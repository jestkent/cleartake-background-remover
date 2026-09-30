"""
Job queue.

A single background worker thread pulls from a FIFO queue and runs jobs
one at a time. Deliberately one thread, not a pool: the models hold GPU
state that is not safe to touch concurrently, and two jobs racing for
8 GB of VRAM is how you get an out-of-memory crash halfway through a
long file. Queueing is the correct behaviour here, not a limitation.

Job records persist to a JSON file so the history survives a restart.
The file is written atomically through a temp file and a rename, so a
crash mid-write cannot leave a truncated database behind.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from backend.pipeline.runner import ProcessOptions, process_file
from backend.settings import get_settings

MAX_HISTORY = 200


@dataclass
class Job:
    id: str
    filename: str
    source_path: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Waiting in queue"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    source_size_bytes: int = 0
    output_size_bytes: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.RLock()
        self._cancelled: set[str] = set()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._load()

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run_worker, name="cleartake-worker", daemon=True
        )
        self._worker.start()

        # Anything left mid-flight from a previous run is stale. Mark it
        # failed rather than leaving a job that claims to be running.
        with self._lock:
            for job in self._jobs.values():
                if job.status in ("running", "queued"):
                    job.status = "failed"
                    job.error = "Interrupted by a restart."
                    job.message = "Interrupted"
        self._save()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")  # unblock the worker so it can notice the flag

    # -- public api -----------------------------------------------------

    def submit(self, source_path: Path, filename: str, options: ProcessOptions) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            filename=filename,
            source_path=str(source_path),
            options=asdict(options),
            source_size_bytes=source_path.stat().st_size if source_path.exists() else 0,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._trim()
        self._save()
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._jobs[i] for i in ids if i in self._jobs]

    def cancel(self, job_id: str) -> bool:
        """
        Request cancellation.

        A queued job is dropped immediately. A running job is flagged and
        stops at the next stage boundary, because killing a job mid-encode
        would leave a half-written file that looks valid.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status == "queued":
                job.status = "cancelled"
                job.message = "Cancelled"
                job.finished_at = time.time()
                self._cancelled.add(job_id)
                self._save()
                return True
            if job.status == "running":
                self._cancelled.add(job_id)
                job.message = "Stopping after the current stage"
                return True
        return False

    def clear_history(self) -> int:
        with self._lock:
            removed = [
                jid for jid, job in self._jobs.items()
                if job.status in ("done", "failed", "cancelled")
            ]
            for jid in removed:
                self._jobs.pop(jid, None)
                if jid in self._order:
                    self._order.remove(jid)
        self._save()
        return len(removed)

    # -- worker ---------------------------------------------------------

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not job_id or self._stop.is_set():
                continue

            with self._lock:
                job = self._jobs.get(job_id)
                if not job or job.status == "cancelled":
                    continue
                job.status = "running"
                job.started_at = time.time()
                job.message = "Starting"
                job.progress = 0.0
            self._save()

            try:
                self._execute(job)
            except Exception as exc:  # noqa: BLE001 - the worker must never die
                with self._lock:
                    job.status = "failed"
                    job.error = str(exc) or exc.__class__.__name__
                    job.message = "Failed"
                    job.finished_at = time.time()
                if os.environ.get("CLEARTAKE_DEBUG"):
                    traceback.print_exc()
            finally:
                self._save()

    def _execute(self, job: Job) -> None:
        def on_progress(fraction: float, message: str) -> None:
            if job.id in self._cancelled:
                raise InterruptedError("Cancelled")
            with self._lock:
                job.progress = round(max(0.0, min(1.0, fraction)), 4)
                job.message = message

        options = ProcessOptions(**job.options)

        try:
            result = process_file(
                job.source_path, options, progress=on_progress, job_id=job.id
            )
        except InterruptedError:
            with self._lock:
                job.status = "cancelled"
                job.message = "Cancelled"
                job.finished_at = time.time()
                self._cancelled.discard(job.id)
            return

        output = Path(result.output) if result.output else None
        with self._lock:
            job.status = "done"
            job.progress = 1.0
            job.message = "Done"
            job.finished_at = time.time()
            job.result = result.to_dict()
            job.duration_seconds = result.duration_seconds
            job.output_size_bytes = (
                output.stat().st_size if output and output.exists() else 0
            )

    # -- persistence ----------------------------------------------------

    def _trim(self) -> None:
        while len(self._order) > MAX_HISTORY:
            oldest = self._order.pop(0)
            self._jobs.pop(oldest, None)

    def _save(self) -> None:
        settings = get_settings()
        try:
            settings.jobs_db.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    "version": 1,
                    "order": self._order,
                    "jobs": {jid: job.to_dict() for jid, job in self._jobs.items()},
                }
            # Write to a sibling temp file, then rename. Rename is atomic
            # on every platform that matters, so a crash cannot leave a
            # half-written database.
            tmp = settings.jobs_db.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, settings.jobs_db)
        except OSError:
            pass  # history is a convenience, never worth failing a job over

    def _load(self) -> None:
        settings = get_settings()
        if not settings.jobs_db.exists():
            return
        try:
            with open(settings.jobs_db, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            valid = set(Job.__dataclass_fields__)
            for jid, data in (payload.get("jobs") or {}).items():
                self._jobs[jid] = Job(**{k: v for k, v in data.items() if k in valid})
            self._order = [j for j in (payload.get("order") or []) if j in self._jobs]
        except (OSError, json.JSONDecodeError, TypeError):
            self._jobs.clear()
            self._order.clear()


_manager: JobManager | None = None


def get_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
