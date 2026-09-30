"""
ClearTake HTTP API.

Binds to 127.0.0.1 by default. Nothing here authenticates anything,
because there is nothing to authenticate against: this is a tool that
runs on your own machine and talks to your own filesystem. Do not put it
on a public interface without a reverse proxy and auth in front.
"""

from __future__ import annotations

import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.jobs import get_manager
from backend.pipeline import media
from backend.pipeline.engines import list_engines, resolve_engine_key
from backend.pipeline.runner import PRESETS, ProcessOptions
from backend.schemas import EngineOut, JobOut, PresetOut, SystemOut
from backend.settings import get_settings

VERSION = "1.0.0"

MEDIA_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv",
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    get_manager().start()
    yield
    get_manager().stop()


app = FastAPI(title="ClearTake", version=VERSION, lifespan=lifespan)

# The dev server runs the UI on a different port than the API. In
# production the API serves the built UI itself and this never fires.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# system
# --------------------------------------------------------------------------


@app.get("/api/system", response_model=SystemOut)
def system_info() -> SystemOut:
    settings = get_settings()
    engines = list_engines()

    ffmpeg_ok = media.ffmpeg_available()
    detail = "Ready." if ffmpeg_ok else (
        "ffmpeg was not found. Install it and put it on your PATH, "
        "or set paths.ffmpeg in config.yaml."
    )

    try:
        active = resolve_engine_key(settings.engines.default)
    except RuntimeError:
        active = "none"

    return SystemOut(
        ffmpeg=ffmpeg_ok,
        ffmpeg_detail=detail,
        engines=[EngineOut(**e.__dict__) for e in engines],
        presets=[
            PresetOut(
                key=key,
                label=cfg["label"],
                blurb=cfg["blurb"],
                attenuation_db=cfg["attenuation_db"],
                declick=cfg["declick"],
                engine=cfg.get("engine", "auto"),
            )
            for key, cfg in PRESETS.items()
        ],
        active_engine=active,
        version=VERSION,
        max_upload_mb=settings.server.max_upload_mb,
    )


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": VERSION, "time": time.time()}


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


@app.post("/api/jobs", response_model=JobOut)
async def create_job(
    file: UploadFile = File(...),
    preset: str = Form("screen_recording"),
    engine: str | None = Form(None),
    attenuation_db: float | None = Form(None),
    declick: bool | None = Form(None),
    declick_sensitivity: float | None = Form(None),
    high_pass_hz: float | None = Form(None),
    normalize: bool | None = Form(None),
    target_lufs: float | None = Form(None),
    keep_original_track: bool | None = Form(None),
) -> JobOut:
    settings = get_settings()
    settings.ensure_dirs()

    original_name = Path(file.filename or "upload").name
    suffix = Path(original_name).suffix.lower()
    if suffix and suffix not in MEDIA_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"{suffix} is not a media file ClearTake can open.",
        )

    # Stream to disk rather than reading into memory. A two hour lecture
    # recording will not fit comfortably in RAM alongside the models.
    stamp = f"{int(time.time() * 1000):x}"
    dest = settings.uploads_dir / f"{stamp}_{original_name}"
    limit = settings.server.max_upload_mb * 1024 * 1024
    written = 0

    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than the {settings.server.max_upload_mb} MB limit.",
                    )
                out.write(chunk)
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file was empty.")

    # Reject anything ffprobe cannot read, before it reaches the queue.
    try:
        info = media.probe(dest)
    except media.MediaError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not info.has_audio:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"{original_name} has no audio track to clean."
        )

    options = ProcessOptions.from_preset(
        preset,
        engine=engine,
        attenuation_db=attenuation_db,
        declick=declick,
        declick_sensitivity=declick_sensitivity,
        high_pass_hz=high_pass_hz,
        normalize=normalize,
        target_lufs=target_lufs,
        keep_original_track=keep_original_track,
    )

    job = get_manager().submit(dest, original_name, options)
    payload = job.to_dict()
    payload["duration_seconds"] = info.duration
    return JobOut(**payload)


@app.get("/api/jobs", response_model=list[JobOut])
def list_jobs(limit: int = 50) -> list[JobOut]:
    return [JobOut(**job.to_dict()) for job in get_manager().list(limit=limit)]


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    job = get_manager().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No job with that id.")
    return JobOut(**job.to_dict())


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not get_manager().cancel(job_id):
        raise HTTPException(
            status_code=409, detail="That job has already finished."
        )
    return {"ok": True}


@app.delete("/api/jobs")
def clear_history() -> dict:
    return {"ok": True, "removed": get_manager().clear_history()}


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def _resolve_output(job_id: str) -> Path:
    job = get_manager().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No job with that id.")
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="That job has not finished.")

    path = Path(job.result.get("output", ""))
    settings = get_settings()

    # Confine downloads to the outputs directory. Without this check a
    # tampered job record could hand out arbitrary files from the disk.
    try:
        path.resolve().relative_to(settings.outputs_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="That path is out of bounds.")

    if not path.exists():
        raise HTTPException(status_code=404, detail="The output file is gone.")
    return path


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    path = _resolve_output(job_id)
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/api/jobs/{job_id}/preview")
def preview(job_id: str) -> FileResponse:
    """Serve the output inline so the browser can play it for an A/B."""
    path = _resolve_output(job_id)
    mime = {
        ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
        ".webm": "video/webm", ".mkv": "video/x-matroska",
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".ogg": "audio/ogg",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=mime)


@app.get("/api/jobs/{job_id}/source")
def source_preview(job_id: str) -> FileResponse:
    """Serve the original upload, for comparing against the cleaned version."""
    job = get_manager().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No job with that id.")

    path = Path(job.source_path)
    settings = get_settings()
    try:
        path.resolve().relative_to(settings.uploads_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="That path is out of bounds.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="The original file is gone.")

    return FileResponse(path)


@app.get("/api/storage")
def storage() -> dict:
    """How much disk the workspace is using, and where."""
    settings = get_settings()

    def folder_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    return {
        "uploads_bytes": folder_size(settings.uploads_dir),
        "outputs_bytes": folder_size(settings.outputs_dir),
        "temp_bytes": folder_size(settings.temp_dir),
        "uploads_path": str(settings.uploads_dir),
        "outputs_path": str(settings.outputs_dir),
    }


@app.post("/api/storage/clear-uploads")
def clear_uploads() -> dict:
    """
    Delete the stored copies of the files you uploaded.

    Cleaned outputs are left alone. Uploads are the bulky half and the
    half you already have on your own disk anyway.
    """
    settings = get_settings()
    removed = 0
    freed = 0
    for f in settings.uploads_dir.glob("*"):
        if f.is_file():
            freed += f.stat().st_size
            f.unlink(missing_ok=True)
            removed += 1
    shutil.rmtree(settings.temp_dir, ignore_errors=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "removed": removed, "freed_bytes": freed}


# --------------------------------------------------------------------------
# static UI
# --------------------------------------------------------------------------

_WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
_FALLBACK = Path(__file__).resolve().parent / "static"

if _WEB_DIST.exists() and (_WEB_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIST), html=True), name="ui")
elif _FALLBACK.exists() and (_FALLBACK / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_FALLBACK), html=True), name="ui")
else:
    @app.get("/")
    def no_ui() -> JSONResponse:
        return JSONResponse(
            {
                "message": "The API is running but no UI was found.",
                "build_it": "cd web && npm install && npm run build",
                "or_use": "python -m backend.cli process <file>",
                "docs": "/docs",
            }
        )
