"""
Configuration.

Everything tunable lives in config.yaml at the repo root. Nothing is
hardcoded to a particular machine, so the same checkout runs on the
desktop, a laptop, or a VPS without edits to the source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("CLEARTAKE_CONFIG", REPO_ROOT / "config.yaml"))


@dataclass
class Paths:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    work_dir: str = "./workspace"
    uploads: str = "./workspace/uploads"
    outputs: str = "./workspace/outputs"
    temp: str = "./workspace/temp"
    jobs_db: str = "./workspace/jobs.json"


@dataclass
class Server:
    host: str = "127.0.0.1"
    port: int = 7788
    open_browser: bool = True
    max_upload_mb: int = 8192
    workers: int = 1  # concurrent jobs; models are not thread-safe, keep at 1


@dataclass
class Audio:
    sample_rate: int = 48_000
    output_codec: str = "aac"
    output_bitrate: str = "256k"
    keep_original_track: bool = True


@dataclass
class Engines:
    default: str = "auto"
    deepfilternet_model: str = "DeepFilterNet3"
    clearervoice_model: str = "MossFormer2_SE_48K"
    device: str = "auto"  # auto | cuda | cpu


@dataclass
class Settings:
    paths: Paths = field(default_factory=Paths)
    server: Server = field(default_factory=Server)
    audio: Audio = field(default_factory=Audio)
    engines: Engines = field(default_factory=Engines)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def resolve(self, value: str) -> Path:
        """Turn a possibly relative config path into an absolute one."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def work_dir(self) -> Path:
        return self.resolve(self.paths.work_dir)

    @property
    def uploads_dir(self) -> Path:
        return self.resolve(self.paths.uploads)

    @property
    def outputs_dir(self) -> Path:
        return self.resolve(self.paths.outputs)

    @property
    def temp_dir(self) -> Path:
        return self.resolve(self.paths.temp)

    @property
    def jobs_db(self) -> Path:
        return self.resolve(self.paths.jobs_db)

    def ensure_dirs(self) -> None:
        for d in (self.work_dir, self.uploads_dir, self.outputs_dir, self.temp_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.jobs_db.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d


def _merge(target: Any, data: dict[str, Any]) -> None:
    for key, value in (data or {}).items():
        if hasattr(target, key):
            setattr(target, key, value)


_cached: Settings | None = None


def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_PATH
    settings = Settings()

    if path.exists() and yaml is not None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        settings.raw = data
        _merge(settings.paths, data.get("paths", {}))
        _merge(settings.server, data.get("server", {}))
        _merge(settings.audio, data.get("audio", {}))
        _merge(settings.engines, data.get("engines", {}))

    # Environment wins over the file, which makes container and CI runs easy.
    if os.environ.get("CLEARTAKE_PORT"):
        settings.server.port = int(os.environ["CLEARTAKE_PORT"])
    if os.environ.get("CLEARTAKE_HOST"):
        settings.server.host = os.environ["CLEARTAKE_HOST"]
    if os.environ.get("CLEARTAKE_WORK_DIR"):
        settings.paths.work_dir = os.environ["CLEARTAKE_WORK_DIR"]
    if os.environ.get("FFMPEG_PATH"):
        settings.paths.ffmpeg = os.environ["FFMPEG_PATH"]

    return settings


def get_settings() -> Settings:
    global _cached
    if _cached is None:
        _cached = load_settings()
    return _cached


def reset_settings_cache() -> None:
    global _cached
    _cached = None
