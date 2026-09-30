"""Request and response shapes for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EngineOut(BaseModel):
    key: str
    name: str
    description: str
    available: bool
    install_hint: str = ""
    requires_gpu: bool = False
    strength: str = "medium"
    detail: str = ""


class PresetOut(BaseModel):
    key: str
    label: str
    blurb: str
    attenuation_db: float
    declick: bool
    engine: str = "auto"


class JobOptionsIn(BaseModel):
    preset: str = "screen_recording"
    engine: str | None = None
    attenuation_db: float | None = Field(default=None, ge=0, le=60)
    declick: bool | None = None
    declick_sensitivity: float | None = Field(default=None, ge=0, le=2.5)
    high_pass_hz: float | None = Field(default=None, ge=0, le=300)
    normalize: bool | None = None
    target_lufs: float | None = Field(default=None, ge=-31, le=-9)
    keep_original_track: bool | None = None


class StageOut(BaseModel):
    name: str
    seconds: float
    detail: str = ""


class JobOut(BaseModel):
    id: str
    filename: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    progress: float = 0.0
    message: str = ""
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    source_size_bytes: int = 0
    output_size_bytes: int = 0
    duration_seconds: float = 0.0


class SystemOut(BaseModel):
    ffmpeg: bool
    ffmpeg_detail: str = ""
    engines: list[EngineOut]
    presets: list[PresetOut]
    active_engine: str
    version: str
    max_upload_mb: int
