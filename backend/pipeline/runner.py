"""
Stage orchestration.

The order matters and is not arbitrary:

  extract -> declick -> denoise -> attenuation blend -> high-pass ->
  loudness -> remux -> verify

Declicking comes before denoising. A click is a broadband impulse; leave
it in and the denoiser's noise estimate gets dragged upward by it, which
costs you reduction everywhere else in the file. Remove the clicks first
and the denoiser sees a cleaner picture of what the steady noise actually
is.

The attenuation blend comes after the denoiser and before everything else,
because every later stage should operate on the signal you are actually
going to ship, not on the full-strength version you are not.

Loudness normalisation comes last of the audio stages. Run it earlier and
the denoiser changes the level underneath it, and your -16 LUFS target
quietly becomes something else.
"""

from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.pipeline import media
from backend.pipeline.audio import (
    apply_attenuation_limit,
    noise_floor_db,
    peak_envelope,
    read_wav,
    rms_db,
    write_wav,
)
from backend.pipeline.declick import declick
from backend.pipeline.engines import build_engine, resolve_engine_key
from backend.settings import get_settings

ProgressCallback = Callable[[float, str], None]


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "screen_recording": {
        "label": "Screen recording",
        "blurb": "Mouse clicks, keyboard, fan hum. The default for tutorials and lessons.",
        "engine": "auto",
        "attenuation_db": 24.0,
        "declick": True,
        "declick_sensitivity": 1.2,
        "high_pass_hz": 80.0,
        "normalize": True,
        "target_lufs": -16.0,
    },
    "gentle": {
        "label": "Gentle",
        "blurb": "Light touch for audio that is already decent. Least risk to the voice.",
        "engine": "auto",
        "attenuation_db": 14.0,
        "declick": True,
        "declick_sensitivity": 0.8,
        "high_pass_hz": 70.0,
        "normalize": True,
        "target_lufs": -16.0,
    },
    "voiceover": {
        "label": "Voiceover",
        "blurb": "Narration over a quiet room. Removes the floor, keeps the delivery.",
        "engine": "auto",
        "attenuation_db": 30.0,
        "declick": True,
        "declick_sensitivity": 1.4,
        "high_pass_hz": 85.0,
        "normalize": True,
        "target_lufs": -16.0,
    },
    "rescue": {
        "label": "Rescue",
        "blurb": "Genuinely bad audio. Aggressive, and it will show on the voice.",
        "engine": "auto",
        "attenuation_db": 42.0,
        "declick": True,
        "declick_sensitivity": 1.8,
        "high_pass_hz": 100.0,
        "normalize": True,
        "target_lufs": -16.0,
    },
    "analyse_only": {
        "label": "Measure only",
        "blurb": "Report the noise floor and draw the waveform. Writes no new file.",
        "engine": "auto",
        "attenuation_db": 0.0,
        "declick": False,
        "declick_sensitivity": 0.0,
        "high_pass_hz": 0.0,
        "normalize": False,
        "target_lufs": -16.0,
        "dry_run": True,
    },
}

DEFAULT_PRESET = "screen_recording"


@dataclass
class ProcessOptions:
    preset: str = DEFAULT_PRESET
    engine: str = "auto"
    attenuation_db: float = 24.0
    declick: bool = True
    declick_sensitivity: float = 1.2
    high_pass_hz: float = 80.0
    normalize: bool = True
    target_lufs: float = -16.0
    keep_original_track: bool = True
    dry_run: bool = False

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ProcessOptions":
        base = dict(PRESETS.get(name, PRESETS[DEFAULT_PRESET]))
        base.pop("label", None)
        base.pop("blurb", None)
        base["preset"] = name if name in PRESETS else DEFAULT_PRESET
        # Only accept overrides that were explicitly supplied.
        base.update({k: v for k, v in overrides.items() if v is not None})
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in base.items() if k in valid})


@dataclass
class StageReport:
    name: str
    seconds: float
    detail: str = ""


@dataclass
class ProcessResult:
    job_id: str
    source: str
    output: str
    engine_used: str
    duration_seconds: float
    elapsed_seconds: float
    video_untouched: bool
    clicks_repaired: int = 0
    noise_floor_before_db: float = 0.0
    noise_floor_after_db: float = 0.0
    level_before_db: float = 0.0
    level_after_db: float = 0.0
    waveform_before: list[float] = field(default_factory=list)
    waveform_after: list[float] = field(default_factory=list)
    stages: list[StageReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def noise_reduction_db(self) -> float:
        return self.noise_floor_before_db - self.noise_floor_after_db

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["noise_reduction_db"] = round(self.noise_reduction_db, 2)
        return d


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------


class _Progress:
    """Maps per-stage progress into a single 0-100 bar."""

    def __init__(self, callback: ProgressCallback | None):
        self.callback = callback
        self.base = 0.0
        self.span = 1.0

    def stage(self, base: float, span: float) -> None:
        self.base, self.span = base, span

    def __call__(self, fraction: float, message: str) -> None:
        if self.callback:
            self.callback(self.base + self.span * max(0.0, min(1.0, fraction)), message)

    def mark(self, fraction: float, message: str) -> None:
        if self.callback:
            self.callback(max(0.0, min(1.0, fraction)), message)


def process_file(
    source: str | Path,
    options: ProcessOptions | None = None,
    output_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
    job_id: str | None = None,
) -> ProcessResult:
    """Run one file end to end. Raises on failure; never returns half-done."""
    settings = get_settings()
    settings.ensure_dirs()

    source = Path(source).resolve()
    options = options or ProcessOptions()
    job_id = job_id or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    report = _Progress(progress)
    stages: list[StageReport] = []
    warnings: list[str] = []

    if not source.exists():
        raise FileNotFoundError(f"No such file: {source}")

    # ---- probe ---------------------------------------------------------
    report.mark(0.01, "Reading file")
    info = media.probe(source)
    if not info.has_audio:
        raise ValueError(f"{source.name} has no audio track to clean.")

    work = settings.temp_dir / job_id
    work.mkdir(parents=True, exist_ok=True)

    try:
        # ---- extract ---------------------------------------------------
        t0 = time.perf_counter()
        report.mark(0.04, "Extracting audio")
        raw_wav = media.extract_audio(
            source, work / "original.wav", sample_rate=settings.audio.sample_rate
        )
        data, sample_rate = read_wav(raw_wav)
        stages.append(StageReport("extract", time.perf_counter() - t0,
                                  f"{data.shape[0]}ch at {sample_rate} Hz"))

        floor_before = noise_floor_db(data, sample_rate)
        level_before = rms_db(data)
        wave_before = peak_envelope(data)
        original = data.copy()

        if options.dry_run:
            return _measure_only(
                job_id, source, info, data, sample_rate, floor_before,
                level_before, wave_before, stages, started, report,
            )

        # ---- declick ---------------------------------------------------
        clicks = 0
        if options.declick and options.declick_sensitivity > 0:
            t0 = time.perf_counter()
            report.mark(0.10, "Repairing clicks")
            data, clicks = declick(
                data, sample_rate, sensitivity=options.declick_sensitivity
            )
            stages.append(StageReport("declick", time.perf_counter() - t0,
                                      f"{clicks} repaired"))

        # ---- denoise ---------------------------------------------------
        t0 = time.perf_counter()
        engine_key = resolve_engine_key(options.engine)
        engine = build_engine(
            engine_key,
            device=settings.engines.device,
            model_name=(
                settings.engines.deepfilternet_model
                if engine_key == "deepfilternet"
                else settings.engines.clearervoice_model
            ),
        )
        report.stage(0.18, 0.55)
        denoised = engine.denoise(data, sample_rate, progress=report)
        stages.append(StageReport("denoise", time.perf_counter() - t0, engine.name))

        # ---- attenuation blend -----------------------------------------
        report.mark(0.76, "Blending to target reduction")
        data = apply_attenuation_limit(data, denoised, options.attenuation_db)

        # ---- high-pass -------------------------------------------------
        if options.high_pass_hz and options.high_pass_hz > 0:
            t0 = time.perf_counter()
            report.mark(0.80, "Removing low rumble")
            data = _high_pass(data, sample_rate, options.high_pass_hz)
            stages.append(StageReport("high_pass", time.perf_counter() - t0,
                                      f"{options.high_pass_hz:.0f} Hz"))

        # Guard against anything upstream pushing past full scale.
        peak = float(np.abs(data).max(initial=0.0))
        if peak > 0.999:
            data = data * (0.999 / peak)
            warnings.append("Peaks were limited to stay under full scale.")

        clean_wav = write_wav(work / "cleaned.wav", data, sample_rate)

        # ---- loudness --------------------------------------------------
        if options.normalize:
            t0 = time.perf_counter()
            report.mark(0.84, "Matching loudness")
            try:
                clean_wav = media.loudness_normalize(
                    clean_wav, work / "normalized.wav",
                    target_lufs=options.target_lufs,
                )
                data, sample_rate = read_wav(clean_wav)
                stages.append(StageReport("loudness", time.perf_counter() - t0,
                                          f"{options.target_lufs:.0f} LUFS"))
            except media.MediaError as exc:
                warnings.append(f"Loudness step skipped: {exc}")

        floor_after = noise_floor_db(data, sample_rate)
        level_after = rms_db(data)
        wave_after = peak_envelope(data)

        # ---- write output ----------------------------------------------
        t0 = time.perf_counter()
        report.mark(0.90, "Writing output file")
        dest = Path(output_path) if output_path else _default_output(source, settings)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if info.has_video:
            media.remux(
                source, clean_wav, dest,
                audio_codec=settings.audio.output_codec,
                audio_bitrate=settings.audio.output_bitrate,
                keep_original_track=options.keep_original_track,
            )
        else:
            media.write_audio_only(
                clean_wav, dest,
                audio_codec=settings.audio.output_codec,
                audio_bitrate=settings.audio.output_bitrate,
            )
        stages.append(StageReport("write", time.perf_counter() - t0, dest.name))

        # ---- verify ------------------------------------------------------
        report.mark(0.97, "Verifying the video is untouched")
        untouched = media.verify_video_untouched(source, dest)
        if info.has_video and not untouched:
            raise RuntimeError(
                "The output video stream does not match the source. "
                "Refusing to report success. This is a bug; please file it."
            )

        report.mark(1.0, "Done")
        return ProcessResult(
            job_id=job_id,
            source=str(source),
            output=str(dest),
            engine_used=engine.name,
            duration_seconds=info.duration,
            elapsed_seconds=time.perf_counter() - started,
            video_untouched=untouched,
            clicks_repaired=clicks,
            noise_floor_before_db=round(floor_before, 2),
            noise_floor_after_db=round(floor_after, 2),
            level_before_db=round(level_before, 2),
            level_after_db=round(level_after, 2),
            waveform_before=wave_before,
            waveform_after=wave_after,
            stages=stages,
            warnings=warnings,
        )

    finally:
        if not _keep_temp():
            shutil.rmtree(work, ignore_errors=True)


def _measure_only(
    job_id, source, info, data, sample_rate, floor_before,
    level_before, wave_before, stages, started, report,
) -> ProcessResult:
    report.mark(1.0, "Measurement complete")
    return ProcessResult(
        job_id=job_id,
        source=str(source),
        output="",
        engine_used="none",
        duration_seconds=info.duration,
        elapsed_seconds=time.perf_counter() - started,
        video_untouched=True,
        noise_floor_before_db=round(floor_before, 2),
        noise_floor_after_db=round(floor_before, 2),
        level_before_db=round(level_before, 2),
        level_after_db=round(level_before, 2),
        waveform_before=wave_before,
        waveform_after=wave_before,
        stages=stages,
        warnings=["Measurement only. No file was written."],
    )


def _high_pass(data: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """
    Second-order Butterworth high-pass, applied forward and backward.

    Zero phase shift matters here: a one-way filter rotates phase near the
    cutoff, and on a stereo track that smears the image. Filtering in both
    directions cancels the shift at the cost of doubling the slope.
    """
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        return data

    nyquist = sample_rate / 2.0
    normalized = max(1e-5, min(cutoff_hz / nyquist, 0.99))
    sos = butter(2, normalized, btype="highpass", output="sos")

    out = np.empty_like(data)
    for ch in range(data.shape[0]):
        if data[ch].size > 24:
            out[ch] = sosfiltfilt(sos, data[ch]).astype(np.float32)
        else:
            out[ch] = data[ch]
    return out


def _default_output(source: Path, settings) -> Path:
    stem = source.stem
    suffix = source.suffix or ".mp4"
    dest = settings.outputs_dir / f"{stem}_clean{suffix}"
    counter = 2
    while dest.exists():
        dest = settings.outputs_dir / f"{stem}_clean_{counter}{suffix}"
        counter += 1
    return dest


def _keep_temp() -> bool:
    import os

    return os.environ.get("CLEARTAKE_KEEP_TEMP", "").lower() in ("1", "true", "yes")
