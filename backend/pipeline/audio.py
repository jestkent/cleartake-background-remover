"""
Audio array helpers.

Every stage in the pipeline speaks the same language: a float32 numpy array
shaped (channels, samples) in the range [-1, 1], plus a sample rate. Reads
and writes go through here so no stage has to think about WAV subtypes.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _parse_riff(raw: bytes) -> tuple[dict, bytes]:
    """
    Walk a RIFF file and return the parsed fmt chunk plus the data bytes.

    Written by hand rather than leaning on the stdlib `wave` module
    because ffmpeg writes float WAV as WAVE_FORMAT_EXTENSIBLE (0xFFFE),
    and `wave` refuses to open those outright. Since extensible float is
    exactly what this pipeline passes between its own stages, that
    refusal would break the tool on a machine without soundfile.
    """
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file.")

    fmt: dict = {}
    data = b""
    pos = 12

    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        (chunk_size,) = struct.unpack_from("<I", raw, pos + 4)
        body_start = pos + 8
        body_end = min(body_start + chunk_size, len(raw))

        if chunk_id == b"fmt ":
            fields = struct.unpack_from("<HHIIHH", raw, body_start)
            fmt = {
                "format": fields[0],
                "channels": fields[1],
                "sample_rate": fields[2],
                "bits": fields[5],
            }
            # Extensible headers carry the true format in a trailing GUID.
            if fmt["format"] == 0xFFFE and chunk_size >= 40:
                (valid_bits,) = struct.unpack_from("<H", raw, body_start + 18)
                subformat = raw[body_start + 24 : body_start + 28]
                (fmt["format"],) = struct.unpack("<H", subformat[:2])
                if valid_bits:
                    fmt["bits"] = valid_bits
        elif chunk_id == b"data":
            data = raw[body_start:body_end]

        pos = body_start + chunk_size + (chunk_size & 1)  # chunks are word aligned

    if not fmt:
        raise ValueError("WAV file has no fmt chunk.")
    return fmt, data


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Read a WAV into (channels, samples) float32.

    soundfile is used when present because it handles every subtype going.
    The built-in parser below covers PCM 8/16/24/32 and IEEE float 32/64,
    including extensible variants, so the tool runs fine with nothing but
    numpy installed.
    """
    path = Path(path)
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), int(sr)
    except ImportError:
        pass

    with open(path, "rb") as fh:
        raw = fh.read()

    fmt, payload = _parse_riff(raw)
    channels = max(1, fmt["channels"])
    bits = fmt["bits"]
    audio_format = fmt["format"]

    if audio_format == 3:  # IEEE float
        if bits == 64:
            arr = np.frombuffer(payload, dtype="<f8").astype(np.float32)
        else:
            arr = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    elif audio_format == 1:  # PCM
        if bits == 16:
            arr = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        elif bits == 32:
            arr = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
        elif bits == 24:
            usable = (len(payload) // 3) * 3
            trio = np.frombuffer(payload[:usable], dtype=np.uint8).reshape(-1, 3)
            as_int = (
                trio[:, 0].astype(np.int32)
                | (trio[:, 1].astype(np.int32) << 8)
                | (trio[:, 2].astype(np.int32) << 16)
            )
            as_int = np.where(as_int & 0x800000, as_int - 0x1000000, as_int)
            arr = as_int.astype(np.float32) / 8388608.0
        elif bits == 8:
            arr = (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {bits}")
    else:
        raise ValueError(
            f"Unsupported WAV format tag {audio_format}. "
            "Install soundfile for wider format support: pip install soundfile"
        )

    usable = (arr.size // channels) * channels
    frames = arr[:usable].reshape(-1, channels)
    return np.ascontiguousarray(frames.T), int(fmt["sample_rate"])


def write_wav(path: str | Path, data: np.ndarray, sample_rate: int) -> Path:
    """Write (channels, samples) float32 out as a 32-bit float WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.atleast_2d(np.asarray(data, dtype=np.float32))
    interleaved = np.ascontiguousarray(data.T).astype("<f4")

    try:
        import soundfile as sf  # type: ignore

        sf.write(str(path), interleaved, sample_rate, subtype="FLOAT")
        return path
    except ImportError:
        pass

    # Minimal WAVE_FORMAT_IEEE_FLOAT writer; the wave module only does PCM.
    payload = interleaved.tobytes()
    channels = data.shape[0]
    byte_rate = sample_rate * channels * 4
    block_align = channels * 4

    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 3, channels, sample_rate, byte_rate, block_align, 32
    )
    header += b"data" + struct.pack("<I", len(payload))

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(payload)
    return path


def to_mono(data: np.ndarray) -> np.ndarray:
    """Average channels down to a single (1, samples) array."""
    data = np.atleast_2d(data)
    return data.mean(axis=0, keepdims=True).astype(np.float32)


def match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim two arrays to their shorter length so they can be mixed."""
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def apply_attenuation_limit(
    original: np.ndarray, processed: np.ndarray, atten_db: float
) -> np.ndarray:
    """
    Blend the processed signal back toward the original by a fixed amount.

    Running a denoiser at full strength is what produces the hollow,
    underwater, swimming-artefact sound that makes cleaned audio obviously
    cleaned. Holding the reduction to a set number of decibels leaves a
    quiet, natural noise floor underneath the speech and keeps the result
    sounding like a room rather than a vacuum.

    atten_db of 25 means the removed noise comes back 25 dB down, so you
    get 25 dB of real reduction and nothing more. 0 leaves the audio
    untouched; anything at or above 60 is effectively full strength.
    """
    if atten_db <= 0:
        return original.copy()
    if atten_db >= 60:
        return processed.copy()

    original, processed = match_length(original, processed)
    removed = original - processed              # whatever the model stripped
    gain = float(10.0 ** (-atten_db / 20.0))    # dB to linear
    return (processed + removed * gain).astype(np.float32)


def peak_envelope(data: np.ndarray, buckets: int = 900) -> list[float]:
    """
    Downsample to per-bucket peak values for drawing a waveform.

    Peaks rather than averages: an average envelope hides exactly the
    transients you want to inspect after a denoise pass.
    """
    mono = to_mono(data)[0]
    if mono.size == 0:
        return [0.0] * buckets

    buckets = max(1, min(buckets, mono.size))
    edges = np.linspace(0, mono.size, buckets + 1, dtype=int)
    out = [
        float(np.abs(mono[start:end]).max()) if end > start else 0.0
        for start, end in zip(edges[:-1], edges[1:])
    ]
    if len(out) < 900:
        out += [0.0] * (900 - len(out))
    return out


def rms_db(data: np.ndarray) -> float:
    """Overall RMS level in dBFS."""
    mono = to_mono(data)[0]
    if mono.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-12))


def noise_floor_db(data: np.ndarray, sample_rate: int, percentile: float = 10.0) -> float:
    """
    Estimate the resting noise floor in dBFS.

    Frame the signal, take the RMS of every frame, and read off a low
    percentile. Those quiet frames are the gaps between words, which is
    where the hiss lives. Comparing this number before and after is the
    clearest single measure of whether a pass did anything.
    """
    mono = to_mono(data)[0]
    frame = max(1, int(sample_rate * 0.02))
    if mono.size < frame:
        return rms_db(data)

    count = mono.size // frame
    frames = mono[: count * frame].reshape(count, frame)
    energies = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    floor = float(np.percentile(energies, percentile))
    return 20.0 * np.log10(max(floor, 1e-12))
