"""
ffmpeg plumbing.

This module is the reason ClearTake cannot degrade your video. The video
stream is never decoded and never re-encoded. It is stream-copied from the
input container to the output container with `-c:v copy`, which moves the
already-compressed packets across byte for byte.

`verify_video_untouched()` proves it by hashing the video stream of both
files and comparing. If those hashes differ, something is wrong and the
pipeline refuses to call the job a success.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.settings import get_settings


class MediaError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails, carrying the tail of stderr."""


# --------------------------------------------------------------------------
# binary discovery
# --------------------------------------------------------------------------


def _binary(name: str) -> str:
    """Resolve ffmpeg/ffprobe from config.yaml, then PATH."""
    configured = getattr(get_settings().paths, name, None)
    if configured:
        p = Path(configured)
        if p.exists():
            return str(p)
        # allow a bare command name in config
        found = shutil.which(configured)
        if found:
            return found
    found = shutil.which(name)
    if not found:
        raise MediaError(
            f"{name} was not found. Install ffmpeg and put it on your PATH, "
            f"or set paths.{name} in config.yaml."
        )
    return found


def ffmpeg_available() -> bool:
    try:
        _binary("ffmpeg")
        _binary("ffprobe")
        return True
    except MediaError:
        return False


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-18:])
        raise MediaError(f"{what} failed (exit {proc.returncode}):\n{tail}")
    return proc


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


@dataclass
class MediaInfo:
    path: Path
    duration: float
    has_video: bool
    has_audio: bool
    video_codec: str | None
    audio_codec: str | None
    sample_rate: int | None
    channels: int | None
    width: int | None = None
    height: int | None = None
    container: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_audio_only(self) -> bool:
        return self.has_audio and not self.has_video


def probe(path: str | Path) -> MediaInfo:
    """Read stream metadata with ffprobe."""
    path = Path(path)
    if not path.exists():
        raise MediaError(f"File not found: {path}")

    proc = _run(
        [
            _binary("ffprobe"),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        "ffprobe",
    )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # Cover art in an mp3 shows up as a video stream. Treat it as audio-only.
    if video is not None and video.get("disposition", {}).get("attached_pic"):
        video = None

    duration = 0.0
    for candidate in (fmt.get("duration"), (audio or {}).get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    return MediaInfo(
        path=path,
        duration=duration,
        has_video=video is not None,
        has_audio=audio is not None,
        video_codec=(video or {}).get("codec_name"),
        audio_codec=(audio or {}).get("codec_name"),
        sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        channels=int(audio["channels"]) if audio and audio.get("channels") else None,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        container=fmt.get("format_name", ""),
        raw=data,
    )


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------


def extract_audio(
    src: str | Path,
    dest_wav: str | Path,
    sample_rate: int = 48_000,
    channels: int | None = None,
) -> Path:
    """
    Pull the first audio track out to a 32-bit float WAV.

    Float WAV is deliberate: every later stage works in float and writes back
    to float, so nothing clips or quantises between stages. The single
    lossy step in the whole pipeline is the final audio encode at remux.
    """
    src, dest_wav = Path(src), Path(dest_wav)
    dest_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _binary("ffmpeg"), "-y",
        "-i", str(src),
        "-vn", "-sn", "-dn",              # drop video, subtitles, data
        "-map", "0:a:0",                  # first audio track only
        "-ar", str(sample_rate),
        "-c:a", "pcm_f32le",
    ]
    if channels:
        cmd += ["-ac", str(channels)]
    cmd.append(str(dest_wav))

    _run(cmd, "Audio extraction")
    return dest_wav


# --------------------------------------------------------------------------
# remux
# --------------------------------------------------------------------------


def remux(
    original_video: str | Path,
    clean_audio_wav: str | Path,
    dest: str | Path,
    audio_codec: str = "aac",
    audio_bitrate: str = "256k",
    keep_original_track: bool = True,
) -> Path:
    """
    Put the cleaned audio back into the original container.

    `-c:v copy` is the whole point: the video packets are copied, not
    re-encoded, so the picture is bit-identical to the source.

    With keep_original_track the output carries two audio tracks. Track 1 is
    the cleaned audio and is flagged default, so any player picks it up
    automatically. Track 2 is the untouched original, stream-copied. That
    gives you an instant A/B in your editor and a way back if a cleanup pass
    went too far, at the cost of a few MB.
    """
    original_video, clean_audio_wav, dest = (
        Path(original_video), Path(clean_audio_wav), Path(dest)
    )
    dest.parent.mkdir(parents=True, exist_ok=True)

    info = probe(original_video)
    if not info.has_video:
        raise MediaError(
            "remux() needs a source with a video stream. "
            "Use write_audio_only() for audio files."
        )

    cmd = [
        _binary("ffmpeg"), "-y",
        "-i", str(original_video),
        "-i", str(clean_audio_wav),
        "-map", "0:v",                    # video from the original
        "-map", "1:a:0",                  # cleaned audio
    ]
    if keep_original_track:
        cmd += ["-map", "0:a:0?"]         # original audio, if present

    cmd += [
        "-c:v", "copy",                   # <-- video is never re-encoded
        "-c:a:0", audio_codec,
        "-b:a:0", audio_bitrate,
        # MKV reads `title`; MP4 has no per-track title field and shows
        # `handler_name` instead. Set both so the track is labelled
        # whichever container the source happened to use.
        "-metadata:s:a:0", "title=Cleaned",
        "-metadata:s:a:0", "handler_name=Cleaned",
        "-disposition:a:0", "default",
    ]
    if keep_original_track and info.has_audio:
        # Copy the source track untouched where the container allows it,
        # otherwise fall back to a re-encode of the original.
        passthrough_ok = _codec_fits_container(info.audio_codec, dest.suffix)
        cmd += ["-c:a:1", "copy" if passthrough_ok else audio_codec]
        if not passthrough_ok:
            cmd += ["-b:a:1", audio_bitrate]
        cmd += [
            "-metadata:s:a:1", "title=Original",
            "-metadata:s:a:1", "handler_name=Original",
            "-disposition:a:1", "0",
        ]

    # Subtitles and chapters ride along untouched when the container allows.
    cmd += ["-map", "0:s?", "-c:s", "copy", "-map_chapters", "0"]
    cmd += ["-movflags", "+faststart", str(dest)]

    try:
        _run(cmd, "Remux")
    except MediaError:
        # Some containers reject subtitle passthrough. Retry without it
        # rather than failing the whole job over a caption track.
        fallback = [c for c in cmd if c not in ("0:s?",)]
        fallback = _drop_pair(fallback, "-map", "0:s?")
        fallback = _drop_pair(fallback, "-c:s", "copy")
        _run(fallback, "Remux (without subtitles)")

    return dest


def write_audio_only(
    clean_audio_wav: str | Path,
    dest: str | Path,
    audio_codec: str = "aac",
    audio_bitrate: str = "256k",
) -> Path:
    """Encode the cleaned audio on its own, for audio-only inputs."""
    clean_audio_wav, dest = Path(clean_audio_wav), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [_binary("ffmpeg"), "-y", "-i", str(clean_audio_wav)]
    if dest.suffix.lower() == ".wav":
        cmd += ["-c:a", "pcm_s24le"]
    else:
        cmd += ["-c:a", audio_codec, "-b:a", audio_bitrate]
    cmd.append(str(dest))

    _run(cmd, "Audio encode")
    return dest


def _codec_fits_container(codec: str | None, suffix: str) -> bool:
    if not codec:
        return False
    suffix = suffix.lower().lstrip(".")
    allowed = {
        "mp4": {"aac", "mp3", "alac", "ac3", "eac3"},
        "m4v": {"aac", "mp3", "alac", "ac3", "eac3"},
        "mov": {"aac", "mp3", "alac", "pcm_s16le", "pcm_s24le", "ac3"},
        "mkv": {"aac", "mp3", "opus", "vorbis", "flac", "ac3", "eac3",
                "pcm_s16le", "pcm_s24le", "alac", "dts", "truehd"},
        "webm": {"opus", "vorbis"},
    }
    return codec in allowed.get(suffix, set())


def _drop_pair(cmd: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == flag and i + 1 < len(cmd) and cmd[i + 1] == value:
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    return out


# --------------------------------------------------------------------------
# proof
# --------------------------------------------------------------------------


def video_stream_hash(path: str | Path) -> str | None:
    """
    MD5 of the video stream's packets, with no decoding involved.

    Two files with the same value hold the same picture data, whatever
    else changed around it in the container.
    """
    info = probe(path)
    if not info.has_video:
        return None

    proc = _run(
        [
            _binary("ffmpeg"), "-v", "error",
            "-i", str(path),
            "-map", "0:v",
            "-c", "copy",
            "-f", "md5", "-",
        ],
        "Video stream hash",
    )
    line = (proc.stdout or "").strip()
    return line.split("=", 1)[1] if "=" in line else line or None


def verify_video_untouched(source: str | Path, output: str | Path) -> bool:
    """
    True when the output's video stream is identical to the source's.

    The pipeline runs this on every job with video and fails loudly if it
    ever comes back False, so a silent re-encode can never slip through.
    """
    src_hash = video_stream_hash(source)
    if src_hash is None:
        return True  # audio-only input, nothing to protect
    return src_hash == video_stream_hash(output)


# --------------------------------------------------------------------------
# loudness
# --------------------------------------------------------------------------


def loudness_normalize(
    src_wav: str | Path,
    dest_wav: str | Path,
    target_lufs: float = -16.0,
    true_peak: float = -1.5,
    loudness_range: float = 11.0,
) -> Path:
    """
    Two-pass EBU R128 normalisation via ffmpeg's loudnorm.

    Two passes rather than one: the first measures the file, the second
    applies a fixed correction using those measurements. Single-pass
    loudnorm works on a moving estimate and audibly pumps on speech.
    """
    src_wav, dest_wav = Path(src_wav), Path(dest_wav)

    measure = subprocess.run(
        [
            _binary("ffmpeg"), "-hide_banner", "-i", str(src_wav),
            "-af",
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, errors="replace",
    )

    stats = _parse_trailing_json(measure.stderr or "")
    filt = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}"
    if stats:
        filt += (
            f":measured_I={stats.get('input_i')}"
            f":measured_TP={stats.get('input_tp')}"
            f":measured_LRA={stats.get('input_lra')}"
            f":measured_thresh={stats.get('input_thresh')}"
            f":offset={stats.get('target_offset', '0.0')}"
            ":linear=true:print_format=summary"
        )

    _run(
        [
            _binary("ffmpeg"), "-y", "-i", str(src_wav),
            "-af", filt,
            "-ar", "48000", "-c:a", "pcm_f32le",
            str(dest_wav),
        ],
        "Loudness normalisation",
    )
    return dest_wav


def _parse_trailing_json(text: str) -> dict[str, Any] | None:
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
