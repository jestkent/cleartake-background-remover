#!/usr/bin/env python3
"""
Build a test video with exactly the problems ClearTake targets.

Run this before you point the tool at real footage. It gives you a file
where you know the ground truth, so you can tell whether a settings change
actually helped instead of guessing.

The synthetic clip contains:
  * a speech-like signal (formant-shaped tone bursts with pauses)
  * steady broadband hiss at a known level
  * 60 Hz mains hum plus its harmonic
  * sharp mouse clicks scattered through the timeline
  * a real H.264 video stream, so the remux path gets exercised

    python scripts/make_test_clip.py
    python scripts/make_test_clip.py --seconds 30 --out samples/noisy.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.pipeline.audio import write_wav  # noqa: E402

SAMPLE_RATE = 48_000


def synth_speech(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """
    A stand-in for a voice: pitched bursts with formants, and pauses.

    The pauses matter more than the bursts. Noise reduction is judged in
    the gaps between words, because that is where a listener hears hiss
    and where a bad denoiser leaves pumping artefacts.
    """
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    signal = np.zeros(n, dtype=np.float64)

    cursor = 0.35
    while cursor < seconds - 0.5:
        burst_len = float(rng.uniform(0.28, 0.75))
        start = int(cursor * SAMPLE_RATE)
        end = min(n, int((cursor + burst_len) * SAMPLE_RATE))
        span = end - start
        if span < 128:
            break

        local_t = t[start:end] - t[start]
        f0 = float(rng.uniform(95, 165))          # male-ish fundamental
        vibrato = 1.0 + 0.012 * np.sin(2 * np.pi * 4.5 * local_t)

        burst = np.zeros(span)
        for harmonic, weight in ((1, 1.0), (2, 0.55), (3, 0.32), (4, 0.18), (6, 0.09)):
            burst += weight * np.sin(2 * np.pi * f0 * harmonic * local_t * vibrato)

        # Two formant resonances give it a vowel-like colour.
        for formant, weight in ((620.0, 0.30), (1180.0, 0.18)):
            burst += weight * np.sin(2 * np.pi * formant * local_t)

        envelope = np.minimum(
            np.minimum(local_t / 0.03, 1.0),
            np.minimum((local_t[-1] - local_t) / 0.06, 1.0),
        )
        signal[start:end] += burst * np.clip(envelope, 0, 1) * rng.uniform(0.5, 0.85)
        cursor += burst_len + float(rng.uniform(0.18, 0.55))

    peak = np.abs(signal).max()
    return (signal / peak * 0.42) if peak > 0 else signal


def add_hiss(signal: np.ndarray, rng: np.random.Generator, level_db: float) -> np.ndarray:
    amplitude = 10.0 ** (level_db / 20.0)
    return signal + rng.normal(0.0, amplitude, signal.size)


def add_hum(signal: np.ndarray, level_db: float = -46.0) -> np.ndarray:
    t = np.arange(signal.size) / SAMPLE_RATE
    amplitude = 10.0 ** (level_db / 20.0)
    hum = amplitude * (np.sin(2 * np.pi * 60 * t) + 0.4 * np.sin(2 * np.pi * 120 * t))
    return signal + hum


def add_clicks(
    signal: np.ndarray, rng: np.random.Generator, count: int
) -> tuple[np.ndarray, list[float]]:
    """Mouse clicks: very short, very steep, broadband."""
    out = signal.copy()
    positions: list[float] = []
    guard = int(0.02 * SAMPLE_RATE)

    for _ in range(count):
        at = int(rng.uniform(guard, signal.size - guard))
        length = int(rng.uniform(0.0008, 0.0022) * SAMPLE_RATE)
        decay = np.exp(-np.linspace(0, 9, length))
        click = rng.normal(0, 1, length) * decay * float(rng.uniform(0.35, 0.72))
        out[at : at + length] += click
        positions.append(round(at / SAMPLE_RATE, 3))

    return out, sorted(positions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a noisy test clip.")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--out", default=str(REPO_ROOT / "samples" / "noisy_test.mp4"))
    parser.add_argument("--hiss-db", type=float, default=-42.0,
                        help="Hiss level in dBFS. Lower is quieter.")
    parser.add_argument("--clicks", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--audio-only", action="store_true")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)

    print(f"Synthesising {args.seconds:.0f}s of speech-like audio")
    clean = synth_speech(args.seconds, rng)

    print(f"Adding hiss at {args.hiss_db:.0f} dBFS, 60 Hz hum, {args.clicks} clicks")
    noisy = add_hum(add_hiss(clean, rng, args.hiss_db))
    noisy, click_times = add_clicks(noisy, rng, args.clicks)
    noisy = np.clip(noisy, -0.99, 0.99).astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        write_wav(wav_path, noisy[None, :], SAMPLE_RATE)

        if args.audio_only:
            shutil.copy(wav_path, out_path)
        else:
            print("Encoding video")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi",
                    "-i", f"testsrc2=size=640x360:rate=30:duration={args.seconds}",
                    "-i", str(wav_path),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(out_path),
                ],
                check=True,
            )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out_path}  ({size_mb:.1f} MB)")
    print(f"Clicks were placed at: {', '.join(f'{t}s' for t in click_times[:10])}"
          + (" ..." if len(click_times) > 10 else ""))
    print("\nNow run:")
    print(f"  python -m backend.cli process {out_path} --preset screen_recording")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
