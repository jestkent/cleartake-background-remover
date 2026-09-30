#!/usr/bin/env python3
"""
ClearTake command line.

    python -m backend.cli process lesson.mp4
    python -m backend.cli process lesson.mp4 --preset voiceover --atten 30
    python -m backend.cli batch ./recordings --preset screen_recording
    python -m backend.cli measure lesson.mp4
    python -m backend.cli engines
    python -m backend.cli serve

The CLI calls straight into the same pipeline the web UI uses, so a
result from one is identical to the other. Batch mode is the one that
earns its keep: a folder of lesson recordings processed overnight beats
dragging them in one at a time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.pipeline import media  # noqa: E402
from backend.pipeline.engines import list_engines, resolve_engine_key  # noqa: E402
from backend.pipeline.runner import (  # noqa: E402
    PRESETS,
    ProcessOptions,
    ProcessResult,
    process_file,
)
from backend.settings import get_settings  # noqa: E402

MEDIA_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv",
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aiff", ".aif",
}

BAR_WIDTH = 34


class Bar:
    """A progress bar that repaints one line, and stays quiet when piped."""

    def __init__(self, label: str, enabled: bool = True):
        self.label = label
        self.enabled = enabled and sys.stdout.isatty()
        self.last = 0.0

    def __call__(self, fraction: float, message: str) -> None:
        if not self.enabled:
            return
        now = time.time()
        if fraction < 1.0 and now - self.last < 0.08:
            return
        self.last = now

        filled = int(BAR_WIDTH * fraction)
        bar = "#" * filled + "." * (BAR_WIDTH - filled)
        text = f"\r  [{bar}] {fraction * 100:5.1f}%  {message[:38]:<38}"
        sys.stdout.write(text)
        sys.stdout.flush()

    def done(self) -> None:
        if self.enabled:
            sys.stdout.write("\r" + " " * (BAR_WIDTH + 58) + "\r")
            sys.stdout.flush()


def human_size(n: int) -> str:
    step = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if step < 1024 or unit == "GB":
            return f"{step:.1f} {unit}" if unit != "B" else f"{int(step)} B"
        step /= 1024
    return f"{step:.1f} GB"


def print_result(result: ProcessResult, verbose: bool = False) -> None:
    reduction = result.noise_reduction_db
    print(f"  engine          {result.engine_used}")
    print(f"  noise floor     {result.noise_floor_before_db:.1f} dB "
          f"-> {result.noise_floor_after_db:.1f} dB   ({reduction:+.1f} dB)")
    if result.clicks_repaired:
        print(f"  clicks repaired {result.clicks_repaired}")
    print(f"  speed           {result.elapsed_seconds:.1f}s for "
          f"{result.duration_seconds:.0f}s of media "
          f"({result.duration_seconds / max(result.elapsed_seconds, 0.01):.1f}x realtime)")

    if result.output:
        mark = "verified identical" if result.video_untouched else "NOT VERIFIED"
        print(f"  video stream    {mark}")
        print(f"  saved to        {result.output}")

    for warning in result.warnings:
        print(f"  note            {warning}")

    if verbose:
        print("  stages:")
        for stage in result.stages:
            print(f"    {stage.name:<12} {stage.seconds:6.2f}s  {stage.detail}")


def options_from_args(args) -> ProcessOptions:
    return ProcessOptions.from_preset(
        args.preset,
        engine=getattr(args, "engine", None),
        attenuation_db=getattr(args, "atten", None),
        declick=(False if getattr(args, "no_declick", False) else None),
        declick_sensitivity=getattr(args, "declick_sensitivity", None),
        high_pass_hz=getattr(args, "high_pass", None),
        normalize=(False if getattr(args, "no_normalize", False) else None),
        target_lufs=getattr(args, "lufs", None),
        keep_original_track=(
            False if getattr(args, "no_original_track", False) else None
        ),
    )


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_process(args) -> int:
    source = Path(args.input)
    if not source.exists():
        print(f"No such file: {source}", file=sys.stderr)
        return 1

    options = options_from_args(args)
    print(f"\n{source.name}")
    print(f"  preset          {options.preset}  "
          f"(reduction target {options.attenuation_db:.0f} dB)")

    bar = Bar(source.name, enabled=not args.quiet)
    try:
        result = process_file(source, options, output_path=args.output, progress=bar)
    except Exception as exc:  # noqa: BLE001
        bar.done()
        print(f"  failed          {exc}", file=sys.stderr)
        return 1

    bar.done()
    print_result(result, verbose=args.verbose)
    print()
    return 0


def cmd_batch(args) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a folder: {folder}", file=sys.stderr)
        return 1

    pattern = "**/*" if args.recursive else "*"
    files = sorted(
        f for f in folder.glob(pattern)
        if f.is_file() and f.suffix.lower() in MEDIA_SUFFIXES
    )
    if not files:
        print(f"No media files found in {folder}")
        return 0

    options = options_from_args(args)
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{len(files)} files, preset '{options.preset}'\n")
    succeeded = failed = 0
    started = time.time()

    for index, path in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {path.name}")
        destination = (
            out_dir / f"{path.stem}_clean{path.suffix}" if out_dir else None
        )
        bar = Bar(path.name, enabled=not args.quiet)
        try:
            result = process_file(path, options, output_path=destination, progress=bar)
            bar.done()
            print_result(result, verbose=args.verbose)
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            bar.done()
            print(f"  failed          {exc}", file=sys.stderr)
            failed += 1
            if args.stop_on_error:
                break
        print()

    elapsed = time.time() - started
    print(f"Finished: {succeeded} cleaned, {failed} failed, "
          f"{elapsed / 60:.1f} min total")
    return 1 if failed and args.stop_on_error else 0


def cmd_measure(args) -> int:
    """Report on a file without writing anything. Good for choosing a preset."""
    source = Path(args.input)
    if not source.exists():
        print(f"No such file: {source}", file=sys.stderr)
        return 1

    info = media.probe(source)
    print(f"\n{source.name}")
    print(f"  container       {info.container}")
    if info.has_video:
        print(f"  video           {info.video_codec} {info.width}x{info.height}")
    print(f"  audio           {info.audio_codec} "
          f"{info.sample_rate} Hz, {info.channels} ch")
    print(f"  duration        {info.duration:.1f}s")
    print(f"  size            {human_size(source.stat().st_size)}")

    options = ProcessOptions.from_preset("analyse_only")
    result = process_file(source, options)

    floor = result.noise_floor_before_db
    print(f"  noise floor     {floor:.1f} dBFS")
    print(f"  average level   {result.level_before_db:.1f} dBFS")
    headroom = result.level_before_db - floor
    print(f"  speech above floor  {headroom:.1f} dB")

    if headroom > 45:
        advice = "Already clean. Try 'gentle', or leave it alone."
    elif headroom > 30:
        advice = "Normal noise. 'screen_recording' will handle it."
    elif headroom > 18:
        advice = "Noisy. Try 'voiceover'."
    else:
        advice = "Very noisy. Try 'rescue', and expect some effect on the voice."
    print(f"\n  suggestion      {advice}\n")
    return 0


def cmd_engines(args) -> int:
    print("\nDenoise engines\n")
    for info in list_engines():
        mark = "available" if info.available else "not installed"
        print(f"  {info.name}")
        print(f"    key           {info.key}")
        print(f"    status        {mark} - {info.detail}")
        print(f"    strength      {info.strength}")
        print(f"    {info.description}")
        if not info.available and info.install_hint:
            print(f"    install with  {info.install_hint}")
        print()

    try:
        print(f"  'auto' currently resolves to: {resolve_engine_key('auto')}\n")
    except RuntimeError as exc:
        print(f"  {exc}\n")

    print("Presets\n")
    for key, cfg in PRESETS.items():
        print(f"  {key:<18} {cfg['label']}")
        print(f"  {'':<18} {cfg['blurb']}")
        print(f"  {'':<18} reduction target {cfg['attenuation_db']:.0f} dB\n")

    print(f"ffmpeg available: {media.ffmpeg_available()}\n")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    settings = get_settings()
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    print(f"\n  ClearTake is running at http://{host}:{port}")
    print("  Press Ctrl+C to stop.\n")

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="warning" if not args.verbose else "info",
    )
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cleartake",
        description="Remove background noise from video without touching the video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(sp) -> None:
        sp.add_argument("--preset", default="screen_recording", choices=list(PRESETS))
        sp.add_argument("--engine", default=None,
                        help="auto, deepfilternet, clearervoice, or spectral")
        sp.add_argument("--atten", type=float, default=None, metavar="DB",
                        help="How much noise to remove, in dB. 14 light, 24 normal, 42 heavy.")
        sp.add_argument("--no-declick", action="store_true",
                        help="Leave mouse clicks and keyboard alone.")
        sp.add_argument("--declick-sensitivity", type=float, default=None, metavar="X",
                        help="0 to 2.5. Above 2 starts eating hard consonants.")
        sp.add_argument("--high-pass", type=float, default=None, metavar="HZ")
        sp.add_argument("--no-normalize", action="store_true")
        sp.add_argument("--lufs", type=float, default=None, metavar="LUFS",
                        help="Loudness target. -16 for web, -14 for YouTube.")
        sp.add_argument("--no-original-track", action="store_true",
                        help="Do not keep the untouched audio as a second track.")
        sp.add_argument("-q", "--quiet", action="store_true")
        sp.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("process", help="Clean one file.")
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None)
    add_shared(p)
    p.set_defaults(func=cmd_process)

    b = sub.add_parser("batch", help="Clean every media file in a folder.")
    b.add_argument("folder")
    b.add_argument("-o", "--output-dir", default=None)
    b.add_argument("-r", "--recursive", action="store_true")
    b.add_argument("--stop-on-error", action="store_true")
    add_shared(b)
    b.set_defaults(func=cmd_batch)

    m = sub.add_parser("measure", help="Report on a file and suggest a preset.")
    m.add_argument("input")
    m.set_defaults(func=cmd_measure)

    e = sub.add_parser("engines", help="Show what is installed.")
    e.set_defaults(func=cmd_engines)

    s = sub.add_parser("serve", help="Start the web interface.")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--reload", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
