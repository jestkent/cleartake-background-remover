# ClearTake

Remove background noise from video without touching the video.

Drop a file in, get the same file back with clean audio. The picture is copied
across byte for byte and verified afterwards, so editing a cleaned file is no
different from editing the original.

Runs entirely on your own machine. No uploads, no API keys, no per-use cost.

---

## What it fixes

- **Hiss, fan noise, air conditioning, preamp whine** — solved
- **Mouse clicks, keyboard, taps** — solved, repaired individually
- **Low rumble and desk thumps** — solved
- **Other people talking in the background** — *not yet*, see below

That last one is worth reading about before you start. A speech denoiser is
trained to keep speech and remove everything else, so a background
conversation is precisely the thing it protects. Turning the strength up
damages your voice without touching theirs. This needs a different kind of
model, and it is scoped as phase 4 in `plan.md`.

---

## Install

Needs **Python 3.10+** and **ffmpeg** on your PATH.

```bash
git clone https://github.com/jestkent/cleartake-background-remover.git cleartake
cd cleartake
pip install -r requirements.txt
```

Check ffmpeg is visible:

```bash
ffmpeg -version
```

If it is not, install it (`winget install ffmpeg` on Windows, `brew install
ffmpeg` on macOS, `apt install ffmpeg` on Debian) or set `paths.ffmpeg` in
`config.yaml` to the full path.

### Better denoising (optional, ~500 MB)

The base install runs the spectral gate, which handles steady noise well. For
everything else, install DeepFilterNet3:

```bash
pip install -r requirements-models.txt
```

Restart ClearTake and it picks the new engine up on its own. Nothing to
configure.

---

## Try it in 60 seconds

Generate a test clip with known problems, then clean it:

```bash
python scripts/make_test_clip.py
python -m backend.cli process samples/noisy_test.mp4
```

You should see something close to:

```
  engine          Spectral gate
  noise floor     -41.4 dB -> -59.1 dB   (+17.7 dB)
  clicks repaired 14
  speed           1.3s for 12s of media (9.2x realtime)
  video stream    verified identical
  saved to        workspace/outputs/noisy_test_clean.mp4
```

The test clip has 14 clicks injected at recorded timestamps, so "14 repaired"
is a real result rather than a plausible-looking number.

---

## Use it

### Web interface

```bash
python -m backend.cli serve
```

Open <http://127.0.0.1:7788>. Drag a file in, pick what kind of recording it
is, and wait. When it finishes you get a before/after waveform, the measured
reduction, and a side-by-side player to hear both.

On Windows you can double-click `run.bat` instead.

### Command line

```bash
# one file
python -m backend.cli process lesson.mp4

# a whole folder, overnight
python -m backend.cli batch ./recordings --preset screen_recording

# check a file and get a preset suggestion, without writing anything
python -m backend.cli measure lesson.mp4

# what is installed
python -m backend.cli engines
```

Useful flags:

| Flag | Does |
| --- | --- |
| `--preset` | `gentle`, `screen_recording`, `voiceover`, `rescue` |
| `--atten 30` | How much noise to remove, in dB. 14 light, 24 normal, 42 heavy. |
| `--no-declick` | Leave mouse clicks alone |
| `--lufs -14` | Loudness target. -14 for YouTube. |
| `--no-original-track` | Do not keep the untouched audio as a second track |
| `-o out.mp4` | Where to write it |

---

## Presets

| Preset | For | Reduction |
| --- | --- | --- |
| `gentle` | Audio that is already decent. Least risk to the voice. | 14 dB |
| `screen_recording` | Tutorials and lessons. Clicks, keyboard, fan. **Default.** | 24 dB |
| `voiceover` | Narration over a quiet room | 30 dB |
| `rescue` | Genuinely bad audio. It will show on the voice. | 42 dB |

Not sure? Run `measure` and it will suggest one based on how far the speech
sits above the noise floor.

---

## The output file

By default you get **two audio tracks**:

1. **Cleaned** — flagged default, so any player picks it up automatically
2. **Original** — untouched, stream-copied

That gives you an instant A/B inside Premiere, Resolve, or CapCut, and a way
back if a pass went too far. It costs a few MB. Turn it off with
`--no-original-track` or the toggle in the UI.

The video stream is identical to your source. Verify it yourself:

```bash
ffmpeg -i original.mp4 -map 0:v -c copy -f md5 -
ffmpeg -i cleaned.mp4  -map 0:v -c copy -f md5 -
```

Same hash. ClearTake runs this check on every job and refuses to report
success if it ever fails.

---

## Fine tuning

If the result sounds hollow, underwater, or like it is swimming, the
attenuation is too high. Drop it to 18 or 20. Removing less noise almost
always sounds better than removing all of it.

If hard consonants sound soft or clipped, lower the click sensitivity. Above
about 2.0 it starts mistaking plosives for clicks.

If it sounds thin, lower the rumble cut or set it to 0.

---

## Configuration

Everything lives in `config.yaml`. Nothing is hardcoded to a machine, so the
same checkout runs anywhere.

```yaml
paths:
  ffmpeg: ffmpeg              # or C:\ffmpeg\bin\ffmpeg.exe
  outputs: ./workspace/outputs
server:
  port: 7788
audio:
  output_codec: aac
  output_bitrate: 256k
engines:
  device: auto                # auto | cuda | cpu
```

Environment variables override the file: `CLEARTAKE_PORT`, `CLEARTAKE_HOST`,
`CLEARTAKE_WORK_DIR`, `FFMPEG_PATH`.

---

## Developing

```bash
# backend with auto-reload
python -m backend.cli serve --reload

# frontend with hot reload, in a second terminal
cd web && npm install && npm run dev     # :5173, proxies /api to :7788

# tests
python -m pytest tests/ -v
```

The UI is served from `web/dist`. After changing anything in `web/src`, run
`npm run build` or the served version stays stale.

API docs are at <http://127.0.0.1:7788/docs> while the server is running.

See `CLAUDE.md` for architecture notes and the traps already hit, and
`plan.md` for the design reasoning and what is planned next.

---

## Disk use

Uploaded copies pile up in `workspace/uploads`. The UI has a button to clear
them, or:

```bash
curl -X POST http://127.0.0.1:7788/api/storage/clear-uploads
```

Cleaned outputs are left alone.

---

## A note on privacy

ClearTake binds to `127.0.0.1` and makes no outbound network calls during
processing. Footage stays on the machine. If you are cleaning recordings that
have students or colleagues in them, that is the reason to run this locally
rather than uploading to a web service.

Do not expose the server on a public interface. There is no authentication,
because there is nothing here to authenticate against.
