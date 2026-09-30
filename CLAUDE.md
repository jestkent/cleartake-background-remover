# CLAUDE.md

Context for Claude Code working in this repo. Read this before changing anything.

## What this is

ClearTake strips background noise out of video without re-encoding the video.
Drop a file in, get the same file back with clean audio. Local only, no API
calls, no per-use cost.

Built for cleaning screen recordings and lesson videos: mouse clicks, keyboard,
fan hum, room tone.

## The one rule that cannot break

**The video stream is never re-encoded.**

`backend/pipeline/media.py` remuxes with `-c:v copy`, so video packets are
copied across rather than decoded and re-compressed. `verify_video_untouched()`
hashes the video stream of the source and the output and compares them. The
runner calls it on every job and raises if they differ.

If you touch anything in `media.py`, run `tests/test_pipeline.py` before
committing. A silent re-encode would be the worst possible bug here, because
the output would still look fine and the quality loss would only show up
generations later.

## Layout

```
backend/
  settings.py          config.yaml loading, path resolution
  main.py              FastAPI app, serves the built UI from web/dist
  cli.py               process / batch / measure / engines / serve
  jobs.py              single-worker queue, JSON persistence
  schemas.py           pydantic request and response shapes
  pipeline/
    media.py           ffmpeg. extract, remux, hash-verify, loudnorm
    audio.py           RIFF reader/writer, attenuation blend, metering
    declick.py         impulsive noise repair (mouse clicks, keyboard)
    runner.py          stage orchestration, presets, ProcessOptions
    engines/
      base.py          DenoiseEngine ABC
      spectral.py      numpy/scipy only, no model download
      deepfilternet.py DeepFilterNet3, python API with CLI fallback
      clearervoice.py  MossFormer2, chunked for long files
web/                   React + Vite + TypeScript + Tailwind
scripts/               setup and test-clip generation
```

## Stage order, and why

```
extract -> declick -> denoise -> attenuation blend -> high-pass ->
loudness -> remux -> verify
```

- **Declick before denoise.** A click is a broadband impulse. Leave it in and
  it drags the denoiser's noise estimate upward, costing reduction across the
  whole file.
- **Blend right after denoise.** Every later stage should work on the signal
  you are actually shipping, not the full-strength version you are not.
- **Loudness last.** Run it earlier and the denoiser moves the level underneath
  it, so the LUFS target silently becomes something else.

Do not reorder these without a reason you can write down.

## Engine contract

An engine takes `(channels, samples)` float32 and returns the same shape,
denoised. That is all. No blending, no loudness, no file handling, and
**no length changes** — the runner mixes engine output against the original
sample for sample, so a length change smears the entire track.

`engines/__init__.py` holds the registry. `auto` walks `AUTO_PREFERENCE` and
takes the first engine that reports available, so a fresh clone runs on the
spectral gate and upgrades itself the moment DeepFilterNet is installed.

To add an engine: subclass `DenoiseEngine`, implement `is_available()` and
`denoise()`, register it in `REGISTRY`, and add it to `AUTO_PREFERENCE` at the
right quality position.

## Traps already hit, do not reintroduce

1. **`wave` cannot read ffmpeg's float WAV.** ffmpeg writes
   `WAVE_FORMAT_EXTENSIBLE` (0xFFFE), which the stdlib module refuses outright.
   `audio.py` has a hand-written RIFF parser. Do not "simplify" it back to
   `wave.open`.

2. **`np.correlate` is O(n²).** It is a direct convolution, not an FFT. On a
   20 minute file at 48 kHz that is the difference between a second and an
   hour. `declick.py` uses `_autocorrelation()`, which goes through the FFT.
   Same trap applies to `scipy.signal.medfilt`, which is O(n·k) — a running
   mean via `uniform_filter1d` is used instead.

3. **Percentile is not a noise floor.** Noise power in a frequency bin is
   roughly exponentially distributed, so its 10th percentile sits at about an
   eighth of its mean. Estimate the noise that way and every noise frame looks
   like it has 8:1 SNR, the gate decides speech is present, and you get 3 dB of
   reduction instead of 30. `spectral.py::_estimate_noise` ranks whole frames
   by energy and averages the quiet ones. This single bug cost 28 dB.

4. **MP4 has no per-track title.** It shows `handler_name` instead. `media.py`
   sets both `title` and `handler_name` so tracks are labelled in MKV and MP4
   alike.

5. **One worker thread, deliberately.** Models hold GPU state that is not
   thread-safe, and two jobs racing for VRAM is how you get an OOM halfway
   through a long file. Queueing is correct behaviour here. Do not add a pool.

6. **`is_available()` only proves the module exists.** `find_spec("df")`
   happily returns a spec for a package that raises on import, so a broken
   DeepFilterNet advertises itself as available and then fails inside the job
   instead of at startup. If you are ever debugging "reports available, always
   fails", import it for real rather than trusting the engine list.

## Running it

```bash
pip install -r requirements.txt
python scripts/make_test_clip.py          # synthetic clip with known problems
python -m backend.cli process samples/noisy_test.mp4
python -m backend.cli serve               # web UI on :7788
```

Frontend dev with hot reload (API must be running separately):

```bash
cd web && npm install && npm run dev      # :5173, proxies /api to :7788
```

The UI is served from `web/dist`. After changing anything in `web/src`, run
`npm run build` or the served version stays stale.

## Environment and isolation

Repo: <https://github.com/jestkent/cleartake-background-remover> (public).

Everything this project installs lives in `.venv` at the repo root. Nothing
goes into the system interpreter, and nothing outside this directory is
modified. Verified 2026-09-29: `include-system-site-packages = false`, and the
system Python's `site-packages` had nothing written to it by the install.

**Never run `pip install` for this project outside the venv.** The system
Python on this machine is a working ML environment (torch with CUDA, numpy
2.x). `requirements-models.txt` pulls DeepFilterNet, which pins `numpy<2.0`,
so one stray global install would downgrade numpy machine wide and break
unrelated projects. Always go through the venv interpreter:

```bash
.venv/Scripts/python.exe -m pip install ...   # Windows
./.venv/bin/python -m pip install ...         # macOS, Linux
```

`pip config set global.require-virtualenv true` makes pip refuse to install
outside a venv at all. Worth setting on any machine with a shared system
Python.

### Pins this environment needs

**`scipy==1.15.3`.** DeepFilterNet pins `numpy<2.0`, so installing it
downgrades numpy to 1.26.4. A scipy built against numpy 2.x then fails to
import and takes the whole pipeline with it, with a misleading
`np.long` AttributeError from deep inside `scipy.sparse`. 1.15.3 supports
numpy 1.26 and still satisfies `requirements.txt`.

### The torchaudio shim

`.venv/Lib/site-packages/torchaudio/backend/` is hand written and belongs to
no package. DeepFilterNet 0.5.6, its last release, does
`from torchaudio.backend.common import AudioMetaData` at import time, and
torchaudio deleted `backend` in 2.2. Python 3.13 forces torchaudio 2.5 or
newer, so there is no version pair where DeepFilterNet imports unpatched. The
shim restores that one symbol and nothing else.

`.venv` is gitignored, so **rebuilding it silently loses the shim.** The
symptom is confusing rather than obvious: `backend.cli engines` still reports
DeepFilterNet as available, and then every job that touches it dies with
`ModuleNotFoundError: No module named 'torchaudio.backend'`. See trap 6.

The `deepFilter` CLI fallback does not rescue this. The console script pip
installs is the same Python entry point and fails identically. Only the
standalone Rust binary from the project's GitHub releases would sidestep it.

### Written outside the repo

Three caches. None of them affect other software, and all are safe to delete
at the cost of a re-download.

| Path | Size | What |
| --- | --- | --- |
| `%LOCALAPPDATA%\DeepFilterNet` | 8.3 MB | DeepFilterNet3 checkpoint, fetched on first use |
| `%LOCALAPPDATA%\pip\cache` | shared | downloaded and locally built wheels |
| `~/.cargo/registry` | 7.3 MB | crate sources, from building `deepfilterlib` |

`deepfilterlib` ships no Windows wheel on PyPI, so pip compiles it from
source. That needs a Rust toolchain already on the machine. Nothing installs
one for you, and the build fails without it.

To keep the model checkpoint inside the repo instead, pass a path to
`init_df(model_base_dir=...)` in `engines/deepfilternet.py`, which currently
passes `None` and so lands in the user cache directory.

### torch is CPU only here

PyPI's Windows torch wheel carries no CUDA, so `requirements-models.txt`
installs `torch+cpu` even on a CUDA machine. DeepFilterNet3 runs faster than
realtime on CPU and this tool is built for voice, so that is a reasonable
place to stop. For the GPU build, install torch and torchaudio from
`download.pytorch.org` instead.

## Testing changes

`python -m pytest tests/ -v` covers the RIFF round trip, the attenuation blend
maths, click detection against known injected positions, and the video-hash
guarantee.

For an audible check, `scripts/make_test_clip.py` generates a clip with hiss,
hum, and clicks at recorded timestamps, so you can verify against ground truth
instead of guessing.

## Known limits, stated plainly

- **Background voices are not removed.** A denoiser is trained to keep speech
  and remove everything else, so a background conversation is exactly what it
  protects. This needs target speaker extraction, which is a different model
  and a different problem. See `plan.md` phase 4. Do not file this as a bug.
- Stereo channels are processed independently, which can shift the stereo image
  slightly on heavily-panned material. Fine for voice, which is where this tool
  lives.
- The spectral gate handles steady noise only. For anything non-stationary,
  install DeepFilterNet.
- **Mouth and lip clicks are not removed, and `declick.py` cannot be tuned to
  do it.** The detector finds impulses through linear-prediction error, which
  spikes on broadband transients. A mouth click is a short *resonant* burst,
  so the predictor handles it comfortably and the residual barely moves.
  Measured on a real five minute recording: spectral flatness 0.072 in the
  source and 0.020 after denoising, against 1.0 for a true impulse, centroids
  spread over 3 to 7 kHz, durations 1.4 to 9.7 ms, about 38 per minute. Of
  the ten loudest, zero cleared the sharpness gate, and the repair count was
  identical at sensitivity 0.6, at 1.8, and with the stage run a second time
  after denoising. Removing these needs spectral repair over the offending
  time-frequency cells, which is a different stage, not a bigger number.
- **The sharpness gate only sees clicks well under its baseline window.** It
  compares a sample against a 4 ms running mean, which is the same width as
  the widest click `max_click_ms` permits. An event that fills its own window
  sits near the window mean and can never clear the 4x ratio, so the real
  ceiling is nearer 2 ms than the nominal 3 ms. Scaling the window with the
  click length does fix that in isolation, but raising `max_click_ms` to 12 ms
  alongside it put 29 false positives into 20 seconds of clean synthetic
  speech. Plosives occupy the same 5 to 15 ms range as the clicks you would be
  reaching for, so this is recorded as a limit rather than patched.
- **Removing hiss makes whatever it was covering louder.** On the same
  recording the high-frequency floor fell 19.1 dB and pre-existing clicks went
  from 3.2 to 8.2 prominence, a factor of 2.55, without a single new click
  being created. Expect "the denoiser added clicks" reports that are really
  unmasking, and check the source before believing them.

## Style

Plain Python 3.10+, type hints on public functions, dataclasses over dicts for
anything structured. Comments explain *why*, not *what* — the code already says
what. No em-dashes in output text or comments.
