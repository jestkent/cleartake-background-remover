# ClearTake: design and plan

## The problem

Screen recordings and lesson videos carry three kinds of unwanted sound, and
they are three different problems wearing the same coat. Treating them as one
is the mistake that makes most noise-removal attempts disappointing.

| What you hear | What it actually is | What fixes it |
| --- | --- | --- |
| Hiss, fan, air conditioning, preamp whine | Stationary broadband noise. Sits still in the spectrum. | Spectral gate, or DeepFilterNet3. Solved. |
| Mouse clicks, keyboard, taps, lip smacks | Impulsive transients. Very short, very steep, broadband. | Time-domain click repair. Solved, see `declick.py`. |
| Other people talking in the background | Competing speech. | **Not a noise problem.** See phase 4. |

That third row is the one that matters most and is least understood.

A speech denoiser is trained to do exactly one thing: keep the speech, remove
everything else. Background conversation *is* speech. The model will identify
it, decide it is the signal rather than the noise, and carefully preserve it.
Running a denoiser harder does not help; it just damages the foreground voice
while leaving the chatter intact.

Removing a competing speaker requires **target speaker extraction**, which is a
separate class of model that needs a reference clip of the voice you want to
keep. It is scoped in phase 4 rather than pretended away.

## The guarantee

The stated requirement was that video quality must not change. This is not a
best-effort claim here, it is enforced.

The video stream is never decoded. `ffmpeg -c:v copy` moves the already
compressed packets from the source container to the output container
unchanged. The only lossy operation in the entire pipeline is the final audio
encode.

To prove it rather than assert it, `media.py::verify_video_untouched()` hashes
the video stream of both files with `ffmpeg -map 0:v -c copy -f md5`, which
hashes packet data without decoding. The runner calls this on every job and
raises if the hashes differ. A silent re-encode cannot ship.

Verified on the test clip:

```
source : 8a5affaf33aca178a4508702431a43bf
output : 8a5affaf33aca178a4508702431a43bf
```

## Architecture

```
                    ┌──────────────┐
   video file  ───► │   ffprobe    │  duration, codecs, stream layout
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ extract audio│  ffmpeg -vn, 48 kHz float32 WAV
                    └──────┬───────┘
                           │  (channels, samples) float32
              ┌────────────▼────────────┐
              │  declick                │  LPC residual + robust threshold
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  denoise engine         │  spectral / DeepFilterNet / MossFormer2
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  attenuation blend      │  hold reduction to N dB
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  high-pass + loudness   │  rumble cut, EBU R128 two-pass
              └────────────┬────────────┘
                           │
   original video ───┐     │  cleaned WAV
                     ▼     ▼
                ┌──────────────────┐
                │  remux -c:v copy │  video packets copied, never decoded
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │  hash verify     │  fail loudly if video changed
                └────────┬─────────┘
                         ▼
                   output file
                   track 1: Cleaned (default)
                   track 2: Original (untouched)
```

### Why the stages sit in that order

**Declick before denoise.** A click is a broadband impulse spread across every
frequency bin. Leave it in and it inflates the denoiser's estimate of the
steady noise floor, which costs reduction across the whole file, not just at
the click. Remove them first and the denoiser gets an honest picture.

**Blend immediately after denoise.** Everything downstream should operate on
the signal being shipped, not the full-strength version that is not.

**Loudness last.** Normalise earlier and the denoiser changes the level
underneath it, so the LUFS target quietly becomes something else.

### The attenuation blend

Running a denoiser at maximum is what produces the hollow, underwater,
swimming sound that makes cleaned audio obviously cleaned. The fix is to not
remove all of it:

```
output = cleaned + (original - cleaned) × 10^(-atten_db / 20)
```

The removed noise is added back at a controlled level, so a setting of 24 dB
gives 24 dB of reduction and leaves a quiet, natural floor underneath. A room
that sounds like a room, rather than a vacuum that opens and closes around
each word.

Measured tracking on the test clip:

| Requested | Achieved |
| --- | --- |
| 14 dB | 13.3 dB |
| 24 dB | 21.5 dB |
| 30 dB | 25.3 dB |
| 42 dB | 29.8 dB |
| 60 dB (full) | 31.8 dB |

The dial does roughly what it says until it runs into the engine's ceiling.

### Engines

Pluggable behind one interface, selected automatically by what is installed.

| Engine | Needs | Good for | Trade |
| --- | --- | --- | --- |
| Spectral gate | scipy only | Steady noise: fans, hiss, hum | Useless on non-stationary noise |
| DeepFilterNet3 | torch, ~500 MB | Everything typical. The default. | None worth mentioning |
| MossFormer2 48K | torch + GPU | Loud rooms, reverb, changing noise | Slower, takes liberties with the voice |

The spectral gate exists so the pipeline is testable the moment the repo is
cloned, before committing to a multi-gigabyte torch install. It is also the
right tool for a genuinely steady noise floor.

The spectral implementation uses a decision-directed SNR estimator (Ephraim
and Malah, 1984) rather than per-frame Wiener. Per-frame SNR estimates are
noisy, so the gain flickers between frames, which is heard as the warbling
musical noise that makes cheap noise reduction obvious. Building each frame's
SNR mostly from the previous frame's cleaned output smooths the estimate and
the artefact largely disappears.

## Build phases

### Phase 1 — Core pipeline ✅ done

- ffmpeg extract and remux with verified video passthrough
- Hand-written RIFF reader (the stdlib `wave` module cannot read ffmpeg's
  extensible float WAV)
- Click detection and repair
- Spectral gate with decision-directed gain
- Attenuation blend, high-pass, two-pass loudness
- Presets, `ProcessOptions`, metering, waveform extraction

**Measured on a synthetic clip with known ground truth** (hiss at -42 dBFS,
60 Hz hum, 14 clicks at recorded timestamps):

```
clicks repaired : 14 of 14, no false positives
noise floor     : -41.4 dB -> -59.1 dB   (17.7 dB reduction)
speech level    : essentially unchanged
video stream    : hash identical
throughput      : 9x realtime end to end
```

### Phase 2 — Interfaces ✅ done

- FastAPI service: upload with streaming to disk, job queue, progress polling,
  download, inline A/B preview, storage management
- Single worker thread, JSON-persisted history, atomic writes
- CLI: `process`, `batch`, `measure`, `engines`, `serve`
- React + Vite + TypeScript + Tailwind UI with before/after waveform

### Phase 3 — Model engines ⬜ install-gated

Code is written and registered. Install with:

```bash
pip install -r requirements-models.txt
```

ClearTake detects them on restart and `auto` upgrades itself. Nothing else to
change.

Remaining work once installed: benchmark DeepFilterNet3 against the spectral
gate on real footage, and tune per-preset attenuation defaults to whatever
sounds right rather than the current estimates.

### Phase 4 — Background voices ⬜ not started

The hard one, and the reason it is a separate phase rather than a setting.

Approach:

1. **Enrol the target voice.** Take 10 to 20 seconds of the foreground speaker
   alone, from the same recording where possible. Compute a speaker embedding
   (ECAPA-TDNN via SpeechBrain, or the embedding model shipped with
   ClearerVoice).
2. **Extract.** Feed audio plus embedding into a target speaker extraction
   model. ClearerVoice-Studio ships one. The `Hush` model on HuggingFace is a
   DeepFilterNet derivative trained specifically for background speaker
   suppression and is the lighter option worth benchmarking first.
3. **Blend conservatively.** These models hallucinate more than denoisers do.
   Gate the output behind a content-integrity check: transcribe before and
   after with faster-whisper and flag the job if the words changed. Never use
   at full strength.

Honest expectation: this works well when the background speaker is meaningfully
quieter than the foreground, and poorly when they are at similar levels. It is
not a solved problem, and anyone claiming otherwise is selling something.

### Phase 5 — Quality and convenience ⬜ ideas

- DNSMOS scoring per stage, so presets can be compared objectively rather than
  by ear
- Level-matched A/B in the UI, because a louder version always sounds better
  and that is a trap
- Dereverb stage (MelBand Roformer via `audio-separator`)
- Watch-folder mode: drop files in, cleaned versions appear
- Export a `.wav` sidecar for editors that prefer a separate audio import

## Deliberate non-goals

- **Real-time processing.** Offline means every stage can use long look-ahead
  and large models. Live noise suppression is a different tool.
- **Cloud or hosted API.** The whole point is that footage never leaves the
  machine. For classroom recordings with students in them that is a
  requirement, not a preference.
- **Music.** These are speech models. They will destroy singing and
  instruments. A vocal-isolation path would be a separate mode.
- **Authentication.** Binds to 127.0.0.1. Anything beyond that needs a reverse
  proxy and real auth in front.

## References

- DeepFilterNet3 — https://github.com/Rikorose/DeepFilterNet
- ClearerVoice-Studio — https://github.com/modelscope/ClearerVoice-Studio
- Hush (background speaker suppression) — https://huggingface.co/Aliados/hush
- Ephraim & Malah, decision-directed estimation, IEEE ASSP 1984
- EBU R128 loudness — https://tech.ebu.ch/docs/r/r128.pdf
