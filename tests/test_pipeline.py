"""
Tests for the parts that must not break.

The important one is `TestVideoIntegrity`. Everything else here is ordinary
correctness checking; that class guards the single promise ClearTake makes,
which is that your video comes back unchanged. A regression there would be
invisible in the output and would only show up as quality loss several
generations later, so it gets a test that hashes real files.

    python -m pytest tests/ -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.pipeline import media  # noqa: E402
from backend.pipeline.audio import (  # noqa: E402
    apply_attenuation_limit,
    noise_floor_db,
    peak_envelope,
    read_wav,
    rms_db,
    write_wav,
)
from backend.pipeline.declick import declick  # noqa: E402
from backend.pipeline.engines import resolve_engine_key  # noqa: E402
from backend.pipeline.engines.spectral import SpectralGateEngine  # noqa: E402
from backend.pipeline.runner import PRESETS, ProcessOptions, process_file  # noqa: E402

SAMPLE_RATE = 48_000

pytestmark = pytest.mark.skipif(
    not media.ffmpeg_available(), reason="ffmpeg is not installed"
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _synth(seconds: float = 4.0, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return (clean, noisy) mono signals with pauses between bursts."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    clean = np.zeros(n)

    cursor = 0.2
    while cursor < seconds - 0.5:
        length = 0.4
        start, end = int(cursor * SAMPLE_RATE), int((cursor + length) * SAMPLE_RATE)
        local = t[start:end] - t[start]
        burst = (
            np.sin(2 * np.pi * 130 * local)
            + 0.5 * np.sin(2 * np.pi * 260 * local)
            + 0.25 * np.sin(2 * np.pi * 640 * local)
        )
        envelope = np.clip(np.minimum(local / 0.02, (local[-1] - local) / 0.05), 0, 1)
        clean[start:end] = burst * envelope * 0.4
        cursor += length + 0.35

    noisy = clean + rng.normal(0, 10 ** (-40 / 20), n)
    return clean.astype(np.float32), noisy.astype(np.float32)


@pytest.fixture(scope="module")
def test_video(tmp_path_factory) -> Path:
    """A real H.264 clip with noisy audio, built once for the whole module."""
    folder = tmp_path_factory.mktemp("cleartake")
    _, noisy = _synth(4.0)
    wav = folder / "audio.wav"
    write_wav(wav, noisy[None, :], SAMPLE_RATE)

    video = folder / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4",
            "-i", str(wav),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video),
        ],
        check=True,
    )
    return video


# --------------------------------------------------------------------------
# WAV round trip
# --------------------------------------------------------------------------


class TestAudioIO:
    def test_float_roundtrip_is_exact(self, tmp_path):
        data = np.random.default_rng(1).uniform(-0.9, 0.9, (2, 5000)).astype(np.float32)
        path = write_wav(tmp_path / "x.wav", data, SAMPLE_RATE)
        back, sr = read_wav(path)

        assert sr == SAMPLE_RATE
        assert back.shape == data.shape
        # Float32 in, float32 out, so this should be bit-exact rather than close.
        np.testing.assert_array_equal(back, data)

    def test_reads_ffmpeg_extensible_float(self, tmp_path):
        """
        ffmpeg writes float WAV as WAVE_FORMAT_EXTENSIBLE, which the stdlib
        `wave` module refuses outright. This is the bug that broke the first
        run of the pipeline, so it gets a test against a file ffmpeg actually
        produced rather than one we wrote ourselves.
        """
        source = tmp_path / "src.wav"
        write_wav(source, np.zeros((1, 2400), dtype=np.float32), SAMPLE_RATE)

        converted = tmp_path / "ffmpeg.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(source),
             "-c:a", "pcm_f32le", str(converted)],
            check=True,
        )
        data, sr = read_wav(converted)
        assert sr == SAMPLE_RATE
        assert data.shape[1] > 0

    @pytest.mark.parametrize("codec", ["pcm_s16le", "pcm_s24le", "pcm_f32le"])
    def test_reads_every_depth_ffmpeg_writes(self, tmp_path, codec):
        tone = np.sin(
            2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE
        ).astype(np.float32) * 0.5
        source = tmp_path / "tone.wav"
        write_wav(source, tone[None, :], SAMPLE_RATE)

        converted = tmp_path / f"{codec}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(source),
             "-c:a", codec, str(converted)],
            check=True,
        )
        data, sr = read_wav(converted)
        assert sr == SAMPLE_RATE
        # Amplitude should survive the format change to within a quantisation step.
        assert abs(float(np.abs(data).max()) - 0.5) < 0.01


# --------------------------------------------------------------------------
# attenuation blend
# --------------------------------------------------------------------------


class TestAttenuationBlend:
    def test_zero_returns_the_original_untouched(self):
        original = np.random.default_rng(2).uniform(-1, 1, (1, 1000)).astype(np.float32)
        processed = np.zeros_like(original)
        np.testing.assert_array_equal(
            apply_attenuation_limit(original, processed, 0.0), original
        )

    def test_full_strength_returns_the_processed_signal(self):
        original = np.ones((1, 1000), dtype=np.float32)
        processed = np.zeros((1, 1000), dtype=np.float32)
        np.testing.assert_array_equal(
            apply_attenuation_limit(original, processed, 60.0), processed
        )

    @pytest.mark.parametrize("atten_db", [6.0, 12.0, 20.0, 30.0])
    def test_removed_noise_returns_at_the_requested_level(self, atten_db):
        """
        With a processed signal of pure silence, everything is 'removed
        noise', so the output should be the original scaled by exactly the
        requested attenuation. That makes the maths checkable in isolation.
        """
        original = np.ones((1, 1000), dtype=np.float32)
        processed = np.zeros((1, 1000), dtype=np.float32)

        result = apply_attenuation_limit(original, processed, atten_db)
        expected = 10 ** (-atten_db / 20)
        assert np.allclose(result, expected, rtol=1e-5)

    def test_monotonic_in_strength(self):
        _, noisy = _synth(2.0)
        engine = SpectralGateEngine()
        cleaned = engine.denoise(noisy[None, :], SAMPLE_RATE)

        floors = [
            noise_floor_db(
                apply_attenuation_limit(noisy[None, :], cleaned, db), SAMPLE_RATE
            )
            for db in (0, 10, 20, 30)
        ]
        # More attenuation must never raise the noise floor.
        assert all(b <= a + 0.5 for a, b in zip(floors, floors[1:]))


# --------------------------------------------------------------------------
# declick
# --------------------------------------------------------------------------


class TestDeclick:
    def test_finds_clicks_at_known_positions(self):
        clean, _ = _synth(4.0)
        rng = np.random.default_rng(11)

        positions = [int(p * SAMPLE_RATE) for p in (0.7, 1.4, 2.1, 2.8, 3.3)]
        noisy = clean.copy()
        for at in positions:
            length = 60
            noisy[at : at + length] += (
                rng.normal(0, 1, length) * np.exp(-np.linspace(0, 9, length)) * 0.6
            )

        repaired, found = declick(noisy[None, :], SAMPLE_RATE, sensitivity=1.2)

        # Every injected click should be found, with little invention.
        assert len(positions) <= found <= len(positions) + 2
        assert repaired.shape == noisy[None, :].shape

        # The repaired signal should sit closer to the clean original.
        before = float(np.abs(noisy - clean).sum())
        after = float(np.abs(repaired[0] - clean).sum())
        assert after < before

    def test_leaves_clean_speech_alone(self):
        clean, _ = _synth(3.0)
        _, found = declick(clean[None, :], SAMPLE_RATE, sensitivity=1.0)
        assert found <= 2, "Clean speech should not trip the click detector"

    def test_disabled_at_zero_sensitivity(self):
        _, noisy = _synth(2.0)
        out, found = declick(noisy[None, :], SAMPLE_RATE, sensitivity=0.0)
        assert found == 0
        np.testing.assert_array_equal(out, noisy[None, :])

    def test_fast_enough_for_long_files(self):
        """
        A guard against the O(n^2) autocorrelation and O(n*k) median filter
        that made the first version take 37 seconds for 12 seconds of audio.
        """
        import time

        _, noisy = _synth(20.0)
        start = time.perf_counter()
        declick(noisy[None, :], SAMPLE_RATE, sensitivity=1.2)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Declick took {elapsed:.1f}s for 20s of audio"


# --------------------------------------------------------------------------
# spectral engine
# --------------------------------------------------------------------------


class TestSpectralEngine:
    def test_reduces_the_noise_floor_substantially(self):
        _, noisy = _synth(4.0)
        engine = SpectralGateEngine()
        cleaned = engine.denoise(noisy[None, :], SAMPLE_RATE)

        before = noise_floor_db(noisy[None, :], SAMPLE_RATE)
        after = noise_floor_db(cleaned, SAMPLE_RATE)
        assert before - after > 15.0, f"Only {before - after:.1f} dB of reduction"

    def test_preserves_the_speech_level(self):
        _, noisy = _synth(4.0)
        cleaned = SpectralGateEngine().denoise(noisy[None, :], SAMPLE_RATE)
        # Overall RMS is dominated by the speech, so it should barely move.
        assert abs(rms_db(noisy[None, :]) - rms_db(cleaned)) < 3.0

    def test_never_changes_length(self):
        for seconds in (0.5, 2.0, 5.0):
            _, noisy = _synth(seconds)
            cleaned = SpectralGateEngine().denoise(noisy[None, :], SAMPLE_RATE)
            assert cleaned.shape == noisy[None, :].shape

    def test_handles_stereo(self):
        _, noisy = _synth(2.0)
        stereo = np.vstack([noisy, noisy * 0.8])
        cleaned = SpectralGateEngine().denoise(stereo, SAMPLE_RATE)
        assert cleaned.shape == stereo.shape


# --------------------------------------------------------------------------
# the guarantee
# --------------------------------------------------------------------------


class TestVideoIntegrity:
    """The promise ClearTake makes. If these fail, the tool is broken."""

    def test_video_stream_hash_is_unchanged(self, test_video, tmp_path):
        result = process_file(
            test_video,
            ProcessOptions.from_preset("screen_recording"),
            output_path=tmp_path / "out.mp4",
        )

        source_hash = media.video_stream_hash(test_video)
        output_hash = media.video_stream_hash(result.output)

        assert source_hash == output_hash, "The video stream was modified"
        assert result.video_untouched is True

    def test_runner_reports_the_verification(self, test_video, tmp_path):
        result = process_file(
            test_video,
            ProcessOptions.from_preset("gentle"),
            output_path=tmp_path / "gentle.mp4",
        )
        assert result.video_untouched is True

    def test_verify_catches_a_real_reencode(self, test_video, tmp_path):
        """
        Prove the check has teeth. Deliberately re-encode the video and
        confirm the verifier notices. Without this, a verifier that always
        returned True would pass every other test in this class.
        """
        reencoded = tmp_path / "reencoded.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(test_video),
             "-c:v", "libx264", "-crf", "40", "-c:a", "copy", str(reencoded)],
            check=True,
        )
        assert media.verify_video_untouched(test_video, reencoded) is False

    def test_output_carries_both_audio_tracks(self, test_video, tmp_path):
        result = process_file(
            test_video,
            ProcessOptions.from_preset("screen_recording"),
            output_path=tmp_path / "tracks.mp4",
        )
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", result.output],
            capture_output=True, text=True, check=True,
        )
        assert len([l for l in probe.stdout.strip().splitlines() if l]) == 2

    def test_single_track_when_asked(self, test_video, tmp_path):
        options = ProcessOptions.from_preset("screen_recording")
        options.keep_original_track = False
        result = process_file(test_video, options, output_path=tmp_path / "one.mp4")

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", result.output],
            capture_output=True, text=True, check=True,
        )
        assert len([l for l in probe.stdout.strip().splitlines() if l]) == 1


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


class TestPipeline:
    def test_produces_a_playable_file(self, test_video, tmp_path):
        result = process_file(
            test_video,
            ProcessOptions.from_preset("screen_recording"),
            output_path=tmp_path / "out.mp4",
        )
        output = Path(result.output)
        assert output.exists() and output.stat().st_size > 0

        info = media.probe(output)
        assert info.has_video and info.has_audio
        assert info.duration > 0

    def test_reports_real_measurements(self, test_video, tmp_path):
        result = process_file(
            test_video,
            ProcessOptions.from_preset("voiceover"),
            output_path=tmp_path / "vo.mp4",
        )
        assert result.noise_reduction_db > 5.0
        assert result.noise_floor_after_db < result.noise_floor_before_db
        assert len(result.waveform_before) == 900
        assert len(result.waveform_after) == 900
        assert result.elapsed_seconds > 0
        assert result.stages

    def test_measure_only_writes_nothing(self, test_video):
        result = process_file(test_video, ProcessOptions.from_preset("analyse_only"))
        assert result.output == ""
        assert result.noise_floor_before_db < 0

    def test_rejects_a_missing_file(self):
        with pytest.raises(FileNotFoundError):
            process_file("does_not_exist.mp4", ProcessOptions())

    @pytest.mark.parametrize("preset", [p for p in PRESETS if p != "analyse_only"])
    def test_every_preset_runs(self, test_video, tmp_path, preset):
        result = process_file(
            test_video,
            ProcessOptions.from_preset(preset),
            output_path=tmp_path / f"{preset}.mp4",
        )
        assert Path(result.output).exists()
        assert result.video_untouched


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------


class TestSupport:
    def test_auto_resolves_to_something_usable(self):
        assert resolve_engine_key("auto") in {
            "spectral", "deepfilternet", "clearervoice"
        }

    def test_unknown_engine_is_rejected(self):
        with pytest.raises(ValueError):
            resolve_engine_key("not_a_real_engine")

    def test_presets_are_well_formed(self):
        for name, config in PRESETS.items():
            assert {"label", "blurb", "attenuation_db"} <= set(config), name
            assert 0 <= config["attenuation_db"] <= 60, name

    def test_peak_envelope_is_fixed_width(self):
        for length in (100, 48_000, 480_000):
            data = np.random.default_rng(5).uniform(-1, 1, (1, length))
            assert len(peak_envelope(data)) == 900

    def test_probe_reads_real_metadata(self, test_video):
        info = media.probe(test_video)
        assert info.has_video and info.has_audio
        assert info.video_codec == "h264"
        assert 3.0 < info.duration < 5.0
        assert info.width == 320 and info.height == 240
