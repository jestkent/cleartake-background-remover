"""
ClearerVoice-Studio / MossFormer2. The heavy option.

Where DeepFilterNet3 is fast and faithful, MossFormer2 SE 48K is slower
and more aggressive. Reach for it when DeepFilterNet leaves too much
behind: loud rooms, strong reverb, noise that moves around instead of
sitting still.

The trade is real. A stronger model takes more liberties with the voice,
and on an already-decent recording it can make things sound processed
where DeepFilterNet would have sounded natural. Try DeepFilterNet first;
come here when it is not enough.

Long files are processed in overlapping chunks with crossfades at the
seams, because the model's memory footprint grows with input length and a
twenty minute lecture will otherwise exhaust an 8 GB card.
"""

from __future__ import annotations

import tempfile
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from backend.pipeline.audio import read_wav, write_wav
from backend.pipeline.engines.base import DenoiseEngine, ProgressFn

CHUNK_SECONDS = 30.0
OVERLAP_SECONDS = 1.0


class ClearerVoiceEngine(DenoiseEngine):
    key = "clearervoice"
    name = "ClearerVoice MossFormer2"
    description = (
        "Heavy duty. For loud rooms, reverb, and noise that keeps changing. "
        "Slower, and takes more liberties with the voice."
    )
    install_hint = "pip install clearvoice"
    requires_gpu = True
    native_sample_rate = 48_000
    strength = "heavy"

    _pipeline = None

    def __init__(self, model_name: str = "MossFormer2_SE_48K", device: str = "auto"):
        self.model_name = model_name
        self.device = device

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if find_spec("clearvoice") is None:
            return False, "Not installed. See requirements-models.txt."
        if find_spec("torch") is None:
            return False, "torch is missing."
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return True, f"Ready on {name}."
            return True, "Ready on CPU. Expect this to be slow."
        except Exception as exc:
            return False, f"torch failed to load: {exc}"

    @classmethod
    def _load(cls, model_name: str):
        if cls._pipeline is not None:
            return cls._pipeline
        from clearvoice import ClearVoice  # type: ignore

        cls._pipeline = ClearVoice(
            task="speech_enhancement", model_names=[model_name]
        )
        return cls._pipeline

    # ----------------------------------------------------------------- #

    def denoise(
        self,
        data: np.ndarray,
        sample_rate: int,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        if find_spec("clearvoice") is None:
            raise RuntimeError(
                "ClearerVoice is not installed. Run:\n"
                "  pip install -r requirements-models.txt"
            )

        data = np.atleast_2d(np.asarray(data, dtype=np.float32))
        target_len = data.shape[-1]

        self._report(progress, 0.05, f"Loading {self.model_name}")
        pipeline = self._load(self.model_name)

        chunk = int(CHUNK_SECONDS * sample_rate)
        overlap = int(OVERLAP_SECONDS * sample_rate)
        out = np.zeros_like(data)

        for ch in range(data.shape[0]):
            out[ch] = self._process_channel(
                pipeline, data[ch], sample_rate, chunk, overlap, progress,
                ch, data.shape[0],
            )

        self._report(progress, 1.0, "MossFormer2 complete")
        return self._fit_length(out, target_len)

    def _process_channel(
        self,
        pipeline,
        signal: np.ndarray,
        sample_rate: int,
        chunk: int,
        overlap: int,
        progress: ProgressFn | None,
        ch_index: int,
        ch_total: int,
    ) -> np.ndarray:
        n = signal.size
        if n <= chunk:
            return self._infer(pipeline, signal, sample_rate, n)

        step = chunk - overlap
        starts = list(range(0, n, step))
        accumulator = np.zeros(n, dtype=np.float64)
        weights = np.zeros(n, dtype=np.float64)

        for i, start in enumerate(starts):
            end = min(start + chunk, n)
            piece = signal[start:end]
            if piece.size < sample_rate // 10:
                accumulator[start:end] += piece
                weights[start:end] += 1.0
                continue

            base = (ch_index + i / max(len(starts), 1)) / ch_total
            self._report(
                progress, 0.1 + 0.85 * base,
                f"Enhancing {start / sample_rate:.0f}s to {end / sample_rate:.0f}s",
            )

            cleaned = self._infer(pipeline, piece, sample_rate, piece.size)

            # Raised-cosine crossfade at the joins, so consecutive chunks
            # sum to unity instead of leaving an audible seam.
            window = np.ones(piece.size)
            fade = min(overlap, piece.size // 2)
            if fade > 0:
                ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade)))
                if start > 0:
                    window[:fade] = ramp
                if end < n:
                    window[-fade:] = ramp[::-1]

            accumulator[start:end] += cleaned * window
            weights[start:end] += window

        weights[weights < 1e-6] = 1.0
        return (accumulator / weights).astype(np.float32)

    def _infer(self, pipeline, signal: np.ndarray, sample_rate: int, target: int) -> np.ndarray:
        """
        Run one chunk through ClearVoice.

        ClearVoice's public interface is file-in, file-out, so each chunk
        makes a round trip through a temp WAV. The write is float32 and
        the read comes straight back, so nothing is lost to the detour.
        """
        with tempfile.TemporaryDirectory(prefix="cleartake_cv_") as tmp:
            tmp_path = Path(tmp)
            in_wav = tmp_path / "chunk.wav"
            out_wav = tmp_path / "chunk_enhanced.wav"

            write_wav(in_wav, signal[None, :], sample_rate)
            result = pipeline(input_path=str(in_wav), online_write=False)

            if isinstance(result, np.ndarray):
                cleaned = np.atleast_2d(result)[0]
            else:
                pipeline.write(result, output_path=str(out_wav))
                cleaned, _ = read_wav(out_wav)
                cleaned = np.atleast_2d(cleaned)[0]

        return self._fit_length(cleaned[None, :], target)[0]
