"""
DeepFilterNet3. The default engine once the models are installed.

Full-band 48 kHz speech enhancement, around 1.1M parameters, and it runs
faster than real time even on CPU. It is the best general answer for the
noise in a screen recording: hiss, hum, fan, room tone, keyboard, traffic
through a window.

Two ways in, tried in order:

  1. The Python API (`from df.enhance import init_df, enhance`). Fastest,
     because the model loads once and stays in memory across jobs.
  2. The `deepFilter` command line tool, shelling out per file. Slower,
     but it works when the Python import is broken, which on Windows it
     sometimes is.

What it will not fix: other people talking. DeepFilterNet is trained to
keep speech and remove everything else, so a background conversation is
precisely the thing it protects. See plan.md, phase 4.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from backend.pipeline.audio import read_wav, write_wav
from backend.pipeline.engines.base import DenoiseEngine, ProgressFn


class DeepFilterNetEngine(DenoiseEngine):
    key = "deepfilternet"
    name = "DeepFilterNet 3"
    description = (
        "The all-rounder. Hiss, hum, fans, keyboard, room tone. "
        "Faithful to the original voice and fast."
    )
    install_hint = "pip install -r requirements-models.txt"
    requires_gpu = False
    native_sample_rate = 48_000
    strength = "medium"

    _model = None
    _df_state = None

    def __init__(self, model_name: str = "DeepFilterNet3", device: str = "auto"):
        self.model_name = model_name
        self.device = device

    # ----------------------------------------------------------------- #

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if find_spec("df") is not None and find_spec("torch") is not None:
            return True, "Python API available."
        if shutil.which("deepFilter"):
            return True, "Command line tool available (slower than the API)."
        return False, "Not installed. See requirements-models.txt."

    @classmethod
    def _load_model(cls, model_name: str, device: str):
        """Load once, reuse. Model init costs seconds; inference costs less."""
        if cls._model is not None:
            return cls._model, cls._df_state

        from df.enhance import init_df  # type: ignore

        kwargs = {}
        if device and device != "auto":
            kwargs["config_allow_defaults"] = True
        model, df_state, _ = init_df(model_base_dir=None, post_filter=True, **kwargs)

        if device != "cpu":
            try:
                import torch  # type: ignore

                if torch.cuda.is_available() and device in ("auto", "cuda"):
                    model = model.to("cuda")
            except Exception:
                pass  # CPU is perfectly usable here, no need to make noise

        cls._model, cls._df_state = model, df_state
        return model, df_state

    # ----------------------------------------------------------------- #

    def denoise(
        self,
        data: np.ndarray,
        sample_rate: int,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        data = np.atleast_2d(np.asarray(data, dtype=np.float32))
        target_len = data.shape[-1]

        if sample_rate != self.native_sample_rate:
            raise ValueError(
                f"DeepFilterNet3 expects {self.native_sample_rate} Hz, got {sample_rate}. "
                "The runner should have resampled before this point."
            )

        if find_spec("df") is not None and find_spec("torch") is not None:
            return self._denoise_api(data, sample_rate, target_len, progress)
        if shutil.which("deepFilter"):
            return self._denoise_cli(data, sample_rate, target_len, progress)

        raise RuntimeError(
            "DeepFilterNet is not installed. Run:\n"
            "  pip install -r requirements-models.txt"
        )

    # -- path 1: python api -------------------------------------------- #

    def _denoise_api(
        self,
        data: np.ndarray,
        sample_rate: int,
        target_len: int,
        progress: ProgressFn | None,
    ) -> np.ndarray:
        import torch  # type: ignore
        from df.enhance import enhance  # type: ignore

        self._report(progress, 0.05, "Loading DeepFilterNet 3")
        model, df_state = self._load_model(self.model_name, self.device)

        channels = data.shape[0]
        out = np.zeros_like(data)

        # Channels are enhanced separately. The model is mono, and mixing
        # to mono first would collapse any stereo image the recording has.
        for ch in range(channels):
            self._report(progress, 0.1 + 0.85 * ch / channels, f"Enhancing channel {ch + 1}")
            tensor = torch.from_numpy(data[ch]).unsqueeze(0)
            with torch.no_grad():
                enhanced = enhance(model, df_state, tensor)
            result = enhanced.squeeze(0).detach().cpu().numpy().astype(np.float32)
            out[ch] = self._fit_length(result[None, :], target_len)[0]

        self._report(progress, 1.0, "DeepFilterNet 3 complete")
        return out

    # -- path 2: command line ------------------------------------------ #

    def _denoise_cli(
        self,
        data: np.ndarray,
        sample_rate: int,
        target_len: int,
        progress: ProgressFn | None,
    ) -> np.ndarray:
        self._report(progress, 0.05, "Running DeepFilterNet (command line)")

        with tempfile.TemporaryDirectory(prefix="cleartake_dfn_") as tmp:
            tmp_path = Path(tmp)
            in_wav = tmp_path / "in.wav"
            out_dir = tmp_path / "out"
            out_dir.mkdir()

            write_wav(in_wav, data, sample_rate)

            # atten-lim is deliberately left at full strength here. The
            # runner applies the attenuation blend afterwards so every
            # engine behaves identically on that control.
            proc = subprocess.run(
                ["deepFilter", str(in_wav), "-o", str(out_dir)],
                capture_output=True, text=True, errors="replace",
            )
            if proc.returncode != 0:
                tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
                raise RuntimeError(f"deepFilter failed:\n{tail}")

            produced = sorted(out_dir.glob("*.wav"))
            if not produced:
                raise RuntimeError("deepFilter produced no output file.")

            self._report(progress, 0.9, "Reading enhanced audio")
            cleaned, _ = read_wav(produced[0])

        self._report(progress, 1.0, "DeepFilterNet complete")
        return self._fit_length(cleaned, target_len)
