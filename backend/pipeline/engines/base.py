"""
The denoise engine contract.

An engine takes noisy audio and returns audio with the noise removed. It
does not do blending, loudness, declicking, or file handling. The runner
owns all of that, so engines stay swappable and the pipeline behaves the
same whichever one you pick.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

ProgressFn = Callable[[float, str], None]


@dataclass
class EngineInfo:
    key: str
    name: str
    description: str
    available: bool
    install_hint: str = ""
    requires_gpu: bool = False
    native_sample_rate: int = 48_000
    strength: str = "medium"  # light | medium | heavy
    detail: str = ""


class DenoiseEngine(ABC):
    key: str = "base"
    name: str = "Base"
    description: str = ""
    install_hint: str = ""
    requires_gpu: bool = False
    native_sample_rate: int = 48_000
    strength: str = "medium"

    @classmethod
    @abstractmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return (usable_right_now, human readable detail)."""

    @classmethod
    def info(cls) -> EngineInfo:
        available, detail = cls.is_available()
        return EngineInfo(
            key=cls.key,
            name=cls.name,
            description=cls.description,
            available=available,
            install_hint=cls.install_hint,
            requires_gpu=cls.requires_gpu,
            native_sample_rate=cls.native_sample_rate,
            strength=cls.strength,
            detail=detail,
        )

    @abstractmethod
    def denoise(
        self,
        data: np.ndarray,
        sample_rate: int,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        """
        Take (channels, samples) float32 and return the same shape, denoised.

        Engines must not change the length of the signal. The runner mixes
        the result against the original sample for sample, and a length
        change there would smear the whole track.
        """

    # -- helpers shared by engines -----------------------------------------

    @staticmethod
    def _report(progress: ProgressFn | None, fraction: float, message: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, fraction)), message)

    @staticmethod
    def _fit_length(result: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or trim an engine's output back to the input length."""
        result = np.atleast_2d(result)
        current = result.shape[-1]
        if current == target_len:
            return result.astype(np.float32)
        if current > target_len:
            return result[..., :target_len].astype(np.float32)
        pad = np.zeros((result.shape[0], target_len - current), dtype=np.float32)
        return np.concatenate([result.astype(np.float32), pad], axis=-1)
