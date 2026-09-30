"""
Engine registry.

The UI asks this module what is installed and greys out the rest, so you
never pick an engine that then fails halfway through a twenty minute file.
"""

from __future__ import annotations

from backend.pipeline.engines.base import DenoiseEngine, EngineInfo
from backend.pipeline.engines.clearervoice import ClearerVoiceEngine
from backend.pipeline.engines.deepfilternet import DeepFilterNetEngine
from backend.pipeline.engines.spectral import SpectralGateEngine

# Ordered best-first. `auto` walks this list and takes the first one that
# is actually installed, so a fresh clone runs on the spectral gate and
# silently upgrades itself to DeepFilterNet the moment you install it.
REGISTRY: dict[str, type[DenoiseEngine]] = {
    DeepFilterNetEngine.key: DeepFilterNetEngine,
    ClearerVoiceEngine.key: ClearerVoiceEngine,
    SpectralGateEngine.key: SpectralGateEngine,
}

AUTO_PREFERENCE = [
    DeepFilterNetEngine.key,
    ClearerVoiceEngine.key,
    SpectralGateEngine.key,
]


def list_engines() -> list[EngineInfo]:
    return [cls.info() for cls in REGISTRY.values()]


def resolve_engine_key(key: str) -> str:
    """Turn 'auto' into a concrete key, and validate anything else."""
    if key and key != "auto":
        if key not in REGISTRY:
            raise ValueError(
                f"Unknown engine '{key}'. Available: {', '.join(REGISTRY)}"
            )
        return key

    for candidate in AUTO_PREFERENCE:
        available, _ = REGISTRY[candidate].is_available()
        if available:
            return candidate

    raise RuntimeError(
        "No denoise engine is usable. At minimum install scipy:\n"
        "  pip install scipy"
    )


def build_engine(key: str, **kwargs) -> DenoiseEngine:
    resolved = resolve_engine_key(key)
    cls = REGISTRY[resolved]

    available, detail = cls.is_available()
    if not available:
        raise RuntimeError(f"{cls.name} is not usable: {detail}\n{cls.install_hint}")

    # Pass only the keyword arguments this engine actually accepts.
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


__all__ = [
    "REGISTRY",
    "DenoiseEngine",
    "EngineInfo",
    "build_engine",
    "list_engines",
    "resolve_engine_key",
]
