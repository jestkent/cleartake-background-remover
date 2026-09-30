"""
Impulsive noise repair: mouse clicks, keyboard chatter, lip smacks, taps.

A click is a very short, very steep burst of broadband energy. It is a
different animal from hiss, and a spectral denoiser handles it badly:
across the 20 ms window the denoiser reasons about, a click looks like a
legitimate transient, so it either survives or it takes the consonant next
to it down with it.

The approach here works in the time domain instead:

  1. Whiten the signal with a short prediction-error filter, which flattens
     out the speech and leaves the sudden bursts sticking up.
  2. Flag samples whose residual exceeds a robust threshold built from the
     median absolute deviation, which a handful of loud clicks cannot skew
     the way a standard deviation would.
  3. Widen each hit into a contiguous region, then rebuild that region by
     interpolating across it from clean audio on both sides.

Regions longer than a few milliseconds are left alone. Anything that long
is a real sound, and patching over it would eat speech.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import lfilter
    from scipy.ndimage import uniform_filter1d
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


def _autocorrelation(x: np.ndarray, order: int) -> np.ndarray:
    """
    First `order + 1` autocorrelation lags, computed through the FFT.

    numpy's correlate() is a direct O(n^2) convolution, which on a twenty
    minute file at 48 kHz is the difference between a second and an hour.
    The Wiener-Khinchin route is O(n log n).
    """
    n = x.size
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(x.astype(np.float64), size)
    full = np.fft.irfft(spectrum * np.conj(spectrum), size)
    return full[: order + 1]


def _prediction_residual(x: np.ndarray, order: int = 16) -> np.ndarray:
    """
    Residual of a short linear predictor.

    Speech is highly predictable sample to sample, so it mostly cancels.
    A click is not predictable at all and survives as a spike.
    """
    n = x.size
    if n <= order * 4:
        return np.abs(np.diff(x, prepend=x[:1]))

    # Autocorrelation, then Levinson-Durbin for the predictor coefficients.
    autocorr = _autocorrelation(x, order)
    if autocorr[0] <= 0:
        return np.abs(np.diff(x, prepend=x[:1]))
    autocorr[0] *= 1.0001  # ridge term keeps the recursion stable

    a = np.zeros(order + 1)
    a[0] = 1.0
    err = autocorr[0]
    for i in range(1, order + 1):
        acc = autocorr[i] + np.dot(a[1:i], autocorr[i - 1 : 0 : -1])
        k = -acc / err if err > 0 else 0.0
        a_new = a.copy()
        a_new[1 : i + 1] += k * a[i - 1 :: -1][: i]
        a = a_new
        err *= max(1.0 - k * k, 1e-8)

    if _HAVE_SCIPY:
        residual = lfilter(a, [1.0], x)
    else:
        residual = np.convolve(x, a)[: x.size]
    return np.abs(residual)


def _regions_from_mask(mask: np.ndarray, pad: int, max_len: int) -> list[tuple[int, int]]:
    """Turn a boolean mask into padded (start, end) spans, dropping long ones."""
    if not mask.any():
        return []

    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    bounds = np.concatenate(([0] if mask[0] else [], edges + 1, [mask.size] if mask[-1] else []))
    spans = bounds.reshape(-1, 2) if bounds.size % 2 == 0 else bounds[:-1].reshape(-1, 2)

    regions: list[tuple[int, int]] = []
    for start, end in spans:
        start = max(0, int(start) - pad)
        end = min(mask.size, int(end) + pad)
        if end - start <= max_len:
            regions.append((start, end))

    # Merge spans that now overlap after padding.
    merged: list[tuple[int, int]] = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            if end - prev_start <= max_len * 2:
                merged[-1] = (prev_start, max(prev_end, end))
                continue
        merged.append((start, end))
    return merged


def _estimate_period(context: np.ndarray, sample_rate: int) -> int | None:
    """
    Find the local pitch period, in samples, or None if unvoiced.

    Searched over 60 Hz to 500 Hz, which covers every speaking voice with
    room to spare at both ends.
    """
    if context.size < 256:
        return None

    lo = max(2, int(sample_rate / 500))
    hi = min(context.size - 1, int(sample_rate / 60))
    if hi <= lo:
        return None

    signal = context - context.mean()
    energy = float(np.dot(signal, signal))
    if energy < 1e-12:
        return None

    size = 1 << int(np.ceil(np.log2(2 * signal.size)))
    spectrum = np.fft.rfft(signal, size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), size)[: hi + 1]

    peak = int(np.argmax(autocorr[lo : hi + 1])) + lo
    # Require a genuine periodic peak. Unvoiced sounds have none, and
    # forcing a period onto them produces a buzzing artefact.
    if autocorr[peak] < 0.35 * autocorr[0]:
        return None
    return peak


def _repair_region(
    x: np.ndarray, start: int, end: int, context: int, sample_rate: int
) -> None:
    """
    Rebuild x[start:end] in place.

    The naive repair, crossfading between the average level either side,
    is wrong in a way that is easy to miss. Clicks usually land *on* the
    speech, not in the gaps between words, so bridging across the damaged
    region punches a hole in whatever word was being said. It measures as
    a successful click removal and sounds like a dropout.

    Speech is quasi-periodic during voiced sounds, so the fix is to take
    the waveform from one pitch period earlier and splice it in. The
    period repeats anyway, which is why this is inaudible, and it restores
    the speech rather than deleting it.

    Unvoiced sounds have no period to borrow. Those fall back to mirroring
    the neighbouring samples, which keeps the right spectral character
    without inventing a pitch that was never there.
    """
    gap = end - start
    if gap <= 0:
        return

    # Use several periods of history so the pitch estimate is stable.
    history_start = max(0, start - max(context, sample_rate // 20))
    history = x[history_start:start]
    future = x[end : min(x.size, end + context)]

    if history.size == 0 and future.size == 0:
        x[start:end] = 0.0
        return

    period = _estimate_period(history, sample_rate) if history.size else None
    patch: np.ndarray | None = None

    if period and history.size >= period + gap:
        # One pitch period back, which is the same part of the same cycle.
        source_start = history.size - period
        candidate = history[source_start : source_start + gap]
        if candidate.size == gap:
            patch = candidate.copy()

    if patch is None and history.size >= gap:
        # Unvoiced or too little history: mirror the preceding samples.
        # Reversing avoids a discontinuity at the splice point.
        patch = history[-gap:][::-1].copy()

    if patch is None and future.size >= gap:
        patch = future[:gap][::-1].copy()

    if patch is None:
        patch = np.zeros(gap, dtype=x.dtype)

    # Match the patch to the level immediately around the gap.
    #
    # The lower bound is deliberately near zero rather than a fraction:
    # when a click lands in a pause between words there is nothing to
    # restore, and the correct repair is near-silence. Clamping the scale
    # higher would splice audible speech into a gap that was quiet.
    edge = min(gap, 32)
    local = np.concatenate([
        history[-edge:] if history.size >= edge else history,
        future[:edge] if future.size >= edge else future,
    ])
    if local.size and patch.size:
        local_rms = float(np.sqrt(np.mean(np.square(local, dtype=np.float64))))
        patch_rms = float(np.sqrt(np.mean(np.square(patch, dtype=np.float64))))
        if patch_rms > 1e-9 and local_rms > 1e-9:
            patch *= np.clip(local_rms / patch_rms, 0.02, 4.0)

    # Crossfade the seams so neither edge clicks in its own right.
    fade = min(gap // 4, max(2, int(sample_rate * 0.0002)))
    if fade > 1 and history.size >= fade and future.size >= fade:
        ramp = np.linspace(0.0, 1.0, fade)
        patch[:fade] = history[-fade:] * (1.0 - ramp) + patch[:fade] * ramp
        patch[-fade:] = patch[-fade:] * (1.0 - ramp) + future[:fade] * ramp

    x[start:end] = patch.astype(x.dtype)


def declick(
    data: np.ndarray,
    sample_rate: int,
    sensitivity: float = 1.0,
    max_click_ms: float = 3.0,
) -> tuple[np.ndarray, int]:
    """
    Find and repair impulsive clicks.

    sensitivity scales the detection threshold. 1.0 is a conservative
    default that catches obvious mouse clicks without touching plosives.
    Above about 2.0 it starts nibbling at hard consonants, so the UI caps
    it there.

    Returns the repaired audio and the number of clicks patched.
    """
    data = np.atleast_2d(np.asarray(data, dtype=np.float32)).copy()
    if sensitivity <= 0:
        return data, 0

    max_len = max(4, int(sample_rate * max_click_ms / 1000.0))
    pad = max(2, int(sample_rate * 0.0004))      # 0.4 ms either side
    context = max(32, int(sample_rate * 0.004))  # 4 ms of context to rebuild from
    total = 0

    for ch in range(data.shape[0]):
        x = data[ch]
        if x.size < sample_rate // 50:
            continue

        residual = _prediction_residual(x)

        # Robust threshold: median plus a multiple of the median absolute
        # deviation. Immune to being dragged upward by the very clicks it
        # is trying to find.
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med))) or 1e-9
        threshold = med + (14.0 / max(sensitivity, 0.05)) * mad

        mask = residual > threshold
        if not mask.any():
            continue

        # A real click is also a local sharpness spike, not just a loud
        # patch. Requiring both cuts false positives on sustained speech.
        # A running mean stands in for a median filter here: it is O(n)
        # rather than O(n*k), and the local baseline it produces is close
        # enough, because the whole point is to compare a sample against
        # its own neighbourhood rather than against the file.
        if _HAVE_SCIPY:
            span = max(3, int(sample_rate * 0.004))
            baseline = uniform_filter1d(residual, size=span, mode="nearest")
            mask &= residual > (baseline * 4.0 + med)

        for start, end in _regions_from_mask(mask, pad, max_len):
            _repair_region(x, start, end, context, sample_rate)
            total += 1

    return data, total
