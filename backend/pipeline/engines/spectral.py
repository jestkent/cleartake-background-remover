"""
Spectral gate. Runs on numpy and scipy alone, with no model download.

This exists so the pipeline is testable the moment you clone the repo,
before you commit to a multi-gigabyte torch install. It is also genuinely
the right tool for one specific job: a steady noise floor. Fan hum, air
conditioning, preamp hiss, laptop whine. Those sit still in the spectrum,
which makes them easy to measure and subtract.

How it works:

  1. STFT the signal.
  2. Build a noise profile from the quietest frames in each frequency bin,
     read at a low percentile. Speech is intermittent, so the quiet tail of
     every bin is the noise sitting underneath it. No "select a silent
     region" step needed.
  3. Build a soft Wiener-style gain mask per bin, smooth it across time and
     frequency, and apply it.

The smoothing is what separates this from a naive noise gate. Gating bins
independently is what produces the warbling, underwater, bubbling artefact
people call musical noise, because isolated bins flicker open and shut
between frames. Smoothing the mask correlates neighbours so they move
together.

Where it falls down: anything non-stationary. Traffic, a door, a chair
scrape, and above all other people talking. For those, use DeepFilterNet3
or ClearerVoice.
"""

from __future__ import annotations

import numpy as np

from backend.pipeline.engines.base import DenoiseEngine, ProgressFn

try:
    from scipy.signal import stft, istft
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


class SpectralGateEngine(DenoiseEngine):
    key = "spectral"
    name = "Spectral gate"
    description = (
        "Steady noise only: fan hum, air conditioning, preamp hiss. "
        "Runs instantly with no model download."
    )
    install_hint = "pip install scipy"
    requires_gpu = False
    strength = "light"

    def __init__(
        self,
        noise_percentile: float = 12.0,
        reduction_db: float = 18.0,
        smooth_freq_bins: int = 3,
        smooth_time_frames: int = 4,
        floor_db: float = -38.0,
    ):
        self.noise_percentile = noise_percentile
        self.reduction_db = reduction_db
        self.smooth_freq_bins = smooth_freq_bins
        self.smooth_time_frames = smooth_time_frames
        self.floor_db = floor_db

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if _HAVE_SCIPY:
            return True, "Ready. No download needed."
        return False, "scipy is not installed."

    # ----------------------------------------------------------------- #

    def denoise(
        self,
        data: np.ndarray,
        sample_rate: int,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        if not _HAVE_SCIPY:
            raise RuntimeError("scipy is required for the spectral engine.")

        data = np.atleast_2d(np.asarray(data, dtype=np.float32))
        target_len = data.shape[-1]
        channels = data.shape[0]
        out = np.zeros_like(data)

        # ~42 ms window at 48 kHz. Long enough to resolve the noise floor,
        # short enough that speech onsets are not smeared across frames.
        nperseg = 1 << int(np.round(np.log2(sample_rate * 0.042)))
        nperseg = int(max(256, min(nperseg, 4096)))
        noverlap = nperseg * 3 // 4

        for ch in range(channels):
            self._report(progress, ch / channels, f"Analysing channel {ch + 1}")

            freqs, times, spec = stft(
                data[ch], fs=sample_rate, nperseg=nperseg,
                noverlap=noverlap, window="hann", boundary="zeros", padded=True,
            )
            power = np.abs(spec) ** 2

            if power.shape[1] < 4:
                out[ch] = data[ch]
                continue

            noise_power = self._estimate_noise(power)
            gain = self._decision_directed_gain(power, noise_power)
            gain = self._smooth(gain)

            self._report(progress, (ch + 0.6) / channels, f"Filtering channel {ch + 1}")

            _, cleaned = istft(
                spec * gain, fs=sample_rate, nperseg=nperseg,
                noverlap=noverlap, window="hann", boundary=True,
            )
            out[ch] = self._fit_length(cleaned[None, :], target_len)[0]

        self._report(progress, 1.0, "Spectral gate complete")
        return out

    def _estimate_noise(self, power: np.ndarray) -> np.ndarray:
        """
        Estimate the mean noise power per frequency bin.

        The obvious approach, taking a low percentile of each bin over
        time, is wrong, and wrong in a way that quietly cripples the
        reduction. Noise power in a bin is roughly exponentially
        distributed, so its 10th percentile sits around an eighth of its
        mean. Use that as the noise estimate and every noise frame looks
        like it has 8:1 signal to noise, the gate concludes there is
        speech present, and it barely attenuates anything.

        What is wanted is the *mean* power during the moments when only
        noise is present. So: rank whole frames by total energy, take the
        quiet ones, and average across those. Ranking by whole frames
        rather than per bin matters, because speech occupies one span of
        time across all frequencies at once.
        """
        frame_energy = power.sum(axis=0)
        cutoff = np.percentile(frame_energy, max(self.noise_percentile, 5.0))

        quiet = power[:, frame_energy <= cutoff]
        if quiet.shape[1] < 3:
            # Wall-to-wall speech with no gaps. Fall back to a per-bin
            # low percentile, scaled up to approximate the mean of an
            # exponential distribution truncated at that point.
            quiet_level = np.percentile(power, 20.0, axis=1, keepdims=True)
            return np.maximum(quiet_level * 4.0, 1e-14)

        noise_power = quiet.mean(axis=1, keepdims=True)
        return np.maximum(noise_power, 1e-14)

    def _decision_directed_gain(
        self, power: np.ndarray, noise_power: np.ndarray
    ) -> np.ndarray:
        """
        Wiener gain driven by a decision-directed SNR estimate.

        The naive move is to compute the gain from the current frame's
        measured SNR alone. That estimate is noisy frame to frame, and the
        resulting gain flickers, which is heard as the warbling musical
        noise that makes cheap noise reduction obvious.

        The decision-directed estimator (Ephraim and Malah, 1984) fixes it
        by building the SNR for each frame mostly from the *previous*
        frame's cleaned output, with only a small contribution from the
        current measurement. The estimate becomes smooth over time, so the
        gain stops flickering, and the artefact largely disappears.

        `alpha` over-subtracts: treating the noise as somewhat louder than
        measured buys real reduction at the cost of a little speech
        distortion. The runner blends the result back toward the original
        afterwards, so being slightly aggressive here is safe.
        """
        beta = 0.96                                       # how much history to keep
        alpha = 1.0 + (self.reduction_db / 12.0)          # over-subtraction factor
        floor = 10.0 ** (self.floor_db / 20.0)

        snr_post = power / noise_power
        gain = np.empty_like(power)

        # Seed the recursion from the first frame's own measurement.
        snr_prior = np.maximum(snr_post[:, 0] - alpha, 1e-6)

        for frame in range(power.shape[1]):
            current = np.maximum(snr_post[:, frame] - alpha, 0.0)
            snr_prior = beta * snr_prior + (1.0 - beta) * current

            g = snr_prior / (1.0 + snr_prior)             # Wiener gain
            g = np.clip(g, floor, 1.0)
            gain[:, frame] = g

            # Feed this frame's cleaned power estimate into the next one.
            snr_prior = np.maximum((g ** 2) * snr_post[:, frame], 1e-6)

        return gain

    def _smooth(self, gain: np.ndarray) -> np.ndarray:
        """
        Blur the gain mask over frequency and time.

        This is the anti-musical-noise step. Isolated bins flickering
        between frames is exactly what produces the bubbling artefact, so
        neighbours are made to move together.
        """
        smoothed = gain

        f_span = max(1, int(self.smooth_freq_bins))
        if f_span > 1:
            kernel = np.hanning(f_span + 2)[1:-1]
            kernel /= kernel.sum()
            smoothed = np.apply_along_axis(
                lambda col: np.convolve(col, kernel, mode="same"), 0, smoothed
            )

        t_span = max(1, int(self.smooth_time_frames))
        if t_span > 1:
            kernel = np.hanning(t_span + 2)[1:-1]
            kernel /= kernel.sum()
            smoothed = np.apply_along_axis(
                lambda row: np.convolve(row, kernel, mode="same"), 1, smoothed
            )

        return np.clip(smoothed, 0.0, 1.0)
