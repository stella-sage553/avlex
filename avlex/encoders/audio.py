"""Lightweight NumPy audio encoders.

A real system would reach for Whisper/WavLM features; here we compute classic
signal-processing descriptors (log-mel energies) so the core stays dependency
free and fully reproducible.
"""

from __future__ import annotations

import numpy as np

from avlex.encoders.base import Encoder
from avlex.types import Array, Modality


def _frame_signal(wav: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice ``wav`` into overlapping frames ``(T, frame_length)`` with zero pad."""
    wav = np.asarray(wav, dtype=np.float64).ravel()
    if wav.size < frame_length:
        wav = np.pad(wav, (0, frame_length - wav.size))
    n_frames = 1 + (wav.size - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return wav[idx]


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    mel_min, mel_max = _hz_to_mel(np.array([0.0])), _hz_to_mel(np.array([sample_rate / 2.0]))
    mel_points = np.linspace(mel_min[0], mel_max[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        rising = (fft_freqs - left) / max(center - left, 1e-8)
        falling = (right - fft_freqs) / max(right - center, 1e-8)
        fb[m - 1] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


class MelStatEncoder(Encoder):
    """Per-frame log-mel energies — a compact spectral descriptor of the audio."""

    modality = Modality.AUDIO

    def __init__(
        self,
        n_mels: int = 40,
        frame_length: int = 400,
        hop_length: int = 160,
        n_fft: int = 512,
        sample_rate: int = 16000,
    ) -> None:
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self._window = np.hanning(frame_length)
        self._filterbank = _mel_filterbank(n_mels, n_fft, sample_rate)

    @property
    def output_dim(self) -> int:
        return self.n_mels

    def encode(self, raw: Array) -> np.ndarray:
        frames = _frame_signal(raw, self.frame_length, self.hop_length)
        spectrum = np.abs(np.fft.rfft(frames * self._window, n=self.n_fft, axis=1)) ** 2
        mel = spectrum @ self._filterbank.T
        return np.log1p(mel)


class EnergyEnvelopeEncoder(Encoder):
    """Per-frame loudness and zero-crossing rate.

    A cheap two-dimensional descriptor: root-mean-square energy tracks loudness,
    and the zero-crossing rate is a rough proxy for noisiness/voicing.
    """

    modality = Modality.AUDIO

    def __init__(self, frame_length: int = 400, hop_length: int = 160) -> None:
        self.frame_length = frame_length
        self.hop_length = hop_length

    @property
    def output_dim(self) -> int:
        return 2

    def encode(self, raw: Array) -> np.ndarray:
        frames = _frame_signal(raw, self.frame_length, self.hop_length)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        crossings = np.abs(np.diff(np.sign(frames), axis=1)) > 0
        zcr = crossings.mean(axis=1)
        return np.stack([rms, zcr], axis=1)
