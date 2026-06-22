"""Time- and frequency-domain features for IMS vibration snapshots (Module B).

Each IMS snapshot is one second of raw acceleration.  A run is the *sequence* of
snapshots over time, so we collapse every snapshot to a single row of scalar
features; the degradation signal then lives in how those scalars evolve across
the run (e.g. kurtosis jumps weeks before the outer race fully fails).

The failing bearing (``ims.target_bearing``, B1 in Set 2) gets the full feature
set; the other bearings keep only RMS as a cheap reference channel.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import kurtosis, skew

from src.utils.paths import load_config

_EPS = 1e-12  # guard against division by zero on a flat / silent channel


def time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """Ten classic time-domain condition indicators for a 1-D signal.

    Kurtosis and crest/impulse factors are the early-warning indicators: a
    nascent point defect produces sharp periodic impacts that spike these long
    before RMS (overall energy) rises appreciably.
    """
    x = np.asarray(signal, dtype=float)
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(abs_x))
    mean_abs = float(np.mean(abs_x))
    return {
        "rms": rms,
        "peak": peak,
        "peak2peak": float(np.max(x) - np.min(x)),
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)),
        "skewness": float(skew(x, bias=False)),
        "crest_factor": peak / (rms + _EPS),
        "shape_factor": rms / (mean_abs + _EPS),
        "impulse_factor": peak / (mean_abs + _EPS),
        "std": float(np.std(x)),
        "mean": float(np.mean(x)),
    }


def band_energy(signal: np.ndarray, fs: float, center_hz: float, halfwidth_hz: float) -> float:
    """Sum of FFT magnitude-squared within ``center_hz ± halfwidth_hz``.

    Used to track energy at a bearing defect frequency (for B1 = outer race,
    that is the BPFO band).
    """
    x = np.asarray(signal, dtype=float)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    mask = (freqs >= center_hz - halfwidth_hz) & (freqs <= center_hz + halfwidth_hz)
    return float(spectrum[mask].sum())


def freq_domain_features(signal: np.ndarray, fs: float, cfg: dict | None = None) -> Dict[str, float]:
    """Total spectral energy plus energy in each bearing defect-frequency band."""
    cfg = cfg or load_config()
    ims = cfg["ims"]
    x = np.asarray(signal, dtype=float)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    feats: Dict[str, float] = {"spectral_energy": float(spectrum.sum())}
    hw = ims["band_halfwidth_hz"]
    for name, freq in ims["defect_freqs"].items():
        feats[f"band_{name}"] = band_energy(x, fs, freq, hw)
    return feats


def extract_file_features(array: np.ndarray, cfg: dict | None = None) -> Dict[str, float]:
    """Collapse one snapshot ``(n_samples, n_bearings)`` into a flat feature row.

    The target bearing gets the full time- and frequency-domain set (prefix
    ``b{N}_``); every other bearing contributes only RMS (prefix ``b{N}_rms``).
    """
    cfg = cfg or load_config()
    ims = cfg["ims"]
    fs = ims["sampling_rate_hz"]
    target = ims["target_bearing"]
    n_bearings = ims["n_bearings"]

    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < n_bearings:
        raise ValueError(
            f"預期 snapshot 形狀為 (n_samples, {n_bearings})，實際為 {arr.shape}。"
        )

    row: Dict[str, float] = {}
    for b in range(1, n_bearings + 1):
        channel = arr[:, b - 1]
        if b == target:
            for k, v in time_domain_features(channel).items():
                row[f"b{b}_{k}"] = v
            for k, v in freq_domain_features(channel, fs, cfg).items():
                row[f"b{b}_{k}"] = v
        else:
            row[f"b{b}_rms"] = float(np.sqrt(np.mean(channel**2)))
    return row
