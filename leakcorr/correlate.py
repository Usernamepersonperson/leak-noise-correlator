"""Cross-correlation and generalised cross-correlation (GCC) time-delay estimation.

Convention: x1(t) = s(t - t1), x2(t) = s(t - t2); the estimated lag is tau = t1 - t2.
Sensor 1 sits at position 0, sensor 2 at position L, the leak at x from sensor 1:
t1 - t2 = (x - (L - x)) / c, hence x = (L + c * tau) / 2.

Weightings (applied to the Welch-averaged cross-spectrum Gxy):
  xcorr  W = 1                         plain cross-correlation
  phat   W = 1 / |Gxy|                 phase transform (Knapp & Carter 1976)
  scot   W = 1 / sqrt(Gxx * Gyy)       smoothed coherence transform (Carter et al. 1973)
  roth   W = 1 / Gxx                   Roth processor (Roth 1971)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHODS = ("xcorr", "phat", "scot", "roth")


def welch_spectra(x1, x2, nfft: int = 4096, overlap: float = 0.5):
    """Segment-averaged auto and cross spectra (Hann window)."""
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    if x1.shape != x2.shape or x1.ndim != 1:
        raise ValueError("x1 and x2 must be 1-D arrays of equal length")
    if len(x1) < nfft:
        nfft = int(2 ** np.floor(np.log2(len(x1))))
    if nfft < 16:
        raise ValueError("signals too short")
    hop = max(1, int(nfft * (1 - overlap)))
    win = np.hanning(nfft)
    starts = range(0, len(x1) - nfft + 1, hop)
    gxx = np.zeros(nfft // 2 + 1)
    gyy = np.zeros(nfft // 2 + 1)
    gxy = np.zeros(nfft // 2 + 1, complex)
    n = 0
    for s in starts:
        a = np.fft.rfft((x1[s : s + nfft] - x1[s : s + nfft].mean()) * win)
        b = np.fft.rfft((x2[s : s + nfft] - x2[s : s + nfft].mean()) * win)
        gxx += np.abs(a) ** 2
        gyy += np.abs(b) ** 2
        gxy += a * np.conj(b)
        n += 1
    return gxx / n, gyy / n, gxy / n, nfft


def gcc(
    x1,
    x2,
    fs: float,
    method: str = "phat",
    nfft: int = 4096,
    fmin: float = 20.0,
    fmax: float | None = None,
    eps: float = 1e-12,
):
    """Return (lags_s, r) of the weighted cross-correlation, zero lag at the centre."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    gxx, gyy, gxy, nfft = welch_spectra(x1, x2, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    if fmax is None:
        fmax = 0.45 * fs
    if method == "xcorr":
        w = np.ones_like(gxx)
    elif method == "phat":
        w = 1.0 / (np.abs(gxy) + eps * (gxx.max() + 1e-30))
    elif method == "scot":
        w = 1.0 / (np.sqrt(gxx * gyy) + eps * (gxx.max() + 1e-30))
    else:
        w = 1.0 / (gxx + eps * (gxx.max() + 1e-30))
    mask = (f >= fmin) & (f <= fmax)
    r = np.fft.irfft(gxy * w * mask, nfft)
    r = np.fft.fftshift(r)
    lags = (np.arange(nfft) - nfft // 2) / fs
    return lags, r


@dataclass
class Peak:
    tau: float  # estimated lag, s
    height: float
    psr: float  # peak-to-sidelobe ratio
    confidence: float  # 1 - 1/psr, in [0, 1)


def pick_peak(lags, r, max_lag: float, guard_s: float = 0.005) -> Peak:
    """Pick the highest peak inside +-max_lag (physical window) with parabolic refinement.

    psr = peak / largest |r| anywhere outside +-guard_s of the peak.
    """
    lags = np.asarray(lags)
    r = np.asarray(r)
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    dt = lags[1] - lags[0]
    inside = np.where(np.abs(lags) <= max_lag + 1e-12)[0]
    if len(inside) == 0:
        raise ValueError("max_lag window contains no samples")
    k = inside[np.argmax(r[inside])]
    height = float(r[k])
    tau = float(lags[k])
    if 0 < k < len(r) - 1:
        a, b, c = r[k - 1], r[k], r[k + 1]
        den = a - 2 * b + c
        if den < 0:
            tau += float(0.5 * (a - c) / den * dt)
    far = np.abs(lags - lags[k]) > guard_s
    side = float(np.max(np.abs(r[far]))) if far.any() else 0.0
    psr = height / side if side > 0 else float("inf")
    if height <= 0:
        psr = 0.0
    conf = 0.0 if psr <= 1 else (1.0 if np.isinf(psr) else 1.0 - 1.0 / psr)
    return Peak(tau, height, float(psr), float(conf))


def locate(tau: float, length_m: float, c: float) -> float:
    """Leak distance from sensor 1 along a pipe of given length; clipped to the pipe."""
    x = 0.5 * (length_m + c * tau)
    return float(min(max(x, 0.0), length_m))


def estimate(x1, x2, fs, length_m, c, method="phat", nfft=4096, fmin=20.0, fmax=None):
    """Full two-sensor chain: correlate, pick peak in the physical window, locate."""
    lags, r = gcc(x1, x2, fs, method, nfft, fmin, fmax)
    pk = pick_peak(lags, r, length_m / c)
    return locate(pk.tau, length_m, c), pk, (lags, r)
