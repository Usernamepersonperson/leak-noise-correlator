"""Synthetic leak-noise recordings for one pipe segment between two sensors.

The leak is a band-limited Gaussian noise source. Each sensor receives it delayed by
distance / c and attenuated by exp(-alpha(f) d). Independent background (sensor floor,
traffic bursts, mains/pump tones) is added at each sensor. SNR is source-referred:
leak power at 1 m from the leak over total background power, so spacing and material
attenuation directly reduce the effective SNR at the sensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .physics import MATERIALS, material_wave_speed

FS = 2048.0
SRC_BAND = (10.0, 900.0)


@dataclass
class Recording:
    x1: np.ndarray
    x2: np.ndarray
    fs: float
    length_m: float
    leak_x_m: float  # true distance of leak from sensor 1
    c_true: float
    eff_snr_db: float  # measured leak/background power ratio at sensors (mean of both)


def background(rng, n: int, fs: float, sigma: float) -> np.ndarray:
    """Floor (white) + traffic bursts (<120 Hz) + mains/pump tones, total variance ~ sigma^2."""
    t = np.arange(n) / fs
    white = rng.standard_normal(n)
    # traffic: low-pass noise with a slow random envelope
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[(f < 15) | (f > 120)] = 0
    traffic = np.fft.irfft(spec, n)
    env = np.abs(fftconvolve(rng.standard_normal(n), np.ones(int(fs)) / fs, "same"))
    traffic = traffic / (traffic.std() + 1e-12) * (0.5 + 3.0 * env / (env.mean() + 1e-12) / 3.0)
    tones = sum(
        a * np.sin(2 * np.pi * h * 50.0 * t + rng.uniform(0, 2 * np.pi))
        for h, a in ((1, 1.0), (2, 0.6), (3, 0.4))
    )
    x = 0.6 * white / 1.0 + 0.5 * traffic / 1.0 + 0.25 * tones
    return sigma * x / (x.std() + 1e-12)


def _src(rng, n: int, fs: float) -> np.ndarray:
    s = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    s[(f < SRC_BAND[0]) | (f > SRC_BAND[1])] = 0
    return np.fft.irfft(s, n)


def propagate(src: np.ndarray, fs: float, delay_s: float, atten_k_db_100hz_m: float):
    """Apply delay and attenuation (sum over path of k * length, dB at 100 Hz) to src."""
    n = len(src)
    f = np.fft.rfftfreq(n, 1 / fs)
    amp = 10.0 ** (-(atten_k_db_100hz_m * f / 100.0) / 20.0)
    return np.fft.irfft(np.fft.rfft(src) * amp * np.exp(-2j * np.pi * f * delay_s), n)


def simulate_pair(
    rng,
    material: str,
    length_m: float,
    leak_x_m: float,
    snr_db: float,
    dur_s: float = 8.0,
    fs: float = FS,
    c_error_frac: float = 0.0,
    clock_offset_s: float = 0.0,
) -> Recording:
    """Two sensors at 0 and length_m, leak at leak_x_m. Returns the recording.

    c_error_frac and clock_offset_s are NOT visible to the estimator: the true wave speed
    is c*(1+c_error_frac) while the estimator gets nominal c; sensor 2's clock is
    clock_offset_s ahead (its record looks delayed by that amount).
    """
    if not 0 <= leak_x_m <= length_m:
        raise ValueError("leak must lie on the pipe segment")
    mat = MATERIALS[material]
    c_nom = material_wave_speed(material)
    c = c_nom * (1.0 + c_error_frac)
    d1, d2 = leak_x_m, length_m - leak_x_m
    t1, t2 = d1 / c, d2 / c + clock_offset_s
    n = int(dur_s * fs)
    pad = int(np.ceil((max(t1, t2) + 0.05) * fs))
    ntot = n + 2 * pad
    src = _src(rng, ntot, fs)
    src /= src.std()
    y1 = propagate(src, fs, t1, mat.atten_db_per_m_100hz * d1 + mat.atten_db_per_m_100hz * 1.0)
    y2 = propagate(src, fs, t2, mat.atten_db_per_m_100hz * d2 + mat.atten_db_per_m_100hz * 1.0)
    # note: the 1 m of path stands in for the leak-to-pipe coupling so d=0 stays finite
    # source power at 1 m reference is 1.0 (unit variance) -> remove that extra 1 m loss:
    y1 = y1 * 10 ** (mat.atten_db_per_m_100hz * 1.0 * 0 / 20)
    sigma_b = 10.0 ** (-snr_db / 20.0) * float(np.std(src))
    b1 = background(rng, ntot, fs, sigma_b)
    b2 = background(rng, ntot, fs, sigma_b)
    sl = slice(pad, pad + n)
    p_sig = 0.5 * (np.var(y1[sl]) + np.var(y2[sl]))
    eff = 10 * np.log10(p_sig / sigma_b**2) if p_sig > 0 else -np.inf
    return Recording((y1 + b1)[sl], (y2 + b2)[sl], fs, length_m, leak_x_m, c, float(eff))
