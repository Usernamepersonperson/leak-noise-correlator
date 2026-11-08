"""Minimum-night-noise screening, as done by noise loggers.

A logger records short windows through the night (02:00-04:00, when demand and traffic are
lowest) and reports the band-level of each. A leak adds a steady noise floor, so the
MINIMUM level over the night rises while sporadic traffic/pump events (which only raise
some windows) are rejected by taking the minimum.

Score variants:
  abs  minimum night level
  rel  minimum night level minus the site's own median over a learning period
"""
from __future__ import annotations

import numpy as np

from .simulate import FS

BAND = (100.0, 800.0)


def _band_mask(n: int, fs: float):
    f = np.fft.rfftfreq(n, 1 / fs)
    return (f >= BAND[0]) & (f <= BAND[1]), f


def window_levels_db(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """In-band level (dB re unit variance) of each row of x (n_windows, n_samples)."""
    x = np.atleast_2d(x)
    mask, _ = _band_mask(x.shape[1], fs)
    spec = np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)
    p = (np.abs(spec[:, mask]) ** 2).sum(axis=1) * 2 / x.shape[1] ** 2
    return 10 * np.log10(p + 1e-30)


def night_windows(
    rng,
    floor_sigma: float,
    leak_to_floor_db: float | None,
    n_win: int = 12,
    win_s: float = 1.0,
    fs: float = FS,
    event_prob: float = 0.2,
    hum_scale: float = 0.3,
    night_sigma_db: float = 1.0,
    persistent_prob: float = 0.08,
) -> np.ndarray:
    """One night of windows: white sensor floor + mains hum + occasional traffic/pump events
    (+ steady leak noise at `leak_to_floor_db` above the in-band floor, if not None)."""
    n = int(win_s * fs)
    mask, f = _band_mask(n, fs)
    t = np.arange(n) / fs
    # night-to-night drift of the whole noise floor (temperature, demand, ground conditions)
    x = floor_sigma * 10 ** (rng.normal(0, night_sigma_db) / 20.0) * rng.standard_normal((n_win, n))
    hum_amp = floor_sigma * hum_scale * rng.lognormal(0, 0.5)
    for h in (1, 2, 3):
        x += hum_amp / h * np.sin(2 * np.pi * 50.0 * h * t[None, :] + rng.uniform(0, 6.28, (n_win, 1)))
    # events: broadband-ish bursts in some windows only
    ev = rng.random(n_win) < event_prob
    if ev.any():
        amp = floor_sigma * rng.lognormal(1.0, 0.6, ev.sum())
        x[ev] += amp[:, None] * rng.standard_normal((ev.sum(), n))
    p_floor = floor_sigma**2 * mask.sum() * 2 / n  # in-band variance of the white floor
    if rng.random() < persistent_prob:
        # a disturbance present ALL night (e.g. a pump cycling): a genuine false-alarm source
        raw = np.fft.rfft(rng.standard_normal(n))
        raw[~mask] = 0
        pers = np.fft.irfft(raw, n)
        pers = pers / pers.std() * np.sqrt(p_floor * 10 ** (rng.uniform(-8.0, 0.0) / 10.0))
        x += pers[None, :]
    if leak_to_floor_db is not None:
        raw = np.fft.rfft(rng.standard_normal((n_win, n)), axis=1)
        raw[:, ~mask] = 0
        leak = np.fft.irfft(raw, n, axis=1)
        leak = leak / leak.std(axis=1, keepdims=True)
        x += leak * np.sqrt(p_floor * 10 ** (leak_to_floor_db / 10.0))
    return x


def roc_auc(scores_neg: np.ndarray, scores_pos: np.ndarray):
    """ROC points (fpr, tpr) and AUC (rank statistic) for 'higher score = leak'."""
    neg = np.asarray(scores_neg, float)
    pos = np.asarray(scores_pos, float)
    if len(neg) == 0 or len(pos) == 0:
        raise ValueError("need both classes")
    thr = np.sort(np.unique(np.concatenate([neg, pos])))[::-1]
    thr = np.concatenate([[np.inf], thr, [-np.inf]])
    tpr = np.array([(pos >= t).mean() for t in thr])
    fpr = np.array([(neg >= t).mean() for t in thr])
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def screening_scores(
    rng,
    leak_to_floor_db: float,
    n_sites: int = 300,
    learn_nights: int = 7,
    site_sigma_db: float = 3.0,
):
    """Simulate n_sites healthy sites and n_sites with a new leak on the test night.

    Returns dict of score arrays: abs_neg, abs_pos, rel_neg, rel_pos.
    Each site has its own sensor floor (lognormal, site_sigma_db), so an absolute
    threshold cannot be shared across sites but a per-site baseline can.
    """
    out = {"abs_neg": [], "abs_pos": [], "rel_neg": [], "rel_pos": []}
    for cls in ("neg", "pos"):
        for _ in range(n_sites):
            sigma = 10 ** (rng.normal(0, site_sigma_db) / 20.0)
            learn = [
                window_levels_db(night_windows(rng, sigma, None)).min()
                for _ in range(learn_nights)
            ]
            test = window_levels_db(
                night_windows(rng, sigma, leak_to_floor_db if cls == "pos" else None)
            ).min()
            out[f"abs_{cls}"].append(test)
            out[f"rel_{cls}"].append(test - float(np.median(learn)))
    return {k: np.array(v) for k, v in out.items()}
