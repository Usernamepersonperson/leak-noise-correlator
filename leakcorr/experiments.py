"""Measured experiments. Every number in the README comes from these functions (seeded)."""
from __future__ import annotations

import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import lorawan
from .correlate import METHODS, estimate, gcc, pick_peak
from .network import demo_network, localise, simulate_loggers
from .physics import MATERIALS, material_wave_speed
from .screening import roc_auc, screening_scores
from .simulate import FS, simulate_pair

DUR_S = 4.0
NFFT = 2048
MATS = list(MATERIALS)


DEF_C_ERR = 0.05  # nominal wave speed wrong by up to +-5 % (uniform)
DEF_CLOCK_S = 0.5e-3  # sensor clock offset sigma
OK_M = 5.0  # a fix within this many metres counts as success


def _summ(err):
    err = np.asarray(err)
    return {
        "median_m": float(np.median(err)),
        "p90_m": float(np.percentile(err, 90)),
        "within_5m": float(np.mean(err <= OK_M)),
    }


def hash_str(s: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s))


def _trial(rng, mat, length, snr, methods, c_err=DEF_C_ERR, clock_s=DEF_CLOCK_S):
    """One recording, evaluated by every method. Returns {method: (abs_err_m, psr)}, eff_snr."""
    c = material_wave_speed(mat)
    x = rng.uniform(0.1 * length, 0.9 * length)
    rec = simulate_pair(
        rng, mat, length, x, snr, DUR_S,
        c_error_frac=rng.uniform(-c_err, c_err), clock_offset_s=rng.normal(0, clock_s),
    )
    out = {}
    for m in methods:
        xe, pk, _ = estimate(rec.x1, rec.x2, rec.fs, length, c, m, NFFT)
        out[m] = (abs(xe - x), pk.psr)
    return out, rec.eff_snr_db


def snr_sweep(trials: int, snrs, length=100.0, seed=1):
    rows = []
    for mat in MATS:
        for snr in snrs:
            rng = np.random.default_rng([seed, hash_str(mat), int(snr + 100)])
            res = [_trial(rng, mat, length, snr, METHODS) for _ in range(trials)]
            for m in METHODS:
                rows.append(
                    {"material": mat, "method": m, "snr_db": snr, "length_m": length,
                     "median_psr": float(np.median([r[0][m][1] for r in res])),
                     "eff_snr_db": float(np.median([r[1] for r in res])),
                     **_summ([r[0][m][0] for r in res])}
                )
    return rows


def spacing_sweep(trials: int, lengths, snr=20.0, method="scot", seed=2):
    rows = []
    for mat in MATS:
        for length in lengths:
            rng = np.random.default_rng([seed, hash_str(mat), int(length)])
            res = [_trial(rng, mat, length, snr, [method])[0][method][0] for _ in range(trials)]
            rows.append({"material": mat, "method": method, "snr_db": snr, "length_m": length,
                         **_summ(res)})
    return rows


def sensitivity(trials: int, snr=30.0, length=100.0, method="scot", seed=3):
    """Ideal conditions except for one imperfection at a time (cast iron)."""
    mat = "cast_iron"
    rows = []
    for ms in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0):
        rng = np.random.default_rng([seed, 1, int(ms * 10)])
        err = [_trial(rng, mat, length, snr, [method], 0.0, ms * 1e-3)[0][method][0]
               for _ in range(trials)]
        rows.append({"kind": "clock_sigma_ms", "value": ms, **_summ(err)})
    for pct in (0.0, 1.0, 2.0, 5.0, 10.0):
        rng = np.random.default_rng([seed, 2, int(pct * 10)])
        err = [_trial(rng, mat, length, snr, [method], pct / 100, 0.0)[0][method][0]
               for _ in range(trials)]
        rows.append({"kind": "c_error_pct", "value": pct, **_summ(err)})
    return rows


def network_trials(trials: int, snrs=(30.0, 50.0), method="scot", seed=4):
    net, loggers = demo_network()
    rows = []
    for snr in snrs:
        rng = np.random.default_rng([seed, int(snr)])
        errs, edge_ok, used = [], 0, []
        for _ in range(trials):
            e = int(rng.integers(len(net.pipes)))
            x = float(rng.uniform(5, net.pipes[e].length_m - 5))
            sig = simulate_loggers(rng, net, loggers, e, x, snr, DUR_S)
            fix = localise(net, sig, FS, method)
            if fix is None:
                errs.append(np.nan)
                continue
            used.append(fix.n_pairs_used)
            errs.append(net.distance(e, x, fix.edge, fix.x))
            edge_ok += fix.edge == e
        err = np.array(errs)
        ok = err[~np.isnan(err)]
        rows.append({
            "snr_db": snr, "trials": trials, "method": method,
            "median_m": float(np.median(ok)), "p90_m": float(np.percentile(ok, 90)),
            "within_10m": float(np.mean(ok <= 10.0)),
            "within_10m_all": float(np.sum(ok <= 10.0) / trials),
            "edge_correct": edge_ok / trials,
            "no_fix": int(np.isnan(err).sum()),
            "median_pairs_used": float(np.median(used)) if used else 0.0,
            "errors_m": [float(v) for v in err],
        })
    return rows, {"pipes": len(net.pipes), "loggers": loggers, "nodes": len(net.nodes)}


def screening(sites: int, levels=(-12.0, -9.0, -6.0, -3.0, 0.0, 3.0), seed=5):
    rows, roc = [], {}
    for lv in levels:
        rng = np.random.default_rng([seed, int(lv * 10) + 1000])
        s = screening_scores(rng, lv, n_sites=sites)
        for kind in ("abs", "rel"):
            fpr, tpr, auc = roc_auc(s[f"{kind}_neg"], s[f"{kind}_pos"])
            tpr5 = float(np.interp(0.05, fpr, tpr))
            rows.append({"leak_to_floor_db": lv, "score": kind, "auc": auc, "tpr_at_fpr5": tpr5})
            if lv == -6.0:
                roc[kind] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return rows, roc


def compute_cost(repeats=200, seed=6):
    rng = np.random.default_rng(seed)
    rec = simulate_pair(rng, "cast_iron", 100.0, 40.0, 20.0, DUR_S)
    c = material_wave_speed("cast_iron")
    out = {"samples_per_signal": len(rec.x1), "nfft": NFFT, "fs_hz": FS,
           "host": platform.processor() or platform.machine(), "python": platform.python_version(),
           "numpy": np.__version__, "methods": {}}
    for m in METHODS:
        estimate(rec.x1, rec.x2, rec.fs, 100.0, c, m, NFFT)  # warm-up
        t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            estimate(rec.x1, rec.x2, rec.fs, 100.0, c, m, NFFT)
            t.append(time.perf_counter() - t0)
        out["methods"][m] = {"median_ms": float(np.median(t) * 1e3)}
    segs = (len(rec.x1) - NFFT) // (NFFT // 2) + 1
    out["segments"] = segs
    out["fft_flops_estimate"] = int((2 * segs + 1) * 2.5 * NFFT * np.log2(NFFT))
    return out


def snippet_effect(trials: int, snr=20.0, length=100.0, method="scot", seed=7):
    """Localisation error with raw vs ADPCM-compressed (uplinked) snippets."""
    rows = []
    for mat in ("cast_iron", "pvc"):
        c = material_wave_speed(mat)
        rng = np.random.default_rng([seed, hash_str(mat)])
        e_raw, e_adp, ratio = [], [], []
        for _ in range(trials):
            x = rng.uniform(10, 90)
            rec = simulate_pair(rng, mat, length, x, snr, DUR_S)
            sc = 3000.0 / (rec.x1.std() + 1e-12)
            sc2 = 3000.0 / (rec.x2.std() + 1e-12)
            q1 = np.round(rec.x1 * sc)
            q2 = np.round(rec.x2 * sc2)
            f1, f2 = lorawan.encode_snippet(q1, 1), lorawan.encode_snippet(q2, 2)
            d1 = lorawan.decode_snippet(f1, len(q1))
            d2 = lorawan.decode_snippet(f2, len(q2))
            e_raw.append(abs(estimate(q1, q2, rec.fs, length, c, method, NFFT)[0] - x))
            e_adp.append(abs(estimate(d1, d2, rec.fs, length, c, method, NFFT)[0] - x))
            ratio.append(float(np.sum(q1**2) / np.sum((q1 - d1) ** 2)))
        rows.append({"material": mat, "median_err_raw_m": float(np.median(e_raw)),
                     "median_err_adpcm_m": float(np.median(e_adp)),
                     "adpcm_snr_db": float(10 * np.log10(np.median(ratio))),
                     "frames_per_snippet": len(f1)})
    plans = {f"sf{sf}": lorawan.snippet_plan(int(DUR_S * FS), sf=sf) for sf in (7, 9)}
    return rows, plans


def demo_example(seed=11):
    """The single example used for the hook figure: cast iron, 100 m, leak at 37 m."""
    rng = np.random.default_rng(seed)
    mat, length, x = "cast_iron", 100.0, 37.0
    c = material_wave_speed(mat)
    rec = simulate_pair(rng, mat, length, x, 10.0, DUR_S, c_error_frac=0.02)
    out = {"material": mat, "length_m": length, "true_x_m": x, "c": c, "eff_snr_db": rec.eff_snr_db}
    for m in ("xcorr", "phat", "scot"):
        lags, r = gcc(rec.x1, rec.x2, rec.fs, m, NFFT, 20.0, 900.0)
        pk = pick_peak(lags, r, length / c)
        xe, _, _ = estimate(rec.x1, rec.x2, rec.fs, length, c, m, NFFT)
        out[m] = {"est_x_m": xe, "tau_ms": pk.tau * 1e3, "psr": pk.psr,
                  "lags": lags.tolist(), "r": (r / np.max(np.abs(r))).tolist()}
    return out


def network_example(seed=21):
    """One network localisation, kept for the figure (not cherry-picked: first trial of the seed)."""
    net, loggers = demo_network()
    rng = np.random.default_rng(seed)
    e = int(rng.integers(len(net.pipes)))
    x = float(rng.uniform(5, net.pipes[e].length_m - 5))
    sig = simulate_loggers(rng, net, loggers, e, x, 40.0, DUR_S)
    fix = localise(net, sig, FS, "scot")
    return {"true_edge": e, "true_x": x, "est_edge": fix.edge, "est_x": fix.x,
            "error_m": net.distance(e, x, fix.edge, fix.x), "pairs_used": fix.n_pairs_used}


def run_all(quick: bool = False, workers: int | None = None) -> dict:
    """Run every experiment (sections in parallel; timing last, alone, so it is not disturbed)."""
    n = 15 if quick else 100
    snrs = [-40, -30, -20, -10, 0, 10, 20, 30, 40] if not quick else [-20, 0, 20]
    lengths = [25, 50, 100, 150, 200, 300] if not quick else [50, 150]
    t0 = time.time()
    res: dict = {"mode": "quick" if quick else "full", "trials": n, "fs_hz": FS, "dur_s": DUR_S,
                 "wave_speed_m_s": {m: material_wave_speed(m) for m in MATS}}
    jobs = {
        "snr_sweep": (snr_sweep, (n, snrs)),
        "spacing_sweep": (spacing_sweep, (n, lengths)),
        "sensitivity": (sensitivity, (n,)),
        "network": (network_trials, (n,)),
        "screening": (screening, (60 if quick else 300,)),
        "snippet": (snippet_effect, (max(n // 2, 5),)),
        "example": (demo_example, ()),
        "netex": (network_example, ()),
    }
    workers = workers or min(os.cpu_count() or 1, 6)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            futs = {k: pool.submit(f, *a) for k, (f, a) in jobs.items()}
            out = {k: f.result() for k, f in futs.items()}
    else:
        out = {k: f(*a) for k, (f, a) in jobs.items()}
    res["snr_sweep"] = out["snr_sweep"]
    res["spacing_sweep"] = out["spacing_sweep"]
    res["sensitivity"] = out["sensitivity"]
    res["network"], res["network_info"] = out["network"]
    res["screening"], res["screening_roc"] = out["screening"]
    res["snippet"], res["snippet_plans"] = out["snippet"]
    res["example"] = out["example"]
    res["network_example"] = out["netex"]
    res["compute"] = compute_cost(30 if quick else 200)
    res["runtime_s"] = time.time() - t0
    return res
