"""Figures: docs/figure.png (1600x900), docs/post-figure.png (960x700), docs/linkedin.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .network import demo_network  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = (
    "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA",
)
MAT_COL = {"steel": GREY, "cast_iron": ACCENT, "pvc": AMBER, "pe": RED}
MAT_LBL = {"steel": "steel", "cast_iron": "cast iron", "pvc": "PVC", "pe": "PE"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.8,
})


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _series(rows, mat, method, key, xkey, **flt):
    r = [x for x in rows if x["material"] == mat and x.get("method") == method
         and all(x.get(k) == v for k, v in flt.items())]
    r.sort(key=lambda x: x[xkey])
    return [x[xkey] for x in r], [x[key] for x in r]


def _floor(v):
    return np.maximum(np.asarray(v, float), 0.01)


def hook_panels(ax_pipe, ax_corr, ex, method="scot", big=False):
    """Pipe schematic with true vs estimated leak, and the correlation peak."""
    fs = 13 if big else 9
    L, xt = ex["length_m"], ex["true_x_m"]
    d = ex[method]
    xe = d["est_x_m"]
    ax_pipe.set_xlim(-6, L + 6)
    ax_pipe.set_ylim(-1.3, 1.9)
    ax_pipe.axis("off")
    ax_pipe.plot([0, L], [0, 0], color=GREY, lw=9 if big else 6, solid_capstyle="round", zorder=1)
    ax_pipe.plot([0, L], [0, 0], color=BG, lw=3.5 if big else 2.5, zorder=2)
    for x, lbl in ((0, "A"), (L, "B")):
        ax_pipe.scatter([x], [0], s=420 if big else 160, color=ACCENT, zorder=4, marker="s")
        ax_pipe.text(x, -0.95, f"sensor {lbl}", color=MUTED, ha="center", va="center", fontsize=fs)
    ax_pipe.scatter([xt], [0.32], s=520 if big else 200, color=RED, zorder=5, marker="v")
    ax_pipe.text(xt, 1.2, f"true leak {xt:.1f} m", color=RED, ha="center", fontsize=fs + 1,
                 fontweight="bold")
    ax_pipe.scatter([xe], [-0.32], s=520 if big else 200, color=GREEN, zorder=6, marker="^")
    ax_pipe.text(xe, -0.95, f"estimated {xe:.1f} m",
                 color=GREEN, ha="center", va="center", fontsize=fs + 1, fontweight="bold",
                 bbox={"fc": BG, "ec": "none", "pad": 1})
    lags = np.array(d["lags"]) * 1e3
    r = np.array(d["r"])
    win = ex["length_m"] / ex["c"] * 1e3
    m = np.abs(lags) <= win * 1.05
    ax_corr.plot(lags[m], r[m], color=ACCENT, lw=2.2 if big else 1.4)
    ax_corr.fill_between(lags[m], 0, r[m], color=ACCENT, alpha=0.25)
    ax_corr.axvline(d["tau_ms"], color=GREEN, lw=2, ls="--")
    true_tau = (2 * xt - L) / ex["c"] * 1e3
    ax_corr.axvline(true_tau, color=RED, lw=2, ls=":")
    ax_corr.set_xlabel("lag between the two sensors (ms)", fontsize=fs + 1)
    ax_corr.set_ylabel(f"correlation ({method.upper()})", fontsize=fs + 1)
    ax_corr.tick_params(labelsize=fs)
    ax_corr.text(d["tau_ms"], 0.97, f"  peak at {d['tau_ms']:.1f} ms", color=GREEN,
                 fontsize=fs + 1, va="top", ha="left" if d["tau_ms"] < 0.4 * win else "right",
                 transform=ax_corr.get_xaxis_transform())
    _style(ax_corr)


def post_figure(res, path):
    ex = res["example"]
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    fig.text(0.05, 0.945, "Two sensors, one correlation peak, one leak position",
             fontsize=14, fontweight="bold", color=TEXT, va="center")
    fig.text(0.05, 0.9, "Simulated cast-iron pipe, 100 m, wave-speed error 2% (open-source code)",
             fontsize=12, color=MUTED, va="center")
    ax_p = fig.add_axes([0.05, 0.60, 0.90, 0.24])
    ax_c = fig.add_axes([0.10, 0.12, 0.85, 0.40])
    hook_panels(ax_p, ax_c, ex, "scot", big=True)
    err = abs(ex["scot"]["est_x_m"] - ex["true_x_m"])
    fig.text(0.95, 0.9, f"error {err:.2f} m", color=GREEN, fontsize=14, fontweight="bold",
             ha="right", va="center")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def main_figure(res, path):
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    fig.text(0.02, 0.965, "leak-noise-correlator: acoustic leak localisation, simulated",
             fontsize=17, fontweight="bold", va="center")
    fig.text(0.02, 0.93, f"All data simulated ({res['trials']} trials per point, "
             f"{res['dur_s']:.0f} s records at {res['fs_hz']:.0f} Hz); attenuation values are "
             "illustrative assumptions", fontsize=10.5, color=MUTED, va="center")
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, bottom=0.075, top=0.88,
                          wspace=0.27, hspace=0.42)
    # (a) hook
    sub = gs[0, 0].subgridspec(2, 1, height_ratios=[1, 2.2], hspace=0.15)
    hook_panels(fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), res["example"], "scot")
    fig.text(0.0585, 0.895, "(a) pipe and correlation peak", color=TEXT, fontsize=11)
    # (b) error vs SNR
    ax = fig.add_subplot(gs[0, 1])
    for mat in MAT_COL:
        x, y = _series(res["snr_sweep"], mat, "scot", "median_m", "snr_db")
        ax.plot(x, _floor(y), "-o", color=MAT_COL[mat], label=MAT_LBL[mat], lw=2, ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("source SNR (dB)")
    ax.set_ylabel("median error (m), 100 m span")
    ax.set_title("(b) error vs SNR, by material (SCOT)", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    _style(ax)
    # (c) error vs spacing
    ax = fig.add_subplot(gs[0, 2])
    for mat in MAT_COL:
        x, y = _series(res["spacing_sweep"], mat, "scot", "median_m", "length_m")
        ax.plot(x, _floor(y), "-o", color=MAT_COL[mat], label=MAT_LBL[mat], lw=2, ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("sensor spacing (m)")
    ax.set_ylabel("median error (m), SNR 20 dB")
    ax.set_title("(c) error vs sensor spacing", color=TEXT, fontsize=11, loc="left")
    _style(ax)
    # (d) network
    ax = fig.add_subplot(gs[1, 0])
    net, loggers = demo_network()
    ne = res["network_example"]
    for p in net.pipes:
        (x0, y0), (x1, y1) = net.coords[p.u], net.coords[p.v]
        ax.plot([x0, x1], [y0, y1], color=MAT_COL[p.material], lw=3, solid_capstyle="round")

    def at(e, x):
        p = net.pipes[e]
        (x0, y0), (x1, y1) = net.coords[p.u], net.coords[p.v]
        t = x / p.length_m
        return x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    for n, (x, y) in net.coords.items():
        ax.scatter([x], [y], s=150 if n in loggers else 30, marker="s" if n in loggers else "o",
                   color=TEXT if n in loggers else GREY, zorder=3)
    tx, ty = at(ne["true_edge"], ne["true_x"])
    ex_, ey = at(ne["est_edge"], ne["est_x"])
    ax.scatter([tx], [ty], s=190, color=RED, marker="v", zorder=5, label="true leak")
    ax.scatter([ex_], [ey], s=190, color=GREEN, marker="^", zorder=6, label="estimate")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"(d) network, 5 loggers (squares); example error {ne['error_m']:.1f} m",
                 color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, 0.02), ncol=2)
    # (e) ROC
    ax = fig.add_subplot(gs[1, 1])
    roc = res["screening_roc"]
    rows = {(r["leak_to_floor_db"], r["score"]): r for r in res["screening"]}
    lv = -6.0
    ax.plot(roc["abs"]["fpr"], roc["abs"]["tpr"], color=GREY, lw=2,
            label=f"absolute night minimum, AUC {rows[(lv, 'abs')]['auc']:.2f}")
    ax.plot(roc["rel"]["fpr"], roc["rel"]["tpr"], color=GREEN, lw=2.4,
            label=f"vs site baseline, AUC {rows[(lv, 'rel')]['auc']:.2f}")
    ax.plot([0, 1], [0, 1], color=GRID, ls="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"(e) night-noise screening, leak {lv:+.0f} dB vs floor", color=TEXT,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="lower right")
    _style(ax)
    # (f) sensitivity
    ax = fig.add_subplot(gs[1, 2])
    for kind, col, lbl in (("c_error_pct", AMBER, "wave-speed error (%)"),
                           ("clock_sigma_ms", VIOLET, "clock offset sigma (ms)")):
        rr = sorted((r for r in res["sensitivity"] if r["kind"] == kind), key=lambda r: r["value"])
        ax.plot([r["value"] for r in rr], [r["median_m"] for r in rr], "-o", color=col, lw=2,
                ms=4, label=lbl)
    ax.set_xlabel("imperfection (% or ms)")
    ax.set_ylabel("median error (m), cast iron 100 m")
    ax.set_title("(f) sensitivity to wave speed and clock", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    _style(ax)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def linkedin(res, path):
    snr = {(r["material"], r["method"], r["snr_db"]): r for r in res["snr_sweep"]}
    ci = [r for r in res["snr_sweep"] if r["material"] == "cast_iron" and r["method"] == "scot"
          and r["within_5m"] >= 0.9 and r["snr_db"] <= 20]
    # lowest source SNR at which >= 90 % of cast-iron trials land within 5 m
    ci_snr = min(r["snr_db"] for r in ci) if ci else None
    r20 = snr[("cast_iron", "scot", 20)]
    comp = res["compute"]["methods"]["scot"]["median_ms"]
    net = [r for r in res["network"] if r["snr_db"] == 50.0][0]
    rows = {(r["leak_to_floor_db"], r["score"]): r for r in res["screening"]}
    pe150 = [r for r in res["spacing_sweep"] if r["material"] == "pe" and r["length_m"] == 150][0]
    d = {
        "claim": "Open-source acoustic leak locator: two sensors, one correlation peak, one position.",
        "data_label": "simulated data (physics model, 100 trials per point)",
        "stats": [
            [f"{r20['median_m']:.1f} m", "cast iron 100 m, 20 dB, +-5% speed error", "accent"],
            [f"{comp:.1f} ms", "per correlation, laptop CPU (Python)", "green"],
            [f"{pe150['within_5m'] * 100:.0f} %", "PE, 150 m spacing: fixes within 5 m",
             "amber"],
        ],
        "limit": "Simulated only: no dispersion, reflections or real field recordings yet.",
        "alt": "Top: a 100 m pipe with two sensors, true leak at 37 m (red) and estimated "
               "position (green). Bottom: correlation function with a clear peak at the delay.",
        "hashtags": ["#IoT", "#Wasserverlust", "#Leckortung", "#SmartWater", "#LoRaWAN"],
        "post": "",
    }
    d["post"] = (
        "A leak in a buried water main often never shows at the surface. The classic "
        "way to find a leak under the road is to listen to the pipe at two points and ask: when "
        "did the same hiss arrive at each end?\n\n"
        "I built an open-source version of that idea in Python. It simulates a leak with the wave "
        "speed taken from pipe material and wall thickness, frequency-dependent attenuation, "
        "traffic and pump noise and sensor clock offset, then locates it with cross-correlation, "
        "GCC-PHAT and SCOT weighting. It extends to a small pipe network, adds night-noise "
        "screening, and defines a 10-byte LoRaWAN payload with raw snippets on request.\n\n"
        f"Measured in simulation: median error {r20['median_m']:.1f} m on a 100 m cast-iron span "
        f"at 20 dB source SNR with up to 5% wave-speed error, "
        f"{comp:.1f} ms per correlation on a laptop CPU, and on a 9-node "
        f"network {net['edge_correct'] * 100:.0f} % of leaks assigned to the correct pipe. "
        f"On PE pipe at 150 m spacing only {pe150['within_5m'] * 100:.0f} % of fixes land within "
        "5 m, because plastic eats the signal.\n\n"
        "Not verified: everything here is synthetic. The attenuation numbers are assumptions, "
        "and there is no real field recording, dispersion or reflection model yet.\n\n"
        "If you have worked with leak correlators in the field: which effect hurts accuracy "
        "most in practice, wave-speed uncertainty or noise?\n\n"
        "Code: https://github.com/VB-Builds/leak-noise-correlator"
    )
    Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return d, ci_snr, rows


def make_all(res, outdir="docs"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    main_figure(res, out / "figure.png")
    post_figure(res, out / "post-figure.png")
    return linkedin(res, out / "linkedin.json")
