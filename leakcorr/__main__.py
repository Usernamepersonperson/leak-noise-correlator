"""CLI: python -m leakcorr {demo,locate,vectors}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import lorawan
from .correlate import METHODS, estimate
from .physics import MATERIALS, material_wave_speed
from .simulate import simulate_pair


def vectors_payload() -> dict:
    """Golden vectors shared by the Python and JavaScript tests."""
    stats = [
        lorawan.Stats(-62.5, -48.0, -20.5, 3.4, 312, 3600, 11, 187, True, False, False),
        lorawan.Stats(-100.0, -100.0, -100.0, 0.0, 0, 2000, -5, 0),
        lorawan.Stats(30.0, 30.0, 30.0, 99.0, 5000, 5000, 127, 1439, True, True, True),
    ]
    rng = np.random.default_rng(0)
    snip = (3000 * np.sin(np.arange(200) * 0.21) + rng.normal(0, 100, 200)).round()
    frames = lorawan.encode_snippet(snip, 7, frame_bytes=40)
    return {
        "stats": [
            {"hex": lorawan.encode_stats(s, i).hex(), "decoded": lorawan.decode_stats(
                lorawan.encode_stats(s, i))}
            for i, s in enumerate(stats)
        ],
        "snippet_frames": [
            {"hex": f.hex(), "decoded": lorawan.decode_frame(f)} for f in frames
        ],
        "downlink": {"hex": lorawan.encode_snippet_request(9, 4).hex(), "snippet_id": 9,
                     "seconds": 4},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="leakcorr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run all experiments and write figures + results")
    d.add_argument("--quick", action="store_true", help="few trials, about a minute")
    d.add_argument("--out", default="docs")
    d.add_argument("--results", default="results/results.json")
    d.add_argument("--workers", type=int, default=None)
    d.add_argument("--no-figures", action="store_true")
    lo = sub.add_parser("locate", help="simulate one leak and locate it")
    lo.add_argument("--material", choices=list(MATERIALS), default="cast_iron")
    lo.add_argument("--length", type=float, default=100.0)
    lo.add_argument("--leak", type=float, default=37.0)
    lo.add_argument("--snr", type=float, default=10.0)
    lo.add_argument("--method", choices=METHODS, default="scot")
    lo.add_argument("--seed", type=int, default=0)
    fg = sub.add_parser("figures", help="redraw figures from a saved results.json")
    fg.add_argument("--results", default="results/results.json")
    fg.add_argument("--out", default="docs")
    v = sub.add_parser("vectors", help="write golden payload vectors for the JS decoder test")
    v.add_argument("--out", default="js/test-vectors.json")
    a = ap.parse_args(argv)

    if a.cmd == "locate":
        rng = np.random.default_rng(a.seed)
        rec = simulate_pair(rng, a.material, a.length, a.leak, a.snr)
        c = material_wave_speed(a.material)
        xe, pk, _ = estimate(rec.x1, rec.x2, rec.fs, a.length, c, a.method)
        print(f"{a.material}: c={c:.0f} m/s, true {a.leak:.2f} m, estimated {xe:.2f} m, "
              f"error {abs(xe - a.leak):.2f} m, tau {pk.tau * 1e3:.2f} ms, psr {pk.psr:.1f}, "
              f"confidence {pk.confidence:.2f}")
        return 0
    if a.cmd == "vectors":
        Path(a.out).write_text(json.dumps(vectors_payload(), indent=1) + "\n", encoding="utf-8")
        print("wrote", a.out)
        return 0
    from . import experiments, figures

    if a.cmd == "figures":
        figures.make_all(json.loads(Path(a.results).read_text(encoding="utf-8")), a.out)
        print("figures + linkedin.json written to", a.out)
        return 0

    res = experiments.run_all(a.quick, a.workers)
    p = Path(a.results)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res), encoding="utf-8")
    print(f"results: {p} ({res['runtime_s']:.0f} s)")
    if not a.no_figures:
        figures.make_all(res, a.out)
        print("figures + linkedin.json written to", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
