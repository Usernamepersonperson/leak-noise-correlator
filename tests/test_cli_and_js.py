import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from leakcorr.__main__ import main, vectors_payload
from leakcorr.experiments import compute_cost, network_trials, snippet_effect

ROOT = Path(__file__).resolve().parent.parent


def test_golden_vectors_file_is_current():
    on_disk = json.loads((ROOT / "js" / "test-vectors.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(vectors_payload()))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_decoder_matches_python():
    r = subprocess.run(["node", str(ROOT / "js" / "test.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_locate_cli_runs(capsys):
    args = ["locate", "--material", "steel", "--length", "80", "--leak", "30", "--snr", "30"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "estimated" in out and "error" in out


def test_compute_cost_reports_all_methods():
    r = compute_cost(repeats=3)
    assert set(r["methods"]) == {"xcorr", "phat", "scot", "roth"}
    assert all(v["median_ms"] > 0 for v in r["methods"].values())


def test_network_trials_smoke():
    rows, info = network_trials(3, snrs=(50.0,))
    assert rows[0]["trials"] == 3 and info["pipes"] == 11
    assert len(rows[0]["errors_m"]) == 3


def test_adpcm_does_not_degrade_localisation():
    rows, plans = snippet_effect(4)
    for r in rows:
        assert r["median_err_adpcm_m"] < 2.0
        assert r["adpcm_snr_db"] > 12
    assert plans["sf7"]["wall_clock_s"] < plans["sf9"]["wall_clock_s"]
    assert np.isfinite(plans["sf7"]["airtime_total_s"])
