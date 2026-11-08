import numpy as np
import pytest

from leakcorr.correlate import estimate
from leakcorr.physics import material_wave_speed
from leakcorr.simulate import propagate, simulate_pair


@pytest.mark.parametrize("mat", ["steel", "cast_iron", "pvc", "pe"])
def test_clean_recovery_each_material(mat):
    rng = np.random.default_rng(5)
    c = material_wave_speed(mat)
    rec = simulate_pair(rng, mat, 60.0, 22.0, 40.0)
    xe, pk, _ = estimate(rec.x1, rec.x2, rec.fs, 60.0, c, "scot", 2048)
    assert abs(xe - 22.0) < 0.5


def test_propagate_delay_is_exact():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(4096)
    y = propagate(s, 2048.0, 10 / 2048.0, 0.0)
    assert np.allclose(y, np.roll(s, 10), atol=1e-9)


def test_attenuation_reduces_power_more_at_high_frequency():
    n, fs = 4096, 2048.0
    t = np.arange(n) / fs
    lo, hi = np.sin(2 * np.pi * 64 * t), np.sin(2 * np.pi * 512 * t)
    a_lo = np.std(propagate(lo, fs, 0, 0.3 * 10)) / np.std(lo)
    a_hi = np.std(propagate(hi, fs, 0, 0.3 * 10)) / np.std(hi)
    assert a_hi < a_lo < 1


def test_snr_controls_effective_snr():
    rng = np.random.default_rng(1)
    a = simulate_pair(rng, "steel", 50.0, 10.0, 0.0)
    b = simulate_pair(rng, "steel", 50.0, 10.0, 20.0)
    assert b.eff_snr_db - a.eff_snr_db == pytest.approx(20.0, abs=3.0)


def test_plastic_loses_more_snr_over_distance():
    rng = np.random.default_rng(2)
    s = simulate_pair(rng, "steel", 100.0, 50.0, 20.0)
    p = simulate_pair(rng, "pe", 100.0, 50.0, 20.0)
    assert p.eff_snr_db < s.eff_snr_db - 10


def test_clock_offset_shifts_estimate_by_c_dt_over_2():
    c = material_wave_speed("cast_iron")
    errs = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        rec = simulate_pair(rng, "cast_iron", 100.0, 50.0, 40.0, clock_offset_s=2e-3)
        xe, _, _ = estimate(rec.x1, rec.x2, rec.fs, 100.0, c, "scot", 2048)
        errs.append(xe - 50.0)
    # sensor 2 delayed by dt -> tau smaller by dt -> estimate shifted by -c*dt/2
    assert np.mean(errs) == pytest.approx(-c * 2e-3 / 2, abs=0.2)


def test_wave_speed_error_shifts_estimate_proportionally():
    c = material_wave_speed("cast_iron")
    rng = np.random.default_rng(0)
    rec = simulate_pair(rng, "cast_iron", 100.0, 80.0, 40.0, c_error_frac=0.05)
    xe, _, _ = estimate(rec.x1, rec.x2, rec.fs, 100.0, c, "scot", 2048)
    # true tau = 60 / (1.05 c); estimate = 50 + c * tau / 2 = 50 + 30 / 1.05
    assert xe == pytest.approx(50 + 30 / 1.05, abs=0.3)


def test_leak_off_pipe_rejected():
    with pytest.raises(ValueError):
        simulate_pair(np.random.default_rng(0), "steel", 50.0, 60.0, 10.0)


def test_reproducible_with_seed():
    a = simulate_pair(np.random.default_rng(9), "pvc", 50.0, 20.0, 10.0)
    b = simulate_pair(np.random.default_rng(9), "pvc", 50.0, 20.0, 10.0)
    assert np.array_equal(a.x1, b.x1)
