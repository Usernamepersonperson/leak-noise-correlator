import numpy as np
import pytest

from leakcorr.correlate import METHODS, estimate, gcc, locate, pick_peak, welch_spectra

FS = 2048.0


def _pair(delay_samples, n=8192, noise=0.0, seed=0):
    """x1(t) = s(t - t1), x2(t) = s(t - t2), tau = t1 - t2 = delay_samples / fs."""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n + 400)
    x1 = s[200 : 200 + n]
    x2 = np.roll(s, -delay_samples)[200 : 200 + n]
    return x1 + noise * rng.standard_normal(n), x2 + noise * rng.standard_normal(n)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("d", [-120, -7, 0, 33, 200])
def test_integer_delay_recovered_all_methods(method, d):
    x1, x2 = _pair(d)
    lags, r = gcc(x1, x2, FS, method, 2048, 20, 900)
    pk = pick_peak(lags, r, 0.2)
    assert pk.tau == pytest.approx(d / FS, abs=0.5 / FS)


def test_subsample_refinement_beats_grid():
    n = 8192
    rng = np.random.default_rng(1)
    s = np.fft.rfft(rng.standard_normal(n + 400))
    f = np.fft.rfftfreq(n + 400, 1 / FS)
    s[(f < 10) | (f > 900)] = 0
    true = 12.4 / FS
    x1 = np.fft.irfft(s, n + 400)[200 : 200 + n]
    x2 = np.fft.irfft(s * np.exp(2j * np.pi * f * true), n + 400)[200 : 200 + n]
    lags, r = gcc(x1, x2, FS, "phat", 2048, 20, 900)
    pk = pick_peak(lags, r, 0.2)
    assert abs(pk.tau - true) < 0.2 / FS


def test_peak_search_respects_physical_window():
    x1, x2 = _pair(400)  # 195 ms
    lags, r = gcc(x1, x2, FS, "phat", 2048, 20, 900)
    pk = pick_peak(lags, r, 0.05)  # window excludes the real peak
    assert abs(pk.tau) <= 0.05 + 1e-9
    assert pk.psr < 2


def test_confidence_drops_with_noise():
    c = []
    for noise in (0.0, 3.0, 30.0):
        x1, x2 = _pair(50, noise=noise)
        lags, r = gcc(x1, x2, FS, "scot", 2048, 20, 900)
        c.append(pick_peak(lags, r, 0.2).psr)
    assert c[0] > c[1] > c[2]


def test_pure_noise_has_low_confidence():
    rng = np.random.default_rng(3)
    lags, r = gcc(rng.standard_normal(8192), rng.standard_normal(8192), FS, "phat", 2048, 20, 900)
    assert pick_peak(lags, r, 0.2).psr < 2.5


def test_locate_formula_and_clipping():
    assert locate(0.0, 100.0, 1000.0) == 50.0
    assert locate(-0.05, 100.0, 1000.0) == pytest.approx(25.0)
    assert locate(0.05, 100.0, 1000.0) == pytest.approx(75.0)
    assert locate(1.0, 100.0, 1000.0) == 100.0
    assert locate(-1.0, 100.0, 1000.0) == 0.0


def test_estimate_end_to_end_known_geometry():
    # leak 30 m from sensor 1 on a 100 m span, c = 1000 m/s
    c, length, x = 1000.0, 100.0, 30.0
    d = round((x - (length - x)) / c * FS)  # t1 - t2 in samples
    x1, x2 = _pair(d)
    xe, pk, _ = estimate(x1, x2, FS, length, c, "phat")
    assert xe == pytest.approx(x, abs=0.5 * c / FS)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        gcc(np.zeros(100), np.zeros(99), FS)
    with pytest.raises(ValueError):
        gcc(np.zeros(100), np.zeros(100), FS, method="nope")
    with pytest.raises(ValueError):
        welch_spectra(np.zeros(8), np.zeros(8))
    with pytest.raises(ValueError):
        pick_peak(np.arange(-5, 5) / FS, np.zeros(10), 0.0)


def test_zero_signal_does_not_crash():
    lags, r = gcc(np.zeros(4096), np.zeros(4096), FS, "phat", 1024)
    pk = pick_peak(lags, r, 0.1)
    assert np.isfinite(pk.tau) and pk.psr == 0.0


def test_short_signal_reduces_nfft():
    x1, x2 = _pair(5, n=1500)
    lags, _ = gcc(x1, x2, FS, "phat", 4096)
    assert len(lags) == 1024
