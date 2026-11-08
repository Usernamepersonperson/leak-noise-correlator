import numpy as np
import pytest

from leakcorr.screening import night_windows, roc_auc, screening_scores, window_levels_db

QUIET = dict(event_prob=0, persistent_prob=0, hum_scale=0, night_sigma_db=0)


def test_level_of_white_noise_scales_with_amplitude():
    rng = np.random.default_rng(0)
    a = window_levels_db(rng.standard_normal((4, 2048)))
    b = window_levels_db(10 * rng.standard_normal((4, 2048)))
    assert np.mean(b - a) == pytest.approx(20.0, abs=1.0)


def test_leak_raises_night_minimum():
    rng = np.random.default_rng(1)
    base = [window_levels_db(night_windows(rng, 1.0, None)).min() for _ in range(40)]
    leak = [window_levels_db(night_windows(rng, 1.0, 6.0)).min() for _ in range(40)]
    assert np.mean(leak) - np.mean(base) > 3.0


def test_leak_level_matches_requested_ratio():
    rng = np.random.default_rng(2)
    base = window_levels_db(night_windows(rng, 1.0, None, **QUIET)).mean()
    leak = window_levels_db(night_windows(rng, 1.0, 0.0, **QUIET)).mean()
    assert leak - base == pytest.approx(3.0, abs=0.7)  # equal power adds 3 dB


def test_minimum_rejects_sporadic_events():
    rng = np.random.default_rng(3)
    lv = window_levels_db(night_windows(rng, 1.0, None, event_prob=0.3, persistent_prob=0))
    assert lv.max() > lv.min() + 1.0


def test_roc_perfect_and_random():
    _, _, auc = roc_auc(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    assert auc == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    _, _, auc = roc_auc(rng.standard_normal(2000), rng.standard_normal(2000))
    assert auc == pytest.approx(0.5, abs=0.05)


def test_roc_inverted_and_empty():
    _, _, auc = roc_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0]))
    assert auc == pytest.approx(0.0)
    with pytest.raises(ValueError):
        roc_auc(np.array([]), np.array([1.0]))


def test_site_baseline_beats_absolute_threshold():
    rng = np.random.default_rng(5)
    s = screening_scores(rng, 0.0, n_sites=60)
    auc_abs = roc_auc(s["abs_neg"], s["abs_pos"])[2]
    auc_rel = roc_auc(s["rel_neg"], s["rel_pos"])[2]
    assert auc_rel > auc_abs + 0.1
