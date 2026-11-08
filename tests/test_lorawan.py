import numpy as np
import pytest

from leakcorr import lorawan as lw


def _stats(**kw):
    base = dict(min_db=-62.5, mean_db=-48.0, max_db=-20.5, spread_db=3.4, dom_hz=312,
                battery_mv=3600, temp_c=11, min_minute=187)
    base.update(kw)
    return lw.Stats(**base)


def test_stats_payload_is_10_bytes_and_roundtrips():
    b = lw.encode_stats(_stats(leak_suspect=True), counter=5)
    assert len(b) == 10
    d = lw.decode_stats(b)
    assert d["min_db"] == -62.5 and d["mean_db"] == -48.0 and d["max_db"] == -20.5
    assert d["spread_db"] == pytest.approx(3.4)
    assert d["dom_hz"] == 312 and d["battery_mv"] == 3600 and d["temp_c"] == 11
    assert d["min_minute"] == 187 and d["leak_suspect"] and d["counter"] == 5
    assert not d["snippet_ready"] and not d["low_battery"]


def test_stats_negative_temperature_and_flags():
    d = lw.decode_stats(lw.encode_stats(_stats(temp_c=-12, low_battery=True, snippet_ready=True)))
    assert d["temp_c"] == -12 and d["low_battery"] and d["snippet_ready"]


def test_stats_saturates_instead_of_wrapping():
    d = lw.decode_stats(lw.encode_stats(_stats(min_db=500, mean_db=-500, spread_db=900,
                                               dom_hz=1e5, battery_mv=9000)))
    assert d["min_db"] == 27.5 and d["mean_db"] == -100.0
    assert d["spread_db"] == 25.5 and d["dom_hz"] == 2040 and d["battery_mv"] == 4550


def test_stats_counter_wraps_at_8():
    assert lw.decode_stats(lw.encode_stats(_stats(), counter=9))["counter"] == 1


def test_stats_validation():
    with pytest.raises(ValueError):
        lw.encode_stats(_stats(min_minute=1440))
    with pytest.raises(ValueError):
        lw.encode_stats(_stats(temp_c=200))
    with pytest.raises(ValueError):
        lw.decode_stats(b"\x40" * 9)
    with pytest.raises(ValueError):
        lw.decode_stats(b"\x00" * 10)  # version 0


def test_downlink_request():
    assert lw.encode_snippet_request(9, 4) == bytes([1, 9, 4])
    with pytest.raises(ValueError):
        lw.encode_snippet_request(256, 4)
    with pytest.raises(ValueError):
        lw.encode_snippet_request(1, 0)


def test_adpcm_roundtrip_quality_on_tone():
    x = (3000 * np.sin(np.arange(4000) * 0.15)).round()
    d = lw.decode_snippet(lw.encode_snippet(x, 1), len(x))
    snr = 10 * np.log10(np.sum(x**2) / np.sum((x - d) ** 2))
    assert snr > 18


def test_adpcm_frames_fit_size_and_are_4x_smaller():
    x = np.random.default_rng(0).normal(0, 3000, 2000).round()
    frames = lw.encode_snippet(x, 2, frame_bytes=48)
    assert all(len(f) <= 48 for f in frames)
    payload = sum(len(f) - 6 for f in frames)
    assert payload == pytest.approx(len(x) * 0.5, rel=0.05)  # 4 bit/sample vs 16


def test_snippet_frames_independent_and_loss_tolerant():
    x = (2500 * np.sin(np.arange(1000) * 0.2)).round()
    frames = lw.encode_snippet(x, 3, frame_bytes=40)
    full = lw.decode_snippet(frames, len(x))
    lost = lw.decode_snippet(frames[:2] + frames[3:], len(x))
    per = (40 - 6) * 2
    assert np.array_equal(full[: 2 * per], lost[: 2 * per])
    assert np.array_equal(full[3 * per :], lost[3 * per :])  # later frames still decode exactly
    assert np.all(lost[2 * per : 3 * per] == 0)


def test_snippet_out_of_order_frames():
    x = (1000 * np.sin(np.arange(500) * 0.3)).round()
    frames = lw.encode_snippet(x, 1)
    assert np.array_equal(lw.decode_snippet(frames[::-1]), lw.decode_snippet(frames))


def test_clipping_extremes_do_not_overflow():
    x = np.array([32767, -32768] * 100, float)
    d = lw.decode_snippet(lw.encode_snippet(x, 1), len(x))
    assert d.max() <= 32767 and d.min() >= -32768


def test_malformed_frames():
    with pytest.raises(ValueError):
        lw.decode_frame(b"\x00\x01")
    with pytest.raises(ValueError):
        lw.decode_frame(bytes([0, 0, 1, 0, 0, 200, 0]))
    with pytest.raises(ValueError):
        lw.decode_snippet([])
    with pytest.raises(ValueError):
        lw.encode_snippet(np.zeros(100), 0, frame_bytes=4)


def test_airtime_matches_published_reference():
    # SF7 / 125 kHz, 20-byte PHY payload (7 app + 13 MAC) = 56.58 ms (standard LoRa calculator)
    assert lw.airtime_s(7, 7) == pytest.approx(0.05658, abs=2e-4)
    # SF12 / 125 kHz, 20-byte PHY payload is about 1.32 s
    assert lw.airtime_s(7, 12) == pytest.approx(1.3189, abs=0.01)


def test_airtime_grows_with_sf_and_payload():
    assert lw.airtime_s(10, 12) > lw.airtime_s(10, 9) > lw.airtime_s(10, 7)
    assert lw.airtime_s(50, 9) > lw.airtime_s(10, 9)


def test_snippet_plan_duty_cycle():
    p = lw.snippet_plan(8192, sf=7)
    assert p["frames"] == 98
    assert p["wall_clock_s"] == pytest.approx(100 * p["airtime_total_s"])
    with pytest.raises(ValueError):
        lw.snippet_plan(8192, sf=12, frame_bytes=100)
