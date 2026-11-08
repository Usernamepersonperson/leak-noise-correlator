import numpy as np
import pytest

from leakcorr.physics import MATERIALS, atten_db_per_m, material_wave_speed, wave_speed


def test_rigid_pipe_limit_is_water_sound_speed():
    # E -> infinity: c -> sqrt(K/rho) = 1483 m/s
    assert wave_speed(1e30, 0.1, 0.01) == pytest.approx(1483.2, abs=0.2)


def test_known_value_cast_iron():
    # K/rho = 2.2e6; K*D/(E*e) = 0.33 -> c = 1483.2 / sqrt(1.33)
    assert material_wave_speed("cast_iron") == pytest.approx(1286.1, abs=0.5)


def test_material_ordering():
    c = {m: material_wave_speed(m) for m in MATERIALS}
    assert c["steel"] > c["pvc"] > c["pe"]
    assert c["cast_iron"] > c["pvc"]


def test_plastic_speeds_in_literature_range():
    assert 300 < material_wave_speed("pvc") < 500
    assert 200 < material_wave_speed("pe") < 350


def test_softer_wall_slows_wave():
    assert wave_speed(1e9, 0.1, 0.01) < wave_speed(1e10, 0.1, 0.01)


def test_thicker_wall_speeds_wave():
    assert wave_speed(1e9, 0.1, 0.02) > wave_speed(1e9, 0.1, 0.01)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError):
        wave_speed(bad, 0.1, 0.01)
    with pytest.raises(ValueError):
        wave_speed(1e9, bad, 0.01)


def test_attenuation_grows_with_frequency_and_plastic_is_worse():
    f = np.array([50.0, 100.0, 400.0])
    a = atten_db_per_m("pe", f)
    assert np.all(np.diff(a) > 0)
    assert atten_db_per_m("pe", 100.0) > atten_db_per_m("pvc", 100.0)
    assert atten_db_per_m("pvc", 100.0) > atten_db_per_m("steel", 100.0)
