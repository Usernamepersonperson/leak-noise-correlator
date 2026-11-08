import numpy as np
import pytest

from leakcorr.network import Network, Pipe, demo_network, localise, simulate_loggers
from leakcorr.physics import material_wave_speed


def _line():
    return Network([Pipe("A", "B", 100, "steel"), Pipe("B", "C", 50, "pvc")])


def test_empty_network_rejected():
    with pytest.raises(ValueError):
        Network([])


def test_arrival_time_matches_speed():
    net = _line()
    t, k = net.arrival(0, 30.0, "A")
    assert t == pytest.approx(30 / material_wave_speed("steel"))
    t2, _ = net.arrival(0, 30.0, "C")
    assert t2 == pytest.approx(70 / material_wave_speed("steel") + 50 / material_wave_speed("pvc"))
    assert k > 0


def test_distance_same_edge_and_across_nodes():
    net = _line()
    assert net.distance(0, 20.0, 0, 70.0) == pytest.approx(50.0)
    assert net.distance(0, 90.0, 1, 10.0) == pytest.approx(20.0)


def test_distance_uses_shorter_loop_path():
    pipes = [Pipe("A", "B", 100, "steel"), Pipe("B", "C", 100, "steel"), Pipe("C", "A", 30, "steel")]
    net = Network(pipes)
    assert net.node_distance("A", "B") == 100
    assert net.node_distance("A", "C") == 30
    assert net.distance(0, 5.0, 1, 95.0) == pytest.approx(5 + 30 + 5)


def test_demo_network_shape():
    net, loggers = demo_network()
    assert len(net.nodes) == 9 and len(net.pipes) == 11 and len(loggers) == 5


def test_localise_between_two_loggers_on_one_pipe():
    net = Network([Pipe("A", "B", 100, "cast_iron")])
    rng = np.random.default_rng(0)
    sig = simulate_loggers(rng, net, ["A", "B"], 0, 35.0, 40.0)
    fix = localise(net, sig)
    assert fix is not None and fix.edge == 0
    assert fix.x == pytest.approx(35.0, abs=2.0)


def test_localise_on_demo_network_high_snr():
    net, loggers = demo_network()
    rng = np.random.default_rng(4)
    ok = 0
    for e in (0, 2, 8, 10):
        sig = simulate_loggers(rng, net, loggers, e, 50.0, 50.0)
        fix = localise(net, sig)
        ok += fix is not None and net.distance(e, 50.0, fix.edge, fix.x) < 10
    assert ok >= 3


def test_localise_returns_none_on_noise_only():
    net, loggers = demo_network()
    rng = np.random.default_rng(1)
    sig = {n: rng.standard_normal(8192) for n in loggers}
    assert localise(net, sig, psr_min=50.0) is None
