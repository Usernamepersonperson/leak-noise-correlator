"""Leak localisation on a small pipe network from pairwise logger correlations.

Every logger pair gives one time-difference-of-arrival tau_ij = T_i - T_j, where T_k is
the travel time from the leak to logger k along the fastest path. A grid search over all
pipes and positions finds the point whose predicted tau_ij best matches the measurements
(truncated quadratic cost, so one wrong pair does not dominate).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .correlate import gcc, pick_peak
from .physics import MATERIALS, material_wave_speed
from .simulate import FS, SRC_BAND, background, propagate


@dataclass(frozen=True)
class Pipe:
    u: str
    v: str
    length_m: float
    material: str

    @property
    def c(self) -> float:
        return material_wave_speed(self.material)

    @property
    def k_atten(self) -> float:
        return MATERIALS[self.material].atten_db_per_m_100hz


class Network:
    def __init__(self, pipes: list[Pipe], coords: dict[str, tuple[float, float]] | None = None):
        if not pipes:
            raise ValueError("network needs at least one pipe")
        self.pipes = list(pipes)
        self.coords = coords or {}
        self.nodes = sorted({n for p in pipes for n in (p.u, p.v)})
        self.adj: dict[str, list[tuple[str, int]]] = {n: [] for n in self.nodes}
        for i, p in enumerate(self.pipes):
            self.adj[p.u].append((p.v, i))
            self.adj[p.v].append((p.u, i))
        self._cache: dict[str, tuple[dict, dict]] = {}

    def _dijkstra(self, src: str):
        """Travel time (s) and (time, atten_k, length) bookkeeping from a node to all nodes."""
        if src in self._cache:
            return self._cache[src]
        dist = {n: float("inf") for n in self.nodes}
        att = {n: 0.0 for n in self.nodes}
        dist[src] = 0.0
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist[n]:
                continue
            for m, ei in self.adj[n]:
                p = self.pipes[ei]
                nd = d + p.length_m / p.c
                if nd < dist[m]:
                    dist[m] = nd
                    att[m] = att[n] + p.k_atten * p.length_m
                    heapq.heappush(heap, (nd, m))
        self._cache[src] = (dist, att)
        return dist, att

    def arrival(self, edge: int, x: float, logger: str):
        """(travel time, attenuation coefficient sum) from a leak at x on pipe `edge`
        (x measured from pipe.u) to a logger node, via the faster pipe end."""
        p = self.pipes[edge]
        du, au = self._dijkstra(p.u)
        dv, av = self._dijkstra(p.v)
        via_u = x / p.c + du[logger]
        via_v = (p.length_m - x) / p.c + dv[logger]
        if via_u <= via_v:
            return via_u, p.k_atten * x + au[logger]
        return via_v, p.k_atten * (p.length_m - x) + av[logger]

    def node_distance(self, a: str, b: str) -> float:
        """Shortest path length in metres (not time)."""
        dist = {n: float("inf") for n in self.nodes}
        dist[a] = 0.0
        heap = [(0.0, a)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist[n]:
                continue
            for m, ei in self.adj[n]:
                nd = d + self.pipes[ei].length_m
                if nd < dist[m]:
                    dist[m] = nd
                    heapq.heappush(heap, (nd, m))
        return dist[b]

    def distance(self, e1: int, x1: float, e2: int, x2: float) -> float:
        """Shortest along-pipe distance between two points on the network."""
        p1, p2 = self.pipes[e1], self.pipes[e2]
        best = abs(x1 - x2) if e1 == e2 else float("inf")
        for n1, d1 in ((p1.u, x1), (p1.v, p1.length_m - x1)):
            for n2, d2 in ((p2.u, x2), (p2.v, p2.length_m - x2)):
                best = min(best, d1 + self.node_distance(n1, n2) + d2)
        return best


def demo_network() -> tuple[Network, list[str]]:
    """A 9-node, 11-pipe district with mixed materials; loggers on 5 nodes."""
    coords = {
        "A": (0, 0), "B": (120, 0), "C": (240, 0), "D": (0, 110), "E": (120, 110),
        "F": (240, 110), "G": (0, 220), "H": (120, 220), "I": (240, 220),
    }
    pipes = [
        Pipe("A", "B", 120, "cast_iron"), Pipe("B", "C", 120, "cast_iron"),
        Pipe("D", "E", 120, "pvc"), Pipe("E", "F", 120, "pvc"),
        Pipe("G", "H", 120, "pe"), Pipe("H", "I", 120, "pe"),
        Pipe("A", "D", 110, "steel"), Pipe("D", "G", 110, "pvc"),
        Pipe("B", "E", 110, "cast_iron"), Pipe("E", "H", 110, "pvc"),
        Pipe("C", "F", 110, "cast_iron"),
    ]
    return Network(pipes, coords), ["A", "C", "E", "G", "I"]


def simulate_loggers(
    rng,
    net: Network,
    loggers: list[str],
    edge: int,
    x: float,
    snr_db: float,
    dur_s: float = 4.0,
    fs: float = FS,
    clock_sigma_s: float = 0.0,
) -> dict[str, np.ndarray]:
    """Recordings at each logger node for a leak on pipe `edge` at x from pipe.u."""
    times = {lg: net.arrival(edge, x, lg) for lg in loggers}
    n = int(dur_s * fs)
    pad = int(np.ceil((max(t for t, _ in times.values()) + 0.1) * fs))
    ntot = n + 2 * pad
    s = np.fft.rfft(rng.standard_normal(ntot))
    f = np.fft.rfftfreq(ntot, 1 / fs)
    s[(f < SRC_BAND[0]) | (f > SRC_BAND[1])] = 0
    src = np.fft.irfft(s, ntot)
    src /= src.std()
    sigma_b = 10.0 ** (-snr_db / 20.0)
    out = {}
    for lg in loggers:
        t, k = times[lg]
        t = t + (rng.normal(0, clock_sigma_s) if clock_sigma_s > 0 else 0.0)
        y = propagate(src, fs, max(t, 0.0), k) + background(rng, ntot, fs, sigma_b)
        out[lg] = y[pad : pad + n]
    return out


@dataclass
class NetFix:
    edge: int
    x: float  # from pipe.u
    cost: float
    n_pairs_used: int


def localise(
    net: Network,
    signals: dict[str, np.ndarray],
    fs: float = FS,
    method: str = "scot",
    psr_min: float = 1.6,
    step_m: float = 1.0,
    nfft: int = 2048,
    cap_s2: float = (0.01) ** 2,
) -> NetFix | None:
    """Estimate the leak position on the network. Returns None when no pair is usable."""
    names = sorted(signals)
    meas = []  # (i, j, tau)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            i, j = names[a], names[b]
            tmax = net._dijkstra(i)[0][j]
            lags, r = gcc(signals[i], signals[j], fs, method, nfft, 20.0, 900.0)
            pk = pick_peak(lags, r, tmax)
            if pk.psr >= psr_min:
                meas.append((i, j, pk.tau))
    if not meas:
        return None
    best = None
    for ei, p in enumerate(net.pipes):
        xs = np.arange(0.0, p.length_m + 1e-9, step_m)
        cost = np.zeros_like(xs)
        for i, j, tau in meas:
            ti = np.array([net.arrival(ei, x, i)[0] for x in xs])
            tj = np.array([net.arrival(ei, x, j)[0] for x in xs])
            cost += np.minimum((tau - (ti - tj)) ** 2, cap_s2)
        k = int(np.argmin(cost))
        if best is None or cost[k] < best.cost:
            best = NetFix(ei, float(xs[k]), float(cost[k]), len(meas))
    return best
