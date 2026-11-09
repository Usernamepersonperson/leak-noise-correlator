"""Pipe acoustics: wave speed and frequency-dependent attenuation.

Wave speed (thin-walled, axially restrained pipe, fluid-filled), the classical
Korteweg / Joukowsky result:

    c = sqrt( (K / rho) / (1 + psi * K * D / (E * e)) )

K bulk modulus of water, rho density, D inner diameter, e wall thickness,
E Young's modulus of the pipe wall, psi ~ 1 (anchoring factor).
References: Wylie, Streeter & Suo, "Fluid Transients in Systems" (Prentice Hall, 1993), ch. 2;
Korteweg (1878); Joukowsky (1898). For leak-noise correlation in buried pipes:
Gao et al., J. Sound Vib. 277(1-2), 133-148 (2004); Hunaidi & Chu, Applied Acoustics 58(3), 235-254 (1999).

Attenuation: leak noise decays with distance and the decay grows with frequency, much
faster in plastic than in metal. The coefficients below are ILLUSTRATIVE assumptions
chosen to reproduce that qualitative ordering. They are not measured values.
"""
from __future__ import annotations

from dataclasses import dataclass

K_WATER = 2.2e9  # Pa, bulk modulus of water (about 15 C)
RHO_WATER = 1000.0  # kg/m3


@dataclass(frozen=True)
class Material:
    name: str
    youngs_pa: float  # Young's modulus of the wall
    diameter_m: float  # typical inner diameter
    wall_m: float  # typical wall thickness
    atten_db_per_m_100hz: float  # illustrative attenuation at 100 Hz, grows linearly with f


MATERIALS: dict[str, Material] = {
    "steel": Material("steel", 210e9, 0.200, 0.006, 0.010),
    "cast_iron": Material("cast_iron", 100e9, 0.150, 0.010, 0.020),
    "pvc": Material("pvc", 3.0e9, 0.110, 0.0066, 0.15),
    "pe": Material("pe", 0.8e9, 0.110, 0.010, 0.40),
}


def wave_speed(
    youngs_pa: float,
    diameter_m: float,
    wall_m: float,
    k_fluid: float = K_WATER,
    rho: float = RHO_WATER,
    psi: float = 1.0,
) -> float:
    """Pressure-wave speed in a fluid-filled thin-walled pipe (m/s)."""
    if min(youngs_pa, diameter_m, wall_m, k_fluid, rho) <= 0:
        raise ValueError("all physical parameters must be positive")
    return float(((k_fluid / rho) / (1.0 + psi * k_fluid * diameter_m / (youngs_pa * wall_m))) ** 0.5)


def material_wave_speed(name: str) -> float:
    m = MATERIALS[name]
    return wave_speed(m.youngs_pa, m.diameter_m, m.wall_m)


def atten_db_per_m(name: str, f_hz):
    """Attenuation in dB per metre at frequency f (linear in f, illustrative)."""
    return MATERIALS[name].atten_db_per_m_100hz * (f_hz / 100.0)
