"""Physical parameters of the moving-support pendulum.

All quantities are SI units. Angle convention (used everywhere in the project):

* ``theta = 0``      -> pendulum upright (the unstable equilibrium / the goal)
* ``theta = +-pi``   -> pendulum hanging straight down (the stable equilibrium)
* ``theta > 0``      -> pendulum tip displaced toward the **+x** side of the support
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsParams:
    """Configurable physical parameters of the translating-support pendulum."""

    # --- which dynamics model to instantiate (see physics.dynamics.MODEL_REGISTRY)
    mechanism: str = "translating_support"

    # --- inertial properties -------------------------------------------------
    support_mass: float = 0.50          # kg, carriage + bearings + belt clamp (M)
    pendulum_mass: float = 0.15         # kg, rod + bob (m)
    pendulum_length: float = 0.40       # m, pivot -> tip (rendering + tip speed)
    com_fraction: float = 0.5           # pivot -> centre of mass, as a fraction of length
    inertia_com: float | None = None    # kg m^2 about the COM; None -> uniform rod m L^2 / 12
    gravity: float = 9.81               # m/s^2

    # --- friction (viscous + smoothed Coulomb) -------------------------------
    support_viscous_friction: float = 0.15    # N s / m, carriage on the guide rails
    support_coulomb_friction: float = 0.05    # N
    pivot_viscous_friction: float = 0.002     # N m s / rad, pendulum bearing
    pivot_coulomb_friction: float = 0.0005    # N m
    support_friction_eps: float = 0.01        # m/s,   tanh smoothing width
    pivot_friction_eps: float = 0.05          # rad/s, tanh smoothing width

    # --- actuator (belt-driven DC motor) --------------------------------------
    max_force: float = 10.0                   # N, stall / peak force at full command
    motor_free_speed: float | None = 4.0      # m/s, no-load speed at full command (back-EMF); None -> ideal force source
    actuator_time_constant: float = 0.02      # s, first-order lag of the motor driver; 0 -> instantaneous

    # --- mechanical limits -----------------------------------------------------
    rail_half_length: float = 0.60            # m, usable travel is [-L, +L] about the rail centre
    max_support_speed: float = 3.0            # m/s, safety limit
    max_pendulum_speed: float = 40.0          # rad/s, divergence guard

    # --- numerical integration ---------------------------------------------------
    dt: float = 0.005                         # s, physics integration step
    integrator: str = "rk4"                   # "rk4" or "scipy" (DOP853 reference)

    # ------------------------------------------------------------------------
    def __post_init__(self) -> None:
        positive = (
            "support_mass", "pendulum_mass", "pendulum_length", "gravity", "max_force",
            "rail_half_length", "max_support_speed", "max_pendulum_speed", "dt",
            "support_friction_eps", "pivot_friction_eps",
        )
        for name in positive:
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"PhysicsParams.{name} must be a finite positive number, got {value!r}")
        for name in ("support_viscous_friction", "support_coulomb_friction",
                     "pivot_viscous_friction", "pivot_coulomb_friction", "actuator_time_constant"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"PhysicsParams.{name} must be finite and >= 0, got {value!r}")
        if not 0.0 < self.com_fraction <= 1.0:
            raise ValueError(f"com_fraction must be in (0, 1], got {self.com_fraction}")
        if self.inertia_com is not None and not (math.isfinite(self.inertia_com) and self.inertia_com >= 0):
            raise ValueError(f"inertia_com must be >= 0 or None, got {self.inertia_com}")
        if self.motor_free_speed is not None and not self.motor_free_speed > 0:
            raise ValueError("motor_free_speed must be > 0 or None (ideal force source)")
        if self.integrator not in ("rk4", "scipy"):
            raise ValueError(f"integrator must be 'rk4' or 'scipy', got {self.integrator!r}")

    # --- derived quantities ---------------------------------------------------------
    @property
    def com_distance(self) -> float:
        """Distance l from the pivot to the pendulum's centre of mass [m]."""
        return self.com_fraction * self.pendulum_length

    @property
    def inertia_about_com(self) -> float:
        if self.inertia_com is not None:
            return float(self.inertia_com)
        return self.pendulum_mass * self.pendulum_length ** 2 / 12.0

    @property
    def inertia_about_pivot(self) -> float:
        """J = I_com + m l^2 [kg m^2]."""
        return self.inertia_about_com + self.pendulum_mass * self.com_distance ** 2

    @property
    def small_oscillation_frequency(self) -> float:
        """Hanging small-oscillation angular frequency sqrt(m g l / J) [rad/s] (no friction)."""
        return math.sqrt(self.pendulum_mass * self.gravity * self.com_distance / self.inertia_about_pivot)

    # --- utilities ---------------------------------------------------------------------
    def replace(self, **changes) -> "PhysicsParams":
        return dataclasses.replace(self, **changes)

    def randomized(self, rng: np.random.Generator, scale: float) -> "PhysicsParams":
        """Return a copy with masses, friction, length and actuator strength perturbed by +-``scale``.

        ``scale = 0.1`` multiplies each perturbed quantity by U(0.9, 1.1). Used by
        the combined training stage to improve robustness; geometry limits and the
        timestep are never perturbed.
        """
        if scale <= 0:
            return self
        names = (
            "support_mass", "pendulum_mass", "pendulum_length",
            "support_viscous_friction", "support_coulomb_friction",
            "pivot_viscous_friction", "pivot_coulomb_friction", "max_force",
        )
        changes = {n: getattr(self, n) * float(rng.uniform(1.0 - scale, 1.0 + scale)) for n in names}
        if self.inertia_com is not None:  # keep the user-specified inertia consistent with the new mass
            changes["inertia_com"] = self.inertia_com * changes["pendulum_mass"] / self.pendulum_mass
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
