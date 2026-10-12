"""Equations of motion for the moving-support pendulum.

Mechanism assumed (see README "Mechanical model")
-------------------------------------------------
Two parallel guide rods are fixed horizontally. A carriage ("support") slides
along them and is driven by a belt/DC motor. A rigid pendulum is hinged to the
carriage about a horizontal axis perpendicular to the rails, so the pendulum
swings in the vertical plane that contains the rails. Only the carriage is
actuated; the pendulum joint is passive.

Three things are kept distinct:

1. the support *translates* along x (this model);
2. a support that *tilts or rotates* about its own pivot (NOT this model - it
   would have different equations, see ``MechanismModel`` below for how to plug
   one in);
3. the pendulum *rotates relative to the support* (``theta`` below is measured
   from the vertical, not from the support, which never rotates here).

Generalized coordinates ``q = (x, theta)``; ``theta = 0`` is upright and the
tip is at ``(x + L sin(theta), L cos(theta))`` above the pivot.

Derivation (Lagrange)
---------------------
Pendulum centre of mass at distance ``l`` from the pivot, mass ``m``, inertia
``I`` about the COM; support mass ``M``::

    T = 1/2 (M + m) xd^2 + m l xd thd cos(th) + 1/2 (I + m l^2) thd^2
    V = m g l cos(th)

With generalized forces ``Qx = F_motor - F_fric`` and ``Qth = -T_fric`` the
Euler-Lagrange equations are::

    (M + m) xdd + m l cos(th) thdd = F_motor - F_fric + m l thd^2 sin(th)
    m l cos(th) xdd + J thdd       = m g l sin(th) - T_fric           , J = I + m l^2

The mass matrix is positive definite for every angle, so the 2x2 system is
solved in closed form. Hanging (``th = pi``) is stable, upright (``th = 0``) is
unstable; both are equilibria with zero velocity and zero force.

Actuator: the command ``u`` in [-1, 1] drives a first-order lag
(``tau_a * a_dot = u - a``). The motor force is
``F = F_max * (a - xd / v_free)`` clipped to ``+-F_max`` (back-EMF term; set
``motor_free_speed: null`` for an ideal force source).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .parameters import PhysicsParams

# Index layout of the state vector shared by every mechanism model.
X, XD, TH, THD, ACT = range(5)
STATE_NAMES = ("x", "x_dot", "theta", "theta_dot", "actuator")


class NumericalInstabilityError(RuntimeError):
    """Raised when the integrator produces a non-finite state."""


def wrap_angle(angle):
    """Wrap an angle (float or ndarray) to ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class EnergyReport:
    """Mechanical energies [J] of a state."""

    support_kinetic: float
    pendulum_kinetic: float
    potential: float
    total: float
    swing: float          # pendulum energy in the support frame: 1/2 J thd^2 + m g l (cos th - 1)
    swing_normalized: float  # 0 = hanging at rest, 1 = upright at rest


class MechanismModel(ABC):
    """Contract between a physical mechanism and the rest of the project.

    State vector (always 5 entries): ``[support coordinate, support rate,
    pendulum angle, pendulum rate, actuator state]``. For a translating support
    the first pair is position/velocity; a rotating support would reuse the
    slots for its own angle/rate. The environment, agents, controllers and
    renderers only talk to this interface, so replacing the mechanism does not
    require touching the RL code.
    """

    #: identifies the geometry so a matching renderer can be chosen.
    geometry: str = "abstract"

    def __init__(self, params: PhysicsParams):
        self.params = params

    # ---- dynamics -----------------------------------------------------------
    @abstractmethod
    def derivatives(self, state: np.ndarray, command: float) -> np.ndarray:
        """Time derivative of the state for a normalized command in [-1, 1]."""

    @abstractmethod
    def energy(self, state: np.ndarray) -> EnergyReport: ...

    @abstractmethod
    def check_limits(self, state: np.ndarray) -> str | None:
        """Return the name of a violated mechanical/numerical limit, else ``None``."""

    @abstractmethod
    def finalize(self, state: np.ndarray, command: float) -> np.ndarray:
        """Post-step clean-up (angle wrapping, quasi-static actuator states)."""

    @abstractmethod
    def linearize_upright(self) -> tuple[np.ndarray, np.ndarray]:
        """``(A, B)`` of ``z_dot = A z + B F`` about the upright equilibrium, ``z = [q, qd, th, thd]``."""

    @abstractmethod
    def force_from_command(self, state: np.ndarray, command: float) -> float:
        """Generalized force actually applied to the support for the given command."""

    # ---- geometry helpers used by renderers and metrics -----------------------
    @property
    @abstractmethod
    def support_limit(self) -> float:
        """Travel limit of the support coordinate (m for translating supports)."""

    @abstractmethod
    def tip_position(self, state: np.ndarray) -> tuple[float, float]: ...

    # ---- convenience accessors (identical for all models) -----------------------
    @staticmethod
    def pendulum_angle(state) -> float:
        return float(wrap_angle(float(state[TH])))

    @staticmethod
    def make_state(q=0.0, qd=0.0, theta=math.pi, theta_dot=0.0, actuator=0.0) -> np.ndarray:
        return np.array([q, qd, theta, theta_dot, actuator], dtype=np.float64)


class TranslatingSupportModel(MechanismModel):
    """Horizontally translating support carrying a hinged pendulum."""

    geometry = "translating_support"

    def __init__(self, params: PhysicsParams):
        super().__init__(params)
        p = params
        # Cache constants so the hot path only does float arithmetic.
        self._M = p.support_mass
        self._m = p.pendulum_mass
        self._l = p.com_distance
        self._J = p.inertia_about_pivot
        self._ml = p.pendulum_mass * p.com_distance
        self._mgl = p.pendulum_mass * p.gravity * p.com_distance
        self._M11 = p.support_mass + p.pendulum_mass

    # ------------------------------------------------------------------------
    def force_from_command(self, state: np.ndarray, command: float) -> float:
        p = self.params
        u = min(1.0, max(-1.0, float(command)))
        a = state[ACT] if p.actuator_time_constant > 0 else u
        if p.motor_free_speed is None:
            force = p.max_force * a
        else:
            force = p.max_force * (a - state[XD] / p.motor_free_speed)
        return min(p.max_force, max(-p.max_force, force))

    def derivatives(self, state: np.ndarray, command: float) -> np.ndarray:
        p = self.params
        _x, xd, th, thd, a = state.tolist()  # python floats: ~3x faster than numpy scalars here
        u = min(1.0, max(-1.0, float(command)))
        tau = p.actuator_time_constant
        if tau > 0.0:
            a_dot = (u - a) / tau
            a_eff = a
        else:
            a_dot = 0.0
            a_eff = u

        if p.motor_free_speed is None:
            motor = p.max_force * a_eff
        else:
            motor = p.max_force * (a_eff - xd / p.motor_free_speed)
        motor = min(p.max_force, max(-p.max_force, motor))

        f_fric = p.support_viscous_friction * xd + p.support_coulomb_friction * math.tanh(xd / p.support_friction_eps)
        t_fric = p.pivot_viscous_friction * thd + p.pivot_coulomb_friction * math.tanh(thd / p.pivot_friction_eps)

        s, c = math.sin(th), math.cos(th)
        m12 = self._ml * c
        r1 = motor - f_fric + self._ml * thd * thd * s
        r2 = self._mgl * s - t_fric
        det = self._M11 * self._J - m12 * m12  # > 0 since J > (m l)^2 / (M + m) for any physical pendulum
        xdd = (r1 * self._J - m12 * r2) / det
        thdd = (self._M11 * r2 - m12 * r1) / det
        return np.array((xd, xdd, thd, thdd, a_dot))

    # ------------------------------------------------------------------------
    def energy(self, state: np.ndarray) -> EnergyReport:
        p = self.params
        _, xd, th, thd, _ = state.tolist()
        support_ke = 0.5 * self._M * xd * xd
        # centre-of-mass velocity: (xd + l thd cos th, -l thd sin th)
        v2 = xd * xd + 2.0 * xd * self._l * thd * math.cos(th) + (self._l * thd) ** 2
        pend_ke = 0.5 * self._m * v2 + 0.5 * p.inertia_about_com * thd * thd
        pot = self._mgl * math.cos(th)
        swing = 0.5 * self._J * thd * thd + self._mgl * (math.cos(th) - 1.0)
        return EnergyReport(
            support_kinetic=support_ke,
            pendulum_kinetic=pend_ke,
            potential=pot,
            total=support_ke + pend_ke + pot,
            swing=swing,
            swing_normalized=1.0 + swing / (2.0 * self._mgl),
        )

    def check_limits(self, state: np.ndarray) -> str | None:
        if not np.all(np.isfinite(state)):
            return "non_finite_state"
        p = self.params
        if abs(state[X]) > p.rail_half_length:
            return "rail_limit"
        if abs(state[XD]) > p.max_support_speed:
            return "support_overspeed"
        if abs(state[THD]) > p.max_pendulum_speed:
            return "pendulum_overspeed"
        return None

    def finalize(self, state: np.ndarray, command: float) -> np.ndarray:
        state[TH] = wrap_angle(float(state[TH]))
        if self.params.actuator_time_constant <= 0.0:
            state[ACT] = min(1.0, max(-1.0, float(command)))
        return state

    @property
    def support_limit(self) -> float:
        return self.params.rail_half_length

    def tip_position(self, state: np.ndarray) -> tuple[float, float]:
        L = self.params.pendulum_length
        return state[X] + L * math.sin(state[TH]), L * math.cos(state[TH])

    # ------------------------------------------------------------------------
    def linearize_upright(self) -> tuple[np.ndarray, np.ndarray]:
        """Linear model about ``theta = 0`` (viscous friction only, ideal actuator).

        ``z = [x, xd, th, thd]``, input = motor force ``F`` [N].
        """
        p = self.params
        mass = np.array([[self._M11, self._ml], [self._ml, self._J]])
        inv = np.linalg.inv(mass)
        # [xdd, thdd] = inv @ ([F - b xd, m g l th - b_th thd])
        A = np.zeros((4, 4))
        A[0, 1] = 1.0
        A[2, 3] = 1.0
        A[1, 1] = -inv[0, 0] * p.support_viscous_friction
        A[1, 2] = inv[0, 1] * self._mgl
        A[1, 3] = -inv[0, 1] * p.pivot_viscous_friction
        A[3, 1] = -inv[1, 0] * p.support_viscous_friction
        A[3, 2] = inv[1, 1] * self._mgl
        A[3, 3] = -inv[1, 1] * p.pivot_viscous_friction
        B = np.array([[0.0], [inv[0, 0]], [0.0], [inv[1, 0]]])
        return A, B


MODEL_REGISTRY: dict[str, type[MechanismModel]] = {
    TranslatingSupportModel.geometry: TranslatingSupportModel,
}


def create_model(params: PhysicsParams) -> MechanismModel:
    """Instantiate the mechanism model named by ``params.mechanism``."""
    try:
        cls = MODEL_REGISTRY[params.mechanism]
    except KeyError:
        known = ", ".join(sorted(MODEL_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown mechanism {params.mechanism!r}. Registered models: {known}") from None
    return cls(params)
