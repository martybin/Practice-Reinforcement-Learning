"""Model-based baseline: energy-shaping swing-up + LQR stabilization.

This is a *reference*, not a replacement for the DQN agent.

Swing-up (Astrom & Furuta, 1996)
--------------------------------
Let ``E = 1/2 J th_dot^2 + m g l (cos th - 1)`` be the pendulum energy in the
support frame (``E = 0`` upright at rest, ``E = -2 m g l`` hanging at rest).
From the equations of motion, the support acceleration ``a = x_ddot`` changes
it at the rate ``dE/dt = -m l a th_dot cos(th)``. Choosing
``a = k (E - E_ref) sign(th_dot cos th)`` therefore drives ``E`` to ``E_ref``.
A PD term on the support removes drift and keeps it away from the rail ends.

Stabilization
-------------
Once the pendulum is inside a capture region, a discrete-time LQR designed on
the linearization about the upright equilibrium takes over (state feedback on
``[x, x_dot, th, th_dot]``, i.e. a PD-style law on the pendulum plus rail
centering). A hysteresis band returns to swing-up if the pendulum falls out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from ..physics import MechanismModel
from ..physics.dynamics import wrap_angle


@dataclass
class BaselineConfig:
    control_dt: float = 0.02
    # --- swing-up (energy shaping) ---
    energy_gain: float = 2.7          # normalized-command gain per unit of (E_ref - E)/(2 m g l)
    energy_margin: float = 0.10       # extra target energy above upright (fraction of 2 m g l)
    kick_command: float = 1.0         # command used while the pendulum is (almost) at rest
    kick_speed: float = 0.15          # rad/s below which the kick applies
    centering_stiffness: float = 0.4  # normalized command per metre of (filtered) displacement from the rail centre
    centering_damping: float = 1.3    # normalized command per m/s of (filtered) support speed
    washout_time: float = 1.6         # s, low-pass time constant: removes drift without fighting the pumping motion
    pump_travel: float = 0.53         # fraction of the rail the pumping push may use before it fades out
    wall_start: float = 0.87          # fraction of the rail where the virtual end-stop begins to brake
    # --- stabilization (LQR) ---
    capture_angle: float = 0.57       # rad
    capture_speed: float = 6.8        # rad/s
    release_angle: float = 1.0        # rad, hysteresis: leave LQR beyond this
    q_diag: tuple = (4.0, 1.0, 60.0, 4.0)   # weights on [x, x_dot, theta, theta_dot]
    r_weight: float = 0.5                   # weight on the (normalized) command


class EnergyShapingLQRController:
    """Switching controller; ``command(state4)`` returns a normalized command in [-1, 1]."""

    name = "baseline"

    def __init__(self, model: MechanismModel, config: BaselineConfig | None = None):
        self.model = model
        self.cfg = config or BaselineConfig()
        p = model.params
        self._mgl = p.pendulum_mass * p.gravity * p.com_distance
        self._J = p.inertia_about_pivot
        self.K = self._design_lqr()
        self.mode = "swing_up"
        self._x_slow = 0.0
        self._v_slow = 0.0

    def _design_lqr(self) -> np.ndarray:
        A, B = self.model.linearize_upright()
        B = B * self.model.params.max_force            # input becomes the normalized command
        Ad, Bd, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), self.cfg.control_dt, method="zoh")
        Q = np.diag(self.cfg.q_diag)
        R = np.array([[self.cfg.r_weight]])
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        return K  # u = -K z

    def reset(self) -> None:
        self.mode = "swing_up"
        self._x_slow = 0.0
        self._v_slow = 0.0

    # ------------------------------------------------------------------------
    def command(self, state) -> float:
        x, xd, th, thd = (float(state[0]), float(state[1]), float(wrap_angle(float(state[2]))), float(state[3]))
        c = self.cfg
        if self.mode == "swing_up":
            if abs(th) < c.capture_angle and abs(thd) < c.capture_speed:
                self.mode = "stabilize"
        elif abs(th) > c.release_angle:
            self.mode = "swing_up"

        if self.mode == "stabilize":
            z = np.array([x, xd, th, thd])
            return float(np.clip(-(self.K @ z)[0], -1.0, 1.0))
        return self._swing_up_command(x, xd, th, thd)

    def _swing_up_command(self, x: float, xd: float, th: float, thd: float) -> float:
        c = self.cfg
        x_lim = self.model.support_limit
        energy = 0.5 * self._J * thd * thd + self._mgl * (math.cos(th) - 1.0)
        e_err = (energy - c.energy_margin * 2.0 * self._mgl) / (2.0 * self._mgl)   # < 0: needs more energy
        direction = thd * math.cos(th)
        if abs(thd) < c.kick_speed and abs(th) > 2.5:
            u = c.kick_command if xd >= 0 else -c.kick_command   # break the symmetry at rest
        else:
            u = c.energy_gain * e_err * (1.0 if direction >= 0 else -1.0)
        # Rail-aware pumping: fade the push out as the support nears the end of its travel.
        # (Sign-based pumping starts with a one-sided drift; without this the first half-swing
        # would run into the rail.)
        if u != 0.0:
            used = x * math.copysign(1.0, u) / (c.pump_travel * x_lim)   # >0 when pushing outward
            u *= min(1.0, max(0.0, 1.0 - used))
        # PD on the low-pass-filtered support state removes the drift but leaves the
        # pendulum-frequency oscillation (which *is* the pumping motion) alone.
        alpha = min(1.0, c.control_dt / c.washout_time)
        self._x_slow += alpha * (x - self._x_slow)
        self._v_slow += alpha * (xd - self._v_slow)
        u += -c.centering_stiffness * self._x_slow - c.centering_damping * self._v_slow
        # Soft virtual end-stop: brake hard when travelling outward inside the wall zone.
        depth = (abs(x) / x_lim - c.wall_start) / (1.0 - c.wall_start)
        if depth > 0.0:
            side = math.copysign(1.0, x)
            u += -side * min(1.0, 3.0 * depth)
            if xd * side > 0:
                u += -side * min(1.0, 1.5 * xd * side)
        return float(np.clip(u, -1.0, 1.0))

    # Policy-protocol alias used by the evaluator.
    def act(self, state) -> float:
        return self.command(state)
