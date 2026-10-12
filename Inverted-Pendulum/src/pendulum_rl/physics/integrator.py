"""Fixed-step and adaptive integrators for mechanism models.

The command is held constant over each call (zero-order hold), which matches a
digital controller driving a motor driver. ``rk4`` is the default fast path;
``scipy`` (DOP853 with tight tolerances) is used as an accuracy reference.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import MechanismModel, NumericalInstabilityError


def rk4_step(model: MechanismModel, state: np.ndarray, command: float, dt: float) -> np.ndarray:
    f = model.derivatives
    half = 0.5 * dt
    k1 = f(state, command)
    k2 = f(state + half * k1, command)
    k3 = f(state + half * k2, command)
    k4 = f(state + dt * k3, command)
    return state + (dt / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)


def scipy_step(model: MechanismModel, state: np.ndarray, command: float, dt: float,
               rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
    sol = solve_ivp(lambda _t, y: model.derivatives(y, command), (0.0, dt), state,
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise NumericalInstabilityError(f"scipy integrator failed: {sol.message}")
    return sol.y[:, -1]


def advance(model: MechanismModel, state: np.ndarray, command: float, duration: float,
            dt: float | None = None, method: str | None = None) -> np.ndarray:
    """Advance ``state`` by ``duration`` seconds with a constant ``command``.

    The interval is split into ``ceil(duration / dt)`` equal substeps. Returns a
    new, finalized state (wrapped angle). Raises ``NumericalInstabilityError`` if
    the result is not finite.
    """
    params = model.params
    dt = params.dt if dt is None else dt
    method = params.integrator if method is None else method
    if duration <= 0 or dt <= 0:
        raise ValueError(f"duration and dt must be positive (got {duration}, {dt})")
    n = max(1, math.ceil(duration / dt - 1e-9))
    h = duration / n
    step = rk4_step if method == "rk4" else scipy_step
    if method not in ("rk4", "scipy"):
        raise ValueError(f"Unknown integrator {method!r}")

    s = np.array(state, dtype=np.float64, copy=True)
    for _ in range(n):
        s = step(model, s, command, h)
    # NaN/inf propagate through every later step, so one check at the end is sufficient.
    if not np.isfinite(s).all():
        raise NumericalInstabilityError(
            f"non-finite state after integration: {s} (command={command}, dt={h})"
        )
    return model.finalize(s, command)
