"""Reward shaping for swing-up and balancing.

Per control step (``theta`` wrapped to [-pi, pi], ``u`` the normalized command)::

    r =  w_up    * (1 + cos(theta)) / 2                       # height: dense swing-up signal
       + w_prec  * exp(-(theta / sigma)^2)                    # precision: small angular error
       - w_spin  * g(theta) * (theta_dot / spin_scale)^2      # speed penalty, gated by closeness
       - w_pos   * (x / x_limit)^4                            # stay away from the rail ends
       - w_speed * (x_dot / v_max)^2                          # no gratuitous support motion
       - w_u     * u^2                                        # actuator effort
       - w_du    * ((u - u_prev) / 2)^2                       # aggressive switching
       + success_bonus * [balanced long enough]               # terminal
       - violation_penalty * [mechanical limit hit]           # terminal

with ``g(theta) = ((1 + cos(theta)) / 2)^2`` close to 1 only near the top. Nothing in
the reward is proportional to support *motion* or to *time near upright by itself*:
lingering near the top while still spinning is penalized by the spin term, and the
precision term is the only large reward at the top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    w_upright: float = 1.0
    w_precision: float = 1.0
    precision_sigma: float = 0.25        # rad
    w_spin: float = 0.4
    spin_scale: float = 6.0              # rad/s
    w_position: float = 0.5
    w_support_speed: float = 0.02
    w_effort: float = 0.02
    w_smoothness: float = 0.02
    success_bonus: float = 50.0
    violation_penalty: float = 20.0


def compute_reward(cfg: RewardConfig, *, theta: float, theta_dot: float, x: float, x_dot: float,
                   command: float, prev_command: float, x_limit: float, v_max: float,
                   success: bool = False, violation: bool = False) -> tuple[float, dict[str, float]]:
    """Return ``(reward, terms)`` where ``terms`` holds every signed contribution."""
    height = (1.0 + math.cos(theta)) / 2.0
    terms = {
        "upright": cfg.w_upright * height,
        "precision": cfg.w_precision * math.exp(-((theta / cfg.precision_sigma) ** 2)),
        "spin": -cfg.w_spin * height ** 2 * (theta_dot / cfg.spin_scale) ** 2,
        "position": -cfg.w_position * min(2.0, abs(x) / x_limit) ** 4,
        "support_speed": -cfg.w_support_speed * (x_dot / v_max) ** 2,
        "effort": -cfg.w_effort * command ** 2,
        "smoothness": -cfg.w_smoothness * ((command - prev_command) / 2.0) ** 2,
        "success": cfg.success_bonus if success else 0.0,
        "violation": -cfg.violation_penalty if violation else 0.0,
    }
    return float(sum(terms.values())), terms
