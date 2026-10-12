"""Gymnasium environment for swing-up and stabilization of the moving-support pendulum.

Observation (float32, normalized, ``obs_dim = 5`` or ``6``)::

    [ x / x_limit,  x_dot / v_scale,  sin(theta),  cos(theta),  theta_dot / omega_scale,  (u_prev) ]

``theta = 0`` is upright. ``u_prev`` (previous normalized command) is included by
default because the actuator has a lag, which makes the closed loop closer to Markov.

Actions
-------
* ``action_mode="discrete"`` (default, used by DQN): ``Discrete(n)``; action ``i`` applies the
  normalized support-drive command ``action_levels[i]`` in [-1, 1]
  (negative = push the support toward -x, positive = toward +x, 0 = release).
* ``action_mode="continuous"``: ``Box(-1, 1, (1,))`` with the same meaning (used by the baseline).

Episode end
-----------
* ``terminated``: success (if ``terminate_on_success``) or a physical constraint violation
  (rail end, support/pendulum overspeed, non-finite state).
* ``truncated``: the time limit ``max_episode_seconds`` was reached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..physics import (
    ACT, TH, THD, X, XD, MechanismModel, NumericalInstabilityError, PhysicsParams, advance, create_model, wrap_angle,
)
from .rewards import RewardConfig, compute_reward

PHASES = ("swinging_up", "stabilizing", "balanced", "failed")
VIOLATIONS = ("rail_limit", "support_overspeed", "pendulum_overspeed", "non_finite_state", "numerical_instability")


@dataclass(frozen=True)
class EnvConfig:
    # --- timing ---
    control_dt: float = 0.02              # s between agent decisions (physics.dt must divide it)
    max_episode_seconds: float = 10.0
    # --- initial states ---
    start_mode: str = "mixed"             # "hanging" | "upright" | "mixed" | "random"
    mixed_upright_probability: float = 0.5
    hanging_angle_noise: float = 0.1      # rad, +- around theta = pi
    hanging_velocity_noise: float = 0.1   # rad/s
    upright_angle_range: float = 0.25     # rad, +- around theta = 0
    upright_velocity_range: float = 0.5   # rad/s
    start_position_range: float = 0.05    # m, +- around the rail centre
    # --- task definition ---
    success_mode: str = "balance"         # "balance": hold balance for hold_seconds | "reach": enter the catch region
    terminate_on_success: bool = True
    upright_region: float = 0.35          # rad, "swing-up reached" if |theta| below this
    balance_angle_tol: float = 0.1        # rad
    balance_velocity_tol: float = 1.0     # rad/s
    hold_seconds: float = 2.0
    catch_angle: float = 0.35             # rad, success region for success_mode="reach"
    catch_velocity: float = 4.0           # rad/s
    # --- actions ---
    action_mode: str = "discrete"         # "discrete" | "continuous"
    action_levels: tuple = (-1.0, 0.0, 1.0)
    # --- observation ---
    omega_scale: float = 15.0             # rad/s mapped to 1.0
    include_prev_action: bool = True
    noise_position: float = 0.0           # std of additive sensor noise (physical units)
    noise_velocity: float = 0.0
    noise_angle: float = 0.0
    noise_angular_velocity: float = 0.0
    # --- robustness ---
    physics_randomization: float = 0.0    # +- fraction applied to masses / friction / force each reset
    disturbance_probability: float = 0.0  # per control step
    disturbance_magnitude: float = 3.0    # rad/s, uniform kick on theta_dot

    def __post_init__(self) -> None:
        if self.start_mode not in ("hanging", "upright", "mixed", "random"):
            raise ValueError(f"start_mode must be hanging|upright|mixed|random, got {self.start_mode!r}")
        if self.success_mode not in ("balance", "reach"):
            raise ValueError(f"success_mode must be 'balance' or 'reach', got {self.success_mode!r}")
        if self.action_mode not in ("discrete", "continuous"):
            raise ValueError(f"action_mode must be 'discrete' or 'continuous', got {self.action_mode!r}")
        levels = tuple(float(v) for v in self.action_levels)
        object.__setattr__(self, "action_levels", levels)
        if len(levels) < 2 or any(abs(v) > 1.0 for v in levels):
            raise ValueError(f"action_levels must contain >= 2 values in [-1, 1], got {levels}")
        if self.control_dt <= 0 or self.max_episode_seconds <= 0:
            raise ValueError("control_dt and max_episode_seconds must be positive")
        if not 0.0 <= self.mixed_upright_probability <= 1.0:
            raise ValueError("mixed_upright_probability must be in [0, 1]")


_OBS_HIGH = (1.5, 2.0, 1.0, 1.0, 2.0, 1.0)


def observation_bounds(cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    high = np.asarray(_OBS_HIGH[: 6 if cfg.include_prev_action else 5], dtype=np.float32)
    return -high, high


def make_observation(x: float, x_dot: float, theta: float, theta_dot: float, prev_command: float,
                     params: PhysicsParams, cfg: EnvConfig) -> np.ndarray:
    """Normalized observation vector. Shared by the environment and the hardware interface so that a
    deployed policy sees exactly the feature scaling it was trained with."""
    obs = [x / params.rail_half_length, x_dot / params.max_support_speed, math.sin(theta), math.cos(theta),
           theta_dot / cfg.omega_scale]
    if cfg.include_prev_action:
        obs.append(prev_command)
    low, high = observation_bounds(cfg)
    return np.clip(np.asarray(obs, dtype=np.float32), low, high)


def classify_phase(theta: float, theta_dot: float, balance_run: float, failed: bool, cfg: EnvConfig) -> str:
    """Human-readable control phase shown on the HUD and stored in ``info['phase']``."""
    if failed:
        return "failed"
    if balance_run >= 0.25:
        return "balanced"
    if abs(theta) <= cfg.upright_region:
        return "stabilizing"
    return "swinging_up"


class PendulumEnv(gym.Env):
    """Swing-up / balance task on a pluggable :class:`MechanismModel`."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, env_config: EnvConfig | None = None, physics: PhysicsParams | None = None,
                 reward: RewardConfig | None = None, render_mode: str | None = None):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None, got {render_mode!r}")
        self.cfg = env_config or EnvConfig()
        self.base_physics = physics or PhysicsParams()
        self.reward_cfg = reward or RewardConfig()
        self.render_mode = render_mode

        ratio = self.cfg.control_dt / self.base_physics.dt
        if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
            raise ValueError(
                f"control_dt ({self.cfg.control_dt}) must be an integer multiple of physics.dt ({self.base_physics.dt})")
        self._substeps = int(round(ratio))
        self.max_steps = int(round(self.cfg.max_episode_seconds / self.cfg.control_dt))

        self.model: MechanismModel = create_model(self.base_physics)
        self._levels = np.asarray(self.cfg.action_levels, dtype=np.float64)

        if self.cfg.action_mode == "discrete":
            self.action_space: spaces.Space = spaces.Discrete(len(self._levels))
        else:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        low, high = observation_bounds(self.cfg)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.obs_dim = int(high.size)

        self._state: np.ndarray | None = None
        self._view = None
        self._reset_trackers()

    # ------------------------------------------------------------------ properties
    @property
    def state(self) -> np.ndarray:
        """Copy of the full internal state ``[x, x_dot, theta, theta_dot, actuator]``."""
        if self._state is None:
            raise RuntimeError("Call reset() before reading the state.")
        return self._state.copy()

    @property
    def physical_state(self) -> np.ndarray:
        """``[x, x_dot, theta, theta_dot]`` with the angle wrapped to [-pi, pi)."""
        s = self.state
        return np.array([s[X], s[XD], wrap_angle(s[TH]), s[THD]])

    @property
    def n_actions(self) -> int:
        return len(self._levels)

    @property
    def time(self) -> float:
        return self._steps * self.cfg.control_dt

    def command_for_action(self, action) -> float:
        if self.cfg.action_mode == "continuous":
            raise TypeError("command_for_action is only defined for discrete action mode")
        return float(self._levels[int(action)])

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        physics = self.base_physics
        if self.cfg.physics_randomization > 0:
            physics = physics.randomized(self.np_random, self.cfg.physics_randomization)
        self.model = create_model(physics)

        if "state" in options:
            state = np.asarray(options["state"], dtype=np.float64)
            if state.shape not in ((4,), (5,)) or not np.all(np.isfinite(state)):
                raise ValueError("options['state'] must be a finite array of 4 or 5 values [x, x_dot, theta, theta_dot(, actuator)]")
            self._state = self.model.make_state(*state)
        else:
            self._state = self._sample_start(options.get("start", self.cfg.start_mode))

        self._reset_trackers()
        self._prev_command = 0.0
        self._init_energy = self.model.energy(self._state).swing_normalized
        self._max_energy = self._init_energy
        obs = self._observation()
        info = self._info(None, {}, phase_failed=False)
        return obs, info

    def step(self, action):
        if self.cfg.action_mode == "discrete":
            if not self.action_space.contains(action if not isinstance(action, np.generic) else action.item()):
                raise ValueError(f"Invalid action {action!r}: expected an integer in [0, {self.n_actions - 1}]")
            command = float(self._levels[int(action)])
            self._action_counts[int(action)] += 1
        else:
            arr = np.asarray(action, dtype=np.float64).reshape(-1)
            if arr.size != 1 or not np.isfinite(arr).all():
                raise ValueError(f"Invalid continuous action {action!r}: expected one finite value in [-1, 1]")
            command = float(np.clip(arr[0], -1.0, 1.0))
        return self._advance(command)

    def step_command(self, command: float):
        """Step with a raw normalized command (any action mode); used by keyboard control and controllers."""
        if not math.isfinite(command):
            raise ValueError(f"command must be finite, got {command}")
        return self._advance(float(np.clip(command, -1.0, 1.0)))

    # ------------------------------------------------------------------ internals
    def _reset_trackers(self) -> None:
        self._steps = 0
        self._done = False
        self._balance_run = 0.0
        self._longest_balance = 0.0
        self._balanced_total = 0.0
        self._reached = False
        self._time_to_upright = math.nan
        self._succeeded = False
        self._return = 0.0
        self._err_sum = 0.0
        self._err_max = 0.0
        self._effort_abs = 0.0
        self._effort_sq = 0.0
        self._violation: str | None = None
        self._action_counts = [0] * len(self._levels)
        self._prev_command = 0.0
        self._max_energy = 0.0
        self._init_energy = 0.0

    def _sample_start(self, mode: str) -> np.ndarray:
        rng, c = self.np_random, self.cfg
        if mode == "mixed":
            mode = "upright" if rng.random() < c.mixed_upright_probability else "hanging"
        x0 = rng.uniform(-c.start_position_range, c.start_position_range)
        if mode == "hanging":
            theta = math.pi + rng.uniform(-c.hanging_angle_noise, c.hanging_angle_noise)
            omega = rng.uniform(-c.hanging_velocity_noise, c.hanging_velocity_noise)
        elif mode == "upright":
            theta = rng.uniform(-c.upright_angle_range, c.upright_angle_range)
            omega = rng.uniform(-c.upright_velocity_range, c.upright_velocity_range)
        elif mode == "random":
            theta = rng.uniform(-math.pi, math.pi)
            omega = rng.uniform(-1.0, 1.0)
        else:
            raise ValueError(f"Unknown start mode {mode!r}")
        return self.model.make_state(x0, 0.0, wrap_angle(theta), omega, 0.0)

    def _observation(self) -> np.ndarray:
        s, c = self._state, self.cfg
        x, xd, th, thd = float(s[X]), float(s[XD]), float(s[TH]), float(s[THD])
        if c.noise_position or c.noise_velocity or c.noise_angle or c.noise_angular_velocity:
            n = self.np_random.normal
            x += n(0, c.noise_position)
            xd += n(0, c.noise_velocity)
            th += n(0, c.noise_angle)
            thd += n(0, c.noise_angular_velocity)
        return make_observation(x, xd, th, thd, self._prev_command, self.model.params, c)

    def _advance(self, command: float):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("step() called after the episode ended; call reset() first.")
        c, p = self.cfg, self.model.params

        if c.disturbance_probability > 0 and self.np_random.random() < c.disturbance_probability:
            self._state[THD] += self.np_random.uniform(-c.disturbance_magnitude, c.disturbance_magnitude)

        violation: str | None = None
        try:
            self._state = advance(self.model, self._state, command, c.control_dt, dt=c.control_dt / self._substeps)
        except NumericalInstabilityError:
            violation = "numerical_instability"
            # the last valid state is kept (``advance`` never mutates its input)
        if violation is None:
            violation = self.model.check_limits(self._state)

        s = self._state
        x, xd, th, thd = float(s[X]), float(s[XD]), float(wrap_angle(s[TH])), float(s[THD])
        dt = c.control_dt
        self._steps += 1
        err = abs(th)
        self._err_sum += err
        self._err_max = max(self._err_max, err)
        self._effort_abs += abs(command) * dt
        self._effort_sq += command * command * dt
        self._max_energy = max(self._max_energy, self.model.energy(s).swing_normalized)

        if not self._reached and err <= c.upright_region:
            self._reached = True
            self._time_to_upright = self.time
        balanced = err <= c.balance_angle_tol and abs(thd) <= c.balance_velocity_tol
        if balanced and violation is None:
            self._balance_run += dt
            self._balanced_total += dt
            self._longest_balance = max(self._longest_balance, self._balance_run)
        else:
            self._balance_run = 0.0

        new_success = False
        if violation is None and not self._succeeded:
            if c.success_mode == "balance":
                new_success = self._balance_run >= c.hold_seconds
            else:
                new_success = err <= c.catch_angle and abs(thd) <= c.catch_velocity
            self._succeeded = new_success

        reward, terms = compute_reward(
            self.reward_cfg, theta=th, theta_dot=thd, x=x, x_dot=xd, command=command,
            prev_command=self._prev_command, x_limit=p.rail_half_length, v_max=p.max_support_speed,
            success=new_success, violation=violation is not None)
        self._return += reward
        self._prev_command = command

        terminated = False
        reason: str | None = None
        if violation is not None:
            terminated, reason, self._violation = True, violation, violation
        elif new_success and c.terminate_on_success:
            terminated, reason = True, "success"
        truncated = (not terminated) and self._steps >= self.max_steps
        if truncated:
            reason = "time_limit"
        self._done = terminated or truncated

        obs = self._observation()
        info = self._info(reason, terms, phase_failed=violation is not None)
        if self._done:
            info["episode_metrics"] = self._episode_metrics(reason)
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def _info(self, reason: str | None, terms: dict, phase_failed: bool) -> dict:
        s = self._state
        th = float(wrap_angle(s[TH]))
        return {
            "time": self.time,
            "x": float(s[X]), "x_dot": float(s[XD]), "theta": th, "theta_dot": float(s[THD]),
            "actuator": float(s[ACT]),
            "command": self._prev_command,
            "phase": classify_phase(th, float(s[THD]), self._balance_run, phase_failed, self.cfg),
            "swing_energy": self.model.energy(s).swing_normalized,
            "balanced_time": self._balance_run,
            "is_success": self._succeeded,
            "termination_reason": reason,
            "reward_terms": terms,
        }

    def _episode_metrics(self, reason: str | None) -> dict:
        n = max(1, self._steps)
        violated = self._violation is not None
        return {
            "steps": self._steps,
            "duration": self.time,
            "return": self._return,
            "termination_reason": reason,
            "success": bool(self._succeeded and not violated),
            "swing_up_success": self._reached,
            "balance_success": bool(self._longest_balance >= self.cfg.hold_seconds and not violated),
            "time_to_upright": self._time_to_upright,
            "balanced_time": self._balanced_total,
            "longest_balance": self._longest_balance,
            "mean_abs_angle_error": self._err_sum / n,
            "max_abs_angle_error": self._err_max,
            "final_abs_angle_error": abs(float(wrap_angle(self._state[TH]))),
            "effort_abs": self._effort_abs,
            "effort_sq": self._effort_sq,
            "violation": violated,
            "violation_reason": self._violation,
            "max_swing_energy": self._max_energy,
            "action_counts": list(self._action_counts),
        }

    def apply_disturbance(self, delta_theta_dot: float) -> None:
        """Instantaneous kick to the pendulum's angular velocity [rad/s] (used by the simulator's 'kick' key)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        self._state[THD] += float(delta_theta_dot)

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode is None:
            return None
        if self._state is None:
            raise RuntimeError("Call reset() before render().")
        if self._view is None:
            from ..visualization.renderer import PendulumView  # lazy: keeps pygame optional for training
            self._view = PendulumView(self.model, headless=self.render_mode == "rgb_array")
        from ..visualization.renderer import HudInfo
        info = self._info(None, {}, self._violation is not None)
        hud = HudInfo(command=self._prev_command, time=self.time, reward=self._return, phase=info["phase"],
                      mode_label="GYMNASIUM ENV")
        return self._view.draw(self._state, hud, show=self.render_mode == "human",
                               fps=self.metadata["render_fps"] if self.render_mode == "human" else 0,
                               as_array=self.render_mode == "rgb_array")

    def close(self) -> None:
        if self._view is not None:
            self._view.close()
            self._view = None


def make_env(env_config: EnvConfig | None = None, physics: PhysicsParams | None = None,
             reward: RewardConfig | None = None, render_mode: str | None = None, **env_overrides) -> PendulumEnv:
    """Convenience constructor; ``env_overrides`` replace fields of ``env_config``."""
    cfg = env_config or EnvConfig()
    if env_overrides:
        cfg = replace(cfg, **env_overrides)
    return PendulumEnv(cfg, physics, reward, render_mode)
