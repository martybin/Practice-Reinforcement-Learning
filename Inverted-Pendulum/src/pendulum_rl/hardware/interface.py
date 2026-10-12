"""Hardware-facing interface between a trained policy and a (future) physical pendulum.

IMPORTANT - this module intentionally contains **no code that can energize an actuator**.
The only concrete backend is :class:`SimulatedBackend`, which runs the software model.
Driving real hardware requires someone to write a :class:`HardwareBackend` subclass for their
specific, calibrated hardware, and :class:`ControlLoop` refuses to run unless every item of
:class:`HardwareConfig` has been explicitly confirmed by a person (see ``docs/HARDWARE.md``).

Data flow of one control cycle (what each piece is responsible for)::

    sensors -> [SensorCalibration] -> [VelocityEstimator] -> make_observation (same scaling as training)
            -> policy.act(obs)  -> action index -> command in [-1, 1]
            -> [SafetyMonitor / CommandLimiter]  -> HardwareBackend.write_command()
                      \\-> latched emergency stop on any violated limit
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..environment import EnvConfig, make_observation
from ..physics import PhysicsParams, create_model, advance


class HardwareNotReadyError(RuntimeError):
    """Raised when a safety prerequisite has not been explicitly satisfied."""


class EmergencyStop(RuntimeError):
    """Raised (and latched) when a safety limit is violated."""


# ---------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class HardwareConfig:
    """Explicit human confirmations required before a non-simulated backend may be armed.

    All default to ``False``. They are *attestations*, not auto-detected facts: set them to ``True``
    only after the corresponding real-world check has been done and documented.
    """

    enabled: bool = False
    sensors_calibrated: bool = False            # encoder offsets/scales measured, zero position verified
    actuator_limits_verified: bool = False      # force/speed limits measured, command saturation tested at low power
    independent_estop_installed: bool = False   # hardware E-stop that cuts motor power without software
    physics_params_identified: bool = False     # simulated parameters replaced by measured ones (see docs)

    def missing(self) -> list[str]:
        return [name for name in ("enabled", "sensors_calibrated", "actuator_limits_verified",
                                  "independent_estop_installed", "physics_params_identified")
                if not getattr(self, name)]


@dataclass(frozen=True)
class SafetyLimits:
    max_command: float = 0.5                    # |command| cap, fraction of full drive (start low!)
    max_command_slew: float = 20.0              # allowed |d command / dt| [1/s]
    position_soft_limit_fraction: float = 0.8   # fraction of rail travel at which the E-stop triggers
    max_support_speed: float = 1.5              # m/s
    max_pendulum_speed: float = 30.0            # rad/s
    max_sensor_age: float = 0.05                # s; older measurements trigger the E-stop
    max_cycle_jitter: float = 0.5               # allowed fractional overrun of the control period


@dataclass(frozen=True)
class SensorCalibration:
    """Maps raw sensor counts to SI units. Must come from measurement, never from defaults."""

    position_m_per_count: float = 0.0
    position_offset_counts: float = 0.0
    angle_rad_per_count: float = 0.0
    angle_offset_counts: float = 0.0   # counts at the *upright* position (theta = 0)

    def is_complete(self) -> bool:
        return self.position_m_per_count != 0.0 and self.angle_rad_per_count != 0.0

    def convert(self, position_counts: float, angle_counts: float) -> tuple[float, float]:
        if not self.is_complete():
            raise HardwareNotReadyError("SensorCalibration is incomplete: measure the position and angle scale factors.")
        return ((position_counts - self.position_offset_counts) * self.position_m_per_count,
                (angle_counts - self.angle_offset_counts) * self.angle_rad_per_count)


@dataclass
class SensorReading:
    timestamp: float            # monotonic seconds at which the measurement was taken
    position: float             # m, support position (rail centre = 0)
    angle: float                # rad, 0 = upright (angle may be unwrapped or wrapped)


# ---------------------------------------------------------------------------- estimation
class VelocityEstimator:
    """Estimates ``x_dot`` and ``theta_dot`` from position/angle samples.

    Uses a first-order filtered ("dirty") derivative with time constant ``tau``:
    ``v <- (tau v + (q - q_prev)) / (tau + dt)``. The same estimator should be used when training
    with simulated sensor noise (``env.noise_*``) so that the learned policy tolerates its lag.
    """

    def __init__(self, tau: float = 0.02):
        self.tau = tau
        self._prev: tuple[float, float, float] | None = None   # (t, x, theta_unwrapped)
        self.x_dot = 0.0
        self.theta_dot = 0.0

    def reset(self) -> None:
        self._prev = None
        self.x_dot = self.theta_dot = 0.0

    def update(self, reading: SensorReading) -> tuple[float, float]:
        if self._prev is None:
            self._prev = (reading.timestamp, reading.position, reading.angle)
            return 0.0, 0.0
        t0, x0, th0 = self._prev
        dt = reading.timestamp - t0
        if dt <= 0:
            raise EmergencyStop(f"non-monotonic sensor timestamps (dt={dt})")
        dth = math.remainder(reading.angle - th0, 2 * math.pi)       # shortest signed angle difference
        self.x_dot = (self.tau * self.x_dot + (reading.position - x0)) / (self.tau + dt)
        self.theta_dot = (self.tau * self.theta_dot + dth) / (self.tau + dt)
        self._prev = (reading.timestamp, reading.position, th0 + dth)
        return self.x_dot, self.theta_dot


# ---------------------------------------------------------------------------- safety
class SafetyMonitor:
    """Checks measurements and limits commands; once tripped the emergency stop stays latched."""

    def __init__(self, physics: PhysicsParams, limits: SafetyLimits | None = None):
        self.physics, self.limits = physics, limits or SafetyLimits()
        self.estop_reason: str | None = None
        self._last_command = 0.0

    @property
    def estopped(self) -> bool:
        return self.estop_reason is not None

    def trip(self, reason: str) -> None:
        if self.estop_reason is None:
            self.estop_reason = reason

    def reset_estop(self) -> None:
        """Release the latch. Must only be called by a person after inspecting the hardware."""
        self.estop_reason = None
        self._last_command = 0.0

    def check_measurements(self, reading: SensorReading, x_dot: float, theta_dot: float, now: float) -> None:
        lim = self.limits
        values = (reading.position, reading.angle, x_dot, theta_dot)
        if not all(math.isfinite(v) for v in values):
            self.trip("non-finite sensor value")
        elif now - reading.timestamp > lim.max_sensor_age:
            self.trip(f"stale sensor data ({now - reading.timestamp:.3f} s old)")
        elif abs(reading.position) > lim.position_soft_limit_fraction * self.physics.rail_half_length:
            self.trip(f"support position {reading.position:+.3f} m beyond soft limit")
        elif abs(x_dot) > lim.max_support_speed:
            self.trip(f"support speed {x_dot:+.2f} m/s beyond limit")
        elif abs(theta_dot) > lim.max_pendulum_speed:
            self.trip(f"pendulum speed {theta_dot:+.1f} rad/s beyond limit")

    def limit_command(self, command: float, dt: float) -> float:
        """Saturate and slew-limit ``command``; returns 0.0 whenever the E-stop is latched."""
        if self.estopped or not math.isfinite(command):
            if not self.estopped:
                self.trip("non-finite command")
            self._last_command = 0.0
            return 0.0
        lim = self.limits
        step = lim.max_command_slew * dt
        cmd = min(max(command, -lim.max_command), lim.max_command)
        cmd = min(max(cmd, self._last_command - step), self._last_command + step)
        self._last_command = cmd
        return cmd


# ---------------------------------------------------------------------------- backends
class HardwareBackend(ABC):
    """What a hardware port must provide. No implementation here drives real motors."""

    #: simulated backends set this to True; only they can run without HardwareConfig confirmations
    simulated: bool = False

    @abstractmethod
    def read_sensors(self) -> SensorReading:
        """Return the latest calibrated measurement (SI units, ``angle = 0`` upright)."""

    @abstractmethod
    def write_command(self, command: float) -> None:
        """Apply a normalized support-drive command in [-1, 1] (already limited by the SafetyMonitor)."""

    @abstractmethod
    def emergency_stop(self) -> None:
        """Remove drive power NOW. Must be idempotent and must not raise."""


class SimulatedBackend(HardwareBackend):
    """Software stand-in for the mechanism, for dry-running the full control loop. Drives nothing."""

    simulated = True

    def __init__(self, physics: PhysicsParams, initial_state=None, position_noise: float = 0.0,
                 angle_noise: float = 0.0, seed: int = 0):
        self.model = create_model(physics)
        self.state = np.array(initial_state, dtype=float) if initial_state is not None else \
            self.model.make_state(0.0, 0.0, math.pi - 0.05, 0.0, 0.0)
        self.time = 0.0
        self._cmd = 0.0
        self._stopped = False
        self._rng = np.random.default_rng(seed)
        self._noise = (position_noise, angle_noise)
        self.commands: list[float] = []

    def read_sensors(self) -> SensorReading:
        x, th = float(self.state[0]), float(self.state[2])
        return SensorReading(self.time, x + self._rng.normal(0, self._noise[0]), th + self._rng.normal(0, self._noise[1]))

    def write_command(self, command: float) -> None:
        self._cmd = 0.0 if self._stopped else float(command)
        self.commands.append(self._cmd)

    def emergency_stop(self) -> None:
        self._stopped = True
        self._cmd = 0.0

    def advance(self, dt: float) -> None:
        """Simulated time step (called by the dry-run harness, not part of the hardware contract)."""
        self.state = advance(self.model, self.state, 0.0 if self._stopped else self._cmd, dt)
        self.time += dt


# ---------------------------------------------------------------------------- policy wrapper
@dataclass
class ControlOutput:
    command: float
    observation: np.ndarray
    estop: bool
    reason: str | None


class PolicyController:
    """Turns a measurement into a safe command. Pure computation: touches no hardware."""

    def __init__(self, act: Callable[[np.ndarray], int], action_levels, physics: PhysicsParams, env_cfg: EnvConfig,
                 safety: SafetyMonitor, estimator: VelocityEstimator | None = None):
        self.act, self.levels = act, [float(v) for v in action_levels]
        self.physics, self.env_cfg, self.safety = physics, env_cfg, safety
        self.estimator = estimator or VelocityEstimator()
        self._prev_command = 0.0

    def reset(self) -> None:
        self.estimator.reset()
        self._prev_command = 0.0

    def step(self, reading: SensorReading, now: float, dt: float) -> ControlOutput:
        x_dot, theta_dot = self.estimator.update(reading)
        self.safety.check_measurements(reading, x_dot, theta_dot, now)
        theta = math.remainder(reading.angle, 2 * math.pi)
        obs = make_observation(reading.position, x_dot, theta, theta_dot, self._prev_command, self.physics, self.env_cfg)
        raw = self.levels[int(self.act(obs))] if not self.safety.estopped else 0.0
        command = self.safety.limit_command(raw, dt)
        self._prev_command = command
        return ControlOutput(command, obs, self.safety.estopped, self.safety.estop_reason)


class ControlLoop:
    """Runs sensor -> policy -> actuator cycles with explicit arming and E-stop handling."""

    def __init__(self, backend: HardwareBackend, controller: PolicyController, config: HardwareConfig | None = None,
                 control_dt: float = 0.02):
        self.backend, self.controller = backend, controller
        self.config = config or HardwareConfig()
        self.control_dt = control_dt
        self.armed = False

    def arm(self) -> None:
        """Enable command output. Real (non-simulated) backends need every HardwareConfig confirmation."""
        if not self.backend.simulated:
            missing = self.config.missing()
            if missing:
                raise HardwareNotReadyError(
                    "Refusing to arm a physical backend. Unconfirmed prerequisites: " + ", ".join(missing) +
                    ". See docs/HARDWARE.md; these must be satisfied by a person, not by software.")
        self.controller.reset()
        self.controller.safety.reset_estop()
        self.armed = True

    def disarm(self) -> None:
        self.armed = False
        self.backend.emergency_stop()

    def cycle(self) -> ControlOutput:
        """One control cycle; any exception or tripped limit ends in an emergency stop."""
        if not self.armed:
            raise HardwareNotReadyError("ControlLoop.arm() has not been called")
        try:
            reading = self.backend.read_sensors()
            out = self.controller.step(reading, now=time.monotonic() if not self.backend.simulated else reading.timestamp,
                                       dt=self.control_dt)
        except Exception as exc:            # sensor failure, non-monotonic time, policy error, ...
            self.controller.safety.trip(f"{type(exc).__name__}: {exc}")
            self.backend.emergency_stop()
            self.armed = False
            return ControlOutput(0.0, np.zeros(0, dtype=np.float32), True, self.controller.safety.estop_reason)
        if out.estop:
            self.backend.emergency_stop()
            self.armed = False
            return out
        self.backend.write_command(out.command)
        return out
