"""Hardware interface: arming rules, estimation, safety limits and emergency stop (no real hardware involved)."""

import math
import time

import numpy as np
import pytest

from pendulum_rl.environment import EnvConfig, PendulumEnv, make_observation
from pendulum_rl.hardware import (
    ControlLoop, HardwareBackend, HardwareConfig, HardwareNotReadyError, PolicyController, SafetyLimits, SafetyMonitor,
    SensorCalibration, SensorReading, SimulatedBackend, VelocityEstimator,
)
from pendulum_rl.physics import PhysicsParams

PHYS, ENV = PhysicsParams(), EnvConfig()
LEVELS = (-1.0, 0.0, 1.0)


class FakePhysicalBackend(HardwareBackend):
    """Stands in for a user-written backend; records what would reach the motor driver."""

    simulated = False

    def __init__(self, position=0.0, angle=0.1):
        self.position, self.angle = position, angle
        self.commands, self.estops = [], 0

    def read_sensors(self):
        return SensorReading(time.monotonic(), self.position, self.angle)     # real backends stamp with a monotonic clock

    def write_command(self, command):
        self.commands.append(command)

    def emergency_stop(self):
        self.estops += 1


def controller(act=lambda obs: 2, limits=None):
    safety = SafetyMonitor(PHYS, limits or SafetyLimits())
    return PolicyController(act, LEVELS, PHYS, ENV, safety)


def test_physical_backend_cannot_be_armed_without_every_confirmation():
    loop = ControlLoop(FakePhysicalBackend(), controller())
    with pytest.raises(HardwareNotReadyError, match="sensors_calibrated"):
        loop.arm()
    assert not loop.armed
    partial = HardwareConfig(enabled=True, sensors_calibrated=True, actuator_limits_verified=True,
                             independent_estop_installed=True)         # physics_params_identified missing
    with pytest.raises(HardwareNotReadyError, match="physics_params_identified"):
        ControlLoop(FakePhysicalBackend(), controller(), partial).arm()
    with pytest.raises(HardwareNotReadyError):
        loop.cycle()                                                   # never runs unarmed


def test_all_confirmations_allow_arming_and_commands_are_limited():
    cfg = HardwareConfig(True, True, True, True, True)
    backend = FakePhysicalBackend()
    loop = ControlLoop(backend, controller(lambda o: 2, SafetyLimits(max_command=0.3)), cfg)
    loop.arm()
    for _ in range(20):
        loop.cycle()
    assert backend.commands and max(backend.commands) <= 0.3 + 1e-12 and backend.estops == 0


def test_simulated_backend_needs_no_hardware_attestation():
    loop = ControlLoop(SimulatedBackend(PHYS), controller())
    loop.arm()
    out = loop.cycle()
    assert not out.estop


def test_slew_rate_limit():
    safety = SafetyMonitor(PHYS, SafetyLimits(max_command=1.0, max_command_slew=10.0))
    assert safety.limit_command(1.0, dt=0.02) == pytest.approx(0.2)
    assert safety.limit_command(1.0, dt=0.02) == pytest.approx(0.4)
    assert safety.limit_command(-1.0, dt=0.02) == pytest.approx(0.2)


@pytest.mark.parametrize("position, reason", [
    (0.55, "soft limit"),                     # beyond 0.8 * rail
    (float("nan"), "non-finite"),
])
def test_emergency_stop_on_unsafe_measurement(position, reason):
    backend = FakePhysicalBackend(position=position)
    loop = ControlLoop(backend, controller(), HardwareConfig(True, True, True, True, True))
    loop.arm()
    out = loop.cycle()
    assert out.estop and reason in out.reason and out.command == 0.0
    assert backend.estops == 1 and not loop.armed and backend.commands == []


def test_stale_sensor_data_triggers_estop():
    safety = SafetyMonitor(PHYS)
    safety.check_measurements(SensorReading(0.0, 0.0, 0.0), 0.0, 0.0, now=1.0)
    assert safety.estopped and "stale" in safety.estop_reason


def test_speed_limits_trigger_estop():
    safety = SafetyMonitor(PHYS)
    safety.check_measurements(SensorReading(1.0, 0.0, 0.0), x_dot=5.0, theta_dot=0.0, now=1.0)
    assert "speed" in safety.estop_reason
    safety2 = SafetyMonitor(PHYS)
    safety2.check_measurements(SensorReading(1.0, 0.0, 0.0), x_dot=0.0, theta_dot=100.0, now=1.0)
    assert "pendulum speed" in safety2.estop_reason


def test_estop_latches_until_explicit_reset():
    safety = SafetyMonitor(PHYS)
    safety.trip("test")
    assert safety.limit_command(1.0, 0.02) == 0.0 and safety.estopped
    safety.check_measurements(SensorReading(1.0, 0.0, 0.0), 0.0, 0.0, now=1.0)   # healthy data does not clear it
    assert safety.estopped
    safety.reset_estop()
    assert not safety.estopped


def test_policy_exception_results_in_estop_not_crash():
    def broken(obs):
        raise RuntimeError("inference failed")

    backend = FakePhysicalBackend()
    loop = ControlLoop(backend, controller(broken), HardwareConfig(True, True, True, True, True))
    loop.arm()
    out = loop.cycle()
    assert out.estop and "inference failed" in out.reason and backend.estops == 1 and backend.commands == []


def test_non_finite_command_is_rejected():
    safety = SafetyMonitor(PHYS)
    assert safety.limit_command(float("nan"), 0.02) == 0.0 and safety.estopped


def test_velocity_estimator_tracks_ramps_and_angle_wraparound():
    est = VelocityEstimator(tau=0.02)
    for k in range(100):
        est.update(SensorReading(k * 0.02, 0.5 * k * 0.02, 0.0))
    assert est.x_dot == pytest.approx(0.5, rel=1e-2)
    est.reset()
    angle = math.pi - 0.5
    for k in range(100):                   # constant +4 rad/s through the +-pi discontinuity
        a = math.remainder(angle + 4.0 * k * 0.02, 2 * math.pi)
        est.update(SensorReading(k * 0.02, 0.0, a))
    assert est.theta_dot == pytest.approx(4.0, rel=2e-2)
    with pytest.raises(Exception):
        est.update(SensorReading(0.5, 0.0, 0.0))        # time going backwards


def test_calibration_must_be_measured():
    cal = SensorCalibration()
    assert not cal.is_complete()
    with pytest.raises(HardwareNotReadyError):
        cal.convert(10, 10)
    x, th = SensorCalibration(1e-4, 100, 2e-3, 50).convert(1100, 150)
    assert x == pytest.approx(0.1) and th == pytest.approx(0.2)


def test_observation_matches_training_environment():
    """A deployed policy must see exactly the features it was trained on."""
    env = PendulumEnv(ENV, PHYS)
    env.reset(seed=0, options={"state": [0.2, 0.3, 0.4, -1.2]})
    env.step_command(0.0)
    x, xd, th, thd = env.physical_state
    expected = env._observation()
    assert np.allclose(make_observation(x, xd, th, thd, 0.0, PHYS, ENV), expected, atol=1e-6)


def test_dry_run_in_simulation_stays_within_command_limits():
    backend = SimulatedBackend(PHYS, seed=0)
    loop = ControlLoop(backend, controller(lambda obs: 1 if obs[4] > 0 else 0, SafetyLimits(max_command=0.5)))
    loop.arm()
    for _ in range(200):
        out = loop.cycle()
        if out.estop:
            break
        backend.advance(0.02)
    assert backend.commands and max(abs(c) for c in backend.commands) <= 0.5 + 1e-12
