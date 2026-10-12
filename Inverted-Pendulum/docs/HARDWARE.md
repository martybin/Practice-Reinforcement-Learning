# Moving to real hardware - interface and prerequisites

> **Safety first.** Nothing in this repository drives a physical actuator, and nothing may be changed
> to do so automatically. A swing-up pendulum on a rail is a fast, energetic machine: a bad command can
> slam the carriage into the end stops, whip the pendulum, or pinch fingers. Everything below must be
> done deliberately by a person, at low power first, with an independent emergency stop in reach.

## What the code provides (`src/pendulum_rl/hardware/interface.py`)

| Concern | Piece | Notes |
|---|---|---|
| Reading sensors | `HardwareBackend.read_sensors() -> SensorReading` | Abstract. Returns SI units, `angle = 0` upright, with a monotonic timestamp. Raw counts are converted by a **measured** `SensorCalibration`; an incomplete calibration raises. |
| Estimating angular velocity | `VelocityEstimator` | First-order filtered derivative with proper angle unwrapping across the +-pi seam. Train with the matching sensor noise / lag (`env.noise_*`) if you rely on it. |
| Building the observation | `environment.make_observation` | The *same function* the training environment uses, so feature scaling cannot drift between simulation and deployment. |
| Running inference | `PolicyController(act=...)` | Any callable `obs -> action index`, e.g. `lambda o: agent.act(o, greedy=True)`. Pure computation, no I/O. |
| Action -> actuator command | `action_levels[idx]` | Normalized command in [-1, 1]; mapping to volts/PWM/current is the backend's job and must be calibrated. |
| Enforcing limits | `SafetyMonitor` + `SafetyLimits` | Command saturation (`max_command`, default 0.5), slew-rate limit, soft position limit (80 % of travel), support / pendulum speed limits, sensor staleness. |
| Emergency stop | `SafetyMonitor.trip`, `HardwareBackend.emergency_stop()` | Any violated limit, any exception (sensor, policy, timing) latches the stop: command forced to 0, backend told to remove drive power, loop disarmed. Released only by `reset_estop()`, which a person must call after inspecting the rig. |
| Arming | `ControlLoop.arm()` + `HardwareConfig` | A non-simulated backend cannot be armed unless *every* attestation flag in `HardwareConfig` is `True` (see checklist). `SimulatedBackend` runs the identical loop in software for dry runs. |

**Software e-stop is not an e-stop.** The code above reduces risk; it does not replace a hardware stop
that cuts motor power without software involvement (see checklist item 4).

## Minimal integration sketch (run it in simulation first)

```python
from pendulum_rl.agents import DQNAgent
from pendulum_rl.environment import EnvConfig
from pendulum_rl.hardware import (ControlLoop, HardwareConfig, PolicyController, SafetyLimits, SafetyMonitor,
                                  SimulatedBackend)
from pendulum_rl.physics import PhysicsParams

agent, meta = DQNAgent.from_checkpoint("checkpoints/combined/best.pt", device="cpu")
physics, env_cfg = PhysicsParams(**meta["physics"]), EnvConfig(**meta["base_env"])
safety = SafetyMonitor(physics, SafetyLimits(max_command=0.3))
controller = PolicyController(lambda obs: agent.act(obs, greedy=True), env_cfg.action_levels, physics, env_cfg, safety)

backend = SimulatedBackend(physics)            # <- replace by YOUR HardwareBackend subclass only after the checklist
loop = ControlLoop(backend, controller, HardwareConfig(), env_cfg.control_dt)
loop.arm()
for _ in range(500):
    out = loop.cycle()
    if out.estop:
        print("E-STOP:", out.reason); break
    backend.advance(env_cfg.control_dt)        # simulation only
```

A real backend subclasses `HardwareBackend`, implements `read_sensors`, `write_command` and an
idempotent, non-raising `emergency_stop`, and keeps `simulated = False`. Provide it from your own code
or repository after the checklist is satisfied; this project deliberately ships none.

## Checklist before any hardware run

Each item corresponds to a `HardwareConfig` flag you may only set to `True` after doing and recording it.

1. **`sensors_calibrated`** - encoder / potentiometer scale and offset measured; the zero angle is the
   true vertical (use a plumb line or level); the rail centre is `x = 0`; direction signs match the
   conventions here (`theta > 0`: tip toward +x; positive command pushes the carriage toward +x).
   Verify by moving the carriage and pendulum by hand and watching the readings.
2. **`actuator_limits_verified`** - peak force at full command, speed limit and the command-to-force
   curve measured; driver current limit set; end-stop behaviour tested at **very low power**; the
   software `max_command` starts well below the force you simulated (e.g. 0.2-0.3) and is raised in steps.
3. **`physics_params_identified`** - the simulated parameters below replaced by measured ones, the
   policy retrained (or at least re-evaluated with `--randomize`) on them.
4. **`independent_estop_installed`** - a physical switch or contactor that removes motor power without
   software, within reach, tested; mechanical end stops / bumpers fitted with margin beyond the
   software travel limit; the pendulum cannot reach people.
5. **`enabled`** - a person decided to run it, with the supervision plan below.

Run supervision: first dry-run the whole loop with `SimulatedBackend`; then run on hardware with the
**pendulum disconnected**, then with a small `max_command`, hand on the e-stop; log everything.

## Parameters to measure or calibrate (sim-to-real gap)

| Simulated parameter (`physics:` in `configs/default.yaml`) | How to obtain it |
|---|---|
| `support_mass` (+ reflected rotor inertia `J_m / r^2`) | Weigh the carriage; add motor/pulley rotor inertia divided by pulley radius squared. Identify the total from a force-step response if unsure. |
| `pendulum_mass`, `pendulum_length`, `com_fraction`, `inertia_com` | Weigh; measure pivot-to-tip; find the COM by balancing; get `inertia_about_pivot` from the small-oscillation period of the hanging pendulum: `J = m g l T^2 / (4 pi^2)`. |
| `support_viscous_friction`, `support_coulomb_friction` | Slide tests / constant-velocity force at several speeds on the rails and bearings. |
| `pivot_viscous_friction`, `pivot_coulomb_friction` | Log decay of free pendulum swings (exponential envelope = viscous, linear = Coulomb). |
| `max_force`, `motor_free_speed`, `actuator_time_constant` | Stall/low-speed force vs command; no-load speed at full command; step response of force or acceleration. Include belt stretch and backlash if present. |
| `rail_half_length`, `max_support_speed` | Measure usable travel with the bumpers; set software limits inside it. |
| control period & latency | Measure end-to-end sensing + inference + actuation latency; set `env.control_dt` accordingly and train with the observed delay (the current model has only the actuator lag). |
| sensor noise / resolution | Record at rest; set `env.noise_*` and the estimator `tau` to match, retrain. |

Known simplifications that matter on hardware: rigid pendulum and belt, no backlash, no sensor
quantization, no timing jitter, Coulomb friction smoothed with tanh, a 1-D (planar) pendulum,
a perfectly horizontal rail, and a first-order actuator lag instead of a full motor/driver model.
Quantify each before trusting a policy trained on this simulator.
