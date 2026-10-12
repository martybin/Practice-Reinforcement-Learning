"""Physics invariants: energy balance, equilibria, small-oscillation frequency, integrator agreement, safeguards."""

import math

import numpy as np
import pytest

from pendulum_rl.physics import (
    ACT, TH, THD, X, XD, MechanismModel, NumericalInstabilityError, PhysicsParams, TranslatingSupportModel, advance,
    create_model, wrap_angle,
)
from pendulum_rl.physics.dynamics import MODEL_REGISTRY

FRICTIONLESS = dict(support_viscous_friction=0.0, support_coulomb_friction=0.0, pivot_viscous_friction=0.0,
                    pivot_coulomb_friction=0.0)


def frictionless(**kw):
    return create_model(PhysicsParams(**FRICTIONLESS, **kw))


def test_state_dimension_and_finite_derivatives():
    model = create_model(PhysicsParams())
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = np.array([rng.uniform(-0.6, 0.6), rng.normal(), rng.uniform(-4, 4), rng.normal() * 5, rng.uniform(-1, 1)])
        d = model.derivatives(s, rng.uniform(-1, 1))
        assert d.shape == (5,) and np.all(np.isfinite(d))


def test_mass_matrix_positive_definite_for_all_angles():
    p = PhysicsParams()
    ml, J, M11 = p.pendulum_mass * p.com_distance, p.inertia_about_pivot, p.support_mass + p.pendulum_mass
    for th in np.linspace(-math.pi, math.pi, 361):
        assert M11 * J - (ml * math.cos(th)) ** 2 > 0


def test_equilibria_have_zero_acceleration():
    model = frictionless(actuator_time_constant=0.0, motor_free_speed=None)
    for theta in (0.0, math.pi):
        d = model.derivatives(model.make_state(0.0, 0.0, theta, 0.0, 0.0), 0.0)
        assert np.allclose(d, 0.0, atol=1e-12)


def test_hanging_is_stable_and_upright_is_unstable():
    model = create_model(PhysicsParams())
    near_down = model.make_state(0, 0, math.pi - 0.05, 0, 0)
    near_up = model.make_state(0, 0, 0.05, 0, 0)
    down_amp = up_amp = 0.0
    for _ in range(100):  # 1 s
        near_down = advance(model, near_down, 0.0, 0.01)
        near_up = advance(model, near_up, 0.0, 0.01)
        down_amp = max(down_amp, abs(wrap_angle(near_down[TH] - math.pi)))
        up_amp = max(up_amp, abs(near_up[TH]))
    assert down_amp < 0.06          # stays within (roughly) the initial amplitude
    assert up_amp > 0.5             # falls away from upright


def test_upright_linearization_has_one_unstable_pole_near_theory():
    A, B = create_model(PhysicsParams()).linearize_upright()
    eig = np.linalg.eigvals(A)
    unstable = eig[eig.real > 1e-6]
    assert len(unstable) == 1
    p = PhysicsParams()
    # for a heavy support the unstable pole approaches sqrt(m g l / J)
    heavy = np.linalg.eigvals(create_model(PhysicsParams(support_mass=1e6, **{k: 0 for k in FRICTIONLESS})).linearize_upright()[0])
    assert max(heavy.real) == pytest.approx(p.small_oscillation_frequency, rel=1e-3)
    assert B.shape == (4, 1)


def test_small_oscillation_period_matches_theory():
    p = PhysicsParams(support_mass=1e6, **FRICTIONLESS)
    model = create_model(p)
    s = model.make_state(0, 0, math.pi - 0.02, 0, 0)
    dt, t, crossings, prev = 0.001, 0.0, [], None
    for _ in range(4000):
        s = advance(model, s, 0.0, dt, dt=dt)
        t += dt
        sign = np.sign(math.sin(s[TH]))
        if prev is not None and sign != prev:
            crossings.append(t)
        prev = sign
    period = 2 * np.mean(np.diff(crossings))
    assert period == pytest.approx(2 * math.pi / p.small_oscillation_frequency, rel=5e-3)


def test_energy_conserved_without_friction_or_force():
    model = frictionless(actuator_time_constant=0.0, motor_free_speed=None)
    s = model.make_state(0.1, 0.4, 2.3, 1.0, 0.0)
    e0 = model.energy(s).total
    for _ in range(1000):
        s = advance(model, s, 0.0, 0.02)
    assert model.energy(s).total == pytest.approx(e0, abs=1e-5)


def test_power_balance_dE_dt_equals_input_minus_dissipation():
    p = PhysicsParams(support_coulomb_friction=0.0, pivot_coulomb_friction=0.0)
    model = create_model(p)
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = np.array([0.0, rng.normal(), rng.uniform(-3, 3), 3 * rng.normal(), rng.uniform(-1, 1)])
        cmd = rng.uniform(-1, 1)
        f = model.derivatives(s, cmd)
        h = 1e-7
        dE = (model.energy(s + h * f).total - model.energy(s - h * f).total) / (2 * h)
        power = (model.force_from_command(s, cmd) - p.support_viscous_friction * s[XD]) * s[XD] \
            - p.pivot_viscous_friction * s[THD] ** 2
        assert dE == pytest.approx(power, abs=1e-6)


def test_friction_only_dissipates_energy():
    model = create_model(PhysicsParams())
    s = model.make_state(0.0, 0.3, 2.0, 2.0, 0.0)
    energies = [model.energy(s).total]
    for _ in range(300):
        s = advance(model, s, 0.0, 0.02)
        energies.append(model.energy(s).total)
    assert all(b <= a + 1e-9 for a, b in zip(energies, energies[1:]))


def test_rk4_agrees_with_adaptive_reference():
    model = create_model(PhysicsParams())
    s0 = model.make_state(0.1, 0.5, 1.0, 2.0, 0.3)
    a = advance(model, s0, 0.7, 1.0, method="rk4")
    b = advance(model, s0, 0.7, 1.0, method="scipy")
    assert np.max(np.abs(a - b)) < 1e-4


def test_actuator_force_respects_limits_and_direction():
    p = PhysicsParams()
    model = create_model(p)
    for xd in (-3.0, 0.0, 3.0):
        for a in (-1.0, 0.0, 1.0):
            f = model.force_from_command(model.make_state(0, xd, 1.0, 0.0, a), 1.0)
            assert abs(f) <= p.max_force + 1e-12
    assert model.force_from_command(model.make_state(0, 0, 1.0, 0.0, 1.0), 1.0) > 0
    assert model.force_from_command(model.make_state(0, 0, 1.0, 0.0, -1.0), -1.0) < 0


def test_positive_command_accelerates_support_positively():
    model = create_model(PhysicsParams())
    s = model.make_state(0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in range(10):
        s = advance(model, s, 1.0, 0.02)
    assert s[XD] > 0 and s[X] > 0


def test_actuator_first_order_lag_time_constant():
    p = PhysicsParams(actuator_time_constant=0.05)
    model = create_model(p)
    s = model.make_state(0, 0, math.pi, 0, 0.0)
    s = advance(model, s, 1.0, 0.05, dt=0.001)
    assert s[ACT] == pytest.approx(1 - math.exp(-1), abs=1e-3)


def test_instant_actuator_when_time_constant_zero():
    model = create_model(PhysicsParams(actuator_time_constant=0.0))
    s = advance(model, model.make_state(0, 0, math.pi, 0, 0), 0.6, 0.02)
    assert s[ACT] == pytest.approx(0.6)


def test_angle_wrapping_convention():
    assert wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_angle(-3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(0.3 + 2 * math.pi) == pytest.approx(0.3)
    model = create_model(PhysicsParams())
    s = model.make_state(0, 0, 3.0, 5.0, 0)   # rotating through the bottom: angle must stay wrapped
    for _ in range(100):
        s = advance(model, s, 0.0, 0.02)
        assert -math.pi <= s[TH] < math.pi


def test_limit_detection():
    p = PhysicsParams()
    model = create_model(p)
    ok = model.make_state(0.0, 0.0, 1.0, 0.0, 0.0)
    assert model.check_limits(ok) is None
    assert model.check_limits(model.make_state(p.rail_half_length + 0.01, 0, 1, 0, 0)) == "rail_limit"
    assert model.check_limits(model.make_state(0, p.max_support_speed + 1, 1, 0, 0)) == "support_overspeed"
    assert model.check_limits(model.make_state(0, 0, 1, p.max_pendulum_speed + 1, 0)) == "pendulum_overspeed"
    assert model.check_limits(model.make_state(np.nan, 0, 1, 0, 0)) == "non_finite_state"


def test_nan_state_raises_instead_of_propagating():
    model = create_model(PhysicsParams())
    with pytest.raises(NumericalInstabilityError):
        advance(model, model.make_state(0, 0, np.nan, 0, 0), 0.0, 0.02)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        PhysicsParams(pendulum_mass=-1.0)
    with pytest.raises(ValueError):
        PhysicsParams(dt=0.0)
    with pytest.raises(ValueError):
        PhysicsParams(integrator="euler")
    with pytest.raises(ValueError):
        PhysicsParams(com_fraction=1.5)


def test_parameter_randomization_is_bounded_and_reproducible():
    p = PhysicsParams()
    a = p.randomized(np.random.default_rng(3), 0.1)
    b = p.randomized(np.random.default_rng(3), 0.1)
    assert a == b
    assert 0.9 * p.pendulum_mass <= a.pendulum_mass <= 1.1 * p.pendulum_mass
    assert a.rail_half_length == p.rail_half_length and a.dt == p.dt
    assert p.randomized(np.random.default_rng(0), 0.0) is p


def test_swing_energy_normalization():
    model = create_model(PhysicsParams())
    assert model.energy(model.make_state(0, 0, math.pi, 0, 0)).swing_normalized == pytest.approx(0.0, abs=1e-12)
    assert model.energy(model.make_state(0, 0, 0.0, 0, 0)).swing_normalized == pytest.approx(1.0, abs=1e-12)


def test_unknown_mechanism_is_reported_and_models_are_replaceable():
    with pytest.raises(ValueError, match="Unknown mechanism"):
        create_model(PhysicsParams(mechanism="tilting_beam"))

    class Dummy(TranslatingSupportModel):          # a drop-in replacement registered under another name
        geometry = "dummy"

    MODEL_REGISTRY["dummy"] = Dummy
    try:
        m = create_model(PhysicsParams(mechanism="dummy"))
        assert isinstance(m, MechanismModel) and m.geometry == "dummy"
    finally:
        MODEL_REGISTRY.pop("dummy")
