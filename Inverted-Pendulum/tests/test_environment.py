"""Gymnasium API compliance, task logic, termination conditions and reproducibility."""

import math
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pendulum_rl.control import EnergyShapingLQRController
from pendulum_rl.environment import (
    ENV_ID, EnvConfig, PendulumEnv, RewardConfig, classify_phase, compute_reward, make_env, make_observation,
)
from pendulum_rl.physics import PhysicsParams


def test_gymnasium_env_checker_discrete_and_continuous():
    check_env(PendulumEnv(), skip_render_check=True)
    check_env(PendulumEnv(EnvConfig(action_mode="continuous")), skip_render_check=True)


def test_registered_with_gymnasium():
    import gymnasium as gym
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


def test_observation_dimension_and_space_validity():
    env = PendulumEnv()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (6,) and obs.dtype == np.float32 and env.obs_dim == 6
    assert PendulumEnv(EnvConfig(include_prev_action=False)).observation_space.shape == (5,)
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(r)
        if term or trunc:
            env.reset()
    # sin/cos columns form a unit vector
    obs, _ = env.reset(seed=2)
    assert obs[2] ** 2 + obs[3] ** 2 == pytest.approx(1.0, abs=1e-5)


def test_state_dimensions():
    env = PendulumEnv()
    env.reset(seed=0)
    assert env.state.shape == (5,) and env.physical_state.shape == (4,)


def test_action_validation():
    env = PendulumEnv()
    env.reset(seed=0)
    for bad in (3, -1, 1.5, "left", None):
        with pytest.raises((ValueError, TypeError)):
            env.step(bad)
    cont = PendulumEnv(EnvConfig(action_mode="continuous"))
    cont.reset(seed=0)
    for bad in (np.array([np.nan]), np.array([0.1, 0.2])):
        with pytest.raises(ValueError):
            cont.step(bad)
    with pytest.raises(ValueError):
        cont.step_command(float("inf"))
    _, _, _, _, info = cont.step(np.array([5.0]))      # out-of-range finite actions are clipped, not rejected
    assert info["command"] == 1.0


def test_step_before_reset_and_after_end_raise():
    env = PendulumEnv()
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    env.reset(seed=0, options={"state": [0.59, 3.0, 0.0, 0.0]})
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(2)
        done = term or trunc
    with pytest.raises(RuntimeError):
        env.step(0)


def test_reset_is_reproducible_for_same_seed_and_differs_across_seeds():
    env = PendulumEnv()
    a, _ = env.reset(seed=42)
    sa = env.state
    b, _ = env.reset(seed=42)
    sb = env.state
    c, _ = env.reset(seed=43)
    assert np.array_equal(a, b) and np.array_equal(sa, sb)
    assert not np.array_equal(sa, env.state)


def test_trajectories_reproducible():
    def rollout(seed):
        env = PendulumEnv(EnvConfig(start_mode="hanging", physics_randomization=0.1, disturbance_probability=0.05))
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(100):
            obs, r, term, trunc, _ = env.step(int(rng.integers(3)))
            out.append((obs.copy(), r))
            if term or trunc:
                break
        return out

    for (o1, r1), (o2, r2) in zip(rollout(5), rollout(5)):
        assert np.array_equal(o1, o2) and r1 == r2


def test_start_modes():
    for mode, check in (("hanging", lambda th: abs(abs(th) - math.pi) <= 0.1 + 1e-9),
                        ("upright", lambda th: abs(th) <= 0.25 + 1e-9)):
        env = PendulumEnv(EnvConfig(start_mode=mode))
        for seed in range(30):
            env.reset(seed=seed)
            assert check(env.physical_state[2])
    env = PendulumEnv(EnvConfig(start_mode="mixed"))
    kinds = set()
    for seed in range(60):
        env.reset(seed=seed)
        kinds.add("up" if abs(env.physical_state[2]) < 1.0 else "down")
    assert kinds == {"up", "down"}


def test_time_limit_truncates_not_terminates():
    env = PendulumEnv(EnvConfig(start_mode="hanging", max_episode_seconds=1.0))
    env.reset(seed=0)
    for k in range(50):
        _, _, term, trunc, info = env.step(1)
    assert trunc and not term and info["termination_reason"] == "time_limit"
    assert info["episode_metrics"]["steps"] == 50


def test_rail_violation_terminates_with_penalty():
    env = PendulumEnv(EnvConfig(start_mode="hanging"))
    env.reset(seed=0)
    term = False
    for _ in range(200):
        _, r, term, trunc, info = env.step(2)       # push right until the rail end
        if term:
            break
    assert term and info["termination_reason"] == "rail_limit"
    assert info["phase"] == "failed" and info["episode_metrics"]["violation"] is True
    assert r < -10                                     # violation penalty dominates the step reward


def test_success_termination_when_balance_held():
    cfg = EnvConfig(start_mode="upright", upright_angle_range=0.0, upright_velocity_range=0.0, start_position_range=0.0,
                    hold_seconds=1.0, action_mode="continuous")
    env = PendulumEnv(cfg)
    env.reset(seed=0)
    ctl = EnergyShapingLQRController(env.model)
    for _ in range(200):
        obs, r, term, trunc, info = env.step(np.array([ctl.command(env.physical_state)], dtype=np.float32))
        if term or trunc:
            break
    assert term and info["termination_reason"] == "success" and info["is_success"]
    assert info["episode_metrics"]["success"] and info["episode_metrics"]["balance_success"]
    assert r > 40                                      # success bonus


def test_success_mode_reach_and_no_termination_option():
    cfg = EnvConfig(success_mode="reach", terminate_on_success=False)
    env = PendulumEnv(cfg)
    env.reset(seed=0, options={"state": [0, 0, 0.1, 0.5]})
    _, r, term, trunc, info = env.step(1)
    assert info["is_success"] and not term           # success flagged, episode continues


def test_phase_classification():
    cfg = EnvConfig()
    assert classify_phase(3.0, 0.0, 0.0, False, cfg) == "swinging_up"
    assert classify_phase(0.2, 1.0, 0.0, False, cfg) == "stabilizing"
    assert classify_phase(0.01, 0.0, 1.0, False, cfg) == "balanced"
    assert classify_phase(0.01, 0.0, 1.0, True, cfg) == "failed"


def test_reward_prefers_upright_and_penalizes_speed_effort_and_rail():
    cfg, kw = RewardConfig(), dict(x=0.0, x_dot=0.0, command=0.0, prev_command=0.0, x_limit=0.6, v_max=3.0)
    up = compute_reward(cfg, theta=0.0, theta_dot=0.0, **kw)[0]
    tilted = compute_reward(cfg, theta=0.5, theta_dot=0.0, **kw)[0]
    down = compute_reward(cfg, theta=math.pi, theta_dot=0.0, **kw)[0]
    assert up > tilted > down
    fast_top = compute_reward(cfg, theta=0.0, theta_dot=10.0, **kw)[0]
    assert fast_top < up - 0.5                       # lingering at the top while spinning is not rewarded
    assert compute_reward(cfg, theta=0.0, theta_dot=0.0, **{**kw, "command": 1.0})[0] < up
    assert compute_reward(cfg, theta=0.0, theta_dot=0.0, **{**kw, "x": 0.58})[0] < up
    # moving the support is never rewarded by itself
    still = compute_reward(cfg, theta=math.pi, theta_dot=0.0, **kw)[0]
    moving = compute_reward(cfg, theta=math.pi, theta_dot=0.0, **{**kw, "x_dot": 2.0, "command": 1.0})[0]
    assert moving < still
    _, terms = compute_reward(cfg, theta=0.0, theta_dot=0.0, violation=True, **kw)
    assert terms["violation"] == -cfg.violation_penalty


def test_physics_randomization_changes_model_each_reset():
    env = PendulumEnv(EnvConfig(physics_randomization=0.2))
    env.reset(seed=1)
    m1 = env.model.params.pendulum_mass
    env.reset(seed=2)
    assert env.model.params.pendulum_mass != m1
    env.reset(seed=1)
    assert env.model.params.pendulum_mass == m1


def test_disturbance_changes_angular_velocity():
    env = PendulumEnv(EnvConfig(disturbance_probability=1.0, disturbance_magnitude=5.0, start_mode="hanging"))
    env.reset(seed=0)
    env.step(1)
    assert abs(env.physical_state[3]) > 0.5


def test_info_contents_and_episode_metrics():
    env = PendulumEnv(EnvConfig(max_episode_seconds=0.2))
    _, info = env.reset(seed=0)
    for key in ("time", "x", "theta", "theta_dot", "phase", "swing_energy", "termination_reason"):
        assert key in info
    for _ in range(10):
        _, _, term, trunc, info = env.step(1)
    m = info["episode_metrics"]
    for key in ("return", "success", "swing_up_success", "balance_success", "time_to_upright", "mean_abs_angle_error",
                "max_abs_angle_error", "effort_abs", "violation", "action_counts"):
        assert key in m
    assert m["action_counts"] == [0, 10, 0]


def test_control_dt_must_be_multiple_of_physics_dt():
    with pytest.raises(ValueError, match="multiple"):
        PendulumEnv(EnvConfig(control_dt=0.02), PhysicsParams(dt=0.0075))
    PendulumEnv(EnvConfig(control_dt=0.02), PhysicsParams(dt=0.01))   # fine


def test_invalid_env_config():
    with pytest.raises(ValueError):
        EnvConfig(start_mode="sideways")
    with pytest.raises(ValueError):
        EnvConfig(action_levels=(2.0, 0.0))
    with pytest.raises(ValueError):
        make_env(render_mode="bogus")


def test_make_observation_matches_env():
    env = PendulumEnv()
    obs, _ = env.reset(seed=3)
    s = env.physical_state
    ref = make_observation(s[0], s[1], s[2], s[3], 0.0, env.model.params, env.cfg)
    assert np.allclose(obs, ref, atol=1e-6)


def test_baseline_swings_up_and_balances_from_hanging():
    """The task is physically feasible: the model-based baseline solves it (also guards the default parameters)."""
    env = PendulumEnv(EnvConfig(start_mode="hanging", action_mode="continuous", terminate_on_success=False,
                                max_episode_seconds=12.0))
    wins = 0
    for seed in range(8):
        env.reset(seed=seed)
        ctl = EnergyShapingLQRController(env.model)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(np.array([ctl.command(env.physical_state)], dtype=np.float32))
            done = term or trunc
        wins += info["episode_metrics"]["balance_success"]
    assert wins >= 7


def test_rendering_rgb_array():
    env = PendulumEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    assert frame.std() > 5                           # not a blank frame
    env.close()
