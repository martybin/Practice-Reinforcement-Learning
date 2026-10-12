"""Renderer and simulator behaviour (headless)."""

import math
from dataclasses import replace

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from pendulum_rl.environment import EnvConfig, RewardConfig  # noqa: E402
from pendulum_rl.physics import PhysicsParams, create_model  # noqa: E402
from pendulum_rl.visualization.pygame_simulator import SimulatorApp  # noqa: E402
from pendulum_rl.visualization.renderer import HudInfo, PendulumView  # noqa: E402
from pendulum_rl.visualization.settings import SimulationConfig  # noqa: E402


def make_app(tmp_path, **sim_kw):
    sim = SimulationConfig(screenshot_dir=str(tmp_path), window_width=1000, window_height=620, **sim_kw)
    return SimulatorApp(PhysicsParams(), EnvConfig(), RewardConfig(), sim, headless=True)


def key(app, k, up=False):
    app.handle_event(pygame.event.Event(pygame.KEYUP if up else pygame.KEYDOWN, key=k))


def test_renderer_draws_all_phases_and_is_not_blank():
    model = create_model(PhysicsParams())
    view = PendulumView(model, (1000, 620), headless=True)
    frames = []
    for phase, theta in (("swinging_up", 2.5), ("stabilizing", 0.2), ("balanced", 0.01), ("failed", 1.0)):
        s = model.make_state(0.1, 0.0, theta, 0.0, 0.0)
        frames.append(view.draw(s, HudInfo(phase=phase, mode_label="TEST", command=0.5, show_help=True,
                                           controls=[("K", "key")], toast=("hello", 1.0)), as_array=True))
    assert frames[0].shape == (620, 1000, 3) and frames[0].dtype == np.uint8
    assert all(f.std() > 8 for f in frames)
    assert not np.array_equal(frames[0], frames[1])      # pendulum position visibly changes the frame
    view.close()


def test_renderer_refuses_unknown_mechanism_geometry():
    class Tilting(type(create_model(PhysicsParams()))):
        geometry = "tilting_support"

    with pytest.raises(NotImplementedError, match="tilting"):
        PendulumView(Tilting(PhysicsParams()), headless=True)


def test_trail_collects_points_and_resets():
    model = create_model(PhysicsParams())
    view = PendulumView(model, (800, 500), headless=True)
    for k in range(5):
        view.draw(model.make_state(0.0, 0, 3.0 - 0.1 * k, 0, 0))
    assert len(view._trail) == 5
    view.reset_trail()
    assert len(view._trail) == 0


def test_trail_is_limited_to_trail_seconds_of_simulation_time():
    model = create_model(PhysicsParams())
    view = PendulumView(model, (800, 500), headless=True, trail_seconds=0.5)
    for k in range(100):                                   # 2 s of simulation time at 50 Hz
        view.draw(model.make_state(0.0, 0, 3.0 - 0.01 * k, 0, 0), HudInfo(time=k * 0.02))
    assert 24 <= len(view._trail) <= 27                    # ~0.5 s / 0.02 s
    view.draw(model.make_state(0.0, 0, 3.0, 0, 0), HudInfo(time=0.0))   # time went backwards: reset
    assert len(view._trail) == 1


def test_manual_keyboard_control_moves_support_and_space_zeroes_it(tmp_path):
    app = make_app(tmp_path, manual_command=0.3)       # gentle: a full-strength push would reach the rail within 0.4 s
    key(app, pygame.K_RIGHT)
    app.simulate_for(0.3)
    x_right = app.env.state[0]
    assert x_right > 0.01 and app.last_command > 0
    key(app, pygame.K_RIGHT, up=True)
    key(app, pygame.K_LEFT)
    key(app, pygame.K_SPACE)                         # space overrides: zero drive while held
    app.simulate_for(0.1)
    assert app.last_command == 0.0
    key(app, pygame.K_SPACE, up=True)
    app.simulate_for(0.3)
    assert app.last_command < 0
    app.close()


def test_arrow_keys_are_ignored_with_a_notice_in_agent_modes(tmp_path):
    app = make_app(tmp_path)
    key(app, pygame.K_4)                              # baseline controller drives
    assert app.mode == "baseline"
    key(app, pygame.K_LEFT)
    assert pygame.K_LEFT not in app.held              # no silent mixing of manual and agent commands
    assert app._toast and "ignored" in app._toast[0]
    app.close()


def test_dqn_mode_without_checkpoint_is_refused_visibly(tmp_path):
    app = make_app(tmp_path)
    key(app, pygame.K_3)
    assert app.mode == "manual" and "No DQN checkpoint" in app._toast[0]
    app.close()


def test_pause_reset_mode_cycle_and_speed_keys(tmp_path):
    app = make_app(tmp_path)
    app.simulate_for(0.4)
    t = app.info["time"]
    key(app, pygame.K_p)
    app.simulate_for(0.4)
    assert app.paused and app.info["time"] == t
    key(app, pygame.K_p)
    key(app, pygame.K_r)
    assert app.info["time"] == 0.0 and abs(abs(app.env.physical_state[2]) - math.pi) < 0.2
    key(app, pygame.K_TAB)
    assert app.mode == "random"
    key(app, pygame.K_TAB)
    assert app.mode == "baseline"                     # dqn is skipped: no checkpoint loaded
    key(app, pygame.K_SPACE)                          # in agent mode space pauses
    assert app.paused
    key(app, pygame.K_n)
    assert app.start_type == "upright"
    s = app.speed
    key(app, pygame.K_EQUALS)
    assert app.speed == 2 * s
    app.close()


def test_baseline_mode_swings_up_and_balances_in_the_simulator(tmp_path):
    app = make_app(tmp_path, mode="baseline")
    app.simulate_for(8.0)
    assert not app.failed
    assert app.phase in ("balanced", "stabilizing")
    assert abs(app.env.physical_state[2]) < 0.1
    app.close()


def test_failure_state_and_auto_reset(tmp_path):
    app = make_app(tmp_path, mode="random", auto_reset_delay=0.5)
    app.env.reset(seed=0, options={"state": [0.59, 3.0, 0.0, 0.0]})
    app.curr_state = app.prev_state = app.env.state
    app.simulate_for(1.0)
    assert app.failed or app.info["time"] < 1.0       # either still showing the failure or already auto-reset
    app.simulate_for(1.0)
    assert not app.failed                             # auto reset after the delay
    manual = make_app(tmp_path, mode="manual")
    manual.env.reset(seed=0, options={"state": [0.59, 3.0, 0.0, 0.0]})
    manual.curr_state = manual.prev_state = manual.env.state
    key(manual, pygame.K_RIGHT)
    manual.simulate_for(0.3)
    manual.simulate_for(5.0)
    assert manual.failed                              # manual mode waits for R
    key(manual, pygame.K_r)
    assert not manual.failed
    app.close(); manual.close()


def test_physics_time_is_independent_of_frame_rate(tmp_path):
    def sim_time(fps):
        app = make_app(tmp_path, mode="baseline", fps=fps)
        app.simulate_for(2.0)
        t = app.info["time"]
        app.close()
        return t

    assert sim_time(30) == pytest.approx(sim_time(120), abs=0.03)


def test_screenshot_key_writes_png(tmp_path):
    app = make_app(tmp_path)
    key(app, pygame.K_s)
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1 and pngs[0].stat().st_size > 5000
    app.close()


def test_disturbance_key_kicks_the_pendulum(tmp_path):
    app = make_app(tmp_path, start="upright")
    app._reset()
    before = app.env.state[3]
    key(app, pygame.K_d)
    assert abs(app.env.state[3] - before) > 2.5
    app.close()


def test_run_loop_exits_on_escape_and_quit(tmp_path):
    app = make_app(tmp_path)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    app.run(max_frames=50)
    assert not app.running and app.frames < 50


def test_simulation_config_validation():
    with pytest.raises(ValueError):
        SimulationConfig(mode="telepathy")
    with pytest.raises(ValueError):
        SimulationConfig(manual_command=0.0)
