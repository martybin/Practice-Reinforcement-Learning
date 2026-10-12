"""Interactive real-time simulator (Pygame).

Physics advances in fixed steps of ``control_dt`` driven by an accumulator, independently of the
render rate; frames interpolate between the last two physics states for smooth motion. Four
control modes share one command path so a human and an agent can never both drive silently:

* ``manual``   - keyboard (left/right arrows). Space releases the drive (zero command).
* ``random``   - uniformly random discrete actions.
* ``dqn``      - greedy trained DQN checkpoint (optionally a swing-up policy that hands over to a stabilizer).
* ``baseline`` - energy-shaping swing-up + LQR stabilization.

In every mode except ``manual`` the arrow keys are *ignored* and an on-screen notice says so;
switching modes is always an explicit key press (1-4 / Tab).
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT
from ..control import BaselineConfig, EnergyShapingLQRController
from ..environment import EnvConfig, PendulumEnv, RewardConfig
from ..evaluation.evaluate import BaselinePolicy, DQNPolicy, RandomPolicy, SwitchingPolicy
from ..physics import PhysicsParams
from ..physics.dynamics import wrap_angle
from .renderer import HudInfo, PendulumView
from .settings import SimulationConfig

MODES = ("manual", "random", "dqn", "baseline")
START_TYPES = ("hanging", "upright", "random")
CONTROLS = [
    ("LEFT/RIGHT", "drive (manual)"), ("SPACE", "zero drive / pause"), ("1 2 3 4", "manual/random/dqn/base"),
    ("TAB", "next mode"), ("R", "reset"), ("P", "pause"), ("N", "next start"), ("D", "kick pendulum"),
    ("T", "trail"), ("-  =", "slower / faster"), ("S", "screenshot"), ("H", "toggle help"), ("ESC", "exit"),
]


class SimulatorApp:
    def __init__(self, physics: PhysicsParams, env_cfg: EnvConfig, reward: RewardConfig, sim: SimulationConfig,
                 baseline_cfg: BaselineConfig | None = None, headless: bool = False):
        self.sim = sim
        self.headless = headless
        self.physics, self.reward = physics, reward
        free_play = replace(env_cfg, action_mode="discrete", terminate_on_success=False,
                            max_episode_seconds=sim.episode_seconds, physics_randomization=0.0,
                            disturbance_probability=0.0, start_mode="hanging" if sim.start == "random" else sim.start)
        self.env = PendulumEnv(free_play, physics, reward)
        self.control_dt = free_play.control_dt
        self.baseline_cfg = baseline_cfg or BaselineConfig(control_dt=self.control_dt)
        self.rng = np.random.default_rng(sim.seed)

        self.policies: dict[str, object] = {}
        self.dqn_label = ""
        self._init_policies()

        self.view = PendulumView(self.env.model, (sim.window_width, sim.window_height), headless=headless,
                                 trail_seconds=sim.trail_seconds, show_trail=sim.show_trail)
        self.view.pg.key.set_repeat()   # we track held keys ourselves

        self.mode = "manual"
        self.start_type = sim.start
        self.show_help = sim.show_help
        self.paused = False
        self.running = True
        self.speed = sim.speed
        self.held: set[int] = set()
        self._toast: tuple[str, float] | None = None
        self._episode_seed = sim.seed
        self.frames = 0
        self._reset()
        self.set_mode(sim.mode, announce=False)

    # ------------------------------------------------------------------ policies
    @classmethod
    def from_config(cls, cfg: dict, sim: SimulationConfig, headless: bool = False) -> "SimulatorApp":
        """Build from a merged config; if ``sim.checkpoint`` is set, its saved physics/env/reward are used."""
        from ..config import build_baseline_config, build_env_config, build_physics, build_reward
        physics, env_cfg, reward = build_physics(cfg), build_env_config(cfg), build_reward(cfg)
        if sim.checkpoint:
            from ..agents import DQNAgent
            _, meta = DQNAgent.from_checkpoint(_resolve(sim.checkpoint), device="cpu")
            if meta.get("physics"):
                physics = PhysicsParams(**meta["physics"])
                env_cfg = EnvConfig(**meta["base_env"])
                reward = RewardConfig(**meta["reward"])
        return cls(physics, env_cfg, reward, sim, build_baseline_config(cfg), headless)

    def _init_policies(self) -> None:
        ctl_cfg = replace(self.baseline_cfg, control_dt=self.control_dt)
        self.policies["baseline"] = BaselinePolicy(lambda env: EnergyShapingLQRController(env.model, ctl_cfg))
        self.policies["random"] = RandomPolicy(self.sim.seed)
        if self.sim.checkpoint:
            from ..agents import DQNAgent
            path = _resolve(self.sim.checkpoint)
            agent, meta = DQNAgent.from_checkpoint(path, device="cpu")
            if agent.obs_dim != self.env.obs_dim or agent.n_actions != self.env.n_actions:
                raise ValueError(f"Checkpoint {path} expects obs_dim={agent.obs_dim}, n_actions={agent.n_actions}, "
                                 f"but the simulator environment has {self.env.obs_dim}/{self.env.n_actions}.")
            label = Path(path).parent.name + "/" + Path(path).stem
            if self.sim.stabilizer_checkpoint:
                stab, _ = DQNAgent.from_checkpoint(_resolve(self.sim.stabilizer_checkpoint), device="cpu")
                self.policies["dqn"] = SwitchingPolicy(agent, stab)
                label += " + stabilizer"
            else:
                self.policies["dqn"] = DQNPolicy(agent)
            self.dqn_label = label

    # ------------------------------------------------------------------ state / modes
    def set_mode(self, mode: str, announce: bool = True) -> bool:
        if mode == "dqn" and "dqn" not in self.policies:
            self.toast("No DQN checkpoint loaded - start with: python scripts/simulate.py --checkpoint <file.pt>")
            return False
        self.mode = mode
        self.held.clear()
        if mode in self.policies:
            self.policies[mode].reset(self.env)
        if announce:
            self.toast(f"Mode: {self.mode_label()}")
        return True

    def mode_label(self) -> str:
        return {"manual": "MANUAL  (keyboard)", "random": "RANDOM POLICY",
                "dqn": f"DQN POLICY  {self.dqn_label}", "baseline": "BASELINE  energy + LQR"}[self.mode]

    def toast(self, message: str, seconds: float = 2.5) -> None:
        self._toast = (message, time.perf_counter() + seconds)

    def _reset(self) -> None:
        self._episode_seed += 1
        self.obs, self.info = self.env.reset(seed=self._episode_seed, options={"start": self.start_type})
        for name in ("baseline", "random"):
            self.policies[name].reset(self.env)
        if "dqn" in self.policies:
            self.policies["dqn"].reset(self.env)
        self.prev_state = self.curr_state = self.env.state
        self.acc = 0.0
        self.total_reward = 0.0
        self.last_command = 0.0
        self.phase = "swinging_up"
        self.failed = False
        self.fail_clock = 0.0
        self.view.reset_trail()

    # ------------------------------------------------------------------ input
    def handle_event(self, event) -> None:
        pg = self.view.pg
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.KEYUP:
            self.held.discard(event.key)
        elif event.type == pg.KEYDOWN:
            self._on_key(event.key)

    def _on_key(self, key: int) -> None:
        pg = self.view.pg
        if key == pg.K_ESCAPE:
            self.running = False
        elif key in (pg.K_LEFT, pg.K_RIGHT):
            if self.mode == "manual":
                self.held.add(key)
            else:
                self.toast(f"{self.mode_label()} is driving - arrow keys ignored. Press 1 for manual control.", 3.0)
        elif key == pg.K_SPACE:
            if self.mode == "manual":
                self.held.add(key)    # while held: zero drive
            else:
                self.toggle_pause()
        elif key == pg.K_p:
            self.toggle_pause()
        elif key == pg.K_r:
            self._reset()
            self.toast(f"Reset ({self.start_type} start)")
        elif key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
            self.set_mode(MODES[key - pg.K_1])
        elif key == pg.K_TAB:
            for i in range(1, len(MODES) + 1):
                if self.set_mode(MODES[(MODES.index(self.mode) + i) % len(MODES)]):
                    break
        elif key == pg.K_n:
            self.start_type = START_TYPES[(START_TYPES.index(self.start_type) + 1) % len(START_TYPES)]
            self.toast(f"Next reset starts: {self.start_type}  (press R)")
        elif key == pg.K_t:
            self.view.show_trail = not self.view.show_trail
            self.view.reset_trail()
        elif key == pg.K_h:
            self.show_help = not self.show_help
        elif key == pg.K_d:
            kick = float(self.rng.choice([-1.0, 1.0]) * 3.0)
            self.env.apply_disturbance(kick)
            self.curr_state = self.env.state
            self.toast(f"Disturbance: {kick:+.1f} rad/s kick")
        elif key == pg.K_s:
            self.screenshot()
        elif key == pg.K_MINUS:
            self.speed = max(0.125, self.speed / 2)
            self.toast(f"Speed x{self.speed:g}")
        elif key in (pg.K_EQUALS, pg.K_PLUS):
            self.speed = min(4.0, self.speed * 2)
            self.toast(f"Speed x{self.speed:g}")

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def screenshot(self) -> Path:
        out_dir = _resolve(self.sim.screenshot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"pendulum_{time.strftime('%Y%m%d_%H%M%S')}_{self.frames}.png"
        self._draw(interpolate=False)
        self.view.save(str(path))
        self.toast(f"Saved {path.name}")
        return path

    # ------------------------------------------------------------------ simulation
    def _manual_command(self) -> float:
        pg = self.view.pg
        if pg.K_SPACE in self.held:
            return 0.0
        return self.sim.manual_command * ((pg.K_RIGHT in self.held) - (pg.K_LEFT in self.held))

    def _physics_step(self) -> None:
        if self.mode == "manual":
            result = self.env.step_command(self._manual_command())
        else:
            action = self.policies[self.mode].act(self.env, self.obs)
            result = self.env.step_command(float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1))) \
                if self.mode == "baseline" else self.env.step(action)
        self.obs, reward, terminated, truncated, self.info = result
        self.total_reward += reward
        self.prev_state, self.curr_state = self.curr_state, self.env.state
        self.last_command = self.info["command"]
        self.phase = self.info["phase"]
        if terminated:
            self.failed = True
            self.fail_clock = 0.0
            self.toast(f"Failed: {self.info['termination_reason'].replace('_', ' ')}", 3.0)
        elif truncated:
            self._reset()

    def update(self, frame_dt: float) -> None:
        """Advance the simulation by ``frame_dt`` seconds of wall-clock time."""
        if self.paused:
            return
        if self.failed:
            self.fail_clock += frame_dt
            if self.mode != "manual" and 0 < self.sim.auto_reset_delay <= self.fail_clock:
                self._reset()
            return
        self.acc += min(frame_dt, 0.1) * self.speed
        steps = 0
        while self.acc >= self.control_dt and steps < 8 and not self.failed:
            self._physics_step()
            self.acc -= self.control_dt
            steps += 1
        if steps == 8:   # fell behind real time: drop the backlog rather than spiral
            self.acc = 0.0

    # ------------------------------------------------------------------ drawing
    def _display_state(self, interpolate: bool) -> np.ndarray:
        s = self.curr_state.copy()
        if interpolate and not self.paused and not self.failed:
            a = min(1.0, self.acc / self.control_dt)
            p, c = self.prev_state, self.curr_state
            s[0] = p[0] + a * (c[0] - p[0])
            s[2] = p[2] + a * wrap_angle(c[2] - p[2])
        return s

    def _hud(self) -> HudInfo:
        now = time.perf_counter()
        toast = None
        if self._toast and now < self._toast[1]:
            toast = (self._toast[0], min(1.0, (self._toast[1] - now) / 0.5))
        notice = ""
        if self.failed:
            notice = "Failed - press R to reset" if self.mode == "manual" or self.sim.auto_reset_delay <= 0 else \
                f"Failed - auto reset in {max(0.0, self.sim.auto_reset_delay - self.fail_clock):.1f} s"
        elif self.mode != "manual":
            notice = "Agent drives the support (arrow keys ignored)"
        return HudInfo(command=self.last_command, time=float(self.info.get("time", 0.0)), reward=self.total_reward,
                       phase="failed" if self.failed else self.phase, mode_label=self.mode_label(), paused=self.paused,
                       toast=toast, show_help=self.show_help, controls=CONTROLS, notice=notice)

    def _draw(self, interpolate: bool = True, show: bool = False):
        return self.view.draw(self._display_state(interpolate), self._hud(), show=show,
                              advance_trail=not self.paused and not self.failed)

    # ------------------------------------------------------------------ main loop
    def run(self, max_frames: int | None = None) -> None:
        pg = self.view.pg
        last = time.perf_counter()
        while self.running and (max_frames is None or self.frames < max_frames):
            for event in pg.event.get():
                self.handle_event(event)
            now = time.perf_counter()
            dt = (1.0 / self.sim.fps) if self.headless else now - last
            last = now
            self.update(dt)
            self._draw(show=not self.headless)
            if not self.headless:
                self.view.clock.tick(self.sim.fps)
            self.frames += 1
        self.close()

    def simulate_for(self, seconds: float) -> None:
        """Advance the simulation by ``seconds`` of wall-clock-equivalent time without drawing (headless runs)."""
        for _ in range(int(seconds * self.sim.fps)):
            self.update(1.0 / self.sim.fps)
            self.frames += 1

    def close(self) -> None:
        self.env.close()
        self.view.close()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (Path.cwd() / p if (Path.cwd() / p).exists() else PROJECT_ROOT / p)
