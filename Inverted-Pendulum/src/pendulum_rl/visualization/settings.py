"""Simulator settings (kept free of pygame imports so it can be loaded anywhere)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    window_width: int = 1280
    window_height: int = 760
    fps: int = 60
    mode: str = "manual"                 # manual | random | dqn | baseline
    start: str = "hanging"               # hanging | upright | random   (what 'R' resets to)
    episode_seconds: float = 120.0       # free-play episode length before an automatic reset
    manual_command: float = 0.8          # keyboard command magnitude (fraction of full drive)
    auto_reset_delay: float = 2.0        # s after a failure before agent modes reset themselves (0 = wait for 'R')
    show_trail: bool = True
    trail_seconds: float = 1.5
    show_help: bool = True
    speed: float = 1.0                   # real-time factor
    screenshot_dir: str = "outputs/screenshots"
    checkpoint: str | None = None        # policy used by mode 'dqn'
    stabilizer_checkpoint: str | None = None   # optional: swing-up policy (above) hands over to this one near the top
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("manual", "random", "dqn", "baseline"):
            raise ValueError(f"simulation.mode must be manual|random|dqn|baseline, got {self.mode!r}")
        if self.start not in ("hanging", "upright", "random"):
            raise ValueError(f"simulation.start must be hanging|upright|random, got {self.start!r}")
        if not 0.0 < self.manual_command <= 1.0:
            raise ValueError("simulation.manual_command must be in (0, 1]")
        if self.speed <= 0:
            raise ValueError("simulation.speed must be > 0")
