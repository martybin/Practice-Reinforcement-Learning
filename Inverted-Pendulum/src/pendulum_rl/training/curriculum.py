"""Training stages and the curriculum scheduler.

Three *different* tasks are trained, and a single ordinary DQN policy is **not** assumed
to solve all of them:

* ``stabilize`` (Stage A): start near upright, learn to hold balance. Policy = stabilizer.
* ``swingup``   (Stage B): start hanging, learn to pump energy and reach the catch region
  (success = entering a region a stabilizer can take over). Policy = swing-up only.
* ``combined``  (Stage C): start from both configurations, with slightly randomized physics
  and small disturbances; success = balance held. Policy = one network for everything.

``mode: curriculum`` runs A -> B -> C with the same network, advancing when the rolling
success rate of the current stage reaches its promotion threshold (or its episode budget
is exhausted).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class StageConfig:
    name: str
    env: dict = field(default_factory=dict)       # EnvConfig overrides for this stage
    reward: dict = field(default_factory=dict)    # RewardConfig overrides for this stage
    max_episodes: int = 500
    promote_success_rate: float = 0.9             # rolling success rate needed to advance (curriculum mode)
    promote_window: int = 50                      # episodes in the rolling window
    min_episodes: int = 100                       # never advance before this many episodes
    exploration_restart: float = 0.3              # epsilon restart value when this stage begins (after stage 0)
    exploration_decay_steps: int | None = None    # None -> keep the DQN default horizon

    def __post_init__(self) -> None:
        if self.max_episodes < 1:
            raise ValueError("max_episodes must be >= 1")
        if not 0.0 < self.promote_success_rate <= 1.0:
            raise ValueError("promote_success_rate must be in (0, 1]")


# Evaluation scenario matching each stage (see evaluation.evaluate.SCENARIOS).
STAGE_SCENARIO = {"stabilize": "stabilize", "swingup": "swingup", "combined": "combined"}


def stages_from_config(stage_table: Mapping[str, Mapping], order: list[str], max_episodes: int | None = None) -> list[StageConfig]:
    """Build the ordered stage list; ``max_episodes`` (if given) overrides every stage's budget."""
    stages = []
    for name in order:
        if name not in stage_table:
            raise KeyError(f"Stage '{name}' is not defined in the 'stages' config section "
                           f"(defined: {sorted(stage_table)})")
        spec = dict(stage_table[name])
        spec.setdefault("name", name)
        if max_episodes is not None:
            spec["max_episodes"] = max_episodes
        stages.append(StageConfig(**spec))
    return stages


class Curriculum:
    """Tracks which stage is active and decides when to advance."""

    def __init__(self, stages: list[StageConfig], promote: bool):
        if not stages:
            raise ValueError("A curriculum needs at least one stage")
        self.stages = stages
        self.promote = promote          # False for single-stage runs: always use the full budget
        self.index = 0
        self.episodes_in_stage = 0
        self._window: deque[bool] = deque(maxlen=stages[0].promote_window)

    @property
    def stage(self) -> StageConfig:
        return self.stages[self.index]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.stages)

    @property
    def rolling_success(self) -> float:
        return sum(self._window) / len(self._window) if self._window else 0.0

    def record(self, success: bool) -> None:
        self.episodes_in_stage += 1
        self._window.append(bool(success))

    def should_advance(self) -> bool:
        """True when the current stage is over (budget exhausted or promotion criterion met)."""
        st = self.stage
        if self.episodes_in_stage >= st.max_episodes:
            return True
        if not self.promote:
            return False
        window_full = len(self._window) >= st.promote_window
        return (window_full and self.episodes_in_stage >= st.min_episodes
                and self.rolling_success >= st.promote_success_rate)

    def advance(self) -> StageConfig | None:
        self.index += 1
        self.episodes_in_stage = 0
        if self.finished:
            return None
        self._window = deque(maxlen=self.stage.promote_window)
        return self.stage
