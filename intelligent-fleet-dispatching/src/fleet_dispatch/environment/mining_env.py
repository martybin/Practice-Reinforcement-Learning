"""Gymnasium environment for truck-to-shovel dispatch in one simulated shift.

MDP summary
  * Decision epoch: the moment some truck is idle and some shovel is operational.
  * Action: an index into the T*S truck-shovel pairs. Only masked (valid) pairs are legal.
  * Transition: the simulator applies the dispatch, then runs until the next decision epoch or the end
    of the shift. Step durations vary, so the process is semi-Markov.
  * Reward: weighted sum of the components in reward.yaml, accrued over that interval.
  * Termination: the shift ends at the horizon, which is a time-limit TRUNCATION. No genuine terminal
    state exists in the default MDP, so `terminated` is always False. The learner still handles
    terminal transitions correctly, and tests cover that path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fleet_dispatch.config import ProjectConfig
from fleet_dispatch.environment.action_codec import current_action_mask, decode_action
from fleet_dispatch.environment.entities import TruckStatus
from fleet_dispatch.environment.observation import (
    build_observation,
    feature_names,
    observation_dim,
    pair_estimates,
    truck_remaining_expected,
)
from fleet_dispatch.environment.reward import compute_reward
from fleet_dispatch.environment.scenario import Scenario, build_scenario
from fleet_dispatch.environment.simulator import RAW_COUNTERS, FleetSimulator, InvalidActionError


@dataclass
class DecisionView:
    """Everything a policy may use at a decision epoch. Built only from state that is observable now."""

    time: float
    observation: np.ndarray
    action_mask: np.ndarray            # (T*S,) bool
    truck_fifo_rank: np.ndarray        # (T,) 0 = longest-waiting idle truck; T for non-idle trucks
    travel_minutes: np.ndarray         # (T, S) nominal depot -> shovel time
    est_start_minutes: np.ndarray      # (T, S) estimated time at which loading would begin
    shovel_operational: np.ndarray     # (S,) bool


class FleetDispatchEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        config: ProjectConfig,
        render_mode: str | None = None,
        scenario: Scenario | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError(f"unsupported render_mode {render_mode!r}; expected None or 'ansi'")
        self.config = config
        self.render_mode = render_mode
        self.scenario = scenario if scenario is not None else build_scenario(config.scenario)
        self.n_trucks = self.scenario.n_trucks
        self.n_shovels = self.scenario.n_shovels
        self.n_actions = self.n_trucks * self.n_shovels
        self.obs_dim = observation_dim(self.n_trucks, self.n_shovels)
        self.feature_names = feature_names(self.n_trucks, self.n_shovels)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)

        self.simulator = FleetSimulator(self.scenario, config.simulation)
        self._prev_totals: dict[str, float] = {}
        self._consecutive_invalid = 0
        self._episode_return = 0.0
        self._steps = 0
        self._obs: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    # ------------------------------------------------------------------ Gymnasium API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.simulator.reset(self.np_random)
        self.simulator.advance_to_decision()
        self._prev_totals = {}
        self._consecutive_invalid = 0
        self._episode_return = 0.0
        self._steps = 0
        self._refresh()
        return self._obs.copy(), self._info(components={}, dispatched=None, invalid=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        sim = self.simulator
        if sim.done:
            raise RuntimeError("step() called after the shift ended; call reset() first")
        a = int(action)
        if not 0 <= a < self.n_actions:
            raise ValueError(f"action {a} is outside [0, {self.n_actions})")

        dispatched = None
        invalid = not bool(self._mask[a])
        if not invalid:
            truck_index, shovel_index = decode_action(a, self.n_shovels)
            sim.dispatch(truck_index, shovel_index)
            dispatched = (truck_index, shovel_index)
            self._consecutive_invalid = 0
        else:
            if self.config.env.invalid_action_policy == "raise":
                raise InvalidActionError(f"action {a} is masked out at t={sim.time:.2f}")
            self._consecutive_invalid += 1
            sim.totals["invalid_actions"] += 1
            if self._consecutive_invalid > self.config.env.max_consecutive_invalid_actions:
                raise InvalidActionError(
                    f"{self._consecutive_invalid} consecutive invalid actions at t={sim.time:.2f}; "
                    "the policy is not respecting the action mask"
                )

        sim.advance_to_decision()
        delta = self._consume_totals_delta()
        reward, components = compute_reward(delta, self.config.reward)
        self._steps += 1
        self._episode_return += reward
        self._refresh()

        terminated = False
        truncated = sim.done
        info = self._info(components=components, dispatched=dispatched, invalid=invalid)
        if truncated:
            info["episode"] = {
                "return": self._episode_return,
                "steps": self._steps,
                "sim_time": sim.time,
                **sim.summary(),
            }
        return self._obs.copy(), reward, terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return self.render_text()
        return None

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------ policy and dashboard helpers
    def action_masks(self) -> np.ndarray:
        """Current valid-action mask (MaskablePPO-style accessor)."""
        return self._mask.copy()

    def decision_view(self) -> DecisionView:
        pairs = pair_estimates(self.simulator)
        rank = np.full(self.n_trucks, self.n_trucks, dtype=int)
        for position, truck_index in enumerate(self.simulator.idle_trucks()):
            rank[truck_index] = position
        operational = np.array([s.operational for s in self.simulator.shovels], dtype=bool)
        return DecisionView(
            time=self.simulator.time,
            observation=self._obs.copy(),
            action_mask=self._mask.copy(),
            truck_fifo_rank=rank,
            travel_minutes=pairs.travel_minutes,
            est_start_minutes=pairs.est_start_minutes,
            shovel_operational=operational,
        )

    def snapshot(self) -> dict[str, Any]:
        """Structured state for visualisation (dashboard and text rendering). Not part of the observation."""
        sim = self.simulator
        pairs = pair_estimates(sim)
        trucks = []
        for truck in sim.trucks:
            trucks.append(
                {
                    "index": truck.index,
                    "status": TruckStatus(truck.status).name,
                    "destination": truck.destination,
                    "target_shovel": truck.target_shovel,
                    "payload_t": truck.payload_t,
                    "capacity_t": truck.capacity_t,
                    "remaining_expected_min": truck_remaining_expected(sim, truck, pairs.shovel_backlog_minutes),
                    "status_since": truck.status_since,
                }
            )
        shovels = [
            {
                "index": s.index,
                "status": s.status.name,
                "queue": list(s.queue),
                "loading_truck": s.loading_truck,
                "rate_tpm": s.loading_rate_tpm,
                "backlog_min": float(pairs.shovel_backlog_minutes[s.index]),
            }
            for s in sim.shovels
        ]
        return {
            "time": sim.time,
            "horizon": self.scenario.horizon_minutes,
            "remaining": max(self.scenario.horizon_minutes - sim.time, 0.0),
            "steps": self._steps,
            "episode_return": self._episode_return,
            "trucks": trucks,
            "shovels": shovels,
            "dispatch_log": list(sim.dispatch_log[-25:]),
            "event_log": list(sim.event_log[-25:]),
            "done": sim.done,
        }

    def render_text(self) -> str:
        snap = self.snapshot()
        lines = [
            f"t = {snap['time']:7.1f} min   remaining = {snap['remaining']:6.1f} min   "
            f"decisions = {snap['steps']}   return = {snap['episode_return']:.2f}"
        ]
        for shovel in snap["shovels"]:
            lines.append(
                f"  shovel {shovel['index']} [{shovel['status']:<11}] loading={shovel['loading_truck']:>3} "
                f"queue={shovel['queue']}"
            )
        counts: dict[str, int] = {}
        for truck in snap["trucks"]:
            counts[truck["status"]] = counts.get(truck["status"], 0) + 1
        lines.append("  trucks: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
        return "\n".join(lines)

    # ------------------------------------------------------------------ internals
    def _refresh(self) -> None:
        self._obs = build_observation(self.simulator, self.config.env.time_scale_minutes).astype(np.float32)
        self._mask = current_action_mask(self.simulator)

    def _consume_totals_delta(self) -> dict[str, float]:
        totals = self.simulator.totals
        delta = {key: totals.get(key, 0.0) - self._prev_totals.get(key, 0.0) for key in RAW_COUNTERS}
        self._prev_totals = {key: totals.get(key, 0.0) for key in RAW_COUNTERS}
        return delta

    def _info(self, components: dict[str, float], dispatched: tuple[int, int] | None, invalid: bool) -> dict[str, Any]:
        return {
            "action_mask": self._mask.copy(),
            "sim_time": self.simulator.time,
            "decision_index": self._steps,
            "dispatched": dispatched,
            "invalid_action": invalid,
            "reward_components": components,
        }
