"""Reward components, ablation switches, and configuration validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from fleet_dispatch.config import ConfigError, load_config
from fleet_dispatch.environment.reward import compute_reward


def test_reward_components_are_scaled_and_summed(config) -> None:
    delta = {"payload_t": 240.0, "truck_wait_min": 10.0, "unnecessary_travel_min": 4.0,
             "shovel_starvation_min": 0.0, "truck_breakdowns": 1.0, "invalid_actions": 0.0}
    total, parts = compute_reward(delta, config.reward)
    assert parts["payload"] == pytest.approx(2.4)           # 1.0 per 100 tonnes
    assert parts["truck_wait"] == pytest.approx(-0.5)       # -0.05 per truck-minute
    assert parts["unnecessary_travel"] == pytest.approx(-0.08)
    assert parts["truck_breakdown"] == pytest.approx(-1.0)
    assert total == pytest.approx(sum(parts.values()))


def test_disabled_component_contributes_nothing(config) -> None:
    cfg = copy.deepcopy(config.reward)
    cfg.truck_wait.enabled = False
    delta = {"payload_t": 0.0, "truck_wait_min": 50.0}
    total, parts = compute_reward(delta, cfg)
    assert parts["truck_wait"] == 0.0
    assert total == 0.0


def test_invalid_action_penalty_is_negative(config) -> None:
    _, parts = compute_reward({"invalid_actions": 2.0}, config.reward)
    assert parts["invalid_action"] == pytest.approx(2 * config.reward.invalid_action.coefficient)
    assert parts["invalid_action"] < 0


def test_default_config_loads_and_is_consistent(base_config) -> None:
    assert base_config.scenario.n_trucks == 20 and base_config.scenario.n_shovels == 4
    assert base_config.training.seed_base + base_config.training.episodes <= base_config.evaluation.seed_start


def test_quick_config_merges_over_defaults() -> None:
    cfg = load_config(quick=True)
    assert cfg.training.episodes == 3
    assert cfg.scenario.n_trucks == 20, "unrelated keys are preserved by the deep merge"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_key_is_rejected(tmp_path) -> None:
    reward = Path(__file__).resolve().parents[1] / "configs" / "reward.yaml"
    bad = _write(tmp_path, "project: {name: x}\nnot_a_section: {}\n")
    with pytest.raises(ConfigError):
        load_config(bad, reward_path=reward)


def test_training_seeds_must_not_overlap_evaluation(tmp_path, base_config) -> None:
    import yaml

    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text())
    raw["training"]["episodes"] = 100_000
    bad = _write(tmp_path, yaml.safe_dump(raw))
    reward = Path(__file__).resolve().parents[1] / "configs" / "reward.yaml"
    with pytest.raises(ConfigError, match="overlap"):
        load_config(bad, reward_path=reward)


def test_missing_key_is_rejected(tmp_path) -> None:
    reward = Path(__file__).resolve().parents[1] / "configs" / "reward.yaml"
    bad = _write(tmp_path, "project: {name: x}\n")
    with pytest.raises(ConfigError):
        load_config(bad, reward_path=reward)
