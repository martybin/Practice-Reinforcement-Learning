"""Typed, validated configuration loading.

YAML files are the single source of truth. They are parsed into frozen-in-spirit dataclasses so that
unknown keys, missing keys and inconsistent values fail loudly at start-up instead of deep in a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration is missing keys, has unknown keys, or holds inconsistent values."""


@dataclass
class ProjectInfo:
    name: str


@dataclass
class ScenarioConfig:
    seed: int
    n_trucks: int
    n_shovels: int
    horizon_minutes: float
    depot_to_shovel_minutes: list[float]
    shovel_to_depot_minutes: list[float]
    truck_capacity_choices_tonnes: list[float]
    truck_speed_factor_range: list[float]
    truck_reliability_factor_range: list[float]
    shovel_loading_rate_tpm: list[float]
    unload_time_minutes: float


@dataclass
class SimulationConfig:
    breakdowns_enabled: bool
    disturbances_enabled: bool
    truck_mtbf_minutes: float
    truck_repair_median_minutes: float
    truck_repair_sigma: float
    shovel_mtbf_minutes: float
    shovel_repair_median_minutes: float
    shovel_repair_sigma: float
    travel_jitter_sigma: float
    loading_jitter_sigma: float
    unload_jitter_sigma: float


@dataclass
class EnvConfig:
    time_scale_minutes: float
    invalid_action_policy: str
    max_consecutive_invalid_actions: int


@dataclass
class AgentConfig:
    seed: int
    hidden_sizes: list[int]
    gamma: float
    learning_rate: float
    batch_size: int
    replay_capacity: int
    learning_starts: int
    train_every_steps: int
    target_update_interval: int
    grad_clip_norm: float
    huber_delta: float
    double_dqn: bool
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    device: str


@dataclass
class TrainingConfig:
    episodes: int
    seed_base: int
    run_name: str
    checkpoint_every_episodes: int
    save_replay_buffer: bool


@dataclass
class EvaluationConfig:
    seed_start: int
    n_episodes: int
    policies: list[str]


@dataclass
class PathsConfig:
    results_dir: str
    checkpoint_dir: str


@dataclass
class RewardComponent:
    enabled: bool
    coefficient: float


@dataclass
class RewardConfig:
    payload: RewardComponent
    truck_wait: RewardComponent
    unnecessary_travel: RewardComponent
    shovel_starvation: RewardComponent
    truck_breakdown: RewardComponent
    invalid_action: RewardComponent


@dataclass
class ProjectConfig:
    project: ProjectInfo
    scenario: ScenarioConfig
    simulation: SimulationConfig
    env: EnvConfig
    agent: AgentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
    reward: RewardConfig

    def validate(self) -> None:
        """Cross-field checks that the type system cannot express."""
        sc, ag, tr, ev = self.scenario, self.agent, self.training, self.evaluation
        if sc.n_trucks < 1 or sc.n_shovels < 1:
            raise ConfigError("scenario.n_trucks and scenario.n_shovels must both be >= 1")
        for name in ("depot_to_shovel_minutes", "shovel_to_depot_minutes", "shovel_loading_rate_tpm"):
            values = getattr(sc, name)
            if len(values) != sc.n_shovels:
                raise ConfigError(f"scenario.{name} has {len(values)} entries; expected n_shovels={sc.n_shovels}")
        for name in ("truck_speed_factor_range", "truck_reliability_factor_range"):
            low, high = getattr(sc, name)
            if not 0 < low <= high:
                raise ConfigError(f"scenario.{name} must satisfy 0 < low <= high, got {[low, high]}")
        if sc.horizon_minutes <= 0:
            raise ConfigError("scenario.horizon_minutes must be positive")
        if not 0.0 < ag.gamma <= 1.0:
            raise ConfigError("agent.gamma must lie in (0, 1]")
        if not 0.0 <= ag.epsilon_end <= ag.epsilon_start <= 1.0:
            raise ConfigError("require 0 <= epsilon_end <= epsilon_start <= 1")
        if ag.batch_size > ag.replay_capacity:
            raise ConfigError("agent.batch_size cannot exceed agent.replay_capacity")
        if self.env.invalid_action_policy not in ("penalize", "raise"):
            raise ConfigError("env.invalid_action_policy must be 'penalize' or 'raise'")
        if ag.device not in ("auto", "cpu", "cuda"):
            raise ConfigError("agent.device must be 'auto', 'cpu' or 'cuda'")
        train_last_seed = tr.seed_base + tr.episodes - 1
        if train_last_seed >= ev.seed_start:
            raise ConfigError(
                f"training seeds [{tr.seed_base}, {train_last_seed}] overlap evaluation seeds starting at "
                f"{ev.seed_start}; evaluation must use held-out seeds"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve(self, relative: str) -> Path:
        """Resolve a path from the config against the project root."""
        return PROJECT_ROOT / relative


def _build(cls: type[T], data: Any, where: str) -> T:
    """Recursively build a dataclass from a mapping, rejecting unknown and missing keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"'{where}' must be a mapping, got {type(data).__name__}")
    declared = {f.name for f in fields(cls)}
    unknown = set(data) - declared
    if unknown:
        raise ConfigError(f"unknown key(s) under '{where}': {sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            raise ConfigError(f"missing key '{where}.{f.name}'")
        hint = hints[f.name]
        if is_dataclass(hint):
            kwargs[f.name] = _build(hint, data[f.name], f"{where}.{f.name}")
        else:
            kwargs[f.name] = data[f.name]
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return content


def config_from_dict(data: Mapping[str, Any]) -> ProjectConfig:
    """Build and validate a ProjectConfig from an already-parsed mapping (e.g. from a checkpoint)."""
    config = _build(ProjectConfig, dict(data), "config")
    config.validate()
    return config


def load_config(
    config_path: str | Path | None = None,
    reward_path: str | Path | None = None,
    quick: bool = False,
) -> ProjectConfig:
    """Load the main YAML, the reward YAML, and optionally the quick-mode overrides."""
    main_path = Path(config_path) if config_path else CONFIG_DIR / "default.yaml"
    reward_file = Path(reward_path) if reward_path else main_path.parent / "reward.yaml"

    data = _read_yaml(main_path)
    if quick:
        data = _deep_merge(data, _read_yaml(main_path.parent / "quick.yaml"))
    reward_data = _read_yaml(reward_file)
    if "reward" not in reward_data:
        raise ConfigError(f"{reward_file} must have a top-level 'reward' key")
    data["reward"] = reward_data["reward"]
    return config_from_dict(data)
