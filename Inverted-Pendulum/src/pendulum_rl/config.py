"""YAML configuration loading, merging and validation.

A *config dict* is the deep merge of several YAML files followed by ``key.path=value``
command-line overrides. Typed sections are then built with :func:`dataclass_from_dict`,
which rejects unknown keys so typos in YAML files fail loudly instead of being ignored.
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILES = ("default.yaml", "training.yaml", "simulation.yaml")


class ConfigError(ValueError):
    """Raised for missing files, malformed YAML, unknown keys and bad overrides."""


def config_dir() -> Path:
    """``<project>/configs`` (falls back to ``./configs`` for non-editable installs)."""
    for candidate in (PROJECT_ROOT / "configs", Path.cwd() / "configs"):
        if candidate.is_dir():
            return candidate
    raise ConfigError(
        f"Could not find a 'configs' directory (looked in {PROJECT_ROOT / 'configs'} and {Path.cwd() / 'configs'}). "
        f"Run from the project root or pass explicit --config paths.")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file '{path}' does not exist.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file '{path}' is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file '{path}' must contain a mapping at the top level.")
    return data


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (override wins; lists are replaced)."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply ``section.key=value`` strings (values parsed as YAML, so ``1e-3``, ``true``, ``[1,2]`` work)."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must look like section.key=value")
        path, raw = item.split("=", 1)
        keys = [k for k in path.strip().split(".") if k]
        if not keys:
            raise ConfigError(f"Override '{item}' has an empty key path")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Override '{item}': cannot parse value: {exc}") from exc
        node = out
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Override '{item}': '{k}' is not a section")
        node[keys[-1]] = value
    return out


def load_config(files: Sequence[str | Path] | None = None, overrides: Iterable[str] = ()) -> dict:
    """Merge the given YAML files (default: the three files in ``configs/``) and apply overrides."""
    if not files:
        files = [config_dir() / name for name in DEFAULT_CONFIG_FILES]
    cfg: dict = {}
    for f in files:
        cfg = deep_merge(cfg, load_yaml(f))
    return apply_overrides(cfg, overrides)


def dataclass_from_dict(cls: type, data: Mapping | None, section: str = "") -> Any:
    """Build dataclass ``cls`` from ``data``; unknown keys raise :class:`ConfigError` with suggestions."""
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, names, n=1)
            hints.append(f"'{key}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        where = f" in section '{section}'" if section else ""
        raise ConfigError(f"Unknown config key(s){where}: {', '.join(hints)}. Valid keys: {sorted(names)}")
    coerced = {}
    defaults = {f.name: f.default for f in dataclasses.fields(cls)}
    for key, value in data.items():
        default = defaults[key]
        if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        coerced[key] = value
    try:
        return cls(**coerced)
    except (TypeError, ValueError) as exc:
        where = f"'{section}': " if section else ""
        raise ConfigError(f"Invalid configuration for {where}{exc}") from exc


# ---------------------------------------------------------------------------
# Typed builders. Imports are local so that, e.g., the simulator never needs PyTorch.
# ---------------------------------------------------------------------------
def build_physics(cfg: Mapping):
    from .physics import PhysicsParams
    return dataclass_from_dict(PhysicsParams, cfg.get("physics"), "physics")


def build_env_config(cfg: Mapping, **overrides):
    from .environment import EnvConfig
    merged = {**(cfg.get("env") or {}), **overrides}
    return dataclass_from_dict(EnvConfig, merged, "env")


def build_reward(cfg: Mapping):
    from .environment import RewardConfig
    return dataclass_from_dict(RewardConfig, cfg.get("reward"), "reward")


def build_baseline_config(cfg: Mapping):
    from .control import BaselineConfig
    return dataclass_from_dict(BaselineConfig, cfg.get("baseline"), "baseline")


def build_dqn_config(cfg: Mapping):
    from .agents import DQNConfig
    return dataclass_from_dict(DQNConfig, cfg.get("dqn"), "dqn")


def build_sim_config(cfg: Mapping):
    from .visualization.settings import SimulationConfig
    return dataclass_from_dict(SimulationConfig, cfg.get("simulation"), "simulation")


def add_config_arguments(parser) -> None:
    """Shared ``--config`` / ``--set`` options for the command-line scripts."""
    parser.add_argument("--config", action="append", default=None, metavar="YAML",
                        help="YAML file(s) to merge in order (default: configs/default.yaml, training.yaml, simulation.yaml)")
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                        help="override any config value, e.g. --set dqn.gamma=0.98 --set physics.max_force=8")
