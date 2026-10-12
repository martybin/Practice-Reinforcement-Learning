"""Training callbacks: CSV / TensorBoard logging, console progress and checkpointing."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path


class Callback:
    """Hook points called by :class:`~pendulum_rl.training.trainer.Trainer`. All are optional."""

    def on_train_start(self, trainer) -> None: ...
    def on_stage_start(self, trainer, stage) -> None: ...
    def on_episode_end(self, trainer, record: dict) -> None: ...
    def on_eval(self, trainer, record: dict) -> None: ...
    def on_stage_end(self, trainer, stage, reason: str) -> None: ...
    def on_train_end(self, trainer, summary: dict) -> None: ...


class CSVLogger(Callback):
    """Writes ``episodes.csv`` and ``eval.csv`` incrementally (so a dashboard can follow a live run)."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self._files: dict[str, tuple] = {}

    def _write(self, name: str, row: dict) -> None:
        if name not in self._files:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            fh = open(self.log_dir / f"{name}.csv", "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()), extrasaction="ignore")
            writer.writeheader()
            self._files[name] = (fh, writer, list(row.keys()))
        fh, writer, _ = self._files[name]
        writer.writerow({k: _csv_value(v) for k, v in row.items()})
        fh.flush()

    def on_episode_end(self, trainer, record: dict) -> None:
        self._write("episodes", record)

    def on_eval(self, trainer, record: dict) -> None:
        self._write("eval", record)

    def on_train_end(self, trainer, summary: dict) -> None:
        for fh, _, _ in self._files.values():
            fh.close()
        self._files.clear()
        (self.log_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")


class TensorBoardLogger(Callback):
    """Logs scalars to TensorBoard if the ``tensorboard`` package is installed."""

    def __init__(self, log_dir: str | Path):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("TensorBoard logging requested but 'tensorboard' is not installed "
                               "(pip install tensorboard), or drop --tensorboard.") from exc
        self.writer = SummaryWriter(str(Path(log_dir) / "tb"))

    def _scalars(self, prefix: str, record: dict, step: int) -> None:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def on_episode_end(self, trainer, record: dict) -> None:
        self._scalars("train", record, record["env_steps"])

    def on_eval(self, trainer, record: dict) -> None:
        self._scalars("eval", record, record["env_steps"])

    def on_train_end(self, trainer, summary: dict) -> None:
        self.writer.close()


class ConsoleProgress(Callback):
    def __init__(self, every: int = 10):
        self.every = max(1, every)
        self._t0 = time.time()

    def on_stage_start(self, trainer, stage) -> None:
        print(f"\n=== stage '{stage.name}' (max {stage.max_episodes} episodes) ===", flush=True)

    def on_episode_end(self, trainer, r: dict) -> None:
        if r["stage_episode"] % self.every == 0:
            print(f"[{r['stage']}] ep {r['stage_episode']:5d} steps {r['env_steps']:8d} | return {r['return']:8.1f} "
                  f"len {r['length']:4d} | success(50) {r['rolling_success']:.2f} | eps {r['epsilon']:.2f} "
                  f"loss {r['loss']:.3f} | {r['termination_reason']} | {time.time() - self._t0:.0f}s", flush=True)

    def on_eval(self, trainer, r: dict) -> None:
        print(f"    eval @ {r['env_steps']} steps ({r['scenario']}): swing-up {r['swing_up_rate']:.2f} "
              f"balance {r['balance_rate']:.2f} violations {r['violation_rate']:.2f} return {r['mean_return']:.1f}", flush=True)

    def on_stage_end(self, trainer, stage, reason: str) -> None:
        print(f"--- stage '{stage.name}' finished: {reason}", flush=True)


class CheckpointCallback(Callback):
    """Keeps ``best.pt`` (best evaluation) and ``final.pt`` (end of training) in the run's checkpoint dir."""

    def __init__(self, ckpt_dir: str | Path):
        self.ckpt_dir = Path(ckpt_dir)
        self.best_score: tuple | None = None

    def on_stage_start(self, trainer, stage) -> None:
        self.best_score = None   # stages are different tasks: their scores are not comparable

    def on_eval(self, trainer, record: dict) -> None:
        score = (record["primary_rate"], record["mean_return"])
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            meta = trainer.metadata(record)
            trainer.agent.save(self.ckpt_dir / "best.pt", metadata=meta, overwrite=True)
            trainer.agent.save(self.ckpt_dir / f"best_{record['stage']}.pt", metadata=meta, overwrite=True)
            record["is_best"] = 1
        else:
            record["is_best"] = 0

    def on_train_end(self, trainer, summary: dict) -> None:
        trainer.agent.save(self.ckpt_dir / "final.pt", metadata=trainer.metadata(summary.get("final_eval")), overwrite=True)


class SnapshotCallback(Callback):
    """Saves the network every ``every`` training episodes to ``<ckpt_dir>/snapshots/ep_<episode>.pt``.

    ``ep_00000.pt`` is the freshly initialised (untrained) network. The snapshots let the dashboard's
    learning arena replay the *same* start with the policy as it was after N episodes of training.
    ``every <= 0`` disables snapshots.
    """

    def __init__(self, ckpt_dir: str | Path, every: int):
        self.dir = Path(ckpt_dir) / "snapshots"
        self.every = max(0, int(every))
        self._last_saved = -1

    def _save(self, trainer) -> None:
        episode = trainer.global_episode
        if self.every <= 0 or episode == self._last_saved:
            return
        trainer.agent.save(self.dir / f"ep_{episode:05d}.pt", metadata=trainer.metadata(trainer.last_eval))
        self._last_saved = episode

    def on_train_start(self, trainer) -> None:
        self._save(trainer)

    def on_episode_end(self, trainer, record: dict) -> None:
        if self.every > 0 and record["episode"] % self.every == 0:
            self._save(trainer)

    def on_train_end(self, trainer, summary: dict) -> None:
        self._save(trainer)      # always keep the last network, whatever the interval


def _csv_value(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, bool):
        return int(v)
    return v


def _json_default(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if hasattr(o, "item"):
        return o.item()
    return str(o)
