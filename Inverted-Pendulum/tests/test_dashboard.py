"""The optional Streamlit dashboard: shows real logs, and says so when there is nothing to show."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "pendulum_rl" / "visualization" / "dashboard.py"


def run_app(root: Path, monkeypatch):
    monkeypatch.setenv("PENDULUM_RL_ROOT", str(root))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()
    return at


def test_dashboard_reports_missing_training_data_instead_of_inventing_it(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception
    messages = " ".join(i.value for i in at.info)
    assert "No training runs found" in messages and "No evaluation reports" in messages and "No checkpoints" in messages
    assert len(at.metric) == 0 and len(at.subheader) == 0


def test_dashboard_shows_metrics_from_a_real_log_directory(tmp_path, monkeypatch):
    n = 40
    run = tmp_path / "logs" / "demo"
    run.mkdir(parents=True)
    pd.DataFrame({
        "episode": range(1, n + 1), "env_steps": np.arange(1, n + 1) * 100, "return": np.linspace(0, 80, n), "length": 100,
        "success": [0, 1] * (n // 2), "swing_up_success": 1, "longest_balance": 1.0, "mean_abs_angle_error": 0.3,
        "epsilon": np.linspace(1, 0.1, n), "loss": np.linspace(1, 0.1, n), "q_max": 1.0, "q_mean": 0.5, "q_min": 0.0,
        "action_frac_0": 0.3, "action_frac_1": 0.3, "action_frac_2": 0.4}).to_csv(run / "episodes.csv", index=False)
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Episodes" in labels and at.metric[0].value == str(n)
    headings = [s.value for s in at.subheader]
    for name in ("Episode reward", "Success rate", "TD loss", "Q-value statistics", "Action distribution"):
        assert name in headings
