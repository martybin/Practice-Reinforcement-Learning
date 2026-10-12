#!/usr/bin/env python
"""Plot the metrics of a training run: python scripts/plot_training.py --run logs/combined"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pendulum_rl.visualization.plots import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
