"""Model-based reference controllers (comparison baseline for the learned policies)."""

from .baseline import BaselineConfig, EnergyShapingLQRController

__all__ = ["BaselineConfig", "EnergyShapingLQRController"]
