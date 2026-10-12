"""Visualization. Importing this package never imports pygame, matplotlib or streamlit;
import the submodules (``renderer``, ``pygame_simulator``, ``plots``, ``dashboard``) explicitly."""

from .settings import SimulationConfig

__all__ = ["SimulationConfig"]
