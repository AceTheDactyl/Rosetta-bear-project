"""
Fano Polarity Feedback Package
==============================

Implements the self-referential polarity engine based on Fano plane axioms:
- Forward polarity (points -> line): "positive arc"
- Backward polarity (lines -> point): "negative arc"

Coherence is gated until both polarities agree.
"""

from .core import line_from_points, point_from_lines
from .loop import GateState, PolarityLoop
from .service import PolarityService

__all__ = [
    "line_from_points",
    "point_from_lines",
    "GateState",
    "PolarityLoop",
    "PolarityService",
]
