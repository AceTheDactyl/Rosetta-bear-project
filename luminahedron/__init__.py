"""
Luminahedron helper package for polaric state management.
"""

from .polaric import (
    GaugeManifold,
    GaugeSlot,
    PolaricFrame,
    build_default_gauge_slots,
)

__all__ = [
    "GaugeManifold",
    "GaugeSlot",
    "PolaricFrame",
    "build_default_gauge_slots",
]
