"""
Runtime loader for the historical PHYSICS TOE ENGINE file.

The original source keeps capitalised filenames with spaces which are
awkward to import.  This helper loads that module exactly once and
re-exports the symbols we need elsewhere in the code base.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

# Path to the original PHYSICS TOE ENGINE source file
TOE_PATH = (
    Path(__file__).resolve().parent / "TOE" / "PHYSICS TOE ENGINE.py"
)


@lru_cache(maxsize=1)
def get_toe_module() -> ModuleType:
    """Load and cache the giant TOE module once."""
    if not TOE_PATH.exists():
        raise FileNotFoundError(f"Missing TOE source at {TOE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "kaelhedron.toe_engine", TOE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TOE module from {TOE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Convenience aliases so consumers can simply import what they need
_toe = get_toe_module()
SacredConstants = _toe.SacredConstants
SO7Algebra = _toe.SO7Algebra
GaugeHierarchy = _toe.GaugeHierarchy
E8Structure = _toe.E8Structure
KaelhedronModel = _toe.Kaelhedron
FanoPlane = _toe.FanoPlane

__all__ = [
    "get_toe_module",
    "SacredConstants",
    "SO7Algebra",
    "GaugeHierarchy",
    "E8Structure",
    "KaelhedronModel",
    "FanoPlane",
]
