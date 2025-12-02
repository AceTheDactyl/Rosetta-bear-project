"""
Kaelhedron Python package
=========================

This module exposes lightweight wrappers around the historical
``PHYSICS TOE ENGINE`` sources so the rest of the repository can import
the Sacred constants, so(7) algebra, and Kaelhedron helpers without
re-parsing giant monolithic scripts.

The package keeps the original files (with their capitals and spaces)
intact while providing conventional snake_case modules for new code.
"""

from __future__ import annotations

from .toe_loader import (
    get_toe_module,
    KaelhedronModel,
    SacredConstants,
    SO7Algebra,
    GaugeHierarchy,
    E8Structure,
)
from .state_bus import KaelCellState, KaelhedronStateBus
from .kformation import KFormationStatus, evaluate_k_formation
from .fano_automorphisms import (
    get_automorphism_for_line,
    get_automorphism_from_word,
    IDENTITY_PERMUTATION,
)

__all__ = [
    "get_toe_module",
    "KaelhedronModel",
    "SacredConstants",
    "SO7Algebra",
    "GaugeHierarchy",
    "E8Structure",
    "KaelCellState",
    "KaelhedronStateBus",
    "KFormationStatus",
    "evaluate_k_formation",
    "get_automorphism_for_line",
    "get_automorphism_from_word",
    "IDENTITY_PERMUTATION",
]
