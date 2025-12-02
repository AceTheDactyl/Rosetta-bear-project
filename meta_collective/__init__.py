# meta_collective/__init__.py
"""
Meta-Collective Architecture
=============================

Hierarchical active inference framework with nested free energy minimization.

Architecture Layers:
    ┌───────────────────────────────────────────────────────────────────┐
    │                     META-COLLECTIVE (z=0.95)                       │
    │  ┌─────────────────────────────────────────────────────────────┐  │
    │  │                    TRIAD-A (z=0.90)                          │  │
    │  │  ┌─────────────────────────────────────────────────────┐    │  │
    │  │  │           TOOL (z=0.867)                             │    │  │
    │  │  │  ┌──────────────────────────────────────────────┐   │    │  │
    │  │  │  │ Internal Model (Kaelhedron + Luminahedron)   │   │    │  │
    │  │  │  │     κ-field   │   λ-field                    │   │    │  │
    │  │  │  └──────────────────────────────────────────────┘   │    │  │
    │  │  └─────────────────────────────────────────────────────┘    │  │
    │  └─────────────────────────────────────────────────────────────┘  │
    │                              ▲                                     │
    │                              │ Interaction (pattern sharing)       │
    │                              ▼                                     │
    │  ┌─────────────────────────────────────────────────────────────┐  │
    │  │                    TRIAD-B (z=0.90)                          │  │
    │  │  (Similar nested structure...)                               │  │
    │  └─────────────────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────────────┘

Each level minimizes its own free energy while contributing
to the free energy minimization of containing levels.
"""

from .fields import KappaField, LambdaField, DualFieldState
from .internal_model import InternalModel
from .free_energy import FreeEnergyMinimizer, VariationalState, Precision
from .tool import Tool, ToolState
from .triad import Triad, TriadInteraction, PatternMessage
from .collective import MetaCollective, CollectiveState

__version__ = "1.0.0"
__author__ = "Rosetta Bear CBS"

# Z-level constants for the architecture
Z_INTERNAL_MODEL = 0.800  # Base internal model coherence
Z_TOOL = 0.867            # Tool layer coherence
Z_TRIAD = 0.900           # Triad layer coherence
Z_META_COLLECTIVE = 0.950 # Meta-collective coherence

__all__ = [
    # Fields
    "KappaField",
    "LambdaField",
    "DualFieldState",
    # Internal Model
    "InternalModel",
    # Free Energy
    "FreeEnergyMinimizer",
    "VariationalState",
    "Precision",
    # Tool
    "Tool",
    "ToolState",
    # Triad
    "Triad",
    "TriadInteraction",
    "PatternMessage",
    # Collective
    "MetaCollective",
    "CollectiveState",
    # Constants
    "Z_INTERNAL_MODEL",
    "Z_TOOL",
    "Z_TRIAD",
    "Z_META_COLLECTIVE",
]
