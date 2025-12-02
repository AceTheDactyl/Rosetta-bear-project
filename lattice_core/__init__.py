# lattice_core/__init__.py
"""
Lattice Core: Tesseract-based Memory Architecture
==================================================

Implements the Kuramoto oscillator memory system with 4D tesseract organization.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ TESSERACT LATTICE ENGINE                                         │
    │ ┌────────────────────────────────────────────────────┐          │
    │ │ Kuramoto Oscillator Network                        │          │
    │ │                                                    │          │
    │ │ Plate₁ ←→ Plate₂ ←→ Plate₃ ←→ ... ←→ PlateN       │          │
    │ │ ↕         ↕         ↕              ↕               │          │
    │ │ Plate₄ ←→ Plate₅ ←→ Plate₆ ←→ ... ←→ PlateM       │          │
    │ │                                                    │          │
    │ │ Each plate: (position, phase, frequency)           │          │
    │ └────────────────────────────────────────────────────┘          │
    │                                                                  │
    │ • update() - Kuramoto dynamics integration                       │
    │ • resonance_retrieval() - Phase perturbation + evolution        │
    │ • Hebbian learning - Connection strengthening                   │
    │ • Order parameter tracking                                       │
    └─────────────────────────────────────────────────────────────────┘

Components:
    - plate.py: Memory Plate dataclass with 4D positioning
    - dynamics.py: Kuramoto oscillator mathematics
    - tesseract_lattice_engine.py: Main lattice engine
"""

from .plate import (
    MemoryPlate,
    EmotionalState,
    create_tesseract_vertices,
)
from .dynamics import (
    kuramoto_update,
    compute_order_parameter,
    compute_quartet_coupling,
    hebbian_update,
    compute_energy,
)
from .tesseract_lattice_engine import (
    TesseractLatticeEngine,
    LatticeConfig,
    RetrievalResult,
)

__version__ = "1.0.0"

__all__ = [
    # Plate
    "MemoryPlate",
    "EmotionalState",
    "create_tesseract_vertices",
    # Dynamics
    "kuramoto_update",
    "compute_order_parameter",
    "compute_quartet_coupling",
    "hebbian_update",
    "compute_energy",
    # Engine
    "TesseractLatticeEngine",
    "LatticeConfig",
    "RetrievalResult",
]
