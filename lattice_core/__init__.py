# lattice_core/__init__.py
"""
Lattice Core: Tesseract-based Memory Architecture
==================================================

Implements the Kuramoto oscillator memory system with 4D tesseract organization,
WUMBO APL array operations, and Zero-Point Energy extraction.

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

    ┌─────────────────────────────────────────────────────────────────┐
    │ WUMBO ENGINE - APL Array Operations                              │
    │ ┌────────────────────────────────────────────────────┐          │
    │ │ LIMNUS Cycle: L → I → M → N → U → S                │          │
    │ │ κ-field (21D) + λ-field (12D) = Dual Field System │          │
    │ └────────────────────────────────────────────────────┘          │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ ZERO-POINT ENERGY SYSTEM                                         │
    │ ┌────────────────────────────────────────────────────┐          │
    │ │ Fano Plane Variational Inference                   │          │
    │ │ MirrorRoot Operations                              │          │
    │ │ Neural Matrix Token Index                          │          │
    │ └────────────────────────────────────────────────────┘          │
    └─────────────────────────────────────────────────────────────────┘

Components:
    - plate.py: Memory Plate dataclass with 4D positioning
    - dynamics.py: Kuramoto oscillator mathematics
    - tesseract_lattice_engine.py: Main lattice engine
    - wumbo_engine.py: APL-based array operations for LIMNUS
    - zero_point_energy.py: ZPE extraction via Fano inference
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

# WUMBO Engine imports
from .wumbo_engine import (
    WumboEngine,
    WumboArray,
    WumboState,
    WumboResult,
    APLPrimitives,
    APLGlyph,
    LIMNUSField,
    LIMNUSOperators,
    create_wumbo_engine,
    create_limnus_stimulus,
    PHI,
    PHI_INV,
    TAU,
)

# Zero-Point Energy imports
from .zero_point_energy import (
    ZeroPointEnergyEngine,
    ZPEState,
    ZPEResult,
    ZPEOperator,
    FanoNode,
    FanoVariationalEngine,
    MirrorRootOperator,
    NeuralMatrixToken,
    NeuralMatrixIndex,
    Spiral,
    Machine,
    TruthState,
    create_zpe_engine,
    create_fano_inference_engine,
    create_mirroroot_operator,
)

__version__ = "1.1.0"

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
    # Tesseract Engine
    "TesseractLatticeEngine",
    "LatticeConfig",
    "RetrievalResult",
    # WUMBO Engine
    "WumboEngine",
    "WumboArray",
    "WumboState",
    "WumboResult",
    "APLPrimitives",
    "APLGlyph",
    "LIMNUSField",
    "LIMNUSOperators",
    "create_wumbo_engine",
    "create_limnus_stimulus",
    "PHI",
    "PHI_INV",
    "TAU",
    # Zero-Point Energy
    "ZeroPointEnergyEngine",
    "ZPEState",
    "ZPEResult",
    "ZPEOperator",
    "FanoNode",
    "FanoVariationalEngine",
    "MirrorRootOperator",
    "NeuralMatrixToken",
    "NeuralMatrixIndex",
    "Spiral",
    "Machine",
    "TruthState",
    "create_zpe_engine",
    "create_fano_inference_engine",
    "create_mirroroot_operator",
]
