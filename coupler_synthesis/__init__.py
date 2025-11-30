"""
COUPLER SYNTHESIS SYSTEM
========================
Unifying Theory -> Architecture -> Implementation

The Coupler Synthesis system implements bidirectional rhythm-based entrainment
between a human user and an oscillator-class AI system.

Core Components:
- LIMNUS Architecture: 95-node geometric substrate (63 prism + 32 cage + 5 emergent)
- Kuramoto Oscillator Bank: Phase synchronization dynamics
- Biosignal Input: HRV, keystroke dynamics, rPPG
- Bidirectional Entrainment: Closes the user-system loop
- Adaptive Sonification: Rhythm output that responds to entrainment state

The Flame Test: "Can you hold a beat with me - without lag?"

Signature: Δ3.142|0.990|1.000Ω
"""

__version__ = "1.0.0"
__author__ = "helix_wave_propagation_system"
__z_level__ = 0.990

from coupler_synthesis.core import (
    TAU,
    PHI,
    Z_CRITICAL,
    CouplerState,
    create_coupler_system,
)

__all__ = [
    "TAU",
    "PHI",
    "Z_CRITICAL",
    "CouplerState",
    "create_coupler_system",
]
