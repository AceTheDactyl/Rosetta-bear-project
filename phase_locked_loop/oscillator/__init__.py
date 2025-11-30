"""
VOLTAGE-CONTROLLED OSCILLATOR SUBMODULE
=======================================
Generates output phase that tracks control voltage variations.

The VCO is the "voice" of the PLL—it produces the output signal
whose phase is steered by the loop filter to match the reference.

The fundamental equation:
    dφ/dt = ω₀ + K_vco · V(t)

Where ω₀ is the center frequency and V(t) is the control voltage.

Signature: Δ3.142|0.995|1.000Ω
"""

__version__ = "1.0.0"
__z_level__ = 0.995

from phase_locked_loop.oscillator.vco import (
    VCOConfig,
    VCOState,
    VoltageControlledOscillator,
    MultiVCOBank,
    create_vco,
    create_vco_bank,
)

__all__ = [
    "VCOConfig",
    "VCOState",
    "VoltageControlledOscillator",
    "MultiVCOBank",
    "create_vco",
    "create_vco_bank",
]
