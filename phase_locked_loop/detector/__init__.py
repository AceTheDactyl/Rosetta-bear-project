"""
PHASE DETECTOR SUBMODULE
========================
Measures the phase difference between reference and VCO signals.

The phase detector is the "ear" of the PLL—it listens to both
the reference and VCO, computes their phase difference, and
outputs an error signal that drives the loop filter.

Multiple detector types are available:
- Multiplying (XOR-like sinusoidal)
- Type II (frequency-sensitive)
- Bang-bang (digital/binary)

Signature: Δ3.142|0.995|1.000Ω
"""

__version__ = "1.0.0"
__z_level__ = 0.995

from phase_locked_loop.detector.phase_detector import (
    DetectorType,
    DetectorConfig,
    PhaseDetectorBase,
    MultiplyingPhaseDetector,
    TypeIIPhaseDetector,
    BangBangPhaseDetector,
    create_phase_detector,
)

__all__ = [
    "DetectorType",
    "DetectorConfig",
    "PhaseDetectorBase",
    "MultiplyingPhaseDetector",
    "TypeIIPhaseDetector",
    "BangBangPhaseDetector",
    "create_phase_detector",
]
