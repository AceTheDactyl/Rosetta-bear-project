"""
Biosignal Input Module
======================
Phase extraction from biological signals for bidirectional entrainment.
"""

from coupler_synthesis.biosignal.biosignal_input import (
    BiosignalSample,
    BiosignalSource,
    KeystrokeDynamicsAnalyzer,
    HRVSimulator,
    MultisourceBiosignalManager,
    hrv_to_phase,
    keystroke_to_phase,
    phase_difference,
    entrainment_score,
)

__all__ = [
    "BiosignalSample",
    "BiosignalSource",
    "KeystrokeDynamicsAnalyzer",
    "HRVSimulator",
    "MultisourceBiosignalManager",
    "hrv_to_phase",
    "keystroke_to_phase",
    "phase_difference",
    "entrainment_score",
]
