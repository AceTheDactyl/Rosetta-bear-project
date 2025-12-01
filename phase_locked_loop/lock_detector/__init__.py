"""
LOCK DETECTOR SUBMODULE
=======================
Estimates coherence and determines lock state.

The lock detector is the "awareness" of the PLL—it monitors
the phase error history and determines whether the system
has achieved stable synchronization.

Uses the order parameter r = |mean(exp(jε))| as the
coherence metric, analogous to Kuramoto's formulation.

When r → 1: Perfect lock, minimal phase error
When r → 0: Unlocked, large/varying errors

Signature: Δ3.142|0.995|1.000Ω
"""

__version__ = "1.0.0"
__z_level__ = 0.995

from phase_locked_loop.lock_detector.detector import (
    LockState,
    LockMetrics,
    LockDetectorConfig,
    CoherenceEstimator,
    LockStateMachine,
    LockDetector,
    create_lock_detector,
)

__all__ = [
    "LockState",
    "LockMetrics",
    "LockDetectorConfig",
    "CoherenceEstimator",
    "LockStateMachine",
    "LockDetector",
    "create_lock_detector",
]
