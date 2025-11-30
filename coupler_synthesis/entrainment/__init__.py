"""
Bidirectional Entrainment Module
================================
Core coupler mechanism for phase synchronization.
"""

from coupler_synthesis.entrainment.bidirectional_entrainment import (
    EntrainmentState,
    EntrainmentMetrics,
    EntrainmentConfig,
    KuramotoOscillatorBank,
    BidirectionalEntrainmentController,
)

__all__ = [
    "EntrainmentState",
    "EntrainmentMetrics",
    "EntrainmentConfig",
    "KuramotoOscillatorBank",
    "BidirectionalEntrainmentController",
]
