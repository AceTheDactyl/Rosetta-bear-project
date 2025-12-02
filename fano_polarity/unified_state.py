# fano_polarity/unified_state.py
"""
Unified System State Contract
=============================

Defines the shared state contract that all subsystems can publish and subscribe to.
This establishes the common language between:
- Scalar Architecture (7 domains, 21 interference nodes)
- Kaelhedron State Bus (21 cells, 7 seals x 3 faces)
- Luminahedron Gauge Manifold (12D gauge structure)
- Polarity Loop (forward/backward polarity gating)

The unified state enables coherent feedback loops across all systems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class LoopState(Enum):
    """Loop controller states (from scalar_architecture)."""
    DIVERGENT = "divergent"
    CONVERGING = "converging"
    CRITICAL = "critical"
    CLOSED = "closed"


class KFormationStatus(Enum):
    """K-Formation status levels."""
    INACTIVE = "inactive"
    APPROACHING = "approaching"
    THRESHOLD = "threshold"
    FORMED = "formed"


class PolarityPhase(Enum):
    """Polarity loop phase states."""
    IDLE = "idle"
    FORWARD_TRIGGERED = "forward_triggered"
    GATED = "gated"
    COHERENCE_RELEASED = "coherence_released"


@dataclass
class DomainSnapshot:
    """Snapshot of a single domain's state."""
    domain_index: int
    name: str
    saturation: float
    loop_state: LoopState
    phase: float


@dataclass
class CellSnapshot:
    """Snapshot of a single Kaelhedron cell."""
    seal_index: int
    face_index: int
    label: str
    theta: float
    kappa: float
    activation: float


@dataclass
class PolaritySnapshot:
    """Snapshot of the polarity loop state."""
    phase: PolarityPhase
    forward_points: Optional[Tuple[int, int]]
    forward_line: Optional[Tuple[int, int, int]]
    gate_remaining: float
    coherence_point: Optional[int]


@dataclass
class UnifiedSystemState:
    """
    Complete unified state across all subsystems.

    This is the master state contract that enables polarity feedback
    to coordinate all mathematical structures.
    """
    # Timestamp
    timestamp: float = field(default_factory=time.time)

    # Global scalar metrics
    kappa: float = 0.0              # Coherence parameter (z-level)
    theta: float = 0.0              # Global phase
    recursion_depth: int = 1        # Number of closed loops
    charge: int = 0                 # Topological charge (closed - divergent)

    # Kaelhedron coherence
    kaelhedron_coherence: float = 0.0
    kaelhedron_phase: float = 0.0

    # Luminahedron divergence
    luminahedron_divergence: float = 0.5
    gauge_coupling: float = 0.0

    # Polaric balance
    polaric_balance: float = 0.5    # Balance between Kaelhedron and Luminahedron

    # K-Formation
    k_formation_status: KFormationStatus = KFormationStatus.INACTIVE
    k_formation_progress: float = 0.0

    # Polarity loop
    polarity_phase: PolarityPhase = PolarityPhase.IDLE

    # Loop states summary
    loops_closed: int = 0
    loops_critical: int = 0
    loops_converging: int = 0
    loops_divergent: int = 7

    # Domain snapshots (optional detail)
    domains: List[DomainSnapshot] = field(default_factory=list)

    # Cell snapshots (optional detail)
    cells: List[CellSnapshot] = field(default_factory=list)

    # Polarity snapshot (optional detail)
    polarity: Optional[PolaritySnapshot] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "kappa": self.kappa,
            "theta": self.theta,
            "recursion_depth": self.recursion_depth,
            "charge": self.charge,
            "kaelhedron_coherence": self.kaelhedron_coherence,
            "kaelhedron_phase": self.kaelhedron_phase,
            "luminahedron_divergence": self.luminahedron_divergence,
            "gauge_coupling": self.gauge_coupling,
            "polaric_balance": self.polaric_balance,
            "k_formation_status": self.k_formation_status.value,
            "k_formation_progress": self.k_formation_progress,
            "polarity_phase": self.polarity_phase.value,
            "loop_counts": {
                "closed": self.loops_closed,
                "critical": self.loops_critical,
                "converging": self.loops_converging,
                "divergent": self.loops_divergent,
            },
        }

    @property
    def composite_coherence(self) -> float:
        """Compute composite coherence across all systems."""
        # Weighted average of Kaelhedron coherence and inverse Luminahedron divergence
        kael_weight = 0.6
        lumi_weight = 0.4
        lumi_coherence = 1.0 - self.luminahedron_divergence
        return kael_weight * self.kaelhedron_coherence + lumi_weight * lumi_coherence

    @property
    def is_coherent(self) -> bool:
        """Check if system has achieved coherence."""
        PHI_INV = 0.618033988749895
        return self.composite_coherence > PHI_INV

    @property
    def signature(self) -> str:
        """Generate state signature."""
        return f"Δ|{self.loops_closed}/7-closed|z{self.kappa:.2f}|η{self.kaelhedron_coherence:.2f}|Ω"


# Type alias for state callbacks
StateCallback = Callable[[UnifiedSystemState], None]


class UnifiedStateRegistry:
    """
    Registry for unified state subscribers.

    Enables pub/sub pattern for state updates across all subsystems.
    """

    def __init__(self):
        self._subscribers: List[StateCallback] = []
        self._latest_state: Optional[UnifiedSystemState] = None

    def subscribe(self, callback: StateCallback) -> None:
        """Subscribe to state updates."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: StateCallback) -> None:
        """Unsubscribe from state updates."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def publish(self, state: UnifiedSystemState) -> None:
        """Publish state update to all subscribers."""
        self._latest_state = state
        for callback in self._subscribers:
            callback(state)

    def latest(self) -> Optional[UnifiedSystemState]:
        """Get the latest published state."""
        return self._latest_state


# Global registry singleton
_global_registry: Optional[UnifiedStateRegistry] = None


def get_state_registry() -> UnifiedStateRegistry:
    """Get or create the global state registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = UnifiedStateRegistry()
    return _global_registry
