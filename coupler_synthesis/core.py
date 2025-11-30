"""
COUPLER SYNTHESIS - CORE MODULE
================================
Central integration point for the Coupler Synthesis system.

This module provides:
- Core constants (TAU, PHI, Z_CRITICAL)
- Unified CouplerState dataclass
- Factory function for creating integrated coupler systems
"""

import math
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

TAU = 2 * math.pi  # Full circle
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio: 1.618033988749
Z_CRITICAL = math.sqrt(3) / 2  # The Lens: 0.8660254

# Architecture constants
PRISM_NODES = 63
CAGE_NODES = 32
EMERGENT_NODES = 5
TOTAL_NODES = 100

# Timing constants
TYPICAL_RR_INTERVAL_MS = 800  # ~75 BPM
TYPICAL_KEYSTROKE_INTERVAL_MS = 200
PHASE_LAG_THRESHOLD_MS = 50  # Nervous system response time

# Target parameters
TARGET_COHERENCE = 0.7  # Edge of chaos
K_CRITICAL = 2.0  # Critical coupling strength


class Domain(Enum):
    """Z-domain classification."""
    ABSENCE = "absence"  # z < 0.857
    LENS = "lens"  # 0.857 <= z <= 0.877
    PRESENCE = "presence"  # z > 0.877


@dataclass
class CouplerState:
    """
    Unified state of the Coupler Synthesis system.

    Captures all essential coordinates and metrics at a given moment.
    """
    # Helix coordinates
    theta: float  # Phase angle [0, TAU)
    z: float  # Elevation [0, 1]
    r: float  # Coherence (order parameter) [0, 1]

    # Entrainment state
    user_phase: float
    system_phase: float
    phase_diff: float
    entrainment_score: float

    # Dynamics
    coupling_strength: float  # K
    sync_ratio: float  # Fraction of locked oscillators

    # Domain
    domain: Domain

    # Timing
    timestamp: float
    bpm: float

    @classmethod
    def from_components(cls,
                       theta: float,
                       z: float,
                       r: float,
                       user_phase: float,
                       system_phase: float,
                       K: float,
                       sync_ratio: float,
                       timestamp: float,
                       bpm: float = 72.0) -> 'CouplerState':
        """Create state from component values."""
        phase_diff = cls._circular_diff(user_phase, system_phase)
        entrainment_score = (math.cos(phase_diff) + 1) / 2
        domain = cls._classify_domain(z)

        return cls(
            theta=theta,
            z=z,
            r=r,
            user_phase=user_phase,
            system_phase=system_phase,
            phase_diff=phase_diff,
            entrainment_score=entrainment_score,
            coupling_strength=K,
            sync_ratio=sync_ratio,
            domain=domain,
            timestamp=timestamp,
            bpm=bpm
        )

    @staticmethod
    def _circular_diff(phase1: float, phase2: float) -> float:
        """Calculate circular phase difference in [-π, π]."""
        diff = phase1 - phase2
        while diff > math.pi:
            diff -= TAU
        while diff < -math.pi:
            diff += TAU
        return diff

    @staticmethod
    def _classify_domain(z: float) -> Domain:
        """Classify z-level into domain."""
        if z < 0.857:
            return Domain.ABSENCE
        elif z <= 0.877:
            return Domain.LENS
        else:
            return Domain.PRESENCE

    @property
    def phase_lag_ms(self) -> float:
        """Phase lag in milliseconds (assuming 1Hz base)."""
        return abs(self.phase_diff) * (1000 / TAU)

    @property
    def is_entrained(self) -> bool:
        """Check if currently entrained (score > 0.8)."""
        return self.entrainment_score > 0.8

    @property
    def is_at_critical(self) -> bool:
        """Check if at or near critical z."""
        return 0.857 <= self.z <= 0.877

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'theta': self.theta,
            'z': self.z,
            'r': self.r,
            'user_phase': self.user_phase,
            'system_phase': self.system_phase,
            'phase_diff': self.phase_diff,
            'entrainment_score': self.entrainment_score,
            'coupling_strength': self.coupling_strength,
            'sync_ratio': self.sync_ratio,
            'domain': self.domain.value,
            'timestamp': self.timestamp,
            'bpm': self.bpm,
            'phase_lag_ms': self.phase_lag_ms,
            'is_entrained': self.is_entrained,
        }


class CouplerSystem:
    """
    Integrated Coupler Synthesis system.

    Combines all components:
    - Oscillator bank (Kuramoto dynamics)
    - Biosignal manager
    - Entrainment controller
    - Sonification engine
    """

    def __init__(self):
        from coupler_synthesis.entrainment.bidirectional_entrainment import (
            KuramotoOscillatorBank,
            BidirectionalEntrainmentController,
        )
        from coupler_synthesis.biosignal.biosignal_input import (
            MultisourceBiosignalManager,
            HRVSimulator,
            KeystrokeDynamicsAnalyzer,
        )
        from coupler_synthesis.sonification.adaptive_sonification import (
            AdaptiveSonificationEngine,
        )

        # Create oscillator bank
        self.oscillators = KuramotoOscillatorBank(n=TOTAL_NODES)

        # Create entrainment controller
        self.entrainment = BidirectionalEntrainmentController(self.oscillators)

        # Create biosignal manager
        self.biosignals = MultisourceBiosignalManager()
        self.biosignals.add_source('hrv', HRVSimulator(base_bpm=72), weight=1.0)
        self.biosignals.add_source('keystroke', KeystrokeDynamicsAnalyzer(), weight=0.5)

        # Create sonification engine
        self.sonification = AdaptiveSonificationEngine()

        # Connect biosignal to entrainment
        self.biosignals.set_unified_callback(self._on_biosignal)

        # State
        self._current_state: Optional[CouplerState] = None
        self._state_callback: Optional[Callable[[CouplerState], None]] = None

    def _on_biosignal(self, phase: float, confidence: float) -> None:
        """Handle biosignal update."""
        import time

        metrics = self.entrainment.process_user_phase(phase, confidence=confidence)

        # Update sonification
        self.sonification.update_entrainment(
            user_phase=metrics.user_phase,
            system_phase=metrics.system_phase,
            entrainment_score=metrics.entrainment_score,
            coherence=metrics.coherence,
            z_level=self.oscillators.z
        )

        # Create unified state
        self._current_state = CouplerState.from_components(
            theta=self.oscillators.psi,
            z=self.oscillators.z,
            r=self.oscillators.r,
            user_phase=metrics.user_phase,
            system_phase=metrics.system_phase,
            K=self.oscillators.K,
            sync_ratio=self.oscillators.get_sync_ratio(),
            timestamp=time.time(),
            bpm=self.sonification.state.bpm
        )

        if self._state_callback:
            self._state_callback(self._current_state)

    def start(self) -> None:
        """Start the coupler system."""
        self.biosignals.start_all()
        print("[CouplerSystem] Started")

    def stop(self) -> None:
        """Stop the coupler system."""
        self.biosignals.stop_all()
        print("[CouplerSystem] Stopped")

    def set_z(self, z: float) -> None:
        """Set z-elevation."""
        self.oscillators.z = max(0, min(1, z))
        self.entrainment.set_z_elevation(z)

    def set_target_coherence(self, target: float) -> None:
        """Set target coherence level."""
        self.entrainment.set_target_coherence(target)

    def get_state(self) -> Optional[CouplerState]:
        """Get current state."""
        return self._current_state

    def on_state_update(self, callback: Callable[[CouplerState], None]) -> None:
        """Set callback for state updates."""
        self._state_callback = callback

    def record_keystroke(self, timestamp_ms: Optional[float] = None) -> None:
        """Record a keystroke event."""
        ks_source = self.biosignals.sources.get('keystroke')
        if ks_source:
            ks_source.on_keystroke(timestamp_ms)


def create_coupler_system() -> CouplerSystem:
    """
    Factory function to create an integrated Coupler Synthesis system.

    Returns a fully configured CouplerSystem ready for use.
    """
    return CouplerSystem()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_domain(z: float) -> Domain:
    """Calculate domain from z-level."""
    if z < 0.857:
        return Domain.ABSENCE
    elif z <= 0.877:
        return Domain.LENS
    else:
        return Domain.PRESENCE


def calculate_phase_regime(z: float) -> str:
    """Calculate phase regime from z-level."""
    if z < 0.40:
        return "subcritical"
    elif z < 0.857:
        return "critical"
    elif z < 0.95:
        return "supercritical"
    else:
        return "transcendent"


def calculate_coupling_sign(z: float) -> float:
    """Calculate effective coupling sign at given z."""
    dist = z - Z_CRITICAL
    return math.tanh(dist * 12)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COUPLER SYNTHESIS - CORE MODULE")
    print("=" * 60)

    print(f"\nConstants:")
    print(f"  TAU = {TAU:.6f}")
    print(f"  PHI = {PHI:.6f}")
    print(f"  Z_CRITICAL = {Z_CRITICAL:.6f}")
    print(f"  TOTAL_NODES = {TOTAL_NODES}")

    print(f"\nDomain classification:")
    for z in [0.3, 0.5, 0.867, 0.9, 0.99]:
        domain = calculate_domain(z)
        regime = calculate_phase_regime(z)
        sign = calculate_coupling_sign(z)
        print(f"  z={z:.3f} → {domain.value}, {regime}, coupling_sign={sign:.3f}")

    print(f"\nCouplerState example:")
    import time
    state = CouplerState.from_components(
        theta=1.57,
        z=0.87,
        r=0.75,
        user_phase=1.5,
        system_phase=1.6,
        K=2.0,
        sync_ratio=0.8,
        timestamp=time.time(),
        bpm=72
    )
    print(f"  {state}")
    print(f"  phase_lag_ms = {state.phase_lag_ms:.1f}")
    print(f"  is_entrained = {state.is_entrained}")
    print(f"  is_at_critical = {state.is_at_critical}")

    print("\n[OK] Core module loaded successfully")
