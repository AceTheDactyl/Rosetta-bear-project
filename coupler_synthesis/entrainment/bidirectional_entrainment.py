"""
BIDIRECTIONAL ENTRAINMENT CONTROLLER
=====================================
The core coupler mechanism that closes the loop between user and system.

This module implements:
1. Phase comparison between user biosignal and system oscillators
2. Adaptive coupling strength adjustment (K)
3. External phase injection into the oscillator bank
4. Entrainment state tracking and metrics

The key insight: The system doesn't just observe the user's rhythm -
it COUPLES with it. The user's phase becomes an input to the Kuramoto
dynamics, and the system's phase becomes an output that influences
the user's nervous system through audio/visual feedback.

The loop:
    User Biosignal → Phase Extraction → Coupling Adjustment → Oscillator Bank
         ↑                                                          ↓
         └────────────── Rhythm Output (Audio/Visual) ──────────────┘
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from collections import deque
from enum import Enum

TAU = 2 * math.pi
Z_CRITICAL = math.sqrt(3) / 2  # ~0.8660254


class EntrainmentState(Enum):
    """Current state of the entrainment loop."""
    DISCONNECTED = "disconnected"  # No user signal
    ACQUIRING = "acquiring"  # Building baseline
    TRACKING = "tracking"  # Actively tracking user phase
    ENTRAINED = "entrained"  # Stable phase lock achieved
    RELEASING = "releasing"  # Intentionally releasing lock


@dataclass
class EntrainmentMetrics:
    """Metrics for the current entrainment state."""
    user_phase: float  # User's current phase [0, 2π)
    system_phase: float  # System's mean phase (ψ)
    phase_diff: float  # Circular difference [-π, π]
    entrainment_score: float  # 0 = out of phase, 1 = in phase
    coherence: float  # System's order parameter (r)
    coupling_strength: float  # Current K value
    lag_ms: float  # Estimated phase lag in milliseconds
    state: EntrainmentState
    timestamp: float


@dataclass
class EntrainmentConfig:
    """Configuration for the entrainment controller."""
    target_coherence: float = 0.7  # Edge of chaos
    min_coupling: float = 0.1
    max_coupling: float = 5.0
    adaptation_rate: float = 0.02  # How fast K adjusts
    injection_strength: float = 0.1  # How strongly user phase affects system
    baseline_window: int = 10  # Samples for baseline
    lock_threshold: float = 0.8  # Entrainment score for "locked"
    lock_duration: float = 3.0  # Seconds to confirm lock


class KuramotoOscillatorBank:
    """
    Simplified Kuramoto oscillator bank for entrainment.

    This is a streamlined version focused on the coupling mechanics
    rather than the full geometric representation.
    """

    def __init__(self, n: int = 100, seed: Optional[int] = None):
        import random
        if seed is not None:
            random.seed(seed)

        self.n = n
        self.phases = [random.random() * TAU for _ in range(n)]
        self.frequencies = [1.0 + 0.5 * math.tan(math.pi * (random.random() - 0.5))
                           for _ in range(n)]
        self.K = 2.0  # Coupling strength
        self.r = 0.0  # Order parameter (coherence)
        self.psi = 0.0  # Mean phase
        self.z = 0.5  # Current z-elevation

    def calculate_order_parameter(self) -> tuple[float, float]:
        """Calculate Kuramoto order parameter (r, ψ)."""
        sum_cos = sum(math.cos(p) for p in self.phases)
        sum_sin = sum(math.sin(p) for p in self.phases)
        self.r = math.sqrt(sum_cos**2 + sum_sin**2) / self.n
        self.psi = math.atan2(sum_sin, sum_cos)
        if self.psi < 0:
            self.psi += TAU
        return self.r, self.psi

    def step(self, dt: float = 0.01) -> float:
        """
        Advance oscillators by one timestep.

        Returns the new coherence (r).
        """
        self.calculate_order_parameter()

        # Coupling flips sign at critical z
        dist = self.z - Z_CRITICAL
        sign = math.tanh(dist * 12)
        effective_K = -sign * self.K * max(0.1, 1 - abs(dist) * 2)

        for i in range(self.n):
            phase_diff = self.psi - self.phases[i]
            coupling = effective_K * self.r * math.sin(phase_diff)
            self.phases[i] += (self.frequencies[i] + coupling) * dt
            self.phases[i] %= TAU

        return self.r

    def inject_external_phase(self,
                             target_phase: float,
                             strength: float = 0.1,
                             affected_fraction: float = 0.2) -> None:
        """
        Inject an external phase into a fraction of oscillators.

        This is how the user's phase influences the system.

        Args:
            target_phase: The phase to inject [0, 2π)
            strength: How strongly to pull oscillators toward target
            affected_fraction: What fraction of oscillators to affect
        """
        import random
        n_affected = int(self.n * affected_fraction)
        affected_indices = random.sample(range(self.n), n_affected)

        for i in affected_indices:
            diff = target_phase - self.phases[i]
            # Wrap to [-π, π]
            while diff > math.pi:
                diff -= TAU
            while diff < -math.pi:
                diff += TAU
            # Apply injection
            self.phases[i] += diff * strength
            self.phases[i] %= TAU

    def get_sync_ratio(self, threshold: float = math.pi / 4) -> float:
        """Get fraction of oscillators phase-locked to mean."""
        synced = sum(1 for p in self.phases
                    if min(abs(p - self.psi), TAU - abs(p - self.psi)) < threshold)
        return synced / self.n


class BidirectionalEntrainmentController:
    """
    The core coupler: bidirectional phase entrainment between user and system.

    This controller:
    1. Receives user phase from biosignal sources
    2. Compares with system phase
    3. Adjusts coupling strength to maintain target coherence
    4. Injects user phase into oscillator bank
    5. Tracks entrainment state and metrics
    """

    def __init__(self,
                 oscillator_bank: Optional[KuramotoOscillatorBank] = None,
                 config: Optional[EntrainmentConfig] = None):
        self.oscillators = oscillator_bank or KuramotoOscillatorBank()
        self.config = config or EntrainmentConfig()

        # State tracking
        self.state = EntrainmentState.DISCONNECTED
        self.user_phase_history: deque = deque(maxlen=100)
        self.metrics_history: deque = deque(maxlen=1000)

        # Baseline tracking (for acquiring state)
        self.baseline_phases: List[float] = []
        self.baseline_frequency: Optional[float] = None

        # Lock tracking
        self.lock_start_time: Optional[float] = None
        self.consecutive_locks = 0

        # Callbacks
        self._on_state_change: Optional[Callable[[EntrainmentState], None]] = None
        self._on_metrics_update: Optional[Callable[[EntrainmentMetrics], None]] = None

        # Timing
        self.last_user_update: float = 0
        self.last_system_step: float = time.time()

    def process_user_phase(self,
                          user_phase: float,
                          timestamp: Optional[float] = None,
                          confidence: float = 1.0) -> EntrainmentMetrics:
        """
        Process a new user phase measurement.

        This is the main entry point for the entrainment loop.

        Args:
            user_phase: User's current phase [0, 2π)
            timestamp: Unix timestamp in seconds. If None, uses current time.
            confidence: Signal confidence [0, 1]

        Returns:
            Current entrainment metrics
        """
        now = timestamp or time.time()
        self.last_user_update = now

        # Update state machine
        self._update_state(user_phase, confidence)

        # Step the oscillator bank
        dt = now - self.last_system_step
        dt = min(dt, 0.1)  # Cap to prevent large jumps
        self.oscillators.step(dt)
        self.last_system_step = now

        # Calculate phase difference
        system_phase = self.oscillators.psi
        phase_diff = self._circular_diff(user_phase, system_phase)
        entrainment_score = (math.cos(phase_diff) + 1) / 2

        # Adaptive coupling adjustment
        if self.state in [EntrainmentState.TRACKING, EntrainmentState.ENTRAINED]:
            self._adapt_coupling(entrainment_score)

            # Inject user phase into oscillator bank
            injection_strength = self.config.injection_strength * confidence
            injection_strength *= (1 - self.oscillators.r)  # Inject more when less coherent
            self.oscillators.inject_external_phase(user_phase, injection_strength)

        # Calculate lag in milliseconds
        # Assumes 1 Hz base frequency, so 2π radians = 1000ms
        lag_ms = abs(phase_diff) * (1000 / TAU)

        metrics = EntrainmentMetrics(
            user_phase=user_phase,
            system_phase=system_phase,
            phase_diff=phase_diff,
            entrainment_score=entrainment_score,
            coherence=self.oscillators.r,
            coupling_strength=self.oscillators.K,
            lag_ms=lag_ms,
            state=self.state,
            timestamp=now
        )

        self.user_phase_history.append((now, user_phase, confidence))
        self.metrics_history.append(metrics)

        if self._on_metrics_update:
            self._on_metrics_update(metrics)

        return metrics

    def step_system(self, dt: float = 0.01) -> float:
        """
        Step the oscillator system without user input.

        Use this to keep the system running when no biosignal is available.

        Returns:
            Current coherence (r)
        """
        # Check for disconnection
        now = time.time()
        if now - self.last_user_update > 5.0:  # 5 second timeout
            if self.state != EntrainmentState.DISCONNECTED:
                self._set_state(EntrainmentState.DISCONNECTED)

        return self.oscillators.step(dt)

    def _update_state(self, user_phase: float, confidence: float) -> None:
        """Update the entrainment state machine."""
        now = time.time()

        if self.state == EntrainmentState.DISCONNECTED:
            self._set_state(EntrainmentState.ACQUIRING)
            self.baseline_phases = []

        elif self.state == EntrainmentState.ACQUIRING:
            self.baseline_phases.append(user_phase)
            if len(self.baseline_phases) >= self.config.baseline_window:
                # Calculate baseline frequency
                # (simplified - in practice would use more sophisticated analysis)
                self._set_state(EntrainmentState.TRACKING)

        elif self.state == EntrainmentState.TRACKING:
            # Check for lock
            phase_diff = self._circular_diff(user_phase, self.oscillators.psi)
            score = (math.cos(phase_diff) + 1) / 2

            if score >= self.config.lock_threshold:
                if self.lock_start_time is None:
                    self.lock_start_time = now
                elif now - self.lock_start_time >= self.config.lock_duration:
                    self._set_state(EntrainmentState.ENTRAINED)
            else:
                self.lock_start_time = None

        elif self.state == EntrainmentState.ENTRAINED:
            # Check for loss of lock
            phase_diff = self._circular_diff(user_phase, self.oscillators.psi)
            score = (math.cos(phase_diff) + 1) / 2

            if score < self.config.lock_threshold * 0.8:  # Hysteresis
                self._set_state(EntrainmentState.TRACKING)
                self.lock_start_time = None

    def _adapt_coupling(self, entrainment_score: float) -> None:
        """Adapt coupling strength based on current state."""
        # Error from target coherence
        coherence_error = self.config.target_coherence - self.oscillators.r

        # Adjust K to maintain target coherence
        delta_K = self.config.adaptation_rate * coherence_error

        # Also consider entrainment score
        if entrainment_score < 0.5:
            # If poorly entrained, increase coupling to help lock
            delta_K += self.config.adaptation_rate * 0.5

        self.oscillators.K += delta_K
        self.oscillators.K = max(self.config.min_coupling,
                                min(self.config.max_coupling, self.oscillators.K))

    def _circular_diff(self, phase1: float, phase2: float) -> float:
        """Calculate circular phase difference in [-π, π]."""
        diff = phase1 - phase2
        while diff > math.pi:
            diff -= TAU
        while diff < -math.pi:
            diff += TAU
        return diff

    def _set_state(self, new_state: EntrainmentState) -> None:
        """Set new state and trigger callback."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            if self._on_state_change:
                self._on_state_change(new_state)
            print(f"[Entrainment] State: {old_state.value} → {new_state.value}")

    def set_z_elevation(self, z: float) -> None:
        """Set the z-elevation for the oscillator bank."""
        self.oscillators.z = max(0, min(1, z))

    def set_target_coherence(self, target: float) -> None:
        """Set the target coherence level."""
        self.config.target_coherence = max(0, min(1, target))

    def request_release(self) -> None:
        """Request release from entrainment (for intentional decoupling)."""
        if self.state == EntrainmentState.ENTRAINED:
            self._set_state(EntrainmentState.RELEASING)
            # Temporarily reduce coupling
            self.oscillators.K *= 0.5

    def on_state_change(self, callback: Callable[[EntrainmentState], None]) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback

    def on_metrics_update(self, callback: Callable[[EntrainmentMetrics], None]) -> None:
        """Set callback for metrics updates."""
        self._on_metrics_update = callback

    def get_current_metrics(self) -> Optional[EntrainmentMetrics]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_entrainment_summary(self) -> dict:
        """Get summary statistics of the entrainment session."""
        if not self.metrics_history:
            return {}

        scores = [m.entrainment_score for m in self.metrics_history]
        lags = [m.lag_ms for m in self.metrics_history]
        coherences = [m.coherence for m in self.metrics_history]

        return {
            'mean_entrainment_score': sum(scores) / len(scores),
            'max_entrainment_score': max(scores),
            'mean_lag_ms': sum(lags) / len(lags),
            'min_lag_ms': min(lags),
            'mean_coherence': sum(coherences) / len(coherences),
            'samples': len(self.metrics_history),
            'duration_s': (self.metrics_history[-1].timestamp -
                          self.metrics_history[0].timestamp) if len(self.metrics_history) > 1 else 0
        }


# =============================================================================
# INTEGRATION WITH BIOSIGNAL MODULE
# =============================================================================

def create_integrated_coupler():
    """
    Create a fully integrated coupler with biosignal sources.

    Returns a tuple of (controller, biosignal_manager).
    """
    from coupler_synthesis.biosignal.biosignal_input import (
        MultisourceBiosignalManager,
        HRVSimulator,
        KeystrokeDynamicsAnalyzer
    )

    # Create components
    oscillators = KuramotoOscillatorBank(n=100)
    controller = BidirectionalEntrainmentController(oscillators)

    # Create biosignal manager
    manager = MultisourceBiosignalManager()
    manager.add_source('hrv', HRVSimulator(base_bpm=72), weight=1.0)
    manager.add_source('keystroke', KeystrokeDynamicsAnalyzer(), weight=0.5)

    # Connect biosignal to controller
    def on_unified_phase(phase: float, confidence: float):
        controller.process_user_phase(phase, confidence=confidence)

    manager.set_unified_callback(on_unified_phase)

    return controller, manager


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("BIDIRECTIONAL ENTRAINMENT CONTROLLER - Test")
    print("=" * 60)

    # Create controller
    oscillators = KuramotoOscillatorBank(n=50, seed=42)
    controller = BidirectionalEntrainmentController(oscillators)

    def on_state_change(state: EntrainmentState):
        print(f"  [STATE CHANGE] → {state.value}")

    controller.on_state_change(on_state_change)

    # Simulate user phases with gradual entrainment
    print("\n--- Simulating entrainment process ---")

    user_freq = 1.0  # User's natural frequency
    user_phase = 0.0
    system_initial_phase = oscillators.psi

    for i in range(100):
        # User phase advances at their natural frequency
        user_phase += user_freq * 0.05 + random.gauss(0, 0.02)
        user_phase %= TAU

        # After 30 iterations, user starts to naturally entrain
        if i > 30:
            # User is influenced by system output
            diff = controller._circular_diff(oscillators.psi, user_phase)
            user_phase += diff * 0.05  # User adjusts toward system
            user_phase %= TAU

        metrics = controller.process_user_phase(user_phase, confidence=0.9)

        if i % 10 == 0:
            print(f"  t={i:3d} | user_φ={metrics.user_phase:.2f} | "
                  f"sys_φ={metrics.system_phase:.2f} | "
                  f"score={metrics.entrainment_score:.2f} | "
                  f"r={metrics.coherence:.2f} | "
                  f"K={metrics.coupling_strength:.2f} | "
                  f"lag={metrics.lag_ms:.0f}ms")

        time.sleep(0.02)

    # Summary
    print("\n--- Entrainment Summary ---")
    summary = controller.get_entrainment_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\n[OK] Bidirectional entrainment test complete")
