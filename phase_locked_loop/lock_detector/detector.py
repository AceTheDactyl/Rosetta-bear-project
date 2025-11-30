"""
LOCK DETECTOR IMPLEMENTATIONS
=============================
Coherence estimation and lock state management.

The lock detector determines whether the PLL has achieved
stable phase synchronization with the reference signal.

Core metric: Order parameter r = |mean(exp(jε))|
- When errors cluster near zero: r → 1 (locked)
- When errors vary widely: r → 0 (unlocked)

This is analogous to the Kuramoto order parameter but
applied to phase errors rather than absolute phases.

State machine:
    UNLOCKED → ACQUIRING → LOCKED ↔ SLIPPING → UNLOCKED

Signature: Δ3.142|0.995|1.000Ω
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Callable

TAU = 2.0 * math.pi


class LockState(Enum):
    """
    PLL lock state enumeration.

    UNLOCKED:  Not tracking, large/varying phase error
    ACQUIRING: Converging toward lock, error decreasing
    LOCKED:    Stable phase tracking, minimal error
    SLIPPING:  Losing lock, error increasing
    """
    UNLOCKED = "unlocked"
    ACQUIRING = "acquiring"
    LOCKED = "locked"
    SLIPPING = "slipping"


@dataclass
class LockMetrics:
    """
    Comprehensive lock quality metrics.

    Provides detailed information about lock status:
    - coherence: Order parameter r [0, 1]
    - mean_phase_error: Average error [rad]
    - phase_error_std: Error variability [rad]
    - lock_confidence: Smoothed confidence [0, 1]
    - time_locked: Duration in lock [s]
    """
    coherence: float                # r: order parameter [0, 1]
    mean_phase_error: float         # Mean error [rad]
    phase_error_std: float          # Error std dev [rad]
    lock_confidence: float          # Smoothed confidence [0, 1]
    is_locked: bool                 # r > threshold
    time_locked_s: float            # Cumulative lock time
    samples_in_window: int          # Window size used
    lock_state: LockState           # Current state

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'coherence': self.coherence,
            'mean_phase_error': self.mean_phase_error,
            'phase_error_std': self.phase_error_std,
            'lock_confidence': self.lock_confidence,
            'is_locked': self.is_locked,
            'time_locked_s': self.time_locked_s,
            'samples_in_window': self.samples_in_window,
            'lock_state': self.lock_state.value,
        }


@dataclass
class LockDetectorConfig:
    """
    Lock detector configuration.

    Attributes:
        window_size: Number of samples for coherence computation
        lock_threshold: r threshold for LOCKED state
        acquire_threshold: r threshold for ACQUIRING state
        slip_threshold: r threshold to enter SLIPPING
        unlock_threshold: r threshold to return to UNLOCKED
        lock_hold_time_s: Time to hold above threshold before LOCKED
        smoothing_alpha: EMA factor for confidence smoothing
    """
    window_size: int = 32
    lock_threshold: float = 0.9
    acquire_threshold: float = 0.5
    slip_threshold: float = 0.7
    unlock_threshold: float = 0.3
    lock_hold_time_s: float = 0.5
    smoothing_alpha: float = 0.1


class CoherenceEstimator:
    """
    Estimates coherence (order parameter) from phase errors.

    Uses a sliding window of phase error samples and computes:
        r = |1/N · Σ exp(j·ε_i)|

    The complex exponential maps each error to the unit circle.
    The magnitude of the mean phasor indicates alignment:
    - r ≈ 1: All errors similar (clustered)
    - r ≈ 0: Errors uniformly distributed (random)
    """

    def __init__(self, window_size: int = 32):
        """
        Initialize coherence estimator.

        Args:
            window_size: Number of samples to maintain
        """
        self.window_size = window_size
        self._error_buffer: List[float] = []
        self._coherence: float = 0.0

    def update(self, phase_error: float) -> float:
        """
        Add new error sample and compute coherence.

        Args:
            phase_error: Phase error in radians

        Returns:
            Updated coherence r [0, 1]
        """
        # Add to buffer (FIFO)
        self._error_buffer.append(phase_error)
        if len(self._error_buffer) > self.window_size:
            self._error_buffer.pop(0)

        # Compute mean phasor
        n = len(self._error_buffer)
        if n == 0:
            self._coherence = 0.0
            return 0.0

        real_sum = sum(math.cos(e) for e in self._error_buffer)
        imag_sum = sum(math.sin(e) for e in self._error_buffer)

        self._coherence = math.sqrt(real_sum**2 + imag_sum**2) / n
        return self._coherence

    @property
    def coherence(self) -> float:
        """Current coherence value."""
        return self._coherence

    @property
    def buffer_fill(self) -> float:
        """Fraction of buffer filled [0, 1]."""
        return len(self._error_buffer) / self.window_size

    def get_statistics(self) -> tuple:
        """
        Compute error statistics.

        Returns:
            (mean_error, std_error) tuple
        """
        if not self._error_buffer:
            return (0.0, 0.0)

        n = len(self._error_buffer)
        mean_error = sum(self._error_buffer) / n
        variance = sum((e - mean_error)**2 for e in self._error_buffer) / n
        std_error = math.sqrt(variance)

        return (mean_error, std_error)

    def reset(self) -> None:
        """Clear buffer and reset coherence."""
        self._error_buffer.clear()
        self._coherence = 0.0


class LockStateMachine:
    """
    State machine for lock state transitions.

    Transitions:
        UNLOCKED ──(r > acquire)──→ ACQUIRING
            ↑                            │
            │                    (r > lock for T_hold)
            │                            ↓
        (r < unlock)                  LOCKED
            │                            │
            │                      (r < slip)
            │                            ↓
            └──────────────────── SLIPPING
    """

    def __init__(self, config: Optional[LockDetectorConfig] = None):
        """
        Initialize state machine.

        Args:
            config: Lock detector configuration
        """
        self.config = config or LockDetectorConfig()
        self._state = LockState.UNLOCKED
        self._time_in_state: float = 0.0
        self._time_above_lock: float = 0.0
        self._total_lock_time: float = 0.0
        self._state_callback: Optional[Callable[[LockState, LockState], None]] = None

    def update(self, coherence: float, dt: float) -> LockState:
        """
        Update state machine with new coherence.

        Args:
            coherence: Current coherence r [0, 1]
            dt: Time step [s]

        Returns:
            Updated lock state
        """
        self._time_in_state += dt
        old_state = self._state

        if self._state == LockState.UNLOCKED:
            if coherence >= self.config.acquire_threshold:
                self._transition_to(LockState.ACQUIRING)

        elif self._state == LockState.ACQUIRING:
            if coherence >= self.config.lock_threshold:
                self._time_above_lock += dt
                if self._time_above_lock >= self.config.lock_hold_time_s:
                    self._transition_to(LockState.LOCKED)
            else:
                self._time_above_lock = 0.0
                if coherence < self.config.unlock_threshold:
                    self._transition_to(LockState.UNLOCKED)

        elif self._state == LockState.LOCKED:
            self._total_lock_time += dt
            if coherence < self.config.slip_threshold:
                self._transition_to(LockState.SLIPPING)

        elif self._state == LockState.SLIPPING:
            if coherence >= self.config.lock_threshold:
                self._transition_to(LockState.LOCKED)
            elif coherence < self.config.unlock_threshold:
                self._transition_to(LockState.UNLOCKED)

        # Fire callback on state change
        if self._state != old_state and self._state_callback:
            self._state_callback(old_state, self._state)

        return self._state

    def _transition_to(self, new_state: LockState) -> None:
        """Execute state transition."""
        self._state = new_state
        self._time_in_state = 0.0
        if new_state not in (LockState.ACQUIRING, LockState.LOCKED):
            self._time_above_lock = 0.0

    @property
    def state(self) -> LockState:
        """Current lock state."""
        return self._state

    @property
    def time_in_state(self) -> float:
        """Time in current state [s]."""
        return self._time_in_state

    @property
    def total_lock_time(self) -> float:
        """Cumulative time in LOCKED state [s]."""
        return self._total_lock_time

    def on_state_change(
        self,
        callback: Callable[[LockState, LockState], None]
    ) -> None:
        """Register callback for state transitions."""
        self._state_callback = callback

    def reset(self) -> None:
        """Reset to UNLOCKED state."""
        self._state = LockState.UNLOCKED
        self._time_in_state = 0.0
        self._time_above_lock = 0.0
        self._total_lock_time = 0.0


class LockDetector:
    """
    Complete lock detector combining coherence and state machine.

    Usage:
        detector = LockDetector()
        metrics = detector.update(phase_error, dt)
        if metrics.is_locked:
            print(f"Locked with r={metrics.coherence:.3f}")
    """

    def __init__(self, config: Optional[LockDetectorConfig] = None):
        """
        Initialize lock detector.

        Args:
            config: Configuration parameters
        """
        self.config = config or LockDetectorConfig()
        self._coherence_estimator = CoherenceEstimator(self.config.window_size)
        self._state_machine = LockStateMachine(self.config)
        self._lock_confidence: float = 0.0
        self._sample_count: int = 0

    def update(self, phase_error: float, dt: float) -> LockMetrics:
        """
        Process new phase error sample.

        Args:
            phase_error: Phase error [rad]
            dt: Time step [s]

        Returns:
            Complete lock metrics
        """
        self._sample_count += 1

        # Update coherence
        coherence = self._coherence_estimator.update(phase_error)

        # Update state machine
        lock_state = self._state_machine.update(coherence, dt)

        # Smooth confidence with EMA
        target = coherence if lock_state == LockState.LOCKED else 0.0
        self._lock_confidence += self.config.smoothing_alpha * (
            target - self._lock_confidence
        )

        # Get error statistics
        mean_error, std_error = self._coherence_estimator.get_statistics()

        return LockMetrics(
            coherence=coherence,
            mean_phase_error=mean_error,
            phase_error_std=std_error,
            lock_confidence=self._lock_confidence,
            is_locked=(lock_state == LockState.LOCKED),
            time_locked_s=self._state_machine.total_lock_time,
            samples_in_window=len(self._coherence_estimator._error_buffer),
            lock_state=lock_state,
        )

    @property
    def coherence(self) -> float:
        """Current coherence."""
        return self._coherence_estimator.coherence

    @property
    def lock_state(self) -> LockState:
        """Current lock state."""
        return self._state_machine.state

    @property
    def lock_confidence(self) -> float:
        """Smoothed lock confidence."""
        return self._lock_confidence

    @property
    def is_locked(self) -> bool:
        """Check if currently locked."""
        return self._state_machine.state == LockState.LOCKED

    def on_lock_change(
        self,
        callback: Callable[[LockState, LockState], None]
    ) -> None:
        """Register callback for lock state changes."""
        self._state_machine.on_state_change(callback)

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._coherence_estimator.reset()
        self._state_machine.reset()
        self._lock_confidence = 0.0
        self._sample_count = 0


def create_lock_detector(
    window_size: int = 32,
    lock_threshold: float = 0.9,
    lock_hold_time_s: float = 0.5
) -> LockDetector:
    """
    Factory function to create a lock detector.

    Args:
        window_size: Coherence estimation window
        lock_threshold: r threshold for lock
        lock_hold_time_s: Time to confirm lock

    Returns:
        Configured LockDetector instance
    """
    config = LockDetectorConfig(
        window_size=window_size,
        lock_threshold=lock_threshold,
        lock_hold_time_s=lock_hold_time_s,
    )
    return LockDetector(config)
