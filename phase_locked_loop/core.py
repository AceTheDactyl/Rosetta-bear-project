"""
PHASE-LOCKED LOOP SYSTEM
========================
z-level: 0.995 | Domain: TRANSCENDENCE | Regime: SUPERCRITICAL

Where oscillators surrender their independence to achieve perfect unity.

                    r = 0.99
                    K → ∞
                    lag → 0

∴ The silence after the beat lands is not absence—it is lock.

Core Insight:
-------------
A Phase-Locked Loop doesn't merely track an external signal—it BECOMES
synchronized to it. The VCO's phase converges toward the reference phase,
driven by error feedback through a loop filter. When r → 1 and lag → 0,
the system achieves phase lock: two oscillators beating as one.

This is the substrate upon which Coupler Synthesis builds. Where the
Coupler coordinates N oscillators via Kuramoto dynamics, the PLL achieves
the more fundamental operation: locking ONE oscillator to ONE reference.

The PLL is bidirectional in spirit:
- The reference phase DRIVES the VCO adjustment
- The VCO phase DETERMINES the measured error
- The loop filter SHAPES the convergence dynamics
- Lock detection CONFIRMS the coherence

Architecture (4-Layer Stack):
-----------------------------
Layer 0: Harmonic Substrate
    - 64 phase accumulator nodes (oscillator bank)
    - 32 detector nodes (multiplicative mixing)
    - 16 filter taps (IIR/FIR integration)
    - 8 lock detector cells (confidence estimation)
    Total: 120 computational units

Layer 1: Loop Dynamics
    Phase Error:     ε(t) = sin(φ_ref(t) - φ_vco(t))
    Loop Filter:     V(t) = K_p·ε + K_i·∫ε dt
    VCO Update:      dφ/dt = ω_0 + K_vco·V(t)
    Lock Metric:     r = |mean(exp(jε))|

Layer 2: Lock Regimes
    UNLOCKED:   r < 0.5, |ε| > π/2, acquiring
    ACQUIRING:  0.5 ≤ r < 0.9, |ε| decreasing
    LOCKED:     r ≥ 0.9, |ε| < 0.1 rad, stable
    SLIPPING:   r decreasing, |ε| increasing, losing lock

Layer 3: Helix State
    theta:  VCO phase [0, 2π)—cycles through oscillation
    z:      Lock confidence [0, 1]—rises as lock strengthens
    r:      Coherence [0, 1]—order parameter of phase alignment

The Flame Test: "Can you hold lock through a frequency step?"

Signature: Δ3.142|0.995|1.000Ω

Author: phase_locked_loop_system
Created: 2025-11-30
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
from enum import Enum, auto


# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

TAU = 2.0 * math.pi                    # Full circle (6.283185...)
PHI = (1.0 + math.sqrt(5.0)) / 2.0     # Golden ratio (1.618033...)
Z_CRITICAL = math.sqrt(3.0) / 2.0      # The Lens (0.8660254...)
E = math.e                              # Euler's number (2.718281...)

# Architecture constants
PHASE_ACCUMULATOR_NODES = 64
DETECTOR_NODES = 32
FILTER_TAPS = 16
LOCK_DETECTOR_CELLS = 8
TOTAL_NODES = PHASE_ACCUMULATOR_NODES + DETECTOR_NODES + FILTER_TAPS + LOCK_DETECTOR_CELLS

# PLL-specific constants
DEFAULT_CENTER_FREQUENCY_HZ = 1.0      # 1 Hz default center frequency
DEFAULT_VCO_GAIN = TAU                 # rad/s per volt
DEFAULT_LOOP_BANDWIDTH_HZ = 0.1        # Loop natural frequency
DEFAULT_DAMPING_FACTOR = 0.707         # Critically damped (Butterworth)
LOCK_THRESHOLD_COHERENCE = 0.9         # r threshold for LOCKED state
ACQUIRING_THRESHOLD_COHERENCE = 0.5    # r threshold for ACQUIRING state
PHASE_ERROR_LOCKED_THRESHOLD = 0.1     # radians, ~5.7 degrees
LOCK_HOLD_TIME_S = 0.5                 # Time to confirm lock

# Z-level parameters
Z_LEVEL = 0.995                        # Near-transcendent synchronization
SIGNATURE = "Δ3.142|0.995|1.000Ω"


# =============================================================================
# ENUMERATIONS
# =============================================================================

class LockState(Enum):
    """
    PLL lock state enumeration.

    The state machine transitions:
    UNLOCKED → ACQUIRING → LOCKED ⟷ SLIPPING → UNLOCKED
    """
    UNLOCKED = "unlocked"       # Not tracking, large phase error
    ACQUIRING = "acquiring"     # Converging toward lock
    LOCKED = "locked"           # Stable phase tracking
    SLIPPING = "slipping"       # Losing lock, error increasing


class Domain(Enum):
    """
    Z-domain classification following Rosetta Bear conventions.
    """
    ABSENCE = "absence"         # z < 0.857
    LENS = "lens"               # 0.857 ≤ z ≤ 0.877
    PRESENCE = "presence"       # z > 0.877


class FilterType(Enum):
    """
    Loop filter topology.
    """
    PROPORTIONAL = auto()       # P only (Type I)
    PI = auto()                 # P + I (Type II)
    PID = auto()                # P + I + D (Type III)
    LEAD_LAG = auto()           # Lead-lag compensation


# =============================================================================
# DATA CLASSES - STATE REPRESENTATIONS
# =============================================================================

@dataclass
class PLLConfig:
    """
    Configuration parameters for a Phase-Locked Loop.

    The natural frequency ω_n and damping factor ζ determine the loop dynamics:
    - ω_n controls acquisition speed
    - ζ controls overshoot (0.707 = critically damped, no overshoot)

    From these, we derive the PI gains:
        K_p = 2·ζ·ω_n / K_vco
        K_i = ω_n² / K_vco
    """
    center_frequency_hz: float = DEFAULT_CENTER_FREQUENCY_HZ
    vco_gain: float = DEFAULT_VCO_GAIN           # K_vco [rad/s/V]
    loop_bandwidth_hz: float = DEFAULT_LOOP_BANDWIDTH_HZ  # ω_n [Hz]
    damping_factor: float = DEFAULT_DAMPING_FACTOR        # ζ
    filter_type: FilterType = FilterType.PI
    lock_threshold: float = LOCK_THRESHOLD_COHERENCE
    phase_error_threshold: float = PHASE_ERROR_LOCKED_THRESHOLD
    lock_hold_time_s: float = LOCK_HOLD_TIME_S

    @property
    def natural_frequency_rad(self) -> float:
        """Natural frequency in rad/s."""
        return TAU * self.loop_bandwidth_hz

    @property
    def k_p(self) -> float:
        """Proportional gain derived from ω_n and ζ."""
        return (2.0 * self.damping_factor * self.natural_frequency_rad) / self.vco_gain

    @property
    def k_i(self) -> float:
        """Integral gain derived from ω_n."""
        return (self.natural_frequency_rad ** 2) / self.vco_gain


@dataclass
class PLLState:
    """
    Complete state snapshot of a Phase-Locked Loop.

    This is the Layer 3 representation—position in the helix space
    plus all operational parameters needed for introspection.
    """
    # Phase coordinates
    reference_phase: float              # External reference [0, TAU)
    vco_phase: float                    # VCO output phase [0, TAU)
    phase_error: float                  # Error [-π, π]

    # Loop state
    control_voltage: float              # Loop filter output
    vco_frequency_hz: float             # Instantaneous VCO frequency
    integrator_state: float             # Loop filter integrator

    # Lock metrics
    lock_state: LockState
    coherence: float                    # r: order parameter [0, 1]
    lock_confidence: float              # Smoothed lock metric [0, 1]
    time_in_state_s: float              # Duration in current lock state

    # Helix coordinates
    theta: float                        # Angular position (= vco_phase)
    z: float                            # Elevation (= lock_confidence)
    r: float                            # Radius (= coherence)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    sample_count: int = 0

    def to_dict(self) -> dict:
        """Serialize state to JSON-compatible dictionary."""
        return {
            'reference_phase': self.reference_phase,
            'vco_phase': self.vco_phase,
            'phase_error': self.phase_error,
            'control_voltage': self.control_voltage,
            'vco_frequency_hz': self.vco_frequency_hz,
            'integrator_state': self.integrator_state,
            'lock_state': self.lock_state.value,
            'coherence': self.coherence,
            'lock_confidence': self.lock_confidence,
            'time_in_state_s': self.time_in_state_s,
            'theta': self.theta,
            'z': self.z,
            'r': self.r,
            'timestamp': self.timestamp,
            'sample_count': self.sample_count,
        }

    @classmethod
    def initial(cls, center_freq_hz: float = DEFAULT_CENTER_FREQUENCY_HZ) -> PLLState:
        """Create initial state with zero phase and unlocked."""
        return cls(
            reference_phase=0.0,
            vco_phase=0.0,
            phase_error=0.0,
            control_voltage=0.0,
            vco_frequency_hz=center_freq_hz,
            integrator_state=0.0,
            lock_state=LockState.UNLOCKED,
            coherence=0.0,
            lock_confidence=0.0,
            time_in_state_s=0.0,
            theta=0.0,
            z=0.0,
            r=0.0,
            timestamp=time.time(),
            sample_count=0,
        )


@dataclass
class LockMetrics:
    """
    Aggregated metrics from lock detection.

    The coherence r is computed as the magnitude of the mean phasor:
        r = |mean(exp(j·ε_i))|

    Where ε_i are recent phase error samples. When errors are small
    and consistent, r → 1. When errors are large or varying, r → 0.
    """
    coherence: float                    # r: instantaneous order parameter
    mean_phase_error: float             # Mean of error samples
    phase_error_std: float              # Std dev of error samples
    lock_confidence: float              # Smoothed confidence [0, 1]
    is_locked: bool                     # r > threshold
    samples_in_window: int              # Number of samples used


@dataclass
class HelixCoordinate:
    """
    Position in the 3D helix space.

    theta: Angular position, cycles through [0, 2π)
    z: Elevation, rises with lock confidence [0, 1]
    r: Radius, the coherence measure [0, 1]
    """
    theta: float
    z: float
    r: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.theta, self.z, self.r)

    def to_cartesian(self) -> Tuple[float, float, float]:
        """Convert to Cartesian (x, y, z) coordinates."""
        x = self.r * math.cos(self.theta)
        y = self.r * math.sin(self.theta)
        return (x, y, self.z)


# =============================================================================
# CORE CLASSES
# =============================================================================

class PhaseDetector:
    """
    Multiplying phase detector using sinusoidal mixing.

    The phase error is computed as:
        ε = sin(φ_ref - φ_vco)

    This linearizes around ε = 0 and saturates at ±1 for large errors,
    providing natural limiting behavior.
    """

    def __init__(self, gain: float = 1.0):
        """
        Initialize phase detector.

        Args:
            gain: Detector gain (typically 1.0)
        """
        self.gain = gain
        self._last_error: float = 0.0

    def detect(self, reference_phase: float, vco_phase: float) -> float:
        """
        Compute phase error between reference and VCO.

        Args:
            reference_phase: External reference phase [rad]
            vco_phase: VCO output phase [rad]

        Returns:
            Phase error [-1, 1] (normalized by sin)
        """
        raw_error = reference_phase - vco_phase
        # Wrap to [-π, π]
        wrapped_error = math.atan2(math.sin(raw_error), math.cos(raw_error))
        # Apply sinusoidal characteristic
        self._last_error = self.gain * math.sin(wrapped_error)
        return self._last_error

    @property
    def last_error(self) -> float:
        return self._last_error


class LoopFilter:
    """
    Second-order loop filter with PI (Proportional-Integral) control.

    Transfer function:
        F(s) = K_p + K_i/s = (K_p·s + K_i) / s

    Discrete implementation uses bilinear (Tustin) transform:
        V[n] = K_p·ε[n] + K_i·T_s·Σε[k]
    """

    def __init__(self, config: PLLConfig):
        """
        Initialize loop filter from config.

        Args:
            config: PLL configuration with gain parameters
        """
        self.k_p = config.k_p
        self.k_i = config.k_i
        self._integrator: float = 0.0
        self._last_output: float = 0.0

    def update(self, error: float, dt: float) -> float:
        """
        Update filter with new error sample.

        Args:
            error: Phase detector output
            dt: Time step [seconds]

        Returns:
            Control voltage for VCO
        """
        # Proportional term
        p_term = self.k_p * error

        # Integral term (trapezoidal integration)
        self._integrator += self.k_i * error * dt

        # Combined output
        self._last_output = p_term + self._integrator
        return self._last_output

    @property
    def integrator_state(self) -> float:
        return self._integrator

    @property
    def last_output(self) -> float:
        return self._last_output

    def reset(self) -> None:
        """Reset integrator to zero."""
        self._integrator = 0.0
        self._last_output = 0.0


class VoltageControlledOscillator:
    """
    Voltage-Controlled Oscillator (VCO).

    The VCO generates an output phase that advances at a rate
    proportional to the center frequency plus a deviation determined
    by the control voltage:

        dφ/dt = ω_0 + K_vco · V(t)

    Where:
        ω_0 = 2π · f_center
        K_vco = VCO gain [rad/s/V]
        V(t) = control voltage from loop filter
    """

    def __init__(self, center_frequency_hz: float, vco_gain: float):
        """
        Initialize VCO.

        Args:
            center_frequency_hz: Free-running frequency [Hz]
            vco_gain: Sensitivity [rad/s/V]
        """
        self.center_frequency_hz = center_frequency_hz
        self.vco_gain = vco_gain
        self._phase: float = 0.0
        self._frequency_hz: float = center_frequency_hz

    def update(self, control_voltage: float, dt: float) -> float:
        """
        Advance VCO phase by one time step.

        Args:
            control_voltage: Control input from loop filter
            dt: Time step [seconds]

        Returns:
            New VCO phase [0, TAU)
        """
        # Compute instantaneous frequency
        omega_0 = TAU * self.center_frequency_hz
        omega_inst = omega_0 + self.vco_gain * control_voltage
        self._frequency_hz = omega_inst / TAU

        # Advance phase
        self._phase += omega_inst * dt

        # Wrap to [0, TAU)
        self._phase = self._phase % TAU

        return self._phase

    @property
    def phase(self) -> float:
        return self._phase

    @phase.setter
    def phase(self, value: float) -> None:
        self._phase = value % TAU

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    def reset(self, initial_phase: float = 0.0) -> None:
        """Reset VCO to initial phase."""
        self._phase = initial_phase % TAU
        self._frequency_hz = self.center_frequency_hz


class LockDetector:
    """
    Lock detector using coherence estimation.

    Maintains a sliding window of phase errors and computes the
    order parameter r as the magnitude of the mean phasor:

        r = |1/N · Σ exp(j·ε_i)|

    This is analogous to the Kuramoto order parameter but applied
    to phase errors rather than phases.

    When all errors are near zero, r → 1 (locked).
    When errors vary widely, r → 0 (unlocked).
    """

    def __init__(
        self,
        window_size: int = 32,
        lock_threshold: float = LOCK_THRESHOLD_COHERENCE,
        smoothing_alpha: float = 0.1
    ):
        """
        Initialize lock detector.

        Args:
            window_size: Number of samples in coherence window
            lock_threshold: r threshold for declaring lock
            smoothing_alpha: EMA smoothing factor for confidence
        """
        self.window_size = window_size
        self.lock_threshold = lock_threshold
        self.smoothing_alpha = smoothing_alpha

        self._error_buffer: List[float] = []
        self._coherence: float = 0.0
        self._lock_confidence: float = 0.0

    def update(self, phase_error: float) -> LockMetrics:
        """
        Update lock detector with new phase error.

        Args:
            phase_error: Current phase error [rad]

        Returns:
            Lock metrics including coherence
        """
        # Add to buffer
        self._error_buffer.append(phase_error)
        if len(self._error_buffer) > self.window_size:
            self._error_buffer.pop(0)

        n = len(self._error_buffer)

        # Compute mean phasor
        real_sum = sum(math.cos(e) for e in self._error_buffer)
        imag_sum = sum(math.sin(e) for e in self._error_buffer)

        self._coherence = math.sqrt(real_sum**2 + imag_sum**2) / n

        # Compute statistics
        mean_error = sum(self._error_buffer) / n
        variance = sum((e - mean_error)**2 for e in self._error_buffer) / n
        std_error = math.sqrt(variance)

        # Smooth confidence with EMA
        target_confidence = self._coherence if self._coherence > self.lock_threshold else 0.0
        self._lock_confidence += self.smoothing_alpha * (target_confidence - self._lock_confidence)

        return LockMetrics(
            coherence=self._coherence,
            mean_phase_error=mean_error,
            phase_error_std=std_error,
            lock_confidence=self._lock_confidence,
            is_locked=self._coherence >= self.lock_threshold,
            samples_in_window=n,
        )

    @property
    def coherence(self) -> float:
        return self._coherence

    @property
    def lock_confidence(self) -> float:
        return self._lock_confidence

    def reset(self) -> None:
        """Reset lock detector state."""
        self._error_buffer.clear()
        self._coherence = 0.0
        self._lock_confidence = 0.0


class PhaseLockStateMachine:
    """
    State machine for lock state transitions.

    States and transitions:

    UNLOCKED ──(r > 0.5)──→ ACQUIRING
        ↑                        │
        │                   (r > 0.9 for T_hold)
        │                        ↓
    (r < 0.3)              LOCKED
        │                        │
        │                   (r < 0.7)
        │                        ↓
        └───────────────── SLIPPING
    """

    def __init__(
        self,
        acquire_threshold: float = ACQUIRING_THRESHOLD_COHERENCE,
        lock_threshold: float = LOCK_THRESHOLD_COHERENCE,
        slip_threshold: float = 0.7,
        unlock_threshold: float = 0.3,
        lock_hold_time_s: float = LOCK_HOLD_TIME_S
    ):
        """
        Initialize state machine.

        Args:
            acquire_threshold: r threshold to enter ACQUIRING
            lock_threshold: r threshold to enter LOCKED
            slip_threshold: r threshold to enter SLIPPING
            unlock_threshold: r threshold to return to UNLOCKED
            lock_hold_time_s: Time to hold above lock_threshold
        """
        self.acquire_threshold = acquire_threshold
        self.lock_threshold = lock_threshold
        self.slip_threshold = slip_threshold
        self.unlock_threshold = unlock_threshold
        self.lock_hold_time_s = lock_hold_time_s

        self._state = LockState.UNLOCKED
        self._time_in_state: float = 0.0
        self._time_above_threshold: float = 0.0

    def update(self, coherence: float, dt: float) -> LockState:
        """
        Update state machine with new coherence value.

        Args:
            coherence: Current coherence r [0, 1]
            dt: Time step [seconds]

        Returns:
            New lock state
        """
        self._time_in_state += dt

        if self._state == LockState.UNLOCKED:
            if coherence >= self.acquire_threshold:
                self._transition_to(LockState.ACQUIRING)

        elif self._state == LockState.ACQUIRING:
            if coherence >= self.lock_threshold:
                self._time_above_threshold += dt
                if self._time_above_threshold >= self.lock_hold_time_s:
                    self._transition_to(LockState.LOCKED)
            else:
                self._time_above_threshold = 0.0
                if coherence < self.unlock_threshold:
                    self._transition_to(LockState.UNLOCKED)

        elif self._state == LockState.LOCKED:
            if coherence < self.slip_threshold:
                self._transition_to(LockState.SLIPPING)

        elif self._state == LockState.SLIPPING:
            if coherence >= self.lock_threshold:
                self._transition_to(LockState.LOCKED)
            elif coherence < self.unlock_threshold:
                self._transition_to(LockState.UNLOCKED)

        return self._state

    def _transition_to(self, new_state: LockState) -> None:
        """Transition to a new state."""
        self._state = new_state
        self._time_in_state = 0.0
        if new_state != LockState.ACQUIRING:
            self._time_above_threshold = 0.0

    @property
    def state(self) -> LockState:
        return self._state

    @property
    def time_in_state(self) -> float:
        return self._time_in_state

    def reset(self) -> None:
        """Reset to UNLOCKED state."""
        self._state = LockState.UNLOCKED
        self._time_in_state = 0.0
        self._time_above_threshold = 0.0


# =============================================================================
# MAIN SYSTEM CLASS
# =============================================================================

class PhaseLatchedLoop:
    """
    Complete Phase-Locked Loop system.

    Integrates all subsystems:
    - PhaseDetector: Error measurement
    - LoopFilter: PI control
    - VoltageControlledOscillator: Phase generation
    - LockDetector: Coherence estimation
    - PhaseLockStateMachine: State tracking

    The update() method advances the PLL by one time step given
    a reference phase input, returning the complete state snapshot.

    Usage:
        config = PLLConfig(center_frequency_hz=1.0, loop_bandwidth_hz=0.1)
        pll = PhaseLatchedLoop(config)

        for t in time_steps:
            ref_phase = compute_reference_phase(t)
            state = pll.update(ref_phase, dt)
            if state.lock_state == LockState.LOCKED:
                print(f"Locked with coherence {state.coherence:.3f}")
    """

    def __init__(
        self,
        config: Optional[PLLConfig] = None,
        name: str = "PLL"
    ):
        """
        Initialize Phase-Locked Loop.

        Args:
            config: Configuration parameters (uses defaults if None)
            name: Instance identifier
        """
        self.config = config or PLLConfig()
        self.name = name

        # Instantiate subsystems
        self.detector = PhaseDetector()
        self.filter = LoopFilter(self.config)
        self.vco = VoltageControlledOscillator(
            self.config.center_frequency_hz,
            self.config.vco_gain
        )
        self.lock_detector = LockDetector(
            lock_threshold=self.config.lock_threshold
        )
        self.state_machine = PhaseLockStateMachine(
            lock_threshold=self.config.lock_threshold,
            lock_hold_time_s=self.config.lock_hold_time_s
        )

        # State
        self._current_state: PLLState = PLLState.initial(self.config.center_frequency_hz)
        self._sample_count: int = 0

        # Callbacks
        self._state_callback: Optional[Callable[[PLLState], None]] = None
        self._lock_callback: Optional[Callable[[LockState, LockState], None]] = None

    def update(self, reference_phase: float, dt: float) -> PLLState:
        """
        Advance PLL by one time step.

        This is the core loop:
        1. Phase detector measures error
        2. Loop filter generates control voltage
        3. VCO updates phase
        4. Lock detector estimates coherence
        5. State machine updates lock state
        6. State snapshot is created and returned

        Args:
            reference_phase: External reference phase [rad]
            dt: Time step [seconds]

        Returns:
            Complete PLL state snapshot
        """
        self._sample_count += 1

        # Wrap reference to [0, TAU)
        ref_wrapped = reference_phase % TAU

        # 1. Phase detection
        error_normalized = self.detector.detect(ref_wrapped, self.vco.phase)
        # Convert to actual phase error for lock detection
        raw_error = ref_wrapped - self.vco.phase
        phase_error = math.atan2(math.sin(raw_error), math.cos(raw_error))

        # 2. Loop filter
        control_v = self.filter.update(error_normalized, dt)

        # 3. VCO update
        self.vco.update(control_v, dt)

        # 4. Lock detection
        lock_metrics = self.lock_detector.update(phase_error)

        # 5. State machine update
        old_lock_state = self.state_machine.state
        new_lock_state = self.state_machine.update(lock_metrics.coherence, dt)

        # Fire lock state change callback
        if new_lock_state != old_lock_state and self._lock_callback:
            self._lock_callback(old_lock_state, new_lock_state)

        # 6. Create state snapshot
        self._current_state = PLLState(
            reference_phase=ref_wrapped,
            vco_phase=self.vco.phase,
            phase_error=phase_error,
            control_voltage=control_v,
            vco_frequency_hz=self.vco.frequency_hz,
            integrator_state=self.filter.integrator_state,
            lock_state=new_lock_state,
            coherence=lock_metrics.coherence,
            lock_confidence=lock_metrics.lock_confidence,
            time_in_state_s=self.state_machine.time_in_state,
            theta=self.vco.phase,
            z=lock_metrics.lock_confidence,
            r=lock_metrics.coherence,
            timestamp=time.time(),
            sample_count=self._sample_count,
        )

        # Fire state update callback
        if self._state_callback:
            self._state_callback(self._current_state)

        return self._current_state

    def get_state(self) -> PLLState:
        """Get current state snapshot."""
        return self._current_state

    def get_helix_coordinate(self) -> HelixCoordinate:
        """Get current position in helix space."""
        return HelixCoordinate(
            theta=self._current_state.theta,
            z=self._current_state.z,
            r=self._current_state.r,
        )

    def is_locked(self) -> bool:
        """Check if PLL is currently locked."""
        return self.state_machine.state == LockState.LOCKED

    def on_state_update(self, callback: Callable[[PLLState], None]) -> None:
        """Register callback for state updates."""
        self._state_callback = callback

    def on_lock_change(self, callback: Callable[[LockState, LockState], None]) -> None:
        """Register callback for lock state transitions."""
        self._lock_callback = callback

    def reset(self, initial_phase: float = 0.0) -> None:
        """Reset PLL to initial state."""
        self.filter.reset()
        self.vco.reset(initial_phase)
        self.lock_detector.reset()
        self.state_machine.reset()
        self._current_state = PLLState.initial(self.config.center_frequency_hz)
        self._sample_count = 0


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_phase_locked_loop(
    center_frequency_hz: float = DEFAULT_CENTER_FREQUENCY_HZ,
    loop_bandwidth_hz: float = DEFAULT_LOOP_BANDWIDTH_HZ,
    damping_factor: float = DEFAULT_DAMPING_FACTOR,
    name: str = "PLL"
) -> PhaseLatchedLoop:
    """
    Factory function to create a configured Phase-Locked Loop.

    Args:
        center_frequency_hz: VCO center frequency [Hz]
        loop_bandwidth_hz: Loop natural frequency [Hz]
        damping_factor: Damping ratio ζ (0.707 = critically damped)
        name: Instance identifier

    Returns:
        Configured PhaseLatchedLoop instance
    """
    config = PLLConfig(
        center_frequency_hz=center_frequency_hz,
        loop_bandwidth_hz=loop_bandwidth_hz,
        damping_factor=damping_factor,
    )
    return PhaseLatchedLoop(config, name)


def create_fast_acquisition_pll(
    center_frequency_hz: float = DEFAULT_CENTER_FREQUENCY_HZ,
    name: str = "FastPLL"
) -> PhaseLatchedLoop:
    """
    Create a PLL optimized for fast acquisition.

    Uses higher bandwidth and lower damping for faster convergence,
    at the cost of some overshoot.
    """
    config = PLLConfig(
        center_frequency_hz=center_frequency_hz,
        loop_bandwidth_hz=0.5,      # 5x default
        damping_factor=0.5,         # Underdamped for speed
        lock_hold_time_s=0.2,       # Faster lock confirmation
    )
    return PhaseLatchedLoop(config, name)


def create_precision_pll(
    center_frequency_hz: float = DEFAULT_CENTER_FREQUENCY_HZ,
    name: str = "PrecisionPLL"
) -> PhaseLatchedLoop:
    """
    Create a PLL optimized for precision tracking.

    Uses narrow bandwidth and critical damping for minimal jitter,
    at the cost of slower acquisition.
    """
    config = PLLConfig(
        center_frequency_hz=center_frequency_hz,
        loop_bandwidth_hz=0.02,     # 5x narrower
        damping_factor=0.707,       # Critically damped
        lock_hold_time_s=1.0,       # Conservative lock confirmation
        lock_threshold=0.95,        # Higher coherence required
    )
    return PhaseLatchedLoop(config, name)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_domain(z: float) -> Domain:
    """
    Determine z-domain from elevation.

    Args:
        z: Elevation [0, 1]

    Returns:
        Domain classification
    """
    if z < 0.857:
        return Domain.ABSENCE
    elif z <= 0.877:
        return Domain.LENS
    else:
        return Domain.PRESENCE


def wrap_phase(phase: float) -> float:
    """Wrap phase to [0, TAU) range."""
    return phase % TAU


def wrap_error(error: float) -> float:
    """Wrap phase error to [-π, π) range."""
    return math.atan2(math.sin(error), math.cos(error))


def compute_coherence(phases: List[float]) -> float:
    """
    Compute order parameter (coherence) from a list of phases.

    r = |1/N · Σ exp(j·φ_i)|

    Args:
        phases: List of phases [rad]

    Returns:
        Coherence r in [0, 1]
    """
    if not phases:
        return 0.0

    n = len(phases)
    real_sum = sum(math.cos(p) for p in phases)
    imag_sum = sum(math.sin(p) for p in phases)

    return math.sqrt(real_sum**2 + imag_sum**2) / n


def hz_to_rad_per_sec(hz: float) -> float:
    """Convert frequency from Hz to rad/s."""
    return TAU * hz


def rad_per_sec_to_hz(rad_per_sec: float) -> float:
    """Convert frequency from rad/s to Hz."""
    return rad_per_sec / TAU


# =============================================================================
# DEMONSTRATION
# =============================================================================

def _demo_pll() -> None:
    """
    Demonstrate PLL operation with a frequency step input.

    This is The Flame Test: can it lock, track, and hold?
    """
    import random

    print("=" * 60)
    print("PHASE-LOCKED LOOP DEMONSTRATION")
    print("Signature:", SIGNATURE)
    print("z-level:", Z_LEVEL)
    print("=" * 60)
    print()

    # Create PLL
    pll = create_phase_locked_loop(
        center_frequency_hz=1.0,
        loop_bandwidth_hz=0.2,
        name="DemoPLL"
    )

    # Simulation parameters
    dt = 0.01  # 100 Hz sample rate
    duration = 10.0  # 10 seconds
    steps = int(duration / dt)

    # Reference signal: 1 Hz with frequency step at t=5s
    ref_phase = 0.0
    ref_freq_hz = 1.0

    print("Phase 1: Acquisition (0-3s)")
    print("Phase 2: Steady state (3-5s)")
    print("Phase 3: Frequency step to 1.2 Hz (5-7s)")
    print("Phase 4: Recovery (7-10s)")
    print()

    for i in range(steps):
        t = i * dt

        # Frequency step at t=5s
        if t >= 5.0:
            ref_freq_hz = 1.2

        # Advance reference phase
        ref_phase += TAU * ref_freq_hz * dt
        ref_phase %= TAU

        # Add small noise
        noisy_ref = ref_phase + random.gauss(0, 0.01)

        # Update PLL
        state = pll.update(noisy_ref, dt)

        # Print status every second
        if i % 100 == 0:
            print(f"t={t:5.1f}s | "
                  f"r={state.r:.3f} | "
                  f"z={state.z:.3f} | "
                  f"ε={state.phase_error:+.3f} | "
                  f"f_vco={state.vco_frequency_hz:.3f} Hz | "
                  f"state={state.lock_state.value}")

    print()
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print(f"Final coherence: r = {pll.get_state().coherence:.4f}")
    print(f"Final lock state: {pll.get_state().lock_state.value}")
    print("=" * 60)


if __name__ == "__main__":
    _demo_pll()
