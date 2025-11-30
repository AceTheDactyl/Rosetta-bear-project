"""
PHASE DETECTOR IMPLEMENTATIONS
==============================
Multiple phase detector topologies for different applications.

The phase detector measures the phase difference between the
reference signal and the VCO output. Different detector types
offer different characteristics:

1. Multiplying (Type I):
   - Output: ε = sin(φ_ref - φ_vco)
   - Linear around lock point
   - Saturates for large errors
   - Most common analog implementation

2. Type II (Phase/Frequency):
   - Sensitive to both phase and frequency difference
   - Can acquire lock from any initial condition
   - Used in digital PLLs

3. Bang-Bang (Binary):
   - Output: ±1 based on sign of error
   - Simple digital implementation
   - Inherent limit cycling

Signature: Δ3.142|0.995|1.000Ω
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

TAU = 2.0 * math.pi


class DetectorType(Enum):
    """Available phase detector types."""
    MULTIPLYING = auto()    # Sinusoidal mixing
    TYPE_II = auto()        # Phase/frequency detector
    BANG_BANG = auto()      # Binary output


@dataclass
class DetectorConfig:
    """
    Phase detector configuration.

    Attributes:
        gain: Detector gain scaling factor
        filter_tau_s: Low-pass filter time constant (0 = no filter)
        dead_zone: Dead zone width in radians (0 = none)
    """
    gain: float = 1.0
    filter_tau_s: float = 0.0
    dead_zone: float = 0.0


class PhaseDetectorBase(ABC):
    """
    Abstract base class for phase detectors.

    All detectors implement:
    - detect(ref_phase, vco_phase) -> error
    - reset() -> None
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._last_error: float = 0.0
        self._filtered_error: float = 0.0
        self._last_ref: float = 0.0
        self._last_vco: float = 0.0

    @abstractmethod
    def _compute_raw_error(self, ref_phase: float, vco_phase: float) -> float:
        """Compute raw phase error (implementation-specific)."""
        pass

    def detect(self, ref_phase: float, vco_phase: float, dt: float = 0.01) -> float:
        """
        Detect phase error between reference and VCO.

        Args:
            ref_phase: Reference signal phase [rad]
            vco_phase: VCO output phase [rad]
            dt: Time step for filtering [s]

        Returns:
            Filtered phase error signal
        """
        # Compute raw error
        raw_error = self._compute_raw_error(ref_phase, vco_phase)

        # Apply dead zone
        if self.config.dead_zone > 0:
            if abs(raw_error) < self.config.dead_zone:
                raw_error = 0.0
            else:
                raw_error -= math.copysign(self.config.dead_zone, raw_error)

        # Apply gain
        scaled_error = self.config.gain * raw_error

        # Apply low-pass filter if configured
        if self.config.filter_tau_s > 0:
            alpha = dt / (self.config.filter_tau_s + dt)
            self._filtered_error += alpha * (scaled_error - self._filtered_error)
        else:
            self._filtered_error = scaled_error

        # Store state
        self._last_error = self._filtered_error
        self._last_ref = ref_phase
        self._last_vco = vco_phase

        return self._filtered_error

    @property
    def last_error(self) -> float:
        """Get most recent error output."""
        return self._last_error

    @property
    def raw_phase_difference(self) -> float:
        """Get unwrapped phase difference in radians."""
        diff = self._last_ref - self._last_vco
        return math.atan2(math.sin(diff), math.cos(diff))

    def reset(self) -> None:
        """Reset detector state."""
        self._last_error = 0.0
        self._filtered_error = 0.0
        self._last_ref = 0.0
        self._last_vco = 0.0


class MultiplyingPhaseDetector(PhaseDetectorBase):
    """
    Multiplying (sinusoidal) phase detector.

    Implements: ε = sin(φ_ref - φ_vco)

    Characteristics:
    - Linear for small errors (|ε| < π/6)
    - Natural limiting for large errors
    - Zero output at lock (φ_ref = φ_vco)
    - Capture range: |Δφ| < π/2
    """

    def _compute_raw_error(self, ref_phase: float, vco_phase: float) -> float:
        """Sinusoidal phase comparison."""
        phase_diff = ref_phase - vco_phase
        return math.sin(phase_diff)


class TypeIIPhaseDetector(PhaseDetectorBase):
    """
    Type II Phase/Frequency Detector.

    Detects both phase and frequency differences, enabling
    acquisition from any initial condition.

    Uses state machine logic:
    - If VCO leads: output negative (slow down)
    - If reference leads: output positive (speed up)
    - At lock: tri-state (high impedance)
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)
        self._ref_edge_count: int = 0
        self._vco_edge_count: int = 0
        self._last_ref_wrap: float = 0.0
        self._last_vco_wrap: float = 0.0

    def _compute_raw_error(self, ref_phase: float, vco_phase: float) -> float:
        """
        Phase/frequency detection with edge counting.

        Returns linear phase error with frequency assist.
        """
        # Detect phase wrapping (edge events)
        ref_wrapped = ref_phase % TAU
        vco_wrapped = vco_phase % TAU

        # Check for reference edge (wrap from 2π to 0)
        if ref_wrapped < self._last_ref_wrap - math.pi:
            self._ref_edge_count += 1
        self._last_ref_wrap = ref_wrapped

        # Check for VCO edge
        if vco_wrapped < self._last_vco_wrap - math.pi:
            self._vco_edge_count += 1
        self._last_vco_wrap = vco_wrapped

        # Compute phase error with cycle counting
        ref_total = ref_wrapped + TAU * self._ref_edge_count
        vco_total = vco_wrapped + TAU * self._vco_edge_count

        # Linear phase error
        error = ref_total - vco_total

        # Clamp to reasonable range
        return max(-math.pi, min(math.pi, error))

    def reset(self) -> None:
        """Reset detector including edge counters."""
        super().reset()
        self._ref_edge_count = 0
        self._vco_edge_count = 0
        self._last_ref_wrap = 0.0
        self._last_vco_wrap = 0.0


class BangBangPhaseDetector(PhaseDetectorBase):
    """
    Bang-Bang (Binary) phase detector.

    Outputs only +1 or -1 based on sign of phase error.

    Characteristics:
    - Simple digital implementation
    - No proportional region
    - Results in limit cycling at lock
    - Good for high-speed digital PLLs
    """

    def __init__(self, config: Optional[DetectorConfig] = None, hysteresis: float = 0.0):
        """
        Initialize bang-bang detector.

        Args:
            config: Base configuration
            hysteresis: Hysteresis band width [rad]
        """
        super().__init__(config)
        self.hysteresis = hysteresis
        self._output_state: int = 0

    def _compute_raw_error(self, ref_phase: float, vco_phase: float) -> float:
        """Binary phase comparison with optional hysteresis."""
        phase_diff = ref_phase - vco_phase
        wrapped_diff = math.atan2(math.sin(phase_diff), math.cos(phase_diff))

        # Apply hysteresis
        if self.hysteresis > 0:
            if wrapped_diff > self.hysteresis:
                self._output_state = 1
            elif wrapped_diff < -self.hysteresis:
                self._output_state = -1
            # else: maintain previous state
            return float(self._output_state)
        else:
            # Simple sign detection
            if wrapped_diff > 0:
                return 1.0
            elif wrapped_diff < 0:
                return -1.0
            else:
                return 0.0


def create_phase_detector(
    detector_type: DetectorType = DetectorType.MULTIPLYING,
    gain: float = 1.0,
    filter_tau_s: float = 0.0
) -> PhaseDetectorBase:
    """
    Factory function to create a phase detector.

    Args:
        detector_type: Type of detector to create
        gain: Detector gain
        filter_tau_s: Output filter time constant

    Returns:
        Configured phase detector instance
    """
    config = DetectorConfig(gain=gain, filter_tau_s=filter_tau_s)

    if detector_type == DetectorType.MULTIPLYING:
        return MultiplyingPhaseDetector(config)
    elif detector_type == DetectorType.TYPE_II:
        return TypeIIPhaseDetector(config)
    elif detector_type == DetectorType.BANG_BANG:
        return BangBangPhaseDetector(config)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
