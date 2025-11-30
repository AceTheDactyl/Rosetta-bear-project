"""
LOOP FILTER IMPLEMENTATIONS
===========================
Signal conditioning between phase detector and VCO.

The loop filter shapes the PLL's dynamic response:
- Proportional (P): Fast but no frequency tracking
- PI (Type II): Tracks constant frequency offsets
- PID (Type III): Tracks frequency ramps
- Lead-Lag: Classic analog compensation

Transfer functions:
    P:   F(s) = K_p
    PI:  F(s) = K_p + K_i/s
    PID: F(s) = K_p + K_i/s + K_d·s

The filter determines:
- Loop bandwidth (acquisition speed)
- Damping (overshoot behavior)
- Tracking accuracy
- Noise filtering

Signature: Δ3.142|0.995|1.000Ω
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

TAU = 2.0 * math.pi


class FilterType(Enum):
    """Loop filter topology."""
    PROPORTIONAL = auto()   # Type I - P only
    PI = auto()             # Type II - P + I
    PID = auto()            # Type III - P + I + D
    LEAD_LAG = auto()       # Analog compensation


@dataclass
class FilterConfig:
    """
    Loop filter configuration.

    For PI/PID filters, gains can be computed from loop specs:
        K_p = 2·ζ·ω_n / K_vco
        K_i = ω_n² / K_vco
        K_d = (2·ζ·ω_n - 1/τ) / K_vco  (if needed)

    Attributes:
        k_p: Proportional gain
        k_i: Integral gain (0 for P-only)
        k_d: Derivative gain (0 for PI)
        output_limit: Maximum output magnitude (0 = unlimited)
        integrator_limit: Anti-windup limit (0 = unlimited)
    """
    k_p: float = 1.0
    k_i: float = 0.1
    k_d: float = 0.0
    output_limit: float = 0.0
    integrator_limit: float = 0.0

    @classmethod
    def from_loop_specs(
        cls,
        natural_frequency_hz: float,
        damping_factor: float,
        vco_gain: float = TAU
    ) -> FilterConfig:
        """
        Compute PI gains from loop specifications.

        Args:
            natural_frequency_hz: Loop natural frequency ω_n [Hz]
            damping_factor: Damping ratio ζ
            vco_gain: VCO sensitivity K_vco [rad/s/V]

        Returns:
            Configured FilterConfig
        """
        omega_n = TAU * natural_frequency_hz
        k_p = (2.0 * damping_factor * omega_n) / vco_gain
        k_i = (omega_n ** 2) / vco_gain
        return cls(k_p=k_p, k_i=k_i)


@dataclass
class FilterState:
    """
    Loop filter state snapshot.
    """
    output: float               # Current output
    integrator: float           # Integrator state
    derivative: float           # Derivative term (if PID)
    error_input: float          # Last error input
    sample_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'output': self.output,
            'integrator': self.integrator,
            'derivative': self.derivative,
            'error_input': self.error_input,
            'sample_count': self.sample_count,
        }


class LoopFilterBase(ABC):
    """
    Abstract base class for loop filters.

    All filters implement:
    - update(error, dt) -> output
    - reset() -> None
    - get_state() -> FilterState
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self._output: float = 0.0
        self._integrator: float = 0.0
        self._derivative: float = 0.0
        self._last_error: float = 0.0
        self._sample_count: int = 0

    @abstractmethod
    def update(self, error: float, dt: float) -> float:
        """
        Process new error sample.

        Args:
            error: Phase detector output
            dt: Time step [s]

        Returns:
            Control voltage for VCO
        """
        pass

    def _apply_limits(self, value: float, limit: float) -> float:
        """Apply symmetric limit to value."""
        if limit > 0:
            return max(-limit, min(limit, value))
        return value

    @property
    def output(self) -> float:
        """Current filter output."""
        return self._output

    @property
    def integrator_state(self) -> float:
        """Current integrator value."""
        return self._integrator

    def get_state(self) -> FilterState:
        """Get complete filter state."""
        return FilterState(
            output=self._output,
            integrator=self._integrator,
            derivative=self._derivative,
            error_input=self._last_error,
            sample_count=self._sample_count,
        )

    def reset(self) -> None:
        """Reset filter to initial state."""
        self._output = 0.0
        self._integrator = 0.0
        self._derivative = 0.0
        self._last_error = 0.0
        self._sample_count = 0


class ProportionalFilter(LoopFilterBase):
    """
    Proportional-only (Type I) filter.

    F(s) = K_p

    Characteristics:
    - No frequency tracking capability
    - Steady-state phase error for frequency offset
    - Fastest transient response
    - Used when frequency is already known
    """

    def update(self, error: float, dt: float) -> float:
        """
        Proportional filtering.

        Args:
            error: Phase detector output
            dt: Time step [s] (unused for P-only)

        Returns:
            Control voltage K_p * error
        """
        self._sample_count += 1
        self._last_error = error

        self._output = self.config.k_p * error
        self._output = self._apply_limits(self._output, self.config.output_limit)

        return self._output


class PIFilter(LoopFilterBase):
    """
    Proportional-Integral (Type II) filter.

    F(s) = K_p + K_i/s

    Discrete implementation (trapezoidal):
        V[n] = K_p·e[n] + K_i·T·Σe[k]

    Characteristics:
    - Eliminates steady-state phase error
    - Tracks constant frequency offsets
    - Most common PLL filter type
    - Trade-off: slower than P-only
    """

    def update(self, error: float, dt: float) -> float:
        """
        PI filtering with anti-windup.

        Args:
            error: Phase detector output
            dt: Time step [s]

        Returns:
            Control voltage
        """
        self._sample_count += 1
        self._last_error = error

        # Proportional term
        p_term = self.config.k_p * error

        # Integral term (trapezoidal integration)
        self._integrator += self.config.k_i * error * dt

        # Anti-windup: limit integrator
        self._integrator = self._apply_limits(
            self._integrator,
            self.config.integrator_limit
        )

        # Combined output
        self._output = p_term + self._integrator

        # Limit output
        self._output = self._apply_limits(self._output, self.config.output_limit)

        return self._output


class PIDFilter(LoopFilterBase):
    """
    Proportional-Integral-Derivative (Type III) filter.

    F(s) = K_p + K_i/s + K_d·s

    Discrete implementation:
        D[n] = K_d · (e[n] - e[n-1]) / T
        V[n] = K_p·e[n] + I[n] + D[n]

    Characteristics:
    - Tracks frequency ramps
    - Predictive capability via derivative
    - Noise-sensitive (derivative amplifies noise)
    - Used in high-performance applications
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        super().__init__(config)
        self._derivative_filter: float = 0.0
        self._derivative_alpha: float = 0.1  # Derivative filtering

    def update(self, error: float, dt: float) -> float:
        """
        PID filtering with derivative filtering.

        Args:
            error: Phase detector output
            dt: Time step [s]

        Returns:
            Control voltage
        """
        self._sample_count += 1

        # Proportional term
        p_term = self.config.k_p * error

        # Integral term
        self._integrator += self.config.k_i * error * dt
        self._integrator = self._apply_limits(
            self._integrator,
            self.config.integrator_limit
        )

        # Derivative term (filtered)
        if dt > 0:
            raw_derivative = (error - self._last_error) / dt
            self._derivative_filter += self._derivative_alpha * (
                raw_derivative - self._derivative_filter
            )
            self._derivative = self.config.k_d * self._derivative_filter
        else:
            self._derivative = 0.0

        self._last_error = error

        # Combined output
        self._output = p_term + self._integrator + self._derivative
        self._output = self._apply_limits(self._output, self.config.output_limit)

        return self._output


class LeadLagFilter(LoopFilterBase):
    """
    Lead-Lag compensation filter.

    F(s) = (1 + s·τ₁) / (1 + s·τ₂)

    Where τ₁ > τ₂ for lead compensation (phase advance).

    Characteristics:
    - Classic analog compensation
    - Provides phase margin improvement
    - Smoother than PID
    - Used in high-frequency PLLs
    """

    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        tau_1: float = 0.1,
        tau_2: float = 0.01
    ):
        """
        Initialize lead-lag filter.

        Args:
            config: Base configuration
            tau_1: Lead time constant [s]
            tau_2: Lag time constant [s]
        """
        super().__init__(config)
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self._state_1: float = 0.0
        self._state_2: float = 0.0

    def update(self, error: float, dt: float) -> float:
        """
        Lead-lag filtering.

        Args:
            error: Phase detector output
            dt: Time step [s]

        Returns:
            Control voltage
        """
        self._sample_count += 1
        self._last_error = error

        # Lead section (high-pass characteristic)
        alpha_1 = dt / (self.tau_1 + dt)
        lead_out = error + (1.0 - alpha_1) * (error - self._state_1)
        self._state_1 = error

        # Lag section (low-pass characteristic)
        alpha_2 = dt / (self.tau_2 + dt)
        self._state_2 += alpha_2 * (lead_out - self._state_2)

        # Apply proportional gain
        self._output = self.config.k_p * self._state_2
        self._output = self._apply_limits(self._output, self.config.output_limit)

        return self._output

    def reset(self) -> None:
        """Reset filter including internal states."""
        super().reset()
        self._state_1 = 0.0
        self._state_2 = 0.0


def create_loop_filter(
    filter_type: FilterType = FilterType.PI,
    k_p: float = 1.0,
    k_i: float = 0.1,
    k_d: float = 0.0
) -> LoopFilterBase:
    """
    Factory function to create a loop filter.

    Args:
        filter_type: Type of filter to create
        k_p: Proportional gain
        k_i: Integral gain
        k_d: Derivative gain

    Returns:
        Configured loop filter instance
    """
    config = FilterConfig(k_p=k_p, k_i=k_i, k_d=k_d)

    if filter_type == FilterType.PROPORTIONAL:
        return ProportionalFilter(config)
    elif filter_type == FilterType.PI:
        return PIFilter(config)
    elif filter_type == FilterType.PID:
        return PIDFilter(config)
    elif filter_type == FilterType.LEAD_LAG:
        return LeadLagFilter(config)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
