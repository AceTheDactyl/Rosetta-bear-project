"""
VOLTAGE-CONTROLLED OSCILLATOR IMPLEMENTATIONS
=============================================
Phase generation with voltage-controlled frequency modulation.

The VCO is the heart of the PLL—it generates an output signal
whose phase advances at a rate controlled by the input voltage:

    dφ/dt = ω₀ + K_vco · V(t)

Where:
    ω₀ = 2π · f_center (center/free-running frequency)
    K_vco = VCO gain [rad/s/V]
    V(t) = control voltage from loop filter

Features:
- Single VCO for basic PLL operation
- VCO bank for multi-oscillator applications
- Phase noise modeling (optional)
- Frequency limiting

Signature: Δ3.142|0.995|1.000Ω
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, List

TAU = 2.0 * math.pi


@dataclass
class VCOConfig:
    """
    VCO configuration parameters.

    Attributes:
        center_frequency_hz: Free-running frequency [Hz]
        vco_gain: Frequency sensitivity [rad/s/V]
        min_frequency_hz: Lower frequency limit [Hz]
        max_frequency_hz: Upper frequency limit [Hz]
        phase_noise_std: Phase noise standard deviation [rad]
    """
    center_frequency_hz: float = 1.0
    vco_gain: float = TAU              # 1 Hz/V
    min_frequency_hz: float = 0.01
    max_frequency_hz: float = 100.0
    phase_noise_std: float = 0.0       # Ideal VCO by default


@dataclass
class VCOState:
    """
    VCO state snapshot.

    Captures the complete state of a VCO at a moment in time.
    """
    phase: float                        # Current phase [0, 2π)
    frequency_hz: float                 # Instantaneous frequency [Hz]
    control_voltage: float              # Applied control voltage
    phase_increment: float              # Last phase step [rad]
    sample_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'phase': self.phase,
            'frequency_hz': self.frequency_hz,
            'control_voltage': self.control_voltage,
            'phase_increment': self.phase_increment,
            'sample_count': self.sample_count,
        }


class VoltageControlledOscillator:
    """
    Single Voltage-Controlled Oscillator.

    Generates a phase signal that advances based on control voltage:
        φ[n+1] = φ[n] + (ω₀ + K_vco·V)·Δt

    The VCO maintains continuous phase, wrapping to [0, 2π).
    """

    def __init__(self, config: Optional[VCOConfig] = None):
        """
        Initialize VCO.

        Args:
            config: VCO configuration (uses defaults if None)
        """
        self.config = config or VCOConfig()

        self._phase: float = 0.0
        self._frequency_hz: float = self.config.center_frequency_hz
        self._last_control_voltage: float = 0.0
        self._last_phase_increment: float = 0.0
        self._sample_count: int = 0

    def update(self, control_voltage: float, dt: float) -> float:
        """
        Advance VCO by one time step.

        Args:
            control_voltage: Control input from loop filter [V]
            dt: Time step [s]

        Returns:
            New phase [0, 2π)
        """
        self._sample_count += 1
        self._last_control_voltage = control_voltage

        # Compute instantaneous angular frequency
        omega_center = TAU * self.config.center_frequency_hz
        omega_deviation = self.config.vco_gain * control_voltage
        omega_instant = omega_center + omega_deviation

        # Apply frequency limits
        freq_hz = omega_instant / TAU
        freq_hz = max(self.config.min_frequency_hz,
                      min(self.config.max_frequency_hz, freq_hz))
        omega_instant = TAU * freq_hz
        self._frequency_hz = freq_hz

        # Compute phase increment
        phase_increment = omega_instant * dt

        # Add phase noise if configured
        if self.config.phase_noise_std > 0:
            phase_increment += random.gauss(0, self.config.phase_noise_std)

        self._last_phase_increment = phase_increment

        # Advance phase
        self._phase += phase_increment
        self._phase = self._phase % TAU

        return self._phase

    @property
    def phase(self) -> float:
        """Current phase [0, 2π)."""
        return self._phase

    @phase.setter
    def phase(self, value: float) -> None:
        """Set phase (wraps to [0, 2π))."""
        self._phase = value % TAU

    @property
    def frequency_hz(self) -> float:
        """Instantaneous frequency [Hz]."""
        return self._frequency_hz

    def get_state(self) -> VCOState:
        """Get complete VCO state."""
        return VCOState(
            phase=self._phase,
            frequency_hz=self._frequency_hz,
            control_voltage=self._last_control_voltage,
            phase_increment=self._last_phase_increment,
            sample_count=self._sample_count,
        )

    def reset(self, initial_phase: float = 0.0) -> None:
        """
        Reset VCO to initial state.

        Args:
            initial_phase: Starting phase [rad]
        """
        self._phase = initial_phase % TAU
        self._frequency_hz = self.config.center_frequency_hz
        self._last_control_voltage = 0.0
        self._last_phase_increment = 0.0
        self._sample_count = 0


class MultiVCOBank:
    """
    Bank of multiple synchronized VCOs.

    Used for applications requiring multiple oscillators:
    - Multi-tone synthesis
    - Parallel lock detection
    - Frequency diversity

    All VCOs in the bank share a common control voltage but
    can have different center frequencies (for tone spacing).
    """

    def __init__(
        self,
        n_oscillators: int,
        base_config: Optional[VCOConfig] = None,
        frequency_spread_hz: float = 0.0
    ):
        """
        Initialize VCO bank.

        Args:
            n_oscillators: Number of VCOs in bank
            base_config: Base configuration for all VCOs
            frequency_spread_hz: Frequency offset between adjacent VCOs
        """
        self.n_oscillators = n_oscillators
        self.base_config = base_config or VCOConfig()
        self.frequency_spread_hz = frequency_spread_hz

        # Create VCOs with spread center frequencies
        self._vcos: List[VoltageControlledOscillator] = []
        center_idx = n_oscillators / 2.0

        for i in range(n_oscillators):
            config = VCOConfig(
                center_frequency_hz=self.base_config.center_frequency_hz
                                    + (i - center_idx) * frequency_spread_hz,
                vco_gain=self.base_config.vco_gain,
                min_frequency_hz=self.base_config.min_frequency_hz,
                max_frequency_hz=self.base_config.max_frequency_hz,
                phase_noise_std=self.base_config.phase_noise_std,
            )
            self._vcos.append(VoltageControlledOscillator(config))

    def update(self, control_voltage: float, dt: float) -> List[float]:
        """
        Update all VCOs with common control voltage.

        Args:
            control_voltage: Shared control input [V]
            dt: Time step [s]

        Returns:
            List of phases from all VCOs
        """
        return [vco.update(control_voltage, dt) for vco in self._vcos]

    def update_individual(
        self,
        control_voltages: List[float],
        dt: float
    ) -> List[float]:
        """
        Update VCOs with individual control voltages.

        Args:
            control_voltages: Per-VCO control inputs
            dt: Time step [s]

        Returns:
            List of phases
        """
        if len(control_voltages) != self.n_oscillators:
            raise ValueError(
                f"Expected {self.n_oscillators} control voltages, "
                f"got {len(control_voltages)}"
            )
        return [
            vco.update(cv, dt)
            for vco, cv in zip(self._vcos, control_voltages)
        ]

    @property
    def phases(self) -> List[float]:
        """Get all VCO phases."""
        return [vco.phase for vco in self._vcos]

    @property
    def frequencies(self) -> List[float]:
        """Get all VCO frequencies [Hz]."""
        return [vco.frequency_hz for vco in self._vcos]

    def get_mean_phase(self) -> float:
        """Compute mean phase across all VCOs."""
        real_sum = sum(math.cos(vco.phase) for vco in self._vcos)
        imag_sum = sum(math.sin(vco.phase) for vco in self._vcos)
        return math.atan2(imag_sum, real_sum) % TAU

    def get_coherence(self) -> float:
        """
        Compute coherence (order parameter) across VCOs.

        r = |mean(exp(jφ))|
        """
        real_sum = sum(math.cos(vco.phase) for vco in self._vcos)
        imag_sum = sum(math.sin(vco.phase) for vco in self._vcos)
        return math.sqrt(real_sum**2 + imag_sum**2) / self.n_oscillators

    def get_vco(self, index: int) -> VoltageControlledOscillator:
        """Get VCO by index."""
        return self._vcos[index]

    def reset(self, initial_phases: Optional[List[float]] = None) -> None:
        """
        Reset all VCOs.

        Args:
            initial_phases: Per-VCO initial phases (0 for all if None)
        """
        if initial_phases is None:
            initial_phases = [0.0] * self.n_oscillators

        for vco, phase in zip(self._vcos, initial_phases):
            vco.reset(phase)


def create_vco(
    center_frequency_hz: float = 1.0,
    vco_gain: float = TAU,
    phase_noise_std: float = 0.0
) -> VoltageControlledOscillator:
    """
    Factory function to create a VCO.

    Args:
        center_frequency_hz: Free-running frequency [Hz]
        vco_gain: Frequency sensitivity [rad/s/V]
        phase_noise_std: Phase noise level [rad]

    Returns:
        Configured VCO instance
    """
    config = VCOConfig(
        center_frequency_hz=center_frequency_hz,
        vco_gain=vco_gain,
        phase_noise_std=phase_noise_std,
    )
    return VoltageControlledOscillator(config)


def create_vco_bank(
    n_oscillators: int,
    center_frequency_hz: float = 1.0,
    frequency_spread_hz: float = 0.1
) -> MultiVCOBank:
    """
    Factory function to create a VCO bank.

    Args:
        n_oscillators: Number of VCOs
        center_frequency_hz: Base center frequency [Hz]
        frequency_spread_hz: Frequency offset between VCOs [Hz]

    Returns:
        Configured VCO bank instance
    """
    config = VCOConfig(center_frequency_hz=center_frequency_hz)
    return MultiVCOBank(n_oscillators, config, frequency_spread_hz)
