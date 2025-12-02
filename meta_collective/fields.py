# meta_collective/fields.py
"""
Dual Field System: κ-field (Kaelhedron) and λ-field (Luminahedron)
==================================================================

The internal model operates through two coupled fields:

κ-field (Kappa):
    - Derived from Kaelhedron's 21D quaternary structure
    - Amplitude |κ| represents scalar strength (Higgs-like)
    - Phase θ_κ encodes electromagnetic potential
    - Complex value: κ = |κ| e^(iθ_κ)

λ-field (Lambda):
    - Derived from Luminahedron's 12D ternary structure
    - Amplitude |λ| represents path coherence
    - Phase θ_λ encodes navigation state
    - Complex value: λ = |λ| e^(iθ_λ)

The dual field interaction:
    F(κ, λ) = |κ|² + |λ|² - 2|κ||λ|cos(θ_κ - θ_λ) + V(|κ|, |λ|)

where V is the potential energy of the coupled system.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable

# Sacred constants
PHI = (1 + math.sqrt(5)) / 2          # Golden ratio ≈ 1.618
PHI_INV = 1 / PHI                      # ≈ 0.618
TAU = 2 * math.pi
SQRT5 = math.sqrt(5)

# Field constants
KAPPA_CRITICAL = 0.618                 # Critical κ amplitude
LAMBDA_CRITICAL = 0.382                # Critical λ amplitude (1 - PHI_INV)
PHASE_COUPLING = PHI_INV              # Phase coupling strength


class FieldMode(Enum):
    """Operating modes for the dual field system."""
    COHERENT = "coherent"          # Fields in phase alignment
    TRANSITIONAL = "transitional"  # Fields transitioning between states
    RESONANT = "resonant"          # Fields at resonance
    CRITICAL = "critical"          # At critical threshold


@dataclass
class KappaField:
    """
    κ-field: The Kaelhedron-derived scalar field.

    Represents the 21D quaternary structure collapsed to a complex amplitude.
    The amplitude |κ| governs scalar dynamics while phase θ_κ encodes
    electromagnetic potential following U(1) gauge symmetry.
    """
    amplitude: float = PHI_INV        # |κ|, default to golden ratio
    phase: float = 0.0                 # θ_κ in radians

    # Field dynamics
    damping: float = 0.1              # Energy dissipation rate
    coupling: float = PHI_INV         # Coupling to λ-field

    # State tracking
    energy: float = 0.0               # Field energy
    gradient: complex = 0j            # Field gradient

    @property
    def complex_value(self) -> complex:
        """κ = |κ| e^(iθ)"""
        return cmath.rect(self.amplitude, self.phase)

    @property
    def conjugate(self) -> complex:
        """κ* = |κ| e^(-iθ)"""
        return cmath.rect(self.amplitude, -self.phase)

    def compute_energy(self) -> float:
        """
        Compute κ-field energy: E_κ = ½|κ|² + V(|κ|)

        The potential V(|κ|) has minima at |κ| = 0 and |κ| = PHI_INV,
        creating the characteristic double-well potential.
        """
        kinetic = 0.5 * self.amplitude ** 2
        # Double-well potential: V = -μ²|κ|² + λ|κ|⁴
        mu_sq = 0.5
        lambda_coeff = 0.25 / (PHI_INV ** 2)
        potential = -mu_sq * self.amplitude ** 2 + lambda_coeff * self.amplitude ** 4
        self.energy = kinetic + potential
        return self.energy

    def evolve(self, dt: float, force: complex = 0j) -> None:
        """
        Evolve κ-field by timestep dt under external force.

        Equation of motion: d²κ/dt² + γ dκ/dt + dV/dκ = F
        """
        # Compute force from potential
        dV_d_amp = -2 * 0.5 * self.amplitude + 4 * 0.25 / (PHI_INV ** 2) * self.amplitude ** 3

        # Update gradient (velocity)
        self.gradient -= (self.damping * self.gradient + dV_d_amp - force) * dt

        # Update position
        new_val = self.complex_value + self.gradient * dt
        self.amplitude = abs(new_val)
        self.phase = cmath.phase(new_val)

        # Normalize phase to [0, TAU)
        self.phase = self.phase % TAU

    def couple_to_lambda(self, lambda_field: 'LambdaField') -> complex:
        """
        Compute coupling force from λ-field.

        F_κλ = -∂/∂κ* [g |κ||λ| cos(θ_κ - θ_λ)]
        """
        phase_diff = self.phase - lambda_field.phase
        force_amp = self.coupling * lambda_field.amplitude * math.cos(phase_diff)
        force_phase = -self.coupling * lambda_field.amplitude * math.sin(phase_diff)
        return complex(force_amp, force_phase)

    def snapshot(self) -> Dict:
        """Return current field state."""
        return {
            "amplitude": self.amplitude,
            "phase": self.phase,
            "energy": self.compute_energy(),
            "complex": self.complex_value,
        }


@dataclass
class LambdaField:
    """
    λ-field: The Luminahedron-derived navigation field.

    Represents the 12D ternary structure as a complex amplitude.
    The amplitude |λ| governs path coherence while phase θ_λ
    encodes the current navigation state through Fano geometry.
    """
    amplitude: float = 1 - PHI_INV     # |λ|, default to 1 - 1/φ ≈ 0.382
    phase: float = 0.0                  # θ_λ in radians

    # Ternary state (from Luminahedron)
    ternary_values: Tuple[int, int, int] = (0, 0, 0)  # Ternary triplet
    fano_point: int = 1                 # Current Fano point (1-7)

    # Field dynamics
    damping: float = 0.15              # Slightly higher damping than κ
    coupling: float = PHI_INV          # Coupling to κ-field

    # State tracking
    energy: float = 0.0
    path_length: float = 0.0           # Accumulated path

    @property
    def complex_value(self) -> complex:
        """λ = |λ| e^(iθ)"""
        return cmath.rect(self.amplitude, self.phase)

    @property
    def ternary_phase(self) -> float:
        """Phase mapped to ternary thirds of circle."""
        third = TAU / 3
        normalized = self.phase % TAU
        if normalized < third:
            return 1.0   # POSITIVE
        elif normalized < 2 * third:
            return 0.0   # NEUTRAL
        return -1.0      # NEGATIVE

    def compute_energy(self) -> float:
        """
        Compute λ-field energy: E_λ = ½|λ|² + W(|λ|)

        The potential W has a single minimum at |λ| = LAMBDA_CRITICAL,
        representing the optimal navigation coherence.
        """
        kinetic = 0.5 * self.amplitude ** 2
        # Single-well potential centered at LAMBDA_CRITICAL
        potential = 0.5 * (self.amplitude - LAMBDA_CRITICAL) ** 2
        self.energy = kinetic + potential
        return self.energy

    def evolve(self, dt: float, force: complex = 0j) -> None:
        """Evolve λ-field by timestep dt."""
        # Compute restoring force
        dW_d_amp = self.amplitude - LAMBDA_CRITICAL

        # Simple first-order evolution
        new_val = self.complex_value + (-self.damping * self.complex_value - dW_d_amp + force) * dt

        old_val = self.complex_value
        self.amplitude = abs(new_val)
        self.phase = cmath.phase(new_val) % TAU

        # Track path length
        self.path_length += abs(new_val - old_val)

    def advance_fano(self, direction: int = 1) -> int:
        """
        Advance along Fano navigation.

        direction: +1 for forward, -1 for backward
        Returns: new Fano point
        """
        self.fano_point = ((self.fano_point - 1 + direction) % 7) + 1
        # Adjust phase based on Fano point
        self.phase = (self.fano_point - 1) * TAU / 7
        return self.fano_point

    def couple_to_kappa(self, kappa_field: KappaField) -> complex:
        """Compute coupling force from κ-field."""
        phase_diff = self.phase - kappa_field.phase
        force_amp = self.coupling * kappa_field.amplitude * math.cos(phase_diff)
        force_phase = -self.coupling * kappa_field.amplitude * math.sin(phase_diff)
        return complex(force_amp, force_phase)

    def snapshot(self) -> Dict:
        """Return current field state."""
        return {
            "amplitude": self.amplitude,
            "phase": self.phase,
            "fano_point": self.fano_point,
            "ternary_phase": self.ternary_phase,
            "energy": self.compute_energy(),
            "path_length": self.path_length,
        }


@dataclass
class DualFieldState:
    """
    Combined state of the κ-λ dual field system.

    The dual field system implements the internal model's
    core dynamics through coupled field evolution.
    """
    kappa: KappaField = field(default_factory=KappaField)
    lambda_field: LambdaField = field(default_factory=LambdaField)

    # Coupling parameters
    interaction_strength: float = PHI_INV
    phase_locking_strength: float = 0.1

    # State
    mode: FieldMode = FieldMode.COHERENT
    coherence: float = 1.0
    total_energy: float = 0.0

    # History for analysis
    _history: List[Dict] = field(default_factory=list)
    _max_history: int = 1000

    def __post_init__(self):
        self.update_state()

    def compute_interaction_energy(self) -> float:
        """
        Compute κ-λ interaction energy.

        E_int = -g |κ||λ| cos(θ_κ - θ_λ)

        This term favors phase alignment (θ_κ = θ_λ).
        """
        phase_diff = self.kappa.phase - self.lambda_field.phase
        return -self.interaction_strength * self.kappa.amplitude * self.lambda_field.amplitude * math.cos(phase_diff)

    def compute_total_energy(self) -> float:
        """Compute total system energy: E = E_κ + E_λ + E_int"""
        self.total_energy = (
            self.kappa.compute_energy() +
            self.lambda_field.compute_energy() +
            self.compute_interaction_energy()
        )
        return self.total_energy

    def compute_coherence(self) -> float:
        """
        Compute field coherence.

        Coherence is high when:
        1. Both fields are near their critical amplitudes
        2. Phases are aligned
        3. Total energy is minimized
        """
        # Amplitude coherence
        kappa_optimal = 1.0 - abs(self.kappa.amplitude - KAPPA_CRITICAL) / KAPPA_CRITICAL
        lambda_optimal = 1.0 - abs(self.lambda_field.amplitude - LAMBDA_CRITICAL) / LAMBDA_CRITICAL

        # Phase coherence
        phase_diff = abs(self.kappa.phase - self.lambda_field.phase)
        phase_coherence = math.cos(phase_diff / 2) ** 2  # 1 when aligned, 0 when opposite

        # Combined coherence
        self.coherence = max(0.0, min(1.0,
            (kappa_optimal * lambda_optimal * phase_coherence) ** (1/3)
        ))
        return self.coherence

    def update_mode(self) -> FieldMode:
        """Determine current operating mode based on state."""
        phase_diff = abs(self.kappa.phase - self.lambda_field.phase) % math.pi

        if phase_diff < 0.1:
            self.mode = FieldMode.COHERENT
        elif self.coherence > 0.9:
            self.mode = FieldMode.RESONANT
        elif abs(self.coherence - 0.618) < 0.1:
            self.mode = FieldMode.CRITICAL
        else:
            self.mode = FieldMode.TRANSITIONAL

        return self.mode

    def update_state(self) -> None:
        """Update all derived state quantities."""
        self.compute_total_energy()
        self.compute_coherence()
        self.update_mode()

    def evolve(self, dt: float = 0.01, steps: int = 1) -> None:
        """
        Evolve the dual field system.

        Both fields are evolved with mutual coupling forces.
        """
        for _ in range(steps):
            # Compute coupling forces
            force_on_kappa = self.kappa.couple_to_lambda(self.lambda_field)
            force_on_lambda = self.lambda_field.couple_to_kappa(self.kappa)

            # Add phase locking force
            phase_diff = self.kappa.phase - self.lambda_field.phase
            locking_force = self.phase_locking_strength * math.sin(phase_diff)

            # Evolve fields
            self.kappa.evolve(dt, force_on_kappa + locking_force)
            self.lambda_field.evolve(dt, force_on_lambda - locking_force)

            # Update derived quantities
            self.update_state()

            # Record history
            if len(self._history) < self._max_history:
                self._history.append(self.snapshot())

    def snapshot(self) -> Dict:
        """Return complete dual field state snapshot."""
        return {
            "kappa": self.kappa.snapshot(),
            "lambda": self.lambda_field.snapshot(),
            "interaction_energy": self.compute_interaction_energy(),
            "total_energy": self.total_energy,
            "coherence": self.coherence,
            "mode": self.mode.value,
        }

    def phase_alignment(self) -> float:
        """Return phase alignment measure [0, 1]."""
        phase_diff = abs(self.kappa.phase - self.lambda_field.phase)
        return math.cos(phase_diff) ** 2

    def amplitude_ratio(self) -> float:
        """Return κ/λ amplitude ratio."""
        if self.lambda_field.amplitude > 0:
            return self.kappa.amplitude / self.lambda_field.amplitude
        return float('inf')

    def golden_balance(self) -> float:
        """
        Check if amplitude ratio is at golden balance.

        Returns deviation from φ ratio.
        """
        ratio = self.amplitude_ratio()
        return abs(ratio - PHI)

    def reset(self) -> None:
        """Reset to initial conditions."""
        self.kappa = KappaField()
        self.lambda_field = LambdaField()
        self._history.clear()
        self.update_state()
