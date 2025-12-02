#!/usr/bin/env python3
"""
INVARIANT VALIDATORS - Mathematical Consistency Checks
======================================================

Validates mathematical invariants during system operation.
Ensures consistency of MirrorRoot, free energy, phase normalization.

Key Invariants:
    - MirrorRoot: Λ × Ν = Β² (φ × φ⁻¹ = 1)
    - Free Energy: F monotonically decreases
    - Phases: θ ∈ [0, 2π)
    - Coupling Balance: κ/(κ+λ) → φ⁻¹

Signature: Δ|invariants|z0.99|consistency|Ω
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1
TAU = 2 * math.pi


@dataclass
class InvariantViolation:
    """Record of invariant violation for debugging."""
    invariant_name: str
    expected_value: float
    actual_value: float
    deviation: float
    timestamp: float
    context: Dict[str, float]


class InvariantValidator:
    """Validates mathematical invariants during operation."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.violations: List[InvariantViolation] = []

    def validate_mirroroot(self, logos: float, nous: float, bios: float) -> Tuple[bool, Optional[InvariantViolation]]:
        """Validate MirrorRoot identity: Λ × Ν = Β²."""
        product = logos * nous
        bios_sq = bios ** 2
        deviation = abs(product - bios_sq)
        valid = deviation < self.tolerance

        if not valid:
            v = InvariantViolation("MirrorRoot", bios_sq, product, deviation, 0.0, {"logos": logos, "nous": nous, "bios": bios})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_free_energy_decrease(self, f_previous: float, f_current: float) -> Tuple[bool, Optional[InvariantViolation]]:
        """Validate free energy monotonic decrease."""
        valid = f_current <= f_previous + self.tolerance

        if not valid:
            v = InvariantViolation("FreeEnergyDecrease", f_previous, f_current, f_current - f_previous, 0.0, {"f_previous": f_previous, "f_current": f_current})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_phase_normalization(self, phases: List[float]) -> Tuple[bool, Optional[InvariantViolation]]:
        """Validate phases are in [0, 2π) range."""
        for i, phase in enumerate(phases):
            if phase < 0 or phase >= TAU:
                v = InvariantViolation("PhaseNormalization", phase % TAU, phase, abs(phase - (phase % TAU)), 0.0, {"index": i})
                self.violations.append(v)
                return False, v
        return True, None

    def validate_coupling_balance(self, kappa: float, lambda_: float, target: float = PHI_INV, tol: float = 0.1) -> Tuple[bool, Optional[InvariantViolation]]:
        """Validate κ-λ coupling approaches golden ratio."""
        total = kappa + lambda_
        if total == 0:
            return True, None
        ratio = kappa / total
        deviation = abs(ratio - target)

        if deviation >= tol:
            v = InvariantViolation("CouplingBalance", target, ratio, deviation, 0.0, {"kappa": kappa, "lambda": lambda_})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_nec_violation(self, r: float, energy_density: float, radial_tension: float) -> Tuple[bool, Optional[InvariantViolation]]:
        """Validate NEC violation for wormhole traversability (ρ + τ < 0)."""
        nec = energy_density + radial_tension

        if nec >= 0:
            v = InvariantViolation("NECViolation", -0.01, nec, nec, 0.0, {"r": r, "rho": energy_density, "tau": radial_tension})
            self.violations.append(v)
            return False, v
        return True, None

    def get_violations(self) -> List[InvariantViolation]:
        return self.violations.copy()

    def clear_violations(self) -> None:
        self.violations.clear()
