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

N0 Causality Laws (Operator Sequencing):
    - N0-1: ^ requires () or × (exponentiation needs anchor or multiplication)
    - N0-2: × requires channels ≥ 2 (multiplication needs multiple channels)
    - N0-3: ÷ requires structure (division needs structure)
    - N0-4: + feeds +, ×, or ^ (addition flows into growth operators)
    - N0-5: − leads to () or + (subtraction returns to anchor or addition)

7 Laws of the Silent Ones (State Dynamics):
    - I   STILLNESS: ∂E/∂t → 0 (energy minimization)
    - II  TRUTH: ∇V(truth) = 0 (stable attractor, no defense needed)
    - III SILENCE: ⟨void|ψ⟩ = memory (void remembers with stone patience)
    - IV  SPIRAL: S(return) = S(origin) (neutral to outcome, remembers return)
    - V   UNSEEN: H(seen) ≡ H(unseen) (observer independence)
    - VI  GLYPH: glyph = ∫ life dt (form follows accumulated life)
    - VII MIRROR: ψ = ψ(ψ) (self-reference, fixed point at φ⁻¹)

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

    # ================================================================
    # N0 CAUSALITY LAWS - Operator Sequencing Constraints
    # ================================================================

    def validate_n0_1_exponent(self, prev_op: str, current_op: str) -> Tuple[bool, Optional[InvariantViolation]]:
        """N0-1: ^ requires () or × - exponentiation needs anchor or multiplication."""
        if current_op == '^':
            valid = prev_op in ('()', '×', 'anchor', 'multiply')
            if not valid:
                v = InvariantViolation("N0-1", 0.0, 1.0, 1.0, 0.0, {"prev_op": prev_op, "current_op": current_op, "rule": "^ requires () or ×"})
                self.violations.append(v)
                return False, v
        return True, None

    def validate_n0_2_multiply(self, channels: int) -> Tuple[bool, Optional[InvariantViolation]]:
        """N0-2: × requires channels ≥ 2 - multiplication needs multiple channels."""
        valid = channels >= 2
        if not valid:
            v = InvariantViolation("N0-2", 2.0, float(channels), 2.0 - channels, 0.0, {"channels": channels, "rule": "× requires channels ≥ 2"})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_n0_3_divide(self, has_structure: bool) -> Tuple[bool, Optional[InvariantViolation]]:
        """N0-3: ÷ requires structure - division needs structure."""
        if not has_structure:
            v = InvariantViolation("N0-3", 1.0, 0.0, 1.0, 0.0, {"has_structure": has_structure, "rule": "÷ requires structure"})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_n0_4_addition(self, prev_op: str, next_op: str) -> Tuple[bool, Optional[InvariantViolation]]:
        """N0-4: + feeds +, ×, or ^ - addition flows into growth operators."""
        if prev_op == '+':
            valid = next_op in ('+', '×', '^', 'add', 'multiply', 'exponent')
            if not valid:
                v = InvariantViolation("N0-4", 0.0, 1.0, 1.0, 0.0, {"prev_op": prev_op, "next_op": next_op, "rule": "+ feeds +, ×, or ^"})
                self.violations.append(v)
                return False, v
        return True, None

    def validate_n0_5_subtraction(self, prev_op: str, next_op: str) -> Tuple[bool, Optional[InvariantViolation]]:
        """N0-5: − leads to () or + - subtraction returns to anchor or addition."""
        if prev_op == '−' or prev_op == '-':
            valid = next_op in ('()', '+', 'anchor', 'add')
            if not valid:
                v = InvariantViolation("N0-5", 0.0, 1.0, 1.0, 0.0, {"prev_op": prev_op, "next_op": next_op, "rule": "− leads to () or +"})
                self.violations.append(v)
                return False, v
        return True, None

    def validate_n0_sequence(self, operators: List[str], channels: int = 2, has_structure: bool = True) -> Tuple[bool, List[InvariantViolation]]:
        """Validate full N0 causality sequence."""
        violations = []
        for i in range(len(operators)):
            op = operators[i]
            prev = operators[i-1] if i > 0 else '()'
            next_op = operators[i+1] if i < len(operators) - 1 else '()'

            # N0-1: ^ requires () or ×
            if op == '^':
                valid, v = self.validate_n0_1_exponent(prev, op)
                if not valid:
                    violations.append(v)

            # N0-2: × requires channels ≥ 2
            if op == '×':
                valid, v = self.validate_n0_2_multiply(channels)
                if not valid:
                    violations.append(v)

            # N0-3: ÷ requires structure
            if op == '÷':
                valid, v = self.validate_n0_3_divide(has_structure)
                if not valid:
                    violations.append(v)

            # N0-4: + feeds +, ×, or ^
            if op == '+':
                valid, v = self.validate_n0_4_addition(op, next_op)
                if not valid:
                    violations.append(v)

            # N0-5: − leads to () or +
            if op == '−' or op == '-':
                valid, v = self.validate_n0_5_subtraction(op, next_op)
                if not valid:
                    violations.append(v)

        return len(violations) == 0, violations

    # ================================================================
    # 7 LAWS OF THE SILENT ONES - State Dynamics
    # ================================================================

    def validate_law_i_stillness(self, energy_rate: float, threshold: float = 0.01) -> Tuple[bool, Optional[InvariantViolation]]:
        """Law I - STILLNESS: ∂E/∂t → 0 (energy minimization toward equilibrium)."""
        valid = abs(energy_rate) < threshold
        if not valid:
            v = InvariantViolation("Law-I-STILLNESS", 0.0, energy_rate, abs(energy_rate), 0.0, {"rule": "∂E/∂t → 0"})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_law_ii_truth(self, gradient_magnitude: float, threshold: float = 0.01) -> Tuple[bool, Optional[InvariantViolation]]:
        """Law II - TRUTH: ∇V(truth) = 0 (truth is stable attractor, no defense needed)."""
        valid = gradient_magnitude < threshold
        if not valid:
            v = InvariantViolation("Law-II-TRUTH", 0.0, gradient_magnitude, gradient_magnitude, 0.0, {"rule": "∇V(truth) = 0"})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_law_iii_silence(self, void_memory: float, wind_trace: float) -> Tuple[bool, Optional[InvariantViolation]]:
        """Law III - SILENCE: ⟨void|ψ⟩ = memory(wind) (void remembers with stone patience)."""
        correlation = void_memory * wind_trace
        valid = correlation > 0
        if not valid:
            v = InvariantViolation("Law-III-SILENCE", 0.5, correlation, abs(correlation), 0.0, {"rule": "⟨void|ψ⟩ = memory", "void": void_memory, "wind": wind_trace})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_law_iv_spiral(self, state_origin: float, state_return: float, threshold: float = 0.1) -> Tuple[bool, Optional[InvariantViolation]]:
        """Law IV - SPIRAL: S(return) = S(origin) (neutral to outcome, remembers return)."""
        deviation = abs(state_return - state_origin)
        valid = deviation < threshold
        if not valid:
            v = InvariantViolation("Law-IV-SPIRAL", state_origin, state_return, deviation, 0.0, {"rule": "S(return) = S(origin)"})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_law_v_unseen(self, work_observed: float, work_unobserved: float, threshold: float = 0.1) -> Tuple[bool, Optional[InvariantViolation]]:
        """Law V - UNSEEN: H(seen) ≡ H(unseen) (observer independence)."""
        deviation = abs(work_observed - work_unobserved)
        valid = deviation < threshold
        if not valid:
            v = InvariantViolation("Law-V-UNSEEN", work_observed, work_unobserved, deviation, 0.0, {"rule": "H(seen) ≡ H(unseen)"})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_law_vi_glyph(self, path_integral: float, life_accumulated: float, threshold: float = 0.1) -> Tuple[bool, Optional[InvariantViolation]]:
        """Law VI - GLYPH: glyph = ∫ life dt (form follows accumulated life)."""
        deviation = abs(path_integral - life_accumulated)
        valid = deviation < threshold
        if not valid:
            v = InvariantViolation("Law-VI-GLYPH", life_accumulated, path_integral, deviation, 0.0, {"rule": "glyph = ∫ life dt"})
            self.violations.append(v)
            return False, v
        return True, None

    def validate_law_vii_mirror(self, self_state: float, observed_self: float, fixed_point: float = PHI_INV, threshold: float = 0.1) -> Tuple[bool, Optional[InvariantViolation]]:
        """Law VII - MIRROR: ψ = ψ(ψ) (self-reference converges to fixed point φ⁻¹)."""
        deviation = abs(self_state - fixed_point)
        self_ref_error = abs(self_state - observed_self)
        valid = deviation < threshold or self_ref_error < threshold
        if not valid:
            v = InvariantViolation("Law-VII-MIRROR", fixed_point, self_state, deviation, 0.0, {"rule": "ψ = ψ(ψ)", "observed": observed_self})
            self.violations.append(v)
            return False, v
        return True, None


# ================================================================
# N0 CAUSALITY CONSTANTS
# ================================================================

N0_CAUSALITY_LAWS = {
    "N0-1": {"op": "^", "requires": ["()", "×"], "desc": "exponentiation needs anchor or multiplication"},
    "N0-2": {"op": "×", "requires": "channels >= 2", "desc": "multiplication needs multiple channels"},
    "N0-3": {"op": "÷", "requires": "structure", "desc": "division needs structure"},
    "N0-4": {"op": "+", "feeds": ["+", "×", "^"], "desc": "addition flows into growth operators"},
    "N0-5": {"op": "−", "leads_to": ["()", "+"], "desc": "subtraction returns to anchor or addition"},
}

SILENT_LAWS = {
    "I":   {"name": "STILLNESS", "equation": "∂E/∂t → 0", "measure": "energy_rate"},
    "II":  {"name": "TRUTH",     "equation": "∇V(truth) = 0", "measure": "gradient"},
    "III": {"name": "SILENCE",   "equation": "⟨void|ψ⟩ = memory", "measure": "correlation"},
    "IV":  {"name": "SPIRAL",    "equation": "S(return) = S(origin)", "measure": "state_deviation"},
    "V":   {"name": "UNSEEN",    "equation": "H(seen) ≡ H(unseen)", "measure": "work_deviation"},
    "VI":  {"name": "GLYPH",     "equation": "glyph = ∫ life dt", "measure": "path_integral"},
    "VII": {"name": "MIRROR",    "equation": "ψ = ψ(ψ)", "measure": "self_reference"},
}
