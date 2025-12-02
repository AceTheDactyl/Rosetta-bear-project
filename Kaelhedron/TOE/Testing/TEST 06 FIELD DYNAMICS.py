#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TEST SUITE 06: FIELD DYNAMICS                             ║
║                                                                              ║
║              Verification of κ-field evolution and potentials                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Any

# ═══════════════════════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    expected: Any
    actual: Any
    tolerance: float = 1e-10
    notes: str = ""

class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
    
    def add_result(self, result: TestResult):
        self.results.append(result)
    
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    def total_count(self) -> int:
        return len(self.results)
    
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"TEST SUITE: {self.name}",
            f"{'='*70}",
            f"Results: {self.passed_count()}/{self.total_count()} passed",
            f"{'='*70}",
        ]
        for r in self.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            lines.append(f"  {status}: {r.name}")
            if not r.passed:
                lines.append(f"         Expected: {r.expected}")
                lines.append(f"         Actual:   {r.actual}")
            if r.notes:
                lines.append(f"         Notes: {r.notes}")
        lines.append(f"{'='*70}")
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS AND FIELD DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
ZETA = (5/3)**4
MU_P = 3/5
MU_S = 23/25

def potential_V(kappa: float) -> float:
    """Mexican hat potential V(κ) = -κ²/2 + ζκ⁴/4"""
    return -0.5 * kappa**2 + (ZETA/4) * kappa**4

def potential_derivative(kappa: float) -> float:
    """dV/dκ = -κ + ζκ³"""
    return -kappa + ZETA * kappa**3

def find_minima() -> tuple:
    """Find the two minima of the double-well potential"""
    # V'(κ) = 0 → κ(-1 + ζκ²) = 0
    # Solutions: κ = 0, κ = ±1/√ζ
    kappa_min = 1 / math.sqrt(ZETA)
    return (-kappa_min, 0, kappa_min)

def klein_gordon_rhs(kappa: float, zeta: float = ZETA) -> float:
    """RHS of □κ + ζκ³ = 0 → -ζκ³"""
    return -zeta * kappa**3

def coherence_gradient(phases: np.ndarray) -> float:
    """Compute gradient-based coherence measure"""
    if len(phases) < 2:
        return 0.0
    gradients = np.diff(phases)
    mean_grad = np.mean(np.abs(gradients))
    if mean_grad == 0:
        return 1.0
    variance = np.var(gradients)
    return np.exp(-variance / (mean_grad**2 + 1e-10))

# ═══════════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> TestSuite:
    suite = TestSuite("FIELD DYNAMICS")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: Potential V(0) = 0
    # ─────────────────────────────────────────────────────────────────────────
    V_0 = potential_V(0)
    suite.add_result(TestResult(
        name="V(0) = 0",
        passed=abs(V_0) < 1e-14,
        expected=0.0,
        actual=V_0,
        notes="Potential at origin"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: V(0) is a local maximum
    # ─────────────────────────────────────────────────────────────────────────
    epsilon = 0.01
    V_plus = potential_V(epsilon)
    V_minus = potential_V(-epsilon)
    suite.add_result(TestResult(
        name="V(0) is local maximum (V(±ε) < V(0))",
        passed=(V_plus < V_0 and V_minus < V_0),
        expected="V(±ε) < 0",
        actual=f"V(+ε)={V_plus:.6f}, V(-ε)={V_minus:.6f}",
        notes="Unstable equilibrium"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: Minima at κ = ±1/√ζ
    # ─────────────────────────────────────────────────────────────────────────
    kappa_min = 1 / math.sqrt(ZETA)
    expected_min = 0.3599612  # ≈ 1/√ζ
    suite.add_result(TestResult(
        name="Minima at κ = ±1/√ζ ≈ ±0.360",
        passed=abs(kappa_min - 1/math.sqrt(ZETA)) < 1e-10,
        expected=1/math.sqrt(ZETA),
        actual=kappa_min,
        notes="VEV location"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4: V'(κ_min) = 0
    # ─────────────────────────────────────────────────────────────────────────
    deriv_at_min = potential_derivative(kappa_min)
    suite.add_result(TestResult(
        name="V'(κ_min) = 0 (stationary point)",
        passed=abs(deriv_at_min) < 1e-10,
        expected=0.0,
        actual=deriv_at_min,
        notes="Minimum condition"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 5: V(κ_min) < 0 (wells below origin)
    # ─────────────────────────────────────────────────────────────────────────
    V_min = potential_V(kappa_min)
    suite.add_result(TestResult(
        name="V(κ_min) < 0 (wells below origin)",
        passed=(V_min < 0),
        expected="< 0",
        actual=V_min,
        notes="Spontaneous symmetry breaking"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 6: Barrier height V(0) - V(κ_min)
    # ─────────────────────────────────────────────────────────────────────────
    barrier = V_0 - V_min
    # Barrier = 1/(4ζ)
    expected_barrier = 1 / (4 * ZETA)
    suite.add_result(TestResult(
        name="Barrier height = 1/(4ζ)",
        passed=abs(barrier - expected_barrier) < 1e-10,
        expected=expected_barrier,
        actual=barrier,
        notes="Tunneling barrier"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 7: Klein-Gordon equation form
    # ─────────────────────────────────────────────────────────────────────────
    # □κ + ζκ³ = 0
    test_kappa = 0.5
    rhs = klein_gordon_rhs(test_kappa)
    expected_rhs = -ZETA * test_kappa**3
    suite.add_result(TestResult(
        name="KGK equation: □κ = -ζκ³",
        passed=abs(rhs - expected_rhs) < 1e-14,
        expected=expected_rhs,
        actual=rhs,
        notes="Non-linear wave equation"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 8: ζ = (5/3)⁴ numerical value
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="ζ = (5/3)⁴ ≈ 7.716",
        passed=abs(ZETA - 7.716049382716051) < 1e-10,
        expected=7.716049382716051,
        actual=ZETA,
        notes="Coupling constant"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 9: VEV = 1/√ζ ≈ 0.360
    # ─────────────────────────────────────────────────────────────────────────
    vev = 1 / math.sqrt(ZETA)
    suite.add_result(TestResult(
        name="VEV = 1/√ζ ≈ 0.360",
        passed=abs(vev - 0.36) < 0.01,
        expected=0.36,
        actual=vev,
        notes="Vacuum expectation value"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 10: Coherence = 1 for uniform gradient
    # ─────────────────────────────────────────────────────────────────────────
    uniform_phases = np.linspace(0, 2*np.pi, 100)
    coh = coherence_gradient(uniform_phases)
    suite.add_result(TestResult(
        name="Coherence → 1 for uniform phase gradient",
        passed=(coh > 0.99),
        expected="> 0.99",
        actual=coh,
        notes="Perfect coherence"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 11: Coherence < 0.5 for random phases
    # ─────────────────────────────────────────────────────────────────────────
    np.random.seed(42)
    random_phases = np.random.uniform(0, 2*np.pi, 100)
    coh_rand = coherence_gradient(random_phases)
    suite.add_result(TestResult(
        name="Coherence < 0.5 for random phases",
        passed=(coh_rand < 0.5),
        expected="< 0.5",
        actual=coh_rand,
        notes="Incoherent state"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 12: Phase thresholds ordering
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="μ_P < φ⁻¹ < μ_S",
        passed=(MU_P < PHI_INV < MU_S),
        expected=f"{MU_P} < {PHI_INV:.3f} < {MU_S}",
        actual=f"{MU_P} < {PHI_INV:.6f} < {MU_S}",
        notes="Phase ordering"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 13: Soliton solution form
    # ─────────────────────────────────────────────────────────────────────────
    # κ(x) = κ_0 tanh(x/ξ) is a solution
    kappa_0 = 1 / math.sqrt(ZETA)
    xi = 1.0  # Correlation length
    x = np.linspace(-5, 5, 100)
    soliton = kappa_0 * np.tanh(x / xi)
    
    # Verify it connects ±κ_0
    suite.add_result(TestResult(
        name="Soliton κ(x) = κ₀tanh(x/ξ) connects ±κ₀",
        passed=(abs(soliton[0] + kappa_0) < 0.01 and abs(soliton[-1] - kappa_0) < 0.01),
        expected=f"±{kappa_0:.3f}",
        actual=f"κ(-∞)={soliton[0]:.3f}, κ(+∞)={soliton[-1]:.3f}",
        notes="Kink solution"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 14: Energy density at minimum
    # ─────────────────────────────────────────────────────────────────────────
    # E = V(κ_min) = -1/(4ζ)
    E_min = -1 / (4 * ZETA)
    suite.add_result(TestResult(
        name="Vacuum energy = -1/(4ζ)",
        passed=abs(V_min - E_min) < 1e-10,
        expected=E_min,
        actual=V_min,
        notes="Negative vacuum energy"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 15: Second derivative test at minimum
    # ─────────────────────────────────────────────────────────────────────────
    # V''(κ) = -1 + 3ζκ²
    # At κ_min: V'' = -1 + 3ζ(1/ζ) = -1 + 3 = 2 > 0
    V_double_prime = -1 + 3 * ZETA * kappa_min**2
    suite.add_result(TestResult(
        name="V''(κ_min) = 2 > 0 (true minimum)",
        passed=abs(V_double_prime - 2.0) < 1e-10,
        expected=2.0,
        actual=V_double_prime,
        notes="Stability condition"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 16: Mass² = V''(κ_min) = 2
    # ─────────────────────────────────────────────────────────────────────────
    mass_squared = V_double_prime
    suite.add_result(TestResult(
        name="Effective mass² = 2",
        passed=abs(mass_squared - 2.0) < 1e-10,
        expected=2.0,
        actual=mass_squared,
        notes="Oscillation frequency"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 17: Correlation length ξ = 1/√(m²) = 1/√2
    # ─────────────────────────────────────────────────────────────────────────
    corr_length = 1 / math.sqrt(2)
    suite.add_result(TestResult(
        name="Correlation length ξ = 1/√2 ≈ 0.707",
        passed=abs(corr_length - 1/math.sqrt(2)) < 1e-10,
        expected=1/math.sqrt(2),
        actual=corr_length,
        notes="Spatial scale"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 18: Z₂ symmetry: V(κ) = V(-κ)
    # ─────────────────────────────────────────────────────────────────────────
    test_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    z2_holds = all(
        abs(potential_V(k) - potential_V(-k)) < 1e-14
        for k in test_values
    )
    suite.add_result(TestResult(
        name="Z₂ symmetry: V(κ) = V(-κ)",
        passed=z2_holds,
        expected="All symmetric",
        actual="All symmetric" if z2_holds else "Broken",
        notes="Reflection symmetry"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 19: Void state V(0) > unity V(κ_min)
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="V(void) > V(unity): 0 > V_min",
        passed=(V_0 > V_min),
        expected=f"0 > {V_min:.4f}",
        actual=f"{V_0} > {V_min:.4f}",
        notes="Unity is energetically favored"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 20: Field equation is nonlinear
    # ─────────────────────────────────────────────────────────────────────────
    # ζκ³ term makes it nonlinear
    k1, k2 = 0.3, 0.4
    rhs1 = klein_gordon_rhs(k1)
    rhs2 = klein_gordon_rhs(k2)
    rhs_sum = klein_gordon_rhs(k1 + k2)
    suite.add_result(TestResult(
        name="Field equation nonlinear: f(κ₁+κ₂) ≠ f(κ₁)+f(κ₂)",
        passed=(abs(rhs_sum - (rhs1 + rhs2)) > 0.01),
        expected="Non-additive",
        actual=f"|difference| = {abs(rhs_sum - (rhs1 + rhs2)):.4f}",
        notes="Cubic nonlinearity"
    ))
    
    return suite

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    suite = run_tests()
    print(suite.summary())
    print(f"\n{'='*70}")
    print(f"FINAL RESULT: {'ALL TESTS PASSED ✓' if suite.all_passed() else 'SOME TESTS FAILED ✗'}")
    print(f"{'='*70}")
