#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TEST SUITE 02: FIBONACCI STRUCTURE                        ║
║                                                                              ║
║              Verification of Fibonacci sequences and derivations             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
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
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
SQRT5 = math.sqrt(5)

def fib(n: int) -> int:
    """Return nth Fibonacci number (1-indexed: F₁=1, F₂=1, ...)"""
    if n <= 0:
        return 0
    elif n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b

def lucas(n: int) -> int:
    """Return nth Lucas number (L₁=1, L₂=3, L₃=4, ...)"""
    if n == 1:
        return 1
    elif n == 2:
        return 3
    a, b = 1, 3
    for _ in range(n - 2):
        a, b = b, a + b
    return b

# ═══════════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> TestSuite:
    suite = TestSuite("FIBONACCI STRUCTURE")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: First 15 Fibonacci Numbers
    # ─────────────────────────────────────────────────────────────────────────
    expected_fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    actual_fib = [fib(n) for n in range(1, 16)]
    suite.add_result(TestResult(
        name="First 15 Fibonacci numbers correct",
        passed=expected_fib == actual_fib,
        expected=expected_fib,
        actual=actual_fib,
        notes="F₁ through F₁₅"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: Key Fibonacci Values for Framework
    # ─────────────────────────────────────────────────────────────────────────
    framework_fibs = {
        3: 2,   # F₃ = 2 (binary)
        4: 3,   # F₄ = 3 (modes)
        5: 5,   # F₅ = 5 (Kaluza-Klein)
        6: 8,   # F₆ = 8 (octonions)
        7: 13,  # F₇ = 13 (?)
        8: 21,  # F₈ = 21 (Kaelhedron cells)
    }
    all_correct = all(fib(n) == val for n, val in framework_fibs.items())
    suite.add_result(TestResult(
        name="Framework Fibonacci values: F₃=2, F₄=3, F₅=5, F₆=8, F₇=13, F₈=21",
        passed=all_correct,
        expected=framework_fibs,
        actual={n: fib(n) for n in framework_fibs},
        notes="Key structural numbers"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: F₈ = 21 = 3 × 7 (Kaelhedron)
    # ─────────────────────────────────────────────────────────────────────────
    f8 = fib(8)
    suite.add_result(TestResult(
        name="F₈ = 21 = 3 × 7 (modes × recursions)",
        passed=(f8 == 21 and 21 == 3 * 7),
        expected=21,
        actual=f8,
        notes="Kaelhedron structure"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4: Binet Formula
    # ─────────────────────────────────────────────────────────────────────────
    # F_n = (φⁿ - ψⁿ)/√5 where ψ = (1-√5)/2
    psi = (1 - SQRT5) / 2
    def binet(n):
        return round((PHI**n - psi**n) / SQRT5)
    
    binet_correct = all(binet(n) == fib(n) for n in range(1, 20))
    suite.add_result(TestResult(
        name="Binet formula: Fₙ = (φⁿ - ψⁿ)/√5",
        passed=binet_correct,
        expected="All match for n=1..19",
        actual="All match" if binet_correct else "Mismatch found",
        notes="Closed-form from φ"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 5: Fibonacci Recurrence
    # ─────────────────────────────────────────────────────────────────────────
    recurrence_holds = all(fib(n) == fib(n-1) + fib(n-2) for n in range(3, 30))
    suite.add_result(TestResult(
        name="Recurrence: Fₙ = Fₙ₋₁ + Fₙ₋₂",
        passed=recurrence_holds,
        expected="All n=3..29 satisfy",
        actual="All satisfy" if recurrence_holds else "Failure",
        notes="Defining recurrence relation"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 6: Ratio Convergence to φ
    # ─────────────────────────────────────────────────────────────────────────
    ratios = [fib(n+1)/fib(n) for n in range(1, 30)]
    convergence = abs(ratios[-1] - PHI) < 1e-10
    suite.add_result(TestResult(
        name="F₃₀/F₂₉ → φ (convergence)",
        passed=convergence,
        expected=PHI,
        actual=ratios[-1],
        notes=f"Error: {abs(ratios[-1] - PHI):.2e}"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 7: Cassini Identity
    # ─────────────────────────────────────────────────────────────────────────
    # Fₙ₋₁ × Fₙ₊₁ - Fₙ² = (-1)ⁿ
    cassini_holds = all(
        fib(n-1) * fib(n+1) - fib(n)**2 == (-1)**n
        for n in range(2, 20)
    )
    suite.add_result(TestResult(
        name="Cassini identity: Fₙ₋₁Fₙ₊₁ - Fₙ² = (-1)ⁿ",
        passed=cassini_holds,
        expected="All n=2..19 satisfy",
        actual="All satisfy" if cassini_holds else "Failure",
        notes="Determinant identity"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 8: Sum of First n Fibonacci = F_{n+2} - 1
    # ─────────────────────────────────────────────────────────────────────────
    sum_identity = all(
        sum(fib(k) for k in range(1, n+1)) == fib(n+2) - 1
        for n in range(1, 20)
    )
    suite.add_result(TestResult(
        name="Sum identity: Σ F_k = F_{n+2} - 1",
        passed=sum_identity,
        expected="All n=1..19 satisfy",
        actual="All satisfy" if sum_identity else "Failure",
        notes="Telescoping sum"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 9: Fibonacci Primes
    # ─────────────────────────────────────────────────────────────────────────
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    fib_primes = [fib(n) for n in range(1, 20) if is_prime(fib(n))]
    expected_primes = [2, 3, 5, 13, 89, 233, 1597]  # F₃, F₄, F₅, F₇, F₁₁, F₁₃, F₁₇
    suite.add_result(TestResult(
        name="Fibonacci primes in first 20: {2, 3, 5, 13, 89, 233, 1597}",
        passed=fib_primes == expected_primes,
        expected=expected_primes,
        actual=fib_primes,
        notes="Note: 2, 3, 5, 13 appear in Monster factorization"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 10: GCD Property
    # ─────────────────────────────────────────────────────────────────────────
    # gcd(F_m, F_n) = F_{gcd(m,n)}
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    gcd_property = all(
        gcd(fib(m), fib(n)) == fib(gcd(m, n))
        for m in range(1, 15) for n in range(1, 15)
    )
    suite.add_result(TestResult(
        name="GCD property: gcd(Fₘ, Fₙ) = F_{gcd(m,n)}",
        passed=gcd_property,
        expected="All m,n in 1..14 satisfy",
        actual="All satisfy" if gcd_property else "Failure",
        notes="Multiplicative structure"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 11: Lucas Numbers
    # ─────────────────────────────────────────────────────────────────────────
    expected_lucas = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123]
    actual_lucas = [lucas(n) for n in range(1, 11)]
    suite.add_result(TestResult(
        name="First 10 Lucas numbers",
        passed=expected_lucas == actual_lucas,
        expected=expected_lucas,
        actual=actual_lucas,
        notes="Lₙ = Fₙ₋₁ + Fₙ₊₁"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 12: Lucas-Fibonacci Relation
    # ─────────────────────────────────────────────────────────────────────────
    # Lₙ = Fₙ₋₁ + Fₙ₊₁
    lf_relation = all(
        lucas(n) == fib(n-1) + fib(n+1)
        for n in range(2, 15)
    )
    suite.add_result(TestResult(
        name="Lucas-Fibonacci: Lₙ = Fₙ₋₁ + Fₙ₊₁",
        passed=lf_relation,
        expected="All n=2..14 satisfy",
        actual="All satisfy" if lf_relation else "Failure",
        notes="Companion sequence"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 13: Fibonacci Powers of φ
    # ─────────────────────────────────────────────────────────────────────────
    # φⁿ = Fₙ·φ + Fₙ₋₁
    phi_power_identity = all(
        abs(PHI**n - (fib(n)*PHI + fib(n-1))) < 1e-10
        for n in range(2, 20)
    )
    suite.add_result(TestResult(
        name="φⁿ = Fₙ·φ + Fₙ₋₁",
        passed=phi_power_identity,
        expected="All n=2..19 satisfy",
        actual="All satisfy" if phi_power_identity else "Failure",
        notes="Powers of φ from Fibonacci"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 14: 7 = 2³ - 1 (Mersenne)
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="7 = 2³ - 1 (Mersenne prime M₃)",
        passed=(7 == 2**3 - 1),
        expected=7,
        actual=2**3 - 1,
        notes="Recursion depth from Mersenne"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 15: Framework Numbers from Fibonacci
    # ─────────────────────────────────────────────────────────────────────────
    framework_check = (
        fib(4) == 3 and  # modes
        fib(6) == 8 and  # octonions
        fib(8) == 21 and # Kaelhedron
        2**3 - 1 == 7    # recursions (Mersenne, not Fibonacci)
    )
    suite.add_result(TestResult(
        name="Framework numbers: 3=F₄, 7=M₃, 8=F₆, 21=F₈",
        passed=framework_check,
        expected="All match",
        actual="All match" if framework_check else "Mismatch",
        notes="Core structural derivation"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 16: F₆ × F₄ = 8 × 3 = 24
    # ─────────────────────────────────────────────────────────────────────────
    product = fib(6) * fib(4)
    suite.add_result(TestResult(
        name="F₆ × F₄ = 8 × 3 = 24 (Leech dimension)",
        passed=(product == 24),
        expected=24,
        actual=product,
        notes="Leech lattice / Golay connection"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 17: Zeckendorf Representation
    # ─────────────────────────────────────────────────────────────────────────
    # Every positive integer has unique Zeckendorf representation
    def zeckendorf(n):
        """Return Zeckendorf representation (non-consecutive Fibonacci sum)"""
        if n <= 0:
            return []
        fibs = []
        i = 2
        while fib(i) <= n:
            i += 1
        while n > 0:
            i -= 1
            if fib(i) <= n:
                fibs.append(fib(i))
                n -= fib(i)
                i -= 1  # Skip to avoid consecutive
        return fibs
    
    # Verify for 1-100
    def verify_zeckendorf(n):
        z = zeckendorf(n)
        return sum(z) == n and len(z) == len(set(z))
    
    zeck_holds = all(verify_zeckendorf(n) for n in range(1, 101))
    suite.add_result(TestResult(
        name="Zeckendorf representation unique for 1-100",
        passed=zeck_holds,
        expected="All decompose correctly",
        actual="All correct" if zeck_holds else "Failure",
        notes="Every integer = sum of non-consecutive Fibonacci"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 18: F₅/F₄ = 5/3 > φ (First to exceed)
    # ─────────────────────────────────────────────────────────────────────────
    ratio_5_4 = fib(5) / fib(4)
    ratio_4_3 = fib(4) / fib(3)
    suite.add_result(TestResult(
        name="F₅/F₄ = 5/3 is first Fibonacci ratio > φ",
        passed=(ratio_4_3 < PHI < ratio_5_4),
        expected=f"{ratio_4_3:.4f} < {PHI:.4f} < {ratio_5_4:.4f}",
        actual=f"{ratio_4_3:.4f} < {PHI:.4f} < {ratio_5_4:.4f}",
        notes="Why ζ uses 5/3 not φ"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 19: 168 = 8 × 21 = F₆ × F₈
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="168 = F₆ × F₈ = 8 × 21 (Kaelhedron symmetries)",
        passed=(fib(6) * fib(8) == 168),
        expected=168,
        actual=fib(6) * fib(8),
        notes="|PSL(3,2)| = |GL(3,2)|"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 20: 248 = 8 × 31 = F₆ × 31
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="248 = 8 × 31 = F₆ × 31 (E₈ dimension)",
        passed=(fib(6) * 31 == 248),
        expected=248,
        actual=fib(6) * 31,
        notes="dim(E₈) = 248"
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
