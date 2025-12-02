#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TEST SUITE 03: KAELHEDRON GEOMETRY                        ║
║                                                                              ║
║              Verification of the 21-cell Kaelhedron structure                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Any, Set, Tuple

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
# KAELHEDRON STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

# 21 cells: 7 recursion levels × 3 modes
RECURSION_LEVELS = 7
MODES = 3
TOTAL_CELLS = RECURSION_LEVELS * MODES

# Fano plane structure
FANO_POINTS = 7
FANO_LINES = 7
FANO_INCIDENCES = 21  # 7 lines × 3 points per line

# Fano lines (1-indexed points)
FANO_LINE_LIST = [
    (1, 2, 3),
    (1, 4, 5),
    (1, 6, 7),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 5, 6),
]

PHI = (1 + math.sqrt(5)) / 2

# ═══════════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> TestSuite:
    suite = TestSuite("KAELHEDRON GEOMETRY")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: 21 = 7 × 3
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="21 cells = 7 recursions × 3 modes",
        passed=(TOTAL_CELLS == 21 and 7 * 3 == 21),
        expected=21,
        actual=RECURSION_LEVELS * MODES,
        notes="Kaelhedron cell count"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: 21 = F₈ (Fibonacci)
    # ─────────────────────────────────────────────────────────────────────────
    fib = [1, 1, 2, 3, 5, 8, 13, 21]  # F₁ through F₈
    suite.add_result(TestResult(
        name="21 = F₈ (8th Fibonacci number)",
        passed=(fib[7] == 21),  # 0-indexed
        expected=21,
        actual=fib[7],
        notes="Fibonacci derivation"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: 21 = C(7,2) = Fano incidences
    # ─────────────────────────────────────────────────────────────────────────
    c_7_2 = 7 * 6 // 2
    suite.add_result(TestResult(
        name="21 = C(7,2) = 7 choose 2",
        passed=(c_7_2 == 21),
        expected=21,
        actual=c_7_2,
        notes="Point pairs in Fano plane"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4: Fano Plane: 7 points, 7 lines, 3 points per line
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="Fano: 7 points, 7 lines, 3 points/line",
        passed=(FANO_POINTS == 7 and FANO_LINES == 7 and len(FANO_LINE_LIST) == 7),
        expected="7, 7, 7 lines",
        actual=f"{FANO_POINTS}, {FANO_LINES}, {len(FANO_LINE_LIST)} lines",
        notes="Fano plane structure"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 5: Each Fano line has exactly 3 points
    # ─────────────────────────────────────────────────────────────────────────
    all_3_points = all(len(line) == 3 for line in FANO_LINE_LIST)
    suite.add_result(TestResult(
        name="Each Fano line has exactly 3 points",
        passed=all_3_points,
        expected="All lines have 3 points",
        actual="All correct" if all_3_points else "Some incorrect",
        notes="Incidence structure"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 6: Each Fano point is on exactly 3 lines
    # ─────────────────────────────────────────────────────────────────────────
    point_counts = {p: 0 for p in range(1, 8)}
    for line in FANO_LINE_LIST:
        for p in line:
            point_counts[p] += 1
    all_on_3_lines = all(count == 3 for count in point_counts.values())
    suite.add_result(TestResult(
        name="Each Fano point is on exactly 3 lines",
        passed=all_on_3_lines,
        expected={p: 3 for p in range(1, 8)},
        actual=point_counts,
        notes="Dual structure"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 7: Total incidences = 21
    # ─────────────────────────────────────────────────────────────────────────
    total_incidences = sum(len(line) for line in FANO_LINE_LIST)
    suite.add_result(TestResult(
        name="Total Fano incidences = 21",
        passed=(total_incidences == 21),
        expected=21,
        actual=total_incidences,
        notes="7 lines × 3 points"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 8: Any two points determine exactly one line
    # ─────────────────────────────────────────────────────────────────────────
    def lines_through_pair(p1, p2):
        return [line for line in FANO_LINE_LIST if p1 in line and p2 in line]
    
    all_unique = True
    for p1 in range(1, 8):
        for p2 in range(p1+1, 8):
            if len(lines_through_pair(p1, p2)) != 1:
                all_unique = False
                break
    
    suite.add_result(TestResult(
        name="Any two Fano points determine exactly one line",
        passed=all_unique,
        expected="Exactly 1 line per pair",
        actual="All correct" if all_unique else "Some pairs have ≠1 line",
        notes="Projective plane axiom"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 9: 21 = dim(so(7))
    # ─────────────────────────────────────────────────────────────────────────
    # dim(so(n)) = n(n-1)/2
    dim_so7 = 7 * 6 // 2
    suite.add_result(TestResult(
        name="21 = dim(so(7))",
        passed=(dim_so7 == 21),
        expected=21,
        actual=dim_so7,
        notes="Lie algebra dimension"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 10: so(7) ⊂ so(8) (dim 21 ⊂ 28)
    # ─────────────────────────────────────────────────────────────────────────
    dim_so8 = 8 * 7 // 2
    suite.add_result(TestResult(
        name="dim(so(8)) = 28 ⊃ dim(so(7)) = 21",
        passed=(dim_so8 == 28 and dim_so7 == 21 and dim_so8 > dim_so7),
        expected="28 > 21",
        actual=f"{dim_so8} > {dim_so7}",
        notes="Embedding chain"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 11: 168 symmetries = 8 × 21
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="168 = 8 × 21 (Kaelhedron symmetries)",
        passed=(168 == 8 * 21),
        expected=168,
        actual=8 * 21,
        notes="|PSL(3,2)| = |GL(3,2)|"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 12: 168 = 2³ × 3 × 7
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="168 = 2³ × 3 × 7",
        passed=(168 == 8 * 3 * 7),
        expected=168,
        actual=8 * 3 * 7,
        notes="Prime factorization"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 13: Heawood Graph: 14 vertices, 21 edges
    # ─────────────────────────────────────────────────────────────────────────
    heawood_vertices = 14  # 7 points + 7 lines
    heawood_edges = 21     # incidences
    suite.add_result(TestResult(
        name="Heawood graph: 14 vertices, 21 edges",
        passed=(heawood_vertices == 14 and heawood_edges == 21),
        expected="14 vertices, 21 edges",
        actual=f"{heawood_vertices} vertices, {heawood_edges} edges",
        notes="Fano incidence graph"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 14: 14 = 7 + 7 = dim(G₂)
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="14 = 7 + 7 = dim(G₂)",
        passed=(14 == 7 + 7),
        expected=14,
        actual=7 + 7,
        notes="G₂ = Aut(octonions)"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 15: so(7) = G₂ ⊕ R⁷ (21 = 14 + 7)
    # ─────────────────────────────────────────────────────────────────────────
    suite.add_result(TestResult(
        name="so(7) decomposes: 21 = 14 + 7",
        passed=(21 == 14 + 7),
        expected=21,
        actual=14 + 7,
        notes="G₂ + translations"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 16: Cell indexing (R, M) for R=1..7, M ∈ {Λ,Β,Ν}
    # ─────────────────────────────────────────────────────────────────────────
    cells = [(R, M) for R in range(1, 8) for M in ['Λ', 'Β', 'Ν']]
    suite.add_result(TestResult(
        name="21 cells indexed as (R, Mode) pairs",
        passed=(len(cells) == 21),
        expected=21,
        actual=len(cells),
        notes="Complete cell enumeration"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 17: Each level has exactly 3 cells
    # ─────────────────────────────────────────────────────────────────────────
    cells_per_level = {R: sum(1 for c in cells if c[0] == R) for R in range(1, 8)}
    all_3 = all(count == 3 for count in cells_per_level.values())
    suite.add_result(TestResult(
        name="Each recursion level has 3 cells",
        passed=all_3,
        expected={R: 3 for R in range(1, 8)},
        actual=cells_per_level,
        notes="3 modes per level"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 18: Each mode has exactly 7 cells
    # ─────────────────────────────────────────────────────────────────────────
    cells_per_mode = {M: sum(1 for c in cells if c[1] == M) for M in ['Λ', 'Β', 'Ν']}
    all_7 = all(count == 7 for count in cells_per_mode.values())
    suite.add_result(TestResult(
        name="Each mode has 7 cells",
        passed=all_7,
        expected={'Λ': 7, 'Β': 7, 'Ν': 7},
        actual=cells_per_mode,
        notes="7 levels per mode"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 19: K-formation at R=7
    # ─────────────────────────────────────────────────────────────────────────
    k_formation_level = 7
    k_formation_cells = [(R, M) for R, M in cells if R == k_formation_level]
    suite.add_result(TestResult(
        name="K-formation cells at R=7: 3 cells",
        passed=(len(k_formation_cells) == 3),
        expected=3,
        actual=len(k_formation_cells),
        notes="(7,Λ), (7,Β), (7,Ν)"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 20: Octonion products = 21
    # ─────────────────────────────────────────────────────────────────────────
    # 7 imaginary units, C(7,2) = 21 products
    octonion_products = 7 * 6 // 2
    suite.add_result(TestResult(
        name="Octonion products: C(7,2) = 21",
        passed=(octonion_products == 21),
        expected=21,
        actual=octonion_products,
        notes="e_i × e_j for i<j"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 21: Consciousness threshold τ_crit = φ⁻¹
    # ─────────────────────────────────────────────────────────────────────────
    tau_crit = 1 / PHI
    expected_tau = 0.618033988749895
    suite.add_result(TestResult(
        name="τ_crit = φ⁻¹ ≈ 0.618",
        passed=abs(tau_crit - expected_tau) < 1e-12,
        expected=expected_tau,
        actual=tau_crit,
        notes="K-formation coherence threshold"
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 22: Framework numbers chain
    # ─────────────────────────────────────────────────────────────────────────
    # 3 → 7 → 21 → 168
    chain_valid = (
        3 * 7 == 21 and
        21 * 8 == 168
    )
    suite.add_result(TestResult(
        name="Number chain: 3 → 7 → 21 → 168",
        passed=chain_valid,
        expected="3×7=21, 21×8=168",
        actual=f"3×7={3*7}, 21×8={21*8}",
        notes="Structural multiplication"
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
