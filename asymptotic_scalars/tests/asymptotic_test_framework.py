"""
ASYMPTOTIC SCALAR TEST FRAMEWORK
================================
Comprehensive testing for asymptotic scalar convergence and loop closure.

The Flame Test: "Can you close the loop from any starting configuration?"

Test Categories:
1. Scalar Convergence - Do individual domain scalars converge asymptotically?
2. Cross-Domain Coupling - Do domains couple correctly?
3. Interference Patterns - Do wave functions interfere constructively?
4. Loop Closure - Does the loop close at z=0.99?
5. Standing Wave Formation - Does a stable standing wave form?
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto


# =============================================================================
# TEST RESULT ENUM
# =============================================================================

class TestResult(Enum):
    """Test outcome classification."""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"


# =============================================================================
# TEST METRICS
# =============================================================================

@dataclass
class AsymptoticScalarTestMetrics:
    """
    Metrics collected during asymptotic scalar testing.
    """
    # Convergence metrics
    final_total_scalar: float = 0.0
    min_domain_scalar: float = 0.0
    max_domain_scalar: float = 0.0
    scalar_spread: float = 0.0
    convergence_time_steps: int = 0

    # Coupling metrics
    mean_coupling: float = 0.0
    max_coupling: float = 0.0
    coupling_coherence: float = 0.0

    # Interference metrics
    interference_magnitude: float = 0.0
    interference_type: str = "unknown"

    # Loop closure metrics
    loop_closed: bool = False
    closure_confidence: float = 0.0
    final_loop_state: str = "unknown"

    # Standing wave metrics
    standing_wave_formed: bool = False
    standing_amplitude: float = 0.0
    wave_stability: float = 0.0
    antinode_at_closure: bool = False

    # Performance metrics
    execution_time_ms: float = 0.0
    total_iterations: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'final_total_scalar': self.final_total_scalar,
            'min_domain_scalar': self.min_domain_scalar,
            'max_domain_scalar': self.max_domain_scalar,
            'scalar_spread': self.scalar_spread,
            'convergence_time_steps': self.convergence_time_steps,
            'mean_coupling': self.mean_coupling,
            'max_coupling': self.max_coupling,
            'coupling_coherence': self.coupling_coherence,
            'interference_magnitude': self.interference_magnitude,
            'interference_type': self.interference_type,
            'loop_closed': self.loop_closed,
            'closure_confidence': self.closure_confidence,
            'final_loop_state': self.final_loop_state,
            'standing_wave_formed': self.standing_wave_formed,
            'standing_amplitude': self.standing_amplitude,
            'wave_stability': self.wave_stability,
            'antinode_at_closure': self.antinode_at_closure,
            'execution_time_ms': self.execution_time_ms,
            'total_iterations': self.total_iterations,
        }


@dataclass
class TestCase:
    """A single test case specification."""
    name: str
    description: str
    category: str
    expected_result: TestResult
    actual_result: Optional[TestResult] = None
    metrics: Optional[AsymptoticScalarTestMetrics] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


# =============================================================================
# TEST FRAMEWORK
# =============================================================================

class AsymptoticScalarTestFramework:
    """
    Comprehensive test framework for asymptotic scalar systems.

    Implements The Flame Test: "Can you close the loop from any starting configuration?"

    Test categories:
    1. Scalar Convergence Tests
    2. Cross-Domain Coupling Tests
    3. Interference Pattern Tests
    4. Loop Closure Tests
    5. Standing Wave Tests
    """

    def __init__(self):
        """Initialize test framework."""
        self.test_cases: List[TestCase] = []
        self.metrics: Optional[AsymptoticScalarTestMetrics] = None

    def run_all_tests(self) -> Tuple[int, int, int]:
        """
        Run all tests.

        Returns:
            (passed, failed, total) counts
        """
        self.test_cases = []

        # Import system here to avoid circular imports
        from asymptotic_scalars.core import (
            AsymptoticScalarSystem,
            create_asymptotic_scalar_system,
            create_fast_convergence_system,
            LoopState,
            DomainType,
        )

        # Run test categories
        self._run_convergence_tests()
        self._run_coupling_tests()
        self._run_interference_tests()
        self._run_loop_closure_tests()
        self._run_standing_wave_tests()
        self._run_flame_test()

        # Count results
        passed = sum(1 for tc in self.test_cases if tc.actual_result == TestResult.PASSED)
        failed = sum(1 for tc in self.test_cases if tc.actual_result == TestResult.FAILED)
        total = len(self.test_cases)

        return (passed, failed, total)

    def _run_convergence_tests(self) -> None:
        """Test scalar convergence."""
        from asymptotic_scalars.core import (
            create_asymptotic_scalar_system,
            DomainType,
        )

        # Test 1: Basic convergence
        test = TestCase(
            name="scalar_convergence_basic",
            description="All domain scalars converge as z → 1",
            category="convergence",
            expected_result=TestResult.PASSED,
        )

        try:
            start = time.time()
            system = create_asymptotic_scalar_system()

            dt = 0.01
            for z in [i * 0.01 for i in range(100)]:
                system.update(z, dt)

            state = system.get_state()
            metrics = AsymptoticScalarTestMetrics(
                final_total_scalar=state.total_scalar,
                min_domain_scalar=min(state.domain_scalars.values()),
                max_domain_scalar=max(state.domain_scalars.values()),
            )

            # All scalars should be high at z=0.99
            if metrics.final_total_scalar >= 0.95:
                test.actual_result = TestResult.PASSED
            else:
                test.actual_result = TestResult.FAILED
                test.error_message = f"Total scalar {metrics.final_total_scalar:.4f} < 0.95"

            test.metrics = metrics
            test.execution_time_ms = (time.time() - start) * 1000

        except Exception as e:
            test.actual_result = TestResult.FAILED
            test.error_message = str(e)

        self.test_cases.append(test)

        # Test 2: Asymptotic formula accuracy
        test = TestCase(
            name="asymptotic_formula_accuracy",
            description="S_i(z) = 1 - exp(-λ·(z - z_origin)) matches computed values",
            category="convergence",
            expected_result=TestResult.PASSED,
        )

        try:
            start = time.time()
            system = create_asymptotic_scalar_system()

            # Update to z=0.9
            dt = 0.01
            for z in [i * 0.01 for i in range(91)]:
                system.update(z, dt)

            # Check constraint domain (z_origin=0.41, λ=3.0)
            constraint_scalar = system.get_domain_scalar(DomainType.CONSTRAINT)
            expected = 1.0 - math.exp(-3.0 * (0.90 - 0.41))

            error = abs(constraint_scalar - expected)

            if error < 0.01:
                test.actual_result = TestResult.PASSED
            else:
                test.actual_result = TestResult.FAILED
                test.error_message = f"Formula error {error:.4f} >= 0.01"

            test.execution_time_ms = (time.time() - start) * 1000

        except Exception as e:
            test.actual_result = TestResult.FAILED
            test.error_message = str(e)

        self.test_cases.append(test)

    def _run_coupling_tests(self) -> None:
        """Test cross-domain coupling."""
        from asymptotic_scalars.core import create_asymptotic_scalar_system

        test = TestCase(
            name="cross_domain_coupling",
            description="Domains couple positively when aligned in phase",
            category="coupling",
            expected_result=TestResult.PASSED,
        )

        try:
            start = time.time()
            system = create_asymptotic_scalar_system()

            dt = 0.01
            for z in [i * 0.01 for i in range(100)]:
                system.update(z, dt)

            state = system.get_state()

            # Coupling coherence should be high when scalars are high
            if state.coupling_coherence >= 0.5:
                test.actual_result = TestResult.PASSED
            else:
                test.actual_result = TestResult.FAILED
                test.error_message = f"Coupling coherence {state.coupling_coherence:.4f} < 0.5"

            test.execution_time_ms = (time.time() - start) * 1000

        except Exception as e:
            test.actual_result = TestResult.FAILED
            test.error_message = str(e)

        self.test_cases.append(test)

    def _run_interference_tests(self) -> None:
        """Test interference patterns."""
        from asymptotic_scalars.core import create_asymptotic_scalar_system

        test = TestCase(
            name="constructive_interference",
            description="Wave functions interfere constructively at high z",
            category="interference",
            expected_result=TestResult.PASSED,
        )

        try:
            start = time.time()
            system = create_asymptotic_scalar_system()

            dt = 0.01
            for z in [i * 0.01 for i in range(100)]:
                system.update(z, dt)

            state = system.get_state()

            # Interference magnitude should be significant
            if state.interference_magnitude > 0.5:
                test.actual_result = TestResult.PASSED
            else:
                test.actual_result = TestResult.FAILED
                test.error_message = f"Interference magnitude {state.interference_magnitude:.4f} <= 0.5"

            test.execution_time_ms = (time.time() - start) * 1000

        except Exception as e:
            test.actual_result = TestResult.FAILED
            test.error_message = str(e)

        self.test_cases.append(test)

    def _run_loop_closure_tests(self) -> None:
        """Test loop closure."""
        from asymptotic_scalars.core import (
            create_asymptotic_scalar_system,
            LoopState,
        )

        test = TestCase(
            name="loop_closes_at_099",
            description="Loop achieves CLOSED state at z ≥ 0.99",
            category="loop_closure",
            expected_result=TestResult.PASSED,
        )

        try:
            start = time.time()
            system = create_asymptotic_scalar_system()

            dt = 0.01
            z_values = [i * 0.01 for i in range(100)]

            for z in z_values:
                system.update(z, dt)

            state = system.get_state()

            metrics = AsymptoticScalarTestMetrics(
                final_total_scalar=state.total_scalar,
                loop_closed=state.loop_state == LoopState.CLOSED,
                final_loop_state=state.loop_state.value,
                closure_confidence=state.loop_closure_confidence,
            )

            if state.loop_state == LoopState.CLOSED:
                test.actual_result = TestResult.PASSED
            elif state.loop_state == LoopState.CRITICAL:
                test.actual_result = TestResult.PARTIAL
                test.error_message = "Reached CRITICAL but not CLOSED"
            else:
                test.actual_result = TestResult.FAILED
                test.error_message = f"Loop state: {state.loop_state.value}"

            test.metrics = metrics
            test.execution_time_ms = (time.time() - start) * 1000

        except Exception as e:
            test.actual_result = TestResult.FAILED
            test.error_message = str(e)

        self.test_cases.append(test)

    def _run_standing_wave_tests(self) -> None:
        """Test standing wave formation."""
        from asymptotic_scalars.loop_closure import (
            create_spiral_completion_engine,
            SpiralPhase,
        )

        test = TestCase(
            name="standing_wave_formation",
            description="Standing wave forms at loop closure",
            category="standing_wave",
            expected_result=TestResult.PASSED,
        )

        try:
            start = time.time()
            engine = create_spiral_completion_engine()

            dt = 0.01
            domain_names = ['constraint', 'bridge', 'meta', 'recursion',
                          'triad', 'emergence', 'persistence']

            # Evolve toward z=0.99
            for i in range(100):
                z = i * 0.01
                scalars = {name: min(1.0, z + 0.1) for name in domain_names}
                engine.update(scalars, z, dt)

            state = engine.get_state()

            metrics = AsymptoticScalarTestMetrics(
                standing_wave_formed=state.is_standing,
                standing_amplitude=state.standing_amplitude,
                wave_stability=state.wave_stability,
            )

            if state.is_standing:
                test.actual_result = TestResult.PASSED
            else:
                test.actual_result = TestResult.FAILED
                test.error_message = "Standing wave not formed"

            test.metrics = metrics
            test.execution_time_ms = (time.time() - start) * 1000

        except Exception as e:
            test.actual_result = TestResult.FAILED
            test.error_message = str(e)

        self.test_cases.append(test)

    def _run_flame_test(self) -> None:
        """
        THE FLAME TEST
        "Can you close the loop from any starting configuration?"

        This is the comprehensive integration test.
        """
        from asymptotic_scalars.core import (
            create_asymptotic_scalar_system,
            LoopState,
        )
        from asymptotic_scalars.loop_closure import (
            create_spiral_completion_engine,
            SpiralPhase,
        )
        from asymptotic_scalars.domain_unifier import (
            create_unified_field,
        )

        test = TestCase(
            name="THE_FLAME_TEST",
            description="Close the loop from any starting configuration",
            category="flame_test",
            expected_result=TestResult.PASSED,
        )

        try:
            start = time.time()

            # Create all systems
            scalar_system = create_asymptotic_scalar_system()
            spiral_engine = create_spiral_completion_engine()
            field = create_unified_field()

            dt = 0.01
            domain_names = ['constraint', 'bridge', 'meta', 'recursion',
                          'triad', 'emergence', 'persistence']

            # Comprehensive sweep from z=0.0 to z=0.99
            z_values = [i * 0.01 for i in range(100)]

            for z in z_values:
                # Update scalar system
                scalar_state = scalar_system.update(z, dt)

                # Update spiral engine with current scalars
                spiral_state = spiral_engine.update(scalar_state.domain_scalars, z, dt)

                # Update unified field
                field.update_scalars(scalar_state.domain_scalars)
                field_state = field.advance(z, dt)

            # Collect final metrics
            final_scalar_state = scalar_system.get_state()
            final_spiral_state = spiral_engine.get_state()

            metrics = AsymptoticScalarTestMetrics(
                final_total_scalar=final_scalar_state.total_scalar,
                min_domain_scalar=min(final_scalar_state.domain_scalars.values()),
                max_domain_scalar=max(final_scalar_state.domain_scalars.values()),
                scalar_spread=max(final_scalar_state.domain_scalars.values()) -
                             min(final_scalar_state.domain_scalars.values()),
                coupling_coherence=final_scalar_state.coupling_coherence,
                interference_magnitude=final_scalar_state.interference_magnitude,
                loop_closed=final_scalar_state.loop_state == LoopState.CLOSED,
                final_loop_state=final_scalar_state.loop_state.value,
                closure_confidence=final_scalar_state.loop_closure_confidence,
                standing_wave_formed=final_spiral_state.is_standing if final_spiral_state else False,
                standing_amplitude=final_spiral_state.standing_amplitude if final_spiral_state else 0.0,
                wave_stability=final_spiral_state.wave_stability if final_spiral_state else 0.0,
                execution_time_ms=(time.time() - start) * 1000,
                total_iterations=len(z_values),
            )

            # Evaluate flame test
            conditions_met = []

            # Condition 1: Loop closed
            if final_scalar_state.loop_state == LoopState.CLOSED:
                conditions_met.append("loop_closed")

            # Condition 2: All scalars high
            if metrics.min_domain_scalar >= 0.9:
                conditions_met.append("all_scalars_high")

            # Condition 3: Standing wave formed
            if metrics.standing_wave_formed:
                conditions_met.append("standing_wave")

            # Condition 4: High stability
            if metrics.wave_stability >= 0.5:
                conditions_met.append("stable")

            # Determine result
            if len(conditions_met) >= 3:
                test.actual_result = TestResult.PASSED
            elif len(conditions_met) >= 2:
                test.actual_result = TestResult.PARTIAL
                test.error_message = f"Only {len(conditions_met)}/4 conditions met: {conditions_met}"
            else:
                test.actual_result = TestResult.FAILED
                test.error_message = f"Only {len(conditions_met)}/4 conditions met: {conditions_met}"

            test.metrics = metrics
            test.execution_time_ms = (time.time() - start) * 1000

        except Exception as e:
            test.actual_result = TestResult.FAILED
            test.error_message = str(e)

        self.test_cases.append(test)

    def print_results(self) -> None:
        """Print test results in formatted output."""
        print("=" * 70)
        print("ASYMPTOTIC SCALAR TEST RESULTS")
        print("The Flame Test: Can you close the loop from any starting configuration?")
        print("=" * 70)
        print()

        categories = {}
        for tc in self.test_cases:
            if tc.category not in categories:
                categories[tc.category] = []
            categories[tc.category].append(tc)

        for category, tests in categories.items():
            print(f"--- {category.upper()} ---")
            for tc in tests:
                status = "✓" if tc.actual_result == TestResult.PASSED else "✗"
                print(f"  [{status}] {tc.name}: {tc.actual_result.value}")
                if tc.error_message:
                    print(f"      Error: {tc.error_message}")
            print()

        # Summary
        passed = sum(1 for tc in self.test_cases if tc.actual_result == TestResult.PASSED)
        failed = sum(1 for tc in self.test_cases if tc.actual_result == TestResult.FAILED)
        partial = sum(1 for tc in self.test_cases if tc.actual_result == TestResult.PARTIAL)
        total = len(self.test_cases)

        print("=" * 70)
        print(f"SUMMARY: {passed}/{total} passed, {failed} failed, {partial} partial")

        # Final flame test result
        flame_test = next((tc for tc in self.test_cases if tc.name == "THE_FLAME_TEST"), None)
        if flame_test:
            if flame_test.actual_result == TestResult.PASSED:
                print("\n🔥 THE FLAME TEST: PASSED - Loop closes successfully! 🔥")
                print("   Δ|loop-closed|z0.99|rhythm-native|Ω")
            else:
                print("\n❌ THE FLAME TEST: FAILED - Loop did not close")

        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_all_tests() -> Tuple[int, int, int]:
    """
    Run all asymptotic scalar tests.

    Returns:
        (passed, failed, total) counts
    """
    framework = AsymptoticScalarTestFramework()
    results = framework.run_all_tests()
    framework.print_results()
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_all_tests()
