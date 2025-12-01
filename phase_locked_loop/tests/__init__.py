"""
PHASE-LOCKED LOOP TEST FRAMEWORK
================================
Validation suite for PLL system functionality.

The Flame Test: "Can you hold lock through a frequency step?"

Test Categories:
1. Acquisition Tests - Lock acquisition timing
2. Precision Tests - Steady-state accuracy
3. Robustness Tests - Frequency step tracking
4. Stability Tests - Long-term performance

Signature: Δ3.142|0.995|1.000Ω
"""

__version__ = "1.0.0"
__z_level__ = 0.995

from phase_locked_loop.tests.pll_test_framework import (
    TestResult,
    TestMetrics,
    PLLTestConfig,
    PLLTestFramework,
    run_flame_test,
    run_acquisition_test,
    run_precision_test,
    run_robustness_test,
    run_stability_test,
)

__all__ = [
    "TestResult",
    "TestMetrics",
    "PLLTestConfig",
    "PLLTestFramework",
    "run_flame_test",
    "run_acquisition_test",
    "run_precision_test",
    "run_robustness_test",
    "run_stability_test",
]
