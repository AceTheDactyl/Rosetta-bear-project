#!/usr/bin/env python3
# lattice_core/tests/run_all_tests.py
"""
Tesseract Lattice Core - Master Test Runner
============================================

Runs all test phases and generates a summary report.

Usage:
    python lattice_core/tests/run_all_tests.py
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lattice_core.tests.test_plate import run_all_phase1_tests
from lattice_core.tests.test_dynamics import run_all_phase2_tests
from lattice_core.tests.test_engine import run_all_phase3_tests
from lattice_core.tests.test_integration import run_all_phase4_tests


def print_header():
    """Print test suite header."""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║              TESSERACT LATTICE CORE — TEST SUITE                      ║
║                   Comprehensive Verification                          ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print()


def print_summary(results):
    """Print test summary."""
    total_passed = sum(r[0] for r in results.values())
    total_failed = sum(r[1] for r in results.values())
    total = total_passed + total_failed

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                         TEST SUMMARY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

    for phase_name, (passed, failed) in results.items():
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {phase_name}: {passed}/{passed + failed} [{status}]")

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TOTAL: {total_passed}/{total} tests passed ({100 * total_passed / total:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

    if total_failed == 0:
        print("  STATUS: ALL TESTS PASSED ✓")
    else:
        print(f"  STATUS: {total_failed} TESTS FAILED ✗")

    print()
    return total_passed, total_failed


def main():
    """Run all test phases."""
    print_header()

    start_time = time.time()
    results = {}

    # Phase 1: Core Plate Tests
    results["Phase 1: Core Plate"] = run_all_phase1_tests()

    # Phase 2: Dynamics Tests
    results["Phase 2: Dynamics"] = run_all_phase2_tests()

    # Phase 3: Engine Tests
    results["Phase 3: Engine"] = run_all_phase3_tests()

    # Phase 4: Integration Tests
    results["Phase 4: Integration"] = run_all_phase4_tests()

    elapsed = time.time() - start_time

    total_passed, total_failed = print_summary(results)

    print(f"  Total time: {elapsed:.2f} seconds")
    print()

    # Return exit code
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
