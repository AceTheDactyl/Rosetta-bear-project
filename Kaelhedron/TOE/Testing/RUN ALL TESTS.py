#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ∃κ FRAMEWORK TEST SUITE                              ║
║                                                                              ║
║                    Master Test Runner & Summary Generator                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Runs all test suites and generates comprehensive summaries.

USAGE:
    python3 RUN_ALL_TESTS.py

OUTPUT:
    - Console output with all test results
    - TEST_RESULTS_SUMMARY.md - Markdown summary
    - TEST_RESULTS_DATA.json - Machine-readable results
"""

import subprocess
import sys
import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TEST_FILES = [
    "TEST_01_SACRED_CONSTANTS.py",
    "TEST_02_FIBONACCI_STRUCTURE.py",
    "TEST_03_KAELHEDRON_GEOMETRY.py",
    "TEST_04_K_FORMATION.py",
    "TEST_05_E8_EMBEDDING.py",
    "TEST_06_FIELD_DYNAMICS.py",
]

# ═══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SuiteResult:
    name: str
    file: str
    total: int
    passed: int
    failed: int
    output: str
    success: bool

def run_test_file(filename: str) -> SuiteResult:
    """Run a single test file and capture results"""
    try:
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        
        # Parse results from output
        lines = output.split('\n')
        total, passed, failed = 0, 0, 0
        name = filename
        
        for line in lines:
            if 'TEST SUITE:' in line:
                name = line.split('TEST SUITE:')[1].strip()
            if 'Results:' in line and '/' in line:
                # Parse "Results: X/Y passed"
                parts = line.split(':')[1].strip().split('/')
                passed = int(parts[0])
                total = int(parts[1].split()[0])
                failed = total - passed
        
        success = (result.returncode == 0) and (failed == 0)
        
        return SuiteResult(
            name=name,
            file=filename,
            total=total,
            passed=passed,
            failed=failed,
            output=output,
            success=success
        )
    except subprocess.TimeoutExpired:
        return SuiteResult(
            name=filename,
            file=filename,
            total=0,
            passed=0,
            failed=1,
            output="TIMEOUT: Test took too long",
            success=False
        )
    except Exception as e:
        return SuiteResult(
            name=filename,
            file=filename,
            total=0,
            passed=0,
            failed=1,
            output=f"ERROR: {str(e)}",
            success=False
        )

def run_all_tests() -> List[SuiteResult]:
    """Run all test suites"""
    results = []
    for test_file in TEST_FILES:
        print(f"\n{'='*70}")
        print(f"Running: {test_file}")
        print('='*70)
        
        result = run_test_file(test_file)
        results.append(result)
        
        # Show output
        print(result.output)
        
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_markdown_summary(results: List[SuiteResult]) -> str:
    """Generate markdown summary of test results"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total_tests = sum(r.total for r in results)
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    all_passed = all(r.success for r in results)
    
    lines = [
        "# ∃κ Framework Test Results",
        "",
        f"**Generated:** {timestamp}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Tests | {total_tests} |",
        f"| Passed | {total_passed} |",
        f"| Failed | {total_failed} |",
        f"| Pass Rate | {100*total_passed/total_tests:.1f}% |",
        f"| Overall Status | {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'} |",
        "",
        "---",
        "",
        "## Test Suites",
        "",
    ]
    
    for r in results:
        status = "✅" if r.success else "❌"
        lines.append(f"### {status} {r.name}")
        lines.append("")
        lines.append(f"- **File:** `{r.file}`")
        lines.append(f"- **Tests:** {r.passed}/{r.total} passed")
        if r.failed > 0:
            lines.append(f"- **Failed:** {r.failed}")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Detailed Results",
        "",
    ])
    
    for r in results:
        lines.append(f"### {r.name}")
        lines.append("")
        lines.append("```")
        # Extract just the test results, not the full output
        in_results = False
        for line in r.output.split('\n'):
            if '✓ PASS' in line or '✗ FAIL' in line:
                lines.append(line)
            elif 'Results:' in line:
                lines.append(line)
        lines.append("```")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Framework Verification Status",
        "",
        "| Component | Status |",
        "|-----------|--------|",
    ])
    
    components = [
        ("Sacred Constants (φ, ζ, thresholds)", any(r.name == "SACRED CONSTANTS" and r.success for r in results)),
        ("Fibonacci Structure (F₈=21, ratios)", any(r.name == "FIBONACCI STRUCTURE" and r.success for r in results)),
        ("Kaelhedron Geometry (21 cells, 168 symmetries)", any(r.name == "KAELHEDRON GEOMETRY" and r.success for r in results)),
        ("K-Formation (R≥7, τ>φ⁻¹, Q≠0)", any(r.name == "K-FORMATION" and r.success for r in results)),
        ("E₈ Embedding (240 roots, 248 dim)", any(r.name == "E₈ EMBEDDING" and r.success for r in results)),
        ("Field Dynamics (potential, coherence)", any(r.name == "FIELD DYNAMICS" and r.success for r in results)),
    ]
    
    for comp, status in components:
        lines.append(f"| {comp} | {'✅ Verified' if status else '❌ Issues'} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Conclusion",
        "",
    ])
    
    if all_passed:
        lines.extend([
            "**All tests passed!** The ∃κ framework is mathematically verified.",
            "",
            "The following have been computationally confirmed:",
            "- All sacred constants derive from φ",
            "- Fibonacci structure underlies the framework",
            "- Kaelhedron = 21 = F₈ = dim(so(7))",
            "- K-formation conditions are properly defined",
            "- E₈ embedding chain is correct",
            "- Field dynamics are consistent",
            "",
            "**∃R → φ → Fibonacci → Fano → Kaelhedron → E₈ → Monster → j(τ) → ∃R**",
            "",
            "🔺∞🌀",
        ])
    else:
        lines.extend([
            f"**{total_failed} tests failed.** Review required.",
            "",
            "Failed suites:",
        ])
        for r in results:
            if not r.success:
                lines.append(f"- {r.name}: {r.failed} failures")
    
    return "\n".join(lines)

def generate_json_data(results: List[SuiteResult]) -> Dict[str, Any]:
    """Generate JSON data structure of test results"""
    timestamp = datetime.now().isoformat()
    
    return {
        "framework": "∃κ Framework",
        "version": "2.0",
        "timestamp": timestamp,
        "summary": {
            "total_tests": sum(r.total for r in results),
            "total_passed": sum(r.passed for r in results),
            "total_failed": sum(r.failed for r in results),
            "all_passed": all(r.success for r in results),
            "suites_count": len(results)
        },
        "suites": [
            {
                "name": r.name,
                "file": r.file,
                "total": r.total,
                "passed": r.passed,
                "failed": r.failed,
                "success": r.success
            }
            for r in results
        ]
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "∃κ FRAMEWORK TEST SUITE" + " "*25 + "║")
    print("║" + " "*15 + "Master Test Runner & Summary Generator" + " "*15 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Run all tests
    results = run_all_tests()
    
    # Calculate totals
    total_tests = sum(r.total for r in results)
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    all_passed = all(r.success for r in results)
    
    # Print grand summary
    print("\n" + "═"*70)
    print("GRAND SUMMARY")
    print("═"*70)
    print(f"  Suites Run:   {len(results)}")
    print(f"  Total Tests:  {total_tests}")
    print(f"  Passed:       {total_passed}")
    print(f"  Failed:       {total_failed}")
    print(f"  Pass Rate:    {100*total_passed/total_tests:.1f}%")
    print("═"*70)
    
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.name}: {r.passed}/{r.total}")
    
    print("═"*70)
    
    if all_passed:
        print("\n" + "🌟"*35)
        print("             ALL TESTS PASSED!")
        print("        ∃κ FRAMEWORK VERIFIED ✓")
        print("🌟"*35)
    else:
        print("\n" + "⚠️"*35)
        print(f"          {total_failed} TESTS FAILED")
        print("         REVIEW REQUIRED")
        print("⚠️"*35)
    
    # Generate summaries
    markdown = generate_markdown_summary(results)
    json_data = generate_json_data(results)
    
    # Save markdown
    with open("TEST_RESULTS_SUMMARY.md", "w") as f:
        f.write(markdown)
    print(f"\n📄 Markdown summary saved: TEST_RESULTS_SUMMARY.md")
    
    # Save JSON
    with open("TEST_RESULTS_DATA.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"📊 JSON data saved: TEST_RESULTS_DATA.json")
    
    # Return exit code
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
