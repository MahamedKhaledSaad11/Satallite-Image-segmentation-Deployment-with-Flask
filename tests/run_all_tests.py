"""
Master Test Runner for AquaSeg Multispectral Water Segmentation.
Executes all test tiers (Unit, Integration, and System / End-to-End API)
and outputs a comprehensive verification report.
"""

import sys
import time
import unittest
from pathlib import Path

# Add project root and deployment to sys.path
deployment_dir = Path(__file__).resolve().parent.parent
project_root = deployment_dir.parent
for p in [str(deployment_dir), str(project_root), str(project_root / "water_segmentation"), str(project_root / "water_segmentation_part2")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from deployment.tests import test_unit
from deployment.tests import test_integration
from deployment.tests import test_system


def run_suite(name: str, module) -> unittest.TestResult:
    print(f"\n{'='*70}")
    print(f"  RUNNING TEST SUITE: {name}")
    print(f"{'='*70}")

    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    return result, elapsed


def main():
    print("\n" + "#" * 70)
    print("   AQUASEG COMPREHENSIVE TEST SUITE — TASK 4 VERIFICATION")
    print("   Unit Tests • Integration Tests • System & API Tests")
    print("#" * 70)

    suites = [
        ("Unit Tests (Formulas, Preprocessing, Contracts)", test_unit),
        ("Integration Tests (Models, Benchmarks, Edge Cases)", test_integration),
        ("System Tests (Flask REST API, E2E Routes, Error Handling)", test_system),
    ]

    summary = []
    total_start = time.perf_counter()

    all_passed = True
    for name, mod in suites:
        res, elapsed = run_suite(name, mod)
        passed = res.wasSuccessful()
        if not passed:
            all_passed = False
        summary.append({
            "name": name,
            "tests_run": res.testsRun,
            "failures": len(res.failures),
            "errors": len(res.errors),
            "time": elapsed,
            "passed": passed
        })

    total_elapsed = time.perf_counter() - total_start

    print("\n\n" + "=" * 70)
    print("                 TEST EXECUTION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Suite Name':<45} | {'Tests':<6} | {'Failures':<8} | {'Status':<8}")
    print("-" * 70)

    total_tests = 0
    total_failures = 0
    total_errors = 0

    for s in summary:
        status_str = "PASS [OK]" if s["passed"] else "FAIL [X]"
        print(f"{s['name']:<45} | {s['tests_run']:<6} | {s['failures'] + s['errors']:<8} | {status_str:<8}")
        total_tests += s["tests_run"]
        total_failures += s["failures"]
        total_errors += s["errors"]

    print("-" * 70)
    print(f"{'TOTAL':<45} | {total_tests:<6} | {total_failures + total_errors:<8} | {'ALL PASSED' if all_passed else 'FAILED'}")
    print(f"Total Execution Time: {total_elapsed:.2f} seconds\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
