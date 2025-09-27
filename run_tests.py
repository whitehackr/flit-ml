#!/usr/bin/env python3
"""
Test runner for FLIT ML project with proper test organization
"""

import subprocess
import sys
import os


def run_tests():
    """Run all tests with proper categorization."""

    print("🧪 Running FLIT ML Test Suite")
    print("=" * 50)

    success = True

    # Unit Tests
    print("\n📦 UNIT TESTS")
    print("-" * 30)

    unit_tests = [
        ("Feature Engineering", "tests/unit/features/test_feature_engineering.py"),
        ("Unknown Categories", "tests/unit/features/test_unknown_categories.py"),
        ("Multi-Model Predictor", "tests/unit/models/test_multi_model_predictor.py")
    ]

    for test_name, test_file in unit_tests:
        print(f"\n🔹 {test_name}")
        try:
            result = subprocess.run([sys.executable, test_file],
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                print(f"✅ PASS")
            else:
                print(f"❌ FAIL")
                print(f"Error: {result.stderr}")
                success = False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            success = False

    # Integration Tests
    print(f"\n🔗 INTEGRATION TESTS")
    print("-" * 30)

    integration_tests = [
        ("Production Artifacts", "poetry run pytest tests/integration/test_production_artifacts.py -v"),
        ("API Endpoints", "tests/integration/test_api_endpoints.py")
    ]

    for test_name, test_command in integration_tests:
        print(f"\n🔹 {test_name}")
        try:
            if test_command.startswith("poetry"):
                result = subprocess.run(test_command.split(),
                                      capture_output=True, text=True, cwd=os.getcwd())
            else:
                result = subprocess.run([sys.executable, test_command],
                                      capture_output=True, text=True, cwd=os.getcwd())

            if result.returncode == 0:
                print(f"✅ PASS")
            else:
                print(f"❌ FAIL")
                print(f"Error: {result.stderr}")
                success = False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            success = False

    # Summary
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())