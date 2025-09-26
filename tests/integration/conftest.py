"""
Pytest configuration for integration tests.
"""

import pytest


def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption(
        "--model-version", action="store", default="v0.1.0",
        help="Model version to test (default: v0.1.0)"
    )
    parser.addoption(
        "--model-domain", action="store", default="bnpl",
        help="Model domain to test (default: bnpl)"
    )