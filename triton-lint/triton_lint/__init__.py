"""
Triton-Lint - Static analyzer for Triton kernel performance and correctness.

This package provides multi-level analysis of Triton kernels to detect
performance anti-patterns and correctness issues.
"""

__version__ = "0.1.0"

from triton_lint.core.finding import Finding, Severity
from triton_lint.core.config import Config

__all__ = ["Finding", "Severity", "Config", "__version__"]
