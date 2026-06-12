"""Smoke test: each example runs and demonstrates divergence (>1 bit-class).

Each example's ``main()`` self-asserts ``len(classes) > 1``, so a clean run IS
the assertion. This test imports the ``main()`` functions only; it never imports
or runs any performance benchmark (project rule: never run perf tests unless
explicitly asked).

Run:  pytest bitequiv/examples/numerical-inconsistency/test_smoke.py
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from _helpers import is_cuda  # noqa: E402

EXAMPLES = [
    "n01_autotuner_picks_silently",
    "n02_sum_reduction_classes",
    "n03_softmax_row_reduction",
    "n04_layernorm_two_reductions",
    "n05_dot_reduce",
]


@pytest.mark.skipif(not is_cuda(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("mod", EXAMPLES)
def test_example_shows_divergence(mod):
    # main() raises AssertionError if it fails to find >1 bit-class.
    importlib.import_module(mod).main()
