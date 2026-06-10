"""Smoke tests for the constraint-pruning examples (project: bitwise equivalence).

Registers the runnable examples in ``examples/constraint_pruning_examples.py`` as
real tests: each compiles a kernel and exercises a static IR-pruning path (M1
reduction-order equivalence + the T4-A/B/C artifact filters), with no kernel-output
comparison. We import and call each example function directly; it raises on failure
and self-skips (Blackwell-only filters) where the hardware isn't available, so this
passes on any CUDA box.
"""
import importlib.util
import os

import pytest

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_CUDA = False

_EXAMPLES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "constraint_pruning_examples.py"))

_EXAMPLE_FNS = [
    "example_static_reduction_equivalence",  # M1: static TTGIR reduction-order equivalence
    "example_vectorization_filter",  # T4-A: PTX vectorization feature selection
    "example_autows_filter",  # T4-B: TTGIR AutoWS (Blackwell — self-skips elsewhere)
    "example_tmem_load_filter",  # T4-C: TTGIR tmem_load drop (Blackwell — self-skips elsewhere)
]


def _load_examples():
    spec = importlib.util.spec_from_file_location("constraint_pruning_examples", _EXAMPLES_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _HAS_CUDA, reason="constraint-pruning examples need a CUDA GPU")
@pytest.mark.parametrize("fn_name", _EXAMPLE_FNS)
def test_example_runs(fn_name):
    mod = _load_examples()
    getattr(mod, fn_name)()
