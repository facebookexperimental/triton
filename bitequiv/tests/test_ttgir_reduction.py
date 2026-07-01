"""Unit tests for the MLIR-native reduction-order checker (`bitequiv.ttgir_reduction`).

These parse REAL committed TTGIR fixtures (`bitequiv/tests/ttgir/*.ttgir`, produced
by `gen_ttgir_fixtures.py`). Parsing is CPU-only but requires the built triton (the
checker calls into `libtriton.bitequiv`); run with the m1 worktree on PYTHONPATH.
We assert *behavior* (the equivalence relation), not the opaque signature internals.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from bitequiv.ttgir_reduction import ttgir_reduction_descriptor, ttgir_reductions_equivalent  # noqa: E402

_TTGIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttgir")


def _load(name):
    with open(os.path.join(_TTGIR, name + ".ttgir")) as f:
        return f.read()


def _eq(a, b):
    return ttgir_reductions_equivalent(_load(a), _load(b))


# --- single-operand reductions (the in-scope core) ---
def test_unordered_different_warps_not_equivalent():
    assert not _eq("sum_uno_nw2", "sum_uno_nw4")
    assert not _eq("sum_uno_nw4", "sum_uno_nw8")


def test_same_config_equivalent():
    assert _eq("sum_uno_nw4", "sum_uno_nw4b")


def test_inner_tree_layout_invariant():
    # inner_tree collapses across num_warps (layout-invariant order)...
    assert _eq("sum_inner_nw2", "sum_inner_nw8")
    # ...and differs from unordered.
    assert not _eq("sum_inner_nw2", "sum_uno_nw2")


def test_shape_enters_signature():
    # same blocked params + num_warps, different BLOCK (axis extent) -> not equivalent.
    assert not _eq("sum_uno_block2048_nw4", "sum_uno_block4096_nw4")


def test_different_combine_not_equivalent():
    assert not _eq("mul_uno_nw4", "sum_uno_nw4")


def test_descriptor_one_per_reduce_and_empty_for_none():
    assert len(ttgir_reduction_descriptor(_load("sum_uno_nw4"))) == 1  # one tt.reduce
    assert ttgir_reduction_descriptor("") == ()  # no reduction-like op


# --- boundary cases: multi-operand + MMA must be SOUND (never a false merge) ---
def test_multi_operand_argmin_sound():
    # a multi-operand reduce is analyzed (non-empty descriptor) and sound across layouts.
    assert ttgir_reduction_descriptor(_load("argmin_nw2")) != ()
    assert not _eq("argmin_nw2", "argmin_nw4")  # different layout -> not merged
    assert _eq("argmin_nw2", "argmin_nw2")  # reflexive


def test_gemm_mma_sound():
    # an MMA accumulation has no tt.reduce; the guard keeps the descriptor non-empty
    # and never falsely equates precision modes.
    assert ttgir_reduction_descriptor(_load("gemm_ieee")) != ()
    assert not _eq("gemm_ieee", "gemm_tf32")  # precision differs -> not merged
    assert _eq("gemm_ieee", "gemm_ieee")  # reflexive
