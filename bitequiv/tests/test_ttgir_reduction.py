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


class _Cfg:
    """Minimal stand-in for an autotuner config exposing just ``enable_fp_fusion``."""

    def __init__(self, enable_fp_fusion):
        self.enable_fp_fusion = enable_fp_fusion


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


# --- enable_fp_fusion (invisible in TTGIR) folded in from the config (diff 2a) ---
def test_fp_fusion_splits_fma_eligible():
    # GEMM (warp_group_dot) is fma-eligible: fma contraction is decided below TTGIR, so
    # fp_fusion on vs off compile to the SAME IR but different bits. With the config
    # supplied they must NOT merge; with the same config they must.
    ir = _load("gemm_ieee")
    assert ttgir_reduction_descriptor(ir, _Cfg(True)) != ttgir_reduction_descriptor(ir, _Cfg(False))
    assert ttgir_reduction_descriptor(ir, _Cfg(True)) == ttgir_reduction_descriptor(ir, _Cfg(True))


def test_fp_fusion_not_split_for_pure_add():
    # a pure-add reduction has no multiply to contract, so fp_fusion cannot change its
    # bits -- on and off must stay equal (no over-split).
    ir = _load("sum_uno_nw4")
    assert ttgir_reduction_descriptor(ir, _Cfg(True)) == ttgir_reduction_descriptor(ir, _Cfg(False))


def test_config_none_is_backward_compatible():
    # no config -> exactly the pre-2a descriptor (the PTX-checker / autotuner path).
    ir = _load("gemm_ieee")
    assert ttgir_reduction_descriptor(ir, None) == ttgir_reduction_descriptor(ir)


# --- diff 2b: real warp_group_dot signature (tiling collapses; tf32x3 splits by dot count) ---
def test_gemm_tiling_and_warps_collapse():
    # M/N tiling and num_warps are FREE for a wgmma GEMM: a different tile (BM 64 vs 128) and a
    # different num_warps, same precision, must land in ONE class -- the signature drops the
    # operand/result shapes and encodings.
    assert _eq("gemm_tf32_bm64", "gemm_tf32_nw8")


def test_gemm_tf32x3_splits_by_dot_count():
    # tf32x3 lowers to 3 chained wgmma passes (a genuinely different bit order); after that
    # decomposition all three print inputPrecision=tf32, so the dot COUNT is the only signal
    # that separates it from a single tf32 dot.
    assert not _eq("gemm_tf32x3", "gemm_tf32_nw8")
