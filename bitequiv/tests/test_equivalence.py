"""Unit tests for the static TTGIR reduction-order equivalence checker
(`bitequiv/equivalence.py`). Pure string fixtures — no GPU, no compilation."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from bitequiv.equivalence import (  # noqa: E402
    reduction_signature, same_reduction_order, classify, reduction_equivalence_key, ptx_reduction_signature, CHECKERS)


def _ttgir(warps_on_axis, ordering="unordered", combine="arith.addf"):
    """Minimal TTGIR with one tt.reduce over axis 1, varying warpsPerCTA[axis]."""
    return f"""
#blocked = #ttg.blocked<{{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, {warps_on_axis}], order = [1, 0]}}>
module {{
  tt.func @k() {{
    %s = "tt.reduce"(%x) <{{axis = 1 : i32, reduction_ordering = "{ordering}"}}> ({{
    ^bb0(%a: f32, %b: f32):
      %r = {combine} %a, %b : f32
      tt.reduce.return %r : f32
    }}) : (tensor<64x1024xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
    tt.return
  }}
}}
"""


def test_signature_nonempty_for_reduction():
    assert reduction_signature(_ttgir(4)) != ()


def test_different_warps_give_different_signature():
    # warpsPerCTA[axis] differs -> different cross-warp tree -> not equivalent.
    assert reduction_signature(_ttgir(2)) != reduction_signature(_ttgir(4))
    assert not same_reduction_order(_ttgir(2), _ttgir(4))


def test_same_layout_is_equivalent():
    assert same_reduction_order(_ttgir(4), _ttgir(4, combine="arith.addf"))


def test_different_combine_op_not_equivalent():
    # add vs mul: same order but different operator -> different bits.
    assert not same_reduction_order(_ttgir(4, combine="arith.addf"), _ttgir(4, combine="arith.mulf"))


def test_inner_tree_is_layout_invariant():
    # inner_tree enforces one canonical order regardless of layout: different warpsPerCTA -> SAME sig.
    assert same_reduction_order(_ttgir(2, ordering="inner_tree"), _ttgir(8, ordering="inner_tree"))
    # ...and inner_tree differs from unordered.
    assert not same_reduction_order(_ttgir(4, ordering="inner_tree"), _ttgir(4, ordering="unordered"))


def test_classify_groups_by_order():
    classes = classify({"nw1": _ttgir(1), "nw2": _ttgir(2), "nw4a": _ttgir(4), "nw4b": _ttgir(4)})
    # nw1, nw2 each alone; nw4a+nw4b together.
    sizes = sorted(len(v) for v in classes.values())
    assert sizes == [1, 1, 2]
    assert len(classes) == 3


def test_no_reduction_yields_empty_signature():
    assert reduction_signature("module { tt.func @k() { tt.return } }") == ()


def test_equivalence_key_adapter_reads_ttgir():
    key_a = reduction_equivalence_key(None, {"ttgir": _ttgir(4)}, None)
    key_b = reduction_equivalence_key(None, {"ttgir": _ttgir(8)}, None)
    assert key_a == reduction_signature(_ttgir(4))
    assert key_a != key_b


# --- level registry (for the autotuner's equivalence_level option) ---
def test_registry_has_ttgir_and_ptx():
    assert set(CHECKERS) == {"ttgir", "ptx"}


def test_registry_ttgir_checker_matches_signature():
    key = CHECKERS["ttgir"](None, {"ttgir": _ttgir(4)}, None)
    assert key == reduction_signature(_ttgir(4))


def test_ptx_checker_is_a_stub_that_raises():
    # PTX-level equivalence is not implemented yet; the registry entry must signal that clearly.
    with pytest.raises(NotImplementedError):
        ptx_reduction_signature("// some ptx")
    with pytest.raises(NotImplementedError):
        CHECKERS["ptx"](None, {"ptx": "// some ptx"}, None)
