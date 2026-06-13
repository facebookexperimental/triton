"""Unit tests for the logical-relation reduction-equivalence core
(`bitequiv/reduction_tree.py`). Pure string fixtures — no GPU, no compilation.

Two parts:
  * descriptor tests — the conservative production relation
    (`reduction_descriptor` / `reductions_equivalent`), including the operand-shape
    soundness fix the old text-tuple signature missed; and
  * oracle tests — the canonical reduction tree over element indices, used as
    independent ground truth: we assert the *refinement invariant*
    (descriptor-equal => oracle-equal) and record a concrete case the conservative
    descriptor misses (oracle-equal but descriptor-different).
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from bitequiv.reduction_tree import (  # noqa: E402
    reduction_descriptor, reductions_equivalent, parse_reductions, oracle_tree_from_blocked, trees_equivalent)


def _arr(xs):
    return ", ".join(str(x) for x in xs)


def _blocked(spt, tpw, wpc, order):
    return (f"#ttg.blocked<{{sizePerThread = [{_arr(spt)}], threadsPerWarp = [{_arr(tpw)}], "
            f"warpsPerCTA = [{_arr(wpc)}], order = [{_arr(order)}]}}>")


def _reduce_body(axis, ordering, combine):
    attr = f"axis = {axis} : i32"
    if ordering is not None:
        attr += f', reduction_ordering = "{ordering}"'
    return f'''    %r = "tt.reduce"(%x) <{{{attr}}}> ({{
    ^bb0(%a: f32, %b: f32):
      %c = {combine} %a, %b : f32
      tt.reduce.return %c : f32
    }})'''


def ttgir_1d(s, spt, tpw, wpc, ordering="unordered", combine="arith.addf"):
    """1-D reduction over `tensor<s x f32>` with a contiguous (order=[0]) blocked layout."""
    blocked = _blocked([spt], [tpw], [wpc], [0])
    body = _reduce_body(0, ordering, combine)
    return f"""
#blocked = {blocked}
module {{
  tt.func @k() {{
{body} : (tensor<{s}xf32, #blocked>) -> f32
    tt.return
  }}
}}
"""


def ttgir_2d(m, n, spt, tpw, wpc, order, axis, ordering="unordered", combine="arith.addf"):
    """2-D reduction over `tensor<m x n x f32>` along `axis`."""
    blocked = _blocked(spt, tpw, wpc, order)
    body = _reduce_body(axis, ordering, combine)
    keep = m if axis == 1 else n
    return f"""
#blocked = {blocked}
module {{
  tt.func @k() {{
{body} : (tensor<{m}x{n}xf32, #blocked>) -> tensor<{keep}xf32>
    tt.return
  }}
}}
"""


_NO_REDUCTION = "module { tt.func @k() { tt.return } }"


# --------------------------------------------------------------------------- #
# descriptor: the conservative production relation
# --------------------------------------------------------------------------- #
def test_no_reduction_is_empty():
    assert reduction_descriptor(_NO_REDUCTION) == ()
    # two reduction-free modules are vacuously equivalent
    assert reductions_equivalent(_NO_REDUCTION, _NO_REDUCTION)


def test_same_layout_equivalent():
    assert reductions_equivalent(ttgir_1d(1024, 1, 32, 4), ttgir_1d(1024, 1, 32, 4))


def test_different_warps_not_equivalent():
    # warpsPerCTA on the axis differs -> different cross-warp tree -> different bits.
    assert not reductions_equivalent(ttgir_1d(1024, 1, 32, 2), ttgir_1d(1024, 1, 32, 4))


def test_different_combine_not_equivalent():
    assert not reductions_equivalent(ttgir_1d(1024, 1, 32, 4, combine="arith.addf"),
                                     ttgir_1d(1024, 1, 32, 4, combine="arith.mulf"))


def test_shape_enters_descriptor_soundness_fix():
    # SAME blocked params, DIFFERENT axis extent (e.g. a different BLOCK_SIZE) =>
    # different #register-groups => different within-thread fold => NOT equivalent.
    # The old text-tuple signature dropped the shape and wrongly merged these.
    assert not reductions_equivalent(ttgir_1d(2048, 1, 32, 4), ttgir_1d(4096, 1, 32, 4))


def test_inner_tree_layout_invariant():
    # inner_tree pins one canonical order regardless of layout: different warps -> SAME.
    assert reductions_equivalent(ttgir_1d(1024, 1, 32, 2, ordering="inner_tree"),
                                 ttgir_1d(1024, 1, 32, 8, ordering="inner_tree"))
    # ...but differs from unordered.
    assert not reductions_equivalent(ttgir_1d(1024, 1, 32, 4, ordering="inner_tree"),
                                     ttgir_1d(1024, 1, 32, 4, ordering="unordered"))


def test_inner_tree_is_shape_sensitive():
    # Even layout-invariant, a different number of leaves is a different canonical tree.
    assert not reductions_equivalent(ttgir_1d(2048, 1, 32, 4, ordering="inner_tree"),
                                     ttgir_1d(4096, 1, 32, 4, ordering="inner_tree"))


def test_absent_ordering_normalizes_to_unordered():
    assert reductions_equivalent(ttgir_1d(1024, 1, 32, 4, ordering=None), ttgir_1d(1024, 1, 32, 4,
                                                                                   ordering="unordered"))


def test_2d_axis_distinguishes():
    # Same tensor + layout, different reduce axis -> different descriptor.
    blk = dict(spt=[1, 1], tpw=[4, 8], wpc=[1, 4], order=[1, 0])
    a0 = ttgir_2d(64, 1024, axis=0, **blk)
    a1 = ttgir_2d(64, 1024, axis=1, **blk)
    assert not reductions_equivalent(a0, a1)


def test_2d_axis1_warps_distinguish():
    a = ttgir_2d(64, 1024, spt=[1, 1], tpw=[4, 8], wpc=[4, 1], order=[1, 0], axis=1)
    b = ttgir_2d(64, 1024, spt=[1, 1], tpw=[4, 8], wpc=[1, 4], order=[1, 0], axis=1)
    assert not reductions_equivalent(a, b)


def test_parse_extracts_shape_and_axis():
    (info, ) = parse_reductions(ttgir_2d(64, 1024, spt=[1, 1], tpw=[4, 8], wpc=[1, 4], order=[1, 0], axis=1))
    assert info.axis == 1 and info.dims == (64, 1024) and info.s_axis == 1024
    assert info.ordering == "unordered" and info.combine == ("arith.addf", )


# --------------------------------------------------------------------------- #
# oracle: canonical reduction tree as independent ground truth
# --------------------------------------------------------------------------- #
# (c, t, w, s) with t,w powers of two and s % (c*t*w) == 0 so the oracle is exact.
_MATRIX = [
    (1, 4, 2, 8),
    (2, 4, 1, 8),
    (1, 2, 4, 8),
    (1, 8, 1, 8),
    (1, 1, 8, 8),
    (4, 2, 1, 8),
    (2, 2, 2, 8),
    (1, 2, 2, 8),
    (1, 4, 1, 8),
    (8, 1, 1, 8),
]
_ORDERINGS = ["unordered", "inner_tree"]


def _descr_and_tree(c, t, w, s, ordering):
    g = ttgir_1d(s, c, t, w, ordering=ordering)
    (info, ) = parse_reductions(g)
    return reduction_descriptor(g), oracle_tree_from_blocked(info)


def test_refinement_invariant_descriptor_equal_implies_oracle_equal():
    cases = [(p, o) for p in _MATRIX for o in _ORDERINGS]
    built = [(_descr_and_tree(*p, ordering=o)) for (p, o) in cases]
    for (da, ta), (db, tb) in itertools.combinations(built, 2):
        if da == db:
            assert trees_equivalent(ta, tb), "descriptor merged two configs the oracle says differ (UNSOUND)"


def test_recorded_conservative_miss_lane_vs_warp():
    # t=2,w=1 (a 2-lane warp shuffle) vs t=1,w=2 (a 2-warp shared-memory combine):
    # identical bits (single add), but the conservative descriptor over-splits them.
    d_lane, tree_lane = _descr_and_tree(1, 2, 1, 2, "unordered")
    d_warp, tree_warp = _descr_and_tree(1, 1, 2, 2, "unordered")
    assert trees_equivalent(tree_lane, tree_warp)  # truly equivalent (oracle)
    assert d_lane != d_warp  # but the descriptor misses it


def test_oracle_inner_tree_layout_invariant():
    # Different layouts, same s -> identical inner_tree oracle tree.
    _, t1 = _descr_and_tree(1, 4, 2, 8, "inner_tree")
    _, t2 = _descr_and_tree(2, 2, 2, 8, "inner_tree")
    assert trees_equivalent(t1, t2)


def test_oracle_unordered_butterfly_pairs_by_xor_not_adjacency():
    # 4 lanes, count-down butterfly pairs (0,2) and (1,3), NOT adjacent (0,1)/(2,3).
    _, tree = _descr_and_tree(1, 4, 1, 4, "unordered")
    # the two depth-1 subtrees are {0,2} and {1,3}
    leaves_of = lambda node: ({node} if isinstance(node, int) else leaves_of(node[0]) | leaves_of(node[1]))
    subtree_leaves = sorted(sorted(leaves_of(child)) for child in tree)
    assert subtree_leaves == [[0, 2], [1, 3]]


# --------------------------------------------------------------------------- #
# soundness guard: a reduce-like op the parser can't analyze (multi-operand
# tt.reduce, or an MMA accumulation) must NOT collapse to an empty descriptor —
# an empty descriptor compares equal to everything (the welford/gemm FP bug).
# --------------------------------------------------------------------------- #
def _ttgir_multi_operand(s, wpc):
    blocked = _blocked([1], [32], [wpc], [0])
    return f"""
#blocked = {blocked}
module {{
  tt.func @k() {{
    %v, %i = "tt.reduce"(%val, %idx) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32, %b: i32, %c: f32, %d: i32):
      %lt = arith.cmpf olt, %a, %c : f32
      %rv = arith.select %lt, %a, %c : f32
      %ri = arith.select %lt, %b, %d : i32
      tt.reduce.return %rv, %ri : f32, i32
    }}) : (tensor<{s}xf32, #blocked>, tensor<{s}xi32, #blocked>) -> (f32, i32)
    tt.return
  }}
}}
"""


def _ttgir_mma(prec, wpc):
    return f"""
#mma = #ttg.nvidia_mma<{{versionMajor = 3, warpsPerCTA = [{wpc}, 1]}}>
module {{
  tt.func @k() {{
    %c = tt.dot %a, %b, %acc, inputPrecision = {prec} : tensor<128x128xf32, #mma>
    tt.return
  }}
}}
"""


def test_multi_operand_reduce_not_empty_descriptor():
    # a multi-operand reduce must NOT yield () (which would equal everything).
    d = reduction_descriptor(_ttgir_multi_operand(8192, 4))
    assert d != () and any(e[0] == "unanalyzed" for e in d)


def test_multi_operand_reduce_is_sound_across_layouts():
    # different warpsPerCTA -> NOT declared equivalent (the welford soundness fix)...
    assert not reductions_equivalent(_ttgir_multi_operand(8192, 2), _ttgir_multi_operand(8192, 4))
    # ...but identical IR is still reflexively equivalent.
    assert reductions_equivalent(_ttgir_multi_operand(8192, 4), _ttgir_multi_operand(8192, 4))


def test_mma_accumulation_is_sound():
    d = reduction_descriptor(_ttgir_mma("tf32", 4))
    assert d != () and any(e[0] == "unanalyzed" for e in d)
    # different precision -> NOT equivalent (the gemm precision case)...
    assert not reductions_equivalent(_ttgir_mma("ieee", 4), _ttgir_mma("tf32", 4))
    # ...identical -> equivalent.
    assert reductions_equivalent(_ttgir_mma("tf32", 4), _ttgir_mma("tf32", 4))


def test_no_reduction_still_empty_and_vacuously_equivalent():
    # a kernel with NO reduction-like op at all still yields () and is equivalent.
    assert reduction_descriptor(_NO_REDUCTION) == ()
    assert reductions_equivalent(_NO_REDUCTION, _NO_REDUCTION)
