"""Tests for the reduction-tree IR canonicalization and the backward builder.

Canonicalization rules are tested directly on the IR nodes; the builder is tested on the
committed real-Triton PTX fixtures (parsed CPU-only, no GPU)."""
import os

from pyptx.parser import parse

from bitequiv.ptx.builder import build_trees, entry_signatures
from bitequiv.ptx.treeir import FpOp, Leaf, ShflCombine, SmemExchange

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "ptx")


def _entry(name):
    m = parse(open(os.path.join(_FIX, f"{name}.ptx")).read())
    return [d for d in m.directives if getattr(d, "is_entry", False)][0]


def _walk(t, acc):
    acc[type(t).__name__] = acc.get(type(t).__name__, 0) + 1
    if isinstance(t, FpOp):
        for c in t.children:
            _walk(c, acc)
    elif isinstance(t, (ShflCombine, SmemExchange)):
        _walk(t.child, acc)
    return acc


def _counts(name):
    acc = {}
    for t in build_trees(_entry(name)):
        _walk(t, acc)
    return acc


# -- canonicalization rules (direct on IR) ------------------------------------


def test_commutative_add_sorts_children():
    x, y = Leaf("x"), Leaf("y")
    assert FpOp("add", (".f32", ), (x, y)).sig() == FpOp("add", (".f32", ), (y, x)).sig()


def test_mul_is_commutative_but_sub_is_not():
    x, y = Leaf("x"), Leaf("y")
    assert FpOp("mul", (".f32", ), (x, y)).sig() == FpOp("mul", (".f32", ), (y, x)).sig()
    assert FpOp("sub", (".f32", ), (x, y)).sig() != FpOp("sub", (".f32", ), (y, x)).sig()


def test_fma_sorts_multiply_pair_but_addend_is_positional():
    a, b, c = Leaf("a"), Leaf("b"), Leaf("c")
    fab = FpOp("fma", (".rn", ".f32"), (a, b, c), fused=True)
    fba = FpOp("fma", (".rn", ".f32"), (b, a, c), fused=True)
    fca = FpOp("fma", (".rn", ".f32"), (c, a, b), fused=True)
    assert fab.sig() == fba.sig()  # a*b+c == b*a+c
    assert fab.sig() != fca.sig()  # a*b+c != c*a+b


def test_modifiers_are_signature_significant():
    x, y = Leaf("x"), Leaf("y")
    assert FpOp("add", (".rn", ".f32"), (x, y)).sig() != FpOp("add", (".f32", ), (x, y)).sig()
    assert FpOp("add", (".ftz", ".f32"), (x, y)).sig() != FpOp("add", (".f32", ), (x, y)).sig()


def test_shfl_offset_sequence_is_positional():
    leaf = Leaf("p")
    a = ShflCombine(16, "add", (".f32", ), ShflCombine(8, "add", (".f32", ), leaf))
    b = ShflCombine(8, "add", (".f32", ), ShflCombine(16, "add", (".f32", ), leaf))
    assert a.sig() != b.sig()


# -- builder on real fixtures -------------------------------------------------


def test_sum_fixtures_have_no_opaque_on_main_path():
    for name in ("sum_nw1", "sum_nw2", "sum_nw4", "sum_nw8", "dot_fuse_on", "dot_fuse_off"):
        assert "OpaqueLeaf" not in _counts(name), name


def test_sum_leaf_count_halves_as_warps_double():
    # Fixed total work split over more warps -> fewer elements (leaves) per thread.
    assert _counts("sum_nw1")["Leaf"] == 128
    assert _counts("sum_nw2")["Leaf"] == 64
    assert _counts("sum_nw4")["Leaf"] == 32
    assert _counts("sum_nw8")["Leaf"] == 16


def test_cross_warp_shuffle_steps_grow_with_log2_num_warps():
    # within-warp = 5 shuffles (32 lanes); cross-warp adds log2(num_warps) more.
    assert _counts("sum_nw1")["ShflCombine"] == 5  # 5 + 0
    assert _counts("sum_nw2")["ShflCombine"] == 6  # 5 + 1
    assert _counts("sum_nw4")["ShflCombine"] == 7  # 5 + 2
    assert _counts("sum_nw8")["ShflCombine"] == 8  # 5 + 3
    # single-warp reduction needs no shared-memory exchange.
    assert "SmemExchange" not in _counts("sum_nw1")
    assert _counts("sum_nw8")["SmemExchange"] == 2


def test_each_num_warps_is_a_distinct_signature():
    sigs = {entry_signatures(_entry(n)) for n in ("sum_nw1", "sum_nw2", "sum_nw4", "sum_nw8")}
    assert len(sigs) == 4


def _all_nodes(name):
    out, seen = [], set()
    for root in build_trees(_entry(name)):
        stack = [root]
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            out.append(n)
            stack.extend(n.children)
    return out


def test_dot_fusion_is_visible_at_ptx():
    # fusion on compiles to fma nodes; off stays mul+add -> the reconstructed trees differ.
    on, off = entry_signatures(_entry("dot_fuse_on")), entry_signatures(_entry("dot_fuse_off"))
    assert on != off
    on_has_fma = any(isinstance(n, FpOp) and n.fused for n in _all_nodes("dot_fuse_on"))
    off_has_fma = any(isinstance(n, FpOp) and n.fused for n in _all_nodes("dot_fuse_off"))
    assert on_has_fma and not off_has_fma


def test_dot_leaves_label_both_input_arrays():
    # the dot reads two operands; leaf coords must distinguish param_0 from param_1.
    coords = "".join(n.coord for n in _all_nodes("dot_fuse_on") if isinstance(n, Leaf))
    assert "param:dot_kernel_param_0" in coords and "param:dot_kernel_param_1" in coords
