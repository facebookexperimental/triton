"""The layout-equivalence gap the redesign closes, and the refinement guarantee.

The *retired* PTX signature was the ordered ``shfl.sync.bfly`` offsets + the fp-opcode
multiset + a fused flag. It is **blind to layout**: two configs with identical offsets and
identical opcode multiset but a different element->thread map collapse to the same key.
The new tree-based descriptor labels each leaf with its recovered element coordinate, so it
splits them. These tests prove:

  1. **gap-closing** — a constructed pair (same offsets, same opcode multiset, different
     load layout) that the OLD signature merges but the NEW one splits;
  2. **real-bits layout split** — the committed ``rowsum2d`` pair (a real 2-D layout
     difference with documented different bits) is split by the new checker;
  3. **refinement** — on every committed fixture pair, NEW-equal implies OLD-equal: the new
     relation only ever sub-divides the old classes (it never merges what the old one split).
     This is a *relative* property between the two checkers; neither is absolutely sound (see
     bitequiv/ptx_reduction.py for the conditional guarantee and its residuals).
"""
import os
from collections import Counter

from pyptx.ir.nodes import Block, Function, Instruction
from pyptx.parser import parse

from bitequiv.ptx_reduction import ptx_reduction_descriptor as NEW

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "ptx")

_FP_OPCODES = frozenset({"fma", "add", "sub", "mul", "min", "max", "div"})
_FP_WIDTHS = frozenset({".f16", ".f16x2", ".f32", ".f64", ".bf16", ".bf16x2"})
_HEADER = ".version 8.5\n.target sm_90a\n.address_size 64\n"


def _insts(stmts):
    for s in stmts:
        if isinstance(s, Block):
            yield from _insts(s.body)
        elif isinstance(s, Instruction):
            yield s


def OLD(ptx):
    """The retired offsets + fp-opcode-multiset + fused signature (layout-blind)."""
    module = parse(ptx if ".version" in ptx else _HEADER + ptx)
    out = []
    for f in (d for d in module.directives if isinstance(d, Function) and d.is_entry):
        offs, ops = [], []
        for i in _insts(f.body):
            if i.opcode == "shfl" and ".bfly" in i.modifiers and len(i.operands) > 2:
                t = getattr(i.operands[2], "text", None) or getattr(i.operands[2], "name", "?")
                offs.append(int(t) if t.lstrip("-").isdigit() else t)
            elif i.opcode in _FP_OPCODES and i.modifiers and i.modifiers[-1] in _FP_WIDTHS:
                ops.append(i.opcode + "".join(i.modifiers))
        out.append((tuple(offs), tuple(sorted(Counter(ops).items())), any(o.startswith("fma") for o in ops)))
    return tuple(sorted(map(repr, out)))


def _fixture(name):
    with open(os.path.join(_FIX, f"{name}.ptx")) as f:
        return f.read()


def _layout(stride):
    """A complete minimal reduction whose single leaf is element ``stride*tid``: a
    within-warp butterfly (offsets 16,8,4,2,1) over one loaded value, then a store. Two
    different strides are two different thread<->data maps with identical offsets and an
    identical add count."""
    steps = "\n".join(f"  shfl.sync.bfly.b32 %rs{k}, %r{3 + k}, {off}, 31, -1;\n"
                      f"  add.f32 %r{4 + k}, %r{3 + k}, %rs{k};" for k, off in enumerate((16, 8, 4, 2, 1)))
    return (f".visible .entry k(.param .u64 .ptr .global k_param_0, .param .u64 .ptr .global k_param_1)\n"
            f".reqntid 32\n{{\n.reg .b32 %r<16>;\n.reg .b32 %rs<8>;\n.reg .b64 %rd<5>;\n"
            f"  ld.param.b64 %rd1, [k_param_0];\n  ld.param.b64 %rd2, [k_param_1];\n"
            f"  mov.u32 %r1, %tid.x;\n  mul.wide.u32 %rd3, %r1, {stride};\n  add.s64 %rd1, %rd1, %rd3;\n"
            f"  ld.global.b32 %r3, [%rd1];\n{steps}\n  st.global.b32 [%rd2], %r8;\n  ret;\n}}\n")


# 1. gap-closing -------------------------------------------------------------
def test_old_signature_merges_what_new_splits():
    a, b = _layout(4), _layout(8)  # same offsets + same op multiset, different layout
    assert OLD(a) == OLD(b)  # the retired checker is blind to the layout -> merges
    assert NEW(a) != NEW(b)  # the new tree checker labels leaves -> splits


def test_same_layout_still_matches():
    assert NEW(_layout(4)) == NEW(_layout(4))  # determinism: identical layout -> identical key


# 2. real-bits layout split --------------------------------------------------
def test_rowsum2d_layout_difference_is_split():
    # rowsum2d_withinwarp (warpsPerCTA=[4,1]) vs crosswarp ([4,2]) reduce the same [4,128]
    # tile with a different 2-D thread<->data map; the README records different real bits
    # (row 0: -425269.0625 vs -425269.0). The new checker must split them.
    assert NEW(_fixture("rowsum2d_withinwarp")) != NEW(_fixture("rowsum2d_crosswarp"))


# 3. refinement (soundness-monotone) -----------------------------------------
_FIXTURES = [
    "sum_nw1", "sum_nw2", "sum_nw4", "sum_nw8", "dot_fuse_on", "dot_fuse_off", "rowsum2d_withinwarp",
    "rowsum2d_crosswarp"
]


def test_new_relation_refines_old_on_all_fixtures():
    # For every pair: NEW-equal => OLD-equal. The new classes are sub-divisions of the old
    # ones, so the new checker never over-merges *relative to* the old one. (A relative
    # property; neither checker is absolutely sound -- see bitequiv/ptx_reduction.py.)
    descs = {n: (OLD(_fixture(n)), NEW(_fixture(n))) for n in _FIXTURES}
    for a in _FIXTURES:
        for b in _FIXTURES:
            old_a, new_a = descs[a]
            old_b, new_b = descs[b]
            if new_a == new_b:
                assert old_a == old_b, f"{a} vs {b}: new merged but old split (not a refinement)"
