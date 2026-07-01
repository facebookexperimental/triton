"""Hermetic unit tests for the conservative PTX affine evaluator (no GPU)."""
from pyptx.parser import parse

from bitequiv.ptx.affine import Affine, AffineEval, Opaque, canon, reqntid_of
from bitequiv.ptx.linker import DefUse

_HEADER = ".version 8.5\n.target sm_90a\n.address_size 64\n"


def _eval(body, reg, reqntid=True):
    src = (f"{_HEADER}.visible .entry k(.param .u64 .ptr .global k_param_0, .param .u32 k_param_1)\n"
           f"{'.reqntid 128' if reqntid else ''}\n{{\n.reg .b32 %r<40>;\n.reg .b64 %rd<20>;\n{body}\nret;\n}}\n")
    m = parse(src)
    f = [d for d in m.directives if getattr(d, "is_entry", False)][0]
    du = DefUse(f)
    ev = AffineEval(du, reqntid_of(f))
    return ev.of_reg(reg, len(du.insts))


def test_shl_is_multiply_by_pow2():
    v = _eval("mov.u32 %r1, %tid.x;\nshl.b32 %r2, %r1, 2;", "%r2")
    assert isinstance(v, Affine) and canon(v) == "4*%tid.x"


def test_and_lowbit_mask_is_noop_within_range():
    # 4*tid with tid<128 spans bits [2,8]; mask 508 covers exactly those -> no-op.
    v = _eval("mov.u32 %r1, %tid.x;\nshl.b32 %r2, %r1, 2;\nand.b32 %r3, %r2, 508;", "%r3")
    assert canon(v) == "4*%tid.x"


def test_and_partial_mask_is_opaque():
    # mask 12 (bits 2,3) does NOT cover all bits 4*tid can set -> not provably no-op.
    v = _eval("mov.u32 %r1, %tid.x;\nshl.b32 %r2, %r1, 2;\nand.b32 %r3, %r2, 12;", "%r3")
    assert isinstance(v, Opaque)


def test_disjoint_or_is_add():
    # 3584 = bits 9,10,11; disjoint from 4*tid's bits [2,8] -> or == add.
    v = _eval("mov.u32 %r1, %tid.x;\nshl.b32 %r2, %r1, 2;\nor.b32 %r3, %r2, 3584;", "%r3")
    assert canon(v) == "3584+4*%tid.x"


def test_overlapping_or_is_opaque():
    # bit 2 overlaps 4*tid's possible bits -> cannot prove or==add.
    v = _eval("mov.u32 %r1, %tid.x;\nshl.b32 %r2, %r1, 2;\nor.b32 %r3, %r2, 4;", "%r3")
    assert isinstance(v, Opaque)


def test_mad_wide_is_affine_times_const_plus_base():
    v = _eval(
        "ld.param.b64 %rd1, [k_param_0];\nmov.u32 %r1, %tid.x;\nshl.b32 %r2, %r1, 2;\nmad.wide.u32 %rd2, %r2, 4, %rd1;",
        "%rd2")
    assert canon(v) == "16*%tid.x+1*param:k_param_0"


def test_shr_exact_when_multiple_else_opaque():
    exact = _eval("mov.u32 %r1, %tid.x;\nshl.b32 %r2, %r1, 2;\nshr.u32 %r3, %r2, 1;", "%r3")
    assert canon(exact) == "2*%tid.x"
    inexact = _eval("mov.u32 %r1, %tid.x;\nshr.u32 %r2, %r1, 5;", "%r2")  # tid has tz 0
    assert isinstance(inexact, Opaque)


def test_mul_reg_reg_is_opaque_and_structural():
    a = _eval("mov.u32 %r1, %tid.x;\nmul.lo.s32 %r2, %r1, %r1;", "%r2")
    b = _eval("mov.u32 %r9, %tid.x;\nmul.lo.s32 %r8, %r9, %r9;", "%r8")  # different reg names
    assert isinstance(a, Opaque)
    assert a.token == b.token  # structural token is register-name independent


def test_ld_param_is_base_symbol():
    v = _eval("ld.param.b64 %rd1, [k_param_0];", "%rd1")
    assert canon(v) == "1*param:k_param_0"


def test_and_mask_opaque_without_range():
    # Same arithmetic but no .reqntid -> tid range unknown -> cannot prove no-op.
    v = _eval("mov.u32 %r1, %tid.x;\nshl.b32 %r2, %r1, 2;\nand.b32 %r3, %r2, 508;", "%r3", reqntid=False)
    assert isinstance(v, Opaque)


def test_affine_value_identity_ignores_range():
    x = Affine(4, frozenset({("%tid.x", 4)}), rng=(0, 512))
    y = Affine(4, frozenset({("%tid.x", 4)}), rng=None)
    assert x == y and hash(x) == hash(y)
