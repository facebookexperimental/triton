"""Tests for the forward interpreter's control-flow sound floor (cfg + predicate). CPU-only."""
from pyptx.parser import parse

from bitequiv.ptx.affine import AffineEval, reqntid_of
from bitequiv.ptx.forward.cfg import has_unknown_control, predicated_insts
from bitequiv.ptx.forward.predicate import PredicateDecoder
from bitequiv.ptx.linker import DefUse, linearize

_HDR = ".version 8.5\n.target sm_90a\n.address_size 64\n"


def _entry(body):
    ptx = _HDR + ".visible .entry k()\n{\n" + body + "}\n"
    return [d for d in parse(ptx).directives if getattr(d, "is_entry", False)][0]


def _decode_first_pred(func):
    du = DefUse(func)
    dec = PredicateDecoder(AffineEval(du, reqntid_of(func)), du)
    for i, inst in enumerate(linearize(func)):
        pred = getattr(inst, "predicate", None)
        if pred is not None:
            return dec.decode(pred.register, i)
    return None


# tid < 64: purely thread-index derived -> safe to drop.
_STRUCT = (".reg .b32 %r<4>;\n.reg .pred %p<2>;\n.reg .f32 %f<4>;\n"
           "mov.u32 %r1, %tid.x;\nsetp.lt.s32 %p1, %r1, 64;\n@%p1 bra LBL;\n"
           "add.f32 %f1, %f2, %f3;\nLBL:\nret;\n")

# (tid & 31) == 0: warp-leader idiom. AffineEval makes the `and` opaque, but there is no DATA load in
# its provenance, so it is structural (must NOT fail closed -- warp-leader stores are everywhere).
_LANE_LEADER = (".reg .b32 %r<4>;\n.reg .pred %p<2>;\n.reg .f32 %f<4>;\n"
                "mov.u32 %r1, %tid.x;\nand.b32 %r2, %r1, 31;\nsetp.eq.s32 %p1, %r2, 0;\n"
                "@%p1 bra LBL;\nadd.f32 %f1, %f2, %f3;\nLBL:\nret;\n")

# branch on a LOADED value -> data-dependent -> must fail closed.
_DATA_DEP = (".reg .b32 %r<4>;\n.reg .b64 %rd<4>;\n.reg .pred %p<2>;\n.reg .f32 %f<4>;\n"
             "ld.global.f32 %f1, [%rd1];\nsetp.lt.f32 %p1, %f1, 0f3F000000;\n@%p1 bra LBL;\n"
             "add.f32 %f2, %f3, %f4;\nLBL:\nret;\n")


def test_structural_tid_guard_is_structural():
    assert _decode_first_pred(_entry(_STRUCT)).is_structural


def test_warp_leader_guard_is_structural():
    assert _decode_first_pred(_entry(_LANE_LEADER)).is_structural


def test_data_dependent_guard_is_not_structural():
    assert not _decode_first_pred(_entry(_DATA_DEP)).is_structural


def test_normal_bra_is_known_control():
    assert not has_unknown_control(_entry(_STRUCT))


def test_brx_is_unknown_control():
    body = ".reg .b32 %r<4>;\nmov.u32 %r1, %tid.x;\nbrx.idx %r1, tab;\nret;\n"
    assert has_unknown_control(_entry(body))


def test_predicated_insts_counts_the_guard():
    assert len(predicated_insts(_entry(_STRUCT))) == 1
