"""Hermetic unit tests for the PTX def-use linker (no GPU, hand-written PTX)."""

from pyptx.ir.nodes import AddressOperand, Instruction, RegisterOperand
from pyptx.parser import parse

from bitequiv.ptx.linker import DefUse, _def_regs, linearize

_HEADER = ".version 8.5\n.target sm_90a\n.address_size 64\n"


def _entry(body, params="(.param .u64 .ptr .global k_param_0, .param .u32 k_param_1)"):
    src = f"{_HEADER}.visible .entry k{params}\n.reqntid 128\n{{\n.reg .b32 %r<40>;\n.reg .b64 %rd<20>;\n.reg .pred %p<8>;\n{body}\nret;\n}}\n"
    m = parse(src)
    return [d for d in m.directives if getattr(d, "is_entry", False)][0]


def test_linearize_recurses_blocks_in_order():
    f = _entry("mov.u32 %r1, %tid.x;\n{\nadd.s32 %r2, %r1, 1;\n}\nshl.b32 %r3, %r2, 2;")
    ops = [i.opcode for i in linearize(f)]
    assert ops == ["mov", "add", "shl", "ret"]


def test_last_writer_picks_most_recent_prior_def():
    # %r1 is written twice; a use after the 2nd write must bind to the 2nd.
    f = _entry("mov.u32 %r1, %tid.x;\nadd.s32 %r2, %r1, 1;\nmov.u32 %r1, %ctaid.x;\nadd.s32 %r3, %r1, 1;")
    du = DefUse(f)
    add3 = next(i for i in du.insts if i.opcode == "add" and i.operands[0].name == "%r3")
    d = du.last_writer("%r1", du.index_of(add3))
    assert d.inst.operands[1].name == "%ctaid.x"  # the 2nd def of %r1
    add2 = next(i for i in du.insts if i.opcode == "add" and i.operands[0].name == "%r2")
    d2 = du.last_writer("%r1", du.index_of(add2))
    assert d2.inst.operands[1].name == "%tid.x"  # the 1st def, for the earlier use


def test_vector_dest_registers_get_slots():
    f = _entry("ld.global.v4.b32 {%r1, %r2, %r3, %r4}, [%rd1];")
    du = DefUse(f)
    slots = {r: du.defs_by_reg[r][0].slot for r in ("%r1", "%r2", "%r3", "%r4")}
    assert slots == {"%r1": 0, "%r2": 1, "%r3": 2, "%r4": 3}


def test_store_defines_no_register():
    f = _entry("st.global.b32 [%rd1], %r1;")
    du = DefUse(f)
    # The store's only register operand (%r1) is a *source*, not a def.
    assert "%rd1" not in du.defs_by_reg and "%r1" not in du.defs_by_reg


def test_masked_load_idiom_last_writer_is_the_predicated_load():
    f = _entry(
        "mov.b32 %r5, 0;\nmov.u32 %r1, %r5;\n@%p1 ld.global.v4.b32 {%r1, %r2, %r3, %r4}, [%rd1];\nadd.f32 %r6, %r1, %r2;"
    )
    du = DefUse(f)
    use = next(i for i in du.insts if i.opcode == "add")
    d = du.last_writer("%r1", du.index_of(use))
    assert d.inst.opcode == "ld" and d.inst.predicate is not None and d.slot == 0


def test_setp_defines_predicate_register():
    f = _entry("mov.u32 %r1, %tid.x;\nsetp.lt.s32 %p1, %r1, %r2;")
    du = DefUse(f)
    assert "%p1" in du.defs_by_reg


def test_def_regs_keys_off_operand0_type():
    reg = Instruction("add", (".s32", ), (RegisterOperand("%r1"), RegisterOperand("%r2"), RegisterOperand("%r3")))
    store = Instruction("st", (".global", ".b32"), (AddressOperand("%rd1", "0"), RegisterOperand("%r2")))
    assert _def_regs(reg) == [("%r1", None)]
    assert _def_regs(store) == []
