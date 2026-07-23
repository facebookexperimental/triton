"""CPU unit tests for the AMDGCN assembly parser (bitequiv/amdgcn/parser.py).

Focused on the structural parse and on the ``.amdgpu_metadata`` workgroup-size extraction —
including the quoted / ``\\01``-mangled name normalization and the per-record pairing that a
reviewer flagged as correctness concerns. No GPU needed."""
import textwrap

from bitequiv.amdgcn import parser
from bitequiv.amdgcn.parser import (
    ImmediateOperand,
    RegisterOperand,
    VectorOperand,
    _normalize_symbol,
    _workgroup_sizes,
)


def _meta(body):
    return "\t.type\tk,@function\nk:\n\ts_endpgm\n\t.amdgpu_metadata\n" + textwrap.dedent(
        body) + "\n\t.end_amdgpu_metadata\n"


# ---- structural parse ---------------------------------------------------


def test_operand_kinds():
    ins = parser.parse_instruction("\tbuffer_load_dwordx4 v[2:5], v2, s[8:11], 0 offen")
    assert ins.opcode == "buffer_load_dwordx4"
    assert isinstance(ins.operands[0], VectorOperand) and len(ins.operands[0].elements) == 4
    assert isinstance(ins.operands[1], RegisterOperand) and ins.operands[1].name == "v2"
    assert isinstance(ins.operands[3], ImmediateOperand) and ins.operands[3].text == "0"
    assert ins.modifiers == ("offen", )


def test_off_and_null_are_immediates_not_registers():
    # documents the module docstring: off/null are ImmediateOperand, not single registers.
    ins = parser.parse_instruction("\tglobal_load_dword v1, v[2:3], off")
    assert isinstance(ins.operands[2], ImmediateOperand) and ins.operands[2].text == "off"


def test_dpp_modifiers_kept():
    ins = parser.parse_instruction("\tv_add_f32_dpp v1, v1, v1 row_shr:8 row_mask:0xf bound_ctrl:1")
    assert ins.opcode == "v_add_f32_dpp"
    assert [o.name for o in ins.operands] == ["v1", "v1", "v1"]
    assert "row_shr:8" in ins.modifiers


# ---- symbol normalization (reviewer nit) --------------------------------


def test_normalize_symbol_strips_quotes_and_mangle_prefix():
    assert _normalize_symbol("'kernel'") == "kernel"
    assert _normalize_symbol('"kernel"') == "kernel"
    assert _normalize_symbol("\\01kernel") == "kernel"
    assert _normalize_symbol("kernel") == "kernel"


def test_quoted_metadata_name_still_matches():
    txt = _meta("""\
        amdhsa.kernels:
          - .max_flat_workgroup_size: 128
            .name:           'k'
            .symbol:         k.kd""")
    assert _workgroup_sizes(txt) == {"k": 128}
    assert parser.parse(txt)[0].reqntid == {"x": 128}  # end-to-end: reqntid recovered


def test_mangled_metadata_name_still_matches():
    txt = _meta("""\
        amdhsa.kernels:
          - .max_flat_workgroup_size: 256
            .name: \\01k""")
    assert _workgroup_sizes(txt) == {"k": 256}


# ---- per-record pairing (reviewer nit) ----------------------------------


def test_malformed_record_does_not_mispair_with_next():
    # record 0 has a size but no .name; it must NOT leak into record 1's name.
    txt = _meta("""\
        amdhsa.kernels:
          - .max_flat_workgroup_size: 64
            .symbol: k0.kd
          - .name: k1
            .max_flat_workgroup_size: 256""")
    wg = _workgroup_sizes(txt)
    assert wg.get("k1") == 256  # k1 paired with ITS OWN size
    assert wg.get("k1") != 64  # not the stale size from the malformed record
    assert "k0" not in wg  # the orphan size is discarded


def test_two_records_both_mapped_either_order():
    txt = _meta("""\
        amdhsa.kernels:
          - .max_flat_workgroup_size: 64
            .name: a
          - .name: b
            .max_flat_workgroup_size: 256""")
    assert _workgroup_sizes(txt) == {"a": 64, "b": 256}
