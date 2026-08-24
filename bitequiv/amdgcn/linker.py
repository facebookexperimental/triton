"""Def-use over an AMDGCN entry function — the AMD peer of ``bitequiv.ptx.linker``.

AMDGCN is the real, register-allocated machine ISA: one physical register ``vN``/``sN``
is reused for many unrelated values across a kernel, so a register NAME alone does not
identify a value. Identity is ``(register, stream position)`` — the *last writer before a
use* is the only correct source. Two AMD-specific facts drive this file:

  * an AMD store's / branch's / barrier's ``operand[0]`` is a SOURCE (the address), not a
    destination, so those opcodes define nothing (``_NO_DEF_PREFIXES``);
  * a range destination ``v[2:5]`` defines four sub-registers, each tracked with its slot
    index (the AMD echo of a PTX ``.v4`` vector dest).

Direction-independent: both a backward walk (start at stores) and the forward interpreter
(walk in program order) use the same ``last_writer`` index.
"""
from __future__ import annotations

from dataclasses import dataclass

from bitequiv.amdgcn.parser import RegisterOperand, VectorOperand


def linearize(func):
    """The flat instruction stream of an entry function.

    The AMDGCN parser already emits a flat ``body`` (labels / directives / comments are
    dropped during parsing), so linearization is just a copy of that list. Kept as a named
    function so the interpreter mirrors the PTX engine, whose ``linearize`` must flatten
    nested blocks.
    """
    return list(func.body)


# Opcodes whose operand[0] is NOT a written register (AMD stores take the address in
# operand[0]; branches / barriers / waits define nothing at all).
_NO_DEF_PREFIXES = (
    "global_store", "buffer_store", "flat_store", "scratch_store", "ds_write", "s_store",
    "global_atomic", "buffer_atomic", "flat_atomic", "ds_add", "ds_atomic",
    "s_branch", "s_cbranch", "s_barrier", "s_waitcnt", "s_endpgm", "s_nop", "s_setprio",
    "s_sendmsg", "s_sleep", "s_setreg", "s_waitcnt_vscnt", "s_dcache",
)


def _defines_nothing(opcode):
    return opcode.startswith(_NO_DEF_PREFIXES)


def _def_regs(inst):
    """The ``(register_name, slot)`` pairs an instruction defines.

    A single-register dest gives ``slot=None``; a range dest ``v[2:5]`` gives one pair per
    element with ``slot=0..N-1``; a no-def opcode or an immediate/label operand[0] gives
    nothing.
    """
    if _defines_nothing(inst.opcode) or not inst.operands:
        return []
    o0 = inst.operands[0]
    if isinstance(o0, RegisterOperand):
        return [(o0.name, None)]
    if isinstance(o0, VectorOperand):
        return [(e.name, k) for k, e in enumerate(o0.elements)]
    return []


@dataclass(frozen=True)
class Def:
    """A single write: the defining instruction, its stream index, the register, and the
    sub-register slot (``None`` for a plain single-register dest)."""

    inst: object
    index: int
    reg: str
    slot: object


class DefUse:
    """Last-writer index over the flat stream. Built once per entry function."""

    def __init__(self, func):
        self.insts = linearize(func)
        self._index_of = {id(inst): i for i, inst in enumerate(self.insts)}
        self.defs_by_reg = {}
        for idx, inst in enumerate(self.insts):
            for reg, slot in _def_regs(inst):
                self.defs_by_reg.setdefault(reg, []).append(Def(inst, idx, reg, slot))

    def index_of(self, inst):
        return self._index_of[id(inst)]

    def last_writer(self, reg, before_index):
        """The most recent ``Def`` of ``reg`` strictly before ``before_index`` (the only
        sound source under post-RA register reuse), or ``None`` if none / hardware-init."""
        defs = self.defs_by_reg.get(reg)
        if not defs:
            return None
        best = None
        for d in defs:  # ascending stream order
            if d.index < before_index:
                best = d
            else:
                break
        return best
