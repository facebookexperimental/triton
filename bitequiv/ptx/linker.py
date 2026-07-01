"""Def-use / last-writer linker over a pyptx function body.

``pyptx`` parses PTX into a flat (per-block) instruction list with **no** SSA / def-use
information. PTX virtual registers are *mostly* single-assignment, but predicated defs
and the masked-load idiom (``mov %r1, 0`` then ``@%p ld.v4 {%r1,...}``) write the same
register twice, and register names are not values. So we build a **last-writer-by-stream
-position** index: a use of ``%rN`` at stream index *i* binds to the most recent prior
def of ``%rN`` (index < *i*). This correctly handles register reuse and the scrambled
within-thread fold chain (the order is recovered from def-use edges, never from the
textual register numbers).

What counts as a *def* is keyed off the **type of operand[0]**, not an opcode allowlist:
a ``RegisterOperand`` (or each element of a ``VectorOperand`` for a ``v2``/``v4`` load,
tagged with its slot) is a register def; an ``AddressOperand`` in operand[0] is a memory
store (``st.*``) and defines no register; ``setp``'s ``PipeOperand`` defines its two
predicate registers. Immediate/label operand[0] (``bar.sync 0``, ``bra L``) define nothing.
"""

from dataclasses import dataclass

from pyptx.ir.nodes import (
    Block,
    Instruction,
    PipeOperand,
    RegisterOperand,
    VectorOperand,
)


@dataclass(frozen=True)
class Def:
    """One register definition at a specific point in the instruction stream."""

    inst: Instruction
    index: int  # position in the linearized instruction stream
    reg: str  # the register name written
    slot: int | None  # vector-element position (0..N-1) for a vector dest, else None


def linearize(func):
    """Flatten a function body (recursing into nested ``Block`` scopes) into a single
    list of ``Instruction`` in source order. Non-instruction statements (labels,
    declarations, comments) are skipped — control flow is recovered, when needed, from
    the instructions themselves (we never reorder)."""
    out = []

    def walk(stmts):
        for s in stmts:
            if isinstance(s, Block):
                walk(s.body)
            elif isinstance(s, Instruction):
                out.append(s)

    walk(func.body)
    return out


def _def_regs(inst):
    """Registers written by ``inst``, as ``(name, slot)`` pairs. Keyed off operand[0]'s
    type so stores (memory dest) and branches/barriers correctly define nothing."""
    if not inst.operands:
        return []
    o0 = inst.operands[0]
    if isinstance(o0, RegisterOperand):
        return [(o0.name, None)]
    if isinstance(o0, VectorOperand):
        return [(e.name, k) for k, e in enumerate(o0.elements) if isinstance(e, RegisterOperand)]
    if isinstance(o0, PipeOperand):  # setp dual-predicate output: %p0|%p1
        return [(s.name, None) for s in (o0.left, o0.right) if isinstance(s, RegisterOperand)]
    return []  # AddressOperand (store), Immediate (bar.sync), Label (bra) -> no reg def


class DefUse:
    """Last-writer index over one entry's linearized instructions."""

    def __init__(self, func):
        self.insts = linearize(func)
        self._index_of = {id(inst): i for i, inst in enumerate(self.insts)}
        self.defs_by_reg: dict[str, list[Def]] = {}
        for idx, inst in enumerate(self.insts):
            for reg, slot in _def_regs(inst):
                self.defs_by_reg.setdefault(reg, []).append(Def(inst, idx, reg, slot))

    def index_of(self, inst):
        """Stream index of an instruction (as returned by :meth:`linearize`)."""
        return self._index_of[id(inst)]

    def last_writer(self, reg, before_index):
        """The most recent def of ``reg`` strictly before ``before_index`` (the using
        instruction's stream index), or ``None`` if there is none (a special register,
        a kernel input not produced in-body, or a true first use)."""
        defs = self.defs_by_reg.get(reg)
        if not defs:
            return None
        best = None
        for d in defs:  # defs_by_reg is built in ascending stream order
            if d.index < before_index:
                best = d
            else:
                break
        return best
