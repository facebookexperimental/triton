"""Control-flow analysis for the forward interpreter's sound floor.

The forward interpreter walks the linearized instruction stream in program order and DROPS
predicates / branches. That is sound EXACTLY when all control flow is thread-STRUCTURAL — the
taken-set of every branch is a function of tid / ctaid / ntid / kernel params (already captured by
the leaf coordinates, the shuffle sequence, and reqntid). It is UNSOUND when control flow is
data-dependent (a branch on a loaded value) or unresolved (an indirect branch), because then the
straight-line trace does not represent what every thread actually computes.

This module detects the unresolved case (:func:`has_unknown_control`) and enumerates the predicated
instructions (:func:`predicated_insts`); :mod:`bitequiv.ptx.forward.predicate` classifies each
predicate as structural or data-dependent. The interpreter fails closed when either fires. A full
basic-block CFG (blocks, edges, back-edges) for loop-carry widening is future work; the floor needs
only these two scans.
"""
from pyptx.ir.nodes import RegisterOperand

from bitequiv.ptx.linker import linearize


def has_unknown_control(func):
    """True if the entry contains control flow the straight-line walk cannot resolve statically: an
    indirect branch (``brx``) or a ``bra`` to a computed (register) target rather than a static
    label. The caller must then fail closed — the trace cannot be trusted to cover every path."""
    for inst in linearize(func):
        op = inst.opcode
        if op == "brx" or op.startswith("brx"):
            return True
        if op == "bra" and any(isinstance(o, RegisterOperand) for o in inst.operands):
            return True  # a register (computed) branch target, not a static label
    return False


def predicated_insts(func):
    """Every guarded instruction paired with its ``Predicate`` (``@%p inst``), in program order."""
    out = []
    for inst in linearize(func):
        pred = getattr(inst, "predicate", None)
        if pred is not None:
            out.append((inst, pred))
    return out
