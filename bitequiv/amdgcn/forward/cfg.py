"""Control-flow inspection for the AMDGCN forward walk — the AMD peer of
``bitequiv.ptx.forward.cfg``.

The forward interpreter walks the flat instruction stream in program order. That is faithful for a
representative ACTIVE lane exactly when the real execution is that straight-line sequence:

  * a BACK-EDGE (a branch to an earlier label) means the body re-executes — a loop. The straight-line
    walk sees the body once, so a loop-carried reduction is NOT faithful and must fail closed (loop
    recovery is handled separately);
  * an INDIRECT branch / call (``s_setpc`` / ``s_call`` / ``s_swappc``) is unknowable -> fail closed;
  * a FORWARD ``s_cbranch_exec{z,nz}`` is an EXEC-mask guard ("skip this block if no lane is active"):
    for an active lane the block runs, so the straight-line walk already covers it — transparent;
  * other FORWARD branches (bounds guards, exits) skip blocks an active lane does not contribute to;
    treated as transparent. The 0-over-merge empirical gate is the backstop — if any forward-branch
    kernel ever over-merges, this must tighten.

Labels + branch targets come from the parser (``func.labels``); a target not found among the labels
is treated as an exit (forward).
"""
from __future__ import annotations

from bitequiv.amdgcn.parser import _normalize_symbol

_INDIRECT = ("s_setpc", "s_call", "s_swappc", "s_rfe")


def branch_target(inst):
    """Normalized label name a branch jumps to, or ``None`` (no label operand)."""
    for o in inst.operands:
        t = getattr(o, "text", None)
        if t:
            return _normalize_symbol(t)
    return None


def has_unknown_control(func):
    """True if the entry has control flow the straight-line walk cannot model at all: an indirect
    branch / call. (Loop back-edges are handled by the loop model in the interpreter, which fails
    closed itself on a loop it cannot reduce; forward guards are transparent — see module docstring.)"""
    return any(inst.opcode.startswith(_INDIRECT) for inst in func.body)


def has_backedge(func):
    """True if the entry contains a loop back-edge (a branch to an at-or-earlier label)."""
    labels = getattr(func, "labels", {}) or {}
    for idx, inst in enumerate(func.body):
        if inst.opcode.startswith(("s_branch", "s_cbranch")):
            ti = labels.get(branch_target(inst))
            if ti is not None and ti <= idx:
                return True
    return False
