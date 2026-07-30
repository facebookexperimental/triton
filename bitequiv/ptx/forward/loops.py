"""Loop-structure recovery, shared by the backward ``_loop_steps`` fence and the forward
``LoopReduce`` reconstruction.

``linearize`` drops labels, so we re-walk ``func.body`` to recover label positions, then a backward
branch (``bra`` to a label at/before it) is a loop back-edge. On top of that we classify a loop as
carrying a floating-point ACCUMULATION (a running total across iterations) — the thing that makes its
chunk size bit-relevant — and expose the loop-carried accumulator register(s) + the accumulating
instruction, which the forward interpreter uses to emit a ``LoopReduce`` (fold summary) instead of
unrolling.
"""
from pyptx.ir.nodes import Block, Label

from bitequiv.ptx.mma import _is_mma

# fp combine ops + widths, used to detect a loop-carried floating-point accumulation.
_FP_WIDTHS = frozenset({".f16", ".f16x2", ".f32", ".f32x2", ".f64", ".bf16", ".bf16x2"})
_FP_COMBINE = frozenset({"add", "sub", "mul", "div", "min", "max", "fma"})


def instrs_and_labels(func):
    """Linearized instructions (recursing ``Block`` scopes, like :func:`bitequiv.ptx.linker.linearize`)
    PLUS a map ``label name -> index of the instruction at/after that label``."""
    insts, label_at, pending = [], {}, []

    def walk(stmts):
        for s in stmts:
            if isinstance(s, Block):
                walk(s.body)
            elif isinstance(s, Label):
                pending.append(s.name)
            elif type(s).__name__ == "Instruction":
                while pending:
                    label_at[pending.pop()] = len(insts)
                insts.append(s)

    walk(func.body)
    for name in pending:  # a trailing label points just past the last instruction
        label_at[name] = len(insts)
    return insts, label_at


def back_edges(insts, label_at):
    """(header_idx, latch_idx) for each backward branch — a ``bra`` to a label at/before it = a loop."""
    loops = []
    for i, inst in enumerate(insts):
        if inst.opcode == "bra":
            for o in inst.operands:
                name = getattr(o, "name", None) or getattr(o, "text", None)
                j = label_at.get(name)
                if j is not None and j <= i:
                    loops.append((j, i))
    return loops


def find_loops(func):
    """Convenience: ``(insts, loops)`` for ``func`` (loops = list of ``(header, latch)`` back-edges)."""
    insts, label_at = instrs_and_labels(func)
    return insts, back_edges(insts, label_at)


def innermost_loop(idx, loops):
    """The smallest loop range ``(header, latch)`` containing ``idx``, or None."""
    best = None
    for lp in loops:
        if lp[0] <= idx <= lp[1] and (best is None or (lp[1] - lp[0]) < (best[1] - best[0])):
            best = lp
    return best


def own_body(insts, loop, loops):
    """Indices in ``loop``'s OWN body — its range minus any nested loop's range."""
    h, latch = loop
    return [k for k in range(h, latch + 1) if innermost_loop(k, loops) == loop]


def _is_fp_combine(inst):
    return inst.opcode in _FP_COMBINE and inst.modifiers and inst.modifiers[-1] in _FP_WIDTHS


def _self_accumulate(inst):
    """True if ``inst`` is an fp combine whose destination is also one of its sources (dst = dst + x
    running total). Returns the accumulator register name, or None."""
    if not (_is_fp_combine(inst) and inst.operands):
        return None
    dname = getattr(inst.operands[0], "name", None)
    if dname and any(getattr(s, "name", None) == dname for s in inst.operands[1:]):
        return dname
    return None


def loop_accumulates(insts, loop, loops):
    """True iff the loop's own body carries an FP accumulation: an MMA, or a self-accumulating fp
    combine. This is what makes the loop's chunk size (BLOCK_K / BLOCK_N) bit-relevant."""
    for k in own_body(insts, loop, loops):
        if _is_mma(insts[k]) or _self_accumulate(insts[k]) is not None:
            return True
    return False


def loop_carried_accumulators(insts, loop, loops):
    """The loop-carried FP accumulations in ``loop``'s own body, as ``(acc_name, inst)``:
    a self-accumulating fp combine (``acc = acc + chunk``), or an MMA (accumulator = its dst operand,
    a register or a TMEM address). The forward interpreter uses these to locate ``acc`` and extract
    the chunk (the non-acc input) for a ``LoopReduce``."""
    out = []
    for k in own_body(insts, loop, loops):
        inst = insts[k]
        if _is_mma(inst):
            name = getattr(inst.operands[0], "name", None) if inst.operands else None
            if name:
                out.append((name, inst))
        else:
            acc = _self_accumulate(inst)
            if acc is not None:
                out.append((acc, inst))
    return out
