"""Loop-structure recovery, shared by the backward ``_loop_steps`` fence and the forward
``LoopReduce`` reconstruction.

``linearize`` drops labels, so we re-walk ``func.body`` to recover label positions, then a backward
branch (``bra`` to a label at/before it) is a loop back-edge. On top of that we classify a loop as
carrying a floating-point ACCUMULATION (a running total across iterations) — the thing that makes its
chunk size bit-relevant — and expose the loop-carried accumulator register(s) + the accumulating
instruction, which the forward interpreter uses to emit a ``LoopReduce`` (fold summary) instead of
unrolling.
"""
from pyptx.ir.nodes import Block, ImmediateOperand, Label, RegisterOperand

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


def loop_self_increments(insts, loop, loops):
    """Sorted immediates of self-increment ``add R, R, IMM`` (dst==src0, IMM != 0) in ``loop``'s own
    body — the chunk step(s): BLOCK_K for a GEMM K-loop, BLOCK_N for a chunked reduction. Used as the
    forward ``LoopReduce`` key so different chunk sizes get different keys (split, sound)."""
    out = []
    for k in own_body(insts, loop, loops):
        inst = insts[k]
        if inst.opcode == "add" and len(inst.operands) == 3:
            d, a, b = inst.operands
            if (isinstance(d, RegisterOperand) and isinstance(a, RegisterOperand) and d.name == a.name
                    and isinstance(b, ImmediateOperand)):
                try:
                    v = int(b.text, 0)
                except (ValueError, TypeError):
                    continue
                if v != 0:
                    out.append(v)
    return tuple(sorted(out))


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


def outer_reduction_loops(insts, loops):
    """Loops whose range contains BOTH a matmul AND a self-accumulating fp combine (``acc = acc + x``)
    — an OUTER reduction that COMBINES matmul partials. Split-K is the canonical case: the split loop's
    ``acc += partial`` wraps the inner K-loop's MMA. A plain tiled GEMM's single K-loop contains the
    MMA but NO in-loop self-accumulate (the MMA itself IS the accumulation into a TMEM/register acc),
    so it is not flagged and keeps its tiling-invariant fence.

    The outer loop's TRIP COUNT (= number of partials combined = num_splits) is bit-relevant: it
    regroups the K sum (``((s0)+(s1))+...`` vs one in-order fold), so different trips are NOT
    bitwise-equivalent even though the static MMA/op structure is identical across trips. This is the
    reusable structural hook for any nested-reduction GEMM (split-K, tree-combine, GEMM+reduction).

    Detection = a loop whose range contains a self-accumulating fp combine ``acc = acc + x`` (dst ==
    src0) — a running fp total folded across iterations. Split-K's split loop (``acc += partial``) is
    exactly that; a plain tiled GEMM accumulates via the MMA/TMEM accumulator itself (no fp add) and
    has none, so it keeps its tiling-invariant fence.

    Keying on the self-accumulate combine (not nested matmul loops) is what survives the compiler
    restructuring at some tilings: at BLOCK_N=256 the split loop and the MMA K-loop are emitted as
    SEPARATE / overlapping loops (the split loop's range holds NO matmul), so a nested-matmul-loop test
    misses the split structure and over-merges the split counts. The split loop still carries the
    ``acc += partial`` combine, so a range scan for it fires in both the nested and the flattened form."""
    out = []
    for lp in loops:
        h, latch = lp
        if any(_self_accumulate(insts[k]) is not None for k in range(h, latch + 1)):
            out.append(lp)
    return out


def _setp_bounds_of(insts, regs):
    """Constants each register in ``regs`` is compared to across the entry's ``setp`` instructions."""
    out = set()
    for inst in insts:
        if inst.opcode == "setp" and len(inst.operands) >= 3:
            for x, y in ((inst.operands[1], inst.operands[2]), (inst.operands[2], inst.operands[1])):
                if isinstance(x, RegisterOperand) and x.name in regs and isinstance(y, ImmediateOperand):
                    try:
                        out.add(int(y.text, 0))
                    except (ValueError, TypeError):
                        continue
    return out


def reduction_trip_signature(func):
    """Hashable fingerprint that distinguishes the outer reduction loops' TRIP COUNTS (the split-K
    regrouping = num_splits), or ``None`` when the entry has no nested-reduction structure (a plain
    GEMM, whose K sum is a single in-order fold that tiling never regroups). Same num_splits across any
    tiling -> same signature (RECOVERS num_warps / BLOCK_M / BLOCK_N); different num_splits -> different
    signature (SOUND).

    A split-K GEMM's static instruction structure is IDENTICAL across split counts — only the outer
    loop's runtime TRIP differs — so we key on the loop-control constants, in two tiers:
    - the ``("trip", ...)`` tier is the split loops' scalar self-increment COUNTER `(step, bound)` (a
      LOOPED split has `s += 1` guarded by `s < num_splits`); the bound is num_splits, tiling-invariant.
    - a small split count is peeled (no counter) at some tilings, so the ``("region", ...)`` tier falls
      back to the split loops' in-region ``setp`` constants (the split + inner-K bounds), which are
      likewise tiling-invariant per split count.
    Both are keyed on num_splits and independent of the tiling axes, so they recover the tiling freedom
    while keeping the split counts distinct. Returns ``None`` unless an outer reduction loop exists, so
    a plain tiled GEMM keeps its tiling-invariant fence."""
    insts, loops = find_loops(func)
    splitloops = outer_reduction_loops(insts, loops)
    if not splitloops:
        return None
    clean = set()
    for h, latch in splitloops:
        steps = {}
        for k in range(h, latch + 1):
            inst = insts[k]
            if inst.opcode == "add" and len(inst.operands) == 3:
                d, a, b = inst.operands
                if (isinstance(d, RegisterOperand) and isinstance(a, RegisterOperand)
                        and d.name == a.name and isinstance(b, ImmediateOperand)):
                    try:
                        steps[d.name] = int(b.text, 0)
                    except (ValueError, TypeError):
                        continue
        for reg, step in steps.items():
            for bound in _setp_bounds_of(insts, {reg}):
                clean.add((step, bound))
    if clean:
        return ("trip", tuple(sorted(clean)))
    region = []
    for h, latch in splitloops:
        cs = set()
        for k in range(h, latch + 1):
            inst = insts[k]
            if inst.opcode == "setp" and len(inst.operands) >= 3:
                for o in inst.operands[1:]:
                    if isinstance(o, ImmediateOperand):
                        try:
                            cs.add(int(o.text, 0))
                        except (ValueError, TypeError):
                            continue
        region.append(tuple(sorted(cs)))
    return ("region", tuple(sorted(region)))
