"""Loop-structure recovery, shared by the backward ``_loop_steps`` fence and the forward
``LoopReduce`` reconstruction.

``linearize`` drops labels, so we re-walk ``func.body`` to recover label positions, then a backward
branch (``bra`` to a label at/before it) is a loop back-edge. On top of that we classify a loop as
carrying a floating-point ACCUMULATION (a running total across iterations) — the thing that makes its
chunk size bit-relevant — and expose the loop-carried accumulator register(s) + the accumulating
instruction, which the forward interpreter uses to emit a ``LoopReduce`` (fold summary) instead of
unrolling. :func:`is_pure_mma_fold` is the matching POSITIVE recognizer for the tensor-core case: it
proves an entry's fp accumulation is carried by MMA instructions alone, which is what lets the MMA
descriptor drop its chunk (BLOCK_K) fence.
"""
from pyptx.ir.nodes import Block, ImmediateOperand, Label, RegisterOperand, VectorOperand

from bitequiv.core.affine_algebra import Affine
from bitequiv.ptx.affine import AffineEval, reqntid_of
from bitequiv.ptx.linker import DefUse
from bitequiv.ptx.mma import _is_mma

# fp combine ops + widths, used to detect a loop-carried floating-point accumulation.
_FP_WIDTHS = frozenset({".f16", ".f16x2", ".f32", ".f32x2", ".f64", ".bf16", ".bf16x2"})
_FP_COMBINE = frozenset({"add", "sub", "mul", "div", "min", "max", "fma"})

# fp opcodes that COMPUTE a value — arithmetic, transcendentals, comparisons, selects. Any of these
# inside a tensor-core fold means the loop does more than sum MMA products, so re-chunking it can
# regroup that arithmetic (see :func:`is_pure_mma_fold`). Format conversion (``cvt``) and the memory /
# movement ops are deliberately NOT here: they are per-element and cannot regroup a fold, and an fp8
# GEMM upcasts its operands with ``cvt`` inside the K loop yet is still a pure fold. A ``cvt`` (or any
# other op) applied to the ACCUMULATOR is caught by the accumulator-read rule instead.
_FP_VALUE_OPS = frozenset({"add", "sub", "mul", "div", "fma", "mad", "min", "max", "rcp", "sqrt",
                           "rsqrt", "sin", "cos", "lg2", "ex2", "tanh", "abs", "neg", "selp", "setp",
                           "redux", "testp", "copysign", "atom", "red"})
# `.tf32` is normally an MMA operand dtype rather than a scalar-op width; it is listed so that an
# arithmetic op carrying only that width still counts as floating point.
_FP_VALUE_WIDTHS = _FP_WIDTHS | frozenset({".tf32"})


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
    return inst.opcode in _FP_COMBINE and inst.modifiers and any(m in _FP_WIDTHS for m in inst.modifiers)


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
                # A non-literal increment immediate (symbolic / unparseable) is not a fixed chunk size,
                # so we skip it rather than guess. Sound: chunk keys are only ever literal constants
                # (BLOCK_K / BLOCK_N); dropping a symbolic step just keys the loop on its remaining literal
                # steps (or none -> the conservative empty key), never merges two distinct chunkings, and
                # the empirical fuzzer backstops. We do NOT fall back to 0 (0 would be dropped anyway).
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


def _operand_regs(operand):
    """Register name(s) an operand mentions (a scalar register, or every element of a vector)."""
    if isinstance(operand, RegisterOperand):
        return {operand.name}
    if isinstance(operand, VectorOperand):
        return {e.name for e in operand.elements if isinstance(e, RegisterOperand)}
    return set()


def _mma_accumulator_regs(inst):
    """Register(s) an MMA writes its accumulator into, or an empty set when it has none in registers.
    ``tcgen05.mma`` accumulates in TENSOR MEMORY — its first operand is an ADDRESS, not a value — so
    there is no accumulator register to track; reading that accumulator back needs a ``tcgen05.ld``,
    which :func:`is_pure_mma_fold` rejects separately."""
    if inst.opcode == "tcgen05" or not inst.operands:
        return set()
    return _operand_regs(inst.operands[0])


def _is_fp_value_op(inst):
    """True iff ``inst`` COMPUTES a floating-point value (see ``_FP_VALUE_OPS``)."""
    return (inst.opcode in _FP_VALUE_OPS and inst.modifiers
            and any(m in _FP_VALUE_WIDTHS for m in inst.modifiers))


def _is_tmem_load(inst):
    """True iff ``inst`` reads the tensor-memory MMA accumulator back into registers."""
    return inst.opcode == "tcgen05" and ".ld" in inst.modifiers


def is_pure_mma_fold(func):
    """True iff every tensor-core fold in ``func`` accumulates by MMA instructions ALONE — the
    positive recognizer that lets an MMA descriptor drop its chunk (BLOCK_K) fence.

    A tensor-core K-loop is re-chunkable without changing a bit only when one iteration contributes
    exactly one batch of MMA products to an accumulator that nothing else touches: any chunking then
    issues the SAME dot products, in the same order, into the same accumulator. The folds are the
    loops that ISSUE the MMAs (an MMA in their own body); for each, over its whole range:

    1. no non-MMA instruction COMPUTES an fp value (:func:`_is_fp_value_op`) — nothing else joins the
       fold. This is what rejects ``input_precision=tf32x3``, whose 3-pass f32 emulation sums its
       compensation products INSIDE the K loop, so its chunking IS bit-relevant;
    2. no non-MMA instruction reads an MMA accumulator register, and no ``tcgen05.ld`` reads the
       tensor-memory accumulator — the accumulator is consumed only AFTER the fold. This is what
       rejects Flash Attention, which rescales the running MMA accumulator every iteration.

    Each fold is scanned over its FULL range, nested loops included, since arithmetic hidden in a
    nested loop is still part of one iteration's contribution. An OUTER loop that combines whole MMA
    partials (split-K: ``acc += partial``, no MMA of its own) is deliberately NOT a fold here — its
    regrouping is num_splits, which :func:`reduction_trip_signature` fences separately.

    Returns False — fail closed — for anything it cannot prove, including an entry with no recovered
    MMA loop (a fully unrolled or unrecognized fold). Asking for a POSITIVE match is the point: the
    fence check this backs up (:func:`bitequiv.ptx.mma._fence_all_matmul`) only rules out an inexact
    MMA dtype, so on its own it lets through every tensor-core kernel that is not a plain GEMM
    (precision emulation, attention, implicit-GEMM convolution, a tensor-core scan). A positive
    recognizer puts those on the conservative side without needing a case for each."""
    insts, loops = find_loops(func)
    folds = [lp for lp in set(loops) if any(_is_mma(insts[k]) for k in own_body(insts, lp, loops))]
    if not folds:
        return False
    for header, latch in folds:
        body = range(header, latch + 1)
        acc_regs = set()
        for k in body:
            if _is_mma(insts[k]):
                acc_regs |= _mma_accumulator_regs(insts[k])
        for k in body:
            inst = insts[k]
            if _is_mma(inst):
                continue
            if _is_fp_value_op(inst) or _is_tmem_load(inst):
                return False
            if acc_regs and any(acc_regs & _operand_regs(o) for o in inst.operands):
                return False
    return True


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


def _symbolic_bounds_of(insts, regs, ev):
    """Canonical form of the RUNTIME (non-literal) values a counter in ``regs`` is compared against.

    :func:`_setp_bounds_of` sees a bound only when it is an immediate. A reduction loop whose bound is
    computed at run time — the common tiled / triangular case ``for j in range(0, program_id * TILE)``
    — is then invisible, and two configs whose folds are split at DIFFERENT points share a signature.
    Evaluating the bound register as an :class:`~bitequiv.ptx.affine.Affine` over the symbol basis
    recovers exactly the missing fact: the bound is ``TILE * %ctaid``, so ``TILE`` (how many chunks
    this CTA's fold covers before the loop hands over to the next region) is in the key.

    A value that is not PROVABLY affine yields one fixed placeholder rather than its structural token:
    that token inlines unresolved register names, which are allocation noise and would split
    bit-identical configs. Recording only "a runtime bound of unknown form is present" keeps the term
    stable while still separating it from a proven affine one."""
    out = set()
    for i, inst in enumerate(insts):
        if inst.opcode != "setp" or len(inst.operands) < 3:
            continue
        comparands = inst.operands[1:]
        if not any(isinstance(o, RegisterOperand) and o.name in regs for o in comparands):
            continue
        for o in comparands:
            if isinstance(o, RegisterOperand) and o.name not in regs:
                value = ev.of_reg(o.name, i)
                out.add(value.to_str() if isinstance(value, Affine) else "?")
    return out


def _with_bounds(base, bounds):
    """``base`` alone when no runtime bound was recovered (byte-identical to the two-tier signature),
    else ``base`` plus the ``("bound", ...)`` tier."""
    return base if not bounds else (base, ("bound", tuple(sorted(bounds))))


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
    a plain tiled GEMM keeps its tiling-invariant fence.

    Neither tier can see a bound that is not a compile-time constant, so a third ``("bound", ...)``
    tier (:func:`_symbolic_bounds_of`) carries the RUNTIME bounds as affine forms. It is appended, not
    substituted, so a signature that has no runtime bound is byte-identical to before — the tier can
    only ever SPLIT. This is what fences a fold whose split point moves with the output tile: causal
    attention runs its unmasked KV blocks in one loop up to ``BLOCK_M * program_id`` and the masked
    diagonal band in a second loop with DIFFERENT arithmetic, so ``BLOCK_M`` decides which KV blocks
    are rounded which way even though both loops exist, with the same op mix, at every ``BLOCK_M``."""
    insts, loops = find_loops(func)
    splitloops = outer_reduction_loops(insts, loops)
    if not splitloops:
        return None
    ev = AffineEval(DefUse(func), reqntid_of(func))
    clean, bounds = set(), set()
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
        bounds |= _symbolic_bounds_of(insts, set(steps), ev)
        for reg, step in steps.items():
            for bound in _setp_bounds_of(insts, {reg}):
                clean.add((step, bound))
    if clean:
        return _with_bounds(("trip", tuple(sorted(clean))), bounds)
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
    return _with_bounds(("region", tuple(sorted(region))), bounds)
