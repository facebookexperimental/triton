"""Standalone **PTX** reduction-equivalence signature — an independent layout-equivalence
checker at the PTX level (M1-PTX).

This is **not** a backstop layered after the TTGIR checker; it is a peer "choice of check
level" that reconstructs the floating-point reduction *tree* directly from PTX and labels
each leaf with the tensor element it reads (the PTX echo of the thread<->data map). Two
configs are bitwise-equivalent reductions iff their reconstructed trees are structurally
equal — which captures, from PTX alone:

  * the **layout** — which element each thread/lane/register holds, recovered from the
    load-address arithmetic (``bitequiv.ptx.affine``); different ``num_warps`` /
    ``sizePerThread`` / transpose => different leaf coordinates;
  * the **within-thread fold order** — the def-use chain of the scalar combine ops
    (left-fold vs balanced tree), not just an opcode multiset;
  * the **within-/cross-warp tree** — the butterfly ``shfl.sync.bfly`` steps and the
    shared-memory exchanges;
  * the **rounding / FMA-contraction** decision — ``fma.rn.f32`` vs ``mul``+``add``,
    ``.rn``/``.ftz`` — carried verbatim in each tree node's modifiers.

The PTX is parsed structurally with ``pyptx`` (a real PTX parser, pure-Python). The
engine lives in :mod:`bitequiv.ptx` (``linker`` -> ``affine`` -> ``leaves`` -> ``builder``).

**Soundness — what it does and does NOT guarantee.** This is not an absolute proof that
two configs are bit-identical. It is **sound only relative to the reconstructed
result-DAG at PTX**: it walks back from each result store through the value graph and turns
every op it does not model into a *verbatim* OPAQUE node, so a difference it *can see* in
that graph forces a split. It does not over-merge on the modeled dimensions (layout,
association order, rounding, fusion) or on a syntactically-different unmodeled op on the
path. "Equal signature => equal bits" therefore holds only **under these conditions**:

  * the backward walk reaches every op that affects the output (control flow / predicate
    conditions it cannot follow are a gap);
  * opaque tokens never collide across the two modules compared (a lost-provenance leaf
    that falls back to a register name is a known collision risk);
  * each canonicalization is bit-preserving (today only commutative ``add``/``mul`` operand
    sorting + ``fma``'s multiply pair, which are IEEE-safe);
  * the two kernels are identical outside the captured computation (true when autotuning a
    single kernel) and you accept the **PTX -> SASS residual** (``ptxas`` can still
    reorder/contract below PTX; real ground truth is SASS / the runtime bits).

So it is a cheap static **pre-filter**, not a proof — the runtime bit-comparison
(``bitequiv/evaluation``) is the ground truth. Within those conditions it is conservative
(it over-splits, costing tuning freedom, rather than over-merging). The thread-count
(``.reqntid``) and a tensor-core (``mma``) guard are folded into each entry's signature so
different launch geometries or un-modeled accumulations never collapse to an empty match.

**Residual caveat.** PTX sits above the ``ptxas`` -> SASS gap; ``ptxas`` can still
contract/reorder. PTX is the practical check, not absolute ground truth (SASS via
``cuobjdump`` would be); ``--fmad=false`` pins the fusion decision. See ``design-doc.md``.
"""
from pyptx.ir.nodes import Function, ImmediateOperand, RegisterOperand
from pyptx.parser import parse
from pyptx.parser.parser import ParseError

from bitequiv.ptx.affine import reqntid_of
from bitequiv.ptx.builder import entry_signatures
from bitequiv.ptx.linker import linearize


def _loop_steps(func):
    """Sorted multiset of loop-induction SELF-INCREMENT constants: every ``add R, R, IMM``
    where dst==src0 is a loop counter stepped by a compile-time constant (for a chunked
    reduction this is exactly BLOCK_N — the cross-chunk column stride). This is a LAYOUT-/
    num_warps-INVARIANT, BLOCK_N-FAITHFUL fence: the reconstruction cannot follow the loop
    back-edge, so a collapsed per-chunk reduction loses the chunk SIZE; folding the loop step
    into the entry signature restores it. Adding it can only SPLIT classes further (it is extra
    distinguishing info), so it is monotonically sound — it never causes an over-merge."""
    steps = []
    for inst in linearize(func):
        if inst.opcode == "add" and len(inst.operands) == 3:
            d, a, b = inst.operands
            if (isinstance(d, RegisterOperand) and isinstance(a, RegisterOperand) and d.name == a.name
                    and isinstance(b, ImmediateOperand)):
                try:
                    v = int(b.text, 0)
                except (ValueError, TypeError):
                    continue
                if v != 0:
                    steps.append(v)
    return tuple(sorted(steps))


# Tensor-core opcodes that accumulate without a tree this engine models; an entry carrying
# one gets a conservative fingerprint so it never matches on an empty signature (mirrors the
# TTGIR checker's `unanalyzed-mma` guard). K-axis MMA order is the M3 target.
_MMA_OPCODES = frozenset({"wgmma", "mma", "wmma", "tcgen05"})

# pyptx requires a module header (`.version` first). Real Triton PTX always has one;
# bare-entry fragments (unit-test snippets) do not, so synthesize a minimal header when
# absent. The synthetic version/target only label the module; they do not affect the
# extracted reduction tree.
_SYNTH_HEADER = ".version 8.5\n.target sm_90a\n.address_size 64\n"


class _Unparseable:
    """Sentinel returned when ``pyptx`` cannot parse the PTX at all.

    It compares unequal to everything (including another ``_Unparseable``) so a parse
    failure can never produce a wrong "EQUAL" merge — each unparseable module becomes its
    own singleton class. Real Triton PTX parses cleanly; this covers only the
    parser-cannot-read-it edge."""

    __slots__ = ()

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __hash__(self):
        return id(self)


def _ensure_header(ptx):
    """Prepend a minimal module header when the text lacks a ``.version`` directive."""
    return ptx if ".version" in ptx else _SYNTH_HEADER + ptx


def _mma_guard(func):
    """Conservative fingerprint of tensor-core ops in an entry, or ``""`` if none."""
    toks = {}
    for inst in linearize(func):
        if inst.opcode in _MMA_OPCODES or "mma" in inst.opcode:
            t = inst.opcode + "".join(inst.modifiers)
            toks[t] = toks.get(t, 0) + 1
    if not toks:
        return ""
    body = ",".join(f"{t}x{n}" for t, n in sorted(toks.items()))
    return f"unanalyzed-mma[{body}]"


def _entry_signature(func):
    """Full canonical signature string for one ``.entry``: the reconstructed reduction tree(s)
    + the MMA guard. NOTE: ``ntid`` (launch geometry = 32*num_warps) is deliberately NOT
    included when there is a real reconstructed reduction, because num_warps is exactly the
    layout dimension we want to be invariant to — the reduction tree signature (with balanced
    reductions collapsed to a layout-invariant form) already captures everything bit-relevant.
    ntid is kept ONLY as the empty-signature guard (no reduction reconstructed), so different
    geometries cannot collapse to an empty match."""
    sigs = list(entry_signatures(func))  # one per result store; sorted, canonical
    parts = list(sigs)
    if not sigs:
        ntid = reqntid_of(func)
        parts.append("ntid=" + (",".join(f"{k}{v}" for k, v in sorted(ntid.items())) if ntid else "?"))
    steps = _loop_steps(func)  # BLOCK_N fence for chunked/looped reductions (monotonically sound)
    if steps:
        parts.append("loops=" + ",".join(map(str, steps)))
    guard = _mma_guard(func)
    if guard:
        parts.append(guard)
    return "|".join(parts)


def ptx_reduction_descriptor(ptx):
    """Conservative, hashable PTX reduction-tree signature of a whole module.

    Returns a tuple with one canonical signature string per ``.entry`` (sorted for
    determinism). Equal descriptors mean the configs reduce in the same order down to
    layout, association, rounding and fusion -- *sound relative to the reconstructed
    result-DAG at PTX, under the conditions in the module docstring* (not an absolute proof;
    PTX->SASS and walk-completeness are residuals). Equal reductions may yield different
    descriptors (conservative / over-split). ``()`` when there is no PTX / no entry; an
    :class:`_Unparseable` sentinel (unequal to everything) when ``pyptx`` cannot parse the
    text, so a parse failure never yields a wrong "EQUAL"."""
    if not ptx:
        return ()
    try:
        module = parse(_ensure_header(ptx))
    except ParseError:
        return _Unparseable()
    entries = (d for d in module.directives if isinstance(d, Function) and d.is_entry)
    return tuple(sorted(_entry_signature(f) for f in entries))


def ptx_reductions_equivalent(ptx_a, ptx_b):
    """True iff the two PTX modules reduce in the same (bitwise-equivalent) order —
    same layout, association, rounding and FMA-contraction."""
    return ptx_reduction_descriptor(ptx_a) == ptx_reduction_descriptor(ptx_b)
