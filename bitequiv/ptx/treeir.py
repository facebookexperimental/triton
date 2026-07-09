"""Reduction-tree IR + canonical serialization.

A reconstructed reduction is a tree whose **leaves** are loaded tensor elements
(labeled by the recovered layout coordinate) and whose **internal nodes** are the
floating-point combine ops, the within-/cross-warp butterfly shuffles, and the
shared-memory exchanges. Two PTX modules reduce in the same bitwise order iff their
canonical tree strings are equal.

Canonicalization rules (the bit-safety frontier — a canonicalization is allowed only when
it provably cannot change the bits; an unsafe one would let the checker wrongly merge):

* MAY reorder children only where IEEE makes it bit-identical: the two children of a
  *separately-rounded* commutative ``add``/``mul`` are sorted; ``fma``'s multiply pair
  (children[:2]) is sorted but the addend (children[2]) stays positional.
* MUST stay positional: ``sub``/``div``; ``min``/``max`` (NaN-payload caution); the
  butterfly **offset sequence**; the overall tree *shape* (left-fold vs balanced tree is
  never re-associated). All op modifiers (``.rn``/``.rz``/``.ftz``/width) ride into the
  node string verbatim — this is where FMA-contraction / rounding sensitivity lives.

Each node exposes ``children`` (sub-nodes) and ``sig_local(child_sigs)`` (this node's
canonical string given its children's strings). ``sig()`` composes them recursively for
convenience; deep trees are serialized iteratively in :func:`bitequiv.ptx.builder.tree_sig`
to avoid Python's recursion limit on long within-thread folds.
"""

from dataclasses import dataclass, field

# Commutative, separately-rounded ops whose two operands may be sorted (bit-safe in IEEE).
_COMMUTATIVE = frozenset({"add", "mul"})


def _norm(tok):
    """Normalize an explicit `.rn` rounding modifier to the implicit default. `.rn`
    (round-to-nearest-even) IS the default for PTX add/mul/cvt, so `add.f32` and `add.rn.f32`
    are bit-IDENTICAL — only `enable_fp_fusion` flips which one Triton emits for a pure-add
    reduction. Dropping it merges those (sound; the only rounding mode normalized — `.rz/.rm/.rp`
    and `.ftz` are NOT touched, they do change bits). fma stays distinct from add+mul by node
    STRUCTURE, not by this token, so FMA-contraction is still detected."""
    return tok.replace(".rn", "")


@dataclass(frozen=True)
class Leaf:
    """A loaded tensor element, labeled by its recovered layout coordinate."""

    coord: str  # canonical element-coordinate string (see bitequiv.ptx.leaves)
    # Layout-invariant element-index image (frozenset[int]) of this load across the thread
    # grid, or None if not recoverable. Excluded from identity/sig; used only by the
    # balanced-reduction collapse pass (bitequiv.ptx.builder.collapse_balanced).
    cols: object = field(default=None, compare=False, repr=False)
    children = ()

    def sig_local(self, child_sigs):
        return f"L[{self.coord}]"

    def sig(self):
        return self.sig_local(())


@dataclass(frozen=True)
class ITreeReduce:
    """A *balanced* (inner_tree) reduction collapsed to a LAYOUT-INVARIANT signature.

    inner_tree's contract is that the global balanced reduction tree is fixed by the logical
    element order, independent of how elements are spread across threads (num_warps / lane
    layout). So two configs that balanced-reduce the SAME element-index set with the SAME
    combine op + rounding + leaf computation are bit-identical regardless of layout. We key on
    exactly those (op+mods, coord-free leaf sig, the column-index image) and DROP the physical
    thread/shuffle structure — recovering the num_warps freedom the structural hash over-split.

    Sound ONLY for balanced reductions (left-fold / unordered keep their physical structure),
    and the column image must be fully recovered (else we do not collapse)."""

    op: str  # reduce op + modifiers, e.g. "add.rn.f32"
    leaf_sig: str  # canonical coord-free signature of the (uniform) leaf computation
    height: int  # reduction tree height (~log2 of element count); distinguishes chunk sizes
    shfl_seq: tuple = ()  # butterfly offsets in depth order — the SHAPE key (count-up vs
    #                       count-down pair lanes differently -> different bits). num_warps-
    #                       invariant (within-warp butterfly is the same for any num_warps), so
    #                       it distinguishes inner_tree from unordered without blocking recovery.
    children = ()

    def sig_local(self, child_sigs):
        return f"ITREE[{_norm(self.op)};{self.leaf_sig};h{self.height};s{self.shfl_seq}]"

    def sig(self):
        return self.sig_local(())


@dataclass(frozen=True)
class OpaqueLeaf:
    """A value whose provenance could not be modeled — matches only an identical token."""

    token: str
    children = ()

    def sig_local(self, child_sigs):
        return f"O[{self.token}]"  # VERBATIM: opaque = unmodeled, must match exactly (no .rn norm)

    def sig(self):
        return self.sig_local(())


@dataclass(frozen=True)
class OpaqueOp:
    """An unmodeled op kept as an internal node: its ``token`` (opcode + modifiers + any
    non-register operands) plus its register operands as ``children``. This keeps the tree
    faithful (e.g. an f64 half-assembly ``or.b64`` or a ``cvt`` over the reduction) and is
    built/serialized iteratively, so two trees match only on identical structure."""

    token: str
    children: tuple

    def sig_local(self, child_sigs):
        # VERBATIM token: an unmodeled op must match exactly (e.g. welford's div/combine). Do NOT
        # .rn-normalize inside opaque tokens — that broke soundness (welford over-merged).
        if not child_sigs:
            return f"O[{self.token}]"
        return f"O[{self.token}]({','.join(child_sigs)})"

    def sig(self):
        return self.sig_local([c.sig() for c in self.children])


@dataclass(frozen=True)
class FpOp:
    """A floating-point combine: ``kind`` in {add,sub,mul,div,min,max} or ``fma``."""

    kind: str
    mods: tuple  # full modifier tuple, e.g. ('.rn', '.f32')
    children: tuple
    fused: bool = False  # True for fma: children = (a, b, c) computing a*b + c

    def sig_local(self, child_sigs):
        # .rn is the DEFAULT rounding for add/mul/fma (op.f32 == op.rn.f32 bit-for-bit), so strip an
        # explicit .rn there to merge enable_fp_fusion's implicit-vs-explicit-.rn difference (e.g.
        # the persistent kernel's cross-chunk add). Do NOT touch div/sub/min/max (kept verbatim) —
        # normalizing those broke welford soundness.
        m = _norm("".join(self.mods)) if self.kind in ("add", "mul", "fma") else "".join(self.mods)
        kids = list(child_sigs)
        if self.fused and len(kids) == 3:
            ab = sorted(kids[:2])
            return f"fma{m}({ab[0]},{ab[1]};{kids[2]})"
        if self.kind in _COMMUTATIVE and not self.fused and len(kids) == 2:
            kids = sorted(kids)
        return f"{self.kind}{m}({','.join(kids)})"

    def sig(self):
        return self.sig_local([c.sig() for c in self.children])


@dataclass(frozen=True)
class ShflCombine:
    """A butterfly-shuffle reduction step: combine a partial with its lane^offset partner.
    The offset (and its position in the sequence) is order-significant."""

    offset: object  # int immediate, or a verbatim token for a dynamic/unresolved offset
    kind: str
    mods: tuple
    child: object  # the partial being reduced (before this shuffle step)

    @property
    def children(self):
        return (self.child, )

    def sig_local(self, child_sigs):
        return f"shfl{self.offset}.{self.kind}{''.join(self.mods)}({child_sigs[0]})"

    def sig(self):
        return self.sig_local([self.child.sig()])


@dataclass(frozen=True)
class SmemExchange:
    """Warp-leader partials crossing through shared memory (a phase boundary in the
    cross-warp reduction). The transpose itself is implied by the surrounding shuffle
    offsets; this node just marks the exchange so the tree shape is faithful."""

    child: object

    @property
    def children(self):
        return (self.child, )

    def sig_local(self, child_sigs):
        return f"smem({child_sigs[0]})"

    def sig(self):
        return self.sig_local([self.child.sig()])
