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

from dataclasses import dataclass

# Commutative, separately-rounded ops whose two operands may be sorted (bit-safe in IEEE).
_COMMUTATIVE = frozenset({"add", "mul"})


@dataclass(frozen=True)
class Leaf:
    """A loaded tensor element, labeled by its recovered layout coordinate."""

    coord: str  # canonical element-coordinate string (see bitequiv.ptx.leaves)
    children = ()

    def sig_local(self, child_sigs):
        return f"L[{self.coord}]"

    def sig(self):
        return self.sig_local(())


@dataclass(frozen=True)
class OpaqueLeaf:
    """A value whose provenance could not be modeled — matches only an identical token."""

    token: str
    children = ()

    def sig_local(self, child_sigs):
        return f"O[{self.token}]"

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
        m = "".join(self.mods)
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
