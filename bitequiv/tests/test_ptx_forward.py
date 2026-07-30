"""Forward interpreter (bitequiv.ptx.forward.interp) — Phase 3 idiom tests.

Idiom 2 (packed .f32x2) is validated on REAL captured PTX (sum_packed_nw1.ptx, a 1-warp inner_tree
sum the compiler folds into .f32x2 + mov.b64): the forward interp must RECONSTRUCT it faithfully
(not fall back to the conservative fingerprint), and its reconstructed tree is pinned by a golden
hash. Idioms 3 (pure-elt in the collapse leaf) and 4 (min/max as collapsible reduce ops) are
collapse-pass properties, tested directly on the reduction IR. All CPU-only (no GPU at test time)."""
import hashlib
import os

from bitequiv.ptx.builder import collapse_balanced
from bitequiv.ptx.forward.interp import forward_module_descriptor
from bitequiv.ptx.treeir import FpOp, ITreeReduce, Leaf, OpaqueOp, ShflCombine

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "ptx")


def _read(name):
    with open(os.path.join(_FIX, f"{name}.ptx")) as fh:
        return fh.read()


def _faithful(desc):
    return bool(desc) and not any("fwd-incomplete" in s for s in desc)


def _digest(desc):
    return hashlib.sha1("␟".join(desc).encode()).hexdigest()[:12]


def _leaf(i):
    """A single-element boundary leaf reading element ``i`` (coord + recoverable column image)."""
    return Leaf(f"c{i}", frozenset({i}))


# -- idiom 2: packed .f32x2 reconstruction (real captured PTX) -----------------


def test_packed_reduction_is_reconstructed_faithfully():
    # Within-thread fold in packed .f32x2 (+ mov.b64 pack/unpack). Before idiom 2 a _Shuffle buried
    # in a b64 op made the forward interp fail closed (fingerprint); idiom 2 decomposes the packed
    # pair per lane, so the reduction tree is reconstructed -> a real tree hash, not "fwd-incomplete".
    assert _faithful(forward_module_descriptor(_read("sum_packed_nw1")))


def test_packed_reduction_golden():
    # Pins the reconstructed packed tree so any regression in the mov.b64 / .f32x2 decomposition is
    # caught on real PTX. Regenerate with _digest(forward_module_descriptor(_read("sum_packed_nw1"))).
    assert _digest(forward_module_descriptor(_read("sum_packed_nw1"))) == "55aefb5132a6"


# -- idiom 4: min/max are collapsible reduce ops -------------------------------


def _balanced4(kind):
    """A balanced 4-leaf reduction ``kind(kind(L0,L1), kind(L2,L3))`` over single-element leaves."""
    lo = FpOp(kind, (".f32", ), (_leaf(0), _leaf(1)))
    hi = FpOp(kind, (".f32", ), (_leaf(2), _leaf(3)))
    return FpOp(kind, (".f32", ), (lo, hi))


def test_min_and_max_reductions_collapse():
    for kind in ("min", "max"):
        c = collapse_balanced(_balanced4(kind))
        assert isinstance(c, ITreeReduce) and c.op == f"{kind}.f32"


def test_min_max_add_collapse_to_distinct_classes():
    # add / min / max over the same leaves must NOT share a descriptor (different ops, different bits).
    sigs = {kind: collapse_balanced(_balanced4(kind)).sig() for kind in ("add", "min", "max")}
    assert len(set(sigs.values())) == 3, sigs


def test_min_butterfly_on_fold_collapses():
    # A within-thread min fold (>= 2 leaves) then a within-warp min butterfly: the whole balanced
    # region collapses (the ShflCombine steps are structural reduce nodes above the fold's leaves),
    # so a min column reduction recovers num_warps the same way a sum does.
    t = FpOp("min", (".f32", ), (_leaf(0), _leaf(1)))
    for off in (1, 2, 4, 8, 16):
        t = ShflCombine(off, "min", (".f32", ), t)
    c = collapse_balanced(t)
    assert isinstance(c, ITreeReduce) and c.op == "min.f32"


# -- idiom 3: pure-elt transcendental kept in the collapse leaf ----------------


def _exp(i):
    return OpaqueOp("ex2.approx.f32", (_leaf(i), ))


def test_transcendental_fed_reduction_collapses():
    # sum(exp(x_i)): the per-element ex2 rides verbatim into the collapse leaf (layout-invariant),
    # so the balanced add over it collapses -- and the leaf_sig records the ex2 so a plain sum(x_i)
    # never merges with it.
    t = FpOp("add", (".f32", ), (FpOp("add", (".f32", ), (_exp(0), _exp(1))),
                                 FpOp("add", (".f32", ), (_exp(2), _exp(3)))))
    c = collapse_balanced(t)
    assert isinstance(c, ITreeReduce) and "ex2.approx.f32" in c.leaf_sig
    plain = collapse_balanced(_balanced4("add"))
    assert isinstance(plain, ITreeReduce) and plain.sig() != c.sig()


def test_squared_reduction_collapses_but_distinct_element_product_does_not():
    # sum(x_i * x_i): mul of ONE element -> per-element -> collapses (rmsnorm / variance recovery).
    sq = FpOp("add", (".f32", ),
              (FpOp("add", (".f32", ), (FpOp("mul", (".f32", ), (_leaf(0), _leaf(0))),
                                        FpOp("mul", (".f32", ), (_leaf(1), _leaf(1))))),
               FpOp("add", (".f32", ), (FpOp("mul", (".f32", ), (_leaf(2), _leaf(2))),
                                        FpOp("mul", (".f32", ), (_leaf(3), _leaf(3)))))))
    assert isinstance(collapse_balanced(sq), ITreeReduce)
    # sum(x_i * x_j), i != j: the pairing is layout-dependent -> the single-element guard REFUSES
    # to collapse (sound; stays over-split). Must remain a plain FpOp tree, not an ITreeReduce.
    prod = FpOp("add", (".f32", ),
                (FpOp("mul", (".f32", ), (_leaf(0), _leaf(1))),
                 FpOp("mul", (".f32", ), (_leaf(2), _leaf(3)))))
    assert not isinstance(collapse_balanced(prod), ITreeReduce)
