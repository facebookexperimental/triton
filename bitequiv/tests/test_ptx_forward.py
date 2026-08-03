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

_HDR = ".version 8.5\n.target sm_90a\n.address_size 64\n"

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


# -- Phase 4: MMA entries take the tiling-invariant fence composition ----------


def test_mma_entry_gets_fence_descriptor():
    # A wgmma entry -> the tiling-invariant fence (K + dtypes kept, m/n dropped), not a tree hash or
    # the old coarse unanalyzed-mma guard.
    ptx = (_HDR + ".visible .entry k()\n{\n.reg .b32 %r<8>;\n"
           "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%r1}, %r2, %r3;\nret;\n}\n")
    (desc, ) = forward_module_descriptor(ptx)
    assert desc.startswith("mma|tok=") and "k16" in desc and "m64" not in desc


# -- Phase 4b: a reduction OVER the MMA output rides the reduction fingerprint (soundness) ----------
# The tiling-invariant fence over-merges a GEMM whose epilogue REDUCES C = A @ B across configs that
# differ only in the reduction ORDER (gemm_reduce_sum / softmax / layernorm): the reduction can lower
# to a within-thread add fold with NO shfl.bfly, which the old shfl-only trigger missed -> the fence
# alone -> unordered and inner_tree collapse to one descriptor -> OVER-MERGE (measured 1/kernel).
# The MMA output is now an ``Mma`` leaf and ``_epilogue_reduces_mma`` detects the fold, so the
# reduction fingerprint rides the descriptor and the orders split.

_WGMMA4 = "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f1, %f2, %f3, %f4}, %rd3, %rd4;\n"


def _mma_body(body):
    return (_HDR + ".visible .entry k(.param .u64 po)\n{\n.reg .b64 %rd<6>;\n.reg .f32 %f<16>;\n"
            "ld.param.u64 %rd1, [po];\ncvta.to.global.u64 %rd2, %rd1;\n" + body + "ret;\n}\n")


def test_mma_within_thread_reduction_rides_fingerprint():
    # A within-thread add fold over the 4 MMA-output lanes (no shfl.bfly) is a reduction over the MMA
    # output -> the descriptor carries "mma+red" (the fingerprint), not the fence alone.
    (desc, ) = forward_module_descriptor(
        _mma_body(_WGMMA4 + "add.f32 %f5, %f1, %f2;\nadd.f32 %f6, %f5, %f3;\n"
                  "add.f32 %f7, %f6, %f4;\nst.global.f32 [%rd2], %f7;\n"))
    assert desc.startswith("mma|") and "mma+red" in desc


def test_mma_elementwise_epilogue_stays_pure_fence():
    # An elementwise epilogue (acc + bias): the add has a non-Mma child (bias from ld.global), so it is
    # NOT a reduction over the MMA output -> the clean tiling-invariant fence, no "mma+red" (must not
    # over-split the pure GEMMs, e.g. gemm_bias_relu).
    (desc, ) = forward_module_descriptor(
        _mma_body(_WGMMA4 + "ld.global.f32 %f5, [%rd2];\nadd.f32 %f6, %f1, %f5;\n"
                  "st.global.f32 [%rd2], %f6;\n"))
    assert desc.startswith("mma|") and "mma+red" not in desc


def test_mma_reduction_orders_do_not_over_merge():
    # Two entries with the IDENTICAL MMA fence but within-thread reductions of different op count (the
    # unordered-255 vs inner_tree-209 shape difference) must NOT share a descriptor. The old shfl-only
    # trigger left both as the bare fence -> one class -> over-merge; riding the fingerprint splits them.
    three = forward_module_descriptor(
        _mma_body(_WGMMA4 + "add.f32 %f5, %f1, %f2;\nadd.f32 %f6, %f5, %f3;\n"
                  "add.f32 %f7, %f6, %f4;\nst.global.f32 [%rd2], %f7;\n"))
    two = forward_module_descriptor(
        _mma_body(_WGMMA4 + "add.f32 %f6, %f1, %f2;\nadd.f32 %f7, %f6, %f3;\n"
                  "st.global.f32 [%rd2], %f7;\n"))
    assert three != two


# -- Phase 5: data-dependent control flow fails closed (sound floor) -----------


def test_data_dependent_branch_fails_closed():
    # A predicate on a LOADED value -> the straight-line walk is unsound -> fingerprint with dd1.
    ptx = (_HDR + ".visible .entry k()\n{\n.reg .b32 %r<4>;\n.reg .b64 %rd<4>;\n"
           ".reg .pred %p<2>;\n.reg .f32 %f<4>;\n"
           "ld.global.f32 %f1, [%rd1];\nsetp.lt.f32 %p1, %f1, 0f3F000000;\n"
           "@%p1 add.f32 %f2, %f1, %f1;\nst.global.f32 [%rd2], %f2;\nret;\n}\n")
    (desc, ) = forward_module_descriptor(ptx)
    assert "fwd-incomplete" in desc and "dd1" in desc


def test_structural_branch_does_not_fail_close():
    # A tid-guarded (structural) predicate is safe to drop -> NOT fail-closed for control flow (dd1).
    ptx = (_HDR + ".visible .entry k()\n{\n.reg .b32 %r<4>;\n.reg .b64 %rd<4>;\n"
           ".reg .pred %p<2>;\n.reg .f32 %f<4>;\n"
           "ld.global.f32 %f1, [%rd1];\nmov.u32 %r1, %tid.x;\nsetp.lt.s32 %p1, %r1, 64;\n"
           "@%p1 add.f32 %f2, %f1, %f1;\nst.global.f32 [%rd2], %f2;\nret;\n}\n")
    (desc, ) = forward_module_descriptor(ptx)
    assert "dd1" not in desc
