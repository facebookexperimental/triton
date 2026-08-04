"""Forward checker: an unrolled loop-carried fold must not over-merge across chunk groupings.

A chunked / persistent reduction (``acc = acc (+|*) chunk`` folded over ``R / R0_BLOCK``
loop iterations) is bit-sensitive to its chunk grouping: a different ``R0_BLOCK`` groups the
FP accumulation differently, so the low mantissa bits differ (fp add/mul is non-associative).
When the compiler FULLY UNROLLS that loop there is no PTX back-edge, so ``_loop_steps`` finds
no self-increment and emits no ``loops=`` fence. On such kernels the forward walk also fails to
reconstruct the within-thread chunk fold: the output tree bottoms out at the accumulator SEED
(a constant) reduced only across warps (butterfly / shared exchange), with NO loaded ``Leaf``.
Its collapsed hash is then blind to the chunk grouping and two different ``R0_BLOCK`` chunkings
collapse to the SAME descriptor -> over-merge (a real soundness violation; verified on the
torch-Inductor ``prod_loop_f32`` / ``splitsum_2kernel_f32`` kernels).

The fix (``forward_descriptor``): when a FAITHFUL reconstruction reduces over a cross-thread
structure but reaches NO loaded ``Leaf``, the reduced input data was lost, so fail closed to
the conservative fingerprint AUGMENTED with the global-load shape (``ldg=`` — the vector-width
x count multiset), a chunk-bearing residual. Different chunkings load different vector widths /
counts, so they split. Monotone-sound: this only ever converts an over-merging hash into a
more-split fingerprint, never the reverse.

CPU-only (parses hand-crafted PTX, no GPU)."""
from bitequiv.ptx.forward.interp import forward_module_descriptor as CHECK

_HEAD = ".version 8.5\n.target sm_90a\n.address_size 64\n"


def _faithful(desc):
    """A descriptor is a faithful reconstruction (not the conservative fingerprint)."""
    return bool(desc) and "fwd-incomplete" not in str(desc[0])


# A reduction whose output tree bottoms out at a CONSTANT accumulator seed reduced across the
# warp (butterfly over a shared exchange), with NO loaded leaf reaching the store -- the shape
# an unrolled loop-carried fold leaves when the within-thread chunk fold is not reconstructed.
# `chunk_loads` are the (dangling) per-chunk global loads: they do NOT feed the store, so the
# reduction tree is IDENTICAL for both variants -- the ONLY difference is the load shape (the
# chunk grouping). Everything else (ntid / shfl / shared store / fp count) is byte-identical, so
# the two descriptors can differ ONLY through the `ldg=` chunk residual the fix appends.
def _unrolled_seed_fold(chunk_loads):
    return _HEAD + """
.visible .entry k(.param .u64 pin, .param .u64 pout)
.reqntid 256
{
  .reg .pred %p<2>;
  .reg .f32 %f<8>;
  .reg .b32 %r<40>;
  .reg .b64 %rd<16>;
  .shared .align 4 .b8 buf[512];
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  cvta.to.global.u64 %rd3, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd4, %r1, 16;
  add.s64 %rd5, %rd3, %rd4;
""" + chunk_loads + """
  mov.f32 %f1, 0f3F800000;
  mov.u32 %r2, buf;
  st.shared.f32 [%r2], %f1;
  bar.sync 0;
  ld.shared.f32 %f2, [%r2];
  shfl.sync.bfly.b32 %r3, %f2, 16, 31, -1;
  add.f32 %f4, %f2, %r3;
  cvta.to.global.u64 %rd6, %rd2;
  st.global.f32 [%rd6], %f4;
  ret;
}
"""


# Chunk grouping A: four .v2 chunk loads (a finer chunking, more/narrower chunks).
_CHUNK_A = (
    "  ld.global.v2.b32 {%r10,%r11}, [%rd5];\n"
    "  ld.global.v2.b32 {%r12,%r13}, [%rd5+8];\n"
    "  ld.global.v2.b32 {%r14,%r15}, [%rd5+16];\n"
    "  ld.global.v2.b32 {%r16,%r17}, [%rd5+24];\n")
# Chunk grouping B: two .v4 chunk loads (a coarser chunking; SAME total 8 elements as A, so the
# reconstructed reduction tree is identical -- only the grouping/vectorization differs).
_CHUNK_B = (
    "  ld.global.v4.b32 {%r10,%r11,%r12,%r13}, [%rd5];\n"
    "  ld.global.v4.b32 {%r14,%r15,%r16,%r17}, [%rd5+16];\n")

PTX_CHUNK_A = _unrolled_seed_fold(_CHUNK_A)
PTX_CHUNK_B = _unrolled_seed_fold(_CHUNK_B)

# Positive control: the SAME butterfly reduction but over a LOADED leaf (not a constant seed).
# A real reduction always reaches its loaded leaves, so the fix must NOT fail-close it.
PTX_LEAF_REDUCTION = _HEAD + """
.visible .entry k(.param .u64 pin, .param .u64 pout)
.reqntid 256
{
  .reg .pred %p<2>;
  .reg .f32 %f<8>;
  .reg .b32 %r<8>;
  .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  cvta.to.global.u64 %rd3, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd4, %r1, 4;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  shfl.sync.bfly.b32 %r3, %f1, 16, 31, -1;
  add.f32 %f4, %f1, %r3;
  cvta.to.global.u64 %rd6, %rd2;
  st.global.f32 [%rd6], %f4;
  ret;
}
"""


def test_unrolled_seed_fold_fails_closed():
    # A reduction that reaches no loaded leaf (only the seed + cross-warp shuffle) cannot certify
    # its chunk grouping -> fail closed to the conservative fingerprint (the soundness floor).
    for ptx in (PTX_CHUNK_A, PTX_CHUNK_B):
        d = CHECK(ptx)
        assert d and "fwd-incomplete" in str(d[0]), d


def test_different_chunking_does_not_over_merge():
    # THE REGRESSION PIN. The two kernels differ ONLY in loop chunking (four .v2 vs two .v4 chunk
    # loads); the reconstructed reduction tree is byte-identical, so WITHOUT the fix both collapse
    # to the same tree hash -> over-merge. With the fix they fail closed with distinct `ldg=` chunk
    # residuals, so they SPLIT -- as they must (different chunkings = different FP order = different bits).
    da, db = CHECK(PTX_CHUNK_A), CHECK(PTX_CHUNK_B)
    assert da != db, (da, db)
    # And the split is due to the chunk residual only: the fingerprints match up to `ldg=`.
    assert str(da[0]).split("|ldg=")[0] == str(db[0]).split("|ldg=")[0], (da, db)
    assert "ldg=" in str(da[0]) and "ldg=" in str(db[0])


def test_leaf_bearing_reduction_still_recovers():
    # Guard against over-triggering: an ordinary butterfly reduction over a LOADED leaf reconstructs
    # faithfully (a real tree hash), so the no-leaf fail-close must NOT fire here.
    assert _faithful(CHECK(PTX_LEAF_REDUCTION))
