"""Forward SmemModel (phase + same-structure guard) soundness + recovery, on hand-crafted PTX.

The model records each ``st.shared`` element (scalar OR vector) into the current write phase, closes
the phase at ``bar.sync``, and resolves a ``ld.shared`` / ``ldmatrix`` against the most-recently closed
phase: it returns a representative write IFF every write in that phase has the SAME coord-free value
structure, else it FAILS CLOSED. This replaces the old blanket "vector st.shared -> fail closed" floor
(D112727538). Sound because the reader's element is interchangeable with any write of the same
structure; a mixed-structure phase can't be resolved without the exact address -> fail closed.

CPU-only (parses hand-crafted PTX, no GPU)."""
from bitequiv.ptx.forward.interp import forward_module_descriptor as CHECK

_HEAD = ".version 8.5\n.target sm_90a\n.address_size 64\n"


def _faithful(desc):
    """A descriptor is a faithful reconstruction (not the conservative fingerprint)."""
    return bool(desc) and "fwd-incomplete" not in str(desc[0])


# Scalar shared exchange: st.shared A -> bar -> ld.shared A -> reconstructs faithfully (real tree hash).
PTX_SCALAR = _HEAD + """
.visible .entry k(.param .u64 pin, .param .u64 pout) {
  .reg .f32 %f<4>;
  .reg .b32 %r<4>;
  .reg .b64 %rd<8>;
  .shared .align 4 .b8 bufA[128];
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  cvta.to.global.u64 %rd3, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd4, %r1, 4;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  mov.u32 %r2, bufA;
  st.shared.f32 [%r2], %f1;
  bar.sync 0;
  ld.shared.f32 %f2, [%r2];
  cvta.to.global.u64 %rd6, %rd2;
  st.global.f32 [%rd6], %f2;
  ret;
}
"""

# VECTOR shared store, both elements the same structure (Leaf f1): now MODELED per element (was a blanket
# fail-close). The load reads the phase {Leaf, Leaf} -> same coord-free sig -> reconstructs faithfully.
PTX_VECTOR = _HEAD + """
.visible .entry k(.param .u64 pin, .param .u64 pout) {
  .reg .f32 %f<4>;
  .reg .b32 %r<4>;
  .reg .b64 %rd<8>;
  .shared .align 8 .b8 bufB[128];
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  cvta.to.global.u64 %rd3, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd4, %r1, 4;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  mov.u32 %r3, bufB;
  st.shared.v2.f32 [%r3], {%f1, %f1};
  bar.sync 0;
  ld.shared.f32 %f2, [%r3];
  cvta.to.global.u64 %rd6, %rd2;
  st.global.f32 [%rd6], %f2;
  ret;
}
"""

# MIXED-structure phase: one write is a Leaf (f1), the other a combine add(f1,f2). Their coord-free sigs
# differ, so a following ld.shared cannot be resolved without the exact address -> FAIL CLOSED (the real
# soundness guard now that vector stores are modeled).
PTX_MIXED = _HEAD + """
.visible .entry k(.param .u64 pin, .param .u64 pout) {
  .reg .f32 %f<8>;
  .reg .b32 %r<4>;
  .reg .b64 %rd<8>;
  .shared .align 4 .b8 bufA[128];
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  cvta.to.global.u64 %rd3, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd4, %r1, 4;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  add.s64 %rd6, %rd5, 4;
  ld.global.f32 %f2, [%rd6];
  add.f32 %f3, %f1, %f2;
  mov.u32 %r2, bufA;
  st.shared.f32 [%r2], %f1;
  add.s32 %r3, %r2, 4;
  st.shared.f32 [%r3], %f3;
  bar.sync 0;
  ld.shared.f32 %f4, [%r2];
  cvta.to.global.u64 %rd7, %rd2;
  st.global.f32 [%rd7], %f4;
  ret;
}
"""

# A ld.shared with NO prior store (empty phase) -> fail closed.
PTX_NOSTORE = _HEAD + """
.visible .entry k(.param .u64 pin, .param .u64 pout) {
  .reg .f32 %f<4>;
  .reg .b32 %r<4>;
  .reg .b64 %rd<8>;
  .shared .align 4 .b8 bufA[128];
  ld.param.u64 %rd2, [pout];
  mov.u32 %r2, bufA;
  ld.shared.f32 %f2, [%r2];
  cvta.to.global.u64 %rd6, %rd2;
  st.global.f32 [%rd6], %f2;
  ret;
}
"""


def test_scalar_shared_exchange_reconstructs():
    assert _faithful(CHECK(PTX_SCALAR))  # scalar exchange is modeled -> faithful tree hash


def test_vector_shared_store_now_modeled():
    # The old floor fail-closed here; the phase model records each vector element, and both elements
    # have the SAME structure, so the load resolves and the reconstruction is faithful (recovery).
    assert _faithful(CHECK(PTX_VECTOR))


def test_scalar_and_uniform_vector_agree():
    # Both entries store the loaded value f1 (scalar vs 2-wide vector of the same f1) and read it back,
    # i.e. they compute the IDENTICAL value with the same structure. Merging them is CORRECT (they are
    # bitwise-equivalent), not an over-merge -- the coord-free structure is what a shared load resolves on.
    assert CHECK(PTX_SCALAR) == CHECK(PTX_VECTOR)


def test_mixed_structure_phase_fails_closed():
    # A phase holding writes of DIFFERENT structure (Leaf vs add) cannot be resolved address-blind:
    # the load could read either -> fail closed (conservative fingerprint), the soundness guard.
    d = CHECK(PTX_MIXED)
    assert d and "fwd-incomplete" in str(d[0])


def test_load_with_no_store_fails_closed():
    d = CHECK(PTX_NOSTORE)
    assert d and "fwd-incomplete" in str(d[0])


# --------------------------------------------------------------------------- #
# Cross-thread min/max recovery (full-fan-in follow-up): a COMPLETE order-invariant
# (min/max) reduction is bit-identical for ANY layout — a cross-warp read resolves to one
# representative partial so the reconstructed column SET is num_warps-VARYING, but the TRUE
# reduced multiset is config-invariant (every autotuner knob preserves it), so the collapse
# drops the varying residual and keys a cross-thread min/max on (op, leaf_sig) alone. This
# recovers col_max across num_warps. Gated on a cross-thread op (butterfly / exchange) so a
# pure within-thread max keeps its column key (conservative).
# --------------------------------------------------------------------------- #
def _minmax_kern(off2, mode="shfl"):
    """A [base + tid*4] load + a second load at ``+off2`` -> within-thread max -> a cross-thread
    reduce (``mode``: ``shfl`` butterfly / ``redux`` hardware warp-reduce / ``within`` none)."""
    body = _HEAD + """
.visible .entry k(
  .param .u64 pin, .param .u64 pout
)
.reqntid 128
{
  .reg .pred %p<2>;
  .reg .f32 %f<8>;
  .reg .b32 %r<8>;
  .reg .b64 %rd<10>;
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  cvta.to.global.u64 %rd3, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd4, %r1, 4;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  add.s64 %rd6, %rd5, %OFF2%;
  ld.global.f32 %f2, [%rd6];
  max.f32 %f3, %f1, %f2;
""".replace("%OFF2%", str(off2))
    if mode == "redux":
        body += "  mov.b32 %r3, -1;\n  redux.sync.max.f32 %r4, %f3, %r3;\n  mov.b32 %f4, %r4;\n"
    elif mode == "shfl":
        body += "  shfl.sync.bfly.b32 %r5, %f3, 16, 31, -1;\n  max.f32 %f4, %f3, %r5;\n"
    else:  # within-thread only (no cross-thread reduce)
        body += "  mov.b32 %f4, %f3;\n"
    body += "  cvta.to.global.u64 %rd7, %rd2;\n  st.global.f32 [%rd7], %f4;\n  ret;\n}\n"
    return body


def test_cross_thread_minmax_is_extent_free():
    # A cross-thread (butterfly) max reconstructs faithfully AND collapses to the extent-free key
    # (`ITREE[max.f32;L[];coi]`) — verified on the collapsed tree sig (the descriptor is its hash).
    from pyptx.ir.nodes import Function
    from pyptx.parser import parse

    from bitequiv.ptx.builder import collapse_balanced, tree_sig
    from bitequiv.ptx.forward.interp import ForwardInterp
    from bitequiv.ptx_reduction import _ensure_header
    assert _faithful(CHECK(_minmax_kern(1024, "shfl")))
    mod = parse(_ensure_header(_minmax_kern(1024, "shfl")))
    entry = next(f for f in mod.directives if isinstance(f, Function) and f.is_entry)
    interp = ForwardInterp(entry)
    sig = tree_sig(collapse_balanced(interp.run()[0]))
    assert "ITREE[max.f32;L[];coi]" in sig


def test_cross_thread_minmax_merges_different_column_layouts():
    # Two cross-thread maxes over DIFFERENT column sets (different second-load offset = a different
    # per-thread layout, as num_warps would produce) merge: the reduced multiset is the same, and
    # min/max is order/shape-invariant, so the num_warps-varying column residual is dropped.
    assert CHECK(_minmax_kern(1024, "shfl")) == CHECK(_minmax_kern(4096, "shfl"))


def test_redux_sync_max_modeled_like_butterfly():
    # `redux.sync.max.f32` (a hardware one-instruction warp max) is modeled as a cross-lane min/max
    # reduce, bit-identical to the shfl-butterfly form the compiler emits at other num_warps.
    d = CHECK(_minmax_kern(1024, "redux"))
    assert _faithful(d)
    assert CHECK(_minmax_kern(1024, "redux")) == CHECK(_minmax_kern(1024, "shfl"))


def test_within_thread_minmax_keeps_column_key():
    # A pure within-thread max (NO cross-thread op) is a partial / elementwise max, not a complete
    # reduction, so it keeps its column key: two different column layouts stay DISTINCT (conservative,
    # so the extent-drop can never merge two genuinely different partial maxes).
    assert CHECK(_minmax_kern(1024, "within")) != CHECK(_minmax_kern(4096, "within"))
