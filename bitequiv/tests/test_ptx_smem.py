"""Forward SmemModel (phase + address match) soundness + recovery, on hand-crafted PTX.

The model records each ``st.shared`` element (scalar OR vector) into the current write phase with the
SLOT IMAGE of its address, closes the phase at ``bar.sync``, and resolves a ``ld.shared`` /
``ldmatrix`` against the most-recently closed phase in two layers: (1) match the load's slot image
against the stores' and take the value of the store(s) that actually wrote it — a PROVEN relocation;
(2) if an address is not provably affine, the older address-blind rule (return a representative iff
every write in the phase has the same coord-free structure), else FAIL CLOSED.

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

# MIXED-structure phase: one write is a Leaf (f1), the other a combine add(f1,f2). Their coord-free
# sigs differ, so an address-BLIND model cannot tell which the load reads. Both slots are provably
# affine here, so the address match picks the store that actually wrote the loaded slot (`bufA+0` =
# the Leaf) and the reconstruction stays faithful.
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

# The same MIXED phase, but the load address is derived from a value LOADED FROM MEMORY, so it is not
# provably affine: the address match cannot run and the address-blind fallback sees two structures ->
# FAIL CLOSED (the layered floor).
PTX_MIXED_OPAQUE_ADDR = _HEAD + """
.visible .entry k(.param .u64 pin, .param .u64 pout) {
  .reg .f32 %f<8>;
  .reg .b32 %r<8>;
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
  ld.global.u32 %r6, [%rd5];
  add.s32 %r7, %r2, %r6;
  ld.shared.f32 %f4, [%r7];
  cvta.to.global.u64 %rd7, %rd2;
  st.global.f32 [%rd7], %f4;
  ret;
}
"""

# A load of a slot NO store in the phase wrote (the store lands at bufA+0, the load reads bufA+64).
# Address-blind the model happily returns the one write it has; with the address it is provably a
# DIFFERENT slot, so the value read was never captured -> fail closed.
PTX_UNWRITTEN_SLOT = _HEAD + """
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
  add.s32 %r3, %r2, 64;
  ld.shared.f32 %f2, [%r3];
  cvta.to.global.u64 %rd6, %rd2;
  st.global.f32 [%rd6], %f2;
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


def test_mixed_structure_phase_resolved_by_address():
    # A phase holding writes of DIFFERENT structure (Leaf vs add) is ambiguous address-blind, but both
    # slots are provably affine, so the load is matched to the store that wrote ITS slot (bufA+0, the
    # Leaf) and the reconstruction stays faithful — and equals the single-store scalar kernel, which
    # computes exactly the same thing.
    assert _faithful(CHECK(PTX_MIXED))
    assert CHECK(PTX_MIXED) == CHECK(PTX_SCALAR)


def test_mixed_structure_unprovable_address_fails_closed():
    # Same mixed phase, but the load address comes from a value read out of memory -> not provably
    # affine -> the address match cannot run and the address-blind fallback fails closed.
    d = CHECK(PTX_MIXED_OPAQUE_ADDR)
    assert d and "fwd-incomplete" in str(d[0])


def test_load_of_unwritten_slot_fails_closed():
    # The loaded slot is provably NOT one this phase wrote, so the value is unmodeled -> fail closed.
    d = CHECK(PTX_UNWRITTEN_SLOT)
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

    from bitequiv.core.canonicalize import collapse_balanced, tree_sig
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


# --------------------------------------------------------------------------- #
# Cross-warp num_warps recovery. A matched shared exchange is a RELOCATION: it moves one partial
# from one thread to another and combines nothing, so it adds no reduction height. Two configs that
# reduce the same element count in the same balanced count-up order therefore agree even though one
# splits the work over two warps (and needs the exchange) and the other over one (and does not).
# --------------------------------------------------------------------------- #
def _bfly(offs, first):
    """A butterfly chain of ``add.f32(p, shfl.bfly(p, off))`` over ``offs``, leaf -> root."""
    out, cur = [], first
    for i, o in enumerate(offs):
        out.append(f"  shfl.sync.bfly.b32 %f{10 + i}, {cur}, {o}, 31, -1;")
        out.append(f"  add.f32 %f{30 + i}, {cur}, %f{10 + i};")
        cur = f"%f{30 + i}"
    return "\n".join(out), cur


def _one_warp(offs):
    """32 threads x 2 elements: a within-thread add then a 32-lane butterfly. 64 elements, no
    shared memory at all."""
    body, cur = _bfly(offs, "%f3")
    return _HEAD + f"""
.visible .entry k(.param .u64 pin, .param .u64 pout) .reqntid 32, 1, 1
{{
  .reg .pred %p<4>;
  .reg .f32 %f<64>;
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  .shared .align 4 .b8 buf[512];
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  cvta.to.global.u64 %rd3, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd4, %r1, 4;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  add.s64 %rd6, %rd5, 128;
  ld.global.f32 %f2, [%rd6];
  add.f32 %f3, %f1, %f2;
{body}
  cvta.to.global.u64 %rd7, %rd2;
  st.global.f32 [%rd7], {cur};
  ret;
}}
"""


def _two_warps(offs):
    """64 threads x 1 element: a 32-lane butterfly, then a cross-warp exchange through shared memory
    (warp leader writes slot = warp id, lane `l < 2` reads slot = l) and one more butterfly step.
    The SAME 64 elements in the same balanced count-up order as :func:`_one_warp`."""
    body, cur = _bfly(offs, "%f1")
    return _HEAD + f"""
.visible .entry k(.param .u64 pin, .param .u64 pout) .reqntid 64, 1, 1
{{
  .reg .pred %p<4>;
  .reg .f32 %f<64>;
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  .shared .align 4 .b8 buf[512];
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  cvta.to.global.u64 %rd3, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd4, %r1, 4;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
{body}
  mov.u32 %r2, buf;
  shr.u32 %r3, %r1, 5;
  shl.b32 %r4, %r3, 2;
  add.s32 %r5, %r2, %r4;
  and.b32 %r6, %r1, 31;
  setp.eq.b32 %p1, %r6, 0;
  @%p1 st.shared.f32 [%r5], {cur};
  bar.sync 0;
  shl.b32 %r7, %r1, 2;
  add.s32 %r8, %r2, %r7;
  setp.lt.u32 %p2, %r1, 2;
  @%p2 ld.shared.f32 %f50, [%r8];
  shfl.sync.bfly.b32 %f51, %f50, 1, 31, -1;
  add.f32 %f52, %f50, %f51;
  cvta.to.global.u64 %rd9, %rd2;
  st.global.f32 [%rd9], %f52;
  ret;
}}
"""


_UP = [1, 2, 4, 8, 16]     # count-up (inner_tree): the canonical balanced order
_DOWN = [16, 8, 4, 2, 1]   # count-down (unordered): pairs lanes differently -> different bits


def test_cross_warp_exchange_recovers_num_warps():
    # The slot <-> warp map is provable (store slot = warp id, load slot = lane), so the exchange is a
    # matched relocation and carries no height: one warp with a two-element within-thread fold and two
    # warps with a shared exchange collapse to the SAME `ITREE[add.f32;L[];h6;s('1',)]`.
    assert _faithful(CHECK(_one_warp(_UP)))
    assert _faithful(CHECK(_two_warps(_UP)))
    assert CHECK(_one_warp(_UP)) == CHECK(_two_warps(_UP))


def test_cross_warp_exchange_splits_on_reduction_order():
    # Same element count and the same exchange, but a count-DOWN butterfly pairs the lanes
    # differently, so the bits differ and the descriptors must stay apart -- in both directions of
    # the comparison, and against the one-warp count-down form too.
    assert CHECK(_two_warps(_UP)) != CHECK(_two_warps(_DOWN))
    assert CHECK(_one_warp(_UP)) != CHECK(_one_warp(_DOWN))
    assert CHECK(_two_warps(_DOWN)) != CHECK(_one_warp(_DOWN))
