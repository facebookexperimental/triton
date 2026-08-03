"""SmemModel soundness: a VECTOR ``st.shared`` must FAIL CLOSED, not let a later scalar ``ld.shared``
resolve to a stale scalar store (the D112727538 reviewer / DCR over-merge concern).

The bug is latent — no natural eval kernel reconstructs faithfully AND has a scalar-load-after-vector-
store, which is why it could not be reproduced at runtime. So the demo is hand-crafted PTX (full control
of the pattern): a scalar-only exchange reconstructs to a real tree hash; adding a vector ``st.shared``
to a second buffer + a scalar ``ld.shared`` from it makes the OLD code resolve the load to the stale
scalar store (address-blind) while ``faithful`` stays True — so two kernels that read DIFFERENT buffers
collapse to ONE descriptor (over-merge). The fix fails closed on the vector store. CPU-only."""
from bitequiv.ptx.forward.interp import forward_module_descriptor as CHECK

_HEAD = ".version 8.5\n.target sm_90a\n.address_size 64\n"

# Scalar shared exchange only: st.shared A -> ld.shared A -> reconstructs faithfully (real tree hash).
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

# Same, plus a VECTOR st.shared to a SECOND buffer B, then a scalar ld.shared from B. WITHOUT the fix
# the vector store is skipped, the scalar load resolves to the stale scalar store A (address-blind),
# faithful stays True -> identical descriptor to PTX_SCALAR though it reads B = OVER-MERGE. WITH the fix
# the vector store fails closed (fingerprint) -> distinct -> sound.
PTX_VECTOR = _HEAD + """
.visible .entry k(.param .u64 pin, .param .u64 pout) {
  .reg .f32 %f<4>;
  .reg .b32 %r<4>;
  .reg .b64 %rd<8>;
  .shared .align 4 .b8 bufA[128];
  .shared .align 8 .b8 bufB[128];
  ld.param.u64 %rd1, [pin];
  ld.param.u64 %rd2, [pout];
  cvta.to.global.u64 %rd3, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd4, %r1, 4;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  mov.u32 %r2, bufA;
  st.shared.f32 [%r2], %f1;
  mov.u32 %r3, bufB;
  st.shared.v2.f32 [%r3], {%f1, %f1};
  bar.sync 0;
  ld.shared.f32 %f2, [%r3];
  cvta.to.global.u64 %rd6, %rd2;
  st.global.f32 [%rd6], %f2;
  ret;
}
"""


def test_scalar_shared_exchange_reconstructs():
    d = CHECK(PTX_SCALAR)
    assert d and "fwd-incomplete" not in str(d[0])   # faithful tree hash (scalar exchange is modeled)


def test_vector_shared_store_fails_closed():
    d = CHECK(PTX_VECTOR)
    assert d and "fwd-incomplete" in str(d[0])        # the fix: vector st.shared -> conservative fingerprint


def test_vector_store_no_over_merge():
    # The two entries read DIFFERENT buffers (A vs B) -> genuinely different computations. The fix keeps
    # their descriptors distinct; WITHOUT it both resolve the scalar load to buffer A -> one descriptor
    # (the reviewer's over-merge). This assertion FAILS on the pre-fix code.
    assert CHECK(PTX_SCALAR) != CHECK(PTX_VECTOR)
