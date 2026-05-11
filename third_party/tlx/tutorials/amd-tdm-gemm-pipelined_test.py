"""
TDM-pipelined GEMM for AMD gfx1250.

This tutorial demonstrates the AMD-specific TLX TDM (Tensor Data Movement)
API on a pipelined matmul:

- ``tl.make_tensor_descriptor`` + ``tlx.async_amd_descriptor_load`` to
  issue an async copy from global memory directly into a user-provided
  LDS buffer.
- ``tlx.amd_descriptor_prefetch_tensor`` for a look-ahead L2 prefetch hint,
  with an ``i1`` pred to gate against the trailing tail.
- ``tlx.async_amd_descriptor_store`` to copy the accumulator tile back
  to global memory via TDM (mirrors ``async_amd_descriptor_load``; no
  token, count-based wait only because the store op produces no SSA
  result).
- ``tlx.async_amd_descriptor_wait(N)`` to drain the ``tensorcnt``
  hardware counter until at most ``N`` outstanding TDM ops remain.
  Counts both load and store directions.
- ``tlx.local_alloc`` *without* an explicit ``layout=``: the
  ``tlx-insert-require-layout`` + ``tlx-propagate-layout`` passes
  rewrite the alloc's encoding from the descriptor automatically.
  A/B allocs get the WMMA-tuned padded layout (they feed ``tl.dot``),
  while the C alloc gets the descriptor-shape default (the TDM store
  hardware verifier requires ``padInterval == innermost block dim``).

The kernel implements ``C = A @ B`` with a two-buffer software pipeline:
prefetch tile k+1 while consuming tile k. Each iteration issues two TDM
copies (A-tile and B-tile), a pair of L2 prefetches for tile k+2, and
waits until at most two TDM ops are outstanding.
"""
import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_gfx1250_available():
    try:
        target = triton.runtime.driver.active.get_current_target()
        return target.arch == "gfx1250"
    except Exception:
        return False


@triton.jit
def matmul_tdm_pipelined_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = A @ B with TDM async loads and a 2-buffer software pipeline.

    Both A and B are assumed row-major contiguous so the inner stride
    is the constexpr 1 that ``tl.make_tensor_descriptor`` expects.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, tl.constexpr(1)],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, tl.constexpr(1)],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, tl.constexpr(1)],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    NUM_BUFFERS: tl.constexpr = 2
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    b_buf = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)
    # One-shot C buffer: filled from the accumulator and TDM-stored to
    # global. Default layout — propagation picks the descriptor-shape
    # default (`[32:+8]` for fp16) which is what the TDM store verifier
    # requires.
    c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_ptr), 1)

    K_ITERS = tl.cdiv(K, BLOCK_K)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Prologue: kick off the first tile's loads (stage 0, slot 0) and
    # prefetch tile 1 into L2 while tile 0 is in flight. The pred
    # guards the K==BLOCK_K case where the prefetched offset would be
    # out of bounds (hardware drops the prefetch silently anyway, but
    # being explicit avoids relying on that fallback).
    tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, 0), [off_m, 0])
    tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, 0), [0, off_n])
    prefetch_pred = BLOCK_K < K
    tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, BLOCK_K], pred=prefetch_pred)
    tlx.amd_descriptor_prefetch_tensor(b_desc, [BLOCK_K, off_n], pred=prefetch_pred)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Steady state: at iter k, issue loads for tile k+1, prefetch tile
    # k+2 into L2, wait for tile k, consume it.
    for k in tl.range(0, K_ITERS - 1):
        next_k = k + 1
        next_slot = next_k % NUM_BUFFERS
        tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, next_slot), [off_m, next_k * BLOCK_K])
        tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, next_slot), [next_k * BLOCK_K, off_n])

        # Look-ahead prefetch for tile k+2 (in bounds when k+2 < K_ITERS).
        prefetch_k = next_k + 1
        prefetch_pred = prefetch_k < K_ITERS
        tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, prefetch_k * BLOCK_K], pred=prefetch_pred)
        tlx.amd_descriptor_prefetch_tensor(b_desc, [prefetch_k * BLOCK_K, off_n], pred=prefetch_pred)

        # Drain everything older than the just-issued pair.
        tlx.async_amd_descriptor_wait(2)

        cur_slot = k % NUM_BUFFERS
        a_reg = tlx.local_load(tlx.local_view(a_buf, cur_slot))
        b_reg = tlx.local_load(tlx.local_view(b_buf, cur_slot))
        acc = tl.dot(a_reg, b_reg, acc)

    # Epilogue: drain everything and consume the last tile.
    tlx.async_amd_descriptor_wait(0)
    last_slot = (K_ITERS - 1) % NUM_BUFFERS
    a_reg = tlx.local_load(tlx.local_view(a_buf, last_slot))
    b_reg = tlx.local_load(tlx.local_view(b_buf, last_slot))
    acc = tl.dot(a_reg, b_reg, acc)

    # TDM-store the result tile: dot output -> LDS via local_store,
    # then LDS -> global via async_amd_descriptor_store. The wait drains
    # the outstanding store before the kernel exits.
    c = acc.to(tlx.dtype_of(c_ptr))
    c_view = tlx.local_view(c_buf, 0)
    tlx.local_store(c_view, c)
    tlx.async_amd_descriptor_store(c_desc, c_view, [off_m, off_n])
    tlx.async_amd_descriptor_wait(0)


def matmul_tdm_pipelined(a: torch.Tensor, b: torch.Tensor, BLOCK_M: int = 128, BLOCK_N: int = 128,
                         BLOCK_K: int = 32) -> torch.Tensor:
    assert a.is_contiguous() and b.is_contiguous(), "A and B must be contiguous"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, f"K mismatch: A={a.shape}, B={b.shape}"
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0, \
        "M, N, K must be multiples of their block sizes for this tutorial"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_tdm_pipelined_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


def test_matmul_tdm_pipelined_compiles_gfx1250():
    """Compile-only check: runs everywhere, validates the kernel still
    lowers cleanly to TDM intrinsics + a propagated padded encoding."""
    from triton.compiler.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget

    src = ASTSource(
        fn=matmul_tdm_pipelined_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
        },
        constexprs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))

    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir, ("expected TDM store of C in TTGIR, got:\n" + ttgir)
    assert "amdg.tdm_prefetch" in ttgir, ("expected TDM prefetch in TTGIR, got:\n" + ttgir)
    assert ("amdg.async_tdm_wait" in ttgir) or ("amdg.async_tdm_intrinsic_wait" in ttgir)
    # Auto-propagation gives:
    #   A: [128, 32] fp16 opIdx=0 -> WMMA-tuned `[128:+8]`
    #   B: [32, 128] fp16 opIdx=1 transposed -> WMMA-tuned `[128:+16]`
    #   C: [128, 128] fp16 -> default `[128:+8]` (innermost = 128)
    # So three distinct encoding strings should be present.
    assert "ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 32]}" in ttgir, (
        "expected WMMA-tuned encoding for A, got:\n" + ttgir)
    assert "ttg.padded_shared<[128:+16] {order = [1, 0], shape = [32, 128]}" in ttgir, (
        "expected WMMA-tuned encoding for B, got:\n" + ttgir)
    assert "ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 128]}" in ttgir, (
        "expected default encoding for C, got:\n" + ttgir)

    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "tensor_store_from_lds" in amdgcn or "tensor.store.from.lds" in amdgcn, (
        "expected tensor_store_from_lds intrinsic in AMDGCN, got:\n" + amdgcn)


@pytest.mark.skipif(not is_gfx1250_available(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("M,N,K", [(128, 128, 64), (256, 256, 128), (512, 512, 256)])
def test_matmul_tdm_pipelined_gfx1250(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    triton_out = matmul_tdm_pipelined(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
    torch_out = torch.matmul(a, b)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    if not is_gfx1250_available():
        raise SystemExit("Requires gfx1250 hardware")
    a = torch.randn((512, 256), device=DEVICE, dtype=torch.float16)
    b = torch.randn((256, 512), device=DEVICE, dtype=torch.float16)
    out = matmul_tdm_pipelined(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
    ref = torch.matmul(a, b)
    print(f"max abs diff: {(out - ref).abs().max().item():.4f}")
