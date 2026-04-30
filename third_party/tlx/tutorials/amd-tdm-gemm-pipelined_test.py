"""
TDM-pipelined GEMM for AMD gfx1250.

This tutorial demonstrates the AMD-specific TLX TDM (Tensor Data Movement)
API on a pipelined matmul:

- ``tl.make_tensor_descriptor`` + ``tlx.async_tdm_load`` to issue an async
  copy from global memory directly into a user-provided LDS buffer.
- ``tlx.async_tdm_wait(N)`` to drain the ``tensorcnt`` hardware counter
  until at most ``N`` outstanding TDM ops remain.
- ``tlx.local_alloc`` *without* an explicit ``layout=``: the
  ``tlx-insert-require-layout`` + ``tlx-propagate-layout`` passes
  rewrite the alloc's encoding from the descriptor automatically, and
  pick the WMMA-tuned padded layout when the buffer feeds ``tl.dot``.

The kernel implements ``C = A @ B`` with a two-buffer software pipeline:
prefetch tile k+1 while consuming tile k. Each iteration issues two TDM
copies (A-tile and B-tile) and waits until at most two ops are
outstanding (the just-issued pair).
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

    NUM_BUFFERS: tl.constexpr = 2
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    b_buf = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    K_ITERS = tl.cdiv(K, BLOCK_K)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Prologue: kick off the first tile's loads (stage 0, slot 0).
    tlx.async_tdm_load(a_desc, tlx.local_view(a_buf, 0), [off_m, 0])
    tlx.async_tdm_load(b_desc, tlx.local_view(b_buf, 0), [0, off_n])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Steady state: at iter k, prefetch tile k+1 into the other slot,
    # wait for tile k, consume it.
    for k in tl.range(0, K_ITERS - 1):
        next_k = k + 1
        next_slot = next_k % NUM_BUFFERS
        tlx.async_tdm_load(a_desc, tlx.local_view(a_buf, next_slot), [off_m, next_k * BLOCK_K])
        tlx.async_tdm_load(b_desc, tlx.local_view(b_buf, next_slot), [next_k * BLOCK_K, off_n])

        # Drain everything older than the just-issued pair.
        tlx.async_tdm_wait(2)

        cur_slot = k % NUM_BUFFERS
        a_reg = tlx.local_load(tlx.local_view(a_buf, cur_slot))
        b_reg = tlx.local_load(tlx.local_view(b_buf, cur_slot))
        acc = tl.dot(a_reg, b_reg, acc)

    # Epilogue: drain everything and consume the last tile.
    tlx.async_tdm_wait(0)
    last_slot = (K_ITERS - 1) % NUM_BUFFERS
    a_reg = tlx.local_load(tlx.local_view(a_buf, last_slot))
    b_reg = tlx.local_load(tlx.local_view(b_buf, last_slot))
    acc = tl.dot(a_reg, b_reg, acc)

    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


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
    assert ("amdg.async_tdm_wait" in ttgir) or ("amdg.async_tdm_intrinsic_wait" in ttgir)
    # Auto-propagation should pick up the WMMA-tuned padded encoding.
    assert "ttg.padded_shared" in ttgir, "expected propagated padded encoding, got:\n" + ttgir

    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn


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
