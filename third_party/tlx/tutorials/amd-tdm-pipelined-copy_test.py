"""
TDM-pipelined elementwise copy/transform on AMD gfx1250.

This tutorial demonstrates the AMD-specific TLX TDM (Tensor Data Movement)
API for descriptor-based async loads on gfx1250:

- ``tl.make_tensor_descriptor`` + ``tlx.async_tdm_load`` to issue an async
  copy from global memory directly into a user-provided LDS buffer.
- ``tlx.async_tdm_wait(N)`` to drain the ``tensorcnt`` hardware counter
  until at most ``N`` outstanding TDM ops remain.
- ``tlx.local_alloc`` *without* an explicit ``layout=``: the
  ``tlx-insert-require-layout`` + ``tlx-propagate-layout`` passes
  rewrite the alloc's encoding from the descriptor automatically.

The kernel applies ``y = scale * x`` over a 2-D row-major tensor with a
two-buffer software pipeline: prefetch tile k+1 while consuming tile k.
This is a deliberately minimal example — feeding the loaded tile into
``tl.dot`` requires a WMMA-aware padded encoding that
``buildDefaultTDMDescriptorEncoding`` does not yet produce, so a TDM-fed
GEMM tutorial is left for a follow-up. The *load + wait + consume*
pattern shown here is the same one a future GEMM would use.
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
def scale_tdm_pipelined_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """y = scale * x over a row-major (M, N) tile grid, TDM-pipelined.

    Each program processes one row of tiles along the N axis: it issues
    successive TDM loads for tiles ``(pid_m, k)`` for ``k = 0..K-1``,
    multiplies by ``scale``, and stores back. With NUM_BUFFERS=2 the
    next tile is prefetched while the current one is being consumed.
    """
    pid_m = tl.program_id(0)
    K_ITERS = tl.cdiv(N, BLOCK_N)

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, N],
        strides=[N, tl.constexpr(1)],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    NUM_BUFFERS: tl.constexpr = 2
    # No explicit `layout=`: tlx-insert-require-layout / tlx-propagate-layout
    # rewrite the alloc's encoding to the descriptor-compatible padded
    # encoding automatically.
    x_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(x_ptr), NUM_BUFFERS)

    off_m = pid_m * BLOCK_M

    # Prologue: prefetch tile 0 into slot 0.
    tlx.async_tdm_load(x_desc, tlx.local_view(x_buf, 0), [off_m, 0])

    # Steady state: at iter k, prefetch tile k+1 into the other slot,
    # wait for tile k, consume it.
    for k in tl.range(0, K_ITERS - 1):
        next_k = k + 1
        next_slot = next_k % NUM_BUFFERS
        tlx.async_tdm_load(x_desc, tlx.local_view(x_buf, next_slot), [off_m, next_k * BLOCK_N])

        # Drain everything older than the just-issued load.
        tlx.async_tdm_wait(1)

        cur_slot = k % NUM_BUFFERS
        tile = tlx.local_load(tlx.local_view(x_buf, cur_slot))
        tile = tile * scale.to(tile.dtype)

        offs_m = off_m + tl.arange(0, BLOCK_M)
        offs_n = k * BLOCK_N + tl.arange(0, BLOCK_N)
        out_ptrs = y_ptr + N * offs_m[:, None] + offs_n[None, :]
        tl.store(out_ptrs, tile, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    # Epilogue: drain the remaining load and consume tile K-1.
    tlx.async_tdm_wait(0)
    last_k = K_ITERS - 1
    last_slot = last_k % NUM_BUFFERS
    tile = tlx.local_load(tlx.local_view(x_buf, last_slot))
    tile = tile * scale.to(tile.dtype)

    offs_m = off_m + tl.arange(0, BLOCK_M)
    offs_n = last_k * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = y_ptr + N * offs_m[:, None] + offs_n[None, :]
    tl.store(out_ptrs, tile, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def scale_tdm_pipelined(x: torch.Tensor, scale: float, BLOCK_M: int = 16, BLOCK_N: int = 32) -> torch.Tensor:
    assert x.is_contiguous(), "x must be contiguous"
    M, N = x.shape
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0, "M, N must be multiples of their block sizes"

    y = torch.empty_like(x)
    grid = (triton.cdiv(M, BLOCK_M), )
    scale_tdm_pipelined_kernel[grid](
        x,
        y,
        M,
        N,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return y


def test_scale_tdm_pipelined_compiles_gfx1250():
    """Compile-only check: runs everywhere, validates the kernel still
    lowers cleanly to TDM intrinsics + a propagated padded encoding."""
    from triton.compiler.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget

    src = ASTSource(
        fn=scale_tdm_pipelined_kernel,
        signature={
            "x_ptr": "*fp16",
            "y_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "scale": "fp32",
        },
        constexprs={"BLOCK_M": 16, "BLOCK_N": 32},
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))

    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert ("amdg.async_tdm_wait" in ttgir) or ("amdg.async_tdm_intrinsic_wait" in ttgir)
    # Auto-propagation should pick up the padded encoding from the descriptor.
    assert "ttg.padded_shared" in ttgir, "expected propagated padded encoding, got:\n" + ttgir

    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn


@pytest.mark.skipif(not is_gfx1250_available(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("M,N", [(32, 64), (128, 128), (256, 512)])
def test_scale_tdm_pipelined_gfx1250(M, N):
    torch.manual_seed(0)
    x = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    scale = 0.5

    triton_out = scale_tdm_pipelined(x, scale, BLOCK_M=16, BLOCK_N=32)
    torch_out = scale * x
    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    if not is_gfx1250_available():
        raise SystemExit("Requires gfx1250 hardware")
    x = torch.randn((128, 256), device=DEVICE, dtype=torch.float16)
    out = scale_tdm_pipelined(x, 0.5, BLOCK_M=16, BLOCK_N=32)
    ref = 0.5 * x
    print(f"max abs diff: {(out - ref).abs().max().item():.4f}")
