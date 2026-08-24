"""auto-TMA promotion of multi-dim strided loads, and composition with autoWS.

Two things are validated:

1. ``test_auto_tma_2d_strided_numerics`` (non-WS, end-to-end): a 2D strided
   masked block load `x[:, :]` (row-major, contiguous innermost) is promoted by
   PromoteLoadToTMA to a host-side TMA descriptor and runs correctly on GPU.
   This exercises the decomposePointer extensions (addptr chain walk,
   stride-after-expand_dims, dense-zero OOB fill, broadcast-wrapped mask cmp).

2. ``test_ws_auto_tma_data_partition_numerics`` (end-to-end): a warp-specialized
   + data-partitioned (dp=2) matmul whose plain masked loads are auto-TMA
   promoted. Asserts the loads become real host-side TMA copies
   (async_tma_copy_global_to_local, no device-side make_tensor_descriptor), that
   WS engaged, that the recipe block_shape reflects the WS-halved descriptor arg
   type (the recipe-from-final-type fix), and that numerics match the reference.

Requires sm_90+: auto-TMA gating is sm_90+, and Meta WS data partitioning runs
when ``use_meta_ws`` is set. Validated on Hopper (sm_90) and Blackwell (sm_100).

Perf (opt-in, ``TRITON_RUN_PERF=1``):

3. ``test_ws_auto_tma_perf``: direct auto-TMA off-vs-on TFLOP/s on the WS+dp GEMM.
4. ``test_ws_auto_tma_autotuner_picks_faster``: exposes ``auto_tma`` as an
   autotune axis (two Configs identical except ``auto_tma``) and asserts the
   autotuner only selects ``auto_tma=True`` when it is actually faster than the
   cp.async baseline, cross-checked against an independent do_bench of each
   variant.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_cuda
from triton.testing import do_bench
import os


def _is_sm90plus() -> bool:
    if not is_cuda():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9  # Hopper (sm_90) and Blackwell (sm_100)


# ---------------------------------------------------------------------------
# 1. Non-WS 2D strided promotion + numerics
# ---------------------------------------------------------------------------
@triton.jit
def _scale_2d_strided(x_ptr, out_ptr, M, N, stride_xm, stride_om, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Row-major x: innermost (offs_n) is contiguous -> auto-TMA eligible.
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :], mask=mask, other=0.0)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], x * 2.0, mask=mask)


@pytest.mark.skipif(not _is_sm90plus(), reason="auto-TMA promotion requires sm_90+")
def test_auto_tma_2d_strided_numerics():
    M, N = 512, 384
    BLOCK_M, BLOCK_N = 64, 64
    dtype = torch.float16
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=dtype, device="cuda")
    out = torch.empty((M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.auto_tma = True
        kernel = _scale_2d_strided[grid](x, out, M, N, x.stride(0), out.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    assert "tt.descriptor_load" in kernel.asm["ttir"], \
        "expected the 2D strided masked load to be auto-TMA promoted"
    torch.testing.assert_close(out, x * 2.0)


@triton.jit
def _gemm_nows(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bn, stride_cm, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(tl.cdiv(K, BLOCK_K)):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc = tl.dot(a, b.T, acc)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :], acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@pytest.mark.skipif(not _is_sm90plus(), reason="auto-TMA promotion requires sm_90+")
def test_auto_tma_gemm_nows():
    """Isolates the auto-TMA DOT-operand path WITHOUT warp specialization."""
    M, N, K = 256, 256, 256
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    dtype = torch.float16
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cuda")
    B = torch.randn((N, K), dtype=dtype, device="cuda")
    C = torch.empty((M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.auto_tma = True
        kernel = _gemm_nows[grid](A, B, C, M, N, K, A.stride(0), B.stride(0), C.stride(0), BLOCK_M=BLOCK_M,
                                  BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    assert "tt.descriptor_load" in kernel.asm["ttir"], "expected auto-TMA promotion"
    ref = (A.to(torch.float32) @ B.T.to(torch.float32)).to(dtype)
    torch.testing.assert_close(ref, C, atol=0.05, rtol=0.05)


@pytest.mark.skipif(not _is_sm90plus(), reason="auto-TMA promotion requires sm_90+")
def test_auto_tma_gemm_nows_device(monkeypatch):
    """Phase A: addptr loads promoted to a DEVICE-built tt.make_tensor_descriptor
    (TRITON_AUTO_TMA_DEVICE=1), not the host-recipe path. Verifies the pass
    synthesizes make_tensor_descriptor from the plain pointer + mask analysis.
    Uses a dedicated kernel so the JIT cache does not collide with the host-mode
    _gemm_nows test (the auto-TMA-device env is not part of the cache key)."""

    @triton.jit
    def _gemm_dev(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bn, stride_cm, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(tl.cdiv(K, BLOCK_K)):
            offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :],
                        mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
            acc = tl.dot(a, b.T, acc)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :], acc.to(tl.float16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    monkeypatch.setenv("TRITON_AUTO_TMA", "1")
    monkeypatch.setenv("TRITON_AUTO_TMA_DEVICE", "1")
    M, N, K = 256, 256, 256
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    dtype = torch.float16
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cuda")
    B = torch.randn((N, K), dtype=dtype, device="cuda")
    C = torch.empty((M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel = _gemm_dev[grid](A, B, C, M, N, K, A.stride(0), B.stride(0), C.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                             BLOCK_K=BLOCK_K)

    ttir = kernel.asm["ttir"]
    assert "tt.make_tensor_descriptor" in ttir, \
        "device mode should synthesize an in-kernel make_tensor_descriptor"
    assert "tt.descriptor_load" in ttir, "expected descriptor_load"
    ref = (A.to(torch.float32) @ B.T.to(torch.float32)).to(dtype)
    torch.testing.assert_close(ref, C, atol=0.05, rtol=0.05)


@triton.jit
def _batched_scale(x_ptr, out_ptr, B, M, N, stride_b, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Per-program base shift folded into the (scalar) base pointer.
    xb = x_ptr + pid_b * stride_b
    ob = out_ptr + pid_b * stride_b
    x = tl.load(xb + offs_m[:, None] * stride_m + offs_n[None, :], mask=mask, other=0.0)
    tl.store(ob + offs_m[:, None] * stride_m + offs_n[None, :], x * 2.0, mask=mask)


@pytest.mark.skipif(not _is_sm90plus(), reason="auto-TMA promotion requires sm_90+")
def test_auto_tma_batched_device(monkeypatch):
    """per-program base (pid_b*stride_b) must fold into the DEVICE descriptor base
    (make_tensor_descriptor(addptr(x, off), ...)), giving a per-batch view."""
    monkeypatch.setenv("TRITON_AUTO_TMA", "1")
    monkeypatch.setenv("TRITON_AUTO_TMA_DEVICE", "1")
    B, M, N = 4, 128, 128
    BLOCK_M, BLOCK_N = 64, 64
    dtype = torch.float16
    torch.manual_seed(0)
    x = torch.randn((B, M, N), dtype=dtype, device="cuda")
    out = torch.empty((B, M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel = _batched_scale[grid](x, out, B, M, N, x.stride(0), x.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    ttir = kernel.asm["ttir"]
    assert "tt.make_tensor_descriptor" in ttir, \
        "per-program base should still produce a device descriptor"
    torch.testing.assert_close(out, x * 2.0)


# ---------------------------------------------------------------------------
# 2. WS + data-partition + auto-TMA (end-to-end numerics)
# ---------------------------------------------------------------------------
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _ws_auto_tma_matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
):
    """C = A @ B, contracting over K. A is [M,K] and B is [K,N], both row-major
    (contiguous in the innermost dim -- A over K, B over N), so the plain masked
    A/B loads are auto-TMA eligible. NOTE: the test allocates B as [N,K] and uses
    square M=N=K, where a [K,N] view aliases it; as written this kernel is only
    correct for square shapes."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True,
                            data_partition_factor=DATA_PARTITION_FACTOR):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + offs_k[:, None] * stride_bn + offs_n[None, :],
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
        c = acc.to(tl.float16)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :], c,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@pytest.mark.skipif(not _is_sm90plus(), reason="WS + auto-TMA requires sm_90+")
def test_ws_auto_tma_data_partition_numerics():
    M, N, K = 512, 512, 512
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 64  # dp=2 needs BLOCK_M=256 (256/2=128) so Blackwell tcgen05.mma M=128 is satisfied per partition
    GROUP_SIZE_M = 8
    DATA_PARTITION_FACTOR = 2
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dtype = torch.float16

    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cuda")
    B = torch.randn((N, K), dtype=dtype, device="cuda")
    C = torch.empty((M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.auto_tma = True
        grid = lambda meta: (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
        kernel = _ws_auto_tma_matmul[grid](
            A, B, C, M, N, K, A.stride(0), B.stride(0), C.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M, NUM_SMS=NUM_SMS, DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR, num_warps=4,
            num_stages=3, maxRegAutoWS=208,  # required for data_partition_factor>1 on Hopper
        )

    ttir = kernel.asm["ttir"]
    ttgir = kernel.asm["ttgir"]
    recipes = list(getattr(kernel.metadata, "auto_tma_recipes", []) or [])
    # Plain tl.load was auto-promoted to a TMA descriptor_load...
    assert "tt.descriptor_load" in ttir, "expected auto-TMA promotion (plain load -> TMA)"
    # ...warp specialization engaged...
    assert "ttg.warp_specialize" in ttgir, "expected warp specialization"
    # ...and the promoted load lowered to a real hardware async TMA copy (not
    # silently demoted back to a plain global load).
    assert "ttng.async_tma_copy_global_to_local" in ttgir, \
        "expected a real async TMA copy in TTGIR (host-side auto-TMA under WS)"
    # Host-side build: the CUtensorMap comes from a recipe (no device-side
    # tt.make_tensor_descriptor) -- so no runtime global-scratch descriptor build.
    assert recipes, "expected at least one auto-TMA recipe (host-side CUtensorMap)"
    assert "tt.make_tensor_descriptor" not in ttgir, \
        "auto-TMA must be host-built (no device-side make_tensor_descriptor)"
    # The recipe block_shape must reflect the WS-halved descriptor arg type
    # (BLOCK_M 256 -> 128 under dp=2): the recipe-from-final-type fix.
    block0 = [(r.get("block_shape") or [None])[0] for r in recipes]
    assert BLOCK_M // DATA_PARTITION_FACTOR in block0, (
        f"expected a recipe block_shape[0] == {BLOCK_M // DATA_PARTITION_FACTOR} "
        f"(WS-halved); got {block0}")

    ref = (A.to(torch.float32) @ B.to(torch.float32)).to(dtype)
    torch.testing.assert_close(ref, C, atol=0.05, rtol=0.05)


# Self-contained kernel for the perf test, decoupled from _ws_auto_tma_matmul
# (which the Hopper WS-fix commit rewrites to a TMA-store form). Plain-load
# A@B, c_ptr store, warp_specialize + data partitioning -> auto-TMA-promotable.
@triton.jit
def _ws_autotma_perf_matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True,
                            data_partition_factor=DATA_PARTITION_FACTOR):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + offs_k[:, None] * stride_bn + offs_n[None, :],
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
        c = acc.to(tl.float16)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :], c,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@pytest.mark.skipif(
    not triton.knobs.nvidia.run_perf,
    reason="perf benchmark (large GEMMs via do_bench); set TRITON_RUN_PERF=1 to run",
)
@pytest.mark.parametrize("M, N, K", [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)])
@pytest.mark.skipif(not _is_sm90plus(), reason="WS + auto-TMA requires sm_90+")
def test_ws_auto_tma_perf(M, N, K):
    """Perf: autoWS + dp=2 GEMM (self-contained _ws_autotma_perf_matmul),
    auto-TMA off vs on (same kernel; only the loads change from cp.async to
    host-built TMA copies). Global knob OFF; the per-call `auto_tma` compile
    option drives promotion so both variants share everything else. Prints
    TFLOP/s for each and the on/off speedup."""
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 64
    GROUP_SIZE_M = 8
    DATA_PARTITION_FACTOR = 2
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dtype = torch.float16

    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cuda")
    B = torch.randn((N, K), dtype=dtype, device="cuda")
    C = torch.empty((M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = lambda meta: (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
    ref = (A.to(torch.float32) @ B.to(torch.float32)).to(dtype)
    flops = 2.0 * M * N * K

    results = {}
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.auto_tma = False
        for at in (False, True):

            def run():
                _ws_autotma_perf_matmul[grid](
                    A,
                    B,
                    C,
                    M,
                    N,
                    K,
                    A.stride(0),
                    B.stride(0),
                    C.stride(0),
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    GROUP_SIZE_M=GROUP_SIZE_M,
                    NUM_SMS=NUM_SMS,
                    DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
                    num_warps=4,
                    num_stages=2,
                    maxRegAutoWS=208,
                    auto_tma=at,
                )

            C.zero_()
            run()
            torch.testing.assert_close(ref, C, atol=0.05, rtol=0.05)
            ms = do_bench(run)
            tflops = flops / (ms * 1e-3) / 1e12
            results[at] = (ms, tflops)
            print(f"\n[ws-autotma-perf] auto_tma={int(at)}: {ms:.4f} ms  {tflops:.1f} TFLOP/s")

    off_ms = results[False][0]
    on_ms = results[True][0]
    print(f"[ws-autotma-perf] autoWS+dp2 {M}x{N}x{K}: "
          f"off={results[False][1]:.1f} on={results[True][1]:.1f} TFLOP/s  "
          f"speedup(on/off)={off_ms / on_ms:.3f}x\n")


# ---------------------------------------------------------------------------
# 3. Autotuner-driven auto-TMA selection (perf study)
# ---------------------------------------------------------------------------
# `auto_tma` is a per-Config autotune axis (Config field in autotuner.py;
# make_ttir gates PromoteLoadToTMA on `opt.auto_tma or knobs.nvidia.auto_tma`).
# This study wraps the WS+dp GEMM in @triton.autotune over two Configs that are
# identical except `auto_tma`, and checks the autotuner selects auto_tma=True
# only when the promoted kernel is actually faster than the cp.async baseline.
# The pick is cross-checked against an INDEPENDENT do_bench of each variant (not
# just the autotuner's own per-config timing), so a corrupted or cross-config
# leaked measurement would be caught. Parametrized across shapes to exercise
# both outcomes (True when TMA overlap wins; False on shapes where it doesn't).
_AUTO_TMA_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2, auto_tma=False),
    triton.Config({}, num_warps=4, num_stages=2, auto_tma=True),
]


def _bench_autotma_variant(grid, A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_SMS, DP, auto_tma):
    """do_bench one variant of _ws_autotma_perf_matmul with auto_tma on/off; all
    other params fixed so the only difference is the load lowering."""

    def run():
        _ws_autotma_perf_matmul[grid](A, B, C, M, N, K, A.stride(0), B.stride(0), C.stride(0), BLOCK_M=BLOCK_M,
                                      BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M, NUM_SMS=NUM_SMS,
                                      DATA_PARTITION_FACTOR=DP, num_warps=4, num_stages=2, maxRegAutoWS=208,
                                      auto_tma=auto_tma)

    return do_bench(run)


@pytest.mark.skipif(
    not triton.knobs.nvidia.run_perf,
    reason="perf benchmark (autotuner over large GEMMs); set TRITON_RUN_PERF=1 to run",
)
@pytest.mark.parametrize("M, N, K", [(256, 256, 256), (4096, 4096, 4096), (8192, 8192, 8192)])
@pytest.mark.skipif(not _is_sm90plus(), reason="WS + auto-TMA requires sm_90+")
def test_ws_auto_tma_autotuner_picks_faster(M, N, K):
    """Perf study: the autotuner selects auto_tma=True only when it is actually
    faster. Two Configs (identical except auto_tma) are autotuned over a WS+dp
    GEMM; the chosen config is cross-checked against an independent do_bench of
    each variant, with a tie band where either choice is legitimate."""
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 64
    GROUP_SIZE_M = 8
    DP = 2
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    dtype = torch.float16

    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cuda")
    B = torch.randn((N, K), dtype=dtype, device="cuda")
    C = torch.empty((M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = lambda meta: (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
    ref = (A.to(torch.float32) @ B.to(torch.float32)).to(dtype)
    flops = 2.0 * M * N * K

    matmul = triton.autotune(configs=_AUTO_TMA_CONFIGS, key=["M", "N", "K"])(_ws_autotma_perf_matmul)

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.auto_tma = False

        # Ground truth: independently time each variant (fresh, outside the autotuner).
        off_ms = _bench_autotma_variant(grid, A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_SMS, DP,
                                        False)
        on_ms = _bench_autotma_variant(grid, A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_SMS, DP,
                                       True)

        # Autotuner: let it A/B the two configs and pick; the final launch runs the winner into C.
        C.zero_()
        matmul[grid](A, B, C, M, N, K, A.stride(0), B.stride(0), C.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                     BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M, NUM_SMS=NUM_SMS, DATA_PARTITION_FACTOR=DP,
                     maxRegAutoWS=208)

    torch.testing.assert_close(ref, C, atol=0.05, rtol=0.05)

    pick = bool(matmul.best_config.auto_tma)
    at_timings = {bool(cfg.auto_tma): t[0] for cfg, t in matmul.configs_timings.items()}
    faster = on_ms < off_ms

    def tflops(ms):
        return flops / (ms * 1e-3) / 1e12

    print(f"\n[autotuner-pick] autoWS+dp2 {M}x{N}x{K}:")
    print(f"  independent do_bench: off={off_ms:.4f} ms ({tflops(off_ms):.1f} TF)  "
          f"on={on_ms:.4f} ms ({tflops(on_ms):.1f} TF)  faster={'on' if faster else 'off'}")
    print(f"  autotuner per-config: off={at_timings.get(False, float('nan')):.4f} ms  "
          f"on={at_timings.get(True, float('nan')):.4f} ms")
    print(f"  autotuner picked auto_tma={int(pick)}  (independent timing => {int(faster)})\n")

    # The autotuner's pick must agree with which variant is actually faster,
    # except within a tie band where either choice is legitimate (no regression).
    TIE_TOL = 0.03
    rel = abs(off_ms - on_ms) / min(off_ms, on_ms)
    if rel > TIE_TOL:
        assert pick == faster, (
            f"autotuner picked auto_tma={int(pick)} but independent timing says faster="
            f"{'on' if faster else 'off'} (off={off_ms:.4f} on={on_ms:.4f} ms, {rel * 100:.1f}% apart)")


# ---------------------------------------------------------------------------
# 4. Autotuner-driven auto-TMA selection on a memory-bound kernel (the "False"
#    direction). A plain elementwise scale has no compute to overlap the copy
#    with (and store promotion is WS-gated, so it never fires here), so promoting
#    the load to a host-built TMA copy only adds per-launch CUtensorMap-build
#    overhead. The autotuner should therefore NOT select auto_tma=True.
# ---------------------------------------------------------------------------
_SCALE_AUTO_TMA_CONFIGS = [
    triton.Config({}, num_warps=4, auto_tma=False),
    triton.Config({}, num_warps=4, auto_tma=True),
]


def _bench_scale_variant(grid, x, out, M, N, BLOCK_M, BLOCK_N, auto_tma):

    def run():
        _scale_2d_strided[grid](x, out, M, N, x.stride(0), out.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
                                auto_tma=auto_tma)

    return do_bench(run)


@pytest.mark.skipif(
    not triton.knobs.nvidia.run_perf,
    reason="perf benchmark (autotuner over elementwise); set TRITON_RUN_PERF=1 to run",
)
@pytest.mark.parametrize("M, N", [(512, 512), (4096, 4096)])
@pytest.mark.skipif(not _is_sm90plus(), reason="auto-TMA requires sm_90+")
def test_scale_auto_tma_autotuner_picks_faster(M, N):
    """Perf study (False direction): on a memory-bound elementwise scale there is
    no compute to overlap the copy with, so auto-TMA is at best neutral and often
    slower (per-launch descriptor build). Verifies the autotuner does not pick
    auto_tma=True unless it is actually faster, cross-checked against an
    independent do_bench of each variant."""
    BLOCK_M, BLOCK_N = 64, 128
    dtype = torch.float16
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=dtype, device="cuda")
    out = torch.empty((M, N), dtype=dtype, device="cuda")

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = lambda meta: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    ref = (x * 2.0).to(dtype)

    scale = triton.autotune(configs=_SCALE_AUTO_TMA_CONFIGS, key=["M", "N"])(_scale_2d_strided)

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.auto_tma = False
        off_ms = _bench_scale_variant(grid, x, out, M, N, BLOCK_M, BLOCK_N, False)
        on_ms = _bench_scale_variant(grid, x, out, M, N, BLOCK_M, BLOCK_N, True)

        out.zero_()
        scale[grid](x, out, M, N, x.stride(0), out.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.testing.assert_close(ref, out, atol=0.05, rtol=0.05)

    pick = bool(scale.best_config.auto_tma)
    at_timings = {bool(cfg.auto_tma): t[0] for cfg, t in scale.configs_timings.items()}
    faster = on_ms < off_ms

    print(f"\n[autotuner-pick] mem-bound scale {M}x{N}:")
    print(f"  independent do_bench: off={off_ms:.5f} ms  on={on_ms:.5f} ms  faster={'on' if faster else 'off'}")
    print(f"  autotuner per-config: off={at_timings.get(False, float('nan')):.5f} ms  "
          f"on={at_timings.get(True, float('nan')):.5f} ms")
    print(f"  autotuner picked auto_tma={int(pick)}  (independent timing => {int(faster)})\n")

    TIE_TOL = 0.03
    rel = abs(off_ms - on_ms) / min(off_ms, on_ms)
    if rel > TIE_TOL:
        assert pick == faster, (
            f"autotuner picked auto_tma={int(pick)} but independent timing says faster="
            f"{'on' if faster else 'off'} (off={off_ms:.5f} on={on_ms:.5f} ms, {rel * 100:.1f}% apart)")


# ---------------------------------------------------------------------------
# 5. Plain-pointer Flash Attention forward (no WS): Q/K/V/O plain tl.load/store
#    auto-TMA promoted to DEVICE descriptors. Exercises per-program base (off_hz),
#    the whole-dim (HEAD_DIM, unmasked) constant-extent path, and a loop-carried
#    K/V load. Numerics vs torch SDPA.
# ---------------------------------------------------------------------------
@triton.jit
def _plain_fa_fwd(Q, K, V, O, sm_scale, N_CTX, stride_h, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    # per-program base for this (batch, head)
    q_base = Q + off_hz * stride_h
    k_base = K + off_hz * stride_h
    v_base = V + off_hz * stride_h
    o_base = O + off_hz * stride_h
    q = tl.load(q_base + offs_m[:, None] * stride_m + offs_d[None, :], mask=offs_m[:, None] < N_CTX, other=0.0)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.44269504  # log2(e)
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + offs_n[:, None] * stride_m + offs_d[None, :], mask=offs_n[:, None] < N_CTX,
                    other=0.0)  # [BLOCK_N, HEAD_DIM]
        v = tl.load(v_base + offs_n[:, None] * stride_m + offs_d[None, :], mask=offs_n[:, None] < N_CTX,
                    other=0.0)  # [BLOCK_N, HEAD_DIM]
        qk = tl.dot(q, k.T) * qk_scale  # [BLOCK_M, BLOCK_N]
        qk = tl.where(offs_n[None, :] < N_CTX, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_ij
    acc = acc / l_i[:, None]
    tl.store(o_base + offs_m[:, None] * stride_m + offs_d[None, :], acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


@pytest.mark.skipif(not _is_sm90plus(), reason="auto-TMA promotion requires sm_90+")
def test_plain_fa_fwd_device(monkeypatch):
    """A plain-pointer FA fwd (no descriptors, no warp_specialize): the Q/K/V
    loads are auto-TMA promoted to DEVICE make_tensor_descriptor + descriptor_load
    (per-program base folded into the descriptor base; unmasked HEAD_DIM extent as
    a constant), numerics match torch SDPA."""
    monkeypatch.setenv("TRITON_AUTO_TMA", "1")
    monkeypatch.setenv("TRITON_AUTO_TMA_DEVICE", "1")
    Z, H, N_CTX, HEAD_DIM = 1, 2, 256, 64
    BLOCK_M, BLOCK_N = 64, 64
    dtype = torch.float16
    sm_scale = 0.5
    torch.manual_seed(0)
    q = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    k = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    v = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    o = torch.empty_like(q)

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    stride_h = q.stride(1)  # one (batch,head) block = N_CTX*HEAD_DIM
    stride_m = q.stride(2)  # seq stride = HEAD_DIM
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    kernel = _plain_fa_fwd[grid](q, k, v, o, sm_scale, N_CTX, stride_h, stride_m, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                 HEAD_DIM=HEAD_DIM)

    ttir = kernel.asm["ttir"]
    assert "tt.make_tensor_descriptor" in ttir, \
        "plain-pointer FA Q/K/V loads should promote to device descriptors"
    assert "tt.descriptor_load" in ttir, "expected descriptor_load"

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=False)
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=0)


# ---------------------------------------------------------------------------
# 4. Plain-pointer Flash Attention forward WITH warp_specialize: the plain loads
#    are auto-TMA promoted (device descriptors) AND the loop is warp-specialized.
#    auto-TMA is the WS enabler (descriptor_load is the partition scheduler seed).
# ---------------------------------------------------------------------------
@triton.jit
def _plain_fa_fwd_ws(Q, K, V, O, sm_scale, N_CTX, stride_h, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_base = Q + off_hz * stride_h
    k_base = K + off_hz * stride_h
    v_base = V + off_hz * stride_h
    o_base = O + off_hz * stride_h
    q = tl.load(q_base + offs_m[:, None] * stride_m + offs_d[None, :], mask=offs_m[:, None] < N_CTX, other=0.0)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.44269504
    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=True, merge_epilogue=True, separate_epilogue_store=True,
                            data_partition_factor=2):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + offs_n[:, None] * stride_m + offs_d[None, :], mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_m + offs_d[None, :], mask=offs_n[:, None] < N_CTX, other=0.0)
        # Mirror 06-fused-attention.py _attn_fwd_inner (non-causal STAGE), but with
        # plain tl.load (auto-TMA promotes) instead of desc_k/desc_v.load.
        qk = tl.dot(q, k.T)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    acc = acc / l_i[:, None]
    tl.store(o_base + offs_m[:, None] * stride_m + offs_d[None, :], acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


@pytest.mark.skipif(not _is_sm90plus(), reason="WS + auto-TMA requires sm_90+")
def test_plain_fa_fwd_ws_device(monkeypatch):
    """plain-pointer FA + warp_specialize: loads auto-TMA'd to device descriptors,
    loop warp-specialized, numerics vs torch SDPA."""
    monkeypatch.setenv("TRITON_AUTO_TMA", "1")
    monkeypatch.setenv("TRITON_AUTO_TMA_DEVICE", "1")
    Z, H, N_CTX, HEAD_DIM = 1, 2, 256, 64
    BLOCK_M, BLOCK_N = 128, 64  # dp=2 -> 64 rows per partition
    dtype = torch.float16
    sm_scale = 0.5
    torch.manual_seed(0)
    q = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    k = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    v = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    o = torch.empty_like(q)

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    with triton.knobs.nvidia.scope():
        # Meta WS + meta partition scheduler (as the hand-written FA WS kernel
        # uses); the default upstream add_warp_specialize does not partition the
        # FA softmax compute.
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True
        triton.knobs.nvidia.disable_wsbarrier_reorder = True
        kernel = _plain_fa_fwd_ws[grid](q, k, v, o, sm_scale, N_CTX, q.stride(1), q.stride(2), BLOCK_M=BLOCK_M,
                                        BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, num_warps=4, num_stages=2, maxRegAutoWS=192)

    ttgir = kernel.asm["ttgir"]
    assert "tt.descriptor_load" in kernel.asm["ttir"], "expected auto-TMA promotion"
    assert "ttg.warp_specialize" in ttgir, "expected warp specialization"
    assert "ttng.async_tma_copy_global_to_local" in ttgir, "expected async TMA copy"

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=False)
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=0)


def _run_plain_fa_ws(q, k, v, o, sm_scale, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM):
    """Launch _plain_fa_fwd_ws under the meta-WS scheduler (as the hand-written
    FA WS kernel does). Requires TRITON_AUTO_TMA[_DEVICE]=1 set by the caller."""
    Z, H = q.shape[0], q.shape[1]

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True
        triton.knobs.nvidia.disable_wsbarrier_reorder = True
        return _plain_fa_fwd_ws[grid](q, k, v, o, sm_scale, N_CTX, q.stride(1), q.stride(2), BLOCK_M=BLOCK_M,
                                      BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, num_warps=4, num_stages=2, maxRegAutoWS=192)


@pytest.mark.skipif(not _is_sm90plus(), reason="WS + auto-TMA requires sm_90+")
@pytest.mark.parametrize("N_CTX", [512, 1024, 2048])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
def test_plain_fa_fwd_ws_shapes(monkeypatch, N_CTX, HEAD_DIM):
    """plain-pointer FA + WS + auto-TMA correctness across larger shapes."""
    monkeypatch.setenv("TRITON_AUTO_TMA", "1")
    monkeypatch.setenv("TRITON_AUTO_TMA_DEVICE", "1")
    Z, H = 2, 4
    BLOCK_M, BLOCK_N = 128, 64
    dtype = torch.float16
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    torch.manual_seed(0)
    q = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    k = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    v = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    o = torch.empty_like(q)
    _run_plain_fa_ws(q, k, v, o, sm_scale, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=False)
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=0)


@pytest.mark.skipif(
    os.environ.get("TRITON_RUN_PERF", "0") != "1",
    reason="perf benchmark (large FA via do_bench); set TRITON_RUN_PERF=1 to run",
)
@pytest.mark.skipif(not _is_sm90plus(), reason="WS + auto-TMA requires sm_90+")
def test_perf_plain_fa_fwd_ws(monkeypatch):
    """Perf of the plain-pointer FA + WS + auto-TMA at a representative shape,
    with accuracy vs torch SDPA. Prints TFLOP/s (pin a free GPU)."""
    monkeypatch.setenv("TRITON_AUTO_TMA", "1")
    monkeypatch.setenv("TRITON_AUTO_TMA_DEVICE", "1")
    Z, H, N_CTX, HEAD_DIM = 4, 32, 4096, 128
    BLOCK_M, BLOCK_N = 128, 64
    dtype = torch.float16
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    torch.manual_seed(0)
    q = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    k = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    v = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    o = torch.empty_like(q)
    _run_plain_fa_ws(q, k, v, o, sm_scale, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=False)
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=0)
    fn = lambda: _run_plain_fa_ws(q, k, v, o, sm_scale, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM)
    ms = triton.testing.do_bench(fn)
    flops = 2 * 2.0 * Z * H * N_CTX * N_CTX * HEAD_DIM  # QK^T + PV, non-causal
    tflops = flops * 1e-12 / (ms * 1e-3)
    at = os.environ.get("TRITON_AUTO_TMA", "0")
    print(f"\n[plain-fa-ws] auto_tma={at} fwd "
          f"({Z}x{H}x{N_CTX}x{HEAD_DIM}): {ms:.4f} ms  {tflops:.1f} TFLOP/s  "
          f"accuracy(vs torch SDPA)=PASS\n")
