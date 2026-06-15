"""
TLX warp-pipelined GEMM tutorial for AMD gfx950.

Uses async_load with 3-buffer pipelining and computes offsets from tile_id
(3 iter_args: acc, a_tile, b_tile). The main loop is split into "mfma" and
"mem" warp-pipeline stages.

Includes an XCD-aware PID remap (chunked) for L2 reuse across the 8 XCDs
of MI300X-class chips.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """Group adjacent PIDs onto the same XCD in chunks of chunk_size for L2 reuse.

    PIDs in the trailing remainder (not a multiple of num_xcds*chunk_size) pass through.
    """
    aligned = (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size)
    if pid >= aligned:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return ((local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size))


@triton.jit
def gemm_wp(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    grid_mn = num_pid_m * num_pid_n
    pid = chiplet_transform_chunked(pid, grid_mn, NUM_XCDS, XCD_CHUNK)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Precompute row/col offsets (these are per-thread, not carried in loop)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Base offsets — recompute full pointer from tile_id * BLOCK_K
    a_base_off = offs_m[:, None] * stride_am
    b_base_off = offs_n[None, :] * stride_bn

    K_ITERS = tl.cdiv(K, BLOCK_K)

    smemA = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    # Prologue: async-copy NUM_BUFFERS tiles
    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        k_start = i * BLOCK_K
        a_offs = a_base_off + (k_start + offs_k[None, :]) * stride_ak
        b_offs = (k_start + offs_k[:, None]) * stride_bk + b_base_off
        tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, i), mask=offs_k[None, :] < K - k_start)
        tlx.async_load_commit_group([tok_a])
        tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, i), mask=offs_k[:, None] < K - k_start)
        tlx.async_load_commit_group([tok_b])

    # Wait for buffer 0
    tlx.async_load_wait_group((NUM_BUFFERS - 1) * 2)
    a_tile = tlx.local_load(tlx.local_view(smemA, 0))
    b_tile = tlx.local_load(tlx.local_view(smemB, 0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop: only 3 iter_args (acc, a_tile, b_tile)
    # Offsets computed from tile_id — no pointer iter_args
    for tile_id in tl.range(0, K_ITERS - NUM_BUFFERS):
        prefetch_buf = tile_id % NUM_BUFFERS
        next_buf = (tile_id + 1) % NUM_BUFFERS
        k_prefetch = (tile_id + NUM_BUFFERS) * BLOCK_K

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

        tlx.async_load_wait_group(2)

        with tlx.warp_pipeline_stage("mem", priority=1):
            a_offs = a_base_off + (k_prefetch + offs_k[None, :]) * stride_ak
            b_offs = (k_prefetch + offs_k[:, None]) * stride_bk + b_base_off
            tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, prefetch_buf), mask=offs_k[None, :]
                                   < K - k_prefetch)
            tlx.async_load_commit_group([tok_a])
            tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, prefetch_buf), mask=offs_k[:, None]
                                   < K - k_prefetch)
            tlx.async_load_commit_group([tok_b])
            a_tile = tlx.local_load(tlx.local_view(smemA, next_buf))
            b_tile = tlx.local_load(tlx.local_view(smemB, next_buf))

    # Epilogue
    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
    tlx.async_load_wait_group(0)
    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        buf = (K_ITERS - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS
        a_tile = tlx.local_load(tlx.local_view(smemA, buf))
        b_tile = tlx.local_load(tlx.local_view(smemB, buf))
        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# gfx950 has 8 XCDs per chip; chunked remap improves L2 reuse.
# XCD_CHUNK=4 was empirically best for our 256x256 tiles at 4K-8K (per-XCD
# group of 4 PIDs aligns with the GROUP_M=8 row partitioning).
NUM_XCDS = 8
XCD_CHUNK = 4


def run(a, b, c, bm, bn, bk, nb, nw, gm):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn), )
    gemm_wp[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        NUM_BUFFERS=nb,
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=XCD_CHUNK,
        num_warps=nw,
        num_stages=1,
    )
    return c


if __name__ == "__main__":
    tflops = lambda ms, M, N, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    configs = [
        ("256x256x32 nw8 gm8", 256, 256, 32, 3, 8, 8),
        ("256x256x32 nw8 gm16", 256, 256, 32, 3, 8, 16),
        ("256x128x32 nw8 gm8", 256, 128, 32, 3, 8, 8),
        ("128x256x32 nw8 gm8", 128, 256, 32, 3, 8, 8),
        ("256x256x32 nw4 gm8", 256, 256, 32, 3, 4, 8),
        ("128x128x32 nw8 gm8", 128, 128, 32, 3, 8, 8),
    ]

    for size in [4096, 8192]:
        M = N = K = size
        torch.manual_seed(42)
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        ref = torch.matmul(a, b)
        c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

        print(f"\n{'='*70}")
        print(f"  M=N=K={size}")
        print(f"{'='*70}")
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b), rep=200)
        print(f"  {'rocBLAS':<32s} {tflops(ms,M,N,K):7.1f} TFLOPS ({ms:.3f} ms)")

        for name, bm, bn, bk, nb, nw, gm in configs:
            lds_kb = (bm * bk + bk * bn) * 2 * nb / 1024
            if lds_kb > 160:
                print(f"  {name:<32s} {'SKIP':>7s} (LDS={lds_kb:.0f}KB)")
                continue
            try:
                run(a, b, c, bm, bn, bk, nb, nw, gm)
                torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
                ok = "OK"
            except Exception:
                ok = "FAIL"
            ms = triton.testing.do_bench(
                lambda bm=bm, bn=bn, bk=bk, nb=nb, nw=nw, gm=gm: run(a, b, c, bm, bn, bk, nb, nw, gm), rep=200)
            print(f"  {name:<32s} {tflops(ms,M,N,K):7.1f} TFLOPS ({ms:.3f} ms) [{ok}]")
