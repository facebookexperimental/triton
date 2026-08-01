"""V3 - A deeper pipeline and persistent scheduling.

Reference kernel for the AMD blog "Optimizing GEMM + Activation on CDNA4 with TLX".
Computes the fused addmm + GLU:  out = X + X*Y,  X = A @ B + bias.

V3 builds on V2 with three changes:
  * NUM_BUFFERS raised from 2 to 3, so the tile a local read consumes was fetched
    two iterations earlier and the hot-loop wait can relax from a full drain to
    "keep one tile in flight" (`async_load_wait_group(1)`).
  * Each tile's A and B loads are committed as a single async group, so one group
    in flight corresponds to one A+B tile.
  * Persistent scheduling: a fixed grid of workgroups (one per CU) loops over its
    share of the output tiles, cutting launch overhead and keeping the V2 grouped
    ordering's L2 reuse warm across tiles.

The epilogue is still the unfused one from V2 (separate bias add and X + X*Y gate);
V4 collapses it. See the blog's "Version 3" section.

DO NOT MERGE - reference code accompanying the blog, not intended for upstreaming.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

M, N = 1024, 21568

NUM_XCDS = 8  # gfx950 (CDNA4, MI350)

BEST_CONFIG = {
    256:  dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=4,
              XCD_CHUNK=4, num_warps=8, matrix_instr_nonkdim=16, waves_per_eu=0),
    512:  dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=4,
              XCD_CHUNK=4, num_warps=8, matrix_instr_nonkdim=16, waves_per_eu=0),
    1024: dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=4,
              XCD_CHUNK=4, num_warps=8, matrix_instr_nonkdim=16, waves_per_eu=0),
}


@triton.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """XCD swizzle: remap tile IDs so co-scheduled tiles share an XCD's L2."""
    aligned = (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size)
    if pid >= aligned:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return ((local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size))


@triton.jit
def addmm_glu_v3(
    a_ptr, b_ptr, bias_ptr, y_ptr, c_ptr,
    M, N, K,
    sa0, sa1, sb0, sb1, sy0, sy1, sc0, sc1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    NUM_CUS: tl.constexpr,
):
    tl.assume(sa0 > 0); tl.assume(sa1 > 0)
    tl.assume(sb0 > 0); tl.assume(sb1 > 0)
    tl.assume(sy0 > 0); tl.assume(sy1 > 0)
    tl.assume(sc0 > 0); tl.assume(sc1 > 0)

    NUM_BUFFERS: tl.constexpr = 3

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # LDS buffers allocated once and reused across tiles (fully drained between tiles).
    smemA = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    # Persistent scheduling: each workgroup strides over its share of the tiles.
    for tile_id in tl.range(start_pid, num_tiles, NUM_CUS):
        swizzled = chiplet_transform_chunked(tile_id, num_tiles, NUM_XCDS, XCD_CHUNK)
        group_id = swizzled // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((swizzled % num_pid_in_group) % group_size_m)
        pid_n = (swizzled % num_pid_in_group) // group_size_m

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_base_off = offs_m[:, None] * sa0
        b_base_off = offs_n[None, :] * sb1
        k_iters = tl.cdiv(K, BLOCK_SIZE_K)

        # Prologue: issue NUM_BUFFERS=3 global reads, A+B combined into one group each.
        k_start = 0
        a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
        b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
        tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 0), mask=offs_k[None, :] < K - k_start)
        tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 0), mask=offs_k[:, None] < K - k_start)
        tlx.async_load_commit_group([tok_a, tok_b])

        k_start = BLOCK_SIZE_K
        a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
        b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
        tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 1), mask=offs_k[None, :] < K - k_start)
        tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 1), mask=offs_k[:, None] < K - k_start)
        tlx.async_load_commit_group([tok_a, tok_b])

        k_start = BLOCK_SIZE_K * 2
        a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
        b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
        tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 2), mask=offs_k[None, :] < K - k_start)
        tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 2), mask=offs_k[:, None] < K - k_start)
        tlx.async_load_commit_group([tok_a, tok_b])

        # Keep one group (the newest) in flight; the tile we consume was fetched earlier.
        wait_tok = tlx.async_load_wait_group(1)
        a_tile = tlx.local_load(tlx.local_view(smemA, 0), token=wait_tok)
        b_tile = tlx.local_load(tlx.local_view(smemB, 0), token=wait_tok)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Hot loop: warp-pipelined MFMA / async-prefetch, wait relaxed to one in flight.
        for i in tl.range(0, k_iters - NUM_BUFFERS, loop_unroll_factor=0):
            prefetch_buf = i % NUM_BUFFERS
            next_buf = (i + 1) % NUM_BUFFERS
            k_prefetch = (i + NUM_BUFFERS) * BLOCK_SIZE_K

            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

            with tlx.warp_pipeline_stage("mem", priority=1):
                a_offs = a_base_off + (k_prefetch + offs_k[None, :]) * sa1
                b_offs = (k_prefetch + offs_k[:, None]) * sb0 + b_base_off
                tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, prefetch_buf),
                                       mask=offs_k[None, :] < K - k_prefetch)
                tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, prefetch_buf),
                                       mask=offs_k[:, None] < K - k_prefetch)
                tlx.async_load_commit_group([tok_a, tok_b])
                a_tile = tlx.local_load(tlx.local_view(smemA, next_buf), token=wait_tok)
                b_tile = tlx.local_load(tlx.local_view(smemB, next_buf), token=wait_tok)

            wait_tok = tlx.async_load_wait_group(1)

        # Epilogue: drain remaining in-flight tiles.
        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
        wait_tok = tlx.async_load_wait_group(0)
        for j in tl.static_range(0, NUM_BUFFERS - 1):
            buf = (k_iters - (NUM_BUFFERS - 1) + j) % NUM_BUFFERS
            a_tile = tlx.local_load(tlx.local_view(smemA, buf), token=wait_tok)
            b_tile = tlx.local_load(tlx.local_view(smemB, buf), token=wait_tok)
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

        # Unfused epilogue (as in V2): bias add, then GLU gate out = X + X*Y.
        bias = tl.load(bias_ptr + offs_n).to(tl.float32)
        x = acc + bias[None, :]
        y_ptrs = y_ptr + offs_m[:, None] * sy0 + offs_n[None, :] * sy1
        y = tl.load(y_ptrs, cache_modifier=".cs").to(tl.float32)
        out = x + x * y

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c_ptrs = c_ptr + offs_cm[:, None] * sc0 + offs_cn[None, :] * sc1
        tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask, cache_modifier=".cs")


def run(a, b, bias, y):
    """Launch the V3 (persistent) kernel. Returns out = (A@B + bias) + (A@B + bias) * Y."""
    Mx, K = a.shape
    _, Nx = b.shape
    cfg = BEST_CONFIG[K]
    num_cus = torch.cuda.get_device_properties(a.device).multi_processor_count
    out = torch.empty((Mx, Nx), device=a.device, dtype=torch.float16)
    grid = (min(num_cus, triton.cdiv(Mx, cfg["BLOCK_SIZE_M"]) * triton.cdiv(Nx, cfg["BLOCK_SIZE_N"])),)
    addmm_glu_v3[grid](
        a, b, bias, y, out, Mx, Nx, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        y.stride(0), y.stride(1), out.stride(0), out.stride(1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"], GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=NUM_XCDS, XCD_CHUNK=cfg["XCD_CHUNK"], NUM_CUS=num_cus,
        num_warps=cfg["num_warps"], num_stages=1,
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"], waves_per_eu=cfg["waves_per_eu"],
    )
    return out


def _reference(bias, a, b, y):
    x = torch.matmul(a, b).to(torch.float32) + bias.to(torch.float32)[None, :]
    return (x + x * y.to(torch.float32)).to(torch.float16)


if __name__ == "__main__":
    for K in (256, 512, 1024):
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        y = torch.randn(M, N, device="cuda", dtype=torch.float16)
        out = run(a, b, bias, y)
        torch.testing.assert_close(out.float(), _reference(bias, a, b, y).float(),
                                   atol=2e-1, rtol=2e-2)
        print(f"V3 K={K}: correctness OK")
