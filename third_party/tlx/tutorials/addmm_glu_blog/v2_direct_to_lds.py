"""V2 - Direct-to-LDS, prefetching, swizzling, autotuning.

Reference kernel for the AMD blog "Optimizing GEMM + Activation on CDNA4 with TLX".
Computes the fused addmm + GLU:  out = X + X*Y,  X = A @ B + bias.

V2 removes V1's two weaknesses:
  * Direct-to-LDS async loads (`tlx.async_load`) copy A/B straight from HBM into
    LDS, skipping the register-staging detour.
  * The hot loop is split into two `tlx.warp_pipeline_stage`s (an MFMA stage and a
    memory stage) so compute overlaps the next prefetch.
  * L2 locality: an XCD swizzle keeps co-scheduled workgroups on the same XCD,
    grouped tile ordering (GROUP_SIZE_M) reuses A rows / B columns, and streaming
    (`.cs`) hints keep the single-use Y and output out of L2.

The hot-loop wait is a full drain (`async_load_wait_group(0)`): with only two
buffers the tile a local read consumes was fetched one step earlier, so the wave
must wait on every in-flight load. V3 deepens the pipeline to relax this. See the
blog's "Version 2" section.

DO NOT MERGE - reference code accompanying the blog, not intended for upstreaming.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

M, N = 1024, 21568

# gfx950 (CDNA4, MI350) has 8 XCDs, each with its own L2 slice.
NUM_XCDS = 8
XCD_CHUNK = 4

# Autotuning winning configs per K.
BEST_CONFIG = {
    256:  dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
              NUM_BUFFERS=2, num_warps=4, matrix_instr_nonkdim=16, waves_per_eu=0),
    512:  dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8,
              NUM_BUFFERS=2, num_warps=4, matrix_instr_nonkdim=16, waves_per_eu=2),
    1024: dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8,
              NUM_BUFFERS=2, num_warps=4, matrix_instr_nonkdim=16, waves_per_eu=0),
}


@triton.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """XCD swizzle: remap program IDs so co-scheduled workgroups share an XCD's L2."""
    aligned = (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size)
    if pid >= aligned:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return ((local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size))


@triton.jit
def addmm_glu_v2(
    a_ptr, b_ptr, bias_ptr, y_ptr, c_ptr,
    M, N, K,
    sa0, sa1, sb0, sb1, sy0, sy1, sc0, sc1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    tl.assume(sa0 > 0); tl.assume(sa1 > 0)
    tl.assume(sb0 > 0); tl.assume(sb1 > 0)
    tl.assume(sy0 > 0); tl.assume(sy1 > 0)
    tl.assume(sc0 > 0); tl.assume(sc1 > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid_mn = num_pid_m * num_pid_n

    # L2 swizzle, then grouped tile ordering.
    pid = chiplet_transform_chunked(pid, grid_mn, NUM_XCDS, XCD_CHUNK)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_base_off = offs_m[:, None] * sa0
    b_base_off = offs_n[None, :] * sb1
    k_iters = tl.cdiv(K, BLOCK_SIZE_K)

    smemA = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    # Prologue: async direct-to-LDS prefetch of NUM_BUFFERS tiles.
    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        k_start = i * BLOCK_SIZE_K
        a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
        b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
        tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, i), mask=offs_k[None, :] < K - k_start)
        tlx.async_load_commit_group([tok_a])
        tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, i), mask=offs_k[:, None] < K - k_start)
        tlx.async_load_commit_group([tok_b])

    tlx.async_load_wait_group(0)
    a_tile = tlx.local_load(tlx.local_view(smemA, 0))
    b_tile = tlx.local_load(tlx.local_view(smemB, 0))

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Hot loop: warp-pipelined MFMA / async-prefetch.
    for tile_id in tl.range(0, k_iters - NUM_BUFFERS):
        prefetch_buf = tile_id % NUM_BUFFERS
        next_buf = (tile_id + 1) % NUM_BUFFERS
        k_prefetch = (tile_id + NUM_BUFFERS) * BLOCK_SIZE_K

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

        # Full drain: with two buffers, the tile consumed next was fetched one step
        # ago, so we must wait on every outstanding load. V3 relaxes this.
        tlx.async_load_wait_group(0)

        with tlx.warp_pipeline_stage("mem", priority=1):
            a_offs = a_base_off + (k_prefetch + offs_k[None, :]) * sa1
            b_offs = (k_prefetch + offs_k[:, None]) * sb0 + b_base_off
            tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, prefetch_buf),
                                   mask=offs_k[None, :] < K - k_prefetch)
            tlx.async_load_commit_group([tok_a])
            tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, prefetch_buf),
                                   mask=offs_k[:, None] < K - k_prefetch)
            tlx.async_load_commit_group([tok_b])

            a_tile = tlx.local_load(tlx.local_view(smemA, next_buf))
            b_tile = tlx.local_load(tlx.local_view(smemB, next_buf))

    # Epilogue: drain the remaining in-flight tiles.
    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
    tlx.async_load_wait_group(0)
    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        buf = (k_iters - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS
        a_tile = tlx.local_load(tlx.local_view(smemA, buf))
        b_tile = tlx.local_load(tlx.local_view(smemB, buf))
        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    # Fused epilogue: bias add, then GLU gate (out = X + X*Y). Streaming (.cs) hints
    # keep the single-use Y and output out of L2.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    x = acc + bias[None, :]

    y_ptrs = y_ptr + offs_cm[:, None] * sy0 + offs_cn[None, :] * sy1
    y = tl.load(y_ptrs, mask=c_mask, other=0.0, cache_modifier=".cs").to(tl.float32)
    out = x + x * y

    c_ptrs = c_ptr + offs_cm[:, None] * sc0 + offs_cn[None, :] * sc1
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask, cache_modifier=".cs")


def run(a, b, bias, y):
    """Launch the V2 kernel. Returns out = (A@B + bias) + (A@B + bias) * Y."""
    Mx, K = a.shape
    _, Nx = b.shape
    cfg = BEST_CONFIG[K]
    out = torch.empty((Mx, Nx), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(Mx, cfg["BLOCK_SIZE_M"]) * triton.cdiv(Nx, cfg["BLOCK_SIZE_N"]),)
    addmm_glu_v2[grid](
        a, b, bias, y, out, Mx, Nx, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        y.stride(0), y.stride(1), out.stride(0), out.stride(1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"], GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_BUFFERS=cfg["NUM_BUFFERS"], NUM_XCDS=NUM_XCDS, XCD_CHUNK=XCD_CHUNK,
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
        print(f"V2 K={K}: correctness OK")
