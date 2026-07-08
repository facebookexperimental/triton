"""Grouped GEMM for AMD gfx950 (MI350/MI355): Warp-pipelined hot loop + L2 swizzle.

Input: A packed [sum(M_g), K], B [G, K, N], C packed [sum(M_g), N]
All tensors are passed as kernel-arg base pointers (per-group located by a 
cumulative-offset array for M and by g*K*N for B).

The kernel consists of a hotloop where we async-copy A/B tiles into an DS ring 
buffer (NUM_BUFFERS deep) and split the K-loop into explicit "mfma"/"mem" 
warp_pipeline_stages so MFMA overlaps the next tile's loads.
"""
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# gfx950 has 8 XCDs
NUM_XCDS = 8

def num_sms():
    return torch.cuda.get_device_properties(DEVICE).multi_processor_count

@triton.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """Permute program ids so adjacent-in-chunk pids land on the same XCD (L2 reuse).
    """
    aligned = (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size)
    if pid >= aligned:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return ((local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size))

@triton.jit
def grouped_gemm_wp_kernel(
    a_ptr,        # packed A [sum(M_g), K]
    b_ptr,        # B [G, K, N]
    c_ptr,        # packed C [sum(M_g), N]
    group_offs,   # [G+1] int32, cumulative M offsets
    group_size,
    N,
    K,
    stride_am,
    stride_bg,
    stride_bk,
    stride_cm,
    NUM_SM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_bg > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)

    pid = tl.program_id(0)

    # Program id after L2 remapping
    pid = chiplet_transform_chunked(pid, NUM_SM, NUM_XCDS, XCD_CHUNK)

    num_n_tiles = tl.cdiv(N, BLOCK_N)

    # LDS ring buffers, allocated once and reused across every tile this program owns
    smemA = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, NUM_BUFFERS)

    tile_idx = pid
    last_problem_end = 0

    for g in range(group_size):
        m_start = tl.load(group_offs + g)
        m_end = tl.load(group_offs + g + 1)

        gm = m_end - m_start

        num_m_tiles = tl.cdiv(gm, BLOCK_M)

        num_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # GROUP_M swizzle within this group's tile grid
            local = tile_idx - last_problem_end
            num_pid_in_group = GROUP_M * num_n_tiles

            group_id = local // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_M)

            # Ensure that consecutive PIDs are assigned tiles along m dimension in
            # in groups of size GROUP_M
            pid_m = first_pid_m + ((local % num_pid_in_group) % group_size_m)
            pid_n = (local % num_pid_in_group) // group_size_m

            b_group_off = g.to(tl.int64) * stride_bg

            # Use modulo for M/N such that out of bounds indices read in bounds 
            # and are discarded at store
            offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % gm
            offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
            offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)

            # Get base pointers for A and B
            a_base_off = (m_start + offs_m)[:, None] * stride_am
            b_base_off = b_group_off + offs_n[None, :]

            K_ITERS = tl.cdiv(K, BLOCK_K)

            # Prologue: async-copy the first NUM_BUFFERS K-tiles.
            for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
                k_start = i * BLOCK_K
                a_offs = a_base_off + (k_start + offs_k[None, :])  # stride_ak = 1
                b_offs = b_base_off + (k_start + offs_k[:, None]) * stride_bk

                # GR i
                tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, i), mask=offs_k[None, :] < K - k_start)
                tlx.async_load_commit_group([tok_a])
                tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, i), mask=offs_k[:, None] < K - k_start)
                tlx.async_load_commit_group([tok_b])

            tlx.async_load_wait_group((NUM_BUFFERS - 1) * 2)

            # LR0
            a_tile = tlx.local_load(tlx.local_view(smemA, 0))
            b_tile = tlx.local_load(tlx.local_view(smemB, 0))

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Hot loop loop: mfma(k) overlaps GR k+NUM_BUFFERS, LR k+1
            for k in tl.range(0, K_ITERS - NUM_BUFFERS):
                prefetch_buf = k % NUM_BUFFERS
                next_buf = (k + 1) % NUM_BUFFERS
                k_prefetch = (k + NUM_BUFFERS) * BLOCK_K

                with tlx.warp_pipeline_stage("mfma", priority=0):
                    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

                tlx.async_load_wait_group((NUM_BUFFERS - 2) * 2)

                with tlx.warp_pipeline_stage("mem", priority=1):
                    a_offs = a_base_off + (k_prefetch + offs_k[None, :])
                    b_offs = b_base_off + (k_prefetch + offs_k[:, None]) * stride_bk
                    tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, prefetch_buf),
                                           mask=offs_k[None, :] < K - k_prefetch)
                    tlx.async_load_commit_group([tok_a])
                    tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, prefetch_buf),
                                           mask=offs_k[:, None] < K - k_prefetch)
                    tlx.async_load_commit_group([tok_b])
                    a_tile = tlx.local_load(tlx.local_view(smemA, next_buf))
                    b_tile = tlx.local_load(tlx.local_view(smemB, next_buf))

            # Epilogue: drain the last NUM_BUFFERS K-tiles
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
            tlx.async_load_wait_group(0)
            for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
                buf = (K_ITERS - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS
                a_tile = tlx.local_load(tlx.local_view(smemA, buf))
                b_tile = tlx.local_load(tlx.local_view(smemB, buf))
                acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

            c = acc.to(tl.float16)
            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + (m_start + offs_cm)[:, None] * stride_cm + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

            # Program p owns tiles p, p+NUM_SM, p+2*NUM_SM, and so on
            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


# Starting config
_CONFIG = {
    "BLOCK_M": 256,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "GROUP_M": 4,
    "NUM_BUFFERS": 2,
    "XCD_CHUNK": 8,
    "num_warps": 8,
}


def grouped_gemm(a_packed, b, group_offs, c_packed=None, config=None):
    """Ragged-M grouped GEMM (uniform N/K), packed layout.

    a_packed:   [sum(M_g), K] fp16, row-major.
    b:          [G, K, N] fp16, row-major.
    group_offs: [G+1] int32 device tensor, cumulative M offsets (group_offs[0] == 0).
    Returns:    c_packed [sum(M_g), N] fp16.
    """
    cfg = dict(_CONFIG)
    if config:
        cfg.update(config)
    G, K, N = b.shape
    total_M = a_packed.shape[0]
    assert a_packed.shape[1] == K
    if c_packed is None:
        c_packed = torch.empty((total_M, N), device=DEVICE, dtype=a_packed.dtype)

    NUM_SM = num_sms()
    grid = (NUM_SM, )
    grouped_gemm_wp_kernel[grid](
        a_packed,
        b,
        c_packed,
        group_offs,
        G,
        N,
        K,
        a_packed.stride(0),
        b.stride(0),
        b.stride(1),
        c_packed.stride(0),
        NUM_SM=NUM_SM,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        NUM_BUFFERS=cfg["NUM_BUFFERS"],
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=cfg["XCD_CHUNK"],
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return c_packed


def _make_packed(m_list, N, K):
    """Build packed A, stacked B, group_offs and a list of per-group (A_i, B_i)."""
    group_A = [torch.randn((M, K), device=DEVICE, dtype=torch.float16) for M in m_list]
    B = torch.randn((len(m_list), K, N), device=DEVICE, dtype=torch.float16)
    a_packed = torch.cat(group_A, dim=0).contiguous()
    offs = torch.tensor([0] + list(torch.tensor(m_list).cumsum(0).tolist()), dtype=torch.int32, device=DEVICE)
    return a_packed, B, offs, group_A


def _check(m_list, N, K, label):
    a_packed, B, offs, group_A = _make_packed(m_list, N, K)
    c = grouped_gemm(a_packed, B, offs)
    start = 0
    for i, M in enumerate(m_list):
        ref = group_A[i] @ B[i]
        torch.testing.assert_close(c[start:start + M], ref, atol=1e-2, rtol=1e-2)
        start += M
    print(f"  [PASS] {label} ({len(m_list)} groups)")


def test_op():
    _check([4096, 2048, 1000, 333], 4096, 4096, "ragged-M (MoE-style), N=K=4096")
    _check([512, 256, 128, 100], 2048, 1024, "ragged-M, N=2048 K=1024")
    _check([4096] * 16, 4096, 4096, "fixed 16x4096^3")
    print("test_op: all correctness checks passed")


def _bench():
    def tflops(ms, total_flops):
        return total_flops * 1e-12 / (ms * 1e-3)

    n = 16
    m_list = [4096] * n
    a_packed, B, offs, group_A = _make_packed(m_list, 4096, 4096)
    total_flops = sum(2 * M * 4096 * 4096 for M in m_list)

    ms = triton.testing.do_bench(lambda: grouped_gemm(a_packed, B, offs), rep=100)
    print(f"  v1 grouped GEMM : {tflops(ms, total_flops):7.1f} TFLOPS ({ms:.3f} ms)")

    ms_torch = triton.testing.do_bench(lambda: [group_A[i] @ B[i] for i in range(n)], rep=100)
    print(f"  torch loop      : {tflops(ms_torch, total_flops):7.1f} TFLOPS ({ms_torch:.3f} ms)")


if __name__ == "__main__":
    test_op()
    print("\n16 x 4096 x 4096 x 4096:")
    _bench()
