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
def grouped_gemm_kernel(
        # device tensors of matrices pointers
        group_a_ptrs, group_b_ptrs, group_c_ptrs,
        # device tensor of gemm sizes. its shape is [group_size, 3]
        # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
        group_gemm_sizes,
        # device tensor of leading dimension sizes. its shape is [group_size, 3]
        # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
        g_lds,
        # number of gemms
        group_size,
        # number of virtual SM
        NUM_SM: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, NUM_XCDS: tl.constexpr, XCD_CHUNK: tl.constexpr, NUM_BUFFERS: tl.constexpr):
    pid = tl.program_id(0)

    # Program id after L2 remapping
    pid = chiplet_transform_chunked(pid, NUM_SM, NUM_XCDS, XCD_CHUNK)

    # LDS ring buffers, allocated once and reused across every tile this program owns
    smemA = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tl.float16, NUM_BUFFERS)

    tile_idx = pid
    last_problem_end = 0

    for g in range(group_size):
        # get pointers to matrices
        a_ptr = tl.multiple_of(tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16)), 16)
        b_ptr = tl.multiple_of(tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16)), 16)
        c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        stride_am = tl.load(g_lds + g * 3)
        stride_bk = tl.multiple_of(tl.load(g_lds + g * 3 + 1), 16)
        stride_cm = tl.load(g_lds + g * 3 + 2)

        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        k_full_chunk_iters = gk // BLOCK_SIZE_K

        # Could try converting this to a for loop and using flatten=True here
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # GROUP_SIZE_M swizzle within this group's tile grid
            local = tile_idx - last_problem_end
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles

            group_id = local // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M

            # Last group might have num rows less than GROUP_SIZE_M because
            # GROUP_SIZE_M does not evenly divide gm
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)

            # Ensure that consecutive PIDs are assigned tiles along m dimension in
            # groups of size GROUP_SIZE_M
            pid_m = first_pid_m + ((local % num_pid_in_group) % group_size_m)
            pid_n = (local % num_pid_in_group) // group_size_m

            # Rows along a are loaded with modulo such that OOB reads wrap around
            # which is fine since the final C store is masked
            offs_am = tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % gm, BLOCK_SIZE_M)

            # Columns along b/c are loaded in continuous chunks since the innermost dimension
            # of the async load needs to be be contiguous for vectorized loads
            offs_n = tl.max_contiguous(tl.multiple_of(pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), BLOCK_SIZE_N),
                                       BLOCK_SIZE_N)

            # Rows for c are contiguous and masked upon store
            offs_cm = tl.max_contiguous(tl.multiple_of(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), BLOCK_SIZE_M),
                                        BLOCK_SIZE_M)

            # Get base offsets for A and B
            a_base_off = offs_am[:, None] * stride_am
            b_base_off = offs_n[None, :]

            # Create accumulator register array
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)

            # The warp-pipelined GEMM only runs when there are at least NUM_BUFFERS
            # full K-tiles. Groups with fewer full tiles take the small-k path in
            # the else block
            if k_full_chunk_iters >= NUM_BUFFERS:
                # Prologue: async-copy the first NUM_BUFFERS K-tiles
                for i in tl.static_range(0, NUM_BUFFERS):
                    k_start = i * BLOCK_SIZE_K
                    a_offs = a_base_off + (k_start + offs_k[None, :])  # stride_ak = 1
                    b_offs = b_base_off + (k_start + offs_k[:, None]) * stride_bk

                    # GR i
                    tok_a = tlx.async_load(a_ptr + a_offs, smemA[i])
                    tok_b = tlx.async_load(b_ptr + b_offs, smemB[i])

                    tlx.async_load_commit_group([tok_a, tok_b])

                # Make GR0 and GR1 finish
                tlx.async_load_wait_group(max((NUM_BUFFERS - 2), 0))

                # LR0
                a_tile = tlx.local_load(smemA[0], relaxed=True)
                b_tile = tlx.local_load(smemB[0], relaxed=True)

                # Hot loop loop: mfma(i) overlaps GR i+NUM_BUFFERS, LR i+1
                for i in tl.range(0, k_full_chunk_iters - NUM_BUFFERS, loop_unroll_factor=0,
                                  disallow_acc_multi_buffer=True):
                    prefetch_buf = i % NUM_BUFFERS
                    next_buf = (i + 1) % NUM_BUFFERS
                    k_prefetch = (i + NUM_BUFFERS) * BLOCK_SIZE_K

                    with tlx.warp_pipeline_stage("mfma", priority=0):
                        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

                    with tlx.warp_pipeline_stage("mem", priority=1):
                        a_offs = a_base_off + (k_prefetch + offs_k[None, :])
                        b_offs = (k_prefetch + offs_k[:, None]) * stride_bk + b_base_off

                        tok_a = tlx.async_load(a_ptr + a_offs, smemA[prefetch_buf])
                        tok_b = tlx.async_load(b_ptr + b_offs, smemB[prefetch_buf])

                        tlx.async_load_commit_group([tok_a, tok_b])

                        a_tile = tlx.local_load(smemA[next_buf], relaxed=True)
                        b_tile = tlx.local_load(smemB[next_buf], relaxed=True)

                    # Most reently committed buffers (GR i + num_buffers) can be in flight
                    tlx.async_load_wait_group(max((NUM_BUFFERS - 2), 0))

                # Epilogue: drain the last NUM_BUFFERS K-tiles
                acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

                tlx.async_load_wait_group(0)

                for i in tl.static_range(0, NUM_BUFFERS - 1):
                    buf = (k_full_chunk_iters - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS

                    a_tile = tlx.local_load(smemA[buf], relaxed=True)
                    b_tile = tlx.local_load(smemB[buf], relaxed=True)

                    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
            else:
                # Small-k path: load the tiles and dot sequentially
                for i in tl.range(0, k_full_chunk_iters):
                    k_start = i * BLOCK_SIZE_K
                    a_offs = a_base_off + (k_start + offs_k[None, :])
                    b_offs = b_base_off + (k_start + offs_k[:, None]) * stride_bk

                    tok_a = tlx.async_load(a_ptr + a_offs, smemA[i])
                    tok_b = tlx.async_load(b_ptr + b_offs, smemB[i])

                    tlx.async_load_commit_group([tok_a, tok_b])

                tlx.async_load_wait_group(0)

                for i in tl.range(0, k_full_chunk_iters):
                    a_tile = tlx.local_load(smemA[i], relaxed=True)
                    b_tile = tlx.local_load(smemB[i], relaxed=True)

                    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

            # Finish the last remainder K-tile (partial tile when gk % BLOCK_SIZE_K != 0).
            # A's K is its contiguous dim and cannot be masked on the async path, so the
            # remainder uses tl.load
            if k_full_chunk_iters * BLOCK_SIZE_K < gk:
                k_start = k_full_chunk_iters * BLOCK_SIZE_K

                a_offs = a_base_off + (k_start + offs_k[None, :])
                b_offs = (k_start + offs_k[:, None]) * stride_bk + b_base_off

                a_tile = tl.load(a_ptr + a_offs, mask=offs_k[None, :] < gk - k_start, other=0.0)
                b_tile = tl.load(b_ptr + b_offs, mask=offs_k[:, None] < gk - k_start, other=0.0)

                acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

            c = acc.to(c_ptr.dtype.element_ty)
            c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_n[None, :]
            c_mask = (offs_cm[:, None] < gm) & (offs_n[None, :] < gn)
            tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".cs")

            # Program p owns tiles p, p+NUM_SM, p+2*NUM_SM, and so on
            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


# Best config
_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 4,
    "NUM_BUFFERS": 4,
    "XCD_CHUNK": 16,
    "num_warps": 4,
}


def grouped_gemm(group_A, group_B, config=None):
    """Ragged M/N/K grouped GEMM via the Blackwell pointer-array table.

    Mirrors blackwell-grouped-gemm_test.py::group_gemm_fn: builds device tables of
    per-group base pointers + sizes + leading dims and launches one persistent grid.

    group_A[i]: fp16 [M_i, K_i] row-major.
    group_B[i]: fp16 [K_i, N_i] row-major.
    Returns:    group_C, a list of fp16 [M_i, N_i] with C_i = A_i @ B_i.
    """
    cfg = dict(_CONFIG)
    if config:
        cfg.update(config)

    G = len(group_A)
    assert len(group_B) == G, "group_A / group_B length mismatch"

    A_addrs, B_addrs, C_addrs = [], [], []
    g_sizes, g_lds = [], []
    group_C = []
    for A, B in zip(group_A, group_B):
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"K mismatch: A has K={K}, B has K={Kb}"
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        # group_gemm_sizes column order: (M, N, K)
        g_sizes += [M, N, K]
        # g_lds column order: (lda, ldb, ldc) = row strides (inner dim is unit stride)
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # Device-side tables (note: ptr tables are int64 addresses from data_ptr()).
    d_a_ptrs = torch.tensor(A_addrs, dtype=torch.int64, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, dtype=torch.int64, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, dtype=torch.int64, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    NUM_SM = num_sms()
    grid = (NUM_SM, )
    grouped_gemm_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        G,
        NUM_SM=NUM_SM,
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=cfg["XCD_CHUNK"],
        NUM_BUFFERS=cfg["NUM_BUFFERS"],
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return group_C


def _rand_groups(shape_spec, seed=0):
    """Per-group A[M,K], B[K,N] fp16 tensors for a list of (M, N, K)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    group_A, group_B = [], []
    for (M, N, K) in shape_spec:
        group_A.append(torch.randn((M, K), device=DEVICE, dtype=torch.float16, generator=g))
        group_B.append(torch.randn((K, N), device=DEVICE, dtype=torch.float16, generator=g))
    return group_A, group_B


def _check(shape_spec, label):
    group_A, group_B = _rand_groups(shape_spec)
    group_C = grouped_gemm(group_A, group_B)
    for i, (A, B) in enumerate(zip(group_A, group_B)):
        ref = torch.matmul(A, B)
        torch.testing.assert_close(group_C[i], ref, atol=1e-2, rtol=1e-2)
    print(f"  [PASS] {label} ({len(shape_spec)} groups)")


def test_op():
    # Full ragged M/N/K parity (each group independent), mirroring blackwell test_op.
    _check([(1024, 1024, 1024), (512, 512, 512), (256, 256, 256), (128, 128, 128)], "blackwell-ragged M=N=K")
    _check([(4096, 4096, 4096), (2048, 4096, 4096), (1000, 4096, 4096), (333, 4096, 4096)],
           "ragged-M (MoE-style), N=K=4096")
    _check([(512, 300, 4000), (333, 1000, 1500), (128, 128, 100), (256, 704, 320)], "k/n-unaligned")
    _check([(1, 64, 64), (7, 7, 7), (33, 128, 50)], "tiny")
    print("test_op: all correctness checks passed")


def _bench():

    def tflops(ms, total_flops):
        return total_flops * 1e-12 / (ms * 1e-3)

    n = 16
    spec = [(4096, 4096, 4096)] * n
    group_A, group_B = _rand_groups(spec)
    total_flops = sum(2 * M * N * K for (M, N, K) in spec)

    ms = triton.testing.do_bench(lambda: grouped_gemm(group_A, group_B), rep=100)
    print(f"  v1 grouped GEMM : {tflops(ms, total_flops):7.1f} TFLOPS ({ms:.3f} ms)")

    ms_torch = triton.testing.do_bench(lambda: [group_A[i] @ group_B[i] for i in range(n)], rep=100)
    print(f"  torch loop      : {tflops(ms_torch, total_flops):7.1f} TFLOPS ({ms_torch:.3f} ms)")


if __name__ == "__main__":
    test_op()
    print("\n16 x 4096 x 4096 x 4096:")
    _bench()
