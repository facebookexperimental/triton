"""Grouped GEMM for AMD gfx950 (MI350/MI355): column-major B, mask-free hot loop.

Design vs v1_4w_switched:
  * B is stored **column-major [K, N]** (K contiguous) == row-major [N, K].
    This makes N the *outer* (non-contiguous) dimension of B, so we can WRAP N
    with `% gn` exactly like we already wrap M for A -- no N-masking, no
    `buffer_load_to_local`, no branch, no v_cndmask, and B gets the compact LDS
    layout (so bigger tiles fit).
  * B is loaded **symmetric to A**: both are [outer, K] tiles with K on the
    innermost (contiguous) axis -> plain coalesced `async_load` into
    smemB=[BLOCK_N,BLOCK_K]. The transpose to the [K,N] operand MFMA needs is
    done for free on the LDS read via `tlx.local_load(tlx.local_trans(smemB))`
    (folded into the shared-memory layout -- no shuffles, no explicit transpose),
    so the dot is a plain `tl.dot(a_tile, b_tile)`.

Constraint: gn must be >= 16. The `tl.multiple_of(gn, 16)` hint enables wide
vectorized C stores (worth ~110 TFLOPS at 4096^3); it degrades correctness only
for gn < 16 (a degenerate case). Unaligned gn >= 16 (e.g. 300, 1000) is correct.
  * K (now B's contiguous dim) is handled by the usual peel: the hot loop runs
    only full K-tiles; the partial last K-tile is peeled into a cold `tl.load`
    that masks on K. So the hot loop carries **no masking at all**.

Host layout: group_B[i] is a column-major [K, N] tensor (K contiguous), i.e. a
`.t()` view of an [N, K] row-major buffer -- the natural weight layout.
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
    """Permute program ids so adjacent-in-chunk pids land on the same XCD (L2 reuse)."""
    aligned = (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size)
    if pid >= aligned:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return ((local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size))


@triton.jit
def grouped_gemm_kernel(group_a_ptrs, group_b_ptrs, group_c_ptrs,
                        # [group_size, 3] of <M, N, K>
                        group_gemm_sizes,
                        # [group_size, 3] of <lda, ldb, ldc>
                        g_lds, group_size, NUM_SM: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, NUM_XCDS: tl.constexpr,
                        XCD_CHUNK: tl.constexpr, NUM_BUFFERS: tl.constexpr):
    pid = tl.program_id(0)
    pid = chiplet_transform_chunked(pid, NUM_SM, NUM_XCDS, XCD_CHUNK)

    # LDS ring buffers. Both A and B are [outer, K] tiles (K innermost/contiguous)
    smemA = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_SIZE_N, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS)

    tile_idx = pid
    last_problem_end = 0

    for g in range(group_size):
        a_ptr = tl.multiple_of(tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16)), 16)
        b_ptr = tl.multiple_of(tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16)), 16)
        c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gn = tl.multiple_of(gn, 16)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        stride_am = tl.load(g_lds + g * 3)  # A row stride
        stride_bn = tl.multiple_of(tl.load(g_lds + g * 3 + 1), 16)  # B N-stride
        stride_cm = tl.load(g_lds + g * 3 + 2)  # C row stride

        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        k_full_chunk_iters = gk // BLOCK_SIZE_K

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            local = tile_idx - last_problem_end
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles
            group_id = local // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((local % num_pid_in_group) % group_size_m)
            pid_n = (local % num_pid_in_group) // group_size_m

            # A rows and B columns are wrapped to keep all reads in bound and keep
            # the the K-contiguous vectorized load. The garbage from wrapped lanes
            # is dropped by the masked C store
            offs_am = tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % gm, BLOCK_SIZE_M)
            offs_bn = tl.multiple_of((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % gn, BLOCK_SIZE_N)

            # K is the contiguous/innermost axis of both A and B tiles
            offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)

            a_base_off = offs_am[:, None] * stride_am  # [BLOCK_M, 1]
            b_base_off = offs_bn[:, None] * stride_bn  # [BLOCK_N, 1]

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            if k_full_chunk_iters >= NUM_BUFFERS:
                # Prologue: async-copy the first NUM_BUFFERS K-tiles
                for pi in tl.static_range(0, NUM_BUFFERS):
                    k_start = pi * BLOCK_SIZE_K
                    a_offs = a_base_off + (k_start + offs_k[None, :])
                    b_offs = b_base_off + (k_start + offs_k[None, :])
                    tok_a = tlx.async_load(a_ptr + a_offs, smemA[pi], cache_modifier=".ca",
                                           eviction_policy="evict_first")
                    tok_b = tlx.async_load(b_ptr + b_offs, smemB[pi], cache_modifier=".ca",
                                           eviction_policy="evict_last")
                    tlx.async_load_commit_group([tok_a, tok_b])

                tlx.async_load_wait_group(max((NUM_BUFFERS - 2), 0))

                a_tile = tlx.local_load(smemA[0])
                b_tile = tlx.local_load(tlx.local_trans(smemB[0]))

                n_steady = k_full_chunk_iters - NUM_BUFFERS

                for i in tl.range(0, n_steady, disallow_acc_multi_buffer=True):
                    # K-tile i
                    prefetch_buf = i % NUM_BUFFERS
                    next_buf = (i + 1) % NUM_BUFFERS
                    k_prefetch = (i + NUM_BUFFERS) * BLOCK_SIZE_K

                    with tlx.warp_pipeline_stage("mfma", priority=0):
                        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

                    with tlx.warp_pipeline_stage("mem", priority=1):
                        a_offs = a_base_off + (k_prefetch + offs_k[None, :])
                        b_offs = b_base_off + (k_prefetch + offs_k[None, :])
                        tok_a = tlx.async_load(a_ptr + a_offs, smemA[prefetch_buf], cache_modifier=".ca",
                                               eviction_policy="evict_first")
                        tok_b = tlx.async_load(b_ptr + b_offs, smemB[prefetch_buf], cache_modifier=".ca",
                                               eviction_policy="evict_last")
                        tlx.async_load_commit_group([tok_a, tok_b])

                        a_tile = tlx.local_load(smemA[next_buf], relaxed=True)
                        b_tile = tlx.local_load(tlx.local_trans(smemB[next_buf]), relaxed=True)

                    tlx.async_load_wait_group(max((NUM_BUFFERS - 2), 0))

                # Epilogue: drain the last NUM_BUFFERS K-tiles
                acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
                tlx.async_load_wait_group(0)

                for i in tl.static_range(0, NUM_BUFFERS - 1):
                    buf = (k_full_chunk_iters - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS
                    a_tile = tlx.local_load(smemA[buf])
                    b_tile = tlx.local_load(tlx.local_trans(smemB[buf]))
                    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
            else:
                # Small-k path: load all full K-tiles, then dot
                for i in tl.range(0, k_full_chunk_iters):
                    k_start = i * BLOCK_SIZE_K
                    a_offs = a_base_off + (k_start + offs_k[None, :])
                    b_offs = b_base_off + (k_start + offs_k[None, :])
                    tok_a = tlx.async_load(a_ptr + a_offs, smemA[i], cache_modifier=".ca",
                                           eviction_policy="evict_first")
                    tok_b = tlx.async_load(b_ptr + b_offs, smemB[i], cache_modifier=".ca", eviction_policy="evict_last")
                    tlx.async_load_commit_group([tok_a, tok_b])

                tlx.async_load_wait_group(0)

                for i in tl.range(0, k_full_chunk_iters):
                    a_tile = tlx.local_load(smemA[i])
                    b_tile = tlx.local_load(tlx.local_trans(smemB[i]))
                    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

            # Peel the partial last K-tile (gk % BLOCK_SIZE_K != 0)
            if k_full_chunk_iters * BLOCK_SIZE_K < gk:
                k_start = k_full_chunk_iters * BLOCK_SIZE_K
                a_offs = a_base_off + (k_start + offs_k[None, :])
                # Load B directly as [BLOCK_K, BLOCK_N] so the dot needs no transpose
                b_offs_t = (k_start + offs_k[:, None]) + offs_bn[None, :] * stride_bn
                a_tile = tl.load(a_ptr + a_offs, mask=offs_k[None, :] < gk - k_start, other=0.0)
                b_tile = tl.load(b_ptr + b_offs_t, mask=offs_k[:, None] < gk - k_start, other=0.0)
                acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

            c = acc.to(c_ptr.dtype.element_ty)
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < gn)
            tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".cs")

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


_CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "NUM_BUFFERS": 3,
    "XCD_CHUNK": 32,
    "num_warps": 8,
}


def grouped_gemm(group_A, group_B, config=None):
    """group_A[i]: fp16 [M_i, K_i] row-major (K contiguous).
       group_B[i]: fp16 [K_i, N_i] COLUMN-major (K contiguous) == [N_i, K_i].t().
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
        assert B.stride(0) == 1, "B must be column-major [K, N] (K contiguous, stride(0)==1)"
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        # lda = A row stride (K); ldb = B N-stride (== K for col-major); ldc = C row stride (N)
        g_lds += [A.stride(0), B.stride(1), C.stride(0)]

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
    """A[M,K] row-major; B[K,N] column-major (K contiguous) via a [N,K].t() view."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    group_A, group_B = [], []
    for (M, N, K) in shape_spec:
        group_A.append(torch.randn((M, K), device=DEVICE, dtype=torch.float16, generator=g))
        Bt = torch.randn((N, K), device=DEVICE, dtype=torch.float16, generator=g)  # [N,K] row-major
        group_B.append(Bt.t())  # [K,N] view, stride (1, K) == column-major
    return group_A, group_B


def _check(shape_spec, label):
    group_A, group_B = _rand_groups(shape_spec)
    group_C = grouped_gemm(group_A, group_B)
    for i, (A, B) in enumerate(zip(group_A, group_B)):
        ref = torch.matmul(A, B)
        torch.testing.assert_close(group_C[i], ref, atol=1e-2, rtol=1e-2)
    print(f"  [PASS] {label} ({len(shape_spec)} groups)")


def test_op():
    _check([(1024, 1024, 1024), (512, 512, 512), (256, 256, 256), (128, 128, 128)], "blackwell-ragged M=N=K")
    _check([(4096, 4096, 4096), (2048, 4096, 4096), (1000, 4096, 4096), (333, 4096, 4096)],
           "ragged-M (MoE-style), N=K=4096")
    _check([(512, 300, 4000), (333, 1000, 1500), (128, 128, 100), (256, 704, 320)], "k/n-unaligned")
    _check([(1, 64, 64), (33, 128, 50)], "tiny")
    print("test_op: all correctness checks passed")


def _bench():

    def tflops(ms, total_flops):
        return total_flops * 1e-12 / (ms * 1e-3)

    n = 16
    spec = [(4096, 4096, 4096)] * n
    group_A, group_B = _rand_groups(spec)
    total_flops = sum(2 * M * N * K for (M, N, K) in spec)

    ms = triton.testing.do_bench(lambda: grouped_gemm(group_A, group_B), rep=100)
    print(f"  v3 grouped GEMM : {tflops(ms, total_flops):7.1f} TFLOPS ({ms:.3f} ms)")

    ms_torch = triton.testing.do_bench(lambda: [group_A[i] @ group_B[i] for i in range(n)], rep=100)
    print(f"  torch loop      : {tflops(ms_torch, total_flops):7.1f} TFLOPS ({ms_torch:.3f} ms)")


if __name__ == "__main__":
    test_op()
    print("\n16 x 4096 x 4096 x 4096:")
    _bench()
