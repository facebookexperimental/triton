"""Grouped GEMM for AMD gfx950: v0's compiler-pipelined loop + column-major B.
"""
import os

import torch
import triton
import triton.language as tl

os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def num_sms():
    return torch.cuda.get_device_properties(DEVICE).multi_processor_count


@triton.jit
def grouped_gemm_v4_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,  # [group_size, 3] of <M, N, K>
    g_lds,  # [group_size, 3] of <lda, ldb, ldc>; ldb is B's N-stride (== K for col-major)
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0

    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)  # B N-stride (== K for column-major B)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # A rows and B columns are wrapped. N-wrap is safe now: N is B's outer
            # (non-contiguous) dim, so it can't produce an out-of-tensor vector read.
            offs_am = tl.multiple_of((tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % gm, BLOCK_SIZE_M)
            offs_bn = (tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % gn
            # K is the contiguous dim (of both A and column-major B) -> hint it.
            offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)

            # A: [BLOCK_M, BLOCK_K], K contiguous on axis 1.
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            # B (column-major): [BLOCK_K, BLOCK_N], K contiguous on axis 0 (stride 1),
            # N strided by ldb on axis 1. Fed straight to tl.dot as the [K, N] operand.
            b_ptrs = b_ptr + offs_k[:, None] + offs_bn[None, :] * ldb

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            k_full_chunk_iters = gk // BLOCK_SIZE_K

            for kk in tl.range(0, k_full_chunk_iters):
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])

                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)

                accumulator += tl.dot(a, b)

                a_ptrs += BLOCK_SIZE_K  # A: advance K (stride 1)
                b_ptrs += BLOCK_SIZE_K  # B col-major: advance K (stride 1)

            # Peel the partial last K-tile; re-materialize offsets (don't carry the
            # advanced pointers) to keep register pressure down
            if k_full_chunk_iters * BLOCK_SIZE_K < gk:
                k_start = k_full_chunk_iters * BLOCK_SIZE_K

                offs_am_2 = tl.multiple_of((tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % gm, BLOCK_SIZE_M)
                offs_bn_2 = (tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % gn
                offs_k_2 = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)

                a_ptrs_2 = a_ptr + offs_am_2[:, None] * lda + (k_start + offs_k_2[None, :])
                b_ptrs_2 = b_ptr + (k_start + offs_k_2[:, None]) + offs_bn_2[None, :] * ldb

                tl.multiple_of(a_ptrs_2, [16, 16])
                tl.multiple_of(b_ptrs_2, [16, 16])

                a = tl.load(a_ptrs_2, mask=offs_k_2[None, :] < gk - k_start, other=0.0)
                b = tl.load(b_ptrs_2, mask=offs_k_2[:, None] < gk - k_start, other=0.0)

                accumulator += tl.dot(a, b)

            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < gn)
            tl.store(c_ptrs, c, mask=c_mask)

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


_BLOCK_M = 256
_BLOCK_N = 256
_BLOCK_K = 64
_NUM_WARPS = 8


def grouped_gemm(group_A, group_B, config=None):
    """group_A[i]: fp16 [M, K] row-major. group_B[i]: fp16 [K, N] COLUMN-major (K contiguous)."""
    BM = (config or {}).get("BLOCK_SIZE_M", _BLOCK_M)
    BN = (config or {}).get("BLOCK_SIZE_N", _BLOCK_N)
    BK = (config or {}).get("BLOCK_SIZE_K", _BLOCK_K)
    W = (config or {}).get("num_warps", _NUM_WARPS)

    group_size = len(group_A)
    assert len(group_B) == group_size

    A_addrs, B_addrs, C_addrs, g_sizes, g_lds = [], [], [], [], []
    out = []
    for i in range(group_size):
        A, B = group_A[i], group_B[i]
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"K mismatch group {i}: {A.shape} x {B.shape}"
        assert B.stride(0) == 1, "B must be column-major [K, N] (K contiguous, stride(0)==1)"
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        out.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(1), C.stride(0)]  # lda, ldb(=B N-stride==K), ldc

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    NUM_SM = num_sms()
    grid = (NUM_SM, )
    grouped_gemm_v4_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        NUM_SM=NUM_SM,
        BLOCK_SIZE_M=BM,
        BLOCK_SIZE_N=BN,
        BLOCK_SIZE_K=BK,
        num_warps=W,
        matrix_instr_nonkdim=16,
    )
    return out


def _rand_groups(shape_spec, seed=0):
    """A[M,K] row-major; B[K,N] column-major (K contiguous) via a [N,K].t() view."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    group_A, group_B = [], []
    for (M, N, K) in shape_spec:
        group_A.append(torch.randn((M, K), device=DEVICE, dtype=torch.float16, generator=g))
        Bt = torch.randn((N, K), device=DEVICE, dtype=torch.float16, generator=g)
        group_B.append(Bt.t())  # [K,N] view, stride (1, K)
    return group_A, group_B


def _check(shape_spec, label):
    group_A, group_B = _rand_groups(shape_spec)
    group_C = grouped_gemm(group_A, group_B)
    for i, (A, B) in enumerate(zip(group_A, group_B)):
        torch.testing.assert_close(group_C[i], torch.matmul(A, B), atol=1e-2, rtol=1e-2)
    print(f"  [PASS] {label} ({len(shape_spec)} groups)")


def test_op():
    _check([(1024, 1024, 1024), (512, 512, 512), (256, 256, 256), (128, 128, 128)], "ragged M=N=K")
    _check([(4096, 4096, 4096), (2048, 4096, 4096), (1000, 4096, 4096), (333, 4096, 4096)], "ragged-M")
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
    print(f"  v4 grouped GEMM : {tflops(ms, total_flops):7.1f} TFLOPS ({ms:.3f} ms)")

    ms_torch = triton.testing.do_bench(lambda: [group_A[i] @ group_B[i] for i in range(n)], rep=100)
    print(f"  torch loop      : {tflops(ms_torch, total_flops):7.1f} TFLOPS ({ms_torch:.3f} ms)")


if __name__ == "__main__":
    test_op()
    print("\n16 x 4096 x 4096 x 4096:")
    _bench()
