"""Grouped GEMM for AMD gfx950 (MI350/MI355): Naive correctness baseline

Many independent FP16 GEMMs (one per group, variable M/N/K) fused into a single
persistent launch.

The input is per-group pointer tables and flat size/stride arrays. Layout is
standard A[M,K] x B[K,N] (K contiguous in A, N contiguous in B), tl.dot(a, b),
FP32 accumulate, FP16 output.
"""
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def num_sms():
    return torch.cuda.get_device_properties(DEVICE).multi_processor_count


@triton.jit
def grouped_gemm_naive_kernel(
    # Per-group base pointers (int64 data pointers indexed by group number)
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # Flat [group_size, 3] of <M, N, K> per group
    group_gemm_sizes,
    # Flat [group_size, 3] of <lda, ldb, ldc> (row strides) per group
    g_lds,
    # Number of gemms
    group_size,
    # Number of persistent programs
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Which global output tile we are computing
    tile_idx = tl.program_id(0)

    # The global tile id for where the current group begins
    last_problem_end = 0

    for g in range(group_size):
        # Load gemm sizes
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        # How many tiles are necessary to compute this specific gemm
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # This program owns the tiles where tile_idx lies within this group's
        # tiles in the range [last_problem_end, +num_tiles)
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # Load strides
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            # Get base pointers
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

            # Get group-relative tile index
            tile_idx_in_gemm = tile_idx - last_problem_end

            # Which tile we are within this gemm along the m axis
            tile_m_idx = tile_idx_in_gemm // num_n_tiles

            # Which tile we are within this gemm along the n axis
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # Row offsets for a
            offs_am = tl.multiple_of((tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % gm, BLOCK_SIZE_M)

            # Column offsets for b
            offs_bn = tl.max_contiguous(
                tl.multiple_of((tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % gn, BLOCK_SIZE_N),
                BLOCK_SIZE_N)

            # Offsets into k reduction dimension
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # Calculate pointers into a and b
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            k_full_chunk_iters = gk // BLOCK_SIZE_K

            for kk in tl.range(0, k_full_chunk_iters):
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])

                # k_start = kk * BLOCK_SIZE_K

                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)

                accumulator += tl.dot(a, b)

                # Move to next k tile
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb

            if k_full_chunk_iters * BLOCK_SIZE_K < gk:
                k_start = k_full_chunk_iters * BLOCK_SIZE_K

                offs_am_2 = tl.multiple_of((tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % gm, BLOCK_SIZE_M)
                offs_bn_2 = tl.max_contiguous(
                    tl.multiple_of((tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % gn, BLOCK_SIZE_N),
                    BLOCK_SIZE_N)
                offs_k_2 = tl.arange(0, BLOCK_SIZE_K)

                a_ptrs_2 = a_ptr + offs_am_2[:, None] * lda + (k_start + offs_k_2[None, :])
                b_ptrs_2 = b_ptr + (k_start + offs_k_2[:, None]) * ldb + offs_bn_2[None, :]

                tl.multiple_of(a_ptrs_2, [16, 16])
                tl.multiple_of(b_ptrs_2, [16, 16])

                a = tl.load(a_ptrs_2, mask=offs_k_2[None, :] < gk - k_start, other=0.0)
                b = tl.load(b_ptrs_2, mask=offs_k_2[:, None] < gk - k_start, other=0.0)

                accumulator += tl.dot(a, b)

            c = accumulator.to(tl.float16)

            # Write back to GMEM
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < gn)
            tl.store(c_ptrs, c, mask=c_mask)

            # Program p owns tiles p, p+NUM_SM, p+2*NUM_SM, and so on
            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


_BLOCK_M = 256
_BLOCK_N = 256
_BLOCK_K = 64


def grouped_gemm(group_A, group_B, group_C=None):
    """Compute [A_i @ B_i for each group i] in one persistent launch.

    group_A[i]: [M_i, K_i] fp16, row-major (K contiguous).
    group_B[i]: [K_i, N_i] fp16, row-major (N contiguous).
    Returns a list of [M_i, N_i] fp16 outputs.
    """
    group_size = len(group_A)
    assert len(group_B) == group_size

    A_addrs, B_addrs, C_addrs, g_sizes, g_lds = [], [], [], [], []
    out = []
    for i in range(group_size):
        A, B = group_A[i], group_B[i]
        assert A.shape[1] == B.shape[0], f"K mismatch group {i}: {A.shape} x {B.shape}"
        M, K = A.shape
        _, N = B.shape
        C = group_C[i] if group_C is not None else torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        out.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    NUM_SM = num_sms()
    grid = (NUM_SM, )
    grouped_gemm_naive_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        NUM_SM=NUM_SM,
        BLOCK_SIZE_M=_BLOCK_M,
        BLOCK_SIZE_N=_BLOCK_N,
        BLOCK_SIZE_K=_BLOCK_K,
        num_warps=8,
        matrix_instr_nonkdim=16,
    )
    return out


def _make_groups(m_list, n_list, k_list):
    group_A, group_B = [], []
    for M, N, K in zip(m_list, n_list, k_list):
        group_A.append(torch.randn((M, K), device=DEVICE, dtype=torch.float16))
        group_B.append(torch.randn((K, N), device=DEVICE, dtype=torch.float16))
    return group_A, group_B


def _check(m_list, n_list, k_list, label):
    group_A, group_B = _make_groups(m_list, n_list, k_list)
    tri = grouped_gemm(group_A, group_B)
    ref = [a @ b for a, b in zip(group_A, group_B)]
    for i, (r, t) in enumerate(zip(ref, tri)):
        torch.testing.assert_close(t, r, atol=1e-2, rtol=1e-2)
    print(f"  [PASS] {label} ({len(m_list)} groups)")


def test_op():
    # Ragged: distinct M, N, K per group
    _check([1024, 512, 256, 128], [1024, 512, 256, 128], [1024, 512, 256, 128], "ragged M/N/K")
    # Ragged-M only
    _check([4096, 2048, 1000, 333], [4096] * 4, [4096] * 4, "ragged-M (MoE-style)")
    # 16 equal 4096^3 groups
    _check([4096] * 16, [4096] * 16, [4096] * 16, "fixed 16x4096^3")
    print("test_op: all correctness checks passed")


def _bench():

    def tflops(ms, m_list, n_list, k_list):
        flops = sum(2 * m * n * k for m, n, k in zip(m_list, n_list, k_list))
        return flops * 1e-12 / (ms * 1e-3)

    m_list = n_list = k_list = [4096] * 16
    group_A, group_B = _make_groups(m_list, n_list, k_list)

    ms = triton.testing.do_bench(lambda: grouped_gemm(group_A, group_B), rep=100)
    print(f"  v0 grouped GEMM : {tflops(ms, m_list, n_list, k_list):7.1f} TFLOPS ({ms:.3f} ms)")

    ms_torch = triton.testing.do_bench(lambda: [a @ b for a, b in zip(group_A, group_B)], rep=100)
    print(f"  torch loop      : {tflops(ms_torch, m_list, n_list, k_list):7.1f} TFLOPS ({ms_torch:.3f} ms)")


if __name__ == "__main__":
    test_op()
    print("\n16 x 4096 x 4096 x 4096:")
    _bench()
