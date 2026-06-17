"""IKBO Linear Compression Embedding (LCE) — stock Triton kernel.

Linear Compression Embedding compresses input embeddings via a learned
projection. The IKBO variant exploits the many-to-one mapping from
candidates to users: user-side matmul is computed once per unique user
and broadcast inside the kernel, avoiding redundant work proportional
to the candidate-to-user ratio.

    Decomposed LCE:
        user_res[u]  = W_user @ E_user[u]           for each unique user u
        out[b]       = W_cand @ E_cand[b] + user_res[u(b)]   for each candidate b

    IKBO fusion (this kernel):
        The candidate GEMM + user broadcast add is fused into a single kernel.
        user_res is pre-computed via cuBLAS/torch.matmul.

Reference diffs: D96879328 (TLX), D89392529 / D105270628 (Triton)
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# =============================================================================
# PyTorch reference
# =============================================================================


def lce_reference(
    compression_w_cand,
    compression_w_user,
    embeddings_cand,
    embeddings_user,
    cand_to_user_index,
):
    """PyTorch eager reference: decomposed LCE with explicit broadcast."""
    cand_res = compression_w_cand @ embeddings_cand
    user_res = compression_w_user @ embeddings_user
    return cand_res + torch.index_select(user_res, dim=0, index=cand_to_user_index)


# =============================================================================
# Input generation
# =============================================================================


def create_inputs(B, M, N, K_USER, K_CAND, cand_to_user_ratio, dtype=torch.float16, device=DEVICE):
    """Create test inputs with a fixed candidate-to-user ratio.

    Args:
        B: Number of candidates (ads batch size).
        M: Compressed output dimension.
        N: Embedding dimension.
        K_USER: Number of user embedding features.
        K_CAND: Number of candidate embedding features.
        cand_to_user_ratio: Candidates per user (Bu = B / ratio).
        dtype: Data type for tensors.
        device: Device for tensors.

    Returns:
        (compression_w_cand, compression_w_user, embeddings_cand,
         embeddings_user, cand_to_user_index)
    """
    num_users = (B + cand_to_user_ratio - 1) // cand_to_user_ratio
    cand_to_user_index = (torch.arange(B, device=device) // cand_to_user_ratio).to(torch.int32)

    compression_w_cand = torch.randn((M, K_CAND), device=device, dtype=dtype)
    compression_w_user = torch.randn((M, K_USER), device=device, dtype=dtype)
    embeddings_cand = torch.randn((B, K_CAND, N), device=device, dtype=dtype)
    embeddings_user = torch.randn((num_users, K_USER, N), device=device, dtype=dtype)

    return (
        compression_w_cand,
        compression_w_user,
        embeddings_cand,
        embeddings_user,
        cand_to_user_index,
    )


# =============================================================================
# Triton kernel
# =============================================================================


@triton.jit
def _swizzle_tile(
    tile_id,
    num_pids_per_batch,
    num_pid_m,
    num_pid_n,
    GROUP_SIZE_M: tl.constexpr,
):
    """Map a linear tile_id to (pid_batch, pid_m, pid_n) with grouped ordering."""
    pid_batch = tile_id // num_pids_per_batch
    local_tile = tile_id % num_pids_per_batch
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = local_tile // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
    pid_n = (local_tile % num_pid_in_group) // group_size_m
    return pid_batch, pid_m, pid_n


def _autotune_configs():
    return [
        triton.Config(
            {"BM": bm, "BN": bn, "BK": bk, "GROUP_SIZE_M": 8},
            num_stages=ns,
            num_warps=nw,
        ) for bm in [64, 128] for bn in [64, 128] for bk in [64, 128] for ns in [3, 4, 5] for nw in [4, 8]
    ]


@triton.autotune(
    configs=_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _ikbo_lce_kernel(
    a_ptr,
    b_ptr,
    cand_to_user_index_ptr,
    user_res_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    a_stride_m,
    a_stride_k,
    b_stride_batch,
    b_stride_k,
    b_stride_n,
    out_stride_batch,
    out_stride_m,
    out_stride_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused candidate GEMM + user broadcast add.

    Computes: out[b] = W_cand @ E_cand[b] + user_res[u(b)]
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pids_per_batch = num_pid_m * num_pid_n

    pid_batch, pid_m, pid_n = _swizzle_tile(
        pid,
        num_pids_per_batch,
        num_pid_m,
        num_pid_n,
        GROUP_SIZE_M,
    )

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    ram = offs_m % M
    rbn = offs_n % N
    rk = tl.arange(0, BK)

    a_ptrs = a_ptr + (ram[:, None] * a_stride_m + rk[None, :] * a_stride_k)
    b_ptrs = (b_ptr + pid_batch * b_stride_batch + rk[:, None] * b_stride_k + rbn[None, :] * b_stride_n)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(K, 0, -BK):
        a = tl.load(a_ptrs, mask=rk[None, :] < k, other=0.0)
        b = tl.load(b_ptrs, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * a_stride_k
        b_ptrs += BK * b_stride_k

    user_idx = tl.load(
        cand_to_user_index_ptr + pid_batch,
        eviction_policy="evict_last",
    )
    idx_m = offs_m[:, None]
    idx_n = offs_n[None, :]
    mask = (idx_m < M) & (idx_n < N)
    in_batch_offset = out_stride_n * idx_n + out_stride_m * idx_m
    user_res_val = tl.load(
        user_res_ptr + (in_batch_offset + out_stride_batch * user_idx),
        mask,
        eviction_policy="evict_last",
    )

    output_val = acc.to(a_ptr.dtype.element_ty) + user_res_val
    out_ptrs = out_ptr + in_batch_offset + out_stride_batch * pid_batch
    tl.store(out_ptrs, output_val, mask)


# =============================================================================
# Wrapper
# =============================================================================


def ikbo_lce(
    compression_w_cand,
    compression_w_user,
    embeddings_cand,
    embeddings_user,
    cand_to_user_index,
):
    """IKBO LCE: candidate GEMM + in-kernel user broadcast.

    Two-phase computation:
      1. User: user_res = W_user @ E_user  (via torch.matmul, computed once)
      2. Fused: out[b] = W_cand @ E_cand[b] + user_res[u(b)]  (Triton kernel)
    """
    B, K, N = embeddings_cand.shape
    M = compression_w_cand.size(0)
    user_res = compression_w_user @ embeddings_user
    out = torch.empty(
        (B, M, N),
        device=embeddings_cand.device,
        dtype=embeddings_cand.dtype,
    )

    grid = lambda META: (B * triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]), )

    _ikbo_lce_kernel[grid](
        compression_w_cand,
        embeddings_cand,
        cand_to_user_index,
        user_res,
        out,
        M,
        N,
        K,
        compression_w_cand.stride(0),
        compression_w_cand.stride(1),
        embeddings_cand.stride(0),
        embeddings_cand.stride(1),
        embeddings_cand.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )

    return out
