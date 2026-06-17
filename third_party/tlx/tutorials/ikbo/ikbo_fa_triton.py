"""IKBO Flash Attention — stock Triton kernel.

In recommendation model inference, query (Q) is per-candidate while
key (K) and value (V) are per-user. Traditional approaches broadcast
user K/V to match the candidate batch size before calling attention.
IKBO eliminates this by accepting mismatched batch sizes and performing
the broadcast lookup inside the kernel.

    Q: [B_cand * n_seed, num_heads, d_head]   — per candidate
    K: [B_user * max_seq_len, num_heads, d_head] — per user
    V: [B_user * max_seq_len, num_heads, d_head] — per user
    cand_to_user_index: [B_cand] int32 — maps candidate to user

    out[b] = softmax(Q[b] @ K[u(b)]^T / sqrt(d)) @ V[u(b)]

Reference diffs: D103791854 (TLX), D105270628 (Triton)
"""

import random

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# =============================================================================
# PyTorch reference
# =============================================================================


def fa_reference(
    query,
    key,
    value,
    cand_to_user_index,
    n_seed,
    num_heads,
    d_head,
    max_seq_len,
):
    """PyTorch eager reference: broadcast user K/V then call SDPA."""
    query_sdpa = query.view(-1, n_seed, num_heads, d_head).permute(0, 2, 1, 3)
    key_sdpa = key.view(-1, max_seq_len, num_heads, d_head)
    key_sdpa_broadcast = torch.index_select(
        key_sdpa,
        dim=0,
        index=cand_to_user_index,
    ).permute(0, 2, 1, 3)
    value_sdpa = value.view(-1, max_seq_len, num_heads, d_head)
    value_sdpa_broadcast = torch.index_select(
        value_sdpa,
        dim=0,
        index=cand_to_user_index,
    ).permute(0, 2, 1, 3)
    return torch.nn.functional.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa_broadcast,
        value_sdpa_broadcast,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )


# =============================================================================
# Input generation
# =============================================================================


def create_inputs(
    B,
    n_seed,
    num_heads,
    d_head,
    max_seq_len,
    cand_to_user_ratio=64,
    dtype=torch.float16,
    device=DEVICE,
):
    """Create IKBO FA test inputs.

    Args:
        B: Number of candidates.
        n_seed: Query sequence length per candidate.
        num_heads: Number of attention heads.
        d_head: Head dimension.
        max_seq_len: KV sequence length per user.
        cand_to_user_ratio: Candidates per user.
        dtype: Data type.
        device: Device.

    Returns:
        (query, key, value, cand_to_user_index, cand_grid)
    """

    def _generate_num_cands_per_user():
        res = []
        cum_sum = 0
        while True:
            cur = random.randint(
                cand_to_user_ratio,
                cand_to_user_ratio,
            ) + random.randint(0, 1)
            if cum_sum + cur >= B:
                res.append(B - cum_sum)
                break
            cum_sum += cur
            res.append(cur)
        return res

    num_cands_per_user = _generate_num_cands_per_user()
    num_cands_per_user_tensor = torch.tensor(num_cands_per_user)
    cand_grid = torch.arange(B, dtype=torch.int32, device=device)

    cand_to_user_index = torch.repeat_interleave(
        torch.arange(num_cands_per_user_tensor.size(0)),
        num_cands_per_user_tensor,
    ).to(dtype=torch.int32, device=device)
    Bu = num_cands_per_user_tensor.size(0)

    query = torch.randn(
        (B * n_seed, num_heads, d_head),
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        (Bu * max_seq_len, num_heads, d_head),
        device=device,
        dtype=dtype,
    )
    value = torch.randn(
        (Bu * max_seq_len, num_heads, d_head),
        device=device,
        dtype=dtype,
    )
    return query, key, value, cand_to_user_index, cand_grid


# =============================================================================
# Triton kernel — inner attention loop
# =============================================================================


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    max_seq_len,
    qk_scale,
    allow_tf32,
    BLOCK_N: tl.constexpr,
):
    offset_seq_n = tl.arange(0, BLOCK_N)
    num_iter = tl.cdiv(max_seq_len, BLOCK_N)
    for i_iter in tl.range(0, num_iter):
        start_n = i_iter * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kT = tl.load(K_block_ptr, boundary_check=[0, 1], padding_option="zero")
        qk = tl.dot(q, kT, allow_tf32=allow_tf32)
        if (i_iter == num_iter - 1) & (max_seq_len % BLOCK_N != 0):
            mask_seq = offset_seq_n[None, :] < max_seq_len - start_n
            qk = tl.where(mask_seq, qk, -1.0e10)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr, boundary_check=[0, 1], padding_option="zero")
        p = p.to(v.dtype)
        acc += tl.dot(p, v, allow_tf32=allow_tf32)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    inv_li = 1.0 / l_i[:, None]
    acc *= inv_li
    return acc, l_i, m_i


# =============================================================================
# Triton kernel — main
# =============================================================================

_is_hip = torch.version.hip is not None

_nvidia_configs = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
]

_amd_configs = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
]


@triton.autotune(
    configs=_amd_configs if _is_hip else _nvidia_configs,
    key=["n_seed", "max_seq_len", "d_head"],
)
@triton.jit
def _ikbo_fa_kernel(
    Q,
    K,
    V,
    output,
    cand_to_user_index,
    cand_grid,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_o_token,
    stride_o_head,
    stride_o_dim,
    n_seed: tl.constexpr,
    num_heads: tl.constexpr,
    d_head: tl.constexpr,
    max_seq_len,
    total_q_tokens,
    total_kv_tokens,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_cand = tl.load(cand_grid + tl.program_id(axis=0))
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)

    allow_tf32 = ALLOW_TF32

    user_idx = tl.load(cand_to_user_index + pid_cand).to(tl.int32)

    seq_start_q = pid_cand * n_seed
    seq_start_kv = user_idx * max_seq_len

    qk_scale = 1.44269504089 * (1.0 / tl.sqrt(tl.cast(d_head, tl.float32)))

    Q_block_ptr = tl.make_block_ptr(
        base=Q + pid_head * stride_q_head,
        shape=(total_q_tokens, d_head),
        strides=(stride_q_token, stride_q_dim),
        offsets=(seq_start_q + pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, d_head),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + pid_head * stride_k_head,
        shape=(d_head, total_kv_tokens),
        strides=(stride_k_dim, stride_k_token),
        offsets=(0, seq_start_kv),
        block_shape=(d_head, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + pid_head * stride_v_head,
        shape=(total_kv_tokens, d_head),
        strides=(stride_v_token, stride_v_dim),
        offsets=(seq_start_kv, 0),
        block_shape=(BLOCK_N, d_head),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=output + pid_head * stride_o_head,
        shape=(total_q_tokens, d_head),
        strides=(stride_o_token, stride_o_dim),
        offsets=(seq_start_q + pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, d_head),
        order=(1, 0),
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, d_head], dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=[0, 1], padding_option="zero")

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,
        max_seq_len,
        qk_scale,
        allow_tf32,
        BLOCK_N,
    )
    tl.store(O_block_ptr, acc.to(output.dtype.element_ty), boundary_check=[0, 1])


# =============================================================================
# Wrapper
# =============================================================================


def ikbo_fa(
    query,
    key,
    value,
    cand_to_user_index,
    cand_grid,
    n_seed,
    num_heads,
    d_head,
    max_seq_len,
):
    """IKBO Flash Attention with in-kernel user broadcast.

    Args:
        query: [B_cand * n_seed, num_heads, d_head]
        key: [B_user * max_seq_len, num_heads, d_head]
        value: [B_user * max_seq_len, num_heads, d_head]
        cand_to_user_index: [B_cand] int32
        cand_grid: [num_cand_blocks] int32 — candidate indices for grid
        n_seed: Query sequence length per candidate.
        num_heads: Number of attention heads.
        d_head: Head dimension.
        max_seq_len: KV sequence length per user.

    Returns:
        output: [B_cand, num_heads, n_seed, d_head] (SDPA-compatible layout)
    """
    output = torch.empty_like(query)

    grid = lambda META: (
        cand_grid.shape[0],
        num_heads,
        triton.cdiv(n_seed, META["BLOCK_M"]),
    )

    _ikbo_fa_kernel[grid](
        query,
        key,
        value,
        output,
        cand_to_user_index,
        cand_grid,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        n_seed=n_seed,
        num_heads=num_heads,
        d_head=d_head,
        max_seq_len=max_seq_len,
        total_q_tokens=query.shape[0],
        total_kv_tokens=key.shape[0],
        ALLOW_TF32=not _is_hip,
    )

    return output.view(-1, n_seed, num_heads, d_head).permute(0, 2, 1, 3)
