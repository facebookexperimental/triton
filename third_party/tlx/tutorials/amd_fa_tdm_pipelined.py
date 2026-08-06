"""
TDM-pipelined Flash-Attention forward for AMD gfx1250 (TLX).

Uses the gfx1250 TDM (tensor-descriptor) async-copy engine to stream K/V
tiles from global memory into double-buffered LDS, with WMMA matmuls via
``tl.dot`` and an online-softmax accumulator.

The software pipeline is hand-written: three iterations are peeled across
the prologue and epilogue, while K/V are double-buffered in LDS. The steady
state is unrolled by two so softmax work can overlap adjacent WMMA shadows.
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _load_k(k_buf, slot, wait_count):
    # async_wait(n): block until at most n TDM ops outstanding, then read
    # the K tile from LDS *transposed* ([BLOCK_N, HEAD_SZ] -> [HEAD_SZ,
    # BLOCK_N]) so the QK dot operand lowering uses a memdesc transpose
    # instead of a register shuffle.
    tlx.async_amd_descriptor_wait(wait_count)
    return tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, slot)))


@triton.jit
def _load_v(v_buf, slot, wait_count):
    tlx.async_amd_descriptor_wait(wait_count)
    return tlx.local_load(tlx.local_view(v_buf, slot))


@triton.jit
def _compute_qk(q, k, cur_seq, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SEQLEN_K: tl.constexpr):
    qk = tl.dot(q, k)
    qk_mask = (cur_seq + tl.arange(0, BLOCK_N))[None, :] < SEQLEN_K
    qk = tl.where(qk_mask, qk, float("-inf"))
    return qk


@triton.jit
def _compute_qk_no_mask(q, k):
    return tl.dot(q, k)


@triton.jit
def _softmax_part0(qk, m_i, scale_ln2):
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    m_ij_scaled = m_ij * scale_ln2
    q_shifted = scale_ln2 * qk - m_ij_scaled[:, None]
    p = tl.math.exp2(q_shifted)
    alpha = tl.math.exp2(scale_ln2 * m_i - m_ij_scaled)
    return p, alpha, m_ij


@triton.jit
def _softmax_part1(p, l_i, acc, alpha):
    l_ij = tl.sum(p, 1)
    acc = acc * alpha[:, None]
    p_bf16 = p.to(tl.bfloat16, fp_downcast_rounding="rtz")
    l_i = l_i * alpha + l_ij
    return p_bf16, l_i, acc


@triton.jit
def attn_fwd_tdm_pipelined_kernel(q_ptr, k_ptr, v_ptr, o_ptr,  #
                                  stride_qz, stride_qh, stride_qm, stride_qk,  #
                                  stride_kz, stride_kh, stride_kn, stride_kk,  #
                                  stride_vz, stride_vh, stride_vn, stride_vk,  #
                                  stride_oz, stride_oh, stride_om, stride_on,  #
                                  SM_SCALE: tl.constexpr,  #
                                  SEQLEN_Q: tl.constexpr,  #
                                  SEQLEN_K: tl.constexpr,  #
                                  BLOCK_M: tl.constexpr,  #
                                  BLOCK_N: tl.constexpr,  #
                                  HEAD_SZ: tl.constexpr,  #
                                  ):
    NUM_BUFFERS: tl.constexpr = 2
    scale_ln2: tl.constexpr = SM_SCALE * RCP_LN2

    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_m = tl.program_id(2) * BLOCK_M

    # --- Q: TDM-load once into LDS, then local_load into the dot-operand
    # layout directly (no register-layout conversion in the prologue). ---
    q_desc = tl.make_tensor_descriptor(
        q_ptr + off_z * stride_qz + off_h * stride_qh,
        shape=[SEQLEN_Q, HEAD_SZ],
        strides=[stride_qm, tl.constexpr(1)],
        block_shape=[BLOCK_M, HEAD_SZ],
    )
    q_buf = tlx.local_alloc((BLOCK_M, HEAD_SZ), tlx.dtype_of(q_ptr), 1)
    tlx.async_amd_descriptor_load(q_desc, tlx.local_view(q_buf, 0), [off_m, 0], clamp_bounds=False)
    tlx.async_amd_descriptor_wait(0)
    q = tlx.local_load(tlx.local_view(q_buf, 0))

    # --- K / V TDM descriptors (block = [BLOCK_N, HEAD_SZ]) ---
    k_desc = tl.make_tensor_descriptor(
        k_ptr + off_z * stride_kz + off_h * stride_kh,
        shape=[SEQLEN_K, HEAD_SZ],
        strides=[stride_kn, tl.constexpr(1)],
        block_shape=[BLOCK_N, HEAD_SZ],
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr + off_z * stride_vz + off_h * stride_vh,
        shape=[SEQLEN_K, HEAD_SZ],
        strides=[stride_vn, tl.constexpr(1)],
        block_shape=[BLOCK_N, HEAD_SZ],
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr + off_z * stride_oz + off_h * stride_oh,
        shape=[SEQLEN_Q, HEAD_SZ],
        strides=[stride_om, tl.constexpr(1)],
        block_shape=[BLOCK_M, HEAD_SZ],
    )
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_SZ), tlx.dtype_of(k_ptr), NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_SZ), tlx.dtype_of(v_ptr), NUM_BUFFERS)
    o_buf = tlx.local_alloc((BLOCK_M, HEAD_SZ), tlx.dtype_of(o_ptr), 1)

    ITERS_IN_PROLOGUE_EPILOGUE: tl.constexpr = 3
    n_blocks_n = max((SEQLEN_K + BLOCK_N - 1) // BLOCK_N - ITERS_IN_PROLOGUE_EPILOGUE, 1)
    has_remainder: tl.constexpr = SEQLEN_K < (ITERS_IN_PROLOGUE_EPILOGUE * BLOCK_N)
    if has_remainder:
        n_blocks_n = n_blocks_n - 1

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SZ], dtype=tl.float32)

    block_min = 0
    block_max = n_blocks_n * BLOCK_N

    # ---------------- Prologue ----------------
    tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [0, 0], clamp_bounds=False)
    tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 1), [BLOCK_N, 0], clamp_bounds=False)
    tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 0), [0, 0], clamp_bounds=False)

    k = _load_k(k_buf, 0, 2)
    qk = _compute_qk_no_mask(q, k)
    p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)

    tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [2 * BLOCK_N, 0], clamp_bounds=False)
    tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 1), [BLOCK_N, 0], clamp_bounds=False)

    k = _load_k(k_buf, 1, 3)

    iter_id = 0
    # ---------------- Steady state (hot loop, no masking) ----------------
    # Unroll by 2 so the scheduler can hoist softmax VALU/TRANS work across
    # the back-edge into adjacent WMMA shadows (better co-execution packing).
    for block_id in tl.range(block_min, block_max, BLOCK_N, loop_unroll_factor=2):
        t_2 = block_id + 2 * BLOCK_N
        t_3 = block_id + 3 * BLOCK_N

        qk = _compute_qk_no_mask(q, k)
        p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
        v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0],
                                      clamp_bounds=False)
        acc = tl.dot(p, v, acc)
        p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)
        k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0],
                                      clamp_bounds=False)
        iter_id += 1

    # ---------------- Remainder (masked steady iter) ----------------
    if has_remainder:
        t_1 = iter_id * BLOCK_N + BLOCK_N
        t_2 = iter_id * BLOCK_N + 2 * BLOCK_N
        t_3 = iter_id * BLOCK_N + 3 * BLOCK_N

        qk = _compute_qk(q, k, t_1, BLOCK_M, BLOCK_N, SEQLEN_K)
        p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
        v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0],
                                      clamp_bounds=False)
        acc = tl.dot(p, v, acc)
        p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)
        k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0],
                                      clamp_bounds=False)
        iter_id += 1

    # ---------------- Epilogue ----------------
    epilogue_offset = (iter_id - 1) * BLOCK_N
    t_2 = epilogue_offset + 2 * BLOCK_N
    t_3 = epilogue_offset + 3 * BLOCK_N

    p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
    v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
    acc = tl.dot(p, v, acc)

    qk = _compute_qk(q, k, t_2, BLOCK_M, BLOCK_N, SEQLEN_K)
    p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)

    k = _load_k(k_buf, iter_id % NUM_BUFFERS, 1)
    tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_3, 0], clamp_bounds=False)

    qk = _compute_qk(q, k, t_3, BLOCK_M, BLOCK_N, SEQLEN_K)
    p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
    v = _load_v(v_buf, (iter_id + 1) % NUM_BUFFERS, 1)
    acc = tl.dot(p, v, acc)

    p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)
    p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
    v = _load_v(v_buf, iter_id % NUM_BUFFERS, 0)
    acc = tl.dot(p, v, acc)

    # ---------------- Output ----------------
    l_recip = 1.0 / l_i[:, None]
    acc = acc * l_recip
    # TDM store via LDS: acc -> LDS (native ds_write from the WMMA layout)
    # -> global via TDM. Avoids the 128-way global_store fan-out of tl.store
    # on the WMMA accumulator.
    o_view = tlx.local_view(o_buf, 0)
    tlx.local_store(o_view, acc.to(o_ptr.dtype.element_ty))
    tlx.async_amd_descriptor_store(o_desc, o_view, [off_m, 0], clamp_bounds=False)
    tlx.async_amd_descriptor_wait(0)


_DEFAULT_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "num_warps": 4,
    "waves_per_eu": 1,
}


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale=None, config=None) -> torch.Tensor:
    """Compute non-causal scaled dot-product attention with the gfx1250 TDM pipeline.

    Inputs use ``[batch, heads, sequence, head_size]`` layout. Q, K, and V
    must be BF16 tensors with matching batch/head/head-size dimensions and a
    contiguous innermost dimension. The result is accumulated and returned in
    FP32.
    """
    cfg = dict(_DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, and V must be rank-4 [batch, heads, sequence, head_size] tensors")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q, K, and V must all have dtype torch.bfloat16")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")

    batch, num_heads, seqlen_q, head_size = q.shape
    batch_k, num_k_heads, seqlen_k, k_head_size = k.shape
    if v.shape != (batch_k, num_k_heads, seqlen_k, k_head_size):
        raise ValueError(f"K and V shapes must match, got K={tuple(k.shape)} and V={tuple(v.shape)}")
    if (batch_k, num_k_heads, k_head_size) != (batch, num_heads, head_size):
        raise ValueError("Q, K, and V must have matching batch, head, and head-size dimensions")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Q, K, and V must have a contiguous innermost dimension")

    block_m = cfg["BLOCK_M"]
    block_n = cfg["BLOCK_N"]
    if block_m != 128 or block_n != 128 or head_size != 128:
        raise ValueError("the validated gfx1250 schedule requires BLOCK_M=BLOCK_N=HEAD_SZ=128")
    if seqlen_q <= 0 or seqlen_q % block_m != 0:
        raise ValueError("Q sequence length must be a positive multiple of BLOCK_M")
    if seqlen_k < 3 * block_n:
        raise ValueError("K/V sequence length must contain at least three BLOCK_N tiles")
    if cfg["num_warps"] != 4 or cfg["waves_per_eu"] != 1:
        raise ValueError("the validated gfx1250 schedule requires num_warps=4 and waves_per_eu=1")

    if sm_scale is None:
        sm_scale = head_size**-0.5

    output = torch.empty_like(q, dtype=torch.float32)
    grid = (batch, num_heads, triton.cdiv(seqlen_q, block_m))
    attn_fwd_tdm_pipelined_kernel[grid](
        q,
        k,
        v,
        output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        SM_SCALE=sm_scale,
        SEQLEN_Q=seqlen_q,
        SEQLEN_K=seqlen_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_SZ=head_size,
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
    )
    return output
