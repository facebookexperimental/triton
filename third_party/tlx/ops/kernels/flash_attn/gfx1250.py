"""gfx1250 TDM/WMMA attention with device-side online-softmax recovery."""

import math

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import InvalidInput

RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _load_k(k_buf, slot, wait_count):
    tlx.async_amd_descriptor_wait(wait_count)
    return tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, slot)))


@triton.jit
def _load_v(v_buf, slot, wait_count):
    tlx.async_amd_descriptor_wait(wait_count)
    value = tlx.local_load(tlx.local_view(v_buf, slot))
    # All waves must finish reading the current K/V stages before any wave
    # can issue the next prefetch into their reusable LDS buffers.
    tlx.workgroup_barrier()
    return value


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
    q_shifted = (qk - m_ij[:, None]) * scale_ln2
    p = tl.math.exp2(q_shifted)
    alpha = tl.math.exp2((m_i - m_ij) * scale_ln2)
    return p, alpha, m_ij


@triton.jit
def _softmax_part1(p, l_i, acc, alpha, P_DTYPE: tl.constexpr, FIXED_REFERENCE: tl.constexpr):
    if FIXED_REFERENCE:
        parts: tl.constexpr = l_i.shape[1]
        terms = tl.reshape(p, [p.shape[0], p.shape[1] // (parts * 8), parts, 8])
        terms = tl.sum(tl.sum(terms, 3), 1)
        l_i += terms
    else:
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
    p_low = p.to(P_DTYPE)
    return p_low, l_i, acc


@triton.jit
def _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE: tl.constexpr):
    if FIXED_REFERENCE:
        p = tl.math.exp2(scale_ln2 * qk - (m_i * scale_ln2)[:, None])
        alpha = tl.full(m_i.shape, 1.0, tl.float32)
        return p, alpha, m_i
    else:
        return _softmax_part0(qk, m_i, scale_ln2)


@triton.jit
def _needs_online_replay(acc, denominator, reference, scale_ln2, row_start, SEQLEN_Q: tl.constexpr):
    # Each valid row has a unit contribution from the first tile. Reject lost
    # normalization or overflow, including overflow when P is converted to FP16.
    max_denominator: tl.constexpr = 1.e30  # Leave FP32 headroom for normalization.
    max_reference_shift: tl.constexpr = 64.0  # Bound scale-minus-shift cancellation.
    fp32_max: tl.constexpr = 3.402823466e38
    bad_output = tl.sum((~(tl.abs(acc) <= fp32_max)).to(tl.int32), 1) > 0
    bad_row = (~((denominator > 0) & (denominator < max_denominator)))
    bad_row |= (tl.abs(reference * scale_ln2) > max_reference_shift) | bad_output
    valid = row_start + tl.arange(0, acc.shape[0]) < SEQLEN_Q
    return tl.sum((valid & bad_row).to(tl.int32), 0) > 0


@triton.jit
def _attention_pass(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
                    stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om,
                    stride_on, SM_SCALE: tl.constexpr, SEQLEN_Q: tl.constexpr, SEQLEN_K: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_SZ: tl.constexpr, FIXED_REFERENCE: tl.constexpr,
                    QUERY_TILES_FIRST: tl.constexpr):
    NUM_BUFFERS: tl.constexpr = 2
    # Carry low-precision P across iterations to shorten FP32 live ranges.
    PREPARE_NEXT_P: tl.constexpr = FIXED_REFERENCE and BLOCK_M == 128 and BLOCK_N == 64
    # Overlap initial Q/K/V transfers where the extra LDS lifetime pays off.
    PREFETCH_Q: tl.constexpr = HEAD_SZ == 128 and (BLOCK_M == 32 or
                                                   (BLOCK_M == 64 and SEQLEN_K < 512)) and SEQLEN_K <= 768
    scale_ln2: tl.constexpr = SM_SCALE * RCP_LN2

    off_z = tl.program_id(0).to(tl.int64)
    if QUERY_TILES_FIRST:
        query_blocks: tl.constexpr = tl.cdiv(SEQLEN_Q, BLOCK_M)
        tile = tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        off_h = tile // query_blocks
        off_m = (tile % query_blocks) * BLOCK_M
    else:
        off_h = tl.program_id(1)
        off_m = tl.program_id(2) * BLOCK_M

    off_h = off_h.to(tl.int64)
    q_desc = tl.make_tensor_descriptor(
        q_ptr + off_z * stride_qz + off_h * stride_qh,
        shape=[SEQLEN_Q, HEAD_SZ],
        strides=[stride_qm, tl.constexpr(1)],
        block_shape=[BLOCK_M, HEAD_SZ],
    )
    q_buf = tlx.local_alloc((BLOCK_M, HEAD_SZ), tlx.dtype_of(q_ptr), 1)
    tlx.async_amd_descriptor_load(q_desc, tlx.local_view(q_buf, 0), [off_m, 0], clamp_bounds=True)
    if not PREFETCH_Q:
        tlx.async_amd_descriptor_wait(0)
        q = tlx.local_load(tlx.local_view(q_buf, 0))

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
    if HEAD_SZ == 128:
        # Pad K rows for the transposed WMMA operand loads.
        k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(HEAD_SZ, 8)], [BLOCK_N, HEAD_SZ],
                                                                                     [1, 0])
        k_buf = tlx.local_alloc((BLOCK_N, HEAD_SZ), tlx.dtype_of(k_ptr), NUM_BUFFERS, layout=k_layout)
    else:
        k_buf = tlx.local_alloc((BLOCK_N, HEAD_SZ), tlx.dtype_of(k_ptr), NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_SZ), tlx.dtype_of(v_ptr), NUM_BUFFERS)
    o_buf = tlx.local_alloc((BLOCK_M, HEAD_SZ), tlx.dtype_of(o_ptr), 1)

    ITERS_IN_PROLOGUE_EPILOGUE: tl.constexpr = 3
    n_blocks_n = max((SEQLEN_K + BLOCK_N - 1) // BLOCK_N - ITERS_IN_PROLOGUE_EPILOGUE, 1)
    has_remainder: tl.constexpr = SEQLEN_K < (ITERS_IN_PROLOGUE_EPILOGUE * BLOCK_N)
    if has_remainder:
        n_blocks_n = n_blocks_n - 1

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if FIXED_REFERENCE:
        # Wider query tiles benefit from packed updates to four partial sums.
        parts: tl.constexpr = min(4 if BLOCK_M >= 128 else 2, BLOCK_N // 8)
        l_i = tl.zeros([BLOCK_M, parts], dtype=tl.float32)
    else:
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SZ], dtype=tl.float32)

    if SEQLEN_K <= 2 * BLOCK_N:
        for key_start in tl.static_range(0, SEQLEN_K, BLOCK_N):
            tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [key_start, 0], clamp_bounds=True)
            tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 0), [key_start, 0], clamp_bounds=True)
            if PREFETCH_Q and key_start == 0:
                tlx.async_amd_descriptor_wait(2)
                q = tlx.local_load(tlx.local_view(q_buf, 0))
            k = _load_k(k_buf, 0, 1)
            qk = _compute_qk(q, k, key_start, BLOCK_M, BLOCK_N, SEQLEN_K)
            if key_start == 0:
                p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)
            else:
                p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
            p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
            v = _load_v(v_buf, 0, 0)
            acc = tl.dot(p, v, acc)
    else:
        block_min = 0
        block_max = n_blocks_n * BLOCK_N

        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [0, 0], clamp_bounds=True)
        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 1), [BLOCK_N, 0], clamp_bounds=True)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 0), [0, 0], clamp_bounds=True)

        if PREFETCH_Q:
            tlx.async_amd_descriptor_wait(3)
            q = tlx.local_load(tlx.local_view(q_buf, 0))
        k = _load_k(k_buf, 0, 2)
        qk = _compute_qk(q, k, 0, BLOCK_M, BLOCK_N, SEQLEN_K)
        p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)

        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [2 * BLOCK_N, 0], clamp_bounds=True)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 1), [BLOCK_N, 0], clamp_bounds=True)

        k = _load_k(k_buf, 1, 3)

        if PREPARE_NEXT_P:
            p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, True)
        iter_id = 0
        unroll: tl.constexpr = 2 if not FIXED_REFERENCE and SEQLEN_K < 512 else 1
        for block_id in tl.range(block_min, block_max, BLOCK_N, loop_unroll_factor=unroll):
            t_2 = block_id + 2 * BLOCK_N
            t_3 = block_id + 3 * BLOCK_N

            qk = _compute_qk_no_mask(q, k)
            if not PREPARE_NEXT_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
            v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
            tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0],
                                          clamp_bounds=True)
            acc = tl.dot(p, v, acc)
            p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
            if PREPARE_NEXT_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, True)
            k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
            tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0],
                                          clamp_bounds=True)
            iter_id += 1

        if has_remainder:
            t_1 = iter_id * BLOCK_N + BLOCK_N
            t_2 = iter_id * BLOCK_N + 2 * BLOCK_N
            t_3 = iter_id * BLOCK_N + 3 * BLOCK_N

            qk = _compute_qk(q, k, t_1, BLOCK_M, BLOCK_N, SEQLEN_K)
            if not PREPARE_NEXT_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
            v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
            tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0],
                                          clamp_bounds=True)
            acc = tl.dot(p, v, acc)
            p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
            if PREPARE_NEXT_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, True)
            k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
            tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0],
                                          clamp_bounds=True)
            iter_id += 1

        epilogue_offset = (iter_id - 1) * BLOCK_N
        t_2 = epilogue_offset + 2 * BLOCK_N
        t_3 = epilogue_offset + 3 * BLOCK_N

        if not PREPARE_NEXT_P:
            p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
        v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
        acc = tl.dot(p, v, acc)

        qk = _compute_qk(q, k, t_2, BLOCK_M, BLOCK_N, SEQLEN_K)
        p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)

        k = _load_k(k_buf, iter_id % NUM_BUFFERS, 1)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_3, 0], clamp_bounds=True)

        qk = _compute_qk(q, k, t_3, BLOCK_M, BLOCK_N, SEQLEN_K)
        p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
        v = _load_v(v_buf, (iter_id + 1) % NUM_BUFFERS, 1)
        acc = tl.dot(p, v, acc)

        p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
        p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
        v = _load_v(v_buf, iter_id % NUM_BUFFERS, 0)
        acc = tl.dot(p, v, acc)

    if FIXED_REFERENCE:
        l_i = tl.sum(l_i, 1)
    l_recip = 1.0 / l_i[:, None]
    acc = acc * l_recip
    o_view = tlx.local_view(o_buf, 0)
    output = acc
    if o_ptr.dtype.element_ty == tl.float16:
        # INVARIANT: convex combinations of finite FP16 values cannot exceed the FP16 range.
        output = tl.where(tl.abs(output) < float("inf"), tl.minimum(tl.maximum(output, -65504.0), 65504.0), output)
    tlx.local_store(o_view, output.to(o_ptr.dtype.element_ty))
    tlx.async_amd_descriptor_store(o_desc, o_view, [off_m, 0], clamp_bounds=True)
    tlx.async_amd_descriptor_wait(0)

    if FIXED_REFERENCE:
        # The completed store above is provisional: recovery overwrites it.
        return _needs_online_replay(acc, l_i, m_i, scale_ln2, off_m, SEQLEN_Q)
    return False


# Keep uncommon recovery out of the fast loop's register schedule.
@triton.jit(noinline=True)
def _online_replay(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
                   stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om,
                   stride_on, SM_SCALE: tl.constexpr, SEQLEN_Q: tl.constexpr, SEQLEN_K: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_SZ: tl.constexpr,
                   QUERY_TILES_FIRST: tl.constexpr):
    _attention_pass(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
                    stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om,
                    stride_on, SM_SCALE, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ, False, QUERY_TILES_FIRST)


@triton.jit
def attn_fwd_tdm_pipelined_kernel(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
                                  stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk,
                                  stride_oz, stride_oh, stride_om, stride_on, SM_SCALE: tl.constexpr,
                                  SEQLEN_Q: tl.constexpr, SEQLEN_K: tl.constexpr, BLOCK_M: tl.constexpr,
                                  BLOCK_N: tl.constexpr, HEAD_SZ: tl.constexpr,
                                  USE_FIXED_REFERENCE: tl.constexpr = False, QUERY_TILES_FIRST: tl.constexpr = False,
                                  BATCH: tl.constexpr = 1, NUM_HEADS: tl.constexpr = 1):
    if USE_FIXED_REFERENCE:
        retry = _attention_pass(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
                                stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz,
                                stride_oh, stride_om, stride_on, SM_SCALE, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N,
                                HEAD_SZ, True, QUERY_TILES_FIRST)
        if retry:
            _online_replay(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
                           stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh,
                           stride_om, stride_on, SM_SCALE, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ,
                           QUERY_TILES_FIRST)
    else:
        _attention_pass(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
                        stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh,
                        stride_om, stride_on, SM_SCALE, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ, False,
                        QUERY_TILES_FIRST)


def _select_tile_sizes(batch, heads, seqlen_q, seqlen_k, head_size, compute_units):
    if head_size != 128:
        return 128, 128
    query_blocks = batch * heads * triton.cdiv(seqlen_q, 128)
    if query_blocks < compute_units:
        narrow_blocks = batch * heads * triton.cdiv(seqlen_q, 64)
        if seqlen_k > 1024:
            return (32 if narrow_blocks <= compute_units // 2 else 64), 128
        if seqlen_k >= 512 and narrow_blocks < compute_units:
            if seqlen_k <= 768 or narrow_blocks <= compute_units // 8:
                return 32, 128
        return 64, 128
    if seqlen_k < 512:
        return 128, 128
    wide = seqlen_k >= 4096 and batch * heads * (seqlen_q // 256) >= compute_units
    return (256 if wide else 128), 64


def _use_query_tiles_first(heads, seqlen_k, head_size):
    return head_size == 128 and seqlen_k >= 65536 and heads >= 32


def attn_fwd_tdm_pipelined(q, k, v, sm_scale, BLOCK_M=None, BLOCK_N=None):
    """Dense noncausal BHSD attention with BF16/FP16 inputs and FP32 output.

    Q/K/V share batch, head count and head size; K/V share sequence length.
    The head dimension must be contiguous and sm_scale positive. Explicit
    tile overrides retain the original default of 128 for the other dimension.
    """
    BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ = q.shape
    SEQLEN_K = k.shape[2]
    if BLOCK_M is None and BLOCK_N is None:
        compute_units = torch.cuda.get_device_properties(q.device).multi_processor_count
        BLOCK_M, BLOCK_N = _select_tile_sizes(BATCH, NUM_Q_HEADS, SEQLEN_Q, SEQLEN_K, HEAD_SZ, compute_units)
    else:
        BLOCK_M = 128 if BLOCK_M is None else BLOCK_M
        BLOCK_N = 128 if BLOCK_N is None else BLOCK_N
    use_fixed = HEAD_SZ == 128 and SEQLEN_K >= 512
    o = torch.empty_like(q, dtype=torch.float32)
    grid = (BATCH, NUM_Q_HEADS, (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M)
    attn_fwd_tdm_pipelined_kernel[grid](q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(), sm_scale,
                                        SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ, use_fixed,
                                        QUERY_TILES_FIRST=_use_query_tiles_first(NUM_Q_HEADS, SEQLEN_K, HEAD_SZ),
                                        num_warps=2 if HEAD_SZ == 128 and BLOCK_M == 32 else 4, waves_per_eu=1)
    return o


_CONFIGS = tuple(
    triton.Config({"BLOCK_M": m, "BLOCK_N": n, "waves_per_eu": 1}, num_warps=warps)
    for m, n, warps in ((32, 128, 2), (64, 128, 4), (128, 128, 4), (128, 64, 4), (256, 64, 4)))


def _prune_configs(configs, named_args, **kwargs):
    max_rows = max(32, triton.next_power_of_2(named_args["SEQLEN_Q"]))
    return [config for config in configs if config.kwargs["BLOCK_M"] <= max_rows]


_autotuned_attention = triton.autotune(
    configs=list(_CONFIGS),
    key=[
        "BATCH", "NUM_HEADS", "SEQLEN_Q", "SEQLEN_K", "SM_SCALE", "stride_qz", "stride_qh", "stride_qm", "stride_kz",
        "stride_kh", "stride_kn", "stride_vz", "stride_vh", "stride_vn"
    ],
    prune_configs_by={"early_config_prune": _prune_configs},
)(attn_fwd_tdm_pipelined_kernel)


def heuristic_config(batch, heads, seqlen_q, seqlen_k, compute_units):
    m, n = _select_tile_sizes(batch, heads, seqlen_q, seqlen_k, 128, compute_units)
    return next(config for config in _CONFIGS if (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"]) == (m, n))


def _validate_inputs(q, k, v, causal, sm_scale, space):
    if space not in ("heuristic", "full", "smoke"):
        raise InvalidInput(f"gfx1250 flash_attn does not provide space={space!r}")
    if causal:
        raise InvalidInput("gfx1250 flash_attn supports noncausal attention only")
    if any(t.ndim != 4 for t in (q, k, v)):
        raise InvalidInput("gfx1250 flash_attn expects rank-4 BHSD tensors")
    if not q.is_cuda or any(t.device != q.device for t in (k, v)):
        raise InvalidInput("gfx1250 flash_attn expects Q/K/V on one GPU")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != q.dtype for t in (k, v)):
        raise InvalidInput("gfx1250 flash_attn expects matching FP16 or BF16 dtypes")
    if q.shape[-1] != 128 or k.shape[:2] != q.shape[:2] or k.shape[-1] != 128 or v.shape != k.shape:
        raise InvalidInput("gfx1250 flash_attn expects MHA with head dimension 128 and matching K/V shapes")
    if any(min(t.shape) <= 0 for t in (q, k, v)):
        raise InvalidInput("gfx1250 flash_attn expects nonempty tensors")
    if any(t.stride(-1) != 1 or min(t.stride()) <= 0 for t in (q, k, v)):
        raise InvalidInput("gfx1250 flash_attn requires positive strides and contiguous head dimensions")
    if not isinstance(sm_scale, (int, float)) or not math.isfinite(sm_scale) or sm_scale <= 0:
        raise InvalidInput("gfx1250 flash_attn requires a positive finite sm_scale")


def flash_attn(q, k, v, causal=False, sm_scale=None, *, space="heuristic"):
    """Dense noncausal D128 MHA; output has the input shape and dtype."""
    sm_scale = q.shape[-1]**-0.5 if sm_scale is None and q.ndim == 4 else sm_scale
    _validate_inputs(q, k, v, causal, sm_scale, space)
    batch, heads, sq, dim = q.shape
    sk = k.shape[2]
    output = torch.empty_like(q)
    grid = lambda meta: (batch, heads, triton.cdiv(sq, meta["BLOCK_M"]))
    options = dict(USE_FIXED_REFERENCE=sk >= 512, QUERY_TILES_FIRST=_use_query_tiles_first(heads, sk, dim), BATCH=batch,
                   NUM_HEADS=heads)
    kernel = _autotuned_attention
    if space != "full":
        units = torch.cuda.get_device_properties(q.device).multi_processor_count
        selected = heuristic_config(batch, heads, sq, sk, units)
        options.update(selected.all_kwargs())
        kernel = attn_fwd_tdm_pipelined_kernel
    kernel[grid](q, k, v, output, *q.stride(), *k.stride(), *v.stride(), *output.stride(), sm_scale, sq, sk,
                 HEAD_SZ=dim, **options)
    return output
