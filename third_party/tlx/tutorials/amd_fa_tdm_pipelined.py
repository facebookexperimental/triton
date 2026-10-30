"""
TDM-pipelined Flash-Attention forward for AMD gfx1250 (TLX).

Uses the gfx1250 TDM (tensor-descriptor) async-copy engine to stream K/V
tiles from global memory into double-buffered LDS, with WMMA matmuls via
``tl.dot`` and an FP32 accumulator. Supports BF16/FP16 dense noncausal MHA.

Key TLX primitives:
  tl.make_tensor_descriptor                       -> TDM tensor descriptor
  tlx.local_alloc(blk, dt, n)                     -> n-buffered LDS allocation
  tlx.async_amd_descriptor_load(desc, view, off)  -> TDM async copy g->LDS
  tlx.async_amd_descriptor_wait(n)                -> wait until <= n TDM ops in flight
  tlx.local_load(tlx.local_trans(view))           -> LDS read w/ memdesc transpose (K)
  tlx.local_load(view)                            -> LDS read (V)
  tl.dot(a, b, c)                                  -> WMMA matmul-accumulate

The software pipeline is hand-written (no auto-pipeliner): 3 iters are
peeled across the prologue+epilogue with a steady-state hot loop in
between, and K/V are double-buffered in LDS via TDM async copies.

For sufficiently long D128 inputs, the fast pass keeps the first key tile's
row maximum and defers denominator reductions. It uses RNE probabilities in
the input dtype and retries a workgroup with stable online softmax if its
shift, denominator, or output is unsafe. Recovery happens on the device on
every invocation, including graph replay. Short inputs retain online softmax.
"""
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

RCP_LN2 = tl.constexpr(1.4426950408889634)


def is_gfx1250_available():
    try:
        target = triton.runtime.driver.active.get_current_target()
        return target.arch == "gfx1250"
    except Exception:
        return False


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
        # Retain two partial sums per row; exchange lanes only at the end.
        terms = tl.reshape(p, [p.shape[0], p.shape[1] // 16, 2, 8])
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
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_SZ: tl.constexpr, FIXED_REFERENCE: tl.constexpr):
    NUM_BUFFERS: tl.constexpr = 2
    # The 128x64 fixed-reference schedule hides K-prefetch latency behind
    # probability conversion and denominator updates; wider tiles prefer the
    # original ordering to limit live operands.
    PREFETCH_BEFORE_P: tl.constexpr = FIXED_REFERENCE and BLOCK_M == 128 and BLOCK_N == 64
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
    tlx.async_amd_descriptor_load(q_desc, tlx.local_view(q_buf, 0), [off_m, 0], clamp_bounds=True)
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
    if BLOCK_M == 64 and BLOCK_N == 128 and HEAD_SZ == 128:
        # Narrow query tiles benefit from eight-element K-row padding.
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
        l_i = tl.zeros([BLOCK_M, 2], dtype=tl.float32)
    else:
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SZ], dtype=tl.float32)

    if SEQLEN_K <= 2 * BLOCK_N:
        # One or two key tiles do not amortize the long pipeline's fill/drain.
        for key_start in tl.static_range(0, SEQLEN_K, BLOCK_N):
            tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [key_start, 0], clamp_bounds=True)
            tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 0), [key_start, 0], clamp_bounds=True)
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

        # ---------------- Prologue ----------------
        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [0, 0], clamp_bounds=True)
        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 1), [BLOCK_N, 0], clamp_bounds=True)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 0), [0, 0], clamp_bounds=True)

        k = _load_k(k_buf, 0, 2)
        qk = _compute_qk(q, k, 0, BLOCK_M, BLOCK_N, SEQLEN_K)
        p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)

        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [2 * BLOCK_N, 0], clamp_bounds=True)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 1), [BLOCK_N, 0], clamp_bounds=True)

        k = _load_k(k_buf, 1, 3)

        iter_id = 0
        # ---------------- Steady state (hot loop, no masking) ----------------
        # The fixed-reference loop benefits from fewer live iterations. Retain
        # the original short online schedule where the extra state is affordable.
        unroll: tl.constexpr = 2 if not FIXED_REFERENCE and SEQLEN_K < 512 else 1
        for block_id in tl.range(block_min, block_max, BLOCK_N, loop_unroll_factor=unroll):
            t_2 = block_id + 2 * BLOCK_N
            t_3 = block_id + 3 * BLOCK_N

            qk = _compute_qk_no_mask(q, k)
            if not PREFETCH_BEFORE_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
            v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
            tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0],
                                          clamp_bounds=True)
            if PREFETCH_BEFORE_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
            acc = tl.dot(p, v, acc)
            p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
            k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
            tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0],
                                          clamp_bounds=True)
            iter_id += 1

        # ---------------- Remainder (masked steady iter) ----------------
        if has_remainder:
            t_1 = iter_id * BLOCK_N + BLOCK_N
            t_2 = iter_id * BLOCK_N + 2 * BLOCK_N
            t_3 = iter_id * BLOCK_N + 3 * BLOCK_N

            qk = _compute_qk(q, k, t_1, BLOCK_M, BLOCK_N, SEQLEN_K)
            if not PREFETCH_BEFORE_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
            v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
            tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0],
                                          clamp_bounds=True)
            if PREFETCH_BEFORE_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
            acc = tl.dot(p, v, acc)
            p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
            k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
            tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0],
                                          clamp_bounds=True)
            iter_id += 1

        # ---------------- Epilogue ----------------
        epilogue_offset = (iter_id - 1) * BLOCK_N
        t_2 = epilogue_offset + 2 * BLOCK_N
        t_3 = epilogue_offset + 3 * BLOCK_N

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

    # ---------------- Output ----------------
    if FIXED_REFERENCE:
        l_i = tl.sum(l_i, 1)
    l_recip = 1.0 / l_i[:, None]
    acc = acc * l_recip
    # TDM store via LDS: acc -> LDS (native ds_write from the WMMA layout)
    # -> global via TDM. Avoids the 128-way global_store fan-out of tl.store
    # on the WMMA accumulator.
    o_view = tlx.local_view(o_buf, 0)
    tlx.local_store(o_view, acc.to(o_ptr.dtype.element_ty))
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
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_SZ: tl.constexpr):
    _attention_pass(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
                    stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om,
                    stride_on, SM_SCALE, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ, False)


@triton.jit
def attn_fwd_tdm_pipelined_kernel(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
                                  stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk,
                                  stride_oz, stride_oh, stride_om, stride_on, SM_SCALE: tl.constexpr,
                                  SEQLEN_Q: tl.constexpr, SEQLEN_K: tl.constexpr, BLOCK_M: tl.constexpr,
                                  BLOCK_N: tl.constexpr, HEAD_SZ: tl.constexpr,
                                  USE_FIXED_REFERENCE: tl.constexpr = False):
    if USE_FIXED_REFERENCE:
        retry = _attention_pass(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
                                stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz,
                                stride_oh, stride_om, stride_on, SM_SCALE, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N,
                                HEAD_SZ, True)
        if retry:
            _online_replay(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
                           stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh,
                           stride_om, stride_on, SM_SCALE, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ)
    else:
        _attention_pass(q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
                        stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh,
                        stride_om, stride_on, SM_SCALE, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ, False)


def _select_tile_sizes(batch, heads, seqlen_q, seqlen_k, head_size, compute_units):
    if head_size != 128:
        return 128, 128
    # Smaller query tiles fill otherwise idle CUs on short inputs.
    query_blocks = batch * heads * triton.cdiv(seqlen_q, 128)
    if seqlen_k <= 1024 and query_blocks < compute_units:
        return 64, 128
    if seqlen_k < 512:
        return 128, 128
    # Wider query tiles amortize K/V traffic, provided enough workgroups remain
    # to fill the GPU. Shorter inputs use the spill-free 128x64 schedule.
    wide = seqlen_k >= 4096 and batch * heads * (seqlen_q // 256) >= compute_units
    return (256 if wide else 128), 64


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
    attn_fwd_tdm_pipelined_kernel[grid](
        q, k, v, o,  #
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),  #
        sm_scale, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ, use_fixed,  #
        num_warps=4, waves_per_eu=1)
    return o


if __name__ == "__main__":
    if not is_gfx1250_available():
        raise SystemExit("Requires gfx1250")
    torch.manual_seed(0)
    q = torch.randn((1, 8, 1024, 128), device=triton.runtime.driver.active.get_active_torch_device(),
                    dtype=torch.bfloat16)
    k = torch.randn((1, 8, 1024, 128), device=triton.runtime.driver.active.get_active_torch_device(),
                    dtype=torch.bfloat16)
    v = torch.randn((1, 8, 1024, 128), device=triton.runtime.driver.active.get_active_torch_device(),
                    dtype=torch.bfloat16)
    out = attn_fwd_tdm_pipelined(q, k, v, 1.0 / (128**0.5))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v).to(torch.float32)
    print("max abs diff:", (out.cpu() - ref.cpu()).abs().max().item())
