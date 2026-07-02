"""AMD CDNA4 Flash Attention forward with a rotated 4-cluster pipeline."""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.core import _aggregate as aggregate

CLUSTER_BUF_DEPTH = 2


@triton.jit
def _assume_strides(
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
):
    tl.assume(stride_qz >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_qm > 0)
    tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_kk >= 0)
    tl.assume(stride_vz >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vk >= 0)
    tl.assume(stride_oz >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_ok >= 0)


@aggregate
class SoftmaxState:
    acc: tl.tensor
    l_i: tl.tensor
    m_i: tl.tensor

    @triton.jit
    def create(BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
        return SoftmaxState(
            tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32),
            tl.full([BLOCK_M], 1.0, dtype=tl.float32),
            tl.full([BLOCK_M], float("-inf"), dtype=tl.float32),
        )

    @triton.jit
    def _nan_max_combine(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def _row_max(qk):
        return tl.reduce(qk, 1, SoftmaxState._nan_max_combine)

    @triton.jit
    def vec1(
        self,
        qk,
        start_n,
        offs_m,
        offs_n,
        QK_SCALE: tl.constexpr,
        DIAG_OFFSET: tl.constexpr,
        MASK_STEPS: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        if MASK_STEPS:
            qk_sm = qk * QK_SCALE
            if IS_CAUSAL:
                kn = start_n + offs_n
                qk_sm = tl.where(offs_m[:, None] + DIAG_OFFSET >= kn[None, :], qk_sm, float("-inf"))
            m_ij = SoftmaxState._row_max(qk_sm)
            m_new = tl.maximum(self.m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp2(qk_sm - m_new[:, None])
            alpha = tl.math.exp2(self.m_i - m_new)
        else:
            m_ij = SoftmaxState._row_max(qk) * QK_SCALE
            m_new = tl.maximum(self.m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp2(qk * QK_SCALE - m_new[:, None])
            alpha = tl.math.exp2(self.m_i - m_new)
        return SoftmaxState(self.acc, self.l_i, m_new), p, alpha

    @triton.jit
    def vec2(self, p, alpha, out_dtype: tl.constexpr):
        l_ij = tl.sum(p, 1)
        acc = self.acc * alpha[:, None]
        l_i = self.l_i * alpha + l_ij
        p_cast = p.to(out_dtype)
        return SoftmaxState(acc, l_i, self.m_i), p_cast


@triton.jit
def _attn_inner_pipelined(
    state,
    q,
    k_ptrs,
    v_ptrs,
    offs_m,
    offs_n,
    block_start,
    block_end,
    k_buf,
    v_buf,
    stride_kn,
    stride_vn,
    QK_SCALE: tl.constexpr,
    DIAG_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Prologue: prime the pipeline for output tile block_start.
    b0 = block_start
    tok_k0 = tlx.async_load(k_ptrs + b0 * BLOCK_N * stride_kn, tlx.local_view(k_buf, 0))
    tok_v0 = tlx.async_load(v_ptrs + b0 * BLOCK_N * stride_vn, tlx.local_view(v_buf, 0))
    tlx.async_load_commit_group([tok_k0])
    tlx.async_load_commit_group([tok_v0])
    tok_k1 = tlx.async_load(k_ptrs + (b0 + 1) * BLOCK_N * stride_kn, tlx.local_view(k_buf, 1))
    tlx.async_load_commit_group([tok_k1])

    wait0 = tlx.async_load_wait_group(2)
    kt0 = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 0)), token=wait0, relaxed=True)
    qk = tl.dot(q, kt0)
    state, p_c, alpha_c = state.vec1(
        qk,
        b0 * BLOCK_N,
        offs_m,
        offs_n,
        QK_SCALE,
        DIAG_OFFSET,
        MASK_STEPS,
        IS_CAUSAL,
    )

    tl.debug_barrier()
    tok_k2 = tlx.async_load(k_ptrs + (b0 + 2) * BLOCK_N * stride_kn, tlx.local_view(k_buf, 0))
    tlx.async_load_commit_group([tok_k2])
    wait1 = tlx.async_load_wait_group(1)
    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 1)), token=wait1, relaxed=True)
    tok_v1 = tlx.async_load(v_ptrs + (b0 + 1) * BLOCK_N * stride_vn, tlx.local_view(v_buf, 1))
    tlx.async_load_commit_group([tok_v1])

    for block_n in tl.range(block_start, block_end - 3, num_stages=0):
        cur_slot = (block_n - block_start) % BUF_DEPTH
        nxt_slot = (block_n + 1 - block_start) % BUF_DEPTH
        ack_n = (block_n + 3) * BLOCK_N
        acv_n = (block_n + 2) * BLOCK_N
        ahead_n = (block_n + 1) * BLOCK_N

        with tlx.warp_pipeline_stage("dot1", priority=0):
            qk = tl.dot(q, kt_dot)
            state, p_dot = state.vec2(p_c, alpha_c, q.dtype)

        tlx.async_load_wait_group(1)

        with tlx.warp_pipeline_stage("mem1", priority=1):
            v_dot = tlx.local_load(tlx.local_view(v_buf, cur_slot), relaxed=True)
            tok_k = tlx.async_load(k_ptrs + ack_n * stride_kn, tlx.local_view(k_buf, nxt_slot))
            tlx.async_load_commit_group([tok_k])

        with tlx.warp_pipeline_stage("dot2", priority=0):
            acc = tl.dot(p_dot, v_dot, state.acc)
            state = SoftmaxState(acc, state.l_i, state.m_i)
            state, p_c, alpha_c = state.vec1(
                qk,
                ahead_n,
                offs_m,
                offs_n,
                QK_SCALE,
                DIAG_OFFSET,
                MASK_STEPS,
                IS_CAUSAL,
            )

        tlx.async_load_wait_group(1)

        with tlx.warp_pipeline_stage("mem2", priority=1):
            kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, cur_slot)), relaxed=True)
            tok_v = tlx.async_load(v_ptrs + acv_n * stride_vn, tlx.local_view(v_buf, cur_slot))
            tlx.async_load_commit_group([tok_v])

    # Drain the last three output tiles without out-of-bounds global prefetches.
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    s_nm3 = (nm3 - block_start) % BUF_DEPTH
    s_nm2 = (nm2 - block_start) % BUF_DEPTH
    s_nm1 = (nm1 - block_start) % BUF_DEPTH

    qk = tl.dot(q, kt_dot)
    tlx.async_load_wait_group(2)
    v_dot = tlx.local_load(tlx.local_view(v_buf, s_nm3), relaxed=True)
    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)
    acc = tl.dot(p_dot, v_dot, state.acc)
    state = SoftmaxState(acc, state.l_i, state.m_i)
    state, p_c, alpha_c = state.vec1(
        qk,
        nm2 * BLOCK_N,
        offs_m,
        offs_n,
        QK_SCALE,
        DIAG_OFFSET,
        MASK_STEPS,
        IS_CAUSAL,
    )
    tl.debug_barrier()
    tok_vlast = tlx.async_load(v_ptrs + nm1 * BLOCK_N * stride_vn, tlx.local_view(v_buf, s_nm1))
    tlx.async_load_commit_group([tok_vlast])
    tlx.async_load_wait_group(2)
    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, s_nm1)), relaxed=True)

    qk = tl.dot(q, kt_dot)
    tlx.async_load_wait_group(1)
    v_dot = tlx.local_load(tlx.local_view(v_buf, s_nm2), relaxed=True)
    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)
    acc = tl.dot(p_dot, v_dot, state.acc)
    state = SoftmaxState(acc, state.l_i, state.m_i)
    state, p_c, alpha_c = state.vec1(
        qk,
        nm1 * BLOCK_N,
        offs_m,
        offs_n,
        QK_SCALE,
        DIAG_OFFSET,
        MASK_STEPS,
        IS_CAUSAL,
    )

    tlx.async_load_wait_group(0)
    v_dot = tlx.local_load(tlx.local_view(v_buf, s_nm1), relaxed=True)
    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)
    acc = tl.dot(p_dot, v_dot, state.acc)
    state = SoftmaxState(acc, state.l_i, state.m_i)

    return state


@triton.jit
def _attn_cluster_tile(
    pid_m,
    off_z,
    off_h,
    Q,
    K,
    V,
    Out,
    k_buf,
    v_buf,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    N_CTX,
    QK_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )

    DIAG_OFFSET: tl.constexpr = 0

    state = SoftmaxState.create(BLOCK_M, HEAD_DIM)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = N_CTX // BLOCK_N

    if IS_CAUSAL:
        masked_blocks: tl.constexpr = BLOCK_M // BLOCK_N
        causal_end = (pid_m + 1) * masked_blocks
        n_full = causal_end - masked_blocks
        if n_full > 0:
            state = _attn_inner_pipelined(
                state,
                q,
                k_ptrs,
                v_ptrs,
                offs_m,
                offs_n,
                0,
                n_full,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                QK_SCALE,
                DIAG_OFFSET,
                BLOCK_N,
                BUF_DEPTH,
                False,
                True,
            )
        state = _attn_inner_pipelined(
            state,
            q,
            k_ptrs,
            v_ptrs,
            offs_m,
            offs_n,
            n_full,
            causal_end,
            k_buf,
            v_buf,
            stride_kn,
            stride_vn,
            QK_SCALE,
            DIAG_OFFSET,
            BLOCK_N,
            BUF_DEPTH,
            True,
            True,
        )
    else:
        state = _attn_inner_pipelined(
            state,
            q,
            k_ptrs,
            v_ptrs,
            offs_m,
            offs_n,
            0,
            n_blocks,
            k_buf,
            v_buf,
            stride_kn,
            stride_vn,
            QK_SCALE,
            DIAG_OFFSET,
            BLOCK_N,
            BUF_DEPTH,
            False,
            False,
        )

    acc = state.acc / state.l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _attn_fwd_cluster_pipeline(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    _assume_strides(
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
    )

    if IS_CAUSAL:
        off_h = tl.program_id(0)
        pid_m = tl.program_id(1)
    else:
        pid_m = tl.program_id(0)
        off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, BUF_DEPTH)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, BUF_DEPTH)
    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089
    _attn_cluster_tile(
        pid_m,
        off_z,
        off_h,
        Q,
        K,
        V,
        Out,
        k_buf,
        v_buf,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        N_CTX,
        QK_SCALE,
        BLOCK_M,
        BLOCK_N,
        BUF_DEPTH,
        HEAD_DIM,
        IS_CAUSAL,
    )


@triton.jit
def _attn_fwd_cluster_persistent_pipeline(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    _assume_strides(
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
    )

    # Persistent scheduler: pin flattened (batch, head) work to an XCD for K/V
    # locality, then round-robin constant-cost work units across local programs.
    # Causal units bundle zig-zag tile pairs; non-causal units are one tile.
    pid = tl.program_id(0)
    xcd = pid % NUM_XCDS
    local = pid // NUM_XCDS
    NUM_LOCAL: tl.constexpr = NUM_SMS // NUM_XCDS

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, BUF_DEPTH)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, BUF_DEPTH)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089
    TILES_PER_UNIT: tl.constexpr = 2 if IS_CAUSAL else 1
    units_per_hz: tl.constexpr = (NUM_M_BLOCKS + TILES_PER_UNIT - 1) // TILES_PER_UNIT
    hz_per_xcd = (Z * H + NUM_XCDS - 1) // NUM_XCDS
    units = hz_per_xcd * units_per_hz

    for unit in tl.range(local, units, NUM_LOCAL, num_stages=0):
        local_hz = unit // units_per_hz
        bundle = unit % units_per_hz
        pid_hz = xcd + local_hz * NUM_XCDS
        if pid_hz < Z * H:
            off_z = pid_hz // H
            off_h = pid_hz % H
            for j in tl.static_range(TILES_PER_UNIT):
                idx = bundle * TILES_PER_UNIT + j
                if idx < NUM_M_BLOCKS:
                    if IS_CAUSAL:
                        half = idx // 2
                        pid_m = tl.where(idx % 2 == 0, half, NUM_M_BLOCKS - 1 - half)
                    else:
                        pid_m = idx
                    # Safe to reuse the LDS slots across units: the outer loop
                    # has num_stages=0 and _attn_cluster_tile drains all async
                    # load groups before it returns.
                    _attn_cluster_tile(
                        pid_m,
                        off_z,
                        off_h,
                        Q,
                        K,
                        V,
                        Out,
                        k_buf,
                        v_buf,
                        stride_qz,
                        stride_qh,
                        stride_qm,
                        stride_qk,
                        stride_kz,
                        stride_kh,
                        stride_kn,
                        stride_kk,
                        stride_vz,
                        stride_vh,
                        stride_vn,
                        stride_vk,
                        stride_oz,
                        stride_oh,
                        stride_om,
                        stride_ok,
                        N_CTX,
                        QK_SCALE,
                        BLOCK_M,
                        BLOCK_N,
                        BUF_DEPTH,
                        HEAD_DIM,
                        IS_CAUSAL,
                    )


def _cluster_default_block_n(causal):
    return 32 if causal else 64


def flash_attn_cluster_pipeline(q, k, v, sm_scale, causal=False, **kw):
    mfma_m = 32
    block_m = kw.pop("BLOCK_M", 256)
    block_n = kw.pop("BLOCK_N", _cluster_default_block_n(causal))
    num_warps = kw.pop("num_warps", min(8, max(1, block_m // mfma_m)))
    waves_per_eu = kw.pop("waves_per_eu", 0 if causal else 2)
    B, H, N_CTX, D = q.shape

    o = torch.empty_like(q)
    m_blocks = triton.cdiv(N_CTX, block_m)
    grid = (H, m_blocks, B) if causal else (m_blocks, H, B)
    _attn_fwd_cluster_pipeline[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        N_CTX,
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BUF_DEPTH=CLUSTER_BUF_DEPTH,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        **kw,
    )
    return o


def flash_attn_cluster_persistent_pipeline(q, k, v, sm_scale, causal=False, **kw):
    mfma_m = 32
    block_m = kw.pop("BLOCK_M", 256)
    block_n = kw.pop("BLOCK_N", _cluster_default_block_n(causal))
    num_warps = kw.pop("num_warps", min(8, max(1, block_m // mfma_m)))
    waves_per_eu = kw.pop("waves_per_eu", 0 if causal else 2)
    B, H, N_CTX, D = q.shape
    num_xcds = kw.pop("NUM_XCDS", 8)
    assert num_xcds > 0, f"cluster attention: NUM_XCDS must be positive, got {num_xcds}"
    cu_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_sms = kw.pop("NUM_SMS", (cu_count // num_xcds) * num_xcds)
    assert num_sms >= num_xcds, f"cluster attention: NUM_SMS ({num_sms}) must be >= NUM_XCDS ({num_xcds})"
    assert num_sms % num_xcds == 0, (
        f"cluster attention: NUM_SMS ({num_sms}) must be divisible by NUM_XCDS ({num_xcds})")

    o = torch.empty_like(q)
    m_blocks = triton.cdiv(N_CTX, block_m)
    grid = (num_sms, )
    _attn_fwd_cluster_persistent_pipeline[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B,
        H,
        N_CTX,
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BUF_DEPTH=CLUSTER_BUF_DEPTH,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        NUM_M_BLOCKS=m_blocks,
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        **kw,
    )
    return o


def attention(q, k, v, sm_scale, causal, config=None):
    config = {} if config is None else dict(config)
    return flash_attn_cluster_pipeline(q, k, v, sm_scale, causal=causal, **config)


def persistent_attention(q, k, v, sm_scale, causal, config=None):
    config = {} if config is None else dict(config)
    return flash_attn_cluster_persistent_pipeline(q, k, v, sm_scale, causal=causal, **config)
