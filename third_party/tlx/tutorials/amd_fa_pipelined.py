import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


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


@triton.jit
def _attn_fwd_async_simple(
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
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    QK_SCALE = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, 1)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, 1)

    k_base = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    for start_n in tl.range(0, hi, BLOCK_N, num_stages=0):
        kn = start_n + offs_n
        k_mask = kn[:, None] < N_CTX
        v_mask = kn[:, None] < N_CTX

        tok_k = tlx.async_load(k_base + start_n * stride_kn, tlx.local_view(k_buf, 0), mask=k_mask)
        tok_v = tlx.async_load(v_base + start_n * stride_vn, tlx.local_view(v_buf, 0), mask=v_mask)
        tlx.async_load_commit_group([tok_k, tok_v])

        wait_tok = tlx.async_load_wait_group(0)
        kt_view = tlx.local_trans(tlx.local_view(k_buf, 0))
        kt_cur = tlx.local_load(kt_view, token=wait_tok)
        v_cur = tlx.local_load(tlx.local_view(v_buf, 0), token=wait_tok)

        qk = tl.dot(q, kt_cur)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk = qk * QK_SCALE - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _attn_fwd_async_prefetch(
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
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Prefetch flash attention with explicit modulo-scheduled prologue,
    hot loop (steady state), and epilogue.

    Design notes:
      * K and V are both double-buffered (2 LDS slots, ping-pong index
        i%2).
      * `local_trans` is applied to K so `local_load` lands directly in
        dot-operand layout 1, skipping the per-iter ds_write+barrier+
        ds_read shuffle that `tl.dot(q, k_cur.T)` would emit.
    Prologue:
    t = 0
    [GLDS_KV]

    Steady State (Hot Loop):
    t = i               t = i+1
    [LR_KV]
    [QK, SM0, SM1, PV]  [GLDS_KV],
    Epilogue:
                        t = i+1
                        [LR_KV]
                        [QK (masked), SM0, SM1, PV]
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    n_main = tl.maximum(n_blocks - 1, 0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    """
    Prologue:
    t = 0
    [GLDS_KV]
    """
    tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0), mask=offs_n[:, None] < N_CTX)
    tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0), mask=offs_n[:, None] < N_CTX)
    tlx.async_load_commit_group([tok_k0, tok_v0])
    """
    Steady State (Hot Loop):
    t = i               t = i+1
    [LR_KV]
    [QK, SM0, SM1, PV]  [GLDS_KV],
    """
    for block_id in tl.range(0, n_main * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        kn = block_id + offs_n
        next_mask = (next_off + offs_n[:, None]) < N_CTX

        i = block_id // BLOCK_N
        slot_cur = i % 2
        slot_nxt = (i + 1) % 2

        # LR_KV_ti
        wait_tok = tlx.async_load_wait_group(0)
        kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_cur))
        kt_cur = tlx.local_load(kt_view, token=wait_tok)
        v_cur = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok)

        # GLDS_KV_t(i+1)
        tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, slot_nxt), mask=next_mask)
        tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, slot_nxt), mask=next_mask)
        tlx.async_load_commit_group([tok_k, tok_v])

        # QK_ti
        qk = tl.dot(q, kt_cur)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))

        # SM_ti
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # PV_ti
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
    """
    Epilogue:
    t = i+1
    [LR_KV]
    [QK (masked), SM0, SM1, PV]
    """
    wait_tok = tlx.async_load_wait_group(0)
    slot_last = n_main % 2
    kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_last))
    kt_cur = tlx.local_load(kt_view, token=wait_tok)
    v_cur = tlx.local_load(tlx.local_view(v_buf, slot_last), token=wait_tok)

    # QK_t(i+1) — with boundary + causal masking
    kn_last = n_main * BLOCK_N + offs_n
    qk = tl.dot(q, kt_cur)
    qk = tl.where(kn_last[None, :] < N_CTX, qk, float("-inf"))
    if IS_CAUSAL:
        qk = tl.where(offs_m[:, None] >= kn_last[None, :], qk, float("-inf"))

    # SM_t(i+1)
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
    p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij

    # PV_t(i+1)
    acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    # Store output
    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


def _launch_kernel(kernel_fn, q, k, v, sm_scale, causal, config):
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = config.get("BLOCK_M", 256)
    BLOCK_N = config.get("BLOCK_N", 64)
    num_warps = config.get("num_warps", 4)

    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
    kernel_fn[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N_CTX, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, IS_CAUSAL=causal,
        num_warps=num_warps,
    )
    return o


def attention(q, k, v, sm_scale, causal, config=None):
    if config is None:
        config = {"BLOCK_M": 256, "BLOCK_N": 64, "num_warps": 4}
    kernel = _attn_fwd_async_prefetch if config.get("PREFETCH", False) else _attn_fwd_async_simple
    return _launch_kernel(kernel, q, k, v, sm_scale, causal, config)
