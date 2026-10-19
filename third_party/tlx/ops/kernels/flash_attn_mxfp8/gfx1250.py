"""gfx1250 MXFP8 Flash Attention forward.

BF16 ``(Z, H, N_CTX, HEAD_DIM)`` in, BF16 out; forward only. The quantization
is Blackwell's, as on gfx950: Q and K are E4M3 with an E8M0 scale per 32x32
block (32 rows by 32 head elements), V is E4M3 with an E8M0 scale per 32 keys
of each head column, and the softmax probabilities P get an E8M0 scale per 32
keys of each row, all by RCEIL of the block max. Q is quantized unscaled, and
sm_scale * log2(e) multiplies the scores. Both WMMAs are scaled
(v_wmma_scale_f32_16x16x128_f8f6f4). A call makes two launches: one quantizes
Q, K and V, the other runs the attention.

The attention kernel runs one workgroup of 8 warps per 256-row Q tile, each
warp owning 32 rows, over 128-key K/V tiles. Every operand moves through the
tensor data mover (TDM) as 256-byte rows into LDS, with four K and V slots so
loads run three tiles ahead. A step runs the next tile's QK WMMAs and the
current tile's PV WMMAs, then the next tile's softmax, and the two halves of
the workgroup ping-pong between the two stages (tlx.warp_pipeline_stage), so
one half's softmax overlaps the other's WMMAs. P's block scales join the exp
argument, so P comes out divided by them. Causal Q tiles stop at the
diagonal, which crosses their last two K/V tiles.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

_FP8 = torch.float8_e4m3fn
_LOG2E = 1.4426950408889634
_HEAD_DIM = 128
_BLOCK_M = 256
_BLOCK_N = 128


@triton.jit
def _mx_scale(amax):
    # E8M0 byte of RCEIL(amax * fp32(1 / 448)), 0 for amax 0: the rule of
    # Blackwell's quantizers (cvt.rp.satfinite.ue8m0x2.f32) below its top byte,
    # which BF16 data and P's block maxes stay far from.
    return ((amax * (1.0 / 448.0)).to(tl.int32, bitcast=True) + 0x7FFFFF) >> 23


@triton.jit
def _quantize_head(X, Out, Scale, stride_xz, stride_xh, stride_xn, stride_xd, H, N_CTX, HEAD_DIM: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    # Each 32x32 block of x (32 rows by 32 head elements) gets the E8M0 byte
    # _mx_scale(amax) for its max magnitude amax, stored for each of its rows,
    # and E4M3 data x / 2**(byte - 127).
    pid_bh = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_b = tl.arange(0, HEAD_DIM // 32)
    x = tl.load(X + (pid_bh // H) * stride_xz + (pid_bh % H) * stride_xh + offs_n[:, None] * stride_xn +
                offs_d[None, :] * stride_xd).to(tl.float32)
    x = tl.reshape(x, (BLOCK_N // 32, 32, HEAD_DIM // 32, 32))
    byte = _mx_scale(tl.max(tl.max(tl.abs(x), 3), 1))
    inv = ((254 - byte) << 23).to(tl.float32, bitcast=True)
    q = tl.clamp(x * inv[:, None, :, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Out + (pid_bh * N_CTX + offs_n[:, None]) * HEAD_DIM + offs_d[None, :], tl.reshape(q, (BLOCK_N, HEAD_DIM)),
             cache_modifier=".cs")
    s = tl.reshape(tl.broadcast_to(byte[:, None, :], (BLOCK_N // 32, 32, HEAD_DIM // 32)), (BLOCK_N, HEAD_DIM // 32))
    tl.store(Scale + (pid_bh * N_CTX + offs_n[:, None]) * (HEAD_DIM // 32) + offs_b[None, :], s.to(tl.uint8),
             cache_modifier=".cs")


@triton.jit
def _quantize_v(V, Out, Scale, stride_vz, stride_vh, stride_vn, stride_vd, H, N_CTX, HEAD_DIM: tl.constexpr,
                BLOCK_N: tl.constexpr):
    # V to MXFP8 along the keys (a 32-key block per column), by
    # _quantize_head's rule. The scales go out per 128-key tile as
    # [column, key block], the order the PV WMMA reads them in.
    pid_bh = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    x = tl.load(V + (pid_bh // H) * stride_vz + (pid_bh % H) * stride_vh + offs_n[:, None] * stride_vn +
                offs_d[None, :] * stride_vd).to(tl.float32)
    x = tl.reshape(x, (BLOCK_N // 32, 32, HEAD_DIM))
    byte = _mx_scale(tl.max(tl.abs(x), 1))
    inv = ((254 - byte) << 23).to(tl.float32, bitcast=True)
    q = tl.reshape(tl.clamp(x * inv[:, None, :], -448.0, 448.0).to(tl.float8e4nv), (BLOCK_N, HEAD_DIM))
    tl.store(Out + (pid_bh * N_CTX + offs_n[:, None]) * HEAD_DIM + offs_d[None, :], q, cache_modifier=".cs")
    # Key block b of the whole sequence sits in tile b // 4 at [column, b % 4].
    offs_b = tl.program_id(0) * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
    offs = (pid_bh * (N_CTX // 128) + offs_b[:, None] // 4) * (HEAD_DIM * 4) + offs_d[None, :] * 4 + offs_b[:, None] % 4
    tl.store(Scale + offs, byte.to(tl.uint8), cache_modifier=".cs")


@triton.jit
def _quantize_mxfp8_qkv_kernel(Q, K, V, Q_out, Q_scale, K_out, K_scale, V_out, V_scale, stride_qz, stride_qh, stride_qn,
                               stride_qd, stride_kz, stride_kh, stride_kn, stride_kd, stride_vz, stride_vh, stride_vn,
                               stride_vd, H, N_CTX, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
    # The three quantizers in one launch, picked by the grid's third axis: each
    # launch costs more host time than the quantizers take on the GPU.
    which = tl.program_id(2)
    if which == 0:
        _quantize_head(Q, Q_out, Q_scale, stride_qz, stride_qh, stride_qn, stride_qd, H, N_CTX, HEAD_DIM, BLOCK_N)
    elif which == 1:
        _quantize_head(K, K_out, K_scale, stride_kz, stride_kh, stride_kn, stride_kd, H, N_CTX, HEAD_DIM, BLOCK_N)
    else:
        _quantize_v(V, V_out, V_scale, stride_vz, stride_vh, stride_vn, stride_vd, H, N_CTX, HEAD_DIM, BLOCK_N)


def _check_shape(x):
    z, h, n, d = x.shape
    if d != _HEAD_DIM or n % 128 != 0:
        raise ValueError(f"gfx1250 MXFP8 quantization needs HEAD_DIM {_HEAD_DIM} and N_CTX % 128 == 0, got {x.shape}")
    return z, h, n, d


def quantize_mxfp8_qkv(q, k, v):
    """E4M3 data and E8M0 scales of Q, K and V as the attention kernel reads
    them, as ``(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)``. The data are
    ``[Z, H, N_CTX, HEAD_DIM]``. Q's and K's scales are ``[Z, H, N_CTX,
    HEAD_DIM // 32]``, each row holding its 32x32 blocks'. V's are ``[Z, H,
    N_CTX // 128, HEAD_DIM, 4]``: per 128-key tile, column ``d`` holds the
    scales of its four 32-key blocks."""
    z, h, n, d = _check_shape(q)
    q_fp8 = torch.empty((z, h, n, d), device=q.device, dtype=_FP8)
    k_fp8 = torch.empty_like(q_fp8)
    v_fp8 = torch.empty_like(q_fp8)
    q_scale = torch.empty((z, h, n, d // 32), device=q.device, dtype=torch.uint8)
    k_scale = torch.empty_like(q_scale)
    v_scale = torch.empty((z, h, n // 128, d, 4), device=q.device, dtype=torch.uint8)
    _quantize_mxfp8_qkv_kernel[(n // 64, z * h, 3)](q, k, v, q_fp8, q_scale, k_fp8, k_scale, v_fp8, v_scale,
                                                    *q.stride(), *k.stride(), *v.stride(), h, n, HEAD_DIM=d, BLOCK_N=64,
                                                    num_warps=2)
    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


@triton.constexpr_function
def _rows_layout(rows):
    # TDM moves 256-byte rows fastest. The 16-byte pad after each one keeps
    # the WMMA operand reads (ds_load_b128 and ds_load_tr8_b64) apart in banks.
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [rows, 256], [1, 0])


@triton.jit
def _desc(base, n_bytes, BLOCK_BYTES: tl.constexpr):
    # A head's data as 256-byte rows, moved BLOCK_BYTES at a time.
    return tl.make_tensor_descriptor(base, shape=[n_bytes // 256, 256], strides=[256, 1],
                                     block_shape=[BLOCK_BYTES // 256, 256])


@triton.jit
def _issue(desc, buf, slot, tile, BLOCK_BYTES: tl.constexpr, pred=True):
    tlx.async_amd_descriptor_load(desc, tlx.local_view(buf, slot), [tile * (BLOCK_BYTES // 256), 0], pred=pred,
                                  clamp_bounds=False)


@triton.jit
def _tile(buf, slot, ROWS: tl.constexpr, COLS: tl.constexpr):
    # A [ROWS, COLS] tile out of its 256-byte LDS rows.
    return tlx.local_load(tlx.local_reshape(tlx.local_view(buf, slot), [ROWS, COLS]))


@triton.jit
def _k_tile(k_buf, slot, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # K^T for the QK WMMA, read transposed out of its [key, head] LDS tile.
    return tlx.local_load(tlx.local_trans(tlx.local_reshape(tlx.local_view(k_buf, slot), [BLOCK_N, HEAD_DIM])))


@triton.jit
def _softmax(qk, m_i, l_i, qk_scale, offs_m, key0, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, MASK: tl.constexpr):
    # P of one K/V tile as E4M3 with a scale per 32 keys, the factor alpha
    # that rescales the output to the new running max, and the new running
    # max (log2 units) and row sums. MASK hides the keys past each row.
    if MASK:
        valid = offs_m[:, None] >= key0 + tl.arange(0, BLOCK_N)[None, :]
        qk = tl.where(valid, qk, -1.0e6)
    qk = tl.reshape(qk, (BLOCK_M, BLOCK_N // 32, 32))
    tmax = tl.max(qk, 2) * qk_scale
    m_ij = tl.maximum(m_i, tl.max(tmax, 1))
    # A block's max P is exp2(its max score - m); the scale's exponent joins
    # the exp argument, so P comes out divided by it.
    byte = _mx_scale(tl.math.exp2(tmax - m_ij[:, None]))
    p = tl.math.exp2(qk * qk_scale - (m_ij[:, None] + (byte - 127).to(tl.float32))[:, :, None])
    if MASK:
        # With qk_scale 0 the hidden scores would count as visible.
        p = tl.where(tl.reshape(valid, (BLOCK_M, BLOCK_N // 32, 32)), p, 0)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + tl.sum(tl.sum(p, 2) * (byte << 23).to(tl.float32, bitcast=True), 1)
    p8 = tl.reshape(p, (BLOCK_M, BLOCK_N)).to(tl.float8e4nv)
    return p8, byte.to(tl.uint8), alpha, m_ij, l_i


@triton.jit
def _qk(q, q_scale, k_buf, ks_buf, slot, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # Each warp owns 32 whole rows, so the softmax reduces within warps.
    k = _k_tile(k_buf, slot, BLOCK_N, HEAD_DIM)
    ks = _tile(ks_buf, slot, BLOCK_N, HEAD_DIM // 32)
    return tlx.dot_scaled(q, q_scale, "e4m3", k, ks, "e4m3", fast_math=True, tiles_per_warp=[2, BLOCK_N // 16])


@triton.jit
def _pv(p8, p_scale, alpha, acc, v_buf, vs_buf, slot, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    v = _tile(v_buf, slot, BLOCK_N, HEAD_DIM)
    vs = _tile(vs_buf, slot, HEAD_DIM, HEAD_DIM // 32)
    return tlx.dot_scaled(p8, p_scale, "e4m3", v, vs, "e4m3", acc * alpha[:, None], fast_math=True,
                          tiles_per_warp=[2, HEAD_DIM // 16])


@triton.jit
def _mfma(q, q_scale, p8, p_scale, alpha, acc, k_buf, ks_buf, v_buf, vs_buf, k_slot, v_slot, BLOCK_N: tl.constexpr,
          HEAD_DIM: tl.constexpr):
    # The next tile's scores and the current tile's PV.
    qk = _qk(q, q_scale, k_buf, ks_buf, k_slot, BLOCK_N, HEAD_DIM)
    tlx.amd_sched_barrier(0)
    acc = _pv(p8, p_scale, alpha, acc, v_buf, vs_buf, v_slot, BLOCK_N, HEAD_DIM)
    return qk, acc


@triton.jit
def _issue_step(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, j, n_tiles, KV_BYTES: tl.constexpr,
                KS_BYTES: tl.constexpr, VS_BYTES: tl.constexpr):
    # V(j + 3) and K(j + 4), into the slots of V(j - 1) and K(j).
    _issue(v_desc, v_buf, (j + 3) % 4, j + 3, KV_BYTES, j + 3 < n_tiles)
    _issue(vs_desc, vs_buf, (j + 3) % 4, j + 3, VS_BYTES, j + 3 < n_tiles)
    _issue(k_desc, k_buf, (j + 4) % 4, j + 4, KV_BYTES, j + 4 < n_tiles)
    _issue(ks_desc, ks_buf, (j + 4) % 4, j + 4, KS_BYTES, j + 4 < n_tiles)


@triton.jit
def _mxfp8_fa_fwd_tdm(
    Q,
    K,
    V,
    Qs,
    Ks,
    Vs,
    Out,
    N_CTX,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # Q, K, V and the output are [Z * H, N_CTX, HEAD_DIM], Q's and K's scales
    # [Z * H, N_CTX, 4] and V's [Z * H, N_CTX // 128, HEAD_DIM, 4], all
    # contiguous.
    tl.static_assert(BLOCK_M == 2 * BLOCK_N, "a causal Q tile's diagonal crosses its last two K/V tiles")
    tl.static_assert(BLOCK_N == 128, "V's scales come in 128-key tiles")
    SCALE_K: tl.constexpr = HEAD_DIM // 32
    if CAUSAL:
        # Every head's longest Q tiles go first.
        off_hz = tl.program_id(0)
        start_m = N_CTX // BLOCK_M - 1 - tl.program_id(1)
        n_tiles = 2 * start_m + 2
        n_plain = tl.maximum(n_tiles - 3, 0)
    else:
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        n_tiles = N_CTX // BLOCK_N
        n_plain = n_tiles - 1
    head = off_hz.to(tl.int64) * N_CTX * HEAD_DIM
    head_scales = off_hz.to(tl.int64) * N_CTX * SCALE_K
    Q_BYTES: tl.constexpr = BLOCK_M * HEAD_DIM
    KV_BYTES: tl.constexpr = BLOCK_N * HEAD_DIM
    QS_BYTES: tl.constexpr = BLOCK_M * SCALE_K
    KS_BYTES: tl.constexpr = BLOCK_N * SCALE_K
    VS_BYTES: tl.constexpr = HEAD_DIM * SCALE_K

    q_desc = _desc(Q + head, N_CTX * HEAD_DIM, Q_BYTES)
    k_desc = _desc(K + head, N_CTX * HEAD_DIM, KV_BYTES)
    v_desc = _desc(V + head, N_CTX * HEAD_DIM, KV_BYTES)
    qs_desc = _desc(Qs + head_scales, N_CTX * SCALE_K, QS_BYTES)
    ks_desc = _desc(Ks + head_scales, N_CTX * SCALE_K, KS_BYTES)
    vs_desc = _desc(Vs + head_scales, N_CTX * SCALE_K, VS_BYTES)
    q_buf = tlx.local_alloc((Q_BYTES // 256, 256), tlx.dtype_of(Q), 1, layout=_rows_layout(Q_BYTES // 256))
    k_buf = tlx.local_alloc((KV_BYTES // 256, 256), tlx.dtype_of(K), 4, layout=_rows_layout(KV_BYTES // 256))
    v_buf = tlx.local_alloc((KV_BYTES // 256, 256), tlx.dtype_of(V), 4, layout=_rows_layout(KV_BYTES // 256))
    qs_buf = tlx.local_alloc((QS_BYTES // 256, 256), tlx.dtype_of(Qs), 1)
    ks_buf = tlx.local_alloc((KS_BYTES // 256, 256), tlx.dtype_of(Ks), 4)
    vs_buf = tlx.local_alloc((VS_BYTES // 256, 256), tlx.dtype_of(Vs), 4)

    # Step j runs tile j + 1's QK WMMAs and tile j's PV WMMAs (the mfma
    # stage), then tile j + 1's softmax, which also issues V(j + 3) and
    # K(j + 4). The two warp groups take turns on the matrix cores, a stage
    # apart. Only the mfma stage reads LDS and only the softmax stage writes
    # it, so a slot is refilled once both groups have read it. A warp's TDM
    # wait covers only its own share of a tile, so the wait at the end of step
    # j covers the tiles step j + 1 reads, before the other group can start
    # that step. TDM order: Q, K0, K1, V0, V1, K2, V2, K3 (each with its
    # scales), then V(j + 3) and K(j + 4) per step. A load past the last tile
    # is issued with its predicate off, which keeps the wait counts.
    _issue(q_desc, q_buf, 0, start_m, Q_BYTES)
    _issue(qs_desc, qs_buf, 0, start_m, QS_BYTES)
    _issue(k_desc, k_buf, 0, 0, KV_BYTES)
    _issue(ks_desc, ks_buf, 0, 0, KS_BYTES)
    _issue(k_desc, k_buf, 1, 1, KV_BYTES)
    _issue(ks_desc, ks_buf, 1, 1, KS_BYTES)
    _issue(v_desc, v_buf, 0, 0, KV_BYTES)
    _issue(vs_desc, vs_buf, 0, 0, VS_BYTES)
    _issue(v_desc, v_buf, 1, 1, KV_BYTES)
    _issue(vs_desc, vs_buf, 1, 1, VS_BYTES)
    _issue(k_desc, k_buf, 2, 2, KV_BYTES, n_tiles > 2)
    _issue(ks_desc, ks_buf, 2, 2, KS_BYTES, n_tiles > 2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    tlx.async_amd_descriptor_wait(8)
    q = _tile(q_buf, 0, BLOCK_M, HEAD_DIM)
    q_scale = _tile(qs_buf, 0, BLOCK_M, SCALE_K)
    qk = _qk(q, q_scale, k_buf, ks_buf, 0, BLOCK_N, HEAD_DIM)
    # Only Q tile 0 meets the diagonal in K/V tile 0; the mask keeps every
    # key of it for the others.
    p8, p_scale, alpha, m_i, l_i = _softmax(qk, m_i, l_i, qk_scale, offs_m, 0, BLOCK_M, BLOCK_N, CAUSAL)
    _issue_step(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, -1, n_tiles, KV_BYTES, KS_BYTES,
                VS_BYTES)
    tlx.async_amd_descriptor_wait(8)

    for j in range(0, n_plain):
        with tlx.warp_pipeline_stage("mfma", priority=1):
            qk, acc = _mfma(q, q_scale, p8, p_scale, alpha, acc, k_buf, ks_buf, v_buf, vs_buf, (j + 1) % 4, j % 4,
                            BLOCK_N, HEAD_DIM)
        tlx.async_amd_descriptor_wait(4)
        with tlx.warp_pipeline_stage("softmax", priority=0):
            p8, p_scale, alpha, m_i, l_i = _softmax(qk, m_i, l_i, qk_scale, offs_m, (j + 1) * BLOCK_N, BLOCK_M, BLOCK_N,
                                                    False)
            _issue_step(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, j, n_tiles, KV_BYTES, KS_BYTES,
                        VS_BYTES)
    if CAUSAL:
        # The steps whose next tile crosses the diagonal.
        for j in range(n_plain, n_tiles - 1):
            with tlx.warp_pipeline_stage("mfma", priority=1):
                qk, acc = _mfma(q, q_scale, p8, p_scale, alpha, acc, k_buf, ks_buf, v_buf, vs_buf, (j + 1) % 4, j % 4,
                                BLOCK_N, HEAD_DIM)
            tlx.async_amd_descriptor_wait(4)
            with tlx.warp_pipeline_stage("softmax", priority=0):
                p8, p_scale, alpha, m_i, l_i = _softmax(qk, m_i, l_i, qk_scale, offs_m, (j + 1) * BLOCK_N, BLOCK_M,
                                                        BLOCK_N, True)
                _issue_step(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, j, n_tiles, KV_BYTES,
                            KS_BYTES, VS_BYTES)
    acc = _pv(p8, p_scale, alpha, acc, v_buf, vs_buf, (n_tiles - 1) % 4, BLOCK_N, HEAD_DIM)

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    o_desc = tl.make_tensor_descriptor(Out + head, shape=[N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    tlx.async_amd_descriptor_wait(0)
    o_buf = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.bfloat16, 1)
    tlx.local_store(tlx.local_view(o_buf, 0), (acc * inv_l[:, None]).to(tl.bfloat16))
    tlx.async_amd_descriptor_store(o_desc, tlx.local_view(o_buf, 0), [start_m * BLOCK_M, 0], clamp_bounds=False)
    tlx.async_amd_descriptor_wait(0)


def _launch_quantized(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal, sm_scale):
    """Launch on the outputs of ``quantize_mxfp8_qkv``."""
    batch, heads, n_ctx, _ = q_fp8.shape
    out = torch.empty(q_fp8.shape, device=q_fp8.device, dtype=torch.bfloat16)
    n_m = n_ctx // _BLOCK_M
    grid = (batch * heads, n_m) if causal else (n_m, batch * heads)
    _mxfp8_fa_fwd_tdm[grid](
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        out,
        n_ctx,
        sm_scale * _LOG2E,
        HEAD_DIM=_HEAD_DIM,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        CAUSAL=causal,
        num_warps=8,
        waves_per_eu=2,
    )
    return out


def flash_attn_mxfp8(q, k, v, causal=False, sm_scale=None, *, space="full", p_expcast=False):
    """BF16 MXFP8 attention. ``space`` selects no separate gfx1250 config yet."""
    del space
    if p_expcast:
        raise ValueError("gfx1250 flash_attn_mxfp8 does not support p_expcast")
    if sm_scale is None:
        sm_scale = q.shape[-1]**-0.5
    if q.dtype != torch.bfloat16:
        raise ValueError(f"gfx1250 flash_attn_mxfp8 expects bf16, got {q.dtype}")
    if q.shape[-1] != _HEAD_DIM or q.shape[-2] % _BLOCK_M != 0:
        raise ValueError(f"gfx1250 flash_attn_mxfp8 expects head dim {_HEAD_DIM} and N_CTX % {_BLOCK_M} == 0, "
                         f"got {tuple(q.shape)}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("gfx1250 flash_attn_mxfp8 expects q, k and v of the same shape")
    return _launch_quantized(*quantize_mxfp8_qkv(q, k, v), causal, sm_scale)
