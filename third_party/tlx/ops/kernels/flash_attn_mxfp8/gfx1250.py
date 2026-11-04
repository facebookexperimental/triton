"""gfx1250 MXFP8 Flash Attention forward.

BF16 ``(Z, H, N_CTX, HEAD_DIM)`` in, BF16 out; forward only. The quantization
is Blackwell's, as on gfx950: Q and K are E4M3 with an E8M0 scale per 32x32
block (32 rows by 32 head elements), V is E4M3 with an E8M0 scale per 32 keys
of each head column, and the softmax probabilities P get an E8M0 scale per 32
keys of each row, all by RCEIL of the block max. Q is quantized unscaled, and
sm_scale * log2(e) multiplies the scores. Both WMMAs are scaled
(v_wmma_scale_f32_16x16x128_f8f6f4). A call makes two launches: one quantizes
Q, K and V, the other runs the attention.

The attention kernels run workgroups of 8 warps on 256-row Q tiles, each
warp owning 32 rows, over 128-key K/V tiles. Non-causal sequences that are a
multiple of 512, and causal ones with a nonzero sm_scale, go to persistent
kernels: one workgroup per CU runs the K/V tiles of its Q tiles as a single
stream, the first tiles of the next Q tile overlapping the end of the last
(causal Q tiles go heaviest first, snaking across the workgroups). Other
calls get one workgroup per Q tile. K, V and their scales move through the
tensor data mover (TDM) as 256-byte rows into LDS, with four K and V slots so
loads run three tiles ahead; Q does too, except in the persistent kernels,
which copy it with async loads. A step takes the current tile's exps and runs
its PV WMMAs and the next tile's QK WMMAs, then the next tile's softmax up to
the exp arguments, and the two halves of the workgroup ping-pong between the
two stages (tlx.warp_pipeline_stage), so one half's softmax overlaps the
other's WMMAs. P's block scales join the exp argument, so P comes out divided
by them. As on gfx950, P is computed with a bit-trick exp2 instead of the
hardware exp, the accumulators are only rescaled when a row max grows past a
threshold, and P's row sums run on the matrix cores. Causal Q tiles stop at
the diagonal, which crosses their last two K/V tiles; there the QK
accumulator starts at -inf for the hidden keys.

The bit-trick exp2 gives 2**x with a linear mantissa, biased high; P and its
row sums share that bias. ``_launch_quantized(..., fast_exp=False)`` runs the
kernel with the hardware exp instead.
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
    # _quantize_head's rule. The data go out transposed per 128-key tile, as
    # [column, key], so the PV WMMA reads them out of LDS with 16-byte loads
    # as it does K, where [key, column] would need the 8-byte transposing
    # loads at half the bandwidth. The scales go out per tile as [column, key
    # block], the order the PV WMMA reads them in.
    pid_bh = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    x = tl.load(V + (pid_bh // H) * stride_vz + (pid_bh % H) * stride_vh + offs_n[:, None] * stride_vn +
                offs_d[None, :] * stride_vd).to(tl.float32)
    x = tl.reshape(x, (BLOCK_N // 32, 32, HEAD_DIM))
    byte = _mx_scale(tl.max(tl.abs(x), 1))
    inv = ((254 - byte) << 23).to(tl.float32, bitcast=True)
    q = tl.reshape(tl.clamp(x * inv[:, None, :], -448.0, 448.0).to(tl.float8e4nv), (BLOCK_N, HEAD_DIM))
    tile = pid_bh * (N_CTX // 128) + offs_n // 128
    tl.store(Out + (tile[None, :] * HEAD_DIM + offs_d[:, None]) * 128 + offs_n[None, :] % 128, tl.trans(q),
             cache_modifier=".cs")
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
    them, as ``(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)``. Q's and K's
    data are ``[Z, H, N_CTX, HEAD_DIM]``, V's ``[Z, H, N_CTX // 128,
    HEAD_DIM, 128]``: transposed per 128-key tile. Q's and K's scales are
    ``[Z, H, N_CTX, HEAD_DIM // 32]``, each row holding its 32x32 blocks'.
    V's are ``[Z, H, N_CTX // 128, HEAD_DIM, 4]``: per 128-key tile, column
    ``d`` holds the scales of its four 32-key blocks."""
    z, h, n, d = _check_shape(q)
    q_fp8 = torch.empty((z, h, n, d), device=q.device, dtype=_FP8)
    k_fp8 = torch.empty_like(q_fp8)
    v_fp8 = torch.empty((z, h, n // 128, d, 128), device=q.device, dtype=_FP8)
    q_scale = torch.empty((z, h, n, d // 32), device=q.device, dtype=torch.uint8)
    k_scale = torch.empty_like(q_scale)
    v_scale = torch.empty((z, h, n // 128, d, 4), device=q.device, dtype=torch.uint8)
    _quantize_mxfp8_qkv_kernel[(n // 64, z * h, 3)](q, k, v, q_fp8, q_scale, k_fp8, k_scale, v_fp8, v_scale,
                                                    *q.stride(), *k.stride(), *v.stride(), h, n, HEAD_DIM=d, BLOCK_N=64,
                                                    num_warps=2)
    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


@triton.constexpr_function
def _qk_layout():
    # The [256, 128] scores of one WMMA dot with tiles_per_warp [2, 8].
    return tlx.layout(shape=((16, 2, 8), (8, 8, 2)), stride=((128, 8, 4096), (1, 16, 2048)))


@triton.constexpr_function
def _acc_layout():
    # A [256, 16] accumulator of a WMMA dot with tiles_per_warp [2, 1]. Each
    # thread already holds 8 consecutive columns of a row, so O is stored from
    # it as is rather than reshuffled across warps for a blocked store.
    return tlx.layout(shape=((16, 2, 8), (8, 2)), stride=((16, 8, 512), (1, 256)))


@triton.constexpr_function
def _rows_layout(rows):
    # TDM moves 256-byte rows fastest. The 16-byte pad after each one keeps
    # the WMMA operand reads (ds_load_b128 and ds_load_tr8_b64) apart in banks.
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [rows // 16, 2, 2, 4, 256], [4, 3, 2, 1, 0])


@triton.jit
def _desc(base, n_bytes, BLOCK_BYTES: tl.constexpr, SWAP: tl.constexpr = False):
    # A head's data as 256-byte rows, moved BLOCK_BYTES at a time. SWAP swaps
    # bits 2 and 3 of the row index on the way into LDS, which stores K's keys
    # with bits 3 and 4 swapped (see _swap34) while the WMMA operand reads
    # still walk consecutive rows.
    ROWS: tl.constexpr = BLOCK_BYTES // 256
    if ROWS >= 16:
        SHAPE: tl.constexpr = [ROWS // 16, 2, 2, 4, 256]
        S2: tl.constexpr = 1024 if SWAP else 2048
        S1: tl.constexpr = 2048 if SWAP else 1024
        return tl.make_tensor_descriptor(base, shape=[n_bytes // 4096, 2, 2, 4, 256], strides=[4096, S2, S1, 256, 1],
                                         block_shape=SHAPE)
    else:
        # Scales, four bytes a key: bits 5 and 6 of the byte are the key's bits
        # 3 and 4.
        SHAPE: tl.constexpr = [ROWS, 2, 2, 2, 32]
        S6: tl.constexpr = 32 if SWAP else 64
        S5: tl.constexpr = 64 if SWAP else 32
        return tl.make_tensor_descriptor(base, shape=[n_bytes // 256, 2, 2, 2, 32], strides=[256, 128, S6, S5, 1],
                                         block_shape=SHAPE)


@triton.jit
def _at(desc, tile, BLOCK_BYTES: tl.constexpr, pred=True):
    STEP: tl.constexpr = BLOCK_BYTES // 4096 if BLOCK_BYTES >= 4096 else BLOCK_BYTES // 256
    return tlx.update_tensor_descriptor(desc, add_offsets=[tile * STEP, 0, 0, 0, 0], pred=pred)


@triton.jit
def _tile(buf, slot, ROWS: tl.constexpr, COLS: tl.constexpr):
    # A [ROWS, COLS] tile out of its 256-byte LDS rows.
    return tlx.local_load(tlx.local_reshape(tlx.local_view(buf, slot), [ROWS, COLS]))


@triton.jit
def _swap34(x):
    # x with bits 3 and 4 of its column index swapped.
    M: tl.constexpr = x.shape[0]
    N: tl.constexpr = x.shape[1]
    return tl.reshape(tl.permute(tl.reshape(x, (M, N // 32, 2, 2, 8)), (0, 1, 3, 2, 4)), (M, N))


@triton.jit
def _k_tile(k_buf, slot, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # K^T for the QK WMMA, read transposed out of its [key, head] LDS tile.
    # The TDM stores the keys with bits 3 and 4 swapped (see _desc).
    return tlx.local_load(tlx.local_trans(tlx.local_reshape(tlx.local_view(k_buf, slot), [BLOCK_N, HEAD_DIM])))


@triton.jit
def _maximum(a, b):
    # NaN-propagating, so the max needs no canonicalized inputs; with finite
    # scores only the sign of a zero max can differ, which nothing reads.
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _exp2_bf16(t):
    # 2**x for t = (x + 127) * 128 / 65535: v_cvt_pknorm_u16_f32 rounds
    # clamp(t, 0, 1) * 65535, the bf16 bit pattern of 2**x with a linear
    # mantissa (biased high; the 65535 / 2**16 slope error is common to P and
    # its sums), two per register.
    return tl.inline_asm_elementwise("v_cvt_pknorm_u16_f32 $0, $1, $2", "=v,v,v", [t], dtype=tl.bfloat16, is_pure=True,
                                     pack=2)


@triton.jit
def _softmax(qk, m_i, qk_scale, offs_m, key0, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, MASK: tl.constexpr,
             FAST_EXP: tl.constexpr, ZERO_SCALE: tl.constexpr):
    # The exp arguments of one K/V tile's P (see _p8), P's E8M0 scale per 32
    # keys, the running max (log2 units), and the factor alpha that rescales
    # the output and row sum accumulators to it, with whether any row of the
    # warp needs that. On a tile that crosses the diagonal the hidden scores
    # come in at -inf from _qk16, or with ZERO_SCALE (MASK) are hidden here
    # and in _p8. A row keeps its max until the tile's grows past it by more
    # than RESCALE, so P stays below 2**RESCALE and alpha is mostly 1. The
    # rescale is left to the next mfma stage, where the vote is long done: a
    # branch on it here waits for the row maxima and keeps the exp arguments
    # from overlapping them.
    RESCALE: tl.constexpr = 8.0
    PKNORM: tl.constexpr = 128.0 / 65535.0
    if MASK and ZERO_SCALE:
        qk = tl.where(_swap34(offs_m[:, None] >= key0 + tl.arange(0, BLOCK_N)[None, :]), qk, -1.0e6)
    qk = tl.reshape(qk, (BLOCK_M, BLOCK_N // 32, 32))
    tmax = tl.reduce(qk, 2, _maximum) * qk_scale
    row = tl.reduce(tmax, 1, _maximum)
    grow = row > m_i + RESCALE
    # The vote takes row r from lane r % 32 of warp r // 32, which holds it in
    # the WMMA layout already; left free it goes through LDS.
    rows: tl.constexpr = tlx.layout(shape=((32, BLOCK_M // 32), (1, )), stride=((1, 32), (1, )))
    rescale = tlx.warp_any(tlx.require_layout(grow, rows))
    m_new = tl.where(grow, row, m_i)
    alpha = tl.math.exp2(m_i - m_new)
    m_i = m_new
    # A block's max P is 2**(its max score - m_i), which sets its scale; the
    # scale's exponent joins the exp argument, so P comes out divided by it.
    if FAST_EXP:
        # Max P's bf16 bits b give _mx_scale's byte ((b + 31) >> 7) - 8, here
        # max(b - 993, 0) >> 7, two per register.
        byte = tl.inline_asm_elementwise(
            "v_cvt_pknorm_u16_f32 $0, $4, $5\n\tv_cvt_pknorm_u16_f32 $1, $6, $7\n\tv_cvt_pknorm_u16_f32 $2, $8, $9\n\tv_cvt_pknorm_u16_f32 $3, $10, $11\n\tv_pk_sub_u16 $0, $0, 0x3e1 op_sel_hi:[1,0] clamp\n\tv_pk_sub_u16 $1, $1, 0x3e1 op_sel_hi:[1,0] clamp\n\tv_pk_sub_u16 $2, $2, 0x3e1 op_sel_hi:[1,0] clamp\n\tv_pk_sub_u16 $3, $3, 0x3e1 op_sel_hi:[1,0] clamp\n\tv_pk_lshrrev_b16 $0, 7, $0 op_sel_hi:[0,1]\n\tv_pk_lshrrev_b16 $1, 7, $1 op_sel_hi:[0,1]\n\tv_pk_lshrrev_b16 $2, 7, $2 op_sel_hi:[0,1]\n\tv_pk_lshrrev_b16 $3, 7, $3 op_sel_hi:[0,1]",
            "=&v,=&v,=&v,=&v,v,v,v,v,v,v,v,v", [(tmax - m_i[:, None] + 127.0) * PKNORM], dtype=tl.uint16, is_pure=True,
            pack=8)
    else:
        byte = _mx_scale(tl.math.exp2(tmax - m_i[:, None]))
    if FAST_EXP and not (MASK and ZERO_SCALE):
        # The exp's top byte is P's E4M3 code: 2**x for t = ((x + 7) * 8 +
        # 0.5) * 256 / 65535, rounded to 3 mantissa bits. The scale keeps x
        # below 8.76, under the code of 448 (0x7E), and x below -7 gives 0.
        # A bf16 P would need v_cvt_scalef32_pk8_fp8_bf16, 7 cycles per 8.
        PK8: tl.constexpr = 2048.0 / 65535.0
        # byte is at most 127, so one conversion per u16 reads it.
        byte_f = tl.inline_asm_elementwise("v_cvt_f32_ubyte0 $0, $2\n\tv_cvt_f32_ubyte2 $1, $2", "=v,=v,v", [byte],
                                           dtype=tl.float32, is_pure=True, pack=2)
        off = ((134.0 - m_i[:, None]) - byte_f) * PK8 + 128.0 / 65535.0
        t = qk * (qk_scale * PK8) + off[:, :, None]
    elif FAST_EXP:
        off = ((254.0 - m_i[:, None]) - byte.to(tl.float32)) * PKNORM
        t = qk * (qk_scale * PKNORM) + off[:, :, None]
    else:
        t = qk * qk_scale - (m_i[:, None] + (byte - 127).to(tl.float32))[:, :, None]
    return tl.reshape(t, (BLOCK_M, BLOCK_N)), byte.to(tl.uint8), m_i, alpha, rescale


@triton.jit
def _p8(t, offs_m, key0, BLOCK_N: tl.constexpr, MASK: tl.constexpr, FAST_EXP: tl.constexpr, ZERO_SCALE: tl.constexpr):
    # P as E4M3 from _softmax's exp arguments.
    if FAST_EXP and not (MASK and ZERO_SCALE):
        w = _exp2_bf16(t).to(tl.uint16, bitcast=True)
    else:
        if FAST_EXP:
            p = _exp2_bf16(t)
        else:
            p = tl.math.exp2(t)
        if MASK and ZERO_SCALE:
            # With qk_scale 0 the hidden scores would count as visible.
            p = tl.where(_swap34(offs_m[:, None] >= key0 + tl.arange(0, BLOCK_N)[None, :]), p, 0)
        w = p.to(tl.float8e4nv).to(tl.uint8, bitcast=True).to(tl.uint16) << 8
    return w


@triton.jit
def _unpack_p8(w):
    # P's E4M3 codes, the top bytes of the u16 exps, gathered four to a
    # register: of each four u16 exps the first two end up holding the four
    # codes and the other two are left as they are.
    w = tl.inline_asm_elementwise("v_perm_b32 $0, $3, $2, 0x07050301", "=v,=v,v,1", [w], dtype=tl.uint16, is_pure=True,
                                  pack=4)
    M: tl.constexpr = w.shape[0]
    N: tl.constexpr = w.shape[1]
    codes, _ = tl.split(tl.permute(tl.reshape(w, (M, N // 4, 2, 2)), (0, 1, 3, 2)))
    p8 = tl.join((codes & 0xFF).to(tl.uint8), (codes >> 8).to(tl.uint8))
    return _swap34(tl.reshape(p8, (M, N)).to(tl.float8e4nv, bitcast=True))


@triton.jit
def _qk(q, q_scale, k_buf, ks_buf, slot, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # Each warp owns 32 whole rows, so the softmax reduces within warps.
    k = _k_tile(k_buf, slot, BLOCK_N, HEAD_DIM)
    ks = _tile(ks_buf, slot, BLOCK_N, HEAD_DIM // 32)
    return tlx.dot_scaled(q, q_scale, "e4m3", k, ks, "e4m3", None, fast_math=True, tiles_per_warp=[2, BLOCK_N // 16])


@triton.jit
def _cat2(a, b):
    # [M, N] and [M, N] side by side; with a and b in one register layout this
    # only renames registers.
    M: tl.constexpr = a.shape[0]
    N: tl.constexpr = a.shape[1]
    return tl.reshape(tl.permute(tl.join(a, b), (0, 2, 1)), (M, 2 * N))


@triton.jit
def _k16(k_buf, slot, n: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # Keys 16 n to 16 n + 15 of _k_tile.
    kv = tlx.local_reshape(tlx.local_view(k_buf, slot), [BLOCK_N, HEAD_DIM])
    return tlx.local_load(tlx.local_trans(tlx.local_slice(kv, [n * 16, 0], [16, HEAD_DIM])))


@triton.jit
def _ks16(ks_buf, slot, n: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    ksv = tlx.local_reshape(tlx.local_view(ks_buf, slot), [BLOCK_N, HEAD_DIM // 32])
    return tlx.local_load(tlx.local_slice(ksv, [n * 16, 0], [16, HEAD_DIM // 32]))


@triton.jit
def _warp():
    # Warp w holds rows 32 w to 32 w + 31 of the scores.
    return tl.inline_asm_elementwise("v_readfirstlane_b32 $0, $1", "=s,v", [tlx.thread_id(0)], dtype=tl.int32,
                                     is_pure=True, pack=1) >> 5


@triton.jit
def _qk16(q, q_scale, k, ks, offs_m, key0, n: tl.constexpr, MASK: tl.constexpr, row0=0):
    init = None
    if MASK:
        # Piece n of _k_tile holds keys 32 (n // 2) + 8 (n % 2) + {0..7, 16..23}.
        # Of the key pairs 32 q to 32 q + 31, a warp sees all for q < qt, none
        # for q > qt, and for q == qt the triangle tri, which is the same for
        # every warp: -2**127 doubles to -inf.
        r = tl.arange(0, offs_m.shape[0]) % 32
        c = tl.arange(0, 16)
        tri = tl.where(r[:, None] - 2 * (c & 8)[None, :] >= (8 * (n % 2) + (c & 7))[None, :], 0.0, -2.0**127)
        qt = _warp() - (key0 - row0) // 32
        init = tri * tl.where(qt == n // 2, 2.0, 0.0) + tl.where(qt < n // 2, float("-inf"), 0.0)
    return tlx.dot_scaled(q, q_scale, "e4m3", k, ks, "e4m3", init, fast_math=True, tiles_per_warp=[2, 1])


@triton.jit
def _v16(v_buf, slot, n: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # Columns 16 n to 16 n + 15 of V, read transposed out of its [column,
    # key] LDS tile, as K is.
    vv = tlx.local_reshape(tlx.local_view(v_buf, slot), [HEAD_DIM, BLOCK_N])
    return tlx.local_load(tlx.local_trans(tlx.local_slice(vv, [n * 16, 0], [16, BLOCK_N])))


@triton.jit
def _vs16(vs_buf, slot, n: tl.constexpr, HEAD_DIM: tl.constexpr):
    vsv = tlx.local_reshape(tlx.local_view(vs_buf, slot), [HEAD_DIM, HEAD_DIM // 32])
    return tlx.local_load(tlx.local_slice(vsv, [n * 16, 0], [16, HEAD_DIM // 32]))


@triton.jit
def _pv16(p8, p_scale, v, vs, acc):
    return tlx.dot_scaled(p8, p_scale, "e4m3", v, vs, "e4m3", acc, fast_math=True, tiles_per_warp=[2, 1])


@triton.jit
def _l_dot(p8, p_scale, l_acc, BLOCK_N: tl.constexpr):
    # P's row sums from one more WMMA per 16 rows, against a block of ones:
    # every column of l_acc holds them.
    ones = tl.full([BLOCK_N // 2, 16], 0x22, tl.uint8)
    ones_scale = tl.full([16, BLOCK_N // 32], 127, tl.uint8)
    return tlx.dot_scaled(p8, p_scale, "e4m3", ones, ones_scale, "e2m1", l_acc, fast_math=True, tiles_per_warp=[2, 1])


@triton.jit
def _rescale8(a0, a1, a2, a3, a4, a5, a6, a7, l_acc, alpha, rescale):
    if rescale:
        a = alpha[:, None]
        a0 = a0 * a
        a1 = a1 * a
        a2 = a2 * a
        a3 = a3 * a
        a4 = a4 * a
        a5 = a5 * a
        a6 = a6 * a
        a7 = a7 * a
        l_acc = l_acc * a
    return a0, a1, a2, a3, a4, a5, a6, a7, l_acc


@triton.jit
def _cat8(x0, x1, x2, x3, x4, x5, x6, x7):
    # The eight [M, 16] pieces as one [M, 128] tensor in the layout of a single
    # tiles_per_warp [2, 8] WMMA dot, which _softmax and P's packing are laid
    # out for.
    x = _cat2(_cat2(_cat2(x0, x1), _cat2(x2, x3)), _cat2(_cat2(x4, x5), _cat2(x6, x7)))
    return tlx.release_layout(tlx.require_layout(x, _qk_layout()), relaxed=True)


@triton.jit
def _pv8(p8, p_scale, a0, a1, a2, a3, a4, a5, a6, a7, l_acc, v_buf, vs_buf, slot, BLOCK_N: tl.constexpr,
         HEAD_DIM: tl.constexpr):
    a0 = _pv16(p8, p_scale, _v16(v_buf, slot, 0, BLOCK_N, HEAD_DIM), _vs16(vs_buf, slot, 0, HEAD_DIM), a0)
    a1 = _pv16(p8, p_scale, _v16(v_buf, slot, 1, BLOCK_N, HEAD_DIM), _vs16(vs_buf, slot, 1, HEAD_DIM), a1)
    a2 = _pv16(p8, p_scale, _v16(v_buf, slot, 2, BLOCK_N, HEAD_DIM), _vs16(vs_buf, slot, 2, HEAD_DIM), a2)
    a3 = _pv16(p8, p_scale, _v16(v_buf, slot, 3, BLOCK_N, HEAD_DIM), _vs16(vs_buf, slot, 3, HEAD_DIM), a3)
    a4 = _pv16(p8, p_scale, _v16(v_buf, slot, 4, BLOCK_N, HEAD_DIM), _vs16(vs_buf, slot, 4, HEAD_DIM), a4)
    a5 = _pv16(p8, p_scale, _v16(v_buf, slot, 5, BLOCK_N, HEAD_DIM), _vs16(vs_buf, slot, 5, HEAD_DIM), a5)
    a6 = _pv16(p8, p_scale, _v16(v_buf, slot, 6, BLOCK_N, HEAD_DIM), _vs16(vs_buf, slot, 6, HEAD_DIM), a6)
    a7 = _pv16(p8, p_scale, _v16(v_buf, slot, 7, BLOCK_N, HEAD_DIM), _vs16(vs_buf, slot, 7, HEAD_DIM), a7)
    l_acc = _l_dot(p8, p_scale, l_acc, BLOCK_N)
    return a0, a1, a2, a3, a4, a5, a6, a7, l_acc


@triton.jit
def _mfma(q, q_scale, p8, p_scale, alpha, rescale, a0, a1, a2, a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf, v_buf, vs_buf,
          k_slot, v_slot, offs_m, key0, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, MASK: tl.constexpr, row0=0):
    # The current tile's PV and row sums, then the next tile's scores, as one
    # WMMA pair per 16 output columns and per 16 keys. Each K tile's reads go
    # out between two PV pairs, so the reads queued behind the LDS do not hold
    # up the WMMAs, and each pair of scores waits only for its own K tile.
    vs0 = _vs16(vs_buf, v_slot, 0, HEAD_DIM)
    vs1 = _vs16(vs_buf, v_slot, 1, HEAD_DIM)
    vs2 = _vs16(vs_buf, v_slot, 2, HEAD_DIM)
    vs3 = _vs16(vs_buf, v_slot, 3, HEAD_DIM)
    vs4 = _vs16(vs_buf, v_slot, 4, HEAD_DIM)
    vs5 = _vs16(vs_buf, v_slot, 5, HEAD_DIM)
    vs6 = _vs16(vs_buf, v_slot, 6, HEAD_DIM)
    vs7 = _vs16(vs_buf, v_slot, 7, HEAD_DIM)
    v0 = _v16(v_buf, v_slot, 0, BLOCK_N, HEAD_DIM)
    v1 = _v16(v_buf, v_slot, 1, BLOCK_N, HEAD_DIM)
    v2 = _v16(v_buf, v_slot, 2, BLOCK_N, HEAD_DIM)
    v3 = _v16(v_buf, v_slot, 3, BLOCK_N, HEAD_DIM)
    v4 = _v16(v_buf, v_slot, 4, BLOCK_N, HEAD_DIM)
    v5 = _v16(v_buf, v_slot, 5, BLOCK_N, HEAD_DIM)
    v6 = _v16(v_buf, v_slot, 6, BLOCK_N, HEAD_DIM)
    v7 = _v16(v_buf, v_slot, 7, BLOCK_N, HEAD_DIM)
    tlx.amd_sched_barrier(0)
    a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _rescale8(a0, a1, a2, a3, a4, a5, a6, a7, l_acc, alpha, rescale)
    tlx.amd_sched_barrier(0)
    l_acc = _l_dot(p8, p_scale, l_acc, BLOCK_N)
    a0 = _pv16(p8, p_scale, v0, vs0, a0)
    a1 = _pv16(p8, p_scale, v1, vs1, a1)
    tlx.amd_sched_barrier(0)
    ks0 = _ks16(ks_buf, k_slot, 0, BLOCK_N, HEAD_DIM)
    ks1 = _ks16(ks_buf, k_slot, 1, BLOCK_N, HEAD_DIM)
    ks2 = _ks16(ks_buf, k_slot, 2, BLOCK_N, HEAD_DIM)
    ks3 = _ks16(ks_buf, k_slot, 3, BLOCK_N, HEAD_DIM)
    ks4 = _ks16(ks_buf, k_slot, 4, BLOCK_N, HEAD_DIM)
    ks5 = _ks16(ks_buf, k_slot, 5, BLOCK_N, HEAD_DIM)
    ks6 = _ks16(ks_buf, k_slot, 6, BLOCK_N, HEAD_DIM)
    ks7 = _ks16(ks_buf, k_slot, 7, BLOCK_N, HEAD_DIM)
    k0 = _k16(k_buf, k_slot, 0, BLOCK_N, HEAD_DIM)
    k1 = _k16(k_buf, k_slot, 1, BLOCK_N, HEAD_DIM)
    tlx.amd_sched_barrier(0)
    a2 = _pv16(p8, p_scale, v2, vs2, a2)
    a3 = _pv16(p8, p_scale, v3, vs3, a3)
    tlx.amd_sched_barrier(0)
    k2 = _k16(k_buf, k_slot, 2, BLOCK_N, HEAD_DIM)
    k3 = _k16(k_buf, k_slot, 3, BLOCK_N, HEAD_DIM)
    tlx.amd_sched_barrier(0)
    a4 = _pv16(p8, p_scale, v4, vs4, a4)
    a5 = _pv16(p8, p_scale, v5, vs5, a5)
    tlx.amd_sched_barrier(0)
    k4 = _k16(k_buf, k_slot, 4, BLOCK_N, HEAD_DIM)
    k5 = _k16(k_buf, k_slot, 5, BLOCK_N, HEAD_DIM)
    tlx.amd_sched_barrier(0)
    a6 = _pv16(p8, p_scale, v6, vs6, a6)
    a7 = _pv16(p8, p_scale, v7, vs7, a7)
    tlx.amd_sched_barrier(0)
    k6 = _k16(k_buf, k_slot, 6, BLOCK_N, HEAD_DIM)
    k7 = _k16(k_buf, k_slot, 7, BLOCK_N, HEAD_DIM)
    tlx.amd_sched_barrier(0)
    qk0 = _qk16(q, q_scale, k0, ks0, offs_m, key0, 0, MASK, row0)
    qk1 = _qk16(q, q_scale, k1, ks1, offs_m, key0, 1, MASK, row0)
    qk2 = _qk16(q, q_scale, k2, ks2, offs_m, key0, 2, MASK, row0)
    qk3 = _qk16(q, q_scale, k3, ks3, offs_m, key0, 3, MASK, row0)
    qk4 = _qk16(q, q_scale, k4, ks4, offs_m, key0, 4, MASK, row0)
    qk5 = _qk16(q, q_scale, k5, ks5, offs_m, key0, 5, MASK, row0)
    qk6 = _qk16(q, q_scale, k6, ks6, offs_m, key0, 6, MASK, row0)
    qk7 = _qk16(q, q_scale, k7, ks7, offs_m, key0, 7, MASK, row0)
    qk = _cat8(qk0, qk1, qk2, qk3, qk4, qk5, qk6, qk7)
    return qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc


@triton.jit
def _issue_step(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, j, v_slot, k_slot, n_tiles,
                KV_BYTES: tl.constexpr, KS_BYTES: tl.constexpr, VS_BYTES: tl.constexpr):
    # V(j + 3) and K(j + 4), into the slots of V(j - 1) and K(j).
    tlx.async_amd_descriptor_load_fused([
        (_at(v_desc, j + 3, KV_BYTES, j + 3 < n_tiles), tlx.local_view(v_buf, v_slot), 0x11),
        (_at(vs_desc, j + 3, VS_BYTES, j + 3 < n_tiles), tlx.local_view(vs_buf, v_slot), 0x22),
        (_at(k_desc, j + 4, KV_BYTES, j + 4 < n_tiles), tlx.local_view(k_buf, k_slot), 0x44),
        (_at(ks_desc, j + 4, KS_BYTES, j + 4 < n_tiles), tlx.local_view(ks_buf, k_slot), 0x88),
    ])


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
    FAST_EXP: tl.constexpr,
    ZERO_SCALE: tl.constexpr,
):
    # Q, K and the output are [Z * H, N_CTX, HEAD_DIM], V [Z * H, N_CTX // 128,
    # HEAD_DIM, 128], Q's and K's scales [Z * H, N_CTX, 4] and V's [Z * H,
    # N_CTX // 128, HEAD_DIM, 4], all contiguous.
    tl.static_assert(BLOCK_M == 2 * BLOCK_N, "a causal Q tile's diagonal crosses its last two K/V tiles")
    tl.static_assert(BLOCK_N == 128, "V's scales come in 128-key tiles")
    # Causal calls come here only with ZERO_SCALE: with qk_scale 0, -inf
    # scores would give NaN, so the softmax masks instead of the QK
    # accumulator.
    tl.static_assert(ZERO_SCALE or not CAUSAL, "causal calls with a nonzero sm_scale take the persistent kernel")
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
    k_desc = _desc(K + head, N_CTX * HEAD_DIM, KV_BYTES, True)
    v_desc = _desc(V + head, N_CTX * HEAD_DIM, KV_BYTES)
    qs_desc = _desc(Qs + head_scales, N_CTX * SCALE_K, QS_BYTES)
    ks_desc = _desc(Ks + head_scales, N_CTX * SCALE_K, KS_BYTES, True)
    vs_desc = _desc(Vs + head_scales, N_CTX * SCALE_K, VS_BYTES)
    q_buf = tlx.local_alloc((Q_BYTES // 4096, 2, 2, 4, 256), tlx.dtype_of(Q), 1, layout=_rows_layout(Q_BYTES // 256))
    k_buf = tlx.local_alloc((KV_BYTES // 4096, 2, 2, 4, 256), tlx.dtype_of(K), 4, layout=_rows_layout(KV_BYTES // 256))
    v_buf = tlx.local_alloc((KV_BYTES // 4096, 2, 2, 4, 256), tlx.dtype_of(V), 4, layout=_rows_layout(KV_BYTES // 256))
    qs_buf = tlx.local_alloc((QS_BYTES // 256, 2, 2, 2, 32), tlx.dtype_of(Qs), 1)
    ks_buf = tlx.local_alloc((KS_BYTES // 256, 2, 2, 2, 32), tlx.dtype_of(Ks), 4)
    vs_buf = tlx.local_alloc((VS_BYTES // 256, 2, 2, 2, 32), tlx.dtype_of(Vs), 4)

    # Step j takes tile j's exps and runs its PV WMMAs and tile j + 1's QK
    # WMMAs (the mfma stage), then tile j + 1's softmax, which also issues
    # V(j + 3) and K(j + 4). The two warp groups take turns on the matrix cores, a stage
    # apart. Only the mfma stage reads LDS and only the softmax stage writes
    # it, so a slot is refilled once both groups have read it. A warp's TDM
    # wait covers only its own share of a tile, so the wait at the end of step
    # j covers the tiles step j + 1 reads, before the other group can start
    # that step. TDM order: Q, K0, K1, V0, V1, K2, V2, K3 (each with its
    # scales), then V(j + 3) and K(j + 4) per step. A load past the last tile
    # is issued with its predicate off, which keeps the wait counts.
    tlx.async_amd_descriptor_load_fused([
        (_at(q_desc, start_m, Q_BYTES), tlx.local_view(q_buf, 0), 0x11),
        (_at(qs_desc, start_m, QS_BYTES), tlx.local_view(qs_buf, 0), 0x22),
        (_at(k_desc, 0, KV_BYTES), tlx.local_view(k_buf, 0), 0x44),
        (_at(ks_desc, 0, KS_BYTES), tlx.local_view(ks_buf, 0), 0x88),
    ])
    tlx.async_amd_descriptor_load_fused([
        (_at(v_desc, 0, KV_BYTES), tlx.local_view(v_buf, 0), 0x11),
        (_at(vs_desc, 0, VS_BYTES), tlx.local_view(vs_buf, 0), 0x22),
        (_at(k_desc, 1, KV_BYTES), tlx.local_view(k_buf, 1), 0x44),
        (_at(ks_desc, 1, KS_BYTES), tlx.local_view(ks_buf, 1), 0x88),
    ])
    tlx.async_amd_descriptor_load_fused([
        (_at(v_desc, 1, KV_BYTES), tlx.local_view(v_buf, 1), 0x11),
        (_at(vs_desc, 1, VS_BYTES), tlx.local_view(vs_buf, 1), 0x22),
        (_at(k_desc, 2, KV_BYTES, n_tiles > 2), tlx.local_view(k_buf, 2), 0x44),
        (_at(ks_desc, 2, KS_BYTES, n_tiles > 2), tlx.local_view(ks_buf, 2), 0x88),
    ])
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    a0 = tl.zeros([BLOCK_M, 16], tl.float32)
    a1 = tl.zeros([BLOCK_M, 16], tl.float32)
    a2 = tl.zeros([BLOCK_M, 16], tl.float32)
    a3 = tl.zeros([BLOCK_M, 16], tl.float32)
    a4 = tl.zeros([BLOCK_M, 16], tl.float32)
    a5 = tl.zeros([BLOCK_M, 16], tl.float32)
    a6 = tl.zeros([BLOCK_M, 16], tl.float32)
    a7 = tl.zeros([BLOCK_M, 16], tl.float32)
    l_acc = tl.zeros([BLOCK_M, 16], tl.float32)

    tlx.async_amd_descriptor_wait(2)
    q = _tile(q_buf, 0, BLOCK_M, HEAD_DIM)
    q_scale = _tile(qs_buf, 0, BLOCK_M, SCALE_K)
    # Only Q tile 0 meets the diagonal in K/V tile 0; the mask keeps every
    # key of it for the others.
    qk = _qk(q, q_scale, k_buf, ks_buf, 0, BLOCK_N, HEAD_DIM)
    t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, offs_m, 0, BLOCK_M, BLOCK_N, CAUSAL, FAST_EXP,
                                               ZERO_SCALE)
    t = _unpack_p8(_p8(t, offs_m, 0, BLOCK_N, CAUSAL, FAST_EXP, ZERO_SCALE))
    _issue_step(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, -1, 2, 3, n_tiles, KV_BYTES, KS_BYTES,
                VS_BYTES)
    tlx.async_amd_descriptor_wait(2)

    # Four steps per trip keep every LDS slot a constant.
    n_quads = n_plain // 4
    for jq in range(0, n_quads):
        for u in tl.static_range(4):
            j = jq * 4 + u
            with tlx.warp_pipeline_stage("mfma", priority=1):
                qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1, a2,
                                                                  a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf, v_buf,
                                                                  vs_buf, (u + 1) % 4, u, offs_m, (j + 1) * BLOCK_N,
                                                                  BLOCK_N, HEAD_DIM, False)
            tlx.async_amd_descriptor_wait(1)
            with tlx.warp_pipeline_stage("softmax", priority=0):
                # MASK only acts with ZERO_SCALE, where it picks _p8's path, which
                # each loop fixes at compile time; tiles below the diagonal get the
                # same P from either, every visible score giving the same P.
                t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, offs_m, (j + 1) * BLOCK_N, BLOCK_M,
                                                           BLOCK_N, CAUSAL, FAST_EXP, ZERO_SCALE)
                t = _p8(t, offs_m, (j + 1) * BLOCK_N, BLOCK_N, CAUSAL, FAST_EXP, ZERO_SCALE)
                tlx.amd_sched_barrier(0)
                _issue_step(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, j, (u + 3) % 4, u, n_tiles,
                            KV_BYTES, KS_BYTES, VS_BYTES)
                tlx.amd_sched_barrier(0)
                t = _unpack_p8(t)
    for j in range(n_quads * 4, n_plain):
        with tlx.warp_pipeline_stage("mfma", priority=1):
            qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1, a2, a3,
                                                              a4, a5, a6, a7, l_acc, k_buf, ks_buf, v_buf, vs_buf,
                                                              (j + 1) % 4, j % 4, offs_m, (j + 1) * BLOCK_N, BLOCK_N,
                                                              HEAD_DIM, False)
        tlx.async_amd_descriptor_wait(1)
        with tlx.warp_pipeline_stage("softmax", priority=0):
            t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, offs_m, (j + 1) * BLOCK_N, BLOCK_M, BLOCK_N,
                                                       CAUSAL, FAST_EXP, ZERO_SCALE)
            t = _p8(t, offs_m, (j + 1) * BLOCK_N, BLOCK_N, CAUSAL, FAST_EXP, ZERO_SCALE)
            tlx.amd_sched_barrier(0)
            _issue_step(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, j, (j + 3) % 4, (j + 4) % 4,
                        n_tiles, KV_BYTES, KS_BYTES, VS_BYTES)
            tlx.amd_sched_barrier(0)
            t = _unpack_p8(t)
    if CAUSAL:
        # The steps whose next tile crosses the diagonal.
        for j in range(n_plain, n_tiles - 1):
            with tlx.warp_pipeline_stage("mfma", priority=1):
                qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1, a2,
                                                                  a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf, v_buf,
                                                                  vs_buf, (j + 1) % 4, j % 4, offs_m, (j + 1) * BLOCK_N,
                                                                  BLOCK_N, HEAD_DIM, False)
            tlx.async_amd_descriptor_wait(1)
            with tlx.warp_pipeline_stage("softmax", priority=0):
                t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, offs_m, (j + 1) * BLOCK_N, BLOCK_M,
                                                           BLOCK_N, True, FAST_EXP, ZERO_SCALE)
                t = _unpack_p8(_p8(t, offs_m, (j + 1) * BLOCK_N, BLOCK_N, True, FAST_EXP, ZERO_SCALE))
                _issue_step(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, j, (j + 3) % 4, (j + 4) % 4,
                            n_tiles, KV_BYTES, KS_BYTES, VS_BYTES)
    a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _rescale8(a0, a1, a2, a3, a4, a5, a6, a7, l_acc, alpha, rescale)
    a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _pv8(t, p_scale, a0, a1, a2, a3, a4, a5, a6, a7, l_acc, v_buf, vs_buf,
                                                 (n_tiles - 1) % 4, BLOCK_N, HEAD_DIM)
    acc = _cat8(a0, a1, a2, a3, a4, a5, a6, a7)

    l_i = tl.max(l_acc, 1)
    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    o_desc = tl.make_tensor_descriptor(Out + head, shape=[N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    tlx.async_amd_descriptor_wait(0)
    o_buf = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.bfloat16, 1)
    tlx.local_store(tlx.local_view(o_buf, 0), (acc * inv_l[:, None]).to(tl.bfloat16))
    tlx.async_amd_descriptor_store(o_desc, tlx.local_view(o_buf, 0), [start_m * BLOCK_M, 0], clamp_bounds=False)
    tlx.async_amd_descriptor_wait(0)


@triton.jit
def _kv_rows(pos, n, pid, G, n_q, HEAD_ROWS):
    # The first 4096-byte row of stream K/V tile pos of this program; its
    # scales start at half that, in 256-byte rows.
    i = pos // n
    j = pos - i * n
    return (pid + i * G) // n_q * HEAD_ROWS + 4 * j


@triton.jit
def _copy_q(Q, Qs, q_buf, qs_buf, slot, g, n_q, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
            SCALE_K: tl.constexpr):
    # Q tile g and its scales, straight into the slot's 256-byte LDS rows.
    row0 = (g // n_q * N_CTX + (g % n_q) * BLOCK_M).to(tl.int64)
    cols = tl.arange(0, 256)[None, :]
    tlx.async_load(Q + row0 * HEAD_DIM + (tl.arange(0, BLOCK_M * HEAD_DIM // 256)[:, None] * 256 + cols),
                   tlx.local_view(q_buf, slot))
    tlx.async_load(Qs + row0 * SCALE_K + (tl.arange(0, BLOCK_M * SCALE_K // 256)[:, None] * 256 + cols),
                   tlx.local_view(qs_buf, slot))
    tlx.async_load_commit_group()


@triton.jit
def _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, v_rows, k_rows, v_ok, k_ok, v_slot,
              k_slot):
    tlx.async_amd_descriptor_load_fused([
        (tlx.update_tensor_descriptor(v_desc, add_offsets=[v_rows, 0, 0, 0, 0],
                                      pred=v_ok), tlx.local_view(v_buf, v_slot), 0x11),
        (tlx.update_tensor_descriptor(vs_desc, add_offsets=[v_rows >> 1, 0, 0, 0, 0],
                                      pred=v_ok), tlx.local_view(vs_buf, v_slot), 0x22),
        (tlx.update_tensor_descriptor(k_desc, add_offsets=[k_rows, 0, 0, 0, 0],
                                      pred=k_ok), tlx.local_view(k_buf, k_slot), 0x44),
        (tlx.update_tensor_descriptor(ks_desc, add_offsets=[k_rows >> 1, 0, 0, 0, 0],
                                      pred=k_ok), tlx.local_view(ks_buf, k_slot), 0x88),
    ])


@triton.jit
def _qk8(q, q_scale, k_buf, ks_buf, slot, offs_m, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # _qk as the loop computes it: a single dot is laid out for a chain head,
    # which a QK with no PV after it in its region is not.
    qk0 = _qk16(q, q_scale, _k16(k_buf, slot, 0, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 0, BLOCK_N, HEAD_DIM), offs_m,
                0, 0, False)
    qk1 = _qk16(q, q_scale, _k16(k_buf, slot, 1, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 1, BLOCK_N, HEAD_DIM), offs_m,
                0, 1, False)
    qk2 = _qk16(q, q_scale, _k16(k_buf, slot, 2, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 2, BLOCK_N, HEAD_DIM), offs_m,
                0, 2, False)
    qk3 = _qk16(q, q_scale, _k16(k_buf, slot, 3, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 3, BLOCK_N, HEAD_DIM), offs_m,
                0, 3, False)
    qk4 = _qk16(q, q_scale, _k16(k_buf, slot, 4, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 4, BLOCK_N, HEAD_DIM), offs_m,
                0, 4, False)
    qk5 = _qk16(q, q_scale, _k16(k_buf, slot, 5, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 5, BLOCK_N, HEAD_DIM), offs_m,
                0, 5, False)
    qk6 = _qk16(q, q_scale, _k16(k_buf, slot, 6, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 6, BLOCK_N, HEAD_DIM), offs_m,
                0, 6, False)
    qk7 = _qk16(q, q_scale, _k16(k_buf, slot, 7, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 7, BLOCK_N, HEAD_DIM), offs_m,
                0, 7, False)
    return _cat8(qk0, qk1, qk2, qk3, qk4, qk5, qk6, qk7)


@triton.jit
def _mxfp8_fa_fwd_persistent(
    Q,
    K,
    V,
    Qs,
    Ks,
    Vs,
    Out,
    N_CTX,
    ZH,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FAST_EXP: tl.constexpr,
    ZERO_SCALE: tl.constexpr,
):
    # Non-causal, N_CTX >= 512. Program pid takes Q tiles pid, pid + G, ...
    # and runs their K/V tiles as one stream of steps, so the pipeline does
    # not drain and refill between Q tiles.
    tl.static_assert(HEAD_DIM == 128 and BLOCK_M == 256 and BLOCK_N == 128)
    SCALE_K: tl.constexpr = HEAD_DIM // 32
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    n_q = N_CTX // BLOCK_M
    n = N_CTX // BLOCK_N
    total = ZH * n_q
    HEAD_ROWS = N_CTX // 32
    Q_BYTES: tl.constexpr = BLOCK_M * HEAD_DIM
    KV_BYTES: tl.constexpr = BLOCK_N * HEAD_DIM
    QS_BYTES: tl.constexpr = BLOCK_M * SCALE_K
    KS_BYTES: tl.constexpr = BLOCK_N * SCALE_K
    VS_BYTES: tl.constexpr = HEAD_DIM * SCALE_K
    k_desc = _desc(K, ZH * N_CTX * HEAD_DIM, KV_BYTES, True)
    v_desc = _desc(V, ZH * N_CTX * HEAD_DIM, KV_BYTES)
    ks_desc = _desc(Ks, ZH * N_CTX * SCALE_K, KS_BYTES, True)
    vs_desc = _desc(Vs, ZH * N_CTX * SCALE_K, VS_BYTES)
    q_buf = tlx.local_alloc((Q_BYTES // 256, 256), tlx.dtype_of(Q), 2,
                            layout=tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]],
                                                                                       [Q_BYTES // 256, 256], [1, 0]))
    k_buf = tlx.local_alloc((KV_BYTES // 4096, 2, 2, 4, 256), tlx.dtype_of(K), 4, layout=_rows_layout(KV_BYTES // 256))
    v_buf = tlx.local_alloc((KV_BYTES // 4096, 2, 2, 4, 256), tlx.dtype_of(V), 4, layout=_rows_layout(KV_BYTES // 256))
    qs_buf = tlx.local_alloc((QS_BYTES // 256, 256), tlx.dtype_of(Qs), 2)
    ks_buf = tlx.local_alloc((KS_BYTES // 256, 2, 2, 2, 32), tlx.dtype_of(Ks), 4)
    vs_buf = tlx.local_alloc((VS_BYTES // 256, 2, 2, 2, 32), tlx.dtype_of(Vs), 4)
    o_buf = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.bfloat16, 1)
    o_desc = tl.make_tensor_descriptor(Out, shape=[total * BLOCK_M, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])

    # Stream tile p is K/V tile p % n of Q tile p // n, and step s runs the
    # PV of p = s and the QK and softmax of p = s + 1, issuing V(s + 3) and
    # K(s + 4) as _mxfp8_fa_fwd_tdm does. Q tile i goes to slot i % 2, issued
    # in the first step of a trip that lands at places 1 to 4 of Q tile
    # i - 2 (in the prologue for i < 2), and is read in the mfma stage of the
    # step before its first QK: two stages apart from the reads of the slot
    # before it and from its load. TDM order: Q0, K0, Q1, V0, K1, V1, K2,
    # V2, K3 (each with its scales), then per step V(s + 3) and K(s + 4),
    # behind the trip's Q load in its first step.
    _copy_q(Q, Qs, q_buf, qs_buf, 0, pid, n_q, N_CTX, BLOCK_M, HEAD_DIM, SCALE_K)
    _copy_q(Q, Qs, q_buf, qs_buf, 1, tl.where(pid + G < total, pid + G, pid), n_q, N_CTX, BLOCK_M, HEAD_DIM, SCALE_K)
    tlx.async_amd_descriptor_load_fused([
        (tlx.update_tensor_descriptor(k_desc, add_offsets=[pid // n_q * HEAD_ROWS, 0, 0, 0,
                                                           0]), tlx.local_view(k_buf, 0), 0x0F),
        (tlx.update_tensor_descriptor(ks_desc, add_offsets=[pid // n_q * HEAD_ROWS // 2, 0, 0, 0,
                                                            0]), tlx.local_view(ks_buf, 0), 0xF0),
    ])
    r0 = _kv_rows(0, n, pid, G, n_q, HEAD_ROWS)
    r1 = _kv_rows(1, n, pid, G, n_q, HEAD_ROWS)
    r2 = _kv_rows(2, n, pid, G, n_q, HEAD_ROWS)
    r3 = _kv_rows(3, n, pid, G, n_q, HEAD_ROWS)
    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, r0, r1, True, True, 0, 1)
    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, r1, r2, True, True, 1, 2)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    a0 = tl.zeros([BLOCK_M, 16], tl.float32)
    a1 = tl.zeros([BLOCK_M, 16], tl.float32)
    a2 = tl.zeros([BLOCK_M, 16], tl.float32)
    a3 = tl.zeros([BLOCK_M, 16], tl.float32)
    a4 = tl.zeros([BLOCK_M, 16], tl.float32)
    a5 = tl.zeros([BLOCK_M, 16], tl.float32)
    a6 = tl.zeros([BLOCK_M, 16], tl.float32)
    a7 = tl.zeros([BLOCK_M, 16], tl.float32)
    l_acc = tl.zeros([BLOCK_M, 16], tl.float32)
    offs_m = tl.arange(0, BLOCK_M)

    tlx.async_load_wait_group(1)
    tlx.async_amd_descriptor_wait(2)
    q = _tile(q_buf, 0, BLOCK_M, HEAD_DIM)
    q_scale = _tile(qs_buf, 0, BLOCK_M, SCALE_K)
    qk = _qk8(q, q_scale, k_buf, ks_buf, 0, offs_m, BLOCK_N, HEAD_DIM)
    t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, offs_m, 0, BLOCK_M, BLOCK_N, False, FAST_EXP,
                                               ZERO_SCALE)
    t = _unpack_p8(_p8(t, offs_m, 0, BLOCK_N, False, FAST_EXP, ZERO_SCALE))
    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, r2, r3, True, True, 2, 3)
    tlx.async_amd_descriptor_wait(2)

    # Tile i's steps j = 0 .. n - 1 run the PV of its K/V tile j and the QK
    # and softmax of j + 1, issuing V(j + 3) and K(j + 4); places from n on
    # are the next tile's. n % 4 == 0, so slots go by j % 4 and the step
    # that crosses into the next tile is the last of a quad. The inner loop
    # runs the quads whose loads stay within the tile; the last quad is
    # unrolled, so the crossing, the Q switch and the O store are static.
    # Q tile i + 1 is loaded in the last step of tile i - 1, stored in step
    # n - 3 and read in step n - 1 of tile i, so the store is two stages
    # apart from the reads around it. Loads past the last tile reread it.
    n_mine = (total - pid + G - 1) // G
    g = pid
    hb = pid // n_q * HEAD_ROWS
    for it in range(0, n_mine):
        g_next = tl.where(g + G < total, g + G, g)
        hb_next = g_next // n_q * HEAD_ROWS
        kr = hb + 16
        for jq in range(0, n // 4 - 1):
            for u in tl.static_range(4):
                with tlx.warp_pipeline_stage("mfma", priority=1):
                    qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1,
                                                                      a2, a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf,
                                                                      v_buf, vs_buf, (u + 1) % 4, u, offs_m, 0, BLOCK_N,
                                                                      HEAD_DIM, False)
                tlx.async_amd_descriptor_wait(1)
                with tlx.warp_pipeline_stage("softmax", priority=0):
                    t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, offs_m, 0, BLOCK_M, BLOCK_N, False,
                                                               FAST_EXP, ZERO_SCALE)
                    t = _p8(t, offs_m, 0, BLOCK_N, False, FAST_EXP, ZERO_SCALE)
                    tlx.amd_sched_barrier(0)
                    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, kr - 4, kr, True, True,
                              (u + 3) % 4, u)
                    tlx.amd_sched_barrier(0)
                    t = _unpack_p8(t)
                    kr += 4
        for u in tl.range(0, 4, loop_unroll_factor=4):
            if u == 2:
                tlx.async_load_wait_group(0)
            with tlx.warp_pipeline_stage("mfma", priority=1):
                if u == 1:
                    # O of the previous tile, written to o_buf in its last
                    # step: both groups' writes are three stages back.
                    tlx.async_amd_descriptor_store(
                        tlx.update_tensor_descriptor(o_desc, add_offsets=[tl.where(it > 0, g - G, g) * BLOCK_M, 0],
                                                     pred=it > 0), tlx.local_view(o_buf, 0))
                if u == 3:
                    q = _tile(q_buf, (it + 1) % 2, BLOCK_M, HEAD_DIM)
                    q_scale = _tile(qs_buf, (it + 1) % 2, BLOCK_M, SCALE_K)
                qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1, a2,
                                                                  a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf, v_buf,
                                                                  vs_buf, (u + 1) % 4, u, offs_m, 0, BLOCK_N, HEAD_DIM,
                                                                  False)
            # The O store is issued in step 1 between loads, so steps 1 and 2
            # leave one more TDM op in flight, and step 3 completes it before
            # o_buf is rewritten.
            if u == 1 or u == 2:
                tlx.async_amd_descriptor_wait(2)
            else:
                tlx.async_amd_descriptor_wait(1)
            with tlx.warp_pipeline_stage("softmax", priority=0):
                if u == 3:
                    # In m_i's layout: a new tensor would be carried as more
                    # copies of m_i.
                    m_i = tl.where(offs_m < 0, m_i, float("-inf"))
                t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, offs_m, 0, BLOCK_M, BLOCK_N, False,
                                                           FAST_EXP, ZERO_SCALE)
                t = _p8(t, offs_m, 0, BLOCK_N, False, FAST_EXP, ZERO_SCALE)
                tlx.amd_sched_barrier(0)
                vr = hb_next + 4 * (u - 1)
                if u == 0:
                    vr = hb + 4 * (n - 1)
                _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, vr, hb_next + 4 * u, True,
                          True, (u + 3) % 4, u)
                tlx.amd_sched_barrier(0)
                t = _unpack_p8(t)
                if u == 3:
                    l_i = tl.max(l_acc, 1)
                    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
                    tlx.local_store(tlx.local_view(o_buf, 0),
                                    (_cat8(a0, a1, a2, a3, a4, a5, a6, a7) * inv_l[:, None]).to(tl.bfloat16))
                    a0 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a1 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a2 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a3 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a4 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a5 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a6 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a7 = tl.zeros([BLOCK_M, 16], tl.float32)
                    l_acc = tl.zeros([BLOCK_M, 16], tl.float32)
                    _copy_q(Q, Qs, q_buf, qs_buf, it % 2, tl.where(g_next + G < total, g_next + G, g_next), n_q, N_CTX,
                            BLOCK_M, HEAD_DIM, SCALE_K)
        g = g_next
        hb = hb_next
    tlx.async_amd_descriptor_store(tlx.update_tensor_descriptor(o_desc, add_offsets=[g * BLOCK_M, 0]),
                                   tlx.local_view(o_buf, 0))
    tlx.async_amd_descriptor_wait(0)
    tlx.async_load_wait_group(0)


@triton.jit
def _qk8m(q, q_scale, k_buf, ks_buf, slot, offs_m, key0, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, row0=0):
    # _qk8 for a K/V tile that may cross the diagonal.
    qk0 = _qk16(q, q_scale, _k16(k_buf, slot, 0, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 0, BLOCK_N, HEAD_DIM), offs_m,
                key0, 0, True, row0)
    qk1 = _qk16(q, q_scale, _k16(k_buf, slot, 1, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 1, BLOCK_N, HEAD_DIM), offs_m,
                key0, 1, True, row0)
    qk2 = _qk16(q, q_scale, _k16(k_buf, slot, 2, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 2, BLOCK_N, HEAD_DIM), offs_m,
                key0, 2, True, row0)
    qk3 = _qk16(q, q_scale, _k16(k_buf, slot, 3, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 3, BLOCK_N, HEAD_DIM), offs_m,
                key0, 3, True, row0)
    qk4 = _qk16(q, q_scale, _k16(k_buf, slot, 4, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 4, BLOCK_N, HEAD_DIM), offs_m,
                key0, 4, True, row0)
    qk5 = _qk16(q, q_scale, _k16(k_buf, slot, 5, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 5, BLOCK_N, HEAD_DIM), offs_m,
                key0, 5, True, row0)
    qk6 = _qk16(q, q_scale, _k16(k_buf, slot, 6, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 6, BLOCK_N, HEAD_DIM), offs_m,
                key0, 6, True, row0)
    qk7 = _qk16(q, q_scale, _k16(k_buf, slot, 7, BLOCK_N, HEAD_DIM), _ks16(ks_buf, slot, 7, BLOCK_N, HEAD_DIM), offs_m,
                key0, 7, True, row0)
    return _cat8(qk0, qk1, qk2, qk3, qk4, qk5, qk6, qk7)


@triton.jit
def _ctile(k, pid, G, ZH, n_q, HEAD_ROWS):
    # Q tile k of a causal program: the programs take the Q tiles longest
    # first in a snake order, which evens out their steps. Returns its K/V
    # rows, its index within its head, its global index and whether it exists.
    t = k * G + tl.where((k & 1) == 0, pid, G - 1 - pid)
    ok = t < ZH * n_q
    t = tl.where(ok, t, 0)
    c = t // ZH
    h = t - c * ZH
    mt = n_q - 1 - c
    return h * HEAD_ROWS, mt, h * n_q + mt, ok


@triton.jit
def _pos_rows(p, hb0, n0, hb1, n1, hb2, n2, hb3, ok1, ok2, ok3):
    # The first 4096-byte row of stream K/V tile p, counted from the current
    # Q tile's first, and whether it exists: Q tiles of two steps bring up to
    # three more Q tiles within reach.
    p1 = p - n0
    p2 = p1 - n1
    p3 = p2 - n2
    rows = tl.where(p < n0, hb0 + 4 * p, tl.where(p1 < n1, hb1 + 4 * p1, tl.where(p2 < n2, hb2 + 4 * p2, hb3 + 4 * p3)))
    ok = (p < n0) | tl.where(p1 < n1, ok1, tl.where(p2 < n2, ok2, ok3))
    return rows, ok


@triton.jit
def _mxfp8_fa_fwd_persistent_causal(
    Q,
    K,
    V,
    Qs,
    Ks,
    Vs,
    Out,
    N_CTX,
    ZH,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FAST_EXP: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # Causal, qk_scale != 0. Q tile m has 2 m + 2 K/V tiles, the last two
    # crossing the diagonal; a program runs its Q tiles' K/V tiles as one
    # stream of steps.
    tl.static_assert(HEAD_DIM == 128 and BLOCK_M == 256 and BLOCK_N == 128)
    SCALE_K: tl.constexpr = HEAD_DIM // 32
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    n_q = N_CTX // BLOCK_M
    total = ZH * n_q
    HEAD_ROWS = N_CTX // 32
    Q_BYTES: tl.constexpr = BLOCK_M * HEAD_DIM
    KV_BYTES: tl.constexpr = BLOCK_N * HEAD_DIM
    QS_BYTES: tl.constexpr = BLOCK_M * SCALE_K
    KS_BYTES: tl.constexpr = BLOCK_N * SCALE_K
    VS_BYTES: tl.constexpr = HEAD_DIM * SCALE_K
    k_desc = _desc(K, ZH * N_CTX * HEAD_DIM, KV_BYTES, True)
    v_desc = _desc(V, ZH * N_CTX * HEAD_DIM, KV_BYTES)
    ks_desc = _desc(Ks, ZH * N_CTX * SCALE_K, KS_BYTES, True)
    vs_desc = _desc(Vs, ZH * N_CTX * SCALE_K, VS_BYTES)
    q_buf = tlx.local_alloc((Q_BYTES // 256, 256), tlx.dtype_of(Q), 2,
                            layout=tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]],
                                                                                       [Q_BYTES // 256, 256], [1, 0]))
    k_buf = tlx.local_alloc((KV_BYTES // 4096, 2, 2, 4, 256), tlx.dtype_of(K), 4, layout=_rows_layout(KV_BYTES // 256))
    v_buf = tlx.local_alloc((KV_BYTES // 4096, 2, 2, 4, 256), tlx.dtype_of(V), 4, layout=_rows_layout(KV_BYTES // 256))
    qs_buf = tlx.local_alloc((QS_BYTES // 256, 256), tlx.dtype_of(Qs), 2)
    ks_buf = tlx.local_alloc((KS_BYTES // 256, 2, 2, 2, 32), tlx.dtype_of(Ks), 4)
    vs_buf = tlx.local_alloc((VS_BYTES // 256, 2, 2, 2, 32), tlx.dtype_of(Vs), 4)
    o_buf = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.bfloat16, 1)
    o_desc = tl.make_tensor_descriptor(Out, shape=[total * BLOCK_M, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])

    # Q tile i's steps j = 0 .. n - 1 run the PV of its K/V tile j and the
    # QK and softmax of j + 1 (of Q tile i + 1's first in the last step),
    # issuing V(j + 3) and K(j + 4) of the stream into slots that go by the
    # stream position. Step 0 issues the O store of Q tile i - 1, which the
    # next step's TDM wait completes before step n - 1 rewrites o_buf, and
    # copies Q tile i + 2 into the slot Q tile i was read from in step n - 1
    # of i - 1. Q tile 0 meets the diagonal in its K/V tiles 0 and 1 too.
    hb, mt, g, ok = _ctile(0, pid, G, ZH, n_q, HEAD_ROWS)
    hb1, mt1, g1, ok1 = _ctile(1, pid, G, ZH, n_q, HEAD_ROWS)
    hb2, mt2, g2, ok2 = _ctile(2, pid, G, ZH, n_q, HEAD_ROWS)
    hb3, mt3, g3, ok3 = _ctile(3, pid, G, ZH, n_q, HEAD_ROWS)
    n_t = 2 * mt + 2
    n1 = 2 * mt1 + 2
    n2 = 2 * mt2 + 2
    _copy_q(Q, Qs, q_buf, qs_buf, 0, g, n_q, N_CTX, BLOCK_M, HEAD_DIM, SCALE_K)
    _copy_q(Q, Qs, q_buf, qs_buf, 1, tl.where(ok1, g1, g), n_q, N_CTX, BLOCK_M, HEAD_DIM, SCALE_K)
    tlx.async_amd_descriptor_load_fused([
        (tlx.update_tensor_descriptor(k_desc, add_offsets=[hb, 0, 0, 0, 0]), tlx.local_view(k_buf, 0), 0x0F),
        (tlx.update_tensor_descriptor(ks_desc, add_offsets=[hb >> 1, 0, 0, 0, 0]), tlx.local_view(ks_buf, 0), 0xF0),
    ])
    r1, o1 = _pos_rows(1, hb, n_t, hb1, n1, hb2, n2, hb3, ok1, ok2, ok3)
    r2, o2 = _pos_rows(2, hb, n_t, hb1, n1, hb2, n2, hb3, ok1, ok2, ok3)
    r3, o3 = _pos_rows(3, hb, n_t, hb1, n1, hb2, n2, hb3, ok1, ok2, ok3)
    r4, o4 = _pos_rows(4, hb, n_t, hb1, n1, hb2, n2, hb3, ok1, ok2, ok3)
    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, hb, r1, True, o1, 0, 1)
    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, r1, r2, o1, o2, 1, 2)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    a0 = tl.zeros([BLOCK_M, 16], tl.float32)
    a1 = tl.zeros([BLOCK_M, 16], tl.float32)
    a2 = tl.zeros([BLOCK_M, 16], tl.float32)
    a3 = tl.zeros([BLOCK_M, 16], tl.float32)
    a4 = tl.zeros([BLOCK_M, 16], tl.float32)
    a5 = tl.zeros([BLOCK_M, 16], tl.float32)
    a6 = tl.zeros([BLOCK_M, 16], tl.float32)
    a7 = tl.zeros([BLOCK_M, 16], tl.float32)
    l_acc = tl.zeros([BLOCK_M, 16], tl.float32)
    rows = tl.arange(0, BLOCK_M)
    offs_m = mt * BLOCK_M + rows

    tlx.async_load_wait_group(1)
    tlx.async_amd_descriptor_wait(2)
    q = _tile(q_buf, 0, BLOCK_M, HEAD_DIM)
    q_scale = _tile(qs_buf, 0, BLOCK_M, SCALE_K)
    qk = _qk8m(q, q_scale, k_buf, ks_buf, 0, offs_m, 0, BLOCK_N, HEAD_DIM, mt * BLOCK_M)
    t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, rows, 0, BLOCK_M, BLOCK_N, False, FAST_EXP, False)
    t = _unpack_p8(_p8(t, rows, 0, BLOCK_N, False, FAST_EXP, False))
    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, r2, r3, o2, o3, 2, 3)
    tlx.async_amd_descriptor_wait(2)
    # Step 0 of Q tile 0.
    qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1, a2, a3, a4, a5,
                                                      a6, a7, l_acc, k_buf, ks_buf, v_buf, vs_buf, 1, 0, offs_m,
                                                      BLOCK_N, BLOCK_N, HEAD_DIM, True, mt * BLOCK_M)
    t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, rows, 0, BLOCK_M, BLOCK_N, False, FAST_EXP, False)
    t = _p8(t, rows, 0, BLOCK_N, False, FAST_EXP, False)
    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, r3, r4, o3, o4, 3, 0)
    t = _unpack_p8(t)
    tlx.async_load_wait_group(0)
    _copy_q(Q, Qs, q_buf, qs_buf, 0, tl.where(ok2, g2, g), n_q, N_CTX, BLOCK_M, HEAD_DIM, SCALE_K)
    tlx.async_amd_descriptor_wait(2)

    n_rounds = total // G
    last = tl.where((n_rounds & 1) == 0, pid, G - 1 - pid)
    n_mine = n_rounds + tl.where(last < total - n_rounds * G, 1, 0)
    b = 0
    # Only Q tile 0 meets the diagonal in its first two K/V tiles, and it comes
    # last. With SPLIT the crossings into the others run without masks, in a
    # second copy of the loop: its first, cold instruction fetch costs more
    # than the masks save unless the sequence is long.
    n_a = 0
    if SPLIT:
        t1 = ZH * (n_q - 1)
        r1 = t1 // G
        last1 = tl.where((r1 & 1) == 0, pid, G - 1 - pid)
        n_a = tl.maximum(tl.minimum(r1 + tl.where(last1 < t1 - r1 * G, 1, 0), n_mine) - 1, 0)
        for it in range(0, n_a):
            n_t = 2 * mt + 2
            n1 = 2 * mt1 + 2
            n2 = 2 * mt2 + 2
            j_mask = tl.maximum(n_t - 3, 1)
            sl_last = (b + n_t - 1) & 3
            b_next = (b + n_t) & 3
            for j in range(1, n_t - 3):
                sl = (b + j) & 3
                with tlx.warp_pipeline_stage("mfma", priority=1):
                    qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1,
                                                                      a2, a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf,
                                                                      v_buf, vs_buf, (sl + 1) & 3, sl, rows, 0, BLOCK_N,
                                                                      HEAD_DIM, False)
                tlx.async_amd_descriptor_wait(1)
                with tlx.warp_pipeline_stage("softmax", priority=0):
                    t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, rows, 0, BLOCK_M, BLOCK_N, False,
                                                               FAST_EXP, False)
                    t = _p8(t, rows, 0, BLOCK_N, False, FAST_EXP, False)
                    tlx.amd_sched_barrier(0)
                    # V(j + 3) is this Q tile's; K(j + 4) is, or the next one's
                    # first.
                    k_in = j + 4 < n_t
                    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, hb + 4 * (j + 3),
                              tl.where(k_in, hb + 4 * (j + 4), hb1), True, k_in | ok1, (sl + 3) & 3, sl)
                    tlx.amd_sched_barrier(0)
                    t = _unpack_p8(t)
            for j in range(j_mask, n_t - 1):
                sl = (b + j) & 3
                with tlx.warp_pipeline_stage("mfma", priority=1):
                    offs_m = mt * BLOCK_M + rows
                    qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1,
                                                                      a2, a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf,
                                                                      v_buf, vs_buf, (sl + 1) & 3, sl, offs_m,
                                                                      (j + 1) * BLOCK_N, BLOCK_N, HEAD_DIM, True,
                                                                      mt * BLOCK_M)
                tlx.async_amd_descriptor_wait(1)
                with tlx.warp_pipeline_stage("softmax", priority=0):
                    t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, rows, 0, BLOCK_M, BLOCK_N, False,
                                                               FAST_EXP, False)
                    t = _p8(t, rows, 0, BLOCK_N, False, FAST_EXP, False)
                    tlx.amd_sched_barrier(0)
                    # V(j + 3) is the next Q tile's first or second, K(j + 4) its
                    # second or third, which for a 2-tile one is the one after's
                    # first.
                    p = j + 4 - n_t
                    k_in = p < n1
                    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, hb1 + 4 * (p - 1),
                              tl.where(k_in, hb1 + 4 * p, hb2), ok1, tl.where(k_in, ok1, ok2), (sl + 3) & 3, sl)
                    tlx.amd_sched_barrier(0)
                    t = _unpack_p8(t)
            hb4 = hb3
            mt4 = mt3
            g4 = g3
            ok4 = ok3
            # Step n - 1 of Q tile i and step 0 of Q tile i + 1.
            for u in tl.range(0, 2, loop_unroll_factor=2):
                if u == 1:
                    tlx.async_load_wait_group(0)
                # Each step's work sits in its branch, which keeps its loop-invariant
                # parts in the stage until the loop is unrolled.
                with tlx.warp_pipeline_stage("mfma", priority=1):
                    if u == 0:
                        v_sl = sl_last
                        q = _tile(q_buf, (it + 1) & 1, BLOCK_M, HEAD_DIM)
                        q_scale = _tile(qs_buf, (it + 1) & 1, BLOCK_M, SCALE_K)
                        qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0,
                                                                          a1, a2, a3, a4, a5, a6, a7, l_acc, k_buf,
                                                                          ks_buf, v_buf, vs_buf, b_next, sl_last, rows,
                                                                          0, BLOCK_N, HEAD_DIM, False)
                    else:
                        v_sl = b_next
                        tlx.async_amd_descriptor_store(
                            tlx.update_tensor_descriptor(o_desc, add_offsets=[g * BLOCK_M, 0]),
                            tlx.local_view(o_buf, 0))
                        qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0,
                                                                          a1, a2, a3, a4, a5, a6, a7, l_acc, k_buf,
                                                                          ks_buf, v_buf, vs_buf, (b_next + 1) & 3,
                                                                          b_next, rows, BLOCK_N, BLOCK_N, HEAD_DIM,
                                                                          False)
                if u == 0:
                    tlx.async_amd_descriptor_wait(1)
                else:
                    tlx.async_amd_descriptor_wait(2)
                with tlx.warp_pipeline_stage("softmax", priority=0):
                    if u == 0:
                        # In m_i's layout: a new tensor would be carried as more
                        # copies of m_i.
                        m_i = tl.where(rows < 0, m_i, float("-inf"))
                    t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, rows, 0, BLOCK_M, BLOCK_N, False,
                                                               FAST_EXP, False)
                    if u == 0:
                        # Its accumulators start at zero.
                        rescale = rescale & False
                    t = _p8(t, rows, 0, BLOCK_N, False, FAST_EXP, False)
                    tlx.amd_sched_barrier(0)
                    vr, vo = _pos_rows(n_t + 2 + u, hb, n_t, hb1, n1, hb2, n2, hb3, ok1, ok2, ok3)
                    kr, ko = _pos_rows(n_t + 3 + u, hb, n_t, hb1, n1, hb2, n2, hb3, ok1, ok2, ok3)
                    _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, vr, kr, vo, ko,
                              (v_sl + 3) & 3, v_sl)
                    tlx.amd_sched_barrier(0)
                    t = _unpack_p8(t)
                    if u == 0:
                        l_i = tl.max(l_acc, 1)
                        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
                        tlx.local_store(tlx.local_view(o_buf, 0),
                                        (_cat8(a0, a1, a2, a3, a4, a5, a6, a7) * inv_l[:, None]).to(tl.bfloat16))
                        a0 = tl.zeros([BLOCK_M, 16], tl.float32)
                        a1 = tl.zeros([BLOCK_M, 16], tl.float32)
                        a2 = tl.zeros([BLOCK_M, 16], tl.float32)
                        a3 = tl.zeros([BLOCK_M, 16], tl.float32)
                        a4 = tl.zeros([BLOCK_M, 16], tl.float32)
                        a5 = tl.zeros([BLOCK_M, 16], tl.float32)
                        a6 = tl.zeros([BLOCK_M, 16], tl.float32)
                        a7 = tl.zeros([BLOCK_M, 16], tl.float32)
                        l_acc = tl.zeros([BLOCK_M, 16], tl.float32)
                    else:
                        _copy_q(Q, Qs, q_buf, qs_buf, (it + 1) & 1, tl.where(ok3, g3, g1), n_q, N_CTX, BLOCK_M,
                                HEAD_DIM, SCALE_K)
                        # In a stage: a stage cannot take the division in it as
                        # an input.
                        hb4, mt4, g4, ok4 = _ctile(it + 4, pid, G, ZH, n_q, HEAD_ROWS)
            b = b_next
            hb = hb1
            mt = mt1
            g = g1
            hb1 = hb2
            mt1 = mt2
            g1 = g2
            ok1 = ok2
            hb2 = hb3
            mt2 = mt3
            g2 = g3
            ok2 = ok3
            hb3 = hb4
            mt3 = mt4
            g3 = g4
            ok3 = ok4
    for it in range(n_a, n_mine):
        n_t = 2 * mt + 2
        n1 = 2 * mt1 + 2
        n2 = 2 * mt2 + 2
        j_mask = tl.maximum(n_t - 3, 1)
        sl_last = (b + n_t - 1) & 3
        b_next = (b + n_t) & 3
        for j in range(1, n_t - 3):
            sl = (b + j) & 3
            with tlx.warp_pipeline_stage("mfma", priority=1):
                qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1, a2,
                                                                  a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf, v_buf,
                                                                  vs_buf, (sl + 1) & 3, sl, rows, 0, BLOCK_N, HEAD_DIM,
                                                                  False)
            tlx.async_amd_descriptor_wait(1)
            with tlx.warp_pipeline_stage("softmax", priority=0):
                t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, rows, 0, BLOCK_M, BLOCK_N, False,
                                                           FAST_EXP, False)
                t = _p8(t, rows, 0, BLOCK_N, False, FAST_EXP, False)
                tlx.amd_sched_barrier(0)
                # V(j + 3) is this Q tile's; K(j + 4) is, or the next one's
                # first.
                k_in = j + 4 < n_t
                _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, hb + 4 * (j + 3),
                          tl.where(k_in, hb + 4 * (j + 4), hb1), True, k_in | ok1, (sl + 3) & 3, sl)
                tlx.amd_sched_barrier(0)
                t = _unpack_p8(t)
        for j in range(j_mask, n_t - 1):
            sl = (b + j) & 3
            with tlx.warp_pipeline_stage("mfma", priority=1):
                offs_m = mt * BLOCK_M + rows
                qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1, a2,
                                                                  a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf, v_buf,
                                                                  vs_buf, (sl + 1) & 3, sl, offs_m, (j + 1) * BLOCK_N,
                                                                  BLOCK_N, HEAD_DIM, True, mt * BLOCK_M)
            tlx.async_amd_descriptor_wait(1)
            with tlx.warp_pipeline_stage("softmax", priority=0):
                t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, rows, 0, BLOCK_M, BLOCK_N, False,
                                                           FAST_EXP, False)
                t = _p8(t, rows, 0, BLOCK_N, False, FAST_EXP, False)
                tlx.amd_sched_barrier(0)
                # V(j + 3) is the next Q tile's first or second, K(j + 4) its
                # second or third, which for a 2-tile one is the one after's
                # first.
                p = j + 4 - n_t
                k_in = p < n1
                _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, hb1 + 4 * (p - 1),
                          tl.where(k_in, hb1 + 4 * p, hb2), ok1, tl.where(k_in, ok1, ok2), (sl + 3) & 3, sl)
                tlx.amd_sched_barrier(0)
                t = _unpack_p8(t)
        offs_m1 = mt1 * BLOCK_M + rows
        hb4 = hb3
        mt4 = mt3
        g4 = g3
        ok4 = ok3
        # Step n - 1 of Q tile i and step 0 of Q tile i + 1.
        for u in tl.range(0, 2, loop_unroll_factor=2):
            if u == 1:
                tlx.async_load_wait_group(0)
            # Each step's work sits in its branch, which keeps its loop-invariant
            # parts in the stage until the loop is unrolled.
            with tlx.warp_pipeline_stage("mfma", priority=1):
                if u == 0:
                    v_sl = sl_last
                    q = _tile(q_buf, (it + 1) & 1, BLOCK_M, HEAD_DIM)
                    q_scale = _tile(qs_buf, (it + 1) & 1, BLOCK_M, SCALE_K)
                    qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1,
                                                                      a2, a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf,
                                                                      v_buf, vs_buf, b_next, sl_last, offs_m1, 0,
                                                                      BLOCK_N, HEAD_DIM, True, mt1 * BLOCK_M)
                else:
                    v_sl = b_next
                    tlx.async_amd_descriptor_store(tlx.update_tensor_descriptor(o_desc, add_offsets=[g * BLOCK_M, 0]),
                                                   tlx.local_view(o_buf, 0))
                    qk, a0, a1, a2, a3, a4, a5, a6, a7, l_acc = _mfma(q, q_scale, t, p_scale, alpha, rescale, a0, a1,
                                                                      a2, a3, a4, a5, a6, a7, l_acc, k_buf, ks_buf,
                                                                      v_buf, vs_buf, (b_next + 1) & 3, b_next, offs_m1,
                                                                      BLOCK_N, BLOCK_N, HEAD_DIM, True, mt1 * BLOCK_M)
            if u == 0:
                tlx.async_amd_descriptor_wait(1)
            else:
                tlx.async_amd_descriptor_wait(2)
            with tlx.warp_pipeline_stage("softmax", priority=0):
                if u == 0:
                    # In m_i's layout: a new tensor would be carried as more
                    # copies of m_i.
                    m_i = tl.where(rows < 0, m_i, float("-inf"))
                t, p_scale, m_i, alpha, rescale = _softmax(qk, m_i, qk_scale, rows, 0, BLOCK_M, BLOCK_N, False,
                                                           FAST_EXP, False)
                if u == 0:
                    # Its accumulators start at zero.
                    rescale = rescale & False
                t = _p8(t, rows, 0, BLOCK_N, False, FAST_EXP, False)
                tlx.amd_sched_barrier(0)
                vr, vo = _pos_rows(n_t + 2 + u, hb, n_t, hb1, n1, hb2, n2, hb3, ok1, ok2, ok3)
                kr, ko = _pos_rows(n_t + 3 + u, hb, n_t, hb1, n1, hb2, n2, hb3, ok1, ok2, ok3)
                _issue_kv(k_desc, ks_desc, v_desc, vs_desc, k_buf, ks_buf, v_buf, vs_buf, vr, kr, vo, ko,
                          (v_sl + 3) & 3, v_sl)
                tlx.amd_sched_barrier(0)
                t = _unpack_p8(t)
                if u == 0:
                    l_i = tl.max(l_acc, 1)
                    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
                    tlx.local_store(tlx.local_view(o_buf, 0),
                                    (_cat8(a0, a1, a2, a3, a4, a5, a6, a7) * inv_l[:, None]).to(tl.bfloat16))
                    a0 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a1 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a2 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a3 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a4 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a5 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a6 = tl.zeros([BLOCK_M, 16], tl.float32)
                    a7 = tl.zeros([BLOCK_M, 16], tl.float32)
                    l_acc = tl.zeros([BLOCK_M, 16], tl.float32)
                else:
                    _copy_q(Q, Qs, q_buf, qs_buf, (it + 1) & 1, tl.where(ok3, g3, g1), n_q, N_CTX, BLOCK_M, HEAD_DIM,
                            SCALE_K)
                    # In a stage: a stage cannot take the division in it as
                    # an input.
                    hb4, mt4, g4, ok4 = _ctile(it + 4, pid, G, ZH, n_q, HEAD_ROWS)
        b = b_next
        hb = hb1
        mt = mt1
        g = g1
        hb1 = hb2
        mt1 = mt2
        g1 = g2
        ok1 = ok2
        hb2 = hb3
        mt2 = mt3
        g2 = g3
        ok2 = ok3
        hb3 = hb4
        mt3 = mt4
        g3 = g4
        ok3 = ok4
    tlx.async_amd_descriptor_wait(0)
    tlx.async_load_wait_group(0)


_NUM_CUS = {}


def _num_cus(device):
    dev = device.index or 0
    if dev not in _NUM_CUS:
        _NUM_CUS[dev] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_CUS[dev]


# From here on the causal kernel's SPLIT copy pays for its instruction fetch.
_SPLIT_N = 4096


def _launch_quantized(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal, sm_scale, *, fast_exp=True):
    """Launch on the outputs of ``quantize_mxfp8_qkv``. ``fast_exp`` computes P
    with a linear-mantissa exp2 bit trick; ``fast_exp=False`` uses the
    hardware exp."""
    batch, heads, n_ctx, _ = q_fp8.shape
    out = torch.empty(q_fp8.shape, device=q_fp8.device, dtype=torch.bfloat16)
    n_m = n_ctx // _BLOCK_M
    if not causal and n_ctx % 512 == 0:
        _mxfp8_fa_fwd_persistent[(min(batch * heads * n_m, _num_cus(q_fp8.device)), )](
            q_fp8,
            k_fp8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            out,
            n_ctx,
            batch * heads,
            sm_scale * _LOG2E,
            HEAD_DIM=_HEAD_DIM,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            FAST_EXP=fast_exp,
            ZERO_SCALE=sm_scale == 0,
            num_warps=8,
            waves_per_eu=2,
        )
        return out
    if causal and sm_scale != 0:
        _mxfp8_fa_fwd_persistent_causal[(min(batch * heads * n_m, _num_cus(q_fp8.device)), )](
            q_fp8,
            k_fp8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            out,
            n_ctx,
            batch * heads,
            sm_scale * _LOG2E,
            HEAD_DIM=_HEAD_DIM,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            FAST_EXP=fast_exp,
            SPLIT=n_ctx >= _SPLIT_N,
            num_warps=8,
            waves_per_eu=2,
        )
        return out
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
        FAST_EXP=fast_exp,
        ZERO_SCALE=sm_scale == 0,
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
