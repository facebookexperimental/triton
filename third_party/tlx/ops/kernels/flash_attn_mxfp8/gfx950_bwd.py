"""gfx950 MXFP8 attention backward with deterministic gradient owners.

All five unique mathematical products use E4M3/E8M0. P has fixed scale
2**-8, and dS uses one RCEIL scale per 32x32 square, following sm100.py.
Separate dQ and dK/dV owners recompute QK and dOV (seven executed products)
to avoid atomics and reduce live accumulator pressure.

Scales are canonical contiguous uint8, not NVIDIA TMA packed:
Q/K/V/dO: [B,H,N,D/32]; Q_dK/K_dQ/dO_dV: [B,H,D,N/32].
Payloads have logical shape [B,H,N,D]. Sequence-reduction payloads may be
ordinary contiguous or physically [B,H,D,N] with logical strides [1,N].
"""

import math

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from .gfx950_quant import _scale_exponent


@triton.jit
def _scaled_e4m3_pairs(x, scales):
    # Packed gfx950 conversion divides both FP32 inputs by the FP32 scale.
    R: tl.constexpr = x.shape[0]
    C: tl.constexpr = x.shape[1]
    x0, x1 = tl.split(x.reshape(R, C // 2, 2))
    packed = tl.inline_asm_elementwise("v_cvt_scalef32_pk_fp8_f32 $0, $1, $2, $3", "=&v,v,v,v", [x0, x1, scales],
                                       dtype=tl.uint16, is_pure=True, pack=1)
    return tl.join(packed.to(tl.uint8), (packed >> 8).to(tl.uint8)).reshape(R, C).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _quantize_ds_square(ds, BN: tl.constexpr, BM: tl.constexpr):
    # The square recipe is transpose invariant, unlike separate 1x32 scales.
    square = ds.reshape(BN // 32, 32, BM // 32, 32)
    amax = tl.max(tl.max(tl.abs(square), 3), 1)
    exponent = _scale_exponent(amax)
    # E8M0 byte0 denotes2^-127, which is subnormal in FP32, not0.0.
    # Byte255 is NaN rather than the FP32 infinity bit pattern.
    scale_bits = tl.where(exponent == 0, 0x00400000, exponent << 23)
    scale_bits = tl.where(exponent == 255, 0x7fc00000, scale_bits)
    fp32_scale = scale_bits.to(tl.float32, bitcast=True)
    pair_scale = tl.broadcast_to(fp32_scale[:, None, :, None], (BN // 32, 32, BM // 32, 16))
    payload = _scaled_e4m3_pairs(ds, pair_scale.reshape(BN, BM // 2))
    scales = tl.broadcast_to(exponent[:, None, :], (BN // 32, 32, BM // 32)).reshape(BN, BM // 32)
    return payload, scales.to(tl.uint8)


@triton.jit
def _bwd_preprocess(O, DO, Delta, N, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, D)
    off = (head * N + row[:, None]) * D + d[None, :]
    o = tl.load(O + off, row[:, None] < N, 0).to(tl.float32)
    do = tl.load(DO + off, row[:, None] < N, 0).to(tl.float32)
    tl.store(Delta + head * N + row, tl.sum(o * do, 1), row < N)


@triton.jit
def _stage_query_inputs(qmem, domem, qdkmem, dodvmem, Q, DO, QDK, DODV, base, start, N, BM: tl.constexpr,
                        D: tl.constexpr, SEQ_K_CONTIG: tl.constexpr, EVEN_N: tl.constexpr):
    rows = start + tl.arange(0, BM)
    offsets = rows[:, None] * D + tl.arange(0, D)[None, :]
    tlx.buffer_load_to_local(qmem[0], Q + base, offsets, EVEN_N | (rows[:, None] < N), 0.0)
    tlx.buffer_load_to_local(domem[0], DO + base, offsets, EVEN_N | (rows[:, None] < N), 0.0)
    if SEQ_K_CONTIG:
        seq_offsets = tl.arange(0, D)[:, None] * N + rows[None, :]
        tlx.buffer_load_to_local(qdkmem[0], QDK + base, seq_offsets, EVEN_N | (rows[None, :] < N), 0.0)
        tlx.buffer_load_to_local(dodvmem[0], DODV + base, seq_offsets, EVEN_N | (rows[None, :] < N), 0.0)
    else:
        tlx.buffer_load_to_local(qdkmem[0], QDK + base, offsets, EVEN_N | (rows[:, None] < N), 0.0)
        tlx.buffer_load_to_local(dodvmem[0], DODV + base, offsets, EVEN_N | (rows[:, None] < N), 0.0)
    tlx.async_load_commit_group()


@triton.jit
def _bwd_kv_owner(Q, QDK, K, V, DO, DODV, QS, QDKS, KS, VS, DOS, DODVS, LSE, Delta, DK, DV, N, sm_scale,
                  D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr, SEQ_K_CONTIG: tl.constexpr,
                  EVEN_N: tl.constexpr):
    """One CTA owns the complete dK/dV reduction for BN keys."""
    key_tile = tl.program_id(1)
    head = tl.program_id(0).to(tl.int64)
    keys = key_tile * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    sd = tl.arange(0, D // 32)
    sq = tl.arange(0, BM // 32)
    base = head * N * D
    sbase = head * N * (D // 32)
    k = tl.load(K + base + keys[:, None] * D + d[None, :], EVEN_N | (keys[:, None] < N), 0.0)
    v = tl.load(V + base + keys[:, None] * D + d[None, :], EVEN_N | (keys[:, None] < N), 0.0)
    ks = tl.load(KS + sbase + keys[:, None] * (D // 32) + sd[None, :], EVEN_N | (keys[:, None] < N), 127)
    vs = tl.load(VS + sbase + keys[:, None] * (D // 32) + sd[None, :], EVEN_N | (keys[:, None] < N), 127)
    dk = tl.full((BN, D), 0.0, tl.float32)
    dv = tl.full((BN, D), 0.0, tl.float32)
    shared_layout: tl.constexpr = tlx.swizzled_layout(0, 0, 0, order=[1, 0])
    qmem = tlx.local_alloc((BM, D), tl.float8e4nv, 1, layout=shared_layout)
    domem = tlx.local_alloc((BM, D), tl.float8e4nv, 1, layout=shared_layout)
    if SEQ_K_CONTIG:
        qdkmem = tlx.local_alloc((D, BM), tl.float8e4nv, 1)
        dodvmem = tlx.local_alloc((D, BM), tl.float8e4nv, 1)
    else:
        qdkmem = tlx.local_alloc((BM, D), tl.float8e4nv, 1, layout=shared_layout)
        dodvmem = tlx.local_alloc((BM, D), tl.float8e4nv, 1, layout=shared_layout)
    begin = 0
    if CAUSAL:
        begin = (key_tile * BN // BM) * BM
    for start in range(begin, N, BM):
        queries = start + tl.arange(0, BM)
        _stage_query_inputs(qmem, domem, qdkmem, dodvmem, Q, DO, QDK, DODV, base, start, N, BM, D, SEQ_K_CONTIG, EVEN_N)
        # All four copies are committed together. Each slot is fully consumed
        # before the next iteration overwrites it.
        tlx.async_load_wait_group(0)
        q = tlx.local_load(qmem[0])
        do = tlx.local_load(domem[0])
        qs = tl.load(QS + sbase + queries[:, None] * (D // 32) + sd[None, :], EVEN_N | (queries[:, None] < N), 127)
        dos = tl.load(DOS + sbase + queries[:, None] * (D // 32) + sd[None, :], EVEN_N | (queries[:, None] < N), 127)
        scores = tlx.dot_scaled(k, ks, "e4m3", q.T, qs, "e4m3")
        lse = tl.load(LSE + head * N + queries, EVEN_N | (queries < N), 0)
        logits = scores * (sm_scale * 1.4426950408889634) - lse[None, :]
        valid = (EVEN_N | (keys[:, None] < N)) & (EVEN_N | (queries[None, :] < N))
        if CAUSAL:
            valid = valid & (keys[:, None] <= queries[None, :])
        p = tl.exp2(tl.where(valid, logits, -float("inf")))
        dp = tlx.dot_scaled(v, vs, "e4m3", do.T, dos, "e4m3")
        delta = tl.load(Delta + head * N + queries, EVEN_N | (queries < N), 0)
        ds = tl.where(valid, p * (dp - delta[None, :]), 0.0)
        ds8, dsk = _quantize_ds_square(ds, BN, BM)
        p8 = _scaled_e4m3_pairs(p, tl.full((BN, BM // 2), 0.00390625, tl.float32))
        ps = tl.full((BN, BM // 32), 119, tl.uint8)
        if SEQ_K_CONTIG:
            dodv = tlx.local_load(dodvmem[0]).T
        else:
            dodv = tlx.local_load(dodvmem[0])
        dodvs = tl.load(DODVS + sbase + d[:, None] * (N // 32) + start // 32 + sq[None, :], start // 32 + sq[None, :]
                        < N // 32, 127)
        dv = tlx.dot_scaled(p8, ps, "e4m3", dodv, dodvs, "e4m3", dv)
        if SEQ_K_CONTIG:
            qdk = tlx.local_load(qdkmem[0]).T
        else:
            qdk = tlx.local_load(qdkmem[0])
        qdks = tl.load(QDKS + sbase + d[:, None] * (N // 32) + start // 32 + sq[None, :], start // 32 + sq[None, :]
                       < N // 32, 127)
        dk = tlx.dot_scaled(ds8, dsk, "e4m3", qdk, qdks, "e4m3", dk)
    out = base + keys[:, None] * D + d[None, :]
    tl.store(DK + out, dk * sm_scale, EVEN_N | (keys[:, None] < N))
    tl.store(DV + out, dv, EVEN_N | (keys[:, None] < N))


@triton.jit
def _bwd_q_owner(Q, K, KDQ, V, DO, QS, KS, KDQS, VS, DOS, LSE, Delta, DQ, N, sm_scale, D: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr, SEQ_K_CONTIG: tl.constexpr,
                 EVEN_N: tl.constexpr):
    """One CTA owns the complete dQ reduction for BM queries."""
    queries = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, D)
    sd = tl.arange(0, D // 32)
    sn = tl.arange(0, BN // 32)
    base = head * N * D
    sbase = head * N * (D // 32)
    off = base + queries[:, None] * D + d[None, :]
    q = tl.load(Q + off, EVEN_N | (queries[:, None] < N), 0.0)
    do = tl.load(DO + off, EVEN_N | (queries[:, None] < N), 0.0)
    qs = tl.load(QS + sbase + queries[:, None] * (D // 32) + sd[None, :], EVEN_N | (queries[:, None] < N), 127)
    dos = tl.load(DOS + sbase + queries[:, None] * (D // 32) + sd[None, :], EVEN_N | (queries[:, None] < N), 127)
    lse = tl.load(LSE + head * N + queries, EVEN_N | (queries < N), 0)
    delta = tl.load(Delta + head * N + queries, EVEN_N | (queries < N), 0)
    dq = tl.full((BM, D), 0.0, tl.float32)
    kmem = tlx.local_alloc((BN, D), tl.float8e4nv, 1)
    vmem = tlx.local_alloc((BN, D), tl.float8e4nv, 1)
    if SEQ_K_CONTIG:
        kdqmem = tlx.local_alloc((D, BN), tl.float8e4nv, 1)
    else:
        kdqmem = tlx.local_alloc((BN, D), tl.float8e4nv, 1)
    end = N
    if CAUSAL:
        end = tl.minimum(N, (tl.program_id(0) + 1) * BM)
    for start in range(0, end, BN):
        keys = start + tl.arange(0, BN)
        offsets = keys[:, None] * D + d[None, :]
        tlx.buffer_load_to_local(kmem[0], K + base, offsets, EVEN_N | (keys[:, None] < N), 0.0)
        tlx.buffer_load_to_local(vmem[0], V + base, offsets, EVEN_N | (keys[:, None] < N), 0.0)
        if SEQ_K_CONTIG:
            seq_offsets = d[:, None] * N + keys[None, :]
            tlx.buffer_load_to_local(kdqmem[0], KDQ + base, seq_offsets, EVEN_N | (keys[None, :] < N), 0.0)
        else:
            tlx.buffer_load_to_local(kdqmem[0], KDQ + base, offsets, EVEN_N | (keys[:, None] < N), 0.0)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(0)
        k = tlx.local_load(kmem[0])
        v = tlx.local_load(vmem[0])
        ks = tl.load(KS + sbase + keys[:, None] * (D // 32) + sd[None, :], EVEN_N | (keys[:, None] < N), 127)
        vs = tl.load(VS + sbase + keys[:, None] * (D // 32) + sd[None, :], EVEN_N | (keys[:, None] < N), 127)
        scores = tlx.dot_scaled(q, qs, "e4m3", k.T, ks, "e4m3")
        logits = scores * (sm_scale * 1.4426950408889634) - lse[:, None]
        valid = (EVEN_N | (queries[:, None] < N)) & (EVEN_N | (keys[None, :] < N))
        if CAUSAL:
            valid = valid & (queries[:, None] >= keys[None, :])
        p = tl.exp2(tl.where(valid, logits, -float("inf")))
        dp = tlx.dot_scaled(do, dos, "e4m3", v.T, vs, "e4m3")
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        ds8, dsq = _quantize_ds_square(ds, BM, BN)
        if SEQ_K_CONTIG:
            kdq = tlx.local_load(kdqmem[0]).T
        else:
            kdq = tlx.local_load(kdqmem[0])
        kdqs = tl.load(KDQS + sbase + d[:, None] * (N // 32) + start // 32 + sn[None, :], start // 32 + sn[None, :]
                       < N // 32, 127)
        dq = tlx.dot_scaled(ds8, dsq, "e4m3", kdq, kdqs, "e4m3", dq)
    tl.store(DQ + off, dq * sm_scale, EVEN_N | (queries[:, None] < N))


def launch_backward(do, do_dv, q, q_dk, k, k_dq, v, o, lse, q_scale, q_dk_scale, k_scale, k_dq_scale, v_scale, do_scale,
                    do_dv_scale, sm_scale, do_bf16, dq, dk, dv, delta, causal=False, block_m=64, block_n=128):
    """Launch deterministic owners into preallocated outputs.

    Public validation constrains this gfx950 kernel to dense MHA, D128 and
    contiguous masters. Low-level sequence lengths must be multiples of32;
    masked loads/stores support partial64/128 tiles.
    """
    b, h, n, d = q.shape
    if d != 128 or n <= 0 or n % 32:
        raise ValueError("gfx950 MXFP8 backward requires D128 and positive N divisible by32")
    # Head bases are int64, but buffer-load intrinsics use signed i32 offsets
    # within each head. Reject overflow before forming a direct-LDS address.
    if n * d >= 2**31:
        raise ValueError("gfx950 MXFP8 backward requires per-head N*D < 2**31")
    if not math.isfinite(sm_scale):
        raise ValueError("gfx950 MXFP8 backward requires a finite sm_scale")
    assert block_m in (32, 64, 128) and block_n in (64, 128)
    seq_k_contig = q_dk.stride(-2) == 1
    assert all((x.stride(-2) == 1) == seq_k_contig for x in (k_dq, do_dv))
    even_n = n % max(block_m, block_n) == 0
    _bwd_preprocess[(triton.cdiv(n, 32), b * h)](o, do_bf16, delta, n, d, 32)
    kv_kernel = _bwd_kv_owner[(b * h, triton.cdiv(n, block_n))](q, q_dk, k, v, do, do_dv, q_scale, q_dk_scale, k_scale,
                                                                v_scale, do_scale, do_dv_scale, lse, delta, dk, dv, n,
                                                                sm_scale, D=d, BM=block_m, BN=block_n, CAUSAL=causal,
                                                                SEQ_K_CONTIG=seq_k_contig, EVEN_N=even_n, num_warps=4,
                                                                num_stages=1, matrix_instr_nonkdim=32)
    _bwd_q_owner[(triton.cdiv(n, block_n), b * h)](q, k, k_dq, v, do, q_scale, k_scale, k_dq_scale, v_scale, do_scale,
                                                   lse, delta, dq, n, sm_scale, D=d, BM=block_n, BN=block_m,
                                                   CAUSAL=causal, SEQ_K_CONTIG=seq_k_contig, EVEN_N=even_n, num_warps=4,
                                                   num_stages=1, matrix_instr_nonkdim=32)
    return kv_kernel
