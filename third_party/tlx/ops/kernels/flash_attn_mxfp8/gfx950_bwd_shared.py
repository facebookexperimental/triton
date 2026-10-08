"""gfx950 shared-square32 MXFP8 attention backward specialization.

Supports contiguous B4/H32/D128 MHA at softmax scale 0.5 for
N=1024/2048/4096/8192 with either causal flag, plus N8192 noncausal at 1.3.
Saved square32 Q/K and newly square32-quantized dO are shared between
reduction orientations; V uses feature32 quantization. This differs from
the general directional recipe in gfx950_bwd. The KV owner materializes
K-major FP8 dS and square32 E8M0 scales, then a separate FP32 dQ consumer
reuses them. Noncausal scales use logical [head, query32, key32] order.
Causal scales instead pack eight bytes per query128/key64 block in
[head, query128, key64, query32-within128, key32-within64] order. The DSS
allocation's square shape describes capacity, not this causal physical layout.
Each gradient has one writer; no atomics are used.

Per call, temporary storage is 128*N*N + 128*(N/32)**2 + 34816*N bytes:
dS, its square scales, preparation and Delta. At N8192 this is
8 GiB + 280 MiB, excluding the three 256 MiB BF16 gradients and saved
inputs. Causal calls retain the same physical workspace size, but only
initialize/read whole diagonal128 and lower-triangular attention blocks.
All allocations and launches use
the input device's current stream. The caller owns input readiness and must
not mutate inputs until their queued uses finish. Same-stream allocator
ordering (or a graph's private pool) owns temporary lifetimes after return;
there is no global workspace cache, hidden stream or allocation-failure retry.
"""

import math
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton import knobs
from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.compiler import CompiledKernel
from triton.runtime import driver
from triton.runtime.jit import _CACHE_STATS_ON, constexpr_function

from .gfx950_quant import _decode_scale, _quantize_store, _scale_exponent


@triton.jit
def _prepare_fused(V, DO, KS, VB, DO8, VS, DOS, KDQS, O, Delta, N: tl.constexpr, D: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    tl.static_assert(D == 128 and BLOCK_N == 32)
    head = tl.program_id(1).to(tl.int64)
    group = tl.program_id(0)
    rows = group * 32 + tl.arange(0, 32)
    d = tl.arange(0, D)
    offsets = (head * N + rows[:, None]) * D + d[None, :]
    valid = rows[:, None] < N
    v = tl.load(V + offsets, valid, 0).to(tl.float32)
    _quantize_store(v, VB, VS, head, rows, N, D, False, BLOCK_N)
    do = tl.load(DO + offsets, valid, 0).to(tl.float32)
    o = tl.load(O + offsets, valid, 0).to(tl.float32)
    delta_layout: tl.constexpr = tlx.layout(shape=((16, 4, 4), (8, 2)), stride=((8, 128, 512), (1, 2048)))
    products = tlx.require_layout(o * do, delta_layout, pin=True)
    delta = tl.sum(products, 1)
    tl.store(Delta + head * N + rows, tlx.release_layout(delta), rows < N)
    square = do.reshape(32, D // 32, 32)
    exponent = _scale_exponent(tl.max(tl.max(tl.abs(square), 2), 0))
    inverse = _decode_scale(exponent, RECIPROCAL=True)
    payload = tl.clamp(square * inverse[None, :, None], -448., 448.).to(tl.float8e4nv)
    tl.store(DO8 + offsets, payload.reshape(32, D), valid)
    feature_offsets = (head * N + rows[:, None]) * (D // 32) + tl.arange(0, D // 32)[None, :]
    tl.store(DOS + feature_offsets, tl.broadcast_to(exponent[None, :], (32, D // 32)).to(tl.uint8), valid)
    saved_offsets = (head * N + group * 32) * (D // 32) + d // 32
    ks = tl.load(KS + saved_offsets, group < N // 32, 0)
    seq_offsets = (head * D + d) * (N // 32) + group
    tl.store(KDQS + seq_offsets, ks, group < N // 32)


@constexpr_function
def _preparation_arena_layout(n):
    """Byte offsets for two FP8 payloads, three E8M0 scales and FP32 Delta."""
    assert n in (1024, 2048)
    payload_bytes = 128 * n * 128
    scale_bytes = 128 * n * 4
    # Each scale stores 128*n*4 bytes. Delta stores 128*n FP32 values.
    offsets = (0, payload_bytes, 2 * payload_bytes, 2 * payload_bytes + scale_bytes,
               2 * payload_bytes + 2 * scale_bytes, 2 * payload_bytes + 3 * scale_bytes)
    total_bytes = 2 * payload_bytes + 4 * scale_bytes
    assert all(offset % 16 == 0 for offset in offsets) and total_bytes % 16 == 0 and total_bytes < 2**31
    return (*offsets, total_bytes)


_PREPARATION_ARENA_BYTES = {n: _preparation_arena_layout(n)[-1] for n in (1024, 2048)}


@constexpr_function
def _backward_arena_layout(n):
    """Disjoint byte regions for preparation, dense dS and its square scales."""
    ds_offset = _preparation_arena_layout(n)[-1]
    dss_offset = ds_offset + 128 * n * n
    total_bytes = dss_offset + 128 * (n // 32)**2
    assert ds_offset % 16 == dss_offset % 16 == total_bytes % 16 == 0 and total_bytes < 2**31
    return ds_offset, dss_offset, total_bytes


_BACKWARD_ARENA_BYTES = {n: _backward_arena_layout(n)[-1] for n in (1024, 2048)}


@triton.jit
def _backward_arena_segments(Arena, ARENA_N: tl.constexpr):
    tl.static_assert(Arena.dtype.element_ty == tl.uint8)
    offsets: tl.constexpr = _backward_arena_layout(ARENA_N)
    ds = (Arena + offsets[0]).to(tl.pointer_type(tl.float8e4nv))
    dss = Arena + offsets[1]
    return ds, dss


@triton.jit
def _preparation_arena_segments(Arena, ARENA_N: tl.constexpr):
    tl.static_assert(Arena.dtype.element_ty == tl.uint8)
    offsets: tl.constexpr = _preparation_arena_layout(ARENA_N)
    vb = (Arena + offsets[0]).to(tl.pointer_type(tl.float8e4nv))
    do8 = (Arena + offsets[1]).to(tl.pointer_type(tl.float8e4nv))
    vs = Arena + offsets[2]
    dos = Arena + offsets[3]
    kdqs = Arena + offsets[4]
    delta = (Arena + offsets[5]).to(tl.pointer_type(tl.float32))
    return vb, do8, vs, dos, kdqs, delta


@triton.jit
def _prepare_fused_arena(V, DO, KS, O, Arena, N: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    vb, do8, vs, dos, kdqs, delta = _preparation_arena_segments(Arena, N)
    _prepare_fused(V, DO, KS, vb, do8, vs, dos, kdqs, O, delta, N, D, BLOCK_N)


@constexpr_function
def _stage_layout(COLS):
    # gfx950 has64 LDS banks. Keep16B chunks and XOR row bits above
    # those already selected by the row stride: perPhase=256/COLS.
    # Avoid 2-way/4-way payload read conflicts for these tile widths.
    if COLS == 128:
        return tlx.swizzled_layout(3, 4, 4, order=[1, 0])
    elif COLS == 64:
        return tlx.swizzled_layout(2, 4, 4, order=[1, 0])
    else:
        return tlx.swizzled_layout(1, 4, 4, order=[1, 0])


@triton.jit
def _load_scale_native(S, head, start, N, R: tl.constexpr, C: tl.constexpr, RHS: tl.constexpr,
                       TRANSPOSED_SQUARE: tl.constexpr):
    tl.static_assert((R == 64 and C == 4) or (R == 128 and C == 2 and RHS))
    tl.static_assert(not TRANSPOSED_SQUARE or (R == 128 and C == 2 and RHS))
    if RHS:
        layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (C // 2, R // 32)), stride=((C, 1, 0), (2, 32 * C)))
    else:
        layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (C // 2, R // 64)), stride=((C, 1, 32 * C), (2, 64 * C)))
    # Only an arange is redistributed: materialize its consumers natively.
    # Full-rank anchoring avoids unsupported slice_layout(linear) parents.
    linear = tlx.require_layout(tl.arange(0, R * C).reshape(R, C), layout, pin=True)
    row = linear // C
    col = linear % C
    if TRANSPOSED_SQUARE:
        offsets = (start + col * 32) * 4 + row // 32
    elif RHS:
        offsets = (start + row // 32 * 32) * 4 + col
    else:
        offsets = (start + row) * 4 + col
    offsets = tlx.require_layout(offsets.to(tl.int32), layout, pin=False)
    value = tlx.buffer_load(S + head.to(tl.int64) * N * 4, offsets, contiguity=1)
    return tlx.require_layout(value, layout, pin=False)


@triton.jit
def _store_ds_scale_native(S, value, head, key_start, start, N):
    layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (1, 1)), stride=((2, 1, 64), (2, 128)))
    linear = tlx.require_layout(tl.arange(0, 128).reshape(64, 2), layout, pin=True)
    row = linear // 2
    col = linear % 2
    offsets = (start // 32 + col) * (N // 32) + key_start // 32 + row // 32
    offsets = tlx.require_layout(offsets.to(tl.int32), layout, pin=False)
    mask = tlx.require_layout(row % 32 == 0, layout, pin=False)
    value = tlx.require_layout(value, layout, pin=False)
    tlx.buffer_store(value, S + head.to(tl.int64) * (N // 32) * (N // 32), offsets, mask)


@triton.jit
def _load_resident_kv64(P, base, rows, N, D: tl.constexpr, EVEN_N: tl.constexpr):
    tl.static_assert(rows.shape[0] == 64 and D == 128 and EVEN_N)
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[2, 1])
    operand: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    offsets = rows[:, None] * D + tl.arange(0, D)[None, :]
    offsets = tlx.require_layout(offsets.to(tl.int32), operand, pin=False)
    value = tlx.buffer_load(P + base, offsets, contiguity=16)
    return tlx.require_layout(value, operand, pin=False)


@triton.jit
def _load_rhs_kv64(mem, TRANS: tl.constexpr, NATIVE: tl.constexpr, RELAXED: tl.constexpr):
    tl.static_assert(NATIVE and not RELAXED)
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[2, 1])
    operand: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    if TRANS:
        value = tlx.local_load(tlx.local_trans(mem), layout=operand, relaxed=False)
    else:
        value = tlx.local_load(mem, layout=operand, relaxed=False)
    return value


@triton.jit
def _lhs_scale_kv64(value):
    R: tl.constexpr = value.shape[0]
    C: tl.constexpr = value.shape[1]
    tl.static_assert(R == 64 and (C == 2 or C == 4))
    layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (C // 2, R // 64)), stride=((C, 1, 32 * C), (2, 64 * C)))
    return tlx.require_layout(value, layout, pin=False)


@triton.jit
def _load_metadata(memory):
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[2, 1])
    columns: tl.constexpr = tlx.slice_layout(mma, 0)
    values = tlx.local_load(memory, layout=columns, relaxed=False)
    return tlx.require_layout(values, columns, pin=False)


@triton.jit
def _stage_tile(qmem, domem, lsemem, deltamem, Q, DO, LSE, Delta, base, head, start, N, BM: tl.constexpr,
                D: tl.constexpr, EVEN_N: tl.constexpr, SLOT: tl.constexpr):
    tl.static_assert(SLOT == 0 or SLOT == 1)
    copy_layout: tl.constexpr = tlx.layout(shape=((64, 2), (1, )), stride=((1, 0), (0, )))
    offsets = tlx.require_layout((start + tl.arange(0, 64)).to(tl.int32), copy_layout, pin=True)
    metadata_base = head.to(tl.int64) * N
    tlx.buffer_load_to_local(lsemem[SLOT], LSE + metadata_base, offsets)
    tlx.buffer_load_to_local(deltamem[SLOT], Delta + metadata_base, offsets)
    rows = start + tl.arange(0, BM)
    offsets = rows[:, None] * D + tl.arange(0, D)[None, :]
    tlx.buffer_load_to_local(qmem[SLOT], Q + base, offsets, EVEN_N | (rows[:, None] < N), 0.0)
    tlx.buffer_load_to_local(domem[SLOT], DO + base, offsets, EVEN_N | (rows[:, None] < N), 0.0)
    tlx.async_load_commit_group()


@triton.jit
def _guarded_p_words(x, scales):
    # P only: each call has a leading nop; low/high remain separate and tied.
    R: tl.constexpr = x.shape[0]
    C: tl.constexpr = x.shape[1]
    tl.static_assert(R == 64 and C == 64)
    tl.static_assert(scales.shape[0] == R and scales.shape[1] == C // 2)
    tl.static_assert(x.dtype == tl.float32 and scales.dtype == tl.float32)
    even, odd = tl.split(x.reshape(R, C // 2, 2))
    x0, x2 = tl.split(even.reshape(R, C // 4, 2))
    x1, x3 = tl.split(odd.reshape(R, C // 4, 2))
    s0, s1 = tl.split(scales.reshape(R, C // 4, 2))
    zero = tl.full((R, C // 4), 0, tl.uint32)
    low = tl.inline_asm_elementwise("s_nop 0\nv_cvt_scalef32_pk_fp8_f32 $0, $2, $3, $4", "=&v,0,v,v,v",
                                    [zero, x0, x1, s0], dtype=tl.uint32, is_pure=True, pack=1)
    word = tl.inline_asm_elementwise("s_nop 0\nv_cvt_scalef32_pk_fp8_f32 $0, $2, $3, $4 op_sel:[0,0,0,1]",
                                     "=&v,0,v,v,v", [low, x2, x3, s1], dtype=tl.uint32, is_pure=True, pack=1)
    halves = tl.join(word.to(tl.uint16), (word >> 16).to(tl.uint16)).reshape(R, C // 2)
    return tl.join(halves.to(tl.uint8), (halves >> 8).to(tl.uint8)).reshape(R, C).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _scaled_e4m3_words(x, scales):
    # Two separately visible native conversions. Keep both original pair scales.
    R: tl.constexpr = x.shape[0]
    C: tl.constexpr = x.shape[1]
    tl.static_assert(R == 64 and C == 64)
    tl.static_assert(scales.shape[0] == R and scales.shape[1] == C // 2)
    tl.static_assert(x.dtype == tl.float32 and scales.dtype == tl.float32)
    even, odd = tl.split(x.reshape(R, C // 2, 2))
    x0, x2 = tl.split(even.reshape(R, C // 4, 2))
    x1, x3 = tl.split(odd.reshape(R, C // 4, 2))
    s0, s1 = tl.split(scales.reshape(R, C // 4, 2))
    zero = tl.full((R, C // 4), 0, tl.uint32)
    low = tl.inline_asm_elementwise("v_cvt_scalef32_pk_fp8_f32 $0, $2, $3, $4", "=&v,0,v,v,v", [zero, x0, x1, s0],
                                    dtype=tl.uint32, is_pure=True, pack=1)
    word = tl.inline_asm_elementwise("v_cvt_scalef32_pk_fp8_f32 $0, $2, $3, $4 op_sel:[0,0,0,1]", "=&v,0,v,v,v",
                                     [low, x2, x3, s1], dtype=tl.uint32, is_pure=True, pack=1)
    halves = tl.join(word.to(tl.uint16), (word >> 16).to(tl.uint16)).reshape(R, C // 2)
    return tl.join(halves.to(tl.uint8), (halves >> 8).to(tl.uint8)).reshape(R, C).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _quantize_ds_square(ds, BN: tl.constexpr, BM: tl.constexpr):
    # A shared 32x32 scale makes the quantized dS transpose-compatible.
    square = ds.reshape(BN // 32, 32, BM // 32, 32)
    amax = tl.max(tl.max(tl.abs(square), 3), 1)
    bits = amax.to(tl.uint32, bitcast=True)
    # Exact modulo-u32 fusion of ceil exponent and FP32 scale construction.
    # Unsigned wrapping is intentional, including ordinary positive inputs.
    scale_bits = tl.add(bits, 0xfc1fffff, sanitize_overflow=False) & 0x7f800000
    scale_bits = tl.where(bits <= 0x04600001, 0x00400000, scale_bits)
    scale_bits = tl.where(bits >= 0x7f800000, 0x7fc00000, scale_bits)
    exponent = scale_bits >> 23
    fp32_scale = scale_bits.to(tl.float32, bitcast=True)
    pair_scale = tl.broadcast_to(fp32_scale[:, None, :, None], (BN // 32, 32, BM // 32, 16))
    payload = _scaled_e4m3_words(ds, pair_scale.reshape(BN, BM // 2))
    scales = tl.broadcast_to(exponent[:, None, :], (BN // 32, 32, BM // 32)).reshape(BN, BM // 32)
    return payload, scales.to(tl.uint8), exponent


@triton.jit
def _load_rhs_words(S, head, start, N):
    # Both query-square words are logically register-local in every thread.
    word_layout: tl.constexpr = tlx.layout(shape=((64, 2), (2, )), stride=((0, 0), (1, )))
    groups = tlx.require_layout(tl.arange(0, 2), word_layout, pin=True)
    word_base = (S + head.to(tl.int64) * N * 4).to(tl.pointer_type(tl.uint32))
    offsets = tlx.require_layout((start + groups * 32).to(tl.int32), word_layout, pin=True)
    words = tlx.buffer_load(word_base, offsets, contiguity=1)
    words = tlx.require_layout(words, word_layout, pin=False)
    return words


@triton.jit
def _decode_rhs_head(words):
    # Keep the asm domain at one i32 per lane half, not four i32 per head.
    half_layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), ()), stride=((0, 1, 0), ()))
    half = tlx.require_layout(tl.arange(0, 2), half_layout, pin=True)
    w0, w1 = tl.split(words.reshape(1, 2))
    selector = (0x06040200 + half * 0x01010101).to(tl.uint32)
    selector = tlx.require_layout(selector, half_layout, pin=True)
    packed = tl.inline_asm_elementwise("v_perm_b32 $0, $1, $2, $3;", "=v,v,v,v", [w1, w0, selector], dtype=tl.uint32,
                                       is_pure=True, pack=1)
    packed = tlx.require_layout(packed, half_layout, pin=True)
    # Bytes are (h, rR, rC); joins split one packed word without arithmetic.
    halves = tl.join(packed.to(tl.uint16), (packed >> 16).to(tl.uint16))
    octets = tl.join(halves.to(tl.uint8), (halves >> 8).to(tl.uint8))
    expanded = tl.broadcast_to(tlx.release_layout(octets.reshape(2, 1, 2, 2)), (2, 32, 2, 2))
    # (h,u,rR,rC) -> (rR,u,rC,h): row=u+32*rR, col=h+2*rC.
    head_scales = expanded.permute(2, 1, 3, 0).reshape(64, 4)
    head_layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (2, 2)), stride=((4, 1, 0), (2, 128)))
    return tlx.require_layout(head_scales, head_layout, pin=False)


@triton.jit
def _decode_rhs_late(words):
    late_layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (1, 4)), stride=((2, 1, 0), (2, 64)))
    late_index = tlx.require_layout(tl.arange(0, 256).reshape(128, 2), late_layout, pin=False)
    late_words = tl.broadcast_to(words.reshape(1, 2), (128, 2))
    late_words = tlx.require_layout(late_words, late_layout, pin=False)
    late_scales = ((late_words >> ((late_index // 2 // 32) * 8)) & 255).to(tl.uint8)
    late_scales = tlx.require_layout(late_scales, late_layout, pin=False)
    return late_scales


@triton.jit
def _store_ds_scale_packed(S, value, head, key_start, start, N):
    # Same four distinct byte writers as the baseline: native row0/32,
    # column0/1. Pack Q128 x K64 metadata into one aligned 8-byte record.
    layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (1, 1)), stride=((2, 1, 64), (2, 128)))
    linear = tlx.require_layout(tl.arange(0, 128).reshape(64, 2), layout, pin=True)
    row = linear // 2
    col = linear % 2
    record = (start // 128) * (N // 64) + key_start // 64
    offsets = record * 8 + ((start % 128) // 32 + col) * 2 + row // 32
    offsets = tlx.require_layout(offsets.to(tl.int32), layout, pin=False)
    mask = tlx.require_layout(row % 32 == 0, layout, pin=False)
    value = tlx.require_layout(value, layout, pin=False)
    tlx.buffer_store(value, S + head.to(tl.int64) * (N // 32) * (N // 32), offsets, mask)


@triton.jit
def _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT, D: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr, SEQ_K_CONTIG: tl.constexpr,
                  EVEN_N: tl.constexpr, IGLP: tl.constexpr, PEEL: tl.constexpr, NATIVE: tl.constexpr,
                  RELAXED: tl.constexpr, qmem, domem, qdkmem, dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head,
                  base, key_tile, keys, start, SLOT: tl.constexpr, PREFETCH, PACK_DSS: tl.constexpr):
    export_mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 64], transposed=True,
                                                   warps_per_cta=[2, 1])
    export_layout: tl.constexpr = tlx.dot_operand_layout(0, export_mma, k_width=16)
    native_rhs: tl.constexpr = tlx.dot_operand_layout(1, export_mma, k_width=16)
    phase: tl.constexpr = 0
    queries = start + tl.arange(0, BM)
    q_words = _load_rhs_words(QS, head, start, N)
    do_words = _load_rhs_words(DOS, head, start, N)
    tlx.async_load_wait_group(0)
    tlx.workgroup_barrier()
    if PREFETCH:
        _stage_tile(qmem, domem, lsemem, deltamem, Q, DO, LSE, Delta, base, head, start + BM, N, BM, D, EVEN_N,
                    1 - SLOT)
    q = _load_rhs_kv64(qmem[SLOT], True, NATIVE, RELAXED)
    do = _load_rhs_kv64(domem[SLOT], True, NATIVE, RELAXED)
    qs = _decode_rhs_head(q_words)
    scores = tlx.release_layout(
        tlx.dot_scaled(tlx.require_layout(k, export_layout, pin=False), _lhs_scale_kv64(ks), 'e4m3',
                       tlx.require_layout(q, native_rhs, pin=False), qs, 'e4m3',
                       tlx.require_layout(tl.full((BN, BM), 0.0, tl.float32), export_mma, pin=False)))
    lse = _load_metadata(lsemem[SLOT])
    logits = scores * (sm_scale * 1.4426950408889634) - lse[None, :]
    valid = (EVEN_N | (keys[:, None] < N)) & (EVEN_N | (queries[None, :] < N))
    if CAUSAL and (not PEEL or phase == 0):
        valid = valid & (keys[:, None] <= queries[None, :])
    p = tl.exp2(tl.where(valid, logits, -float('inf')))
    if IGLP:
        tlx.amd_iglp_opt(3)
    dos = _decode_rhs_head(do_words)
    dp = tlx.release_layout(
        tlx.dot_scaled(tlx.require_layout(v, export_layout, pin=False), _lhs_scale_kv64(vs), 'e4m3',
                       tlx.require_layout(do, native_rhs, pin=False), dos, 'e4m3',
                       tlx.require_layout(tl.full((BN, BM), 0.0, tl.float32), export_mma, pin=False)))
    delta = _load_metadata(deltamem[SLOT])
    ds = tl.where(valid, p * (dp - delta[None, :]), 0.0)
    ds8, dsk, unused_compact_exponent = _quantize_ds_square(ds, BN, BM)
    dsk = _lhs_scale_kv64(dsk)
    p8 = _guarded_p_words(p, tl.full((BN, BM // 2), 0.00390625, tl.float32))
    ps = tl.full((BN, BM // 32), 119, tl.uint8)
    if SEQ_K_CONTIG:
        dodv = _load_rhs_kv64(dodvmem[SLOT], True, NATIVE, RELAXED)
    else:
        dodv = _load_rhs_kv64(dodvmem[SLOT], False, NATIVE, RELAXED)
    dodvs = _decode_rhs_late(do_words)
    # LLVM 850a2b1b: mask-zero ends the head IGLP scheduling region.
    tlx.amd_sched_barrier(0)
    dv = tlx.dot_scaled(tlx.require_layout(p8, export_layout, pin=False), _lhs_scale_kv64(ps), 'e4m3',
                        tlx.require_layout(dodv, native_rhs, pin=False), dodvs, 'e4m3',
                        tlx.require_layout(dv, export_mma, pin=False))
    if SEQ_K_CONTIG:
        qdk = _load_rhs_kv64(qdkmem[SLOT], True, NATIVE, RELAXED)
    else:
        qdk = _load_rhs_kv64(qdkmem[SLOT], False, NATIVE, RELAXED)
    qdks = _decode_rhs_late(q_words)
    dk = tlx.dot_scaled(tlx.require_layout(ds8, export_layout, pin=False), _lhs_scale_kv64(dsk), 'e4m3',
                        tlx.require_layout(qdk, native_rhs, pin=False), qdks, 'e4m3',
                        tlx.require_layout(dk, export_mma, pin=False))
    export_base = head * N * N
    export_rows = tlx.require_layout(tl.arange(0, BN), tlx.slice_layout(export_layout, 1), pin=False)
    export_cols = tlx.require_layout(tl.arange(0, BM), tlx.slice_layout(export_layout, 0), pin=False)
    export_keys = key_tile * BN + export_rows
    export_queries = start + export_cols
    export_key_column = tlx.require_layout(export_keys[:, None], export_layout, pin=True)
    export_query_row = tlx.require_layout(export_queries[None, :], export_layout, pin=True)
    export_offsets = (export_key_column * N + export_query_row).to(tl.int32)
    export_offsets = tlx.require_layout(export_offsets, export_layout, pin=False)
    export_value = tlx.require_layout(ds8, export_layout, pin=False)
    tlx.buffer_store(export_value, DS_EXPORT + export_base, export_offsets)
    # Workspace ABI follows the original kernel mask mode, not this tile's
    # peeled fine-mask flag (which is False in causal bulk/drain calls).
    if PACK_DSS:
        _store_ds_scale_packed(DSS_EXPORT, dsk, head, key_tile * BN, start, N)
    else:
        _store_ds_scale_native(DSS_EXPORT, dsk, head, key_tile * BN, start, N)
    return (dk, dv)


@triton.jit
def _bwd_kv_owner(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT, D: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr, SEQ_K_CONTIG: tl.constexpr,
                  EVEN_N: tl.constexpr, IGLP: tl.constexpr = False, PEEL: tl.constexpr = False,
                  NATIVE: tl.constexpr = False, RELAXED: tl.constexpr = False, XCD_KEY_TILES: tl.constexpr = 0):
    """One CTA owns the complete dK/dV reduction for BN keys."""
    tl.static_assert(not SEQ_K_CONTIG, 'Shared-LDS specialization requires ordinary contiguous payloads')
    tl.static_assert(D == 128 and BM == 64 and (BN == 64) and NATIVE and (not RELAXED) and (not PEEL))
    tl.static_assert(EVEN_N, 'Unmasked DS export requires complete 128-row tiles')
    export_mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 64], transposed=True,
                                                   warps_per_cta=[2, 1])
    if XCD_KEY_TILES:
        # Interleave eight 32-owner chunks, as in gfx950 grouped GEMM.
        # Key-fast logical owners share a head within each chunk, grouping
        # Q/dO accesses. The host enables this only for qualified N4096/N8192 NC.
        heads: tl.constexpr = 128
        xcds: tl.constexpr = 8
        chunk: tl.constexpr = 32
        group: tl.constexpr = xcds * chunk
        tl.static_assert(not CAUSAL and (heads * XCD_KEY_TILES) % group == 0)
        physical = tl.program_id(0) + heads * tl.program_id(1)
        logical = (physical // group) * group + (physical % xcds) * chunk + (physical // xcds) % chunk
        head = (logical // XCD_KEY_TILES).to(tl.int64)
        key_tile = logical % XCD_KEY_TILES
    else:
        key_tile = tl.program_id(1)
        head = tl.program_id(0).to(tl.int64)
    keys = key_tile * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    base = head * N * D
    k = _load_resident_kv64(K, base, keys, N, D, EVEN_N)
    v = _load_resident_kv64(V, base, keys, N, D, EVEN_N)
    ks = _load_scale_native(KS, head, key_tile * BN, N, 64, 4, False, False)
    vs = _load_scale_native(VS, head, key_tile * BN, N, 64, 4, False, False)
    dk = tlx.require_layout(tl.full((BN, D), 0.0, tl.float32), export_mma, pin=False)
    dv = tlx.require_layout(tl.full((BN, D), 0.0, tl.float32), export_mma, pin=False)
    shared_layout: tl.constexpr = _stage_layout(D)
    qmem = tlx.local_alloc((BM, D), tl.float8e4nv, 2, layout=shared_layout)
    domem = tlx.local_alloc((BM, D), tl.float8e4nv, 2, layout=shared_layout)
    metadata_layout: tl.constexpr = tlx.swizzled_layout(0, 1, 1, order=[0])
    lsemem = tlx.local_alloc((BM, ), tl.float32, 2, layout=metadata_layout)
    deltamem = tlx.local_alloc((BM, ), tl.float32, 2, layout=metadata_layout)
    if SEQ_K_CONTIG:
        qdkmem = tlx.local_alloc((D, BM), tl.float8e4nv, 1, layout=_stage_layout(BM))
        dodvmem = tlx.local_alloc((D, BM), tl.float8e4nv, 1, layout=_stage_layout(BM))
    else:
        qdkmem = qmem
        dodvmem = domem
    begin = 0
    if CAUSAL:
        begin = key_tile * BN // 128 * 128
    _stage_tile(qmem, domem, lsemem, deltamem, Q, DO, LSE, Delta, base, head, begin, N, BM, D, EVEN_N, 0)
    if CAUSAL:
        # The first physical128 block is always masked. Only its second
        # tile needs a runtime prefetch decision for the final-only CTA.
        dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT, D,
                               BM, BN, CAUSAL, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem, qdkmem,
                               dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys, begin, 0,
                               True, PACK_DSS=CAUSAL)
        dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT, D,
                               BM, BN, CAUSAL, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem, qdkmem,
                               dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys, begin + 64,
                               1, begin < N - 128, PACK_DSS=CAUSAL)
        for pair_start in range(begin + 128, N - 128, 128):
            dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT,
                                   D, BM, BN, False, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem,
                                   qdkmem, dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys,
                                   pair_start, 0, True, PACK_DSS=CAUSAL)
            dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT,
                                   D, BM, BN, False, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem,
                                   qdkmem, dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys,
                                   pair_start + 64, 1, True, PACK_DSS=CAUSAL)
        if begin < N - 128:
            dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT,
                                   D, BM, BN, False, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem,
                                   qdkmem, dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys,
                                   N - 128, 0, True, PACK_DSS=CAUSAL)
            dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT,
                                   D, BM, BN, False, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem,
                                   qdkmem, dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys,
                                   N - 64, 1, False, PACK_DSS=CAUSAL)
    else:
        for pair_start in range(begin, N - 128, 128):
            dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT,
                                   D, BM, BN, CAUSAL, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem,
                                   qdkmem, dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys,
                                   pair_start, 0, True, PACK_DSS=CAUSAL)
            dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT,
                                   D, BM, BN, CAUSAL, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem,
                                   qdkmem, dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys,
                                   pair_start + 64, 1, True, PACK_DSS=CAUSAL)
        dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT, D,
                               BM, BN, CAUSAL, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem, qdkmem,
                               dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys, N - 128, 0,
                               True, PACK_DSS=CAUSAL)
        dk, dv = _compute_tile(Q, K, V, DO, QS, KS, VS, DOS, LSE, Delta, DK, DV, N, sm_scale, DS_EXPORT, DSS_EXPORT, D,
                               BM, BN, CAUSAL, SEQ_K_CONTIG, EVEN_N, IGLP, PEEL, NATIVE, RELAXED, qmem, domem, qdkmem,
                               dodvmem, lsemem, deltamem, k, v, ks, vs, dk, dv, head, base, key_tile, keys, N - 64, 1,
                               False, PACK_DSS=CAUSAL)
    out = base + keys[:, None] * D + d[None, :]
    tl.store(DK + out, dk * sm_scale, EVEN_N | (keys[:, None] < N))
    tl.store(DV + out, dv, EVEN_N | (keys[:, None] < N))


@triton.jit
def _kdqs_pair(S, head, group: tl.constexpr, start, N):
    # The prepared ABI repeats KS[group] across32 consecutive feature rows.
    groups = (N // 32).to(tl.int64)
    kg = (start // 32).to(tl.int64)
    byte_offset = (head * 128 + group * 32) * groups + kg
    # N%128==0 and BK64 starts imply kg%2==0: no partial pair exists.
    return tl.load(S.to(tl.pointer_type(tl.uint16)) + byte_offset // 2)


@triton.jit
def _uniform_kdqs_scale(S, head, start, N):
    # Host admission requires complete query128/key64 tiles. Every loop
    # start is a multiple of64 below an end that is a multiple of128.
    w0 = _kdqs_pair(S, head, 0, start, N)
    w1 = _kdqs_pair(S, head, 1, start, N)
    w2 = _kdqs_pair(S, head, 2, start, N)
    w3 = _kdqs_pair(S, head, 3, start, N)
    layout: tl.constexpr = tlx.layout(shape=((32, 2, 2, 2), (2, )), stride=((2, 1, 64, 0), (128, )))
    index = tlx.require_layout(tl.arange(0, 256).reshape(128, 2), layout, pin=True)
    # index=2*(lane&31)+(lane>>5)+64*(wave&1)+128*register.
    # The final select is register-static in this layout, not a runtime4way mux.
    low = tl.where((index & 64) == 0, w0, w1)
    high = tl.where((index & 64) == 0, w2, w3)
    word = tl.where(index < 128, low, high)
    return ((word >> ((index & 1) * 8)) & 255).to(tl.uint8)


@triton.jit
def _uniform_dss_scale_packed(S, head, logical_tile, start, N):
    # Q128 contains four square32 qgroups, K64 contains two kgroups.
    # Each record is [qgroup0:kgroup0/1, ..., qgroup3:kgroup0/1].
    record_index = (head * (N // 128) + logical_tile) * (N // 64) + start // 64
    record = tl.load(S.to(tl.pointer_type(tl.uint64)) + record_index)
    low_word = record.to(tl.uint32)
    high_word = (record >> 32).to(tl.uint32)
    native: tl.constexpr = tlx.layout(shape=((32, 2, 2, 2), (2, )), stride=((2, 1, 0, 64), (128, )))
    index = tlx.require_layout(tl.arange(0, 256).reshape(128, 2), native, pin=True)
    word = tl.where(index < 128, low_word, high_word)
    shift = (index & 64) // 4 + (index & 1) * 8
    return ((word >> shift) & 255).to(tl.uint8)


@triton.jit
def _dss_pair(S, head, query_group, start, N):
    offset = (head * (N // 32) + query_group) * (N // 32) + start // 32
    return tl.load(S.to(tl.pointer_type(tl.uint16)) + offset // 2).to(tl.uint32)


@triton.jit
def _uniform_dss_scale(S, head, logical_tile, start, N):
    # BM128 spans four qgroups; BK64 consumes two adjacent kscale bytes.
    # N/32 and start/32 are even, so every uint16 pair is aligned/in-bounds.
    qg = logical_tile.to(tl.int64) * 4
    w0 = _dss_pair(S, head, qg, start, N)
    w1 = _dss_pair(S, head, qg + 1, start, N)
    w2 = _dss_pair(S, head, qg + 2, start, N)
    w3 = _dss_pair(S, head, qg + 3, start, N)
    native: tl.constexpr = tlx.layout(shape=((32, 2, 2, 2), (2, )), stride=((2, 1, 0, 64), (128, )))
    index = tlx.require_layout(tl.arange(0, 256).reshape(128, 2), native, pin=True)
    low = tl.where((index & 64) == 0, w0, w1)
    high = tl.where((index & 64) == 0, w2, w3)
    word = tl.where(index < 128, low, high)
    return ((word >> ((index & 1) * 8)) & 255).to(tl.uint8)


@triton.jit
def _bwd_q_consume(DS, DSS, K, KDQS, DQ, N, sm_scale, D: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                   CAUSAL: tl.constexpr, HEADS: tl.constexpr):
    """Same ascending K64 chain; K-major dS is transposed as an LDS view."""
    tl.static_assert(D == 128 and BM == 128 and BK == 64 and HEADS == 128)
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[2, 2])
    lhs_layout: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    rhs_layout: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    query_tiles = tl.cdiv(N, BM).to(tl.int64)
    linear = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(tl.int64) * query_tiles
    head = linear % HEADS
    logical_tile = (query_tiles - 1 - linear // HEADS).to(tl.int32)
    queries = logical_tile * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    base = head * N * D
    dsbase = head * N * N
    dq = tlx.zeros((BM, D), tl.float32, layout=mma)
    kmem = tlx.local_alloc((BK, D), tl.float8e4nv, 1, layout=_stage_layout(D))
    dsmem = tlx.local_alloc((BK, BM), tl.float8e4nv, 1, layout=_stage_layout(BM))
    end = N
    if CAUSAL:
        end = (logical_tile + 1) * BM
    for start in range(0, end, BK):
        keys = start + tl.arange(0, BK)
        offsets = keys[:, None] * D + d[None, :]
        tlx.buffer_load_to_local(kmem[0], K + base, offsets, True, 0.0)
        # Producer visits whole key64/query64 blocks.
        # The physical mask is [key,query], not a fine-grained triangle.
        ds_offsets = (keys[:, None] * N + queries[None, :]).to(tl.int32)
        tlx.buffer_load_to_local(dsmem[0], DS + dsbase, ds_offsets, True, 0.0, cache_modifier=".cs")
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(0)
        if CAUSAL:
            dsq = _uniform_dss_scale_packed(DSS, head, logical_tile, start, N)
        else:
            dsq = _uniform_dss_scale(DSS, head, logical_tile, start, N)
        k = tlx.local_load(kmem[0], layout=rhs_layout, relaxed=False)
        ds8 = tlx.local_load(tlx.local_trans(dsmem[0]), layout=lhs_layout, relaxed=False)
        kdqs = _uniform_kdqs_scale(KDQS, head, start, N)
        dsq = tlx.require_layout(dsq, tlx.layout(shape=((32, 2, 2, 2), (2, )), stride=((2, 1, 0, 64), (128, ))),
                                 pin=False)
        kdqs = tlx.require_layout(kdqs, tlx.layout(shape=((32, 2, 2, 2), (2, )), stride=((2, 1, 64, 0), (128, ))),
                                  pin=False)
        dq = tlx.require_layout(dq, mma, pin=False)
        dq = tlx.dot_scaled(ds8, dsq, "e4m3", k, kdqs, "e4m3", dq)
        # The removed DSS LDS conversion formerly supplied this backedge WAR barrier.
        tl.debug_barrier()
    out = base + queries[:, None] * D + d[None, :]
    tl.store(DQ + out, dq * sm_scale)


@triton.jit
def _bwd_kv_owner_arena(Q, K, QS, KS, LSE, Arena, DK, DV, N, sm_scale, ARENA_N: tl.constexpr, D: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr):
    vb, do8, vs, dos, _, delta = _preparation_arena_segments(Arena, ARENA_N)
    DS, DSS = _backward_arena_segments(Arena, ARENA_N)
    _bwd_kv_owner(Q, K, vb, do8, QS, KS, vs, dos, LSE, delta, DK, DV, N, sm_scale, DS, DSS, D, BM, BN, CAUSAL,
                  SEQ_K_CONTIG=False, EVEN_N=True, IGLP=True, PEEL=False, NATIVE=True, RELAXED=False)


@triton.jit
def _bwd_q_consume_arena(K, Arena, DQ, N, sm_scale, ARENA_N: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
                         BK: tl.constexpr, CAUSAL: tl.constexpr, HEADS: tl.constexpr):
    _, _, _, _, kdqs, _ = _preparation_arena_segments(Arena, ARENA_N)
    DS, DSS = _backward_arena_segments(Arena, ARENA_N)
    _bwd_q_consume(DS, DSS, K, kdqs, DQ, N, sm_scale, D, BM, BK, CAUSAL, HEADS)


# Only compiled kernels and their JIT validity metadata are retained. Input
# tensors, temporary allocations, pointers and streams always belong to a call.
_ARENA_LAUNCH_PLANS = {}


def _arena_launch_controls():
    # A direct compiled launch skips JIT hooks and option binding. Use normal
    # JIT for instrumentation and any non-default compiler/runtime controls.
    if (_CACHE_STATS_ON or not knobs.propagate_env or knobs.runtime.interpret or knobs.runtime.debug
            or knobs.runtime.sanitize_overflow or knobs.runtime.launch_enter_hook or knobs.runtime.launch_exit_hook
            or knobs.runtime.kernel_load_start_hook or knobs.runtime.kernel_load_end_hook
            or knobs.runtime.kernel_unload_hook or knobs.runtime.jit_cache_hook is not None
            or knobs.runtime.jit_post_compile_hook is not None or knobs.runtime.add_stages_inspection_hook is not None
            or knobs.compilation.listener is not None or knobs.compilation.always_compile or knobs.compilation.override
            or knobs.compilation.dump_ir or knobs.compilation.instrumentation_mode
            or any(vars(group)
                   for group in (knobs.compilation, knobs.language, knobs.amd)) or get_cache_invalidating_env_vars()
            or os.environ.get("TRITON_COMPILE_IQ_APPLY") or os.environ.get("TRITON_COMPILE_IQ_COLLECT")):
        return False
    return True


def _arena_launch_plan(device, n, causal, sm_scale):
    if device.type != "cuda" or device.index is None or not _arena_launch_controls():
        return None
    plan = _ARENA_LAUNCH_PLANS.get((device.index, n, causal, sm_scale))
    if plan is None:
        return None
    for current, entry in zip((_prepare_fused_arena, _bwd_kv_owner_arena, _bwd_q_consume_arena), plan):
        jit, cache_key, kernel, source_hash, globals_used, runner = entry
        cache = jit.device_caches.get(device.index)
        if (jit is not current or jit.pre_run_hooks or jit.launch_metadata is not None or jit.debug
                or jit.hash != source_hash or cache is None or cache[0].get(cache_key) is not kernel
                or getattr(kernel, "_compile_iq_acf_cubin", None) is not None):
            return None
        for name, value, namespace in globals_used:
            if name not in namespace or namespace[name] != value:
                return None
    return tuple(entry[5] for entry in plan)


def _remember_arena_launch_plan(device, n, causal, sm_scale, compiled):
    if device.type != "cuda" or device.index is None or not _arena_launch_controls():
        return
    entries = []
    grids = ((n // 32, 128, 1), (128, n // 64, 1), (n // 128, 128, 1))
    for jit, kernel, grid in zip((_prepare_fused_arena, _bwd_kv_owner_arena, _bwd_q_consume_arena), compiled, grids):
        cache = jit.device_caches.get(device.index)
        if (not isinstance(kernel, CompiledKernel) or cache is None or jit.pre_run_hooks
                or jit.launch_metadata is not None or jit.debug or kernel.src.fn is not jit):
            return
        cache_key = next((key for key, cached in cache[0].items() if cached is kernel), None)
        if cache_key is None:
            return
        globals_used = tuple((name, value, namespace) for (name, _), (value, namespace) in jit.used_global_vals.items())
        # JIT source edits require callers' hashes/caches to be invalidated,
        # as documented by JITCallable._unsafe_update_src. Track that contract.
        entries.append((jit, cache_key, kernel, jit.hash, globals_used, kernel[grid]))
    _ARENA_LAUNCH_PLANS[(device.index, n, causal, sm_scale)] = tuple(entries)


def _check_shared_square_inputs(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse, sm_scale, causal):
    """Metadata-only admission; this never reads payload or scale values."""
    if type(causal) is not bool:
        raise ValueError("Shared-square backward requires an actual bool causal flag")
    if not isinstance(q_fp8, torch.Tensor) or q_fp8.device.type != "cuda":
        raise ValueError("Shared-square backward requires GPU tensors")
    shape = tuple(q_fp8.shape)
    if len(shape) != 4 or shape[:2] != (4, 32) or shape[2] not in (1024, 2048, 4096, 8192) or shape[3] != 128:
        raise ValueError("Shared-square backward requires B4/H32/D128 and N1024/2048/4096/8192")
    n = shape[2]
    if (type(sm_scale) is not float or not math.isfinite(sm_scale)
            or not (sm_scale == 0.5 or (sm_scale == 1.3 and n == 8192 and not causal))):
        raise ValueError("Shared-square backward requires Python float sm_scale=0.5, or N8192 noncausal at 1.3")
    scale_shape = (4, 32, n, 4)
    payload_bytes = 4 * 32 * n * 128
    scale_bytes = 4 * 32 * n * 4
    specs = (
        ("q_fp8", q_fp8, shape, torch.float8_e4m3fn, payload_bytes),
        ("k_fp8", k_fp8, shape, torch.float8_e4m3fn, payload_bytes),
        ("q_scale", q_scale, scale_shape, torch.uint8, scale_bytes),
        ("k_scale", k_scale, scale_shape, torch.uint8, scale_bytes),
        ("v_bf16", v_bf16, shape, torch.bfloat16, 2 * payload_bytes),
        ("do_bf16", do_bf16, shape, torch.bfloat16, 2 * payload_bytes),
        ("out_bf16", out_bf16, shape, torch.bfloat16, 2 * payload_bytes),
        ("lse", lse, shape[:-1], torch.float32, scale_bytes),
    )
    device = q_fp8.device
    for name, tensor, expected_shape, dtype, expected_bytes in specs:
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.shape != expected_shape
                or tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous()
                or tensor.storage_offset() != 0 or tensor.is_conj() or tensor.is_neg()):
            raise ValueError("Invalid shared-square tensor metadata: " + name)
        storage = tensor.untyped_storage()
        data_ptr = tensor.data_ptr()
        if (storage.nbytes() != expected_bytes or data_ptr != storage.data_ptr() or data_ptr % 16 != 0):
            raise ValueError("Whole, 16-byte-aligned storage required: " + name)
    return device, shape


def _is_gfx950(device):
    if torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0] == "gfx950"


def can_use_shared_square(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse, sm_scale, *, causal=False):
    """Metadata-only dispatch guard; unsupported inputs use the general path."""
    try:
        device, _ = _check_shared_square_inputs(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse,
                                                sm_scale, causal)
    except ValueError:
        return False
    return _is_gfx950(device)


def _try_launch_backward_shared_square(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse, sm_scale, *,
                                       causal=False):
    """Internal autograd dispatch; the caller already holds the input device.

    Return None only for unsupported metadata or architecture. Validation
    and launch share one invocation; no validation result is cached across
    calls. Allocation and kernel errors must propagate, not trigger fallback.
    """
    try:
        device, shape = _check_shared_square_inputs(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse,
                                                    sm_scale, causal)
    except ValueError:
        return None
    if not _is_gfx950(device):
        return None
    return _launch_backward_shared_square(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse, sm_scale,
                                          causal, device, shape)


def launch_backward_shared_square(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse, sm_scale, *,
                                  causal=False):
    """Return BF16 (dQ, dK, dV) for the admitted shared-square recipe.

    q_fp8/k_fp8 and their uint8 scales MUST be the saved output of the
    production square32 forward quantizer, including repetition over each
    group of 32 sequence rows. out_bf16 and base-2 lse MUST belong to that
    same forward invocation and sm_scale. V/dO/O are original BF16 operands;
    Delta is reduced from original O*dO, not dequantized payloads.

    Those value/provenance invariants are the caller's responsibility: this
    entry validates metadata only, with no host copies or GPU-value checks.
    It does not normalize unsupported inputs or silently fall back after
    allocation or launch failure. Other scales use the general backward path.

    Per-call workspace is allocated before the first kernel submission.
    Capturing callers must retain the graph/private pool according to Torch's
    normal CUDA/HIP graph contract. Calls on different streams get distinct
    workspaces; producer-to-consumer ordering is entirely on the same stream.
    """
    device, shape = _check_shared_square_inputs(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse,
                                                sm_scale, causal)
    with torch.cuda.device(device):
        if not _is_gfx950(device):
            raise ValueError("Shared-square backward is gfx950-only")
        return _launch_backward_shared_square(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse, sm_scale,
                                              causal, device, shape)


def _launch_backward_shared_square(q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse, sm_scale, causal,
                                   device, shape):
    """Launch after metadata/architecture checks, inside the input device context."""
    n = shape[2]
    feature_scale_shape = (4, 32, n, 4)
    sequence_scale_shape = (4, 32, 128, n // 32)
    # Derive temporary pointers inside JIT kernels instead of creating Torch
    # views. Each N1024/N2048 call owns preparation, DS and DSS in one arena.
    if n in _PREPARATION_ARENA_BYTES:
        arena = torch.empty((_BACKWARD_ARENA_BYTES[n], ), dtype=torch.uint8, device=device)
    else:
        vb = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
        do8 = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
        vs = torch.empty(feature_scale_shape, dtype=torch.uint8, device=device)
        dos = torch.empty(feature_scale_shape, dtype=torch.uint8, device=device)
        kdqs = torch.empty(sequence_scale_shape, dtype=torch.uint8, device=device)
        delta = torch.empty(shape[:-1], dtype=torch.float32, device=device)
    # Never cap or reinterpret the whole DS allocation as range32.
    # Kernels use i64 head bases and bounded per-head i32 offsets.
    # For causal calls the producer starts at floor(key_start/128)*128,
    # writing both query64 tiles (including fine-diagonal masked zeros).
    # The query128 consumer reads only these initialized DS/DSS blocks.
    # Capacity is square32 in both modes. Causal physical scale records are
    # [4, 32, n // 128, n // 64, 4, 2]; each key64 owner writes one full
    # eight-byte record across its two query64 visits, without shared words.
    if n not in _PREPARATION_ARENA_BYTES:
        ds = torch.empty((4, 32, n, n), dtype=torch.float8_e4m3fn, device=device)
        dss = torch.empty((4, 32, n // 32, n // 32), dtype=torch.uint8, device=device)
    dq = torch.empty(shape, dtype=torch.bfloat16, device=device)
    dk = torch.empty(shape, dtype=torch.bfloat16, device=device)
    dv = torch.empty(shape, dtype=torch.bfloat16, device=device)

    if n in _PREPARATION_ARENA_BYTES:
        # Every input was admitted in this call. Subclasses retain normal JIT
        # specialization because they may override tensor metadata access.
        ordinary_inputs = all(
            type(tensor) is torch.Tensor for tensor in (q_fp8, k_fp8, q_scale, k_scale, v_bf16, do_bf16, out_bf16, lse))
        plan = _arena_launch_plan(device, n, causal, sm_scale) if ordinary_inputs else None
        if plan is not None:
            prepare, kv, query = plan
            # Resolve one invocation-local stream for this producer/consumer
            # chain. The cached public runners contain only kernels and grids.
            stream = driver.active.get_current_stream(device.index)
            # Pass the complete original ABI, including constexpr positions.
            prepare(v_bf16, do_bf16, k_scale, out_bf16, arena, n, 128, 32, stream=stream)
            kv(q_fp8, k_fp8, q_scale, k_scale, lse, arena, dk, dv, n, sm_scale, n, 128, 64, 64, causal,
               stream=stream)
            query(k_fp8, arena, dq, n, sm_scale, n, 128, 128, 64, causal, 128, stream=stream)
        else:
            # N stays runtime in both reduction kernels; ARENA_N only fixes
            # preparation byte offsets. Cold JIT errors propagate unchanged.
            prepare = _prepare_fused_arena.run(v_bf16, do_bf16, k_scale, out_bf16, arena, n, 128, 32, num_warps=4,
                                               num_stages=2, grid=(n // 32, 128), warmup=False)
            kv = _bwd_kv_owner_arena.run(q_fp8, k_fp8, q_scale, k_scale, lse, arena, dk, dv, n, sm_scale,
                                         ARENA_N=n, D=128, BM=64, BN=64, CAUSAL=causal, num_warps=2, num_stages=1,
                                         matrix_instr_nonkdim=32, waves_per_eu=0, grid=(128, n // 64), warmup=False)
            query = _bwd_q_consume_arena.run(k_fp8, arena, dq, n, sm_scale, ARENA_N=n, D=128, BM=128, BK=64,
                                             CAUSAL=causal, HEADS=128, num_warps=4, num_stages=1,
                                             matrix_instr_nonkdim=32, waves_per_eu=0, grid=(n // 128, 128),
                                             warmup=False)
            if ordinary_inputs:
                _remember_arena_launch_plan(device, n, causal, sm_scale, (prepare, kv, query))
    else:
        _prepare_fused.run(v_bf16, do_bf16, k_scale, vb, do8, vs, dos, kdqs, out_bf16, delta, n, 128, 32, num_warps=4,
                           num_stages=2, grid=(n // 32, 128), warmup=False)
        # The KV owner reuses Q/dO payloads and their saved/prepared square32
        # scales directly; no directional exports or second Delta launch.
        _bwd_kv_owner.run(q_fp8, k_fp8, vb, do8, q_scale, k_scale, vs, dos, lse, delta, dk, dv, n, sm_scale, ds, dss,
                          D=128, BM=64, BN=64, CAUSAL=causal, SEQ_K_CONTIG=False, EVEN_N=True, IGLP=True, PEEL=False,
                          NATIVE=True, RELAXED=False,
                          XCD_KEY_TILES=(n // 64 if not causal and n in (4096, 8192) else 0), num_warps=2, num_stages=1,
                          matrix_instr_nonkdim=32, waves_per_eu=0, grid=(128, n // 64), warmup=False)
        _bwd_q_consume.run(ds, dss, k_fp8, kdqs, dq, n, sm_scale, D=128, BM=128, BK=64, CAUSAL=causal, HEADS=128,
                           num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=0, grid=(n // 128, 128),
                           warmup=False)
    return dq, dk, dv


@triton.jit
def _prepare_mxfp8(DO, KS, DO8, DOS, KDQS, O, Delta, N: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    tl.static_assert(D == 128 and BLOCK_N == 32)
    head = tl.program_id(1).to(tl.int64)
    group = tl.program_id(0)
    rows = group * 32 + tl.arange(0, 32)
    d = tl.arange(0, D)
    offsets = (head * N + rows[:, None]) * D + d[None, :]
    valid = rows[:, None] < N
    do = tl.load(DO + offsets, valid, 0).to(tl.float32)
    o = tl.load(O + offsets, valid, 0).to(tl.float32)
    delta_layout: tl.constexpr = tlx.layout(shape=((16, 4, 4), (8, 2)), stride=((8, 128, 512), (1, 2048)))
    products = tlx.require_layout(o * do, delta_layout, pin=True)
    delta = tl.sum(products, 1)
    tl.store(Delta + head * N + rows, tlx.release_layout(delta), rows < N)
    square = do.reshape(32, D // 32, 32)
    exponent = _scale_exponent(tl.max(tl.max(tl.abs(square), 2), 0))
    inverse = _decode_scale(exponent, RECIPROCAL=True)
    payload = tl.clamp(square * inverse[None, :, None], -448., 448.).to(tl.float8e4nv)
    tl.store(DO8 + offsets, payload.reshape(32, D), valid)
    feature_offsets = (head * N + rows[:, None]) * (D // 32) + tl.arange(0, D // 32)[None, :]
    tl.store(DOS + feature_offsets, tl.broadcast_to(exponent[None, :], (32, D // 32)).to(tl.uint8), valid)
    saved_offsets = (head * N + group * 32) * (D // 32) + d // 32
    ks = tl.load(KS + saved_offsets, group < N // 32, 0)
    seq_offsets = (head * D + d) * (N // 32) + group
    tl.store(KDQS + seq_offsets, ks, group < N // 32)


def _check_mxfp8_backward_shape(reference, sm_scale, causal):
    """Shared metadata-only shape/scale admission for explicit-input APIs."""
    if type(causal) is not bool:
        raise ValueError("Shared-square backward requires an actual bool causal flag")
    if not isinstance(reference, torch.Tensor) or reference.device.type != "cuda":
        raise ValueError("Shared-square backward requires GPU tensors")
    shape = tuple(reference.shape)
    if len(shape) != 4 or shape[:2] != (4, 32) or shape[2] not in (1024, 2048, 4096, 8192) or shape[3] != 128:
        raise ValueError("Shared-square backward requires B4/H32/D128 and N1024/2048/4096/8192")
    if (type(sm_scale) is not float or not math.isfinite(sm_scale)
            or not (sm_scale == 0.5 or (sm_scale == 1.3 and shape[2] == 8192 and not causal))):
        raise ValueError("Shared-square backward requires Python float sm_scale=0.5, or N8192 noncausal at1.3")
    return reference.device, shape


def _check_mxfp8_backward_tensors(device, specs):
    for name, tensor, expected_shape, dtype in specs:
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tuple(tensor.shape) != expected_shape or tensor.dtype != dtype or tensor.device != device
                or not tensor.is_contiguous() or tensor.storage_offset() != 0 or tensor.is_conj() or tensor.is_neg()):
            raise ValueError("Invalid shared-square tensor metadata: " + name)
        storage = tensor.untyped_storage()
        data_ptr = tensor.data_ptr()
        if (storage.nbytes() != tensor.numel() * tensor.element_size() or data_ptr != storage.data_ptr()
                or data_ptr % 16 != 0):
            raise ValueError("Whole,16-byte-aligned storage required: " + name)


def _mxfp8_backward_specs(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, lse, shape):
    scale_shape = shape[:-1] + (4, )
    return (
        ("q_fp8", q_fp8, shape, torch.float8_e4m3fn),
        ("k_fp8", k_fp8, shape, torch.float8_e4m3fn),
        ("v_fp8", v_fp8, shape, torch.float8_e4m3fn),
        ("q_scale", q_scale, scale_shape, torch.uint8),
        ("k_scale", k_scale, scale_shape, torch.uint8),
        ("v_scale", v_scale, scale_shape, torch.uint8),
        ("lse", lse, shape[:-1], torch.float32),
    )


def _allocate_mxfp8_backward_preparation(device, shape):
    n = shape[2]
    do_fp8 = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    do_scale = torch.empty(shape[:-1] + (4, ), dtype=torch.uint8, device=device)
    delta = torch.empty(shape[:-1], dtype=torch.float32, device=device)
    k_dq_scale = torch.empty((4, 32, 128, n // 32), dtype=torch.uint8, device=device)
    return do_fp8, do_scale, delta, k_dq_scale


def _allocate_mxfp8_backward_workspace(device, shape):
    n = shape[2]
    ds = torch.empty((4, 32, n, n), dtype=torch.float8_e4m3fn, device=device)
    # Physical causal DSS is [B,H,N//128,N//64,4,2], as in the legacy API.
    dss = torch.empty((4, 32, n // 32, n // 32), dtype=torch.uint8, device=device)
    dq = torch.empty(shape, dtype=torch.bfloat16, device=device)
    dk = torch.empty(shape, dtype=torch.bfloat16, device=device)
    dv = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return ds, dss, dq, dk, dv


def _launch_mxfp8_backward_preparation(k_scale, do_bf16, out_bf16, prepared, shape):
    do_fp8, do_scale, delta, k_dq_scale = prepared
    n = shape[2]
    _prepare_mxfp8[(n // 32, 128)](do_bf16, k_scale, do_fp8, do_scale, k_dq_scale, out_bf16, delta, n, 128, 32,
                                   num_warps=4, num_stages=2)


def _launch_mxfp8_backward_core(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, lse, sm_scale, causal, prepared,
                                workspace, shape):
    do_fp8, do_scale, delta, k_dq_scale = prepared
    ds, dss, dq, dk, dv = workspace
    n = shape[2]
    _bwd_kv_owner[(128, n // 64)](q_fp8, k_fp8, v_fp8, do_fp8, q_scale, k_scale, v_scale, do_scale, lse, delta, dk, dv,
                                  n, sm_scale, ds, dss, D=128, BM=64, BN=64, CAUSAL=causal, SEQ_K_CONTIG=False,
                                  EVEN_N=True, IGLP=True, PEEL=False, NATIVE=True, RELAXED=False, num_warps=2,
                                  num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=0)
    _bwd_q_consume[(n // 128, 128)](ds, dss, k_fp8, k_dq_scale, dq, n, sm_scale, D=128, BM=128, BK=64, CAUSAL=causal,
                                    HEADS=128, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=0)
    return dq, dk, dv


def prepare_backward_shared_square_mxfp8(k_scale, do_bf16, out_bf16, sm_scale, *, causal=False):
    """Return caller-owned (dO8, dO_scale, Delta, K_dQ_scale) for the core API.

    dO/O are original BF16 tensors. Delta is the original FP32 reduction
    sum(O*dO), NOT a reduction using dequantized dO8. dO is square32-quantized
    once; its payload/scales serve both backward reduction orientations.
    k_scale must be the saved square32 K scale with repeated sequence rows.
    The result contains E4M3 [B,H,N,128], E8M0 [B,H,N,4], FP32 [B,H,N],
    and E8M0 [B,H,128,N/32], respectively.

    This checked API allocates four outputs and launches only preparation,
    making its timing boundary explicit. sm_scale/causal select the same
    supported configurations as full backward, but do not change preparation.
    Metadata is checked; scale provenance/row repetition are caller invariants.
    Work uses the input device's current stream. Callers own result lifetime,
    cross-stream readiness and graph retention; nothing is cached globally.
    """
    device, shape = _check_mxfp8_backward_shape(do_bf16, sm_scale, causal)
    _check_mxfp8_backward_tensors(device, (
        ("k_scale", k_scale, shape[:-1] + (4, ), torch.uint8),
        ("do_bf16", do_bf16, shape, torch.bfloat16),
        ("out_bf16", out_bf16, shape, torch.bfloat16),
    ))
    with torch.cuda.device(device):
        if not _is_gfx950(device):
            raise ValueError("Shared-square backward is gfx950-only")
        prepared = _allocate_mxfp8_backward_preparation(device, shape)
        _launch_mxfp8_backward_preparation(k_scale, do_bf16, out_bf16, prepared, shape)
        return prepared


def launch_backward_shared_square_mxfp8(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, do_bf16, out_bf16, lse,
                                        sm_scale, *, causal=False):
    """Checked full backward from MXFP8 Q/K/V and BF16 dO/O; return BF16 dQ/dK/dV.

    Q/K use the existing square32 quantization and repeated [B,H,N,4] scales.
    V MUST already use feature32 quantization: each [key,32 features] block
    has its own E8M0 scale, also [B,H,N,4]. Forward V's sequence32 scales
    [B,H,N/32,128] are incompatible. No V transpose or requantization occurs.
    Equal scale tensor shapes do not establish equal quantization semantics:
    values/provenance remain the caller's responsibility.

    O and base-2 LSE must come from the matching forward and sm_scale. Full
    backward includes FP32 Delta from original BF16 O*dO, square32 dO quantization, K-scale
    preparation, KV and Q kernels. Q/K/V preparation is excluded by this API.
    Supplying V from the original BF16 feature32 quantizer reproduces the
    legacy backward recipe; requantizing sequence-scaled FP8 V need not do so.

    All metadata and architecture checks precede every allocation/launch.
    All nine temporary/output allocations precede the first submission.
    Calls use the input device's current stream, with ordinary allocator or
    graph-private-pool lifetime; callers own readiness and graph retention.
    No normalization, hidden workspace, fallback or allocation retry is used.
    """
    device, shape = _check_mxfp8_backward_shape(q_fp8, sm_scale, causal)
    specs = _mxfp8_backward_specs(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, lse, shape)
    _check_mxfp8_backward_tensors(
        device, specs + (
            ("do_bf16", do_bf16, shape, torch.bfloat16),
            ("out_bf16", out_bf16, shape, torch.bfloat16),
        ))
    with torch.cuda.device(device):
        if not _is_gfx950(device):
            raise ValueError("Shared-square backward is gfx950-only")
        prepared = _allocate_mxfp8_backward_preparation(device, shape)
        workspace = _allocate_mxfp8_backward_workspace(device, shape)
        _launch_mxfp8_backward_preparation(k_scale, do_bf16, out_bf16, prepared, shape)
        return _launch_mxfp8_backward_core(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, lse, sm_scale, causal,
                                           prepared, workspace, shape)


def launch_backward_shared_square_mxfp8_core(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, do_fp8, do_scale, delta,
                                             k_dq_scale, lse, sm_scale, *, causal=False):
    """Checked prequantized core; return BF16 dQ/dK/dV using only KV/Q kernels.

    Q/K square32 and V feature32 contracts match the full MXFP8-input API.
    dO8/dO_scale/Delta/K_dQ_scale must be the preparation API's matching
    outputs, or be value-equivalent. In particular Delta must come from the
    ORIGINAL BF16 O*dO, not dequantized dO8, and K_dQ_scale must correspond
    to k_scale. Q/K/V and LSE must match the forward; prepared dO and Delta
    must match its backward invocation.
    These value/provenance contracts are not GPU-checked by metadata admission.

    Preparation/Delta costs are EXCLUDED: this API must not be reported as
    full backward. It allocates fresh DS/DSS and three gradient outputs per
    call; it does not cache workspace or modify inputs/prepared tensors.
    Input-device/current-stream, graph lifetime and readiness responsibilities
    are the same as the full API. All checks precede allocation and launch.
    """
    device, shape = _check_mxfp8_backward_shape(q_fp8, sm_scale, causal)
    specs = _mxfp8_backward_specs(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, lse, shape)
    _check_mxfp8_backward_tensors(
        device, specs + (
            ("do_fp8", do_fp8, shape, torch.float8_e4m3fn),
            ("do_scale", do_scale, shape[:-1] + (4, ), torch.uint8),
            ("delta", delta, shape[:-1], torch.float32),
            ("k_dq_scale", k_dq_scale, (4, 32, 128, shape[2] // 32), torch.uint8),
        ))
    with torch.cuda.device(device):
        if not _is_gfx950(device):
            raise ValueError("Shared-square backward is gfx950-only")
        workspace = _allocate_mxfp8_backward_workspace(device, shape)
        prepared = do_fp8, do_scale, delta, k_dq_scale
        return _launch_mxfp8_backward_core(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, lse, sm_scale, causal,
                                           prepared, workspace, shape)
