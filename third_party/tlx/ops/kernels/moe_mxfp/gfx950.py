"""MXFP4 W4A4 routed MoE on gfx950.

Activations are quantized to MXFP4 (E2M1 + one E8M0 scale per 32 along K) before each GEMM, and both
GEMMs run on the CDNA4 scaled MFMA through ``tl.dot_scaled``. One call runs: local-expert block
alignment, per-token activation quantization, the stage-1 gate/up GEMM with SiLU and the MXFP4
requantization of its output fused into the epilogue, the stage-2 down GEMM, and the shared finalize
(optional per-expert post-norm and routing-weighted sum). Launch shapes depend only on the token count,
so the op is HIP-graph capturable.

Checkpoints use the natural (Quark) layout: ``w13`` uint8 ``[E, 2I, K/2]`` with ``w13_scale`` uint8
``[E, 2I, K/32]`` (gate rows first), and ``w2`` uint8 ``[E, H, I/2]`` with ``w2_scale`` uint8
``[E, H, I/32]``. Packed bytes hold the even element in the low nibble. ``prepare_w13`` / ``prepare_w2``
reorder them once at load time into the kernels' layout: weights in 1 KiB blocks in MFMA lane order (the
same byte order as AITER's ``shuffle_weight``) and scales packed so one dword feeds two K-steps.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from triton.tlx.ops.kernels.moe.gfx950 import _align, _finalize_kernel

_SCALE_GROUP = 32


@triton.jit
def _mxfp4_quant(x, BM: tl.constexpr, BN: tl.constexpr):
    """fp32 [BM, BN] -> (packed e2m1 uint8 [BM, BN//2], e8m0 uint8 [BM, BN//32]); same rounding as AITER."""
    NG: tl.constexpr = BN // 32
    x = tl.reshape(x, (BM, NG, 32))
    amax = tl.max(tl.abs(x), axis=2)
    # floor(log2(round_up(amax))) - 2 straight from the exponent bits; integer-only so it lowers in any layout.
    bits = (amax.to(tl.int32, bitcast=True) + 0x200000).to(tl.uint32, bitcast=True)
    biased = ((bits >> 23) & 0xFF).to(tl.int32)
    e = tl.where(biased == 255, 127, tl.maximum(biased - 129, -127))
    scale = (e + 127).to(tl.uint8)
    # gfx950 scaled pack: x / 2^e rounded to E2M1 (RNE, saturating), two values per byte.
    sf = ((e + 127) << 23).to(tl.float32, bitcast=True)
    lo, hi = tl.split(tl.reshape(x, (BM, NG, 16, 2)))
    sf = tl.broadcast_to(sf[:, :, None], (BM, NG, 16))
    packed = tl.inline_asm_elementwise("v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3", "=v,v,v,v", [lo, hi, sf],
                                       dtype=tl.int32, is_pure=True, pack=1)
    # AITER zeroes a group whose rounded-up amax overflows (e == 127); the hardware would saturate it instead.
    packed = tl.where(tl.broadcast_to((e == 127)[:, :, None], (BM, NG, 16)), 0, packed)
    return tl.reshape(packed.to(tl.uint8), (BM, BN // 2)), tl.reshape(scale, (BM, NG))


@triton.jit
def _silu(x):
    # Raw v_exp_f32 / v_rcp_f32 (1 ulp each) instead of tl.exp's denormal fixups and tl.sigmoid's IEEE division; the
    # result is rounded to bf16 next, which this changes for ~2e-6 of activations, by 1 ulp.
    t = tl.inline_asm_elementwise("v_exp_f32 $0, $1", "=v,v", [x * -1.4426950408889634], dtype=tl.float32, is_pure=True,
                                  pack=1)
    r = tl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [1.0 + t], dtype=tl.float32, is_pure=True, pack=1)
    return x * r


@triton.jit
def _shuf_off(n, kb, KP: tl.constexpr):
    """Byte offset of weight (row n, packed column kb) in ``prepare_w13``/``prepare_w2``'s layout: 1 KiB blocks of
    16 rows x 64 bytes, each in MFMA lane order (lane = kb // 16 % 4 * 16 + n % 16, 16 bytes per lane)."""
    return (n // 16) * (KP * 16) + (kb // 64) * 1024 + ((kb % 64) // 16) * 256 + (n % 16) * 16 + kb % 16


@triton.jit
def _scale_off(r, g, R):
    """Byte offset of scale group g of row r in the packed layout [K/512, R, 16]: per row, 16 bytes cover two
    K-steps (8 groups each) ordered [g % 4][g // 4 % 2][step], so one dword per lane feeds both steps' MFMAs."""
    return (g // 16) * (R * 16) + r * 16 + (g % 4) * 4 + ((g // 4) % 2) * 2 + (g // 8) % 2


@triton.jit
def _scale_pair_off(r):
    """Offsets of one packed K-step pair for rows r: [R, 4, 2, 2] = [row, g % 4, g // 4, step]."""
    return r[:, None, None, None] * 16 + tl.arange(0, 4)[None, :, None, None] * 4 + \
        tl.arange(0, 2)[None, None, :, None] * 2 + tl.arange(0, 2)[None, None, None, :]


@triton.jit
def _unpack_pair(x, R: tl.constexpr):
    """One packed K-step pair [R, 4, 2, 2] -> (step 0, step 1) scales [R, 8] in group order."""
    s0, s1 = tl.split(x)
    return tl.reshape(tl.permute(s0, (0, 2, 1)), (R, 8)), tl.reshape(tl.permute(s1, (0, 2, 1)), (R, 8))


@triton.jit
def _b_rows(pid_n, N, DUAL_B: tl.constexpr, NATURAL_B: tl.constexpr, STACKED_B: tl.constexpr = False):
    """First weight rows of a 256-row tile's left and right halves. Prepared stage-1 weights interleave gate and up
    in 128-row blocks, so the halves are rows [256 * pid_n, +256); natural stage-1 weights, and shuffled ones with
    STACKED_B, keep [gate; up], so the left half is gate block pid_n and the right half the same up block, N (= I)
    rows further."""
    if (NATURAL_B or STACKED_B) and DUAL_B:
        return pid_n * 128, N + pid_n * 128
    else:
        return pid_n * 256, pid_n * 256 + 128


@triton.jit
def _quant_rows_kernel(x_ptr, q_ptr, s_ptr, M, stride_xm, stride_xk, K: tl.constexpr, BM: tl.constexpr,
                       BK: tl.constexpr, PACKED: tl.constexpr, UNIT_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_k * BK + tl.arange(0, BK)
    mask = rows < M
    # A runtime column stride would hide the contiguity the vectorized load relies on.
    col_off = cols[None, :] if UNIT_K else cols[None, :].to(tl.int64) * stride_xk
    x = tl.load(x_ptr + rows[:, None].to(tl.int64) * stride_xm + col_off, mask=mask[:, None], other=0.0)
    q, s = _mxfp4_quant(x.to(tl.float32), BM, BK)
    qc = pid_k * (BK // 2) + tl.arange(0, BK // 2)
    sc = pid_k * (BK // 32) + tl.arange(0, BK // 32)
    tl.store(q_ptr + rows[:, None].to(tl.int64) * (K // 2) + qc[None, :], q, mask=mask[:, None])
    if PACKED:
        tl.store(s_ptr + _scale_off(rows[:, None].to(tl.int64), sc[None, :], M), s, mask=mask[:, None])
    else:
        tl.store(s_ptr + rows[:, None].to(tl.int64) * (K // 32) + sc[None, :], s, mask=mask[:, None])


def quantize_rows(x: torch.Tensor, packed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [M, K] -> (packed uint8 [M, K/2], E8M0 uint8 [M, K/32]); ``packed`` returns the scales as
    [K/512, M, 16] (``_scale_off``)."""
    M, K = x.shape
    assert K % 512 == 0 or not packed
    assert K % 256 == 0
    q = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    s = torch.empty((K // 512, M, 16) if packed else (M, K // 32), dtype=torch.uint8, device=x.device)
    if M:
        bk = 512 if K % 512 == 0 else 256
        _quant_rows_kernel[(triton.cdiv(M, 16), K // bk)](x, q, s, M, x.stride(0), x.stride(1), K=K, BM=16, BK=bk,
                                                          PACKED=packed, UNIT_K=x.stride(1) == 1, num_warps=4)
    return q, s


@triton.jit
def _moe_mxfp4_small_kernel(
    a_ptr,  # uint8 [a_rows, K/2]: token rows (A_ROW_DIV_TOPK) or slot rows
    as_ptr,  # uint8 packed scales [K/512, a_rows, 16]
    b_ptr,  # uint8 [E, NB, K/2] pre-shuffled (``prepare_w13`` / ``prepare_w2``)
    bs_ptr,  # uint8 packed scales [E, K/512, NB, 16]
    c_ptr,  # DUAL_B: uint8 [slots, N/2]; else bf16 [slots, N]
    cs_ptr,  # DUAL_B: uint8 packed scales [N/512, slots, 16]
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_ptr,
    num_valid,
    a_rows,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_be,
    stride_bse,
    NB: tl.constexpr,
    TOP_K: tl.constexpr,
    A_ROW_DIV_TOPK: tl.constexpr,
    DUAL_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    WN: tl.constexpr,
    A_LDS: tl.constexpr,
    NATURAL_B: tl.constexpr = False,
    STACKED_B: tl.constexpr = False,
    SORTED: tl.constexpr = False,
    cap=0,
    R4: tl.constexpr = False,
    B_CACHE: tl.constexpr = ".cg",
    UNROLL: tl.constexpr = None,
):
    """Small/mid-T engine: BLOCK_M rows x two 128-row weight tiles (gate and up, or 256 output columns) per program,
    WN waves across the columns; two weight tiles keep as many bytes in flight per wave in both GEMMs.
    B and both scales go global -> MFMA operand registers one K-step ahead (scales one packed pair ahead). A goes
    straight to registers too, or (A_LDS) once per tile into double-buffered LDS by async DMA, which saves the
    per-wave copies once BLOCK_M=64.

    SORTED (the prefill engine, 64 or 96 rows x 4 waves): rows that the token-order layout gathers are kept in tile order
    instead, so a tile reads and writes contiguous rows. Stage 1 reads its A scales in tile order (``_sort_rows``) and
    writes its MXFP4 output and scales at each slot's tile position ([cap, ...]); stage 2 reads those rows directly
    and stores its BF16 rows 16 bytes per lane. With B_CACHE "" the weights stay in L1 for the expert's other m-blocks
    on the same CU, and UNROLL unrolls the K loop. A 96-row block keeps its activations as three 32-row LDS tiles
    (SPLIT_A, R4, WN) and is compiled with waves_per_eu=2, so the 3 x 2 accumulator tiles still leave two waves per SIMD."""
    SM: tl.constexpr = 32
    NS: tl.constexpr = BLOCK_M // SM
    BLOCK_N: tl.constexpr = 128
    KP: tl.constexpr = K // 2
    BKP: tl.constexpr = 128
    n_steps: tl.constexpr = K // 256
    # Natural weights: a K-step is 128 bytes along each row; prepared: two 1 KiB blocks per 16-row group.
    B_KSTEP: tl.constexpr = BKP if NATURAL_B else BKP * 16
    BS_PAIR: tl.constexpr = NB * 16
    tl.static_assert(BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == 96)
    SPLIT_A: tl.constexpr = BLOCK_M == 96
    tl.static_assert(WN == 4 or WN == 8)
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 128], transposed=True,
                                            warps_per_cta=[1, WN])
    a_op: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=16)
    b_op: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=16)
    # Packed scale pairs [rows, g%4, g//4, step]: the MFMA wants lanes over (row % 16, g % 4), registers over
    # (step, g // 4), both steps' scale bytes in one dword; A broadcasts over warps, B splits columns over them. With
    # WN = 4 the loads use lanes over consecutive dwords (g % 4 fastest, then row), which touches 4x fewer cache
    # sectors per load, and _small_tile converts to the MFMA order (_ASP_MMA / _BSP_MMA) in registers.
    if WN == 4:
        bsp_layout: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
                                              stride=((4, 8, 16, 32, 64, 128, 256, 512), (1, 2, 1024)))
        asp_layout: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
                                              stride=((4, 8, 16, 32, 64, 128, 0, 0), (1, 2, 256)))
    else:
        bsp_layout: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
                                              stride=((16, 32, 64, 128, 4, 8, 256, 512, 1024), (1, 2)))
        asp_layout: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
                                              stride=((16, 32, 64, 128, 4, 8, 0, 0, 0), (1, 2, 256)))
    # A_LDS: lanes fill consecutive 16-byte LDS slots (``a_tile`` address bits 4..9); warps / registers take the rest.
    if BLOCK_M == 32 or BLOCK_M == 96:
        a_tile: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [[1024, 32]],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [16, 0], [2, 0], [4, 0], [8, 0]],
            [BLOCK_M, BKP])
        if WN == 4:
            a_ld: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
                                            stride=((16, 32, 64, 128, 2048, 256, 512, 1024), (1, 2, 4, 8)))
        else:
            a_ld: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
                                            stride=((16, 32, 64, 128, 2048, 256, 512, 1024, 0), (1, 2, 4, 8)))
    else:
        a_tile: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [[1024, 32]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [2, 0], [4, 0],
                           [8, 0], [16, 0]], [BLOCK_M, BKP])
        if WN == 4:
            a_ld: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
                                            stride=((16, 32, 64, 128, 4096, 256, 512, 1024), (1, 2, 4, 8, 2048)))
        else:
            a_ld: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
                                            stride=((16, 32, 64, 128, 4096, 256, 512, 1024, 2048), (1, 2, 4, 8)))

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = N // BLOCK_N if DUAL_B else N // (2 * BLOCK_N)
    num_pid_m = tl.cdiv(tl.load(num_post_ptr), BLOCK_M)
    if pid >= num_pid_m * num_pid_n:
        return
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert < 0:
        return
    m0 = pid_m * BLOCK_M

    a_off = tl.tuple([])
    asp_off = tl.tuple([])
    if SORTED:
        tl.static_assert((BLOCK_M == 64 or BLOCK_M == 96) and WN == 4 and A_LDS)
    for i in tl.static_range(NS):
        tok = tl.load(sorted_ids_ptr + m0 + i * SM + tl.arange(0, SM))
        if A_ROW_DIV_TOPK:
            row = tl.where(tok < num_valid, tok // TOP_K, 0)
        else:
            row = tl.where(tok < num_valid, tok, 0)
        # SORTED: A scales (both stages) and stage-2 A rows are in tile order.
        srow = m0 + i * SM + tl.arange(0, SM) if SORTED else row
        if not A_LDS:
            a_off += tl.tuple(
                [tlx.require_layout((row[:, None] * KP + tl.arange(0, BKP)[None, :]).to(tl.int32), a_op, pin=True)])
        asp_off += tl.tuple([tlx.require_layout(_scale_pair_off(srow).to(tl.int32), asp_layout, pin=True)])
    smem_a = None
    a_ld_off = None
    if SPLIT_A:
        nrows = 0
        a_ld_off = tl.tuple([])
        for i in tl.static_range(NS):
            tok_i = tl.load(sorted_ids_ptr + m0 + i * SM + tl.arange(0, SM))
            nrows += tl.sum((tok_i < num_valid).to(tl.int32), axis=0)
            if A_ROW_DIV_TOPK:
                row_i = tl.where(tok_i < num_valid, tok_i // TOP_K, 0)
            else:
                row_i = m0 + i * SM + tl.arange(0, SM)
            a_ld_off += tl.tuple(
                [tlx.require_layout((row_i[:, None] * KP + tl.arange(0, BKP)[None, :]).to(tl.int32), a_ld)])
        smem_a = tlx.local_alloc((SM, BKP), tl.uint8, (4 if R4 else 2) * NS, layout=a_tile)
    elif A_LDS:
        tok_all = tl.load(sorted_ids_ptr + m0 + tl.arange(0, BLOCK_M))
        if A_ROW_DIV_TOPK:
            row_all = tl.where(tok_all < num_valid, tok_all // TOP_K, 0)
        elif SORTED:
            row_all = m0 + tl.arange(0, BLOCK_M)
        else:
            row_all = tl.where(tok_all < num_valid, tok_all, 0)
        a_ld_off = tlx.require_layout((row_all[:, None] * KP + tl.arange(0, BKP)[None, :]).to(tl.int32), a_ld)
        smem_a = tlx.local_alloc((BLOCK_M, BKP), tl.uint8, 4 if R4 else 2, layout=a_tile)
    if NATURAL_B:
        b_off = tlx.require_layout((tl.arange(0, BLOCK_N)[None, :] * KP + tl.arange(0, BKP)[:, None]).to(tl.int32),
                                   b_op, pin=True)
    else:
        b_off = tlx.require_layout(
            _shuf_off(tl.arange(0, BLOCK_N)[None, :],
                      tl.arange(0, BKP)[:, None], KP).to(tl.int32), b_op, pin=True)
    bsp_off = tlx.require_layout(_scale_pair_off(tl.arange(0, BLOCK_N)).to(tl.int32), bsp_layout, pin=True)
    n0 = pid_n * 256
    row_l, row_r = _b_rows(pid_n, N, DUAL_B, NATURAL_B, STACKED_B)
    b_base = b_ptr + expert * stride_be + row_l * KP
    b2_delta = (row_r - row_l) * KP
    bs_base = bs_ptr + expert * stride_bse + n0 * 16
    as_pair = a_rows * 16

    if SORTED:
        # Padding sits at the end of an expert's last block: a block with at most SM valid rows (an expert's tail)
        # computes only its first sub-block. One uniform branch per program; each path is straight-line code.
        if not SPLIT_A:
            nrows = tl.sum((tok_all < num_valid).to(tl.int32), axis=0)
        if nrows <= SM:
            _small_tile(a_ptr, as_ptr, c_ptr, cs_ptr, sorted_ids_ptr, smem_a, a_ld_off, a_off, asp_off, b_base, b_off,
                        b2_delta, bs_base, bsp_off, as_pair, m0, num_valid, pid_n, n0, cap, N, 1, SM, BLOCK_N, BKP,
                        n_steps, B_KSTEP, BS_PAIR, UNROLL, B_CACHE, A_LDS, DUAL_B, SORTED, mma, a_op, SPLIT_A, R4, WN)
        else:
            if SPLIT_A:
                if nrows <= 2 * SM:
                    _small_tile(a_ptr, as_ptr, c_ptr, cs_ptr, sorted_ids_ptr, smem_a, a_ld_off, a_off, asp_off, b_base,
                                b_off, b2_delta, bs_base, bsp_off, as_pair, m0, num_valid, pid_n, n0, cap, N, 2, SM,
                                BLOCK_N, BKP, n_steps, B_KSTEP, BS_PAIR, UNROLL, B_CACHE, A_LDS, DUAL_B, SORTED, mma,
                                a_op, SPLIT_A, R4, WN)
                else:
                    _small_tile(a_ptr, as_ptr, c_ptr, cs_ptr, sorted_ids_ptr, smem_a, a_ld_off, a_off, asp_off, b_base,
                                b_off, b2_delta, bs_base, bsp_off, as_pair, m0, num_valid, pid_n, n0, cap, N, NS, SM,
                                BLOCK_N, BKP, n_steps, B_KSTEP, BS_PAIR, UNROLL, B_CACHE, A_LDS, DUAL_B, SORTED, mma,
                                a_op, SPLIT_A, R4, WN)
            else:
                _small_tile(a_ptr, as_ptr, c_ptr, cs_ptr, sorted_ids_ptr, smem_a, a_ld_off, a_off, asp_off, b_base,
                            b_off, b2_delta, bs_base, bsp_off, as_pair, m0, num_valid, pid_n, n0, cap, N, NS, SM,
                            BLOCK_N, BKP, n_steps, B_KSTEP, BS_PAIR, UNROLL, B_CACHE, A_LDS, DUAL_B, SORTED, mma, a_op,
                            SPLIT_A, R4, WN)
    else:
        _small_tile(a_ptr, as_ptr, c_ptr, cs_ptr, sorted_ids_ptr, smem_a, a_ld_off, a_off, asp_off, b_base, b_off,
                    b2_delta, bs_base, bsp_off, as_pair, m0, num_valid, pid_n, n0, cap, N, NS, SM, BLOCK_N, BKP,
                    n_steps, B_KSTEP, BS_PAIR, UNROLL, B_CACHE, A_LDS, DUAL_B, SORTED, mma, a_op, SPLIT_A, R4, WN)


_ASP_MMA = tl.constexpr(
    tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)), stride=((16, 32, 64, 128, 4, 8, 0, 0), (1, 2, 256))))
_BSP_MMA = tl.constexpr(
    tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)), stride=((16, 32, 64, 128, 4, 8, 256, 512), (1, 2, 1024))))


@triton.jit
def _mma_scales(x, mma_layout: tl.constexpr, R: tl.constexpr, CONTIG: tl.constexpr):
    """Unpacked (step 0, step 1) scales of a packed pair; CONTIG loads are first permuted to the MFMA lane order."""
    if CONTIG:
        x = tlx.require_layout(x, mma_layout, pin=True)
    return _unpack_pair(x, R)


@triton.jit
def _small_tile(a_ptr, as_ptr, c_ptr, cs_ptr, sorted_ids_ptr, smem_a, a_ld_off, a_off, asp_off, b_base, b_off, b2_delta,
                bs_base, bsp_off, as_pair, m0, num_valid, pid_n, n0, cap, N: tl.constexpr, NSR: tl.constexpr,
                SM: tl.constexpr, BLOCK_N: tl.constexpr, BKP: tl.constexpr, n_steps: tl.constexpr,
                B_KSTEP: tl.constexpr, BS_PAIR: tl.constexpr, UNROLL: tl.constexpr, B_CACHE: tl.constexpr,
                A_LDS: tl.constexpr, DUAL_B: tl.constexpr, SORTED: tl.constexpr, mma: tl.constexpr, a_op: tl.constexpr,
                SPLIT_A: tl.constexpr, R4: tl.constexpr, WN: tl.constexpr):
    """K loop and epilogue of ``_moe_mxfp4_small_kernel`` for the block's first NSR SM-row sub-blocks."""
    CONTIG: tl.constexpr = WN == 4
    accs = tl.tuple([tlx.zeros((SM, BLOCK_N), tl.float32, layout=mma) for _ in range(NSR)])
    accs2 = tl.tuple([tlx.zeros((SM, BLOCK_N), tl.float32, layout=mma) for _ in range(NSR)])
    RB: tl.constexpr = 4 if R4 else 2
    if SPLIT_A:
        for i in tl.static_range(NSR):
            tlx.buffer_load_to_local(smem_a[RB * i], a_ptr, a_ld_off[i])
            if R4:
                tlx.buffer_load_to_local(smem_a[RB * i + 1], a_ptr + BKP, a_ld_off[i])
        tlx.async_load_commit_group()
    elif A_LDS:
        tlx.buffer_load_to_local(smem_a[0], a_ptr, a_ld_off)
        if R4:
            tlx.buffer_load_to_local(smem_a[1], a_ptr + BKP, a_ld_off)
        tlx.async_load_commit_group()
    else:
        a = tl.tuple([tlx.buffer_load(a_ptr, a_off[i], contiguity=16) for i in range(NSR)])
    b = tlx.buffer_load(b_base, b_off, cache=B_CACHE, contiguity=16)
    a_sp = tl.tuple([tlx.buffer_load(as_ptr, asp_off[i], contiguity=4) for i in range(NSR)])
    b_sp = tlx.buffer_load(bs_base, bsp_off, cache=B_CACHE, contiguity=4)
    b2 = tlx.buffer_load(b_base + b2_delta, b_off, cache=B_CACHE, contiguity=16)
    b2_sp = tlx.buffer_load(bs_base + 128 * 16, bsp_off, cache=B_CACHE, contiguity=4)
    if R4:
        # Four activation buffers: one wait + barrier per pair of K-steps publishes A[k0], A[k0 + 1] and retires the
        # reads of the buffers that then take A[k0 + 2], A[k0 + 3].
        tl.static_assert(A_LDS)
        for k0 in tl.static_range(0, n_steps, 2):
            kp = min(k0 // 2 + 1, n_steps // 2 - 1)
            na_sp = tl.tuple([tlx.buffer_load(as_ptr + kp * as_pair, asp_off[i], contiguity=4) for i in range(NSR)])
            nb_sp = tlx.buffer_load(bs_base + kp * BS_PAIR, bsp_off, cache=B_CACHE, contiguity=4)
            a_s_pair = tl.tuple([_mma_scales(a_sp[i], _ASP_MMA, SM, CONTIG) for i in range(NSR)])
            b_s_pair = _mma_scales(b_sp, _BSP_MMA, BLOCK_N, CONTIG)
            nb2_sp = tlx.buffer_load(bs_base + kp * BS_PAIR + 128 * 16, bsp_off, cache=B_CACHE, contiguity=4)
            b2_s_pair = _mma_scales(b2_sp, _BSP_MMA, BLOCK_N, CONTIG)
            tlx.async_load_wait_group(0)
            tlx.workgroup_barrier()
            if k0 + 2 < n_steps:
                for j in tl.static_range(2):
                    if SPLIT_A:
                        for i in tl.static_range(NSR):
                            tlx.buffer_load_to_local(smem_a[RB * i + (k0 + 2 + j) % 4], a_ptr + (k0 + 2 + j) * BKP,
                                                     a_ld_off[i])
                    else:
                        tlx.buffer_load_to_local(smem_a[(k0 + 2 + j) % 4], a_ptr + (k0 + 2 + j) * BKP, a_ld_off)
                tlx.async_load_commit_group()
            for s in tl.static_range(2):
                kn = min(k0 + s + 1, n_steps - 1)
                if SPLIT_A:
                    acur = tl.tuple(
                        [tlx.local_load(smem_a[RB * i + (k0 + s) % 4], layout=a_op, relaxed=False) for i in range(NSR)])
                else:
                    acur = tl.tuple([
                        tlx.local_load(tlx.local_slice(smem_a[(k0 + s) % 4], [i * SM, 0], [SM, BKP]), layout=a_op,
                                       relaxed=False) for i in range(NSR)
                    ])
                nb = tlx.buffer_load(b_base + kn * B_KSTEP, b_off, cache=B_CACHE, contiguity=16)
                nb2 = tlx.buffer_load(b_base + b2_delta + kn * B_KSTEP, b_off, cache=B_CACHE, contiguity=16)
                new_accs = tl.tuple([])
                new_accs2 = tl.tuple([])
                for i in tl.static_range(NSR):
                    new_accs += tl.tuple(
                        [tl.dot_scaled(acur[i], a_s_pair[i][s], "e2m1", b, b_s_pair[s], "e2m1", accs[i])])
                    new_accs2 += tl.tuple(
                        [tl.dot_scaled(acur[i], a_s_pair[i][s], "e2m1", b2, b2_s_pair[s], "e2m1", accs2[i])])
                accs, accs2 = new_accs, new_accs2
                b, b2 = nb, nb2
            a_sp, b_sp, b2_sp = na_sp, nb_sp, nb2_sp
    for k0 in tl.range(0, 0 if R4 else n_steps, 2, num_stages=1, loop_unroll_factor=UNROLL):
        kp_n = tl.minimum(k0 // 2 + 1, n_steps // 2 - 1)
        na_sp = tl.tuple([tlx.buffer_load(as_ptr + kp_n * as_pair, asp_off[i], contiguity=4) for i in range(NSR)])
        nb_sp = tlx.buffer_load(bs_base + kp_n * BS_PAIR, bsp_off, cache=B_CACHE, contiguity=4)
        a_s_pair = tl.tuple([_mma_scales(a_sp[i], _ASP_MMA, SM, CONTIG) for i in range(NSR)])
        b_s_pair = _mma_scales(b_sp, _BSP_MMA, BLOCK_N, CONTIG)
        nb2_sp = tlx.buffer_load(bs_base + kp_n * BS_PAIR + 128 * 16, bsp_off, cache=B_CACHE, contiguity=4)
        b2_s_pair = _mma_scales(b2_sp, _BSP_MMA, BLOCK_N, CONTIG)
        for s in tl.static_range(2):
            kn = tl.minimum(k0 + s + 1, n_steps - 1)
            if SPLIT_A:
                tlx.async_load_wait_group(0)
                tlx.workgroup_barrier()
                if k0 + s + 1 < n_steps:
                    for i in tl.static_range(NSR):
                        tlx.buffer_load_to_local(smem_a[2 * i + 1 - s], a_ptr + kn * BKP, a_ld_off[i])
                    tlx.async_load_commit_group()
                acur = tl.tuple([tlx.local_load(smem_a[2 * i + s], layout=a_op, relaxed=False) for i in range(NSR)])
            elif A_LDS:
                # A[k] landed for this thread; the barrier publishes it and retires all reads of the other buffer.
                tlx.async_load_wait_group(0)
                tlx.workgroup_barrier()
                # The last step has no next activation tile; a copy there would only add a full wait before the
                # epilogue.
                if k0 + s + 1 < n_steps:
                    tlx.buffer_load_to_local(smem_a[1 - s], a_ptr + kn * BKP, a_ld_off)
                    tlx.async_load_commit_group()
                acur = tl.tuple([
                    tlx.local_load(tlx.local_slice(smem_a[s], [i * SM, 0], [SM, BKP]), layout=a_op, relaxed=False)
                    for i in range(NSR)
                ])
            else:
                acur = a
                a = tl.tuple([tlx.buffer_load(a_ptr + kn * BKP, a_off[i], contiguity=16) for i in range(NSR)])
            nb = tlx.buffer_load(b_base + kn * B_KSTEP, b_off, cache=B_CACHE, contiguity=16)
            nb2 = tlx.buffer_load(b_base + b2_delta + kn * B_KSTEP, b_off, cache=B_CACHE, contiguity=16)
            new_accs = tl.tuple([])
            new_accs2 = tl.tuple([])
            for i in tl.static_range(NSR):
                new_accs += tl.tuple([tl.dot_scaled(acur[i], a_s_pair[i][s], "e2m1", b, b_s_pair[s], "e2m1", accs[i])])
                new_accs2 += tl.tuple(
                    [tl.dot_scaled(acur[i], a_s_pair[i][s], "e2m1", b2, b2_s_pair[s], "e2m1", accs2[i])])
            accs, accs2 = new_accs, new_accs2
            b, b2 = nb, nb2
        a_sp, b_sp, b2_sp = na_sp, nb_sp, nb2_sp
    if A_LDS:
        tlx.async_load_wait_group(0)

    # SORTED stage-2 stores: lanes span a 128-column row segment, 16 bytes each (the layout change goes through LDS).
    row_st: tl.constexpr = tlx.layout(shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
                                      stride=((8, 16, 32, 64, 128, 256, 512, 1024), (1, 2, 4, 2048)))
    for i in tl.static_range(NSR):
        tok = tl.load(sorted_ids_ptr + m0 + i * SM + tl.arange(0, SM))
        valid = (tok < num_valid)[:, None]
        slot = tok.to(tl.int64)
        if DUAL_B:
            # Same rounding as an unfused path: the activation is BF16 before requantization.
            h = (_silu(accs[i]) * accs2[i]).to(tl.bfloat16).to(tl.float32)
            q, qs = _mxfp4_quant(h, SM, BLOCK_N)
            qc = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
            sg = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
            if SORTED:
                pos = (m0 + i * SM + tl.arange(0, SM)).to(tl.int64)
                tl.store(c_ptr + pos[:, None] * (N // 2) + qc[None, :], q, mask=valid)
                tl.store(cs_ptr + _scale_off(pos[:, None], sg[None, :], cap), qs, mask=valid)
            else:
                tl.store(c_ptr + slot[:, None] * (N // 2) + qc[None, :], q, mask=valid)
                tl.store(cs_ptr + _scale_off(slot[:, None], sg[None, :], num_valid), qs, mask=valid)
        elif SORTED:
            offs = tlx.require_layout(slot[:, None] * N + (n0 + tl.arange(0, BLOCK_N))[None, :], row_st)
            vmask = tlx.require_layout(tl.broadcast_to(valid, (SM, BLOCK_N)), row_st)
            # Streaming output: non-temporal stores keep it from evicting weights and activations in L2.
            tl.store(c_ptr + offs, tlx.require_layout(accs[i].to(tl.bfloat16), row_st), mask=vmask,
                     eviction_policy="evict_first")
            tl.store(c_ptr + offs + BLOCK_N, tlx.require_layout(accs2[i].to(tl.bfloat16), row_st), mask=vmask,
                     eviction_policy="evict_first")
        else:
            offs_n = n0 + tl.arange(0, BLOCK_N)
            tl.store(c_ptr + slot[:, None] * N + offs_n[None, :], accs[i].to(tl.bfloat16), mask=valid)
            tl.store(c_ptr + slot[:, None] * N + BLOCK_N + offs_n[None, :], accs2[i].to(tl.bfloat16), mask=valid)


@triton.jit
def _sort_scale_rows_kernel(src, dst, sorted_ids_ptr, num_post_ptr, num_valid, src_rows, cap, TOP_K: tl.constexpr,
                            NCH: tl.constexpr, BR: tl.constexpr):
    """Packed scales [NCH, src_rows, 16] in token order -> [NCH, cap, 16] in tile order: row r takes token
    sorted_ids[r] // TOP_K; padding rows get zeros."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = r < tl.load(num_post_ptr)
    tok = tl.load(sorted_ids_ptr + r, mask=live, other=num_valid)
    ok = live & (tok < num_valid)
    srow = tl.where(ok, tok // TOP_K, 0).to(tl.int64)
    cols = tl.arange(0, 16)
    for c in tl.static_range(NCH):
        v = tl.load(src + c * src_rows * 16 + srow[:, None] * 16 + cols[None, :], mask=ok[:, None], other=0)
        tl.store(dst + c * cap * 16 + r[:, None].to(tl.int64) * 16 + cols[None, :], v, mask=live[:, None])


@triton.jit
def _finalize_local_kernel(y_ptr, w_ptr, scores_ptr, ids_ptr, out_ptr, H, local_start, num_local, eps, gain,
                           TOP_K: tl.constexpr, HAS_NORM: tl.constexpr, BLOCK_H: tl.constexpr):
    """``_finalize_kernel`` that visits only this rank's slots of each token (a uniform branch per slot) instead of
    normalizing all TOP_K rows and zeroing the others' weights: the same rounding points, slots added in order."""
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    if HAS_NORM:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        scale = (w.to(tl.float32) + gain).to(w_ptr.dtype.element_ty).to(tl.float32)
    acc = tl.zeros((BLOCK_H, ), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        e = tl.load(ids_ptr + m * TOP_K + k)
        if (e >= local_start) & (e < local_start + num_local):
            sc = tl.load(scores_ptr + m * TOP_K + k).to(tl.float32)
            y = tl.load(y_ptr + (m * TOP_K + k).to(tl.int64) * H + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_NORM:
                y = (y * scale).to(tl.bfloat16).to(tl.float32)
                wgt = sc / tl.sqrt(tl.sum(y * y, axis=0) / H + eps)
            else:
                wgt = sc
            acc = acc + y * wgt
    tl.store(out_ptr + m.to(tl.int64) * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def pick_config(num_valid: int, num_experts: int = 64) -> dict:
    """Keyed on the expected rows per expert, T * topk / num_experts (no host sync, graph-capturable), so one rule
    holds for any expert split. Cutoffs measured with 32 and 64 local experts at H = I = 4096 under HIP graph replay
    on MI350X; another hidden size or GPU may move them: pass one of these dicts as ``moe_mxfp4``'s ``cfg`` to pin an
    engine. Above 64 rows per expert (an 8K-token SP prefill at EP4 is ~256) the SORTED engine runs: 64-row blocks up
    to 160 rows per expert, 96-row blocks above (fewer programs per expert, each weight tile feeding 96 rows; measured
    up to 1024 rows per expert, where it also beats the 256-row quad tile)."""
    rows = num_valid / num_experts
    if rows <= 16:
        return dict(BLOCK_M=32, WN=8, A_LDS=False)
    if rows <= 64:
        return dict(BLOCK_M=128, GROUP_M=4, QUAD=True)
    if rows <= 160:
        return dict(BLOCK_M=64, WN=4, A_LDS=True, SORTED=True)
    return dict(BLOCK_M=96, WN=4, A_LDS=True, SORTED=True, WPE=2)


@triton.jit
def _sorted_gather_kernel(q_ptr, s_ptr, gq_ptr, gs_ptr, sorted_ids_ptr, num_post_ptr, num_valid, cap, K: tl.constexpr,
                          TOP_K: tl.constexpr, BM: tl.constexpr):
    """Copy token-order MXFP4 rows (q [T, K/2], s [T, K/32]) into sorted order: gq [cap, K/2] and K-group-major
    gs [K/32, cap]. Padding rows get zero data and the smallest scale (idle MFMA lanes draw less power)."""
    pid_m = tl.program_id(0)
    if pid_m * BM >= tl.load(num_post_ptr):
        return
    KP: tl.constexpr = K // 2
    KS: tl.constexpr = K // 32
    pos = pid_m * BM + tl.arange(0, BM)
    tok = tl.load(sorted_ids_ptr + pos)
    valid = (tok < num_valid)[:, None]
    rows = tl.where(tok < num_valid, tok // TOP_K, 0).to(tl.int64)
    cq = tl.arange(0, KP)
    q = tl.load(q_ptr + rows[:, None] * KP + cq[None, :], mask=valid, other=0)
    tl.store(gq_ptr + pos[:, None].to(tl.int64) * KP + cq[None, :], q)
    cs = tl.arange(0, KS)
    sc = tl.load(s_ptr + rows[:, None] * KS + cs[None, :], mask=valid, other=0)
    tl.store(gs_ptr + cs[None, :].to(tl.int64) * cap + pos[:, None], sc)


@triton.jit
def _align_count_kernel(ids_ptr, counts_ptr, n, local_start, E: tl.constexpr, E_BLOCK: tl.constexpr,
                        CHUNK: tl.constexpr):
    """counts[c, e] = number of slots in chunk c routed to local expert e."""
    c = tl.program_id(0)
    slot = c * CHUNK + tl.arange(0, CHUNK)
    local = tl.load(ids_ptr + slot, mask=slot < n, other=-1) - local_start
    hist = tl.histogram(tl.where((slot < n) & (local >= 0) & (local < E), local, E).to(tl.int32), E_BLOCK)
    experts = tl.arange(0, E_BLOCK)
    tl.store(counts_ptr + c * E + experts, hist, mask=experts < E)


@triton.jit
def _align_scatter_kernel(ids_ptr, counts_ptr, sorted_ptr, expert_ptr, num_post_ptr, n, local_start, num_chunks,
                          E: tl.constexpr, E_BLOCK: tl.constexpr, C_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr,
                          CHUNK: tl.constexpr, SUB: tl.constexpr, PADB: tl.constexpr = 0):
    """Stable sort of slots by local expert into BLOCK_M-padded blocks. Program c places chunk c's slots (rank =
    earlier chunks' counts + in-chunk rank) and, for experts e = c, c + num_chunks, ..., writes padding and expert
    ids. Padding slots hold n; num_post = padded rows."""
    c = tl.program_id(0)
    experts = tl.arange(0, E_BLOCK)
    real = experts < E
    chunks = tl.arange(0, C_BLOCK)
    cnt2d = tl.load(counts_ptr + chunks[:, None] * E + experts[None, :],
                    mask=(chunks < num_chunks)[:, None] & real[None, :], other=0)
    counts = tl.sum(cnt2d, axis=0)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    start = tl.cumsum(padded, axis=0) - padded

    running = tl.sum(tl.where((chunks < c)[:, None], cnt2d, 0), axis=0)
    for sub in tl.static_range(CHUNK // SUB):
        slot = c * CHUNK + sub * SUB + tl.arange(0, SUB)
        local = tl.load(ids_ptr + slot, mask=slot < n, other=-1) - local_start
        onehot = ((local[:, None] == experts[None, :]) & real[None, :] & (slot < n)[:, None]).to(tl.int32)
        dst = tl.sum((tl.cumsum(onehot, axis=0) - 1 + running[None, :] + start[None, :]) * onehot, axis=1)
        tl.store(sorted_ptr + dst, slot.to(tl.int32), mask=tl.sum(onehot, axis=1) > 0)
        running += tl.sum(onehot, axis=0)

    pad = tl.arange(0, PADB)
    for e in range(c, E, num_chunks):
        sel = experts == e
        cnt = tl.sum(tl.where(sel, counts, 0), axis=0)
        rows = tl.sum(tl.where(sel, padded, 0), axis=0)
        s0 = tl.sum(tl.where(sel, start, 0), axis=0)
        tl.store(sorted_ptr + s0 + cnt + pad, tl.full((PADB, ), 0, tl.int32) + n, mask=cnt + pad < rows)
        for b in range(0, rows // BLOCK_M):
            tl.store(expert_ptr + s0 // BLOCK_M + b, e)
    if c == 0:
        tl.store(num_post_ptr, tl.sum(padded, axis=0))


def _align_quad(topk_ids, num_local, local_start, block_m):
    """``_align`` equivalent (same sorted order) for the quad engine, parallel over slot chunks."""
    n = topk_ids.numel()
    dev = topk_ids.device
    cap = triton.cdiv(n + num_local * (block_m - 1), block_m) * block_m
    sorted_ids = torch.empty(cap, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(cap // block_m, dtype=torch.int32, device=dev)
    num_post = torch.empty(1, dtype=torch.int32, device=dev)
    # Each program prefix-sums all chunk counts, so chunk size trades per-program work against that prefix.
    chunk = 128 if n <= 65536 else 256
    num_chunks = triton.cdiv(n, chunk)
    counts = torch.empty((num_chunks, num_local), dtype=torch.int32, device=dev)
    ids = topk_ids.reshape(-1)
    _align_count_kernel[(num_chunks, )](ids, counts, n, local_start, E=num_local,
                                        E_BLOCK=triton.next_power_of_2(num_local + 1), CHUNK=chunk, num_warps=4)
    _align_scatter_kernel[(num_chunks, )](ids, counts, sorted_ids, expert_ids, num_post, n, local_start, num_chunks,
                                          E=num_local, E_BLOCK=triton.next_power_of_2(num_local),
                                          C_BLOCK=triton.next_power_of_2(num_chunks), BLOCK_M=block_m, CHUNK=chunk,
                                          SUB=128, num_warps=4, PADB=triton.next_power_of_2(block_m))
    return sorted_ids, expert_ids, num_post


@triton.jit
def _quad_tile(tile, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (tile // num_pid_in_group) * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    return first_pid_m + ((tile % num_pid_in_group) % group_size_m), (tile % num_pid_in_group) // group_size_m


@triton.jit
def _quad_prologue(tile, num_pid_m, num_pid_n, a_ptr, as_ptr, b_ptr, bs_ptr, expert_ids_ptr, stride_be, stride_bse,
                   smem_a_top, smem_a_bot, smem_b_left, smem_b_right, smem_a_sc_t, smem_a_sc_b, smem_b_sc, tile_off,
                   a_tile_off, a_sc_offsets, b_sc_offsets, a_sc_k, b_sc_k, KP: tl.constexpr, BLOCK_M: tl.constexpr,
                   GROUP_M: tl.constexpr, N, DUAL_B: tl.constexpr, NATURAL_B: tl.constexpr, STACKED_B: tl.constexpr):
    """Issue the first two K-stages of ``tile``: eight commit groups, in the order the main loop waits on them. B
    scales come one packed K-step pair at a time; pair p goes to ``smem_b_sc[p % 2]``."""
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_K: tl.constexpr = 128
    pid_m, pid_n = _quad_tile(tile, num_pid_m, num_pid_n, GROUP_M)
    expert = tl.load(expert_ids_ptr + pid_m)
    B_KSTEP: tl.constexpr = HALF_K if NATURAL_B else HALF_K * 16
    row_l, row_r = _b_rows(pid_n, N, DUAL_B, NATURAL_B, STACKED_B)
    a_base = a_ptr + (pid_m * BLOCK_M).to(tl.int64) * KP
    b_base = b_ptr + expert.to(tl.int64) * stride_be + row_l.to(tl.int64) * KP
    b_right = (row_r - row_l).to(tl.int64) * KP
    a_scales = as_ptr + pid_m * BLOCK_M
    b_scales = bs_ptr + expert.to(tl.int64) * stride_bse + pid_n * 256 * 16
    for s in tl.static_range(2):
        tlx.buffer_load_to_local(smem_b_left[s], b_base + s * B_KSTEP, tile_off, contiguity=16)
        if s == 0:
            tlx.buffer_load_to_local(smem_b_sc[0], b_scales, b_sc_offsets, contiguity=4)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_top[s], a_base + s * HALF_K, a_tile_off)
        tlx.buffer_load_to_local(smem_a_sc_t[s], a_scales + s * a_sc_k, a_sc_offsets)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_bot[s], a_base + HALF_M * KP + s * HALF_K, a_tile_off)
        tlx.buffer_load_to_local(smem_a_sc_b[s], a_scales + HALF_M + s * a_sc_k, a_sc_offsets)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_b_right[s], b_base + b_right + s * B_KSTEP, tile_off, contiguity=16)
        tlx.async_load_commit_group()


@triton.jit
def _wsum_rows(acc_l, acc_r, slots_ptr, out_ptr, w_ptr, offs_l, offs_r, num_valid, stride_om, ROWS: tl.constexpr,
               TOP_K: tl.constexpr):
    slot = tl.load(slots_ptr + tl.arange(0, ROWS))
    valid = slot < num_valid
    w = tl.load(w_ptr + slot, mask=valid, other=0.0)
    o = out_ptr + (slot // TOP_K)[:, None].to(tl.int64) * stride_om
    # Same rounding as the unfused path up to the sum: the expert output is BF16 before weighting.
    vl = (acc_l.to(tl.bfloat16).to(tl.float32) * w[:, None]).to(tl.bfloat16)
    vr = (acc_r.to(tl.bfloat16).to(tl.float32) * w[:, None]).to(tl.bfloat16)
    tl.atomic_add(o + offs_l[None, :], vl, mask=valid[:, None], sem="relaxed")
    tl.atomic_add(o + offs_r[None, :], vr, mask=valid[:, None], sem="relaxed")


@triton.jit
def _b_scales_step(buf, STEP: tl.constexpr, pair_layout: tl.constexpr, scale_b_layout: tl.constexpr,
                   HALF_N: tl.constexpr):
    """K-step STEP of the packed B-scale pair in LDS ``buf`` [256, 16] -> (left, right) MFMA scale operands."""
    pair = tl.reshape(tlx.local_load(buf, layout=pair_layout), (2 * HALF_N, 4, 2, 2))
    s0, s1 = tl.split(pair)
    st = s0 if STEP == 0 else s1
    comb = tl.reshape(tl.permute(st, (0, 2, 1)), (2 * HALF_N, 8))
    left, right = tl.split(tl.trans(tl.reshape(comb, 2, HALF_N, 8), 1, 2, 0))
    return tlx.require_layout(left, scale_b_layout), tlx.require_layout(right, scale_b_layout)


@triton.jit
def _quad_tile_body(tile, num_pid_m, num_pid_n, num_tiles, num_progs, a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, cs_ptr,
                    sorted_ids_ptr, expert_ids_ptr, N, num_valid, cap, stride_be, stride_bse, stride_cm, smem_a_top,
                    smem_a_bot, smem_b_left, smem_b_right, smem_a_sc_t, smem_a_sc_b, smem_b_sc, tile_off, a_tile_off,
                    a_sc_offsets, b_sc_offsets, a_sc_k, b_sc_k, scale_a_layout: tl.constexpr,
                    scale_b_layout: tl.constexpr, scale_b_pair_layout: tl.constexpr, group_layout: tl.constexpr,
                    K: tl.constexpr, DUAL_B: tl.constexpr, BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
                    PERSISTENT: tl.constexpr, TOP_ONLY: tl.constexpr, TOP_K: tl.constexpr = 1,
                    FUSE_WSUM: tl.constexpr = False, NATURAL_B: tl.constexpr = False, STACKED_B: tl.constexpr = False):
    """One 256x256 tile whose first two K-stages are already in flight (see ``_moe_mxfp4_quad_kernel``).

    TOP_ONLY tiles have only padding in their bottom half: they skip its MFMAs and stores but still issue its loads,
    so the commit-group sequence (and the next tile's prefetch) is unchanged."""
    BLOCK_N: tl.constexpr = 256
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2
    HALF_K: tl.constexpr = 128
    KP: tl.constexpr = K // 2
    n_steps: tl.constexpr = K // 256
    B_KSTEP: tl.constexpr = HALF_K if NATURAL_B else HALF_K * 16
    a_half_m = HALF_M * KP
    a_sc_half_m = HALF_M
    pid_m, pid_n = _quad_tile(tile, num_pid_m, num_pid_n, GROUP_M)
    expert = tl.load(expert_ids_ptr + pid_m)
    m0 = pid_m * BLOCK_M
    n_left = pid_n * BLOCK_N
    n_right = n_left + HALF_N
    row_l, row_r = _b_rows(pid_n, N, DUAL_B, NATURAL_B, STACKED_B)
    b_half_n = (row_r - row_l).to(tl.int64) * KP
    # The first two K-stages of this tile are already in flight.
    a_base = a_ptr + m0.to(tl.int64) * KP + 2 * HALF_K
    b_base = b_ptr + expert.to(tl.int64) * stride_be + row_l.to(tl.int64) * KP + 2 * B_KSTEP
    a_scales_ptr = as_ptr + m0 + 2 * a_sc_k
    b_scales_ptr = bs_ptr + expert.to(tl.int64) * stride_bse + n_left * 16 + b_sc_k

    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)
    a_sc_top = tlx.local_load(smem_a_sc_t[0], layout=scale_a_layout)
    b_sc_left, b_sc_right = _b_scales_step(smem_b_sc[0], 0, scale_b_pair_layout, scale_b_layout, HALF_N)

    for k in tl.range(0, n_steps - 2, 2, num_stages=1):
        for s in tl.static_range(2):
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tl = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_left, b_sc_left, "e2m1", acc_tl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                if not TOP_ONLY:
                    a_bot = tlx.local_load(smem_a_bot[s], relaxed=True)
                    a_sc_bot = tlx.local_load(smem_a_sc_b[s], layout=scale_a_layout)
                tlx.amd_sched_barrier(0)
                tlx.buffer_load_to_local(smem_b_left[s], b_base + s * B_KSTEP, tile_off, contiguity=16)
                if s == 0:
                    # Pair k/2 is being computed; its buffer's other half is read below, so fill the other buffer.
                    tlx.buffer_load_to_local(smem_b_sc[(k // 2 + 1) % 2], b_scales_ptr, b_sc_offsets, contiguity=4)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                if not TOP_ONLY:
                    acc_bl = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_left, b_sc_left, "e2m1", acc_bl)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_right = tlx.local_load(tlx.local_trans(smem_b_right[s]), relaxed=True)
                tlx.amd_sched_barrier(0)
                tlx.buffer_load_to_local(smem_a_top[s], a_base + s * HALF_K, a_tile_off)
                tlx.buffer_load_to_local(smem_a_sc_t[s], a_scales_ptr + s * a_sc_k, a_sc_offsets)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_right, b_sc_right, "e2m1", acc_tr)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_left = tlx.local_load(tlx.local_trans(smem_b_left[1 - s]), relaxed=True)
                tlx.amd_sched_barrier(0)
                tlx.buffer_load_to_local(smem_a_bot[s], a_base + a_half_m + s * HALF_K, a_tile_off)
                tlx.buffer_load_to_local(smem_a_sc_b[s], a_scales_ptr + a_sc_half_m + s * a_sc_k, a_sc_offsets)
                tlx.async_load_commit_group()
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                if not TOP_ONLY:
                    acc_br = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_right, b_sc_right, "e2m1", acc_br)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_top = tlx.local_load(smem_a_top[1 - s], relaxed=True)
                a_sc_top = tlx.local_load(smem_a_sc_t[1 - s], layout=scale_a_layout)
                if s == 0:
                    b_sc_left, b_sc_right = _b_scales_step(smem_b_sc[(k // 2) % 2], 1, scale_b_pair_layout,
                                                           scale_b_layout, HALF_N)
                else:
                    b_sc_left, b_sc_right = _b_scales_step(smem_b_sc[(k // 2 + 1) % 2], 0, scale_b_pair_layout,
                                                           scale_b_layout, HALF_N)
                tlx.amd_sched_barrier(0)
                tlx.buffer_load_to_local(smem_b_right[s], b_base + b_half_n + s * B_KSTEP, tile_off, contiguity=16)
                tlx.async_load_commit_group()
                if s == 1:
                    a_base += 2 * HALF_K
                    b_base += 2 * B_KSTEP
                    a_scales_ptr += 2 * a_sc_k
                    b_scales_ptr += b_sc_k

    acc_tl = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_left, b_sc_left, "e2m1", acc_tl)
    tlx.async_load_wait_group(5)
    if not TOP_ONLY:
        a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
        a_sc_bot = tlx.local_load(smem_a_sc_b[0], layout=scale_a_layout)
        acc_bl = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_left, b_sc_left, "e2m1", acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
    acc_tr = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_right, b_sc_right, "e2m1", acc_tr)
    tlx.async_load_wait_group(3)
    b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
    if not TOP_ONLY:
        acc_br = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_right, b_sc_right, "e2m1", acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(smem_a_top[1], relaxed=True)
    a_sc_top = tlx.local_load(smem_a_sc_t[1], layout=scale_a_layout)
    b_sc_left, b_sc_right = _b_scales_step(smem_b_sc[(n_steps // 2 - 1) % 2], 1, scale_b_pair_layout, scale_b_layout,
                                           HALF_N)
    acc_tl = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_left, b_sc_left, "e2m1", acc_tl)
    tlx.async_load_wait_group(1)
    if not TOP_ONLY:
        a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
        a_sc_bot = tlx.local_load(smem_a_sc_b[1], layout=scale_a_layout)
        acc_bl = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_left, b_sc_left, "e2m1", acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
    if PERSISTENT:
        # Every LDS buffer is now in registers: start the next tile (or re-read this one on the last pass).
        nxt = tile + num_progs
        nxt = tl.where(nxt < num_tiles, nxt, tile)
        _quad_prologue(nxt, num_pid_m, num_pid_n, a_ptr, as_ptr, b_ptr, bs_ptr, expert_ids_ptr, stride_be, stride_bse,
                       smem_a_top, smem_a_bot, smem_b_left, smem_b_right, smem_a_sc_t, smem_a_sc_b, smem_b_sc, tile_off,
                       a_tile_off, a_sc_offsets, b_sc_offsets, a_sc_k, b_sc_k, KP, BLOCK_M, GROUP_M, N, DUAL_B,
                       NATURAL_B, STACKED_B)
    acc_tr = tl.dot_scaled(a_top, a_sc_top, "e2m1", b_right, b_sc_right, "e2m1", acc_tr)
    if not TOP_ONLY:
        acc_br = tl.dot_scaled(a_bot, a_sc_bot, "e2m1", b_right, b_sc_right, "e2m1", acc_br)

    if DUAL_B:
        rows_top = m0 + tl.arange(0, HALF_M)
        rows_bot = rows_top + HALF_M
        qc = pid_n * (HALF_N // 2) + tl.arange(0, HALF_N // 2)
        sg = pid_n * (HALF_N // 32) + tl.arange(0, HALF_N // 32)
        h_top = tlx.require_layout((_silu(acc_tl) * acc_tr).to(tl.bfloat16), group_layout)
        q_top, s_top = _mxfp4_quant(h_top.to(tl.float32), HALF_M, HALF_N)
        tl.store(c_ptr + rows_top[:, None].to(tl.int64) * (N // 2) + qc[None, :], q_top)
        tl.store(cs_ptr + sg[None, :] * cap + rows_top[:, None], s_top)
        if not TOP_ONLY:
            h_bot = tlx.require_layout((_silu(acc_bl) * acc_br).to(tl.bfloat16), group_layout)
            q_bot, s_bot = _mxfp4_quant(h_bot.to(tl.float32), HALF_M, HALF_N)
            tl.store(c_ptr + rows_bot[:, None].to(tl.int64) * (N // 2) + qc[None, :], q_bot)
            tl.store(cs_ptr + sg[None, :] * cap + rows_bot[:, None], s_bot)
    elif FUSE_WSUM:
        # c is the zeroed [T, H] output and cs the flat routing weights: each slot adds bf16(y) * w for its token.
        offs_l = n_left + tl.arange(0, HALF_N)
        offs_r = n_right + tl.arange(0, HALF_N)
        _wsum_rows(acc_tl, acc_tr, sorted_ids_ptr + m0, c_ptr, cs_ptr, offs_l, offs_r, num_valid, stride_cm, HALF_M,
                   TOP_K)
        if not TOP_ONLY:
            _wsum_rows(acc_bl, acc_br, sorted_ids_ptr + m0 + HALF_M, c_ptr, cs_ptr, offs_l, offs_r, num_valid,
                       stride_cm, HALF_M, TOP_K)
    else:
        tok_top = tl.load(sorted_ids_ptr + m0 + tl.arange(0, HALF_M))
        tok_bot = tl.load(sorted_ids_ptr + m0 + HALF_M + tl.arange(0, HALF_M))
        c_top = c_ptr + tok_top[:, None].to(tl.int64) * stride_cm
        c_bot = c_ptr + tok_bot[:, None].to(tl.int64) * stride_cm
        offs_l = n_left + tl.arange(0, HALF_N)
        offs_r = n_right + tl.arange(0, HALF_N)
        vt = (tok_top < num_valid)[:, None]
        vb = (tok_bot < num_valid)[:, None]
        tl.store(c_top + offs_l[None, :], acc_tl.to(tl.bfloat16), mask=vt)
        tl.store(c_top + offs_r[None, :], acc_tr.to(tl.bfloat16), mask=vt)
        if not TOP_ONLY:
            tl.store(c_bot + offs_l[None, :], acc_bl.to(tl.bfloat16), mask=vb)
            tl.store(c_bot + offs_r[None, :], acc_br.to(tl.bfloat16), mask=vb)


@triton.jit
def _moe_mxfp4_quad_kernel(
    a_ptr,  # uint8 [cap, K/2], rows in sorted order
    as_ptr,  # uint8 [K/32, cap]
    b_ptr,  # uint8 [E, NB, K/2]
    bs_ptr,  # uint8 [E, K/32, NB]
    c_ptr,  # DUAL_B: uint8 [cap, N/2] sorted; else bf16 [slots, N]
    cs_ptr,  # DUAL_B: uint8 [N/32, cap]
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_ptr,
    N,
    NB,
    K: tl.constexpr,
    num_valid,
    cap,
    stride_be,
    stride_bse,
    stride_cm,
    DUAL_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    PERSISTENT: tl.constexpr,
    TOP_K: tl.constexpr = 1,
    FUSE_WSUM: tl.constexpr = False,
    NATURAL_B: tl.constexpr = False,
    STACKED_B: tl.constexpr = False,
):
    """2x2-quadrant inter-wave e2m1 pipeline over sorted, BLOCK_M-padded rows.

    Layouts, scale XOR swizzles and the pipeline follow the 8-wave gfx9 a4w4 tutorial. Each program computes a
    256x256 tile; B rows [256 * pid_n, +256) of the expert are its left and right halves (gate/up interleaved in
    128-row blocks for the stage-1 weights, see ``prepare_w13``). With K = 4096 a tile is only 16 K-steps; when
    PERSISTENT, each program loops over tiles and issues the next tile's first two stages before its own last
    MFMAs and epilogue, hiding the cold expert-weight loads that would otherwise start every tile. That keeps the
    pipeline LDS live through the epilogue (it would not fit next to the 256-row stage-1 epilogue's conversions)."""
    BLOCK_N: tl.constexpr = 256
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2
    HALF_K: tl.constexpr = 128  # packed bytes per K-step (256 elements)
    NG: tl.constexpr = 8  # scale groups per K-step
    KP: tl.constexpr = K // 2
    n_steps: tl.constexpr = K // 256
    tl.static_assert(BLOCK_M == 256 or BLOCK_M == 128)
    tl.static_assert(n_steps % 2 == 0 and n_steps >= 4)

    # B side (and A at BLOCK_M=256) uses the tutorial's [128, 128]-quadrant layouts verbatim. The 64-row A
    # variants drop the row-64 basis (or broadcast it, where a DMA layout needs all nine thread bits).
    g_load_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((16, 32, 64, 128, 4096, 8192, 256, 512, 1024), (1, 2, 4, 8, 2048)),
    )
    shared_tile: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0],
         [16, 0]],
        [HALF_N, HALF_K],
    )
    # B tiles follow the weights' 1 KiB blocks (16 rows x 64 bytes in MFMA lane order): each wave's DMA moves one
    # contiguous block, and the LDS copy is already in operand order.
    b_load_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=((128, 256, 512, 1024, 16, 32, 64, 2048, 4096), (1, 2, 4, 8, 8192)),
    )
    b_tile: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0],
         [64, 0]],
        [HALF_N, HALF_K],
    )
    if HALF_M == 128:
        a_load_layout: tl.constexpr = g_load_layout
        a_tile: tl.constexpr = shared_tile
        shared_a_scales: tl.constexpr = tlx.shared_linear_layout_encoding(
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2], [16, 4]],
            block_bases=[],
            alignment=16,
        )
        scale_load_a_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((32, 64, 128, 256, 512, 1, 2, 132, 0), (8, 16)),
        )
        scale_a_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
            stride=((8, 16, 32, 64, 1, 2, 0, 0, 128), (4, 256, 512)),
        )
        # One 32-column quant group per thread: rows over lanes/warps, 32 contiguous columns in registers.
        group_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
            stride=((32, 64, 128, 256, 512, 1024, 2048, 4096, 8192), (1, 2, 4, 8, 16)),
        )
    else:
        # Lanes must fill consecutive 16-byte LDS slots: lane bits = LDS address bits 4..9 of ``a_tile``. Rows go in
        # the 128-row tile's order, with row 16 in its row-64 slot: rows 1, 32, 16 share a 1 KiB padding interval and
        # rows 2, 4, 8 sit in separate ones, so the padding staggers the banks of the rows one MFMA operand read
        # touches. With row 2 inside the first interval instead, 31% of LDS cycles were bank conflicts; now 8%.
        a_load_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
            stride=((16, 32, 64, 128, 4096, 2048, 256, 512, 1024), (1, 2, 4, 8)),
        )
        a_tile: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [[1024, 32]],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [16, 0], [2, 0], [4, 0],
             [8, 0]],
            [HALF_M, HALF_K],
        )
        shared_a_scales: tl.constexpr = tlx.shared_linear_layout_encoding(
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 1], [0, 2], [16, 4]],
            block_bases=[],
            alignment=16,
        )
        # 128 dwords: lanes = LDS dword slots, one warp bit, two broadcast warp bits.
        scale_load_a_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((32, 64, 128, 256, 1, 2, 132, 0, 0), (8, 16)),
        )
        scale_a_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((8, 16, 32, 64, 1, 2, 0, 0, 128), (4, 256)),
        )
        # 16 contiguous columns per thread; a quant group spans two adjacent lanes.
        group_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
            stride=((16, 128, 256, 512, 1024, 2048, 4096, 32, 64), (1, 2, 4, 8)),
        )
    # B scales: one packed K-step pair [256 rows, 16 bytes] per buffer, row-major, filled a dword per lane.
    shared_b_scales: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
                      [128, 0]],
        block_bases=[],
        alignment=16,
    )
    scale_load_b_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
        stride=((4, 8, 16, 32, 64, 128, 256, 512, 1024), (1, 2, 2048)),
    )
    scale_b_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
        stride=((8, 16, 32, 64, 1, 2, 128, 256, 0), (4, 512)),
    )
    # A pair [256, 16] read so that each step's bytes land as scale_b_layout's [256, 8] combined tile.
    scale_b_pair_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((16, 32, 64, 128, 4, 8, 256, 512, 0), (1, 2, 1024, 2048)),
    )

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(tl.load(num_post_ptr), BLOCK_M)
    num_pid_n = N // HALF_N if DUAL_B else N // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    if pid >= num_tiles:
        return

    smem_a_top = tlx.local_alloc((HALF_M, HALF_K), tl.uint8, 2, layout=a_tile)
    smem_a_bot = tlx.local_alloc((HALF_M, HALF_K), tl.uint8, 2, layout=a_tile)
    smem_b_left = tlx.local_alloc((HALF_N, HALF_K), tl.uint8, 2, layout=b_tile)
    smem_b_right = tlx.local_alloc((HALF_N, HALF_K), tl.uint8, 2, layout=b_tile)
    smem_a_sc_t = tlx.local_alloc((HALF_M, NG), tl.uint8, 2, layout=shared_a_scales)
    smem_a_sc_b = tlx.local_alloc((HALF_M, NG), tl.uint8, 2, layout=shared_a_scales)
    smem_b_sc = tlx.local_alloc((BLOCK_N, 16), tl.uint8, 2, layout=shared_b_scales)

    # Natural weights load the same 16 bytes per lane, at their row-major addresses: the LDS tile is identical.
    if NATURAL_B:
        tile_off = tlx.require_layout(tl.arange(0, HALF_N)[:, None] * KP + tl.arange(0, HALF_K)[None, :], b_load_layout)
    else:
        tile_off = tlx.require_layout(_shuf_off(tl.arange(0, HALF_N)[:, None],
                                                tl.arange(0, HALF_K)[None, :], KP), b_load_layout)
    a_tile_off = tlx.require_layout(tl.arange(0, HALF_M)[:, None] * KP + tl.arange(0, HALF_K)[None, :], a_load_layout)
    a_sc_offsets = tlx.require_layout(
        tl.mul(tl.arange(0, HALF_M)[:, None], 1, sanitize_overflow=False) +
        tl.mul(tl.arange(0, NG)[None, :], cap, sanitize_overflow=False), scale_load_a_layout)
    b_sc_offsets = tlx.require_layout(
        tl.mul(tl.arange(0, BLOCK_N)[:, None], 16, sanitize_overflow=False) + tl.arange(0, 16)[None, :],
        scale_load_b_layout)
    a_sc_k = NG * cap
    b_sc_k = NB * 16

    _quad_prologue(pid, num_pid_m, num_pid_n, a_ptr, as_ptr, b_ptr, bs_ptr, expert_ids_ptr, stride_be, stride_bse,
                   smem_a_top, smem_a_bot, smem_b_left, smem_b_right, smem_a_sc_t, smem_a_sc_b, smem_b_sc, tile_off,
                   a_tile_off, a_sc_offsets, b_sc_offsets, a_sc_k, b_sc_k, KP, BLOCK_M, GROUP_M, N, DUAL_B, NATURAL_B,
                   STACKED_B)
    if PERSISTENT:
        for tile in range(pid, num_tiles, num_progs):
            pid_m, _ = _quad_tile(tile, num_pid_m, num_pid_n, GROUP_M)
            # Sorted blocks put an expert's real rows first, so this row decides whether the bottom half is padding.
            if tl.load(sorted_ids_ptr + pid_m * BLOCK_M + HALF_M) >= num_valid:
                _quad_tile_body(tile, num_pid_m, num_pid_n, num_tiles, num_progs, a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr,
                                cs_ptr, sorted_ids_ptr, expert_ids_ptr, N, num_valid, cap, stride_be, stride_bse,
                                stride_cm, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, smem_a_sc_t, smem_a_sc_b,
                                smem_b_sc, tile_off, a_tile_off, a_sc_offsets, b_sc_offsets, a_sc_k, b_sc_k,
                                scale_a_layout, scale_b_layout, scale_b_pair_layout, group_layout, K, DUAL_B, BLOCK_M,
                                GROUP_M, True, True, TOP_K, FUSE_WSUM, NATURAL_B, STACKED_B)
            else:
                _quad_tile_body(tile, num_pid_m, num_pid_n, num_tiles, num_progs, a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr,
                                cs_ptr, sorted_ids_ptr, expert_ids_ptr, N, num_valid, cap, stride_be, stride_bse,
                                stride_cm, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, smem_a_sc_t, smem_a_sc_b,
                                smem_b_sc, tile_off, a_tile_off, a_sc_offsets, b_sc_offsets, a_sc_k, b_sc_k,
                                scale_a_layout, scale_b_layout, scale_b_pair_layout, group_layout, K, DUAL_B, BLOCK_M,
                                GROUP_M, True, False, TOP_K, FUSE_WSUM, NATURAL_B, STACKED_B)
        tlx.async_load_wait_group(0)
    else:
        # No loop: the pipeline LDS dies before the epilogue, which reuses it for layout conversions. A TOP_ONLY
        # branch here would keep two epilogues' conversion buffers live and overflow LDS at BLOCK_M=256.
        _quad_tile_body(pid, num_pid_m, num_pid_n, num_tiles, num_progs, a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, cs_ptr,
                        sorted_ids_ptr, expert_ids_ptr, N, num_valid, cap, stride_be, stride_bse, stride_cm, smem_a_top,
                        smem_a_bot, smem_b_left, smem_b_right, smem_a_sc_t, smem_a_sc_b, smem_b_sc, tile_off,
                        a_tile_off, a_sc_offsets, b_sc_offsets, a_sc_k, b_sc_k, scale_a_layout, scale_b_layout,
                        scale_b_pair_layout, group_layout, K, DUAL_B, BLOCK_M, GROUP_M, False, False, TOP_K, FUSE_WSUM,
                        NATURAL_B, STACKED_B)


def _pack_scales(s: torch.Tensor) -> torch.Tensor:
    """[E, N, K/32] -> the packed layout of ``_scale_off`` per expert: [E, K/512, N, 16]."""
    E, N, KS = s.shape
    assert KS % 16 == 0
    return s.view(E, N, KS // 16, 2, 2, 4).permute(0, 2, 1, 5, 4, 3).contiguous().view(E, KS // 16, N, 16)


def _shuffle_weights(w: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] -> the 1 KiB-block layout of ``_shuf_off`` (same shape, bytes reordered)."""
    E, N, KP = w.shape
    assert N % 16 == 0 and KP % 64 == 0
    return w.view(E, N // 16, 16, KP // 64, 4, 16).permute(0, 1, 3, 4, 2, 5).contiguous().view(E, N, KP)


def _interleave_128(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Reorder [gate; up] along ``dim`` into 128-row blocks g0, u0, g1, u1, ..."""
    n = t.shape[dim] // 2
    g, u = t.narrow(dim, 0, n), t.narrow(dim, n, n)
    shp = list(g.shape)
    shp[dim:dim + 1] = [n // 128, 128]
    return torch.stack([g.reshape(shp), u.reshape(shp)], dim=dim + 1).flatten(dim, dim + 2).contiguous()


def _check_weights(name: str, w: torch.Tensor, s: torch.Tensor, row_multiple: int) -> None:
    """Natural-layout checks for ``prepare_w13`` / ``prepare_w2``: w uint8 [E, N, K/2], s uint8 [E, N, K/32]."""
    if w.dtype != torch.uint8 or s.dtype != torch.uint8 or w.ndim != 3 or s.ndim != 3:
        raise ValueError(f"{name}: expected uint8 [E, N, K/2] weights and uint8 [E, N, K/32] E8M0 scales; got "
                         f"{w.dtype} {tuple(w.shape)} and {s.dtype} {tuple(s.shape)}")
    E, N, KP = w.shape
    if s.shape != (E, N, KP // 16):
        raise ValueError(f"{name}: scales {tuple(s.shape)} do not match weights {tuple(w.shape)}; expected "
                         f"{(E, N, KP // 16)} (one scale per 32 elements, two elements per byte)")
    if N % row_multiple:
        raise ValueError(f"{name}: {N} rows is not a multiple of {row_multiple}")
    if (2 * KP) % 512:
        raise ValueError(f"{name}: reduction dim {2 * KP} is not a multiple of 512 (one packed scale pair per row)")


def prepare_w13(w13: torch.Tensor, w13_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Natural stage-1 weights -> kernel layout: gate/up interleaved in 128-row blocks, then pre-shuffled
    (``_shuf_off``), with packed scales [E, K/512, 2I, 16] (``_scale_off``)."""
    # Gate and up halves each split into whole 128-row blocks.
    _check_weights("prepare_w13", w13, w13_scale, 256)
    return _shuffle_weights(_interleave_128(w13, 1)), _pack_scales(_interleave_128(w13_scale, 1))


def prepare_w2(w2: torch.Tensor, w2_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Natural stage-2 weights -> kernel layout: pre-shuffled, with packed scales [E, I/512, H, 16]."""
    _check_weights("prepare_w2", w2, w2_scale, 16)
    return _shuffle_weights(w2.contiguous()), _pack_scales(w2_scale.contiguous())


def prepare_w13_scales(w13: torch.Tensor, w13_scale: torch.Tensor) -> torch.Tensor:
    """Scales only, for ``moe_mxfp4(..., natural_weights=True)``: the weights stay in the checkpoint layout (the one
    ``moe_mxfp4_a16w4`` reads), so a server keeps a single copy; the scales are 1/16 of the weight bytes."""
    _check_weights("prepare_w13_scales", w13, w13_scale, 256)
    return _pack_scales(_interleave_128(w13_scale, 1))


def prepare_w2_scales(w2: torch.Tensor, w2_scale: torch.Tensor) -> torch.Tensor:
    """Scales only, for ``moe_mxfp4(..., natural_weights=True)``."""
    _check_weights("prepare_w2_scales", w2, w2_scale, 16)
    return _pack_scales(w2_scale.contiguous())


def _launch_quad(a, a_s, b, b_st, c, c_s, sorted_ids, expert_ids, num_post, N, K, num_valid, dual_b, cfg, topk=1,
                 fuse_wsum=False, natural=False, stacked=False):
    bm = cfg["BLOCK_M"]
    cap = sorted_ids.numel()
    num_m_blocks = min(triton.cdiv(cap, bm), max(num_valid, 1))
    num_n = N // 128 if dual_b else N // 256
    # Measured at T=4096: persistence speeds up both 128-row GEMMs but slows the 256-row stage-2 GEMM by ~30%.
    persistent = bm == 128
    grid = min(num_m_blocks * num_n, _num_cus(a.device)) if persistent else num_m_blocks * num_n
    _moe_mxfp4_quad_kernel[(grid, )](a, a_s, b, b_st, c, c_s if c_s is not None else c,
                                     sorted_ids, expert_ids, num_post, N, b.shape[1], K, num_valid, cap, b.stride(0),
                                     b_st.stride(0), c.stride(0), DUAL_B=dual_b, BLOCK_M=bm, GROUP_M=cfg["GROUP_M"],
                                     PERSISTENT=persistent, TOP_K=topk, FUSE_WSUM=fuse_wsum, NATURAL_B=natural,
                                     STACKED_B=stacked, num_warps=8, num_stages=1, matrix_instr_nonkdim=16,
                                     llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ("amdgpu-post-sched-strategy",
                                                                                   "nop")))


@functools.cache
def _num_cus(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _launch(a, a_s, b, b_s, c, c_s, sorted_ids, expert_ids, num_post, N, K, num_valid, a_rows, topk, a_row_div_topk,
            dual_b, cfg, natural=False, stacked=False):
    bm = cfg["BLOCK_M"]
    num_m_blocks = min(triton.cdiv(sorted_ids.numel(), bm), max(num_valid, 1))
    sorted_cfg = cfg.get("SORTED", False)
    extra = dict(SORTED=True, cap=sorted_ids.numel(), B_CACHE="", UNROLL=K // 512) if sorted_cfg else {}
    if sorted_cfg and bm == 96 and not dual_b:
        # Stage 2 with 96-row blocks: four activation buffers, one wait + barrier per pair of K-steps.
        extra["R4"] = True
    _moe_mxfp4_small_kernel[(num_m_blocks * (N // 128 if dual_b else N // 256), )](
        a, a_s, b, b_s, c, c_s if c_s is not None else c, sorted_ids, expert_ids, num_post, num_valid, a_rows, N, K,
        b.stride(0), b_s.stride(0), b.shape[1], TOP_K=topk, A_ROW_DIV_TOPK=a_row_div_topk, DUAL_B=dual_b, BLOCK_M=bm,
        WN=cfg["WN"], A_LDS=cfg["A_LDS"], NATURAL_B=natural, STACKED_B=stacked, num_warps=cfg["WN"], num_stages=1,
        matrix_instr_nonkdim=16, **extra, **({"waves_per_eu": cfg["WPE"]} if cfg.get("WPE") else {}))


def moe_mxfp4(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale, expert_start=0, post_norm_weight=None,
              post_norm_eps=1e-6, post_norm_gain=0.0, out=None, cfg=None, num_experts=None, natural_weights=False,
              gate_up_stacked=False):
    """Weights come from ``prepare_w13`` / ``prepare_w2``: w13 [E, 2I, K/2] gate/up interleaved in 128-row
    blocks, w2 [E, H, I/2], both pre-shuffled, with packed scales [E, K/512, 2I, 16] and [E, I/512, H, 16].
    With ``natural_weights``, w13 [E, 2I, K/2] ([gate; up]) and w2 are the checkpoint's natural uint8 weights and
    only the scales come from ``prepare_w13_scales`` / ``prepare_w2_scales``; the result is bitwise the same.
    With ``gate_up_stacked``, w13 is pre-shuffled (``_shuf_off`` 1 KiB blocks, the layout of aiter's
    ``shuffle_weight(w13, (16, 16))``) but keeps [gate; up] instead of interleaving 128-row blocks.
    ``num_experts`` is how many experts ``topk_ids`` route over (all of them under expert parallelism); it only
    picks the engine and defaults to the E local experts."""
    T, K = x.shape
    E, two_i, _ = w13.shape
    I = two_i // 2
    H = w2.shape[1]
    topk = topk_ids.shape[1]
    num_valid = T * topk
    dev = x.device
    assert w13.dtype == torch.uint8 and w13.shape[2] * 2 == K and w13_scale.shape == (E, K // 512, two_i, 16)
    assert w2.dtype == torch.uint8 and w2.shape == (E, H, I // 2) and w2_scale.shape == (E, I // 512, H, 16)
    assert K % 1024 == 0 and I % 1024 == 0 and H % 256 == 0
    for w in (w13, w13_scale, w2, w2_scale):
        assert w.is_contiguous()
    if out is None:
        out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    if T == 0:
        return out

    cfg = cfg or pick_config(num_valid, num_experts or E)
    ids = topk_ids.contiguous()
    scores32 = topk_weights.to(torch.float32).contiguous()
    has_norm = post_norm_weight is not None
    # The 128-row quad stage-2 GEMM can apply the routing weights and accumulate into ``out`` itself with BF16
    # atomics: each add rounds to BF16 and the order varies run to run. Measured: ~1.5% faster at T=4096, but ~1.5%
    # slower with 256-row tiles (T=8192/16384). The post-norm needs whole expert rows first.
    fuse_wsum = cfg.get("QUAD", False) and cfg["BLOCK_M"] == 128 and not has_norm and cfg.get("FUSE_WSUM", True)
    y = None if fuse_wsum else torch.empty((num_valid, H), dtype=torch.bfloat16, device=dev)
    if cfg.get("QUAD", False):
        sorted_ids, expert_ids, num_post = _align_quad(ids, E, expert_start, cfg["BLOCK_M"])
        cap = sorted_ids.numel()
        # Quantize each token once, then copy rows into sorted order (each token has ~topk*E/E_total copies).
        tq, ts = quantize_rows(x)
        xq = torch.empty((cap, K // 2), dtype=torch.uint8, device=dev)
        xs = torch.empty((K // 32, cap), dtype=torch.uint8, device=dev)
        _sorted_gather_kernel[(cap // 16, )](tq, ts, xq, xs, sorted_ids, num_post, num_valid, cap, K=K, TOP_K=topk,
                                             BM=16, num_warps=4)
        inter_q = torch.empty((cap, I // 2), dtype=torch.uint8, device=dev)
        inter_s = torch.empty((I // 32, cap), dtype=torch.uint8, device=dev)
        _launch_quad(xq, xs, w13, w13_scale, inter_q, inter_s, sorted_ids, expert_ids, num_post, I, K, num_valid, True,
                     cfg, natural=natural_weights, stacked=gate_up_stacked)
        if fuse_wsum:
            out.zero_()
            _launch_quad(inter_q, inter_s, w2, w2_scale, out, scores32, sorted_ids, expert_ids, num_post, H, I,
                         num_valid, False, cfg, topk=topk, fuse_wsum=True, natural=natural_weights)
            return out
        _launch_quad(inter_q, inter_s, w2, w2_scale, y, None, sorted_ids, expert_ids, num_post, H, I, num_valid, False,
                     cfg, natural=natural_weights)
    elif cfg.get("SORTED", False):
        sorted_ids, expert_ids, num_post = _align_quad(ids, E, expert_start, cfg["BLOCK_M"])
        cap = sorted_ids.numel()
        xq, ts = quantize_rows(x, packed=True)
        xs = torch.empty((K // 512, cap, 16), dtype=torch.uint8, device=dev)
        _sort_scale_rows_kernel[(triton.cdiv(cap, 64), )](ts, xs, sorted_ids, num_post, num_valid, T, cap, TOP_K=topk,
                                                          NCH=K // 512, BR=64, num_warps=4)
        inter_q = torch.empty((cap, I // 2), dtype=torch.uint8, device=dev)
        inter_s = torch.empty((I // 512, cap, 16), dtype=torch.uint8, device=dev)
        # Stage 1 reads token rows of xq (A_ROW_DIV_TOPK) with tile-order scales [K/512, cap, 16].
        _launch(xq, xs, w13, w13_scale, inter_q, inter_s, sorted_ids, expert_ids, num_post, I, K, num_valid, cap, topk,
                True, True, cfg, natural=natural_weights, stacked=gate_up_stacked)
        _launch(inter_q, inter_s, w2, w2_scale, y, None, sorted_ids, expert_ids, num_post, H, I, num_valid, cap, topk,
                False, False, cfg, natural=natural_weights)
        _finalize_local_kernel[(T, )](y, post_norm_weight.contiguous() if has_norm else out, scores32, ids, out, H,
                                      expert_start, E, float(post_norm_eps), float(post_norm_gain), TOP_K=topk,
                                      HAS_NORM=has_norm, BLOCK_H=triton.next_power_of_2(H), num_warps=4)
        return out
    else:
        sorted_ids, expert_ids, num_post = _align(ids, E, expert_start, cfg["BLOCK_M"])
        xq, xs = quantize_rows(x, packed=True)
        inter_q = torch.empty((num_valid, I // 2), dtype=torch.uint8, device=dev)
        inter_s = torch.empty((I // 512, num_valid, 16), dtype=torch.uint8, device=dev)
        _launch(xq, xs, w13, w13_scale, inter_q, inter_s, sorted_ids, expert_ids, num_post, I, K, num_valid, T, topk,
                True, True, cfg, natural=natural_weights, stacked=gate_up_stacked)
        _launch(inter_q, inter_s, w2, w2_scale, y, None, sorted_ids, expert_ids, num_post, H, I, num_valid, num_valid,
                topk, False, False, cfg, natural=natural_weights)

    _finalize_kernel[(T, )](
        y,
        post_norm_weight.contiguous() if has_norm else out,
        scores32,
        ids,
        out,
        H,
        expert_start,
        E,
        float(post_norm_eps),
        float(post_norm_gain),
        0,
        TOP_K=topk,
        TOP_K_BLOCK=triton.next_power_of_2(topk),
        HAS_NORM=has_norm,
        BLOCK_H=triton.next_power_of_2(H),
        SPLIT_K=1,
        num_warps=4,
    )
    return out
