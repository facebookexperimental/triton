"""Decode-sized MXFP4 MoE prologue on gfx950: expert-aligned slot sort plus MXFP4 activation quantization in one launch.

An A4W4 routed MoE with AITER's FlyDSL / asm stage-1 kernels needs, before its first GEMM: the slots sorted by local
expert into ``block_m``-row blocks (``sorted_ids`` with ``token | slot << 24`` entries and ``token_num | topk << 24``
padding, one local expert id per block, the padded slot count), the activations quantized to MXFP4 per row, and the
E8M0 scales reordered into the sorted rows' 32x8 MFMA scale tiles. Serving code issues these as five or six launches
(an expert-mask fill, two sorting phases, the row quantization and the scale sort). At decode every launch costs
4-6 us, so this op does all of it in one: each program recomputes the (tiny) routing layout from ``topk_ids``,
quantizes its tile of rows, and writes each row's scales straight to the sorted positions of the row's slots; one
extra program writes ``sorted_ids``, ``sorted_expert_ids`` and ``num_valid_ids``. Scale rows of padding slots are not
written (the GEMMs never use them for a real token's output). Every output byte has exactly one writer, so no grid-wide
synchronization is needed.

Bit-exact with AITER: the quantization is ``dynamic_mxfp4_quant`` (round-up amax, RNE E2M1), the sort order is
``moe_sorting`` with an expert mask (local experts ascending, tokens ascending within an expert), and the scale
layout is ``moe_mxfp4_sort``'s ``[M/32, N/8, 4, 16]`` uint32 tiles (byte ``m // 16 % 2 + 2 * (n // 4 % 2)`` of the
word at ``[m // 32, n // 8, n % 4, m % 16]``).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

MAX_TOKENS = 64


@triton.jit
def _layout(ids_ptr, T, local_start, TOP_K: tl.constexpr, TOP_K_PAD: tl.constexpr, T_PAD: tl.constexpr, E: tl.constexpr,
            E_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr):
    """Routing layout from topk_ids: local expert of every slot [T_PAD, TOP_K_PAD] (-1 if not local or padding),
    slot counts and block-aligned base row of every local expert [E_BLOCK], and each slot's sorted row
    [T_PAD, TOP_K_PAD]."""
    t = tl.arange(0, T_PAD)
    k = tl.arange(0, TOP_K_PAD)
    ids = tl.load(ids_ptr + t[:, None] * TOP_K + k[None, :], mask=(t < T)[:, None] & (k < TOP_K)[None, :], other=-1)
    loc = ids - local_start
    loc = tl.where((ids >= 0) & (loc >= 0) & (loc < E), loc, -1)
    e = tl.arange(0, E_BLOCK)
    hit3 = loc[:, :, None] == e[None, None, :]  # [T_PAD, TOP_K, E_BLOCK]
    onehot = tl.max(hit3.to(tl.int32), axis=1)  # [T_PAD, E_BLOCK]: a token routes to an expert at most once
    cnt = tl.sum(onehot, axis=0)
    span = (cnt + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    base = tl.cumsum(span, axis=0) - span
    prefix = tl.cumsum(onehot, axis=0) - onehot  # tokens before t routed to the same expert
    row3 = base[None, None, :] + prefix[:, None, :]
    pos = tl.sum(tl.where(hit3, row3, 0), axis=2)
    pos = tl.where(loc >= 0, pos, -1)
    total = tl.sum(span, axis=0)
    return loc, cnt, span, base, pos, total


@triton.jit
def _scale_byte_off(r, c, N_TILES: tl.constexpr):
    """Byte offset of scale (sorted row r, group c) in moe_mxfp4_sort's [M/32, N/8, 4, 16] uint32 tiles."""
    word = (r // 32) * (N_TILES * 64) + (c // 8) * 64 + (c % 4) * 16 + (r % 16)
    return word * 4 + (r // 16) % 2 + 2 * ((c // 4) % 2)


@triton.jit
def _moe_mxfp4_sort_quant_kernel(x_ptr, stride_xm, ids_ptr, q_ptr, s_ptr, sorted_ids_ptr, sorted_eids_ptr,
                                 num_valid_ptr, T, local_start, K: tl.constexpr, TOP_K: tl.constexpr,
                                 TOP_K_PAD: tl.constexpr, T_PAD: tl.constexpr, E: tl.constexpr, E_BLOCK: tl.constexpr,
                                 BLOCK_M: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, MAX_BLOCKS: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)
    NG: tl.constexpr = BK // 32
    N_TILES: tl.constexpr = K // 32 // 8
    loc, cnt, span, base, pos, total = _layout(ids_ptr, T, local_start, TOP_K, TOP_K_PAD, T_PAD, E, E_BLOCK, BLOCK_M)
    if pid_k == K // BK:
        # metadata column: padding ids, one local expert id per block, counts
        if pid_t == 0:
            # Per expert: padding rows [base + cnt, base + span) (fewer than BLOCK_M) and its blocks
            # [base / BLOCK_M, (base + span) / BLOCK_M) (at most MAX_BLOCKS), so no scan over row positions.
            j = tl.arange(0, BLOCK_M)
            prow = base[:, None] + cnt[:, None] + j[None, :]
            tl.store(sorted_ids_ptr + prow,
                     tl.full((E_BLOCK, BLOCK_M), 0, tl.int32) + (T | (TOP_K << 24)), mask=prow < (base + span)[:, None])
            b = tl.arange(0, MAX_BLOCKS)
            blk = base[:, None] // BLOCK_M + b[None, :]
            eid = tl.broadcast_to(tl.arange(0, E_BLOCK)[:, None], (E_BLOCK, MAX_BLOCKS))
            tl.store(sorted_eids_ptr + blk, eid, mask=b[None, :] * BLOCK_M < span[:, None])
            t = tl.arange(0, T_PAD)
            k = tl.arange(0, TOP_K_PAD)
            val = t[:, None] | (k[None, :] << 24)
            tl.store(sorted_ids_ptr + tl.where(pos >= 0, pos, 0), val, mask=pos >= 0)
            tl.store(num_valid_ptr, total)
            tl.store(num_valid_ptr + 1, T)
        return

    # ---- quantize rows [pid_t * BT, +BT) x cols [pid_k * BK, +BK) exactly as AITER dynamic_mxfp4_quant ----
    rows = pid_t * BT + tl.arange(0, BT)
    cols = pid_k * BK + tl.arange(0, BK)
    rmask = rows < T
    x = tl.load(x_ptr + rows[:, None].to(tl.int64) * stride_xm + cols[None, :], mask=rmask[:, None], other=0.0)
    x = tl.reshape(x.to(tl.float32), (BT, NG, 32))
    amax = tl.max(tl.abs(x), axis=2, keep_dims=True)
    amax = ((amax.to(tl.int32, bitcast=True) + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000).to(
        tl.float32, bitcast=True)
    unb = tl.clamp(tl.floor(tl.log2(amax)) - 2, min=-127, max=127)
    # 2**-127 is subnormal in fp32 and the backend flushes it to zero, so unb = 127 scales in two normal steps.
    qx = (x * tl.exp2(-tl.minimum(unb, 126)) * tl.where(unb > 126, 0.5, 1.0)).to(tl.uint32, bitcast=True)
    sgn = qx & 0x80000000
    qx = qx ^ sgn
    qf = qx.to(tl.float32, bitcast=True)
    sat = qf >= 6.0
    den = (not sat) & (qf < 1.0)
    nrm = not (sat | den)
    DEN_EXP: tl.constexpr = (127 - 1) + (23 - 1) + 1
    DEN_INT: tl.constexpr = DEN_EXP << 23
    den_f: tl.constexpr = tl.cast(DEN_INT, tl.float32, bitcast=True)
    dx = ((qf + den_f).to(tl.uint32, bitcast=True) - DEN_INT).to(tl.uint8)
    # ((1 - 127) << 23) + (1 << 21) - 1 as uint32 (exponent rebias plus the first half of the RNE bias).
    nx = qx + 3240099839
    nx = ((nx + ((qx >> 22) & 1)) >> 22).to(tl.uint8)
    v = tl.full((BT, NG, 32), 0x7, tl.uint8)
    v = tl.where(nrm, nx, v)
    v = tl.where(den, dx, v)
    v = v | (sgn >> 28).to(tl.uint8)
    ev, od = tl.split(tl.reshape(v, (BT, BK // 2, 2)))
    qcols = pid_k * (BK // 2) + tl.arange(0, BK // 2)
    tl.store(q_ptr + rows[:, None] * (K // 2) + qcols[None, :], ev | (od << 4), mask=rmask[:, None])
    scale = tl.reshape((unb.to(tl.int32) + 127).to(tl.uint8), (BT, NG))

    # ---- each row's scales go to the sorted rows of its local slots (padding rows are left unwritten: the GEMMs
    # never use them for a real token's output) ----
    rsel = (tl.arange(0, T_PAD)[None, :] == rows[:, None]).to(tl.int32)  # [BT, T_PAD]
    rpos = tl.sum(rsel[:, :, None] * (pos[None, :, :] + 1), axis=1) - 1  # [BT, TOP_K_PAD]; -1 = not local
    g = pid_k * NG + tl.arange(0, NG)
    off = _scale_byte_off(rpos[:, :, None], g[None, None, :], N_TILES)
    smask = (rpos >= 0)[:, :, None] & rmask[:, None, None]
    tl.store(s_ptr + tl.where(smask, off, 0), tl.broadcast_to(scale[:, None, :], (BT, TOP_K_PAD, NG)), mask=smask)


def sorted_capacity(num_tokens: int, topk: int, num_experts: int, block_m: int) -> int:
    """Rows of ``sorted_ids`` as AITER's moe_sorting allocates them (num_experts = all experts, not only local)."""
    return num_tokens * topk + num_experts * block_m - topk


def moe_mxfp4_sort_quant(x: torch.Tensor, topk_ids: torch.Tensor, local_start: int, num_local: int, num_experts: int,
                         block_m: int = 32, cfg: dict | None = None):
    """x bf16 [T, K], topk_ids int32 [T, topk] (global expert ids, distinct within a token as a top-k is; ids outside
    [local_start, local_start + num_local) are skipped) -> (x_fp4 uint8 [T, K/2], sorted_ids int32 [cap],
    sorted_expert_ids int32 [cap/block_m], num_valid_ids int32 [2] = (padded slots, T), sorted scales uint8 [cap
    rounded up to 32, K/32])."""
    T, K = x.shape
    topk = topk_ids.shape[1]
    assert 0 < T <= MAX_TOKENS and K % 256 == 0 and block_m == 32 and x.stride(1) == 1
    assert x.dtype == torch.bfloat16 and topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
    dev = x.device
    cap = sorted_capacity(T, topk, num_experts, block_m)
    q = torch.empty((T, K // 2), dtype=torch.uint8, device=dev)
    sorted_ids = torch.empty((cap, ), dtype=torch.int32, device=dev)
    sorted_eids = torch.empty((triton.cdiv(cap, block_m), ), dtype=torch.int32, device=dev)
    num_valid = torch.empty((2, ), dtype=torch.int32, device=dev)
    scales = torch.empty((triton.cdiv(cap, 32), K // 32 // 8, 4, 16), dtype=torch.uint32, device=dev)
    t_pad = max(triton.next_power_of_2(T), 2)
    e_block = triton.next_power_of_2(num_local)
    # Every program recomputes the routing layout over all T tokens, so programs stay small (one or two rows) and
    # widen along K as T grows. Measured on MI350X (HIP-graph replay, K = 4096, 32 of 256 experts local).
    cfg = cfg or {}
    bt = cfg.get("BT", 1 if T <= 32 else 2)
    # BK divides K: the grid has K // BK quantization programs and the tiles have no K-axis masks.
    bk = cfg.get("BK", math.gcd(K, 512 if T <= 16 else 1024))
    assert K % bk == 0
    num_warps = cfg.get("num_warps", 2 if T <= 16 else 4 if T <= 32 else 8)
    max_blocks = max(triton.next_power_of_2(triton.cdiv(T, block_m)), 2)
    _moe_mxfp4_sort_quant_kernel[(triton.cdiv(T, bt),
                                  K // bk + 1)](x, x.stride(0), topk_ids, q, scales.view(torch.uint8), sorted_ids,
                                                sorted_eids, num_valid, T, local_start, K=K, TOP_K=topk,
                                                TOP_K_PAD=triton.next_power_of_2(topk), T_PAD=t_pad, E=num_local,
                                                E_BLOCK=e_block, BLOCK_M=block_m, BT=bt, BK=bk, MAX_BLOCKS=max_blocks,
                                                num_warps=num_warps)
    return q, sorted_ids, sorted_eids, num_valid, scales.view(torch.uint8).view(-1, K // 32)
