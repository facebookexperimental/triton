"""Shared-A batched GEMM (BMM) for gfx950 / CDNA4 — ROW-major B (shared-LHS).

Companion to ``amd_bmm.py``. Both files are shared-A; what differs is B's memory
layout: ``amd_bmm.py`` takes COLUMN-major B (``stride_bk == 1``), this file takes
ROW-major B (``stride_bn == 1``), the standard torch.bmm / inductor layout.

LAYOUT:
  * A: shared-A — one (M, K) matrix reused across the whole batch,
    ``a.stride(0) == 0`` (mat1 batch-stride 0). Benchmark against shared-A, not
    distinct-A: rocBLAS reads shared-A once and keeps it L2-resident, so a
    distinct-A comparison flatters TLX.
  * B: (B, K, N) ROW-major (N-contiguous, ``stride_bn == 1``).
  * C: (B, M, N) row-major.

CONFIG: the default paths use num_warps=8 and matrix_instr_nonkdim=32. Large,
deep odd-K shapes use a 4-wave MI16 path with explicitly decomposed accumulator
streams.

Two load paths, selected by K alignment (``K % BLOCK_K == 0`` -> aligned A rows,
no K-tail):
  * aligned -> direct-to-LDS (``buffer_load_to_local``) + swizzled LDS.
  * odd K   -> register path (``tl.load`` -> ``tlx.local_store``), masked K-tail.
    Required because odd K gives 2-byte-aligned rows, where direct-to-LDS is
    illegal on CDNA4.
"""
from dataclasses import dataclass

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BLOCK_N = 256
BLOCK_K = 32
VENDOR_BLOCK_M = 256
VENDOR_BLOCK_K = 64
NUM_XCDS = 8
LARGE_BATCH_GROUP = 64
BMM_262_256_BATCH_GROUP = 256
BMM_448_160_BATCH_GROUP = 256
MIN_GROUPED_OUTPUT_TILES = 5
NB = 3

_RESIDENT_OPERAND_AUTO = tl.constexpr(0)
_RESIDENT_OPERAND_A = tl.constexpr(1)
_RESIDENT_OPERAND_B = tl.constexpr(2)


@dataclass(frozen=True)
class _RegisterStagedKernelSpec:
    block_m: int
    block_n: int
    a_row_tile_sizes: tuple[int, ...]
    b_col_tile_sizes: tuple[int, ...]
    num_stages: int
    instr_shape: tuple[int, int, int]
    warps_per_cta: tuple[int, int]
    loop_unroll: int
    swizzle_b_local: bool
    global_load_b_cg: bool
    global_tail_load_before_last_dot: bool
    local_load_b_before_global_prefetch: bool
    resident_operand_policy: tl.constexpr


_MT64X256_MI32_KERNEL_SPEC = _RegisterStagedKernelSpec(
    block_m=64,
    block_n=256,
    a_row_tile_sizes=(64, ),
    b_col_tile_sizes=(256, ),
    num_stages=2,
    instr_shape=(32, 32, 16),
    warps_per_cta=(1, 4),
    loop_unroll=2,
    swizzle_b_local=False,
    global_load_b_cg=True,
    global_tail_load_before_last_dot=False,
    local_load_b_before_global_prefetch=True,
    resident_operand_policy=_RESIDENT_OPERAND_AUTO,
)
# These two shapes keep their measured resident-A schedules. Resident B is
# valid and lowers operand VGPR pressure, but the evaluated load placements and
# small scheduler-cover search regress latency; the policy remains an explicit,
# bounded tuning dimension for future specializations.
_MT144X256_MI16_KERNEL_SPEC = _RegisterStagedKernelSpec(
    block_m=144,
    block_n=256,
    a_row_tile_sizes=(128, 16),
    b_col_tile_sizes=(256, ),
    num_stages=2,
    instr_shape=(16, 16, 32),
    warps_per_cta=(1, 4),
    loop_unroll=1,
    swizzle_b_local=True,
    global_load_b_cg=False,
    global_tail_load_before_last_dot=False,
    local_load_b_before_global_prefetch=True,
    resident_operand_policy=_RESIDENT_OPERAND_A,
)
_MT224X160_MI16_KERNEL_SPEC = _RegisterStagedKernelSpec(
    block_m=224,
    block_n=160,
    a_row_tile_sizes=(32, 32, 32, 32, 32, 32, 32),
    b_col_tile_sizes=(32, 32, 32, 32, 32),
    num_stages=2,
    instr_shape=(16, 16, 32),
    warps_per_cta=(2, 2),
    loop_unroll=1,
    swizzle_b_local=False,
    global_load_b_cg=False,
    global_tail_load_before_last_dot=True,
    local_load_b_before_global_prefetch=False,
    resident_operand_policy=_RESIDENT_OPERAND_A,
)

# Compiler policy stays separate from data decomposition:
# (MFMA_PER_DWORDX4, DISABLE_HIGH_RP_RESCHEDULE).
_REGISTER_STAGED_SCHEDULE_SPEC = (4, False)
_MT224X160_REGISTER_STAGED_SCHEDULE_SPEC = (4, True)

# Coalesced [128, 128] store layout: eight contiguous fp16 values per lane.
_C4_128 = tlx.layout(shape=((16, 16), (8, 8)), stride=((8, 128), (1, 2048)))


def _swz(shape, cd):

    def basis(d, i):
        return [1 << i, 0] if d == 0 else [0, 1 << i]

    fd = 1 - cd
    cb = int(shape[cd]).bit_length() - 1
    fb = int(shape[fd]).bit_length() - 1
    return ([basis(cd, i) for i in range(cb)] + [basis(fd, i)
                                                 for i in range(4, fb)] + [basis(fd, i) for i in range(min(4, fb))])


@triton.jit
def _chip(pid, nt, nx: tl.constexpr, cs: tl.constexpr):
    """L2 XCD-chunk remap: keep a batch's MN-tiles on one XCD (B stays hot in L2)."""
    al = (nt // (nx * cs)) * (nx * cs)
    if pid >= al:
        return pid
    x = pid % nx
    lp = pid // nx
    return (lp // cs) * nx * cs + x * cs + (lp % cs)


@triton.jit
def _bmm_direct(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, AB: tl.constexpr, BB: tl.constexpr, NUM_XCDS: tl.constexpr,
                GMN: tl.constexpr, NT: tl.constexpr, NB: tl.constexpr):
    """Aligned rows, no K-tail (K % BLOCK_K == 0): direct-to-LDS + swizzled LDS."""
    npn = tl.cdiv(N, BN)
    pidf = _chip(tl.program_id(0), NT, NUM_XCDS, GMN)
    bid = pidf // GMN
    pid = pidf % GMN
    pm = pid // npn
    pn = pid % npn
    ash: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], AB, [BM, BK])
    bsh: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], BB, [BK, BN])
    sA = tlx.local_alloc((BM, BK), tlx.dtype_of(a_ptr), NB, layout=ash)
    sB = tlx.local_alloc((BK, BN), tlx.dtype_of(b_ptr), NB, layout=bsh)
    om = (pm * BM + tl.arange(0, BM)) % M
    on = (pn * BN + tl.arange(0, BN)) % N
    ok = tl.arange(0, BK)
    a_ptr = a_ptr + bid.to(tl.int64) * sab
    b_ptr = b_ptr + bid.to(tl.int64) * sbb
    ao = om[:, None] * sam
    bo = on[None, :] * sbn
    KI = tl.cdiv(K, BK)
    for i in tl.range(0, NB, loop_unroll_factor=NB):
        kk = i * BK
        tlx.buffer_load_to_local(tlx.local_view(sA, i), a_ptr, ao + (kk + ok[None, :]) * sak)
        tlx.buffer_load_to_local(tlx.local_view(sB, i), b_ptr, (kk + ok[:, None]) * sbk + bo)
        tlx.async_load_commit_group()
    tlx.async_load_wait_group(NB - 2)
    a = tlx.local_load(tlx.local_view(sA, 0))
    b = tlx.local_load(tlx.local_view(sB, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, KI - NB):
        cur = (k + 1) % NB
        pf = k % NB
        kp = (k + NB) * BK
        acc = tl.dot(a, b, acc)
        tlx.buffer_load_to_local(tlx.local_view(sA, pf), a_ptr, ao + (kp + ok[None, :]) * sak)
        tlx.buffer_load_to_local(tlx.local_view(sB, pf), b_ptr, (kp + ok[:, None]) * sbk + bo)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(NB - 2)
        a = tlx.local_load(tlx.local_view(sA, cur))
        b = tlx.local_load(tlx.local_view(sB, cur))
    acc = tl.dot(a, b, acc)
    tlx.async_load_wait_group(0)
    for i in tl.range(0, NB - 1, loop_unroll_factor=NB - 1):
        bf = (KI - (NB - 1) + i) % NB
        acc = tl.dot(tlx.local_load(tlx.local_view(sA, bf)), tlx.local_load(tlx.local_view(sB, bf)), acc)
    et = c_ptr.dtype.element_ty
    cb = c_ptr + bid.to(tl.int64) * scb
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    tl.store(cb + scm * rm[:, None] + scn * rn[None, :], acc.to(et), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _bmm_register(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr, NUM_XCDS: tl.constexpr, GMN: tl.constexpr, NT: tl.constexpr,
                  NB: tl.constexpr):
    """Odd / unaligned K: register path (tl.load -> local_store), masked K-tail."""
    npn = tl.cdiv(N, BN)
    pidf = _chip(tl.program_id(0), NT, NUM_XCDS, GMN)
    bid = pidf // GMN
    pid = pidf % GMN
    pm = pid // npn
    pn = pid % npn
    sA = tlx.local_alloc((BM, BK), tlx.dtype_of(a_ptr), NB)
    sB = tlx.local_alloc((BK, BN), tlx.dtype_of(b_ptr), NB)
    om = (pm * BM + tl.arange(0, BM)) % M
    on = (pn * BN + tl.arange(0, BN)) % N
    ok = tl.arange(0, BK)
    a_ptr = a_ptr + bid.to(tl.int64) * sab
    b_ptr = b_ptr + bid.to(tl.int64) * sbb
    ao = om[:, None] * sam
    bo = on[None, :] * sbn
    KI = tl.cdiv(K, BK)
    for i in tl.range(0, NB, loop_unroll_factor=NB):
        kk = i * BK
        km = (kk + ok) < K
        ar = tl.load(a_ptr + ao + (kk + ok[None, :]) * sak, mask=km[None, :], other=0.0)
        br = tl.load(b_ptr + (kk + ok[:, None]) * sbk + bo, mask=km[:, None], other=0.0)
        tlx.local_store(tlx.local_view(sA, i), ar)
        tlx.local_store(tlx.local_view(sB, i), br)
    tl.debug_barrier()
    a = tlx.local_load(tlx.local_view(sA, 0))
    b = tlx.local_load(tlx.local_view(sB, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, KI - NB):
        cur = (k + 1) % NB
        pf = k % NB
        kp = (k + NB) * BK
        acc = tl.dot(a, b, acc)
        km = (kp + ok) < K
        ar = tl.load(a_ptr + ao + (kp + ok[None, :]) * sak, mask=km[None, :], other=0.0)
        br = tl.load(b_ptr + (kp + ok[:, None]) * sbk + bo, mask=km[:, None], other=0.0)
        tlx.local_store(tlx.local_view(sA, pf), ar)
        tlx.local_store(tlx.local_view(sB, pf), br)
        tl.debug_barrier()
        a = tlx.local_load(tlx.local_view(sA, cur))
        b = tlx.local_load(tlx.local_view(sB, cur))
    acc = tl.dot(a, b, acc)
    for i in tl.range(0, NB - 1, loop_unroll_factor=NB - 1):
        bf = (KI - (NB - 1) + i) % NB
        acc = tl.dot(tlx.local_load(tlx.local_view(sA, bf)), tlx.local_load(tlx.local_view(sB, bf)), acc)
    et = c_ptr.dtype.element_ty
    cb = c_ptr + bid.to(tl.int64) * scb
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    tl.store(cb + scm * rm[:, None] + scn * rn[None, :], acc.to(et), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _bmm_mi16_quad(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   NUM_XCDS: tl.constexpr, GMN: tl.constexpr, NT: tl.constexpr,
                   B_BASES: tl.constexpr):
    """2x2 operand decomposition: defer A-high/B-high LDS reads under MFMA."""
    HM: tl.constexpr = BM // 2
    HN: tl.constexpr = BN // 2
    tl.static_assert(HM == 128 and HN == 128,
                     "_C4_128 requires 128x128 accumulator quadrants")
    npn = tl.cdiv(N, BN)
    pidf = _chip(tl.program_id(0), NT, NUM_XCDS, GMN)
    bid = pidf // GMN
    pid = pidf % GMN
    pm = pid // npn
    pn = pid % npn
    b_sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], B_BASES, [BK, HN])
    buf_a_lo = tlx.local_alloc((HM, BK), tlx.dtype_of(a_ptr), 2)
    buf_a_hi = tlx.local_alloc((HM, BK), tlx.dtype_of(a_ptr), 2)
    buf_b_lo = tlx.local_alloc((BK, HN), tlx.dtype_of(b_ptr), 2, layout=b_sh)
    buf_b_hi = tlx.local_alloc((BK, HN), tlx.dtype_of(b_ptr), 2, layout=b_sh)
    om_lo = (pm * BM + tl.arange(0, HM)) % M
    om_hi = (pm * BM + HM + tl.arange(0, HM)) % M
    on_lo = (pn * BN + tl.arange(0, HN)) % N
    on_hi = (pn * BN + HN + tl.arange(0, HN)) % N
    ok = tl.arange(0, BK)
    a_ptr += bid.to(tl.int64) * sab
    b_ptr += bid.to(tl.int64) * sbb
    ao_lo = om_lo[:, None] * sam
    ao_hi = om_hi[:, None] * sam
    bo_lo = on_lo[None, :] * sbn
    bo_hi = on_hi[None, :] * sbn
    nf = K // BK
    tlx.local_store(tlx.local_view(buf_a_lo, 0), tl.load(a_ptr + ao_lo + ok[None, :] * sak))
    tlx.local_store(tlx.local_view(buf_a_hi, 0), tl.load(a_ptr + ao_hi + ok[None, :] * sak))
    tlx.local_store(tlx.local_view(buf_b_lo, 0), tl.load(b_ptr + ok[:, None] * sbk + bo_lo))
    tlx.local_store(tlx.local_view(buf_b_hi, 0), tl.load(b_ptr + ok[:, None] * sbk + bo_hi))
    tl.debug_barrier()
    c00 = tl.zeros((HM, HN), dtype=tl.float32)
    c10 = tl.zeros((HM, HN), dtype=tl.float32)
    c01 = tl.zeros((HM, HN), dtype=tl.float32)
    c11 = tl.zeros((HM, HN), dtype=tl.float32)
    for k in tl.range(0, nf - 1):
        cur = k % 2
        nxt = (k + 1) % 2
        kp = (k + 1) * BK
        a_lo = tlx.local_load(tlx.local_view(buf_a_lo, cur))
        b_lo = tlx.local_load(tlx.local_view(buf_b_lo, cur))
        next_a_lo = tl.load(a_ptr + ao_lo + (kp + ok[None, :]) * sak)
        tlx.buffer_load_to_local(tlx.local_view(buf_b_lo, nxt), b_ptr,
                                 (kp + ok[:, None]) * sbk + bo_lo)
        c00 = tl.dot(a_lo, b_lo, c00)
        a_hi = tlx.local_load(tlx.local_view(buf_a_hi, cur))
        next_a_hi = tl.load(a_ptr + ao_hi + (kp + ok[None, :]) * sak)
        c10 = tl.dot(a_hi, b_lo, c10)
        b_hi = tlx.local_load(tlx.local_view(buf_b_hi, cur))
        tlx.buffer_load_to_local(tlx.local_view(buf_b_hi, nxt), b_ptr,
                                 (kp + ok[:, None]) * sbk + bo_hi)
        c01 = tl.dot(a_lo, b_hi, c01)
        c11 = tl.dot(a_hi, b_hi, c11)
        tlx.local_store(tlx.local_view(buf_a_lo, nxt), next_a_lo)
        tlx.local_store(tlx.local_view(buf_a_hi, nxt), next_a_hi)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(0)
        tl.debug_barrier()
    last = (nf - 1) % 2
    a_lo = tlx.local_load(tlx.local_view(buf_a_lo, last))
    a_hi = tlx.local_load(tlx.local_view(buf_a_hi, last))
    b_lo = tlx.local_load(tlx.local_view(buf_b_lo, last))
    b_hi = tlx.local_load(tlx.local_view(buf_b_hi, last))
    c00 = tl.dot(a_lo, b_lo, c00)
    c01 = tl.dot(a_lo, b_hi, c01)
    c10 = tl.dot(a_hi, b_lo, c10)
    c11 = tl.dot(a_hi, b_hi, c11)
    # `k` above is a full-tile ordinal; `kt` is an absolute K-element offset
    # because this loop starts at the first incomplete BK tile.
    for kt in tl.range(nf * BK, K, BK):
        km = (kt + ok) < K
        ta_lo = tl.load(a_ptr + ao_lo + (kt + ok[None, :]) * sak, mask=km[None, :], other=0.0)
        ta_hi = tl.load(a_ptr + ao_hi + (kt + ok[None, :]) * sak, mask=km[None, :], other=0.0)
        tb_lo = tl.load(b_ptr + (kt + ok[:, None]) * sbk + bo_lo, mask=km[:, None], other=0.0)
        tb_hi = tl.load(b_ptr + (kt + ok[:, None]) * sbk + bo_hi, mask=km[:, None], other=0.0)
        c00 = tl.dot(ta_lo, tb_lo, c00)
        c01 = tl.dot(ta_lo, tb_hi, c01)
        c10 = tl.dot(ta_hi, tb_lo, c10)
        c11 = tl.dot(ta_hi, tb_hi, c11)
    et = c_ptr.dtype.element_ty
    cb = c_ptr + bid.to(tl.int64) * scb
    rm_lo = pm * BM + tl.arange(0, HM)
    rm_hi = rm_lo + HM
    rn_lo = pn * BN + tl.arange(0, HN)
    rn_hi = rn_lo + HN
    tl.store(cb + scm * rm_lo[:, None] + scn * rn_lo[None, :], tlx.require_layout(c00.to(et), _C4_128),
             mask=(rm_lo[:, None] < M) & (rn_lo[None, :] < N))
    tl.store(cb + scm * rm_lo[:, None] + scn * rn_hi[None, :], tlx.require_layout(c01.to(et), _C4_128),
             mask=(rm_lo[:, None] < M) & (rn_hi[None, :] < N))
    tl.store(cb + scm * rm_hi[:, None] + scn * rn_lo[None, :], tlx.require_layout(c10.to(et), _C4_128),
             mask=(rm_hi[:, None] < M) & (rn_lo[None, :] < N))
    tl.store(cb + scm * rm_hi[:, None] + scn * rn_hi[None, :], tlx.require_layout(c11.to(et), _C4_128),
             mask=(rm_hi[:, None] < M) & (rn_hi[None, :] < N))


@tl.core.builtin
def _load_all_a_tiles_from_local(a_local_tiles, stage,
                                 a_tile_count: tl.constexpr,
                                 a_layout: tl.constexpr, _semantic=None):
    """Load every A operand tile from LDS into its dot layout."""
    a_tile_count = tl.core._unwrap_if_constexpr(a_tile_count)
    a_layout = tl.core._unwrap_if_constexpr(a_layout)
    a_local_tiles = list(a_local_tiles)
    values = []
    for mi in range(a_tile_count):
        view = tlx.local_view(a_local_tiles[mi], stage, _semantic=_semantic)
        a_value = tlx.local_load(view, _semantic=_semantic)
        a_value = tlx.require_layout(
            a_value, a_layout, pin=False, _semantic=_semantic
        )
        values.append(a_value)
    return tl.tuple(values)


@tl.core.builtin
def _load_all_b_tiles_from_local(b_local_tiles, stage,
                                 b_tile_count: tl.constexpr,
                                 b_layout: tl.constexpr, _semantic=None):
    """Load every B operand tile from LDS into its dot layout."""
    b_tile_count = tl.core._unwrap_if_constexpr(b_tile_count)
    b_layout = tl.core._unwrap_if_constexpr(b_layout)
    b_local_tiles = list(b_local_tiles)
    values = []
    for nj in range(b_tile_count):
        view = tlx.local_view(b_local_tiles[nj], stage, _semantic=_semantic)
        b_value = tlx.local_load(view, _semantic=_semantic)
        b_value = tlx.require_layout(
            b_value, b_layout, pin=False, _semantic=_semantic
        )
        values.append(b_value)
    return tl.tuple(values)


@tl.core.builtin
def _dot_preloaded_a_and_b_tiles(a_dot_operands, b_dot_operands, acc,
                                 a_tile_count: tl.constexpr,
                                 b_tile_count: tl.constexpr,
                                 _semantic=None):
    """Expand the dot grid when both operand families are preloaded."""
    a_tile_count = tl.core._unwrap_if_constexpr(a_tile_count)
    b_tile_count = tl.core._unwrap_if_constexpr(b_tile_count)
    values = list(acc)
    a_dot_operands = list(a_dot_operands)
    b_dot_operands = list(b_dot_operands)
    for nj in range(b_tile_count):
        for mi in range(a_tile_count):
            index = mi * b_tile_count + nj
            values[index] = tl.dot(
                a_dot_operands[mi], b_dot_operands[nj], values[index],
                _semantic=_semantic
            )
    return tl.tuple(values)


@tl.core.builtin
def _load_each_b_tile_and_dot_with_preloaded_a(
    a_dot_operands,
    b_local_tiles,
    stage,
    acc,
    a_tile_count: tl.constexpr,
    b_tile_count: tl.constexpr,
    b_layout: tl.constexpr,
    _semantic=None,
):
    """Stream B tiles from LDS and dot each with every preloaded A tile."""
    a_tile_count = tl.core._unwrap_if_constexpr(a_tile_count)
    b_tile_count = tl.core._unwrap_if_constexpr(b_tile_count)
    b_layout = tl.core._unwrap_if_constexpr(b_layout)
    values = list(acc)
    a_dot_operands = list(a_dot_operands)
    b_local_tiles = list(b_local_tiles)
    for nj in range(b_tile_count):
        view = tlx.local_view(b_local_tiles[nj], stage, _semantic=_semantic)
        b_value = tlx.local_load(view, _semantic=_semantic)
        b_value = tlx.require_layout(
            b_value, b_layout, pin=False, _semantic=_semantic
        )
        for mi in range(a_tile_count):
            index = mi * b_tile_count + nj
            values[index] = tl.dot(
                a_dot_operands[mi], b_value, values[index],
                _semantic=_semantic
            )
    return tl.tuple(values)


@tl.core.builtin
def _load_each_a_tile_and_dot_with_preloaded_b(
    a_local_tiles,
    b_dot_operands,
    stage,
    acc,
    a_tile_count: tl.constexpr,
    b_tile_count: tl.constexpr,
    a_layout: tl.constexpr,
    _semantic=None,
):
    """Stream A tiles from LDS and dot each with every preloaded B tile."""
    a_tile_count = tl.core._unwrap_if_constexpr(a_tile_count)
    b_tile_count = tl.core._unwrap_if_constexpr(b_tile_count)
    a_layout = tl.core._unwrap_if_constexpr(a_layout)
    values = list(acc)
    a_local_tiles = list(a_local_tiles)
    b_dot_operands = list(b_dot_operands)
    for mi in range(a_tile_count):
        view = tlx.local_view(a_local_tiles[mi], stage, _semantic=_semantic)
        a_value = tlx.local_load(view, _semantic=_semantic)
        a_value = tlx.require_layout(
            a_value, a_layout, pin=False, _semantic=_semantic
        )
        for nj in range(b_tile_count):
            index = mi * b_tile_count + nj
            values[index] = tl.dot(
                a_value, b_dot_operands[nj], values[index],
                _semantic=_semantic
            )
    return tl.tuple(values)


@tl.core.builtin
def _load_preloaded_operand_tiles(
    a_local_tiles,
    b_local_tiles,
    stage,
    a_tile_count: tl.constexpr,
    b_tile_count: tl.constexpr,
    a_layout: tl.constexpr,
    b_layout: tl.constexpr,
    preload_a: tl.constexpr,
    _semantic=None,
):
    """Load the compile-time-selected resident operand family from LDS."""
    preload_a = tl.core._unwrap_if_constexpr(preload_a)
    if preload_a:
        return _load_all_a_tiles_from_local(
            a_local_tiles,
            stage,
            a_tile_count,
            a_layout,
            _semantic=_semantic,
        )
    return _load_all_b_tiles_from_local(
        b_local_tiles,
        stage,
        b_tile_count,
        b_layout,
        _semantic=_semantic,
    )


@tl.core.builtin
def _load_each_streamed_operand_and_dot(
    a_local_tiles,
    b_local_tiles,
    preloaded_operands,
    stage,
    acc,
    a_tile_count: tl.constexpr,
    b_tile_count: tl.constexpr,
    a_layout: tl.constexpr,
    b_layout: tl.constexpr,
    preload_a: tl.constexpr,
    _semantic=None,
):
    """Stream the nonresident operand family and update the full dot grid."""
    preload_a = tl.core._unwrap_if_constexpr(preload_a)
    if preload_a:
        return _load_each_b_tile_and_dot_with_preloaded_a(
            preloaded_operands,
            b_local_tiles,
            stage,
            acc,
            a_tile_count,
            b_tile_count,
            b_layout,
            _semantic=_semantic,
        )
    return _load_each_a_tile_and_dot_with_preloaded_b(
        a_local_tiles,
        preloaded_operands,
        stage,
        acc,
        a_tile_count,
        b_tile_count,
        a_layout,
        _semantic=_semantic,
    )


@tl.core.builtin
def _spec_length(values: tl.constexpr, _semantic=None):
    values = tl.core._unwrap_if_constexpr(values)
    return tl.constexpr(len(values))


@tl.core.builtin
def _spec_item(values: tl.constexpr, index: tl.constexpr, _semantic=None):
    values = tl.core._unwrap_if_constexpr(values)
    index = tl.core._unwrap_if_constexpr(index)
    return tl.constexpr(values[index])


@tl.core.builtin
def _tile_extent(index: tl.constexpr, tile_sizes: tl.constexpr,
                 _semantic=None):
    """Select one heterogeneous tile extent from the compile-time spec."""
    index = tl.core._unwrap_if_constexpr(index)
    tile_sizes = tl.core._unwrap_if_constexpr(tile_sizes)
    return tl.constexpr(tile_sizes[index])


@tl.core.builtin
def _tile_start(index: tl.constexpr, tile_sizes: tl.constexpr,
                _semantic=None):
    """Return the compile-time prefix sum preceding one operand tile."""
    index = tl.core._unwrap_if_constexpr(index)
    tile_sizes = tl.core._unwrap_if_constexpr(tile_sizes)
    return tl.constexpr(sum(tile_sizes[:index]))


@triton.jit
def _make_output_tile_coordinates(
    m_block,
    n_block,
    kernel_spec: tl.constexpr,
):
    """Build the logical C row/column coordinates for one macro tile."""
    block_m: tl.constexpr = tl.constexpr(kernel_spec.block_m)
    block_n: tl.constexpr = tl.constexpr(kernel_spec.block_n)
    a_tile_count: tl.constexpr = _spec_length(kernel_spec.a_row_tile_sizes)
    b_tile_count: tl.constexpr = _spec_length(kernel_spec.b_col_tile_sizes)
    output_rows = tl.tuple([])
    for mi in tl.static_range(a_tile_count):
        output_rows += tl.tuple([
            m_block * block_m
            + _tile_start(mi, kernel_spec.a_row_tile_sizes)
            + tl.arange(0, _tile_extent(mi, kernel_spec.a_row_tile_sizes))
        ])
    output_cols = tl.tuple([])
    for nj in tl.static_range(b_tile_count):
        output_cols += tl.tuple([
            n_block * block_n
            + _tile_start(nj, kernel_spec.b_col_tile_sizes)
            + tl.arange(0, _tile_extent(nj, kernel_spec.b_col_tile_sizes))
        ])
    return tl.tuple([output_rows, output_cols])


@triton.jit
def _make_global_tile_offsets(
    output_rows,
    output_cols,
    m,
    n,
    stride_am,
    stride_bn,
    even_m: tl.constexpr,
    even_n: tl.constexpr,
    kernel_spec: tl.constexpr,
):
    """Build wrapped A/B offsets for full-width unmasked global loads."""
    a_tile_count: tl.constexpr = _spec_length(kernel_spec.a_row_tile_sizes)
    b_tile_count: tl.constexpr = _spec_length(kernel_spec.b_col_tile_sizes)
    a_global_offsets = tl.tuple([])
    for mi in tl.static_range(a_tile_count):
        if even_m:
            a_global_rows = output_rows[mi]
        else:
            a_global_rows = tl.where(
                output_rows[mi] < m,
                output_rows[mi],
                output_rows[mi] - m,
            )
        a_global_offsets += tl.tuple([
            a_global_rows[:, None] * stride_am
        ])
    b_global_offsets = tl.tuple([])
    for nj in tl.static_range(b_tile_count):
        if even_n:
            b_global_cols = output_cols[nj]
        else:
            b_global_cols = tl.where(
                output_cols[nj] < n,
                output_cols[nj],
                output_cols[nj] - n,
            )
        b_global_offsets += tl.tuple([
            b_global_cols[None, :] * stride_bn
        ])
    return tl.tuple([a_global_offsets, b_global_offsets])


@triton.jit
def _local_alloc_pipeline(a_ptr, b_ptr, block_k: tl.constexpr,
                          kernel_spec: tl.constexpr):
    """Allocate every A/B tile in the explicit multi-stage LDS pipeline."""
    num_stages: tl.constexpr = tl.constexpr(kernel_spec.num_stages)
    swizzle_b_local: tl.constexpr = tl.constexpr(kernel_spec.swizzle_b_local)
    a_tile_count: tl.constexpr = _spec_length(kernel_spec.a_row_tile_sizes)
    b_tile_count: tl.constexpr = _spec_length(kernel_spec.b_col_tile_sizes)
    a_local_tiles = tl.tuple([])
    for mi in tl.static_range(a_tile_count):
        a_local_tiles += tl.tuple([
            tlx.local_alloc(
                (_tile_extent(mi, kernel_spec.a_row_tile_sizes), block_k),
                tlx.dtype_of(a_ptr),
                num_stages,
            )
        ])
    b_local_tiles = tl.tuple([])
    if swizzle_b_local:
        b_shared_layout: tl.constexpr = tlx.swizzled_layout(4, 3, 5)
        for nj in tl.static_range(b_tile_count):
            b_local_tiles += tl.tuple([
                tlx.local_alloc(
                    (block_k, _tile_extent(nj, kernel_spec.b_col_tile_sizes)),
                    tlx.dtype_of(b_ptr),
                    num_stages,
                    layout=b_shared_layout,
                )
            ])
    else:
        for nj in tl.static_range(b_tile_count):
            b_local_tiles += tl.tuple([
                tlx.local_alloc(
                    (block_k, _tile_extent(nj, kernel_spec.b_col_tile_sizes)),
                    tlx.dtype_of(b_ptr),
                    num_stages,
                )
            ])
    return tl.tuple([a_local_tiles, b_local_tiles])


@triton.jit
def _bmm_register_staged(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sab,
    sam,
    sak,
    sbb,
    sbk,
    sbn,
    scb,
    scm,
    scn,
    KERNEL_SPEC: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    HAS_K_TAIL: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BATCH_GROUP: tl.constexpr,
    GMN: tl.constexpr,
    NT: tl.constexpr,
):
    """Register-staged K32 BMM driven by compile-time operand tiles."""
    NUM_STAGES: tl.constexpr = tl.constexpr(KERNEL_SPEC.num_stages)
    LOOP_UNROLL: tl.constexpr = tl.constexpr(KERNEL_SPEC.loop_unroll)
    GLOBAL_LOAD_B_CG: tl.constexpr = tl.constexpr(KERNEL_SPEC.global_load_b_cg)
    GLOBAL_TAIL_LOAD_BEFORE_LAST_DOT: tl.constexpr = tl.constexpr(
        KERNEL_SPEC.global_tail_load_before_last_dot
    )
    LOCAL_LOAD_B_BEFORE_GLOBAL_PREFETCH: tl.constexpr = tl.constexpr(
        KERNEL_SPEC.local_load_b_before_global_prefetch
    )
    A_TILE_COUNT: tl.constexpr = _spec_length(KERNEL_SPEC.a_row_tile_sizes)
    B_TILE_COUNT: tl.constexpr = _spec_length(KERNEL_SPEC.b_col_tile_sizes)
    INSTR_M: tl.constexpr = _spec_item(KERNEL_SPEC.instr_shape, 0)
    INSTR_N: tl.constexpr = _spec_item(KERNEL_SPEC.instr_shape, 1)
    INSTR_K: tl.constexpr = _spec_item(KERNEL_SPEC.instr_shape, 2)
    WARPS_M: tl.constexpr = _spec_item(KERNEL_SPEC.warps_per_cta, 0)
    WARPS_N: tl.constexpr = _spec_item(KERNEL_SPEC.warps_per_cta, 1)
    # Keep the smaller per-wave operand family resident and stream the larger
    # family one tile at a time. A is partitioned across WARPS_M and B across
    # WARPS_N; their common K32 depth cancels from this VGPR-footprint estimate.
    # Cross multiplication keeps the comparison integral:
    #   BLOCK_M / WARPS_M <= BLOCK_N / WARPS_N.
    RESIDENT_OPERAND_POLICY: tl.constexpr = tl.constexpr(
        KERNEL_SPEC.resident_operand_policy
    )
    PRELOAD_A_OPERANDS: tl.constexpr = tl.constexpr(
        RESIDENT_OPERAND_POLICY == _RESIDENT_OPERAND_A
        or (
            RESIDENT_OPERAND_POLICY == _RESIDENT_OPERAND_AUTO
            and KERNEL_SPEC.block_m * WARPS_N
            <= KERNEL_SPEC.block_n * WARPS_M
        )
    )
    BLOCK_K: tl.constexpr = 32

    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[INSTR_M, INSTR_N, INSTR_K],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)

    # Program mapping: recover one batch and one output macro tile from the
    # XCD-friendly one-dimensional launch order.
    remapped_program = _chip(tl.program_id(0), NT, BATCH_GROUP, GMN)
    batch_id = remapped_program // GMN
    tile_id = remapped_program % GMN
    m_block = tile_id // N_BLOCKS
    n_block = tile_id % N_BLOCKS
    rk = tl.arange(0, BLOCK_K)
    a_ptr += batch_id.to(tl.int64) * sab
    b_ptr += batch_id.to(tl.int64) * sbb

    # Tile setup: build logical output coordinates, wrapped global input
    # offsets, and the heterogeneous A/B LDS images described by KERNEL_SPEC.
    output_rows, output_cols = _make_output_tile_coordinates(
        m_block, n_block, KERNEL_SPEC
    )
    a_global_offsets, b_global_offsets = _make_global_tile_offsets(
        output_rows, output_cols, M, N, sam, sbn,
        EVEN_M, EVEN_N, KERNEL_SPEC,
    )
    a_local_tiles, b_local_tiles = _local_alloc_pipeline(
        a_ptr, b_ptr, BLOCK_K, KERNEL_SPEC
    )

    # Prologue: globally prefetch K0, publish it to LDS stage 0, then make the
    # complete A/B stage visible to every wave before the first iteration.
    for mi in tl.static_range(A_TILE_COUNT):
        first_a_global = tl.load(
            a_ptr + a_global_offsets[mi] + rk[None, :] * sak
        )
        tlx.local_store(
            tlx.local_view(a_local_tiles[mi], 0), first_a_global
        )
    for nj in tl.static_range(B_TILE_COUNT):
        b_global_ptrs = b_ptr + rk[:, None] * sbk + b_global_offsets[nj]
        if GLOBAL_LOAD_B_CG:
            first_b_global = tl.load(
                b_global_ptrs, cache_modifier=".cg"
            )
        else:
            first_b_global = tl.load(b_global_ptrs)
        tlx.local_store(
            tlx.local_view(b_local_tiles[nj], 0), first_b_global
        )
    tl.debug_barrier()
    # Keep accumulator construction in the same JIT scope as the K loop. A
    # helper return would erase the native #mma encoding from the tuple type.
    acc = tl.tuple([])
    for mi in tl.static_range(A_TILE_COUNT):
        for nj in tl.static_range(B_TILE_COUNT):
            acc += tl.tuple([
                tlx.zeros(
                    (
                        _tile_extent(mi, KERNEL_SPEC.a_row_tile_sizes),
                        _tile_extent(nj, KERNEL_SPEC.b_col_tile_sizes),
                    ),
                    tl.float32,
                    layout=mma,
                )
            ])

    # Steady state: consume LDS K(t), globally prefetch K(t+1), execute the
    # current dot grid, then publish K(t+1) into the alternate LDS stage.
    full_tiles = K // BLOCK_K
    for k in tl.range(
        0, full_tiles - 1, loop_unroll_factor=LOOP_UNROLL
    ):
        current_stage = k % NUM_STAGES
        next_stage = (k + 1) % NUM_STAGES
        next_k_offset = (k + 1) * BLOCK_K
        if PRELOAD_A_OPERANDS or LOCAL_LOAD_B_BEFORE_GLOBAL_PREFETCH:
            preloaded_operands = _load_preloaded_operand_tiles(
                a_local_tiles, b_local_tiles, current_stage,
                A_TILE_COUNT, B_TILE_COUNT, dot0, dot1,
                PRELOAD_A_OPERANDS,
            )
        if PRELOAD_A_OPERANDS and LOCAL_LOAD_B_BEFORE_GLOBAL_PREFETCH:
            b_dot_operands = _load_all_b_tiles_from_local(
                b_local_tiles, current_stage, B_TILE_COUNT, dot1
            )
        a_global_prefetch = tl.tuple([])
        for mi in tl.static_range(A_TILE_COUNT):
            a_global_prefetch += tl.tuple([tl.load(
                a_ptr
                + a_global_offsets[mi]
                + (next_k_offset + rk[None, :]) * sak
            )])
        b_global_prefetch = tl.tuple([])
        for nj in tl.static_range(B_TILE_COUNT):
            b_global_ptrs = (
                b_ptr
                + (next_k_offset + rk[:, None]) * sbk
                + b_global_offsets[nj]
            )
            if GLOBAL_LOAD_B_CG:
                b_global_prefetch += tl.tuple([
                    tl.load(b_global_ptrs, cache_modifier=".cg")
                ])
            else:
                b_global_prefetch += tl.tuple([tl.load(b_global_ptrs)])
        if PRELOAD_A_OPERANDS:
            if LOCAL_LOAD_B_BEFORE_GLOBAL_PREFETCH:
                acc = _dot_preloaded_a_and_b_tiles(
                    preloaded_operands, b_dot_operands, acc,
                    A_TILE_COUNT, B_TILE_COUNT,
                )
            else:
                acc = _load_each_streamed_operand_and_dot(
                    a_local_tiles, b_local_tiles, preloaded_operands,
                    current_stage, acc, A_TILE_COUNT, B_TILE_COUNT,
                    dot0, dot1, PRELOAD_A_OPERANDS,
                )
        else:
            if not LOCAL_LOAD_B_BEFORE_GLOBAL_PREFETCH:
                preloaded_operands = _load_preloaded_operand_tiles(
                    a_local_tiles, b_local_tiles, current_stage,
                    A_TILE_COUNT, B_TILE_COUNT, dot0, dot1,
                    PRELOAD_A_OPERANDS,
                )
            acc = _load_each_streamed_operand_and_dot(
                a_local_tiles, b_local_tiles, preloaded_operands,
                current_stage, acc, A_TILE_COUNT, B_TILE_COUNT,
                dot0, dot1, PRELOAD_A_OPERANDS,
            )
        for mi in tl.static_range(A_TILE_COUNT):
            tlx.local_store(
                tlx.local_view(a_local_tiles[mi], next_stage),
                a_global_prefetch[mi],
            )
        for nj in tl.static_range(B_TILE_COUNT):
            tlx.local_store(
                tlx.local_view(b_local_tiles[nj], next_stage),
                b_global_prefetch[nj],
            )
        tl.debug_barrier()

    # Drain: consume the final complete K32 stage and, when K is not divisible
    # by 32, one masked and zero-padded tail stage.
    current_stage = (full_tiles - 1) % NUM_STAGES
    preloaded_operands = _load_preloaded_operand_tiles(
        a_local_tiles, b_local_tiles, current_stage,
        A_TILE_COUNT, B_TILE_COUNT, dot0, dot1, PRELOAD_A_OPERANDS,
    )

    if HAS_K_TAIL:
        tail_k_offset = full_tiles * BLOCK_K
        tail_mask = tail_k_offset + rk < K
        if GLOBAL_TAIL_LOAD_BEFORE_LAST_DOT:
            a_tail_global = tl.tuple([])
            for mi in tl.static_range(A_TILE_COUNT):
                a_tail_global += tl.tuple([tl.load(
                    a_ptr
                    + a_global_offsets[mi]
                    + (tail_k_offset + rk[None, :]) * sak,
                    mask=tail_mask[None, :],
                    other=0.0,
                )])
            b_tail_global = tl.tuple([])
            for nj in tl.static_range(B_TILE_COUNT):
                b_global_ptrs = (
                    b_ptr
                    + (tail_k_offset + rk[:, None]) * sbk
                    + b_global_offsets[nj]
                )
                if GLOBAL_LOAD_B_CG:
                    b_tail_global += tl.tuple([tl.load(
                        b_global_ptrs,
                        mask=tail_mask[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )])
                else:
                    b_tail_global += tl.tuple([tl.load(
                        b_global_ptrs,
                        mask=tail_mask[:, None],
                        other=0.0,
                    )])
            acc = _load_each_streamed_operand_and_dot(
                a_local_tiles, b_local_tiles, preloaded_operands,
                current_stage, acc, A_TILE_COUNT, B_TILE_COUNT,
                dot0, dot1, PRELOAD_A_OPERANDS,
            )
        else:
            acc = _load_each_streamed_operand_and_dot(
                a_local_tiles, b_local_tiles, preloaded_operands,
                current_stage, acc, A_TILE_COUNT, B_TILE_COUNT,
                dot0, dot1, PRELOAD_A_OPERANDS,
            )
            a_tail_global = tl.tuple([])
            for mi in tl.static_range(A_TILE_COUNT):
                a_tail_global += tl.tuple([tl.load(
                    a_ptr
                    + a_global_offsets[mi]
                    + (tail_k_offset + rk[None, :]) * sak,
                    mask=tail_mask[None, :],
                    other=0.0,
                )])
            b_tail_global = tl.tuple([])
            for nj in tl.static_range(B_TILE_COUNT):
                b_global_ptrs = (
                    b_ptr
                    + (tail_k_offset + rk[:, None]) * sbk
                    + b_global_offsets[nj]
                )
                if GLOBAL_LOAD_B_CG:
                    b_tail_global += tl.tuple([tl.load(
                        b_global_ptrs,
                        mask=tail_mask[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )])
                else:
                    b_tail_global += tl.tuple([tl.load(
                        b_global_ptrs,
                        mask=tail_mask[:, None],
                        other=0.0,
                    )])
        tail_stage = full_tiles % NUM_STAGES
        for mi in tl.static_range(A_TILE_COUNT):
            tlx.local_store(
                tlx.local_view(a_local_tiles[mi], tail_stage),
                a_tail_global[mi],
            )
        for nj in tl.static_range(B_TILE_COUNT):
            tlx.local_store(
                tlx.local_view(b_local_tiles[nj], tail_stage),
                b_tail_global[nj],
            )
        tl.debug_barrier()
        preloaded_operands = _load_preloaded_operand_tiles(
            a_local_tiles, b_local_tiles, tail_stage,
            A_TILE_COUNT, B_TILE_COUNT, dot0, dot1, PRELOAD_A_OPERANDS,
        )
        acc = _load_each_streamed_operand_and_dot(
            a_local_tiles, b_local_tiles, preloaded_operands,
            tail_stage, acc, A_TILE_COUNT, B_TILE_COUNT,
            dot0, dot1, PRELOAD_A_OPERANDS,
        )
    else:
        acc = _load_each_streamed_operand_and_dot(
            a_local_tiles, b_local_tiles, preloaded_operands,
            current_stage, acc, A_TILE_COUNT, B_TILE_COUNT,
            dot0, dot1, PRELOAD_A_OPERANDS,
        )

    # Epilogue: keep the high-RP accumulator tuple in this JIT scope while
    # converting and storing every logical C tile.
    base = c_ptr + batch_id.to(tl.int64) * scb
    element_type = c_ptr.dtype.element_ty
    for mi in tl.static_range(A_TILE_COUNT):
        for nj in tl.static_range(B_TILE_COUNT):
            offsets = tlx.require_layout(
                base
                + output_rows[mi][:, None] * scm
                + output_cols[nj][None, :] * scn,
                mma,
                pin=False,
            )
            value = acc[mi * B_TILE_COUNT + nj].to(element_type)
            if EVEN_M and EVEN_N:
                tl.store(offsets, value)
            else:
                mask = tlx.require_layout(
                    (output_rows[mi][:, None] < M)
                    & (output_cols[nj][None, :] < N),
                    mma,
                    pin=False,
                )
                tl.store(offsets, value, mask=mask)


def _validate_register_staged_spec(kernel_spec):
    assert kernel_spec.block_m == sum(kernel_spec.a_row_tile_sizes)
    assert kernel_spec.block_n == sum(kernel_spec.b_col_tile_sizes)
    assert kernel_spec.a_row_tile_sizes and kernel_spec.b_col_tile_sizes
    assert all(tile_size > 0 and tile_size & (tile_size - 1) == 0
               for tile_size in kernel_spec.a_row_tile_sizes + kernel_spec.b_col_tile_sizes)
    assert kernel_spec.num_stages > 0 and kernel_spec.loop_unroll > 0
    assert kernel_spec.instr_shape in ((16, 16, 32), (32, 32, 16))
    assert len(kernel_spec.warps_per_cta) == 2
    assert kernel_spec.warps_per_cta[0] * kernel_spec.warps_per_cta[1] in (1, 2, 4, 8)
    assert isinstance(kernel_spec.swizzle_b_local, bool)
    assert isinstance(kernel_spec.global_load_b_cg, bool)
    assert isinstance(kernel_spec.global_tail_load_before_last_dot, bool)
    assert isinstance(kernel_spec.local_load_b_before_global_prefetch, bool)
    assert kernel_spec.resident_operand_policy in (
        _RESIDENT_OPERAND_AUTO,
        _RESIDENT_OPERAND_A,
        _RESIDENT_OPERAND_B,
    )


def _launch_register_staged_bmm(a, b, c, *, kernel_spec, batch_group,
                                schedule_spec=None):
    """Launch one specialization of the register-staged K32 pipeline."""
    # Validate the compile-time kernel policy independently from the runtime
    # tensor contract. A new tile shape should only need a new kernel_spec.
    _validate_register_staged_spec(kernel_spec)
    if schedule_spec is not None:
        assert len(schedule_spec) == 2
        assert schedule_spec[0] > 0
        assert isinstance(schedule_spec[1], bool)
    assert a.ndim == 3 and b.ndim == 3 and c.ndim == 3
    assert a.shape[0] == b.shape[0] == c.shape[0]
    assert a.shape[1] == c.shape[1]
    assert a.shape[2] == b.shape[1]
    assert b.shape[2] == c.shape[2]
    assert a.dtype == b.dtype == c.dtype
    assert batch_group > 0
    block_m, block_n = kernel_spec.block_m, kernel_spec.block_n
    instr_shape = kernel_spec.instr_shape
    warps_per_cta = kernel_spec.warps_per_cta
    batch, m, k = a.shape
    n = b.shape[-1]
    assert k >= BLOCK_K, f"K must be >= BLOCK_K={BLOCK_K}, got K={k}"
    assert block_m <= 2 * m, (
        f"BLOCK_M={block_m} requires M >= {(block_m + 1) // 2}, "
        f"got M={m}"
    )
    assert block_n <= 2 * n, (
        f"BLOCK_N={block_n} requires N >= {(block_n + 1) // 2}, "
        f"got N={n}"
    )

    # Flatten [batch, M-tile, N-tile] into one launch dimension. The kernel
    # recovers batch_id and the two output-tile coordinates from program_id.
    n_blocks = triton.cdiv(n, block_n)
    tiles_per_batch = triton.cdiv(m, block_m) * n_blocks
    program_count = batch * tiles_per_batch
    strides = (
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
    )
    launch_options = {}
    if schedule_spec is not None:
        # This compiler policy is optional and orthogonal to tile geometry.
        launch_options = dict(
            enable_sched_group_barrier_scheduler=True,
            sched_group_barrier_mfma_per_dwordx4=schedule_spec[0],
            disable_unclustered_high_rp_reschedule=schedule_spec[1],
        )
    _bmm_register_staged[(program_count, )](
        a, b, c, m, n, k, *strides,
        KERNEL_SPEC=kernel_spec,
        EVEN_M=m % block_m == 0,
        EVEN_N=n % block_n == 0,
        HAS_K_TAIL=k % 32 != 0,
        N_BLOCKS=n_blocks,
        BATCH_GROUP=batch_group,
        GMN=tiles_per_batch,
        NT=program_count,
        num_warps=warps_per_cta[0] * warps_per_cta[1],
        num_stages=1,
        matrix_instr_nonkdim=instr_shape[0],
        **launch_options,
    )
    return c


def bmm_register_staged_template(a, b, kernel_spec, batch_group=NUM_XCDS,
                                 schedule_spec=None):
    batch, m, _ = a.shape
    n = b.shape[-1]
    c = torch.empty((batch, m, n), device=a.device, dtype=a.dtype)
    return _launch_register_staged_bmm(
        a, b, c,
        kernel_spec=kernel_spec,
        batch_group=batch_group,
        schedule_spec=schedule_spec,
    )


def bmm(a, b):
    """C = A @ B, shared-A, ROW-major B (stride_bn == 1)."""
    # sA / sB take their element type from a_ptr / b_ptr independently, so a dtype
    # mismatch would silently give the two LDS buffers different types.
    assert a.dtype == b.dtype, f"A and B must have the same dtype, got {a.dtype} and {b.dtype}"
    Bs, M, K = a.shape
    N = b.shape[-1]
    bm = 64 if M <= 64 else 128
    KI = triton.cdiv(K, BLOCK_K)
    assert KI >= 2, f"K must span >= 2 BLOCK_K={BLOCK_K} tiles for the pipeline, got K={K}"
    # The 2-stage pipeline needs 2 <= nb <= KI: the prologue unconditionally issues
    # nb loads and the drain indexes (KI - (nb - 1) + i) % nb, so nb > KI would
    # over-read past K and re-accumulate tile 0. Requiring KI >= 2 makes
    # min(NB, KI) >= 2 on its own -- a max(2, ...) here would defeat the clamp.
    nb = min(NB, KI)
    GMN = triton.cdiv(M, bm) * triton.cdiv(N, BLOCK_N)
    NT = Bs * GMN
    c = torch.empty((Bs, M, N), device=a.device, dtype=a.dtype)
    attrs = (("amdgpu-agpr-alloc", "0,0"), )
    common = dict(num_warps=8, num_stages=1, matrix_instr_nonkdim=32, llvm_fn_attrs=attrs)
    st = (a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2), c.stride(0), c.stride(1),
          c.stride(2))
    if M == 40 and N == 256 and K == 1956:
        batch_group = LARGE_BATCH_GROUP if Bs >= LARGE_BATCH_GROUP else NUM_XCDS
        return _launch_register_staged_bmm(
            a, b, c,
            kernel_spec=_MT64X256_MI32_KERNEL_SPEC,
            batch_group=batch_group,
            schedule_spec=_REGISTER_STAGED_SCHEDULE_SPEC,
        )
    if M == 262 and N == 256 and K == 294:
        batch_group = (
            BMM_262_256_BATCH_GROUP
            if Bs >= BMM_262_256_BATCH_GROUP
            else NUM_XCDS
        )
        return _launch_register_staged_bmm(
            a, b, c,
            kernel_spec=_MT144X256_MI16_KERNEL_SPEC,
            batch_group=batch_group,
        )
    # Two MT224x160 tile-grid programs cover the 448x160 output. Peeling the
    # odd-K tail lets every full tile use gfx950's unaligned dwordx2 loads even
    # though consecutive A rows are only naturally aligned. Grouping the same
    # 224-row A tile across 256 batches keeps shared-A cache lines resident.
    if M == 448 and N == 160 and K >= BLOCK_K and K % BLOCK_K != 0:
        batch_group = BMM_448_160_BATCH_GROUP if Bs >= BMM_448_160_BATCH_GROUP else NUM_XCDS
        return _launch_register_staged_bmm(
            a, b, c,
            kernel_spec=_MT224X160_MI16_KERNEL_SPEC,
            batch_group=batch_group,
            schedule_spec=_MT224X160_REGISTER_STAGED_SCHEDULE_SPEC,
        )
    # A large/deep shape has enough arithmetic to amortize MI16's larger
    # accumulator bank, while the 256-row tile halves repeated B traffic versus
    # the default 128-row path. Keep this as a shape class, not an exact-shape
    # special case; smaller/shallower BMMs retain the lower-register path below.
    if M >= 512 and N == BLOCK_N and K >= 1024 and K % VENDOR_BLOCK_K != 0:
        vgmn = triton.cdiv(M, VENDOR_BLOCK_M) * triton.cdiv(N, BLOCK_N)
        vnt = Bs * vgmn
        vbb = tuple(tuple(x) for x in _swz((VENDOR_BLOCK_K, BLOCK_N // 2), 1))
        # Grouping only pays once there are enough output tiles per batch.  On
        # gfx950, paired A/B tests show GMN=2/4 regresses slightly, while
        # GMN=5..8 improves.  Requiring a full 64-batch group also avoids
        # changing the launch order for smaller batches where reuse cannot form.
        use_large_batch_group = Bs >= LARGE_BATCH_GROUP and vgmn >= MIN_GROUPED_OUTPUT_TILES
        _bmm_mi16_quad[(vnt, )](
            a, b, c, M, N, K, *st, BM=VENDOR_BLOCK_M, BN=BLOCK_N, BK=VENDOR_BLOCK_K,
            # Group 64 batches of the same M tile consecutively. A is shared
            # across batches, so its cache lines are reused while B streams.
            NUM_XCDS=LARGE_BATCH_GROUP if use_large_batch_group else NUM_XCDS,
            GMN=vgmn, NT=vnt, B_BASES=vbb,
            num_warps=4, num_stages=1,
            matrix_instr_nonkdim=16,
            reverse_local_assignment=True,
            disable_unclustered_high_rp_reschedule=use_large_batch_group,
            enable_sched_group_barrier_scheduler=use_large_batch_group,
            sched_group_barrier_required_region_count=4)
        return c
    # K % BLOCK_K, not K % 8: the direct path does no K-tail masking on
    # buffer_load_to_local, so a K that is 8-aligned but not BLOCK_K-aligned
    # (e.g. 264) reads past the K extent. Matches the guard in amd_bmm.py.
    if K % BLOCK_K == 0:  # 16-byte-aligned A rows, no K-tail -> direct-to-LDS (wins)
        AB = tuple(tuple(x) for x in _swz([bm, BLOCK_K], 1))
        BB = tuple(tuple(x) for x in _swz([BLOCK_K, BLOCK_N], 1))
        _bmm_direct[(NT, )](a, b, c, M, N, K, *st, BM=bm, BN=BLOCK_N, BK=BLOCK_K, AB=AB, BB=BB, NUM_XCDS=NUM_XCDS,
                            GMN=GMN, NT=NT, NB=nb, **common)
    else:  # odd / unaligned K -> register path
        _bmm_register[(NT, )](a, b, c, M, N, K, *st, BM=bm, BN=BLOCK_N, BK=BLOCK_K, NUM_XCDS=NUM_XCDS, GMN=GMN, NT=NT,
                              NB=nb, **common)
    return c


def make_bmm_inputs(B, M, N, K, device, dtype=torch.float16, seed=0):
    """SHARED-A: one (M,K) reused across the batch -> a.stride(0)==0. B row-major.

    We always use shared-A: it is the shared-LHS layout, and distinct-A flatters TLX
    (rocBLAS reads shared-A once from HBM, so it is much faster on distinct-A).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((M, K), device=device, dtype=dtype, generator=g).unsqueeze(0).expand(B, M, K)
    b = torch.randn((B, K, N), device=device, dtype=dtype, generator=g)  # row-major (stride_bn == 1)
    return a, b


def _warm_ms(fn, iters=60, warmup=20):
    """Warm device time (L2 hot, back-to-back) — matches rocprofv3 kernel-trace, no launch tax."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    dev = triton.runtime.driver.active.get_active_torch_device()
    # (B, M, N, K): representative shared-LHS shapes.  Always shared-A.
    shapes = [(320, 1024, 256, 256), (1024, 395, 256, 320), (1024, 40, 256, 1956), (1024, 262, 256, 294),
              (1024, 1195, 256, 2309)]
    print("mode: shared-A (shared-LHS)   (B row-major)")
    print(f"{'M x N x K (B)':<22}{'path':<8}{'TLX':>9}{'rocBLAS':>10}{'ratio':>8}  {'ok'}")
    for B, M, N, K in shapes:
        a, b = make_bmm_inputs(B, M, N, K, dev)
        ref = torch.bmm(a, b)
        out = bmm(a, b)
        ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
        t = _warm_ms(lambda: bmm(a, b)) * 1e3
        rb = _warm_ms(lambda: torch.bmm(a, b)) * 1e3
        path = "direct" if K % BLOCK_K == 0 else "reg"
        print(f"{f'{M}x{N}x{K} ({B})':<22}{path:<8}{t:8.0f}u{rb:9.0f}u{rb / t:7.2f}x  {'OK' if ok else 'WRONG'}")
