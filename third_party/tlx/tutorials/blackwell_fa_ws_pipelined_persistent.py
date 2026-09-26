import os
from contextvars import copy_context
from enum import IntEnum
from typing import NamedTuple

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language import core
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.language.extra.cuda.inline_ptx_lib import _mul_f32x2, _fma_f32x2, _sub_f32x2
from triton.language.extra.subtile_ops import _join_n_2D, _split_n_2D
from triton.tools.tensor_descriptor import TensorDescriptor


class Policy(IntEnum):
    DENSE = 0
    CAUSAL_64K = 1
    CAUSAL_128K = 2
    CAUSAL_256K = 3
    CAUSAL_512K = 4


class ForwardPlan(NamedTuple):
    stage: int
    pipelined: bool
    policy: int
    num_ctas: int


POLICY_DENSE = tl.constexpr(int(Policy.DENSE))
CAUSAL_POLICY_BIT_OFFSET = 16
FWD_BLOCK_M = tl.constexpr(256)
DEVICE = triton.runtime.driver.active.get_active_torch_device()
BWD_2CTA_MIN_N_CTX = 256
PERSISTENT_CTA_FRACTION = 0.92

BWD_DIRECT_DQ_CONFIG = (False, 8, 2)


def _forward_layout_bshd(q, k, v):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("attention requires matching four-dimensional Q, K, and V tensors")
    values = (q, k, v)
    if all(value.is_contiguous() for value in values):
        return False
    if all(
        tuple(value.stride()) == (value.shape[2] * value.shape[1] * value.shape[3],
                                  value.shape[3], value.shape[1] * value.shape[3], 1)
        for value in values
    ):
        return True
    raise ValueError("attention requires contiguous BHSD or matching BSHD-backed BHSD views")


def _forward_descriptor(value, block_shape, layout_bshd):
    b, h, n, d = value.shape
    shape = [b * n, h * d] if layout_bshd else [b * h * n, d]
    strides = [h * d, 1] if layout_bshd else [d, 1]
    return TensorDescriptor(value, shape=shape, strides=strides, block_shape=block_shape)


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    # In 2-CTA mode Q/O stay per-CTA (own M rows, full head-dim). K/V are the
    # collective-MMA B operands, split along their FREE dim and HW-reassembled:
    #   QK (contraction=HEAD_DIM): K split along BLOCK_N rows.
    #   PV (contraction=BLOCK_N):  V split along HEAD_DIM cols.
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N // NUM_CTAS, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM // NUM_CTAS]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]


configs = [
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": kv,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
            "GROUP_SIZE_N": grp_n,
            "RESCALE_OPT": rescale_opt,
            "USE_WHERE": where,  # used when RESCALE_OPT is True
            "USE_WARP_BARRIER": uwb,
            "FAST_FIXED": fast_fixed,
            "COMPACT_CLC": compact_clc,
            "GLOBAL_LPT": global_lpt,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    ) for kv in [3, 6] for grp_n in [1, 2, 4, 8, 16, 32, 64] for (rescale_opt, where, fast_fixed) in [
        (False, False, True),
        (False, False, False),
        (True, False, False),
        (True, True, False),
    ] for uwb in [False, True]
    for compact_clc in ([False, True] if grp_n > 4 and uwb and not fast_fixed else [False])
    for global_lpt in ([False, True] if compact_clc else [False])
] + [
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": kv,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
            "GROUP_SIZE_N": grp_n,
            "RESCALE_OPT": rescale_opt,
            "USE_WHERE": where,
            "USE_WARP_BARRIER": False,
            "NUM_CTAS": 2,
            "FAST_FIXED": fast_fixed,
            "COMPACT_CLC": False,
            "GLOBAL_LPT": False,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
        ctas_per_cga=(2, 1, 1),
    ) for kv in [3, 6] for grp_n in [1] for (rescale_opt, where, fast_fixed) in [
        (False, False, True),
        (False, False, False),
        (True, False, False),
        (True, True, False),
    ]
]

_fwd_num_ctas = os.environ.get("TLX_FWD_NUM_CTAS")
if _fwd_num_ctas is not None:
    configs = [config for config in configs if config.kwargs.get("NUM_CTAS", 1) == int(_fwd_num_ctas)]

_fwd_num_buffers_kv = os.environ.get("TLX_FWD_NUM_BUFFERS_KV")
if _fwd_num_buffers_kv is not None:
    configs = [config for config in configs if config.kwargs["NUM_BUFFERS_KV"] == int(_fwd_num_buffers_kv)]

_fwd_rescale_opt = os.environ.get("TLX_FWD_RESCALE_OPT")
if _fwd_rescale_opt is not None:
    configs = [config for config in configs if config.kwargs["RESCALE_OPT"] == bool(int(_fwd_rescale_opt))]

_fwd_use_where = os.environ.get("TLX_FWD_USE_WHERE")
if _fwd_use_where is not None:
    configs = [config for config in configs if config.kwargs["USE_WHERE"] == bool(int(_fwd_use_where))]


def prune_configs_by_hdim(configs, named_args, **kwargs):
    HEAD_DIM = kwargs["HEAD_DIM"]
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]
    target_kv_buffers = 6 if HEAD_DIM == 64 else 3
    if STAGE == 3:
        h = int(named_args["H"])
        # Size a Q/K section against the usable L2 budget. This kernel supports
        # fp16 and bf16 inputs, so every Q/K element is two bytes.
        bytes_per_element = 2
        l2_budget_bytes = 50 * 1024 * 1024
        bytes_per_head = 2 * N_CTX * HEAD_DIM * bytes_per_element
        capacity_heads = max(1, l2_budget_bytes // bytes_per_head)
        if capacity_heads >= h:
            l2_group_size_n = 1 << (h - 1).bit_length()
        else:
            l2_group_size_n = 1 << (capacity_heads.bit_length() - 1)
        l2_group_size_n = min(64, l2_group_size_n)
        target_group_sizes = ((4, l2_group_size_n) if l2_group_size_n > 4 else (4,))
    else:
        target_group_sizes = (1,)
    pruned = []
    for conf in configs:
        kv = conf.kwargs.get("NUM_BUFFERS_KV", 0)
        grp_n = conf.kwargs.get("GROUP_SIZE_N", 0)
        is_2cta = conf.kwargs.get("NUM_CTAS", 1) > 1
        if grp_n not in target_group_sizes:
            continue
        if is_2cta:
            num_ctas = conf.kwargs["NUM_CTAS"]
            if N_CTX % (conf.kwargs["BLOCK_M"] * num_ctas) != 0:
                continue
            if kv != target_kv_buffers:
                continue
        else:
            if kv != target_kv_buffers:
                continue
        pruned.append(conf)
    return pruned


_RCP_LN2 = tl.constexpr(1.4426950408889634)
_ROUNDING_GAUGE = tl.constexpr(0.055517269)
_BF16_FIXED_GAUGE = tl.constexpr(4.055517269)
_EXP2_MAGIC = tl.constexpr(12582912.0)
_EXP2_BF16_SCALE = tl.constexpr(128.0)
_EXP2_BF16_BIAS = tl.constexpr(126.0)
_EXP2_F16_SCALE = tl.constexpr(1024.0)
_EXP2_F16_BIAS = tl.constexpr(14.0)


@core.builtin
def _s2_exp2_f16(x, a2, b2m, _semantic=None):
    return core.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            .reg .b32 v0, v1, pk, zz, uu, mk, c0;
            mov.b64 ra, { $1, $2 };
            mov.b64 rb, { $3, $4 };
            mov.b64 rc, { $5, $6 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { v0, v1 }, rd;
            prmt.b32 pk, v0, v1, 0x5410;
            mov.b32 zz, 0;
            mov.b32 mk, 0x02000200;
            add.s16x2 uu, pk, mk;
            mov.b32 c0, 0x39AB39AB;
            fma.rn.f16x2 pk, uu, c0, pk;
            max.f16x2 pk, pk, zz;
            mov.b32 $0, pk;
        }
        """,
        "=r,r,r,r,r,r,r",
        [x, a2, b2m],
        dtype=core.float16,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )


@core.builtin
def _s2_exp2_bf16(x, a2, b2m, _semantic=None):
    return core.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            .reg .b32 v0, v1, pk, uu, mk, c0;
            mov.b64 ra, { $1, $2 };
            mov.b64 rb, { $3, $4 };
            mov.b64 rc, { $5, $6 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { v0, v1 }, rd;
            prmt.b32 pk, v0, v1, 0x5410;
            mov.b32 mk, 0x00400040;
            add.s16x2 uu, pk, mk;
            mov.b32 c0, 0x3f3b3f3b;
            fma.rn.relu.bf16x2 pk, uu, c0, pk;
            mov.b32 $0, pk;
        }
        """,
        "=r,r,r,r,r,r,r",
        [x, a2, b2m],
        dtype=core.bfloat16,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )


@core.builtin
def _mul_bf16x2(x, y, _semantic=None):
    return core.inline_asm_elementwise(
        "mul.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        [x, y],
        dtype=core.bfloat16,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )


@triton.jit
def _reduce_bf16(a, b):
    return (a + b).to(tl.bfloat16)


@triton.jit
def _reduce_or(x, y):
    return x | y


@triton.jit
def _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE: tl.constexpr):
    if STAGE == 1:
        # First part of STAGE == 3 in _get_fused_loop_bounds
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # Second part of STAGE == 3 in _get_fused_loop_bounds
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        tl.static_assert(STAGE == 3)
        # Maps to STAGE=1 in _get_fused_loop_bounds
        lo, hi = 0, N_CTX
    return lo, hi


@triton.jit
def _get_start_m_bwd(start_n, BLOCK_N1, STAGE: tl.constexpr):
    if STAGE == 1:
        return 0
    else:
        tl.static_assert(STAGE == 3)
        return start_n * BLOCK_N1


@triton.jit
def _get_unfused_bwd_loop_bounds(start_n, N_CTX, BLOCK_N1, STAGE: tl.constexpr):
    if STAGE == 1:
        # First part of STAGE == 3
        lo, hi = start_n * BLOCK_N1, (start_n + 1) * BLOCK_N1
    elif STAGE == 2:
        # Second part of STAGE == 3 in this function
        lo, hi = (start_n + 1) * BLOCK_N1, N_CTX
    else:
        tl.static_assert(STAGE == 3)
        lo, hi = 0, N_CTX
    return lo, hi


@triton.jit
def _get_fused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE: tl.constexpr):
    if STAGE == 1:
        return 0, N_CTX
    else:
        tl.static_assert(STAGE == 3)
        return 0, (start_m + 1) * BLOCK_M


@triton.jit
def _compute_offsets(
    tile_idx,
    H,
    num_pid_n,
    num_pid_in_group,
    N_CTX,
    BLOCK_M: tl.constexpr,
    STAGE: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    NUM_PID_M_STATIC: tl.constexpr,
    GRID_X_STATIC: tl.constexpr,
    GLOBAL_LPT: tl.constexpr,
    DENSE_PAIRED: tl.constexpr = False,
):
    if STAGE == 3 and GROUP_SIZE_N > 4:
        if GRID_X_STATIC > 0:
            head_lane = tile_idx % GRID_X_STATIC
            grid_row = tile_idx // GRID_X_STATIC
            head_chunks = H // GRID_X_STATIC
            if GLOBAL_LPT:
                num_batches = num_pid_n // H
                batch_chunks = num_batches * head_chunks
                m_in_section = grid_row // batch_chunks
                batch_chunk = grid_row % batch_chunks
                off_z = batch_chunk // head_chunks
                head_chunk = batch_chunk % head_chunks
                off_h = head_chunk * GRID_X_STATIC + head_lane
            else:
                rows_per_batch: tl.constexpr = NUM_PID_M_STATIC * head_chunks
                off_z = grid_row // rows_per_batch
                row_in_batch = grid_row % rows_per_batch
                if H <= GROUP_SIZE_N:
                    m_in_section = row_in_batch // head_chunks
                    head_chunk = row_in_batch % head_chunks
                    off_h = head_chunk * GRID_X_STATIC + head_lane
                else:
                    chunks_per_full_section: tl.constexpr = GROUP_SIZE_N // GRID_X_STATIC
                    rows_per_full_section: tl.constexpr = NUM_PID_M_STATIC * chunks_per_full_section
                    full_sections: tl.constexpr = H // GROUP_SIZE_N
                    tail_heads: tl.constexpr = H % GROUP_SIZE_N
                    if tail_heads == 0:
                        section_id = row_in_batch // rows_per_full_section
                        row_in_section = row_in_batch % rows_per_full_section
                        m_in_section = row_in_section // chunks_per_full_section
                        head_chunk = row_in_section % chunks_per_full_section
                    else:
                        full_rows: tl.constexpr = full_sections * rows_per_full_section
                        in_tail = row_in_batch >= full_rows
                        full_row = row_in_batch % rows_per_full_section
                        tail_chunks: tl.constexpr = tail_heads // GRID_X_STATIC
                        tail_row = row_in_batch - full_rows
                        section_id = tl.where(in_tail, full_sections,
                                              row_in_batch // rows_per_full_section)
                        m_in_section = tl.where(in_tail, tail_row // tail_chunks,
                                                full_row // chunks_per_full_section)
                        head_chunk = tl.where(in_tail, tail_row % tail_chunks,
                                              full_row % chunks_per_full_section)
                    off_h = section_id * GROUP_SIZE_N + head_chunk * GRID_X_STATIC + head_lane
            start_m = NUM_PID_M_STATIC - 1 - m_in_section
            off_hz = off_z * H + off_h
        else:
            num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
            tiles_per_batch = num_pid_m * H
            off_z = tile_idx // tiles_per_batch
            tile_in_batch = tile_idx % tiles_per_batch
            section_id = tile_in_batch // (num_pid_m * GROUP_SIZE_N)
            first_head = section_id * GROUP_SIZE_N
            group_size_n = min(H - first_head, GROUP_SIZE_N)
            tile_in_section = tile_in_batch - section_id * num_pid_m * GROUP_SIZE_N
            start_m = tile_in_section // group_size_n
            off_h = first_head + tile_in_section % group_size_n
            start_m = num_pid_m - 1 - start_m
            off_hz = off_z * H + off_h
    else:
        group_id = tile_idx // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        group_size_n = 1 if DENSE_PAIRED and STAGE == 1 and GROUP_SIZE_N == 1 else min(num_pid_n - first_pid_n, GROUP_SIZE_N)
        start_m = (tile_idx % num_pid_in_group) // group_size_n
        off_hz = first_pid_n + (tile_idx % group_size_n)
        off_z = off_hz // H
        off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    lo, hi = _get_fused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)
    kv_offset_y = offset_y + lo
    return start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y


@triton.jit
def _forward_descriptor_offsets(qo_y, kv_y, off_hz, H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr,
                                LAYOUT_BSHD: tl.constexpr):
    if LAYOUT_BSHD:
        batch = off_hz // H
        head = off_hz % H
        row_shift = (batch - off_hz) * N_CTX
        return qo_y + row_shift, kv_y + row_shift, head * HEAD_DIM
    return qo_y, kv_y, 0


@triton.jit
def _mask_scalar(qk, col_limit, s, i, keep_ge: tl.constexpr):
    # Bitmask for a block of 16 elements: bit i is set iff column (s + i) >= col_limit.
    cur = max(col_limit - s, 0)
    bit = (-1 << cur) & (1 << i)
    # keep_ge=True keeps columns >= col_limit (left limit); keep_ge=False keeps
    # columns < col_limit (right limit).
    keep = (bit != 0) if keep_ge else (bit == 0)
    return tl.where(keep, qk, -float("inf"))


@triton.jit
def _mask_scalar_right(qk, col_limit, s, i):
    return _mask_scalar(qk, col_limit, s, i, False)


@triton.jit
def _mask_scalar_left(qk, col_limit, s, i):
    return _mask_scalar(qk, col_limit, s, i, True)


@triton.jit
def _apply_causal_mask(qk, col_limit, BLOCK: tl.constexpr, keep_ge: tl.constexpr = False):
    # Apply causal mask via a bitmask calculated for each block of 16 elements.
    # This allows the efficient R2P (register to predicate) instruction to be used at the SASS level.
    # Credit to Tri Dao,
    # https://github.com/Dao-AILab/flash-attention/commit/bac1001e4f6caa09d70537495d6746a685a2fa78
    #
    # NOTE: We use map_elementwise here in order to generate an interleaved sequence of instructions
    # that processes one element of qk at a time. This improves ptxas's resulting SASS.
    #
    # keep_ge=False: qk is [..., N], keep keys < col_limit (forward, right limit).
    # keep_ge=True: qk is transposed [..., M], keep queries >= col_limit
    # (backward, left limit).
    offs = tl.arange(0, BLOCK)[None, :]
    s = offs & ~0xF
    i = offs & 0xF
    if keep_ge:
        return tl.map_elementwise(_mask_scalar_left, qk, col_limit, s, i)
    return tl.map_elementwise(_mask_scalar_right, qk, col_limit, s, i)


@triton.jit
def _certificate_maximum_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

@core.builtin
def _add_f32x2_half_sum(a, b, _semantic=None):
    return core.inline_asm_elementwise(
        r"""{
    .reg .b64 aa,bb,cc;
    mov.b64 aa,{$2,$3};
    mov.b64 bb,{$4,$5};
    add.rn.f32x2 cc,aa,bb;
    mov.b64 {$0,$1},cc;
}""",
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=core.float32,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )

@triton.jit
def _sum_p_four_pairs(p):
    pieces = _split_n_2D(p, 8)
    accum = pieces[0]
    for part in tl.static_range(1, 8):
        accum = _add_f32x2_half_sum(accum, pieces[part])
    pairs = _split_n_2D(accum, 4)
    left = _add_f32x2_half_sum(pairs[0], pairs[1])
    right = _add_f32x2_half_sum(pairs[2], pairs[3])
    total = _add_f32x2_half_sum(left, right)
    return tl.sum(total, 1)


@triton.jit
def _fwd_softmax_tile_1cta(
    qk_fulls,
    qk_tiles,
    p_fulls,
    p_tiles,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    cid,
    accum_cnt_qk,
    gauge_cnt,
    qk_scale,
    offs_m,
    m_i,
    l_i,
    start_m,
    N_CTX,
    out_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    STAGE: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
    SCALAR_N: tl.constexpr,
    FAST_FIXED: tl.constexpr,
    SKIP_CAUSAL_DIAG: tl.constexpr,
    CERTIFIED_CAUSAL: tl.constexpr = False,
    MERGED_STAGE: tl.constexpr = False,
):
    _ROUNDING_GAUGE: tl.constexpr = 0.055517269
    _EXP2_MAGIC: tl.constexpr = 12582912.0
    _EXP2_BF16_SCALE: tl.constexpr = 128.0
    _EXP2_BF16_BIAS: tl.constexpr = 126.0
    _EXP2_F16_SCALE: tl.constexpr = 1024.0
    _EXP2_F16_BIAS: tl.constexpr = 14.0
    FAST_BF16: tl.constexpr = FAST_FIXED and out_dtype == tl.bfloat16
    FAST_F16: tl.constexpr = FAST_FIXED and out_dtype == tl.float16
    NUM_P_SLICES: tl.constexpr = 2 if FAST_F16 else NUM_MMA_SLICES
    if CERTIFIED_CAUSAL:
        m_scaled = m_i * qk_scale
        if MERGED_STAGE:
            lo, hi = 0, (start_m + 1) * BLOCK_M
        else:
            lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)
        for start_n in tl.range(lo, hi, BLOCK_N):
            _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
            tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
            skip_q0 = SKIP_CAUSAL_DIAG and (STAGE == 2 or MERGED_STAGE) and cid == 0 and start_n + BLOCK_N >= hi
            if skip_q0:
                for slice_id in tl.static_range(0, NUM_P_SLICES):
                    tlx.barrier_arrive(tlx.local_view(p_fulls, cid * NUM_P_SLICES + slice_id))
            else:
                qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
                if STAGE == 2 or (MERGED_STAGE and start_n >= start_m * BLOCK_M):
                    qk = _apply_causal_mask(qk, (offs_m - start_n + 1)[:, None], BLOCK_N)
                if start_n == 0:
                    m_i = tl.max(qk, 1)
                    m_scaled = m_i * qk_scale
                qk = _fma_f32x2(qk, qk_scale, -m_scaled[:, None])
                qks = _split_n_2D(qk, NUM_MMA_SLICES)
                l_ij = tl.zeros_like(l_i)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    p_i = tl.math.exp2(qks[slice_id])
                    p_h = p_i.to(out_dtype)
                    tlx.local_store(tlx.local_view(p_tiles, cid * NUM_P_SLICES + slice_id), p_h)
                    tlx.barrier_arrive(tlx.local_view(p_fulls, cid * NUM_P_SLICES + slice_id))
                    l_partial = _sum_p_four_pairs(p_i)
                    l_ij += l_partial
                    gauge_cnt = tl.maximum(gauge_cnt, l_partial)
                l_i += l_ij
            accum_cnt_qk += 1
        return m_i, l_i, accum_cnt_qk, gauge_cnt
    elif FAST_FIXED:
        lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)
        for start_n in tl.range(lo, hi, BLOCK_N):
            _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
            tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
            skip_q0 = SKIP_CAUSAL_DIAG and STAGE == 2 and cid == 0 and start_n + BLOCK_N >= hi
            if skip_q0:
                for slice_id in tl.static_range(0, NUM_P_SLICES):
                    p_bufIdx = cid * NUM_P_SLICES + slice_id
                    tlx.barrier_arrive(tlx.local_view(p_fulls, p_bufIdx))
            else:
                if gauge_cnt == 0:
                    for fragment_id in tl.static_range(0, 4):
                        qk_fragment = tlx.local_load(tlx.subslice(
                            tlx.local_view(qk_tiles, cid),
                            fragment_id * 32,
                            32,
                        ))
                        if STAGE == 2:
                            col_limit_right = (offs_m - (start_n + fragment_id * 32) + 1)[:, None]
                            qk_fragment = _apply_causal_mask(qk_fragment, col_limit_right, 32)
                        m_i = tl.maximum(m_i, tl.max(qk_fragment, 1) * qk_scale)
                    m_i = tl.ceil(m_i) + _ROUNDING_GAUGE
                l_ij = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
                p_pending = tl.zeros([BLOCK_M // 2, 32], dtype=out_dtype)
                for fragment_id in tl.static_range(0, 4):
                    qk_fragment = tlx.local_load(tlx.subslice(
                        tlx.local_view(qk_tiles, cid),
                        fragment_id * 32,
                        32,
                    ))
                    if STAGE == 2:
                        col_limit_right = (offs_m - (start_n + fragment_id * 32) + 1)[:, None]
                        qk_fragment = _apply_causal_mask(qk_fragment, col_limit_right, 32)
                    if FAST_BF16:
                        p_h = _s2_exp2_bf16(
                            qk_fragment,
                            qk_scale * _EXP2_BF16_SCALE,
                            _EXP2_MAGIC + _EXP2_BF16_SCALE * (_EXP2_BF16_BIAS - m_i[:, None]),
                        )
                    elif STAGE == 2:
                        p_i = tl.math.exp2(_fma_f32x2(qk_fragment, qk_scale, -m_i[:, None]))
                        p_h = p_i.to(out_dtype)
                    else:
                        p_h = _s2_exp2_f16(
                            qk_fragment,
                            qk_scale * _EXP2_F16_SCALE,
                            _EXP2_MAGIC + _EXP2_F16_SCALE * (_EXP2_F16_BIAS - m_i[:, None]),
                        )
                    if (FAST_BF16 or FAST_F16) and fragment_id % 2 == 0:
                        p_pending = p_h
                    elif FAST_BF16 or FAST_F16:
                        p_bufIdx = cid * 2 + fragment_id // 2
                        tlx.local_store(tlx.local_view(p_tiles, p_bufIdx), _join_n_2D([p_pending, p_h]))
                        tlx.barrier_arrive(tlx.local_view(p_fulls, p_bufIdx))
                    if FAST_BF16:
                        l_ij += tl.reduce(p_h, axis=1, combine_fn=_reduce_bf16).to(tl.float32)
                    elif STAGE == 2:
                        l_ij += tl.sum(p_i, 1)
                    else:
                        l_ij += tl.sum(p_h, 1).to(tl.float32)
                l_i += l_ij
            accum_cnt_qk += 1
            gauge_cnt += 1
        return m_i, l_i, accum_cnt_qk, gauge_cnt

    if MERGED_STAGE:
        lo, hi = 0, (start_m + 1) * BLOCK_M
    else:
        lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)
    for start_n in tl.range(lo, hi, BLOCK_N):
        qk_buf = cid
        qk_buf_phase = accum_cnt_qk & 1
        alpha_phase = accum_cnt_qk & 1
        tlx.barrier_wait(tlx.local_view(qk_fulls, qk_buf), qk_buf_phase)
        skip_q0 = SKIP_CAUSAL_DIAG and (STAGE == 2 or MERGED_STAGE) and cid == 0 and start_n + BLOCK_N >= hi
        if skip_q0:
            alpha = tl.full([BLOCK_M // 2], 1.0, tl.float32)
            tlx.barrier_wait(tlx.local_view(alpha_empties, cid), alpha_phase ^ 1)
            tlx.local_store(
                tlx.local_view(alpha_tiles, cid),
                tl.join(alpha, alpha) if SCALAR_N == 2 else alpha[:, None],
            )
            tlx.barrier_arrive(tlx.local_view(alpha_fulls, cid))
            for slice_id in tl.static_range(0, NUM_P_SLICES):
                p_bufIdx = qk_buf * NUM_P_SLICES + slice_id
                tlx.barrier_arrive(tlx.local_view(p_fulls, p_bufIdx))
        else:
            qk = tlx.local_load(tlx.local_view(qk_tiles, qk_buf))

            if STAGE == 2 or (MERGED_STAGE and start_n >= start_m * BLOCK_M):
                col_limit_right = (offs_m - start_n + 1)[:, None]
                qk = _apply_causal_mask(qk, col_limit_right, BLOCK_N)

            if RESCALE_OPT:
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

            if RESCALE_OPT:
                alpha_ = (m_i - m_ij) * qk_scale
                alpha = tl.math.exp2(alpha_)
                rescale_mask = alpha_ >= -8.0
                alpha = tl.where(rescale_mask, 1.0, alpha)
                m_ij = tl.where(rescale_mask, m_i, m_ij)
            else:
                alpha = tl.math.exp2(m_i - m_ij)
            tlx.barrier_wait(tlx.local_view(alpha_empties, cid), alpha_phase ^ 1)
            tlx.local_store(
                tlx.local_view(alpha_tiles, cid),
                tl.join(alpha, alpha) if SCALAR_N == 2 else alpha[:, None],
            )
            tlx.barrier_arrive(tlx.local_view(alpha_fulls, cid))
            l_i *= alpha

            if RESCALE_OPT:
                m_scaled = m_ij * qk_scale
                qk = _fma_f32x2(qk, qk_scale, -m_scaled[:, None])
            else:
                qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
            qks = _split_n_2D(qk, NUM_MMA_SLICES)
            l_ij = tl.zeros_like(l_i)
            p_pending = tl.zeros([BLOCK_M // 2, BLOCK_N // NUM_MMA_SLICES], dtype=out_dtype)
            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                p_i = tl.math.exp2(qks[slice_id])
                p_h = p_i.to(out_dtype)
                if FAST_F16:
                    if slice_id % 2 == 0:
                        p_pending = p_h
                    else:
                        p_bufIdx = qk_buf * NUM_P_SLICES + slice_id // 2
                        tlx.local_store(
                            tlx.local_view(p_tiles, p_bufIdx),
                            _join_n_2D([p_pending, p_h]),
                        )
                        tlx.barrier_arrive(tlx.local_view(p_fulls, p_bufIdx))
                else:
                    p_bufIdx = qk_buf * NUM_P_SLICES + slice_id
                    tlx.local_store(tlx.local_view(p_tiles, p_bufIdx), p_h)
                    tlx.barrier_arrive(tlx.local_view(p_fulls, p_bufIdx))
                l_ij += tl.sum(p_i, 1)

            l_i += l_ij
            m_i = m_ij
        accum_cnt_qk += 1
    return m_i, l_i, accum_cnt_qk, gauge_cnt


@core.builtin
def _bf16_dual_start(a0, a1, a2, a3, b0, b1, b2, b3, _semantic=None):
    return core.inline_asm_elementwise(
        r"""{
    .reg .b32 s, t;
    prmt.b32 s, $1, $5, 0x5410;
    prmt.b32 t, $1, $5, 0x7632;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $2, $6, 0x5410;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $2, $6, 0x7632;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $3, $7, 0x5410;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $3, $7, 0x7632;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $4, $8, 0x5410;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $4, $8, 0x7632;
    add.rn.bf16x2 s, s, t;
    mov.b32 $0, s;
}""",
        "=r,r,r,r,r,r,r,r,r",
        [a0, a1, a2, a3, b0, b1, b2, b3],
        dtype=core.bfloat16,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )

@core.builtin
def _bf16_dual_step(acc, a0, a1, a2, a3, b0, b1, b2, b3, _semantic=None):
    return core.inline_asm_elementwise(
        r"""{
    .reg .b32 s, t;
    mov.b32 s, $1;
    prmt.b32 t, $2, $6, 0x5410;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $2, $6, 0x7632;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $3, $7, 0x5410;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $3, $7, 0x7632;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $4, $8, 0x5410;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $4, $8, 0x7632;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $5, $9, 0x5410;
    add.rn.bf16x2 s, s, t;
    prmt.b32 t, $5, $9, 0x7632;
    add.rn.bf16x2 s, s, t;
    mov.b32 $0, s;
}""",
        "=r,r,r,r,r,r,r,r,r,r",
        [acc, a0, a1, a2, a3, b0, b1, b2, b3],
        dtype=core.bfloat16,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )

@triton.jit
def _sum_bf16_row32_dual(p, q):
    tl.static_assert(p.shape[1] == 32 and q.shape[1] == 32)
    a = _split_n_2D(p, 16)
    b = _split_n_2D(q, 16)
    acc = _bf16_dual_start(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])
    for i in tl.static_range(1, 4):
        acc = _bf16_dual_step(acc, a[4*i], a[4*i+1], a[4*i+2], a[4*i+3], b[4*i], b[4*i+1], b[4*i+2], b[4*i+3])
    return acc


@triton.jit
def _fwd_softmax_tile_2cta(
    qk_fulls,
    qk_tiles,
    p_fulls,
    p_tiles,
    cid,
    accum_cnt_qk,
    gauge_cnt,
    qk_scale,
    m_i,
    l_i,
    start_m,
    N_CTX,
    out_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    STAGE: tl.constexpr,
    DENSE_PAIRED: tl.constexpr = False,
):
    _BF16_FIXED_GAUGE: tl.constexpr = 4.055517269
    _EXP2_MAGIC: tl.constexpr = 12582912.0
    _EXP2_BF16_SCALE: tl.constexpr = 128.0
    _EXP2_BF16_BIAS: tl.constexpr = 126.0
    tl.static_assert(NUM_MMA_SLICES == 2)
    lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)
    for start_n in tl.range(lo, hi, BLOCK_N):
        _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
        tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
        l_ij = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
        p_pending = tl.zeros([BLOCK_M // 2, 32], dtype=out_dtype)
        p_sum_pending = p_pending
        for fragment_id in tl.static_range(0, 4):
            qk_fragment = tlx.local_load(tlx.subslice(
                tlx.local_view(qk_tiles, cid),
                fragment_id * 32,
                32,
            ))
            p_h = _s2_exp2_bf16(
                qk_fragment,
                qk_scale * _EXP2_BF16_SCALE,
                _EXP2_MAGIC + _EXP2_BF16_SCALE * (_EXP2_BF16_BIAS - _BF16_FIXED_GAUGE),
            )
            if fragment_id < 2:
                p_fragment = tlx.local_slice(
                    tlx.local_view(p_tiles, cid * 2),
                    [0, fragment_id * 32],
                    [BLOCK_M // 2, 32],
                )
                tlx.local_store(p_fragment, p_h)
                tlx.barrier_arrive(
                    tlx.local_view(p_fulls, cid * 3 + fragment_id),
                    1,
                    remote_cta_rank=0,
                )
            elif fragment_id == 2:
                p_pending = p_h
            else:
                tlx.local_store(
                    tlx.local_view(p_tiles, cid * 2 + 1),
                    _join_n_2D([p_pending, p_h]),
                )
                tlx.barrier_arrive(
                    tlx.local_view(p_fulls, cid * 3 + 2),
                    1,
                    remote_cta_rank=0,
                )
            if DENSE_PAIRED and STAGE == 3:
                if fragment_id % 2 == 0:
                    p_sum_pending = p_h
                else:
                    sum_pair = _sum_bf16_row32_dual(p_sum_pending, p_h)
                    sum_lo, sum_hi = sum_pair.split()
                    l_ij += sum_lo.to(tl.float32)
                    l_ij += sum_hi.to(tl.float32)
            else:
                l_ij += tl.reduce(p_h, axis=1, combine_fn=_reduce_bf16).to(tl.float32)
            if gauge_cnt == 0:
                m_i = tl.maximum(m_i, tl.max(qk_fragment, 1) * qk_scale)
        if gauge_cnt == 0:
            m_i = tl.ceil(m_i)
        l_i += l_ij
        accum_cnt_qk += 1
        gauge_cnt += 1
    return m_i, l_i, accum_cnt_qk, gauge_cnt


@triton.jit
def _fwd_softmax_tile_2cta_online(
    qk_fulls,
    qk_tiles,
    p_fulls,
    p_tiles,
    acc_fulls,
    acc_tiles,
    cid,
    accum_cnt_qk,
    gauge_cnt,
    qk_scale,
    m_i,
    l_i,
    start_m,
    N_CTX,
    out_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    STAGE: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
):
    tl.static_assert(NUM_MMA_SLICES == 2)
    lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)
    for start_n in tl.range(lo, hi, BLOCK_N):
        _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
        tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
        qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
        if RESCALE_OPT:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            alpha_ = (m_i - m_ij) * qk_scale
            alpha = tl.math.exp2(alpha_)
            rescale_mask = alpha_ >= -8.0
            alpha = tl.where(rescale_mask, 1.0, alpha)
            m_ij = tl.where(rescale_mask, m_i, m_ij)
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            alpha = tl.math.exp2(m_i - m_ij)
        if gauge_cnt > 0:
            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                acc_slice = tlx.subslice(
                    tlx.local_view(acc_tiles, cid),
                    HEAD_DIM * slice_id // NUM_MMA_SLICES,
                    HEAD_DIM // NUM_MMA_SLICES,
                )
                acc = tlx.local_load(acc_slice)
                tlx.local_store(acc_slice, _mul_f32x2(acc, alpha[:, None]))
        tlx.barrier_arrive(
            tlx.local_view(acc_fulls, cid),
            1,
            remote_cta_rank=0,
        )
        l_i *= alpha
        if RESCALE_OPT:
            qk = _fma_f32x2(qk, qk_scale, -(m_ij * qk_scale)[:, None])
        else:
            qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        qk_fragments = _split_n_2D(qk, 4)
        l_ij = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
        p_pending = tl.zeros([BLOCK_M // 2, 32], dtype=out_dtype)
        for fragment_id in tl.static_range(0, 4):
            p_i = tl.math.exp2(qk_fragments[fragment_id])
            p_h = p_i.to(out_dtype)
            if fragment_id < 2:
                p_fragment = tlx.local_slice(
                    tlx.local_view(p_tiles, cid * 2),
                    [0, fragment_id * 32],
                    [BLOCK_M // 2, 32],
                )
                tlx.local_store(p_fragment, p_h)
                tlx.barrier_arrive(
                    tlx.local_view(p_fulls, cid * 3 + fragment_id),
                    1,
                    remote_cta_rank=0,
                )
            elif fragment_id == 2:
                p_pending = p_h
            else:
                tlx.local_store(
                    tlx.local_view(p_tiles, cid * 2 + 1),
                    _join_n_2D([p_pending, p_h]),
                )
                tlx.barrier_arrive(
                    tlx.local_view(p_fulls, cid * 3 + 2),
                    1,
                    remote_cta_rank=0,
                )
            l_ij += tl.sum(p_i, 1)
        l_i += l_ij
        m_i = m_ij
        accum_cnt_qk += 1
        gauge_cnt += 1
    return m_i, l_i, accum_cnt_qk, gauge_cnt


@triton.jit
def _fwd_softmax_tile(
    qk_fulls,
    qk_tiles,
    p_fulls,
    p_tiles,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    acc_fulls,
    acc_tiles,
    cid,
    accum_cnt_qk,
    gauge_cnt,
    qk_scale,
    offs_m,
    m_i,
    l_i,
    start_m,
    N_CTX,
    out_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    STAGE: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
    SCALAR_N: tl.constexpr,
    USE_2CTA: tl.constexpr,
    FAST_FIXED: tl.constexpr,
    SKIP_CAUSAL_DIAG: tl.constexpr,
    CERTIFIED_CAUSAL: tl.constexpr = False,
    DENSE_PAIRED: tl.constexpr = False,
):
    if USE_2CTA:
        if FAST_FIXED:
            return _fwd_softmax_tile_2cta(
                qk_fulls,
                qk_tiles,
                p_fulls,
                p_tiles,
                cid,
                accum_cnt_qk,
                gauge_cnt,
                qk_scale,
                m_i,
                l_i,
                start_m,
                N_CTX,
                out_dtype,
                BLOCK_M,
                BLOCK_N,
                NUM_MMA_SLICES,
                STAGE,
                DENSE_PAIRED=DENSE_PAIRED,
            )
        return _fwd_softmax_tile_2cta_online(
            qk_fulls,
            qk_tiles,
            p_fulls,
            p_tiles,
            acc_fulls,
            acc_tiles,
            cid,
            accum_cnt_qk,
            gauge_cnt,
            qk_scale,
            m_i,
            l_i,
            start_m,
            N_CTX,
            out_dtype,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            NUM_MMA_SLICES,
            STAGE,
            RESCALE_OPT,
        )
    return _fwd_softmax_tile_1cta(
        qk_fulls,
        qk_tiles,
        p_fulls,
        p_tiles,
        alpha_empties,
        alpha_fulls,
        alpha_tiles,
        cid,
        accum_cnt_qk,
        gauge_cnt,
        qk_scale,
        offs_m,
        m_i,
        l_i,
        start_m,
        N_CTX,
        out_dtype,
        BLOCK_M,
        BLOCK_N,
        NUM_MMA_SLICES,
        STAGE,
        RESCALE_OPT,
        SCALAR_N,
        FAST_FIXED,
        SKIP_CAUSAL_DIAG,
        CERTIFIED_CAUSAL,
    )


@triton.jit
def _fwd_fixed_2cta_control_tile(
    tile_id,
    tile_count,
    accum_cnt_qk,
    gauge_cnt,
    sm_scale,
    M,
    H,
    num_pid_n,
    num_pid_in_group,
    N_CTX,
    desc_v,
    desc_o,
    qk_fulls,
    qk_tiles,
    p_fulls,
    p_tiles,
    acc_empties,
    acc_tiles,
    qk_empties,
    o_empties,
    o_fulls,
    o_tiles,
    cluster_cta_rank,
    BLOCK_M: tl.constexpr,
    BLOCK_M_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EFFECTIVE_BLOCK_M: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    NUM_PID_M_STATIC: tl.constexpr,
    GRID_X_STATIC: tl.constexpr,
    GLOBAL_LPT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
    STAGE: tl.constexpr,
    PIPELINED: tl.constexpr,
    CERTIFIED_CAUSAL: tl.constexpr = False,
    alpha_empties=None,
    alpha_fulls=None,
    alpha_tiles=None,
    Failure=None,
    clc_context=None,
    clc_phase_producer=None,
    DENSE_PAIRED: tl.constexpr = False,
):
    _RCP_LN2: tl.constexpr = 1.4426950408889634
    _BF16_FIXED_GAUGE: tl.constexpr = 4.055517269
    start_m, off_hz, lo, hi, _, _ = _compute_offsets(
        tile_id,
        H,
        num_pid_n,
        num_pid_in_group,
        N_CTX,
        EFFECTIVE_BLOCK_M,
        STAGE,
        GROUP_SIZE_N,
        NUM_PID_M_STATIC,
        GRID_X_STATIC,
        GLOBAL_LPT,
        DENSE_PAIRED=DENSE_PAIRED,
    )
    m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32)
    qk_scale = sm_scale * _RCP_LN2
    gauge_cnt -= gauge_cnt
    cid = 0
    offs_m = (start_m * EFFECTIVE_BLOCK_M) + (cluster_cta_rank * BLOCK_M_SPLIT +
                                                 tl.arange(0, BLOCK_M_SPLIT))
    if CERTIFIED_CAUSAL:
        m_i, l_i, accum_cnt_qk, gauge_cnt = _fwd_softmax_tile_1cta(
            qk_fulls, qk_tiles, p_fulls, p_tiles, alpha_empties, alpha_fulls, alpha_tiles, cid,
            accum_cnt_qk, gauge_cnt, qk_scale, offs_m, m_i, l_i, start_m,
            N_CTX, tlx.dtype_of(desc_v), BLOCK_M, BLOCK_N, NUM_MMA_SLICES,
            STAGE=3, RESCALE_OPT=False, SCALAR_N=1, FAST_FIXED=True,
            SKIP_CAUSAL_DIAG=STAGE == 3 and (NUM_PID_M_STATIC <= 16 or NUM_PID_M_STATIC == 32) and BLOCK_N * 2 <= BLOCK_M,
            CERTIFIED_CAUSAL=True, MERGED_STAGE=True,
        )
    if not CERTIFIED_CAUSAL and STAGE & 1:
        m_i, l_i, accum_cnt_qk, gauge_cnt = _fwd_softmax_tile_2cta(
            qk_fulls,
            qk_tiles,
            p_fulls,
            p_tiles,
            cid,
            accum_cnt_qk,
            gauge_cnt,
            qk_scale,
            m_i,
            l_i,
            start_m,
            N_CTX,
            tlx.dtype_of(desc_v),
            BLOCK_M,
            BLOCK_N,
            NUM_MMA_SLICES,
            STAGE=4 - STAGE,
            DENSE_PAIRED=DENSE_PAIRED,
        )
    if not CERTIFIED_CAUSAL and STAGE & 2:
        m_i, l_i, accum_cnt_qk, gauge_cnt = _fwd_softmax_tile_2cta(
            qk_fulls,
            qk_tiles,
            p_fulls,
            p_tiles,
            cid,
            accum_cnt_qk,
            gauge_cnt,
            qk_scale,
            m_i,
            l_i,
            start_m,
            N_CTX,
            tlx.dtype_of(desc_v),
            BLOCK_M,
            BLOCK_N,
            NUM_MMA_SLICES,
            STAGE=2,
            DENSE_PAIRED=DENSE_PAIRED,
        )

    if CERTIFIED_CAUSAL and (NUM_PID_M_STATIC == 32 or NUM_PID_M_STATIC == 64):
        tlx.clc_producer(clc_context, clc_phase_producer)
    if not CERTIFIED_CAUSAL:
        tlx.barrier_arrive(qk_empties[cid], 1, remote_cta_rank=0)
    _, phase = get_bufidx_phase(tile_count, 1)
    tlx.barrier_wait(acc_empties[cid], phase)
    tlx.barrier_wait(o_empties[cid], phase ^ 1)
    scale = (1 / l_i)[:, None]
    if CERTIFIED_CAUSAL:
        bad_rows = ~((gauge_cnt <= 128.0) & (l_i > 0.0) & (l_i <= N_CTX) & (tl.abs(m_i * qk_scale) <= 32.0))
    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
        subslice = tlx.subslice(
            acc_tiles[cid],
            HEAD_DIM * slice_id // NUM_MMA_SLICES,
            HEAD_DIM // NUM_MMA_SLICES,
        )
        acc = tlx.local_load(subslice)
        if CERTIFIED_CAUSAL:
            bad_rows |= ~(tl.reduce(tl.abs(acc), 1, _certificate_maximum_nan) < float("inf"))
        acc = _mul_f32x2(acc, scale)
        acc = acc.to(tlx.dtype_of(desc_o))
        subslice_o = tlx.local_slice(
            o_tiles[cid],
            [0, HEAD_DIM * slice_id // NUM_MMA_SLICES],
            [BLOCK_M_SPLIT, HEAD_DIM // NUM_MMA_SLICES],
        )
        tlx.local_store(subslice_o, acc)
    if CERTIFIED_CAUSAL:
        tl.store(Failure + 2 * (off_hz * (N_CTX // BLOCK_M) + start_m) + cid,
                 (tl.sum(bad_rows.to(tl.int32), 0) != 0).to(tl.int32))
    tlx.fence("async_shared")
    if CERTIFIED_CAUSAL:
        tlx.barrier_arrive(qk_empties[cid])
    tlx.barrier_arrive(o_fulls[cid])
    if CERTIFIED_CAUSAL:
        saved_m = m_i * sm_scale * _RCP_LN2 + tl.math.log2(l_i)
    else:
        saved_m = 1.0 / l_i if PIPELINED else _BF16_FIXED_GAUGE + tl.math.log2(l_i)
    tl.store(M + off_hz * N_CTX + offs_m, saved_m)
    return accum_cnt_qk, gauge_cnt


@triton.jit
def _fwd_softmax_stages(
    qk_fulls,
    qk_tiles,
    p_fulls,
    p_tiles,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    acc_fulls,
    acc_tiles,
    cid,
    accum_cnt_qk,
    gauge_cnt,
    qk_scale,
    offs_m,
    m_i,
    l_i,
    start_m,
    N_CTX,
    out_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    STAGE: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
    SCALAR_N: tl.constexpr,
    USE_2CTA: tl.constexpr,
    FAST_FIXED: tl.constexpr,
    SKIP_CAUSAL_DIAG: tl.constexpr,
    CERTIFIED_CAUSAL: tl.constexpr = False,
    MERGED_RECOVERY: tl.constexpr = False,
    DENSE_PAIRED: tl.constexpr = False,
):
    if (CERTIFIED_CAUSAL or MERGED_RECOVERY) and STAGE == 3:
        return _fwd_softmax_tile_1cta(
            qk_fulls, qk_tiles, p_fulls, p_tiles, alpha_empties, alpha_fulls, alpha_tiles, cid,
            accum_cnt_qk, gauge_cnt, qk_scale, offs_m, m_i, l_i, start_m,
            N_CTX, out_dtype, BLOCK_M, BLOCK_N, NUM_MMA_SLICES,
            STAGE=3, RESCALE_OPT=RESCALE_OPT, SCALAR_N=SCALAR_N, FAST_FIXED=FAST_FIXED,
            SKIP_CAUSAL_DIAG=SKIP_CAUSAL_DIAG, CERTIFIED_CAUSAL=CERTIFIED_CAUSAL, MERGED_STAGE=True,
        )
    for stage_bit in tl.static_range(1, 3):
        if STAGE & stage_bit:
            m_i, l_i, accum_cnt_qk, gauge_cnt = _fwd_softmax_tile(
                qk_fulls,
                qk_tiles,
                p_fulls,
                p_tiles,
                alpha_empties,
                alpha_fulls,
                alpha_tiles,
                acc_fulls,
                acc_tiles,
                cid,
                accum_cnt_qk,
                gauge_cnt,
                qk_scale,
                offs_m,
                m_i,
                l_i,
                start_m,
                N_CTX,
                out_dtype,
                BLOCK_M,
                BLOCK_N,
                HEAD_DIM,
                NUM_MMA_SLICES,
                STAGE=4 - STAGE if stage_bit == 1 else 2,
                RESCALE_OPT=RESCALE_OPT,
                SCALAR_N=SCALAR_N,
                USE_2CTA=USE_2CTA,
                FAST_FIXED=FAST_FIXED,
                SKIP_CAUSAL_DIAG=SKIP_CAUSAL_DIAG,
                CERTIFIED_CAUSAL=CERTIFIED_CAUSAL,
                DENSE_PAIRED=DENSE_PAIRED,
            )
    return m_i, l_i, accum_cnt_qk, gauge_cnt


@triton.jit
def _fwd_control_tile(
    tile_id,
    tile_count,
    accum_cnt,
    sm_scale,
    M,
    H,
    num_pid_n,
    num_pid_in_group,
    N_CTX,
    desc_o,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    acc_empties,
    acc_fulls,
    acc_tiles,
    l_fulls,
    l_tiles,
    m_tiles,
    qk_empties,
    o_empties,
    o_fulls,
    o_tiles,
    cluster_cta_rank,
    BLOCK_M_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EFFECTIVE_BLOCK_M: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    NUM_PID_M_STATIC: tl.constexpr,
    GRID_X_STATIC: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    NUM_GROUPS_PER_CTA: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
    SCALAR_N: tl.constexpr,
    STAGE: tl.constexpr,
    USE_2CTA: tl.constexpr,
    USE_WHERE: tl.constexpr,
    SKIP_RESCALE,
    FUSE_EPILOG,
    GLOBAL_LPT: tl.constexpr,
    LAYOUT_BSHD: tl.constexpr = False,
):
    start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
        tile_id // NUM_CTAS,
        H,
        num_pid_n,
        num_pid_in_group,
        N_CTX,
        EFFECTIVE_BLOCK_M,
        STAGE,
        GROUP_SIZE_N,
        NUM_PID_M_STATIC,
        GRID_X_STATIC,
        GLOBAL_LPT,
    )
    qo_offset_y, kv_offset_y, offset_x = _forward_descriptor_offsets(
        qo_offset_y, kv_offset_y, off_hz, H, N_CTX, HEAD_DIM, LAYOUT_BSHD)
    control_lo = hi if SKIP_RESCALE else lo
    # Rescale the live output accumulator when the online-softmax gauge changes.
    for _ in tl.range(control_lo, hi, BLOCK_N):
        _, phase = get_bufidx_phase(accum_cnt, 1)
        for cid in tl.static_range(0, NUM_GROUPS_PER_CTA):
            tlx.barrier_wait(alpha_fulls[cid], phase)
            alpha_loaded = tlx.local_load(alpha_tiles[cid])
            alpha_1 = tl.split(alpha_loaded)[0][:, None] if SCALAR_N == 2 else alpha_loaded
            tlx.barrier_arrive(alpha_empties[cid])
            if RESCALE_OPT:
                # One warp vote skips the whole accumulator update when no lane
                # changed its online-softmax gauge.
                pred = alpha_1 < 1.0
                ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
                should_rescale = ballot_result != 0

            if USE_WHERE:
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    subslice = tlx.subslice(
                        acc_tiles[cid],
                        HEAD_DIM * slice_id // NUM_MMA_SLICES,
                        HEAD_DIM // NUM_MMA_SLICES,
                    )
                    acc = tlx.local_load(subslice)
                    if RESCALE_OPT:
                        scaled_acc = _mul_f32x2(acc, alpha_1)
                        acc = tl.where(should_rescale, scaled_acc, acc)
                    else:
                        acc = _mul_f32x2(acc, alpha_1)
                    tlx.local_store(subslice, acc)
            else:
                if RESCALE_OPT:
                    should_rescale_red = tl.reduce(should_rescale, axis=0, combine_fn=_reduce_or)
                    should_rescale_scalar = tl.reshape(should_rescale_red, ())
                if not RESCALE_OPT or (RESCALE_OPT and should_rescale_scalar):
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        subslice = tlx.subslice(
                            acc_tiles[cid],
                            HEAD_DIM * slice_id // NUM_MMA_SLICES,
                            HEAD_DIM // NUM_MMA_SLICES,
                        )
                        acc = tlx.local_load(subslice)
                        acc = _mul_f32x2(acc, alpha_1)
                        tlx.local_store(subslice, acc)
            if USE_2CTA:
                tlx.barrier_arrive(acc_fulls[cid], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(acc_fulls[cid])
        accum_cnt += 1

    _, phase = get_bufidx_phase(tile_count, 1)
    for cid in tl.static_range(0, NUM_GROUPS_PER_CTA):
        group_id = cid * NUM_CTAS + cluster_cta_rank
        # l and m share a synchronization group, so release QK only after both loads.
        tlx.barrier_wait(l_fulls[cid], phase)
        l_loaded = tlx.local_load(l_tiles[cid])
        m_loaded = tlx.local_load(m_tiles[cid])
        l = tl.split(l_loaded)[0][:, None] if SCALAR_N == 2 else l_loaded
        m = tl.split(m_loaded)[0][:, None] if SCALAR_N == 2 else m_loaded
        if USE_2CTA:
            tlx.barrier_arrive(qk_empties[cid], 1, remote_cta_rank=0)
        else:
            tlx.barrier_arrive(qk_empties[cid])
        if RESCALE_OPT:
            m = m * sm_scale * _RCP_LN2
        m += tl.math.log2(l)
        offs_m = start_m * EFFECTIVE_BLOCK_M + group_id * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]), mask=offs_m < N_CTX)

        # Normalize the completed accumulator into the output staging tile;
        # the epilog task owns publication to global memory.
        tlx.barrier_wait(acc_empties[cid], phase)
        tlx.barrier_wait(o_empties[cid], phase ^ 1)
        scale = 1 / l
        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
            subslice = tlx.subslice(
                acc_tiles[cid],
                HEAD_DIM * slice_id // NUM_MMA_SLICES,
                HEAD_DIM // NUM_MMA_SLICES,
            )
            acc = tlx.local_load(subslice)
            acc = _mul_f32x2(acc, scale)
            acc = acc.to(tlx.dtype_of(desc_o))
            subslice_o = tlx.local_slice(
                o_tiles[cid],
                [0, HEAD_DIM * slice_id // NUM_MMA_SLICES],
                [BLOCK_M_SPLIT, HEAD_DIM // NUM_MMA_SLICES],
            )
            tlx.local_store(subslice_o, acc)
        if FUSE_EPILOG:
            qo_offset_y_split = qo_offset_y + group_id * BLOCK_M_SPLIT
            tlx.async_descriptor_store(
                desc_o, o_tiles[cid], [qo_offset_y_split, offset_x], eviction_policy="evict_first")
            tlx.async_descriptor_store_wait(0)
            tlx.barrier_arrive(o_empties[cid])
        else:
            tlx.barrier_arrive(o_fulls[cid])
    return accum_cnt


@triton.jit
def _fwd_load_first_k_tile(
    accum_cnt_k,
    lo,
    hi,
    kv_offset_y,
    desc_k,
    k_tiles,
    k_fulls,
    k_empties,
    K_BYTES_PER_ELEM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    cluster_cta_rank,
    NUM_CTAS: tl.constexpr,
    offset_x=0,
):
    buf, phase = get_bufidx_phase(accum_cnt_k, NUM_BUFFERS_KV)
    tlx.barrier_wait(k_empties[buf], phase ^ 1)
    if cluster_cta_rank == 0:
        tlx.barrier_expect_bytes(k_fulls[buf], K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
    tlx.async_descriptor_load(
        desc_k,
        k_tiles[buf],
        [kv_offset_y + cluster_cta_rank * (BLOCK_N // NUM_CTAS), offset_x],
        k_fulls[buf],
        two_ctas=True,
    )
    return accum_cnt_k + (hi - lo) // BLOCK_N


@triton.jit
def _fwd_load_tile(
    tile_count,
    accum_cnt_kv,
    lo,
    hi,
    qo_offset_y,
    kv_offset_y,
    desc_q,
    desc_k,
    desc_v,
    q_tiles,
    kv_tiles,
    q_fulls,
    q_empties,
    kv_fulls,
    kv_empties,
    Q_BYTES_PER_ELEM: tl.constexpr,
    K_BYTES_PER_ELEM: tl.constexpr,
    V_BYTES_PER_ELEM: tl.constexpr,
    BLOCK_M_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    cluster_cta_rank=0,
    v_tiles=None,
    v_fulls=None,
    v_empties=None,
    NUM_GROUPS_PER_CTA: tl.constexpr = 2,
    NUM_CTAS: tl.constexpr = 1,
    SKIP_FIRST_K: tl.constexpr = False,
    offset_x=0,
):
    USE_2CTA: tl.constexpr = NUM_CTAS == 2
    # load q0
    if USE_2CTA:
        q_bufIdx, q_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_Q)
        for group_id in tl.static_range(0, NUM_GROUPS_PER_CTA):
            q_id = q_bufIdx + group_id * NUM_BUFFERS_Q
            tlx.barrier_wait(q_empties[q_id], q_phase ^ 1)
            if cluster_cta_rank == 0:
                tlx.barrier_expect_bytes(q_fulls[q_id], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM * NUM_CTAS)
            tlx.async_descriptor_load(
                desc_q,
                q_tiles[q_id],
                [qo_offset_y + (group_id * NUM_CTAS + cluster_cta_rank) * BLOCK_M_SPLIT, offset_x],
                q_fulls[q_id],
                two_ctas=True,
            )
        if SKIP_FIRST_K:
            buf, phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
            tlx.barrier_wait(v_empties[buf], phase ^ 1)
            if cluster_cta_rank == 0:
                tlx.barrier_expect_bytes(v_fulls[buf], V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
            tlx.async_descriptor_load(
                desc_v,
                v_tiles[buf],
                [kv_offset_y, offset_x + cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
                v_fulls[buf],
                two_ctas=True,
            )
            kv_offset_y += BLOCK_N
            accum_cnt_kv += 1

        loop_start = lo + BLOCK_N if SKIP_FIRST_K else lo
        for _ in tl.range(loop_start, hi, BLOCK_N):
            buf, phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
            tlx.barrier_wait(kv_empties[buf], phase ^ 1)
            tlx.barrier_wait(v_empties[buf], phase ^ 1)
            if cluster_cta_rank == 0:
                tlx.barrier_expect_bytes(kv_fulls[buf], K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                tlx.barrier_expect_bytes(v_fulls[buf], V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
            tlx.async_descriptor_load(
                desc_k,
                kv_tiles[buf],
                [kv_offset_y + cluster_cta_rank * (BLOCK_N // NUM_CTAS), offset_x],
                kv_fulls[buf],
                two_ctas=True,
            )
            tlx.async_descriptor_load(
                desc_v,
                v_tiles[buf],
                [kv_offset_y, offset_x + cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
                v_fulls[buf],
                two_ctas=True,
            )
            kv_offset_y += BLOCK_N
            accum_cnt_kv += 1
        return accum_cnt_kv
    q_bufIdx, q_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_Q)
    tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
    tlx.barrier_expect_bytes(q_fulls[q_bufIdx], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
    qo_offset_y_split = qo_offset_y
    tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, offset_x], q_fulls[q_bufIdx])

    # loop over loading k, v
    k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
    k_empty = tlx.local_view(kv_empties, k_bufIdx)
    tlx.barrier_wait(k_empty, k_phase ^ 1)

    # load K
    k_full = tlx.local_view(kv_fulls, k_bufIdx)
    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
    tlx.barrier_expect_bytes(k_full, K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
    tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, offset_x], k_full)

    # load q1
    q_bufIdx += NUM_BUFFERS_Q
    tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
    tlx.barrier_expect_bytes(q_fulls[q_bufIdx], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
    qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
    tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, offset_x], q_fulls[q_bufIdx])

    v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
    v_empty = tlx.local_view(kv_empties, v_bufIdx)
    tlx.barrier_wait(v_empty, v_phase ^ 1)
    # load V
    v_full = tlx.local_view(kv_fulls, v_bufIdx)
    v_tile = tlx.local_view(kv_tiles, v_bufIdx)
    tlx.barrier_expect_bytes(v_full, V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
    tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, offset_x], v_full)

    kv_offset_y += BLOCK_N
    accum_cnt_kv += 2

    for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
        k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
        k_empty = tlx.local_view(kv_empties, k_bufIdx)
        tlx.barrier_wait(k_empty, k_phase ^ 1)
        # load K
        k_full = tlx.local_view(kv_fulls, k_bufIdx)
        k_tile = tlx.local_view(kv_tiles, k_bufIdx)
        tlx.barrier_expect_bytes(k_full, K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
        tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, offset_x], k_full)

        v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
        v_empty = tlx.local_view(kv_empties, v_bufIdx)
        tlx.barrier_wait(v_empty, v_phase ^ 1)
        # load V
        v_full = tlx.local_view(kv_fulls, v_bufIdx)
        v_tile = tlx.local_view(kv_tiles, v_bufIdx)
        tlx.barrier_expect_bytes(v_full, V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
        tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, offset_x], v_full)

        kv_offset_y += BLOCK_N
        accum_cnt_kv += 2

    return accum_cnt_kv


@triton.jit
def _fwd_pv_32_32_64(p0, p1, b0, b1, b2, phase, v, acc, use_acc, done, HEAD_DIM: tl.constexpr):
    tlx.barrier_wait(b0, phase)
    tlx.async_dot(tlx.local_slice(p0, [0, 0], [128, 32]), tlx.local_slice(v, [0, 0], [32, HEAD_DIM // 2]), acc,
                  use_acc=use_acc, force_async=True, two_ctas=True)
    tlx.barrier_wait(b1, phase)
    tlx.async_dot(tlx.local_slice(p0, [0, 32], [128, 32]), tlx.local_slice(v, [32, 0], [32, HEAD_DIM // 2]), acc,
                  use_acc=True, force_async=True, two_ctas=True)
    tlx.barrier_wait(b2, phase)
    tlx.async_dot(p1, tlx.local_slice(v, [64, 0], [64, HEAD_DIM // 2]), acc, use_acc=True, mBarriers=done,
                  force_async=True, two_ctas=True)


@triton.jit
def _fwd_mma_tile(
    tile_count,
    accum_cnt_kv,
    accum_cnt_qk,
    lo,
    hi,
    q_tiles,
    kv_tiles,
    qk_tiles,
    p_tiles,
    acc_tiles,
    q_fulls,
    q_empties,
    kv_fulls,
    kv_empties,
    qk_fulls,
    qk_empties,
    p_fulls,
    acc_fulls,
    acc_empties,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_P_SLICES: tl.constexpr,
    v_tiles=None,
    v_fulls=None,
    v_empties=None,
    USE_2CTA: tl.constexpr = False,
    SKIP_RESCALE: tl.constexpr = False,
    SKIP_CAUSAL_Q0_LAST: tl.constexpr = False,
):
    v_tiles = v_tiles if USE_2CTA else kv_tiles
    v_fulls = v_fulls if USE_2CTA else kv_fulls
    v_empties = v_empties if USE_2CTA else kv_empties
    q_bufIdx, q_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_Q)
    k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
    v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv if USE_2CTA else accum_cnt_kv + 1, NUM_BUFFERS_KV)

    # wait for the K buffer to be populated by the producer
    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)

    # wait for the Q buffer to be populated by the producer
    tlx.barrier_wait(q_fulls[q_bufIdx], q_phase)

    # -- compute q0 @ k ----
    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
    tlx.barrier_wait(qk_empties[0], q_phase ^ 1)
    tlx.async_dot(
        q_tiles[0],
        k_tile,
        qk_tiles[0],
        use_acc=False,
        mBarriers=[qk_fulls[0]],
        two_ctas=USE_2CTA,
    )
    if USE_2CTA and SKIP_RESCALE and lo + BLOCK_N >= hi:
        tlx.tcgen05_commit(q_empties[q_bufIdx], two_ctas=USE_2CTA)

    # -- compute q1 @ k ----
    tlx.barrier_wait(q_fulls[q_bufIdx + NUM_BUFFERS_Q], q_phase)
    tlx.barrier_wait(qk_empties[1], q_phase ^ 1)
    tlx.async_dot(
        q_tiles[1],
        k_tile,
        qk_tiles[1],
        use_acc=False,
        mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
        two_ctas=USE_2CTA,
    )
    if USE_2CTA and SKIP_RESCALE and lo + BLOCK_N >= hi:
        tlx.tcgen05_commit(q_empties[q_bufIdx + NUM_BUFFERS_Q], two_ctas=USE_2CTA)

    _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)

    # -- compute p0 @ v ----
    # wait for the V buffer to be populated by the producer
    tlx.barrier_wait(v_fulls[v_bufIdx], v_phase)
    if USE_2CTA:
        if not SKIP_RESCALE:
            tlx.barrier_wait(acc_fulls[0], qk_phase)
        _fwd_pv_32_32_64(
            p_tiles[0],
            p_tiles[1],
            p_fulls[0],
            p_fulls[1],
            p_fulls[2],
            qk_phase,
            v_tiles[v_bufIdx],
            acc_tiles[0],
            False,
            [],
            HEAD_DIM_KV * 2,
        )
    else:
        if not SKIP_RESCALE:
            tlx.barrier_wait(acc_fulls[0], qk_phase)
        for slice_id in tl.static_range(0, NUM_P_SLICES):
            p_bufIdx = slice_id
            tlx.barrier_wait(p_fulls[p_bufIdx], qk_phase)
            kv_slice = tlx.local_slice(
                v_tiles[v_bufIdx],
                [BLOCK_N * slice_id // NUM_P_SLICES, 0],
                [BLOCK_N // NUM_P_SLICES, HEAD_DIM_KV],
            )
            tlx.async_dot(
                p_tiles[p_bufIdx],
                kv_slice,
                acc_tiles[0],
                use_acc=slice_id > 0,
                force_async=True,
                two_ctas=USE_2CTA,
            )

    acc1_init = False

    for i in tl.range(lo + BLOCK_N, hi, BLOCK_N):
        v_bufIdx_prev = v_bufIdx
        qk_phase_prev = qk_phase

        accum_cnt_qk += 1
        accum_cnt_kv += 1 if USE_2CTA else 2
        k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
        v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv if USE_2CTA else accum_cnt_kv + 1, NUM_BUFFERS_KV)

        # -- compute q0 @ k ----
        # wait for the K buffer to be populated by the producer
        tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
        k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
        _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)

        skip_q0 = SKIP_CAUSAL_Q0_LAST and i + BLOCK_N >= hi
        if skip_q0:
            tlx.tcgen05_commit(qk_fulls[0], two_ctas=USE_2CTA)
        else:
            tlx.async_dot(
                q_tiles[0],
                k_tile,
                qk_tiles[0],
                use_acc=False,
                mBarriers=[qk_fulls[0]],
                two_ctas=USE_2CTA,
            )
        if USE_2CTA and SKIP_RESCALE and i + BLOCK_N >= hi:
            tlx.tcgen05_commit(q_empties[q_bufIdx], two_ctas=USE_2CTA)

        # -- compute p1 @ v from the previous iteration----
        if USE_2CTA:
            if not SKIP_RESCALE:
                tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
            _fwd_pv_32_32_64(
                p_tiles[2],
                p_tiles[3],
                p_fulls[3],
                p_fulls[4],
                p_fulls[5],
                qk_phase_prev,
                v_tiles[v_bufIdx_prev],
                acc_tiles[1],
                acc1_init,
                [v_empties[v_bufIdx_prev]],
                HEAD_DIM_KV * 2,
            )
        else:
            if not SKIP_RESCALE:
                tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
            for slice_id in tl.static_range(0, NUM_P_SLICES):
                p_bufIdx = slice_id + NUM_P_SLICES
                tlx.barrier_wait(p_fulls[p_bufIdx], qk_phase_prev)
                kv_slice = tlx.local_slice(
                    v_tiles[v_bufIdx_prev],
                    [BLOCK_N * slice_id // NUM_P_SLICES, 0],
                    [BLOCK_N // NUM_P_SLICES, HEAD_DIM_KV],
                )
                use_acc = acc1_init if slice_id == 0 else True
                mBarriers = [v_empties[v_bufIdx_prev]] if slice_id == NUM_P_SLICES - 1 else []
                tlx.async_dot(
                    p_tiles[p_bufIdx],
                    kv_slice,
                    acc_tiles[1],
                    use_acc=use_acc,
                    mBarriers=mBarriers,
                    force_async=SKIP_RESCALE,
                    two_ctas=USE_2CTA,
                )

        acc1_init = True

        # -- compute q1 @ k ----
        tlx.async_dot(
            q_tiles[1],
            k_tile,
            qk_tiles[1],
            use_acc=False,
            mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
            two_ctas=USE_2CTA,
        )
        if USE_2CTA and SKIP_RESCALE and i + BLOCK_N >= hi:
            tlx.tcgen05_commit(q_empties[q_bufIdx + NUM_BUFFERS_Q], two_ctas=USE_2CTA)

        # -- compute p0 @ v ----
        # wait for the V buffer to be populated by the producer
        tlx.barrier_wait(v_fulls[v_bufIdx], v_phase)

        if USE_2CTA:
            if not SKIP_RESCALE:
                tlx.barrier_wait(acc_fulls[0], qk_phase)
            _fwd_pv_32_32_64(
                p_tiles[0],
                p_tiles[1],
                p_fulls[0],
                p_fulls[1],
                p_fulls[2],
                qk_phase,
                v_tiles[v_bufIdx],
                acc_tiles[0],
                True,
                [],
                HEAD_DIM_KV * 2,
            )
        else:
            if not SKIP_RESCALE:
                tlx.barrier_wait(acc_fulls[0], qk_phase)
            for slice_id in tl.static_range(0, NUM_P_SLICES):
                p_bufIdx = slice_id
                tlx.barrier_wait(p_fulls[p_bufIdx], qk_phase)
                kv_slice = tlx.local_slice(
                    v_tiles[v_bufIdx],
                    [BLOCK_N * slice_id // NUM_P_SLICES, 0],
                    [BLOCK_N // NUM_P_SLICES, HEAD_DIM_KV],
                )
                if not skip_q0:
                    tlx.async_dot(
                        p_tiles[p_bufIdx],
                        kv_slice,
                        acc_tiles[0],
                        use_acc=True,
                        force_async=True,
                        two_ctas=USE_2CTA,
                    )

    if not SKIP_RESCALE or not USE_2CTA:
        tlx.tcgen05_commit(q_empties[q_bufIdx], two_ctas=USE_2CTA)
        tlx.tcgen05_commit(q_empties[q_bufIdx + NUM_BUFFERS_Q], two_ctas=USE_2CTA)
    tlx.tcgen05_commit(acc_empties[0], two_ctas=USE_2CTA)

    # -- compute p1 @ v ----
    if USE_2CTA:
        if not SKIP_RESCALE:
            tlx.barrier_wait(acc_fulls[1], qk_phase)
        _fwd_pv_32_32_64(
            p_tiles[2],
            p_tiles[3],
            p_fulls[3],
            p_fulls[4],
            p_fulls[5],
            qk_phase,
            v_tiles[v_bufIdx],
            acc_tiles[1],
            acc1_init,
            [acc_empties[1], v_empties[v_bufIdx]],
            HEAD_DIM_KV * 2,
        )
    else:
        if not SKIP_RESCALE:
            tlx.barrier_wait(acc_fulls[1], qk_phase)
        for slice_id in tl.static_range(0, NUM_P_SLICES):
            p_bufIdx = slice_id + NUM_P_SLICES
            tlx.barrier_wait(p_fulls[p_bufIdx], qk_phase)
            kv_slice = tlx.local_slice(
                v_tiles[v_bufIdx],
                [BLOCK_N * slice_id // NUM_P_SLICES, 0],
                [BLOCK_N // NUM_P_SLICES, HEAD_DIM_KV],
            )
            use_acc = acc1_init if slice_id == 0 else True
            mBarriers = [acc_empties[1], v_empties[v_bufIdx]] if slice_id == NUM_P_SLICES - 1 else []
            tlx.async_dot(
                p_tiles[p_bufIdx],
                kv_slice,
                acc_tiles[1],
                use_acc=use_acc,
                mBarriers=mBarriers,
                two_ctas=USE_2CTA,
            )

    accum_cnt_qk += 1
    accum_cnt_kv += 1 if USE_2CTA else 2
    return accum_cnt_kv, accum_cnt_qk


@triton.autotune(
    configs=configs,
    key=["N_CTX", "HEAD_DIM", "H", "STAGE", "LAYOUT_BSHD"],
    prune_configs_by={"early_config_prune": prune_configs_by_hdim},
)
@triton.jit
def _attn_fwd_ws(
    sm_scale,
    M,  #
    Z: tl.constexpr,
    H: tl.constexpr,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_BUFFERS_Q: tl.constexpr,  #
    NUM_BUFFERS_KV: tl.constexpr,  #
    NUM_BUFFERS_QK: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    NUM_MMA_SLICES: tl.constexpr,  #
    GROUP_SIZE_N: tl.constexpr,  #
    RESCALE_OPT: tl.constexpr,  #
    USE_WHERE: tl.constexpr,  #
    USE_WARP_BARRIER: tl.constexpr,  #
    NUM_CTAS: tl.constexpr = 1,  #
    PIPELINED: tl.constexpr = False,
    POLICY: tl.constexpr = POLICY_DENSE,
    DENSE_REGS: tl.constexpr = 200,
    FAST_FIXED: tl.constexpr = True,
    NUM_PID_M_STATIC: tl.constexpr = 1,
    GRID_X_STATIC: tl.constexpr = 1,
    COMPACT_CLC: tl.constexpr = False,
    GLOBAL_LPT: tl.constexpr = False,
    LAYOUT_BSHD: tl.constexpr = False,
    N_CTX_STATIC: tl.constexpr = 0,
):
    _attn_fwd_ws_kernel(sm_scale, M, Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, STAGE,
                        NUM_BUFFERS_Q, NUM_BUFFERS_KV, NUM_BUFFERS_QK, NUM_MMA_GROUPS, NUM_MMA_SLICES, GROUP_SIZE_N,
                        RESCALE_OPT, USE_WHERE, USE_WARP_BARRIER, NUM_CTAS, PIPELINED, POLICY, DENSE_REGS, FAST_FIXED,
                        NUM_PID_M_STATIC, GRID_X_STATIC, COMPACT_CLC, GLOBAL_LPT, LAYOUT_BSHD=LAYOUT_BSHD, N_CTX_STATIC=N_CTX_STATIC)


@triton.jit
def _attn_fwd_ws_kernel(
    sm_scale,
    M,  #
    Z: tl.constexpr,
    H: tl.constexpr,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_BUFFERS_Q: tl.constexpr,  #
    NUM_BUFFERS_KV: tl.constexpr,  #
    NUM_BUFFERS_QK: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    NUM_MMA_SLICES: tl.constexpr,  #
    GROUP_SIZE_N: tl.constexpr,  #
    RESCALE_OPT: tl.constexpr,  #
    USE_WHERE: tl.constexpr,  #
    USE_WARP_BARRIER: tl.constexpr,  #
    NUM_CTAS: tl.constexpr = 1,  #
    PIPELINED: tl.constexpr = False,
    POLICY: tl.constexpr = POLICY_DENSE,
    DENSE_REGS: tl.constexpr = 200,
    FAST_FIXED: tl.constexpr = True,
    NUM_PID_M_STATIC: tl.constexpr = 1,
    GRID_X_STATIC: tl.constexpr = 1,
    COMPACT_CLC: tl.constexpr = False,
    GLOBAL_LPT: tl.constexpr = False,
    N_CTX_STATIC: tl.constexpr = 0,
    Failure=None,
    LAYOUT_BSHD: tl.constexpr = False,
    SPARSE_FALLBACK: tl.constexpr = False,
    Recovery=None,
):
    _RCP_LN2: tl.constexpr = 1.4426950408889634
    _BF16_FIXED_GAUGE: tl.constexpr = 4.055517269
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(NUM_BUFFERS_QK == 1)
    tl.static_assert(NUM_BUFFERS_Q == 1)

    USE_2CTA: tl.constexpr = NUM_CTAS == 2
    tl.static_assert(not USE_2CTA or BLOCK_M == 256)
    FAST_F16_CAPABLE: tl.constexpr = (tlx.dtype_of(desc_v) == tl.float16 and NUM_MMA_SLICES == 4 and not RESCALE_OPT
                                      and not USE_2CTA)
    FAST_2CTA_CAPABLE: tl.constexpr = (
        USE_2CTA
        and PIPELINED
        and tlx.dtype_of(desc_v) == tl.bfloat16
        and NUM_MMA_SLICES == 2
        and not RESCALE_OPT
    )
    CERTIFIED_CAUSAL: tl.constexpr = Failure is not None
    if CERTIFIED_CAUSAL:
        tl.static_assert(FAST_FIXED and not RESCALE_OPT and not PIPELINED and STAGE == 3 and NUM_CTAS == 1)
        tl.static_assert(tlx.dtype_of(desc_q) == tl.bfloat16 and tlx.dtype_of(desc_k) == tl.bfloat16
                         and tlx.dtype_of(desc_v) == tl.bfloat16)
        tl.static_assert((HEAD_DIM == 128 or (HEAD_DIM == 64 and Z == 4 and H == 48
                          and (N_CTX_STATIC == 2048 or N_CTX_STATIC == 4096 or N_CTX_STATIC == 8192
                               or N_CTX_STATIC == 16384 or N_CTX_STATIC == 32768) and not LAYOUT_BSHD))
                         and BLOCK_M == 256 and BLOCK_N == 128 and NUM_MMA_SLICES == 2)
        tl.static_assert((N_CTX_STATIC == 2048 and (
            (Z * N_CTX_STATIC == 32768 and H == 16 and LAYOUT_BSHD) or (Z == 4 and H == 48 and not LAYOUT_BSHD)))
            or (N_CTX_STATIC >= 4096 and N_CTX_STATIC <= 32768 and N_CTX_STATIC % 512 == 0))
        tl.static_assert(tlx.num_warps() == 4)
    USE_FAST_FIXED: tl.constexpr = FAST_FIXED and (FAST_F16_CAPABLE or FAST_2CTA_CAPABLE or CERTIFIED_CAUSAL)
    USE_FAST_F16: tl.constexpr = FAST_FIXED and FAST_F16_CAPABLE
    DENSE_EXCLUSIVE: tl.constexpr = (
        (N_CTX_STATIC == 1024 or N_CTX_STATIC == 2048 or N_CTX_STATIC == 4096
             or N_CTX_STATIC == 8192 or N_CTX_STATIC == 16384 or N_CTX_STATIC == 32768)
        and N_CTX_STATIC == NUM_PID_M_STATIC * 512
        and USE_2CTA and USE_FAST_FIXED and STAGE == 1
        and HEAD_DIM == 128 and BLOCK_N == 128 and NUM_BUFFERS_KV == 3 and DENSE_REGS == 176
    )
    DENSE_PAIRED: tl.constexpr = DENSE_EXCLUSIVE and N_CTX_STATIC != 32768
    SKIP_CAUSAL_DIAG: tl.constexpr = (
        STAGE == 3
        and (NUM_PID_M_STATIC <= 16 or (CERTIFIED_CAUSAL and NUM_PID_M_STATIC == 32))
        and BLOCK_N * 2 <= BLOCK_M
    )
    FUSE_EPILOG: tl.constexpr = USE_FAST_F16
    DIRECT_SCHED: tl.constexpr = USE_2CTA or USE_FAST_F16 or SPARSE_FALLBACK
    if SPARSE_FALLBACK:
        tl.static_assert(NUM_CTAS == 1 and STAGE == 3 and GROUP_SIZE_N == 1)
        tl.static_assert(not COMPACT_CLC and not CERTIFIED_CAUSAL and RESCALE_OPT)
    USE_GRID_LPT: tl.constexpr = (
        STAGE == 3
        and COMPACT_CLC
        and GROUP_SIZE_N > 4
    )
    LPT_GRID_X: tl.constexpr = (
        GRID_X_STATIC
        if USE_GRID_LPT and GRID_X_STATIC <= GROUP_SIZE_N
        else min(GROUP_SIZE_N, GRID_X_STATIC & -GRID_X_STATIC)
        if USE_GRID_LPT
        else 1
    )
    OFFSET_GRID_X: tl.constexpr = LPT_GRID_X if USE_GRID_LPT else 0
    cluster_cta_rank = tlx.cluster_cta_rank() if USE_2CTA else 0
    is_leader = (not USE_2CTA) or (cluster_cta_rank % 2 == 0)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2
    tl.static_assert(NUM_MMA_GROUPS == 2)
    NUM_GROUPS_PER_CTA: tl.constexpr = NUM_MMA_GROUPS
    # Per-CTA head-dim of the multicast K/V (B) operands: the collective 2-CTA
    # MMA reassembles the full HEAD_DIM contraction from both CTAs' halves.
    HEAD_DIM_KV: tl.constexpr = HEAD_DIM // NUM_CTAS
    # Effective M-block: 2-CTA covers BLOCK_M * NUM_CTAS total Q rows per work tile.
    EFFECTIVE_BLOCK_M: tl.constexpr = BLOCK_M * NUM_CTAS
    if DENSE_EXCLUSIVE:
        N_CTX = NUM_PID_M_STATIC * EFFECTIVE_BLOCK_M

    # Compute bytes per element for each tensor type
    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))
    qk_dtype = tl.float32

    # original grid
    #   triton.cdiv(q.shape[2], META["BLOCK_M"]),
    #   q.shape[0] * q.shape[1],
    num_pid_m = tl.cdiv(N_CTX, EFFECTIVE_BLOCK_M)
    num_pid_n = Z * H
    num_pid_in_group = num_pid_m * GROUP_SIZE_N
    num_tiles = num_pid_m * num_pid_n
    if USE_GRID_LPT:
        start_pid = tl.program_id(0) + LPT_GRID_X * tl.program_id(1)
        persistent_stride = tl.num_programs(0) * tl.num_programs(1)
    else:
        start_pid = tl.program_id(0) // NUM_CTAS
        persistent_stride = tl.num_programs(0) // NUM_CTAS
    if SPARSE_FALLBACK:
        needs_recovery = tl.load(Recovery + 2 * start_pid) | tl.load(Recovery + 2 * start_pid + 1)
        start_pid = tl.where(needs_recovery != 0, start_pid, -1)

    # allocate SMEM buffers and barriers
    NUM_Q_BUFS: tl.constexpr = NUM_GROUPS_PER_CTA * NUM_BUFFERS_Q
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_Q_BUFS)
    BLOCK_N_KV: tl.constexpr = BLOCK_N // NUM_CTAS
    if USE_2CTA:
        # Separate k_tiles and v_tiles. K loaded as (N/2, D), local_trans to (D, N/2).
        # V loaded as (N, D/2).
        k_tiles = tlx.local_alloc((BLOCK_N_KV, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
        v_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM_KV), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)
        o_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_o), NUM_GROUPS_PER_CTA)
    else:
        kv_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
        o_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_o), NUM_MMA_GROUPS)

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_Q_BUFS)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_Q_BUFS)
    if USE_2CTA:
        k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
        k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
        v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
        v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    else:
        kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
        kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    o_empties = tlx.alloc_barriers(num_barriers=NUM_GROUPS_PER_CTA)

    # Define the buffer for sharing. Offsets are currently manually specified
    # via buffer count.
    qk_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    qk_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_N), qk_dtype, NUM_MMA_GROUPS, tlx.storage_kind.tmem,
                               reuse=qk_storage_alias)
    NUM_P_SLICES: tl.constexpr = 2 if USE_FAST_F16 else NUM_MMA_SLICES
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N // NUM_P_SLICES),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * NUM_P_SLICES,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )
    # When BLOCK_M_SPLIT == 64 == blockM, the TMEM lowering selects the
    # I16x32bx2 message whose secondHalfOffset=0 hits a ptxas bug. Pad to
    # blockN=2 so secondHalfOffset is naturally non-zero.
    SCALAR_N: tl.constexpr = 2 if BLOCK_M_SPLIT == 64 else 1
    alpha_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, SCALAR_N),
        tl.float32,
        NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )
    l_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, SCALAR_N),
        tl.float32,
        NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )
    m_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, SCALAR_N),
        tl.float32,
        NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_storage_alias,
    )
    # Define the buffer reuse strategy:
    # QK is shared by (P, alpha, l, and m)
    #   - First half  : stores P
    #   - Second half  : stores Alpha, l, and m
    #   QK : |                                                   BLK_M/2 * BLOCK_N * fp32                         |
    #   P:   |  BLK_M/(2*SLICES) * fp16| BLK_M/(2*SLICES) * fp16|...
    # Alpha:                                                        |BLK_M/2*1*fp32|
    #   l  :                                                                        |BLK_M/2*1*fp32|
    #   m  :                                                                                       |BLK_M/2*1*fp32|
    qk_storage_alias.set_buffer_overlap(
        tlx.reuse_group(
            qk_tiles,
            tlx.reuse_group(
                tlx.reuse_group(p_tiles, group_size=NUM_P_SLICES),
                alpha_tiles,
                l_tiles,
                m_tiles,
                group_type=tlx.reuse_group_type.distinct,
            ),
            group_type=tlx.reuse_group_type.shared,
        ))

    acc_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem)

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    # Cross-CTA barriers: must use mbarriers for 2-CTA (arrive_count=NUM_CTAS).
    if USE_2CTA:
        qk_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=NUM_CTAS)
        p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * 3, arrive_count=NUM_CTAS)
        acc_fulls = (acc_empties
                     if USE_FAST_FIXED else tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=NUM_CTAS))
    elif USE_WARP_BARRIER:
        qk_empties = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
        p_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS * NUM_P_SLICES, num_warps=4)
        acc_fulls = (acc_empties
                     if USE_FAST_FIXED else tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4))
    else:
        qk_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=NUM_CTAS)
        p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_P_SLICES, arrive_count=NUM_CTAS)
        acc_fulls = (acc_empties
                     if USE_FAST_FIXED else tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=NUM_CTAS))
    # Alpha/l stay within the compute role and use warp barriers. The 2-CTA
    # output handoff below uses an mbarrier for the separate epilog role.
    alpha_fulls = (qk_fulls if USE_FAST_FIXED else tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4))
    alpha_empties = (qk_empties if USE_FAST_FIXED else tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4))
    l_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
    o_fulls = (tlx.alloc_barriers(
        num_barriers=NUM_MMA_GROUPS) if USE_2CTA else tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4))

    # CLC consumers per CTA: correction(1) + softmax(NUM_GROUPS_PER_CTA) + mma(1) + load(1) + epilog(1).
    if not DIRECT_SCHED:
        clc_context = tlx.clc_create_context(num_consumers=3 + NUM_GROUPS_PER_CTA if FUSE_EPILOG else 4 +
                                             NUM_GROUPS_PER_CTA - (1 if CERTIFIED_CAUSAL else 0))

    # In 2-CTA mode, cross-CTA barrier_arrive (to the leader's mbarriers) requires
    # the mbarrier.init to be visible cluster-wide before any remote arrive.
    # Must be AFTER all barrier allocations (including CLC context barriers).
    if DENSE_EXCLUSIVE or CERTIFIED_CAUSAL or SPARSE_FALLBACK:
        ws_done = tlx.alloc_barriers(num_barriers=1, arrive_count=3 + (1 if CERTIFIED_CAUSAL else NUM_GROUPS_PER_CTA))
    if USE_2CTA:
        tlx.fence_mbarrier_init_cluster()

    with tlx.async_tasks(exclusive=CERTIFIED_CAUSAL or SPARSE_FALLBACK or DENSE_EXCLUSIVE):
        # correction group
        with tlx.async_task("default"):
            accum_cnt = 0
            accum_cnt_qk = 0
            gauge_cnt = tl.zeros([BLOCK_M_SPLIT], tl.float32) if CERTIFIED_CAUSAL else 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_producer = 1
            clc_phase_consumer = 0
            while tile_id != -1:
                # Publish this persistent tile, then finalize its M and O outputs.
                if (USE_2CTA or CERTIFIED_CAUSAL) and USE_FAST_FIXED:
                    accum_cnt_qk, gauge_cnt = _fwd_fixed_2cta_control_tile(
                        tile_id,
                        tile_count,
                        accum_cnt_qk,
                        gauge_cnt,
                        sm_scale,
                        M,
                        H,
                        num_pid_n,
                        num_pid_in_group,
                        N_CTX,
                        desc_v,
                        desc_o,
                        qk_fulls,
                        qk_tiles,
                        p_fulls,
                        p_tiles,
                        acc_empties,
                        acc_tiles,
                        qk_empties,
                        o_empties,
                        o_fulls,
                        o_tiles,
                        cluster_cta_rank,
                        BLOCK_M,
                        BLOCK_M_SPLIT,
                        BLOCK_N,
                        EFFECTIVE_BLOCK_M,
                        GROUP_SIZE_N,
                        NUM_PID_M_STATIC,
                        OFFSET_GRID_X,
                        GLOBAL_LPT,
                        HEAD_DIM,
                        NUM_MMA_SLICES,
                        RESCALE_OPT,
                        STAGE,
                        PIPELINED,
                        CERTIFIED_CAUSAL, alpha_empties, alpha_fulls, alpha_tiles, Failure,
                        clc_context if CERTIFIED_CAUSAL and not DIRECT_SCHED else None,
                        clc_phase_producer,
                        DENSE_PAIRED=DENSE_PAIRED,
                    )
                    if CERTIFIED_CAUSAL and not DIRECT_SCHED:
                        if NUM_PID_M_STATIC != 32 and NUM_PID_M_STATIC != 64:
                            tlx.clc_producer(clc_context, clc_phase_producer)
                        clc_phase_producer ^= 1
                elif not USE_2CTA:
                    if not DIRECT_SCHED:
                        tlx.clc_producer(clc_context, clc_phase_producer)
                        clc_phase_producer ^= 1
                    accum_cnt = _fwd_control_tile(
                        tile_id,
                        tile_count,
                        accum_cnt,
                        sm_scale,
                        M,
                        H,
                        num_pid_n,
                        num_pid_in_group,
                        N_CTX,
                        desc_o,
                        alpha_empties,
                        alpha_fulls,
                        alpha_tiles,
                        acc_empties,
                        acc_fulls,
                        acc_tiles,
                        l_fulls,
                        l_tiles,
                        m_tiles,
                        qk_empties,
                        o_empties,
                        o_fulls,
                        o_tiles,
                        cluster_cta_rank,
                        BLOCK_M_SPLIT,
                        BLOCK_N,
                        EFFECTIVE_BLOCK_M,
                        GROUP_SIZE_N,
                        NUM_PID_M_STATIC,
                        OFFSET_GRID_X,
                        HEAD_DIM,
                        NUM_CTAS,
                        NUM_GROUPS_PER_CTA,
                        4 if SPARSE_FALLBACK else NUM_MMA_SLICES,
                        RESCALE_OPT,
                        SCALAR_N,
                        STAGE,
                        USE_2CTA,
                        USE_WHERE,
                        USE_FAST_FIXED,
                        FUSE_EPILOG,
                        GLOBAL_LPT,
                        LAYOUT_BSHD=LAYOUT_BSHD,
                    )
                tile_count += 1
                if DIRECT_SCHED:
                    next_tile_id = tile_id + persistent_stride
                    tile_id = -1 if SPARSE_FALLBACK else tl.where(next_tile_id < num_tiles, next_tile_id, -1)
                else:
                    if USE_GRID_LPT:
                        cta_x, cta_y, cta_z = tlx.clc_consumer(
                            clc_context, clc_phase_consumer, return_3d=True
                        )
                        tile_id = tl.where(
                            cta_x >= 0,
                            cta_x + LPT_GRID_X * (cta_y + tl.num_programs(1) * cta_z),
                            -1,
                        )
                    else:
                        tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer ^= 1

            if CERTIFIED_CAUSAL or SPARSE_FALLBACK or DENSE_EXCLUSIVE:
                tlx.barrier_wait(ws_done[0], 0)

        # softmax groups
        with tlx.async_task(num_warps=4, registers=DENSE_REGS if USE_2CTA else 232 if CERTIFIED_CAUSAL else 176 if SPARSE_FALLBACK else 168,
                            replicate=1 if (USE_2CTA or CERTIFIED_CAUSAL) and USE_FAST_FIXED else NUM_GROUPS_PER_CTA):
            accum_cnt_qk = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                gauge_cnt = tl.zeros([BLOCK_M_SPLIT], tl.float32) if CERTIFIED_CAUSAL else 0
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    EFFECTIVE_BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                    NUM_PID_M_STATIC,
                    OFFSET_GRID_X,
                    GLOBAL_LPT,
                    DENSE_PAIRED=DENSE_PAIRED,
                )
                # initialize pointer to m and l
                m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                # FA4 update_row_sum has init_val being None for the first iteration, here
                # we use initial value of 1.0
                l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32)
                if not USE_FAST_FIXED:
                    l_i += 1.0
                qk_scale = sm_scale
                qk_scale *= _RCP_LN2
                p_dtype = tlx.dtype_of(desc_v)

                if (USE_2CTA or CERTIFIED_CAUSAL) and USE_FAST_FIXED:
                    cid = 1
                    group_id = cid * NUM_CTAS + cluster_cta_rank
                else:
                    cid = tlx.async_task_replica_id()
                    group_id = cid * NUM_CTAS + cluster_cta_rank if USE_2CTA else cid
                offs_m = (start_m * EFFECTIVE_BLOCK_M) + ((group_id * BLOCK_M_SPLIT) + tl.arange(0, BLOCK_M_SPLIT))
                m_i, l_i, accum_cnt_qk, gauge_cnt = _fwd_softmax_stages(
                    qk_fulls,
                    qk_tiles,
                    p_fulls,
                    p_tiles,
                    alpha_empties,
                    alpha_fulls,
                    alpha_tiles,
                    acc_fulls,
                    acc_tiles,
                    cid,
                    accum_cnt_qk,
                    gauge_cnt,
                    qk_scale,
                    offs_m,
                    m_i,
                    l_i,
                    start_m,
                    N_CTX,
                    p_dtype,
                    BLOCK_M,
                    BLOCK_N,
                    HEAD_DIM,
                    NUM_MMA_SLICES,
                    STAGE,
                    RESCALE_OPT,
                    SCALAR_N,
                    USE_2CTA,
                    USE_FAST_FIXED,
                    SKIP_CAUSAL_DIAG,
                    CERTIFIED_CAUSAL,
                    MERGED_RECOVERY=SPARSE_FALLBACK,
                    DENSE_PAIRED=DENSE_PAIRED,
                )

                if USE_2CTA or CERTIFIED_CAUSAL:
                    if USE_2CTA:
                        tlx.barrier_arrive(qk_empties[cid], 1, remote_cta_rank=0)
                    _, phase = get_bufidx_phase(tile_count, 1)
                    tlx.barrier_wait(acc_empties[cid], phase)
                    tlx.barrier_wait(o_empties[cid], phase ^ 1)
                    scale = (1 / l_i)[:, None]
                    if CERTIFIED_CAUSAL:
                        bad_rows = ~((gauge_cnt <= 128.0) & (l_i > 0.0) & (l_i <= N_CTX) & (tl.abs(m_i * qk_scale) <= 32.0))
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        subslice = tlx.subslice(
                            acc_tiles[cid],
                            HEAD_DIM * slice_id // NUM_MMA_SLICES,
                            HEAD_DIM // NUM_MMA_SLICES,
                        )
                        acc = tlx.local_load(subslice)
                        if CERTIFIED_CAUSAL:
                            bad_rows |= ~(tl.reduce(tl.abs(acc), 1, _certificate_maximum_nan) < float("inf"))
                        acc = _mul_f32x2(acc, scale)
                        acc = acc.to(tlx.dtype_of(desc_o))
                        subslice_o = tlx.local_slice(
                            o_tiles[cid],
                            [0, HEAD_DIM * slice_id // NUM_MMA_SLICES],
                            [BLOCK_M_SPLIT, HEAD_DIM // NUM_MMA_SLICES],
                        )
                        tlx.local_store(subslice_o, acc)
                    if CERTIFIED_CAUSAL:
                        tl.store(Failure + 2 * (off_hz * (N_CTX // BLOCK_M) + start_m) + cid,
                                 (tl.sum(bad_rows.to(tl.int32), 0) != 0).to(tl.int32))
                    tlx.fence("async_shared")
                    if CERTIFIED_CAUSAL:
                        tlx.barrier_arrive(qk_empties[cid])
                    tlx.barrier_arrive(o_fulls[cid])
                    if PIPELINED and USE_FAST_FIXED:
                        saved_m = 1.0 / l_i
                    elif CERTIFIED_CAUSAL:
                        saved_m = m_i * sm_scale * _RCP_LN2 + tl.math.log2(l_i)
                    elif USE_FAST_FIXED:
                        saved_m = _BF16_FIXED_GAUGE + tl.math.log2(l_i)
                    else:
                        saved_m = m_i * qk_scale if RESCALE_OPT else m_i
                        saved_m += tl.math.log2(l_i)
                    tl.store(
                        M + off_hz * N_CTX + offs_m,
                        saved_m,
                        mask=offs_m < N_CTX,
                    )
                else:
                    tlx.local_store(l_tiles[cid], tl.join(l_i, l_i) if SCALAR_N == 2 else l_i[:, None])
                    tlx.local_store(m_tiles[cid], tl.join(m_i, m_i) if SCALAR_N == 2 else m_i[:, None])
                    tlx.barrier_arrive(l_fulls[cid])
                tile_count += 1
                if DIRECT_SCHED:
                    next_tile_id = tile_id + persistent_stride
                    tile_id = -1 if SPARSE_FALLBACK else tl.where(next_tile_id < num_tiles, next_tile_id, -1)
                else:
                    if USE_GRID_LPT:
                        cta_x, cta_y, cta_z = tlx.clc_consumer(
                            clc_context, clc_phase_consumer, return_3d=True
                        )
                        tile_id = tl.where(
                            cta_x >= 0,
                            cta_x + LPT_GRID_X * (cta_y + tl.num_programs(1) * cta_z),
                            -1,
                        )
                    else:
                        tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer ^= 1

            if CERTIFIED_CAUSAL or SPARSE_FALLBACK or DENSE_EXCLUSIVE:
                tlx.barrier_arrive(ws_done[0])

        # mma group
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            accum_cnt_qk = 0

            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                _, _, lo, hi, _, _ = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    EFFECTIVE_BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                    NUM_PID_M_STATIC,
                    OFFSET_GRID_X,
                    GLOBAL_LPT,
                    DENSE_PAIRED=DENSE_PAIRED,
                )
                if USE_2CTA:
                    if is_leader:
                        accum_cnt_kv, accum_cnt_qk = _fwd_mma_tile(
                            tile_count,
                            accum_cnt_kv,
                            accum_cnt_qk,
                            lo,
                            hi,
                            q_tiles,
                            k_tiles,
                            qk_tiles,
                            p_tiles,
                            acc_tiles,
                            q_fulls,
                            q_empties,
                            k_fulls,
                            k_empties,
                            qk_fulls,
                            qk_empties,
                            p_fulls,
                            acc_fulls,
                            acc_empties,
                            BLOCK_N,
                            HEAD_DIM_KV,
                            NUM_BUFFERS_Q,
                            NUM_BUFFERS_KV,
                            NUM_P_SLICES,
                            v_tiles,
                            v_fulls,
                            v_empties,
                            True,
                            SKIP_RESCALE=USE_FAST_FIXED,
                        )
                else:
                    accum_cnt_kv, accum_cnt_qk = _fwd_mma_tile(
                        tile_count,
                        accum_cnt_kv,
                        accum_cnt_qk,
                        lo,
                        hi,
                        q_tiles,
                        kv_tiles,
                        qk_tiles,
                        p_tiles,
                        acc_tiles,
                        q_fulls,
                        q_empties,
                        kv_fulls,
                        kv_empties,
                        qk_fulls,
                        qk_empties,
                        p_fulls,
                        acc_fulls,
                        acc_empties,
                        BLOCK_N,
                        HEAD_DIM_KV,
                        NUM_BUFFERS_Q,
                        NUM_BUFFERS_KV,
                        NUM_P_SLICES,
                        None,
                        None,
                        None,
                        False,
                        SKIP_RESCALE=USE_FAST_FIXED,
                        SKIP_CAUSAL_Q0_LAST=SKIP_CAUSAL_DIAG,
                    )
                tile_count += 1
                if DIRECT_SCHED:
                    next_tile_id = tile_id + persistent_stride
                    tile_id = -1 if SPARSE_FALLBACK else tl.where(next_tile_id < num_tiles, next_tile_id, -1)
                else:
                    if USE_GRID_LPT:
                        cta_x, cta_y, cta_z = tlx.clc_consumer(
                            clc_context, clc_phase_consumer, return_3d=True
                        )
                        tile_id = tl.where(
                            cta_x >= 0,
                            cta_x + LPT_GRID_X * (cta_y + tl.num_programs(1) * cta_z),
                            -1,
                        )
                    else:
                        tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer ^= 1

            if CERTIFIED_CAUSAL or SPARSE_FALLBACK or DENSE_EXCLUSIVE:
                tlx.barrier_arrive(ws_done[0])

        if USE_2CTA and USE_FAST_FIXED:
            with tlx.async_task(num_warps=1, registers=24):
                accum_cnt_k = 0
                tile_id = start_pid
                while tile_id != -1:
                    _, off_hz, lo, hi, _, kv_offset_y = _compute_offsets(
                        tile_id,
                        H,
                        num_pid_n,
                        num_pid_in_group,
                        N_CTX,
                        EFFECTIVE_BLOCK_M,
                        STAGE,
                        GROUP_SIZE_N,
                        NUM_PID_M_STATIC,
                        OFFSET_GRID_X,
                        GLOBAL_LPT,
                        DENSE_PAIRED=DENSE_PAIRED,
                    )
                    _, kv_offset_y, offset_x = _forward_descriptor_offsets(
                        0, kv_offset_y, off_hz, H, N_CTX, HEAD_DIM, LAYOUT_BSHD)
                    accum_cnt_k = _fwd_load_first_k_tile(
                        accum_cnt_k,
                        lo,
                        hi,
                        kv_offset_y,
                        desc_k,
                        k_tiles,
                        k_fulls,
                        k_empties,
                        K_BYTES_PER_ELEM,
                        BLOCK_N,
                        HEAD_DIM,
                        NUM_BUFFERS_KV,
                        cluster_cta_rank,
                        NUM_CTAS,
                        offset_x=offset_x,
                    )
                    tlx.named_barrier_wait(15, 64)
                    next_tile_id = tile_id + persistent_stride
                    tile_id = tl.where(next_tile_id < num_tiles, next_tile_id, -1)
                if DENSE_EXCLUSIVE:
                    tlx.barrier_arrive(ws_done[0])

        # Q/K/V loader role; output publication remains in the epilog task.
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                _, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    EFFECTIVE_BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                    NUM_PID_M_STATIC,
                    OFFSET_GRID_X,
                    GLOBAL_LPT,
                    DENSE_PAIRED=DENSE_PAIRED,
                )
                qo_offset_y, kv_offset_y, offset_x = _forward_descriptor_offsets(
                    qo_offset_y, kv_offset_y, off_hz, H, N_CTX, HEAD_DIM, LAYOUT_BSHD)
                if USE_2CTA:
                    accum_cnt_kv = _fwd_load_tile(
                        tile_count,
                        accum_cnt_kv,
                        lo,
                        hi,
                        qo_offset_y,
                        kv_offset_y,
                        desc_q,
                        desc_k,
                        desc_v,
                        q_tiles,
                        k_tiles,
                        q_fulls,
                        q_empties,
                        k_fulls,
                        k_empties,
                        Q_BYTES_PER_ELEM,
                        K_BYTES_PER_ELEM,
                        V_BYTES_PER_ELEM,
                        BLOCK_M_SPLIT,
                        BLOCK_N,
                        HEAD_DIM,
                        NUM_BUFFERS_Q,
                        NUM_BUFFERS_KV,
                        cluster_cta_rank,
                        v_tiles,
                        v_fulls,
                        v_empties,
                        NUM_GROUPS_PER_CTA,
                        NUM_CTAS,
                        SKIP_FIRST_K=USE_FAST_FIXED,
                        offset_x=offset_x,
                    )
                    if USE_FAST_FIXED:
                        tlx.named_barrier_arrive(15, 64)
                else:
                    accum_cnt_kv = _fwd_load_tile(
                        tile_count,
                        accum_cnt_kv,
                        lo,
                        hi,
                        qo_offset_y,
                        kv_offset_y,
                        desc_q,
                        desc_k,
                        desc_v,
                        q_tiles,
                        kv_tiles,
                        q_fulls,
                        q_empties,
                        kv_fulls,
                        kv_empties,
                        Q_BYTES_PER_ELEM,
                        K_BYTES_PER_ELEM,
                        V_BYTES_PER_ELEM,
                        BLOCK_M_SPLIT,
                        BLOCK_N,
                        HEAD_DIM,
                        NUM_BUFFERS_Q,
                        NUM_BUFFERS_KV,
                        offset_x=offset_x,
                    )

                tile_count += 1
                if DIRECT_SCHED:
                    next_tile_id = tile_id + persistent_stride
                    tile_id = -1 if SPARSE_FALLBACK else tl.where(next_tile_id < num_tiles, next_tile_id, -1)
                else:
                    if USE_GRID_LPT:
                        cta_x, cta_y, cta_z = tlx.clc_consumer(
                            clc_context, clc_phase_consumer, return_3d=True
                        )
                        tile_id = tl.where(
                            cta_x >= 0,
                            cta_x + LPT_GRID_X * (cta_y + tl.num_programs(1) * cta_z),
                            -1,
                        )
                    else:
                        tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer ^= 1

            if CERTIFIED_CAUSAL or SPARSE_FALLBACK or DENSE_EXCLUSIVE:
                tlx.barrier_arrive(ws_done[0])

        # epilog group
        if USE_2CTA or not FUSE_EPILOG:
            with tlx.async_task(num_warps=1, registers=24):
                # initialize offsets
                tile_count = 0
                tile_id = start_pid
                clc_phase_consumer = 0
                while tile_id != -1:
                    # initialize offsets
                    _, off_hz, _, _, qo_offset_y, _ = _compute_offsets(
                        tile_id,
                        H,
                        num_pid_n,
                        num_pid_in_group,
                        N_CTX,
                        EFFECTIVE_BLOCK_M,
                        STAGE,
                        GROUP_SIZE_N,
                        NUM_PID_M_STATIC,
                        OFFSET_GRID_X,
                        GLOBAL_LPT,
                        DENSE_PAIRED=DENSE_PAIRED,
                    )
                    qo_offset_y, _, offset_x = _forward_descriptor_offsets(
                        qo_offset_y, 0, off_hz, H, N_CTX, HEAD_DIM, LAYOUT_BSHD)
                    _, phase = get_bufidx_phase(tile_count, 1)
                    if NUM_PID_M_STATIC == 1 and NUM_CTAS == 1:
                        for cid in tl.static_range(0, NUM_GROUPS_PER_CTA):
                            tlx.barrier_wait(o_fulls[cid], phase)
                            if cid * BLOCK_M_SPLIT < N_CTX:
                                tlx.async_descriptor_store(
                                    desc_o, o_tiles[cid], [qo_offset_y + cid * BLOCK_M_SPLIT, offset_x],
                                    eviction_policy="evict_first")
                            tlx.async_descriptor_store_wait(0)
                            tlx.barrier_arrive(o_empties[cid])
                    elif USE_FAST_FIXED and NUM_GROUPS_PER_CTA == 2:
                        for cid in tl.static_range(0, NUM_GROUPS_PER_CTA):
                            group_id = cid * NUM_CTAS + cluster_cta_rank
                            tlx.barrier_wait(o_fulls[cid], phase)
                            qo_offset_y_split = qo_offset_y + group_id * BLOCK_M_SPLIT
                            tlx.async_descriptor_store(
                                desc_o, o_tiles[cid], [qo_offset_y_split, offset_x], eviction_policy="evict_first")
                        tlx.async_descriptor_store_wait(1)
                        tlx.barrier_arrive(o_empties[0])
                        tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(o_empties[1])
                    else:
                        for cid in tl.static_range(0, NUM_GROUPS_PER_CTA):
                            group_id = cid * NUM_CTAS + cluster_cta_rank
                            tlx.barrier_wait(o_fulls[cid], phase)
                            qo_offset_y_split = qo_offset_y + group_id * BLOCK_M_SPLIT
                            tlx.async_descriptor_store(
                                desc_o, o_tiles[cid], [qo_offset_y_split, offset_x], eviction_policy="evict_first")
                        for cid in tl.static_range(0, NUM_GROUPS_PER_CTA):
                            tlx.async_descriptor_store_wait(NUM_GROUPS_PER_CTA - 1 - cid)
                            tlx.barrier_arrive(o_empties[cid])

                    tile_count += 1
                    if DIRECT_SCHED:
                        next_tile_id = tile_id + persistent_stride
                        tile_id = -1 if SPARSE_FALLBACK else tl.where(next_tile_id < num_tiles, next_tile_id, -1)
                    else:
                        if USE_GRID_LPT:
                            cta_x, cta_y, cta_z = tlx.clc_consumer(
                                clc_context, clc_phase_consumer, return_3d=True
                            )
                            tile_id = tl.where(
                                cta_x >= 0,
                                cta_x + LPT_GRID_X * (cta_y + tl.num_programs(1) * cta_z),
                                -1,
                            )
                        else:
                            tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                        clc_phase_consumer ^= 1

                if CERTIFIED_CAUSAL or SPARSE_FALLBACK or DENSE_EXCLUSIVE:
                    tlx.barrier_arrive(ws_done[0])


@triton.jit
def _dq_fixed_scale(n_ctx):
    return tl.where(n_ctx <= 8192, 2048.0, 4096.0)


@triton.jit
def _dq_pack_fixed_pair(values, scale, contributors, RangeBudget, budget_offsets):
    shift = tl.inline_asm_elementwise(
        "bfind.u32 $0, $1;", constraints="=r,r", args=[contributors - 1],
        dtype=tl.int32, is_pure=True, pack=1,
    ) + 1
    limit = 32767 >> (shift + 1)
    lo, hi = tl.split(tl.reshape(values, (values.shape[0], values.shape[1] // 2, 2)))
    packed, amplitude = tl.inline_asm_elementwise(
        """
        {
            .reg .f32 a0, a1;
            .reg .s32 q0, q1, t1;
            .reg .b32 b0, b1;
            mul.rn.f32 a0, $2, $4;
            mul.rn.f32 a1, $3, $4;
            mov.b32 b0, a0;
            mov.b32 b1, a1;
            and.b32 b0, b0, 0x7fffffff;
            and.b32 b1, b1, 0x7fffffff;
            max.u32 $1, b0, b1;
            cvt.rni.s32.f32 q0, a0;
            cvt.rni.s32.f32 q1, a1;
            shl.b32 t1, q1, 16;
            add.u32 $0, t1, q0;
        }
        """,
        constraints="=r,=r,f,f,f", args=[lo, hi, scale],
        dtype=(tl.int32, tl.int32), is_pure=True, pack=1,
    )
    row_amplitude = tl.max(amplitude, 1)
    invalid = row_amplitude >= tl.full((), 32767.5, tl.float32).to(tl.int32, bitcast=True)
    safe_amplitude = tl.where(invalid, 0, row_amplitude).to(tl.float32, bitcast=True)
    row_bound = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;", constraints="=r,f", args=[safe_amplitude],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    excess = tl.where(invalid, 32768, tl.maximum(row_bound - limit, 0))
    tl.atomic_add(RangeBudget + budget_offsets, excess, mask=excess > 0, sem="relaxed")
    return packed

@triton.jit
def _dq_pack_bf16_pair(values):
    bits = values.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    lo, hi = tl.split(tl.reshape(bits, (values.shape[0], values.shape[1] // 2, 2)))
    return (lo.to(tl.uint32) | (hi.to(tl.uint32) << 16)).to(tl.int32)


@triton.jit
def _attn_bwd_unpack_fixed(DQ_PACKED, DQ_OUT, TOTAL: tl.constexpr,
                           N_CTX: tl.constexpr, ROW_STRIDE: tl.constexpr,
                           Failure, BLOCK: tl.constexpr, RangeBudget, N_HEAD: tl.constexpr):
    if 2 * (TOTAL + BLOCK) > 2147483647:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    elif BLOCK == 1024:
        offsets = tl.program_id(0).to(tl.uint32) * BLOCK + tl.arange(0, BLOCK).to(tl.uint32)
    else:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(DQ_PACKED + offsets, offsets < TOTAL, other=0)
    lo = (packed << 16) >> 16
    hi = ((packed - lo) >> 16).to(tl.int16).to(tl.int32)
    if BLOCK == 1024 or 2 * (TOTAL + BLOCK) > 2147483647:
        row = (offsets // ROW_STRIDE % N_CTX).to(tl.int32)
    else:
        row = offsets // ROW_STRIDE % N_CTX
    inv_scale = tl.div_rn(1.0, _dq_fixed_scale(N_CTX))
    values = tl.join(lo.to(tl.float32) * inv_scale, hi.to(tl.float32) * inv_scale)
    exact_lo = (packed & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    exact_hi = (packed >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    values = tl.where(row[:, None] < 256, tl.join(exact_lo, exact_hi), values)
    converted = values.to(tl.bfloat16)
    budget = (row // 256 + 1).to(tl.float32) * (0.5 / _dq_fixed_scale(N_CTX))
    error = tl.abs(converted.to(tl.float32) - values) + budget[:, None]
    contributors = row // 256 + 1
    shift = tl.inline_asm_elementwise(
        "bfind.u32 $0, $1;", constraints="=r,r", args=[contributors - 1],
        dtype=tl.int32, is_pure=True, pack=1,
    ) + 1
    base_bound = contributors * (32767 >> (shift + 1))
    if ROW_STRIDE == 64:
        batch_head = offsets // (N_CTX * 64)
    else:
        batch_head = offsets // (N_CTX * N_HEAD * 64) * N_HEAD + offsets // 64 % N_HEAD
    head_half = offsets % 64 // 32
    budget_offsets = (batch_head * 2 + head_half) * N_CTX + row
    excess = tl.load(RangeBudget + budget_offsets, offsets < TOTAL, other=0)
    range_bad = base_bound + excess > 32767
    bad = (row[:, None] >= 256) & (offsets[:, None] < TOTAL) & ((error > 0.01) | range_bad[:, None])
    if tl.max(tl.max(bad.to(tl.int32), 1), 0) != 0:
        tl.atomic_or(Failure, 1, sem="relaxed")
    converted = tl.reshape(converted, (BLOCK * 2,))
    if 2 * (TOTAL + BLOCK) > 2147483647:
        out_offsets = tl.program_id(0).to(tl.int64) * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    elif BLOCK == 1024:
        out_offsets = tl.program_id(0).to(tl.uint32) * BLOCK * 2 + tl.arange(0, BLOCK * 2).to(tl.uint32)
    else:
        out_offsets = tl.program_id(0) * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    tl.store(DQ_OUT + out_offsets, converted, out_offsets < TOTAL * 2)


@triton.jit
def _bwd_prefix_delta(Q, K, V, DO, M, Delta, scale,
                         N_CTX: tl.constexpr, N_HEAD: tl.constexpr,
                         Q_STRIDES: tl.constexpr, K_STRIDES: tl.constexpr,
                         V_STRIDES: tl.constexpr, DO_STRIDES: tl.constexpr,
                         BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 128,
                         HEAD_DIM: tl.constexpr = 128):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    key = tl.arange(0, BLOCK_N)
    dim = tl.arange(0, HEAD_DIM)
    hz = tl.program_id(1)
    batch = hz // N_HEAD
    head = hz % N_HEAD
    q = tl.load(Q + batch * Q_STRIDES[0] + head * Q_STRIDES[1] + row[:, None] * Q_STRIDES[2] + dim[None, :] * Q_STRIDES[3])
    k = tl.load(K + batch * K_STRIDES[0] + head * K_STRIDES[1] + key[:, None] * K_STRIDES[2] + dim[None, :] * K_STRIDES[3])
    qk = tl.dot(q, tl.trans(k))
    m = tl.load(M + hz * N_CTX + row)
    exponent = tl.fma(qk, scale * 1.4426950408889634, -m[:, None])
    exponent = tl.where(key[None, :] <= row[:, None], exponent, -float('inf'))
    p = tl.math.exp2(exponent)
    do = tl.load(DO + batch * DO_STRIDES[0] + head * DO_STRIDES[1] + row[:, None] * DO_STRIDES[2] + dim[None, :] * DO_STRIDES[3])
    v = tl.load(V + batch * V_STRIDES[0] + head * V_STRIDES[1] + key[:, None] * V_STRIDES[2] + dim[None, :] * V_STRIDES[3])
    dp = tl.dot(do, tl.trans(v))
    delta = tl.sum(p * dp, 1)
    tl.store(Delta + hz * N_CTX + row, delta)



@triton.jit
def _bwd_prefix_dv_residual(Q, K, DO, M, DV, scale,
                               N_CTX: tl.constexpr, N_HEAD: tl.constexpr,
                               Q_STRIDES: tl.constexpr, K_STRIDES: tl.constexpr,
                               DO_STRIDES: tl.constexpr, DV_STRIDES: tl.constexpr,
                               BLOCK_K: tl.constexpr = 16, PREFIX_Q: tl.constexpr = 128,
                               HEAD_DIM: tl.constexpr = 128):
    key = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    query = tl.arange(0, PREFIX_Q)
    dim = tl.arange(0, HEAD_DIM)
    hz = tl.program_id(1)
    batch = hz // N_HEAD
    head = hz % N_HEAD
    k = tl.load(K + batch * K_STRIDES[0] + head * K_STRIDES[1] + key[:, None] * K_STRIDES[2] + dim[None, :] * K_STRIDES[3])
    q = tl.load(Q + batch * Q_STRIDES[0] + head * Q_STRIDES[1] + query[:, None] * Q_STRIDES[2] + dim[None, :] * Q_STRIDES[3])
    qk = tl.dot(k, tl.trans(q))
    m = tl.load(M + hz * N_CTX + query)
    exponent = tl.fma(qk, scale * 1.4426950408889634, -m[None, :])
    exponent = tl.where(key[:, None] <= query[None, :], exponent, -float('inf'))
    p = tl.math.exp2(exponent)
    residual = p - p.to(tl.bfloat16).to(tl.float32)
    do = tl.load(DO + batch * DO_STRIDES[0] + head * DO_STRIDES[1] + query[:, None] * DO_STRIDES[2] + dim[None, :] * DO_STRIDES[3])
    high = residual.to(tl.bfloat16)
    correction = tl.dot(high, do)
    pointer = DV + batch * DV_STRIDES[0] + head * DV_STRIDES[1] + key[:, None] * DV_STRIDES[2] + dim[None, :] * DV_STRIDES[3]
    original = tl.load(pointer).to(tl.float32)
    tl.store(pointer, original + correction)



@triton.jit
def _bwd_row_offset(off_hz, off_m, N_CTX, HEAD_DIM: tl.constexpr,
                    N_HEAD: tl.constexpr, STRIDES: tl.constexpr):
    if STRIDES is None:
        return off_hz * HEAD_DIM * N_CTX + off_m * HEAD_DIM
    else:
        return ((off_hz // N_HEAD) * STRIDES[0] + (off_hz % N_HEAD) * STRIDES[1]
                + off_m * STRIDES[2])


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         M, DO_SCALED, Delta, DQ_ACCUM,  #
                         N_CTX,  #
                         SCALE_DO_BY_INV_L: tl.constexpr,
                         ZERO_DQ: tl.constexpr,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,  #
                         N_HEAD: tl.constexpr = 1,
                         O_STRIDES: tl.constexpr = None,
                         DO_STRIDES: tl.constexpr = None,
                         SCALED_DO_STRIDES: tl.constexpr = None,
                         DQ_STRIDES: tl.constexpr = None,
                         PACKED_DQ: tl.constexpr = False,
                         RangeFailure=None,
                         RangeBudget=None,
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o_rows = _bwd_row_offset(off_hz, off_m, N_CTX, HEAD_DIM, N_HEAD, O_STRIDES)
    do_rows = _bwd_row_offset(off_hz, off_m, N_CTX, HEAD_DIM, N_HEAD, DO_STRIDES)
    o = tl.load(O + o_rows[:, None] + off_n[None, :])
    do_ptrs = DO + do_rows[:, None] + off_n[None, :]
    do = tl.load(do_ptrs).to(tl.float32)
    if SCALE_DO_BY_INV_L:
        inv_l = tl.load(M + off_hz * N_CTX + off_m)
        do = (do * inv_l[:, None]).to(DO.dtype.element_ty)
        scaled_rows = _bwd_row_offset(off_hz, off_m, N_CTX, HEAD_DIM, N_HEAD, SCALED_DO_STRIDES)
        tl.store(DO_SCALED + scaled_rows[:, None] + off_n[None, :], do)
        do = do.to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)
    if PACKED_DQ:
        packed_cols = tl.arange(0, HEAD_DIM // 2)
        packed_rows = _bwd_row_offset(off_hz, off_m, N_CTX, HEAD_DIM // 2, N_HEAD, DQ_STRIDES)
        tl.store(DQ_ACCUM + packed_rows[:, None] + packed_cols[None, :], 0)
        budget_offsets = (off_hz * 2 + tl.arange(0, 2)[:, None]) * N_CTX + off_m[None, :]
        tl.store(RangeBudget + budget_offsets, 0)
        if tl.program_id(0) == 0 and off_hz == 0:
            tl.store(RangeFailure, 0)
    elif ZERO_DQ:
        dq_rows = _bwd_row_offset(off_hz, off_m, N_CTX, HEAD_DIM, N_HEAD, DQ_STRIDES)
        tl.store(
            DQ_ACCUM + dq_rows[:, None] + off_n[None, :],
            0.0,
        )


@triton.jit
def _attn_bwd_dq_postprocess(DQ_ACCUM, DQ_OUT,  #
                             N_CTX,  #
                             BLK: tl.constexpr, HALF_HD: tl.constexpr,  #
                             BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,  #
                             N_HEAD: tl.constexpr = 1,
                             DQ_STRIDES: tl.constexpr = None,
                             ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_h = tl.arange(0, HEAD_DIM)
    q = off_m[:, None]
    h = off_h[None, :]
    tile_base = (q // BLK) * BLK
    local = q % BLK
    half = h // HALF_HD
    col = h % HALF_HD
    packed_row = 2 * tile_base + local + BLK * half
    src = DQ_ACCUM + off_hz * N_CTX * HEAD_DIM + packed_row * HALF_HD + col
    val = tl.load(src)
    output_rows = _bwd_row_offset(off_hz, off_m, N_CTX, HEAD_DIM, N_HEAD, DQ_STRIDES)
    dst = DQ_OUT + output_rows[:, None] + h
    tl.store(dst, val.to(DQ_OUT.dtype.element_ty))


_bwd_selected_meta = {}


def _bwd_host_descriptor_pre_hook_tlx(nargs):
    block_m = nargs["BLOCK_M1"]
    block_n = nargs["BLOCK_N1"]
    head_dim = nargs["HEAD_DIM"]
    num_ctas = nargs.get("NUM_CTAS", 1)
    _bwd_selected_meta["BLOCK_M1"] = block_m
    _bwd_selected_meta["NUM_CTAS"] = num_ctas
    # Reset dQ before launches where preprocessing does not own initialization.
    # dQ uses TMA reduce-add, so stale values accumulate across runs.
    # dk/dv don't need zeroing — they use use_acc=False on the first iteration.
    if (
        (not nargs.get("PERSISTENT_BWD", False) or nargs.get("PACKED_DQ", False))
        and not nargs.get("PREPROCESS_ZEROES_DQ", False)
    ):
        nargs["desc_dq"].base.zero_()
    nargs["desc_q"].block_shape = [1, 1, block_m, head_dim // num_ctas]
    nargs["desc_do"].block_shape = [1, 1, block_m, head_dim // num_ctas]
    nargs["desc_v"].block_shape = [1, 1, block_n, head_dim]
    nargs["desc_k"].block_shape = [1, 1, block_n, head_dim]
    direct_dq = nargs.get("SCALE_QK_IN_KERNEL", False) and num_ctas > 1 and head_dim == 128
    if direct_dq:
        dq_m = block_m // num_ctas
        dq_slice = head_dim // nargs["EPILOGUE_SUBTILE"]
        nargs["desc_dq"].block_shape = [1, 1, 2, dq_m, dq_slice]
    elif num_ctas > 1:
        dq_m = block_m
        dq_slice = head_dim // nargs["EPILOGUE_SUBTILE"]
        nargs["desc_dq"].block_shape = [1, 1, dq_m, dq_slice]
    else:
        dq_m = block_m
        dq_slice = nargs["DQ_REDUCE_NCOL"]
        nargs["desc_dq"].block_shape = [1, 1, dq_m, dq_slice]
    dkv_slice = nargs["DKV_STORE_NCOL"]
    nargs["desc_dv"].block_shape = [1, 1, block_n, dkv_slice]
    nargs["desc_dk"].block_shape = [1, 1, block_n, dkv_slice]
    if "desc_m" in nargs:
        nargs["desc_m"].block_shape = [block_m]
    if "desc_delta" in nargs:
        nargs["desc_delta"].block_shape = [block_m]
    # 2-CTA: separate B-operand descriptors for the transposed views.
    if "desc_kt" in nargs and "desc_qt" in nargs:
        nargs["desc_kt"].block_shape = [1, 1, block_n * num_ctas, head_dim // num_ctas]
        nargs["desc_qt"].block_shape = [1, 1, block_m // num_ctas, head_dim]
        nargs["desc_dot"].block_shape = [1, 1, block_m // num_ctas, head_dim]


configs_bwd_1cta = [
    triton.Config(
        {
            "BLOCK_M1": 64,
            "BLOCK_N1": 128,
            "NUM_BUFFERS_KV": 1,
            "NUM_BUFFERS_Q": 2,
            "NUM_BUFFERS_DO": 1,
            "NUM_BUFFERS_DS": 1,
            "NUM_BUFFERS_TMEM": 1,
            "NUM_COMPUTE_SLICES": 2,
            "DKV_STORE_NCOL": 64,
            "DQ_REDUCE_STAGES": 2,
            "DQ_REDUCE_NCOL": 32,
            "DQ_STAGE_COUNT": 2,
            "EPILOGUE_SUBTILE": 4,
            "GROUP_SIZE_M": 1,
            "USE_WARP_BARRIER": use_warp_barrier,
            "NUM_CTAS": 1,
        },
        num_warps=8,
        num_stages=1,
        pre_hook=_bwd_host_descriptor_pre_hook_tlx,
    )
    for use_warp_barrier in (False, True)
]

configs_bwd_2cta = [
    triton.Config(
        {
            "BLOCK_M1": 128,
            "BLOCK_N1": 128,
            "NUM_BUFFERS_KV": 1,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_DO": 1,
            "NUM_BUFFERS_DS": 1,
            "NUM_BUFFERS_TMEM": 1,
            "NUM_COMPUTE_SLICES": 2,
            "DKV_STORE_NCOL": 64,
            "DQ_REDUCE_STAGES": 2,
            "DQ_REDUCE_NCOL": 32,
            "DQ_STAGE_COUNT": 2,
            "EPILOGUE_SUBTILE": 8,
            "GROUP_SIZE_M": 1,
            "USE_WARP_BARRIER": False,
            "NUM_CTAS": 2,
        },
        num_warps=8,
        num_stages=1,
        pre_hook=_bwd_host_descriptor_pre_hook_tlx,
        ctas_per_cga=(2, 1, 1),
    )
]


def _make_bwd_runtime_config(num_ctas, epilogue_subtile, dq_stage_count, use_warp_barrier=False):
    block_m = 128 if num_ctas == 2 else 64
    kwargs = {
        "BLOCK_M1": block_m,
        "BLOCK_N1": 128,
        "NUM_BUFFERS_KV": 1,
        "NUM_BUFFERS_Q": 1 if num_ctas == 2 else 2,
        "NUM_BUFFERS_DO": 1,
        "NUM_BUFFERS_DS": 1,
        "NUM_BUFFERS_TMEM": 1,
        "NUM_COMPUTE_SLICES": 2,
        "DKV_STORE_NCOL": 64,
        "DQ_REDUCE_STAGES": 2,
        "DQ_REDUCE_NCOL": 32,
        "EPILOGUE_SUBTILE": epilogue_subtile,
        "GROUP_SIZE_M": 1,
        "USE_WARP_BARRIER": use_warp_barrier,
        "NUM_CTAS": num_ctas,
        "DQ_STAGE_COUNT": dq_stage_count,
    }
    return triton.Config(
        kwargs,
        num_warps=8,
        num_stages=1,
        pre_hook=_bwd_host_descriptor_pre_hook_tlx,
        **({"ctas_per_cga": (2, 1, 1)} if num_ctas == 2 else {}),
    )


_configs_bwd_1cta_runtime = [
    _make_bwd_runtime_config(1, 4, 2, use_warp_barrier)
    for use_warp_barrier in (False, True)
]
_bwd_d64_config = _make_bwd_runtime_config(1, 4, 2, True)
_bwd_d64_config.kwargs["BLOCK_M1"] = 128
_configs_bwd_1cta_runtime.append(_bwd_d64_config)
_configs_bwd_2cta_runtime = [
    _make_bwd_runtime_config(2, epilogue_subtile, dq_stage_count)
    for epilogue_subtile, dq_stage_count in ((2, 1), (4, 2), (8, 2), (8, 4))
]
BWD_CONFIGS = _configs_bwd_1cta_runtime + _configs_bwd_2cta_runtime


def prune_bwd_configs(configs, named_args, **kwargs):
    kwargs = {**named_args, **kwargs}
    n_ctx = kwargs["N_CTX"]
    configs = [
        config
        for config in configs
        if (config.kwargs["EPILOGUE_SUBTILE"] != 2 or kwargs.get("PRENORMALIZED_DO", False))
        and (
            config.kwargs["NUM_CTAS"] != 1
            or config.kwargs["BLOCK_M1"] != 128
            or (
                kwargs.get("HEAD_DIM") == 64
                and kwargs.get("STAGE") in (1, 3)
            )
        )
        and (
            (n_ctx + config.kwargs["BLOCK_N1"] - 1)
            // config.kwargs["BLOCK_N1"]
        )
        % config.kwargs.get("NUM_CTAS", 1)
        == 0
    ]
    if kwargs.get("SCALE_QK_IN_KERNEL", False) and kwargs.get("HEAD_DIM") == 128:
        configs = [config for config in configs if config.kwargs.get("NUM_CTAS", 1) == 2]
        if kwargs.get("PERSISTENT_BWD", False):
            configs = [
                config
                for config in configs
                if config.kwargs["EPILOGUE_SUBTILE"] == 8
                and config.kwargs.get("DQ_STAGE_COUNT", 2) == 2
            ]
        else:
            if kwargs.get("PRENORMALIZED_DO", False):
                if kwargs.get("BSHD_CONFIG", False) and n_ctx == 8192:
                    epilogue_subtile, dq_stage_count = 4, 2
                elif n_ctx >= 4096 and kwargs.get("STAGE") == 1:
                    epilogue_subtile, dq_stage_count = 8, 2
                else:
                    epilogue_subtile, dq_stage_count = 2, 1
            else:
                _, epilogue_subtile, dq_stage_count = BWD_DIRECT_DQ_CONFIG
            configs = [
                config
                for config in configs
                if config.kwargs["EPILOGUE_SUBTILE"] == epilogue_subtile
                and config.kwargs.get("DQ_STAGE_COUNT", 2) == dq_stage_count
            ]
        assert configs
    elif n_ctx < BWD_2CTA_MIN_N_CTX:
        configs = [
            config
            for config in configs
            if config.kwargs.get("NUM_CTAS", 1) == 1
            and not config.kwargs.get("USE_WARP_BARRIER", False)
        ]
    return configs


@triton.jit
def _bwd_mma_dots_1cta(
    blk_idx,
    num_steps,
    kv_buf_id,
    kv_phase,
    k_tiles,
    v_tiles,
    q_tiles,
    do_tiles,
    qk_tiles,
    qk_fulls,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_tiles,
    dp_fulls,
    dp_empties,
    dv_tiles,
    dv_fulls,
    dv_empties,
    dk_tiles,
    dk_fulls,
    dk_empties,
    dq_tiles,
    dq_fulls,
    dq_empties,
    ds_tiles,
    ds_fulls,
    dsT_tmem_tiles,
    dsT_tmem_fulls,
    do_fulls,
    do_empties,
    q_fulls,
    q_empties,
    k_mma_done,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
):
    """1-CTA MMA dot sequence: prolog + main loop + epilog.

    This is the original base code, untouched.
    """
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

    # -----------------------------------------------------------
    # Prolog
    #
    # 1. qkT = tl.dot(k, qT)
    # 2. dpT = tl.dot(v, tl.trans(do))
    # 3. dv += tl.dot(ppT, do)
    # -----------------------------------------------------------

    q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
    do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
    tmem_buf_id, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

    # Compute qkT = tl.dot(k, qT)
    tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
    tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
    qT = tlx.local_trans(q_tiles[q_buf_id])
    tlx.async_dot(
        k_tiles[kv_buf_id],
        qT,
        qk_tiles[tmem_buf_id],
        use_acc=False,
        mBarriers=[qk_fulls[tmem_buf_id]],
    )

    # Compute dpT = tl.dot(v, tl.trans(do))
    tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
    tlx.barrier_wait(dp_empties[tmem_buf_id], tmem_phase ^ 1)
    doT = tlx.local_trans(do_tiles[do_buf_id])
    tlx.async_dot(
        v_tiles[kv_buf_id],
        doT,
        dp_tiles[tmem_buf_id],
        use_acc=False,
        mBarriers=[dp_fulls[tmem_buf_id]],
    )

    # Compute dv += tl.dot(ppT, do)
    tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
    tlx.barrier_wait(dv_empties[kv_buf_id], kv_phase ^ 1)
    tlx.async_dot(
        p_tiles[tmem_buf_id],
        do_tiles[do_buf_id],
        dv_tiles[kv_buf_id],
        use_acc=False,
        mBarriers=[do_empties[do_buf_id]],
    )
    blk_idx += 1
    # -----------------------------------------------------------
    # Main loop
    # 1. qkT = tl.dot(k, qT)
    # 2. dq = tl.dot(tl.trans(dsT), k) from previous iteration
    # 3. dk += tl.dot(dsT, tl.trans(qT)) from previous iteration
    # 4. dpT = tl.dot(v, tl.trans(do))
    # 5. dv += tl.dot(ppT, do)
    # -----------------------------------------------------------
    tlx.barrier_wait(dk_empties[kv_buf_id], kv_phase ^ 1)
    for j in range(1, num_steps):
        q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
        tmem_buf_id, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
        # Compute qkT = tl.dot(k, qT)
        tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
        tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
        qT = tlx.local_trans(q_tiles[q_buf_id])
        tlx.async_dot(
            k_tiles[kv_buf_id],
            qT,
            qk_tiles[tmem_buf_id],
            use_acc=False,
            mBarriers=[qk_fulls[tmem_buf_id]],
        )

        prev_blk_idx = blk_idx - 1
        q_buf_id_prev, _ = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
        tmem_buf_id_prev, tmem_phase_prev = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
        ds_buf_id_prev, ds_phase_prev = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

        # Compute dk += tl.dot(dsT, tl.trans(qT)) from previous iteration
        # Read dsT from TMEM (faster MMA read path than SMEM).
        # dk must read dsT_tmem BEFORE dq writes dq_tiles (same TMEM slot).
        tlx.barrier_wait(dsT_tmem_fulls[ds_buf_id_prev], ds_phase_prev)
        tlx.async_dot(
            dsT_tmem_tiles[ds_buf_id_prev],
            q_tiles[q_buf_id_prev],
            dk_tiles[kv_buf_id],
            use_acc=(j - 1) > 0,
            mBarriers=[
                q_empties[q_buf_id_prev],
            ],
        )

        # Compute dq = tl.dot(tl.trans(dsT), k) from previous iteration
        tlx.barrier_wait(ds_fulls[ds_buf_id_prev], ds_phase_prev)
        tlx.barrier_wait(dq_empties[tmem_buf_id_prev], tmem_phase_prev ^ 1)
        dsT_view = tlx.local_trans(ds_tiles[ds_buf_id_prev])
        tlx.async_dot(
            dsT_view,
            k_tiles[kv_buf_id],
            dq_tiles[tmem_buf_id_prev],
            use_acc=False,
            mBarriers=[
                dq_fulls[tmem_buf_id_prev],
            ],
        )

        do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
        # Compute dpT = tl.dot(v, tl.trans(do))
        tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
        tlx.barrier_wait(dp_empties[tmem_buf_id], tmem_phase ^ 1)
        doT = tlx.local_trans(do_tiles[do_buf_id])
        tlx.async_dot(
            v_tiles[kv_buf_id],
            doT,
            dp_tiles[tmem_buf_id],
            use_acc=False,
            mBarriers=[dp_fulls[tmem_buf_id]],
        )

        # Compute dv += tl.dot(ppT, do)
        tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
        tlx.async_dot(
            p_tiles[tmem_buf_id],
            do_tiles[do_buf_id],
            dv_tiles[kv_buf_id],
            use_acc=True,
            mBarriers=[do_empties[do_buf_id]],
        )
        blk_idx += 1

    tlx.tcgen05_commit(dv_fulls[kv_buf_id])

    # -----------------------------------------------------------
    # Epilog
    # 4. dk += tl.dot(dsT, tl.trans(qT))
    # 5. dq = tl.dot(tl.trans(dsT), k)
    # -----------------------------------------------------------
    prev_blk_idx = blk_idx - 1
    q_buf_id, _ = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
    tmem_buf_id, tmem_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
    ds_buf_id, ds_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)
    # Compute dk += tl.dot(dsT, tl.trans(qT))
    # Read dsT from TMEM (faster MMA read path than SMEM).
    tlx.barrier_wait(dsT_tmem_fulls[ds_buf_id], ds_phase)
    tlx.async_dot(
        dsT_tmem_tiles[ds_buf_id],
        q_tiles[q_buf_id],
        dk_tiles[kv_buf_id],
        use_acc=num_steps > 1,
        mBarriers=[q_empties[q_buf_id], dk_fulls[kv_buf_id]],
    )

    # Compute dq = tl.dot(tl.trans(dsT), k)
    tlx.barrier_wait(ds_fulls[ds_buf_id], ds_phase)
    tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
    dsT_view = tlx.local_trans(ds_tiles[ds_buf_id])
    tlx.async_dot(
        dsT_view,
        k_tiles[kv_buf_id],
        dq_tiles[tmem_buf_id],
        use_acc=False,
        mBarriers=[
            dq_fulls[tmem_buf_id],
        ],
    )
    tlx.tcgen05_commit(k_mma_done[kv_buf_id])

    return blk_idx


@triton.jit
def _bwd_mma_dots_2cta(
    blk_idx,
    num_steps,
    kv_buf_id,
    kv_phase,
    k_tiles,
    v_tiles,
    q_tiles,
    do_tiles,
    qk_tiles,
    qk_fulls,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_tiles,
    dp_fulls,
    dp_empties,
    dv_tiles,
    dv_fulls,
    dv_empties,
    dk_tiles,
    dk_fulls,
    dk_empties,
    dq_tiles,
    dq_fulls,
    dq_empties,
    ds_tiles,
    ds_fulls,
    dsT_tmem_tiles,
    dsT_tmem_fulls,
    do_fulls,
    do_empties,
    q_fulls,
    q_empties,
    k_mma_done,
    k_empties,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    qt_tiles,
    dot_tiles,
    kt_tiles,
    qt_fulls,
    qt_empties,
    dot_fulls,
    dot_empties,
    kt_fulls,
    kt_empties,
    k_fulls,
    v_fulls,
    ds_empties,
    DQ_BUF_OFFSET: tl.constexpr = 0,
    P_BUF_OFFSET: tl.constexpr = 0,
):
    """2-CTA MMA dot sequence: prolog + main loop + epilog.

    Uses qt_tiles/dot_tiles for dots 1,2 and kt_tiles for dot 5.
    All dots use two_ctas=True.

    Differences from 1-CTA:
    - Dots 1,2 use qt_tiles/dot_tiles (transposed views, split along M)
      instead of q_tiles/do_tiles.
    - Dot 5 uses kt_tiles instead of k_tiles.
    - All dots use two_ctas=True for collaborative MMA.
    - K/V have separate barrier waits (not bundled into q_fulls/do_fulls).
    """

    if blk_idx > 0:
        prev_tmem_buf_id, prev_tmem_phase = get_bufidx_phase(
            blk_idx - 1, NUM_BUFFERS_TMEM
        )
        tlx.barrier_wait(
            dq_empties[prev_tmem_buf_id], prev_tmem_phase
        )

    tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)

    q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
    do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
    tmem_buf_id, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

    # Dot 1: qkT = tl.dot(k, qT)
    tlx.barrier_wait(qt_fulls[q_buf_id], q_phase)
    tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
    qT = tlx.local_trans(qt_tiles[q_buf_id])
    tlx.async_dot(
        k_tiles[kv_buf_id],
        qT,
        qk_tiles[tmem_buf_id],
        use_acc=False,
        mBarriers=[qk_fulls[tmem_buf_id], qt_empties[q_buf_id]],
        two_ctas=True,
    )

    # Dot 2: dpT = tl.dot(v, tl.trans(do))
    tlx.barrier_wait(dot_fulls[do_buf_id], do_phase)
    tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)
    doT = tlx.local_trans(dot_tiles[do_buf_id])
    tlx.async_dot(
        v_tiles[kv_buf_id],
        doT,
        dp_tiles[tmem_buf_id],
        use_acc=False,
        mBarriers=[dp_fulls[tmem_buf_id], dot_empties[do_buf_id]],
        two_ctas=True,
    )

    # Dot 3: dv += tl.dot(ppT, do)
    # Wait for do_tiles to be loaded (2-CTA: not bundled into dot_fulls)
    tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
    tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
    tlx.barrier_wait(dv_empties[kv_buf_id], kv_phase ^ 1)
    tlx.async_dot(
        p_tiles[tmem_buf_id + P_BUF_OFFSET],
        do_tiles[do_buf_id],
        dv_tiles[kv_buf_id],
        use_acc=False,
        mBarriers=[do_empties[do_buf_id]],
        two_ctas=True,
    )
    blk_idx += 1

    # -----------------------------------------------------------
    # Main loop
    # Order: S → dK → dP → dQ → dV
    # -----------------------------------------------------------
    tlx.barrier_wait(dk_empties[kv_buf_id], kv_phase ^ 1)
    # kt is loaded once per n-block and reused across the whole m-loop, so wait
    # on it once here (like k_fulls/v_fulls) instead of every iteration.
    tlx.barrier_wait(kt_fulls[kv_buf_id], kv_phase)
    for j in range(1, num_steps):
        q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
        tmem_buf_id, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

        tlx.barrier_wait(qt_fulls[q_buf_id], q_phase)
        tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
        prev_blk_idx = blk_idx - 1
        tmem_buf_id_prev, tmem_phase_prev = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
        tlx.barrier_wait(dq_empties[tmem_buf_id_prev], tmem_phase_prev ^ 1)
        qT = tlx.local_trans(qt_tiles[q_buf_id])
        tlx.async_dot(
            k_tiles[kv_buf_id],
            qT,
            qk_tiles[tmem_buf_id],
            use_acc=False,
            mBarriers=[qk_fulls[tmem_buf_id], qt_empties[q_buf_id]],
            two_ctas=True,
        )

        q_buf_id_prev, q_phase_prev = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
        ds_buf_id_prev, ds_phase_prev = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

        tlx.barrier_wait(q_fulls[q_buf_id_prev], q_phase_prev)
        tlx.barrier_wait(dsT_tmem_fulls[ds_buf_id_prev], ds_phase_prev)
        tlx.async_dot(
            dsT_tmem_tiles[ds_buf_id_prev],
            q_tiles[q_buf_id_prev],
            dk_tiles[kv_buf_id],
            use_acc=(j - 1) > 0,
            mBarriers=[q_empties[q_buf_id_prev], dp_empties[ds_buf_id_prev]],
            two_ctas=True,
        )

        do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
        tlx.barrier_wait(dot_fulls[do_buf_id], do_phase)
        tlx.barrier_wait(dp_empties[tmem_buf_id], tmem_phase ^ 1)
        doT = tlx.local_trans(dot_tiles[do_buf_id])
        tlx.async_dot(
            v_tiles[kv_buf_id],
            doT,
            dp_tiles[tmem_buf_id],
            use_acc=False,
            mBarriers=[dp_fulls[tmem_buf_id], dot_empties[do_buf_id]],
            two_ctas=True,
        )
        # Dot 5: dq = tl.dot(tl.trans(dsT), k)
        # dq_empties[prev] was already waited before Dot 1 (qk aliases the dq
        # TMEM region), and kt_fulls is now waited once before the loop, so
        # neither needs re-waiting here.
        tlx.barrier_wait(ds_fulls[ds_buf_id_prev], ds_phase_prev)
        # dq (cols 0-63) aliases the qk TMEM region, which still holds qk(j);
        # ds_fulls above only proves compute consumed qk(j-1) (one iteration
        # stale on the single-buffered TMEM ring). Wait for qk(j)'s release
        # before Dot 5 overwrites it, else the write races compute's read.
        tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase)
        dsT_view = tlx.local_trans(ds_tiles[ds_buf_id_prev])
        tlx.async_dot(
            dsT_view,
            kt_tiles[kv_buf_id],
            dq_tiles[tmem_buf_id_prev + DQ_BUF_OFFSET],
            use_acc=False,
            mBarriers=[dq_fulls[tmem_buf_id_prev], ds_empties[ds_buf_id_prev]],
            two_ctas=True,
        )
        # Dot 3: dv += tl.dot(ppT, do)
        tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
        tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
        tlx.async_dot(
            p_tiles[tmem_buf_id + P_BUF_OFFSET],
            do_tiles[do_buf_id],
            dv_tiles[kv_buf_id],
            use_acc=True,
            mBarriers=[do_empties[do_buf_id]],
            two_ctas=True,
        )
        blk_idx += 1

    # Commit dv accumulation after all loop iterations
    tlx.tcgen05_commit(dv_fulls[kv_buf_id], two_ctas=True)

    # -----------------------------------------------------------
    # Epilog
    # 4. dk += tl.dot(dsT, q) (TMEM path)
    # 5. dq = tl.dot(tl.trans(dsT), k)
    # -----------------------------------------------------------
    prev_blk_idx = blk_idx - 1
    q_buf_id, q_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
    tmem_buf_id, tmem_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
    ds_buf_id, ds_phase = get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)
    # Compute dk += tl.dot(dsT, q)
    # Read dsT from TMEM (faster MMA read path than SMEM).
    # Wait for q_tiles load (2-CTA: Dot 1 uses qt_tiles, not q_tiles).
    tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
    tlx.barrier_wait(dsT_tmem_fulls[ds_buf_id], ds_phase)
    tlx.async_dot(
        dsT_tmem_tiles[ds_buf_id],
        q_tiles[q_buf_id],
        dk_tiles[kv_buf_id],
        use_acc=num_steps > 1,
        mBarriers=[q_empties[q_buf_id], dk_fulls[kv_buf_id], dp_empties[ds_buf_id]],
        two_ctas=True,
    )

    # Compute dq = tl.dot(tl.trans(dsT), k)
    tlx.barrier_wait(ds_fulls[ds_buf_id], ds_phase)
    tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
    dsT_view = tlx.local_trans(ds_tiles[ds_buf_id])
    tlx.barrier_wait(kt_fulls[kv_buf_id], kv_phase)
    tlx.async_dot(
        dsT_view,
        kt_tiles[kv_buf_id],
        dq_tiles[tmem_buf_id + DQ_BUF_OFFSET],
        use_acc=False,
        mBarriers=[
            dq_fulls[tmem_buf_id],
            ds_empties[ds_buf_id],
        ],
        two_ctas=True,
    )
    tlx.tcgen05_commit(k_mma_done[kv_buf_id], two_ctas=True)
    tlx.tcgen05_commit(kt_empties[kv_buf_id], two_ctas=True)
    # Release k_empties from the leader's mma group (two_ctas updates each CTA's
    # copy). CAVEAT: tracks only the last K read, not the dK/dV staging stores
    # that alias k_tiles/v_tiles — inert today (one-tile-per-block, no refill);
    # once the KV ring cycles, also gate refill on those staging stores.
    tlx.tcgen05_commit(k_empties[kv_buf_id], two_ctas=True)

    return blk_idx


@triton.jit
def _bwd_load_1cta(
    blk_idx,
    off_chz,
    batch,
    head,
    start_m,
    start_n,
    num_steps,
    tile_count,
    desc_k,
    desc_v,
    desc_q,
    desc_do,
    desc_m,
    desc_delta,
    M_ptr,
    delta_ptr,
    k_tiles,
    v_tiles,
    q_tiles,
    do_tiles,
    sM_tiles,
    sD_tiles,
    k_empties,
    q_fulls,
    q_empties,
    do_fulls,
    do_empties,
    m_fulls,
    m_empties,
    d_fulls,
    d_empties,
    K_BYTES_PER_ELEM: tl.constexpr,
    V_BYTES_PER_ELEM: tl.constexpr,
    Q_BYTES_PER_ELEM: tl.constexpr,
    DO_BYTES_PER_ELEM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    cluster_cta_rank,
    is_leader,
):
    start_block_n = start_n * BLOCK_N1
    kv_buf_id, kv_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_KV)

    # Load K+Q bundled on q_fulls (prologue: first m_block includes K)
    curr_m = start_m
    step_m = BLOCK_M1
    q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
    tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
    tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
    tlx.barrier_expect_bytes(
        q_fulls[q_buf_id],
        K_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM + Q_BYTES_PER_ELEM * BLOCK_M1 * (HEAD_DIM // NUM_CTAS))
    tlx.async_descriptor_load(
        desc_k,
        k_tiles[kv_buf_id],
        [batch, head, start_block_n, 0],
        q_fulls[q_buf_id],
    )
    tlx.async_descriptor_load(
        desc_q,
        q_tiles[q_buf_id],
        [batch, head, curr_m, cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
        q_fulls[q_buf_id],
    )

    # Load M (raw bulk copy — no TMA descriptor needed)
    m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
    tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
    tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
    tlx.async_load(M_ptr + off_chz + curr_m, sM_tiles[m_buf_id], bulk=True, barrier=m_fulls[m_buf_id])

    # Load V+dO bundled on do_fulls (prologue: first m_block includes V)
    do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
    tlx.barrier_expect_bytes(
        do_fulls[do_buf_id],
        V_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM + DO_BYTES_PER_ELEM * BLOCK_M1 * (HEAD_DIM // NUM_CTAS))
    tlx.async_descriptor_load(
        desc_v,
        v_tiles[kv_buf_id],
        [batch, head, start_block_n, 0],
        do_fulls[do_buf_id],
    )
    tlx.async_descriptor_load(
        desc_do,
        do_tiles[do_buf_id],
        [batch, head, curr_m, cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
        do_fulls[do_buf_id],
    )

    # Load D (raw bulk copy — no TMA descriptor needed)
    d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
    tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
    tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
    tlx.async_load(delta_ptr + off_chz + curr_m, sD_tiles[d_buf_id], bulk=True, barrier=d_fulls[d_buf_id])

    curr_m += step_m
    blk_idx += 1

    for _ in range(1, num_steps):
        q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
        do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
        # Load Q
        tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
        tlx.barrier_expect_bytes(q_fulls[q_buf_id], Q_BYTES_PER_ELEM * BLOCK_M1 * (HEAD_DIM // NUM_CTAS))
        tlx.async_descriptor_load(
            desc_q,
            q_tiles[q_buf_id],
            [batch, head, curr_m, cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
            q_fulls[q_buf_id],
        )

        # Load M (raw bulk copy)
        m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
        tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
        tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
        tlx.async_load(M_ptr + off_chz + curr_m, sM_tiles[m_buf_id], bulk=True, barrier=m_fulls[m_buf_id])

        # Load dO
        tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
        tlx.barrier_expect_bytes(do_fulls[do_buf_id], DO_BYTES_PER_ELEM * BLOCK_M1 * (HEAD_DIM // NUM_CTAS))
        tlx.async_descriptor_load(
            desc_do,
            do_tiles[do_buf_id],
            [batch, head, curr_m, cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
            do_fulls[do_buf_id],
        )

        # Load D (raw bulk copy)
        d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
        tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
        tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
        tlx.async_load(delta_ptr + off_chz + curr_m, sD_tiles[d_buf_id], bulk=True, barrier=d_fulls[d_buf_id])

        curr_m += step_m
        blk_idx += 1

    return blk_idx


@triton.jit
def _bwd_load_2cta(
    blk_idx,
    off_chz,
    batch,
    head,
    start_m,
    start_n,
    num_steps,
    tile_count,
    desc_k,
    desc_v,
    desc_q,
    desc_do,
    desc_m,
    desc_delta,
    M_ptr,
    delta_ptr,
    k_tiles,
    v_tiles,
    q_tiles,
    do_tiles,
    sM_tiles,
    sD_tiles,
    k_empties,
    dv_empties,
    dk_empties,
    q_fulls,
    q_empties,
    do_fulls,
    do_empties,
    m_fulls,
    m_empties,
    d_fulls,
    d_empties,
    K_BYTES_PER_ELEM: tl.constexpr,
    V_BYTES_PER_ELEM: tl.constexpr,
    Q_BYTES_PER_ELEM: tl.constexpr,
    DO_BYTES_PER_ELEM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    PRENORMALIZED_DO: tl.constexpr,
    cluster_cta_rank,
    is_leader,
    # 2-CTA specific
    k_fulls,
    v_fulls,
    desc_kt,
    desc_qt,
    desc_dot,
    kt_tiles,
    kt_fulls,
    kt_empties,
    qt_tiles,
    qt_fulls,
    qt_empties,
    dot_tiles,
    dot_fulls,
    dot_empties,
):
    start_block_n = start_n * BLOCK_N1
    # Load K — both CTAs load same N-block via two_ctas
    kv_buf_id, kv_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_KV)
    tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
    tlx.barrier_wait(dv_empties[kv_buf_id], kv_phase ^ 1, pred=is_leader)
    tlx.barrier_wait(dk_empties[kv_buf_id], kv_phase ^ 1, pred=is_leader)
    if is_leader:
        tlx.barrier_expect_bytes(k_fulls[kv_buf_id], K_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM * NUM_CTAS)
    tlx.async_descriptor_load(
        desc_k,
        k_tiles[kv_buf_id],
        [batch, head, start_block_n, 0],
        k_fulls[kv_buf_id],
        two_ctas=tl.constexpr(True),
    )

    # Load V
    if is_leader:
        tlx.barrier_expect_bytes(v_fulls[kv_buf_id], V_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM * NUM_CTAS)
    tlx.async_descriptor_load(
        desc_v,
        v_tiles[kv_buf_id],
        [batch, head, start_block_n, 0],
        v_fulls[kv_buf_id],
        two_ctas=tl.constexpr(True),
    )

    # In 2-CTA, skip q_tiles prolog load — dot1 uses qt_tiles, not q_tiles.
    # q_tiles will be first loaded in the inner loop for dk.
    curr_m = start_m
    step_m = BLOCK_M1
    q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
    # Load Qt [HEAD_DIM, BLOCK_M1//2] per CTA (for dots 1,2)
    tlx.barrier_wait(qt_empties[q_buf_id], q_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(qt_fulls[q_buf_id], Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM)
    tlx.async_descriptor_load(
        desc_qt,
        qt_tiles[q_buf_id],
        [batch, head, curr_m + cluster_cta_rank * (BLOCK_M1 // NUM_CTAS), 0],
        qt_fulls[q_buf_id],
        two_ctas=tl.constexpr(True),
    )

    if not PRENORMALIZED_DO:
        m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
        tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
        tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
        tlx.async_load(M_ptr + off_chz + curr_m, sM_tiles[m_buf_id], bulk=True, barrier=m_fulls[m_buf_id])

    # Load dO: [BLOCK_M1, HEAD_DIM//NUM_CTAS] per CTA
    do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(do_fulls[do_buf_id], DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM)
    tlx.async_descriptor_load(
        desc_do,
        do_tiles[do_buf_id],
        [batch, head, curr_m, cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
        do_fulls[do_buf_id],
        two_ctas=tl.constexpr(True),
    )
    # Load dOt [HEAD_DIM, BLOCK_M1//2] per CTA (for dots 1,2)
    tlx.barrier_wait(dot_empties[do_buf_id], do_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(dot_fulls[do_buf_id], DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM)
    tlx.async_descriptor_load(
        desc_dot,
        dot_tiles[do_buf_id],
        [batch, head, curr_m + cluster_cta_rank * (BLOCK_M1 // NUM_CTAS), 0],
        dot_fulls[do_buf_id],
        two_ctas=tl.constexpr(True),
    )

    # Load D (raw bulk copy)
    d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
    tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
    tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
    tlx.async_load(delta_ptr + off_chz + curr_m, sD_tiles[d_buf_id], bulk=True, barrier=d_fulls[d_buf_id])

    # Load Kt (B for dQ = dS @ K), [BLOCK_N1*2, HEAD_DIM//2] per CTA.
    tlx.barrier_wait(kt_empties[kv_buf_id], kv_phase ^ 1)
    lower_start_block_n = start_block_n - cluster_cta_rank * BLOCK_N1
    if is_leader:
        tlx.barrier_expect_bytes(kt_fulls[kv_buf_id], K_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM * NUM_CTAS)
    tlx.async_descriptor_load(
        desc_kt,
        kt_tiles[kv_buf_id],
        [batch, head, lower_start_block_n, cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
        kt_fulls[kv_buf_id],
        two_ctas=tl.constexpr(True),
    )

    curr_m += step_m
    blk_idx += 1

    for _ in range(1, num_steps):
        q_buf_id, q_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
        do_buf_id, do_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)

        tlx.barrier_wait(qt_empties[q_buf_id], q_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(qt_fulls[q_buf_id], Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM)
        tlx.async_descriptor_load(
            desc_qt,
            qt_tiles[q_buf_id],
            [batch, head, curr_m + cluster_cta_rank * (BLOCK_M1 // NUM_CTAS), 0],
            qt_fulls[q_buf_id],
            two_ctas=tl.constexpr(True),
        )

        tlx.barrier_wait(dot_empties[do_buf_id], do_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(dot_fulls[do_buf_id], DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM)
        tlx.async_descriptor_load(
            desc_dot,
            dot_tiles[do_buf_id],
            [batch, head, curr_m + cluster_cta_rank * (BLOCK_M1 // NUM_CTAS), 0],
            dot_fulls[do_buf_id],
            two_ctas=tl.constexpr(True),
        )

        prev_q_buf_id, prev_q_phase = get_bufidx_phase(blk_idx - 1, NUM_BUFFERS_Q)
        tlx.barrier_wait(q_empties[prev_q_buf_id], prev_q_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(q_fulls[prev_q_buf_id], Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM)
        tlx.async_descriptor_load(
            desc_q,
            q_tiles[prev_q_buf_id],
            [batch, head, curr_m - step_m, cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
            q_fulls[prev_q_buf_id],
            two_ctas=tl.constexpr(True),
        )

        if not PRENORMALIZED_DO:
            m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
            tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
            tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
            tlx.async_load(M_ptr + off_chz + curr_m, sM_tiles[m_buf_id], bulk=True, barrier=m_fulls[m_buf_id])

        # Load dO: [BLOCK_M1, HEAD_DIM//NUM_CTAS] per CTA
        tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(do_fulls[do_buf_id], DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM)
        tlx.async_descriptor_load(
            desc_do,
            do_tiles[do_buf_id],
            [batch, head, curr_m, cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
            do_fulls[do_buf_id],
            two_ctas=tl.constexpr(True),
        )

        # Load D (raw bulk copy)
        d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
        tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
        tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
        tlx.async_load(delta_ptr + off_chz + curr_m, sD_tiles[d_buf_id], bulk=True, barrier=d_fulls[d_buf_id])

        curr_m += step_m
        blk_idx += 1

    # Load q_tiles for the last M-block (epilog dk will consume)
    last_q_buf_id, last_q_phase = get_bufidx_phase(blk_idx - 1, NUM_BUFFERS_Q)
    tlx.barrier_wait(q_empties[last_q_buf_id], last_q_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(q_fulls[last_q_buf_id], Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM)
    tlx.async_descriptor_load(
        desc_q,
        q_tiles[last_q_buf_id],
        [batch, head, curr_m - step_m, cluster_cta_rank * (HEAD_DIM // NUM_CTAS)],
        q_fulls[last_q_buf_id],
        two_ctas=tl.constexpr(True),
    )

    return blk_idx


@triton.jit
def _bwd_compute_inner_loop(
    start_n,
    qk_fulls,
    qk_tiles,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_empties,
    dp_fulls,
    dp_tiles,
    ds_tiles,
    ds_fulls,
    dsT_tmem_tiles,
    dsT_tmem_fulls,
    sM_tiles,
    sD_tiles,
    m_fulls,
    m_empties,
    d_fulls,
    d_empties,
    curr_m,
    blk_idx,
    step_m,
    do_out_dtype,
    q_out_dtype,
    qk_scale,
    N_CTX,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    NUM_COMPUTE_SLICES: tl.constexpr,
    STAGE: tl.constexpr,
    REUSE_DP_FOR_DQ: tl.constexpr,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    PRENORMALIZED_DO: tl.constexpr,
    # 2-CTA params (defaults for 1-CTA)
    USE_2CTA: tl.constexpr = False,
    SCALE_QK_IN_KERNEL: tl.constexpr = False,
    NUM_CTAS: tl.constexpr = 1,
    dsT_xchg_tiles=None,
    ds_xchg_tiles=None,
    ds_peer_fulls=None,
    ds_empties=None,
    dsT_fulls=None,
    cluster_cta_rank=0,
    P_BUF_OFFSET: tl.constexpr = 0,
    num_steps_override=0,
):
    _BF16_FIXED_GAUGE: tl.constexpr = 4.055517269
    _EXP2_BF16_BIAS: tl.constexpr = 126.0
    _EXP2_BF16_SCALE: tl.constexpr = 128.0
    _EXP2_MAGIC: tl.constexpr = 12582912.0
    tl.static_assert(not PRENORMALIZED_DO or STAGE != 1)
    start_block_n = start_n * BLOCK_N1
    offs_n = start_block_n + tl.arange(0, BLOCK_N1)
    # dsT (f16) aliases dp's (f32) TMEM region, but TMemBarrierInsertion stays
    # silent there: the mbarrier arrives between the dp read and the dsT store
    # clear its tracking state. This rendezvous is load-bearing; keep it.
    DP_READ_DONE_BAR: tl.constexpr = 11
    NUM_COMPUTE_THREADS: tl.constexpr = 8 * 32
    if num_steps_override > 0:
        num_steps = num_steps_override
    else:
        lo, hi = _get_unfused_bwd_loop_bounds(start_n, N_CTX, BLOCK_N1, STAGE)
        num_steps = (hi - lo) // BLOCK_M1
    for mask_stage in tl.static_range(2 if USE_2CTA and STAGE == 1 else 1):
        if USE_2CTA and STAGE == 1:
            if mask_stage == 0:
                stage_steps = min(num_steps, tl.cdiv(BLOCK_N1 * NUM_CTAS, BLOCK_M1))
            else:
                stage_steps = max(num_steps - tl.cdiv(BLOCK_N1 * NUM_CTAS, BLOCK_M1), 0)
        else:
            stage_steps = num_steps
        for _ in range(stage_steps):
            tmem_buf_id, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
            ds_buf_id, _ = get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)

            d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
            tlx.barrier_wait(qk_fulls[tmem_buf_id], tmem_phase)

            qkT = tlx.local_load(qk_tiles[tmem_buf_id])
            if not PRENORMALIZED_DO:
                m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
                tlx.barrier_wait(m_fulls[m_buf_id], m_phase)
                m = tlx.local_load(sM_tiles[m_buf_id])
            if PRENORMALIZED_DO:
                p_h = _s2_exp2_bf16(
                    qkT,
                    qk_scale * _EXP2_BF16_SCALE,
                    _EXP2_MAGIC + _EXP2_BF16_SCALE * (_EXP2_BF16_BIAS - _BF16_FIXED_GAUGE),
                )
                pT = p_h.to(tl.float32)
            else:
                if SCALE_QK_IN_KERNEL:
                    sT = _fma_f32x2(qkT, qk_scale, -m[None, :])
                else:
                    sT = _sub_f32x2(qkT, m[None, :])
                if STAGE == 1 and (not USE_2CTA or mask_stage == 0):
                    col_limit_left = (offs_n - curr_m)[:, None]
                    sT = _apply_causal_mask(sT, col_limit_left, BLOCK_M1, keep_ge=True)
                pT = tl.math.exp2(sT)

            if PRENORMALIZED_DO and do_out_dtype == tl.bfloat16:
                ppT = p_h
            else:
                ppT = pT.to(do_out_dtype)
            # P (f16) aliases the upper half of the qk (f32) TMEM region; this
            # intra-task WAR is ordered by the compiler's TMemBarrierInsertion pass.
            tlx.local_store(p_tiles[tmem_buf_id + P_BUF_OFFSET], ppT)
            if USE_2CTA:
                tlx.barrier_arrive(qk_empties[tmem_buf_id], 1, remote_cta_rank=0)
                tlx.barrier_arrive(p_fulls[tmem_buf_id], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(qk_empties[tmem_buf_id])
                tlx.barrier_arrive(p_fulls[tmem_buf_id])

            tlx.barrier_wait(dp_fulls[tmem_buf_id], tmem_phase)
            dpT = tlx.local_load(dp_tiles[tmem_buf_id])
            tlx.barrier_wait(d_fulls[d_buf_id], d_phase)
            Di = tlx.local_load(sD_tiles[d_buf_id])
            if not PRENORMALIZED_DO:
                tlx.barrier_arrive(m_empties[m_buf_id])
            tlx.barrier_arrive(d_empties[d_buf_id])
            if PRENORMALIZED_DO and q_out_dtype == tl.bfloat16:
                dp_delta_h = _sub_f32x2(dpT, Di[None, :]).to(tl.bfloat16)
                dsT = _mul_bf16x2(p_h, dp_delta_h)
            else:
                dsT = _mul_f32x2(pT, _sub_f32x2(dpT, Di[None, :]))
                dsT = dsT.to(q_out_dtype)
            # Intra-task WAR rendezvous (see above); TMemBarrierInsertion does not
            # cover this store, so the explicit wait is required.
            tlx.named_barrier_wait(DP_READ_DONE_BAR, NUM_COMPUTE_THREADS)
            tlx.local_store(dsT_tmem_tiles[ds_buf_id], dsT)
            if not REUSE_DP_FOR_DQ and not USE_2CTA:
                tlx.barrier_arrive(dp_empties[tmem_buf_id])
            if USE_2CTA:
                tlx.barrier_arrive(dsT_tmem_fulls[ds_buf_id], 1, remote_cta_rank=0)
                _, ds_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)
                tlx.barrier_wait(ds_empties[ds_buf_id], ds_phase ^ 1)
                peer_rank = 1 - cluster_cta_rank
                if cluster_cta_rank == 0:
                    own_tmem = tlx.local_slice(dsT_tmem_tiles[ds_buf_id], [0, 0], [BLOCK_N1, BLOCK_M1 // NUM_CTAS])
                    peer_tmem = tlx.local_slice(dsT_tmem_tiles[ds_buf_id], [0, BLOCK_M1 // NUM_CTAS],
                                                [BLOCK_N1, BLOCK_M1 // NUM_CTAS])
                    own_smem = tlx.local_slice(ds_tiles[ds_buf_id], [0, 0], [BLOCK_N1, BLOCK_M1 // NUM_CTAS])
                else:
                    own_tmem = tlx.local_slice(dsT_tmem_tiles[ds_buf_id], [0, BLOCK_M1 // NUM_CTAS],
                                               [BLOCK_N1, BLOCK_M1 // NUM_CTAS])
                    peer_tmem = tlx.local_slice(dsT_tmem_tiles[ds_buf_id], [0, 0], [BLOCK_N1, BLOCK_M1 // NUM_CTAS])
                    own_smem = tlx.local_slice(ds_tiles[ds_buf_id], [BLOCK_N1, 0], [BLOCK_N1, BLOCK_M1 // NUM_CTAS])
                own_data = tlx.local_load(own_tmem)
                tlx.local_store(own_smem, own_data)
                peer_data = tlx.local_load(peer_tmem)
                tlx.barrier_arrive(dp_empties[tmem_buf_id], 1, remote_cta_rank=0)
                tlx.local_store(ds_xchg_tiles[ds_buf_id], peer_data)
                tlx.fence("async_shared")
                remote_dst = own_smem
                tlx.barrier_expect_bytes(ds_peer_fulls[ds_buf_id], 2 * BLOCK_N1 * (BLOCK_M1 // NUM_CTAS))
                tlx.async_remote_shmem_copy(
                    dst=remote_dst,
                    src=ds_xchg_tiles[ds_buf_id],
                    remote_cta_rank=peer_rank,
                    barrier=ds_peer_fulls[ds_buf_id],
                )
            else:
                tlx.local_store(ds_tiles[ds_buf_id], dsT)
                tlx.fence("async_shared")
                tlx.barrier_arrive(ds_fulls[ds_buf_id])
                tlx.barrier_arrive(dsT_tmem_fulls[ds_buf_id])

            curr_m += step_m
            blk_idx += 1
    return curr_m, blk_idx


@triton.jit
def _bwd_task_loop_init(
    H,
    Z,
    N_CTX,
    BLOCK_N1: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    PERSISTENT_BWD: tl.constexpr,
    PACKED_DQ: tl.constexpr = False,
    Failure=None,
):
    batch_heads = H * Z
    num_kv_pairs = tl.cdiv(N_CTX, BLOCK_N1 * NUM_CTAS)
    if PERSISTENT_BWD:
        cluster_id = tl.program_id(0) // NUM_CTAS
    else:
        cluster_id = (
            (tl.program_id(2) * H + tl.program_id(1)) * num_kv_pairs
            + tl.program_id(0) // NUM_CTAS
        )
    if PERSISTENT_BWD and PACKED_DQ:
        persistent_stride = tl.num_programs(0) // NUM_CTAS
        num_units = batch_heads * (num_kv_pairs // 2)
        num_outer_tasks = 2 * tl.cdiv(num_units - cluster_id, persistent_stride)
        task_id = tl.where(cluster_id < num_units, 0, -1)
    elif PERSISTENT_BWD:
        persistent_stride = tl.num_programs(0) // NUM_CTAS
        num_owned_heads = tl.cdiv(batch_heads - cluster_id, persistent_stride)
        num_outer_tasks = num_owned_heads * num_kv_pairs
        task_id = tl.where(cluster_id < batch_heads, 0, -1)
    else:
        persistent_stride = 1
        num_outer_tasks = 1
        task_id = cluster_id
    if Failure is not None:
        task_id = tl.where(tl.load(Failure) != 0, task_id, -1)
    return cluster_id, persistent_stride, num_kv_pairs, num_outer_tasks, task_id


@triton.jit
def _decode_bwd_kv_task(
    task_id,
    cluster_id,
    persistent_stride,
    num_kv_pairs,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    STAGE: tl.constexpr,
    PERSISTENT_BWD: tl.constexpr,
    NATIVE_COORDS: tl.constexpr,
    cluster_cta_rank,
    PACKED_DQ: tl.constexpr = False,
):
    if NATIVE_COORDS:
        kv_pair = tl.program_id(0) // NUM_CTAS
        head = tl.program_id(1)
        batch = tl.program_id(2)
        batch_head = batch * H + head
    elif PERSISTENT_BWD and PACKED_DQ:
        pairs_per_head = num_kv_pairs // 2
        unit = cluster_id + (task_id // 2) * persistent_stride
        batch_head = unit // pairs_per_head
        pair = unit % pairs_per_head
        kv_pair = tl.where(task_id % 2 == 0, pair, num_kv_pairs - 1 - pair)
    elif PERSISTENT_BWD:
        kv_pair = task_id % num_kv_pairs
        head_slot = task_id // num_kv_pairs
        batch_head = cluster_id + head_slot * persistent_stride
    else:
        batch_head = task_id // num_kv_pairs
        kv_pair = task_id - batch_head * num_kv_pairs
    if not NATIVE_COORDS:
        head = batch_head % H
        batch = batch_head // H
    base_start_n = kv_pair * NUM_CTAS
    start_n = base_start_n + cluster_cta_rank
    start_m = _get_start_m_bwd(base_start_n, BLOCK_N1, STAGE)
    num_steps = (N_CTX - start_m) // BLOCK_M1
    off_chz = ((batch * H + head) * N_CTX).to(tl.int64)
    start_block_n = start_n * BLOCK_N1
    return kv_pair, start_n, head, batch, off_chz, start_m, num_steps, start_block_n


def _bwd_tuning_reset_pre_hook(nargs, reset_only=False):
    if nargs.get("PACKED_DQ", False):
        nargs["desc_dq"].base.zero_()
        nargs["RangeBudget"].zero_()
        nargs["RangeFailure"].zero_()
    elif (
        nargs.get("PREPROCESS_ZEROES_DQ", False)
        and not nargs.get("PERSISTENT_BWD", False)
    ):
        nargs["desc_dq"].base.zero_()


@triton.jit
def _cold_head_stripe_io(
    DQ64, DQ_OUT, Failure, H, Z, N_CTX,
    DQ64_STRIDES: tl.constexpr, WRITE_ZERO: tl.constexpr,
):
    if tl.load(Failure) != 0:
        rank = tlx.cluster_cta_rank()
        cluster_id = tl.program_id(0) // 2
        cluster_stride = tl.num_programs(0) // 2
        elements_per_rank = N_CTX * 64
        lanes = tl.arange(0, 4096)
        for batch_head in range(cluster_id, H * Z, cluster_stride):
            batch = batch_head // H
            head = batch_head % H
            for offset in range(0, elements_per_rank, 4096):
                linear = offset + lanes
                half_row = linear // 128
                query = (half_row // 64) * 128 + rank * 64 + half_row % 64
                column = linear % 128
                address = (batch.to(tl.int64) * DQ64_STRIDES[0]
                           + head.to(tl.int64) * DQ64_STRIDES[1]
                           + query.to(tl.int64) * DQ64_STRIDES[2]
                           + column.to(tl.int64) * DQ64_STRIDES[3])
                if WRITE_ZERO:
                    tl.store(DQ64 + address, 0.0, linear < elements_per_rank)
                else:
                    value = tl.load(DQ64 + address, linear < elements_per_rank, other=0)
                    tl.store(DQ_OUT + address, value.to(tl.bfloat16), linear < elements_per_rank)



@triton.autotune(
    configs=BWD_CONFIGS,
    key=["N_CTX", "HEAD_DIM", "H", "Z", "STAGE", "PERSISTENT_BWD",
         "SCALE_QK_IN_KERNEL", "PRENORMALIZED_DO", "BSHD_CONFIG"],
    prune_configs_by={"early_config_prune": prune_bwd_configs},
    pre_hook=_bwd_tuning_reset_pre_hook,
)
@triton.jit
def _attn_bwd_ws(
    desc_q,
    desc_k,
    desc_v,
    sm_scale,  #
    desc_do,  #
    desc_dq,
    desc_dk,
    desc_dv,  #
    desc_m,
    desc_delta,
    M_ptr,
    delta_ptr,
    H,
    Z,
    N_CTX,  #
    # 2-CTA descriptors (pass dummy descriptors for 1-CTA).
    desc_kt,
    desc_qt,
    desc_dot,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_COMPUTE_SLICES: tl.constexpr,
    DQ_REDUCE_STAGES: tl.constexpr,
    DQ_REDUCE_NCOL: tl.constexpr,
    DQ_STAGE_COUNT: tl.constexpr,
    DKV_STORE_NCOL: tl.constexpr,
    STAGE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_WARP_BARRIER: tl.constexpr = False,
    EPILOGUE_SUBTILE: tl.constexpr = 4,
    NUM_CTAS: tl.constexpr = 1,
    SCALE_QK_IN_KERNEL: tl.constexpr = False,
    PERSISTENT_BWD: tl.constexpr = False,
    PRENORMALIZED_DO: tl.constexpr = False,
    PREPROCESS_ZEROES_DQ: tl.constexpr = False,
    PACKED_DQ: tl.constexpr = False,
    RangeFailure=None,
    PACKED_N_CTX: tl.constexpr = 0,
    RangeBudget=None,
    Failure=None,
    DQ64=None,
    DQ64_STRIDES: tl.constexpr = None,
    DQ_OUT=None,
    BSHD_CONFIG: tl.constexpr = False,
):
    _RCP_LN2: tl.constexpr = 1.4426950408889634
    if Failure is not None:
        tl.static_assert(not PACKED_DQ)
        tl.static_assert(DQ_OUT is not None and DQ_OUT.dtype.element_ty == tl.bfloat16)
        tl.static_assert(PERSISTENT_BWD and STAGE == 3 and SCALE_QK_IN_KERNEL
                         and not PRENORMALIZED_DO and not USE_WARP_BARRIER
                         and BLOCK_M1 == 128 and BLOCK_N1 == 128 and EPILOGUE_SUBTILE == 8
                         and DQ_STAGE_COUNT == 2 and NUM_BUFFERS_KV == 1 and NUM_BUFFERS_Q == 1
                         and NUM_BUFFERS_DS == 1 and NUM_BUFFERS_TMEM == 1
                         and tlx.dtype_of(desc_q) == tl.bfloat16 and DQ64 is not None
                         and DQ64.dtype.element_ty == tl.float64 and DQ64_STRIDES is not None)
    if PACKED_DQ:
        tl.static_assert(PACKED_N_CTX >= 256 and PACKED_N_CTX <= 16384 and PACKED_N_CTX % 256 == 0)
        tl.static_assert(not PERSISTENT_BWD or PACKED_N_CTX % 512 == 0)
        N_CTX = PACKED_N_CTX
    # Runtime error if NUM_BUFFERS_DO != 1
    tl.static_assert(NUM_BUFFERS_DO == 1)

    # If we have BLOCK_M1 == 128 and HEAD_DIM == 128 we don't have enough
    # TMEM. We may need to expand this condition across other configs in
    # the future.
    # Note: Setting REUSE_DP_FOR_DQ=False with BLOCK_M1 == 64 and
    # HEAD_DIM == 128 will result in an accuracy issue.
    REUSE_DP_FOR_DQ: tl.constexpr = (BLOCK_M1 == 128) and (HEAD_DIM == 128) and (NUM_CTAS == 1)

    USE_2CTA: tl.constexpr = NUM_CTAS == 2
    tl.static_assert(
        not PACKED_DQ or (
            STAGE == 3 and SCALE_QK_IN_KERNEL
            and tlx.dtype_of(desc_q) == tl.bfloat16
            and (not PERSISTENT_BWD or (not USE_WARP_BARRIER and NUM_BUFFERS_KV == 1
                 and NUM_BUFFERS_Q == 1 and NUM_BUFFERS_DS == 1 and NUM_BUFFERS_TMEM == 1))
            and tlx.dtype_of(desc_dq) == tl.int32 and EPILOGUE_SUBTILE == 8
            and BLOCK_M1 == 128 and BLOCK_N1 == 128 and DQ_STAGE_COUNT == 2
            and RangeFailure is not None and RangeBudget is not None
        ),
        "packed dQ requires the guarded causal BF16 direct two-CTA configuration",
    )
    tl.static_assert(
        not SCALE_QK_IN_KERNEL or HEAD_DIM == 64 or (USE_2CTA and HEAD_DIM == 128),
        "QK scaling requires D64 or two-CTA D128",
    )
    tl.static_assert(
        not PRENORMALIZED_DO or SCALE_QK_IN_KERNEL,
        "prenormalized dO requires direct dQ scaling",
    )
    DIRECT_DQ_OUTPUT: tl.constexpr = SCALE_QK_IN_KERNEL and USE_2CTA and HEAD_DIM == 128
    NATIVE_COORDS: tl.constexpr = USE_2CTA and not PERSISTENT_BWD
    qk_scale = sm_scale * _RCP_LN2

    # Compute bytes per element for each tensor type
    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))
    DO_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_do))

    # =========================================================================
    # Allocate all barriers (before SMEM/TMEM allocations)
    # =========================================================================
    M_STAGE: tl.constexpr = 1 if USE_2CTA else 2
    D_STAGE: tl.constexpr = 2

    # K/V are bundled into Q/dO barriers (loaded once per n_block in prologue).
    # k_mma_done: signaled by MMA task after dq dot (last k_tiles read).
    k_mma_done = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_KV,
        arrive_count=2 if USE_2CTA else 1,
    )
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    if PRENORMALIZED_DO:
        m_fulls = None
        m_empties = None
    else:
        m_fulls = tlx.alloc_barriers(num_barriers=M_STAGE)
        m_empties = tlx.alloc_barriers(num_barriers=M_STAGE)
    d_fulls = tlx.alloc_barriers(num_barriers=D_STAGE)
    d_empties = tlx.alloc_barriers(num_barriers=D_STAGE)
    if USE_WARP_BARRIER:
        ds_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_TMEM, num_warps=8)
    else:
        ds_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS)
    dsT_tmem_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS, arrive_count=NUM_CTAS)

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    if USE_WARP_BARRIER:
        qk_empties = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_TMEM, num_warps=8)
        p_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_TMEM, num_warps=8)
    else:
        qk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS)
        p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS)
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    if USE_WARP_BARRIER:
        dq_empties = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_TMEM, num_warps=4)
    else:
        dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS)
    if DIRECT_DQ_OUTPUT and Failure is None:
        dq_stage_fulls = tlx.alloc_warp_barrier(
            num_barriers=DQ_STAGE_COUNT, num_warps=4
        )
        dq_stage_empties = tlx.alloc_barriers(num_barriers=DQ_STAGE_COUNT)
    else:
        dq_stage_fulls = None
        dq_stage_empties = None
    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    if USE_WARP_BARRIER:
        dv_empties = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_KV, num_warps=8)
    else:
        dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=NUM_CTAS)
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    if USE_WARP_BARRIER:
        dk_empties = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_KV, num_warps=8)
    else:
        dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=NUM_CTAS)

    if REUSE_DP_FOR_DQ:
        dp_empties = dq_empties
    else:
        if USE_WARP_BARRIER:
            dp_empties = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_TMEM, num_warps=8)
        elif USE_2CTA:
            # 2-CTA: dp_empties needs arrivals from both MMA (Dot 4 mBarrier)
            # and compute (after DSMEM exchange) before Dot 2 can write dp.
            dp_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS + 1)
        else:
            dp_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    # 2-CTA barriers for transposed views
    if USE_2CTA:
        k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)  # noqa: F841
        v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)  # noqa: F841
        kt_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)  # noqa: F841
        kt_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)  # noqa: F841
        qt_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)  # noqa: F841
        qt_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)  # noqa: F841
        dot_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)  # noqa: F841
        dot_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)  # noqa: F841
        ds_peer_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)  # noqa: F841
        ds_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)  # noqa: F841

    tile_count = 0

    # =========================================================================
    # Allocate SMEM and TMEM buffers
    # =========================================================================
    k_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    v_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)
    q_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM // NUM_CTAS), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    do_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM // NUM_CTAS), tlx.dtype_of(desc_do), NUM_BUFFERS_DO)

    DS_ROWS: tl.constexpr = BLOCK_N1 * NUM_CTAS
    DS_COLS: tl.constexpr = BLOCK_M1 // NUM_CTAS
    ds_tiles = tlx.local_alloc((DS_ROWS, DS_COLS), tlx.dtype_of(desc_q), NUM_BUFFERS_DS)

    DQ_STORE_M: tl.constexpr = BLOCK_M1 // NUM_CTAS
    DQ_SLICE_N: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
    if USE_2CTA and Failure is None:
        DQ_STORE_STAGES: tl.constexpr = 1 if EPILOGUE_SUBTILE == 4 else 2
        DQ_BUFFER_STAGES: tl.constexpr = DQ_STAGE_COUNT if DIRECT_DQ_OUTPUT else DQ_STORE_STAGES
        dq_store_buf = tlx.local_alloc((BLOCK_M1, DQ_SLICE_N), tlx.dtype_of(desc_dq), DQ_BUFFER_STAGES)
    elif Failure is None:
        DQ_REDUCE_ITERS: tl.constexpr = HEAD_DIM // DQ_REDUCE_NCOL
        dq_store_buf = tlx.local_alloc((BLOCK_M1, DQ_REDUCE_NCOL), tlx.dtype_of(desc_dq), DQ_REDUCE_STAGES)

    DKV_STORE_ITERS: tl.constexpr = HEAD_DIM // DKV_STORE_NCOL
    # - sdv reuses v_tiles (free after dv_fulls; MMA's last v_tiles read —
    #   the dpT dot — precedes dv_fulls).
    # - sdk reuses k_tiles (MMA's dq dot still reads k_tiles after dk_fulls,
    #   so the compute task must wait on k_mma_done before writing sdk).
    sdv_store_buf = tlx.local_alloc(
        (BLOCK_N1, DKV_STORE_NCOL), tlx.dtype_of(desc_dv),
        NUM_BUFFERS_KV * DKV_STORE_ITERS, reuse=v_tiles)
    sdk_store_buf = tlx.local_alloc(
        (BLOCK_N1, DKV_STORE_NCOL), tlx.dtype_of(desc_dk),
        NUM_BUFFERS_KV * DKV_STORE_ITERS, reuse=k_tiles)

    if PRENORMALIZED_DO:
        sM_tiles = None
    else:
        sM_tiles = tlx.local_alloc((BLOCK_M1, ), tl.float32, M_STAGE)
    sD_tiles = tlx.local_alloc((BLOCK_M1, ), tl.float32, D_STAGE)

    # S/P/dQ share TMEM via storage alias. S and P fully overlap (shared).
    # In 2-CTA, P and dQ must be distinct (non-overlapping) so that
    # Dot 5 (dQ) doesn't overwrite P before Dot 3 (dV) reads it.
    qk_p_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    qk_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem,
                               reuse=qk_p_storage_alias)
    P_NUM_BUFFERS: tl.constexpr = NUM_BUFFERS_TMEM
    P_BUF_IDX: tl.constexpr = 0
    p_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_do),
        P_NUM_BUFFERS,
        tlx.storage_kind.tmem,
        reuse=qk_p_storage_alias,
    )
    # dP, dS (TMEM for dk dot), and dQ share TMEM via storage alias.
    # dP and dS occupy the same offset (sequential lifetime: dpT consumed
    # before dsT written). dQ occupies a distinct offset (it may overlap
    # with dsT in the mma pipeline).
    dp_dq_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    dp_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=dp_dq_storage_alias,
    )
    dsT_tmem_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_q),
        NUM_BUFFERS_DS,
        tlx.storage_kind.tmem,
        reuse=dp_dq_storage_alias,
    )

    dv_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem)
    dk_tiles = tlx.local_alloc((BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem)

    # dQ uses the same storage alias group as dP/dS — all three share
    # the same TMEM slot.
    # Lifecycle within one block: dpT → dsT → dq (sequential, no overlap).
    if REUSE_DP_FOR_DQ:
        DQ_BUF_IDX: tl.constexpr = 0
        dq_tiles = tlx.local_alloc(
            (BLOCK_M1, HEAD_DIM),
            tl.float32,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
            reuse=dp_dq_storage_alias,
        )
        dp_dq_storage_alias.set_buffer_overlap(
            tlx.reuse_group(
                dp_tiles,
                dsT_tmem_tiles,
                dq_tiles,
                group_type=tlx.reuse_group_type.shared,
            ))
    else:
        if USE_2CTA:
            # 2-CTA: place P and dQ explicitly via set_buffer_overlap.
            # SWAP: P at column 0, dQ at column 64 (was the reverse). dQ's real
            # TwoCTA_RHS footprint is 128x64 (64 i32 cols); storage-alias
            # lowering runs after layout propagation, so it knows this and can
            # place dQ at a non-zero column offset.
            DQ_BUF_IDX: tl.constexpr = 0
            dq_tiles = tlx.local_alloc(
                (BLOCK_M1 // NUM_CTAS, HEAD_DIM),
                tl.float32,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=qk_p_storage_alias,
            )
            dq_phys = tlx.local_alloc(
                (BLOCK_M1, HEAD_DIM // NUM_CTAS),
                tl.float32,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
                reuse=qk_p_storage_alias,
            )
            # qk shares the whole region; within it, P and dQ occupy distinct
            # 64-col halves (P first -> col 0, dQ next -> col 64). dq_tiles and
            # dq_phys are two views of the same dQ storage (shared subgroup).
            qk_p_storage_alias.set_buffer_overlap(
                tlx.reuse_group(
                    qk_tiles,
                    tlx.reuse_group(
                        p_tiles,
                        tlx.reuse_group(
                            dq_tiles,
                            dq_phys,
                            group_type=tlx.reuse_group_type.shared,
                        ),
                        group_type=tlx.reuse_group_type.distinct,
                    ),
                    group_type=tlx.reuse_group_type.shared,
                ))
        else:
            # 1-CTA with bm1=64: separate dQ TMEM
            DQ_BUF_IDX: tl.constexpr = 0
            dq_tiles = tlx.local_alloc(
                (BLOCK_M1, HEAD_DIM),
                tl.float32,
                NUM_BUFFERS_TMEM,
                tlx.storage_kind.tmem,
            )

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    # 2-CTA setup
    if USE_2CTA:
        cluster_cta_rank = tlx.cluster_cta_rank()
        is_leader = cluster_cta_rank == 0
        # Kt tiles: B operand for dQ = dS @ K, shape [BLOCK_N1*2, HEAD_DIM//2] per CTA.
        kt_tiles = tlx.local_alloc((BLOCK_N1 * NUM_CTAS, HEAD_DIM // NUM_CTAS), tlx.dtype_of(desc_k),
                                   NUM_BUFFERS_KV)  # noqa: F841
        # Qt tiles: [BLOCK_M1//2, HEAD_DIM] — for dots 1,2 (transposed B, split along M)
        qt_tiles = tlx.local_alloc((BLOCK_M1 // NUM_CTAS, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)  # noqa: F841
        # dOt tiles: [BLOCK_M1//2, HEAD_DIM] — for dots 1,2
        dot_tiles = tlx.local_alloc((BLOCK_M1 // NUM_CTAS, HEAD_DIM), tlx.dtype_of(desc_do),
                                    NUM_BUFFERS_DO)  # noqa: F841
        # DSMEM exchange staging buffer
        ds_xchg_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1 // NUM_CTAS), tlx.dtype_of(desc_q),
                                        NUM_BUFFERS_DS)  # noqa: F841
        # dp_tiles and dsT_tmem_tiles share the same TMEM
        # (sequential lifetime). Both use dp_dq_storage_alias.
        dp_dq_storage_alias.set_buffer_overlap(
            tlx.reuse_group(
                dp_tiles,
                dsT_tmem_tiles,
                group_type=tlx.reuse_group_type.shared,
            ))
    else:
        cluster_cta_rank = 0
        is_leader = True  # noqa: F841

    if USE_2CTA:
        tlx.fence_mbarrier_init_cluster()

    # `less_reg_mma`: the MMA task issues many dots over the same SMEM
    # operands, so a CSE-d operand descriptor stays live across all of them and
    # spills. Give each MMA its own address computation instead.
    if Failure is not None:
        _cold_head_stripe_io(DQ64, DQ_OUT, Failure, H, Z, N_CTX, DQ64_STRIDES, True)

    with tlx.async_tasks(exclusive=not DIRECT_DQ_OUTPUT, less_reg_mma=True):
        # compute
        with tlx.async_task("default"):
            (cluster_id, persistent_stride, num_kv_pairs,
             num_outer_tasks, task_id) = _bwd_task_loop_init(
                 H, Z, N_CTX, BLOCK_N1, NUM_CTAS, PERSISTENT_BWD, PACKED_DQ, Failure)
            task_iter = 0
            phase_base = 0
            while task_id != -1:
                (kv_pair, start_n, head, batch, off_chz, start_m,
                 num_steps, start_block_n) = _decode_bwd_kv_task(
                     task_id, cluster_id, persistent_stride,
                     num_kv_pairs, H, N_CTX, BLOCK_M1, BLOCK_N1,
                     NUM_CTAS, STAGE, PERSISTENT_BWD, NATIVE_COORDS,
                     cluster_cta_rank, PACKED_DQ)
                tile_count = task_iter
                blk_idx = phase_base
                curr_m = start_m
                step_m = BLOCK_M1
                do_out_dtype = tlx.dtype_of(desc_do)
                q_out_dtype = tlx.dtype_of(desc_q)
                if USE_2CTA and STAGE == 3:
                    curr_m, blk_idx = _bwd_compute_inner_loop(
                        start_n,
                        qk_fulls,
                        qk_tiles,
                        qk_empties,
                        p_tiles,
                        p_fulls,
                        dp_empties,
                        dp_fulls,
                        dp_tiles,
                        ds_tiles,
                        ds_fulls,
                        dsT_tmem_tiles,
                        dsT_tmem_fulls,
                        sM_tiles,
                        sD_tiles,
                        m_fulls,
                        m_empties,
                        d_fulls,
                        d_empties,
                        curr_m,
                        blk_idx,
                        step_m,
                        do_out_dtype,
                        q_out_dtype,
                        qk_scale,
                        N_CTX,
                        NUM_BUFFERS_TMEM,
                        NUM_BUFFERS_DS,
                        BLOCK_M1,
                        BLOCK_N1,
                        NUM_COMPUTE_SLICES,
                        STAGE=1,
                        REUSE_DP_FOR_DQ=REUSE_DP_FOR_DQ,
                        M_STAGE=M_STAGE,
                        D_STAGE=D_STAGE,
                        PRENORMALIZED_DO=PRENORMALIZED_DO,
                        USE_2CTA=USE_2CTA,
                        SCALE_QK_IN_KERNEL=SCALE_QK_IN_KERNEL,
                        NUM_CTAS=NUM_CTAS,
                        dsT_xchg_tiles=None,
                        ds_xchg_tiles=ds_xchg_tiles if USE_2CTA else None,
                        ds_peer_fulls=ds_peer_fulls if USE_2CTA else None,
                        ds_empties=ds_empties if USE_2CTA else None,
                        dsT_fulls=None,
                        cluster_cta_rank=cluster_cta_rank,
                        P_BUF_OFFSET=P_BUF_IDX,
                        num_steps_override=num_steps,
                    )
                else:
                    if STAGE & 1:
                        curr_m, blk_idx = _bwd_compute_inner_loop(
                            start_n,
                            qk_fulls,
                            qk_tiles,
                            qk_empties,
                            p_tiles,
                            p_fulls,
                            dp_empties,
                            dp_fulls,
                            dp_tiles,
                            ds_tiles,
                            ds_fulls,
                            dsT_tmem_tiles,
                            dsT_tmem_fulls,
                            sM_tiles,
                            sD_tiles,
                            m_fulls,
                            m_empties,
                            d_fulls,
                            d_empties,
                            curr_m,
                            blk_idx,
                            step_m,
                            do_out_dtype,
                            q_out_dtype,
                            qk_scale,
                            N_CTX,
                            NUM_BUFFERS_TMEM,
                            NUM_BUFFERS_DS,
                            BLOCK_M1,
                            BLOCK_N1,
                            NUM_COMPUTE_SLICES,
                            STAGE=4 - STAGE,
                            REUSE_DP_FOR_DQ=REUSE_DP_FOR_DQ,
                            M_STAGE=M_STAGE,
                            D_STAGE=D_STAGE,
                            PRENORMALIZED_DO=PRENORMALIZED_DO,
                            USE_2CTA=USE_2CTA,
                            SCALE_QK_IN_KERNEL=SCALE_QK_IN_KERNEL,
                            NUM_CTAS=NUM_CTAS,
                            dsT_xchg_tiles=None,
                            ds_xchg_tiles=ds_xchg_tiles if USE_2CTA else None,
                            ds_peer_fulls=ds_peer_fulls if USE_2CTA else None,
                            ds_empties=ds_empties if USE_2CTA else None,
                            dsT_fulls=None,
                            cluster_cta_rank=cluster_cta_rank,
                            P_BUF_OFFSET=P_BUF_IDX,
                        )
                    if STAGE & 2:
                        curr_m, blk_idx = _bwd_compute_inner_loop(
                            start_n,
                            qk_fulls,
                            qk_tiles,
                            qk_empties,
                            p_tiles,
                            p_fulls,
                            dp_empties,
                            dp_fulls,
                            dp_tiles,
                            ds_tiles,
                            ds_fulls,
                            dsT_tmem_tiles,
                            dsT_tmem_fulls,
                            sM_tiles,
                            sD_tiles,
                            m_fulls,
                            m_empties,
                            d_fulls,
                            d_empties,
                            curr_m,
                            blk_idx,
                            step_m,
                            do_out_dtype,
                            q_out_dtype,
                            qk_scale,
                            N_CTX,
                            NUM_BUFFERS_TMEM,
                            NUM_BUFFERS_DS,
                            BLOCK_M1,
                            BLOCK_N1,
                            NUM_COMPUTE_SLICES,
                            STAGE=2,
                            REUSE_DP_FOR_DQ=REUSE_DP_FOR_DQ,
                            M_STAGE=M_STAGE,
                            D_STAGE=D_STAGE,
                            PRENORMALIZED_DO=PRENORMALIZED_DO,
                            USE_2CTA=USE_2CTA,
                            SCALE_QK_IN_KERNEL=SCALE_QK_IN_KERNEL,
                            NUM_CTAS=NUM_CTAS,
                            dsT_xchg_tiles=None,
                            ds_xchg_tiles=ds_xchg_tiles if USE_2CTA else None,
                            ds_peer_fulls=ds_peer_fulls if USE_2CTA else None,
                            ds_empties=ds_empties if USE_2CTA else None,
                            dsT_fulls=None,
                            cluster_cta_rank=cluster_cta_rank,
                            P_BUF_OFFSET=P_BUF_IDX,
                        )

                kv_buf_id, kv_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_KV)

                tlx.barrier_wait(dv_fulls[kv_buf_id], kv_phase)
                for slice_id in tl.static_range(DKV_STORE_ITERS):
                    dv_slice = tlx.local_slice(
                        dv_tiles[kv_buf_id],
                        [0, slice_id * DKV_STORE_NCOL],
                        [BLOCK_N1, DKV_STORE_NCOL],
                    )
                    dv = tlx.local_load(dv_slice)
                    store_buf_id = kv_buf_id * DKV_STORE_ITERS + slice_id
                    tlx.async_descriptor_store_wait(DKV_STORE_ITERS - 1)
                    tlx.local_store(sdv_store_buf[store_buf_id], dv.to(tlx.dtype_of(desc_dv)))
                    tlx.async_descriptor_store(
                        desc_dv,
                        sdv_store_buf[store_buf_id],
                        [batch, head, start_block_n, slice_id * DKV_STORE_NCOL],
                    )
                if USE_2CTA:
                    tlx.barrier_arrive(dv_empties[kv_buf_id], 1, remote_cta_rank=0)
                else:
                    tlx.barrier_arrive(dv_empties[kv_buf_id])
                tlx.barrier_wait(dk_fulls[kv_buf_id], kv_phase)
                tlx.barrier_wait(k_mma_done[kv_buf_id], kv_phase)
                for slice_id in tl.static_range(DKV_STORE_ITERS):
                    dk_slice = tlx.local_slice(
                        dk_tiles[kv_buf_id],
                        [0, slice_id * DKV_STORE_NCOL],
                        [BLOCK_N1, DKV_STORE_NCOL],
                    )
                    dk = tlx.local_load(dk_slice)
                    dk *= sm_scale
                    store_buf_id = kv_buf_id * DKV_STORE_ITERS + slice_id
                    tlx.async_descriptor_store_wait(DKV_STORE_ITERS - 1)
                    tlx.local_store(sdk_store_buf[store_buf_id], dk.to(tlx.dtype_of(desc_dk)))
                    tlx.async_descriptor_store(
                        desc_dk,
                        sdk_store_buf[store_buf_id],
                        [batch, head, start_block_n, slice_id * DKV_STORE_NCOL],
                    )
                tlx.async_descriptor_store_wait(0)
                if USE_2CTA:
                    tlx.barrier_arrive(k_empties[kv_buf_id])
                    tlx.barrier_arrive(dk_empties[kv_buf_id], 1, remote_cta_rank=0)
                else:
                    tlx.barrier_arrive(k_empties[kv_buf_id])
                    tlx.barrier_arrive(dk_empties[kv_buf_id])
                phase_base += num_steps
                task_iter += 1
                task_id = tl.where(task_iter < num_outer_tasks, task_iter, -1)

        # reduction
        with tlx.async_task(num_warps=4, registers=88):
            (cluster_id, persistent_stride, num_kv_pairs,
             num_outer_tasks, task_id) = _bwd_task_loop_init(
                 H, Z, N_CTX, BLOCK_N1, NUM_CTAS, PERSISTENT_BWD, PACKED_DQ, Failure)
            task_iter = 0
            phase_base = 0
            while task_id != -1:
                (kv_pair, start_n, head, batch, off_chz, start_m,
                 num_steps, start_block_n) = _decode_bwd_kv_task(
                     task_id, cluster_id, persistent_stride,
                     num_kv_pairs, H, N_CTX, BLOCK_M1, BLOCK_N1,
                     NUM_CTAS, STAGE, PERSISTENT_BWD, NATIVE_COORDS,
                     cluster_cta_rank, PACKED_DQ)
                tile_count = task_iter
                blk_idx = phase_base
                curr_m = start_m
                step_m = BLOCK_M1
                for _ in range(num_steps):
                    tmem_buf_id, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

                    tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                    if Failure is not None:
                        rows = tl.arange(0, 128)
                        columns = tl.arange(0, 16)
                        query_rows = curr_m + cluster_cta_rank * 64 + rows % 64
                        dq_base = (batch.to(tl.int64) * DQ64_STRIDES[0]
                                   + head.to(tl.int64) * DQ64_STRIDES[1]
                                   + query_rows.to(tl.int64) * DQ64_STRIDES[2])
                        for slice_id in tl.static_range(4):
                            dq_slice = tlx.local_slice(
                                dq_phys[tmem_buf_id + DQ_BUF_IDX],
                                [0, slice_id * 16],
                                [128, 16],
                            )
                            dq = tlx.local_load(dq_slice)
                            dq = _mul_f32x2(dq, sm_scale)
                            dq_columns = slice_id * 16 + columns[None, :] + (rows[:, None] // 64) * 64
                            dq_offsets = dq_base[:, None] + dq_columns.to(tl.int64) * DQ64_STRIDES[3]
                            tl.atomic_add(DQ64 + dq_offsets, dq.to(tl.float64), sem="relaxed")
                        tlx.barrier_arrive(dq_empties[tmem_buf_id], 1, remote_cta_rank=0)

                    elif USE_2CTA:
                        dq_m_offset = cluster_cta_rank * DQ_STORE_M
                        DQ_PACK_ITERS: tl.constexpr = (HEAD_DIM // NUM_CTAS) // (DQ_SLICE_N * (2 if PACKED_DQ else 1))
                        if DIRECT_DQ_OUTPUT:
                            dq_full = tlx.local_load(dq_phys[tmem_buf_id + DQ_BUF_IDX])
                            tlx.barrier_arrive(
                                dq_empties[tmem_buf_id], 1, remote_cta_rank=0
                            )
                            if PACKED_DQ:
                                if curr_m < 256:
                                    dq_packed = _dq_pack_bf16_pair(_mul_f32x2(dq_full, sm_scale))
                                else:
                                    budget_rows = tl.arange(0, BLOCK_M1)
                                    budget_offsets = (2 * off_chz + (budget_rows // DQ_STORE_M) * N_CTX
                                                      + curr_m + dq_m_offset + budget_rows % DQ_STORE_M)
                                    dq_packed = _dq_pack_fixed_pair(
                                        dq_full, sm_scale * _dq_fixed_scale(N_CTX), curr_m // 256 + 1,
                                        RangeBudget, budget_offsets,
                                    )
                                dq_full = dq_packed
                            dq_slices = _split_n_2D(dq_full, DQ_PACK_ITERS)
                            for slice_id in tl.static_range(DQ_PACK_ITERS):
                                dq_stage_count = blk_idx * DQ_PACK_ITERS + slice_id
                                dq_stage_buf_id, dq_stage_phase = get_bufidx_phase(
                                    dq_stage_count, DQ_STAGE_COUNT
                                )
                                tlx.barrier_wait(
                                    dq_stage_empties[dq_stage_buf_id],
                                    dq_stage_phase ^ 1,
                                )
                                dq_smem = dq_store_buf[dq_stage_buf_id]
                                tlx.local_store(
                                    dq_smem,
                                    (dq_slices[slice_id] if PACKED_DQ else dq_slices[slice_id] * sm_scale).to(
                                        tlx.dtype_of(desc_dq)
                                    ),
                                )
                                tlx.fence("async_shared")
                                tlx.barrier_arrive(dq_stage_fulls[dq_stage_buf_id])
                        else:
                            packed_row_base = 2 * (curr_m + dq_m_offset)
                            dq_full = tlx.local_load(dq_phys[tmem_buf_id + DQ_BUF_IDX])
                            if USE_WARP_BARRIER:
                                tlx.barrier_arrive(dq_empties[tmem_buf_id])
                            else:
                                tlx.barrier_arrive(
                                    dq_empties[tmem_buf_id], 1, remote_cta_rank=0
                                )
                            dq_full = dq_full * (sm_scale if SCALE_QK_IN_KERNEL else LN2)
                            dq_slices = _split_n_2D(dq_full, DQ_PACK_ITERS)
                            for slice_id in tl.static_range(DQ_PACK_ITERS):
                                dq_smem = dq_store_buf[slice_id % DQ_STORE_STAGES]
                                tlx.async_descriptor_store_wait(DQ_STORE_STAGES - 1)
                                tlx.local_store(
                                    dq_smem,
                                    dq_slices[slice_id].to(tlx.dtype_of(desc_dq)),
                                )
                                tlx.async_descriptor_store(
                                    desc_dq,
                                    dq_smem,
                                    [
                                        batch,
                                        head,
                                        packed_row_base,
                                        slice_id * DQ_SLICE_N,
                                    ],
                                    store_reduce="add",
                                )
                    else:
                        HALF_HD: tl.constexpr = HEAD_DIM // 2
                        SLICES_PER_HALF: tl.constexpr = HALF_HD // DQ_REDUCE_NCOL
                        for slice_id in tl.static_range(DQ_REDUCE_ITERS):
                            dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                            dq_slice = tlx.local_slice(
                                dq_tiles[tmem_buf_id],
                                [0, slice_id * DQ_REDUCE_NCOL],
                                [BLOCK_M1, DQ_REDUCE_NCOL],
                            )
                            dq = tlx.local_load(dq_slice)
                            if SCALE_QK_IN_KERNEL:
                                dq = dq * sm_scale
                            else:
                                dq = dq * LN2
                            tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                            tlx.local_store(
                                dq_store_buf[dq_smem_idx],
                                dq.to(tlx.dtype_of(desc_dq)),
                            )
                            packed_half = slice_id // SLICES_PER_HALF
                            packed_col = (slice_id % SLICES_PER_HALF) * DQ_REDUCE_NCOL
                            tlx.async_descriptor_store(
                                desc_dq,
                                dq_store_buf[dq_smem_idx],
                                [
                                    batch,
                                    head,
                                    2 * curr_m + BLOCK_M1 * packed_half,
                                    packed_col,
                                ],
                                store_reduce="add",
                            )
                        tlx.barrier_arrive(dq_empties[tmem_buf_id])

                    curr_m += step_m
                    blk_idx += 1

                if Failure is None:
                    tlx.async_descriptor_store_wait(0)
                phase_base += num_steps
                task_iter += 1
                task_id = tl.where(task_iter < num_outer_tasks, task_iter, -1)

        if USE_2CTA and DIRECT_DQ_OUTPUT and Failure is None:
            with tlx.async_task(num_warps=1, registers=88):
                (cluster_id, persistent_stride, num_kv_pairs,
                 num_outer_tasks, task_id) = _bwd_task_loop_init(
                     H, Z, N_CTX, BLOCK_N1, NUM_CTAS, PERSISTENT_BWD, PACKED_DQ, Failure)
                task_iter = 0
                phase_base = 0
                while task_id != -1:
                    (kv_pair, start_n, head, batch, off_chz, start_m,
                     num_steps, start_block_n) = _decode_bwd_kv_task(
                         task_id, cluster_id, persistent_stride,
                         num_kv_pairs, H, N_CTX, BLOCK_M1, BLOCK_N1,
                         NUM_CTAS, STAGE, PERSISTENT_BWD, NATIVE_COORDS,
                         cluster_cta_rank, PACKED_DQ)
                    tile_count = task_iter
                    curr_m = start_m
                    dq_m_offset = cluster_cta_rank * DQ_STORE_M
                    DQ_PACK_ITERS: tl.constexpr = (
                        HEAD_DIM // NUM_CTAS // (DQ_SLICE_N * (2 if PACKED_DQ else 1))
                    )
                    for blk_local_idx in range(num_steps):
                        blk_idx = phase_base + blk_local_idx
                        for slice_id in tl.static_range(DQ_PACK_ITERS):
                            dq_stage_count = blk_idx * DQ_PACK_ITERS + slice_id
                            dq_stage_buf_id, dq_stage_phase = get_bufidx_phase(
                                dq_stage_count, DQ_STAGE_COUNT
                            )
                            local_stage_count = blk_local_idx * DQ_PACK_ITERS + slice_id
                            if local_stage_count >= DQ_STAGE_COUNT:
                                tlx.async_descriptor_store_wait(DQ_STAGE_COUNT - 1)
                                tlx.barrier_arrive(
                                    dq_stage_empties[dq_stage_buf_id]
                                )
                            tlx.barrier_wait(
                                dq_stage_fulls[dq_stage_buf_id], dq_stage_phase
                            )
                            dq_smem = dq_store_buf[dq_stage_buf_id]
                            if (PERSISTENT_BWD and not PACKED_DQ and kv_pair == 0) or (PACKED_DQ and curr_m < 256):
                                tlx.async_descriptor_store(
                                    desc_dq,
                                    dq_smem,
                                    [
                                        batch,
                                        head,
                                        0,
                                        curr_m + dq_m_offset,
                                        slice_id * DQ_SLICE_N,
                                    ],
                                )
                            else:
                                tlx.async_descriptor_store(
                                    desc_dq,
                                    dq_smem,
                                    [
                                        batch,
                                        head,
                                        0,
                                        curr_m + dq_m_offset,
                                        slice_id * DQ_SLICE_N,
                                    ],
                                    store_reduce="add",
                                )
                        curr_m += BLOCK_M1
                    tlx.async_descriptor_store_wait(0)
                    for final_stage_id in tl.static_range(DQ_STAGE_COUNT):
                        tlx.barrier_arrive(dq_stage_empties[final_stage_id])
                    phase_base += num_steps
                    task_iter += 1
                    task_id = tl.where(task_iter < num_outer_tasks, task_iter, -1)

        with tlx.async_task(num_warps=1, registers=88):
            (cluster_id, persistent_stride, num_kv_pairs,
             num_outer_tasks, task_id) = _bwd_task_loop_init(
                 H, Z, N_CTX, BLOCK_N1, NUM_CTAS, PERSISTENT_BWD, PACKED_DQ, Failure)
            task_iter = 0
            phase_base = 0
            while task_id != -1:
                (kv_pair, start_n, head, batch, off_chz, start_m,
                 num_steps, start_block_n) = _decode_bwd_kv_task(
                     task_id, cluster_id, persistent_stride,
                     num_kv_pairs, H, N_CTX, BLOCK_M1, BLOCK_N1,
                     NUM_CTAS, STAGE, PERSISTENT_BWD, NATIVE_COORDS,
                     cluster_cta_rank, PACKED_DQ)
                tile_count = task_iter
                blk_idx = phase_base
                if is_leader:
                    kv_buf_id, kv_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_KV)
                    if USE_2CTA:
                        blk_idx = _bwd_mma_dots_2cta(
                            blk_idx=blk_idx,
                            num_steps=num_steps,
                            kv_buf_id=kv_buf_id,
                            kv_phase=kv_phase,
                            k_tiles=k_tiles,
                            v_tiles=v_tiles,
                            q_tiles=q_tiles,
                            do_tiles=do_tiles,
                            qk_tiles=qk_tiles,
                            qk_fulls=qk_fulls,
                            qk_empties=qk_empties,
                            p_tiles=p_tiles,
                            p_fulls=p_fulls,
                            dp_tiles=dp_tiles,
                            dp_fulls=dp_fulls,
                            dp_empties=dp_empties,
                            dv_tiles=dv_tiles,
                            dv_fulls=dv_fulls,
                            dv_empties=dv_empties,
                            dk_tiles=dk_tiles,
                            dk_fulls=dk_fulls,
                            dk_empties=dk_empties,
                            dq_tiles=dq_tiles,
                            dq_fulls=dq_fulls,
                            dq_empties=dq_empties,
                            ds_tiles=ds_tiles,
                            ds_fulls=ds_fulls,
                            dsT_tmem_tiles=dsT_tmem_tiles,
                            dsT_tmem_fulls=dsT_tmem_fulls,
                            do_fulls=do_fulls,
                            do_empties=do_empties,
                            q_fulls=q_fulls,
                            q_empties=q_empties,
                            k_mma_done=k_mma_done,
                            k_empties=k_empties,
                            NUM_BUFFERS_Q=NUM_BUFFERS_Q,
                            NUM_BUFFERS_DO=NUM_BUFFERS_DO,
                            NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                            NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                            BLOCK_N1=BLOCK_N1,
                            qt_tiles=qt_tiles,
                            dot_tiles=dot_tiles,
                            kt_tiles=kt_tiles,
                            qt_fulls=qt_fulls,
                            qt_empties=qt_empties,
                            dot_fulls=dot_fulls,
                            dot_empties=dot_empties,
                            kt_fulls=kt_fulls,
                            kt_empties=kt_empties,
                            k_fulls=k_fulls,
                            v_fulls=v_fulls,
                            ds_empties=ds_empties,
                            DQ_BUF_OFFSET=DQ_BUF_IDX,
                            P_BUF_OFFSET=P_BUF_IDX,
                        )
                    else:
                        blk_idx = _bwd_mma_dots_1cta(
                            blk_idx=blk_idx,
                            num_steps=num_steps,
                            kv_buf_id=kv_buf_id,
                            kv_phase=kv_phase,
                            k_tiles=k_tiles,
                            v_tiles=v_tiles,
                            q_tiles=q_tiles,
                            do_tiles=do_tiles,
                            qk_tiles=qk_tiles,
                            qk_fulls=qk_fulls,
                            qk_empties=qk_empties,
                            p_tiles=p_tiles,
                            p_fulls=p_fulls,
                            dp_tiles=dp_tiles,
                            dp_fulls=dp_fulls,
                            dp_empties=dp_empties,
                            dv_tiles=dv_tiles,
                            dv_fulls=dv_fulls,
                            dv_empties=dv_empties,
                            dk_tiles=dk_tiles,
                            dk_fulls=dk_fulls,
                            dk_empties=dk_empties,
                            dq_tiles=dq_tiles,
                            dq_fulls=dq_fulls,
                            dq_empties=dq_empties,
                            ds_tiles=ds_tiles,
                            ds_fulls=ds_fulls,
                            dsT_tmem_tiles=dsT_tmem_tiles,
                            dsT_tmem_fulls=dsT_tmem_fulls,
                            do_fulls=do_fulls,
                            do_empties=do_empties,
                            q_fulls=q_fulls,
                            q_empties=q_empties,
                            k_mma_done=k_mma_done,
                            NUM_BUFFERS_Q=NUM_BUFFERS_Q,
                            NUM_BUFFERS_DO=NUM_BUFFERS_DO,
                            NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                            NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                            BLOCK_M1=BLOCK_M1,
                            BLOCK_N1=BLOCK_N1,
                        )
                    tile_count += 1
                phase_base += num_steps
                task_iter += 1
                task_id = tl.where(task_iter < num_outer_tasks, task_iter, -1)

        with tlx.async_task(num_warps=1, registers=88):
            (cluster_id, persistent_stride, num_kv_pairs,
             num_outer_tasks, task_id) = _bwd_task_loop_init(
                 H, Z, N_CTX, BLOCK_N1, NUM_CTAS, PERSISTENT_BWD, PACKED_DQ, Failure)
            task_iter = 0
            phase_base = 0
            while task_id != -1:
                (kv_pair, start_n, head, batch, off_chz, start_m,
                 num_steps, start_block_n) = _decode_bwd_kv_task(
                     task_id, cluster_id, persistent_stride,
                     num_kv_pairs, H, N_CTX, BLOCK_M1, BLOCK_N1,
                     NUM_CTAS, STAGE, PERSISTENT_BWD, NATIVE_COORDS,
                     cluster_cta_rank, PACKED_DQ)
                tile_count = task_iter
                blk_idx = phase_base
                if USE_2CTA:
                    blk_idx = _bwd_load_2cta(
                        blk_idx=blk_idx,
                        off_chz=off_chz,
                        batch=batch,
                        head=head,
                        start_m=start_m,
                        start_n=start_n,
                        num_steps=num_steps,
                        tile_count=tile_count,
                        desc_k=desc_k,
                        desc_v=desc_v,
                        desc_q=desc_q,
                        desc_do=desc_do,
                        desc_m=desc_m,
                        desc_delta=desc_delta,
                        M_ptr=M_ptr,
                        delta_ptr=delta_ptr,
                        k_tiles=k_tiles,
                        v_tiles=v_tiles,
                        q_tiles=q_tiles,
                        do_tiles=do_tiles,
                        sM_tiles=sM_tiles,
                        sD_tiles=sD_tiles,
                        k_empties=k_empties,
                        dv_empties=dv_empties,
                        dk_empties=dk_empties,
                        q_fulls=q_fulls,
                        q_empties=q_empties,
                        do_fulls=do_fulls,
                        do_empties=do_empties,
                        m_fulls=m_fulls,
                        m_empties=m_empties,
                        d_fulls=d_fulls,
                        d_empties=d_empties,
                        K_BYTES_PER_ELEM=K_BYTES_PER_ELEM,
                        V_BYTES_PER_ELEM=V_BYTES_PER_ELEM,
                        Q_BYTES_PER_ELEM=Q_BYTES_PER_ELEM,
                        DO_BYTES_PER_ELEM=DO_BYTES_PER_ELEM,
                        BLOCK_M1=BLOCK_M1,
                        BLOCK_N1=BLOCK_N1,
                        NUM_BUFFERS_KV=NUM_BUFFERS_KV,
                        NUM_BUFFERS_Q=NUM_BUFFERS_Q,
                        NUM_BUFFERS_DO=NUM_BUFFERS_DO,
                        M_STAGE=M_STAGE,
                        D_STAGE=D_STAGE,
                        HEAD_DIM=HEAD_DIM,
                        STAGE=STAGE,
                        NUM_CTAS=NUM_CTAS,
                        PRENORMALIZED_DO=PRENORMALIZED_DO,
                        cluster_cta_rank=cluster_cta_rank,
                        is_leader=is_leader,
                        k_fulls=k_fulls,
                        v_fulls=v_fulls,
                        desc_kt=desc_kt,
                        desc_qt=desc_qt,
                        desc_dot=desc_dot,
                        kt_tiles=kt_tiles,
                        kt_fulls=kt_fulls,
                        kt_empties=kt_empties,
                        qt_tiles=qt_tiles,
                        qt_fulls=qt_fulls,
                        qt_empties=qt_empties,
                        dot_tiles=dot_tiles,
                        dot_fulls=dot_fulls,
                        dot_empties=dot_empties,
                    )
                else:
                    blk_idx = _bwd_load_1cta(
                        blk_idx=blk_idx,
                        off_chz=off_chz,
                        batch=batch,
                        head=head,
                        start_m=start_m,
                        start_n=start_n,
                        num_steps=num_steps,
                        tile_count=tile_count,
                        desc_k=desc_k,
                        desc_v=desc_v,
                        desc_q=desc_q,
                        desc_do=desc_do,
                        desc_m=desc_m,
                        desc_delta=desc_delta,
                        M_ptr=M_ptr,
                        delta_ptr=delta_ptr,
                        k_tiles=k_tiles,
                        v_tiles=v_tiles,
                        q_tiles=q_tiles,
                        do_tiles=do_tiles,
                        sM_tiles=sM_tiles,
                        sD_tiles=sD_tiles,
                        k_empties=k_empties,
                        q_fulls=q_fulls,
                        q_empties=q_empties,
                        do_fulls=do_fulls,
                        do_empties=do_empties,
                        m_fulls=m_fulls,
                        m_empties=m_empties,
                        d_fulls=d_fulls,
                        d_empties=d_empties,
                        K_BYTES_PER_ELEM=K_BYTES_PER_ELEM,
                        V_BYTES_PER_ELEM=V_BYTES_PER_ELEM,
                        Q_BYTES_PER_ELEM=Q_BYTES_PER_ELEM,
                        DO_BYTES_PER_ELEM=DO_BYTES_PER_ELEM,
                        BLOCK_M1=BLOCK_M1,
                        BLOCK_N1=BLOCK_N1,
                        NUM_BUFFERS_KV=NUM_BUFFERS_KV,
                        NUM_BUFFERS_Q=NUM_BUFFERS_Q,
                        NUM_BUFFERS_DO=NUM_BUFFERS_DO,
                        M_STAGE=M_STAGE,
                        D_STAGE=D_STAGE,
                        HEAD_DIM=HEAD_DIM,
                        STAGE=STAGE,
                        NUM_CTAS=NUM_CTAS,
                        cluster_cta_rank=cluster_cta_rank,
                        is_leader=is_leader,
                    )
                phase_base += num_steps
                task_iter += 1
                task_id = tl.where(task_iter < num_outer_tasks, task_iter, -1)

        # relay — waits for peer's DSMEM to arrive, then signals ds_fulls
        # so the MMA task can read the combined ds_tiles.
        if USE_2CTA:
            with tlx.async_task(num_warps=1, registers=40):
                (cluster_id, persistent_stride, num_kv_pairs,
                 num_outer_tasks, task_id) = _bwd_task_loop_init(
                     H, Z, N_CTX, BLOCK_N1, NUM_CTAS, PERSISTENT_BWD, PACKED_DQ, Failure)
                task_iter = 0
                phase_base = 0
                while task_id != -1:
                    (kv_pair, start_n, head, batch, off_chz, start_m,
                     num_steps, start_block_n) = _decode_bwd_kv_task(
                         task_id, cluster_id, persistent_stride,
                         num_kv_pairs, H, N_CTX, BLOCK_M1, BLOCK_N1,
                         NUM_CTAS, STAGE, PERSISTENT_BWD, NATIVE_COORDS,
                         cluster_cta_rank, PACKED_DQ)
                    tile_count = task_iter
                    for blk_local_idx_relay in range(num_steps):
                        blk_idx_relay = phase_base + blk_local_idx_relay
                        ds_buf_id_relay, ds_phase_relay = get_bufidx_phase(blk_idx_relay, NUM_BUFFERS_DS)
                        tlx.barrier_wait(ds_peer_fulls[ds_buf_id_relay], ds_phase_relay)
                        tlx.fence("async_shared")
                        tlx.barrier_arrive(ds_fulls[ds_buf_id_relay], 1, remote_cta_rank=0)
                    phase_base += num_steps
                    task_iter += 1
                    task_id = tl.where(task_iter < num_outer_tasks, task_iter, -1)


    if Failure is not None:
        _cold_head_stripe_io(DQ64, DQ_OUT, Failure, H, Z, N_CTX, DQ64_STRIDES, False)

        # TODO: empty task to absorb warps — needs num_warps bump in configs
        # EMPTY_WARPS: tl.constexpr = 1 if USE_2CTA else 2
        # with tlx.async_task(num_warps=EMPTY_WARPS, registers=24):
        #     pass


def _certified_forward(q, k, v, sm_scale):
    layout_bshd = _forward_layout_bshd(q, k, v)
    b, h, n, d = q.shape
    safe = torch.zeros((b*h*(n//256), 2), device=q.device, dtype=torch.int32)
    o = torch.empty_like(q)
    m = torch.empty((b,h,n), device=q.device, dtype=torch.float32)
    desc_q = _forward_descriptor(q, [128,d], layout_bshd)
    desc_k = _forward_descriptor(k, [128,d], layout_bshd)
    desc_v = _forward_descriptor(v, [128,d], layout_bshd)
    desc_o = _forward_descriptor(o, [128,d], layout_bshd)
    capacity = max(1,50*1024*1024 // (4*n*d))
    group = (1 << (h-1).bit_length()) if capacity >= h else (1 << (capacity.bit_length()-1))
    group = max(4,min(64,group))
    if group == 4 and h > 4:
        group = 8
    grid = _compact_clc_grid(h,group,n//256,b) if group > 4 else (n//256*b*h,)
    allocator_context = copy_context()
    allocator_context.run(triton.set_allocator, lambda size,align,stream: torch.empty(size,dtype=torch.int8,device=q.device))
    common = dict(LAYOUT_BSHD=layout_bshd,N_CTX=n,N_CTX_STATIC=n,HEAD_DIM=d,BLOCK_M=256,BLOCK_N=128,STAGE=3,
                  NUM_BUFFERS_Q=1,NUM_BUFFERS_KV=3,NUM_BUFFERS_QK=1,NUM_MMA_GROUPS=2,
                  NUM_MMA_SLICES=2,GROUP_SIZE_N=group,USE_WHERE=False,USE_WARP_BARRIER=True,
                  NUM_CTAS=1,PIPELINED=False,DENSE_REGS=168,NUM_PID_M_STATIC=n//256,GRID_X_STATIC=h,
                  COMPACT_CLC=group>4,GLOBAL_LPT=False,POLICY=0,
                  num_warps=4,num_stages=1,num_ctas=1,multicast=False)
    allocator_context.run(_attn_fwd_ws_kernel[grid],sm_scale,m,b,h,desc_q,desc_k,desc_v,desc_o,
                             RESCALE_OPT=False,FAST_FIXED=True,Failure=safe,**common)
    fallback = dict(common, GROUP_SIZE_N=1, COMPACT_CLC=False, GLOBAL_LPT=False, N_CTX_STATIC=0)
    allocator_context.run(_attn_fwd_ws_kernel[(b*h*(n//256),)],sm_scale,m,b,h,desc_q,desc_k,desc_v,desc_o,
                                       RESCALE_OPT=True,FAST_FIXED=False,SPARSE_FALLBACK=True,Recovery=safe,**fallback)
    return o,m,safe


def _select_forward_plan(q, k, v, causal):
    head_dim = q.shape[-1]
    n_ctx = q.shape[2]
    pipelined = (head_dim == 64 and q.shape == k.shape and q.shape == v.shape and n_ctx >= 32768
                 and n_ctx % 256 == 0) or (head_dim == 128 and not causal and q.shape == k.shape and q.shape == v.shape
                                           and n_ctx >= 1024 and n_ctx % 256 == 0)
    policy = int(Policy.DENSE) if not causal else min(
        int(Policy.CAUSAL_512K),
        max(int(Policy.CAUSAL_64K),
            n_ctx.bit_length() - CAUSAL_POLICY_BIT_OFFSET),
    )
    return ForwardPlan(
        stage=3 if causal else 1,
        pipelined=pipelined,
        policy=policy,
        num_ctas=(2 if head_dim == 128 and q.dtype == torch.bfloat16 and not causal and q.shape == k.shape
                  and q.shape == v.shape and n_ctx >= 1024 else 1),
    )


def _compact_clc_grid(heads, group, num_pid_m, batches):
    grid_x = heads if heads <= group else min(group, heads & -heads)
    return grid_x, num_pid_m * (heads // grid_x) * batches


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        layout_bshd = _forward_layout_bshd(q, k, v)
        paper_shape = (
            type(q) is torch.Tensor and type(k) is torch.Tensor and type(v) is torch.Tensor
            and type(causal) is bool and type(sm_scale) in (float, int) and sm_scale in (128 ** -0.5, 1.0 / 128 ** 0.5)
            and q.ndim == k.ndim == v.ndim == 4 and q.shape == k.shape == v.shape
            and q.shape[2] in (1024, 2048, 4096, 8192, 16384, 32768)
            and (q.shape[0], q.shape[1], q.shape[3]) == (32768 // q.shape[2], 16, 128)
            and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and q.is_cuda and q.device == k.device == v.device and layout_bshd
        )
        certified_tb_shape = (
            type(q) is torch.Tensor and type(k) is torch.Tensor and type(v) is torch.Tensor
            and type(causal) is bool and type(sm_scale) in (float, int) and sm_scale in (HEAD_DIM_K ** -0.5, 1.0 / HEAD_DIM_K ** 0.5)
            and q.ndim == k.ndim == v.ndim == 4 and q.shape == k.shape == v.shape
            and q.shape[2] in (2048, 4096, 8192, 16384, 32768)
            and (q.shape[0], q.shape[1]) == (4, 48) and HEAD_DIM_K in (64, 128)
            and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and q.is_cuda and q.device == k.device == v.device
            and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        )
        if causal and ((paper_shape and q.shape[2] != 1024) or certified_tb_shape):
            o, M, _ = _certified_forward(q, k, v, sm_scale)
            ctx.save_for_backward(q, k, v, o, M)
            ctx.sm_scale = sm_scale
            ctx.HEAD_DIM = HEAD_DIM_K
            ctx.causal = causal
            ctx.saved_inverse_normalizer = False
            return o
        plan = _select_forward_plan(q, k, v, causal)

        o = torch.empty_like(q)
        extra_kern_args = {}
        if (not causal and plan.pipelined and HEAD_DIM_K == 128
                and q.dtype == k.dtype == v.dtype == torch.bfloat16
                and q.is_cuda and q.device == k.device == v.device
                and q.shape[2] in (1024, 2048, 4096, 8192, 16384, 32768)):
            extra_kern_args["N_CTX_STATIC"] = q.shape[2]

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        use_2cta = plan.num_ctas == 2
        dummy_block = [128, HEAD_DIM_K] if plan.pipelined else [1, 1]
        desc_q = _forward_descriptor(q, dummy_block, layout_bshd)
        desc_v = _forward_descriptor(v, [128, 64] if use_2cta else dummy_block, layout_bshd)
        desc_k = _forward_descriptor(k, [64, 128] if use_2cta else dummy_block, layout_bshd)
        desc_o = _forward_descriptor(o, dummy_block, layout_bshd)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        pipelined_num_mma_slices = (
            2
            if HEAD_DIM_K == 128
            or (q.dtype != torch.float16 and plan.policy in (1, 2))
            else 4
        )
        pipelined_rescale_opt = False
        pipelined_fast_fixed = True
        if plan.pipelined:
            work_ctas = (triton.cdiv(q.shape[2], FWD_BLOCK_M.value * plan.num_ctas) * q.shape[0] * q.shape[1] *
                         plan.num_ctas)
            if use_2cta:
                full_ctas = min(torch.cuda.get_device_properties(q.device).multi_processor_count, work_ctas)
                full_clusters = full_ctas // plan.num_ctas
                work_clusters = work_ctas // plan.num_ctas
                persistent_waves = triton.cdiv(work_clusters, full_clusters)
                packed_clusters = triton.cdiv(work_clusters, persistent_waves)
                grid = (packed_clusters * plan.num_ctas, )
            else:
                grid = (work_ctas, )
            _attn_fwd_ws.fn[grid](
                sm_scale,
                M,  #
                q.shape[0],
                q.shape[1],  #
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                LAYOUT_BSHD=layout_bshd,
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                BLOCK_M=256,
                BLOCK_N=128,
                STAGE=plan.stage,  #
                NUM_BUFFERS_Q=1,
                NUM_BUFFERS_KV=3 if HEAD_DIM_K == 128 else 6,
                NUM_BUFFERS_QK=1,
                NUM_MMA_GROUPS=2,
                NUM_MMA_SLICES=pipelined_num_mma_slices,
                GROUP_SIZE_N=4 if plan.policy == 1 else 1,
                RESCALE_OPT=pipelined_rescale_opt,
                USE_WHERE=False,
                USE_WARP_BARRIER=True,
                NUM_CTAS=plan.num_ctas,
                PIPELINED=True,
                POLICY=plan.policy,
                DENSE_REGS=176 if use_2cta else 168,
                FAST_FIXED=pipelined_fast_fixed,
                NUM_PID_M_STATIC=triton.cdiv(q.shape[2], 256 * plan.num_ctas),
                GRID_X_STATIC=q.shape[1],
                COMPACT_CLC=False,
                num_stages=1,
                num_warps=4,
                **({"ctas_per_cga": (2, 1, 1)} if use_2cta else {}),
                **extra_kern_args,
            )
        else:

            def grid(META):
                num_ctas = META.get("NUM_CTAS") or 1
                num_pid_m = triton.cdiv(q.shape[2], META["BLOCK_M"] * num_ctas)
                group = META["GROUP_SIZE_N"]
                if (
                    causal
                    and group > 4
                    and META.get("COMPACT_CLC", False)
                    and num_ctas == 1
                ):
                    return _compact_clc_grid(q.shape[1], group, num_pid_m, q.shape[0])
                n_clusters = num_pid_m * q.shape[0] * q.shape[1]
                return (n_clusters * num_ctas, )

            _attn_fwd_ws[grid](
                sm_scale,
                M,
                q.shape[0],
                q.shape[1],
                desc_q,
                desc_k,
                desc_v,
                desc_o,
                LAYOUT_BSHD=layout_bshd,
                N_CTX=q.shape[2],
                HEAD_DIM=HEAD_DIM_K,
                STAGE=plan.stage,
                PIPELINED=False,
                POLICY=plan.policy,
                DENSE_REGS=168,
                NUM_PID_M_STATIC=triton.cdiv(q.shape[2], 256),
                GRID_X_STATIC=q.shape[1],
                **extra_kern_args,
            )
        ctx.grid = grid

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        # Keep this predicate aligned with the kernel-side USE_FAST_FIXED gate.
        ctx.saved_inverse_normalizer = (
            plan.pipelined
            and use_2cta
            and q.dtype == torch.bfloat16
            and pipelined_num_mma_slices == 2
            and not pipelined_rescale_opt
            and pipelined_fast_fixed
        )
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert all(tensor.is_contiguous() or tensor.transpose(1, 2).is_contiguous()
                   for tensor in (q, k, v, o, do))
        assert ctx.HEAD_DIM in (64, 128), "backward requires head dimension 64 or 128"
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        paper_backward = (
            ctx.HEAD_DIM == 128 and N_HEAD == 16 and q.is_cuda
            and N_CTX in (1024, 2048, 4096, 8192, 16384, 32768)
            and BATCH * N_CTX == 32768 and ctx.sm_scale in (128 ** -0.5, 1.0 / 128 ** 0.5)
            and all(type(t) is torch.Tensor and t.dtype == torch.bfloat16 and t.shape == q.shape
                    and t.stride() == (N_CTX * N_HEAD * 128, 128, N_HEAD * 128, 1)
                    for t in (q, k, v, o, do))
        )
        prefix_correction = (
            paper_backward or (
                ctx.HEAD_DIM == 128 and BATCH == 4 and N_HEAD == 48
                and N_CTX in (2048, 4096, 8192, 16384, 32768) and ctx.sm_scale in (128 ** -0.5, 1.0 / 128 ** 0.5)
                and all(type(t) is torch.Tensor and t.is_cuda
                        and t.dtype == torch.bfloat16 and t.shape == q.shape
                        and t.device == q.device and t.is_contiguous()
                        for t in (q, k, v, o, do))
            )
        ) and ctx.causal and not ctx.saved_inverse_normalizer
        direct_dq_output = (
            ctx.HEAD_DIM == 128
            and N_CTX >= BWD_2CTA_MIN_N_CTX
            and N_CTX % 256 == 0
        )
        bf16_dq_output = (
            direct_dq_output
            and q.dtype == torch.bfloat16
            and not ctx.causal
        )
        packed_dq_output = direct_dq_output and ctx.causal and q.dtype == torch.bfloat16 and N_CTX <= 16384 and BATCH * N_HEAD * N_CTX >= 131072
        range_failure = torch.empty((1,), device=q.device, dtype=torch.int32) if packed_dq_output else None
        range_budget = torch.empty((BATCH, N_HEAD, 2, N_CTX), device=q.device, dtype=torch.int32) if packed_dq_output else None
        if packed_dq_output:
            dq = torch.empty_like(q)
        elif direct_dq_output:
            if bf16_dq_output:
                dq = torch.empty_like(q, dtype=torch.bfloat16)
            else:
                dq = torch.empty_like(q, dtype=torch.float32)
        else:
            dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _HALF_HD = ctx.HEAD_DIM // 2
        if packed_dq_output:
            if q.is_contiguous():
                dq_accum = torch.empty((BATCH, N_HEAD, N_CTX, ctx.HEAD_DIM // 2), device=q.device, dtype=torch.int32)
            else:
                dq_accum = torch.empty((BATCH, N_CTX, N_HEAD, ctx.HEAD_DIM // 2), device=q.device,
                                       dtype=torch.int32).transpose(1, 2)
        elif direct_dq_output:
            dq_accum = dq
        else:
            dq_accum = torch.empty([BATCH, N_HEAD, N_CTX, ctx.HEAD_DIM], device=q.device, dtype=torch.float32)
        PRE_BLOCK = 128
        BLK_SLICE_FACTOR = 2
        scale_qk_in_kernel = direct_dq_output or ctx.HEAD_DIM == 64
        arg_k = k if scale_qk_in_kernel else k * (ctx.sm_scale * _RCP_LN2.value)
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        dense_policy = BWD_DIRECT_DQ_CONFIG if direct_dq_output else None
        persistent_bwd = (
            direct_dq_output
            and not ctx.causal
            and dense_policy is not None
            and dense_policy[0]
        ) or (packed_dq_output and N_CTX % 512 == 0)
        scale_do_by_inv_l = ctx.saved_inverse_normalizer
        assert not scale_do_by_inv_l or bf16_dq_output
        do_scaled = torch.empty_like(do) if scale_do_by_inv_l else do
        preprocess_zeroes_dq = (scale_do_by_inv_l and not persistent_bwd) or packed_dq_output
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            M, do_scaled, delta, dq_accum,  #
            N_CTX,  #
            SCALE_DO_BY_INV_L=scale_do_by_inv_l,
            ZERO_DQ=preprocess_zeroes_dq,
            PACKED_DQ=packed_dq_output,
            RangeFailure=range_failure,
            RangeBudget=range_budget,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM,  #
            N_HEAD=N_HEAD,
            O_STRIDES=None if o.is_contiguous() else o.stride()[:3],
            DO_STRIDES=None if do.is_contiguous() else do.stride()[:3],
            SCALED_DO_STRIDES=None if do_scaled.is_contiguous() else do_scaled.stride()[:3],
            DQ_STRIDES=None if dq_accum.is_contiguous() else dq_accum.stride()[:3],
        )

        if prefix_correction or (ctx.HEAD_DIM == 64 and ctx.causal and q.dtype == torch.bfloat16):
            prefix_block = 64 if N_CTX == 1024 else 16
            _bwd_prefix_delta[(128 // prefix_block, BATCH * N_HEAD)](
                q, k, v, do, M, delta, ctx.sm_scale,
                N_CTX=N_CTX, N_HEAD=N_HEAD,
                Q_STRIDES=q.stride(), K_STRIDES=k.stride(),
                V_STRIDES=v.stride(), DO_STRIDES=do.stride(),
                BLOCK_M=prefix_block, BLOCK_N=128, HEAD_DIM=ctx.HEAD_DIM, num_warps=4,
            )

        dummy_block = [1, 1, 1, 1]
        HEAD_DIM = ctx.HEAD_DIM
        desc_shape = [BATCH, N_HEAD, N_CTX, HEAD_DIM]
        desc_k = TensorDescriptor(
            arg_k,
            shape=desc_shape,
            strides=arg_k.stride(),
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=desc_shape,
            strides=v.stride(),
            block_shape=dummy_block,
        )
        desc_q = TensorDescriptor(
            q,
            shape=desc_shape,
            strides=q.stride(),
            block_shape=dummy_block,
        )
        desc_do = TensorDescriptor(
            do_scaled,
            shape=desc_shape,
            strides=do_scaled.stride(),
            block_shape=dummy_block,
        )
        if direct_dq_output:
            dq_half_hd = _HALF_HD // 2 if packed_dq_output else _HALF_HD
            packed_shape = [BATCH, N_HEAD, 2, N_CTX, dq_half_hd]
            packed_strides = [
                dq_accum.stride(0),
                dq_accum.stride(1),
                dq_half_hd,
                dq_accum.stride(2),
                1,
            ]
        else:
            packed_shape = [BATCH, N_HEAD, 2 * N_CTX, _HALF_HD]
            packed_strides = [N_HEAD * N_CTX * HEAD_DIM, N_CTX * HEAD_DIM, _HALF_HD, 1]
        desc_dq = TensorDescriptor(
            dq_accum,
            shape=packed_shape,
            strides=packed_strides,
            block_shape=[1, 1, 1, 1, 1] if direct_dq_output else dummy_block,
        )
        desc_dk = TensorDescriptor(
            dk,
            shape=desc_shape,
            strides=dk.stride(),
            block_shape=dummy_block,
        )
        desc_dv = TensorDescriptor(
            dv,
            shape=desc_shape,
            strides=dv.stride(),
            block_shape=dummy_block,
        )
        desc_m = TensorDescriptor(
            M,
            shape=[BATCH * N_HEAD * N_CTX],
            strides=[1],
            block_shape=[1],
        )
        desc_delta = TensorDescriptor(
            delta,
            shape=[BATCH * N_HEAD * N_CTX],
            strides=[1],
            block_shape=[1],
        )

        desc_kt = TensorDescriptor(arg_k, shape=desc_shape, strides=arg_k.stride(), block_shape=dummy_block)
        desc_qt = TensorDescriptor(q, shape=desc_shape, strides=q.stride(), block_shape=dummy_block)
        desc_dot = TensorDescriptor(do_scaled, shape=desc_shape, strides=do_scaled.stride(), block_shape=dummy_block)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # NUM_SMS = torch.cuda.get_device_properties(q.device).multi_processor_count

        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        persistent_cta_cap = max(2, int(sm_count * PERSISTENT_CTA_FRACTION) // 2 * 2)

        def grid_bwd(meta):
            n_tiles = triton.cdiv(N_CTX, meta["BLOCK_N1"])
            num_ctas = meta.get("NUM_CTAS", 1)
            n_tiles = triton.cdiv(n_tiles, num_ctas) * num_ctas
            if persistent_bwd and packed_dq_output:
                pair_units = BATCH * N_HEAD * (n_tiles // (2 * num_ctas))
                return (num_ctas * min(pair_units, max(1, sm_count // num_ctas)), )
            if persistent_bwd and num_ctas == 2:
                return (min(n_tiles * N_HEAD * BATCH, persistent_cta_cap), )
            if num_ctas == 2:
                return (n_tiles, N_HEAD, BATCH)
            return (n_tiles, N_HEAD, BATCH)

        stage = 3 if ctx.causal else 1
        bwd_args = (
            desc_q, desc_k, desc_v, ctx.sm_scale, desc_do, desc_dq, desc_dk, desc_dv,
            desc_m, desc_delta, M, delta, N_HEAD, BATCH, N_CTX, desc_kt, desc_qt, desc_dot,
        )
        bwd_kwargs = dict(
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            SCALE_QK_IN_KERNEL=scale_qk_in_kernel,
            PERSISTENT_BWD=persistent_bwd,
            PRENORMALIZED_DO=scale_do_by_inv_l,
            PREPROCESS_ZEROES_DQ=preprocess_zeroes_dq,
            PACKED_DQ=packed_dq_output,
            RangeFailure=range_failure,
            PACKED_N_CTX=N_CTX if packed_dq_output else 0,
            RangeBudget=range_budget,
            BSHD_CONFIG=paper_backward,
        )
        _attn_bwd_ws[grid_bwd](*bwd_args, **bwd_kwargs)

        if packed_dq_output:
            unpack_block = 1024 if prefix_correction and N_CTX in (8192, 16384) else 512
            _attn_bwd_unpack_fixed[(triton.cdiv(dq_accum.numel(), unpack_block),)](
                dq_accum, dq, dq_accum.numel(), N_CTX, dq_accum.stride(2), range_failure, BLOCK=unpack_block,
                RangeBudget=range_budget, N_HEAD=N_HEAD,
            )
            fallback_dq = torch.empty_like(q, dtype=torch.float64)
            cold_kwargs = dict(
                bwd_kwargs, PERSISTENT_BWD=True, PREPROCESS_ZEROES_DQ=True,
                PACKED_DQ=False, RangeFailure=None, PACKED_N_CTX=0, RangeBudget=None,
                Failure=range_failure, DQ64=fallback_dq, DQ64_STRIDES=fallback_dq.stride(), DQ_OUT=dq,
            )
            cold_grid = (2 * min(BATCH * N_HEAD, max(1, sm_count // 2)), 1, 1)
            _attn_bwd_ws[cold_grid](*bwd_args, **cold_kwargs)
        elif not direct_dq_output:
            _blk = _bwd_selected_meta["BLOCK_M1"] // _bwd_selected_meta["NUM_CTAS"]
            post_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
            _attn_bwd_dq_postprocess[post_grid](
                dq_accum, dq,  #
                N_CTX,  #
                BLK=_blk, HALF_HD=_HALF_HD,  #
                BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM,  #
                N_HEAD=N_HEAD,
                DQ_STRIDES=None if dq.is_contiguous() else dq.stride()[:3],
            )

        if ((prefix_correction and (N_CTX == 1024 or (BATCH == 4 and N_HEAD == 48 and N_CTX in (8192, 16384))))
                or (ctx.HEAD_DIM == 64 and ctx.causal and q.dtype == torch.bfloat16)):
            _bwd_prefix_dv_residual[(4, BATCH * N_HEAD)](
                q, k, do, M, dv, ctx.sm_scale, N_CTX=N_CTX, N_HEAD=N_HEAD,
                Q_STRIDES=q.stride(), K_STRIDES=k.stride(),
                DO_STRIDES=do.stride(), DV_STRIDES=dv.stride(),
                BLOCK_K=32, PREFIX_Q=128, HEAD_DIM=ctx.HEAD_DIM, num_warps=4,
            )

        return dq, dk, dv, None, None


def attention(q, k, v, sm_scale, causal, config=None):
    if config is None:
        return _attention.apply(q, k, v, sm_scale, causal)

    layout_bshd = _forward_layout_bshd(q, k, v)
    # Non-autotuned path with explicit config
    HEAD_DIM_K = q.shape[-1]
    stage = 3 if causal else 1
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    dummy_block = [1, 1]
    desc_q = _forward_descriptor(q, dummy_block, layout_bshd)
    desc_v = _forward_descriptor(v, dummy_block, layout_bshd)
    desc_k = _forward_descriptor(k, dummy_block, layout_bshd)
    desc_o = _forward_descriptor(o, dummy_block, layout_bshd)

    # Apply pre_hook to set block shapes
    nargs = {**config, "HEAD_DIM": HEAD_DIM_K, "desc_q": desc_q, "desc_k": desc_k, "desc_v": desc_v, "desc_o": desc_o}
    _host_descriptor_pre_hook(nargs)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    num_ctas = config.get("NUM_CTAS", 1)
    assert q.shape[2] % (config["BLOCK_M"] * num_ctas) == 0 or num_ctas == 1
    num_pid_m = triton.cdiv(q.shape[2], config["BLOCK_M"] * num_ctas)
    if (
        causal
        and config["GROUP_SIZE_N"] > 4
        and config.get("COMPACT_CLC", False)
        and num_ctas == 1
    ):
        grid = _compact_clc_grid(q.shape[1], config["GROUP_SIZE_N"], num_pid_m, q.shape[0])
    else:
        grid0 = num_pid_m * q.shape[0] * q.shape[1] * num_ctas
        grid = (grid0, 1, 1)
    launch_kwargs = {}
    if num_ctas > 1:
        launch_kwargs["ctas_per_cga"] = (num_ctas, 1, 1)
    _attn_fwd_ws.fn[grid](
        sm_scale,
        M,
        q.shape[0],
        q.shape[1],
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        LAYOUT_BSHD=layout_bshd,
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        NUM_PID_M_STATIC=num_pid_m,
        GRID_X_STATIC=q.shape[1],
        num_stages=1,
        **launch_kwargs,
        **config,
    )
    return o
