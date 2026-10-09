"""Blackwell MXFP8 grouped GEMM for ``tlx.ops.grouped_gemm_mxfp8``.

This is a forward-only, persistent SM100 kernel for E4M3 operands, E8M0
microscales, and BF16 output. The 1CTA specialization uses a graph-safe dynamic
scheduler; the 2CTA specialization follows a static cluster-stride schedule.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.runtime import _allocation
from triton.tlx.ops.kernels.grouped_gemm_mxfp8._scales import prepare_scales

_BLOCK_SIZE_M = 128
_BLOCK_SIZE_N = 256
_BLOCK_SIZE_K = 128
_VEC_SIZE = 32
_NUM_DATA_BUFFERS = 4
_NUM_SCALE_BUFFERS = 4
_NUM_TMEM_BUFFERS = 1
# Two accumulator slots that share one epilogue subtile of TMEM columns. The
# epilogue drains the shared subtile first and releases the accumulator, so the
# next tile's MMA overlaps the remaining epilogue subtiles.
_OVERLAP_ACC = True
_NUM_TILE_BUFFERS = 3
_EPILOGUE_SUBTILE = 4
_NUM_WARPS = 4
_NUM_STAGES = 1
_TILE_SENTINEL = 0x7FFFFFFF - 1

_SM100_SMEM_BYTES = 232448
_SM100_TMEM_COLUMNS = 512
_SMEM_SAFETY_MARGIN_BYTES = 1024
_MBARRIER_BYTES = 8
_FP8_BYTES = 1
_BF16_BYTES = 2
_INT32_BYTES = 4

_CONFIG_SPEC = {
    "BLOCK_SIZE_M": _BLOCK_SIZE_M,
    "BLOCK_SIZE_N": _BLOCK_SIZE_N,
    "BLOCK_SIZE_K": _BLOCK_SIZE_K,
    "NUM_DATA_BUFFERS": _NUM_DATA_BUFFERS,
    "NUM_SCALE_BUFFERS": _NUM_SCALE_BUFFERS,
    "NUM_TMEM_BUFFERS": _NUM_TMEM_BUFFERS,
    "NUM_TILE_BUFFERS": _NUM_TILE_BUFFERS,
    "EPILOGUE_SUBTILE": _EPILOGUE_SUBTILE,
    "OVERLAP_ACC": _OVERLAP_ACC,
    "NUM_CTAS": 1,
}
_CONFIG_2CTA_SPEC = dict(
    _CONFIG_SPEC,
    BLOCK_SIZE_K=128,
    NUM_DATA_BUFFERS=6,
    NUM_SCALE_BUFFERS=6,
    NUM_CTAS=2,
)


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _select_num_ctas(*, gm: int, n: int, k: int, launch_sms: int) -> int:
    if launch_sms < 2 or launch_sms % 2 != 0 or n < 3072 or k < 2048:
        return 1
    num_clusters = launch_sms // 2
    min_logical_tiles = _cdiv(gm, 2 * _BLOCK_SIZE_M) * _cdiv(n, _BLOCK_SIZE_N)
    return 2 if min_logical_tiles >= num_clusters else 1


def _estimate_operand_smem_bytes(config: dict[str, int]) -> int:
    return (config["NUM_DATA_BUFFERS"] * (config["BLOCK_SIZE_M"] + config["BLOCK_SIZE_N"] // config["NUM_CTAS"]) *
            config["BLOCK_SIZE_K"] * _FP8_BYTES)


def _estimate_scale_smem_bytes(config: dict[str, int]) -> int:
    rep_m = config["BLOCK_SIZE_M"] // 128
    rep_n = config["BLOCK_SIZE_N"] // 128
    rep_k = _cdiv(config["BLOCK_SIZE_K"] // _VEC_SIZE, 4)
    return config["NUM_SCALE_BUFFERS"] * (rep_m + rep_n) * rep_k * 2 * 256


def _estimate_epilogue_smem_bytes(config: dict[str, int]) -> int:
    return (config["BLOCK_SIZE_M"] * (config["BLOCK_SIZE_N"] // config["EPILOGUE_SUBTILE"]) * _BF16_BYTES)


def _estimate_tile_id_smem_bytes(config: dict[str, int]) -> int:
    if config["NUM_CTAS"] == 2:
        return 0
    return config["NUM_TILE_BUFFERS"] * _INT32_BYTES


def _estimate_barrier_smem_bytes(config: dict[str, int]) -> int:
    barrier_count = 2 * config["NUM_DATA_BUFFERS"] + 2 * config["NUM_TMEM_BUFFERS"]
    if config["NUM_CTAS"] == 1:
        barrier_count += 2 * config["NUM_TILE_BUFFERS"]
    return barrier_count * _MBARRIER_BYTES


def _estimate_smem_bytes(config: dict[str, int]) -> int:
    return (_estimate_operand_smem_bytes(config) + _estimate_scale_smem_bytes(config) +
            _estimate_epilogue_smem_bytes(config) + _estimate_tile_id_smem_bytes(config) +
            _estimate_barrier_smem_bytes(config))


def _accumulator_tmem_columns(config: dict[str, int]) -> int:
    if config["OVERLAP_ACC"]:
        slice_n = config["BLOCK_SIZE_N"] // config["EPILOGUE_SUBTILE"]
        return 2 * config["BLOCK_SIZE_N"] - slice_n
    return config["BLOCK_SIZE_N"] * config["NUM_TMEM_BUFFERS"]


def _estimate_tmem_columns(config: dict[str, int]) -> int:
    if config["OVERLAP_ACC"]:
        # The TMEM storage alias spans 2N columns: both overlapped accumulator
        # slots plus the explicitly staged block scales past them.
        return 2 * config["BLOCK_SIZE_N"]
    return config["BLOCK_SIZE_N"] * config["NUM_TMEM_BUFFERS"]


def _config_error(config: dict[str, int]) -> str | None:
    expected = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "NUM_TMEM_BUFFERS": 1,
        "NUM_TILE_BUFFERS": 3,
        "EPILOGUE_SUBTILE": 4,
        "OVERLAP_ACC": True,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            return f"{name} must be {value}"
    num_ctas = config.get("NUM_CTAS")
    if num_ctas not in (1, 2):
        return "NUM_CTAS must be 1 or 2"
    expected_pipeline = ({"BLOCK_SIZE_K": 128, "NUM_DATA_BUFFERS": 4, "NUM_SCALE_BUFFERS": 4}
                         if num_ctas == 1 else {"BLOCK_SIZE_K": 128, "NUM_DATA_BUFFERS": 6, "NUM_SCALE_BUFFERS": 6})
    for name, value in expected_pipeline.items():
        if config.get(name) != value:
            return f"{name} must be {value}"
    if _estimate_smem_bytes(config) + _SMEM_SAFETY_MARGIN_BYTES > _SM100_SMEM_BYTES:
        return "configuration exceeds the SM100 shared-memory budget"
    if _estimate_tmem_columns(config) > _SM100_TMEM_COLUMNS:
        return "configuration exceeds the SM100 tensor-memory budget"
    return None


for _CONFIG_NAME, _CANDIDATE_CONFIG in (
    ("1cta", _CONFIG_SPEC),
    ("2cta", _CONFIG_2CTA_SPEC),
):
    if _CONFIG_ERROR := _config_error(_CANDIDATE_CONFIG):
        raise RuntimeError(f"invalid SM100 MXFP8 grouped GEMM {_CONFIG_NAME} config: {_CONFIG_ERROR}")


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS: tl.constexpr):
    buf_idx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return buf_idx, phase


@triton.jit
def _device_trap_if(condition):
    """Trap lanes where condition is true without requiring Triton debug mode."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred failed;
            setp.ne.u32 failed, $1, 0;
            @failed trap;
            mov.u32 $0, 0;
        }
        """,
        "=r,r",
        [condition.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_validated_split(split_sizes_ptr, group_idx, running_m, total_m):
    m_size = tl.load(split_sizes_ptr + group_idx, cache_modifier=".ca").to(tl.int32)
    next_m = running_m + m_size
    _device_trap_if((m_size < 0) | (running_m % 128 != 0) | (next_m > total_m))
    return m_size


@triton.jit
def _preread_tile_idx(accum_cnt, NUM_BUFFERS: tl.constexpr, bars, tile_id_smem):
    buf, phase = _get_bufidx_phase(accum_cnt, NUM_BUFFERS)
    tlx.barrier_wait(bars[buf], phase)
    tile_idx = tl.reshape(tlx.local_load(tile_id_smem[buf]), (1, ))
    return tl.sum(tile_idx)


@triton.jit
def _producer_fetch_tile_idx_1cta(counter_ptr):
    return tl.atomic_add(counter_ptr, 1, sem="relaxed")


@triton.jit
def _producer_wait_tile_empty(
    tile_id_producer_bars,
    accum_cnt,
    NUM_TILE_BUFFERS: tl.constexpr,
):
    if accum_cnt >= NUM_TILE_BUFFERS:
        buf, phase = _get_bufidx_phase(accum_cnt, NUM_TILE_BUFFERS)
        tlx.barrier_wait(tile_id_producer_bars[buf], phase ^ 1)


@triton.jit
def _producer_signal_tile_ready(
    tile_id_smem,
    tile_id_consumer_bars,
    tile_buf,
    tile_idx,
):
    tlx.local_store(tile_id_smem[tile_buf], tl.full((1, ), tile_idx, tl.int32))
    tlx.barrier_arrive(tile_id_consumer_bars[tile_buf], 1)


@triton.jit
def _epilogue_signal_tile_done(tile_id_producer_bars, tile_buf):
    tlx.barrier_arrive(tile_id_producer_bars[tile_buf], 1)


@triton.jit
def _release_acc(tmem_empty_bar, NUM_CTAS: tl.constexpr):
    if NUM_CTAS == 2:
        tlx.barrier_arrive(tmem_empty_bar, 1, remote_cta_rank=0)
    else:
        tlx.barrier_arrive(tmem_empty_bar, 1)


@triton.jit
def _mxfp8_grouped_gemm_kernel(  # noqa: C901
    a_ptr,
    stride_am,
    stride_ak,
    b_ptr,
    stride_bn,
    stride_bk,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    scale_n_chunks,
    scale_k_chunks,
    stride_b_scale_g,
    split_sizes_ptr,
    counter_ptr,
    G: tl.constexpr,
    M,
    N: tl.constexpr,
    K,
    SENTINEL: tl.constexpr,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_DATA_BUFFERS: tl.constexpr,
    NUM_SCALE_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_TILE_BUFFERS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    OVERLAP_ACC: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    """Persistent E4M3 grouped GEMM with E8M0 microscaling."""
    VEC_SIZE: tl.constexpr = 32
    REP_M: tl.constexpr = BLOCK_SIZE_M // 128
    REP_N: tl.constexpr = BLOCK_SIZE_N // 128
    REP_K: tl.constexpr = triton.cdiv(BLOCK_SIZE_K // VEC_SIZE, 4)
    BLOCK_N_PER_CTA: tl.constexpr = BLOCK_SIZE_N // NUM_CTAS
    NUM_SMEM_BUFFERS: tl.constexpr = NUM_DATA_BUFFERS
    tl.static_assert(NUM_DATA_BUFFERS == NUM_SCALE_BUFFERS)

    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
    else:
        cluster_cta_rank = 0
        pred_cta0 = True

    buffers_a = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_K),
        tl.float8e4nv,
        NUM_SMEM_BUFFERS,
    )
    buffers_b = tlx.local_alloc(
        (BLOCK_N_PER_CTA, BLOCK_SIZE_K),
        tl.float8e4nv,
        NUM_SMEM_BUFFERS,
    )
    buffers_a_scale = tlx.local_alloc(
        (1, REP_M, REP_K, 2, 256),
        tl.uint8,
        NUM_SMEM_BUFFERS,
    )
    buffers_b_scale = tlx.local_alloc(
        (1, REP_N, REP_K, 2, 256),
        tl.uint8,
        NUM_SMEM_BUFFERS,
    )
    if OVERLAP_ACC:
        # Overlapping-accumulator layout. Slot 0 owns columns [0, N)
        # and slot 1 owns [N - N / SUBTILE, 2N - N / SUBTILE); only the last
        # subtile of slot 0 aliases the first subtile of slot 1, and the
        # epilogue drains that subtile first. The block scales are placed
        # explicitly in the columns past 2N - N / SUBTILE so the whole layout
        # fits in the 512 TMEM columns.
        tl.static_assert(NUM_TMEM_BUFFERS == 1)
        tl.static_assert(BLOCK_SIZE_N == 256 and EPILOGUE_SUBTILE == 4)
        ACC_SHIFT: tl.constexpr = BLOCK_SIZE_N - BLOCK_SIZE_N // EPILOGUE_SUBTILE
        # Scale TMEM shapes match the compiler's SMEM->TMEM scale lowering:
        # rows are the per-CTA MMA M / N, columns hold the remaining bytes.
        A_SCALE_TMEM_COLS: tl.constexpr = REP_M * REP_K * 2 * 256 // BLOCK_SIZE_M
        B_SCALE_TMEM_COLS: tl.constexpr = REP_N * REP_K * 2 * 256 // BLOCK_N_PER_CTA
        acc_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
        acc_full = tlx.local_alloc(
            (BLOCK_SIZE_M, 2 * BLOCK_SIZE_N),
            tl.float32,
            1,
            tlx.storage_kind.tmem,
            reuse=acc_alias,
        )
        # Placeholders that push the scales past column 2N - N / SUBTILE;
        # distinct children are aligned to their own column width.
        acc_pad_n = tlx.local_alloc(
            (BLOCK_SIZE_M, BLOCK_SIZE_N),
            tl.float32,
            1,
            tlx.storage_kind.tmem,
            reuse=acc_alias,
        )
        acc_pad_half = tlx.local_alloc(
            (BLOCK_SIZE_M, BLOCK_SIZE_N // 2),
            tl.float32,
            1,
            tlx.storage_kind.tmem,
            reuse=acc_alias,
        )
        acc_pad_quarter = tlx.local_alloc(
            (BLOCK_SIZE_M, BLOCK_SIZE_N // 4),
            tl.float32,
            1,
            tlx.storage_kind.tmem,
            reuse=acc_alias,
        )
        a_scale_tmem = tlx.local_alloc(
            (BLOCK_SIZE_M, A_SCALE_TMEM_COLS),
            tl.uint8,
            1,
            tlx.storage_kind.tmem,
            reuse=acc_alias,
        )
        b_scale_tmem = tlx.local_alloc(
            (BLOCK_N_PER_CTA, B_SCALE_TMEM_COLS),
            tl.uint8,
            1,
            tlx.storage_kind.tmem,
            reuse=acc_alias,
        )
        acc_alias.set_buffer_overlap(
            tlx.reuse_group(
                acc_full,
                tlx.reuse_group(
                    acc_pad_n,
                    acc_pad_half,
                    acc_pad_quarter,
                    a_scale_tmem,
                    b_scale_tmem,
                    group_type=tlx.reuse_group_type.distinct,
                ),
                group_type=tlx.reuse_group_type.shared,
            ))
        ACC_RELEASE_ARRIVES: tl.constexpr = 1
    else:
        ACC_SHIFT: tl.constexpr = 0
        buffers_c = tlx.local_alloc(
            (BLOCK_SIZE_M, BLOCK_SIZE_N),
            tl.float32,
            NUM_TMEM_BUFFERS,
            tlx.storage_kind.tmem,
        )
        ACC_RELEASE_ARRIVES: tl.constexpr = EPILOGUE_SUBTILE
    SLICE_N: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    if NUM_CTAS == 1:
        tile_id_smem = tlx.local_alloc((1, ), tl.int32, NUM_TILE_BUFFERS)

    smem_empty_bars = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    smem_full_bars = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(NUM_TMEM_BUFFERS, arrive_count=1)
    # With OVERLAP_ACC one tmem_full/tmem_empty pair coordinates both slots.
    # The MMA and epilogue tasks walk the same tile sequence, so the barrier
    # phase (tile parity) also selects the slot:
    #   MMA:      wait tmem_empty(phase ^ 1) -> MMAs into slot[phase] ->
    #             tcgen05_commit(tmem_full)
    #   epilogue: wait tmem_full(phase) -> load the shared subtile ->
    #             arrive tmem_empty once -> drain the other subtiles while the
    #             next tile's MMA writes the other slot
    # The MMA task runs at most one tile ahead: reusing tile t's slot for
    # tile t + 2 needs the release of tile t + 1, which the in-order epilogue
    # issues only after draining all of tile t, so the parity cannot alias.
    # In 2CTA mode the leader issues the MMA, tcgen05_commit signals both
    # CTAs, and both epilogues arrive on the leader's tmem_empty.
    tmem_empty_bars = tlx.alloc_barriers(
        NUM_TMEM_BUFFERS,
        arrive_count=ACC_RELEASE_ARRIVES * NUM_CTAS,
    )
    if NUM_CTAS == 1:
        tile_id_consumer_bars = tlx.alloc_barriers(
            NUM_TILE_BUFFERS,
            arrive_count=1,
        )
        tile_id_producer_bars = tlx.alloc_barriers(
            NUM_TILE_BUFFERS,
            arrive_count=1,
        )
    else:
        tl.static_assert(NUM_SM % NUM_CTAS == 0)
        tlx.fence_mbarrier_init_cluster()

    with tlx.async_tasks(
            exclusive=True,
            no_ending_cluster_sync=True,
            mbarrier_try_wait_suspend_ns=50000,
    ):
        with tlx.async_task("default"):
            cm_start = 0
            tile_start = 0
            accum_cnt_tmem = 0
            if NUM_CTAS == 2:
                tile_idx = tl.program_id(0) // NUM_CTAS
                tile_stride: tl.constexpr = NUM_SM // NUM_CTAS
            else:
                accum_cnt_tile = 0
                tile_idx = _preread_tile_idx(
                    accum_cnt_tile,
                    NUM_TILE_BUFFERS,
                    tile_id_consumer_bars,
                    tile_id_smem,
                )

            for g in range(G):
                m_size = _load_validated_split(split_sizes_ptr, g, cm_start, M)
                num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M * NUM_CTAS)
                num_n_tiles: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
                num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
                num_tiles = num_m_tiles * num_n_tiles
                tile_end = tile_start + num_tiles

                if (tile_idx >= tile_start) and (tile_idx < tile_end):
                    c_desc = tl.make_tensor_descriptor(
                        c_ptr + cm_start.to(tl.int64) * N,
                        shape=[m_size, N],
                        strides=[N, 1],
                        block_shape=[
                            BLOCK_SIZE_M,
                            BLOCK_SIZE_N // EPILOGUE_SUBTILE,
                        ],
                    )
                    while (tile_idx >= tile_start) and (tile_idx < tile_end):
                        cur_tile_idx = tile_idx - tile_start
                        if m_size < N:
                            tile_m_idx = cur_tile_idx % num_m_tiles
                            tile_n_idx = cur_tile_idx // num_m_tiles
                        else:
                            tile_n_idx = cur_tile_idx % num_n_tiles
                            tile_m_idx = cur_tile_idx // num_n_tiles

                        tmem_buf, tmem_phase = _get_bufidx_phase(
                            accum_cnt_tmem,
                            NUM_TMEM_BUFFERS,
                        )
                        tlx.barrier_wait(tmem_full_bars[tmem_buf], tmem_phase)
                        offs_cm = (tile_m_idx * BLOCK_SIZE_M * NUM_CTAS + cluster_cta_rank * BLOCK_SIZE_M)
                        offs_cn = tile_n_idx * BLOCK_SIZE_N

                        for i in tl.static_range(EPILOGUE_SUBTILE):
                            if OVERLAP_ACC:
                                # Even tiles use slot 0 drained in reverse and
                                # odd tiles use slot 1 drained forward, so the
                                # first subtile is always the shared columns.
                                if tmem_phase == 0:
                                    result = tlx.local_load(
                                        tlx.subslice(
                                            acc_full[0],
                                            (EPILOGUE_SUBTILE - 1 - i) * SLICE_N,
                                            SLICE_N,
                                        ))
                                    out_n = offs_cn + (EPILOGUE_SUBTILE - 1 - i) * SLICE_N
                                else:
                                    result = tlx.local_load(
                                        tlx.subslice(
                                            acc_full[0],
                                            ACC_SHIFT + i * SLICE_N,
                                            SLICE_N,
                                        ))
                                    out_n = offs_cn + i * SLICE_N
                                if i == 0:
                                    # The shared columns are in registers, so
                                    # the next tile's MMA may overwrite them.
                                    # The non-per-thread arrive synchronizes
                                    # the warp group after the TMEM load.
                                    _release_acc(tmem_empty_bars[0], NUM_CTAS)
                            else:
                                result = tlx.local_load(
                                    tlx.local_slice(
                                        buffers_c[tmem_buf],
                                        [0, i * SLICE_N],
                                        [BLOCK_SIZE_M, SLICE_N],
                                    ))
                                out_n = offs_cn + i * SLICE_N
                            c_desc.store(
                                [offs_cm, out_n],
                                result.to(tl.bfloat16),
                            )
                            if not OVERLAP_ACC:
                                _release_acc(tmem_empty_bars[tmem_buf], NUM_CTAS)

                        accum_cnt_tmem += 1
                        if NUM_CTAS == 2:
                            tile_idx += tile_stride
                        else:
                            tile_buf, _ = _get_bufidx_phase(
                                accum_cnt_tile,
                                NUM_TILE_BUFFERS,
                            )
                            _epilogue_signal_tile_done(
                                tile_id_producer_bars,
                                tile_buf,
                            )
                            accum_cnt_tile += 1
                            tile_idx = _preread_tile_idx(
                                accum_cnt_tile,
                                NUM_TILE_BUFFERS,
                                tile_id_consumer_bars,
                                tile_id_smem,
                            )

                cm_start += m_size
                tile_start += num_tiles

            _device_trap_if(cm_start != M)

        with tlx.async_task(num_warps=1, num_regs=48):
            tile_start = 0
            running_m = 0
            accum_cnt_smem = 0
            accum_cnt_tmem = 0
            if NUM_CTAS == 2:
                tile_idx = tl.program_id(0) // NUM_CTAS
                tile_stride: tl.constexpr = NUM_SM // NUM_CTAS
            else:
                accum_cnt_tile = 0
                tile_idx = _preread_tile_idx(
                    accum_cnt_tile,
                    NUM_TILE_BUFFERS,
                    tile_id_consumer_bars,
                    tile_id_smem,
                )

            for g in range(G):
                m_size = _load_validated_split(split_sizes_ptr, g, running_m, M)
                num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M * NUM_CTAS)
                num_n_tiles: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
                num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
                num_tiles = num_m_tiles * num_n_tiles
                tile_end = tile_start + num_tiles

                if (tile_idx >= tile_start) and (tile_idx < tile_end):
                    while (tile_idx >= tile_start) and (tile_idx < tile_end):
                        if (NUM_CTAS == 1) or (cluster_cta_rank == 0):
                            tmem_buf, tmem_phase = _get_bufidx_phase(
                                accum_cnt_tmem,
                                NUM_TMEM_BUFFERS,
                            )
                            tlx.barrier_wait(
                                tmem_empty_bars[tmem_buf],
                                tmem_phase ^ 1,
                            )

                            for kk in range(0, num_k_tiles):
                                smem_buf, smem_phase = _get_bufidx_phase(
                                    accum_cnt_smem,
                                    NUM_SMEM_BUFFERS,
                                )
                                tlx.barrier_wait(
                                    smem_full_bars[smem_buf],
                                    smem_phase,
                                )
                                if OVERLAP_ACC:
                                    # Stage this k-block's scales into one TMEM
                                    # slot; tcgen05.cp and tcgen05.mma issue in
                                    # order from this thread.
                                    tlx.tmem_copy(
                                        buffers_a_scale[smem_buf],
                                        a_scale_tmem[0],
                                    )
                                    tlx.tmem_copy(
                                        buffers_b_scale[smem_buf],
                                        b_scale_tmem[0],
                                    )
                                    if tmem_phase == 0:
                                        tlx.async_dot_scaled(
                                            buffers_a[smem_buf],
                                            tlx.local_trans(buffers_b[smem_buf]),
                                            tlx.subslice(acc_full[0], 0, BLOCK_SIZE_N),
                                            a_scale_tmem[0],
                                            "e4m3",
                                            b_scale_tmem[0],
                                            "e4m3",
                                            use_acc=kk > 0,
                                            mBarriers=[smem_empty_bars[smem_buf]],
                                            two_ctas=NUM_CTAS == 2,
                                        )
                                    else:
                                        tlx.async_dot_scaled(
                                            buffers_a[smem_buf],
                                            tlx.local_trans(buffers_b[smem_buf]),
                                            tlx.subslice(
                                                acc_full[0],
                                                ACC_SHIFT,
                                                BLOCK_SIZE_N,
                                            ),
                                            a_scale_tmem[0],
                                            "e4m3",
                                            b_scale_tmem[0],
                                            "e4m3",
                                            use_acc=kk > 0,
                                            mBarriers=[smem_empty_bars[smem_buf]],
                                            two_ctas=NUM_CTAS == 2,
                                        )
                                else:
                                    tlx.async_dot_scaled(
                                        buffers_a[smem_buf],
                                        tlx.local_trans(buffers_b[smem_buf]),
                                        buffers_c[tmem_buf],
                                        buffers_a_scale[smem_buf],
                                        "e4m3",
                                        buffers_b_scale[smem_buf],
                                        "e4m3",
                                        use_acc=kk > 0,
                                        mBarriers=[smem_empty_bars[smem_buf]],
                                        two_ctas=NUM_CTAS == 2,
                                    )
                                accum_cnt_smem += 1

                            tlx.tcgen05_commit(
                                tmem_full_bars[tmem_buf],
                                two_ctas=NUM_CTAS == 2,
                            )
                        else:
                            accum_cnt_smem += num_k_tiles
                        accum_cnt_tmem += 1
                        if NUM_CTAS == 2:
                            tile_idx += tile_stride
                        else:
                            accum_cnt_tile += 1
                            tile_idx = _preread_tile_idx(
                                accum_cnt_tile,
                                NUM_TILE_BUFFERS,
                                tile_id_consumer_bars,
                                tile_id_smem,
                            )

                running_m += m_size
                tile_start += num_tiles

            _device_trap_if(running_m != M)

        with tlx.async_task(num_warps=1, num_regs=48):
            start_am = 0
            start_bn = 0
            tile_start = 0
            accum_cnt_smem = 0
            data_bytes: tl.constexpr = (BLOCK_SIZE_M + BLOCK_N_PER_CTA) * BLOCK_SIZE_K
            a_scale_bytes: tl.constexpr = REP_M * REP_K * 2 * 256
            b_scale_bytes: tl.constexpr = REP_N * REP_K * 2 * 256
            smem_bytes: tl.constexpr = data_bytes + a_scale_bytes + b_scale_bytes
            # Multicast completion counts bytes once per destination CTA.
            cooperative_smem_bytes: tl.constexpr = smem_bytes * NUM_CTAS
            if NUM_CTAS == 2:
                tile_idx = tl.program_id(0) // NUM_CTAS
                tile_stride: tl.constexpr = NUM_SM // NUM_CTAS
            else:
                accum_cnt_out = 0
                tile_idx = _producer_fetch_tile_idx_1cta(counter_ptr)

            # Full-domain load descriptors are built once per worker instead of
            # once per group; groups are addressed through coordinate offsets.
            # Rows or columns past a group's end read neighboring data that only
            # feeds output rows/columns clipped by the per-group C store.
            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[M, K],
                strides=[stride_am, stride_ak],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[G * N, K],
                strides=[stride_bn, stride_bk],
                block_shape=[BLOCK_N_PER_CTA, BLOCK_SIZE_K],
            )
            a_scale_desc = tl.make_tensor_descriptor(
                a_scale_ptr,
                shape=[1, M // 128, scale_k_chunks, 2, 256],
                strides=[
                    (M // 128) * scale_k_chunks * 2 * 256,
                    scale_k_chunks * 2 * 256,
                    2 * 256,
                    256,
                    1,
                ],
                block_shape=[1, REP_M, REP_K, 2, 256],
            )
            b_scale_desc = tl.make_tensor_descriptor(
                b_scale_ptr,
                shape=[1, G * scale_n_chunks, scale_k_chunks, 2, 256],
                strides=[
                    G * scale_n_chunks * scale_k_chunks * 2 * 256,
                    scale_k_chunks * 2 * 256,
                    2 * 256,
                    256,
                    1,
                ],
                block_shape=[1, REP_N, REP_K, 2, 256],
            )

            for g in range(G):
                m_size = _load_validated_split(split_sizes_ptr, g, start_am, M)
                num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M * NUM_CTAS)
                num_n_tiles: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
                num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
                num_tiles = num_m_tiles * num_n_tiles
                tile_end = tile_start + num_tiles
                prev_accum_cnt_smem = accum_cnt_smem

                if (tile_idx >= tile_start) and (tile_idx < tile_end):
                    a_row_base = start_am
                    b_row_base = start_bn
                    # Global blocked A scales pack 128-row atoms. The running
                    # group prefix is validated before this truncating divide.
                    a_scale_row_base = start_am // 128
                    b_scale_row_base = g * scale_n_chunks

                    while (tile_idx >= tile_start) and (tile_idx < tile_end):
                        if NUM_CTAS == 1:
                            tile_buf = accum_cnt_out % NUM_TILE_BUFFERS
                            _producer_signal_tile_ready(
                                tile_id_smem,
                                tile_id_consumer_bars,
                                tile_buf,
                                tile_idx,
                            )
                        cur_tile_idx = tile_idx - tile_start
                        if m_size < N:
                            tile_m_idx = cur_tile_idx % num_m_tiles
                            tile_n_idx = cur_tile_idx // num_m_tiles
                        else:
                            tile_n_idx = cur_tile_idx % num_n_tiles
                            tile_m_idx = cur_tile_idx // num_n_tiles

                        offs_am = (tile_m_idx * BLOCK_SIZE_M * NUM_CTAS + cluster_cta_rank * BLOCK_SIZE_M)
                        offs_bn = (tile_n_idx * BLOCK_SIZE_N + cluster_cta_rank * BLOCK_N_PER_CTA)
                        scale_m_idx = (tile_m_idx * REP_M * NUM_CTAS + cluster_cta_rank * REP_M)
                        scale_n_idx = tile_n_idx * REP_N

                        for kk in range(0, num_k_tiles):
                            smem_buf, smem_phase = _get_bufidx_phase(
                                accum_cnt_smem,
                                NUM_SMEM_BUFFERS,
                            )
                            tlx.barrier_wait(
                                smem_empty_bars[smem_buf],
                                smem_phase ^ 1,
                            )
                            if NUM_CTAS == 2:
                                tlx.barrier_expect_bytes(
                                    smem_full_bars[smem_buf],
                                    cooperative_smem_bytes,
                                    pred=pred_cta0,
                                )
                            else:
                                tlx.barrier_expect_bytes(
                                    smem_full_bars[smem_buf],
                                    smem_bytes,
                                )
                            tlx.async_descriptor_load(
                                a_desc,
                                buffers_a[smem_buf],
                                [a_row_base + offs_am, kk * BLOCK_SIZE_K],
                                smem_full_bars[smem_buf],
                                eviction_policy=("evict_last" if NUM_CTAS == 2 else ""),
                                two_ctas=NUM_CTAS == 2,
                            )
                            tlx.async_descriptor_load(
                                b_desc,
                                buffers_b[smem_buf],
                                [b_row_base + offs_bn, kk * BLOCK_SIZE_K],
                                smem_full_bars[smem_buf],
                                eviction_policy=("evict_last" if NUM_CTAS == 2 else ""),
                                two_ctas=NUM_CTAS == 2,
                            )

                            scale_k_idx = kk * REP_K
                            tlx.async_descriptor_load(
                                a_scale_desc,
                                buffers_a_scale[smem_buf],
                                [
                                    0,
                                    a_scale_row_base + scale_m_idx,
                                    scale_k_idx,
                                    0,
                                    0,
                                ],
                                smem_full_bars[smem_buf],
                                eviction_policy=("evict_last" if NUM_CTAS == 2 else ""),
                                two_ctas=NUM_CTAS == 2,
                            )
                            if NUM_CTAS == 2:
                                tlx.async_descriptor_load(
                                    b_scale_desc,
                                    buffers_b_scale[smem_buf],
                                    [
                                        0,
                                        b_scale_row_base + scale_n_idx,
                                        scale_k_idx,
                                        0,
                                        0,
                                    ],
                                    smem_full_bars[smem_buf],
                                    pred=pred_cta0,
                                    eviction_policy="evict_last",
                                    multicast_targets=[0, 1],
                                    two_ctas=True,
                                )
                            else:
                                tlx.async_descriptor_load(
                                    b_scale_desc,
                                    buffers_b_scale[smem_buf],
                                    [
                                        0,
                                        b_scale_row_base + scale_n_idx,
                                        scale_k_idx,
                                        0,
                                        0,
                                    ],
                                    smem_full_bars[smem_buf],
                                )
                            accum_cnt_smem += 1

                        if NUM_CTAS == 2:
                            tile_idx += tile_stride
                        else:
                            accum_cnt_out += 1
                            _producer_wait_tile_empty(
                                tile_id_producer_bars,
                                accum_cnt_out,
                                NUM_TILE_BUFFERS,
                            )
                            tile_idx = _producer_fetch_tile_idx_1cta(counter_ptr)

                if accum_cnt_smem > prev_accum_cnt_smem:
                    if (NUM_CTAS == 1) or (cluster_cta_rank == 0):
                        smem_buf, smem_phase = _get_bufidx_phase(
                            accum_cnt_smem - 1,
                            NUM_SMEM_BUFFERS,
                        )
                        tlx.barrier_wait(
                            smem_full_bars[smem_buf],
                            smem_phase,
                        )

                start_am += m_size
                start_bn += N
                tile_start += num_tiles

            _device_trap_if(start_am != M)
            if NUM_CTAS == 1:
                tile_buf = accum_cnt_out % NUM_TILE_BUFFERS
                _producer_wait_tile_empty(
                    tile_id_producer_bars,
                    accum_cnt_out,
                    NUM_TILE_BUFFERS,
                )
                _producer_signal_tile_ready(
                    tile_id_smem,
                    tile_id_consumer_bars,
                    tile_buf,
                    SENTINEL,
                )


@contextlib.contextmanager
def _tma_descriptor_allocator(device: torch.device) -> Iterator[None]:
    """Install the descriptor workspace allocator only for this launch context."""

    def allocate(size: int, _alignment: int, _stream: int | None) -> torch.Tensor:
        return torch.empty(size, device=device, dtype=torch.int8)

    token = _allocation._allocator.set(allocate)
    try:
        yield
    finally:
        _allocation._allocator.reset(token)


def grouped_gemm_mxfp8(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    split_sizes: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    num_sms: int | None = None,
    sf_layout: str = "natural",
) -> torch.Tensor:
    """Launch the forward-only SM100 MXFP8 grouped GEMM backend."""
    if x.ndim != 2 or w.ndim not in (2, 3) or split_sizes.ndim != 1:
        raise ValueError("expected rank-2 x, rank-2/rank-3 w, and rank-1 split_sizes")
    if x.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn:
        raise ValueError("SM100 MXFP8 grouped GEMM supports E4M3 operands only")
    if (x_scale.dtype != torch.float8_e8m0fnu or w_scale.dtype != torch.float8_e8m0fnu):
        raise ValueError("SM100 MXFP8 grouped GEMM requires E8M0 scales")
    if split_sizes.dtype != torch.int32:
        raise ValueError("split_sizes must have dtype torch.int32")

    gm, k = x.shape
    g = split_sizes.shape[0]
    if w.ndim == 3:
        weight_groups, n, weight_k = w.shape
        if weight_groups != g:
            raise ValueError("w group count must match split_sizes")
        w_2d = w.view(weight_groups * n, weight_k)
    else:
        grouped_n, weight_k = w.shape
        if g == 0 or grouped_n % g != 0:
            raise ValueError("rank-2 w rows must be divisible by the group count")
        n = grouped_n // g
        w_2d = w.view(grouped_n, weight_k)

    if min(gm, g, n, k) <= 0 or weight_k != k:
        raise ValueError("grouped GEMM dimensions must be positive and K must match")
    if gm % 128 or k % 128:
        raise ValueError("GM and K must be divisible by 128")
    if n % 8:
        raise ValueError("N must be divisible by 8 for BF16 TMA output")

    tensors = (x, x_scale, w, w_scale, split_sizes)
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("x, scales, w, and split_sizes must be contiguous")
    if x.device.type != "cuda" or any(tensor.device != x.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on x's CUDA device")

    device = x.device
    with torch.cuda.device(device):
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise ValueError("SM100 MXFP8 grouped GEMM requires an SM10x device")
        if out is None:
            out = torch.empty((gm, n), device=device, dtype=torch.bfloat16)
        elif (out.shape != (gm, n) or out.dtype != torch.bfloat16 or out.device != device or not out.is_contiguous()):
            raise ValueError("out must be contiguous BF16 [GM, N] on x's device")

        x_scale_5d, w_scale_5d = prepare_scales(
            x_scale,
            w_scale,
            gm=gm,
            g=g,
            n=n,
            k=k,
            sf_layout=sf_layout,
        )
        launch_sms = (torch.cuda.get_device_properties(device).multi_processor_count if num_sms is None else num_sms)
        if launch_sms <= 0:
            raise ValueError("num_sms must be positive")

        num_ctas = _select_num_ctas(gm=gm, n=n, k=k, launch_sms=launch_sms)
        config = _CONFIG_2CTA_SPEC if num_ctas == 2 else _CONFIG_SPEC
        # The full-domain B-scale descriptor views all groups as one blocked
        # tensor, which requires the per-group scale blocks to be contiguous.
        if w_scale_5d.stride(0) != w_scale_5d[0].numel():
            raise ValueError("w_scale blocks must be contiguous across groups")

        if num_ctas == 1:
            # The explicit zero is captured and replayed with the kernel, so every
            # CUDA graph replay starts dynamic dispatch from tile zero.
            counter = torch.empty(1, dtype=torch.int32, device=device)
            counter.zero_()
        else:
            # The static 2CTA specialization compiles out every counter access.
            counter = split_sizes

        with _tma_descriptor_allocator(device):
            _mxfp8_grouped_gemm_kernel[(launch_sms, )](
                a_ptr=x,
                stride_am=x.stride(0),
                stride_ak=x.stride(1),
                b_ptr=w_2d,
                stride_bn=w_2d.stride(0),
                stride_bk=w_2d.stride(1),
                c_ptr=out,
                a_scale_ptr=x_scale_5d,
                b_scale_ptr=w_scale_5d,
                scale_n_chunks=w_scale_5d.shape[1],
                scale_k_chunks=x_scale_5d.shape[2],
                stride_b_scale_g=w_scale_5d.stride(0),
                split_sizes_ptr=split_sizes,
                counter_ptr=counter,
                G=g,
                M=gm,
                N=n,
                K=k,
                SENTINEL=_TILE_SENTINEL,
                NUM_SM=launch_sms,
                BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
                BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
                BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
                NUM_DATA_BUFFERS=config["NUM_DATA_BUFFERS"],
                NUM_SCALE_BUFFERS=config["NUM_SCALE_BUFFERS"],
                NUM_TMEM_BUFFERS=config["NUM_TMEM_BUFFERS"],
                NUM_TILE_BUFFERS=config["NUM_TILE_BUFFERS"],
                EPILOGUE_SUBTILE=config["EPILOGUE_SUBTILE"],
                OVERLAP_ACC=config["OVERLAP_ACC"],
                NUM_CTAS=num_ctas,
                num_warps=_NUM_WARPS,
                num_stages=_NUM_STAGES,
                ctas_per_cga=(num_ctas, 1, 1) if num_ctas == 2 else None,
            )
    return out


__all__ = ["grouped_gemm_mxfp8"]
