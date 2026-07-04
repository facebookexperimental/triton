# TLX (warp-specialized) HSTU cross-attention BACKWARD reduce_dq kernel
# (attn_bwd_ws), ported verbatim from fbcode hammer
# (hammer/v2/ops/triton/template/tlx_bw_cross_attention.py) for standalone OSS
# use under MetaMain2 triton. The kernel math is unchanged; only the fbcode/hammer
# leaf-dep imports were swapped for the local stubs / utils, mirroring how
# triton_bw_cross_attention.py (the redq port) was adapted.
#
# Buffer plan (single-K reduce_dq):
#   SMEM:  k_tiles / v_tiles / q_tiles / do_tiles / dqk_trans_tiles
#   TMEM:  qk_trans_tiles (+ p_tiles reuse), dq_tiles (+ dp_tiles reuse),
#          standalone dk_tiles, dv_tiles
#   dq is use_acc=False MMA + desc_dq_t.store(store_reduce="add") (not loop
#   carried); dk/dv ARE loop-carried in standalone TMEM.
import os
from typing import List, Optional, Tuple

import torch

import triton
import triton.language as tl

# TLX (Triton Language Extensions) -- present in MetaMain2 triton.
import triton.language.extra.tlx as tlx  # type: ignore[attr-defined]

from triton.tools.tensor_descriptor import TensorDescriptor

from stubs import (
    autotune_max_seq_len,
    next_power_of_2,
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from triton_attention_utils import fast_fma, fast_mul, fast_silu  # noqa: F401
from triton_hstu_cross_attention import (
    _attn_bwd_preprocess,
    backward_d_silu_activation,
    backward_silu_activation,
    backward_valid_mask,
)


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    buf_id = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return buf_id, phase


@triton.jit
def _compute_bwd_reduce_dq_offsets(H, BLOCK_N, seq_offsets_q, seq_offsets):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    start_n = tl.program_id(1) * BLOCK_N
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
    seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    return off_h, start_n, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q


def _bwd_seq_parallel_pre_hook(nargs):
    # Only zero DQ (store_reduce="add" accumulation across CTAs).
    # DK/DV are overwritten via tl.store, so no zeroing needed.
    if "DQ" in nargs:
        nargs["DQ"].zero_()
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    DimQ = nargs["DimQ"]
    DimV = nargs["DimV"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    BLOCK_D_V = nargs["BLOCK_D_V"]
    nargs["desc_q"].block_shape = [BLOCK_M, DimQ]
    nargs["desc_k"].block_shape = [BLOCK_N, DimQ]
    nargs["desc_v"].block_shape = [BLOCK_N, DimV]
    nargs["desc_do"].block_shape = [BLOCK_M, DimV]
    # Backward reduce_dq supports either a shared epilogue subtile or separate
    # DQ/DKV subtile knobs. Handle both to keep older configs working.
    dq_subtile = nargs["EPILOGUE_DQ_SUBTILE"]
    dkv_subtile = nargs["EPILOGUE_DKV_SUBTILE"]
    if dq_subtile is not None:
        nargs["desc_dq"].block_shape = [BLOCK_M, DimQ // dq_subtile]
    if dkv_subtile is not None:
        nargs["desc_dk"].block_shape = [BLOCK_N, DimQ // dkv_subtile]
        nargs["desc_dv"].block_shape = [BLOCK_N, BLOCK_D_V // dkv_subtile]


def _get_bw_pipeline_reduce_dq_configs() -> List[triton.Config]:
    # Minimal config set (the non-full-autotune branch from fbcode): BLOCK_M=64,
    # BLOCK_N=128, NUM_BUFFERS_TMEM=1, TRANSPOSE toggled.
    configs = [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "NUM_BUFFERS_Q": 2,
                "NUM_BUFFERS_KV": 1,
                "NUM_BUFFERS_DO": num_buffers_do,
                "NUM_BUFFERS_DS": 1,
                "NUM_BUFFERS_TMEM": 1,
                "EPILOGUE_DQ_SUBTILE": 1,
                "EPILOGUE_DKV_SUBTILE": 2,
                "COMPUTE_REG": 176,
                "MMA_REG": 48,
                "LOAD_REG": 24,
                "TRANSPOSE": transpose,
            },
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_seq_parallel_pre_hook,
        )
        for num_buffers_do, transpose in [
            (2, True),
            #(1, True),
            #(2, False),
            #(1, False),
        ]
    ]
    return configs


@triton_autotune(
    configs=_get_bw_pipeline_reduce_dq_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_Q_LEN",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "SHARED_KV",
    ],
)
@triton.jit
def attn_bwd_ws(  # noqa C901
    desc_q,
    desc_k,
    desc_v,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    desc_do,
    desc_dq,
    desc_dk,
    desc_dv,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    stride_mm,
    Z,
    AUTOTUNE_Z,
    H,
    AUTOTUNE_MAX_Q_LEN,  # Quantized MAX_Q_LEN used as an autotuning key
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    EPILOGUE_DQ_SUBTILE: tl.constexpr,
    EPILOGUE_DKV_SUBTILE: tl.constexpr,
    COMPUTE_REG: tl.constexpr,
    MMA_REG: tl.constexpr,
    LOAD_REG: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    SHARED_KV: tl.constexpr,
    SOFTMAX: tl.constexpr,
    # When True, store dq transposed as dq^T [DimQ, BLOCK_M] (fewer TMEM columns
    # at BLOCK_M < DimQ); when False, use the [BLOCK_M, DimQ] layout. Autotuned.
    TRANSPOSE: tl.constexpr,
):
    tl.static_assert(NUM_BUFFERS_Q >= 2)

    # allocate SMEM buffers and barriers
    # pyrefly: ignore [missing-attribute]
    k_tiles = tlx.local_alloc((BLOCK_N, DimQ), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    if not SHARED_KV:
        # pyrefly: ignore [missing-attribute]
        v_tiles = tlx.local_alloc((BLOCK_N, DimV), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)
    # pyrefly: ignore [missing-attribute]
    q_tiles = tlx.local_alloc((BLOCK_M, DimQ), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    # pyrefly: ignore [missing-attribute]
    do_tiles = tlx.local_alloc((BLOCK_M, DimV), tlx.dtype_of(desc_do), NUM_BUFFERS_DO)

    # Use SMEM for dqk_trans
    # pyrefly: ignore [missing-attribute]
    dqk_trans_tiles = tlx.local_alloc(
        # pyrefly: ignore [missing-attribute]
        (BLOCK_N, BLOCK_M),
        # pyrefly: ignore [missing-attribute]
        tlx.dtype_of(desc_q),
        NUM_BUFFERS_DS,
    )

    # pyrefly: ignore [missing-attribute]
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    # pyrefly: ignore [missing-attribute]
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # pyrefly: ignore [missing-attribute]
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    if not SHARED_KV:
        # pyrefly: ignore [missing-attribute]
        v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
        # pyrefly: ignore [missing-attribute]
        v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # pyrefly: ignore [missing-attribute]
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    # pyrefly: ignore [missing-attribute]
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    # pyrefly: ignore [missing-attribute]
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    # pyrefly: ignore [missing-attribute]
    dqk_trans_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    # allocate TMEM buffers and barriers
    # pyrefly: ignore [missing-attribute]
    qk_trans_tiles = tlx.local_alloc(
        (BLOCK_N, BLOCK_M),
        tl.float32,
        NUM_BUFFERS_TMEM,
        # pyrefly: ignore [missing-attribute]
        tlx.storage_kind.tmem,
    )

    # P
    # pyrefly: ignore [missing-attribute]
    p_tiles = tlx.local_alloc(
        (BLOCK_N, BLOCK_M),
        # pyrefly: ignore [missing-attribute]
        tlx.dtype_of(desc_do),
        NUM_BUFFERS_TMEM,
        # pyrefly: ignore [missing-attribute]
        tlx.storage_kind.tmem,
        reuse=qk_trans_tiles,
    )
    # TRANSPOSE: store dq transposed as dq^T ([DimQ, BLOCK_M]) instead of
    # [BLOCK_M, DimQ]. With BLOCK_M < DimQ this shrinks its TMEM column footprint
    # from DimQ to BLOCK_M columns (e.g. 128 -> 64 at BLOCK_M=64), and makes dq^T
    # the same [*, BLOCK_M] footprint as dp ([BLOCK_N, BLOCK_M]) so dp can
    # time-share it cleanly. dq^T = (dS @ K)^T = K^T @ dS^T; the reduce epilogue
    # transposes it back before the dQ store.
    if TRANSPOSE:
        # pyrefly: ignore [missing-attribute]
        dq_tiles = tlx.local_alloc(
            (DimQ, BLOCK_M),
            tl.float32,
            NUM_BUFFERS_TMEM,
            # pyrefly: ignore [missing-attribute]
            tlx.storage_kind.tmem,
        )
    else:
        # pyrefly: ignore [missing-attribute]
        dq_tiles = tlx.local_alloc(
            (BLOCK_M, DimQ),
            tl.float32,
            NUM_BUFFERS_TMEM,
            # pyrefly: ignore [missing-attribute]
            tlx.storage_kind.tmem,
        )

    # pyrefly: ignore [missing-attribute]
    dp_tiles = tlx.local_alloc(
        (BLOCK_N, BLOCK_M),
        tl.float32,
        1,
        # pyrefly: ignore [missing-attribute]
        tlx.storage_kind.tmem,
        # dp time-shares dq's (larger) TMEM. Without this reuse the standalone
        # tiles overflow 512 TMEM columns at BLOCK_M=128, so dk_tiles overlaps a
        # neighbor and its upper column-subtiles get corrupted.
        reuse=dq_tiles,
    )
    # pyrefly: ignore [missing-attribute]
    dk_tiles = tlx.local_alloc(
        (BLOCK_N, DimQ),
        tl.float32,
        1,
        # pyrefly: ignore [missing-attribute]
        tlx.storage_kind.tmem,
    )
    # SHARED_KV folds dv into dk's tmem (alpha is pre-scaled into dqk_trans), so
    # the separate dv_tiles tile is only needed when K and V are distinct.
    if not SHARED_KV:
        # pyrefly: ignore [missing-attribute]
        dv_tiles = tlx.local_alloc(
            (BLOCK_N, DimV),
            tl.float32,
            1,
            # pyrefly: ignore [missing-attribute]
            tlx.storage_kind.tmem,
        )

    # pyrefly: ignore [missing-attribute]
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # pyrefly: ignore [missing-attribute]
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # pyrefly: ignore [missing-attribute]
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # pyrefly: ignore [missing-attribute]
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # pyrefly: ignore [missing-attribute]
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # pyrefly: ignore [missing-attribute]
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # dk_tiles / dv_tiles are single-buffered, so their full/empty handshakes are
    # depth-1 and must NOT be sized to NUM_BUFFERS_KV (which only buffers the K/V
    # *loads*). Sizing these to NUM_BUFFERS_KV let consecutive N-blocks use
    # different barrier slots for the same physical tile -> dk/dv corruption at
    # NUM_BUFFERS_KV>1. Index these with [0] and a depth-1 phase everywhere.
    # pyrefly: ignore [missing-attribute]
    dk_fulls = tlx.alloc_barriers(num_barriers=1)
    # pyrefly: ignore [missing-attribute]
    dk_empties = tlx.alloc_barriers(num_barriers=1)
    # pyrefly: ignore [missing-attribute]
    dv_fulls = tlx.alloc_barriers(num_barriers=1)
    # pyrefly: ignore [missing-attribute]
    dv_empties = tlx.alloc_barriers(num_barriers=1)

    # pyrefly: ignore [missing-attribute]
    with tlx.async_tasks():
        # Reduce
        # pyrefly: ignore [missing-attribute]
        with tlx.async_task("default"):
            off_h, _start_n, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (
                _compute_bwd_reduce_dq_offsets(H, BLOCK_N, seq_offsets_q, seq_offsets)
            )
            # dq/dk/dv stores use on-device TMA descriptors bounded per-program to
            # [end, ...]: partial M/N blocks are dropped by TMA bounds (no
            # adjacent-batch corruption, no mask needed) and the addressing is
            # int64 internally, so the global seq_start_* * stride never overflows
            # int32 (the multi-GB-tensor IMA). dk/dv use plain stores; dq uses
            # store_reduce="add" to accumulate its contribution across N blocks.
            if seq_len_kv > 0:
                dq_end_q = (seq_start_q + seq_len_q).to(tl.int32)
                # TRANSPOSE stores dq from a transposed accumulator as one full
                # [BLOCK_M, DimQ] block (no DQ subtile); otherwise subtile DimQ.
                dq_block_d: tl.constexpr = (
                    BLOCK_D_Q if TRANSPOSE else BLOCK_D_Q // EPILOGUE_DQ_SUBTILE
                )
                desc_dq_t = tl.make_tensor_descriptor(
                    DQ,
                    shape=[dq_end_q, H * DimQ],
                    # pyrefly: ignore [bad-argument-type]
                    strides=[stride_dqm, 1],
                    block_shape=[BLOCK_M, dq_block_d],
                )
                dk_end_kv = (seq_start_kv + seq_len_kv).to(tl.int32)
                desc_dk_t = tl.make_tensor_descriptor(
                    DK,
                    shape=[dk_end_kv, H * DimQ],
                    # pyrefly: ignore [bad-argument-type]
                    strides=[stride_dkn, 1],
                    block_shape=[BLOCK_N, DimQ // EPILOGUE_DKV_SUBTILE],
                )
                if not SHARED_KV:
                    desc_dv_t = tl.make_tensor_descriptor(
                        DV,
                        shape=[dk_end_kv, H * DimV],
                        # pyrefly: ignore [bad-argument-type]
                        strides=[stride_dvn, 1],
                        block_shape=[BLOCK_N, BLOCK_D_V // EPILOGUE_DKV_SUBTILE],
                    )

                accum_cnt_q = 0
                accum_cnt_n = 0
                for start_n in tl.range(0, seq_len_kv, BLOCK_N):
                    for start_m in tl.range(0, seq_len_q, BLOCK_M):
                        tmem_buf_id, tmem_phase = _get_bufidx_phase(
                            accum_cnt_q, NUM_BUFFERS_TMEM
                        )
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                        # alpha already folded into dqk_trans (-> dq).
                        if TRANSPOSE:
                            # dq_tiles holds dq^T [DimQ, BLOCK_M]; load and
                            # transpose back to [BLOCK_M, DimQ] for the store.
                            # pyrefly: ignore [missing-attribute]
                            dq_t = tlx.local_load(dq_tiles[tmem_buf_id])
                            dq = tl.trans(dq_t)
                            desc_dq_t.store(
                                [
                                    (seq_start_q + start_m).to(tl.int32),
                                    off_h * stride_dqh,
                                ],
                                dq,
                                store_reduce="add",
                            )
                        else:
                            slice_size: tl.constexpr = BLOCK_D_Q // EPILOGUE_DQ_SUBTILE
                            for slice_id in tl.static_range(EPILOGUE_DQ_SUBTILE):
                                # pyrefly: ignore [missing-attribute]
                                dq_slice = tlx.local_slice(
                                    dq_tiles[tmem_buf_id],
                                    [0, slice_id * slice_size],
                                    [BLOCK_M, slice_size],
                                )
                                # pyrefly: ignore [missing-attribute]
                                dq = tlx.local_load(dq_slice)
                                desc_dq_t.store(
                                    [
                                        (seq_start_q + start_m).to(tl.int32),
                                        off_h * stride_dqh + slice_id * slice_size,
                                    ],
                                    dq,
                                    store_reduce="add",
                                )

                        # release dq
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_arrive(dq_empties[tmem_buf_id])
                        accum_cnt_q += 1

                    # dk/dv epilogue: store per N block after all M blocks.
                    # dk/dv tiles are single-buffered -> depth-1 barrier index[0]
                    # and phase, independent of the NUM_BUFFERS_KV load pipeline.
                    dkv_buf_id, dkv_phase = _get_bufidx_phase(accum_cnt_n, 1)

                    # Store dk
                    dk_slice_size: tl.constexpr = DimQ // EPILOGUE_DKV_SUBTILE
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(dk_fulls[dkv_buf_id], dkv_phase)
                    for slice_id in tl.static_range(EPILOGUE_DKV_SUBTILE):
                        # pyrefly: ignore [missing-attribute]
                        dk_slice = tlx.local_slice(
                            dk_tiles[0],
                            [0, slice_id * dk_slice_size],
                            [BLOCK_N, dk_slice_size],
                        )
                        # pyrefly: ignore [missing-attribute]
                        dk = tlx.local_load(dk_slice)
                        # alpha already folded into dqk_trans (-> dk); for
                        # SHARED_KV the MMA already accumulated dv into dk_tiles.
                        desc_dk_t.store(
                            [
                                (seq_start_kv + start_n).to(tl.int32),
                                off_h * stride_dkh + slice_id * dk_slice_size,
                            ],
                            dk.to(desc_k.dtype),
                        )
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_arrive(dk_empties[dkv_buf_id])

                    # Store dv
                    if not SHARED_KV:
                        dv_slice_size: tl.constexpr = BLOCK_D_V // EPILOGUE_DKV_SUBTILE
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(dv_fulls[dkv_buf_id], dkv_phase)
                        for slice_id in tl.static_range(EPILOGUE_DKV_SUBTILE):
                            # pyrefly: ignore [missing-attribute]
                            dv_slice = tlx.local_slice(
                                dv_tiles[0],
                                [0, slice_id * dv_slice_size],
                                [BLOCK_N, dv_slice_size],
                            )
                            # pyrefly: ignore [missing-attribute]
                            dv = tlx.local_load(dv_slice)
                            # pyrefly: ignore [unbound-name]
                            desc_dv_t.store(
                                [
                                    (seq_start_kv + start_n).to(tl.int32),
                                    off_h * stride_dvh + slice_id * dv_slice_size,
                                ],
                                # pyrefly: ignore [missing-attribute]
                                dv.to(tlx.dtype_of(desc_v)),
                            )
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_arrive(dv_empties[dkv_buf_id])

                    accum_cnt_n += 1

        # Compute
        # pyrefly: ignore [missing-attribute]
        with tlx.async_task(num_warps=8, registers=COMPUTE_REG):
            off_h, _start_n, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (
                _compute_bwd_reduce_dq_offsets(H, BLOCK_N, seq_offsets_q, seq_offsets)
            )
            if SOFTMAX:
                # base pointers into the per-(row, head) softmax statistics
                M_off = M + off_h + seq_start_q * stride_mm
                Delta_off = Delta + off_h + seq_start_q * stride_mm

            accum_cnt_q = 0
            for start_n in tl.range(0, seq_len_kv, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len_kv
                if not SOFTMAX:
                    if ATTN_SCALE_TYPE == "scalar":
                        scale = tl.load(attn_scale).to(tl.float32)
                    else:
                        tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
                        scale = tl.load(attn_scale + offs_n, mask=mask_n).to(tl.float32)

                for start_m in tl.range(0, seq_len_q, BLOCK_M):
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(
                        accum_cnt_q, NUM_BUFFERS_TMEM
                    )
                    ds_buf_id, _ = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_DS)

                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    mask_m = offs_m < seq_len_q
                    # wait for qkT
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(qk_fulls[tmem_buf_id], tmem_phase)
                    # pyrefly: ignore [missing-attribute]
                    qk_trans = tlx.local_load(qk_trans_tiles[tmem_buf_id])
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_arrive(qk_empties[tmem_buf_id])
                    valid_mask_trans = backward_valid_mask(
                        offs_m,
                        offs_n,
                        0,  # uih_len_q
                        seq_len_q,
                        seq_len_kv,
                        False,  # HAS_CAUSAL
                    )
                    if SOFTMAX:
                        # recompute pT = exp2(qk * alpha * log2(e) - M)
                        # pyrefly: ignore [unbound-name]
                        m = tl.load(M_off + offs_m * stride_mm, mask=mask_m)
                        # Fuse the alpha*log2(e) scale and the -M subtract into a
                        # single f32x2 fma before exp2.
                        qk_trans = fast_fma(qk_trans, alpha * 1.44269504, -m[None, :])
                        pT = tl.math.exp2(qk_trans)
                        pT = tl.where(valid_mask_trans, pT, 0.0)
                        # pyrefly: ignore [missing-attribute]
                        act_qk_trans = pT.to(tlx.dtype_of(desc_do))
                    else:
                        qk_trans, act_qk_trans, pT = backward_silu_activation(
                            qk_trans,
                            alpha,
                            valid_mask_trans,
                            # pyrefly: ignore [missing-attribute]
                            tlx.dtype_of(desc_do),
                            # pyrefly: ignore [unbound-name]
                            scale,
                        )
                    # pyrefly: ignore [missing-attribute]
                    tlx.local_store(p_tiles[tmem_buf_id], act_qk_trans)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_arrive(p_fulls[tmem_buf_id])

                    # dk and dq
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(dp_fulls[tmem_buf_id], tmem_phase)
                    # pyrefly: ignore [missing-attribute]
                    dact_qk_trans = tlx.local_load(dp_tiles[tmem_buf_id])
                    if SOFTMAX:
                        # dS = pT * (dP - Delta)
                        # pyrefly: ignore [unbound-name]
                        Di = tl.load(Delta_off + offs_m * stride_mm, mask=mask_m)
                        dqk_trans = fast_mul(pT, dact_qk_trans - Di[None, :])
                    else:
                        dqk_trans = backward_d_silu_activation(
                            # pyrefly: ignore [unbound-name]
                            dact_qk_trans,
                            pT,
                            qk_trans,
                            # pyrefly: ignore [unbound-name]
                            scale,
                            valid_mask_trans,
                        )
                    # Pre-scale by alpha (fp32, pre-cast) so the dk dot
                    # (dqk_trans @ q) and dq dot (dqk @ k) already carry alpha;
                    # the MMA/Reduce skip the post-scale and dk can fold dv into a
                    # single tmem buffer for SHARED_KV.
                    dqk_trans = fast_mul(dqk_trans, alpha)
                    # pyrefly: ignore [missing-attribute]
                    dqk_trans = dqk_trans.to(tlx.dtype_of(desc_do))
                    # pyrefly: ignore [missing-attribute]
                    tlx.local_store(dqk_trans_tiles[ds_buf_id], dqk_trans)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_arrive(dqk_trans_fulls[ds_buf_id])

                    accum_cnt_q += 1
                # dk/dv epilogue skipped - Reduce task handles stores

        # MMA
        # pyrefly: ignore [missing-attribute]
        with tlx.async_task(num_warps=1, registers=MMA_REG):
            off_h, _start_n, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (
                _compute_bwd_reduce_dq_offsets(H, BLOCK_N, seq_offsets_q, seq_offsets)
            )

            accum_cnt_q = 0
            accum_cnt_n = 0
            for _start_n in tl.range(0, seq_len_kv, BLOCK_N):
                accum_cnt_q_n_start = accum_cnt_q
                # wait for K (and V) for this N block
                kv_buf_id, kv_phase = _get_bufidx_phase(accum_cnt_n, NUM_BUFFERS_KV)
                # dk/dv tiles are single-buffered: depth-1 index[0]/phase, and the
                # producer must wait for the consumer to drain the tile before
                # EVERY reuse (every N-block after the first) -- not just after
                # NUM_BUFFERS_KV blocks (that count belongs to the K/V load tiles).
                dkv_buf_id, dkv_phase = _get_bufidx_phase(accum_cnt_n, 1)
                if accum_cnt_n >= 1:
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(dk_empties[dkv_buf_id], dkv_phase ^ 1)
                    if not SHARED_KV:
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(dv_empties[dkv_buf_id], dkv_phase ^ 1)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
                if not SHARED_KV:
                    # pyrefly: ignore [missing-attribute, unbound-name]
                    tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)

                # prologue
                q_buff_id, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                do_buf_id, do_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_DO)
                tmem_buf_id, tmem_phase = _get_bufidx_phase(
                    accum_cnt_q, NUM_BUFFERS_TMEM
                )
                # wait for q
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(q_fulls[q_buff_id], q_phase)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
                # q @ kT
                # pyrefly: ignore [missing-attribute]
                q_trans = tlx.local_trans(q_tiles[q_buff_id])
                # k0 @ q
                # pyrefly: ignore [missing-attribute]
                tlx.async_dot(
                    k_tiles[kv_buf_id],
                    q_trans,
                    qk_trans_tiles[tmem_buf_id],
                    use_acc=False,
                    mBarriers=[qk_fulls[tmem_buf_id]],
                )
                # compute dpT
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                # As dP uses the same tmem as dQ, wait for dQ release.
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
                # pyrefly: ignore [missing-attribute]
                do_trans = tlx.local_trans(do_tiles[do_buf_id])
                # v @ do_trans (or k @ do_trans when SHARED_KV)
                if SHARED_KV:
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_dot(
                        k_tiles[kv_buf_id],
                        do_trans,
                        dp_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[dp_fulls[tmem_buf_id]],
                    )
                else:
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_dot(
                        # pyrefly: ignore [missing-attribute, unbound-name]
                        v_tiles[kv_buf_id],
                        do_trans,
                        dp_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[dp_fulls[tmem_buf_id]],
                    )
                # dv0: for SHARED_KV this initializes the merged dk_tiles
                # (alpha-free dv); the dk dots accumulate onto it below.
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
                if SHARED_KV:
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_dot(
                        p_tiles[tmem_buf_id],
                        do_tiles[do_buf_id],
                        dk_tiles[0],
                        use_acc=False,
                        mBarriers=[do_empties[do_buf_id]],
                    )
                else:
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_dot(
                        p_tiles[tmem_buf_id],
                        do_tiles[do_buf_id],
                        # pyrefly: ignore [unbound-name]
                        dv_tiles[0],
                        use_acc=False,
                        mBarriers=[do_empties[do_buf_id]],
                    )

                accum_cnt_q += 1
                # Mainloop
                for _start_m in tl.range(BLOCK_M, seq_len_q, BLOCK_M):
                    q_buff_id, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(
                        accum_cnt_q, NUM_BUFFERS_TMEM
                    )

                    # q @ kT
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(q_fulls[q_buff_id], q_phase)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
                    # pyrefly: ignore [missing-attribute]
                    q_trans = tlx.local_trans(q_tiles[q_buff_id])
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_dot(
                        k_tiles[kv_buf_id],
                        q_trans,
                        qk_trans_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[qk_fulls[tmem_buf_id]],
                    )

                    prev_cnt_q = accum_cnt_q - 1
                    q_buff_id_prev, _ = _get_bufidx_phase(prev_cnt_q, NUM_BUFFERS_Q)
                    tmem_buf_id_prev, tmem_phase_prev = _get_bufidx_phase(
                        prev_cnt_q, NUM_BUFFERS_TMEM
                    )
                    ds_buf_id_prev, ds_phase_prev = _get_bufidx_phase(
                        prev_cnt_q, NUM_BUFFERS_DS
                    )
                    # compute dQ
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(dqk_trans_fulls[tmem_buf_id_prev], ds_phase_prev)
                    # pyrefly: ignore [missing-attribute]
                    tlx.fence_async_shared()
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(dq_empties[tmem_buf_id_prev], tmem_phase_prev ^ 1)
                    if TRANSPOSE:
                        # dq^T = K^T @ dS^T (dqk_trans is dS^T): transpose K instead
                        # of dqk so the accumulator lands as [DimQ, BLOCK_M].
                        # pyrefly: ignore [missing-attribute]
                        k_trans = tlx.local_trans(k_tiles[kv_buf_id])
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            k_trans,
                            dqk_trans_tiles[ds_buf_id_prev],
                            # pyrefly: ignore [missing-attribute]
                            dq_tiles[tmem_buf_id_prev],
                            use_acc=False,
                            mBarriers=[dq_fulls[tmem_buf_id_prev]],
                        )
                    else:
                        # dq = dS @ K -> [BLOCK_M, DimQ]
                        # pyrefly: ignore [missing-attribute]
                        dqk = tlx.local_trans(dqk_trans_tiles[ds_buf_id_prev])
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            dqk,
                            k_tiles[kv_buf_id],
                            # pyrefly: ignore [missing-attribute]
                            dq_tiles[tmem_buf_id_prev],
                            use_acc=False,
                            mBarriers=[dq_fulls[tmem_buf_id_prev]],
                        )

                    # compute dk(i-1). For SHARED_KV always accumulate: dv0
                    # already initialized the merged dk_tiles for this N block.
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_dot(
                        dqk_trans_tiles[ds_buf_id_prev],
                        q_tiles[q_buff_id_prev],
                        dk_tiles[tmem_buf_id_prev],
                        use_acc=SHARED_KV or prev_cnt_q > accum_cnt_q_n_start,
                        mBarriers=[q_empties[q_buff_id_prev]],
                    )

                    # compute dpT
                    do_buf_id, do_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_DO)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                    # As dP uses the same tmem as dQ, wait for dQ release.
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
                    # pyrefly: ignore [missing-attribute]
                    do_trans = tlx.local_trans(do_tiles[do_buf_id])
                    # v @ do_trans (or k @ do_trans when SHARED_KV)
                    if SHARED_KV:
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            k_tiles[kv_buf_id],
                            do_trans,
                            dp_tiles[tmem_buf_id],
                            use_acc=False,
                            mBarriers=[dp_fulls[tmem_buf_id]],
                        )
                    else:
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            # pyrefly: ignore [missing-attribute, unbound-name]
                            v_tiles[kv_buf_id],
                            do_trans,
                            dp_tiles[tmem_buf_id],
                            use_acc=False,
                            mBarriers=[dp_fulls[tmem_buf_id]],
                        )

                    # compute dv(i) (accumulate into merged dk_tiles for shared)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
                    if SHARED_KV:
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            p_tiles[tmem_buf_id],
                            do_tiles[do_buf_id],
                            dk_tiles[0],
                            use_acc=True,
                            mBarriers=[do_empties[do_buf_id]],
                        )
                    else:
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            p_tiles[tmem_buf_id],
                            do_tiles[do_buf_id],
                            # pyrefly: ignore [unbound-name]
                            dv_tiles[0],
                            use_acc=True,
                            mBarriers=[do_empties[do_buf_id]],
                        )

                    accum_cnt_q += 1
                if not SHARED_KV:
                    # dv published via dk_fulls (merged tile) for SHARED_KV.
                    # pyrefly: ignore [missing-attribute]
                    tlx.tcgen05_commit(dv_fulls[0])

                # epilogue
                q_buff_id, q_phase = _get_bufidx_phase(accum_cnt_q - 1, NUM_BUFFERS_Q)
                tmem_buf_id, tmem_phase = _get_bufidx_phase(
                    accum_cnt_q - 1, NUM_BUFFERS_TMEM
                )
                ds_buf_id, ds_phase = _get_bufidx_phase(accum_cnt_q - 1, NUM_BUFFERS_DS)

                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(dqk_trans_fulls[ds_buf_id], ds_phase)
                # pyrefly: ignore [missing-attribute]
                tlx.fence_async_shared()
                # pyrefly: ignore [missing-attribute]
                tlx.async_dot(
                    dqk_trans_tiles[ds_buf_id],
                    q_tiles[q_buff_id],
                    dk_tiles[0],
                    # SHARED_KV: always accumulate onto the merged dv/dk tile.
                    use_acc=SHARED_KV or (accum_cnt_q - accum_cnt_q_n_start) > 1,
                    mBarriers=[q_empties[q_buff_id], dk_fulls[0]],
                )

                # acc dq (last use of K/V for this N block)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
                if SHARED_KV:
                    kv_release_barriers = [k_empties[kv_buf_id]]
                else:
                    kv_release_barriers = [
                        k_empties[kv_buf_id],
                        # pyrefly: ignore [unbound-name]
                        v_empties[kv_buf_id],
                    ]
                if TRANSPOSE:
                    # dq^T = K^T @ dS^T (see mainloop dq dot above).
                    # pyrefly: ignore [missing-attribute]
                    k_trans = tlx.local_trans(k_tiles[kv_buf_id])
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_dot(
                        k_trans,
                        dqk_trans_tiles[ds_buf_id],
                        dq_tiles[tmem_buf_id],
                        use_acc=False,
                        # pyrefly: ignore [missing-attribute]
                        mBarriers=[dq_fulls[tmem_buf_id]] + kv_release_barriers,
                    )
                else:
                    # dq = dS @ K -> [BLOCK_M, DimQ]
                    # pyrefly: ignore [missing-attribute]
                    dqk = tlx.local_trans(dqk_trans_tiles[ds_buf_id])
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_dot(
                        dqk,
                        k_tiles[kv_buf_id],
                        dq_tiles[tmem_buf_id],
                        use_acc=False,
                        # pyrefly: ignore [missing-attribute]
                        mBarriers=[dq_fulls[tmem_buf_id]] + kv_release_barriers,
                    )

                accum_cnt_n += 1

        # Load
        # pyrefly: ignore [missing-attribute]
        with tlx.async_task(num_warps=1, registers=LOAD_REG):
            # initialize offsets
            off_h, _start_n, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (
                _compute_bwd_reduce_dq_offsets(H, BLOCK_N, seq_offsets_q, seq_offsets)
            )

            accum_cnt_q = 0
            accum_cnt_n = 0
            for start_n in tl.range(0, seq_len_kv, BLOCK_N):
                # load K for this N block
                k_buff_id, kv_phase = _get_bufidx_phase(accum_cnt_n, NUM_BUFFERS_KV)
                if accum_cnt_n >= NUM_BUFFERS_KV:
                    _, kv_empty_phase = _get_bufidx_phase(
                        accum_cnt_n - NUM_BUFFERS_KV, NUM_BUFFERS_KV
                    )
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(k_empties[k_buff_id], kv_empty_phase)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_expect_bytes(
                    k_fulls[k_buff_id], 2 * BLOCK_N * DimQ
                )  # float16
                kv_offset = seq_start_kv + start_n
                # pyrefly: ignore [missing-attribute]
                tlx.async_descriptor_load(
                    desc_k,
                    k_tiles[k_buff_id],
                    [kv_offset.to(tl.int32), off_h * stride_kh],
                    k_fulls[k_buff_id],
                )
                if not SHARED_KV:
                    # load V for this N block
                    if accum_cnt_n >= NUM_BUFFERS_KV:
                        _, kv_empty_phase_v = _get_bufidx_phase(
                            accum_cnt_n - NUM_BUFFERS_KV, NUM_BUFFERS_KV
                        )
                        # pyrefly: ignore [missing-attribute, unbound-name]
                        tlx.barrier_wait(v_empties[k_buff_id], kv_empty_phase_v)
                    # pyrefly: ignore [missing-attribute, unbound-name]
                    tlx.barrier_expect_bytes(
                        # pyrefly: ignore [unbound-name]
                        v_fulls[k_buff_id],
                        2 * BLOCK_N * DimV,
                    )  # float16
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_descriptor_load(
                        desc_v,
                        # pyrefly: ignore [missing-attribute, unbound-name]
                        v_tiles[k_buff_id],
                        [kv_offset.to(tl.int32), off_h * stride_vh],
                        # pyrefly: ignore [missing-attribute]
                        v_fulls[k_buff_id],
                    )

                # load Q and dO for all M blocks
                q_buff_id, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(q_empties[q_buff_id], q_phase ^ 1)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_expect_bytes(
                    q_fulls[q_buff_id], 2 * BLOCK_M * DimQ
                )  # float16
                q_offset = seq_start_q
                # pyrefly: ignore [missing-attribute]
                tlx.async_descriptor_load(
                    desc_q,
                    q_tiles[q_buff_id],
                    [q_offset.to(tl.int32), off_h * stride_qh],
                    q_fulls[q_buff_id],
                )

                # load DO
                do_buf_id, do_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_DO)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_expect_bytes(
                    do_fulls[do_buf_id], 2 * BLOCK_M * DimV
                )  # float16
                # pyrefly: ignore [missing-attribute]
                tlx.async_descriptor_load(
                    desc_do,
                    do_tiles[do_buf_id],
                    [q_offset.to(tl.int32), off_h * stride_doh],
                    do_fulls[do_buf_id],
                )

                accum_cnt_q += 1
                for start_m in tl.range(BLOCK_M, seq_len_q, BLOCK_M):
                    # load Q
                    q_buff_id, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    do_buf_id, do_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_DO)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(q_empties[q_buff_id], q_phase ^ 1)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_expect_bytes(
                        q_fulls[q_buff_id], 2 * BLOCK_M * DimQ
                    )  # float16
                    q_offset = seq_start_q + start_m
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_buff_id],
                        [q_offset.to(tl.int32), off_h * stride_qh],
                        q_fulls[q_buff_id],
                    )
                    # load dO
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_expect_bytes(
                        do_fulls[do_buf_id], 2 * BLOCK_M * DimV
                    )  # float16
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_descriptor_load(
                        desc_do,
                        do_tiles[do_buf_id],
                        [q_offset.to(tl.int32), off_h * stride_doh],
                        do_fulls[do_buf_id],
                    )
                    accum_cnt_q += 1
                accum_cnt_n += 1


def tlx_bw_reduce_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    shared_kv: bool = False,
    num_softmax_heads: int = 0,
    out: Optional[torch.Tensor] = None,
    M: Optional[torch.Tensor] = None,
    use_2kv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the TLX warp-specialized reduce_dq backward (attn_bwd_ws).

    When use_2kv=True, launches the 2-KV-block data-partitioned variant
    (attn_bwd_ws_2kv), which processes 2 KV blocks per program as two independent
    MMA groups. It is shared-KV + transpose by construction, so it requires
    shared_kv=True and takes no SHARED_KV/TRANSPOSE kernel args.

    Mirrors the fbcode tlx_hstu_attention_bwd dispatch (reduce_dq path). Returns
    (dq, dk, dv); dv aliases dk when shared_kv.
    """
    dout = switch_to_contiguous_if_needed(dout)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = dk if shared_kv else torch.empty_like(v)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    if not shared_kv:
        dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets

    total_seq_len_q, H, DimQ = q.shape
    total_seq_len_kv, _, _ = k.shape
    _, _, DimV = v.shape

    if attn_scale.ndim == 0:
        attn_scale_type = "scalar"
    else:
        attn_scale_type = "dynamic"

    # Softmax statistics: when enabled, M (base-2 logsumexp from forward) is
    # required and Delta = rowsum(O * dO) is computed here via a preprocess pass.
    # All-or-nothing: num_softmax_heads is either 0 (SiLU) or H (softmax).
    SOFTMAX = num_softmax_heads != 0
    if SOFTMAX:
        assert num_softmax_heads == H, (
            "tlx cross attention softmax requires num_softmax_heads == 0 or H"
        )
        assert M is not None and out is not None, (
            "softmax backward requires forward M and out"
        )
        stride_mm = M.stride(0)
        Delta = torch.empty_like(M)
        pre_grid = (triton.cdiv(out.shape[0], 128), num_softmax_heads)
        _attn_bwd_preprocess[pre_grid](
            out,
            dout,
            Delta,
            out.shape[0],
            H=H,
            softmax_heads=num_softmax_heads,
            BLOCK_M=128,
            HEAD_DIM=DimV,
        )
    else:
        M = torch.empty(0, device=q.device, dtype=torch.float32)
        Delta = torch.empty(0, device=q.device, dtype=torch.float32)
        stride_mm = 0

    # TMA descriptors (host-side dummy block_shape, filled by the pre_hook).
    dummy_block = [1, 1]
    desc_q = TensorDescriptor(
        q, shape=[total_seq_len_q, H * DimQ], strides=[H * DimQ, 1],
        block_shape=dummy_block,
    )
    desc_v = TensorDescriptor(
        v, shape=[total_seq_len_kv, H * DimV], strides=[H * DimV, 1],
        block_shape=dummy_block,
    )
    desc_k = TensorDescriptor(
        k, shape=[total_seq_len_kv, H * DimQ], strides=[H * DimQ, 1],
        block_shape=dummy_block,
    )
    desc_do = TensorDescriptor(
        dout, shape=[total_seq_len_q, H * DimV], strides=[H * DimV, 1],
        block_shape=dummy_block,
    )

    AUTOTUNE_Z = Z

    # On-device TMA descriptors (tl.make_tensor_descriptor) for the grad stores
    # need a runtime scratch allocator registered.
    def _bwd_alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device=q.device, dtype=torch.int8)

    triton.set_allocator(_bwd_alloc_fn)

    grid = lambda meta: (Z * H, 1)  # noqa E731
    desc_dq = TensorDescriptor(
        dq, shape=[total_seq_len_q, H * DimQ], strides=[H * DimQ, 1],
        block_shape=dummy_block,
    )
    desc_dk = TensorDescriptor(
        dk, shape=[total_seq_len_kv, H * DimQ], strides=[H * DimQ, 1],
        block_shape=dummy_block,
    )
    desc_dv = TensorDescriptor(
        dv, shape=[total_seq_len_kv, H * DimV], strides=[H * DimV, 1],
        block_shape=dummy_block,
    )
    _bwd_kwargs = dict(
        desc_q=desc_q,
        desc_k=desc_k,
        desc_v=desc_v,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        DQ=dq,
        DK=dk,
        DV=dv,
        desc_do=desc_do,
        desc_dq=desc_dq,
        desc_dk=desc_dk,
        desc_dv=desc_dv,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        max_seq_len=max_seq_len,
        attn_scale=attn_scale,
        M=M,
        Delta=Delta,
        stride_mm=stride_mm,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        AUTOTUNE_MAX_Q_LEN=autotune_max_seq_len(max_q_len),
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        DimQ=DimQ,
        DimV=DimV,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        ATTN_SCALE_TYPE=attn_scale_type,
        SOFTMAX=SOFTMAX,
    )
    if use_2kv:
        assert shared_kv, "attn_bwd_ws_2kv requires shared_kv"
        # attn_bwd_ws_2kv is shared-KV + transpose by construction (no
        # SHARED_KV/TRANSPOSE args).
        attn_bwd_ws_2kv[grid](**_bwd_kwargs)
    else:
        attn_bwd_ws[grid](**_bwd_kwargs, SHARED_KV=shared_kv)

    dq = dq.to(q.dtype)
    if shared_kv:
        dv = torch.empty(0, device=q.device, dtype=v.dtype)
    return dq, dk, dv


# ============================================================================
# attn_bwd_ws_2kv: 2-KV-block data-partitioned reduce_dq (shared-KV + transpose
# only). Ported verbatim from fbcode hammer
# (hammer/v2/ops/triton/template/tlx_bw_cross_attention.py); all helpers it uses
# already exist above. Processes 2 KV blocks (n0,n1) per program with depth-2
# qk/p/dk and a persistent dq accumulator -> two independent MMA groups (DP=2).
# ============================================================================

def _get_bw_pipeline_2kv_configs() -> List[triton.Config]:
    return [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "NUM_BUFFERS_Q": 2,
                "NUM_BUFFERS_KV": 2,
                "NUM_BUFFERS_DO": 2,
                "NUM_BUFFERS_DS": 2,
                "NUM_BUFFERS_TMEM": 1,
                "EPILOGUE_DQ_SUBTILE": 1,
                "EPILOGUE_DKV_SUBTILE": 2,
                "COMPUTE_REG": 192,
                "MMA_REG": 48,
                "LOAD_REG": 24,
            },
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_seq_parallel_pre_hook,
        )
    ]


@triton_autotune(
    configs=_get_bw_pipeline_2kv_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_Q_LEN",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "SHARED_KV",
    ],
)
@triton.jit
def attn_bwd_ws_2kv(  # noqa C901
    desc_q,
    desc_k,
    desc_v,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    desc_do,
    desc_dq,
    desc_dk,
    desc_dv,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    stride_mm,
    Z,
    AUTOTUNE_Z,
    H,
    AUTOTUNE_MAX_Q_LEN,  # Quantized MAX_Q_LEN used as an autotuning key
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    EPILOGUE_DQ_SUBTILE: tl.constexpr,
    EPILOGUE_DKV_SUBTILE: tl.constexpr,
    COMPUTE_REG: tl.constexpr,
    MMA_REG: tl.constexpr,
    LOAD_REG: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    SOFTMAX: tl.constexpr,
):
    tl.static_assert(NUM_BUFFERS_Q >= 2)
    # attn_bwd_ws_2kv is shared-KV (V aliases K) + transposed-dq only by
    # construction -- it takes no SHARED_KV/TRANSPOSE constexprs. The launcher
    # routes the non-shared-KV case to attn_bwd_ws.

    # allocate SMEM buffers and barriers
    # pyrefly: ignore [missing-attribute]
    k_tiles = tlx.local_alloc((BLOCK_N, DimQ), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    # pyrefly: ignore [missing-attribute]
    q_tiles = tlx.local_alloc((BLOCK_M, DimQ), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    # pyrefly: ignore [missing-attribute]
    do_tiles = tlx.local_alloc((BLOCK_M, DimV), tlx.dtype_of(desc_do), NUM_BUFFERS_DO)

    # Use SMEM for dqk_trans
    # pyrefly: ignore [missing-attribute]
    dqk_trans_tiles = tlx.local_alloc(
        # pyrefly: ignore [missing-attribute]
        (BLOCK_N, BLOCK_M),
        # pyrefly: ignore [missing-attribute]
        tlx.dtype_of(desc_q),
        NUM_BUFFERS_DS,
    )

    # pyrefly: ignore [missing-attribute]
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    # pyrefly: ignore [missing-attribute]
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # pyrefly: ignore [missing-attribute]
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # pyrefly: ignore [missing-attribute]
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    # pyrefly: ignore [missing-attribute]
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    # pyrefly: ignore [missing-attribute]
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    # pyrefly: ignore [missing-attribute]
    dqk_trans_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)  # depth-2 dS

    # allocate TMEM buffers and barriers
    # pyrefly: ignore [missing-attribute]
    qk_trans_tiles = tlx.local_alloc(
        (BLOCK_N, BLOCK_M),
        tl.float32,
        2,  # double-KV: depth-2 (n0, n1) so the two pT recomputes overlap
        # pyrefly: ignore [missing-attribute]
        tlx.storage_kind.tmem,
    )

    # P
    # pyrefly: ignore [missing-attribute]
    p_tiles = tlx.local_alloc(
        (BLOCK_N, BLOCK_M),
        # pyrefly: ignore [missing-attribute]
        tlx.dtype_of(desc_do),
        2,  # double-KV: matches qk_trans depth (reuse aliasing)
        # pyrefly: ignore [missing-attribute]
        tlx.storage_kind.tmem,
        reuse=qk_trans_tiles,
    )
    # Transpose-only variant: dq is stored transposed as dq^T
    # ([DimQ, BLOCK_M]) instead of [BLOCK_M, DimQ]. With BLOCK_M < DimQ this shrinks
    # its TMEM column footprint from DimQ to BLOCK_M columns (128 -> 64 at
    # BLOCK_M=64) and gives dq^T the same [*, BLOCK_M] footprint as dp so they fit
    # the 512-col budget. dq^T = (dS @ K)^T = K^T @ dS^T; the Reduce epilogue
    # transposes it back before the dQ store.
    # pyrefly: ignore [missing-attribute]
    dq_tiles = tlx.local_alloc(
        (DimQ, BLOCK_M),
        tl.float32,
        NUM_BUFFERS_TMEM,
        # pyrefly: ignore [missing-attribute]
        tlx.storage_kind.tmem,
    )

    # double-KV: dp gets its OWN TMEM tile (depth-1, option A). It can't reuse
    # dq_tiles because dq is now an n0+n1 accumulator that persists across both KV
    # sub-steps. Budget: dq(64)+dp(64)+dk(2x128)+qk(64)=448 <= 512.
    # pyrefly: ignore [missing-attribute]
    dp_tiles = tlx.local_alloc(
        (BLOCK_N, BLOCK_M),
        tl.float32,
        1,
        # pyrefly: ignore [missing-attribute]
        tlx.storage_kind.tmem,
    )
    # pyrefly: ignore [missing-attribute]
    # double-KV: TWO dk accumulators (n0, n1), accumulated across the inner M loop
    # and stored after it (dv folded in for SHARED_KV).
    dk_tiles = tlx.local_alloc(
        (BLOCK_N, DimQ),
        tl.float32,
        2,
        # pyrefly: ignore [missing-attribute]
        tlx.storage_kind.tmem,
    )
    # SHARED_KV: dv is folded into dk's TMEM (alpha pre-scaled into dqk_trans), so
    # there is no separate dv tile.
    # pyrefly: ignore [missing-attribute]
    qk_fulls = tlx.alloc_barriers(num_barriers=2)  # depth-2 (n0, n1)
    # pyrefly: ignore [missing-attribute]
    qk_empties = tlx.alloc_barriers(num_barriers=2)
    # pyrefly: ignore [missing-attribute]
    p_fulls = tlx.alloc_barriers(num_barriers=2)  # depth-2 (n0, n1)
    # pyrefly: ignore [missing-attribute]
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # dp dealiased -> own recycle barrier (Compute arrives after consuming dp).
    # pyrefly: ignore [missing-attribute]
    dp_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # pyrefly: ignore [missing-attribute]
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # pyrefly: ignore [missing-attribute]
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    # dk accumulators (n0, n1): one full/empty pair each (depth-2, indexed by nj).
    # NOT sized to NUM_BUFFERS_KV, which only buffers the K loads.
    # pyrefly: ignore [missing-attribute]
    dk_fulls = tlx.alloc_barriers(num_barriers=2)
    # pyrefly: ignore [missing-attribute]
    dk_empties = tlx.alloc_barriers(num_barriers=2)

    # pyrefly: ignore [missing-attribute]
    with tlx.async_tasks():
        # Reduce
        # pyrefly: ignore [missing-attribute]
        with tlx.async_task("default"):
            off_h, _start_n, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (
                _compute_bwd_reduce_dq_offsets(H, BLOCK_N, seq_offsets_q, seq_offsets)
            )
            if seq_len_kv > 0:
                dq_end_q = (seq_start_q + seq_len_q).to(tl.int32)
                desc_dq_t = tl.make_tensor_descriptor(
                    DQ,
                    shape=[dq_end_q, H * DimQ],
                    # pyrefly: ignore [bad-argument-type]
                    strides=[stride_dqm, 1],
                    block_shape=[BLOCK_M, BLOCK_D_Q],
                )
                dk_end_kv = (seq_start_kv + seq_len_kv).to(tl.int32)
                desc_dk_t = tl.make_tensor_descriptor(
                    DK,
                    shape=[dk_end_kv, H * DimQ],
                    # pyrefly: ignore [bad-argument-type]
                    strides=[stride_dkn, 1],
                    block_shape=[BLOCK_N, DimQ // EPILOGUE_DKV_SUBTILE],
                )
                dk_slice_size: tl.constexpr = DimQ // EPILOGUE_DKV_SUBTILE
                cnt_q = 0
                cnt_pair = 0
                for start_n0 in tl.range(0, seq_len_kv, 2 * BLOCK_N):
                    for start_m in tl.range(0, seq_len_q, BLOCK_M):
                        _, m_ph = _get_bufidx_phase(cnt_q, 1)
                        # one dq atomic per (m, pair): n0+n1 already summed in TMEM
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(dq_fulls[0], m_ph)
                        # pyrefly: ignore [missing-attribute]
                        dq_t = tlx.local_load(dq_tiles[0])
                        dq = tl.trans(dq_t)
                        desc_dq_t.store(
                            [(seq_start_q + start_m).to(tl.int32), off_h * stride_dqh],
                            dq,
                            store_reduce="add",
                        )
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_arrive(dq_empties[0])
                        cnt_q += 1
                    # store the two dk accumulators after the M loop (dv folded in).
                    # OOB n1 rows on the last pair are bounds-dropped by desc_dk_t.
                    _, pair_ph = _get_bufidx_phase(cnt_pair, 1)
                    for nj in tl.static_range(2):
                        start_n = start_n0 + nj * BLOCK_N
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(dk_fulls[nj], pair_ph)
                        for slice_id in tl.static_range(EPILOGUE_DKV_SUBTILE):
                            # pyrefly: ignore [missing-attribute]
                            dk_slice = tlx.local_slice(
                                dk_tiles[nj],
                                [0, slice_id * dk_slice_size],
                                [BLOCK_N, dk_slice_size],
                            )
                            # pyrefly: ignore [missing-attribute]
                            dk = tlx.local_load(dk_slice)
                            desc_dk_t.store(
                                [
                                    (seq_start_kv + start_n).to(tl.int32),
                                    off_h * stride_dkh + slice_id * dk_slice_size,
                                ],
                                dk.to(desc_k.dtype),
                            )
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_arrive(dk_empties[nj])
                    cnt_pair += 1

        # Compute
        # pyrefly: ignore [missing-attribute]
        with tlx.async_task(num_warps=8, registers=COMPUTE_REG):
            off_h, _start_n, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (
                _compute_bwd_reduce_dq_offsets(H, BLOCK_N, seq_offsets_q, seq_offsets)
            )
            if SOFTMAX:
                M_off = M + off_h + seq_start_q * stride_mm
                Delta_off = Delta + off_h + seq_start_q * stride_mm
                scaled_alpha = alpha * 1.44269504

            cnt_q = 0
            for start_n0 in tl.range(0, seq_len_kv, 2 * BLOCK_N):
                for start_m in tl.range(0, seq_len_q, BLOCK_M):
                    _, m_ph = _get_bufidx_phase(cnt_q, 1)
                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    mask_m = offs_m < seq_len_q
                    if SOFTMAX:
                        # pyrefly: ignore [unbound-name]
                        m = tl.load(M_off + offs_m * stride_mm, mask=mask_m)
                        # pyrefly: ignore [unbound-name]
                        Di = tl.load(Delta_off + offs_m * stride_mm, mask=mask_m)
                    for nj in tl.static_range(2):
                        start_n = start_n0 + nj * BLOCK_N
                        offs_n = start_n + tl.arange(0, BLOCK_N)
                        mask_n = offs_n < seq_len_kv
                        if not SOFTMAX:
                            if ATTN_SCALE_TYPE == "scalar":
                                scale = tl.load(attn_scale).to(tl.float32)
                            else:
                                tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
                                scale = tl.load(attn_scale + offs_n, mask=mask_n).to(
                                    tl.float32
                                )
                        _, dp_ph = _get_bufidx_phase(2 * cnt_q + nj, 1)
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(qk_fulls[nj], m_ph)
                        # pyrefly: ignore [missing-attribute]
                        qk_trans = tlx.local_load(qk_trans_tiles[nj])
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_arrive(qk_empties[nj])
                        valid_mask_trans = backward_valid_mask(
                            offs_m, offs_n, 0, seq_len_q, seq_len_kv, False
                        )
                        if SOFTMAX:
                            qkt = fast_fma(qk_trans, scaled_alpha, -m[None, :])
                            pT = tl.math.exp2(qkt)
                            pT = tl.where(valid_mask_trans, pT, 0.0)
                            # pyrefly: ignore [missing-attribute]
                            act_qk_trans = pT.to(tlx.dtype_of(desc_do))
                        else:
                            qk_trans, act_qk_trans, pT = backward_silu_activation(
                                qk_trans,
                                alpha,
                                valid_mask_trans,
                                # pyrefly: ignore [missing-attribute]
                                tlx.dtype_of(desc_do),
                                # pyrefly: ignore [unbound-name]
                                scale,
                            )
                        # pyrefly: ignore [missing-attribute]
                        tlx.local_store(p_tiles[nj], act_qk_trans)
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_arrive(p_fulls[nj])

                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(dp_fulls[0], dp_ph)
                        # pyrefly: ignore [missing-attribute]
                        dact_qk_trans = tlx.local_load(dp_tiles[0])
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_arrive(dp_empties[0])
                        if SOFTMAX:
                            # pyrefly: ignore [unbound-name]
                            dqk_trans = fast_mul(pT, dact_qk_trans - Di[None, :])
                        else:
                            dqk_trans = backward_d_silu_activation(
                                dact_qk_trans,
                                pT,
                                qk_trans,
                                # pyrefly: ignore [unbound-name]
                                scale,
                                valid_mask_trans,
                            )
                        dqk_trans = fast_mul(dqk_trans, alpha)
                        # pyrefly: ignore [missing-attribute]
                        dqk_trans = dqk_trans.to(tlx.dtype_of(desc_do))
                        # pyrefly: ignore [missing-attribute]
                        tlx.local_store(dqk_trans_tiles[nj], dqk_trans)
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_arrive(dqk_trans_fulls[nj])
                    cnt_q += 1

        # MMA
        # pyrefly: ignore [missing-attribute]
        with tlx.async_task(num_warps=1, registers=MMA_REG):
            off_h, _start_n, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (
                _compute_bwd_reduce_dq_offsets(H, BLOCK_N, seq_offsets_q, seq_offsets)
            )
            cnt_q = 0
            cnt_pair = 0
            for start_n0 in tl.range(0, seq_len_kv, 2 * BLOCK_N):
                _, pair_ph = _get_bufidx_phase(cnt_pair, 1)
                # recycle the two dk accumulators (first-pair waits are no-ops)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(dk_empties[0], pair_ph ^ 1)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(dk_empties[1], pair_ph ^ 1)
                # K[n0], K[n1] resident across the whole M loop
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(k_fulls[0], pair_ph)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(k_fulls[1], pair_ph)
                for start_m in tl.range(0, seq_len_q, BLOCK_M):
                    q_buf, q_ph = _get_bufidx_phase(cnt_q, NUM_BUFFERS_Q)
                    do_buf, do_ph = _get_bufidx_phase(cnt_q, NUM_BUFFERS_DO)
                    _, m_ph = _get_bufidx_phase(cnt_q, 1)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(q_fulls[q_buf], q_ph)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(do_fulls[do_buf], do_ph)
                    # pyrefly: ignore [missing-attribute]
                    q_trans = tlx.local_trans(q_tiles[q_buf])
                    # pyrefly: ignore [missing-attribute]
                    do_trans = tlx.local_trans(do_tiles[do_buf])
                    # FORWARD both KV blocks (qk depth-2 per nj; dp depth-1 serialized)
                    for nj in tl.static_range(2):
                        _, dp_ph = _get_bufidx_phase(2 * cnt_q + nj, 1)
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(qk_empties[nj], m_ph ^ 1)
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            k_tiles[nj],
                            q_trans,
                            qk_trans_tiles[nj],
                            use_acc=False,
                            mBarriers=[qk_fulls[nj]],
                        )
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(dp_empties[0], dp_ph ^ 1)
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            k_tiles[nj],
                            do_trans,
                            dp_tiles[0],
                            use_acc=False,
                            mBarriers=[dp_fulls[0]],
                        )
                    # BACKWARD both KV blocks (non-deferred: wait Compute's dS)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(dq_empties[0], m_ph ^ 1)
                    for nj in tl.static_range(2):
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(dqk_trans_fulls[nj], m_ph)
                        # pyrefly: ignore [missing-attribute]
                        tlx.fence_async_shared()
                        # dq^T += K[nj]^T @ dS[nj]^T (accumulate n0 then n1)
                        # pyrefly: ignore [missing-attribute]
                        k_trans = tlx.local_trans(k_tiles[nj])
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            k_trans,
                            dqk_trans_tiles[nj],
                            dq_tiles[0],
                            use_acc=(nj == 1),
                            mBarriers=[dq_fulls[0]] if nj == 1 else [],
                        )
                        # dv init/accumulate: p[nj] @ dO -> dk_tiles[nj]
                        # pyrefly: ignore [missing-attribute]
                        tlx.barrier_wait(p_fulls[nj], m_ph)
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            p_tiles[nj],
                            do_tiles[do_buf],
                            dk_tiles[nj],
                            use_acc=(start_m > 0),
                            mBarriers=[do_empties[do_buf]] if nj == 1 else [],
                        )
                        # dk += dS[nj] @ Q (Q reused across both KV blocks)
                        # pyrefly: ignore [missing-attribute]
                        tlx.async_dot(
                            dqk_trans_tiles[nj],
                            q_tiles[q_buf],
                            dk_tiles[nj],
                            use_acc=True,
                            mBarriers=[q_empties[q_buf]] if nj == 1 else [],
                        )
                    cnt_q += 1
                # publish dk[0], dk[1] and release K[n0], K[n1]
                # pyrefly: ignore [missing-attribute]
                tlx.tcgen05_commit(dk_fulls[0])
                # pyrefly: ignore [missing-attribute]
                tlx.tcgen05_commit(dk_fulls[1])
                # pyrefly: ignore [missing-attribute]
                tlx.tcgen05_commit(k_empties[0])
                # pyrefly: ignore [missing-attribute]
                tlx.tcgen05_commit(k_empties[1])
                cnt_pair += 1

        # Load
        # pyrefly: ignore [missing-attribute]
        with tlx.async_task(num_warps=1, registers=LOAD_REG):
            # initialize offsets
            off_h, _start_n, seq_start_kv, seq_len_kv, seq_start_q, seq_len_q = (
                _compute_bwd_reduce_dq_offsets(H, BLOCK_N, seq_offsets_q, seq_offsets)
            )

            accum_cnt_q = 0
            accum_cnt_pair = 0
            # OUTER: KV pairs (n0, n1). K[n0]/K[n1] live in k_tiles[0]/[1] across
            # the whole inner M loop; released (k_empties) only after it.
            for start_n0 in tl.range(0, seq_len_kv, 2 * BLOCK_N):
                start_n1 = start_n0 + BLOCK_N
                _, kpair_phase = _get_bufidx_phase(accum_cnt_pair, 1)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(k_empties[0], kpair_phase ^ 1)

                # --- PROLOGUE (m=0): staggered K[n0] -> Q[0] -> K[n1] -> dO[0] ---
                # K[n0]
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_expect_bytes(k_fulls[0], 2 * BLOCK_N * DimQ)
                # pyrefly: ignore [missing-attribute]
                tlx.async_descriptor_load(
                    desc_k,
                    k_tiles[0],
                    [(seq_start_kv + start_n0).to(tl.int32), off_h * stride_kh],
                    k_fulls[0],
                )
                # Q[0]
                q_buff_id, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(q_empties[q_buff_id], q_phase ^ 1)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_expect_bytes(q_fulls[q_buff_id], 2 * BLOCK_M * DimQ)
                # pyrefly: ignore [missing-attribute]
                tlx.async_descriptor_load(
                    desc_q,
                    q_tiles[q_buff_id],
                    [seq_start_q.to(tl.int32), off_h * stride_qh],
                    q_fulls[q_buff_id],
                )
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(k_empties[1], kpair_phase ^ 1)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_expect_bytes(k_fulls[1], 2 * BLOCK_N * DimQ)
                # pyrefly: ignore [missing-attribute]
                tlx.async_descriptor_load(
                    desc_k,
                    k_tiles[1],
                    [(seq_start_kv + start_n1).to(tl.int32), off_h * stride_kh],
                    k_fulls[1],
                )
                # dO[0]
                do_buf_id, do_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_DO)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                # pyrefly: ignore [missing-attribute]
                tlx.barrier_expect_bytes(do_fulls[do_buf_id], 2 * BLOCK_M * DimV)
                # pyrefly: ignore [missing-attribute]
                tlx.async_descriptor_load(
                    desc_do,
                    do_tiles[do_buf_id],
                    [seq_start_q.to(tl.int32), off_h * stride_doh],
                    do_fulls[do_buf_id],
                )
                accum_cnt_q += 1

                # --- Main loop (m = 1 .. M-1): only Q[m]/dO[m] ---
                for start_m in tl.range(BLOCK_M, seq_len_q, BLOCK_M):
                    q_buff_id, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    do_buf_id, do_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_DO)
                    q_offset = seq_start_q + start_m
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(q_empties[q_buff_id], q_phase ^ 1)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_expect_bytes(q_fulls[q_buff_id], 2 * BLOCK_M * DimQ)
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_buff_id],
                        [q_offset.to(tl.int32), off_h * stride_qh],
                        q_fulls[q_buff_id],
                    )
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                    # pyrefly: ignore [missing-attribute]
                    tlx.barrier_expect_bytes(do_fulls[do_buf_id], 2 * BLOCK_M * DimV)
                    # pyrefly: ignore [missing-attribute]
                    tlx.async_descriptor_load(
                        desc_do,
                        do_tiles[do_buf_id],
                        [q_offset.to(tl.int32), off_h * stride_doh],
                        do_fulls[do_buf_id],
                    )
                    accum_cnt_q += 1
                accum_cnt_pair += 1
