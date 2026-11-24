"""1-CTA warp-specialized backward of the sm100 varlen attention (ported from ads_mkl TLX JFA)."""

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.cuda.inline_ptx_lib import _mul_f32x2, _sub_f32x2
from triton.language.extra.tlx.warp_spec import get_bufidx_phase as _get_bufidx_phase

from ._sm100_common import _apply_column_range, _tile_in_band, bwd_calculate_num_steps, bwd_lookup_offsets


@triton.jit  # pragma: no cover
def bwd_calculate_offsets(
    H,
    G,
    tile_idx,
    n_tile_num,
    BROADCAST_Q: tl.constexpr,
):
    off_seq_h = tile_idx // n_tile_num
    off_z = off_seq_h // H
    if BROADCAST_Q:
        off_q_z = 0
    else:
        off_q_z = off_z
    off_h = off_seq_h % H
    off_h_kv = off_h // G
    pid = tile_idx % n_tile_num
    return off_z, off_h, off_h_kv, off_q_z, pid


@triton.jit  # pragma: no cover
def _bwd_dq_epilogue_iter(
    blk_idx,
    curr_m,
    qlen,
    begin_q,
    off_h2,
    desc_dq,
    dq_tiles,
    dq_fulls,
    dq_empties,
    dq_store_buf,
    stride_qh,
    sm_scale,
    BLOCK_M1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    DQ_REDUCE_NCOL: tl.constexpr,
    DQ_REDUCE_ITERS: tl.constexpr,
    DQ_REDUCE_STAGES: tl.constexpr,
    EARLY_RELEASE_SUBTILES: tl.constexpr,
    APPLY_M_MASK: tl.constexpr,
):
    # Reduction-warp body for one m-step of the dq epilogue. APPLY_M_MASK is
    # constexpr so the bulk-loop call (False) emits zero mask code; only the
    # peeled tail call (True) emits the `offs_m < qlen` `tl.where`.
    tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

    # wait for dq = tl.dot(tl.trans(dsT), k)
    tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
    offs_k_base = tl.arange(0, DQ_REDUCE_NCOL)
    if APPLY_M_MASK:
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m_mask_full = offs_m[:, None] < qlen
    for slice_id in tl.static_range(DQ_REDUCE_ITERS - EARLY_RELEASE_SUBTILES):
        dq_smem_idx = slice_id % DQ_REDUCE_STAGES
        dq_slice = tlx.local_slice(
            dq_tiles[tmem_buf_id],
            [0, slice_id * DQ_REDUCE_NCOL],
            [BLOCK_M1, DQ_REDUCE_NCOL],
        )
        dq = tlx.local_load(dq_slice)
        dq = dq * sm_scale
        if APPLY_M_MASK:
            qmask = (offs_k_base[None, :] < HEAD_DIM - slice_id * DQ_REDUCE_NCOL) & m_mask_full
            dq = tl.where(qmask, dq, 0.0)
        tlx.local_store(
            dq_store_buf[dq_smem_idx],
            dq.to(tlx.dtype_of(desc_dq)),
        )
        tlx.fence_async_shared()
        tlx.async_descriptor_store(
            desc_dq,
            dq_store_buf[dq_smem_idx],
            [
                (begin_q + curr_m).to(tl.int32),
                (off_h2 * stride_qh).to(tl.int32) + slice_id * DQ_REDUCE_NCOL,
            ],
            store_reduce="add",
        )
        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
    if EARLY_RELEASE_SUBTILES == 1:
        er0: tl.constexpr = DQ_REDUCE_ITERS - 1
        dq_smem_idx = er0 % DQ_REDUCE_STAGES
        dq_er0 = tlx.local_load(
            tlx.local_slice(
                dq_tiles[tmem_buf_id],
                [0, er0 * DQ_REDUCE_NCOL],
                [BLOCK_M1, DQ_REDUCE_NCOL],
            ))
        tlx.barrier_arrive(dq_empties[tmem_buf_id])
        dq_er0 = dq_er0 * sm_scale
        if APPLY_M_MASK:
            qmask_er0 = (offs_k_base[None, :] < HEAD_DIM - er0 * DQ_REDUCE_NCOL) & m_mask_full
            dq_er0 = tl.where(qmask_er0, dq_er0, 0.0)
        tlx.local_store(
            dq_store_buf[dq_smem_idx],
            dq_er0.to(tlx.dtype_of(desc_dq)),
        )
        tlx.fence_async_shared()
        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
        tlx.async_descriptor_store(
            desc_dq,
            dq_store_buf[dq_smem_idx],
            [
                (begin_q + curr_m).to(tl.int32),
                (off_h2 * stride_qh).to(tl.int32) + er0 * DQ_REDUCE_NCOL,
            ],
            store_reduce="add",
        )
    elif EARLY_RELEASE_SUBTILES == 2:
        er0: tl.constexpr = DQ_REDUCE_ITERS - 2
        dq_smem_idx_0 = er0 % DQ_REDUCE_STAGES
        er1: tl.constexpr = DQ_REDUCE_ITERS - 1
        dq_smem_idx_1 = er1 % DQ_REDUCE_STAGES
        dq_er0 = tlx.local_load(
            tlx.local_slice(
                dq_tiles[tmem_buf_id],
                [0, er0 * DQ_REDUCE_NCOL],
                [BLOCK_M1, DQ_REDUCE_NCOL],
            ))
        dq_er1 = tlx.local_load(
            tlx.local_slice(
                dq_tiles[tmem_buf_id],
                [0, er1 * DQ_REDUCE_NCOL],
                [BLOCK_M1, DQ_REDUCE_NCOL],
            ))
        tlx.barrier_arrive(dq_empties[tmem_buf_id])
        dq_er0 = dq_er0 * sm_scale
        if APPLY_M_MASK:
            qmask_er0 = (offs_k_base[None, :] < HEAD_DIM - er0 * DQ_REDUCE_NCOL) & m_mask_full
            dq_er0 = tl.where(qmask_er0, dq_er0, 0.0)
        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
        tlx.local_store(
            dq_store_buf[dq_smem_idx_0],
            dq_er0.to(tlx.dtype_of(desc_dq)),
        )
        tlx.fence_async_shared()
        tlx.async_descriptor_store(
            desc_dq,
            dq_store_buf[dq_smem_idx_0],
            [
                (begin_q + curr_m).to(tl.int32),
                (off_h2 * stride_qh).to(tl.int32) + er0 * DQ_REDUCE_NCOL,
            ],
            store_reduce="add",
        )
        dq_er1 = dq_er1 * sm_scale
        if APPLY_M_MASK:
            qmask_er1 = (offs_k_base[None, :] < HEAD_DIM - er1 * DQ_REDUCE_NCOL) & m_mask_full
            dq_er1 = tl.where(qmask_er1, dq_er1, 0.0)
        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
        tlx.local_store(
            dq_store_buf[dq_smem_idx_1],
            dq_er1.to(tlx.dtype_of(desc_dq)),
        )
        tlx.fence_async_shared()
        tlx.async_descriptor_store(
            desc_dq,
            dq_store_buf[dq_smem_idx_1],
            [
                (begin_q + curr_m).to(tl.int32),
                (off_h2 * stride_qh).to(tl.int32) + er1 * DQ_REDUCE_NCOL,
            ],
            store_reduce="add",
        )


@triton.jit  # pragma: no cover
def _bwd_softmax_iter(
    blk_idx,
    curr_m,
    qlen,
    klen,
    start_n,
    offs_n,
    M_off,
    D_off,
    qk_fulls,
    qk_tiles,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_fulls,
    dp_tiles,
    ds_tiles,
    ds_fulls,
    desc_q,
    desc_do,
    sm_scale,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    LN2: tl.constexpr,
    APPLY_M_MASK: tl.constexpr,
):
    # Compute-warp body for one m-step of the bwd softmax/dsT pipeline.
    # APPLY_M_MASK constexpr-controls whether the `offs_m < qlen` term is
    # emitted; the runtime `start_n + BLOCK_N1 >= klen` check stays inside
    # because klen is loaded per-tile (not constexpr).
    tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
    ds_buf_id, _ = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)

    offs_m = curr_m + tl.arange(0, BLOCK_M1)
    m = tl.load(M_off + offs_m, mask=offs_m < qlen, other=0.0)

    # wait for qkT = tl.dot(k, qT)
    tlx.barrier_wait(qk_fulls[tmem_buf_id], tmem_phase)
    qkT = tlx.local_load(qk_tiles[tmem_buf_id])
    tlx.barrier_arrive(qk_empties[tmem_buf_id])

    # Scale the recomputed score on the fp32 accumulator (matching the forward
    # and triton_jfa_v2) rather than pre-scaling K in bf16 on the host:
    # bf16-operand rounding of a large QK^T pushed qkT - m slightly positive on
    # the row-max column -> exp2 overflow -> +inf -> NaN grads at high activation
    # magnitude. tl.minimum bounds the argument for masked/padding columns
    # (whose stored LSE is 0). Both are fp32 register ops -- the tl.dot operands
    # stay bf16, so no slowdown.
    qkT = qkT * (sm_scale / LN2)
    pT = tl.math.exp2(tl.minimum(_sub_f32x2(qkT, m[None, :]), 0.0))
    HAS_BAND: tl.constexpr = WINDOW_LEFT is not None or WINDOW_RIGHT is not None
    diag = klen - qlen

    # Key row r of pT keeps query columns [r + lo0, min(r + hi0, cap)).
    lo0 = -BLOCK_N1
    hi0 = BLOCK_M1
    if WINDOW_RIGHT is not None:
        lo0 = start_n - diag - WINDOW_RIGHT - curr_m
    if WINDOW_LEFT is not None:
        hi0 = start_n - diag + WINDOW_LEFT + 1 - curr_m
    if APPLY_M_MASK:
        # Tail (last m-step): the `offs_m < qlen` term always applies.
        pT = tl.where(offs_n[:, None] < klen, pT, 0.0)
        pT = _apply_column_range(pT, lo0, hi0, qlen - curr_m, 0.0, BLOCK_N1, BLOCK_M1)
    else:
        # Bulk (non-last m-step). The n-mask only on the boundary K tile; the
        # band mask only when the tile crosses a band edge.
        if start_n + BLOCK_N1 >= klen:
            pT = tl.where(offs_n[:, None] < klen, pT, 0.0)
            if HAS_BAND:
                pT = _apply_column_range(pT, lo0, hi0, BLOCK_M1, 0.0, BLOCK_N1, BLOCK_M1)
        elif HAS_BAND:
            if not _tile_in_band(curr_m, start_n, BLOCK_M1, BLOCK_N1, diag, WINDOW_LEFT, WINDOW_RIGHT):
                pT = _apply_column_range(pT, lo0, hi0, BLOCK_M1, 0.0, BLOCK_N1, BLOCK_M1)

    # ppT *= qk_scale
    ppT = pT
    ppT = ppT.to(tlx.dtype_of(desc_do))
    tlx.local_store(p_tiles[tmem_buf_id], ppT)
    tlx.barrier_arrive(p_fulls[tmem_buf_id])

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D_off + offs_m, mask=offs_m < qlen, other=0.0)

    # Wait for dpT = tl.dot(v, tl.trans(do))
    tlx.barrier_wait(dp_fulls[tmem_buf_id], tmem_phase)
    dpT = tlx.local_load(dp_tiles[tmem_buf_id])
    # No need to release dP, as dP uses the same tmem as dQ
    # in the same iteration. Release dQ instead later.
    dsT = _mul_f32x2(pT, _sub_f32x2(dpT, Di[None, :]))
    dsT = dsT.to(tlx.dtype_of(desc_q))
    tlx.local_store(ds_tiles[ds_buf_id], dsT)
    tlx.fence_async_shared()
    tlx.barrier_arrive(ds_fulls[ds_buf_id])


_BWD_CONFIG = dict(BLOCK_M1=128, BLOCK_N1=128, NUM_BUFFERS_KV=1, NUM_BUFFERS_Q=2, NUM_BUFFERS_DO=1, NUM_BUFFERS_DS=1,
                   NUM_BUFFERS_TMEM=1, EPILOGUE_SUBTILE=4, ENABLE_CLC=True, EARLY_RELEASE_SUBTILES=2, num_warps=4,
                   num_stages=1)


@triton.jit  # pragma: no cover
def _attn_bwd_ws(  # noqa: C901, TR001
    desc_q,
    Q_offsets: tl.tensor,
    desc_k,
    K_offsets: tl.tensor,
    desc_v,
    desc_do,  #
    Out_offsets: tl.tensor,
    desc_dq,
    dQ: tl.tensor,
    dK: tl.tensor,
    dV: tl.tensor,  #
    sm_scale: float,  #
    M: tl.tensor,
    D: tl.tensor,
    stride_km: int,
    stride_qh: int,
    stride_kh: int,
    stride_mh: int,
    stride_d: int,
    Z,
    H: int,
    G: int,
    N_CTX_KV,
    tile_to_batch: tl.tensor,
    tile_to_head: tl.tensor,
    tile_to_block: tl.tensor,
    total_valid_tiles: int,
    total_len_q,
    total_len_kv,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    EARLY_RELEASE_SUBTILES: tl.constexpr,  # How many subtiles to pre-load before releasing TMEM (1 or 2)
    BROADCAST_Q: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    USE_I64_IDX: tl.constexpr,
    ENABLE_CLC: tl.constexpr,
    ENABLE_LB: tl.constexpr,
    USE_ON_DEVICE_TMA: tl.constexpr = False,
) -> None:
    tl.static_assert(
        EARLY_RELEASE_SUBTILES >= 1 and EARLY_RELEASE_SUBTILES <= 2,
        "EARLY_RELEASE_SUBTILES must be 1 or 2",
    )
    if USE_ON_DEVICE_TMA:
        # The desc_* arguments are raw (tokens, H * HEAD_DIM) tensors; building the descriptors here saves
        # five host-side constructions per call.
        desc_q = tl.make_tensor_descriptor(desc_q, shape=[total_len_q, HEAD_DIM * H], strides=[HEAD_DIM * H, 1],
                                           block_shape=[BLOCK_M1, BLOCK_D])
        desc_do = tl.make_tensor_descriptor(desc_do, shape=[total_len_q, HEAD_DIM * H], strides=[HEAD_DIM * H, 1],
                                            block_shape=[BLOCK_M1, BLOCK_D])
        desc_k = tl.make_tensor_descriptor(desc_k, shape=[total_len_kv, HEAD_DIM * H], strides=[HEAD_DIM * H, 1],
                                           block_shape=[BLOCK_N1, BLOCK_D])
        desc_v = tl.make_tensor_descriptor(desc_v, shape=[total_len_kv, HEAD_DIM * H], strides=[HEAD_DIM * H, 1],
                                           block_shape=[BLOCK_N1, BLOCK_D])

    n_tile_num = tl.cdiv(N_CTX_KV, BLOCK_N1)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    if USE_I64_IDX:
        prog_id = prog_id.to(tl.int64)
        num_progs = num_progs.to(tl.int64)

    if ENABLE_LB:
        total_tiles = total_valid_tiles
    else:
        total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    # allocate smem buffers
    k_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_D), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    v_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_D), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)
    q_tiles = tlx.local_alloc((BLOCK_M1, BLOCK_D), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    do_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_DO)

    # Use SMEM for dsT
    ds_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tlx.dtype_of(desc_q), NUM_BUFFERS_DS)

    # SMEM staging buffer for async TMA store of dK/dV epilogue
    slice_size_alloc: tl.constexpr = BLOCK_D // EPILOGUE_SUBTILE
    dkv_store_buf = tlx.local_alloc((BLOCK_N1, slice_size_alloc), tlx.dtype_of(desc_k), 1)

    # SMEM staging buffer for async TMA reduce-add of dQ (double-buffered).
    # Uses smaller column width (DQ_REDUCE_NCOL) than dK/dV to fit in SMEM.
    DQ_REDUCE_NCOL: tl.constexpr = BLOCK_D // (EPILOGUE_SUBTILE * 2)  # 16 cols
    DQ_REDUCE_STAGES: tl.constexpr = 2
    DQ_REDUCE_ITERS: tl.constexpr = BLOCK_D // DQ_REDUCE_NCOL  # 8 iters
    dq_store_buf = tlx.local_alloc((BLOCK_M1, DQ_REDUCE_NCOL), tlx.dtype_of(desc_q), DQ_REDUCE_STAGES)

    # allocate barriers for smem buffers
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    ds_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)

    # allocate tmem buffers
    qk_p_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    qk_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_p_storage_alias,
    )
    p_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_do),
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_p_storage_alias,
    )
    qk_p_storage_alias.set_buffer_overlap(tlx.reuse_group(qk_tiles, p_tiles, group_type=tlx.reuse_group_type.shared))
    dp_dq_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    dp_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=dp_dq_storage_alias,
    )

    dq_tiles = tlx.local_alloc(
        (BLOCK_M1, BLOCK_D),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=dp_dq_storage_alias,
    )
    dp_dq_storage_alias.set_buffer_overlap(tlx.reuse_group(dp_tiles, dq_tiles, group_type=tlx.reuse_group_type.shared))
    dv_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_D), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem)
    dk_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_D), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem)

    # allocate barriers for tmem buffers
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    if ENABLE_CLC:
        clc_context = tlx.clc_create_context(4)

    with tlx.async_tasks():
        # reduction
        with tlx.async_task(num_warps=4, registers=64):
            if USE_ON_DEVICE_TMA:
                # Built inside its only user: a narrow descriptor passed into the warp-specialized region
                # loses its shared-memory layout at the region boundary.
                desc_dq = tl.make_tensor_descriptor(desc_dq, shape=[total_len_q,
                                                                    HEAD_DIM * H], strides=[HEAD_DIM * H, 1],
                                                    block_shape=[BLOCK_M1, BLOCK_D // (EPILOGUE_SUBTILE * 2)])
            blk_idx = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            i = 0
            has_more_tile = True
            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = (bwd_lookup_offsets(
                    tile_to_batch,
                    tile_to_head,
                    tile_to_block,
                    G,
                    tile_idx,
                    BROADCAST_Q,
                ) if ENABLE_LB else bwd_calculate_offsets(
                    H,
                    G,
                    tile_idx,
                    n_tile_num,
                    BROADCAST_Q,
                ))
                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                qlen = end_q - begin_q
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k
                start_n = pid * BLOCK_N1
                off_h2 = off_h.to(tl.int64)

                if start_n < klen:
                    num_steps, start_m = bwd_calculate_num_steps(qlen, klen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_LEFT,
                                                                 WINDOW_RIGHT)
                    curr_m = start_m
                    step_m = BLOCK_M1
                    # Loop peeling: bulk pass (num_steps - 1 iters, mask code
                    # dead-stripped via APPLY_M_MASK=False constexpr) + tail
                    # (1 iter, mask code emitted). Only the last m-step can
                    # have offs_m >= qlen.
                    for _ in range(num_steps - 1):
                        _bwd_dq_epilogue_iter(
                            blk_idx,
                            curr_m,
                            qlen,
                            begin_q,
                            off_h2,
                            desc_dq,
                            dq_tiles,
                            dq_fulls,
                            dq_empties,
                            dq_store_buf,
                            stride_qh,
                            sm_scale,
                            BLOCK_M1=BLOCK_M1,
                            HEAD_DIM=HEAD_DIM,
                            NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                            DQ_REDUCE_NCOL=DQ_REDUCE_NCOL,
                            DQ_REDUCE_ITERS=DQ_REDUCE_ITERS,
                            DQ_REDUCE_STAGES=DQ_REDUCE_STAGES,
                            EARLY_RELEASE_SUBTILES=EARLY_RELEASE_SUBTILES,
                            APPLY_M_MASK=False,
                        )
                        curr_m += step_m
                        blk_idx += 1
                    if num_steps > 0:
                        _bwd_dq_epilogue_iter(
                            blk_idx,
                            curr_m,
                            qlen,
                            begin_q,
                            off_h2,
                            desc_dq,
                            dq_tiles,
                            dq_fulls,
                            dq_empties,
                            dq_store_buf,
                            stride_qh,
                            sm_scale,
                            BLOCK_M1=BLOCK_M1,
                            HEAD_DIM=HEAD_DIM,
                            NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                            DQ_REDUCE_NCOL=DQ_REDUCE_NCOL,
                            DQ_REDUCE_ITERS=DQ_REDUCE_ITERS,
                            DQ_REDUCE_STAGES=DQ_REDUCE_STAGES,
                            EARLY_RELEASE_SUBTILES=EARLY_RELEASE_SUBTILES,
                            APPLY_M_MASK=True,
                        )
                        curr_m += step_m
                        blk_idx += 1
                if ENABLE_CLC:
                    tlx.clc_producer(clc_context, clc_phase_producer)
                    clc_phase_producer = clc_phase_producer ^ 1
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(tl.int64)
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    i += 1
                    tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm

        # compute
        with tlx.async_task("default", replicate=1):
            blk_idx = 0
            kv_tile_idx = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            i = 0
            has_more_tile = True
            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = (bwd_lookup_offsets(
                    tile_to_batch,
                    tile_to_head,
                    tile_to_block,
                    G,
                    tile_idx,
                    BROADCAST_Q,
                ) if ENABLE_LB else bwd_calculate_offsets(
                    H,
                    G,
                    tile_idx,
                    n_tile_num,
                    BROADCAST_Q,
                ))
                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                qlen = end_q - begin_q
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k
                if not BROADCAST_Q:
                    begin_o = begin_q
                else:
                    begin_o = qlen * off_z
                start_n = pid * BLOCK_N1
                off_h2 = off_h.to(tl.int64)
                off_chz = off_h2 * stride_mh + begin_o
                kv_offset = off_h_kv.to(tl.int64) * stride_kh

                if start_n < klen:
                    # offset pointers for batch/head
                    M_off = M + off_chz
                    D_off = D + off_chz
                    num_steps, start_m = bwd_calculate_num_steps(qlen, klen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_LEFT,
                                                                 WINDOW_RIGHT)
                    curr_m = start_m
                    step_m = BLOCK_M1
                    offs_n = start_n + tl.arange(0, BLOCK_N1)
                    # Loop peeling: bulk pass (num_steps - 1 iters, no
                    # offs_m < qlen mask code emitted) + tail (1 iter with
                    # mask). The runtime `start_n + BLOCK_N1 >= klen` check
                    # stays inside the helper since klen is loaded per-tile.
                    for _ in range(num_steps - 1):
                        _bwd_softmax_iter(
                            blk_idx,
                            curr_m,
                            qlen,
                            klen,
                            start_n,
                            offs_n,
                            M_off,
                            D_off,
                            qk_fulls,
                            qk_tiles,
                            qk_empties,
                            p_tiles,
                            p_fulls,
                            dp_fulls,
                            dp_tiles,
                            ds_tiles,
                            ds_fulls,
                            desc_q,
                            desc_do,
                            sm_scale,
                            BLOCK_M1=BLOCK_M1,
                            BLOCK_N1=BLOCK_N1,
                            NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                            NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                            WINDOW_LEFT=WINDOW_LEFT,
                            WINDOW_RIGHT=WINDOW_RIGHT,
                            LN2=LN2,
                            APPLY_M_MASK=False,
                        )
                        curr_m += step_m
                        blk_idx += 1
                    if num_steps > 0:
                        _bwd_softmax_iter(
                            blk_idx,
                            curr_m,
                            qlen,
                            klen,
                            start_n,
                            offs_n,
                            M_off,
                            D_off,
                            qk_fulls,
                            qk_tiles,
                            qk_empties,
                            p_tiles,
                            p_fulls,
                            dp_fulls,
                            dp_tiles,
                            ds_tiles,
                            ds_fulls,
                            desc_q,
                            desc_do,
                            sm_scale,
                            BLOCK_M1=BLOCK_M1,
                            BLOCK_N1=BLOCK_N1,
                            NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                            NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                            WINDOW_LEFT=WINDOW_LEFT,
                            WINDOW_RIGHT=WINDOW_RIGHT,
                            LN2=LN2,
                            APPLY_M_MASK=True,
                        )
                        curr_m += step_m
                        blk_idx += 1

                    # num_steps <= 0: no query block reaches this K tile (an empty query
                    # sequence, or every query row outside the window), so the MMA and
                    # load tasks skip it. Tested here, not in the tile guard: a guard
                    # implying num_steps > 0 makes the tail check above unconditional
                    # and ptxas then spills in the softmax loop.
                    if num_steps > 0:
                        # epilogue - async TMA store for dK/dV
                        kv_buf_id, kv_phase = _get_bufidx_phase(kv_tile_idx, NUM_BUFFERS_KV)

                        tlx.barrier_wait(dv_fulls[kv_buf_id], kv_phase)
                        slice_size: tl.constexpr = BLOCK_D // EPILOGUE_SUBTILE
                        desc_dv = tl.make_tensor_descriptor(
                            dV,
                            shape=[end_k.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_N1, BLOCK_D // EPILOGUE_SUBTILE],
                        )
                        desc_dk = tl.make_tensor_descriptor(
                            dK,
                            shape=[end_k.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_N1, BLOCK_D // EPILOGUE_SUBTILE],
                        )
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            dv_slice = tlx.local_slice(
                                dv_tiles[kv_buf_id],
                                [0, slice_id * slice_size],
                                [BLOCK_N1, slice_size],
                            )
                            dv = tlx.local_load(dv_slice)
                            tlx.local_store(dkv_store_buf[0], dv.to(tlx.dtype_of(desc_dv)))
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dv,
                                dkv_store_buf[0],
                                [
                                    (begin_k + start_n).to(tl.int32),
                                    (kv_offset + slice_id * slice_size).to(tl.int32),
                                ],
                            )
                            tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(dv_empties[kv_buf_id])

                        tlx.barrier_wait(dk_fulls[kv_buf_id], kv_phase)
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            dk_slice = tlx.local_slice(
                                dk_tiles[kv_buf_id],
                                [0, slice_id * slice_size],
                                [BLOCK_N1, slice_size],
                            )
                            dk = tlx.local_load(dk_slice)
                            dk *= sm_scale
                            tlx.local_store(dkv_store_buf[0], dk.to(tlx.dtype_of(desc_dk)))
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dk,
                                dkv_store_buf[0],
                                [
                                    (begin_k + start_n).to(tl.int32),
                                    (kv_offset + slice_id * slice_size).to(tl.int32),
                                ],
                            )
                            tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(dk_empties[kv_buf_id])
                        kv_tile_idx += 1
                    else:
                        # dK/dV of the tile are zero: store them without the dK/dV
                        # barriers or kv_tile_idx, which only count tiles the MMA runs.
                        zero_ncol: tl.constexpr = BLOCK_D // EPILOGUE_SUBTILE
                        zero_rows = (begin_k + offs_n).to(tl.int64) * (HEAD_DIM * H)
                        zero_row_mask = offs_n[:, None] < klen
                        zeros = tl.zeros([BLOCK_N1, zero_ncol], dK.dtype.element_ty)
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            offs_d = slice_id * zero_ncol + tl.arange(0, zero_ncol)
                            zero_offs = zero_rows[:, None] + kv_offset + offs_d[None, :]
                            zero_mask = zero_row_mask & (offs_d[None, :] < HEAD_DIM)
                            tl.store(dV + zero_offs, zeros, mask=zero_mask)
                            tl.store(dK + zero_offs, zeros, mask=zero_mask)
                if ENABLE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(tl.int64)
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    i += 1
                    tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm

        # mma
        with tlx.async_task(num_warps=1, registers=104):
            blk_idx = 0
            kv_tile_idx = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            i = 0
            has_more_tile = True
            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = (bwd_lookup_offsets(
                    tile_to_batch,
                    tile_to_head,
                    tile_to_block,
                    G,
                    tile_idx,
                    BROADCAST_Q,
                ) if ENABLE_LB else bwd_calculate_offsets(
                    H,
                    G,
                    tile_idx,
                    n_tile_num,
                    BROADCAST_Q,
                ))

                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                qlen = end_q - begin_q
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k
                start_n = pid * BLOCK_N1
                off_h2 = off_h.to(tl.int64)

                num_steps, start_m = bwd_calculate_num_steps(qlen, klen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_LEFT,
                                                             WINDOW_RIGHT)
                if start_n < klen and num_steps > 0:
                    kv_buf_id, kv_phase = _get_bufidx_phase(kv_tile_idx, NUM_BUFFERS_KV)
                    tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
                    tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)

                    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
                    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

                    # -----------------------------------------------------------
                    # Prolog
                    #
                    # 1. qkT = tl.dot(k, qT)
                    # 2. dpT = tl.dot(v, tl.trans(do))
                    # 3. dv += tl.dot(ppT, do)
                    # -----------------------------------------------------------

                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

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
                    # As dP uses the same tmem as dQ, wait for dQ release.
                    tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
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
                        q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                        tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
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
                        q_buf_id_prev, _ = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                        tmem_buf_id_prev, tmem_phase_prev = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
                        ds_buf_id_prev, ds_phase_prev = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

                        # Compute dq = tl.dot(tl.trans(dsT), k) from previous iteration
                        tlx.barrier_wait(ds_fulls[ds_buf_id_prev], ds_phase_prev)
                        tlx.barrier_wait(dq_empties[tmem_buf_id_prev], tmem_phase_prev ^ 1)
                        dsT_view = tlx.local_trans(ds_tiles[ds_buf_id_prev])
                        tlx.async_dot(
                            dsT_view,
                            k_tiles[kv_buf_id],
                            dq_tiles[tmem_buf_id_prev],
                            use_acc=False,
                            mBarriers=[dq_fulls[tmem_buf_id_prev]],
                        )

                        # Compute dk += tl.dot(dsT, tl.trans(qT)) from previous iteration
                        tlx.async_dot(
                            ds_tiles[ds_buf_id_prev],
                            q_tiles[q_buf_id_prev],
                            dk_tiles[kv_buf_id],
                            use_acc=(j - 1) > 0,
                            force_async=True,
                            mBarriers=[q_empties[q_buf_id_prev]],
                        )

                        do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                        # Compute dpT = tl.dot(v, tl.trans(do))
                        tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                        # As dP uses the same tmem as dQ, wait for dQ release.
                        tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
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
                            force_async=True,
                            mBarriers=[do_empties[do_buf_id]],
                        )
                        blk_idx += 1

                    tlx.tcgen05_commit(dv_fulls[kv_buf_id])
                    # tlx.tcgen05_commit(v_empties[kv_buf_id])

                    # -----------------------------------------------------------
                    # Epilog
                    # 4. dk += tl.dot(dsT, tl.trans(qT))
                    # 5. dq = tl.dot(tl.trans(dsT), k)
                    # -----------------------------------------------------------
                    prev_blk_idx = blk_idx - 1
                    q_buf_id, _ = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
                    ds_buf_id, ds_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)
                    # Compute dk += tl.dot(dsT, tl.trans(qT))
                    tlx.barrier_wait(ds_fulls[ds_buf_id], ds_phase)
                    tlx.async_dot(ds_tiles[ds_buf_id], q_tiles[q_buf_id], dk_tiles[kv_buf_id], use_acc=num_steps > 1,
                                  mBarriers=[
                                      q_empties[q_buf_id],
                                      dk_fulls[kv_buf_id],
                                  ],  # could be a issue
                                  )

                    # Compute dq = tl.dot(tl.trans(dsT), k)
                    tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
                    dsT_view = tlx.local_trans(ds_tiles[ds_buf_id])
                    tlx.async_dot(
                        dsT_view,
                        k_tiles[kv_buf_id],
                        dq_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[dq_fulls[tmem_buf_id]],
                    )
                    tlx.tcgen05_commit(k_empties[kv_buf_id])
                    kv_tile_idx += 1
                if ENABLE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(tl.int64)
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    i += 1
                    tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm
        # load
        with tlx.async_task(num_warps=1, registers=64):
            blk_idx = 0
            kv_tile_idx = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            i = 0
            has_more_tile = True
            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = (bwd_lookup_offsets(
                    tile_to_batch,
                    tile_to_head,
                    tile_to_block,
                    G,
                    tile_idx,
                    BROADCAST_Q,
                ) if ENABLE_LB else bwd_calculate_offsets(
                    H,
                    G,
                    tile_idx,
                    n_tile_num,
                    BROADCAST_Q,
                ))

                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                qlen = end_q - begin_q
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k
                start_n = pid * BLOCK_N1
                off_h2 = off_h.to(tl.int64)
                num_steps, start_m = bwd_calculate_num_steps(qlen, klen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_LEFT,
                                                             WINDOW_RIGHT)
                if start_n < klen and num_steps > 0:
                    curr_m = start_m
                    step_m = BLOCK_M1
                    # Load K
                    kv_buf_id, kv_phase = _get_bufidx_phase(kv_tile_idx, NUM_BUFFERS_KV)
                    tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                    tlx.barrier_expect_bytes(k_fulls[kv_buf_id], 2 * BLOCK_N1 * HEAD_DIM)  # float16
                    tlx.async_descriptor_load(
                        desc_k,
                        k_tiles[kv_buf_id],
                        [
                            (begin_k + start_n).to(tl.int32),
                            (off_h_kv * stride_kh).to(tl.int32),
                        ],
                        k_fulls[kv_buf_id],
                    )

                    # Load first Q
                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                    tlx.barrier_expect_bytes(q_fulls[q_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_buf_id],
                        [
                            (begin_q + curr_m).to(tl.int32),
                            (off_h2 * stride_qh).to(tl.int32),
                        ],
                        q_fulls[q_buf_id],
                    )

                    # Load V
                    # tlx.barrier_wait(v_empties[kv_buf_id], kv_phase ^ 1)
                    tlx.barrier_expect_bytes(v_fulls[kv_buf_id], 2 * BLOCK_N1 * HEAD_DIM)  # float16
                    tlx.async_descriptor_load(
                        desc_v,
                        v_tiles[kv_buf_id],
                        [
                            (begin_k + start_n).to(tl.int32),
                            (off_h_kv * stride_kh).to(tl.int32),
                        ],
                        v_fulls[kv_buf_id],
                    )

                    # Load first dO
                    if not BROADCAST_Q:
                        begin_o = begin_q
                    else:
                        begin_o = qlen * off_z
                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                    tlx.barrier_expect_bytes(do_fulls[do_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                    tlx.async_descriptor_load(
                        desc_do,
                        do_tiles[do_buf_id],
                        [
                            (begin_o + curr_m).to(tl.int32),
                            (off_h2 * stride_qh).to(tl.int32),
                        ],
                        do_fulls[do_buf_id],
                    )
                    curr_m += step_m
                    blk_idx += 1

                    for _ in range(1, num_steps):
                        q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                        do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                        # Load Q
                        tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                        tlx.barrier_expect_bytes(q_fulls[q_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                        tlx.async_descriptor_load(
                            desc_q,
                            q_tiles[q_buf_id],
                            [
                                (begin_q + curr_m).to(tl.int32),
                                (off_h2 * stride_qh).to(tl.int32),
                            ],
                            q_fulls[q_buf_id],
                        )

                        # Load dO
                        tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                        tlx.barrier_expect_bytes(do_fulls[do_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                        tlx.async_descriptor_load(
                            desc_do,
                            do_tiles[do_buf_id],
                            [
                                (begin_o + curr_m).to(tl.int32),
                                (off_h2 * stride_qh).to(tl.int32),
                            ],
                            do_fulls[do_buf_id],
                        )
                        curr_m += step_m
                        blk_idx += 1
                    kv_tile_idx += 1
                if ENABLE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(tl.int64)
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    i += 1
                    tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm
