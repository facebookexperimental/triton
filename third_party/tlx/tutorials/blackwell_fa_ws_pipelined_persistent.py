import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.language.extra.cuda.inline_ptx_lib import _mul_f32x2, _fma_f32x2, _sub_f32x2
from triton.language.extra.subtile_ops import _join_n_2D, _split_n_2D
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
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
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    )
    for kv in [3, 6]
    for grp_n in [1, 4]
    for (rescale_opt, where) in [(False, False), (True, False), (True, True)]
    for uwb in [False, True]
]


def prune_configs_by_hdim(configs, named_args, **kwargs):
    HEAD_DIM = kwargs["HEAD_DIM"]
    STAGE = kwargs["STAGE"]
    target_kv_buffers = 6 if HEAD_DIM == 64 else 3
    target_group_size_n = 4 if STAGE == 3 else 1
    return [
        conf for conf in configs if conf.kwargs.get("NUM_BUFFERS_KV", 0) == target_kv_buffers
        and conf.kwargs.get("GROUP_SIZE_N", 0) == target_group_size_n
    ]


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
):
    group_id = tile_idx // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
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
def _softmax_inner_loop(
    qk_fulls,
    qk_tiles,
    p_fulls,
    p_tiles,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    cid,
    accum_cnt_qk,
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
):
    lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)

    for start_n in tl.range(lo, hi, BLOCK_N):
        _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)
        tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
        qk = tlx.local_load(tlx.local_view(qk_tiles, cid))

        if STAGE == 2:
            col_limit_right = (offs_m - start_n + 1)[:, None]
            qk = _apply_causal_mask(qk, col_limit_right, BLOCK_N)

        # compute m_i, p in registers
        # update_row_max: row_max_new = _compute_row_max(qk, row_max[0])
        # -> FA4 handles one row per thread (32 threads per warp * 4)
        # -> use fmax_reduce(one row of qk, m_i[0])
        # -> m_i|m_ij = row_max[0] * scale
        if RESCALE_OPT:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

        # -- compute correction factor
        # update_row_max: acc_scale_ = (row_max[0] - row_max_new) * scale
        # -> acc_scale = exp2(acc_scale_)
        # -> if (acc_scale_ >= -8.0):
        # ->   row_max_new = row_max[0]; acc_scale = 1.0
        # -> row_max[0] = row_max_new
        if RESCALE_OPT:
            alpha_ = (m_i - m_ij) * qk_scale  # alpha_ is 1D distributed over the warp group
            alpha = tl.math.exp2(alpha_)
            rescale_mask = alpha_ >= -8.0
            alpha = tl.where(rescale_mask, 1.0, alpha)
            m_ij = tl.where(rescale_mask, m_i, m_ij)
        else:
            alpha = tl.math.exp2(m_i - m_ij)
        tlx.barrier_wait(tlx.local_view(alpha_empties, cid), qk_phase ^ 1)
        tlx.local_store(tlx.local_view(alpha_tiles, cid), tl.join(alpha, alpha) if SCALAR_N == 2 else alpha[:, None])
        tlx.barrier_arrive(tlx.local_view(alpha_fulls, cid))

        # scale_subtract_rowmax:
        # -> row_max_scaled = row_max_new * scale
        # -> s[i], s[i+1] = fma_packed_f32x2((s[i], s[i+1]), (scale, scale), (-row_max_scaled, -row_max_scaled))
        if RESCALE_OPT:
            m_scaled = m_ij * qk_scale
            qk = _fma_f32x2(qk, qk_scale, -m_scaled[:, None])
        else:
            qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        # apply_epx2_convert in FA4:
        # 128 elements per row is divided into 4 fragments, first fragment covers [0] to [31]
        # for last fragment, always use SFU, for first 3 fragments, elements 0 to 11 use SFU,
        # elements 12 to 15 use emulation, elements 16 to 27 use SFU, elements 28 to 31 use emulation
        # the loop is unrolled twice likely for vectorization
        qks = _split_n_2D(qk, NUM_MMA_SLICES)
        ps = ()
        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
            # prepare p for the v dot
            p_bufIdx = cid * NUM_MMA_SLICES + slice_id
            p_i = tl.math.exp2(qks[slice_id])
            tlx.local_store(tlx.local_view(p_tiles, p_bufIdx), p_i.to(out_dtype))
            tlx.barrier_arrive(tlx.local_view(p_fulls, p_bufIdx))
            ps = ps + (p_i, )

        p = _join_n_2D(ps)
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        accum_cnt_qk += 1

    return m_i, l_i, accum_cnt_qk


@triton.autotune(
    configs=configs,
    key=["N_CTX", "HEAD_DIM", "STAGE"],
    prune_configs_by={"early_config_prune": prune_configs_by_hdim},
)
@triton.jit
def _attn_fwd_ws(sm_scale, M,  #
                 Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
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
                 ):
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(NUM_BUFFERS_QK == 1)
    tl.static_assert(NUM_BUFFERS_Q == 1)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2

    # Compute bytes per element for each tensor type
    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))
    qk_dtype = tl.float32

    # original grid
    #   triton.cdiv(q.shape[2], META["BLOCK_M"]),
    #   q.shape[0] * q.shape[1],
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = Z * H
    num_pid_in_group = num_pid_m * GROUP_SIZE_N

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    o_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_o), NUM_MMA_GROUPS)

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    o_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    # Define the buffer for sharing. Offsets are currently manually specified
    # via buffer count.
    qk_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    qk_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_N), qk_dtype, NUM_MMA_GROUPS, tlx.storage_kind.tmem,
                               reuse=qk_storage_alias)
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N // NUM_MMA_SLICES),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * NUM_MMA_SLICES,
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
                tlx.reuse_group(p_tiles, group_size=NUM_MMA_SLICES),
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

    if USE_WARP_BARRIER:
        qk_empties = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
        p_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS * NUM_MMA_SLICES, num_warps=4)
        acc_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
        alpha_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
        alpha_empties = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
        l_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
        o_fulls = tlx.alloc_warp_barrier(num_barriers=NUM_MMA_GROUPS, num_warps=4)
    else:
        qk_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
        p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_MMA_SLICES)
        acc_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
        alpha_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
        alpha_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
        l_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
        o_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    # 6 consumers: correction(1) + softmax(2 replicas) + mma(1) + load(1) + epilog(1)
    clc_context = tlx.clc_create_context(num_consumers=6)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            accum_cnt = 0
            phase = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_producer = 1
            clc_phase_consumer = 0
            while tile_id != -1:
                # CLC producer: announce work to all consumer tasks
                tlx.clc_producer(clc_context, clc_phase_producer)
                clc_phase_producer ^= 1

                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )
                for _ in tl.range(lo, hi, BLOCK_N):
                    _, phase = get_bufidx_phase(accum_cnt, 1)
                    for cid in tl.static_range(0, NUM_MMA_GROUPS):
                        # -- update output accumulator --
                        tlx.barrier_wait(alpha_fulls[cid], phase)
                        alpha_loaded = tlx.local_load(alpha_tiles[cid])
                        alpha_1 = tl.split(alpha_loaded)[0][:, None] if SCALAR_N == 2 else alpha_loaded
                        tlx.barrier_arrive(alpha_empties[cid])
                        # Perform warp-level ballot vote to check if any thread needs rescaling
                        # 0xFFFFFFFF means all 32 threads in the warp participate
                        if RESCALE_OPT:
                            pred = alpha_1 < 1.0
                            # ballot_result is a tensor with the same shape as pred
                            # All elements contain the same warp-level ballot value
                            # Non-zero means at least one thread has alpha_1 < 1.0
                            ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
                            should_rescale = ballot_result != 0

                        # FA4: each thread handles one row, 128 elements
                        #   128 threads handle 128 rows
                        #   each thread breaks one row into 8 fragments, each fragment 16 elements, unrolls by 2
                        # TLX: with NUM_MMA_SLICES of 2, we handle 128x64, then another 128x64
                        # Since Triton doesn't support ifOp on a tensor value, we try to combine the values
                        # option 1: use tl.where
                        if USE_WHERE:
                            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                                subslice = tlx.subslice(
                                    acc_tiles[cid],
                                    HEAD_DIM * slice_id // NUM_MMA_SLICES,
                                    HEAD_DIM // NUM_MMA_SLICES,
                                )
                                acc = tlx.local_load(subslice)
                                # Use tl.where to conditionally apply rescaling
                                # acc = acc * alpha_1 where should_rescale, else acc unchanged
                                if RESCALE_OPT:
                                    scaled_acc = _mul_f32x2(acc, alpha_1)
                                    acc = tl.where(should_rescale, scaled_acc, acc)
                                else:
                                    acc = _mul_f32x2(acc, alpha_1)
                                tlx.local_store(subslice, acc)
                        else:
                            # option 2: use a single scalar IfOp
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
                        tlx.barrier_arrive(acc_fulls[cid])
                    accum_cnt += 1

                _, phase = get_bufidx_phase(tile_count, 1)
                for cid in tl.static_range(0, NUM_MMA_GROUPS):
                    # epilogue
                    tlx.barrier_wait(l_fulls[cid], phase)
                    l_loaded = tlx.local_load(l_tiles[cid])
                    m_loaded = tlx.local_load(m_tiles[cid])
                    l = tl.split(l_loaded)[0][:, None] if SCALAR_N == 2 else l_loaded
                    m = tl.split(m_loaded)[0][:, None] if SCALAR_N == 2 else m_loaded
                    # Signal qk_empties after both l and m loads complete,
                    # since both tiles share the same synchronization group.
                    tlx.barrier_arrive(qk_empties[cid])
                    if RESCALE_OPT:
                        # RESCALE_OPT stores unscaled row-max in m_tiles.
                        # The bwd kernel expects scaled values (m * qk_scale),
                        # so we scale here before storing M.
                        m = m * sm_scale * 1.44269504
                    m += tl.math.log2(l)
                    offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                    m_ptrs = M + off_hz * N_CTX + offs_m
                    tl.store(m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]))

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
                    tlx.barrier_arrive(o_fulls[cid])

                tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # softmax groups
        with tlx.async_task(num_warps=4, registers=168, replicate=NUM_MMA_GROUPS):
            accum_cnt_qk = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )
                # initialize pointer to m and l
                m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                # FA4 update_row_sum has init_val being None for the first iteration, here
                # we use initial value of 1.0
                l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)
                qk_scale = sm_scale
                qk_scale *= 1.44269504  # 1/log(2)
                p_dtype = tlx.dtype_of(desc_v)

                cid = tlx.async_task_replica_id()
                offs_m = (start_m * BLOCK_M) + ((cid * BLOCK_M_SPLIT) + tl.arange(0, BLOCK_M_SPLIT))
                if STAGE & 1:
                    m_i, l_i, accum_cnt_qk = _softmax_inner_loop(
                        qk_fulls,
                        qk_tiles,
                        p_fulls,
                        p_tiles,
                        alpha_empties,
                        alpha_fulls,
                        alpha_tiles,
                        cid,
                        accum_cnt_qk,
                        qk_scale,
                        offs_m,
                        m_i,
                        l_i,
                        start_m,
                        N_CTX,
                        p_dtype,
                        BLOCK_M,
                        BLOCK_N,
                        NUM_MMA_SLICES,
                        STAGE=4 - STAGE,
                        RESCALE_OPT=RESCALE_OPT,
                        SCALAR_N=SCALAR_N,
                    )
                if STAGE & 2:
                    m_i, l_i, accum_cnt_qk = _softmax_inner_loop(
                        qk_fulls,
                        qk_tiles,
                        p_fulls,
                        p_tiles,
                        alpha_empties,
                        alpha_fulls,
                        alpha_tiles,
                        cid,
                        accum_cnt_qk,
                        qk_scale,
                        offs_m,
                        m_i,
                        l_i,
                        start_m,
                        N_CTX,
                        p_dtype,
                        BLOCK_M,
                        BLOCK_N,
                        NUM_MMA_SLICES,
                        STAGE=2,
                        RESCALE_OPT=RESCALE_OPT,
                        SCALAR_N=SCALAR_N,
                    )

                # prepare l_i for the epilog
                tlx.local_store(l_tiles[cid], tl.join(l_i, l_i) if SCALAR_N == 2 else l_i[:, None])
                tlx.local_store(m_tiles[cid], tl.join(m_i, m_i) if SCALAR_N == 2 else m_i[:, None])
                tlx.barrier_arrive(l_fulls[cid])
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # mma group
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            accum_cnt_qk = 0

            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                # initialize offsets
                _, _, lo, hi, _, _ = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )

                q_bufIdx, q_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_Q)
                k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

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
                )

                # -- compute q1 @ k ----
                tlx.barrier_wait(q_fulls[q_bufIdx + NUM_BUFFERS_Q], q_phase)
                tlx.barrier_wait(qk_empties[1], q_phase ^ 1)
                tlx.async_dot(
                    q_tiles[1],
                    k_tile,
                    qk_tiles[1],
                    use_acc=False,
                    mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                )

                _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)

                # -- compute p0 @ v ----
                # wait for the V buffer to be populated by the producer
                tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                tlx.barrier_wait(acc_fulls[0], qk_phase)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    p_bufIdx = slice_id
                    tlx.barrier_wait(p_fulls[p_bufIdx], qk_phase)
                    kv_slice = tlx.local_slice(
                        kv_tiles[v_bufIdx],
                        [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                        [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                    )
                    tlx.async_dot(
                        p_tiles[p_bufIdx],
                        kv_slice,
                        acc_tiles[0],
                        use_acc=slice_id > 0,
                        force_async=True,
                    )

                acc1_init = False

                for i in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    v_bufIdx_prev = v_bufIdx
                    qk_phase_prev = qk_phase

                    accum_cnt_qk += 1
                    accum_cnt_kv += 2
                    k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

                    # -- compute q0 @ k ----
                    # wait for the K buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                    _, qk_phase = get_bufidx_phase(accum_cnt_qk, 1)

                    tlx.async_dot(
                        q_tiles[0],
                        k_tile,
                        qk_tiles[0],
                        use_acc=False,
                        mBarriers=[qk_fulls[0]],
                    )

                    # -- compute p1 @ v from the previous iteration----
                    tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        p_bufIdx = slice_id + NUM_MMA_SLICES
                        tlx.barrier_wait(p_fulls[p_bufIdx], qk_phase_prev)
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx_prev],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                        )
                        use_acc = acc1_init if slice_id == 0 else True
                        mBarriers = [kv_empties[v_bufIdx_prev]] if slice_id == NUM_MMA_SLICES - 1 else []
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_slice,
                            acc_tiles[1],
                            use_acc=use_acc,
                            mBarriers=mBarriers,
                        )

                    acc1_init = True

                    # -- compute q1 @ k ----
                    tlx.async_dot(
                        q_tiles[1],
                        k_tile,
                        qk_tiles[1],
                        use_acc=False,
                        mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                    )

                    # -- compute p0 @ v ----
                    # wait for the V buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)

                    tlx.barrier_wait(acc_fulls[0], qk_phase)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        p_bufIdx = slice_id
                        tlx.barrier_wait(p_fulls[p_bufIdx], qk_phase)
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                        )
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_slice,
                            acc_tiles[0],
                            use_acc=True,
                            force_async=True,
                        )

                tlx.tcgen05_commit(q_empties[q_bufIdx])
                tlx.tcgen05_commit(q_empties[q_bufIdx + NUM_BUFFERS_Q])
                tlx.tcgen05_commit(acc_empties[0])

                # -- compute p1 @ v ----
                tlx.barrier_wait(acc_fulls[1], qk_phase)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    p_bufIdx = slice_id + NUM_MMA_SLICES
                    tlx.barrier_wait(p_fulls[p_bufIdx], qk_phase)
                    kv_slice = tlx.local_slice(
                        kv_tiles[v_bufIdx],
                        [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                        [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                    )
                    use_acc = acc1_init if slice_id == 0 else True
                    mBarriers = [acc_empties[1], kv_empties[v_bufIdx]] if slice_id == NUM_MMA_SLICES - 1 else []
                    tlx.async_dot(
                        p_tiles[p_bufIdx],
                        kv_slice,
                        acc_tiles[1],
                        use_acc=use_acc,
                        mBarriers=mBarriers,
                    )

                accum_cnt_qk += 1
                accum_cnt_kv += 2
                tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # load
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                # initialize offsets
                _, _, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )

                # load q0
                q_bufIdx, q_phase = get_bufidx_phase(tile_count, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
                qo_offset_y_split = qo_offset_y
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])

                # loop over loading k, v
                k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(kv_empties, k_bufIdx)
                tlx.barrier_wait(k_empty, k_phase ^ 1)

                # load K
                k_full = tlx.local_view(kv_fulls, k_bufIdx)
                k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                tlx.barrier_expect_bytes(k_full, K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                # load q1
                q_bufIdx += NUM_BUFFERS_Q
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
                qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])

                v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(kv_empties, v_bufIdx)
                tlx.barrier_wait(v_empty, v_phase ^ 1)
                # load V
                v_full = tlx.local_view(kv_fulls, v_bufIdx)
                v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                tlx.barrier_expect_bytes(v_full, V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                kv_offset_y += BLOCK_N
                accum_cnt_kv += 2

                for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(kv_empties, k_bufIdx)
                    tlx.barrier_wait(k_empty, k_phase ^ 1)
                    # load K
                    k_full = tlx.local_view(kv_fulls, k_bufIdx)
                    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                    tlx.barrier_expect_bytes(k_full, K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                    v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(kv_empties, v_bufIdx)
                    tlx.barrier_wait(v_empty, v_phase ^ 1)
                    # load V
                    v_full = tlx.local_view(kv_fulls, v_bufIdx)
                    v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                    tlx.barrier_expect_bytes(v_full, V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                    kv_offset_y += BLOCK_N
                    accum_cnt_kv += 2

                tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # epilog group
        with tlx.async_task(num_warps=1, registers=24):
            # initialize offsets
            tile_count = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            while tile_id != -1:
                # initialize offsets
                _, _, _, _, qo_offset_y, _ = _compute_offsets(
                    tile_id,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    STAGE,
                    GROUP_SIZE_N,
                )
                _, phase = get_bufidx_phase(tile_count, 1)
                for cid in tl.static_range(0, NUM_MMA_GROUPS):
                    tlx.barrier_wait(o_fulls[cid], phase)
                    qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                    tlx.async_descriptor_store(desc_o, o_tiles[cid], [qo_offset_y_split, 0])
                    tlx.async_descriptor_store_wait(0)
                    tlx.barrier_arrive(o_empties[cid])

                tile_count += 1
                tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dq_postprocess(DQ_ACCUM, DQ_OUT,  #
                             N_CTX,  #
                             BLK: tl.constexpr, HALF_HD: tl.constexpr,  #
                             BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,  #
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
    dst = DQ_OUT + off_hz * N_CTX * HEAD_DIM + q * HEAD_DIM + h
    tl.store(dst, val.to(DQ_OUT.dtype.element_ty))


@triton.jit
def bwd_calculate_offsets(
    tile_idx,
    n_tile_num,
    num_pid_m,
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    STAGE: tl.constexpr,
):
    bhid = tile_idx // n_tile_num
    pid = tile_idx % n_tile_num
    pid, bhid = tl.swizzle2d(pid, bhid, n_tile_num, num_pid_m, GROUP_SIZE_M)
    batch = bhid // H
    head = bhid % H
    off_chz = (bhid * N_CTX).to(tl.int64)
    start_n = pid
    start_m = _get_start_m_bwd(start_n, BLOCK_N1, STAGE)
    num_steps = (N_CTX - start_m) // BLOCK_M1
    return off_chz, batch, head, start_m, start_n, num_steps


_bwd_selected_meta = {}


def _bwd_host_descriptor_pre_hook_tlx(nargs):
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_CTAS = nargs.get("NUM_CTAS", 1)

    _bwd_selected_meta["BLOCK_M1"] = BLOCK_M1
    _bwd_selected_meta["NUM_CTAS"] = NUM_CTAS

    # Reset dq accumulator to zeros before each autotuner warmup run.
    # dq uses TMA reduce-add, so stale values accumulate across runs.
    # dk/dv don't need zeroing — they use use_acc=False on the first iteration.
    nargs["desc_dq"].base.zero_()

    nargs["desc_q"].block_shape = [1, 1, BLOCK_M1, HEAD_DIM // NUM_CTAS]
    nargs["desc_do"].block_shape = [1, 1, BLOCK_M1, HEAD_DIM // NUM_CTAS]
    nargs["desc_v"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM]
    nargs["desc_k"].block_shape = [1, 1, BLOCK_N1, HEAD_DIM]
    if NUM_CTAS > 1:
        EPILOGUE_SUBTILE = nargs["EPILOGUE_SUBTILE"]
        DQ_SLICE_N = HEAD_DIM // EPILOGUE_SUBTILE
        nargs["desc_dq"].block_shape = [1, 1, BLOCK_M1, DQ_SLICE_N]
    else:
        DQ_REDUCE_NCOL = nargs["DQ_REDUCE_NCOL"]
        nargs["desc_dq"].block_shape = [1, 1, BLOCK_M1, DQ_REDUCE_NCOL]
    DKV_STORE_NCOL = nargs["DKV_STORE_NCOL"]
    nargs["desc_dv"].block_shape = [1, 1, BLOCK_N1, DKV_STORE_NCOL]
    nargs["desc_dk"].block_shape = [1, 1, BLOCK_N1, DKV_STORE_NCOL]
    nargs["desc_m"].block_shape = [BLOCK_M1]
    nargs["desc_delta"].block_shape = [BLOCK_M1]
    # 2-CTA: separate B-operand descriptors for the transposed views.
    if "desc_kt" in nargs and "desc_qt" in nargs:
        nargs["desc_kt"].block_shape = [1, 1, BLOCK_N1 * NUM_CTAS, HEAD_DIM // NUM_CTAS]
        nargs["desc_qt"].block_shape = [1, 1, BLOCK_M1 // NUM_CTAS, HEAD_DIM]
        nargs["desc_dot"].block_shape = [1, 1, BLOCK_M1 // NUM_CTAS, HEAD_DIM]


configs_bwd_1cta = [
    triton.Config(
        {
            "BLOCK_M1": bm1,
            "BLOCK_N1": 128,
            "NUM_BUFFERS_KV": 1,
            "NUM_BUFFERS_Q": 2,
            "NUM_BUFFERS_DO": 1,
            "NUM_BUFFERS_DS": 1,
            "NUM_BUFFERS_TMEM": 1,
            "DKV_STORE_NCOL": 64,
            "NUM_COMPUTE_SLICES": 2,
            "DQ_REDUCE_STAGES": 2,
            "DQ_REDUCE_NCOL": 32,
            "EPILOGUE_SUBTILE": 4,
            "GROUP_SIZE_M": 1,
            "USE_WARP_BARRIER": uwb,
            "NUM_CTAS": 1,
        },
        num_warps=8,
        num_stages=1,
        pre_hook=_bwd_host_descriptor_pre_hook_tlx,
    ) for bm1 in [64] for uwb in [False, True]
]

configs_bwd_2cta = [
    # 2-CTA config
    triton.Config(
        {
            "BLOCK_M1": 128,
            "BLOCK_N1": 128,
            "NUM_BUFFERS_KV": 1,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_DO": 1,
            "NUM_BUFFERS_DS": 1,
            "NUM_BUFFERS_TMEM": 1,
            "DKV_STORE_NCOL": 64,
            "NUM_COMPUTE_SLICES": 2,
            "DQ_REDUCE_STAGES": 2,
            "DQ_REDUCE_NCOL": 32,
            "EPILOGUE_SUBTILE": 8,
            "GROUP_SIZE_M": 1,
            "USE_WARP_BARRIER": False,
            "NUM_CTAS": 2,
        },
        num_warps=8,
        num_stages=1,
        pre_hook=_bwd_host_descriptor_pre_hook_tlx,
        ctas_per_cga=(2, 1, 1),
    ),
]

configs_bwd_tlx = configs_bwd_1cta + configs_bwd_2cta


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
        mBarriers=[q_empties[q_buf_id], dk_fulls[tmem_buf_id]],
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

    # Wait for K and V loads to complete (not bundled into q_fulls/do_fulls in 2-CTA).
    tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
    tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)

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
        # dq (cols 0-63) also aliases the qk TMEM region, which at this point
        # holds qk(j) that the compute task may still be reading: ds_fulls
        # above only proves compute consumed qk(j-1) — one iteration stale on
        # the single-buffered TMEM ring. Wait for compute to release qk(j)
        # before Dot 5's write lands in its storage (the same guard Dot 1
        # uses for the full overwrite). Latent hazard: without this, Dot 5's
        # write is ordered against compute's qk(j) read only by MMA-pipe
        # timing.
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
        ],
        two_ctas=True,
    )
    tlx.tcgen05_commit(k_mma_done[kv_buf_id], two_ctas=True)
    tlx.tcgen05_commit(kt_empties[kv_buf_id], two_ctas=True)

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

    # Load M (raw bulk copy)
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

        # Load M (raw bulk copy)
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
    # 2-CTA params (defaults for 1-CTA)
    USE_2CTA: tl.constexpr = False,
    NUM_CTAS: tl.constexpr = 1,
    dsT_xchg_tiles=None,
    ds_xchg_tiles=None,
    ds_peer_fulls=None,
    ds_empties=None,
    dsT_fulls=None,
    cluster_cta_rank=0,
    P_BUF_OFFSET: tl.constexpr = 0,
    num_steps_override=0,
    qk_read_done=None,
    dp_read_done=None,
):
    start_block_n = start_n * BLOCK_N1
    offs_n = start_block_n + tl.arange(0, BLOCK_N1)
    if num_steps_override > 0:
        num_steps = num_steps_override
    else:
        lo, hi = _get_unfused_bwd_loop_bounds(start_n, N_CTX, BLOCK_N1, STAGE)
        num_steps = (hi - lo) // BLOCK_M1
    for _ in range(num_steps):
        tmem_buf_id, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
        ds_buf_id, _ = get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)

        # Wait for QK first (from MMA, typically ready sooner), then M.
        # D wait is deferred to right before dS computation (like FA4).
        m_buf_id, m_phase = get_bufidx_phase(blk_idx, M_STAGE)
        d_buf_id, d_phase = get_bufidx_phase(blk_idx, D_STAGE)
        tlx.barrier_wait(qk_fulls[tmem_buf_id], tmem_phase)
        tlx.barrier_wait(m_fulls[m_buf_id], m_phase)

        qkT = tlx.local_load(qk_tiles[tmem_buf_id])
        m = tlx.local_load(sM_tiles[m_buf_id])
        # qkT/pT are transposed: [BLOCK_N1 (keys), BLOCK_M1 (queries)]. Apply the
        # causal mask to the logits via the R2P bitmask helper (keep query-cols
        # m >= key-row n), then exp2 (exp2(-inf) = 0), avoiding the per-element
        # ISETP arithmetic of `offs_m >= offs_n`.
        sT = _sub_f32x2(qkT, m[None, :])
        if STAGE == 1:
            col_limit_left = (offs_n - curr_m)[:, None]
            sT = _apply_causal_mask(sT, col_limit_left, BLOCK_M1, keep_ge=True)
        pT = tl.math.exp2(sT)

        # Store P to TMEM.
        ppT = pT.to(do_out_dtype)
        # Hazard 1 (intra-task WAR): P (f16) aliases the upper half of the qk
        # (f32) TMEM region; tcgen05 ld/st warp->chunk maps differ, so a fast
        # warp's P store can overwrite a 32x32 chunk a slow warp has not read as
        # qkT. Rendezvous all 8 compute warps between the read and the store.
        tlx.barrier_arrive(qk_read_done[tmem_buf_id])
        tlx.barrier_wait(qk_read_done[tmem_buf_id], tmem_phase)
        tlx.local_store(p_tiles[tmem_buf_id + P_BUF_OFFSET], ppT)
        # P aliases the QK TMEM region, so qk_empties (which frees that region for
        # reuse) must be signaled after P is stored, not before. The
        # local_store->TMEM lowering auto-emits tcgen05.wait::st, so p_fulls
        # already observes the completed P store; no manual wait.
        if USE_2CTA:
            tlx.barrier_arrive(qk_empties[tmem_buf_id], 1, remote_cta_rank=0)
            tlx.barrier_arrive(p_fulls[tmem_buf_id], 1, remote_cta_rank=0)
        else:
            tlx.barrier_arrive(qk_empties[tmem_buf_id])
            tlx.barrier_arrive(p_fulls[tmem_buf_id])

        # --- Phase 3: Compute dS = pT * (dpT - Di). ---
        tlx.barrier_wait(dp_fulls[tmem_buf_id], tmem_phase)
        dpT = tlx.local_load(dp_tiles[tmem_buf_id])
        tlx.barrier_wait(d_fulls[d_buf_id], d_phase)
        Di = tlx.local_load(sD_tiles[d_buf_id])
        tlx.barrier_arrive(m_empties[m_buf_id])
        tlx.barrier_arrive(d_empties[d_buf_id])
        dsT = _mul_f32x2(pT, _sub_f32x2(dpT, Di[None, :]))
        dsT = dsT.to(q_out_dtype)
        # Hazard 1 (intra-task WAR): dsT (f16) aliases dp's (f32) region -- same
        # warp->chunk mismatch as the P store above.
        tlx.barrier_arrive(dp_read_done[tmem_buf_id])
        tlx.barrier_wait(dp_read_done[tmem_buf_id], tmem_phase)
        tlx.local_store(dsT_tmem_tiles[ds_buf_id], dsT)
        # dsT aliases the dP TMEM region, so dp_empties (which frees that region
        # for reuse) must be signaled after dsT is stored, not before. The
        # local_store->TMEM lowering auto-emits tcgen05.wait::st (+barrier), so the
        # dsT_tmem_fulls arrive and the 2-CTA TMEM read-back below both observe the
        # completed store; no manual wait.
        if not REUSE_DP_FOR_DQ and not USE_2CTA:
            tlx.barrier_arrive(dp_empties[tmem_buf_id])
        # 2-CTA: exchange half of dS with peer via DSMEM, then
        # overwrite ds_tiles so it contains mixed dS from both CTAs.
        if USE_2CTA:
            tlx.barrier_arrive(dsT_tmem_fulls[ds_buf_id], 1, remote_cta_rank=0)
            _, ds_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)
            # Wait for MMA Dot 5 to finish reading ds_tiles before overwriting.
            tlx.barrier_wait(ds_empties[ds_buf_id], ds_phase ^ 1)
            peer_rank = 1 - cluster_cta_rank
            # Load own/peer M-columns from TMEM (dsT_tmem_tiles), store to SMEM.
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
            # Signal dp_empties right after TMEM reload is done —
            # dsT_tmem is no longer needed, MMA can overwrite dp/dq TMEM.
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
            # NOTE: ds_peer_fulls wait + ds_fulls signal moved to relay task.
        else:
            tlx.local_store(ds_tiles[ds_buf_id], dsT)
            tlx.fence("async_shared")
            tlx.barrier_arrive(ds_fulls[ds_buf_id])
            tlx.barrier_arrive(dsT_tmem_fulls[ds_buf_id])

        curr_m += step_m
        blk_idx += 1
    return curr_m, blk_idx


@triton.autotune(configs=configs_bwd_tlx, key=["N_CTX", "HEAD_DIM"])
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
    DKV_STORE_NCOL: tl.constexpr,
    STAGE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_WARP_BARRIER: tl.constexpr = False,
    EPILOGUE_SUBTILE: tl.constexpr = 4,
    NUM_CTAS: tl.constexpr = 1,
):
    # Runtime error if NUM_BUFFERS_DO != 1
    tl.static_assert(NUM_BUFFERS_DO == 1)

    # If we have BLOCK_M1 == 128 and HEAD_DIM == 128 we don't have enough
    # TMEM. We may need to expand this condition across other configs in
    # the future.
    # Note: Setting REUSE_DP_FOR_DQ=False with BLOCK_M1 == 64 and
    # HEAD_DIM == 128 will result in an accuracy issue.
    REUSE_DP_FOR_DQ: tl.constexpr = (BLOCK_M1 == 128) and (HEAD_DIM == 128) and (NUM_CTAS == 1)

    USE_2CTA: tl.constexpr = NUM_CTAS == 2

    # Compute bytes per element for each tensor type
    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))
    DO_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_do))

    # 3D grid: (n_tile_num, H, Z) — batch/head from grid dims, no div/rem.
    # n_tile_num = tl.cdiv(N_CTX, BLOCK_N1)
    start_n = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    off_chz = ((batch * H + head) * N_CTX).to(tl.int64)
    if USE_2CTA and STAGE == 3:
        # Causal 2-CTA: both CTAs must iterate the same M-blocks
        # because the MMA is collaborative (two_ctas=True).
        # Use the lower CTA's start_n to compute start_m.
        cluster_cta_rank_early = tlx.cluster_cta_rank()
        base_start_n = start_n - cluster_cta_rank_early
        start_m = _get_start_m_bwd(base_start_n, BLOCK_N1, STAGE)
    else:
        start_m = _get_start_m_bwd(start_n, BLOCK_N1, STAGE)
    num_steps = (N_CTX - start_m) // BLOCK_M1
    start_block_n = start_n * BLOCK_N1

    # =========================================================================
    # Allocate all barriers (before SMEM/TMEM allocations)
    # =========================================================================
    M_STAGE: tl.constexpr = 1 if USE_2CTA else 2
    D_STAGE: tl.constexpr = 2

    # K/V are bundled into Q/dO barriers (loaded once per n_block in prologue).
    # k_mma_done: signaled by MMA task after dq dot (last k_tiles read).
    # k_empties: signaled by compute task after dKV staging stores complete
    #            AND k_mma_done is received.  Gates both k_tiles and v_tiles
    #            (v_tiles aliased by sdv_store_buf) since V load follows K
    #            load in the load task.
    k_mma_done = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
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
    # Dedicated 8-warp rendezvous barriers for the intra-task aliased-TMEM WAR
    # (Hazard 1): all compute warps must finish reading qk/dp before any warp
    # overwrites that region with P/dsT. Intra-CTA (per-CTA) regardless of
    # NUM_CTAS -- a first-class TLX warp barrier in place of tl.debug_barrier().
    qk_read_done = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_TMEM, num_warps=8)
    dp_read_done = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_TMEM, num_warps=8)
    if USE_WARP_BARRIER:
        dq_empties = tlx.alloc_warp_barrier(num_barriers=NUM_BUFFERS_TMEM, num_warps=4)
    else:
        dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS)

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
    if USE_2CTA:
        dq_store_buf = tlx.local_alloc((BLOCK_M1, DQ_SLICE_N), tlx.dtype_of(desc_dq), 2)
    else:
        DQ_REDUCE_ITERS: tl.constexpr = HEAD_DIM // DQ_REDUCE_NCOL
        dq_store_buf = tlx.local_alloc((BLOCK_M1, DQ_REDUCE_NCOL), tlx.dtype_of(desc_dq), DQ_REDUCE_STAGES)

    # - sdv reuses v_tiles (free after dv_fulls; MMA's last v_tiles read —
    #   the dpT dot — precedes dv_fulls).
    # - sdk reuses k_tiles (MMA's dq dot still reads k_tiles after dk_fulls,
    #   so the compute task must wait on k_mma_done before writing sdk).
    sdv_store_buf = tlx.local_alloc((BLOCK_N1, DKV_STORE_NCOL), tlx.dtype_of(desc_dv), NUM_BUFFERS_KV, reuse=v_tiles)
    sdk_store_buf = tlx.local_alloc((BLOCK_N1, DKV_STORE_NCOL), tlx.dtype_of(desc_dk), NUM_BUFFERS_KV, reuse=k_tiles)

    sM_tiles = tlx.local_alloc((BLOCK_M1, ), tl.float32, M_STAGE)
    sD_tiles = tlx.local_alloc((BLOCK_M1, ), tl.float32, D_STAGE)

    # S/P/dQ share TMEM via storage alias. S and P fully overlap (shared).
    # In 2-CTA, P and dQ must be distinct (non-overlapping) so that
    # Dot 5 (dQ) doesn't overwrite P before Dot 3 (dV) reads it.
    qk_p_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    qk_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem,
                               reuse=qk_p_storage_alias)
    # In 2-CTA mode, P is offset to column 64 (after dQ's 64 cols at column 0).
    # P's per-buffer stride is 64 i32 cols (128x128 f16), so num=2 + index 1
    # naturally places P at column 64. In 1-CTA mode, P stays at column 0.
    P_NUM_BUFFERS: tl.constexpr = 2 if USE_2CTA and not REUSE_DP_FOR_DQ else NUM_BUFFERS_TMEM
    P_BUF_IDX: tl.constexpr = 1 if USE_2CTA and not REUSE_DP_FOR_DQ else 0
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
            # 2-CTA: dQ at column 0, P offset to column 64.
            # dQ (64x128 f32 twocta_rhs) = 64 i32 cols at columns 0-63.
            # P (128x128 f16) = 64 i32 cols at columns 64-127.
            # P uses num=2 with index 1 to get the offset; P's per-buffer
            # stride is naturally 64 from getTmemAllocSizes(128x128 f16).
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

    with tlx.async_tasks():
        # compute
        with tlx.async_task("default"):
            blk_idx = 0
            curr_m = start_m
            step_m = BLOCK_M1
            do_out_dtype = tlx.dtype_of(desc_do)
            q_out_dtype = tlx.dtype_of(desc_q)
            if USE_2CTA and STAGE == 3:
                # 2-CTA causal: single loop with mask applied every iteration.
                # Both CTAs have the same num_steps (from base_start_n).
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
                    USE_2CTA=USE_2CTA, qk_read_done=qk_read_done, dp_read_done=dp_read_done,
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
                        USE_2CTA=USE_2CTA, qk_read_done=qk_read_done, dp_read_done=dp_read_done,
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
                        USE_2CTA=USE_2CTA, qk_read_done=qk_read_done, dp_read_done=dp_read_done,
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
            DKV_STORE_ITERS: tl.constexpr = HEAD_DIM // DKV_STORE_NCOL
            for slice_id in tl.static_range(DKV_STORE_ITERS):
                dv_slice = tlx.local_slice(
                    dv_tiles[kv_buf_id],
                    [0, slice_id * DKV_STORE_NCOL],
                    [BLOCK_N1, DKV_STORE_NCOL],
                )
                dv = tlx.local_load(dv_slice)
                tlx.async_descriptor_store_wait(0)
                tlx.local_store(sdv_store_buf[kv_buf_id], dv.to(tlx.dtype_of(desc_dv)))
                tlx.async_descriptor_store(
                    desc_dv,
                    sdv_store_buf[kv_buf_id],
                    [batch, head, start_block_n, slice_id * DKV_STORE_NCOL],
                )
            if USE_2CTA:
                tlx.barrier_arrive(dv_empties[kv_buf_id], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(dv_empties[kv_buf_id])
            tlx.barrier_wait(dk_fulls[kv_buf_id], kv_phase)
            # Wait for MMA's dq dot (last k_tiles read) before writing
            # sdk_store_buf which aliases k_tiles.
            tlx.barrier_wait(k_mma_done[kv_buf_id], kv_phase)
            for slice_id in tl.static_range(DKV_STORE_ITERS):
                dk_slice = tlx.local_slice(
                    dk_tiles[kv_buf_id],
                    [0, slice_id * DKV_STORE_NCOL],
                    [BLOCK_N1, DKV_STORE_NCOL],
                )
                dk = tlx.local_load(dk_slice)
                dk *= sm_scale
                tlx.async_descriptor_store_wait(0)
                tlx.local_store(sdk_store_buf[kv_buf_id], dk.to(tlx.dtype_of(desc_dk)))
                tlx.async_descriptor_store(
                    desc_dk,
                    sdk_store_buf[kv_buf_id],
                    [batch, head, start_block_n, slice_id * DKV_STORE_NCOL],
                )
            tlx.async_descriptor_store_wait(0)
            # All staging stores done + MMA done reading k_tiles →
            # safe for load task to refill both k_tiles and v_tiles.
            tlx.barrier_arrive(k_empties[kv_buf_id])
            if USE_2CTA:
                tlx.barrier_arrive(dk_empties[kv_buf_id], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(dk_empties[kv_buf_id])

        # reduction
        with tlx.async_task(num_warps=4, registers=88):
            blk_idx = 0
            curr_m = start_m
            step_m = BLOCK_M1
            for _ in range(num_steps):
                tmem_buf_id, tmem_phase = get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

                # wait for dq = tl.dot(tl.trans(dsT), k)
                tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                if USE_2CTA:
                    dq_m_offset = cluster_cta_rank * DQ_STORE_M
                    packed_row_base = 2 * (curr_m + dq_m_offset)
                    DQ_PACK_ITERS: tl.constexpr = (HEAD_DIM // NUM_CTAS) // DQ_SLICE_N
                    dq_full = tlx.local_load(dq_phys[tmem_buf_id + DQ_BUF_IDX])
                    if USE_WARP_BARRIER:
                        tlx.barrier_arrive(dq_empties[tmem_buf_id])
                    else:
                        tlx.barrier_arrive(dq_empties[tmem_buf_id], 1, remote_cta_rank=0)
                    dq_full = dq_full * LN2
                    dq_slices = _split_n_2D(dq_full, DQ_PACK_ITERS)
                    for slice_id in tl.static_range(DQ_PACK_ITERS):
                        dq_smem = dq_store_buf[slice_id % 2]
                        tlx.async_descriptor_store_wait(1)
                        tlx.local_store(dq_smem, dq_slices[slice_id].to(tlx.dtype_of(desc_dq)))
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

                # Increment pointers.
                curr_m += step_m
                blk_idx += 1

            # Wait for the final tile
            tlx.async_descriptor_store_wait(0)

        # mma
        with tlx.async_task(num_warps=1, registers=88):
            blk_idx = 0
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

        # load
        with tlx.async_task(num_warps=1, registers=88):
            blk_idx = 0
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

        # relay — waits for peer's DSMEM to arrive, then signals ds_fulls
        # so the MMA task can read the combined ds_tiles.
        if USE_2CTA:
            with tlx.async_task(num_warps=1, registers=40):
                for blk_idx_relay in range(num_steps):
                    ds_buf_id_relay, ds_phase_relay = get_bufidx_phase(blk_idx_relay, NUM_BUFFERS_DS)
                    tlx.barrier_wait(ds_peer_fulls[ds_buf_id_relay], ds_phase_relay)
                    tlx.fence("async_shared")
                    tlx.barrier_arrive(ds_fulls[ds_buf_id_relay], 1, remote_cta_rank=0)

        # TODO: empty task to absorb warps — needs num_warps bump in configs
        # EMPTY_WARPS: tl.constexpr = 1 if USE_2CTA else 2
        # with tlx.async_task(num_warps=EMPTY_WARPS, registers=24):
        #     pass


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        stage = 3 if causal else 1

        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            q,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_o = TensorDescriptor(
            o,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        grid = lambda META: (triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1], )

        ctx.grid = grid
        _attn_fwd_ws[grid](
            sm_scale,
            M,  #
            q.shape[0],
            q.shape[1],  #
            desc_q,
            desc_k,
            desc_v,
            desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert o.is_contiguous() and do.is_contiguous()
        dq = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        _HALF_HD = ctx.HEAD_DIM // 2
        dq_accum = torch.zeros([BATCH, N_HEAD, N_CTX, ctx.HEAD_DIM], device=q.device, dtype=torch.float32)
        PRE_BLOCK = 128
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM,  #
        )

        dummy_block = [1, 1, 1, 1]
        HEAD_DIM = ctx.HEAD_DIM
        desc_shape = [BATCH, N_HEAD, N_CTX, HEAD_DIM]
        desc_strides = [N_HEAD * N_CTX * HEAD_DIM, N_CTX * HEAD_DIM, HEAD_DIM, 1]
        desc_k = TensorDescriptor(
            arg_k,
            shape=desc_shape,
            strides=desc_strides,
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=desc_shape,
            strides=desc_strides,
            block_shape=dummy_block,
        )
        desc_q = TensorDescriptor(
            q,
            shape=desc_shape,
            strides=desc_strides,
            block_shape=dummy_block,
        )
        desc_do = TensorDescriptor(
            do,
            shape=desc_shape,
            strides=desc_strides,
            block_shape=dummy_block,
        )
        packed_shape = [BATCH, N_HEAD, 2 * N_CTX, _HALF_HD]
        packed_strides = [N_HEAD * N_CTX * HEAD_DIM, N_CTX * HEAD_DIM, _HALF_HD, 1]
        desc_dq = TensorDescriptor(
            dq_accum,
            shape=packed_shape,
            strides=packed_strides,
            block_shape=dummy_block,
        )
        desc_dk = TensorDescriptor(
            dk,
            shape=desc_shape,
            strides=desc_strides,
            block_shape=dummy_block,
        )
        desc_dv = TensorDescriptor(
            dv,
            shape=desc_shape,
            strides=desc_strides,
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

        desc_kt = TensorDescriptor(arg_k, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
        desc_qt = TensorDescriptor(q, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
        desc_dot = TensorDescriptor(do, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # NUM_SMS = torch.cuda.get_device_properties(q.device).multi_processor_count

        def grid_persistent(meta):
            n_tiles = triton.cdiv(N_CTX, meta["BLOCK_N1"])
            num_ctas = meta.get("NUM_CTAS", 1)
            n_tiles = triton.cdiv(n_tiles, num_ctas) * num_ctas
            return (n_tiles, N_HEAD, BATCH)

        stage = 3 if ctx.causal else 1
        _attn_bwd_ws[grid_persistent](
            desc_q, desc_k, desc_v, ctx.sm_scale, desc_do, desc_dq, desc_dk, desc_dv,  #
            desc_m, desc_delta,  #
            M, delta,  #
            N_HEAD, BATCH,  #
            N_CTX,  #
            desc_kt, desc_qt, desc_dot,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            STAGE=stage,  #
        )

        _blk = _bwd_selected_meta["BLOCK_M1"] // _bwd_selected_meta["NUM_CTAS"]
        post_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        _attn_bwd_dq_postprocess[post_grid](
            dq_accum, dq,  #
            N_CTX,  #
            BLK=_blk, HALF_HD=_HALF_HD,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM,  #
        )

        return dq, dk, dv, None, None


def attention(q, k, v, sm_scale, causal, config=None):
    if config is None:
        return _attention.apply(q, k, v, sm_scale, causal)

    # Non-autotuned path with explicit config
    HEAD_DIM_K = q.shape[-1]
    stage = 3 if causal else 1
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    dummy_block = [1, 1]
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

    # Apply pre_hook to set block shapes
    nargs = {**config, "HEAD_DIM": HEAD_DIM_K, "desc_q": desc_q, "desc_k": desc_k, "desc_v": desc_v, "desc_o": desc_o}
    _host_descriptor_pre_hook(nargs)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(q.shape[2], config["BLOCK_M"]) * q.shape[0] * q.shape[1], 1, 1)
    _attn_fwd_ws.fn[grid](
        sm_scale,
        M,
        q.shape[0],
        q.shape[1],
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        num_stages=1,
        **config,
    )
    return o
