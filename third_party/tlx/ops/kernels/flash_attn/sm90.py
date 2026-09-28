"""Hopper (sm90) Flash Attention implementation for ``tlx.ops``.

Promoted from ``tutorials/hopper_fa_ws_pipelined_pingpong.py``. The supported
contract is contiguous square FP16/BF16 attention with ``HEAD_DIM`` 64 or 128.
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    if nargs["USE_BM192"]:
        nargs["desc_q"].block_shape = [1, BLOCK_M_SPLIT, HEAD_DIM]
        nargs["desc_v"].block_shape = [1, BLOCK_N, HEAD_DIM]
        nargs["desc_k"].block_shape = [1, BLOCK_N, HEAD_DIM]
        nargs["desc_o"].block_shape = [1, BLOCK_M_SPLIT, HEAD_DIM]
    else:
        nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
        if nargs["FP8_OUTPUT"]:
            nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
        else:
            nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
        nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
        nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]


_DEFAULT_BLOCK_M = 128
# Profile-selected on H100 to avoid consumer spills without reducing CTA residency.
_FWD_CONSUMER_REGISTERS = tl.constexpr(240)
_DEFAULT_ROW_SCHEDULE = {
    "ROW_REVERSE_HEAD_GROUP": 0,
    "ROW_REVERSE_MAX": 0,
    "ROW_AFFINE_MUL": 1,
    "ROW_AFFINE_ADD": 0,
    "ROW_AFFINE_MOD": 0,
    "ROW_SWAP_SRC_0": 0,
    "ROW_SWAP_SRC_1": 0,
    "ROW_SWAP_XOR": 0,
}
_CAUSAL_ROW_SCHEDULES = {
    8: {
        **_DEFAULT_ROW_SCHEDULE,
        "ROW_REVERSE_HEAD_GROUP": 1,
        "ROW_REVERSE_MAX": 7,
    },
    16: {
        **_DEFAULT_ROW_SCHEDULE,
        "ROW_REVERSE_HEAD_GROUP": 1,
        "ROW_REVERSE_MAX": 15,
    },
    32: {
        **_DEFAULT_ROW_SCHEDULE,
        "ROW_AFFINE_MUL": 5,
        "ROW_AFFINE_ADD": 29,
        "ROW_AFFINE_MOD": 32,
        "ROW_SWAP_SRC_0": 7,
        "ROW_SWAP_SRC_1": 15,
        "ROW_SWAP_XOR": 8,
    },
    64: {
        **_DEFAULT_ROW_SCHEDULE,
        "ROW_REVERSE_HEAD_GROUP": 2,
        "ROW_REVERSE_MAX": 63,
    },
}
_DEFAULT_LAUNCH_TUNING = {
    "STEADY_UNROLL": 2,
    "WORKER_CAP": None,
}
_NONCAUSAL_WORKLOAD_LAUNCH_TUNING = {(torch.bfloat16, 4, 48, n_ctx, 64): {
                                         "STEADY_UNROLL": 1,
                                         "WORKER_CAP": None,
                                         "WORKER_MULTIPLIER": 1,
                                     }
                                     for n_ctx in (1024, 2048, 4096, 8192)}
_CAUSAL_WORKLOAD_LAUNCH_TUNING = {
    (torch.bfloat16, 4, 48, 1024, 128): {
        "STEADY_UNROLL": 1,
        "WORKER_CAP": None,
    },
    (torch.bfloat16, 4, 48, 2048, 64): {
        "STEADY_UNROLL": 1,
        "WORKER_CAP": None,
    },
    (torch.bfloat16, 4, 48, 4096, 64): {
        "STEADY_UNROLL": 2,
        "WORKER_CAP": 129,
    },
    (torch.bfloat16, 4, 48, 2048, 128): {
        "STEADY_UNROLL": 1,
        "WORKER_CAP": None,
    },
    (torch.bfloat16, 4, 48, 4096, 128): {
        "STEADY_UNROLL": 1,
        "WORKER_CAP": 131,
    },
}


def _select_row_schedule(causal, n_ctx, block_m):
    if not causal:
        return dict(_DEFAULT_ROW_SCHEDULE)
    num_row_tiles = triton.cdiv(n_ctx, block_m)
    return dict(_CAUSAL_ROW_SCHEDULES.get(num_row_tiles, _DEFAULT_ROW_SCHEDULE))


def _select_forward_policy(causal, shape, dtype, block_m, num_sms):
    row_schedule = _select_row_schedule(causal, shape[2], block_m)
    workload_key = (dtype, *shape)
    workload_tuning = _CAUSAL_WORKLOAD_LAUNCH_TUNING if causal else _NONCAUSAL_WORKLOAD_LAUNCH_TUNING
    launch_tuning = workload_tuning.get(workload_key, _DEFAULT_LAUNCH_TUNING)
    worker_cap = launch_tuning["WORKER_CAP"]
    target_workers = num_sms * launch_tuning.get("WORKER_MULTIPLIER", 1)
    if worker_cap is not None:
        target_workers = min(target_workers, worker_cap)
    return row_schedule, launch_tuning["STEADY_UNROLL"], target_workers


configs = [
    triton.Config(
        {
            "BLOCK_M": _DEFAULT_BLOCK_M,
            "BLOCK_N": block_n,
            "NUM_BUFFERS": num_buffers,
            "NUM_MMA_WARPS": 8,
            "NUM_MMA_GROUPS": 2,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    ) for block_n, num_buffers in ((128, 2), (128, 3), (64, 2))
] + [
    triton.Config(
        {
            "BLOCK_M": 192,  # noqa: TR002 - FA3's D64 tile lowers TMA traffic.
            "BLOCK_N": 128,
            "NUM_BUFFERS": 3,
            "NUM_MMA_WARPS": 12,
            "NUM_MMA_GROUPS": 3,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    )
]


def _prune_configs_by_head_dim(configs, named_args, **kwargs):
    head_dim = kwargs["HEAD_DIM"]
    causal = kwargs["CAUSAL"]
    use_bm192 = kwargs["USE_BM192"]
    block_n = 128 if head_dim == 64 else head_dim
    num_buffers = 3 if head_dim == 64 and not causal else 2
    block_m = 192 if use_bm192 else _DEFAULT_BLOCK_M
    return [
        config for config in configs if config.kwargs["BLOCK_M"] == block_m and config.kwargs["BLOCK_N"] == block_n
        and config.kwargs["NUM_BUFFERS"] == num_buffers
    ]


@triton.jit
def _compute_offsets(
    tile_idx,
    H,
    num_pid_n,
    num_pid_in_group,
    N_CTX,
    BLOCK_M: tl.constexpr,
    CAUSAL: tl.constexpr,
    ROW_REVERSE_HEAD_GROUP: tl.constexpr,
    ROW_REVERSE_MAX: tl.constexpr,
    ROW_AFFINE_MUL: tl.constexpr,
    ROW_AFFINE_ADD: tl.constexpr,
    ROW_AFFINE_MOD: tl.constexpr,
    ROW_SWAP_SRC_0: tl.constexpr,
    ROW_SWAP_SRC_1: tl.constexpr,
    ROW_SWAP_XOR: tl.constexpr,
):
    group_id = tile_idx // num_pid_in_group
    first_pid_n = group_id
    off_hz = first_pid_n
    start_m = tile_idx % num_pid_in_group
    if CAUSAL:
        if ROW_REVERSE_HEAD_GROUP != 0:
            reverse_rows = ((off_hz // ROW_REVERSE_HEAD_GROUP) & 1) != 0
            start_m = tl.where(reverse_rows, ROW_REVERSE_MAX - start_m, start_m)

        if ROW_AFFINE_MOD != 0:
            source_m = start_m
            mapped_m = (source_m * ROW_AFFINE_MUL + ROW_AFFINE_ADD) % ROW_AFFINE_MOD
            if ROW_SWAP_XOR != 0:
                swap_rows = (source_m == ROW_SWAP_SRC_0) | (source_m == ROW_SWAP_SRC_1)
                mapped_m = tl.where(swap_rows, mapped_m ^ ROW_SWAP_XOR, mapped_m)
            start_m = mapped_m
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    lo = 0
    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M) if CAUSAL else N_CTX
    kv_offset_y = offset_y + lo
    return start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y


@triton.autotune(
    configs=configs,
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "CAUSAL", "USE_BM192"],
    prune_configs_by={"early_config_prune": _prune_configs_by_head_dim},
)
@triton.jit
def _attn_fwd_ws_pipelined_pingpong(sm_scale, M,  #
                                    Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                                    HEAD_DIM: tl.constexpr,  #
                                    BLOCK_M: tl.constexpr,  #
                                    BLOCK_N: tl.constexpr,  #
                                    FP8_OUTPUT: tl.constexpr,  #
                                    CAUSAL: tl.constexpr,  #
                                    USE_BM192: tl.constexpr,  #
                                    ROW_REVERSE_HEAD_GROUP: tl.constexpr,  #
                                    ROW_REVERSE_MAX: tl.constexpr,  #
                                    ROW_AFFINE_MUL: tl.constexpr,  #
                                    ROW_AFFINE_ADD: tl.constexpr,  #
                                    ROW_AFFINE_MOD: tl.constexpr,  #
                                    ROW_SWAP_SRC_0: tl.constexpr,  #
                                    ROW_SWAP_SRC_1: tl.constexpr,  #
                                    ROW_SWAP_XOR: tl.constexpr,  #
                                    STEADY_UNROLL: tl.constexpr,  #
                                    NUM_BUFFERS: tl.constexpr,  #
                                    NUM_MMA_WARPS: tl.constexpr,  #
                                    NUM_MMA_GROUPS: tl.constexpr,  #
                                    ):
    tl.static_assert(BLOCK_N <= HEAD_DIM or (HEAD_DIM == 64 and BLOCK_N == 128))
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    SERIALIZE_QK: tl.constexpr = CAUSAL or HEAD_DIM == 128
    USE_SCHEDULER_BARRIER: tl.constexpr = USE_BM192
    USE_K_AHEAD: tl.constexpr = not CAUSAL and HEAD_DIM == 64 and not USE_BM192
    CONSUMER_REGISTERS: tl.constexpr = 160 if USE_BM192 else _FWD_CONSUMER_REGISTERS

    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))

    # Persistent kernel setup
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = Z * H
    num_pid_in_group = num_pid_m
    total_tiles = num_pid_m * Z * H

    tiles_per_prog = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_prog += 1

    tile_idx = prog_id

    # allocate buffers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS)
    k_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS)
    v_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS)

    # allocate barriers
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    q_empties = tlx.alloc_warp_barrier(
        num_barriers=NUM_MMA_GROUPS,
        num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
        num_arrivals=1,
    )
    k_empties = tlx.alloc_warp_barrier(
        num_barriers=NUM_BUFFERS,
        num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
        num_arrivals=NUM_MMA_GROUPS,
    )
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v_empties = tlx.alloc_warp_barrier(
        num_barriers=NUM_BUFFERS,
        num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
        num_arrivals=NUM_MMA_GROUPS,
    )
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)

    with tlx.async_tasks(exclusive=True):
        # producer group
        with tlx.async_task("default"):
            if not CAUSAL and HEAD_DIM == 64:
                if tlx.thread_id(0) == 0:
                    tlx.prefetch(desc_q, tensormap=True)
                    tlx.prefetch(desc_k, tensormap=True)
                    tlx.prefetch(desc_v, tensormap=True)
                    tlx.prefetch(desc_o, tensormap=True)
            accum_cnt_kv = 0

            for i in range(0, tiles_per_prog):
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_idx,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    CAUSAL=CAUSAL,
                    ROW_REVERSE_HEAD_GROUP=ROW_REVERSE_HEAD_GROUP,
                    ROW_REVERSE_MAX=ROW_REVERSE_MAX,
                    ROW_AFFINE_MUL=ROW_AFFINE_MUL,
                    ROW_AFFINE_ADD=ROW_AFFINE_ADD,
                    ROW_AFFINE_MOD=ROW_AFFINE_MOD,
                    ROW_SWAP_SRC_0=ROW_SWAP_SRC_0,
                    ROW_SWAP_SRC_1=ROW_SWAP_SRC_1,
                    ROW_SWAP_XOR=ROW_SWAP_XOR,
                )

                # Give QK priority on the D64 noncausal producer TMA stream.
                # Other paths retain their established K/V issue order.
                if USE_K_AHEAD:
                    kv_buf_id, kv_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)
                    kv_offset = kv_offset_y + lo
                    tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                    tlx.barrier_expect_bytes(k_fulls[kv_buf_id], K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    tlx.async_descriptor_load(desc_k, k_tiles[kv_buf_id], [kv_offset, 0], k_fulls[kv_buf_id])

                _, q_phase = get_bufidx_phase(i, 1)
                q_cid: tl.constexpr = 0
                tlx.barrier_wait(q_empties[q_cid], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_cid], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
                qo_offset_y_split = qo_offset_y + q_cid * BLOCK_M_SPLIT
                if USE_BM192:
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_cid],
                        [off_hz, start_m * BLOCK_M + q_cid * BLOCK_M_SPLIT, 0],
                        q_fulls[q_cid],
                    )
                else:
                    tlx.async_descriptor_load(desc_q, q_tiles[q_cid], [qo_offset_y_split, 0], q_fulls[q_cid])

                for cid in tl.range(1, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                    # TR051 heuristically unrolls the persistent loop and misses the matching consumer arrival.
                    tlx.barrier_wait(q_empties[cid], q_phase ^ 1)  # noqa: TR051
                    tlx.barrier_expect_bytes(q_fulls[cid], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
                    qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                    if USE_BM192:
                        tlx.async_descriptor_load(
                            desc_q,
                            q_tiles[cid],
                            [off_hz, start_m * BLOCK_M + cid * BLOCK_M_SPLIT, 0],
                            q_fulls[cid],
                        )
                    else:
                        tlx.async_descriptor_load(desc_q, q_tiles[cid], [qo_offset_y_split, 0], q_fulls[cid])

                if USE_K_AHEAD:
                    for kv_idx in tl.range(lo + BLOCK_N, hi, BLOCK_N, loop_unroll_factor=STEADY_UNROLL):
                        next_kv_count = accum_cnt_kv + 1
                        k_buf_id, k_phase = get_bufidx_phase(next_kv_count, NUM_BUFFERS)
                        k_offset = kv_offset_y + kv_idx
                        tlx.barrier_wait(k_empties[k_buf_id], k_phase ^ 1)
                        tlx.barrier_expect_bytes(k_fulls[k_buf_id], K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                        tlx.async_descriptor_load(desc_k, k_tiles[k_buf_id], [k_offset, 0], k_fulls[k_buf_id])

                        v_buf_id, v_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)
                        v_offset = kv_offset_y + kv_idx - BLOCK_N
                        tlx.barrier_wait(v_empties[v_buf_id], v_phase ^ 1)
                        tlx.barrier_expect_bytes(v_fulls[v_buf_id], V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                        tlx.async_descriptor_load(desc_v, v_tiles[v_buf_id], [v_offset, 0], v_fulls[v_buf_id])
                        accum_cnt_kv = next_kv_count

                    v_buf_id, v_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)
                    v_offset = kv_offset_y + hi - BLOCK_N
                    tlx.barrier_wait(v_empties[v_buf_id], v_phase ^ 1)
                    tlx.barrier_expect_bytes(v_fulls[v_buf_id], V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    tlx.async_descriptor_load(desc_v, v_tiles[v_buf_id], [v_offset, 0], v_fulls[v_buf_id])
                    accum_cnt_kv += 1
                else:
                    kv_buf_id, kv_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)
                    kv_offset = kv_offset_y + lo
                    tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                    tlx.barrier_expect_bytes(k_fulls[kv_buf_id], K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    if USE_BM192:
                        tlx.async_descriptor_load(desc_k, k_tiles[kv_buf_id], [off_hz, lo, 0], k_fulls[kv_buf_id])
                    else:
                        tlx.async_descriptor_load(desc_k, k_tiles[kv_buf_id], [kv_offset, 0], k_fulls[kv_buf_id])

                    tlx.barrier_wait(v_empties[kv_buf_id], kv_phase ^ 1)
                    tlx.barrier_expect_bytes(v_fulls[kv_buf_id], V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    if USE_BM192:
                        tlx.async_descriptor_load(desc_v, v_tiles[kv_buf_id], [off_hz, lo, 0], v_fulls[kv_buf_id])
                    else:
                        tlx.async_descriptor_load(desc_v, v_tiles[kv_buf_id], [kv_offset, 0], v_fulls[kv_buf_id])
                    accum_cnt_kv += 1

                    for kv_idx in tl.range(lo + BLOCK_N, hi, BLOCK_N, loop_unroll_factor=STEADY_UNROLL):
                        kv_buf_id, kv_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)
                        kv_offset = kv_offset_y + kv_idx
                        tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                        tlx.barrier_expect_bytes(k_fulls[kv_buf_id], K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                        if USE_BM192:
                            tlx.async_descriptor_load(
                                desc_k,
                                k_tiles[kv_buf_id],
                                [off_hz, kv_idx, 0],
                                k_fulls[kv_buf_id],
                            )
                        else:
                            tlx.async_descriptor_load(desc_k, k_tiles[kv_buf_id], [kv_offset, 0], k_fulls[kv_buf_id])

                        tlx.barrier_wait(v_empties[kv_buf_id], kv_phase ^ 1)
                        tlx.barrier_expect_bytes(v_fulls[kv_buf_id], V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                        if USE_BM192:
                            tlx.async_descriptor_load(
                                desc_v,
                                v_tiles[kv_buf_id],
                                [off_hz, kv_idx, 0],
                                v_fulls[kv_buf_id],
                            )
                        else:
                            tlx.async_descriptor_load(desc_v, v_tiles[kv_buf_id], [kv_offset, 0], v_fulls[kv_buf_id])
                        accum_cnt_kv += 1

                tile_idx += num_progs

        # consumer group
        with tlx.async_task(
                num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
                registers=CONSUMER_REGISTERS,
                replicate=NUM_MMA_GROUPS,
        ):
            accum_cnt_kv = 0
            cid: tl.constexpr = tlx.async_task_replica_id()

            # Bootstrap FA3's pairwise three-warpgroups scheduler ring.
            if USE_SCHEDULER_BARRIER and cid == 0:
                tlx.named_barrier_arrive(11, 256)

            # Bootstrap the pingpong sequence once before the persistent tile loop.
            if SERIALIZE_QK and cid == 1:
                tlx.named_barrier_arrive(9, 256)

            for i in range(0, tiles_per_prog):
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_idx,
                    H,
                    num_pid_n,
                    num_pid_in_group,
                    N_CTX,
                    BLOCK_M,
                    CAUSAL=CAUSAL,
                    ROW_REVERSE_HEAD_GROUP=ROW_REVERSE_HEAD_GROUP,
                    ROW_REVERSE_MAX=ROW_REVERSE_MAX,
                    ROW_AFFINE_MUL=ROW_AFFINE_MUL,
                    ROW_AFFINE_ADD=ROW_AFFINE_ADD,
                    ROW_AFFINE_MOD=ROW_AFFINE_MOD,
                    ROW_SWAP_SRC_0=ROW_SWAP_SRC_0,
                    ROW_SWAP_SRC_1=ROW_SWAP_SRC_1,
                    ROW_SWAP_XOR=ROW_SWAP_XOR,
                )

                # initialize pointer to m and l
                m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)

                # load scales
                qk_scale = sm_scale
                qk_scale *= 1.44269504  # 1/log(2)

                # wait for the Q buffer to be populated by the producer
                _, q_phase = get_bufidx_phase(i, 1)
                tlx.barrier_wait(q_fulls[cid], q_phase)

                k_buf_id, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)

                # wait for the K[0] buffer to be populated by the producer
                tlx.barrier_wait(k_fulls[k_buf_id], k_phase)

                # -- compute qk[0] ----
                k_tile = tlx.local_trans(k_tiles[k_buf_id])

                if SERIALIZE_QK:
                    if cid == 0:
                        # Consumer 0 waits for Consumer 1 to reach synchronization point at barrier 9.
                        tlx.named_barrier_wait(9, 256)
                    else:
                        # Then waits at barrier 10 until Consumer 0 finishes issuing its async_dot.
                        tlx.named_barrier_wait(10, 256)

                qk = tlx.async_dot(q_tiles[cid], k_tile)

                if SERIALIZE_QK:
                    if cid == 0:
                        # After issuing async_dot, Consumer 0 signals barrier 10 to unblock Consumer 1.
                        tlx.named_barrier_arrive(10, 256)
                    else:
                        # Consumer 1 signals barrier 9 to unblock Consumer 0.
                        tlx.named_barrier_arrive(9, 256)

                # wait for the MMA to complete
                qk = tlx.async_dot_wait(0, qk)
                # release the K buffer
                tlx.barrier_arrive(k_empties[k_buf_id], 1)
                # The single-tile path has completed its final QK read.
                if lo + BLOCK_N == hi:
                    tlx.barrier_arrive(q_empties[cid], 1)

                # -- compute m_i and l_i ----
                if CAUSAL:
                    offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                    if lo + BLOCK_M >= hi:
                        offs_n = lo + tl.arange(0, BLOCK_N)
                        qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
                p = tl.math.exp2(qk)
                # -- compute correction factor
                alpha = tl.math.exp2(m_i - m_ij)
                # -- update output accumulator[0] --
                acc = acc * alpha[:, None]
                l_ij = tl.sum(p, 1)
                if USE_BM192:
                    p = p.to(tlx.dtype_of(desc_k))
                l_i = l_i * alpha + l_ij
                m_i = m_ij
                accum_cnt_kv += 1

                # Keep the steady-state loop branch-free. In the causal case,
                # peel every KV tile that intersects the BLOCK_M-wide diagonal
                # region. The max keeps the initial tile out of the later loops.
                steady_hi = tl.maximum(lo + BLOCK_N, hi - BLOCK_M) if CAUSAL else hi
                steady_tiles = (steady_hi - (lo + BLOCK_N)) // BLOCK_N
                paired_hi = steady_hi - (steady_tiles % 2) * BLOCK_N if CAUSAL else steady_hi
                for kv_idx in tl.range(
                        lo + BLOCK_N,
                        paired_hi,
                        BLOCK_N,
                        loop_unroll_factor=STEADY_UNROLL,
                ):
                    k_buf_id, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)

                    # WG0 publishes K readiness to the other consumers through
                    # FA3's pairwise scheduler ring.
                    if not USE_SCHEDULER_BARRIER or cid == 0:
                        tlx.barrier_wait(k_fulls[k_buf_id], k_phase)
                    if USE_SCHEDULER_BARRIER:
                        if cid == 0:
                            tlx.named_barrier_wait(11, 256)
                        elif cid == 1:
                            tlx.named_barrier_wait(12, 256)
                        else:
                            tlx.named_barrier_wait(13, 256)

                    # compute qk for the current iteration
                    k_tile = tlx.local_trans(k_tiles[k_buf_id])

                    if SERIALIZE_QK:
                        if cid == 0:
                            # Consumer 0 waits for Consumer 1 to reach synchronization point at barrier 9.
                            tlx.named_barrier_wait(9, 256)
                        else:
                            # Then waits at barrier 10 until Consumer 0 finishes issuing its async_dot.
                            tlx.named_barrier_wait(10, 256)

                    qk = tlx.async_dot(q_tiles[cid], k_tile)

                    if SERIALIZE_QK:
                        if cid == 0:
                            # After issuing async_dot, Consumer 0 signals barrier 10 to unblock Consumer 1.
                            tlx.named_barrier_arrive(10, 256)
                        else:
                            # Consumer 1 signals barrier 9 to unblock Consumer 0.
                            tlx.named_barrier_arrive(9, 256)

                    # compute pv from the previous iteration
                    # wait for the previous V buffer to be populated by the producer
                    v_buf_id, v_phase = get_bufidx_phase(accum_cnt_kv - 1, NUM_BUFFERS)
                    if not USE_SCHEDULER_BARRIER or cid == 0:
                        tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
                    # prepare p and v for the dot
                    p = p.to(tlx.dtype_of(desc_k))
                    acc = tlx.async_dot(p, v_tiles[v_buf_id], acc)
                    if USE_SCHEDULER_BARRIER:
                        if cid == 0:
                            tlx.named_barrier_arrive(12, 256)
                        elif cid == 1:
                            tlx.named_barrier_arrive(13, 256)
                        else:
                            tlx.named_barrier_arrive(11, 256)

                    # wait for the current qk MMA to complete
                    qk = tlx.async_dot_wait(1, qk)
                    # release the K buffer
                    tlx.barrier_arrive(k_empties[k_buf_id], 1)

                    # -- compute m_i and l_i ----
                    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, None]
                    p = tl.math.exp2(qk)
                    # -- compute correction factor
                    alpha = tl.math.exp2(m_i - m_ij)
                    l_ij = tl.sum(p, 1)
                    if USE_BM192:
                        p = p.to(tlx.dtype_of(desc_k))
                    # update m_i and l_i
                    l_i = l_i * alpha + l_ij
                    m_i = m_ij

                    # -- update output accumulator --
                    # wait for the previous pv MMA to complete
                    acc = tlx.async_dot_wait(0, acc)
                    # release the V buffer
                    tlx.barrier_arrive(v_empties[v_buf_id], 1)
                    acc = acc * alpha[:, None]
                    accum_cnt_kv += 1

                if CAUSAL:
                    for kv_idx in tl.range(paired_hi, hi, BLOCK_N):
                        k_buf_id, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS)
                        tlx.barrier_wait(k_fulls[k_buf_id], k_phase)
                        k_tile = tlx.local_trans(k_tiles[k_buf_id])

                        if SERIALIZE_QK:
                            if cid == 0:
                                tlx.named_barrier_wait(9, 256)
                            else:
                                tlx.named_barrier_wait(10, 256)

                        qk = tlx.async_dot(q_tiles[cid], k_tile)

                        if SERIALIZE_QK:
                            if cid == 0:
                                tlx.named_barrier_arrive(10, 256)
                            else:
                                tlx.named_barrier_arrive(9, 256)

                        v_buf_id, v_phase = get_bufidx_phase(accum_cnt_kv - 1, NUM_BUFFERS)
                        tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
                        p = p.to(tlx.dtype_of(desc_k))
                        acc = tlx.async_dot(p, v_tiles[v_buf_id], acc)

                        qk = tlx.async_dot_wait(1, qk)
                        tlx.barrier_arrive(k_empties[k_buf_id], 1)
                        if kv_idx + BLOCK_N == hi:
                            tlx.barrier_arrive(q_empties[cid], 1)

                        if kv_idx + BLOCK_M >= hi:
                            offs_n = kv_idx + tl.arange(0, BLOCK_N)
                            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
                        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                        qk = qk * qk_scale - m_ij[:, None]
                        p = tl.math.exp2(qk)
                        alpha = tl.math.exp2(m_i - m_ij)
                        l_ij = tl.sum(p, 1)
                        l_i = l_i * alpha + l_ij
                        m_i = m_ij

                        acc = tlx.async_dot_wait(0, acc)
                        tlx.barrier_arrive(v_empties[v_buf_id], 1)
                        acc = acc * alpha[:, None]
                        accum_cnt_kv += 1

                # compute pv from the last iteration
                # wait for the V buffer to be populated by the producer
                v_buf_id, v_phase = get_bufidx_phase(accum_cnt_kv - 1, NUM_BUFFERS)
                tlx.barrier_wait(v_fulls[v_buf_id], v_phase)
                # prepare p and v for the dot
                p = p.to(tlx.dtype_of(desc_k))
                acc = tlx.async_dot(p, v_tiles[v_buf_id], acc)
                acc = tlx.async_dot_wait(1, acc)
                if not CAUSAL:
                    if lo + BLOCK_N != hi:
                        tlx.barrier_arrive(q_empties[cid], 1)
                inv_l_i = 1.0 / l_i
                m_i += tl.math.log2(l_i)
                offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                m_ptrs = M + off_hz * N_CTX + offs_m
                tl.store(m_ptrs, m_i, mask=offs_m < N_CTX if USE_BM192 else None)
                # wait for the final PV to complete before releasing V
                acc = tlx.async_dot_wait(0, acc)
                tlx.barrier_arrive(v_empties[v_buf_id], 1)

                # epilogue
                qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                acc = acc * inv_l_i[:, None]
                if USE_BM192:
                    o_tile = acc.to(tlx.dtype_of(desc_o)).reshape(1, BLOCK_M_SPLIT, HEAD_DIM)
                    desc_o.store(
                        [off_hz, start_m * BLOCK_M + cid * BLOCK_M_SPLIT, 0],
                        o_tile,
                    )
                else:
                    desc_o.store([qo_offset_y_split, 0], acc.to(tlx.dtype_of(desc_o)))

                tile_idx += num_progs


@triton.jit
# Triton TR001: backward preprocess uses the fixed 128-row FA tile from the wrapper.
def _attn_bwd_preprocess(  # noqa: TR001
        O,
        DO,
        Delta,
        DQ,
        N_CTX,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    mask = (off_m[:, None] < N_CTX) & (off_n[None, :] < HEAD_DIM)
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :],
        mask=mask,
        other=0.0,
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=off_m < N_CTX)
    tl.store(
        DQ + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :],
        tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32),
        mask=mask,
    )


def _host_descriptor_bwd_pre_hook(nargs):
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_MMA_GROUPS_BWD = nargs["NUM_MMA_GROUPS_BWD"]
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_do"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_dq"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_dk"].block_shape = [BLOCK_N // NUM_MMA_GROUPS_BWD, HEAD_DIM]
    nargs["desc_dv"].block_shape = [BLOCK_N // NUM_MMA_GROUPS_BWD, HEAD_DIM]


configs_bwd = [
    triton.Config(
        {
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 2,
            "NUM_BUFFERS_DQ_STORE": 2,
            "NUM_MMA_GROUPS_BWD": 2,
            "NUM_MMA_WARPS_BWD": 8,
            "BWD_REGISTERS": 240,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=_host_descriptor_bwd_pre_hook,
    ),
]


@triton.autotune(configs=configs_bwd, key=["N_CTX", "HEAD_DIM", "CAUSAL"])
@triton.jit
def _attn_bwd_tlx(
    desc_q,
    desc_k,
    desc_v,
    sm_scale,
    desc_do,
    desc_dq,
    desc_dk,
    desc_dv,
    M,
    D,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DQ_STORE: tl.constexpr,
    NUM_MMA_GROUPS_BWD: tl.constexpr,
    NUM_MMA_WARPS_BWD: tl.constexpr,
    BWD_REGISTERS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int32)
    pid = tl.program_id(0)

    M += off_chz
    D += off_chz

    row_base = off_chz
    start_n = pid * BLOCK_N
    global_start_n = row_base + start_n
    first_q_block = start_n // BLOCK_M if CAUSAL else 0
    num_steps = N_CTX // BLOCK_M - first_q_block
    CID_BLOCK_N: tl.constexpr = BLOCK_N // NUM_MMA_GROUPS_BWD

    kv_atom_layout: tl.constexpr = tlx.nv_mma_shared_layout_encoding(
        (CID_BLOCK_N, HEAD_DIM // NUM_MMA_GROUPS_BWD),
        [1, 0],
        tlx.dtype_of(desc_k),
        [1, 1],
        [1, 1],
        [1, 0],
        False,
        True,
    )
    kv_smem_layout: tl.constexpr = kv_atom_layout.tile_to_shape((BLOCK_N, HEAD_DIM))
    k_smem = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), 1, layout=kv_smem_layout)
    v_smem = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_v), 1, layout=kv_smem_layout)
    qdo_atom_layout: tl.constexpr = tlx.nv_mma_shared_layout_encoding(
        (BLOCK_M, HEAD_DIM // NUM_MMA_GROUPS_BWD),
        [1, 0],
        tlx.dtype_of(desc_q),
        [1, 1],
        [1, 1],
        [1, 0],
        False,
        True,
    )
    qdo_smem_layout: tl.constexpr = qdo_atom_layout.tile_to_shape((BLOCK_M, HEAD_DIM))
    q_smem = tlx.local_alloc((BLOCK_M, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q, layout=qdo_smem_layout)
    do_smem = tlx.local_alloc((BLOCK_M, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_Q, layout=qdo_smem_layout)
    dq_store_smem = tlx.local_alloc(
        (BLOCK_M, HEAD_DIM),
        tlx.dtype_of(desc_dq),
        NUM_BUFFERS_DQ_STORE * NUM_MMA_GROUPS_BWD,
    )
    score_atom_layout: tl.constexpr = tlx.nv_mma_shared_layout_encoding(
        (CID_BLOCK_N, BLOCK_M),
        [1, 0],
        tlx.dtype_of(desc_q),
        [1, 1],
        [1, 1],
        [1, 0],
        False,
        True,
    )
    score_smem_layout: tl.constexpr = score_atom_layout.tile_to_shape((BLOCK_N, BLOCK_M))
    score_smem_full = tlx.local_alloc(
        (BLOCK_N, BLOCK_M),
        tlx.dtype_of(desc_q),
        1,
        layout=score_smem_layout,
    )

    kv_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=NUM_MMA_GROUPS_BWD)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)

    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))
    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    DO_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_do))

    with tlx.async_tasks():
        with tlx.async_task("default"):
            k_load_view = tlx.local_reinterpret(k_smem[0], tlx.dtype_of(desc_k), [BLOCK_N, HEAD_DIM],
                                                layout=kv_atom_layout, pin=False)
            v_load_view = tlx.local_reinterpret(v_smem[0], tlx.dtype_of(desc_v), [BLOCK_N, HEAD_DIM],
                                                layout=kv_atom_layout, pin=False)
            tlx.barrier_expect_bytes(
                kv_full[0],
                BLOCK_N * HEAD_DIM * (K_BYTES_PER_ELEM + V_BYTES_PER_ELEM),
            )
            tlx.async_descriptor_load(desc_k, k_load_view, [global_start_n, 0], kv_full[0])
            tlx.async_descriptor_load(desc_v, v_load_view, [global_start_n, 0], kv_full[0])

            for q_iter in range(num_steps):
                q_buf = q_iter % NUM_BUFFERS_Q
                q_phase = (q_iter // NUM_BUFFERS_Q) & 1
                blk = first_q_block + q_iter
                start_m = blk * BLOCK_M
                global_start_m = row_base + start_m
                q_load_view = tlx.local_reinterpret(
                    q_smem[q_buf],
                    tlx.dtype_of(desc_q),
                    [BLOCK_M, HEAD_DIM],
                    layout=qdo_atom_layout,
                    pin=False,
                )
                do_load_view = tlx.local_reinterpret(
                    do_smem[q_buf],
                    tlx.dtype_of(desc_do),
                    [BLOCK_M, HEAD_DIM],
                    layout=qdo_atom_layout,
                    pin=False,
                )
                # Each consumer releases this slot only after its final Q and dO reads.
                tlx.barrier_wait(q_empties[q_buf], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_buf], BLOCK_M * HEAD_DIM * Q_BYTES_PER_ELEM)
                tlx.async_descriptor_load(desc_q, q_load_view, [global_start_m, 0], q_fulls[q_buf])

                tlx.barrier_expect_bytes(do_fulls[q_buf], BLOCK_M * HEAD_DIM * DO_BYTES_PER_ELEM)
                tlx.async_descriptor_load(desc_do, do_load_view, [global_start_m, 0], do_fulls[q_buf])

        with tlx.async_task(
                num_warps=NUM_MMA_WARPS_BWD // NUM_MMA_GROUPS_BWD,
                registers=BWD_REGISTERS,
                replicate=NUM_MMA_GROUPS_BWD,
        ):
            cid: tl.constexpr = tlx.async_task_replica_id()
            cid_start_n: tl.constexpr = cid * CID_BLOCK_N
            k_view = tlx.local_reinterpret(k_smem[0], tlx.dtype_of(desc_k), [BLOCK_N, HEAD_DIM], layout=kv_atom_layout,
                                           pin=False)
            v_view = tlx.local_reinterpret(v_smem[0], tlx.dtype_of(desc_v), [BLOCK_N, HEAD_DIM], layout=kv_atom_layout,
                                           pin=False)
            score_view = tlx.local_reinterpret(
                score_smem_full[0],
                tlx.dtype_of(desc_q),
                [BLOCK_N, BLOCK_M],
                layout=score_atom_layout,
                pin=False,
            )
            score_view_t = tlx.local_trans(score_view)
            k_slice = tlx.local_slice(k_view, [cid_start_n, 0], [CID_BLOCK_N, HEAD_DIM])
            v_slice = tlx.local_slice(v_view, [cid_start_n, 0], [CID_BLOCK_N, HEAD_DIM])
            score_smem = tlx.local_slice(score_view, [cid_start_n, 0], [CID_BLOCK_N, BLOCK_M])
            score_smem_t = tlx.local_slice(score_view_t, [0, cid_start_n], [BLOCK_M, CID_BLOCK_N])
            tlx.barrier_wait(kv_full[0], 0)
            dk = tl.zeros([CID_BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv = tl.zeros([CID_BLOCK_N, HEAD_DIM], dtype=tl.float32)
            for q_iter in range(num_steps):
                q_buf = q_iter % NUM_BUFFERS_Q
                q_phase = (q_iter // NUM_BUFFERS_Q) & 1
                blk = first_q_block + q_iter
                start_m = blk * BLOCK_M
                offs_m = start_m + tl.arange(0, BLOCK_M)

                tlx.barrier_wait(q_fulls[q_buf], q_phase)
                q = tlx.local_reinterpret(
                    q_smem[q_buf],
                    tlx.dtype_of(desc_q),
                    [BLOCK_M, HEAD_DIM],
                    layout=qdo_atom_layout,
                    pin=False,
                )
                qT = tlx.local_trans(q)

                qkT = tlx.async_dot(k_slice, qT)
                valid_m = offs_m < N_CTX
                m = tl.load(M + offs_m, mask=valid_m)
                Di = tl.load(D + offs_m, mask=valid_m)
                tlx.barrier_wait(do_fulls[q_buf], q_phase)
                do = tlx.local_reinterpret(
                    do_smem[q_buf],
                    tlx.dtype_of(desc_do),
                    [BLOCK_M, HEAD_DIM],
                    layout=qdo_atom_layout,
                    pin=False,
                )
                doT = tlx.local_trans(do)
                qkT = tlx.async_dot_wait(0, qkT)
                qkT *= sm_scale * RCP_LN2
                if CAUSAL and start_m < start_n + BLOCK_N:
                    offs_n = start_n + cid_start_n + tl.arange(0, CID_BLOCK_N)
                    qkT = tl.where(offs_m[None, :] >= offs_n[:, None], qkT, -float("inf"))
                pT = tl.math.exp2(qkT - m[None, :])
                dpT = tlx.async_dot(v_slice, doT)
                dv = tlx.async_dot(pT.to(tlx.dtype_of(desc_q)), do, dv)
                dpT = tlx.async_dot_wait(1, dpT).to(tl.float32)
                dsT = pT * (dpT - Di[None, :])
                dk = tlx.async_dot(dsT.to(tlx.dtype_of(desc_q)), q, dk)
                dv = tlx.async_dot_wait(1, dv)
                tlx.local_store(score_smem, dsT.to(tlx.dtype_of(desc_q)))
                tlx.fence_async_shared()

                dq = tlx.async_dot(score_smem_t, k_slice)
                dk = tlx.async_dot_wait(1, dk)
                tlx.barrier_arrive(q_empties[q_buf], 1)
                dq = tlx.async_dot_wait(0, dq)
                dq *= sm_scale
                dq_store_buf = cid * NUM_BUFFERS_DQ_STORE + q_iter % NUM_BUFFERS_DQ_STORE
                tlx.async_descriptor_store_wait(NUM_BUFFERS_DQ_STORE - 1)
                tlx.local_store(dq_store_smem[dq_store_buf], dq.to(tlx.dtype_of(desc_dq)))
                tlx.fence_async_shared()
                tlx.async_descriptor_store(
                    desc_dq,
                    dq_store_smem[dq_store_buf],
                    [row_base + start_m, 0],
                    store_reduce="add",
                )

            dk *= sm_scale
            tlx.async_descriptor_store_wait(0)
            dkv_store_buf: tl.constexpr = cid * NUM_BUFFERS_DQ_STORE
            tlx.local_store(dq_store_smem[dkv_store_buf], dk.to(tlx.dtype_of(desc_dk)))
            tlx.fence_async_shared()
            tlx.async_descriptor_store(
                desc_dk,
                dq_store_smem[dkv_store_buf],
                [global_start_n + cid_start_n, 0],
            )
            tlx.async_descriptor_store_wait(0)
            tlx.local_store(dq_store_smem[dkv_store_buf], dv.to(tlx.dtype_of(desc_dv)))
            tlx.fence_async_shared()
            tlx.async_descriptor_store(
                desc_dv,
                dq_store_smem[dkv_store_buf],
                [global_start_n + cid_start_n, 0],
            )
            tlx.async_descriptor_store_wait(0)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal, config):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in (64, 128)
        assert q.shape[2] % _DEFAULT_BLOCK_M == 0
        o = torch.empty_like(q)
        extra_kern_args = {}
        use_bm192 = (config is None and not causal and q.dtype == torch.bfloat16 and q.shape[0] == 4
                     and q.shape[1] == 48 and q.shape[2] in (1024, 2048, 4096, 8192) and q.shape[3] == 64)

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Rank-3 descriptors safely truncate BM192's final 128-row output tile.
        if use_bm192:
            batch_heads = q.shape[0] * q.shape[1]
            dummy_block = [1, 1, 1]
            desc_shape = [batch_heads, q.shape[2], HEAD_DIM_K]
            desc_strides = [q.shape[2] * HEAD_DIM_K, HEAD_DIM_K, 1]
            desc_q = TensorDescriptor(q, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
            desc_v = TensorDescriptor(v, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=desc_shape, strides=desc_strides, block_shape=dummy_block)
        else:
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1],
                                          block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                                          block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        NUM_SMS = torch.cuda.get_device_properties(q.device).multi_processor_count
        if config is not None:
            config = dict(config)
        block_m = 192 if use_bm192 else (_DEFAULT_BLOCK_M if config is None else config["BLOCK_M"])
        row_schedule, steady_unroll, target_workers = _select_forward_policy(
            causal,
            q.shape,
            q.dtype,
            block_m,
            NUM_SMS,
        )

        def grid(META):
            total_tiles = triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1]
            return (min(target_workers, total_tiles), 1, 1)

        ctx.grid = grid
        if config is None:
            _attn_fwd_ws_pipelined_pingpong[grid](
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
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                CAUSAL=causal,  #
                USE_BM192=use_bm192,  #
                STEADY_UNROLL=steady_unroll,  #
                **row_schedule,  #
                **extra_kern_args,
            )
        else:
            nargs = {
                **config,
                "HEAD_DIM": HEAD_DIM_K,
                "desc_q": desc_q,
                "desc_k": desc_k,
                "desc_v": desc_v,
                "desc_o": desc_o,
                "FP8_OUTPUT": q.dtype == torch.float8_e5m2,
                "USE_BM192": use_bm192,
            }
            _host_descriptor_pre_hook(nargs)
            _attn_fwd_ws_pipelined_pingpong.fn[grid](
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
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,
                CAUSAL=causal,
                USE_BM192=use_bm192,
                STEADY_UNROLL=steady_unroll,
                **row_schedule,
                **config,
            )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        BLOCK_N = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o,
            do,
            delta,
            dq,
            N_CTX,
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,
        )
        grid = (N_CTX // BLOCK_N, 1, BATCH * N_HEAD)
        y_dim = BATCH * N_HEAD * N_CTX
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_v = TensorDescriptor(v, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_do = TensorDescriptor(do, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_dq = TensorDescriptor(dq, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_dk = TensorDescriptor(dk, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        desc_dv = TensorDescriptor(dv, shape=[y_dim, ctx.HEAD_DIM], strides=[ctx.HEAD_DIM, 1], block_shape=dummy_block)
        _attn_bwd_tlx[grid](
            desc_q,
            desc_k,
            desc_v,
            ctx.sm_scale,
            desc_do,
            desc_dq,
            desc_dk,
            desc_dv,
            M,
            delta,
            N_HEAD,
            N_CTX,
            HEAD_DIM=ctx.HEAD_DIM,
            CAUSAL=ctx.causal,
        )

        return dq, dk, dv, None, None, None


def attention(q, k, v, sm_scale, causal=False, config=None):
    if isinstance(causal, dict) and config is None:
        config = causal
        causal = False
    return _attention.apply(q, k, v, sm_scale, causal, config)


def flash_attn(q, k, v, causal=False, sm_scale=None, *, space="full"):
    """Fused D64/D128 attention over ``(Z, H, N_CTX, HEAD_DIM)``."""
    # SM90 uses one head-dimension-specific configuration in both spaces.
    if space not in ("full", "smoke"):
        raise ValueError(f"space must be 'full' or 'smoke', got {space!r}")
    if sm_scale is None:
        sm_scale = q.shape[-1]**-0.5
    return _attention.apply(q, k, v, sm_scale, causal, None)
