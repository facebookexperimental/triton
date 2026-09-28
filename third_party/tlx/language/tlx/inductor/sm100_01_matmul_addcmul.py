# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# Fused matmul + bias + store_m + addcmul kernel.
# Epilogue: m = s @ W.T + b2; out = layer_input + x0 * m (all in FP32 on accumulator).
# Adapted from GEO's 1152x1024x12800 production kernel.

# pyre-ignore-all-errors

import functools
import os

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import triton.language.extra.tlx as tlx  # @manual=//triton:triton
from torch.library import triton_op, wrap_triton
from triton.tools.tensor_descriptor import TensorDescriptor  # @manual=//triton:triton

# Proton is deliberately disabled in the TorchInductor integration.  Importing
# its optional extension would make ordinary TorchTLX registration depend on a
# matching profiler build even though all profiling branches specialize away.
pl = None


@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def get_heuristic_config(*_args):
    return None


def get_cuda_autotune_config():  # noqa: C901
    # The production-shape benchmark pins this configuration.  Keep one valid
    # autotune entry for the decorated kernel; the TorchTLX wrapper below uses
    # ``matmul_addcmul_kernel.fn`` with the same fixed configuration directly.
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 64,
                "NUM_SMEM_BUFFERS": 6,
                "NUM_TMEM_BUFFERS": 3,
                "NUM_MMA_GROUPS": 1,
                "EPILOGUE_SUBTILE": 4,
                "NUM_CTAS": 2,
                "SPLIT_K": 1,
                "USE_WARP_BARRIER": False,
            },
            num_warps=4,
            num_stages=1,
            ctas_per_cga=(2, 1, 1),
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ]


def preprocess_configs(configs, named_args, **kwargs):  # noqa: C901
    """Custom pruner for linear_addcmul__returns_linear.

    Replicates the base addmm pruner logic but:
    1. Uses actual SMEM (including epilogue buffers) instead of base estimate.
    2. Removes the BM_SPLIT=64 + 2CTA filter — the 4-task WS kernel
       handles this correctly (verified by correctness tests)."""
    import math

    NUM_SMS = _get_num_sms()
    MAX_SMEM = 232 * 1024
    MAX_TMEM = 256 * 1024

    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]

    pruned_configs = []
    for c in configs:
        BM = c.kwargs["BLOCK_SIZE_M"]
        BN = c.kwargs["BLOCK_SIZE_N"]
        BK = c.kwargs["BLOCK_SIZE_K"]
        s = c.kwargs["NUM_SMEM_BUFFERS"]
        t = c.kwargs["NUM_TMEM_BUFFERS"]
        NCTA = c.kwargs["NUM_CTAS"]
        NMG = c.kwargs["NUM_MMA_GROUPS"]
        SK = c.kwargs.get("SPLIT_K", 1)
        SUB = c.kwargs["EPILOGUE_SUBTILE"]
        GSM = c.kwargs["GROUP_SIZE_M"]

        BM_SPLIT = BM // NMG

        if BM_SPLIT > 128 or BM_SPLIT < 64:
            continue
        # BM_SPLIT=64 + 2CTA generally causes illegal memory access.
        # Exception: BM=128 BN=256 s<=4 t=1 NMG=2 SUB=8 (verified correct).
        if NCTA == 2 and BM_SPLIT == 64:
            if not (
                BM == 128 and BN == 256 and SUB == 8 and t == 1 and NMG == 2 and s <= 4
            ):
                continue
        if GSM % NCTA != 0:
            continue
        if BN % SUB != 0:
            continue
        if BM == 64 and math.ceil(M / 128) * math.ceil(N / 128) > 16:
            continue

        # Split-K gating
        num_mn_tiles = math.ceil(M / BM) * math.ceil(N / BN)
        if SK > 1:
            if num_mn_tiles >= NUM_SMS:
                continue
            k_tiles = math.ceil(K / BK)
            if k_tiles < SK:
                continue
            k_per_split = math.ceil(k_tiles / SK)
            if k_per_split * (SK - 1) >= k_tiles:
                continue
            if k_tiles // SK < 4:
                continue

        # SMEM: use actual epilogue buffers (not base estimate)
        slice_size = BN // SUB
        smem_a = BM * BK * 2 * s
        smem_b = BK * (BN // NCTA) * 2 * s
        NUM_X0LI_BUFS_EST = 3
        smem_x0li = 2 * NUM_X0LI_BUFS_EST * BM_SPLIT * slice_size * 2
        NUM_EPI_SMEM = NMG if NMG > 2 else 2
        smem_cm = 2 * NUM_EPI_SMEM * BM_SPLIT * slice_size * 2
        total_smem = smem_a + smem_b + smem_x0li + smem_cm
        if total_smem > MAX_SMEM:
            continue

        # TMEM
        if BM * BN * 4 * t > MAX_TMEM:
            continue

        pruned_configs.append(c)

    # Two-level Split-K filter
    if pruned_configs:

        def _total_tiles(c):
            return (
                math.ceil(M / c.kwargs["BLOCK_SIZE_M"])
                * math.ceil(N / c.kwargs["BLOCK_SIZE_N"])
                * c.kwargs.get("SPLIT_K", 1)
            )

        def _num_waves(c):
            return math.ceil(_total_tiles(c) / NUM_SMS)

        def _tile_key(c):
            return (
                c.kwargs["BLOCK_SIZE_M"],
                c.kwargs["BLOCK_SIZE_N"],
                c.kwargs["BLOCK_SIZE_K"],
            )

        tile_groups = {}
        for c in pruned_configs:
            tile_groups.setdefault(_tile_key(c), []).append(c)
        result = []
        for group_configs in tile_groups.values():
            min_waves = min(_num_waves(c) for c in group_configs)
            best = [c for c in group_configs if _num_waves(c) == min_waves]
            max_sk = max(c.kwargs.get("SPLIT_K", 1) for c in best)
            best = [c for c in best if c.kwargs.get("SPLIT_K", 1) == max_sk]
            result.extend(best)
        pruned_configs = result

    # GROUP_SIZE_M golden rule
    if pruned_configs:
        IMBALANCE_THRESHOLD = 10
        if M > N * IMBALANCE_THRESHOLD:
            pruned_configs = [
                c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] == 1
            ]
        elif N > M * IMBALANCE_THRESHOLD:
            pruned_configs = [
                c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] >= 32
            ]
        else:
            pruned_configs = [
                c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] == 8
            ]

    # Pareto-optimal filtering on (s, t, NMG)
    if pruned_configs:

        def _group_key(c):
            return (
                c.kwargs["BLOCK_SIZE_M"],
                c.kwargs["BLOCK_SIZE_N"],
                c.kwargs["BLOCK_SIZE_K"],
                c.kwargs["EPILOGUE_SUBTILE"],
                c.kwargs["NUM_CTAS"],
                c.kwargs.get("SPLIT_K", 1),
            )

        def _val(c):
            return (
                c.kwargs["NUM_SMEM_BUFFERS"],
                c.kwargs["NUM_TMEM_BUFFERS"],
                c.kwargs["NUM_MMA_GROUPS"],
            )

        def _dominates(a, b):
            va, vb = _val(a), _val(b)
            return all(x >= y for x, y in zip(va, vb)) and any(
                x > y for x, y in zip(va, vb)
            )

        groups = {}
        for c in pruned_configs:
            groups.setdefault(_group_key(c), []).append(c)
        pruned_configs = []
        for members in groups.values():
            for c in members:
                if not any(_dominates(other, c) for other in members if other is not c):
                    pruned_configs.append(c)

    return pruned_configs


def preprocess_configs_tma(configs, named_args, **kwargs):
    named_args = dict(named_args)
    named_args["_has_tma_epilogue"] = True
    return preprocess_configs(configs, named_args, **kwargs)


def _workspace_rows_per_split(
    m: int,
    block_m: int,
    num_ctas: int,
) -> int:
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_m = (num_pid_m + num_ctas - 1) // num_ctas * num_ctas
    return num_pid_m * block_m


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    NUM_MMA_GROUPS = nargs.get("NUM_MMA_GROUPS", 1)
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    BLOCK_N_PER_CTA = BLOCK_N // NUM_CTAS
    # For column-major inputs, TMA descriptor block shape matches the transposed view
    if nargs.get("A_ROW_MAJOR", True):
        nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    else:
        nargs["a_desc"].block_shape = [BLOCK_K, BLOCK_M_SPLIT]
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N_PER_CTA]
    else:
        nargs["b_desc"].block_shape = [BLOCK_N_PER_CTA, BLOCK_K]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    epi_block = [BLOCK_M // NUM_MMA_GROUPS, BLOCK_N // EPILOGUE_SUBTILE]
    nargs["c_desc"].block_shape = epi_block
    if "x0_desc" in nargs:
        nargs["x0_desc"].block_shape = [
            BLOCK_M // NUM_MMA_GROUPS,
            BLOCK_N // EPILOGUE_SUBTILE,
        ]
    if "li_desc" in nargs:
        nargs["li_desc"].block_shape = [
            BLOCK_M // NUM_MMA_GROUPS,
            BLOCK_N // EPILOGUE_SUBTILE,
        ]
    if "m_desc" in nargs:
        nargs["m_desc"].block_shape = epi_block
    SPLIT_K = nargs.get("SPLIT_K", 1)
    if SPLIT_K > 1:
        M = nargs["M"]
        N = nargs["N"]
        rows_per_split = _workspace_rows_per_split(M, BLOCK_M, NUM_CTAS)
        workspace = torch.empty(
            (SPLIT_K * rows_per_split, N),
            device=nargs["c_desc"].base.device,
            dtype=torch.float32,
        )
        nargs["workspace_desc"].base = workspace
        nargs["workspace_desc"].shape = list(workspace.shape)
    else:
        nargs["workspace_desc"].base = nargs["c_desc"].base
        nargs["workspace_desc"].shape = list(nargs["c_desc"].base.shape)
    nargs["workspace_desc"].block_shape = [
        BLOCK_M // NUM_MMA_GROUPS,
        BLOCK_N // EPILOGUE_SUBTILE,
    ]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit
def _compute_grid_info(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    SPLIT_K,
    NUM_CTAS: tl.constexpr,
):
    """Compute common grid information used across async tasks."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Pad num_pid_m to multiple of NUM_CTAS so CTA clusters tile evenly along M.
    num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_mn_tiles = num_pid_m * num_pid_n
    num_tiles = num_mn_tiles * SPLIT_K
    k_tiles_total = tl.cdiv(K, BLOCK_SIZE_K)
    return (
        start_pid,
        num_pid_m,
        num_pid_n,
        num_pid_in_group,
        num_mn_tiles,
        num_tiles,
        k_tiles_total,
    )


@triton.jit
def _fwd_epilogue_ops(
    result,  # FP32 accumulator from TMEM
    b2_ptr,
    x0_buf,
    li_buf,
    offs_bn_slice,
    N,
    m_desc,
    m_smem_buffers,
    m_store_idx,
    offs_am,
    slice_size: tl.constexpr,
    FUSE_ADDCMUL: tl.constexpr,
    ENABLE_PROTON: tl.constexpr = False,
    PROTON_ITER: tl.constexpr = 10,
    idx=0,
):
    """Apply bias, store m, load x0/li from SMEM buffers, compute out = li + x0 * m in FP32."""
    if FUSE_ADDCMUL:
        b2_offsets = offs_bn_slice + tl.arange(0, slice_size)
        b2_tile = tl.load(
            b2_ptr + b2_offsets,
            mask=b2_offsets < N,
            other=0.0,
        )
        result = result + b2_tile.to(tl.float32)

        m_bf16 = result.to(tl.bfloat16)
        m_smem = m_smem_buffers[m_store_idx]
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(m_smem, m_bf16)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(
            m_desc, m_smem, [offs_am, offs_bn_slice], eviction_policy="evict_first"
        )

        x0_tile = tlx.local_load(x0_buf)
        li_tile = tlx.local_load(li_buf)
        result = li_tile.to(tl.float32) + x0_tile.to(tl.float32) * result

    return result


@triton.jit
def _fwd_epilogue_ops_tma(
    result,
    b2_ptr,
    x0_buf,
    li_buf,
    offs_bn_slice,
    N,
    m_desc,
    m_smem_buffers,
    m_store_idx,
    offs_am,
    slice_size: tl.constexpr,
    FUSE_ADDCMUL: tl.constexpr,
    ENABLE_PROTON: tl.constexpr = False,
    PROTON_ITER: tl.constexpr = 10,
    idx=0,
):
    """TMA path: x0/li pre-loaded to SMEM by 4th warp group."""
    if FUSE_ADDCMUL:
        b2_offsets = offs_bn_slice + tl.arange(0, slice_size)
        b2_tile = tl.load(
            b2_ptr + b2_offsets,
            mask=b2_offsets < N,
            other=0.0,
        )
        result = result + b2_tile.to(tl.float32)

        m_bf16 = result.to(tl.bfloat16)
        m_smem = m_smem_buffers[m_store_idx]
        tlx.async_descriptor_store_wait(1)
        tlx.local_store(m_smem, m_bf16)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(
            m_desc, m_smem, [offs_am, offs_bn_slice], eviction_policy="evict_first"
        )

        x0_tile = tlx.local_load(x0_buf)
        li_tile = tlx.local_load(li_buf)
        result = li_tile.to(tl.float32) + x0_tile.to(tl.float32) * result

    return result


@triton.jit
def _process_tile_epilogue_inner(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    GROUP_SIZE_M,
    M,
    N,
    BLOCK_M_SPLIT,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    NUM_CTAS,
    EPILOGUE_SUBTILE,
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    SPLIT_K,
    c_desc,
    workspace_desc,
    c_smem_buffers,
    tmem_buffers,
    tmem_full_bars,
    tmem_empty_bars,
    cur_tmem_buf,
    tmem_read_phase,
    # Forward epilogue args
    b2_ptr,
    buffers_x0,
    buffers_li,
    x0li_full_bars,
    x0li_empty_bars,
    x0li_slice_cnt,
    NUM_X0LI_BUFS: tl.constexpr,
    m_desc,
    m_smem_buffers,
    FUSE_ADDCMUL: tl.constexpr,
    ENABLE_PROTON: tl.constexpr = False,
    PROTON_ITER: tl.constexpr = 10,
    idx=0,
):
    """Process epilogue for a single tile with optional fused addcmul."""
    mn_tile_id = tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N

    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    if SPLIT_K > 1:
        split_id = tile_id // num_mn_tiles
        out_desc = workspace_desc
        num_pid_m_padded = (
            (tl.cdiv(M, BLOCK_SIZE_M) + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
        )
        rows_per_split = num_pid_m_padded * BLOCK_SIZE_M
        row_base = split_id * rows_per_split
    else:
        out_desc = c_desc
        row_base = 0

    for group_id in tl.static_range(NUM_MMA_GROUPS):
        # Wait for TMEM first to free it for MMA
        buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

        if ENABLE_PROTON:
            pl.enter_scope("epi_wait_tmem")
        tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_read_phase)
        if ENABLE_PROTON:
            pl.exit_scope("epi_wait_tmem")

        # load the result from TMEM to registers
        acc_tmem = tmem_buffers[buf_idx]
        offs_am = pid_m * BLOCK_SIZE_M + group_id * BLOCK_M_SPLIT

        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
            if ENABLE_PROTON:
                pl.enter_scope("epi_tmem_load")
            acc_tmem_subslice = tlx.local_slice(
                acc_tmem,
                [0, slice_id * slice_size],
                [BLOCK_M_SPLIT, slice_size],
            )
            result = tlx.local_load(acc_tmem_subslice)
            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
            if ENABLE_PROTON:
                pl.exit_scope("epi_tmem_load")

            if FUSE_ADDCMUL:
                x0li_buf, x0li_phase = _get_bufidx_phase(x0li_slice_cnt, NUM_X0LI_BUFS)
                tlx.barrier_wait(x0li_full_bars[x0li_buf], x0li_phase)

            x0_buf_ref = buffers_x0[x0li_buf] if FUSE_ADDCMUL else buffers_x0[0]
            li_buf_ref = buffers_li[x0li_buf] if FUSE_ADDCMUL else buffers_li[0]
            result = _fwd_epilogue_ops(
                result,
                b2_ptr,
                x0_buf_ref,
                li_buf_ref,
                offs_bn + slice_id * slice_size,
                N,
                m_desc,
                m_smem_buffers,
                (group_id * EPILOGUE_SUBTILE + slice_id) % 2,
                offs_am,
                slice_size,
                FUSE_ADDCMUL,
                ENABLE_PROTON,
                PROTON_ITER,
                idx,
            )
            if ENABLE_PROTON:
                pl.enter_scope("epi_store_c")
            c = result.to(tlx.dtype_of(out_desc))
            c_smem = c_smem_buffers[(group_id * EPILOGUE_SUBTILE + slice_id) % 2]
            tlx.async_descriptor_store_wait(1)
            tlx.local_store(c_smem, c)
            tlx.fence_async_shared()
            tlx.async_descriptor_store(
                out_desc,
                c_smem,
                [row_base + offs_am, offs_bn + slice_id * slice_size],
                eviction_policy="evict_first",
            )
            if ENABLE_PROTON:
                pl.exit_scope("epi_store_c")

            if FUSE_ADDCMUL:
                tlx.barrier_arrive(x0li_empty_bars[x0li_buf])
                x0li_slice_cnt += 1

    # Wait for all TMA stores to complete
    tlx.async_descriptor_store_wait(0)

    return x0li_slice_cnt


@triton.jit
def _process_tile_mma_inner(  # noqa: C901
    k_tiles,
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    buffers_A,
    buffers_B,
    tmem_buffers,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    tmem_full_bars,
    cur_tmem_buf,
    tmem_empty_bars,
    tmem_write_phase,
    smem_accum_cnt,
    NUM_CTAS,
    cta_bars,
    pred_cta0,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    ENABLE_PROTON: tl.constexpr = False,
    PROTON_ITER: tl.constexpr = 10,
    idx=0,
):
    """Process MMA for a single tile over [k_tile_start, k_tile_end). Returns updated smem_accum_cnt."""
    local_k_tiles = k_tile_end - k_tile_start

    # Peeled first K-iteration: wait for data before acquiring TMEM
    buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

    if ENABLE_PROTON:
        pl.enter_scope("mma_wait_B")
    tlx.barrier_wait(B_smem_full_bars[buf], phase)
    if ENABLE_PROTON:
        pl.exit_scope("mma_wait_B")

    # Process first K iteration (peeled) with use_acc=False
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        a_buf = group_id * NUM_SMEM_BUFFERS + buf
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

        if ENABLE_PROTON:
            pl.enter_scope("mma_wait_A_tmem")
        tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
        cur_barrier_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.barrier_wait(tmem_empty_bars[cur_barrier_idx], tmem_write_phase ^ 1)
        if ENABLE_PROTON:
            pl.exit_scope("mma_wait_A_tmem")

        if NUM_CTAS == 2:
            tlx.barrier_arrive(cta_bars[a_buf], arrive_count=1, remote_cta_rank=0)
            tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_cta0)

        a_operand = (
            tlx.local_trans(buffers_A[a_buf]) if not A_ROW_MAJOR else buffers_A[a_buf]
        )
        b_operand = (
            tlx.local_trans(buffers_B[buf]) if not B_ROW_MAJOR else buffers_B[buf]
        )

        if ENABLE_PROTON:
            pl.enter_scope("mma_dot")
        tlx.async_dot(
            a_operand,
            b_operand,
            tmem_buffers[acc_buf],
            use_acc=False,
            mBarriers=[A_smem_empty_bars[a_buf]],
            two_ctas=NUM_CTAS == 2,
            out_dtype=tl.float32,
        )
        if ENABLE_PROTON:
            pl.exit_scope("mma_dot")

    smem_accum_cnt += 1

    # Remaining K iterations with use_acc=True
    for _ in range(1, local_k_tiles):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

        if ENABLE_PROTON:
            pl.enter_scope("mma_wait_B")
        tlx.barrier_wait(B_smem_full_bars[buf], phase)
        if ENABLE_PROTON:
            pl.exit_scope("mma_wait_B")

        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

            if ENABLE_PROTON:
                pl.enter_scope("mma_wait_A")
            tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
            if ENABLE_PROTON:
                pl.exit_scope("mma_wait_A")

            if NUM_CTAS == 2:
                tlx.barrier_arrive(cta_bars[a_buf], arrive_count=1, remote_cta_rank=0)
                tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_cta0)

            a_operand = (
                tlx.local_trans(buffers_A[a_buf])
                if not A_ROW_MAJOR
                else buffers_A[a_buf]
            )
            b_operand = (
                tlx.local_trans(buffers_B[buf]) if not B_ROW_MAJOR else buffers_B[buf]
            )

            if ENABLE_PROTON:
                pl.enter_scope("mma_dot")
            tlx.async_dot(
                a_operand,
                b_operand,
                tmem_buffers[acc_buf],
                use_acc=True,
                mBarriers=[A_smem_empty_bars[a_buf]],
                two_ctas=NUM_CTAS == 2,
                out_dtype=tl.float32,
            )
            if ENABLE_PROTON:
                pl.exit_scope("mma_dot")

        smem_accum_cnt += 1

    # Wait for last MMA to complete and signal epilogue
    if ENABLE_PROTON:
        pl.enter_scope("mma_signal_epi")
    last_buf, last_phase = _get_bufidx_phase(smem_accum_cnt - 1, NUM_SMEM_BUFFERS)
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        a_buf = group_id * NUM_SMEM_BUFFERS + last_buf
        tlx.barrier_wait(A_smem_empty_bars[a_buf], last_phase)
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.barrier_arrive(tmem_full_bars[acc_buf], 1)
    if ENABLE_PROTON:
        pl.exit_scope("mma_signal_epi")

    return smem_accum_cnt


@triton.jit
def _process_tile_producer_inner(
    tile_id,
    start_pid,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    GROUP_SIZE_M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    NUM_MMA_GROUPS,
    BLOCK_M_SPLIT,
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    smem_accum_cnt,
    NUM_CTAS,
    cluster_cta_rank,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    ENABLE_PROTON: tl.constexpr = False,
    PROTON_ITER: tl.constexpr = 10,
    idx=0,
):
    """Process TMA loads for a single tile with all subtiles over [k_tile_start, k_tile_end).
    Epilogue tensor loading (b2, x0, li) is handled by a separate 4th warp group."""
    mn_tile_id = tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))
    offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // NUM_CTAS)
    expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K // NUM_CTAS

    local_k_tiles = k_tile_end - k_tile_start

    # Iterate along K dimension for this split's range
    for k_idx in range(0, local_k_tiles):
        if ENABLE_PROTON:
            pl.enter_scope("prod_k_iter")
        k = k_tile_start + k_idx
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        offs_k = k * BLOCK_SIZE_K

        # Load A for the first group
        a_buf = buf
        if ENABLE_PROTON:
            pl.enter_scope("prod_wait_A_empty")
        tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
        if ENABLE_PROTON:
            pl.exit_scope("prod_wait_A_empty")
        offs_am = pid_m * BLOCK_SIZE_M
        tlx.barrier_expect_bytes(
            A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
        )
        if not A_ROW_MAJOR:
            tlx.async_descriptor_load(
                a_desc,
                buffers_A[a_buf],
                [offs_k, offs_am],
                A_smem_full_bars[a_buf],
                eviction_policy="evict_last",
            )
        else:
            tlx.async_descriptor_load(
                a_desc,
                buffers_A[a_buf],
                [offs_am, offs_k],
                A_smem_full_bars[a_buf],
                eviction_policy="evict_last",
            )

        # Load B once per K iteration (shared across all subtiles)
        last_a_buf = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS + buf
        tlx.barrier_wait(A_smem_empty_bars[last_a_buf], phase ^ 1)
        tlx.barrier_expect_bytes(B_smem_full_bars[buf], expected_bytes)
        if not B_ROW_MAJOR:
            tlx.async_descriptor_load(
                b_desc,
                buffers_B[buf],
                [offs_bn, offs_k],
                B_smem_full_bars[buf],
                eviction_policy="evict_last",
            )
        else:
            tlx.async_descriptor_load(
                b_desc,
                buffers_B[buf],
                [offs_k, offs_bn],
                B_smem_full_bars[buf],
                eviction_policy="evict_last",
            )

        # Load all remaining A subtiles for this K iteration
        for group_id in tl.static_range(1, NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf

            tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)

            offs_am2 = offs_am + group_id * BLOCK_M_SPLIT

            tlx.barrier_expect_bytes(
                A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
            )
            if not A_ROW_MAJOR:
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_A[a_buf],
                    [offs_k, offs_am2],
                    A_smem_full_bars[a_buf],
                    eviction_policy="evict_last",
                )
            else:
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_A[a_buf],
                    [offs_am2, offs_k],
                    A_smem_full_bars[a_buf],
                    eviction_policy="evict_last",
                )

        smem_accum_cnt += 1
        if ENABLE_PROTON:
            pl.exit_scope("prod_k_iter")

    return smem_accum_cnt


TORCH_DTYPE_TO_TRITON = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.jit
def _reduce_k_addcmul_kernel(
    workspace_ptr,
    out_ptr,
    m_ptr,
    bias_ptr,
    x0_ptr,
    layer_input_ptr,
    M,
    N,
    WORKSPACE_ROWS_PER_SPLIT,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        ws_offs = base_offs + s * WORKSPACE_ROWS_PER_SPLIT * N
        partial = tl.load(workspace_ptr + ws_offs, mask=mask, other=0.0)
        acc += partial.to(tl.float32)

    bias = tl.load(bias_ptr + offs_n[None, :], mask=offs_n[None, :] < N, other=0.0)
    m = acc + bias.to(tl.float32)
    x0 = tl.load(x0_ptr + base_offs, mask=mask, other=0.0)
    layer_input = tl.load(layer_input_ptr + base_offs, mask=mask, other=0.0)
    out = layer_input.to(tl.float32) + x0.to(tl.float32) * m
    tl.store(m_ptr + base_offs, m.to(OUTPUT_DTYPE), mask=mask)
    tl.store(out_ptr + base_offs, out.to(OUTPUT_DTYPE), mask=mask)


def reduce_post_hook(nargs, exception=None):
    if exception is not None:
        return
    split_k = nargs.get("SPLIT_K", 1)
    if split_k > 1:
        M = nargs["M"]
        N = nargs["N"]
        workspace = nargs["workspace_desc"].base
        c = nargs["c_desc"].base
        rows_per_split = _workspace_rows_per_split(
            M,
            nargs["BLOCK_SIZE_M"],
            nargs.get("NUM_CTAS", 1),
        )
        reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
        _reduce_k_addcmul_kernel[reduce_grid](
            workspace,
            c,
            nargs["m_desc"].base,
            nargs["b2_ptr"],
            nargs["x0_desc"].base,
            nargs["li_desc"].base,
            M,
            N,
            rows_per_split,
            SPLIT_K=split_k,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[c.dtype],
            num_warps=4,
        )


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": preprocess_configs},
    post_hook=reduce_post_hook,
)
@triton.jit
def matmul_addcmul_kernel(  # noqa: C901
    a_desc,
    b_desc,
    c_desc,
    workspace_desc,
    b2_ptr,
    x0_desc,
    li_desc,
    m_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    USE_WARP_BARRIER: tl.constexpr = False,
    FUSE_ADDCMUL: tl.constexpr = True,
    ENABLE_PROTON: tl.constexpr = False,
    PROTON_ITER: tl.constexpr = 10,
):
    # allocate NUM_SMEM_BUFFERS buffers
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    if not A_ROW_MAJOR:
        buffers_A = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_M_SPLIT),
            tlx.dtype_of(a_desc),
            NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        )
    else:
        buffers_A = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_SIZE_K),
            tlx.dtype_of(a_desc),
            NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        )
    # In 2-CTA mode, each CTA only needs to load BLOCK_N // NUM_CTAS of B.
    if not B_ROW_MAJOR:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_N // NUM_CTAS, BLOCK_SIZE_K),
            tlx.dtype_of(b_desc),
            NUM_SMEM_BUFFERS,
        )
    else:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_SIZE_N // NUM_CTAS),
            tlx.dtype_of(b_desc),
            NUM_SMEM_BUFFERS,
        )
    # NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    # Each buffer holds one subtile: BLOCK_M_SPLIT x BLOCK_SIZE_N
    # Total buffers: NUM_TMEM_BUFFERS * NUM_MMA_GROUPS
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )

    # Allocate SMEM buffers for epilogue TMA store (at least 2 for multi-buffering)
    NUM_EPILOGUE_SMEM_BUFFERS: tl.constexpr = (
        NUM_MMA_GROUPS if NUM_MMA_GROUPS > 2 else 2
    )
    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    c_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, slice_size),
        tlx.dtype_of(workspace_desc),
        NUM_EPILOGUE_SMEM_BUFFERS,
    )

    m_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, slice_size),
        tlx.dtype_of(m_desc),
        NUM_EPILOGUE_SMEM_BUFFERS,
    )

    NUM_X0LI_BUFS: tl.constexpr = 3
    buffers_x0 = tlx.local_alloc(
        (BLOCK_M_SPLIT, slice_size),
        tlx.dtype_of(x0_desc),
        NUM_X0LI_BUFS,
    )
    buffers_li = tlx.local_alloc(
        (BLOCK_M_SPLIT, slice_size),
        tlx.dtype_of(li_desc),
        NUM_X0LI_BUFS,
    )
    x0li_full_bars = tlx.alloc_barriers(NUM_X0LI_BUFS, arrive_count=1)
    x0li_empty_bars = tlx.alloc_barriers(NUM_X0LI_BUFS, arrive_count=1)

    # CTA pairs are placed along M dim
    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(
            num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=2
        )  # CTA0 waits for CTA1's data before mma
    else:
        cluster_cta_rank = 0
        pred_cta0 = False
        cta_bars = None

    # allocate barriers - each subtile needs its own barriers
    # NUM_SMEM_BUFFERS barriers per subtile for synchronization
    A_smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    A_smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    B_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    # NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    if USE_WARP_BARRIER:
        tmem_full_bars = tlx.alloc_warp_barrier(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, num_warps=1
        )
        tmem_empty_bars = tlx.alloc_warp_barrier(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
            num_warps=4,
            num_arrivals=EPILOGUE_SUBTILE,
        )
    else:
        tmem_full_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
        )
        tmem_empty_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
            arrive_count=EPILOGUE_SUBTILE,
        )

    with tlx.async_tasks():
        with tlx.async_task("default"):  # epilogue consumer
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )

            tmem_accum_cnt = 0
            x0li_slice_cnt = 0
            tile_id = start_pid
            idx = 0
            # Only fuse addcmul when SPLIT_K == 1 (fusing per-split is wrong)
            DO_FUSE: tl.constexpr = FUSE_ADDCMUL and (SPLIT_K == 1)

            while tile_id < num_tiles:
                # Skip tiles whose split has zero K-tiles (last split
                # can be empty when cdiv(k_tiles_total, SPLIT_K) * (SPLIT_K-1)
                # >= k_tiles_total).
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)
                if ENABLE_PROTON:
                    pl.enter_scope("epi_tile")
                if k_tile_end > k_tile_start:
                    cur_tmem_buf, tmem_read_phase = _get_bufidx_phase(
                        tmem_accum_cnt, NUM_TMEM_BUFFERS
                    )
                    x0li_slice_cnt = _process_tile_epilogue_inner(
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        M=M,
                        N=N,
                        BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        NUM_CTAS=NUM_CTAS,
                        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        SPLIT_K=SPLIT_K,
                        c_desc=c_desc,
                        workspace_desc=workspace_desc,
                        c_smem_buffers=c_smem_buffers,
                        tmem_buffers=tmem_buffers,
                        tmem_full_bars=tmem_full_bars,
                        tmem_empty_bars=tmem_empty_bars,
                        cur_tmem_buf=cur_tmem_buf,
                        tmem_read_phase=tmem_read_phase,
                        b2_ptr=b2_ptr,
                        buffers_x0=buffers_x0,
                        buffers_li=buffers_li,
                        x0li_full_bars=x0li_full_bars,
                        x0li_empty_bars=x0li_empty_bars,
                        x0li_slice_cnt=x0li_slice_cnt,
                        NUM_X0LI_BUFS=NUM_X0LI_BUFS,
                        m_desc=m_desc,
                        m_smem_buffers=m_smem_buffers,
                        FUSE_ADDCMUL=DO_FUSE,
                        ENABLE_PROTON=ENABLE_PROTON,
                        PROTON_ITER=PROTON_ITER,
                        idx=idx,
                    )
                    tmem_accum_cnt += 1
                if ENABLE_PROTON:
                    pl.exit_scope("epi_tile")
                idx += 1
                tile_id += NUM_SMS

        with tlx.async_task(num_warps=1, num_regs=24):  # MMA consumer
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )

            tmem_accum_cnt = 0
            smem_accum_cnt = 0
            tile_id = start_pid
            idx = 0

            while tile_id < num_tiles:
                # Compute K range for this split
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

                # Skip tiles whose split has zero K-tiles
                if ENABLE_PROTON:
                    pl.enter_scope("mma_tile")
                if k_tile_end > k_tile_start:
                    cur_tmem_buf, tmem_write_phase = _get_bufidx_phase(
                        tmem_accum_cnt, NUM_TMEM_BUFFERS
                    )
                    smem_accum_cnt = _process_tile_mma_inner(
                        k_tiles=k_tiles_total,
                        k_tile_start=k_tile_start,
                        k_tile_end=k_tile_end,
                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        buffers_A=buffers_A,
                        buffers_B=buffers_B,
                        tmem_buffers=tmem_buffers,
                        A_smem_full_bars=A_smem_full_bars,
                        B_smem_full_bars=B_smem_full_bars,
                        A_smem_empty_bars=A_smem_empty_bars,
                        tmem_full_bars=tmem_full_bars,
                        cur_tmem_buf=cur_tmem_buf,
                        tmem_empty_bars=tmem_empty_bars,
                        tmem_write_phase=tmem_write_phase,
                        smem_accum_cnt=smem_accum_cnt,
                        NUM_CTAS=NUM_CTAS,
                        cta_bars=cta_bars,
                        pred_cta0=pred_cta0,
                        A_ROW_MAJOR=A_ROW_MAJOR,
                        B_ROW_MAJOR=B_ROW_MAJOR,
                        ENABLE_PROTON=ENABLE_PROTON,
                        PROTON_ITER=PROTON_ITER,
                        idx=idx,
                    )
                    tmem_accum_cnt += 1
                if ENABLE_PROTON:
                    pl.exit_scope("mma_tile")
                idx += 1
                tile_id += NUM_SMS

        with tlx.async_task(num_warps=1, num_regs=24):  # producer, TMA load (A/B only)
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )

            smem_accum_cnt = 0
            tile_id = start_pid
            idx = 0

            while tile_id < num_tiles:
                # Compute K range for this split
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

                # Skip tiles whose split has zero K-tiles
                if ENABLE_PROTON:
                    pl.enter_scope("prod_tile")
                if k_tile_end > k_tile_start:
                    smem_accum_cnt = _process_tile_producer_inner(
                        tile_id=tile_id,
                        start_pid=start_pid,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                        k_tile_start=k_tile_start,
                        k_tile_end=k_tile_end,
                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                        a_desc=a_desc,
                        b_desc=b_desc,
                        buffers_A=buffers_A,
                        buffers_B=buffers_B,
                        A_smem_full_bars=A_smem_full_bars,
                        B_smem_full_bars=B_smem_full_bars,
                        A_smem_empty_bars=A_smem_empty_bars,
                        smem_accum_cnt=smem_accum_cnt,
                        NUM_CTAS=NUM_CTAS,
                        cluster_cta_rank=cluster_cta_rank,
                        A_ROW_MAJOR=A_ROW_MAJOR,
                        B_ROW_MAJOR=B_ROW_MAJOR,
                        ENABLE_PROTON=ENABLE_PROTON,
                        PROTON_ITER=PROTON_ITER,
                        idx=idx,
                    )
                if ENABLE_PROTON:
                    pl.exit_scope("prod_tile")
                idx += 1
                tile_id += NUM_SMS

        with tlx.async_task(num_warps=1, num_regs=24):  # 4th warp: epilogue TMA loader
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )
            x0li_load_cnt = 0
            tile_id = start_pid
            DO_FUSE: tl.constexpr = FUSE_ADDCMUL and (SPLIT_K == 1)
            epi_dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(x0_desc))
            slice_bytes: tl.constexpr = epi_dsize * BLOCK_M_SPLIT * slice_size

            while tile_id < num_tiles:
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)
                if k_tile_end > k_tile_start and DO_FUSE:
                    mn_tile_id = tile_id % num_mn_tiles
                    pid_m, pid_n = _compute_pid(
                        mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M
                    )
                    offs_bn_epi = pid_n * BLOCK_SIZE_N
                    for g in tl.static_range(NUM_MMA_GROUPS):
                        offs_am_g = pid_m * BLOCK_SIZE_M + g * BLOCK_M_SPLIT
                        for s_id in tl.static_range(EPILOGUE_SUBTILE):
                            x0li_buf, x0li_phase = _get_bufidx_phase(
                                x0li_load_cnt, NUM_X0LI_BUFS
                            )
                            if x0li_load_cnt >= NUM_X0LI_BUFS:
                                tlx.barrier_wait(
                                    x0li_empty_bars[x0li_buf], x0li_phase ^ 1
                                )
                            tlx.barrier_expect_bytes(
                                x0li_full_bars[x0li_buf], 2 * slice_bytes
                            )
                            tlx.async_descriptor_load(
                                x0_desc,
                                buffers_x0[x0li_buf],
                                [offs_am_g, offs_bn_epi + s_id * slice_size],
                                x0li_full_bars[x0li_buf],
                            )
                            tlx.async_descriptor_load(
                                li_desc,
                                buffers_li[x0li_buf],
                                [offs_am_g, offs_bn_epi + s_id * slice_size],
                                x0li_full_bars[x0li_buf],
                            )
                            x0li_load_cnt += 1
                tile_id += NUM_SMS


_ENABLE_PROTON = False
_printed_heuristic_configs = set()


def matmul_addcmul_epi_prefetch(  # noqa: C901
    s, W, b2, x0, layer_input, config=None, enable_proton=_ENABLE_PROTON
):
    """4-task epilogue-prefetch variant: fused matmul + bias + store_m + addcmul.

    The epilogue ops are done in FP32 on the matmul accumulator before casting,
    saving separate kernel launches and extra reads of the matmul output.

    Args:
        s: Input matrix of shape (M, K)
        W: Weight matrix of shape (N, K) — matmul computes s @ W.T
        b2: Bias vector of shape (N,)
        x0: Multiplicand of shape (M, N)
        layer_input: Addend of shape (M, N)
        config: Optional dict with kernel config.

    Returns:
        (out, m) where out = layer_input + x0 * m, m = s @ W.T + b2
    """
    assert s.shape[1] == W.shape[1], "Incompatible dimensions"
    M, K = s.shape
    N = W.shape[0]
    assert x0.shape == (M, N), f"x0 shape {x0.shape} != ({M}, {N})"
    assert layer_input.shape == (M, N), (
        f"layer_input shape {layer_input.shape} != ({M}, {N})"
    )

    # Allocate outputs: out (c_desc) and m (m_desc)
    out = torch.empty((M, N), device=s.device, dtype=s.dtype)
    m_out = torch.empty((M, N), device=s.device, dtype=s.dtype)

    b2_2d = b2.unsqueeze(0)

    # Detect column-major inputs.
    a_row_major = s.is_contiguous()
    b_mat = W.t()  # (K, N)
    b_row_major = b_mat.is_contiguous()

    dummy_block = [1, 1]
    if not a_row_major:
        a_t = s.T
        a_desc = TensorDescriptor(a_t, a_t.shape, a_t.stride(), dummy_block)
    else:
        a_desc = TensorDescriptor(s, s.shape, s.stride(), dummy_block)
    if not b_row_major:
        b_t = b_mat.T
        b_desc = TensorDescriptor(b_t, b_t.shape, b_t.stride(), dummy_block)
    else:
        b_desc = TensorDescriptor(b_mat, b_mat.shape, b_mat.stride(), dummy_block)
    c_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)
    x0_desc = TensorDescriptor(x0, x0.shape, x0.stride(), dummy_block)
    li_desc = TensorDescriptor(
        layer_input, layer_input.shape, layer_input.stride(), dummy_block
    )
    m_desc = TensorDescriptor(m_out, m_out.shape, m_out.stride(), dummy_block)

    NUM_SMS = _get_num_sms()

    # Use heuristic config if no config provided and env var is set
    use_heuristic = os.environ.get("TLX_GEMM_USE_HEURISTIC", "0") == "1"
    if config is None and use_heuristic:
        config = get_heuristic_config(M, N, K, NUM_SMS)
        if config is not None and os.environ.get("TRITON_PRINT_AUTOTUNING") == "1":
            shape_key = (M, N, K)
            if shape_key not in _printed_heuristic_configs:
                _printed_heuristic_configs.add(shape_key)
                config_str = ", ".join(
                    f"{k}: {v}"
                    for k, v in config.items()
                    if k not in ("pre_hook", "ctas_per_cga")
                )
                print(f"heuristic config selected: {config_str};")

    if config is not None:
        config = dict(config)
        ctas_per_cga = config.pop("ctas_per_cga", None)
        config.pop("pre_hook", None)
        config.pop("INTERLEAVE_EPILOGUE", None)
        config.pop("EPILOGUE_PREFETCH_K", None)
        split_k = config.get("SPLIT_K", 1)
        if split_k > 1:
            rows_per_split = _workspace_rows_per_split(
                M,
                config["BLOCK_SIZE_M"],
                config.get("NUM_CTAS", 1),
            )
            workspace = torch.empty(
                (split_k * rows_per_split, N),
                device=s.device,
                dtype=torch.float32,
            )
            workspace_desc = TensorDescriptor(
                workspace, workspace.shape, workspace.stride(), dummy_block
            )
        else:
            workspace_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)
        hook_args = {
            "a_desc": a_desc,
            "b_desc": b_desc,
            "c_desc": c_desc,
            "b2_ptr": b2_2d,
            "x0_desc": x0_desc,
            "li_desc": li_desc,
            "m_desc": m_desc,
            "workspace_desc": workspace_desc,
            "M": M,
            "N": N,
            "K": K,
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            **config,
        }
        matmul_tma_set_block_size_hook(hook_args)
        NUM_CTAS = config.get("NUM_CTAS", 1)
        num_pid_m = triton.cdiv(M, config["BLOCK_SIZE_M"])
        num_pid_n = triton.cdiv(N, config["BLOCK_SIZE_N"])
        num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
        total_tiles = num_pid_m * num_pid_n * split_k
        grid = (min(NUM_SMS, total_tiles),)
        if enable_proton:
            import triton.profiler as proton

            mode = proton.mode.Default(
                metric_type="cycle",
                optimizations="clock32",
                buffer_size=131072,
                buffer_type="global",
            )
            proton.start("proton", data="trace", backend="instrumentation", mode=mode)
        wrap_triton(matmul_addcmul_kernel.fn)[grid](
            a_desc,
            b_desc,
            c_desc,
            workspace_desc,
            b2_2d,
            x0_desc,
            li_desc,
            m_desc,
            M,
            N,
            K,
            A_ROW_MAJOR=a_row_major,
            B_ROW_MAJOR=b_row_major,
            NUM_SMS=NUM_SMS,
            FUSE_ADDCMUL=True,
            ENABLE_PROTON=enable_proton,
            PROTON_ITER=10,
            ctas_per_cga=ctas_per_cga,
            **config,
        )
        if enable_proton:
            proton.finalize()
        if split_k > 1:
            reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
            _reduce_k_addcmul_kernel[reduce_grid](
                workspace_desc.base,
                out,
                m_out,
                b2,
                x0,
                layer_input,
                M,
                N,
                rows_per_split,
                SPLIT_K=split_k,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[s.dtype],
                num_warps=4,
            )
    else:
        workspace_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

        def grid(META):
            NUM_CTAS = META["NUM_CTAS"]
            num_pid_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
            num_pid_n = triton.cdiv(N, META["BLOCK_SIZE_N"])
            num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
            mn_tiles = num_pid_m * num_pid_n
            total_tiles = mn_tiles * META["SPLIT_K"]
            return (min(NUM_SMS, total_tiles),)

        if enable_proton:
            import triton.profiler as proton

            mode = proton.mode.Default(
                metric_type="cycle",
                optimizations="clock32",
                buffer_size=131072,
                buffer_type="global",
            )
            proton.start("proton", data="trace", backend="instrumentation", mode=mode)
        matmul_addcmul_kernel[grid](
            a_desc,
            b_desc,
            c_desc,
            workspace_desc,
            b2_2d,
            x0_desc,
            li_desc,
            m_desc,
            M,
            N,
            K,
            A_ROW_MAJOR=a_row_major,
            B_ROW_MAJOR=b_row_major,
            NUM_SMS=NUM_SMS,
            FUSE_ADDCMUL=True,
            ENABLE_PROTON=enable_proton,
        )
        if enable_proton:
            proton.finalize()
        best = matmul_addcmul_kernel.best_config
        split_k = best.kwargs.get("SPLIT_K", 1)
        if split_k > 1:
            workspace = workspace_desc.base
            rows_per_split = _workspace_rows_per_split(
                M,
                best.kwargs["BLOCK_SIZE_M"],
                best.kwargs.get("NUM_CTAS", 1),
            )
            reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
            _reduce_k_addcmul_kernel[reduce_grid](
                workspace,
                out,
                m_out,
                b2,
                x0,
                layer_input,
                M,
                N,
                rows_per_split,
                SPLIT_K=split_k,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[s.dtype],
                num_warps=4,
            )
    return out, m_out


_FIXED_PRODUCTION_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 64,
    "NUM_SMEM_BUFFERS": 6,
    "NUM_TMEM_BUFFERS": 3,
    "NUM_MMA_GROUPS": 1,
    "EPILOGUE_SUBTILE": 4,
    "NUM_CTAS": 2,
    "SPLIT_K": 1,
    "INTERLEAVE_EPILOGUE": 0,
    "USE_WARP_BARRIER": False,
    "ctas_per_cga": (2, 1, 1),
}


def _sm100_01_fused_kernel_impl(
    s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.compiler.is_compiling():
        return _inductor_sm100_01_fused_impl(
            s,
            weight,
            bias,
            multiplier,
            residual,
        )
    return matmul_addcmul_epi_prefetch(
        s,
        weight,
        bias,
        multiplier,
        residual,
        _FIXED_PRODUCTION_CONFIG,
    )


sm100_01_fused_kernel = triton_op(
    "torch_tlx::sm100_01_fused_kernel",
    mutates_args={},
)(_sm100_01_fused_kernel_impl)


_SM100_01_TRACEABLE_KERNEL = triton.autotune(
    configs=[
        triton.Config(
            {},
            num_warps=4,
            num_stages=1,
            ctas_per_cga=(2, 1, 1),
        )
    ],
    key=[],
)(matmul_addcmul_kernel.fn)


def _inductor_sm100_01_fused_impl(
    s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Visible HOP form used by the late Inductor subgraph replacement.

    ``wrap_triton`` captures host TMA descriptors through Dynamo, but the
    subgraph autotuner traces decompositions with ``make_fx`` after AOT.  Pass
    the equivalent stable-descriptor metadata directly to its Triton HOP.
    """
    from torch._higher_order_ops.triton_kernel_wrap import (
        create_tma_stable_metadata,
        kernel_side_table,
        triton_kernel_wrapper_mutation,
    )

    m, k = s.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=s.device, dtype=s.dtype)
    linear = torch.empty((m, n), device=s.device, dtype=s.dtype)
    # SPLIT_K=1 specializes every workspace access away.  Keep a distinct,
    # minimally sized descriptor backing so AOT functionalization does not
    # split the aliased c_desc/workspace_desc and return the stale clone.
    workspace = torch.empty((128, 32), device=s.device, dtype=s.dtype)
    num_sms = _get_num_sms()
    config = {
        "M": m,
        "N": n,
        "K": k,
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 64,
        "NUM_SMEM_BUFFERS": 6,
        "NUM_TMEM_BUFFERS": 3,
        "NUM_MMA_GROUPS": 1,
        "EPILOGUE_SUBTILE": 4,
        "NUM_CTAS": 2,
        "SPLIT_K": 1,
        "NUM_SMS": num_sms,
        "A_ROW_MAJOR": True,
        "B_ROW_MAJOR": False,
        "USE_WARP_BARRIER": False,
        "FUSE_ADDCMUL": True,
        "ENABLE_PROTON": False,
        "PROTON_ITER": 10,
        "ctas_per_cga": (2, 1, 1),
    }
    num_pid_m = triton.cdiv(m, config["BLOCK_SIZE_M"])
    num_pid_m = triton.cdiv(num_pid_m, config["NUM_CTAS"]) * config["NUM_CTAS"]
    num_pid_n = triton.cdiv(n, config["BLOCK_SIZE_N"])
    total_tiles = num_pid_m * num_pid_n
    triton_kernel_wrapper_mutation(
        kernel_idx=kernel_side_table.add_kernel(_SM100_01_TRACEABLE_KERNEL),
        constant_args_idx=kernel_side_table.add_constant_args(config),
        grid=[(min(num_sms, total_tiles), 1, 1)],
        tma_descriptor_metadata={
            "a_desc": create_tma_stable_metadata([128, 64]),
            "b_desc": create_tma_stable_metadata([64, 64]),
            "c_desc": create_tma_stable_metadata([128, 32]),
            "workspace_desc": create_tma_stable_metadata([128, 32]),
            "x0_desc": create_tma_stable_metadata([128, 32]),
            "li_desc": create_tma_stable_metadata([128, 32]),
            "m_desc": create_tma_stable_metadata([128, 32]),
        },
        kwargs={
            "a_desc": s,
            "b_desc": weight,
            "c_desc": out,
            "workspace_desc": workspace,
            "b2_ptr": bias,
            "x0_desc": multiplier,
            "li_desc": residual,
            "m_desc": linear,
        },
        launch_kwargs=("ctas_per_cga",),
    )
    return out, linear


def _aten_sm100_01(
    s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    accumulator = s.float() @ weight.float().T
    linear_fp32 = accumulator + bias.float()
    linear = linear_fp32.to(torch.bfloat16)
    output = (residual.float() + multiplier.float() * linear_fp32).to(
        torch.bfloat16
    )
    return output, linear


@torch.library.custom_op("torch_tlx::sm100_01_semantic", mutates_args=())
def sm100_01_semantic(
    s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aten_sm100_01(s, weight, bias, multiplier, residual)


@sm100_01_semantic.register_fake
def _fake_sm100_01_semantic(
    s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del bias, multiplier, residual
    shape = (s.shape[0], weight.shape[0])
    return (
        torch.empty(shape, device=s.device, dtype=torch.bfloat16),
        torch.empty(shape, device=s.device, dtype=torch.bfloat16),
    )


def _fused_sm100_01(
    s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _inductor_sm100_01_fused_impl(
        s,
        weight,
        bias,
        multiplier,
        residual,
    )


def _eligible_sm100_01(match) -> bool:
    from torch._inductor import config

    from ..hw.target import current_target

    if config.triton.tlx_mode not in ("allow", "force"):
        return False
    if not current_target().is_blackwell:
        return False
    tensors = [
        match.kwargs[name].meta.get("val")
        for name in ("s", "weight", "bias", "multiplier", "residual")
    ]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    s, weight, bias, multiplier, residual = tensors
    return bool(
        s.shape == (1152, 1024)
        and weight.shape == (12800, 1024)
        and bias.shape == (12800,)
        and multiplier.shape == (1152, 12800)
        and residual.shape == (1152, 12800)
        and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
        and all(tensor.device == s.device for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
    )


@functools.cache
def register_sm100_01_pattern() -> None:
    from torch._inductor.fx_passes.post_grad import pass_patterns
    from torch._inductor.kernel.custom_op import CustomOpConfig
    from torch._inductor.pattern_matcher import fwd_only, register_replacement

    from .subgraph import register_tlx_subgraph_autotuning

    register_tlx_subgraph_autotuning(
        sm100_01_semantic,
        name="tlx_sm100_01_matmul_addcmul",
        tlx_configs=[CustomOpConfig(_fused_sm100_01)],
        aten_impl=_aten_sm100_01,
    )

    example_inputs = (
        torch.empty((2, 64), dtype=torch.bfloat16),
        torch.empty((32, 64), dtype=torch.bfloat16),
        torch.empty((32,), dtype=torch.bfloat16),
        torch.empty((2, 32), dtype=torch.bfloat16),
        torch.empty((2, 32), dtype=torch.bfloat16),
    )

    def pattern(s, weight, bias, multiplier, residual):
        return _aten_sm100_01(s, weight, bias, multiplier, residual)

    def replacement(s, weight, bias, multiplier, residual):
        return sm100_01_semantic(s, weight, bias, multiplier, residual)

    register_replacement(
        pattern,
        replacement,
        example_inputs,
        fwd_only,
        pass_patterns[0],
        extra_check=_eligible_sm100_01,
        pattern_name="tlx_sm100_01_matmul_addcmul",
    )
