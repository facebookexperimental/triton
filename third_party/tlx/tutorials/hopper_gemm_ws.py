import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BM"]
    BLOCK_N = nargs["BN"]
    BLOCK_K = nargs["BK"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N // NUM_CTAS]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]
    # Add NUM_SMS
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    nargs["NUM_SMS"] = NUM_SMS


def preprocess_configs(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]

    IMBALANCE_THRESHOLD = 10
    if M > N * IMBALANCE_THRESHOLD:
        # M >> N: keep only small GROUP_SIZE_M to sweep M, reuse B
        configs = [c for c in configs if c.kwargs["GROUP_SIZE_M"] == 1]
    elif N > M * IMBALANCE_THRESHOLD:
        # N >> M: keep only large GROUP_SIZE_M to sweep N, reuse A
        configs = [c for c in configs if c.kwargs["GROUP_SIZE_M"] >= 32]
    else:
        # Balanced: keep moderate GROUP_SIZE_M for L2 locality
        configs = [c for c in configs if c.kwargs["GROUP_SIZE_M"] == 8]

    return configs


def get_autotune_configs():
    return [
        triton.Config(
            {
                "BM": BM,
                "BN": BN,
                "BK": BK,
                "GROUP_SIZE_M": g,
                "NUM_STAGES": s,
                "NUM_MMA_WARPS": 8,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": epilogue,
                "NUM_CTAS": num_ctas,
                "USE_WARP_BARRIER": uwb,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(num_ctas, 1, 1),
        )
        for BM in [128, 256]
        for BN in [128, 256]
        for BK in [64]
        for s in [3, 4]
        for epilogue in [True, False]
        for g in [1, 8, 64]
        for num_ctas in [1, 2]
        for uwb in [False, True]
    ]


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": preprocess_configs},
)
@triton.jit
def matmul_kernel_tlx_ws(a_desc, b_desc, c_desc,  #
                         M, N, K,  #
                         BM: tl.constexpr,  #
                         BN: tl.constexpr,  #
                         BK: tl.constexpr,  #
                         GROUP_SIZE_M: tl.constexpr,  #
                         NUM_STAGES: tl.constexpr,  #
                         NUM_MMA_WARPS: tl.constexpr,  #
                         NUM_MMA_GROUPS: tl.constexpr,  #
                         EPILOGUE_SUBTILE: tl.constexpr,  #
                         NUM_CTAS: tl.constexpr,  #
                         NUM_SMS: tl.constexpr,  #
                         USE_WARP_BARRIER: tl.constexpr = False,  #
                         ):
    # Descriptor
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    # Need NUM_STAGES sets of SMEM buffers for A and B
    # where each set contains two for A and one for B.
    # Split A into two in M-dimension to have two consumer tasks for wgmma
    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    # Need NUM_STAGES sets of mbarriers for A and B
    # where each set contains two for A and one for B.
    # Do the above for both empty states and full states respectively.
    if USE_WARP_BARRIER:
        bars_empty_a = tlx.alloc_warp_barrier(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, num_warps=4)
        bars_empty_b = tlx.alloc_warp_barrier(num_barriers=NUM_STAGES, num_warps=4, num_arrivals=NUM_MMA_GROUPS)
    else:
        bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
        bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    # Barriers for cross-CTA synchronization before multicast TMA loads
    if NUM_CTAS == 2:
        cta_bars = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=2)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            # Persistent loop - each SM processes tiles with stride NUM_SMS
            tile_id = sm_id
            smem_accum_cnt = 0
            while tile_id < num_tiles:
                # Convert tile_id to pid_m and pid_n
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                for k in range(0, tl.cdiv(K, BK)):
                    buf, p = _get_bufidx_phase(smem_accum_cnt, NUM_STAGES)
                    offset_k = k * BK

                    # Async load to a[buf]
                    empty_a_1st = tlx.local_view(bars_empty_a, buf)  # mbar
                    full_a_1st = tlx.local_view(bars_full_a, buf)  # mbar
                    tlx.barrier_wait(bar=empty_a_1st, phase=p ^ 1)  # EmptyBar A1 wait
                    tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    data_a_1st = tlx.local_view(a, buf)  # smem data
                    tlx.async_descriptor_load(a_desc, data_a_1st, [offset_am, offset_k], full_a_1st)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p ^ 1)
                    tlx.barrier_expect_bytes(full_b, BN * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    data_b = tlx.local_view(b, buf)

                    if NUM_CTAS == 2:
                        # Sync cluster: ensure both CTAs' buffers are ready for multicast
                        cta_id = tlx.cluster_cta_rank()
                        cta_bar = tlx.local_view(cta_bars, buf)
                        tlx.barrier_arrive(cta_bar, 1)
                        tlx.barrier_arrive(cta_bar, 1, remote_cta_rank=cta_id ^ 1)
                        tlx.barrier_wait(cta_bar, p)

                        # Each CTA loads half of B and multicasts to both CTAs
                        if cta_id == 0:
                            buf_b_slice = tlx.local_slice(data_b, [0, 0], [BK, BN // 2])
                        else:
                            buf_b_slice = tlx.local_slice(data_b, [0, BN // 2], [BK, BN // 2])
                        tlx.async_descriptor_load(
                            b_desc,
                            buf_b_slice,
                            [offset_k, offset_bn + cta_id * BN // 2],
                            full_b,
                            multicast_targets=[cta_id, cta_id ^ 1],
                        )
                    else:
                        tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                    # Async load to a[buf+NUM_STAGES]
                    empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                    full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p ^ 1)
                    tlx.barrier_expect_bytes(bar=full_a_2nd,
                                             size=BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)  # smem data
                    tlx.async_descriptor_load(a_desc, data_a_2nd, [offset_am + BLOCK_M_SPLIT, offset_k], full_a_2nd)

                    smem_accum_cnt += 1

                # Move to next tile with stride NUM_SMS
                tile_id += NUM_SMS

        # consumers (wgmma + async store)
        with tlx.async_task(num_warps=4, replicate=2):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            # Persistent loop - each SM processes tiles with stride NUM_SMS
            tile_id = sm_id
            smem_accum_cnt = 0
            while tile_id < num_tiles:
                # Convert tile_id to pid_m and pid_n
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                acc = tl.zeros([BM // 2, BN], dtype=tl.float32)
                for k in range(0, tl.cdiv(K, BK)):
                    buf, p = _get_bufidx_phase(smem_accum_cnt, NUM_STAGES)

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id())  # noqa
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id())  # noqa
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(
                        data_a,
                        data_b,
                        acc,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id())  # noqa
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    smem_accum_cnt += 1

                offset_cm = offset_am + BLOCK_M_SPLIT * tlx.async_task_replica_id()
                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn], c0)
                    c1 = acc1.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn + BN // 2], c1)
                else:
                    c_desc.store([offset_cm, offset_bn], acc.to(tlx.dtype_of(c_desc)))  # noqa

                # Move to next tile with stride NUM_SMS
                tile_id += NUM_SMS


def matmul(a, b, config=None):
    """Matrix multiplication using TLX GEMM kernel."""
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    triton.set_allocator(alloc_fn)

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.empty(
        (M, N),
        dtype=torch.float16,
        device=DEVICE,
    )

    # Get number of SMs
    NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(
        a,
        shape=[M, K],
        strides=[K, 1],
        block_shape=dummy_block,
    )

    desc_in_2 = TensorDescriptor(
        b,
        shape=[K, N],
        strides=[N, 1],
        block_shape=dummy_block,
    )
    desc_out = TensorDescriptor(
        c,
        shape=[M, N],
        strides=[N, 1],
        block_shape=dummy_block,
    )

    if config is not None:
        # Set descriptor block shapes according to config
        NUM_MMA_GROUPS = config["NUM_MMA_GROUPS"]
        BLOCK_M_SPLIT = config["BM"] // NUM_MMA_GROUPS
        desc_in_1.block_shape = [BLOCK_M_SPLIT, config["BK"]]
        desc_in_2.block_shape = [config["BK"], config["BN"] // config.get("NUM_CTAS", 1)]
        if config.get("EPILOGUE_SUBTILE", False):
            desc_out.block_shape = [BLOCK_M_SPLIT, config["BN"] // 2]
        else:
            desc_out.block_shape = [BLOCK_M_SPLIT, config["BN"]]

        # Use persistent kernel with min(NUM_SMS, total_tiles) blocks
        num_pid_m = triton.cdiv(M, config["BM"])
        num_pid_n = triton.cdiv(N, config["BN"])
        total_tiles = num_pid_m * num_pid_n
        grid = (min(NUM_SMS, total_tiles), )
        matmul_kernel_tlx_ws.fn[grid](
            desc_in_1,
            desc_in_2,
            desc_out,
            M,
            N,
            K,
            NUM_SMS=NUM_SMS,
            **config,
        )
    else:
        # Use persistent kernel with min(NUM_SMS, total_tiles) blocks
        grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"])), )  # noqa: E731
        matmul_kernel_tlx_ws[grid](
            desc_in_1,
            desc_in_2,
            desc_out,
            M,
            N,
            K,
            NUM_SMS=NUM_SMS,
        )
    return c
