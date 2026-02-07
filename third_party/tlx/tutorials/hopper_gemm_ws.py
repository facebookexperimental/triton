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


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BM"]
    BLOCK_N = nargs["BN"]
    BLOCK_K = nargs["BK"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BM": 128,
                "BN": 256,
                "BK": 64,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": 4,
                "NUM_MMA_WARPS": 8,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": True,
            }, num_stages=1, num_warps=4, pre_hook=matmul_tma_set_block_size_hook),
        triton.Config(
            {
                "BM": 128,
                "BN": 256,
                "BK": 64,
                "GROUP_SIZE_M": 8,
                "NUM_STAGES": 3,
                "NUM_MMA_WARPS": 8,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": False,
            }, num_stages=1, num_warps=4, pre_hook=matmul_tma_set_block_size_hook),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
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
    bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            # Assuming NUM_STAGES = 2
            # p should be 1, 1, 0, 0, 1, 1, 0, 0, ...
            p = 1

            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES
                offset_k = k * BK

                # Async load to a[buf]
                empty_a_1st = tlx.local_view(bars_empty_a, buf)  # mbar
                full_a_1st = tlx.local_view(bars_full_a, buf)  # mbar
                tlx.barrier_wait(bar=empty_a_1st, phase=p)  # EmptyBar A1 wait
                tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                data_a_1st = tlx.local_view(a, buf)  # smem data
                tlx.async_descriptor_load(a_desc, data_a_1st, [offset_am, offset_k], full_a_1st)

                # Async load to b[buf]
                empty_b = tlx.local_view(bars_empty_b, buf)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=empty_b, phase=p)
                tlx.barrier_expect_bytes(full_b, BN * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                data_b = tlx.local_view(b, buf)
                tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                # Async load to a[buf+NUM_STAGES]
                empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                tlx.barrier_expect_bytes(bar=full_a_2nd, size=BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)  # smem data
                tlx.async_descriptor_load(a_desc, data_a_2nd, [offset_am + BLOCK_M_SPLIT, offset_k], full_a_2nd)

                # Flip phase after every NUM_STAGES iterations finish
                p = p ^ (buf == (NUM_STAGES - 1))

        # consumers (wgmma + async store)
        with tlx.async_task(num_warps=4, replicate=2):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
            offset_am = pid_m * BM
            offset_bn = pid_n * BN

            p = 0
            # Assuming NUM_STAGES = 2
            # p should be 0, 0, 1, 1, 0, 0, ...
            acc = tl.zeros([BM // 2, BN], dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BK)):
                buf = k % NUM_STAGES

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

                # Flip phase after every NUM_STAGES iterations finish
                p = p ^ (buf == (NUM_STAGES - 1))

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


def matmul(a, b, config=None):
    """Matrix multiplication using TLX GEMM kernel."""
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    triton.set_allocator(alloc_fn)

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros(
        (M, N),
        dtype=torch.float16,
        device=DEVICE,
    )

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
        desc_in_2.block_shape = [config["BK"], config["BN"]]
        if config.get("EPILOGUE_SUBTILE", False):
            desc_out.block_shape = [BLOCK_M_SPLIT, config["BN"] // 2]
        else:
            desc_out.block_shape = [BLOCK_M_SPLIT, config["BN"]]

        grid = (triton.cdiv(M, config['BM']) * triton.cdiv(N, config['BN']), )
        matmul_kernel_tlx_ws.fn[grid](
            desc_in_1,
            desc_in_2,
            desc_out,
            M,
            N,
            K,
            **config,
        )
    else:
        grid = lambda META: (  # noqa E731
            triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']), )
        matmul_kernel_tlx_ws[grid](
            desc_in_1,
            desc_in_2,
            desc_out,
            M,
            N,
            K,
        )
    return c
