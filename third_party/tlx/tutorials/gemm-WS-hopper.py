import pytest
import torch

import triton
import triton.language as tl
import triton.tlx.language as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"
def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'

def get_cuda_autotune_config():
    return [
        triton.Config({'BM': 128, 'BK': 128, 'BN': 128},
                      num_warps=8),
    ]

@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires Nvidia Hopper",
)
def test_ws_gemm():

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=DEVICE)


    @triton.jit
    def matmul_kernel_tlx_ws(
        desc_in_1, desc_in_2, desc_out, #
        K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
    ):
        # Descriptor
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offset_am = pid_m * BM
        offset_bn = pid_n * BN

        # Need NUM_STAGES sets of SMEM buffers for A and B
        # where each set contains two for A and one for B.
        # Split A into two in M-dimension to have two consumer tasks for wgmma
        a = tlx.local_alloc((BM // 2, BK), tl.float16, NUM_STAGES * 2)
        b = tlx.local_alloc((BK, BN), tl.float16, NUM_STAGES)

        # Need NUM_STAGES sets of mbarriers for A and B
        # where each set contains two for A and one for B.
        # Do the above for both empty states and full states respectively.
        bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * 2,)
        bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * 2, arrive_count=1)
        bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=2)
        bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

        # Warp specilization
        with tlx.async_tasks():
            # Producer (async load)
            with tlx.async_task("default"):
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
                    tlx.barrier_expect_bytes(full_a_1st, (BM//2) * BK * 2)
                    data_a_1st = tlx.local_view(a, buf)  # smem data
                    tlx.async_descriptor_load(
                        desc_in_1,
                        data_a_1st,
                        [offset_am, offset_k],
                        full_a_1st)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        desc_in_2,
                        data_b,
                        [offset_k, offset_bn],
                        full_b)

                    # Async load to a[buf+NUM_STAGES]
                    empty_a_2nd = tlx.local_view(bars_empty_a, buf+NUM_STAGES)
                    full_a_2nd = tlx.local_view(bars_full_a, buf+NUM_STAGES)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                    tlx.barrier_expect_bytes(bar=full_a_2nd, size=(BM//2) * BK * 2)
                    data_a_2nd = tlx.local_view(a, buf+NUM_STAGES)  # smem data
                    tlx.async_descriptor_load(
                        desc_in_1,
                        data_a_2nd,
                        [offset_am + (BM//2), offset_k],
                        full_a_2nd)

                    # Flip phase after every NUM_STAGES iterations finish
                    p = (p^1) if (buf == (NUM_STAGES-1)) else p

            # consumers (wgmma + async store)
            with tlx.async_task(num_warps=4, replicate=2):
                p = 0
                # Assuming NUM_STAGES = 2
                # p should be 0, 0, 1, 1, 0, 0, ...
                for k in range(0, tl.cdiv(K, BK)):
                    buf = k % NUM_STAGES

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(
                        data_a,
                        data_b,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    desc_out.store([offset_am + (BM // 2) * tlx.async_task_replica_id(), offset_bn], acc.to(tlx.dtype_of(desc_out)))  # noqa

                    # Flip phase after every NUM_STAGES iterations finish
                    p = (p^1) if (buf == (NUM_STAGES-1)) else p

    def matmul(a, b):
        # Check constraints.
        assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
        assert a.is_contiguous(), "Matrix A must be contiguous"

        (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
        c = torch.zeros((M, N), dtype=torch.float16, device=DEVICE, )

        BM, BN, BK = (128, 64, 128)

        desc_in_1 = TensorDescriptor(
            a,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BM // 2, BK],
        )

        desc_in_2 = TensorDescriptor(
            b,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BK, BN],
        )
        desc_out = TensorDescriptor(
            c,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BM // 2, BN],
        )

        grid = lambda META: (  # noqa E731
            triton.cdiv(M, META['BM']), triton.cdiv(N, META['BN']),
        )
        matmul_kernel_tlx_ws[grid](
            desc_in_1, desc_in_2, desc_out,  #
            K=tl.constexpr(K),
            BM=tl.constexpr(BM), 
            BN=tl.constexpr(BN), 
            BK=tl.constexpr(BK), 
            NUM_STAGES=tl.constexpr(2),  #
        )
        return c

    triton.set_allocator(alloc_fn)

    torch.manual_seed(0)
    # M, N, K = (256, 128, 64)
    M, N, K = (128, 64, 64)

    a = torch.randn((M, K), dtype=torch.float16, device=DEVICE)
    b = torch.randn((K, N), dtype=torch.float16, device=DEVICE)

    rtol = 1e-2 if is_hip_cdna2() else 0
    output = matmul(a, b)
    output_ref = torch.matmul(a, b)
    print("output.shape", output.shape)
    print("output", output)
    # print("output.split", torch.split(output, 64))
    # print("output_ref.shape", output_ref.shape)
    print("output_ref", output_ref)
    # print("output - output_ref", output - output_ref)
    assert torch.allclose(output, output_ref, atol=1e-2, rtol=rtol), "❌ Triton and Torch differ"
