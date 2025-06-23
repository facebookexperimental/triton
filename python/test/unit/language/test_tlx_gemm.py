import pytest
import torch

import triton
import triton.language as tl
import triton.tlx.language as tlx

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
def test_ws_gemm(device):

    @triton.jit
    def matmul_kernel_tlx_ws(
        a_ptr, b_ptr, c_ptr, #
        M, N, K, #
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,  #
    ):
        NUM_STAGES=2
        # Descriptor
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        # offset_am = pid_m * BM
        # offset_bn = pid_n * BN

        # tma_desc_a = tl.make_tensor_descriptor(
        #         a_ptr,
        #         shape=[M, K],
        #         strides=[K, 1],
        #         block_shape=[BM, BK],
        #     )

        # tma_desc_b = tl.make_tensor_descriptor(
        #         b_ptr,
        #         shape=[K, N],
        #         strides=[N, 1],
        #         block_shape=[BK, BN],
        #     )

        # tma_desc_c = tl.make_tensor_descriptor(
        #         c_ptr,
        #         shape=[M, N],
        #         strides=[N, 1],
        #         block_shape=[BM, BN],
        #     )

        # Need NUM_STAGES sets of SMEM buffers for A and B
        # where each set contains two for A and one for B.
        # Split A into two in M-dimension to have two consumer tasks for wgmma
        a = tlx.local_alloc((BM // 2, BK), tl.float16, tl.constexpr(4))
        b = tlx.local_alloc((BN, BK), tl.float16, tl.constexpr(2))

        size_a = (BM//2) * BK * 2 # sizeof(dtype)
        size_b = BN * BK * 2 # * sizeof(dtype)

        # Need NUM_STAGES sets of mbarriers for A and B
        # where each set contains two for A and one for B.
        # Do the above for both empty states and full states respectively.
        # bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * 2)
        # bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * 2)
        # bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES)
        # bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES)

        # # Warp specilization
        # with tlx.async_tasks():
        #     # Producer (async load)
        #     with tlx.async_task("default"):
        #         p = 1  # phase
        #         for k in range(0, tl.cdiv(K, BK)):
        #             buf = k % NUM_STAGES
        #             offset_k = k * BK

        #             # Async load to a[buf]
        #             empty_a_1st = tlx.local_view(bars_empty_a, buf)
        #             full_a_1st = tlx.local_view(bars_full_a, buf)
        #             data_a_1st = tlx.local_view(a, buf)
        #             tlx.barrier_wait(bar=empty_a_1st, phase=p)
        #             tlx.barrier_expect_bytes(full_a_1st, size_a)
        #             tlx.async_descriptor_load(
        #                 tma_desc_a,
        #                 data_a_1st,
        #                 [offset_am, offset_k],
        #                 full_a_1st)

        #             # Async load to b[buf]
        #             empty_b = tlx.local_view(bars_empty_b, buf)
        #             full_b = tlx.local_view(bars_full_b, buf)
        #             data_b = tlx.local_view(a, buf)
        #             tlx.barrier_wait(bar=empty_b, phase=p)
        #             tlx.barrier_expect_bytes(bar=full_b, size=size_b)
        #             tlx.async_descriptor_load(
        #                 tma_desc_b,
        #                 data_b,
        #                 [offset_k, offset_bn],
        #                 full_a_1st)

        #             # Async load to a[buf+NUM_STAGES]
        #             empty_a_2nd = tlx.local_view(bars_empty_a, buf+NUM_STAGES)
        #             full_a_2nd = tlx.local_view(bars_full_a, buf+NUM_STAGES)
        #             data_a_2nd = tlx.local_view(a, buf+NUM_STAGES)
        #             tlx.barrier_wait(bar=empty_a_2nd, phase=p)
        #             tlx.barrier_expect_bytes(bar=full_a_2nd, size=size_a)
        #             tlx.async_descriptor_load(
        #                 tma_desc_a,
        #                 data_a_2nd,
        #                 [offset_am + (BM//2), offset_k],
        #                 full_a_2nd)

        #             # Flip phase after every NUM_STAGES iterations finish
        #             p = p if (buf < (NUM_STAGES-1)) else (p^1)

        #     # consumers (wgmma + async store)
        #     with tlx.async_task(num_warps=4, replicate=2):
        #         p = 0  # phase
        #         acc = tl.zeros((BM//2, BN), dtype=tl.bfloat16)
        #         for k in range(0, tl.cdiv(K, BK)):
        #             buf = k % NUM_STAGES
        #             # Flip phase before every NUM_STAGES iterations begin
        #             p = p if (buf>0) else (p^1)

        #             # Local vars for this iteration
        #             full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
        #             empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
        #             data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id()) # noqa
        #             full_b = tlx.local_view(bars_full_b, buf)
        #             empty_b = tlx.local_view(bars_empty_b, buf)
        #             data_b = tlx.local_view(b, buf)

        #             # Wait
        #             tlx.barrier_wait(bar=full_a, phase=p)
        #             tlx.barrier_wait(bar=full_b, phase=p)

        #             # async_dot
        #             # acc = tlx.async_dot(
        #             #     data_a,
        #             #     data_b,
        #             #     col_input=0,
        #             #     col_other=0,
        #             # )
        #             # async_wait
        #             # acc = tlx.async_dot_wait(tl.constexpr(0), acc)

        #             # Release buffers
        #             tlx.barrier_arrive(empty_a)
        #             tlx.barrier_arrive(empty_b)

                # tlx.async_descriptor_store(tma_desc_c, acc, [offset_am, offset_bn])

    def matmul(a, b):
        # Check constraints.
        assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        M, K = a.shape
        K, N = b.shape
        BM, BN, BK = 128, 128, 128
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (  # noqa E731
            triton.cdiv(M, META['BM']), triton.cdiv(N, META['BN']),
        )
        matmul_kernel_tlx_ws[grid](
            a, b, c,  #
            M, N, K,  #
            BM=tl.constexpr(BM), 
            BN=tl.constexpr(BN), 
            BK=tl.constexpr(BK), 
        )
        return c


    torch.manual_seed(0)
    M, K, N = 512, 512, 512
    a = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16) # M x K
    b = torch.randn((K, N), device=DEVICE, dtype=torch.bfloat16) # K x N
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    rtol = 1e-2 if is_hip_cdna2() else 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
