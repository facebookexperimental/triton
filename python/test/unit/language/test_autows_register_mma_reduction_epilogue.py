"""autoWS persistent GEMM whose dot stays a register tt.dot (BLOCK_M=32) with a
row-reduction epilogue (partition-scheduler bug #48)."""

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def matmul_row_center_kernel(a_desc, b_desc, c_desc, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                             NUM_SMS: tl.constexpr, MERGE_EPILOGUE: tl.constexpr,
                             MERGE_EPILOGUE_TO_COMPUTATION: tl.constexpr):
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = tl.cdiv(M, BM) * num_pid_n
    k_tiles = tl.cdiv(K, BK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=False, warp_specialize=True,
                            data_partition_factor=1, separate_epilogue_store=True, merge_epilogue=MERGE_EPILOGUE,
                            merge_epilogue_to_computation=MERGE_EPILOGUE_TO_COMPUTATION):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([pid_m * BM, ki * BK])
            b = b_desc.load([ki * BK, pid_n * BN])
            acc = tl.dot(a, b, acc)
        # Row reduction in the epilogue (LayerNorm-style centering).
        mean = tl.sum(acc, axis=1) / BN
        c_desc.store([pid_m * BM, pid_n * BN], acc - mean[:, None])


@pytest.mark.parametrize("merge", ["none", "merge_epilogue", "merge_epilogue_to_computation"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_register_mma_reduction_epilogue(merge):
    """The reduction epilogue used to send this kernel down the no-MMA
    reduction layout, which split the register dot from the K loop carrying its
    accumulator: the first persistent tile of every CTA was wrong, and the
    result changed from launch to launch. With merge_epilogue or
    merge_epilogue_to_computation there is no separate epilogue partition and
    the reduction layout keeps the dot, its K loop and the epilogue together."""
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        # The shape of the Inductor addmm template that hit the bug: BM=32
        # keeps the dot a register mma.sync, k_tiles=2, and 193 tiles is more
        # than one wave on every Blackwell part.
        M, N, K = 6167, 64, 64
        BM, BN, BK = 32, 64, 32
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        assert num_tiles > NUM_SMS
        torch.manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        acc = A.float() @ B.float()
        ref = acc - acc.mean(dim=1, keepdim=True)
        a_desc = TensorDescriptor.from_tensor(A, [BM, BK])
        b_desc = TensorDescriptor.from_tensor(B, [BK, BN])
        grid = (min(NUM_SMS, num_tiles), )
        for _ in range(10):
            C = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float32)
            c_desc = TensorDescriptor.from_tensor(C, [BM, BN])
            kernel = matmul_row_center_kernel[grid](a_desc, b_desc, c_desc, M, N, K, BM, BN, BK, NUM_SMS,
                                                    merge == "merge_epilogue", merge == "merge_epilogue_to_computation",
                                                    num_warps=4, num_stages=3)
            torch.testing.assert_close(C, ref, atol=1e-5, rtol=1e-5)
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" not in ttgir, "Expected a register tt.dot"
