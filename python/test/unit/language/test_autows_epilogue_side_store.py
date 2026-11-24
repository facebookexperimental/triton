"""autoWS persistent matmul with an epilogue store that does not depend on the accumulator."""

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell


@triton.jit
def matmul_side_store_kernel(A, B, C, X, D, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                             BN: tl.constexpr, BK: tl.constexpr, OWN_MASK: tl.constexpr, NUM_SMS: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(A, [M, K], [K, 1], [BM, BK])
    b_desc = tl.make_tensor_descriptor(B, [K, N], [N, 1], [BK, BN])
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = tl.cdiv(M, BM) * num_pid_n
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=False, warp_specialize=True,
                            data_partition_factor=1, separate_epilogue_store=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            a = a_desc.load([pid_m * BM, ki * BK])
            b = b_desc.load([ki * BK, pid_n * BN])
            acc += tl.dot(a, b)
        rm = (pid_m * BM + tl.arange(0, BM))[:, None]
        rn = (pid_n * BN + tl.arange(0, BN))[None, :]
        mask = (rm < M) & (rn < N)
        idx = rm * N + rn
        tl.store(C + idx, acc.to(tl.bfloat16), mask)
        # Side store: its value comes from a pointer load, not from the accumulator.
        x = tl.load(X + idx, mask)
        if OWN_MASK:
            tl.store(D + idx, x * 2.0, idx < M * N)
        else:
            tl.store(D + idx, x * 2.0, mask)


@pytest.mark.parametrize("OWN_MASK", [False, True])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_epilogue_side_store(OWN_MASK):
    """The side store used to get no partition: with its own mask it was
    silently dropped, and with the accumulator store's mask its pointer was
    captured into the warp-specialized region as a tensor and overran shared
    memory (illegal memory access)."""
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        M, N, K = 1024, 128, 2048
        BLOCK = 128
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        torch.manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        X = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        C = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        D = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        num_tiles = triton.cdiv(M, BLOCK) * triton.cdiv(N, BLOCK)
        kernel = matmul_side_store_kernel[(min(NUM_SMS, num_tiles), )](A, B, C, X, D, M, N, K, BLOCK, BLOCK, BLOCK,
                                                                       OWN_MASK, NUM_SMS, num_warps=8, num_stages=3)
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert ttgir.count("tt.store") == 2, "Expected both epilogue stores to survive"
        torch.testing.assert_close(D, X * 2.0, atol=0, rtol=0)
        ref_c = (A.double() @ B.double()).to(torch.bfloat16)
        torch.testing.assert_close(C, ref_c, atol=1e-2, rtol=1e-2)
