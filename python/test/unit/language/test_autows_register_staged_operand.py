"""autoWS persistent TMA matmul whose MMA operand may be staged through registers.

An FP32 (TF32) operand that is not K-major is staged through registers into the
MMA's own partition, so warp specialization must bail out for it. Every other
layout must still be warp-specialized: correctness alone cannot see a bail-out,
so the TTGIR is checked too.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell


@triton.jit
def persistent_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, A_KMAJOR: tl.constexpr, B_KMAJOR: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NUM_SMS: tl.constexpr):
    if A_KMAJOR:
        a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    else:
        a_desc = tl.make_tensor_descriptor(a_ptr, [K, M], [M, 1], [BK, BM])
    if B_KMAJOR:
        b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BN, BK])
    else:
        b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = tl.cdiv(M, BM) * num_pid_n
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, warp_specialize=True):
        om = (tile_id // num_pid_n) * BM
        on = (tile_id % num_pid_n) * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            if A_KMAJOR:
                a = a_desc.load([om, k])
            else:
                a = a_desc.load([k, om]).T
            if B_KMAJOR:
                b = b_desc.load([on, k]).T
            else:
                b = b_desc.load([k, on])
            acc = tl.dot(a, b, acc)
        rm = om + tl.arange(0, BM)
        rn = on + tl.arange(0, BN)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


@pytest.mark.parametrize("num_stages", [1, 3])
@pytest.mark.parametrize("B_KMAJOR", [True, False])
@pytest.mark.parametrize("A_KMAJOR", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_register_staged_operand(dtype, A_KMAJOR, B_KMAJOR, num_stages):
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        M, N, K = 2048, 2048, 512
        BM, BN, BK = 128, 128, 32
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        torch.manual_seed(0)
        # Small integers are exact in TF32/bf16 and their products and sums are
        # exact in FP32, so the result must match the reference bitwise.
        A = torch.randint(-4, 5, (M, K), device="cuda").to(dtype)
        B = torch.randint(-4, 5, (K, N), device="cuda").to(dtype)
        a_arg = A if A_KMAJOR else A.t().contiguous()
        b_arg = B.t().contiguous() if B_KMAJOR else B
        C = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float32)
        kernel = persistent_matmul_kernel[(NUM_SMS, )](a_arg, b_arg, C, M, N, K, A_KMAJOR, B_KMAJOR, BM, BN, BK,
                                                       NUM_SMS, num_warps=4, num_stages=num_stages)
        staged = dtype == torch.float32 and not (A_KMAJOR and B_KMAJOR)
        ttgir = kernel.asm["ttgir"]
        if staged:
            assert "ttg.warp_specialize" not in ttgir, "Register-staged MMA operand must not be warp-specialized"
        else:
            assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        ref = (A.double() @ B.double()).float()
        torch.testing.assert_close(C, ref, atol=0, rtol=0)
