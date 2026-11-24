"""autoWS persistent matmul whose code after the warp-specialized loop reads C."""

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell


@triton.jit
def matmul_post_loop_readback_kernel(A, B, C, R, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                                     BN: tl.constexpr, BK: tl.constexpr, STORE: tl.constexpr, NUM_SMS: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(A, [M, K], [K, 1], [BM, BK])
    b_desc = tl.make_tensor_descriptor(B, [K, N], [N, 1], [BK, BN])
    c_desc = tl.make_tensor_descriptor(C, [M, N], [N, 1], [BM, BN])
    c_half_desc = tl.make_tensor_descriptor(C, [M, N], [N, 1], [BM, BN // 2])
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = tl.cdiv(M, BM) * num_pid_n
    pid = tl.program_id(0)
    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=False, warp_specialize=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            a = a_desc.load([pid_m * BM, ki * BK])
            b = b_desc.load([ki * BK, pid_n * BN])
            acc += tl.dot(a, b)
        if STORE == "pointer":
            rm = (pid_m * BM + tl.arange(0, BM))[:, None]
            rn = (pid_n * BN + tl.arange(0, BN))[None, :]
            tl.store(C + rm * N + rn, acc.to(tl.bfloat16))
        elif STORE == "descriptor":
            c_desc.store([pid_m * BM, pid_n * BN], acc.to(tl.bfloat16))
        else:
            # Two subtile stores through double-buffered TMA staging.
            acc0, acc1 = acc.reshape(BM, 2, BN // 2).permute(0, 2, 1).split()
            c_half_desc.store([pid_m * BM, pid_n * BN], acc0.to(tl.bfloat16))
            c_half_desc.store([pid_m * BM, pid_n * BN + BN // 2], acc1.to(tl.bfloat16))
    # After the loop, read back the last element of this CTA's last C tile.
    last = pid + ((num_tiles - 1 - pid) // NUM_SMS) * NUM_SMS
    v = tl.load(C + ((last // num_pid_n) * BM + BM - 1) * N + (last % num_pid_n) * BN + BN - 1)
    tl.atomic_add(R + pid, v.to(tl.float32))


@pytest.mark.parametrize("STORE", ["pointer", "descriptor", "descriptor_subtiled"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_post_loop_readback(STORE):
    """Code after the warp-specialized loop used to run before the partitions'
    stores of C landed, so every CTA read back a stale value: the post-loop ops
    stayed ahead of the warp_specialize op, the single-WS lowering did not join
    the partitions, and the last subtile's TMA store was still in flight."""
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        M, N, K = 4096, 4096, 512
        BLOCK = 128
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        torch.manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        num_pid_n = triton.cdiv(N, BLOCK)
        num_tiles = triton.cdiv(M, BLOCK) * num_pid_n
        grid = min(NUM_SMS, num_tiles)
        pids = torch.arange(grid, device="cuda")
        last = pids + ((num_tiles - 1 - pids) // NUM_SMS) * NUM_SMS
        for _ in range(5):
            C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
            R = torch.zeros(grid, device="cuda", dtype=torch.float32)
            kernel = matmul_post_loop_readback_kernel[(grid, )](A, B, C, R, M, N, K, BLOCK, BLOCK, BLOCK, STORE,
                                                                NUM_SMS, num_warps=4, num_stages=3)
            assert "ttg.warp_specialize" in kernel.asm["ttgir"], "Expected warp specialization in IR"
            expected = C[(last // num_pid_n) * BLOCK + BLOCK - 1, (last % num_pid_n) * BLOCK + BLOCK - 1].float()
            torch.testing.assert_close(R, expected, atol=0, rtol=0)
        ref_c = (A.float() @ B.float()).to(torch.bfloat16)
        torch.testing.assert_close(C, ref_c, atol=1e-2, rtol=1e-2)
