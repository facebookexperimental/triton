"""MetaAutoWS warp-spec tmem_load -> arrive -> reduce fusion runtime test.

Covers T280816632: when MetaAutoWS (warp_specialize=True) places a
tmem_load on the computation partition boundary, the code partitioner
inserts an `arrive` immediately after the load. The
`triton-nvidia-tmem-load-reduce` pass must push that arrive past the
row reduction (per subtile, no intervening memory ops) and fuse to
`tcgen05.ld.red`.

Runtime Blackwell Ultra-only: drives the real AutoWS pipeline, then asserts
the fusion survived it and that the row maxima the kernel computed are correct.
The compile-only counterpart lives in
test_tmem_sm100_compile_load_reduce_autows.py.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell_ultra
from triton.tools.tensor_descriptor import TensorDescriptor


# ---------------------------------------------------------------------------
# Runtime AutoWS kernel: persistent TMA matmul with warp_specialize whose only
# epilogue is a post-MMA row reduction, lowering to the same tmem_load+reduce
# pattern. With `separate_epilogue_store` and per-subtile TMEM, each subtile's
# tmem_load sits on the partition edge with an arrive before the reduce.
# ---------------------------------------------------------------------------
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def autows_matmul_reduce_kernel(
    a_desc, b_desc, mx_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in tl.range(
            start_pid, num_tiles, NUM_SMS,
            flatten=False,
            warp_specialize=True,
            disallow_acc_multi_buffer=True,
            data_partition_factor=1,
            separate_epilogue_store=True,
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS)
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            acc = tl.dot(a, b.T, acc)
        # The reduction result has to be stored: an unused tl.max is DCE'd
        # before the fusion pass runs, leaving no pattern to fuse.
        mx = tl.max(acc, axis=1)
        offs_m = offs_am + tl.arange(0, BLOCK_M)
        tl.store(mx_ptr + offs_m, mx, mask=offs_m < M)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
def test_runtime_autows_tmem_load_reduce_with_arrive():
    """Runtime: AutoWS puts an arrive on the tmem_load edge and it still fuses.

    The compile-only test pins the isolated TTGIR pattern. This one drives the
    real AutoWS pipeline end to end and asserts the fusion survives it, so the
    assertions below are all made against the launched kernel's own asm --
    re-checking the template here would just restate the compile-only test.
    N == BLOCK_N so one tile covers each row and no cross-tile combine is needed.
    """
    if not is_blackwell_ultra():
        pytest.skip("Requires Blackwell Ultra")

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        M, N, K = 256, 128, 128
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        GROUP_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        dtype = torch.float16
        A = torch.randn((M, K), dtype=dtype, device="cuda")
        B = torch.randn((N, K), dtype=dtype, device="cuda")
        MX = torch.empty((M, ), dtype=torch.float32, device="cuda")

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)
        a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_N, BLOCK_K])

        grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])),)

        kernel = autows_matmul_reduce_kernel[grid](
            a_desc, b_desc, MX,
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M, NUM_SMS=NUM_SMS,
            num_warps=4, num_stages=3,
        )
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir
        assert "redOp" in ttgir, "expected fused ttng.tmem_load {redOp} in the AutoWS kernel"
        ptx = kernel.asm["ptx"]
        assert "tcgen05.ld.red" in ptx, "expected tcgen05.ld.red in PTX"

        ref = torch.max(torch.matmul(A.float(), B.T.float()), dim=1).values
        torch.testing.assert_close(ref, MX, atol=1e-2, rtol=1e-2)
