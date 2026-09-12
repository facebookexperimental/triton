"""MetaAutoWS warp-spec tmem_load -> arrive -> reduce fusion compile-only test.

Covers T280816632: when MetaAutoWS (warp_specialize=True) places a
tmem_load on the computation partition boundary, the code partitioner
inserts an `arrive` immediately after the load, and the
`triton-nvidia-tmem-load-reduce` pass must still fuse to `tcgen05.ld.red`.
This is the compile-only version of the runtime test in
test_tmem_sm103_load_reduce_autows.py: it compiles the same kernel for
sm103 and checks the fused instruction in PTX instead of launching it.
It runs on sm100 CI, where no sm103 hardware is available.
"""

# NOTE: delete this file once GB300 or Rubin is available in CI: the sm103
# runtime test will cover this pattern on real hardware, and this
# compile-only stand-in will no longer be needed.

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


# Duplicated from test_tmem_sm103_load_reduce_autows.py; keep in sync.
# Persistent TMA matmul with warp_specialize whose only epilogue is a
# post-MMA row reduction. With `separate_epilogue_store` and per-subtile
# TMEM, each subtile's tmem_load sits on the partition edge with an arrive
# before the reduce.
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


# Compile-only (no launch); gated to Blackwell just to reduce test bloat.
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_compile_only_autows_tmem_load_reduce_with_arrive():
    """Compile-only: the runtime kernel fuses to tcgen05.ld.red for sm103."""
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        src = ASTSource(
            fn=autows_matmul_reduce_kernel,
            signature={
                # Must match the JIT's tensordesc<dtype[block_shape]>
                # specialization for the runtime test's TensorDescriptors.
                "a_desc": "tensordesc<fp16[128, 64]>",
                "b_desc": "tensordesc<fp16[128, 64]>",
                "mx_ptr": "*fp32",
                "M": "i32",
                "N": "i32",
                "K": "i32",
                "BLOCK_M": "constexpr",
                "BLOCK_N": "constexpr",
                "BLOCK_K": "constexpr",
                "GROUP_M": "constexpr",
                "NUM_SMS": "constexpr",
            },
            constexprs={
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
                "NUM_SMS": torch.cuda.get_device_properties("cuda").multi_processor_count,
            },
        )
        k = triton.compile(src, target=GPUTarget("cuda", 103, 32), options={"num_warps": 4, "num_stages": 3})
        assert k.asm["cubin"] != b""
        assert "tcgen05.ld.red" in k.asm["ptx"], "expected tcgen05.ld.red in PTX"
