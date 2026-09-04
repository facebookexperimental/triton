"""MetaAutoWS warp-spec tmem_load -> arrive -> reduce fusion tests.

Covers T280816632: when MetaAutoWS (warp_specialize=True) places a
tmem_load on the computation partition boundary, the code partitioner
inserts an `arrive` immediately after the load. The
`triton-nvidia-tmem-load-reduce` pass must push that arrive past the
row reduction (per subtile, no intervening memory ops) and fuse to
`tcgen05.ld.red`.

Two tests:
  - compile-only (GPUTarget cuda:103) verifies IR contains pure barrier-slot
    indexing and arrive after fused ld.red
  - runtime Blackwell-only drives the real AutoWS pipeline, then asserts the
    fusion survived it and that the row maxima the kernel computed are correct
"""

import re
import pathlib

import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir, nvidia
from triton.backends.compiler import GPUTarget
from triton._internal_testing import is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor


_AUTOWS_TMPL = """
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#out = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @autows_tmem_load_reduce_with_arrive(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, %bars: !ttg.memdesc<2x1xi64, #shared, #ttg.shared_memory, mutable>, %bar_index: i32, %out: !tt.ptr<f32>) {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %bar = ttg.memdesc_index %bars[%bar_index] : !ttg.memdesc<2x1xi64, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = ttg.convert_layout %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #out>
    %3 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #out>
    %4 = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #out>
    %5 = tt.addptr %4, %3 : tensor<128x!tt.ptr<f32>, #out>, tensor<128xi32, #out>
    tt.store %5, %2 : tensor<128x!tt.ptr<f32>, #out>
    tt.return
  }
}
"""


def _compile_autows_tmpl(tmp_path: pathlib.Path):
    f = tmp_path / "autows_tmem_load_reduce_with_arrive.ttgir"
    f.write_text(_AUTOWS_TMPL)
    context = ir.context()
    ir.load_dialects(context)
    nvidia.load_dialects(context)
    module = ir.parse_mlir_module(str(f), context)
    pm = ir.pass_manager(context)
    nvidia.passes.ttnvgpuir.add_tmem_load_reduce(pm)
    pm.run(module, "test_compile_only_autows_tmem_load_reduce_with_arrive")
    f.write_text(str(module))
    k = triton.compile(str(f), target=GPUTarget("cuda", 103, 32))
    return k


def test_compile_only_autows_tmem_load_reduce_with_arrive(tmp_path):
    """Compile-only: AutoWS-generated arrive after tmem_load must still fuse."""
    k = _compile_autows_tmpl(tmp_path)
    ttgir = k.asm["ttgir"]
    assert "redOp" in ttgir, "expected fused ld.red in TTGIR"
    assert "memdesc_index" in ttgir
    assert "arrive_barrier" in ttgir
    assert re.search(
        r"ttng\.tmem_load.*redOp[\s\S]*memdesc_index[\s\S]*arrive_barrier",
        ttgir,
    ), "expected pure barrier indexing and arrive after fused load"
    assert k.asm["cubin"] != b""
    assert "tcgen05.ld.red" in k.asm["ptx"] or "ld.red" in k.asm["ptx"]


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


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
                    reason="Requires Blackwell (sm100+)")
def test_runtime_autows_tmem_load_reduce_with_arrive():
    """Runtime: AutoWS puts an arrive on the tmem_load edge and it still fuses.

    The compile-only test pins the isolated TTGIR pattern. This one drives the
    real AutoWS pipeline end to end and asserts the fusion survives it, so the
    assertions below are all made against the launched kernel's own asm --
    re-checking the template here would just restate the compile-only test.
    N == BLOCK_N so one tile covers each row and no cross-tile combine is needed.
    """
    if not is_blackwell():
        pytest.skip("Requires Blackwell")

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
        assert "tcgen05.ld.red" in ptx or "ld.red" in ptx

        ref = torch.max(torch.matmul(A.float(), B.T.float()), dim=1).values
        torch.testing.assert_close(ref, MX, atol=1e-2, rtol=1e-2)
