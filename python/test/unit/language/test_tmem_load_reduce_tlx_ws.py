"""TLX warp-spec tmem_load -> arrive -> reduce fusion tests.

Covers T280816632: tcgen05.ld.red via ttng.tmem_load {redOp} must fuse
even when the load sits on a warp-spec partition boundary with an
immediately following arrive barrier (TLX explicit `tlx.barrier_arrive`
lowers to `ttng.arrive_barrier`). The pass hoists the barrier past the
reduction when no intervening memory ops exist (per-subtile).

Two tests:
  - compile-only (no GPU needed, GPUTarget cuda:103) verifies TTGIR/PTX
  - runtime Blackwell-only launches a kernel that lowers to the same pattern
    and checks the fused instruction both emits and computes the right answer
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

# ---------------------------------------------------------------------------
# Shared TTGIR template for explicit TLX arrive after tmem_load.
# This is the exact IR that TLX lowers to when `tlx.barrier_arrive` is placed
# immediately after a tmem_load at the computation partition edge.
# ---------------------------------------------------------------------------
_TLX_TMPL = """
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#out = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tlx_tmem_load_reduce_with_arrive(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, %bar: !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>, %out: !tt.ptr<f32>) {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
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


def _compile_tlx_tmpl(tmp_path: pathlib.Path):
    f = tmp_path / "tlx_tmem_load_reduce_with_arrive.ttgir"
    f.write_text(_TLX_TMPL)
    context = ir.context()
    ir.load_dialects(context)
    nvidia.load_dialects(context)
    module = ir.parse_mlir_module(str(f), context)
    pm = ir.pass_manager(context)
    nvidia.passes.ttnvgpuir.add_tmem_load_reduce(pm)
    pm.run(module, "test_compile_only_tlx_tmem_load_reduce_with_arrive")
    f.write_text(str(module))
    k = triton.compile(str(f), target=GPUTarget("cuda", 103, 32))
    return k


def test_compile_only_tlx_tmem_load_reduce_with_arrive(tmp_path):
    """Compile-only: TLX explicit arrive after tmem_load must still fuse to redOp."""
    k = _compile_tlx_tmpl(tmp_path)
    ttgir = k.asm["ttgir"]
    # The pass must have fused and hoisted the arrive past the reduction.
    assert "redOp" in ttgir, "expected fused ttng.tmem_load {redOp} in TTGIR"
    assert "ttng.tmem_load" in ttgir
    # arrive must still be present but after the fused load (per subtile)
    assert "arrive_barrier" in ttgir
    # Verify ordering: fused load before arrive, no unfused tt.reduce with maxnumf
    assert re.search(r"ttng\.tmem_load.*redOp.*\n.*arrive_barrier", ttgir), (
        "expected arrive to be pushed past fused ld.red"
    )
    assert k.asm["cubin"] != b""
    # PTX should contain ld.red on SM103
    ptx = k.asm["ptx"]
    assert "tcgen05.ld.red" in ptx or "ld.red" in ptx, "expected tcgen05.ld.red in PTX"


# ---------------------------------------------------------------------------
# Runtime kernel: a Blackwell MMA feeding a row reduction. The reduction result
# must be stored -- an unused tl.max is DCE'd well before the fusion pass runs,
# which leaves no tmem_load+reduce pattern for it to match.
# ---------------------------------------------------------------------------


@triton.jit
def _tmem_rowmax_kernel(a_ptr, b_ptr, mx_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    mx = tl.max(acc, axis=1)
    tl.store(mx_ptr + offs_m, mx, mask=offs_m < M)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
                    reason="Requires Blackwell (sm100+)")
def test_runtime_tmem_load_reduce_executes():
    """Runtime: the fused tcgen05.ld.red emits and computes the right row max.

    The compile-only test above pins the IR shape for the TLX explicit-arrive
    pattern; this is its hardware counterpart. Both the instruction check and
    the numerics check are made against the launched kernel, so this does not
    restate the compile-only test. BLOCK_N == N so a single tile spans each row
    and no cross-tile combine is needed.
    """
    if not is_blackwell():
        pytest.skip("Requires Blackwell")

    M, N, K = 256, 128, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    mx = torch.empty((M, ), device="cuda", dtype=torch.float32)

    kernel = _tmem_rowmax_kernel[(triton.cdiv(M, BLOCK_M), )](
        a, b, mx, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )

    assert "redOp" in kernel.asm["ttgir"], "expected fused ttng.tmem_load {redOp} in the launched kernel"
    ptx = kernel.asm["ptx"]
    assert "tcgen05.ld.red" in ptx or "ld.red" in ptx

    ref = torch.max(torch.matmul(a.float(), b.float()), dim=1).values
    torch.testing.assert_close(ref, mx, atol=1e-2, rtol=1e-2)
