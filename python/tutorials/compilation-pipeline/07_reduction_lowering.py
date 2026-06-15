"""07 — Reduction lowering: tt.reduce becomes a warp-shuffle tree.

Pipeline: a ``tl.sum`` is a ``tt.reduce`` op (with a combine region holding
``arith.addf``) all the way through TTGIR. The tree is materialized in the
**TTGIR -> LLVM** step, ``convert-triton-gpu-to-llvm``
(``lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp``): each thread first folds
its own elements, then warps combine across a butterfly **shuffle** tree
(``nvvm.shfl.sync bfly`` -> PTX ``shfl.sync.bfly``).

What to notice: the ``tt.reduce`` region survives untouched until the LLVM step,
where ``shfl.sync`` instructions appear — that is the cross-lane reduction tree.

Bit-neutral mechanic: for a *fixed* layout the tree is fixed, so the result is
deterministic (re-running is bitwise-identical). The reduction order is only
``layout``-dependent — change the layout and the bits can move (see 08).

Run:  python python/tutorials/compilation-pipeline/07_reduction_lowering.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, dump_passes, is_cuda, pass_diff, show


@triton.jit
def sum_kernel(src, dst, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))


def main():
    N = 4096
    torch.manual_seed(0)
    src = torch.randn(N, device="cuda", dtype=torch.float32)
    dst = torch.empty(1, device="cuda", dtype=torch.float32)

    ck = compile_only(sum_kernel, src, dst, N, BLOCK=N, num_warps=4, grid=(1, ))

    banner("07 — the reduction is a tt.reduce op with a combine region (TTGIR)")
    show(ck, "ttgir", grep=["tt.reduce", "arith.addf", "reduce.return"], limit=6)

    banner("07 — lowering to LLVM materializes the warp-shuffle tree")
    print(f"    shuffle ops in LLVM IR: nvvm.shfl.sync = {count(ck, 'llir', 'shfl')}")
    print(f"    shuffle ops in PTX    : shfl.sync       = {count(ck, 'ptx', 'shfl.sync')}")

    # The pass that builds the tree: `convert-triton-gpu-to-llvm`.
    banner("07 — the pass responsible: convert-triton-gpu-to-llvm (grep=shfl)")
    dumps = dump_passes(sum_kernel, src, dst, N, BLOCK=N, num_warps=4, grid=(1, ))
    pass_diff(dumps, "convert-triton-gpu-to-llvm", grep="shfl", limit=8)

    # Bit-neutral for a fixed layout: the tree is fixed => deterministic result.
    a = torch.empty(1, device="cuda")
    b = torch.empty(1, device="cuda")
    sum_kernel[(1, )](src, a, N, BLOCK=N, num_warps=4)
    sum_kernel[(1, )](src, b, N, BLOCK=N, num_warps=4)
    torch.cuda.synchronize()
    assert torch.equal(a, b), "same layout must give a bitwise-identical reduction"
    torch.testing.assert_close(a, src.sum().reshape(1), atol=1e-3, rtol=1e-3)
    print("\n[OK] the tree-reduction is deterministic for a fixed layout and matches torch.sum"
          " (which reduction *order* the layout picks is the subject of 08).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
