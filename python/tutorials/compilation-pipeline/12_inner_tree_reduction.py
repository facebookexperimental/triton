"""12 — inner_tree ordering: a compiler change that makes a reduction layout-stable.

This is the *fix* for the negative result in 08. There, a ``tl.sum`` gave a
different bit pattern at each ``num_warps`` because the layout chose a different
shuffle-tree order (FP add is non-associative). The bitwise-equivalence project
adds a **reduction ordering** knob that constrains the lowering so the order no
longer depends on the layout:

    tl.sum(x, axis=0, reduction_ordering=tl.ReductionOrdering.INNER_TREE)

The handling lives in the **TTGIR -> LLVM** pass ``convert-triton-gpu-to-llvm``
(``lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`` — see ``isInnerTree``,
``reduceValueSequence``, ``warpReduce``). With INNER_TREE the warp reduction uses
a **count-up** shuffle order (``shuffleXor`` with N = 1, 2, 4, ...) that builds
the tree from adjacent lanes first, so the tree structure depends on lane
*proximity*, not on the total number of active lanes — hence the same bits at any
``num_warps``. The default (``UNORDERED``) keeps the classic count-down butterfly,
which is faster to express but layout-sensitive.

BIT-CHANGING but stabilized: we run num_warps in {1,2,4,8} both ways and group
the exact results. UNORDERED splits into several bitwise classes (as in 08);
INNER_TREE collapses to exactly one — that single class is what an equivalence
checker can rely on across configs.

Run:  python python/tutorials/compilation-pipeline/12_inner_tree_reduction.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, dump_passes, is_cuda, pass_diff


@triton.jit
def sum_kernel(src, dst, N, BLOCK: tl.constexpr, ORD: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0, reduction_ordering=ORD))


def _classes(src, N, ordering, warps):
    """Run the sum at each num_warps; group results into bitwise-equivalence classes."""
    classes = {}
    for nw in warps:
        out = torch.empty(1, device="cuda", dtype=torch.float32)
        sum_kernel[(1, )](src, out, N, BLOCK=N, ORD=ordering, num_warps=nw)
        torch.cuda.synchronize()
        classes.setdefault(out.item().hex(), []).append(nw)
    return classes


def main():
    N = 4096
    torch.manual_seed(0)
    src = torch.randn(N, device="cuda", dtype=torch.float32)
    warps = [1, 2, 4, 8]

    banner("12 — UNORDERED: the reduction order (and bits) still moves with num_warps")
    unordered = _classes(src, N, tl.ReductionOrdering.UNORDERED, warps)
    for bits, members in unordered.items():
        print(f"    {bits}  <- num_warps={members}")

    banner("12 — INNER_TREE: one bitwise class across every num_warps")
    inner = _classes(src, N, tl.ReductionOrdering.INNER_TREE, warps)
    for bits, members in inner.items():
        print(f"    {bits}  <- num_warps={members}")

    # The pass that implements the ordering: `convert-triton-gpu-to-llvm`. Grep the
    # shuffle ops to see the warp-reduction tree it emits for INNER_TREE.
    banner("12 — the pass responsible: convert-triton-gpu-to-llvm (grep=shfl)")
    dumps = dump_passes(sum_kernel, src, torch.empty(1, device="cuda"), N, BLOCK=N, ORD=tl.ReductionOrdering.INNER_TREE,
                        num_warps=4, grid=(1, ))
    pass_diff(dumps, "convert-triton-gpu-to-llvm", grep="shfl", limit=8)

    ref = src.sum()
    for cls in (unordered, inner):
        for bits in cls:
            torch.testing.assert_close(torch.tensor(float.fromhex(bits)), ref.cpu(), atol=1e-3, rtol=1e-3)
    assert len(unordered) > 1, "expected UNORDERED to split into several bitwise classes (like 08)"
    assert len(inner) == 1, "INNER_TREE must give ONE bitwise class for all num_warps"
    print(f"\n[OK] UNORDERED produced {len(unordered)} bitwise classes but INNER_TREE produced 1 —"
          " the inner_tree handling makes the reduction layout-stable (and bitwise-checkable).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
