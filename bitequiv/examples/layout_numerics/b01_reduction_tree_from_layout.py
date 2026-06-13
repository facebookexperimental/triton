"""b01 — Data layout fixes the reduction TREE, which fixes the bits.

THE core mechanism of the bitwise-equivalence project. Source:
``lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`` (within-thread fold +
cross-warp shuffle tree). Deep dive:
``bitequiv/knowledge-base/tree-reduction-in-ptx-and-triton.md``.

A ``tl.sum`` over one axis is lowered to: each thread folds its own elements, then
warps combine across a shuffle tree. WHICH element lives in which lane/warp is the
**layout** (``sizePerThread`` / ``threadsPerWarp`` / ``warpsPerCTA`` along the
reduce axis), and ``num_warps`` changes it. A different layout => a different tree
=> a different summation order => (because FP add is non-associative) **different
bits**.

BIT-CHANGING. We compile the same kernel at ``num_warps in {1,2,4,8}``, show the
layout differs in TTGIR, then run each and group the *exact* results into
bitwise-equivalence classes.

Subtlety worth seeing: not every layout difference actually changes the bits here
(``num_warps`` 2/4/8 happen to coincide; 1 does not). You cannot eyeball which
configs are equivalent — that is exactly why the project builds a checker. The
static TTGIR signature is *conservative*: it separates configs whose
``warpsPerCTA@axis`` differs even when the bits happen to match, so the kept set is
always a safe subset. (Enforcing equivalence regardless of layout: see b02.)

Run:  python bitequiv/examples/layout_numerics/b01_reduction_tree_from_layout.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, first, hexbits, is_cuda, show


@triton.jit
def sum_kernel(src, dst, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))


def _run(src, N, num_warps):
    out = torch.empty(1, device="cuda", dtype=torch.float32)
    sum_kernel[(1, )](src, out, N, BLOCK=N, num_warps=num_warps)
    torch.cuda.synchronize()
    return out


def main():
    N = 4096
    torch.manual_seed(0)
    src = torch.randn(N, device="cuda", dtype=torch.float32)
    warps = [1, 2, 4, 8]

    banner("b01 — same reduction, layout along the reduce axis changes with num_warps")
    for nw in warps:
        ck = compile_only(sum_kernel, src, torch.empty(1, device="cuda"), N, BLOCK=N, num_warps=nw, grid=(1, ))
        layout = first(ck, "ttgir", "#blocked =")
        print(f"    num_warps={nw}:  {layout.split('= ', 1)[-1]}")
    print("\n    tt.reduce op (note reduction_ordering):")
    ck = compile_only(sum_kernel, src, torch.empty(1, device="cuda"), N, BLOCK=N, num_warps=4, grid=(1, ))
    show(ck, "ttgir", grep='"tt.reduce"', limit=1)

    banner("b01 — exact results grouped into bitwise-equivalence classes")
    ref = src.double().sum().item()
    classes = {}
    for nw in warps:
        out = _run(src, N, nw)
        bits = hexbits(out)
        classes.setdefault(bits, []).append(nw)
        print(f"    num_warps={nw}:  {out.item():.10f}  (raw i32 bits={bits})  "
              f"err vs fp64 sum={abs(out.double().item() - ref):.3e}")
    print("\n    bitwise-equivalence classes (by exact bits):")
    for bits, members in classes.items():
        print(f"        num_warps {members}  ->  bits={bits}")

    # The claim, asserted: layout DOES change the bits (nw1 vs nw4), yet every
    # config is numerically close to the true sum ("different bits != wrong").
    o1, o4 = _run(src, N, 1), _run(src, N, 4)
    assert not torch.equal(o1, o4), "expected num_warps 1 vs 4 to differ in bits"
    assert torch.allclose(o4, torch.tensor([[ref]], device="cuda", dtype=torch.float32).reshape(1), atol=1e-3), \
        "every config is still numerically close to the true sum"
    print(f"\n[OK] layout changed the bits ({len(classes)} distinct results across {len(warps)} configs); "
          "all are numerically close. Bits are determinism vs a reference, not correctness.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
