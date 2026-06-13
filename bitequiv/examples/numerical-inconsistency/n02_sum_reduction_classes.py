"""n02 — 1-D sum: different autotuner configs -> different bits (the clean repro).

THE simplest, most reliable demonstration. A ``tl.sum`` over one axis lowers to:
each thread folds its own elements, then warps combine across a shuffle tree
(``lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp``). WHICH element lives in
which lane/warp is the **layout** (``sizePerThread`` / ``threadsPerWarp`` /
``warpsPerCTA`` along the reduce axis), and an autotuner config's ``num_warps``
changes it. A different layout => a different tree => a different summation order
=> (FP add is non-associative) **different bits**.

This script enumerates the configs an autotuner would try (``num_warps`` in
{1,2,4,8}), runs each, and groups the *exact* outputs into bitwise-equivalence
classes. We assert MORE THAN ONE class exists -- i.e. the choice of config alone
changes the numerics. Every config stays numerically close to the fp64 reference:
the problem is **determinism**, not correctness.

Deep dive: ``bitequiv/knowledge-base/tree-reduction-in-ptx-and-triton.md``.
The existing fix (collapse all configs to one class) is
``reduction_ordering="inner_tree"`` / ``TRITON_STRICT_REDUCTION_ORDERING=1``.

Run:  python bitequiv/examples/numerical-inconsistency/n02_sum_reduction_classes.py
"""
import torch

import triton
import triton.language as tl

from _helpers import (adversarial_1d, banner, compile_only, first, group_by_bits,
                      is_cuda, short, show)

# The configs a stock autotuner would enumerate for this kernel. Only num_warps
# varies, so the story stays about the autotuner picking a *layout*.
CONFIG_WARPS = [1, 2, 4, 8]


@triton.jit
def sum_kernel(src, dst, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))


def run_config(src, N, num_warps):
    out = torch.empty(1, device="cuda", dtype=torch.float32)
    sum_kernel[(1, )](src, out, N, BLOCK=N, num_warps=num_warps)
    torch.cuda.synchronize()
    return out


def main():
    N = 8192
    src = adversarial_1d(N, seed=0)
    ref = src.double().sum().item()

    banner("n02 — same tl.sum, layout along the reduce axis changes with num_warps")
    for nw in CONFIG_WARPS:
        ck = compile_only(sum_kernel, src, torch.empty(1, device="cuda"), N,
                          BLOCK=N, num_warps=nw, grid=(1, ))
        layout = first(ck, "ttgir", "#blocked =")
        print(f"    num_warps={nw}:  {layout.split('= ', 1)[-1]}")
    print("\n    the tt.reduce op (note reduction_ordering = unordered by default):")
    ck = compile_only(sum_kernel, src, torch.empty(1, device="cuda"), N,
                      BLOCK=N, num_warps=4, grid=(1, ))
    show(ck, "ttgir", grep='"tt.reduce"', limit=1)

    banner("n02 — exact results grouped into bitwise-equivalence classes")
    outputs = []
    for nw in CONFIG_WARPS:
        out = run_config(src, N, nw)
        outputs.append((f"num_warps={nw}", out))
        print(f"    num_warps={nw}:  {out.item():.10f}  (i32 bits={out.view(torch.int32).item()})  "
              f"err vs fp64={abs(out.double().item() - ref):.3e}")

    classes = group_by_bits(outputs)
    print("\n    bitwise-equivalence classes (by exact bits):")
    for key, members in classes.items():
        print(f"        {members}  ->  bitclass {short(key)}")

    assert len(classes) > 1, (
        f"expected >1 bit-class across num_warps {CONFIG_WARPS}; got {len(classes)}. "
        "If configs coincided, increase N or change the seed (see _helpers.adversarial_1d).")
    # "different bits != wrong": every config is still close to the true sum.
    for label, out in outputs:
        assert abs(out.double().item() - ref) < max(1e-2, abs(ref) * 1e-3), \
            f"{label} drifted too far from the fp64 reference"

    print(f"\n[OK] the autotuner's num_warps choice alone produced {len(classes)} distinct "
          f"bit-classes across {len(CONFIG_WARPS)} configs; all are numerically close to "
          "the fp64 sum. Bits are determinism vs a reference, not correctness.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
