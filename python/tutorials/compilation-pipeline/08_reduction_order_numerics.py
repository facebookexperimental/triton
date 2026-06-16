"""08 — Reduction order is layout-dependent: a POSITIVE numerics example.

This is the core mechanism of the bitwise-equivalence project. A ``tl.sum`` over
one axis lowers to a per-thread fold followed by a cross-warp shuffle tree (see
07). WHICH element lives in which lane/warp is the **layout** along the reduce
axis, and ``num_warps`` changes it. A different layout => a different tree => a
different summation order => (FP add is non-associative) **different bits**.

Unlike 01-06 (all bit-neutral), this one is BIT-CHANGING: we compile the same
kernel at ``num_warps in {1,2,4,8}``, show the shuffle tree differs, then run each
and group the *exact* results into bitwise-equivalence classes.

Subtlety worth seeing: not every layout difference moves the bits — here 2/4/8
coincide and only 1 stands apart. You cannot eyeball which configs are equivalent;
that is exactly why the project builds an equivalence checker (see
``bitequiv/examples/layout_numerics/`` for the static-signature + runtime checker).

Run:  python python/tutorials/compilation-pipeline/08_reduction_order_numerics.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, is_cuda


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

    banner("08 — same reduction, the layout (and its shuffle tree) changes with num_warps")
    for nw in warps:
        ck = compile_only(sum_kernel, src, torch.empty(1, device="cuda"), N, BLOCK=N, num_warps=nw, grid=(1, ))
        layout = next((ln for ln in ck.asm["ttgir"].splitlines() if "#blocked =" in ln), "")
        print(f"    num_warps={nw}:  shfl.sync={count(ck, 'llir', 'shfl'):>2}   {layout.split('= ', 1)[-1]}")

    banner("08 — run each config and group EXACT results into equivalence classes")
    results = {nw: _run(src, N, nw).item() for nw in warps}
    classes = {}
    for nw in warps:
        classes.setdefault(results[nw].hex(), []).append(nw)
    for bits, members in classes.items():
        print(f"    {bits}  <- num_warps={members}")

    ref = src.sum()
    for nw in warps:
        torch.testing.assert_close(torch.tensor(results[nw]), ref.cpu(), atol=1e-3, rtol=1e-3)
    assert len(classes) > 1, "expected at least two bitwise-equivalence classes (1 vs the rest)"
    print(f"\n[OK] all {len(warps)} configs match torch.sum within tolerance, but split into"
          f" {len(classes)} bitwise classes — layout changes the reduction order, hence the bits.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
