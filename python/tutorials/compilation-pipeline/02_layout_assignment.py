"""02 — Layout assignment: TTIR -> TTGIR (the defining TritonGPU step).

Pipeline stage: **TTGIR**, entry pass ``convert-triton-to-tritongpu``
(``third_party/nvidia/backend/compiler.py`` -> ``make_ttgir``).

What to notice: every tensor type gains a **layout encoding** describing the
thread/warp/CTA -> element mapping, e.g.
``tensor<256xf32, #ttg.blocked<{sizePerThread=[..], threadsPerWarp=[32],
warpsPerCTA=[N], order=[0]}>>``. ``num_warps`` flows straight into
``warpsPerCTA``: compile the SAME kernel with ``num_warps=4`` vs ``8`` and the
encoding changes.

Bit-neutral here, but FOUNDATIONAL: this layout is exactly what later fixes a
reduction's tree shape, which is what can make a reduction's FP result depend on
accumulation order. For a plain elementwise add the layout does not change the
bits.

Run:  python python/tutorials/compilation-pipeline/02_layout_assignment.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, diff, is_cuda, show


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


def main():
    n, BLOCK = 4096, 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    o = torch.empty_like(x)

    ck4 = compile_only(add_kernel, x, y, o, n, BLOCK=BLOCK, num_warps=4, grid=(1, ))
    ck8 = compile_only(add_kernel, x, y, o, n, BLOCK=BLOCK, num_warps=8, grid=(1, ))

    banner("02 — same op, before vs after layout assignment")
    print("TTIR (no layout) — note the plain tensor type on arith.addf:")
    show(ck4, "ttir", grep="arith.addf", limit=3)
    print("\nTTGIR (layout attached, num_warps=4):")
    show(ck4, "ttgir", grep="#blocked =", limit=3)
    print("TTGIR (layout attached, num_warps=8):")
    show(ck8, "ttgir", grep="#blocked =", limit=3)

    banner("02 — num_warps flows into warpsPerCTA (the thread mapping)")
    diff(ck4, ck8, "ttgir", grep="#blocked =", label_a="num_warps=4", label_b="num_warps=8")

    # Bit-neutral for elementwise add: both layouts give the exact same result.
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]), )
    o4 = torch.empty_like(x)
    o8 = torch.empty_like(x)
    add_kernel[grid](x, y, o4, n, BLOCK=BLOCK, num_warps=4)
    add_kernel[grid](x, y, o8, n, BLOCK=BLOCK, num_warps=8)
    torch.cuda.synchronize()
    assert torch.equal(o4, o8) and torch.equal(o4, x + y), "elementwise add is layout-invariant"
    print("\n[OK] num_warps=4 and num_warps=8 give bitwise-identical results"
          " (elementwise add is layout-invariant).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
