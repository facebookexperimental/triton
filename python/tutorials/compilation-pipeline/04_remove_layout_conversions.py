"""04 — RemoveLayoutConversions: the convert_layout-elimination workhorse.

Pipeline: **TTGIR** ``RemoveLayoutConversions``
(``lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp``), which runs
5+ times during ``make_ttgir``. A ``ttg.convert_layout`` shuffles data between
threads (often through shared memory) — pure overhead — so the pass propagates
layouts to delete every conversion that isn't genuinely required.

What to notice: an **elementwise** kernel needs ZERO conversions (one layout
serves the whole kernel). A **transpose** needs exactly ONE: transposing changes
which thread owns which element, so the data must physically move — that lone
``convert_layout`` is irreducible and survives the pass.

Bit-neutral mechanic: moving an element between threads does not change its value.
(But *which* layout survives is what later fixes a reduction's tree, so for a
reduction kernel it can change the FP accumulation order.)

Watch redundant conversions disappear pass-by-pass:
    MLIR_ENABLE_DUMP=transpose_kernel TRITON_ALWAYS_COMPILE=1 \\
        python python/tutorials/compilation-pipeline/04_remove_layout_conversions.py

Run:  python python/tutorials/compilation-pipeline/04_remove_layout_conversions.py
"""
import torch

import triton
import triton.language as tl

from _ir_utils import banner, compile_only, count, is_cuda, show


@triton.jit
def elementwise_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2.0, mask=offs < n)


@triton.jit
def transpose_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rn[:, None] * M + rm[None, :], tl.trans(x))


def main():
    n, BLOCK = 1024, 1024
    x1 = torch.randn(n, device="cuda")
    o1 = torch.empty_like(x1)
    M = N = 64
    x2 = torch.randn(M, N, device="cuda")
    o2 = torch.empty(N, M, device="cuda")

    ck_elem = compile_only(elementwise_kernel, x1, o1, n, BLOCK=BLOCK, grid=(1, ))
    ck_trans = compile_only(transpose_kernel, x2, o2, M, N, grid=(1, ))

    banner("04 — convert_layout count = irreducible cross-thread data movement")
    print(f"    elementwise (x*2):  ttg.convert_layout ops = {count(ck_elem, 'ttgir', 'convert_layout')}")
    print(f"    transpose  (x.T) :  ttg.convert_layout ops = {count(ck_trans, 'ttgir', 'convert_layout')}")
    print("\n    The pass deleted every *redundant* conversion. The transpose's one")
    print("    survivor is necessary — transposing genuinely re-assigns elements to threads.")

    banner("04 — the surviving conversion and the two layouts it bridges")
    show(ck_trans, "ttgir", grep="#blocked", limit=2,
         label="layout encodings (note the transposed order [1,0] vs [0,1]):")
    show(ck_trans, "ttgir", grep="convert_layout", limit=4, label="the convert_layout op:")

    # Bit-neutral: the transpose moves elements but preserves their values exactly.
    transpose_kernel[(1, )](x2, o2, M, N)
    torch.cuda.synchronize()
    assert torch.equal(o2, x2.T.contiguous()), "transpose must be exact"
    print("\n[OK] transpose result is bitwise-equal to x.T (layout conversion is bit-neutral).")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
