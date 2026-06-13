"""b02 — Enforcing bitwise equivalence: the inner_tree reduction ordering.

The flip side of b01. b01 showed that the *unordered* reduction's tree (and bits)
follow the layout, so ``num_warps`` can change the result. ``inner_tree`` instead
pins a single **canonical, layout-invariant** combination order over the original
element indices, so EVERY layout reduces in the same order and produces the same
bits.

Mechanism: ``ReduceOpToLLVM.cpp`` (``isInnerTree``) emits a balanced within-thread
tree + count-up butterfly shuffles defined over element indices, not over the
physical lane layout. Enabled by ``TRITON_STRICT_REDUCTION_ORDERING=1`` (read at
import time -> this script sets it BEFORE importing triton), which makes
``tl.sum`` default to ``reduction_ordering = "inner_tree"``.

BIT-CHANGING relative to b01's unordered result, but **layout-INVARIANT**: this is
how the project recovers tuning freedom (vary ``num_warps`` for speed) while
keeping bitwise-identical output.

Run:  python bitequiv/examples/layout_numerics/b02_reduction_ordering_inner_tree.py
"""
import os

# Must be set before importing triton (the knob is read at import time).
os.environ.setdefault("TRITON_STRICT_REDUCTION_ORDERING", "1")

import torch  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from _ir_utils import banner, compile_only, hexbits, is_cuda, show  # noqa: E402


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

    banner("b02 — tl.sum now lowers to reduction_ordering = inner_tree")
    ck = compile_only(sum_kernel, src, torch.empty(1, device="cuda"), N, BLOCK=N, num_warps=4, grid=(1, ))
    show(ck, "ttgir", grep='"tt.reduce"', limit=1)
    assert 'reduction_ordering = "inner_tree"' in ck.asm["ttgir"], \
        "TRITON_STRICT_REDUCTION_ORDERING must make tl.sum use inner_tree"

    banner("b02 — every num_warps now produces the SAME bits (layout-invariant)")
    classes = {}
    for nw in warps:
        out = _run(src, N, nw)
        classes.setdefault(hexbits(out), []).append(nw)
        print(f"    num_warps={nw}:  {out.item():.10f}  (raw i32 bits={hexbits(out)})")
    print("\n    bitwise-equivalence classes (by exact bits):")
    for bits, members in classes.items():
        print(f"        num_warps {members}  ->  bits={bits}")

    base = _run(src, N, 1)
    for nw in warps[1:]:
        assert torch.equal(base, _run(src, N,
                                      nw)), f"inner_tree must be bitwise-equal across num_warps (failed nw={nw})"
    assert len(classes) == 1, "inner_tree must collapse all layouts into ONE class"
    print("\n[OK] inner_tree gives ONE bitwise-equivalence class across all num_warps "
          "(contrast b01's layout-dependent split). This is how tuning freedom is kept "
          "under a bitwise constraint.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
