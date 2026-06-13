"""n03 — softmax: exp() between two reductions amplifies the divergence.

A row-wise softmax does a max-reduce, then ``exp``, then a sum-reduce, then a
divide. Both reductions are layout-dependent (n02), and the ``exp`` in between
amplifies tiny ordering differences in the max before they reach the sum. So
across autotuner configs (``num_warps``) the per-row outputs split into more
bit-classes than a bare sum does.

We compile each config, show the two ``tt.reduce`` ops, run each, and group the
*whole output tensor's* exact bits into equivalence classes; assert >1 class.

Run:  python bitequiv/examples/numerical-inconsistency/n03_softmax_row_reduction.py
"""
import torch

import triton
import triton.language as tl

from _helpers import (banner, compile_only, group_by_bits, is_cuda, short, show)

CONFIG_WARPS = [1, 2, 4, 8]


def _softmax_input(rows, cols, seed=0):
    """Moderate-range rows: softmax divergence needs MANY comparable exp() terms.

    NOTE: do NOT reuse the extreme ``adversarial_1d`` recipe here. A huge dynamic
    range makes softmax collapse to one-hot (every term but the max underflows to
    0), so the output becomes bit-identical regardless of reduction order. We want
    the opposite: ~unit-scale values so thousands of exp() terms contribute to the
    sum, making the sum-reduction ORDER decide the low bits.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(rows, cols, generator=g, dtype=torch.float32).to("cuda")


@triton.jit
def softmax_kernel(src, dst, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(src + row * stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(dst + row * stride + offs, num / den, mask=mask)


def run_config(src, num_warps):
    rows, cols = src.shape
    out = torch.empty_like(src)
    softmax_kernel[(rows, )](src, out, cols, src.stride(0), BLOCK=cols, num_warps=num_warps)
    torch.cuda.synchronize()
    return out


def _ref(src):
    return torch.softmax(src.double(), dim=1)


def main():
    rows, cols = 64, 8192
    src = _softmax_input(rows, cols, seed=0)
    ref = _ref(src)

    banner("n03 — softmax has two layout-dependent reductions (max, then sum)")
    ck = compile_only(softmax_kernel, src, torch.empty_like(src), cols, src.stride(0),
                      BLOCK=cols, num_warps=4, grid=(rows, ))
    show(ck, "ttgir", grep='"tt.reduce"', limit=4)

    banner("n03 — exact outputs grouped into bitwise-equivalence classes")
    outputs = []
    for nw in CONFIG_WARPS:
        out = run_config(src, nw)
        outputs.append((f"num_warps={nw}", out))
        max_err = (out.double() - ref).abs().max().item()
        print(f"    num_warps={nw}:  max err vs fp64 softmax={max_err:.3e}")

    classes = group_by_bits(outputs)
    print("\n    bitwise-equivalence classes (by exact bits of the whole output):")
    for key, members in classes.items():
        print(f"        {members}  ->  bitclass {short(key)}")

    assert len(classes) > 1, (
        f"expected >1 bit-class across num_warps {CONFIG_WARPS}; got {len(classes)}. "
        "Increase cols/rows or change the seed if configs coincided.")
    for label, out in outputs:
        assert (out.double() - ref).abs().max().item() < 1e-3, \
            f"{label} drifted too far from the fp64 softmax"

    print(f"\n[OK] softmax produced {len(classes)} distinct bit-classes across "
          f"{len(CONFIG_WARPS)} autotuner configs; all numerically close to the fp64 reference.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
