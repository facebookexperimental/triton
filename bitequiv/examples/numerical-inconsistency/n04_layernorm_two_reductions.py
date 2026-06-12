"""n04 — layer-norm: two DEPENDENT reductions, the strongest amplifier.

Layer-norm forward computes a mean (reduction #1), then a variance that depends
on that mean (reduction #2), then normalizes. Any ordering difference in the mean
feeds into the variance, so layout-driven divergence compounds across the two
reductions. This is the production-relevant case: the project's ``STABLE_REDUCTION``
layer-norm workaround exists precisely because this divergence caused a real
~5.24% NE gap (see ``bitequiv/CLAUDE.md``).

We enumerate autotuner configs (``num_warps``), run each, and group the exact
outputs into bitwise-equivalence classes; assert >1 class.

Run:  python bitequiv/examples/numerical-inconsistency/n04_layernorm_two_reductions.py
"""
import torch

import triton
import triton.language as tl

from _helpers import (adversarial_2d, banner, compile_only, group_by_bits,
                      is_cuda, short, show)

CONFIG_WARPS = [1, 2, 4, 8]


@triton.jit
def layernorm_kernel(src, dst, n_cols, stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(src + row * stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(dst + row * stride + offs, xc * rstd, mask=mask)


def run_config(src, eps, num_warps):
    rows, cols = src.shape
    out = torch.empty_like(src)
    layernorm_kernel[(rows, )](src, out, cols, src.stride(0), eps, BLOCK=cols, num_warps=num_warps)
    torch.cuda.synchronize()
    return out


def _ref(src, eps):
    x = src.double()
    mean = x.mean(dim=1, keepdim=True)
    xc = x - mean
    var = (xc * xc).mean(dim=1, keepdim=True)
    return xc / torch.sqrt(var + eps)


def main():
    rows, cols, eps = 64, 8192, 1e-5
    src = adversarial_2d(rows, cols, seed=0)
    ref = _ref(src, eps)

    banner("n04 — layer-norm: mean then variance (variance depends on mean)")
    ck = compile_only(layernorm_kernel, src, torch.empty_like(src), cols, src.stride(0), eps,
                      BLOCK=cols, num_warps=4, grid=(rows, ))
    show(ck, "ttgir", grep='"tt.reduce"', limit=4)

    banner("n04 — exact outputs grouped into bitwise-equivalence classes")
    outputs = []
    for nw in CONFIG_WARPS:
        out = run_config(src, eps, nw)
        outputs.append((f"num_warps={nw}", out))
        max_err = (out.double() - ref).abs().max().item()
        print(f"    num_warps={nw}:  max err vs fp64 layer-norm={max_err:.3e}")

    classes = group_by_bits(outputs)
    print("\n    bitwise-equivalence classes (by exact bits of the whole output):")
    for key, members in classes.items():
        print(f"        {members}  ->  bitclass {short(key)}")

    assert len(classes) > 1, (
        f"expected >1 bit-class across num_warps {CONFIG_WARPS}; got {len(classes)}. "
        "Increase cols/rows or change the seed if configs coincided.")
    for label, out in outputs:
        assert (out.double() - ref).abs().max().item() < 1e-2, \
            f"{label} drifted too far from the fp64 layer-norm"

    print(f"\n[OK] layer-norm produced {len(classes)} distinct bit-classes across "
          f"{len(CONFIG_WARPS)} autotuner configs; all numerically close. This is the "
          "pattern behind the production STABLE_REDUCTION workaround.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
