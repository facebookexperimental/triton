"""n01 — the headline: the stock autotuner silently picks the numerics.

Same kernel + same input, wrapped in a real ``@triton.autotune`` over configs
that differ only in ``num_warps``. The autotuner times them and keeps the
*fastest* -- but those configs do NOT all produce the same bits (see n02 for the
mechanism). So whichever config happens to win the timing race silently
determines the floating-point result.

We:
  1. launch the autotuned kernel once and read which config won
     (``sum_kernel.best_config``) and its output bits;
  2. enumerate every config individually and group outputs into
     bitwise-equivalence classes;
  3. assert there is MORE THAN ONE class -- so the tuning decision alone changes
     the numerics.

We deliberately do NOT assert *which* config wins: that depends on the GPU and
on timing noise, and that nondeterminism is exactly the point. Two identical
builds on two machines can pick different configs and produce different bits.

Uses only stock ``@triton.autotune`` / ``triton.Config`` -- no project-local
autotuner changes (those are the "after"). Mechanism + fix:
``bitequiv/knowledge-base/tree-reduction-in-ptx-and-triton.md``.

Run:  python bitequiv/examples/numerical-inconsistency/n01_autotuner_picks_silently.py
"""
import torch

import triton
import triton.language as tl

from _helpers import (adversarial_1d, banner, group_by_bits, is_cuda, short)

CONFIG_WARPS = [1, 2, 4, 8]


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in CONFIG_WARPS],
    key=["N"],
)
@triton.jit
def autotuned_sum(src, dst, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))


@triton.jit
def _sum_fixed(src, dst, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))


def run_config(src, N, num_warps):
    out = torch.empty(1, device="cuda", dtype=torch.float32)
    _sum_fixed[(1, )](src, out, N, BLOCK=N, num_warps=num_warps)
    torch.cuda.synchronize()
    return out


def main():
    N = 8192
    src = adversarial_1d(N, seed=0)
    ref = src.double().sum().item()

    banner("n01 — let the stock autotuner choose, then look at what it chose")
    out = torch.empty(1, device="cuda", dtype=torch.float32)
    autotuned_sum[(1, )](src, out, N, BLOCK=N)
    torch.cuda.synchronize()
    winner = autotuned_sum.best_config
    print(f"    autotuner picked: num_warps={winner.num_warps}")
    print(f"    its result:       {out.item():.10f}  (i32 bits={out.view(torch.int32).item()})")

    banner("n01 — but every valid config is a different bit-class")
    outputs = []
    for nw in CONFIG_WARPS:
        o = run_config(src, N, nw)
        outputs.append((f"num_warps={nw}", o))
        print(f"    num_warps={nw}:  {o.item():.10f}  (i32 bits={o.view(torch.int32).item()})  "
              f"err vs fp64={abs(o.double().item() - ref):.3e}")

    classes = group_by_bits(outputs)
    print("\n    bitwise-equivalence classes (by exact bits):")
    for key, members in classes.items():
        print(f"        {members}  ->  bitclass {short(key)}")

    assert winner is not None, "autotuner did not record a best_config"
    assert len(classes) > 1, (
        f"expected the config choice to change the bits; got {len(classes)} class(es). "
        "Increase N or change the seed if configs coincided.")
    print(f"\n[OK] the autotuner silently selected num_warps={winner.num_warps} out of "
          f"{len(CONFIG_WARPS)} configs spanning {len(classes)} bit-classes — so the tuning "
          "decision alone determined the numerics. A different machine/timing could pick a "
          "different bit-class.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
