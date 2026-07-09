"""Example: the bitequiv equivalence checkers, and turning on pruning in the autotuner.

Two parts, run in order by ``main()``:

PART 1 — RUN THE TWO CHECKERS STANDALONE
----------------------------------------
bitequiv ships two static reduction-equivalence checkers. Both answer "do these two
compiled configs reduce in the same floating-point order, hence produce the same bits?"
purely from the compiled IR — no kernel launch, no reference output:

  * the TTGIR checker (`bitequiv.ttgir_reduction.ttgir_reduction_descriptor`) reads the **data
    layout** of the reduce operand from TTGIR. It is cheap (just parses TTGIR text via the
    `libtriton.bitequiv` analysis) but is **blind to FMA contraction**, which is decided
    below TTGIR.
  * the PTX checker (`bitequiv.ptx_reduction.ptx_reduction_descriptor`) reconstructs the
    reduction tree from PTX, so it *also* sees `mul`+`add` -> `fma` fusion (gated by
    `enable_fp_fusion`).

Two configs are equivalent under a checker iff their descriptors are equal.
`run_checkers_standalone()` shows the instructive contrast:
  - pure-add row sum, num_warps 4 vs 8: under `unordered` both checkers say NOT equivalent
    (the left-fold order is layout-dependent); under `inner_tree` both say equivalent (the
    balanced tree is layout-invariant).
  - row dot (`sum(x*y)`), `enable_fp_fusion` on vs off: the TTGIR checker says EQUIVALENT
    (the TTGIR is identical) but the PTX checker says NOT — it sees `fma.rn.f32` vs
    `add`+`mul`. This is exactly why the autotuner pruning below uses the PTX checker.

PART 2 — TURN ON PRUNING IN THE AUTOTUNER (PTX checker)
------------------------------------------------------
The autotuner already has the hook — `prune_configs_by={"ir_config_prune": ...}` (see
`python/triton/runtime/autotuner.py`). bitequiv turns its PTX checker into such a predicate
via `reduction_equivalence_prune(level, reference)` (see `bitequiv/equivalence_ptx.py`).
Wiring those two together — no core or API changes — makes the autotuner keep ONLY the
configs whose compiled reduction is bitwise-equivalent to a chosen reference, then pick the
fastest of those. We sweep a deliberately diverse ~100-config space (reduction_ordering x
num_warps x num_stages x BLOCK_N over a looped row-sum kernel) so the prune has real work
to do: most configs collapse into a few equivalence classes, and only the reference's class
survives to be benchmarked. Dropped configs land in `kernel.pruned_by_ir`; the classes are
on the predicate's `.classes`.

The reference has two modes, both shown:
  1. DEFAULT (`reference=None`)  — the autotuner supplies its FIRST config as the reference.
  2. EXPLICIT (`reference=<compiled kernel | asm dict | raw IR text>`) — you compile a chosen
     config and hand it in as the standard of bit-exactness.

Run it (on a GPU host, with triton importable):
    python -m bitequiv.examples.autotune_equivalence_pruning
"""

import itertools
import os
import sys

# Make `bitequiv` importable whether run as `-m bitequiv.examples...` or as a plain script.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from bitequiv.ptx_reduction import ptx_reduction_descriptor  # noqa: E402
from bitequiv.ttgir_reduction import ttgir_reduction_descriptor  # noqa: E402

from bitequiv.equivalence_ptx import reduction_equivalence_prune  # noqa: E402

DEVICE = "cuda"

_INNER_TREE = tl.ReductionOrdering.INNER_TREE
_UNORDERED = tl.ReductionOrdering.UNORDERED

# One program per row; a single tile covers the whole row for the standalone-checker kernels.
ROWS, N_COLS, BLOCK = 128, 4096, 4096


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
@triton.jit
def rowsum_kernel(src, dst, n_cols, stride, BLOCK: tl.constexpr, REDUCTION_ORDERING: tl.constexpr):
    """Single-tile per-row sum (pure add)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(src + row * stride + offs, mask=mask, other=0.0)
    tl.store(dst + row, tl.sum(x, axis=0, reduction_ordering=REDUCTION_ORDERING))


@triton.jit
def rowdot_kernel(a, b, dst, n_cols, stride, BLOCK: tl.constexpr, REDUCTION_ORDERING: tl.constexpr):
    """Single-tile per-row dot product: a multiply feeds the reduction, so `enable_fp_fusion`
    decides `fma` vs `mul`+`add` — visible in PTX, invisible in TTGIR."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(a + row * stride + offs, mask=mask, other=0.0)
    y = tl.load(b + row * stride + offs, mask=mask, other=0.0)
    tl.store(dst + row, tl.sum(x * y, axis=0, reduction_ordering=REDUCTION_ORDERING))


@triton.jit
def rowsum_looped_kernel(src, dst, n_cols, stride, BLOCK_N: tl.constexpr, REDUCTION_ORDERING: tl.constexpr):
    """Per-row sum over the row in BLOCK_N-wide tiles. The loop makes BLOCK_N and num_stages
    real autotuner knobs (tile count + pipelining), so the config space is genuinely diverse."""
    row = tl.program_id(0)
    acc = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < n_cols
        acc += tl.load(src + row * stride + offs, mask=mask, other=0.0)
    tl.store(dst + row, tl.sum(acc, axis=0, reduction_ordering=REDUCTION_ORDERING))


def _sum_inputs():
    g = torch.Generator(device="cpu").manual_seed(0)
    src = torch.randn(ROWS, N_COLS, generator=g, dtype=torch.float32).to(DEVICE)
    dst = torch.empty(ROWS, device=DEVICE, dtype=torch.float32)
    return src, dst


def _dot_inputs():
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(ROWS, N_COLS, generator=g, dtype=torch.float32).to(DEVICE)
    b = torch.randn(ROWS, N_COLS, generator=g, dtype=torch.float32).to(DEVICE)
    dst = torch.empty(ROWS, device=DEVICE, dtype=torch.float32)
    return a, b, dst


# --------------------------------------------------------------------------- #
# PART 1 — run the two checkers standalone
# --------------------------------------------------------------------------- #
def _compile_rowsum(ordering, num_warps):
    src, dst = _sum_inputs()
    return rowsum_kernel.warmup(src, dst, N_COLS, src.stride(0), grid=(ROWS, ), BLOCK=BLOCK,
                                REDUCTION_ORDERING=ordering, num_warps=num_warps)


def _compile_rowdot(ordering, num_warps, enable_fp_fusion):
    a, b, dst = _dot_inputs()
    return rowdot_kernel.warmup(a, b, dst, N_COLS, a.stride(0), grid=(ROWS, ), BLOCK=BLOCK, REDUCTION_ORDERING=ordering,
                                num_warps=num_warps, enable_fp_fusion=enable_fp_fusion)


def _verdicts(ck_a, ck_b):
    """(ttgir_equivalent, ptx_equivalent) for two compiled kernels, from each checker."""
    ttgir_eq = ttgir_reduction_descriptor(ck_a.asm["ttgir"]) == ttgir_reduction_descriptor(ck_b.asm["ttgir"])
    ptx_eq = ptx_reduction_descriptor(ck_a.asm["ptx"]) == ptx_reduction_descriptor(ck_b.asm["ptx"])
    return ttgir_eq, ptx_eq


def _yn(b):
    return "equivalent" if b else "NOT equivalent"


def run_checkers_standalone():
    """PART 1: run the TTGIR and PTX checkers directly on compiled configs and print their
    verdicts. Returns a dict of (ttgir_equivalent, ptx_equivalent) per comparison."""
    print("\n=== PART 1: run the two checkers standalone (no autotuner) ===")
    results = {}

    # 1) pure-add row sum, num_warps 4 vs 8, under each reduction ordering.
    for name, ordering in (("unordered", _UNORDERED), ("inner_tree", _INNER_TREE)):
        ttgir_eq, ptx_eq = _verdicts(_compile_rowsum(ordering, 4), _compile_rowsum(ordering, 8))
        results[f"rowsum_{name}_nw4_vs_nw8"] = (ttgir_eq, ptx_eq)
        print(f"  rowsum  {name:10s} num_warps 4 vs 8:  TTGIR {_yn(ttgir_eq):14s} | PTX {_yn(ptx_eq)}")

    # 2) row dot, enable_fp_fusion on vs off (same ordering, same num_warps): the FMA gap.
    ck_on = _compile_rowdot(_UNORDERED, 4, enable_fp_fusion=True)
    ck_off = _compile_rowdot(_UNORDERED, 4, enable_fp_fusion=False)
    ttgir_eq, ptx_eq = _verdicts(ck_on, ck_off)
    results["rowdot_fp_fusion_on_vs_off"] = (ttgir_eq, ptx_eq)
    print(f"  rowdot  fp_fusion on vs off:        TTGIR {_yn(ttgir_eq):14s} | PTX {_yn(ptx_eq)}")
    print("  ^ the TTGIR checker is blind to FMA contraction (identical TTGIR), the PTX checker is not.")
    return results


# --------------------------------------------------------------------------- #
# PART 2 — turn on PTX pruning in the autotuner over a diverse config space
# --------------------------------------------------------------------------- #
# enable_fp_fusion is a compile flag, not a triton.Config field, so it is not an autotuner
# knob here (it is exercised in PART 1). The grid below spans the four knobs the autotuner
# can vary, ~100 configs total.
_ORDERINGS = (("inner_tree", _INNER_TREE), ("unordered", _UNORDERED))
_NUM_WARPS = (1, 2, 4, 8)
_NUM_STAGES = (2, 3)
_BLOCK_N = (128, 256, 512, 1024, 2048, 4096)


def _make_config(ordering_val, num_warps, num_stages, block_n):
    return triton.Config({"BLOCK_N": block_n, "REDUCTION_ORDERING": ordering_val}, num_warps=num_warps,
                         num_stages=num_stages)


def _configs():
    """~96 diverse configs across reduction_ordering x num_warps x num_stages x BLOCK_N. The
    FIRST config is the autotuner's default reference."""
    return [
        _make_config(ordering_val, num_warps, num_stages, block_n)
        for (_, ordering_val
             ), num_warps, num_stages, block_n in itertools.product(_ORDERINGS, _NUM_WARPS, _NUM_STAGES, _BLOCK_N)
    ]


def _configs_quick():
    """A small subset (8 configs) for the regression test — same code path, fast."""
    return [
        _make_config(ordering_val, num_warps, 3, block_n)
        for (_, ordering_val), num_warps, block_n in itertools.product(_ORDERINGS, (2, 4), (2048, 4096))
    ]


def _ordering_name(config):
    return "inner_tree" if config.kwargs["REDUCTION_ORDERING"] == _INNER_TREE else "unordered"


def _label(config):
    return (f"reduction_ordering={_ordering_name(config)} num_warps={config.num_warps} "
            f"num_stages={config.num_stages} BLOCK_N={config.kwargs['BLOCK_N']}")


def _summarize(mode, configs, prune, kernel):
    """Print the equivalence classes + kept/pruned counts, and return a small result dict."""
    pruned = set(kernel.pruned_by_ir)  # configs the autotuner dropped via ir_config_prune
    kept = [c for c in configs if c not in pruned]
    print(f"\n--- reference mode: {mode} ---")
    print(f"  config space: {len(configs)} configs; checker found {len(prune.classes)} equivalence class(es)")
    print(f"  KEPT (bitwise-equivalent to the reference): {len(kept)}")
    print(f"  PRUNED (not equivalent): {len(pruned)}")
    for c in kept[:6]:
        print(f"    kept: {_label(c)}")
    if len(kept) > 6:
        print(f"    ... (+{len(kept) - 6} more kept)")
    print(f"  best config selected by the autotuner: {_label(kernel.best_config)}")
    return {
        "mode": mode,
        "n_configs": len(configs),
        "n_classes": len(prune.classes),
        "kept_labels": sorted(_label(c) for c in kept),
        "pruned_labels": sorted(_label(c) for c in pruned),
        "best_label": _label(kernel.best_config),
    }


def run_default_reference(configs=None):
    """Mode 1: equivalence measured against the autotuner's FIRST config (reference=None)."""
    src, dst = _sum_inputs()
    configs = configs if configs is not None else _configs()
    prune = reduction_equivalence_prune("ptx")  # reference=None -> first config is the reference
    kernel = triton.autotune(configs=configs, key=["n_cols"], prune_configs_by={"ir_config_prune":
                                                                                prune})(rowsum_looped_kernel)
    kernel[(ROWS, )](src, dst, N_COLS, src.stride(0))
    return _summarize("default (autotuner's first config)", configs, prune, kernel)


def run_explicit_reference(configs=None):
    """Mode 2: equivalence measured against an EXPLICIT anchor we compile ourselves.

    We pick `unordered, num_warps=4, BLOCK_N=4096` as the standard of bit-exactness. Because
    an unordered reduction is layout-dependent, only configs with that exact reduction match —
    a different (more restrictive) outcome than mode 1, showing the reference drives the result.
    """
    src, dst = _sum_inputs()
    configs = configs if configs is not None else _configs()
    anchor = rowsum_looped_kernel.warmup(src, dst, N_COLS, src.stride(0), grid=(ROWS, ), BLOCK_N=4096,
                                         REDUCTION_ORDERING=_UNORDERED, num_warps=4, num_stages=3)
    prune = reduction_equivalence_prune("ptx", reference=anchor)
    kernel = triton.autotune(configs=configs, key=["n_cols"], prune_configs_by={"ir_config_prune":
                                                                                prune})(rowsum_looped_kernel)
    kernel[(ROWS, )](src, dst, N_COLS, src.stride(0))
    return _summarize("explicit anchor (unordered, num_warps=4, BLOCK_N=4096)", configs, prune, kernel)


def main():
    if not torch.cuda.is_available():
        print("no CUDA GPU available; this example needs one to compile and benchmark kernels.")
        return
    print(f"device: {torch.cuda.get_device_name()}")
    run_checkers_standalone()
    print("\n=== PART 2: turn on PTX equivalence pruning in the autotuner ===")
    run_default_reference()
    run_explicit_reference()


if __name__ == "__main__":
    main()
