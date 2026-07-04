# bitequiv examples

Runnable examples of using the bitequiv equivalence checkers.

## `autotune_equivalence_pruning.py` — run the checkers, then prune the autotuner

A two-part walkthrough.

### Part 1 — run the two checkers standalone

bitequiv has two static reduction-equivalence checkers; both decide, from the compiled IR
alone (no launch, no reference output), whether two configs reduce in the same float order:

- **TTGIR checker** — `bitequiv.ttgir_reduction.ttgir_reduction_descriptor(ttgir)`. Reads the data
  layout of the reduce operand from TTGIR. Cheap, but **blind to FMA contraction**.
- **PTX checker** — `bitequiv.ptx_reduction.ptx_reduction_descriptor(ptx)`. Reconstructs the
  reduction tree from PTX, so it also sees `mul`+`add` → `fma` fusion.

Two configs are equivalent under a checker iff their descriptors are equal:

```python
from bitequiv.ttgir_reduction import ttgir_reduction_descriptor   # TTGIR checker
from bitequiv.ptx_reduction import ptx_reduction_descriptor       # PTX checker

ck_a = kernel.warmup(...); ck_b = kernel.warmup(...)             # compile two configs (no launch)
ttgir_equiv = ttgir_reduction_descriptor(ck_a.asm["ttgir"]) == ttgir_reduction_descriptor(ck_b.asm["ttgir"])
ptx_equiv   = ptx_reduction_descriptor(ck_a.asm["ptx"]) == ptx_reduction_descriptor(ck_b.asm["ptx"])
```

The example shows the instructive contrast: on a pure-add row sum the two checkers agree
(`unordered` num_warps split, `inner_tree` num_warps merge — it is layout-invariant); but on a
row dot (`sum(x*y)`) with `enable_fp_fusion` on vs off, the **TTGIR checker says equivalent
(identical TTGIR) while the PTX checker says NOT** — it sees `fma.rn.f32` vs `add`+`mul`. That
FMA gap is why the autotuner pruning below uses the PTX checker.

### Part 2 — turn on equivalence pruning in the autotuner (PTX checker)

Make `@triton.autotune` keep only the configs whose compiled reduction is **bitwise-equivalent**
to a chosen reference, then pick the fastest of those — with **no core or API changes**, just
wiring two existing pieces together:

```python
from bitequiv.equivalence_ptx import reduction_equivalence_prune

prune = reduction_equivalence_prune("ptx")          # the equivalence predicate (PTX checker)
kernel = triton.autotune(
    configs=...,
    key=["n_cols"],
    prune_configs_by={"ir_config_prune": prune},    # <-- the hook
)(rowsum_looped_kernel)
```

The autotuner compiles each config, then `ir_config_prune` inspects its PTX (reusing the
already-compiled artifact, no recompile) and drops any config whose reduction order differs from
the reference. Dropped configs are recorded in `kernel.pruned_by_ir`; the equivalence classes are
on `prune.classes`.

To make the prune do real work, the example sweeps a **diverse ~100-config space** —
`reduction_ordering × num_warps × num_stages × BLOCK_N` over a *looped* row-sum kernel (so
`num_stages` and `BLOCK_N` are genuine knobs). Most of those configs collapse into a few
equivalence classes; only the reference's class survives to be benchmarked. (`enable_fp_fusion`
is a compile flag, not a `triton.Config` knob, so it is not an autotuner axis here — it is the
thing Part 1 demonstrates.)

### The reference — two modes

Equivalence is measured against a *reference*, set via the `reference=` parameter:

| Mode | How |
|------|-----|
| **Default** | `reduction_equivalence_prune("ptx")` (`reference=None`) — the autotuner supplies its **first config** |
| **Explicit choice** | `reduction_equivalence_prune("ptx", reference=anchor)` where `anchor` is a compiled kernel / `asm` dict / raw IR text you provide |

A balanced `inner_tree` reduction is layout-invariant, so configs differing only in `num_warps`
merge; an `unordered` left-fold is layout-dependent, so they stay separate; a different `BLOCK_N`
changes the loop fold, so it splits too. The checker reads all of this from PTX.

### Run

```bash
# on a GPU host, with triton importable (PYTHONPATH=$PWD/python or the venv)
python -m bitequiv.examples.autotune_equivalence_pruning
```

It prints the Part 1 checker verdicts, then for each reference mode the config space size, the
number of equivalence classes, the kept/pruned counts, and the best config the autotuner picked.

The GPU-gated regression test is `bitequiv/tests/test_autotune_equivalence_example.py`.
