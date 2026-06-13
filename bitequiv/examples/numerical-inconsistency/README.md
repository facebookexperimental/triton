# Numerical inconsistency from autotuner config selection

> **Upstream base:** `ce3c8447e02d428f18fb5ae9be64a5fa8e9b2109`
> (facebookexperimental/triton #1636). See [`UPSTREAM_COMMIT.txt`](UPSTREAM_COMMIT.txt).
> These examples use **only stock** `@triton.autotune` / `triton.Config` /
> `kernel.warmup()` — no project-local autotuner changes. They are a **standalone**
> repro: nothing here imports from the rest of `bitequiv/`.

## The story

The same kernel, on the same input, compiled under different autotuner configs
(`num_warps` / `BLOCK_SIZE` / `num_stages`), produces **bitwise-different**
floating-point outputs. The stock autotuner keeps the *fastest* config — so
**whichever config happens to win the timing race silently determines the
numerics.** Two identical builds on two machines can pick different configs and
return different bits.

This is the **"before"** state that the rest of the project (constraint-aware
autotuning) fixes by pruning the config space to a single bitwise-equivalence
class.

## Why (mechanism)

`tl.sum` / `tl.reduce` over an axis lowers to a reduction **tree**: each thread
folds its own elements, then warps combine across a shuffle tree
(`lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`). Which element lives in which
lane/warp is the **layout** (`sizePerThread` / `threadsPerWarp` / `warpsPerCTA`
along the reduce axis), and a config's `num_warps` changes it. A different layout
⇒ a different tree ⇒ a different summation order ⇒ (FP add is non-associative)
**different bits**.

The existing fix collapses all configs to one bit-class:
`reduction_ordering="inner_tree"` (env `TRITON_STRICT_REDUCTION_ORDERING=1`), which
enforces a canonical, layout-invariant order. Full write-up:
[`../../knowledge-base/tree-reduction-in-ptx-and-triton.md`](../../knowledge-base/tree-reduction-in-ptx-and-triton.md).

## Files

| File | Kernel | What it shows |
|------|--------|---------------|
| `n01_autotuner_picks_silently.py` | 1-D sum, **autotuned** | The headline: read `best_config`, then show the configs span >1 bit-class — the tuning decision alone sets the numerics. |
| `n02_sum_reduction_classes.py` | 1-D sum | The clean repro: enumerate `num_warps`, group exact outputs into bit-classes. Simplest, most reliable. |
| `n03_softmax_row_reduction.py` | softmax row | `exp` between max- and sum-reduce amplifies divergence. |
| `n04_layernorm_two_reductions.py` | layer-norm fwd | Two **dependent** reductions (mean→variance) — strongest amplifier; the pattern behind the production `STABLE_REDUCTION` workaround. |
| `n05_dot_reduce.py` | `tl.dot` + reduce | Divergence in a matmul-shaped kernel — bridges to the M3 GEMM-equivalence milestone. |
| `_helpers.py` | — | Self-contained IR-capture + exact-bit grouping + adversarial inputs. |
| `run_all.py` | — | Runs `n01..n05`; self-skips with no GPU. |
| `test_smoke.py` | — | Pytest: each example's `main()` self-asserts `>1` bit-class. |

Inputs are **adversarial by construction** (`_helpers.adversarial_1d/2d`): random
mantissas × a wide dynamic range (`logspace(-6, 6)`) × alternating signs, seeded
for reproducibility. The wide spread plus catastrophic cancellation make the
result maximally order-sensitive, so different reduction trees reliably land in
different bit-classes.

## What to expect

- Reductions (`n01`–`n04`) across `num_warps ∈ {1,2,4,8}` produce **2–4 distinct
  bit-classes**. Some configs may *coincidentally* share a class — you can't
  eyeball which, which is the whole point.
- `n03`/`n04` typically split *more* than `n02` because `exp` and the chained
  mean→variance reduction amplify ordering differences.
- For the GEMM (`n05`) the **tiling** knobs (`num_warps`, `BLOCK_K`) are bitwise-
  *stable* — the MMA accumulates K in fixed instruction-sized chunks regardless of
  tiling — while the **precision** knob (`input_precision`) diverges. This is the
  precise insight motivating M3: MMA equivalence is a different sub-problem from
  reduction equivalence.
- Every config stays numerically **close to the fp64 reference**: the issue is
  **determinism**, not accuracy. "Different bits ≠ wrong."

For the softmax (`n03`) note the inputs are deliberately **unit-scale** (not the
extreme `adversarial` range): a huge range makes softmax collapse to one-hot, which
is bit-identical regardless of order. If any example ever yields one class on some
GPU, its assertion message tells you what to bump.

### Observed (2026-06-08, GB300 / sm_103a, Triton 3.6.0+fb.beta)

| Example | Knob swept | Distinct bit-classes |
|---------|------------|----------------------|
| n01 autotuner | `num_warps` {1,2,4,8} | **3** (autotuner picked num_warps=8) |
| n02 sum | `num_warps` {1,2,4,8} | **3** |
| n03 softmax | `num_warps` {1,2,4,8} | **4** |
| n04 layer-norm | `num_warps` {1,2,4,8} | **4** |
| n05 GEMM tiling | `num_warps`,`BLOCK_K` | **1** (stable) |
| n05 GEMM precision | `input_precision` {ieee,tf32} | **2** |

## How to run

GPU host (this project): **devgpu013** (GB300 / Blackwell), Python at
`/home/youngzt/gpu_test_venv/bin/python`.

**Important:** the PyPI `triton==3.3.1` wheel in that venv **cannot target sm_103a**
(its bundled LLVM is too old) — it crashes on these reduction/MMA kernels. On
Blackwell, run against the fork's in-tree build by putting it on `PYTHONPATH` and
enabling in-tree backend discovery (the fork is built but not pip-installed):

```bash
export TRITON_BACKENDS_IN_TREE=1
export PYTHONPATH=/home/youngzt/bitwise-equiv/triton/python
PY=/home/youngzt/gpu_test_venv/bin/python

# one example
$PY bitequiv/examples/numerical-inconsistency/n02_sum_reduction_classes.py
# all of them
$PY bitequiv/examples/numerical-inconsistency/run_all.py
# smoke test (needs pytest installed)
$PY -m pytest bitequiv/examples/numerical-inconsistency/test_smoke.py
```

On a non-Blackwell GPU with a matching Triton install, the two env vars are
unnecessary. On a host without any CUDA GPU every script self-skips cleanly.

## Reproducing after the bug is fixed

The pinned sha in `UPSTREAM_COMMIT.txt` is the base where this divergence
reproduces. If a future upstream default makes reductions ordered (collapsing the
classes), the examples still *run* but `len(classes)` becomes 1 and the assertions
fail **loudly** — signaling that the behavior changed. To reproduce the original
divergence, build Triton at the pinned commit and re-run.

## Out of scope (future work)

- **Before/after benchmark** (`config → latency → bit-class`, the perf cost of
  enforcing equivalence) — intentionally deferred. The design exists; build it
  once the project's equivalence pruner is wired in so "after" comes from the real
  feature rather than a manual bit-class filter.
