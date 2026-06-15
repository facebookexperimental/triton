# Evaluation — M1 reduction-equivalence checks

GPU evaluation of the static reduction-equivalence checkers against **real compiled
kernels**, comparing each checker's verdict to empirical bit-equality (`torch.equal`)
over an adversarial config matrix. Two harnesses on this branch (the PTX/FMA backstop
eval lives on the sibling `bitequiv-m1-ptx` branch):

- `evaluate_reduction_equivalence.py` — the **TTGIR** checker (`bitequiv.reduction_tree`)
  on simple reductions (sum / softmax / layernorm).
- `evaluate_complex_kernels.py` — the TTGIR checker on larger / boundary kernels
  (looped, 2-D, bf16/fp16, argmin, welford, gemm).

Both share one confusion matrix (per static relation):

|                | emp-equal | emp-different |
|----------------|-----------|---------------|
| **stat-equiv** | TP        | **FP — UNSOUND, must be 0** |
| **stat-noteq** | FN (conservative over-split, OK) | TN (detected) |

`PYTHONPATH` points `import triton` at a built in-tree fork (the checkers call into
`libtriton.bitequiv`); all harnesses require a CUDA GPU.

---

# Part A — TTGIR reduction-equivalence check

Validates `bitequiv.reduction_tree` against real compiled kernels — two-way, over a
config matrix per kernel:

- **Part 1 — detection (recall):** for config pairs whose outputs differ bit-for-bit,
  does the static check flag them as *not* equivalent?
- **Part 2 — soundness (FP rate):** for pairs the check declares *equivalent*, run `R`
  random inputs and measure the rate they are **not** actually bit-equal. Must be 0.

It reuses the standalone repro kit in `../examples/numerical-inconsistency/`
(`_helpers.py`: `compile_only`, `bitclass_key`/`group_by_bits`, `adversarial_*`).

## Method

Per kernel, over `ordering ∈ {unordered, inner_tree} × num_warps ∈ {1,2,4,8} ×
num_stages ∈ {2,3}`: compile each config to TTGIR (no launch) and run it on `R=8`
adversarial random inputs. For every config pair compute `stat = reduction_descriptor`
equality and `emp = bit-identical on every input`. `unordered × num_warps` supplies the
emp-different pairs to detect; `inner_tree` (layout-invariant) and `num_stages`-only
differences supply the non-trivial stat-equiv pairs that test soundness. It also flags
implementation/design problems (empty descriptor; an `inner_tree` pair that is
empirically different) and exits nonzero.

## Run

```bash
PYTHONPATH=/home/youngzt/bitwise-equiv/triton/python \
  /home/youngzt/bitwise-equiv/triton/.venv/bin/python \
  bitequiv/evaluation/evaluate_reduction_equivalence.py
```

## Result (2026-06-11, NVIDIA H100, cc 9.0, triton 3.6.0+fb.beta)

| kernel | reduces | unordered bit-classes | inner_tree bit-classes | TP | FP | TN | FN | detection | FP rate |
|--------|---------|----------------------|------------------------|----|----|----|----|-----------|---------|
| sum       | 1 (addf)          | 4 | 1 | 32 | 0 | 88 | 0 | 100% | 0% |
| softmax   | 2 (maxnumf, addf) | 4 | 1 | 32 | 0 | 88 | 0 | 100% | 0% |
| layernorm | 2 (addf, addf)    | 4 | 1 | 32 | 0 | 88 | 0 | 100% | 0% |

**No soundness violations.** `inner_tree` collapses to a single bit-class across all
configs; the parser matched real TTGIR (multi-reduce included); for these kernels the
descriptor was in fact *exact* (FN = 0).

## Larger / boundary kernels — `evaluate_complex_kernels.py`

Same confusion framework, re-run against the **MLIR-native** checker (the signature is
the real axis-projected `LinearLayout`, not regex — see `design-doc.md` §5).

### Result (2026-06-15, NVIDIA H100, cc 9.0, triton 3.6.0+fb.beta)

IN-SCOPE — all sound (`FP=0`):

| kernel | TP | FP | TN | FN | note |
|--------|----|----|----|----|------|
| sum_looped (N=1M, loop+mask) | 32 | 0 | 88 | 0 | detection 100%; runtime-N & num_stages correctly irrelevant |
| reduce2d_axis0 | 6 | 0 | 22 | 0 | detection 100%; 2-D axis-0 validated |
| reduce2d_axis1 | 6 | 0 | 16 | 6 | detection 100%; 2-D axis-1 (sound; minor over-split) |
| sum_bf16 / sum_fp16 | 6 | 0 | 0 | 22 | 16-bit rounding collapses configs to 1 class → conservative over-split (sound) |

BOUNDARY — out of full scope, sound (and now *analyzed*, not just abstained):

| kernel | empty descr. | FP | result |
|--------|--------------|----|--------|
| argmin (multi-operand value+index) | 0 | 0 | conservatively split (index is order-invariant → FN; sound) |
| welford (multi-operand, order-sensitive) | 0 | **0** | **detection 100% (22 TN)** — divergence detected via the axis-LL, not merged |
| gemm (`tl.dot` / MMA) | 0 | **0** | detection 100% (9 TN: `ieee`≠`tf32`); tiling over-split — full MMA is M3 |

The earlier (regex) empty-descriptor soundness hole is structurally gone: the C++ analysis
emits a real signature for every `tt.reduce` and a conservative `unanalyzed-mma` guard for
tensor-core accumulation, so `()` now means only "no reduction-like op." Unit tests:
`test_multi_operand_argmin_sound` / `test_gemm_mma_sound` in `tests/test_reduction_tree.py`.

---

# Part B — PTX/FMA backstop (sibling branch)

The PTX/FMA backstop evaluation (`evaluate_ptx_reduction_equivalence.py`) lives on the
sibling **`bitequiv-m1-ptx`** branch, together with the `bitequiv.ptx_reduction` engine.
It validates — same confusion methodology — that the **PTX** descriptor catches the
`mul`+`add → fma` fusion (decided below TTGIR, gated by `enable_fp_fusion`) that the
TTGIR checker is provably blind to: identical TTGIR, different PTX (`fma.rn.f32` vs
`add.rn.f32`+`mul.rn.f32`), different bits. On a `tl.sum(x*y)` dot reduction it closes
**20 TTGIR false-positive pairs** with `FP=0` and 0 refinement violations (PTX refines
TTGIR — never merges a pair TTGIR splits). See that branch's `evaluation/README.md` for
method, the per-kernel result table, and the implicit-`.rn` normalization follow-up.
