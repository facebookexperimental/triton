# Evaluation — M1 TTGIR reduction-equivalence check

Validates `bitequiv.reduction_tree` (the M1 static reduction-equivalence check)
against **real compiled kernels** — the first time it runs on real TTGIR rather than
hand-written fixtures. Two-way, over a config matrix per kernel:

- **Part 1 — detection (recall):** for config pairs whose outputs differ bit-for-bit,
  does our static check flag them as *not* equivalent?
- **Part 2 — soundness (FP rate):** for pairs our check declares *equivalent*, run
  `R` random inputs and measure the rate they are **not** actually bit-equal. Must be 0.

It reuses the standalone repro kit on `../examples/numerical-inconsistency/`
(`_helpers.py`: `compile_only`, `bitclass_key`/`group_by_bits`, `adversarial_*`).

## Method

Per kernel, over `ordering ∈ {unordered, inner_tree} × num_warps ∈ {1,2,4,8} ×
num_stages ∈ {2,3}`: compile each config to TTGIR (no launch) and run it on `R=8`
adversarial random inputs. For every config pair compute

- `stat`  = `reduction_descriptor(ttgir_a) == reduction_descriptor(ttgir_b)`
- `emp`   = outputs bit-identical on **every** one of the `R` inputs

and bin into the confusion matrix:

|                | emp-equal | emp-different |
|----------------|-----------|---------------|
| **stat-equiv** | TP        | **FP — UNSOUND, must be 0** |
| **stat-noteq** | FN (conservative over-split, OK) | TN (detected) |

`unordered × num_warps` supplies the emp-different pairs to detect; `inner_tree`
(layout-invariant) and `num_stages`-only differences supply the non-trivial
stat-equiv pairs that test soundness.

It also **flags implementation/design problems** (and exits nonzero) without fixing
them: an empty descriptor (parser didn't match real TTGIR), or an `inner_tree` pair
that is empirically different (inner_tree not actually layout-invariant).

## Run

```bash
cd <m1 worktree>
PYTHONPATH=/home/youngzt/bitwise-equiv/triton/python \
  /home/youngzt/bitwise-equiv/triton/.venv/bin/python \
  bitequiv/evaluation/evaluate_reduction_equivalence.py
```

`PYTHONPATH` points `import triton` at the built in-tree fork. Requires a CUDA GPU
(self-skips without one).

## Result (2026-06-11, NVIDIA H100, cc 9.0, triton 3.6.0+fb.beta)

| kernel | reduces | unordered bit-classes | inner_tree bit-classes | TP | FP | TN | FN | detection | FP rate |
|--------|---------|----------------------|------------------------|----|----|----|----|-----------|---------|
| sum       | 1 (addf)          | 4 | 1 | 32 | 0 | 88 | 0 | 100% | 0% |
| softmax   | 2 (maxnumf, addf) | 4 | 1 | 32 | 0 | 88 | 0 | 100% | 0% |
| layernorm | 2 (addf, addf)    | 4 | 1 | 32 | 0 | 88 | 0 | 100% | 0% |

**No soundness violations.** `inner_tree` collapses to a single bit-class across all
configs (assumption confirmed); the parser matched real TTGIR (multi-reduce included);
and for these kernels the conservative descriptor was in fact *exact* (FN = 0, static
classes = empirical classes).

## Larger / more complex kernels — `evaluate_complex_kernels.py`

Same confusion framework, two groups. **In-scope** (gated on soundness `FP==0`) and
**boundary** (out of the current checker's scope — mapped, not gated). Run identically
but with `evaluate_complex_kernels.py`.

### Result (2026-06-11, NVIDIA H100, cc 9.0, triton 3.6.0+fb.beta, R=5 inputs/config)

IN-SCOPE — all sound (`FP=0`):

| kernel | static / emp classes | TP | FP | TN | FN | note |
|--------|----------------------|----|----|----|----|------|
| sum_looped (N=1M, loop+mask) | 5 / 5 | 32 | 0 | 88 | 0 | detection 100%; runtime-N & num_stages correctly irrelevant |
| reduce2d_axis0 | 5 / 5 | 6 | 0 | 22 | 0 | detection 100%; 2-D axis-0 projection validated |
| reduce2d_axis1 | 2 / 2 | 12 | 0 | 16 | 0 | detection 100%; 2-D axis-1 projection validated |
| sum_bf16 (bf16 in/out) | 5 / 1 | 6 | 0 | 0 | 22 | 16-bit rounding collapses all configs to 1 class → conservative over-split (sound) |
| sum_fp16 (fp16 in/out) | 5 / 1 | 6 | 0 | 0 | 22 | same |

For f32 (looped, 2-D) the conservative descriptor was again **exact** (static == empirical).
Half-precision is *sound but conservative* (it splits configs whose 16-bit outputs coincide).

BOUNDARY — out of full scope, now handled *conservatively* (sound), after the fix below:

| kernel | empty descr. | FP | result |
|--------|--------------|----|--------|
| argmin (multi-operand value+index) | 0/8 | 0 | conservatively split (the index is order-invariant, so all 28 emp-equal pairs are FN — sound) |
| welford (multi-operand, order-sensitive) | 0/8 | **0** | detection 100% (22 TN), 6 FN — the divergence is now **detected**, not falsely merged |
| gemm (`tl.dot` / MMA) | 0/6 | **0** | detection 100% (9 TN: `ieee`≠`tf32`), 6 FN (tiling over-split) — sound; full MMA analysis is M3 |

### Finding (FIXED this session): empty descriptor was treated as "equivalent to everything"

The first cut assumed an unparsed (multi-operand / MMA) reduce simply "doesn't match, no
false merge." That was **wrong**: an unmatched reduce produced an *empty* descriptor `()`,
and `() == ()` made the checker declare such configs **equivalent**. The evaluation caught
this as an unsound false-positive — welford **22 FP**, gemm **9 FP** (the single-operand
in-scope path was never affected).

**Fix applied** (`reduction_tree.reduction_descriptor`): a reduce-like op we cannot
structurally analyze (a multi-operand `tt.reduce`, or a tensor-core/MMA accumulation) now
emits a conservative `("unanalyzed", kinds, fingerprint)` descriptor entry instead of being
dropped. The fingerprint includes every layout-encoding body and reduce/dot line, so two
such configs are called equivalent **only when their IR is identical** — never on an empty
signature. `()` is now returned only when there is genuinely no reduction-like op.
Re-run result: **welford and gemm FP → 0**; no empty descriptors anywhere. Unit tests:
`test_multi_operand_*` / `test_mma_accumulation_is_sound` in `tests/test_reduction_tree.py`.

## Scope / next steps

All tested kernels are now **sound** (in-scope exact for f32; boundary conservatively split).
Next, to recover the tuning freedom the conservatism gives up: full multi-operand reduce
analysis (variadic operands, 2N-arg region); MMA/precision via `#mma` LinearLayout +
PTX-FMA backstop (M3); a before/after perf-vs-bit-class study.
