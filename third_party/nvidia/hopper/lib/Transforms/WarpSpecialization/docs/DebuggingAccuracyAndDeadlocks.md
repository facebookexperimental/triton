# Debugging AutoWS Accuracy / Deadlock Issues

A reusable triage process for AutoWS (automatic warp specialization) kernels
that produce **wrong results** or **hang**, especially when a known-good
reference exists (a TLX / manual-WS kernel, a non-persistent variant, or a
different config that works). Distilled from the FA-bwd persistent
staging-reuse race triage (see "Worked example" below).

The single most useful framing: **find the smallest delta between a passing and
a failing variant, then diff their synchronization.** Most AutoWS correctness
bugs are a *missing or degenerate barrier*, not wrong math.

---

## Tools & skills

| Tool / skill | Use |
|---|---|
| `autows-docs` skill | **Read first.** Maps the task to the right design doc (`ReuseGroups.md`, `TMAStoreWaitPipeline.md`, `TokenBarrierLowering.md`, `BarrierFusion.md`, `AccumulationCounters.md`). The docs encode the invariants you're about to check. |
| `ir-debugging` skill | `TRITON_KERNEL_DUMP=1` → `~/.triton/dump/` (`.ttir/.ttgir/.llir/.ptx/.cubin`); `MLIR_ENABLE_DUMP=1` for per-pass IR; `TRITON_ALWAYS_COMPILE=1` to bypass cache. |
| `barrier-visualization` skill | The **mbarrier phase model** (the key reasoning tool) + the coverage-table method + the column-packed TMEM aliasing hazard. |
| `autows-testing` skill | Correctness test commands (do **not** use the TLX `test_correctness.py` for AutoWS). |
| `compute-sanitizer` | `--tool racecheck` (SMEM data hazards), `--tool synccheck` (illegal/divergent barriers). At `/usr/local/cuda-*/bin/compute-sanitizer`. |
| `debug-failing-gpu` skill | Recover from GPU-busy/OOM before/while running repros. |
| `git log -S` / `git show <rev>:<file>` | Find the *right* reference revision — the matching reference kernel may be an **older** version of a file (kernels get refactored). |

---

## Process

### 1. Reproduce and isolate the exact config
Pin the failing config precisely (variant, `bwd_config_idx`, head dim,
`early_tma_store_lowering`, causal, shapes). Many configs are gated by
`pytest.skip` — the failing combo may have **zero test coverage**, so reproduce
it directly by setting `kernel.configs = [...]; kernel.cache = {}` like
`test_op` does.

Write an **accuracy benchmark** that runs the AutoWS kernel and a known-good
reference (TLX) on identical inputs, compares both to a torch reference *and to
each other*, and **runs N times** to expose non-determinism. (Template:
`third_party/tlx/tutorials/accuracy_bench_bwd_autows_vs_tlx.py`.)

### 2. Narrow to the smallest passing/failing delta
Sweep the axes and find the one flip:
- **config idx** (which scheduling/reuse config fails?),
- **persistent vs non-persistent** (same config) — isolates outer-tile reuse,
- **AutoWS vs TLX** — isolates the compiler vs the algorithm,
- **head dim / shape**.

The delta localizes the bug. (Here: only `ws_persistent` + idx 2 failed → the
bug is in persistent cross-tile reuse, not the algorithm or the config's math.)

### 3. Race or deterministic?
Run-to-run **variance** ≈ 0 ⇒ deterministic logic bug. Large variance ⇒ a race.
A race that doesn't always corrupt output still corrupts *sometimes* — rely on
variance, not a single pass.

### 4. compute-sanitizer on a *small* compiling repro
`racecheck` finds SMEM hazards; `synccheck` finds illegal/divergent barriers.
Precompile once, then run the sanitizer (it's slow). **Caveats that will mislead
you if ignored:**
- racecheck **cannot see TMEM** races — only shared memory.
- racecheck **false-positives on mbarrier kernels** (it models
  `__syncthreads`/named barriers, not the producer/consumer phase protocol).
  *Cross-check:* run it on a **known-correct** kernel (e.g. the forward) — the
  same hazard class there means it's a false positive.
- Use the variance/output signal as the ground truth for "is it actually
  wrong"; use the sanitizer to *locate* candidate SMEM accesses.

### 5. Rule out hypotheses with the PTX
Check the lowered PTX before trusting an IR-level theory. Example: TMA-store
draining — `cp.async.bulk.wait_group.read 0` everywhere means every TMA store
fully drains, so any "store-still-in-flight" theory for that buffer is dead.

### 6. Diff the synchronization: broken vs reference
Dump the **final TTGIR** for the broken kernel and the working reference, and
diff their barrier/reuse structure. Drive it with the `barrier-visualization`
**coverage table**: for every physically-reused/aliased buffer, enumerate the
ordering edges that *should* exist (forward producer→consumer **and** backward /
cross-iteration consumer→next-producer) and mark each ✓ present or ✗ MISSING.
Count-balancing (`#arrive == #wait`) gives false passes — an *absent*
cross-alias edge is invisible to it.

If the obvious reference isn't the same shape (e.g. the current TLX bwd is
one-tile-per-CTA while AutoWS is grid-stride persistent), **`git log` the file**
for an older revision that matches.

### 7. Root-cause with the mbarrier phase model
Reason, don't tally. Genuine deadlock requires one of: (1) no arrive ever
produces the awaited phase, (2) parity-cadence starvation, (3) a cross-partition
cycle. Genuine race = a missing/degenerate ordering edge. Degeneracy signatures
seen in practice:
- **Constant phase** on a reuse wait (literal `0`/`1` instead of an
  `accumCnt`/IV-derived phase) ⇒ never alternates across the loop ⇒ no-op after
  the first iteration.
- **Same-task channel**: a producer/consumer pair in the *same* partition has
  its consumer-release **elided** by `WSLowerToken`'s same-partition elision, so
  the surviving acquire is a no-op. A cross-partition edge must be modeled with
  a token whose producer and consumer are in **different** tasks.
- **Different warp counts**: you cannot add a second consumer-release of a
  different warp count to a host buffer's empty barrier (`consumerWarps ==
  nWarps` assert) — use a dedicated token.

### 8. Fix and verify
Verify the *output* is correct **and deterministic** across many runs/seeds;
re-run `racecheck`/`synccheck`; run the full WarpSpecialization lit suite for
regressions; confirm the other configs and the passing variants are unchanged.
**Add regression coverage** for the previously-uncovered config.

---

## Worked example: FA-bwd persistent staging-reuse race

- **Symptom:** `fused_attention_ws_device_tma` bwd, `ws_persistent` +
  `early_tma_store_lowering` + `bwd_config_idx=2`, hDim 128 — wrong, non-
  deterministic gradients (~3-4 vs ~3e-3). Non-persistent and TLX correct.
- **Isolation:** only `ws_persistent`+idx2 failed; non-persistent idx2 and TLX
  passed ⇒ persistent cross-tile reuse.
- **Ruled out:** the 3-group `{dpT,dq,dsT}` *TMEM* reuse (racecheck is TMEM-
  blind; its cross-iter WAR is present); `can_rotate_by_buffer_count` (PTX =
  `wait_group 0`, stores drain); operand↔MMA reuse (armed backward barriers
  present).
- **Reference:** the current TLX bwd is one-tile-per-CTA; the matching
  *persistent* TLX bwd was an **April** revision (`git log` the file) — CLC
  work-stealing with the same `sdv reuses v_tiles` / `sdk reuses k_tiles`
  pattern, guarded by a dedicated `k_empties` arrived *after* the store drain.
- **Root cause:** dv/dk TMA-staging buffers alias the v/do operand SMEM
  (`allocation.reuseTarget`); the Step-7.5 guard was degenerate (constant
  `phase=0`, and a same-task channel whose release is elided) ⇒ the next tile's
  load raced the previous tile's staging store.
- **Fix:** a dedicated single-buffered cross-partition reuse token — load task
  acquires at the outer-loop top, staging task releases at the bottom (loop-
  carried phase). See `TMAStoreWaitPipeline.md` §"Step 1 … Step 7.5" and commit
  `e9135d565`.
- **Lesson:** the bug was *not* where racecheck pointed (TMEM is invisible to
  it) nor where the first hypothesis aimed (TMA drain) — the
  passing-vs-failing-variant delta plus the TLX barrier diff localized it.
