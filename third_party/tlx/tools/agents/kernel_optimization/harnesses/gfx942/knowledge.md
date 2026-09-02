# gfx942 — CDNA3 / MI300X

Architecture-wide knowledge, shared by every target under `harnesses/gfx942/`.
Per-op structure lives in that op's `targets/<kernel>/optimization_guidance.md`.

Two things are deliberately **not** here. Hardware quantities: `Gfx942` in
`third_party/tlx/language/tlx/hw/resources.py` is ground truth for CU count, XCD
count and LDS budget, and executable heuristics tune against it — cite the
attribute, never copy its value. Measured figures: they belong in the run
artifacts and the perf suite, and a number copied into a prompt goes stale
without anything noticing. What follows is mechanism and method.

## 1. Mechanism

- **Workgroups are dispatched round-robin across the XCDs** (`Gfx942.num_xcds`).
  Tiles that should share an operand in one L2 land on different chiplets unless
  the grid is remapped. A GROUP_M swizzle targets the same reuse, so the two are
  substitutes rather than additive — expect the second one applied to do little.
- **LDS is per-workgroup and small** (`Gfx942.lds_bytes`; CDNA4's
  `Gfx950.lds_bytes` is substantially larger). Ring depth and tile width compete
  for the same bytes, linearly. This is why a tile ported from a gfx950 kernel
  usually will not fit, and why depth is a weaker lever here than the tile
  dimensions it displaces.
- **fp16 MFMA is `v_mfma_f32_16x16x16_f16` / `v_mfma_f32_32x32x8_f16`** — half
  the K depth of CDNA4's equivalents, so a K-blocking choice tuned on CDNA4
  under-feeds the pipe here. Pass `matrix_instr_nonkdim=16`.
- **No C launch dispatcher.** It needs the NVIDIA backend's
  `asm["launch_metadata"]`, which the AMD backend never emits, so every launch
  goes through Python and carries that fixed cost. `TRITON_USE_C_DISPATCHER=0`
  skips a branch that cannot succeed.
- **Infinity Cache is far larger than the `L2_cache_size` torch reports.** Sizing
  a cache-flush rotation against the torch value under-rotates by more than an
  order of magnitude and leaves the measurement warm.

## 2. Method

- **`do_bench` in short bursts is not a valid gate on this part.** Bursts
  separated by sleeps prevent clocks from ever settling, and the burst-to-burst
  spread swamps the promotion threshold. Warm continuously, take one timed
  window, report p50 with a p20/p80 band. `gfx942_perf_harness.measure_samples`
  implements this; do not hand-roll a timing loop.
- **Distinguish launch-bound rows from kernel rows before reading them.** With no
  C dispatcher, a small shape measures the autotuner wrapper plus a Python launch
  rather than the kernel. A row whose per-call time is the same order as the
  launch cost is an end-to-end statement, not a kernel statement, and optimizing
  the kernel will not move it.
- **Dropping `do_bench` also drops its cache flush**, so iterations run warm
  against whatever the previous one left resident. Cold-cache numbers need
  deliberate buffer rotation sized to Infinity Cache.
- Lock clocks with `third_party/tlx/denoise.sh` (`rocm-smi --setperfdeterminism`
  plus power overdrive). Pin to an idle GPU; a concurrent job perturbs power and
  thermals even on another GPU of the same node.

_Rationale and the measurements behind §2:
`third_party/tlx/tutorials/testing/gfx942_perf_harness.py` docstring._

## 3. Ported from gfx950 (CDNA4) — hypotheses, not established here

From the v0→v9 ladder in
`third_party/tlx/tutorials/gfx9_gemm/a16w16/README.md` (MI355, fp16), which
holds the per-step figures. **That ladder stages operands direct-to-LDS via
`buffer_load_to_local`. The load mechanism does not transfer; the scheduling
ideas are separable from it, which is why this section exists.**

- **`tlx.warp_pipeline_stage` (→ `s_setprio`) is the largest hot-loop win there.**
  Each region splits into an MFMA stage (priority 0, yields) and a memory stage
  (priority 1, issues first) so loads overlap the math. Needs 8 warps so
  accumulators stay in AGPRs. (Mechanism: `v8_warp_pipeline/matmul_kernel.py`.)
- **N-slicing B into halves with a manual 4-region pipeline**, and **step-2 loop
  unrolling with alternating register sets**, are the two steps below it.
- **Never pass `other=0.0` to `buffer_load`.** The compiler emits register copies
  to implement the fallback for masked-out lanes, and on AMD a masked
  `buffer_load` already returns zero — so the fallback is pure overhead. This one
  is a large regression, not a marginal one.
- **`TRITON_DISABLE_POST_MISCHED=1` protects a hand-scheduled hot loop** from the
  LLVM post-RA machine scheduler re-ordering the interleave.
- **The residual gap to rocBLAS there is LLVM instruction scheduling**, which
  Gluon's best closes with custom `llirSched` / `amdgcnSched` passes the default
  backend scheduler does not match. Treat that as the ceiling of what schedule
  tuning at this level can reach.

## 4. Open — no evidence either way

- Whether `s_setprio` scheduling helps a *register-staged* hot loop as much as a
  direct-to-LDS one. The mechanism — issue memory ahead of MFMA — is not
  obviously tied to the staging path, but nothing has been run. The 8-warp
  requirement is already met by the wide gfx942 configs, so that is not the
  blocker it would be at 4 warps.
- `waves_per_eu` is pinned at 0 in every shipped config and has never been swept.
- The Python launch cost is not a kernel problem and no work has gone into it.
