# AMD TLX Plan IR prototype

This directory implements M1.1--M1.5b.4a of the profile-guided TLX scheduling
design without changing the existing TLX kernels. M1.5b.2 reuses and extends
Meta's shared modulo-scheduling DDG through backend-neutral APIs.

## Implemented milestones

- **M1.1 — reproducible baseline:** a fixed BF16, non-causal, hDim-128 FA
  backward case catalog; fixed plain-Triton fused and TLX fused-bridge tuning
  configurations; revision/device/measurement manifests; content-addressed
  compiler-artifact capture.
- **M1.2 — PlanBundle extraction:** deterministic JSON for operations, semantic
  dot fragments, layouts, LDS allocations, synchronization, and final textual
  schedule. Scheduled-MFMA contracts include output role, resident operand,
  accumulator lifetime, register class, initialization, shapes, slices, and MMA
  layout.
- **M1.3 — exact replay verification:** location-independent TTGIR
  normalization, layer hashes, PlanBundle diff, and re-extraction verification.
- **M1.4a — stable native value graph:** an opt-in, analysis-only AMD compiler
  pass after final structured scheduling and before SCFToCF. It emits stable
  operation/value IDs, structured-loop iteration-distance edges, derived-value
  lineage, logical tensor sizes, and explicit identity-quality diagnostics.
- **M1.4b — static liveness and logical LDS model:** structured block program
  points, half-open value intervals, alias roots/views, normalized constant and
  modulo slot paths, memory effects, and per-allocation alias-union intervals.
- **M1.4c — asynchronous LDS lifetime model:** async write transactions,
  commit groups, partial waits, visibility and release barriers, consumers,
  structured-loop iteration distance, symbolic slot overwrite, and explicit
  reuse hazards.
- **M1.4d — verified audit contract:** explicit SSA, memory, async, barrier,
  and slot-reuse dependencies; block-local peak logical live sets; logical
  resource summaries; typed unresolved facts; and a strict machine-readable
  and Markdown audit for pinned kernels.
- **M1.5a — intra-iteration plan application:** a versioned schedule-delta
  contract and AMD final-structured-TTGIR pass that resolve stable IDs,
  validate exact baselines, anchors, and distance-zero dependencies, then
  apply complete block-local permutations without changing iteration,
  staging, synchronization, or dot decomposition.
- **M1.5b.1 — pipeline-delta contract:** a separate versioned intent schema for
  cross-iteration async-group placement and global/register-to-LDS staging. A
  dry-run validator pins the input value graph, resolves structured loops,
  complete async groups, LDS slot paths, staged tensor uses, depth, and
  alignment, and reports the changes a later native materializer must make.
- **M1.5b.2 — existing-LDS pipeline application:** an AMD pass immediately
  before async wait-count adjustment resolves complete async families, exact
  loop distance and modulo slot depth, projects Plan dependencies into Meta's
  shared DDG, and uses constrained modulo scheduling to reorder existing
  producer slices. It preserves LDS allocations, waits/barriers, iteration
  storage, prefetch distance, buffer depth, and dot decomposition.
- **M1.5b.3 — existing-ring depth and synchronization:** the native pass
  resolves complete existing-LDS roots, checks target capacity, rewrites a
  canonical leading ring dimension and producer/consumer modulo indices,
  derives retained counts for complete or partial waits, and inserts missing
  visibility/release barriers. It then re-extracts aliases, async lifetimes,
  hazards, resources, and a second DDG; acceptance requires the requested
  depth/distance, no open important fact, no LDS reuse hazard, a legal modulo
  schedule, and an unchanged dot contract.
- **M1.5b.4a — single-slot register-to-LDS staging:** the native pass resolves
  a produced tensor and its complete set of direct in-loop consumers, checks
  known size, alignment, layout preservation, and target LDS capacity, then
  inserts a typed mutable LDS allocation, local store/load, and visibility and
  release barriers. It re-extracts Plan IR, proves that the original register
  value ends at the store, verifies the rebuilt distance-zero DDG, and freezes
  the dot/scheduled-MFMA contract.

M1.5a lowers only verified intra-iteration schedule permutations. M1.5b.1
validates cross-iteration intent. M1.5b.2 applies schedules already represented
by an existing ring. M1.5b.3 changes depth/distance and synchronization only
for that existing ring. M1.5b.4a introduces synchronous single-slot
`register_to_lds`; buffered staging, derived-use rewriting, mixed ring/staging
plans, and `global_to_lds` remain later M1.5b.4 work.

## Reproduction

Run from the Triton repository root:

```bash
export PYTHONPATH=third_party/tlx/tools/plan_ir

python3 -m tlx_plan catalog --output /tmp/fa_bwd_catalog.json
python3 -m tlx_plan manifest \
  --case mha_n2048_d128 \
  --schedule tlx_fused_bridge \
  --source-root "$PWD" \
  --output /tmp/baseline.json
python3 -m tlx_plan extract \
  --ttgir /path/to/final.ttgir \
  --value-graph /tmp/plan-values/<fingerprint>.plan-values.json \
  --manifest /tmp/baseline.json \
  --output /tmp/plan.json
python3 -m tlx_plan replay \
  --ttgir /path/to/final.ttgir \
  --value-graph /tmp/plan-values/<fingerprint>.plan-values.json \
  --plan /tmp/plan.json \
  --normalized-output /tmp/replayed.ttgir \
  --report /tmp/replay.json
```

Generate the native sidecar during AMD compilation by setting:

```bash
export TRITON_TLX_PLAN_ANALYSIS_DIR=/tmp/plan-values
```

The pass is disabled by default and does not mutate TTGIR. The sidecar reports
logical tensor/LDS bytes, static TTGIR program-order intervals, and structured
iteration distances. Physical VGPR allocation and physical LDS placement remain
unknown. The async model reports commit-count and CTA-barrier frontiers, not
hardware cycles.

Compare two extracted plans with:

```bash
python3 -m tlx_plan diff /tmp/baseline-plan.json /tmp/candidate-plan.json
```

Run the strict M1.4d audit with:

```bash
python3 -m tlx_plan audit \
  --plan /tmp/plan.json \
  --output /tmp/plan-audit.json \
  --markdown-output /tmp/plan-audit.md
```

Create and validate an identity M1.5a block-local schedule delta with:

```bash
python3 -m tlx_plan schedule-delta \
  --plan /tmp/plan.json \
  --output /tmp/schedule-delta.json
python3 -m tlx_plan validate-schedule-delta \
  --delta /tmp/schedule-delta.json \
  --plan /tmp/plan.json
```

The desired order may then be edited as a complete permutation. Native
application checks the exact baseline order and all distance-zero dependencies
before changing final structured TTGIR.

Apply a delta during AMD compilation with:

```bash
export TRITON_TLX_SCHEDULE_PLAN=/tmp/schedule-delta.json
export TRITON_TLX_PLAN_APPLY_REPORT=/tmp/plan-apply-report.json
```

The compiler hook skips unrelated helper-kernel modules, but the standalone
pass rejects a missing target kernel unless `allow-missing-kernel=true` is
explicitly requested. The report records stable input/output fingerprints,
checked dependencies, pinned anchors, moved operation counts, and confirms
that iteration placement, storage, synchronization, and dot decomposition are
unchanged.

Create and dry-run an identity M1.5b.1 pipeline delta with:

```bash
python3 -m tlx_plan pipeline-delta \
  --plan /tmp/plan.json \
  --output /tmp/pipeline-delta.json
python3 -m tlx_plan validate-pipeline-delta \
  --delta /tmp/pipeline-delta.json \
  --plan /tmp/plan.json \
  --output /tmp/pipeline-delta-validation.json
```

Edit `loops` to request `set_prefetch_distance`, `global_to_lds`, or
`register_to_lds`. Validation is analysis-only: the report always records
`materialization_status: not_applied` and never treats a valid request as
proof that a physical modulo schedule, LDS allocation, or synchronization
sequence exists.

Dump the exact pre-wait-count sidecar and apply an existing-LDS pipeline delta
during AMD compilation with:

```bash
export TRITON_TLX_PIPELINE_ANALYSIS_DIR=/tmp/pipeline-plan-values
export TRITON_TLX_PIPELINE_PLAN=/tmp/pipeline-delta.json
export TRITON_TLX_PIPELINE_APPLY_REPORT=/tmp/pipeline-apply-report.json
```

Every requested async group sharing a positive-distance wait must be present.
The M1.5b.2 identity path requires exact analyzed distance/depth and preserves
the stable fingerprint. M1.5b.3 accepts a changed depth/distance only when all
readers and writers of the existing root are selected and directly indexed. It
rejects capacity overflow and new-staging intents. Its report records LDS bytes,
rewritten slots, waits/barriers, pre/post fingerprints, the post-rewrite audit,
and second-DDG verification.

A staging-only loop may instead contain one or more `register_to_lds` entries
with `buffer_depth: 1`. M1.5b.4a requires a produced ranked tensor, all direct
uses and named consumers in the selected `scf.for`, known logical bytes, and a
power-of-two alignment. The report uses
`materialization_scope: register_to_lds_staging` and records the new staging
and synchronization changes.

The fixed configuration names and kernel symbols are sourced from
`third_party/tlx/tutorials/amd_fa_bwd.py`:

| Plan | Kernel | Fixed config | Algorithm |
|---|---|---|---|
| `triton_fused_bf16` | `_attn_bwd_dkdv_dq_d128_triton_fused_kernel` | BM=32, BN=128, warps=4, stages=2, dQ subtile=2 | short persistent fused |
| `tlx_fused_bridge` | `_attn_bwd_dkdv_dq_d128_gqa_kernel` | BM=16, BN=256, warps=4, stages=1 | long fused bridge |

The core catalog contains MHA N=2048 and production-like GQA N=1024 through
16384, all with hDim=128 and non-causal execution.
