# AMD TLX Plan IR prototype

This directory implements M1.1--M1.4d of the profile-guided TLX scheduling
design without changing the existing TLX kernels or modulo scheduler.

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

M1.3 verifies that a captured baseline is reproducible. It does **not** lower an
arbitrarily mutated storage or schedule plan back into TTGIR. That mutation path
is intentionally left for the later candidate-generation/lowering milestone.

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

The fixed configuration names and kernel symbols are sourced from
`third_party/tlx/tutorials/amd_fa_bwd.py`:

| Plan | Kernel | Fixed config | Algorithm |
|---|---|---|---|
| `triton_fused_bf16` | `_attn_bwd_dkdv_dq_d128_triton_fused_kernel` | BM=32, BN=128, warps=4, stages=2, dQ subtile=2 | short persistent fused |
| `tlx_fused_bridge` | `_attn_bwd_dkdv_dq_d128_gqa_kernel` | BM=16, BN=256, warps=4, stages=1 | long fused bridge |

The core catalog contains MHA N=2048 and production-like GQA N=1024 through
16384, all with hDim=128 and non-causal execution.
