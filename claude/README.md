# TTGIR-SCHED — docs index

A TTGIR-level replacement for the AMD `TRITON_ENABLE_LLIR_SCHED`
LLVM-IR scheduler pass. Lives in
`third_party/amd/lib/TritonAMDGPUTransforms/DotDecomposeAndSchedule.cpp`.

## Doc map

| File | What |
|---|---|
| [`llir_sched_at_ttgir_design.md`](llir_sched_at_ttgir_design.md) | **Why TTGIR.** Design rationale — what's wrong with the LLIR pass, why TTGIR is structurally safer, why M + N split is the right starting point, K-split deferred caveat. |
| [`llir_sched_at_ttgir_plan.md`](llir_sched_at_ttgir_plan.md) | **The phased plan.** 7-phase implementation roadmap (Phase 0 scaffold through Phase 6 docs). |
| [`ttgir_sched_status.md`](ttgir_sched_status.md) | **What's landed + how to use it.** Per-phase commit list, lit-test inventory, env-var contract, worked v8/v10 example, full e2e validation matrix. |
| [`phase4_coverage.py`](phase4_coverage.py) | **Driver script.** Runs the e2e coverage matrix on stand-alone matmul (K sweep) and produces a markdown summary table. |

## Quick reference

### Two opt-in env vars

| Env var | Effect | Default |
|---|---|---|
| `TRITON_ENABLE_TTGIR_SCHED` | Enable the pass (planning-only, no IR mutation, just remarks) | off |
| `TRITON_TTGIR_SCHED_APPLY` | Mutate IR: replace each candidate MFMA `tt.dot` with M × N sub-dots glued via `amdgpu.concat`, with `ROCDL::SchedBarrier(0)` between M-rows | off |
| `TRITON_TTGIR_SCHED_BARRIER_STRIDE` | Override the sched-barrier stride (0 = none, 1 = per-sub-dot) | numPartitionsN |

### One-line e2e (stand-alone matmul, K=4096):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate metamain2
cd ~/AMD/triton/claude/triton_kernels_baseline
TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 \
    HIP_VISIBLE_DEVICES=0 python _one_run_envcompare.py 4096
# → OK 888.70 (vs 853.75 baseline → +4.1 %)
```

### Lit tests

Six lit tests in `test/TritonGPU/amd/`:

```
amd-ttgir-sched-phase0-noop.mlir       # pass scaffold; no-op without env var
amd-ttgir-sched-phase1a-plan.mlir      # partition-plan detection
amd-ttgir-sched-phase1b-bwd.mlir       # backward walker (producer ops)
amd-ttgir-sched-phase1c-fwd.mlir       # forward walker (user ops)
amd-ttgir-sched-phase1d-apply.mlir     # M × N apply (32 sub-dots + concat)
amd-ttgir-sched-phase3-barrier.mlir    # SchedBarrier insertion, 3 stride modes
```

All pass via FileCheck (`triton-opt -tritonamdgpu-dot-decompose-and-schedule`).

## Headline results

| Workload | LLIR-SCHED (matmul_4waves) | TTGIR-SCHED (this work) |
|---|---|---|
| v8/v10 main loop                                  | ✅ +17–22 %               | (proxied via stand-alone) |
| Stand-alone autotuned matmul, K=4096              | ❌ crash on autotune       | ✅ +4.1 % |
| Stand-alone autotuned matmul, K=8192              | ❌ crash on autotune       | ✅ **+11.0 %** |
| FA-fwd tutorial                                   | ❌ SSA dominance crash     | ✅ correct, within ±2 % |

The matmul_4waves LLIR pass *crashes* on FA-fwd with `Instruction does
not dominate all uses!`. The TTGIR pass produces correct + competitive
output on the same kernel, because MLIR's typed SSA verifier rejects
ill-formed rewrites at construction time — the LLIR pass has no such
safety net.

## Commit log

```
0ab3e00e3 [claude] Phase 4: e2e coverage matrix (stand-alone K sweep + FA-fwd)
05cc4613b [claude] ttgir_sched_status.md: Phase 3 landed; e2e +1.8 % perf bump
b48a1bc43 [AMD][TTGIR-SCHED] Phase 3: insert ROCDL::SchedBarrier(0) between M-rows
381e4099a [claude] ttgir_sched_status.md: e2e validation results
48e3f0fdc [claude] ttgir_sched_status.md: Phase 0-2 progress tracker
7b7191047 [AMD][TTGIR-SCHED] Phase 2: compose N-split on top of M-split (M × N grid)
6cc0e7545 [AMD][TTGIR-SCHED] Phase 1d: apply M-split SSA rewrite
0c16f8e26 [AMD][TTGIR-SCHED] Phase 1c: forward walker for dot-result user ops
a2c8db5b0 [AMD][TTGIR-SCHED] Phase 1b: backward walker for producer ops
280370f92 [AMD][TTGIR-SCHED] Phase 1a: compute M-split partition plan per dot
cbfbda28f [AMD][TTGIR-SCHED] Phase 0: scaffold opt-in TRITON_ENABLE_TTGIR_SCHED pass
```

## What's NOT done

| Phase | Description | Why deferred |
|---|---|---|
| 5 | Default-disable LLIR pass when TTGIR-SCHED is active | Needs cross-repo coordination (LLIR pass lives on the matmul_4waves branch of ROCm/triton, not here in MetaMain2). Can be done via a Python-side gate in `compiler.py` when `TRITON_TTGIR_SCHED_APPLY=1`. |
| K-split | Decompose `local_load` to per-MFMA-vector grain | Requires extending the producer chain rewrite (Phase 1b's backward walker would need to slice the LDS reads too) — see design doc. |
| Dim-flipping ops | `BroadcastOp` / `ExpandDimsOp` / `TransOp` / `ReshapeOp` in the walkers | Only relevant if user kernels rely on these in the producer/user chain; v8/v10 / stand-alone matmul / FA-fwd don't need it. |
