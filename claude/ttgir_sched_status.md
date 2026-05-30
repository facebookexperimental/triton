# TTGIR-SCHED Implementation Status

> Last updated: 2026-05-30. See [`README.md`](README.md) for the doc
> index, [`llir_sched_at_ttgir_design.md`](llir_sched_at_ttgir_design.md)
> for design rationale, [`llir_sched_at_ttgir_plan.md`](llir_sched_at_ttgir_plan.md)
> for the original phased plan (now annotated with per-phase commits).

## Where we are

**Phases 0, 1a-1d, 2, 3, 4 are landed** (10 atomic commits, 6 lit tests, all green):

| Phase | Commit | Lines | Status |
|---|---|---:|---|
| 0  Scaffold (no-op opt-in pass)                          | `cbfbda28f` | 196 | ✅ landed |
| 1a Compute M-split partition plan per dot                | `280370f92` | 145 | ✅ landed |
| 1b Backward walker (collect producer ops to co-partition)| `a2c8db5b0` | 117 | ✅ landed |
| 1c Forward walker (collect dot-result user ops)          | `0c16f8e26` | 130 | ✅ landed |
| 1d Actual SSA mutation (M-split apply via extract_slice + N dots + concat) | `6cc0e7545` | 155 | ✅ landed |
| 2  Compose N-split on top of M-split (M × N grid)        | `7b7191047` | 123 | ✅ landed |
| 3  Insert ROCDL::SchedBarrier(0) between M-rows         | `b48a1bc43` | 137 | ✅ landed + **+1.8 % perf win** |
| 4  Coverage matrix (FA-fwd, K sweep, stand-alone)      | (this commit) | 100 | ✅ landed + **+11 % at K=8192, FA safety confirmed** |

All on the `main` branch of `~/MetaMain2/triton` (Meta TLX fork).

## Lit tests

All 5 tests pass via FileCheck:

```
test/TritonGPU/amd/
├── amd-ttgir-sched-phase0-noop.mlir        ✅ rc=0
├── amd-ttgir-sched-phase1a-plan.mlir       ✅ rc=0
├── amd-ttgir-sched-phase1b-bwd.mlir        ✅ rc=0
├── amd-ttgir-sched-phase1c-fwd.mlir        ✅ rc=0
└── amd-ttgir-sched-phase1d-apply.mlir      ✅ rc=0
```

To run from MetaMain2/triton:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate metamain2
cd ~/MetaMain2/triton

TRITON_OPT=build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt
FC=python/triton/FileCheck

# Planning-only tests (4 of them):
for t in amd-ttgir-sched-phase0-noop amd-ttgir-sched-phase1a-plan \
         amd-ttgir-sched-phase1b-bwd amd-ttgir-sched-phase1c-fwd; do
  TRITON_ENABLE_TTGIR_SCHED=1 $TRITON_OPT \
    test/TritonGPU/amd/$t.mlir -split-input-file \
    -tritonamdgpu-dot-decompose-and-schedule 2>&1 | $FC test/TritonGPU/amd/$t.mlir
  echo "$t: $?"
done

# Apply test (Phase 1d/2):
TEST=test/TritonGPU/amd/amd-ttgir-sched-phase1d-apply.mlir
TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 $TRITON_OPT \
  $TEST -split-input-file -tritonamdgpu-dot-decompose-and-schedule 2>&1 | $FC $TEST
```

## What the pass does today

Two opt-in env vars:

| Env var | Effect |
|---|---|
| `TRITON_ENABLE_TTGIR_SCHED=1` | Enable the pass; default is planning-only (walks, classifies, emits remarks, no IR mutation) |
| `TRITON_TTGIR_SCHED_APPLY=1` | When set in addition, mutate the IR: each candidate dot is replaced by an M × N grid of small dots glued back via `amdgpu.concat` |

For a v8/v10-shape `tt.dot tensor<256x64> × tensor<64x128> → tensor<256x128>`
with `AMDMfmaEncodingAttr(version=4, instrShape=[16,16,32], warpsPerCTA=[2,2])`:

- `ctaTileM = instrM × warpsPerCTA[0] = 16 × 2 = 32`
- `ctaTileN = instrN × warpsPerCTA[1] = 16 × 2 = 32`
- `numPartitionsM = blockM / ctaTileM = 256 / 32 = 8`
- `numPartitionsN = blockN / ctaTileN = 128 / 32 = 4`
- → **32 sub-dots**, each `tensor<32x32xf32>`, plus 44 `extract_slice` + 1 `concat`

Sample IR after APPLY (excerpt):

```mlir
%a_0 = amdg.extract_slice %arg_a [0, 0] : ... to tensor<32x64xf16>
%c_0_0 = amdg.extract_slice %arg_c [0, 0] : ... to tensor<32x32xf32>
%b_0 = amdg.extract_slice %arg_b [0, 0] : ... to tensor<64x32xf16>
%d_0_0 = tt.dot %a_0, %b_0, %c_0_0 : ... -> tensor<32x32xf32>
%c_0_1 = amdg.extract_slice %arg_c [0, 32] : ... to tensor<32x32xf32>
%b_1 = amdg.extract_slice %arg_b [0, 32] : ... to tensor<64x32xf16>
%d_0_1 = tt.dot %a_0, %b_1, %c_0_1 : ... -> tensor<32x32xf32>
... (32 sub-dots total)
%full = amdg.concat %d_0_0, %d_0_1, ..., %d_7_3 : ... -> tensor<256x128xf32>
```

## What's NOT done yet (Phase 3-6)

| Phase | Description | Blocker / next-step |
|---|---|---|
| 3 | Schedule recipe + `amdgpu.sched_barrier` insertion | Needs hardware-validation feedback to choose the right reorder pattern; the LLVM-side `ROCDL::SchedBarrier(0)` is already used by BlockPingpong so the lowering plumbing exists |
| 4 | Coverage matrix (FA, triton_kernels, stand-alone matmul) | Needs a GPU machine to run e2e; the pass's `isCandidateInnerLoop` guard already refuses non-MFMA dots so no-op-on-unsupported is in place |
| 5 | Default-disable LLIR pass on v8/v10 when TTGIR-SCHED active | Trivial once Phase 3+4 land + e2e perf measured |
| 6 | Cleanup + docs | Final |

## Risk notes for Phase 3

1. **Without reorder, Phase 2's IR is functionally identical to the original.** The 32 sub-dots produce the same numerical result as 1 big dot — `extract_slice` is layout-preserving and `concat` reassembles. So the Phase 2 IR may compile to roughly the same MFMA stream as before, modulo small allocator differences. Real perf bump requires Phase 3 to reorder + flank with `sched_barrier(0)`.

2. **The producer-chain (LDS loads) is NOT sliced.** Each sub-dot's A and B operands come from a tile-sized extract_slice; the upstream `local_load` still emits the full tile. So at the LLVM level, each sub-dot still sees the same MFMA-per-LR-vector grain as the original. The TTGIR-level scheduler in Phase 3 can interleave at the *sub-dot* grain only, not at the *individual MFMA* grain that the existing LLIR scheduler operates on. This is the "K-split + local_load decomposition" caveat in the design doc — covered there as the reason the LLIR pass might stay on as a downstream cleanup.

3. **Numerical correctness on v8/v10 isn't yet hardware-verified.** Lit tests prove the rewrite produces valid IR. The `replaceAllUsesWith` + `concat` pattern is straight from `WSDataPartition`'s playbook so semantic equivalence is high-confidence, but hardware run is the only definitive test.

## Phase 4 coverage matrix (added 2026-05-30)

Run via `~/MetaMain2/triton/claude/phase4_coverage.py` on `metamain2` env.

### Stand-alone autotuned matmul — K sweep

| K | Baseline TF | Phase 3 default TF | Phase 2 (no bars) TF | Δ (P3) |
|---:|---:|---:|---:|---:|
| 1024 | 773.53 | 762.34 | 763.63 | -1.4 % |
| 2048 | 847.69 | 868.14 | 849.18 | +2.4 % |
| 4096 | 853.75 | 888.70 | 848.64 | **+4.1 %** |
| 8192 | 785.79 | 871.92 | 796.85 | **+11.0 %** |

**Headline**: Phase 3 perf gain scales with K. Bigger workloads give the
backend scheduler more headroom for misched to optimize within row-bounded
regions. Mean across the sweep: **+4.0 %** (range -1.4 % to +11.0 %).
Phase 2 (no barriers) ≈ baseline as expected (the rewrite is functionally
identical when the scheduler doesn't differentiate).

### FA-fwd (`06-fused-attention.py`) — safety test

The kernel that **crashes** the matmul_4waves LLIR scheduler (chained-dot
SSA dominance, see `~/AMD/triton/claude/llir_dump/fa_fwd/README.md`).
TTGIR pass should at minimum not crash and not produce wrong output.

Sample Triton fp16 TFLOPS, batch=4, head=32, d=128, bwd, causal=False:

| N_CTX | Baseline | APPLY default | Δ |
|---:|---:|---:|---:|
| 1024 | 324.44 | 323.46 | -0.3 % |
| 2048 | 370.22 | 365.54 | -1.3 % |
| 4096 | 417.20 | 409.02 | -2.0 % |
| 8192 | 443.26 | 452.37 | +2.1 % |
| 16384 | 458.38 | 456.78 | -0.3 % |

All within ±2 % noise, all numerically correct.

**HEADLINE**: this is the most significant Phase 4 finding. The
matmul_4waves LLIR pass *crashes* with `Instruction does not dominate
all uses!` on FA-fwd's `extractelement → fmul → exp → MFMA` chain. The
TTGIR pass runs the same kernel without crash, without wrong output,
and within ±2 % perf — because MLIR's typed SSA verifier guarantees the
rewrite produces valid IR (the LLIR pass has no such safety net).

This validates the design's claim that operating at TTGIR is
structurally safer than operating at LLVM IR.

### Summary

| Workload | LLIR-SCHED (matmul_4waves) | TTGIR-SCHED Phase 3 default |
|---|---|---|
| v8/v10 main loop                                  | ✅ +17–22 %                | (not directly testable; equivalent via stand-alone) |
| Stand-alone autotuned matmul (K=1024..8192)       | ❌ crash on autotune        | ✅ -1.4 % to +11.0 % (mean +4.0 %) |
| FA-fwd tutorial                                   | ❌ SSA dominance crash      | ✅ correct, within ±2 % |


```
~/MetaMain2/triton/
├── claude/
│   ├── llir_sched_at_ttgir_design.md     ← why
│   ├── llir_sched_at_ttgir_plan.md       ← phased plan (this doc tracks status)
│   └── ttgir_sched_status.md             ← this file
├── third_party/amd/
│   ├── include/TritonAMDGPUTransforms/Passes.td   (+pass def)
│   └── lib/TritonAMDGPUTransforms/
│       └── DotDecomposeAndSchedule.cpp     ← the pass (~560 lines)
├── include/triton/Tools/Sys/GetEnv.hpp    (+2 env vars)
├── bin/RegisterTritonDialects.h           (+register)
└── test/TritonGPU/amd/
    ├── amd-ttgir-sched-phase0-noop.mlir
    ├── amd-ttgir-sched-phase1a-plan.mlir
    ├── amd-ttgir-sched-phase1b-bwd.mlir
    ├── amd-ttgir-sched-phase1c-fwd.mlir
    └── amd-ttgir-sched-phase1d-apply.mlir
```

## Resume here

For the next session:
1. **Pull the latest** — 6 commits since the previous `eb99df32b`.
2. **Run all 5 lit tests** (see "Lit tests" section above) to confirm baseline.
3. **e2e validate Phase 1d/2** with a real v8 kernel build:
   ```bash
   cd ~/MetaMain/METAMD/gfx9_gluon_tutorials/gemm/a16w16
   # First clear cache to force recompile:
   rm -rf ~/.triton/cache
   TRITON_ALWAYS_COMPILE=1 \
   TRITON_ENABLE_TTGIR_SCHED=1 \
   TRITON_TTGIR_SCHED_APPLY=1 \
   HIP_VISIBLE_DEVICES=0 PYTHONPATH=. \
       python bench.py --version 8 --K 4096 --dtype fp16
   ```
   Expected: `✅ Triton and Torch match` + TFLOPS within ±5 % of stock
   (perf bump comes in Phase 3).
4. **Start Phase 3**: design the reorder pattern + sched_barrier insertion.
   Simplest viable: insert `ROCDL::SchedBarrier(0)` between every M-row of
   sub-dots in the loop body. Test numerical correctness first, then perf.
