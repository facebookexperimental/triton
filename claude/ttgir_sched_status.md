# TTGIR-SCHED Implementation Status

> Last updated: 2026-05-30. See `llir_sched_at_ttgir_design.md` for the
> design rationale and `llir_sched_at_ttgir_plan.md` for the original
> 7-phase plan.

## Where we are

**Phases 0, 1a-1d, 2 are landed** (6 atomic commits, all lit-tested):

| Phase | Commit | Lines | Status |
|---|---|---:|---|
| 0  Scaffold (no-op opt-in pass)                          | `cbfbda28f` | 196 | ✅ landed |
| 1a Compute M-split partition plan per dot                | `280370f92` | 145 | ✅ landed |
| 1b Backward walker (collect producer ops to co-partition)| `a2c8db5b0` | 117 | ✅ landed |
| 1c Forward walker (collect dot-result user ops)          | `0c16f8e26` | 130 | ✅ landed |
| 1d Actual SSA mutation (M-split apply via extract_slice + N dots + concat) | `6cc0e7545` | 155 | ✅ landed |
| 2  Compose N-split on top of M-split (M × N grid)        | `7b7191047` | 123 | ✅ landed |

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

## e2e validation (added 2026-05-30)

Stand-alone autotuned matmul (`~/AMD/triton/claude/triton_kernels_baseline/_one_run_envcompare.py`),
K=4096, BM=BN=256, BK=64, num_warps=8 (autotune-picked), on the
`metamain2` conda env:

| Mode | TFLOPS | Δ vs baseline | PyTorch match |
|---|---:|---:|---|
| Baseline (no TTGIR_SCHED)                | 850.24 | — | ✅ |
| `TTGIR_SCHED=1` (planning only)          | 850.53 | +0.03 % (noise) | ✅ |
| `TTGIR_SCHED=1 + TTGIR_SCHED_APPLY=1`   | 839.03 | **-1.3 %** (within ±5 % bar) | ✅ |

Headline:
- **Planning mode is a true no-op** (delta is run-to-run noise).
- **APPLY mode is numerically correct** (PyTorch reference matched after
  the 8 × 8 = 64 sub-dots + concat rewrite).
- **APPLY perf is within the ±5 % success criterion** — the small delta
  reflects LLVM-side scheduling/RA differences on the rewritten IR;
  Phase 2 doesn't yet add a beneficial reorder or sched_barrier (that's
  Phase 3's job).

**v8 from METAMD cannot be tested directly on `metamain2`** because
`v8_beyond_hotloop` imports `triton.experimental.gluon.language.amd.cdna3.extract_slice`,
which exists only on the matmul_4waves branch of ROCm/triton (the
`amd-triton` conda env). MetaMain2/triton's gluon.amd.cdna3 module
doesn't have it. This is the same `ImportError: cannot import name
'extract_slice'` we saw earlier when trying v8 on the `oss` env (see
the `gl_matmul_passes_summary.md` discussion).

Workarounds for the next session if a v8/v10-style validation is wanted:
  1. Port the `cdna3.extract_slice` op from matmul_4waves to MetaMain2's
     gluon AMD language (small port), OR
  2. Use a vanilla autotuned matmul (as done here) as the e2e proxy — it
     also lowers to MFMA `tt.dot` and exercises the same pass paths, OR
  3. Build a synthetic Gluon-equivalent v8 kernel using only ops
     MetaMain2 has natively.


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
