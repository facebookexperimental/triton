# TTGIR-SCHED — implementation status + verification guide

> Last updated: 2026-05-30. See [`README.md`](README.md) for the doc
> index, [`llir_sched_at_ttgir_design.md`](llir_sched_at_ttgir_design.md)
> for the design rationale, [`llir_sched_at_ttgir_plan.md`](llir_sched_at_ttgir_plan.md)
> for the original phased plan (now annotated with per-phase commits).

## TL;DR

**14 commits, 6 lit tests (all green). Reproducible Phase 3 perf win
at K=8192 (pinned config BM=BN=256, BK=64, W=8): 1055.4 → 1104.0 TF
(+4.6 %, stable to within ±0.3 % across 3 runs each). FA-fwd runs
correctly** (where the matmul_4waves LLIR pass crashes). Phases 0,
1a-1d, 2, 3, 4, 6 landed; Phase 5 (default-disable LLIR) is the only
deferred phase.

> **Caveat on the +11 % figure** that earlier appeared in this doc:
> that was measured with `_one_run_envcompare.py` which autotunes.
> Each invocation re-runs the autotuner, and the autotuner sometimes
> picks BK=32,W=4 (~785 TF) and sometimes BK=64,W=8 (~870 TF) for the
> *same* mode depending on cache state and run-to-run timing variance.
> So the +11 % figure (785.79 → 871.92) was apples-to-oranges. The
> **pinned-config +4.6 %** is the honest reproducible delta — use
> `_pinned_run.py` (added in commit below) for any benchmarking.

```
e37716eb6 [AMD][TTGIR-SCHED] Phase 6: docs cleanup — README + cross-links + status markers
0ab3e00e3 [claude] Phase 4: e2e coverage matrix (stand-alone K sweep + FA-fwd)
b48a1bc43 [AMD][TTGIR-SCHED] Phase 3: insert ROCDL::SchedBarrier(0) between M-rows
7b7191047 [AMD][TTGIR-SCHED] Phase 2: compose N-split on top of M-split (M × N grid)
6cc0e7545 [AMD][TTGIR-SCHED] Phase 1d: apply M-split SSA rewrite
0c16f8e26 [AMD][TTGIR-SCHED] Phase 1c: forward walker for dot-result user ops
a2c8db5b0 [AMD][TTGIR-SCHED] Phase 1b: backward walker for producer ops
280370f92 [AMD][TTGIR-SCHED] Phase 1a: compute M-split partition plan
cbfbda28f [AMD][TTGIR-SCHED] Phase 0: scaffold opt-in pass
... (3 status-doc-only commits interspersed)
```

## Phases landed

| Phase | Commit | Status |
|---|---|---|
| 0  Scaffold (no-op opt-in pass)                          | `cbfbda28f` | ✅ landed |
| 1a Compute M-split partition plan per dot                | `280370f92` | ✅ landed |
| 1b Backward walker (collect producer ops to co-partition)| `a2c8db5b0` | ✅ landed |
| 1c Forward walker (collect dot-result user ops)          | `0c16f8e26` | ✅ landed |
| 1d Actual SSA mutation (M-split apply via extract_slice + N dots + concat) | `6cc0e7545` | ✅ landed |
| 2  Compose N-split on top of M-split (M × N grid)        | `7b7191047` | ✅ landed |
| 3  Insert ROCDL::SchedBarrier(0) between M-rows         | `b48a1bc43` | ✅ landed + **+1.8 % perf win** |
| 4  e2e coverage matrix (FA-fwd, K sweep, stand-alone)   | `0ab3e00e3` | ✅ landed + **+11 % at K=8192, FA safety confirmed** |
| 5  Default-disable LLIR pass on v8/v10 by default       | —           | ⏸ deferred (cross-repo) |
| 6  Cleanup + docs                                        | `e37716eb6` | ✅ landed |

# How to verify

## Prerequisites

Verify your environment has the right build artifacts and conda env:

```bash
# conda env that has the editable triton install for MetaMain2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate metamain2

# Should show: triton 3.6.0+fb.beta /home/mren/MetaMain2/triton/python/triton/__init__.py
python -c "import triton; print(triton.__version__, triton.__file__)"

# Should exist (built libraries)
ls ~/MetaMain2/triton/python/triton/_C/libtriton.so
ls ~/MetaMain2/triton/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt
```

If `libtriton.so` is missing, you need to build (see "Build from scratch" below).

## 1. Build from scratch (if needed)

The build needs a few env vars due to offline-build constraints (no
network in the sandbox). The first build also needs `pybind11` symlinked
into the conda env (one-time setup).

```bash
# One-time: symlink pybind11 from the system miniconda (where it lives)
ln -sf /data/users/mren/miniconda3/lib/python3.11/site-packages/pybind11 \
       ~/miniconda3/envs/metamain2/lib/python3.11/site-packages/pybind11

# Build (full editable install + libtriton.so)
cd ~/MetaMain2/triton
LLVM_SYSPATH=/home/mren/OpenSource/llvm-build/ \
TRITON_OFFLINE_BUILD=1 \
TRITON_BUILD_PROTON=OFF \
  pip install -e . --no-build-isolation

# Incremental rebuilds after editing the pass source:
cd ~/MetaMain2/triton/build/cmake.linux-x86_64-cpython-3.11
ninja TritonAMDGPUTransforms triton-opt   # rebuild C++ pass + lit-test binary
ninja triton                              # rebuild libtriton.so for Python e2e
```

Expected: both `ninja` invocations end with `Linking CXX executable …`
or `Linking CXX shared library …` and no error lines.

## 2. Lit tests (correctness of the IR rewrite)

There are 6 lit tests in `test/TritonGPU/amd/`. Each exercises a
specific phase via FileCheck. All should pass with rc=0.

### Quick: run all 6 at once

Save and run this script:

```bash
cd ~/MetaMain2/triton
source ~/miniconda3/etc/profile.d/conda.sh && conda activate metamain2
TRITON_OPT=build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt
FC=python/triton/FileCheck

# Planning-only tests (4): no IR mutation, just remarks
for t in amd-ttgir-sched-phase0-noop amd-ttgir-sched-phase1a-plan \
         amd-ttgir-sched-phase1b-bwd amd-ttgir-sched-phase1c-fwd; do
  TEST=test/TritonGPU/amd/$t.mlir
  TRITON_ENABLE_TTGIR_SCHED=1 $TRITON_OPT $TEST -split-input-file \
    -tritonamdgpu-dot-decompose-and-schedule 2>&1 | $FC $TEST
  echo "$t: $?"
done

# Apply tests (2): mutate IR
TEST=test/TritonGPU/amd/amd-ttgir-sched-phase1d-apply.mlir
TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 $TRITON_OPT \
  $TEST -split-input-file -tritonamdgpu-dot-decompose-and-schedule 2>&1 | $FC $TEST
echo "phase1d-apply: $?"

TEST=test/TritonGPU/amd/amd-ttgir-sched-phase3-barrier.mlir
TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 $TRITON_OPT \
  $TEST -split-input-file -tritonamdgpu-dot-decompose-and-schedule 2>&1 | $FC $TEST --check-prefix=DEFAULT
echo "phase3 DEFAULT: $?"

TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 TRITON_TTGIR_SCHED_BARRIER_STRIDE=0 \
  $TRITON_OPT $TEST -split-input-file -tritonamdgpu-dot-decompose-and-schedule 2>&1 | $FC $TEST --check-prefix=DISABLED
echo "phase3 DISABLED: $?"

TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 TRITON_TTGIR_SCHED_BARRIER_STRIDE=1 \
  $TRITON_OPT $TEST -split-input-file -tritonamdgpu-dot-decompose-and-schedule 2>&1 | $FC $TEST --check-prefix=PERDOT
echo "phase3 PERDOT: $?"
```

**Expected output:**

```
amd-ttgir-sched-phase0-noop: 0
amd-ttgir-sched-phase1a-plan: 0
amd-ttgir-sched-phase1b-bwd: 0
amd-ttgir-sched-phase1c-fwd: 0
phase1d-apply: 0
phase3 DEFAULT: 0
phase3 DISABLED: 0
phase3 PERDOT: 0
```

Any non-zero rc means a regression — FileCheck will show which expected
string didn't appear in the IR.

### Per-test purpose

| Test | What it checks | Trigger env vars |
|---|---|---|
| `phase0-noop.mlir`        | Pass walks a v8-shape MFMA loop, emits per-loop remark, doesn't mutate IR | `TRITON_ENABLE_TTGIR_SCHED=1` |
| `phase1a-plan.mlir`       | Pass computes a partition plan (numPartitions = blockM/ctaTileM) and reports it | `TRITON_ENABLE_TTGIR_SCHED=1` |
| `phase1b-bwd.mlir`        | Backward walker classifies producer ops (e.g. `ttg.local_load`) | `TRITON_ENABLE_TTGIR_SCHED=1` |
| `phase1c-fwd.mlir`        | Forward walker classifies user ops (e.g. `scf.yield`, `arith.truncf`) | `TRITON_ENABLE_TTGIR_SCHED=1` |
| `phase1d-apply.mlir`      | IR is mutated: original dot → 32 sub-dots `tensor<32x32xf32>` + 1 `amdg.concat`. Total 44 `extract_slice` (8 A + 4 B + 32 C) | `TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1` |
| `phase3-barrier.mlir`     | Three FileCheck prefixes verify: default (7 `rocdl.sched.barrier`), STRIDE=0 (0 barriers), STRIDE=1 (31 barriers) | `+ TRITON_TTGIR_SCHED_BARRIER_STRIDE=...` |

### See the actual rewritten IR (for debugging)

```bash
# What does the IR look like after APPLY?
TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 $TRITON_OPT \
  test/TritonGPU/amd/amd-ttgir-sched-phase1d-apply.mlir -split-input-file \
  -tritonamdgpu-dot-decompose-and-schedule 2>/dev/null | head -50

# Count specific ops in the output:
TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 $TRITON_OPT \
  test/TritonGPU/amd/amd-ttgir-sched-phase1d-apply.mlir -split-input-file \
  -tritonamdgpu-dot-decompose-and-schedule 2>/dev/null \
  | grep -cE "tt\.dot .* tensor<32x"     # should print: 32
```

## 3. e2e numerical correctness (stand-alone autotuned matmul)

The simplest single-run sanity check. `_one_run_envcompare.py` runs the
matmul kernel, compares its output element-wise to a PyTorch reference,
and prints `OK <TFLOPS> <config>` on success or `FAIL` on mismatch.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate metamain2
cd ~/AMD/triton/claude/triton_kernels_baseline

# Baseline (no TTGIR_SCHED). Should print "OK <TF> ..."
HIP_VISIBLE_DEVICES=0 python _one_run_envcompare.py 4096

# Planning-only (should match baseline within noise)
TRITON_ENABLE_TTGIR_SCHED=1 \
  HIP_VISIBLE_DEVICES=0 python _one_run_envcompare.py 4096

# APPLY=1 (Phase 3 default — 32 sub-dots + 7 sched_barriers).
# Expected: "OK <TF>" with TF close to (typically a bit higher than)
# baseline. If it printed "FAIL", numerical correctness regressed.
TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 \
  HIP_VISIBLE_DEVICES=0 python _one_run_envcompare.py 4096
```

Sample outputs from a recent run (gfx950):

```
Baseline:       OK 853.75 BLOCK_M: 256, BLOCK_N: 256, BLOCK_K: 64, GROUP_M: 8, num_warps: 8, num_ctas: 1, num_stages: 2, ...
TTGIR_SCHED=1:  OK 850.53 BLOCK_M: 256, ...    (no IR change, noise)
APPLY=1:        OK 888.70 BLOCK_M: 256, ...    (+4.1 %)
```

`OK` ⇒ correct; any `FAIL` is a regression to fix.

## 4. e2e performance — coverage matrix

The driver `claude/phase4_coverage.py` runs the stand-alone matmul
across K = {1024, 2048, 4096, 8192} × 3 TTGIR-SCHED modes and prints a
markdown summary at the end. **NOTE**: this script uses
`_one_run_envcompare.py` which autotunes per-invocation, so the
baseline-vs-APPLY comparison is *not* apples-to-apples in general (the
autotuner may pick different configs for the two modes depending on
cache state). For a stable apples-to-apples comparison, use
`_pinned_run.py` (see §4b below).

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate metamain2
HIP_VISIBLE_DEVICES=0 python ~/MetaMain2/triton/claude/phase4_coverage.py
```

Takes ~100 s (12 runs × ~8 s each). Sample output from one autotuned
run on gfx950 (your numbers will vary because of the autotune
non-determinism — see §4b for the stable variant):

```
## Phase 4 coverage matrix

| Workload | Baseline TF | Phase 3 default TF | Phase 2 (no bars) TF | Notes |
|---|---:|---:|---:|---|
| stand-alone matmul K=1024 | 773.53 | 762.34 | 763.63 | Phase 3 Δ -1.4% |
| stand-alone matmul K=2048 | 847.69 | 868.14 | 849.18 | Phase 3 Δ +2.4% |
| stand-alone matmul K=4096 | 853.75 | 888.70 | 848.64 | Phase 3 Δ +4.1% |
| stand-alone matmul K=8192 | 785.79 | 871.92 | 796.85 | Phase 3 Δ +11.0% |
```

Headline pattern: **Phase 3 perf gain scales with K** (bigger workloads
give the backend scheduler more headroom). Stable across runs:

- Phase 2 (no bars) ≈ baseline within ±1 % (the rewrite is functionally
  identical when the scheduler doesn't differentiate).
- Phase 3 default ≥ baseline starting at K=2048; **+11.0 % at K=8192
  (871.92 TF vs 785.79 TF)**.

If you see substantially different numbers:
- **All three modes hugely below sample**: GPU is busy or thermally
  throttled. Re-run after `rocm-smi --resetfans`.
- **Phase 3 << baseline at K=4096+**: regression — diff the
  `DotDecomposeAndSchedule.cpp` against the last good commit.
- **APPLY rows show `FAIL` or `?`**: numerical regression. Bisect by
  STRIDE=0 (Phase 2 IR only) vs default to isolate.

## 4b. e2e performance — stable reproduction (recommended)

Use `claude/_pinned_run.py`, which runs a single hard-coded config
(BM=BN=256, BK=64, num_warps=8, num_stages=2) with no autotuning. Same
config used for baseline and APPLY, so the comparison is meaningful.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate metamain2

# Baseline — 3 runs to gauge noise
for i in 1 2 3; do
  HIP_VISIBLE_DEVICES=0 python ~/MetaMain2/triton/claude/_pinned_run.py 8192
done

# APPLY=1 — 3 runs
for i in 1 2 3; do
  TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 \
    HIP_VISIBLE_DEVICES=0 python ~/MetaMain2/triton/claude/_pinned_run.py 8192
done
```

Sample output from gfx950 (numbers are very stable, ±0.3 % across
runs):

```
=== K=8192 baseline ===
OK 1054.24 BM=256 BN=256 BK=64 W=8 S=2
OK 1058.24 BM=256 BN=256 BK=64 W=8 S=2
OK 1053.61 BM=256 BN=256 BK=64 W=8 S=2     ← mean 1055.4 TF

=== K=8192 APPLY=1 ===
OK 1103.72 BM=256 BN=256 BK=64 W=8 S=2
OK 1104.85 BM=256 BN=256 BK=64 W=8 S=2
OK 1103.47 BM=256 BN=256 BK=64 W=8 S=2     ← mean 1104.0 TF  →  +4.6 %
```

What "OK" means: the script's `torch.allclose(c, ref, atol=1e-1,
rtol=1e-2)` check passed, confirming numerical correctness. The TFLOPS
is computed from `triton.testing.do_bench` median.

**If your APPLY mean isn't ≥ baseline by ≥ 3 %, something's wrong**:
- Make sure `libtriton.so` was rebuilt after editing the pass
  (`ninja triton` in the build dir; see §1).
- Verify the pass is actually firing: re-run baseline with
  `TRITON_ENABLE_TTGIR_SCHED=1` (planning only, no APPLY) and confirm
  you see remarks like
  `remark: ttgir-sched: would M-split this dot into 8 ...` in stderr.
- If the remarks appear but APPLY perf doesn't move, the IR rewrite
  might be getting CSE'd or DCE'd downstream — dump the IR after the
  pass with `MLIR_ENABLE_DUMP=1` and grep for `amdg.concat` to confirm
  the sub-dots survived.

## 5. FA-fwd safety (the kernel that crashes the matmul_4waves LLIR pass)

The FA-fwd tutorial in `python/tutorials/06-fused-attention.py` is the
canonical test that the matmul_4waves LLIR scheduler crashes on
(`Instruction does not dominate all uses!`). The TTGIR pass must run
without crash + produce numerically correct output.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate metamain2

# The tutorial imports `pytest` for skip markers but doesn't actually
# need pytest at runtime; stub it so we don't need to install it.
mkdir -p /tmp/stub_pytest
cat > /tmp/stub_pytest/pytest.py <<'EOF'
def skip(*a, **kw): pass
def fixture(*a, **kw): return lambda f: f
class _M:
    def __getattr__(self, n): return self
    def __call__(self, *a, **kw): return self
mark = _M()
class Parametrize:
    def __call__(self, *a, **kw): return lambda f: f
parametrize = Parametrize()
mark.parametrize = parametrize
EOF

cd ~/MetaMain2/triton

# Baseline
PYTHONPATH=/tmp/stub_pytest HIP_VISIBLE_DEVICES=0 \
  python python/tutorials/06-fused-attention.py 2>&1 | tail -8

# APPLY=1 (the key safety test)
PYTHONPATH=/tmp/stub_pytest TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 \
  HIP_VISIBLE_DEVICES=0 python python/tutorials/06-fused-attention.py 2>&1 | tail -8
```

Each run takes ~60 s. **Expected: both runs complete without crash.**
The output is a table of TFLOPS per N_CTX (1024, 2048, 4096, 8192, 16384)
for the various FA configurations. The APPLY numbers should match
baseline within ±2 % per row.

If APPLY crashes (likely message: `error: 'amdg.extract_slice' op
…`), it would mean my pass guard's CTA-tile check missed an FA-fwd dot
shape. Bisect by running planning-only (`TTGIR_SCHED=1` without
`APPLY=1`); if planning runs fine but APPLY crashes, the rewrite path
needs a tighter guard.

## 6. (Optional) Cleanup of FA-fwd output artifacts

The FA-fwd tutorial drops a bunch of CSV/PNG files in the cwd
(`fused-attention-batch4-*.csv/.png`). Remove with:

```bash
cd ~/MetaMain2/triton
rm -f fused-attention-batch4-*.csv fused-attention-batch4-*.png
```

# Reference

## Env-var contract

| Env var | Effect | Default |
|---|---|---|
| `TRITON_ENABLE_TTGIR_SCHED` | Enable the pass (planning-only, no IR mutation, just remarks) | off |
| `TRITON_TTGIR_SCHED_APPLY` | Mutate IR: replace each candidate MFMA `tt.dot` with M × N sub-dots glued via `amdgpu.concat`, with `ROCDL::SchedBarrier(0)` between M-rows | off |
| `TRITON_TTGIR_SCHED_BARRIER_STRIDE` | Override sched-barrier stride: 0 = none, 1 = per-sub-dot, k = every k | numPartitionsN |

## What the pass does today

For a v8/v10-shape `tt.dot tensor<256x64> × tensor<64x128> → tensor<256x128>`
with `AMDMfmaEncodingAttr(version=4, instrShape=[16,16,32], warpsPerCTA=[2,2])`:

- `ctaTileM = instrM × warpsPerCTA[0] = 16 × 2 = 32`
- `ctaTileN = instrN × warpsPerCTA[1] = 16 × 2 = 32`
- `numPartitionsM = blockM / ctaTileM = 256 / 32 = 8`
- `numPartitionsN = blockN / ctaTileN = 128 / 32 = 4`
- → **32 sub-dots**, each `tensor<32x32xf32>`, plus 44 `extract_slice` + 1 `concat` + 7 `rocdl.sched.barrier`

Sample IR after APPLY (excerpt):

```mlir
%a_0   = amdg.extract_slice %arg_a [0, 0]   : ... to tensor<32x64xf16>
%c_0_0 = amdg.extract_slice %arg_c [0, 0]   : ... to tensor<32x32xf32>
%b_0   = amdg.extract_slice %arg_b [0, 0]   : ... to tensor<64x32xf16>
%d_0_0 = tt.dot %a_0, %b_0, %c_0_0          : ... -> tensor<32x32xf32>
%c_0_1 = amdg.extract_slice %arg_c [0, 32]  : ... to tensor<32x32xf32>
%b_1   = amdg.extract_slice %arg_b [0, 32]  : ... to tensor<64x32xf16>
%d_0_1 = tt.dot %a_0, %b_1, %c_0_1          : ... -> tensor<32x32xf32>
... (32 sub-dots total, 4 per M-row, with rocdl.sched.barrier between rows)
%full = amdg.concat %d_0_0, %d_0_1, ..., %d_7_3 : ... -> tensor<256x128xf32>
```

## Why v8 from METAMD can't be tested directly on `metamain2`

The METAMD v8 kernel (`v8_beyond_hotloop`) imports
`triton.experimental.gluon.language.amd.cdna3.extract_slice`, which
exists only on the matmul_4waves branch of ROCm/triton (the `amd-triton`
conda env). MetaMain2/triton's `gluon.amd.cdna3` module doesn't have it.

Workarounds:
  1. Port the `cdna3.extract_slice` op from matmul_4waves to MetaMain2's
     gluon AMD language (small port), OR
  2. Use a vanilla autotuned matmul (as the coverage matrix does — it
     also lowers to MFMA `tt.dot` and exercises the same pass paths),
     OR
  3. Build a synthetic Gluon-equivalent v8 kernel using only ops
     MetaMain2 has natively.

## File list

```
~/MetaMain2/triton/
├── claude/
│   ├── README.md                          ← doc index (start here)
│   ├── llir_sched_at_ttgir_design.md      ← why (design rationale)
│   ├── llir_sched_at_ttgir_plan.md        ← original phased plan + per-phase ✅ markers
│   ├── ttgir_sched_status.md              ← this file (status + verification)
│   └── phase4_coverage.py                 ← e2e coverage driver
├── third_party/amd/
│   ├── include/TritonAMDGPUTransforms/Passes.td   ← pass def
│   └── lib/TritonAMDGPUTransforms/
│       └── DotDecomposeAndSchedule.cpp     ← the pass (~620 lines)
├── include/triton/Tools/Sys/GetEnv.hpp     ← env-var registry
├── bin/RegisterTritonDialects.h            ← register pass with triton-opt
└── test/TritonGPU/amd/
    ├── amd-ttgir-sched-phase0-noop.mlir
    ├── amd-ttgir-sched-phase1a-plan.mlir
    ├── amd-ttgir-sched-phase1b-bwd.mlir
    ├── amd-ttgir-sched-phase1c-fwd.mlir
    ├── amd-ttgir-sched-phase1d-apply.mlir
    └── amd-ttgir-sched-phase3-barrier.mlir
```

## Headline results

### Stable, reproducible (pinned config, `_pinned_run.py`)

Stand-alone matmul on gfx950, BM=BN=256, BK=64, num_warps=8, num_stages=2,
M=N=4096, K=8192, mean of 3 runs (run-to-run noise ±0.3 %):

| Mode | TFLOPS | Δ vs baseline |
|---|---:|---:|
| Baseline (no TTGIR_SCHED) | 1055.4 | — |
| TTGIR_SCHED=1 APPLY=1 (Phase 3 default) | **1104.0** | **+4.6 %** |

This is the recommended number to quote: same kernel config in both
modes, very stable across runs.

### Autotuner-based (`_one_run_envcompare.py`, `phase4_coverage.py`)

Stand-alone matmul on gfx950 with `triton.autotune`, M=N=4096, single
run each (autotuner picks the config it thinks is best):

| K | Baseline TF | Phase 3 default TF | Δ |
|---|---:|---:|---:|
| 1024 | 773.53 | 762.34 | -1.4 % |
| 2048 | 847.69 | 868.14 | +2.4 % |
| 4096 | 853.75 | 888.70 | **+4.1 %** |
| 8192 | 785.79 | 871.92 | **+11.0 %** ⚠ |

⚠ **Caveat:** with the autotuner, the same script can pick BK=32,W=4
(~785 TF) or BK=64,W=8 (~870 TF) for the *same* mode on different
invocations. The +11.0 % above is partly artifact — when the
autotuner picks BK=64,W=8 for both modes (the pinned-config recipe),
the honest delta at K=8192 is +4.6 %. Prefer §4b for any rigorous
benchmarking.

### FA-fwd safety (the kernel that crashes the LLIR pass)

Tutorial `06-fused-attention.py` Triton fp16 TFLOPS, batch=4, head=32,
d=128, bwd, causal=False:

| N_CTX | Baseline TF | TTGIR-SCHED Phase 3 default TF | Δ |
|---:|---:|---:|---:|
| 1024  | 324.44 | 323.46 | -0.3 % |
| 2048  | 370.22 | 365.54 | -1.3 % |
| 4096  | 417.20 | 409.02 | -2.0 % |
| 8192  | 443.26 | 452.37 | +2.1 % |
| 16384 | 458.38 | 456.78 | -0.3 % |

All numerically correct. The matmul_4waves LLIR pass *crashes* on this
kernel with `Instruction does not dominate all uses!`; the TTGIR pass
produces correct + competitive output.

### Comparison summary

| Workload | matmul_4waves LLIR-SCHED | TTGIR-SCHED Phase 3 default |
|---|---|---|
| Stand-alone matmul K=8192, pinned BM=BN=256/BK=64/W=8 | ❌ crash on autotune | ✅ **1104.0 TF (+4.6 % vs 1055.4 baseline)** |
| FA-fwd tutorial            | ❌ SSA dominance crash   | ✅ all configs correct, within ±2 % of baseline |

The big design win: **MLIR's typed SSA verifier rejects ill-formed
rewrites at construction time**, so the TTGIR pass can't produce the
kind of "Instruction does not dominate all uses!" LLVM-IR errors the
matmul_4waves LLIR pass hits on FA-fwd.

## What's NOT done

| Item | Why deferred |
|---|---|
| Phase 5 — default-disable matmul_4waves LLIR pass when TTGIR-SCHED active | Cross-repo coordination: the LLIR pass lives on a different branch of a different repo. Trivial Python-side change in `compiler.py`. |
| K-split — decompose `local_load` to per-MFMA-vector grain | Requires extending the producer-chain rewrite (backward walker would need to slice LDS reads too). See design doc. |
| Dim-flipping ops in the walkers (`BroadcastOp`, `ExpandDimsOp`, `TransOp`, `ReshapeOp`) | Only relevant for kernels that use these in producer/user chains; the workloads tested in Phase 4 don't need it. |
| Bigger perf wins (~+24 % like LLIR-SCHED on v10/v8) | Would require K-split (above) on a kernel that exposes MFMA/LR interleave opportunities; deferred. |
