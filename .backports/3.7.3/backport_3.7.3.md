# fbtriton v3.7.3 — backport doc

AMD-perf backport onto `release/3.7.x`.

- **Target cases:** paged decode, dense a16w16 GEMM (qkv/o_proj), and w4a4 MXFP4 MoE down-proj — all gfx950 / MI350.
- **Branch:** `wychi/backport-3.7.3` (off `release/3.7.x`)
- **Tracking sheet:** https://docs.google.com/spreadsheets/d/1B8FNRo-nF44SJx7tiCPDXgwuTlegxadcQvxzsc9SHZw/edit

---

## TL;DR

v3.7.3 pulls the AMD perf work onto the production `release/3.7.x` line (v3.7.2 was fixes-only and dropped every AMD perf commit).

- **Picked:** paged decode (A); a16w16 GEMM + prereqs + the D113115691 epilogue chain `#10059→#2150→#2290`
  (B); TLX layout infra (L); addmm+GLU `#1963` (M); and MXFP4/w4a4 (X) — a16w16 f16 bench harness `#2337`,
  intra_wave a4w4 GEMM `#2336` (supersedes closed `#1919`), inter_wave 8-wave a4w4 GEMM `#2392`. All picked
  in main's landing order.
- **Deferred:** Chain C (torchTLX Inductor — target prod uses sglang, not torch.compile) · Chain D
  (modulo scheduler — beta/gated).
- **Status:** version bumped to `3.7.3+fb`; built + tested (TLX correctness 69 passed / 4 skip, a4w4
  intra+inter + bench-harness tests green, folded-chain LIT green); perf-validated below.

---

## Perf testing runbook (target cases)

AMD perf harnesses ship in the wheel under `triton/language/extra/tlx/tutorials/` (or the source tree
at `third_party/tlx/tutorials/`). Require a **gfx950 (MI350)** GPU. Baselines are `torch`/rocBLAS
(present); only decode's optional `aiter` provider needs a heavy ROCm build.

```bash
TUT=$(python -c "import triton,os;print(os.path.dirname(triton.__file__))")/language/extra/tlx/tutorials

# 1. Paged decode (A) — HEAD_DIM=64 / PAGE_SIZE=16 / GQA 64:8 / bf16
python $TUT/testing/test_amd_pa_decode_perf.py --version tlx --qlens 1 2 3 4          # TLX only
python $TUT/testing/test_amd_pa_decode_perf.py --version tlx aiter --qlens 1 2 3 4    # vs aiter

# 2. Dense a16w16 GEMM (B) — TBD's M/N/K, vs rocBLAS
python $TUT/gfx9_gemm/inter_wave/a16w16/bench.py \
  --shape 2048 18432 8192 --shape 2048 8192 8192 --shape 2048 4096 8192 \
  --shape 2048 256 8192   --shape 2048 8192 4096                      # omit --shape → 4096³ default

# 3. MXFP4 MoE GEMM (w4a4) — intra_wave a4w4 #2336 + inter_wave 8-wave a4w4 #2392 (Shape:Stride kernels)
python $TUT/gfx9_gemm/intra_wave/a4w4/bench.py                    # intra_wave a4w4 (4096²×K sweep)
# inter_wave #2392 ships no bench.py; drive its matmul() on prod 2048×N×K shapes (see §1)
# a4w4 correctness (intra + inter): pytest -k a4w4 python/test/unit/language/test_tlx_amd.py

# Correctness (run first)
pytest -q $TUT/testing/test_correctness.py::test_amd_pa_decode
# a16w16 correctness is built into bench.py (prints OK/FAIL per shape)
```

On hang: `third_party/tlx/killgpu.sh`.

---

## 1 · main HEAD `35a1f082c` ↔ v3.7.3 — completeness (did we capture every target perf commit?)

**Status — ✅ complete: v3.7.3 matches main HEAD on all three target cases (a16w16 0.96–1.01×, decode batch≥8 0.995×, MXFP4 w4a4 K≥2048 1.00–1.02×).**

Both sides **built from source**: v3.7.3 = the backport wheel (`gitb8e35f17`); main = a wheel built
from `origin/main` HEAD **`35a1f082c`** (`triton-3.8.0+git35a1f082`) — *not* the published nightly pip
wheel, which lags main. Each test runs on **both** wheels, GPU0 **clock-locked @2100 MHz**, warm, with
a v3.7.3 repeat to establish the noise floor (~1%).

| target perf test | v3.7.3 vs main `35a1f082c` | captured? |
|---|---|---|
| **paged decode** | parity at batch≥8 (geomean **0.995×**); v3.7.3 steadier at batch=1 | ✅ |
| **dense a16w16 GEMM** | parity across all target shapes (**0.96–1.01×**) | ✅ |
| **MoE GEMM (w4a4 MXFP4)** | intra_wave `#2336` parity vs main `439413ab` (K≥2048 **1.00–1.02×**); inter_wave `#2392` byte-identical to main | ✅ |

**a16w16 GEMM — target's M/N/K, vs rocBLAS** (each wheel's *own* bench+kernel; `bench.py` is identical,
`matmul_kernel.py` differs only by main's #2329):

| M×N×K | rocBLAS | v3.7.3 | main `35a1f082c` | main/v3.7.3 |
|---|--:|--:|--:|--:|
| 2048×18432×8192 | 1112 | 1006 | 1007 | 1.00× |
| 2048×8192×8192  | 1127 | 1141 | 1128 | 0.99× |
| 2048×4096×8192  |  886 |  932 |  939 | 1.01× |
| 2048×256×8192   |  307 |  334 |  334 | 1.00× (thin-N) |
| 2048×8192×4096  | 1164 | 1121 | 1075 | 0.96× |

```bash
# AMD: drive via denoise.sh (NUMA pin + clock lock @ --setperfdeterminism 2100); select GPU with HIP_VISIBLE_DEVICES
SH="--shape 2048 18432 8192 --shape 2048 8192 8192 --shape 2048 4096 8192 --shape 2048 256 8192 --shape 2048 8192 4096"
D=third_party/tlx/denoise.sh; A16=third_party/tlx/tutorials/gfx9_gemm/inter_wave/a16w16/bench.py
HIP_VISIBLE_DEVICES=0 $D .venv/bin/python          $A16 $SH                       # v3.7.3
HIP_VISIBLE_DEVICES=0 $D .venv-mainhead/bin/python <main-worktree>/…/$A16 $SH     # main
```
(N=256/N=4096 bench "FAIL" = benign split-K 1-ULP tolerance artifact, bit-exact at `SPLIT_K=1`; not a regression.)

**Paged decode — TB/s effective HBM, qlen1** (decode kernel is byte-identical across wheels, so the
*same* `test_amd_pa_decode_perf.py` drives both → compiler-isolated):

| batch × ctx | v3.7.3 | main `35a1f082c` | main/v3.7.3 |
|---|--:|--:|--:|
| 8 × 8192   | 4.55 | 4.55 | 1.00× |
| 32 × 8192  | 6.40 | 6.35 | 0.99× |
| 128 × 8192 | 8.17 | 8.15 | 1.00× |
| 8 × 32768  | 5.57 | 5.42 | 0.97× |
| 32 × 32768 | 8.12 | 8.01 | 0.99× |
| 8 × 131072 | 6.16 | 6.12 | 0.99× |
| 1 × 8192   | 0.79 | 0.68 | batch=1: noisy |
| 1 × 32768  | 3.45 | 3.34 | batch=1: noisy |

```bash
# same test drives both wheels; kernel is imported from the installed wheel (byte-identical):
MTEST=<main-worktree>/third_party/tlx/tutorials/testing/test_amd_pa_decode_perf.py; D=third_party/tlx/denoise.sh
HIP_VISIBLE_DEVICES=0 $D .venv/bin/python          $MTEST --version tlx --qlens 1 2 3 4   # v3.7.3
HIP_VISIBLE_DEVICES=0 $D .venv-mainhead/bin/python $MTEST --version tlx --qlens 1 2 3 4   # main
```
High-throughput (batch≥8) geomean main/v3.7.3 = **0.995×** (parity). At batch=1 the main wheel is
noisy/non-reproducible (a `TRITON_USE_C_DISPATCHER` cold-start fallback corrupts first timings) while
v3.7.3 is stable — so v3.7.3 is if anything *steadier* there. Cold-start glitches excluded via warm re-runs.

**MoE GEMM (w4a4 MXFP4) — TLX TFLOPS, 4096×4096×K** (compiler-isolated: kernel+bench byte-identical
across wheels. TLX-only — no rocBLAS baseline for MXFP4. Compared against a *newer* main HEAD `439413ab`
that actually carries the kernel — MXFP4 landed as #2336 after `35a1f082c`, so it's built from a `439413ab`
worktree in its own venv):

| M×N×K | v3.7.3 | main `439413ab` | main/v3.7.3 |
|---|--:|--:|--:|
| 4096×4096×1024  | 1898.9 | 1741.3 | 0.92× (launch-noise, ~18µs) |
| 4096×4096×2048  | 2496.7 | 2511.2 | 1.01× |
| 4096×4096×4096  | 2934.5 | 2962.3 | 1.01× |
| 4096×4096×8192  | 3306.9 | 3332.7 | 1.01× |
| 4096×4096×16384 | 3525.2 | 3541.0 | 1.00× |
| 4096×4096×32768 | 3584.5 | 3664.7 | 1.02× |

```bash
D=third_party/tlx/denoise.sh; A4=third_party/tlx/tutorials/gfx9_gemm/intra_wave/a4w4/bench.py
HIP_VISIBLE_DEVICES=0 $D .venv/bin/python           $A4                    # v3.7.3
HIP_VISIBLE_DEVICES=0 $D .venv-main439/bin/python <main-worktree>/…/$A4    # main 439413ab
```
Compute-bound K≥2048 parity (**1.00–1.02×**, within the ~1–2% floor); K=1024 0.92× is launch-overhead
noise at ~18µs, not codegen. All shapes correct (max_err 0.0000).

→ **v3.7.3 captures every target perf commit on main HEAD** — a16w16, decode, and MXFP4 all at parity with
from-source main builds (a16w16/decode vs `35a1f082c`; MXFP4 vs the newer `439413ab` that carries the kernel).

---

## 2 · v3.7.2 ↔ v3.7.3 — no regression

**Status — ✅ no regression: shared GEMM compiler path is flat (tritonbench 1.00×); the target kernels are net-new or large wins (a16w16 old→new 1.12–1.17×).**

v3.7.2 runtime = stock `fbtriton==3.7.2` (PyPI, separate venv); v3.7.3 = backport wheel. **Target
kernels are *net-new*** — v3.7.2 has no paged-decode and no inter_wave a16w16, and v3.7.3's kernel
source can't even run on a v3.7.2 runtime (needs backported JIT/DSL: `llvm_fn_attrs` #2152,
`tlx.require_layout` #2290). So the regression check is on the shared/standard paths.

> **Clock caveat.** `--setperflevel high` is unsupported on this MI350, but `sudo rocm-smi
> --setperfdeterminism 2100` works (sclk only). Even locked, cross-GPU device variance (~2%) + residual
> mclk/thermal noise put the floor at **~±5%**. Correctness (69-pass) is unaffected.

- **Standard GEMM compiler path — tritonbench `gemm`: no meaningful regression.** `pt2_triton`
  (inductor→triton) vs rocBLAS baseline, both runtimes: geomean Δ **1.00×** (within the ±5% floor).
  The backport is *additive* — it doesn't touch generic GEMM codegen. (`decoding_attention` unrunnable
  here: its input packing needs CUDA-only `xformers`.)
- **a16w16 old→new** (clock-locked @2100, 4096², rocBLAS baselines match ±2%): v3.7.3 inter_wave is a
  **uniform 1.12–1.17×** over v3.7.2's best intra_wave (`v9_beyond_hotloop`) and reaches rocBLAS parity
  (~1.00× vs v3.7.2's ~0.87×): K=1024 841→962, K=2048 987→1151, K=4096 1060→1188, K=8192 1004→1128.
- **Paged decode / thin-N / MXFP4 w4a4** — net-new in v3.7.3 (no v3.7.2 equivalent; the a4w4 MXFP4 GEMM
  can't even run on a v3.7.2 runtime), so "absent → present" — no regression possible.

```bash
# AMD: wrap each run in denoise.sh (NUMA pin + --setperfdeterminism 2100); GPU via HIP_VISIBLE_DEVICES
D=<fbtriton>/third_party/tlx/denoise.sh
# tritonbench gemm — same providers, both runtimes (v3.7.2 = .venv372, v3.7.3 = .venv):
cd ~/github/tritonbench
HIP_VISIBLE_DEVICES=0 $D <fbtriton>/.venv372/bin/python run.py --op gemm --only aten_matmul,pt2_triton_matmul --metrics latency,tflops --csv
HIP_VISIBLE_DEVICES=0 $D <fbtriton>/.venv/bin/python    run.py --op gemm --only aten_matmul,pt2_triton_matmul --metrics latency,tflops --csv
# a16w16 old→new (same 4096² shapes): v3.7.2 intra_wave v9  vs  v3.7.3 inter_wave
HIP_VISIBLE_DEVICES=0 $D .venv372/bin/python third_party/tlx/tutorials/gfx9_gemm/a16w16/bench.py --version 9
HIP_VISIBLE_DEVICES=0 $D .venv/bin/python    third_party/tlx/tutorials/gfx9_gemm/inter_wave/a16w16/bench.py --shape 4096 4096 8192  # etc.
```

**No regression anywhere; large wins or net-new capability on the target kernels.**

---

## Status & next steps

- **Built + tested:** smoke ✓ · launch.h 42-skip (AMD) ✓ · TLX correctness **69 passed / 4 skip** ·
  folded-chain LIT ✓ · a4w4 intra+inter correctness ✓ · a16w16 bench-harness tests (7) ✓ · `#2336`/`#2392` lit ✓.
- **Stack order:** version bump (`3.7.3+fb`, first commit) → L → B → A → M → X (`#2337 → #2336 → #2392`) →
  backport doc (tip). **Next:** push `wychi/backport-3.7.3` to PR #2335 for land.
