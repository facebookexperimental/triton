---
name: tlx-amd-testing
description: >
  Test and run TLX-AMD tutorial kernels (gfx950/CDNA4 and gfx1250) and
  understand their CI. Use when working on AMD TLX tutorial kernels — GEMM
  (warp-pipeline, LDS-pipelined, TDM, MXFP), Flash Attention (simple, prefetch,
  persistent), addmm+GLU, or IKBO (FA, LCE) — running their correctness or perf,
  checking arch gating (gfx950 vs gfx1250), or the MI350 CI workflow. Covers the
  standardized layout (one correctness file, one perf file per op×arch).
---

# TLX-AMD Tutorial Kernel Testing

AMD tutorial kernels follow the same standardized layout as the NVIDIA
(Hopper/Blackwell) reference: **one shared correctness file**
(`test_correctness.py`, arch-gated) and **one perf script per (op, arch)**.
Each kernel is an importable module under `third_party/tlx/tutorials/`.

## Kernel inventory

| Kernel module | Op | Correctness test(s) | Perf script | Arch gate |
|---|---|---|---|---|
| `amd_gemm_warp_pipeline.py` | GEMM | `test_amd_gemm_warp_pipeline` | `test_amd_gemm_perf.py` (`warp_pipeline`) | `is_hip_cdna4` |
| `amd_gemm_pipelined.py` | GEMM (LDS pipeline) | `test_amd_gemm_pipelined` | `test_amd_gemm_perf.py` (`pipelined`) | `is_hip` |
| `amd_fa_pipelined.py` | Flash Attention | `test_amd_fa_pipelined` | `test_amd_fa_perf.py` (`simple`, `prefetch`) | `is_hip_cdna4` |
| `amd_fa_persistent.py` | Flash Attention (persistent) | `test_amd_fa_persistent`, `test_amd_fa_persistent_cross_attention` | `test_amd_fa_perf.py` (`persistent`) | `is_hip_cdna4` |
| `amd_addmm_glu.py` | addmm + GLU (gated linear unit, **not** GELU) | `test_amd_addmm_glu` | `test_amd_addmm_glu_perf.py` | `is_hip_cdna4` |
| `ikbo/ikbo_fa_triton.py` | IKBO Flash Attention | `test_ikbo_fa` | `test_amd_ikbo_fa_perf.py` | none (any HIP/CUDA) |
| `ikbo/ikbo_lce_triton.py` | IKBO LCE (logit cross-entropy — **not** attention) | `test_ikbo_lce` | `test_amd_ikbo_lce_perf.py` | none (any HIP/CUDA) |
| `amd_tdm_gemm_pipelined.py` | GEMM (TDM) | `test_amd_tdm_gemm_pipelined` | — | `is_hip_gfx1250` |
| `amd_mxfp_gemm_tdm_pipelined.py` | GEMM (MXFP, TDM) | `test_amd_mxfp_gemm_tdm_pipelined` | `test_amd_mxfp_gemm_perf.py` | `is_hip_gfx1250` |

`gfx950` = CDNA4 = MI350-class (`is_hip_cdna4()`). `gfx1250` is a separate, newer
target (`is_hip_gfx1250()`). On gfx950, the gfx1250-only GEMM tests auto-skip.

## Correctness

All AMD correctness lives in the single shared file; tests self-gate via
`@pytest.mark.skipif`, so only the relevant cases run per GPU.

```bash
# All AMD + IKBO (gfx1250-only cases auto-skip on gfx950):
pytest third_party/tlx/tutorials/testing/test_correctness.py -v -k "amd or ikbo"

# Whole file — Hopper/Blackwell cases auto-skip on AMD (what CI runs, no -k):
pytest third_party/tlx/tutorials/testing/test_correctness.py -v
```

`-k "amd"` alone does **not** select the IKBO tests (`test_ikbo_*` has no "amd" in
its node id) — use `-k "amd or ikbo"`.

## Perf

Never run perf unless explicitly asked. Use the `kernel-perf-testing` skill for
run mechanics. `denoise.sh` works on AMD (it runs the command with NUMA pinning);
its GPU clock/power lock is `nvidia-smi`-based and is skipped on AMD, so expect
slightly higher run-to-run variance. Pick a free GPU with `rocm-smi`.

## CI

`.github/workflows/mi350.yml` runs on a gfx950 (MI350/CDNA4) runner and mirrors
`.github/workflows/h100.yml`:

- **`mi350-tlx-test`** — TLX unit tests (`python/test/unit/language/test_tlx_*.py`)
  + the tutorial correctness suite (`test_correctness.py`). AMD/IKBO run;
  Hopper/Blackwell and gfx1250 cases auto-skip.
- **`mi350-meta-triton-test`** — TritonBench perf coverage (perf-regression lives
  here, not in the perf scripts above).

Nightly failures are filed as issues via `report-nightly-failure.yml`.

## Local run note

After any C++ change (or a stale checkout), the in-tree `libtriton.so` can lag
the Python source and every AMD kernel fails at compile with
`AttributeError: module '...amd.passes.ttgpuir' has no attribute '<pass>'`.
Fix: rebuild with `make dev-install-llvm`. If GPU tests hang, run
`third_party/tlx/killgpu.sh`.
