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
| `amd_gemm_gfx942.py` | GEMM (MI300X, autotuned) | `test_amd_gemm_gfx942`, `test_amd_gemm_gfx942_odd_shapes` | `test_amd_gemm_gfx942_perf.py` | `is_hip_cdna3` |
| `amd_addmm_gfx942.py` | addmm (MI300X, autotuned) | `test_amd_addmm_gfx942` | `test_amd_addmm_gfx942_perf.py` | `is_hip_cdna3` |
| `amd_bmm_gfx942.py` | BMM (MI300X, autotuned) | `test_amd_bmm_gfx942`, `test_amd_bmm_gfx942_distinct_a` | `test_amd_bmm_gfx942_perf.py` | `is_hip_cdna3` |
| `amd_fa_pipelined.py` | Flash Attention | `test_amd_fa_pipelined` | `test_amd_fa_perf.py` (`simple`, `prefetch`) | `is_hip_cdna4` |
| `amd_fa_persistent.py` | Flash Attention (persistent) | `test_amd_fa_persistent`, `test_amd_fa_persistent_cross_attention` | `test_amd_fa_perf.py` (`persistent`) | `is_hip_cdna4` |
| `amd_addmm_glu.py` | addmm + GLU (gated linear unit, **not** GELU) | `test_amd_addmm_glu` | `test_amd_addmm_glu_perf.py` | `is_hip_cdna4` |
| `ikbo/ikbo_fa_triton.py` | IKBO Flash Attention | `test_ikbo_fa` | `test_amd_ikbo_fa_perf.py` | none (any HIP/CUDA) |
| `ikbo/ikbo_lce_triton.py` | IKBO LCE (logit cross-entropy — **not** attention) | `test_ikbo_lce` | `test_amd_ikbo_lce_perf.py` | none (any HIP/CUDA) |
| `amd_tdm_gemm_pipelined.py` | GEMM (TDM) | `test_amd_tdm_gemm_pipelined` | — | `is_hip_gfx1250` |
| `amd_mxfp_gemm_tdm_pipelined.py` | GEMM (MXFP, TDM) | `test_amd_mxfp_gemm_tdm_pipelined` | `test_amd_mxfp_gemm_perf.py` | `is_hip_gfx1250` |

`gfx950` = CDNA4 = MI350-class (`is_hip_cdna4()`). `gfx942` = CDNA3 = MI300X-class
(`is_hip_cdna3()`). `gfx1250` is a separate, newer target (`is_hip_gfx1250()`). On
gfx950, both the gfx1250-only and the gfx942-only GEMM tests auto-skip.

The MI300X kernel is the only gfx942 entry and has **no CI runner** — the MI350
workflow is gfx950, so `test_amd_gemm_gfx942*` always skips there. Run it by hand
on an MI300X box.

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
run mechanics. `denoise.sh` **does** lock clocks on AMD: it identifies the part by
PCI id (MI300X `0x74a0`/`0x74a1`/gfx942, MI350X, MI355X), then applies
`rocm-smi --setperfdeterminism` (default 2100 MHz, override `DETERMINISM_CLK`) and
`--setpoweroverdrive` (750 W on MI300X, override `DESIRED_POWER`), NUMA-binds to
the GPU's node, and resets both on exit. It needs sudo and is best-effort. It
defaults `HIP_VISIBLE_DEVICES` to 4 — set it to a free GPU (`rocm-smi`) yourself.

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
