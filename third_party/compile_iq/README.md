# compile_iq — zero-surface, PTX-direct ptxas-ACF tuning for Triton

`compile_iq` tunes **ptxas Advanced Control Files (ACFs)** for Triton kernels with **no user-code
surface**: a normal `@triton.jit` kernel (see `examples/user_kernel.py`) gets faster SASS without
importing or referencing anything from this package. The system is **PTX-direct** — it tunes by
assembling a *fixed* `kernel.ptx` with ptxas and launching the cubin via the CUDA driver API, and it
**never recompiles the kernel from source**. Three stages communicate only through disk:

| Stage | What | Trigger | Requirements |
|-------|------|---------|--------------|
| **1. collect** | a gated hook in `jit.py` dumps a **source-free** task (`kernel.ptx` + `spec.json`) per kernel | `FBTRITON_COMPILE_IQ_COLLECT=1` | none (no `source.py`, no cuda-python) |
| **2. factory** | offline search over ACFs scored by `ptxas(fixed PTX)` + driver launch; keep the best → ACF store | EVO route under `smoke/` (`run_e2e_cuda12{8,30}.sh`) | EVO engine + ptxas + driver matching the tier |
| **3. consume** | a gated `make_cubin` hook appends `--apply-controls=<stored.acf>` | `FBTRITON_COMPILE_IQ_APPLY=1` | ptxas matching the ACF version |

Content-addressed: the store key is `sha256(normalized PTX) × arch`, so one ACF transfers across
runtime shapes (M,N,K) of the same kernel.

## What each stage captures / does

- **collect** (`collector.py`) writes only `kernel.ptx` + `spec.json` — the source-free launch
  description (entry name, arch, `shared`, block/grid, and the post-specialization kernel-param
  layout; see `ptx_launch.build_spec`). No `source.py` is dumped, because nothing downstream
  recompiles. Kernels the spec can't yet express (non-null scratch, multi-CTA/cluster, tensordesc
  args) are skipped fail-open — the user's run is never affected.
- **factory** never runs Triton: it assembles the fixed PTX with `ptxas [--apply-controls=<acf>]`
  and launches via the driver (`ptx_launch.py`). The reference implementation is the EVO route in
  `smoke/` (`ptx_evo_search.py` + `ptx_bench_one.py`), driven by `smoke/run_e2e_cuda12{8,30}.sh`.
- **consume** (`consume.py`, hook in `nvidia/backend/compiler.py:make_cubin`) re-assembles with the
  stored ACF. ⚠️ The applied ACF is invisible to Triton's compile cache, so consume currently needs
  `TRITON_ALWAYS_COMPILE=1` (see the TODO in `consume.py`).

## Version matching (required, two axes)

1. A `cuda-X.Y` ACF is only accepted by **ptxas X.Y** (a mismatched ptxas rejects it: "Invalid
   compiler controls file").
2. The **PTX ISA `.version`** is chosen by Triton from the ptxas seen at compile time, and a CUDA-X.Y
   driver cannot run a newer-ISA cubin. So the ptxas used for collect **and** factory/consume must
   match the tier version **and** the installed driver.

The `smoke/run_e2e_cuda12{8,30}.sh` wrappers pin a matching set (ptxas 12.8 / 13.0 + tier).

## End-to-end

See `smoke/README.md`. Quick collect demo (source-free task, any matching ptxas):

```bash
FBTRITON_COMPILE_IQ_COLLECT=1 FBTRITON_COMPILE_IQ_DEBUG=1 COMPILE_IQ_TASK_DIR=/tmp/ciq_tasks \
  TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.0/bin/ptxas \
  python third_party/compile_iq/examples/user_kernel.py
# -> /tmp/ciq_tasks/<sha16>/{kernel.ptx, spec.json}   (no source.py)
```

Full tune+apply on the naive matmul:

```bash
PTXAS_KNOBS=/data/users/$USER/ptxas_knobs bash third_party/compile_iq/smoke/run_e2e_cuda130.sh
```

## Environment variables

| Var | Purpose |
|-----|---------|
| `FBTRITON_COMPILE_IQ_COLLECT` | enable Stage 1 (collect) |
| `FBTRITON_COMPILE_IQ_APPLY` | enable Stage 3 (consume) |
| `FBTRITON_COMPILE_IQ_DEBUG` | trace collect / lookup hit-miss |
| `COMPILE_IQ_TASK_DIR` | where collect writes tasks (default `~/.compile_iq/tasks`) |
| `COMPILE_IQ_STORE` | ACF store root (default `~/.compile_iq/store`) |
| `COMPILE_IQ_PTXAS_MIN_VERSION` | consume's minimum ptxas version (default 13.3; lower it to match a 12.8/13.0 tier) |
| `TRITON_PTXAS_BLACKWELL_PATH` | ptxas to use (must match the tier + driver) |

Paths are illustrative — keep them in your env, **never** hard-code them in source.

## Mechanics

- **Correctness** is checked by *self-consistency* — an ACF must reproduce the no-ACF output
  (rel ≤ 1e-2) — so it is op-agnostic. A diverging/erroring candidate scores INVALID.
- **Isolation**: each candidate is benchmarked in a throwaway spawn subprocess with a timeout, so an
  ACF that wedges the GPU is killed and scored INVALID rather than hanging the search.
- **Search**: the EVO engine over a constrained ptxas `.config` tier; candidates are scored in the
  base env through the PTX-direct driver-launch bridge (the EVO env has no torch/triton).

## Scope

Naive (non-warp-specialized) kernels with null global/profile scratch. Warp-specialized / TMA
kernels (non-zero scratch, tensordesc args) extend `build_spec` and are the next PR in the stack.
"""
