# compile_iq — zero-surface ptxas-ACF tuning for Triton

`compile_iq` tunes **ptxas Advanced Control Files (ACFs)** for Triton kernels with
**no user-code surface**: a normal `@triton.jit` kernel (see `examples/user_kernel.py`)
gets faster SASS without importing or referencing anything from this package. The whole
system is decoupled into three stages that communicate only through disk:

| Stage | What | Trigger | Requirements |
|-------|------|---------|--------------|
| **1. collect** | a gated hook in `jit.py` dumps a *compileIQ task* (source + args + grid + PTX) per kernel | `FBTRITON_COMPILE_IQ_COLLECT=1` | none — stdlib only, any box |
| **2. factory** | offline search over ACFs (replay each candidate, keep the best) → ACF store | `python -m triton.compile_iq.factory <task_dir>` | CompileIQ engine + ptxas ≥ 13.3 + search-space bin |
| **3. consume** | a gated `make_cubin` hook re-assembles with the stored ACF (`--apply-controls`) | `FBTRITON_COMPILE_IQ_APPLY=1` | ptxas ≥ 13.3 |

Content-addressed: the store key is `sha256(normalized PTX) × arch`, so one ACF transfers
across runtime shapes (M,N,K) of the same kernel.

## The requirement: ptxas ≥ 13.3

`--apply-controls` (the ACF flag) is only understood by **ptxas 13.3+**, so both the factory
and the consume hook need a 13.3+ ptxas (auto-discovered from the `nvidia-cuda-nvcc` wheel, or
set `TRITON_PTXAS_BLACKWELL_PATH`). The consume hook is version-guarded and fail-open: with an
older ptxas it silently skips the ACF and compiles unchanged, so nothing breaks.

There is **no GPU-driver requirement** beyond what the kernel itself needs — by CUDA
minor-version compatibility, a 13.3-assembled cubin runs on any driver of the same major
(e.g. CUDA 13.0). The per-candidate subprocess isolation exists to survive a genuinely bad ACF
(illegal access / divergent scheduling), not a driver mismatch. **collect** has no deps at all.

## Factory environment (uv)

On a box with ptxas ≥ 13.3 (or use the `nvidia-cuda-nvcc` wheel below):

```bash
uv venv --python 3.13 .venv-ciq
uv pip install -r third_party/compile_iq/requirements.txt
# torch matching the fbtriton build (cu130):
uv pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
# proprietary CompileIQ "Evo" engine (NOT on PyPI) — from a CompileIQ checkout:
#   git lfs install && git lfs pull && uv pip install .
# or an internal Evo wheel:
#   uv pip install <compileiq-evo-wheel>
```

### Environment variables

| Var | Purpose | Example (machine-specific) |
|-----|---------|----------------------------|
| `TRITON_PTXAS_BLACKWELL_PATH` | 13.3+ ptxas (else auto-discovered from the `nvidia-cuda-nvcc` wheel / PATH) | `.venv-ciq/lib/python3.13/site-packages/nvidia/cu13/bin/ptxas` |
| `COMPILE_IQ_SEARCH_SPACE_BIN` | ptxas13.3 search-space `.bin` (separate NVIDIA artifact) | `/data/users/daohang/CompileIQ/tests/compiler_ss/data/ptxas13.3.bin` |
| `COMPILE_IQ_TASK_DIR` | where collect writes tasks | `/tmp/ciq_tasks` |
| `COMPILE_IQ_STORE` | ACF store root (default `~/.compile_iq/store`) | `~/.compile_iq/store` |
| `FBTRITON_COMPILE_IQ_COLLECT` | enable Stage 1 | `1` |
| `FBTRITON_COMPILE_IQ_APPLY` | enable Stage 3 | `1` |
| `FBTRITON_COMPILE_IQ_DEBUG` | trace collect/lookup hit-miss | `1` |

Paths above are illustrative — keep them in your env, **never** hard-code them in source.

## End-to-end run (naive matmul)

Use the interpreter that has torch + fbtriton + the `compileiq` engine
(here: `/data/users/daohang/miniconda3/bin/python`, referred to as `$PY`), and set
`$PTXAS` to a 13.3+ ptxas.

```bash
# 1. collect  (any box)
FBTRITON_COMPILE_IQ_COLLECT=1 COMPILE_IQ_TASK_DIR=/tmp/ciq_tasks FBTRITON_COMPILE_IQ_DEBUG=1 \
  TRITON_PTXAS_BLACKWELL_PATH=$PTXAS \
  $PY third_party/compile_iq/examples/user_kernel.py
# -> /tmp/ciq_tasks/<sha16>/{kernel.ptx,task.json,source.py}

# 2. factory  (needs the CompileIQ engine + ptxas 13.3 + search-space bin)
COMPILE_IQ_SEARCH_SPACE_BIN=/path/to/ptxas13.3.bin TRITON_PTXAS_BLACKWELL_PATH=$PTXAS \
  $PY -m triton.compile_iq.factory /tmp/ciq_tasks/<sha16>
# -> "[factory] wrote ACF (+X%)" -> ~/.compile_iq/store/<arch>/<sha>.acf

# 3. consume  (TRITON_ALWAYS_COMPILE=1 forces re-assembly so the ACF is actually applied;
#              otherwise step-1's cached no-ACF cubin is reused and the hook never re-runs ptxas)
FBTRITON_COMPILE_IQ_APPLY=1 FBTRITON_COMPILE_IQ_DEBUG=1 TRITON_ALWAYS_COMPILE=1 \
  TRITON_PTXAS_BLACKWELL_PATH=$PTXAS \
  $PY third_party/compile_iq/examples/user_kernel.py
# -> "[compile_iq.consume] HIT <sha16> <arch>" and runtime <= baseline
```

**Definition of done:** collect produces a task; factory stores an ACF (when one beats the
no-ACF baseline); consume logs a HIT with a speedup. An empty store is a clean no-op — consume
MISSes and compiles the unchanged baseline.

## How it works (mechanics)

- **Correctness** is checked by *self-consistency* — an ACF must reproduce the no-ACF output
  (rel ≤ 1e-2) — so it's op-agnostic (the no-ACF run is the reference).
- **Per-candidate isolation**: each ACF is benchmarked in its own spawn subprocess. A
  candidate that wedges (timeout), crashes (IMA), or diverges scores INVALID and the search
  keeps moving — the factory never hangs.
- **ACF application**: the per-launch `ptx_options="--apply-controls=<file>"` kwarg in the
  factory/replay; the consume hook injects the same flag at `make_cubin`.
- **Search**: `LocalSearchSpaceBin(COMPILE_IQ_SEARCH_SPACE_BIN)` (offline; no network),
  benchmarked under `gpu_benchmark_mode(clock_mhz=1965)` for stable measurements.

## Scope

Naive (non-warp-specialized) kernels only. Warp-specialized / TMA kernels and a co-tuned
Triton+PTX *mixed* search space (bigger wins, but needs disabling autotune = user surface)
are deferred to keep this stage zero-surface.
