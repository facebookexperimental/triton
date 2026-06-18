# compile_iq smoke tests

Two smoke tests for the compile_iq ptxas-ACF tuning pipeline. Both run on a single B200.

## Engines & search spaces (important)

The optimizer ships in two forms: the older **CIQ** engine (`compileiq`, base Python 3.13)
and the newer **EVO** engine (`evo_solar`/`evo_nda`, in the `evo` conda env, Python 3.10).
They take *different* search-space artifacts:

| Engine | Search space | Notes |
|--------|--------------|-------|
| CIQ    | `ptxas13.3.bin` (binary blob) | full/aggressive space; random candidates wedge the GPU |
| EVO    | `cuda-<ver>-ptxas-p{0,1,2}.config` | constrained tiers (p0 safest); CIQ can't parse these |

**Version matching is required:** a `cuda-X.Y` space yields ACFs that only ptxas **X.Y**
accepts (a mismatched ptxas rejects them with "Invalid compiler controls file"). The
EVO route below pairs the `cuda-13.0` tier with `/usr/local/cuda-13.0/bin/ptxas`.

The EVO env has no triton/torch, so it cannot benchmark a Triton kernel itself; each
candidate is benchmarked in base 3.13 via the `bench_one.py` subprocess bridge.

## Setup (internal): fetch artifacts -- NOT checked in

The EVO wheel and the search-space `.config` files are NVIDIA-provided binaries (the
EVO wheel is under NDA) and are deliberately **not committed**. Internal users fetch
them from Manifold:

    manifold getr tc_bench_ci/tree/compileiq/ptxas_knobs /data/users/$USER/ptxas_knobs
    # getr nests one level -- flatten so the files sit directly in $PTXAS_KNOBS:
    mv /data/users/$USER/ptxas_knobs/ptxas_knobs/* /data/users/$USER/ptxas_knobs/ && \
      rmdir /data/users/$USER/ptxas_knobs/ptxas_knobs

That yields the search-space tiers (`cuda-{12.8,13.0}-ptxas-p{0,1,2}.config`) and the
EVO wheel (`evo_nda-*-linux-nodep.tgz`). Then build the `evo` conda env (Python 3.10):

    conda create -n evo python=3.10 -y
    tar xzf /data/users/$USER/ptxas_knobs/evo_nda-*-linux-nodep.tgz -C /tmp
    conda run -n evo pip install /tmp/evo_nda-*.whl

That's all the setup. The scripts auto-discover via `PTXAS_KNOBS` (default
`/data/users/$USER/ptxas_knobs`) and the `evo` env; override `EVO_PY` / `PTXAS` /
`SS_CONFIG` if your layout differs. Base only needs this fbtriton + torch -- the CIQ
`compileiq` engine is NOT required for the EVO route. ptxas comes from a standard CUDA
toolkit (`/usr/local/cuda-13.0`), matched to the `.config` version. (`ptxas13.3.bin` is
CIQ-route-only and ships in the CompileIQ repo, not this Manifold path.)

## 1. EVO engine smoke (`evo_smoke.py`)

Confirms the EVO Racket core ingests a real ptxas `.config` and runs one generation
(no GPU / no compile -- a no-op objective). Fast sanity check of the engine wiring.

    conda run -n evo python evo_smoke.py $PTXAS_KNOBS/cuda-13.0-ptxas-p0.config

## 2. End-to-end 3-step flow (`run_e2e_search.sh`)

The full collect -> factory -> consume flow on the unmodified naive matmul
(`../examples/user_kernel.py`), with the factory's search driven by EVO over the
constrained p0 space. After the setup above it needs no arguments or edits:

    PTXAS_KNOBS=/data/users/$USER/ptxas_knobs bash third_party/compile_iq/smoke/run_e2e_search.sh

- **collect**: gated jit.py hook dumps a task (stdlib only, zero kernel surface)
- **factory**: `evo_search.py` searches the constrained space (each candidate benched
  in base via `bench_one.py`), then `store_best.py` persists the best ACF to the store
- **consume**: gated make_cubin hook applies the stored ACF on a store HIT

Helpers: `bench_one.py` (base-env compile+benchmark of one ACF), `store_best.py`
(persist best ACF), `evo_search.py` (EVO orchestrator).
