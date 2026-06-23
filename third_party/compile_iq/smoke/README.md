# compile_iq smoke tests

PTX-direct ptxas-ACF tuning, exercised end to end on a single B200.

## Route: PTX-direct (no source dumped)

Tuning and applying an ACF assembles a **fixed `kernel.ptx`** with ptxas and launches the cubin
via the CUDA **driver API** -- the kernel is never recompiled from Python source, and collection
captures **only** PTX + a launch spec (no `source.py` is ever dumped):

    ptxas <fixed kernel.ptx> [--apply-controls=<acf>] -o cand.cubin   # the per-candidate compile
    cuModuleLoadData(cand.cubin) -> cuLaunchKernel(...)               # the objective body (run + time)

Because the PTX is frozen, the tuned cubin is byte-identical to production and the whole "did the
source recompile to the same PTX?" parity problem disappears; it is also op-agnostic and needs no
source. The reusable launcher is `../compile_iq/ptx_launch.py`.

## Engine & search spaces (important)

The ACF search is driven by the **EVO** engine (`evo_solar`/`evo_nda`, in the `evo` conda env,
Python 3.10) over a **constrained** ptxas knob tier (`cuda-<ver>-ptxas-p{0,1,2}.config`, p0 safest).
The EVO env has no triton/torch, so each candidate is benchmarked in base 3.13 via the
`ptx_bench_one.py` subprocess bridge (ptxas + driver launch).

**Version matching is required, on two axes:**
- A `cuda-X.Y` tier yields ACFs that only ptxas **X.Y** accepts (a mismatched ptxas rejects them
  with "Invalid compiler controls file").
- The **PTX ISA `.version`** is chosen by Triton from the ptxas it sees at compile time, and a
  CUDA-X.Y driver cannot run a newer-ISA cubin. So `PTXAS` must match the tier version **and** the
  installed driver.

The `run_e2e_cuda12{8,30}.sh` wrappers pin a matching set for you.

## Setup (internal): fetch artifacts -- NOT checked in

The EVO wheel and the search-space `.config` files are NVIDIA-provided binaries (the EVO wheel is
under NDA) and are deliberately **not committed**. Internal users fetch them from Manifold:

    manifold getr tc_bench_ci/tree/compileiq/ptxas_knobs /data/users/$USER/ptxas_knobs
    # getr nests one level -- flatten so the files sit directly in $PTXAS_KNOBS:
    mv /data/users/$USER/ptxas_knobs/ptxas_knobs/* /data/users/$USER/ptxas_knobs/ && \
      rmdir /data/users/$USER/ptxas_knobs/ptxas_knobs

That yields the search-space tiers (`cuda-{12.8,13.0}-ptxas-p{0,1,2}.config`) and the EVO wheel
(`evo_nda-*-linux-nodep.tgz`). Then build the `evo` conda env (Python 3.10):

    conda create -n evo python=3.10 -y
    tar xzf /data/users/$USER/ptxas_knobs/evo_nda-*-linux-nodep.tgz -C /tmp
    conda run -n evo pip install /tmp/evo_nda-*.whl

The scripts auto-discover via `PTXAS_KNOBS` (default `/data/users/$USER/ptxas_knobs`) and the `evo`
env; override `EVO_PY` / `PTXAS` / `SS_CONFIG` if your layout differs. Base needs this fbtriton +
torch + cuda-python; ptxas comes from a standard CUDA toolkit (e.g. `/usr/local/cuda-13.0`), matched
to the `.config` version AND the driver.

## 1. EVO engine smoke (`evo_smoke.py`)

Confirms the EVO core ingests a real ptxas `.config` and runs one generation (no GPU / no compile --
a no-op objective). Fast sanity check of the engine wiring.

    conda run -n evo python evo_smoke.py $PTXAS_KNOBS/cuda-13.0-ptxas-p0.config

## 2. End-to-end (`run_e2e_cuda128.sh` / `run_e2e_cuda130.sh`)

The version-pinned wrappers are the entry points; each runs the full PTX-direct flow with a matching
ptxas + tier:

    bash third_party/compile_iq/smoke/run_e2e_cuda128.sh   # ptxas 12.8 (B200 production toolkit)
    bash third_party/compile_iq/smoke/run_e2e_cuda130.sh   # ptxas 13.0

Both accept `TIER` (p0/p1/p2, default p0) and the usual `PTXAS_KNOBS` / `CUDA_VISIBLE_DEVICES` /
`PER_CAND_TIMEOUT` overrides. They are thin wrappers over `run_ptx_e2e.sh`, which runs three stages:

- **emit** -- `ptx_direct_smoke.py` compiles the naive matmul once -> `kernel.ptx` + `spec.json`
  (the source-free launch description), and proves baseline `ptxas(kernel.ptx)` -> driver-launch
  == torch + Triton bit-for-bit.
- **mint** -- `ptx_evo_search.py` runs EVO over the constrained tier, each candidate scored through
  the PTX-direct driver-launch bridge `ptx_bench_one.py` in base 3.13; writes `best.acf.hex`.
- **apply** -- driver-launch with `--apply-controls=best.acf`, validated (self-consistent) + timed.

To run unpinned (defaults to cuda-13.0): `bash run_ptx_e2e.sh`.

`spec.json` previews the schema the production collector must capture for a source-free apply
(today's `collector.py` stops at args/grid; it does not yet record `num_warps`/`shared`/scratch
sizes / the post-specialization arg order). Scope: naive matmul with null global/profile scratch;
WS/TMA (non-zero scratch, tensordesc args) is future work.

Helpers: `ptx_launch.py` (ptxas + driver launch + launch-spec), `ptx_bench_one.py` (isolated
per-candidate PTX-direct bench bridge), `ptx_evo_search.py` (EVO orchestrator), `ptx_direct_smoke.py`
(emit/baseline/apply orchestrator).
