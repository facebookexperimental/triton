# compile_iq — decoupled ptxas-ACF tuning for Triton

Productionizes the ptx-acf smoke test: instead of CompileIQ wrapping the kernel,
the kernel run and the search are **decoupled via disk**, with **zero user surface**.

```
(1) COLLECTION            (2) FACTORY (offline)          (3) CONSUMPTION
user runs kernel  ──►  compileIQ task  ──► CompileIQ ──► ACF store ──► user runs again
 jit.py hook            on disk            search         on disk        make_cubin hook
 (FBTRITON_COMPILE_IQ_COLLECT)                    best ACF→store     (FBTRITON_COMPILE_IQ_APPLY) applies ACF if hash hits
```

## Layout
```
third_party/compile_iq/compile_iq/
  __init__.py    install/enable
  store.py       sha256(PTX) hashing + ACF store ($COMPILE_IQ_STORE/<arch>/<sha>.acf)
  collector.py   Stage 1: dump a self-contained task
  replay.py      Stage 2a: re-run a kernel from a task, ACF applied via PTX_OPTIONS
  factory.py     Stage 2b: task -> CompileIQ search -> best ACF -> store
  consume.py     Stage 3: look up ACF by PTX hash -> --apply-controls args
# symlink: python/triton/compile_iq -> ../../third_party/compile_iq/compile_iq
# core hooks (env-gated, no-op by default):
#   collection: python/triton/runtime/jit.py
#   apply:      third_party/nvidia/backend/compiler.py  (make_cubin)
```

## Enable (zero user code change)
```
# collection (Triton users): just an env var; no kernel/Triton-user changes
export FBTRITON_COMPILE_IQ_COLLECT=1                  # turn on collection
export COMPILE_IQ_TASK_DIR=/path/to/tasks    # default ~/.compile_iq/tasks

# factory (offline operators only): needs ptxas >= 13.3 + the search-space bin
pip install nvidia-cuda-nvcc                 # ptxas 13.3 — auto-discovered, no path needed
export COMPILE_IQ_STORE=/path/to/store       # default ~/.compile_iq/store
export COMPILE_IQ_SEARCH_SPACE_BIN=/path/to/ptxas13.3_search_space.bin
# (TRITON_PTXAS_BLACKWELL_PATH only needed to override auto-discovery)

# consumption (gradual prod rollout): apply stored ACFs at compile time
export FBTRITON_COMPILE_IQ_APPLY=1           # default OFF; needs ptxas>=13.3 (version-guarded, fail-open)
```

## Test plan / checkpoints

- [x] **CP1 — collection (no-autotune, zero user surface)**: run a normal kernel
      program (no compile_iq references) with collection enabled purely via env:
      `FBTRITON_COMPILE_IQ_COLLECT=1 COMPILE_IQ_TASK_DIR=/tmp/ciq_tasks python examples/user_kernel.py`
      (collection doesn't *apply* an ACF, but see the ptxas-version PIN below: the
      captured PTX sha tracks the ptxas in effect, so collection & consumption must
      use the same ptxas for the store key to line up — the Stage-3 hook runs inside
      Triton's compile with that same ptxas, so it's consistent by construction.)
      → `<task>/{kernel.ptx,source.py,task.json}` appears with ptx_sha256, arch,
      launch dims, grid, arg metadata. `examples/user_kernel.py` is a plain Triton
      matmul with zero compile_iq imports — proof the kernel needs no changes. ✅
- [x] **CP2 — replay**: `replay.load_task/load_kernel/build_args/run_once` reproduces
      correct output from the task alone (rel-err 0 vs torch). ✅
- [x] **CP3 — factory → store**: `python -m triton.compile_iq.factory <task_dir>`
      runs the search and writes `$COMPILE_IQ_STORE/<arch>/<sha>.acf` + `.acf.json`. ✅
- [ ] **CP4 — autotuned kernel (ws)**: collection fires per compiled config; capture/
      select the *winning*-config task; replay needs `TensorDescriptor` (TMA) + cluster
      launch support in `replay.build_args`/`run_once`. (no-autotune vs autotune is the
      only axis that matters; kernel complexity is otherwise equivalent.)
- [x] **CP5 — consumption (in-Triton `make_cubin` hook)**: a gated hook in
      `nvidia/backend/compiler.py::make_cubin` (env `FBTRITON_COMPILE_IQ_APPLY`, default
      OFF) hashes the PTX, looks up `store.read_acf(sha, arch)`, and appends
      `--apply-controls` on a hit — fires per config during autotuning. Version-guarded
      (only ptxas ≥13.3, via `packaging.Version`) + try/except → fail-open. Verified:
      hook sha == collection sha (parity); HIT changes the cubin (40800→65304 B) and
      applies in ~0.05s; MISS compiles unchanged; old ptxas → skipped (fail-open). ✅

## Decisions / pins
- **Replay = recompile-through-Triton with ACF via PTX_OPTIONS**, not raw cubin launch.
  Rationale: Triton specializes/reorders/drops kernel args (the PTX `.entry` param list
  != Python signature; `_dispatch_arg_indices` is often `None`), so raw-cubin launch
  would mean reimplementing Triton's ABI. Recompiling the captured source is robust,
  decoupled, and the regenerated PTX matches the captured sha. Raw-cubin = future
  hardening (source-free / exact-byte fidelity).
- **PIN (hash-key design)**: store key is `sha256(PTX) × arch` for now. PTX subsumes
  autotune config + dtype/alignment specialization but NOT runtime shape (M,N,K). Tasks
  capture shapes/identity so we *can* extend the key (per-shape ACFs) later.

Status: CP1–CP3 + CP5 done+verified (no-autotune naive GEMM). CP4 (autotune) is the remaining checkpoint.

## Dependencies (who needs what)
- **Triton users (collection)**: nothing — the hook is a no-op unless `FBTRITON_COMPILE_IQ_COLLECT`
  is set; collection imports only stdlib. CP1 is testable with just an fbtriton build.
- **Factory operators (offline)**: `pip install nvidia-cuda-nvcc` (ptxas ≥13.3, auto-discovered),
  the **CompileIQ engine** (the NVIDIA "Evo" wheel — bundles the proprietary core; no source
  build / git-lfs), and the **search-space `.bin`** (a separate NVIDIA artifact) via
  `COMPILE_IQ_SEARCH_SPACE_BIN`. None of these are vendored in this repo.

## Factory setup — install the CompileIQ engine
The factory imports `compileiq` (NVIDIA's proprietary search engine). Get it via either:
```
# (a) PREFERRED: the internal CompileIQ/Evo wheel (bundles the engine binary; no git-lfs)
pip install <compileiq-evo-wheel>

# (b) FALLBACK: from the CompileIQ source checkout (engine binary comes via git-lfs)
git clone <NVIDIA CompileIQ repo> CompileIQ
cd CompileIQ && git lfs install && git lfs pull   # fetch core / libciq.so
pip install .                                     # installs the `compileiq` package

# plus, for both paths:
pip install nvidia-cuda-nvcc                       # ptxas 13.3 (auto-discovered)
# obtain the ptxas13.3 search-space .bin (NVIDIA artifact) and point at it:
export COMPILE_IQ_SEARCH_SPACE_BIN=/path/to/ptxas13.3_search_space.bin
```
Collection (Stage 1) needs **none** of the above — only an fbtriton build.

## One-shot commands (collect → factory)
```
# 0) clean
rm -rf /tmp/ciq_tasks /tmp/ciq_store

# 1) collect: a normal kernel run; collection on via env only (no extra deps)
FBTRITON_COMPILE_IQ_COLLECT=1 COMPILE_IQ_TASK_DIR=/tmp/ciq_tasks python third_party/compile_iq/examples/user_kernel.py

# 2) factory: task -> CompileIQ search -> ACF store  (needs the Factory setup above)
COMPILE_IQ_STORE=/tmp/ciq_store COMPILE_IQ_SEARCH_SPACE_BIN=/path/to/ptxas13.3_search_space.bin python -m triton.compile_iq.factory $(ls -d /tmp/ciq_tasks/*/ | head -1)
```
