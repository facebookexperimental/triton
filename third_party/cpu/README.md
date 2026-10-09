# CPU backend (vendored)

This directory is a copy of the out-of-tree CPU backend,
[triton-lang/triton-cpu](https://github.com/triton-lang/triton-cpu).
Develop the backend there, not here; this copy only tracks it.

## Sync history

### First sync: initial import
triton-cpu [#284](https://github.com/triton-lang/triton-cpu/pull/284)
(on upstream triton
[`72259b1`](https://github.com/triton-lang/triton/commit/72259b1cc3c543c361dcd185a6ff89662e8ed52f)),
imported by fbtriton
[#2981](https://github.com/facebookexperimental/triton/pull/2981).
(Note: the first import was unusually labor-intensive — in particular,
triton-cpu OSS did substantial one-time pre-work to isolate its source
code and minimize shared in-tree code. Future syncs should be simpler.)

### Sync 2026-09: CPU unit tests + post-#284 functional deltas
triton-cpu [#285](https://github.com/triton-lang/triton-cpu/pull/285)
through [#301](https://github.com/triton-lang/triton-cpu/pull/301), plus
the `test_core.py` allowlist isolation (lands in triton-cpu OSS first —
fbtriton only imports, never originates, CPU changes).

Verbatim copy:
- `python/test/unit/cpu/` (all 9 files incl.
  `test_core_cpu_allowlist.py`).
- The `test_core.py` selection hook in `python/test/conftest.py`; the
  16 `is_cpu()` guards in `test_core.py`, 1 in `test_cache.py`; the
  `is_cpu()` helper in `_internal_testing.py`.
- `ConvertDotToNanokernel.cpp` (#286/#297), `compiler.py` options/bf16
  deltas, `FuncOpToLLVM.cpp` (#288 comment only — the code was already
  here), `tutorials/cpu/08-sfc-matmul.py`.

Modified port (fbtriton's core predates the OSS base):
- `test_cache.py`: the 2 `#289` mock tests carry a 1-line
  fbtriton-only mock adaptation (Meta's interceptor API).
- `ConvertMemoryOps.cpp`, `ConvertElemManipOps.cpp`,
  `ScalarizeUsingForOps.cpp`: base-bump churn referencing newer core
  APIs (rich cache-policy API) reverted.
- `compiler.py`: keeps `vec_lib = None` (#3365).
- Allowlist drops 4 tests Meta removed from `test_core.py` (see the
  module docstring).

Ignored (not ported):
- `backend/build.py`, `backend/driver.py`: fbtriton forks (hermetic
  toolchain, tuple API, NativeRT export). The OSS #285/#287
  equivalents are already present here via hand-ports.
- `test/TritonCPU/` lit tests (16 files; no registration needed —
  lit auto-discovers them).

## Syncing Timing

`triton/stable` in fbsource is promoted to the next version periodically.
On each promo, also sync this directory to the latest triton-cpu:

1. Copy `third_party/cpu/` over; it applies mostly verbatim.
2. Hand-port the in-tree integration below. It has no counterpart in
   triton-cpu and must be reconciled with fbtriton's side on every sync.
3. Keep the LLVM revision in agreement. Both fbtriton and triton-cpu build
   on upstream Triton and its pinned LLVM.
4. Keep the SLEEF revision in `CMakeLists.txt` in sync. CMake fetches the
   pinned revision; offline builds must provide `TRITON_SLEEF_SOURCE_DIR`.

## In-tree integration (outside this directory)

- `bin/RegisterTritonDialects.h`, `bin/CMakeLists.txt`
  (`TRITON_ENABLE_CPU_BACKEND`) — dialect/tool registration
- `setup.py` — `cpu` in the backend copy list
- `python/triton/runtime/{driver,jit,autotuner}.py`,
  `python/triton/compiler/compiler.py` — driver/cache selection, JIT, autotune
- `python/triton/language/semantic.py` — tensor-descriptor handling
- `python/triton/testing.py` — benchmark timing
- `include/triton/Tools/Sys/GetEnv.h` — cache-invalidating `TRITON_CPU_*`
  env vars
- `python/tutorials/cpu/` — CPU tutorials, copied from triton-cpu
- `.github/workflows/`, `.github/scripts/bisect_failure.sh` — CI builds the
  CPU backend

## Verify

Rebuild with `cpu` in `TRITON_CODEGEN_BACKENDS`, run the CPU tutorials,
and keep the runtime/descriptor suites green (see the
[#2981](https://github.com/facebookexperimental/triton/pull/2981) test plan).
