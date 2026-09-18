# CPU backend (vendored)

This directory is a copy of the out-of-tree CPU backend,
[triton-lang/triton-cpu](https://github.com/triton-lang/triton-cpu).
Develop the backend there, not here; this copy only tracks it.

Last sync: triton-cpu [#284](https://github.com/triton-lang/triton-cpu/pull/284)
(on upstream triton
[`72259b1`](https://github.com/triton-lang/triton/commit/72259b1cc3c543c361dcd185a6ff89662e8ed52f)),
imported by [#2981](https://github.com/facebookexperimental/triton/pull/2981).
(Note: the last sync was the first import of the CPU backend into
fbtriton, so it was unusually labor-intensive — in particular, triton-cpu
OSS did substantial one-time pre-work to isolate its source code and
minimize shared in-tree code. Future syncs should be simpler.)

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
