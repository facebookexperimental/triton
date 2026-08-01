# Working on Triton

## Build and Testing Guidelines
- Before running tests for native/compiler changes, run `make` in the triton directory to rebuild triton. DO NOT RUN `make` if you only changed Python code or code in `python/triton_kernels`.
- For compiler changes, add tests in `python/test/` (pytest) or test (lit). Keep GPU-only tests in `python/test/unit/` or `python/test/gluon/`, name them `test_<feature>_<condition>`, and avoid creating new test files unless requested.
- Run pytest with `-n auto -s --tb=short`. Run a single test with `pytest -n auto file.py::test_name`.
- The build dir is given by `BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
- Run lit from the build dir:  `cd BUILD_DIR; ninja triton-opt; lit -v test/<path>.mlir` (example: `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir`).
- Lit tests can be run locally (no GPU required).
- Compiler crashes sometimes print an MLIR reproducer (external_resources / mlir_reproducer). Save the full MLIR + {-# ... #-} metadata to `/tmp/<file>.mlir`, then run `triton-opt /tmp/<file>.mlir --run-reproducer` to reproduce locally.

## Performance Sweep Regression Policy
- Every kernel in the full performance sweep is a protected gate. Compiler or
  backend changes must preserve compilation, correctness, runtime execution,
  and performance for every listed configuration.
- Run the full sweep after changes that can affect Wave code generation. A
  failure inherited from a recent branch revision is still a regression: find
  the last known-good revision and fix it. Never waive a failure by treating
  the immediately preceding broken revision as the baseline.
- Do not accept aggregate or flagship-kernel wins that hide another kernel's
  regression. Report and compare every configuration against LLVM and the
  last known-good Wave baseline.
- Any statistically credible slowdown in any sweep configuration blocks the
  change. Resolve noise with matched, repeated A/B measurements; never waive
  an individual regression against gains elsewhere.

## TLX Wave async DMA
- Direct-to-LDS async DMA completion is synchronized only by an explicit
  `wait_group`. Never infer a DMA dependency from LDS aliasing, destination
  accesses, pending reads/writes, or allocation history, and never compensate
  by changing the Wave scheduler.
- Preserve the explicit wait result as a token dependency on dominated DS
  operations. If the source protocol does not provide the required wait, fix
  or reject the source protocol instead of manufacturing an implicit wait.

## Wave scheduling boundary
- Greedy scheduling only traverses legal ready candidates and fills stalls
  reported by the Wave model. Target, occupancy, latency, resource, and filler
  compatibility policy belongs in the model, never in the scheduler.
- Represent new scheduling opportunities as named model stalls, such as
  `CoexecWindow`. Do not add target-specific ranking, a second order, or a
  post-schedule veto to the scheduler.
