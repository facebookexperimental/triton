# LLIR Scheduler — out-of-tree LLVM pass plugin

The LLIR scheduler is shipped here as an **out-of-tree LLVM pass plugin**. It classifies
each scheduling region and routes it to one of two models:

- **MFMA ↔ memory — a throughput problem.** The original model, used by the GEMM hot loops
  and described in the
  [a16w16 v5 README](../../kernels/gemm/intra_wave/a16w16/v5_local_prefetch/README.md). It
  reorders the MFMA/`ds_read`/`buffer_load` instructions in the LLVM-IR hot loop and pins
  the order with `llvm.amdgcn.sched.barrier(i32 0)` after each memory anchor, so LLVM's
  machine scheduler preserves the interleave (no misched-disable needed).
- **MFMA ↔ VALU — a co-execution problem.** Used by the flash-attention dot clusters
  ([`kernels/attention/`](../../kernels/attention/README.md) §6). Every vector op has to be
  assigned to a specific MFMA's 24-cycle shadow, in the right form, or it falls outside and
  costs cycles. Here the plugin does not reorder: it *declares* the intended pipeline with
  `sched_group_barrier` and lets AMDGPU's IGroupLP construct it. A second algorithm handles
  regions whose VALU demand exceeds the available shadow.

> **Learn the algorithm** → [`llir_scheduler.html`](llir_scheduler.html). The illustrated
> design reference walks through the whole pass: instruction classification, dependency-safe
> region formation, the MFMA↔memory interleaving budget and its cost model, and how the
> schedule is pinned with `sched.barrier` instead of disabling misched.

## Files
- `LlirSchedPlugin.cpp` — the pass, as a new-PassManager plugin (`llvmGetPassPluginInfo`,
  auto-inserted at the `OptimizerLast` extension point).
- `libLlirSched.so` — prebuilt plugin (see pin below).
- `llir_scheduler.html` — design reference: how the pass forms regions, sizes the
  MFMA↔memory interleave, and pins the result with `llvm.amdgcn.sched.barrier`.

## Pinned toolchain (important — ABI lock)
The `.so` is a native LLVM plugin and is **ABI-locked to the exact LLVM that
Triton is built with**. This tutorial pins Triton to [`gfx950-tutorial-v1.1`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v1.1) for the GEMM
kernels and [`gfx950-tutorial-v2.0`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v2.0) for the attention kernels.
**Both tags carry the same LLVM, commit `850a2b1b`** (see `cmake/llvm-info.json`), so the
one prebuilt `.so` works with either. If the LLVM pin itself moves, **rebuild the `.so`**.

## Build
Build against the same LLVM Triton uses (downloaded to `~/.triton/llvm/llvm-850a2b1b-*`):

```bash
LLVM=$(dirname $(dirname $(find ~/.triton/llvm -name llvm-config | head -1)))
g++ -shared -fPIC -fvisibility=default \
    $("$LLVM/bin/llvm-config" --cxxflags) \
    -o libLlirSched.so LlirSchedPlugin.cpp
```
The plugin does **not** link LLVM; it resolves LLVM symbols from `libtriton` at
load time (see prerequisites).

## Triton prerequisites
Both pins carry the source change this plugin needs: it
*keeps the TargetMachine for LLVM plugins* via `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1`,
without which `optimize_module` runs all of O3 with no target machine and codegen
regresses. The one thing left to the builder is symbol visibility:

- **Build with default visibility:** `TRITON_EXT_ENABLED=1 pip install -e .`
  (the default `-fvisibility=hidden` build exports no LLVM symbols, and
  `PassPlugin::Load` fails with `undefined symbol`).

`bench.py` handles the runtime requirement automatically: when `LLVM_PASS_PLUGIN_PATH`
is set it loads `libtriton` with `RTLD_GLOBAL` so the plugin can resolve symbols.

## Use
```bash
LLVM_PASS_PLUGIN_PATH=/abs/path/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
    python bench.py --version 8 --K 8192 --dtype fp16
```
`scripts/run_perf_table.py` wires both env vars into the `llir`, `llir+force-agpr`, and
`llir+force-agpr+amdgcnas` configs automatically.

## The plugin source
`LlirSchedPlugin.cpp` is the maintained plugin source — a self-contained
new-PassManager LLVM pass plugin: it carries no Triton headers and registers
itself via `llvmGetPassPluginInfo`,
auto-inserted at the `OptimizerLast` extension point. Edit it and rebuild the
`.so` with the `g++` command in **Build** above.
