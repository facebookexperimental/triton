# LLIR scheduler pass plugin (gfx950)

An out-of-tree LLVM pass plugin that schedules the hot loop of a 4-wave
(intra-wave) MFMA GEMM: it interleaves the MFMA instructions with the LDS reads
and direct-to-LDS loads that are independent of them, and pins the result with
`llvm.amdgcn.sched.barrier`, so LLVM's machine schedulers keep the interleave.
Nothing is disabled globally and kernels without such a loop are left alone.

`LlirSchedPlugin.cpp` is the plugin of the
[gfx950 Gluon tutorials](https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/plugins/llir_scheduler)
(MIT, AMD), unchanged apart from formatting. Its design reference, with the
region formation and the interleaving cost model, is
[`llir_scheduler.html`](https://github.com/ROCm/gfx950-gluon-tutorials/blob/main/plugins/llir_scheduler/llir_scheduler.html)
in that repository.

## When to use it

Use it for kernels where one wave both loads and computes, such as
[`a16w16/v7_slice`](../../a16w16/v7_slice/matmul_kernel.py). The pass
needs the next iteration's LDS reads to be independent of the current MFMAs
(local prefetch), and it keeps `cd_regclass` accumulator pins
(`tlx.amd_dot(..., cd_regclass="a")`) attached to their MFMAs and fences them.

Do not load it for warp-pipelined kernels (`tlx.warp_pipeline_stage`, e.g.
`a16w16/v8_warp_pipeline`, `v9_beyond_hotloop`, `tlx.ops.mm`). Those already
get their mem/MFMA interleave from the two wave groups; the pass declines their
regions but still pads them, which measured 2-5% slower on gfx950.

## Build

```bash
./build.sh        # writes libLlirSched.so next to the source
```

The plugin does not link LLVM. It resolves LLVM symbols from `libtriton` when
it is loaded, so it is ABI-locked to the LLVM in `cmake/llvm-info.json` and
has to be rebuilt when that pin moves. A plugin built for another LLVM
revision crashes at load.

Triton has to be built with default symbol visibility so `libtriton` exports
the LLVM symbols:

```bash
TRITON_EXT_ENABLED=1 pip install -e . --no-build-isolation
```

## Use

```bash
LLVM_PASS_PLUGIN_PATH=$PWD/libLlirSched.so \
  python ../../a16w16/bench.py --version 7 --K 8192
```

`bench.py` loads `libtriton` with `RTLD_GLOBAL` when `LLVM_PASS_PLUGIN_PATH` is
set. A driver of your own has to do the same before the first `import triton`:

```python
import os, sys
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
import triton
```

`LLVM_PASS_PLUGIN_PATH` is part of Triton's cache key, so scheduled and
unscheduled builds of one kernel do not collide in the cache.
