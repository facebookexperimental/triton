# Register Budgets: `minRegAutoWS`, `maxRegAutoWS`, and `maxnreg`

**Frontend**: `third_party/nvidia/backend/compiler.py` (`CUDAOptions`,
`CUDABackend.make_ttgir`), `python/triton/runtime/autotuner.py` (`Config`)

**Passes**:
- `OptimizePartitionWarps.cpp` (`optimizePartitionNumWarps`)
- `lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp` (`AllocateWarpGroups`)
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp`
  (`lowerWarpSpecialize`, `createRegRealloc`)
- `lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp`
- `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` (`getContextualMaxNReg`)

A warp-specialized kernel shares one register file (65,536 32-bit registers per
SM) across all of its warps. On sm_90+ the kernel can move registers between
warp groups at run time with `setmaxnreg` (`nvvm.setmaxregister`). These three
compile options decide how many registers per thread each warp group gets
after that reallocation. `maxnreg` also sets the static per-thread cap that
ptxas compiles against.

## The options

| Option | Type / legal values | Default | Becomes | Validation |
|---|---|---|---|---|
| `minRegAutoWS` | int, multiple of 8 | `24` | module attr `ttg.min_reg_auto_ws` | `_check_reg_auto_ws_alignment` in `CUDAOptions.__post_init__` and `Config` raises `ValueError` if not divisible by 8 |
| `maxRegAutoWS` | int, multiple of 8, or `None` | `None` | module attr `ttg.max_reg_auto_ws` (only if not `None`) | same |
| `maxnreg` | int or `None` | `None` | module attr `ttg.maxnreg` (only if not `None`) | none in Python |

All three are `CUDAOptions` fields, so they are part of the compile cache key.
They can be set on `triton.Config(...)` or as launch kwargs. `make_ttgir`
writes the module attributes before any TTGIR pass runs.

## Pipeline

### 1. `OptimizePartitionWarps`: per-partition requests

The pass runs on every `ttg.warp_specialize` that is not a manual TLX warp
specialization (`shouldBail`). The Meta WS Blackwell pipeline runs it twice:
after software pipelining, and again after TMA lowering. It does two things.

**Shrinks partition warp counts.** It repeatedly halves a partition's warps
while the largest tensor in it still fits. The fit test assumes the register
file is split evenly across all warps (`65536 / 32 / totalWarps`). It ignores
`maxnreg`, `minRegAutoWS`, and `maxRegAutoWS`.

**Writes `requestedRegisters`.** It sets one entry per partition region (not
for the default region):

| Partition | `maxRegAutoWS` unset | `maxRegAutoWS` set |
|---|---|---|
| Holds a tensor with more than one element | `-1` (sentinel: take an even share of what is left) | `maxRegAutoWS` |
| Otherwise (load, MMA-issue, TMA-store, relay partitions) | `minRegAutoWS` | `minRegAutoWS` |

If the `ttg.min_reg_auto_ws` attribute is absent, `24` is used. The default
region, i.e. the `num_warps` warp group, never gets a request from AutoWS.
`defaultRequestedRegisters` is only set by TLX. So the default region always
takes the even share of the remaining pool, whether or not `maxRegAutoWS` is
set.

### 2. `AllocateWarpGroups`: the pool and the split

This pass runs during TTGIR→LLVM conversion.

1. **Pad to whole warp groups.** It pads every `ttg.warp_specialize` with
   empty partitions to the same number of full 4-warp groups. Padding
   partitions request 24. It also records
   `ttg.total-num-warps = num_warps + 4 × numExtraWarpGroups`.
2. **Skip if there are no requests.** If no `ttg.warp_specialize` has register
   requests, it stops here.
3. **Pick the per-thread starting budget** `maxnreg`. This is `ttg.maxnreg` if
   the user set it. Otherwise it is
   `floor(65536 / (totalWarps × 32))`, rounded down to a multiple of 8.
4. **Pool** = `maxnreg × totalWarps × 32`.
5. **Fixed requests.** Each 4-warp group's request is the maximum request
   among the partitions it contains, rounded up to a multiple of 8. A
   partition containing `tt.assert`/`tt.print`, or any partition when
   instrumentation is on, is raised to at least 32 (`minRegistersForRegion`).
   This also applies to a tensor partition's share sentinel, so such a tensor
   partition gets a fixed 32-register request instead of a share.
   Fixed requests are subtracted from the pool.
6. **Even split.** What remains is split evenly, rounded down to a multiple of
   8, across the default region and every warp group whose request is the
   `-1` sentinel.
7. **Bail-out checks.** It gives up without recording anything if the
   remainder is negative, or if the even share is below 24 (below 32 for a
   default region containing assert/print).
8. **Record.** On success it writes `actualRegisters` (default region first)
   and sets `ttg.maxnreg` to the starting budget. That means `ttg.maxnreg` is
   now present even when the user did not set it.

The pool is computed from `maxnreg` and is not checked against the hardware's
65,536 registers. A user `maxnreg` larger than `65536 / (totalWarps × 32)`
produces a kernel whose static register demand exceeds the SM. A smaller one
shrinks the pool available to every partition.

### 3. `lowerWarpSpecialize`: `setmaxnreg` emission

Dynamic reallocation is emitted only when `ttg.maxnreg` is present and no
`ttg.warp_specialize` carries `tti.disable_setmaxregister` (set by the global
sanitizer). In that case:

1. **Kernel start.** Worker warps are assumed to drop to 24 registers.
   `defRegs` is how many registers that frees for the default warps. If
   `defRegs` would exceed 256, it is capped at 256 and the workers drop less.
2. **Default warps.** They `setmaxnreg` from `maxnreg` to `defRegs` at kernel
   start, then to `actualRegisters[0]` at the start of their region, and back
   at the end.
3. **Worker warps.** They drop to the low value at the top of the switch loop,
   rise to their partition's `actualRegisters` entry at partition start, and
   return at partition end.

`createRegRealloc` clamps every value to `[24, 256]` and rounds it down to a
multiple of 8. It emits nothing when the value does not change.

### 4. Other consumers of `ttg.maxnreg`

- `FuncOpToLLVM` copies `ttg.maxnreg` to `nvvm.maxnreg`, which becomes the
  PTX `.maxnreg` directive. This applies to non-WS kernels too: for them,
  `maxnreg` is only a ptxas cap.
- `getContextualMaxNReg` limits TMEM load/store message width (`tcgen05.ld` /
  `tcgen05.st`) to the smallest of these:
  - 256;
  - the enclosing partition's `actualRegisters` (or `requestedRegisters`)
    entry;
  - the module `ttg.maxnreg`.

  The lowering uses at most half of that limit for one message.

## Resources affected

- **Registers per thread, per warp group.** This is the direct effect of all
  three options, through `actualRegisters` and `setmaxnreg`.
- **TMEM load vectorization.** Fewer registers mean narrower `tcgen05.ld`
  messages and more of them.
- **Warps.** These options do not change warp counts.
  `OptimizePartitionWarps` shrinks warps using its own even-split estimate,
  independently of them.

## Silent fallbacks

- **`AllocateWarpGroups` gives up** (negative remainder, or even share below
  24). This happens, for example, when `maxRegAutoWS` times the
  tensor-partition warps does not fit in the pool. No `actualRegisters` are
  recorded and nothing is reported:
  - If the user did not set `maxnreg`, `ttg.maxnreg` stays absent. No
    `setmaxnreg` is emitted, and every warp runs with ptxas's uniform
    allocation.
  - If the user did set `maxnreg`, the kernel-start and switch-loop
    reallocations are still emitted, but the per-partition ones are not.
    Worker partitions therefore keep the low value computed in step 3.1
    (normally 24) for their whole lifetime.
- **Out-of-range values are clamped at emission.** A `minRegAutoWS` or
  `maxRegAutoWS` outside `[24, 256]` is not rejected. `createRegRealloc`
  clamps it to the nearest legal value.
- **TLX kernels with manual warp specialization** skip `OptimizePartitionWarps`.
  For them, `minRegAutoWS` and `maxRegAutoWS` have no effect.

The chosen values are visible in the TTGIR as `requestedRegisters` /
`actualRegisters` on `ttg.warp_specialize` and as `ttg.maxnreg` on the module,
and in PTX as `setmaxnreg` instructions.
