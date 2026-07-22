# HSTU Self-Attention Forward — AutoWS Case Matrix & DP=2 Investigation

> **RESOLVED.** Root cause = the compiler DP=2 (`WSDataPartition`) transform split the
> SMEM buffers/MMAs into 128-row halves but left the device-side TMA descriptor
> `box_dim` at the un-partitioned `BLOCK_M=256`, so the TMA overran the 128-row SMEM
> buffer → SMEM corruption → deadlock/IMA. Localized by a PTX diff (sole functional
> delta: `box_dim[1] = 256` vs `128`). **Fixed** in `WSDataPartition.cpp`
> (`scaleDeviceTensormapBoxDims`, scales box_dim by the partition factor). Compiler
> DP=2 now correct; `test_self_attention_autows.py` 7 passed. Full detail +
> ttgir/tlx/ptx artifacts in `devmate/ttgir_dp2_cmp/` (`ROOT_CAUSE.md`).


Investigation of the HSTU self-attn **forward** kernel (`_hstu_attn_fwd` in
`third_party/tlx/tutorials/hstu_self_attn/triton_hstu_attention.py`) under
automatic warp specialization (meta-WS), on **GB200**, focused on the DP=2
runtime failure reported via paste `P2427737220`.

## Repos / build

- Repo: `/home/mren/MetaMain/triton`
- Env: conda `metamain` (`PATH=/home/mren/.conda/envs/metamain/bin`,
  `LIBRARY_PATH=LD_LIBRARY_PATH=/home/mren/.conda/envs/metamain/lib`)
- Build: `ninja -C build/cmake.linux-aarch64-cpython-3.12` (rebuilds
  `python/triton/_C/libtriton.so`, the runtime lib the editable install loads)
- Commits under test:
  - **base** `95a42269c` — `[AutoWS][HSTU] Enable autoWS on self-attn backward + TLX-matching dq-reduce`
  - **tip** `216057806` — regroup tip (`main`), = base + 8 ported AutoWS/HSTU commits folded into 3 diffs (A DP fixes, B debug+test, C HSTU fwd fold)

## Test configs

`P2427737220` runs the fwd for `L=256, Z=4, H=2, D=128` with:
`HSTU_SELF_AUTOWS=1`, `HSTU_SELF_DP=2`, `HSTU_SELF_PIN=1`,
`HSTU_SELF_AUTOWS_WARPS=4`, `TRITON_USE_META_WS=1`,
`TRITON_DISABLE_WSBARRIER_REORDER=1`. Reference: torch-autograd float causal-SiLU.

Two tutorial-supported knobs matter:
- `HSTU_SELF_DP` — data-partition factor. **Default 2** in the kernel, but the
  official test **pins DP=1** (see below).
- `HSTU_SELF_AUTOWS_WARPS` — sets `num_warps` on the pinned autoWS fwd config.
  **Default 8** (`triton_hstu_attention.py`: `int(os.environ.get(..., "8"))`).

## Result matrix

| # | Commit | DP | WARPS | reorder-disable | Result |
|---|--------|----|-------|-----------------|--------|
| 1 | base `95a42269c` | 2 | 4 | 1 | ❌ **compile** `OutOfResources: tensor memory, Required 1024 > 512` |
| 2 | base `95a42269c` | 1 | 4 | 1 | 🔒 **hang** (100% util, killed 350s) |
| 3 | base `95a42269c` | 1 | 8 (default, official test) | 1 | ✅ **PASS** — `test_self_attention_autows.py` 3/3, rel-L2 dq/dk/dv = 2.81e-3/2.35e-3/2.35e-3 |
| 4 | tip `216057806` | 2 | 4 | 1 | ❌ **runtime** CUDA illegal memory access (async, at sync) |
| 5 | tip `216057806` | 2 | 4 | 0 (reorder ON) | ❌ **runtime** same illegal memory access |
| 6 | tip `216057806` | 2 | 4 | — (under compute-sanitizer memcheck) | 🔒 **deadlock** — 100% util spin, no OOB report emitted (timing-dependent) |
| 7 | tip `216057806` | 1 | 8 | 1 | ✅ **PASS** — `kernel done 2.6s`, fwd rel-L2 = 2.34e-3 |
| 8 | tip `216057806` | 1 | 4 | 1 | 🔒 **hang** (100% util, killed) |
| 9 | tip `216057806` | 1 | 8 (default, official test) | 1 | ✅ **PASS** — `test_self_attention_autows.py` 3/3 (identical numerics to base) |

## Findings

1. **DP=2 is off the supported envelope at the base.** The official test hard-codes
   `HSTU_SELF_DP=1` with the comment: *"data_partition_factor=2 would need
   BLOCK_M=256 (each slice >=128 TMEM rows) which OOMs TMEM on this kernel."* That
   is exactly case #1 (needs 1024 TMEM cols vs 512 hw limit) — DP=2 never compiled
   at the base.

2. **The ported commits made DP=2 *compile* but not *correct*.** At the tip DP=2
   now fits TMEM and launches, but the kernel is wrong: **illegal memory access**
   under fast/racy timing (#4, #5), **deadlock** under serialized timing
   (memcheck, #6). Fast→OOB / serialized→hang is the signature of a **WS
   producer/consumer barrier ordering bug**, not a plain index overflow (a pure
   OOB would fault under memcheck too).

3. **`TRITON_DISABLE_WSBARRIER_REORDER` is not the trigger.** DP=2 fails
   identically with the wsbarrier-reorder pass on (#5) and off (#4) → the bug is
   in the **DP=2 data-partition path**, not that reorder pass.

4. **`num_warps` is the sole difference between pass and hang at DP=1.** WARPS=8
   passes (#7), WARPS=4 hangs (#8); same shape, everything else equal.
   `HSTU_SELF_AUTOWS_WARPS=4` is intended **only** paired with
   `HSTU_SELF_DQ_REDUCE=1` (dq-reduce adds a reduction partition, so PSM's
   warp-budget check needs the default partition at 4 warps). On the plain RMW fwd
   path, 4 warps is off-envelope.

5. `P2427737220` combined **two** off-envelope knobs (`DP=2` **and** `WARPS=4`),
   each independently broken.

6. **The regroup did not regress the tested path** — the official DP=1 autoWS test
   passes identically at base (#3) and tip (#9).

## TTGIR comparison (tip, DP=1, WARPS 8 vs 4)

Dumped via `TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=... TRITON_ALWAYS_COMPILE=1`.

Both: one `ttg.warp_specialize`, 4 partitions `[epilogue, gemm, load, computation]`,
`requestedRegisters = [24, 24, -1]`. Differences:

| | WARPS=8 (PASS) | WARPS=4 (HANG) |
|---|---|---|
| `ttg.num-warps` | 8 | 4 |
| computation partition (`partition2`) | `num_warps(8)` | `num_warps(4)` |
| `qk` staging buffer | `memdesc<2x128x128xbf16>` (double-buffered) | `memdesc<3x128x128xbf16>` (triple-buffered) |
| `async_task` ops | 328 | 344 |
| `memdesc` ops | 202 | 216 |
| total lines | 819 | 847 |

Going 8→4 warps also flips the pipeliner to a **deeper (3-slot) `qk` buffer** with a
**half-width consumer** — a producer/consumer slot-accounting mismatch that is the
likely deadlock source (barrier arrive/wait counts get materialized from these in
the WS→LLVM lowering, `lib/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.cpp`).

## Experiment: low-overhead WS lowering (PR #2054, commit `2054eb494623`) on DP=2

**Question:** does the low-overhead single-WS lowering fix the DP=2 fwd failure?

The low-overhead lowering is compiled into the build (`2054eb494623` is an ancestor
of HEAD) but is gated on `hasSingleWarpSpecialize(module)`, which is only set by TLX
`Fixup.cpp` for `tlx.async_tasks(exclusive=True)` kernels — never for meta-WS. To
try it on the meta-WS HSTU kernel I added an **experimental env-gated hook**:

- `include/triton/Tools/Sys/GetEnv.h` — register `TRITON_WS_FORCE_SINGLE`.
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp`
  (`runOnOperation`, start): if `TRITON_WS_FORCE_SINGLE` is set and the module has
  exactly one `ttg.warp_specialize` op, call `setHasSingleWarpSpecialize(mod, true)`
  before the WS→LLVM lowering runs.

**Prerequisite — does DP=2 even warp-specialize?** Depends on `num_warps`:

| DP | WARPS | `ttg.warp_specialize` ops | Notes |
|----|-------|---------------------------|-------|
| 2 | 8 | **0** | meta-WS **declines** → falls back to non-WS (SIMT + pipeline). Low-overhead lowering is inapplicable (nothing to lower). |
| 2 | 4 | **1** (412 async_task refs) | warp-specializes → single-WS, hook **fires**. |

So the low-overhead path is only reachable for the **DP=2 WARPS=4** config (which is
exactly the original paste `P2427737220`).

**Result — hook fires, lowering changes, bug persists:**

| DP=2 WARPS=4 | PTX lines | `setmaxnreg` (reg realloc) | `bar.warp` | Runtime |
|---|---|---|---|---|
| force **OFF** (default lowering) | 9434 | 14 | 123 | ❌ hang / IMA |
| force **ON** (low-overhead) | 9362 | **8** | **114** | ❌ hang / IMA |

The PTX diff confirms the low-overhead lowering **did activate** (register-realloc
`setmaxnreg` 14→8 and ~72 fewer PTX lines — exactly PR #2054's removal of
per-region register realloc + bar.sync). But the kernel **still deadlocks/IMAs**.

**Conclusion:** the low-overhead WS lowering does **not** fix the DP=2 failure. The
bug is therefore **not** in the WS→LLVM lowering (which #2054 replaces); it is
**upstream** — in the DP=2 data-partition / barrier scheduling that produces the WS
structure itself. Confirmed independently by: DP=2 fails identically with the
wsbarrier-reorder pass on/off, and with default vs low-overhead lowering.

## Root-cause: `tt.print` + IR override, and cuda-gdb (DP=2 WARPS=4)

Technique per `68498f2c2` (MetaMain2) / `DebuggingAccuracyAndDeadlocks.md`
§"Locate a hang with `tt.print` + IR override".

**Setup:** dump the DP=2 WARPS=4 final TTGIR (it *does* warp-specialize: 1 WS op,
5 partitions), keep only `.ttgir` in the override dir, insert `tt.print` markers,
re-run with `TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=... TRITON_ALWAYS_COMPILE=1`.
Markers: `prologue-before-tmem` / `prologue-after-tmem` (bracketing the shared
`ttng.tmem_alloc` block at TTGIR lines 172-176) and `gemm-loop-iter` (task-1 loop).

**Hang face — `tt.print` result:**
- `WSDBG prologue-before-tmem` — **printed** (flushed, 32+ times across CTAs)
- `WSDBG prologue-after-tmem` — **never printed**
- `WSDBG gemm-loop-iter` — **never printed**

→ the kernel stalls **inside the shared-prologue `ttng.tmem_alloc` / `tcgen05.alloc`
block** (between the two markers), before any partition loop. This is the classic
**TMEM-oversubscription stall**: DP=2 allocates 4× `1x128x128xf32` TMEM buffers
(`acc_0`,`acc_1`,`qk_0`,`qk_1` = 512 cols, at the 512 hw limit) and `tcgen05.alloc`
spins waiting for columns that never free.

**IMA face — cuda-gdb `info cuda warps` (non-perturbing cross-check):** running the
repro under cuda-gdb, the timing-shifted run instead **faulted**:
```
CUDA Exception: Warp Out-of-range Address
triggered at triton_hstu_attention.py:1259  (tl._experimental_descriptor_store)
  inlined from :1327 ; block (0,2,0), _hstu_attn_fwd<<<(1,8,1),(512,1,1)>>>
```
The OOB is the **output-epilogue TMA store** (lines 1249-1266): the descriptor is
built with `load_size=[BLOCK_M, BLOCK_D_V]` / `global_size=[seq_end_q, H*DimV]` and
stored at row `seq_start_q + pid*BLOCK_M`. Under DP=2 `BLOCK_M = 128*DP = 256` and
the accumulator is split across the 2 computation partitions (tasks 3,4); the
epilogue that reassembles/stores that DP-split 256-row `acc` computes an
out-of-range address.

**Reconciling the two faces (they are NOT one run):** the epilogue store and the
prologue `tmem_alloc` execute on the **same warp group**. The store (src 1259)
lowers to `ttng.async_tma_copy_local_to_global` at ttgir lines 259/265,
`async_task_id = 0`, **inside the `default {}` region**; the `tmem_alloc` block
(172-176) and the inserted prints are root-scope, *before* `ttg.warp_specialize`.
In the WS model the **default warp group** runs the root-scope prologue and then the
`default {}` region (task 0 = epilogue); worker warps (tasks 1-4) run only their
partition and touch neither. So `prologue-after-tmem` and the store are sequential
on the same warps — within one run you cannot miss the former yet reach the latter.
The two observations came from **different runs** (timing-dependent) plus:
(a) the `tt.print` insertion **perturbed** the schedule (doc caveat: prints can flip
a WS failure into a different one), stalling the default group earlier at
`tcgen05.alloc`; (b) **per-block divergence** — the 8-block grid has different blocks
at different points (prints came from blocks (0,4,0)/(0,5,0); the fault from (0,2,0)).

**Conclusion (authoritative = cuda-gdb, non-perturbing):** the primary DP=2 defect is
a **data-partition epilogue address bug — an out-of-range output TMA store at
line 1259** (`BLOCK_M=256`, DP-split `acc` stored at `seq_start_q + pid*BLOCK_M`,
`global_size=[seq_end_q, H*DimV]`). The `tt.print` "stall in `tmem_alloc`" is a
**perturbation-induced / secondary** manifestation (4-way TMEM at the 512 limit is a
contributing fragility), not the root cause. It is a DP=2 correctness bug, not a
WS-lowering bug — corroborated by: fails with wsbarrier-reorder on/off, and with
default vs low-overhead WS lowering.

## Can DP be isolated from autoWS? (No — they are inseparable)

Two experiments to test whether the failure is "DP" vs "autoWS":

**1. autoWS OFF, DP set** — `data_partition_factor` is a `tl.range` meta-WS
annotation (kernel lines 1185-1190) paired with `warp_specialize`; with WS off it is
**ignored** and the kernel picks a normal ≤128 tile.

| config | result |
|---|---|
| autoWS=OFF, DP=1 | ✅ rel-L2 2.34e-3 |
| autoWS=OFF, DP=2 | ✅ rel-L2 2.34e-3 (byte-identical) |

→ DP does nothing without autoWS; it cannot be exercised in isolation this way.

**2. "Manual DP": decouple `BLOCK_M=256` from the split factor.** Added a temp knob
`HSTU_SELF_DP_FACTOR` (reverted after) so the 256-row tile runs with
`data_partition_factor=1` under autoWS:

| config | result |
|---|---|
| BLOCK_M=256, split=**1** | ❌ **compile error**: `blockM must be 64 or 128 but got 256` (pass `nvgpu-ws-data-partition{num-warp-groups=1}`; the 256-row `tt.dot` accumulator can't be placed in TMEM's 128-row limit) |
| BLOCK_M=256, split=**2** (normal) | 🔒 hang (known DP=2 failure) |

→ **`BLOCK_M=256` *requires* the DP=2 split** (num-warp-groups=2) to legalize the
256-row MMA into two 128-row TMEM tiles. The tile size and the DP transform are
inseparable by construction.

**Answer:** you cannot test "DP without autoWS" (annotation inert) nor "the 256 tile
without the DP split" (hard TMEM compile error). The failure is intrinsic to the
**meta-WS DP=2 transform path** — the only path that produces a compilable 256-tile,
and the one that is runtime-broken. The correctness reference for this 2-group
tiling is the hand-written **TLX kernel** (manual DP), which is what the
`test_self_attention_bwd.py` triton-vs-TLX cross-check validates.

## Manual data-partition — FA-fwd style (WORKS **and** warp-specializes)

Two manual-DP shapes were tried:

**Attempt 1 (rejected): unrolled tiles.** A `tl.static_range` loop in the entry ran N
native BLOCK_M(=128) tiles per program (needed `num_stages=1` to dodge the SWP
pipeliner's `'tt.descriptor_load' op ... doesn't know how to predicate` error). It
was numerically correct but `num_stages=1` **suppressed WS** → 0 `ttg.warp_specialize`
ops (plain SIMT), and it reloaded K/V per group. Not a real data partition; dropped.

**Attempt 2 (kept): FA-fwd split-M DP**, mirroring
`fused_attention_ws_device_tma_dp.py`. Split BLOCK_M into two halves *within one
program*; load each KV block **once** and feed both halves' MMAs (shared K/V); two
accumulators; two output stores. Gated by env `HSTU_SELF_FA_DP` (default 0 = off).

Implementation (`triton_hstu_attention.py`):
- `_hstu_attn_fwd_subtile`: per-half compute (qk = q@kᵀ, mask, activation, acc+=act@v)
  taking a **pre-loaded** k/v (shared across the two halves).
- `_hstu_attn_fwd_compute_dp`: TMA-only split-M path — `offs_m0/offs_m1`, `q0/q1`,
  `acc0/acc1`; one warp-specialized KV loop (`data_partition_factor=1`) that loads
  k/v once and calls the subtile twice; two `descriptor_store`s (rows [start_m,·) and
  [start_m+BLOCK_M/2,·)).
- entry `_hstu_attn_fwd` dispatches: `if _HSTU_SELF_FA_DP: _hstu_attn_fwd_compute_dp(...)
  else: _hstu_attn_fwd_compute(...)` (both variants kept).
- Config: use `HSTU_SELF_DP=2` for the BLOCK_M=256 config (→ two 128-row halves);
  `HSTU_SELF_AUTOWS_WARPS=4` (at W8 the WS budget overflows → SIMT fallback, same as
  compiler DP).

**Result** (`HSTU_SELF_AUTOWS=1 HSTU_SELF_DP=2 HSTU_SELF_AUTOWS_WARPS=4
HSTU_SELF_FA_DP=1`): **correct, no hang/OOB, and genuinely warp-specialized** — TTGIR
has 1 `ttg.warp_specialize` op with partitions `[epilogue, gemm, load, computation,
computation]` (2 MMA groups), 4 MMAs, 4 TMEM allocs.

| shape | FA-DP (W4) | warp-specialized? |
|---|---|---|
| L=256 Z=4 | ✅ rel-L2 2.344e-03 | ✅ **yes** (1 WS op, 5 partitions) |
| L=512 Z=2 | ✅ rel-L2 2.349e-03 | ✅ **yes** |

**Both variants coexist**, selected by `HSTU_SELF_FA_DP`:
- `=0` (default): original single-tile `_hstu_attn_fwd_compute` — DP=1/W8 warp-specializes, unchanged.
- `=1`: FA-style split-M `_hstu_attn_fwd_compute_dp` — DP=2/W4 warp-specialized 2-group.

So the FA-style manual DP is a working, warp-specialized 2×128 partition that shares
K/V — the correct alternative to the broken compiler `data_partition_factor=2` path.

**Why "DP without autoWS" and "compiler DP not gated on autoWS" are dead ends:**
DP without WS is inert (annotation ignored); and DP *is* a warp-group split, so a
256-row tile with compiler split=1 is a hard compile error (`blockM must be 64 or 128
but got 256`). The compiler split cannot be exercised outside the WS pipeline. The
FA-style manual split is the viable path.

## Notes on the WS→LLVM lowering path

- Low-overhead single-WS lowering (commit `2054eb494623`, `#2054`) is gated on
  `hasSingleWarpSpecialize(module)` — a module attr set **only** in TLX
  `Fixup.cpp` when a WS op carries `tlx.exclusive` (from
  `tlx.async_tasks(exclusive=True)`), there is exactly one WS op, and target is
  NVIDIA. **Not on by default**; `exclusive` defaults to `False`. The HSTU tutorial
  does not set it, so all cases above use the **original** (robust multi-WS)
  lowering.
