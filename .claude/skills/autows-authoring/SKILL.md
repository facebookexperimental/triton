---
name: autows-authoring
description: >
  Author Triton kernels with automatic warp specialization (AutoWS). Use when
  writing new AutoWS kernels, adding warp_specialize=True to tl.range loops,
  choosing tl.range kwargs and JIT options, debugging why WS was not applied,
  or structuring a kernel to work with both Meta WS and upstream OAI Triton.
  Covers GEMM and Flash Attention patterns on Hopper and Blackwell.
---

# AutoWS Kernel Authoring Guide

AutoWS is Meta's compiler-driven warp specialization for Triton kernels. Instead
of manually writing producer/consumer partitions with TLX primitives (barriers,
`tlx.async_tasks`, `local_alloc`), you annotate `tl.range()` with
`warp_specialize=True` and the compiler automatically partitions ops, inserts
barriers, allocates SMEM/TMEM buffers, and handles multi-buffering.

**Minimal recipe:**
1. Set `TRITON_USE_META_WS=1` (or `triton.knobs.nvidia.use_meta_ws = True`)
2. Use `tl.range(..., warp_specialize=True)` on your loop
3. Pass `num_warps >= 4` at launch

Related skills: `autows-testing` (run tests), `ir-debugging` (IR dumps),
`autows-docs` (compiler internals).

---

## Key Authoring Rules

1. **Prefer a persistent kernel — give AutoWS a loop to specialize.** AutoWS
   pipelines work across *loop iterations* (the producer loads tile N+1 while
   the consumers compute tile N), so a kernel with no loop gives the compiler
   nothing to specialize and WS is silently stripped. Single-shot
   pointwise / memory-bound ops — e.g. a fused SwiGLU forward that handles one
   row-block per program — must be restructured into a **persistent** kernel
   (`grid = #SMs`; each program loops over output tiles with
   `tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True)`) so the
   load→compute→store pipeline spans iterations. This applies even when the
   per-tile body has no reduction loop: the persistent tile loop is itself the
   loop AutoWS specializes.

2. **Place `warp_specialize=True` on the outermost/persistent loop.** For
   persistent kernels, annotate the tile loop
   (`tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True)`), not the
   inner K-reduction loop. The inner loop uses plain `range()`. For
   non-persistent kernels, annotate the main compute loop.

3. **Use TMA loads.** AutoWS partitions loads into a producer warp group and
   compute into consumer warp groups. This partitioning is most effective with
   TMA descriptor loads (`a_desc.load(...)`) rather than pointer-based
   `tl.load()`. TMA enables async bulk copies that the producer can issue
   independently while consumers compute. All reference kernels use
   `TensorDescriptor` + `desc.load()`/`desc.store()`.

4. **Memory allocation and partition scheduling kwargs are Meta-only.** The
   `tl.range()` kwargs for controlling memory allocation (`smem_alloc_algo`,
   `tmem_alloc_algo`, `smem_budget`, `smem_circular_reuse`) and partition
   scheduling (`merge_epilogue`, `merge_correction`,
   `merge_epilogue_to_computation`, `separate_epilogue_store`) are consumed
   exclusively by Meta's WS passes. They do not exist in OSS Triton's
   `tl.range()` — see [OSS Fallback](#oss-triton-fallback) for how to gate them.

---

## Enabling AutoWS

**Environment variable (recommended for running kernels from the command line):**
```bash
TRITON_USE_META_WS=1 python my_kernel.py
```
Also add `TRITON_USE_META_WS=1` to the kernel script's module docstring so
users know it's required:
```python
"""
My AutoWS GEMM kernel.

Usage:
    TRITON_USE_META_WS=1 python my_gemm.py
"""
```

**Programmatic (recommended for correctness tests only):**
```python
with triton.knobs.nvidia.scope():
    triton.knobs.nvidia.use_meta_ws = True
    # ... launch kernel ...
```
Use `triton.knobs.nvidia.scope()` in correctness tests so the knob is
automatically restored after the test, preventing state leakage between test
cases. Do NOT use `scope()` for actual runtime/benchmark scripts — use the
`TRITON_USE_META_WS=1` env var instead.

`TRITON_USE_META_WS` is a cache-invalidating env var (listed in
`include/triton/Tools/Sys/GetEnv.hpp`), meaning changing it forces
recompilation.

### Additional Environment Variables

| Env Var | Default | Purpose |
|---------|---------|---------|
| `TRITON_USE_META_WS` | `False` | Master switch for Meta WS vs upstream OAI WS |
| `TRITON_DISABLE_WSBARRIER_REORDER` | `False` | Disable WS barrier reordering |
| `TRITON_ENABLE_INTERLEAVE_TMEM` | `True` | Interleave TMEM pass (Blackwell) |

Source: `python/triton/knobs.py` lines 502-534

---

## `tl.range()` Kwargs Reference

Defined in `python/triton/language/core.py` (`tl.range.__init__`).

### Core

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `warp_specialize` | bool | `False` | Enable AutoWS on this loop |
| `flatten` | bool | `False` | Loop flattening for persistent kernels. **WARNING:** `flatten=True` currently does NOT warp-specialize — the kernel runs but skips WS |
| `data_partition_factor` | int/None | `None` | Split work across N data partitions. `None`/1 = no split, 2 = splits BLOCK_M in half. Requires sufficient BLOCK_SIZE_M (256 for Blackwell dp=2, 128 for Hopper dp=2) |

### Memory Allocation (Meta WS only)

These kwargs control how the compiler allocates SMEM/TMEM buffers. They are
consumed by `WSMemoryPlanner.cpp` via loop attributes.

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `smem_alloc_algo` | int/None | `None` | SMEM allocation strategy (0 or 1). Strategy 1 is preferred for FA kernels |
| `tmem_alloc_algo` | int/None | `None` | TMEM allocation strategy (Blackwell only) |
| `smem_budget` | int/None | `None` | Override SMEM budget in bytes |
| `smem_circular_reuse` | bool/None | `None` | Enable circular reuse of SMEM buffers |

### Partition Scheduling (Meta WS only)

These kwargs control how `PartitionSchedulingMeta` assigns ops to partitions.
They override pass-level defaults (all false) via per-loop attributes, read at
`PartitionSchedulingMeta.cpp` lines 2614-2629.

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `merge_epilogue` | bool | `False` | Merge epilogue ops into the computation/correction/reduction partition |
| `merge_correction` | bool | `False` | Merge softmax correction ops into the computation partition |
| `merge_epilogue_to_computation` | bool | `False` | Merge epilogue ops directly to the computation partition |
| `separate_epilogue_store` | bool | `False` | Separate epilogue store ops into their own 1-warp partition |

---

## JIT-Level Options

Passed at kernel launch time. Defined in `CUDAOptions` at
`third_party/nvidia/backend/compiler.py` lines 145-180.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_warps` | int | 4 | Total warps. Must be >= 4 and power of 2 for WS |
| `num_stages` | int | 3 | Pipeline depth / multi-buffer count |
| `minRegAutoWS` | int | 24 | Min registers for non-tensor partitions. Divisible by 8 |
| `maxRegAutoWS` | int/None | None | Max registers for tensor partitions. Divisible by 8 |
| `pingpongAutoWS` | bool | False | Enable ping-pong barriers between two consumer partitions |
| `early_tma_store_lowering` | bool | False | Lower TMA stores before partition scheduling |

### Register Budget (`minRegAutoWS` / `maxRegAutoWS`)

These control how the 64K hardware register file is divided across
warp-specialized partitions. Each partition runs a subset of warps and can have
a different register cap (emitted as PTX `setmaxnreg` instructions).

**Partition types:**
- **Non-tensor partitions** — partitions without MMA/dot ops (typically the TMA
  load producer). These get `minRegAutoWS` registers.
- **Tensor partitions** — partitions with MMA/dot ops (computation, correction,
  reduction). These get `maxRegAutoWS` registers (if set) or split the remainder
  evenly (if not set).
- **Default partition** — runs outside the WS region. Gets leftover registers.

**When `maxRegAutoWS` is NOT set (default):**
Non-tensor partitions get `minRegAutoWS` (24). Tensor partitions AND the default
partition all receive sentinel value `-1`, meaning "split the remaining register
pool evenly after deducting the fixed allocations":
```
regsPerThread = (totalRegs - fixedRegs) / leftoverThreads
```
Computed in `AllocateWarpGroups.cpp` lines 244-280.

**When `maxRegAutoWS` IS set:**
Non-tensor partitions get `minRegAutoWS`. Tensor partitions get `maxRegAutoWS`.
The default partition gets ALL leftover registers.
Computed in `OptimizePartitionWarps.cpp` lines 295-314.

**Both values must be divisible by 8** (`compiler.py:140-142`).

### Checking Register Allocations

To see which partitions map to which register budget and verify the actual
allocation:

1. **IR inspection:** Set `MLIR_ENABLE_DUMP=1`. Look for the
   `ttg.warp_specialize` op which carries:
   - `requestedRegisters = array<i32: ...>` — what `OptimizePartitionWarps` requested
   - `actualRegisters = array<i32: ...>` — what `AllocateWarpGroups` computed
   - Array order: `[default_partition, partition_0, partition_1, ...]`
   - A `-1` in `requestedRegisters` means "split evenly"
   - Example: `requestedRegisters = array<i32: 24, -1, 24>` means load partition
     gets 24, computation splits leftovers, epilogue store gets 24

2. **PTXAS log:** `TRITON_DUMP_PTXAS_LOG=1` prints ptxas verbose output showing
   register usage.

3. **PTX inspection:** `kernel.asm['ptx']` — search for
   `setmaxnreg.inc.sync.aligned` and `setmaxnreg.dec.sync.aligned` to see
   register reallocation at partition boundaries.

---

## Kernel Patterns

### GEMM (K-loop WS)

Warp-specialize the inner K-reduction loop. Based on `matmul_kernel_tma_ws` in
`python/test/unit/language/test_tutorial09_warp_specialization.py` lines 34-94.

```python
"""
GEMM with AutoWS (K-loop warp specialization).

Usage:
    TRITON_USE_META_WS=1 python my_gemm.py
"""

@triton.jit
def matmul_kernel_ws(a_desc, b_desc, c_desc, M, N, K,
                     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                     BLOCK_SIZE_K: tl.constexpr,
                     DATA_PARTITION_FACTOR: tl.constexpr,
                     SMEM_ALLOC_ALGO: tl.constexpr,
                     SEPARATE_EPILOGUE_STORE: tl.constexpr, ...):
    # ... pid computation ...
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(
            k_tiles,
            warp_specialize=True,
            data_partition_factor=DATA_PARTITION_FACTOR,
            smem_alloc_algo=SMEM_ALLOC_ALGO,
            separate_epilogue_store=SEPARATE_EPILOGUE_STORE,
    ):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c_desc.store([offs_cm, offs_cn], accumulator.to(dtype))
```

### Persistent GEMM (tile-loop WS)

Warp-specialize the outer persistent loop. Inner K-loop uses plain `range()`.
Based on `matmul_kernel_tma_persistent_ws` in same file, lines 102-167.

```python
"""
Persistent GEMM with AutoWS (tile-loop warp specialization).

Usage:
    TRITON_USE_META_WS=1 python my_persistent_gemm.py
"""

@triton.jit
def matmul_persistent_ws(a_desc, b_desc, c_desc, M, N, K,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                         BLOCK_SIZE_K: tl.constexpr, NUM_SMS: tl.constexpr,
                         DATA_PARTITION_FACTOR: tl.constexpr,
                         SMEM_ALLOC_ALGO: tl.constexpr,
                         SEPARATE_EPILOGUE_STORE: tl.constexpr, ...):
    start_pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for tile_id in tl.range(
            start_pid, num_tiles, NUM_SMS,
            warp_specialize=True,                           # on the OUTER loop
            data_partition_factor=DATA_PARTITION_FACTOR,
            smem_alloc_algo=SMEM_ALLOC_ALGO,
            separate_epilogue_store=SEPARATE_EPILOGUE_STORE,
    ):
        # ... compute pid_m, pid_n from tile_id ...
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):                           # plain range()
            a = a_desc.load([offs_am, ki * BLOCK_SIZE_K])
            b = b_desc.load([offs_bn, ki * BLOCK_SIZE_K])
            accumulator = tl.dot(a, b.T, accumulator)

        c_desc.store([offs_am, offs_bn], accumulator.to(dtype))
```

### Flash Attention (inner-loop WS)

Warp-specialize the KV-iteration loop with partition scheduling hints.
Based on `python/tutorials/fused-attention-ws-device-tma-hopper-or-blackwell.py`
lines 150-155.

```python
for start_n in tl.range(
        lo, hi, BLOCK_N,
        warp_specialize=warp_specialize,
        merge_epilogue=True,
        merge_correction=True,
        smem_alloc_algo=1,
        data_partition_factor=DP_FACTOR,
):
    # TMA descriptor loads for K and V
    k = k_desc.load([start_n, offs_k])
    v = v_desc.load([start_n, offs_d])
    # QK dot product
    qk = tl.dot(q, k)
    # ... softmax ...
    # PV dot product
    acc = tl.dot(p.to(dtype), v, acc)
```

---

## Test & Launch Boilerplate

Correctness tests should use `triton.knobs.nvidia.use_meta_ws = True` (not the
env var) inside a `scope()` block to avoid state leakage between test cases.

Based on `test_tutorial09_matmul_tma_warp_specialize` in
`python/test/unit/language/test_tutorial09_warp_specialization.py` lines 416-492.

```python
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

def test_my_kernel():
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        M, N, K = 8192, 8192, 1024
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        dtype = torch.float16

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device="cuda")
        B = torch.randn((N, K), dtype=dtype, device="cuda")
        C = torch.empty((M, N), dtype=dtype, device="cuda")

        # TMA requires a custom allocator
        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_N, BLOCK_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N])

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"])
                           * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

        # Launch — capture handle to inspect IR
        kernel = matmul_kernel_ws[grid](
            a_desc, b_desc, c_desc, M, N, K,
            BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
            DATA_PARTITION_FACTOR=1, SMEM_ALLOC_ALGO=0,
            SEPARATE_EPILOGUE_STORE=False,
            num_stages=3, num_warps=4,
        )

        # 1. Verify WS was applied
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir

        # 2. Verify correct HW instructions (pick one per arch)
        # Blackwell: assert "ttng.tc_gen5_mma" in ttgir
        # Hopper:    assert "ttng.warp_group_dot" in ttgir
        assert "ttng.async_tma_copy_global_to_local" in ttgir

        # 3. Compare against reference
        ref = torch.matmul(A.float(), B.T.float()).to(dtype)
        torch.testing.assert_close(ref, C, atol=0.03, rtol=0.03)
```

---

## Verifying AutoWS is Working

1. **IR check:** `kernel.asm["ttgir"]` must contain `"ttg.warp_specialize"` —
   confirms WS was applied.

2. **MMA check:** Look for `"ttng.tc_gen5_mma"` (Blackwell) or
   `"ttng.warp_group_dot"` (Hopper) — confirms tensor core usage.

3. **TMA check:** Look for `"ttng.async_tma_copy_global_to_local"` — confirms
   async TMA copies in the producer partition.

4. **Partition check:** Set `MLIR_ENABLE_DUMP=1` and look for
   `ttg.partition = array<i32: N>` attributes in the IR after
   `PartitionSchedulingMeta`. This shows how ops were assigned to partitions.

5. **Full IR dump:** Set `TRITON_KERNEL_DUMP=<kernel_name>` +
   `TRITON_ALWAYS_COMPILE=1` to dump IR at each compilation stage. See
   `ir-debugging` skill for details.

6. **AutoWS vs TLX comparison:**
   ```bash
   TRITON_USE_META_WS=1 python python/tutorials/test_hopper_fwd_autows_vs_tlx.py
   ```

---

## 2-CTA (Multi-CTA) with AutoWS

2-CTA allows two CTAs in a cluster to cooperatively execute a single MMA,
doubling the N dimension of the output tile. This is a **Blackwell-only**,
**Meta WS-only** feature.

### Requirements

1. **`ctas_per_cga=(2, 1, 1)`** — pass this at kernel launch time (not
   `cluster_dims` or `num_ctas`). `ctas_per_cga` is the correct way to enable
   2-CTA because it bypasses PlanCTA's unreliable `CTASplitNum` encoding by
   forcing `num_ctas=1` internally, then using `Transform2CTALoads` +
   `Insert2CTASync` for B-operand splitting and cross-CTA synchronization.

2. **`two_ctas=True` on `tl.dot()`** — tells the compiler this dot product
   should use 2-CTA MMA. The compiler automatically splits the B load across
   CTAs and inserts cross-CTA barriers.

3. **`TRITON_USE_META_WS=1`** — 2-CTA with WS is only supported by Meta's WS
   pipeline. Upstream OAI WS does not support `cluster_dims >= 2`.

4. **`num_stages=1`** — current 2-CTA implementations use 1 pipeline stage.

5. **Grid M dimension must be >= 2** — the grid must launch at least 2 CTAs in
   the cluster dimension: `grid = (max(triton.cdiv(M, BLOCK_M), 2), ...)`.

### Kernel Pattern

```python
"""
2-CTA GEMM with AutoWS.

Usage:
    TRITON_USE_META_WS=1 python my_2cta_gemm.py
"""

@triton.jit
def matmul_2cta_ws_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                          stride_am, stride_ak, stride_bk, stride_bn,
                          stride_cm, stride_cn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N])

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)

    for k in tl.range(0, k_tiles, warp_specialize=True):
        offs_k = k * BLOCK_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, accumulator, two_ctas=True)  # <-- 2-CTA MMA

    c_desc.store([offs_am, offs_bn], accumulator.to(tl.float16))
```

### Launch Pattern

```python
grid = (max(triton.cdiv(M, BLOCK_M), 2), triton.cdiv(N, BLOCK_N))

matmul_2cta_ws_kernel[grid](
    a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
    num_stages=1,
    ctas_per_cga=(2, 1, 1),     # <-- enables 2-CTA cluster
)
```

### What the compiler does

When `ctas_per_cga` is set with `two_ctas=True` on `tl.dot()`:
1. **`Transform2CTALoads`** splits the B-operand load: each CTA loads half of
   BLOCK_N (`[BLOCK_K, BLOCK_N/2]`), offset by `cluster_cta_id * BLOCK_N/2`
2. **`Insert2CTASync`** inserts cross-CTA barriers before the 2-CTA MMA using
   the "arrive remote, wait local" pattern via `mapa` instructions
3. Both CTAs issue the MMA cooperatively

### Reference files

- 2-CTA AutoWS test: `third_party/tlx/tutorials/blackwell-triton-addmm-2cta_test.py`
- TLX 2-CTA GEMM (manual WS): `third_party/tlx/tutorials/blackwell_gemm_2cta.py`
- Design doc: `docs/design/2cta-autoWS-sync.md`
- Transform2CTALoads: `third_party/nvidia/hopper/lib/Transforms/Transform2CTALoads.cpp`
- Insert2CTASync: `third_party/nvidia/hopper/lib/Transforms/Insert2CTASync.cpp`

---

## Restrictions (When WS Bails Out)

If any of these conditions are violated, the compiler silently strips WS
annotations and the kernel runs without specialization.

- **Minimum 4 warps** — `num_warps >= 4` required
  (`WarpSpecialization.cpp:148-153`)
- **No else blocks** — `scf.if` with non-trivial else blocks not supported
  (`WarpSpecialization.cpp:156-170`)
- **Max 16 total warps** — if estimated warp budget exceeds 16, WS is stripped
  (`PartitionSchedulingMeta.cpp:2664-2698`)
- **No distance > 1 loop-carried deps** (`ScheduleLoops.cpp:40-41`)
- **No outer loops** for pipelining eligibility (`ScheduleLoops.cpp:42-43`)
- **No barriers, asserts, or prints** in the WS loop body
  (`ScheduleLoops.cpp:44-47`)
- **Register alignment** — `minRegAutoWS` and `maxRegAutoWS` must be divisible
  by 8 (`compiler.py:140-142`)
- **2-CTA + upstream WS not supported** — only Meta's WS supports
  `cluster_dims >= 2` (`compiler.py:703-706`)
- **`flatten=True` skips WS** — the kernel runs but WS is not applied
- **`data_partition_factor != 1`** requires sufficient `BLOCK_SIZE_M` (256 for
  Blackwell dp=2, 128 for Hopper dp=2)
- **Pointer-typed tensors** should not be cross-partition

---

## OSS Triton Fallback

The basic `tl.range(warp_specialize=True)` syntax works with both Meta WS and
upstream OAI WS. The difference is entirely which compiler passes run, controlled
by `TRITON_USE_META_WS`:

- **Blackwell:** upstream uses `add_warp_specialize`; Meta uses
  `add_partition_scheduling_meta` + `add_hopper_warpspec`
- **Hopper:** upstream uses `add_hopper_warpspec` only (internal
  `doTaskPartition`); Meta runs the full pipeline

**Meta WS-specific features do NOT work with OSS Triton.** The following kwargs
and options are consumed only by Meta's compiler passes. They do not exist in
OSS Triton's `tl.range()` signature — passing them will cause errors, not silent
no-ops. If the kernel must run on both, use completely separate code paths gated
by `tl.constexpr`:

- **Partition scheduling kwargs:** `merge_epilogue`, `merge_correction`,
  `merge_epilogue_to_computation`, `separate_epilogue_store`
- **Memory allocation kwargs:** `smem_alloc_algo`, `tmem_alloc_algo`,
  `smem_budget`, `smem_circular_reuse`
- **Register budget options:** `minRegAutoWS`, `maxRegAutoWS`, `pingpongAutoWS`
- **2-CTA / multi-CTA:** `multi_cta=True` and `cluster_dims >= 2` with WS
- **`early_tma_store_lowering`**

### Dual-mode kernel pattern

Because these kwargs do not exist in OSS Triton's `tl.range()`, you cannot
conditionally pass them (e.g., `smem_alloc_algo=1 if X else None`). You must
use completely separate `tl.range()` calls:

```python
@triton.jit
def _kernel_body(a_desc, b_desc, c_desc, ...):
    # ... shared load/dot/store logic ...

@triton.jit
def my_kernel(..., USE_META_WS: tl.constexpr):
    if USE_META_WS:
        for tile_id in tl.range(
                start_pid, num_tiles, NUM_SMS,
                warp_specialize=True,
                separate_epilogue_store=True,
                smem_alloc_algo=1,
                merge_epilogue=True,
        ):
            _kernel_body(...)
    else:
        for tile_id in tl.range(
                start_pid, num_tiles, NUM_SMS,
                warp_specialize=True,
        ):
            _kernel_body(...)
```

---

## Exhaustive Reference

### All `tl.range()` kwargs (from `python/triton/language/core.py`)

This is the complete list of every kwarg accepted by `tl.range()`. Items marked
**(Meta only)** do not exist in OSS Triton and require separate code paths.

| Kwarg | Type | Default | Available in OSS | Description |
|-------|------|---------|-----------------|-------------|
| `num_stages` | int/None | `None` | Yes | Pipeline depth override at the loop level |
| `loop_unroll_factor` | int/None | `None` | Yes | Loop unroll factor |
| `flatten` | bool | `False` | Yes | Loop flattening for persistent kernels. `True` currently skips WS |
| `warp_specialize` | bool | `False` | Yes | Enable AutoWS on this loop |
| `multi_cta` | bool | `False` | **Meta only** | Enable multi-CTA (2-CTA) mode |
| `disable_licm` | bool | `False` | Yes | Disable loop-invariant code motion |
| `data_partition_factor` | int/None | `None` | **Meta only** | Split work across N data partitions |
| `disallow_acc_multi_buffer` | bool | `False` | **Meta only** | Prevent multi-buffering of accumulators |
| `merge_epilogue` | bool | `False` | **Meta only** | Merge epilogue into computation/correction/reduction partition |
| `merge_epilogue_to_computation` | bool | `False` | **Meta only** | Merge epilogue directly to computation partition |
| `merge_correction` | bool | `False` | **Meta only** | Merge softmax correction into computation partition |
| `separate_epilogue_store` | bool | `False` | **Meta only** | Separate epilogue store into its own 1-warp partition |
| `tmem_alloc_algo` | int/None | `None` | **Meta only** | TMEM allocation strategy (Blackwell only) |
| `smem_alloc_algo` | int/None | `None` | **Meta only** | SMEM allocation strategy (0 or 1) |
| `smem_budget` | int/None | `None` | **Meta only** | Override SMEM budget in bytes |
| `smem_circular_reuse` | bool/None | `None` | **Meta only** | Enable circular reuse of SMEM buffers |

### All AutoWS-relevant environment variables (from `python/triton/knobs.py`)

| Env Var | Knob | Type | Default | Description |
|---------|------|------|---------|-------------|
| `TRITON_USE_META_WS` | `knobs.nvidia.use_meta_ws` | bool | `False` | Master switch: Meta WS vs upstream OAI WS |
| `TRITON_DISABLE_WSBARRIER_REORDER` | `knobs.nvidia.disable_wsbarrier_reorder` | bool | `False` | Disable WS barrier reordering |
| `TRITON_ENABLE_INTERLEAVE_TMEM` | `knobs.nvidia.enable_interleave_tmem` | bool | `True` | Interleave TMEM pass (Blackwell) |
| `TRITON_DUMP_PTXAS_LOG` | `knobs.nvidia.dump_ptxas_log` | bool | `False` | Print ptxas verbose output (register usage) |
| `MLIR_ENABLE_DUMP` | — | bool | `False` | Dump MLIR IR after each pass (for inspecting partitions) |
| `TRITON_KERNEL_DUMP` | — | str | unset | Dump IR at each stage for the named kernel |
| `TRITON_ALWAYS_COMPILE` | `knobs.compilation.always_compile` | bool | `False` | Force recompilation (useful with IR dumps) |

### All AutoWS-relevant JIT-level options (from `CUDAOptions`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_warps` | int | 4 | Total warps per CTA. Must be >= 4 and power of 2 for WS |
| `num_stages` | int | 3 | Pipeline depth / multi-buffer count |
| `minRegAutoWS` | int | 24 | Registers for non-tensor partitions. Must be divisible by 8 |
| `maxRegAutoWS` | int/None | None | Registers for tensor partitions. Must be divisible by 8 |
| `pingpongAutoWS` | bool | False | Ping-pong barriers between two consumer partitions |
| `early_tma_store_lowering` | bool | False | Lower TMA stores before partition scheduling |

---

## Reference Files

- `tl.range` definition: `python/triton/language/core.py` (lines 3454-3484)
- Knobs: `python/triton/knobs.py` (line 516)
- Backend options: `third_party/nvidia/backend/compiler.py` (lines 145-180)
- Compiler pipeline: `third_party/nvidia/backend/compiler.py` (lines 659-716)
- GEMM test: `python/test/unit/language/test_tutorial09_warp_specialization.py`
- AddMM test: `python/test/unit/language/test_autows_addmm.py`
- FA test: `third_party/tlx/tutorials/testing/test_correctness_autows.py`
- FA kernel: `third_party/tlx/tutorials/fused_attention_ws_device_tma.py`
- FA kernel (DP): `third_party/tlx/tutorials/fused_attention_ws_device_tma_dp.py`
- AutoWS vs TLX: `python/tutorials/test_hopper_fwd_autows_vs_tlx.py`
- LIT tests: `test/Hopper/WarpSpecialization/`
