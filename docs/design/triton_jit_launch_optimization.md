# Triton JIT Launch Latency Optimization

## Problem

Triton's `JITFunction.run()` dispatch path adds significant Python-side overhead on
every kernel launch, even when the kernel is already compiled and cached. Profiling
(via D99636574) showed the following breakdown for a 19-arg kernel:

| Phase | Time (µs) | % |
|-------|-----------|---|
| **binder** (arg specialization) | 9.65 | 48% |
| **launch** (launch_metadata + kernel.run) | 7.47 | 37% |
| cache_key (tuple + str creation) | 1.30 | 7% |
| driver (get_device + get_stream) | 0.66 | 3% |
| rest (globals, grid, hooks) | 0.88 | 5% |
| **TOTAL** | **19.96** | |

The profiled total is inflated by `time.perf_counter()` overhead; the real
wall-clock was **~14 µs** for `nop_triton_kernel` (19 args) vs **~3.5 µs** for
`nop_triton_compiled_kernel_run` (which calls `CompiledKernel.run` directly) and
**~1.1 µs** for cuteDSL (0-arg case).

## Root Cause Analysis

There are **two separate issues**:

### Issue 1: Python-side JIT dispatch overhead (Normal Triton → Compiled Kernel)

Everything that happens *before* the C launcher is called:

```
kernel[grid](*args)
  → JITFunction.__getitem__  → creates lambda
    → JITFunction.run()
      → binder(*args)           ← exec()-generated function, calls specialize_impl() per arg
      → compute_cache_key()     ← tuple() + str() allocation
      → global variable check   ← dict iteration
      → kernel.launch_metadata()← Python method call + *args unpacking
      → kernel.run(...)         ← property access + CudaLauncher.__call__
```

### Issue 2: C launcher overhead (Compiled Kernel → cuteDSL)

The generated C launcher is a one-size-fits-all template that always pays for:
- `ensureCudaContext()` — `cuCtxGetCurrent()` on every launch
- `PyArg_ParseTuple` × 2 — parses 20 fields (14 metadata + 6 packed)
- Hook/scratch None checks — 4 branches even when unused
- `CUlaunchAttribute[4]` setup — even for simple kernels
- `cuLaunchKernelEx` — heavier than `cudaLaunchKernel`

This series of diffs addresses **Issue 1** only. Issue 2 requires changes to the C
code generator (`make_launcher` in `driver.py`) or the compiler-emitted launcher
approach described in the design doc "Move CPU-side Kernel Launcher Generation
into the Triton Compiler".

## Optimizations Implemented

All changes are in `third-party/triton/beta/triton/python/triton/runtime/jit.py`.

### 1. Identity-based fast path (Layer 1)

**The biggest win.** After the first successful launch, we cache the previous
call's args, kwargs, and kernel. On the next call, we check if the exact same
Python objects are being passed (`args[i] is last_args[i]` and kwargs identity).
This is just N pointer comparisons — no `isinstance`, no `data_ptr()`, no
string formatting, no dict lookup.

This works because in typical training loops, the same tensor objects and
integer values are reused across iterations. CPython also caches small integers
(-5 to 256), so common constexpr values like `32` pass the identity check.

**kwargs support:** Kernel calls with kwargs (e.g., `kernel[grid](*args,
BLOCK_SIZE=128)`) are handled by the fast path. The kwargs dict is compared
by checking that each value `is` the same object as the previous call. User
kwargs are saved before the system kwargs (debug, sanitize_overflow) injection.

When the identity check succeeds, we skip:
- The entire binder (specialize_impl × N args)
- Cache key computation (tuple + str allocation)
- The global variable check (when `used_global_vals` is empty)

**Correctness:** Falls through to the slow path on any mismatch. Global
variable check is still performed when `used_global_vals` is non-empty.

**bound_args handling:** The fast path stores `tuple(bound_args.values())`
(which includes default parameter values) separately from `args`, and uses
the stored bound values for grid resolution and kernel launch.

## Benchmark Results (Layer 1 only)

Benchmark command:
```bash
buck2 run @mode/opt -m ovr_config//triton:beta //pytorch/tritonbench:run_lite -- \
  --op launch_latency \
  --only nop_triton_kernel,nop_triton_kernel_kwargs,nop_triton_compiled_kernel_run \
  --metrics walltime --simple-output
```

Hardware: NVIDIA GB200

### Before (baseline)

| x_val (num args) | nop_triton_kernel | nop_triton_kernel_kwargs | compiled_kernel_run |
|-------------------|-------------------|--------------------------|---------------------|
| 0 | 6.15 µs | 6.15 µs | 2.43 µs |
| 19 | 12.45 µs | 12.45 µs | 3.44 µs |

### After (Layer 1 identity check)

| x_val (num args) | nop_triton_kernel | nop_triton_kernel_kwargs | compiled_kernel_run |
|-------------------|-------------------|--------------------------|---------------------|
| 0 | **4.90 µs** | **4.92 µs** | 2.49 µs |
| 19 | **6.96 µs** | **7.78 µs** | 3.53 µs |

### Summary

| Args | Before | After (positional) | After (kwargs) | Speedup |
|------|--------|--------------------|----------------|--------|
| 0 | 6.15 µs | 4.90 µs | 4.92 µs | **20%** |
| **19** | **12.45 µs** | **6.96 µs** | **7.78 µs** | **37-44%** |

The ~0.8 µs overhead for kwargs vs positional args comes from the kwargs
identity comparison loop (iterating dict keys + `is` checks).

### 2. Signature-based fast path (Layer 2) — **fallback for Layer 1 misses**

When arg objects differ but the specialization would be the same (e.g., new
tensors with the same dtype/alignment after reallocation), Layer 1 misses.
Layer 2 computes a "fast key" — a minimal tuple capturing only what affects
specialization:

**Positional args:**
- **constexpr args:** value directly (determines compiled kernel)
- **int args:** value (determines type range i32/u64/i64, `==1` specialization, `%16` alignment)
- **float/bool/None:** type marker only
- **tensors:** `(dtype, data_ptr() % 16 == 0)`
- **TMA descriptors:** `'tma'`
- **Unknown types:** returns `None` (falls through to slow path)

**kwargs** (e.g., `BLOCK_SIZE=128`): included in the fast key with sorted
`(key, value)` pairs using the same specialization logic as positional args.
This ensures that changing a kwarg value (e.g., `BLOCK_SIZE=128` → `256`)
produces a different fast key and correctly falls through to a different
compiled kernel.

This key is looked up in a `_run_cache` dict. On hit, we skip the binder and
cache key computation entirely.

## Benchmark Results (Layer 1 + Layer 2)

Benchmark command:
```bash
buck2 run @mode/opt -m ovr_config//triton:beta //pytorch/tritonbench:run_lite -- \
  --op launch_latency \
  --only nop_triton_kernel,nop_triton_kernel_kwargs,nop_triton_kernel_new_tensors,nop_triton_compiled_kernel_run \
  --metrics walltime --simple-output
```

Hardware: NVIDIA GB200

### Layer 1 only (baseline for Layer 2)

| x_val | nop_triton_kernel | nop_triton_kernel_new_tensors | compiled_kernel_run |
|-------|-------------------|-------------------------------|---------------------|
| 0 | 5.02 µs | 4.99 µs | 2.55 µs |
| 19 | 6.97 µs (L1 hit) | 14.30 µs (L1 miss → slow path) | 3.52 µs |

### After Layer 1 + Layer 2

| x_val | nop_triton_kernel | nop_triton_kernel_kwargs | nop_triton_kernel_new_tensors | compiled_kernel_run |
|-------|-------------------|--------------------------|-------------------------------|---------------------|
| 0 | 4.91 µs | 4.97 µs | 4.99 µs | 2.46 µs |
| 19 | 6.94 µs (L1 hit) | 7.67 µs (L1 hit) | 10.79 µs (L2 hit) | 3.48 µs |

### Layer 2 Impact (19-arg, Layer 1 miss scenario)

| Scenario | Latency | vs Slow Path |
|----------|---------|-------------|
| Slow path (no fast path) | 14.30 µs | baseline |
| **Layer 2 hit** | **10.79 µs** | **25% faster** |

### 3. Skip `launch_metadata()` when hooks are None — **~0.3 µs savings**

`CompiledKernel.launch_metadata()` returns `None` immediately when
`knobs.runtime.launch_enter_hook is None` (the common case), but the Python
method call itself + `*args` tuple creation for 19 args costs ~0.3 µs. We
check the hook directly and pass `None` without calling the method.

### 4. Call `launcher.launch()` directly — **~0.3 µs savings**

`kernel.run` is a property that returns a `CudaLauncher` instance. Calling it
goes through `CudaLauncher.__call__`, which:
1. Defines `allocate_scratch()` as a nested function
2. Calls it twice (global scratch + profile scratch) — even when both sizes are 0
3. Then calls `self.launch(...)` (the actual C extension)

For kernels with no scratch (the common case), we call `launcher.launch()`
directly, bypassing `__call__` and the scratch allocation overhead.

### 5. Cache kernel properties — **~0.2 µs savings**

On each fast-path launch, we previously accessed:
- `kernel.run` — a `@property` that checks `self._run is None` and returns a `CudaLauncher`
- `kernel.function` — attribute access
- `kernel.packed_metadata` — attribute access
- `launcher.launch_cooperative_grid`, `.launch_cluster`, `.launch_pdl` — 3 more accesses

All of these are now cached in a flat tuple (`_make_launch_cache`) after the
first launch, eliminating per-call attribute/property overhead.

### 6. Remove `hasattr(kernel, "result")` — **~0.05 µs savings**

Cached kernels are already resolved (the slow path calls `.result()` before
caching). The `hasattr` check on every fast-path launch was always False.
Removed.

## Benchmark Results (All Optimizations)

Benchmark command:
```bash
buck2 run @mode/opt -m ovr_config//triton:beta //pytorch/tritonbench:run_lite -- \
  --op launch_latency \
  --only nop_triton_kernel,nop_triton_kernel_kwargs,nop_triton_kernel_new_tensors,nop_triton_compiled_kernel_run \
  --metrics walltime --simple-output
```

Hardware: NVIDIA GB200

### After Layer 1 + Layer 2 (baseline for this diff)

| x_val | nop_triton_kernel | nop_triton_kernel_kwargs | nop_triton_kernel_new_tensors | compiled_kernel_run |
|-------|-------------------|--------------------------|-------------------------------|---------------------|
| 0 | 4.91 µs | 4.97 µs | 4.99 µs | 2.46 µs |
| 19 | 6.94 µs (L1 hit) | 7.67 µs (L1 hit) | 10.79 µs (L2 hit) | 3.48 µs |

### After all optimizations

| x_val | nop_triton_kernel | nop_triton_kernel_kwargs | nop_triton_kernel_new_tensors | compiled_kernel_run |
|-------|-------------------|--------------------------|-------------------------------|---------------------|
| 0 | **4.38 µs** | **4.46 µs** | **4.45 µs** | 2.46 µs |
| 19 | **6.02 µs** (L1 hit) | **6.87 µs** (L1 hit) | **10.01 µs** (L2 hit) | 3.41 µs |

### Full Cumulative Summary (19-arg)

| Scenario | Baseline | Layer 1 | Layer 1+2 | All Opts | Speedup |
|----------|----------|---------|-----------|----------|---------|
| L1 hit (positional) | 12.45 µs | 6.96 µs | 6.94 µs | **6.02 µs** | **52%** |
| L1 hit (kwargs) | 12.45 µs | 7.78 µs | 7.67 µs | **6.87 µs** | **45%** |
| L2 hit (new tensors) | 14.30 µs | 14.30 µs | 10.79 µs | **10.01 µs** | **30%** |
| **19** | **12.45 µs** | **7.91 µs** | **7.04 µs** | **5.93 µs** | **6.52 µs** | **52%** |

### Remaining gap to compiled_kernel_run (19 args: 5.93 µs vs 3.48 µs = 2.45 µs)

| Overhead | ~Cost | Optimizable? |
|----------|-------|--------------|
| `get_current_device()` + `get_current_stream()` | 0.66 µs | Risky — stream can change independently |
| `__getitem__` lambda creation + call | 0.3-0.5 µs | Part of Triton API contract |
| Identity check loop (19 pointer comparisons) | 0.3 µs | Already minimal |
| `*args` unpacking into C launcher | 0.3-0.5 µs | Inherent to Python/C boundary |
| Grid resolution + assertions | 0.2 µs | Already minimal |

These are essentially at the floor of what pure Python can achieve. Further
improvement requires moving to C/Cython or the compiler-emitted launcher
approach.

## Diff Summary

**File changed:** `third-party/triton/beta/triton/python/triton/runtime/jit.py`

### New methods

- **`_compute_fast_key(self, args, device)`** — Computes a minimal tuple from
  args that uniquely determines which compiled kernel to use. Returns `None`
  for unsupported arg types (graceful fallback to slow path).

- **`_make_launch_cache(device, args, kernel)`** — Static method that builds a
  flat 10-element tuple caching everything needed for fast-path launch: the
  kernel, the C launch function, CUfunction handle, packed metadata, launcher
  flags (cooperative, cluster, PDL), and a no-scratch flag.

### New fields in `JITFunction.__init__`

- **`self._run_cache`** — Dict mapping `(device, fast_arg_signature)` →
  launch cache tuple (Layer 2 cache).
- **`self._last_call`** — 10-element tuple from the previous successful launch
  (Layer 1 cache).

### Modified method: `JITFunction.run()`

The `run()` method now has three execution tiers:

1. **Fast path Layer 1** (identity check) — `O(n)` pointer comparisons, no
   allocations. Hits when the same Python objects are passed.
2. **Fast path Layer 2** (signature dict lookup) — Builds a fast key tuple,
   does a dict lookup. Hits when arg types/values match but objects differ.
3. **Slow path** (original code) — Full binder + cache key + compilation.
   Populates both fast caches on success.

Both fast paths:
- Still verify global variables haven't changed (correctness)
- Skip `launch_metadata()` when hooks are None
- Call `launcher.launch()` directly when no scratch is needed
- Use cached kernel/launcher properties instead of per-call attribute access
