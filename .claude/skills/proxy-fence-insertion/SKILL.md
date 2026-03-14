# Proxy Fence Insertion

Use when working on fence-related compiler passes, TMA store lowering, proxy
fence insertion, investigating missing or spurious fences, or debugging correctness
issue in TLX kernels that use tlx.async_descriptor_store or MMA operations.

---

## Why fences are needed

Hopper+ (sm90+) has separate **generic** and **async** memory proxies. Writes
through one proxy are not visible to reads through the other without an explicit
proxy fence (`fence.proxy.async.shared::cta`). For example, a register→SMEM
store (generic proxy) followed by a TMA store from SMEM (async proxy) requires
a fence between the two.

## TLX DSL API

Source: `third_party/tlx/language/tlx/mem_ops.py`

### `tlx.fence(scope)`

Unified fence entry point.

| `scope`          | PTX emitted                        | Use case |
|------------------|------------------------------------|----------|
| `"async_shared"` | `fence.proxy.async.shared::cta`    | Bridge generic↔async proxy (e.g. between `local_store` and TMA store) |
| `"gpu"`          | `fence.acq_rel.gpu`                | Device-scope ordering of global/shared memory |
| `"sys"`          | `fence.acq_rel.sys`                | System-scope ordering (visible to host CPU) |

### `tlx.fence_async_shared()`

Deprecated alias for `tlx.fence("async_shared")`.

### Canonical TMA store pattern

```python
tlx.local_store(smem, data)
tlx.fence("async_shared")           # proxy fence
tlx.async_descriptor_store(desc, smem)
tlx.async_descriptor_store_wait(0)
```

## Common proxy-crossing patterns

### 1. Register → SMEM → TMA store

`local_store` (generic proxy write) followed by `async_descriptor_store` (async
proxy read). The TMA hardware reads SMEM via the async proxy, so a fence is
needed after the generic-proxy store. This is handled by **TMALowering** and
covered by the canonical TMA store pattern above.

### 2. Register → SMEM → MMA (wgmma / tcgen5)

When MMA operands are populated by writing registers to SMEM (via `LocalAllocOp`
with a source or `LocalStoreOp`), the write goes through the generic proxy.
wgmma and tcgen5 MMA instructions read their SMEM operands through the async
proxy. A proxy fence is required between the register→SMEM copy and the MMA.
This is handled automatically by **FenceInsertionPass**.

In TLX kernels this shows up when, for example, scales or other data are
written to SMEM from registers and then consumed by a `wgmma` — the compiler
inserts the fence, but understanding the pattern helps when debugging
correctness issues where the fence might be missing.

## Compiler fence insertion

Three passes insert proxy fences at different stages of the compilation
pipeline. They are listed in the order they run.

### 1. FenceInsertionPass (optimization phase)

**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp`

Walks every `DotOpInterface` op (wgmma / tcgen5 MMA). If an operand traces
back to a register→SMEM copy (generic proxy write feeding an async proxy read),
inserts a `FenceAsyncSharedOp` before the dot. Can hoist the fence out of loops
when safe. Only runs on sm90+.

### 2. TMALowering (TTGIR → TTGIR rewrite)

**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp`

Rewrites high-level TMA store ops. Unconditionally inserts a
`FenceAsyncSharedOp` between the `LocalAllocOp` (register→SMEM) and the
lowered TMA store:

```
LocalAllocOp  →  FenceAsyncSharedOp  →  TMA store  →  TMAStoreWaitOp
```

### 3. ProxyFenceInsertionPass (post-allocation safety net)

**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/ProxFenceInsertion.cpp`

Runs **after** shared memory allocation. Uses alias analysis over allocated
buffers to find remaining generic↔async proxy conflicts not caught by earlier
passes. Conservatively inserts fences to avoid races. Only runs on sm90+
(`computeCapability >= 90`).

## PTX lowering chain

```
FenceAsyncSharedOp (TritonNvidiaGPU dialect)
  → NVVM::FenceProxyOp (NVVM dialect)
    → fence.proxy.async.shared::cta  (PTX)
```

Lowering lives in
`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp`
(`FenceAsyncSharedOpConversion`). The `bCluster` attribute selects
`shared::cluster` vs `shared::cta` scope.

## When a fence is NOT needed

- **Async→async** (same proxy domain) — no proxy crossing
- **Pre-Hopper** (< sm90) — no separate async proxy
- **Fence already present** between the conflicting ops (all three passes check
  for existing `FenceAsyncSharedOp`)
