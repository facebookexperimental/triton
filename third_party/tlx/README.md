# TLX - Triton Low-level Extensions

## Introduction

TLX (Triton Low-level Language Extensions) is a low-level, warp-aware, hardware-near extension of the Triton DSL. It provides intrinsics and warp-specialized operations for fine-grained GPU control, hardware-oriented primitives for advanced kernel development, and explicit constructs for GPU memory, compute, and asynchronous control flow, designed for power users pushing Triton to the metal.

## Memory Fences

`tlx.fence(scope)` issues a memory fence. The `scope` argument is required:

| Scope | PTX | Description |
|-------|-----|-------------|
| `"gpu"` | `fence.acq_rel.gpu` | Device-scope fence. Orders prior global/shared memory writes to be visible to all GPU threads. |
| `"sys"` | `fence.acq_rel.sys` | System-scope fence. Like `"gpu"` but also visible to the host CPU. |
| `"async_shared"` | `fence.proxy.async.shared::cta` | Proxy fence for async shared memory. Required between `local_store` and a subsequent TMA store (`async_descriptor_store`) to the same shared memory. |

Example:
```python
tlx.local_store(smem_buf, data)
tlx.fence("async_shared")
tlx.async_descriptor_store(desc, smem_buf, offsets)
```

## Prefetch

`tlx.prefetch(pointer, level="L2", mask=None)` issues a non-blocking prefetch hint for pointer-based scattered/gather loads. This complements `tlx.async_descriptor_prefetch_tensor` (which works on TMA tensor descriptors) by supporting raw pointer tensors.

| Level | PTX | Description |
|-------|-----|-------------|
| `"L1"` | `prefetch.global.L1` | Prefetch into L1 and L2 cache |
| `"L2"` | `prefetch.global.L2` | Prefetch into L2 cache only (default) |

Example:
```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
tlx.prefetch(input_ptr + offsets, level="L2", mask=mask)
x = tl.load(input_ptr + offsets, mask=mask)
```
