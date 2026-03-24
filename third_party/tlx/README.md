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
