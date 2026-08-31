> **[gfx950]** — available on AMD MI350 (CDNA4) only; not available on MI300.

Buffer operations access global memory via a scalar base pointer and a tensor of i32 element offsets, rather than a tensor of pointers. This maps directly to AMD's hardware buffer instructions, which use a resource descriptor and byte offsets, enabling the hardware to do out-of-bounds checking and cache optimization.

## `tlx.buffer_load`

Load a tensor of values from global memory.

```python
result = tlx.buffer_load(
    ptr, offsets, mask=None, other=None, cache=None, contiguity=1
)
```

| Argument | Type | Description |
|----------|------|-------------|
| `ptr` | scalar pointer | Base address in global memory. |
| `offsets` | i32 tensor | Per-element byte offsets from `ptr`. |
| `mask` | bool tensor, optional | When `mask[i]` is `False`, the element is not loaded. |
| `other` | tensor or scalar, optional | Value used for masked-out elements (where `mask[i]` is `False`). |
| `cache` | str, optional | Cache modifier (e.g. `".ca"`, `".cg"`). |
| `contiguity` | constexpr int | Trusted positive power-of-two vectorization width; it must divide each thread's element count. |

**Returns**: A tensor with the same shape as `offsets` and element type matching the pointee type of `ptr`.

Lowers to `amdg.buffer_load`, which is eventually lowered to `rocdl.raw.ptr.buffer.load`.

## `tlx.buffer_store`

Store a tensor of values to global memory.

```python
tlx.buffer_store(stored_value, ptr, offsets, mask=None, cache=None)
```

| Argument | Type | Description |
|----------|------|-------------|
| `stored_value` | tensor | Values to write. |
| `ptr` | scalar pointer | Base address in global memory. |
| `offsets` | i32 tensor | Per-element byte offsets from `ptr`. |
| `mask` | bool tensor, optional | When `mask[i]` is `False`, the element is not written. |
| `cache` | str, optional | Cache modifier. |

**Returns**: Nothing.

Lowers to `amdg.buffer_store`, which is eventually lowered to `rocdl.raw.ptr.buffer.store`.

## `tlx.buffer_atomic_add`

Atomically add a tensor through an AMD scalar resource descriptor.

```python
previous = tlx.buffer_atomic_add(
    ptr, offsets, value, mask=None, sem=None, scope=None, contiguity=1
)
```

`contiguity` is the same trusted per-thread adjacency width accepted by
`buffer_load`; values greater than one also preserve the selected layout. The
operation returns the values observed before the additions. FP16 and BF16 use
packed two-element instructions, so `contiguity` must be at least two and any
mask must be uniform for each adjacent pair. Compilation fails if mask analysis
reduces the legal vector width below two.


## Example: buffer load and store

```python
import triton.language.extra.tlx as tlx

@triton.jit
def kernel(src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int32)
    mask = offsets < BLOCK_SIZE

    # Load from global memory using buffer semantics
    data = tlx.buffer_load(src_ptr, offsets, mask=mask, other=0.0)

    # Store to global memory using buffer semantics
    tlx.buffer_store(data, dst_ptr, offsets, mask=mask)
```

For the async global-to-shared variant, see the warp-pipeline GEMM example (`third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py`).
