### Other operations

- `tlx.cluster_cta_rank()` **[Hopper+]**

  Returns the rank (unique ID) of the current CTA within the cluster.

- `tlx.thread_id(axis)` **[Hopper+]**

    Returns the id of the current thread instance along the given `axis`.

- `tlx.dtype_of(v)` **[Hopper+]**

    Returns the dtype of a tensor or tensor descriptor.

- `tlx.size_of(dtype)` **[Hopper+]**

    Returns the size in bytes of a given Triton dtype. This is useful for dynamically computing memory sizes based on dtype, especially in barrier synchronization code.

    Example:
    ```python
    # Instead of hardcoding size values
    tlx.barrier_expect_bytes(barrier, 2 * BLOCK_M * BLOCK_K)  # Assumes float16

    # Use size_of for dtype-aware computation
    tlx.barrier_expect_bytes(barrier,
                           tlx.size_of(tlx.dtype_of(desc)) * BLOCK_M * BLOCK_K)
    ```

- `tlx.clock64()` **[Hopper+]**

    Returns the current 64-bit hardware clock value. E.g,
    ```
        start = tlx.clock64()
        # ... kernel code ...
        end = tlx.clock64()
        elapsed = end - start  # Number of clock cycles elapsed
    ```

- `tlx.stoch_round(src, dst_dtype, rand_bits)` **[Blackwell]**

    Performs hardware-accelerated stochastic rounding for FP32→FP8/BF16/F16 conversions on Blackwell GPUs (compute capability ≥ 100). Uses PTX `cvt.rs.satfinite` instructions for probabilistic rounding.

    **Why Use Stochastic Rounding:**
    - Reduces bias in low-precision training/inference by randomly rounding up or down
    - Improves numerical accuracy compared to deterministic rounding (e.g., round-to-nearest-even)
    - Particularly beneficial when accumulating many small updates in FP8/FP16

    **Performance Characteristics:**
    - Hardware-accelerated: Uses native Blackwell instructions (cvt.rs.satfinite)
    - Minimal overhead: Similar throughput to deterministic rounding
    - Memory bandwidth: Requires additional random bits (uint32 per element)

    Parameters:
    - `src`: Source FP32 tensor
    - `dst_dtype`: Destination dtype (FP8 E5M2, FP8 E4M3FN, BF16, or FP16)
    - `rand_bits`: Random bits (uint32 tensor) for entropy, same shape as src
      - **Important:** Use `n_rounds=7` with `tl.randint4x()` for sufficient entropy
      - Fewer rounds may result in biased rounding behavior
      - Different seeds produce different rounding decisions for better statistical properties

    Example:
    ```python
        # Generate random bits for entropy
        # n_rounds=7 provides sufficient randomness for unbiased stochastic rounding
        offsets = tl.arange(0, BLOCK_SIZE // 4)
        r0, r1, r2, r3 = tl.randint4x(seed, offsets, n_rounds=7)
        rbits = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(x.shape)

        # Apply stochastic rounding
        y = tlx.stoch_round(x, tlx.dtype_of(y_ptr), rbits)
    ```

- `tlx.vote_ballot_sync(mask, pred)` **[Hopper+]**

    Collects a predicate from each thread in the warp and returns a 32-bit
    mask where each bit represents the predicate value from the corresponding
    lane. Only threads specified by `mask` participate in the vote.
    ```
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
    ```

- `tlx.prefetch(pointer, level="L2", mask=None, tensormap=False)` **[Hopper+]** issues a non-blocking prefetch hint for pointer-based scattered/gather loads. This complements `tlx.async_descriptor_prefetch_tensor` (which works on TMA tensor descriptors) by supporting raw pointer tensors.
  Additionally, if `tensormap` is specified to `True`, the API instead does a prefetch of tensor map object (TMA descriptor) and ignores other parameters other than `pointer`.

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

  ...
  # desc_in can be host side descriptor or device side like this:
  desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )
  tlx.prefetch(desc_in, tensormap=True)
  ```

- `tlx.dump_layout(x)` **[Hopper+, MI300+]**

    Compile-time diagnostic that prints the resolved layout of a value to the
    compiler log. `x` may be a register tensor or a shared/tensor-memory buffer
    (memdesc). It emits **no** device code and returns nothing — this is a
    static, host-side diagnostic, distinct from the runtime `tl.device_print` /
    `tl.print`. The op is rendered at the end of the TTGIR pipeline, so the
    printed layout reflects all compiler optimizations, then it is erased.

    The layout is printed in CuTe (CUTLASS) `Shape:Stride` notation (`_N` marks
    a static integer):
    - Register tensors → a thread-value (TV) layout
      `((thread...),(value...)):((thread...),(value...))`, where the thread
      group comes from the hardware lane/warp/block dims and the value group
      from the per-thread registers (stride `_0` denotes a broadcast).
    - Shared/tensor-memory buffers → a single strided layout, e.g. `_64:_1`.
    - Swizzled shared buffers → `Swizzle<B,M,S> o (base):(stride)`.

    In all cases the layout maps a coordinate to the **logical tensor's
    row-major element index** (its codomain): for a register tensor a
    `(thread, value)` coordinate → the logical element index it holds, and for
    a buffer an offset → the buffer element index. The strides are offsets in
    that logical index space, not physical byte/bank addresses.

    Layouts that are not representable as a CuTe layout fall back to the raw
    linear-layout string.

    Example:
    ```python
    x = tl.load(x_ptr + offs)          # register tensor
    tlx.dump_layout(x)                  # -> // cute: ((_32,_2,_2),_1):((_1,_32,_0),_0)

    buf = tlx.local_alloc((BLOCK,), tl.float32, 1)
    v = tlx.local_view(buf, 0)
    tlx.dump_layout(v)                  # -> // cute: _64:_1
    ```

- `x = tlx.require_layout(x, layout, pin=True, late_address_compute=False)` **[Hopper+, MI300+]**

    Require a register tensor `x` to use `layout`, expressed as a
    `tlx.layout(...)` (Shape:Stride). With the default `pin=True`, the `#linear`
    encoding is wrapped as `#tlx.no_verify_layout(#tlx.user_layout(...))`. The
    inner `#tlx.user_layout` carries `PinnedEncodingTrait`, so downstream passes
    (`tritongpu-coalesce`, `remove-layout-conversions`, AMD `optimize-epilogue`)
    treat it as fixed and never rewrite it; the outer `#tlx.no_verify_layout` defers
    operand-layout verification until the pin is peeled by
    `TLXResolvePlaceholderLayouts` in `make_ttgir`. Example: pin an FP16 epilogue
    `tl.store` to a coalesced layout so `OptimizeEpilogue` keeps the wide
    `buffer_store_dwordx4` instead of narrowing it to the MMA-accumulator store,
    without staging the value through LDS.

    Pass `pin=False` for an optimizer-flexible requirement. This emits the
    requested encoding without the `#tlx.user_layout` hard anchor, allowing
    later layout passes to propagate the requirement or materialize a layout
    conversion. The default remains `pin=True` for existing callers.

    On AMD, `late_address_compute=True` asks shared-memory-backed layout
    conversions to compute their addresses at this use. The backend does this
    by rematerializing inexpensive lane/warp coordinates, shortening their live
    ranges across register-heavy regions.

    Pair with `tlx.assert_same_layout(x, layout)` (below) to statically verify the
    pin survived to the final TTGIR.

- `tlx.assert_same_layout(lhs, rhs)` **[Hopper+, MI300+]**

    Compile-time assertion that two layouts are equivalent after layout
    propagation and all other TTGIR layout optimizations have completed. Like
    `tlx.dump_layout`, it emits no device code and is consumed at the end of the
    TTGIR pipeline.

    `rhs` supports two forms:

    - **Value/value:** `rhs` is another register tensor or shared/tensor-memory
      buffer. The frontend emits `tlx.assert_same_layout`, whose two operands
      retain their independently resolved final types.
    - **Value/layout:** `rhs` is a constant `tlx.layout_encoding`. The frontend
      lowers the constant to an encoding attribute and emits
      `tlx.assert_same_layout_expected`. At assertion time, the pass combines
      that encoding with `lhs`'s shape, element type, and (for buffers) memory
      properties to construct an expected tensor or memdesc type.

    These are separate internal operations only because an SSA value is an MLIR
    operand while a constant layout is an MLIR attribute. They share the same
    comparison path and the same public Python API.

    Before comparison, both final types are converted with
    `ttg::toLinearLayout`. The assertion compares the resulting
    `LinearLayout`s, not the original encoding attributes. Consequently,
    structurally different encodings pass if they describe the same logical
    mapping. A mismatch reports both normalized LinearLayouts and fails
    compilation.

    Example:

    ```python
    x = tlx.local_load(x_buf, layout=REGISTER_LAYOUT)
    y = tlx.local_load(y_buf, layout=REGISTER_LAYOUT)

    tlx.assert_same_layout(x, y)                # value/value
    tlx.assert_same_layout(x, REGISTER_LAYOUT)  # value/layout
    ```


## Buffer Operations (AMD)

> **[MI350]** — available on AMD MI350 (CDNA4) only; not available on MI300.

Buffer operations access global memory via a scalar base pointer and a tensor of i32 element offsets, rather than a tensor of pointers. This maps directly to AMD's hardware buffer instructions, which use a resource descriptor and byte offsets, enabling the hardware to do out-of-bounds checking and cache optimization.

### `tlx.buffer_load`

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

### `tlx.buffer_store`

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

### `tlx.buffer_atomic_add`

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

### `tlx.buffer_load_to_local`

Async load from global memory directly into shared (local) memory, bypassing registers. This is useful for producer warps that prefetch data into shared memory for other warps to consume.

```python
token = tlx.buffer_load_to_local(dest, ptr, offsets, mask=None, other=None, cache_modifier="")
```

| Argument | Type | Description |
|----------|------|-------------|
| `dest` | `tlx.buffered_tensor` | Destination slice in shared memory. |
| `ptr` | scalar pointer | Base address in global memory. |
| `offsets` | i32 tensor | Per-element byte offsets from `ptr`. |
| `mask` | bool tensor, optional | When `mask[i]` is `False`, the element is not loaded. |
| `other` | tensor or scalar, optional | Value used for masked-out elements. |
| `cache_modifier` | str, optional | Cache modifier string (default `""`). |

**Returns**: A `tlx.async_token` that can be used with `tlx.async_load_wait_group()` to synchronize on the completion of the transfer.

Lowers to `amdg.buffer_load_to_local`, which is eventually lowered to `rocdl.raw.ptr.buffer.load.async.lds` — a single hardware instruction that moves data from global memory to LDS without going through VGPRs.

**Requirements.** The direct-to-LDS copy has the following hardware constraints:
- **Vector width.** Each thread's load must reach a supported direct-to-LDS width (**32 or 128 bits**). If it can only be vectorized to a smaller width, it cannot be lowered.
- **Provable pointer/offset alignment.** The compiler must be able to *prove* the alignment that vector width needs.
- **Mask alignment.** If `mask` is given it must be aligned to the vector width: each group of (vector width) consecutive mask values must be identical. The copy transfers each lane's whole vector in one transaction, so a mask whose `True`/`False` boundary cannot be proven vector-aligned (e.g. `offs < K` for a runtime `K`) forces per-element vectorization and cannot lower.

### Example

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

## Wave Uniformity (AMD)

### `tlx.assume_uniform`

Assert that a scalar holds the same value in every lane of the wave.

```python
value = tlx.assume_uniform(value)
```

| Argument | Type | Description |
|----------|------|-------------|
| `value` | scalar pointer, or 16/32/64-bit int or float | Value asserted to be wave-uniform. Narrower types are not supported. |

**Returns**: `value`, unchanged.

The main use is buffer operations, which keep their base pointer in the scalar (SGPR) resource descriptor, so it has to be wave-uniform. When the backend cannot prove that it is — most commonly because the pointer was loaded from memory — it falls back to a per-lane waterfall loop around every access. `tlx.assume_uniform` tells the backend to take uniformity as given:

```python
base = tl.load(ptr_array + gid).to(tl.pointer_type(tl.float16))
base = tlx.assume_uniform(base)
data = tlx.buffer_load(base, offsets)
```

Lowers to `amdg.assume_uniform`, which is eventually lowered to `llvm.amdgcn.readfirstlane` — that makes the result uniform by construction as far as LLVM's uniformity analysis is concerned. On non-AMD backends it is a no-op that returns its argument. Nothing verifies the assertion: if the value is not actually uniform, every lane silently gets lane 0's value.

## Explicit MFMA Scheduling (AMD)

> **[MI350]** — the source-scheduled operations are restricted to CDNA4
> (`gfx950`) native BF16 MFMA layouts.

- `tl.dot(a, b, acc)` preserves a TLX-pinned accumulator layout, so whole dots
  need no AMD-specific wrapper.
- `tlx.extract_slice(source, shape, offsets)` selects an aligned register
  fragment without cross-thread movement.
- `tlx.rematerialized_range(start, end, anchor, placement=None)` recreates
  inexpensive distributed coordinates near a use instead of carrying them
  through a long software pipeline.
- `tlx.amd_register_resident(value, register_class="agpr", registers_per_group=1)`
  keeps a distributed value in allocator-visible native register tuples.
- `tlx.amd_scheduled_mfma(...)` exposes independent native MFMA accumulator
  chains in deterministic N-major, M-minor, K-reduction source order.
- `tlx.amd_mfma_commit(value, preserve)` applies the CDNA4 MFMA result hazard
  boundary while threading a live dot-operand dependency.
- `tlx.amd_sched_barrier(mask=0)` prevents selected AMD machine-instruction
  classes from crossing a source boundary. It is a scheduling marker, not a
  workgroup barrier or memory fence.

These primitives describe fragments, lifetimes, and ordering without assigning
physical registers. Their verifiers reject unsupported targets, layouts,
element types, and native fragment widths before lowering.

Unlike `tl.dot`, `amd_scheduled_mfma` carries an explicit `accumulator_role`.
`transient` selects the latency-aware intrinsic path for phase-local work;
`persistent` selects register-constrained lowering for a chain carried across
phases. Neither role changes the numerical matrix operation.

## AMD TDM Descriptor Loads

`tlx.update_tensor_descriptor(desc, add_offsets=None, set_bounds=None,
pred=None, clamp_bounds=False)` produces a positioned descriptor SSA value.
`add_offsets` advances the tile position without changing bounds;
`set_bounds` rewrites absolute bounds; and `pred` replaces the inherited
predicate. Use `clamp_bounds=True` with `add_offsets` to derive the remaining
OOB extent of an advanced tile.

`tlx.async_amd_descriptor_load(desc, result, offsets=None, pred=None,
clamp_bounds=True)` issues an AMD TDM descriptor load from global memory to a
TLX local buffer. If `offsets` is omitted, `desc` is used as already positioned
and its predicate is preserved. The analogous store accepts `offsets=None` for
the same reason. Both operations are available on TDM-capable AMD targets
(`gfx1250+`) and should be synchronized with
`tlx.async_amd_descriptor_wait`.

`tlx.async_amd_descriptor_load_fused(members, cache_modifier="")` fuses two to
four AMD TDM descriptor loads behind one static hardware instruction. Each
member is a `(positioned_desc, destination, warp_used_hint)` tuple. Position
descriptors with `tlx.update_tensor_descriptor` before issuing the fused load.
The hints must be non-empty, pairwise disjoint, and legal axis-aligned subsets
of the CTA's waves; they do not need to cover every wave. Members must have the
same descriptor rank but may have different shapes and element types. All
members share one cache modifier.

Example:
```python
a_load_desc = tlx.update_tensor_descriptor(
    a_desc, add_offsets=[off_m, k * BLOCK_K], pred=True, clamp_bounds=True)
b_load_desc = tlx.update_tensor_descriptor(
    b_desc, add_offsets=[k * BLOCK_K, off_n], pred=True, clamp_bounds=True)
a_tok = tlx.async_amd_descriptor_load_fused([
    (a_load_desc, tlx.local_view(a_buf, slot), 0b0011),
    (b_load_desc, tlx.local_view(b_buf, slot), 0b1100),
])
tlx.async_amd_descriptor_wait(0, [a_tok])
```

## Warp Pipeline (AMD)

> **[MI350]** — AMD MI350 (CDNA4); not available on MI300.

`tlx.warp_pipeline_stage(label, *, priority=None)` is a context manager that marks explicit pipeline stage boundaries inside a loop. The compiler partitions the loop body at these boundaries and inserts conditional barriers so that one warp group executes one stage ahead of the other, overlapping memory latency with compute.

**This is an explicit partitioning marker, not an automatic optimization.** Correctness depends on the user's buffering and synchronization structure. In particular:
- Use multi-buffered shared memory (typically triple buffering with `NUM_BUFFERS=3`) to prevent data races between warp groups accessing the same buffer.
- Use explicit `tlx.async_load_wait_group()` to ensure data is ready before consumption.
- Handle prologue (prefetch) and epilogue (drain) around the main loop.

See the gfx1250 warp-pipeline GEMM example (`third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py`) for the full pattern.

| Parameter | Type | Description |
|-----------|------|-------------|
| `label` | `str` | Stage name for diagnostics (e.g. `"load"`, `"compute"`) |
| `priority` | `int` (0-3), optional | Hardware scheduling hint, maps to `s_setprio`. Higher = more urgent. |

Auto software pipelining is automatically disabled on loops that contain warp pipeline stages.

Example (simplified — see gfx1250 example for production pattern):
```python
import triton.language.extra.tlx as tlx

@triton.jit
def gemm_kernel(..., BLOCK_K: tl.constexpr, NUM_BUFFERS: tl.constexpr):
    buf_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, NUM_BUFFERS)
    buf_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, NUM_BUFFERS)

    # Prologue: prefetch NUM_BUFFERS-1 tiles into shared memory
    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        tlx.async_load(a_ptrs, tlx.local_view(buf_A, i), mask=...)
        tlx.async_load(b_ptrs, tlx.local_view(buf_B, i), mask=...)
        tlx.async_load_commit_group()
        a_ptrs += BLOCK_K * stride_ak; b_ptrs += BLOCK_K * stride_bk
    tlx.async_load_wait_group(NUM_BUFFERS - 2)

    # Main loop with warp pipelining
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(NUM_BUFFERS - 1, K_ITERS):
        consumer = (k - (NUM_BUFFERS - 1)) % NUM_BUFFERS
        producer = k % NUM_BUFFERS
        with tlx.warp_pipeline_stage("lds_load", priority=1):
            a_tile = tlx.local_load(tlx.local_view(buf_A, consumer))
            b_tile = tlx.local_load(tlx.local_view(buf_B, consumer))
        tlx.async_load_wait_group(0)
        with tlx.warp_pipeline_stage("compute_and_load", priority=0):
            tlx.async_load(a_ptrs, tlx.local_view(buf_A, producer), mask=...)
            tlx.async_load_commit_group()
            acc = tl.dot(a_tile, b_tile, acc)

    # Epilogue: drain remaining buffers
    ...
```

## Scaled Dot (AMD)

`tlx.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, *, fast_math=False, lhs_k_pack=True, rhs_k_pack=True, out_dtype=tl.float32, tiles_per_warp=None)`
is a thin wrapper around `tl.dot_scaled`. Without `tiles_per_warp` it is exactly
equivalent to `tl.dot_scaled` — pass it only when you need the AMD-specific
WMMA scheduling hint described below.

### `tiles_per_warp` — what it controls

| Concept | Controlled by | What it means |
|---------|---------------|---------------|
| **Warp distribution** | `warpsPerCTA` (chosen automatically by `AccelerateAMDMatmul::planWarps`) | How the total result tile is split *across* warps along M/N. |
| **Per-warp tiling** | `tiles_per_warp` (this hint) | How many `instrShape`-sized WMMA tiles *each warp* covers contiguously before the layout repeats. |

So `tiles_per_warp=[2, 2]` does **not** mean "distribute 4 tiles across 4
warps." It means *each* warp emits a 2×2 block of WMMA instruction tiles,
holding the corresponding 2×2 accumulator registers. Concretely, for a
`tt.dot_scaled` lowered to gfx1250 WMMA (`instrShape = [16, 16, K]`),
4 warps, `warpsPerCTA = [2, 2]`:

| `tiles_per_warp` | Per-warp coverage (M × N) | Per-CTA coverage before repeat (M × N) |
|------------------|---------------------------|----------------------------------------|
| `[1, 1]` (default) | `16 × 16` | `32 × 32` |
| `[2, 2]`           | `32 × 32` | `64 × 64` |

For a `256 × 256` result, `[1, 1]` repeats the layout `8 × 8` times,
`[2, 2]` repeats it `4 × 4`. Larger `tiles_per_warp` gives each warp more
contiguous accumulator state (better register reuse for preshuffled
MXFP scales, fewer warp-level reductions), at the cost of more registers
per warp.

Together, `instrShape`, `warpsPerCTA`, and `tiles_per_warp` define the M/N
extent of one CTA-level WMMA layout period:
`period[d] = instrShape[d] * warpsPerCTA[d] * tiles_per_warp[d]`. If the result
tile is larger than this period, the period repeats. The K entry of
`instrShape` is the per-instruction reduction depth and is handled separately
from this M/N tiling.

`tiles_per_warp` is validated by `AccelerateAMDMatmul`: it must have one
entry per result-tile dim, each entry must be positive, and
`instrShape[d] * warpsPerCTA[d] * tiles_per_warp[d]` must fit in the result
tile shape.

### Example

```python
import triton.language.extra.tlx as tlx

acc = tlx.dot_scaled(
    a, a_scale, "e5m2",
    b, b_scale, "e5m2",
    acc,
    tiles_per_warp=[2, 2],   # pack 2x2 WMMA tiles per warp for preshuffled MXFP
)
```

### Mechanism (for IR-level users)

The wrapper attaches `amdg.wmma_tiles_per_warp = array<i32: m, n>` on the
resulting `tt.dot_scaled` op. `ScaledBlockedToScaledWMMAF8F6F4` reads the
attribute and substitutes `m, n` for the default `1, 1` when building the
WMMA encoding. Setting the attribute directly on a `tt.dot[_scaled]` op
in MLIR has the same effect; the wrapper just spares Python kernels from
hand-poking attributes.

Currently consumed only by the scaled-WMMA pattern (gfx1250). Regular
`tt.dot` WMMA and the MFMA patterns do not read it.

### Explicit WMMA layout pinning

`tlx.require_amd_wmma_layout(x, version=3, transposed=True, warp_bases=...,
reg_bases=..., instr_shape=(16, 16, 128))` pins a tensor to an explicit AMD
WMMA register/warp layout. This is useful for tuned gfx1250 epilogues that must
retain the accumulator ownership chosen by `tiles_per_warp` across otherwise
layout-neutral tensor operations. The bases contain one linear-layout basis
vector per register or warp bit and must match the tensor rank.

The helper lowers to a pinned `tlx.require_layout`; omit it when automatic
layout propagation is sufficient.
