- `tlx.thread_id(axis)` **[sm90+]**

    Returns the id of the current thread instance along the given `axis`.

- `tlx.dtype_of(v)` **[sm90+]**

    Returns the dtype of a tensor or tensor descriptor.

- `tlx.size_of(dtype)` **[sm90+]**

    Returns the size in bytes of a given Triton dtype. This is useful for dynamically computing memory sizes based on dtype, especially in barrier synchronization code.

    Example:
    ```python
    # Instead of hardcoding size values
    tlx.barrier_expect_bytes(barrier, 2 * BLOCK_M * BLOCK_K)  # Assumes float16

    # Use size_of for dtype-aware computation
    tlx.barrier_expect_bytes(barrier,
                           tlx.size_of(tlx.dtype_of(desc)) * BLOCK_M * BLOCK_K)
    ```

- `tlx.clock64()` **[sm90+]**

    Returns the current 64-bit hardware clock value. E.g,
    ```
        start = tlx.clock64()
        # ... kernel code ...
        end = tlx.clock64()
        elapsed = end - start  # Number of clock cycles elapsed
    ```

- `tlx.stoch_round(src, dst_dtype, rand_bits)` **[sm100]**

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

- `tlx.vote_ballot_sync(mask, pred)` **[sm90+]**

    Collects a predicate from each thread in the warp and returns a 32-bit
    mask where each bit represents the predicate value from the corresponding
    lane. Only threads specified by `mask` participate in the vote.
    ```
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
    ```

- `tlx.warp_all(pred)` / `tlx.warp_any(pred)` **[sm90+, gfx942+]**

    Reduce one distributed predicate per physical lane to a warp-uniform
    scalar `i1`. Non-boolean inputs are compared with zero first. `warp_all`
    is true only if every lane contributes true; `warp_any` is true if at
    least one lane contributes true.

    These are physical-lane votes, not logical tensor-axis reductions.
    `tl.all` and `tl.max` reduce a tensor dimension and retain tensor layout
    semantics, whereas these operations produce a scalar suitable for a
    uniform branch. The predicate must distribute exactly one element per
    lane; compilation fails if its resolved layout gives a lane zero or
    multiple elements.

    ```python
    all_safe = tlx.warp_all(per_lane_safe)
    any_active = tlx.warp_any(per_lane_active)
    ```

    Prefer these semantic reductions when code only needs an all/any result.
    They do not expose a hardware ballot bit mask.

- `tlx.warp_predicate(predicate, inits, body, args=(), wave_uniform=False)` **[AMD]**

    Executes `body(*inits, *args)` with the hardware execution mask restricted
    to lanes where `predicate` is true. Active lanes receive the values returned
    by `body`; inactive lanes keep their corresponding values from `inits`. A
    single carried tensor is returned directly, while multiple carried tensors
    are returned as a tuple.

    A scalar predicate controls its physical lane. For a tensor predicate, the
    elements owned by each lane are OR-reduced to form that lane's execution
    bit. Unlike `tlx.warp_all` and `tlx.warp_any`, this does not make the
    predicate uniform across the wave.

    ```python
    @triton.jit
    def scale_active(acc, scale):
        return acc * scale

    acc = tlx.warp_predicate(
        active,
        acc,
        scale_active,
        args=(scale,),
    )
    ```

    `body` must be an `@triton.jit` function that returns the same number and
    types of tensors as `inits`. It must contain straight-line computation and
    no cross-wave synchronization. Reductions, dots, and layout shuffles are
    allowed only with `wave_uniform=True`, which asserts that every lane in a
    wave observes the same predicate. Cross-warp reductions, nested dynamic
    control flow, and other nested regions are unsupported. This operation is
    currently lowered only by the AMD backend.

- `tlx.prefetch(pointer, level="L2", mask=None, tensormap=False)` **[sm90+]** issues a non-blocking prefetch hint for pointer-based scattered/gather loads. This complements `tlx.async_descriptor_prefetch_tensor` (which works on TMA tensor descriptors) by supporting raw pointer tensors.
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

## `tlx.assume_uniform`

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
