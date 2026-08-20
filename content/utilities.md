### Other operations

- `tlx.cluster_cta_rank()`

  Returns the rank (unique ID) of the current CTA within the cluster.

- `tlx.thread_id(axis)`

    Returns the id of the current thread instance along the given `axis`.

- `tlx.dtype_of(v)`

    Returns the dtype of a tensor or tensor descriptor.

- `tlx.size_of(dtype)`

    Returns the size in bytes of a given Triton dtype. This is useful for dynamically computing memory sizes based on dtype, especially in barrier synchronization code.

    Example:
    ```python
    # Instead of hardcoding size values
    tlx.barrier_expect_bytes(barrier, 2 * BLOCK_M * BLOCK_K)  # Assumes float16

    # Use size_of for dtype-aware computation
    tlx.barrier_expect_bytes(barrier,
                           tlx.size_of(tlx.dtype_of(desc)) * BLOCK_M * BLOCK_K)
    ```

- `tlx.clock64()`

    Returns the current 64-bit hardware clock value. E.g,
    ```
        start = tlx.clock64()
        # ... kernel code ...
        end = tlx.clock64()
        elapsed = end - start  # Number of clock cycles elapsed
    ```

- `tlx.stoch_round(src, dst_dtype, rand_bits)`

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
