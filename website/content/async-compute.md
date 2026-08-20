### Async tensor core operations

- `acc = tlx.async_dot(a[i], b[i], acc)` **[Hopper+]**
- `acc = tlx.async_dot(a_reg, b[i], acc)` **[Hopper]**
- `acc[i] = tlx.async_dot(a[i], b[i], acc[i], barrier)` **[Blackwell]**
- `acc[i] = tlx.async_dot_scaled(a[i], b[i], acc[i], a_scale[i], a_format, b_scale[i], b_format, use_acc, two_ctas, mBarriers)` **[Blackwell]**

    **Parameters:**
    - `a[i]`: A tile in shared memory (FP8 format)
    - `b[i]`: B tile in shared memory (FP8 format)
    - `acc[i]`: Accumulator tile in tensor memory (TMEM)
    - `a_scale[i]`: Per-block scaling factors for A (E8M0 format in SMEM)
    - `a_format`: FP8 format string for A: `"e4m3"`, `"e5m2"`, or `"e2m1"`
    - `b_scale[i]`: Per-block scaling factors for B (E8M0 format in SMEM)
    - `b_format`: FP8 format string for B: `"e4m3"`, `"e5m2"`, or `"e2m1"`
    - `use_acc`: If `True`, compute D = A@B + D; if `False`, compute D = A@B
    - `two_ctas`: If `True`, enables 2-CTA collective MMA (generates `tcgen05.mma.cta_group::2`)
    - `mBarriers`: Optional list of mbarriers for MMA completion signaling

    **2-CTA Scaled MMA:** When `two_ctas=True`, the scaled MMA operates across two CTAs in a cluster. Key considerations:
    - **B data is split**: Each CTA loads half of B (`BLOCK_N // 2`)
    - **B scale is NOT split**: Both CTAs need the full B scale for correct MMA computation
    - **CTA synchronization**: Use "Arrive Remote, Wait Local" pattern before MMA
    - **MMA predication**: Compiler auto-generates predicate so only CTA 0 issues the MMA

    **Example: 2-CTA Scaled MMA**
    ```python
    # B data split across CTAs, but B scale is full
    desc_b = tl.make_tensor_descriptor(b_ptr, ..., block_shape=[BLOCK_K, BLOCK_N // 2])
    desc_b_scale = tl.make_tensor_descriptor(b_scale_ptr, ..., block_shape=[BLOCK_N // 128, ...])  # Full scale

    # Load B with CTA offset, B scale without offset
    tlx.async_descriptor_load(desc_b, b_tile[0], [0, cluster_cta_rank * BLOCK_N // 2], bar_b)
    tlx.async_descriptor_load(desc_b_scale, b_scale_tile[0], [0, 0, 0, 0], bar_b_scale)  # Full B scale

    # CTA sync: "Arrive Remote, Wait Local"
    tlx.barrier_arrive(cta_bars[0], 1, remote_cta_rank=0)
    tlx.barrier_wait(cta_bars[0], phase=0, pred=pred_cta0)

    # 2-CTA scaled MMA with mBarriers for completion tracking
    tlx.async_dot_scaled(
        a_tile[0], b_tile[0], c_tile[0],
        a_scale_tile[0], "e4m3",
        b_scale_tile[0], "e4m3",
        use_acc=False,
        two_ctas=True,
        mBarriers=[mma_done_bar],
    )
    tlx.barrier_wait(mma_done_bar, tl.constexpr(0))
    ```

    **Alternative: Using tcgen05_commit for MMA completion**
    ```python
    # Issue MMA without mBarriers
    tlx.async_dot_scaled(..., two_ctas=True)

    # Use tcgen05_commit to track all prior MMA ops
    tlx.tcgen05_commit(mma_done_bar, two_ctas=True)
    tlx.barrier_wait(mma_done_bar, tl.constexpr(0))
    ```

    **TMEM-backed MX Scales:**

    For scaled MMA operations on Blackwell GPUs, scales can be stored in Tensor Memory (TMEM) for efficient access. TLX provides automatic layout resolution for TMEM scale buffers.

    *Allocating TMEM Scale Buffers:*

    When allocating TMEM buffers for uint8/int8 types (used for MX scales), TLX uses a placeholder layout (`DummyTMEMLayoutAttr`) that gets automatically resolved to `TensorMemoryScalesEncodingAttr` during compilation when the buffer is used with `async_dot_scaled`.

    ```python
    # Allocate TMEM buffers for scales (layout is automatically resolved)
    a_scale_tmem = tlx.local_alloc((128, 8), tl.uint8, num=1, storage=tlx.storage_kind.tmem)
    b_scale_tmem = tlx.local_alloc((256, 4), tl.uint8, num=1, storage=tlx.storage_kind.tmem)
    ```

    *Copying Scales from SMEM to TMEM:*

    Use `tlx.tmem_copy` **[Blackwell]** to efficiently transfer scale data from shared memory to tensor memory:

    ```python
    # Copy scales from SMEM to TMEM (asynchronous, uses tcgen05.cp instruction)
    tlx.tmem_copy(a_scale_smem, a_scale_tmem)
    tlx.tmem_copy(b_scale_smem, b_scale_tmem)
    ```

    *Using TMEM Scales with Scaled MMA:*

    ```python
    # TMEM scales are automatically detected and used with the correct layout
    tlx.async_dot_scaled(
        a_smem, b_smem, acc_tmem,
        A_scale=a_scale_tmem, A_format="e4m3",
        B_scale=b_scale_tmem, B_format="e4m3",
        use_acc=True,
        mBarriers=[mma_bar],
    )
    ```

    *Complete Example: TMEM-backed Scaled GEMM:*

    ```python
    @triton.jit
    def scaled_gemm_kernel(...):
        # Allocate TMEM for accumulator and scales
        acc = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, num=1, storage=tlx.storage_kind.tmem)
        a_scale_tmem = tlx.local_alloc((BLOCK_M // 128, BLOCK_K // 32), tl.uint8, num=1, storage=tlx.storage_kind.tmem)
        b_scale_tmem = tlx.local_alloc((BLOCK_N // 128, BLOCK_K // 32), tl.uint8, num=1, storage=tlx.storage_kind.tmem)

        # Load scales from global memory to SMEM
        tlx.async_descriptor_load(a_scale_desc, a_scale_smem, [...], barrier=bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_smem, [...], barrier=bar)
        tlx.barrier_wait(bar, phase)

        # Copy scales from SMEM to TMEM
        tlx.tmem_copy(a_scale_smem[0], a_scale_tmem[0])
        tlx.tmem_copy(b_scale_smem[0], b_scale_tmem[0])

        # Perform scaled MMA with TMEM scales
        tlx.async_dot_scaled(
            a_smem[0], b_smem[0], acc[0],
            A_scale=a_scale_tmem[0], A_format="e4m3",
            B_scale=b_scale_tmem[0], B_format="e4m3",
            use_acc=False,
        )
    ```

    **Note:** Multibuffering is automatically cancelled for scale buffers since TMEM scales don't support multibuffering. 3D allocations (1×M×K) are automatically flattened to 2D (M×K).

- `acc = tlx.async_dot_wait(pendings, acc)` **[Hopper+]**

    Wait for completion of prior asynchronous dot operations. The pendings argument indicates the number of in-flight operations not completed.

    Example:
    ```python
    acc = tlx.async_dot(a_smem, b_smem)
    acc = tlx.async_dot_wait(tl.constexpr(0), acc)
    tl.store(C_ptrs, acc)
    ```
