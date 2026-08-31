- `acc = tlx.async_dot(a[i], b[i], acc)` **[sm90+]**
- `acc = tlx.async_dot(a_reg, b[i], acc)` **[sm90]**
- `acc[i] = tlx.async_dot(a[i], b[i], acc[i], barrier)` **[sm100]**
- `acc[i] = tlx.async_dot_scaled(a[i], b[i], acc[i], a_scale[i], a_format, b_scale[i], b_format, use_acc, two_ctas, mBarriers)` **[sm100]**

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

    Use `tlx.tmem_copy` **[sm100]** to efficiently transfer scale data from shared memory to tensor memory:

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

- `acc = tlx.async_dot_wait(pendings, acc)` **[sm90+]**

    Wait for completion of prior asynchronous dot operations. The pendings argument indicates the number of in-flight operations not completed.

    Example:
    ```python
    acc = tlx.async_dot(a_smem, b_smem)
    acc = tlx.async_dot_wait(tl.constexpr(0), acc)
    tl.store(C_ptrs, acc)
    ```

## Explicit MFMA scheduling

> **[gfx942, gfx950]** — the source-scheduled operations support CDNA3 and
> CDNA4 native BF16/F16 MFMA layouts.

- `tl.dot(a, b, acc)` preserves a TLX-pinned accumulator layout, so whole dots
  need no AMD-specific wrapper.
- `tlx.extract_slice(source, shape, offsets)` selects an aligned register
  fragment without cross-thread movement.
- `tlx.rematerialized_range(start, end, anchor, placement=None)` recreates
  inexpensive distributed coordinates near a use instead of carrying them
  through a long software pipeline.
- `tlx.amd_register_resident(value, register_class="agpr", registers_per_group=1)`
  keeps every native-register group in one allocator-visible whole-tensor
  residency interval.
- `tlx.amd_register_handoff(value, register_class="vgpr")` starts an
  independent allocation interval for each 32-bit native register value,
  shortening a local live range without requiring simultaneous whole-tensor
  residency. Both register-boundary operations accept `"vgpr"` or `"agpr"`;
  `amd_register_resident` additionally accepts a power-of-two
  `registers_per_group` from 1 through 32.
- `tlx.amd_scheduled_mfma(...)` exposes independent native MFMA accumulator
  chains in deterministic N-major, M-minor, K-reduction source order.
- `tlx.amd_mfma_commit(value, preserve)` applies the target MFMA result hazard
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
phases. Its default `auto` accumulator storage follows the role alone, the same
way on every target: VGPRs for `transient` and AGPRs for `persistent`. An
explicit register class overrides that choice, and CDNA3 requires one — it
cannot order an AGPR accumulator read against the MFMA drain, so a persistent
accumulator there must ask for `"vgpr"` rather than be silently downgraded.
Neither role changes the numerical matrix operation.

## Scaled dot

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

### Example: `tiles_per_warp`

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
