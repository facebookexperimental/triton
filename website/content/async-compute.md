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

> `amd_scheduled_mfma` and `amd_mfma_commit` support **[gfx942, gfx950]**
> native BF16/F16 MFMA layouts. The register-class constraint operations are
> AMD-only and currently verified on **[gfx950]**.

These are low-level backend-tuning controls. Prefer `tl.dot` unless a kernel
needs explicit native accumulator chains, register-class constraints, or an
MFMA completion boundary. Inspect the generated code and benchmark any use;
the register allocator still chooses physical registers.

| API | Verified targets | Purpose |
|-----|------------------|---------|
| `tlx.amd_scheduled_mfma` | gfx942, gfx950 | Update explicitly ordered native MFMA accumulator chains. |
| `tlx.amd_mfma_commit` | gfx942, gfx950 | Join one or more chains at an MFMA completion and liveness boundary. |
| `tlx.amd_register_class_anchor` | gfx950 | Add a separate local register-class anchor for each native 32-bit value. |
| `tlx.amd_register_resident` | gfx950 | Require every tensor group to be allocatable together in native register tuples. |

Related operations are `tlx.extract_slice`, which selects an aligned register
fragment without cross-thread movement; `tlx.rematerialized_range`, which
recreates inexpensive coordinates near a use; and `tlx.amd_sched_barrier`,
which prevents selected AMD instruction classes from crossing a source
boundary. None of these operations is a workgroup barrier or memory fence.

### `tlx.amd_register_resident`

```python
value = tlx.amd_register_resident(
    value,
    register_class="agpr",
    registers_per_group=1,
)
```

Returns `value` unchanged, with the same shape, element type, and distributed
layout. At one allocator-visible point, every packed per-thread group must be
available in the requested `"agpr"` or `"vgpr"` class.

`registers_per_group` is the number of consecutive 32-bit registers in each
group, not the number of groups. It must be one of `1`, `2`, `4`, `8`, `16`,
or `32`. A group of width `R` contains `R` 32-bit elements or `2 * R` 16-bit
elements, and the per-thread element count must divide evenly into groups.
Registers are consecutive within one group; different groups need not be
adjacent.

This constraint does not select physical register numbers, reserve registers
for the value's full lifetime, prevent copies or spills, or guarantee a
particular occupancy. Wider groups strengthen the tuple constraint and can
reduce allocator flexibility. Always use the returned value.

### `tlx.amd_register_class_anchor`

```python
value = tlx.amd_register_class_anchor(value, register_class="vgpr")
```

Returns `value` unchanged after passing each packed 32-bit register value
through a separate, side-effecting tied `"agpr"` or `"vgpr"` constraint. This
provides a source-local allocation and scheduling anchor without requiring all
tensor groups to meet at one point.

A tied input and output may be coalesced by LLVM. The operation does not
guarantee a distinct physical register or live interval, emit a copy, shorten
a live range, prevent spills, or improve occupancy. Use
`amd_register_resident` when simultaneous all-group allocation is required,
and use the value returned by `amd_register_class_anchor` for downstream work.

Both register-class operations accept distributed integer or floating-point
tensors with 16-bit or 32-bit elements. They constrain allocation at a source
point; they do not wait for MFMA completion or synchronize lanes, waves, or
memory.

### `tlx.amd_scheduled_mfma`

```python
acc = tlx.amd_scheduled_mfma(
    a,
    b,
    acc,
    accumulator_role="transient",
    resident_operand=None,
    accumulator_register_class=None,
    initialize=False,
)
```

With `initialize=False`, the operation computes the native-fragment equivalent
of `acc + a @ b`. It keeps one SSA chain per output fragment and creates
updates in K-major, N-major, M-minor order:

```python
for k in native_k_fragments:
    for n in output_n_fragments:
        for m in output_m_fragments:
            acc[m, n] = mfma(a[m, k], b[k, n], acc[m, n])
```

This source order round-robins a K slice over independent accumulators before
returning to the same dependency chain. LLVM may still reschedule independent
instructions on the transient intrinsic path.

`a` and `b` must be matching rank-two BF16 or F16 tensors with dot-operand
layouts using `kWidth=4` or `8`. `acc` must be rank-two F32 with the
corresponding unit-tile MFMA layout. Matrix shapes and per-wave native
fragments must match.

| Argument | Meaning |
|----------|---------|
| `accumulator_role="transient"` | Phase-local chain lowered through LLVM-visible MFMA intrinsics, so LLVM models latency and hazards. |
| `accumulator_role="persistent"` | Chain carried across phases and lowered through register-constrained, side-effecting inline assembly. |
| `resident_operand=None` | No persistent source operand is selected for AGPR placement. |
| `resident_operand=0` / `1` | On the persistent path, select `a` / `b` for AGPR placement; the other source uses VGPRs. |
| `accumulator_register_class=None` | `auto`: persistent work uses AGPR; transient placement is left to LLVM. |
| `accumulator_register_class="vgpr"` / `"agpr"` | Select the persistent accumulator class explicitly. |
| `initialize=True` | Start each output chain's first native K update from zero, ignoring the supplied accumulator. |

`resident_operand` and an explicit accumulator class impose hard class
constraints only on the current persistent lowering. The transient intrinsic
path leaves physical placement to LLVM; use `amd_register_resident` separately
when a transient source needs an explicit allocation point. Set
`initialize=True` only for the first band of a multi-band accumulation, or the
earlier accumulated value will be discarded.

Because LLVM cannot model the latency or hazards of an MFMA hidden in inline
assembly, persistent lowering currently adds target-specific input wait padding
and a result drain. Those waits and the inline-assembly representation are
implementation details and should be included when comparing the two roles.

| Target | MFMA layout | Native instruction shapes | Persistent accumulator |
|--------|-------------|---------------------------|------------------------|
| gfx942 / CDNA3 | version 3 | `32x32x8`, `16x16x16` | Must explicitly use `accumulator_register_class="vgpr"`. |
| gfx950 / CDNA4 | version 4 | `32x32x16`, `16x16x32` | `auto` selects AGPR; explicit VGPR or AGPR is supported. |

gfx942 rejects every explicit `accumulator_register_class="agpr"` request.
Because persistent `auto` also resolves to AGPR, persistent gfx942 calls must
select `"vgpr"` explicitly.

All active lanes of a wave must execute `amd_scheduled_mfma` uniformly.

### `tlx.amd_mfma_commit`

```python
committed = tlx.amd_mfma_commit(acc)
committed, live = tlx.amd_mfma_commit(acc, preserve=live)
```

Applies the target-specific MFMA result-readiness boundary and returns every
input numerically unchanged. `value` may be one F32 MFMA-layout tensor or a
nonempty tuple of independent results. Optional `preserve` is one BF16 or F16
dot-operand tensor threaded through the same liveness and allocation boundary;
it is not another numerical MFMA operand.

| Call | Return value |
|------|--------------|
| `amd_mfma_commit(acc)` | `committed_acc` |
| `amd_mfma_commit((acc0, acc1))` | `(committed0, committed1)` |
| `amd_mfma_commit(acc, live)` | `(committed_acc, live_out)` |
| `amd_mfma_commit((acc0, acc1), live)` | `(committed0, committed1, live_out)` |

Each F32 input must be consumed only by this boundary, and downstream code must
use the returned result. When a `preserve` dependency remains live after the
boundary, its downstream uses must similarly use the returned `live_out`; the
returned dependency may be discarded when it is no longer needed.

The current lowering constrains a results-only boundary to AGPRs. A boundary
with `preserve` constrains its F32 results to VGPRs and the preserved dot
operand to AGPRs; directly committing an AGPR-resident scheduled-MFMA result
with `preserve` is rejected. These placement choices and the exact wait padding
are implementation details, not portable synchronization semantics.

`amd_mfma_commit` is not an MFMA queue instruction, `s_waitcnt`, hardware
memory fence, workgroup barrier, or cross-wave synchronization operation.

### Transient-chain example

Assume `a`, `b`, and `acc` already carry matching dot-operand and MFMA layouts:

```python
b_live = tlx.amd_register_resident(
    b,
    register_class="agpr",
    registers_per_group=4,
)
acc = tlx.amd_scheduled_mfma(
    a,
    b_live,
    acc,
    accumulator_role="transient",
    initialize=True,
)
acc, _ = tlx.amd_mfma_commit(acc, b_live)
tl.store(output_ptrs, acc)
```

For later accumulation bands, pass `initialize=False`. Here,
`amd_register_resident` creates a common allocation point for `b_live`, while
`preserve` keeps it live through the commit boundary. If later code needs that
operand, consume the returned value instead of `_`. `resident_operand` controls
only persistent lowering.

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
