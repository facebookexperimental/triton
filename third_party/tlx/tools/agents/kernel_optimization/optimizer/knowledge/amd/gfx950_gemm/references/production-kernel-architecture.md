# gfx950 Production GEMM Catalog Architecture

Use this reference when optimizing or extending the public gfx950 implementation
in `third_party/tlx/ops/kernels/mm/gfx950.py`. For another kernel, preserve the
same separation of concerns rather than copying names or exact plans.

## Follow Five Layers

```text
public API and validation
    chooses the supported workload contract
        ↓
family dispatcher and bounded geometry selector
    chooses a measured algorithm and legal plan
        ↓
compile-time plan
    fixes MFMA geometry, staging, mapping, and compiler controls
        ↓
shared kernel family
    materializes loads, LDS, MFMA ownership, and epilogue
        ↓
compiler and final AMDGCN
    lower, schedule, allocate registers, and expose the real artifact
```

Keep these layers distinct. A high-ranked geometry that the current source
cannot execute is a capability gap, not proof that the geometry is poor. A
kernel that compiles is not automatically safe for production dispatch.

## Read One Selected Path, Not The Whole File

Start from the public path and follow one concrete workload:

1. `mm` owns public validation, `out=` reuse/validation, and full versus
   heuristic search-space behavior.
2. `_dispatch_for`, `_dispatch_plan`, and `heuristic_config` choose a family and
   complete plan. Record which selector supplied the decision.
3. `_launch_dispatch` maps the family label to its launcher. A transposed plan
   may launch `B.T @ A.T` into `out.T` without a copy.
4. `_wg_regular_wave_grid_plan` and the relevant launcher define the parameter
   schema, grid, compiler options, and legal geometry.
5. `_wg_kernel_regular_mi16_wave_grid` or
   `_wg_kernel_regular_mi32_wave_grid` owns the regular wave-grid allocation,
   pipeline selection, MFMA layout, and epilogue.
6. Follow the selected compute-tile, stage, load, and local-view helpers for one
   K iteration only.
7. Inspect final TTGIR/AMDGCN and profiles to verify the intended plan survived
   lowering.

Do not count every `@triton.jit` helper as a launchable GPU kernel. Identify
actual `kernel[grid](...)` launch sites, then group them by dataflow. The public
catalog currently uses reusable register-resident, regular wave-grid,
inter-wave LDS, Stream-K, persistent, and LocalSplitU compute families, with
auxiliary reduction launches where required. Recompute the inventory after
structural changes rather than freezing historical counts.

Other important family entry points include `_register_kernel_impl`, the
inter-wave launcher, `streamk_kernel`, `_persistent_kernel`, and
`_local_split_u_kernel`. These are bounded alternatives, not a request to force
every workload through one implementation.

## Make Plans Complete And Inspectable

A plan should identify all fields that can change generated code or launch
behavior:

```text
family and orientation
matrix instruction and logical K width
MI wave tile and wave group
BlockM, BlockN, BlockK, and stage/buffer count
register, LDS, direct-to-LDS, and PGR/future-bank choices per operand
Split-K, Stream-K, LocalSplitU, or persistent ownership
grouping, workgroup mapping, and XCD policy
cache modifiers and trusted alignment/contiguity facts
waves_per_eu, warps, compiler scheduling, and register policy
launch grid, auxiliary workspace/reduction, and output-store mapping
```

The dispatch manifest used for rebase or refactor validation must include these
fields, not only a family label.

## Share Geometry Without Forcing One Dataflow

For MI16 regular grids:

```text
BlockM = 16 * MIWaveTileM * WarpsM
BlockN = 16 * MIWaveTileN * WarpsN
waves  = WarpsM * WarpsN
```

`MIWaveTileM/N` are native-MFMA counts and may be non-powers of two. Logical
compute fragments, transfer grouping, and physical LDS backing are independent.
For example, adjacent logical operands can share a wider transfer and then be
recovered through constant LDS views without merging their compute lifetimes.

Keep common plan validation, mapping, boundary handling, and launch code shared.
Add a new family only when the algorithm, synchronization, or resource regime
is genuinely different. Do not create one device kernel per shape.

## Distinguish Data-Movement Terms Precisely

```text
register resident : global -> operand VGPR -> MFMA
VGPR-to-LDS staged: global -> payload VGPR -> LDS -> operand VGPR -> MFMA
direct-to-LDS     : global -> LDS -> operand VGPR -> MFMA
PGR/future banks  : multiple global/operand banks overlap future K states
```

Direct-to-LDS is gated per operand. First prove legality of the lane/global
mapping, vector width, alignment, destination representation, views, padding,
tail, and stage reuse. Then benchmark against VGPR staging. Mixed operand
policies are valid. Power-of-two geometry neither proves legality nor predicts
profitability.

Fragmented register paths should decompose only the axis where padding is
expensive and reuse the opposite operand. Choose the resident side by both live
bytes/occupancy and the number of independent MFMAs each streamed read can
cover. Fewer registers can lose when it replaces useful residency with repeated
short-runway waits.

## Treat Dispatch As Measured Policy

The full capability space, heuristic selector, and promotion cache have
different jobs:

- the full space exposes legal candidate plans;
- the heuristic maps broad workload features to a small starting plan set;
- measured promotions admit bounded winners and preserve fallbacks.

Exact-shape entries may be a temporary tuning cache, but they do not define
support. Generalize only when a compact predicate based on work, layout,
resource, or device-wave features survives endpoints, interior cases, negative
neighbors, and the old corpus.

After a refactor, compare public dispatch manifests and final artifacts across
the protected corpus. AST-equivalent helpers do not prove equivalent behavior
when selectors, defaults, cache keys, or compiler options changed.
