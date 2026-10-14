# TMA multicast (compiler-directed, autoWS GEMM, sm100)

Scope: whether to spend search budget on compiler-directed TMA multicast
(`desc.load(..., multicast=True)` or the `multicast` compile option) for a
GEMM. The design is in
[triton_tma_multicast.md](../../../../../../../../docs/design/triton_tma_multicast.md).
TLX's explicit multicast is out of scope.

Code paths are relative to the repository root.

## Knobs

| Knob | How set | What it does | Existing doc |
|---|---|---|---|
| `multicast` | compile option (`triton.Config(multicast=True)`, default False) | Kernel-wide permission for TMA loads to multicast | [triton_tma_multicast.md, "Global opt-in"](../../../../../../../../docs/design/triton_tma_multicast.md#global-opt-in) |
| `multicast=` on a load | `desc.load(offsets, multicast=None/True/False)` | Per-load override of the kernel default | [triton_tma_multicast.md, "Per-load policy"](../../../../../../../../docs/design/triton_tma_multicast.md#per-load-policy) |
| `ctas_per_cga` | compile option | The cluster that multicast would broadcast within | [triton_tma_multicast.md, "Cluster geometry and program IDs under `ctas_per_cga`"](../../../../../../../../docs/design/triton_tma_multicast.md#cluster-geometry-and-program-ids-under-ctas_per_cga) |

## Never try (any problem)

| Condition | Status | Why | Code |
|---|---|---|---|
| any use of `multicast=True` / the `multicast` option in this tree | silently ignored | The planning pass records its choice only as the `tt.multicast_axes` attribute, and nothing reads it. Every descriptor-load lowering creates the TMA copy without multicast targets: the Meta WS load lowering, the non-WS TMA lowering and the software pipeliner | `lib/Dialect/TritonNvidiaGPU/Transforms/TMAMulticast.cpp`, `TritonNvidiaGPUTMAMulticastPass`; `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSLowerMem.cpp`, `optimizeTMALoads`; `lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp`, `TMALoadLowering`; `lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp`, `createTMAAsyncLoad` |

[triton_tma_multicast.md, "Current implementation"](../../../../../../../../docs/design/triton_tma_multicast.md#current-implementation) describes lowering steps (recipient masks, `.multicast::cluster` PTX) that the code does not perform yet. The same doc gives conflicting statements on Meta AutoWS support; with no lowering, neither applies.

## Mechanism

**What the planner would select.** It plans only when all of these hold (`TMAMulticast.cpp`, `analyzeTMAMulticast`, `TMAClusterGeometry::get`):
- the load is a tiled `tt.descriptor_load` in a single-block kernel;
- the cluster dimensions are powers of two, with 2 to 16 CTAs in total;
- the load's indices and descriptor are provably the same along a cluster axis;
- no enclosing `for`/`while`/`if` control value depends on a cluster axis of size > 1. A static persistent loop starting at `program_id(0)` under a `(2,1,1)` cluster fails this check. Only the validity bit of a CLC result counts as cluster-uniform; its tile coordinates do not (`ProgramIdDependencyAnalysis`).

**What multicast would relieve.** L2-to-SM traffic for an A or B tile that several CTAs of a cluster load. One elected CTA issues the copy and every recipient's barrier completes.

**What it would cost.** A cluster launch, cluster-wide barriers around each multicast transaction, and a schedule that keeps the loads cluster-uniform.

## When to try it (depends on the problem)

- **IF** tuning an autoWS GEMM in this tree, **THEN** skip `multicast=True` and any cluster shape chosen only to enable multicast, **BECAUSE** the generated code is identical to `multicast=False` (duplicate config).
- **IF** multicast lowering lands later, **THEN** re-derive these rules from the planner conditions above, **BECAUSE** a static persistent loop bounded by `program_id(0)` would still be rejected under an x-dimension cluster.

## Interactions

See [interactions.md](interactions.md). Pairs these rules depend on:
- `multicast` × `ctas_per_cga`
- `multicast` × `two_ctas` (see [2cta.md](2cta.md))
- `multicast` × tile scheduler (see [clc_persistence.md](clc_persistence.md))
- `multicast` × TMA operand loads

## Hazards

None known; see [hazards.md](hazards.md).

## Not yet established

See [TRACKER.md](TRACKER.md):
- Everything about multicast performance on GEMM: there is no lowering to measure.
