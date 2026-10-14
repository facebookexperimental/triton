# TMA multicast (compiler-directed, autoWS GEMM, sm100)

Scope: whether to spend search budget on compiler-directed TMA multicast
(`desc.load(..., multicast=True)` or the `multicast` compile option) for a
GEMM. The design is in
[triton_tma_multicast.md](../../../../../../../../docs/design/triton_tma_multicast.md).
TLX's explicit multicast is out of scope.

Code paths are relative to the repository root.

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| `multicast` (per load) | `desc.load(multicast=)` | None / True / False | None | P1 | Silently ignored in this tree: no TMA lowering reads the planned axes. | Nick's multicast ghstack: lowering #2260, AutoWS support #2893, #2894, #2876, e2e #2259 (all OPEN); then a power-of-2 cluster of 2-16 CTAs. |
| `multicast` | compile option | bool | False | P1 | Same as the per-load row. | Same as the per-load row. |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| any use of `multicast=True` / the `multicast` option in this tree | silently ignored | The planning pass records its choice only as the `tt.multicast_axes` attribute, and nothing reads it. |

[triton_tma_multicast.md, "Current implementation"](../../../../../../../../docs/design/triton_tma_multicast.md#current-implementation) describes lowering steps (recipient masks, `.multicast::cluster` PTX) that the code does not perform yet. The same doc gives conflicting statements on Meta AutoWS support; with no lowering, neither applies.

## Mechanism

- **Eligible loads:** the planner plans only a tiled `tt.descriptor_load` in a single-block kernel whose indices and descriptor are provably the same along a cluster axis (`TMAMulticast.cpp`, `analyzeTMAMulticast`); all conditions in these first three bullets must hold.
- **Cluster geometry:** cluster dimensions must be powers of two, 2 to 16 CTAs in total (`TMAMulticast.cpp`, `TMAClusterGeometry::get`).
- **Cluster-uniform control flow:** no enclosing `for`/`while`/`if` control value may depend on a cluster axis of size > 1; a static persistent loop starting at `program_id(0)` under a `(2,1,1)` cluster fails this, and only a CLC result's validity bit counts as cluster-uniform, not its tile coordinates (`ProgramIdDependencyAnalysis`).
- **Multicast copy:** one elected CTA would issue the copy and every recipient's barrier completes -> would relieve L2-to-SM traffic for an A or B tile that several CTAs of a cluster load -> would cost a cluster launch, cluster-wide barriers around each multicast transaction, and a schedule that keeps the loads cluster-uniform.

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
