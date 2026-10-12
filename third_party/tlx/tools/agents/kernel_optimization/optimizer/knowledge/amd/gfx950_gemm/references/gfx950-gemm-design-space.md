# gfx950 GEMM Design Space

Use these mechanisms to form hypotheses. They are alternatives with measurable
crossovers, not a fixed preference order.

## Separate Work Distribution, Compute, And Movement

Choose device-level work distribution first: ordinary output tiling,
persistence, Stream-K or global Split-K for sparse device waves, and LocalSplitU
when very small M and deep K can profitably parallelize K inside one workgroup.
Count useful workgroups, complete CU rounds, and final-wave occupancy. Include
launch, workspace, reduction, and numerical-order costs.

For a regular native-MFMA wave grid:

```text
BlockM = MatrixInstM * MIWaveTileM * MIWaveGroupM
BlockN = MatrixInstN * MIWaveTileN * MIWaveGroupN
waves  = MIWaveGroupM * MIWaveGroupN
```

`MIWaveTile` counts native MFMAs and need not be a power of two. Irregular
logical macro tiles can remain homogeneous native-MFMA grids. Keep logical
compute decomposition, global/LDS transfer grouping, and physical LDS backing
as separate choices. A power-of-two backing may pad storage for an irregular
logical tile; load and compute only the logical windows, and never turn backing
padding into an out-of-bounds global access.

When multiple transfers populate subregions of one larger LDS backing, use
static non-reordering views and preserve the authored offset layout through
lowering. Verify that direct-to-LDS recognition follows those views to the
backing allocation; otherwise a legal logical decomposition can silently fall
back to extra register or LDS traffic.

Movement choices include register-resident operands, fragmented register
tiles, VGPR-to-LDS staging, direct-to-LDS, future-register/PGR2 banks, and
four-wave intra-wave or higher-wave inter-wave pipelines. Preserve several
bounded families when their dataflow or resource regimes genuinely differ;
share geometry, validation, and launch machinery rather than cloning a kernel
for individual shapes.

| Family | Useful signal | Primary risk |
|---|---|---|
| Register resident | Short K and low fixed overhead | Operand VGPR pressure and padding |
| Fragmented register | One output axis has costly tail waste | More fragments and source expansion |
| VGPR-to-LDS staged | K amortizes flexible load/publish scheduling | Payload VGPRs, LDS writes, barriers |
| Direct-to-LDS | Staging VGPR pressure dominates | Address/order legality and less scheduling freedom |
| PGR2/future banks | Independent memory and compute can overlap | Long live ranges, spills, occupancy loss |
| Inter-wave | More resident waves can hide latency | Synchronization and handoff gaps |
| Persistent/Stream-K | Natural grid leaves sparse device waves | Coordination and reduction overhead |
| LocalSplitU | Small M and deep K underfill the device | LDS reduction and a changed reduction tree |

Direct-to-LDS legality and profitability are separate decisions, per operand.
Prove lane mapping, vector width, alignment, LDS representation, tail handling,
and buffer reuse before benchmarking. Mixed direct/staged operands are valid.
Power-of-two geometry neither proves legality nor predicts a win.

For extreme rectangles, include the zero-copy algebraic orientation
`C.T = B.T @ A.T` when the input and output transposes are legal views. Recompute
tile utilization, CU rounds, load contiguity, locality, and store lowering in
the transposed view.

## Preserve Pipeline Invariants

Write the steady-state lifetime explicitly. A two-stage K pipeline may hold:

```text
operand registers : K(t), consumed now
current LDS       : K(t+1), ready to read
future memory     : K(t+2), in flight or prefetched
```

Every transition must preserve buffer ownership, async commit/wait depth,
barriers, and this next-iteration invariant. Source statement count is not a
machine schedule: one source load may lower to several VMEM or LDSDMA
instructions, and one local load may lower to several LDS reads.

For irregular K, prove alignment separately for each operand. Keep full K tiles
unmasked and at the widest legal vector width, isolate the reduction tail, and
zero-fill invalid reduction lanes. Retune the pipeline against the resulting
machine instructions. Never broaden an unaligned-load fact from unmasked loads
to masked loads, stores, atomics, or direct-to-LDS without a separate proof.

Partial M/N lanes may redirect input reads to a valid address when their output
is discarded by a masked store. Reduction-tail lanes must contribute zero. No
logical MFMA may consume uninitialized backing storage.

## Expose Legal Intra-Wave Windows

Use `tlx.warp_pipeline_stage(scope="intra_wave")` only after geometry, dataflow,
and resource allocation are competitive. The source owns semantic independence,
buffer-lifetime boundaries, and the operations admitted to a scheduling window;
the compiler owns lowered instruction pricing, exact placement, dependency
validation, waits, architectural hazards, and register allocation.

The current interface supports two structural forms:

- one region without `pair` contains independent read-only memory and MFMA work;
  the compiler partitions and interleaves the streams;
- exactly two adjacent regions with the same non-negative `pair` explicitly
  provide memory and compute streams for one window.

`pair` guards structure; it is not a cover-ratio tuning knob. Labels are
diagnostic. Cover assignment is compiler-owned. Both forms must fail closed on
dependencies, side effects, unsupported control flow, or malformed pairs.

Treat interleaving and outstanding prefetch depth as one budget. Moving loads
closer to MFMAs can look more regular while shortening the useful `vmcnt`
distance and reducing latency cover. Compare final waits and timing, not only
the visual load/MFMA alternation.

Do not place layout conversions that materialize through shared scratch inside
a phase-sensitive warp pipeline. Load directly into the consumer layout or pin
an already-valid layout. Inspect final AMDGCN: finer source windows may expose
more placement opportunities but also add address setup, scalar state, commit
groups, waits, or `m0` hazards.

## Treat Locality And Resources As Coupled

Reason about reuse at its actual scope: registers/LDS within a workgroup, GL1
across nearby workgroups on one CU, shared cache across CUs or XCDs, and HBM
when reuse does not survive scheduling. Tune cache policy per operand only after
measuring a residency or interference hypothesis. A streaming operand may
benefit from bypassing GL1 while the other retains it; symmetric policies are
not a default.

More prefetch, larger block K, or wider accumulator tiles can improve overlap
while increasing VGPR or LDS pressure enough to lose a resident wave. Repeated
source loads may already hit cache, so explicit shared-operand grouping can
lose to a simpler grid through extra live accumulators.

## Gate On The Final Artifact

For every serious candidate record:

```text
MFMA opcode and count
global/buffer load form and count
LDS read/write and direct-to-LDS count
wait, barrier, nop, and scratch traffic
VGPR, AGPR, LDS, scratch, and occupancy
grid, waves/workgroup, stage count, and store width
```

Use instruction traces to decide whether lower occupancy is offset by better
MFMA coverage or vice versa. Tune post-lowering scheduling only after the
kernel structure is competitive, and keep it configuration-scoped: a schedule
that helps one tile shape can regress another because instruction count,
dependencies, and live ranges change.

Keep workload-dependent choices such as tile ranking, wave family, mapping,
cache policy, and stage count in the library or autotuner. Move only reusable
semantic mechanisms, legality checks, and lowering-aware scheduling support
into the compiler.
