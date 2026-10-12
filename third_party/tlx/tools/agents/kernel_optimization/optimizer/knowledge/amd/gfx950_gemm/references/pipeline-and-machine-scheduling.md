# gfx950 GEMM Pipeline And Machine Scheduling

Use this reference after a competitive tile, work distribution, and data-
movement family exist. Scheduling can expose latency already present in a good
dataflow; it rarely repairs excess padded work, an unsuitable MFMA grid, or an
occupancy collapse caused by the kernel architecture.

## Describe The Steady State First

Write the state carried across one K-loop boundary before moving operations. A
typical staged loop may contain:

```text
compute state : accumulator and operand fragments for K(t)
published LDS : complete tiles for K(t+1)
future state  : addresses, transactions, or payload fragments for K(t+2)
```

For every stage, identify who owns each buffer, when its writes become visible,
which wait protects its first read, and which barrier prevents premature reuse.
Prologue, steady state, and epilogue must establish and discharge the same
invariants. An optimization is invalid if it works only after the first loop
trip or reads a future bank on the final trip.

Do not infer the machine pipeline from source order. One source load may lower
to several VMEM or LDSDMA instructions; a local load or dot may expand into
multiple LDS reads, moves, and MFMAs. Layout conversion can add scratch-backed
LDS traffic. Price and validate the lowered instruction stream.

## Choose Inter-Wave Or Intra-Wave Deliberately

The two forms solve different latency-hiding problems:

- **Inter-wave scheduling** assigns memory and compute phases to different
  waves or wave groups and phase-shifts their progress. It can increase the
  number of independent instructions visible to the machine at once, but its
  handoff barriers, ownership rules, and phase offsets are semantic.
- **Intra-wave scheduling** exposes independent memory and MFMA work within one
  wave so the backend can interleave their lowered instructions before register
  allocation. It avoids cross-wave ownership but can lengthen fragment live
  ranges or consume the prefetch runway too early.

Do not assume that a successful form should replace the other globally. Compare
resident waves, barrier cost, live ranges, and the final issue pattern for the
specific plan.

For inter-wave pipelines, treat every synchronization operation as load-bearing
until the ownership proof says otherwise. Check that each producer publishes a
complete stage before a consumer reads it and that no wave overwrites the stage
while another wave still consumes it. Measure the handoff gap around the first
consumer MFMA and the last producer memory operation; more nominal overlap can
lose if the phase boundary drains the machine.

## Use The Current Intra-Wave Source Contract

`tlx.warp_pipeline_stage(scope="intra_wave")` marks a source-approved scheduling
window. It has two structural forms:

```python
# One mixed region. The compiler partitions independent read-only memory and
# MFMA work and validates whether they can be interleaved.
with tlx.warp_pipeline_stage("load_and_compute", scope="intra_wave"):
    future = tl.load(future_ptrs)
    acc = tl.dot(a_tile, b_tile, acc)

# One explicit window. Exactly two adjacent regions with the same pair ID
# provide the memory and compute streams.
with tlx.warp_pipeline_stage("publish", scope="intra_wave", pair=0):
    future = tl.load(future_ptrs)
with tlx.warp_pipeline_stage("compute", scope="intra_wave", pair=0):
    acc = tl.dot(a_tile, b_tile, acc)
```

The `pair` ID prevents accidental association with a neighboring region. It is
a structural guard, not a scheduling ratio or autotuning knob. Labels describe
intent but do not establish legality. Do not use removed tuning arguments or
invent source-level instruction counts.

The source is responsible for:

- admitting only operations intentionally exposed for motion;
- ensuring memory and compute streams are semantically independent;
- preserving buffer ownership and loop-carried lifetime boundaries; and
- excluding side effects, unsupported control flow, and hidden scratch-backed
  layout conversion.

The compiler is responsible for partitioning a mixed region, validating an
explicit pair, pricing the represented machine instructions, choosing cover,
respecting dependencies and architectural hazards, and materializing waits.
Both forms must fail closed when these facts cannot be proved.

An unannotated source sequence and a mixed annotated region are not equivalent.
The annotation is the source authorization that the backend may partition and
move the admitted lowered work across its original operation order. Raw AMD
scheduling barriers can constrain already-lowered groups, but they cannot
recover source-level independence or safely invent a scheduling window.

## Separate Three K-Loop Levers

Do not use `BlockK`, source unrolling, and buffer count as synonyms:

- Increasing **BlockK** changes one logical stage's K extent. It can reduce
  barrier and loop-control density, but increases LDS footprint and often the
  amount of data simultaneously live.
- Increasing **unroll or K work per loop body** keeps the physical stage
  geometry but exposes several stage transitions in one compiler scheduling
  window. It may improve instruction visibility without enlarging each LDS
  buffer, at the cost of code size, address state, and longer live ranges.
- Increasing **stage or future-bank count** adds concurrently owned K states.
  It may hide longer latency but consumes LDS or registers and changes wait and
  reuse distances.

Test these separately. A larger K body that helps a shallow dependency chain
can hurt a configuration whose occupancy is already limited by VGPRs or LDS.
Non-power-of-two K bodies are legal only when every stage and tail transition is
proved, not because the aggregate K happens to be divisible.

## Treat Interleaving And Prefetch As One Budget

Moving future loads closer to current MFMAs can produce a visually regular
load/MFMA sequence while reducing the distance between issue and the protecting
`vmcnt`. Conversely, issuing every future load at the top may create enough
outstanding traffic but retain too many addresses and fragments.

For each candidate compare:

```text
future-load issue -> first wait distance
first LDS read -> first dependent MFMA distance
MFMA issue gaps and dependency chains
simultaneously live future/current fragments
VMEM, LDS, and MFMA instructions between waits
barrier and stage-reuse locations
```

Vary load grouping, MFMA cover, and future distance as one bounded experiment.
Reject schedules that improve alternation but add wait drains, address setup,
scratch, or enough registers to lose a resident wave.

## Price Cover After Lowering

A source operation is not a stable unit of scheduler cover. Compute cover using
the actual lowering width and instruction class represented by the admitted
TTGIR operations. Static subslices, direct-to-LDS loads, and different MFMA
shapes can change the count without changing the number of source statements.

Keep cover selection compiler-owned when it can be derived from the lowered
stream. Expose a source option only after a real workload proves that semantic
intent cannot express the necessary choice and a stable user-visible contract
exists.

Scheduling is configuration-scoped. A ratio that is effective for one
BlockM/BlockN/BlockK and wave grid can regress another due to different
instruction multiplicity, dependency shape, register allocation, or post-RA
movement. Do not encode one historical cover ratio as an architecture default.

## Validate The Final Schedule

Check both pre-register-allocation intent and final AMDGCN. Register allocation
can lengthen or collapse the intended groups, add spills, or force a different
wait pattern. Record:

- VGPR/AGPR use, scratch, LDS, and resulting occupancy;
- VMEM/LDS/MFMA ordering and normalized instruction counts;
- `vmcnt` and `lgkmcnt` placement and whether waits frequently drain to zero;
- barriers, nops, scalar address work, and direct-to-LDS setup;
- first-iteration, steady-state, and final-iteration behavior; and
- wall-clock distribution under the production timing contract.

Promote a scheduling change only when the final artifact contains the intended
mechanism, correctness covers multiple K-loop and tail cases, and the measured
gain survives neighbors and the protected regression corpus.
