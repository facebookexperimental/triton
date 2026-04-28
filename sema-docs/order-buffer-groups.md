# Ordering NVWS buffer groups by dependency readiness

[← Back to README](README.md)

## Contents

- [Purpose](#purpose)
- [Bridge position](#bridge-position)
- [Function and group eligibility](#function-and-group-eligibility)
- [Memory predecessors](#memory-predecessors)
- [Readiness model](#readiness-model)
- [Pairwise preferences](#pairwise-preferences)
- [From preferences to an order](#from-preferences-to-an-order)
- [Movement-safety checks and exact mutation](#movement-safety-checks-and-exact-mutation)
- [Output and non-goals](#output-and-non-goals)
- [Test example](#test-example)
- [Implementation and test map](#implementation-and-test-map)

## Purpose

`NVWSOrderBufferGroups` (`--nvws-order-buffer-groups`) is a narrow scheduling
policy for the MetaAWS–NVWS bridge. Group collection emits TMEM groups before
SMEM groups. The pass reorders only eligible TMEM groups within that TMEM
prefix; it never reorders SMEM groups. Buffer IDs initially follow allocation
first-sighting order. `NVWSInsertSemas` later renders active groups in
collection order. Consequently, semaphore acquires inserted at one consumer
can inherit the canonical MetaAWS memory planner's TMEM allocation order
instead of an explicit latency policy.

This pass makes that order deliberate for a restricted class of TMEM groups.
When two such groups have incoming cross-owner dependency sources at the same
access and those sources have sufficiently different modeled readiness, the
group with the later-ready source path is placed first. `NVWSInsertSemas` then
discovers and renders that group first, so the wait for the modeled later-ready
source precedes the wait for the earlier-ready source.

The transformation is heuristic and conservative. Missing schedule data,
small readiness differences, contradictory observations, preference cycles,
or an unsafe allocation layout cause constraints to be omitted or ordering to
remain unchanged; they do not make otherwise valid IR invalid.

[↑ Back to contents](#contents)

## Bridge position

The relevant bridge order, including its verification boundaries, is:

```text
canonical MemoryPlanner
-> AnnotateTMAStoreWaits and TMA-store-wait validation
-> final MetaToNVWSConvert
-> partition verifier

-> NVWSOrderBufferGroups

-> partition verifier
-> NVWSInsertSemas
```

The pass runs after the **final** `MetaToNVWSConvert` because it consumes the
completed memory plan and NVWS ownership representation:

- `buffer.id` defines the groups whose discovery order will matter;
- `buffer.copy` identifies planned allocation members eligible for this
  policy;
- source-free `ttng.tmem_alloc` operations are in their final NVWS form; and
- partition plus `loop.stage`/`loop.cluster` metadata lets the access analysis
  identify cross-owner dependencies and compare incoming dependency sources in
  one schedule coordinate.

The optional early conversion used by
`TRITON_NVWS_USE_META_NVWS_ALLOCAS=1` is not this boundary. Canonical memory
planning still runs after that conversion and completes the memory plan, so
ordering is deferred until the common final conversion.

The pass must run before `NVWSInsertSemas`. Group collection keys the TMEM
group sequence by the first allocation encountered for each buffer ID, and
the semaphore emitter processes active groups in that sequence. Reordering
the representative allocations before collection therefore controls the
order of co-located acquires without coupling this latency policy to semaphore
construction. After `NVWSInsertSemas` has materialized backings, tokens, and
protocol operations, the allocation order is no longer a useful policy
boundary.

In the current driver this pass is part of the MetaAWS–NVWS bridge only. It is
followed by the warp-specialization partition verifier, just like the final
conversion and the mechanism passes that follow it.

[↑ Back to contents](#contents)

## Function and group eligibility

The pass visits every `tt.func` in its module. A function without an
`scf.for` marked `tt.warp_specialize` is ignored. Otherwise it uses the same
`collectGroups` and `buildAccessDag` implementation as `NVWSInsertSemas`.
The shared locality validator runs first. The pass then collects groups and
builds access DAGs separately in each top-level function block. If one block
contains at least two groups, construction is performed for every group in
that block before the ordering heuristic is applied; malformed input that the
shared builder rejects is a pass failure.

A group can participate in ordering only when all of these conditions hold:

- it is a TMEM group;
- it has at least one allocation member;
- its `buffer.id` is nonnegative, excluding the negative synthetic IDs that
  group collection assigns to allocations without an ID; and
- every member is a source-free `ttng.tmem_alloc` directly in the current
  top-level function block
  and carries `buffer.copy`.

The first allocation member is the group's representative. Moving that one
operation is enough to change the group's first-sighting position the next
time `collectGroups` runs, even when several allocations share the same
`buffer.id`.

At least two groups in one block must exist for analysis to proceed, and at
least two eligible groups in that block must exist for any ordering mutation.
SMEM groups, sourceful TMEM allocations, unplanned TMEM allocations without
`buffer.copy`, negative-ID groups, and groups whose members are not directly
in the block being processed never receive ordering preferences.

[↑ Back to contents](#contents)

## Memory predecessors

SSA use-def edges do not represent ordering through a mutable buffer, so the
pass augments them with simple memory predecessors derived from each group's
access DAG.

Access nodes are partitioned by their actual containing block and processed in
lexical order within that block. State is maintained independently for every
exact alias piece identified by the access DAG. For each piece the pass tracks
the last writer and the outstanding readers:

- a read depends on the last writer, then joins the reader set;
- a write depends on the last writer and on every outstanding reader, then
  clears the readers and becomes the new last writer; and
- multiple touches by one access node are merged per piece, with write taking
  precedence over read.

Thus the added edges cover same-block read-after-write, write-after-write, and
write-after-read ordering. Exact pieces prevent disjoint members of one
backing group from creating false memory dependencies. The scan does not infer
additional memory order between different blocks.

Every such source becomes a memory predecessor for readiness scoring,
regardless of ownership. The subset whose owner differs from the current
access owner is also recorded as that group's incoming cross-owner sources at
the access. Those incoming sources are what allow two groups to be compared at
a common consumer.

[↑ Back to contents](#contents)

## Readiness model

`ReadinessScorer` recursively computes a latency score for an operation from
two predecessor classes:

1. the defining operation of each SSA operand, when one exists; and
2. the same-block memory predecessors described above.

Conceptually:

```text
start(op) = max(ready(ssa-defs), ready(memory-predecessors), 0)
ready(op) = min(INT32_MAX,
                start(op) + max(LatencyModel(op), 0))
```

Scores are memoized. A dependency cycle makes that score unavailable rather
than guessing a value. Negative latency-model results contribute zero, and
the accumulated result saturates at `INT32_MAX`.

For one eligible group at one consumer, the pass summarizes the incoming
cross-owner sources as follows:

- every source must have a readiness score;
- every source must have `loop.stage` and `loop.cluster` schedule data;
- all sources for that group must have the same stage/cluster coordinate; and
- the group readiness is the maximum readiness of those sources.

If any condition is missing, that group has no readiness summary at that
consumer. The comparison uses the incoming sources' common schedule
coordinate; it does not use the consumer's schedule as a substitute.

[↑ Back to contents](#contents)

## Pairwise preferences

Eligible groups are compared only when the same access operation has nonempty
incoming cross-owner sources from both groups. A comparison creates a
preference only when:

- both group summaries are available;
- the two summaries have the same stage/cluster coordinate; and
- their readiness scores differ by at least 32 latency-model units.

The threshold is inclusive: a difference of exactly 32 qualifies. The group
with the larger readiness score is preferred before the group with the smaller
score. Scores closer than 32 are treated as indistinguishable, preserving
stability instead of ordering on latency-model noise.

The same unordered group pair can meet at more than one consumer. Repeated
observations in the same direction reinforce one preference. If different
consumers prefer opposite directions, the pair is marked conflicting and no
edge is retained for it. A missing or nonqualifying observation is simply
ignored; it does not conflict with an existing qualifying observation.

[↑ Back to contents](#contents)

## From preferences to an order

Each nonconflicting preference becomes a directed edge from the preferred
earlier group to the other group. The pass computes a stable topological order
of eligible groups. Whenever several groups currently have zero indegree, it
selects the one earliest in the original discovery order. Unconstrained groups
therefore receive deterministic, authored-order tie-breaking, although a
retained edge can indirectly change the relative order of groups with no
direct edge between them.

Conflicting pairs contribute no edge, while other valid pairs may still
reorder groups. If the retained graph contains a cycle, topological sorting
cannot complete and the pass preserves the authored order for that top-level
function block, then continues with the remaining blocks. No allocations in
that block have been moved at that point.

[↑ Back to contents](#contents)

## Movement-safety checks and exact mutation

Before mutating IR, the pass finds the earliest and latest eligible
representatives in the current top-level function block. Every operation in
that inclusive lexical span must be one of:

- a source-free `ttng.tmem_alloc`; or
- a source-free `ttg.local_alloc`.

Any sourceful allocation or any other operation in the span causes a
successful no-op for the current block; processing continues in the remaining
function blocks. This prevents the heuristic from moving an allocation across
computation, control flow, or an operation whose source would impose an SSA
dependency.

The accepted topological order is realized from right to left. When a
representative that should be earlier currently follows its successor, the
pass moves that representative immediately before the successor. A per-move
check again requires both operations to be in the same block, requires the
moved operation to be source-free, and permits it to cross only source-free
TMEM or local allocations. Operations already in the required relative order
are left in place.

Only the first `ttng.tmem_alloc` representing each eligible group is moved.
No operation is cloned or erased, and no operands, results, types, attributes,
owners, or schedule coordinates are changed. Other source-free allocations in
the span are not selected by the latency policy, although an eligible
representative may move across them.

[↑ Back to contents](#contents)

## Output and non-goals

The output is the same NVWS IR with, at most, a different lexical order for
eligible group representatives. That lexical change is an input policy for
the subsequent group collector and semaphore emitter.

The pass does **not**:

- insert, remove, combine, or relocate semaphore operations;
- change a buffer ID, offset, copy count, reuse decision, or backing;
- choose partitions, owners, stages, or clusters;
- reschedule producer or consumer computation;
- order SMEM, sourceful TMEM, or otherwise ineligible groups by readiness;
- infer cross-block mutable-memory ordering (only SSA defining-operation edges
  cross blocks in the readiness recursion); or
- claim a globally optimal wait order.

Unavailable heuristic evidence is not diagnosed as an error. By contrast,
failure of the shared group collector or access-DAG builder is a real pass
failure, because `NVWSInsertSemas` would be unable to consume that input
reliably as well.

[↑ Back to contents](#contents)

## Test example

`test/NVWS/order_buffer_groups.mlir` contains two focused cases.

In `@orders_latest_ready_first`, the accumulator group (`buffer.id = 5`) is
authored before the input group (`buffer.id = 7`). Both are source-free,
copy-one entry-block TMEM allocations. Stores in partitions 0 and 2 produce
the two buffers at the same stage/cluster coordinate, and one
`ttng.tc_gen5_mma` in partition 1 consumes both groups. The input producer has
the longer modeled path:

```text
accumulator: addf -> tmem_store ---------+
                                            -> tc_gen5_mma
input:       exp2 -> fp_to_fp -> tmem_store +
```

The readiness gap qualifies, so the pass moves the input representative before
the accumulator representative:

```text
// Authored
ttng.tmem_alloc {buffer.id = 5, buffer.copy = 1}
ttng.tmem_alloc {buffer.id = 7, buffer.copy = 1}

// After --nvws-order-buffer-groups
ttng.tmem_alloc {buffer.id = 7, buffer.copy = 1}
ttng.tmem_alloc {buffer.id = 5, buffer.copy = 1}
```

In `@equal_readiness_is_stable`, the two modeled producer paths do not create a
qualifying readiness gap. The authored order, ID 11 before ID 10, remains
unchanged. Together the cases verify later-ready-first ordering and
equal-readiness stability.

[↑ Back to contents](#contents)

## Implementation and test map

| File | Responsibility |
|---|---|
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/OrderBufferGroups.cpp` | Eligibility, exact-piece memory predecessors, readiness scoring, pairwise preferences, stable topological ordering, safety checks, and allocation movement. |
| `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td` | Pass registration, command-line name, summary, and dependent dialects. |
| `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp` | Placement after final `MetaToNVWSConvert`; its partition verifier runs before `NVWSInsertSemas`. |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasAccessDag.cpp` | Shared group discovery and access-DAG construction whose first-sighting order the pass controls. |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp` | Re-collects groups and constructs synchronization plans in discovery order. |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitIR.cpp` | Renders active groups sequentially, preserving that order for co-located acquires. |
| `third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/LatencyModel.h` | Latency model used by readiness scoring. |
| `test/NVWS/order_buffer_groups.mlir` | Later-ready-first reordering and equal-readiness stability coverage. |

[↑ Back to contents](#contents)
