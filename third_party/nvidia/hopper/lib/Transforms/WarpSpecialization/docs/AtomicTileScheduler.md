# Atomic Tile Scheduler

`AtomicTileScheduler.cpp` extends Triton's dynamic persistent tile scheduler to
physical CTA clusters. It changes a scheduler claim from "reserve one tile" to
"reserve one contiguous group of tiles," then gives one tile ID to each CTA in
the cluster.

The transformation is deliberately split into two passes around automatic warp
specialization (AutoWS):

```text
atomic-tile-scheduler-prepare
            |
            v
     optional AutoWS
            |
            v
atomic-tile-scheduler-materialize
```

The prepare pass proves that the scheduler has a supported shape, rewrites its
initial tile ID into physical-cluster order, and marks the atomic claim. The
materialize pass lowers the marked claim after AutoWS has finished partitioning
the loop. This keeps cluster-specific synchronization out of generic code
partitioning.

## Supported IR Shape

The pass recognizes the canonical dynamic persistent scheduler:

* An `scf.while` containing a two-CTA dot carries the next tile ID.
* The loop's after-region contains exactly one candidate tile-claim atomic.
* The claim is a scalar i32 `atomic_add` of constant one, with a true mask and
  GPU or system scope.
* The counter pointer is a uniform kernel argument.
* The atomic result is forwarded directly through `scf.yield`.
* The initial carried tile ID is directly `program_id(0)`.
* The loop condition is the signed comparison `tile_id < num_tiles`.
* `num_tiles` is cluster-uniform and provably divisible by the physical cluster
  size.
* The launch uses an explicit physical CTA cluster while the logical Triton
  program still has one CTA (`ttg.num-ctas = 1`).

The transformation is intentionally narrow. Loops that are not candidate
two-CTA dot schedulers are ignored. Once a loop is identified as a clustered
dynamic scheduler, however, an invalid atomic, cluster configuration, or
tile-count contract is a compilation error. The pass never partially rewrites
an invalid candidate.

## Prepare Pass

`atomic-tile-scheduler-prepare` runs before AutoWS. It first requires an
explicit physical cluster with logical `ttg.num-ctas == 1`; then, for each
candidate claim, it:

1. Recognizes the scheduler atomic and its enclosing loop.
2. Computes the physical cluster size
   `K = clusterDimX * clusterDimY * clusterDimZ`.
3. Verifies the assumptions needed to reserve `K` consecutive tile IDs.
4. Replaces the initial `program_id(0)` with a cluster-major, X-fastest linear
   tile ID whose `K` consecutive values belong to one physical cluster.
5. Attaches the cluster-claim marker and the proven cluster size to the atomic.

No synchronization is introduced at this stage. The marker and recorded size
survive optional AutoWS and tell the late materializer which claim to lower.
AutoWS independently recognizes the loop-carried atomic as a run-once producer,
assigns it to one owner partition, and broadcasts its result.

The registered pass name is
`triton-nvidia-gpu-atomic-tile-scheduler-prepare`. The NVIDIA pipeline places it
immediately after conversion to TTGIR and the two-CTA legality check, before
TTGIR optimization and optional AutoWS.

## AutoWS Interaction

When AutoWS accepts the loop, its existing atomic broadcast machinery places
the marked atomic in one owner partition and forwards the result to consumer
partitions through the normal buffered partition channel. The atomic tile
scheduler does not create a second cross-partition synchronization mechanism.

The late materialization therefore has two useful properties:

* Only the owner partition executes the cross-CTA claim protocol.
* Compute partitions consume the resulting tile ID through the same channel
  used for other run-once atomics.

Without AutoWS, the materialize pass still lowers the marked atomic in the
ordinary loop. The cluster protocol is independent of whether the loop was
partitioned.

If AutoWS cannot represent the run-once atomic broadcast, its existing bailout
removes the specialization metadata before code partitioning. The late pass can
then materialize the same marked claim in the unspecialized loop. No
cluster-specific case is required in `doCodePartition`.

## Materialize Pass

`atomic-tile-scheduler-materialize` runs after optional AutoWS. It replaces the
marked scalar claim with a cluster-wide handoff:

1. Physical cluster rank zero waits until the previous handoff buffer is free.
2. Rank zero performs `atomic_add(counter, K)`, reserving `K` consecutive tile
   IDs with one global atomic operation.
3. Rank zero stores the returned base tile ID in its local shared-memory slot
   and DSM-stores it into the corresponding slot of every other CTA.
4. Rank zero signals each CTA's full barrier.
5. Each CTA waits on its local full barrier and loads the base tile ID.
6. Each CTA computes `tileId = base + clusterCtaRank`.
7. Each CTA arrives at rank zero's empty barrier, allowing the buffer to be
   reused after all CTAs have consumed the value.

The registered pass name is
`triton-nvidia-gpu-atomic-tile-scheduler-materialize`. The NVIDIA pipeline runs
it late, after optional warp specialization has created the physical owner
partition and before CLC materialization and final cross-CTA synchronization.

In pseudocode:

```text
if clusterCtaRank == 0:
  wait(empty, previousPhase)
  base = atomic_add(counter, clusterSize)
  for rank in cluster:
    dsm_store(pidSlot[rank], base)
    arrive(full[rank])

wait(full[clusterCtaRank], phase)
base = load(pidSlot)
tileId = base + clusterCtaRank
remote_arrive(empty[0])
```

The pass uses ordinary TTGIR mbarrier wait/arrive operations and distributed
shared-memory addressing. It does not require a cluster-wide barrier: a cluster
barrier would make every thread wait and would duplicate the synchronization
already represented by the producer/consumer handoff. Under AutoWS, only the
owner warp group participates in this protocol.

## Buffering and Phase

The PID slot and its full/empty barriers are function-lifetime allocations. The
loop carries a phase bit so a barrier instance can be reused safely across
iterations. The empty barrier has an arrival count of `K`, because every CTA
must finish reading the current base before rank zero overwrites the slot. Each
full barrier has one producer arrival from rank zero.

When the atomic lives inside an isolated warp-specialized partition, the pass
captures these allocations into that partition. The shared
[cluster handoff utilities](ClusterHandoff.md) keep this capture and
remote-addressing behavior consistent with CLC lowering.

The materializer also preserves the atomic's `async_task_id` on the replacement
operations and carries a phase value through the rebuilt `scf.while`. This keeps
the new protocol in the run-once owner and toggles the mbarrier parity once per
persistent iteration.

## Physical and Logical Clusters

This pass is a backend physical-cluster extension. It does not reinterpret
logical multi-CTA Triton programs. The supported configuration has:

* a logical CTA count of one in the TTGIR layout, and
* explicit physical cluster dimensions on the kernel.

The physical cluster rank is linearized with X as the fastest-varying
dimension. That rank selects the tile offset within the contiguous block
reserved by rank zero.

## Correctness Invariants

The lowering relies on the following invariants established or checked by the
two passes:

* Exactly one CTA performs the global atomic claim for a cluster iteration.
* The atomic increment is the physical cluster size.
* Every CTA receives the same base tile ID before adding its unique rank.
* Rank zero cannot reuse the PID slot until all CTAs have consumed it.
* A marked claim remains in the expected scheduler loop through AutoWS.
* The physical cluster configuration at materialization matches the size
  recorded by preparation.

A mismatch in a prepared clustered scheduler is diagnosed as a pass failure,
not partially lowered into a synchronized loop.

## Tests

`test/Hopper/TwoCTA/check_matmul_two_cta.mlir` checks preparation and
materialization together, including nontrivial physical cluster shapes, the
single `atomic_add(counter, K)`, DSM publication, and full/empty barriers.
`test/TritonNvidiaGPU/invalid.mlir` covers rejected scheduler shapes and
diagnostics. The tutorial AutoWS tests in
`python/test/unit/language/test_tutorial09_warp_specialization.py` exercise the
integrated compiler pipeline.

## Related Documentation

* [Overview](Overview.md)
* [Cluster Handoff Utilities](ClusterHandoff.md)
* [Cross-Partition Atomic Support](CrossPartitionAtomicSupport.md)
* [Code Partition](CodePartition.md)
