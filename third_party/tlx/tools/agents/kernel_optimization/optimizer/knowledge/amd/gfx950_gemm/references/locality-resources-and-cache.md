# gfx950 GEMM Locality, Resources, And Cache

Use this reference to reason about occupancy, operand reuse, LDS, and cache as a
coupled system. Resource counts are constraints on the achievable schedule, not
independent scores to minimize.

## Start From The Limiting Resource

Record per workgroup and per wave:

```text
VGPR and AGPR allocation
scratch or private-memory traffic
LDS bytes, alignment, and simultaneously live buffers
waves per workgroup and requested waves per execution unit
hardware-limited resident waves and workgroups
```

Separate payload registers, future-address or fragment registers, operand
registers, accumulators, and epilogue state. This reveals whether a proposed
prefetch, wider tile, or second accumulator bank actually crosses an occupancy
boundary.

Higher nominal occupancy is not automatically faster. A lower-occupancy plan
can win when it removes padded work, increases useful MFMA reuse, or exposes a
better instruction pipeline. Conversely, saving a few registers has no value
if the allocation stays within the same residency bucket and adds more memory
traffic or shorter wait runway. Use the final allocation and measured timing.

Treat barriers as semantic until the buffer-ownership proof removes them. A
barrier that appears redundant in one trace may protect reuse across another
wave, stage, or loop trip.

## Reason About Reuse At Its Real Scope

Classify each operand's reuse:

- **wave/register reuse**: one loaded fragment feeds several MFMAs;
- **workgroup/LDS reuse**: several waves consume one staged tile;
- **CU-local cache reuse**: nearby workgroups on one CU revisit lines;
- **XCD/shared-cache reuse**: mapped workgroups share a larger cache domain;
- **HBM streaming**: scheduling distance or interference destroys reuse.

Estimate the number of consumers, bytes retained, and reuse distance. Do not
call an operand reusable solely because mathematical GEMM reuses it; the launch
mapping must schedule those consumers close enough on the relevant hardware
scope.

For a strongly shared operand, first test inexpensive locality mechanisms:
tile traversal, grouped PID mapping, XCD-aware distribution, persistence, and
per-operand cache policy. Loading several output tiles in one program to reuse
an operand explicitly also retains multiple accumulator tiles and often causes
a large VGPR increase. Promote it only if the cache and mapping alternatives do
not preserve enough reuse and the final occupancy remains competitive.

## Tune Cache Policy Per Operand

A and B can have different streaming and reuse behavior. Use a four-way
ablation when testing a global-load cache modifier:

```text
default/default
bypass/default
default/bypass
bypass/bypass
```

Rotate execution order and run under the production cache regime. A policy may
look beneficial in a cold scanner and regress a warm reused workload, or vice
versa. Keep output and workspace treatment equivalent.

Interpret a bypass win through a concrete mechanism: reduced GL1 pollution for
a streaming operand, better retention for the reused operand, or less harmful
cross-workgroup interference. Do not turn a single symmetric result into a
global default. Revalidate after changes to tile orientation, mapping,
persistence, or K partitioning because they change reuse scope.

## Account For Concurrent LDS Footprint

LDS cost is the sum of simultaneously owned backing, not the sum of logical
tile extents. Include double or triple buffers, padded physical backing,
direct-to-LDS destinations, layout-conversion scratch, and LocalSplitU partial
accumulators. Check alignment and allocator rounding.

Larger BlockK can reduce barrier density while increasing every stage's LDS
footprint. More stages can hide memory latency while multiplying that footprint.
Source unrolling can increase live address or payload state without changing
LDS bytes. Measure each lever independently.

Direct-to-LDS removes payload VGPRs only when the intended instruction survives
lowering. It can still increase address setup, constrain ordering, or require
wider backing. Mixed policies are valid: stage one operand directly and keep
the other in VGPRs when their legality or scheduling needs differ.

## Map Work With Cache Domains In Mind

PID grouping and XCD mapping affect both device utilization and cache locality.
For each mapping, calculate:

- consecutive workgroups sharing A or B;
- reuse distance in workgroups and approximate bytes;
- full and partial CU rounds;
- balance across XCDs and compute units; and
- whether persistent or Stream-K ownership changes the sequence.

A locality-friendly traversal can lose if it creates an imbalanced final wave.
An evenly distributed mapping can lose if it destroys a large shared operand's
reuse. Keep mapping in the compile-time plan and promotion evidence rather than
assuming one architecture-wide order.

For Stream-K, include partial-tile ownership, reduction workspace, and the
cache behavior of multiple contributors. For LocalSplitU, include LDS reduction
traffic and the changed per-wave accumulator lifetime. The best mapping for a
data-parallel tile need not be the best for either K-parallel form.

## Apply A Resource Gate Before Timing

Reject a candidate before expensive benchmarking when the final artifact:

- spills or introduces material scratch traffic;
- loses a resident-wave or resident-workgroup tier without a mechanism likely
  to compensate;
- allocates unexpected LDS scratch through a layout conversion;
- loses the intended direct-to-LDS or vector load form;
- retains multiple future or accumulator banks outside the designed lifetime;
  or
- changes the launch mapping so the locality hypothesis is no longer tested.

Passing the gate is not a performance result. Benchmark survivors and retain
the resource record with the measurements so later dispatch or compiler changes
can detect why a formerly good plan moved regimes.
