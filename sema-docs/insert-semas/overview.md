# InsertSemas overview

[← Back to README](../README.md)

## Contents

- [Contract](#contract)
- [Running example](#running-example)
- [One model, three steps](#one-model-three-steps)
- [Core objects](#core-objects)
- [Source and use state](#source-and-use-state)
- [Point-of-use placement](#point-of-use-placement)
- [Mutation boundary](#mutation-boundary)
- [Step documents](#step-documents)
- [Code map](#code-map)

## Contract

`nvws-insert-semas` turns cross-partition accesses to mutable SMEM or TMEM
allocations into explicit semaphore synchronization. Each analyzed allocation
has a logical group id: an authored `buffer.id` when present, otherwise a
private synthetic id. Every access is assigned an `Owner` from its partition
metadata.

The pass must preserve three things at once:

1. memory versions: a read observes the intended write, and a later write does
   not overwrite a value while another partition may still use it;
2. physical storage: overlapping allocations and buffer copies refer to the
   correct backing slot; and
3. scheduled execution: inserted operations respect the Triton pipeliner's
   authored `loop.stage` placements and the ordering constraints required by
   the new synchronization edges.

The implementation separates those concerns into three models:

  - `ACCESS-DAG` describes memory and ownership.
  - `SYNC-DAG` derives the required synchronization edges and builds a
    complete, verified synchronization plan.
  - `EMIT-IR` materializes that plan without changing it.

This distinction is the central design rule:

> SYNC-DAG chooses every acquire, release, token producer, region result,
> semaphore, schedule, and copy offset. EMIT-IR only renders those
> choices.

Before building those models, the shared managed-allocation locality
validator requires every complete buffer group and its recognized
memdesc/token use closure to reside in one top-level function CFG block. It
rejects cross-block groups, aliases, accesses, and CFG block-argument
forwarding. InsertSemas then performs group discovery and ACCESS-DAG
construction independently in each validated function block. The resulting
block-local DAGs are accumulated into one function-level synchronization plan
and one function-atomic EMIT-IR mutation.

[↑ Back to contents](#contents)

## Running example

The documents build one example in stages. The input contains one local-memory
object, one writer partition, and one reader partition:

The excerpt is IR-shaped text; type details are abbreviated, but the
warp-specialization attributes that establish owners are shown:

```text
%m0 = ttg.local_alloc {buffer.id = 104} : !memdesc
scf.for ... {
  ttg.local_store %value, %m0 {ttg.partition = array<i32: 0>}
  %v = ttg.local_load %m0 {ttg.partition = array<i32: 1>}
  "use"(%v) {ttg.partition = array<i32: 1>}
} {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
   ttg.warp_specialize.tag = 0 : i32}
```

The notation used in the diagrams is deliberately shorter:

```text
for {
  W m0 {0}
  R m0 {1}
}
```

ACCESS-DAG discovers that `m0` is one group member, its whole footprint is one
piece `P0`, and the loop first touches that piece as owner `{0}`:

```text
func root
`- for pieces{P0:W:{0}}
   |- ENTER pieces{P0:W:{0}}
   |- W m0 {0}
   |- R m0 {1}
   `- EXIT pieces{P0:W:{0}}
```

SYNC-DAG then derives two synchronization edges with stable names:

```text
e1: W m0 {0} -> R m0 {1}       current value becomes readable
e2: R m0 {1} -> EXIT {0}        piece P0; next iteration may overwrite it
```

After reduction, both remain. Point-of-use placement maps them to two
semaphores:

```text
EMPTY:  count=1, initially released, owner {0}
FULL:   count=1, initially blocked

for {
  a EMPTY {0}
  W m0 {0}
  r FULL  {0}
  a FULL  {1}
  R m0 {1}
  r EMPTY {1}
}
```

The next iteration's `a EMPTY` closes `e2`. The first iteration succeeds
because `EMPTY` starts released. A zero-trip loop performs none of the body
synchronization. Later sections derive every line rather than assuming this
diagram.

[↑ Back to contents](#contents)

## One model, three steps

```text
input IR
  |
  v
ACCESS-DAG
  collect groups, members, pieces, accesses, owners, and region boundaries
  |
  v
SYNC-DAG
  derive and reduce synchronization edges
  choose backing copies
  place point-of-use acquires and releases, carrying tokens when needed
  form semaphores and validate the complete plan
  finalize schedules and copy offsets
  |
  v
EMIT-IR
  materialize backing, semaphores, tokens, views, and region signatures
  verify the emitted IR
  |
  v
output IR
```

Before InsertSemas, upstream allocation materialization must ensure that a
synchronized `ttg.local_alloc` is not still sourced by TT-form
`tt.descriptor_load` or `tt.descriptor_gather`. The NVWS allocation path
performs that conversion in `nvws-insert-allocas` or IR conversion bridge.
InsertSemas reports a hard
pass-order invariant failure if such a source remains in a group that requires
semaphores.

ACCESS-DAG assigns owners and creates `ENTER`/`EXIT` boundaries while it
recursively builds each region; ownership is not a separate pass step.

SYNC-DAG constructs and verifies one point-of-use plan before EMIT-IR starts.

[↑ Back to contents](#contents)

## Core objects

- **GroupDag**: allocations and views analyzed together because they reuse the
  same backing. An authored `buffer.id` selects the group when present; an
  allocation without one receives a private synthetic id. Circular local
  rules may split one physical `buffer.id` into several logical groups.
- **Member**: one allocation in the group. Members may overlap in physical
  address space.
- **PieceId**: one disjoint interval induced by all member endpoints. A
  member's `footprint` is the list of pieces it covers.
- **Owner**: `(ttg.partition, warp-specialization tag)`. An empty `Owner`,
  written `root`, means owner resolution did not find exactly one partition
  id and a tag: partition metadata may be absent or contain multiple ids, or
  no tag may be reachable. It does not necessarily mean the operation is
  lexically outside warp-specialized code.
- **Node**: one access, region, boundary marker, acquire, or release.
- **Chain**: the ordered nodes in a function, loop body, or `if` branch.
- **PieceInfo**: the owner and aggregate `R`/`W` effect of one piece at a node
  or region boundary.
- **RegionFlow**: the token carried through a `for` or `if`: one owner and one
  exact producer per exit path. A null producer means that path returns the
  input token unchanged. Semaphore formation records which semaphore the
  region result uses after acquire and release placement is final.
- **Sema**: one logical semaphore, its pending count, optional entry
  owner, and eventually its emitted SSA value.

Several fields form the contract between planning and emission:

- `slotEffect` seals the aggregate memory effect of an access or region
  summary; `ENTER` and `EXIT` carry `pieceInfo` instead;
- `tokenSource` names the exact acquire or region producer consumed by an
  access or release;
- `producedTokenOwner` states the effective owner of a token-producing node;
- `sat` pairs a release with the acquire it supplies;
- `scheduleAnchor` identifies the source or completion anchor that determines
  a release's placement and schedule;
- `recurrenceDistance` records a cross-iteration dependency explicitly; and
- `stageOffset` and `bufferStageOffset` select semaphore and backing copies.

Because token identity is exact, several same-owner tokens may remain live at
once. `tokenSource` distinguishes them, and EMIT-IR resolves that specific
producer.

[↑ Back to contents](#contents)

## Source and use state

SYNC-DAG derives synchronization edges per piece. For each piece it tracks:

- `source`: the latest write to the piece; and
- `uses`: the latest read or write of the piece by each owner since that
  source.

When a child region receives a piece from its parent, the source remains the
outside write. Inside the child, `ENTER` marks where those contents become
available so the child can derive its edges without moving the write into the
child.

After a structured region, a new owner receives the current token from the
region summary, not directly from an older write. This keeps a following
handoff after the region and prevents it from bypassing work performed inside
the region.

Readers of one stable version may fan out. A later writer must wait for every
latest reader that is not already ordered before it. The full rules, including
region composition and edge reduction, are derived in
[SYNC-DAG](sync-dag.md).

[↑ Back to contents](#contents)

## Point-of-use placement

The pass builds one point-of-use (POU) plan. It reuses a valid token when the
current owner already has one; otherwise it acquires a token at the buffer use
that needs it. When ownership changes, the previous owner releases its token
and the new owner acquires the semaphore before using the buffer.

One current exception is not at a buffer use: an acquire can consume a
semaphore's initially released state without using its token for a buffer
access. This makes a later acquire wait for a real release. The nested
parent-continuation example in SYNC-DAG shows this case and a simpler carried-
token form left as a future investigation.

A loop can pass an exact token in an `iter_arg` and return it as a loop result.
Zero iterations return the input token. Nonzero iterations return the final
token produced in the body. The planner carries a token this way when:

- the exact incoming token must remain available after the loop;
- the token passed to the next iteration cannot be represented by an acquire
  at that buffer use.

The loop also forwards a valid token already returned by its body instead of
adding a same-owner release and acquire. All of these choices are part of the
same POU plan.

An `if` with one boundary owner gives the same incoming token to both
branches. When the token is needed afterward, each branch performs its own
ownership changes and returns a token for that boundary owner. An unchanged
branch returns the input token. A later ownership change uses the `if` result,
so its release is placed after the `if`, not before it. When a branch returns
the input token, its unfinished async completion is retained for a later
cross-partition release.

[↑ Back to contents](#contents)

## Mutation boundary

Group collection, ACCESS-DAG construction, and per-group SYNC-DAG construction
build the model without emitting semaphore IR. Schedule finalization may
adjust `loop.cluster` attributes on existing operations. The driver saves the
original attributes and restores them if planning or schedule finalization
fails.

EMIT-IR starts only after the single plan has placed every synchronization
edge, passed token-connectivity and structural checks, and finalized the
schedule. Its ordinary rendering rewrites SSA signatures and memory objects
without choosing another token or changing the SYNC-DAG plan. Acquires and
releases placed inside an `if` branch remain in that branch; EMIT-IR does not
split the conditional or move them across its boundary.

[↑ Back to contents](#contents)

## Step documents

1. [ACCESS-DAG: accesses, owners, and boundaries](access-dag.md)
2. [SYNC-DAG: edges, placement, semaphores, and schedule](sync-dag.md)
3. [EMIT-IR: materializing the sealed plan](emit-ir.md)

[↑ Back to contents](#contents)

## Code map

- Shared model: `InsertSemas.h`
- Groups, pieces, accesses, owners, and boundaries:
  `InsertSemasAccessDag.cpp`
- Synchronization edges, direct placement, semaphore formation, verification,
  and scheduling: `InsertSemasSyncDag.cpp`
- Symbolic dump and IR materialization: `InsertSemasEmitIR.cpp`
- Pass driver and the single build, schedule, and emit sequence:
  `InsertSemas.cpp`
- Function-CFG locality validator and public contract:
  `MetaToNVWSConvert.cpp` / `MetaToNVWSConvert.h`

[↑ Back to contents](#contents)
