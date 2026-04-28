# EMIT-IR

[← Back to README](../README.md)

## Contents

- [Contract: materialize a sealed plan](#contract-materialize-a-sealed-plan)
- [Running example: semaphore DAG to MLIR](#running-example-semaphore-dag-to-mlir)
- [Emission order](#emission-order)
- [Backing and semaphore creates](#backing-and-semaphore-creates)
  - [Ordinary groups](#ordinary-groups)
  - [Tokenless planned local backing](#tokenless-planned-local-backing)
  - [Circular local groups](#circular-local-groups)
  - [Entry state](#entry-state)
- [Exact token routing](#exact-token-routing)
  - [Exact fan-out and reuse](#exact-fan-out-and-reuse)
- [Buffer views and accesses](#buffer-views-and-accesses)
  - [Alias replay](#alias-replay)
  - [Access rewriting](#access-rewriting)
- [Synchronization-node mapping](#synchronization-node-mapping)
- [Tokens through regions](#tokens-through-regions)
  - [POU without a carried token](#pou-without-a-carried-token)
  - [POU with a carried token](#pou-with-a-carried-token)
  - [An `if` result](#an-if-result)
  - [Several groups threading tokens through one region](#several-groups-threading-tokens-through-one-region)
- [Schedule, offsets, and partition metadata](#schedule-offsets-and-partition-metadata)
- [Cleanup](#cleanup)
- [Emitted-IR verification](#emitted-ir-verification)
  - [Exact cached-view reuse](#exact-cached-view-reuse)
  - [Partition outputs](#partition-outputs)
  - [Token and view locality](#token-and-view-locality)
  - [Lifetime and loop slots](#lifetime-and-loop-slots)
- [Output contract](#output-contract)
- [Code map](#code-map)

## Contract: materialize a sealed plan

EMIT-IR receives finalized `GroupDag`s. A planned group has backing storage to
materialize. An active group also has semaphores and a synchronization chain
to render. SYNC-DAG has already fixed:

- group, member, and semaphore copy counts;
- which allocation supplies storage reused by each member;
- semaphores, entry state, and pending counts;
- exact acquire and release positions;
- every access and release `tokenSource`;
- every token producer's owner;
- every release's matching acquire and source/completion anchor;
- every `RegionFlow` and exact path result;
- partition requirements, schedules, recurrence distances, and copy offsets.

The emitter allocates physical objects, rewrites structured-control-flow
signatures, renders those nodes, and verifies the result. It does not choose
acquire placement, infer an owner token, revise the sealed plan, or move
synchronization across a structured-control-flow boundary.

The central invariant is:

> Every Access, Release, and region input names an exact `tokenSource`.
> EMIT-IR looks up that producer directly; owner and lexical order are not
> routing inputs.

[↑ Back to contents](#contents)

## Running example: semaphore DAG to MLIR

SYNC-DAG finished the running loop as:

```text
|- scf.for pieces{P0:W:{0}}
|  |- a EMPTY {0}
|  |- W m0 {0}
|  |- r FULL {0} [none]
|  |- a FULL {1}
|  |- R m0 {1}
|  |- r EMPTY {1} [none]

EMPTY: count=1, entry owner={0}
FULL:  count=1, initially blocked
BACKING: numCopies=1
```

The following snippets are schematic MLIR with long type signatures omitted.
The emitter first creates staged backing and the two semaphore objects beside
the original allocation. Rendering later redirects group-allocation uses, and
cleanup removes dead originals:

```text
%base = ttg.local_alloc
  : !ttg.memdesc<1x1xi32, ...>

%empty = nvws.semaphore.create %base released = -1
  {pending_count = 1}
%full = nvws.semaphore.create %base
  {pending_count = 1}
```

`released = -1` means every physical stage of the entry semaphore begins
released. Omitting `released` means every stage begins blocked. The body is
then rendered in the same order as the symbolic chain:

```text
scf.for ... {
  %tw = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 0>}
  %bw = nvws.semaphore.buffer %empty, %tw {ttg.partition = array<i32: 0>}
  ttg.local_store %value, %bw {ttg.partition = array<i32: 0>}
  nvws.semaphore.release %full, %tw [#nvws.async_op<none>]
    {arrive_count = 1, ttg.partition = array<i32: 0>}

  %tr = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>}
  %br = nvws.semaphore.buffer %full, %tr {ttg.partition = array<i32: 1>}
  %v = ttg.local_load %br {ttg.partition = array<i32: 1>}
  nvws.semaphore.release %empty, %tr [#nvws.async_op<none>]
    {arrive_count = 1, ttg.partition = array<i32: 1>}
}
```

Notice that the token may cross semaphore names:

```text
producer a EMPTY -> token %tw -> W m0 and r FULL
producer a FULL  -> token %tr -> R m0 and r EMPTY
```

The release destination is not the semaphore that produced the token. The
token is used to access the current buffer; the release posts arrivals to the
next semaphore selected by SYNC-DAG.

[↑ Back to contents](#contents)

## Emission order

`emitIR` performs these steps:

1. Create one function-level poison async token. It replaces uses of detached
   legacy token results and also serves as a temporary signature placeholder.
2. Select planned groups with backing storage and active groups with
   semaphores. A tokenless local reuse group is planned but not active.
3. Clear legacy TMEM dependency operands, replace their old token-result uses
   with poison, and repeatedly erase dead pre-existing loop/`if` token slots
   for active groups.
4. Materialize backing objects for every planned group and semaphore creates
   for active groups.
5. For planned groups without semaphores, replace the original local
   allocations with views of their planned backing storage.
6. Aggregate every group's requested `RegionFlow` slot and rewrite each
   affected `scf.for` or `scf.if` exactly once, outermost first.
7. Render each active group's finalized chain.
8. Repeatedly remove newly dead token slots.
9. Erase dead alias operations and original allocations.
10. Erase the poison token when it is unused.
11. Verify emitted SSA, partition, locality, and lifetime contracts.

There is no separate “entry acquire” pass. An entry acquire is an ordinary
`Acquire` node rendered at its exact chain position.

[↑ Back to contents](#contents)

## Backing and semaphore creates

[↑ Back to contents](#contents)

### Ordinary groups

Each member's emitted backing type reflects the group's copy count. For
ordinary local or TMEM memory, that copy dimension is added in front of the
authored shape. For planned SMEM reuse, each internal `Member` record retains
its authored `buffer.copy`. TMEM scale encodings retain their special shape
convention.

Consider the ACCESS-DAG overlap example:

```text
m0=[0,256), footprint={P0,P1,P2}
m1=[64,192), footprint={P1}
```

Pieces guide synchronization, but every semaphore create carries one backing
value/type per group member. One emitted allocation can supply another
member's storage when the memory kind, offsets, types, and authored planning
metadata permit it. TMEM containment uses `ttng.tmem_subslice` and, when
needed, `ttg.memdesc_reinterpret`.

For ordinary mixed-copy SMEM reuse, the allocation supplying storage has one
copy and enough space for a member's logical copies. When a member has more
than the group's one copy, EMIT-IR reinterprets that storage as:

```text
[member buffer.copy] + member authored shape
```

It selects the member's explicit `buffer.start` with `ttg.memdesc_index`, then
reinterprets that slot to the member's emitted backing type. For a copy-one
planned reuse member with `buffer.start=0`, a direct
`ttg.memdesc_reinterpret` is sufficient. The semaphore's backing bundle
contains the allocation supplying storage and all such member views.

The mapping from every member to its emitted allocation or view is fixed before
signature rewriting and chain rendering. There is no later “fold the backing
after emission” step.

[↑ Back to contents](#contents)

### Tokenless planned local backing

A compatible ordinary local group can require backing/view materialization
even when all accesses have one owner and no semaphore is needed. This applies
only when the group has at least two source-free allocations in one block and
every member can share one emitted allocation under SYNC-DAG's type, offset,
copy, and `buffer.start` checks.

EMIT-IR materializes the shared storage and member views, replaces each
original allocation with a `ttg.memdesc_index` selecting index zero from its
emitted backing, and erases the original allocations. It emits no semaphore,
acquire, release, or async-token plumbing. A separate single allocation with
`buffer.copy=2` is not changed by this path.

`test/NVWS/insert_semas_local_mixed_copy_reuse.mlir` checks both mixed-copy
SMEM views and tokenless copy-one reuse.

[↑ Back to contents](#contents)

### Circular local groups

Circular groups retain separate logical SYNC-DAGs but share one physical
backing by `buffer.id`. All members must agree on backing type and be defined
in one block. The group with `buffer.start=0` supplies the authored backing
identity.

Compatible circular semaphore creates are shared by `(buffer.id, entry-state)`.
Their pending counts must agree. Copy and stage offsets were already assigned
by SYNC-DAG.

[↑ Back to contents](#contents)

### Entry state

`Sema::entryOwner` supplies the legacy default released-stage mask, while an
explicit `Sema::releasedMask` computed by SYNC-DAG overrides that default:

```text
releasedMask present -> semaphore.create ... released = releasedMask
entryOwner present   -> semaphore.create ... released = -1
otherwise            -> semaphore.create ...
```

An explicit zero mask is printed as an omitted `released` clause. Creates for
entry semaphores are emitted before the others. The create's `pending_count` is
the uniform count already proved during semaphore assignment.

[↑ Back to contents](#contents)

## Exact token routing

`RenderState` stores records of this form:

```text
Token
  value       emitted async-token SSA value
  sema        emitted semaphore SSA value that produced the token
  ref
    producer  exact symbolic producer node
    sema      symbolic semaphore
    owner     effective owner
```

`recordToken` replaces only a record with the same producer. Tokens produced
by different nodes may coexist even when they have the same owner.

Before rendering an Access or Release, `renderChain` calls
`tokenForSource(node->tokenSource)`. Missing or owner-incompatible records are
errors; only the named producer can satisfy the consumer.

[↑ Back to contents](#contents)

### Exact fan-out and reuse

The fan-out example from SYNC-DAG has three tokens live over time:

```text
a EMPTY(2) {0} -> producer token T0
  T0 -> W m0 {0}
  T0 -> r F1 {0}
  T0 -> r F2 {0}
  T0 -> later R m0 {0}

a F1 {1} -> T1 -> R m0 {1} -> r EMPTY {1}
a F2 {2} -> T2 -> R m0 {2} -> r EMPTY {2}
```

When `{0}` rereads the buffer, `T1` and `T2` may have been emitted more
recently. The access still names producer `T0`, so the emitter selects `T0`
exactly. Owner order cannot change that choice.

[↑ Back to contents](#contents)

## Buffer views and accesses

An access needs both a token and a member view. `getView` emits:

```text
%m0_view, %m1_view, ... = nvws.semaphore.buffer %sema, %token
```

The result bundle includes a view for every group member. The current bundle
is reused only when all token, semaphore, owner, and copy facts match:

- symbolic producer;
- symbolic semaphore;
- token SSA value;
- semaphore SSA value;
- owner;
- `bufferStageOffset`; and
- a `sameViewType`-compatible requested member type.

`sameViewType` compares shape, element type, encoding, memory space, and
mutability. It deliberately does not make allocation shape part of the cache
identity.

An owner-only cache would be unsound because the same owner may hold tokens
from different producers.

[↑ Back to contents](#contents)

### Alias replay

For a `Touch`, the emitter selects its member view and replays the alias path
recorded by ACCESS-DAG. Each cloned alias replaces the recorded alias operand
with the current acquired view. Result types are re-inferred where possible so
staged allocation shapes propagate correctly.

A `memdesc_index` whose result is `sameViewType`-compatible with its source can
be elided. Other supported aliases are cloned in order:

```text
base semaphore view
  -> memdesc_index/subview/subslice/trans/reinterpret/reshape
  -> exact access view
```

[↑ Back to contents](#contents)

### Access rewriting

Known accesses are rewritten as follows:

- general operations have the exact `Touch::accessValue` operand replaced;
- a sourceful local allocation becomes an explicit `ttg.local_store` through
  the acquired view;
- a sourceful `ttng.tmem_alloc` becomes an explicit `ttng.tmem_store` through
  the acquired view; eligible uses of the allocation result are redirected to
  that view, excluding semaphore creates, the view-defining operation, and
  operations represented by access nodes;
- a scalar local source is splatted when the destination needs a tensor;
- local-allocation result uses unrelated to semaphore creates and access nodes
  are redirected to the new view; and
- the rendered access returns its real completion anchor, including an
  ACCESS-DAG-selected descriptor store.

The returned anchor becomes `lastReal`, which determines the exact insertion
point for a following release.

[↑ Back to contents](#contents)

## Synchronization-node mapping

| Symbolic node | Emitted action |
| --- | --- |
| `Acquire` | Insert before the next real node; otherwise before the containing region terminator, or after the last root-level real operation. Apply owner, schedule, and optional `stageOffset`; record the node as producer. |
| `Release` | Insert after the last rendered completion anchor, or at the containing block start when no real node precedes it; use its exact source token, destination semaphore, release-kind array, count, schedule, and optional `stageOffset`; mark the source released for lifetime auditing. |
| `Access` | Resolve exact source, materialize/reuse the exact buffer bundle, replay aliases, and rewrite the operation. |
| `ENTER` / `EXIT` | Emit nothing; the parent region renderer wires path inputs/results. |
| `For` / `If` | Render the child chain and fill the preallocated token slot only when a `RegionFlow` exists. |

`chainBlock` locates the exact child block for a synchronization-only or otherwise
empty chain. Thus a branch containing only a release still receives that
release in its own block.

[↑ Back to contents](#contents)

## Tokens through regions

Both loop forms use POU placement. The only difference here is whether the
loop signature must carry a token from one iteration to the next and return
that token after the loop.

[↑ Back to contents](#contents)

### POU without a carried token

The running example does not need a token result from the loop. It has no
`RegionFlow`, so its `scf.for` signature is unchanged. Body-local acquires and
releases render exactly where SYNC-DAG put them.

`renderPlainLoop` still creates a nested render state. If the body uses an
exact incoming token without replacing it, the emitter records that token
under the loop producer while rendering the body. The same incoming token
remains available after the loop. No iter-arg or loop result is added.

[↑ Back to contents](#contents)

### POU with a carried token

When the token after an iteration can differ from the token before that
iteration, SYNC-DAG seals a `RegionFlow` such as:

```text
exact incoming producer -> loop tokenSource
loop RegionFlow.owner = {0}
loop RegionFlow.exits[0] = exact body producer returned by an iteration
```

Signature rewriting first adds a poison placeholder:

```text
%result = scf.for ... iter_args(%carry = %poison) -> !ttg.async.token {
  ...
  scf.yield %poison
}
```

`renderCarriedLoop` replaces the init with the exact incoming token, records
the body iter-arg as the loop's token producer, renders the body, and replaces
the yield with the exact producer named by `RegionFlow.exits[0]`:

```text
%result = scf.for ... iter_args(%carry = %initial) -> !ttg.async.token {
  ...
  %next = nvws.semaphore.acquire %empty
  scf.yield %next
}
```

The loop result is recorded under:

- the loop node itself;
- the exact incoming producer alias when needed; and
- the exact exit producer alias.

Downstream `tokenSource` pointers can therefore resolve the result without an
owner-based search. The result is simple: a zero-trip loop returns `%initial`;
a nonzero loop returns the exact token yielded by its final iteration.
EMIT-IR does not decide whether the token should be carried. It renders the
sealed `RegionFlow` supplied by SYNC-DAG.

[↑ Back to contents](#contents)

### An `if` result

Suppose SYNC-DAG recorded:

```text
if thread{{4}}
  then ... EXIT yield{a Sback}
  else ... EXIT yield{pass}
```

Signature rewriting adds one result and one operand to each yield. During
rendering:

- the then path resolves the exact acquire named in `flow.exits[0]`;
- the else path sees `nullptr` and yields the exact incoming token; and
- the `scf.if` result becomes a new producer with owner `{4}`.

When a path returns the incoming token, SYNC-DAG also retains that token's
unfinished completion. A later cross-partition release using the `if` result
therefore emits the required release kind, such as `[tc5mma]`, rather than
turning it into `[none]`.

Schematic emitted IR:

```mlir
%out = scf.if %cond -> !ttg.async.token {
  ...
  %then_token = nvws.semaphore.acquire %Sback
  scf.yield %then_token
} else {
  scf.yield %incoming
}
```

Rendering normalizes an `if` represented in `RegionFlow` with no authored else
to an empty else. If `RegionFlow` needs a result, that branch supplies the exact
pass-through. The region renderer does not split the `if` or move
synchronization outside it.
An ownership change performed inside one branch is completed inside that
branch before its yield. If the first access after the `if` has another owner,
the release for that later handoff consumes the `if` result and therefore
appears after the conditional.

[↑ Back to contents](#contents)

### Several groups threading tokens through one region

`rewriteSignatures` collects all groups' requested slots before touching an
operation. The `for` or `if` is rebuilt once with all extra token types. Each
group remembers its absolute slot index and later fills only that slot.

Operations are rewritten outermost first, and every graph node pointing at an
old region is retargeted to the replacement. This avoids repeated nested
signature surgery.

[↑ Back to contents](#contents)

## Schedule, offsets, and partition metadata

Generated synchronization nodes receive the `stageCluster` already stored in
SYNC-DAG:

- acquire schedule from its placement anchor;
- release schedule from its exact source or completion anchor; and
- buffer view schedule from the access operation.

`stageOffset` is emitted as a signed `i32` operand on acquire/release and
selects a semaphore copy. `bufferStageOffset` is emitted on
`nvws.semaphore.buffer` and selects a backing copy. Neither alters `loop.stage`
or `loop.cluster`.

For the two-copy alias example:

```text
R m0 slot 0 -> release Shandoff with stageOffset=+1
W m1 slot 1 -> acquire Shandoff at its selected slot
```

The emitter transcribes `+1`; it does not replay the slot schedule.

For a structured operation that already carries partition metadata,
`requiredParts` extends its partition set when generated synchronization needs
additional owners inside it. Signature rewriting also extends
`ttg.partition.outputs` for every new token result and gives terminators the
region partition metadata when absent.

[↑ Back to contents](#contents)

## Cleanup

After rendering all active groups, EMIT-IR:

- erases supported alias operations whose results are dead;
- erases original group allocations that no longer have uses; and
- removes the temporary poison operation when no detached legacy use remains.

The initial dead-slot sweep may rebuild old `scf.for`/`scf.if` operations to
remove obsolete async-token results before new exact slots are added. This is
signature cleanup, not synchronization placement.

[↑ Back to contents](#contents)

## Emitted-IR verification

The final verifier checks the materialized contract.

[↑ Back to contents](#contents)

### Exact cached-view reuse

If a cached buffer view is intentionally reused after a release, the verifier
requires that:

- the view came from `nvws.semaphore.buffer` with the recorded exact token;
- a release of that token exists in the same block; and
- the reused view has a witnessed use after that release.

This exception is admitted only for an exact reuse chosen in SYNC-DAG.

[↑ Back to contents](#contents)

### Partition outputs

For every `scf.for` or `scf.if` carrying `ttg.partition.outputs`:

- the attribute arity must equal the operation's result count; and
- each yielded nonconstant producer with partition metadata must intersect the
  declared output partitions.

[↑ Back to contents](#contents)

### Token and view locality

`verifyTokenLocality` traces every release/buffer token backward through:

- direct `nvws.semaphore.acquire` results;
- `scf.if` then/else yields; and
- `scf.for` iter-args, yields, and init values.

For partition-marked consumers and acquires, the partition sets must be equal.
An acquire without partition metadata is outside that comparison. For a
partition-marked semaphore buffer, every partition-marked view user must have
the same set; unpartitioned users are outside this check.

[↑ Back to contents](#contents)

### Lifetime and loop slots

Within one block, an ordinary token may not create a new buffer view after it
has been released. A newly materialized exact-reuse buffer is recorded in
`exactReuseBufferOps` and exempted from that generic check. Reuse of an already
cached view records a `CachedReuseContract` and must pass the proof above.

For token slots that resolve directly, or through a bounded `scf.if` result
trace, to a semaphore acquire, a loop may not carry two slots for the create's
first physical backing. Downstream `AssignStagePhase` cannot represent
that state. The check excludes circular local backing, which explicitly uses
several semaphores.

Any failure here is a plan/emitter contract violation or malformed memory plan
and is reported as an error.

[↑ Back to contents](#contents)

## Output contract

After EMIT-IR succeeds:

- every consumer in a synchronized group uses a view derived from its exact
  planned token producer;
- every release uses that same exact token and the planned destination
  semaphore;
- every structured path supplies its planned region result or exact
  pass-through;
- schedules, counts, stage offsets, and backing offsets match SYNC-DAG;
- partition-marked token and view consumers agree with their acquire or
  materialization partition sets; and
- old group allocations and legacy token plumbing no longer define the active
  synchronization.

[↑ Back to contents](#contents)

## Code map

- Symbolic dump: `SyncDagDumper`, `dumpSyncDags`
- Shared emission state: `EmitCtx`, `RenderState`
- Legacy-token cleanup: `nukeGroupTokens`, `eraseDeadTokenSlots`
- Backing and creates: `materializeLogicalBacking`,
  `materializeTokenlessMembers`, `emitPhysicalIR`
- Aggregated region slots: `rewriteSignatures`
- Views and accesses: `getView`, `renderAccess`
- Structured control flow: `renderPlainLoop`, `renderCarriedLoop`,
  `renderRegion`
- Chain materialization: `renderChain`
- Locality proof: `verifyTokenLocality`
- Full emitted checks: `verifyEmittedIR`
- Entry point: `emitIR`

[↑ Back to contents](#contents)
