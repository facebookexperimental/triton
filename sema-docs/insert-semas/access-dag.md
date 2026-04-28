# ACCESS-DAG: accesses, owners, and boundaries

[← Back to README](../README.md)

## Contents

- [Purpose](#purpose)
- [The running example](#the-running-example)
- [What is analyzed](#what-is-analyzed)
- [Groups and members](#groups-and-members)
- [Pieces](#pieces)
  - [Pieces must form one component](#pieces-must-form-one-component)
- [Recognizing accesses](#recognizing-accesses)
- [Owners](#owners)
- [Ordered chains](#ordered-chains)
- [Effects](#effects)
- [When an access finishes](#when-an-access-finishes)
  - [Why `tt.descriptor_store` needs special handling](#why-ttdescriptor_store-needs-special-handling)
  - [Required `nvws.descriptor_store` counterpart](#required-nvwsdescriptor_store-counterpart)
- [Regions and boundaries](#regions-and-boundaries)
  - [Choosing a boundary owner](#choosing-a-boundary-owner)
  - [A loop can have a different owner per piece](#a-loop-can-have-a-different-owner-per-piece)
  - [An `if` prefers the preceding owner](#an-if-prefers-the-preceding-owner)
- [Construction algorithm](#construction-algorithm)
- [Output contract](#output-contract)
- [Code map](#code-map)

## Purpose

ACCESS-DAG answers the questions that must be settled before synchronization
is derived:

1. Which allocations are one synchronization group?
2. Which disjoint memory pieces does each allocation cover?
3. Which operations read or write those pieces?
4. Which partition owns each access?
5. What owner and memory effect does each `for` or `if` present at its
   boundary?

This pass does not insert semaphore nodes. It builds the region tree and
ordered access chains that SYNC-DAG consumes.

The diagrams below are schematic. The implementation dumps only the completed
SYNC-DAG, not a standalone ACCESS-DAG.

[↑ Back to contents](#contents)

## The running example

Start with the loop introduced in the overview:

The excerpt is IR-shaped text with abbreviated types. The loop attributes are
included because they establish the `{0}` and `{1}` owners:

```text
%m0 = ttg.local_alloc {buffer.id = 104} : !memdesc
scf.for ... {
  ttg.local_store %value, %m0 {ttg.partition = array<i32: 0>}
  %v = ttg.local_load %m0 {ttg.partition = array<i32: 1>}
  "use"(%v) {ttg.partition = array<i32: 1>}
} {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
   ttg.warp_specialize.tag = 0 : i32}
```

ACCESS-DAG first gives the allocation a member name and a piece footprint:

```text
group 104
  m0 = local_alloc [0, 1)

piece       interval    covered by
P0          [0, 1)      m0

member      footprint
m0          {P0}
```

It recognizes the store as `W m0 {0}` and the load as `R m0 {1}`. The loop's
boundary owner for `P0` is its first toucher, `{0}`:

```text
func root
`- for pieces{P0:W:{0}}
   |- ENTER pieces{P0:W:{0}}
   |- W m0 pieces{P0} {0}
   |- R m0 pieces{P0} {1}
   `- EXIT pieces{P0:W:{0}}
```

The `W` effect at the boundary means that some path through the loop writes
`P0`; it does not mean the boundary itself performs a write. SYNC-DAG will use
this tree to derive the two edges shown in the overview.

[↑ Back to contents](#contents)

## What is analyzed

The pass recognizes mutable:

- `ttng.tmem_alloc`; and
- `ttg.local_alloc`.

It ignores immutable local allocations. Allocations with the same memory kind
and `buffer.id` normally form one `GroupDag`. An allocation without a
`buffer.id` receives a private synthetic group id, so unrelated anonymous
allocations are never merged accidentally.

The implementation recursively analyzes the first block of each structured
`scf.for` or `scf.if` region. At the function CFG level, the managed-allocation
locality validator first proves that every complete group and its recognized
memdesc/token use closure belong to one top-level function block. It rejects
groups spanning blocks, cross-block uses and aliases, and managed storage
forwarded through `cf.br`/`cf.cond_br` block arguments.

After validation, InsertSemas visits each top-level function block
independently for group discovery and ACCESS-DAG construction.
`collectGroups(func, &functionBlock)` sees only allocations nested under that
block, and `buildAccessDag` walks that explicit block. An early-return CFG is
therefore supported when each complete managed lifetime is contained in one
of its blocks; ACCESS-DAG never silently falls back to the function entry
block or attempts general CFG dataflow.

A synchronized `ttg.local_alloc` must not retain a source produced by
TT-form `tt.descriptor_load` or `tt.descriptor_gather`. Upstream allocation
materialization must convert that form before InsertSemas; the NVWS allocation
path does so in `nvws-insert-allocas`. If the group needs semaphores, SYNC-DAG
diagnoses a remaining TT-form source as a pass-order invariant violation.

[↑ Back to contents](#contents)

## Groups and members

A group is one access-analysis unit and, when ownership changes, one
synchronization unit. Each allocation in it is a `Member`:

```text
Member
  allocOp       original allocation
  type          authored memdesc type
  offset        buffer.offset, or zero
  extent        TMEM column count or SMEM leading shape extent
  copies        buffer.copy, or one
  circularStart buffer.start, or zero
```

For a circular local allocation, `buffer.start` is its initial position in the
shared ring. For an ordinary local allocation, `buffer.start` can instead name
the allocation's logical slot within storage reused from another allocation.
Only `buffer.circular` makes an allocation a circular group.

Grouping has three relevant rules:

- A circular local allocation must have `buffer.id`, `buffer.copy`, and
  `buffer.start`, and must not have `buffer.offset`. Each circular allocation
  gets its own logical group even when another circular group has the same
  physical `buffer.id`; physical sharing is validated later.
- TMEM allocations with one `buffer.id` must agree on every explicit
  `buffer.copy`; conflicting depths are rejected rather than analyzed as
  independent synchronization plans over aliased storage.
- Ordinary local allocations with one `buffer.id` remain one group. Their
  explicit `buffer.copy` values can differ when the memory plan describes
  several logical slots within storage reused from another allocation.
  ACCESS-DAG records each member's copy count and start; SYNC-DAG later checks
  whether that reuse is valid.

Keeping logical groups separate where necessary is important: synchronization
is derived per group. SYNC-DAG backing selection can map compatible members of
one ordinary group, or several circular groups, onto storage supplied by one
allocation.

[↑ Back to contents](#contents)

## Pieces

Members in one group may overlap. Synchronizing each whole member would either
miss an overlap or serialize disjoint storage, so ACCESS-DAG partitions their
address span into disjoint conceptual pieces.

For example:

```text
m0 = [0, 256)
m1 = [64, 192)

address: 0          64                    192         256
         |----------|---------------------|-----------|
piece:   P0         P1                    P2
cover:   {m0}       {m0,m1}               {m0}

footprint(m0) = {P0, P1, P2}
footprint(m1) = {P1}
```

The implementation collects every member endpoint, orders the intervals, and
assigns a `PieceId` when the covering-member set changes. It persists only
each member's `PieceId` footprint; the numeric intervals are construction
facts.

An access to `m1` touches `P1`. An access to `m0` touches all three pieces:

```text
W m1  => W P1
R m0  => R P0, R P1, R P2
```

SYNC-DAG tracks memory versions independently for `P0`, `P1`, and `P2`.

<a id="pieces-must-connect"></a>

[↑ Back to contents](#contents)

### Pieces must form one component

One `buffer.id` group must describe one connected storage component.
ACCESS-DAG rejects either:

- an uncovered gap between ordered member endpoints; or
- adjacent coverage sets that share no member.

Valid overlap:

```text
m0 [0, 128) --------+
                    +--- shared member across the boundary
m1   [64, 192) -----+
```

Rejected disjoint members:

```text
m0 [0, 64)          m1 [128, 192)
   no member connects the two components
```

The rest of InsertSemas relies on “one group equals one connected
synchronization unit,” so rejecting a disconnected group is safer than
silently constructing two unrelated synchronization plans under one id.

[↑ Back to contents](#contents)

## Recognizing accesses

For each group, ACCESS-DAG maintains a map from memdesc SSA values to:

```text
(member id, alias path from the original allocation)
```

The original allocation result starts with an empty alias path. A supported
alias operation extends the path for its result. The recognized alias names
are:

- `ttg.memdesc_index`
- `ttg.memdesc_subview`
- `ttg.memdesc_subslice`
- `ttg.memdesc_trans`
- `ttg.memdesc_reinterpret`
- `ttg.memdesc_reshape`

Each `AliasStep` stores the operation, the aliased operand number, and the
authored result type. EMIT-IR later replays that path on an acquired semaphore
buffer.

Known memory operations receive explicit effects:

| Operation | Group operand | Effect |
| --- | --- | --- |
| `ttng.tmem_load` | source | `R` |
| `ttg.local_load` | source | `R` |
| `ttng.tmem_store` | destination | `W` |
| `ttg.local_store` | destination | `W` |
| `nvws.descriptor_load` / `nvws.descriptor_gather` | destination | `W` |
| Sourceful `ttng.tmem_alloc` / `ttg.local_alloc` | result | `W` |
| MMA accumulator | accumulator operand | `W` |
| Other group memdesc operands | operand | `R` |

For an otherwise unknown operation, an alias operand already present in this
group's value map is treated as a write. This conservative default prevents an
unrecognized mutation from being misclassified as a read.

Structured `scf.for`/`scf.if` operations, function return, and `scf.yield` may
not carry a memdesc derived from a group allocation directly; discovered direct
uses are rejected.
General `cf.br`/`cf.cond_br` block-argument transport is also unsupported but,
as described above, is not comprehensively discovered or rejected.

[↑ Back to contents](#contents)

## Owners

An access owner is resolved from its partition metadata:

```text
Owner = (ttg.partition, warp-specialization tag)
```

An operation resolves to a non-root owner only when it names exactly one
partition and has a warp-specialization tag on itself or a reachable enclosing
warp-specialized loop. No partition, several partitions, or no such tag
produces the empty owner, written `root` in the documents. Owners are attached
to access nodes immediately; ownership is not assigned later.

An access may touch several members in the same group. Its `Touch` records,
for each member:

- member id;
- `R` or `W` effect;
- the exact memdesc value used by the operation;
- the authored access type; and
- the alias path.

[↑ Back to contents](#contents)

## Ordered chains

Each function body, loop body, and `if` branch becomes a doubly linked chain.
Only memory-relevant nodes appear:

```text
Access <-> Access <-> For <-> Access <-> If <-> Access
```

Ordinary arithmetic remains in the IR but is absent from the ACCESS-DAG.
Structured regions are nested nodes, not flattened into their parent:

```text
parent chain
  A -> [For] -> B
        |
        `- child chain: ENTER -> C -> D -> EXIT
```

This separation is essential. Parent analysis sees a summarized region event;
child analysis sees only the version supplied through its own `ENTER` and the
version returned through its own `EXIT`.

[↑ Back to contents](#contents)

## Effects

`Effect` has two values, `R` and `W`, with `W` dominating:

```text
join(R, R) = R
join(R, W) = W
join(W, R) = W
join(W, W) = W
```

An access node's `slotEffect` is the join of its touches. A region's effect for
one piece is the join across all of its branches. This is deliberately a may
summary: if either branch writes a piece, the region boundary says `W`.

The summary also records a boundary owner for each piece. Effect and owner are
separate facts:

```text
pieces{P0:W:{0}, P1:R:{2}}
       ^  ^  ^
       |  |  boundary owner
       |  aggregate effect
       piece
```

[↑ Back to contents](#contents)

## When an access finishes

ACCESS-DAG records every read and write as a separate access node and continues
scanning after each node. This section only determines where one
`ttg.local_load` is considered complete for release placement and scheduling.
The selected anchor does not by itself represent completion of asynchronous
work.

A `ttg.local_load` normally stops using its SMEM buffer at the load operation.
One pattern is different: the load can feed exactly one `tt.descriptor_store`,
directly or through one `ttg.convert_layout`. Later lowering can make the
descriptor store read directly from the SMEM buffer, so the buffer must remain
protected through that store. The access node records the descriptor store as
its `completionAnchor`, and SYNC-DAG uses the store instead of the load for
release placement and scheduling.

[↑ Back to contents](#contents)

### Why `tt.descriptor_store` needs special handling

At the input to InsertSemas, `tt.descriptor_store` accepts a tensor rather than
an SMEM buffer:

```mlir
%loaded = ttg.local_load %buf
%value = ttg.convert_layout %loaded  // optional
tt.descriptor_store %desc, ..., %value
```

The buffer operand appears only on `ttg.local_load`, so ordinary access
discovery records the read there. The optional conversion can later be folded
into a load that produces the required layout directly. TMA lowering can then
reuse `%buf` and replace the load-to-store path with an asynchronous store from
SMEM:

```mlir
ttng.async_tma_copy_local_to_global %desc, ..., %buf
```

The buffer dependency is therefore implicit in the tensor passed from the
local load, optionally through one conversion, to the descriptor store.
Recording the descriptor store as `completionAnchor` keeps `%buf` protected
through the operation that consumes it.

`deriveCompletionAnchor` recognizes only those two exact paths and requires:

- the load result and, when present, conversion result each have exactly one
  user;
- load, optional conversion, and store are in the same block;
- the store follows the load; and
- the store has the same owner as the load.

For a store found by that search, fan-out, a cross-block path, reverse order,
or an owner change is rejected. A descriptor store reached through a more
indirect path is not discovered or rejected; the access retains no explicit
completion anchor, so release scheduling uses the load itself. Such an
indirect path is outside the supported input.

[↑ Back to contents](#contents)

### Required `nvws.descriptor_store` counterpart

NVWS does not currently have a descriptor-store counterpart to
`nvws.descriptor_load` and `nvws.descriptor_gather`. A buffer-taking operation
would make the SMEM dependency explicit. Allocation materialization could
first produce:

```mlir
%loaded = ttg.local_load %buf
%value = ttg.convert_layout %loaded
ttg.local_store %value, %buf
nvws.descriptor_store %desc, ..., %buf
```

ACCESS-DAG could record the local store as a write to `%buf` and
`nvws.descriptor_store` as the following read. When the load-convert-store
round trip preserves the SMEM representation and its intermediate values have
no other users, it could be folded to:

```mlir
nvws.descriptor_store %desc, ..., %buf
```

The operation must also expose when the asynchronous TMA store finishes reading
the SMEM buffer, so the semaphore arrival returning that buffer is not posted
early.

[↑ Back to contents](#contents)

## Regions and boundaries

ACCESS-DAG constructs owners, summaries, and boundary nodes in one recursive
walk.

For every path of each retained memory-relevant `for` or `if`, it creates:

```text
ENTER -> child accesses/regions -> EXIT
```

`ENTER` means “the version visible when this path starts.” `EXIT` means “the
version returned when this path ends.” They are structural records, not
existing MLIR operations.

[↑ Back to contents](#contents)

### Choosing a boundary owner

The boundary owner is selected per piece:

- `for`: the first owner in the body that touches the piece;
- `if`: the last preceding owner in the parent chain if one exists; otherwise
  the first owner touching the piece in then/else order;
- a warp-specialized `for` presents `root` to its enclosing parent chain, even
  though its child boundary retains the selected partition owner.

The `if` rule preserves an already-established parent version. The loop rule
chooses the owner that must receive the iteration's input before any later
body owner can use it.

[↑ Back to contents](#contents)

### A loop can have a different owner per piece

Consider two overlapping members:

```text
m0 footprint = {P0, P1}
m1 footprint = {P1, P2}

for {
  W m0 {0}        // first touches P0 and P1
  R m1 {1}        // first touches P2
}
```

The loop summary is:

```text
P0: W, owner {0}
P1: W, owner {0}
P2: R, owner {1}
```

The boundary is therefore per-piece, not one owner for the whole region:

```text
for pieces{P0:W:{0}, P1:W:{0}, P2:R:{1}}
|- ENTER pieces{P0:W:{0}, P1:W:{0}, P2:R:{1}}
|- W m0 {0}
|- R m1 {1}
`- EXIT pieces{P0:W:{0}, P1:W:{0}, P2:R:{1}}
```

[↑ Back to contents](#contents)

### An `if` prefers the preceding owner

```text
W m0 {2}
if %cond {
  R m0 {0}
} else {
  R m0 {1}
}
```

The version entering the `if` belongs to `{2}`, so both branch boundaries use
`{2}` even though neither branch's first access does:

```text
W m0 {2}
if pieces{P0:R:{2}}
|- then
|  |- ENTER pieces{P0:R:{2}}
|  |- R m0 {0}
|  `- EXIT pieces{P0:R:{2}}
`- else
   |- ENTER pieces{P0:R:{2}}
   |- R m0 {1}
   `- EXIT pieces{P0:R:{2}}
```

SYNC-DAG can now derive each branch from the same parent source and later join
the exact token supplied by each path.

[↑ Back to contents](#contents)

## Construction algorithm

The shared locality validator runs once for the function. The driver then
iterates its top-level blocks. For each block, `collectGroups` classifies the
allocations nested there, creates each group's members, and seeds each
allocation-result alias map. Then, for each block-local group:

1. `buildPieces` partitions the covered span and validates connectivity.
2. `buildChainForBlock` walks the explicit function block supplied by the
   driver in lexical order.
3. `collectTouches` either extends an alias path or recognizes memory effects.
4. An access node receives its owner, touches, `slotEffect`, and optional
   completion anchor.
5. A nested `for` or `if` is built recursively.
6. Child effects, first owners, and last owners are summarized per piece.
7. The region selects its boundary owner per piece and receives `ENTER`/`EXIT`
   nodes for every path.
8. The summarized region is appended as one event in the parent chain.

The transient `Chain` summary contains effects and first/last owners only while
construction is in progress. The persistent model remains `Node`-based.

[↑ Back to contents](#contents)

## Output contract

After ACCESS-DAG:

- every group is one connected piece component;
- every member has a `PieceId` footprint and its authored copy/start metadata;
- every recognized access has exact touches, effects, owner, and alias paths;
- every access/region has a sealed aggregate `slotEffect`;
- every retained memory-relevant `for` and `if` has child chains with explicit
  `ENTER` and `EXIT`;
- every touched region piece has an effect and boundary owner; and
- recognized direct or one-`convert_layout` descriptor-store completion is
  explicit.

No synchronization edge or semaphore has been chosen yet. The next document
starts from exactly these facts.

[↑ Back to contents](#contents)

## Code map

- Group formation: `collectGroups`
- Piece partition and connectivity: `buildPieces`
- Alias and access recognition: `collectTouches`
- Descriptor-store completion: `deriveCompletionAnchor`
- Function-CFG locality and validation:
  `validateNVWSManagedAllocationLocality` in `MetaToNVWSConvert`
- Effect/owner accumulation: `appendNode`
- Recursive regions and boundaries: `buildChainForBlock`
- Per-group entry point: `buildAccessDag`

[↑ Back to contents](#contents)
