# SYNC-DAG

[← Back to README](../README.md)

## Contents

- [Notation](#notation)
- [Purpose](#purpose)
- [How the pass works](#how-the-pass-works)
- [From buffer accesses to synchronization edges](#from-buffer-accesses-to-synchronization-edges)
  - [What the pass remembers for each piece](#what-the-pass-remembers-for-each-piece)
  - [Read and write rules](#read-and-write-rules)
  - [Ensuring every access has a token](#ensuring-every-access-has-a-token)
  - [Example: one writer and one reader](#example-one-writer-and-one-reader)
  - [Example: several readers and token reuse](#example-several-readers-and-token-reuse)
  - [Example: different pieces synchronize separately](#example-different-pieces-synchronize-separately)
  - [Nested loops and branches](#nested-loops-and-branches)
  - [Example: nested loops with a count-2 semaphore](#example-nested-loops-with-a-count-2-semaphore)
  - [Example: the boundary owner is unchanged](#example-the-boundary-owner-is-unchanged)
- [Removing synchronization edges already guaranteed by other paths](#removing-synchronization-edges-already-guaranteed-by-other-paths)
  - [Synchronization edges in one path](#synchronization-edges-in-one-path)
  - [Synchronization edges between loop iterations](#synchronization-edges-between-loop-iterations)
  - [Async operations and release positions](#async-operations-and-release-positions)
  - [Example: one release provides two arrivals](#example-one-release-provides-two-arrivals)
  - [Example: an async writer keeps its direct edge](#example-an-async-writer-keeps-its-direct-edge)
  - [Example: one release after two reads](#example-one-release-after-two-reads)
  - [Example: a direct edge is unnecessary](#example-a-direct-edge-is-unnecessary)
  - [Example: a loop-exit edge is unnecessary](#example-a-loop-exit-edge-is-unnecessary)
- [Placing acquires and releases](#placing-acquires-and-releases)
  - [Straight-line code](#straight-line-code)
  - [`if` branches](#if-branches)
  - [Loops](#loops)
  - [Choosing whether a loop carries a token](#choosing-whether-a-loop-carries-a-token)
  - [Example: acquire at the buffer use](#example-acquire-at-the-buffer-use)
  - [Example: fixed stages with tokenless cross-stage POU](#example-fixed-stages-with-tokenless-cross-stage-pou)
  - [Example: an `if` returns the boundary token](#example-an-if-returns-the-boundary-token)
  - [Example: a buffer-use acquire is carried](#example-a-buffer-use-acquire-is-carried)
  - [Example: nested loops without carried tokens](#example-nested-loops-without-carried-tokens)
  - [Example: reading the buffer after the inner loop](#example-reading-the-buffer-after-the-inner-loop)
  - [Example: fixed stages in a nested POU loop](#example-fixed-stages-in-a-nested-pou-loop)
  - [Example: branch-local handback with fixed stages](#example-branch-local-handback-with-fixed-stages)
- [Assigning semaphores and counts](#assigning-semaphores-and-counts)
  - [Releases on the same path or different paths](#releases-on-the-same-path-or-different-paths)
  - [One pending count per semaphore](#one-pending-count-per-semaphore)
  - [Initial released-stage masks](#initial-released-stage-masks)
- [Buffer and semaphore copies](#buffer-and-semaphore-copies)
  - [Buffer copies](#buffer-copies)
  - [Example: a TMEM accumulator gets two copies](#example-a-tmem-accumulator-gets-two-copies)
  - [Semaphore copies](#semaphore-copies)
  - [Example: a TMA load uses the lowering stage count](#example-a-tma-load-uses-the-lowering-stage-count)
- [Scheduling synchronization in a pipelined loop](#scheduling-synchronization-in-a-pipelined-loop)
  - [Release before acquire](#release-before-acquire)
  - [Synchronization between iterations](#synchronization-between-iterations)
  - [Selecting the matching copy](#selecting-the-matching-copy)
  - [Example: circular K and V select different copies](#example-circular-k-and-v-select-different-copies)
  - [Example: a non-circular alias advances the copy](#example-a-non-circular-alias-advances-the-copy)
  - [Example: one buffer copy](#example-one-buffer-copy)
- [Checks before changing IR](#checks-before-changing-ir)
- [Build order and code map](#build-order-and-code-map)

## Notation

The document uses the following terms throughout:

```text
group          allocations and views with a shared buffer.id that reuse the same backing
backing        one or more physical SMEM or TMEM values used by the group
m0, m1         allocation members in the group, not their derived views
P0, P1         non-overlapping pieces of the group storage
{0}, {1}       owners: partitions 0 and 1 with the enclosing loop's WS tag
root           code with no partition owner
source         latest write to the piece
uses           latest access to the piece by each owner since that write
token          value returned by an acquire and used by releases and semaphore.buffer
```

Ordinary mutable local and TMEM allocations are bucketed separately by memory
kind and `buffer.id`. Each allocation without an ID receives a private
synthetic key. A circular local allocation is instead its own logical group,
even when several circular groups have the same physical `buffer.id`; emission
later folds those groups onto shared physical storage. Supported memdesc view
operations add aliases of an allocation member, not new `Member` entries. A
group can materialize several backing values, although compatible members can
reuse one value or a view of it.

An owner contains both a partition number and a WS tag. Two operations have
the same owner only when both values match. Most diagrams show one loop at a
time, so `{0}` abbreviates `(partition 0, the WS tag of that loop)`. Nested
examples name the outer and inner loops when their tags differ.

The examples use this short form:

```text
W m0 {0}              owner {0} writes buffer member m0
R m0 {1}              owner {1} reads buffer member m0
ENTER / EXIT           start and end used to describe one loop or branch path
[region P0:W:{1}]      the region writes piece P0 and its boundary owner is {1}
e1: A -> B             synchronization edge: B waits for A
a FULL(2) {2}          owner {2} acquires FULL and waits for 2 arrivals
r FULL(2) {3}          owner {3} releases FULL with arrive_count=2
[none]                 the semaphore receives an immediate arrival
[tma_load]             the semaphore tracks completion of a TMA load
[tc5mma]               the semaphore tracks completion of prior TCGen5 work
R m0 [t] {1}           the read uses the buffer selected by token t
[c2,s0]                loop.cluster=2 and loop.stage=0
```

`a` and `r` are short for acquire and release. The bracketed names are release
kinds. A release does not wait: `[none]` gives the semaphore an immediate
arrival, `[tma_load]` makes the semaphore track completion of a TMA load, and
`[tc5mma]` makes it track completion of prior TCGen5 work. The acquire waits
until the semaphore has received its `pending_count` arrivals and all tracked
work has completed.

The diagrams use three views:

```text
initial edge DAG    every synchronization edge found from buffer accesses
reduced edge DAG    edges left after other paths already guarantee their ordering
semaphore DAG       remaining edges realized as acquires, releases, and token uses
```

An arrow labeled `walk` shows program order, not synchronization. An arrow
labeled with a semaphore connects a release to an acquire. Paths from opposite
`if` branches are shown separately because only one branch runs.

The pass has one point-of-use (POU) planner. It normally places an acquire at
the first buffer use that needs a token. When the exact token must remain valid
across a loop, the planner instead passes that token into the loop and returns
the next token. Both forms are POU decisions from the same plan.

[↑ Back to contents](#contents)

## Purpose

SYNC-DAG finds the synchronization edges required when different partitions
use the same backing buffer. It realizes those edges with semaphores, acquires,
releases, and tokens. It also chooses backing-buffer and semaphore depth and
plans where every acquire and release is placed.

ACCESS-DAG has already found the buffer accesses and the loops and branches
that contain them. SYNC-DAG builds the final symbolic plan. EMIT-IR preserves
the chosen synchronization positions and exact token links.

[↑ Back to contents](#contents)

## How the pass works

```text
derive synchronization edges from the buffer accesses
  -> remove edges whose ordering other paths already guarantee
  -> choose physical buffer copies
  -> place acquires and releases and choose each access's token
  -> assign semaphores and counts
  -> choose semaphore copies and scheduled placement
  -> check the complete plan
  -> emit IR
```

SYNC-DAG decides all synchronization edges, semaphores, and tokens before
EMIT-IR starts.

Every worked example names its test under `test/NVWS` except
`@same_owner_nested`, `@doc_preserved_async_edge`, and
`@doc_repeated_same_owner_sources`. Those three inputs were run through the
pass but are not test cases. The DAG dump can be printed with:

```text
NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt input.mlir \
  -allow-unregistered-dialect --nvws-insert-semas
```

The dump names semaphores `S0`, `S1`, and so on. Examples use names such as
`EMPTY` and `FULL` to make their purpose easier to follow.

Each worked example ends with `Compact output IR`. This is readable IR-shaped
text, not literal MLIR: long types and unrelated attributes are omitted, but
loop and branch results, token arguments, acquires, releases, and buffer uses
match the pass output. Within each loop or branch scope, operations are grouped
into contiguous runs by partition, with owner tags in one column. A partition
can appear again after ownership returns to it. Nested loops and branches
remain explicit. Semaphore dependencies determine cross-partition execution
order; the visual order of partition blocks does not.

[↑ Back to contents](#contents)

## From buffer accesses to synchronization edges

The pass visits buffer accesses in program order. For each buffer piece it
remembers the latest write and the latest access by each owner. It adds an edge
whenever one owner must wait for another.

[↑ Back to contents](#contents)

### What the pass remembers for each piece

```text
source    latest write to the piece
uses      latest access to the piece by each owner since that write
```

A read updates only that owner's entry in `uses`. It does not replace the
source, so readers that can run separately still start from the same source.
A write becomes the new source and removes the earlier entries from `uses`.

There is one control-flow exception to where a new-reader edge starts. If the
source owner's latest use is a structured-region summary, the edge starts at
that summary instead of bypassing it to the older write. The contents still
come from the same latest write, but the new owner receives the token returned
by the region.

Inside a nested loop or branch, the source is still the latest write. The
child DAG uses `ENTER` as the local point where those contents enter the
child. A write inside the child becomes the new source.

[↑ Back to contents](#contents)

### Read and write rules

For one buffer piece `P`:

```text
first write of P
  remember this access as the source and first use
  add no edge

read P again by the same owner
  update that owner's latest use
  add no edge

read P by a new owner
  add source -> read
  remember the new reader

write P
  add latest use -> write for every other owner that is not already ordered
  remember the write as the new source and only use
```

A write does not need another edge when an existing path already makes the
other owner finish first. When a WS loop receives a buffer from code outside
a partition, its first partition does not need an extra edge from that code.

If an async operation wrote the buffer, a direct edge from that operation
also records the completion that the semaphore must track. The pass
keeps that edge when a different path would lose this completion requirement.

For a group with several members, a later non-async write by the same owner may
reuse the current token. In that case its later release also carries the
release kind for any unfinished async write associated with the same token.

The parent treats a nested loop or branch as one summary node. The child has
its own DAG from `ENTER`, through its buffer accesses, to `EXIT`. Parent and
child edges are never mixed.

[↑ Back to contents](#contents)

### Ensuring every access has a token

Once a group needs synchronization, every buffer access in that group needs a
token:

- One or more synchronization edges that remain and end at an access provide
  one acquire for the new owner.
- If the owner already has a token valid for every piece touched by the
  access, the access reuses that token.
- Otherwise, the pass adds an edge from the last place that holds a usable
  token, so the new owner can acquire one.

For a read, reuse is valid only when that owner already has the current
contents for every piece it reads. For a write, reuse is valid only when all
other owners have already finished using the old contents.

[↑ Back to contents](#contents)

### Example: one writer and one reader

`test/NVWS/insert_semas.mlir` `@local_loop_carried_and_result` contains two
groups. For the per-iteration group `buffer.id=104`, the relevant input is:

```text
for {
  W m0 {0}
  R m0 {1}
}
```

There is one piece, P0. The write sets its contents, and the read by another
owner creates `e1`. Because the loop may run again, owner `{0}` must wait for
the read before overwriting P0. Edge `e2` records that cross-iteration
ordering at `EXIT`:

```text
edge    source             destination       reason
e1      W m0(i) {0}       R m0(i) {1}       read after write
e2      R m0(i) {1}       EXIT(i) {0}        next iteration may overwrite P0
```

No edge is removed. The two edges have different destinations, so no edges
are merged or grouped together:

```text
                  ENTER(i) {0}
                        | walk
                        v
                   W m0(i) {0}
                        +----------------- e1 > ------------------+
                   walk |                                         v
                        |                                    R m0(i) {1}
                        +----------------- < e2 ------------------+
                        v
                   EXIT(i) {0}
```

The acquire for `e2` is placed at the write that needs its token. `e1` uses
semaphore `FULL`. `e2` uses semaphore `EMPTY`, which starts released so that
iteration zero can begin:

```text
edge    semaphore    release owner    pending_count    initial state
e1      FULL         {0}              1                blocked
e2      EMPTY        {1}              1                released for owner {0}
```

The semaphore DAG keeps owner `{0}` on the left and owner `{1}` on the right,
matching the edge DAG. The `EMPTY` arrow bypasses the loop boundary and reaches
the acquire in the next iteration:

```text
                  ENTER(i) {0}
                        | walk
                        v
            tw = acquire EMPTY(i) {0}
                     tw |
                        v
                W m0(i) [tw] {0}
                     tw | walk
                        v
             release FULL, tw {0} e1
                        +---------------- FULL > -----------------+
                   walk |                                         v
                        |                               tr = acquire FULL {1}
                        |                                      tr |
                        |                                         v
                        |                                 R m0(i) [tr] {1}
                        |                                      tr | walk
                        |                                         v
                        |                             release EMPTY, tr {1} e2
                        |                                         |
                   EXIT(i) {0}                                    |
                        | next iteration                          |
                        v                                         |
                 ENTER(i+1) {0}                                   |
                        | walk                                    |
                        v                                         |
                        +--------------- < EMPTY -----------------+
                        v
          tw2 = acquire EMPTY(i+1) {0}
                    tw2 |
                        v
               W m0(i+1) [tw2] {0}
```

For iteration zero, `EMPTY`'s initially released state lets `tw` complete
without a preceding `e2` arrival. Re-entry consumes the arrival posted by the
preceding iteration's `e2` release. After the final iteration, no later acquire
consumes its final `EMPTY` arrival. The loop carries no token in or out. A
zero-trip loop runs none of these semaphore operations and leaves `EMPTY`
released.

Compact output IR:

```text
for {                                         // no async-token iter_arg
  // partition {0}
  tw = acquire EMPTY                          {0}
  W m0 [tw]                                   {0}
  release FULL, tw                            {0}

  // partition {1}
  tr = acquire FULL                           {1}
  R m0 [tr]                                   {1}
  release EMPTY, tr                           {1}
}
```

[↑ Back to contents](#contents)

### Example: several readers and token reuse

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@fanout_not_reduced` has this one-piece loop:

```text
for {
  W m0 {0}
  R m0 {1}
  R m0 {2}
  R m0 {0}
}
```

Both reads by other owners receive the contents written by owner `{0}`. The
final owner-`{0}` read reuses owner `{0}`'s original token and adds no edge.
The next iteration waits for both other readers:

```text
edge    source          destination
f1      W m0 {0}       R m0 {1}
f2      W m0 {0}       R m0 {2}
f3      R m0(i) {1}    EXIT(i) {0}
f4      R m0(i) {2}    EXIT(i) {0}
```

```text
                                  ENTER(i) {0}
                                     | walk
                                     v
                                  W m0 {0}
                         +-----------+-----------+
                      f1 |      walk |           | f2
                         v           v           v
                   R m0 {1}     R m0 {0}     R m0 {2}
                      f3 |      walk |           | f4
                         +-----------+-----------+
                                     v
                                 EXIT {0}
```

No edge is removed: `f1` and `f2` start two independent reader paths, and
`f3` and `f4` both make the next iteration wait. No edges have both the same
source owner and destination, so no edges are merged. Edges `f3` and `f4`
have the same destination and destination owner, so they share one semaphore
and acquire while keeping two releases. The two edges from the write use
separate semaphores:

```text
edge      semaphore    release owner    pending_count    initial state
f1        TO_R1        {0}              1                blocked
f2        TO_R2        {0}              1                blocked
f3        EMPTY        {1}              2                released for owner {0}
f4        EMPTY        {2}              2                released for owner {0}
```

The semaphore DAG keeps owner `{0}` in the middle and the two readers on the
outside, as in the edge DAG. Both owner-`{0}` releases remain on one
program-order path. The two reader releases join at the next iteration's
`EMPTY` acquire:

```text
                                            ENTER(i) {0}
                                                  | walk
                                                  v
                                      t0 = acquire EMPTY(2) {0}
                                               t0 |
                                                  v
                                          W m0(i) [t0] {0}
                                                  | walk
                                                  v
                                      release TO_R1, t0 {0} f1
                +---------------------------------+
          TO_R1 |                                 | walk
                v                                 v
     t1 = acquire TO_R1 {1}           release TO_R2, t0 {0} f2
             t1 |                                 +---------------------------------+
                v                            walk |                                 | TO_R2
        R m0(i) [t1] {1}                          v                                 v
                | walk                    R m0(i) [t0] {0}               t2 = acquire TO_R2 {2}
                v                                 | walk                         t2 |
    release EMPTY, t1 {1} f3                      v                                 v
          EMPTY |                            EXIT(i) {0}                    R m0(i) [t2] {2}
                |                                 | next iteration                  | walk
                |                                 v                                 v
                |                            ENTER(i+1) {0}              release EMPTY, t2 {2} f4
                |                                 | walk                      EMPTY |
                |                                 v                                 |
                +------------------> next = acquire EMPTY(2) {0} <------------------+
                                             next |
                                                  v
                                        W m0(i+1) [next] {0}
```

For iteration zero, `EMPTY` starts released with count 2 and supplies `t0`.
For re-entry, `next` consumes the arrivals posted by `f3` and `f4`. The later
`R m0 {0}` uses `t0`, not either reader token. After the final iteration, no
later acquire consumes the two `EMPTY` arrivals. The loop returns no token. A
zero-trip loop executes no acquire or release and leaves `EMPTY` released.

Compact output IR:

```text
for {                                         // no async-token iter_arg
  // partition {0}
  t0 = acquire EMPTY(2)                       {0}
  W m0 [t0]                                   {0}
  release TO_R1, t0                           {0}
  release TO_R2, t0                           {0}
  R m0 [t0]                                   {0}   // same-iteration reuse

  // partition {1}
  t1 = acquire TO_R1                          {1}
  R m0 [t1]                                   {1}
  release EMPTY, t1                           {1}

  // partition {2}
  t2 = acquire TO_R2                          {2}
  R m0 [t2]                                   {2}
  release EMPTY, t2                           {2}
}
```

[↑ Back to contents](#contents)

### Example: different pieces synchronize separately

`test/NVWS/insert_semas_tmem_container_subviews.mlir`
`@container_with_disjoint_subviews` uses one large buffer and three smaller
views. The three smaller views do not overlap:

```text
member    range
m0        [0,256)       large buffer
m1        [0,128)       left view
m2        [128,192)     middle view
m3        [192,256)     right view

piece     range          members
P0        [0,128)        m0,m1
P1        [128,192)      m0,m2
P2        [192,256)      m0,m3
```

`[a,b)` means offsets starting at `a` and ending just before `b`. The relevant
input is:

```text
for {
  W m0 {0}              writes P0,P1,P2
  R m0 {1}              reads  P0,P1,P2
  W m1 {1}              writes P0
  W m2 {2}              writes P1
  W m3 {3}              writes P2
  R m1 {1}
  R m2 {2}
  R m3 {3}
}
```

The pass first finds these synchronization edges:

```text
DAG node          piece        synchronization edge ending here
ENTER(i) {0}      P0,P1,P2     none
W m0(i) {0}       P0,P1,P2     none
R m0(i) {1}       P0           e1: W m0(i) {0} -> R m0(i) {1}
                  P1           d1: W m0(i) {0} -> R m0(i) {1}
                  P2           d2: W m0(i) {0} -> R m0(i) {1}
W m1(i) {1}       P0           none; owner {1} already has the token
W m2(i) {2}       P1           d3: W m0(i) {0} -> W m2(i) {2}
                                e2: R m0(i) {1} -> W m2(i) {2}
W m3(i) {3}       P2           d4: W m0(i) {0} -> W m3(i) {3}
                                e3: R m0(i) {1} -> W m3(i) {3}
R m1(i) {1}       P0           none; same-owner program order
R m2(i) {2}       P1           none; same-owner program order
R m3(i) {3}       P2           none; same-owner program order
EXIT(i) {0}       P0           e4: R m1(i) {1} -> EXIT(i) {0}
                  P1           e5: R m2(i) {2} -> EXIT(i) {0}
                  P2           e6: R m3(i) {3} -> EXIT(i) {0}
```

Edges `d1` and `d2` have the same endpoints as `e1`. Once `e1` gives owner
`{1}` the token for the group, separate edges for P1 and P2 are unnecessary.
The path through `e1` and `e2` already orders `W m0` before `W m2`, so `d3`
is unnecessary. The path through `e1` and `e3` does the same for `W m3`, so
`d4` is unnecessary.

The synchronization-edge DAG after removing those edges is:

```text
                                                                                                                      ENTER(i) {0}
                                                                                                                            | walk
                                                                                                                            v
                                                                                                                       W m0(i) {0}
                                                              +--------------------------- e1 < ----------------------------+
                                                              v                                                             | walk
                                                         R m0(i) {1}                                                        |
                    +----------------- e2 < ------------------+------------ e3 > -------------+                             |
                    v                                         | walk                          v                             |
               W m2(i) {2}                               W m1(i) {1}                     W m3(i) {3}                        |
                    | walk                                    | walk                          | walk                        |
                    v                                         v                               v                             |
               R m2(i) {2}                               R m1(i) {1}                     R m3(i) {3}                        |
                    | e5                                      | e4                            | e6                          | walk
                    +-----------------------------------------+-------------------------------+-------- e4,e5,e6 > ---------+
                                                                                                                            v
                                                                                                                       EXIT(i) {0}
```

Edges `e1`, `e2`, and `e3` have different destinations. Edges `e4`, `e5`,
and `e6` have the same destination and destination owner, so they share one
semaphore and acquire while retaining three releases. The semaphore
assignment is:

```text
edge    semaphore     release owner    pending_count    initial state
e1      WHOLE_FULL    {0}              1                blocked
e2      P1_FULL       {1}              1                blocked
e3      P2_FULL       {1}              1                blocked
e4      EMPTY         {1}              3                released for owner {0}
e5      EMPTY         {2}              3                released for owner {0}
e6      EMPTY         {3}              3                released for owner {0}
```

POU places the count-3 acquire at the large write. Owner `{0}` sends the
whole buffer to owner `{1}`. Owner `{1}` reads all three pieces, sends P1 to
owner `{2}` and P2 to owner `{3}`, then writes and reads P0 with the token it
already has. All three `EMPTY` releases join at the next acquire:

```text
                                                                                                                      ENTER(i) {0}
                                                                                                                            | walk
                                                                                                                            v
                                                                                                              whole = acquire EMPTY(3) {0}
                                                                                                                      whole |
                                                                                                                            v
                                                                                                                   W m0(i) [whole] {0}
                                                                                                                            | walk
                                                                                                                            v
                                                                                                            release WHOLE_FULL, whole {0} e1
                                                              +----------------------- WHOLE_FULL < ------------------------+
                                                              v                                                             | walk
                                                 r0 = acquire WHOLE_FULL {1}                                                |
                                                           r0 |                                                             |
                                                              v                                                             |
                                                      R m0(i) [r0] {1}                                                      |
                                                              | walk                                                        |
                                                              v                                                             |
                                                 release P1_FULL, r0 {1} e2                                                 |
                    +-------------- P1_FULL < ----------------+                                                             |
                    v                                         | walk                                                        |
        p1 = acquire P1_FULL {2}                              v                                                             |
                 p1 |                            release P2_FULL, r0 {1} e3                                                 |
                    v                                         +--------- P2_FULL > -----------+                             |
            W m2(i) [p1] {2}                                  | walk                          v                             |
                    | walk                                    v                   p2 = acquire P2_FULL {3}                  |
                    v                                 W m1(i) [r0] {1}                     p2 |                             |
            R m2(i) [p1] {2}                                  | walk                          v                             |
                    | walk                                    v                       W m3(i) [p2] {3}                 EXIT(i) {0}
                    v                                 R m1(i) [r0] {1}                        | walk                        | next iteration
        release EMPTY, p1 {2} e5                              | walk                          v                             v
              EMPTY |                                         v                       R m3(i) [p2] {3}               ENTER(i+1) {0}
                    |                             release EMPTY, r0 {1} e4                    | walk                        | walk
                    |                                   EMPTY |                               v                             v
                    |                                         |                   release EMPTY, p2 {3} e6                  |
                    |                                         |                         EMPTY |                             |
                    |                                         |                               |                             v
                    +-----------------------------------------+-------------------------------+-- EMPTY > ---- next = acquire EMPTY(3) {0}
                                                                                                                       next |
                                                                                                                            v
                                                                                                                  W m0(i+1) [next] {0}
```

The buffer still has two physical copies. The whole-buffer read uses one
semaphore, while the later P1 and P2 writes use separate semaphores. `EMPTY`'s
initially released count-3 state supplies `whole` on the first iteration. The
three releases post the arrivals consumed by `next` on re-entry. After the
final iteration, no later acquire consumes those arrivals. A zero-trip loop
runs no semaphore operation and leaves `EMPTY` initially released.

Compact output IR:

```text
for {                                         // no async-token iter_arg
  // partition {0}
  whole = acquire EMPTY(3)                    {0}
  W m0 [whole]                                {0}
  release WHOLE_FULL, whole                   {0}

  // partition {1}
  r0 = acquire WHOLE_FULL                     {1}
  R m0 [r0]                                   {1}
  release P1_FULL, r0                         {1}
  release P2_FULL, r0                         {1}
  W m1 [r0]                                   {1}
  R m1 [r0]                                   {1}
  release EMPTY, r0                           {1}

  // partition {2}
  p1 = acquire P1_FULL                        {2}
  W m2 [p1]                                   {2}
  R m2 [p1]                                   {2}
  release EMPTY, p1                           {2}

  // partition {3}
  p2 = acquire P2_FULL                        {3}
  W m3 [p2]                                   {3}
  R m3 [p2]                                   {3}
  release EMPTY, p2                           {3}
}
```

[↑ Back to contents](#contents)

### Nested loops and branches

A loop or branch is one summary node in its parent's DAG. The summary says
which pieces the region reads or writes and gives each piece a boundary
owner:

```text
parent DAG: ... -> [region summary] -> ...

child DAG:         ENTER -> buffer accesses -> EXIT
```

The parent applies the same read and write rules to the summary. Each child
starts with the incoming buffer contents and builds its own edges. After an
`if`, the parent keeps only orderings established by every returning branch. A
loop that may run zero times cannot establish that ordering using only its body.

The parent uses only the summary node. It does not replace that node with the
child's accesses. An outer edge can therefore end at a loop while a separate
set of edges exists inside that loop.

[↑ Back to contents](#contents)

### Example: nested loops with a count-2 semaphore

`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain` is the first complete nested
example. Its relevant input is:

```text
outer for {
  W m0 {3}

  inner for {
    R m0 {2}
    R m0 {1}
    W m0 {1}
    R m0 {0}
  }
}
```

There is one piece P0. The inner loop first touches P0 with a read owned by
`{2}` and later writes it, so its parent summary is `P0:W:{2}`.

The two loop levels are analyzed separately. At the outer level, the inner
loop is one summary node. The complete parent edge list is:

```text
edge    source                  destination
p1      W m0 {3}               inner summary {2}
p2      inner summary {2}      EXIT outer(i) {3}
```

```text
                 ENTER outer(i) {3}
                          | walk
                          v
                     W m0(i) {3}
                          | p1
                          v
           [inner summary P0:W:{2}]
                          | p2
                          v
                  EXIT outer(i) {3}
```

The child has its own edge list:

```text
edge    source             destination
c1      ENTER inner {2}    R m0 {1}
c2      R m0 {2}           W m0 {1}
c3      W m0 {1}           R m0 {0}
c4      W m0(i,j) {1}      EXIT inner(i,j) {2}
c5      R m0(i,j) {0}      EXIT inner(i,j) {2}
```

```text
              ENTER inner(i,j) {2}
                    +------------ c1 > ------------+
                    | walk                         v
                    v                           R m0 {1}
                R m0 {2}                           | walk
                    | c2                           |
                    +------------ c2 > ------------+
                    |                              v
                    |                           W m0 {1}
                    |                              |
                    |                              v
                    +------------ < c4 ------------+------- c3 > -------+
                    |                                                   v
                    |                                                R m0 {0}
                    |                                                   | c5
                    +------------------------ < c5 ---------------------+
                    v
              EXIT inner(i,j) {2}
```

The path through `c3` and `c5` already orders the owner-`{1}` write before
`EXIT`, so ordering alone would make `c4` unnecessary. However, the pass does
not remove a loop-exit edge when its destination owner also owns the loop's
first buffer access. Owner `{2}` meets that condition here, so both `c4` and
`c5` remain.

Parent edges `p1` and `p2` end at the inner-loop summary; child edges
`c1`–`c5` stay inside the inner loop. If the inner loop continues, the `c4`
and `c5` releases supply the `FULL` acquire in the next iteration. If the loop
finishes, they supply the `FULL` acquire after the loop, whose token is used by
`p2`. The diagrams below show these cases separately.

All five child edges remain. Because `c4` and `c5` have the same destination
and destination owner, they use one `FULL` semaphore with `pending_count=2`,
while owners `{1}` and `{0}` each keep a separate release. Parent edge `p1`
uses the same semaphore for the first inner iteration. Its single release
posts two arrivals, so every `FULL` acquire waits for two arrivals:

```text
edge    semaphore      release owner / count    pending_count    initial state
c1      R1_READY       {2} x1                   1                blocked
c2      WRITE_READY    {2} x1                   1                blocked
c3      R0_READY       {1} x1                   1                blocked
p1      FULL           {3} x2                   2                blocked
c4      FULL           {1} x1                   2                blocked
c5      FULL           {0} x1                   2                blocked
p2      OUTER_EMPTY    {2} x1                   1                released for owner {3}
```

`p1`, `c4`, and `c5` use the same semaphore at different times. Before the
first inner iteration, or when the inner loop has zero trips, `p1` releases
`FULL` with `arrive_count=2`. After a nonempty inner iteration, `c4` and `c5`
both run and contribute one arrival each. Every `FULL` acquire therefore has
`pending_count=2`.

The diagrams keep each control path continuous and draw semaphore signals on
separate side paths. First, `p1` supplies the first inner acquire when the
loop runs:

```text
           ENTER outer(i) {3}
                    | walk
                    v
     outer = acquire OUTER_EMPTY {3}
              outer |
                    v
           W m0(i) [outer] {3}
                    | walk
                    v
release FULL(2), outer {3} p1 ----------- FULL(2) > -----------+
                    | enter inner                              |
                    v                                          |
          ENTER inner(i,0) {2}                                 |
                    | walk                                     |
                    v                                          |
       t2 = acquire FULL(2) {2} <------------------------------+
                 t2 |
```

If the inner loop has zero trips, the same real `p1` release instead supplies
the post-loop `done` acquire:

```text
release FULL(2), outer {3} p1 ------- FULL(2) > -------+
                    | enter inner                      |
                    v                                  |
        inner scf.for executes zero trips              |
                    | loop finishes                    |
                    v                                  |
       done = acquire FULL(2) {2} <--------------------+
                 done |
```

For an executed inner iteration, the semaphore DAG uses the same owner lanes
and branch structure as the child edge DAG. Owner `{2}` stays on the left
control path. Releases `c4` and `c5` feed a separate `FULL(2)` path that
bypasses `EXIT` and the next `ENTER`, then ends directly at the next POU
acquire:

```text
              ENTER inner(i,j) {2}
                        | walk
                        v
            t2 = acquire FULL(2) {2}
                     t2 |
                        v
           release R1_READY, t2 {2} c1
                        +---------------------------->----------------------------+
                        | walk                                           R1_READY |
                        v                                                         v
              R m0(i,j) [t2] {2}                                     t1r = acquire R1_READY {1}
                        | walk                                                t1r |
                        v                                                         v
         release WRITE_READY, t2 {2} c2                                 R m0(i,j) [t1r] {1}
                        |                                                         | walk
                        +---------------------------->----------------------------+
                        | walk                                                    v
                        v                                           t1w = acquire WRITE_READY {1}
                        |                                                     t1w |
                        |                                                         v
                        |                                               W m0(i,j) [t1w] {1}
                        |                                                         | walk
                        |                                                         v
                        |                                           release R0_READY, t1w {1} c3
                        |                                                         +----------------------->-----------------------+
                        |                                                         | walk                                 R0_READY |
                        |                                                         v                                               v
                        |                                             release FULL, t1w {1} c4                        t0 = acquire R0_READY {0}
                        |                       +----------------<----------------+                                            t0 |
                        |                  FULL |                                                                                 v
                        |                       |                                                                       R m0(i,j) [t0] {0}
                        |                       |                                                                                 | walk
                        |                       |                                                                                 v
                        |                       |                                                                      release FULL, t0 {0} c5
                        |                       +----------------------------------------<----------------------------------------+
                        |               FULL(2) |
                        v                       |
               EXIT inner(i,j) {2}              |
                        | next iteration        |
                        v                       |
             ENTER inner(i,j+1) {2}             |
                        | walk                  |
                        v                       |
                        +-----------<-----------+
                        v
           next = acquire FULL(2) {2}
                   next |
```

After the final inner iteration, the same two releases instead supply
`done`. This is the same executed body, but the control path finishes the
loop instead of entering another iteration:

```text
              ENTER inner(i,j) {2}
                        | walk
                        v
            t2 = acquire FULL(2) {2}
                     t2 |
                        v
           release R1_READY, t2 {2} c1
                        +---------------------------->----------------------------+
                        | walk                                           R1_READY |
                        v                                                         v
              R m0(i,j) [t2] {2}                                     t1r = acquire R1_READY {1}
                        | walk                                                t1r |
                        v                                                         v
         release WRITE_READY, t2 {2} c2                                 R m0(i,j) [t1r] {1}
                        |                                                         | walk
                        +---------------------------->----------------------------+
                        | walk                                                    v
                        v                                           t1w = acquire WRITE_READY {1}
                        |                                                     t1w |
                        |                                                         v
                        |                                               W m0(i,j) [t1w] {1}
                        |                                                         | walk
                        |                                                         v
                        |                                           release R0_READY, t1w {1} c3
                        |                                                         +----------------------->-----------------------+
                        |                                                         | walk                                 R0_READY |
                        |                                                         v                                               v
                        |                                             release FULL, t1w {1} c4                        t0 = acquire R0_READY {0}
                        |                       +----------------<----------------+                                            t0 |
                        |                  FULL |                                                                                 v
                        |                       |                                                                       R m0(i,j) [t0] {0}
                        |                       |                                                                                 | walk
                        |                       |                                                                                 v
                        |                       |                                                                      release FULL, t0 {0} c5
                        |                       +----------------------------------------<----------------------------------------+
                        |               FULL(2) |
                        v                       |
            EXIT inner(i,last) {2}              |
                        | loop finishes         |
                        v                       |
                        +-----------<-----------+
                        v
           done = acquire FULL(2) {2}
                   done |
```

Either real `done` acquire above implements `p2`. The control path continues
through the outer boundary, while `OUTER_EMPTY` bypasses that boundary and
ends directly at the next outer acquire:

```text
           done = acquire FULL(2) {2}
                   done |
                        v
release OUTER_EMPTY, done {2} p2 ---------------- OUTER_EMPTY > ----------------+
                        | finish outer body                                     |
                        v                                                       |
              EXIT outer(i) {3}                                                 |
                        | next iteration                                        |
                        v                                                       |
            ENTER outer(i+1) {3}                                                |
                        | walk                                                  |
                        v                                                       |
nextOuter = acquire OUTER_EMPTY {3} <-------------------------------------------+
              nextOuter |
                        v
          W m0(i+1) [nextOuter] {3}
```

Thus every `FULL` acquire waits for exactly two arrivals. Neither loop has a
token `iter_arg` in the emitted IR. The `OUTER_EMPTY` acquire is placed at the
owner-`{3}` write. Its initially released state starts outer iteration zero;
later outer iterations consume the arrival posted by the preceding `p2`
release. After the final outer iteration, no later acquire consumes `p2`'s
arrival. A zero-trip outer loop executes no semaphore operation and leaves
`OUTER_EMPTY` released.

Compact output IR:

```text
outer for {                                   // no async-token iter_arg
  // partition {3}
  outer = acquire OUTER_EMPTY                 {3}
  W m0 [outer]                                {3}
  release FULL, outer arrive_count=2          {3}

  inner for {                                 // no async-token iter_arg
    // partition {2}
    t2 = acquire FULL(2)                      {2}
    release R1_READY, t2                      {2}
    R m0 [t2]                                 {2}
    release WRITE_READY, t2                   {2}

    // partition {1}
    t1r = acquire R1_READY                    {1}
    R m0 [t1r]                                {1}
    t1w = acquire WRITE_READY                 {1}
    W m0 [t1w]                                {1}
    release R0_READY, t1w                     {1}
    release FULL, t1w                         {1}

    // partition {0}
    t0 = acquire R0_READY                     {0}
    R m0 [t0]                                 {0}
    release FULL, t0                          {0}
  }

  // partition {2}
  done = acquire FULL(2)                      {2}
  release OUTER_EMPTY, done                   {2}
}
```

[↑ Back to contents](#contents)

### Example: the boundary owner is unchanged

The preceding example changes owner from outer `{3}` to inner boundary `{2}`.
The inline `@same_owner_nested` input covers the other case: the outer
writer and inner boundary are both `{3}`.

No current test case covers this case. The plan below was generated by
running the shown input through the pass, so it is checked by the DAG dump but
not by a test case.

```text
outer for {
  W m0 {3}

  inner for {
    R m0 {3}
    R m0 {2}
    W m0 {1}
    R m0 {0}
  }
}
```

The inner loop first touches P0 with a read owned by `{3}` and later writes
P0, so its parent summary is `P0:W:{3}`. At the parent level every boundary
owner is `{3}`:

```text
node                              generated synchronization edge
ENTER outer(i) {3}                none
W m0 {3}                          none
[inner summary P0:W:{3}]          none
EXIT outer(i) {3}                 none
```

```text
           ENTER outer(i) {3}
                    | walk
                    v
               W m0(i) {3}
                    | walk
                    v
       [inner summary P0:W:{3}]
                    | walk
                    v
            EXIT outer(i) {3}
```

No parent synchronization edge is needed. The child still has six initial
edges between different owners:

```text
edge    source                 destination
c1      ENTER inner {3}        R m0 {2}
c2      R m0 {3}               W m0 {1}
c3      R m0 {2}               W m0 {1}
c4      W m0 {1}               R m0 {0}
c5      W m0(i,j) {1}          EXIT inner(i,j) {3}
c6      R m0(i,j) {0}          EXIT inner(i,j) {3}
```

```text
              ENTER inner(i,j) {3}
                        +------------------- c1 > --------------------+
                        | walk                                        v
                    R m0 {3}                                      R m0 {2}
                        | c2                                          | c3
                        +------------------- c2 > --------------------+------------------- c3 > --------------------+
                        |                                                                                           v
                        |                                                                                       W m0 {1}
                        |                                                                                           |
                        |                                                                                           v
                        +------------------------------------------ < c5 -------------------------------------------+------------------- c4 > --------------------+
                        |                                                                                                                                         v
                        |                                                                                                                                     R m0 {0}
                        |                                                                                                                                         | c6
                        +----------------------------------------------------------------- < c6 ------------------------------------------------------------------+
                        v
               EXIT inner(i,j) {3}
```

The path through `c4` and `c6` already orders the owner-`{1}` write before
`EXIT`. Even so, the pass keeps every loop-exit edge to owner `{3}`, because
that owner starts the next iteration. Both `c5` and `c6` therefore remain and
run on every iteration. No edges are removed or merged. Edges `c2` and `c3`
share a destination and destination owner, so they share one count-2 acquire.
Edges `c5` and `c6` do the same at `EXIT` and share `READY`:

```text
edge    semaphore      release owner    pending_count    initial state
c1      R2_READY       {3}              1                blocked
c2      WRITE_READY    {3}              2                blocked
c3      WRITE_READY    {2}              2                blocked
c4      R0_READY       {1}              1                blocked
c5      READY          {1}              2                released for owner {3}
c6      READY          {0}              2                released for owner {3}
```

The key difference is that the initially acquired `READY` token can be used
by the outer write and then passed directly into the first inner iteration.
There is no parent release/acquire pair between them. The entry token passes
through both same-owner boundaries:

```text
     initial = acquire READY(2) {3}
            initial |
                    v
           ENTER outer(i) {3}
            initial |
                    v
          W m0(i) [initial] {3}
            initial |
                    v
          ENTER inner(i,0) {3}
              itok = initial
```

An executed inner iteration keeps those same four owner lanes. Owner `{3}`
stays on the uninterrupted left path. Releases `c2` and `c3` meet at the
count-2 write acquire. Releases `c5` and `c6` meet at the count-2 `READY`
acquire on the owner-`{3}` path. The resulting `next` token then crosses
`EXIT` and the next `ENTER`:

```text
              ENTER inner(i,j) {3}
                   itok |
                        v
          release R2_READY, itok {3} c1
                        +---------------- R2_READY > -----------------+
                        | walk                               R2_READY |
                        v                                             v
              R m0(i,j) [itok] {3}                        t2 = acquire R2_READY {2}
                   itok |                                          t2 |
                        v                                             v
        release WRITE_READY, itok {3} c2                     R m0(i,j) [t2] {2}
                        |                                             | walk
                        |                                             v
                        |                              release WRITE_READY, t2 {2} c3
                        +-------------- WRITE_READY > ----------------+-------------- WRITE_READY > ----------------+
                        | walk                                                                       WRITE_READY(2) |
                        |                                                                                           v
                        |                                                                            t1 = acquire WRITE_READY(2) {1}
                        |                                                                                        t1 |
                        |                                                                                           v
                        |                                                                                  W m0(i,j) [t1] {1}
                        |                                                                                           | walk
                        |                                                                                           v
                        |                                                                              release R0_READY, t1 {1} c4
                        |                                                                                           +---------------- R0_READY > -----------------+
                        |                                                                                           | walk                               R0_READY |
                        |                                                                                           v                                             v
                        |                                                                               release READY, t1 {1} c5                      t0 = acquire R0_READY {0}
                        |                      +----------------------------- < READY ------------------------------+
                        |                READY |                                                                                                               t0 |
                        |                      |                                                                                                                  v
                        |                      |                                                                                                         R m0(i,j) [t0] {0}
                        |                      |                                                                                                                  | walk
                        |                      |                                                                                                                  v
                        |                      |                                                                                                      release READY, t0 {0} c6
                        |                      +---------------------------------------------------- < READY -----------------------------------------------------+
                        |             READY(2) |
                        |                      |
                        +--------- < ----------+
                        v
           next = acquire READY(2) {3}
                   next |
                        v
               EXIT inner(i,j) {3}
                   next | next inner iteration
                        v
             ENTER inner(i,j+1) {3}
            itok = next |
```

If a nonempty inner loop finishes, that same token becomes its result:

```text
       next = acquire READY(2) {3}
               next |
                    v
       EXIT inner(i,last) {3}
               next |
                    v
          result = next
```

If the inner loop has zero trips, it returns its incoming outer token instead:

```text
                  outer
              outer | inner loop has zero trips
                    v
             result = outer
```

When the outer loop continues, `result` crosses its boundary and supplies the
next outer write:

```text
                 result
             result |
                    v
            EXIT outer(i) {3}
                    | next outer iteration
                    v
          ENTER outer(i+1) {3}
             result |
                    v
         W m0(i+1) [result] {3}
```

When the outer loop finishes, it returns `result`:

```text
                 result
             result |
                    v
          EXIT outer(last) {3}
                    |
                    v
             final = result
```

A zero-trip outer loop returns the original acquired token:

```text
     initial = acquire READY(2) {3}
            initial | outer loop has zero trips
                    v
             final = initial
```

The two `READY` releases always execute together, so the acquire count is
two. If the inner loop has zero trips, it returns `outer` unchanged. If the
outer loop has zero trips, it returns `initial` unchanged. In both cases the
same token is returned; no synchronization edge is added.

Compact output IR:

```text
initial = acquire READY(2)                    {3}

final = outer for iter_arg outer = initial {
  // partition {3}
  W m0 [outer]                                {3}

  result = inner for iter_arg itok = outer {
    // partition {3}
    release R2_READY, itok                    {3}
    R m0 [itok]                               {3}
    release WRITE_READY, itok                 {3}
    next = acquire READY(2)                   {3}

    // partition {2}
    t2 = acquire R2_READY                     {2}
    R m0 [t2]                                 {2}
    release WRITE_READY, t2                   {2}

    // partition {1}
    t1 = acquire WRITE_READY(2)               {1}
    W m0 [t1]                                 {1}
    release R0_READY, t1                      {1}
    release READY, t1                         {1}

    // partition {0}
    t0 = acquire R0_READY                     {0}
    R m0 [t0]                                 {0}
    release READY, t0                         {0}

    yield next
  }

  yield result
}

// zero-trip inner: result = outer
// zero-trip outer: final = initial
```

[↑ Back to contents](#contents)

## Removing synchronization edges already guaranteed by other paths

The pass first records every required synchronization edge. It then removes an
edge only when the remaining edges do both of these jobs:

- they already make the destination owner wait for the source; and
- they give the destination owner a token it can use.

This step runs before buffer-copy selection and acquire placement. Removing an
edge therefore does not depend on whether a later loop carries a token or
acquires it at a buffer use.

If one remaining edge represents several initial edges, its release cannot be
placed before the latest represented source that uses the same token. The
pass records those sources before it removes any edge. It groups them by the
destination node, the owner that releases, and the owner that acquires.

Each loop, branch path, and parent DAG is processed separately:

```text
one path              remove an edge already guaranteed by that path
loop boundary         check the kept path into the next iteration
nested region         process each child DAG separately
```

An edge is protected from reduction when its source is still the exact active
use of the current version source and it carries an async completion
requirement. If a later same-owner access has replaced that active use, an
otherwise redundant edge from the version source is not automatically
protected and may be removed when the kept edges provide both the required
ordering and a usable token.

[↑ Back to contents](#contents)

### Synchronization edges in one path

An edge between two buffer accesses may be removed only when all of these are
true:

1. removing it does not lose a required async completion;
2. already-kept edges make the destination owner wait for its source;
3. those kept edges leave a usable token for the destination owner; and
4. the destination is a buffer access, not `ENTER`, `EXIT`, or a region.

The token check is important. A path that gives the right execution order is
not enough when the destination has no usable token.

When an edge is removed, its destination reuses the token established by the
kept edges:

```text
removed edge
  -> no new acquire
  -> destination reuses a token returned by an earlier acquire
```

Only kept edges can prove that another edge is unnecessary. A removed edge
can never be used to justify another removal. Edges without owners and edges
whose source is not a buffer access are left to the other steps.

[↑ Back to contents](#contents)

### Synchronization edges between loop iterations

An edge from a buffer access to `EXIT` makes the next iteration wait. The next
iteration's boundary owner cannot reuse the piece until the source in the
current iteration has finished.

To decide whether that edge can be removed, the pass follows the kept edges
through a simulated next iteration:

1. record the orderings established by one loop iteration;
2. find the destination owner's first access to all affected pieces in the
   next iteration;
3. follow the kept body edges to that access;
4. check that the destination owner has a usable token there; and
5. remove the edge only when the kept path already provides the ordering and the
   destination owner is not the loop's first access owner.

The last rule keeps an edge needed to give the first owner a token for the
next iteration.

[↑ Back to contents](#contents)

### Async operations and release positions

Two details must remain correct after edges are removed:

- An async source edge is kept when removing it would lose a required async
  completion; this protection applies while the source remains the version's
  exact active use.
- If one remaining edge represents several initial edges, its release stays
  after the latest represented source that uses the same token.

The second rule applies only when the sources use the same token and the later
source follows the earlier one in the same path. The release also carries every
release kind represented by those initial edges.

The examples below show these rules in the emitted acquires and releases.

[↑ Back to contents](#contents)

### Example: one release provides two arrivals

`test/NVWS/insert_semas_same_owner_mixed_completion.mlir`
`@same_owner_mixed_completion` checks that a later write by the same owner does
not lose an earlier async completion requirement. It does not test the separate
rule that keeps a release after the latest represented source. The two SMEM
members name the same buffer. Owner `{0}` writes both before owner `{1}` reads
either one:

```text
A: W m0 {0}      nvws.descriptor_load       release kind [tma_load]
B: W m1 {0}      ttg.local_store            release kind [none]
C: R m1 {1}
D: R m0 {1}
```

A starts a TMA load. B is a non-async write by the same owner using the same
token. The release after B therefore provides an immediate `[none]` arrival
for B, and `[tma_load]` makes the same semaphore track A's TMA transfer.
Because `m0` and `m1` are exact aliases, they contain one piece. The exact edge
inventory is:

```text
DAG node             synchronization edge ending here
ENTER(i) {0}         none
A: W async m0 {0}    none
B: W sync m1 {0}     none
C: R m1 {1}          q1: B {0} -> C {1}       [none,tma_load]
D: R m0 {1}          none; reuse C's token
EXIT(i) {0}           q2: D {1} -> EXIT(i) {0} [none]
```

No edge is removed. `q1` starts at B because B is the latest write, but its
release kinds also include `[tma_load]` for A. `D` replaces C as owner `{1}`'s
latest access, so `q2` starts at D. The synchronization-edge DAG is:

```text
                                                              ENTER(i) {0}
                                                                    | walk
                                                                    v
                                                          A: W async m0(i) {0}
                                                                    | walk
                                                                    v
                                                           B: W sync m1(i) {0}
                        +-------------------------------------------+
                     q1 |                                           | walk
                        v                                           v
                 C: R m1(i) {1}                                     |
                        | walk                                      |
                        v                                           |
                 D: R m0(i) {1}                                     |
                     q2 |                                           |
                        +-------------------------------------------+
                                                                    v
                                                               EXIT(i) {0}
```

The semaphore assignment is:

```text
edge / role    semaphore    release owner    pending_count    initial state
entry          EMPTY        -                1                released for owner {0}
q2             EMPTY        {1}              1                same semaphore
q1             FULL         {0}              2                blocked
```

One release provides two arrivals, so `FULL` has `pending_count=2`.
Owner `{0}` continues directly toward `EXIT` while owner `{1}` executes the
reader chain:

```text
                                                              ENTER(i) {0}
                                                                    | walk
                                                                    v
                                                     producer = acquire EMPTY(i) {0}
                                                           producer |
                                                                    v
                                                   A: W async m0(i) [producer] {0}
                                                                    | walk
                                                                    v
                                                    B: W sync m1(i) [producer] {0}
                                                                    | walk
                                                                    v
                                              release FULL, producer [none,tma_load] {0} q1
                        +-------------------------------------------+
                FULL(2) |                                           | walk
                        v                                           v
         consumer = acquire FULL(2) {1}                        EXIT(i) {0}
               consumer |                                           | next iteration
                        v                                           v
             C: R m1(i) [consumer] {1}                       ENTER(i+1) {0}
                        | walk                                      | walk
                        v                                           v
             D: R m0(i) [consumer] {1}                              |
                        | walk                                      |
                        v                                           |
      release EMPTY, consumer [none] {1} q2                         |
                  EMPTY |                                           |
                        +---------------------------> next = acquire EMPTY(i+1) {0}
                                                               next |
                                                                    v
                                                 A: W async m0(i+1) [next] {0}
```

The `FULL` acquire therefore waits for two arrivals. `[none]` posts one
immediately when the release runs; `[tma_load]` makes `FULL` track A's TMA
transfer. The acquire completes after both arrivals and the tracked TMA
transfer. This test does not cover
the separate rule for choosing the latest release position. On the final
iteration, no later acquire consumes `q2`'s arrival. A zero-trip loop executes
no acquire or release and leaves `EMPTY` released.

Compact output IR:

```text
for {                                             // no async-token iter_arg
  // partition {0}
  producer = acquire EMPTY                        {0}
  A: W async m0 [producer]                        {0}
  B: W sync m1 [producer]                         {0}
  release FULL, producer [none,tma_load]          {0}

  // partition {1}
  consumer = acquire FULL(2)                      {1}
  C: R m1 [consumer]                              {1}
  D: R m0 [consumer]                              {1}
  release EMPTY, consumer [none]                  {1}
}
```

[↑ Back to contents](#contents)

### Example: an async writer keeps its direct edge

The inline `@doc_preserved_async_edge` input shows the async rule above. It is
not a current test case; the edge and semaphore plans below come from running
this input through the pass:

```text
for {
  A: W async m0 {0}    descriptor_load, release kind [tma_load]
  B: R m0 {1}
  C: W m0 {2}
}
```

The pass records three forward edges and one edge to the next iteration. Both
direct edges from A carry the TMA completion requirement and must be kept:

```text
DAG node              synchronization edge ending here
ENTER(i) {0}          none
A: W async m0 {0}     none
B: R m0 {1}           a1: A {0} -> B {1}       [tma_load]
C: W m0 {2}           a2: A {0} -> C {2}       [tma_load]
                       a3: B {1} -> C {2}       [none]
EXIT(i) {0}            a4: C {2} -> EXIT(i) {0} [none]
```

```text
                  ENTER(i) {0}
                        | walk
                        v
              A: W async m0(i) {0}
                        +-------------------->----------------------+-------------------->----------------------+
                        | walk                                   a1 |                                           | a2
                        v                                           v                                           |
                        |                                    B: R m0(i) {1}                                     |
                        |                                        a3 |                                           |
                        |                                           +-------------------->----------------------+
                        |                                                                                       v
                        |                                                                                C: W m0(i) {2}
                        |                                                                                    a4 |
                        +------------------------------------------<--------------------------------------------+
                        v
                   EXIT(i) {0}
```

No other path makes B wait for A, so `a1` must remain. The path `a1 -> a3`
already puts C after A. The pass deliberately keeps C's direct edge from A,
so C's acquire waits for two arrivals: one tied to A's TMA completion and one
posted immediately by B's release. No edge is removed in this example. Edges
`a2` and `a3` share one destination and semaphore, but their source owners are
different, so they remain two releases:

```text
edge / role    semaphore      release owner    pending_count    initial state
entry          EMPTY          -                1                released for owner {0}
a4             EMPTY          {2}              1                same semaphore
a1             READ_READY     {0}              1                blocked
a2,a3          WRITE_READY    {0}, {1}         2                blocked
```

The semaphore DAG keeps owner `{0}` on one uninterrupted spine. Owner `{1}`
branches from `a1`; its `a3` release and owner `{0}`'s later `a2` release
provide the two arrivals required by owner `{2}`:

```text
                  ENTER(i) {0}
                        | walk
                        v
            t0 = acquire EMPTY(i) {0}
                     t0 |
                        v
       A: W async m0(i) [t0] {0}
                        | walk
                        v
    release READ_READY, t0 [tma_load] {0} a1
                        +-------------------->----------------------+
                        | walk                           READ_READY |
                        v                                           v
                        |                              t1 = acquire READ_READY {1}
                        |                                        t1 |
                        |                                           v
                        |                                B: R m0(i) [t1] {1}
                        |                                           | walk
                        |                                           v
                        |                         release WRITE_READY, t1 [none] {1} a3
                        |                                           +-------------------->----------------------+
                        |                                                                           WRITE_READY |
                        |                                                                                       v
    release WRITE_READY, t0 [tma_load] {0} a2                                                                   |
                        +------------------------------------------->-------------------------------------------+
                   walk |                                                                        WRITE_READY(2) |
                        v                                                                                       v
                   EXIT(i) {0}                                                                   t2 = acquire WRITE_READY(2) {2}
                        | next iteration                                                                     t2 |
                        v                                                                                       v
                 ENTER(i+1) {0}                                                                      C: W m0(i) [t2] {2}
                        | walk                                                                                  | walk
                        v                                                                                       v
                        |                                                                        release EMPTY, t2 [none] {2} a4
                        |                                                                                 EMPTY |
                        +------------------------------------------ < ------------------------------------------+
                        v
          next = acquire EMPTY(i+1) {0}
                   next |
                        v
     A: W async m0(i+1) [next] {0}
```

The direct and reader-path releases use different tokens, so InsertSemas keeps
them as two arrivals rather than merging them into one release. This inline
case documents the InsertSemas plan only; no current test checks the emitted
IR for this two-release async case. For iteration zero, `EMPTY` starts
released. On later iterations, `a4` supplies the acquire before A, as shown in
the same diagram.

The final `a4` arrival has no later acquire. A zero-trip loop executes none of
the shown operations and leaves `EMPTY` released.

Compact output IR:

```text
for {                                             // no async-token iter_arg
  // partition {0}
  t0 = acquire EMPTY                              {0}
  A: W async m0 [t0]                              {0}
  release READ_READY, t0 [tma_load]               {0}
  release WRITE_READY, t0 [tma_load]              {0}

  // partition {1}
  t1 = acquire READ_READY                         {1}
  B: R m0 [t1]                                    {1}
  release WRITE_READY, t1 [none]                  {1}

  // partition {2}
  t2 = acquire WRITE_READY(2)                     {2}
  C: W m0 [t2]                                    {2}
  release EMPTY, t2 [none]                        {2}
}
```

[↑ Back to contents](#contents)

### Example: one release after two reads

The inline `@doc_repeated_same_owner_sources` input shows the other
release-position rule. `whole` spans P0 and P1; `part` is another name for P0:

```text
for {
  W whole(P0,P1) {0}
  R whole(P0,P1) {1}
  R part(P0)     {1}
}
```

The first owner-`{1}` read is the latest use of P1. The later read replaces
owner `{1}`'s latest use only for P0. The complete edge inventory is:

```text
DAG node           synchronization edge ending here
ENTER(i) {0}       none
W whole {0}        none
R whole {1}        f1a: W whole {0} -> R whole {1}    P0
                   f1b: W whole {0} -> R whole {1}    P1
R part {1}         none; reuse R whole's token
EXIT(i) {0}        m2: R whole {1} -> EXIT(i) {0}     P1
                   m3: R part  {1} -> EXIT(i) {0}     P0
```

```text
              ENTER(i) {0}
                    | walk
                    v
               W whole {0}
                    +-------------------->--------------------+
                    | walk                                    | f1a,f1b
                    v                                         v
                    |                                    R whole {1}
                    |                             +-----<-----+
                    |                             | m2   walk |
                    |                             v           v
                    |                             |      R part {1}
                    |                             |           | m3
                    +--------------<--------------+-----<-----+
                    v
               EXIT(i) {0}
```

Both reads use the token returned by the same owner-`{1}` acquire. The pass
keeps `f1a` first. That edge already gives owner `{1}` the token used by
`R whole`, so the duplicate edge `f1b` is removed:

```text
after removing f1b

              ENTER(i) {0}
                    | walk
                    v
               W whole {0}
                    +-------------------->--------------------+
                    | walk                                    | f1a
                    v                                         v
                    |                                    R whole {1}
                    |                             +-----<-----+
                    |                             | m2   walk |
                    |                             v           v
                    |                             |      R part {1}
                    |                             |           | m3
                    +--------------<--------------+-----<-----+
                    v
               EXIT(i) {0}
```

No other edge is removed. The pass records both exit-edge source nodes before
forming releases. Edges `m2` and `m3` have the same destination and source
owner, so they become one release. Because both reads use the same token, that
release is placed after the later source, `R part`. The result is one count-1
release after `R part`, not two arrivals and not an early release after
`R whole`:

```text
edge / role    semaphore    release owner    pending_count    initial state
entry          EMPTY        -                1                released for owner {0}
m2,m3          EMPTY        {1}              1                same semaphore
f1a            FULL         {0}              1                blocked
```

```text
              ENTER(i) {0}
                    | walk
                    v
         t0 = acquire EMPTY {0}
                 t0 |
                    v
           W whole(i) [t0] {0}
                 t0 |
                    v
     release FULL, t0 [none] {0} f1a
                    +-------------------->--------------------+
                    | walk                                    | FULL
                    v                                         v
                    |                               t1 = acquire FULL {1}
                    |                                      t1 |
                    |                                         v
                    |                                R whole(i) [t1] {1}
                    |                                      t1 | walk
                    |                                         v
                    |                                R part(i) [t1] {1}
                    |                                      t1 |
                    |                                         v
                    |                        release EMPTY, t1 [none] {1} m2,m3
                    |                           +------<------+
               EXIT(i) {0}                      | EMPTY
                    | next iteration            |
                    v                           |
             ENTER(i+1) {0}                     |
                    | walk                      |
                    v                           |
                    +------------ < ------------+
                    v
        next = acquire EMPTY {0}
               next |
                    v
         W whole(i+1) [next] {0}
```

The current DAG dump has exactly two semaphores with pending count 1 and
places the `EMPTY` release after the second read. The pass uses the list of
earlier reads only to choose that release position; it emits no extra IR
operation for the list. The same diagram shows `EMPTY` bypassing `EXIT` and
`ENTER` to reach the next owner-`{0}` acquire.

On the final iteration, no later acquire consumes the release's arrival. A
zero-trip loop executes no acquire or release and leaves `EMPTY` released.

Compact output IR:

```text
for {                                             // no async-token iter_arg
  // partition {0}
  t0 = acquire EMPTY                              {0}
  W whole [t0]                                    {0}
  release FULL, t0 [none]                         {0}

  // partition {1}
  t1 = acquire FULL                               {1}
  R whole [t1]                                    {1}
  R part [t1]                                     {1}
  release EMPTY, t1 [none]                        {1}
}
```

[↑ Back to contents](#contents)

### Example: a direct edge is unnecessary

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces` has overlapping members:

```text
m0 = [0,256)
m1 = [64,192)

P0 = [0,64)       m0 only
P1 = [64,192)     m0 and m1
P2 = [192,256)    m0 only
```

The loop access order is:

```text
W m0 {0}
R m0 {1}
W m1 {2}
R m1 {0}
```

The complete initial edge inventory is:

```text
DAG node       synchronization edge ending here
ENTER(i) {0}   none
W m0 {0}       none
R m0 {1}       s1a: W m0 {0} -> R m0 {1}       P0
                s1b: W m0 {0} -> R m0 {1}       P1
                s1c: W m0 {0} -> R m0 {1}       P2
W m1 {2}       s2:  W m0 {0} -> W m1 {2}       P1
                s3:  R m0 {1} -> W m1 {2}       P1
R m1 {0}       s4:  W m1 {2} -> R m1 {0}       P1
EXIT(i) {0}    c0a: R m0 {1} -> EXIT(i) {0}    P0
                c0b: R m0 {1} -> EXIT(i) {0}    P2
```

The complete initial DAG keeps every operation as one node. Labels on the
same arrow name the separate piece edges with those endpoints:

```text
                  ENTER(i) {0}
                        | walk
                        v
                   W m0(i) {0}
                        +------------------------------------------------- s1a,s1b,s1c > ---------------------------------------------------+
                        | walk                                                                                                              v
                        |                                                                                                              R m0(i) {1}
                        |                                                                                                                   | c0a,c0b
                        +------------------------- s2 > --------------------------+------------------------- < s3 --------------------------+
                        |                                                         v                                                         |
                        |                                                    W m1(i) {2}                                                    |
                        |                                                         | s4                                                      |
                        +------------------------- < s4 --------------------------+                                                         |
                        v                                                                                                                   |
                   R m1(i) {0}                                                                                                              |
                        | walk                                                                                                              |
                        +--------------------------------------------------- < c0a,c0b -----------------------------------------------------+
                        v
                   EXIT(i) {0}
```

There is no edge from `W m1` to `EXIT` for P1. Edge `s4` already makes owner
`{0}` wait for that write at `R m1`, and `R m1` precedes the same owner's
`EXIT`.

At `R m0`, the pass keeps `s1a` first. That edge gives owner `{1}` a token for
the whole group, so `s1b` and `s1c` are unnecessary. The path through `s1a`
and `s3` then makes owner `{2}` wait for the owner-`{0}` write, so `s2` is
unnecessary:

```text
removed edge    ordering already provided by
s1b             s1a
s1c             s1a
s2              s1a followed by s3
```

Edges `c0a` and `c0b` survive reduction. They have the same destination and
source owner, so they become one release when semaphores are formed.

The resulting synchronization-edge DAG is:

```text
                  ENTER(i) {0}
                        | walk
                        v
                   W m0(i) {0}
                        +----------------------------------------------------- s1a > -------------------------------------------------------+
                        | walk                                                                                                              v
                        |                                                                                                              R m0(i) {1}
                        |                                                                                                                   | c0a,c0b
                        |                                                         +------------------------- < s3 --------------------------+
                        |                                                         v                                                         |
                        |                                                    W m1(i) {2}                                                    |
                        |                                                         | s4                                                      |
                        +------------------------- < s4 --------------------------+                                                         |
                        v                                                                                                                   |
                   R m1(i) {0}                                                                                                              |
                        | walk                                                                                                              |
                        +--------------------------------------------------- < c0a,c0b -----------------------------------------------------+
                        v
                   EXIT(i) {0}
```

The four remaining edges become four count-1 semaphores. The entry row is the
initial state of `EMPTY`; it is not another synchronization edge:

```text
edges          semaphore    release owner    pending_count    initial state
s1a            F01          {0}              1                blocked
s3             F12          {1}              1                blocked
s4             F20          {2}              1                blocked
entry           EMPTY        none             1                released
c0a,c0b         EMPTY        {1}              1                same semaphore
```

The semaphore DAG uses the same lane order as the edge DAG: owner `{0}` on
the left, owner `{2}` in the middle, and owner `{1}` on the right. After
`R m0`, owner `{1}` releases `F12` and then immediately releases `EMPTY` on
one vertical path. `F12` branches left to owner `{2}`; `EMPTY` branches right
to an outside path that bypasses the other owners and the loop boundary:

```text
                  ENTER(i) {0}
                        | walk
                        v
            t0 = acquire EMPTY(i) {0}
                     t0 |
                        v
                W m0(i) [t0] {0}
                     t0 |
                        v
             release F01, t0 {0} s1a
                        +------------------------------------------------------- F01 > ---------------------------------------------------------+
                        | walk                                                                                                              F01 |
                        |                                                                                                                       v
                        |                                                                                                             t1 = acquire F01 {1}
                        |                                                                                                                    t1 |
                        |                                                                                                                       v
                        |                                                                                                               R m0(i) [t1] {1}
                        |                                                                                                                    t1 | walk
                        |                                                                                                                       v
                        |                                                                                                            release F12, t1 {1} s3
                        |                                                           +------------------------- F12 < ---------------------------+
                        |                                                           |                                                      walk |
                        |                                                           |                                                           v
                        |                                                           |                                             release EMPTY, t1 {1} c0a,c0b
                        |                                                           |                                                           +--------- EMPTY > -----------+
                        |                                                           v                                                                                         |
                        |                                                 t2 = acquire F12 {2}                                                                                |
                        |                                                        t2 |                                                                                         |
                        |                                                           v                                                                                         |
                        |                                                   W m1(i) [t2] {2}                                                                                  |
                        |                                                        t2 |                                                                                         |
                        |                                                           v                                                                                         |
                        |                                                release F20, t2 {2} s4                                                                               |
                        +------------------------- F20 < ---------------------------+                                                                                         |
                        v                                                                                                                                                     |
              t0b = acquire F20 {0}                                                                                                                                           |
                    t0b |                                                                                                                                                     |
                        v                                                                                                                                                     |
                R m1(i) [t0b] {0}                                                                                                                                             |
                        | walk                                                                                                                                                |
                        v                                                                                                                                                     |
                   EXIT(i) {0}                                                                                                                                                |
                        | next iteration                                                                                                                                      |
                        v                                                                                                                                                     |
                 ENTER(i+1) {0}                                                                                                                                               |
                        | walk                                                                                                                                                |
                        v                                                                                                                                                     |
          next = acquire EMPTY(i+1) {0} -------------------------------------------------------------- < EMPTY ---------------------------------------------------------------+
                   next |
                        v
              W m0(i+1) [next] {0}
```

There is no extra `{0}->{2}` semaphore. The test checks these four semaphores
and checks that the loop has no token argument. On iteration zero, the
initially released `EMPTY` supplies `t0`. Re-entry consumes the arrivals posted
by the preceding iteration's `c0a,c0b` release. On the final iteration, no
later acquire consumes those arrivals. A zero-trip loop executes no semaphore
operation and leaves `EMPTY` released.

Compact output IR:

```text
for {                                             // no async-token iter_arg
  // partition {0}
  t0 = acquire EMPTY                              {0}
  W m0 [t0]                                       {0}
  release F01, t0 [none]                          {0}
  t0b = acquire F20                               {0}
  R m1 [t0b]                                      {0}

  // partition {1}
  t1 = acquire F01                                {1}
  R m0 [t1]                                       {1}
  release F12, t1 [none]                          {1}
  release EMPTY, t1 [none]                        {1}

  // partition {2}
  t2 = acquire F12                                {2}
  W m1 [t2]                                       {2}
  release F20, t2 [none]                          {2}
}
```

[↑ Back to contents](#contents)

### Example: a loop-exit edge is unnecessary

`test/NVWS/insert_semas_local_buffer_reuse.mlir`
`@local_n_owner_aliased_buffers` uses two partly overlapping members:

```text
m0 = [0,128)
m1 = [64,192)

P0 = [0,64)      in m0
P1 = [64,128)    in m0 and m1
P2 = [128,192)   in m1
```

The loop access order is:

```text
W m0 {0}
R m0 {1}
W m1 {2}
R m1 {0}
```

The complete initial edge inventory is:

```text
DAG node       synchronization edge ending here
ENTER(i)       none
W m0 {0}       none
R m0 {1}       l1a: W m0 {0} -> R m0 {1}       P0
                l1b: W m0 {0} -> R m0 {1}       P1
W m1 {2}       l2a: W m0 {0} -> W m1 {2}       P1
                l2b: R m0 {1} -> W m1 {2}       P1
R m1 {0}       l3a: W m1 {2} -> R m1 {0}       P1
                l3b: W m1 {2} -> R m1 {0}       P2
EXIT(i)        c0:  R m0 {1} -> EXIT(i) {0}    P0
                c2:  R m1 {0} -> EXIT(i) {2}    P2
```

There is no edge from `W m1` to `EXIT` for P1. Edges `l3a` and `l3b` already
make owner `{0}` wait for that write at `R m1`, and `R m1` precedes the same
owner's `EXIT`.

The complete initial DAG keeps one node for each operation. Labels on one
arrow retain every piece edge:

```text
                  ENTER(i)
                        | walk
                        v
                   W m0(i) {0}
                        +--------------------------------------------------- l1a,l1b > -----------------------------------------------------+
                        | walk                                                                                                              v
                        |                                                                                                              R m0(i) {1}
                        |                                                                                                                   | c0
                        +------------------------ l2a > --------------------------+------------------------ < l2b --------------------------+
                        |                                                         v                                                         |
                        |                                                    W m1(i) {2}                                                    |
                        |                                                     | l3a,l3b                                                      |
                        +---------------------- < l3a,l3b -----------------------+                                                         |
                        v                                                                                                                   |
                   R m1(i) {0}                                                                                                              |
                        | c2                                                                                                                |
                        +------------------------------------------------------ < c0 -------------------------------------------------------+
                        v
                    EXIT(i)
```

The pass removes four edges. For `c2`, the kept path starts with owner `{0}`'s
program order from `R m1(i)` to `W m0(i+1)`, then uses `l1a` and `l2b` to
reach owner `{2}` at `W m1(i+1)`:

```text
removed edge    ordering already provided by
l1b             l1a
l2a             l1a followed by l2b
l3b             l3a
c2              same-owner next-iteration order, then l1a followed by l2b
```

Edge `c0` remains because owner `{0}` needs a token for P0 before the first
access of the next iteration. No other edge is removed.

The final synchronization-edge DAG is:

```text
                  ENTER(i)
                        | walk
                        v
                   W m0(i) {0}
                        +----------------------------------------------------- l1a > -------------------------------------------------------+
                        | walk                                                                                                              v
                        |                                                                                                              R m0(i) {1}
                        |                                                                                                                   | c0
                        |                                                         +------------------------ < l2b --------------------------+
                        |                                                         v                                                         |
                        |                                                    W m1(i) {2}                                                    |
                        |                                                        | l3a                                                       |
                        +------------------------ < l3a -------------------------+                                                         |
                        v                                                                                                                   |
                   R m1(i) {0}                                                                                                              |
                        | walk                                                                                                              |
                        +------------------------------------------------------ < c0 -------------------------------------------------------+
                        v
                    EXIT(i)
```

The emitted POU plan therefore has exactly four semaphores with pending count
1. The entry row records `EMPTY`'s initial state; it is not an edge:

```text
edges        semaphore    release owner    pending_count    initial state
entry         EMPTY        none             1                released
c0            EMPTY        {1}              1                same semaphore
l1a          F01          {0}              1                blocked
l2b          F12          {1}              1                blocked
l3a          F20          {2}              1                blocked
```

As in the previous example, the edge and semaphore DAGs use lanes `{0}`,
`{2}`, `{1}` from left to right. Owner `{1}` releases `F12` and then
immediately releases `EMPTY` on one vertical path. `F12` branches left to
owner `{2}`, while `EMPTY` branches right to the outside recurrence path:

```text
                  ENTER(i)
                        | walk
                        v
            t0 = acquire EMPTY(i) {0}
                     t0 |
                        v
                W m0(i) [t0] {0}
                     t0 |
                        v
             release F01, t0 {0} l1a
                        +------------------------------------------------------- F01 > ---------------------------------------------------------+
                        | walk                                                                                                              F01 |
                        |                                                                                                                       v
                        |                                                                                                             t1 = acquire F01 {1}
                        |                                                                                                                    t1 |
                        |                                                                                                                       v
                        |                                                                                                               R m0(i) [t1] {1}
                        |                                                                                                                    t1 | walk
                        |                                                                                                                       v
                        |                                                                                                           release F12, t1 {1} l2b
                        |                                                           +------------------------- F12 < ---------------------------+
                        |                                                           |                                                      walk |
                        |                                                           |                                                           v
                        |                                                           |                                               release EMPTY, t1 {1} c0
                        |                                                           |                                                           +--------- EMPTY > -----------+
                        |                                                           v                                                                                         |
                        |                                                 t2 = acquire F12 {2}                                                                                |
                        |                                                        t2 |                                                                                         |
                        |                                                           v                                                                                         |
                        |                                                   W m1(i) [t2] {2}                                                                                  |
                        |                                                        t2 |                                                                                         |
                        |                                                           v                                                                                         |
                        |                                                release F20, t2 {2} l3a                                                                              |
                        +------------------------- F20 < ---------------------------+                                                                                         |
                        v                                                                                                                                                     |
              t0b = acquire F20 {0}                                                                                                                                           |
                    t0b |                                                                                                                                                     |
                        v                                                                                                                                                     |
                R m1(i) [t0b] {0}                                                                                                                                             |
                        | walk                                                                                                                                                |
                        v                                                                                                                                                     |
                   EXIT(i)                                                                                                                                                    |
                        | next iteration                                                                                                                                      |
                        v                                                                                                                                                     |
                 ENTER(i+1)                                                                                                                                                   |
                        | walk                                                                                                                                                |
                        v                                                                                                                                                     |
          next = acquire EMPTY(i+1) {0} -------------------------------------------------------------- < EMPTY ---------------------------------------------------------------+
                   next |
                        v
              W m0(i+1) [next] {0}
```

The kept path from `R m1(i)` through the next `W m0`, `l1a`, and `l2b`
already reaches owner `{2}` at `W m1(i+1)`, so `c2` is unnecessary. The loop
carries no semaphore token. Iteration zero uses `EMPTY`'s initial state;
re-entry uses `c0`. The final `EMPTY` arrival has no later acquire. A
zero-trip loop executes no semaphore operation and leaves `EMPTY` released.

Compact output IR:

```text
for {                                             // no async-token iter_arg
  // partition {0}
  t0 = acquire EMPTY                              {0}
  W m0 [t0]                                       {0}
  release F01, t0 [none]                          {0}
  t0b = acquire F20                               {0}
  R m1 [t0b]                                      {0}

  // partition {1}
  t1 = acquire F01                                {1}
  R m0 [t1]                                       {1}
  release F12, t1 [none]                          {1}
  release EMPTY, t1 [none]                        {1}

  // partition {2}
  t2 = acquire F12                                {2}
  W m1 [t2]                                       {2}
  release F20, t2 [none]                          {2}
}
```

[↑ Back to contents](#contents)

## Placing acquires and releases

After unnecessary edges are removed and physical copies are chosen, the pass
places every acquire and release. These symbolic locations are final before
EMIT-IR starts. EMIT-IR renders those locations without splitting an `if` or
moving synchronization across a branch boundary.

Every buffer access and every release records which acquire produced its
token. When a token crosses a loop or an `if`, the pass also records the same
token returned by that region. EMIT-IR follows those recorded links; it does
not guess from the owner or from the nearest token in the code.

The scheduled examples show `loop.stage` and `loop.cluster`. The pass does not
change an input `loop.stage`. It may change `loop.cluster` so releases happen
before their matching acquires.

[↑ Back to contents](#contents)

### Straight-line code

At each buffer access, the pass does one of two things:

1. Reuse a token already held by that owner when it is still valid for every
   buffer piece touched by the access.
2. Otherwise, place an acquire immediately before the access.

The matching releases are placed after their source buffer-access operations.
Each release records its source token, the acquire it supplies, and the release
kinds that determine how the semaphore gets its arrivals.

An edge that enters or leaves a surrounding loop or `if` is handled the same
way. The acquire is normally placed at the buffer access that needs the token.
An acquire after an inner loop may instead connect the inner synchronization
to later outer code. A loop may carry the exact token when later code or the
next iteration needs it.

[↑ Back to contents](#contents)

### `if` branches

Both branches start with the same incoming tokens. When a boundary token is
needed after the `if`, the pass handles each branch separately and combines
their results:

```text
both branches keep the same incoming token
  keep that token after the `if`

one branch changes ownership and the other does not
  complete the ownership change inside that branch
  changed branch returns its new boundary-owner token
  unchanged branch returns the incoming token

the first access after the `if` has another owner
  first return the boundary-owner token from every branch
  release that `if` result after the conditional
  the next owner acquires before its buffer access
```

The `if` result also keeps the completion carried by a pass-through input. A
later cross-partition release therefore still tracks an unfinished async
operation instead of posting an immediate arrival.

#### A following owner uses the `if` result

`test/NVWS/insert_semas_local_cfg.mlir`
`@local_if_consumption_continues_after_join` checks the handoff after an
`if`. Its shape is:

```text
W buffer {0}

if cond {
  R buffer {1}
} else {
  no buffer access
}

R buffer {1}
```

The parent sees the `if` as owner `{0}`. The then branch temporarily moves
ownership to `{1}` and returns it to `{0}`. The following owner-`{1}`
read is a separate handoff:

```text
parent DAG node                 synchronization edge ending here
W before {0}                   none
[if summary P0:R:{0}]          none
R after {1}                    e3: if summary {0} -> R after {1}

then DAG node                  synchronization edge ending here
ENTER if {0}                   none
R inside {1}                   e1: ENTER if {0} -> R inside {1}
EXIT if {0}                    e2: R inside {1} -> EXIT if {0}

else DAG node                  synchronization edge ending here
ENTER if {0}                   none
EXIT if {0}                    none
```

```text
parent synchronization-edge DAG

                  W before {0}
                       | walk
                       v
             [if summary P0:R:{0}]
                       +------------------- e3 > -------------------+
                                                                  v
                                                        R after {1}

then-child synchronization-edge DAG

                  ENTER if {0}
                       +------------------- e1 > -------------------+
                                                                  v
                                                        R inside {1}
                       +------------------- < e2 -------------------+
                       v
                   EXIT if {0}

else-child synchronization-edge DAG

                  ENTER if {0}
                       | walk
                       v
                   EXIT if {0}
```

Each edge uses a count-1 blocked semaphore:

```text
edge    semaphore    release owner    acquire owner    pending_count
e1      TO_1         {0}              {1}              1
e2      BACK_0       {1}              {0}              1
e3      NEXT_1       {0}              {1}              1
```

The then path completes `e1` and `e2` inside the branch:

```text
                         W before [t0] {0}
                              t0 |
                                  v
                             scf.if cond
                                  | then
                                  v
                    release TO_1, t0 {0} e1
                                  +-------------------- TO_1 > --------------------+
                                                                                   v
                                                                      t1 = acquire TO_1 {1}
                                                                                 t1 |
                                                                                    v
                                                                      R inside [t1] {1}
                                                                                 t1 | walk
                                                                                    v
                                                                 release BACK_0, t1 {1} e2
                                                                                    |
                back0 = acquire BACK_0 {0} <---------------- BACK_0 ----------------+
                             back0 |
                                   v
                             yield back0
```

The else path yields `t0` unchanged. Both paths therefore return owner
`{0}`. Only after the merge does `e3` hand the token to the following
reader:

```text
                       out0 = if result {0}
                                   out0 |
                                        v
                    release NEXT_1, out0 {0} e3
                                        +---------------- NEXT_1 > ----------------+
                                                                                   v
                                                                    next1 = acquire NEXT_1 {1}
                                                                               next1 |
                                                                                     v
                                                                      R after [next1] {1}
```

Compact output IR:

```text
t0 = W buffer {0}

out0 = if cond {
  release TO_1, t0 [none]                         {0}
  t1 = acquire TO_1                               {1}
  R buffer [t1]                                   {1}

  release BACK_0, t1 [none]                       {1}
  back0 = acquire BACK_0                          {0}
  yield back0                                     {0,1}
} else {
  yield t0                                        {0,1}
}

release NEXT_1, out0 [none]                       {0}
next1 = acquire NEXT_1                            {1}
R buffer [next1]                                  {1}
```

The release to `NEXT_1` cannot be placed before the `if`: that would use
`t0` and bypass the token returned by the branch that ran.

If the following access is still owner `{0}`, it reuses the `if` result and
there is no handoff after the conditional:

```text
out0 = if cond { ... yield back0 } else { yield t0 }
op buffer [out0] {0}
```

Releases in different branches do not add their counts because only one
branch runs. Releases that execute together on one path do add their counts.
If different pieces have different boundary owners, there is no single token
result; the pass keeps each piece's branch-local releases separate and joins
them at the later acquire.

[↑ Back to contents](#contents)

### Loops

A loop has two relevant kinds of edges:

```text
edge into the loop       earlier code -> first matching buffer use in the body
edge to the next turn    last body use -> EXIT -> first matching use next turn
```

If different pieces have different boundary owners, the pass places an
acquire at the first use for each owner. There is no single loop token to
carry.

When the pieces share one boundary owner, the pass chooses among these cases:

- If the body ends with a valid boundary-owner token, return that token from
  the loop.
- If an exact incoming token must remain valid, pass it into the loop and
  return the exact token produced for the next turn.
- Otherwise, place the acquire at the first buffer use that needs it. If an
  inner loop already uses the same semaphore, an acquire after that inner loop
  can connect the inner and outer synchronization.
- If the loop has no buffer use that needs a token, no token is added merely
  because a loop exists.

[↑ Back to contents](#contents)

### Choosing whether a loop carries a token

The planner first processes the loop body and learns which exact token reaches
each use and each exit. It then chooses one of two loop forms:

```text
acquire at use
  no valid token reaches the buffer use
  place the acquire immediately before that use
  loop entry and exit carry no token merely because a loop exists

carry through loop
  pass the exact input token as a loop argument
  reuse it while it remains valid
  if ownership changes, obtain the returned token at a buffer use or loop tail
  return the exact token the body ends with, possibly the unchanged input
```

The loop carries a token when any of these facts requires it:

- the body already ends with an exact boundary-owner token the loop can
  return;
- a valid incoming token must remain available after the loop;
- the required next-turn acquire cannot be represented safely at one buffer
  use.

The planner makes this decision while building the final POU plan.

An owner change by itself does not require a loop token. Consider:

```text
op {1}
for boundary owner {2}
op {2}
```

Owner `{2}` has no token before the loop, so there is nothing to carry into
it. The planner releases the owner-`{1}` token and acquires it at owner `{2}`'s
first buffer use. Loop-closing releases then supply either the next
iteration's acquire or the acquire before the final `op {2}`. A zero-trip
loop uses the entry release for that final acquire.

Now consider:

```text
op {1}
for boundary owner {1}
op {2}
```

The loop already receives the token used by the first owner-`{1}` operation.
After the loop, a completed owner-`{1}` token is needed for the release to
owner `{2}`. The planner therefore carries the token:

```text
zero trips:     output token = input token
nonzero trips:  output token = exact token returned by the final iteration
```

The returned owner-`{1}` token is then released to owner `{2}`, which acquires
it before the final buffer operation. This avoids an extra same-owner
release/acquire round trip at the loop boundary.

Before emitting IR, the pass checks that every used token comes from an
acquire or was passed through a loop or branch. It also checks that the
placement does not conflict with existing `loop.stage` annotations.

[↑ Back to contents](#contents)

### Example: acquire at the buffer use

`test/NVWS/insert_semas.mlir` `@local_loop_carried_and_result` contains the
running loop. For its `buffer.id=104` group, no valid owner-`{0}` token reaches
the first write:

```text
for {
  W m0 {0}
  R m0 {1}
}
```

The complete edge inventory is:

```text
DAG node       synchronization edge ending here
ENTER(i) {0}   none
W m0 {0}       none
R m0 {1}       e1: W m0 {0} -> R m0 {1}
EXIT(i) {0}    e2: R m0 {1} -> EXIT(i) {0}
```

No edge is removed or merged. The synchronization-edge DAG is:

```text
                  ENTER(i) {0}
                        | walk
                        v
                   W m0(i) {0}
                        +----------------- e1 > ------------------+
                   walk |                                         v
                        |                                    R m0(i) {1}
                        +----------------- < e2 ------------------+
                        v
                   EXIT(i) {0}
```

The edges use this semaphore assignment:

```text
edge / role    semaphore    release owner    pending_count    initial state
e1             FULL         {0}              1                blocked
entry          EMPTY        none             1                released
e2             EMPTY        {1}              1                same semaphore
```

The planner places the `EMPTY` acquire at the write that needs its token:

```text
                  ENTER(i) {0}
                        | walk
                        v
            tw = acquire EMPTY(i) {0}
                     tw |
                        v
                W m0(i) [tw] {0}
                     tw | walk
                        v
             release FULL, tw {0} e1
                        +---------------- FULL > -----------------+
                   walk |                                         v
                        |                               tr = acquire FULL {1}
                        |                                      tr |
                        |                                         v
                        |                                 R m0(i) [tr] {1}
                        |                                      tr | walk
                        |                                         v
                        |                             release EMPTY, tr {1} e2
                        |                                         |
                   EXIT(i) {0}                                    |
                        | next iteration                          |
                        v                                         |
                 ENTER(i+1) {0}                                   |
                        | walk                                    |
                        v                                         |
                        +--------------- < EMPTY -----------------+
                        v
          tw2 = acquire EMPTY(i+1) {0}
                    tw2 |
                        v
               W m0(i+1) [tw2] {0}
```

The loop carries no semaphore token. Iteration zero consumes `EMPTY`'s initial
state at its first write. Re-entry consumes the preceding iteration's `e2`
release. After the final iteration, no later acquire consumes the final
`EMPTY` arrival. A zero-trip loop executes none of these operations and leaves
`EMPTY` released.

Compact output IR:

```text
result_value = for iter_arg value = init {       // no token iter_arg
  // partition {0}
  tw = acquire EMPTY                             {0}
  W m0(value) [tw]                               {0}
  release FULL, tw [none]                        {0}
  next_value = next(value)                       {0}

  // partition {1}
  tr = acquire FULL                              {1}
  R m0 [tr]                                      {1}
  release EMPTY, tr [none]                       {1}

  yield next_value                               {0,1}   // scalar only
}
```

[↑ Back to contents](#contents)

### Example: fixed stages with tokenless cross-stage POU

`test/NVWS/insert_semas_staged_pou.mlir`
`@staged_tokenless_cross_stage_pou` has one buffer copy, nested loops, and
fixed stage assignments:

```text
outer for {
  inner for {
    touch0 m0 {0} stage 0
    touch1 m0 {0} stage 1
    touch2 m0 {1} stage 1
  }
}
```

The pass treats each touch as a write, so the inner edge inventory is:

```text
DAG node            synchronization edge ending here
ENTER inner {0}     none
touch0 m0 {0}       none
touch1 m0 {0}       none; same owner as touch0
touch2 m0 {1}       e1: touch1 {0} -> touch2 {1}
EXIT inner {0}      e2: touch2 {1} -> EXIT inner {0}
```

There is one piece, P0. The outer DAG contains only the inner summary between
owner-`{0}` boundaries, so it has no synchronization edge:

```text
                  ENTER outer {0}
                        | walk
                        v
         [inner summary P0:W:{0}]
                        | walk
                        v
                   EXIT outer {0}
```

No inner edge is removed or merged. Its synchronization-edge DAG is:

```text
                  ENTER inner {0}
                        | walk
                        v
                 touch0 m0 {0}
                        | walk
                        v
                 touch1 m0 {0}
                        +----------------- e1 > ------------------+
                   walk |                                         v
                        |                                touch2 m0 {1}
                        +----------------- < e2 ------------------+
                        v
                   EXIT inner {0}
```

The completed plan uses exactly two count-1 semaphores:

```text
edge / role    semaphore    release owner    pending_count    initial state
entry          EMPTY        none             1                released
e1             FULL         {0}              1                blocked
e2             EMPTY        {1}              1                same semaphore
```

The emitted schedules are:

```text
operation                                      loop.stage    loop.cluster
acquire EMPTY, buffer, touch0                  0             2
touch1, release FULL                           1             1
acquire FULL, buffer, touch2, release EMPTY    1             1
```

`touch0` acquires `EMPTY` at its point of use. `touch1` reuses that exact
token in the same inner iteration, even though it has a different stage.
Owner `{0}` then releases `FULL`, and owner `{1}` acquires it for `touch2`.
Neither loop has a semaphore-token argument or result.

The semaphore DAG keeps owner `{0}` on the left and owner `{1}` on the right:

```text
                     ENTER inner(i,j) {0}
                              | walk
                              v
           t0 = acquire EMPTY(i,j) {0} [c2,s0]
                           t0 |
                              v
              touch0 m0(i,j) [t0] {0} [c2,s0]
                              | walk
                              v
              touch1 m0(i,j) [t0] {0} [c1,s1]
                           t0 |
                              v
            release FULL, t0 {0} e1 [c1,s1]
                              +---------------- FULL > ----------------+
                         walk |                                        v
                              |                      t1 = acquire FULL {1} [c1,s1]
                              |                                     t1 |
                              |                                        v
                              |                     touch2 m0(i,j) [t1] {1} [c1,s1]
                              |                                     t1 | walk
                              |                                        v
                              |                   release EMPTY, t1 {1} e2 [c1,s1]
                              |                                        |
                     EXIT inner(i,j) {0}                               |
                              | next inner iteration                    |
                              v                                        |
                    ENTER inner(i,j+1) {0}                              |
                              | walk                                   |
                              v                                        |
                              +---------------- < EMPTY ----------------+
                              v
         next = acquire EMPTY(i,j+1) {0} [c2,s0]
                           next |
                              v
            touch0 m0(i,j+1) [next] {0} [c2,s0]
```

For the first executed inner iteration, `EMPTY`'s initially released state
supplies `t0`. The `release EMPTY` at stage 1 of iteration `j` supplies the
`acquire EMPTY` at stage 0 of iteration `j+1`. The differing stage numbers do
not require a token to be carried through the loop: the semaphore edge orders
the two operations.

If the inner loop finishes and the outer loop later starts another inner
iteration, the same `EMPTY` arrival supplies that next `acquire EMPTY`. After
the final executed `touch2`, no later acquire consumes the final arrival. A
zero-trip inner or outer loop executes no acquire or release and leaves
`EMPTY` initially released.

Compact output IR:

```text
outer for {                                      // no async-token iter_arg or result
  inner for {                                    // no async-token iter_arg or result
    // partition {0}
    t0 = acquire EMPTY                           {0}   [c2,s0]
    touch0 m0 [t0]                               {0}   [c2,s0]
    touch1 m0 [t0]                               {0}   [c1,s1]
    release FULL, t0 [none]                      {0}   [c1,s1]

    // partition {1}
    t1 = acquire FULL                            {1}   [c1,s1]
    touch2 m0 [t1]                               {1}   [c1,s1]
    release EMPTY, t1 [none]                     {1}   [c1,s1]
  }
}

// release EMPTY at stage 1 supplies the next acquire EMPTY at stage 0
// zero trip: no acquire or release executes
```

[↑ Back to contents](#contents)

### Example: an `if` returns the boundary token

`test/NVWS/insert_semas_conditional_multi_result.mlir`
`@conditional_multi_result_if_token` has a two-copy TMEM accumulator. The
relevant access shape is:

```text
for {
  W acc {1}                 MMA

  if cond {
    R acc {0}
  } else {
    no acc access
  }
}
```

The `if` starts and ends with owner `{1}`, the owner immediately before
the conditional. The then branch temporarily moves ownership to `{0}`; the
else branch keeps `{1}`. The exact edge inventory is:

```text
parent DAG node                  synchronization edge ending here
ENTER loop {1}                  none
W MMA {1}                       none
[if summary P0:R:{1}]           none
EXIT loop {1}                   none

then DAG node                   synchronization edge ending here
ENTER if {1}                    none
R acc {0}                       e1: ENTER if {1} -> R acc {0}
EXIT if {1}                     e2: R acc {0} -> EXIT if {1}

else DAG node                   synchronization edge ending here
ENTER if {1}                    none
EXIT if {1}                     none
```

The parent and child DAGs are separate:

```text
parent

                  ENTER loop {1}
                        | walk
                        v
                   W MMA {1}
                        | walk
                        v
         [if summary P0:R:{1}]
                        | walk
                        v
                   EXIT loop {1}

then child

                  ENTER if {1}
                        +----------------- e1 > ------------------+
                   walk |                                         v
                        |                                    R acc {0}
                        +----------------- < e2 ------------------+
                        v
                   EXIT if {1}

else child

                  ENTER if {1}
                        | walk
                        v
                   EXIT if {1}
```

The parent has no synchronization edge around the `if`. The then path keeps
both cross-partition edges. The else path has no synchronization edge.

`EMPTY` supplies the initial owner-`{1}` token before the loop and also
implements the then-path handback `e2`. The semaphore assignment is:

```text
edge / role    semaphore    release owner    pending_count    initial state
entry           EMPTY        none             1                released
e1              FULL         {1}              1                blocked
e2              EMPTY        {0}              1                same semaphore as entry
```

The loop carries the owner-`{1}` token. On the then path, both ownership
changes stay inside the branch:

```text
                initial = acquire EMPTY {1}
                             initial |
                                     v
                    for iter_arg tok = initial
                                  tok |
                                      v
                         ENTER loop(i) {1}
                                  tok |
                                      v
                           W MMA(i) [tok] {1}
                                  tok | walk
                                      v
                                 scf.if cond
                                      | then
                                      v
                    release FULL, tok [tc5mma] {1} e1
                                      +-------------------- FULL > ---------------------+
                                                                                       v
                                                                          tr = acquire FULL {0}
                                                                                     tr |
                                                                                        v
                                                                                R acc(i) [tr] {0}
                                                                                     tr | walk
                                                                                        v
                                                                 release EMPTY, tr [none] {0} e2
                                                                                        |
              back = acquire EMPTY {1} <-------------------- EMPTY ---------------------+
                            back |
                                 v
                           yield back
                            back |
                                 v
                 next = if result {1}
                            next |
                                 v
                      loop yields next
                            next |
                                 v
               W MMA(i+1) [next] {1}
```

On the else path, ownership never leaves `{1}`, so the branch returns the
incoming token without a semaphore operation:

```text
                initial = acquire EMPTY {1}
                             initial |
                                     v
                    for iter_arg tok = initial
                                  tok |
                                      v
                         ENTER loop(i) {1}
                                  tok |
                                      v
                           W MMA(i) [tok] {1}
                                  tok | walk
                                      v
                                 scf.if cond
                                      | else
                                      v
                                 yield tok
                                      |
                                      v
                           next = if result {1}
                                    next |
                                         v
                              loop yields next
                                    next |
                                         v
                         W MMA(i+1) [next] {1}
```

The paths are alternatives. The then path performs
`{1} -> {0} -> {1}` and returns the new owner-`{1}` token. The else path
returns the old owner-`{1}` token. There is no else-path release and no
unconditional acquire after the `if`.

`EMPTY.pending_count` and `FULL.pending_count` are both one. A zero-trip
loop returns `initial`. After a nonzero loop, its result is the exact token
returned by the final executed `if`.

Compact output IR:

```text
initial = acquire EMPTY                              {1}

loop_use, loop_value, loop_token =
    for iter_args (use_acc, value, tok = initial) {
  // partition {1}
  W MMA [tok]                                        {1}

  branch_value, branch_use, next_tok = if cond {
    // partition {1}
    release FULL, tok [tc5mma]                       {1}

    // partition {0}
    tr = acquire FULL                                {0}
    R acc [tr]                                       {0}
    release EMPTY, tr [none]                         {0}

    // partition {1}
    back = acquire EMPTY                             {1}
    yield iv, true, back                             {0,1}
  } else {
    // partition {1}
    yield value, use_acc, tok                        {0,1}
  }

  next_value = next(branch_value)                    {0,1}
  yield branch_use, next_value, next_tok             {0,1}
}
```

[↑ Back to contents](#contents)

### Example: a buffer-use acquire is carried

A token acquired at a buffer use may also become the token for the next
iteration.

`test/NVWS/insert_semas_per_edge_tmem.mlir`
`@tmem_single_producer_multi_consumer_fanout` uses a TMEM buffer with two
physical copies:

```text
for {
  W first {0}
  R reader1 {1}
  R reader2 {2}
  W final {0}
}
```

There is one piece, P0. The complete edge inventory is:

```text
DAG node         synchronization edge ending here
ENTER(i) {0}     none
W first {0}      none
R reader1 {1}    e1: W first {0} -> R reader1 {1}
R reader2 {2}    e2: W first {0} -> R reader2 {2}
W final {0}      e3: R reader1 {1} -> W final {0}
                 e4: R reader2 {2} -> W final {0}
EXIT(i) {0}      none
```

No edge is removed or merged. The two reader paths are independent and join
at the final write:

```text
                                                        ENTER(i) {0}
                                                              | walk
                                                              v
                                                       W first(i) {0}
                    +----------------- < e1 ------------------+----------------- e2 > ------------------+
                    v                                    walk |                                         v
            R reader1(i) {1}                                  |                                 R reader2(i) {2}
                    +----------------- e3 > ------------------+----------------- < e4 ------------------+
                                                              v
                                                       W final(i) {0}
                                                              | walk
                                                              v
                                                         EXIT(i) {0}
```

Edges `e3` and `e4` share a destination and destination owner, so they use
one semaphore and acquire. Their source owners differ, so they remain two
releases. The entry row is the initial `EMPTY` state:

```text
edges        semaphore    release owner    pending_count    initial state
e1           TO_R1        {0}              1                blocked
e2           TO_R2        {0}              1                blocked
entry         EMPTY        none             2                released
e3            EMPTY        {1}              2                same semaphore
e4            EMPTY        {2}              2                same semaphore
```

Both readers release the count-2 `EMPTY` semaphore. Its acquire must occur
before `W final(i)`, so that token already exists at the end of iteration `i`.
The next iteration's `W first(i+1)` can reuse the same token. The complete
semaphore DAG keeps the two owner-`{0}` releases on one ordered spine and the
reader paths separate:

```text
                                             initial = acquire EMPTY(2) at root
                                                      initial |
                                                              v
                                               scf.for iter_arg carry=initial
                                                              +---------------------------------- > ------------------------------------+
                                                     executes |                                                               zero trip |
                                                              v                                                                         v
                                                        ENTER(i) {0}                                                             result=initial
                                                        carry |
                                                              v
                                                   W first(i) [carry] {0}
                                                        carry | walk
                                                              v
                                                 release TO_R1, carry {0} e1
                    +--------------- < TO_R1 -----------------+
                    v                                    walk |
         t1 = acquire TO_R1 {1}                  release TO_R2, carry {0} e2
                 t1 |                                         +--------------- TO_R2 > -----------------+
                    v                                         |                                         v
          R reader1(i) [t1] {1}                               |                              t2 = acquire TO_R2 {2}
                 t1 | walk                                    |                                      t2 |
                    v                                         |                                         v
        release EMPTY, t1 {1} e3                              |                               R reader2(i) [t2] {2}
                    |                                         |                                      t2 | walk
                    |                                         |                                         v
                    |                                         |                             release EMPTY, t2 {2} e4
                    +--------------- EMPTY > -----------------+--------------- < EMPTY -----------------+
                                                              v
                                                 next = acquire EMPTY(2) {0}
                                                         next |
                                                              v
                                                    W final(i) [next] {0}
                                                         next | walk
                                                              v
                                                     EXIT(i) yields next
                                                              +---------------------------------- > ------------------------------------+
                                               next iteration |                                                           loop finishes |
                                                              v                                                                         v
                                                       ENTER(i+1) {0}                                                              result=next
                                                   carry=next |
                                                              v
                                                  W first(i+1) [carry] {0}
```

The count-2 acquire is needed by `W final`, and its token remains valid for
`W first` in the next iteration. The loop therefore carries `next`. A
zero-trip loop returns the `initial` token, which was acquired before the
loop, unchanged.

After the final iteration, the loop returns `next` as its result.

Compact output IR:

```text
initial = acquire EMPTY(2)                       root

result_value, result_token = for iter_args (value = 0, carry = initial) {
  // partition {0}
  W first [carry]                                {0}
  release TO_R1, carry [none]                    {0}
  release TO_R2, carry [none]                    {0}
  next = acquire EMPTY(2)                        {0}
  W final [next]                                 {0}

  // partition {1}
  t1 = acquire TO_R1                             {1}
  R reader1 [t1]                                 {1}
  release EMPTY, t1 [none]                       {1}

  // partition {2}
  t2 = acquire TO_R2                             {2}
  R reader2 [t2]                                 {2}
  release EMPTY, t2 [none]                       {2}

  next_value = next(value)                       {0,1,2}
  yield next_value, next                         {0,1,2}
}

// zero trip: result_token = initial
```

[↑ Back to contents](#contents)

### Example: nested loops without carried tokens

`test/NVWS/insert_semas_nested_ws_inner_loop.mlir`
`@nested_ws_inner_loop` has one outer WS loop and one inner loop:

```text
outer for {
  inner for {
    W acc {1}       tc_gen5_mma
    R acc {0}       tmem_load
  }
}
```

At the outer level, the inner loop starts and ends with owner `{1}`, the same
owner as the outer `ENTER` and `EXIT`. The parent inventory has no
synchronization edge:

```text
parent DAG node                  synchronization edge ending here
ENTER outer(i) {1}              none
[inner summary P0:W:{1}]        none
EXIT outer(i) {1}               none
```

The parent synchronization-edge DAG is:

```text
                         ENTER outer(i) {1}
                                  | walk
                                  v
                   [inner summary P0:W:{1}]
                                  | walk
                                  v
                         EXIT outer(i) {1}
```

The child inventory has two edges:

```text
DAG node                 synchronization edge ending here
ENTER inner(i,j) {1}     none
W acc(i,j) {1}           none
R acc(i,j) {0}           c1: W acc(i,j) {1} -> R acc(i,j) {0}
EXIT inner(i,j) {1}      c2: R acc(i,j) {0} -> EXIT inner(i,j) {1}
```

```text
                  ENTER inner(i,j) {1}
                            | walk
                            v
                     W acc(i,j) {1}
                            +-------------------c1 >--------------------+
                            | walk                                      v
                            |                                    R acc(i,j) {0}
                            +-------------------< c2--------------------+
                            v
                   EXIT inner(i,j) {1}
```

No edge is removed or merged.

`c1` becomes `FULL`. Edge `c2` and the initial ready state use `EMPTY`:

```text
edge / role    semaphore    release owner    pending_count    initial state
c1             FULL         {1}              1                blocked
entry           EMPTY        none             1                released
c2              EMPTY        {0}              1                same semaphore
```

The pass places both acquires immediately before the inner buffer accesses
that need them. The parent DAG contains the child as one summary. Each
complete alternative below includes the body so that its `c2` release has a
visible source. If the next inner turn is in the same outer iteration,
the owner-`{1}` control path crosses the loop boundary while the `EMPTY` rail
bypasses it and ends at the next acquire:

```text
                  ENTER inner(i,j) {1}
                            | walk
                            v
                 tw = acquire EMPTY {1}
                         tw |
                            v
                   W acc(i,j) [tw] {1}
                            | walk
                            v
            release FULL, tw [tc5mma] {1} c1
                            +----------------- FULL > ------------------+
                       walk |                                           v
                            |                                 tr = acquire FULL {0}
                            |                                        tr |
                            |                                           v
                            |                                  R acc(i,j) [tr] {0}
                            |                                        tr | walk
                            |                                           v
                            |                            release EMPTY, tr [none] {0} c2
                            |                                           +-------------- EMPTY > ----------------+
                            v                                                                                   |
                   EXIT inner(i,j) {1}                                                                          |
                            | next inner iteration                                                              |
                            v                                                                                   |
                 ENTER inner(i,j+1) {1}                                                                         |
                            | walk                                                                              |
                            v                                                                                   |
                next = acquire EMPTY {1} -----------------------------------<-----------------------------------+
                       next |
                            v
                 W acc(i,j+1) [next] {1}
```

If the next executed inner turn is in a later outer iteration, the arrival
posted by that release remains available while control crosses both loop
boundaries. The first acquire in that later inner-loop execution consumes it:

```text
                 ENTER inner(i,last) {1}
                            | walk
                            v
                 tw = acquire EMPTY {1}
                         tw |
                            v
                 W acc(i,last) [tw] {1}
                            | walk
                            v
            release FULL, tw [tc5mma] {1} c1
                            +----------------- FULL > ------------------+
                       walk |                                           v
                            |                                 tr = acquire FULL {0}
                            |                                        tr |
                            |                                           v
                            |                                R acc(i,last) [tr] {0}
                            |                                        tr | walk
                            |                                           v
                            |                            release EMPTY, tr [none] {0} c2
                            |                                           +-------------- EMPTY > ----------------+
                            v                                                                                   |
                 EXIT inner(i,last) {1}                                                                         |
                            | inner finishes                                                                    |
                            v                                                                                   |
                    EXIT outer(i) {1}                                                                           |
                            | later outer iteration                                                             |
                            v                                                                                   |
                   ENTER outer(k) {1}                                                                           |
                            | walk                                                                              |
                            v                                                                                   |
                  ENTER inner(k,0) {1}                                                                          |
                            | walk                                                                              |
                            v                                                                                   |
                first = acquire EMPTY {1} -----------------------------------<----------------------------------+
                      first |
                            v
                 W acc(k,0) [first] {1}
```

The two alternatives are exclusive. Neither loop carries a semaphore token.

After the final inner iteration, its `EMPTY` arrival has no following acquire
in that run of the inner loop. It remains ready for the next time the inner
loop executes, including in a later outer iteration. Neither loop carries a
semaphore token, and no acquire or release is moved outside both loops.
For the first executed inner iteration, `EMPTY`'s initially released state
supplies `tw`. EMIT-IR removes the old tokens attached to the TMEM operations
because the semaphores now order the accesses. A zero-trip inner or outer loop
executes no acquire or release, so the ready `EMPTY` state remains available.

Compact output IR:

```text
outer for {                                      // no token iter_arg or result
  inner for {                                    // no token iter_arg or result
    // partition {1}
    tw = acquire EMPTY                           {1}
    W acc [tw]                                   {1}
    release FULL, tw [tc5mma]                    {1}

    // partition {0}
    tr = acquire FULL                            {0}
    R acc [tr]                                   {0}
    release EMPTY, tr [none]                     {0}
  }
}
```

[↑ Back to contents](#contents)

### Example: reading the buffer after the inner loop

The next function in the same file,
`@nested_ws_inner_loop_parent_continuation`, adds a read after the inner loop:

```text
outer for {
  inner for {
    W acc {1}
    R acc {0}
  }

  R acc {0}          outer read after the inner loop
}
```

The inner loop and outer loop have separate edge sets:

```text
edge    source                       destination
c1      W inner {1}                  R inner {0}
c2      R inner {0}                  EXIT inner {1}
p1      inner summary P0:W:{1}       R outer {0}
p2      R outer {0}                  EXIT outer {1}
```

```text
parent synchronization-edge DAG

                  ENTER outer(i) {1}
                        | walk
                        v
          [inner summary P0:W:{1}]
                        +----------------- p1 > ------------------+
                   walk |                                         v
                        |                                 R outer(i) {0}
                        +----------------- < p2 ------------------+
                        v
                  EXIT outer(i) {1}

child synchronization-edge DAG

                  ENTER inner(i,j) {1}
                        | walk
                        v
                  W inner(i,j) {1}
                        +----------------- c1 > ------------------+
                   walk |                                         v
                        |                                 R inner(i,j) {0}
                        +----------------- < c2 ------------------+
                        v
                  EXIT inner(i,j) {1}
```

No edge is removed or merged. Parent and child edges remain separate.

The pass forms four count-1 semaphores:

```text
LOCAL_EMPTY    next inner write or acquire after the inner loop; initially released
LOCAL_FULL     inner write -> inner read
OUTER_FULL     completed inner loop -> outer read
OUTER_EMPTY    outer read -> owner {1}; initially released
```

The semaphore assignment is:

```text
edge / role        semaphore      release owner       pending_count    initial state
c1                 LOCAL_FULL     {1}                 1                blocked
entry               LOCAL_EMPTY    none                1                released
c2                  LOCAL_EMPTY    {0}                 1                same semaphore
prepare             LOCAL_EMPTY    {1}                 1                same semaphore
p1                 OUTER_FULL     {1}                 1                blocked
entry               OUTER_EMPTY    none                1                released
p2                  OUTER_EMPTY    {0}                 1                same semaphore
```

An acquire before the outer loop consumes `OUTER_EMPTY`'s initially released
state. Its token is not used by a buffer access. This is deliberate: it makes
the later `prepare` acquire wait for the release from `R outer` instead of
completing from the initial state:

```text
OUTER_EMPTY starts released
  -> entry = acquire OUTER_EMPTY             consumes the initial release
  -> R outer {0}
  -> release OUTER_EMPTY {0}                 posts the next arrival
  -> prepare = acquire OUTER_EMPTY           consumes that arrival
```

Thus `entry` and `prepare` acquire the same semaphore at different times, but
they do not consume the same arrival.

When the inner loop continues, `c2` supplies the next inner acquire. Owner
`{1}` stays on the left, owner `{0}` stays on the right, and the
`LOCAL_EMPTY` rail bypasses the loop boundary:

```text
              ENTER inner(i,j) {1}
                        | walk
                        v
         wtok = acquire LOCAL_EMPTY {1}
                   wtok |
                        v
             W inner(i,j) [wtok] {1}
                        | walk
                        v
    release LOCAL_FULL, wtok [tc5mma] {1} c1
                        +------------- LOCAL_FULL > --------------+
                   walk |                                         v
                        |                           rtok = acquire LOCAL_FULL {0}
                        |                                    rtok |
                        |                                         v
                        |                              R inner(i,j) [rtok] {0}
                        |                                    rtok | walk
                        |                                         v
                        |                      release LOCAL_EMPTY, rtok [none] {0} c2
                        |                                         +------------- LOCAL_EMPTY > ---------------+
                        v                                                                                     |
               EXIT inner(i,j) {1}                                                                            |
                        | next inner iteration                                                                |
                        v                                                                                     |
             ENTER inner(i,j+1) {1}                                                                           |
                        | walk                                                                                |
                        v                                                                                     |
         next = acquire LOCAL_EMPTY {1} ----------------------------------<-----------------------------------+
                   next |
                        v
            W inner(i,j+1) [next] {1}
```

If an executed inner loop finishes, `done` consumes `c2`. For a zero-trip
inner loop, `done` instead consumes `LOCAL_EMPTY`'s initial state on the first
outer iteration or the previous `prepare` release on a later iteration. The
complete executed-inner path and the post-loop semaphore path are:

```text
             ENTER inner(i,last) {1}
                        | walk
                        v
         wtok = acquire LOCAL_EMPTY {1}
                   wtok |
                        v
           W inner(i,last) [wtok] {1}
                        | walk
                        v
    release LOCAL_FULL, wtok [tc5mma] {1} c1
                        +------------- LOCAL_FULL > --------------+
                   walk |                                         v
                        |                           rtok = acquire LOCAL_FULL {0}
                        |                                    rtok |
                        |                                         v
                        |                            R inner(i,last) [rtok] {0}
                        |                                    rtok | walk
                        |                                         v
                        |                      release LOCAL_EMPTY, rtok [none] {0} c2
                        |                                         +------------- LOCAL_EMPTY > ---------------+
                        v                                                                                     |
             EXIT inner(i,last) {1}                                                                           |
                        | loop finishes                                                                       |
                        v                                                                                     |
         done = acquire LOCAL_EMPTY {1} ----------------------------------<-----------------------------------+
                   done |
                        v
     release OUTER_FULL, done [none] {1} p1
                        +------------- OUTER_FULL > --------------+
                   walk |                                         v
                        |                            to = acquire OUTER_FULL {0}
                        |                                      to |
                        |                                         v
                        |                                R outer(i) [to] {0}
                        |                                      to | walk
                        |                                         v
                        |                       release OUTER_EMPTY, to [none] {0} p2
                        |                                         |
        prepare = acquire OUTER_EMPTY {1} -----< OUTER_EMPTY -----+
                prepare |
                        v
     release LOCAL_EMPTY, prepare [none] {1}
```

The `prepare` release has two exclusive consumers in the next outer
iteration. When the inner loop executes, it supplies the first inner acquire:

```text
        prepare = acquire OUTER_EMPTY {1}
                prepare |
                        v
     release LOCAL_EMPTY, prepare [none] {1}
                        +---------------------------------- LOCAL_EMPTY > ------------------------------------+
                   walk |                                                                                     |
                        v
                EXIT outer(i) {1}                                                                             |
                        | next outer iteration                                                                |
                        v
              ENTER outer(i+1) {1}                                                                            |
                        | walk                                                                                |
                        v
             ENTER inner(i+1,0) {1}                                                                           |
                        | walk                                                                                |
                        v                                                                                     |
         first = acquire LOCAL_EMPTY {1} ----------------------------------<----------------------------------+
                  first |
                        v
           W inner(i+1,0) [first] {1}
```

When that inner loop has zero trips, the post-loop `done` acquire consumes the
arrival posted by the same `prepare` release instead:

```text
        prepare = acquire OUTER_EMPTY {1}
                prepare |
                        v
     release LOCAL_EMPTY, prepare [none] {1}
                        +---------------------------------- LOCAL_EMPTY > ------------------------------------+
                   walk |                                                                                     |
                        v
                EXIT outer(i) {1}                                                                             |
                        | next outer iteration                                                                |
                        v
              ENTER outer(i+1) {1}                                                                            |
                        | inner scf.for executes zero trips                                                   |
                        v                                                                                     |
         done = acquire LOCAL_EMPTY {1} ----------------------------------<-----------------------------------+
                   done |
                        v
```

The root acquire makes the later `prepare` acquire wait for the arrival posted
by the outer read's `p2` release. Neither loop carries a semaphore token. If
the outer loop is zero-trip, only the root acquire executes.

Compact output IR:

```text
entry = acquire OUTER_EMPTY                      {1}   // consumes initial release

outer for {                                      // no token iter_arg or result
  inner for {                                    // no token iter_arg or result
    // partition {1}
    wtok = acquire LOCAL_EMPTY                   {1}
    W inner [wtok]                               {1}
    release LOCAL_FULL, wtok [tc5mma]            {1}

    // partition {0}
    rtok = acquire LOCAL_FULL                    {0}
    R inner [rtok]                               {0}
    release LOCAL_EMPTY, rtok [none]             {0}
  }

  // partition {1}
  done = acquire LOCAL_EMPTY                     {1}
  release OUTER_FULL, done [none]                {1}
  prepare = acquire OUTER_EMPTY                  {1}
  release LOCAL_EMPTY, prepare [none]            {1}

  // partition {0}
  to = acquire OUTER_FULL                        {0}
  R outer [to]                                   {0}
  release OUTER_EMPTY, to [none]                 {0}
}
```

**Future investigation:** Another possible outcome is to carry the owner-`{1}`
token through the inner loop. This would remove the unused root acquire and
the separate `prepare` bridge, and it would use two semaphores instead of four.
The current pass does not produce this form:

```text
S1 = semaphore(initially released)
S0 = semaphore(blocked)

outer for {
  t1 = acquire S1 {1}

  after_inner = inner for iter_args(tok = t1) {
    W acc [tok] {1}
    release S0, tok [tc5mma] {1}

    t0 = acquire S0 {0}
    R acc [t0] {0}
    release S1, t0 [none] {0}

    next1 = acquire S1 {1}
    yield next1
  }

  release S0, after_inner [none] {1}
  outer0 = acquire S0 {0}
  R acc [outer0] {0}
  release S1, outer0 [none] {0}
}
```

[↑ Back to contents](#contents)

### Example: fixed stages in a nested POU loop

`test/NVWS/insert_semas_nested_carrier.mlir`
`@scheduled_relocated_acquire_boundaries` has fixed stages on the inner write,
inner read, and read after the inner loop. Acquires for the inner accesses stay
at those buffer uses. A post-inner `LOCAL_EMPTY` acquire connects the inner
completion to later outer code.

The relevant input is:

```text
outer for {
  W acc {0}                         tmem_alloc/store

  inner for {
    W acc {1} stage=0               tc_gen5_mma
    R acc {0} stage=1               tmem_load
  }

  R acc {0} stage=0                 post-inner load
}
```

The remaining edge DAG is:

```text
parent edges
  p1: W outer {0}             -> inner summary P0:W:{1}
  p2: inner summary P0:W:{1}  -> R post {0}

child edges
  c1: W MMA {1}               -> R inner {0}
  c2: R inner {0}             -> EXIT inner {1}
```

```text
parent synchronization-edge DAG

                  ENTER outer(i) {0}
                        | walk
                        v
                   W outer(i) {0}
                        +----------------- p1 > ------------------+
                   walk |                                         v
                        |                       [inner summary P0:W:{1}]
                        +----------------- < p2 ------------------+
                        v
                   R post(i) {0}
                        | walk
                        v
                  EXIT outer(i) {0}

child synchronization-edge DAG

                                                        ENTER inner(i,j) {1}
                                                                  | walk
                                                                  v
                                                           W MMA(i,j) {1}
                        +----------------- < c1 ------------------+
                        v                                    walk |
                R inner(i,j) {0}                                  |
                        +----------------- c2 > ------------------+
                                                                  v
                                                         EXIT inner(i,j) {1}
```

No edge is removed or merged.

All four edges use count-one semaphores:

```text
edge / role    semaphore      release owner    pending_count    initial state
entry,p2       OUTER_EMPTY    {1}              1                released
p1,c2          LOCAL_EMPTY    {0}              1                blocked
c1             LOCAL_FULL     {1}              1                blocked
```

The generated schedule locations are:

```text
operation                                    owner    cluster    stage
acquire LOCAL_EMPTY at inner MMA             {1}      3          0
release LOCAL_FULL after MMA [tc5mma]         {1}      3          0
acquire LOCAL_FULL at inner read              {0}      2          1
release LOCAL_EMPTY after inner read          {0}      2          1
post-inner acquire LOCAL_EMPTY                {1}      inner boundary
release OUTER_EMPTY after inner loop          {1}      3          0
acquire OUTER_EMPTY at post-inner read        {0}      4          0
```

The root acquire and outer token stay in owner `{0}`. The inner loop has no
token `iter_arg`. The diagrams use `[cN,sM]` for cluster `N`, stage `M` and
keep owner `{0}` on the left and owner `{1}` on the right.

When the inner loop executes and continues, `p1` supplies its first acquire
and `c2` supplies the next one. Both semaphore rails end at the acquires;
neither rail flows through an `ENTER` or `EXIT`:

```text
            initial = acquire OUTER_EMPTY at root
                      initial |
                              v
                scf.for iter_arg out=initial
                              | executes
                              v
                     ENTER outer(i) {0}
                          out |
                              v
                    W outer(i) [out] {0}
                          out | walk
                              v
               release LOCAL_EMPTY, out {0} p1
                              +----------------- enter inner > -------------------+
                  LOCAL_EMPTY |                                                   v
                              |                                         ENTER inner(i,0) {1}
                              |                                                   | walk
                              |                                                   v
                              +-------- LOCAL_EMPTY > ---------first = acquire LOCAL_EMPTY {1} [c3,s0]
                                                                            first |
                                                                                  v
                                                                   W MMA(i,0) [first] {1} [c3,s0]
                                                                            first | walk
                                                                                  v
                                                          release LOCAL_FULL, first [tc5mma] {1} c1 [c3,s0]
                              +------------------ LOCAL_FULL < -------------------+
                              v                                              walk |
             tr = acquire LOCAL_FULL {0} [c2,s1]                                  |
                           tr |                                                   |
                              v                                                   |
                R inner(i,0) [tr] {0} [c2,s1]                                     |
                           tr | walk                                              |
                              v                                                   |
        release LOCAL_EMPTY, tr [none] {0} c2 [c2,s1]                             |
                  LOCAL_EMPTY |                                                   v
                              |                                          EXIT inner(i,0) {1}
                              |                                                   | next inner iteration
                              |                                                   v
                              |                                         ENTER inner(i,1) {1}
                              |                                                   | walk
                              |                                                   v
                              +-------- LOCAL_EMPTY > ---------next = acquire LOCAL_EMPTY {1} [c3,s0]
                                                                             next |
                                                                                  v
                                                                    W MMA(i,1) [next] {1} [c3,s0]
```

On the final executed inner iteration, `c2` bypasses `EXIT inner` and ends at
the unstamped post-loop acquire. The following `p2` release uses `[c3,s0]`;
the post-inner read acquires `OUTER_EMPTY` at `[c4,s0]`. That token is the
outer loop's carried `out` token:

```text
                                                                       ENTER inner(i,last) {1}
                                                                                  | walk
                                                                                  v
                                                               wtok = acquire LOCAL_EMPTY {1} [c3,s0]
                                                                             wtok |
                                                                                  v
                                                                  W MMA(i,last) [wtok] {1} [c3,s0]
                                                                             wtok | walk
                                                                                  v
                                                          release LOCAL_FULL, wtok [tc5mma] {1} c1 [c3,s0]
                              +------------------ LOCAL_FULL < -------------------+
                              v                                              walk |
             tr = acquire LOCAL_FULL {0} [c2,s1]                                  |
                           tr |                                                   |
                              v                                                   |
              R inner(i,last) [tr] {0} [c2,s1]                                    |
                           tr | walk                                              |
                              v                                                   |
        release LOCAL_EMPTY, tr [none] {0} c2 [c2,s1]                             |
                  LOCAL_EMPTY |                                                   v
                              |                                        EXIT inner(i,last) {1}
                              |                                                   | loop finishes
                              |                                                   v
                              +------- LOCAL_EMPTY > --------done = acquire LOCAL_EMPTY {1} [unstamped]
                                                                             done |
                                                                                  v
                                                           release OUTER_EMPTY, done [none] {1} p2 [c3,s0]
                                                                                  |
            out = acquire OUTER_EMPTY {0} [c4,s0] -------- < OUTER_EMPTY ---------+
                          out |
                              v
                 R post(i) [out] {0} [c4,s0]
                          out | walk
                              v
                  EXIT outer(i) yields out
                              +------------------------------ loop finishes > --------------------------------+
                              | next outer iteration                                                          v
                              v                                                                          result=out
                ENTER outer(i+1) receives out
                          out |
                              v
                   W outer(i+1) [out] {0}
                          out | walk
                              v
               release LOCAL_EMPTY, out {0} p1
```

When the inner loop has zero trips, the same unstamped `done` acquire consumes
the arrival posted by the real `p1` release. The control path completes the
zero-trip loop; the `LOCAL_EMPTY` rail remains separate until `done`:

```text
                     ENTER outer(i) {0}
                          out |
                              v
                    W outer(i) [out] {0}
                          out | walk
                              v
               release LOCAL_EMPTY, out {0} p1
                              +----------------- enter inner > -------------------+
                  LOCAL_EMPTY |                                                   v
                              |                                   inner scf.for executes zero trips
                              |                                                   | loop finishes
                              |                                                   v
                              +------- LOCAL_EMPTY > --------done = acquire LOCAL_EMPTY {1} [unstamped]
                                                                             done |
                                                                                  v
                                                           release OUTER_EMPTY, done [none] {1} p2 [c3,s0]
```

Thus `p1` chooses first-inner versus zero-trip `done`, and `c2` chooses
next-inner versus final `done`; none of those alternatives execute together.
If the outer loop is zero-trip, it returns `initial`; after the final executed
outer iteration, it returns `out`.

Compact output IR:

```text
initial = acquire OUTER_EMPTY                    root

outer_value, outer_token = outer for iter_args (value, out = initial) {
  // partition {0}
  W outer [out]                                  {0}
  release LOCAL_EMPTY, out [none]                {0}

  inner for {                                    // no token iter_arg or result
    // partition {1}
    wtok = acquire LOCAL_EMPTY                   {1}   [c3,s0]
    W MMA [wtok]                                 {1}   [c3,s0]
    release LOCAL_FULL, wtok [tc5mma]            {1}   [c3,s0]

    // partition {0}
    tr = acquire LOCAL_FULL                      {0}   [c2,s1]
    R inner [tr]                                 {0}   [c2,s1]
    release LOCAL_EMPTY, tr [none]               {0}   [c2,s1]
  }

  // partition {1}: finish the inner loop
  done = acquire LOCAL_EMPTY                     {1}   [unstamped]
  release OUTER_EMPTY, done [none]               {1}   [c3,s0]

  // partition {0}
  out_next = acquire OUTER_EMPTY                 {0}   [c4,s0]
  R post [out_next]                              {0}   [c4,s0]
  next_value = next(value)                       {0}

  yield next_value, out_next                     {0,1}
}

// zero-trip outer: outer_token = initial
```

[↑ Back to contents](#contents)

### Example: branch-local handback with fixed stages

`test/NVWS/insert_semas_nested_carrier.mlir`
`@branch_completion_requires_carrier` has fixed stages inside an `if`.
The boundary owner of the `if` is `{1}`.

```text
outer for {
  W acc {0}

  inner for {
    W mma0 acc {1} stage=0

    if cond {
      W mma1 acc {1} stage=1
      R branch acc {0} stage=1
    } else {
      no acc access
    }

    R final acc {0} stage=1
  }

  R post acc {0}
}
```

The parent and inner edge inventories are:

```text
parent
  p1: W outer {0}          -> inner summary P0:W:{1}
  p2: inner summary {1}    -> R post {0}

inner
  c1: W mma1 {1}           -> R branch {0}       (then only)
  b2: R branch {0}         -> EXIT if {1}         (then only)
  c2: if summary P0:W:{1}  -> R final {0}
  c3: R final {0}          -> EXIT inner {1}
```

```text
parent synchronization-edge DAG

                                                               ENTER outer(i) {0}
                                                                        | walk
                                                                        v
                                                                 W outer(i) {0}
                            +------------------- < p1 -------------------+
                            v
              [inner summary P0:W:{1}]
                            +------------------- p2 > -------------------+
                                                                        v
                                                                  R post(i) {0}
                                                                        | walk
                                                                        v
                                                               EXIT outer(i) {0}
```

```text
inner synchronization-edge DAG

                  ENTER inner(i,j) {1}
                            | walk
                            v
                     W mma0(i,j) {1}
                            | walk
                            v
                  [if summary P0:W:{1}]
                            +------------------- c2 > -------------------+
                                                                        v
                                                               R final(i,j) {0}
                            +------------------- < c3 -------------------+
                            v
                   EXIT inner(i,j) {1}
```

```text
then-child synchronization-edge DAG

                       ENTER if {1}
                            | walk
                            v
                       W mma1 {1}
                            +------------------- c1 > -------------------+
                                                                        v
                                                                 R branch {0}
                            +------------------- < b2 -------------------+
                            v
                        EXIT if {1}
```

```text
else-child synchronization-edge DAG

                       ENTER if {1}
                            | walk
                            v
                        EXIT if {1}
```

No edge is removed or merged. The parent, inner, and child DAGs remain
separate. The important placement rule is that `b2` is completed inside the
then branch. The `if` therefore returns owner `{1}` on both paths:

```text
then path   R branch {0} -> release/acquire back to {1} -> yield new token
else path   no buffer access                              -> yield input token
```

Only after that merge does `c2` hand the buffer to owner `{0}` for
`R final`. The semaphore assignment is:

```text
edge / role    semaphore         release owner    pending_count    initial state
entry,p2       OUTER_EMPTY       p2:{1}           1                released
p1,c3          LOOP_BACK         {0}              1                blocked
c1             TO_BRANCH_READ    {1}              1                blocked
b2             BACK_TO_WRITER    {0}              1                blocked
c2             TO_POST_READ      {1}              1                blocked
```

`[cN,sM]` means cluster `N`, stage `M`. Owner `{1}` stays on the
left and owner `{0}` stays on the right.

On the then path, the branch-local handback happens before the `if` result:

```text
                    tw = acquire LOOP_BACK {1} [c5,s0]
                                  tw |
                                     v
                         W mma0(i,j) [tw] {1} [c5,s0]
                                  tw | walk
                                     v
                                scf.if cond
                                     | then
                                     v
                         W mma1 [tw] {1} [c2,s1]
                                  tw | walk
                                     v
          release TO_BRANCH_READ, tw [tc5mma] {1} c1 [c2,s1]
                                     +---------------- TO_BRANCH_READ > ----------------+
                                                                                         v
                                                            tb = acquire TO_BRANCH_READ {0} [c3,s1]
                                                                                       tb |
                                                                                          v
                                                                            R branch [tb] {0} [c3,s1]
                                                                                       tb | walk
                                                                                          v
                                                   release BACK_TO_WRITER, tb [none] {0} b2 [c3,s1]
                                                                                          |
        back = acquire BACK_TO_WRITER {1} [c2,s1] <------------- BACK_TO_WRITER ----------+
                                back |
                                     v
                              yield back
                                back |
                                     v
                     branch = if result {1}
                              branch | walk
                                     v
       release TO_POST_READ, branch [tc5mma] {1} c2 [c2,s1]
                                     +------------------ TO_POST_READ > ------------------+
                                                                                           v
                                                               tf = acquire TO_POST_READ {0} [c4,s1]
                                                                                         tf |
                                                                                            v
                                                                              R final [tf] {0} [c4,s1]
                                                                                         tf | walk
                                                                                            v
                                                         release LOOP_BACK, tf [none] {0} c3 [c4,s1]
                                                                                            |
              next = acquire LOOP_BACK {1} [c5,s0] <---------------- LOOP_BACK -------------+
                                next |
                                     v
                       W mma0(i,j+1) [next] {1}
```

On the else path, the `if` returns `tw` unchanged. The following handoff
still starts after the `if` and consumes its result:

```text
                    tw = acquire LOOP_BACK {1} [c5,s0]
                                  tw |
                                     v
                         W mma0(i,j) [tw] {1} [c5,s0]
                                  tw | walk
                                     v
                                scf.if cond
                                     | else
                                     v
                                yield tw
                                  tw |
                                     v
                     branch = if result {1}
                              branch | walk
                                     v
       release TO_POST_READ, branch [tc5mma] {1} c2 [c2,s1]
                                     +------------------ TO_POST_READ > ------------------+
                                                                                           v
                                                               tf = acquire TO_POST_READ {0} [c4,s1]
                                                                                         tf |
                                                                                            v
                                                                              R final [tf] {0} [c4,s1]
                                                                                         tf | walk
                                                                                            v
                                                         release LOOP_BACK, tf [none] {0} c3 [c4,s1]
                                                                                            |
              next = acquire LOOP_BACK {1} [c5,s0] <---------------- LOOP_BACK -------------+
```

The `TO_POST_READ` release carries `[tc5mma]`. The else path passes through
the stage-0 MMA token, so a later cross-partition release must retain that
unfinished completion. The then path has already completed its branch MMA
through `TO_BRANCH_READ`, but using the same release kind after the merge is
safe and keeps one exact result token for both paths.

The rest of the outer flow is independent of which branch ran:

```text
initial = acquire OUTER_EMPTY at root
      initial |
              v
outer for iter_arg out = initial
      out |
          v
W outer [out] {0}
      out |
          v
release LOOP_BACK, out {0} p1
          |
          v
inner loop executes the flow above
          |
          v
done = acquire LOOP_BACK {1}
     done |
          v
release OUTER_EMPTY, done {1} p2
          |
          v
out_next = acquire OUTER_EMPTY {0}
 out_next |
          v
R post [out_next] {0}
          |
          v
outer loop yields out_next
```

If the inner loop is zero-trip, `p1` supplies `done`. If the outer loop is
zero-trip, it returns `initial`.

Compact output IR:

```text
initial = acquire OUTER_EMPTY                         root

outer_value, outer_token =
    outer for iter_args (value, out = initial) {
  // partition {0}
  W outer [out]                                       {0}
  release LOOP_BACK, out [none]                       {0}

  inner for {                                         // no token iter_arg
    // partition {1}
    tw = acquire LOOP_BACK                            {1}   [c5,s0]
    W mma0 [tw]                                       {1}   [c5,s0]

    branch = if cond {
      // partition {1}
      W mma1 [tw]                                     {1}   [c2,s1]
      release TO_BRANCH_READ, tw [tc5mma]             {1}   [c2,s1]

      // partition {0}
      tb = acquire TO_BRANCH_READ                     {0}   [c3,s1]
      R branch [tb]                                   {0}   [c3,s1]
      release BACK_TO_WRITER, tb [none]               {0}   [c3,s1]

      // partition {1}
      back = acquire BACK_TO_WRITER                   {1}   [c2,s1]
      yield back                                      {0,1}
    } else {
      // partition {1}
      yield tw                                        {0,1}
    }

    // partition {1}
    release TO_POST_READ, branch [tc5mma]             {1}   [c2,s1]

    // partition {0}
    tf = acquire TO_POST_READ                         {0}   [c4,s1]
    R final [tf]                                      {0}   [c4,s1]
    release LOOP_BACK, tf [none]                      {0}   [c4,s1]
  }

  // partition {1}
  done = acquire LOOP_BACK                            {1}
  release OUTER_EMPTY, done [none]                    {1}   [c2,s1]

  // partition {0}
  out_next = acquire OUTER_EMPTY                      {0}
  R post [out_next]                                   {0}
  next_value = next(value)                            {0,1}

  yield next_value, out_next                          {0,1}
}
```

[↑ Back to contents](#contents)

## Assigning semaphores and counts

The pass first decides where every acquire and release goes. It then gives
matching acquires and releases the same semaphore. An acquire at the start of
a loop, an acquire in the next iteration, and an acquire after the loop may
use the same semaphore when only one runs on a path and each requires the same
arrival count.

For each semaphore, the pass:

1. counts how many arrivals each acquire requires;
2. makes that count the same at every acquire of the semaphore;
3. assigns the semaphore to those acquires and their releases; and
4. records which semaphore carries a token through each `for` or `if`.

The symbolic acquire and release locations do not change after this step.
EMIT-IR renders them in the planned branch and loop scope.

[↑ Back to contents](#contents)

### Releases on the same path or different paths

When two releases always run, one acquire waits for arrivals from both:

```text
one release contribution = arrive_count * number of release kinds
pending_count = sum of all release contributions on the path
```

When releases are in opposite `if` branches, only one runs. Both branches
must contribute the same number of arrivals, and the acquire waits for that
count:

```text
then path contributes N arrivals
else path contributes N arrivals
acquire pending_count = N
```

Most releases have `arrive_count=1` and one release kind, so they contribute
one arrival. A release with two release kinds contributes two arrivals, one
for each kind.

This explains the earlier count example:

```text
release-count loop
  c4 and c5 both execute on one path  -> FULL count 2
```

[↑ Back to contents](#contents)

### One pending count per semaphore

Every acquire of one semaphore uses the same `pending_count`. Sometimes the
first acquire receives all required arrivals from one release, while later
loop iterations receive the same count from several releases. The single
release can use a larger `arrive_count` so that both cases provide the same
total.

A path can be scaled only when it has exactly one release. Every release kind
on that release must be `[none]` or WGMMA; other kinds cannot be scaled. The
semaphore's `pending_count` must be evenly divisible by the number of release
kinds. The pass then sets:

```text
arrive_count = pending_count / number of release kinds
```

The release-count example has:

```text
first acquire    p1 by owner {3}: one [none] release
later iteration  c4 by {1} plus c5 by {0}: two releases

FULL pending_count = 2
p1 arrive_count   = 2
c4 arrive_count   = 1
c5 arrive_count   = 1
```

If the releases cannot provide the same count at every acquire, the pass
reports an error instead of creating a semaphore with inconsistent counts.

[↑ Back to contents](#contents)

### Initial released-stage masks

Some semaphores start released for one owner. The released-stage mask lets the
first acquire of each selected physical stage run before a release executes:

```text
starts released       semaphore.create ... released = -1
starts blocked        semaphore.create ...
```

Every acquire has the semaphore's `pending_count`. A released mask changes
only the initial phase of each stage; it does not partially satisfy a pending
count, and no initial release or arrival operations are emitted.

The legacy entry state releases every physical stage. For a non-circular,
authored multi-member ring whose buffer and semaphore depths already match,
SYNC-DAG can refine that state. It replays the ring's one direct entry acquire
and loop-closing release for one complete stage orbit. For each physical stage,
only its first event matters:

```text
first event is acquire  -> released-mask bit 1
first event is release  -> released-mask bit 0
```

The replay uses the canonical ASP cursor ordinal plus the final authored stage
offset. Full masks remain the legacy `released = -1`; mixed masks are stored on
`Sema::releasedMask`. For example, two advances on a depth-3 ring produce
acquire stages `0,2,1` and release stages `1,0,2`, so the initial mask is
`0b101 = 5`. Two advances on a depth-5 ring similarly produce `0b10101 = 21`.

The refinement deliberately excludes circular groups, auto-expanded
semaphore rings, nested or ambiguous control flow, and entry channels with
multiple acquire/release sites. Those retain the legacy entry state. A later
acquire of any stage waits for new arrivals. The token belongs to the owner
that performs the acquire.

[↑ Back to contents](#contents)

## Buffer and semaphore copies

InsertSemas records a group buffer-copy count, each member's authored copy
count, and a semaphore-copy count. They can differ while this pass builds and
schedules the DAG:

```text
group buffer copies    copies in each backing emitted for the group
member copies          buffer.copy on one member, or one when omitted
semaphore copies       copies of each semaphore
```

The member count records the allocation's memory-plan value. The group count
selects the copies in a backing emitted by InsertSemas. In a valid mixed-copy
SMEM reuse group, one allocation supplies that backing while the other members
retain the logical copy counts authored for their views.
For a local buffer filled by a TMA load, InsertSemas can also record one group
buffer copy while checking the schedule with several semaphore copies.
LowerSemaphore later creates the staged SMEM buffer and semaphore storage.
InsertSemas's `num-stages` option must match the stage count that
LowerSemaphore will actually use.

[↑ Back to contents](#contents)

### Buffer copies

The pass chooses the group buffer-copy count after it removes unnecessary
synchronization edges. Every explicit `buffer.copy` must be positive. Members
of one group must either all provide `buffer.copy` or all omit it.

TMEM members sharing one `buffer.id` must provide the same explicit copy
count. The supported bridge-to-InsertSemas form with different ordinary SMEM
counts is:

- the allocation supplying storage has `buffer.copy=1` and enough storage for
  every member;
- that allocation and each member have the same element type, encoding, and
  memory space;
- its storage is at least `member buffer.copy * member storage size`; and
- each member reusing that storage carries an explicit `buffer.start` in
  `[0, buffer.copy)`.

For that shape, the group buffer-copy count is one. Each member reusing the
storage keeps its own copy count and `buffer.start`, which EMIT-IR uses to
construct its logical view. InsertSemas does not synthesize the
`buffer.start` attribute; it must be present for reuse-view materialization.

```text
start with one buffer copy

if no synchronization edges remain
  emit no semaphore synchronization
  a valid source-free local reuse group can still materialize shared storage

otherwise, if buffer.copy is specified
  use the common count, or one for the valid mixed-copy SMEM shape

otherwise, if this is a TMEM buffer on the default NVWS route
and the two-copy checks pass
  use two buffer copies
```

The edge-free exception applies only to an ordinary local group with at least
two source-free allocations in one block when every member can share one
emitted allocation under the type, offset, copy, and `buffer.start` checks.
EMIT-IR then materializes the backing and member views without creating a
semaphore, acquire, or release. An unrelated single same-owner allocation with
`buffer.copy=2` remains unchanged.

The MetaAWS–NVWS bridge route is selected by `TRITON_NVWS_USE_META=1`. Its
memory planner runs before InsertSemas and writes `buffer.copy`. Automatic
warp specialization then calls InsertSemas with `use-meta-partitioner=true`.
InsertSemas still uses any `buffer.copy` value, but it does not guess two TMEM
copies when that value is absent. A missing value therefore means one copy on
this route.

On the default NVWS route, a synchronized TMEM buffer without `buffer.copy`
can be given two copies automatically. It does not need an MMA user. When it
does have an MMA user, it stays single-copy if any of these are true:

- the MMA reads the old accumulator while writing the new value;
- that MMA and loop do not support two accumulator copies;
- the surrounding WS loop disables accumulator copies;
- two copies would make total TMEM use exceed `128 * 512`; or
- this is a scaled MMA whose accumulator's N dimension is 256.

`test/NVWS/insert_semas_root_entry_tmem.mlir`
`@root_entry_accumulator_uses_native_carried_pou` checks a two-copy
accumulator. The acquire before the loop creates a token that partition `{1}`
uses directly, so no extra semaphore is needed between root code and that
partition. The loop must carry this token from one iteration to the next.
The planner recognizes that the exact root token must remain available and
uses it as the loop's input token. The later MMA in partition `{2}` still
needs its own semaphore.

[↑ Back to contents](#contents)

### Example: a TMEM accumulator gets two copies

The complete access shape of that accumulator is:

```text
W acc root                    initial tmem_store

for {
  R acc {1}
  W acc {1}
  W acc {2}                  tc_gen5_mma
}

R acc root                    final tmem_load
```

Partition `{1}` is the first partition to use the buffer in the loop. It uses
the token created by the acquire before the loop, so no synchronization edge
is needed from root code to partition `{1}`. The exact edges are:

```text
parent DAG node                 synchronization edge ending here
W acc root                     none
[loop summary P0:W:{1}]        none; it uses the root token
R acc root                     p1: loop summary -> R acc root

child DAG node                 synchronization edge ending here
ENTER(i) {1}                   none
R acc {1}                      none
W acc {1}                      none
W MMA {2}                      e1: W acc {1} -> W MMA {2}
EXIT(i) {1}                    e2: W MMA {2} -> EXIT(i) {1}
```

The parent and child are separate DAGs. The parent has no synchronization
edge into the loop; only `p1` leaves the loop summary. The exact root-token
adoption is shown later in the semaphore/token view:

```text
parent synchronization-edge DAG

                         W acc root
                              | walk
                              v
                    [loop summary P0:W:{1}]
                              | p1
                              v
                         R acc root
```

```text
child synchronization-edge DAG

                                                      ENTER(i) {1}
                                                            | walk
                                                            v
                                                      R acc(i) {1}
                                                            | walk
                                                            v
                                                      W acc(i) {1}
                                                            +-----------------e1 >------------------+
                                                            | walk                                  v
                                                            |                                 W MMA(i) {2}
                                                            +-----------------< e2------------------+
                                                            v
                                                       EXIT(i) {1}
```

The semaphores are:

```text
edge / role    semaphore    release owner    pending_count    initial state
entry          EMPTY        -                1                released at root
e2             EMPTY        {2}              1                same semaphore
e1             TO_MMA       {1}              1                blocked
p1             AFTER        {1}              1                blocked
```

The planner carries the token created before the loop because it must remain
available through the loop. The entry view shows token adoption, the two
semaphore paths, and the owner-`{1}` acquire carried through `EXIT`:

```text
          root = acquire EMPTY
               root |
                    v
              W acc [root] root
                    +-------------root token >--------------+
                                                            v
                                                      ENTER(0) {1}
                                                       root |
                                                            v
                                                   R acc(0) [root] {1}
                                                            | walk
                                                            v
                                                   W acc(0) [root] {1}
                                                            | walk
                                                            v
                                               release TO_MMA, root {1} e1
                                                            +---------------TO_MMA >----------------+
                                                            | walk                                  v
                                                            |                           mma = acquire TO_MMA {2}
                                                            |                                   mma |
                                                            |                                       v
                                                            |                              W MMA(0) [mma] {2}
                                                            |                                       | walk
                                                            |                                       v
                                                            |                      release EMPTY, mma [tc5mma] {2} e2
                                                            +----------------< EMPTY----------------+
                                                            v
                                                next = acquire EMPTY {1}
                                                       next |
                                                            v
                                                   EXIT(0) yields next
```

If the loop continues, the same owner-`{1}` token crosses the boundary and is
used by the next iteration's first read and write:

```text
                                                          next
                                                       next |
                                                            v
                                                   EXIT(i) yields next
                                                            | next iteration
                                                            v
                                                ENTER(i+1) receives next
                                                       next |
                                                            v
                                                 R acc(i+1) [next] {1}
                                                            | walk
                                                            v
                                                 W acc(i+1) [next] {1}
                                                            | walk
                                                            v
                                               release TO_MMA, next {1} e1
```

If the loop finishes, `next` becomes `result`. The release to `AFTER`
implements parent edge `p1`:

```text
                                                          next
                                                       next |
                                                            v
                                                EXIT(last) yields result
                                                     result |
                                                            v
                                              release AFTER, result {1} p1
                    +----------------< AFTER----------------+
                    v
        out = acquire AFTER root
                out |
                    v
              R acc [out] root
```

For a zero-trip loop, the continuation view is skipped: `result` is the
original root token and enters the same owner-`{1}` `AFTER` release. This
buffer has synchronization edges, no explicit `buffer.copy`, an MMA shape
that permits two copies, and enough TMEM capacity. The result is:

```text
input accumulator     memdesc<128x128xf32>
generated buffer      memdesc<2x128x128xf32>
physical copies       2
semaphore copies      2
```

Compact output IR:

```text
root_token = acquire EMPTY                   root
W acc [root_token]                           root

result = for iter_arg carry = root_token {
  // partition {1}
  R acc [carry]                              {1}
  W acc [carry]                              {1}
  release TO_MMA, carry                      {1}
  next = acquire EMPTY                       {1}

  // partition {2}
  mma = acquire TO_MMA                       {2}
  W MMA [mma]                                {2}
  release EMPTY, mma [tc5mma]                {2}

  yield next
}

release AFTER, result                        {1}
out = acquire AFTER                          root
R acc [out]                                  root

// zero trip: result = root_token
```

[↑ Back to contents](#contents)

### Semaphore copies

After placing the acquires and releases, the pass chooses how many semaphore
copies to assume while it checks schedules and offsets:

```text
SMEM buffer
and no input buffer.copy
and at least one release includes [tma_load]
  semaphore copies = max(1, num-stages)

otherwise
  semaphore copies = buffer copies
```

For example, `test/NVWS/insert_semas.mlir` `@local_release_after_mma` has one
buffer copy in the InsertSemas DAG. The buffer is filled by
`nvws.descriptor_load`, so the `FULL` release includes `[tma_load]`.
InsertSemas checks its schedule using `max(1, num-stages)`, where `num-stages`
is the option on `--nvws-insert-semas`.

```text
InsertSemas DAG
  buffer copies = 1

schedule model
  semaphore copies = max(1, num-stages)
```

LowerSemaphore uses the owning WS loop's `tt.num_stages` when that attribute
is present. Otherwise it uses the option on `--nvws-lower-semaphore`. If one
physical semaphore group is shared by several TMA-producing WS loops, it uses
the largest of their stage counts. The option on `--nvws-insert-semas` must
match that effective count. LowerSemaphore then creates the staged semaphore
storage and replaces the eligible one-copy SMEM allocation with a matching
staged allocation.

[↑ Back to contents](#contents)

### Example: a TMA load uses the lowering stage count

The buffer with `buffer.id=102` in `@local_release_after_mma` has this input:

```text
for {
  W m0 {0} stage=0       nvws.descriptor_load
  R m0 {1} stage=1       tc_gen5_mma operand
}
```

The same two synchronization edges repeat on every iteration:

```text
edge    source                  destination
e1      W descriptor_load {0}   R MMA operand {1}
e2      R MMA operand {1}       EXIT {0}
```

```text
                      ENTER(i) {0}
                            | walk
                            v
                W descriptor_load(i) {0}
                            +-------------------e1 >--------------------+
                            | walk                                      v
                            |                                 R MMA operand(i) {1}
                            +-------------------< e2--------------------+
                            v
                       EXIT(i) {0}
```

`e1` uses `FULL`. `e2` uses `EMPTY`, which starts released for iteration zero.
Both semaphores have `pending_count=1`:

```text
edge    semaphore    release owner    pending_count    initial state
e1      FULL         {0}              1                blocked
e2      EMPTY        {1}              1                released for owner {0}
```

The two fixed lanes keep each owner's operations vertical. `[cN,sM]` means
cluster `N`, stage `M`. The `EMPTY` rail remains separate from owner `{0}`'s
control spine while control crosses `EXIT` and `ENTER`:

```text
                      ENTER(i) {0}
                            | walk
                            v
          empty = acquire EMPTY(i) {0} [c1,s0]
                      empty |
                            v
       W descriptor_load(i) [empty] {0} [c1,s0]
                            | walk
                            v
      release FULL, empty [tma_load] {0} e1 [c1,s0]
                            +------------------FULL >-------------------+
                            | walk                                      v
                            |                            full = acquire FULL {1} [c0,s1]
                            |                                      full |
                            |                                           v
                            |                          R MMA operand(i) [full] {1} [c0,s1]
                            |                                           | walk
                            |                                           v
                            |                      release EMPTY, full [tc5mma] {1} e2 [c0,s1]
                            |                                     EMPTY |
                            v                                           |
                       EXIT(i) {0}                                      |
                            | next iteration                            |
                            v                                           |
                     ENTER(i+1) {0}                                     |
                            | walk                                      |
                            v                                           |
          next = acquire EMPTY(i+1) {0} [c1,s0] --------< EMPTY --------+
                       next |
                            v
     W descriptor_load(i+1) [next] {0} [c1,s0]
```

Both tracked arrivals are required. `[tma_load]` makes `FULL` track the
descriptor load, so the `FULL` acquire cannot complete until the load fills
SMEM. The next `EMPTY` acquire waits for the `[tc5mma]` arrival posted when the
MMA finishes reading that buffer copy.

SMEM does not receive automatic TMEM double buffering, and the input has no
`buffer.copy`. Therefore the InsertSemas DAG records:

```text
buffer copies              1
semaphore copies used
for schedule checks        max(1, num-stages)
```

LowerSemaphore later creates the matching staged SMEM and semaphore storage.
There is no buffer use after the loop, so no acquire consumes the final
`EMPTY` arrival. A zero-trip loop executes none of the shown operations and
leaves every initially released semaphore copy ready for its first acquire.

Compact output IR:

```text
for {                                        // no async-token iter_arg
  // partition {0}
  empty = acquire EMPTY                      {0} [c1,s0]
  W descriptor_load [empty]                  {0} [c1,s0]
  release FULL, empty [tma_load]             {0} [c1,s0]

  // partition {1}
  full = acquire FULL                        {1} [c0,s1]
  R MMA operand [full]                       {1} [c0,s1]
  release EMPTY, full [tc5mma]               {1} [c0,s1]
}

// zero trip: no acquire or release executes
```

[↑ Back to contents](#contents)

## Scheduling synchronization in a pipelined loop

The pass schedules acquires and releases only after their locations and
semaphores are fixed. Scheduling does not move them to different buffer
accesses.

A scheduled loop uses three values:

```text
loop.stage       stage assigned by the Triton software pipeliner
loop.cluster     order of operations within and across those stages
stage-offset     which buffer and semaphore copy an operation uses
```

The pass keeps every `loop.stage` fixed. It changes `loop.cluster` when a
release must execute before its acquire. It changes `stage-offset` when the
release, acquire, and buffer access must select a different copy.

[↑ Back to contents](#contents)

### Release before acquire

For every release and its matching acquire, the pass knows:

```text
the release's source/completion anchor
the acquire satisfied by that release's arrivals
the loop.stage of both anchors
whether the ordering crosses a loop iteration
```

The required delay between their owners is:

```text
release anchor stage - acquire anchor stage - iteration distance
```

The pass keeps the input stages and adjusts clusters to satisfy these
orderings. If the required orderings form a cycle that cannot execute, it
reports the cycle as an error. The pass does not add another ordering rule when
existing data flow already orders the operations.

[↑ Back to contents](#contents)

### Synchronization between iterations

A release can post arrivals for an acquire in the same iteration or in a later
iteration. When this is already known, the ordering records that positive
iteration distance. Otherwise the pass follows the physical buffer and
semaphore copies used by successive iterations.

The pass replays the ordered buffer accesses and uses the semaphore-copy
count. With one semaphore copy, the distance is one iteration. With several,
it finds the first later iteration whose acquire returns to the release's
semaphore copy.

If the pass cannot find a later iteration that uses the matching copy, it
reports an error.

[↑ Back to contents](#contents)

### Selecting the matching copy

Circular buffers can have several names with one `buffer.id` and one shared
set of physical copies. The pass visits their reads and writes in program
order, tracks which write advances the shared copy number, and assigns an
offset to each buffer access, acquire, and release.

It also checks that every input `buffer.start` agrees with this write order
and that no read selects a copy before a write has produced it.

That circular meaning applies only when `buffer.circular` is present. On an
ordinary planned-reuse SMEM member, `buffer.start` selects the member's logical
slot within the reused allocation's storage. That selection is fixed while
materializing the member view; it is not a semaphore stage offset and does not
participate in the circular write-order check.

`test/NVWS/insert_semas_circular_smem.mlir`
`@circular_tutorial_1_1_to_2_2` checks the circular two-copy case. K and V are
analyzed separately but share physical storage. Each access, acquire, and
release receives the offset of the copy it uses.

A non-circular alias is another name or view of the same physical buffer. If
that buffer has several copies, the operations on both sides of a
synchronization edge must be directly inside the same loop body. The pass
selects a release offset that uses the same copy as the acquire, including for
an edge between iterations.

`test/NVWS/insert_semas_fused_alias_handoff.mlir` `@fused_alias_depth_two`
checks this case. Its semaphores contain both buffer names, and each
`nvws.semaphore.buffer` returns both views. The copy offset belongs to the
acquire or release; the buffer view does not choose a different copy.

[↑ Back to contents](#contents)

### Example: circular K and V select different copies

`test/NVWS/insert_semas_circular_smem.mlir`
`@circular_tutorial_1_1_to_2_2` has K and V sharing one circular buffer with
two physical copies:

```text
K = local_alloc {buffer.id=301, buffer.copy=2,
                 buffer.circular, buffer.start=0}
V = local_alloc {buffer.id=301, buffer.copy=2,
                 buffer.circular, buffer.start=1}

for {
  W K {1}
  W V {1}
  R K {2}
  R V {2}
}
```

K and V are two logical groups during synchronization analysis. Their edge
DAGs are separate views, not parallel execution. The exact edges are:

```text
group  DAG node          synchronization edge ending here
K      ENTER(i) {1}      none
K      W K(i) {1}        none
K      R K(i) {2}        k1: W K(i) {1} -> R K(i) {2}
K      EXIT(i) {1}       k2: R K(i) {2} -> EXIT(i) {1}

V      ENTER(i) {1}      none
V      W V(i) {1}        none
V      R V(i) {2}        v1: W V(i) {1} -> R V(i) {2}
V      EXIT(i) {1}       v2: R V(i) {2} -> EXIT(i) {1}
```

```text
K synchronization-edge DAG

                      ENTER(i) {1}
                            | walk
                            v
                       W K(i) {1}
                            +-------------------k1 >--------------------+
                            | walk                                      v
                            |                                      R K(i) {2}
                            +-------------------< k2--------------------+
                            v
                       EXIT(i) {1}

V synchronization-edge DAG

                      ENTER(i) {1}
                            | walk
                            v
                       W V(i) {1}
                            +-------------------v1 >--------------------+
                            | walk                                      v
                            |                                      R V(i) {2}
                            +-------------------< v2--------------------+
                            v
                       EXIT(i) {1}
```

No synchronization edge is removed or merged within either logical group.
The loop still executes K before V in each owner. The pass folds the two
logical groups onto one physical `FULL` semaphore and one physical `EMPTY`
semaphore because they share one circular `buffer.id`:

```text
edge    semaphore    release owner    pending_count    initial state
k1      FULL         {1}              1                blocked
k2      EMPTY        {2}              1                released for owner {1}
v1      FULL         {1}              1                blocked
v2      EMPTY        {2}              1                released for owner {1}
```

The loop still executes the accesses in input order: K before V in each owner.
The synchronization views stay separate, as they do in the edge DAGs above.
Both views use the same physical `FULL`/`EMPTY` pair. Their two external
`EMPTY` rails select different stages of that pair; they are not different
semaphores.

```text
K semaphore view

                      ENTER(i) {1}
                            | walk
                            v
                 kt = acquire EMPTY {1}
                         kt |
                            v
                     W K(i) [kt] {1}
                            | walk
                            v
                 release FULL, kt {1} k1
                            +------------------ FULL > -----------------+
                       walk |                                           v
                            |                                kr = acquire FULL {2}
                            |                                         kr |
                            |                                            v
                            |                                    R K(i) [kr] {2}
                            |                                            | walk
                            |                                            v
                            |                          release EMPTY, kr {2} k2 -------- EMPTY (K stage) > ------------+
                            |                                                                                          |
                            v                                                                                          |
                       EXIT(i) {1}                                                                                     |
                            | next iteration                                                                           |
                            v                                                                                          |
                     ENTER(i+1) {1}                                                                                    |
                            | walk                                                                                     |
                            v                                                                                          |
                nextK = acquire EMPTY {1} -------------------- EMPTY (K stage) < --------------------------------------+
                      nextK |
                            v
                   W K(i+1) [nextK] {1}

V semaphore view

                      ENTER(i) {1}
                            | walk
                            v
                 vt = acquire EMPTY {1}
                         vt |
                            v
                     W V(i) [vt] {1}
                            | walk
                            v
                 release FULL, vt {1} v1
                            +------------------ FULL > -----------------+
                       walk |                                           v
                            |                                vr = acquire FULL {2}
                            |                                         vr |
                            |                                            v
                            |                                    R V(i) [vr] {2}
                            |                                            | walk
                            |                                            v
                            |                          release EMPTY, vr {2} v2 -------- EMPTY (V stage) > ------------+
                            |                                                                                          |
                            v                                                                                          |
                       EXIT(i) {1}                                                                                     |
                            | next iteration                                                                           |
                            v                                                                                          |
                     ENTER(i+1) {1}                                                                                    |
                            | walk                                                                                     |
                            v                                                                                          |
                nextV = acquire EMPTY {1} -------------------- EMPTY (V stage) < --------------------------------------+
                      nextV |
                            v
                   W V(i+1) [nextV] {1}
```

Each write therefore acquires `EMPTY` immediately before it, and the loop
does not carry a semaphore token.

The two buffers share one write number. Each write advances it:

```text
event       current write number    required write number    offset
W K         -1 -> 0                 K producer = 0           0
W V          0 -> 1                 V producer = 1           0
R K          1                      K producer = 0          -1
R V          1                      V producer = 1           0
```

Adding offsets to the same two-view scaffold gives:

```text
K semaphore view with offsets

                      ENTER(i) {1}
                            | walk
                            v
            kt = acquire EMPTY {1} [offset 0]
                         kt |
                            v
                 W K(i) [kt] {1} [copy 0]
                            | walk
                            v
           release FULL, kt {1} k1 [offset 0]
                            +-------------- FULL (copy 0) > ------------+
                       walk |                                           v
                            |                     kr = acquire FULL {2} [offset -1]
                            |                                         kr |
                            |                                            v
                            |                            R K(i) [kr] {2} [copy 0]
                            |                                            | walk
                            |                                            v
                            |        release EMPTY, kr {2} k2 [offset -1] -------- EMPTY (copy 0) > -------------------+
                            |                                                                                          |
                            v                                                                                          |
                       EXIT(i) {1}                                                                                     |
                            | next iteration                                                                           |
                            v                                                                                          |
                     ENTER(i+1) {1}                                                                                    |
                            | walk                                                                                     |
                            v                                                                                          |
     nextK = acquire EMPTY {1} [offset 0] -------------------- EMPTY (copy 0) < ---------------------------------------+
                      nextK |
                            v
            W K(i+1) [nextK] {1} [copy 0]

V semaphore view with offsets

                      ENTER(i) {1}
                            | walk
                            v
            vt = acquire EMPTY {1} [offset 0]
                         vt |
                            v
                 W V(i) [vt] {1} [copy 1]
                            | walk
                            v
           release FULL, vt {1} v1 [offset 0]
                            +-------------- FULL (copy 1) > ------------+
                       walk |                                           v
                            |                      vr = acquire FULL {2} [offset 0]
                            |                                         vr |
                            |                                            v
                            |                            R V(i) [vr] {2} [copy 1]
                            |                                            | walk
                            |                                            v
                            |         release EMPTY, vr {2} v2 [offset 0] -------- EMPTY (copy 1) > -------------------+
                            |                                                                                          |
                            v                                                                                          |
                       EXIT(i) {1}                                                                                     |
                            | next iteration                                                                           |
                            v                                                                                          |
                     ENTER(i+1) {1}                                                                                    |
                            | walk                                                                                     |
                            v                                                                                          |
     nextV = acquire EMPTY {1} [offset 0] -------------------- EMPTY (copy 1) < ---------------------------------------+
                      nextV |
                            v
            W V(i+1) [nextV] {1} [copy 1]
```

The offsets select these physical copies relative to the latest shared write:

```text
operation                 stage offset       physical copy
W K / release FULL        0                  copy 0
W V / release FULL        0                  copy 1
acquire FULL / R K       -1                  copy 0
acquire FULL / R V        0                  copy 1
R K / release EMPTY      -1                  copy 0
R V / release EMPTY       0                  copy 1
```

The `-1` means the previous copy after V advanced the shared write number. It
is not a negative physical index. With two copies, wrapping `-1` selects copy
0. The generated DAG gives K's acquire and closing release `stage-offset=-1`
and V's offset zero. The test also checks that the IR contains the one shared
pair of physical semaphores.

For iteration zero, the initially released state of each physical `EMPTY`
copy supplies `kt` and `vt`. On re-entry, `k2` supplies the next K acquire and
`v2` supplies the next V acquire. After the final iteration, the arrivals from
their final releases have no later acquires. A zero-trip loop executes none of
these operations and leaves both physical `EMPTY` copies initially released.

Compact output IR:

```text
for {                                        // no async-token iter_arg
  // partition {1}
  kt = acquire EMPTY [offset 0]              {1}
  W K [kt] [copy 0]                          {1}
  release FULL, kt [offset 0]                {1}

  vt = acquire EMPTY [offset 0]              {1}
  W V [vt] [copy 1]                          {1}
  release FULL, vt [offset 0]                {1}

  // partition {2}
  kr = acquire FULL [offset -1]              {2}
  R K [kr] [copy 0]                          {2}
  release EMPTY, kr [offset -1]              {2}

  vr = acquire FULL [offset 0]               {2}
  R V [vr] [copy 1]                          {2}
  release EMPTY, vr [offset 0]               {2}
}

// zero trip: no acquire or release executes
```

[↑ Back to contents](#contents)

### Example: a non-circular alias advances the copy

`test/NVWS/insert_semas_fused_alias_handoff.mlir`
`@fused_alias_depth_two` uses two names for one two-copy SMEM buffer:

```text
m0 = local_alloc {buffer.id=500, buffer.copy=2}
m1 = local_alloc {buffer.id=500, buffer.copy=2}

for {
  W m0 {4}
  R m0 {2}
  W m1 {4}
  R m1 {2}
}
```

Both names refer to the same bytes, so the synchronization-edge DAG is one
ordered path:

```text
edge    source        destination
e1      W m0 {4}      R m0 {2}
e2      R m0 {2}      W m1 {4}
e3      W m1 {4}      R m1 {2}
e4      R m1 {2}      EXIT {4}
```

```text
                      ENTER(i) {4}
                            | walk
                            v
                       W m0(i) {4}
                            +-------------------e1 >--------------------+
                            | walk                                      v
                            |                                      R m0(i) {2}
                            +-------------------< e2--------------------+
                            v
                       W m1(i) {4}
                            +-------------------e3 >--------------------+
                            | walk                                      v
                            |                                      R m1(i) {2}
                            +-------------------< e4--------------------+
                            v
                       EXIT(i) {4}
```

No edge is removed or merged.

Every semaphore has `pending_count=1`; `ENTRY` starts released:

```text
edge    semaphore    release owner    pending_count    initial state
e1      M0_FULL      {4}              1                blocked
e2      M1_READY     {2}              1                blocked
e3      M1_FULL      {4}              1                blocked
e4      ENTRY        {2}              1                released for owner {4}
```

The POU semaphore DAG for iteration `i` is:

```text
                      ENTER(i) {4}
                            | walk
                            v
                t0 = acquire ENTRY(i) {4}
                         t0 |
                            v
                    W m0(i) [t0] {4}
                            | walk
                            v
               release M0_FULL, t0 {4} e1
                            +-----------------M0_FULL >-----------------+
                            | walk                                      v
                            |                               t1 = acquire M0_FULL {2}
                            |                                        t1 |
                            |                                           v
                            |                                   R m0(i) [t1] {2}
                            |                                           | walk
                            |                                           v
                            |                              release M1_READY, t1 {2} e2
                            +----------------< M1_READY-----------------+
                            v
                t2 = acquire M1_READY {4}
                         t2 |
                            v
                    W m1(i) [t2] {4}
                            | walk
                            v
               release M1_FULL, t2 {4} e3
                            +-----------------M1_FULL >-----------------+
                            | walk                                      v
                            |                               t3 = acquire M1_FULL {2}
                            |                                        t3 |
                            |                                           v
                            |                                   R m1(i) [t3] {2}
                            |                                           | walk
                            |                                           v
                            |                               release ENTRY, t3 {2} e4
                            |                                     ENTRY |
                            v                                           |
                       EXIT(i) {4}                                      |
                            | next iteration                            |
                            v                                           |
                     ENTER(i+1) {4}                                     |
                            | walk                                      |
                            v                                           |
              next = acquire ENTRY(i+1) {4} ----------< ENTRY ----------+
                       next |
                            v
                  W m0(i+1) [next] {4}
```

The first write/read pair uses copy `s`; the second uses `(s+1) mod 2`:

```text
synchronization edge       release offset    acquire offset
W m0 -> R m0               0                 0
R m0 -> W m1              +1                 0
W m1 -> R m1               0                 0
R m1 -> W m0(i+1)         +1                 0
```

The generated DAG therefore places `stage-offset=1` on the `M1_READY` and
`ENTRY` releases. The full offset overlay uses the same two-lane scaffold:

```text
                      ENTER(i) {4}
                            | walk
                            v
            t0 = acquire ENTRY {4} [offset 0]
                         t0 |
                            v
                    W m0(i) [t0] {4} [copy s]
                            | walk
                            v
          release M0_FULL, t0 {4} e1 [offset 0]
                            +-----------------M0_FULL >-----------------+
                            | walk                                      v
                            |                          t1 = acquire M0_FULL {2} [offset 0]
                            |                                        t1 |
                            |                                           v
                            |                                   R m0(i) [t1] {2} [copy s]
                            |                                           | walk
                            |                                           v
                            |                        release M1_READY, t1 {2} e2 [offset +1]
                            +----------------< M1_READY-----------------+
                            v
          t2 = acquire M1_READY {4} [offset 0]
                         t2 |
                            v
                    W m1(i) [t2] {4} [copy s+1]
                            | walk
                            v
          release M1_FULL, t2 {4} e3 [offset 0]
                            +-----------------M1_FULL >-----------------+
                            | walk                                      v
                            |                          t3 = acquire M1_FULL {2} [offset 0]
                            |                                        t3 |
                            |                                           v
                            |                                   R m1(i) [t3] {2} [copy s+1]
                            |                                           | walk
                            |                                           v
                            |                         release ENTRY, t3 {2} e4 [offset +1]
                            |                                     ENTRY |
                            v                                           |
                       EXIT(i) {4}                                      |
                            | next iteration                            |
                            v                                           |
                     ENTER(i+1) {4}                                     |
                            | walk                                      |
                            v                                           |
           next = acquire ENTRY {4} [offset 0] --------< ENTRY ---------+
                       next |
                            v
                  W m0(i+1) [next] {4} [copy s]
```

Without the two `+1` release offsets, the following acquire would wait on a
different physical semaphore copy. For iteration zero, `ENTRY` is initially
released and supplies `t0`. On re-entry, `e4` supplies `next`. After the final
iteration, no later acquire consumes the final `ENTRY` arrival. A zero-trip
loop executes none of these operations and leaves `ENTRY` initially released.

Compact output IR:

```text
for {                                        // no async-token iter_arg
  // partition {4}
  t0 = acquire ENTRY [offset 0]              {4}
  W m0 [t0] [copy s]                         {4}
  release M0_FULL, t0 [offset 0]             {4}

  t2 = acquire M1_READY [offset 0]           {4}
  W m1 [t2] [copy s+1]                       {4}
  release M1_FULL, t2 [offset 0]             {4}

  // partition {2}
  t1 = acquire M0_FULL [offset 0]            {2}
  R m0 [t1] [copy s]                         {2}
  release M1_READY, t1 [offset +1]           {2}

  t3 = acquire M1_FULL [offset 0]            {2}
  R m1 [t3] [copy s+1]                       {2}
  release ENTRY, t3 [offset +1]              {2}
}

// zero trip: no acquire or release executes
```

[↑ Back to contents](#contents)

### Example: one buffer copy

`test/NVWS/insert_semas_recurrence_schedule.mlir`
`@one_slot_recurrence` has one SMEM copy and this scheduled loop:

```text
buffer.copy = 1

for {
  W buffer {3}      loop.stage 0
  R first {1}       loop.stage 0
  R last {1}        loop.stage 1
}
```

The two reads have the same owner. The final read replaces that owner's latest
access without adding another synchronization edge. The exact edges are:

```text
DAG node                  synchronization edge ending here
ENTER(i) {3}              none
W buffer(i) {3}           none
R first(i) {1}            e1: W buffer(i) {3} -> R first(i) {1}
R last(i) {1}             none; same-owner program order
EXIT(i) {3}               e2: R last(i) {1} -> EXIT(i) {3}
```

```text
                      ENTER(i) {3}
                            | walk
                            v
                     W buffer(i) {3}
                            +-------------------e1 >--------------------+
                            | walk                                      v
                            |                                    R first(i) {1}
                            |                                           | walk
                            |                                           v
                            |                                     R last(i) {1}
                            +-------------------< e2--------------------+
                            v
                       EXIT(i) {3}
```

No edge is removed or merged.

The schedule can overlap the final read of iteration `i` with work from
iteration `i+1`. The `EMPTY` acquire is immediately before the next store,
and the matching `EMPTY` release is after the final read. Although edge `e2`
ends at `EXIT(i)`, POU places its acquire at `W buffer(i+1)`. With one
semaphore copy, that edge has distance one iteration.

The semaphore assignment is:

```text
edge / role    semaphore    release owner    pending_count    initial state
e1             FULL         {3}              1                blocked
entry          EMPTY        -                1                released for owner {3}
e2             EMPTY        {1}              1                same semaphore
```

The generated semaphore DAG, including the synchronization into iteration `i+1`, is
below. `[cN,sM]` means cluster `N`, stage `M`:

```text
                      ENTER(i) {3}
                            | walk
                            v
          empty = acquire EMPTY(i) {3} [c3,s0]
                      empty |
                            v
                W buffer(i) [empty] {3} [c3,s0]
                            | walk
                            v
        release FULL, empty [none] {3} e1 [c3,s0]
                            +------------------FULL >-------------------+
                            | walk                                      v
                            |                            full = acquire FULL {1} [c3,s0]
                            |                                      full |
                            |                                           v
                            |                                R first(i) [full] {1} [c3,s0]
                            |                                           | walk
                            |                                           v
                            |                                 R last(i) [full] {1} [c2,s1]
                            |                                           | walk
                            |                                           v
                            |                       release EMPTY, full [none] {1} e2 [c2,s1]
                            |                                     EMPTY |
                            v                                           |
                       EXIT(i) {3}                                      |
                            | next iteration                            |
                            v                                           |
                     ENTER(i+1) {3}                                     |
                            | walk                                      |
                            v                                           |
          next = acquire EMPTY(i+1) {3} [c3,s0] --------< EMPTY --------+
                       next |
                            v
              W buffer(i+1) [next] {3} [c3,s0]
```

The loop does not carry a semaphore token. Each iteration acquires `EMPTY`
immediately before the store. `EMPTY` starts released for iteration zero. On
re-entry, partition `{1}` posts the arrival consumed by the next `EMPTY`
acquire. After the final iteration, no later acquire consumes that arrival. A
zero-trip loop executes none of these operations and leaves `EMPTY` initially
released.

The pass adjusts `loop.cluster` so the next store and its first reader are
ordered after the final read. The test checks those clusters and the final
acquire and release operations.

After the ordering and copy offsets are fixed, each release uses the schedule
of its source/completion anchor. An acquire uses the schedule of the operation
that next needs its token. For an acquire used by the next iteration, the pass
also accounts for the loop boundary.

Compact output IR:

```text
for {                                        // no async-token iter_arg
  // partition {3}
  empty = acquire EMPTY                      {3} [c3,s0]
  W buffer [empty]                           {3} [c3,s0]
  release FULL, empty [none]                 {3} [c3,s0]

  // partition {1}
  full = acquire FULL                        {1} [c3,s0]
  R first [full]                             {1} [c3,s0]
  R last [full]                              {1} [c2,s1]
  release EMPTY, full [none]                 {1} [c2,s1]
}

// zero trip: no acquire or release executes
```

[↑ Back to contents](#contents)

## Checks before changing IR

The pass checks the complete acquire/release DAG before it changes the input
IR.

While choosing placements, the planner uses these rules:

- carry a token rather than lose an exact token needed after a `for` or `if`;

For the complete plan, it checks that:

- every token leaving a `for` or `if` can be traced to an incoming token or an
  acquire inside that region;
- every buffer access has a token owned by the correct owner;
- every release has a token, one or more release kinds, a positive arrival
  count, and exactly one matching acquire;
- each matching release and acquire use the same semaphore;
- every acquire has the semaphore's positive `pending_count` and either is
  supplied by release arrivals or uses a semaphore that starts released;
- an acquire repeated inside a loop has release arrivals that let the next
  iteration continue;
- no semaphore is acquired by two distinct partition owners (root is excluded
  from this check, so root and one partition owner may acquire the same
  semaphore);
- every synchronization edge between iterations has a positive distance;
- every path through a `for` or `if` returns the token needed after that
  path; and
- every synchronization edge was consumed by the acquire/release DAG.

After emitting IR, the pass checks that every token and buffer view is used
inside the region where it exists. Within one block, a later non-exempt
`nvws.semaphore.buffer` materialization after a token release is rejected.
Fan-out through multiple releases is allowed, as are exact-view reuse cases
recorded by the emitter, including cached-view use after release under its
reuse contracts. See [EMIT-IR](emit-ir.md) for those checks.

[↑ Back to contents](#contents)

## Build order and code map

This final section names C++ functions and types for readers changing the
implementation. The earlier sections do not require these names.

The function driver in `InsertSemas.cpp` builds one plan as follows:

```text
validateNVWSManagedAllocationLocality
  -> for each top-level function block
       collectGroups(func, block)
       buildAccessDag(group, func, block) for every block-local group
       append groups to the function candidate list
  -> buildSyncDag for every group
       ChainWalker
       reduceEdges
       computeBackingCopies
       DirectBuilder::run
       computeRequiredParts
       computeSemaphoreCopies
       validateTokenConnectivity
       reject synchronized tt.descriptor-fed local_alloc members
       verifySyncDag
  -> finalizeSyncSchedule across all groups
       planPhysicalResources
  -> emitIR
```

A synchronized group is unsupported when one of its `ttg.local_alloc` members
is still sourced directly by `tt.descriptor_load` or `tt.descriptor_gather`.
Upstream allocation materialization must convert that input before
`nvws-insert-semas`; the NVWS allocation path does so in
`nvws-insert-allocas`. `buildSyncDag` diagnoses the violated pass-order invariant
before `verifySyncDag`.

Source map:

| Responsibility | Implementation |
| --- | --- |
| Build, schedule, and emit driver | `InsertSemas.cpp` |
| Function-CFG locality contract | `validateNVWSManagedAllocationLocality` in `MetaToNVWSConvert.cpp` |
| Shared `Node`, `RegionFlow`, `Sema`, `GroupDag` model | `InsertSemas.h` |
| Groups, pieces, owners, accesses, region summaries | `InsertSemasAccessDag.cpp` |
| Initial synchronization edges | `ChainWalker`, `applyTouch` in `InsertSemasSyncDag.cpp` |
| Remove unnecessary synchronization edges | `reduceStraightEdges`, `reduceLoopCloses`, `reduceEdges` |
| Place acquires and releases | `DirectBuilder` |
| Assign semaphores and counts | `DirectBuilder::formSemaphores` |
| TT descriptor-fed local pass-order invariant | `buildSyncDag` |
| Checks before changing IR | `validateTokenConnectivity`, `verifySyncDag` |
| Copies and schedule | `computeBackingCopies`, `computeSemaphoreCopies`, `finalizeSyncSchedule` |
| Select which member's emitted allocation supplies another member's storage | `planPhysicalResources`, `Member::backingPrimary` |
| DAG dump and IR emission | `InsertSemasEmitIR.cpp` |

The design can be summarized in one sentence:

> Find synchronization edges, remove edges whose ordering other paths already
> guarantee, place the acquires and releases, assign semaphores, check the
> result, then emit IR.

[↑ Back to contents](#contents)
