# Ping-Pong Scheduling

Ping-pong scheduling enforces mutual exclusion around "expensive" GPU
operations across warp partitions. When two consumer partitions both execute
expensive ops on shared hardware resources (tensor cores on Hopper, SFU on
Blackwell), they alternate execution via mbarrier synchronization rather than
competing simultaneously.

## Pipeline Integration

Both passes are gated by the `pingpongAutoWS` option (`--pingpong-auto-ws`).
See [Overview.md](Overview.md) for the full pipeline and Hopper/Blackwell
differences.

`doPingPongPrep` runs **before** code partitioning (ops still have
`async_task_id` but are not physically separated). `doPingPongSync` runs
**after** code partitioning (ops are inside `WarpSpecializeOp` regions).

**File**: `PingPong.cpp`

## Expensive Op Identification

Identification is architecture-dependent (`CriticalRegionManager::isExpensiveOp`):

| Architecture | Expensive Ops | Rationale |
|-------------|--------------|-----------|
| Hopper (SM90) | `WarpGroupDotOp` (wgmma) | Shared tensor core resources |
| Blackwell (SM100) | `math::{Exp,Exp2,Sin,Cos,Tanh,Sqrt,Rsqrt}Op`, pure `tt.elementwise_inline_asm` containing an SFU `.approx` mnemonic (`sin`/`cos`/`ex2`/`lg2`/`tanh`/`rcp`/`rsqrt`/`sqrt`/`div`) and no barrier/memory/control/collective ops (rank > 1 tensors only) | SFU bottleneck for large tensors |

Expensive ops are further classified as:
- **NonReorderable** (e.g., `WarpGroupDotOp`): has memory effects, so the
  critical region boundary is the op itself. This self-effect (e.g. a wgmma's
  SMEM-operand read) marks the op's own boundary; it must **not** be treated as
  an intervening memory effect when deciding whether two expensive ops can be
  grouped (see Step 1), otherwise consecutive WGMMAs never group.
- **PureArithmetic** (e.g., `math::ExpOp`, pure SFU `.approx` inline
  asm): memory-effect-free, so the boundary extends forward to the next op
  with memory effects.

## Barrier Allocation

Each ping-pong region allocates a two-element mbarrier group in shared memory.
Both barriers are initialized with one arrival credit, threaded into the two
warp-specialization partitions, invalidated, and deallocated after the
specialized region. Wait phases are derived from the normalized, linearized
iteration count of the enclosing loops.

The disabled-by-default mbarrier-to-named-barrier pass may later promote the
whole pair when two compiler IDs are available. If promotion is disabled or
the pool is exhausted, PingPong remains correct using the original mbarriers.

## `doPingPongPrep` Algorithm

### Step 1: Group Expensive Ops

Walk the function and group expensive ops. An op joins an existing group if:

1. **Same operation type** as all ops in the group.
2. **Same control flow context**: same block, no intervening `scf::ForOp` /
   `scf::IfOp` / `scf::WhileOp`.
3. **No intervening memory effects** between ops in the same partition. This is
   evaluated *strictly between* the two ops: the **endpoint ops are excluded**,
   and any **peer expensive ops in between are skipped** (they belong to the
   same ping-pong region, so they do not split it). Only a non-expensive op
   with memory side effects rejects grouping. Implemented by
   `hasInterveningMemEffect`, which is distinct from `findEndOp` (the latter
   finds a region's *end* boundary and, for NonReorderable ops, returns the op
   itself).

If no group matches, a new group is created.

### Step 2: Validate and Assign `pingpong_id`

For each group:

1. Categorize ops by partition. Require **exactly 2 partitions** — ping-pong
   only applies with two consumer partitions sharing the same expensive op type.
2. Require a parent `scf::ForOp` — ping-pong needs iteration.
3. Validate schedule alternation via `arrivesFirst()`: the two partitions' ops
   must alternate cleanly in the linearized schedule:
   ```
   [partition A ops] [partition B ops] [partition A ops] [partition B ops] ...
   ```
   If ops interleave within a "round," the group is skipped.
4. Set attributes: `pingpong_id` (region identifier) and
   `pingpong_first_partition_id` (which partition's ops appear first).

## `doPingPongSync` Algorithm

After code partitioning, walk `WarpSpecializeOp` regions and insert barriers.

### Step 1: Discover Regions

Scan partition regions for ops with `pingpong_id` attributes. Allocate and
initialize an mbarrier pair for each region.

### Step 2: Compute Boundaries

For each partition in a ping-pong region:
- **Start**: the expensive op itself.
- **End**: the first subsequent op with memory side effects (found by
  `findEndOp`). If the expensive op itself has memory effects (NonReorderable),
  the end is the op itself.

Multiple expensive ops in the same partition are unioned — start is the earliest,
end is the latest.

### Step 3: Insert Barriers

The partition that executes first (from `pingpong_first_partition_id`) is the
**pong** partition. The other is **ping**.

```
Ping partition:                      Pong partition:
─────────────────────                ─────────────────────
arrive(pongBarrier)  ─────────┐
  ...                         │
                              ├───>  wait(pongBarrier)
                              │      [expensive ops]
wait(pingBarrier)  <──────────┤      arrive(pingBarrier)
[expensive ops]               │        ...
arrive(pongBarrier)  ─────────┤
  ...                         │
                              ├───>  wait(pongBarrier)
                              │      [expensive ops]
wait(pingBarrier)  <──────────┤      arrive(pingBarrier)
[expensive ops]               │        ...
arrive(pongBarrier)  ─────────┘
  ...
```

**Why the initial arrive at ping's region entry**: The ping partition issues an
initial `arrive(pongBarrier)` before entering the loop body. This primes the
pump — it allows the pong partition's first `wait(pongBarrier)` to proceed
immediately, since pong goes first by definition. Without this, pong would
deadlock on the first iteration.

The concrete ops inserted are `ArriveBarrierOp` and `WaitBarrierOp`. Each
arrival is a leader arrival with count one. When promotion is enabled, the pair
is rewritten atomically to `NamedBarrierArriveOp` and `NamedBarrierWaitOp`, with
the participant count derived from both partitions.
