# Ping-Pong Scheduling

Ping-pong scheduling enforces mutual exclusion around "expensive" GPU
operations across warp partitions. When two consumer partitions both execute
expensive ops on shared hardware resources (tensor cores on Hopper, SFU on
Blackwell), they alternate execution via named barrier synchronization rather
than competing simultaneously.

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
| Blackwell (SM100) | `math::ExpOp`, `math::Exp2Op` (rank > 1 tensors only) | SFU bottleneck for large tensors |

Expensive ops are further classified as:
- **NonReorderable** (e.g., `WarpGroupDotOp`): has memory effects, so the
  critical region boundary is the op itself.
- **PureArithmetic** (e.g., `math::ExpOp`): memory-effect-free, so the
  boundary extends forward to the next op with memory effects.

## Named Barrier Allocation

Named barriers use indices **7 through 15** (indices 0-6 are reserved for
producer-consumer mbarriers and warp group sync). Each ping-pong region
consumes **two** barrier indices — one for "ping" and one for "pong".

Maximum concurrent ping-pong regions: **(15 - 7 + 1) / 2 = 4** (pairs
`{7,8}`, `{9,10}`, `{11,12}`, `{13,14}`). If barriers are exhausted, the
region is silently skipped.

## `doPingPongPrep` Algorithm

### Step 1: Group Expensive Ops

Walk the function and group expensive ops. An op joins an existing group if:

1. **Same operation type** as all ops in the group.
2. **Same control flow context**: same block, no intervening `scf::ForOp` /
   `scf::IfOp` / `scf::WhileOp`.
3. **No intervening memory effects** between ops in the same partition.

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

Scan partition regions for ops with `pingpong_id` attributes. Allocate a barrier
pair for each region.

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

The concrete ops inserted are `NamedBarrierArriveOp` and `NamedBarrierWaitOp`,
with the thread count set to `(numWarps_ping + numWarps_pong) * 32`.
