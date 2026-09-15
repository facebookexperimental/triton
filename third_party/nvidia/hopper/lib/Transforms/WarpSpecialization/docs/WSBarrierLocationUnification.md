# WS Barrier Location Unification

`triton-nvidia-unify-ws-barrier-locations` co-locates related AutoWS wait
barriers and orders their TMEM and streamable SMEM operands. The initial use
cases are an epilogue that combines a broadcast bias with an MMA accumulator
and the corresponding Flash Attention softmax chain.

The pass runs immediately after `triton-nvidia-interleave-tmem`. TMEM
interleaving first chooses load locations for latency and register liveness;
barrier unification first exposes one scheduling region, then puts the TMEM
load before the SMEM broadcast chain without moving either wait again.

## Motivation

After TMEM interleaving, a one-dimensional-bias `addmm` epilogue can have this
shape:

```mlir
ttng.wait_barrier %bias_ready, %bias_phase
%bias = ttg.local_load %bias_smem
ttng.arrive_barrier %bias_empty, 1
%bias_f32 = arith.extf %bias
%bias_layout = ttg.convert_layout %bias_f32
%bias_tile = tt.broadcast %bias_layout
ttng.wait_barrier %acc_ready, %acc_phase
%acc = ttng.tmem_load %accumulator
ttng.arrive_barrier %acc_empty, 1
%out = arith.addf %acc, %bias_tile
```

There is no substantive independent work between the waits. Keeping them
separate can prevent ptxas from scheduling the broadcast and accumulator add
as one region, while extending the lifetime of the broadcast value. The pass
raises the later wait next to the earlier wait, then orders the TMEM operand
first:

```mlir
ttng.wait_barrier %bias_ready, %bias_phase
ttng.wait_barrier %acc_ready, %acc_phase
%acc = ttng.tmem_load %accumulator
ttng.arrive_barrier %acc_empty, 1
%bias = ttg.local_load %bias_smem
...
%bias_tile = tt.broadcast %bias_layout
%out = arith.addf %acc, %bias_tile
```

Both waits remain distinct. Their operands, phases, predicates, and constraints
are unchanged. The operand reorder moves the SMEM release with its load and
moves no acquire after the waits have been unified.

### The Trade

This is not a free win, and the two sides of it do not scale together.

What is gained is register relief, and it is proportional to the size of the
broadcast value. In the original order `%bias_tile` is materialized before
`%acc_ready` is awaited, so it stays live across that wait; afterwards it is
produced only once the MMA has landed and lives just as far as the `addf`. On
the measured `addmm` epilogue that value is a `128x128xf32` bias tile at 128
elements per thread, and removing it from the wait's live range is what
eliminates the spill.

What is lost is overlap, and that cost is the same at any size. Before the
move, `local_load` → `extf` → `convert_layout` → `broadcast` runs while the
MMA is still in flight. After it, the consumer blocks on the accumulator
before doing any bias preparation at all, so that chain no longer hides MMA
latency. Once the waits are co-located, the subsequent operand reorder does not
further extend the asynchronous MMA interval: it changes only which ready
operand is materialized first.

For a large broadcast the relief dominates. For a small one there is no
meaningful relief and the serialization is pure loss, which is why eligibility
carries a minimum broadcast size rather than firing on the shape alone.

## Candidate Recognition

The pass operates within one basic block. It collects waits carrying
`constraints.WSBarrier` and examines consecutive pairs in program order. A
pair is eligible only when:

- every operation between the waits is an allowed load, shape change,
  broadcast, or cast;
- every intervening barrier is also a `WSBarrier` and passes the barrier-order
  proof;
- the later wait's operands dominate the earlier insertion point; and
- the interval contains a `tt.broadcast` whose result is at least
  `kMinRegisterHeavyBroadcastElemsPerThread` (32) elements per thread.

The size floor is what keeps the pass on the side of the trade described
above. A `tt.broadcast` alone is a shape-only signal: it identifies the PTXAS
optimization opportunity but says nothing about how many registers the hoist
frees, while the overlap it costs is unconditional. Requiring a register-heavy
result means the pass fires on the case it was measured on and declines the
ones where it would only serialize. The `128x128xf32` bias tile that motivated
the pass sits at 128 elements per thread, so the floor leaves a wide margin.
After moving one wait, the pass rescans the block. This continues until no
consecutive pair can be unified, allowing a sequence of compatible regions to
collapse to one wait location.

For each successful unification, the pass records the intervening local loads.
It then finds a TMEM operand and one of those SMEM operands meeting at a common
pure operation. If the SMEM side is a single-use cheap preparation chain ending
in a qualifying broadcast, it moves the load, release, and preparation chain
after the TMEM channel. Both waits stay at the common location, so the reorder
does not change when the consumer begins waiting for the asynchronous result.

## Profitability Rule

The first version deliberately uses a narrow allowlist between the two waits:

- `ttg.local_load` and `ttng.tmem_load`;
- `ttg.convert_layout`;
- `tt.expand_dims` and `tt.broadcast`;
- `arith.extf` and `arith.truncf`;
- memdesc views and scalar integer barrier-phase bookkeeping; and
- WS barrier operations that pass the movement proof.

At least one `tt.broadcast` must be present. The same allowlist, single-use
rule, and size floor gate the operand reorder. Loads and casts alone do not
make a region profitable. Arithmetic, reductions, unknown side effects, and
every other operation keep the waits and operands in their original order.

## Safety

The later wait is moved immediately after the earlier wait only if all of its
operands already dominate that location. Every crossed operation is checked:

- regions and unsupported barrier-like operations reject the move;
- another wait must carry `WSBarrier` constraints;
- crossing an arrive requires
  `canAdvanceWSBarrierArrivePastWait`, using either disjoint channel graphs or
  the ordered-region proof;
- TMA, MMAv5, `tc_gen5_commit`, and other arrive-like operations reject the
  move; and
- any operation outside the explicit profitability allowlist rejects the
  move.

The operand reorder validates that the arrive following the SMEM load shares
the acquire's ordered parent and proves every crossed barrier safe. After wait
unification it moves only the SMEM load/release/preparation chain; the waits
remain fixed. If global WS-barrier reordering is disabled, it skips wait
unification and moves the complete SMEM channel as the compatibility fallback.

## Pipeline and Controls

The NVIDIA backend schedules the pass directly after
`triton-nvidia-interleave-tmem` and before generic data-duplication and
instruction-reordering passes. It is enabled independently by default; set
`TRITON_ENABLE_UNIFY_WS_BARRIER_LOCATIONS=0` to disable both transformations.
`TRITON_DISABLE_WSBARRIER_REORDER=1` disables wait unification but retains the
whole-channel operand-ordering fallback used by the FA kernels.

Focused coverage is in
`test/TritonNvidiaGPU/unify_ws_barrier_locations.mlir`.
