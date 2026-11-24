# Named Barrier API Changes

Source design: [Named Barrier API Changes](https://docs.google.com/document/d/12Ue2PtbEQq9ih_MfLOjqdBroMEdSs1_ZdXFZw0Qjng0/edit?tab=t.0)

## 1. Problem statement

Before this stack, a named-barrier ID had no first-class representation in the
Triton stack. The barrier operand of `ttng.wait_barrier_named` and
`ttng.arrive_barrier_named` was an `i32`, usually a folded `arith.constant`,
but the IR had no type, attribute, symbol, wrapper, or allocation that
identified the value as a hardware named-barrier ID.

Named-barrier IDs came from two independent sources that did not share a
namespace:

- The user/TLX path lowered the literal supplied to
  `tlx.named_barrier_wait` or `tlx.named_barrier_arrive` directly to an
  `arith.constant i32`.
- Compiler passes assigned IDs with local counters and materialized them as
  constants. `PingPong.cpp` allocated two IDs per critical region starting at
  7. `ConvertWarpSpecializeToLLVM` reserved IDs 1 and 2 for fixed lowering
  paths.

The raw-`i32` representation had four problems:

- There was no validation that an ID was in the hardware range `[0, 15]`, and
  no protection for IDs with reserved semantics.
- User-selected and compiler-selected IDs could silently collide.
- A bare integer operand did not communicate that it named a hardware
  barrier.
- The IR did not distinguish a compile-time ID from a runtime value, which
  prevented compiler allocation when existing IDs could not be proven.

## 2. Goals and non-goals

### Goals

- Validate user-specified named barriers in TLX with useful diagnostics.
- Represent named-barrier intent explicitly in IR and share that
  representation between Triton and TLX lowering.
- Centralize compiler allocation of named barriers.
- Allow profitable mbarriers to be promoted to named barriers without
  colliding with user reservations or fixed compiler lowering.

### Non-goals

- The frontend will not expose a named-barrier allocator in this change.
- Reusing IDs through named-barrier liveness analysis is deferred. Each
  compiler-promoted mbarrier owns one ID for the whole kernel.
- The initial profitability model will not attempt to identify the kernel's
  true critical path.
- Cluster-scoped or byte-counted asynchronous mbarriers are not promotable.

## 3. IR change

The implementation introduces explicit user and compiler named-barrier ID
wrappers. Each wrapper is created during IR translation and remains opaque to
the source-level user.
Both wrappers carry an integer SSA value, but only compiler-created IDs must be
constant.

### 3.1 Frontend

The public TLX API is unchanged. Users continue to pass an integer ID to the
named-barrier operations. The Python frontend validates statically known IDs
before translation so common mistakes produce a source-facing error.

A future extension may expose an allocator so users do not have to choose IDs
directly.

### 3.2 IR translation and bindings

The TLX bindings wrap the ID operand of the named-barrier arrive and wait
operations as a user named-barrier ID. The operations consume the explicit
named-barrier-ID interface instead of an unqualified `i32`.

Compiler transformations create compiler named-barrier IDs through a separate
builder API. A compiler ID must be constant and must have been reserved from
the shared pool described in section 4.1.

### 3.3 Validation

Users may reuse an ID and may provide a non-static ID. When a user-provided ID
is statically known, both the Python frontend and the IR verifier enforce:

- ID 0 is rejected with a diagnostic that it is reserved for the compiler.
- IDs 1 and 2 are rejected while they remain reserved by fixed lowering paths.
- IDs outside the hardware range `[0, 15]` are rejected.

Each rule has a distinct diagnostic. Dynamic user IDs remain legal, but their
presence makes compiler allocation unsafe because the compiler cannot prove
which hardware IDs are free.

Compiler named-barrier IDs must always be static and within the hardware range
`[0, 15]`. That is wider than the allocator pool of section 4.1: the fixed
lowering paths own IDs 0, 1 and 2 as compiler IDs without drawing them from the
pool, so the verifier cannot require pool membership here.

## 4. Compiler named-barrier allocation

Before the unified allocator and the PingPong refactor, named barriers could
be introduced in several places:

- NVVM barrier lowering used ID 0 when no ID was specified.
- `ConvertWarpSpecializeToLLVM` used IDs 1 and 2 for fixed lowering paths.
- Ping-pong synchronization allocated two IDs per critical region from a
  hard-coded 7 through 15 range. It did not inspect existing named-barrier
  use, so it could hand out an ID a user had already reserved.

These local policies did not protect against collisions when the compiler and
user both needed named barriers. Allocation is now centralized; PingPong emits
mbarriers instead of allocating named-barrier IDs itself (§6).

### 4.1 Unified named-barrier pool

The compiler-owned pool starts with IDs `[3, 15]`. IDs 0, 1, and 2
remain reserved until the corresponding fixed lowerings are migrated to the
same API.

At the point where the pool is constructed, scan all existing user
named-barrier definitions:

1. Hoist or canonicalize definitions where doing so produces a single SSA
   value for each statically known reservation.
2. Remove every statically known user ID from the pool.
3. If any user ID cannot be inferred at compile time, mark the pool
   unavailable. Optional compiler optimizations must then decline to allocate
   named barriers.

Compiler passes request IDs from the shared pool and create explicit compiler
named-barrier IDs. Required allocations must report exhaustion; optional
optimizations must remain correct when no ID is available. The initial
implementation does not recycle IDs based on liveness.

## 5. mbarrier-to-named-barrier promotion

Compiler-generated Triton mbarriers can sometimes be replaced with named
barriers. `TritonNvidiaGPUPromoteMBarrierToNamedBarrierPass` runs in the NVIDIA
TTGIR pipeline after TMA lowering and optional partition-warp optimization,
before fence insertion and MMA lowering, when
`TRITON_PROMOTE_MBAR_TO_NAMED_BARRIER` is enabled (off by default). A successful
promotion replaces each arrive/wait with a named-barrier operation and erases
the now-unused shared-memory allocation, initialization, invalidation, views,
and captures. TLX's explicit named-barrier APIs and their ID validation remain
independent of this optional promotion.

The pass applies to **Triton-generated** barriers, not TLX-authored ones.
`triton-tlx-fixup` examines source-level TTIR immediately after AST lowering
and sets the module's `triton.skip_generic_pipeline` attribute when it finds
TLX dialect ops, explicit local-memory/TMEM operations, manually written
`ttg.warp_specialize` ops, or TLX 2-CTA MMA. The promotion pass checks that
attribute first and returns **before allocating named-barrier IDs**. Checking
for surviving TLX ops in late TTGIR would miss kernels expressed solely with
manual warp specialization or whose TLX ops have already been lowered. This
source marker conservatively excludes any other module with these explicitly
authored GPU constructs as well; it does not disable unrelated TLX passes or
user-authored named barriers.

### 5.1 Promotion safety

The implemented pass collects complete `1xi64` or `Kx1xi64` local-alloc
barrier groups. Every physical slot must have exactly one initialization (and,
if present, one invalidation); all uses must be recognized barrier operations,
memdesc views, partition captures, or deallocations. No partial promotion of
an allocation is attempted. For an eligible group, it checks:

1. Arrivals are neither per-thread nor multicast, have no nonconstant-true
   predicate, and execute uniformly within their task. Waits have no async
   dependencies, have a recognized phase, and likewise execute uniformly.
   Only `scf.for`, `scf.if`, and provably uniform `scf.while` ancestors are
   allowed. Matching guard chains must choose the same `if` branches and
   `while` iterations across **every** arrival and wait task; matching loop
   nests must have equivalent iteration spaces. Unprovable or divergent
   control flow is rejected.
2. For each slot, producer tasks are distinct from consumer tasks. The sum
   of producer arrival counts equals its initialization count; a producer
   task can have one looped arrival and, where the schedule permits, one
   unlooped priming arrival, but not two looped arrivals. A consumer task has
   **at most one wait site per slot**. This matters after software pipelining:
   two waits can legally observe the same completed mbarrier phase, whereas
   two named-barrier waits participate in successive generations and may
   hang. Dynamic indices are conservatively counted as potentially touching
   every slot; two waits for different *static* slots are allowed.
3. The barriers are CTA-local (including the descriptor's shape and memory
   space), all active slots belong to the same warp-specialize operation, and
   each slot has the same participant thread count. Named-barrier arrivals and
   waits use the sum of the producer and consumer task warp counts, not the
   mbarrier initialization count.
4. The shared named-barrier allocator can reserve one ID per active slot
   without colliding with other compiler or user reservations. Allocation
   failure leaves the candidate as an mbarrier without failing compilation.

These are deliberately recognizable schedule shapes, not a proof of arbitrary
mbarrier protocols. In particular, the pass does not establish that two
syntactically different pipelined wait sites each correspond to a distinct
generation, nor does it prove lifetime/reuse for an arbitrary source-written
barrier. Unsupported schedules remain mbarriers; keep the promotion knob off
unless the producing protocol has been validated.

### 5.2 Phase-offset and backward-edge barriers

An mbarrier wait may be initially satisfied because its requested parity does
not match the phase established by `init_barrier`. This is common for backward
edges in a pipelined or circular-buffer schedule: the consumer must not wait
before the buffer's first use, but subsequent uses must wait for the producer
from an earlier iteration. A named barrier has no initial parity state, so an
unconditional named-barrier wait would introduce a synchronization that was
not present in the original program.

The implementation recognizes only these wait-phase and slot schedules:

- A constant phase zero, provided the wait is unconditionally executed and
  its producer and consumer loop nests and guard chains match.
- A counter-derived parity, `(iteration & 1)` or
  `((iteration / K) & 1)` (also accepting modulo-two and the compiler's
  equivalent division/multiplication/subtraction form of modulo). The
  iteration must be a positive-step normalized `scf.for` induction count or
  a nonnegative, loop-carried counter incremented by a positive constant.
  The accepted dynamic schedules start at iteration zero, advance once per
  iteration, have no phase offset, and use the same counter and equivalent
  iteration spaces in producer and consumer tasks. For a dynamic `K`-slot
  group, both sides select `iteration % K` and phase
  `(iteration / K) % 2`.
- A separate static-slot pattern: exactly one wait loop per static slot,
  phase `iteration % 2`, and matching looped arrivals per slot. A producer
  may also make one unlooped priming arrival *before* its matching loop in
  the same task. A phase that starts satisfied without a matching arrival is
  not promotable.

The pass rejects a mix of constant and counter-derived waits (for example,
a peeled prologue), nonconstant-true wait predicates, missing required priming
arrivals, and schedules whose guards, slot indices, or per-task sites cannot
be matched. A warp-uniform dynamic loop bound is permitted when producer and
consumer bounds are provably equivalent; dynamic exit conditions that cannot
be proved equal are not. Named-barrier waits do not observe an mbarrier phase
passively: each participating warp enters the next named-barrier generation.
The per-task, per-slot site check (§5.1) deliberately leaves SWP-expanded
multiple-arrive/wait patterns as mbarriers even if their individual phase
expressions look valid.

### 5.3 Profitability

The implemented policy is deliberately conservative:

- Promote an entire multi-buffer group or none of it. Partial promotion does
  not eliminate the group's allocation and phase-management cost.
- Do not promote per-thread-arrival barriers. Independent arrival is an
  mbarrier advantage and removing it may require extra synchronization.
- Do not promote predicated, initially-skipped, or loop-peeled schedules.
  They fail the current safety and phase-shape checks (§5.2). Synthesizing
  control flow to make them eligible is outside this pass; rejected candidates
  retain their mbarriers.

A stronger cost model cannot relax the §5.2 safety rules. Admitting predicated
or initially-skipped schedules would first require a soundness proof (for
example, predicated named-barrier waits with proven generation alignment),
and only then a cost argument.

### 5.4 Priority

After reserving warp-specialize IDs, the pass validates candidates before
rewriting any of them. It sorts eligible allocations by descending
partition-relative `scf.for` depth, then ascending number of active slots
(IDs required), and finally original IR walk order for deterministic ties.
If the allocator cannot supply the whole group's IDs, that group is skipped.
A future model may include critical-path or bottleneck information.

## 6. Ping-pong-to-mbarrier refactor

PingPong now allocates an mbarrier pair and emits arrive/wait operations with
loop-carried phases, including a priming arrival for the first ping-pong
round. It first checks that the two task loops have equivalent iteration
spaces. The optional promotion pass can turn an eligible pair into named
barriers, but the generated protocol remains valid as mbarriers when the knob
is off, IDs are unavailable, or eligibility fails. TLX-marked modules keep
their mbarriers because the promotion pass exits before analyzing them.

## Testing requirements

Focused lit coverage in `test/TritonNvidiaGPU/promote-mbarrier-to-named-barrier.mlir`
checks eligible Triton rewrites, multibuffer/phase schedules, guard and
participant rejection, SWP-duplicated wait-site rejection, and preservation
of an otherwise-promotable TLX-marked module (including its barrier-ID state).
`test/TLX/attach-metadata.mlir` verifies that TLX fixup attaches the source
marker even for manually warp-specialized kernels with no TLX dialect ops.
The Blackwell AutoWS and Hopper PingPong tests exercise enabled promotion
for Triton. TLX correctness tests should validate their native mbarrier and
named-barrier protocols unchanged, **not** require TLX promotion.
