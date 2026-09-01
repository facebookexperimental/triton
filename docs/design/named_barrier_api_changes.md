# Named Barrier API Changes

Source design: [Named Barrier API Changes](https://docs.google.com/document/d/12Ue2PtbEQq9ih_MfLOjqdBroMEdSs1_ZdXFZw0Qjng0/edit?tab=t.0)

## 1. Problem statement

A named-barrier ID has no first-class representation in the Triton stack. The
barrier operand of `ttng.wait_barrier_named` and
`ttng.arrive_barrier_named` is an `i32`. In practice, it is usually a folded
`arith.constant`, but the IR has no type, attribute, symbol, wrapper, or
allocation that identifies the value as a hardware named-barrier ID.

Named-barrier IDs currently come from two independent sources that do not
share a namespace:

- The user/TLX path lowers the literal supplied to
  `tlx.named_barrier_wait` or `tlx.named_barrier_arrive` directly to an
  `arith.constant i32`.
- Compiler passes assign IDs with local counters and materialize them as
  constants. `PingPong.cpp` allocates two IDs per critical region starting at
  7. `ConvertWarpSpecializeToLLVM` reserves IDs 1 and 2 for fixed lowering
  paths.

The raw-`i32` representation has four problems:

- There is no validation that an ID is in the hardware range `[0, 15]`, and
  no protection for IDs with reserved semantics.
- User-selected and compiler-selected IDs can silently collide.
- A bare integer operand does not communicate that it names a hardware
  barrier.
- The IR does not distinguish a compile-time ID from a runtime value, which
  prevents compiler allocation when the existing IDs cannot be proven.

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

Introduce explicit user and compiler named-barrier ID wrappers. The wrapper is
created during IR translation and remains opaque to the source-level user.
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

Named barriers can currently be introduced in several places:

- NVVM barrier lowering uses ID 0 when no ID is specified.
- `ConvertWarpSpecializeToLLVM` uses IDs 1 and 2 for fixed lowering paths.
- Ping-pong synchronization allocates two IDs per critical region from a
  hard-coded 7 through 15 range. It does not inspect existing named-barrier
  use, so it will hand out an ID a user has already reserved.

These local policies do not protect against collisions when the compiler and
user both need named barriers. Allocation must instead be centralized.

### 4.1 Unified named-barrier pool

Create a compiler-owned pool initialized with IDs `[3, 15]`. IDs 0, 1, and 2
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

Triton source generally should not expose barriers directly, but compiler
generated mbarriers can sometimes be replaced with named barriers. Promotion
removes shared-memory allocation, barrier initialization, shared-memory
communication, and phase computation. A dedicated pass provides one policy
for both ordinary Triton and TLX instead of teaching every producer pass to
allocate named barriers independently.

The pass is disabled by default. End-to-end coverage must exercise all three
configurations:

1. Regular AutoWS Triton.
2. TLX without source-level named barriers.
3. TLX with source-level named barriers that reserve IDs from the shared pool.

### 5.1 Promotion safety

An mbarrier is promotable only when all of the following are proven:

1. Its uses consist only of arrivals without an expected byte count and wait
   operations. Asynchronous-proxy operations are excluded.
2. Its arrival count and total participating thread count are statically
   known.
3. Every arrival and wait is warp-uniform. The participating thread count must
   be a multiple of the warp size, and no use may be nested in warp-divergent
   control flow.
4. It does not have cluster scope.

Failure to prove any condition leaves the mbarrier unchanged.

### 5.2 Profitability

The initial profitability policy is deliberately conservative:

- Promote an entire multi-buffer group or none of it. Partial promotion does
  not eliminate the group's allocation and phase-management cost.
- Do not promote per-thread-arrival barriers. Independent arrival is an
  mbarrier advantage and removing it may require extra synchronization.

These restrictions can be relaxed after evaluation provides a stronger cost
model.

### 5.3 Priority

When there are fewer named-barrier IDs than promotable groups, rank candidates
with these initial rules:

1. Exclude candidates that require additional synchronization, including
   every-thread mbarriers.
2. Prefer candidates used at greater loop depth as a proxy for execution
   frequency.
3. Prefer groups requiring fewer named-barrier IDs so complete groups are more
   likely to be promoted.

Tie-breaking must be deterministic. A future model may include critical-path
or bottleneck information.

## 6. Ping-pong-to-mbarrier refactor

After promotion is available, refactor PingPong to generate mbarriers and let
the promotion pass decide whether named barriers are profitable and available.
This removes its independent ID allocator and makes promotion the single
source of truth.

Before enabling this path by default, evaluate:

- Whether ping-pong remains profitable when only some or none of its
  mbarriers can be promoted.
- Whether the pass needs a configurable barrier limit in addition to the
  hardware limit enforced by the shared pool.

## Testing requirements

Each implementation change must include focused lit coverage for its IR
contract, verifier failures, allocator behavior, safety analysis, candidate
ranking, and rewrites. Final end-to-end tests must expose the
disabled-by-default promotion knob and demonstrate correct execution and code
generation for regular AutoWS Triton, TLX without named barriers, and TLX with
named barriers.
