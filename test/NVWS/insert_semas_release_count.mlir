// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s --check-prefix=EMIT
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas --nvws-lower-semaphore -cse | FileCheck %s --check-prefix=LOWER
//
// EMIT/LOWER pin the first-class count contract end to end on THIS
// pass-produced shape (fable/integrate-pending-count-plan.md): the
// scaled release carries arrive_count = 2 in emitted IR, and the
// lowering transcribes both counts into the mbarrier init/arrive.
// LOWER: ttng.init_barrier {{.*}}, 2
// LOWER: ttng.arrive_barrier {{.*}}, 2

// Release arrive-multiplicity (spec section 5.2, uniform pending count):
// a semaphore's pending count is a per-semaphore constant — every acquire
// site sees the same count and every acquire cycle must receive exactly
// that many arrives. Shape: producer {3} stores outside the inner loop;
// inside, {2} and {1} read, {1} CORRECTS the buffer in place after an
// explicit {2}->{1} WAR handoff, and {0} consumes the corrected value
// AFTER the store. The last version's holders at the inner EXIT are
// therefore {1} (the writer) and {0} (its reader) — a fan-in-2 regain —
// and the outer single-source ready edge lands on the SAME semaphore.
// The lone outer release must arrive twice: r S(2). The emitted and
// lowered IR checks below pin that multiplicity.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // EMIT-LABEL: @release_multiplicity_unified_fanin_regain
  tt.func @release_multiplicity_unified_fanin_regain(%lb: i32, %ub: i32, %step: i32) {
    // EMIT: [[V1:%.*]] = ttg.local_alloc {buffer.id = 700 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %alloc = ttg.local_alloc {buffer.id = 700 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // V2 = outer ready (initially released — supplies iteration zero),
    // V3 = {2}->{1} read handoff, V4 = {2}->{1} WAR handoff, V5 =
    // {1}->{0} corrected-value edge, V6 = the unified full semaphore
    // with pending_count = 2.
    // EMIT: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT: [[V5:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT: [[V6:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT-NOT: iter_args
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 3>} : () -> !ty
      // {3}'s acquire of the outer ready semaphore sits at its point of
      // use inside the loop body; no token is threaded through the loop.
      // EMIT: [[V7:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // EMIT: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // EMIT: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]] {ttg.partition = array<i32: 3>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // EMIT: nvws.semaphore.release [[V6]], [[V7]] [#nvws.async_op<none>] {arrive_count = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 3>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // EMIT-NOT: iter_args
      scf.for %j = %lb to %ub step %step : i32 {
        // EMIT: [[V9:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // EMIT: nvws.semaphore.release [[V3]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // EMIT: [[V10:%.*]] = nvws.semaphore.buffer [[V6]], [[V9]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: ttg.local_load [[V10]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        // EMIT: nvws.semaphore.release [[V4]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "use2"(%l2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        // EMIT: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // EMIT: [[V12:%.*]] = nvws.semaphore.buffer [[V3]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: ttg.local_load [[V12]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        %l1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        %c = "correct"(%l1) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
        // EMIT: [[V13:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // EMIT: [[V14:%.*]] = nvws.semaphore.buffer [[V4]], [[V13]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V14]] {ttg.partition = array<i32: 1>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: nvws.semaphore.release [[V5]], [[V13]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // EMIT: nvws.semaphore.release [[V6]], [[V13]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        ttg.local_store %c, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: [[V15:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // EMIT: [[V16:%.*]] = nvws.semaphore.buffer [[V5]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: ttg.local_load [[V16]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        %l0 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        // EMIT: nvws.semaphore.release [[V6]], [[V15]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "use0"(%l0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      } {ttg.partition = array<i32: 0, 1, 2>}
      // After the inner loop, {2} regains the last version — its acquire
      // cycle absorbs the fan-in-2 arrives from {1} and {0} — and only
      // then releases the outer ready semaphore for {3}'s next store.
      // EMIT: } {ttg.partition = array<i32: 0, 1, 2>}
      // EMIT: [[V17:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // EMIT: nvws.semaphore.release [[V2]], [[V17]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// The outer ready release by {3} carries arrive multiplicity 2 — one
// release op, two arrives — because it shares the pending_count = 2
// semaphore with the inner fan-in-2 regain ({1} the corrector and {0}
// its post-store reader each arrive once into {2}'s next acquire cycle);
// both acquire sites read the uniform pending count off the one create.
// Every acquire sits at its point of use: neither loop threads a token,
// and the outer ready semaphore is created initially released to supply
// iteration zero. The stable ENTER source lets {1}'s read overlap {2}'s
// read, so the in-loop store takes an explicit {2}->{1} WAR edge; {1}'s
// own read is ordered by program order.
