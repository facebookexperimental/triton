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
// inside, {2} (carried) and {1} read, {1} CORRECTS the buffer in place
// (the store's WAR obligations resolve early: {0}-less here — {2}'s via
// the transitive-sync skip), and {0} consumes the corrected value AFTER
// the store. The last version's holders at the inner EXIT are therefore
// {1} (the writer) and {0} (its reader) — a fan-in-2 regain — while the
// For-row unification merges the outer single-source ready edge onto the
// SAME semaphore. The lone outer release must arrive twice: r S(2). At
// commit 3 this is checked on the SYNC-DAG dump; commit 4 extends it with
// IR-level checks (async_ops arity carries the multiplicity).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // EMIT-LABEL: @release_multiplicity_unified_fanin_regain
  tt.func @release_multiplicity_unified_fanin_regain(%lb: i32, %ub: i32, %step: i32) {
    // EMIT: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 700 : i32}
    %alloc = ttg.local_alloc {buffer.id = 700 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // EMIT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32}
    // EMIT: [[FULL2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 2 : i32}
    // EMIT: [[S3:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // EMIT: [[S4:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // EMIT: [[INIT:%.*]] = nvws.semaphore.acquire [[EMPTY]] : {{.*}} -> !ttg.async.token
    // EMIT: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[CARRY:%.*]] = [[INIT]]) -> (!ttg.async.token)
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 3>} : () -> !ty
      // EMIT: [[BUF3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRY]] {ttg.partition = array<i32: 3>}
      // EMIT: ttg.local_store {{%.*}}, [[BUF3]] {ttg.partition = array<i32: 3>}
      // EMIT: nvws.semaphore.release [[FULL2]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 2 : i32, ttg.partition = array<i32: 3>}
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 3>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // EMIT: [[ACQ2:%.*]] = nvws.semaphore.acquire [[FULL2]] {ttg.partition = array<i32: 2>} : {{.*}} -> !ttg.async.token
      // EMIT: [[INNER:%.*]] = scf.for {{.*}} iter_args([[JCARRY:%.*]] = [[ACQ2]]) -> (!ttg.async.token)
      scf.for %j = %lb to %ub step %step : i32 {
        // EMIT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL2]], [[JCARRY]] {ttg.partition = array<i32: 2>}
        // EMIT: ttg.local_load [[BUF2]] {ttg.partition = array<i32: 2>}
        // EMIT: nvws.semaphore.release [[S3]], [[JCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
        %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "use2"(%l2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        // EMIT: [[ACQ1:%.*]] = nvws.semaphore.acquire [[S3]] {ttg.partition = array<i32: 1>} : {{.*}} -> !ttg.async.token
        // EMIT: [[BUF1:%.*]] = nvws.semaphore.buffer [[S3]], [[ACQ1]] {ttg.partition = array<i32: 1>}
        // EMIT: ttg.local_load [[BUF1]] {ttg.partition = array<i32: 1>}
        %l1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        %c = "correct"(%l1) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
        // EMIT: ttg.local_store {{%.*}}, [[BUF1]] {ttg.partition = array<i32: 1>}
        // EMIT: nvws.semaphore.release [[S4]], [[ACQ1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // EMIT: nvws.semaphore.release [[FULL2]], [[ACQ1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        ttg.local_store %c, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: [[ACQ0:%.*]] = nvws.semaphore.acquire [[S4]] {ttg.partition = array<i32: 0>} : {{.*}} -> !ttg.async.token
        // EMIT: [[BUF0:%.*]] = nvws.semaphore.buffer [[S4]], [[ACQ0]] {ttg.partition = array<i32: 0>}
        // EMIT: ttg.local_load [[BUF0]] {ttg.partition = array<i32: 0>}
        %l0 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        // EMIT: nvws.semaphore.release [[FULL2]], [[ACQ0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        "use0"(%l0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
        // EMIT: [[ACQ2B:%.*]] = nvws.semaphore.acquire [[FULL2]] {ttg.partition = array<i32: 2>} : {{.*}} -> !ttg.async.token
        // EMIT: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[ACQ2B]]
      } {ttg.partition = array<i32: 0, 1, 2>}
      // EMIT: nvws.semaphore.release [[EMPTY]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // EMIT: [[ACQ3:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 3>} : {{.*}} -> !ttg.async.token
      // EMIT: scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} [[ACQ3]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// The SYNC-DAG view (stderr dump). The outer ready release by {3} carries
// arrive multiplicity 2 — one release op, two arrives — because it shares
// the semaphore with the inner fan-in-2 regain ({1} the corrector and {0}
// its post-store reader both close into carried {2}); both acquire sites
// show the uniform pending count (2). The in-loop store itself takes NO
// edge: {2}'s WAR is discharged transitively ({1} synced behind {2} via
// its own read edge), {1}'s own read by program order. Under the v5 uniform
// hold rule, the outer S3 component remains carrier-bearing: the prefix's
// release has arrive multiplicity 2, so condition E rejects point-of-use
// (`rel-count`) to preserve the one-carrier-slot rule.
