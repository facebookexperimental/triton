// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // S1: prefix For, p1-anchored transparent region.
  // The p1 hold is carried into the inner loop (transparent): the store acquires
  // EMPTY once at the top, the inner loop carries that same token, and the trailing
  // p2 read after the inner loop gates a fresh acquire on the regained handle.
  // CHECK-LABEL: @uniform_hold_s1_prefix_for_p1
  tt.func @uniform_hold_s1_prefix_for_p1(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 981 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F4:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 981 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[CARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F2]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 2>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
        // CHECK: nvws.semaphore.release [[F4]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F4]], [[T3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S2: owner-change at op1 -> inner; negative expected.
  // The p1 hold cuts at op1 (store releases F3 immediately); the inner loop stays
  // plain — each iteration's p2 load acquires F3 at its point of use — and the
  // trailing p2 read takes its own F3 acquire after the loop.
  // CHECK-LABEL: @uniform_hold_s2_owner_change_cut
  tt.func @uniform_hold_s2_owner_change_cut(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 982 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 982 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F3]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F3]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F3]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>}
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F3]], [[T3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S3: trailing read after regain; negative expected.
  // A function-level EMPTY acquire threads the outer and middle loops. The p1
  // store reuses the carried handle and releases F3; the inner loop stays plain
  // (point-of-use acquires). A p2 close-acquire on F3 after the inner loop hands
  // EMPTY back, and the trailing p1 read's fresh EMPTY acquire is the yielded carrier.
  // CHECK-LABEL: @uniform_hold_s3_trailing_read_after_regain
  tt.func @uniform_hold_s3_trailing_read_after_regain(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 983 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 983 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: [[MID:%.*]] = scf.for {{.*}} iter_args([[MCARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[MCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F3]], [[MCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F3]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F3]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>}
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[T4:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 1>}
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T4]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[MID]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S4: prefix For, p2-anchored owner mirror of S1.
  // CHECK-LABEL: @uniform_hold_s4_prefix_for_p2
  tt.func @uniform_hold_s4_prefix_for_p2(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 984 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F4:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 984 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 2>}
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[CARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRY]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F2]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
        // CHECK: nvws.semaphore.release [[F4]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F4]], [[T3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S5: region-spanning at WS-body depth 1, no middle, no op4.
  // A function-level EMPTY acquire is carried by the outer loop; the p1 store
  // buffers on the carried token. The inner loop carries it too and yields the
  // bottom p1 re-acquire; the outer loop carries the regained handle onward
  // (no post-loop drain).
  // CHECK-LABEL: @uniform_hold_s5_ws_body_depth1
  tt.func @uniform_hold_s5_ws_body_depth1(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 985 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 985 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[OCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[ICARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ICARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[ICARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[INNER]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S6: same-owner trailing read after the inner region.
  // The p1 store reuses the function-level carried EMPTY handle and releases
  // EMPTY at once (hold cut); the inner loop stays plain (point-of-use EMPTY
  // acquires), and the trailing p1 read takes a fresh EMPTY acquire whose token
  // the outer loop yields.
  // CHECK-LABEL: @uniform_hold_s6_same_owner_trailing_read
  tt.func @uniform_hold_s6_same_owner_trailing_read(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 986 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 986 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[OCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[OCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>}
      // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 1>}
      %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T3]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S7: cross-owner store out / load in.
  // p1 store acquires EMPTY then releases FULL; the inner p2 load acquires FULL
  // above the inner loop, uses the same token inside (no carrier needed — the
  // inner loop is plain), and the regained EMPTY handle is released after.
  // CHECK-LABEL: @uniform_hold_s7_cross_owner_store_load
  tt.func @uniform_hold_s7_cross_owner_store_load(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 987 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 987 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[F2]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: scf.for {{.*}}  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S8: If-as-prefix-region inside the WS body.
  // A function-level EMPTY acquire is carried by the loop; the p1 store buffers
  // on the carried token. The scf.if yields the token (bottom p1 re-acquire in
  // then, pass-through in else) and the loop carries the regained handle onward
  // (no post-if drain).
  // CHECK-LABEL: @uniform_hold_s8_if_prefix_region
  tt.func @uniform_hold_s8_if_prefix_region(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 988 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 988 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[OCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[IF:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
      scf.if %cond {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[OCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[OCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"() {ttg.partition = array<i32: 2>} : () -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
      // CHECK: } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[OCARRY]] : !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[IF]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S9: If inside the inner loop.
  // The store/load/store all live inside the conditional; a single carrier is
  // threaded from a function-level acquire through outer loop, inner loop, and
  // the scf.if (point-of-use store on the carried buffer, else passes through).
  // CHECK-LABEL: @uniform_hold_s9_if_inside_inner_loop
  tt.func @uniform_hold_s9_if_inside_inner_loop(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 989 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 989 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[ICARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[IF:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        scf.if %cond {
          // CHECK: "producer1"
          // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ICARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
          // CHECK: ttg.local_load [[B0]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F2]], [[ICARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"() {ttg.partition = array<i32: 2>} : () -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        // CHECK: } else {
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[ICARRY]] : !ttg.async.token
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[IF]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[INNER]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S10: fan-out / multi-consumer.
  // The store's fan-out sema (last create, pending_count = 2) collects two
  // consumer arrives per cycle. Both loops stay plain: the p3 store and each
  // inner iteration's p2 read acquire at point of use, the correcting store
  // takes a separate {2}->{1} WAR handoff, and a p2 close-acquire after the
  // inner loop hands EMPTY back to the p3 store.
  // CHECK-LABEL: @uniform_hold_s10_fanout_multi_consumer
  tt.func @uniform_hold_s10_fanout_multi_consumer(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 990 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F4:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F5:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 990 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer0"
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 3>}
      // CHECK: nvws.semaphore.release [[F5]], [[T0]] [#nvws.async_op<none>] {arrive_count = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %v0 = "producer0"(%i0) {ttg.partition = array<i32: 3>} : (i32) -> !ty
      ttg.local_store %v0, %alloc {ttg.partition = array<i32: 3>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F5]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: nvws.semaphore.release [[F2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F5]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v1 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer1"(%v1) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B2]] {ttg.partition = array<i32: 1>}
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F3]], [[T3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B3]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F4]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: nvws.semaphore.release [[F5]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T4:%.*]] = nvws.semaphore.acquire [[F4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B4:%.*]] = nvws.semaphore.buffer [[F4]], [[T4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B4]] {ttg.partition = array<i32: 0>}
        // CHECK: nvws.semaphore.release [[F5]], [[T4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      } {ttg.partition = array<i32: 0, 1, 2, 3>}
      // CHECK: } {ttg.partition = array<i32: 0, 1, 2, 3>}
      // CHECK: [[T5:%.*]] = nvws.semaphore.acquire [[F5]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
