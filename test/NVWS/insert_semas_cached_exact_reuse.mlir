// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s --check-prefix=IR
// RUN: env NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse 2>&1 | FileCheck %s --check-prefix=DAG

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // IR-LABEL: @cached_exact_reuse_after_release
  // DAG-LABEL: function: @cached_exact_reuse_after_release
  // DAG: |- a  S1  {1}
  // DAG: |- r  S0  {1} [none]
  // DAG: |- a  S0  {0}
  // DAG: |- r  S1  {0} [none]
  tt.func @cached_exact_reuse_after_release(%lb: i32, %ub: i32, %step: i32) {
    // IR: [[BASE:%[-A-Za-z0-9_.$#]+]] = ttg.local_alloc {buffer.id = 9820 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // IR: [[ENTRY:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.create [[BASE]] released = -1 {pending_count = 1 : i32}
    // IR: [[FULL:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32}
    %buf = ttg.local_alloc {buffer.id = 9820 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %value = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
    scf.for %i = %lb to %ub step %step : i32 {
      // IR: [[P1_TOKEN:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[ENTRY]] {ttg.partition = array<i32: 1>}
      // IR: [[P1_WRITE_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[ENTRY]], [[P1_TOKEN]] {ttg.partition = array<i32: 1>}
      // IR: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[P1_WRITE_BUF]] {ttg.partition = array<i32: 1>}
      ttg.local_store %value, %buf {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // IR: nvws.semaphore.release [[FULL]], [[P1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}

      // IR: [[P0_TOKEN:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // IR: [[P0_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[FULL]], [[P0_TOKEN]] {ttg.partition = array<i32: 0>}
      // IR: [[R0:%[-A-Za-z0-9_.$#]+]] = ttg.local_load [[P0_BUF]] {ttg.partition = array<i32: 0>}
      %r0 = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // IR: nvws.semaphore.release [[ENTRY]], [[P0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      "use0"(%r0) {ttg.partition = array<i32: 0>} : (!ty) -> ()

      // The first post-release read materializes an approved exact-reuse
      // buffer.  The second read must reuse that same view without making the
      // emitted-IR verifier demand a second release.
      // IR: [[P1_REUSE_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[ENTRY]], [[P1_TOKEN]] {ttg.partition = array<i32: 1>}
      // IR: [[R1A:%[-A-Za-z0-9_.$#]+]] = ttg.local_load [[P1_REUSE_BUF]] {ttg.partition = array<i32: 1>}
      %r1a = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "use1a"(%r1a) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      // IR-NOT: nvws.semaphore.buffer
      // IR: [[R1B:%[-A-Za-z0-9_.$#]+]] = ttg.local_load [[P1_REUSE_BUF]] {ttg.partition = array<i32: 1>}
      %r1b = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "use1b"(%r1b) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
