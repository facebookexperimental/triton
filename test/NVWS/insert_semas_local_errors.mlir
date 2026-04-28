// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @unsupported_local_memdesc_forwarding(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 203 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      // CHECK: nvws-insert-semas: unsupported memdesc alias use test.memdesc_view
      %view = "test.memdesc_view"(%alloc) {ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> !ttg.memdesc<1xi32, #shared, #smem>
      %l = ttg.local_load %view {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> !ty
      "use"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
