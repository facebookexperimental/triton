// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reject_managed_flow_across_cfg_blocks(
      %early: i1, %lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 1201 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: error: nvws-insert-semas: managed memdesc flow across function CFG blocks is unsupported
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
