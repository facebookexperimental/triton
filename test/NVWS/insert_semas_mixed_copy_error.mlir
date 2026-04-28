// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %s

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @mixed_copy_reject(%lb: i32, %ub: i32, %step: i32) {
    %a = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 77 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: nvws-insert-semas: TMEM allocations sharing buffer.id 77 have conflicting buffer.copy values 1 and 2
    // CHECK: first buffer.copy value is 1
    %b = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 77 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0>,
       ttg.partition.stages = [0 : i32], ttg.partition.types = ["default"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
