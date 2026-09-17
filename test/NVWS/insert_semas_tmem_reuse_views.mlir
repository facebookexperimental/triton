// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // This function emits NO semaphores: enforce their total absence.
  // CHECK-LABEL: @tmem_reuse_views_end_of_insert_semas
  // CHECK-NOT: nvws.semaphore
  tt.func @tmem_reuse_views_end_of_insert_semas() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %alias = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    %base = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %half = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    "use"(%alias, %base, %half) : (!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>) -> ()

    scf.for %iv = %c0 to %c1 step %c1 {
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}

    tt.return
  }
}
