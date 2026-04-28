// RUN: triton-opt %s -allow-unregistered-dialect --nvws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=50000 smem-circular-reuse=true" | FileCheck %s --check-prefix=ODD
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-memory-planner="num-buffers=2 smem-alloc-algo=1 smem-budget=50000 smem-circular-reuse=true" | FileCheck %s --check-prefix=EVEN
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=50000" | FileCheck %s --check-prefix=DEFAULT

// Meta phase 4 forms a reuse group only because this priority has exactly two
// records. Depth 3 remains one odd circular pool; depth 2 splits into two
// independent copy-1 pools.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // ODD-LABEL: @two_phase4_candidates
  // The allocs are declared a,b but produced b,a. Starts follow producer order.
  // ODD: ttg.local_alloc {buffer.circular, buffer.copy = 3 : i32, buffer.id = [[ID:[0-9]+]] : i32, buffer.start = 1 : i32}
  // ODD-NEXT: ttg.local_alloc {buffer.circular, buffer.copy = 3 : i32, buffer.id = [[ID]] : i32, buffer.start = 0 : i32}
  // EVEN-LABEL: @two_phase4_candidates
  // EVEN: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[ID0:[0-9]+]] : i32}
  // EVEN-NEXT: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[ID1:[0-9]+]] : i32}
  // EVEN-NOT: buffer.circular
  // DEFAULT-LABEL: @two_phase4_candidates
  // DEFAULT: ttg.local_alloc {buffer.copy = {{[0-9]+}} : i32, buffer.id = 0 : i32}
  // DEFAULT-NEXT: ttg.local_alloc {buffer.copy = {{[0-9]+}} : i32, buffer.id = 1 : i32}
  // DEFAULT-NOT: buffer.circular
  tt.func @two_phase4_candidates(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %lb: i32, %ub: i32, %step: i32) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      nvws.descriptor_load %desc[%iv, %iv] 16384 %b {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc[%iv, %iv] 16384 %a {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %av = ttg.local_load %a {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %bv = ttg.local_load %b {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "use"(%av, %bv) : (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>) -> ()
    }
    tt.return
  }
}
