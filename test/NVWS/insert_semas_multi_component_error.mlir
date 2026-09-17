// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %s

// Two allocations share buffer.id=402 but occupy disjoint, non-overlapping
// column ranges ([0,128) and [256,384)) with no covering member, so the
// piece table splits into two connected components. The memory planner never
// emits this (reusers are stacked within their owner's columns), so InsertSemas
// rejects it rather than mis-synchronizing the single-component protocol.
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @disjoint_members_reject(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK: buffer.id group has disjoint pieces (more than one connected component)
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttg.local_alloc %cst0 {buffer.id = 402 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %av = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use_a"(%av) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      %b = ttg.local_alloc %cst1 {buffer.id = 402 : i32, buffer.offset = 256 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %bv = ttg.local_load %b {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use_b"(%bv) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
