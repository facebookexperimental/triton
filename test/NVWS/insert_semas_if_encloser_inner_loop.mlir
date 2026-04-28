// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// M0 PIN (nested-loop hold-rule extension, plan v3 §M0 item 5 — the
// canDrop(If) golden; the true If-ENCLOSER target is ABSENT from the corpus).
//
// WS-tagged outer loop -> scf.if -> non-WS inner loop, with ONE inner-confined
// ping-pong buffer in the if-branch. The scf.if sits BETWEEN the WS loop and
// the inner for, so it is the inner loop's encloser. The SYNC-DAG today is
// outer holdrule{gated(nested-final)} + inner holdrule{gated(non-ws-loop)};
// the if threads the carrier.
//
// This pins TODAY's all-gated emission. After M2, canDrop(If)=false keeps the
// inner loop GATED (clause (ii) of edit 1 finds the scf.if in the chain) and
// the only change is the dump gate-reason label (non-ws-loop -> if-encloser);
// the EMITTED IR here must stay BYTE-IDENTICAL. This is the only artifact that
// lets M2 verify the C3 claim ("if-encloser stays byte-identical, only the
// label changes"). Optimizing this shape is the UNSCHEDULED M4.2 follow-up
// (flip canDrop(If) to true + build the else pass-through).

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL:   tt.func @if_encloser_inner_loop
  tt.func @if_encloser_inner_loop(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %true = arith.constant true
    // Single-buffered (1x) ping-pong: alloc, create EMPTY/FULL pair, pre-loop acquire of EMPTY.
    // CHECK:           [[POISON:%.*]] = ub.poison : !ttg.async.token
    // CHECK:           [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[ACQ0:%.*]] = nvws.semaphore.acquire [[EMPTY]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %res, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // WS outer loop carries the poison data-token and the EMPTY acquire-token.
    // CHECK:           {{.*}}:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[OUTER_DATA:%.*]] = [[POISON]], [[OUTER_TOK:%.*]] = [[ACQ0]]) -> (!ttg.async.token, !ttg.async.token)  : i32 {
    %o = scf.for %iv0 = %lb to %ub step %step iter_args(%t0 = %tok) -> (!ttg.async.token) : i32 {
      // scf.if encloser threads the outer carriers into the then-branch.
      // CHECK:             [[IF:%.*]]:2 = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token, !ttg.async.token) {
      %r = scf.if %cond -> (!ttg.async.token) {
        // Inner non-WS loop carries the if-region carriers.
        // CHECK:               [[INNER:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[INNER_DATA:%.*]] = [[OUTER_DATA]], [[INNER_TOK:%.*]] = [[OUTER_TOK]]) -> (!ttg.async.token, !ttg.async.token)  : i32 {
        %i = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %t0) -> (!ttg.async.token) : i32 {
          // CHECK:                 [[SA:%.*]] = "loadA"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
          %sA = "loadA"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
          // CHECK:                 [[SB:%.*]] = "loadB"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
          %sB = "loadB"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
          // Producer (partition 1): buffer from EMPTY's carried token feeds the MMA, then release FULL.
          // CHECK:                 [[MMA_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[INNER_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK:                 ttng.tc_gen5_mma [[SA]], [[SB]], [[MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          %mma = ttng.tc_gen5_mma %sA, %sB, %res[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK:                 nvws.semaphore.release [[FULL]], [[INNER_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          // Consumer (partition 0): acquire FULL, buffer feeds the load, release EMPTY.
          // CHECK:                 [[ACQ_FULL:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK:                 [[LOAD_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[ACQ_FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK:                 %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[LOAD_BUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
          %val, %t2 = ttng.tmem_load %res[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          // CHECK:                 nvws.semaphore.release [[EMPTY]], [[ACQ_FULL]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          // CHECK:                 "use"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          // Re-acquire EMPTY for the next iteration's producer; carry it as the loop token.
          // CHECK:                 [[REACQ:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK:                 scf.yield {ttg.partition = array<i32: 0, 1>} [[POISON]], [[REACQ]] : !ttg.async.token, !ttg.async.token
          scf.yield {ttg.partition = array<i32: 0, 1>} %t2 : !ttg.async.token
        // CHECK:               } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
        // CHECK:               scf.yield {ttg.partition = array<i32: 0, 1>} [[INNER]]#0, [[INNER]]#1 : !ttg.async.token, !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %i : !ttg.async.token
      } else {
        // Else branch: pass the outer carriers through unchanged (no semaphore op).
        // CHECK:             } else {
        // CHECK:               scf.yield {ttg.partition = array<i32: 0, 1>} [[OUTER_DATA]], [[OUTER_TOK]] : !ttg.async.token, !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %t0 : !ttg.async.token
      // CHECK:             } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // CHECK:             scf.yield {ttg.partition = array<i32: 0, 1>} [[IF]]#0, [[IF]]#1 : !ttg.async.token, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %r : !ttg.async.token
    // CHECK:           } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    // CHECK:           tt.return
    tt.return
  }
}
