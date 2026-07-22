// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1" | FileCheck %s

// Simplified (no-attention) audit of the chained operand-D accumulator channel
// behavior in handleOperandD (T278685041). Two MMAv5 ops write the SAME tmem
// accumulator in one loop (acc = A@B; acc += C@D). Both are in the gemm
// partition (task 1); a tmem_load of the accumulator is in the computation
// partition (task 0). This is the minimal load/mma/store shape (no attention)
// requested in the D111277168 review, exercising both the "no intermediate" and
// "with intermediate output" cases and showing the produced channels.
//
// How the channel is (not) created between the two MMAs: handleOperandD walks
// every op whose accumulator IS this tmem tile (each is a D-writer / chain link)
// and calls needsChannel(producerTask, consumerTask) for consecutive writers.
// Both dots carry the same ttg.partition (gemm, task 1), so needsChannel is
// false for that pair -> NO MMA->MMA channel; the second dot just extends the
// producer set (chains in place). A channel is emitted ONLY when the accumulator
// is read from a DIFFERENT partition, i.e. a tmem_load in the computation
// partition (task 0) -> a cross-partition MMA->tmem_load channel. So an
// intermediate read between the dots does not create an MMA->MMA channel; it
// creates an extra cross-partition read channel (case 2 below).

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32} {

// ---- Case 1: chained accumulator, NO intermediate output between the dots ----
// The two dots share one gemm partition (partition0); the accumulator handoff
// between them is same-partition, so NO MMA->MMA channel (wait_barrier) is
// emitted. The only channel is the cross-partition MMA -> tmem_load read into
// the computation partition (default region: wait_barrier, tmem_load).
//
// CHECK-LABEL: @chained_accum_no_intermediate
// CHECK: ttg.warp_specialize
// computation partition (default region): waits on the one cross-partition
// accumulator channel, then reads it.
// CHECK: ttng.wait_barrier
// CHECK: ttng.tmem_load
// gemm partition: both chained dots, with NO wait_barrier (no MMA->MMA channel)
// between them.
// CHECK: partition0
// CHECK: ttng.tc_gen5_mma
// CHECK-NOT: ttng.wait_barrier
// CHECK: ttng.tc_gen5_mma
  tt.func public @chained_accum_no_intermediate(
      %a: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>,
      %b: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>,
      %c: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>,
      %d: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>,
      %o: !ttg.memdesc<128x128xf32, #shared32, #smem, mutable>,
      %lb: i32, %ub: i32, %step: i32) attributes {noinline = false} {
    %true = arith.constant {ttg.partition = array<i32: 0, 1>} true
    %false = arith.constant {ttg.partition = array<i32: 0, 1>} false
    %acc, %acc_tok = ttng.tmem_alloc {ttg.partition = array<i32: 1>, buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %iv = %lb to %ub step %step iter_args(%tok = %acc_tok) -> (!ttg.async.token) : i32 {
      // MMA1: use_acc = false -> first producer, overwrites the accumulator.
      %t1 = ttng.tc_gen5_mma %a, %b, %acc[%tok], %false, %true {ttg.partition = array<i32: 1>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // MMA2: use_acc = true into the SAME acc -> chained; same task -> no channel.
      %t2 = ttng.tc_gen5_mma %c, %d, %acc[%t1], %true, %true {ttg.partition = array<i32: 1>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // cross-partition read of the finished accumulator (computation, task 0)
      %val, %t3 = ttng.tmem_load %acc[%t2] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      ttg.local_store %val, %o {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #shared32, #smem, mutable>
      scf.yield %t3 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1>}
    tt.return
  }

// ---- Case 2: an intermediate tmem_load of the acc BETWEEN the two dots ----
// The intermediate read (task 0) after the first dot forces a cross-partition
// channel; the second dot still chains into the same accumulator in the gemm
// partition (still NO MMA->MMA channel), and the final read is a second
// cross-partition channel. So: two tmem_loads (two channels), two dots in one
// partition.
//
// CHECK-LABEL: @chained_accum_with_intermediate
// CHECK: ttg.warp_specialize
// two cross-partition accumulator reads in the computation partition:
// CHECK: ttng.tmem_load
// CHECK: ttng.tmem_load
// both chained dots in the gemm partition, no MMA->MMA channel between them:
// CHECK: partition0
// CHECK: ttng.tc_gen5_mma
// CHECK-NOT: ttng.wait_barrier
// CHECK: ttng.tc_gen5_mma
  tt.func public @chained_accum_with_intermediate(
      %a: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>,
      %b: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>,
      %c: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>,
      %d: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>,
      %o0: !ttg.memdesc<128x128xf32, #shared32, #smem, mutable>,
      %o1: !ttg.memdesc<128x128xf32, #shared32, #smem, mutable>,
      %lb: i32, %ub: i32, %step: i32) attributes {noinline = false} {
    %true = arith.constant {ttg.partition = array<i32: 0, 1>} true
    %false = arith.constant {ttg.partition = array<i32: 0, 1>} false
    %cst_half = arith.constant {ttg.partition = array<i32: 0>} dense<5.000000e-01> : tensor<128x128xf32, #blocked>
    %acc, %acc_tok = ttng.tmem_alloc {ttg.partition = array<i32: 1>, buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %iv = %lb to %ub step %step iter_args(%tok = %acc_tok) -> (!ttg.async.token) : i32 {
      %t1 = ttng.tc_gen5_mma %a, %b, %acc[%tok], %false, %true {ttg.partition = array<i32: 1>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // intermediate read of the partial accumulator (task 0), BETWEEN the dots,
      // with a compute + store on it (mirrors `output = acc / 2.0; store(output)`).
      %mid, %t2 = ttng.tmem_load %acc[%t1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %half = arith.mulf %mid, %cst_half {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked>
      ttg.local_store %half, %o0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #shared32, #smem, mutable>
      // second dot chains into the same acc, same task -> still no MMA->MMA channel
      %t3 = ttng.tc_gen5_mma %c, %d, %acc[%t2], %true, %true {ttg.partition = array<i32: 1>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %val, %t4 = ttng.tmem_load %acc[%t3] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      ttg.local_store %val, %o1 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #shared32, #smem, mutable>
      scf.yield %t4 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1>}
    tt.return
  }
}
