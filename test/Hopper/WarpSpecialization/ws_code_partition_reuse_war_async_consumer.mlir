// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" | FileCheck %s

// A TMEM reuse-group channel that ping-pongs through async MMAs must keep its
// WAR reuse barrier; needExplicitReuseWait must NOT elide it just because the
// producer and consumer are in the same partition.
//
// Minimal pattern (not a full attention kernel): two channels share one
// single-buffered TMEM allocation (same buffer.id; the sibling at
// buffer.offset 0). The owner is written by an async tc_gen5_mma with
// useC=false (a full overwrite, "QK") in the gemm partition; the sibling is
// read by a second async tc_gen5_mma ("PV") in the *same* gemm partition. The
// PV read only *issues* in program order and drains asynchronously, so program
// order does not guarantee it finishes before the next QK overwrite. Code
// partitioning must therefore gate the QK MMA with a backward (P-empty) reuse
// acquire (dstTask = 2 is the gemm partition). Without the fix the barrier is
// elided and the CHECK-NEXT below does not match.

// CHECK-LABEL: tt.func public @reuse_war_async
// CHECK: ttng.wait_barrier {{.*}}direction = "backward"{{.*}}dstTask = 2 : i32{{.*}}ttg.partition = array<i32: 2>
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, %false, %true,

#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reuse_war_async(%desc_k: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared>>) attributes {noinline = false} {
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 0 : i32
    %c128_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 128 : i32
    %c1024_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 1024 : i32
    %true = arith.constant {ttg.partition = array<i32: 0, 2>} true
    %false = arith.constant {ttg.partition = array<i32: 2>} false
    // QK operand A (Q in SMEM), loaded once.
    %q = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // PV accumulator (operand D).
    %acc, %acc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // K / V SMEM operands.
    %k = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // TMEM reuse group (shared buffer.id): the QK result (owner) and P (reuser at offset 0).
    %qk, %qk_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 6 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %p = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 6 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %qload = tt.descriptor_load %desc_k[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
    ttg.local_store %qload, %q {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    scf.for %t = %c0_i32 to %c128_i32 step %c128_i32  : i32 {
      %r:4 = scf.for %i = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%off = %c0_i32, %uc = %false, %qkt = %qk_tok, %at = %acc_tok) -> (i32, i1, !ttg.async.token, !ttg.async.token)  : i32 {
        // load K (stage 0) and V (stage 1) into SMEM
        %kl = tt.descriptor_load %desc_k[%off, %c0_i32] {ttg.partition = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
        ttg.local_store %kl, %k {ttg.partition = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %kt = ttg.memdesc_trans %k {ttg.partition = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        %vl = tt.descriptor_load %desc_v[%off, %c0_i32] {ttg.partition = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
        ttg.local_store %vl, %v {ttg.partition = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // QK MMA: writes the reuse owner (operand D, useC=false => full overwrite), stage 0.
        %qkmma = ttng.tc_gen5_mma %q, %kt, %qk[%qkt], %false, %true {ttg.partition = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndB,smem,2,0\22]}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // computation reads the QK result (channel A consumer) and writes P into the
        // reuse sibling (channel B producer), stage 1.
        %qkres, %qkt2 = ttng.tmem_load %qk[%qkmma] {ttg.partition = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %pf16 = arith.truncf %qkres {ttg.partition = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        ttng.tmem_store %pf16, %p, %true {ttg.partition = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        // PV MMA: reads P (operand A) async, same (gemm) partition as the QK MMA, stage 1.
        %pvmma = ttng.tc_gen5_mma %p, %v, %acc[%at], %uc, %true {ttg.partition = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndB,smem,2,1\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %off2 = arith.addi %off, %c128_i32 {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
        scf.yield {ttg.partition = array<i32: 0>} %off2, %true, %qkt2, %pvmma : i32, i1, !ttg.async.token, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2>, tt.data_partition_factor = 2 : i32, tt.merge_correction = true, tt.merge_epilogue = true, tt.scheduled_max_stage = 1 : i32}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>}
    } {ttg.partition = array<i32: 0, 1, 2>, tt.data_partition_factor = 2 : i32, tt.merge_correction = true, tt.merge_epilogue = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32], ttg.partition.types = ["computation", "load", "gemm"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
