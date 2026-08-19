// RUN: triton-opt %s -split-input-file --nvgpu-warp-specialization="num-stages=3 capability=90" | FileCheck %s

// Tests that warp specialization bails out before doConvertDescriptorLoadsToNVWS
// when PartitionSchedulingMeta assigned no partitions.
//
// The loop below carries the `tt.warp_specialize` marker, so the pass is
// entered, but nothing carries a `ttg.partition` / `async_task_id`: PSM found
// no MMA to build producer/consumer roles around, because the `tt.dot` is an
// IEEE fp32 dot that lowers to FMA rather than wgmma. Specialization needs at
// least two partitions (producer + consumer), so the pass must reject here.
//
// Without the gate, doConvertDescriptorLoadsToNVWS still rewrites the
// `tt.descriptor_load` into an `nvws.descriptor_load`. Nothing erases it --
// optimizeTMALoads only runs from insertAsyncComm during code partitioning,
// which creates no channels without partitions, and there is no rollback path.
// The stranded NVWS op then reaches LLVM translation and fails as
// "LLVM Translation failed for operation: builtin.unrealized_conversion_cast"
// on the backward casts feeding it.
//
// CHECK-LABEL: @no_partitions_descriptor_load
// The descriptor load must survive unconverted, for ordinary TMA lowering.
// CHECK: tt.descriptor_load
// CHECK-NOT: nvws.descriptor_load
// CHECK-NOT: ttg.warp_specialize
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func public @no_partitions_descriptor_load(%desc: !tt.tensordesc<16x16xf32, #shared>, %acc0: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %result = scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%acc = %acc0) -> (tensor<16x16xf32, #blocked>) : i32 {
      %a = tt.descriptor_load %desc[%iv, %c0_i32] : !tt.tensordesc<16x16xf32, #shared> -> tensor<16x16xf32, #blocked>
      %sum = arith.addf %acc, %a : tensor<16x16xf32, #blocked>
      scf.yield %sum : tensor<16x16xf32, #blocked>
    } {tt.warp_specialize}
    tt.return %result : tensor<16x16xf32, #blocked>
  }
}
