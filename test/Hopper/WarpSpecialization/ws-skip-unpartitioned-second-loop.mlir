// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-partition-scheduling-meta --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448" | FileCheck %s

// The first loop gets partitions; partition scheduling skips the second one
// (no descriptor load, no MMA). Without the bail-out the second loop is cloned
// into every partition and its TMA reduce runs once per partition.

// CHECK-LABEL: @two_loops_skipped
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: nvws.descriptor_load
// CHECK: tt.descriptor_reduce
// CHECK-NOT: tt.descriptor_reduce
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#slice = #ttg.slice<{dim = 1, parent = #mma}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

tt.func public @two_loops_skipped(
  %a_desc: !tt.tensordesc<32x32xbf16, #shared>,
  %b_desc: !tt.tensordesc<32x64xbf16, #shared>,
  %y_desc: !tt.tensordesc<32x64xf32, #shared1>,
  %n_tiles: i32,
  %z_desc: !tt.tensordesc<32x64xf32, #shared1>,
  %x_ptr: !tt.ptr<f32>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %zero = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
  scf.for %i = %c0_i32 to %n_tiles step %c1_i32 : i32 {
    %offs = arith.muli %i, %c32_i32 : i32
    %acc = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_k = %zero) -> (tensor<32x64xf32, #mma>) : i32 {
      %offs_k = arith.muli %k, %c32_i32 : i32
      %a = tt.descriptor_load %a_desc[%offs, %offs_k] : !tt.tensordesc<32x32xbf16, #shared> -> tensor<32x32xbf16, #blocked>
      %b = tt.descriptor_load %b_desc[%offs_k, %c0_i32] : !tt.tensordesc<32x64xbf16, #shared> -> tensor<32x64xbf16, #blocked>
      %a_op = ttg.convert_layout %a : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %b_op = ttg.convert_layout %b : tensor<32x64xbf16, #blocked> -> tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %d = tt.dot %a_op, %b_op, %acc_k : tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x64xf32, #mma>
      scf.yield %d : tensor<32x64xf32, #mma>
    }
    %stage = ttg.local_alloc %acc : (tensor<32x64xf32, #mma>) -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>
    %tok = ttng.async_tma_copy_local_to_global %y_desc[%offs, %c0_i32] %stage : !tt.tensordesc<32x64xf32, #shared1>, !ttg.memdesc<32x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    scf.yield
  } {tt.warp_specialize, tt.separate_epilogue_store = true}
  scf.for %j = %c0_i32 to %n_tiles step %c1_i32 : i32 {
    %offs2 = arith.muli %j, %c32_i32 : i32
    %xp = tt.addptr %x_ptr, %offs2 : !tt.ptr<f32>, i32
    %xps = tt.splat %xp : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #mma>
    %xv = tt.load %xps : tensor<32x64x!tt.ptr<f32>, #mma>
    tt.descriptor_reduce add, %z_desc[%offs2, %c0_i32], %xv : !tt.tensordesc<32x64xf32, #shared1>, tensor<32x64xf32, #mma>
  } {tt.warp_specialize}
  tt.return
}


}
