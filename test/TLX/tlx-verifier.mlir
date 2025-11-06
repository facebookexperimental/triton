
// RUN: triton-opt -split-input-file -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=cuda:90 num-ctas=1 threads-per-warp=32})' --verify-diagnostics %s

module attributes {tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @legalize_warp_partition(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    // expected-error @+1 {{WarpSpecializeOp should not capture RankedTensorType}}
    ttg.warp_specialize(%arg3, %3, %arg5)
    default {
      %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %4 = arith.addi %3, %2 : tensor<1024xi32>
      ttg.warp_yield
    }
    partition0(%arg7: !tt.ptr<f32>, %arg8: tensor<1024xi32>, %arg9: !tt.ptr<f32>) num_warps(1) {
      %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %4 = arith.addi %arg8, %2 : tensor<1024xi32>
      %5 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %8 = tt.splat %arg9 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      ttg.warp_return
    } : (!tt.ptr<f32>, tensor<1024xi32>, !tt.ptr<f32>) -> ()
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @tc_gen5_mma(%a: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async, two_ctas}:
       !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    // expected-error @+1 {{Expecting all dot ops to be 2cta together}}
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async}:
           !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
           !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
           !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
           !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @map_smem_to_remote(%arg: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    %c1_i32 = arith.constant 1 : i32
    // expected-error @+1 {{Unexpected buffer remote view in 1cta mode}}
    %0 = ttng.map_to_remote_buffer %arg, %c1_i32: !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }
}
