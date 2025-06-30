#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ws_tma(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.extsi %arg3 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg0, [%arg2, %arg3], [%2, %c1_i64] : <i16>, <tensor<64x64xsi16>>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xi16, #shared, #smem, mutable>
    %5 = ttg.memdesc_subview %4[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xi16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xi16, #shared, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %7 = ttg.memdesc_subview %6[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %8 = ttg.memdesc_subview %6[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %8, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %7, 8192, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %9 = arith.muli %0, %c64_i32 : i32
    %10 = arith.muli %1, %c64_i32 : i32
    ttg.warp_specialize(%7)
    default {
      ttng.wait_barrier %8, %c1_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %11 = tlx.require_layout %5 : !ttg.memdesc<64x64xi16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xi16, #shared2, #smem, mutable>
      %12 = ttng.tensor_desc_to_tma_ptr %3 : !tt.tensordesc<tensor<64x64xsi16>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %12[%9, %10] %11, %7, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xi16, #shared2, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg4: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      %c0_i32_0 = arith.constant 0 : i32
      ttng.wait_barrier %arg4, %c0_i32_0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(convert-triton-to-tritongpu{enable-source-remat=false num-ctas=1 num-warps=4 target=cuda:90 threads-per-warp=32}, tritongpu-coalesce, tlx-propagate-layout, tritongpu-F32DotTC, triton-nvidia-gpu-plan-cta, tritongpu-remove-layout-conversions, tritongpu-optimize-thread-locality, tritongpu-accelerate-matmul, tritongpu-remove-layout-conversions, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, triton-nvidia-optimize-descriptor-encoding, triton-loop-aware-cse, tritongpu-fuse-nested-loops, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, triton-licm, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, tritongpu-combine-tensor-select-and-if, tritongpu-assign-latencies{num-stages=3}, tritongpu-schedule-loops, tritongpu-pipeline{dump-intermediate-steps=true num-stages=3}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, triton-loop-aware-cse, tritongpu-prefetch, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, tritongpu-coalesce-async-copy, triton-nvidia-optimize-tmem-layouts, tritongpu-remove-layout-conversions, triton-nvidia-interleave-tmem, tritongpu-reduce-data-duplication, tritongpu-reorder-instructions, triton-loop-aware-cse, symbol-dce, triton-nvidia-tma-lowering, triton-nvidia-gpu-fence-insertion{compute-capability=90}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true})",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
