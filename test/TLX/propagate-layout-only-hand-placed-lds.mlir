// RUN: triton-opt -split-input-file --tlx-propagate-layout="only-hand-placed-lds=true" %s | FileCheck %s

// Plain Triton keeps its dot-operand staging.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @plain_triton_keeps_staging
  tt.func public @plain_triton_keeps_staging(%src: tensor<64x32xf16, #blocked>) -> tensor<64x32xf16, #dot> {
    // CHECK: ttg.local_alloc %
    // CHECK: ttg.local_load
    // CHECK-NOT: ttg.convert_layout
    %alloc = ttg.local_alloc %src : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #shared, #smem>
    %a = ttg.local_load %alloc : !ttg.memdesc<64x32xf16, #shared, #smem> -> tensor<64x32xf16, #dot>
    tt.return %a : tensor<64x32xf16, #dot>
  }
}

// -----

// TLX modules fold it.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @tlx_folds_staging
  tt.func public @tlx_folds_staging(%src: tensor<64x32xf16, #blocked>) -> tensor<64x32xf16, #dot> {
    // CHECK-NOT: ttg.local_alloc
    // CHECK: ttg.convert_layout %{{.*}} : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op
    %alloc = ttg.local_alloc %src : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #shared, #smem>
    %a = ttg.local_load %alloc : !ttg.memdesc<64x32xf16, #shared, #smem> -> tensor<64x32xf16, #dot>
    tt.return %a : tensor<64x32xf16, #dot>
  }
}

// -----

// So do warp-pipelined modules, which carry no TLX module attributes.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @warp_pipeline_folds_staging
  tt.func public @warp_pipeline_folds_staging(%src: tensor<64x32xf16, #blocked>) -> tensor<64x32xf16, #dot> {
    // CHECK-NOT: ttg.local_alloc
    // CHECK: ttg.convert_layout %{{.*}} : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op
    %alloc = ttg.local_alloc %src : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #shared, #smem>
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage"}
    %a = ttg.local_load %alloc : !ttg.memdesc<64x32xf16, #shared, #smem> -> tensor<64x32xf16, #dot>
    tt.return %a : tensor<64x32xf16, #dot>
  }
}
