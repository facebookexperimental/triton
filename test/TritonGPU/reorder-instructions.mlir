// RUN: triton-opt %s -split-input-file -tritongpu-reorder-instructions | FileCheck %s

// check that we don't hoist convert_layout above its operand definition.
// CHECK-LABEL: convert_cannot_hoist
//       CHECK:   %[[CVTS:.+]] = ttg.local_alloc
//       CHECK:   ttg.local_load %[[CVTS]]
//       CHECK:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @convert_cannot_hoist(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %10 = ttg.local_alloc %9 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %11 = ttg.local_load %10 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %11, %cst_0, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: no_move_alloc_for_scalar_src
//       CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
//       CHECK: %[[SPLAT:.*]] = tt.splat %{{.*}} : f32 -> tensor<32x32xf32, #blocked>
//       CHECK: ttg.async_wait {num = 0 : i32}
//       CHECK: ttg.local_alloc %[[SPLAT]] : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_move_alloc_for_scalar_src() {
    %cst = arith.constant 0.000000e+00 : f32
    %t = tt.splat %cst : f32 -> tensor<32x32xf32, #blocked>
    ttg.async_wait {num = 0 : i32}
    %alloc = ttg.local_alloc %t : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_dealloc
//       CHECK: ttg.async_wait {num = 0 : i32}
//       CHECK: ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
//       CHECK: ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
//       CHECK: %3 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_dealloc(%arg0: tensor<32x32xf32, #blocked>) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %2 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %3 = arith.addf %2, %2 : tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_idx_1
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_idx_1(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD = ttg.local_load %AS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_idx_1_negative
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #{{.*}}, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: ttng.arrive_barrier
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #{{.*}}, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_idx_1_negative(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    ttng.arrive_barrier %bar, 2, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %AD = ttg.local_load %AS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// check that we don't sink convert_layout if it has multi users
// CHECK-LABEL: convert_cannot_sink
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @convert_cannot_sink(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A0 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS0 = ttg.local_alloc %A0 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD0 = ttg.local_load %AS0 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD0, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %A1 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS1 = ttg.local_alloc %A1 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD1 = ttg.local_load %AS1 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %13 = tt.dot %AD1, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// With a register cap, an unpipelined loop issues both global loads before
// its first shared store.
// CHECK-LABEL: sink_loop_allocs_below_loads
//       CHECK: scf.for
//  CHECK-NEXT:   %[[A:.+]] = tt.load
//  CHECK-NEXT:   %[[B:.+]] = tt.load
//  CHECK-NEXT:   %[[AS:.+]] = ttg.local_alloc %[[A]]
//  CHECK-NEXT:   %[[BS:.+]] = ttg.local_alloc %[[B]]
//  CHECK-NEXT:   ttng.warp_group_dot %[[AS]], %[[BS]]
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.maxnreg" = 128 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_loop_allocs_below_loads(%a: tensor<64x64x!tt.ptr<f16>, #blocked>, %b: tensor<64x64x!tt.ptr<f16>, #blocked1>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %A = tt.load %a : tensor<64x64x!tt.ptr<f16>, #blocked>
      %AS = ttg.local_alloc %A : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %B = tt.load %b : tensor<64x64x!tt.ptr<f16>, #blocked1>
      %BS = ttg.local_alloc %B : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %d = ttng.warp_group_dot %AS, %BS, %acc : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %d : tensor<64x64xf32, #mma>
    }
    tt.return %r : tensor<64x64xf32, #mma>
  }
}

// -----

// Without a register cap, the same loop keeps each alloc after its load.
// CHECK-LABEL: no_sink_loop_allocs_without_maxnreg
//       CHECK: scf.for
//  CHECK-NEXT:   %[[A:.+]] = tt.load
//  CHECK-NEXT:   ttg.local_alloc %[[A]]
//  CHECK-NEXT:   %[[B:.+]] = tt.load
//  CHECK-NEXT:   ttg.local_alloc %[[B]]
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_sink_loop_allocs_without_maxnreg(%a: tensor<64x64x!tt.ptr<f16>, #blocked>, %b: tensor<64x64x!tt.ptr<f16>, #blocked1>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %A = tt.load %a : tensor<64x64x!tt.ptr<f16>, #blocked>
      %AS = ttg.local_alloc %A : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %B = tt.load %b : tensor<64x64x!tt.ptr<f16>, #blocked1>
      %BS = ttg.local_alloc %B : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %d = ttng.warp_group_dot %AS, %BS, %acc : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %d : tensor<64x64xf32, #mma>
    }
    tt.return %r : tensor<64x64xf32, #mma>
  }
}

// -----

// An alloc used before the second load stays after its own load.
// CHECK-LABEL: no_sink_loop_alloc_used_before_load
//       CHECK: scf.for
//  CHECK-NEXT:   %[[A:.+]] = tt.load
//  CHECK-NEXT:   %[[AS:.+]] = ttg.local_alloc %[[A]]
//  CHECK-NEXT:   ttng.warp_group_dot %[[AS]]
//  CHECK-NEXT:   %[[B:.+]] = tt.load
//  CHECK-NEXT:   ttg.local_alloc %[[B]]
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.maxnreg" = 128 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_sink_loop_alloc_used_before_load(%a: tensor<64x64x!tt.ptr<f16>, #blocked>, %b: tensor<64x64x!tt.ptr<f16>, #blocked1>, %bs0: !ttg.memdesc<64x64xf16, #shared, #smem>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %A = tt.load %a : tensor<64x64x!tt.ptr<f16>, #blocked>
      %AS = ttg.local_alloc %A : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %d0 = ttng.warp_group_dot %AS, %bs0, %acc : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
      %B = tt.load %b : tensor<64x64x!tt.ptr<f16>, #blocked1>
      %BS = ttg.local_alloc %B : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %d1 = ttng.warp_group_dot %AS, %BS, %d0 : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %d1 : tensor<64x64xf32, #mma>
    }
    tt.return %r : tensor<64x64xf32, #mma>
  }
}
