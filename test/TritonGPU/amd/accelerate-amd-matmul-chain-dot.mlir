// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=16" | FileCheck %s --check-prefixes MFMA16,CHECK,CHAIN
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=32" | FileCheck %s --check-prefixes MFMA32,CHECK,CHAIN
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx950 matrix-instruction-size=32" | FileCheck %s --check-prefixes CHECK-GFX950,CHAIN
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx950 matrix-instruction-size=16" | FileCheck %s --check-prefixes CHECK-GFX950,CHAIN

// Accumulating into one loop result does not create a chain through the
// independent A/B operands returned by that same loop.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHAIN: #mma = #ttg.amd_mfma<{{.*}}warpsPerCTA = [2, 2]
// CHAIN-LABEL: @independent_loop_results
// CHAIN: tt.dot {{.*}} -> tensor<256x256xf32, #mma>
// CHAIN: tt.dot {{.*}} -> tensor<256x256xf32, #mma>
// CHAIN: tt.dot {{.*}} -> tensor<256x256xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @independent_loop_results(
      %a: tensor<256x32xf16, #dotOp0>, %b: tensor<32x256xf16, #dotOp1>,
      %lb: index, %ub: index, %step: index) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %first = tt.dot %a, %b, %zero : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #blocked>
    %r:3 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %first, %a_iter = %a, %b_iter = %b)
        -> (tensor<256x256xf32, #blocked>, tensor<256x32xf16, #dotOp0>, tensor<32x256xf16, #dotOp1>) {
      %next = tt.dot %a_iter, %b_iter, %acc : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #blocked>
      %next_a = arith.addf %a_iter, %a : tensor<256x32xf16, #dotOp0>
      %next_b = arith.addf %b_iter, %b : tensor<32x256xf16, #dotOp1>
      scf.yield %next, %next_a, %next_b : tensor<256x256xf32, #blocked>, tensor<256x32xf16, #dotOp0>, tensor<32x256xf16, #dotOp1>
    }
    %last = tt.dot %r#1, %r#2, %r#0 : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #blocked>
    tt.return %last : tensor<256x256xf32, #blocked>
  }
}

// -----

// Real operand dependencies must survive both the loop and conditional yields.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHAIN: #mma = #ttg.amd_mfma<{{.*}}warpsPerCTA = [4, 1]
// CHAIN-LABEL: @chain_through_loop_and_if
// CHAIN: tt.dot {{.*}} -> tensor<128x128xf32, #mma>
// CHAIN: tt.dot {{.*}} -> tensor<128x128xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @chain_through_loop_and_if(
      %a: tensor<128x128xf16, #dotOp0>, %b: tensor<128x128xf16, #dotOp1>,
      %lb: index, %ub: index, %step: index, %pred: i1) -> tensor<128x128xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %first = tt.dot %a, %b, %zero : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    %half = arith.truncf %first : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %p = ttg.convert_layout %half : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #dotOp0>
    %r:2 = scf.for %iv = %lb to %ub step %step iter_args(%p_iter = %p, %other = %a)
        -> (tensor<128x128xf16, #dotOp0>, tensor<128x128xf16, #dotOp0>) {
      %next_p = arith.addf %p_iter, %p : tensor<128x128xf16, #dotOp0>
      %next_other = arith.addf %other, %a : tensor<128x128xf16, #dotOp0>
      scf.yield %next_p, %next_other : tensor<128x128xf16, #dotOp0>, tensor<128x128xf16, #dotOp0>
    }
    %selected = scf.if %pred -> tensor<128x128xf16, #dotOp0> {
      scf.yield %r#0 : tensor<128x128xf16, #dotOp0>
    } else {
      %sum = arith.addf %r#0, %r#1 : tensor<128x128xf16, #dotOp0>
      scf.yield %sum : tensor<128x128xf16, #dotOp0>
    }
    %last = tt.dot %selected, %b, %zero : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    tt.return %last : tensor<128x128xf32, #blocked>
  }
}

// -----

// An if's accumulator result must not contaminate its independent A result.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHAIN: #mma = #ttg.amd_mfma<{{.*}}warpsPerCTA = [2, 2]
// CHAIN-LABEL: @independent_if_results
// CHAIN: tt.dot {{.*}} -> tensor<256x256xf32, #mma>
// CHAIN: tt.dot {{.*}} -> tensor<256x256xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @independent_if_results(
      %a: tensor<256x32xf16, #dotOp0>, %b: tensor<32x256xf16, #dotOp1>,
      %pred: i1) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %first = tt.dot %a, %b, %zero : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #blocked>
    %r:2 = scf.if %pred -> (tensor<256x256xf32, #blocked>, tensor<256x32xf16, #dotOp0>) {
      scf.yield %first, %a : tensor<256x256xf32, #blocked>, tensor<256x32xf16, #dotOp0>
    } else {
      %next_a = arith.addf %a, %a : tensor<256x32xf16, #dotOp0>
      scf.yield %first, %next_a : tensor<256x256xf32, #blocked>, tensor<256x32xf16, #dotOp0>
    }
    %last = tt.dot %r#1, %b, %r#0 : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #blocked>
    tt.return %last : tensor<256x256xf32, #blocked>
  }
}

// -----

// A while forwards condition arguments to its results even on zero iterations.
// The before/after regions deliberately use different types and value orders.
// A nested execute_region also exercises the generic region-branch mapping.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHAIN: #mma = #ttg.amd_mfma<{{.*}}warpsPerCTA = [4, 1]
// CHAIN-LABEL: @chain_through_while
// CHAIN: tt.dot {{.*}} -> tensor<128x128xf32, #mma>
// CHAIN: tt.dot {{.*}} -> tensor<128x128xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @chain_through_while(
      %a: tensor<128x128xf16, #dotOp0>, %b: tensor<128x128xf16, #dotOp1>,
      %limit: i32) -> tensor<128x128xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %first = tt.dot %a, %b, %zero : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    %r:2 = scf.while (%i = %i0, %acc = %first) : (i32, tensor<128x128xf32, #blocked>) -> (tensor<128x128xf16, #dotOp0>, i32) {
      %p = scf.execute_region -> tensor<128x128xf16, #dotOp0> {
        %half = arith.truncf %acc : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %converted = ttg.convert_layout %half : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #dotOp0>
        scf.yield %converted : tensor<128x128xf16, #dotOp0>
      }
      %cond = arith.cmpi slt, %i, %limit : i32
      scf.condition(%cond) %p, %i : tensor<128x128xf16, #dotOp0>, i32
    } do {
    ^bb0(%p: tensor<128x128xf16, #dotOp0>, %i: i32):
      %next_i = arith.addi %i, %i1 : i32
      %half = ttg.convert_layout %p : tensor<128x128xf16, #dotOp0> -> tensor<128x128xf16, #blocked>
      %acc = arith.extf %half : tensor<128x128xf16, #blocked> to tensor<128x128xf32, #blocked>
      scf.yield %next_i, %acc : i32, tensor<128x128xf32, #blocked>
    }
    %last = tt.dot %r#0, %b, %zero : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    tt.return %last : tensor<128x128xf32, #blocked>
  }
}

// -----

// The operand dependency exists only through the while back-edge. Visit the
// tail before the head so its layout also exercises the backward traversal.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHAIN: #mma = #ttg.amd_mfma<{{.*}}warpsPerCTA = [4, 1]
// CHAIN-LABEL: @chain_through_while_backedge
// CHAIN: tt.dot {{.*}} -> tensor<128x128xf32, #mma>
// CHAIN: tt.dot {{.*}} -> tensor<128x128xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @chain_through_while_backedge(
      %a: tensor<128x128xf16, #dotOp0>, %b: tensor<128x128xf16, #dotOp1>,
      %limit: i32) -> tensor<128x128xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %r:3 = scf.while (%i = %i0, %p = %a, %acc = %zero) : (i32, tensor<128x128xf16, #dotOp0>, tensor<128x128xf32, #blocked>) -> (i32, tensor<128x128xf16, #dotOp0>, tensor<128x128xf32, #blocked>) {
      %cond = arith.cmpi slt, %i, %limit : i32
      scf.condition(%cond) %i, %p, %acc : i32, tensor<128x128xf16, #dotOp0>, tensor<128x128xf32, #blocked>
    } do {
    ^bb0(%i: i32, %p: tensor<128x128xf16, #dotOp0>, %acc: tensor<128x128xf32, #blocked>):
      %tail = tt.dot %p, %b, %acc : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
      %head = tt.dot %a, %b, %zero : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
      %half = arith.truncf %head : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      %next_p = ttg.convert_layout %half : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #dotOp0>
      %next_i = arith.addi %i, %i1 : i32
      scf.yield %next_i, %next_p, %tail : i32, tensor<128x128xf16, #dotOp0>, tensor<128x128xf32, #blocked>
    }
    tt.return %r#2 : tensor<128x128xf32, #blocked>
  }
}

// -----

// Reordered condition arguments must not make the accumulator a dependency of
// the independent A/B results, including through subsequent loop iterations.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHAIN: #mma = #ttg.amd_mfma<{{.*}}warpsPerCTA = [2, 2]
// CHAIN-LABEL: @independent_while_results
// CHAIN: tt.dot {{.*}} -> tensor<256x256xf32, #mma>
// CHAIN: tt.dot {{.*}} -> tensor<256x256xf32, #mma>
// CHAIN: tt.dot {{.*}} -> tensor<256x256xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @independent_while_results(
      %a: tensor<256x32xf16, #dotOp0>, %b: tensor<32x256xf16, #dotOp1>,
      %limit: i32) -> tensor<256x256xf32, #blocked> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #blocked>
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %first = tt.dot %a, %b, %zero : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #blocked>
    %r:4 = scf.while (%i = %i0, %acc = %first, %a_iter = %a, %b_iter = %b) : (i32, tensor<256x256xf32, #blocked>, tensor<256x32xf16, #dotOp0>, tensor<32x256xf16, #dotOp1>) -> (tensor<256x32xf16, #dotOp0>, tensor<32x256xf16, #dotOp1>, tensor<256x256xf32, #blocked>, i32) {
      %cond = arith.cmpi slt, %i, %limit : i32
      scf.condition(%cond) %a_iter, %b_iter, %acc, %i : tensor<256x32xf16, #dotOp0>, tensor<32x256xf16, #dotOp1>, tensor<256x256xf32, #blocked>, i32
    } do {
    ^bb0(%a_iter: tensor<256x32xf16, #dotOp0>, %b_iter: tensor<32x256xf16, #dotOp1>, %acc: tensor<256x256xf32, #blocked>, %i: i32):
      %next = tt.dot %a_iter, %b_iter, %acc : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #blocked>
      %next_a = arith.addf %a_iter, %a : tensor<256x32xf16, #dotOp0>
      %next_b = arith.addf %b_iter, %b : tensor<32x256xf16, #dotOp1>
      %next_i = arith.addi %i, %i1 : i32
      scf.yield %next_i, %next, %next_a, %next_b : i32, tensor<256x256xf32, #blocked>, tensor<256x32xf16, #dotOp0>, tensor<32x256xf16, #dotOp1>
    }
    %last = tt.dot %r#0, %r#1, %r#2 : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #blocked>
    tt.return %last : tensor<256x256xf32, #blocked>
  }
}

// -----

// Check the warpsPerCTA parameter of #mma layout of the two dot's.
// The 1st dot always has warpsPerCTA = [4, 1].
// The warpsPerCTA for the 2nd dot depends on mfma instruction size and BLOCK_M size.


// BLOCK_M = 128
// warpsPerCTA = [4, 1] for mfma16 and mfma32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK-LABEL: mfma_chain_dot_BM128
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<128x16xf32, #mma>
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<128x128xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_BM128(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %o_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<128x16xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst1 : tensor<128x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// BLOCK_M = 64
// warpsPerCTA = [4, 1] for mfma16
// warpsPerCTA = [2, 2] for mfma32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// MFMA32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK-LABEL: mfma_chain_dot_BM64
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<64x16xf32, #mma>
// MFMA16: tt.dot {{.*}} : {{.*}} -> tensor<64x128xf32, #mma>
// MFMA32: tt.dot {{.*}} : {{.*}} -> tensor<64x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_BM64(
      %q: tensor<64x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %o_ptr: tensor<64x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<64x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<64x16xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<64x16xf32, #blocked> to tensor<64x16xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst1 : tensor<64x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<64x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<64x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// BLOCK_M = 32
// warpsPerCTA = [2, 2] for mfma16
// warpsPerCTA = [1, 4] for mfma32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// MFMA16{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK-LABEL: mfma_chain_dot_BM32
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<32x16xf32, #mma>
// MFMA16: tt.dot {{.*}} : {{.*}} -> tensor<32x128xf32, #mma1>
// MFMA32: tt.dot {{.*}} : {{.*}} -> tensor<32x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_BM32(
      %q: tensor<32x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %o_ptr: tensor<32x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<32x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<32x16xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<32x16xf32, #blocked> to tensor<32x16xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<32x16xf16, #blocked> -> tensor<32x16xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst1 : tensor<32x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<32x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<32x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// BLOCK_M = 16, only check mfma16 since it's too small for mfma32
// warpsPerCTA = [1, 4] for mfma16
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA16{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = true}>
// CHECK-LABEL: mfma_chain_dot_BM16
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<16x16xf32, #mma>
// MFMA16: tt.dot {{.*}} : {{.*}} -> tensor<16x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_BM16(
      %q: tensor<16x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %o_ptr: tensor<16x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<16x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<16x16xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst1 : tensor<16x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<16x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<16x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// Check kWidth of both operands of the 2nd dot. To avoid in-warp shuffle for
// the layout conversion from #mma to #dotOp, kWidth should be set to 4

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK-LABEL: mfma_chain_dot_kWidth_f16
// CHECK-GFX950: tt.dot {{.*}} : {{.*}} -> tensor<128x128xf32, #mma>
// CHECK-GFX950: tt.dot {{.*}} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> {{.*}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_kWidth_f16(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x128xf16, #dotOp1>,
      %v: tensor<128x128xf16, #dotOp1>,
      %o_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK-LABEL: mfma_chain_dot_kWidth_bf16
// CHECK-GFX950: tt.dot {{.*}} : {{.*}} -> tensor<128x128xf32, #mma>
// CHECK-GFX950: tt.dot {{.*}} : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> {{.*}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_kWidth_bf16(
      %q: tensor<128x128xbf16, #dotOp0>,
      %k: tensor<128x128xbf16, #dotOp1>,
      %v: tensor<128x128xbf16, #dotOp1>,
      %o_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xbf16, #dotOp0> * tensor<128x128xbf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    %qk_bf16 = arith.truncf %qk :  tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %p = ttg.convert_layout %qk_bf16 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #dotOp0>
    %o = tt.dot %p, %v, %cst : tensor<128x128xbf16, #dotOp0> * tensor<128x128xbf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// Cross-iteration chain-dot: in pipelined FA the QK dot result is yielded and
// consumed as operand A of the PV dot on the next iteration.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK-GFX950-LABEL: cross_iter_chain_dot_fa
// CHECK-GFX950: tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
// CHECK-GFX950: tt.dot {{.*}} -> tensor<128x64xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @cross_iter_chain_dot_fa(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x64xf16, #dotOp1>,
      %v: tensor<64x128xf16, #dotOp1>,
      %acc_init: tensor<128x128xf32, #blocked>,
      %p_init: tensor<128x64xf16, #dotOp0>,
      %lb: index, %ub: index, %step: index) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %result:2 = scf.for %iv = %lb to %ub step %step
        iter_args(%acc = %acc_init, %p_prev = %p_init) -> (tensor<128x128xf32, #blocked>, tensor<128x64xf16, #dotOp0>) {
      %pv = tt.dot %p_prev, %v, %acc : tensor<128x64xf16, #dotOp0> * tensor<64x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
      %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x64xf16, #dotOp1> -> tensor<128x64xf32, #blocked>
      %qk_f16 = arith.truncf %qk : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
      %p_next = ttg.convert_layout %qk_f16 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dotOp0>
      scf.yield %pv, %p_next : tensor<128x128xf32, #blocked>, tensor<128x64xf16, #dotOp0>
    }
    tt.return %result#0 : tensor<128x128xf32, #blocked>
  }
}


// -----

// Cross-iteration chain-dot: in pipelined FA the QK dot result is yielded and
// consumed as operand A of the PV dot on the next iteration.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK-GFX950-LABEL: cross_iter_chain_dot_fa
// CHECK-GFX950: tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
// CHECK-GFX950: tt.dot {{.*}} -> tensor<128x64xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @cross_iter_chain_dot_fa(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x64xf16, #dotOp1>,
      %v: tensor<64x128xf16, #dotOp1>,
      %acc_init: tensor<128x128xf32, #blocked>,
      %p_init: tensor<128x64xf16, #dotOp0>,
      %lb: index, %ub: index, %step: index) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %result:2 = scf.for %iv = %lb to %ub step %step
        iter_args(%acc = %acc_init, %p_prev = %p_init) -> (tensor<128x128xf32, #blocked>, tensor<128x64xf16, #dotOp0>) {
      %pv = tt.dot %p_prev, %v, %acc : tensor<128x64xf16, #dotOp0> * tensor<64x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
      %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x64xf16, #dotOp1> -> tensor<128x64xf32, #blocked>
      %qk_f16 = arith.truncf %qk : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
      %p_next = ttg.convert_layout %qk_f16 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dotOp0>
      scf.yield %pv, %p_next : tensor<128x128xf32, #blocked>, tensor<128x64xf16, #dotOp0>
    }
    tt.return %result#0 : tensor<128x128xf32, #blocked>
  }
}
