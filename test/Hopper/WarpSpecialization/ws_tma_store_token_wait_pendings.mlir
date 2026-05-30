// RUN: triton-opt %s -split-input-file --nvgpu-tma-store-token-wait-lowering | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Direct case: no intervening stores → pendings = 0
// CHECK-LABEL: direct_no_intervening
  tt.func public @direct_no_intervening(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Loop-carried token after an unrelated loop-carried value. The wait token's
// block argument must map to the token yield operand, not the dead value.
// CHECK-LABEL: loop_carried_token_with_dead_iter_arg
  tt.func public @loop_carried_token_with_dead_iter_arg(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %init_tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %result:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%dead = %i, %carried = %init_tok) -> (i32, !ttg.async.token) {
      %tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      ttng.async_tma_store_token_wait %carried : !ttg.async.token
      scf.yield %dead, %tok : i32, !ttg.async.token
    }
    // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.async_tma_store_token_wait %result#1 : !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Direct case: 1 intervening store → pendings = 1 for first, 0 for second
// CHECK-LABEL: direct_one_intervening
  tt.func public @direct_one_intervening(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
    ttng.async_tma_store_token_wait %tok0 : !ttg.async.token
    // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.async_tma_store_token_wait %tok1 : !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Loop-carried case: wait at top, 2 stores, yield first token.
// After tok0 there is 1 store (tok1) before end of body, and 0 stores before
// the wait at the top → pendings = 1.
// CHECK-LABEL: loop_carried
  tt.func public @loop_carried(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // Create an initial token for the loop.
    %init_tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %result = scf.for %iv = %c0 to %c8 step %c1 iter_args(%carried = %init_tok) -> (!ttg.async.token) {
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      ttng.async_tma_store_token_wait %carried : !ttg.async.token
      scf.yield %tok0 : !ttg.async.token
    }
    // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.async_tma_store_token_wait %result : !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Option C: planner-annotated rotation. The wait carries
// can_rotate_by_buffer_count = 2 from the memory planner (buffer.copy = 2),
// but the source order has 0 stores between the defining store and the
// wait — this happens for WS partition loops where SWP ExpandLoops does
// not source-reorder the body, so source-order counting alone would yield
// pendings = 0 (full drain) and waste the rotating slots. The lowering
// must trust the planner's K-1 intent and emit pendings = 1.
// CHECK-LABEL: rotate_by_buffer_count_trust_planner
  tt.func public @rotate_by_buffer_count_trust_planner(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
    ttng.async_tma_store_token_wait %tok {can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Option C is gated to only fire when a defining store is matched.
// Here the token's defining op is the scf.for itself (a loop result), not a
// TMA store-like op — so neither the direct nor the loop-carried trace
// matches, and we keep the conservative pendings = 0 (full drain) for the
// kernel-exit drain, even with the annotation present.
// CHECK-LABEL: rotate_annotation_outside_loop_still_drains
  tt.func public @rotate_annotation_outside_loop_still_drains(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %init_tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %result = scf.for %iv = %c0 to %c8 step %c1 iter_args(%carried = %init_tok) -> (!ttg.async.token) {
      %tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %tok : !ttg.async.token
    }
    // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.async_tma_store_token_wait %result {can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token
    tt.return
  }
}
