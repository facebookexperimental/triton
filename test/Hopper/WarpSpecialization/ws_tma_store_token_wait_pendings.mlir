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
// scf.for result tokens use the minimum of the loop-yield path and the
// zero-iteration initial iter_arg path.
// CHECK-LABEL: for_result_token_drain
  tt.func public @for_result_token_drain(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %init0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %init1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %init2 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src2 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %result:3 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%carried0 = %init0, %carried1 = %init1, %carried2 = %init2) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      %tok2 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src2 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %tok0, %tok1, %tok2 : !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    // CHECK: ttng.async_tma_store_wait {pendings = 2 : i32}
    ttng.async_tma_store_token_wait %result#0 : !ttg.async.token
    // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
    ttng.async_tma_store_token_wait %result#1 : !ttg.async.token
    // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.async_tma_store_token_wait %result#2 : !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Intervening scf.if contributes the minimum store count across branches.
// CHECK-LABEL: direct_with_intervening_if
  tt.func public @direct_with_intervening_if(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32,
      %cond: i1) {
    %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    scf.if %cond {
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    } else {
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      %tok2 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src2 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    }
    // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
    ttng.async_tma_store_token_wait %tok0 : !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// scf.if result tokens exclude the defining if and count later stores.
// CHECK-LABEL: if_result_token_min
  tt.func public @if_result_token_min(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32,
      %cond: i1) {
    %tok = scf.if %cond -> (!ttg.async.token) {
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %tok0 : !ttg.async.token
    } else {
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      %tok2 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src2 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %tok0 : !ttg.async.token
    }
    %tok3 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src2 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Matching wait guard selects the same branch for later sibling scf.if ops.
// CHECK-LABEL: if_result_token_same_guard
  tt.func public @if_result_token_same_guard(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32,
      %cond: i1) {
    %init = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %tok = scf.if %cond -> (!ttg.async.token) {
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %tok0 : !ttg.async.token
    } else {
      scf.yield %init : !ttg.async.token
    }
    scf.if %cond {
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    }
    scf.if %cond {
      %tok2 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src2 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    }
    scf.if %cond {
      // CHECK: ttng.async_tma_store_wait {pendings = 2 : i32}
      ttng.async_tma_store_token_wait %tok : !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Equivalent pure conditions also select the same branch.
// CHECK-LABEL: if_result_token_equivalent_guard
  tt.func public @if_result_token_equivalent_guard(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %cond0 = arith.cmpi eq, %i, %i : i32
    %cond1 = arith.cmpi eq, %i, %i : i32
    %init = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %tok = scf.if %cond0 -> (!ttg.async.token) {
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %tok0 : !ttg.async.token
    } else {
      scf.yield %init : !ttg.async.token
    }
    scf.if %cond0 {
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    }
    scf.if %cond1 {
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      ttng.async_tma_store_token_wait %tok : !ttg.async.token
    }
    tt.return
  }
}
