// Program-order pendings of barrierful TMA store token waits in a loop. These
// waits carry no planned_pending_count (barrierful waits must use program-order
// counting), and token-wait lowering runs after scf-to-cf, where a loop-carried
// token is a CFG block argument it cannot trace: without the annotation every
// such wait drains all stores (pendings = 0, the first RUN line). The annotation pass
// records the count on structured IR.
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --nvgpu-tma-store-token-wait-lowering | FileCheck %s --check-prefix=NOANNOT
// RUN: triton-opt %s -split-input-file --nvgpu-tma-store-token-wait-annotate-pendings --convert-scf-to-cf --nvgpu-tma-store-token-wait-lowering | FileCheck %s
// RUN: triton-opt %s -split-input-file --nvgpu-tma-store-token-wait-annotate-pendings | FileCheck %s --check-prefix=ATTR

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// Two staging slots, two stores per iteration (separate store partition at
// EPILOGUE_SUBTILE=2 with two staging copies): each wait guards the store issued
// two stores earlier, so one store may stay in flight.
// CHECK-LABEL: @one_hop_two_slots
// NOANNOT-LABEL: @one_hop_two_slots
  tt.func public @one_hop_two_slots(%desc: !tt.tensordesc<128x64xf16, #shared>, %s0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %s1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %bar0: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %bar1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %i: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %true = arith.constant true
    %t0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %r:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%a0 = %t0, %a1 = %t1) -> (!ttg.async.token, !ttg.async.token) {
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      // NOANNOT: ttng.async_tma_store_wait {pendings = 0 : i32}
      ttng.async_tma_store_token_wait %a0, %bar0[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      // NOANNOT: ttng.async_tma_store_wait {pendings = 0 : i32}
      ttng.async_tma_store_token_wait %a1, %bar1[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %n0, %n1 : !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// Four staging slots, two stores per iteration: the token rotates through two
// iter_args before reaching its store (two hops), so each wait guards the store
// issued four stores earlier and three may stay in flight.
// CHECK-LABEL: @two_hops_four_slots
// NOANNOT-LABEL: @two_hops_four_slots
  tt.func public @two_hops_four_slots(%desc: !tt.tensordesc<128x64xf16, #shared>, %s0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %s1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %bar0: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %bar1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %i: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %true = arith.constant true
    %t0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t2 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t3 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %r:4 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%a0 = %t0, %a1 = %t1, %b0 = %t2, %b1 = %t3) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      // CHECK: ttng.async_tma_store_wait {pendings = 3 : i32}
      // NOANNOT: ttng.async_tma_store_wait {pendings = 0 : i32}
      ttng.async_tma_store_token_wait %a0, %bar0[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      // CHECK: ttng.async_tma_store_wait {pendings = 3 : i32}
      // NOANNOT: ttng.async_tma_store_wait {pendings = 0 : i32}
      ttng.async_tma_store_token_wait %a1, %bar1[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %b0, %b1, %n0, %n1 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// Two hops, but the second iteration waits on a prologue token with only one
// store after it: the count is the minimum over the steady state (3) and every
// init path (1 for iteration 0, 3 for iteration 1).
// CHECK-LABEL: @two_hops_short_prologue
// NOANNOT-LABEL: @two_hops_short_prologue
  tt.func public @two_hops_short_prologue(%desc: !tt.tensordesc<128x64xf16, #shared>, %s0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %s1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %bar0: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %bar1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %i: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %true = arith.constant true
    %t0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %r:4 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%a0 = %t0, %a1 = %t1, %b0 = %t0, %b1 = %t1) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      ttng.async_tma_store_token_wait %a0, %bar0[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      ttng.async_tma_store_token_wait %a1, %bar1[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %b0, %b1, %n0, %n1 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// An init token that cannot be traced to a store keeps the conservative full
// drain.
// CHECK-LABEL: @untraceable_init
  tt.func public @untraceable_init(%desc: !tt.tensordesc<128x64xf16, #shared>, %s0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %s1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %bar0: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %bar1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %i: i32, %tok_in: !ttg.async.token) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %true = arith.constant true
    %r = scf.for %iv = %c0 to %c8 step %c1 iter_args(%a0 = %tok_in) -> (!ttg.async.token) {
      // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
      ttng.async_tma_store_token_wait %a0, %bar0[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %n0 : !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// Guarded prologue (persistent loop): each prologue store sits in an scf.if on
// the same "first tile exists" condition, and the else branch yields poison.
// A token from the then branch implies the later guarded store also ran; the
// poison branch constrains nothing.
// CHECK-LABEL: @guarded_prologue
// NOANNOT-LABEL: @guarded_prologue
  tt.func public @guarded_prologue(%desc: !tt.tensordesc<128x64xf16, #shared>, %s0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %s1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %bar0: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %bar1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %i: i32, %has_tile: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %true = arith.constant true
    %poison = ub.poison : !ttg.async.token
    %t0 = scf.if %has_tile -> (!ttg.async.token) {
      %s = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %s : !ttg.async.token
    } else {
      scf.yield %poison : !ttg.async.token
    }
    %t1 = scf.if %has_tile -> (!ttg.async.token) {
      %s = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %s : !ttg.async.token
    } else {
      scf.yield %poison : !ttg.async.token
    }
    %r:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%a0 = %t0, %a1 = %t1) -> (!ttg.async.token, !ttg.async.token) {
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      // NOANNOT: ttng.async_tma_store_wait {pendings = 0 : i32}
      ttng.async_tma_store_token_wait %a0, %bar0[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      // NOANNOT: ttng.async_tma_store_wait {pendings = 0 : i32}
      ttng.async_tma_store_token_wait %a1, %bar1[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %n0, %n1 : !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// A condition computed in the loop body may differ in the previous iteration,
// so the guarded store after the token's store must not be counted.
// CHECK-LABEL: @loop_variant_condition
// ATTR-LABEL: @loop_variant_condition
  tt.func public @loop_variant_condition(%desc: !tt.tensordesc<128x64xf16, #shared>, %s0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %s1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %bar0: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %bar1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %i: i32, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %true = arith.constant true
    %t0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %p1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %p2 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %r = scf.for %iv = %c0 to %c8 step %c1 iter_args(%a0 = %t0) -> (!ttg.async.token) {
      %c = arith.cmpi sgt, %iv, %n : index
      scf.if %c {
        // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
        // ATTR: ttng.async_tma_store_token_wait {{.*}}program_order_pending_count = 0
        ttng.async_tma_store_token_wait %a0, %bar0[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      }
      %n0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.if %c {
        %x = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      }
      scf.yield %n0 : !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// Guarded prologue stores on different conditions: the first slot's token
// does not imply the second guarded store ran.
// CHECK-LABEL: @guarded_prologue_different_conditions
  tt.func public @guarded_prologue_different_conditions(%desc: !tt.tensordesc<128x64xf16, #shared>, %s0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %s1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %bar0: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %bar1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %i: i32, %c_a: i1, %c_b: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %true = arith.constant true
    %poison = ub.poison : !ttg.async.token
    %t0 = scf.if %c_a -> (!ttg.async.token) {
      %s = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %s : !ttg.async.token
    } else {
      scf.yield %poison : !ttg.async.token
    }
    %t1 = scf.if %c_b -> (!ttg.async.token) {
      %s = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %s : !ttg.async.token
    } else {
      scf.yield %poison : !ttg.async.token
    }
    %r:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%a0 = %t0, %a1 = %t1) -> (!ttg.async.token, !ttg.async.token) {
      // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
      ttng.async_tma_store_token_wait %a0, %bar0[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
      ttng.async_tma_store_token_wait %a1, %bar1[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
      %n1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %s1 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %n0, %n1 : !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// A wait on a token no store produced gets no annotation (the lowering keeps
// its conservative drain).
// CHECK-LABEL: @poison_token
// ATTR-LABEL: @poison_token
  tt.func public @poison_token(%desc: !tt.tensordesc<128x64xf16, #shared>, %s0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %s1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %bar0: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %bar1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>, %i: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %true = arith.constant true
    %poison = ub.poison : !ttg.async.token
    // ATTR-NOT: program_order_pending_count
    // ATTR: ttng.async_tma_store_token_wait
    // ATTR-NOT: program_order_pending_count
    // CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.async_tma_store_token_wait %poison, %bar0[%true] : !ttg.async.token, !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>
    tt.return
  }
}
