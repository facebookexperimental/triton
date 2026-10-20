// RUN: triton-opt %s -split-input-file --nvgpu-insert-2cta-sync -verify-diagnostics

// The 2-CTA sync phase is linearized over the enclosing loops. When an inner
// loop's trip count changes between iterations of an outer loop and the nest
// is not the single persistent loop that gets a running count, the phase
// cannot be derived: refuse instead of emitting a deadlocking barrier.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  // MMA loop nested two levels below the persistent loop.
  tt.func @varying_trip_count_two_level_nest(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token, %k_tiles: i32, %num_tiles: i32) {
    ttg.warp_specialize(%a, %b, %acc, %acc_tok, %k_tiles, %num_tiles)
    default {
      ttg.warp_yield
    }
    partition0(%part_a: !ttg.memdesc<128x64xf16, #shared, #smem>,
               %part_b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
               %part_acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
               %part_acc_tok: !ttg.async.token, %part_k_tiles: i32, %part_num_tiles: i32) num_warps(4) {
      %true = arith.constant true
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      scf.for %tile = %c0_i32 to %part_num_tiles step %c1_i32 : i32 {
        scf.for %sub = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
          %k_lo = arith.addi %sub, %tile : i32
          scf.for %k = %k_lo to %part_k_tiles step %c1_i32 : i32 {
            // expected-error @below {{the cross-CTA barrier phase cannot be derived}}
            %tok = ttng.tc_gen5_mma %part_a, %part_b, %part_acc[%part_acc_tok], %true, %true {two_ctas} :
              !ttg.memdesc<128x64xf16, #shared, #smem>,
              !ttg.memdesc<64x128xf16, #shared1, #smem>,
              !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          }
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<128x64xf16, #shared, #smem>,
         !ttg.memdesc<64x128xf16, #shared1, #smem>,
         !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
         !ttg.async.token, i32, i32) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  // MMA loop under an scf.if inside the persistent loop.
  tt.func @varying_trip_count_under_if(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token, %k_tiles: i32, %num_tiles: i32) {
    ttg.warp_specialize(%a, %b, %acc, %acc_tok, %k_tiles, %num_tiles)
    default {
      ttg.warp_yield
    }
    partition0(%part_a: !ttg.memdesc<128x64xf16, #shared, #smem>,
               %part_b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
               %part_acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
               %part_acc_tok: !ttg.async.token, %part_k_tiles: i32, %part_num_tiles: i32) num_warps(4) {
      %true = arith.constant true
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      scf.for %tile = %c0_i32 to %part_num_tiles step %c1_i32 : i32 {
        %k_lo = arith.remsi %tile, %c3_i32 : i32
        %valid = arith.cmpi slt, %k_lo, %part_k_tiles : i32
        scf.if %valid {
          scf.for %k = %k_lo to %part_k_tiles step %c1_i32 : i32 {
            // expected-error @below {{the cross-CTA barrier phase cannot be derived}}
            %tok = ttng.tc_gen5_mma %part_a, %part_b, %part_acc[%part_acc_tok], %true, %true {two_ctas} :
              !ttg.memdesc<128x64xf16, #shared, #smem>,
              !ttg.memdesc<64x128xf16, #shared1, #smem>,
              !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          }
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<128x64xf16, #shared, #smem>,
         !ttg.memdesc<64x128xf16, #shared1, #smem>,
         !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
         !ttg.async.token, i32, i32) -> ()
    tt.return
  }
}

// -----

// Without warp specialization the barrier is initialized at kernel entry, so
// the phase counts every loop. A two-level nest above a varying MMA loop is
// refused like its warp-specialized counterpart.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @non_ws_varying_trip_count_two_level_nest(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token, %k_tiles: i32, %num_tiles: i32) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    scf.for %batch = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      scf.for %tile = %c0_i32 to %num_tiles step %c1_i32 : i32 {
        %k_hi = arith.addi %tile, %c1_i32 : i32
        scf.for %k = %c0_i32 to %k_hi step %c1_i32 : i32 {
          // expected-error @below {{the cross-CTA barrier phase cannot be derived}}
          %tok = ttng.tc_gen5_mma %a, %b, %acc[%acc_tok], %true, %true {two_ctas} :
            !ttg.memdesc<128x64xf16, #shared, #smem>,
            !ttg.memdesc<64x128xf16, #shared1, #smem>,
            !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        }
      }
    }
    tt.return
  }
}

// -----

// Without warp specialization, an MMA loop under an scf.if inside the
// persistent loop: tiles that skip the if skip rendezvous that the persistent
// loop's iteration index still counts.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @non_ws_mma_loop_under_if(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token, %k_tiles: i32, %num_tiles: i32, %m: i32) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %tile = %c0_i32 to %num_tiles step %c1_i32 : i32 {
      %valid = arith.cmpi slt, %tile, %m : i32
      scf.if %valid {
        scf.for %k = %c0_i32 to %k_tiles step %c1_i32 : i32 {
          // expected-error @below {{the cross-CTA barrier phase cannot be derived}}
          %tok = ttng.tc_gen5_mma %a, %b, %acc[%acc_tok], %true, %true {two_ctas} :
            !ttg.memdesc<128x64xf16, #shared, #smem>,
            !ttg.memdesc<64x128xf16, #shared1, #smem>,
            !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        }
      }
    }
    tt.return
  }
}

// -----

// Without warp specialization, a while loop's iteration counter restarts in
// every iteration of an enclosing loop, and a while loop has no trip count to
// scale that loop by: refuse.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @non_ws_while_in_loop(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token, %n: i32) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    scf.for %o = %c0_i32 to %c3_i32 step %c1_i32 : i32 {
      %r = scf.while (%i = %c0_i32) : (i32) -> i32 {
        %cond = arith.cmpi slt, %i, %n : i32
        scf.condition(%cond) %i : i32
      } do {
      ^bb0(%j: i32):
        // expected-error @below {{the cross-CTA barrier phase cannot be derived}}
        %tok = ttng.tc_gen5_mma %a, %b, %acc[%acc_tok], %true, %true {two_ctas} :
          !ttg.memdesc<128x64xf16, #shared, #smem>,
          !ttg.memdesc<64x128xf16, #shared1, #smem>,
          !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %nx = arith.addi %j, %c1_i32 : i32
        scf.yield %nx : i32
      }
    }
    tt.return
  }
}
