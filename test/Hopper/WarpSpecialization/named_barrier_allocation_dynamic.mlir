// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=84' | FileCheck %s
// Both modules must lower cleanly. Assert that directly on stderr too: a
// diagnostic here would mean allocation rejected a legal kernel.
// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=84' 2>&1 | FileCheck %s --check-prefix=NODIAG
// NODIAG-NOT: error:

// A dynamic user ID only blocks allocation once an ID must be drawn from the
// pool. One partition is covered by the reserved partition ID on its own, so
// nothing is allocated and the kernel stays legal. The two-partition form,
// which does need a draw and is correctly rejected, lives in
// Conversion/named_barrier_allocation_invalid.mlir.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @dynamic_user_id_single_partition_is_legal
  tt.func @dynamic_user_id_single_partition_is_legal(%id: i32) {
    %c128 = arith.constant 128 : i32
    %user = ttng.user_named_barrier_id %id : i32
    ttng.wait_barrier_named %user, %c128 : !ttng.named_barrier_id, i32
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// A statically known user ID entering the partition as an explicit capture is
// a block argument. The allocator has to resolve the capture before deciding
// the ID is dynamic -- otherwise it marks the pool unavailable and rejects a
// kernel that is fully static. Two partitions here, so a pool draw really does
// happen: it must succeed, routing around the captured reservation.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32} {
  // CHECK-LABEL: @captured_static_user_id_is_not_dynamic
  tt.func @captured_static_user_id_is_not_dynamic() {
    %c3 = arith.constant 3 : i32
    %c128 = arith.constant 128 : i32
    ttg.warp_specialize(%c3, %c128) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: i32, %arg1: i32) num_warps(4) {
      %user = ttng.user_named_barrier_id %arg0 : i32
      ttng.wait_barrier_named %user, %arg1 : !ttng.named_barrier_id, i32
      ttg.warp_return
    }
    partition1(%arg0: i32, %arg1: i32) num_warps(4) {
      ttg.warp_return
    } : (i32, i32) -> ()
    tt.return
  }
}
