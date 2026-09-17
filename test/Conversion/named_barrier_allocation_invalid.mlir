// RUN: not triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=84' 2>&1 | FileCheck %s

// Two partitions: the reserved partition ID covers only the first, so the
// second must be drawn from the pool -- which a dynamic user ID makes unsafe.
// A single-partition kernel needs no draw and stays legal; that case is
// covered in Hopper/WarpSpecialization/named_barrier_allocation_dynamic.mlir.
// CHECK: error: cannot allocate warp-specialize named barriers with a dynamic user named-barrier ID
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32} {
  tt.func @dynamic_user_id_blocks_compiler_allocation(%id: i32) {
    %c128 = arith.constant 128 : i32
    %user = ttng.user_named_barrier_id %id : i32
    ttng.wait_barrier_named %user, %c128 : !ttng.named_barrier_id, i32
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    }
    partition1() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// CHECK: error: not enough named barriers for warp-specialize partitions
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32} {
  tt.func @user_ids_exhaust_compiler_allocation() {
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %c9 = arith.constant 9 : i32
    %c10 = arith.constant 10 : i32
    %c11 = arith.constant 11 : i32
    %c12 = arith.constant 12 : i32
    %c13 = arith.constant 13 : i32
    %c14 = arith.constant 14 : i32
    %c15 = arith.constant 15 : i32
    %user3 = ttng.user_named_barrier_id %c3 : i32
    %user4 = ttng.user_named_barrier_id %c4 : i32
    %user5 = ttng.user_named_barrier_id %c5 : i32
    %user6 = ttng.user_named_barrier_id %c6 : i32
    %user7 = ttng.user_named_barrier_id %c7 : i32
    %user8 = ttng.user_named_barrier_id %c8 : i32
    %user9 = ttng.user_named_barrier_id %c9 : i32
    %user10 = ttng.user_named_barrier_id %c10 : i32
    %user11 = ttng.user_named_barrier_id %c11 : i32
    %user12 = ttng.user_named_barrier_id %c12 : i32
    %user13 = ttng.user_named_barrier_id %c13 : i32
    %user14 = ttng.user_named_barrier_id %c14 : i32
    %user15 = ttng.user_named_barrier_id %c15 : i32
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    }
    partition1() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
