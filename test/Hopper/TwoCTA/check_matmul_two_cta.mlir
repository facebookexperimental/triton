// RUN: triton-opt %s -split-input-file --triton-nvidia-check-matmul-two-cta --verify-diagnostics | FileCheck %s
// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-atomic-tile-scheduler-prepare --triton-nvidia-gpu-atomic-tile-scheduler-materialize | FileCheck %s --check-prefix=ATOMIC

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK-LABEL: module
// CHECK-SAME: "ttng.two-ctas" = true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @independent_two_cta_mmas(
      %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %a1: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc1: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok0: !ttg.async.token,
      %acc_tok1: !ttg.async.token) {
    %true = arith.constant true
    %tok0 = ttng.tc_gen5_mma %a0, %b, %acc0[%acc_tok0], %true, %true {two_ctas} :
      !ttg.memdesc<128x64xf16, #shared, #smem>,
      !ttg.memdesc<64x128xf16, #shared1, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %tok1 = ttng.tc_gen5_mma %a1, %b, %acc1[%acc_tok1], %true, %true {two_ctas} :
      !ttg.memdesc<128x64xf16, #shared, #smem>,
      !ttg.memdesc<64x128xf16, #shared1, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @dependent_two_cta_dot_chain(
      %q: tensor<128x64xf16>,
      %k: tensor<64x128xf16>,
      %v: tensor<128x128xf16>,
      %acc: tensor<128x128xf32>) {
    // expected-note @+1 {{producer 2-CTA dot result is consumed by this dot.}}
    %qk = tt.dot %q, %k, %acc {two_ctas} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    %qk_f16 = arith.truncf %qk : tensor<128x128xf32> to tensor<128x128xf16>
    // expected-error @+1 {{two_ctas=True does not currently support dependent matmul chains}}
    %pv = tt.dot %qk_f16, %v, %acc {two_ctas} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
    tt.return
  }
}

// -----

// A 2x2 physical cluster gets one contiguous group of four logical PIDs. The
// compiler linearizes the physical cluster coordinate, reserves four PIDs with
// one leader atomic, and distributes the base to the other three ranks.
// ATOMIC-LABEL: @clustered_atomic_scheduler_2x2
// ATOMIC: tt.get_program_id y
// ATOMIC: arith.divui {{.*}}, %c2_i32
// ATOMIC: %[[SEED_RANK:[0-9]+]] = nvg.cluster_id
// ATOMIC: arith.muli {{.*}}, %c4_i32
// ATOMIC: arith.addi {{.*}}, %[[SEED_RANK]]
// ATOMIC: ttng.init_barrier {{.*}}, 4
// ATOMIC: scf.while
// ATOMIC: %[[RANK:[0-9]+]] = nvg.cluster_id
// ATOMIC: scf.if
// ATOMIC-COUNT-1: tt.atomic_rmw add, acq_rel, gpu, {{.*}}, %c4_i32
// ATOMIC-COUNT-3: ttg.remote_shmem_store
// ATOMIC: arith.addi {{.*}}, %[[RANK]]
// ATOMIC: ttng.map_to_remote_buffer
// ATOMIC: ttng.arrive_barrier
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 2 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @clustered_atomic_scheduler_2x2(
      %counter: !tt.ptr<i32>, %tile_groups: i32,
      %a: tensor<128x64xf16>, %b: tensor<64x128xf16>,
      %acc: tensor<128x128xf32>) {
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %num_tiles = arith.muli %tile_groups, %c4 : i32
    %start = tt.get_program_id x : i32
    %result = scf.while (%tile = %start) : (i32) -> i32 {
      %valid = arith.cmpi slt, %tile, %num_tiles : i32
      scf.condition(%valid) %tile : i32
    } do {
    ^bb0(%tile: i32):
      %dot = tt.dot %a, %b, %acc {two_ctas} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
      %next = tt.atomic_rmw add, acq_rel, gpu, %counter, %c1, %true : (!tt.ptr<i32>, i32, i1) -> i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// -----

// A 4x1 cluster follows the same protocol; the shape only changes how the
// physical cluster coordinate is linearized.
// ATOMIC-LABEL: @clustered_atomic_scheduler_4x1
// ATOMIC: arith.divui {{.*}}, %c4_i32
// ATOMIC: arith.muli {{.*}}, %c4_i32
// ATOMIC: ttng.init_barrier {{.*}}, 4
// ATOMIC: scf.if
// ATOMIC-COUNT-1: tt.atomic_rmw add, acq_rel, gpu, {{.*}}, %c4_i32
// ATOMIC-COUNT-3: ttg.remote_shmem_store
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 4 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @clustered_atomic_scheduler_4x1(
      %counter: !tt.ptr<i32>, %a: tensor<128x64xf16>,
      %b: tensor<64x128xf16>, %acc: tensor<128x128xf32>) {
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    %true = arith.constant true
    %start = tt.get_program_id x : i32
    %result = scf.while (%tile = %start) : (i32) -> i32 {
      %valid = arith.cmpi slt, %tile, %c8 : i32
      scf.condition(%valid) %tile : i32
    } do {
    ^bb0(%tile: i32):
      %dot = tt.dot %a, %b, %acc {two_ctas} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
      %next = tt.atomic_rmw add, acq_rel, gpu, %counter, %c1, %true : (!tt.ptr<i32>, i32, i1) -> i32
      scf.yield %next : i32
    }
    tt.return
  }
}
