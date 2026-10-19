// RUN: triton-opt %s -split-input-file --nvgpu-2cta-transform-loads | FileCheck %s

// A 2-CTA MMA reads half of B (along N) from each CTA of the pair. When B is
// built in registers from descriptor loads, the pass must still give each CTA
// its half: halve every N-carrying descriptor load (with the CTA offset on the
// N coordinate) and rebuild the register chain at half width. Leaving the
// full-width B made each CTA feed the wrong half (silent wrong answer). The MMA
// is tagged two_cta_register_b so FenceInsertion fences at cluster scope.

// fp32 B tile scaled by an N-independent vector (broadcast along N), then cast.
// CHECK-LABEL: @k
// CHECK: nvg.cluster_id
// CHECK: tt.descriptor_load {{.*}}{two_cta_b{{.*}}} : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32
// CHECK: tt.broadcast {{.*}} -> tensor<64x64xf32
// CHECK: arith.mulf {{.*}} : tensor<64x64xf32
// CHECK: arith.truncf {{.*}} to tensor<64x64xbf16
// CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<64x64xbf16
// CHECK: ttng.tc_gen5_mma {{.*}} {two_cta_register_b, two_ctas} : !ttg.memdesc<128x64xbf16, {{.*}}>, !ttg.memdesc<64x64xbf16
// CHECK-NOT: tensor<64x128xf32

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @k(%A: !tt.ptr<bf16>, %B: !tt.ptr<f32>, %B2: !tt.ptr<f32>, %E1: !tt.ptr<f32>, %C: !tt.ptr<f32>, %M: i32, %N: i32, %K: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %a_desc = arith.extsi %K : i32 to i64
    %a_desc_0 = tt.make_tensor_descriptor %A, [%M, %K], [%a_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16>
    %b_desc = arith.extsi %N : i32 to i64
    %b_desc_1 = tt.make_tensor_descriptor %B, [%K, %N], [%b_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
    %e1_desc = tt.make_tensor_descriptor %E1, [%K], [%c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64xf32>
    %c_desc = tt.make_tensor_descriptor %C, [%M, %N], [%b_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x128xf32>
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_2 = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_3 = arith.divsi %num_pid_n, %c128_i32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %num_pid_m_2, %num_pid_n_3 : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    scf.for %t = %0 to %1 step %c148_i32  : i32 {
      %pid_m = arith.remsi %t, %num_pid_m_2 : i32
      %pid_n = arith.divsi %t, %num_pid_m_2 : i32
      %a = arith.muli %pid_m, %c128_i32 : i32
      %b = arith.muli %pid_n, %c128_i32 : i32
      %acc = scf.for %ki = %c0_i32 to %3 step %c1_i32 iter_args(%acc_4 = %cst) -> (tensor<128x128xf32, #linear>)  : i32 {
        %a_5 = arith.muli %ki, %c64_i32 : i32
        %a_6 = tt.descriptor_load %a_desc_0[%a, %a_5] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
        %a_7 = ttg.local_alloc %a_6 : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %b_8 = tt.descriptor_load %b_desc_1[%a_5, %b] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked1>
        %b_9 = tt.descriptor_load %e1_desc[%a_5] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked2>
        %b_10 = ttg.convert_layout %b_9 : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %b_11 = tt.expand_dims %b_10 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xf32, #blocked1>
        %b_12 = tt.broadcast %b_11 : tensor<64x1xf32, #blocked1> -> tensor<64x128xf32, #blocked1>
        %b_13 = arith.mulf %b_8, %b_12 : tensor<64x128xf32, #blocked1>
        %b_14 = arith.truncf %b_13 : tensor<64x128xf32, #blocked1> to tensor<64x128xbf16, #blocked1>
        %b_15 = ttg.local_alloc %b_14 : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %acc_16, %acc_17 = ttng.tmem_alloc %acc_4 : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %acc_18 = ttng.tc_gen5_mma %a_7, %b_15, %acc_16[%acc_17], %true, %true {two_ctas} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_19, %acc_20 = ttng.tmem_load %acc_16[%acc_18] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        scf.yield %acc_19 : tensor<128x128xf32, #linear>
      }
      %4 = arith.muli %pid_m, %c128_i32 : i32
      %5 = arith.muli %pid_n, %c128_i32 : i32
      %6 = ttg.convert_layout %acc : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
      tt.descriptor_store %c_desc[%4, %5], %6 : !tt.tensordesc<128x128xf32>, tensor<128x128xf32, #blocked1>
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// Rank-3 descriptor with a leading unit block dim, cast to bf16.
// CHECK-LABEL: @k
// CHECK: tt.descriptor_load {{.*}}{two_cta_b{{.*}}} : !tt.tensordesc<1x64x64xf32> -> tensor<64x64xf32
// CHECK: arith.truncf {{.*}} to tensor<64x64xbf16
// CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<64x64xbf16
// CHECK: ttng.tc_gen5_mma {{.*}} {two_cta_register_b, two_ctas}

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @k(%A: !tt.ptr<bf16>, %B: !tt.ptr<f32>, %B2: !tt.ptr<f32>, %E1: !tt.ptr<f32>, %C: !tt.ptr<f32>, %M: i32, %N: i32, %K: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c148_i32 = arith.constant 148 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %a_desc = arith.extsi %K : i32 to i64
    %a_desc_0 = tt.make_tensor_descriptor %A, [%M, %K], [%a_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16>
    %b_desc = arith.muli %K, %N : i32
    %b_desc_1 = arith.extsi %b_desc : i32 to i64
    %b_desc_2 = arith.extsi %N : i32 to i64
    %b_desc_3 = tt.make_tensor_descriptor %B, [%c1_i32, %K, %N], [%b_desc_1, %b_desc_2, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<1x64x128xf32>
    %c_desc = tt.make_tensor_descriptor %C, [%M, %N], [%b_desc_2, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x128xf32>
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_4 = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_5 = arith.divsi %num_pid_n, %c128_i32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %num_pid_m_4, %num_pid_n_5 : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    scf.for %t = %0 to %1 step %c148_i32  : i32 {
      %pid_m = arith.remsi %t, %num_pid_m_4 : i32
      %pid_n = arith.divsi %t, %num_pid_m_4 : i32
      %a = arith.muli %pid_m, %c128_i32 : i32
      %b = arith.muli %pid_n, %c128_i32 : i32
      %acc = scf.for %ki = %c0_i32 to %3 step %c1_i32 iter_args(%acc_6 = %cst) -> (tensor<128x128xf32, #linear>)  : i32 {
        %a_7 = arith.muli %ki, %c64_i32 : i32
        %a_8 = tt.descriptor_load %a_desc_0[%a, %a_7] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
        %a_9 = ttg.local_alloc %a_8 : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %b_10 = tt.descriptor_load %b_desc_3[%c0_i32, %a_7, %b] : !tt.tensordesc<1x64x128xf32> -> tensor<64x128xf32, #blocked1>
        %b_11 = arith.truncf %b_10 : tensor<64x128xf32, #blocked1> to tensor<64x128xbf16, #blocked1>
        %b_12 = ttg.local_alloc %b_11 : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %acc_13, %acc_14 = ttng.tmem_alloc %acc_6 : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %acc_15 = ttng.tc_gen5_mma %a_9, %b_12, %acc_13[%acc_14], %true, %true {two_ctas} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_16, %acc_17 = ttng.tmem_load %acc_13[%acc_15] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        scf.yield %acc_16 : tensor<128x128xf32, #linear>
      }
      %4 = arith.muli %pid_m, %c128_i32 : i32
      %5 = arith.muli %pid_n, %c128_i32 : i32
      %6 = ttg.convert_layout %acc : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
      tt.descriptor_store %c_desc[%4, %5], %6 : !tt.tensordesc<128x128xf32>, tensor<128x128xf32, #blocked1>
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// Two loads of one device descriptor halved along different dims (one is
// transposed) need two half-width descriptors.
// CHECK-LABEL: @k
// CHECK-DAG: tt.make_tensor_descriptor {{.*}} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16>
// CHECK-DAG: tt.make_tensor_descriptor {{.*}} : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16>
// CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16
// CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16
// CHECK: ttng.tc_gen5_mma {{.*}} {two_cta_register_b, two_ctas}

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @k(%a: !ttg.memdesc<128x128xbf16, #shared, #smem>, %bp: !tt.ptr<bf16>, %n: i32, %s64: i64, %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %i: i32, %j: i32) {
    %true = arith.constant true
    %c1 = arith.constant 1 : i64
    %bd = tt.make_tensor_descriptor %bp, [%n, %n], [%s64, %c1] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16>
    %b = tt.descriptor_load %bd[%i, %j] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #blocked1>
    %c = tt.descriptor_load %bd[%j, %i] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #blocked1>
    %ct = tt.trans %c {order = array<i32: 1, 0>} : tensor<128x128xbf16, #blocked1> -> tensor<128x128xbf16, #blocked3>
    %ctc = ttg.convert_layout %ct : tensor<128x128xbf16, #blocked3> -> tensor<128x128xbf16, #blocked1>
    %sum = arith.addf %b, %ctc : tensor<128x128xbf16, #blocked1>
    %s = ttg.local_alloc %sum : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    ttng.tc_gen5_mma %a, %s, %acc, %true, %true {two_ctas} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
