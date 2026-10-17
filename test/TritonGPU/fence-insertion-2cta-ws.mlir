// RUN: triton-opt %s --triton-nvidia-gpu-fence-insertion | FileCheck %s

// Warp-specialized 2-CTA GEMM whose B tile is built in registers (fp32 TMA tile
// cast to bf16) by the default partition and read by the 2-CTA MMA in
// partition0. The leader CTA's MMA reads the peer CTA's shared memory, so the
// fence before it must be cluster-scoped. ProxyFenceInsertion does not add one
// here because the write and the read are in different partitions.

// CHECK: partition0(
// CHECK: ttng.fence_async_shared {bCluster = true}
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}}two_ctas

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @k(%A: !tt.ptr<bf16>, %B: !tt.ptr<f32>, %B2: !tt.ptr<f32>, %E1: !tt.ptr<f32>, %C: !tt.ptr<f32>, %M: i32, %N: i32, %K: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant {async_task_id = array<i32: 0>} true
    %c3_i64 = arith.constant {async_task_id = array<i32: 0>} 3 : i64
    %c4_i64 = arith.constant {async_task_id = array<i32: 0>} 4 : i64
    %c0_i64 = arith.constant {async_task_id = array<i32: 0>} 0 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
    %c128_i32 = arith.constant {async_task_id = array<i32: 0>} 128 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %c148_i32 = arith.constant {async_task_id = array<i32: 0>} 148 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0>} 127 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0>} 63 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %b = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    %b_0 = ttg.memdesc_index %b[%c0_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_1 = ttg.memdesc_index %b[%c1_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_2 = ttg.memdesc_index %b[%c2_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_2, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %a = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    %a_3 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %a_3, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %a_4 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %a_4, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %a_5 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %a_5, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %a_6 = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    %a_7 = ttg.memdesc_index %a_6[%c0_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %a_7, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %a_8 = ttg.memdesc_index %a_6[%c1_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %a_8, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %a_9 = ttg.memdesc_index %a_6[%c2_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %a_9, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_10 = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    %b_11 = ttg.memdesc_index %b_10[%c0_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_11, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_12 = ttg.memdesc_index %b_10[%c1_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_12, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_13 = ttg.memdesc_index %b_10[%c2_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_13, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<4x1xi64, #shared, #smem, mutable>
    %acc_14 = ttg.memdesc_index %acc[%c0_i32] : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_14, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_15 = ttg.memdesc_index %acc[%c1_i32] : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_15, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_16 = ttg.memdesc_index %acc[%c2_i32] : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_16, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_17 = ttg.memdesc_index %acc[%c3_i32] : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_17, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_18 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<4x1xi64, #shared, #smem, mutable>
    %acc_19 = ttg.memdesc_index %acc_18[%c0_i32] : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_19, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_20 = ttg.memdesc_index %acc_18[%c1_i32] : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_20, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_21 = ttg.memdesc_index %acc_18[%c2_i32] : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_21, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_22 = ttg.memdesc_index %acc_18[%c3_i32] : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_22, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_23 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    %b_24 = ttg.memdesc_index %b_23[%c0_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_24, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_25 = ttg.memdesc_index %b_23[%c1_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_25, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_26 = ttg.memdesc_index %b_23[%c2_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_26, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_27 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    %b_28 = ttg.memdesc_index %b_27[%c0_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_28, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_29 = ttg.memdesc_index %b_27[%c1_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_29, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b_30 = ttg.memdesc_index %b_27[%c2_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b_30, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    %b_31 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<3x64x64xbf16, #shared1, #smem, mutable>
    %a_32 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<3x128x64xbf16, #shared1, #smem, mutable>
    %b_33 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<3x64x64xf32, #shared2, #smem, mutable>
    %c_desc_staging = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %2 = ttg.memdesc_index %0[%c1_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %2, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %3 = ttg.memdesc_index %0[%c2_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %3, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_34 = ttng.tmem_alloc {buffer.copy = 4 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<4x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_specialize(%M, %N, %K, %acc_34, %b_31, %a_32, %b_10, %a, %a_6, %acc, %A, %B, %b, %b_33, %acc_18, %b_23, %b_27, %0) attributes {requestedRegisters = array<i32: 24, 24>, ttg.partition.types = ["epilogue", "gemm", "load"]}
    default {
      %b_desc = arith.extsi %N {async_task_id = array<i32: 0>} : i32 to i64
      %c_desc = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
      %c_desc_35 = arith.muli %b_desc, %c4_i64 : i64
      ttng.tensormap_create %c_desc, %C, [%c32_i32, %c128_i32], [%N, %M], [%c_desc_35], [%c1_i32, %c1_i32] {elem_type = 7 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<f32>, i32, i32, i32, i32, i64, i32, i32) -> ()
      ttng.tensormap_fenceproxy_acquire %c_desc : !tt.ptr<i8>
      %c_desc_36 = ttng.reinterpret_tensor_descriptor %c_desc : !tt.ptr<i8> to !tt.tensordesc<128x128xf32, #shared2>
      %num_pid_m = arith.addi %M, %c127_i32 {async_task_id = array<i32: 0>} : i32
      %num_pid_m_37 = arith.divsi %num_pid_m, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %num_pid_n = arith.addi %N, %c127_i32 {async_task_id = array<i32: 0>} : i32
      %num_pid_n_38 = arith.divsi %num_pid_n, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %4 = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
      %5 = arith.muli %num_pid_m_37, %num_pid_n_38 {async_task_id = array<i32: 0>} : i32
      %6 = arith.addi %K, %c63_i32 {async_task_id = array<i32: 0>} : i32
      %7 = arith.divsi %6, %c64_i32 {async_task_id = array<i32: 0>} : i32
      %8:2 = scf.for %arg8 = %4 to %5 step %c148_i32 iter_args(%arg9 = %c0_i64, %arg10 = %c0_i64) -> (i64, i64)  : i32 {
        %pid_m = arith.remsi %arg8, %num_pid_m_37 {async_task_id = array<i32: 0>} : i32
        %pid_n = arith.divsi %arg8, %num_pid_m_37 {async_task_id = array<i32: 0>} : i32
        %a_39 = arith.muli %pid_m, %c128_i32 {async_task_id = array<i32: 0>} : i32
        %b_40 = arith.muli %pid_n, %c128_i32 {async_task_id = array<i32: 0>} : i32
        %acc_41 = arith.divui %arg9, %c4_i64 {async_task_id = array<i32: 0>} : i64
        %acc_42 = arith.muli %acc_41, %c4_i64 {async_task_id = array<i32: 0>} : i64
        %acc_43 = arith.subi %arg9, %acc_42 {async_task_id = array<i32: 0>} : i64
        %acc_44 = arith.trunci %acc_43 {async_task_id = array<i32: 0>} : i64 to i32
        %acc_45 = arith.andi %acc_41, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %acc_46 = arith.trunci %acc_45 {async_task_id = array<i32: 0>} : i64 to i1
        %acc_47 = scf.for %acc_54 = %c0_i32 to %7 step %c1_i32 iter_args(%arg12 = %arg10) -> (i64)  : i32 {
          %b_55 = arith.divui %arg12, %c3_i64 {async_task_id = array<i32: 0>} : i64
          %b_56 = arith.muli %b_55, %c3_i64 {async_task_id = array<i32: 0>} : i64
          %b_57 = arith.subi %arg12, %b_56 {async_task_id = array<i32: 0>} : i64
          %b_58 = arith.trunci %b_57 {async_task_id = array<i32: 0>} : i64 to i32
          %b_59 = arith.andi %b_55, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %b_60 = arith.trunci %b_59 {async_task_id = array<i32: 0>} : i64 to i1
          %b_61 = ttg.memdesc_index %b_33[%b_58] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x64x64xf32, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared2, #smem, mutable>
          %b_62 = ttg.memdesc_index %b[%b_58] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %b_63 = arith.extui %b_60 {async_task_id = array<i32: 0>} : i1 to i32
          %b_64 = nvg.cluster_id {async_task_id = array<i32: 0>}
          %b_65 = arith.remui %b_64, %c2_i32 {async_task_id = array<i32: 0>} : i32
          %b_66 = arith.cmpi eq, %b_65, %c0_i32 {async_task_id = array<i32: 0>} : i32
          %b_67 = arith.cmpi ne, %b_65, %c0_i32 {async_task_id = array<i32: 0>} : i32
          %b_68 = arith.xori %b_64, %c1_i32 {async_task_id = array<i32: 0>} : i32
          %b_69 = ttng.map_to_remote_buffer %b_62, %b_68 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %b_62, %b_63, %b_66 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.fence_async_shared {async_task_id = array<i32: 0>, bCluster = false}
          ttng.arrive_barrier %b_69, 1, %b_66 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %b_62, %b_63, %b_67 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %b_70 = ttg.local_load %b_61 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared2, #smem, mutable> -> tensor<64x64xf32, #blocked>
          %b_71 = ttg.memdesc_index %b_27[%b_58] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.arrive_barrier %b_71, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %b_72 = arith.truncf %b_70 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
          %b_73 = ttg.memdesc_index %b_31[%b_58] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
          %b_74 = ttg.memdesc_index %b_10[%b_58] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %b_75 = arith.xori %b_60, %true {async_task_id = array<i32: 0>} : i1
          %b_76 = arith.extui %b_75 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %b_74, %b_76 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttg.local_store %b_72, %b_73 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
          %b_77 = ttg.memdesc_index %b_23[%b_58] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.arrive_barrier %b_77, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %accum_cnt_78 = arith.addi %arg12, %c1_i64 {async_task_id = array<i32: 0>} : i64
          scf.yield {async_task_id = array<i32: 0>} %accum_cnt_78 : i64
        } {async_task_id = array<i32: 0>, tt.warp_specialize}
        %acc_48 = ttg.memdesc_index %acc_34[%acc_44] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_49 = ttg.memdesc_index %acc[%acc_44] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_50 = arith.extui %acc_46 {async_task_id = array<i32: 0>} : i1 to i32
        %acc_51 = ttg.memdesc_index %acc_18[%acc_44] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %c_desc_staging_52 = ttg.memdesc_index %c_desc_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128x128xf32, #shared2, #smem, mutable>
        ttng.wait_barrier %acc_49, %acc_50 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2>, direction = "forward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_53 = ttng.tmem_load %acc_48 {async_task_id = array<i32: 0>, tmem.end = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        ttng.arrive_barrier %acc_51, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttg.local_store %acc_53, %c_desc_staging_52 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #shared2, #smem, mutable>
        %9 = ttng.async_tma_copy_local_to_global %c_desc_36[%a_39, %b_40] %c_desc_staging_52 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x128xf32, #shared2>, !ttg.memdesc<128x128xf32, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %9   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %accum_cnt = arith.addi %arg9, %c1_i64 {async_task_id = array<i32: 0>} : i64
        scf.yield {async_task_id = array<i32: 0>} %accum_cnt, %acc_47 : i64, i64
      } {async_task_id = array<i32: 0>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_yield
    }
    partition0(%M_35: i32, %N_36: i32, %K_37: i32, %acc_38: !ttg.memdesc<4x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %b_39: !ttg.memdesc<3x64x64xbf16, #shared1, #smem, mutable>, %a_40: !ttg.memdesc<3x128x64xbf16, #shared1, #smem, mutable>, %b_41: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %a_42: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %a_43: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %acc_44: !ttg.memdesc<4x1xi64, #shared, #smem, mutable>, %A_45: !tt.ptr<bf16>, %B_46: !tt.ptr<f32>, %b_47: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %b_48: !ttg.memdesc<3x64x64xf32, #shared2, #smem, mutable>, %acc_49: !ttg.memdesc<4x1xi64, #shared, #smem, mutable>, %b_50: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %b_51: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %arg25: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>) num_warps(1) {
      %c-2_i32 = arith.constant -2 : i32
      %c2_i32_52 = arith.constant {async_task_id = array<i32: 1>} 2 : i32
      %c3_i64_53 = arith.constant {async_task_id = array<i32: 1>} 3 : i64
      %c1_i64_54 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
      %c4_i64_55 = arith.constant {async_task_id = array<i32: 1>} 4 : i64
      %c0_i64_56 = arith.constant {async_task_id = array<i32: 1>} 0 : i64
      %c63_i32_57 = arith.constant {async_task_id = array<i32: 1>} 63 : i32
      %c127_i32_58 = arith.constant {async_task_id = array<i32: 1>} 127 : i32
      %c1_i32_59 = arith.constant {async_task_id = array<i32: 1>} 1 : i32
      %c0_i32_60 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
      %c148_i32_61 = arith.constant {async_task_id = array<i32: 1>} 148 : i32
      %b_62 = arith.constant {async_task_id = array<i32: 1>} 64 : i32
      %c128_i32_63 = arith.constant {async_task_id = array<i32: 1>} 128 : i32
      %true_64 = arith.constant {async_task_id = array<i32: 1>} true
      %false = arith.constant {async_task_id = array<i32: 1>} false
      %num_pid_m = arith.addi %M_35, %c127_i32_58 {async_task_id = array<i32: 1>} : i32
      %num_pid_m_65 = arith.divsi %num_pid_m, %c128_i32_63 {async_task_id = array<i32: 1>} : i32
      %num_pid_n = arith.addi %N_36, %c127_i32_58 {async_task_id = array<i32: 1>} : i32
      %num_pid_n_66 = arith.divsi %num_pid_n, %c128_i32_63 {async_task_id = array<i32: 1>} : i32
      %4 = tt.get_program_id x {async_task_id = array<i32: 1>} : i32
      %5 = arith.muli %num_pid_m_65, %num_pid_n_66 {async_task_id = array<i32: 1>} : i32
      %6 = arith.addi %K_37, %c63_i32_57 {async_task_id = array<i32: 1>} : i32
      %7 = arith.divsi %6, %b_62 {async_task_id = array<i32: 1>} : i32
      %8:2 = scf.for %arg26 = %4 to %5 step %c148_i32_61 iter_args(%arg27 = %c0_i64_56, %arg28 = %c0_i64_56) -> (i64, i64)  : i32 {
        %acc_67 = arith.divui %arg27, %c4_i64_55 {async_task_id = array<i32: 1>} : i64
        %acc_68 = arith.muli %acc_67, %c4_i64_55 {async_task_id = array<i32: 1>} : i64
        %acc_69 = arith.subi %arg27, %acc_68 {async_task_id = array<i32: 1>} : i64
        %acc_70 = arith.trunci %acc_69 {async_task_id = array<i32: 1>} : i64 to i32
        %acc_71 = arith.andi %acc_67, %c1_i64_54 {async_task_id = array<i32: 1>} : i64
        %acc_72 = arith.trunci %acc_71 {async_task_id = array<i32: 1>} : i64 to i1
        %acc_73 = ttg.memdesc_index %acc_49[%acc_70] {async_task_id = array<i32: 1>} : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_74 = arith.xori %acc_72, %true_64 : i1
        %acc_75 = arith.extui %acc_74 : i1 to i32
        ttng.wait_barrier %acc_73, %acc_75 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_76:2 = scf.for %acc_78 = %c0_i32_60 to %7 step %c1_i32_59 iter_args(%arg30 = %false, %arg31 = %arg28) -> (i1, i64)  : i32 {
          %a_79 = arith.divui %arg31, %c3_i64_53 {async_task_id = array<i32: 1>} : i64
          %a_80 = arith.muli %a_79, %c3_i64_53 {async_task_id = array<i32: 1>} : i64
          %a_81 = arith.subi %arg31, %a_80 {async_task_id = array<i32: 1>} : i64
          %a_82 = arith.trunci %a_81 {async_task_id = array<i32: 1>} : i64 to i32
          %a_83 = arith.andi %a_79, %c1_i64_54 {async_task_id = array<i32: 1>} : i64
          %a_84 = arith.trunci %a_83 {async_task_id = array<i32: 1>} : i64 to i1
          %acc_85 = ttg.memdesc_index %acc_38[%acc_70] {async_task_id = array<i32: 1>} : !ttg.memdesc<4x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %b_86 = ttg.memdesc_index %b_39[%a_82] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
          %a_87 = ttg.memdesc_index %a_40[%a_82] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x128x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
          %b_88 = ttg.memdesc_index %b_41[%a_82] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %b_89 = ttg.memdesc_index %b_50[%a_82] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_90 = arith.extui %a_84 : i1 to i32
          ttng.wait_barrier %b_89, %acc_90 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %a_91 = ttg.memdesc_index %a_42[%a_82] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %a_92 = ttg.memdesc_index %a_43[%a_82] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %a_93 = arith.extui %a_84 {async_task_id = array<i32: 1>} : i1 to i32
          %a_94 = nvg.cluster_id {async_task_id = array<i32: 1>}
          %a_95 = arith.remui %a_94, %c2_i32_52 {async_task_id = array<i32: 1>} : i32
          %a_96 = arith.cmpi eq, %a_95, %c0_i32_60 {async_task_id = array<i32: 1>} : i32
          %a_97 = arith.cmpi ne, %a_95, %c0_i32_60 {async_task_id = array<i32: 1>} : i32
          %a_98 = arith.xori %a_94, %c1_i32_59 {async_task_id = array<i32: 1>} : i32
          %a_99 = ttng.map_to_remote_buffer %a_92, %a_98 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %a_92, %a_93, %a_96 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.fence_async_shared {async_task_id = array<i32: 1>, bCluster = false}
          ttng.arrive_barrier %a_99, 1, %a_96 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %a_92, %a_93, %a_97 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_100 = arith.extui %acc_78 : i32 to i64
          %acc_101 = arith.extui %7 : i32 to i64
          %acc_102 = arith.subi %arg26, %4 : i32
          %acc_103 = arith.divui %acc_102, %c148_i32_61 : i32
          %acc_104 = arith.extui %acc_103 : i32 to i64
          %acc_105 = arith.muli %acc_104, %acc_101 : i64
          %acc_106 = arith.addi %acc_105, %acc_100 : i64
          %acc_107 = arith.remui %acc_106, %c3_i64_53 : i64
          %acc_108 = arith.divui %acc_106, %c3_i64_53 : i64
          %acc_109 = arith.andi %acc_108, %c1_i64_54 : i64
          %acc_110 = arith.trunci %acc_107 : i64 to i32
          %acc_111 = arith.trunci %acc_109 : i64 to i32
          %9 = ttg.memdesc_index %arg25[%acc_110] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_112 = nvg.cluster_id
          %acc_113 = arith.andi %acc_112, %c-2_i32 : i32
          %acc_114 = ttng.map_to_remote_buffer %9, %acc_113 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
          ttng.arrive_barrier %acc_114, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
          %acc_115 = arith.remui %acc_112, %c2_i32_52 : i32
          %acc_116 = arith.cmpi eq, %acc_115, %c0_i32_60 : i32
          ttng.wait_barrier %9, %acc_111, %acc_116 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.tc_gen5_mma %a_87, %b_86, %acc_85, %arg30, %true_64, %b_88[%true_64], %a_91[%true_64] {async_task_id = array<i32: 1>, is_async, tmem.start = array<i32: 0>, two_ctas} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %accum_cnt_117 = arith.addi %arg31, %c1_i64_54 {async_task_id = array<i32: 1>} : i64
          scf.yield {async_task_id = array<i32: 1>} %true_64, %accum_cnt_117 : i1, i64
        } {async_task_id = array<i32: 1>, tt.warp_specialize}
        %acc_77 = ttg.memdesc_index %acc_44[%acc_70] {async_task_id = array<i32: 1>} : !ttg.memdesc<4x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.tc_gen5_commit %acc_77 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %accum_cnt = arith.addi %arg27, %c1_i64_54 {async_task_id = array<i32: 1>} : i64
        scf.yield {async_task_id = array<i32: 1>} %accum_cnt, %acc_76#1 : i64, i64
      } {async_task_id = array<i32: 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition1(%M_35: i32, %N_36: i32, %K_37: i32, %acc_38: !ttg.memdesc<4x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %b_39: !ttg.memdesc<3x64x64xbf16, #shared1, #smem, mutable>, %a_40: !ttg.memdesc<3x128x64xbf16, #shared1, #smem, mutable>, %b_41: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %a_42: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %a_43: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %acc_44: !ttg.memdesc<4x1xi64, #shared, #smem, mutable>, %A_45: !tt.ptr<bf16>, %B_46: !tt.ptr<f32>, %b_47: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %b_48: !ttg.memdesc<3x64x64xf32, #shared2, #smem, mutable>, %acc_49: !ttg.memdesc<4x1xi64, #shared, #smem, mutable>, %b_50: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %b_51: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %arg25: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>) num_warps(1) {
      %b_desc = arith.constant 4 : i64
      %b_desc_52 = arith.constant 32 : i32
      %a_desc = arith.constant 2 : i64
      %c-2_i32 = arith.constant {async_task_id = array<i32: 2>} -2 : i32
      %true_53 = arith.constant {async_task_id = array<i32: 2>} true
      %c3_i64_54 = arith.constant {async_task_id = array<i32: 2>} 3 : i64
      %c0_i64_55 = arith.constant {async_task_id = array<i32: 2>} 0 : i64
      %c63_i32_56 = arith.constant {async_task_id = array<i32: 2>} 63 : i32
      %c127_i32_57 = arith.constant {async_task_id = array<i32: 2>} 127 : i32
      %c1_i32_58 = arith.constant {async_task_id = array<i32: 2>} 1 : i32
      %c0_i32_59 = arith.constant {async_task_id = array<i32: 2>} 0 : i32
      %c148_i32_60 = arith.constant {async_task_id = array<i32: 2>} 148 : i32
      %c64_i32_61 = arith.constant {async_task_id = array<i32: 2>} 64 : i32
      %c128_i32_62 = arith.constant {async_task_id = array<i32: 2>} 128 : i32
      %c1_i64_63 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
      %c2_i32_64 = arith.constant {async_task_id = array<i32: 2>} 2 : i32
      %a_desc_65 = arith.extsi %K_37 {async_task_id = array<i32: 2>} : i32 to i64
      %a_desc_66 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
      %a_desc_67 = arith.muli %a_desc_65, %a_desc : i64
      ttng.tensormap_create %a_desc_66, %A_45, [%c64_i32_61, %c128_i32_62], [%K_37, %M_35], [%a_desc_67], [%c1_i32_58, %c1_i32_58] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
      ttng.tensormap_fenceproxy_acquire %a_desc_66 : !tt.ptr<i8>
      %a_desc_68 = ttng.reinterpret_tensor_descriptor %a_desc_66 : !tt.ptr<i8> to !tt.tensordesc<128x64xbf16, #shared1>
      %b_desc_69 = arith.extsi %N_36 {async_task_id = array<i32: 2>} : i32 to i64
      %b_desc_70 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
      %b_desc_71 = arith.muli %b_desc_69, %b_desc : i64
      ttng.tensormap_create %b_desc_70, %B_46, [%b_desc_52, %c64_i32_61], [%N_36, %K_37], [%b_desc_71], [%c1_i32_58, %c1_i32_58] {elem_type = 7 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<f32>, i32, i32, i32, i32, i64, i32, i32) -> ()
      ttng.tensormap_fenceproxy_acquire %b_desc_70 : !tt.ptr<i8>
      %b_desc_72 = ttng.reinterpret_tensor_descriptor %b_desc_70 : !tt.ptr<i8> to !tt.tensordesc<64x64xf32, #shared2>
      %num_pid_m = arith.addi %M_35, %c127_i32_57 {async_task_id = array<i32: 2>} : i32
      %num_pid_m_73 = arith.divsi %num_pid_m, %c128_i32_62 {async_task_id = array<i32: 2>} : i32
      %num_pid_n = arith.addi %N_36, %c127_i32_57 {async_task_id = array<i32: 2>} : i32
      %num_pid_n_74 = arith.divsi %num_pid_n, %c128_i32_62 {async_task_id = array<i32: 2>} : i32
      %4 = tt.get_program_id x {async_task_id = array<i32: 2>} : i32
      %5 = arith.muli %num_pid_m_73, %num_pid_n_74 {async_task_id = array<i32: 2>} : i32
      %6 = arith.addi %K_37, %c63_i32_56 {async_task_id = array<i32: 2>} : i32
      %7 = arith.divsi %6, %c64_i32_61 {async_task_id = array<i32: 2>} : i32
      %b_75 = nvg.cluster_id {async_task_id = array<i32: 2>}
      %b_76 = arith.remsi %b_75, %c2_i32_64 {async_task_id = array<i32: 2>} : i32
      %b_77 = arith.muli %b_76, %c64_i32_61 {async_task_id = array<i32: 2>} : i32
      %8 = scf.for %arg26 = %4 to %5 step %c148_i32_60 iter_args(%arg27 = %c0_i64_55) -> (i64)  : i32 {
        %pid_m = arith.remsi %arg26, %num_pid_m_73 {async_task_id = array<i32: 2>} : i32
        %pid_n = arith.divsi %arg26, %num_pid_m_73 {async_task_id = array<i32: 2>} : i32
        %a_78 = arith.muli %pid_m, %c128_i32_62 {async_task_id = array<i32: 2>} : i32
        %b_79 = arith.muli %pid_n, %c128_i32_62 {async_task_id = array<i32: 2>} : i32
        %b_80 = arith.addi %b_79, %b_77 {async_task_id = array<i32: 2>} : i32
        %acc_81 = scf.for %acc_82 = %c0_i32_59 to %7 step %c1_i32_58 iter_args(%arg29 = %arg27) -> (i64)  : i32 {
          %a_83 = arith.muli %acc_82, %c64_i32_61 {async_task_id = array<i32: 2>} : i32
          %a_84 = arith.divui %arg29, %c3_i64_54 {async_task_id = array<i32: 2>} : i64
          %a_85 = arith.muli %a_84, %c3_i64_54 {async_task_id = array<i32: 2>} : i64
          %a_86 = arith.subi %arg29, %a_85 {async_task_id = array<i32: 2>} : i64
          %a_87 = arith.trunci %a_86 {async_task_id = array<i32: 2>} : i64 to i32
          %a_88 = arith.andi %a_84, %c1_i64_63 {async_task_id = array<i32: 2>} : i64
          %a_89 = arith.trunci %a_88 {async_task_id = array<i32: 2>} : i64 to i1
          %a_90 = ttg.memdesc_index %a_42[%a_87] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %a_91 = arith.xori %a_89, %true_53 {async_task_id = array<i32: 2>} : i1
          %a_92 = arith.extui %a_91 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %a_90, %a_92 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %a_93 = ttg.memdesc_index %a_43[%a_87] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %a_94 = arith.andi %b_75, %c-2_i32 {async_task_id = array<i32: 2>} : i32
          %a_95 = ttng.map_to_remote_buffer %a_93, %a_94 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
          %a_96 = arith.remui %b_75, %c2_i32_64 {async_task_id = array<i32: 2>} : i32
          %a_97 = arith.cmpi eq, %a_96, %c0_i32_59 {async_task_id = array<i32: 2>} : i32
          ttng.barrier_expect %a_93, 32768 {async_task_id = array<i32: 2>}, %a_97 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %a_98 = ttg.memdesc_index %a_40[%a_87] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x128x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
          ttng.async_tma_copy_global_to_local %a_desc_68[%a_78, %a_83] %a_98, %a_95, %true_53 {async_task_id = array<i32: 2>, two_cta = true} : !tt.tensordesc<128x64xbf16, #shared1>, !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable> -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
          %b_99 = ttg.memdesc_index %b_51[%a_87] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %b_100 = arith.xori %a_89, %true_53 : i1
          %b_101 = arith.extui %b_100 : i1 to i32
          ttng.wait_barrier %b_99, %b_101 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %b_102 = ttg.memdesc_index %b_47[%a_87] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %b_103 = ttng.map_to_remote_buffer %b_102, %a_94 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
          ttng.barrier_expect %b_102, 32768 {async_task_id = array<i32: 2>}, %a_97 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %b_104 = ttg.memdesc_index %b_48[%a_87] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x64x64xf32, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared2, #smem, mutable>
          ttng.async_tma_copy_global_to_local %b_desc_72[%a_83, %b_80] %b_104, %b_103, %true_53 {async_task_id = array<i32: 2>, two_cta = true} : !tt.tensordesc<64x64xf32, #shared2>, !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable> -> !ttg.memdesc<64x64xf32, #shared2, #smem, mutable>
          %accum_cnt = arith.addi %arg29, %c1_i64_63 {async_task_id = array<i32: 2>} : i64
          scf.yield {async_task_id = array<i32: 2>} %accum_cnt : i64
        } {async_task_id = array<i32: 2>, tt.warp_specialize}
        scf.yield {async_task_id = array<i32: 2>} %acc_81 : i64
      } {async_task_id = array<i32: 2>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    } : (i32, i32, i32, !ttg.memdesc<4x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<3x64x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<3x128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, !ttg.memdesc<4x1xi64, #shared, #smem, mutable>, !tt.ptr<bf16>, !tt.ptr<f32>, !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, !ttg.memdesc<3x64x64xf32, #shared2, #smem, mutable>, !ttg.memdesc<4x1xi64, #shared, #smem, mutable>, !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, !ttg.memdesc<3x1xi64, #shared, #smem, mutable>) -> ()
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %3 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_11 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_12 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_13 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %a_3 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %a_4 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %a_5 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %a_7 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %a_8 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %a_9 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %acc_14 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %acc_15 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %acc_16 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %acc_17 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %acc_19 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %acc_20 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %acc_21 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %acc_22 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_24 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_25 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_26 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_28 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_29 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b_30 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}


