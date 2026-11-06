// RUN: triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=1 | FileCheck %s

// Slicing a descriptor store along M for partition 1 creates off_m + 128 for
// its row coordinate. That shifted value must stay local to the store: the
// later pointer index splat(off_m) + make_range is sliced with a [128, 256)
// range, so it must still splat the unshifted off_m. Leaking the shifted
// coordinate through the shared IRMapping applied the M offset twice and
// stored rows 128-255 of every tile 128 rows too low.
//
// Reduced from a persistent BM=256 GEMM with data_partition_factor=2: a bf16
// TMA store of the accumulator plus a pointer-stored row sum of the stored
// value. %vf is placed before the convert_layout feeding the descriptor store
// so the descriptor store is sliced before the pointer store's operands.

// CHECK-LABEL: @desc_store_coord_does_not_leak_into_pointer_index
// CHECK: %[[OFFM:.*]] = arith.muli %{{.*}}, %c256_i32 : i32
// CHECK: %[[LDOFF:.*]] = arith.addi %[[OFFM]], %c128_i32 : i32
// CHECK: %[[STOFF:.*]] = arith.addi %[[OFFM]], %c128_i32 : i32
// CHECK: tt.descriptor_load %{{.*}}[%[[OFFM]], %{{.*}}]
// CHECK: tt.descriptor_load %{{.*}}[%[[LDOFF]], %{{.*}}]
// CHECK: tt.splat %[[OFFM]] : i32 -> tensor<128xi32
// CHECK: tt.descriptor_store %{{.*}}[%[[OFFM]], %c0_i32]
// CHECK-NOT: tt.splat %[[STOFF]]
// CHECK-NOT: tt.splat %[[LDOFF]]
// CHECK: tt.splat %[[OFFM]] : i32 -> tensor<128xi32
// CHECK: tt.descriptor_store %{{.*}}[%[[STOFF]], %c0_i32]

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @desc_store_coord_does_not_leak_into_pointer_index(%x_ptr: !tt.ptr<bf16>, %y_ptr: !tt.ptr<bf16>, %out_ptr: !tt.ptr<bf16>, %r1_ptr: !tt.ptr<f32>) {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear>
    %x_desc = tt.make_tensor_descriptor %x_ptr, [%c4096_i32, %c256_i32], [%c256_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<256x64xbf16, #shared>
    %y_desc = tt.make_tensor_descriptor %y_ptr, [%c128_i32, %c256_i32], [%c256_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared>
    %o_desc = tt.make_tensor_descriptor %out_ptr, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<256x128xbf16, #shared>
    %pid = tt.get_program_id x : i32
    %rows = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %r1_base = tt.splat %r1_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    scf.for %t = %pid to %c16_i32 step %c16_i32  : i32 {
      %off_m = arith.muli %t, %c256_i32 : i32
      %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_init = ttng.tmem_store %cst_0, %acc[%acc_tok], %true : tensor<256x128xf32, #linear> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %loop:2 = scf.for %ki = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%use_acc = %false, %tok = %acc_init) -> (i1, !ttg.async.token)  : i32 {
        %k = arith.muli %ki, %c64_i32 : i32
        %a = tt.descriptor_load %x_desc[%off_m, %k] : !tt.tensordesc<256x64xbf16, #shared> -> tensor<256x64xbf16, #blocked1>
        %a_smem = ttg.local_alloc %a : (tensor<256x64xbf16, #blocked1>) -> !ttg.memdesc<256x64xbf16, #shared, #smem>
        %b = tt.descriptor_load %y_desc[%c0_i32, %k] : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked1>
        %b_smem = ttg.local_alloc %b : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %b_t = ttg.memdesc_trans %b_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared1, #smem>
        %mma = ttng.tc_gen5_mma %a_smem, %b_t, %acc[%tok], %use_acc, %true : !ttg.memdesc<256x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared1, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %mma : i1, !ttg.async.token
      }
      %rows_splat = tt.splat %off_m : i32 -> tensor<256xi32, #blocked>
      %row_idx = arith.addi %rows_splat, %rows : tensor<256xi32, #blocked>
      %acc_val, %acc_ld_tok = ttng.tmem_load %acc[%loop#1] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
      %vb = arith.truncf %acc_val : tensor<256x128xf32, #linear> to tensor<256x128xbf16, #linear>
      %vf = arith.extf %vb : tensor<256x128xbf16, #linear> to tensor<256x128xf32, #linear>
      %vb_cvt = ttg.convert_layout %vb : tensor<256x128xbf16, #linear> -> tensor<256x128xbf16, #blocked2>
      tt.descriptor_store %o_desc[%off_m, %c0_i32], %vb_cvt : !tt.tensordesc<256x128xbf16, #shared>, tensor<256x128xbf16, #blocked2>
      %sq = arith.mulf %vf, %vf : tensor<256x128xf32, #linear>
      %rsum = "tt.reduce"(%sq) <{axis = 1 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        tt.reduce.return %sum : f32
      }) : (tensor<256x128xf32, #linear>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %ss = arith.addf %rsum, %cst : tensor<256xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %r1_ptrs = tt.addptr %r1_base, %row_idx : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
      %ss_cvt = ttg.convert_layout %ss : tensor<256xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<256xf32, #blocked>
      tt.store %r1_ptrs, %ss_cvt : tensor<256x!tt.ptr<f32>, #blocked>
    } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
