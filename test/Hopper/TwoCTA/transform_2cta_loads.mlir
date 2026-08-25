// RUN: triton-opt %s -split-input-file --nvgpu-2cta-transform-loads | FileCheck %s
// Cooperative marking is a performance choice; the knob selects the per-CTA
// protocol for A/B benchmarking while leaving B splitting in place.
// RUN: env TRITON_DISABLE_2CTA_COOPERATIVE_LOADS=1 triton-opt %s -split-input-file --nvgpu-2cta-transform-loads | FileCheck %s --check-prefix=PERCTA

// Test: Transform B descriptor loads for 2-CTA mode.
// When TCGen5MMAOp has two_ctas=true, the pass should:
// 1. Clone the MakeTensorDescOp with half-width block shape (64x64 instead of 64x128)
// 2. Insert ClusterCTAIdOp + CTA-based offset computation
// 3. Create new DescriptorLoadOp with half-width result

// CHECK-LABEL: @matmul_2cta_transform_loads
// PERCTA-LABEL: @matmul_2cta_transform_loads
// PERCTA: ttng.tc_gen5_mma
// PERCTA-NOT: two_cta_load
// Two make_tensor_descriptor ops: original A and cloned half-width B
// CHECK: tt.make_tensor_descriptor %{{.*}} : !tt.ptr<f16>, !tt.tensordesc<128x64xf16>
// CHECK: tt.make_tensor_descriptor %{{.*}} : !tt.ptr<f16>, !tt.tensordesc<64x64xf16>
// Rank-2 A operand also uses the hardware CTA-group TMA protocol.
// CHECK: tt.descriptor_load %{{.*}} {{.*}}two_cta_load{{.*}} : !tt.tensordesc<128x64xf16>
// CTA offset computation inside the loop
// CHECK: %[[CTA_ID:.*]] = nvg.cluster_id
// CHECK: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: %[[MOD:.*]] = arith.remsi %[[CTA_ID]], %[[C2]]
// CHECK: %[[HALF:.*]] = arith.constant 64 : i32
// CHECK: %[[OFF:.*]] = arith.muli %[[MOD]], %[[HALF]]
// CHECK: arith.addi %{{.*}}, %[[OFF]]
// Rank-2 B uses the same protocol. Rank-1 metadata remains per-CTA.
// CHECK: tt.descriptor_load %{{.*}} {{.*}}two_cta_load{{.*}} : !tt.tensordesc<64x64xf16>
// Half-width B alloc
// CHECK: ttg.local_alloc %{{.*}} : {{.*}} -> !ttg.memdesc<64x64xf16
// CHECK: tt.descriptor_load %{{.*}} : !tt.tensordesc<128xf32>
// MMA with two_ctas and half-width B
// CHECK: ttng.tc_gen5_mma {{.*}} {two_ctas}

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_2cta_transform_loads(
      %a_ptr: !tt.ptr<f16>,
      %b_ptr: !tt.ptr<f16>,
      %m_desc: !tt.tensordesc<128xf32>,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cN_i64 = arith.extsi %N : i32 to i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>

    %pid = tt.get_program_id x : i32
    %offs_am = arith.muli %pid, %c128_i32 : i32
    %offs_bn = arith.muli %pid, %c128_i32 : i32

    // A descriptor: 128x64 block
    %a_desc = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%cN_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x64xf16>
    // B descriptor: 64x128 block (pass should clone this with 64x64)
    %b_desc = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%cN_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x128xf16>

    %accumulator = scf.for %k = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : i32 {
      %offs_k = arith.muli %k, %c64_i32 : i32

      // A load (not modified)
      %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked1>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

      // B load (should be transformed to half-width with CTA offset)
      %b = tt.descriptor_load %b_desc[%offs_k, %offs_bn] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %m = tt.descriptor_load %m_desc[%offs_am] : !tt.tensordesc<128xf32> -> tensor<128xf32>

      %acc_layout = ttg.convert_layout %acc : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked3>
      %acc_tmem, %token = ttng.tmem_alloc %acc_layout : (tensor<128x128xf32, #blocked3>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // 2-CTA MMA: pass should transform B load above
      %mma_token = ttng.tc_gen5_mma %a_smem, %b_smem, %acc_tmem[%token], %true, %true {two_ctas} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      %result, %load_token = ttng.tmem_load %acc_tmem[%mma_token] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked3>
      %result_layout = ttg.convert_layout %result : tensor<128x128xf32, #blocked3> -> tensor<128x128xf32, #blocked>

      scf.yield %result_layout : tensor<128x128xf32, #blocked>
    }

    tt.return
  }
}

// -----

// Test: A single full V descriptor load split between two 2-CTA PV MMAs must
// become one cooperative half-width load and two zero-copy shared-memory
// views. This avoids two TMA loads without staging V through registers.
// CHECK-LABEL: @fa_split_v_load
// CHECK-SAME: %[[DESC:.*]]: !tt.tensordesc<128x64xbf16>
// CHECK: %[[V:.*]] = tt.descriptor_load %[[DESC]]{{.*}}two_cta_b{{.*}} : !tt.tensordesc<128x64xbf16>
// CHECK: %[[FULL:.*]] = ttg.local_alloc %[[V]] : {{.*}} -> !ttg.memdesc<128x64xbf16
// CHECK: %[[V0:.*]] = ttg.memdesc_subslice %[[FULL]][0, 0] : {{.*}} -> !ttg.memdesc<64x64xbf16, {{.*}}, {{.*}}, 128x64>
// CHECK: %[[V1:.*]] = ttg.memdesc_subslice %[[FULL]][64, 0] : {{.*}} -> !ttg.memdesc<64x64xbf16, {{.*}}, {{.*}}, 128x64>
// CHECK-NOT: tt.split
// CHECK: ttng.tc_gen5_mma %{{.*}}, %[[V0]], {{.*}} {two_ctas}
// CHECK: ttng.tc_gen5_mma %{{.*}}, %[[V1]], {{.*}} {two_ctas}

#v_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#v_reshape = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#v_trans = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [2, 16, 1], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>
#v_leaf = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [8, 0], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0]], warp = [[2, 0], [4, 0]], block = []}>
#v_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#v_smem = #ttg.shared_memory
#v_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fa_split_v_load(
      %v_desc: !tt.tensordesc<128x128xbf16>,
      %a0: !ttg.memdesc<256x64xbf16, #v_shared, #v_smem>,
      %a1: !ttg.memdesc<256x64xbf16, #v_shared, #v_smem>,
      %acc0: !ttg.memdesc<256x128xf32, #v_tmem, #ttng.tensor_memory, mutable>,
      %acc1: !ttg.memdesc<256x128xf32, #v_tmem, #ttng.tensor_memory, mutable>,
      %token0: !ttg.async.token,
      %token1: !ttg.async.token) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32

    %v = tt.descriptor_load %v_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #v_blocked>
    %v_reshape_value = tt.reshape %v : tensor<128x128xbf16, #v_blocked> -> tensor<2x64x128xbf16, #v_reshape>
    %v_trans_value = tt.trans %v_reshape_value {order = array<i32: 1, 2, 0>} : tensor<2x64x128xbf16, #v_reshape> -> tensor<64x128x2xbf16, #v_trans>
    %v0, %v1 = tt.split %v_trans_value : tensor<64x128x2xbf16, #v_trans> -> tensor<64x128xbf16, #v_leaf>
    %v0_smem = ttg.local_alloc %v0 : (tensor<64x128xbf16, #v_leaf>) -> !ttg.memdesc<64x128xbf16, #v_shared, #v_smem>
    %v1_smem = ttg.local_alloc %v1 : (tensor<64x128xbf16, #v_leaf>) -> !ttg.memdesc<64x128xbf16, #v_shared, #v_smem>

    %mma0 = ttng.tc_gen5_mma %a0, %v0_smem, %acc0[%token0], %true, %true {two_ctas} : !ttg.memdesc<256x64xbf16, #v_shared, #v_smem>, !ttg.memdesc<64x128xbf16, #v_shared, #v_smem>, !ttg.memdesc<256x128xf32, #v_tmem, #ttng.tensor_memory, mutable>
    %mma1 = ttng.tc_gen5_mma %a1, %v1_smem, %acc1[%token1], %true, %true {two_ctas} : !ttg.memdesc<256x64xbf16, #v_shared, #v_smem>, !ttg.memdesc<64x128xbf16, #v_shared, #v_smem>, !ttg.memdesc<256x128xf32, #v_tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Test: the same split chain, but the descriptor argument also feeds a second,
// unrelated descriptor_load. With no MakeTensorDescOp to clone, the transform
// retypes the argument in place, so firing here would silently hand the other
// load a half-width descriptor and make it address the wrong tile. The pass
// must bail and leave the split chain intact.
// CHECK-LABEL: @fa_split_v_load_shared_desc
// CHECK-SAME: %[[DESC:.*]]: !tt.tensordesc<128x128xbf16>
// CHECK: tt.split
// CHECK-NOT: ttg.memdesc_subslice

#v2_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#v2_reshape = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#v2_trans = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [2, 16, 1], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>
#v2_leaf = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [8, 0], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0]], warp = [[2, 0], [4, 0]], block = []}>
#v2_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#v2_smem = #ttg.shared_memory
#v2_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fa_split_v_load_shared_desc(
      %v_desc: !tt.tensordesc<128x128xbf16>,
      %a0: !ttg.memdesc<256x64xbf16, #v2_shared, #v2_smem>,
      %a1: !ttg.memdesc<256x64xbf16, #v2_shared, #v2_smem>,
      %acc0: !ttg.memdesc<256x128xf32, #v2_tmem, #ttng.tensor_memory, mutable>,
      %acc1: !ttg.memdesc<256x128xf32, #v2_tmem, #ttng.tensor_memory, mutable>,
      %token0: !ttg.async.token,
      %token1: !ttg.async.token) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32

    %v = tt.descriptor_load %v_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #v2_blocked>
    %v_reshape_value = tt.reshape %v : tensor<128x128xbf16, #v2_blocked> -> tensor<2x64x128xbf16, #v2_reshape>
    %v_trans_value = tt.trans %v_reshape_value {order = array<i32: 1, 2, 0>} : tensor<2x64x128xbf16, #v2_reshape> -> tensor<64x128x2xbf16, #v2_trans>
    %v0, %v1 = tt.split %v_trans_value : tensor<64x128x2xbf16, #v2_trans> -> tensor<64x128xbf16, #v2_leaf>
    %v0_smem = ttg.local_alloc %v0 : (tensor<64x128xbf16, #v2_leaf>) -> !ttg.memdesc<64x128xbf16, #v2_shared, #v2_smem>
    %v1_smem = ttg.local_alloc %v1 : (tensor<64x128xbf16, #v2_leaf>) -> !ttg.memdesc<64x128xbf16, #v2_shared, #v2_smem>

    // Second consumer of the same descriptor.
    %v_other = tt.descriptor_load %v_desc[%c128_i32, %c0_i32] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #v2_blocked>
    %v_other_smem = ttg.local_alloc %v_other : (tensor<128x128xbf16, #v2_blocked>) -> !ttg.memdesc<128x128xbf16, #v2_shared, #v2_smem>

    %mma0 = ttng.tc_gen5_mma %a0, %v0_smem, %acc0[%token0], %true, %true {two_ctas} : !ttg.memdesc<256x64xbf16, #v2_shared, #v2_smem>, !ttg.memdesc<64x128xbf16, #v2_shared, #v2_smem>, !ttg.memdesc<256x128xf32, #v2_tmem, #ttng.tensor_memory, mutable>
    %mma1 = ttng.tc_gen5_mma %a1, %v1_smem, %acc1[%token1], %true, %true {two_ctas} : !ttg.memdesc<256x64xbf16, #v2_shared, #v2_smem>, !ttg.memdesc<64x128xbf16, #v2_shared, #v2_smem>, !ttg.memdesc<256x128xf32, #v2_tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Test: Host-side TMA descriptor (function argument) with 2-CTA.
// The pass should update the argument's TensorDescType to half-width and
// transform the load, same as device-side but without cloning MakeTensorDescOp.
// CHECK-LABEL: @matmul_2cta_host_tma
// The function argument type should be updated to half-width
// CHECK-SAME: !tt.tensordesc<64x64xf16>
// CTA offset computation
// CHECK: %[[CTA_ID:.*]] = nvg.cluster_id
// CHECK: arith.remsi
// CHECK: arith.muli
// CHECK: arith.addi
// Half-width B load
// CHECK: tt.descriptor_load %{{.*}} : !tt.tensordesc<64x64xf16>
// CHECK: ttg.local_alloc %{{.*}} : {{.*}} -> !ttg.memdesc<64x64xf16
// CHECK: ttng.tc_gen5_mma {{.*}} {two_ctas}

#blocked_h = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1_h = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3_h = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_h = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem_h = #ttg.shared_memory
#tmem_h = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_2cta_host_tma(
      %a_desc: !tt.tensordesc<128x64xf16>,
      %b_desc: !tt.tensordesc<64x128xf16>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_h>

    %pid = tt.get_program_id x : i32
    %offs_am = arith.muli %pid, %c128_i32 : i32
    %offs_bn = arith.muli %pid, %c128_i32 : i32

    %accumulator = scf.for %k = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked_h>) : i32 {
      %offs_k = arith.muli %k, %c64_i32 : i32

      %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked1_h>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked1_h>) -> !ttg.memdesc<128x64xf16, #shared_h, #smem_h>

      // Host-side B descriptor — pass should update the func arg type to half-width
      %b = tt.descriptor_load %b_desc[%offs_k, %offs_bn] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1_h>
      %b_smem = ttg.local_alloc %b : (tensor<64x128xf16, #blocked1_h>) -> !ttg.memdesc<64x128xf16, #shared_h, #smem_h>

      %acc_layout = ttg.convert_layout %acc : tensor<128x128xf32, #blocked_h> -> tensor<128x128xf32, #blocked3_h>
      %acc_tmem, %token = ttng.tmem_alloc %acc_layout : (tensor<128x128xf32, #blocked3_h>) -> (!ttg.memdesc<128x128xf32, #tmem_h, #ttng.tensor_memory, mutable>, !ttg.async.token)

      %mma_token = ttng.tc_gen5_mma %a_smem, %b_smem, %acc_tmem[%token], %true, %true {two_ctas} : !ttg.memdesc<128x64xf16, #shared_h, #smem_h>, !ttg.memdesc<64x128xf16, #shared_h, #smem_h>, !ttg.memdesc<128x128xf32, #tmem_h, #ttng.tensor_memory, mutable>

      %result, %load_token = ttng.tmem_load %acc_tmem[%mma_token] : !ttg.memdesc<128x128xf32, #tmem_h, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked3_h>
      %result_layout = ttg.convert_layout %result : tensor<128x128xf32, #blocked3_h> -> tensor<128x128xf32, #blocked_h>

      scf.yield %result_layout : tensor<128x128xf32, #blocked_h>
    }

    tt.return
  }
}

// -----

// Test: Multiple B loads from the same host-side TMA descriptor should all use
// the same original block shape. The descriptor argument must be halved once,
// not once per MMA user.
// CHECK-LABEL: @matmul_2cta_host_tma_reused_desc
// CHECK-SAME: !tt.tensordesc<64x64xf16>
// CHECK-NOT: tensor<64x32xf16>
// CHECK: tt.descriptor_load %{{.*}} : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16
// CHECK: tt.descriptor_load %{{.*}} : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16
// CHECK: ttng.tc_gen5_mma {{.*}} {two_ctas}
// CHECK: ttng.tc_gen5_mma {{.*}} {two_ctas}

#blocked_r = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1_r = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3_r = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_r = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem_r = #ttg.shared_memory
#tmem_r = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_2cta_host_tma_reused_desc(
      %a_desc: !tt.tensordesc<128x64xf16>,
      %b_desc: !tt.tensordesc<64x128xf16>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_r>

    %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked1_r>
    %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked1_r>) -> !ttg.memdesc<128x64xf16, #shared_r, #smem_r>

    %b0 = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1_r>
    %b0_smem = ttg.local_alloc %b0 : (tensor<64x128xf16, #blocked1_r>) -> !ttg.memdesc<64x128xf16, #shared_r, #smem_r>

    %b1 = tt.descriptor_load %b_desc[%c64_i32, %c128_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1_r>
    %b1_smem = ttg.local_alloc %b1 : (tensor<64x128xf16, #blocked1_r>) -> !ttg.memdesc<64x128xf16, #shared_r, #smem_r>

    %acc_layout0 = ttg.convert_layout %cst : tensor<128x128xf32, #blocked_r> -> tensor<128x128xf32, #blocked3_r>
    %acc_tmem0, %token0 = ttng.tmem_alloc %acc_layout0 : (tensor<128x128xf32, #blocked3_r>) -> (!ttg.memdesc<128x128xf32, #tmem_r, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_token0 = ttng.tc_gen5_mma %a_smem, %b0_smem, %acc_tmem0[%token0], %true, %true {two_ctas} : !ttg.memdesc<128x64xf16, #shared_r, #smem_r>, !ttg.memdesc<64x128xf16, #shared_r, #smem_r>, !ttg.memdesc<128x128xf32, #tmem_r, #ttng.tensor_memory, mutable>

    %acc_layout1 = ttg.convert_layout %cst : tensor<128x128xf32, #blocked_r> -> tensor<128x128xf32, #blocked3_r>
    %acc_tmem1, %token1 = ttng.tmem_alloc %acc_layout1 : (tensor<128x128xf32, #blocked3_r>) -> (!ttg.memdesc<128x128xf32, #tmem_r, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_token1 = ttng.tc_gen5_mma %a_smem, %b1_smem, %acc_tmem1[%token1], %true, %true {two_ctas} : !ttg.memdesc<128x64xf16, #shared_r, #smem_r>, !ttg.memdesc<64x128xf16, #shared_r, #smem_r>, !ttg.memdesc<128x128xf32, #tmem_r, #ttng.tensor_memory, mutable>

    tt.return
  }
}

// -----

// Test: No two_ctas attribute - pass should not modify anything
// CHECK-LABEL: @matmul_no_2cta
// CHECK-NOT: nvg.cluster_id
// CHECK: ttng.tc_gen5_mma
// CHECK-NOT: two_ctas

#blocked_b = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3_b = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem_b = #ttg.shared_memory
#tmem_b = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_no_2cta(
      %a_ptr: !tt.ptr<f16>,
      %b_ptr: !tt.ptr<f16>,
      %M: i32, %N: i32, %K: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cN_i64 = arith.extsi %N : i32 to i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_b>

    %pid = tt.get_program_id x : i32
    %offs = arith.muli %pid, %c128_i32 : i32

    %a_desc = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%cN_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x64xf16>
    %b_desc = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%cN_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x128xf16>

    %a = tt.descriptor_load %a_desc[%offs, %c0_i32] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked1_b>
    %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked1_b>) -> !ttg.memdesc<128x64xf16, #shared_b, #smem_b>

    %b = tt.descriptor_load %b_desc[%c0_i32, %offs] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked2_b>
    %b_smem = ttg.local_alloc %b : (tensor<64x128xf16, #blocked2_b>) -> !ttg.memdesc<64x128xf16, #shared_b, #smem_b>

    %acc_layout = ttg.convert_layout %cst : tensor<128x128xf32, #blocked_b> -> tensor<128x128xf32, #blocked3_b>
    %acc_tmem, %token = ttng.tmem_alloc %acc_layout : (tensor<128x128xf32, #blocked3_b>) -> (!ttg.memdesc<128x128xf32, #tmem_b, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // No two_ctas - pass should not modify
    %mma_token = ttng.tc_gen5_mma %a_smem, %b_smem, %acc_tmem[%token], %true, %true : !ttg.memdesc<128x64xf16, #shared_b, #smem_b>, !ttg.memdesc<64x128xf16, #shared_b, #smem_b>, !ttg.memdesc<128x128xf32, #tmem_b, #ttng.tensor_memory, mutable>

    %result, %load_token = ttng.tmem_load %acc_tmem[%mma_token] : !ttg.memdesc<128x128xf32, #tmem_b, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked3_b>
    tt.return
  }
}

// -----

// Test: halving the B block along the contiguous dimension must shrink the
// swizzle byte width to one that is legal for the halved shape. A 64-element
// fp16 contiguous dimension supports a 128B swizzle, but its 32-element half
// only supports 64B; carrying the 128B swizzle over produces an invalid
// NVMMASharedLayout ("block shape along the contiguous dimension 1 is too
// small for the swizzle byte size 128").
// CHECK-LABEL: @matmul_2cta_shrink_swizzle
// CHECK-SAME: !tt.tensordesc<64x32xf16, #[[SW64:[a-z0-9_]+]]>
// CHECK: tt.descriptor_load %{{.*}} : !tt.tensordesc<64x32xf16, #[[SW64]]>
// CHECK: ttg.local_alloc %{{.*}} -> !ttg.memdesc<64x32xf16, #[[SW64]]
// CHECK: ttng.tc_gen5_mma {{.*}} {two_ctas}

#blocked_s = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1_s = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3_s = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_s = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem_s = #ttg.shared_memory
#tmem_s = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_2cta_shrink_swizzle(
      %a_desc: !tt.tensordesc<128x64xf16, #shared_s>,
      %b_desc: !tt.tensordesc<64x64xf16, #shared_s>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked_s>

    %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf16, #shared_s> -> tensor<128x64xf16, #blocked1_s>
    %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked1_s>) -> !ttg.memdesc<128x64xf16, #shared_s, #smem_s>

    %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16, #shared_s> -> tensor<64x64xf16, #blocked1_s>
    %b_smem = ttg.local_alloc %b : (tensor<64x64xf16, #blocked1_s>) -> !ttg.memdesc<64x64xf16, #shared_s, #smem_s>

    %acc_layout = ttg.convert_layout %cst : tensor<128x64xf32, #blocked_s> -> tensor<128x64xf32, #blocked3_s>
    %acc_tmem, %token = ttng.tmem_alloc %acc_layout : (tensor<128x64xf32, #blocked3_s>) -> (!ttg.memdesc<128x64xf32, #tmem_s, #ttng.tensor_memory, mutable>, !ttg.async.token)

    %mma_token = ttng.tc_gen5_mma %a_smem, %b_smem, %acc_tmem[%token], %true, %true {two_ctas} : !ttg.memdesc<128x64xf16, #shared_s, #smem_s>, !ttg.memdesc<64x64xf16, #shared_s, #smem_s>, !ttg.memdesc<128x64xf32, #tmem_s, #ttng.tensor_memory, mutable>

    tt.return
  }
}
