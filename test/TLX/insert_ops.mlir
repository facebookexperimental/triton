// RUN: triton-opt -split-input-file -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=cuda:90 num-ctas=1 threads-per-warp=32})' %s| FileCheck %s


#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tlx_bar_init
  tt.func public @tlx_bar_init(%a: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
                                                     %b: !ttg.memdesc<128x64xf16, #shared1, #ttg.shared_memory>,
                                                     %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                                                     %useAcc: i1,
                                                     %pred: i1,
                                                     %barrier: !ttg.memdesc<1xi64, #shared, #ttg.shared_memory>,
                                                     %barrierPred: i1) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttng.init_barrier
    // CHECK: ttng.cluster_arrive {relaxed = false}
    // CHECK: ttng.cluster_wait
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %2 = ttng.map_to_remote_buffer %1, %c0_i32 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %2, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async, two_ctas}:
           !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
           !ttg.memdesc<128x64xf16, #shared1, #ttg.shared_memory>,
           !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
           !ttg.memdesc<1xi64, #shared, #ttg.shared_memory>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tlx_bar_init_ws_partition
  tt.func public @tlx_bar_init_ws_partition(%4: !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>,
                                                %5: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>,
                                                %6: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttng.init_barrier
    // CHECK: ttng.cluster_arrive {relaxed = false}
    // CHECK: ttng.cluster_wait
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize(%4, %6, %5, %0)
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg2: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg3: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(1) {
      %true = arith.constant true
      %false = arith.constant false
      %c0_i32_0 = arith.constant 0 : i32
      %7 = ttg.memdesc_index %arg3[%c0_i32_0] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %8 = ttng.map_to_remote_buffer %7, %c0_i32_0 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttng.arrive_barrier %8, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttng.tc_gen5_mma %arg0, %arg2, %arg1[], %false, %true {two_ctas} : !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tlx_bar_init_ws_default
  tt.func public @tlx_bar_init_ws_default(%arg0: !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>,
                                                %arg1: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>,
                                                %arg2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttng.init_barrier
    // CHECK: ttng.cluster_arrive {relaxed = false}
    // CHECK: ttng.cluster_wait
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize()
    default {
      %true = arith.constant true
      %false = arith.constant false
      %c0_i32_0 = arith.constant 0 : i32
      %7 = ttg.memdesc_index %0[%c0_i32_0] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %8 = ttng.map_to_remote_buffer %7, %c0_i32_0 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttng.arrive_barrier %8, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttng.tc_gen5_mma %arg0, %arg1, %arg2, %false, %true {two_ctas} : !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module {
  // CHECK-LABEL: @tlx_bar_init_for_block
  tt.func public @tlx_bar_init_for_block(%arg0: !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>,
                                                %arg1: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>,
                                                %arg2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttng.init_barrier
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c1_i32 = arith.constant 1 : i32
    %2 = ttg.memdesc_index %0[%c1_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttng.init_barrier
    // CHECK: ttng.cluster_arrive {relaxed = false}
    // CHECK: ttng.cluster_wait
    ttng.init_barrier %2, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c0_i32_0 = arith.constant 0 : i32
    %c300_i32 = arith.constant 300 : i32
    %c1_i32_1 = arith.constant 1 : i32

    scf.for %arg6 = %c0_i32_0 to %c300_i32 step %c1_i32_1  : i32 {
      %8 = ttg.memdesc_index %0[%arg6] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %c0_i32_3 = arith.constant 0 : i32
      %9 = ttng.map_to_remote_buffer %8, %c0_i32_3 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttng.arrive_barrier %9, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    }
    %true = arith.constant true
    %false = arith.constant false
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %false, %true {two_ctas} : !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
