// RUN: triton-opt %s -split-input-file --triton-nvidia-mma-collector-reuse | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_tail = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, colStride = 1>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Two MMAs share A: re-issued per 16-wide K slice, A filled then last-used,
  // the original use_d kept on the first slice only, the barrier on the end.
  // CHECK-LABEL: @shared_a
  // CHECK: %[[PRED:.*]] = arith.constant true
  // CHECK: %[[ONE:.*]] = arith.constant true
  // CHECK: %[[A0:.*]] = ttg.memdesc_subslice %[[A:.*]][0, 0]
  // CHECK: %[[B0:.*]] = ttg.memdesc_subslice %[[B:.*]][0, 0]
  // CHECK: ttng.tc_gen5_mma %[[A0]], %[[B0]], %[[D:.*]], %[[USE:.*]], %[[PRED]] {is_async, ttng.collector_a = "fill"}
  // CHECK: %[[T0:.*]] = ttg.memdesc_subslice %[[BT:.*]][0, 0]
  // CHECK: ttng.tc_gen5_mma %[[A0]], %[[T0]], %[[DT:.*]], %[[USE]], %[[PRED]] {is_async, ttng.collector_a = "lastuse"}
  // CHECK: %[[A1:.*]] = ttg.memdesc_subslice %[[A]][0, 16]
  // CHECK: %[[B1:.*]] = ttg.memdesc_subslice %[[B]][16, 0]
  // CHECK: ttng.tc_gen5_mma %[[A1]], %[[B1]], %[[D]], %[[ONE]], %[[PRED]] {is_async, ttng.collector_a = "fill"}
  // CHECK-COUNT-4: ttng.tc_gen5_mma
  // CHECK: ttng.tc_gen5_mma {{.*}}, %[[DT]], %[[ONE]], %[[PRED]], %{{.*}}[%[[PRED]]] {is_async, ttng.collector_a = "lastuse"}
  // CHECK-NOT: ttng.tc_gen5_mma
  tt.func public @shared_a(
    %a: !ttg.memdesc<128x64xbf16, #shared, #smem>,
    %b: !ttg.memdesc<64x128xbf16, #shared, #smem>,
    %bt: !ttg.memdesc<64x32xbf16, #shared_t, #smem>,
    %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %dt: !ttg.memdesc<128x32xf32, #tmem_tail, #ttng.tensor_memory, mutable>,
    %use_d: i1,
    %bar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    %true = arith.constant true
    ttng.tc_gen5_mma %a, %b, %d, %use_d, %true {is_async} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %bt, %dt, %use_d, %true, %bar[%true] {is_async} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x32xbf16, #shared_t, #smem>, !ttg.memdesc<128x32xf32, #tmem_tail, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }

  // Different A operands: nothing to share.
  // CHECK-LABEL: @different_a
  // CHECK-NOT: collector_a
  // CHECK: tt.return
  tt.func public @different_a(
    %a: !ttg.memdesc<128x64xbf16, #shared, #smem>,
    %a2: !ttg.memdesc<128x64xbf16, #shared, #smem>,
    %b: !ttg.memdesc<64x128xbf16, #shared, #smem>,
    %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %d2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    ttng.tc_gen5_mma %a, %b, %d, %true, %true {is_async} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a2, %b, %d2, %true, %true {is_async} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // A barrier wait between the MMAs orders them, so they are not re-issued.
  // CHECK-LABEL: @side_effect_between
  // CHECK-NOT: collector_a
  // CHECK: tt.return
  tt.func public @side_effect_between(
    %a: !ttg.memdesc<128x64xbf16, #shared, #smem>,
    %b: !ttg.memdesc<64x128xbf16, #shared, #smem>,
    %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %d2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %bar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
    %phase: i32) {
    %true = arith.constant true
    ttng.tc_gen5_mma %a, %b, %d, %true, %true {is_async} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.tc_gen5_mma %a, %b, %d2, %true, %true {is_async} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // Synchronous MMAs wait for completion and are left alone.
  // CHECK-LABEL: @sync_mma
  // CHECK-NOT: collector_a
  // CHECK: tt.return
  tt.func public @sync_mma(
    %a: !ttg.memdesc<128x64xbf16, #shared, #smem>,
    %b: !ttg.memdesc<64x128xbf16, #shared, #smem>,
    %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %d2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    ttng.tc_gen5_mma %a, %b, %d, %true, %true : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %d2, %true, %true : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
