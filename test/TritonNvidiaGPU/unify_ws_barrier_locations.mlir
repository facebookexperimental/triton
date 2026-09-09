// RUN: triton-opt %s --triton-nvidia-unify-ws-barrier-locations --allow-unregistered-dialect | FileCheck %s
// RUN: env TRITON_DISABLE_WSBARRIER_REORDER=1 triton-opt %s --triton-nvidia-unify-ws-barrier-locations --allow-unregistered-dialect | FileCheck %s --check-prefix=DISABLED

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear64 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#slice64 = #ttg.slice<{dim = 0, parent = #linear64}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#shared1d = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_n64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @unify_cast_broadcast
// CHECK:       ttng.wait_barrier %{{.*}}, %{{.*}}, %{{.*}} {{.*}}dstTask = 3
// CHECK-NEXT:  ttng.wait_barrier %{{.*}}, %{{.*}} {{.*}}dstTask = 1
// CHECK-NEXT:  %[[ACC:.*]] = ttng.tmem_load
// CHECK-NEXT:  ttng.arrive_barrier
// CHECK-NEXT:  %[[BIAS:.*]] = ttg.local_load
// CHECK-NEXT:  ttng.arrive_barrier
// CHECK-NEXT:  %[[EXT:.*]] = arith.extf %[[BIAS]]
// CHECK-NEXT:  %[[CVT:.*]] = ttg.convert_layout %[[EXT]]
// CHECK-NEXT:  %[[BCAST:.*]] = tt.broadcast %[[CVT]]
// CHECK-NEXT:  arith.addf %[[ACC]], %[[BCAST]]
// DISABLED-LABEL: @unify_cast_broadcast
// DISABLED:       ttng.wait_barrier %{{.*}}, %{{.*}} {{.*}}dstTask = 1
// DISABLED-NEXT:  %[[ACC:.*]] = ttng.tmem_load
// DISABLED-NEXT:  ttng.arrive_barrier
// DISABLED-NEXT:  ttng.wait_barrier %{{.*}}, %{{.*}}, %{{.*}} {{.*}}dstTask = 3
// DISABLED-NEXT:  %[[BIAS:.*]] = ttg.local_load
tt.func @unify_cast_broadcast(%desc: !tt.tensordesc<1x128xf16, #shared>) {
  %c0 = arith.constant 0 : i32
  %true = arith.constant true
  %tma_barrier = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %tmem_barrier = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias_buffer = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  %accumulator = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.barrier_expect %tma_barrier, 256, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %bias_buffer, %tma_barrier, %true :
    !tt.tensordesc<1x128xf16, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  ttng.tc_gen5_commit %tmem_barrier : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  ttng.wait_barrier %tma_barrier, %c0, %true {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias = ttg.local_load %bias_buffer : !ttg.memdesc<1x128xf16, #shared, #smem, mutable> -> tensor<1x128xf16, #blocked>
  ttng.arrive_barrier %tma_barrier, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias_f32 = arith.extf %bias : tensor<1x128xf16, #blocked> to tensor<1x128xf32, #blocked>
  %bias_layout = ttg.convert_layout %bias_f32 : tensor<1x128xf32, #blocked> -> tensor<1x128xf32, #linear>
  %bias_tile = tt.broadcast %bias_layout : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
  ttng.wait_barrier %tmem_barrier, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %acc = ttng.tmem_load %accumulator : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
  ttng.arrive_barrier %tmem_barrier, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %out = arith.addf %acc, %bias_tile : tensor<128x128xf32, #linear>
  "use"(%out) : (tensor<128x128xf32, #linear>) -> ()
  tt.return
}

// The FA shape includes expand_dims. First co-locate the waits, then put the
// TMEM load before the SMEM preparation without moving either acquire again.
// This changes register materialization order without lengthening the async
// interval. With global barrier reordering disabled, preserve D114's fallback
// behavior by moving the complete SMEM channel after the TMEM channel.
// CHECK-LABEL: @unify_then_prioritize_tmem
// CHECK:       ttng.wait_barrier %{{.*}} {{.*}}dstTask = 3
// CHECK-NEXT:  ttng.wait_barrier %{{.*}} {{.*}}dstTask = 1
// CHECK-NEXT:  %[[TV:.*]] = ttng.tmem_load
// CHECK-NEXT:  ttng.arrive_barrier {{.*}}dstTask = 1
// CHECK-NEXT:  %[[LV:.*]] = ttg.local_load
// CHECK-NEXT:  ttng.arrive_barrier {{.*}}dstTask = 3
// CHECK-NEXT:  %[[EXPAND:.*]] = tt.expand_dims %[[LV]]
// CHECK-NEXT:  %[[BCAST:.*]] = tt.broadcast %[[EXPAND]]
// CHECK-NEXT:  arith.addf %[[TV]], %[[BCAST]]
// DISABLED-LABEL: @unify_then_prioritize_tmem
// DISABLED:       ttng.wait_barrier %{{.*}} {{.*}}dstTask = 1
// DISABLED-NEXT:  %[[TV:.*]] = ttng.tmem_load
// DISABLED-NEXT:  ttng.arrive_barrier {{.*}}dstTask = 1
// DISABLED-NEXT:  ttng.wait_barrier %{{.*}} {{.*}}dstTask = 3
// DISABLED-NEXT:  %[[LV:.*]] = ttg.local_load
tt.func @unify_then_prioritize_tmem(
    %smem: !ttg.memdesc<64xf32, #shared1d, #smem, mutable>,
    %tmem: !ttg.memdesc<128x64xf32, #tmem_n64, #ttng.tensor_memory, mutable, 128x128>,
    %local_full: !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
    %local_empty: !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
    %tmem_full: !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
    %tmem_empty: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
  %phase = arith.constant 0 : i32
  ttng.wait_barrier %local_full, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, parentId = 1 : i32, minRegionId = 1 : i32, maxRegionId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %local = ttg.local_load %smem : !ttg.memdesc<64xf32, #shared1d, #smem, mutable> -> tensor<64xf32, #slice64>
  ttng.arrive_barrier %local_empty, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, parentId = 1 : i32, minRegionId = 2 : i32, maxRegionId = 2 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %local_row = tt.expand_dims %local {axis = 0 : i32} : tensor<64xf32, #slice64> -> tensor<1x64xf32, #linear64>
  %local_tile = tt.broadcast %local_row : tensor<1x64xf32, #linear64> -> tensor<128x64xf32, #linear64>
  ttng.wait_barrier %tmem_full, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, parentId = 1 : i32, minRegionId = 1 : i32, maxRegionId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %tmem_value = ttng.tmem_load %tmem : !ttg.memdesc<128x64xf32, #tmem_n64, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #linear64>
  ttng.arrive_barrier %tmem_empty, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, parentId = 1 : i32, minRegionId = 2 : i32, maxRegionId = 2 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %sum = arith.addf %tmem_value, %local_tile : tensor<128x64xf32, #linear64>
  "use"(%sum) : (tensor<128x64xf32, #linear64>) -> ()
  tt.return
}

// A broadcast small enough that hoisting the accumulator wait buys no register
// relief does not qualify. The register saving scales with the broadcast value,
// the lost overlap does not, so the waits stay where they are and the bias
// preparation keeps covering MMA latency. Same shape as @unify_cast_broadcast
// except the broadcast result is 1x128 (1 element per thread) rather than
// 128x128 (128 per thread).
// CHECK-LABEL: @keep_small_broadcast
// CHECK:       ttng.wait_barrier %{{.*}}, %{{.*}}, %{{.*}} {{.*}}dstTask = 3
// CHECK-NEXT:  ttg.local_load
// CHECK:       tt.broadcast
// CHECK-NEXT:  ttng.wait_barrier {{.*}}dstTask = 1
tt.func @keep_small_broadcast(%desc: !tt.tensordesc<1x128xf16, #shared>) {
  %c0 = arith.constant 0 : i32
  %true = arith.constant true
  %tma_barrier = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %tmem_barrier = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias_buffer = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  %accumulator = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.barrier_expect %tma_barrier, 256, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %bias_buffer, %tma_barrier, %true :
    !tt.tensordesc<1x128xf16, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  ttng.tc_gen5_commit %tmem_barrier : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  ttng.wait_barrier %tma_barrier, %c0, %true {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias = ttg.local_load %bias_buffer : !ttg.memdesc<1x128xf16, #shared, #smem, mutable> -> tensor<1x128xf16, #blocked>
  ttng.arrive_barrier %tma_barrier, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias_f32 = arith.extf %bias : tensor<1x128xf16, #blocked> to tensor<1x128xf32, #blocked>
  %bias_row = tt.broadcast %bias_f32 : tensor<1x128xf32, #blocked> -> tensor<1x128xf32, #blocked>
  ttng.wait_barrier %tmem_barrier, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %acc = ttng.tmem_load %accumulator : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
  ttng.arrive_barrier %tmem_barrier, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  "use"(%acc) : (tensor<128x128xf32, #linear>) -> ()
  "use"(%bias_row) : (tensor<1x128xf32, #blocked>) -> ()
  tt.return
}

// CHECK-LABEL: @unify_to_fixed_point
// CHECK:       ttng.wait_barrier {{.*}}dstTask = 3
// CHECK-NEXT:  ttng.wait_barrier {{.*}}dstTask = 2
// CHECK-NEXT:  ttng.wait_barrier {{.*}}dstTask = 1
// CHECK-NEXT:  %[[BIAS0:.*]] = ttg.local_load
// CHECK:       tt.broadcast
// CHECK:       %[[BIAS1:.*]] = ttg.local_load
// CHECK:       tt.broadcast
// CHECK:       ttng.tmem_load
tt.func @unify_to_fixed_point() {
  %c0 = arith.constant 0 : i32
  %barrier0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %barrier1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %barrier2 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias_buffer0 = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  %bias_buffer1 = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  %accumulator = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.wait_barrier %barrier0, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias0 = ttg.local_load %bias_buffer0 : !ttg.memdesc<1x128xf16, #shared, #smem, mutable> -> tensor<1x128xf16, #blocked>
  %bias0_f32 = arith.extf %bias0 : tensor<1x128xf16, #blocked> to tensor<1x128xf32, #blocked>
  %bias0_layout = ttg.convert_layout %bias0_f32 : tensor<1x128xf32, #blocked> -> tensor<1x128xf32, #linear>
  %bias0_tile = tt.broadcast %bias0_layout : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
  ttng.wait_barrier %barrier1, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias1 = ttg.local_load %bias_buffer1 : !ttg.memdesc<1x128xf16, #shared, #smem, mutable> -> tensor<1x128xf16, #blocked>
  %bias1_f32 = arith.extf %bias1 : tensor<1x128xf16, #blocked> to tensor<1x128xf32, #blocked>
  %bias1_layout = ttg.convert_layout %bias1_f32 : tensor<1x128xf32, #blocked> -> tensor<1x128xf32, #linear>
  %bias1_tile = tt.broadcast %bias1_layout : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
  ttng.wait_barrier %barrier2, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %acc = ttng.tmem_load %accumulator : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
  %sum0 = arith.addf %acc, %bias0_tile : tensor<128x128xf32, #linear>
  %sum1 = arith.addf %sum0, %bias1_tile : tensor<128x128xf32, #linear>
  "use"(%sum1) : (tensor<128x128xf32, #linear>) -> ()
  tt.return
}

// CHECK-LABEL: @keep_region_without_broadcast
// CHECK:       ttng.wait_barrier
// CHECK-NEXT:  %[[BIAS:.*]] = ttg.local_load
// CHECK-NEXT:  arith.extf %[[BIAS]]
// CHECK-NEXT:  ttg.convert_layout
// CHECK-NEXT:  ttng.wait_barrier
tt.func @keep_region_without_broadcast() {
  %c0 = arith.constant 0 : i32
  %barrier0 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %barrier1 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias_buffer = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  %accumulator = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.wait_barrier %barrier0, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias = ttg.local_load %bias_buffer : !ttg.memdesc<1x128xf16, #shared, #smem, mutable> -> tensor<1x128xf16, #blocked>
  %bias_f32 = arith.extf %bias : tensor<1x128xf16, #blocked> to tensor<1x128xf32, #blocked>
  %bias_layout = ttg.convert_layout %bias_f32 : tensor<1x128xf32, #blocked> -> tensor<1x128xf32, #linear>
  ttng.wait_barrier %barrier1, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %acc = ttng.tmem_load %accumulator : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
  "use"(%bias_layout, %acc) : (tensor<1x128xf32, #linear>, tensor<128x128xf32, #linear>) -> ()
  tt.return
}

// CHECK-LABEL: @keep_non_ws_barrier
// CHECK:       ttng.wait_barrier {{.*}}dstTask = 3
// CHECK-NEXT:  %[[BIAS:.*]] = ttg.local_load
// CHECK:       %[[BCAST:.*]] = tt.broadcast
// CHECK-NEXT:  ttng.wait_barrier
// CHECK-NEXT:  ttng.wait_barrier {{.*}}dstTask = 1
tt.func @keep_non_ws_barrier() {
  %c0 = arith.constant 0 : i32
  %barrier0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %barrier1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %plain_barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias_buffer = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  %accumulator = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.wait_barrier %barrier0, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias = ttg.local_load %bias_buffer : !ttg.memdesc<1x128xf16, #shared, #smem, mutable> -> tensor<1x128xf16, #blocked>
  %bias_f32 = arith.extf %bias : tensor<1x128xf16, #blocked> to tensor<1x128xf32, #blocked>
  %bias_tile = tt.broadcast %bias_f32 : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
  ttng.wait_barrier %plain_barrier, %c0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  ttng.wait_barrier %barrier1, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %acc = ttng.tmem_load %accumulator : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
  "use"(%bias_tile, %acc) : (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #linear>) -> ()
  tt.return
}

// CHECK-LABEL: @keep_substantive_work
// CHECK:       ttng.wait_barrier
// CHECK-NEXT:  ttg.local_load
// CHECK:       arith.addf
// CHECK-NEXT:  ttng.wait_barrier
tt.func @keep_substantive_work(%desc: !tt.tensordesc<1x128xf16, #shared>) {
  %c0 = arith.constant 0 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #linear>
  %tma_barrier = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %tmem_barrier = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias_buffer = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  %accumulator = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.barrier_expect %tma_barrier, 256, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %bias_buffer, %tma_barrier, %true :
    !tt.tensordesc<1x128xf16, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
  ttng.tc_gen5_commit %tmem_barrier : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  ttng.wait_barrier %tma_barrier, %c0, %true {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %bias = ttg.local_load %bias_buffer : !ttg.memdesc<1x128xf16, #shared, #smem, mutable> -> tensor<1x128xf16, #blocked>
  %bias_f32 = arith.extf %bias : tensor<1x128xf16, #blocked> to tensor<1x128xf32, #blocked>
  %bias_layout = ttg.convert_layout %bias_f32 : tensor<1x128xf32, #blocked> -> tensor<1x128xf32, #linear>
  %bias_tile = tt.broadcast %bias_layout : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
  %work = arith.addf %zero, %zero : tensor<128x128xf32, #linear>
  ttng.wait_barrier %tmem_barrier, %c0 {constraints = {WSBarrier = {channelGraph = array<i32: 2>, direction = "forward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %acc = ttng.tmem_load %accumulator : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
  %out = arith.addf %acc, %bias_tile : tensor<128x128xf32, #linear>
  "use"(%out, %work) : (tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear>) -> ()
  tt.return
}

// A foreign arrive between the SMEM load and its release must not be mistaken
// for that channel's release. This rejects both the unified-wait path and the
// disabled-unification fallback.
// CHECK-LABEL: @keep_foreign_release
// CHECK:       ttng.wait_barrier {{.*}}dstTask = 3
// CHECK-NEXT:  %[[LOCAL:.*]] = ttg.local_load
// CHECK-NEXT:  ttng.arrive_barrier {{.*}}parentId = 2
// CHECK-NEXT:  ttng.arrive_barrier {{.*}}parentId = 1
// CHECK:       ttng.wait_barrier {{.*}}dstTask = 1
// CHECK-NEXT:  %[[TMEM:.*]] = ttng.tmem_load
// DISABLED-LABEL: @keep_foreign_release
// DISABLED:       ttng.wait_barrier {{.*}}dstTask = 3
// DISABLED-NEXT:  ttg.local_load
// DISABLED-NEXT:  ttng.arrive_barrier {{.*}}parentId = 2
// DISABLED-NEXT:  ttng.arrive_barrier {{.*}}parentId = 1
// DISABLED:       ttng.wait_barrier {{.*}}dstTask = 1
// DISABLED-NEXT:  ttng.tmem_load
tt.func @keep_foreign_release(
    %smem: !ttg.memdesc<64xf32, #shared1d, #smem, mutable>,
    %tmem: !ttg.memdesc<128x64xf32, #tmem_n64, #ttng.tensor_memory, mutable, 128x128>,
    %local_full: !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
    %local_empty: !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
    %foreign_empty: !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
    %tmem_full: !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
    %tmem_empty: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
  %phase = arith.constant 0 : i32
  ttng.wait_barrier %local_full, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, parentId = 1 : i32, minRegionId = 1 : i32, maxRegionId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %local = ttg.local_load %smem : !ttg.memdesc<64xf32, #shared1d, #smem, mutable> -> tensor<64xf32, #slice64>
  ttng.arrive_barrier %foreign_empty, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, parentId = 2 : i32, minRegionId = 2 : i32, maxRegionId = 2 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  ttng.arrive_barrier %local_empty, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, parentId = 1 : i32, minRegionId = 2 : i32, maxRegionId = 2 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %local_row = tt.expand_dims %local {axis = 0 : i32} : tensor<64xf32, #slice64> -> tensor<1x64xf32, #linear64>
  %local_tile = tt.broadcast %local_row : tensor<1x64xf32, #linear64> -> tensor<128x64xf32, #linear64>
  ttng.wait_barrier %tmem_full, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, parentId = 1 : i32, minRegionId = 1 : i32, maxRegionId = 1 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %tmem_value = ttng.tmem_load %tmem : !ttg.memdesc<128x64xf32, #tmem_n64, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #linear64>
  ttng.arrive_barrier %tmem_empty, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, parentId = 1 : i32, minRegionId = 2 : i32, maxRegionId = 2 : i32}}} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
  %sum = arith.addf %tmem_value, %local_tile : tensor<128x64xf32, #linear64>
  "use"(%sum) : (tensor<128x64xf32, #linear64>) -> ()
  tt.return
}

}
