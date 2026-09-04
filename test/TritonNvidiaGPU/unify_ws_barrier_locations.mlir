// RUN: triton-opt %s --triton-nvidia-unify-ws-barrier-locations --allow-unregistered-dialect | FileCheck %s
// RUN: env TRITON_DISABLE_WSBARRIER_REORDER=1 triton-opt %s --triton-nvidia-unify-ws-barrier-locations --allow-unregistered-dialect | FileCheck %s --check-prefix=DISABLED

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @unify_cast_broadcast
// CHECK:       ttng.wait_barrier %{{.*}}, %{{.*}}, %{{.*}} {{.*}}dstTask = 3
// CHECK-NEXT:  ttng.wait_barrier %{{.*}}, %{{.*}} {{.*}}dstTask = 1
// CHECK-NEXT:  %[[BIAS:.*]] = ttg.local_load
// CHECK-NEXT:  ttng.arrive_barrier
// CHECK-NEXT:  %[[EXT:.*]] = arith.extf %[[BIAS]]
// CHECK-NEXT:  %[[CVT:.*]] = ttg.convert_layout %[[EXT]]
// CHECK-NEXT:  %[[BCAST:.*]] = tt.broadcast %[[CVT]]
// CHECK:       %[[ACC:.*]] = ttng.tmem_load
// CHECK-NEXT:  ttng.arrive_barrier
// CHECK-NEXT:  arith.addf %[[ACC]], %[[BCAST]]
// DISABLED-LABEL: @unify_cast_broadcast
// DISABLED:       ttng.wait_barrier
// DISABLED-NEXT:  ttg.local_load
// DISABLED:       tt.broadcast
// DISABLED-NEXT:  ttng.wait_barrier
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

}
