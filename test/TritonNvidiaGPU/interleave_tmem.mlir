// RUN: triton-opt %s --triton-nvidia-interleave-tmem --allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#linear64 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#linear128 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {

tt.func public @sink_load(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                          %arg1: tensor<128x128xf16, #blocked>,
                          %arg2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>)
                          -> (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x128xf16, #blocked>) {

  // CHECK: ttg.local_alloc
  // CHECK: ttng.tmem_load
  // CHECK: ttg.convert_layout
  // CHECK: arith.truncf
  %subslice0 = ttng.tmem_subslice %arg0 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %subtile0 = ttng.tmem_load %subslice0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear64>
  %outLHS = ttg.convert_layout %subtile0 : tensor<128x64xf32, #linear64> -> tensor<128x64xf32, #blocked>
  %subslice1 = ttng.tmem_subslice %arg0 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %subtile1 = ttng.tmem_load %subslice1 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear64>
  %outRHS = ttg.convert_layout %subtile1 : tensor<128x64xf32, #linear64> -> tensor<128x64xf32, #blocked>

  // CHECK: ttng.tmem_load
  // CHECK: ttg.convert_layout
  // CHECK: ttng.tmem_store
  // CHECK: arith.truncf
  %4 = ttg.local_alloc %arg1 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
  %5 = arith.truncf %outLHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

  %true = arith.constant true
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear128>
  ttng.tmem_store %cst, %arg2, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %6 = arith.truncf %outRHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

  // CHECK: ttng.tmem_load
  // CHECK: ttg.convert_layout
  // CHECK: "unknow_may_side_effect"() : () -> ()
  // CHECK: arith.truncf
  %7 = ttng.tmem_load %arg2 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  %8 = ttg.convert_layout %7 : tensor<128x128xf32, #linear128> -> tensor<128x128xf32, #blocked>
  "unknow_may_side_effect"() : () -> ()
  %9 = arith.truncf %8 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>

  ttg.local_dealloc %4 : !ttg.memdesc<128x128xf16, #shared, #smem>
  tt.return %5, %6, %9 : tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x128xf16, #blocked>
}

// CHECK-LABEL: @interleave_load_store_ws
tt.func @interleave_load_store_ws() {
  %0 = ttng.tmem_alloc : () -> (!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>)
  ttg.warp_specialize(%0)
  default{
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0(%arg0: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(8) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %alpha = arith.constant dense<0.5> : tensor<128x64xf32, #linear64>
    %true = arith.constant true

    // CHECK: scf.for
    scf.for %i = %c0 to %c32 step %c1 : i32 {
      // CHECK: memdesc_index
      %cur_acc = ttg.memdesc_index %arg0[%i] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: [[S0:%.+]] = ttng.tmem_subslice %{{.+}} {N = 0 : i32}
      // CHECK-NEXT: [[S1:%.+]] = ttng.tmem_subslice %{{.+}} {N = 64 : i32}

      // CHECK-NEXT: [[L0:%.+]] = ttng.tmem_load [[S0]]
      // CHECK-NEXT: [[M0:%.+]] = arith.mulf [[L0]]
      // CHECK-NEXT: ttng.tmem_store [[M0]], [[S0]]
      %slice0 = ttng.tmem_subslice %cur_acc {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %val0 = ttng.tmem_load %slice0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear64>
      %mul0 = arith.mulf %val0, %alpha : tensor<128x64xf32, #linear64>

      // CHECK-NEXT: [[L1:%.+]] = ttng.tmem_load [[S1]]
      // CHECK-NEXT: [[M1:%.+]] = arith.mulf [[L1]]
      // CHECK-NEXT: ttng.tmem_store [[M1]], [[S1]]
      %slice1 = ttng.tmem_subslice %cur_acc {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %val1 = ttng.tmem_load %slice1 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear64>
      %mul1 = arith.mulf %val1, %alpha : tensor<128x64xf32, #linear64>

      ttng.tmem_store %mul0, %slice0, %true : tensor<128x64xf32, #linear64> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %mul1, %slice1, %true : tensor<128x64xf32, #linear64> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    }
    ttg.warp_return
  } : (!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
  tt.return
}

// CHECK-LABEL: @arrive_barrier
tt.func @arrive_barrier(%arg0: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %cst = arith.constant dense<0.0> : tensor<128x128xf32, #linear128>

  // CHECK-COUNT-2: ttng.tmem_alloc
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %noalias_alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: tmem_store
  // CHECK-NEXT: tmem_load
  %0 = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.tmem_store %cst, %noalias_alloc, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: arrive_barrier
  ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  "user"(%0) : (tensor<128x128xf32, #linear128>) -> ()
  tt.return
}

// CHECK-LABEL: @arrive_restore_after_operand_defs
tt.func @arrive_restore_after_operand_defs(
    %arg0: !ttg.memdesc<1x1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %cst = arith.constant dense<0.0> : tensor<128x128xf32, #linear128>
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  // CHECK: ttng.tmem_store
  ttng.tmem_store %cst, %alloc, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[BAR:%.+]] = ttg.memdesc_index
  %bar = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x1xi64, #barrier_shared, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %use0 = arith.addi %c0, %c0 : i32
  // CHECK-NEXT: ttng.arrive_barrier [[BAR]], 1
  ttng.arrive_barrier %bar, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  // CHECK-NEXT: arith.addi
  %use1 = arith.addi %use0, %c0 : i32
  "user"(%unused, %use1) : (tensor<128x128xf32, #linear128>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @sink_alloc_op
tt.func @sink_alloc_op(%arg0: tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %true = arith.constant true

  %alloc0 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %subview0 = ttg.memdesc_index %alloc0[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK: [[ALLOC1:%.+]] = ttng.tmem_alloc
  %alloc1 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK: [[SUBVIEW1:%.+]] = ttg.memdesc_index [[ALLOC1]]
  %subview1 = ttg.memdesc_index %alloc1[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: tmem_store %arg0, [[SUBVIEW1]]
  ttng.tmem_store %arg0, %subview1, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[ALLOC0:%.+]] = ttng.tmem_alloc
  // CHECK: [[SUBVIEW0:%.+]] = ttg.memdesc_index [[ALLOC0]]
  // CHECK-NEXT: tmem_store %arg0, [[SUBVIEW0]]
  ttng.tmem_store %arg0, %subview0, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}

// An arrive with channelGraph disjoint from a wait's channelGraph should be
// sunk past the wait.
// CHECK-LABEL: @sink_arrive_past_wait_disjoint
tt.func @sink_arrive_past_wait_disjoint(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %phase: i32) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  // CHECK: ttng.wait_barrier {{.*}}channelGraph = array<i32: 2>
  // CHECK: ttng.wait_barrier {{.*}}channelGraph = array<i32: 1>
  // CHECK: ttng.arrive_barrier {{.*}}channelGraph = array<i32: 2>
  // CHECK: ttng.arrive_barrier {{.*}}channelGraph = array<i32: 1>
  ttng.wait_barrier %bar1, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.arrive_barrier %bar2, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// An arrive whose channelGraph overlaps the wait's channelGraph must NOT be
// sunk past the wait.
// CHECK-LABEL: @no_reorder_overlapping_graph
tt.func @no_reorder_overlapping_graph(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %phase: i32) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  // CHECK: ttng.arrive_barrier
  // CHECK-SAME: channelGraph = array<i32: 1, 2>
  // CHECK-NEXT: ttng.wait_barrier
  // CHECK-SAME: channelGraph = array<i32: 2, 3>
  ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// Barriers without constraints are not moved.
// CHECK-LABEL: @no_reorder_without_constraints
tt.func @no_reorder_without_constraints(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %phase: i32) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  // CHECK: ttng.arrive_barrier
  // CHECK-NEXT: ttng.wait_barrier
  ttng.arrive_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.wait_barrier %bar2, %phase : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// WS barriers are not reordered in a parent block without a direct tmem_load,
// even if a nested region contains one.
// CHECK-LABEL: @no_reorder_without_tmem_load_in_parent_block
tt.func @no_reorder_without_tmem_load_in_parent_block(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %phase: i32) {
  // CHECK: ttng.arrive_barrier
  // CHECK-SAME: channelGraph = array<i32: 2>
  // CHECK-NEXT: ttng.wait_barrier
  // CHECK-SAME: channelGraph = array<i32: 1>
  ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %c1 step %c1 : i32 {
    %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  }
  tt.return
}

// WS arrives cannot sink past a non-WS arrive barrier.
// CHECK-LABEL: @sink_arrive_stops_at_non_ws_arrive
tt.func @sink_arrive_stops_at_non_ws_arrive(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  // CHECK: ttng.arrive_barrier
  // CHECK-SAME: WSBarrier
  // CHECK-NEXT: ttng.arrive_barrier
  // CHECK-SAME: loweringMask
  ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.arrive_barrier %bar2, 1 {constraints = {loweringMask = array<i32: 0, 1>}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// WS waits cannot rise past a non-WS wait barrier.
// CHECK-LABEL: @raise_wait_stops_at_non_ws_wait
tt.func @raise_wait_stops_at_non_ws_wait(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %phase: i32) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  // CHECK: ttng.wait_barrier
  // CHECK-SAME: loweringMask
  // CHECK-NEXT: ttng.wait_barrier
  // CHECK-SAME: WSBarrier
  ttng.wait_barrier %bar1, %phase {constraints = {loweringMask = array<i32: 1, 0>}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// WS barriers cannot move past non-barrier ops with arrive-like semantics.
// CHECK-LABEL: @no_reorder_across_arrive_like_op
tt.func @no_reorder_across_arrive_like_op(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %phase: i32) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  // CHECK: ttng.arrive_barrier
  // CHECK-SAME: channelGraph = array<i32: 2>
  // CHECK-NEXT: ttng.async_tma_store_wait
  // CHECK-NEXT: ttng.wait_barrier
  // CHECK-SAME: channelGraph = array<i32: 1>
  ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.async_tma_store_wait {pendings = 0 : i32}
  ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// WS barriers cannot move past tcgen05 commits.
// CHECK-LABEL: @no_reorder_across_tcgen5_commit
tt.func @no_reorder_across_tcgen5_commit(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %phase: i32) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  // CHECK: ttng.arrive_barrier
  // CHECK-SAME: channelGraph = array<i32: 2>
  // CHECK-NEXT: ttng.tc_gen5_commit
  // CHECK-NEXT: ttng.wait_barrier
  // CHECK-SAME: channelGraph = array<i32: 1>
  ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.tc_gen5_commit %bar1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// WS barriers cannot move past control-flow ops.
// CHECK-LABEL: @no_reorder_across_control_flow
tt.func @no_reorder_across_control_flow(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %phase: i32) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %unused = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  // CHECK: ttng.arrive_barrier
  // CHECK-SAME: channelGraph = array<i32: 2>
  // CHECK-NEXT: scf.for
  // CHECK: ttng.wait_barrier
  // CHECK-SAME: channelGraph = array<i32: 1>
  ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %c1 step %c1 : i32 {
  }
  ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// After barrier reordering, tmem_load can sink past the wait that was
// previously blocked by an arrive from a different channel.
// CHECK-LABEL: @tmem_load_sinks_after_barrier_reorder
tt.func @tmem_load_sinks_after_barrier_reorder(
    %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %phase: i32) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // tmem_load is followed by its own arrive (channel 2), then a wait from
  // channel 1. The arrive should sink past the wait, letting the tmem_load
  // sink further.
  //
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: tmem_load
  // CHECK-NEXT: ttng.arrive_barrier
  // CHECK-SAME: channelGraph = array<i32: 2>
  // CHECK-NEXT: ttng.wait_barrier
  // CHECK-SAME: channelGraph = array<i32: 1>
  // CHECK-NEXT: "user"
  %0 = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 1>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  "user"(%0) : (tensor<128x128xf32, #linear128>) -> ()
  tt.return
}

// All split tmem_loads should inherit the channelGraph from their arrive
// barrier and sink past store-channel barriers independently.
// CHECK-LABEL: @split_tmem_loads_all_sink
tt.func @split_tmem_loads_all_sink(
    %tmem_wait_bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %store_bar0: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %store_bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %smem_buf: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
    %phase: i32) {
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %s0 = ttng.tmem_subslice %alloc {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %s1 = ttng.tmem_subslice %alloc {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

  // tmem_load wait (no constraints — from MMA channel)
  ttng.wait_barrier %tmem_wait_bar, %phase : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>

  // Two split tmem_loads
  %v0 = ttng.tmem_load %s0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear64>
  %v1 = ttng.tmem_load %s1 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear64>

  // tmem_load arrive (channelGraph disjoint from store channel)
  ttng.arrive_barrier %tmem_wait_bar, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>

  // Store channel: wait → local_store → arrive, repeated for each subtile
  ttng.wait_barrier %store_bar0, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %t0 = arith.truncf %v0 : tensor<128x64xf32, #linear64> to tensor<128x64xf16, #linear64>
  ttg.local_store %t0, %smem_buf : tensor<128x64xf16, #linear64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  ttng.arrive_barrier %store_bar0, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>

  ttng.wait_barrier %store_bar1, %phase {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %t1 = arith.truncf %v1 : tensor<128x64xf32, #linear64> to tensor<128x64xf16, #linear64>
  ttg.local_store %t1, %smem_buf : tensor<128x64xf16, #linear64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  ttng.arrive_barrier %store_bar1, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 2>}}} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>

  // Expected: both tmem_loads sink past the store waits, interleaved with
  // the store pipeline.
  //
  // CHECK:      ttng.wait_barrier %{{.*}}, %{{.*}} :
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: arith.truncf
  // CHECK-NEXT: ttng.wait_barrier {{.*}}channelGraph = array<i32: 2>
  // CHECK-NEXT: ttg.local_store
  // CHECK-NEXT: ttng.arrive_barrier {{.*}}channelGraph = array<i32: 2>
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: ttng.arrive_barrier {{.*}}channelGraph = array<i32: 1, 3>
  // CHECK-NEXT: arith.truncf
  // CHECK-NEXT: ttng.wait_barrier {{.*}}channelGraph = array<i32: 2>
  // CHECK-NEXT: ttg.local_store
  // CHECK-NEXT: ttng.arrive_barrier {{.*}}channelGraph = array<i32: 2>
  tt.return
}

}
