// RUN: triton-opt --split-input-file -proton-mpp-store-barrier-info %s | FileCheck %s

// Test 1: Basic barrier record resolution - simple wait_barrier
// The ReadCounterOp should be replaced with allocOpId (start) and index (end)

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @test_simple_wait_barrier_resolution
  tt.func @test_simple_wait_barrier_resolution() {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true

    %barriers = ttg.local_alloc {mpp.op.id = 100 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %barrier = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>

    // CHECK: %[[ALLOC_ID:.*]] = arith.constant 100 : i32
    // CHECK-NEXT: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID]] {scopeId = 0 : i32}
    %start_counter = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %segment, %start_counter {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    ttng.wait_barrier %barrier, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    // CHECK: proton_gpu.circular_store end %{{.*}}, %c0_i32{{.*}} {scopeId = 0 : i32}
    %end_counter = proton_gpu.read_counter : i32
    proton_gpu.circular_store end %segment, %end_counter {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    gpu.barrier
    proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
    tt.return
  }
}

// -----

// Test 2: Dynamic index from loop

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @test_dynamic_index_from_loop
  tt.func @test_dynamic_index_from_loop() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true

    %barriers = ttg.local_alloc {mpp.op.id = 200 : i64} : () -> !ttg.memdesc<4xi64, #shared, #smem, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>

    // CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} : i32
    scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
      %barrier = ttg.memdesc_index %barriers[%i] : !ttg.memdesc<4xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

      // CHECK: %[[ALLOC_ID:.*]] = arith.constant 200 : i32
      // CHECK-NEXT: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID]] {scopeId = 1 : i32}
      %start_counter = proton_gpu.read_counter : i32
      proton_gpu.circular_store start %segment, %start_counter {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

        ttng.wait_barrier %barrier, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

        // CHECK: proton_gpu.circular_store end %{{.*}}, %[[IV]] {scopeId = 1 : i32}
        %end_counter = proton_gpu.read_counter : i32
        proton_gpu.circular_store end %segment, %end_counter {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
      }

      gpu.barrier
      proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
      tt.return
    }
}

// -----

// Test 3: TMA copy operation with barrier

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: @test_tma_copy_barrier_resolution
  tt.func @test_tma_copy_barrier_resolution(%a_desc: !tt.tensordesc<tensor<64x32xbf16, #shared>>) {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true

    %data_smem = ttg.local_alloc : () -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable>
    %barriers = ttg.local_alloc {mpp.op.id = 300 : i64} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    ttng.init_barrier %barriers, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %barriers, 4096, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared1, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared1, #smem, mutable> -> <1024, #smem, warp>

    // CHECK: %[[ALLOC_ID:.*]] = arith.constant 300 : i32
    // CHECK: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID]]
    %start_counter = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %segment, %start_counter {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    ttng.async_tma_copy_global_to_local %a_desc[%c0_i32, %c0_i32] %data_smem, %barriers, %true : !tt.tensordesc<tensor<64x32xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable>

    // CHECK: proton_gpu.circular_store end %{{.*}}, %{{.*}}
    %end_counter = proton_gpu.read_counter : i32
    proton_gpu.circular_store end %segment, %end_counter {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    gpu.barrier
    proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
    tt.return
  }
}

// -----

// Test 4: Multiple barriers with different allocOpIds

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @test_multiple_barriers_different_allocs
  tt.func @test_multiple_barriers_different_allocs() {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true

    %barriers_a = ttg.local_alloc {mpp.op.id = 400 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %barriers_b = ttg.local_alloc {mpp.op.id = 401 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %barrier_a = ttg.memdesc_index %barriers_a[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %barrier_b = ttg.memdesc_index %barriers_b[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

    ttng.init_barrier %barrier_a, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier_b, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>

    // CHECK: %[[ALLOC_ID_A:.*]] = arith.constant 400 : i32
    // CHECK: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID_A]] {scopeId = 0 : i32}
    %start_counter_a = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %segment, %start_counter_a {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    ttng.wait_barrier %barrier_a, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %end_counter_a = proton_gpu.read_counter : i32
    proton_gpu.circular_store end %segment, %end_counter_a {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    // CHECK: %[[ALLOC_ID_B:.*]] = arith.constant 401 : i32
    // CHECK: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID_B]] {scopeId = 1 : i32}
    %start_counter_b = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %segment, %start_counter_b {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    ttng.wait_barrier %barrier_b, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %end_counter_b = proton_gpu.read_counter : i32
    proton_gpu.circular_store end %segment, %end_counter_b {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    gpu.barrier
    proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
    tt.return
  }
}

// -----

// Test 5: Index selected via scf.if

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @test_index_via_scf_if
  tt.func @test_index_via_scf_if(%cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true

    %barriers = ttg.local_alloc {mpp.op.id = 800 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>

    // CHECK: %[[SELECTED_INDEX:.*]] = scf.if %{{.*}} -> (i32)
    %selected_index = scf.if %cond -> i32 {
      scf.yield %c0_i32 : i32
    } else {
      scf.yield %c1_i32 : i32
    }

    %barrier = ttg.memdesc_index %barriers[%selected_index] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>

    // CHECK: %[[ALLOC_ID:.*]] = arith.constant 800 : i32
    // CHECK-NEXT: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID]] {scopeId = 0 : i32}
    %start_counter = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %segment, %start_counter {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    ttng.wait_barrier %barrier, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    // CHECK: proton_gpu.circular_store end %{{.*}}, %[[SELECTED_INDEX]] {scopeId = 0 : i32}
    %end_counter = proton_gpu.read_counter : i32
    proton_gpu.circular_store end %segment, %end_counter {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    gpu.barrier
    proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
    tt.return
  }
}

// -----

// Test 6: Loop variable with memdesc_index - barrier yielded through loop

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @test_loop_memdesc_index_barrier
  tt.func @test_loop_memdesc_index_barrier() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true

    %barriers = ttg.local_alloc {mpp.op.id = 900 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>

    %init_barrier = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %barrier_0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %barrier_1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier_0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier_1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>

    // CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[BARRIER_ARG:.*]] = %{{.*}}, %{{.*}} = %{{.*}}) -> (!ttg.memdesc<1xi64,{{.*}}, i32)
    %result = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32
        iter_args(%curr_barrier = %init_barrier)
        -> (!ttg.memdesc<1xi64, #shared, #smem, mutable>) : i32 {

      // CHECK: %[[ALLOC_ID_IN_LOOP:.*]] = arith.constant 900 : i32
      // CHECK: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID_IN_LOOP]]
      %start_counter = proton_gpu.read_counter : i32
      proton_gpu.circular_store start %segment, %start_counter {scopeId = 6 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

      ttng.wait_barrier %curr_barrier, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

      // CHECK: proton_gpu.circular_store end %{{.*}}, %{{.*}}
      %end_counter = proton_gpu.read_counter : i32
      proton_gpu.circular_store end %segment, %end_counter {scopeId = 6 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

      // CHECK: %[[NEXT_IDX:.*]] = arith.remsi %{{.*}}, %{{.*}} : i32
      %next_idx = arith.remsi %i, %c2_i32 : i32
      // CHECK: ttg.memdesc_index %{{.*}}[%[[NEXT_IDX]]]
      %next_barrier = ttg.memdesc_index %barriers[%next_idx] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

      // CHECK: scf.yield %{{.*}}, %[[NEXT_IDX]] : !ttg.memdesc<1xi64,{{.*}}, i32
      scf.yield %next_barrier : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    }

    // CHECK: %[[ALLOC_ID_AFTER:.*]] = arith.constant 900 : i32
    // CHECK: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID_AFTER]]
    %start_after = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %segment, %start_after {scopeId = 7 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    ttng.wait_barrier %result, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    // CHECK: proton_gpu.circular_store end %{{.*}}, %{{.*}}
    %end_after = proton_gpu.read_counter : i32
    proton_gpu.circular_store end %segment, %end_after {scopeId = 7 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

    gpu.barrier
    proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
    tt.return
  }
}

// -----

// Test 7: Nested loops with different barrier arrays

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @test_outer_loop_barrier_in_inner_loop
  tt.func @test_outer_loop_barrier_in_inner_loop() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true

    %outer_barriers = ttg.local_alloc {mpp.op.id = 1800 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %inner_barriers = ttg.local_alloc {mpp.op.id = 1801 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>

    %outer_bar_0 = ttg.memdesc_index %outer_barriers[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %inner_bar_0 = ttg.memdesc_index %inner_barriers[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

    ttng.init_barrier %outer_bar_0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %inner_bar_0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>

    // CHECK: scf.for
    %outer_result = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32
        iter_args(%outer_barrier = %outer_bar_0)
        -> (!ttg.memdesc<1xi64, #shared, #smem, mutable>) : i32 {

      // CHECK: %[[OUTER_ALLOC_ID:.*]] = arith.constant 1800 : i32
      // CHECK: proton_gpu.circular_store start %{{.*}}, %[[OUTER_ALLOC_ID]] {scopeId = 23 : i32}
      %outer_start = proton_gpu.read_counter : i32
      proton_gpu.circular_store start %segment, %outer_start {scopeId = 23 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

      ttng.wait_barrier %outer_barrier, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

      // CHECK: proton_gpu.circular_store end %{{.*}}, %{{.*}} {scopeId = 23 : i32}
      %outer_end = proton_gpu.read_counter : i32
      proton_gpu.circular_store end %segment, %outer_end {scopeId = 23 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

      // CHECK: scf.for
      %inner_result = scf.for %j = %c0_i32 to %c2_i32 step %c1_i32
          iter_args(%inner_barrier = %inner_bar_0)
          -> (!ttg.memdesc<1xi64, #shared, #smem, mutable>) : i32 {

        // CHECK: %[[INNER_ALLOC_ID:.*]] = arith.constant 1801 : i32
        // CHECK: proton_gpu.circular_store start %{{.*}}, %[[INNER_ALLOC_ID]] {scopeId = 24 : i32}
        %inner_start = proton_gpu.read_counter : i32
        proton_gpu.circular_store start %segment, %inner_start {scopeId = 24 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

        ttng.wait_barrier %inner_barrier, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

        // CHECK: proton_gpu.circular_store end %{{.*}}, %{{.*}} {scopeId = 24 : i32}
        %inner_end = proton_gpu.read_counter : i32
        proton_gpu.circular_store end %segment, %inner_end {scopeId = 24 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32

        %next_j_phase = arith.xori %j, %c1_i32 : i32
        %next_inner_barrier = ttg.memdesc_index %inner_barriers[%next_j_phase] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        scf.yield %next_inner_barrier : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      }

      %next_i_phase = arith.xori %i, %c1_i32 : i32
      %next_outer_barrier = ttg.memdesc_index %outer_barriers[%next_i_phase] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      scf.yield %next_outer_barrier : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    }

    gpu.barrier
    proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
    tt.return
  }
}

// -----

// Test 8: CF dialect control flow pattern (lowered from scf.if)

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @test_cf_branch_control_flow
  tt.func @test_cf_branch_control_flow() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cond = arith.constant true

    %barriers = ttg.local_alloc {mpp.op.id = 61 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>

    %barrier_0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %barrier_1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier_0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier_1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>

    cf.br ^bb2(%barrier_1, %c0_i32 : !ttg.memdesc<1xi64, #shared, #smem, mutable>, i32)

  ^bb2(%block_barrier: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %phase: i32):
    cf.cond_br %cond, ^bb3, ^bb_exit

  ^bb3:
    cf.cond_br %cond, ^bb4, ^bb5

  ^bb4:
    %start = proton_gpu.read_counter : i32
    // CHECK: %[[ALLOC_ID:.*]] = arith.constant 61 : i32
    // CHECK: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID]] {scopeId = 23 : i32}
    proton_gpu.circular_store start %segment, %start {scopeId = 23 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
    cf.br ^bb5

  ^bb5:
    ttng.wait_barrier %block_barrier, %phase, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    cf.cond_br %cond, ^bb6, ^bb7

  ^bb6:
    %end = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %{{.*}}, %{{.*}} {scopeId = 23 : i32}
    proton_gpu.circular_store end %segment, %end {scopeId = 23 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
    cf.br ^bb7

  ^bb7:
    cf.br ^bb_exit

  ^bb_exit:
    gpu.barrier
    proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
    tt.return
  }
}

// -----

// Test 9: Multi-barrier tc_gen5_mma with nested circular_store patterns

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @test_tc_gen5_mma_multi_barrier_nested_stores
  tt.func @test_tc_gen5_mma_multi_barrier_nested_stores() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cond = arith.constant true

    %barrier_array_59 = ttg.local_alloc {mpp.op.id = 59 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %barrier_array_84 = ttg.local_alloc {mpp.op.id = 84 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>

    %barrier_59_0 = ttg.memdesc_index %barrier_array_59[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %barrier_84_0 = ttg.memdesc_index %barrier_array_84[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier_59_0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier_84_0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %a_smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
    %b_smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared3, #smem, mutable>
    %acc_tmem = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>

    cf.br ^bb20

  ^bb20:
    // CHECK: %[[ALLOC_59:.*]] = arith.constant 59 : i32
    // CHECK-NEXT: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_59]] {scopeId = 21 : i32}
    %start_21 = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %segment, %start_21 {scopeId = 21 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
    cf.br ^bb21

  ^bb21:
    cf.cond_br %cond, ^bb22, ^bb23

  ^bb22:
    // CHECK: %[[ALLOC_84:.*]] = arith.constant 84 : i32
    // CHECK: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_84]] {scopeId = 22 : i32}
    %start_22 = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %segment, %start_22 {scopeId = 22 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
    cf.br ^bb23

  ^bb23:
    ttng.tc_gen5_mma %a_smem, %b_smem, %acc_tmem, %false, %true, %barrier_59_0[%true], %barrier_84_0[%true] {is_async, mpp.op.id = 302 : i64} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
    cf.cond_br %cond, ^bb24, ^bb25

  ^bb24:
    // CHECK: proton_gpu.circular_store end %{{.*}}, %c0_i32{{.*}} {scopeId = 22 : i32}
    %end_22 = proton_gpu.read_counter : i32
    proton_gpu.circular_store end %segment, %end_22 {scopeId = 22 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
    cf.br ^bb25

  ^bb25:
    cf.cond_br %cond, ^bb26, ^bb27

  ^bb26:
    // CHECK: proton_gpu.circular_store end %{{.*}}, %c0_i32{{.*}} {scopeId = 21 : i32}
    %end_21 = proton_gpu.read_counter : i32
    proton_gpu.circular_store end %segment, %end_21 {scopeId = 21 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
    cf.br ^bb27

  ^bb27:
    gpu.barrier
    proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
    tt.return
  }
}

// -----

// Test 10: HSTU pattern - barrier from loop arg with SEPARATE phase counter

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @test_barrier_loop_arg_separate_phase_counter
  tt.func @test_barrier_loop_arg_separate_phase_counter() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %cond = arith.constant true

    %acc_36 = ttg.local_alloc {mpp.op.id = 61 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %acc_44 = ttg.local_alloc {mpp.op.id = 74 : i64} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>

    %acc_37 = ttg.memdesc_index %acc_36[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_38 = ttg.memdesc_index %acc_36[%c1_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_37, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_38, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %acc_45 = ttg.memdesc_index %acc_44[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %acc_46 = ttg.memdesc_index %acc_44[%c1_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_45, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %acc_46, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %scratch = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %scratch : !tt.ptr<i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %segment = proton_gpu.segment_alloc %buf : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>

    // CHECK: scf.for
    %result:4 = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32
        iter_args(%acc_98 = %acc_38, %arg33 = %c0_i32,
                  %acc_134_barrier = %acc_45, %acc_133 = %c0_i32)
        -> (!ttg.memdesc<1xi64, #shared, #smem, mutable>, i32,
            !ttg.memdesc<1xi64, #shared, #smem, mutable>, i32) : i32 {

      scf.if %cond {
        %start_142 = proton_gpu.read_counter : i32
        // CHECK: %[[ALLOC_ID_142:.*]] = arith.constant 61 : i32
        // CHECK: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID_142]] {scopeId = 33 : i32}
        proton_gpu.circular_store start %segment, %start_142 {scopeId = 33 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
      }

      ttng.wait_barrier %acc_98, %arg33, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

      scf.if %cond {
        %end_142 = proton_gpu.read_counter : i32
        // CHECK: proton_gpu.circular_store end %{{.*}}, %{{.*}} {scopeId = 33 : i32}
        proton_gpu.circular_store end %segment, %end_142 {scopeId = 33 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
      }

      %acc_132 = arith.xori %acc_133, %c1_i32 : i32
      %acc_134 = ttg.memdesc_index %acc_44[%acc_132] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

      scf.if %cond {
        %start_165 = proton_gpu.read_counter : i32
        // CHECK: %[[ALLOC_ID_165:.*]] = arith.constant 74 : i32
        // CHECK: proton_gpu.circular_store start %{{.*}}, %[[ALLOC_ID_165]] {scopeId = 34 : i32}
        proton_gpu.circular_store start %segment, %start_165 {scopeId = 34 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
      }

      ttng.wait_barrier %acc_134, %acc_133, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

      scf.if %cond {
        %end_165 = proton_gpu.read_counter : i32
        // CHECK: proton_gpu.circular_store end %{{.*}}, %{{.*}} {scopeId = 34 : i32}
        proton_gpu.circular_store end %segment, %end_165 {scopeId = 34 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
      }

      %next_phase = arith.xori %arg33, %c1_i32 : i32
      %next_acc_98 = ttg.memdesc_index %acc_36[%next_phase] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

      scf.yield %next_acc_98, %next_phase, %acc_134, %acc_132 :
        !ttg.memdesc<1xi64, #shared, #smem, mutable>, i32,
        !ttg.memdesc<1xi64, #shared, #smem, mutable>, i32
    }

    gpu.barrier
    proton_gpu.finalize %segment, %scratch : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
    tt.return
  }
}
