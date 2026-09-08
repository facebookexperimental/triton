// RUN: split-file %s %t
// RUN: triton-opt %t/insert_semas.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas.mlir
// RUN: triton-opt %t/insert_semas_async_entry_fanin.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_async_entry_fanin.mlir
// RUN: triton-opt %t/insert_semas_async_entry_fanin.mlir -allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore -cse -o /dev/null
// RUN: triton-opt %t/insert_semas_branch_local_init.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_branch_local_init.mlir
// RUN: triton-opt %t/insert_semas_cached_exact_reuse.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_cached_exact_reuse.mlir --check-prefix=IR
// RUN: env NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt %t/insert_semas_cached_exact_reuse.mlir -allow-unregistered-dialect --nvws-insert-semas -cse 2>&1 | FileCheck %t/insert_semas_cached_exact_reuse.mlir --check-prefix=DAG
// RUN: triton-opt %t/insert_semas_circular_backing_dominance.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_circular_backing_dominance.mlir
// RUN: triton-opt %t/insert_semas_circular_smem.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_circular_smem.mlir
// RUN: triton-opt %t/insert_semas_circular_smem.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_circular_smem.mlir --check-prefix=COUNT
// RUN: triton-opt %t/insert_semas_conditional_multi_result.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_conditional_multi_result.mlir
// RUN: triton-opt %t/insert_semas_descriptor_store_completion.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_descriptor_store_completion.mlir --check-prefix=SEMA
// RUN: triton-opt %t/insert_semas_descriptor_store_completion.mlir -split-input-file -allow-unregistered-dialect --verify-each=false --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore --triton-nvidia-tma-lowering -cse | FileCheck %t/insert_semas_descriptor_store_completion.mlir --check-prefix=LOWER
// RUN: triton-opt %t/insert_semas_direct_builder_composition.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_direct_builder_composition.mlir
// RUN: triton-opt %t/insert_semas_function_cfg.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_function_cfg.mlir
// RUN: not triton-opt %t/insert_semas_function_cfg_errors.mlir -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %t/insert_semas_function_cfg_errors.mlir
// RUN: triton-opt %t/insert_semas_fused_alias_handoff.mlir -allow-unregistered-dialect --nvws-insert-semas | FileCheck %t/insert_semas_fused_alias_handoff.mlir --check-prefix=SEMA
// RUN: triton-opt %t/insert_semas_fused_alias_handoff.mlir -allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase -cse | FileCheck %t/insert_semas_fused_alias_handoff.mlir --check-prefix=ASP
// RUN: triton-opt %t/insert_semas_if_encloser_inner_loop.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_if_encloser_inner_loop.mlir
// RUN: triton-opt %t/insert_semas_if_split_metadata.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_if_split_metadata.mlir
// RUN: triton-opt %t/insert_semas_live_tag_source.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_live_tag_source.mlir
// RUN: triton-opt %t/insert_semas_local_buffer_reuse.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_local_buffer_reuse.mlir
// RUN: triton-opt %t/insert_semas_local_cfg.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_local_cfg.mlir
// RUN: not triton-opt %t/insert_semas_local_errors.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %t/insert_semas_local_errors.mlir
// RUN: not triton-opt %t/insert_semas_local_errors.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas=num-stages=33 2>&1 | FileCheck %t/insert_semas_local_errors.mlir --check-prefix=NUM-STAGES
// RUN: triton-opt %t/insert_semas_local_mixed_copy_reuse.mlir -allow-unregistered-dialect --nvws-insert-semas | FileCheck %t/insert_semas_local_mixed_copy_reuse.mlir
// RUN: triton-opt %t/insert_semas_local_no_buffer_id.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_local_no_buffer_id.mlir
// RUN: triton-opt %t/insert_semas_local_read_lifetime.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_local_read_lifetime.mlir
// RUN: triton-opt %t/insert_semas_memdesc_subslice.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_memdesc_subslice.mlir
// RUN: triton-opt %t/insert_semas_memdesc_trans_alloc_shape.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_memdesc_trans_alloc_shape.mlir
// RUN: triton-opt %t/insert_semas_meta_fa_fwd.mlir -allow-unregistered-dialect --nvws-insert-semas | FileCheck %t/insert_semas_meta_fa_fwd.mlir
// RUN: triton-opt %t/insert_semas_meta_fa_fwd.mlir -allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase -cse | FileCheck %t/insert_semas_meta_fa_fwd.mlir --check-prefix=ASP
// RUN: not triton-opt %t/insert_semas_mixed_copy_error.mlir -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %t/insert_semas_mixed_copy_error.mlir
// RUN: triton-opt %t/insert_semas_mixed_overlap_members.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_mixed_overlap_members.mlir
// RUN: not triton-opt %t/insert_semas_multi_component_error.mlir -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %t/insert_semas_multi_component_error.mlir
// RUN: triton-opt %t/insert_semas_nested_carrier.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_nested_carrier.mlir
// RUN: triton-opt %t/insert_semas_nested_region_access.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_nested_region_access.mlir
// RUN: triton-opt %t/insert_semas_nested_ws_inner_loop.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_nested_ws_inner_loop.mlir
// RUN: triton-opt %t/insert_semas_per_edge_tmem.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_per_edge_tmem.mlir
// RUN: triton-opt %t/insert_semas_post_ws_read_tag.mlir -allow-unregistered-dialect --nvws-insert-semas | FileCheck %t/insert_semas_post_ws_read_tag.mlir
// RUN: triton-opt %t/insert_semas_post_ws_read_tag.mlir -allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore | FileCheck %t/insert_semas_post_ws_read_tag.mlir --check-prefix=LOWER
// RUN: triton-opt %t/insert_semas_post_ws_read_tag.mlir -allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore --tritongpu-partition-loops | FileCheck %t/insert_semas_post_ws_read_tag.mlir --check-prefix=PARTITION
// RUN: triton-opt %t/insert_semas_raw_if_token.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_raw_if_token.mlir
// RUN: triton-opt %t/insert_semas_recurrence_owner_cycle.mlir -allow-unregistered-dialect --nvws-insert-semas | FileCheck %t/insert_semas_recurrence_owner_cycle.mlir
// RUN: triton-opt %t/insert_semas_recurrence_owner_cycle.mlir -allow-unregistered-dialect --nvws-insert-semas=num-stages=4 --nvws-semaphore-optimize=num-stages=4 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=4 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=4 --tritongpu-pipeline=num-stages=4 | FileCheck %t/insert_semas_recurrence_owner_cycle.mlir --check-prefix=PIPE
// RUN: triton-opt %t/insert_semas_recurrence_schedule.mlir -allow-unregistered-dialect --nvws-insert-semas | FileCheck %t/insert_semas_recurrence_schedule.mlir
// RUN: triton-opt %t/insert_semas_recurrence_schedule.mlir -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=2 --tritongpu-pipeline=num-stages=2 | FileCheck %t/insert_semas_recurrence_schedule.mlir --check-prefix=PIPE
// RUN: not triton-opt %t/insert_semas_recurrence_schedule_errors.mlir -allow-unregistered-dialect --nvws-insert-semas 2>&1 | FileCheck %t/insert_semas_recurrence_schedule_errors.mlir
// RUN: triton-opt %t/insert_semas_region_drain.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_region_drain.mlir
// RUN: triton-opt %t/insert_semas_region_drain_continuation.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_region_drain_continuation.mlir
// RUN: triton-opt %t/insert_semas_release_count.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_release_count.mlir --check-prefix=EMIT
// RUN: triton-opt %t/insert_semas_release_count.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore -cse | FileCheck %t/insert_semas_release_count.mlir --check-prefix=LOWER
// RUN: triton-opt %t/insert_semas_root_entry_tmem.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_root_entry_tmem.mlir
// RUN: triton-opt %t/insert_semas_same_owner_mixed_completion.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_same_owner_mixed_completion.mlir
// RUN: triton-opt %t/insert_semas_same_owner_mixed_completion.mlir -allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore -cse | FileCheck %t/insert_semas_same_owner_mixed_completion.mlir --check-prefix=LOWER --implicit-check-not=nvws.descriptor_load
// RUN: triton-opt %t/insert_semas_scheduled_region_slots.mlir -allow-unregistered-dialect --nvws-insert-semas | FileCheck %t/insert_semas_scheduled_region_slots.mlir --check-prefix=SEMA
// RUN: triton-opt %t/insert_semas_scheduled_region_slots.mlir -allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase -cse | FileCheck %t/insert_semas_scheduled_region_slots.mlir --check-prefix=ASP
// RUN: triton-opt %t/insert_semas_sequential_ws_loops.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_sequential_ws_loops.mlir
// RUN: triton-opt %t/insert_semas_slack_zero_delay_schedule.mlir -allow-unregistered-dialect --nvws-insert-semas | FileCheck %t/insert_semas_slack_zero_delay_schedule.mlir
// RUN: triton-opt %t/insert_semas_staged_pou.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_staged_pou.mlir
// RUN: triton-opt %t/insert_semas_staged_pou.mlir -allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=2 --tritongpu-pipeline=num-stages=2 -o /dev/null
// RUN: triton-opt %t/insert_semas_tail_schedule.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_tail_schedule.mlir
// RUN: triton-opt %t/insert_semas_tmem_alias.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_tmem_alias.mlir
// RUN: triton-opt %t/insert_semas_tmem_alias.mlir -allow-unregistered-dialect --nvws-insert-semas="use-meta-partitioner=true" -cse | FileCheck %t/insert_semas_tmem_alias.mlir --check-prefix=META
// RUN: triton-opt %t/insert_semas_tmem_container_subviews.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_tmem_container_subviews.mlir
// RUN: triton-opt %t/insert_semas_tmem_no_loop_exit_drain.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_tmem_no_loop_exit_drain.mlir
// RUN: triton-opt %t/insert_semas_tmem_reuse_views.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_tmem_reuse_views.mlir
// RUN: triton-opt %t/insert_semas_transitive_reduction.mlir -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_transitive_reduction.mlir
// RUN: triton-opt %t/insert_semas_uniform_hold_transparency.mlir -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %t/insert_semas_uniform_hold_transparency.mlir

//--- insert_semas.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, rank = 3}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // Single-buffered (1x): alloc, create semaphores, initial acquire+store
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[INIT:%.*]] = nvws.semaphore.acquire [[EMPTY]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[INIT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[INIT]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[INIT_BUF]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %3 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %7 = ttg.memdesc_trans %6 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // Buffer from EMPTY sem used in MMA
      // CHECK: [[CURRENT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[INIT]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[CURRENT_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %8 = ttng.tc_gen5_mma %5, %7, %result[%arg6], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %8 : !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After the loop, the captured input token releases FULL for the final load.
    // CHECK: nvws.semaphore.release [[FULL]], [[INIT]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[FINAL:%.*]] = nvws.semaphore.acquire [[FULL]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[FINAL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[FINAL]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[FINAL_BUF]][] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // CHECK-NOT: nvws.semaphore.release
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_unconditional_user
  tt.func @matmul_tma_acc_with_unconditional_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered (2x): alloc, create semaphores, initial acquire+store
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %0 = ttng.tmem_store %cst_0, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer from EMPTY sem, then release FULL
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // Consumer: acquire FULL, buffer, load, release EMPTY
      // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V11]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %result_1, %token_2 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "acc_user"(%result_1) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // Re-acquire EMPTY for next store
      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V2]], [[V12]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V14:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V13]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %8 = ttng.tmem_store %cst, %result[%token_2], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: scf.yield {{.*}}[[V12]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %8 : !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_user
  tt.func @matmul_tma_acc_with_conditional_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered (2x): alloc, create, initial acquire+store
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %0 = ttng.tmem_store %cst_0, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer from EMPTY sem
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V9]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>}: i32
      // The conditional returns the owner-{1} token: the then path hands the
      // buffer to {0} and back, while the else path passes its input through.
      %9 = scf.if %8 -> (!ttg.async.token) {
        // CHECK: [[IF_TOKEN:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V11]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %result_1, %token_2 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_1) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[BACK:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: scf.yield {{.*}}[[BACK]] : !ttg.async.token
        scf.yield %token_2 : !ttg.async.token
      } else {
        // CHECK: } else {
        // CHECK-NEXT: scf.yield {{.*}}[[V7]] : !ttg.async.token
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK-NEXT: [[V13:%.*]] = nvws.semaphore.buffer [[V2]], [[IF_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V14:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V13]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tmem_store %cst, %result[%9], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: scf.yield {{.*}}[[IF_TOKEN]]
      scf.yield %10 : !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs= [array<i32: 1>]}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_def
  tt.func @matmul_tma_acc_with_conditional_def(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered: alloc, create, initial acquire+store
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer, then release
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 : i32
      // Consumer: acquire FULL, buffer, load, release EMPTY
      // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V11]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // Re-acquire EMPTY, buffer, conditional store
      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V2]], [[V12]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V14:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V13]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %9 = ttng.tmem_store %cst, %result[%token_1], %8 {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: scf.yield {{.*}}[[V12]]
      scf.yield %9 : !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 6 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 6 : i32}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use
  tt.func @matmul_tma_acc_with_conditional_def_and_use(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered: alloc, create, initial acquire+store
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V9]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>}: i32
      // The conditional returns the owner-{1} token: the then path hands the
      // buffer to {0} and back, while the else path passes its input through.
      %9 = scf.if %8 -> (!ttg.async.token) {
        // CHECK: [[IF_TOKEN:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V11]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[BACK:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: scf.yield {{.*}}[[BACK]] : !ttg.async.token
        scf.yield %token_1 : !ttg.async.token
      } else {
        // CHECK: } else {
        // CHECK-NEXT: scf.yield {{.*}}[[V7]] : !ttg.async.token
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK-NEXT: [[V13:%.*]] = nvws.semaphore.buffer [[V2]], [[IF_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V14:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V13]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tmem_store %cst, %result[%9], %8 {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: scf.yield {{.*}}[[IF_TOKEN]]
      scf.yield %10 : !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use_no_multibuf_flag
  tt.func @matmul_tma_acc_with_conditional_def_and_use_no_multibuf_flag(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Single-buffered (1x) because of tt.disallow_acc_multi_buffer
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V9:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V8:%.*]] = [[V4]]) -> (i1, !ttg.async.token)  : i32 {
    %1:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0) -> (i1, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V2]], [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg4], %arg3, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>}: i32
      %9 = arith.cmpi ne, %arg2, %c0_i32 {ttg.partition = array<i32: 1>} : i32
      // The if carries the owner-{1} token. The then branch performs the
      // complete {1}->{0}->{1} handoff; the else branch passes [[V8]] through.
      %10 = scf.if %8 -> (!ttg.async.token) {
        // CHECK: [[IF_TOKEN:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[V3]], [[V8]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "some_op"() {ttg.partition = array<i32: 0>} : () -> ()
        // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V3]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[BACK:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: scf.yield {{.*}}[[BACK]] : !ttg.async.token
        scf.yield %token_1 : !ttg.async.token
      } else {
        // CHECK: } else {
        // CHECK-NEXT: scf.yield {{.*}}[[V8]] : !ttg.async.token
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK-NOT: nvws.semaphore.acquire [[V2]]
      // CHECK: scf.yield {{.*}}[[IF_TOKEN]] : i1, !ttg.async.token
      scf.yield %9, %10 : i1, !ttg.async.token
    // CHECK: } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 8 : i32}
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>], tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 8 : i32}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }

  // CHECK-LABEL: @matmul_scaled_rhs_scales_tma
  tt.func @matmul_scaled_rhs_scales_tma(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf8E4M3FN, #shared2>, %arg4: !tt.tensordesc<128x64xf8E4M3FN, #shared2>, %arg5: !tt.tensordesc<128x8xi8, #shared3>) {
    %cst = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // LHS scales (no semaphore - static)
    %result = ttng.tmem_alloc %cst : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    // ACC buffer: alloc, create, initial acquire+store
    %result_1, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[ACC_EMPTY:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[ACC_FULL:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[ACC_INIT:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[ACC_INIT_BUF:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]], [[ACC_INIT]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[ACC_INIT_BUF]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %0 = ttng.tmem_store %cst_0, %result_1[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // RHS scales: alloc + semaphore pair
    // CHECK: [[V7:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V7]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V9:%.*]] = nvws.semaphore.create [[V7]] {pending_count = 1 : i32} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    %1 = scf.for %arg6 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg7 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg6, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %3 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf8E4M3FN, #shared2> -> tensor<128x64xf8E4M3FN, #blocked1>
      %4 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf8E4M3FN, #shared2> -> tensor<128x64xf8E4M3FN, #blocked1>
      %5 = tt.descriptor_load %arg5[%arg1, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x8xi8, #shared3> -> tensor<128x8xi8, #linear>
      %6 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>
      %7 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>
      %8 = ttg.memdesc_trans %7 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>
      // RHS scales: acquire SEMPTY, buffer, store, release SFULL
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V8]], [[V10]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V11]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x8xi8, #linear> -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      %result_2 = ttng.tmem_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      // ACC buffer + RHS scales buffer for MMA
      // CHECK: nvws.semaphore.release [[V9]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[ACC_BUF:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]], [[ACC_INIT]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.buffer [[V9]], [[V13]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma_scaled %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[ACC_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, [[V14]], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 1>}
      %9 = ttng.tc_gen5_mma_scaled %6, %8, %result_1[%arg7], %result, %result_2, %true, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %9 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 9 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: release FULL, acquire FULL, buffer, load; no trailing EMPTY release (no further access)
    // CHECK: nvws.semaphore.release [[V8]], [[V13]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 9 : i32}
    // CHECK: nvws.semaphore.release [[ACC_FULL]], [[ACC_INIT]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 9 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[FINAL:%.*]] = nvws.semaphore.acquire [[ACC_FULL]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[FINAL_BUF:%.*]] = nvws.semaphore.buffer [[ACC_FULL]], [[FINAL]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[FINAL_BUF]][] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %val, %tok = ttng.tmem_load %result_1[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // CHECK-NOT: nvws.semaphore.release
    "use"(%val) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @user_partition_has_cycle
  tt.func @user_partition_has_cycle(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %0 = tt.descriptor_load %arg3[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
    %1 = ttg.local_alloc %0 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    // Double-buffered producer/consumer cycle in loop
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V4:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (tensor<128x128xf32, #blocked>)  : i32 {
    %2:2 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %token) -> (tensor<128x128xf32, #blocked>, !ttg.async.token)  : i32 {
      %3 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %4 = tt.descriptor_load %arg4[%arg2, %3] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.memdesc_trans %5 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %7 = ttng.tc_gen5_mma %1, %6, %result[%arg7], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.addf %arg6, %arg6 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V3]], [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V9]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %9 = arith.mulf %8, %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      scf.yield %9, %token_1 : tensor<128x128xf32, #blocked>, !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 11 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 11 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
    // After loop: no drain; there is no post-loop TMEM access.
    "use"(%2#0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use_flag
  tt.func @matmul_tma_acc_with_conditional_def_and_use_flag(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered with use_d flag
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V9:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V8:%.*]] = [[V4]]) -> (i1, !ttg.async.token)  : i32 {
    %1:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0) -> (i1, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V2]], [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg4], %arg3, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %9 = arith.cmpi ne, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %10 = scf.if %8 -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[V3]], [[V8]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "some_op"() {ttg.partition = array<i32: 0>} : () -> ()
        // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V3]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_1 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      scf.yield %9, %10 : i1, !ttg.async.token
    // CHECK: } {tt.num_stages = 4 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 12 : i32}
    } {tt.num_stages = 4 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 12 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1>, array<i32: 1>]}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }

  // CHECK-LABEL: @specialize_mma_only
  tt.func @specialize_mma_only(%arg0: !tt.tensordesc<64x128xf16, #shared>, %arg1: !ttg.memdesc<128x64xf16, #shared, #smem>, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Reversed pattern: partition 0 stores, partition 1 does MMA
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V6:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: [[V7:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V6]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V9:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V8:%.*]] = [[V5]]) -> (!ttg.async.token)  : i32 {
    %1 = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg4 = %0) -> (!ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg0[%arg3, %arg3] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V2]], [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V10]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %result_2, %token_3 = ttng.tmem_load %result[%arg4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %3:2 = "some_producer"(%2, %result_2) {ttg.partition = array<i32: 0>} : (tensor<64x128xf16, #blocked1>, tensor<128x128xf32, #blocked>) -> (tensor<128x64xf16, #blocked1>, tensor<128x128xf32, #blocked>)
      %4 = ttg.local_alloc %3#0 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %5 = ttg.memdesc_trans %4 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: [[V11:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V10]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %6 = ttng.tmem_store %3#1, %result[%token_3], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V3]], [[V12]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %7 = ttng.tc_gen5_mma %arg1, %5, %result[%6], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V2]], [[V12]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V14]]
      scf.yield %7 : !ttg.async.token
    // CHECK: } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 15 : i32}
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 15 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>]}
    // After loop: release/acquire/buffer/load/release
    // CHECK: nvws.semaphore.release [[V4]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 15 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V4]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V4]], [[V15]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V16]][] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @load_scale_mma_user
  tt.func @load_scale_mma_user(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem>, %arg2: !tt.tensordesc<8x128xi8, #shared>, %arg3: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // ACC buffer + scale buffer each get their own semaphore pairs
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V6:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: [[V7:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V6]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: [[V9:%.*]] = nvws.semaphore.create [[V8]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V10:%.*]] = nvws.semaphore.create [[V8]] {pending_count = 1 : i32} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V12:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V11:%.*]] = [[V5]]) -> (!ttg.async.token)  : i32 {
    %1 = scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg2[%arg5, %arg5] {ttg.partition = array<i32: 2>} : !tt.tensordesc<8x128xi8, #shared> -> tensor<8x128xi8, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = array<i32: 2>} : (tensor<8x128xi8, #blocked1>) -> !ttg.memdesc<8x128xi8, #shared, #smem>
      // CHECK: [[V13:%.*]] = ttg.local_load %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : !ttg.memdesc<8x128xi8, #shared, #smem> -> tensor<8x128xi8, #linear1>
      %4 = ttg.local_load %3 {ttg.partition = array<i32: 0>} : !ttg.memdesc<8x128xi8, #shared, #smem> -> tensor<8x128xi8, #linear1>
      %5 = tt.trans %4 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<8x128xi8, #linear1> -> tensor<128x8xi8, #linear>
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V9]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V9]], [[V14]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V15]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x8xi8, #linear> -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      %result_2 = ttng.tmem_alloc %5 {ttg.partition = array<i32: 0>} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      // CHECK: nvws.semaphore.release [[V10]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V2]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V17:%.*]] = nvws.semaphore.acquire [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V18:%.*]] = nvws.semaphore.buffer [[V10]], [[V17]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma_scaled %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V16]][], [[V18]], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 1>}
      %6 = ttng.tc_gen5_mma_scaled %arg0, %arg1, %result[%arg6], %result_2, %arg3, %true, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>

      // CHECK: nvws.semaphore.release [[V9]], [[V17]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V3]], [[V11]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V19:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V20:%.*]] = nvws.semaphore.buffer [[V3]], [[V19]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V20]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %result_3, %token_4 = ttng.tmem_load %result[%6] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V19]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V21]]
      scf.yield %token_4 : !ttg.async.token
    // CHECK: } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 16 : i32}
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 16 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: release rides the loop-carried EMPTY-acquire token (payload none; MMA completion
    // already ordered through the in-loop FULL release)
    // CHECK: nvws.semaphore.release [[V4]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 16 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V4]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V23:%.*]] = nvws.semaphore.buffer [[V4]], [[V22]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V23]][] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @store_mma_load
  tt.func @store_mma_load(%arg0: i32, %arg1: !tt.tensordesc<128x64xf16, #shared>, %arg2: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // disallow_acc_multi_buffer => single-buffered
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 17 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V6:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V5:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %0 = scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg4 = %token) -> (!ttg.async.token)  : i32 {
      %1 = tt.descriptor_load %arg1[%arg3, %arg3] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %2 = arith.addf %1, %1 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %4 = "make_acc"() {ttg.partition = array<i32: 0>} : () -> tensor<128x128xf32, #blocked>
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V8:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %5 = ttng.tmem_store %4, %result[%arg4], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %6 = ttng.tc_gen5_mma %3, %arg2, %result[%5], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V2]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %result_0, %token_1 = ttng.tmem_load %result[%6] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: scf.yield {{.*}}[[V11]]
      scf.yield %token_1 : !ttg.async.token
    // CHECK: } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 17 : i32}
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 17 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>]}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }

  // CHECK-LABEL: @local_alloc_into_mma
  tt.func @local_alloc_into_mma(%arg0: i32, %arg1: tensor<128x64xf16, #blocked1>, %arg2: !tt.tensordesc<64x128xf16, #shared>) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // Acquire at the MMA's point of use: tagged with the acquiring partition
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 18 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    %5 = scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg4 = %token) -> (!ttg.async.token)  : i32 {
      %0 = ttg.local_alloc %arg1 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %1 = tt.descriptor_load %arg2[%arg3, %arg3] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %2 = arith.addf %1, %1 {ttg.partition = array<i32: 0>} : tensor<64x128xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = array<i32: 0>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %4 = ttng.tc_gen5_mma %0, %3, %result[%arg4], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %4 : !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 18 : i32}
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 18 : i32}
    // After loop: the dead load (unused result) is dropped; its acquire+buffer remain, no trailing release
    ttng.tmem_load %result[%5] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 18 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: nvws.semaphore.release
    tt.return
  }

  // CHECK-LABEL: @shmem_sink_iterator_invalidation
  tt.func @shmem_sink_iterator_invalidation(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // Two TMEM allocs: ACC (single-buffered) + LHS (TMEM operand, single-buffered)
    // ACC semaphore pair
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // LHS semaphore pair
    // CHECK: [[V7:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V7]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V9:%.*]] = nvws.semaphore.create [[V7]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %3 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK: [[V10:%.*]] = ttg.local_load %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked2>
      %6 = ttg.local_load %5 {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked2>
      %7 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.memdesc_trans %7 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // LHS: acquire, buffer, store, release
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V8]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 1x128x64>
      // CHECK: ttng.tmem_store [[V10]], [[V12]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked2> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 1x128x64>
      %result_2 = ttng.tmem_alloc %6 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory>
      // ACC buffer + LHS acquire for MMA
      // CHECK: nvws.semaphore.release [[V9]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V9]], [[V14]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 1x128x64>
      // CHECK: ttng.tc_gen5_mma [[V15]], %{{[-A-Za-z0-9_.$#]+}}, [[V13]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %9 = ttng.tc_gen5_mma %result_2, %8, %result[%arg6], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %9 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 19 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: release FULL, acquire FULL, buffer, load; no trailing EMPTY release (no further access)
    // CHECK: nvws.semaphore.release [[V8]], [[V14]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 19 : i32}
    // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 19 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V3]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V17:%.*]] = nvws.semaphore.buffer [[V3]], [[V16]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V17]][] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // CHECK-NOT: nvws.semaphore.release
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_smem_fanout
  tt.func @local_smem_fanout(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 100 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 100 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V7:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = [[V5]]) -> (!ttg.async.token)  : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V4]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V11:%.*]] = ttg.local_load [[V10]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "use_b"(%l1) {ttg.partition = array<i32: 1>} : (!ty) -> ()

      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V4]], [[V12]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V14:%.*]] = ttg.local_load [[V13]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V2]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "use_c"(%l2) {ttg.partition = array<i32: 2>} : (!ty) -> ()

      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V2]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V16]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v2, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: scf.yield {{.*}}[[V15]]
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @local_reg_and_smem_use
  tt.func @local_reg_and_smem_use(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 106 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 106 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V6:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V6]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V3]], [[V7]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V9:%.*]] = ttg.local_load [[V8]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l = ttg.local_load %alloc {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "use_reg"(%l) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ty) -> ()

      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V4]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V4]], [[V10]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "use_smem"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @local_same_owner_no_semaphore
  // CHECK-NOT: nvws.semaphore
  tt.func @local_same_owner_no_semaphore() {
    %alloc = ttg.local_alloc {buffer.id = 101 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 101 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NOT: nvws.semaphore
    %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V1]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NOT: nvws.semaphore
    ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = ttg.local_load [[V1]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
    // CHECK-NOT: nvws.semaphore
    %l = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
    "use"(%l) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    tt.return
  }

  // CHECK-LABEL: @local_loop_carried_and_result
  tt.func @local_loop_carried_and_result(%lb: i32, %ub: i32, %step: i32, %init: !ty) {
    %iter_alloc = ttg.local_alloc {buffer.id = 104 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %result_alloc = ttg.local_alloc {buffer.id = 105 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 104 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = ttg.local_alloc {buffer.id = 105 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V4]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V4]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (tensor<1xi32, #blocked>)  : i32 {
    %r = scf.for %i = %lb to %ub step %step iter_args(%arg = %init) -> (!ty) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V2]], [[V9]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store [[V7]], [[V10]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %arg, %iter_alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V3]], [[V11]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V13:%.*]] = ttg.local_load [[V12]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l = ttg.local_load %iter_alloc {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V2]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "use_iter"(%l) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ty) -> ()

      %next = "next"(%arg) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (!ty) -> !ty
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : !ty
    // CHECK: } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}

    // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V5]], [[V14]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: ttg.local_store [[V8]], [[V15]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    ttg.local_store %r, %result_alloc {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

    // CHECK: nvws.semaphore.release [[V6]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V17:%.*]] = nvws.semaphore.buffer [[V6]], [[V16]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V18:%.*]] = ttg.local_load [[V17]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
    %lr = ttg.local_load %result_alloc {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
    // No trailing EMPTY release: the result buffer is not accessed again
    // CHECK-NOT: nvws.semaphore.release
    "use_result"(%lr) {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : (!ty) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @local_release_after_mma
  tt.func @local_release_after_mma(%desc: !tt.tensordesc<128x64xf16, #shared>, %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %alloc = ttg.local_alloc {buffer.id = 102 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 102 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %iv = %lb to %ub step %step : i32 {
      // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc[%i, %i] 16384 %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %nt = ttng.tc_gen5_mma %alloc, %rhs, %acc[%tok], %true, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @local_release_after_descriptor_store
  tt.func @local_release_after_descriptor_store(%desc: !tt.tensordesc<128x128xf16, #shared>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc {buffer.id = 103 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 103 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %iv = %lb to %ub step %step : i32 {
      %v = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : () -> tensor<128x128xf16, #linear>
      // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V8:%.*]] = ttg.local_load [[V7]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %l = ttg.local_load %alloc {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %cvt = ttg.convert_layout %l {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc[%i, %c0], %cvt {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x128xf16, #shared>, tensor<128x128xf16, #blocked1>
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @attention_forward
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<64x64xf16, #shared>, %arg2: !tt.tensordesc<64x64xf16, #shared>, %arg3: f32, %arg4: i32) {
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    // Three TMEM allocs: S is double-buffered; O and P are single-slot.
    // S semaphore pair
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // O semaphore pair
    %result_2, %token_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V4]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V4]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V4]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V5]] : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V5]], [[V8]] : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    // CHECK: [[V10:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %0 = ttng.tmem_store %cst_0, %result_2[%token_3], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // P semaphore pair
    // CHECK: [[V11:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V12:%.*]] = nvws.semaphore.create [[V11]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V13:%.*]] = nvws.semaphore.create [[V11]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V17:%.*]]:3 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V14:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V15:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V16:%.*]] = [[V8]]) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token)  : i32 {
    %1:4 = scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg6 = %cst, %arg7 = %cst_1, %arg8 = %token, %arg9 = %0) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg1[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      // S: buffer, mma, release
      // CHECK: [[V18:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V19:%.*]] = nvws.semaphore.buffer [[V2]], [[V18]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %5 = ttng.tc_gen5_mma %arg0, %4, %result[%arg8], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      // S: acquire, buffer, load, release
      // CHECK: nvws.semaphore.release [[V3]], [[V18]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V20:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V21:%.*]] = nvws.semaphore.buffer [[V3]], [[V20]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V21]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
      %result_6, %token_7 = ttng.tmem_load %result[%5] {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

      // CHECK: nvws.semaphore.release [[V2]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %6 = "compute_row_max"(%result_6, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %7 = "sub_row_max"(%result_6, %6, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %8 = math.exp2 %7 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked>
      %9 = arith.subf %arg7, %6 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %10 = arith.subf %arg7, %6 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %11 = math.exp2 %9 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = math.exp2 %10 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = "tt.reduce"(%8) <{axis = 1 : i32}> ({
      ^bb0(%arg10: f32, %arg11: f32):
        %24 = arith.addf %arg10, %arg11 {ttg.partition = array<i32: 0>}: f32
        tt.reduce.return %24 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = arith.mulf %arg6, %12 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %15 = arith.addf %14, %13 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = tt.expand_dims %11 {axis = 1 : i32, ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %17 = tt.broadcast %16 {ttg.partition = array<i32: 3>} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>

      // O: buffer, load
      // CHECK: [[V22:%.*]] = nvws.semaphore.buffer [[V5]], [[V16]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V22]][] {ttg.partition = array<i32: 3>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
      %result_8, %token_9 = ttng.tmem_load %result_2[%arg9] {ttg.partition = array<i32: 3>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

      %18 = arith.mulf %result_8, %17 {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked>
      %19 = tt.descriptor_load %arg2[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked1>
      %20 = ttg.local_alloc %19 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %21 = arith.truncf %8 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      // P: buffer, store, release
      // CHECK: [[V23:%.*]] = nvws.semaphore.acquire [[V12]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V24:%.*]] = nvws.semaphore.buffer [[V12]], [[V23]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V24]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %result_10 = ttng.tmem_alloc %21 {ttg.partition = array<i32: 0>} : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>

      // O: store, release
      // CHECK: nvws.semaphore.release [[V13]], [[V23]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V25:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V22]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %22 = ttng.tmem_store %18, %result_2[%token_9], %true {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      // O: acquire, buffer for MMA
      // P: acquire, buffer for MMA
      // P+O: release after MMA
      // CHECK: nvws.semaphore.release [[V6]], [[V16]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V26:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V27:%.*]] = nvws.semaphore.buffer [[V6]], [[V26]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: [[V28:%.*]] = nvws.semaphore.acquire [[V13]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V29:%.*]] = nvws.semaphore.buffer [[V13]], [[V28]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %23 = ttng.tc_gen5_mma %result_10, %20, %result_2[%22], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      // S+O: acquire for next iter
      scf.yield %15, %6, %token_7, %23 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>]}
    // After loop: only O is consumed after the loop.
    // CHECK: nvws.semaphore.release [[V12]], [[V28]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: nvws.semaphore.release [[V5]], [[V26]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V30:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 3>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: nvws.semaphore.release [[V7]], [[V17]]#2 [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V31:%.*]] = nvws.semaphore.acquire [[V7]] : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V32:%.*]] = nvws.semaphore.buffer [[V7]], [[V31]] : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V32]][] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    %result_4, %token_5 = ttng.tmem_load %result_2[%1#3] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    "use"(%1#0, %result_4, %1#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @hoisted_alloc
  tt.func @hoisted_alloc(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // Hoisted alloc with nested loops
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %res, %tok = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V7:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
      %ptrub = tt.addptr %ptr0, %iv0 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>, i32
      %ub1 = tt.load %ptrub {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>
      %lb1 = "lb1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      %step1 = "step1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %tok4 = scf.for %iv = %lb1 to %ub1 step %step1 iter_args(%tok2 = %tok1) -> (!ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %tok3 = ttng.tc_gen5_mma %sA, %sB, %res[%tok2], %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %tok3 : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V10]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V11]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    // After outer loop: no drain; there is no post-loop TMEM access.
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @if_split_workaround
  tt.func @if_split_workaround(%arg0: !tt.tensordesc<1x64xf16, #shared>, %arg1: tensor<64x128x!tt.ptr<f16>, #blocked3> {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    // Single-buffered (disallow_acc_multi_buffer)
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V10:%.*]]:3 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V8:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V9:%.*]] = [[V4]]) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.async.token)  : i32 {
    %1:3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %arg1, %arg5 = %0) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : (i32) -> (i32, tensor<64x128xi32, #blocked3>, i32)
      %3 = tt.splat %2#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #blocked2>
      %4 = tt.descriptor_gather %arg0[%3, %2#2] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (!tt.tensordesc<1x64xf16, #shared>, tensor<128xi32, #blocked2>, i32) -> tensor<128x64xf16, #blocked1>
      %5 = tt.addptr %arg4, %2#1 {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked3>, tensor<64x128xi32, #blocked3>
      %6 = tt.load %5 {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked3>
      %7 = ttg.local_alloc %4 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V2]], [[V9]] {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %9 = ttng.tc_gen5_mma %7, %8, %result[%arg5], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %10 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %11 = arith.select %10, %false, %true {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : i1
      %12 = scf.if %10 -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[V3]], [[V9]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V3]], [[V12]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V13]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked1>
        %result_0, %token_1 = ttng.tmem_load %result[%9] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %token_1 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %9 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %11, %5, %12 : i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token
    // CHECK: } {tt.disallow_acc_multi_buffer, tt.num_stages = 3 : i32, tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 3 : i32, tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @nested_loop_yes_double_buffer
  tt.func @nested_loop_yes_double_buffer(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // Double-buffered: inner loop store is in partition 2 (same as MMA producer)
    %res, %tok = ttng.tmem_alloc : () ->(!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %toka = ttng.tmem_store %cst, %res[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %toka) -> (!ttg.async.token) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V10:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %tok1a = ttng.tmem_store %cst, %res[%tok1], %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V12:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V11:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i1)  : i32 {
      %useD, %tok4 = scf.for %iv = %lb to %ub step %step iter_args(%useD = %false, %tok2 = %tok1a) -> (i1, !ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %tok3 = ttng.tc_gen5_mma %sA, %sB, %res[%tok2], %useD, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %tok3 : i1, !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V3]], [[V14]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V15]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V16]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @nested_loop_cross_partition_fresh_mma_double_buffer
  tt.func @nested_loop_cross_partition_fresh_mma_double_buffer(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // The cross-partition store -> fresh-MMA handoff uses two accumulator copies.
    %res, %tok = ttng.tmem_alloc : () ->(!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[NESTED_ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[NESTED_EMPTY:%.*]] = nvws.semaphore.create [[NESTED_ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[NESTED_FULL:%.*]] = nvws.semaphore.create [[NESTED_ALLOC]] released = 2 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[NESTED_INIT:%.*]] = nvws.semaphore.acquire [[NESTED_EMPTY]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[NESTED_INIT_BUF:%.*]] = nvws.semaphore.buffer [[NESTED_EMPTY]], [[NESTED_INIT]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[NESTED_INIT_BUF]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %toka = ttng.tmem_store %cst, %res[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[NESTED_OUTER_TOKEN:%.*]] = [[NESTED_INIT]]) -> (!ttg.async.token)  : i32 {
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %toka) -> (!ttg.async.token) : i32 {
      // CHECK: [[NESTED_STORE_BUF:%.*]] = nvws.semaphore.buffer [[NESTED_EMPTY]], [[NESTED_OUTER_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[NESTED_STORE_BUF]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %tok1a = ttng.tmem_store %cst, %res[%tok1], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[NESTED_FULL]], [[NESTED_OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[NESTED_MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[NESTED_FULL]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i1)  : i32 {
      %useD, %tok4 = scf.for %iv = %lb to %ub step %step iter_args(%useD = %false, %tok2 = %tok1a) -> (i1, !ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[NESTED_MMA_BUF:%.*]] = nvws.semaphore.buffer [[NESTED_FULL]], [[NESTED_MMA_TOKEN]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[NESTED_MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>}
        %tok3 = ttng.tc_gen5_mma %sA, %sB, %res[%tok2], %useD, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %tok3 : i1, !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      // CHECK: nvws.semaphore.release [[NESTED_EMPTY]], [[NESTED_MMA_TOKEN]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[NESTED_READ_TOKEN:%.*]] = nvws.semaphore.acquire [[NESTED_EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[NESTED_READ_BUF:%.*]] = nvws.semaphore.buffer [[NESTED_EMPTY]], [[NESTED_READ_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[NESTED_READ_BUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: scf.yield {{.*}}[[NESTED_READ_TOKEN]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @cross_partition_constant_fresh_mma
  tt.func @cross_partition_constant_fresh_mma(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %res, %tok = ttng.tmem_alloc {buffer.copy = 2 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[CONST_ALLOC:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[CONST_EMPTY:%.*]] = nvws.semaphore.create [[CONST_ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[CONST_FULL:%.*]] = nvws.semaphore.create [[CONST_ALLOC]] released = 2 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[CONST_INIT:%.*]] = nvws.semaphore.acquire [[CONST_EMPTY]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[CONST_OUTER_TOKEN:%.*]] = [[CONST_INIT]]) -> (!ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%outerTok = %tok) -> (!ttg.async.token) : i32 {
      // CHECK: [[CONST_STORE_BUF:%.*]] = nvws.semaphore.buffer [[CONST_EMPTY]], [[CONST_OUTER_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[CONST_STORE_BUF]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %storeTok = ttng.tmem_store %cst, %res[%outerTok], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[CONST_FULL]], [[CONST_OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[CONST_MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[CONST_FULL]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %inner = scf.for %iv = %lb to %ub step %step iter_args(%innerTok = %storeTok) -> (!ttg.async.token) : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[CONST_MMA_BUF:%.*]] = nvws.semaphore.buffer [[CONST_FULL]], [[CONST_MMA_TOKEN]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[CONST_MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>}
        %mmaTok = ttng.tc_gen5_mma %sA, %sB, %res[%innerTok], %false, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %mmaTok : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
      // CHECK: nvws.semaphore.release [[CONST_EMPTY]], [[CONST_MMA_TOKEN]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[CONST_READ_TOKEN:%.*]] = nvws.semaphore.acquire [[CONST_EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[CONST_READ_BUF:%.*]] = nvws.semaphore.buffer [[CONST_EMPTY]], [[CONST_READ_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[CONST_READ_BUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %value, %readTok = ttng.tmem_load %res[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: scf.yield {{.*}}[[CONST_READ_TOKEN]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %readTok : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }

  // CHECK-LABEL: @cross_partition_accumulating_mma
  tt.func @cross_partition_accumulating_mma(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %res, %tok = ttng.tmem_alloc {buffer.copy = 2 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: ttng.tmem_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%outerTok = %tok) -> (!ttg.async.token) : i32 {
      %storeTok = ttng.tmem_store %cst, %res[%outerTok], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %inner = scf.for %iv = %lb to %ub step %step iter_args(%innerTok = %storeTok) -> (!ttg.async.token) : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        %mmaTok = ttng.tc_gen5_mma %sA, %sB, %res[%innerTok], %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %mmaTok : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
      %value, %readTok = ttng.tmem_load %res[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %readTok : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: @nested_loop_yes_double_buffer_scaled
  tt.func @nested_loop_yes_double_buffer_scaled(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>,
    %scalesA: tensor<128x8xi8, #linear>, %scalesB: tensor<128x8xi8, #linear>) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // Double-buffered with scaled MMA
    %res, %tok = ttng.tmem_alloc : () ->(!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %toka = ttng.tmem_store %cst, %res[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %lhs_scales = ttng.tmem_alloc %scalesA: (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %rhs_scales = ttng.tmem_alloc %scalesB : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %toka) -> (!ttg.async.token) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V10:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %tok1a = ttng.tmem_store %cst, %res[%tok1], %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V12:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V11:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i1)  : i32 {
      %useD, %tok4 = scf.for %iv = %lb to %ub step %step iter_args(%useD = %false, %tok2 = %tok1a) -> (i1, !ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %tok3 = ttng.tc_gen5_mma_scaled %sA, %sB, %res[%tok2], %lhs_scales, %rhs_scales, %useD, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %tok3 : i1, !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V3]], [[V14]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V15]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V16]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @nested_loop_no_double_buffer_scaled
  tt.func @nested_loop_no_double_buffer_scaled(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>,
    %scalesA: tensor<128x8xi8, #linear>, %scalesB: tensor<128x8xi8, #linear>) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // Single-buffered: inner loop store in partition 2 but 128x256 is too large
    %res, %tok = ttng.tmem_alloc : () ->(!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
    // CHECK: [[V6:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
    %toka = ttng.tmem_store %cst, %res[%tok], %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %lhs_scales = ttng.tmem_alloc %scalesA : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %rhs_scales = ttng.tmem_alloc %scalesB : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %toka) -> (!ttg.async.token) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
      // CHECK: [[V10:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
      %tok1a = ttng.tmem_store %cst, %res[%tok1], %true {ttg.partition = array<i32: 2>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V12:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V11:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i1)  : i32 {
      %useD, %tok4 = scf.for %iv = %lb to %ub step %step iter_args(%useD = %false, %tok2 = %tok1a) -> (i1, !ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x256xf32, #shared, #smem>
        // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
        %tok3 = ttng.tc_gen5_mma_scaled %sA, %sB, %res[%tok2], %lhs_scales, %rhs_scales, %useD, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x256xf32, #shared, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %tok3 : i1, !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V3]], [[V14]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V15]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256> -> tensor<128x256xf32, #blocked>
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x256xf32, #blocked>) -> ()
      // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V16]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// Test that tmem allocations in functions that do not use warp specialization
// do not trigger an assert if they have multiple uses.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @test_tmem_no_ws
  tt.func public @test_tmem_no_ws(%arg0: !ttg.memdesc<128x128xi8, #shared, #smem>, %arg1: !ttg.memdesc<128x128xi8, #shared1, #smem>, %arg2: !ttg.memdesc<128x128xi8, #shared1, #smem>, %arg3: tensor<128x16xf8E4M3FN, #linear>, %arg4: tensor<128x16xf8E4M3FN, #linear>, %arg5: tensor<128x16xf8E4M3FN, #linear>) {
    %true = arith.constant true
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_0, %token_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_2 = ttng.tmem_alloc %arg3 : (tensor<128x16xf8E4M3FN, #linear>) -> !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    %result_3 = ttng.tmem_alloc %arg4 : (tensor<128x16xf8E4M3FN, #linear>) -> !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    %result_4 = ttng.tmem_alloc %arg5 : (tensor<128x16xf8E4M3FN, #linear>) -> !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    %0 = ttng.tc_gen5_mma_scaled %arg0, %arg1, %result[%token], %result_2, %result_3, %true, %true lhs = e2m1 rhs = e2m1 : !ttg.memdesc<128x128xi8, #shared, #smem>, !ttg.memdesc<128x128xi8, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    %1 = ttng.tc_gen5_mma_scaled %arg0, %arg2, %result_0[%token_1], %result_2, %result_4, %true, %true lhs = e2m1 rhs = e2m1 : !ttg.memdesc<128x128xi8, #shared, #smem>, !ttg.memdesc<128x128xi8, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    tt.return
  }
}

//--- insert_semas_async_entry_fanin.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @async_entry_fanin
  tt.func @async_entry_fanin(
      %desc: !tt.tensordesc<128x128xf16, #shared>,
      %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 1701 : i32}
    // CHECK: [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32}
    // CHECK: [[INNER_READY:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 2 : i32}
    %alloc = ttg.local_alloc {buffer.id = 1701 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[OUTER_TOKEN:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: [[OUTER_BUFFER:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_TOKEN]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[OUTER_BUFFER]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: nvws.semaphore.release [[INNER_READY]], [[OUTER_TOKEN]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>}
      // CHECK-NEXT: nvws.semaphore.release [[INNER_READY]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>}
      nvws.descriptor_load %desc[%i, %i] 32768 %alloc {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      scf.for %j = %lb to %ub step %step : i32 {
        // CHECK-NEXT: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[INNER_READY]] {ttg.partition = array<i32: 2>}
        %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
        "use2"(%l2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        %l1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
        %corrected = "correct"(%l1) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
        ttg.local_store %corrected, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %l0 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
        "use0"(%l0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      } {ttg.partition = array<i32: 0, 1, 2>}
      // CHECK: [[DONE:%.*]] = nvws.semaphore.acquire [[INNER_READY]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[OUTER_EMPTY]], [[DONE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_branch_local_init.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @branch_local_initial_acquire_stays_with_create
  tt.func @branch_local_initial_acquire_stays_with_create(
      %guard: i1,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    // CHECK: [[POISON:%.*]] = ub.poison : !ttg.async.token
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %false = arith.constant false
    %true = arith.constant true

    // The semaphore storage is branch-local here. Both nvws.semaphore.create
    // ops must stay in the same branch as the tmem_alloc; hoisting them above
    // the scf.if violates SSA dominance. The EMPTY semaphore is created
    // initially released, which supplies iteration zero to the in-loop
    // acquire.
    scf.if %guard {
      scf.yield
    } else {
      %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
      // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
      // CHECK: [[V5:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V4:%.*]] = [[POISON]]) -> (!ttg.async.token)  : i32 {
      %loop = scf.for %iv = %c0 to %c4 step %c1 iter_args(%carry = %tok) -> (!ttg.async.token) : i32 {
        // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V7]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V3]], [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V9]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %loaded, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%loaded) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[POISON]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %load_tok : !ttg.async.token
      // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
      } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
      // CHECK: "token_user"([[V5]]) : (!ttg.async.token) -> ()
      "token_user"(%loop) : (!ttg.async.token) -> ()
      scf.yield
    }
    tt.return
  }
}

//--- insert_semas_cached_exact_reuse.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // IR-LABEL: @cached_exact_reuse_after_release
  // DAG-LABEL: function: @cached_exact_reuse_after_release
  // DAG: |- a  S1  {1}
  // DAG: |- r  S0  {1} [none]
  // DAG: |- a  S0  {0}
  // DAG: |- r  S1  {0} [none]
  tt.func @cached_exact_reuse_after_release(%lb: i32, %ub: i32, %step: i32) {
    // IR: [[BASE:%[-A-Za-z0-9_.$#]+]] = ttg.local_alloc {buffer.id = 9820 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // IR: [[ENTRY:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.create [[BASE]] released = 1 {pending_count = 1 : i32}
    // IR: [[FULL:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32}
    %buf = ttg.local_alloc {buffer.id = 9820 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %value = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
    scf.for %i = %lb to %ub step %step : i32 {
      // IR: [[P1_TOKEN:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[ENTRY]] {ttg.partition = array<i32: 1>}
      // IR: [[P1_WRITE_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[ENTRY]], [[P1_TOKEN]] {ttg.partition = array<i32: 1>}
      // IR: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[P1_WRITE_BUF]] {ttg.partition = array<i32: 1>}
      ttg.local_store %value, %buf {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // IR: nvws.semaphore.release [[FULL]], [[P1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}

      // IR: [[P0_TOKEN:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // IR: [[P0_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[FULL]], [[P0_TOKEN]] {ttg.partition = array<i32: 0>}
      // IR: [[R0:%[-A-Za-z0-9_.$#]+]] = ttg.local_load [[P0_BUF]] {ttg.partition = array<i32: 0>}
      %r0 = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // IR: nvws.semaphore.release [[ENTRY]], [[P0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      "use0"(%r0) {ttg.partition = array<i32: 0>} : (!ty) -> ()

      // The first post-release read materializes an approved exact-reuse
      // buffer.  The second read must reuse that same view without making the
      // emitted-IR verifier demand a second release.
      // IR: [[P1_REUSE_BUF:%[-A-Za-z0-9_.$#]+]] = nvws.semaphore.buffer [[ENTRY]], [[P1_TOKEN]] {ttg.partition = array<i32: 1>}
      // IR: [[R1A:%[-A-Za-z0-9_.$#]+]] = ttg.local_load [[P1_REUSE_BUF]] {ttg.partition = array<i32: 1>}
      %r1a = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "use1a"(%r1a) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      // IR-NOT: nvws.semaphore.buffer
      // IR: [[R1B:%[-A-Za-z0-9_.$#]+]] = ttg.local_load [[P1_REUSE_BUF]] {ttg.partition = array<i32: 1>}
      %r1b = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "use1b"(%r1b) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_circular_backing_dominance.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The start-1 allocation appears first, while producer order remains
  // start0, start1. Folding must move the canonical start-0 backing before the
  // merged semaphore creates.
  // CHECK-LABEL: @circular_start_zero_backing_dominates_creates
  tt.func @circular_start_zero_backing_dominates_creates(
      %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[S1_PAYLOAD:%.*]] = "make_start1"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %start1_payload = "make_start1"() {ttg.partition = array<i32: 1>} : () -> !ty
    // CHECK: [[S0_PAYLOAD:%.*]] = "make_start0"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %start0_payload = "make_start0"() {ttg.partition = array<i32: 1>} : () -> !ty
    %start1 = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 700 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 700 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %start0 = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 700 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      // CHECK: [[P1_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[S0_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[P1_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[S0_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[P1_STAGE]]{{\]}}, [[S0_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store [[S0_PAYLOAD]], [[S0_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[P1_STAGE]]{{\]}}, [[S0_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %start0_payload, %start0 {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[P2_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[S0_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[P2_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[S0_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[P2_STAGE]]{{\]}}, [[S0_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[S0_VAL:%.*]] = ttg.local_load [[S0_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[P2_STAGE]]{{\]}}, [[S0_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %start0_value = ttg.local_load %start0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[S1_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[P1_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[S1_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[P1_STAGE]]{{\]}}, [[S1_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store [[S1_PAYLOAD]], [[S1_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[P1_STAGE]]{{\]}}, [[S1_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %start1_payload, %start1 {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[S1_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[P2_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[S1_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[P2_STAGE]]{{\]}}, [[S1_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[S1_VAL:%.*]] = ttg.local_load [[S1_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[P2_STAGE]]{{\]}}, [[S1_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %start1_value = ttg.local_load %start1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[S0_VAL]], [[S1_VAL]]) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%start0_value, %start1_value) {ttg.partition = array<i32: 2>} : (!ty, !ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_circular_smem.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_model_descriptor_load_mma
  // COUNT-LABEL: @circular_model_descriptor_load_mma
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_model_descriptor_load_mma(
      %desc_k: !tt.tensordesc<128x128xf16, #shared>,
      %desc_v: !tt.tensordesc<128x128xf16, #shared>,
      %lhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %tok: !ttg.async.token) {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    scf.for %iv = %c0 to %c1 step %c1 : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.descriptor_load {{.*}} 32768 [[K_EMPTY_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_k[%c0, %c0] 32768 %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kt = ttg.memdesc_trans %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared_t, #smem, mutable>
      // CHECK: [[V_EMPTY_STAGE:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.descriptor_load {{.*}} 32768 [[V_EMPTY_BUF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_v[%c0, %c0] 32768 %v {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_TRANS:%.*]] = ttg.memdesc_trans [[K_FULL_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
      // CHECK: [[QK:%.*]] = ttng.tc_gen5_mma {{.*}}, [[K_TRANS]], {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %qk = ttng.tc_gen5_mma %lhs, %kt, %acc[%tok], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[PV:%.*]] = ttng.tc_gen5_mma {{.*}}, [[V_FULL_BUF]], {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %pv = ttng.tc_gen5_mma %lhs, %v, %acc[%qk], %true, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      "use_token"(%pv) {ttg.partition = array<i32: 1>} : (!ttg.async.token) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 3>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_1_to_2_2
  // COUNT-LABEL: @circular_tutorial_1_1_to_2_2
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_tutorial_1_1_to_2_2(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> !ty
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> !ty
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[K_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_k, %k {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[V_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_v, %v {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %kval = ttg.local_load %k {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %vval = ttg.local_load %v {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%kval, %vval) {ttg.partition = array<i32: 2>} : (!ty, !ty) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 1 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_2_to_3_4
  // COUNT-LABEL: @circular_tutorial_1_2_to_3_4
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_tutorial_1_2_to_3_4(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> !ty
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> !ty
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[K_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_k, %k {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[V_EMPTY_BUF]] {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_v, %v {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_FULL_BUF]] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %kval = ttg.local_load %k {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_FULL_BUF]] {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %vval = ttg.local_load %v {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 3, 4>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%kval, %vval) {ttg.partition = array<i32: 3, 4>} : (!ty, !ty) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      scf.yield {ttg.partition = array<i32: 1, 2, 3, 4>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3, 4>], ttg.warp_specialize.tag = 2 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_1_to_2_3
  // COUNT-LABEL: @circular_tutorial_1_1_to_2_3
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_tutorial_1_1_to_2_3(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> !ty
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> !ty
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[K_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_k, %k {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[V_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_v, %v {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %kval = ttg.local_load %k {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_FULL_BUF]] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %vval = ttg.local_load %v {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 2, 3>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%kval, %vval) {ttg.partition = array<i32: 2, 3>} : (!ty, !ty) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1, 2, 3>} : i32
      scf.yield {ttg.partition = array<i32: 1, 2, 3>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3>], ttg.warp_specialize.tag = 3 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_2_to_3_3
  // COUNT-LABEL: @circular_tutorial_1_2_to_3_3
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_tutorial_1_2_to_3_3(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> !ty
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> !ty
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[K_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_k, %k {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[V_EMPTY_BUF]] {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_v, %v {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_FULL_BUF]] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %kval = ttg.local_load %k {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_FULL_BUF]] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %vval = ttg.local_load %v {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 3>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%kval, %vval) {ttg.partition = array<i32: 3>} : (!ty, !ty) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1, 2, 3>} : i32
      scf.yield {ttg.partition = array<i32: 1, 2, 3>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3>], ttg.warp_specialize.tag = 3 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_conditional_multi_result.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @conditional_multi_result_if_token
  tt.func @conditional_multi_result_if_token(%lhs: !ttg.memdesc<128x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %false = arith.constant false

    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[INITIAL:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[LOOP:%.*]]:3 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[USE_ACC:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[CARRY_VALUE:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[CARRY_TOKEN:%.*]] = [[INITIAL]]) -> (i1, i32, !ttg.async.token)  : i32 {
    %loop:3 = scf.for %iv = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%use_acc = %false, %tok = %acc_tok, %carry = %c0_i32) -> (i1, !ttg.async.token, i32) : i32 {
      // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[CARRY_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V8]][], [[USE_ACC]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok], %use_acc, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cond = arith.cmpi eq, %iv, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32

      // CHECK: [[BRANCH:%.*]]:3 = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (i32, i1, !ttg.async.token) {
      %epilogue:3 = scf.if %cond -> (i32, !ttg.async.token, i1) {
        // CHECK: nvws.semaphore.release [[V3]], [[CARRY_TOKEN]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V10]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %value, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[HAND_BACK:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {{.*}}[[HAND_BACK]] : i32, i1, !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %iv, %load_tok, %true : i32, !ttg.async.token, i1
      } else {
        // CHECK: scf.yield {{.*}}[[CARRY_VALUE]], [[USE_ACC]], [[CARRY_TOKEN]] : i32, i1, !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %carry, %mma, %use_acc : i32, !ttg.async.token, i1
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1>, array<i32: 1>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 0, 1>]}
      %next = arith.addi %epilogue#0, %c1_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.yield {{.*}}[[BRANCH]]#1, %{{[-A-Za-z0-9_.$#]+}}, [[BRANCH]]#2 : i1, i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %epilogue#2, %epilogue#1, %next : i1, !ttg.async.token, i32
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @guarded_tokenless_if_deferred_initial_acquire
  tt.func @guarded_tokenless_if_deferred_initial_acquire(%lhs: !ttg.memdesc<128x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>, %guard: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false

    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    scf.if %guard {
      // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
      // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %loop = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%tok = %acc_tok) -> (!ttg.async.token) : i32 {
        // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V5]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V7]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %value, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %load_tok : !ttg.async.token
      // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
      } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
      scf.yield
    }
    tt.return
  }
}

//--- insert_semas_descriptor_store_completion.mlir

// TMA lowering does not propagate partition attrs to its new helper ops, so
// this synthetic pre-partition fixture disables per-pass verification only
// for the cross-pass order check. The production pipeline partitions first.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @direct_descriptor_store_completion
  // LOWER-LABEL: tt.func @direct_descriptor_store_completion
  // LOWER: ttng.async_tma_copy_local_to_global
  // LOWER-NEXT: ttng.async_tma_store_wait
  // LOWER: ttng.arrive_barrier
  tt.func @direct_descriptor_store_completion(%desc: !tt.tensordesc<128x64xf16, #shared>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    // SEMA: [[V1:%.*]] = ttg.local_alloc {buffer.id = 600 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // SEMA: [[EMPTY:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32}
    // SEMA-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32}
    %alloc = ttg.local_alloc {buffer.id = 600 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // SEMA-NEXT: [[LOOP:%.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[WRITE:%.*]] = [[ENTRY]]) -> (!ttg.async.token)  : i32 {
    scf.for %iv = %lb to %ub step %step : i32 {
      %first = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #blocked>
      // SEMA: [[WRITE_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[WRITE_BUF]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: nvws.semaphore.release [[FULL]], [[WRITE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %first, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: [[READ:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[LOADED:%.*]] = ttg.local_load [[READ_BUF]] {ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // The consumer release comes AFTER the descriptor store it must cover.
      // SEMA-NEXT: tt.descriptor_store %{{.*}}[%{{.*}}, %{{.*}}], [[LOADED]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: nvws.semaphore.release [[EMPTY]], [[READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      tt.descriptor_store %desc[%i, %i], %loaded {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked>
      %next = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #blocked>
      // SEMA: [[NEXT_WRITE:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: [[NEXT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[NEXT_WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[NEXT_BUF]] {ttg.partition = array<i32: 0>}
      ttg.local_store %next, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[NEXT_WRITE]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    // SEMA: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @already_lowered_tma_store_handoffs
  tt.func @already_lowered_tma_store_handoffs(
      %desc: !tt.tensordesc<128x64xf32, #shared>,
      %lb: i32, %ub: i32, %step: i32) {
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x64xf32, #blocked>
    // These exact-alias members model two consecutive output slices in one
    // depth-2 physical staging allocation.
    // SEMA: [[BASE:%.*]] = ttg.local_alloc
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = 3
    // SEMA-NEXT: [[COPY_READY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]]
    // SEMA-NEXT: [[M1_READY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]]
    // SEMA-NEXT: [[REDUCE_READY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]]
    %m0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 602 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %m1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 602 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // SEMA: [[ZERO_P0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // SEMA-NEXT: [[W0_TOKEN:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[ZERO_P0]]]
      // SEMA-NEXT: [[W0_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]], [[W0_TOKEN]]
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[W0_BUFFER]]#0
      // SEMA-NEXT: nvws.semaphore.release [[COPY_READY]][[[ZERO_P0]]], [[W0_TOKEN]]
      ttg.local_store %v0, %m0 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      // The TMA copy is a read of slot 0. Its release must stay after the
      // token wait and hand the next writer slot 1.
      // SEMA-NEXT: [[ZERO_P1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // SEMA-NEXT: [[COPY_TOKEN:%.*]] = nvws.semaphore.acquire [[COPY_READY]][[[ZERO_P1]]]
      // SEMA-NEXT: [[COPY_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[COPY_READY]], [[COPY_TOKEN]]
      // SEMA-NEXT: [[COPY:%.*]] = ttng.async_tma_copy_local_to_global %{{.*}} [[COPY_BUFFER]]#0
      // SEMA-NEXT: ttng.async_tma_store_token_wait [[COPY]]
      // SEMA-NEXT: [[TO_M1:%.*]] = arith.constant {{.*}} 1 : i32
      // SEMA-NEXT: nvws.semaphore.release [[M1_READY]][[[TO_M1]]], [[COPY_TOKEN]]
      %copy = ttng.async_tma_copy_local_to_global %desc[%i, %i] %m0 {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %copy {ttg.partition = array<i32: 1>} : !ttg.async.token
      // SEMA-NEXT: [[W1_TOKEN:%.*]] = nvws.semaphore.acquire [[M1_READY]][[[ZERO_P0]]]
      // SEMA-NEXT: [[W1_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[M1_READY]], [[W1_TOKEN]]
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[W1_BUFFER]]#1
      // SEMA-NEXT: nvws.semaphore.release [[REDUCE_READY]][[[ZERO_P0]]], [[W1_TOKEN]]
      ttg.local_store %v1, %m1 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      // Async reduce has the same SMEM-read lifetime and must release only
      // after its completion wait, back to slot 0 of the next iteration.
      // SEMA-NEXT: [[REDUCE_TOKEN:%.*]] = nvws.semaphore.acquire [[REDUCE_READY]][[[ZERO_P1]]]
      // SEMA-NEXT: [[REDUCE_BUFFER:%.*]]:2 = nvws.semaphore.buffer [[REDUCE_READY]], [[REDUCE_TOKEN]]
      // SEMA-NEXT: [[REDUCE:%.*]] = ttng.async_tma_reduce add, %{{.*}} [[REDUCE_BUFFER]]#1
      // SEMA-NEXT: ttng.async_tma_store_token_wait [[REDUCE]]
      // SEMA-NEXT: nvws.semaphore.release [[ENTRY]][[[TO_M1]]], [[REDUCE_TOKEN]]
      %reduce = ttng.async_tma_reduce add, %desc[%i, %i] %m1 {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %reduce {ttg.partition = array<i32: 1>} : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @converted_descriptor_store_completion
  tt.func @converted_descriptor_store_completion(%desc: !tt.tensordesc<128x64xf16, #shared>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    // SEMA: [[V1:%.*]] = ttg.local_alloc {buffer.id = 601 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // SEMA: [[EMPTY:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32}
    // SEMA-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32}
    %alloc = ttg.local_alloc {buffer.id = 601 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // SEMA-NEXT: [[LOOP:%.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[WRITE:%.*]] = [[ENTRY]]) -> (!ttg.async.token)  : i32 {
    scf.for %iv = %lb to %ub step %step : i32 {
      %first = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #linear>
      // SEMA: [[WRITE_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[WRITE_BUF]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: nvws.semaphore.release [[FULL]], [[WRITE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %first, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: [[READ:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: [[LOADED:%.*]] = ttg.local_load [[READ_BUF]] {ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #linear>
      // SEMA-NEXT: [[CONVERTED:%.*]] = ttg.convert_layout [[LOADED]] {ttg.partition = array<i32: 1>}
      %converted = ttg.convert_layout %loaded {ttg.partition = array<i32: 1>} : tensor<128x64xf16, #linear> -> tensor<128x64xf16, #blocked>
      // The consumer release comes AFTER the descriptor store even with the
      // intervening layout conversion between the load and the store.
      // SEMA-NEXT: tt.descriptor_store %{{.*}}[%{{.*}}, %{{.*}}], [[CONVERTED]] {ttg.partition = array<i32: 1>}
      // SEMA-NEXT: nvws.semaphore.release [[EMPTY]], [[READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      tt.descriptor_store %desc[%i, %i], %converted {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked>
      %next = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #linear>
      // SEMA: [[NEXT_WRITE:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: [[NEXT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[NEXT_WRITE]] {ttg.partition = array<i32: 0>}
      // SEMA-NEXT: ttg.local_store %{{.*}}, [[NEXT_BUF]] {ttg.partition = array<i32: 0>}
      ttg.local_store %next, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[NEXT_WRITE]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    // SEMA: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_direct_builder_composition.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @branch_pass_with_outstanding_completion
  tt.func @branch_pass_with_outstanding_completion(
      %cond: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[PASS_BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10100 : i32}
    // CHECK: [[LOOP_SEMA:%[0-9]+]] = nvws.semaphore.create [[PASS_BASE]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[TO_0:%[0-9]+]] = nvws.semaphore.create [[PASS_BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[TO_1_THEN:%[0-9]+]] = nvws.semaphore.create [[PASS_BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[TO_1_ELSE:%[0-9]+]] = nvws.semaphore.create [[PASS_BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[THEN_BACK:%[0-9]+]] = nvws.semaphore.create [[PASS_BASE]] {pending_count = 2 : i32}
    // CHECK-NEXT: [[ELSE_BACK:%[0-9]+]] = nvws.semaphore.create [[PASS_BASE]] {pending_count = 2 : i32}
    %buf = ttg.local_alloc {buffer.id = 10100 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %x = "value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
    // CHECK: [[W1_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[LOOP_SEMA]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[W1_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[LOOP_SEMA]], [[W1_TOKEN]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[W1_BUFFER]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.semaphore.release [[TO_0]], [[W1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
    ttg.local_store %x, %buf {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[R0_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_0]]
    // CHECK-NEXT: [[R0_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_0]], [[R0_TOKEN]]
    // CHECK-NEXT: ttg.local_load [[R0_BUFFER]]
    %r0 = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "use0"(%r0) {ttg.partition = array<i32: 0>} : (!one) -> ()
    scf.if %cond {
      // CHECK: [[IF_TOKEN:%[0-9]+]] = scf.if
      // CHECK: nvws.semaphore.release [[TO_1_THEN]], [[R0_TOKEN]] [#nvws.async_op<none>]
      // CHECK-NEXT: [[R1_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_1_THEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[R1_THEN_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_1_THEN]], [[R1_THEN_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_load [[R1_THEN_BUFFER]] {ttg.partition = array<i32: 1>}
      %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "then1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
      // CHECK: nvws.semaphore.release [[THEN_BACK]], [[R1_THEN_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[R0_THEN_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_0]], [[R0_TOKEN]]
      // CHECK-NEXT: ttg.local_load [[R0_THEN_BUFFER]]
      // CHECK-NEXT: nvws.semaphore.release [[THEN_BACK]], [[R0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %r0b = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "then0"(%r0b) {ttg.partition = array<i32: 0>} : (!one) -> ()
      // CHECK: [[THEN_RETURN:%[0-9]+]] = nvws.semaphore.acquire [[THEN_BACK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: scf.yield {{.*}}[[THEN_RETURN]] : !ttg.async.token
    } else {
      // CHECK: } else {
      // CHECK: nvws.semaphore.release [[TO_1_ELSE]], [[R0_TOKEN]] [#nvws.async_op<none>]
      // CHECK-NEXT: [[R1_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_1_ELSE]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[R1_ELSE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_1_ELSE]], [[R1_ELSE_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_load [[R1_ELSE_BUFFER]] {ttg.partition = array<i32: 1>}
      %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "else1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
      // CHECK: nvws.semaphore.release [[ELSE_BACK]], [[R1_ELSE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[R0_ELSE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_0]], [[R0_TOKEN]]
      // CHECK-NEXT: ttg.local_load [[R0_ELSE_BUFFER]]
      // CHECK-NEXT: nvws.semaphore.release [[ELSE_BACK]], [[R0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %r0b = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "else0"(%r0b) {ttg.partition = array<i32: 0>} : (!one) -> ()
      // CHECK: [[ELSE_RETURN:%[0-9]+]] = nvws.semaphore.acquire [[ELSE_BACK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: scf.yield {{.*}}[[ELSE_RETURN]] : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1>}
    // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
    // CHECK-NEXT: nvws.semaphore.release [[LOOP_SEMA]], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK-NOT: nvws.semaphore.acquire [[LOOP_SEMA]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @conditional_fanout_restores_boundary_owner
  tt.func @conditional_fanout_restores_boundary_owner(
      %cond: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10105 : i32}
    // CHECK-NEXT: [[ENTRY:%[0-9]+]] = nvws.semaphore.create [[BASE]] released = 1 {pending_count = 2 : i32}
    // CHECK-NEXT: [[THEN_TO_ONE:%[0-9]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[THEN_TO_TWO:%[0-9]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[ELSE_TO_ONE:%[0-9]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[ELSE_RETURN:%[0-9]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32}
    %buf = ttg.local_alloc {buffer.id = 10105 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %x = "value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[START:%[0-9]+]] = nvws.semaphore.acquire [[ENTRY]]
      // CHECK-NEXT: scf.for {{.*}} iter_args([[CARRY:%[-A-Za-z0-9_.$#]+]] = [[START]]) -> (!ttg.async.token)  : i32 {
      // CHECK-NEXT: [[FIRST_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[ENTRY]], [[CARRY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[FIRST_BUFFER]] {ttg.partition = array<i32: 0>}
      ttg.local_store %x, %buf {ttg.partition = array<i32: 0>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      // CHECK: [[IF_TOKEN:%[0-9]+]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
      scf.if %cond {
        // CHECK-NEXT: nvws.semaphore.release [[THEN_TO_TWO]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        // CHECK-NEXT: nvws.semaphore.release [[THEN_TO_ONE]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        // CHECK-NEXT: [[THEN_ONE:%[0-9]+]] = nvws.semaphore.acquire [[THEN_TO_ONE]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[THEN_ONE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[THEN_TO_ONE]], [[THEN_ONE]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: ttg.local_load [[THEN_ONE_BUFFER]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[ENTRY]], [[THEN_ONE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
        "then1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
        // CHECK: [[THEN_TWO:%[0-9]+]] = nvws.semaphore.acquire [[THEN_TO_TWO]] {ttg.partition = array<i32: 2>}
        // CHECK-NEXT: [[THEN_TWO_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[THEN_TO_TWO]], [[THEN_TWO]] {ttg.partition = array<i32: 2>}
        // CHECK-NEXT: ttg.local_load [[THEN_TWO_BUFFER]] {ttg.partition = array<i32: 2>}
        // CHECK-NEXT: nvws.semaphore.release [[ENTRY]], [[THEN_TWO]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
        %r2 = ttg.local_load %buf {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
        "then2"(%r2) {ttg.partition = array<i32: 2>} : (!one) -> ()
        // CHECK: [[THEN_BACK:%[0-9]+]] = nvws.semaphore.acquire [[ENTRY]] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: scf.yield {{.*}}[[THEN_BACK]] : !ttg.async.token
      } else {
        // CHECK: } else {
        // CHECK-NEXT: nvws.semaphore.release [[ELSE_TO_ONE]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        // CHECK-NEXT: [[ELSE_ONE:%[0-9]+]] = nvws.semaphore.acquire [[ELSE_TO_ONE]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[ELSE_ONE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[ELSE_TO_ONE]], [[ELSE_ONE]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: ttg.local_load [[ELSE_ONE_BUFFER]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[ELSE_RETURN]], [[ELSE_ONE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
        "else1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
        // CHECK: [[ELSE_BACK:%[0-9]+]] = nvws.semaphore.acquire [[ELSE_RETURN]] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: scf.yield {{.*}}[[ELSE_BACK]] : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>]}
      // CHECK-NEXT: [[LAST_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[ENTRY]], [[IF_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[LAST_BUFFER]] {ttg.partition = array<i32: 0>}
      ttg.local_store %x, %buf {ttg.partition = array<i32: 0>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @nested_same_owner_reuses_outer_token
  tt.func @nested_same_owner_reuses_outer_token(
      %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[REUSE_BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10101 : i32}
    // CHECK: [[OUTER_ENTRY:%[0-9]+]] = nvws.semaphore.create [[REUSE_BASE]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[TO_ZERO:%[0-9]+]] = nvws.semaphore.create [[REUSE_BASE]] {pending_count = 1 : i32}
    // CHECK-NOT: nvws.semaphore.create
    %buf = ttg.local_alloc {buffer.id = 10101 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %x = "value"() {ttg.partition = array<i32: 1>} : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[W_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[OUTER_ENTRY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[W_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[OUTER_ENTRY]], [[W_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[W_BUFFER]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[TO_ZERO]], [[W_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %x, %buf {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[ZERO_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_ZERO]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[ZERO_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_ZERO]], [[ZERO_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_load [[ZERO_BUFFER]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[OUTER_ENTRY]], [[ZERO_TOKEN]] [#nvws.async_op<none>]
      %r0 = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "use0"(%r0) {ttg.partition = array<i32: 0>} : (!one) -> ()
      scf.for %j = %lb to %ub step %step : i32 {
        // CHECK: scf.for
        // CHECK-NEXT: [[INNER_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[OUTER_ENTRY]], [[W_TOKEN]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: ttg.local_load [[INNER_BUFFER]] {ttg.partition = array<i32: 1>}
        %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
        "use1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
        // CHECK-NOT: nvws.semaphore.acquire
        // CHECK-NOT: nvws.semaphore.release
      // CHECK: } {ttg.partition = array<i32: 1>}
      } {ttg.partition = array<i32: 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @conditional_completion_is_later_edge_source
  tt.func @conditional_completion_is_later_edge_source(
      %cond: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[SOURCE_BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10102 : i32}
    // CHECK: [[SOURCE_ENTRY:%[0-9]+]] = nvws.semaphore.create [[SOURCE_BASE]] released = 1
    %buf = ttg.local_alloc {buffer.id = 10102 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %x = "value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
    // CHECK: [[SOURCE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[SOURCE_ENTRY]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[SOURCE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[SOURCE_ENTRY]], [[SOURCE_TOKEN]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[SOURCE_BUFFER]] {ttg.partition = array<i32: 1>}
    ttg.local_store %x, %buf {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.if %cond {
      // CHECK: [[IF_TOKEN:%[0-9]+]] = scf.if
      // CHECK: nvws.semaphore.release [[TO_ZERO:%[0-9]+]], [[SOURCE_TOKEN]] [#nvws.async_op<none>]
      // CHECK-NEXT: [[ZERO_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_ZERO]]
      // CHECK-NEXT: [[ZERO_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_ZERO]], [[ZERO_TOKEN]]
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[ZERO_BUFFER]]
      // CHECK-NEXT: nvws.semaphore.release [[JOIN:%[0-9]+]], [[ZERO_TOKEN]] [#nvws.async_op<none>]
      ttg.local_store %x, %buf {ttg.partition = array<i32: 0>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[THEN_RETURN:%[0-9]+]] = nvws.semaphore.acquire [[JOIN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.yield {{.*}}[[THEN_RETURN]] : !ttg.async.token
    } else {
      // CHECK: } else {
      // CHECK-NEXT: scf.yield {{.*}}[[SOURCE_TOKEN]] : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // CHECK-NEXT: [[R1_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[SOURCE_ENTRY]], [[IF_TOKEN]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_load [[R1_BUFFER]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.semaphore.release [[TO_TWO:%[0-9]+]], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
    %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "use1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
    // CHECK: [[R2_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_TWO]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: [[R2_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_TWO]], [[R2_TOKEN]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: ttg.local_load [[R2_BUFFER]] {ttg.partition = array<i32: 2>}
    %r2 = ttg.local_load %buf {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "use2"(%r2) {ttg.partition = array<i32: 2>} : (!one) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
    // CHECK-LABEL: @conditional_alternatives_normalize_counts
  tt.func @conditional_alternatives_normalize_counts(
      %cond: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[COUNT_BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10103 : i32}
    // CHECK: [[COUNT_ENTRY:%[0-9]+]] = nvws.semaphore.create [[COUNT_BASE]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[COUNT_TO_ZERO:%[0-9]+]] = nvws.semaphore.create [[COUNT_BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[THEN_TO_ONE:%[0-9]+]] = nvws.semaphore.create [[COUNT_BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[THEN_TO_TWO:%[0-9]+]] = nvws.semaphore.create [[COUNT_BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[ELSE_TO_ONE:%[0-9]+]] = nvws.semaphore.create [[COUNT_BASE]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[THEN_BACK:%[0-9]+]] = nvws.semaphore.create [[COUNT_BASE]] {pending_count = 2 : i32}
    // CHECK-NEXT: [[ELSE_BACK:%[0-9]+]] = nvws.semaphore.create [[COUNT_BASE]] {pending_count = 1 : i32}
    %buf = ttg.local_alloc {buffer.id = 10103 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %x = "value"() : () -> !one
    // CHECK: [[COUNT_INITIAL:%[0-9]+]] = nvws.semaphore.acquire [[COUNT_ENTRY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: scf.for {{.*}} iter_args([[COUNT_CARRY:%[-A-Za-z0-9_.$#]+]] = [[COUNT_INITIAL]]) -> (!ttg.async.token) : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
    // CHECK: [[COUNT_WRITE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[COUNT_ENTRY]], [[COUNT_CARRY]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[COUNT_WRITE_BUFFER]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.semaphore.release [[COUNT_TO_ZERO]], [[COUNT_CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
    ttg.local_store %x, %buf {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[COUNT_ZERO_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[COUNT_TO_ZERO]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: [[COUNT_ZERO_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[COUNT_TO_ZERO]], [[COUNT_ZERO_TOKEN]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttg.local_load [[COUNT_ZERO_BUFFER]] {ttg.partition = array<i32: 0>}
    %r0 = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "use0"(%r0) {ttg.partition = array<i32: 0>} : (!one) -> ()
    scf.if %cond {
      // CHECK: [[IF_TOKEN:%[0-9]+]] = scf.if
      // CHECK-NEXT: nvws.semaphore.release [[THEN_TO_TWO]], [[COUNT_ZERO_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[THEN_TO_ONE]], [[COUNT_ZERO_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[THEN_ONE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[THEN_TO_ONE]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[THEN_ONE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[THEN_TO_ONE]], [[THEN_ONE_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_load [[THEN_ONE_BUFFER]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[THEN_BACK]], [[THEN_ONE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "then1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
      // CHECK: [[THEN_TWO_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[THEN_TO_TWO]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[THEN_TWO_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[THEN_TO_TWO]], [[THEN_TWO_TOKEN]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttg.local_load [[THEN_TWO_BUFFER]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[THEN_BACK]], [[THEN_TWO_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %r2 = ttg.local_load %buf {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "then2"(%r2) {ttg.partition = array<i32: 2>} : (!one) -> ()
      // CHECK: [[THEN_RETURN:%[0-9]+]] = nvws.semaphore.acquire [[THEN_BACK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: scf.yield {{.*}}[[THEN_RETURN]] : !ttg.async.token
    } else {
      %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      // CHECK: } else {
      // CHECK-NEXT: nvws.semaphore.release [[ELSE_TO_ONE]], [[COUNT_ZERO_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[ELSE_ONE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[ELSE_TO_ONE]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[ELSE_ONE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[ELSE_TO_ONE]], [[ELSE_ONE_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_load [[ELSE_ONE_BUFFER]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[ELSE_BACK]], [[ELSE_ONE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      "else1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
      // CHECK: [[ELSE_RETURN:%[0-9]+]] = nvws.semaphore.acquire [[ELSE_BACK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: scf.yield {{.*}}[[ELSE_RETURN]] : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>]}
    // CHECK-NEXT: nvws.semaphore.release [[COUNT_ENTRY]], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: [[JOIN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[COUNT_ENTRY]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[JOIN_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[COUNT_ENTRY]], [[JOIN_TOKEN]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_load [[JOIN_BUFFER]] {ttg.partition = array<i32: 1>}
    %r1c = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "continue1"(%r1c) {ttg.partition = array<i32: 1>} : (!one) -> ()
    // CHECK-NEXT: "continue1"
    // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[JOIN_TOKEN]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @mixed_owner_conditional_joins_at_whole_write
  tt.func @mixed_owner_conditional_joins_at_whole_write(
      %cond: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[MIXED_WHOLE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[MIXED_LEFT:%[0-9]+]] = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[MIXED_RIGHT:%[0-9]+]] = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 1 : i32}
    // CHECK-NEXT: [[MIXED_ENTRY_0:%[0-9]+]] = nvws.semaphore.create [[MIXED_WHOLE]], [[MIXED_LEFT]], [[MIXED_RIGHT]] released = 1
    // CHECK-NEXT: [[MIXED_ENTRY_1:%[0-9]+]] = nvws.semaphore.create [[MIXED_WHOLE]], [[MIXED_LEFT]], [[MIXED_RIGHT]] released = 1
    // CHECK-NEXT: [[MIXED_JOIN:%[0-9]+]] = nvws.semaphore.create [[MIXED_WHOLE]], [[MIXED_LEFT]], [[MIXED_RIGHT]] {pending_count = 2 : i32}
    %whole = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %one = "one"() : () -> !one
    %two = "two"() : () -> !two
    scf.for %i = %lb to %ub step %step : i32 {
    scf.if %cond {
      // CHECK: scf.if
      // CHECK: [[LEFT_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_ENTRY_0]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LEFT_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_ENTRY_0]], [[LEFT_THEN_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[LEFT_THEN_BUFFER]]#1 {ttg.partition = array<i32: 1>}
      ttg.local_store %one, %left {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[RIGHT_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_ENTRY_1]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[RIGHT_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_ENTRY_1]], [[RIGHT_THEN_TOKEN]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[RIGHT_THEN_BUFFER]]#2 {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[MIXED_JOIN]], [[RIGHT_THEN_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[MIXED_JOIN]], [[LEFT_THEN_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %one, %right {ttg.partition = array<i32: 2>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } else {
      // CHECK: } else {
      // CHECK: [[LEFT_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_ENTRY_0]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LEFT_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_ENTRY_0]], [[LEFT_ELSE_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[LEFT_ELSE_BUFFER]]#1 {ttg.partition = array<i32: 1>}
      ttg.local_store %one, %left {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[RIGHT_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_ENTRY_1]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[RIGHT_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_ENTRY_1]], [[RIGHT_ELSE_TOKEN]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[RIGHT_ELSE_BUFFER]]#2 {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[MIXED_JOIN]], [[RIGHT_ELSE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[MIXED_JOIN]], [[LEFT_ELSE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %one, %right {ttg.partition = array<i32: 2>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } {ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: [[WHOLE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_JOIN]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: [[WHOLE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_JOIN]], [[WHOLE_TOKEN]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[WHOLE_BUFFER]]#0 {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.semaphore.release [[MIXED_ENTRY_0]], [[WHOLE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.semaphore.release [[MIXED_ENTRY_1]], [[WHOLE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
    ttg.local_store %two, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_function_cfg.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @managed_group_in_non_entry_block
  tt.func @managed_group_in_non_entry_block(
      %early: i1, %lb: i32, %ub: i32, %step: i32) {
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    // CHECK: ^bb2:
    // CHECK: [[BACKING:%.*]] = ttg.local_alloc {buffer.id = 1200 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BACKING]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BACKING]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 1200 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[WRITE_TOKEN:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[WRITE_BUFFER:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WRITE_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[WRITE_BUFFER]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]], [[WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[READ_TOKEN:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[READ_BUFFER:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[LOADED:%.*]] = ttg.local_load [[READ_BUFFER]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[EMPTY]], [[READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume"([[LOADED]]) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()
      "consume"(%loaded) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_function_cfg_errors.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reject_managed_flow_across_cfg_blocks(
      %early: i1, %lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 1201 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: error: nvws-insert-semas: managed memdesc flow across function CFG blocks is unsupported
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_fused_alias_handoff.mlir

// Two exact-alias epilogue members share one depth-2 physical allocation.
// The first member uses slot 0 and the second uses slot 1, so each
// read-to-next-write release must target the successor slot rather than the
// source read's slot.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @fused_alias_depth_two
  // ASP-LABEL: @fused_alias_depth_two
  tt.func @fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    // Both member allocs collapse onto one fused depth-2 backing allocation;
    // every semaphore lists both (identical) member views as its buffers.
    // SEMA: [[BASE:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // SEMA: [[FULL0:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // SEMA: [[EMPTY1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // SEMA: [[FULL1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    // ASP: [[BASE:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32}
    // ASP: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = 3 {pending_count = 1 : i32}
    // ASP: [[FULL0:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32}
    // ASP: [[EMPTY1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32}
    // ASP: [[FULL1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32}
    %m0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %m1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>

    // The loop-close release partition (2) differs from the first-acquire
    // partition (4), so no acquire token is threaded through iter_args at the
    // SEMA stage; ASP threads the slot cursor plus one phase word per
    // acquirer.
    // SEMA: scf.for
    // ASP: scf.for {{.*}} iter_args([[CURSOR:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_R0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_R1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_W0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_W1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}})
    scf.for %iv = %lb to %ub step %step : i32 {
      // Member 0 write: acquire ENTRY, store through view #0, release FULL0.
      // SEMA: [[W0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[W0_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[W0_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[W0_BUF:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]], [[W0_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[W0_BUF]]#0 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // SEMA: [[W0_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[FULL0]][[[W0_REL_ZERO]]], [[W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[SLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: arith.shli {{%.*}}, [[SLOT0]] {ttg.partition = array<i32: 4>} : i32
      // ASP: [[PHN_W0:%.*]] = arith.xori [[PH_W0]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[W0_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[SLOT0]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[W0_BUF:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]][[[SLOT0]]], [[W0_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttg.local_store {{%.*}}, [[W0_BUF]]#0 {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[FULL0]][[[SLOT0]]], [[W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %v0, %m0 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Member 0 read: acquire FULL0, load view #0, release EMPTY1 at the
      // successor slot (SLOT0 + 1).
      // SEMA: [[R0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[R0_TOK:%.*]] = nvws.semaphore.acquire [[FULL0]][[[R0_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[R0_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL0]], [[R0_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      // SEMA: ttg.local_load [[R0_BUF]]#0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // SEMA: [[TO_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[EMPTY1]][[[TO_M1]]], [[R0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PHN_R0:%.*]] = arith.xori [[PH_R0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[R0_TOK:%.*]] = nvws.semaphore.acquire [[FULL0]][[[SLOT0]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[R0_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL0]][[[SLOT0]]], [[R0_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttg.local_load [[R0_BUF]]#0 {ttg.partition = array<i32: 2>}
      // ASP: [[TO_M1_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_M1_REM:%.*]] = arith.remsi [[TO_M1_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_M1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M1_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[EMPTY1]][[[TO_M1]]], [[R0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %r0 = ttg.local_load %m0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume0"(%r0) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
      // Member 1 write: acquire EMPTY1, store through view #1, release FULL1.
      // SEMA: [[W1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[W1_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY1]][[[W1_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[W1_BUF:%.*]]:2 = nvws.semaphore.buffer [[EMPTY1]], [[W1_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[W1_BUF]]#1 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // SEMA: [[W1_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[FULL1]][[[W1_REL_ZERO]]], [[W1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[NEXT_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[SLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[NEXT_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: arith.shli {{%.*}}, [[SLOT1]] {ttg.partition = array<i32: 4>} : i32
      // ASP: [[PHN_W1:%.*]] = arith.xori [[PH_W1]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[W1_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY1]][[[SLOT1]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[W1_BUF:%.*]]:2 = nvws.semaphore.buffer [[EMPTY1]][[[SLOT1]]], [[W1_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttg.local_store {{%.*}}, [[W1_BUF]]#1 {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[FULL1]][[[SLOT1]]], [[W1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %v1, %m1 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Member 1 read: acquire FULL1, load view #1, close the loop by
      // releasing ENTRY at the successor slot (SLOT1 + 1) mod 2.
      // SEMA: [[R1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[R1_TOK:%.*]] = nvws.semaphore.acquire [[FULL1]][[[R1_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[R1_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL1]], [[R1_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_load [[R1_BUF]]#1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // SEMA: [[TO_NEXT_M0:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[ENTRY]][[[TO_NEXT_M0]]], [[R1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PHN_R1:%.*]] = arith.xori [[PH_R1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[R1_TOK:%.*]] = nvws.semaphore.acquire [[FULL1]][[[SLOT1]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[R1_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL1]][[[SLOT1]]], [[R1_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttg.local_load [[R1_BUF]]#1 {ttg.partition = array<i32: 2>}
      // ASP: [[TO_M0_RAW:%.*]] = arith.addi [[SLOT1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_M0_REM:%.*]] = arith.remsi [[TO_M0_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_M0:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M0_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[ENTRY]][[[TO_M0]]], [[R1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // ASP: scf.yield {ttg.partition = array<i32: 2, 4>} [[SLOT1]], [[PHN_R0]], [[PHN_R1]], [[PHN_W0]], [[PHN_W1]] : i32, i32, i32, i32, i32
      %r1 = ttg.local_load %m1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume1"(%r1) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Planner-authored aliases may be different views of one staged backing.
  // Here the smaller member covers the prefix of the larger member.  The
  // read-to-next-write handoff must still target the following physical slot.
  // SEMA-LABEL: @fused_partial_alias_depth_three
  // ASP-LABEL: @fused_partial_alias_depth_three
  tt.func @fused_partial_alias_depth_three(%lb: i32, %ub: i32, %step: i32) {
    // Entry stages 0 and 2 are acquired before their first release; stage 1
    // is released before its first acquire, so the bootstrap mask is 0b101.
    // SEMA: [[PLARGE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>
    // SEMA: [[PSMALL:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    // SEMA: [[PENTRY:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] released = 5 {pending_count = 1 : i32} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // SEMA: [[PFULL0:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // SEMA: [[PHANDOFF:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // SEMA: [[PFULL1:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // ASP: [[PLARGE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32}
    // ASP: [[PSMALL:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32}
    // ASP: [[PENTRY:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] released = 5 {pending_count = 1 : i32}
    // ASP: [[PFULL0:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32}
    // ASP: [[PHANDOFF:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32}
    // ASP: [[PFULL1:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32}
    %large = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %small = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %small_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %large_value = arith.constant dense<1.000000e+00> : tensor<256x64xf16, #blocked64>

    // SEMA: scf.for
    // ASP: scf.for {{.*}} iter_args([[PCURSOR:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_R0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_R1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_W0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_W1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}})
    scf.for %iv = %lb to %ub step %step : i32 {
      // Small-member write: acquire PENTRY, store through view #1 (the small
      // member), release PFULL0.
      // SEMA: [[PW0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[PW0_TOK:%.*]] = nvws.semaphore.acquire [[PENTRY]][[[PW0_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PENTRY]], [[PW0_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[PW0_BUF]]#1 {ttg.partition = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: [[PW0_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[PFULL0]][[[PW0_REL_ZERO]]], [[PW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PSLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[PPHN_W0:%.*]] = arith.xori [[PPH_W0]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[PW0_TOK:%.*]] = nvws.semaphore.acquire [[PENTRY]][[[PSLOT0]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[PW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PENTRY]][[[PSLOT0]]], [[PW0_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttg.local_store {{%.*}}, [[PW0_BUF]]#1 {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[PFULL0]][[[PSLOT0]]], [[PW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %small_value, %small {ttg.partition = array<i32: 4>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // Small-member read: acquire PFULL0, load view #1, then hand off to the
      // large write at the following physical slot (PSLOT0 + 1) mod 3.
      // SEMA: [[PR0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[PR0_TOK:%.*]] = nvws.semaphore.acquire [[PFULL0]][[[PR0_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL0]], [[PR0_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_load [[PR0_BUF]]#1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked1>
      // SEMA: [[TO_LARGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[PHANDOFF]][[[TO_LARGE]]], [[PR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PPHN_R0:%.*]] = arith.xori [[PPH_R0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[PR0_TOK:%.*]] = nvws.semaphore.acquire [[PFULL0]][[[PSLOT0]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[PR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL0]][[[PSLOT0]]], [[PR0_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttg.local_load [[PR0_BUF]]#1 {ttg.partition = array<i32: 2>}
      // ASP: [[TO_LARGE_RAW:%.*]] = arith.addi [[PSLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_LARGE_REM:%.*]] = arith.remsi [[TO_LARGE_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TO_LARGE_SLOT:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_LARGE_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[PHANDOFF]][[[TO_LARGE_SLOT]]], [[PR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %small_read = ttg.local_load %small {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_small"(%small_read) {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked64>) -> ()
      // Large-member write: acquire PHANDOFF at the successor slot, store
      // through view #0 (the large member), release PFULL1.
      // SEMA: [[PW1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[PW1_TOK:%.*]] = nvws.semaphore.acquire [[PHANDOFF]][[[PW1_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PHANDOFF]], [[PW1_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
      // SEMA: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[PW1_BUF]]#0 {ttg.partition = array<i32: 4>} : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      // SEMA: [[PW1_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[PFULL1]][[[PW1_REL_ZERO]]], [[PW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PSLOT1_RAW:%.*]] = arith.addi [[PSLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[PSLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[PSLOT1_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[PPHN_W1:%.*]] = arith.xori [[PPH_W1]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[PW1_TOK:%.*]] = nvws.semaphore.acquire [[PHANDOFF]][[[PSLOT1]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[PW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PHANDOFF]][[[PSLOT1]]], [[PW1_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttg.local_store {{%.*}}, [[PW1_BUF]]#0 {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[PFULL1]][[[PSLOT1]]], [[PW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %large_value, %large {ttg.partition = array<i32: 4>} : tensor<256x64xf16, #blocked64> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      // Large-member read: acquire PFULL1, load view #0, close the loop by
      // releasing PENTRY at the reader's own slot (constant 0 / PSLOT1: the
      // slot the next small write reaches two iterations later).
      // SEMA: [[PR1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[PR1_TOK:%.*]] = nvws.semaphore.acquire [[PFULL1]][[[PR1_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // SEMA: [[PR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL1]], [[PR1_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
      // SEMA: ttg.local_load [[PR1_BUF]]#0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #blocked1>
      // SEMA: [[PBACK_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: nvws.semaphore.release [[PENTRY]][[[PBACK_ZERO]]], [[PR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // ASP: [[PPHN_R1:%.*]] = arith.xori [[PPH_R1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[PR1_TOK:%.*]] = nvws.semaphore.acquire [[PFULL1]][[[PSLOT1]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[PR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL1]][[[PSLOT1]]], [[PR1_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttg.local_load [[PR1_BUF]]#0 {ttg.partition = array<i32: 2>}
      // ASP: nvws.semaphore.release [[PENTRY]][[[PSLOT1]]], [[PR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // ASP: scf.yield {ttg.partition = array<i32: 2, 4>} [[PSLOT1]], [[PPHN_R0]], [[PPHN_R1]], [[PPHN_W0]], [[PPHN_W1]] : i32, i32, i32, i32, i32
      %large_read = ttg.local_load %large {ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #blocked64>
      "consume_large"(%large_read) {ttg.partition = array<i32: 2>} : (tensor<256x64xf16, #blocked64>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }

  // SEMA-LABEL: @tmem_fused_alias_depth_two
  // ASP-LABEL: @tmem_fused_alias_depth_two
  tt.func @tmem_fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>

    // The fused tmem allocation and its semaphores are hoisted to function
    // scope, ahead of the loop that contains the source tmem_allocs.
    // SEMA: [[TBASE:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // SEMA: [[TENTRY:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // SEMA: [[TFULL0:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // SEMA: [[TEMPTY1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // SEMA: [[TFULL1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // ASP: [[TBASE:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32}
    // ASP: [[TENTRY:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] released = 3 {pending_count = 1 : i32}
    // ASP: [[TFULL0:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32}
    // ASP: [[TEMPTY1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32}
    // ASP: [[TFULL1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32}
    // SEMA: scf.for
    // ASP: scf.for {{.*}} iter_args([[TCURSOR:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_R0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_R1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_W0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_W1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}})
    scf.for %iv = %lb to %ub step %step : i32 {
      // Member 0 write: the value-carrying tmem_alloc becomes a tmem_store
      // through view #0 with no token bracket.
      // SEMA: [[TW0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[TW0_TOK:%.*]] = nvws.semaphore.acquire [[TENTRY]][[[TW0_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // SEMA: [[TW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TENTRY]], [[TW0_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[TW0_BUF]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: [[TW0_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[TFULL0]][[[TW0_REL_ZERO]]], [[TW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // ASP: [[TSLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[TPHN_W0:%.*]] = arith.xori [[TPH_W0]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[TW0_TOK:%.*]] = nvws.semaphore.acquire [[TENTRY]][[[TSLOT0]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[TW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TENTRY]][[[TSLOT0]]], [[TW0_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttng.tmem_store {{%.*}}, [[TW0_BUF]]#0, {{%.*}} {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[TFULL0]][[[TSLOT0]]], [[TW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      %m0 = ttng.tmem_alloc %v0 {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      // Member 0 read: acquire TFULL0, load view #0 with an empty token
      // bracket, release TEMPTY1 at the successor slot.
      // SEMA: [[TR0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[TR0_TOK:%.*]] = nvws.semaphore.acquire [[TFULL0]][[[TR0_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // SEMA: [[TR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL0]], [[TR0_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[TR0_BUF]]#0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      // SEMA: [[T_TO_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[TEMPTY1]][[[T_TO_M1]]], [[TR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // ASP: [[TPHN_R0:%.*]] = arith.xori [[TPH_R0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TR0_TOK:%.*]] = nvws.semaphore.acquire [[TFULL0]][[[TSLOT0]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[TR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL0]][[[TSLOT0]]], [[TR0_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttng.tmem_load [[TR0_BUF]]#0[] {ttg.partition = array<i32: 2>}
      // ASP: [[T_TO_M1_RAW:%.*]] = arith.addi [[TSLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[T_TO_M1_REM:%.*]] = arith.remsi [[T_TO_M1_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[T_TO_M1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[T_TO_M1_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[TEMPTY1]][[[T_TO_M1]]], [[TR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %r0, %t0 = ttng.tmem_load %m0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "consume0"(%r0) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
      // Member 1 write: acquire TEMPTY1, store through view #1, release
      // TFULL1.
      // SEMA: [[TW1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: [[TW1_TOK:%.*]] = nvws.semaphore.acquire [[TEMPTY1]][[[TW1_ZERO]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // SEMA: [[TW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TEMPTY1]], [[TW1_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[TW1_BUF]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: [[TW1_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // SEMA: nvws.semaphore.release [[TFULL1]][[[TW1_REL_ZERO]]], [[TW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // ASP: [[TSLOT1_RAW:%.*]] = arith.addi [[TSLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[TSLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TSLOT1_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
      // ASP: [[TPHN_W1:%.*]] = arith.xori [[TPH_W1]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // ASP: [[TW1_TOK:%.*]] = nvws.semaphore.acquire [[TEMPTY1]][[[TSLOT1]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // ASP: [[TW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TEMPTY1]][[[TSLOT1]]], [[TW1_TOK]] {ttg.partition = array<i32: 4>}
      // ASP: ttng.tmem_store {{%.*}}, [[TW1_BUF]]#1, {{%.*}} {ttg.partition = array<i32: 4>}
      // ASP: nvws.semaphore.release [[TFULL1]][[[TSLOT1]]], [[TW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      %m1 = ttng.tmem_alloc %v1 {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      // Member 1 read: acquire TFULL1, load view #1, close the loop by
      // releasing TENTRY at the successor slot (TSLOT1 + 1) mod 2.
      // SEMA: [[TR1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // SEMA: [[TR1_TOK:%.*]] = nvws.semaphore.acquire [[TFULL1]][[[TR1_ZERO]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // SEMA: [[TR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL1]], [[TR1_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // SEMA: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[TR1_BUF]]#1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      // SEMA: [[T_TO_M0:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // SEMA: nvws.semaphore.release [[TENTRY]][[[T_TO_M0]]], [[TR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // ASP: [[TPHN_R1:%.*]] = arith.xori [[TPH_R1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[TR1_TOK:%.*]] = nvws.semaphore.acquire [[TFULL1]][[[TSLOT1]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // ASP: [[TR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL1]][[[TSLOT1]]], [[TR1_TOK]] {ttg.partition = array<i32: 2>}
      // ASP: ttng.tmem_load [[TR1_BUF]]#1[] {ttg.partition = array<i32: 2>}
      // ASP: [[T_TO_M0_RAW:%.*]] = arith.addi [[TSLOT1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[T_TO_M0_REM:%.*]] = arith.remsi [[T_TO_M0_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // ASP: [[T_TO_M0:%.*]] = arith.select {{%.*}}, {{%.*}}, [[T_TO_M0_REM]] {ttg.partition = array<i32: 2>} : i32
      // ASP: nvws.semaphore.release [[TENTRY]][[[T_TO_M0]]], [[TR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // ASP: scf.yield {ttg.partition = array<i32: 2, 4>} [[TSLOT1]], [[TPHN_R0]], [[TPHN_R1]], [[TPHN_W0]], [[TPHN_W1]] : i32, i32, i32, i32, i32
      %r1, %t1 = ttng.tmem_load %m1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "consume1"(%r1) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // A: both fresh-write epochs are inside one loop.
  // SEMA-LABEL: @case_a
  // ASP-LABEL: @case_a
  tt.func @case_a(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // SEMA: [[A_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // SEMA: [[A_ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]] released = 1 {pending_count = 1 : i32}
    // SEMA: [[A_NEXT:%.*]] = nvws.semaphore.create [[A_BASE]] released = 6 {pending_count = 1 : i32}
    // SEMA: [[A_INIT_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 10 : i32} :
    // ASP: [[A_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // ASP: [[A_ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]] released = 1 {pending_count = 1 : i32}
    // ASP: [[A_NEXT:%.*]] = nvws.semaphore.create [[A_BASE]] released = 6 {pending_count = 1 : i32}
    // ASP: [[A_INITIAL_CURRENT_STAGE:%.*]] = arith.constant 2 : i32
    // ASP: [[A_INITIAL_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} 1 : i32
    // ASP: [[A_INITIAL_NEXT_RAW:%.*]] = arith.addi [[A_INITIAL_CURRENT_STAGE]], [[A_INITIAL_ONE]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} : i32
    // ASP: [[A_INITIAL_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} 3 : i32
    // ASP: [[A_INITIAL_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[A_INITIAL_NEXT_RAW]], [[A_INITIAL_DEPTH]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} : i32
    // ASP: [[A_INITIAL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} 0 : i32
    // ASP: [[A_INITIAL_NEXT_STAGE:%.*]] = arith.select [[A_INITIAL_NEEDS_WRAP]], [[A_INITIAL_ZERO]], [[A_INITIAL_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} : i32
    // ASP: [[A_INIT_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]][[[A_INITIAL_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 10 : i32}
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: scf.for {{.*}} iter_args([[A_W0_TOK:%.*]] = [[A_INIT_TOK]])
    // ASP: scf.for {{.*}} iter_args([[A_W0_TOK:%.*]] = [[A_INIT_TOK]], [[A_CURRENT_STAGE:%.*]] = [[A_INITIAL_NEXT_STAGE]],
    %outer = scf.for %i = %lb to %ub step %step iter_args(%carry = %tok) -> (!ttg.async.token) : i32 {
      // SEMA: [[A_W0_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]], [[A_W0_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: ttng.tmem_store {{%.*}}, [[A_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // SEMA: nvws.semaphore.release [[A_NEXT]], [[A_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[A_W0_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]][[[A_CURRENT_STAGE]]], [[A_W0_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: ttng.tmem_store {{%.*}}, [[A_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[A_NEXT]][[[A_CURRENT_STAGE]]], [[A_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %w0 = ttng.tmem_store %zero, %acc[%carry], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: [[A_W1_TOK:%.*]] = nvws.semaphore.acquire [[A_NEXT]] {ttg.partition = array<i32: 1>}
      // SEMA: [[A_W1_BUF:%.*]] = nvws.semaphore.buffer [[A_NEXT]], [[A_W1_TOK]] {ttg.partition = array<i32: 1>}
      // SEMA: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[A_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // SEMA: nvws.semaphore.release [[A_ENTRY]], [[A_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[A_NEXT_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // ASP: [[A_NEXT_RAW:%.*]] = arith.addi [[A_CURRENT_STAGE]], [[A_NEXT_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[A_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // ASP: [[A_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[A_NEXT_RAW]], [[A_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[A_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // ASP: [[A_NEXT_STAGE:%.*]] = arith.select [[A_NEEDS_WRAP]], [[A_ZERO]], [[A_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[A_W1_TOK:%.*]] = nvws.semaphore.acquire [[A_NEXT]][[[A_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      // ASP: [[A_W1_BUF:%.*]] = nvws.semaphore.buffer [[A_NEXT]][[[A_NEXT_STAGE]]], [[A_W1_TOK]] {ttg.partition = array<i32: 1>}
      // ASP: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[A_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // ASP: nvws.semaphore.release [[A_ENTRY]][[[A_NEXT_STAGE]]], [[A_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%w0], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]] {ttg.partition = array<i32: 0>}
      // SEMA: [[A_READ_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]], [[A_READ_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: {{%.*}}, {{%.*}} = ttng.tmem_load [[A_READ_BUF]][] {ttg.partition = array<i32: 0>}
      // ASP: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]][[[A_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // ASP: [[A_READ_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]][[[A_NEXT_STAGE]]], [[A_READ_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: {{%.*}}, {{%.*}} = ttng.tmem_load [[A_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %value, %read = ttng.tmem_load %acc[%w1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_a"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 10 : i32}
    tt.return
  }

  // B: the first write precedes the loop, so the default all/none masks stay.
  // SEMA-LABEL: @case_b
  // ASP-LABEL: @case_b
  tt.func @case_b(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // SEMA: [[B_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // SEMA: [[B_ENTRY:%.*]] = nvws.semaphore.create [[B_BASE]] released = 7 {pending_count = 1 : i32}
    // SEMA: [[B_FULL:%.*]] = nvws.semaphore.create [[B_BASE]] {pending_count = 1 : i32}
    // SEMA: [[B_INIT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]] :
    // ASP: [[B_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // ASP: [[B_ENTRY:%.*]] = nvws.semaphore.create [[B_BASE]] released = 7 {pending_count = 1 : i32}
    // ASP: [[B_FULL:%.*]] = nvws.semaphore.create [[B_BASE]] {pending_count = 1 : i32}
    // ASP: [[B_INITIAL_CURRENT_STAGE:%.*]] = arith.constant 2 : i32
    // ASP: [[B_INITIAL_ONE:%.*]] = arith.constant 1 : i32
    // ASP: [[B_INITIAL_NEXT_RAW:%.*]] = arith.addi [[B_INITIAL_CURRENT_STAGE]], [[B_INITIAL_ONE]] : i32
    // ASP: [[B_INITIAL_DEPTH:%.*]] = arith.constant 3 : i32
    // ASP: [[B_INITIAL_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[B_INITIAL_NEXT_RAW]], [[B_INITIAL_DEPTH]] : i32
    // ASP: [[B_INITIAL_ZERO:%.*]] = arith.constant 0 : i32
    // ASP: [[B_INITIAL_NEXT_STAGE:%.*]] = arith.select [[B_INITIAL_NEEDS_WRAP]], [[B_INITIAL_ZERO]], [[B_INITIAL_NEXT_RAW]] : i32
    // ASP: [[B_INIT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]][[[B_INITIAL_NEXT_STAGE]], {{%.*}}]
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: [[B_W0_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]], [[B_INIT_TOK]]
    // SEMA: ttng.tmem_store {{%.*}}, [[B_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
    // ASP: [[B_W0_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]][[[B_INITIAL_NEXT_STAGE]]], [[B_INIT_TOK]]
    // ASP: ttng.tmem_store {{%.*}}, [[B_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
    %w0 = ttng.tmem_store %zero, %acc[%tok], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // SEMA: scf.for {{.*}} iter_args([[B_MMA_TOK:%.*]] = [[B_INIT_TOK]])
    // ASP: scf.for {{.*}} iter_args([[B_MMA_TOK:%.*]] = [[B_INIT_TOK]], [[B_CURRENT_STAGE:%.*]] = [[B_INITIAL_NEXT_STAGE]],
    %outer = scf.for %i = %lb to %ub step %step iter_args(%carry = %w0) -> (!ttg.async.token) : i32 {
      // SEMA: [[B_MMA_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]], [[B_MMA_TOK]] {ttg.partition = array<i32: 1>}
      // SEMA: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[B_MMA_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // SEMA: nvws.semaphore.release [[B_FULL]], [[B_MMA_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[B_MMA_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]][[[B_CURRENT_STAGE]]], [[B_MMA_TOK]] {ttg.partition = array<i32: 1>}
      // ASP: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[B_MMA_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // ASP: nvws.semaphore.release [[B_FULL]][[[B_CURRENT_STAGE]]], [[B_MMA_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]] {ttg.partition = array<i32: 0>}
      // SEMA: [[B_READ_BUF:%.*]] = nvws.semaphore.buffer [[B_FULL]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: {{%.*}}, {{%.*}} = ttng.tmem_load [[B_READ_BUF]][] {ttg.partition = array<i32: 0>}
      // SEMA: nvws.semaphore.release [[B_ENTRY]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]][[[B_CURRENT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // ASP: [[B_READ_BUF:%.*]] = nvws.semaphore.buffer [[B_FULL]][[[B_CURRENT_STAGE]]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: {{%.*}}, {{%.*}} = ttng.tmem_load [[B_READ_BUF]][] {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[B_ENTRY]][[[B_CURRENT_STAGE]]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %value, %read = ttng.tmem_load %acc[%w1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_b"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // SEMA: [[B_NEXT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]] {ttg.partition = array<i32: 1>}
      // ASP: [[B_NEXT_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // ASP: [[B_NEXT_RAW:%.*]] = arith.addi [[B_CURRENT_STAGE]], [[B_NEXT_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // ASP: [[B_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[B_NEXT_RAW]], [[B_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // ASP: [[B_NEXT_STAGE:%.*]] = arith.select [[B_NEEDS_WRAP]], [[B_ZERO]], [[B_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_NEXT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]][[[B_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 11 : i32}
    tt.return
  }

  // C: the loop-exit relay reserves the successor slot for the next W0.
  // SEMA-LABEL: @case_c
  // ASP-LABEL: @case_c
  tt.func @case_c(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // SEMA: [[C_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // SEMA: [[C_ENTRY:%.*]] = nvws.semaphore.create [[C_BASE]] released = 1 {pending_count = 1 : i32}
    // SEMA: [[C_FULL:%.*]] = nvws.semaphore.create [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[C_FREE:%.*]] = nvws.semaphore.create [[C_BASE]] released = 6 {pending_count = 1 : i32}
    // ASP: [[C_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // ASP: [[C_ENTRY:%.*]] = nvws.semaphore.create [[C_BASE]] released = 1 {pending_count = 1 : i32}
    // ASP: [[C_FULL:%.*]] = nvws.semaphore.create [[C_BASE]] {pending_count = 1 : i32}
    // ASP: [[C_FREE:%.*]] = nvws.semaphore.create [[C_BASE]] released = 6 {pending_count = 1 : i32}
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: scf.for
    // ASP: scf.for {{.*}} iter_args([[C_OUTER_CURRENT_STAGE:%[-A-Za-z0-9_.$#]+]] = {{%[-A-Za-z0-9_.$#]+}},
    %outer = scf.for %i = %lb to %ub step %step iter_args(%outer_token = %tok) -> (!ttg.async.token) : i32 {
      // SEMA: [[C_W0_TOK:%.*]] = nvws.semaphore.acquire [[C_ENTRY]] {ttg.partition = array<i32: 0>}
      // SEMA: [[C_W0_BUF:%.*]] = nvws.semaphore.buffer [[C_ENTRY]], [[C_W0_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: ttng.tmem_store {{%.*}}, [[C_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // SEMA: nvws.semaphore.release [[C_FREE]], [[C_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[C_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // ASP: [[C_OUTER_NEXT_RAW:%.*]] = arith.addi [[C_OUTER_CURRENT_STAGE]], [[C_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // ASP: [[C_OUTER_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[C_OUTER_NEXT_RAW]], [[C_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // ASP: [[C_NEXT_OF_OUTER_STAGE:%.*]] = arith.select [[C_OUTER_NEEDS_WRAP]], [[C_ZERO]], [[C_OUTER_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_W0_TOK:%.*]] = nvws.semaphore.acquire [[C_ENTRY]][[[C_NEXT_OF_OUTER_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // ASP: [[C_W0_BUF:%.*]] = nvws.semaphore.buffer [[C_ENTRY]][[[C_NEXT_OF_OUTER_STAGE]]], [[C_W0_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: ttng.tmem_store {{%.*}}, [[C_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[C_FREE]][[[C_NEXT_OF_OUTER_STAGE]]], [[C_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %w0 = ttng.tmem_store %zero, %acc[%outer_token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: scf.for
      // ASP: [[C_INNER_LOOP:%.*]]:3 = scf.for {{.*}} iter_args([[C_INNER_CURRENT_STAGE:%[-A-Za-z0-9_.$#]+]] = [[C_NEXT_OF_OUTER_STAGE]],
      %inner = scf.for %j = %lb to %ub step %step iter_args(%inner_token = %w0) -> (!ttg.async.token) : i32 {
        // SEMA: [[C_W1_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]] {ttg.partition = array<i32: 1>}
        // SEMA: [[C_W1_BUF:%.*]] = nvws.semaphore.buffer [[C_FREE]], [[C_W1_TOK]] {ttg.partition = array<i32: 1>}
        // SEMA: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[C_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        // SEMA: nvws.semaphore.release [[C_FULL]], [[C_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // ASP: [[C_INNER_NEXT_RAW:%.*]] = arith.addi [[C_INNER_CURRENT_STAGE]], [[C_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
        // ASP: [[C_INNER_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[C_INNER_NEXT_RAW]], [[C_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
        // ASP: [[C_NEXT_OF_INNER_CURRENT_STAGE:%.*]] = arith.select [[C_INNER_NEEDS_WRAP]], [[C_ZERO]], [[C_INNER_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
        // ASP: [[C_W1_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]][[[C_NEXT_OF_INNER_CURRENT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
        // ASP: [[C_W1_BUF:%.*]] = nvws.semaphore.buffer [[C_FREE]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_W1_TOK]] {ttg.partition = array<i32: 1>}
        // ASP: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[C_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        // ASP: nvws.semaphore.release [[C_FULL]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%inner_token], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // SEMA: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]] {ttg.partition = array<i32: 0>}
        // SEMA: [[C_READ_BUF:%.*]] = nvws.semaphore.buffer [[C_FULL]], [[C_READ_TOK]] {ttg.partition = array<i32: 0>}
        // SEMA: {{%.*}}, {{%.*}} = ttng.tmem_load [[C_READ_BUF]][] {ttg.partition = array<i32: 0>}
        // SEMA: nvws.semaphore.release [[C_FREE]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        // ASP: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[C_NEXT_OF_INNER_CURRENT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
        // ASP: [[C_READ_BUF:%.*]] = nvws.semaphore.buffer [[C_FULL]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_READ_TOK]] {ttg.partition = array<i32: 0>}
        // ASP: {{%.*}}, {{%.*}} = ttng.tmem_load [[C_READ_BUF]][] {ttg.partition = array<i32: 0>}
        // ASP: nvws.semaphore.release [[C_FREE]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        %value, %read = ttng.tmem_load %acc[%w1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "use_c"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // The accessless relay targets next(inner-stage) on both sides.
      // SEMA: [[C_ACQUIRE_AUTHORED_OFFSET_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // SEMA: [[C_EXIT_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]][[[C_ACQUIRE_AUTHORED_OFFSET_ONE]]] {ttg.partition = array<i32: 1>}
      // SEMA: [[C_RELEASE_AUTHORED_OFFSET_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // SEMA: nvws.semaphore.release [[C_ENTRY]][[[C_RELEASE_AUTHORED_OFFSET_ONE]]], [[C_EXIT_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[C_NEXT_OF_INNER_RAW:%.*]] = arith.addi [[C_INNER_LOOP]]#0, [[C_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_NEXT_OF_INNER_REM:%.*]] = arith.remsi [[C_NEXT_OF_INNER_RAW]], [[C_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_NEXT_OF_INNER_STAGE:%.*]] = arith.select {{.*}}, {{.*}}, [[C_NEXT_OF_INNER_REM]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[C_REL_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // ASP: [[C_EXIT_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]][[[C_NEXT_OF_INNER_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      // ASP: [[C_RELEASE_NEXT_OF_INNER_RAW:%.*]] = arith.addi [[C_INNER_LOOP]]#0, [[C_REL_ONE]] {ttg.partition = array<i32: 1>} : i32
      // ASP: [[C_REL_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 3 : i32
      // ASP: [[C_RELEASE_NEXT_OF_INNER_REM:%.*]] = arith.remsi [[C_RELEASE_NEXT_OF_INNER_RAW]], [[C_REL_DEPTH]] {ttg.partition = array<i32: 1>} : i32
      // ASP: [[C_RELEASE_NEXT_OF_INNER_STAGE:%.*]] = arith.select {{.*}}, {{.*}}, [[C_RELEASE_NEXT_OF_INNER_REM]] {ttg.partition = array<i32: 1>} : i32
      // ASP: nvws.semaphore.release [[C_ENTRY]][[[C_RELEASE_NEXT_OF_INNER_STAGE]]], [[C_EXIT_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      scf.yield {ttg.partition = array<i32: 0, 1>} %inner : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 12 : i32}
    tt.return
  }

  // D: the acquire preceding the nonempty inner loop is the fresh epoch.
  // SEMA-LABEL: @case_d
  // ASP-LABEL: @case_d
  tt.func @case_d(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // SEMA: [[D_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // SEMA: [[D_ENTRY:%.*]] = nvws.semaphore.create [[D_BASE]] released = 1 {pending_count = 1 : i32}
    // SEMA: [[D_NEXT:%.*]] = nvws.semaphore.create [[D_BASE]] released = 6 {pending_count = 1 : i32}
    // SEMA: [[D_INIT_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 13 : i32} :
    // ASP: [[D_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // ASP: [[D_ENTRY:%.*]] = nvws.semaphore.create [[D_BASE]] released = 1 {pending_count = 1 : i32}
    // ASP: [[D_NEXT:%.*]] = nvws.semaphore.create [[D_BASE]] released = 6 {pending_count = 1 : i32}
    // ASP: [[D_INITIAL_CURRENT_STAGE:%.*]] = arith.constant 2 : i32
    // ASP: [[D_INITIAL_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} 1 : i32
    // ASP: [[D_INITIAL_NEXT_RAW:%.*]] = arith.addi [[D_INITIAL_CURRENT_STAGE]], [[D_INITIAL_ONE]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} : i32
    // ASP: [[D_INITIAL_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} 3 : i32
    // ASP: [[D_INITIAL_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[D_INITIAL_NEXT_RAW]], [[D_INITIAL_DEPTH]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} : i32
    // ASP: [[D_INITIAL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} 0 : i32
    // ASP: [[D_INITIAL_NEXT_STAGE:%.*]] = arith.select [[D_INITIAL_NEEDS_WRAP]], [[D_INITIAL_ZERO]], [[D_INITIAL_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} : i32
    // ASP: [[D_INIT_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]][[[D_INITIAL_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 13 : i32}
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // SEMA: scf.for {{.*}} iter_args([[D_W0_TOK:%.*]] = [[D_INIT_TOK]])
    // ASP: scf.for {{.*}} iter_args([[D_W0_TOK:%.*]] = [[D_INIT_TOK]], [[D_CURRENT_STAGE:%.*]] = [[D_INITIAL_NEXT_STAGE]],
    %outer = scf.for %i = %lb to %ub step %step iter_args(%outer_token = %tok) -> (!ttg.async.token) : i32 {
      // SEMA: [[D_W0_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]], [[D_W0_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: ttng.tmem_store {{%.*}}, [[D_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // SEMA: nvws.semaphore.release [[D_NEXT]], [[D_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[D_W0_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]][[[D_CURRENT_STAGE]]], [[D_W0_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: ttng.tmem_store {{%.*}}, [[D_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[D_NEXT]][[[D_CURRENT_STAGE]]], [[D_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %w0 = ttng.tmem_store %zero, %acc[%outer_token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // SEMA: [[D_W1_TOK:%.*]] = nvws.semaphore.acquire [[D_NEXT]] {ttg.partition = array<i32: 1>}
      // SEMA: scf.for
      // ASP: [[D_NEXT_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // ASP: [[D_NEXT_RAW:%.*]] = arith.addi [[D_CURRENT_STAGE]], [[D_NEXT_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[D_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // ASP: [[D_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[D_NEXT_RAW]], [[D_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[D_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // ASP: [[D_NEXT_STAGE:%.*]] = arith.select [[D_NEEDS_WRAP]], [[D_ZERO]], [[D_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[D_W1_TOK:%.*]] = nvws.semaphore.acquire [[D_NEXT]][[[D_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      // ASP: scf.for
      %inner = scf.for %j = %lb to %ub step %step iter_args(%inner_token = %w0) -> (!ttg.async.token) : i32 {
        // SEMA: [[D_W1_BUF:%.*]] = nvws.semaphore.buffer [[D_NEXT]], [[D_W1_TOK]] {ttg.partition = array<i32: 1>}
        // SEMA: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[D_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        // ASP: [[D_W1_BUF:%.*]] = nvws.semaphore.buffer [[D_NEXT]][[[D_NEXT_STAGE]]], [[D_W1_TOK]] {ttg.partition = array<i32: 1>}
        // ASP: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[D_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%inner_token], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1>} %w1 : !ttg.async.token
      } {ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      // SEMA: nvws.semaphore.release [[D_ENTRY]], [[D_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[D_READ_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]] {ttg.partition = array<i32: 0>}
      // SEMA: [[D_READ_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]], [[D_READ_TOK]] {ttg.partition = array<i32: 0>}
      // SEMA: {{%.*}}, {{%.*}} = ttng.tmem_load [[D_READ_BUF]][] {ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[D_ENTRY]][[[D_NEXT_STAGE]]], [[D_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[D_READ_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]][[[D_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // ASP: [[D_READ_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]][[[D_NEXT_STAGE]]], [[D_READ_TOK]] {ttg.partition = array<i32: 0>}
      // ASP: {{%.*}}, {{%.*}} = ttng.tmem_load [[D_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %value, %read = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_d"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 13 : i32}
    tt.return
  }
}

//--- insert_semas_if_encloser_inner_loop.mlir

// M0 PIN (nested-loop hold-rule extension, plan v3 §M0 item 5 — the
// canDrop(If) golden; the true If-ENCLOSER target is ABSENT from the corpus).
//
// WS-tagged outer loop -> scf.if -> non-WS inner loop, with ONE inner-confined
// ping-pong buffer in the if-branch. The scf.if sits BETWEEN the WS loop and
// the inner for, so it is the inner loop's encloser.
//
// Point-of-use construction: no carrier token threads the outer loop, the
// scf.if, or the inner loop (loop-close release is partition 0, first acquire
// is partition 1, so no yield-carried token). EMPTY is created initially
// released and acquired inside the inner-loop body at the producer's first
// use; the else branch stays empty.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL:   tt.func @if_encloser_inner_loop
  tt.func @if_encloser_inner_loop(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %true = arith.constant true
    // Single-buffered (1x) ping-pong: alloc, create EMPTY (initially released) / FULL pair; no pre-loop acquire.
    // CHECK:           [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %res, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // WS outer loop carries no tokens.
    // CHECK:           scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    %o = scf.for %iv0 = %lb to %ub step %step iter_args(%t0 = %tok) -> (!ttg.async.token) : i32 {
      // scf.if encloser yields no carriers.
      // CHECK:             scf.if %{{[-A-Za-z0-9_.$#]+}} {
      %r = scf.if %cond -> (!ttg.async.token) {
        // Inner non-WS loop carries no tokens.
        // CHECK:               scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
        %i = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %t0) -> (!ttg.async.token) : i32 {
          // CHECK:                 [[SA:%.*]] = "loadA"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
          %sA = "loadA"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
          // CHECK:                 [[SB:%.*]] = "loadB"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
          %sB = "loadB"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
          // Producer (partition 1): acquire EMPTY at point of use, buffer feeds the MMA, then release FULL.
          // CHECK:                 [[ACQ_EMPTY:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK:                 [[MMA_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ACQ_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK:                 ttng.tc_gen5_mma [[SA]], [[SB]], [[MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          %mma = ttng.tc_gen5_mma %sA, %sB, %res[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK:                 nvws.semaphore.release [[FULL]], [[ACQ_EMPTY]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          // Consumer (partition 0): acquire FULL, buffer feeds the load, release EMPTY.
          // CHECK:                 [[ACQ_FULL:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK:                 [[LOAD_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[ACQ_FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK:                 %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[LOAD_BUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
          %val, %t2 = ttng.tmem_load %res[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          // CHECK:                 nvws.semaphore.release [[EMPTY]], [[ACQ_FULL]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          // CHECK:                 "use"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          scf.yield {ttg.partition = array<i32: 0, 1>} %t2 : !ttg.async.token
        // CHECK:               } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
        scf.yield {ttg.partition = array<i32: 0, 1>} %i : !ttg.async.token
      } else {
        // Else branch: empty (no carriers to pass through, no semaphore op).
        // CHECK:             } else {
        // CHECK-NEXT:        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
        scf.yield {ttg.partition = array<i32: 0, 1>} %t0 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      scf.yield {ttg.partition = array<i32: 0, 1>} %r : !ttg.async.token
    // CHECK:           } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    // CHECK:           tt.return
    tt.return
  }
}

//--- insert_semas_if_split_metadata.mlir

// Regression for branch-carried conditional metadata with non-default
// partitions. The taken branch returns ownership to partition 4, the other
// branch passes partition 4's token through, and the if result remains the
// loop-carried token. Nothing may assume partitions 0/1.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @if_split_workaround_nondefault_partitions
  tt.func @if_split_workaround_nondefault_partitions(%arg0: !tt.tensordesc<1x64xf16, #shared>, %arg1: tensor<64x128x!tt.ptr<f16>, #blocked3> {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    // Single-buffered (disallow_acc_multi_buffer): alloc grows to 1x, then the
    // EMPTY/FULL semaphore pair, the initial acquire of EMPTY, its buffer, and
    // the init store writing through that buffer.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[INITTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[INITBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[INITTOK]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[INITBUF]][], %{{[-A-Za-z0-9_.$#]+}} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // The acquire token is threaded as the loop's third iter_arg.
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}, [[CARRYTOK:%.*]] = [[INITTOK]]) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.async.token)  : i32 {
    %1:3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %arg1, %arg5 = %0) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4, 5>} : (i32) -> (i32, tensor<64x128xi32, #blocked3>, i32)
      %3 = tt.splat %2#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : i32 -> tensor<128xi32, #blocked2>
      %4 = tt.descriptor_gather %arg0[%3, %2#2] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : (!tt.tensordesc<1x64xf16, #shared>, tensor<128xi32, #blocked2>, i32) -> tensor<128x64xf16, #blocked1>
      %5 = tt.addptr %arg4, %2#1 {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>, ttg.partition = array<i32: 4>} : tensor<64x128x!tt.ptr<f16>, #blocked3>, tensor<64x128xi32, #blocked3>
      %6 = tt.load %5 {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<64x128x!tt.ptr<f16>, #blocked3>
      %7 = ttg.local_alloc %4 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 5>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA reads its accumulator through a buffer derived from the carried
      // EMPTY token.
      // CHECK: [[MMABUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRYTOK]] {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[MMABUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 5 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 4>}
      %9 = ttng.tc_gen5_mma %7, %8, %result[%arg5], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %10 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 4>} : i32
      %11 = arith.select %10, %false, %true {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 4>} : i1
      // The taken branch performs the complete {4}->{0}->{4} handoff. The
      // other branch passes the loop-carried partition-4 token through.
      // CHECK: [[NEXTTOK:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
      // CHECK: nvws.semaphore.release [[FULL]], [[CARRYTOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[CONSTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[CONSBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CONSTOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_load [[CONSBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[CONSTOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[HAND_BACK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 4>} [[HAND_BACK]] : !ttg.async.token
      // CHECK: } else {
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 4>} [[CARRYTOK]] : !ttg.async.token
      // CHECK: } {loop.cluster = 4 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 4>, ttg.partition.outputs = [array<i32: 4>]}
      %12 = scf.if %10 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%9] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 4>} %token_1 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 4>} %9 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 4>, ttg.partition.outputs = [array<i32: 4>]}
      // CHECK: scf.yield {{.*}}[[NEXTTOK]] : i1, tensor<64x128x!tt.ptr<f16>, #blocked>, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 4, 5>} %11, %5, %12 : i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token
    // Loop close: the schedule/partition attrs are preserved verbatim.
    // CHECK: } {tt.disallow_acc_multi_buffer, tt.num_stages = 3 : i32, tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 4, 5>, ttg.partition.outputs = [array<i32: 4>, array<i32: 4>, array<i32: 4>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 3 : i32, tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 4, 5>, ttg.partition.outputs = [array<i32: 4>, array<i32: 4>, array<i32: 4>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }

  // CHECK-LABEL: @if_split_yield_routing_three_partitions
  tt.func @if_split_yield_routing_three_partitions(%lhs: !ttg.memdesc<128x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %false = arith.constant false

    // Double-buffered (2x): alloc, then the EMPTY/FULL semaphore pair.
    // CHECK: [[ALLOC2:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY2:%.*]] = nvws.semaphore.create [[ALLOC2]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[FULL2:%.*]] = nvws.semaphore.create [[ALLOC2]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[INITIAL2:%.*]] = nvws.semaphore.acquire [[EMPTY2]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: [[LOOP2:%.*]]:3 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[USE2:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[VALUE2:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[CARRY2:%.*]] = [[INITIAL2]]) -> (i1, i32, !ttg.async.token)  : i32 {
    %loop:3 = scf.for %iv = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%use_acc = %false, %tok = %acc_tok, %carry = %c0_i32) -> (i1, !ttg.async.token, i32) : i32 {
      // The MMA uses the token carried from the preceding iteration.
      // CHECK: [[BODYBUF:%.*]] = nvws.semaphore.buffer [[EMPTY2]], [[CARRY2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[BODYBUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok], %use_acc, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %aux = "p2_work"(%iv) {ttg.partition = array<i32: 2>} : (i32) -> i32
      "p2_sink"(%aux) {ttg.partition = array<i32: 2>} : (i32) -> ()
      %cond = arith.cmpi eq, %iv, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32

      // The branch-carried if must not pick up partition 2. The taken branch
      // performs {1}->{0}->{1}; the other branch passes [[CARRY2]] through.
      // CHECK: [[BRANCH2:%.*]]:3 = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (i32, i1, !ttg.async.token) {
      // CHECK: nvws.semaphore.release [[FULL2]], [[CARRY2]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[CONS2TOK:%.*]] = nvws.semaphore.acquire [[FULL2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[CONS2BUF:%.*]] = nvws.semaphore.buffer [[FULL2]], [[CONS2TOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_load [[CONS2BUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY2]], [[CONS2TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[BACK2:%.*]] = nvws.semaphore.acquire [[EMPTY2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[BACK2]] : i32, i1, !ttg.async.token
      // CHECK: } else {
      // CHECK: scf.yield {{.*}}[[VALUE2]], [[USE2]], [[CARRY2]] : i32, i1, !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1>, array<i32: 1>]}
      %epilogue:3 = scf.if %cond -> (i32, !ttg.async.token, i1) {
        %value, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %iv, %load_tok, %true : i32, !ttg.async.token, i1
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %carry, %mma, %use_acc : i32, !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 0, 1>]}
      %next = arith.addi %epilogue#0, %c1_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.yield {{.*}}[[BRANCH2]]#1, %{{[-A-Za-z0-9_.$#]+}}, [[BRANCH2]]#2 : i1, i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %epilogue#2, %epilogue#1, %next : i1, !ttg.async.token, i32
    // Loop close: the conditional token remains the third loop result.
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}

//--- insert_semas_live_tag_source.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @live_tag_source_after_prior_loop_threading
  tt.func @live_tag_source_after_prior_loop_threading(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %scratch = ttg.local_alloc {buffer.id = 910 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 910 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = ttng.tmem_alloc {buffer.id = 900 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V4]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V4]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V4]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V11:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V9:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V10:%.*]] = [[V8]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V5]], [[V10]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: nvws.semaphore.release [[V7]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner_tmem = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load_lhs"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load_rhs"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V14:%.*]] = nvws.semaphore.buffer [[V7]], [[V13]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V14]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[V6]], [[V13]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V6]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V16]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V7]], [[V15]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use_tmem"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // CHECK: [[V17:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[V5]], [[V17]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V18:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V19:%.*]] = nvws.semaphore.buffer [[V5]], [[V18]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V19]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner_tmem] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_tmem_post"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      %payload = "local_payload"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
      // CHECK: [[V20:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V21:%.*]] = nvws.semaphore.buffer [[V2]], [[V20]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V21]] {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload, %scratch {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      scf.for %iv2 = %lb to %ub step %step : i32 {
        // CHECK: [[V23:%.*]] = nvws.semaphore.buffer [[V3]], [[V22]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK: [[V24:%.*]] = ttg.local_load [[V23]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
        %loaded = ttg.local_load %scratch {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
        "use_local"(%loaded) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      } {ttg.partition = array<i32: 1>}

      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: nvws.semaphore.release [[V2]], [[V22]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V18]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_local_buffer_reuse.mlir

// Local-memory mirrors of TMEM buffer-reuse tests. These exercise the
// same v4 §Physical Conflict Key behaviors (buffer.id grouping +
// buffer.offset overlap classification) on ttg.local_alloc instead of
// ttng.tmem_alloc. Until the make-group path is unified the local
// allocs are treated as independent groups; once unified they will
// share a logical buffer group and the dump / emit shape will match
// the TMEM mirrors.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Mirror of @sourceful_tokenless_alias from tmem-buffer-reuse-semas.mlir.
  // Two local_allocs share buffer.id=400 and overlap at offsets 0 and 64
  // (extent 128 each → physical-conflict-key match). Two partitions
  // alternate: {1} writes/reads member 0, then {0} writes/reads member 1.
  // CHECK-LABEL: @local_sourceful_aliased_buffers
  tt.func @local_sourceful_aliased_buffers(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 400 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = ttg.local_alloc {buffer.id = 400 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V2]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V2]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V6:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V5:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[V7:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V8:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]]#0 {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %a = ttg.local_alloc %cst {buffer.id = 400 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V9:%.*]] = ttg.local_load [[V8]]#0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]]:2 = nvws.semaphore.buffer [[V4]], [[V10]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V11]]#1 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %b = ttg.local_alloc %cst_0 {buffer.id = 400 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V12:%.*]] = ttg.local_load [[V11]]#1 {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %vb = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[V3]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // Mirror of @n_owner_alias_sequence from tmem-buffer-reuse-semas.mlir.
  // Two local_allocs share buffer.id=401 and overlap at offsets 0 and 64.
  // Three partitions form a linear chain: {0} writes m0, {1} reads m0,
  // {2} writes m1, {0} reads m1 — alternating EMPTY/FULL semaphore shape.
  // CHECK-LABEL: @local_n_owner_aliased_buffers
  tt.func @local_n_owner_aliased_buffers(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = ttg.local_alloc {buffer.id = 401 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V2]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V2]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V1]], [[V2]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V1]], [[V2]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V8:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // {0} writes m0 through the carried token, hands to {1}; {1}
      // reads and releases both onward ({2}) and the carrier regain.
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V10]]#0 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %a = ttg.local_alloc %cst0 {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[V4]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]]:2 = nvws.semaphore.buffer [[V4]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: [[V13:%.*]] = ttg.local_load [[V12]]#0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[V5]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V3]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
	      // {2} writes m1 (its regain S4 was reduced away — traversal
	      // closure), hands to {0}; {0} reads; the carrier regain closes.
	      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
	      // CHECK: [[V15:%.*]]:2 = nvws.semaphore.buffer [[V5]], [[V14]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
	      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V15]]#1 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
	      %b = ttg.local_alloc %cst1 {buffer.id = 401 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
	      // CHECK: nvws.semaphore.release [[V6]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
	      // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
	      // CHECK: [[V17:%.*]]:2 = nvws.semaphore.buffer [[V6]], [[V16]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
	      // CHECK: [[V18:%.*]] = ttg.local_load [[V17]]#1 {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
	      %vb = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
	      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 1 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

}

//--- insert_semas_local_cfg.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_if_conditional_only
  tt.func @local_if_conditional_only(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V6:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V5:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %cond = "cond"() {ttg.partition = array<i32: 0, 1>} : () -> i1
      // CHECK: [[IF_TOKEN:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
      scf.if %cond {
        // CHECK: nvws.semaphore.release [[V3]], [[V5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V3]], [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[V10:%.*]] = ttg.local_load [[V9]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        // CHECK: nvws.semaphore.release [[V2]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "use_then"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: [[HAND_BACK:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {{.*}}[[HAND_BACK]] : !ttg.async.token
      } else {
        // The non-consuming branch keeps the producer token.
        // CHECK: } else {
        // CHECK: scf.yield {{.*}}[[V5]] : !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V2]], [[IF_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v2, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: scf.yield {{.*}}[[IF_TOKEN]]
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_if_conditional_only_else
  tt.func @local_if_conditional_only_else(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 203 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 203 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V6:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V5:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %cond = "cond"() {ttg.partition = array<i32: 0, 1>} : () -> i1
      // The non-consuming then branch keeps the producer token.
      // CHECK: [[IF_TOKEN:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
      scf.if %cond {
        // CHECK: scf.yield {{.*}}[[V5]] : !ttg.async.token
      } else {
        // CHECK: } else {
        // CHECK: nvws.semaphore.release [[V3]], [[V5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V3]], [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[V10:%.*]] = ttg.local_load [[V9]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        // CHECK: nvws.semaphore.release [[V2]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "use_else"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: [[HAND_BACK:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {{.*}}[[HAND_BACK]] : !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V2]], [[IF_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v2, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: scf.yield {{.*}}[[IF_TOKEN]]
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_if_consumption_continues_after_join
  tt.func @local_if_consumption_continues_after_join(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 201 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 201 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[JOIN:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V7:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = [[V5]]) -> (!ttg.async.token)  : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %cond = "cond"() {ttg.partition = array<i32: 0, 1>} : () -> i1
      // The if first returns owner {0}. Only then does the following {0}->{1}
      // handoff use the if result; it must not bypass the conditional.
      // CHECK-NOT: nvws.semaphore.release [[JOIN]]
      // CHECK: [[IF_TOKEN:%.*]] = scf.if
      scf.if %cond {
        // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[V11:%.*]] = ttg.local_load [[V10]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        // CHECK: nvws.semaphore.release [[V4]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "use_then"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: [[BRANCH_BACK:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: scf.yield {{.*}}[[BRANCH_BACK]] : !ttg.async.token
      } else {
        // CHECK: } else {
        // CHECK-NEXT: scf.yield {{.*}}[[V6]] : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // CHECK-NEXT: nvws.semaphore.release [[JOIN]], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[JOIN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[JOIN]], [[V12]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V14:%.*]] = ttg.local_load [[V13]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V2]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "use_after"(%l2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V2]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V16]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v2, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: scf.yield {{.*}}[[V15]]
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_root_external_distinct_from_ws_tag_zero
  // CHECK-NOT: nvws.semaphore
  tt.func @local_root_external_distinct_from_ws_tag_zero(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 202 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 202 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NOT: nvws.semaphore
    %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V1]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NOT: nvws.semaphore
    ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    // CHECK-NOT: nvws.semaphore
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[V2:%.*]] = ttg.local_load [[V1]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      // CHECK-NOT: nvws.semaphore
      %l = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "use"(%l) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NOT: nvws.semaphore
    } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_local_errors.mlir

// NUM-STAGES: nvws-insert-semas: num-stages must be in [1, 32], got 33

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @unsupported_local_memdesc_forwarding(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 203 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      // CHECK: nvws-insert-semas: unsupported memdesc alias use test.memdesc_view
      %view = "test.memdesc_view"(%alloc) {ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> !ttg.memdesc<1xi32, #shared, #smem>
      %l = ttg.local_load %view {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> !ty
      "use"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reject_zero_buffer_copy(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: nvws-insert-semas: buffer.copy must be in [1, 32], got 0
    %alloc = ttg.local_alloc {buffer.copy = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
    } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reject_buffer_copy_above_mask_width() {
    // CHECK: nvws-insert-semas: buffer.copy must be in [1, 32], got 33
    %alloc = ttg.local_alloc {buffer.copy = 33 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

//--- insert_semas_local_mixed_copy_reuse.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_small = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @mixed_copy_same_backing
  // One 256x128 host covers the smaller two-copy 128x64 view.
  tt.func @mixed_copy_same_backing(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: %[[BASE:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 600 : i32} : () -> !ttg.memdesc<1x256x128xf16
    %host = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 600 : i32} : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    // CHECK-NEXT: %[[RING0:.*]] = ttg.memdesc_reinterpret %[[BASE]] : {{.*}} -> !ttg.memdesc<2x128x64xf16
    // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[SLOT0:.*]] = ttg.memdesc_index %[[RING0]][%[[ZERO]]]
    // CHECK-NEXT: %[[VIEW0:.*]] = ttg.memdesc_reinterpret %[[SLOT0]] : {{.*}} -> !ttg.memdesc<1x128x64xf16
    %slot0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 600 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK-NEXT: %[[RING1:.*]] = ttg.memdesc_reinterpret %[[BASE]] : {{.*}} -> !ttg.memdesc<2x128x64xf16
    // CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK-NEXT: %[[SLOT1:.*]] = ttg.memdesc_index %[[RING1]][%[[ONE]]]
    // CHECK-NEXT: %[[VIEW1:.*]] = ttg.memdesc_reinterpret %[[SLOT1]] : {{.*}} -> !ttg.memdesc<1x128x64xf16
    // CHECK-NEXT: %[[ENTRY:.*]] = nvws.semaphore.create %[[BASE]], %[[VIEW0]], %[[VIEW1]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: %[[TO_R0:.*]] = nvws.semaphore.create %[[BASE]], %[[VIEW0]], %[[VIEW1]] {pending_count = 1 : i32}
    // CHECK-NEXT: %[[TO_W1:.*]] = nvws.semaphore.create %[[BASE]], %[[VIEW0]], %[[VIEW1]] {pending_count = 1 : i32}
    %slot1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 600 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %wide = arith.constant dense<0.0> : tensor<256x128xf16, #blocked>
    %half = arith.constant dense<1.0> : tensor<128x64xf16, #blocked64>
    // CHECK: %[[INIT:.*]] = nvws.semaphore.acquire %[[ENTRY]]
    // CHECK-NEXT: %[[LOOP_RESULT:.*]] = scf.for {{.*}} iter_args(%[[CARRY:.*]] = %[[INIT]])
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: %[[HOST_WRITE_BUF:.*]]:3 = nvws.semaphore.buffer %[[ENTRY]], %[[CARRY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store %{{.*}}, %[[HOST_WRITE_BUF]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release %[[TO_R0]], %[[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %wide, %host {ttg.partition = array<i32: 0>} : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[R0_TOKEN:.*]] = nvws.semaphore.acquire %[[TO_R0]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[R0_BUF:.*]]:3 = nvws.semaphore.buffer %[[TO_R0]], %[[R0_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[R0_VALUE:.*]] = ttg.local_load %[[R0_BUF]]#1 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release %[[TO_W1]], %[[R0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %r0 = ttg.local_load %slot0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "use0"(%r0) {ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked64>) -> ()
      // CHECK: %[[W1_TOKEN:.*]] = nvws.semaphore.acquire %[[TO_W1]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: %[[W1_BUF:.*]]:3 = nvws.semaphore.buffer %[[TO_W1]], %[[W1_TOKEN]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttg.local_store %{{.*}}, %[[W1_BUF]]#2 {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release %[[ENTRY]], %[[W1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      ttg.local_store %half, %slot1 {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[HOST_READ_TOKEN:.*]] = nvws.semaphore.acquire %[[ENTRY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: %[[HOST_READ_BUF:.*]]:3 = nvws.semaphore.buffer %[[ENTRY]], %[[HOST_READ_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: %[[HOST_VALUE:.*]] = ttg.local_load %[[HOST_READ_BUF]]#0 {ttg.partition = array<i32: 0>}
      %r1 = ttg.local_load %host {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blocked>
      "use1"(%r1) {ttg.partition = array<i32: 0>} : (tensor<256x128xf16, #blocked>) -> ()
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %[[HOST_READ_TOKEN]] : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @same_copy_smaller_view
  tt.func @same_copy_smaller_view(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: %[[LARGE_BASE:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 602 : i32} : () -> !ttg.memdesc<1x128x128xf16
    %large = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 602 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK-NEXT: %[[SMALL_VIEW:.*]] = ttg.memdesc_reinterpret %[[LARGE_BASE]] : {{.*}} -> !ttg.memdesc<1x64x64xf16
    // CHECK-NEXT: %[[SMALL_ENTRY:.*]] = nvws.semaphore.create %[[LARGE_BASE]], %[[SMALL_VIEW]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: %[[SMALL_FULL:.*]] = nvws.semaphore.create %[[LARGE_BASE]], %[[SMALL_VIEW]] {pending_count = 1 : i32}
    %small = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 602 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %large_value = arith.constant dense<0.0> : tensor<128x128xf16, #blocked>
    // CHECK: scf.for
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: %[[LARGE_WRITE_TOKEN:.*]] = nvws.semaphore.acquire %[[SMALL_ENTRY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: %[[LARGE_WRITE_BUF:.*]]:2 = nvws.semaphore.buffer %[[SMALL_ENTRY]], %[[LARGE_WRITE_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store %{{.*}}, %[[LARGE_WRITE_BUF]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release %[[SMALL_FULL]], %[[LARGE_WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %large_value, %large {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[SMALL_TOKEN:.*]] = nvws.semaphore.acquire %[[SMALL_FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[SMALL_READ_BUF:.*]]:2 = nvws.semaphore.buffer %[[SMALL_FULL]], %[[SMALL_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %[[SMALL_VALUE:.*]] = ttg.local_load %[[SMALL_READ_BUF]]#1 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release %[[SMALL_ENTRY]], %[[SMALL_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %value = ttg.local_load %small {ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked_small>
      "use_small"(%value) {ttg.partition = array<i32: 1>} : (tensor<64x64xf16, #blocked_small>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.partition.outputs = [], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }

  // CHECK-LABEL: @tokenless_copy_one_alias
  // CHECK-NOT: nvws.semaphore
  tt.func @tokenless_copy_one_alias(%lb: i32, %ub: i32, %step: i32,
                                     %value: tensor<128x64xf16, #blocked64>) {
    // CHECK: %[[ONE_BASE:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 601 : i32}
    // CHECK-NEXT: %[[A_ZERO:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[A_VIEW:.*]] = ttg.memdesc_index %[[ONE_BASE]][%[[A_ZERO]]]
    %a = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 601 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK-NEXT: %[[B_ZERO:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[B_VIEW:.*]] = ttg.memdesc_index %[[ONE_BASE]][%[[B_ZERO]]]
    // CHECK-NOT: nvws.semaphore
    %b = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 601 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: ttg.local_store %{{.*}}, %[[A_VIEW]] {ttg.partition = array<i32: 0>}
      ttg.local_store %value, %a {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[LOADED:.*]] = ttg.local_load %[[B_VIEW]] {ttg.partition = array<i32: 0>}
      %loaded = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "use"(%loaded) {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      scf.yield {ttg.partition = array<i32: 0>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0>,
       ttg.partition.outputs = [], ttg.warp_specialize.tag = 2 : i32}
    // CHECK-NOT: nvws.semaphore
    // CHECK: tt.return
    tt.return
  }
}

//--- insert_semas_local_no_buffer_id.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_no_buffer_id
  tt.func @local_no_buffer_id(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = nvws.semaphore.create %{{[-A-Za-z0-9_.$#]+}} released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create %{{[-A-Za-z0-9_.$#]+}} {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V3:%.*]] = nvws.semaphore.acquire [[V1]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V4:%.*]] = nvws.semaphore.buffer [[V1]], [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V4]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V2]], [[V3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V6:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V7:%.*]] = ttg.local_load [[V6]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V1]], [[V5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_local_read_lifetime.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @outer_produced_inner_consumed
  tt.func @outer_produced_inner_consumed(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %outer = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 2>} : () -> tensor<128x64xf16, #blocked>
      // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]] {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      // CHECK-NEXT: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // The inner loop acquires once before entry. Every read uses that token,
      // and one release after the loop returns the buffer to owner {2}.
      // CHECK-NEXT: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
      scf.for %inner = %lb to %ub step %step : i32 {
        // CHECK-NEXT: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[V8:%.*]] = ttg.local_load [[V7]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: "use_tensor"([[V8]]) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        %loaded = ttg.local_load %alloc {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
        "use_tensor"(%loaded) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
      // CHECK-NEXT: } {ttg.partition = array<i32: 1>}
      } {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_memdesc_subslice.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_memdesc_subslice_alias
  tt.func @local_memdesc_subslice_alias(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 9920 : i32} : () -> !ttg.memdesc<1x2xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[EMPTY:%[0-9]+]] = nvws.semaphore.create [[BASE]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>
    // CHECK-NEXT: [[FULL:%[0-9]+]] = nvws.semaphore.create [[BASE]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 9920 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[VALUE:%[0-9]+]] = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<2xi32, #blocked>
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      // CHECK: [[EMPTY_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[WHOLE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[EMPTY]], [[EMPTY_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: ttg.local_store [[VALUE]], [[WHOLE_BUFFER]] {ttg.partition = array<i32: 0>} : tensor<2xi32, #blocked> -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]], [[EMPTY_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[FULL_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[WHOLE_READ_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[FULL]], [[FULL_TOKEN]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[SLICE:%[0-9]+]] = ttg.memdesc_subslice [[WHOLE_READ_BUFFER]][0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %view = ttg.memdesc_subslice %alloc[0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[LOADED:%[0-9]+]] = ttg.local_load [[SLICE]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %loaded = ttg.local_load %view {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      // CHECK: nvws.semaphore.release [[EMPTY]], [[FULL_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x2xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!one) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_memdesc_trans_alloc_shape.mlir

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The first MMA operand gives member A its exact view. Member B is cached at
  // the same time as a generic view, whose allocShape includes the three-stage
  // backing. Replaying B's transpose must infer its result type from that view.
  // CHECK-LABEL: @memdesc_trans_preserves_staged_alloc_shape
  tt.func @memdesc_trans_preserves_staged_alloc_shape(
      %desc_a: !tt.tensordesc<128x64xf16, #shared>,
      %desc_b: !tt.tensordesc<256x64xf16, #shared>,
      %acc: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
      %tok: !ttg.async.token) {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V2]] released = 7 {pending_count = 1 : i32} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V2]] {pending_count = 1 : i32} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>
    %r = scf.for %iv = %c0 to %c1 step %c1 iter_args(%flag = %false) -> (i1) : i32 {
      // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V6:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V5]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>
      // CHECK: nvws.descriptor_load %{{[-A-Za-z0-9_.$#]+}}[%{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}] 16384 [[V6]]#0 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc_a[%c0, %c0] 16384 %a {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: nvws.descriptor_load %{{[-A-Za-z0-9_.$#]+}}[%{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}] 32768 [[V6]]#1 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<256x64xf16, #shared>, i32, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>
      // CHECK: nvws.semaphore.release [[V4]], [[V5]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_b[%c0, %c0] 32768 %b {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<256x64xf16, #shared>, i32, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      // CHECK: [[V7:%.*]] = nvws.semaphore.acquire [[V4]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V8:%.*]]:2 = nvws.semaphore.buffer [[V4]], [[V7]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>
      // CHECK-NEXT: [[V9:%.*]] = ttg.memdesc_trans [[V8]]#1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256>
      %bt = ttg.memdesc_trans %b {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared_t, #smem, mutable>
      // CHECK-NEXT: [[V10:%.*]] = ttng.tc_gen5_mma [[V8]]#0, [[V9]], %{{[-A-Za-z0-9_.$#]+}}[%{{[-A-Za-z0-9_.$#]+}}], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %mma = ttng.tc_gen5_mma %a, %bt, %acc[%tok], %flag, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: "use_token"([[V10]]) {ttg.partition = array<i32: 0>}
      "use_token"(%mma) {ttg.partition = array<i32: 0>} : (!ttg.async.token) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %true : i1
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i1"(%r) : (i1) -> ()
    tt.return
  }
}

//--- insert_semas_meta_fa_fwd.mlir

// Meta flash attention forward (persistent) IR, captured from
// meta-aws-logs/run-22may26-nvws-meta-tmem-crash/passes/
// 062-anonymous_VerifyWarpSpecializationPartitions.mlir (loc() stripped).
// High-coverage exercise of native point-of-use insert-semas with a persistent
// outer loop wrapping a pipelined inner loop. Native POU opens buffer.id=5 at
// its first inner-loop use and closes the recurrence with releases from the
// stage-1 uses; no buffer.id=5 token is carried through either loop.
// buffer.id=4 stays behind its own in-loop gate. The
// buffer.id=2/3 per-iteration accumulators thread tokens through the outer
// loop only (bottom acquire at the final readout) and re-acquire in-body
// inside the inner loop. The Q/K/V descriptor SMEM, stats stores, and epilogue
// O buffers exercise the remaining protocols. Hand-curated, semaphore-focused
// CHECKs.
//
// NOTE: the pass canonicalizes the TMEM encoding aliases, so in the captured
// output #tmem is blockN=128 (the 128x128 acc/result/f16-view buffers) and
// #tmem1 is blockN=1 (the alpha/l_i stats subslices) — the opposite of the
// input aliases below. CHECK lines therefore use the *output* alias names.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
// CHECK: module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32,
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL:   tt.func public @_attn_fwd_persist(
// ASP-LABEL:     tt.func public @_attn_fwd_persist(
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %n_tile_num = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 1.44269502 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %prog_id = tt.get_program_id x : i32
    %num_progs = tt.get_num_programs x : i32
    %total_tiles = arith.muli %Z, %n_tile_num : i32
    %total_tiles_3 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_3, %num_progs : i32
    %0 = arith.remsi %total_tiles_3, %num_progs : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_19 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_19 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %desc_q_4 = arith.muli %Z, %H : i32
    %desc_q_5 = arith.muli %desc_q_4, %c16384_i32 : i32
    %desc_q_6 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_q_7 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_k_8 = tt.make_tensor_descriptor %desc_k, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_v_9 = tt.make_tensor_descriptor %desc_v, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_o_10 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_o_11 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %offset_y = arith.muli %H, %c16384_i32 : i32
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %offs_m0_12 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked>
    %qk_scale = arith.mulf %sm_scale, %cst : f32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %m_ij_13 = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear>
    %qk_14 = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear>

    // The single-buffer Q/K/V SMEM allocs each get a true(EMPTY)/false(FULL)
    // semaphore pair (pending_count = 1).
    // CHECK:           [[Q0:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[Q0_E:%.*]] = nvws.semaphore.create [[Q0]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[Q0_F:%.*]] = nvws.semaphore.create [[Q0]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %q0_0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[Q1:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[Q1_E:%.*]] = nvws.semaphore.create [[Q1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[Q1_F:%.*]] = nvws.semaphore.create [[Q1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %q0_1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[K:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[K_E:%.*]] = nvws.semaphore.create [[K]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[K_F:%.*]] = nvws.semaphore.create [[K]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[V:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[V_E:%.*]] = nvws.semaphore.create [[V]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[V_F:%.*]] = nvws.semaphore.create [[V]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // The buffer.id=4 accumulator-class TMEM allocation is hoisted to a single
    // 1x128x128 alloc whose subslices (alpha/offsetkv stats + acc + f16 view)
    // form one multi-member semaphore group: two released gates (pending_count =
    // 2) plus five false(FULL) phases (pending_count = 1).
    // CHECK:           [[R4:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[R4_IN:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} released = 1 {pending_count = 2 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_E:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} released = 1 {pending_count = 2 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F1:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F2:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F3:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F4:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F5:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    %alpha = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %alpha_15 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 66 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y_16 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 65 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    // The buffer.id=5 accumulator-class TMEM allocation forms the second
    // multi-member group: two released gates (pending_count = 2) plus five
    // false(FULL) semaphores (pending_count = 1).
    // CHECK:           [[R5:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[R5_IN:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} released = 1 {pending_count = 2 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_E:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} released = 1 {pending_count = 2 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F1:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F2:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F3:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F4:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F5:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    %offsetkv_y_17 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 66 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y_18 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 65 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    // Epilogue-store SMEM scratch (%3,%4) each get a true/false pair.
    // CHECK:           [[O0:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[O0_E:%.*]] = nvws.semaphore.create [[O0]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[O0_F:%.*]] = nvws.semaphore.create [[O0]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[O1:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[O1_E:%.*]] = nvws.semaphore.create [[O1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[O1_F:%.*]] = nvws.semaphore.create [[O1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // The two per-iteration acc TMEM allocs (buffer.id 2/3) are hoisted to
    // single-component 1x buffers with a true/false pair each.
    // CHECK:           [[ACC0:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[ACC0_E:%.*]] = nvws.semaphore.create [[ACC0]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[ACC0_F:%.*]] = nvws.semaphore.create [[ACC0]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[ACC1:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[ACC1_E:%.*]] = nvws.semaphore.create [[ACC1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[ACC1_F:%.*]] = nvws.semaphore.create [[ACC1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>

    // Initial outer-gate acquires for R4/R5 (partition 1), followed by the two
    // per-iteration accumulator tokens. Only ACC0/ACC1 thread through the
    // outer loop; the R5 token does not.
    // CHECK:           [[IA_R4:%.*]] = nvws.semaphore.acquire [[R4_E]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[{{.*}}]> -> !ttg.async.token
    // CHECK:           [[IA_R5:%.*]] = nvws.semaphore.acquire [[R5_E]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[{{.*}}]> -> !ttg.async.token
    // CHECK:           [[IA_ACC0:%.*]] = nvws.semaphore.acquire [[ACC0_E]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK:           [[IA_ACC1:%.*]] = nvws.semaphore.acquire [[ACC1_E]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token

    // R5 is not carried through the outer persistent loop. R4 also needs no
    // carried token.
    // CHECK-NOT:       scf.for {{.*}}[[IA_R5]]
    // CHECK:           scf.for {{.*}} iter_args({{.*}}, [[C_ACC0:%.*]] = [[IA_ACC0]], [[C_ACC1:%.*]] = [[IA_ACC1]]) -> (i32, !ttg.async.token, !ttg.async.token)  : i32 {
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_19 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_19, %n_tile_num {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_hz = arith.divsi %tile_idx_19, %n_tile_num {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_z = arith.divsi %off_hz, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_h = arith.remsi %off_hz, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_20 = arith.muli %off_z, %offset_y {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_21 = arith.muli %off_h, %c16384_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_22 = arith.addi %offset_y_20, %offset_y_21 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %qo_offset_y_23 = arith.addi %offset_y_22, %qo_offset_y {ttg.partition = array<i32: 2, 3>} : i32
      %5 = arith.addi %qo_offset_y_23, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %q0 = arith.addi %qo_offset_y_23, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %offs_m0_24 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_25 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_26 = arith.addi %offs_m0_24, %offs_m0 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %offs_m0_27 = arith.addi %offs_m0_25, %offs_m0_12 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>

      // Q0 load: acquire EMPTY, point-of-use buffer feeds the descriptor_load,
      // release FULL with a tma_load arrive.
      // CHECK:           [[Q0_AE:%.*]] = nvws.semaphore.acquire [[Q0_E]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[Q0_BUF:%.*]] = nvws.semaphore.buffer [[Q0_E]], [[Q0_AE]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[Q0_BUF]] {ttg.partition = array<i32: 3>}
      // CHECK:           nvws.semaphore.release [[Q0_F]], [[Q0_AE]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_q_6[%qo_offset_y_23, %c0_i32] 32768 %q0_0 {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Q1 load mirrors Q0.
      // CHECK:           [[Q1_AE:%.*]] = nvws.semaphore.acquire [[Q1_E]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[Q1_BUF:%.*]] = nvws.semaphore.buffer [[Q1_E]], [[Q1_AE]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[Q1_BUF]] {ttg.partition = array<i32: 3>}
      // CHECK:           nvws.semaphore.release [[Q1_F]], [[Q1_AE]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_q_7[%q0, %c0_i32] 32768 %q0_1 {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %qk_0, %qk_0_32 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0, 1, 5>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %qk_1, %qk_1_33 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0, 1, 4>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // The acc_0 / acc_1 init stores reuse the carried ACC0/ACC1 EMPTY tokens
      // (no re-acquire here), write zero into the point-of-use buffer, then
      // release the EMPTY gate for the first in-body acquire in the inner loop.
      // CHECK:           [[ACC0_BUF0:%.*]] = nvws.semaphore.buffer [[ACC0_E]], [[C_ACC0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           ttng.tmem_store %{{.*}}, [[ACC0_BUF0]], %{{.*}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           nvws.semaphore.release [[ACC0_E]], [[C_ACC0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_0, %acc_0_34 = ttng.tmem_alloc %cst_0 {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK:           [[ACC1_BUF0:%.*]] = nvws.semaphore.buffer [[ACC1_E]], [[C_ACC1]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           ttng.tmem_store %{{.*}}, [[ACC1_BUF0]], %{{.*}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           nvws.semaphore.release [[ACC1_E]], [[C_ACC1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_1, %acc_1_35 = ttng.tmem_alloc %cst_0 {buffer.copy = 1 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Outside the inner loop, the K/V FULL tokens consumed by the trailing
      // tc5mma releases are acquired up front (partition 1) on the Q FULL sems.
      // CHECK:           [[K_PRE:%.*]] = nvws.semaphore.acquire [[Q0_F]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[V_PRE:%.*]] = nvws.semaphore.acquire [[Q1_F]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token

      // R5 is not carried through the inner pipelined loop either. The R4
      // inner gate and Q FULL tokens are loop-invariant captures; ACC0/ACC1
      // are re-acquired in-body.
      // CHECK:           [[INNER:%.*]]:5 = scf.for {{.*}} iter_args({{.*}}) -> (i32, tensor<128xf32, {{.*}}>, tensor<128xf32, {{.*}}>, tensor<128xf32, {{.*}}>, tensor<128xf32, {{.*}}>)  : i32 {
      %offsetkv_y_40:9 = scf.for %offsetkv_y_88 = %c0_i32 to %c16384_i32 step %c128_i32 iter_args(%offset_y_89 = %offset_y_22, %arg12 = %cst_2, %arg13 = %cst_1, %qk_0_90 = %qk_0_32, %acc_91 = %acc_0_34, %arg16 = %cst_2, %arg17 = %cst_1, %qk_1_92 = %qk_1_33, %acc_93 = %acc_1_35) -> (i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        // K descriptor load: acquire EMPTY (K_E), point-of-use buffer, release FULL.
        // CHECK:             [[KIN_AE:%.*]] = nvws.semaphore.acquire [[K_E]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK:             [[KIN_BUF:%.*]] = nvws.semaphore.buffer [[K_E]], [[KIN_AE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK:             nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[KIN_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
        // CHECK:             nvws.semaphore.release [[K_F]], [[KIN_AE]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        nvws.descriptor_load %desc_k_8[%offset_y_89, %c0_i32] 32768 %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_97 = ttg.memdesc_reinterpret %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_98 = ttg.memdesc_trans %k_97 {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        // V descriptor load: acquire EMPTY (V_E), point-of-use buffer, release FULL.
        // CHECK:             [[VIN_AE:%.*]] = nvws.semaphore.acquire [[V_E]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK:             [[VIN_BUF:%.*]] = nvws.semaphore.buffer [[V_E]], [[VIN_AE]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK:             nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[VIN_BUF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
        // CHECK:             nvws.semaphore.release [[V_F]], [[VIN_AE]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        nvws.descriptor_load %desc_v_9[%offset_y_89, %c0_i32] 32768 %v {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // First QK MMA (partition 1): acquire and buffer the R4 inner gate,
        // buffer the Q0 FULL token (lhs), then acquire+buffer the inner K FULL
        // token (rhs, transposed). MMA lhs is the Q0 buffer, acc is R4_QK#3.
        // CHECK:             [[R4_QK_A:%.*]] = nvws.semaphore.acquire [[R4_IN]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[R4_QK:%.*]]:5 = nvws.semaphore.buffer [[R4_IN]], [[R4_QK_A]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             [[Q0_QK:%.*]] = nvws.semaphore.buffer [[Q0_F]], [[K_PRE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK:             [[KMMA_AF:%.*]] = nvws.semaphore.acquire [[K_F]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK:             [[KMMA_BUF:%.*]] = nvws.semaphore.buffer [[K_F]], [[KMMA_AF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qk_101 = ttng.tc_gen5_mma %q0_0, %k_98, %qk_0[%qk_0_90], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // CHECK:             ttng.tc_gen5_mma [[Q0_QK]], %{{.*}}, [[R4_QK]]#3[], %{{.*}}, %{{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK:             nvws.semaphore.release [[R4_F1]], [[R4_QK_A]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        // Second QK MMA mirrors the first on the R5 (id=5) acc-class set. Its
        // R5 token is acquired at this first use, not carried into the loop.
        // After it, the inner K_E EMPTY is released using the K consumer token.
        // CHECK:             [[R5_QK_A:%.*]] = nvws.semaphore.acquire [[R5_IN]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[R5_QK:%.*]]:5 = nvws.semaphore.buffer [[R5_IN]], [[R5_QK_A]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             [[Q1_QK:%.*]] = nvws.semaphore.buffer [[Q1_F]], [[V_PRE]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qk_102 = ttng.tc_gen5_mma %q0_1, %k_98, %qk_1[%qk_1_92], %false, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // CHECK:             ttng.tc_gen5_mma [[Q1_QK]], %{{.*}}, [[R5_QK]]#3[], %{{.*}}, %{{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK:             nvws.semaphore.release [[K_E]], [[KMMA_AF]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK:             nvws.semaphore.release [[R5_F1]], [[R5_QK_A]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        // QK tmem_load (partition 5): acquire R4_F1, buffer, load #3.
        // CHECK:             [[QK0_AF:%.*]] = nvws.semaphore.acquire [[R4_F1]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[QK0_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F1]], [[QK0_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_load [[QK0_BUF]]#3[] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>}
        %qk_103, %qk_104 = ttng.tmem_load %qk_0[%qk_101] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_105 = ttg.convert_layout %qk_103 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        // QK tmem_load (partition 4): acquire R5_F1, buffer, load #3.
        // CHECK:             [[QK1_AF:%.*]] = nvws.semaphore.acquire [[R5_F1]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[QK1_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F1]], [[QK1_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_load [[QK1_BUF]]#3[] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>}
        %qk_106, %qk_107 = ttng.tmem_load %qk_1[%qk_102] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_108 = ttg.convert_layout %qk_106 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %m_ij_109 = "tt.reduce"(%qk_105) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_176: f32, %m_ij_177: f32):
          %m_ij_178 = arith.maxnumf %m_ij_176, %m_ij_177 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %m_ij_178 {ttg.partition = array<i32: 5>} : f32
        }) {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_110 = "tt.reduce"(%qk_108) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_176: f32, %m_ij_177: f32):
          %m_ij_178 = arith.maxnumf %m_ij_176, %m_ij_177 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %m_ij_178 {ttg.partition = array<i32: 4>} : f32
        }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_111 = arith.mulf %m_ij_109, %m_ij {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_112 = arith.mulf %m_ij_110, %m_ij_13 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_113 = arith.maxnumf %arg13, %m_ij_111 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_114 = arith.maxnumf %arg17, %m_ij_112 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %qk_115 = arith.mulf %qk_105, %qk {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %qk_116 = arith.mulf %qk_108, %qk_14 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %qk_117 = tt.expand_dims %m_ij_113 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_118 = tt.expand_dims %m_ij_114 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_119 = tt.broadcast %qk_117 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_120 = tt.broadcast %qk_118 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_121 = arith.subf %qk_115, %qk_119 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %qk_122 = arith.subf %qk_116, %qk_120 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %p = math.exp2 %qk_121 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %p_123 = math.exp2 %qk_122 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %alpha_124 = arith.subf %arg13, %m_ij_113 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_125 = arith.subf %arg17, %m_ij_114 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_126 = math.exp2 %alpha_124 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_127 = tt.expand_dims %alpha_126 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %alpha_128 = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} true
        // alpha stats store (partition 5) writes into the R4 subslice #0 buffer
        // already opened by [[QK0_BUF]] (point-of-use), then releases R4 FULL.
        // CHECK:             ttng.tmem_store %{{.*}}, [[QK0_BUF]]#0, %{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
        // CHECK:             nvws.semaphore.release [[R4_F2]], [[QK0_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>}
        // CHECK:             nvws.semaphore.release [[R4_IN]], [[QK0_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>}
        ttng.tmem_store %alpha_127, %alpha, %alpha_128 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
        %alpha_129 = math.exp2 %alpha_125 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_130 = tt.expand_dims %alpha_129 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %alpha_131 = arith.constant {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} true
        // alpha_15 stats store (partition 4) mirrors on R5 subslice #0.
        // CHECK:             ttng.tmem_store %{{.*}}, [[QK1_BUF]]#0, %{{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
        // CHECK:             nvws.semaphore.release [[R5_F2]], [[QK1_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>}
        // CHECK:             nvws.semaphore.release [[R5_IN]], [[QK1_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>}
        ttng.tmem_store %alpha_130, %alpha_15, %alpha_131 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_176: f32, %l_ij_177: f32):
          %l_ij_178 = arith.addf %l_ij_176, %l_ij_177 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %l_ij_178 {ttg.partition = array<i32: 5>} : f32
        }) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_ij_132 = "tt.reduce"(%p_123) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_176: f32, %l_ij_177: f32):
          %l_ij_178 = arith.addf %l_ij_176, %l_ij_177 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %l_ij_178 {ttg.partition = array<i32: 4>} : f32
        }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        // acc_0 tmem_load (partition 0) re-acquires ACC0 EMPTY in-body and
        // buffers under the fresh token.
        // CHECK:             [[ACC0_AE:%.*]] = nvws.semaphore.acquire [[ACC0_E]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:             [[ACC0_LD:%.*]] = nvws.semaphore.buffer [[ACC0_E]], [[ACC0_AE]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             ttng.tmem_load [[ACC0_LD]][] {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 8>, ttg.partition = array<i32: 0>}
        %acc_133, %acc_134 = ttng.tmem_load %acc_0[%acc_91] {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 8>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        // CHECK:             [[ACC1_AE:%.*]] = nvws.semaphore.acquire [[ACC1_E]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:             [[ACC1_LD:%.*]] = nvws.semaphore.buffer [[ACC1_E]], [[ACC1_AE]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             ttng.tmem_load [[ACC1_LD]][] {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 11>, ttg.partition = array<i32: 0>}
        %acc_135, %acc_136 = ttng.tmem_load %acc_1[%acc_93] {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 11>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %18 = tt.reshape %acc_133 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %19 = tt.reshape %acc_135 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %20 = tt.trans %18 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %21 = tt.trans %19 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %outLHS, %outRHS = tt.split %20 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        %outLHS_137, %outRHS_138 = tt.split %21 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        // alpha stats reload (partition 0): acquire R4_F2, buffer, load subslice
        // #0, release the second arrival to R4_IN.
        // CHECK:             [[A0_AF:%.*]] = nvws.semaphore.acquire [[R4_F2]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[A0_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F2]], [[A0_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_load [[A0_BUF]]#0[] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        // CHECK:             nvws.semaphore.release [[R4_IN]], [[A0_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        %alpha_139, %alpha_140 = ttng.tmem_load %alpha[] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %alpha_141 = tt.reshape %alpha_139 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %alpha_142 = ttg.convert_layout %alpha_141 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_143 = tt.expand_dims %alpha_142 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        // alpha_15 stats reload (partition 0): acquire R5_F2, buffer, load, and
        // release the second arrival to R5_IN for the next iteration's POU
        // acquire.
        // CHECK:             [[A1_AF:%.*]] = nvws.semaphore.acquire [[R5_F2]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[A1_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F2]], [[A1_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_load [[A1_BUF]]#0[] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
        // CHECK:             nvws.semaphore.release [[R5_IN]], [[A1_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
        %alpha_144, %alpha_145 = ttng.tmem_load %alpha_15[] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %alpha_146 = tt.reshape %alpha_144 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %alpha_147 = ttg.convert_layout %alpha_146 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_148 = tt.expand_dims %alpha_147 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %acc0_149 = ttg.convert_layout %acc0_143 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_150 = ttg.convert_layout %acc0_148 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_151 = tt.broadcast %acc0_149 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_152 = tt.broadcast %acc0_150 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_153 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS, %acc0_151 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_154 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS_137, %acc0_152 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc1 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS, %acc0_151 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc1_155 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS_138, %acc0_152 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc_156 = tt.join %acc0_153, %acc1 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_157 = tt.join %acc0_154, %acc1_155 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_158 = tt.trans %acc_156 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_159 = tt.trans %acc_157 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_160 = tt.reshape %acc_158 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %acc_161 = tt.reshape %acc_159 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %p_162 = arith.truncf %p {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %p_163 = arith.truncf %p_123 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        // p (f16) store into the R4 f16 view subslice #4 (partition 5) reuses
        // the retained R4_F1 token and releases R4_F3.
        // CHECK:             [[P0_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F1]], [[QK0_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_store %{{.*}}, [[P0_BUF]]#4, %{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             nvws.semaphore.release [[R4_F3]], [[QK0_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>}
        %acc_164 = ttng.tmem_alloc %p_162 {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
        // p_123 (f16) store into R5 f16 view subslice #4 (partition 4) reuses
        // the retained R5_F1 token and releases R5_F3.
        // CHECK:             [[P1_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F1]], [[QK1_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_store %{{.*}}, [[P1_BUF]]#4, %{{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             nvws.semaphore.release [[R5_F3]], [[QK1_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>}
        %acc_165 = ttng.tmem_alloc %p_163 {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
        // acc_0 update store (partition 0) into the in-body acquired ACC0
        // buffer, release ACC0_F on the same token.
        // CHECK:             ttng.tmem_store %{{.*}}, [[ACC0_LD]][], %{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.start = array<i32: 9>, ttg.partition = array<i32: 0>}
        // CHECK:             nvws.semaphore.release [[ACC0_F]], [[ACC0_AE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %acc_166 = ttng.tmem_store %acc_160, %acc_0[%acc_134], %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.start = array<i32: 9>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // CHECK:             ttng.tmem_store %{{.*}}, [[ACC1_LD]][], %{{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 12>, ttg.partition = array<i32: 0>}
        // CHECK:             nvws.semaphore.release [[ACC1_F]], [[ACC1_AE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %acc_167 = ttng.tmem_store %acc_161, %acc_1[%acc_136], %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 12>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // PV MMA #1 (partition 1): buffer R4_F3 (p f16 view #4), acquire ACC0_F
        // FULL, acquire V_F FULL, tc5mma, release ACC0_E.
        // CHECK:             [[PV0_AF:%.*]] = nvws.semaphore.acquire [[R4_F3]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[PV0_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F3]], [[PV0_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             [[PV0_ACC_AF:%.*]] = nvws.semaphore.acquire [[ACC0_F]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:             [[PV0_ACC_BUF:%.*]] = nvws.semaphore.buffer [[ACC0_F]], [[PV0_ACC_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             [[PV0_V_AF:%.*]] = nvws.semaphore.acquire [[V_F]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK:             [[PV0_V_BUF:%.*]] = nvws.semaphore.buffer [[V_F]], [[PV0_V_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK:             ttng.tc_gen5_mma [[PV0_BUF]]#4, [[PV0_V_BUF]], [[PV0_ACC_BUF]][], %{{.*}}, %{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 9>, tmem.start = array<i32: 8, 10>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK:             nvws.semaphore.release [[ACC0_E]], [[PV0_ACC_AF]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %acc_170 = ttng.tc_gen5_mma %acc_164, %v, %acc_0[%acc_166], %true, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 9>, tmem.start = array<i32: 8, 10>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // PV MMA #2 (partition 1): R5_F3 / ACC1_F / re-use V buffer, release the
        // inner V_E EMPTY and ACC1_E.
        // CHECK:             [[PV1_AF:%.*]] = nvws.semaphore.acquire [[R5_F3]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[PV1_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F3]], [[PV1_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             [[PV1_ACC_AF:%.*]] = nvws.semaphore.acquire [[ACC1_F]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:             [[PV1_ACC_BUF:%.*]] = nvws.semaphore.buffer [[ACC1_F]], [[PV1_ACC_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             ttng.tc_gen5_mma [[PV1_BUF]]#4, [[PV0_V_BUF]], [[PV1_ACC_BUF]][], %{{.*}}, %{{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 12>, tmem.start = array<i32: 11, 13>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK:             nvws.semaphore.release [[V_E]], [[PV0_V_AF]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK:             nvws.semaphore.release [[ACC1_E]], [[PV1_ACC_AF]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %acc_171 = ttng.tc_gen5_mma %acc_165, %v, %acc_1[%acc_167], %true, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 12>, tmem.start = array<i32: 11, 13>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg12, %alpha_126 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_172 = arith.mulf %arg16, %alpha_129 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_173 = arith.addf %l_i0, %l_ij {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_174 = arith.addf %l_i0_172, %l_ij_132 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %offsetkv_y_175 = arith.addi %offset_y_89, %c128_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : i32
        // No R5 token is re-acquired at the loop boundary or yielded.
        // CHECK-NOT:         nvws.semaphore.acquire [[R5_IN]]
        // CHECK:             scf.yield {ttg.partition = array<i32: 0, 1, 3, 4, 5>} %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} :
        scf.yield {ttg.partition = array<i32: 0, 1, 3, 4, 5>} %offsetkv_y_175, %l_i0_173, %m_ij_113, %qk_104, %acc_170, %l_i0_174, %m_ij_114, %qk_107, %acc_171 : i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token
      // Inner-loop close: pinned pipelining attrs.
      // CHECK:           } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 1 : i32, tt.separate_epilogue_store = true, ttg.partition = array<i32: 0, 1, 3, 4, 5>, ttg.partition.outputs = {{\[}}array<i32: 3>, array<i32: 5>, array<i32: 5>, array<i32: 4>, array<i32: 4>]}
      } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 1 : i32, tt.separate_epilogue_store = true, ttg.partition = array<i32: 0, 1, 3, 4, 5>, ttg.partition.outputs = [array<i32: 3>, array<i32: 5>, array<i32: 5>, array<i32: 1>, array<i32: 0>, array<i32: 4>, array<i32: 4>, array<i32: 1>, array<i32: 0>]}

      // Post-inner-loop epilogue (still inside the persistent outer loop). The
      // inner Q FULL tokens release back as EMPTY for the next outer iteration.
      // R5 is acquired at the post-inner use, while R4 uses its point-of-use
      // drain; each opens one final FULL semaphore.
      // CHECK:           nvws.semaphore.release [[Q1_E]], [[V_PRE]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK:           nvws.semaphore.release [[Q0_E]], [[K_PRE]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK:           [[R5_POST:%.*]] = nvws.semaphore.acquire [[R5_IN]] {ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           nvws.semaphore.release [[R5_F4]], [[R5_POST]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}

      // Both post-loop stats stores (partition 4) land in the single R5_F4
      // phase: one acquire, one buffer, store #2 then #1.
      // CHECK:           [[OK18_AF:%.*]] = nvws.semaphore.acquire [[R5_F4]] {ttg.partition = array<i32: 4>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           [[OK18_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F4]], [[OK18_AF]] {ttg.partition = array<i32: 4>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
      // CHECK:           ttng.tmem_store %{{.*}}, [[OK18_BUF]]#2, %{{.*}} {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      %offsetkv_y_41 = tt.expand_dims %offsetkv_y_40#6 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_42 = arith.constant {ttg.partition = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_41, %offsetkv_y_18, %offsetkv_y_42 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // offsetkv_y_17 stats store (partition 4) reuses the same token for
      // buffer #1, then releases R5_F5 and the first R5_E arrival.
      // CHECK:           ttng.tmem_store %{{.*}}, [[OK18_BUF]]#1, %{{.*}} {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      // CHECK:           nvws.semaphore.release [[R5_F5]], [[OK18_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      // CHECK:           nvws.semaphore.release [[R5_E]], [[OK18_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      %offsetkv_y_43 = tt.expand_dims %offsetkv_y_40#5 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_44 = arith.constant {ttg.partition = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_43, %offsetkv_y_17, %offsetkv_y_44 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK:           [[R4_DRAIN:%.*]] = nvws.semaphore.acquire [[R4_IN]] {ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           nvws.semaphore.release [[R4_F4]], [[R4_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}

      // Both post-loop stats stores (partition 5) land in the single R4_F4
      // phase: acquire, buffer, store #2 then #1.
      // CHECK:           [[OK16_AF:%.*]] = nvws.semaphore.acquire [[R4_F4]] {ttg.partition = array<i32: 5>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           [[OK16_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F4]], [[OK16_AF]] {ttg.partition = array<i32: 5>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
      // CHECK:           ttng.tmem_store %{{.*}}, [[OK16_BUF]]#2, %{{.*}} {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      %offsetkv_y_45 = tt.expand_dims %offsetkv_y_40#2 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_46 = arith.constant {ttg.partition = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_45, %offsetkv_y_16, %offsetkv_y_46 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // offsetkv_y stats store (partition 5) reuses the same token for buffer
      // #1, then releases R4_F5 and the first R4_E arrive.
      // CHECK:           ttng.tmem_store %{{.*}}, [[OK16_BUF]]#1, %{{.*}} {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      // CHECK:           nvws.semaphore.release [[R4_F5]], [[OK16_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>}
      // CHECK:           nvws.semaphore.release [[R4_E]], [[OK16_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>}
      %offsetkv_y_47 = tt.expand_dims %offsetkv_y_40#1 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_48 = arith.constant {ttg.partition = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_47, %offsetkv_y, %offsetkv_y_48 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // offsetkv_y reload (partition 0): acquire R4_F5, buffer, load #1.
      // CHECK:           [[OKR_AF:%.*]] = nvws.semaphore.acquire [[R4_F5]] {ttg.partition = array<i32: 0>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           [[OKR_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F5]], [[OKR_AF]] {ttg.partition = array<i32: 0>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
      // CHECK:           ttng.tmem_load [[OKR_BUF]]#1[] {ttg.partition = array<i32: 0>}
      %offsetkv_y_49, %offsetkv_y_50 = ttng.tmem_load %offsetkv_y[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_51 = tt.reshape %offsetkv_y_49 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_52 = ttg.convert_layout %offsetkv_y_51 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0 = math.log2 %offsetkv_y_52 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      // offsetkv_y_17 reload (partition 0): acquire R5_F5, buffer, load #1.
      // CHECK:           [[OK17R_AF:%.*]] = nvws.semaphore.acquire [[R5_F5]] {ttg.partition = array<i32: 0>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           [[OK17R_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F5]], [[OK17R_AF]] {ttg.partition = array<i32: 0>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
      // CHECK:           ttng.tmem_load [[OK17R_BUF]]#1[] {ttg.partition = array<i32: 0>}
      %offsetkv_y_53, %offsetkv_y_54 = ttng.tmem_load %offsetkv_y_17[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_55 = tt.reshape %offsetkv_y_53 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_56 = ttg.convert_layout %offsetkv_y_55 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_57 = math.log2 %offsetkv_y_56 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      // offsetkv_y_16 reload (partition 0): load #2 from the R4_F5 buffer,
      // then the second R4_E arrive.
      // CHECK:           ttng.tmem_load [[OKR_BUF]]#2[] {ttg.partition = array<i32: 0>}
      // CHECK:           nvws.semaphore.release [[R4_E]], [[OKR_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %offsetkv_y_58, %offsetkv_y_59 = ttng.tmem_load %offsetkv_y_16[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_60 = tt.reshape %offsetkv_y_58 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_61 = ttg.convert_layout %offsetkv_y_60 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_62 = arith.addf %offsetkv_y_61, %m_i0 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      // offsetkv_y_18 reload (partition 0): load #2 from the R5_F5 buffer,
      // then the second R5_E arrive.
      // CHECK:           ttng.tmem_load [[OK17R_BUF]]#2[] {ttg.partition = array<i32: 0>}
      // CHECK:           nvws.semaphore.release [[R5_E]], [[OK17R_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %offsetkv_y_63, %offsetkv_y_64 = ttng.tmem_load %offsetkv_y_18[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_65 = tt.reshape %offsetkv_y_63 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_66 = ttg.convert_layout %offsetkv_y_65 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_67 = arith.addf %offsetkv_y_66, %m_i0_57 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %acc0 = tt.expand_dims %offsetkv_y_52 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_68 = tt.expand_dims %offsetkv_y_56 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_69 = tt.broadcast %acc0 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %acc0_70 = tt.broadcast %acc0_68 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      // Final acc_0 readout (partition 0): bottom re-acquire of ACC0 EMPTY
      // (carried to the next outer iteration via yield), buffer, load.
      // CHECK:           [[ACCF0_AE:%.*]] = nvws.semaphore.acquire [[ACC0_E]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK:           [[ACCF0_BUF:%.*]] = nvws.semaphore.buffer [[ACC0_E]], [[ACCF0_AE]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           ttng.tmem_load [[ACCF0_BUF]][] {tmem.end = array<i32: 10>, ttg.partition = array<i32: 0>}
      %acc, %acc_71 = ttng.tmem_load %acc_0[%offsetkv_y_40#4] {tmem.end = array<i32: 10>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_72 = ttg.convert_layout %acc {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      // Final acc_1 readout (partition 0) mirrors acc_0.
      // CHECK:           [[ACCF1_AE:%.*]] = nvws.semaphore.acquire [[ACC1_E]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK:           [[ACCF1_BUF:%.*]] = nvws.semaphore.buffer [[ACC1_E]], [[ACCF1_AE]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           ttng.tmem_load [[ACCF1_BUF]][] {tmem.end = array<i32: 13>, ttg.partition = array<i32: 0>}
      %acc_73, %acc_74 = ttng.tmem_load %acc_1[%offsetkv_y_40#8] {tmem.end = array<i32: 13>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_75 = ttg.convert_layout %acc_73 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %acc0_76 = arith.divf %acc_72, %acc0_69 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %acc0_77 = arith.divf %acc_75, %acc0_70 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %m_ptrs0 = arith.muli %off_hz, %c16384_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %m_ptrs0_78 = tt.addptr %M, %m_ptrs0 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : !tt.ptr<f32>, i32
      %m_ptrs0_79 = tt.splat %m_ptrs0_78 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_80 = tt.splat %m_ptrs0_78 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_81 = tt.addptr %m_ptrs0_79, %offs_m0_26 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %m_ptrs0_82 = tt.addptr %m_ptrs0_80, %offs_m0_27 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %6 = ttg.convert_layout %m_i0_62 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      %7 = ttg.convert_layout %m_i0_67 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      tt.store %m_ptrs0_81, %6 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      tt.store %m_ptrs0_82, %7 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      // Epilogue O0 local_store (partition 0): acquire O0 EMPTY, point-of-use
      // buffer, store, release O0 FULL.
      // CHECK:           [[O0_AE:%.*]] = nvws.semaphore.acquire [[O0_E]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[O0_BUF:%.*]] = nvws.semaphore.buffer [[O0_E]], [[O0_AE]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           ttg.local_store %{{.*}}, [[O0_BUF]] {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           nvws.semaphore.release [[O0_F]], [[O0_AE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %8 = arith.truncf %acc0_76 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      ttg.local_store %8, %3 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Epilogue O1 local_store (partition 0) mirrors O0.
      // CHECK:           [[O1_AE:%.*]] = nvws.semaphore.acquire [[O1_E]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[O1_BUF:%.*]] = nvws.semaphore.buffer [[O1_E]], [[O1_AE]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           ttg.local_store %{{.*}}, [[O1_BUF]] {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           nvws.semaphore.release [[O1_F]], [[O1_AE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %10 = arith.truncf %acc0_77 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      ttg.local_store %10, %4 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Epilogue O0 local_load (partition 2): acquire O0 FULL, buffer and
      // load. Ownership remains live through its descriptor store below.
      // CHECK:           [[O0L_AF:%.*]] = nvws.semaphore.acquire [[O0_F]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[O0L_BUF:%.*]] = nvws.semaphore.buffer [[O0_F]], [[O0L_AF]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           ttg.local_load [[O0L_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %13 = ttg.local_load %3 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %14 = ttg.convert_layout %13 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      // Epilogue O1 local_load (partition 2) mirrors O0. Each empty release
      // follows the descriptor store that completes that channel's read.
      // CHECK:           [[O1L_AF:%.*]] = nvws.semaphore.acquire [[O1_F]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[O1L_BUF:%.*]] = nvws.semaphore.buffer [[O1_F]], [[O1L_AF]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           ttg.local_load [[O1L_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      // CHECK:           tt.descriptor_store
      // CHECK:           nvws.semaphore.release [[O0_E]], [[O0L_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK:           tt.descriptor_store
      // CHECK:           nvws.semaphore.release [[O1_E]], [[O1L_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %16 = ttg.local_load %4 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %17 = ttg.convert_layout %16 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_10[%qo_offset_y_23, %c0_i32], %14 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared>, tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_11[%5, %c0_i32], %17 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared>, tensor<128x128xf16, #blocked1>
      %tile_idx_87 = arith.addi %tile_idx_19, %num_progs {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      // Outer-loop end bridges R4 and R5 back to their point-of-use gates
      // (two arrivals each). Only the ACC0/ACC1 readout tokens ride the yield.
      // CHECK:           [[OX_R4:%.*]] = nvws.semaphore.acquire [[R4_E]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           nvws.semaphore.release [[R4_IN]], [[OX_R4]] [#nvws.async_op<none>] {arrive_count = 2 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK:           [[OX_R5:%.*]] = nvws.semaphore.acquire [[R5_E]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           nvws.semaphore.release [[R5_IN]], [[OX_R5]] [#nvws.async_op<none>] {arrive_count = 2 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK:           scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %{{.*}}, [[ACCF0_AE]], [[ACCF1_AE]] : i32, !ttg.async.token, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %tile_idx_87 : i32
    // Outer-loop close: pinned warp-specialize attrs (stages, tag, types).
    // CHECK:           } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, ttg.partition.outputs = {{.*}}, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3, 4, 5>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_mixed_copy_error.mlir

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @mixed_copy_reject(%lb: i32, %ub: i32, %step: i32) {
    %a = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 77 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: nvws-insert-semas: TMEM allocations sharing buffer.id 77 have conflicting buffer.copy values 1 and 2
    // CHECK: first buffer.copy value is 1
    %b = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 77 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0>,
       ttg.partition.stages = [0 : i32], ttg.partition.types = ["default"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_mixed_overlap_members.mlir

// Dedicated mirror of the meta-FA stats group (GROUP buffer.id=4 in
// insert_semas_meta_fa_fwd: m0[64,65) m1[66,67) m2[65,66) m3[0,128)
// m4[0,64)): many members on ONE backing buffer, with a MIX of
// overlapping and non-overlapping extents. A spanning member (the FA
// accumulator) covers the whole buffer, so the planner keeps every
// member on one backing and the group synchronizes as a single unit.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#half_blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#col_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // memory:   0        64  65  66  67      128
  // m3 (acc)  [==============================)    <- spans everything
  // m4 (p)    [========)
  // m0 (alpha)         [--)
  // m2 (l)                 [--)
  // m1 (m)                     [--)
  //
  // conflict graph (who overlaps whom):
  //
  //       m4 --- m3 --- m0       m3 is the HUB: it overlaps every
  //               | \            sliver, so every sliver must sync
  //              m2  m1          with m3's writes/reads.
  //                              No sliver<->sliver edge exists!
  //
  // m3 spans and bridges the whole group into one backing. One carrier
  // chain threads it: every sliver's W/R weaves acquire/release against
  // the shared semaphores, but slivers never sync with each other
  // directly - overlap is priced per shared piece.
  //
  // Owners form a ring: slivers W{1}->R{2}, acc W{2}->R{0}, p W{0}->R{1}.
  // CHECK-LABEL: @tmem_mixed_overlap_spanning_member
  tt.func @tmem_mixed_overlap_spanning_member(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #half_blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #col_blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = ttng.tmem_subslice [[V1]] {offset = 0 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V3:%.*]] = ttg.memdesc_reinterpret [[V2]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = ttng.tmem_subslice [[V1]] {offset = 65 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V5:%.*]] = ttg.memdesc_reinterpret [[V4]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V6:%.*]] = ttng.tmem_subslice [[V1]] {offset = 66 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V7:%.*]] = ttg.memdesc_reinterpret [[V6]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = ttng.tmem_subslice [[V1]] {offset = 64 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V9:%.*]] = ttg.memdesc_reinterpret [[V8]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V10:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] released = 3 {pending_count = 2 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V11:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V13:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V15:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V16:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V17:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V19:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V18:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // m0: alpha sliver at column 64, produced by {1}, consumed by {2}.
      %alpha = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V20:%.*]] = nvws.semaphore.acquire [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V21:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V21]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %alpha, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V11]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V11]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V23:%.*]]:5 = nvws.semaphore.buffer [[V11]], [[V22]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V23]]#0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_alpha"(%av) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m1: m sliver at column 66, produced by {1}, consumed by {2}.
      %m = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 66 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V25:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V25]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %m, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V13]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V26:%.*]] = nvws.semaphore.acquire [[V13]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V27:%.*]]:5 = nvws.semaphore.buffer [[V13]], [[V26]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V27]]#1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %mv, %mt = ttng.tmem_load %m[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_m"(%mv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m2: l sliver at column 65, produced by {1}, consumed by {2}.
      %l = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 65 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V29:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V29]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %l, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V15]], [[V20]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V30:%.*]] = nvws.semaphore.acquire [[V15]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V31:%.*]]:5 = nvws.semaphore.buffer [[V15]], [[V30]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V31]]#2[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %lv, %lt = ttng.tmem_load %l[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_l"(%lv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m3: the spanning accumulator [0,128), produced by {2}, consumed
      // by {0}. It overlaps ALL other members.
      %acc, %tacc = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V32:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V31]]#3[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %acc0 = ttng.tmem_store %cst, %acc[%tacc], %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V16]], [[V30]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V10]], [[V30]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V33:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V34:%.*]]:5 = nvws.semaphore.buffer [[V16]], [[V33]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V34]]#3[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %accv, %acct = ttng.tmem_load %acc[%acc0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V10]], [[V33]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_acc"(%accv) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      // m4: p at [0,64) - disjoint from every sliver, overlaps only the
      // accumulator. Produced by {0}, consumed by {1}.
      %p = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V34]]#4, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      ttng.tmem_store %cst64, %p, %true {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #half_blocked> -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V17]], [[V33]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V35:%.*]] = nvws.semaphore.acquire [[V17]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V36:%.*]]:5 = nvws.semaphore.buffer [[V17]], [[V35]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V36]]#4[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      %pv, %pt = ttng.tmem_load %p[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #half_blocked>
      "use_p"(%pv) {ttg.partition = array<i32: 1>} : (tensor<128x64xf32, #half_blocked>) -> ()

      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_multi_component_error.mlir

// Two allocations share buffer.id=402 but occupy disjoint, non-overlapping
// column ranges ([0,128) and [256,384)) with no covering member, so the
// piece table splits into two connected components. The memory planner never
// emits this (reusers are stacked within their owner's columns), so InsertSemas
// rejects it rather than mis-synchronizing the single-component protocol.
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @disjoint_members_reject(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK: buffer.id group has disjoint pieces (more than one connected component)
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttg.local_alloc %cst0 {buffer.id = 402 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %av = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use_a"(%av) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      %b = ttg.local_alloc %cst1 {buffer.id = 402 : i32, buffer.offset = 256 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %bv = ttg.local_load %b {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use_b"(%bv) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_nested_carrier.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @outer_sourceful_alloc_inner_loop_reentry
  tt.func @outer_sourceful_alloc_inner_loop_reentry(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V8:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V7:%.*]] = [[V5]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // The inner loop carries no semaphore token: partition 1 acquires [[V4]]
      // at the point of use adjacent to the MMA on every iteration.
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load1"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load2"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[V11:%.*]] = nvws.semaphore.buffer [[V4]], [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V11]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[V3]], [[V10]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V3]], [[V12]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V13]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V4]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // Post-loop bridge: partition 1 re-acquires [[V4]] after the last in-loop
      // read and releases [[V2]]; partition 0 then reads the final value and
      // carries the fresh [[V2]] permit to the next outer iteration.
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: nvws.semaphore.release [[V2]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V2]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V16]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      // CHECK: scf.yield {{.*}}[[V15]]
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Completion differs between the conditional's stage-1 path and its
  // passthrough path.  The inner loop carries no token: the first stage-0 MMA
  // acquires at the point of use.  The conditional itself returns an owner-1
  // token: the then-path hands the partition-0 read back to owner 1, and the
  // else-path passes the stage-0 MMA token through unchanged.
  // CHECK-LABEL: @branch_completion_requires_carrier
  tt.func @branch_completion_requires_carrier(
      %cond: i1, %lb: i32, %ub: i32, %step: i32,
      %lhs: !ttg.memdesc<128x64xf32, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf32, #shared, #smem>) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 920 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[ENTRY:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[TO_BRANCH_READ:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[BACK_TO_WRITER:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[TO_POST_READ:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[LOOP_BACK:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[OUTER_INPUT:%.*]] = nvws.semaphore.acquire [[ENTRY]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[OUTER_RESULTS:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[OUTER_SCALAR:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_TOKEN:%.*]] = [[OUTER_INPUT]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %i = %lb to %ub step %step iter_args(%tile = %c0) -> (i32) : i32 {
      // CHECK: [[OUTER_BUFFER:%.*]] = nvws.semaphore.buffer [[ENTRY]], [[OUTER_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_BUFFER]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[LOOP_BACK]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc, %tok = ttng.tmem_alloc %cst {buffer.copy = 1 : i32, buffer.id = 920 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner = scf.for %j = %lb to %ub step %step iter_args(%iter = %tok) -> (!ttg.async.token) : i32 {
        // CHECK: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[LOOP_BACK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[INNER_BUFFER:%.*]] = nvws.semaphore.buffer [[LOOP_BACK]], [[INNER_TOKEN]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[INNER_BUFFER]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma0 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%iter], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

        // The conditional returns owner 1 on both paths.
        // CHECK: [[BRANCH_TOKEN:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
        %branch = scf.if %cond -> (!ttg.async.token) {
          // CHECK: [[BRANCH_BUFFER:%.*]] = nvws.semaphore.buffer [[LOOP_BACK]], [[INNER_TOKEN]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK-NEXT: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[BRANCH_BUFFER]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK-NEXT: nvws.semaphore.release [[TO_BRANCH_READ]], [[INNER_TOKEN]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          %mma1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%mma0], %true, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: [[BRANCH_READ_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_BRANCH_READ]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK-NEXT: [[BRANCH_READ_BUFFER:%.*]] = nvws.semaphore.buffer [[TO_BRANCH_READ]], [[BRANCH_READ_TOKEN]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[BRANCH_READ_BUFFER]][] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
          // CHECK-NEXT: nvws.semaphore.release [[BACK_TO_WRITER]], [[BRANCH_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          %value0, %read0 = ttng.tmem_load %acc[%mma1] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          "consume.branch"(%value0) {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          // CHECK: [[RETURN_TOKEN:%.*]] = nvws.semaphore.acquire [[BACK_TO_WRITER]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
          // CHECK-NEXT: scf.yield {{.*}}[[RETURN_TOKEN]] : !ttg.async.token
          scf.yield {ttg.partition = array<i32: 0, 1>} %read0 : !ttg.async.token
        } else {
          // The passthrough path returns the unchanged stage-0 MMA token.
          // CHECK: } else {
          // CHECK-NEXT: scf.yield {{.*}}[[INNER_TOKEN]] : !ttg.async.token
          scf.yield {ttg.partition = array<i32: 0, 1>} %mma0 : !ttg.async.token
        // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}

        // CHECK: nvws.semaphore.release [[TO_POST_READ]], [[BRANCH_TOKEN]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[POST_READ_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_POST_READ]] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[POST_READ_BUFFER:%.*]] = nvws.semaphore.buffer [[TO_POST_READ]], [[POST_READ_TOKEN]] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[POST_READ_BUFFER]][] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        // CHECK-NEXT: nvws.semaphore.release [[LOOP_BACK]], [[POST_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %value1, %read1 = ttng.tmem_load %acc[%branch] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume"(%value1) {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read1 : !ttg.async.token
      // CHECK: } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // Post-loop bridge: partition 1 re-acquires LOOP_BACK after the last
      // in-loop read and releases ENTRY; partition 0 reads the final value
      // under a fresh ENTRY permit carried to the next outer iteration.
      // CHECK: [[POST_LOOP_TOKEN:%.*]] = nvws.semaphore.acquire [[LOOP_BACK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: nvws.semaphore.release [[ENTRY]], [[POST_LOOP_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[NEXT_OUTER_TOKEN:%.*]] = nvws.semaphore.acquire [[ENTRY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[NEXT_OUTER_BUFFER:%.*]] = nvws.semaphore.buffer [[ENTRY]], [[NEXT_OUTER_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[NEXT_OUTER_BUFFER]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "consume.post"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.yield {{.*}}[[NEXT_OUTER_TOKEN]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "consume.outer"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The recurrence acquire is constructed at the point of use with the stage-0
  // MMA's own schedule, so no stale scheduling arc is left from the stage-1
  // read to the next stage-0 MMA.  The post-loop bridge stays at the
  // partition-1 boundary instead of moving to the partition-0 final read.
  // CHECK-LABEL: @scheduled_relocated_acquire_boundaries
  tt.func @scheduled_relocated_acquire_boundaries(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V8:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V7:%.*]] = [[V5]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // The inner loop carries no token; the acquire sits next to the MMA.
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load1"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load2"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V4]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[V11:%.*]] = nvws.semaphore.buffer [[V4]], [[V10]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V11]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[V3]], [[V10]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[V13:%.*]] = nvws.semaphore.buffer [[V3]], [[V12]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V13]][] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %read_tok = ttng.tmem_load %acc[%mma] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V4]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use"(%val) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      // CHECK: } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // Post-loop bridge at the partition-1 boundary.
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: nvws.semaphore.release [[V2]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[V16:%.*]] = nvws.semaphore.buffer [[V2]], [[V15]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK-NEXT: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V16]][] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%out) {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      // CHECK: scf.yield {{.*}}[[V15]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @three_level_reentry_without_post_access
  tt.func @three_level_reentry_without_post_access(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V5:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[V7:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Neither the middle nor the inner loop carries a token.
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
      %middle:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%mid = %c0_i32, %mtok = %tok) -> (i32, !ttg.async.token) : i32 {
        // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
        %inner = scf.for %iv2 = %lb to %ub step %step iter_args(%tok1 = %mtok) -> (!ttg.async.token) : i32 {
          %lhs = "load1"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
          %rhs = "load2"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
          // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK-NEXT: [[V10:%.*]] = nvws.semaphore.buffer [[V4]], [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V10]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: nvws.semaphore.release [[V3]], [[V9]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V3]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
          %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          // CHECK: nvws.semaphore.release [[V4]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
        // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

        %mid_next = arith.addi %mid, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        scf.yield {ttg.partition = array<i32: 0, 1>} %mid_next, %inner : i32, !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}

      %next = arith.addi %tile, %middle#0 {ttg.partition = array<i32: 0>} : i32
      // Without a post-loop access, partition 1 bridges from the last in-loop
      // read straight to the [[V2]] release for the next outer iteration.
      // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: nvws.semaphore.release [[V2]], [[V13]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @three_level_sourceful_alloc_reentry
  tt.func @three_level_sourceful_alloc_reentry(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // The innermost recurrence gets its own initial permit.  OUTER_EMPTY still
    // carries the sourceful allocation across outer iterations.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[INNER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32}
    // CHECK: [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32}
    // CHECK: [[INNER_FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    // CHECK: [[MIDDLE_FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    // CHECK: [[OUTER_TO_MIDDLE:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    // CHECK: [[OUTER_ENTRY:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: [[OUTER:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[OUTER_IV:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_TOKEN:%.*]] = [[OUTER_ENTRY]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[OUTER_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_BUF]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>}
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Partition 1 blocks on the store handoff before the middle loop; the
      // acquired token itself is unused, and the middle loop carries no token.
      // CHECK: nvws.semaphore.release [[OUTER_TO_MIDDLE]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.acquire [[OUTER_TO_MIDDLE]] {ttg.partition = array<i32: 1>}
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
      %middle:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%mid = %c0_i32, %mtok = %tok) -> (i32, !ttg.async.token) : i32 {
        // The innermost loop carries no semaphore token; its EMPTY acquire is
        // adjacent to the MMA on every iteration.
        // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
        %inner = scf.for %iv2 = %lb to %ub step %step iter_args(%tok1 = %mtok) -> (!ttg.async.token) : i32 {
          %lhs = "load1"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
          %rhs = "load2"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
          // CHECK: [[INNER_ACQ:%.*]] = nvws.semaphore.acquire [[INNER_EMPTY]] {ttg.partition = array<i32: 1>}
          // CHECK-NEXT: [[INNER_BUF:%.*]] = nvws.semaphore.buffer [[INNER_EMPTY]], [[INNER_ACQ]] {ttg.partition = array<i32: 1>}
          // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[INNER_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
          %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: nvws.semaphore.release [[INNER_FULL]], [[INNER_ACQ]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
          // CHECK: [[INNER_READ:%.*]] = nvws.semaphore.acquire [[INNER_FULL]] {ttg.partition = array<i32: 0>}
          // CHECK: [[INNER_READ_BUF:%.*]] = nvws.semaphore.buffer [[INNER_FULL]], [[INNER_READ]] {ttg.partition = array<i32: 0>}
          // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[INNER_READ_BUF]][] {ttg.partition = array<i32: 0>}
          %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          // CHECK: nvws.semaphore.release [[INNER_EMPTY]], [[INNER_READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
          "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

        // CHECK: [[FINAL_INNER:%.*]] = nvws.semaphore.acquire [[INNER_EMPTY]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[MIDDLE_FULL]], [[FINAL_INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK: [[MIDDLE_READ:%.*]] = nvws.semaphore.acquire [[MIDDLE_FULL]] {ttg.partition = array<i32: 0>}
        // CHECK: [[MIDDLE_READ_BUF:%.*]] = nvws.semaphore.buffer [[MIDDLE_FULL]], [[MIDDLE_READ]] {ttg.partition = array<i32: 0>}
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[MIDDLE_READ_BUF]][] {ttg.partition = array<i32: 0>}
        %mid_out, %mid_tok = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[OUTER_TO_MIDDLE]], [[MIDDLE_READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        "use"(%mid_out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        %mid_next = arith.addi %mid, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        // The middle regain re-arms INNER_EMPTY for the next inner loop; it is
        // no longer yielded because the middle loop carries no token.
        // CHECK: [[MIDDLE_REGAIN:%.*]] = nvws.semaphore.acquire [[OUTER_TO_MIDDLE]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[INNER_EMPTY]], [[MIDDLE_REGAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        scf.yield {ttg.partition = array<i32: 0, 1>} %mid_next, %mid_tok : i32, !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}

      // Post-middle bridge: a fresh INNER_EMPTY acquire anchors the
      // OUTER_EMPTY release for the outer recurrence.
      // CHECK: [[POST_MIDDLE:%.*]] = nvws.semaphore.acquire [[INNER_EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[OUTER_EMPTY]], [[POST_MIDDLE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[OUTER_READ:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: [[OUTER_READ_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_READ]] {ttg.partition = array<i32: 0>}
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[OUTER_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %out, %out_tok = ttng.tmem_load %acc[%middle#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %middle#0 {ttg.partition = array<i32: 0>} : i32
      // CHECK: scf.yield {{.*}}[[OUTER_READ]]
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_nested_region_access.mlir

// The pass inserts ZERO semaphores for every nested-region access pattern
// below. Each function asserts that no nvws.semaphore op is emitted.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 1: depth-2 nesting, outer = scf.for (WS), inner = scf.if.
  // Access is inside the then-region. Verify outer for and inner if are
  // both annotated in OWNERSHIP-DAG via transitive-event propagation;
  // the "something" arith op contributes no row.
  // CHECK-LABEL: tt.func @for_outer_if_inner_access
  // CHECK-NOT: nvws.semaphore
  tt.func @for_outer_if_inner_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 800 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      %something = arith.addi %iv, %iv {ttg.partition = array<i32: 0>} : i32
      scf.if %cond {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 2: depth-2 nesting, outer = scf.if at function scope (outside
  // any WS loop), inner = scf.for (WS-tagged). Access is in the for body.
  // Verify the outer if is annotated transitively, AND its annotation
  // uses tagged display `{@0.X}` because it is anchored outside the WS
  // loop.
  // CHECK-LABEL: tt.func @if_outer_for_inner_access
  // CHECK-NOT: nvws.semaphore
  tt.func @if_outer_for_inner_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 801 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.if %cond {
      %something = arith.constant 42 : i32
      scf.for %iv = %lb to %ub step %step : i32 {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {tt.warp_specialize, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 3: depth-2 if->if nesting, both inside a WS-tagged scf.for.
  // Access is in the inner if's then-region. Both outer and inner if
  // must be annotated via transitive access; the outer if has no direct
  // event.
  // CHECK-LABEL: tt.func @if_outer_if_inner_access
  // CHECK-NOT: nvws.semaphore
  tt.func @if_outer_if_inner_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 802 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.if %cond {
        %something = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
        scf.if %cond {
          %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
          ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 0, 1>}
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 4: depth-2 for->for nesting. Outer scf.for is WS-tagged,
  // inner scf.for is plain. Access is in the inner body. Verify outer
  // for is annotated transitively via the inner for's annotation.
  // CHECK-LABEL: tt.func @for_outer_for_inner_access
  // CHECK-NOT: nvws.semaphore
  tt.func @for_outer_for_inner_access(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 803 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      %something = arith.addi %iv, %iv {ttg.partition = array<i32: 0>} : i32
      scf.for %jv = %lb to %ub step %step : i32 {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 5: depth-3 alternating for->if->for, access in the inner
  // for body. Outer scf.for is WS-tagged. Every ancestor on the path
  // (outer for, if, inner for) must be annotated transitively.
  // CHECK-LABEL: tt.func @triple_for_if_for_access
  // CHECK-NOT: nvws.semaphore
  tt.func @triple_for_if_for_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 804 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.if %cond {
        scf.for %jv = %lb to %ub step %step : i32 {
          %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
          ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 0, 1>}
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 6: depth-3 alternating if->for->if, outermost scf.if at
  // function scope. The middle scf.for is WS-tagged. Access is in the
  // innermost if. The outer if is anchored outside the WS loop and must
  // show tagged display `{@0.X}` on its branches. All three regioned
  // ops on the path must be annotated.
  // CHECK-LABEL: tt.func @triple_if_for_if_access
  // CHECK-NOT: nvws.semaphore
  tt.func @triple_if_for_if_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 805 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.if %cond {
      scf.for %iv = %lb to %ub step %step : i32 {
        scf.if %cond {
          %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
          ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1>}
      } {tt.warp_specialize, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 7: depth-4 alternating for->if->for->if, with WS-tagged
  // outermost scf.for and access at the leaf. Verifies arbitrary
  // depth: every ancestor on the path is annotated, every sibling
  // empty region (else of each if) is reconciled.
  // CHECK-LABEL: tt.func @quad_for_if_for_if_access
  // CHECK-NOT: nvws.semaphore
  tt.func @quad_for_if_for_if_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 806 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.if %cond {
        scf.for %jv = %lb to %ub step %step : i32 {
          scf.if %cond {
            %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
            ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          } {ttg.partition = array<i32: 0, 1>}
        } {ttg.partition = array<i32: 0, 1>}
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 8: sibling scf.ifs at the same nesting level inside a
  // WS-tagged scf.for. Only one of the two ifs has access. The other
  // must be absent from the OWNERSHIP-DAG and ACCESS-DAG entirely;
  // the one with access must appear with proper annotation.
  // CHECK-LABEL: tt.func @sibling_if_only_one_with_access
  // CHECK-NOT: nvws.semaphore
  tt.func @sibling_if_only_one_with_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 807 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.if %cond {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 0, 1>}
      scf.if %cond {
        %side = "side_effect"() {ttg.partition = array<i32: 0>} : () -> i32
      } {ttg.partition = array<i32: 0>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 9: sibling scf.fors at the same nesting level inside a
  // WS-tagged outer scf.for. Only the first inner for has access. The
  // second inner for must be absent from both DAGs.
  // CHECK-LABEL: tt.func @sibling_for_only_one_with_access
  // CHECK-NOT: nvws.semaphore
  tt.func @sibling_for_only_one_with_access(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 808 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.for %jv = %lb to %ub step %step : i32 {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 0, 1>}
      scf.for %kv = %lb to %ub step %step : i32 {
        %side = "side_effect"() {ttg.partition = array<i32: 0>} : () -> i32
      } {ttg.partition = array<i32: 0>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_nested_ws_inner_loop.mlir

// M2 PIN (nested-loop hold-rule extension, plan v3).
//
// A WS-tagged OUTER loop wrapping a non-WS INNER loop with ONE inner-confined
// ping-pong buffer (the persistent-FA qk shape, design v3 §8): tc_gen5_mma
// (partition 1) writes the tmem accumulator, tmem_load (partition 0) reads it,
// BOTH confined to the inner loop; nothing touches the buffer at the outer
// level.
//
// This pins the enabled native shape: no root entry acquire, the original
// token iter_args dropped from both loops, and the writer acquire at the
// first toucher inside the inner loop.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL:   tt.func @nested_ws_inner_loop(
  tt.func @nested_ws_inner_loop(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // Alloc grows to 1x (single-buffered) and loses its token result; the two
    // ping-pong semaphores are created at the root: released EMPTY gates the
    // writer, blocked FULL the reader.
    // CHECK:           [[BUF_ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[EMPTY:%.*]] = nvws.semaphore.create [[BUF_ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[FULL:%.*]] = nvws.semaphore.create [[BUF_ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %res, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Both loops drop their token iter_args entirely: no acquire hoisted to
    // the root, no carrier threaded through either loop.
    // CHECK:           scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    %o = scf.for %iv0 = %lb to %ub step %step iter_args(%t0 = %tok) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT:        scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      %i = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %t0) -> (!ttg.async.token) : i32 {
        // CHECK:               {{.*}} = "loadA"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %sA = "loadA"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        // CHECK:               {{.*}} = "loadB"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %sB = "loadB"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        // Writer (partition 1): acquire EMPTY at first toucher, buffer, MMA, release FULL.
        // CHECK:               [[WTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[WBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WTOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               {{.*}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[WBUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %sA, %sB, %res[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK:               nvws.semaphore.release [[FULL]], [[WTOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // Reader (partition 0): acquire FULL, buffer, tmem_load, release EMPTY.
        // CHECK:               [[RTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[RBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[RTOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[RBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %t2 = ttng.tmem_load %res[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK:               nvws.semaphore.release [[EMPTY]], [[RTOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK:               "use"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // The token results are gone, so neither loop yields a carrier.
        // CHECK-NOT:           scf.yield
        scf.yield {ttg.partition = array<i32: 0, 1>} %t2 : !ttg.async.token
      // CHECK:             } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // CHECK-NOT:           scf.yield
      scf.yield {ttg.partition = array<i32: 0, 1>} %i : !ttg.async.token
    // CHECK:           } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    // No post-loop TMEM access: nothing emitted after the outer loop.
    // CHECK:           tt.return
    tt.return
  }

  // The inner recurrence is identical to nested_ws_inner_loop, but the outer
  // body consumes the same allocation after the inner loop.  The recurrence
  // acquire stays next to the inner MMA; one final LOCAL_EMPTY acquire after
  // the inner loop hands the buffer to the outer continuation.
  // CHECK-LABEL:   tt.func @nested_ws_inner_loop_parent_continuation(
  tt.func @nested_ws_inner_loop_parent_continuation(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // Four semaphores share the allocation: LOCAL EMPTY/FULL ping-pong the
    // inner recurrence, OUTER EMPTY/FULL fence the outer continuation.
    // CHECK:           [[LIVE_ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[LOCAL_EMPTY:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[LOCAL_FULL:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[OUTER_FULL:%.*]] = nvws.semaphore.create [[LIVE_ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // The pre-loop entry acquire drains OUTER_EMPTY's initial permit so the
    // in-body tail acquire waits on the outer reader; its token is unused.
    // CHECK:           [[OUTER_ENTRY:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %res, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Neither loop carries a token: the original iter_args are dropped.
    // CHECK:           scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    %o = scf.for %iv0 = %lb to %ub step %step iter_args(%t0 = %tok) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT:        scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      %i = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %t0) -> (!ttg.async.token) : i32 {
        %sA = "loadA"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %sB = "loadB"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        // Writer (partition 1): acquire LOCAL_EMPTY at first toucher, buffer, MMA, release LOCAL_FULL.
        // CHECK:               [[LACQ:%.*]] = nvws.semaphore.acquire [[LOCAL_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[LBUF:%.*]] = nvws.semaphore.buffer [[LOCAL_EMPTY]], [[LACQ]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               {{.*}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[LBUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %sA, %sB, %res[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK:               nvws.semaphore.release [[LOCAL_FULL]], [[LACQ]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // Reader (partition 0): acquire LOCAL_FULL, buffer, tmem_load, release LOCAL_EMPTY.
        // CHECK:               [[LRTOK:%.*]] = nvws.semaphore.acquire [[LOCAL_FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[LRBUF:%.*]] = nvws.semaphore.buffer [[LOCAL_FULL]], [[LRTOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[LRBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %t2 = ttng.tmem_load %res[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK:               nvws.semaphore.release [[LOCAL_EMPTY]], [[LRTOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK:               "use_inner"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        "use_inner"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK-NOT:           scf.yield
        scf.yield {ttg.partition = array<i32: 0, 1>} %t2 : !ttg.async.token
      // CHECK:             } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // Handoff: the final LOCAL_EMPTY acquire after the inner loop is the
      // token released to OUTER_FULL (async_op none: the last inner reader
      // already arrived after MMA completion).
      // CHECK-NEXT:        [[FINAL_LOCAL:%.*]] = nvws.semaphore.acquire [[LOCAL_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT:        nvws.semaphore.release [[OUTER_FULL]], [[FINAL_LOCAL]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Outer reader (partition 0): acquire OUTER_FULL, buffer, tmem_load, release OUTER_EMPTY.
      // CHECK:             [[OACQ:%.*]] = nvws.semaphore.acquire [[OUTER_FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK:             [[OBUF:%.*]] = nvws.semaphore.buffer [[OUTER_FULL]], [[OACQ]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:             %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[OBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %outerVal, %t3 = ttng.tmem_load %res[%i] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK:             nvws.semaphore.release [[OUTER_EMPTY]], [[OACQ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK:             "use_outer"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      "use_outer"(%outerVal) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // Bridge: regain OUTER_EMPTY and refeed LOCAL_EMPTY for the next outer
      // iteration; without this, iteration two deadlocks.
      // CHECK:             [[OUTER_TAIL:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT:        nvws.semaphore.release [[LOCAL_EMPTY]], [[OUTER_TAIL]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK-NOT:           scf.yield
      scf.yield {ttg.partition = array<i32: 0, 1>} %t3 : !ttg.async.token
    // CHECK:           } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 1 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 1 : i32}
    // CHECK:           tt.return
    tt.return
  }
}

//--- insert_semas_per_edge_tmem.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_single_producer_multi_consumer_fanout
  tt.func @tmem_single_producer_multi_consumer_fanout(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 300 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 2 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V8:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V7:%.*]] = [[V5]]) -> (i32, !ttg.async.token)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a, %ta = ttng.tmem_alloc {buffer.id = 300 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V10:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %s0 = ttng.tmem_store %cst, %a[%ta], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V3]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %v1, %t1 = ttng.tmem_load %a[%s0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_p1"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()

      // CHECK: nvws.semaphore.release [[V2]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.buffer [[V4]], [[V13]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V14]][] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %v2, %t2 = ttng.tmem_load %a[%s0] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_p2"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()

      // CHECK: nvws.semaphore.release [[V2]], [[V13]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V2]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V17:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V16]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %s1 = ttng.tmem_store %cst, %a[%t2], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: scf.yield {{.*}}[[V15]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#alpha_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_qk_alpha_pacc_three_member_edges
  tt.func @tmem_qk_alpha_pacc_three_member_edges(
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst16 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %alpha_val = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #alpha_blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = ttng.tmem_subslice [[V1]] {offset = 0 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V3:%.*]] = ttg.memdesc_reinterpret [[V2]] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = ttng.tmem_subslice [[V1]] {offset = 64 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: [[V5:%.*]] = ttg.memdesc_reinterpret [[V4]] : !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V1]], [[V5]], [[V3]] released = 1 {pending_count = 2 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V1]], [[V5]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V1]], [[V5]], [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V13:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V11:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V12:%.*]] = [[V10]]) -> (i32, !ttg.async.token)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %qk, %tq = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %alpha = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 5>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>

      // CHECK: [[V14:%.*]]:3 = nvws.semaphore.buffer [[V6]], [[V12]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V15:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V14]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %qk0 = ttng.tmem_store %cst, %qk[%tq], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK: nvws.semaphore.release [[V7]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V17:%.*]]:3 = nvws.semaphore.buffer [[V7]], [[V16]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V17]]#0[] {ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %qkv, %qkt = ttng.tmem_load %qk[%qk0] {ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_qk"(%qkv) {ttg.partition = array<i32: 5>} : (tensor<128x128xf32, #blocked>) -> ()

      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V17]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #blocked1> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      ttng.tmem_store %alpha_val, %alpha, %true {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #alpha_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>

      // CHECK: nvws.semaphore.release [[V8]], [[V16]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V18:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V19:%.*]]:3 = nvws.semaphore.buffer [[V8]], [[V18]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V19]]#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1> -> tensor<128x1xf32, #blocked1>
      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #alpha_blocked>
      // CHECK: nvws.semaphore.release [[V6]], [[V18]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_alpha"(%av) {ttg.partition = array<i32: 0>} : (tensor<128x1xf32, #alpha_blocked>) -> ()

      // CHECK: [[V21:%.*]]:3 = nvws.semaphore.buffer [[V7]], [[V16]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V21]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 5>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %pacc = ttng.tmem_alloc %cst16 {buffer.id = 301 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>

      // CHECK: nvws.semaphore.release [[V6]], [[V16]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V23:%.*]]:3 = nvws.semaphore.buffer [[V6]], [[V22]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tc_gen5_mma [[V23]]#2, %{{[-A-Za-z0-9_.$#]+}}, [[V23]]#0[], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %mma = ttng.tc_gen5_mma %pacc, %rhs, %qk[%qkt], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK: [[V24:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V23]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %qk1 = ttng.tmem_store %cst, %qk[%mma], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 5>} : i32
      // CHECK: scf.yield {{.*}}[[V22]]
      scf.yield {ttg.partition = array<i32: 0, 1, 5>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 5>, ttg.partition.outputs = [array<i32: 0, 1, 5>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_full_aliased_members_different_partitions
  tt.func @tmem_full_aliased_members_different_partitions(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 302 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]], [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V4:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]]:2 = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %a = ttng.tmem_alloc %cst0 {buffer.id = 302 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V7]]#0[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %av, %at = ttng.tmem_load %a[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_a"(%av) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V9:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %b = ttng.tmem_alloc %cst1 {buffer.id = 302 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V9]]#1[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %bv, %bt = ttng.tmem_load %b[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_b"(%bv) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_nested_linear_chain_no_outer_drain
  tt.func @tmem_nested_linear_chain_no_outer_drain(
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %true = arith.constant true

    %acc, %atok = ttng.tmem_alloc {buffer.id = 704 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V1:%.*]], [[V2:%.*]] = ttng.tmem_alloc {buffer.id = 704 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V3:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 705 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V3]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V7:%.*]] = [[V2]]) -> (i32, !ttg.async.token)  : i32 {
    %outer:2 = scf.for %iv0 = %lb to %ub step %step iter_args(%i = %c0, %outer_tok = %atok) -> (i32, !ttg.async.token) : i32 {
      // CHECK: [[V10:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V9:%.*]] = [[V7]]) -> (!ttg.async.token)  : i32 {
      %inner = scf.for %iv1 = %lb to %ub step %step iter_args(%inner_tok = %outer_tok) -> (!ttg.async.token) : i32 {
        // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V4]], [[V11]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 5>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %src = ttng.tmem_alloc %cst {buffer.copy = 1 : i32, buffer.id = 705 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>

        // CHECK: nvws.semaphore.release [[V5]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V14:%.*]] = nvws.semaphore.buffer [[V5]], [[V13]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: [[V15:%.*]] = ttng.tc_gen5_mma [[V14]], %{{[-A-Za-z0-9_.$#]+}}, [[V1]]{{\[}}[[V9]]{{\]}}, %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %mma = ttng.tc_gen5_mma %src, %rhs, %acc[%inner_tok], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[V4]], [[V13]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: scf.yield {{.*}}[[V15]]
        scf.yield {ttg.partition = array<i32: 1, 5>} %mma : !ttg.async.token
      } {ttg.partition = array<i32: 1, 5>, ttg.partition.outputs = [array<i32: 1>]}
      %next = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 5>} : i32
      // CHECK: scf.yield {{.*}}, [[V10]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 5>} %next, %inner : i32, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 5>, ttg.partition.outputs = [array<i32: 0, 1, 5>, array<i32: 1>], ttg.warp_specialize.tag = 7 : i32}
    "use_i32"(%outer#0) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_same_owner_reads_close_at_yield
  tt.func @tmem_same_owner_reads_close_at_yield(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 706 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V4:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a, %ta = ttng.tmem_alloc {buffer.id = 706 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V8:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %s = ttng.tmem_store %cst, %a[%ta], %true {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V10]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %v0, %t0 = ttng.tmem_load %a[%s] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_first"(%v0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V10]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %v1, %t1 = ttng.tmem_load %a[%s] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_second"(%v1) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 5>} : i32
      scf.yield {ttg.partition = array<i32: 0, 5>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 5>, ttg.partition.outputs = [array<i32: 0, 5>], ttg.warp_specialize.tag = 8 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_post_ws_read_tag.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // LOWER-LABEL: @post_ws_tmem_read_tag
  // LOWER: ttng.tc_gen5_commit {{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
  // LOWER: ttng.wait_barrier {{.*}} :
  // LOWER: [[READ:%.*]] = ttg.memdesc_index {{.*}} : !ttg.memdesc<1x128x128xf32
  // LOWER: [[OUT:%.*]], {{%.*}} = ttng.tmem_load [[READ]][]
  // LOWER-NEXT: "use"([[OUT]])
  // PARTITION-LABEL: @post_ws_tmem_read_tag
  // PARTITION: nvws.warp_group
  // PARTITION: ttng.tc_gen5_commit {{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
  // PARTITION: nvws.warp_group.return
  // PARTITION: [[POST_READ:%.*]] = ttg.memdesc_index {{.*}} : !ttg.memdesc<1x128x128xf32
  // PARTITION: [[POST_OUT:%.*]], {{%.*}} = ttng.tmem_load [[POST_READ]][]
  // PARTITION-NEXT: "use"([[POST_OUT]])
  // CHECK-LABEL: @post_ws_tmem_read_tag
  tt.func @post_ws_tmem_read_tag(
      %ub: i32,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[HELD:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[INIT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[HELD]]
    // CHECK-NEXT: ttng.tmem_store %{{.*}}, [[INIT_BUF]][], %{{.*}}
    %init = ttng.tmem_store %cst, %acc[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // The same semaphore token remains live across the loop. Every MMA uses
    // its buffer, and one release after the loop tracks the final MMA.
    // CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%carry = %init) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT: [[BODY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[HELD]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, [[BODY_BUF]][], %{{.*}}, %{{.*}} {ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 1>} %mma : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[HELD]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: [[READ:%.*]] = nvws.semaphore.acquire [[FULL]]
    // CHECK-NEXT: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ]]
    // CHECK-NEXT: %{{.*}}, %{{.*}} = ttng.tmem_load [[READ_BUF]][]
    %out, %load_tok = ttng.tmem_load %acc[%loop] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // The post-loop read is the last access; no release of [[EMPTY]] follows it.
    // CHECK-NOT: nvws.semaphore.release
    "use"(%out) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}

//--- insert_semas_raw_if_token.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @raw_edge_token_carried_if
  tt.func @raw_edge_token_carried_if(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]], [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V1]], [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V7:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // Different physical resources in the same logical buffer group force the
      // raw-edge scheduler. The taken branch hands ownership to partition 1 and
      // back to partition 0; the untouched branch passes partition 0's token
      // through the if.
      %a, %ta = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %b, %tb = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V9:%.*]]:2 = nvws.semaphore.buffer [[V2]], [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V10:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %a0 = ttng.tmem_store %cst0, %a[%ta], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cond = arith.cmpi eq, %iv, %c0 {ttg.partition = array<i32: 0, 1>} : i32

      // CHECK: [[IF_TOKEN:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
      %if_tok = scf.if %cond -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[V3]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V12:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]]#0[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %av, %at = ttng.tmem_load %a[%a0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V4]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use_a"(%av) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[HAND_BACK:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[HAND_BACK]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %at : !ttg.async.token
      } else {
        // CHECK: } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V8]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %a0 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // CHECK: [[V14:%.*]]:2 = nvws.semaphore.buffer [[V2]], [[IF_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V15:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V14]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %a1 = ttng.tmem_store %cst1, %a[%if_tok], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK: [[V16:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V14]]#1[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %b0 = ttng.tmem_store %cst0, %b[%tb], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V5]], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V17:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V18:%.*]]:2 = nvws.semaphore.buffer [[V5]], [[V17]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V18]]#1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %bv, %bt = ttng.tmem_load %b[%b0] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V17]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_b"(%bv) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: scf.yield {{.*}} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_recurrence_owner_cycle.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Two fresh writes advance a four-slot physical ring on every iteration.
  // Each logical buffer therefore reuses its slots after two iterations. Its
  // stage-3 EMPTY release followed by a stage-0 reacquire has required owner
  // delay +1, but the reverse FULL handoff has delay -3. The complete owner
  // cycle has delay -2 and is legal: the producer may block while the
  // independent consumer releases the old slot.
  // The pass folds both logical buffers onto one physical ring alloc and gives
  // every protocol op an explicit slot-offset operand: offset 0 rides the
  // current fresh-write cursor, while buffer a's consumer trio sits one fresh
  // write behind it (offset -1) because b's store advanced the cursor between
  // a's store and a's load.
  // CHECK-LABEL: @legal_cross_partition_backpressure
  // PIPE-LABEL: @legal_cross_partition_backpressure
  // PIPE: ttg.warp_specialize
  // PIPE: default {
  // PIPE: scf.for
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_load
  // PIPE: partition0
  // PIPE: scf.for
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_store
  // PIPE: partition1
  // PIPE: scf.for
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_store
  tt.func @legal_cross_partition_backpressure(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 15 {pending_count = 1 : i32} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>
    // CHECK-NOT: iter_args
    %a = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %av = "producer_a"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : () -> tensor<128x64xf16, #blocked>
      // CHECK: [[V4:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]][[[V4]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V6:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]][[[V6]]], [[V5]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %av, %a {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: [[V8:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: nvws.semaphore.release [[V3]][[[V8]]], [[V5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %bv = "producer_b"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : () -> tensor<128x64xf16, #blocked>
      // CHECK: [[V9:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V2]][[[V9]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V2]][[[V11]]], [[V10]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %bv, %b {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: [[V13:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: nvws.semaphore.release [[V3]][[[V13]]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token

      // CHECK: [[V14:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V3]][[[V14]]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V16:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[V17:%.*]] = nvws.semaphore.buffer [[V3]][[[V16]]], [[V15]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: [[V18:%.*]] = ttg.local_load [[V17]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %br = ttg.local_load %b {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: [[V19:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: nvws.semaphore.release [[V2]][[[V19]]], [[V15]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume_b"([[V18]]) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      "consume_b"(%br) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
      // CHECK: [[V20:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} -1 : i32
      // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V3]][[[V20]]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V22:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} -1 : i32
      // CHECK: [[V23:%.*]] = nvws.semaphore.buffer [[V3]][[[V22]]], [[V21]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: [[V24:%.*]] = ttg.local_load [[V23]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %ar = ttg.local_load %a {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: [[V25:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} -1 : i32
      // CHECK: nvws.semaphore.release [[V2]][[[V25]]], [[V21]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume_a"([[V24]]) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      "consume_a"(%ar) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_recurrence_schedule.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // This is the one-slot Q shape from attention backward. The final read at
  // loop.stage 1 releases the slot reused by the loop.stage 0 store in a future
  // iteration. Because the loop-carried dependency distance is one, the final
  // read and next store execute in the same pipelined iteration; loop.cluster
  // must order the store and its first consumer after the final read.
  // CHECK-LABEL: @one_slot_recurrence
  // PIPE-LABEL: @one_slot_recurrence
  // PIPE: partition0
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  // PIPE: scf.for
  // PIPE: ttg.local_load
  // PIPE: ttng.arrive_barrier
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  // PIPE: "consume_last"
  tt.func @one_slot_recurrence(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 420 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 420 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %value = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : () -> tensor<128x64xf16, #blocked>

      // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[V7]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %first = ttg.local_load %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: "consume_first"({{.*}}) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      "consume_first"(%first) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()

      // CHECK: ttg.local_load [[V7]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %last = ttg.local_load %alloc {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume_last"({{.*}}) {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      "consume_last"(%last) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 1, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // This is the same ownership cycle shifted across the two partitions. The
  // EMPTY handoff requires owner delay +1 and the FULL handoff contributes -1,
  // so the cycle is feasible but both handoffs meet in the same retimed wave.
  // Cluster legalization must still put the final read before the next write.
  // CHECK-LABEL: @retimed_zero_delay_cycle
  // PIPE-LABEL: @retimed_zero_delay_cycle
  // PIPE: partition0
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  // PIPE: scf.for
  // PIPE: ttg.local_load
  // PIPE: ttng.arrive_barrier
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_load
  // PIPE: "consume_first"
  // PIPE: "consume_last"
  tt.func @retimed_zero_delay_cycle(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 423 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 423 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %value = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : () -> tensor<128x64xf16, #blocked>
      // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      // CHECK: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[V7]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %first = ttg.local_load %alloc {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: "consume_first"({{.*}}) {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      "consume_first"(%first) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()

      // CHECK: ttg.local_load [[V7]] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %last = ttg.local_load %alloc {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: "consume_last"({{.*}}) {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
      "consume_last"(%last) {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 1, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}

//--- insert_semas_recurrence_schedule_errors.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The EMPTY handoff from partition 1 to partition 3 requires one iteration
  // of owner delay, while the reverse FULL handoff has zero delay. Their
  // positive cycle cannot be satisfied by cross-partition backpressure.
  // CHECK: error: nvws-insert-semas: fixed loop.stage assignments form an unsatisfiable semaphore handoff cycle (cycle requires 1 additional pipeline iteration)
  // CHECK-DAG: note: handoff {1} -> {3} has producer loop.stage 2, consumer loop.stage 0, loop-carried dependency distance 1, and required delay 1
  // CHECK-DAG: note: handoff {3} -> {1} has producer loop.stage 0, consumer loop.stage 0, loop-carried dependency distance 0, and required delay 0
  tt.func @invalid_loop_carried_schedule(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 421 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      %value = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : () -> tensor<128x64xf16, #blocked>
      ttg.local_store %value, %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %first = ttg.local_load %alloc {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume_first"(%first) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
      %last = ttg.local_load %alloc {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume_last"(%last) {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 1, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_region_drain.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // One completed WS-region phase supplies both root writes. The drain acquire
  // is one exact token producer; the two handoff releases fan out from it.
  // CHECK-LABEL: @region_drain_fanout
  tt.func @region_drain_fanout(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[WHOLE:%.*]] = ttg.local_alloc
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[WHOLE]]
    %whole = ttg.local_alloc {buffer.id = 9903 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 9903 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 9903 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %root = "root.value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      ttg.local_store %value, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      %loaded = ttg.local_load %whole {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !two
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!two) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: [[DRAIN:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: nvws.semaphore.release {{%.*}}, [[DRAIN]]
    // CHECK-NEXT: nvws.semaphore.release {{%.*}}, [[DRAIN]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    ttg.local_store %root, %left : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    ttg.local_store %root, %right : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // A branch-only handoff drains the completed WS-region phase inside the
  // branch that executes. The two static drain sites are mutually exclusive.
  // CHECK-LABEL: @guarded_region_drain
  tt.func @guarded_region_drain(%lb: i32, %ub: i32, %step: i32,
                                %cond: i1) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
    // CHECK: [[BRANCH_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]]
    %alloc = ttg.local_alloc {buffer.id = 9904 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !one
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!one) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: scf.if
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    scf.if %cond {
      // CHECK-NEXT: [[THEN_DRAIN:%.*]] = nvws.semaphore.acquire [[BRANCH_EMPTY]]
      // CHECK-NEXT: nvws.semaphore.release {{%.*}}, [[THEN_DRAIN]]
      %then_value = ttg.local_load %alloc : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "then.use"(%then_value) : (!one) -> ()
    } else {
      // CHECK: } else {
      // CHECK-NEXT: [[ELSE_DRAIN:%.*]] = nvws.semaphore.acquire [[BRANCH_EMPTY]]
      // CHECK-NEXT: nvws.semaphore.release {{%.*}}, [[ELSE_DRAIN]]
      %else_value = ttg.local_load %alloc : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "else.use"(%else_value) : (!one) -> ()
    }
    tt.return
  }
}

//--- insert_semas_region_drain_continuation.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @root_continuation
  tt.func @root_continuation(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    // CHECK: [[ROOT_WHOLE:%[0-9]+]] = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[ROOT_LEFT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[ROOT_RIGHT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 1 : i32}
    // CHECK-NEXT: [[ROOT_EMPTY:%[0-9]+]] = nvws.semaphore.create [[ROOT_WHOLE]], [[ROOT_LEFT]], [[ROOT_RIGHT]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[ROOT_FULL:%[0-9]+]] = nvws.semaphore.create [[ROOT_WHOLE]], [[ROOT_LEFT]], [[ROOT_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[ROOT_LEFT_READY:%[0-9]+]] = nvws.semaphore.create [[ROOT_WHOLE]], [[ROOT_LEFT]], [[ROOT_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[ROOT_RIGHT_READY:%[0-9]+]] = nvws.semaphore.create [[ROOT_WHOLE]], [[ROOT_LEFT]], [[ROOT_RIGHT]] {pending_count = 1 : i32}
    %whole = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 9910 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[ROOT_VALUE:%[0-9]+]] = "root.value"()
    %root = "root.value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[ROOT_LOOP_VALUE:%[0-9]+]] = "producer"() {ttg.partition = array<i32: 0>}
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      // CHECK-NEXT: [[ROOT_WRITE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[ROOT_WRITE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_EMPTY]], [[ROOT_WRITE_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store [[ROOT_LOOP_VALUE]], [[ROOT_WRITE_BUFFER]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[ROOT_FULL]], [[ROOT_WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %value, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[ROOT_READ_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[ROOT_READ_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_FULL]], [[ROOT_READ_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[ROOT_LOADED:%[0-9]+]] = ttg.local_load [[ROOT_READ_BUFFER]]#0 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[ROOT_EMPTY]], [[ROOT_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %whole {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !two
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!two) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: [[ROOT_DRAIN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_EMPTY]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[ROOT_RIGHT_READY]], [[ROOT_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[ROOT_LEFT_READY]], [[ROOT_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: [[ROOT_LEFT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_LEFT_READY]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    scf.if %cond {
      // CHECK: scf.if
      // CHECK-NEXT: [[ROOT_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_LEFT_READY]], [[ROOT_LEFT_TOKEN]]
      // CHECK-NEXT: ttg.local_store [[ROOT_VALUE]], [[ROOT_THEN_BUFFER]]#1
      ttg.local_store %root, %left : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } else {
      // CHECK: } else {
      // CHECK-NEXT: [[ROOT_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_LEFT_READY]], [[ROOT_LEFT_TOKEN]]
      // CHECK-NEXT: ttg.local_store [[ROOT_VALUE]], [[ROOT_ELSE_BUFFER]]#1
      ttg.local_store %root, %left : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    }
    // CHECK: [[ROOT_RIGHT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[ROOT_RIGHT_READY]]
    // CHECK-NEXT: [[ROOT_FINAL_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[ROOT_RIGHT_READY]], [[ROOT_RIGHT_TOKEN]]
    // CHECK-NEXT: ttg.local_store [[ROOT_VALUE]], [[ROOT_FINAL_BUFFER]]#2
    ttg.local_store %root, %right : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @partition_2_then_partition_3
  tt.func @partition_2_then_partition_3(%lb: i32, %ub: i32, %step: i32,
                                        %cond: i1) {
    // CHECK: [[P23_WHOLE:%[0-9]+]] = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[P23_LEFT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[P23_RIGHT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 1 : i32}
    // CHECK-NEXT: [[P23_EMPTY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_FULL:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_ROOT_READY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_THEN_READY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_ELSE_READY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P23_RIGHT_READY:%[0-9]+]] = nvws.semaphore.create [[P23_WHOLE]], [[P23_LEFT]], [[P23_RIGHT]] {pending_count = 1 : i32}
    %whole = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 9911 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[P23_VALUE:%[0-9]+]] = "root.value"()
    %root = "root.value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[P23_LOOP_VALUE:%[0-9]+]] = "producer"() {ttg.partition = array<i32: 0>}
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      // CHECK-NEXT: [[P23_WRITE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[P23_WRITE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_EMPTY]], [[P23_WRITE_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store [[P23_LOOP_VALUE]], [[P23_WRITE_BUFFER]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[P23_FULL]], [[P23_WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %value, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[P23_READ_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[P23_READ_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_FULL]], [[P23_READ_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[P23_LOADED:%[0-9]+]] = ttg.local_load [[P23_READ_BUFFER]]#0 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[P23_EMPTY]], [[P23_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %whole {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !two
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!two) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: [[P23_DRAIN:%[0-9]+]] = nvws.semaphore.acquire [[P23_EMPTY]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[P23_RIGHT_READY]], [[P23_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[P23_ROOT_READY]], [[P23_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: [[P23_ROOT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_ROOT_READY]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>,
       ttg.warp_specialize.tag = 0 : i32}
    scf.if %cond {
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[P23_THEN_READY]], [[P23_ROOT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
      // CHECK-NEXT: [[P23_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_THEN_READY]] {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: [[P23_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_THEN_READY]], [[P23_THEN_TOKEN]] {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: ttg.local_store [[P23_VALUE]], [[P23_THEN_BUFFER]]#1 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      ttg.local_store %root, %left {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } else {
      // CHECK: } else {
      // CHECK-NEXT: nvws.semaphore.release [[P23_ELSE_READY]], [[P23_ROOT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
      // CHECK-NEXT: [[P23_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_ELSE_READY]] {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: [[P23_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_ELSE_READY]], [[P23_ELSE_TOKEN]] {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: ttg.local_store [[P23_VALUE]], [[P23_ELSE_BUFFER]]#1 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
      ttg.local_store %root, %left {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    }
    // CHECK: [[P23_RIGHT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P23_RIGHT_READY]] {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: [[P23_FINAL_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P23_RIGHT_READY]], [[P23_RIGHT_TOKEN]] {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[P23_VALUE]], [[P23_FINAL_BUFFER]]#2 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32}
    ttg.local_store %root, %right {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @partition_1_then_partition_0
  tt.func @partition_1_then_partition_0(%lb: i32, %ub: i32, %step: i32,
                                        %cond: i1) {
    // CHECK: [[P10_WHOLE:%[0-9]+]] = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[P10_LEFT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[P10_RIGHT:%[0-9]+]] = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 1 : i32}
    // CHECK-NEXT: [[P10_EMPTY:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[P10_FULL:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P10_ROOT_READY:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P10_THEN_READY:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[P10_ELSE_READY:%[0-9]+]] = nvws.semaphore.create [[P10_WHOLE]], [[P10_LEFT]], [[P10_RIGHT]] {pending_count = 1 : i32}
    %whole = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 9912 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[P10_VALUE:%[0-9]+]] = "root.value"()
    %root = "root.value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[P10_LOOP_VALUE:%[0-9]+]] = "producer"() {ttg.partition = array<i32: 0>}
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !two
      // CHECK-NEXT: [[P10_WRITE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[P10_WRITE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_EMPTY]], [[P10_WRITE_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store [[P10_LOOP_VALUE]], [[P10_WRITE_BUFFER]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[P10_FULL]], [[P10_WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttg.local_store %value, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[P10_READ_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[P10_READ_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_FULL]], [[P10_READ_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[P10_LOADED:%[0-9]+]] = ttg.local_load [[P10_READ_BUFFER]]#0 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[P10_EMPTY]], [[P10_READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %loaded = ttg.local_load %whole {ttg.partition = array<i32: 1>} : !ttg.memdesc<2xi32, #shared, #smem, mutable> -> !two
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!two) -> ()
    // CHECK: } {tt.warp_specialize
    // CHECK-NEXT: [[P10_DRAIN:%[0-9]+]] = nvws.semaphore.acquire [[P10_EMPTY]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[P10_ROOT_READY]], [[P10_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: [[P10_ROOT_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_ROOT_READY]]
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    scf.if %cond {
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[P10_THEN_READY]], [[P10_ROOT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
      // CHECK-NEXT: [[P10_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_THEN_READY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: [[P10_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_THEN_READY]], [[P10_THEN_TOKEN]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: ttg.local_store [[P10_VALUE]], [[P10_THEN_BUFFER]]#1 {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      ttg.local_store %root, %left {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } else {
      // CHECK: } else {
      // CHECK-NEXT: nvws.semaphore.release [[P10_ELSE_READY]], [[P10_ROOT_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32}
      // CHECK-NEXT: [[P10_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[P10_ELSE_READY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: [[P10_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_ELSE_READY]], [[P10_ELSE_TOKEN]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      // CHECK-NEXT: ttg.local_store [[P10_VALUE]], [[P10_ELSE_BUFFER]]#1 {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
      ttg.local_store %root, %left {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    }
    // CHECK: [[P10_FINAL_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[P10_EMPTY]], [[P10_DRAIN]]
    // CHECK-NEXT: ttg.local_store [[P10_VALUE]], [[P10_FINAL_BUFFER]]#2 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    ttg.local_store %root, %right {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

//--- insert_semas_release_count.mlir

//
// EMIT/LOWER pin the first-class count contract end to end on this
// pass-produced shape: the
// scaled release carries arrive_count = 2 in emitted IR, and the
// lowering transcribes both counts into the mbarrier init/arrive.
// LOWER: ttng.init_barrier {{.*}}, 2
// LOWER: ttng.arrive_barrier {{.*}}, 2

// Release arrive-multiplicity (spec section 5.2, uniform pending count):
// a semaphore's pending count is a per-semaphore constant — every acquire
// site sees the same count and every acquire cycle must receive exactly
// that many arrives. Shape: producer {3} stores outside the inner loop;
// inside, {2} and {1} read, {1} CORRECTS the buffer in place after an
// explicit {2}->{1} WAR handoff, and {0} consumes the corrected value
// AFTER the store. The last version's holders at the inner EXIT are
// therefore {1} (the writer) and {0} (its reader) — a fan-in-2 regain —
// and the outer single-source ready edge lands on the SAME semaphore.
// The lone outer release must arrive twice: r S(2). The emitted and
// lowered IR checks below pin that multiplicity.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // EMIT-LABEL: @release_multiplicity_unified_fanin_regain
  tt.func @release_multiplicity_unified_fanin_regain(%lb: i32, %ub: i32, %step: i32) {
    // EMIT: [[V1:%.*]] = ttg.local_alloc {buffer.id = 700 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %alloc = ttg.local_alloc {buffer.id = 700 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // V2 = outer ready (initially released — supplies iteration zero),
    // V3 = {2}->{1} read handoff, V4 = {2}->{1} WAR handoff, V5 =
    // {1}->{0} corrected-value edge, V6 = the unified full semaphore
    // with pending_count = 2.
    // EMIT: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT: [[V5:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT: [[V6:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // EMIT-NOT: iter_args
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 3>} : () -> !ty
      // {3}'s acquire of the outer ready semaphore sits at its point of
      // use inside the loop body; no token is threaded through the loop.
      // EMIT: [[V7:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // EMIT: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V7]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // EMIT: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]] {ttg.partition = array<i32: 3>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // EMIT: nvws.semaphore.release [[V6]], [[V7]] [#nvws.async_op<none>] {arrive_count = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 3>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // EMIT-NOT: iter_args
      scf.for %j = %lb to %ub step %step : i32 {
        // EMIT: [[V9:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // EMIT: nvws.semaphore.release [[V3]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // EMIT: [[V10:%.*]] = nvws.semaphore.buffer [[V6]], [[V9]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: ttg.local_load [[V10]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        // EMIT: nvws.semaphore.release [[V4]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "use2"(%l2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        // EMIT: [[V11:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // EMIT: [[V12:%.*]] = nvws.semaphore.buffer [[V3]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: ttg.local_load [[V12]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        %l1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        %c = "correct"(%l1) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
        // EMIT: [[V13:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // EMIT: [[V14:%.*]] = nvws.semaphore.buffer [[V4]], [[V13]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V14]] {ttg.partition = array<i32: 1>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: nvws.semaphore.release [[V5]], [[V13]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // EMIT: nvws.semaphore.release [[V6]], [[V13]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        ttg.local_store %c, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: [[V15:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // EMIT: [[V16:%.*]] = nvws.semaphore.buffer [[V5]], [[V15]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: ttg.local_load [[V16]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        %l0 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        // EMIT: nvws.semaphore.release [[V6]], [[V15]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "use0"(%l0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      } {ttg.partition = array<i32: 0, 1, 2>}
      // After the inner loop, {2} regains the last version — its acquire
      // cycle absorbs the fan-in-2 arrives from {1} and {0} — and only
      // then releases the outer ready semaphore for {3}'s next store.
      // EMIT: } {ttg.partition = array<i32: 0, 1, 2>}
      // EMIT: [[V17:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // EMIT: nvws.semaphore.release [[V2]], [[V17]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// The outer ready release by {3} carries arrive multiplicity 2 — one
// release op, two arrives — because it shares the pending_count = 2
// semaphore with the inner fan-in-2 regain ({1} the corrector and {0}
// its post-store reader each arrive once into {2}'s next acquire cycle);
// both acquire sites read the uniform pending count off the one create.
// Every acquire sits at its point of use: neither loop threads a token,
// and the outer ready semaphore is created initially released to supply
// iteration zero. The stable ENTER source lets {1}'s read overlap {2}'s
// read, so the in-loop store takes an explicit {2}->{1} WAR edge; {1}'s
// own read is ordered by program order.

//--- insert_semas_root_entry_tmem.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @root_entry_accumulator_uses_native_carried_pou
  tt.func @root_entry_accumulator_uses_native_carried_pou(
      %ub: i32,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true

    // Root initializes the accumulator before entering the WS loop. Native
    // carried POU passes that exact token into the loop. A zero-trip loop
    // returns it unchanged; a nonzero loop returns partition 1's final token.
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[ACC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[ROOT:%.*]] = nvws.semaphore.create [[ACC]] released = 3 {pending_count = 1 : i32}
    // CHECK-NEXT: [[TO_MMA:%.*]] = nvws.semaphore.create [[ACC]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[TO_ROOT:%.*]] = nvws.semaphore.create [[ACC]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[ROOT_TOKEN:%.*]] = nvws.semaphore.acquire [[ROOT]]
    // CHECK-NEXT: [[ROOT_BUFFER:%.*]] = nvws.semaphore.buffer [[ROOT]], [[ROOT_TOKEN]]
    // CHECK-NEXT: ttng.tmem_store %{{.*}}, [[ROOT_BUFFER]][], %{{.*}}
    %init = ttng.tmem_store %cst, %acc[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[LOOP_RESULT:%.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[LOOP_TOKEN:%.*]] = [[ROOT_TOKEN]]) -> (!ttg.async.token)  : i32 {
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%carry = %init) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT: [[LOOP_BUFFER:%.*]] = nvws.semaphore.buffer [[ROOT]], [[LOOP_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: %{{.*}}, %{{.*}} = ttng.tmem_load [[LOOP_BUFFER]][] {ttg.partition = array<i32: 1>}
      %loaded, %load_tok = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

      // CHECK: ttng.tmem_store %{{.*}}, [[LOOP_BUFFER]][], %{{.*}} {ttg.partition = array<i32: 1>}
      %store = ttng.tmem_store %loaded, %acc[%load_tok], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: nvws.semaphore.release [[TO_MMA]], [[LOOP_TOKEN]] [#nvws.async_op<none>]
      // CHECK-NEXT: [[MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_MMA]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[MMA_BUFFER:%.*]] = nvws.semaphore.buffer [[TO_MMA]], [[MMA_TOKEN]] {ttg.partition = array<i32: 2>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%store], %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, [[MMA_BUFFER]][]
      // CHECK-NEXT: nvws.semaphore.release [[ROOT]], [[MMA_TOKEN]] [#nvws.async_op<tc5mma>]
      // CHECK-NEXT: [[NEXT_TOKEN:%.*]] = nvws.semaphore.acquire [[ROOT]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.yield {{.*}}[[NEXT_TOKEN]] : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 1, 2>} %mma : !ttg.async.token
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[TO_ROOT]], [[LOOP_RESULT]] [#nvws.async_op<none>]
    // CHECK-NEXT: [[OUT_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_ROOT]]
    // CHECK-NEXT: [[OUT_BUFFER:%.*]] = nvws.semaphore.buffer [[TO_ROOT]], [[OUT_TOKEN]]
    // CHECK-NEXT: %{{.*}}, %{{.*}} = ttng.tmem_load [[OUT_BUFFER]][]
    %out, %out_tok = ttng.tmem_load %acc[%loop] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%out) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}

//--- insert_semas_same_owner_mixed_completion.mlir

// Two exact-alias members are filled by one partition before either is
// consumed by another partition.  The first fill is a TMA load and the second
// is synchronous.  Their one ownership handoff must retain both completion
// signals: the TMA completion and the explicit arrival after the local store.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @same_owner_mixed_completion
  // LOWER-LABEL: @same_owner_mixed_completion
  // LOWER: ttng.init_barrier %{{.*}}, 2
  // LOWER: ttng.barrier_expect %{{.*}}, 16384
  // LOWER: ttng.async_tma_copy_global_to_local
  // LOWER: ttng.arrive_barrier %{{.*}}, 1
  tt.func @same_owner_mixed_completion(%desc: !tt.tensordesc<128x64xf16, #shared>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    %tma = ttg.local_alloc {buffer.id = 610 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %sync = ttg.local_alloc {buffer.id = 610 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>

    // CHECK: [[TMA_BASE:%.*]] = ttg.local_alloc {buffer.id = 610 : i32, buffer.offset = 0 : i32}
    // CHECK: [[SYNC_BASE:%.*]] = ttg.local_alloc {buffer.id = 610 : i32, buffer.offset = 0 : i32}
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[TMA_BASE]], [[SYNC_BASE]] released = 1 {pending_count = 1 : i32}
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[TMA_BASE]], [[SYNC_BASE]] {pending_count = 2 : i32}
    // CHECK: scf.for
    scf.for %iv = %lb to %ub step %step : i32 {
      // CHECK: [[PRODUCER_TOKEN:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: [[PRODUCER_BUFFERS:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[PRODUCER_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 16384 [[PRODUCER_BUFFERS]]#0 {ttg.partition = array<i32: 0>}
      nvws.descriptor_load %desc[%i, %i] 16384 %tma {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[PRODUCER_BUFFERS]]#1 {ttg.partition = array<i32: 0>}
      ttg.local_store %value, %sync {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]], [[PRODUCER_TOKEN]] [#nvws.async_op<none>, #nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}

      // CHECK: [[CONSUMER_TOKEN:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK: [[CONSUMER_BUFFERS:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[CONSUMER_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load [[CONSUMER_BUFFERS]]#1 {ttg.partition = array<i32: 1>}
      %sync_value = ttg.local_load %sync {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: ttg.local_load [[CONSUMER_BUFFERS]]#0 {ttg.partition = array<i32: 1>}
      %tma_value = ttg.local_load %tma {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume"(%sync_value, %tma_value) {ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>) -> ()
      // CHECK: nvws.semaphore.release [[EMPTY]], [[CONSUMER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // The same owner-token wave may fill partially overlapping members.  The
  // later synchronous write must retain the earlier TMA completion even
  // though the group has more than one physical piece.
  // CHECK-LABEL: @same_owner_partial_overlap_mixed_completion
  // LOWER-LABEL: @same_owner_partial_overlap_mixed_completion
  // LOWER: ttng.init_barrier %{{.*}}, 2
  // LOWER: ttng.barrier_expect %{{.*}}, 16384
  // LOWER: ttng.async_tma_copy_global_to_local
  // LOWER: ttng.arrive_barrier %{{.*}}, 1
  tt.func @same_owner_partial_overlap_mixed_completion(%desc: !tt.tensordesc<128x64xf16, #shared>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    %tma = ttg.local_alloc {buffer.id = 611 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %sync = ttg.local_alloc {buffer.id = 611 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %value = arith.constant dense<0.000000e+00> : tensor<256x64xf16, #blocked>

    // CHECK: [[PARTIAL_TMA_BASE:%.*]] = ttg.local_alloc {buffer.id = 611 : i32, buffer.offset = 0 : i32}
    // CHECK: [[PARTIAL_SYNC_BASE:%.*]] = ttg.local_alloc {buffer.id = 611 : i32, buffer.offset = 0 : i32}
    // CHECK: [[PARTIAL_EMPTY:%.*]] = nvws.semaphore.create [[PARTIAL_TMA_BASE]], [[PARTIAL_SYNC_BASE]] released = 1 {pending_count = 1 : i32}
    // CHECK: [[PARTIAL_FULL:%.*]] = nvws.semaphore.create [[PARTIAL_TMA_BASE]], [[PARTIAL_SYNC_BASE]] {pending_count = 2 : i32}
    // CHECK: scf.for
    scf.for %iv = %lb to %ub step %step : i32 {
      // CHECK: [[PARTIAL_PRODUCER_TOKEN:%.*]] = nvws.semaphore.acquire [[PARTIAL_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: [[PARTIAL_PRODUCER_BUFFERS:%.*]]:2 = nvws.semaphore.buffer [[PARTIAL_EMPTY]], [[PARTIAL_PRODUCER_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 16384 [[PARTIAL_PRODUCER_BUFFERS]]#0 {ttg.partition = array<i32: 0>}
      nvws.descriptor_load %desc[%i, %i] 16384 %tma {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[PARTIAL_PRODUCER_BUFFERS]]#1 {ttg.partition = array<i32: 0>}
      ttg.local_store %value, %sync {ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[PARTIAL_FULL]], [[PARTIAL_PRODUCER_TOKEN]] [#nvws.async_op<none>, #nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}

      // CHECK: [[PARTIAL_CONSUMER_TOKEN:%.*]] = nvws.semaphore.acquire [[PARTIAL_FULL]] {ttg.partition = array<i32: 1>}
      // CHECK: [[PARTIAL_CONSUMER_BUFFERS:%.*]]:2 = nvws.semaphore.buffer [[PARTIAL_FULL]], [[PARTIAL_CONSUMER_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load [[PARTIAL_CONSUMER_BUFFERS]]#1 {ttg.partition = array<i32: 1>}
      %sync_value = ttg.local_load %sync {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #blocked>
      // CHECK: ttg.local_load [[PARTIAL_CONSUMER_BUFFERS]]#0 {ttg.partition = array<i32: 1>}
      %tma_value = ttg.local_load %tma {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume_partial"(%sync_value, %tma_value) {ttg.partition = array<i32: 1>} : (tensor<256x64xf16, #blocked>, tensor<128x64xf16, #blocked>) -> ()
      // CHECK: nvws.semaphore.release [[PARTIAL_EMPTY]], [[PARTIAL_CONSUMER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}

//--- insert_semas_scheduled_region_slots.mlir

// These tests cover slot replay across an atomic scheduled region.  The
// scf.if itself owns the stage-2 schedule; its child operations intentionally
// have no loop.stage/loop.cluster attributes.

#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked128 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Two fresh writes advance the depth-5 cursor from A's slot to B's slot.
  // The conditional C wave is a same-owner overwrite after B has been read,
  // so it reuses B's slot and does not advance the cursor.  The region-closing
  // release therefore supplies A three logical iterations later:
  //
  //   A(i) = 2i mod 5, B/C(i) = 2i+1 mod 5, A(i+3) = B/C(i).
  //
  // The only authored non-zero displacement is the A-read to B-write handoff.
  // The scf.if returns an owner-0 token: the then-branch hands the C-read
  // token back to owner 0, while the else-branch passes the B-read token
  // through.  One common ENTRY release follows the conditional.
  // SEMA-LABEL: @depth5_regular_atomic_if
  // ASP-LABEL: @depth5_regular_atomic_if
  tt.func @depth5_regular_atomic_if(%lb: i32, %ub: i32, %step: i32,
                                    %cond: i1) {
    // SEMA: [[A_BASE:%.*]] = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32}
    // SEMA: [[B_BASE:%.*]] = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32}
    // SEMA: [[C_BASE:%.*]] = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32}
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] released = 21 {pending_count = 1 : i32}
    // SEMA: [[A_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[A_TO_B:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[B_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[C_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[IF_BACK:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // ASP: [[ENTRY:%.*]] = nvws.semaphore.create {{.*}} released = 21 {pending_count = 1 : i32}
    // ASP: [[A_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[A_TO_B:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[B_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[C_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[IF_BACK:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    %a = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %a_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %b_value = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #blocked64>
    %c_value = arith.constant dense<2.000000e+00> : tensor<128x128xf16, #blocked128>

    // No token iter_args: every acquire is at its point of use in the body.
    // After ASP the loop carries the cursor plus phase words, all i32.
    // SEMA: scf.for
    // ASP: %{{[0-9]+}}:7 = scf.for {{.*}} iter_args([[CURSOR:%.*]] = %{{[-A-Za-z0-9_.$#]+}},
    scf.for %iv = %lb to %ub step %step : i32 {
      // SEMA: [[ZA:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: [[A_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[ZA]]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[A_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]], [[A_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: ttg.local_store {{.*}}, [[A_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[ZAF:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: nvws.semaphore.release [[A_FULL]][[[ZAF]]], [[A_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[A_SLOT:%.*]] = arith.select {{.*}} : i32
      // ASP: [[A_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[A_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[A_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]][[[A_SLOT]]], [[A_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: ttg.local_store {{.*}}, [[A_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: nvws.semaphore.release [[A_FULL]][[[A_SLOT]]], [[A_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %a_value, %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // The A read hands its slot off to the B write at displacement +1.
      // SEMA: [[ZAR:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_FULL]][[[ZAR]]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: [[A_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_FULL]], [[A_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: ttg.local_load [[A_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: [[TO_B:%.*]] = arith.constant {{.*}} 1 : i32
      // SEMA: nvws.semaphore.release [[A_TO_B]][[[TO_B]]], [[A_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_FULL]][[[A_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[A_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_FULL]][[[A_SLOT]]], [[A_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: ttg.local_load [[A_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[TO_B_RAW:%.*]] = arith.addi [[A_SLOT]], {{%.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      // ASP: [[TO_B_REM:%.*]] = arith.remsi [[TO_B_RAW]], {{%.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      // ASP: [[TO_B_SLOT:%.*]] = arith.select {{.*}}, {{.*}}, [[TO_B_REM]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      // ASP: nvws.semaphore.release [[A_TO_B]][[[TO_B_SLOT]]], [[A_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %a_read = ttg.local_load %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_a"(%a_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      // B's fresh acquire advances the cursor to [[B_SLOT]] = [[A_SLOT]] + 1.
      // SEMA: [[ZB:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: [[B_TOK:%.*]] = nvws.semaphore.acquire [[A_TO_B]][[[ZB]]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[B_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_TO_B]], [[B_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: ttg.local_store {{.*}}, [[B_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // SEMA: [[ZBF:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: nvws.semaphore.release [[B_FULL]][[[ZBF]]], [[B_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[B_RAW:%.*]] = arith.addi [[A_SLOT]], {{%.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_SLOT:%.*]] = arith.select {{.*}}, {{.*}}, [[B_RAW]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : i32
      // ASP: [[B_TOK:%.*]] = nvws.semaphore.acquire [[A_TO_B]][[[B_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: [[B_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_TO_B]][[[B_SLOT]]], [[B_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: ttg.local_store {{.*}}, [[B_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // ASP: nvws.semaphore.release [[B_FULL]][[[B_SLOT]]], [[B_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %b_value, %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: [[ZBR:%.*]] = arith.constant {{.*}} 0 : i32
      // SEMA: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]][[[ZBR]]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: [[B_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]], [[B_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: ttg.local_load [[B_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]][[[B_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[B_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]][[[B_SLOT]]], [[B_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: ttg.local_load [[B_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %b_read = ttg.local_load %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_b"(%b_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      // The scf.if yields its owner-0 token.  ASP also returns the shared slot
      // and the phase words updated inside the branch.
      // SEMA: [[IF_TOKEN:%.*]] = scf.if
      // ASP: [[IF_RESULTS:%.*]]:4 = scf.if
      scf.if %cond {
        // C overwrites B's slot under the still-held B-read token: it is
        // rendered through the C member of B_FULL's buffer tuple, so the
        // atomic region does not add a third cursor advance.
        // SEMA: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
        // SEMA: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // SEMA: nvws.semaphore.release [[C_FULL]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        // ASP: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]][[[B_SLOT]]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
        // ASP: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // ASP: nvws.semaphore.release [[C_FULL]][[[B_SLOT]]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        ttg.local_store %c_value, %c {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked128> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // The C read hands ownership back to owner 0 before the branch yields.
        // SEMA: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]] {ttg.partition = array<i32: 1>}
        // SEMA: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // SEMA: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // SEMA: nvws.semaphore.release [[IF_BACK]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // SEMA: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]] {ttg.partition = array<i32: 0>}
        // SEMA: scf.yield {{.*}}[[IF_THEN_TOKEN]] : !ttg.async.token
        // ASP: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[B_SLOT]], {{%.*}}] {ttg.partition = array<i32: 1>}
        // ASP: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]][[[B_SLOT]]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // ASP: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // ASP: nvws.semaphore.release [[IF_BACK]][[[B_SLOT]]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // ASP: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]][[[B_SLOT]], {{%.*}}] {ttg.partition = array<i32: 0>}
        // ASP: scf.yield {{.*}}[[IF_THEN_TOKEN]], [[B_SLOT]],
        %c_read = ttg.local_load %c {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked128>
        "consume_c"(%c_read) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked128>) -> ()
      } else {
        // Without the C wave, the branch passes the B-read token through.
        // SEMA: scf.yield {{.*}}[[B_READ_TOK]] : !ttg.async.token
        // ASP: scf.yield {{.*}}[[B_READ_TOK]], [[B_SLOT]],
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      // SEMA: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // ASP: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>]}
      // SEMA: nvws.semaphore.release [[ENTRY]][{{%.*}}], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[ENTRY]][[[IF_RESULTS]]#1], [[IF_RESULTS]]#0 [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: scf.yield {{.*}}[[IF_RESULTS]]#1,
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Consecutive A/B writes by the same owner share one token and therefore
  // make one fresh depth-3 cursor advance.  Both reads and the atomic C region
  // use that stage.  The scf.if returns an owner-0 token from either the C read
  // or the unchanged A/B-read path.  One common release then returns the slot
  // to A after one full three-iteration orbit.
  // SEMA-LABEL: @depth3_same_owner_atomic_if
  // ASP-LABEL: @depth3_same_owner_atomic_if
  tt.func @depth3_same_owner_atomic_if(%lb: i32, %ub: i32, %step: i32,
                                       %cond: i1) {
    // SEMA: [[A_BASE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32}
    // SEMA: [[B_BASE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32}
    // SEMA: [[C_BASE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32}
    // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] released = 7 {pending_count = 1 : i32}
    // SEMA: [[AB_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[C_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // SEMA: [[IF_BACK:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] {pending_count = 1 : i32}
    // ASP: [[ENTRY:%.*]] = nvws.semaphore.create {{.*}} released = 7 {pending_count = 1 : i32}
    // ASP: [[AB_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[C_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // ASP: [[IF_BACK:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %a_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %b_value = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #blocked64>
    %c_value = arith.constant dense<2.000000e+00> : tensor<128x128xf16, #blocked128>

    // SEMA: scf.for
    // ASP: %{{[0-9]+}}:5 = scf.for {{.*}} iter_args([[CURSOR:%.*]] = %{{[-A-Za-z0-9_.$#]+}},
    scf.for %iv = %lb to %ub step %step : i32 {
      // Both writes share the single ENTRY acquire of this iteration.
      // SEMA: [[AB_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // SEMA: [[AB_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]], [[AB_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // SEMA: ttg.local_store {{.*}}, [[AB_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: [[AB_SLOT:%.*]] = arith.select {{.*}} : i32
      // ASP: [[AB_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[AB_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: [[AB_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]][[[AB_SLOT]]], [[AB_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: ttg.local_store {{.*}}, [[AB_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      ttg.local_store %a_value, %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // SEMA: ttg.local_store {{.*}}, [[AB_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // SEMA: nvws.semaphore.release [[AB_FULL]], [[AB_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: ttg.local_store {{.*}}, [[AB_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // ASP: nvws.semaphore.release [[AB_FULL]][[[AB_SLOT]]], [[AB_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      ttg.local_store %b_value, %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // Both reads share the single AB_FULL acquire.
      // SEMA: [[AB_READ_TOK:%.*]] = nvws.semaphore.acquire [[AB_FULL]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: [[AB_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]], [[AB_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // SEMA: ttg.local_load [[AB_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[AB_READ_TOK:%.*]] = nvws.semaphore.acquire [[AB_FULL]][[[AB_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: [[AB_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: ttg.local_load [[AB_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %a_read = ttg.local_load %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      // SEMA: ttg.local_load [[AB_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // ASP: ttg.local_load [[AB_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %b_read = ttg.local_load %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_ab"(%a_read, %b_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>, tensor<128x64xf16, #blocked64>) -> ()
      // The scf.if yields its owner-0 token.  ASP also returns the shared slot
      // and the phase words updated inside the branch.
      // SEMA: [[IF_TOKEN:%.*]] = scf.if
      // ASP: [[IF_RESULTS:%.*]]:4 = scf.if
      scf.if %cond {
        // C is rendered through the C member of AB_FULL's buffer tuple: the
        // atomic region reuses the shared slot without a cursor advance.
        // SEMA: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]], [[AB_READ_TOK]] {ttg.partition = array<i32: 0>}
        // SEMA: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // SEMA: nvws.semaphore.release [[C_FULL]], [[AB_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        // ASP: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]] {ttg.partition = array<i32: 0>}
        // ASP: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // ASP: nvws.semaphore.release [[C_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        ttg.local_store %c_value, %c {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked128> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // The C read hands ownership back to owner 0 before the branch yields.
        // SEMA: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]] {ttg.partition = array<i32: 1>}
        // SEMA: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // SEMA: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // SEMA: nvws.semaphore.release [[IF_BACK]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // SEMA: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]] {ttg.partition = array<i32: 0>}
        // SEMA: scf.yield {{.*}}[[IF_THEN_TOKEN]] : !ttg.async.token
        // ASP: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[AB_SLOT]], {{%.*}}] {ttg.partition = array<i32: 1>}
        // ASP: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]][[[AB_SLOT]]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // ASP: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // ASP: nvws.semaphore.release [[IF_BACK]][[[AB_SLOT]]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // ASP: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]][[[AB_SLOT]], {{%.*}}] {ttg.partition = array<i32: 0>}
        // ASP: scf.yield {{.*}}[[IF_THEN_TOKEN]], [[AB_SLOT]],
        %c_read = ttg.local_load %c {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked128>
        "consume_c"(%c_read) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked128>) -> ()
      } else {
        // Without the C wave, the branch passes the A/B-read token through.
        // SEMA: scf.yield {{.*}}[[AB_READ_TOK]] : !ttg.async.token
        // ASP: scf.yield {{.*}}[[AB_READ_TOK]], [[AB_SLOT]],
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      // SEMA: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // ASP: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1, 2>, array<i32: 0>, array<i32: 1>]}
      // SEMA: nvws.semaphore.release [[ENTRY]], [[IF_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: nvws.semaphore.release [[ENTRY]][[[IF_RESULTS]]#1], [[IF_RESULTS]]#0 [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // ASP: scf.yield {{.*}}[[IF_RESULTS]]#1,
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}

//--- insert_semas_sequential_ws_loops.mlir

// Sequential-loop seam golden: one buffer is shared by two sequential
// ws-tagged loops. This pins the seam
// behavior of the CURRENT pass so any change shows up in review.
//
// What this pins (point-of-use construction): neither loop carries a
// token iter_arg. V2, loop A's full semaphore, is created initially
// released so it supplies iteration zero; A's store acquires it in-body
// directly before the store and A's consumer releases it each
// iteration. The seam is one acquire/release pair at function scope
// between the loops: `acquire V2` drains A's final permit and
// `release V5, <that token>` converts it into loop B's entry permit.
// Loop B repeats the same in-body shape on V5 (store acquire, consumer
// release) and holds one surplus permit on V5 at exit. V3 and V4 are
// the intra-loop store->load edges of A and B respectively, released
// then acquired within each iteration.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @sequential_ws_loops_shared_buffer
  tt.func @sequential_ws_loops_shared_buffer(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 950 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 950 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producerA"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V7]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V3]], [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V10:%.*]] = ttg.local_load [[V9]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V2]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumerA"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: nvws.semaphore.release [[V5]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %j = %lb to %ub step %step : i32 {
      %w = "producerB"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V5]], [[V12]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V13]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %w, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[V4]], [[V12]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V4]], [[V14]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[V16:%.*]] = ttg.local_load [[V15]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      %m = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      // CHECK: nvws.semaphore.release [[V5]], [[V14]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumerB"(%m) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}

//--- insert_semas_slack_zero_delay_schedule.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // The q recurrence forces owner 2 one wave after owner 0:
  //   q store {2}, stage 2 -> q load {0}, stage 4
  //   q load {0}, stage 4 -> next q store {2}, distance 1, delay +1
  // The p current handoff has raw delay zero but one wave of solved slack:
  //   p store {0}, stage 4 -> p load {2}, stage 4
  // while the reverse recurrence is tight at delay -1. Schedule projection
  // must not turn the slack current handoff and tight recurrence into opposite
  // static cluster edges.
  // CHECK-LABEL: tt.func @slack_zero_delay_edge_is_not_projected
  // CHECK: [[Q_BACKING:%.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 430 : i32}
  // CHECK: [[Q_EMPTY:%.*]] = nvws.semaphore.create [[Q_BACKING]] released = 1
  // CHECK: [[Q_FULL:%.*]] = nvws.semaphore.create [[Q_BACKING]]
  // CHECK: [[P_BACKING:%.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 431 : i32}
  // CHECK: [[P_EMPTY:%.*]] = nvws.semaphore.create [[P_BACKING]] released = 1
  // CHECK: [[P_FULL:%.*]] = nvws.semaphore.create [[P_BACKING]]
  // CHECK: scf.for
  // CHECK: [[Q_WRITE_BUF:%.*]] = nvws.semaphore.buffer [[Q_EMPTY]], %{{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>}
  // CHECK: ttg.local_store %{{.*}}, [[Q_WRITE_BUF]] {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>}
  // CHECK: [[Q_READ_BUF:%.*]] = nvws.semaphore.buffer [[Q_FULL]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>}
  // CHECK: [[Q_READ:%.*]] = ttg.local_load [[Q_READ_BUF]] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>}
  // CHECK: [[P_WRITE_BUF:%.*]] = nvws.semaphore.buffer [[P_EMPTY]], %{{.*}} {loop.cluster = 1 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>}
  // CHECK: ttg.local_store [[Q_READ]], [[P_WRITE_BUF]] {loop.cluster = 1 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>}
  // CHECK: [[P_READ_BUF:%.*]] = nvws.semaphore.buffer [[P_FULL]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
  // CHECK: ttg.local_load [[P_READ_BUF]] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
  tt.func @slack_zero_delay_edge_is_not_projected(
      %lb: i32, %ub: i32, %step: i32) {
    %q = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 430 : i32} :
        () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %p = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 431 : i32} :
        () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %q_value = "q_producer"() {loop.cluster = 0 : i32, loop.stage = 2 : i32,
                                  ttg.partition = array<i32: 2>} :
          () -> tensor<1xi32, #blocked>
      ttg.local_store %q_value, %q {loop.cluster = 0 : i32,
                                    loop.stage = 2 : i32,
                                    ttg.partition = array<i32: 2>} :
          tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %q_read = ttg.local_load %q {loop.cluster = 0 : i32,
                                   loop.stage = 4 : i32,
                                   ttg.partition = array<i32: 0>} :
          !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>

      ttg.local_store %q_read, %p {loop.cluster = 0 : i32,
                                   loop.stage = 4 : i32,
                                   ttg.partition = array<i32: 0>} :
          tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %p_read = ttg.local_load %p {loop.cluster = 0 : i32,
                                   loop.stage = 4 : i32,
                                   ttg.partition = array<i32: 2>} :
          !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      "consume"(%p_read) {loop.cluster = 0 : i32, loop.stage = 4 : i32,
                           ttg.partition = array<i32: 2>} :
          (tensor<1xi32, #blocked>) -> ()
    } {tt.scheduled_max_stage = 4 : i32, tt.warp_specialize,
       ttg.partition = array<i32: 0, 2>,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_staged_pou.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @staged_tokenless_cross_stage_pou
  tt.func @staged_tokenless_cross_stage_pou(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 991 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // The cross-iteration EMPTY edge goes from stage 1 to stage 0.  POU keeps
    // both acquires inside the inner loop; neither loop carries an async token.
    // The consecutive checks after FULL also prove that no third semaphore or
    // pre-loop acquire was inserted.
    // CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
    scf.for %outer = %lb to %ub step %step : i32 {
      // CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
      scf.for %inner = %lb to %ub step %step : i32 {
        // CHECK-NEXT: [[P0_TOKEN:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        // CHECK-NEXT: [[P0_BUFFER:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[P0_TOKEN]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        // CHECK-NEXT: "touch0"([[P0_BUFFER]]) {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        "touch0"(%alloc) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
        // CHECK-NEXT: "touch1"([[P0_BUFFER]]) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
        // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[P0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
        "touch1"(%alloc) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
        // CHECK-NEXT: [[P1_TOKEN:%.*]] = nvws.semaphore.acquire [[FULL]] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[P1_BUFFER:%.*]] = nvws.semaphore.buffer [[FULL]], [[P1_TOKEN]] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: "touch2"([[P1_BUFFER]]) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[P1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        "touch2"(%alloc) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      // CHECK-NEXT: } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>}
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>}
    // CHECK-NEXT: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- insert_semas_tail_schedule.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#scalar = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#local_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Each TMEM group sees its own partition-1 access at cluster 1 or 3; the
  // independent local buffer adds a partition-1 access at cluster 4. The
  // inner loop carries no tokens: each group's regain is emitted after the
  // inner loop in partition 1, acquiring the group's last-round semaphore
  // and releasing its entry semaphore for the next outer tile.
  // CHECK-LABEL: @cross_group_tail_acquire_schedule
  tt.func @cross_group_tail_acquire_schedule(
      %lhs: !ttg.memdesc<128x64xf32, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf32, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %frontier = ttg.local_alloc {buffer.id = 402 : i32} : () -> !ttg.memdesc<1xi32, #local_shared, #smem, mutable>

    // Both in-loop tmem_allocs are hoisted; each group gets an initially
    // released entry semaphore plus two in-loop semaphores.
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 400 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V5]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V5]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V5]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32) : i32
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0) -> (i32) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V2]], [[V9]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V10]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[V4]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_a, %tok_a = ttng.tmem_alloc %cst {buffer.id = 400 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V6]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[V8]], [[V11]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_b, %tok_b = ttng.tmem_alloc %cst {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32
      %inner:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%a_token = %tok_a, %b_token = %tok_b) -> (!ttg.async.token, !ttg.async.token) : i32 {
        // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V4]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V14:%.*]] = nvws.semaphore.buffer [[V4]], [[V13]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V14]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[V3]], [[V13]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %mma_a = ttng.tc_gen5_mma %lhs, %rhs, %acc_a[%a_token], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V16:%.*]] = nvws.semaphore.buffer [[V3]], [[V15]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V16]][] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V4]], [[V15]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %a_value, %a_read = ttng.tmem_load %acc_a[%mma_a] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume_a"(%a_value) {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

        // CHECK: [[V17:%.*]] = nvws.semaphore.acquire [[V8]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V18:%.*]] = nvws.semaphore.buffer [[V8]], [[V17]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[V18]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[V7]], [[V17]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %mma_b = ttng.tc_gen5_mma %lhs, %rhs, %acc_b[%b_token], %true, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: [[V19:%.*]] = nvws.semaphore.acquire [[V7]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V20:%.*]] = nvws.semaphore.buffer [[V7]], [[V19]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V20]][] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[V8]], [[V19]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %b_value, %b_read = ttng.tmem_load %acc_b[%mma_b] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume_b"(%b_value) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

        %frontier_value = "frontier_value"() {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : () -> tensor<1xi32, #scalar>
        // CHECK: ttg.local_store {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        ttg.local_store %frontier_value, %frontier {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<1xi32, #scalar> -> !ttg.memdesc<1xi32, #local_shared, #smem, mutable>

        scf.yield {ttg.partition = array<i32: 0, 1>} %a_read, %b_read : !ttg.async.token, !ttg.async.token
      } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}
      // The token results and yields of the inner loop are dropped entirely.
      // CHECK: } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}

      %next = arith.addi %tile, %c0 {ttg.partition = array<i32: 0>} : i32
      // Loop-close regains: partition 1 acquires each group's last-round
      // semaphore and releases its entry semaphore for the next outer tile.
      // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[V2]], [[V21]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[V6]], [[V22]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_tmem_alias.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_reinterpret_alias
  // META-LABEL: @tmem_reinterpret_alias
  tt.func @tmem_reinterpret_alias(%ub: i32) {
    // CHECK: [[V0:%.*]] = ub.poison : !ttg.async.token
    // META: [[V0:%.*]] = ub.poison : !ttg.async.token
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 3 {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // META: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // META: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // META: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %alloc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V5:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V4:%.*]] = [[V0]]) -> (!ttg.async.token)  : i32 {
    // META: [[V5:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V4:%.*]] = [[V0]]) -> (!ttg.async.token)  : i32 {
    %r = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%t = %tok) -> (!ttg.async.token) : i32 {
      // CHECK: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V8:%.*]] = ttg.memdesc_reinterpret [[V7]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // META: [[V6:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // META: [[V7:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // META: [[V8:%.*]] = ttg.memdesc_reinterpret [[V7]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %view0 = ttg.memdesc_reinterpret %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V9:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // META: [[V9:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %t0 = ttng.tmem_store %cst, %view0[%t], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[V12:%.*]] = ttg.memdesc_reinterpret [[V11]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // META: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // META: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // META: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // META: [[V12:%.*]] = ttg.memdesc_reinterpret [[V11]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %view1 = ttg.memdesc_reinterpret %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // META: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V12]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %val, %t1 = ttng.tmem_load %view1[%t0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // META: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use"(%val) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      // The source token thread is dropped: the yield forwards the poison
      // placeholder, not an acquire token (loop-close release partition 1
      // differs from the first-acquire partition 0, so no token is carried).
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V0]] : !ttg.async.token
      // META: scf.yield {ttg.partition = array<i32: 0, 1>} [[V0]] : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %t1 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: "use_token"([[V5]])
    // META: "use_token"([[V5]])
    "use_token"(%r) : (!ttg.async.token) -> ()
    tt.return
  }
}

//--- insert_semas_tmem_container_subviews.mlir

// v4 container-pattern tests. All multi-member buffer.id groups in real
// Triton IR have one member that acts as the physical slot container
// (largest extent, covers [0, slot_size)); other members are sub-views
// inside the container. Union-find on overlap intervals unites all
// members through the container, so each buffer.id group collapses to
// a single resourceKey.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked256 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem256 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Container m0 = [0, 256). Disjoint sub-views m1 = [0, 128),
  // m2 = [128, 192), m3 = [192, 256). m0 unions with m1, m2, m3 via
  // overlap; m1, m2, m3 are pairwise disjoint. All collapse to one
  // resourceKey via m0.

  // CHECK-LABEL: @container_with_disjoint_subviews
  tt.func @container_with_disjoint_subviews(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst256 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked256>
    %cst128 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked64>
    %true = arith.constant true

    // Container alloc, then the three sub-view memdescs that union into it.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc {buffer.id = 900 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[S3:%.*]] = ttng.tmem_subslice [[ALLOC]] {offset = 192 : i32}
    // CHECK: [[M3:%.*]] = ttg.memdesc_reinterpret [[S3]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[S2:%.*]] = ttng.tmem_subslice [[ALLOC]] {offset = 128 : i32}
    // CHECK: [[M2:%.*]] = ttg.memdesc_reinterpret [[S2]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[S1:%.*]] = ttng.tmem_subslice [[ALLOC]] {offset = 0 : i32}
    // CHECK: [[M1:%.*]] = ttg.memdesc_reinterpret [[S1]] : !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>

    // One EMPTY semaphore (pending=3) and three FULL semaphores (pending=1),
    // all over the single collapsed resourceKey of four memdesc members.
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]], [[M3]] released = 3 {pending_count = 3 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F0:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]], [[M3]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F1:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]], [[M3]] {pending_count = 1 : i32}
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]], [[M3]] {pending_count = 1 : i32}

    // No carried token: EMPTY is initially released and acquired inside the loop body.
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {

      // Producer (p0): acquire EMPTY at point of use, buffer the container
      // slot, store the full extent, and release it to p1.
      // CHECK: [[A0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]]:4 = nvws.semaphore.buffer [[EMPTY]], [[A0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B0]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      %m0, %t0 = ttng.tmem_alloc %cst256 {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x256xf32, #blocked256>) -> (!ttg.memdesc<128x256xf32, #tmem256, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %v0, %l0 = ttng.tmem_load %m0[%t0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x256xf32, #tmem256, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked256>
      %m1, %t1 = ttng.tmem_alloc %cst128 {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m2, %t2 = ttng.tmem_alloc %cst64 {buffer.id = 900 : i32, buffer.offset = 128 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> (!ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m3, %t3 = ttng.tmem_alloc %cst64 {buffer.id = 900 : i32, buffer.offset = 192 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf32, #blocked64>) -> (!ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Whole-buffer reader p1 acquires F0 and reads all three pieces.
      // CHECK: nvws.semaphore.release [[F0]], [[A0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK-NOT: nvws.semaphore.release [[F1]], [[A0]]
      // CHECK-NOT: nvws.semaphore.release [[F2]], [[A0]]
      // CHECK: [[A1:%.*]] = nvws.semaphore.acquire [[F0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B1:%.*]]:4 = nvws.semaphore.buffer [[F0]], [[A1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B1]]#0[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256> -> tensor<128x256xf32, #blocked>

      // Reader p1 sends P1 and P2 to their writers, then writes P0 itself.
      // CHECK: nvws.semaphore.release [[F1]], [[A1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[F2]], [[A1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B1]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>

      // Sub-view writer p2 (m2 = [128,192)): acquire F1, buffer, store member #2.
      // CHECK: [[A2:%.*]] = nvws.semaphore.acquire [[F1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B2:%.*]]:4 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B2]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>

      // Sub-view writer p3 (m3 = [192,256)): acquire F2, buffer, store member #3.
      // CHECK: [[A3:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B3:%.*]]:4 = nvws.semaphore.buffer [[F2]], [[A3]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B3]]#3, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      %v1, %l1 = ttng.tmem_load %m1[%t1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %v2, %l2 = ttng.tmem_load %m2[%t2] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked64>
      %v3, %l3 = ttng.tmem_load %m3[%t3] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked64>

      // Reader p1 reuses its retained F0 token, loads member #1, and recycles EMPTY.
      // CHECK: [[B4:%.*]]:4 = nvws.semaphore.buffer [[F0]], [[A1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B4]]#1[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Reader p2 reuses its retained F1 token, loads member #2, and recycles EMPTY.
      // CHECK: [[B5:%.*]]:4 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B5]]#2[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Reader p3 reuses its retained F2 token, loads member #3, and recycles EMPTY.
      // CHECK: [[B6:%.*]]:4 = nvws.semaphore.buffer [[F2]], [[A3]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B6]]#3[] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use1"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      "use2"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> ()
      "use3"(%v3) {ttg.partition = array<i32: 3>} : (tensor<128x64xf32, #blocked64>) -> ()
      "use0"(%v0) {ttg.partition = array<i32: 1>} : (tensor<128x256xf32, #blocked256>) -> ()

      // No bottom re-acquire: the yield carries only the original i32.
      // CHECK-NOT: nvws.semaphore.acquire
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %{{[-A-Za-z0-9_.$#]+}} : i32
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2, 3>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3>], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked256 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem256 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Container m0 = [0, 256). Two overlapping sub-views m1 = [0, 128),
  // m2 = [64, 128). m1 and m2 overlap each other AND both overlap m0.
  // All three collapse to one resourceKey.

  // CHECK-LABEL: @container_with_overlapping_subviews
  tt.func @container_with_overlapping_subviews(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst256 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked256>
    %cst128 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked64>
    %true = arith.constant true

    // Container alloc, then two overlapping sub-view memdescs that union into it.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc {buffer.id = 901 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[S2:%.*]] = ttng.tmem_subslice [[ALLOC]] {offset = 64 : i32}
    // CHECK: [[M2:%.*]] = ttg.memdesc_reinterpret [[S2]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[S1:%.*]] = ttng.tmem_subslice [[ALLOC]] {offset = 0 : i32}
    // CHECK: [[M1:%.*]] = ttg.memdesc_reinterpret [[S1]] : !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x256> -> !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>

    // One EMPTY semaphore (pending=2) and three FULL semaphores (pending=1)
    // over the single collapsed resourceKey of three memdesc members.
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]] released = 3 {pending_count = 2 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F0:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]] {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[F1:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]] {pending_count = 1 : i32}
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]], [[M1]], [[M2]] {pending_count = 1 : i32}

    // No carried token here: EMPTY is acquired inside the loop body.
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {

      // Producer (p0): acquire EMPTY, buffer the container slot, store full extent, release F0.
      // CHECK: [[A0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]]:3 = nvws.semaphore.buffer [[EMPTY]], [[A0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B0]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>
      // CHECK: nvws.semaphore.release [[F0]], [[A0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %m0, %t0 = ttng.tmem_alloc %cst256 {buffer.id = 901 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x256xf32, #blocked256>) -> (!ttg.memdesc<128x256xf32, #tmem256, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m1, %t1 = ttng.tmem_alloc %cst128 {buffer.id = 901 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m2, %t2 = ttng.tmem_alloc %cst64 {buffer.id = 901 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> (!ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Sub-view writer p1 (m1 = [0,128)): acquire F0, buffer, store member #1, release F1.
      // CHECK: [[A1:%.*]] = nvws.semaphore.acquire [[F0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B1:%.*]]:3 = nvws.semaphore.buffer [[F0]], [[A1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B1]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: nvws.semaphore.release [[F1]], [[A1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Sub-view writer p2 (m2 = [64,128)): acquire F1, buffer, store member #2, release F2.
      // CHECK: [[A2:%.*]] = nvws.semaphore.acquire [[F1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B2:%.*]]:3 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B2]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: nvws.semaphore.release [[F2]], [[A2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %v1, %l1 = ttng.tmem_load %m1[%t1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %v2, %l2 = ttng.tmem_load %m2[%t2] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked64>

      // Reader p1: acquire F2, buffer, load member #1, and recycle EMPTY.
      // CHECK: [[A3:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B3:%.*]]:3 = nvws.semaphore.buffer [[F2]], [[A3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B3]]#1[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Reader p2 reuses its retained F1 token, loads member #2, and recycles EMPTY.
      // CHECK: [[B4:%.*]]:3 = nvws.semaphore.buffer [[F1]], [[A2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x256>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B4]]#2[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[A2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use1"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      "use2"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{[-A-Za-z0-9_.$#]+}} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_tmem_no_loop_exit_drain.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_loop_carried_linear_chain_no_exit_drain
  tt.func @tmem_loop_carried_linear_chain_no_exit_drain(
      %lb: i32, %ub: i32, %step: i32,
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %cst_f16 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: [[V1:%.*]], [[V2:%.*]] = ttng.tmem_alloc %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc, %acc_tok = ttng.tmem_alloc %cst_f32 {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: [[V3:%.*]] = ttng.tmem_alloc {buffer.id = 920 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V3]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V3]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V7:%.*]] = [[V2]]) -> (i32, !ttg.async.token)  : i32 {
    %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0, %tok = %acc_tok) -> (i32, !ttg.async.token) : i32 {
      // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V4]], [[V9]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V10]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 5>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %frag = ttng.tmem_alloc %cst_f16 {buffer.id = 920 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V5]], [[V9]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V5]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: [[V13:%.*]] = ttng.tc_gen5_mma [[V12]], %{{[-A-Za-z0-9_.$#]+}}, [[V1]]{{\[}}[[V7]]{{\]}}, %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %read = ttng.tc_gen5_mma %frag, %rhs, %acc[%tok], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %next_i = arith.addi %i, %c0 {ttg.partition = array<i32: 5>} : i32
      // CHECK: nvws.semaphore.release [[V4]], [[V11]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 5>} %{{[-A-Za-z0-9_.$#]+}}, [[V13]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 1, 5>} %next_i, %read : i32, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 5>, ttg.partition.outputs = [array<i32: 5>, array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}

    // No loop-exit drain: no acquire/release may appear after the loop.
    // CHECK-NOT: nvws.semaphore.acquire
    // CHECK-NOT: nvws.semaphore.release
    "use"(%loop#0) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_tmem_reuse_views.mlir

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // This function emits NO semaphores: enforce their total absence.
  // CHECK-LABEL: @tmem_reuse_views_end_of_insert_semas
  // CHECK-NOT: nvws.semaphore
  tt.func @tmem_reuse_views_end_of_insert_semas() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %alias = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    %base = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %half = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    "use"(%alias, %base, %half) : (!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>) -> ()

    scf.for %iv = %c0 to %c1 step %c1 {
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}

    tt.return
  }
}

//--- insert_semas_transitive_reduction.mlir

// Transitive reduction (spec TRANSITIVE REDUCTION section): implied
// same-chain edges are dropped pay-for-play — but never wave-opening
// acquires (the wave guard) and never across distinct destinations'
// closed waves. Corner cases pinned here:
//   1. serialized ring: the {0}->{2} fan-in arm is implied through
//      {0}->{1}->{2} and DROPPED (one release per handoff survives);
//   2. genuine fan-out to two reader partitions: both edges are wave
//      openers — NOTHING is dropped;
//   3. a returning owner with an earlier token reuses that token without
//      adding a carrier-only handoff.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @serialized_ring_reduces
  tt.func @serialized_ring_reduces(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant {ttg.partition = array<i32: 0>} dense<0.0> : tensor<256x128xf16, #blocked>
    %cst1 = arith.constant {ttg.partition = array<i32: 2>} dense<1.0> : tensor<128x128xf16, #blocked>
    // Containment shape (the planner-realistic layout): the offset-0 owner
    // m0 spans [0,256) and the offset-64 reuser m1[64,192) nests inside it.
    // Both members of buffer.id 500 are co-allocated into one semaphore
    // set. One initially released EMPTY and three blocked FULL handoff semaphores
    // serialize the {0}->{1}->{2}->{0} chain.
    // CHECK: [[A:%.*]] = ttg.local_alloc {buffer.id = 500 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>
    // CHECK: [[B:%.*]] = ttg.local_alloc {buffer.id = 500 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[A]], [[B]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F01:%.*]] = nvws.semaphore.create [[A]], [[B]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F12:%.*]] = nvws.semaphore.create [[A]], [[B]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F20:%.*]] = nvws.semaphore.create [[A]], [[B]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // Owner/reuser pair of one buffer.id: {0} writes a (the spanning
    // owner), {1} reads a, {2} writes b (nested inside a), {0} reads b.
    // The W-after-R edge {0}->{2} for the overlap piece is implied via
    // {0}->{1}->{2} and dropped (in-chain sweep). The minimal serialized
    // chain survives: three handoffs plus the carrier close — four
    // releases. No piece is reuser-only, so every EXIT carries owner {0}
    // and no {2} regain edge is raised at all.
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // {0} acquires EMPTY, writes a, releases FULL for {1}.
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF0:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: ttg.local_store %{{.*}}, [[BUF0]]#0 {ttg.partition = array<i32: 0>} : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[F01]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %a = ttg.local_alloc %cst0 {buffer.id = 500 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
      // {1} acquires FULL, reads a, releases FULL for {2} and EMPTY back.
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F01]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF1:%.*]]:2 = nvws.semaphore.buffer [[F01]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: ttg.local_load [[BUF1]]#0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[F12]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<256x128xf16, #blocked>) -> ()
      // {2} acquires FULL, writes b, releases FULL for {0}.
      // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F12]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF2:%.*]]:2 = nvws.semaphore.buffer [[F12]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable, 1x256x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[BUF2]]#1 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[F20]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %b = ttg.local_alloc %cst1 {buffer.id = 500 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // {0} acquires FULL, reads b. No re-release: every piece's EXIT
      // carries owner {0}, so no {2} regain edge exists to close here.
      // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F20]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF3:%.*]]:2 = nvws.semaphore.buffer [[F20]], [[T3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable, 1x256x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[BUF3]]#1 {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %vb = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      // Exactly four releases survive across the body (F01, F12+EMPTY, F20).
      // No carrier token threaded through scf.yield in this serialized ring.
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{.*}} : i32
      %j = arith.addi %i, %iv {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
      // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#scalar_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#scalar_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#scalar_smem = #ttg.shared_memory
!scalar_ty = tensor<1xi32, #scalar_blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // The producer reread moves holder {0}'s WAR row, but it does not replace
  // the current version's RAW source. The foreign writer must wait for both
  // readers, including the producer reread.
  // CHECK-LABEL: @stable_producer_row_and_foreign_war
  tt.func @stable_producer_row_and_foreign_war(%lb: i32, %ub: i32,
                                                %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 991 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[READY:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[WAR:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // No pre-loop seed acquire and no token iter_arg: EMPTY is created
    // initially released and acquired at the point of use inside the body.
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    %alloc = ttg.local_alloc {buffer.id = 991 : i32} : () -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %w0 = "w0"() {ttg.partition = array<i32: 0>} : () -> !scalar_ty
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[READY]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %w0, %alloc {ttg.partition = array<i32: 0>} : !scalar_ty -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
      // The reread uses the same producer token and buffer view, then
      // contributes a real WAR arrival to the later foreign writer.
      // CHECK: ttg.local_load [[B0]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      // CHECK: nvws.semaphore.release [[WAR]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %r0 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable> -> !scalar_ty
      "use0"(%r0) {ttg.partition = array<i32: 0>} : (!scalar_ty) -> ()
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[READY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[READY]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      // CHECK: nvws.semaphore.release [[WAR]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %r1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable> -> !scalar_ty
      "use1"(%r1) {ttg.partition = array<i32: 1>} : (!scalar_ty) -> ()
      %w2 = "w2"() {ttg.partition = array<i32: 2>} : () -> !scalar_ty
      // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[WAR]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[WAR]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // The foreign writer closes the ring: its release of EMPTY supplies
      // the producer's next-iteration acquire.
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %w2, %alloc {ttg.partition = array<i32: 2>} : !scalar_ty -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#scalar_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#scalar_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#scalar_smem = #ttg.shared_memory
!scalar_ty = tensor<1xi32, #scalar_blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // An outer write is represented by the child ENTER. The second child reader
  // fans out from ENTER, not from the first child reader or the outer row.
  // CHECK-LABEL: @region_resets_producer_row_to_enter
  tt.func @region_resets_producer_row_to_enter(%lb: i32, %ub: i32,
                                                %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 992 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // {2}'s turn semaphore is created before {1}'s; {0} hands off to [[R1]].
    // CHECK: [[R2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[R1:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 992 : i32} : () -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      %w0 = "w0"() {ttg.partition = array<i32: 0>} : () -> !scalar_ty
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[R1]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %w0, %alloc {ttg.partition = array<i32: 0>} : !scalar_ty -> !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable>
      // No token iter_arg on the inner loop: {1} acquires [[R1]] at its
      // point of use inside the body.
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      scf.for %j = %lb to %ub step %step : i32 {
        // The entry acquire of [[R1]] is the inner chain's source: its
        // fan-out release to [[R2]] precedes the first reader's load, which
        // reuses the same token.
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[R1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: nvws.semaphore.release [[R2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[R1]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        %r1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable> -> !scalar_ty
        "use1"(%r1) {ttg.partition = array<i32: 1>} : (!scalar_ty) -> ()
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[R2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[R2]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B2]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
        // {2}'s release of [[R1]] feeds the next inner iteration's entry
        // acquire.
        // CHECK: nvws.semaphore.release [[R1]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %r2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #scalar_shared, #scalar_smem, mutable> -> !scalar_ty
        "use2"(%r2) {ttg.partition = array<i32: 2>} : (!scalar_ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
      // Loop close: after the inner region, {1} re-acquires [[R1]] (the last
      // inner release) and returns EMPTY to the outer producer.
      // CHECK: } {ttg.partition = array<i32: 1, 2>}
      // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[R1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @fanout_not_reduced
  tt.func @fanout_not_reduced(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant {ttg.partition = array<i32: 0>} dense<0.0> : tensor<128x128xf16, #blocked>
    // One producer, two independent reader partitions: both edges open
    // their waves — the reduction must keep both reader acquires. The
    // EMPTY semaphore fans out (pending_count = 2) to two FULL sems.
    // CHECK: [[BUF:%.*]] = ttg.local_alloc {buffer.id = 501 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BUF]] released = 1 {pending_count = 2 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F1:%.*]] = nvws.semaphore.create [[BUF]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[BUF]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // No pre-loop seed acquire and no token iter_arg: the producer acquires
    // EMPTY at its point of use inside the body.
    // CHECK: %{{.*}} = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}) -> (i32) : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // {0} acquires EMPTY in-body, writes a, releases to both readers.
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[PBUF]] {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[F1]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[F2]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %a = ttg.local_alloc %cst0 {buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Reader {1}: wave-opening acquire of F1 (KEPT, not reduced).
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[RBUF1:%.*]] = nvws.semaphore.buffer [[F1]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[RBUF1]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %v1 = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      // Reader {2}: wave-opening acquire of F2 (KEPT, not reduced).
      // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[RBUF2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[RBUF2]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %v2 = ttg.local_load %a {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
      // {0} re-reads a with its retained producer token: no handoff edge.
      // CHECK: [[RBUF0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[RBUF0]] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %v0 = ttg.local_load %a {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      // No re-acquire and no token threaded through scf.yield: the next
      // iteration's in-body acquire waits on both reader releases.
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{.*}} : i32
      %j = arith.addi %i, %iv {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
      // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

//--- insert_semas_uniform_hold_transparency.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // S1: prefix For, p1-anchored transparent region.
  // The p1 hold is carried into the inner loop (transparent): the store acquires
  // EMPTY once at the top, the inner loop carries that same token, and the trailing
  // p2 read after the inner loop gates a fresh acquire on the regained handle.
  // CHECK-LABEL: @uniform_hold_s1_prefix_for_p1
  tt.func @uniform_hold_s1_prefix_for_p1(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 981 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F4:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 981 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[CARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F2]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 2>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
        // CHECK: nvws.semaphore.release [[F4]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F4]], [[T3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S2: owner-change at op1 -> inner; negative expected.
  // The p1 hold cuts at op1 (store releases F3 immediately); the inner loop stays
  // plain — each iteration's p2 load acquires F3 at its point of use — and the
  // trailing p2 read takes its own F3 acquire after the loop.
  // CHECK-LABEL: @uniform_hold_s2_owner_change_cut
  tt.func @uniform_hold_s2_owner_change_cut(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 982 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 982 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F3]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F3]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F3]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>}
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F3]], [[T3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S3: trailing read after regain; negative expected.
  // A function-level EMPTY acquire threads the outer and middle loops. The p1
  // store reuses the carried handle and releases F3; the inner loop stays plain
  // (point-of-use acquires). A p2 close-acquire on F3 after the inner loop hands
  // EMPTY back, and the trailing p1 read's fresh EMPTY acquire is the yielded carrier.
  // CHECK-LABEL: @uniform_hold_s3_trailing_read_after_regain
  tt.func @uniform_hold_s3_trailing_read_after_regain(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 983 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 983 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: [[MID:%.*]] = scf.for {{.*}} iter_args([[MCARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[MCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F3]], [[MCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F3]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F3]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>}
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[T4:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 1>}
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T4]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[MID]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S4: prefix For, p2-anchored owner mirror of S1.
  // CHECK-LABEL: @uniform_hold_s4_prefix_for_p2
  tt.func @uniform_hold_s4_prefix_for_p2(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 984 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F4:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 984 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 2>}
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[CARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRY]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F2]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
        // CHECK: nvws.semaphore.release [[F4]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F4]], [[T3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S5: region-spanning at WS-body depth 1, no middle, no op4.
  // A function-level EMPTY acquire is carried by the outer loop; the p1 store
  // buffers on the carried token. The inner loop carries it too and yields the
  // bottom p1 re-acquire; the outer loop carries the regained handle onward
  // (no post-loop drain).
  // CHECK-LABEL: @uniform_hold_s5_ws_body_depth1
  tt.func @uniform_hold_s5_ws_body_depth1(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 985 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 985 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[OCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[ICARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ICARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[ICARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[INNER]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S6: same-owner trailing read after the inner region.
  // The p1 store reuses the function-level carried EMPTY handle and releases
  // EMPTY at once (hold cut); the inner loop stays plain (point-of-use EMPTY
  // acquires), and the trailing p1 read takes a fresh EMPTY acquire whose token
  // the outer loop yields.
  // CHECK-LABEL: @uniform_hold_s6_same_owner_trailing_read
  tt.func @uniform_hold_s6_same_owner_trailing_read(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 986 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 986 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[OCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[OCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>}
      // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 1>}
      %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T3]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S7: cross-owner store out / load in.
  // p1 store acquires EMPTY then releases FULL; the inner p2 load acquires FULL
  // above the inner loop, uses the same token inside (no carrier needed — the
  // inner loop is plain), and the regained EMPTY handle is released after.
  // CHECK-LABEL: @uniform_hold_s7_cross_owner_store_load
  tt.func @uniform_hold_s7_cross_owner_store_load(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 987 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 987 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[F2]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: scf.for {{.*}}  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S8: If-as-prefix-region inside the WS body.
  // A function-level EMPTY acquire is carried by the loop; the p1 store buffers
  // on the carried token. The scf.if yields the token (bottom p1 re-acquire in
  // then, pass-through in else) and the loop carries the regained handle onward
  // (no post-if drain).
  // CHECK-LABEL: @uniform_hold_s8_if_prefix_region
  tt.func @uniform_hold_s8_if_prefix_region(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 988 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 988 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[OCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[IF:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
      scf.if %cond {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[OCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[OCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"() {ttg.partition = array<i32: 2>} : () -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
      // CHECK: } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[OCARRY]] : !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[IF]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S9: If inside the inner loop.
  // The store/load/store all live inside the conditional; a single carrier is
  // threaded from a function-level acquire through outer loop, inner loop, and
  // the scf.if (point-of-use store on the carried buffer, else passes through).
  // CHECK-LABEL: @uniform_hold_s9_if_inside_inner_loop
  tt.func @uniform_hold_s9_if_inside_inner_loop(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 989 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 989 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[ICARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[IF:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        scf.if %cond {
          // CHECK: "producer1"
          // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ICARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
          // CHECK: ttg.local_load [[B0]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F2]], [[ICARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"() {ttg.partition = array<i32: 2>} : () -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        // CHECK: } else {
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[ICARRY]] : !ttg.async.token
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[IF]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[INNER]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S10: fan-out / multi-consumer.
  // The store's fan-out sema (last create, pending_count = 2) collects two
  // consumer arrives per cycle. Both loops stay plain: the p3 store and each
  // inner iteration's p2 read acquire at point of use, the correcting store
  // takes a separate {2}->{1} WAR handoff, and a p2 close-acquire after the
  // inner loop hands EMPTY back to the p3 store.
  // CHECK-LABEL: @uniform_hold_s10_fanout_multi_consumer
  tt.func @uniform_hold_s10_fanout_multi_consumer(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 990 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F4:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F5:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 990 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer0"
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 3>}
      // CHECK: nvws.semaphore.release [[F5]], [[T0]] [#nvws.async_op<none>] {arrive_count = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %v0 = "producer0"(%i0) {ttg.partition = array<i32: 3>} : (i32) -> !ty
      ttg.local_store %v0, %alloc {ttg.partition = array<i32: 3>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F5]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: nvws.semaphore.release [[F2]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F5]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v1 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer1"(%v1) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B2]] {ttg.partition = array<i32: 1>}
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F3]], [[T3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B3]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F4]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: nvws.semaphore.release [[F5]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T4:%.*]] = nvws.semaphore.acquire [[F4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B4:%.*]] = nvws.semaphore.buffer [[F4]], [[T4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B4]] {ttg.partition = array<i32: 0>}
        // CHECK: nvws.semaphore.release [[F5]], [[T4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      } {ttg.partition = array<i32: 0, 1, 2, 3>}
      // CHECK: } {ttg.partition = array<i32: 0, 1, 2, 3>}
      // CHECK: [[T5:%.*]] = nvws.semaphore.acquire [[F5]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T5]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
