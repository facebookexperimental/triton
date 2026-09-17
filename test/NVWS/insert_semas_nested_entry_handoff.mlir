// RUN: split-file %s %t
// RUN: triton-opt %t/cases.mlir --allow-unregistered-dialect --nvws-insert-semas | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %t/cases.mlir --allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase -cse | FileCheck %s --check-prefix=ASP
// RUN: triton-opt %t/cases.mlir --allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore -cse | FileCheck %s --check-prefix=LOWER
// RUN: triton-opt %t/pipeline.mlir --allow-unregistered-dialect --nvws-insert-semas=num-stages=2 --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=2 --tritongpu-pipeline=num-stages=2 -cse | FileCheck %s --check-prefix=PIPE
// RUN: triton-opt %t/pipeline.mlir --allow-unregistered-dialect --nvws-insert-semas='num-stages=2 use-meta-partitioner=true' --nvws-semaphore-optimize=num-stages=2 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore=num-stages=2 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops='num-stages=2 use-meta-ws=true' --tritongpu-pipeline=num-stages=2 -cse | FileCheck %s --check-prefix=PIPE

// A nested recurrence must receive the permit its parent already acquired.
// Merely counting acquires misses the deadlock: an unused outer token followed
// by another acquire of the same initially released semaphore consumes the
// initial permit twice. Inputs contain no manually authored semaphore events.
// Multi-copy reader-first cases make every MMA a fresh overwrite; write-first
// controls separately cover ordinary fresh-write multi-buffering.
// Positive inner bounds ensure the post-loop observer follows an MMA. The first
// correction's value is overwritten by the first useAccumulator=false MMA.

//--- cases.mlir
#reg1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#reg = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
#tm = #ttng.tensor_memory
!acc = !ttg.memdesc<128x128xf32, #tmem, #tm, mutable>
!tile = tensor<128x128xf32, #reg>
!lhs = !ttg.memdesc<128x64xf16, #shared, #smem>
!rhs = !ttg.memdesc<64x128xf16, #shared, #smem>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @reader_first_depth_1
  // ASP-LABEL: @reader_first_depth_1
  // LOWER-LABEL: @reader_first_depth_1
  // SEMA: [[READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: scf.for {{.*}} iter_args([[OUTER_IN:%.*]] = [[ENTRY]])
  // SEMA: nvws.semaphore.release [[READY]], [[OUTER_IN]] [#nvws.async_op<none>]
  // SEMA-NEXT: {{.*}}scf.for
  // SEMA: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: ttng.tmem_load
  // SEMA: nvws.semaphore.release [[TO_MMA:%.*]], [[INNER_TOKEN]] [#nvws.async_op<none>]
  // SEMA: [[MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_MMA]]
  // SEMA: ttng.tc_gen5_mma
  // SEMA: nvws.semaphore.release [[READY]], [[MMA_TOKEN]] [#nvws.async_op<tc5mma>]
  // SEMA: [[RETURNED:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: "consume"
  // The post-inner acquisition is the next outer iteration's incoming token.
  // SEMA: scf.yield {{.*}}[[RETURNED]] : !ttg.async.token
  // ASP: [[INITIAL_SLOT:%.*]] = arith.constant 0 : i32
  // ASP: [[READY_ASP:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // ASP: [[TO_MMA_ASP:%.*]] = nvws.semaphore.create
  // ASP: [[INITIAL_BITS:%.*]] = arith.constant -2 : i32
  // ASP: [[PHASE_ONE:%.*]] = arith.constant {{.*}} 1 : i32
  // The entry mask and selected bit produce phase 1 for initial availability.
  // ASP: [[INITIAL_BIT:%.*]] = arith.shli [[PHASE_ONE]], [[INITIAL_SLOT]]
  // ASP: [[ENTRY_BITS:%.*]] = arith.xori [[INITIAL_BITS]], [[INITIAL_BIT]]
  // ASP: [[INITIAL_SHIFT:%.*]] = arith.shrui [[ENTRY_BITS]], [[INITIAL_SLOT]]
  // ASP: [[ENTRY_PHASE:%.*]] = arith.andi [[INITIAL_SHIFT]], [[PHASE_ONE]]
  // ASP: [[ENTRY_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INITIAL_SLOT]], [[ENTRY_PHASE]]]
  // ASP: scf.for {{.*}} iter_args([[OUTER_TOKEN_ASP:%.*]] = [[ENTRY_ASP]], [[OUTER_SLOT:%.*]] = [[INITIAL_SLOT]], [[OUTER_BITS:%.*]] = [[ENTRY_BITS]],
  // The relay does not select a new slot or reset the carried phase word.
  // ASP: nvws.semaphore.release [[READY_ASP]][[[OUTER_SLOT]]], [[OUTER_TOKEN_ASP]] [#nvws.async_op<none>]
  // ASP: [[INNER_RESULT:%[a-zA-Z0-9_]+]]:{{[0-9]+}} = scf.for {{.*}} iter_args({{.*}}[[INNER_SLOT:%.*]] = [[OUTER_SLOT]], [[INNER_BITS:%.*]] = [[OUTER_BITS]],
  // ASP: [[INNER_BIT:%.*]] = arith.shli {{%.*}}, [[INNER_SLOT]]
  // ASP: [[NEXT_READER_BITS:%.*]] = arith.xori [[INNER_BITS]], [[INNER_BIT]]
  // ASP: [[INNER_SHIFT:%.*]] = arith.shrui [[NEXT_READER_BITS]], [[INNER_SLOT]]
  // ASP: [[INNER_PHASE:%.*]] = arith.andi [[INNER_SHIFT]],
  // ASP: [[READ_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INNER_SLOT]], [[INNER_PHASE]]]
  // ASP: ttng.tmem_load
  // ASP: [[MMA_SLOT:%.*]] = arith.select
  // ASP: [[MMA_ASP:%.*]] = nvws.semaphore.acquire [[TO_MMA_ASP]][[[MMA_SLOT]],
  // ASP: ttng.tc_gen5_mma
  // ASP: nvws.semaphore.release [[READY_ASP]][[[MMA_SLOT]]], [[MMA_ASP]] [#nvws.async_op<tc5mma>]
  // ASP: scf.yield {{.*}}[[MMA_SLOT]], [[NEXT_READER_BITS]],
  // ASP: [[FINAL_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INNER_RESULT]]#1,
  // ASP: "consume"
  // Returning the token AND evolved slot/phase state re-arms the next outer trip.
  // ASP: scf.yield {{.*}}[[FINAL_ASP]], [[INNER_RESULT]]#1,
  // LOWER: [[READY_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // LOWER: ttng.init_barrier
  // LOWER: scf.for
  // LOWER: [[RELAY_VIEW:%.*]] = ttg.memdesc_index [[READY_STORAGE]][
  // LOWER: ttng.arrive_barrier [[RELAY_VIEW]], 1
  // LOWER-NEXT: {{.*}}scf.for
  // LOWER: [[FIRST_WAIT_VIEW:%.*]] = ttg.memdesc_index [[READY_STORAGE]][
  // LOWER: ttng.wait_barrier [[FIRST_WAIT_VIEW]],
  // LOWER: ttng.tmem_load
  tt.func @reader_first_depth_1(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @reader_first_depth_2
  // ASP-LABEL: @reader_first_depth_2
  // LOWER-LABEL: @reader_first_depth_2
  // SEMA: [[READY:%.*]] = nvws.semaphore.create {{.*}} released = {{[0-9]+}} {pending_count = 1 : i32}
  // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: scf.for {{.*}} iter_args([[OUTER_IN:%.*]] = [[ENTRY]])
  // SEMA: nvws.semaphore.release [[READY]], [[OUTER_IN]] [#nvws.async_op<none>]
  // SEMA-NEXT: {{.*}}scf.for
  // SEMA: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: ttng.tmem_load
  // SEMA: nvws.semaphore.release [[TO_MMA:%.*]], [[INNER_TOKEN]] [#nvws.async_op<none>]
  // SEMA: [[MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_MMA]]
  // SEMA: ttng.tc_gen5_mma
  // SEMA: nvws.semaphore.release [[READY]], [[MMA_TOKEN]] [#nvws.async_op<tc5mma>]
  // SEMA: [[RETURNED:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: "consume"
  // The post-inner acquisition is the next outer iteration's incoming token.
  // SEMA: scf.yield {{.*}}[[RETURNED]] : !ttg.async.token
  // ASP: [[INITIAL_SLOT:%.*]] = arith.constant 1 : i32
  // ASP: [[READY_ASP:%.*]] = nvws.semaphore.create {{.*}} released = 2 {pending_count = 1 : i32}
  // ASP: [[TO_MMA_ASP:%.*]] = nvws.semaphore.create
  // ASP: [[INITIAL_BITS:%.*]] = arith.constant -3 : i32
  // ASP: [[PHASE_ONE:%.*]] = arith.constant {{.*}} 1 : i32
  // The entry mask and selected bit produce phase 1 for initial availability.
  // ASP: [[INITIAL_BIT:%.*]] = arith.shli [[PHASE_ONE]], [[INITIAL_SLOT]]
  // ASP: [[ENTRY_BITS:%.*]] = arith.xori [[INITIAL_BITS]], [[INITIAL_BIT]]
  // ASP: [[INITIAL_SHIFT:%.*]] = arith.shrui [[ENTRY_BITS]], [[INITIAL_SLOT]]
  // ASP: [[ENTRY_PHASE:%.*]] = arith.andi [[INITIAL_SHIFT]], [[PHASE_ONE]]
  // ASP: [[ENTRY_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INITIAL_SLOT]], [[ENTRY_PHASE]]]
  // ASP: scf.for {{.*}} iter_args([[OUTER_TOKEN_ASP:%.*]] = [[ENTRY_ASP]], [[OUTER_SLOT:%.*]] = [[INITIAL_SLOT]], [[OUTER_BITS:%.*]] = [[ENTRY_BITS]],
  // The relay does not select a new slot or reset the carried phase word.
  // ASP: nvws.semaphore.release [[READY_ASP]][[[OUTER_SLOT]]], [[OUTER_TOKEN_ASP]] [#nvws.async_op<none>]
  // ASP: [[INNER_RESULT:%[a-zA-Z0-9_]+]]:{{[0-9]+}} = scf.for {{.*}} iter_args({{.*}}[[INNER_SLOT:%.*]] = [[OUTER_SLOT]], [[INNER_BITS:%.*]] = [[OUTER_BITS]],
  // ASP: [[INNER_BIT:%.*]] = arith.shli {{%.*}}, [[INNER_SLOT]]
  // ASP: [[NEXT_READER_BITS:%.*]] = arith.xori [[INNER_BITS]], [[INNER_BIT]]
  // ASP: [[INNER_SHIFT:%.*]] = arith.shrui [[NEXT_READER_BITS]], [[INNER_SLOT]]
  // ASP: [[INNER_PHASE:%.*]] = arith.andi [[INNER_SHIFT]],
  // ASP: [[READ_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INNER_SLOT]], [[INNER_PHASE]]]
  // ASP: ttng.tmem_load
  // ASP: [[MMA_SLOT:%.*]] = arith.select
  // ASP: [[MMA_ASP:%.*]] = nvws.semaphore.acquire [[TO_MMA_ASP]][[[MMA_SLOT]],
  // ASP: ttng.tc_gen5_mma
  // ASP: nvws.semaphore.release [[READY_ASP]][[[MMA_SLOT]]], [[MMA_ASP]] [#nvws.async_op<tc5mma>]
  // ASP: scf.yield {{.*}}[[MMA_SLOT]], [[NEXT_READER_BITS]],
  // ASP: [[FINAL_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INNER_RESULT]]#1,
  // ASP: "consume"
  // Returning the token AND evolved slot/phase state re-arms the next outer trip.
  // ASP: scf.yield {{.*}}[[FINAL_ASP]], [[INNER_RESULT]]#1,
  // LOWER: [[READY_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64,
  // LOWER: ttng.init_barrier
  // LOWER: scf.for
  // LOWER: [[RELAY_VIEW:%.*]] = ttg.memdesc_index [[READY_STORAGE]][
  // LOWER: ttng.arrive_barrier [[RELAY_VIEW]], 1
  // LOWER-NEXT: {{.*}}scf.for
  // LOWER: [[FIRST_WAIT_VIEW:%.*]] = ttg.memdesc_index [[READY_STORAGE]][
  // LOWER: ttng.wait_barrier [[FIRST_WAIT_VIEW]],
  // LOWER: ttng.tmem_load
  tt.func @reader_first_depth_2(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %inner:2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %false, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @reader_first_depth_3
  // ASP-LABEL: @reader_first_depth_3
  // LOWER-LABEL: @reader_first_depth_3
  // SEMA: [[READY:%.*]] = nvws.semaphore.create {{.*}} released = {{[0-9]+}} {pending_count = 1 : i32}
  // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: scf.for {{.*}} iter_args([[OUTER_IN:%.*]] = [[ENTRY]])
  // SEMA: nvws.semaphore.release [[READY]], [[OUTER_IN]] [#nvws.async_op<none>]
  // SEMA-NEXT: {{.*}}scf.for
  // SEMA: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: ttng.tmem_load
  // SEMA: nvws.semaphore.release [[TO_MMA:%.*]], [[INNER_TOKEN]] [#nvws.async_op<none>]
  // SEMA: [[MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_MMA]]
  // SEMA: ttng.tc_gen5_mma
  // SEMA: nvws.semaphore.release [[READY]], [[MMA_TOKEN]] [#nvws.async_op<tc5mma>]
  // SEMA: [[RETURNED:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: "consume"
  // The post-inner acquisition is the next outer iteration's incoming token.
  // SEMA: scf.yield {{.*}}[[RETURNED]] : !ttg.async.token
  // ASP: [[INITIAL_SLOT:%.*]] = arith.constant 2 : i32
  // ASP: [[READY_ASP:%.*]] = nvws.semaphore.create {{.*}} released = 4 {pending_count = 1 : i32}
  // ASP: [[TO_MMA_ASP:%.*]] = nvws.semaphore.create
  // ASP: [[INITIAL_BITS:%.*]] = arith.constant -5 : i32
  // ASP: [[PHASE_ONE:%.*]] = arith.constant {{.*}} 1 : i32
  // The entry mask and selected bit produce phase 1 for initial availability.
  // ASP: [[INITIAL_BIT:%.*]] = arith.shli [[PHASE_ONE]], [[INITIAL_SLOT]]
  // ASP: [[ENTRY_BITS:%.*]] = arith.xori [[INITIAL_BITS]], [[INITIAL_BIT]]
  // ASP: [[INITIAL_SHIFT:%.*]] = arith.shrui [[ENTRY_BITS]], [[INITIAL_SLOT]]
  // ASP: [[ENTRY_PHASE:%.*]] = arith.andi [[INITIAL_SHIFT]], [[PHASE_ONE]]
  // ASP: [[ENTRY_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INITIAL_SLOT]], [[ENTRY_PHASE]]]
  // ASP: scf.for {{.*}} iter_args([[OUTER_TOKEN_ASP:%.*]] = [[ENTRY_ASP]], [[OUTER_SLOT:%.*]] = [[INITIAL_SLOT]], [[OUTER_BITS:%.*]] = [[ENTRY_BITS]],
  // The relay does not select a new slot or reset the carried phase word.
  // ASP: nvws.semaphore.release [[READY_ASP]][[[OUTER_SLOT]]], [[OUTER_TOKEN_ASP]] [#nvws.async_op<none>]
  // ASP: [[INNER_RESULT:%[a-zA-Z0-9_]+]]:{{[0-9]+}} = scf.for {{.*}} iter_args({{.*}}[[INNER_SLOT:%.*]] = [[OUTER_SLOT]], [[INNER_BITS:%.*]] = [[OUTER_BITS]],
  // ASP: [[INNER_BIT:%.*]] = arith.shli {{%.*}}, [[INNER_SLOT]]
  // ASP: [[NEXT_READER_BITS:%.*]] = arith.xori [[INNER_BITS]], [[INNER_BIT]]
  // ASP: [[INNER_SHIFT:%.*]] = arith.shrui [[NEXT_READER_BITS]], [[INNER_SLOT]]
  // ASP: [[INNER_PHASE:%.*]] = arith.andi [[INNER_SHIFT]],
  // ASP: [[READ_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INNER_SLOT]], [[INNER_PHASE]]]
  // ASP: ttng.tmem_load
  // ASP: [[MMA_SLOT:%.*]] = arith.select
  // ASP: [[MMA_ASP:%.*]] = nvws.semaphore.acquire [[TO_MMA_ASP]][[[MMA_SLOT]],
  // ASP: ttng.tc_gen5_mma
  // ASP: nvws.semaphore.release [[READY_ASP]][[[MMA_SLOT]]], [[MMA_ASP]] [#nvws.async_op<tc5mma>]
  // ASP: scf.yield {{.*}}[[MMA_SLOT]], [[NEXT_READER_BITS]],
  // ASP: [[FINAL_ASP:%.*]] = nvws.semaphore.acquire [[READY_ASP]][[[INNER_RESULT]]#1,
  // ASP: "consume"
  // Returning the token AND evolved slot/phase state re-arms the next outer trip.
  // ASP: scf.yield {{.*}}[[FINAL_ASP]], [[INNER_RESULT]]#1,
  // LOWER: [[READY_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
  // LOWER: ttng.init_barrier
  // LOWER: scf.for
  // LOWER: [[RELAY_VIEW:%.*]] = ttg.memdesc_index [[READY_STORAGE]][
  // LOWER: ttng.arrive_barrier [[RELAY_VIEW]], 1
  // LOWER-NEXT: {{.*}}scf.for
  // LOWER: [[FIRST_WAIT_VIEW:%.*]] = ttg.memdesc_index [[READY_STORAGE]][
  // LOWER: ttng.wait_barrier [[FIRST_WAIT_VIEW]],
  // LOWER: ttng.tmem_load
  tt.func @reader_first_depth_3(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 3 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %inner:2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %false, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @reader_first_all_mmas_fresh
  // ASP-LABEL: @reader_first_all_mmas_fresh
  // LOWER-LABEL: @reader_first_all_mmas_fresh
  // SEMA: [[READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // SEMA: [[ENTRY:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: scf.for {{.*}} iter_args([[OUTER_IN:%.*]] = [[ENTRY]])
  // SEMA: nvws.semaphore.release [[READY]], [[OUTER_IN]] [#nvws.async_op<none>]
  // SEMA-NEXT: {{.*}}scf.for
  // SEMA: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: ttng.tmem_load
  // SEMA: nvws.semaphore.release [[TO_MMA:%.*]], [[INNER_TOKEN]] [#nvws.async_op<none>]
  // SEMA: [[MMA_TOKEN:%.*]] = nvws.semaphore.acquire [[TO_MMA]]
  // SEMA: ttng.tc_gen5_mma
  // SEMA: nvws.semaphore.release [[READY]], [[MMA_TOKEN]] [#nvws.async_op<tc5mma>]
  // SEMA: [[RETURNED:%.*]] = nvws.semaphore.acquire [[READY]]
  // SEMA: "consume"
  // The post-inner acquisition is the next outer iteration's incoming token.
  // SEMA: scf.yield {{.*}}[[RETURNED]] : !ttg.async.token
  // ASP: nvws.semaphore.create
  // LOWER: ttng.init_barrier
  tt.func @reader_first_all_mmas_fresh(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %false, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @initialized_reader_first
  // ASP-LABEL: @initialized_reader_first
  // LOWER-LABEL: @initialized_reader_first
  // SEMA: [[INIT_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // SEMA: [[INIT_ENTRY:%.*]] = nvws.semaphore.acquire [[INIT_READY]]
  // SEMA: scf.for {{.*}} iter_args([[INIT_IN:%.*]] = [[INIT_ENTRY]])
  // SEMA: [[INIT_BUFFER:%.*]] = nvws.semaphore.buffer [[INIT_READY]], [[INIT_IN]]
  // SEMA: ttng.tmem_store {{.*}}, [[INIT_BUFFER]][]
  // SEMA-NEXT: nvws.semaphore.release [[INIT_READY]], [[INIT_IN]] [#nvws.async_op<none>]
  // The real initializer already supplies entry; do not add another arrival.
  // SEMA-NEXT: {{.*}}scf.for
  // SEMA: nvws.semaphore.release [[INIT_READY]]{{.*}}[#nvws.async_op<tc5mma>]
  // SEMA: [[INIT_RETURN:%.*]] = nvws.semaphore.acquire [[INIT_READY]]
  // SEMA: "consume"
  // SEMA: scf.yield {{.*}}[[INIT_RETURN]] : !ttg.async.token
  // ASP: nvws.semaphore.create
  // LOWER: ttng.init_barrier
  tt.func @initialized_reader_first(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %init = ttng.tmem_store %zero, %acc[%alloc_tok], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
      %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %init, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @initialized_zero_inner
  // ASP-LABEL: @initialized_zero_inner
  // LOWER-LABEL: @initialized_zero_inner
  // SEMA: [[INIT_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // SEMA: [[INIT_ENTRY:%.*]] = nvws.semaphore.acquire [[INIT_READY]]
  // SEMA: scf.for {{.*}} iter_args([[INIT_IN:%.*]] = [[INIT_ENTRY]])
  // SEMA: [[INIT_BUFFER:%.*]] = nvws.semaphore.buffer [[INIT_READY]], [[INIT_IN]]
  // SEMA: ttng.tmem_store {{.*}}, [[INIT_BUFFER]][]
  // SEMA-NEXT: nvws.semaphore.release [[INIT_READY]], [[INIT_IN]] [#nvws.async_op<none>]
  // The real initializer already supplies entry; do not add another arrival.
  // SEMA-NEXT: {{.*}}scf.for {{%.*}} = [[ZERO_INNER_BOUND:%.*]] to [[ZERO_INNER_BOUND]] step
  // SEMA: nvws.semaphore.release [[INIT_READY]]{{.*}}[#nvws.async_op<tc5mma>]
  // SEMA: [[INIT_RETURN:%.*]] = nvws.semaphore.acquire [[INIT_READY]]
  // SEMA: "consume"
  // SEMA: scf.yield {{.*}}[[INIT_RETURN]] : !ttg.async.token
  // ASP: nvws.semaphore.create
  // LOWER: ttng.init_barrier
  tt.func @initialized_zero_inner(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %init = ttng.tmem_store %zero, %acc[%alloc_tok], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
      %inner:2 = scf.for %j = %c0 to %c0 step %c1 iter_args(%carry = %init, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @initialized_zero_outer
  // ASP-LABEL: @initialized_zero_outer
  // LOWER-LABEL: @initialized_zero_outer
  // SEMA: [[ZERO_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // SEMA: [[ZERO_ENTRY:%.*]] = nvws.semaphore.acquire [[ZERO_READY]]
  // SEMA: ttng.tmem_store
  // The zero-trip result must forward the initialized incoming token.
  // SEMA: [[ZERO_OUTER:%[a-zA-Z0-9_]+]] = scf.for {{%.*}} = [[ZERO_BOUND:%.*]] to [[ZERO_BOUND]] step {{.*}} iter_args([[ZERO_INPUT:%.*]] = [[ZERO_ENTRY]])
  // SEMA: scf.yield
  // SEMA: [[ZERO_RESULT_BUFFER:%.*]] = nvws.semaphore.buffer [[ZERO_READY]], [[ZERO_OUTER]]
  // SEMA: ttng.tmem_load [[ZERO_RESULT_BUFFER]][]
  // SEMA: "consume_final"
  // ASP: nvws.semaphore.create
  // LOWER: ttng.init_barrier
  tt.func @initialized_zero_outer(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    %init = ttng.tmem_store %zero, %acc[%alloc_tok], %true {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : !tile -> !acc
    scf.for %i = %c0 to %c0 step %c1 : i32 {
      %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %init, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %final, %final_token = ttng.tmem_load %acc[%init] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : !acc -> !tile
    "consume_final"(%final) {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : (!tile) -> ()
    tt.return
  }

  // SEMA-LABEL: @write_first_control
  // ASP-LABEL: @write_first_control
  // LOWER-LABEL: @write_first_control
  // SEMA: ttng.tc_gen5_mma
  // ASP: nvws.semaphore.create
  // LOWER: ttng.init_barrier
  tt.func @write_first_control(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        %value, %read = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !acc -> !tile
        "consume_inner"(%value) {ttg.partition = array<i32: 0>} : (!tile) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read, %true : !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @write_first_fresh_depth_2
  // SEMA: [[FRESH_READY:%.*]] = nvws.semaphore.create {{.*}} released = 3 {pending_count = 1 : i32}
  // SEMA-NEXT: [[FRESH_DONE:%.*]] = nvws.semaphore.create
  // No enclosing reservation is needed for this write-first control.
  // SEMA-NEXT: {{.*}}scf.for
  // SEMA: [[FRESH_WRITER:%.*]] = nvws.semaphore.acquire [[FRESH_READY]]
  // SEMA: ttng.tc_gen5_mma
  // SEMA: nvws.semaphore.release [[FRESH_DONE]], [[FRESH_WRITER]] [#nvws.async_op<tc5mma>]
  // SEMA: [[FRESH_READER:%.*]] = nvws.semaphore.acquire [[FRESH_DONE]]
  // SEMA: ttng.tmem_load
  // SEMA: nvws.semaphore.release [[FRESH_READY]], [[FRESH_READER]] [#nvws.async_op<none>]
  // ASP-LABEL: @write_first_fresh_depth_2
  // ASP: nvws.semaphore.create
  // LOWER-LABEL: @write_first_fresh_depth_2
  // LOWER: ttng.init_barrier
  tt.func @write_first_fresh_depth_2(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    %inner:2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %false, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
      %value, %read = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume_inner"(%value) {ttg.partition = array<i32: 0>} : (!tile) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %read, %true : !ttg.async.token, i1
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @write_first_fresh_depth_3
  // SEMA: [[FRESH_READY:%.*]] = nvws.semaphore.create {{.*}} released = 7 {pending_count = 1 : i32}
  // SEMA-NEXT: [[FRESH_DONE:%.*]] = nvws.semaphore.create
  // No enclosing reservation is needed for this write-first control.
  // SEMA-NEXT: {{.*}}scf.for
  // SEMA: [[FRESH_WRITER:%.*]] = nvws.semaphore.acquire [[FRESH_READY]]
  // SEMA: ttng.tc_gen5_mma
  // SEMA: nvws.semaphore.release [[FRESH_DONE]], [[FRESH_WRITER]] [#nvws.async_op<tc5mma>]
  // SEMA: [[FRESH_READER:%.*]] = nvws.semaphore.acquire [[FRESH_DONE]]
  // SEMA: ttng.tmem_load
  // SEMA: nvws.semaphore.release [[FRESH_READY]], [[FRESH_READER]] [#nvws.async_op<none>]
  // ASP-LABEL: @write_first_fresh_depth_3
  // ASP: nvws.semaphore.create
  // LOWER-LABEL: @write_first_fresh_depth_3
  // LOWER: ttng.init_barrier
  tt.func @write_first_fresh_depth_3(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 3 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    %inner:2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %false, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
      %value, %read = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume_inner"(%value) {ttg.partition = array<i32: 0>} : (!tile) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %read, %true : !ttg.async.token, i1
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }


  // SEMA-LABEL: @reader_first_three_levels
  // SEMA: [[DEEP_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // SEMA: [[DEEP_ENTRY:%.*]] = nvws.semaphore.acquire [[DEEP_READY]]
  // SEMA: scf.for {{.*}} iter_args([[DEEP_OUTER_IN:%.*]] = [[DEEP_ENTRY]])
  // SEMA: [[MIDDLE:%.*]] = scf.for {{.*}} iter_args([[MIDDLE_IN:%.*]] = [[DEEP_OUTER_IN]])
  // SEMA: nvws.semaphore.release [[DEEP_READY]], [[MIDDLE_IN]] [#nvws.async_op<none>]
  // SEMA-NEXT: {{.*}}scf.for
  // SEMA: ttng.tc_gen5_mma
  // SEMA: nvws.semaphore.release [[DEEP_READY]]{{.*}}[#nvws.async_op<tc5mma>]
  // SEMA: [[MIDDLE_OUT:%.*]] = nvws.semaphore.acquire [[DEEP_READY]]
  // SEMA: "consume"
  // SEMA: scf.yield {{.*}}[[MIDDLE_OUT]] : !ttg.async.token
  // SEMA: scf.yield {{.*}}[[MIDDLE]] : !ttg.async.token
  // ASP-LABEL: @reader_first_three_levels
  // ASP: nvws.semaphore.create
  // LOWER-LABEL: @reader_first_three_levels
  // LOWER: [[NESTED_READY_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // LOWER: ttng.init_barrier
  // LOWER: scf.for
  // LOWER: scf.for
  // LOWER: [[NESTED_RELAY_VIEW:%.*]] = ttg.memdesc_index [[NESTED_READY_STORAGE]][
  // LOWER: ttng.arrive_barrier [[NESTED_RELAY_VIEW]], 1
  // LOWER-NEXT: {{.*}}scf.for
  // LOWER: [[NESTED_WAIT_VIEW:%.*]] = ttg.memdesc_index [[NESTED_READY_STORAGE]][
  // LOWER: ttng.wait_barrier [[NESTED_WAIT_VIEW]],
  // LOWER: ttng.tmem_load
  tt.func @reader_first_three_levels(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      scf.for %middle = %c0 to %c2 step %c1 : i32 {
        %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
          %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
          %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
          %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
          %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
          scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
        %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
        "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @reader_first_conditional
  // SEMA: [[IF_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // SEMA: [[IF_ENTRY:%.*]] = nvws.semaphore.acquire [[IF_READY]]
  // SEMA: scf.for {{.*}} iter_args([[IF_IN:%.*]] = [[IF_ENTRY]])
  // SEMA: [[BRANCH:%.*]] = scf.if
  // SEMA: [[IF_MIDDLE:%.*]] = scf.for {{.*}} iter_args([[IF_MIDDLE_IN:%.*]] = [[IF_IN]])
  // Supply belongs to the taken branch, never to the unchanged alternative.
  // SEMA: nvws.semaphore.release [[IF_READY]], [[IF_MIDDLE_IN]] [#nvws.async_op<none>]
  // SEMA-NEXT: {{.*}}scf.for
  // SEMA: ttng.tc_gen5_mma
  // SEMA: nvws.semaphore.release [[IF_READY]]{{.*}}[#nvws.async_op<tc5mma>]
  // SEMA: [[IF_OUT:%.*]] = nvws.semaphore.acquire [[IF_READY]]
  // SEMA: "consume"
  // SEMA: scf.yield {{.*}}[[IF_OUT]] : !ttg.async.token
  // SEMA: scf.yield {{.*}}[[IF_MIDDLE]] : !ttg.async.token
  // SEMA: } else {
  // SEMA-NOT: nvws.semaphore.release
  // SEMA: scf.yield {{.*}}[[IF_IN]] : !ttg.async.token
  // SEMA: scf.yield {{.*}}[[BRANCH]] : !ttg.async.token
  // ASP-LABEL: @reader_first_conditional
  // ASP: nvws.semaphore.create
  // LOWER-LABEL: @reader_first_conditional
  // LOWER: [[NESTED_READY_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // LOWER: ttng.init_barrier
  // LOWER: scf.for
  // LOWER: scf.if
  // LOWER: scf.for
  // LOWER: [[NESTED_RELAY_VIEW:%.*]] = ttg.memdesc_index [[NESTED_READY_STORAGE]][
  // LOWER: ttng.arrive_barrier [[NESTED_RELAY_VIEW]], 1
  // LOWER-NEXT: {{.*}}scf.for
  // LOWER: [[NESTED_WAIT_VIEW:%.*]] = ttg.memdesc_index [[NESTED_READY_STORAGE]][
  // LOWER: ttng.wait_barrier [[NESTED_WAIT_VIEW]],
  // LOWER: ttng.tmem_load
  // LOWER: } else {
  // LOWER-NOT: ttng.arrive_barrier
  // LOWER: scf.yield
  tt.func @reader_first_conditional(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      scf.if %guard {
        scf.for %middle = %c0 to %c2 step %c1 : i32 {
          %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
            %value, %read = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 0>} : !acc -> !tile
            %corrected = math.exp2 %value {ttg.partition = array<i32: 0>} : !tile
            %written = ttng.tmem_store %corrected, %acc[%read], %true {ttg.partition = array<i32: 0>} : !tile -> !acc
            %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
            scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
          } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
          %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
          "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
        } {ttg.partition = array<i32: 0, 1>}
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // SEMA-LABEL: @smem_reader_first
  // SEMA: [[SMEM_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 1 : i32}
  // SEMA: [[SMEM_ENTRY:%.*]] = nvws.semaphore.acquire [[SMEM_READY]]
  // SEMA: scf.for {{.*}} iter_args([[SMEM_IN:%.*]] = [[SMEM_ENTRY]])
  // SEMA: nvws.semaphore.release [[SMEM_READY]], [[SMEM_IN]] [#nvws.async_op<none>]
  // SEMA-NEXT: scf.for
  // SEMA: [[SMEM_READER:%.*]] = nvws.semaphore.acquire [[SMEM_READY]]
  // SEMA: ttg.local_load
  // SEMA: nvws.semaphore.release [[SMEM_WRITE:%.*]], [[SMEM_READER]] [#nvws.async_op<none>]
  // SEMA: [[SMEM_WRITER:%.*]] = nvws.semaphore.acquire [[SMEM_WRITE]]
  // SEMA: ttg.local_store
  // SEMA: nvws.semaphore.release [[SMEM_READY]], [[SMEM_WRITER]] [#nvws.async_op<none>]
  // SEMA: [[SMEM_OUT:%.*]] = nvws.semaphore.acquire [[SMEM_READY]]
  // SEMA: "consume_final"
  // SEMA: scf.yield {{.*}}[[SMEM_OUT]] : !ttg.async.token
  // ASP-LABEL: @smem_reader_first
  // ASP: nvws.semaphore.create
  // LOWER-LABEL: @smem_reader_first
  // LOWER: [[SMEM_READY_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // LOWER: ttng.init_barrier
  // LOWER: scf.for
  // LOWER: [[SMEM_RELAY_VIEW:%.*]] = ttg.memdesc_index [[SMEM_READY_STORAGE]][
  // LOWER: ttng.arrive_barrier [[SMEM_RELAY_VIEW]], 1
  // LOWER-NEXT: scf.for
  // LOWER: [[SMEM_WAIT_VIEW:%.*]] = ttg.memdesc_index [[SMEM_READY_STORAGE]][
  // LOWER: ttng.wait_barrier [[SMEM_WAIT_VIEW]],
  // LOWER: ttg.local_load
  tt.func @smem_reader_first(%value: tensor<1xi32, #reg1>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 101 : i32} : () -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      scf.for %j = %c0 to %c2 step %c1 : i32 {
        %before = ttg.local_load %buffer {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
        "observe_before_overwrite"(%before) {ttg.partition = array<i32: 0>} : (tensor<1xi32, #reg1>) -> ()
        ttg.local_store %value, %buffer {ttg.partition = array<i32: 1>} : tensor<1xi32, #reg1> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
      } {ttg.partition = array<i32: 0, 1>}
      %after = ttg.local_load %buffer {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
      "consume_final"(%after) {ttg.partition = array<i32: 0>} : (tensor<1xi32, #reg1>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  // A real incoming token can also require the new deferred handoff. The
  // initial store is outside both loops; each inner recurrence must receive
  // two arrivals, matching its two independent completion sources.
  // SEMA-LABEL: @smem_nested_fanin
  // SEMA: [[FANIN_READY:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 2 : i32}
  // SEMA: [[FANIN_ENTRY:%.*]] = nvws.semaphore.acquire [[FANIN_READY]]
  // SEMA: ttg.local_store
  // SEMA: scf.for {{.*}} iter_args([[FANIN_IN:%.*]] = [[FANIN_ENTRY]])
  // SEMA: nvws.semaphore.release [[FANIN_READY]], [[FANIN_IN]] [#nvws.async_op<none>] {arrive_count = 2 : i32, ttg.partition = array<i32: 2>}
  // SEMA-NEXT: scf.for
  // SEMA: nvws.semaphore.acquire [[FANIN_READY]]
  // SEMA: ttg.local_store
  // SEMA: nvws.semaphore.release [[FANIN_READY]]{{.*}}[#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
  // SEMA: ttg.local_load
  // SEMA: nvws.semaphore.release [[FANIN_READY]]{{.*}}[#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
  // SEMA: [[FANIN_OUT:%.*]] = nvws.semaphore.acquire [[FANIN_READY]]
  // SEMA: "consume_final"
  // SEMA: scf.yield {{.*}}[[FANIN_OUT]] : !ttg.async.token
  // ASP-LABEL: @smem_nested_fanin
  // ASP: [[FANIN_ASP:%.*]] = nvws.semaphore.create {{.*}} released = 1 {pending_count = 2 : i32}
  // ASP: scf.for
  // ASP: nvws.semaphore.release [[FANIN_ASP]]{{.*}}[#nvws.async_op<none>] {arrive_count = 2 : i32,
  // ASP-NEXT: {{.*}}scf.for
  // LOWER-LABEL: @smem_nested_fanin
  // LOWER: [[FANIN_STORAGE:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // LOWER: ttng.init_barrier {{.*}}, 2
  // LOWER: scf.for
  // LOWER: [[FANIN_RELAY:%.*]] = ttg.memdesc_index [[FANIN_STORAGE]][
  // LOWER: ttng.arrive_barrier [[FANIN_RELAY]], 2
  // LOWER-NEXT: scf.for
  // LOWER: ttng.wait_barrier
  // LOWER: ttg.local_load
  tt.func @smem_nested_fanin(%value: tensor<1xi32, #reg1>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 102 : i32} : () -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
    ttg.local_store %value, %buffer {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : tensor<1xi32, #reg1> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      scf.for %j = %c0 to %c2 step %c1 : i32 {
        %first = ttg.local_load %buffer {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
        "consume_first"(%first) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #reg1>) -> ()
        %second = ttg.local_load %buffer {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
        %corrected = arith.addi %second, %second {ttg.partition = array<i32: 1>} : tensor<1xi32, #reg1>
        ttg.local_store %corrected, %buffer {ttg.partition = array<i32: 1>} : tensor<1xi32, #reg1> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
        %last = ttg.local_load %buffer {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
        "consume_last"(%last) {ttg.partition = array<i32: 0>} : (tensor<1xi32, #reg1>) -> ()
      } {ttg.partition = array<i32: 0, 1, 2>}
      %after = ttg.local_load %buffer {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #reg1>
      "consume_final"(%after) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #reg1>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

//--- pipeline.mlir
#reg = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
#tm = #ttng.tensor_memory
!acc = !ttg.memdesc<128x128xf32, #tmem, #tm, mutable>
!tile = tensor<128x128xf32, #reg>
!lhs = !ttg.memdesc<128x64xf16, #shared, #smem>
!rhs = !ttg.memdesc<64x128xf16, #shared, #smem>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // PIPE-LABEL: @scheduled_reader_first
  // PIPE: [[PIPE_READY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // PIPE: ttg.warp_specialize
  // PIPE: default {
  // PIPE: ttng.wait_barrier
  // PIPE: scf.for
  // PIPE: [[PIPE_RELAY:%.*]] = ttg.memdesc_index [[PIPE_READY]][
  // PIPE: ttng.arrive_barrier [[PIPE_RELAY]], 1
  // PIPE-NEXT: {{.*}}scf.for
  // PIPE: [[PIPE_WAIT:%.*]] = ttg.memdesc_index [[PIPE_READY]][
  // PIPE: ttng.wait_barrier [[PIPE_WAIT]],
  // PIPE: ttng.tmem_load
  // PIPE: ttng.arrive_barrier
  tt.func @scheduled_reader_first(%lhs: !lhs, %rhs: !rhs, %guard: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : !tile
    %acc, %alloc_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 100 : i32} : () -> (!acc, !ttg.async.token)
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%carry = %alloc_tok, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
        %value, %read = ttng.tmem_load %acc[%carry] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !acc -> !tile
        %corrected = math.exp2 %value {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !tile
        %written = ttng.tmem_store %corrected, %acc[%read], %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !tile -> !acc
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%written], %use_acc, %true {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !lhs, !rhs, !acc
        scf.yield {ttg.partition = array<i32: 0, 1>} %mma, %true : !ttg.async.token, i1
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      %out, %read_out = ttng.tmem_load %acc[%inner#0] {ttg.partition = array<i32: 0>} : !acc -> !tile
      "consume"(%out) {ttg.partition = array<i32: 0>} : (!tile) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
