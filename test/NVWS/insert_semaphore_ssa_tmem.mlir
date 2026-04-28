// RUN: env NVWS_USE_SSA_TMEM=1 triton-opt %s --nvws-insert-semaphore -allow-unregistered-dialect | FileCheck %s
// RUN: env NVWS_USE_SSA_TMEM=1 triton-opt %s --nvws-insert-semaphore --nvws-insert-tmem-semaphore -allow-unregistered-dialect | FileCheck %s

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#slice = #ttg.slice<{dim = 1, parent = #linear}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @ssa_rank1_fp_uses_tmem
  // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x1xf32, {{.*}}#ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @ssa_rank1_fp_uses_tmem(%lb: i32, %ub: i32, %step: i32, %input: tensor<128xf32, #slice>) {
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[SRC:%.*]] = "make_alpha"
      %alpha = "make_alpha"(%input) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128xf32, #slice>) -> tensor<128xf32, #slice>

      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[EXP:%.*]] = tt.expand_dims [[SRC]] {axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[STORE_SRC:%.*]] = ttg.convert_layout [[EXP]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[PRED:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} true
      // CHECK-NEXT: ttng.tmem_store [[STORE_SRC]], [[PBUF]], [[PRED]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NOT: ttg.local_store
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[CBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[LOAD:%[A-Za-z0-9_]+]], {{%[A-Za-z0-9_]+}} = ttng.tmem_load [[CBUF]][] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[RESHAPE:%.*]] = tt.reshape [[LOAD]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[RELOADED:%.*]] = ttg.convert_layout [[RESHAPE]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NOT: ttg.local_load
      // CHECK: "use"([[RELOADED]]) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      "use"(%alpha) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128xf32, #slice>) -> ()
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
