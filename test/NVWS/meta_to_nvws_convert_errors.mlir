// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-meta-to-nvws-convert -verify-diagnostics

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @missing_meta_assignment(%lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      // expected-error @below {{MetaToNVWSConvert requires a non-empty, non-negative async_task_id or ttg.partition assignment}}
      "test.missing"() : () -> ()
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @reject_mismatched_promoted_root(
      %lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{MetaToNVWSConvert found an enclosing loop with a mismatched async_task_id task domain}}
    scf.for %outer = %lb to %ub step %step : i32 {
      scf.for %inner = %lb to %ub step %step : i32 {
        "test.inner"() {async_task_id = array<i32: 1>} : () -> ()
        scf.yield {async_task_id = array<i32: 0, 1>}
      } {async_task_id = array<i32: 0, 1>, tt.warp_specialize,
         ttg.partition.stages = [0 : i32, 1 : i32],
         ttg.partition.types = ["default", "gemm"],
         ttg.warp_specialize.tag = 12 : i32}
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @reject_uses_in_two_cfg_blocks(
      %early: i1, %lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{MetaToNVWSConvert: managed memdesc flow across function CFG blocks is unsupported}}
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 40 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      "test.touch"(%alloc) {async_task_id = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"], ttg.warp_specialize.tag = 0 : i32}
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    scf.for %i = %lb to %ub step %step : i32 {
      "test.touch"(%alloc) {async_task_id = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @reject_group_definitions_in_two_cfg_blocks(
      %early: i1, %lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{MetaToNVWSConvert: one buffer group spans function CFG blocks}}
    %a = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 41 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    %b = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 41 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      "test.touch"(%a, %b) {async_task_id = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, !ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @reject_tmem_results_in_two_cfg_blocks(
      %early: i1, %lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{MetaToNVWSConvert: managed memdesc flow across function CFG blocks is unsupported}}
    %alloc, %token = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %i = %lb to %ub step %step : i32 {
      "test.touch_memdesc"(%alloc) {async_task_id = array<i32: 0>} : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"], ttg.warp_specialize.tag = 0 : i32}
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    scf.for %i = %lb to %ub step %step : i32 {
      "test.touch_token"(%token) {async_task_id = array<i32: 0>} : (!ttg.async.token) -> ()
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @reject_alias_chain_across_cfg_blocks(
      %early: i1, %lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{MetaToNVWSConvert: managed memdesc flow across function CFG blocks is unsupported}}
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 43 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %view0 = ttg.memdesc_reinterpret %alloc {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %view1 = ttg.memdesc_reinterpret %view0 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    scf.for %i = %lb to %ub step %step : i32 {
      "test.touch"(%view1) {async_task_id = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @reject_cfg_block_argument(
      %lb: i32, %ub: i32, %step: i32) {
    // expected-error @below {{MetaToNVWSConvert: managed memdesc flow through function CFG block arguments is unsupported}}
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 44 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    cf.br ^work(%alloc : !ttg.memdesc<1xi32, #shared, #smem, mutable>)
  ^work(%buffer: !ttg.memdesc<1xi32, #shared, #smem, mutable>):
    scf.for %i = %lb to %ub step %step : i32 {
      "test.touch"(%buffer) {async_task_id = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
