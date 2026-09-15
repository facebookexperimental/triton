// RUN: triton-opt --tlx-print-ttgir-to-tlx -split-input-file %s | FileCheck %s

// Test that the predicates the software pipeliner puts on its prefetched ops
// round-trip, and that a constant-true one is left implicit.
//
// The pipeliner predicates the prefetched wait and dot with `k < k_tiles - 1`
// so the drain iteration neither waits on a buffer that is never filled nor
// runs a dot on unloaded operands. A constant-true predicate is the default
// and carries no information, so only a real one is emitted.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // A real predicate guards whether the wait executes, so it must survive.
  // CHECK-LABEL: def wait_with_predicate(
  // CHECK: [[PRED:[a-zA-Z_0-9]+]] = {{.*}} < {{.*}}
  // CHECK: tlx.barrier_wait({{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}, pred=[[PRED]])
  tt.func public @wait_with_predicate(%k: i32, %k_tiles: i32, %phase: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %last = arith.subi %k_tiles, %c1_i32 : i32
    %pred = arith.cmpi slt, %k, %last : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A constant-true predicate is the default, so it is left implicit.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def wait_with_constant_true(
  // CHECK: tlx.barrier_wait(
  // CHECK-NOT: pred=
  tt.func public @wait_with_constant_true(%phase: i32) attributes {noinline = false} {
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// The same holds for the dot: dropping its predicate would let the drain
// iteration accumulate garbage into the tmem accumulator.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def dot_with_predicate(
  // CHECK: [[PRED:[a-zA-Z_0-9]+]] = {{.*}} < {{.*}}
  // CHECK: tlx.async_dot({{.*}}pred=[[PRED]]{{.*}})
  tt.func public @dot_with_predicate(%k: i32, %k_tiles: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %true = arith.constant true
    %last = arith.subi %k_tiles, %c1_i32 : i32
    %pred = arith.cmpi slt, %k, %last : i32
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.tc_gen5_mma %a, %b, %acc, %false, %pred, %bar[%true] {is_async} : !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A constant-true dot predicate is likewise left implicit. This is the half of
// the dot change that is observable: a real predicate was already emitted
// before, a constant-true one was emitted too and should not be.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def dot_with_constant_true(
  // CHECK: tlx.async_dot(
  // CHECK-NOT: pred=
  tt.func public @dot_with_constant_true() attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.tc_gen5_mma %a, %b, %acc, %false, %true, %bar[%true] {is_async} : !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// The predicate is found by skipping the variadic memdesc dependency buffers,
// which are a scheduling hint rather than something the wait's behavior
// depends on, so they are not emitted.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def wait_with_predicate_and_deps(
  // CHECK: [[PRED:[a-zA-Z_0-9]+]] = {{.*}} < {{.*}}
  // CHECK: tlx.barrier_wait({{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}, pred=[[PRED]])
  tt.func public @wait_with_predicate_and_deps(%k: i32, %k_tiles: i32, %phase: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %last = arith.subi %k_tiles, %c1_i32 : i32
    %pred = arith.cmpi slt, %k, %last : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase, %pred deps %a : !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

// Deps with no predicate: nothing is emitted for either.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def wait_with_deps_only(
  // CHECK: tlx.barrier_wait(
  // CHECK-NOT: pred=
  tt.func public @wait_with_deps_only(%phase: i32) attributes {noinline = false} {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase deps %a : !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
    tt.return
  }
}
