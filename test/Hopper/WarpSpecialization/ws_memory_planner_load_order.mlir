// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=2 smem-alloc-algo=1 smem-budget=196608" | FileCheck %s

// Test: Phase 4 chooses equal-priority multi-buffer candidates in producer
// usage order, not local_alloc order. W is allocated first, but X's TMA load is
// issued first. With only enough SMEM for one extra 128x256xf16 copy, X should
// get buffer.copy = 2 and W should stay at buffer.copy = 1.

// CHECK-LABEL: @load_order_breaks_multibuffer_tie
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32}
// CHECK-SAME: !ttg.memdesc<128x256xf16
// CHECK: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = {{[0-9]+}} : i32}
// CHECK-SAME: !ttg.memdesc<128x256xf16

#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @load_order_breaks_multibuffer_tie(
      %x_desc: !tt.tensordesc<tensor<128x256xf16, #shared>>,
      %w_desc: !tt.tensordesc<tensor<128x256xf16, #shared>>) {
    %W_smem = ttg.local_alloc : () -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    %X_smem = ttg.local_alloc : () -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    %c0 = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
    %c1 = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
    %c10 = arith.constant {ttg.partition = array<i32: 0, 1>} 10 : i32
    scf.for %iv = %c0 to %c10 step %c1 : i32 {
      %x = tt.descriptor_load %x_desc[%c0, %c0] {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x256xf16, #shared>> -> tensor<128x256xf16, #blocked1>
      ttg.local_store %x, %X_smem {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x256xf16, #blocked1> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
      %w = tt.descriptor_load %w_desc[%c0, %c0] {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x256xf16, #shared>> -> tensor<128x256xf16, #blocked1>
      ttg.local_store %w, %W_smem {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x256xf16, #blocked1> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
      %x_val = ttg.local_load %X_smem {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked1>
      %w_val = ttg.local_load %W_smem {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked1>
      %sum = arith.addf %x_val, %w_val {ttg.partition = array<i32: 0>} : tensor<128x256xf16, #blocked1>
      scf.yield {ttg.partition = array<i32: 0, 1>}
    } {ttg.partition = array<i32: 0, 1>, tt.warp_specialize}
    tt.return
  }
}
