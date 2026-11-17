// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-partition-scheduling-meta '--nvgpu-warp-specialization=capability=90 num-stages=3 smem-budget=200000' | FileCheck %s

// A pre-loop atomic whose result feeds the warp-specialized loop runs once,
// ahead of ttg.warp_specialize, and reaches the partitions as a capture. It
// used to be tagged with every partition, which made WS bail.

// CHECK-LABEL: @pre_loop_atomic
// CHECK: %[[TOK:.*]] = tt.atomic_rmw add, acq_rel, gpu
// CHECK: ttg.warp_specialize(%[[TOK]],
// CHECK: partition0(%[[ARG:[a-z0-9]+]]: i32
// CHECK: arith.addi %{{.*}}, %[[ARG]]
// CHECK-NOT: tt.atomic_rmw
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @pre_loop_atomic(
      %a_desc: !tt.tensordesc<128x64xf16, #shared>,
      %b_desc: !tt.tensordesc<64x256xf16, #shared>,
      %out: !tt.ptr<f32>, %flag: !tt.ptr<i32>, %k_tiles: i32, %num_tiles: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cm1 = arith.constant -1 : i32
    %c148 = arith.constant 148 : i32
    %true = arith.constant true
    %acc_init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %pid = tt.get_program_id x : i32
    %slot = tt.addptr %flag, %pid : !tt.ptr<i32>, i32
    %tok = tt.atomic_rmw add, acq_rel, gpu, %slot, %c1, %true : (!tt.ptr<i32>, i32, i1) -> i32
    %start = arith.addi %pid, %tok : i32
    %result = scf.while (%tile = %start) : (i32) -> i32 {
      %valid = arith.cmpi slt, %tile, %num_tiles : i32
      scf.condition(%valid) %tile : i32
    } do {
    ^bb0(%tile: i32):
      %inner = scf.for %ki = %c0 to %k_tiles step %c1 iter_args(%acc = %acc_init) -> (tensor<128x256xf32, #mma>) : i32 {
        %a = tt.descriptor_load %a_desc[%tile, %ki] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %a_alloc = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%ki, %tile] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
        %b_alloc = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %dot = ttng.warp_group_dot %a_alloc, %b_alloc, %acc {inputPrecision = 0 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        scf.yield %dot : tensor<128x256xf32, #mma>
      } {tt.scheduled_max_stage = 2 : i32}
      %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
      %cvt = ttg.convert_layout %inner : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
      tt.store %ptrs, %cvt : tensor<128x256x!tt.ptr<f32>, #blocked1>
      %next = arith.addi %tile, %c148 : i32
      scf.yield %next : i32
    } attributes {tt.warp_specialize}
    tt.return
  }
}
