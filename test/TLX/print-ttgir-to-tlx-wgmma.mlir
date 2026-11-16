// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Test that Hopper wgmma (ttng.warp_group_dot and its wait) round-trips to
// tlx.async_dot / tlx.async_dot_wait. Every tl.dot compiled for sm_90 lowers
// to these, so without them no Hopper matmul or attention kernel round-trips.

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

  // A pipelined dot waits on its accumulator and on the shared-memory operands
  // it is still reading, all in one wait, and keeps its outstanding-group count.
  // CHECK-LABEL: def async_dot_and_wait(
  // CHECK: [[D:[a-zA-Z_0-9]+]] = tlx.async_dot([[A:[a-zA-Z_0-9]+]], [[B:[a-zA-Z_0-9]+]], {{[a-zA-Z_0-9]+}})
  // CHECK-NOT: input_precision
  // CHECK: ({{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}) = tlx.async_dot_wait(1, ([[D]], [[A]], [[B]]))
  tt.func public @async_dot_and_wait(%a: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %d = ttng.warp_group_dot %a, %b, %cst {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
    %w:3 = ttng.warp_group_dot_wait %d, %a, %b {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    tt.return
  }

  // A single-input wait is not a tuple.
  // CHECK-LABEL: def single_input_wait(
  // CHECK: {{[a-zA-Z_0-9]+}} = tlx.async_dot_wait(0, {{[a-zA-Z_0-9]+}})
  tt.func public @single_input_wait(%acc: tensor<128x128xf32, #mma>) {
    %w = ttng.warp_group_dot_wait %acc {pendings = 0 : i32} : tensor<128x128xf32, #mma>
    tt.return
  }

  // A register A operand (attention's P @ V) is passed through as a tensor.
  // A synchronous dot has no TLX spelling of its own; it is an async dot
  // drained immediately.
  // CHECK-LABEL: def sync_dot_register_a(
  // CHECK: {{[a-zA-Z_0-9]+}} = tlx.async_dot_wait(0, tlx.async_dot({{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}))
  tt.func public @sync_dot_register_a(%p: tensor<128x128xf16, #dotA>, %v: !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %d = ttng.warp_group_dot %p, %v, %cst {inputPrecision = 0 : i32} : tensor<128x128xf16, #dotA> * !ttg.memdesc<128x128xf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
    tt.return
  }

  // For fp32 operands the precision changes the instruction (tf32 vs. 3xtf32),
  // so it is carried through.
  // CHECK-LABEL: def fp32_precision(
  // CHECK: tlx.async_dot({{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}, input_precision="tf32x3")
  tt.func public @fp32_precision(%a: !ttg.memdesc<128x32xf32, #shared32, #smem, mutable>, %b: !ttg.memdesc<32x128xf32, #shared32, #smem, mutable>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %d = ttng.warp_group_dot %a, %b, %cst {inputPrecision = 1 : i32, isAsync = true} : !ttg.memdesc<128x32xf32, #shared32, #smem, mutable> * !ttg.memdesc<32x128xf32, #shared32, #smem, mutable> -> tensor<128x128xf32, #mma>
    tt.return
  }
}
