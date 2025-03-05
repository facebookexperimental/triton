// RUN: triton-opt %s -split-input-file --fix-ws-barrier | FileCheck %s

// CHECK-LABEL: @no_rewrite_barrier
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warp-groups-per-cta" = 2 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 171056 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  llvm.func @no_rewrite_barrier(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 256>} {
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    nvvm.barrier0
    nvvm.barrier0
    llvm.br ^bb1
  ^bb1:
    llvm.return
  }
}

// -----

// CHECK-LABEL: @mixed_control_flow
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warp-groups-per-cta" = 2 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 171056 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  llvm.func @mixed_control_flow(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 256>} {
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    // CHECK: bar.sync 0x1, 0x80;
    // CHECK: bar.sync 0x2, 0x80;
    nvvm.barrier0
    nvvm.barrier0
    %1 = llvm.mlir.constant(1 : i1) : i1
    llvm.br ^bb1
  ^bb1:
    nvvm.barrier0
    llvm.cond_br %1, ^bb2, ^bb3
  ^bb2:
    nvvm.barrier0
    llvm.br ^bb3
  ^bb3:
    llvm.return
  }
}

// -----

// CHECK-LABEL: @no_warp_specialize
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 171056 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  llvm.func @no_warp_specialize(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 256>} {
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    nvvm.barrier0
    nvvm.barrier0
    %1 = llvm.mlir.constant(1 : i1) : i1
    llvm.br ^bb1
  ^bb1:
    nvvm.barrier0
    llvm.cond_br %1, ^bb2, ^bb3
  ^bb2:
    nvvm.barrier0
    llvm.br ^bb3
  ^bb3:
    llvm.return
  }
}

// -----

// CHECK-LABEL: @reuse_barrier_id
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warp-groups-per-cta" = 2 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 171056 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  llvm.func @reuse_barrier_id(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 256>} {
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    // CHECK: bar.sync 0x2, 0x80;
    // CHECK: bar.sync 0x2, 0x80;
    // CHECK: bar.sync 0x3, 0x80;
    // CHECK: bar.sync 0x4, 0x80;
    // CHECK: bar.sync 0x1, 0x80;
    nvvm.barrier0
    nvvm.barrier0
    %1 = llvm.mlir.constant(1 : i1) : i1
    llvm.br ^bb1
  ^bb1:
    nvvm.barrier0
    %2 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "bar.sync 0x2, 0x80;", ""  : () -> !llvm.void
    %3 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "bar.sync 0x3, 0x80;", ""  : () -> !llvm.void
    %4 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "bar.sync 0x4, 0x80;", ""  : () -> !llvm.void
    llvm.cond_br %1, ^bb2, ^bb3
  ^bb2:
    nvvm.barrier0
    llvm.br ^bb3
  ^bb3:
    llvm.return
  }
}
