// RUN: triton-opt %s -split-input-file --fix-ws-barrier | FileCheck %s

// CHECK-LABEL: @no_rewrite_barrier
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warp-groups-per-cta" = 2 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 171056 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  llvm.func @no_rewrite_barrier(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 256>} {
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    nvvm.barrier0
    nvvm.barrier0
    llvm.br ^bb1
  ^bb1:
    nvvm.barrier0
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
    // CHECK: nvvm.barrier0
    nvvm.barrier0
    nvvm.barrier0
    %cst4 = llvm.mlir.constant(4 : i32) : i32
    %cst0 = llvm.mlir.constant(0 : i32) : i32
    %warpid = llvm.inline_asm asm_dialect = att operand_attrs = [] "{\0A.reg .u32 a<5>;              \0Amov.u32 a0, %tid.x;          \0Amov.u32 a1, %tid.y;          \0Amov.u32 a2, %tid.z;          \0Amov.u32 a3, %ntid.x;         \0Amov.u32 a4, %ntid.y;         \0Amad.lo.u32 a1, a2, a4, a1;   \0Amad.lo.u32 a0, a1, a3, a0;   \0Ashr.u32 a0, a0, 5;           \0A.reg .b32         %tmp<3>;   \0Amov.u32   %tmp0, -1;         \0Amov.u32   %tmp1, 31;         \0Amov.u32   %tmp2, 0;          \0Ashfl.sync.idx.b32         $0, a0, %tmp2, %tmp1, %tmp0;           \0A}", "=r"  : () -> i32
    %wgid = llvm.udiv %warpid, %cst4  : i32
    %flag = llvm.icmp "eq" %wgid, %cst0 : i32
    llvm.cond_br %flag, ^bb1, ^bb2
  ^bb1:
    nvvm.barrier0
    llvm.br ^bb3
  ^bb2:
    nvvm.barrier0
    llvm.br ^bb3
  ^bb3:
    nvvm.barrier0
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

// CHECK-LABEL: @no_warp_specialize_2
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warp-groups-per-cta" = 2 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 171056 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  llvm.func @no_warp_specialize_2(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 256>} {
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
    nvvm.barrier0
    nvvm.barrier0
    %flag = llvm.mlir.constant(1 : i1) : i1
    llvm.cond_br %flag, ^bb1, ^bb2
  ^bb1:
    nvvm.barrier0
    llvm.br ^bb3
  ^bb2:
    nvvm.barrier0
    llvm.br ^bb3
  ^bb3:
    nvvm.barrier0
    llvm.return
  }
}

// -----

// CHECK-LABEL: @reuse_barrier_id
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warp-groups-per-cta" = 2 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 171056 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  llvm.func @reuse_barrier_id(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 256>} {
    // CHECK: nvvm.barrier0
    // CHECK: nvvm.barrier0
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
    llvm.cond_br %1, ^bb1, ^bb2
  ^bb1:
    nvvm.barrier0
    llvm.br ^bb2
  ^bb2:
    %cst4 = llvm.mlir.constant(4 : i32) : i32
    %cst0 = llvm.mlir.constant(0 : i32) : i32
    %warpid = llvm.inline_asm asm_dialect = att operand_attrs = [] "{\0A.reg .u32 a<5>;              \0Amov.u32 a0, %tid.x;          \0Amov.u32 a1, %tid.y;          \0Amov.u32 a2, %tid.z;          \0Amov.u32 a3, %ntid.x;         \0Amov.u32 a4, %ntid.y;         \0Amad.lo.u32 a1, a2, a4, a1;   \0Amad.lo.u32 a0, a1, a3, a0;   \0Ashr.u32 a0, a0, 5;           \0A.reg .b32         %tmp<3>;   \0Amov.u32   %tmp0, -1;         \0Amov.u32   %tmp1, 31;         \0Amov.u32   %tmp2, 0;          \0Ashfl.sync.idx.b32         $0, a0, %tmp2, %tmp1, %tmp0;           \0A}", "=r"  : () -> i32
    %wgid = llvm.udiv %warpid, %cst4  : i32
    nvvm.barrier0
    %flag = llvm.icmp "eq" %wgid, %cst0 : i32
    llvm.cond_br %flag, ^bb3, ^bb4
  ^bb3:
    nvvm.barrier0
    %2 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "bar.sync 0x2, 0x80;", ""  : () -> !llvm.void
    %3 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "bar.sync 0x3, 0x80;", ""  : () -> !llvm.void
    %4 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "bar.sync 0x4, 0x80;", ""  : () -> !llvm.void
    llvm.br ^bb5
  ^bb4:
    nvvm.barrier0
    llvm.br ^bb5
  ^bb5:
    llvm.return
  }
}
