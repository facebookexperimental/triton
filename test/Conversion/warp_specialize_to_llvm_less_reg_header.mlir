// RUN: triton-opt %s -split-input-file -mlir-print-local-scope -allow-unregistered-dialect -convert-warp-specialize-to-llvm -cse | FileCheck %s

// Regression test for `tlx.less_reg_header`, a hacky tuning knob propagated to
// the module by the TLX Fixup pass from `tlx.async_tasks(less_reg_header=True)`.
//
// By default the warp-specialize header computes the dispatch warp ID from a
// plain %tid.x read. That read dominates every partition, so CSE (this file's
// RUN line runs -cse) folds the partitions' own %tid.x reads into it, and
// %tid.x is computed in the header and kept live all the way into the
// partitions. For unknown reasons, ptxas may then spill it to local memory.
// With the knob set, the header reads %tid.x inside one opaque inline-PTX
// block, so each partition keeps its own %tid.x read.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, tlx.less_reg_header} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @less_reg_header
llvm.func @less_reg_header() attributes {allocation.offset = 32 : i32} {
  // The header has no plain %tid.x read; the warp ID comes from inline PTX.
  // CHECK-NOT: nvvm.read.ptx.sreg.tid.x
  // CHECK: [[WID:%.*]] = llvm.inline_asm {{.*}}"{\0A mov.u32 $0, %tid.x;\0A shr.u32 $0, $0, 5;\0A}", "=r"
  // CHECK: [[WARP_ID:%.*]] = nvvm.shfl.sync idx %{{.*}}, [[WID]],
  // CHECK: llvm.icmp "ult" [[WARP_ID]]

  // Both partitions keep their own %tid.x read.
  // CHECK: [[TID0:%.*]] = nvvm.read.ptx.sreg.tid.x
  // CHECK-NEXT: "partition0"([[TID0]])
  // CHECK: [[TID1:%.*]] = nvvm.read.ptx.sreg.tid.x
  // CHECK-NEXT: "default"([[TID1]])
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    "default"(%tid) : (i32) -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    "partition0"(%tid) : (i32) -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

// Same kernel without `tlx.less_reg_header`: the header reads %tid.x with a
// plain read, and CSE folds both partitions' reads into it.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @default_header
llvm.func @default_header() attributes {allocation.offset = 32 : i32} {
  // CHECK: [[TID:%.*]] = nvvm.read.ptx.sreg.tid.x
  // CHECK: llvm.udiv [[TID]]
  // CHECK-NOT: llvm.inline_asm
  // CHECK-NOT: nvvm.read.ptx.sreg.tid.x
  // CHECK: "partition0"([[TID]])
  // CHECK-NOT: nvvm.read.ptx.sreg.tid.x
  // CHECK: "default"([[TID]])
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    "default"(%tid) : (i32) -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    "partition0"(%tid) : (i32) -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}
