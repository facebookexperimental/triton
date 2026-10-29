// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -convert-warp-specialize-to-llvm -verify-diagnostics
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -triton-amdgpu-convert-warp-specialize-to-llvm=gfx-arch=gfx1250 -verify-diagnostics

// Single-WS dispatch runs each worker warp in one partition and then exits, so
// a second warp_specialize region (e.g. one synthesized to make worker warps
// join a cluster sync) must be rejected rather than miscompiled.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, "ttg.single-warp-specialize" = true} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// expected-error @below {{single-warp-specialize lowering requires exactly one warp_specialize op, but found 2}}
llvm.func @single_ws_two_regions() attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    "default0"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "partition0"() : () -> ()
    ttg.warp_return
  } : () -> ()
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    "default1"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "partition1"() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}
