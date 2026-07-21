// RUN: triton-opt %s --triton-nvidia-plan-tma-multicast | FileCheck %s
// RUN: triton-opt %s --triton-nvidia-plan-tma-multicast --triton-nvidia-tma-lowering | FileCheck %s --check-prefix=LOWERING

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {
  "ttg.ctas-per-cga" = true,
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 4 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.multicast" = true,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  ttg.target = "cuda:90",
  "ttg.threads-per-warp" = 32 : i32
} {
  // CHECK-LABEL: @rectangular
  tt.func public @rectangular(
      %a: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %b: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    %k = arith.constant 0 : i32
    // CHECK: tt.descriptor_load {{.*}} {tt.multicast_axes = array<i32: 1>}
    %av = tt.descriptor_load %a[%x, %k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    // CHECK: tt.descriptor_load {{.*}} {tt.multicast_axes = array<i32: 0>}
    %bv = tt.descriptor_load %b[%y, %k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @no_reuse
  tt.func public @no_reuse(%a: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    // CHECK-NOT: tt.multicast_axes
    %v = tt.descriptor_load %a[%x, %y] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @per_load_disable
  tt.func public @per_load_disable(%a: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %x = tt.get_program_id x : i32
    %k = arith.constant 0 : i32
    // CHECK-NOT: tt.multicast_axes
    %v = tt.descriptor_load %a[%x, %k] {tt.multicast = false} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @dynamic_tile
  tt.func public @dynamic_tile(
      %a: !tt.tensordesc<tensor<128x64xf16, #shared>>, %counter: !tt.ptr<i32>) {
    %tile = tt.load %counter : !tt.ptr<i32>
    %k = arith.constant 0 : i32
    // CHECK-NOT: tt.multicast_axes
    %v = tt.descriptor_load %a[%tile, %k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @divergent_loop
  tt.func public @divergent_loop(
      %a: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %y step %c1 : i32 {
      // CHECK-NOT: tt.multicast_axes
      %v = tt.descriptor_load %a[%x, %c0] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // CHECK-LABEL: @induction_dependency
  tt.func public @induction_dependency(
      %a: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %x = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    scf.for %i = %x to %c4 step %c1 : i32 {
      // CHECK: tt.descriptor_load {{.*}} {tt.multicast_axes = array<i32: 1>}
      %v = tt.descriptor_load %a[%i, %c0] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // LOWERING-LABEL: @lowering_preserves_plan
  // CHECK-LABEL: @lowering_preserves_plan
  tt.func public @lowering_preserves_plan(
      %b: !tt.tensordesc<tensor<128x64xf16, #shared>>) -> tensor<128x64xf16, #blocked> {
    %y = tt.get_program_id y : i32
    %k = arith.constant 0 : i32
    // CHECK: tt.descriptor_load {{.*}} {tt.multicast_axes = array<i32: 0>}
    // LOWERING: ttng.cluster_barrier
    // LOWERING: ttng.async_tma_copy_global_to_local {{.*}} {tt.multicast_axes = array<i32: 0>}
    %bv = tt.descriptor_load %b[%y, %k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    tt.return %bv : tensor<128x64xf16, #blocked>
  }
}

module attributes {
  "ttg.ctas-per-cga" = true,
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 2 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.multicast" = true,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.use-meta-ws" = true,
  ttg.target = "cuda:90"
} {
  // CHECK-LABEL: @meta_ws_fallback
  tt.func public @meta_ws_fallback(
      %a: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %x = tt.get_program_id x : i32
    %k = arith.constant 0 : i32
    // CHECK-NOT: tt.multicast_axes
    %v = tt.descriptor_load %a[%x, %k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
    tt.return
  }
}
