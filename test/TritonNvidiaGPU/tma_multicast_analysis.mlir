// RUN: triton-opt %s --triton-nvidia-plan-tma-multicast | FileCheck %s
// RUN: triton-opt %s --triton-nvidia-plan-tma-multicast --triton-nvidia-tma-lowering | FileCheck %s --check-prefix=LOWERING

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 4 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  ttg.target = "cuda:90",
  "ttg.threads-per-warp" = 32 : i32
} {
  // CHECK-LABEL: @rectangular
  tt.func public @rectangular(
      %a: !tt.tensordesc<128x64xf16, #shared>,
      %b: !tt.tensordesc<128x64xf16, #shared>) {
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    %k = arith.constant 0 : i32
    // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 1>}
    %av = tt.descriptor_load %a[%x, %k] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 0>}
    %bv = tt.descriptor_load %b[%y, %k] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @no_reuse
  tt.func public @no_reuse(%a: !tt.tensordesc<128x64xf16, #shared>) {
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    // CHECK-NOT: tt.multicast_axes
    %v = tt.descriptor_load %a[%x, %y] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @per_load_disable
  tt.func public @per_load_disable(%a: !tt.tensordesc<128x64xf16, #shared>) {
    %x = tt.get_program_id x : i32
    %k = arith.constant 0 : i32
    // CHECK-NOT: tt.multicast_axes
    %v = tt.descriptor_load %a[%x, %k] {multicast = false} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @dynamic_tile
  tt.func public @dynamic_tile(
      %a: !tt.tensordesc<128x64xf16, #shared>, %counter: !tt.ptr<i32>) {
    %tile = tt.load %counter : !tt.ptr<i32>
    %k = arith.constant 0 : i32
    // CHECK-NOT: tt.multicast_axes
    %v = tt.descriptor_load %a[%tile, %k] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @divergent_loop
  tt.func public @divergent_loop(
      %a: !tt.tensordesc<128x64xf16, #shared>) {
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %y step %c1 : i32 {
      // CHECK-NOT: tt.multicast_axes
      %v = tt.descriptor_load %a[%x, %c0] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // The load's recipient groups span Y at a fixed X, but lowering uses a
  // full-cluster barrier. An X-dependent branch is therefore unsafe even
  // though it is uniform within each data recipient group.
  // CHECK-LABEL: @subgroup_uniform_cluster_divergent
  tt.func public @subgroup_uniform_cluster_divergent(
      %a: !tt.tensordesc<128x64xf16, #shared>) {
    %x = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %is_x0 = arith.cmpi eq, %x, %c0 : i32
    scf.if %is_x0 {
      // CHECK-NOT: tt.multicast_axes
      %v = tt.descriptor_load %a[%x, %c0] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // CHECK-LABEL: @cluster_uniform_condition
  tt.func public @cluster_uniform_condition(
      %a: !tt.tensordesc<128x64xf16, #shared>, %enabled: i1) {
    %x = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    scf.if %enabled {
      // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 1>}
      %v = tt.descriptor_load %a[%x, %c0] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // CHECK-LABEL: @induction_dependency
  tt.func public @induction_dependency(
      %a: !tt.tensordesc<128x64xf16, #shared>) {
    %x = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    scf.for %i = %x to %c4 step %c1 : i32 {
      // CHECK-NOT: tt.multicast_axes
      %v = tt.descriptor_load %a[%i, %c0] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // Dividing each physical program coordinate by its exact cluster dimension
  // produces a cluster-uniform work id suitable for a persistent loop bound.
  // CHECK-LABEL: @cluster_uniform_quotient
  tt.func public @cluster_uniform_quotient(
      %a: !tt.tensordesc<128x64xf16, #shared>, %limit: i32) {
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %cluster_x = arith.divsi %x, %c2 : i32
    %cluster_y = arith.divui %y, %c4 : i32
    %start = arith.addi %cluster_x, %cluster_y : i32
    scf.for %i = %start to %limit step %c1 : i32 {
      // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 1>}
      %v = tt.descriptor_load %a[%x, %i] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // CHECK-LABEL: @wrong_cluster_divisor
  tt.func public @wrong_cluster_divisor(
      %a: !tt.tensordesc<128x64xf16, #shared>, %limit: i32) {
    %x = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %bad_cluster_x = arith.divui %x, %c3 : i32
    scf.for %i = %bad_cluster_x to %limit step %c1 : i32 {
      // CHECK-NOT: tt.multicast_axes
      %v = tt.descriptor_load %a[%x, %c0] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // LOWERING-LABEL: @lowering_preserves_plan
  // CHECK-LABEL: @lowering_preserves_plan
  tt.func public @lowering_preserves_plan(
      %b: !tt.tensordesc<128x64xf16, #shared>) -> tensor<128x64xf16, #blocked> {
    %y = tt.get_program_id y : i32
    %k = arith.constant 0 : i32
    // CHECK: tt.descriptor_load {{.*}}tt.multicast_axes = array<i32: 0>}
    // LOWERING: ttng.cluster_barrier
    // LOWERING: arith.cmpi eq
    // LOWERING: ttng.async_tma_copy_global_to_local {{.*}}tt.multicast_axes = array<i32: 0>}
    %bv = tt.descriptor_load %b[%y, %k] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return %bv : tensor<128x64xf16, #blocked>
  }

  // CHECK-LABEL: @clc_cluster_schedule
  tt.func public @clc_cluster_schedule(
      %a: !tt.tensordesc<128x64xf16, #shared>,
      %b: !tt.tensordesc<128x64xf16, #shared>) {
    %true = arith.constant true
    %x0 = tt.get_program_id x : i32
    %y0 = tt.get_program_id y : i32
    %z0 = tt.get_program_id z : i32
    %k = arith.constant 0 : i32
    %results:4 = scf.while (%valid = %true, %x = %x0, %y = %y0, %z = %z0) :
        (i1, i32, i32, i32) -> (i1, i32, i32, i32) {
      scf.condition(%valid) %valid, %x, %y, %z : i1, i32, i32, i32
    } do {
    ^bb0(%valid: i1, %x: i32, %y: i32, %z: i32):
      // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 1>}
      %av = tt.descriptor_load %a[%x, %k] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 0>}
      %bv = tt.descriptor_load %b[%y, %k] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %next_valid, %next_x, %next_y, %next_z = ttng.clc_advance : i1, i32, i32, i32
      scf.yield %next_valid, %next_x, %next_y, %next_z : i1, i32, i32, i32
    }
    tt.return
  }
}

module attributes {
  "ttg.cluster-dim-x" = 1 : i32,
  "ttg.cluster-dim-y" = 1 : i32,
  "ttg.cluster-dim-z" = 2 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:90"
} {
  // CHECK-LABEL: @z_axis_broadcast
  tt.func public @z_axis_broadcast(
      %a: !tt.tensordesc<128x64xf16, #shared>) {
    %c0 = arith.constant 0 : i32
    // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 2>}
    %v = tt.descriptor_load %a[%c0, %c0] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @z_axis_dependent
  tt.func public @z_axis_dependent(
      %a: !tt.tensordesc<128x64xf16, #shared>) {
    %z = tt.get_program_id z : i32
    %c0 = arith.constant 0 : i32
    // CHECK-NOT: tt.multicast_axes
    %v = tt.descriptor_load %a[%z, %c0] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @z_axis_uniform_quotient
  tt.func public @z_axis_uniform_quotient(
      %a: !tt.tensordesc<128x64xf16, #shared>, %limit: i32) {
    %z = tt.get_program_id z : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %cluster_z = arith.divui %z, %c2 : i32
    scf.for %i = %cluster_z to %limit step %c1 : i32 {
      // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 2>}
      %v = tt.descriptor_load %a[%c0, %c0] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // CHECK-LABEL: @z_axis_wrong_divisor
  tt.func public @z_axis_wrong_divisor(
      %a: !tt.tensordesc<128x64xf16, #shared>, %limit: i32) {
    %z = tt.get_program_id z : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %bad_cluster_z = arith.divui %z, %c3 : i32
    scf.for %i = %bad_cluster_z to %limit step %c1 : i32 {
      // CHECK-NOT: tt.multicast_axes
      %v = tt.descriptor_load %a[%c0, %c0] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }
}

module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 2 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:90"
} {
  // CHECK-LABEL: @meta_ws_plan
  tt.func public @meta_ws_plan(
      %a: !tt.tensordesc<128x64xf16, #shared>) {
    %x = tt.get_program_id x : i32
    %k = arith.constant 0 : i32
    // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 1>}
    %v = tt.descriptor_load %a[%x, %k] {multicast = true} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }
}
