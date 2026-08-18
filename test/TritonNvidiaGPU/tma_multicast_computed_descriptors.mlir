// RUN: triton-opt %s --triton-nvidia-plan-tma-multicast | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 2 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:90"
} {
  // CHECK-LABEL: @invariant_computed_descriptors
  tt.func public @invariant_computed_descriptors(
      %base: !tt.ptr<f16>, %raw_desc: !tt.ptr<i8>) {
    %pid_x = tt.get_program_id x : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c128_i32 = arith.constant 128 : i32
    %desc = tt.make_tensor_descriptor %base, [%c128_i32, %c64_i32],
        [%c64_i64, %c1_i64] : !tt.ptr<f16>,
        !tt.tensordesc<128x64xf16, #shared>
    // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 1>}
    %0 = tt.descriptor_load %desc[%pid_x, %c0_i32] {multicast = true} :
        !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    %reinterpreted = ttng.reinterpret_tensor_descriptor %raw_desc :
        !tt.ptr<i8> to !tt.tensordesc<128x64xf16, #shared>
    // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 1>}
    %1 = tt.descriptor_load %reinterpreted[%pid_x, %c0_i32] {multicast = true} :
        !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // The recipient group spans Y at a fixed X, but lowering synchronizes the
  // full cluster. An X-dependent branch must therefore remain non-multicast.
  // CHECK-LABEL: @subgroup_uniform_cluster_divergent
  tt.func public @subgroup_uniform_cluster_divergent(
      %raw_desc: !tt.ptr<i8>) {
    %pid_x = tt.get_program_id x : i32
    %c0_i32 = arith.constant 0 : i32
    %is_x0 = arith.cmpi eq, %pid_x, %c0_i32 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw_desc :
        !tt.ptr<i8> to !tt.tensordesc<128x64xf16, #shared>
    scf.if %is_x0 {
      // CHECK-NOT: tt.multicast_axes
      // CHECK: tt.descriptor_load
      %0 = tt.descriptor_load %desc[%pid_x, %c0_i32] {multicast = true} :
          !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // CHECK-LABEL: @cluster_uniform_condition
  tt.func public @cluster_uniform_condition(
      %raw_desc: !tt.ptr<i8>, %enabled: i1) {
    %pid_x = tt.get_program_id x : i32
    %c0_i32 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw_desc :
        !tt.ptr<i8> to !tt.tensordesc<128x64xf16, #shared>
    scf.if %enabled {
      // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 1>}
      %0 = tt.descriptor_load %desc[%pid_x, %c0_i32] {multicast = true} :
          !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    }
    tt.return
  }

  // CHECK-LABEL: @cluster_divergent_while_condition
  tt.func public @cluster_divergent_while_condition(
      %raw_desc: !tt.ptr<i8>) {
    %pid_x = tt.get_program_id x : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %desc = ttng.reinterpret_tensor_descriptor %raw_desc :
        !tt.ptr<i8> to !tt.tensordesc<128x64xf16, #shared>
    %result = scf.while (%keep = %true) : (i1) -> i1 {
      %next = arith.cmpi eq, %pid_x, %c0_i32 : i32
      scf.condition(%next) %next : i1
    } do {
    ^bb0(%keep: i1):
      // CHECK-NOT: tt.multicast_axes
      // CHECK: tt.descriptor_load
      %0 = tt.descriptor_load %desc[%pid_x, %c0_i32] {multicast = true} :
          !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      scf.yield %keep : i1
    }
    tt.return
  }

  // CHECK-LABEL: @cluster_uniform_while_condition
  tt.func public @cluster_uniform_while_condition(
      %raw_desc: !tt.ptr<i8>, %enabled: i1) {
    %pid_x = tt.get_program_id x : i32
    %c0_i32 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw_desc :
        !tt.ptr<i8> to !tt.tensordesc<128x64xf16, #shared>
    %result = scf.while (%keep = %enabled) : (i1) -> i1 {
      scf.condition(%keep) %keep : i1
    } do {
    ^bb0(%keep: i1):
      // CHECK: tt.descriptor_load {{.*}}multicast = true{{.*}}tt.multicast_axes = array<i32: 1>}
      %0 = tt.descriptor_load %desc[%pid_x, %c0_i32] {multicast = true} :
          !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      scf.yield %keep : i1
    }
    tt.return
  }

  // CHECK-LABEL: @pid_dependent_descriptor
  tt.func public @pid_dependent_descriptor(
      %base0: !tt.ptr<f16>, %base1: !tt.ptr<f16>) {
    %pid_x = tt.get_program_id x : i32
    %pid_y = tt.get_program_id y : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c128_i32 = arith.constant 128 : i32
    %is_y0 = arith.cmpi eq, %pid_y, %c0_i32 : i32
    %base = arith.select %is_y0, %base0, %base1 : !tt.ptr<f16>
    %desc = tt.make_tensor_descriptor %base, [%c128_i32, %c64_i32],
        [%c64_i64, %c1_i64] : !tt.ptr<f16>,
        !tt.tensordesc<128x64xf16, #shared>
    // CHECK-NOT: tt.multicast_axes
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%pid_x, %c0_i32] {multicast = true} :
        !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @unsupported_region
  tt.func public @unsupported_region(%raw_desc: !tt.ptr<i8>) {
    %pid_x = tt.get_program_id x : i32
    %c0_i32 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw_desc :
        !tt.ptr<i8> to !tt.tensordesc<128x64xf16, #shared>
    // CHECK: scf.execute_region
    // CHECK-NOT: tt.multicast_axes
    // CHECK: tt.descriptor_load
    scf.execute_region {
      %0 = tt.descriptor_load %desc[%pid_x, %c0_i32] {multicast = true} :
          !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      scf.yield
    }
    tt.return
  }

  // CHECK-LABEL: @private_helper
  tt.func private @private_helper(%raw_desc: !tt.ptr<i8>, %offset: i32) {
    %c0_i32 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw_desc :
        !tt.ptr<i8> to !tt.tensordesc<128x64xf16, #shared>
    // CHECK-NOT: tt.multicast_axes
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%offset, %c0_i32] {multicast = true} :
        !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    tt.return
  }
}
