// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups="instrumented=1" | FileCheck %s --check-prefix=INSTRUMENTED

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {
}

// -----

// A single AutoWS-generated warp-specialize region enables direct warp-ID
// dispatch. The tag is placed on an operation in the region, matching the IR
// emitted by AutoWS.
// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.single-warp-specialize" = true, "ttg.total-num-warps" = 8 : i32} {
// CHECK: tt.func @single_autows_region
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @single_autows_region() {
  ttg.warp_specialize()
  default {
    %c0 = arith.constant {ttg.warp_specialize.tag = 0 : i32} 0 : i32
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// A manual warp-specialize region has no AutoWS tag and must remain under the
// frontend's explicit exclusive-task control.
// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {
// CHECK: tt.func @single_manual_region
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @single_manual_region() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 20 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @kernel() {
  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 18, 4, 12, 16, 19>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  }
  partition2() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  // CHECK: partition3() num_warps(2)
  // CHECK: partition4() num_warps(1)
  tt.return
}

}

// -----

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 16 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @two_warp_specialize() {
  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 12, 14, 4, 15>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  // CHECK: partition2() num_warps(8)
  // CHECK: partition3() num_warps(1)

  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 14, 4, 12, 15>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  } : () -> ()

  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 168 : i32
module attributes {"ttg.num-warps" = 8 : i32} {

tt.func @setmaxnreg() {
  // CHECK: actualRegisters = array<i32: 208, 80, 80, 80>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 48, 80, 48>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(2) {
    ttg.warp_return
  }
  partition2() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 128 : i32} {

// CHECK-LABEL: tt.func @fixed_default_registers
tt.func @fixed_default_registers() {
  // CHECK: actualRegisters = array<i32: 80, 24>
  ttg.warp_specialize() attributes {defaultRequestedRegisters = 80 : i32, requestedRegisters = array<i32: 24>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 128 : i32} {

// CHECK-LABEL: tt.func @fixed_default_shared_worker_registers
tt.func @fixed_default_shared_worker_registers() {
  // CHECK: actualRegisters = array<i32: 80, 176>
  ttg.warp_specialize() attributes {defaultRequestedRegisters = 80 : i32}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 40 : i32
module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 40 : i32} {

tt.func @assert_and_print_min_registers() {
  // CHECK: actualRegisters = array<i32: 48, 32>
  // CHECK: requestedRegisters = array<i32: 24>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24>}
  default {
    %zero = arith.constant 0 : i32
    tt.print "zero: " {hex = false, isSigned = array<i32: 1>} : %zero : i32
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %true = arith.constant true
    tt.assert %true, "assert text" : i1
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// INSTRUMENTED: module attributes {instrumented.test, ttg.maxnreg = 40 : i32
module attributes {"instrumented.test", "ttg.num-warps" = 4 : i32, ttg.maxnreg = 40 : i32} {

tt.func @instrumented_min_registers() {
  // INSTRUMENTED: actualRegisters = array<i32: 48, 32>
  // INSTRUMENTED: requestedRegisters = array<i32: 16>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 16>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 128 : i32
module attributes {"ttg.num-warps" = 8 : i32} {

tt.func @steal_from_default() {
  // CHECK: actualRegisters = array<i32: 64, 192>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 192>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(8) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// Test that user-provided warpGroupStartIds are preserved and padding
// partitions are assigned IDs after the real partitions. This prevents
// padding warps from displacing real task warps to higher IDs.
module attributes {"ttg.num-warps" = 8 : i32} {

// CHECK-LABEL: tt.func @respect_user_start_ids
tt.func @respect_user_start_ids() {
  // User provided [8, 12, 13] for 3 real partitions (4+1+1 = 6 warps).
  // Padding adds 2 warps to reach 8 (next multiple of 4).
  // Padding partition should get startId=14, after the real partitions.
  // CHECK: warpGroupStartIds = array<i32: 8, 12, 13, 14>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 88, 24, 24>, warpGroupStartIds = array<i32: 8, 12, 13>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    ttg.warp_return
  }
  partition2() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  // CHECK: partition3() num_warps(2)
  tt.return
}

}

// -----

// Do-nothing padding partitions must request the legal setmaxregister floor
// (24 on all sm_90+ targets), not a below-minimum value. If a padding partition
// forms a warp group in which no partition requests >= 24, a below-floor value
// reaches nvvm.setmaxregister and fails the verifier ("must be in between 24 to
// 256"). Two 1-warp partitions (2 warps) are padded up to a full warp group (4
// warps); the appended padding partition's request must be 24, not 16.
module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: tt.func @padding_register_floor
tt.func @padding_register_floor() {
  // CHECK: ttg.warp_specialize() attributes {actualRegisters = array<i32: {{[0-9]+}}, 24, 24, 24>, requestedRegisters = array<i32: 24, 24, 24>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24, 24>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}
