// RUN: triton-opt %s -triton-materialize-integer-facts -test-integer-facts="generic=true" -test-print-alignment -verify-diagnostics=only-expected -split-input-file -o /dev/null
// RUN: triton-opt %s -triton-materialize-integer-facts -test-print-alignment -split-input-file -verify-diagnostics=only-expected -o %t.once
// RUN: triton-opt %t.once -triton-materialize-integer-facts -split-input-file -o %t.twice
// RUN: diff %t.once %t.twice
// RUN: triton-opt %s -triton-materialize-integer-facts -test-integer-facts -test-print-alignment -split-input-file -verify-diagnostics=only-expected -o /dev/null
// RUN: triton-opt %s -triton-materialize-integer-facts -triton-materialize-integer-facts -test-print-alignment -verify-diagnostics=only-expected -split-input-file | FileCheck %s --check-prefix=REFINE
// RUN: triton-opt %s -triton-materialize-integer-facts -test-print-alignment -triton-erase-integer-refinements -verify-diagnostics=only-expected -split-input-file | FileCheck %s --check-prefix=ERASE
// RUN: triton-opt %s -triton-materialize-integer-facts -triton-narrow-integer-ranges -canonicalize -split-input-file | FileCheck %s --check-prefix=NARROW

// ERASE-NOT: tt.refine
// ERASE: llvm.intr.assume
// ERASE-NOT: tt.refine

// -----

module {
  tt.func @unbounded_call(%x: i32, %y: i32, %bound: i32) -> i32 {
    %bounded = tt.refine %x, %bound {predicate = 2 : i64} : i32
    %r = tt.call @identity(%bounded, %bound) : (i32, i32) -> i32
    %s = tt.call @identity(%y, %bound) : (i32, i32) -> i32
    %no_leak = arith.cmpi slt, %s, %bound {test.proof = "unknown"} : i32
    tt.return %s : i32
  }
  tt.func private @identity(%value: i32, %bound: i32) -> i32 {
    tt.return %value : i32
  }
}

// -----

tt.func @unsigned_bound_is_not_signed(%value: i64) -> i64 {
  %zero = arith.constant 0 : i64
  %positive = tt.refine %value, %zero {predicate = 8 : i64} : i64
  %unsigned = arith.cmpi ugt, %positive, %zero {test.proof = "proven"} : i64
  %signed = arith.cmpi sgt, %positive, %zero {test.proof = "unknown"} : i64
  tt.return %positive : i64
}

// -----

// REFINE-LABEL: tt.func @refine_branch_scope
// REFINE: scf.if
// REFINE: %[[LOW:.*]] = tt.refine %{{.*}}, %{{.*}} {predicate = 5 : i64}
// REFINE: %[[HIGH:.*]] = tt.refine %[[LOW]], %{{.*}} {predicate = 2 : i64}
// REFINE-NOT: tt.refine
// REFINE: tt.splat %[[HIGH]]
// REFINE: } else {
// REFINE-NOT: tt.refine
// REFINE: tt.splat %arg0
tt.func @refine_branch_scope(%x: i32) {
  %zero = arith.constant 0 : i32
  %upper = arith.constant 1024 : i32
  %nonnegative = arith.cmpi sge, %x, %zero : i32
  %bounded = arith.cmpi slt, %x, %upper : i32
  %both = arith.andi %bounded, %nonnegative : i1
  scf.if %both {
    %base = tt.splat %x : i32 -> tensor<4xi32>
    %stride = arith.constant dense<4> : tensor<4xi32>
    %scaled = arith.muli %base, %stride : tensor<4xi32>
    %lane = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %index = arith.addi %scaled, %lane : tensor<4xi32>
    // expected-remark @below {{contiguity = [4], divisibility = [4], constancy = [1]}}
    %rem = arith.remsi %index, %stride : tensor<4xi32>
  } else {
    %base = tt.splat %x : i32 -> tensor<4xi32>
    %stride = arith.constant dense<4> : tensor<4xi32>
    %scaled = arith.muli %base, %stride : tensor<4xi32>
    %lane = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %index = arith.addi %scaled, %lane : tensor<4xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
    %rem = arith.remsi %index, %stride : tensor<4xi32>
  }
  tt.return
}

// -----

// REFINE-LABEL: tt.func @refine_dynamic_bound
// REFINE: tt.refine
tt.func @refine_dynamic_bound(%x: i32, %limit: i32) {
  %zero = arith.constant 0 : i32
  %upper = arith.constant 1024 : i32
  %limit_bound = tt.refine %limit, %upper {predicate = 3 : i64} : i32
  %positive = tt.refine %x, %zero {predicate = 5 : i64} : i32
  %bounded = tt.refine %positive, %limit_bound {predicate = 2 : i64} : i32
  %proof = arith.cmpi slt, %bounded, %upper {test.proof = "proven"} : i32
  %inverse = arith.cmpi sge, %bounded, %upper {test.proof = "disproven"} : i32
  %sibling = arith.cmpi slt, %x, %limit_bound {test.proof = "unknown"} : i32
  %base = tt.splat %bounded : i32 -> tensor<4xi32>
  %stride = arith.constant dense<4> : tensor<4xi32>
  %scaled = arith.muli %base, %stride : tensor<4xi32>
  %lane = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %index = arith.addi %scaled, %lane : tensor<4xi32>
  // expected-remark @below {{contiguity = [4], divisibility = [4], constancy = [1]}}
  %rem = arith.remsi %index, %stride : tensor<4xi32>
  %original = tt.splat %x : i32 -> tensor<4xi32>
  %original_scaled = arith.muli %original, %stride : tensor<4xi32>
  %original_index = arith.addi %original_scaled, %lane : tensor<4xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
  %original_rem = arith.remsi %original_index, %stride : tensor<4xi32>
  tt.return
}

// -----

tt.func @refine_existing_chain(%x: i32) {
  %zero = arith.constant 0 : i32
  %upper = arith.constant 1024 : i32
  %positive = tt.refine %x, %zero {predicate = 5 : i64} : i32
  %condition = arith.cmpi slt, %x, %upper : i32
  llvm.intr.assume %condition : i1
  %lower = arith.cmpi sge, %x, %zero {test.proof = "unknown"} : i32
  %refined_lower = arith.cmpi sge, %positive, %zero {test.proof = "proven"} : i32
  %refined_upper = arith.cmpi slt, %positive, %upper {test.proof = "proven"} : i32
  %bound = arith.cmpi slt, %x, %upper {test.proof = "proven"} : i32
  tt.return
}

// -----

tt.func @refine_merge_and_freeze(%x: i32, %limit: i32, %cond: i1) {
  %zero = arith.constant 0 : i32
  %negative = tt.refine %x, %zero {predicate = 2 : i64} : i32
  %frozen = llvm.freeze %negative : i32
  %freeze_proof = arith.cmpi slt, %frozen, %zero {test.proof = "unknown"} : i32
  %merged = scf.if %cond -> (i32) {
    %low = tt.refine %x, %limit {predicate = 2 : i64} : i32
    scf.yield %low : i32
  } else {
    %high = tt.refine %x, %limit {predicate = 5 : i64} : i32
    scf.yield %high : i32
  }
  %merge_proof = arith.cmpi slt, %merged, %limit {test.proof = "unknown"} : i32
  %one = arith.constant 1 : i32
  %loop = scf.for %iv = %zero to %limit step %one iter_args(%carry = %x) -> (i32) : i32 {
    %next = tt.refine %carry, %iv {predicate = 2 : i64} : i32
    scf.yield %next : i32
  }
  %loop_proof = arith.cmpi slt, %loop, %limit {test.proof = "unknown"} : i32
  %positive = tt.refine %x, %zero {predicate = 5 : i64} : i32
  %contradiction = tt.refine %positive, %zero {predicate = 2 : i64} : i32
  %sibling = arith.cmpi sge, %x, %zero {test.proof = "unknown"} : i32
  tt.return
}

// -----

tt.func @refine_missing_tile_bound(%x: i32, %rows: i32, %columns: i32) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %eight = arith.constant 8 : i32
  %limit = arith.constant 255 : i32
  %m0 = tt.refine %rows, %one {predicate = 5 : i64} : i32
  %m = tt.refine %m0, %limit {predicate = 3 : i64} : i32
  %n0 = tt.refine %columns, %one {predicate = 5 : i64} : i32
  %n = tt.refine %n0, %limit {predicate = 3 : i64} : i32
  %positive = tt.refine %x, %zero {predicate = 5 : i64} : i32
  %width = arith.muli %eight, %n : i32
  %chunk = arith.divsi %positive, %width : i32
  %base = arith.muli %chunk, %eight : i32
  %remaining = arith.subi %m, %base : i32
  %divisor = arith.minsi %remaining, %eight : i32
  %sign = arith.cmpi sgt, %divisor, %zero {test.proof = "unknown"} : i32
  tt.return
}

// -----

// REFINE-LABEL: tt.func @refine_loop_assumption_chain
// REFINE: scf.for %[[IV:.*]] =
// REFINE: %[[LOWER:.*]] = tt.refine %[[IV]], %{{.*}} {predicate = 5 : i64}
// REFINE: %[[UPPER:.*]] = tt.refine %[[LOWER]], %{{.*}} {predicate = 2 : i64}
// REFINE: llvm.intr.assume
// REFINE: %[[BOTH:.*]] = tt.refine %[[UPPER]], %{{.*}} {predicate = 5 : i64}
// REFINE-NOT: tt.refine
// REFINE: arith.cmpi slt, %[[BOTH]]
tt.func @refine_loop_assumption_chain(%start: i32, %limit: i32, %step: i32) {
  %zero = arith.constant 0 : i32
  scf.for %iv = %start to %limit step %step : i32 {
    %condition = arith.cmpi sge, %iv, %zero : i32
    llvm.intr.assume %condition : i1
    %bound = arith.cmpi slt, %iv, %limit {test.proof = "unknown"} : i32
    %sign = arith.cmpi sge, %iv, %zero {test.proof = "proven"} : i32
  }
  tt.return
}

// -----

// REFINE-LABEL: tt.func @refine_signed_loop_lower_bound
// REFINE: scf.for
// REFINE: %[[LOW:.*]] = tt.refine %{{.*}}, %{{.*}} {predicate = 5 : i64}
// REFINE: tt.refine %[[LOW]], %{{.*}} {predicate = 2 : i64}
tt.func @refine_signed_loop_lower_bound(%start: i32, %limit: i32, %step: i32) {
  %zero = arith.constant 0 : i32
  scf.for %iv = %start to %limit step %step : i32 {
    %lower = arith.cmpi sge, %iv, %start {test.proof = "unknown"} : i32
    %upper = arith.cmpi slt, %iv, %limit {test.proof = "unknown"} : i32
    %sign = arith.cmpi sge, %iv, %zero {test.proof = "unknown"} : i32
  }
  %outside = arith.cmpi slt, %start, %limit {test.proof = "unknown"} : i32
  tt.return
}

// -----

// REFINE-LABEL: tt.func @refine_unsigned_loop_lower_bound
// REFINE: scf.for
// REFINE: %[[LOW:.*]] = tt.refine %{{.*}}, %{{.*}} {predicate = 9 : i64}
// REFINE: tt.refine %[[LOW]], %{{.*}} {predicate = 6 : i64}
tt.func @refine_unsigned_loop_lower_bound(%start: i32, %limit: i32, %step: i32) {
  %zero = arith.constant 0 : i32
  scf.for unsigned %iv = %start to %limit step %step : i32 {
    %lower = arith.cmpi uge, %iv, %start {test.proof = "unknown"} : i32
    %upper = arith.cmpi ult, %iv, %limit {test.proof = "unknown"} : i32
    %signed = arith.cmpi sge, %iv, %start {test.proof = "unknown"} : i32
    %sign = arith.cmpi sge, %iv, %zero {test.proof = "unknown"} : i32
  }
  %outside = arith.cmpi ult, %start, %limit {test.proof = "unknown"} : i32
  tt.return
}

// -----

tt.func @refine_unselected_relation(%x: i32, %limit: i32, %condition: i1) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %positive = tt.refine %limit, %one {predicate = 5 : i64} : i32
  %bounded = tt.refine %x, %positive {predicate = 2 : i64} : i32
  %selected = arith.select %condition, %bounded, %x : i32
  %identity = arith.addi %selected, %zero : i32
  %proof = arith.cmpi slt, %identity, %positive {test.proof = "unknown"} : i32
  tt.return
}

// -----

tt.func @refine_wrapping_product_quotient(%x: i32, %groups: i32, %columns: i32) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %sixteen = arith.constant 16 : i32
  %limit = arith.constant 268435456 : i32
  %g0 = tt.refine %groups, %one {predicate = 5 : i64} : i32
  %g = tt.refine %g0, %sixteen {predicate = 3 : i64} : i32
  %n0 = tt.refine %columns, %one {predicate = 5 : i64} : i32
  %n = tt.refine %n0, %limit {predicate = 3 : i64} : i32
  %total = arith.muli %g, %n : i32
  %positive = tt.refine %x, %zero {predicate = 5 : i64} : i32
  %bounded = tt.refine %positive, %total {predicate = 2 : i64} : i32
  %q = arith.divsi %bounded, %g : i32
  %base = tt.splat %q : i32 -> tensor<4xi32>
  %stride = arith.constant dense<4> : tensor<4xi32>
  %scaled = arith.muli %base, %stride : tensor<4xi32>
  %lane = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %index = arith.addi %scaled, %lane : tensor<4xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
  %rem = arith.remsi %index, %stride : tensor<4xi32>
  tt.return
}

// -----

// REFINE-LABEL: tt.func @refine_shared_successor
// REFINE: cf.cond_br
// REFINE: tt.refine
// REFINE: cf.br
// REFINE: tt.refine
// REFINE: cf.br
tt.func @refine_shared_successor(%x: i32, %limit: i32) -> i32 {
  %condition = arith.cmpi slt, %x, %limit : i32
  cf.cond_br %condition, ^join(%x : i32), ^join(%x : i32)
^join(%arg: i32):
  %unknown = arith.cmpi slt, %arg, %limit {test.proof = "unknown"} : i32
  tt.return %arg : i32
}

// -----

tt.func @negative_dividend_dynamic_remainder(%input: i32, %divisor: i32) {
  %zero = arith.constant 0 : i32
  %lower = arith.constant -1024 : i32
  %bounded = arith.cmpi sge, %input, %lower : i32
  %negative = arith.cmpi slt, %input, %zero : i32
  llvm.intr.assume %bounded : i1
  llvm.intr.assume %negative : i1
  %rem = arith.remsi %input, %divisor : i32
  %minus_one = arith.constant -1 : i32
  %mapped = arith.divsi %rem, %minus_one : i32
  %tile = arith.constant 128 : i32
  %base = arith.muli %mapped, %tile : i32
  %bases = tt.splat %base : i32 -> tensor<128xi32>
  %lanes = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %indices = arith.addi %bases, %lanes : tensor<128xi32>
  %four = arith.constant dense<4> : tensor<128xi32>
  // expected-remark @below {{contiguity = [4], divisibility = [4], constancy = [1]}}
  %result = arith.remsi %indices, %four : tensor<128xi32>
  tt.return
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx950"} {
  tt.func @launch_bound_unknown_divisor_sign(%group: i32) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id x : i32
    %mapped = arith.divsi %pid, %group : i32
    %tile = arith.constant 128 : i32
    %base = arith.muli %mapped, %tile : i32
    %bases = tt.splat %base : i32 -> tensor<128xi32>
    %lane = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %coordinate = arith.addi %bases, %lane : tensor<128xi32>
    %divisor = arith.constant dense<16> : tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
    %remainder = arith.remsi %coordinate, %divisor : tensor<128xi32>
    tt.return
  }

  tt.func @launch_bound_dynamic_remapping(%modulus: i32, %group: i32) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id x : i32
    %within_group = arith.remsi %pid, %modulus : i32
    %mapped = arith.divsi %within_group, %group : i32
    %zero = arith.constant 0 : i32
    %positive = arith.cmpi sge, %mapped, %zero : i32
    llvm.intr.assume %positive : i1
    %tile = arith.constant 128 : i32
    %base = arith.muli %mapped, %tile : i32
    %bases = tt.splat %base : i32 -> tensor<128xi32>
    %lane = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %coordinate = arith.addi %bases, %lane : tensor<128xi32>
    %divisor = arith.constant dense<16> : tensor<128xi32>
    // expected-remark @below {{contiguity = [16], divisibility = [16], constancy = [1]}}
    %remainder = arith.remsi %coordinate, %divisor : tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [16]}}
    %quotient = arith.divsi %coordinate, %divisor : tensor<128xi32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx950"} {
  tt.func @launch_bound_x_safe(%divisor: i32 {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id x : i32
    %tile = arith.constant 128 : i32
    %base = arith.muli %pid, %tile : i32
    %base_tensor = tt.splat %base : i32 -> tensor<128xi32>
    %lane = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %coordinate = arith.addi %base_tensor, %lane : tensor<128xi32>
    %divisor_tensor = tt.splat %divisor : i32 -> tensor<128xi32>
    // expected-remark @below {{contiguity = [16], divisibility = [16], constancy = [1], constant_value = <none>}}
    %remainder = arith.remsi %coordinate, %divisor_tensor : tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
    %quotient = arith.divsi %coordinate, %divisor_tensor : tensor<128xi32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx950"} {
  tt.func @launch_bound_x_overflow(%divisor: i32 {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id x : i32
    %tile = arith.constant 256 : i32
    %base = arith.muli %pid, %tile : i32
    %base_tensor = tt.splat %base : i32 -> tensor<256xi32>
    %lane = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %coordinate = arith.addi %base_tensor, %lane : tensor<256xi32>
    %divisor_tensor = tt.splat %divisor : i32 -> tensor<256xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %remainder = arith.remsi %coordinate, %divisor_tensor : tensor<256xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %quotient = arith.divsi %coordinate, %divisor_tensor : tensor<256xi32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx1100"} {
  tt.func @launch_bound_wave32_safe(%divisor: i32 {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id x : i32
    %tile = arith.constant 64 : i32
    %base = arith.muli %pid, %tile : i32
    %base_tensor = tt.splat %base : i32 -> tensor<64xi32>
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %coordinate = arith.addi %base_tensor, %lane : tensor<64xi32>
    %divisor_tensor = tt.splat %divisor : i32 -> tensor<64xi32>
    // expected-remark @below {{contiguity = [16], divisibility = [16], constancy = [1], constant_value = <none>}}
    %remainder = arith.remsi %coordinate, %divisor_tensor : tensor<64xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
    %quotient = arith.divsi %coordinate, %divisor_tensor : tensor<64xi32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx1100"} {
  tt.func @launch_bound_wave32_overflow(%divisor: i32 {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id x : i32
    %tile = arith.constant 128 : i32
    %base = arith.muli %pid, %tile : i32
    %base_tensor = tt.splat %base : i32 -> tensor<128xi32>
    %lane = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %coordinate = arith.addi %base_tensor, %lane : tensor<128xi32>
    %divisor_tensor = tt.splat %divisor : i32 -> tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %remainder = arith.remsi %coordinate, %divisor_tensor : tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %quotient = arith.divsi %coordinate, %divisor_tensor : tensor<128xi32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx950"} {
  tt.func @launch_bound_y_unbounded(%divisor: i32 {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id y : i32
    %tile = arith.constant 128 : i32
    %base = arith.muli %pid, %tile : i32
    %base_tensor = tt.splat %base : i32 -> tensor<128xi32>
    %lane = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %coordinate = arith.addi %base_tensor, %lane : tensor<128xi32>
    %divisor_tensor = tt.splat %divisor : i32 -> tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %remainder = arith.remsi %coordinate, %divisor_tensor : tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %quotient = arith.divsi %coordinate, %divisor_tensor : tensor<128xi32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx950"} {
  tt.func @launch_bound_z_unbounded(%divisor: i32 {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id z : i32
    %tile = arith.constant 128 : i32
    %base = arith.muli %pid, %tile : i32
    %base_tensor = tt.splat %base : i32 -> tensor<128xi32>
    %lane = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %coordinate = arith.addi %base_tensor, %lane : tensor<128xi32>
    %divisor_tensor = tt.splat %divisor : i32 -> tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %remainder = arith.remsi %coordinate, %divisor_tensor : tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %quotient = arith.divsi %coordinate, %divisor_tensor : tensor<128xi32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 4 : i32, ttg.target = "hip:gfx1250"} {
  tt.func @launch_bound_cluster_safe(%divisor: i32 {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id x : i32
    %tile = arith.constant 128 : i32
    %base = arith.muli %pid, %tile : i32
    %base_tensor = tt.splat %base : i32 -> tensor<128xi32>
    %lane = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %coordinate = arith.addi %base_tensor, %lane : tensor<128xi32>
    %divisor_tensor = tt.splat %divisor : i32 -> tensor<128xi32>
    // expected-remark @below {{contiguity = [16], divisibility = [16], constancy = [1], constant_value = <none>}}
    %remainder = arith.remsi %coordinate, %divisor_tensor : tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
    %quotient = arith.divsi %coordinate, %divisor_tensor : tensor<128xi32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:90"} {
  tt.func @launch_bound_other_target(%divisor: i32 {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id x : i32
    %tile = arith.constant 64 : i32
    %base = arith.muli %pid, %tile : i32
    %base_tensor = tt.splat %base : i32 -> tensor<64xi32>
    %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %coordinate = arith.addi %base_tensor, %lane : tensor<64xi32>
    %divisor_tensor = tt.splat %divisor : i32 -> tensor<64xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %remainder = arith.remsi %coordinate, %divisor_tensor : tensor<64xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %quotient = arith.divsi %coordinate, %divisor_tensor : tensor<64xi32>
    tt.return
  }
}



// -----

tt.func @cast() {
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %cst = arith.constant 1 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %0 = arith.extsi %cst : i32 to i64
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %cst_tensor = arith.constant dense<1> : tensor<128xi32>
  // Bitcast preserves axis bounds for same-width types, but float results have
  // no integer constant.
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %1 = tt.bitcast %cst_tensor : tensor<128xi32> -> tensor<128xf32>
  tt.return
}

// -----

tt.func @require_layout(%arg0: tensor<128xi32> {tt.contiguity = 16 : i32, tt.divisibility = 4 : i32, tt.constancy = 2 : i32}) {
  // expected-remark @below {{contiguity = [16], divisibility = [4], constancy = [2], constant_value = <none>}}
  %0 = ttg.require_layout %arg0 : tensor<128xi32> -> tensor<128xi32>
  tt.return
}

// -----

tt.func @integer_cast_constants() {
  %minus_one = arith.constant -1 : i8
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 255}}
  %unsigned = arith.extui %minus_one : i8 to i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = -1}}
  %signed = arith.extsi %minus_one : i8 to i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = -1}}
  %narrow = arith.trunci %unsigned : i32 to i8
  %true = arith.constant true
  // expected-remark @below {{constant_value = 1}}
  %unsigned_bool = arith.extui %true : i1 to i32
  // expected-remark @below {{constant_value = -1}}
  %signed_bool = arith.extsi %true : i1 to i32
  %splat = arith.constant dense<256> : tensor<4xi16>
  // Truncation to zero improves the known divisibility.
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [4], constant_value = 0}}
  %zero = arith.trunci %splat : tensor<4xi16> to tensor<4xi8>
  tt.return
}

// -----

tt.func @release_layout(%arg0: tensor<128xi32> {tt.contiguity = 16 : i32, tt.divisibility = 4 : i32, tt.constancy = 2 : i32}) {
  // expected-remark @below {{contiguity = [16], divisibility = [4], constancy = [2], constant_value = <none>}}
  %0 = ttg.release_layout %arg0 : tensor<128xi32> -> tensor<128xi32>
  tt.return
}

// -----

tt.func @wide_integer_cast_constants() {
  %minus_one = arith.constant -1 : i64
  // expected-remark @below {{constant_value = -1}}
  %signed = arith.extsi %minus_one : i64 to i128
  // The positive 2^64 - 1 remains exact in the wider constant field.
  // expected-remark @below {{constant_value = 18446744073709551615}}
  %unsigned = arith.extui %minus_one : i64 to i128
  // expected-remark @below {{constant_value = -1}}
  %round_trip = arith.trunci %unsigned : i128 to i64
  // expected-remark @below {{constant_value = -1}}
  %narrow = arith.trunci %signed : i128 to i8
  tt.return
}

// -----

tt.func @tlx_layout_boundaries(%arg0: tensor<128xi32> {tt.contiguity = 16 : i32, tt.divisibility = 4 : i32, tt.constancy = 2 : i32}) {
  // expected-remark @below {{contiguity = [16], divisibility = [4], constancy = [2], constant_value = <none>}}
  %0 = tlx.require_layout %arg0 : tensor<128xi32> -> tensor<128xi32>
  // expected-remark @below {{contiguity = [16], divisibility = [4], constancy = [2], constant_value = <none>}}
  %1 = tlx.release_layout %0 : tensor<128xi32> -> tensor<128xi32>
  tt.return
}

// -----

tt.func @narrow_integer_arithmetic_constants() {
  %min = arith.constant -128 : i8
  %one = arith.constant 1 : i8
  // Wrapping to zero recovers the strongest divisibility.
  // expected-remark @below {{divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %sum = arith.addi %min, %min : i8
  // expected-remark @below {{constant_value = 127}}
  %difference = arith.subi %min, %one : i8
  %sixteen = arith.constant dense<16> : tensor<4xi8>
  // expected-remark @below {{divisibility = [4611686018427387904], constancy = [4], constant_value = 0}}
  %product = arith.muli %sixteen, %sixteen : tensor<4xi8>
  // expected-remark @below {{constant_value = 0}}
  %shift = arith.shli %min, %one : i8
  tt.return
}

// -----

tt.func @signed_unsigned_integer_constants() {
  %negative = arith.constant -7 : i8
  %one = arith.constant 1 : i8
  %two = arith.constant 2 : i8
  // The same i8 bits represent either -7 or 249.
  // expected-remark @below {{constant_value = -3}}
  %signed_div = arith.divsi %negative, %two : i8
  // expected-remark @below {{constant_value = 124}}
  %unsigned_div = arith.divui %negative, %two : i8
  // expected-remark @below {{constant_value = -1}}
  %signed_rem = arith.remsi %negative, %two : i8
  // expected-remark @below {{constant_value = 1}}
  %unsigned_rem = arith.remui %negative, %two : i8
  // expected-remark @below {{constant_value = -4}}
  %signed_shift = arith.shrsi %negative, %one : i8
  // expected-remark @below {{constant_value = 124}}
  %unsigned_shift = arith.shrui %negative, %one : i8
  // expected-remark @below {{constant_value = -7}}
  %signed_min = arith.minsi %negative, %two : i8
  // expected-remark @below {{constant_value = 2}}
  %unsigned_min = arith.minui %negative, %two : i8
  // expected-remark @below {{constant_value = 2}}
  %signed_max = arith.maxsi %negative, %two : i8
  // expected-remark @below {{constant_value = -7}}
  %unsigned_max = arith.maxui %negative, %two : i8
  // expected-remark @below {{constant_value = -1}}
  %signed_cmp = arith.cmpi slt, %negative, %two : i8
  // expected-remark @below {{constant_value = 0}}
  %unsigned_cmp = arith.cmpi ult, %negative, %two : i8
  tt.return
}

// -----

tt.func @invalid_integer_constants() {
  %zero = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %min = arith.constant -128 : i8
  %minus_one = arith.constant -1 : i8
  %width = arith.constant 8 : i8
  // Undefined division and remainder must not produce exact constants.
  // expected-remark @below {{constant_value = <none>}}
  %div_zero = arith.divsi %one, %zero : i8
  // expected-remark @below {{constant_value = <none>}}
  %rem_zero = arith.remui %one, %zero : i8
  // expected-remark @below {{constant_value = <none>}}
  %div_overflow = arith.divsi %min, %minus_one : i8
  // Unlike signed division, this remainder is defined.
  // expected-remark @below {{constant_value = 0}}
  %rem_min = arith.remsi %min, %minus_one : i8
  // A shift by the element width produces poison.
  // expected-remark @below {{constant_value = <none>}}
  %left_overshift = arith.shli %one, %width : i8
  // expected-remark @below {{constant_value = <none>}}
  %right_overshift = arith.shrui %one, %width : i8
  tt.return
}

// -----

tt.func @integer_overflow_flags() {
  %zero = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %two = arith.constant 2 : i8
  %min = arith.constant -128 : i8
  %max = arith.constant 127 : i8
  // Each flagged operation rejects the corresponding kind of wrap.
  // expected-remark @below {{constant_value = <none>}}
  %signed_add = arith.addi %max, %one overflow<nsw> : i8
  // expected-remark @below {{constant_value = <none>}}
  %unsigned_sub = arith.subi %zero, %one overflow<nuw> : i8
  // expected-remark @below {{constant_value = <none>}}
  %unsigned_mul = arith.muli %min, %two overflow<nuw> : i8
  // expected-remark @below {{constant_value = <none>}}
  %signed_shift = arith.shli %max, %one overflow<nsw> : i8
  // expected-remark @below {{constant_value = 3}}
  %valid = arith.addi %one, %two overflow<nsw, nuw> : i8
  tt.return
}

// -----

tt.func @integer_exact_flags() {
  %odd = arith.constant -7 : i8
  %even = arith.constant -8 : i8
  %one = arith.constant 1 : i8
  %two = arith.constant 2 : i8
  // Exact division and shifts cannot discard nonzero bits.
  // expected-remark @below {{constant_value = <none>}}
  %signed_div = arith.divsi %odd, %two exact : i8
  // expected-remark @below {{constant_value = <none>}}
  %unsigned_div = arith.divui %odd, %two exact : i8
  // expected-remark @below {{constant_value = <none>}}
  %inexact_shift = arith.shrui %odd, %one exact : i8
  // expected-remark @below {{constant_value = 124}}
  %valid_div = arith.divui %even, %two exact : i8
  // expected-remark @below {{constant_value = -4}}
  %valid_shift = arith.shrsi %even, %one exact : i8
  tt.return
}

// -----

tt.func @integer_cast_flags() {
  %negative = arith.constant -1 : i8
  %positive = arith.constant 1 : i8
  %unsigned_byte = arith.constant 255 : i16
  %signed_byte = arith.constant -1 : i16
  // expected-remark @below {{constant_value = <none>}}
  %negative_ext = arith.extui %negative nneg : i8 to i16
  // expected-remark @below {{constant_value = 1}}
  %positive_ext = arith.extui %positive nneg : i8 to i16
  // 255 fits unsigned i8, while -1 fits signed i8.
  // expected-remark @below {{constant_value = <none>}}
  %signed_trunc_overflow = arith.trunci %unsigned_byte overflow<nsw> : i16 to i8
  // expected-remark @below {{constant_value = -1}}
  %unsigned_trunc = arith.trunci %unsigned_byte overflow<nuw> : i16 to i8
  // expected-remark @below {{constant_value = <none>}}
  %unsigned_trunc_overflow = arith.trunci %signed_byte overflow<nuw> : i16 to i8
  // expected-remark @below {{constant_value = -1}}
  %signed_trunc = arith.trunci %signed_byte overflow<nsw> : i16 to i8
  tt.return
}

// -----

tt.func @bitcast_pointer_element_width(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i1> {tt.divisibility = 16 : i32}) {
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  // expected-remark @below {{contiguity = [128], divisibility = [16], constancy = [1], constant_value = <none>}}
  %ptrs = tt.addptr %base, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [16], constancy = [1], constant_value = <none>}}
  %same_width = tt.bitcast %ptrs : tensor<128x!tt.ptr<f32>> -> tensor<128x!tt.ptr<i32>>
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %cast = tt.bitcast %ptrs : tensor<128x!tt.ptr<f32>> -> tensor<128x!tt.ptr<f16>>
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %cast_fp8 = tt.bitcast %cast : tensor<128x!tt.ptr<f16>> -> tensor<128x!tt.ptr<f8E4M3FN>>
  %byte_base = tt.splat %arg1 : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
  %byte_ptrs = tt.addptr %byte_base, %range : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
  // Sub-byte and i8 pointers both use an effective one-byte element size.
  // expected-remark @below {{contiguity = [128], divisibility = [16], constancy = [1], constant_value = <none>}}
  %same_storage_width = tt.bitcast %byte_ptrs : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
  tt.return
}

// -----

tt.func @add(%arg0: tensor<128xi32> {tt.contiguity = 1 : i32, tt.divisibility = 4 : i32, tt.constancy = 2: i32}, %arg1: tensor<128xi32> {tt.contiguity = 4 : i32, tt.divisibility = 4 : i32, tt.constancy = 1: i32}) {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [1], constancy = [1], constant_value = <none>}}
  %2 = arith.addi %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 127}}
  %3 = arith.constant dense<127> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %4 = arith.addi %1, %3 : tensor<128xi32>
  // Contiguous + contiguous can overestimate divisibility if we only take
  // gcd(divisibility) (divisibility is defined on contiguity-group bases).
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = <none>}}
  %5 = arith.addi %0, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [1], constancy = [1], constant_value = <none>}}
  %odd = tt.make_range {end = 129 : i32, start = 1 : i32} : tensor<128xi32>
  // Contiguous + contiguous with different base parity yields an odd sequence.
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %even_plus_odd = arith.addi %0, %odd : tensor<128xi32>
  // Contiguous + contiguous with odd bases may yield an even sequence but we still conservatively estimate the
  // divisibility as 1.
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %odd_plus_odd = arith.addi %odd, %odd : tensor<128xi32>
  // Partial constant + contiguous. The resultant contiguity is smaller than the operands'.
  // [4, 4, 8, 8, 12, 12, ...] + [0, 1, 2, 3, 4, 5, ...]
  // expected-remark @below {{contiguity = [2], divisibility = [2], constancy = [1], constant_value = <none>}}
  %9 = arith.addi %0, %arg0 : tensor<128xi32>
  // Partial contiguous + partial contiguous can also overestimate divisibility.
  // [0, 1, 2, 3, 0, 1, 2, 3, ...] + [0, 1, 2, 3, 0, 1, 2, 3, ...]
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = <none>}}
  %11 = arith.addi %arg1, %arg1 : tensor<128xi32>
  // Partial constant + partial constant.
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [2], constant_value = <none>}}
  %12 = arith.addi %arg0, %arg0 : tensor<128xi32>
  tt.return
}

// -----

tt.func @addptr(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %cst1 = arith.constant 1 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %0 = tt.addptr %arg0, %cst1 : !tt.ptr<i1>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %1 = tt.addptr %arg1, %cst1 : !tt.ptr<i8>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = <none>}}
  %2 = tt.addptr %arg2, %cst1 : !tt.ptr<i16>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %3 = tt.addptr %arg3, %cst1 : !tt.ptr<i32>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>}}
  %4 = tt.addptr %arg4, %cst1 : !tt.ptr<i64>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = 4}}
  %cst4 = arith.constant 4 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %5 = tt.addptr %arg0, %cst4 : !tt.ptr<i1>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %6 = tt.addptr %arg1, %cst4 : !tt.ptr<i8>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>}}
  %7 = tt.addptr %arg2, %cst4 : !tt.ptr<i16>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
  %8 = tt.addptr %arg3, %cst4 : !tt.ptr<i32>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
  %9 = tt.addptr %arg4, %cst4 : !tt.ptr<i64>, i32
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>}}
  %11 = tt.expand_dims %10 {axis = 0: i32} : tensor<128xi32> -> tensor<1x128xi32>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [128, 1], constant_value = <none>}}
  %12 = tt.broadcast %11 : tensor<1x128xi32> -> tensor<128x128xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %13 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x128x!tt.ptr<i1>>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %14 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %15 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<128x128x!tt.ptr<i16>>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<128x128x!tt.ptr<i32>>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %17 = tt.splat %arg4 : !tt.ptr<i64> -> tensor<128x128x!tt.ptr<i64>>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [1, 16], constancy = [128, 1], constant_value = <none>}}
  %18 = tt.addptr %13, %12 : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [1, 16], constancy = [128, 1], constant_value = <none>}}
  %19 = tt.addptr %14, %12 : tensor<128x128x!tt.ptr<i8>>, tensor<128x128xi32>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [2, 16], constancy = [128, 1], constant_value = <none>}}
  %20 = tt.addptr %15, %12 : tensor<128x128x!tt.ptr<i16>>, tensor<128x128xi32>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [4, 16], constancy = [128, 1], constant_value = <none>}}
  %21 = tt.addptr %16, %12 : tensor<128x128x!tt.ptr<i32>>, tensor<128x128xi32>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [8, 16], constancy = [128, 1], constant_value = <none>}}
  %22 = tt.addptr %17, %12 : tensor<128x128x!tt.ptr<i64>>, tensor<128x128xi32>
  tt.return
}

// -----

tt.func @sub(%arg0: tensor<128xi32> {tt.contiguity = 1 : i32, tt.divisibility = 4 : i32, tt.constancy = 2: i32}, %arg1: tensor<128xi32> {tt.contiguity = 4 : i32, tt.divisibility = 4 : i32, tt.constancy = 1: i32}) {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [1], constancy = [1], constant_value = <none>}}
  %2 = arith.subi %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %3 = arith.subi %1, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 129}}
  %4 = arith.constant dense<129> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %5 = arith.subi %4, %1 : tensor<128xi32>
  // Result contiguity depends on RHS constancy for subi; ensure divisibility is
  // clamped when result contiguity is smaller than operand contiguity.
  // expected-remark @below {{contiguity = [128], divisibility = [16], constancy = [1], constant_value = <none>}}
  %rhs_range = tt.make_range {end = 144 : i32, start = 16 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
  %sub_clamp = arith.subi %0, %rhs_range : tensor<128xi32>
  // Both operands are contiguous, but with different contiguity groups. In this
  // case, we conservatively infer divisibility from parity: even - even is
  // divisible by 2.
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = <none>}}
  %even_even_diff_contig = arith.subi %0, %arg1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [4], divisibility = [1], constancy = [1], constant_value = <none>}}
  %arg1_plus_one = arith.addi %arg1, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %6 = arith.subi %0, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [1], constancy = [1], constant_value = <none>}}
  %odd = tt.make_range {end = 129 : i32, start = 1 : i32} : tensor<128xi32>
  // Contiguous - contiguous with different base parity yields an odd constant.
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %even_minus_odd = arith.subi %0, %odd : tensor<128xi32>
  // Contiguous - contiguous. With both only odd divisibilities.
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %odd_minus_arg1_plus_one = arith.subi %odd, %arg1_plus_one : tensor<128xi32>
  // Partial constant - contiguous. The resultant contiguity is smaller than the LHS'.
  // [0, 1, 2, 3, 4, 5, ...] - [4, 4, 8, 8, 12, 12, ...]
  // expected-remark @below {{contiguity = [2], divisibility = [2], constancy = [1], constant_value = <none>}}
  %7 = arith.subi %0, %arg0 : tensor<128xi32>
  // Same contiguity - shortcut optimization for divisibility
  // [4, 5, 6, 7, 4, 5, 6, 7, ...] - [4, 5, 6, 7, 4, 5, 6, 7, ...]
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %8 = arith.subi %arg1, %arg1 : tensor<128xi32>
  // Partial constant - partial constant.
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [2], constant_value = <none>}}
  %9 = arith.subi %arg0, %arg0 : tensor<128xi32>
  tt.return
}

// -----

tt.func @mul(%arg0: i64 {tt.divisibility = 16 : i32}) {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %2 = arith.muli %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %3 = arith.constant dense<128> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %4 = arith.muli %3, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [128], constant_value = 2}}
  %5 = arith.constant dense<2> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [256], constancy = [128], constant_value = 256}}
  %6 = arith.muli %4, %5 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 4611686018427387904}}
  %7 = arith.constant 4611686018427387904: i64
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = <none>}}
  %8 = arith.muli %arg0, %7 : i64
  tt.return
}

// -----

tt.func @div(%arg0: i32 {tt.divisibility = 16 : i32}) {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %2 = arith.divsi %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %3 = arith.divui %1, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %4 = arith.constant dense<64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [64], constant_value = <none>}}
  %5 = arith.divsi %0, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %6 = arith.divsi %4, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %7 = arith.divsi %4, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [128], constant_value = 66}}
  %8 = arith.constant dense<66> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [2], constant_value = <none>}}
  %9 = arith.divui %0, %8 : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [8192], constancy = [1], constant_value = <none>}}
  %10 = tt.make_range {end = 8320 : i32, start = 8192 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [64], constant_value = <none>}}
  %11 = arith.divsi %10, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = 2}}
  %12 = arith.constant 2 : i32
  // dividing a scalar by a power of two should give predictable divisibility
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>}}
  %13 = arith.divsi %arg0, %12 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [32], constancy = [1], constant_value = 32}}
  %14 = arith.constant 32 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %15 = arith.divsi %arg0, %14 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = 6}}
  %16 = arith.constant 6 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %17 = arith.divsi %arg0, %16 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [128], constant_value = 2}}
  %18 = arith.constant dense<2> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [2], constant_value = <none>}}
  %19 = arith.divsi %0, %18 : tensor<128xi32>
  tt.return
}


// -----

tt.func @rem() {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %2 = arith.remsi %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %3 = arith.remui %1, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %4 = arith.constant dense<64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [64], divisibility = [64], constancy = [1], constant_value = <none>}}
  %5 = arith.remsi %0, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %6 = arith.remsi %4, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [128], constant_value = 66}}
  %7 = arith.constant dense<66> : tensor<128xi32>
  // expected-remark @below {{contiguity = [2], divisibility = [2], constancy = [1], constant_value = <none>}}
  %8 = arith.remui %0, %7 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 192}}
  %9 = arith.constant dense<192> : tensor<128xi32>
  // expected-remark @below {{contiguity = [64], divisibility = [64], constancy = [1], constant_value = <none>}}
  %10 = arith.remsi %0, %9 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %11 = arith.remsi %9, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [32], constancy = [1], constant_value = <none>}}
  %12 = tt.make_range {end = 160 : i32, start = 32 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %13 = arith.remsi %0, %12 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %14 = arith.remsi %12, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [32], divisibility = [32], constancy = [1], constant_value = <none>}}
  %15 = arith.remsi %12, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %16 = arith.remsi %4, %12 : tensor<128xi32>
  %short_range = tt.make_range {end = 36 : i32, start = 32 : i32} : tensor<4xi32>
  %large_divisor = arith.constant dense<64> : tensor<4xi32>
  // Preserving the input group preserves divisibility larger than contiguity.
  // expected-remark @below {{contiguity = [4], divisibility = [32]}}
  %preserved_group = arith.remui %short_range, %large_divisor : tensor<4xi32>
  tt.return
}

// -----

tt.func @rem_partial_groups(%divisor: tensor<32xi32> {tt.divisibility = 2 : i32, tt.constancy = 16 : i32}) {
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  // Inferred unit contiguity exposes odd elements within the input group.
  // expected-remark @below {{contiguity = [1], divisibility = [1]}}
  %partial_rhs_signed = arith.remsi %range, %divisor : tensor<32xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1]}}
  %partial_rhs_unsigned = arith.remui %range, %divisor : tensor<32xi32>

  %eight = arith.constant dense<8> : tensor<32xi32>
  %two = arith.constant dense<2> : tensor<32xi32>
  %partial_range = arith.remui %range, %eight : tensor<32xi32>
  // The same issue occurs with a partially contiguous dividend.
  // expected-remark @below {{contiguity = [1], divisibility = [1]}}
  %partial_lhs_signed = arith.remsi %partial_range, %two : tensor<32xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1]}}
  %partial_lhs_unsigned = arith.remui %partial_range, %two : tensor<32xi32>
  tt.return
}

// -----

tt.func @rem_unit_contiguity(
    %lhs: tensor<8xi32> {tt.divisibility = 4 : i32},
    %rhs: tensor<8xi32> {tt.divisibility = 2 : i32},
    %scalar_lhs: i32 {tt.divisibility = 8 : i32},
    %scalar_rhs: i32 {tt.divisibility = 4 : i32}) {
  // Divisibility holds for every element, even when the divisor varies.
  // expected-remark @below {{contiguity = [1], divisibility = [2]}}
  %tensor_signed = arith.remsi %lhs, %rhs : tensor<8xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [2]}}
  %tensor_unsigned = arith.remui %lhs, %rhs : tensor<8xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4]}}
  %scalar_signed = arith.remsi %scalar_lhs, %scalar_rhs : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4]}}
  %scalar_unsigned = arith.remui %scalar_lhs, %scalar_rhs : i32
  tt.return
}

// -----

// NOTE: unsigned remui no longer keeps partial-group contiguity (upstream
// PR11546 soundness test pins unit below); %within_group collapses. Re-derive
// the D122462298 MI350 relaxation in rem_partial_groups-compatible form if
// paged-decode perf regresses.
tt.func @partial_div_rem_groups() {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %c64 = arith.constant dense<64> : tensor<128xi32>
  // Two contiguous groups of [0, ..., 63].
  // expected-remark @below {{contiguity = [64], divisibility = [64], constancy = [1], constant_value = <none>}}
  %token = arith.remui %n, %c64 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %c8 = arith.constant dense<8> : tensor<128xi32>
  // [0 x 8, 1 x 8, ..., 7 x 8] repeated twice.
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %group = arith.divui %token, %c8 : tensor<128xi32>
  // Eight [0, ..., 7] runs exist, but partial-group inference is disabled:
  // unit result (see NOTE above).
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %within_group = arith.remui %token, %c8 : tensor<128xi32>
  tt.return
}

// -----

// Signed div/rem need a sign proof for partial groups. These facts can
// describe local runs such as [-4, -3, -2, -1, 8, 9, 10, 11], for which
// truncation toward zero produces neither four equal quotients nor a
// four-element contiguous remainder run.
tt.func @signed_partial_div_rem_is_conservative(
    %lhs: tensor<8xi32> {tt.contiguity = 4 : i32, tt.divisibility = 4 : i32, tt.constancy = 1 : i32}) {
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [8], constant_value = 4}}
  %four = arith.constant dense<4> : tensor<8xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %quotient = arith.divsi %lhs, %four : tensor<8xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %remainder = arith.remsi %lhs, %four : tensor<8xi32>
  tt.return
}

// -----

tt.func @signed_bounded_tile() {
  %pid = tt.get_program_id x : i32
  %limit = arith.constant 16777216 : i32
  %valid = arith.cmpi ult, %pid, %limit : i32
  llvm.intr.assume %valid : i1
  %tile = arith.constant 128 : i32
  %base = arith.muli %pid, %tile : i32
  %bases = tt.splat %base : i32 -> tensor<128xi32>
  %lanes = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %indices = arith.addi %bases, %lanes : tensor<128xi32>
  %divisor = arith.constant dense<64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [64]}}
  %quotient = arith.divsi %indices, %divisor : tensor<128xi32>
  // expected-remark @below {{contiguity = [64], divisibility = [64], constancy = [1]}}
  %remainder = arith.remsi %indices, %divisor : tensor<128xi32>
  tt.return
}

// -----

tt.func @signed_tile_can_wrap() {
  %pid = tt.get_program_id x : i32
  %tile = arith.constant 128 : i32
  %base = arith.muli %pid, %tile : i32
  %bases = tt.splat %base : i32 -> tensor<128xi32>
  %lanes = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %indices = arith.addi %bases, %lanes : tensor<128xi32>
  %divisor = arith.constant dense<64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
  %quotient = arith.divsi %indices, %divisor : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
  %remainder = arith.remsi %indices, %divisor : tensor<128xi32>
  tt.return
}

// -----

tt.func @signed_assumption_follows_ssa(%base: i32 {tt.divisibility = 128 : i32}, %cond: i1) {
  %zero = arith.constant 0 : i32
  %limit = arith.constant 2147483520 : i32
  %positive = arith.cmpi sge, %base, %zero : i32
  %bounded = arith.cmpi sle, %base, %limit : i32
  %bases = tt.splat %base : i32 -> tensor<128xi32>
  %lanes = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %indices = arith.addi %bases, %lanes : tensor<128xi32>
  %divisor = arith.constant dense<64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
  %before = arith.divsi %indices, %divisor : tensor<128xi32>
  scf.if %cond {
    llvm.intr.assume %positive : i1
    llvm.intr.assume %bounded : i1
    // Earlier computations retain their original SSA facts.
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
    %inside = arith.divsi %indices, %divisor : tensor<128xi32>
    %local_bases = tt.splat %base : i32 -> tensor<128xi32>
    %local_indices = arith.addi %local_bases, %lanes : tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [64]}}
    %local = arith.divsi %local_indices, %divisor : tensor<128xi32>
  } else {
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
    %sibling = arith.divsi %indices, %divisor : tensor<128xi32>
  }
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
  %outside = arith.divsi %indices, %divisor : tensor<128xi32>
  llvm.intr.assume %positive : i1
  llvm.intr.assume %bounded : i1
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
  %after = arith.divsi %indices, %divisor : tensor<128xi32>
  %new_bases = tt.splat %base : i32 -> tensor<128xi32>
  %new_indices = arith.addi %new_bases, %lanes : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [64]}}
  %new = arith.divsi %new_indices, %divisor : tensor<128xi32>
  tt.return
}

// -----

tt.func @signed_partial_groups_with_proof() {
  %lanes = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %sixtyfour = arith.constant dense<64> : tensor<128xi32>
  %eight = arith.constant dense<8> : tensor<128xi32>
  %groups = arith.remui %lanes, %sixtyfour : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8]}}
  %quotient = arith.divsi %groups, %eight : tensor<128xi32>
  // expected-remark @below {{contiguity = [8], divisibility = [8], constancy = [1]}}
  %remainder = arith.remsi %groups, %eight : tensor<128xi32>
  tt.return
}

// -----

tt.func @signed_bounded_loop() {
  %zero = arith.constant 0 : i32
  %end = arith.constant 1024 : i32
  %step = arith.constant 128 : i32
  %lanes = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %divisor = arith.constant dense<64> : tensor<128xi32>
  scf.for %iv = %zero to %end step %step : i32 {
    %bases = tt.splat %iv : i32 -> tensor<128xi32>
    %indices = arith.addi %bases, %lanes : tensor<128xi32>
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [64]}}
    %quotient = arith.divsi %indices, %divisor : tensor<128xi32>
    // expected-remark @below {{contiguity = [64], divisibility = [64], constancy = [1]}}
    %remainder = arith.remsi %indices, %divisor : tensor<128xi32>
  }
  tt.return
}

// -----

tt.func @signed_wrapping_add() {
  %base = arith.constant dense<2147483584> : tensor<128xi32>
  %lanes = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %indices = arith.addi %base, %lanes : tensor<128xi32>
  %divisor = arith.constant dense<64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
  %quotient = arith.divsi %indices, %divisor : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1]}}
  %remainder = arith.remsi %indices, %divisor : tensor<128xi32>
  tt.return
}

// -----

// Full contiguous inputs may start at a negative divisor multiple.
tt.func @signed_full_div_rem_is_conservative(
    %lhs: tensor<128xi32> {tt.contiguity = 128 : i32, tt.divisibility = 64 : i32}) {
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %positive = arith.constant dense<64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = -64}}
  %negative = arith.constant dense<-64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %quotient_positive = arith.divsi %lhs, %positive : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %quotient_negative = arith.divsi %lhs, %negative : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %remainder_positive = arith.remsi %lhs, %positive : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %remainder_negative = arith.remsi %lhs, %negative : tensor<128xi32>
  tt.return
}

// -----

tt.func @signed_negative_range_div_rem() {
  // expected-remark @below {{contiguity = [128], divisibility = [64], constancy = [1], constant_value = <none>}}
  %range = tt.make_range {start = -64 : i32, end = 64 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = -64}}
  %divisor = arith.constant dense<-64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %quotient = arith.divsi %range, %divisor : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %remainder = arith.remsi %range, %divisor : tensor<128xi32>
  tt.return
}

// -----

tt.func @signed_remainder_preserves_element_divisibility(
    %lhs: tensor<128xi32> {tt.divisibility = 16 : i32}) {
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = -64}}
  %divisor = arith.constant dense<-64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
  %remainder = arith.remsi %lhs, %divisor : tensor<128xi32>
  tt.return
}

// -----

// A divisor that changes within the dimension does not preserve group bases.
tt.func @remainder_partial_divisor_constancy(
    %rhs: tensor<128xi32> {tt.constancy = 2 : i32, tt.divisibility = 4 : i32}) {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %signed = arith.remsi %range, %rhs : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %unsigned = arith.remui %range, %rhs : tensor<128xi32>
  tt.return
}

// -----

tt.func @expanddims() {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [128], constant_value = 2}}
  %1 = arith.constant dense<2> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = <none>}}
  %2 = arith.muli %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [2, 2], constancy = [1, 1], constant_value = <none>}}
  %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  tt.return
}

// -----

tt.func @reshape(%arg0: tensor<8xi32> {tt.contiguity = 1 : i32, tt.divisibility = 8 : i32, tt.constancy = 4 : i32}) {
  // expected-remark @below {{contiguity = [4], divisibility = [4], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  // expected-remark @below {{contiguity = [1, 1, 4], divisibility = [1, 1, 4], constancy = [1, 1, 1], constant_value = <none>}}
  %1 = tt.reshape %0 : tensor<4xi32> -> tensor<1x1x4xi32>
  // expected-remark @below {{contiguity = [16], divisibility = [4], constancy = [1], constant_value = <none>}}
  %2 = tt.make_range {end = 20 : i32, start = 4 : i32} : tensor<16xi32>
  // expected-remark @below {{contiguity = [1, 4], divisibility = [1, 4], constancy = [1, 1], constant_value = <none>}}
  %3 = tt.reshape %2 : tensor<16xi32> -> tensor<4x4xi32>
  // expected-remark @below {{contiguity = [4], divisibility = [4], constancy = [1], constant_value = <none>}}
  %4 = tt.reshape %3 : tensor<4x4xi32> -> tensor<16xi32>
  // expected-remark @below {{contiguity = [4, 1], divisibility = [4, 1], constancy = [1, 1], constant_value = <none>}}
  %5 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<4x4xi32> -> tensor<4x4xi32>
  // expected-remark @below {{contiguity = [1, 2, 1], divisibility = [1, 2, 1], constancy = [1, 1, 1], constant_value = <none>}}
  %6 = tt.reshape %5 : tensor<4x4xi32> -> tensor<2x2x4xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %7 = tt.reshape %5 : tensor<4x4xi32> -> tensor<16xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [2, 2], constant_value = <none>}}
  %8 = tt.reshape %arg0 : tensor<8xi32> -> tensor<4x2xi32>
  // expected-remark @below {{contiguity = [8], divisibility = [16], constancy = [1], constant_value = <none>}}
  %9 = tt.make_range {end = 24 : i32, start = 16 : i32} : tensor<8xi32>
  // expected-remark @below {{contiguity = [1, 8], divisibility = [1, 16], constancy = [1, 1], constant_value = <none>}}
  %10 = tt.reshape %9 : tensor<8xi32> -> tensor<1x8xi32>
  // expected-remark @below {{contiguity = [4], divisibility = [1], constancy = [1], constant_value = <none>}}
  %11 = tt.make_range {end = 5 : i32, start = 1 : i32} : tensor<4xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [4], constant_value = 2}}
  %12 = arith.constant dense<2> : tensor<4xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = <none>}}
  %13 = arith.muli %11, %12 : tensor<4xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [2, 2], constancy = [1, 1], constant_value = <none>}}
  %14 = tt.reshape %13 : tensor<4xi32> -> tensor<1x4xi32>
  tt.return
}

// -----

tt.func @reshape_refined_piece_merge(
    %arg0: tensor<4x4xi32> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<[1, 1]> : tensor<2xi32>, tt.constancy = dense<[2, 4]> : tensor<2xi32>},
    %arg1: tensor<2x2x2xi32> {tt.contiguity = dense<[1, 1, 2]> : tensor<3xi32>, tt.divisibility = dense<[1, 1, 1]> : tensor<3xi32>, tt.constancy = dense<[2, 2, 1]> : tensor<3xi32>}) {
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %0 = tt.reshape %arg0 : tensor<4x4xi32> -> tensor<16xi32>
  // expected-remark @below {{contiguity = [1, 2], divisibility = [1, 1], constancy = [4, 1], constant_value = <none>}}
  %1 = tt.reshape %arg1 : tensor<2x2x2xi32> -> tensor<4x2xi32>
  tt.return
}

// -----

tt.func @reshape_global_divisibility(
    %arg0: tensor<2x2xi32> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<[8, 2]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
    %arg1: tensor<2x2xi32> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<[2, 8]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
    %arg2: tensor<2x1x2xi32> {tt.contiguity = dense<[1, 1, 1]> : tensor<3xi32>, tt.divisibility = dense<[2, 32, 4]> : tensor<3xi32>, tt.constancy = dense<[1, 1, 1]> : tensor<3xi32>},
    %arg3: tensor<4x4xi32> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<[8, 2]> : tensor<2xi32>, tt.constancy = dense<[2, 4]> : tensor<2xi32>},
    %arg4: tensor<2x2x!tt.ptr<i32>> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<[4, 8]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
  // Divisibility from a unit-contiguity source axis applies to every output
  // axis, including axes whose flat-index bits do not overlap the source axis.
  // expected-remark @below {{contiguity = [1, 1, 1], divisibility = [8, 8, 8], constancy = [1, 1, 1], constant_value = <none>}}
  %0 = tt.reshape %arg0 : tensor<2x2xi32> -> tensor<1x2x2xi32>
  // expected-remark @below {{contiguity = [1, 1, 1], divisibility = [8, 8, 8], constancy = [1, 1, 1], constant_value = <none>}}
  %1 = tt.reshape %arg1 : tensor<2x2xi32> -> tensor<1x2x2xi32>
  // The strongest global divisor may come from a singleton source axis.
  // expected-remark @below {{contiguity = [1], divisibility = [32], constancy = [1], constant_value = <none>}}
  %2 = tt.reshape %arg2 : tensor<2x1x2xi32> -> tensor<4xi32>
  // Global divisibility is independent of merging constant source factors.
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [8], constant_value = <none>}}
  %3 = tt.reshape %arg3 : tensor<4x4xi32> -> tensor<16xi32>
  // Pointer contiguity is measured in elements while divisibility is measured
  // in bytes, so global divisibility can coexist with contiguity greater than
  // one. Reshape must preserve the global divisor even when splitting that
  // contiguous axis.
  // expected-remark @below {{contiguity = [2, 1], divisibility = [8, 4], constancy = [1, 1], constant_value = <none>}}
  %4 = tt.reshape %arg4 : tensor<2x2x!tt.ptr<i32>> -> tensor<4x1x!tt.ptr<i32>>
  tt.return
}

// -----

tt.func @global_divisibility_is_axis_independent(
    %arg0: tensor<2x2xi32> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<[8, 2]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
    %arg1: tensor<2x2xi32> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<[2, 8]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
    %cond: i1) {
  // Unit contiguity makes every element a group base, so the strongest such
  // divisibility applies to every axis through shape and arithmetic ops.
  // expected-remark @below {{contiguity = [1, 1, 1], divisibility = [8, 8, 8], constancy = [1, 1, 1], constant_value = <none>}}
  %0 = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<2x2xi32> -> tensor<1x2x2xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [1, 1], constant_value = <none>}}
  %1 = tt.trans %arg0 {order = array<i32: 1, 0>} : tensor<2x2xi32> -> tensor<2x2xi32>
  // expected-remark @below {{contiguity = [1, 1, 1], divisibility = [8, 8, 8], constancy = [4, 1, 1], constant_value = <none>}}
  %2 = tt.broadcast %0 : tensor<1x2x2xi32> -> tensor<4x2x2xi32>
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [1, 1], constant_value = <none>}}
  %3 = arith.addi %arg0, %zero : tensor<2x2xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [1, 1], constant_value = <none>}}
  %4 = scf.if %cond -> tensor<2x2xi32> {
    scf.yield %arg0 : tensor<2x2xi32>
  } else {
    scf.yield %arg1 : tensor<2x2xi32>
  }
  tt.return
}

// -----

tt.func @broadcast() {
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %0 = arith.constant dense<64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 1], constant_value = 64}}
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 128], constant_value = 64}}
  %2 = tt.broadcast %1 : tensor<128x1xi32> -> tensor<128x128xi32>
  tt.return
}

// -----

tt.func @splat(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @cmp_all_contiguous() {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %1 = arith.constant dense<0> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %2 = arith.cmpi eq, %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %3 = arith.cmpi ne, %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %4 = arith.cmpi slt, %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %5 = arith.cmpi sle, %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %6 = arith.cmpi sge, %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %7 = arith.cmpi sgt, %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %8 = arith.cmpi eq, %1, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %9 = arith.cmpi ne, %1, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %10 = arith.cmpi slt, %1, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %11 = arith.cmpi sle, %1, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %12 = arith.cmpi sge, %1, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %13 = arith.cmpi sgt, %1, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %14 = arith.constant dense<8> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %15 = arith.cmpi sgt, %14, %0 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = -1}}
  %16 = arith.cmpi sgt, %14, %1 : tensor<128xi32>
  tt.return
}

tt.func @cmp_partial_contiguous() {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %1 = arith.constant dense<8> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [32], constancy = [128], constant_value = 32}}
  %3 = arith.constant dense<32> : tensor<128xi32>
  // expected-remark @below {{contiguity = [32], divisibility = [32], constancy = [1], constant_value = <none>}}
  %4 = arith.remsi %0, %3 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %5 = arith.cmpi eq, %4, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %6 = arith.cmpi ne, %4, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %7 = arith.cmpi slt, %4, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %8 = arith.cmpi sle, %4, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %9 = arith.cmpi sge, %4, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %10 = arith.cmpi sgt, %4, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %11 = arith.cmpi eq, %1, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %12 = arith.cmpi ne, %1, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %13 = arith.cmpi slt, %1, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %14 = arith.cmpi sle, %1, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %15 = arith.cmpi sge, %1, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %16 = arith.cmpi sgt, %1, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [128], constant_value = 48}}
  %17 = arith.constant dense<48> : tensor<128xi32>
  // expected-remark @below {{contiguity = [16], divisibility = [16], constancy = [1], constant_value = <none>}}
  %18 = arith.remsi %0, %17 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %19 = arith.cmpi eq, %18, %3 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %20 = arith.cmpi ne, %18, %3 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %21 = arith.cmpi slt, %18, %3 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %22 = arith.cmpi sle, %18, %3 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %23 = arith.cmpi sge, %18, %3 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %24 = arith.cmpi sgt, %18, %3 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %25 = arith.cmpi eq, %3, %18 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %26 = arith.cmpi ne, %3, %18 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %27 = arith.cmpi slt, %3, %18 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %28 = arith.cmpi sle, %3, %18 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %29 = arith.cmpi sge, %3, %18 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none}}
  %30 = arith.cmpi sgt, %3, %18 : tensor<128xi32>
  tt.return
}

// -----

tt.func @logic() {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %1 = arith.constant dense<64> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [64], constant_value = <none>}}
  %2 = arith.divsi %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %3 = arith.constant dense<8> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %4 = arith.divsi %0, %3 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %5 = arith.andi %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %6 = arith.ori %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %7 = arith.xori %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %8 = arith.andi %2, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %9 = arith.ori %2, %4 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %10 = arith.xori %2, %4 : tensor<128xi32>
  tt.return
}

// -----

tt.func @select(%arg0 : i1, %arg1 : tensor<4xi1>) {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %1 = arith.constant dense<0> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %2 = arith.cmpi eq, %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %3 = arith.cmpi slt, %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %4 = arith.constant 0 : i1
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %7 = tt.splat %4 : i1 -> tensor<128xi1>
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %5 = arith.select %4, %3, %7 : tensor<128xi1>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %8 = arith.select %7, %3, %2 : tensor<128xi1>, tensor<128xi1>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %9 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 1], constant_value = <none>}}
  %10 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %11 = arith.select %arg0, %9, %10 : tensor<128x1xi1>
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [4], constant_value = 4}}
  %cst = arith.constant dense<4> : tensor<4xi32>
  // expected-remark @below {{contiguity = [4], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %13 = arith.muli %12, %cst : tensor<4xi32>
  // expected-remark @below {{contiguity = [4], divisibility = [16], constancy = [1], constant_value = <none>}}
  %14 = tt.make_range {end = 20 : i32, start = 16 : i32} : tensor<4xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %15 = arith.select %arg1, %12, %13 : tensor<4xi1>, tensor<4xi32>
  tt.return
}

// -----

tt.func @shift(%arg0: i32 {tt.divisibility = 4 : i32}) {
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [128], constant_value = <none>}}
  %s = tt.splat %arg0 : i32 -> tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %1 = arith.constant dense<8> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [128], constant_value = 4}}
  %2 = arith.constant dense<4> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [256], constancy = [1], constant_value = <none>}}
  %3 = arith.shli %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %4 = arith.shrsi %0, %2 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %5 = arith.shli %1, %2 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = <none>}}
  %6 = arith.shli %1, %s : tensor<128xi32>
  // Unknown shifts make every range element a group base, including odd ones.
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %unknown_left = arith.shli %0, %s : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %7 = arith.shrsi %0, %s : tensor<128xi32>
  tt.return
}

// -----

tt.func @max_min() {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [64], constancy = [1], constant_value = <none>}}
  %1 = tt.make_range {end = 192 : i32, start = 64 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [64], constancy = [1], constant_value = <none>}}
  %2 = arith.maxsi %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [64], constancy = [1], constant_value = <none>}}
  %3 = arith.minsi %0, %1 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %4 = arith.constant dense<8> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [128], constant_value = 4}}
  %5 = arith.constant dense<4> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %6 = arith.maxsi %4, %5 : tensor<128xi32>
  tt.return
}

// -----

// A complicated example with different contiguity and divisibility in lhs and rhs.
// To simplify construction of the test we just pass attributes from the arguments
tt.func @contiguity_dependent_divisibility(%arg0: tensor<8xi32> {tt.contiguity = 8 : i32, tt.divisibility = 4 : i32, tt.constancy = 1 : i32}, %arg1: tensor<8xi32> {tt.contiguity = 2 : i32, tt.divisibility = 8 : i32, tt.constancy = 1 : i32}) {
  // expected-remark @below {{contiguity = [2], divisibility = [2], constancy = [1], constant_value = <none>}}
  %0 = arith.maxsi %arg0, %arg1 : tensor<8xi32>
  // expected-remark @below {{contiguity = [2], divisibility = [2], constancy = [1], constant_value = <none>}}
  %1 = arith.minsi %arg0, %arg1 : tensor<8xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %2 = arith.constant 0 : i1
  // expected-remark @below {{contiguity = [2], divisibility = [2], constancy = [1], constant_value = <none>}}
  %3 = arith.select %2, %0, %1 : tensor<8xi32>
  tt.return
}

// -----

tt.func @if(%i1 : i1) {
  // expected-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 32], constant_value = 64}}
  %cst_64 = arith.constant dense<64> : tensor<128x32xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = 1}}
  %cst_1 = arith.constant dense<1> : tensor<128x32xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 32], constant_value = 64}}
  %a = arith.muli %cst_64, %cst_1 : tensor<128x32xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = <none>}}
  %ret = scf.if %i1 -> tensor<128x32xi32> {
    scf.yield %a : tensor<128x32xi32>
  } else {
    scf.yield %cst_1 : tensor<128x32xi32>
  }
  tt.return
}

// -----

tt.func @for() {
  // expected-remark @below {{contiguity = [1, 1], divisibility = [4611686018427387904, 4611686018427387904], constancy = [128, 32], constant_value = 0}}
  %a_init = arith.constant dense<0> : tensor<128x32xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = 1}}
  %b_init = arith.constant dense<1> : tensor<128x32xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [4, 4], constancy = [128, 32], constant_value = 4}}
  %c_init = arith.constant dense<4> : tensor<128x32xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [128], constancy = [1], constant_value = 128}}
  %ub = arith.constant 128 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %lb = arith.constant 0 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = 16}}
  %step = arith.constant 16 : i32
  %a, %b, %c = scf.for %iv = %lb to %ub step %step iter_args(%a = %a_init, %b = %b_init, %c = %c_init) -> (tensor<128x32xi32>, tensor<128x32xi32>, tensor<128x32xi32>) : i32 {
    // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
    %t = arith.addi %iv, %lb : i32
    // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = <none>}}
    // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = <none>}}
    // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [4, 4], constancy = [128, 32], constant_value = 4}}
    scf.yield %b, %a, %c : tensor<128x32xi32>, tensor<128x32xi32>, tensor<128x32xi32>
  }
  tt.return
}

// -----

tt.func @for_dynamic(%lb: i32 {tt.divisibility = 16 : i32}, %step: i32 {tt.divisibility = 8 : i32}, %ub: i32) {
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0 = arith.constant 0 : i32
  scf.for %iv = %lb to %ub step %step : i32 {
    // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>}}
    %t = arith.addi %iv, %c0 : i32
  }
  tt.return
}

// -----

tt.func @for_if(%i1: i1, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0_i32 = arith.constant 0 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %c1_i32 = arith.constant 1 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = 10}}
  %c10_i32 = arith.constant 10 : i32
  // expected-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 64], constant_value = 64}}
  %cst = arith.constant dense<64> : tensor<128x64xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
  %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
  %2 = scf.for %arg9 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg1 = %1) -> (tensor<128x64x!tt.ptr<f16>>): i32 {
    // TODO-remark(this remark is wrong, needs to be fixed) @below {{scf.if}}
    // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
    %3 = scf.if %i1 -> (tensor<128x64x!tt.ptr<f16>>) {
      scf.yield %arg1 : tensor<128x64x!tt.ptr<f16>>
    } else {
      scf.yield %arg1 : tensor<128x64x!tt.ptr<f16>>
    }
    // TODO-remark(this remark is wrong, needs to be fixed) @below {{tt.addptr}}
    // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
    %4 = tt.addptr %3, %cst : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    // TODO-remark(this remark is wrong, needs to be fixed) @below {{scf.for}}
    // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
    scf.yield %1 : tensor<128x64x!tt.ptr<f16>>
  }
  tt.return
}

// -----

tt.func @for_if_for(%i1: i1, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 8 : i32}) {
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0_i32 = arith.constant 0 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %c1_i32 = arith.constant 1 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = 10}}
  %c10_i32 = arith.constant 10 : i32
  // expected-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 64], constant_value = 64}}
  %cst = arith.constant dense<64> : tensor<128x64xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
  %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [128, 64], constant_value = <none>}}
  %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{scf.for}}
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [128, 64], constant_value = <none>}}
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{scf.if}}
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [128, 64], constant_value = <none>}}
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{tt.addptr}}
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [128, 64], constant_value = <none>}}
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{scf.for}}
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
  %3 = scf.for %arg9 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg2 = %1) -> (tensor<128x64x!tt.ptr<f16>>) : i32 {
    %4 = scf.if %i1 -> (tensor<128x64x!tt.ptr<f16>>) {
      %5 = scf.for %arg10 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg3 = %2) -> (tensor<128x64x!tt.ptr<f16>>) : i32 {
        scf.yield %arg3 : tensor<128x64x!tt.ptr<f16>>
      }
      scf.yield %5 : tensor<128x64x!tt.ptr<f16>>
    } else {
      scf.yield %arg2 : tensor<128x64x!tt.ptr<f16>>
    }
    %6 = tt.addptr %4, %cst : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    scf.yield %1 : tensor<128x64x!tt.ptr<f16>>
  }
  tt.return
}

// -----

tt.func @permute_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 128], constant_value = -1}}
  %cst = arith.constant dense<true> : tensor<128x128xi1>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [128, 1], divisibility = [1073741824, 1], constancy = [1, 1], constant_value = <none>}}
  %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>}}
  %3 = tt.splat %arg1 : i32 -> tensor<128x1xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %4 = arith.muli %2, %3 : tensor<128x1xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>}}
  %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %6 = tt.addptr %5, %4 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>}}
  %7 = tt.expand_dims %1 {axis = 0 : i32}: tensor<128xi32> -> tensor<1x128xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 128], constant_value = <none>}}
  %8 = tt.broadcast %6 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [128, 1], constant_value = <none>}}
  %9 = tt.broadcast %7 : tensor<1x128xi32> -> tensor<128x128xi32>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [4, 16], constancy = [1, 1], constant_value = <none>}}
  %10 = tt.addptr %8, %9 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
  // expected-remark @below {{contiguity = [128, 1], divisibility = [1073741824, 1], constancy = [1, 1], constant_value = <none>}}
  %11 = tt.expand_dims %0 {axis = 1 : i32}: tensor<128xi32> -> tensor<128x1xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>}}
  %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
  // expected-remark @below {{contiguity = [128, 1], divisibility = [16, 4], constancy = [1, 1], constant_value = <none>}}
  %13 = tt.addptr %12, %11 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
  // expected-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>}}
  %14 = tt.expand_dims %1 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 128], constant_value = <none>}}
  %15 = tt.splat %arg3 : i32 -> tensor<1x128xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %16 = arith.muli %14, %15 : tensor<1x128xi32>
  // expected-remark @below {{contiguity = [128, 1], divisibility = [16, 4], constancy = [1, 128], constant_value = <none>}}
  %17 = tt.broadcast %13 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>}}
  %18 = tt.broadcast %16 : tensor<1x128xi32> -> tensor<128x128xi32>
  // expected-remark @below {{contiguity = [128, 1], divisibility = [16, 4], constancy = [1, 1], constant_value = <none>}}
  %19 = tt.addptr %17, %18 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %20 = tt.load %10, %cst, %cst_0 : tensor<128x128x!tt.ptr<f32>>
  tt.store %19, %20, %cst : tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @load_constancy(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 1 : i32}) {
  // expected-remark @below {{divisibility = [16]}}
  %sixteen = arith.constant dense<16> : tensor<1024xi32>
  // expected-remark @below {{divisibility = [8]}}
  %eight = arith.constant dense<8> : tensor<1024xi32>
  // expected-remark @below {{contiguity = [1024], divisibility = [1073741824], constancy = [1]}}
  %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  // expected-remark @below {{constancy = [16]}}
  %2 = arith.divsi %1, %sixteen : tensor<1024xi32>
  // expected-remark @below {{constancy = [1024]}}
  %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  // expected-remark @below {{constancy = [1024]}}
  %4 = tt.splat %arg1 : i32 -> tensor<1024xi32>
  // expected-remark @below {{constancy = [8]}}
  %5 = arith.divsi %1, %eight : tensor<1024xi32>
  // expected-remark @below {{constancy = [8]}}
  %6 = arith.cmpi slt, %5, %4 : tensor<1024xi32>
  // expected-remark @below {{constancy = [16]}}
  %7 = tt.addptr %3, %2 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  // expected-remark @below {{constancy = [16]}}
  %8 = tt.load %7 : tensor<1024x!tt.ptr<i32>>
  // expected-remark @below {{constancy = [8]}}
  %9 = tt.load %7, %6 : tensor<1024x!tt.ptr<i32>>
  // expected-remark @below {{constancy = [1]}}
  %10 = tt.load %7, %6, %1 : tensor<1024x!tt.ptr<i32>>
  // expected-remark @below {{constancy = [8]}}
  %11 = tt.load %7, %6, %5 : tensor<1024x!tt.ptr<i32>>
  tt.return
}

// -----

// This is a tiny test for verifying StoreOp-related alignment, It simply store a constant to a buffer.
tt.func @store_constant_align(%addr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32 {tt.divisibility = 16 : i32}) {
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %pid = tt.get_program_id x : i32
  // expected-remark @below {{contiguity = [1], divisibility = [128], constancy = [1], constant_value = 128}}
  %c128_i32 = arith.constant 128 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [128], constancy = [1], constant_value = <none>}}
  %1 = arith.muli %pid, %c128_i32 : i32
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
 // expected-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = <none>}}
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
 // expected-remark @below {{contiguity = [128], divisibility = [128], constancy = [1], constant_value = <none>}}
  %4 = arith.addi %3, %2 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [128], constant_value = <none>}}
  %5 = tt.splat %addr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  // expected-remark @below {{contiguity = [128], divisibility = [16], constancy = [1], constant_value = <none>}}
  %6 = tt.addptr %5, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [128], constant_value = <none>}}
  %9 = tt.splat %n : i32 -> tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %mask = arith.cmpi slt, %4, %9 : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %cst = arith.constant dense<0.0> : tensor<128xf32>
  tt.store %5, %cst, %mask : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----

// This IR is dumped from vecadd test.
// Note, the hint {tt.divisibility = 16 : i32} for %n_elements affects the alignment of mask.
tt.func @vecadd_mask_align_16(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32 {tt.divisibility = 16 : i32}) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : i32 -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %9 = tt.splat %n_elements : i32 -> tensor<64xi32>
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{arith.cmpi slt, %{{.*}} => contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %mask = arith.cmpi slt, %4, %9 : tensor<64xi32>
  %11 = tt.load %6, %mask : tensor<64x!tt.ptr<f32>>
  %12 = tt.load %8, %mask : tensor<64x!tt.ptr<f32>>
  %13 = arith.addf %11, %12 : tensor<64xf32>
  %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{tt.addptr %{{.*}} => contiguity = [64], divisibility = [16], constancy = [1], constant_value = <none>}}
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  tt.store %15, %13, %mask : tensor<64x!tt.ptr<f32>>
  tt.return
}

// -----

// This IR is dumped from vecadd test.
// Note, there is no divisibility hint for %n_elements, Triton should assume its divisibility to be 1 by default.
tt.func @vecadd_mask_align_1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : i32 -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %9 = tt.splat %n_elements : i32 -> tensor<64xi32>
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{arith.cmpi slt, %{{.*}} => contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %10 = arith.cmpi slt, %4, %9 : tensor<64xi32>
  %11 = tt.load %6, %10 : tensor<64x!tt.ptr<f32>>
  %12 = tt.load %8, %10 : tensor<64x!tt.ptr<f32>>
  %13 = arith.addf %11, %12 : tensor<64xf32>
  %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  tt.store %15, %13, %10 : tensor<64x!tt.ptr<f32>>
  tt.return
}

// -----

module {

// We don't use function cloning here, so the alignment info is the gcd of all call sites.
tt.func @addptr_hints(%arg0: !tt.ptr<i32>) {
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %cst1 = arith.constant 1 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %1 = tt.addptr %arg0, %cst1 : !tt.ptr<i32>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = 4}}
  %cst4 = arith.constant 4 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %2 = tt.addptr %arg0, %cst4 : !tt.ptr<i32>, i32
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = 16}}
  %cst16 = arith.constant 16 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %3 = tt.addptr %arg0, %cst4 : !tt.ptr<i32>, i32
  tt.return
}

tt.func @kernel_div16(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  tt.call @addptr_hints(%arg0) : (!tt.ptr<i32>) -> ()
  tt.return
}

tt.func @kernel_div8(%arg0: !tt.ptr<i32> {tt.divisibility = 8 : i32}) {
  tt.call @addptr_hints(%arg0) : (!tt.ptr<i32>) -> ()
  tt.return
}

tt.func @kernel_div4(%arg0: !tt.ptr<i32> {tt.divisibility = 4 : i32}) {
  tt.call @addptr_hints(%arg0) : (!tt.ptr<i32>) -> ()
  tt.return
}

}

// -----

module {

// We don't use function cloning here, so the alignment info is the gcd of all call sites.
tt.func @mul(%arg0: i32) {
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %cst1 = arith.constant 1 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %1 = arith.muli %arg0, %cst1 : i32
  tt.return
}

tt.func @bar(%arg0: i32) {
  tt.call @mul(%arg0) : (i32) -> ()
  tt.return
}

tt.func @foo(%arg0: i32) {
  tt.call @mul(%arg0) : (i32) -> ()
  tt.return
}

tt.func @call_graph(%arg0: i32) {
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = 12}}
  %cst12 = arith.constant 12 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %0 = arith.muli %arg0, %cst12 : i32
  tt.call @foo(%0) : (i32) -> ()
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = 8}}
  %cst8 = arith.constant 8 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>}}
  %1 = arith.muli %arg0, %cst8 : i32
  tt.call @bar(%1) : (i32) -> ()
  tt.return
}

}

// -----

tt.func public @chained_for(%8: tensor<128x64x!tt.ptr<bf16>> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>}) {
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %cst = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = 16}}
  %c16_i32 = arith.constant 16 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %c1_i32 = arith.constant 1 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0_i32 = arith.constant 0 : i32
  // expected-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 64], constant_value = 64}}
  %cst_0 = arith.constant dense<64> : tensor<128x64xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %9 = scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg8 = %8) -> (tensor<128x64x!tt.ptr<bf16>>)  : i32 {
    %11 = tt.addptr %arg8, %cst_0 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    scf.yield %11 : tensor<128x64x!tt.ptr<bf16>>
  }
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  // TODO-remark(this remark is wrong, needs to be fixed) @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %10 = scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg8 = %9) -> (tensor<128x64x!tt.ptr<bf16>>)  : i32 {
    tt.store %arg8, %cst : tensor<128x64x!tt.ptr<bf16>>
    %11 = tt.addptr %arg8, %cst_0 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    scf.yield %11 : tensor<128x64x!tt.ptr<bf16>>
  }
  tt.return
}

// -----

module {
  tt.func @int_min_does_not_underflow_in_analysis() -> i64 {
    // expected-remark @below {{divisibility = [4611686018427387904]}}
    %int_min = arith.constant -9223372036854775808 : i64
    tt.return %int_min : i64
  }
}

// -----

tt.func @test_warp_specialize_propagation(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}) {
  ttg.warp_specialize(%arg0, %arg1)
  default {
    // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
    tt.addptr %arg0, %arg1 : !tt.ptr<f16>, i32
    ttg.warp_yield
  }
  partition0(%arg2: !tt.ptr<f16>, %arg3: i32) num_warps(1) {
    // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
    tt.addptr %arg2, %arg3 : !tt.ptr<f16>, i32
    ttg.warp_return
  }
  partition1(%arg2: !tt.ptr<f16>, %arg3: i32) num_warps(1) {
    // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
    tt.addptr %arg2, %arg3 : !tt.ptr<f16>, i32
    ttg.warp_return
  } : (!tt.ptr<f16>, i32) -> ()
  tt.return
}

// -----

tt.func @if_into_for_init(%i1 : i1) {
  %c0 = arith.constant 0 : i32
  %cst_64 = arith.constant 64 : i32
  %cst128 = arith.constant 128 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = <none>}}
  %ret = scf.if %i1 -> i32 {
    scf.yield %cst_64 : i32
  } else {
    scf.yield %cst128 : i32
  }
  scf.for %i = %ret to %cst128 step %cst_64 : i32 {
    // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = <none>}}
    %t = arith.addi %i, %c0 : i32
  }
  tt.return
}

// -----

tt.func @if_into_for_step(%i1 : i1) {
  %c0 = arith.constant 0 : i32
  %cst_64 = arith.constant 64 : i32
  %cst128 = arith.constant 128 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = <none>}}
  %ret = scf.if %i1 -> i32 {
    scf.yield %cst_64 : i32
  } else {
    scf.yield %cst128 : i32
  }
  scf.for %i = %c0 to %cst128 step %ret : i32 {
    // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = <none>}}
    %t = arith.addi %i, %c0 : i32
  }
  tt.return
}

// -----

tt.func @op_annotation(%i32 : i32) {
  %c0 = arith.constant 0 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4096], constancy = [1], constant_value = <none>}}
  %ret0 = arith.addi %c0, %i32 { tt.divisibility = 4096 : i32 } : i32
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1024, 1024], constancy = [128, 64], constant_value = <none>}}
  %ret1 = tt.splat %ret0 { tt.divisibility = dense<[1024, 1024]> : tensor<2xi32> } : i32 -> tensor<128x64xi32>
  tt.return
}

// -----

tt.func public @trans_4d_tensor_kernel(%arg0: tensor<32x32x32x32xi32> {tt.contiguity = dense<[32, 1, 1, 1]> : tensor<4xi32>, tt.divisibility = dense<[16, 1, 1, 1]> : tensor<4xi32>}) attributes {noinline = false} {
  // expected-remark @below {{contiguity = [1, 1, 1, 32], divisibility = [1, 1, 1, 16], constancy = [1, 1, 1, 1], constant_value = <none>}}
  %101 = tt.trans %arg0 {order = array<i32: 3, 2, 1, 0>} : tensor<32x32x32x32xi32> -> tensor<32x32x32x32xi32>
  // expected-remark @below {{contiguity = [1, 32, 1, 1], divisibility = [1, 16, 1, 1], constancy = [1, 1, 1, 1], constant_value = <none>}}
  %102 = tt.trans %arg0 {order = array<i32: 1, 0, 2, 3>} : tensor<32x32x32x32xi32> -> tensor<32x32x32x32xi32>
  tt.return
}

// -----

tt.func @unrealized_conversion_cast(%arg0: tensor<128x128xi32> {tt.contiguity = dense<[16, 32]> : tensor<2xi32>}) {
  // Case 1: AxisInfo is propagated through a sequence of
  // unrealized_conversion_cast ops.
  // expected-remark @below {{contiguity = [16, 32], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<128x128xi32> to !llvm.struct<(i32, i32, i32, i32)>
  // expected-remark @below {{contiguity = [16, 32], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %1 = builtin.unrealized_conversion_cast %0 : !llvm.struct<(i32, i32, i32, i32)> to tensor<128x128xi32>

  // Case 2: AxisInfo is falling back to the pessimistic state if the
  // propagated AxisInfo would be invalid.
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %2 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %3 = builtin.unrealized_conversion_cast %2 : !llvm.struct<(i32, i32, i32, i32)> to tensor<128x128xi32>
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<128x128xi32> -> tensor<128x128xi32>
  tt.return
}

// -----

// An unrealized conversion cast can materialize a value without an input.
// AxisInfo must return a pessimistic state with the result rank instead of
// indexing an empty operand lattice.
tt.func @empty_unrealized_conversion_cast() {
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %0 = "builtin.unrealized_conversion_cast"() : () -> tensor<4x8xi32>
  tt.return
}

// -----

// Axis analysis does not support multi-dimensional function arguments. Make
// sure that we don't crash.
tt.func @callee(%arg0: tensor<128x1xi32>) {
  tt.return
}

tt.func @caller() {
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [128, 1], divisibility = [1073741824, 1], constancy = [1, 1], constant_value = <none>}}
  %1 = tt.expand_dims %0 {axis = 1: i32} : tensor<128xi32> -> tensor<128x1xi32>
  tt.call @callee(%1) : (tensor<128x1xi32>) -> ()
  tt.return
}

// -----

tt.func @mul_zero_constancy() {
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %zeros = arith.constant dense<0> : tensor<128xi32>
  // expected-remark @below {{constancy = [128]}}
  %product = arith.muli %zeros, %range : tensor<128xi32>
  tt.return
}

// -----

tt.func @max_constancy() {
  %c5 = arith.constant dense<5> : tensor<4xi32>
  %c7 = arith.constant dense<7> : tensor<4xi32>
  // expected-remark @below {{constancy = [4], constant_value = 7}}
  %max = arith.maxsi %c5, %c7 : tensor<4xi32>
  tt.return
}

// -----

tt.func @select_same_value_constancy() {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %two = arith.constant dense<2> : tensor<4xi32>
  %mod = arith.remsi %range, %two : tensor<4xi32>
  %zero = arith.constant dense<0> : tensor<4xi32>
  %cond = arith.cmpi ne, %mod, %zero : tensor<4xi32>
  %lhs = arith.constant dense<42> : tensor<4xi32>
  %rhs = arith.constant dense<42> : tensor<4xi32>
  // expected-remark @below {{constancy = [4], constant_value = 42}}
  %sel = arith.select %cond, %lhs, %rhs : tensor<4xi1>, tensor<4xi32>
  tt.return
}

// -----

// Regression: SelectOp must clamp divisibility when condConstancy reduces the
// output contiguity below either input's contiguity. Otherwise the helper
// getDivisibilityFromContiguity overestimates divisibility because it does not
// see condConstancy. See issue triton-lang/triton#10067.
tt.func @select_cond_constancy_clamps_divisibility(%arg0: tensor<8xi1>) {
  // expected-remark @below {{contiguity = [8], divisibility = [8], constancy = [1], constant_value = <none>}}
  %lhs = tt.make_range {end = 16 : i32, start = 8 : i32} : tensor<8xi32>
  // expected-remark @below {{contiguity = [8], divisibility = [16], constancy = [1], constant_value = <none>}}
  %rhs = tt.make_range {end = 24 : i32, start = 16 : i32} : tensor<8xi32>
  // %arg0 has unknown contents, so condConstancy = 1. Output contiguity must
  // collapse to gcd(8, 8, 1) = 1; divisibility must clamp to 1 (not gcd(8, 16) = 8).
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %sel = arith.select %arg0, %lhs, %rhs : tensor<8xi1>, tensor<8xi32>
  tt.return
}

// -----

tt.func @cmp_after_max_constancy() {
  %c5 = arith.constant dense<5> : tensor<4xi32>
  %c7 = arith.constant dense<7> : tensor<4xi32>
  %max = arith.maxsi %c5, %c7 : tensor<4xi32>
  // expected-remark @below {{constancy = [4], constant_value = -1}}
  %cmp = arith.cmpi sgt, %max, %c5 : tensor<4xi32>
  tt.return
}

// -----

tt.func public @test_inductor_for() {
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = 64}}
  %c64_i32 = arith.constant 64 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0_i64 = arith.constant 0 : i64
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0_i32 = arith.constant 0 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %c1_i32 = arith.constant 1 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = 64}}
  %c64_i64 = arith.constant 64 : i64
  // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = -1}}
  %0 = arith.cmpi slt, %c0_i32, %c1_i32 : i32

  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = 64}}
  %1:2 = scf.if %0 -> (i32, i32) {
    scf.yield %c0_i32, %c64_i32 : i32, i32
  } else {
    scf.yield %c1_i32, %c64_i32 : i32, i32
  }

  // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = <none>}}
  %2 = scf.for %arg0 = %1#0 to %1#1 step %c64_i32 iter_args(%arg1 = %c0_i64) -> (i64)  : i32 {
    // expected-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = <none>}}
    %3 = arith.addi %arg1, %c64_i64 : i64
    scf.yield %3 : i64
  }
  tt.return
}

// -----

// Verify that if an operation is statically determined to be dead, we fall back
// to assigning it a pessimistic value, rather than skipping it entirely.
tt.func @dead_op_pessimistic() {
  %c5 = arith.constant dense<5> : tensor<4xi32>
  %c7 = arith.constant dense<7> : tensor<4xi32>
  %false = arith.constant false
  scf.if %false {
    // expected-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
    %add = arith.addi %c5, %c7 : tensor<4xi32>
  }
  tt.return
}

// -----

tt.func @negative_constants() {
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = -8}}
  %neg8_scalar = arith.constant -8 : i32
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = -8}}
  %neg8_dense = arith.constant dense<-8> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [128], constant_value = 16}}
  %sixteen = arith.constant dense<16> : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %sum = arith.addi %neg8_dense, %sixteen : tensor<128xi32>
  tt.return
}

// -----

// NARROW-LABEL: tt.func @narrow_branch_only
// NARROW: scf.if
// NARROW: arith.divsi {{.*}} : i32
// NARROW: } else {
// NARROW: arith.divsi {{.*}} : i64
tt.func @narrow_branch_only(%x: i64) -> i64 {
  %zero = arith.constant 0 : i64
  %limit = arith.constant 1024 : i64
  %three = arith.constant 3 : i64
  %lo = arith.cmpi sge, %x, %zero : i64
  %hi = arith.cmpi slt, %x, %limit : i64
  %bounded = arith.andi %lo, %hi : i1
  %result = scf.if %bounded -> i64 {
    %q = arith.divsi %x, %three : i64
    scf.yield %q : i64
  } else {
    %q = arith.divsi %x, %three : i64
    scf.yield %q : i64
  }
  tt.return %result : i64
}

// -----

// NARROW-LABEL: tt.func @narrow_unsigned_keep_product
// NARROW: arith.muli {{.*}} : i64
// NARROW: arith.divui {{.*}} : i32
// NARROW: arith.extui {{.*}} : i32 to i64
tt.func @narrow_unsigned_keep_product(%x: i32, %y: i32) -> (i64, i64) {
  %wide_x = arith.extui %x : i32 to i64
  %wide_y = arith.extui %y : i32 to i64
  %product = arith.muli %wide_x, %wide_y : i64
  %three = arith.constant 3 : i64
  %quotient = arith.divui %wide_x, %three : i64
  tt.return %product, %quotient : i64, i64
}

// -----

// NARROW-LABEL: tt.func @keep_unsigned_sign_extended
// NARROW: arith.remui {{.*}} : i64
// NARROW: arith.shrui {{.*}} : i64
tt.func @keep_unsigned_sign_extended(%x: i32) -> (i64, i64) {
  %wide = arith.extsi %x : i32 to i64
  %seven = arith.constant 7 : i64
  %shift = arith.constant 33 : i64
  %rem = arith.remui %wide, %seven : i64
  %shr = arith.shrui %wide, %shift : i64
  tt.return %rem, %shr : i64, i64
}

// -----

// NARROW-LABEL: tt.func @keep_wide_shift
// NARROW: arith.shrui {{.*}} : i64
tt.func @keep_wide_shift(%x: i32, %shift: i32) -> i64 {
  %wide = arith.extui %x : i32 to i64
  %amount = arith.extui %shift : i32 to i64
  %result = arith.shrui %wide, %amount : i64
  tt.return %result : i64
}

// -----

// NARROW-LABEL: tt.func @narrow_range_loop
// NARROW: scf.for {{.*}} : i32
tt.func @narrow_range_loop(%n: i64, %ptr: !tt.ptr<i64>) {
  %zero = arith.constant 0 : i64
  %limit = arith.constant 1024 : i64
  %one = arith.constant 1 : i64
  %lo = arith.cmpi sgt, %n, %zero : i64
  %hi = arith.cmpi sle, %n, %limit : i64
  %bounded = arith.andi %lo, %hi : i1
  scf.if %bounded {
    scf.for %i = %zero to %n step %one : i64 {
      %p = tt.addptr %ptr, %i : !tt.ptr<i64>, i64
      tt.store %p, %i : !tt.ptr<i64>
    }
  }
  tt.return
}


// -----

// NARROW-LABEL: @unsigned_result
// NARROW: arith.addi {{.*}} overflow<nuw> : i32
// NARROW: arith.extui {{.*}} : i32 to i64
tt.func @unsigned_result(%x: i64) -> i64 {
  %lo = arith.constant 2147483600 : i64
  %hi = arith.constant 2147483647 : i64
  %one = arith.constant 1 : i64
  %lower = arith.cmpi sge, %x, %lo : i64
  %upper = arith.cmpi sle, %x, %hi : i64
  %inside = arith.andi %lower, %upper : i1
  llvm.intr.assume %inside : i1
  %sum = arith.addi %x, %one overflow<nsw, nuw> : i64
  tt.return %sum : i64
}

// -----

// NARROW-LABEL: @negative_result
// NARROW: arith.subi {{.*}} overflow<nsw> : i32
// NARROW: arith.extsi {{.*}} : i32 to i64
tt.func @negative_result(%x: i64) -> i64 {
  %lo = arith.constant -2147483647 : i64
  %hi = arith.constant -2147483600 : i64
  %one = arith.constant 1 : i64
  %lower = arith.cmpi sge, %x, %lo : i64
  %upper = arith.cmpi sle, %x, %hi : i64
  %inside = arith.andi %lower, %upper : i1
  llvm.intr.assume %inside : i1
  %difference = arith.subi %x, %one overflow<nsw, nuw> : i64
  tt.return %difference : i64
}

// -----

// NARROW-LABEL: @safe_flags
// NARROW: arith.addi {{.*}} overflow<nsw, nuw> : i32
// NARROW: arith.ext{{[su]}}i {{.*}} : i32 to i64
tt.func @safe_flags(%x: i64) -> i64 {
  %lo = arith.constant 0 : i64
  %hi = arith.constant 100 : i64
  %one = arith.constant 1 : i64
  %lower = arith.cmpi sge, %x, %lo : i64
  %upper = arith.cmpi sle, %x, %hi : i64
  %inside = arith.andi %lower, %upper : i1
  llvm.intr.assume %inside : i1
  %sum = arith.addi %x, %one overflow<nsw, nuw> : i64
  tt.return %sum : i64
}

// -----

// NARROW-LABEL: @already_narrow
// NARROW: arith.addi {{.*}} overflow<nsw, nuw> : i32
tt.func @already_narrow(%x: i32, %y: i32) -> i32 {
  %sum = arith.addi %x, %y overflow<nsw, nuw> : i32
  tt.return %sum : i32
}

// -----

// NARROW-LABEL: @keep_wide
// NARROW: arith.addi {{.*}} overflow<nsw, nuw> : i64
tt.func @keep_wide(%x: i64, %y: i64) -> i64 {
  %sum = arith.addi %x, %y overflow<nsw, nuw> : i64
  tt.return %sum : i64
}

// -----

// Duplicate operands must receive the same edge fact.
tt.func @single_predecessor(%x: i32) -> i32 {
  %bound = arith.constant 1024 : i32
  %condition = arith.cmpi slt, %x, %bound : i32
  cf.cond_br %condition, ^then(%x, %x : i32, i32), ^else(%x : i32)
^then(%a: i32, %alias: i32):
  %check_a = arith.cmpi slt, %a, %bound {test.proof = "proven"} : i32
  %check_alias = arith.cmpi slt, %alias, %bound {test.proof = "proven"} : i32
  tt.return %a : i32
^else(%b: i32):
  %check_b = arith.cmpi sge, %b, %bound {test.proof = "proven"} : i32
  %check_false = arith.cmpi slt, %b, %bound {test.proof = "disproven"} : i32
  tt.return %b : i32
}

// An unbounded predecessor must discard the edge fact at the join.
tt.func @multiple_predecessors(%x: i32, %y: i32) -> i32 {
  %bound = arith.constant 1024 : i32
  %condition = arith.cmpi slt, %x, %bound : i32
  cf.cond_br %condition, ^merge(%x : i32), ^other
^other:
  cf.br ^merge(%y : i32)
^merge(%arg: i32):
  %check = arith.cmpi slt, %arg, %bound {test.proof = "unknown"} : i32
  tt.return %arg : i32
}

// Opposite edges into one block must not leak either outcome.
tt.func @same_successor(%x: i32) -> i32 {
  %bound = arith.constant 1024 : i32
  %condition = arith.cmpi slt, %x, %bound : i32
  cf.cond_br %condition, ^merge(%x : i32), ^merge(%x : i32)
^merge(%arg: i32):
  %check = arith.cmpi slt, %arg, %bound {test.proof = "unknown"} : i32
  tt.return %arg : i32
}

// Both incoming edges establish the same bound.
tt.func @bounded_predecessors(%x: i32, %y: i32) -> i32 {
  %bound = arith.constant 1024 : i32
  %first = arith.cmpi slt, %x, %bound : i32
  cf.cond_br %first, ^merge(%x : i32), ^other
^other:
  %second = arith.cmpi slt, %y, %bound : i32
  cf.cond_br %second, ^merge(%y : i32), ^exit
^merge(%arg: i32):
  %check = arith.cmpi slt, %arg, %bound {test.proof = "proven"} : i32
  tt.return %arg : i32
^exit:
  tt.return %y : i32
}

// Both predicate operands can cross the edge, including duplicate aliases.
tt.func @transported_bound(%x: i32) -> i32 {
  %bound = arith.constant 1024 : i32
  %condition = arith.cmpi slt, %x, %bound : i32
  cf.cond_br %condition, ^then(%x, %x, %bound, %bound : i32, i32, i32, i32), ^else(%x, %bound : i32, i32)
^then(%a: i32, %alias_a: i32, %b: i32, %alias_b: i32):
  %check_a = arith.cmpi slt, %a, %b {test.proof = "proven"} : i32
  %check_alias_a = arith.cmpi slt, %alias_a, %b {test.proof = "proven"} : i32
  %check_alias_b = arith.cmpi slt, %a, %alias_b {test.proof = "proven"} : i32
  %check_both = arith.cmpi slt, %alias_a, %alias_b {test.proof = "proven"} : i32
  %check_original = arith.cmpi slt, %a, %bound {test.proof = "proven"} : i32
  tt.return %a : i32
^else(%c: i32, %d: i32):
  %check_else = arith.cmpi sge, %c, %d {test.proof = "proven"} : i32
  tt.return %c : i32
}

// Equal predicates with different bounds do not establish a common relation.
tt.func @different_bounds(%choose: i1, %x: i32, %y: i32, %bound_a: i32, %bound_b: i32) -> i32 {
  cf.cond_br %choose, ^left, ^right
^left:
  %a = tt.refine %x, %bound_a {predicate = 2 : i64} : i32
  cf.br ^merge(%a : i32)
^right:
  %b = tt.refine %y, %bound_b {predicate = 2 : i64} : i32
  cf.br ^merge(%b : i32)
^merge(%arg: i32):
  %check_a = arith.cmpi slt, %arg, %bound_a {test.proof = "unknown"} : i32
  %check_b = arith.cmpi slt, %arg, %bound_b {test.proof = "unknown"} : i32
  tt.return %arg : i32
}

// -----

tt.func @generic_range_interfaces(%input: i32) {
  %zero = arith.constant 0 : i32
  %limit = arith.constant 16 : i32
  %low = tt.refine %input, %zero {predicate = 5 : i64} : i32
  %bounded = tt.refine %low, %limit {predicate = 2 : i64} : i32
  %lower = arith.cmpi sge, %bounded, %zero {test.generic_proof = "proven"} : i32
  %upper = arith.cmpi slt, %bounded, %limit {test.generic_proof = "proven"} : i32
  %sibling = arith.cmpi slt, %input, %limit {test.generic_proof = "unknown"} : i32
  %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %expanded = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %broadcast = tt.broadcast %expanded : tensor<4x1xi32> -> tensor<4x2xi32>
  %transposed = tt.trans %broadcast {order = array<i32: 1, 0>} : tensor<4x2xi32> -> tensor<2x4xi32>
  %flat = tt.reshape %transposed : tensor<2x4xi32> -> tensor<8xi32>
  %splat = tt.splat %bounded : i32 -> tensor<8xi32>
  %joined = tt.join %flat, %splat : tensor<8xi32> -> tensor<8x2xi32>
  %left, %right = tt.split %joined : tensor<8x2xi32> -> tensor<8xi32>
  %indices = arith.constant dense<0> : tensor<8xi32>
  %gather = tt.gather %left[%indices] {axis = 0 : i32} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  %tensor_limit = arith.constant dense<16> : tensor<8xi32>
  %gather_bound = arith.cmpi slt, %gather, %tensor_limit {test.generic_proof = "proven"} : tensor<8xi32>
  %split_bound = arith.cmpi slt, %right, %tensor_limit {test.generic_proof = "proven"} : tensor<8xi32>
  tt.return
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#other = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx950"} {
  tt.func @generic_target_and_layout_ranges() {
    %true = arith.constant true
    amdg.cond_barrier %true
    %pid = tt.get_program_id x : i32
    %max = arith.constant 16777214 : i32
    %bound = arith.cmpi sle, %pid, %max {test.generic_proof = "proven"} : i32
    %splat = tt.splat %pid : i32 -> tensor<256xi32, #blocked>
    %converted = ttg.convert_layout %splat : tensor<256xi32, #blocked> -> tensor<256xi32, #other>
    %required = ttg.require_layout %converted : tensor<256xi32, #other> -> tensor<256xi32, #other>
    %released = ttg.release_layout %required : tensor<256xi32, #other> -> tensor<256xi32, #other>
    %limit = arith.constant dense<16777214> : tensor<256xi32, #other>
    %layout_bound = arith.cmpi sle, %released, %limit {test.generic_proof = "proven"} : tensor<256xi32, #other>
    tt.return
  }
}
