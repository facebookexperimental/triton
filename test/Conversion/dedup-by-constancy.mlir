// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm --llvm-optimize-for-nvvm-target | FileCheck %s

// CHECK-LABEL: dedup_by_constancy_full
// CHECK-COUNT-2: llvm.add
// CHECK-NOT: llvm.add
// CHECK: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
// CHECK: llvm.sdiv
// CHECK-NOT: llvm.sdiv
// CHECK: llvm.getelementptr %arg0[[[REGISTER:%[0-9]+]]]
// CHECK-COUNT-7: llvm.getelementptr %arg0[[[REGISTER]]]
// CHECK-NOT: llvm.getelementptr %arg0[[[REGISTER]]]
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dedup_by_constancy_full(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<256> : tensor<1024xi32, #blocked>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg2 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    %7 = arith.divsi %4, %cst : tensor<1024xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %10 = tt.load %9, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    %11 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %12 = tt.addptr %11, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %12, %10, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: dedup_by_constancy_partial
// CHECK-COUNT-4: llvm.add
// CHECK-NOT: llvm.add
// CHECK: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
// CHECK-COUNT-2: llvm.sdiv
// CHECK-NOT: llvm.sdiv
// CHECK: llvm.getelementptr %arg0[[[REGISTER1:%[0-9]+]]]
// CHECK-COUNT-3: llvm.getelementptr %arg0[[[REGISTER1]]]
// CHECK-NOT: llvm.getelementptr %arg0[[[REGISTER1]]]
// CHECK: llvm.getelementptr %arg0[[[REGISTER2:%[0-9]+]]]
// CHECK-COUNT-3: llvm.getelementptr %arg0[[[REGISTER2]]]
// CHECK-NOT: llvm.getelementptr %arg0[[[REGISTER2]]]
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dedup_by_constancy_partial(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<4> : tensor<1024xi32, #blocked>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg2 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    %7 = arith.divsi %4, %cst : tensor<1024xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %10 = tt.load %9, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    %11 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %12 = tt.addptr %11, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %12, %10, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: dedup_i1_row_repeats
// CHECK-COUNT-16: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
// CHECK: llvm.sdiv
// CHECK-COUNT-3: llvm.sdiv
// CHECK-NOT: llvm.sdiv
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dedup_i1_row_repeats(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}) {
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %c16 = arith.constant dense<16> : tensor<64x128xi32, #blocked>
    %c128 = arith.constant dense<128> : tensor<64x128xi32, #blocked>
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bs = tt.splat %base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows = arith.addi %bs, %r : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %re = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rb = tt.broadcast %re : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %n = tt.splat %arg1 : i32 -> tensor<64x128xi32, #blocked>
    %m = arith.cmpi slt, %rb, %n : tensor<64x128xi32, #blocked>
    %q = arith.divsi %rb, %c16 : tensor<64x128xi32, #blocked>
    %c = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %ce = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %cb = tt.broadcast %ce : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %qo = arith.muli %q, %c128 : tensor<64x128xi32, #blocked>
    %off = arith.addi %qo, %cb : tensor<64x128xi32, #blocked>
    %p = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %pp = tt.addptr %p, %off : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    tt.store %pp, %cst, %m : tensor<64x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: dedup_i1_row_repeats_sm90
// CHECK-COUNT-16: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
// CHECK: llvm.sdiv
// CHECK-COUNT-3: llvm.sdiv
// CHECK-NOT: llvm.sdiv
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dedup_i1_row_repeats_sm90(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}) {
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %c16 = arith.constant dense<16> : tensor<64x128xi32, #blocked>
    %c128 = arith.constant dense<128> : tensor<64x128xi32, #blocked>
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bs = tt.splat %base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows = arith.addi %bs, %r : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %re = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rb = tt.broadcast %re : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %n = tt.splat %arg1 : i32 -> tensor<64x128xi32, #blocked>
    %m = arith.cmpi slt, %rb, %n : tensor<64x128xi32, #blocked>
    %q = arith.divsi %rb, %c16 : tensor<64x128xi32, #blocked>
    %c = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %ce = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %cb = tt.broadcast %ce : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %qo = arith.muli %q, %c128 : tensor<64x128xi32, #blocked>
    %off = arith.addi %qo, %cb : tensor<64x128xi32, #blocked>
    %p = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %pp = tt.addptr %p, %off : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    tt.store %pp, %cst, %m : tensor<64x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: dedup_i1_row_repeats_sm80
// CHECK-COUNT-4: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
// CHECK-COUNT-4: llvm.sdiv
// CHECK-NOT: llvm.sdiv
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dedup_i1_row_repeats_sm80(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}) {
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %c16 = arith.constant dense<16> : tensor<64x128xi32, #blocked>
    %c128 = arith.constant dense<128> : tensor<64x128xi32, #blocked>
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bs = tt.splat %base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows = arith.addi %bs, %r : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %re = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rb = tt.broadcast %re : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %n = tt.splat %arg1 : i32 -> tensor<64x128xi32, #blocked>
    %m = arith.cmpi slt, %rb, %n : tensor<64x128xi32, #blocked>
    %q = arith.divsi %rb, %c16 : tensor<64x128xi32, #blocked>
    %c = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %ce = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %cb = tt.broadcast %ce : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %qo = arith.muli %q, %c128 : tensor<64x128xi32, #blocked>
    %off = arith.addi %qo, %cb : tensor<64x128xi32, #blocked>
    %p = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %pp = tt.addptr %p, %off : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    tt.store %pp, %cst, %m : tensor<64x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
