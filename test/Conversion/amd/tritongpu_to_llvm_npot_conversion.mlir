// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-amdgpu-shared-memory=arch=gfx950 --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --convert-builtin-func-to-llvm | FileCheck %s

// Lane bits 0 and 1 swap within a wave64; the odd digit stays in registers,
// so each thread stores its three values.
#src = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[64]], lane = [[2], [1], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: module attributes {{.*}}ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @wave_shuffle
  // CHECK-NOT: ptr<3>
  // CHECK-COUNT-3: rocdl.ds_bpermute
  // CHECK-NOT: {{ptr<3>|barrier}}
  // CHECK-COUNT-3: llvm.store {{.*}} : vector<1xi32>, !llvm.ptr<1>
  // CHECK-NEXT: llvm.return
  tt.func public @wave_shuffle(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #src>
    %c = ttg.convert_layout %r : tensor<192xi32, #src> -> tensor<192xi32, #dst>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<192x!tt.ptr<i32>, #dst>
    tt.store %s, %c : tensor<192x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

// Destination register j holds source register 2j mod 11 (11 for j = 11).
#src = #ttg.linear<{register = [[1], [2], [256]], lane = [[4], [8], [16], [32], [64], [128]], warp = [], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [768]}>
#dst = #ttg.linear<{register = [[2], [256], [1]], lane = [[4], [8], [16], [32], [64], [128]], warp = [], block = [], inputRadices = {register = [2, 3, 2], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [768]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: module attributes {{.*}}ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @registers_i8
  // CHECK: %[[T0:.*]] = llvm.trunc %{{.*}} : i32 to i8
  // CHECK-NEXT: %[[T1:.*]] = llvm.trunc %{{.*}} : i32 to i8
  // CHECK-NEXT: %[[T2:.*]] = llvm.trunc %{{.*}} : i32 to i8
  // CHECK-NOT: {{ptr<3>|ds_bpermute|permlane}}
  // CHECK: llvm.insertelement %[[T0]]
  // CHECK-NEXT: llvm.store
  // CHECK-NEXT: llvm.insertelement %[[T2]]
  // CHECK-COUNT-4: llvm.insertelement
  // CHECK-NEXT: llvm.store
  // CHECK-NEXT: llvm.insertelement %[[T1]]
  tt.func public @registers_i8(%p: !tt.ptr<i8>) {
    %r = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32, #src>
    %t = arith.trunci %r : tensor<768xi32, #src> to tensor<768xi8, #src>
    %c = ttg.convert_layout %t : tensor<768xi8, #src> -> tensor<768xi8, #dst>
    %s = tt.splat %p : !tt.ptr<i8> -> tensor<768x!tt.ptr<i8>, #dst>
    tt.store %s, %c : tensor<768x!tt.ptr<i8>, #dst>
    tt.return
  }
}

// -----

// Register 1 moves to lane 32. Of each broadcast lane pair only lane bit 5 = 0
// stores, i1 takes a byte, and a wave barrier suffices.
#src = #ttg.linear<{register = [[64], [32]], lane = [[1], [2], [4], [8], [16], [0]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: module attributes {{.*}}ttg.shared = 192 : i32
  // CHECK-LABEL: llvm.func @wave_shared_i1
  // CHECK: llvm.zext {{.*}} : i1 to i8
  // CHECK: %[[Q:.*]] = llvm.udiv %{{.*}}, %{{.*}} : i32
  // CHECK: %[[DIGIT:.*]] = llvm.urem %[[Q]], %{{.*}} : i32
  // CHECK: %[[OWNS:.*]] = llvm.icmp "eq" %[[DIGIT]], %{{.*}} : i32
  // CHECK: %[[PRED:.*]] = llvm.and %{{.*}}, %[[OWNS]] : i1
  // CHECK: llvm.cond_br %[[PRED]]
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT: llvm.store {{.*}} : i8, !llvm.ptr<3>
  // CHECK: rocdl.wave.barrier
  // CHECK-NOT: rocdl.s.barrier
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> i8
  // CHECK: llvm.trunc {{.*}} : i8 to i1
  tt.func public @wave_shared_i1(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #src>
    %k = arith.constant 100 : i32
    %z = tt.splat %k : i32 -> tensor<192xi32, #src>
    %b = arith.cmpi slt, %r, %z : tensor<192xi32, #src>
    %c = ttg.convert_layout %b : tensor<192xi1, #src> -> tensor<192xi1, #dst>
    %e = arith.extui %c : tensor<192xi1, #dst> to tensor<192xi32, #dst>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<192x!tt.ptr<i32>, #dst>
    tt.store %s, %e : tensor<192x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

// The register run [1], [2] ends at the odd digit, so each thread moves its
// six 16-byte-aligned values as three two-element vectors.
#p = #ttg.linear<{register = [[1], [2]], lane = [[0], [0], [0], [0], [0], [0]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [6]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @odd_digit_ends_vector
  // CHECK-COUNT-3: llvm.load {{.*}} -> vector<2xf32>
  // CHECK-COUNT-3: llvm.store {{.*}} : vector<2xf32>
  tt.func public @odd_digit_ends_vector(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %r = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32, #p>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<6x!tt.ptr<f32>, #p>
    %a = tt.addptr %s, %r {tt.divisibility = dense<16> : tensor<1xi32>} : tensor<6x!tt.ptr<f32>, #p>, tensor<6xi32, #p>
    %v = tt.load %a : tensor<6x!tt.ptr<f32>, #p>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<6x!tt.ptr<f32>, #p>
    %b = tt.addptr %t, %r {tt.divisibility = dense<16> : tensor<1xi32>} : tensor<6x!tt.ptr<f32>, #p>, tensor<6xi32, #p>
    tt.store %b, %v : tensor<6x!tt.ptr<f32>, #p>
    tt.return
  }
}

// -----

// Register digit 1 broadcasts, so registers 3-5 repeat registers 0-2.
#p = #ttg.linear<{register = [[64], [0]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @broadcast_register_store
  // CHECK: llvm.store %{{.*}}, %[[P0:[0-9]+]]
  // CHECK: llvm.store %{{.*}}, %[[P1:[0-9]+]]
  // CHECK: llvm.store %{{.*}}, %[[P2:[0-9]+]]
  // CHECK: llvm.store %{{.*}}, %[[P0]]
  // CHECK: llvm.store %{{.*}}, %[[P1]]
  // CHECK: llvm.store %{{.*}}, %[[P2]]
  tt.func public @broadcast_register_store(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #p>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<192x!tt.ptr<i32>, #p>
    %a = tt.addptr %s, %r : tensor<192x!tt.ptr<i32>, #p>, tensor<192xi32, #p>
    tt.store %a, %r : tensor<192x!tt.ptr<i32>, #p>
    tt.return
  }
}

// -----

// Warp bit 0 broadcasts, so a buffer store keeps one owner per element.
#p = #ttg.linear<{register = [[128]], lane = [[1], [2], [4], [8], [16], [32]], warp = [[0], [64]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [384]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @buffer_store_warp_broadcast
  // CHECK-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[W:.*]] = rocdl.readfirstlane
  // CHECK: rocdl.make.buffer.rsrc
  // CHECK-NEXT: %[[BIT:.*]] = llvm.and %[[W]], %[[ONE]]
  // CHECK-NEXT: %[[OWN:.*]] = llvm.icmp "eq" %[[BIT]], %[[ZERO]]
  // CHECK-COUNT-3: llvm.select %[[OWN]]
  tt.func public @buffer_store_warp_broadcast(%p: !tt.ptr<i32>) {
    %v = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32, #p>
    amdg.buffer_store %v, %p[%v] : !tt.ptr<i32> -> tensor<384xi32, #p>
    tt.return
  }
}

// -----

// A stores 2 f32 per access (lane 0 steps by 2) and B loads 4 (its odd
// register digit starts at 128); B's register count differs, so no map. The
// odd digit is a register on both sides, so each round moves one odd value.
#src = #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [8], [16], [32], [64]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [384]}>
#dst = #ttg.linear<{register = [[1], [2], [128]], lane = [[4], [8], [16], [32], [64], [0]], warp = [], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [384]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: module attributes {{.*}}ttg.shared = 512 : i32
  // CHECK-LABEL: llvm.func @wave_shared_vectorized
  // CHECK: llvm.store {{.*}} : vector<2xf32>, !llvm.ptr<3>
  // CHECK-NOT: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK: rocdl.wave.barrier
  // CHECK-NOT: llvm.store {{.*}} !llvm.ptr<3>
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
  // CHECK-NOT: !llvm.ptr<3> -> f32
  // CHECK: rocdl.wave.barrier
  // CHECK-COUNT-2: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
  // CHECK-NOT: !llvm.ptr<3> -> f32
  tt.func public @wave_shared_vectorized(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>, #src>
    %v = tt.load %s : tensor<384x!tt.ptr<f32>, #src>
    %c = ttg.convert_layout %v : tensor<384xf32, #src> -> tensor<384xf32, #dst>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>, #dst>
    tt.store %d, %c : tensor<384x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// -----

// No 192 layout keeps these registers. Of each broadcast lane pair only lane
// bit 5 = 0 stores, at row * 3 + column; every lane reloads after a barrier.
#b = #ttg.linear<{register = [[32, 0], [0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [64, 3]}>
#flat = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: module attributes {{.*}}ttg.shared = 768 : i32
  // CHECK-LABEL: llvm.func @wave_moving_reshape
  // CHECK: %[[THREE:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %[[Q:.*]] = llvm.udiv %{{.*}}, %{{.*}} : i32
  // CHECK: %[[DIGIT:.*]] = llvm.urem %[[Q]], %{{.*}} : i32
  // CHECK: %[[OWNS:.*]] = llvm.icmp "eq" %[[DIGIT]], %{{.*}} : i32
  // CHECK: %[[PRED:.*]] = llvm.and %{{.*}}, %[[OWNS]] : i1
  // CHECK: llvm.mul %{{.*}}, %[[THREE]] : i32
  // CHECK: llvm.cond_br %[[PRED]]
  // CHECK-NEXT: ^bb1:
  // CHECK-NEXT: llvm.store {{.*}} : f32, !llvm.ptr<3>
  // CHECK: rocdl.s.barrier
  // CHECK-COUNT-3: llvm.load {{.*}} : !llvm.ptr<3> -> f32
  tt.func public @wave_moving_reshape(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<64x3x!tt.ptr<f32>, #b>
    %v = tt.load %s : tensor<64x3x!tt.ptr<f32>, #b>
    %r = tt.reshape %v : tensor<64x3xf32, #b> -> tensor<192xf32, #flat>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #flat>
    tt.store %d, %r : tensor<192x!tt.ptr<f32>, #flat>
    tt.return
  }
}
