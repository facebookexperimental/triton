// RUN: triton-opt %s -split-input-file -tritonamdgpu-fold-lds-staging-over-budget | FileCheck %s

// One 32 KiB staging fits in the 64 KiB gfx942 LDS and is kept.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @keep_staging_within_budget
  tt.func public @keep_staging_within_budget(%src: tensor<64x256xf16, #blocked>) -> tensor<64x256xf16, #dot> {
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    // CHECK-NOT: ttg.convert_layout
    %alloc = ttg.local_alloc %src : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %a = ttg.local_load %alloc : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    tt.return %a : tensor<64x256xf16, #dot>
  }
}

// -----

// Three live 32 KiB stagings need 96 KiB, so all of them fold.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @fold_staging_over_budget
  tt.func public @fold_staging_over_budget(%x: tensor<64x256xf16, #blocked>, %y: tensor<64x256xf16, #blocked>, %z: tensor<64x256xf16, #blocked>) -> (tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>) {
    // CHECK-NOT: ttg.local_alloc
    // CHECK-NOT: ttg.local_load
    // CHECK-COUNT-3: ttg.convert_layout %{{.*}} : tensor<64x256xf16, #blocked> -> tensor<64x256xf16, #ttg.dot_op
    %ax = ttg.local_alloc %x : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %ay = ttg.local_alloc %y : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %az = ttg.local_alloc %z : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %lx = ttg.local_load %ax : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %ly = ttg.local_load %ay : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %lz = ttg.local_load %az : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    tt.return %lx, %ly, %lz : tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>
  }
}

// -----

// TLX modules are left to tlx_propagate_layout, even over budget.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @skip_tlx_module
  tt.func public @skip_tlx_module(%x: tensor<64x256xf16, #blocked>, %y: tensor<64x256xf16, #blocked>, %z: tensor<64x256xf16, #blocked>) -> (tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>) {
    // CHECK-COUNT-3: ttg.local_alloc
    // CHECK-COUNT-3: ttg.local_load
    // CHECK-NOT: ttg.convert_layout
    %ax = ttg.local_alloc %x : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %ay = ttg.local_alloc %y : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %az = ttg.local_alloc %z : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %lx = ttg.local_load %ax : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %ly = ttg.local_load %ay : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %lz = ttg.local_load %az : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    tt.return %lx, %ly, %lz : tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>
  }
}

// -----

// Each case below is over budget: the two stagings fold, the third buffer
// stays. A mutable buffer is not staging.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @keep_mutable_alloc
  tt.func public @keep_mutable_alloc(%x: tensor<64x256xf16, #blocked>, %y: tensor<64x256xf16, #blocked>, %z: tensor<64x256xf16, #blocked>) -> (tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>) {
    // CHECK: %[[Z:.*]] = ttg.local_alloc %{{.*}} : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<{{.*}}mutable>
    // CHECK-COUNT-2: ttg.convert_layout %{{.*}} : tensor<64x256xf16, #blocked> -> tensor<64x256xf16, #ttg.dot_op
    // CHECK: ttg.local_load %[[Z]]
    %ax = ttg.local_alloc %x : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %ay = ttg.local_alloc %y : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %az = ttg.local_alloc %z : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %lx = ttg.local_load %ax : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %ly = ttg.local_load %ay : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %lz = ttg.local_load %az : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #dot>
    tt.return %lx, %ly, %lz : tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>
  }
}

// -----

// A load ordered by a token is left alone.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @keep_load_with_token
  tt.func public @keep_load_with_token(%x: tensor<64x256xf16, #blocked>, %y: tensor<64x256xf16, #blocked>, %z: tensor<64x256xf16, #blocked>, %tok: !ttg.async.token) -> (tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>) {
    // CHECK: %[[Z:.*]] = ttg.local_alloc %arg2
    // CHECK-COUNT-2: ttg.convert_layout %{{.*}} : tensor<64x256xf16, #blocked> -> tensor<64x256xf16, #ttg.dot_op
    // CHECK: ttg.local_load %[[Z]] token
    %ax = ttg.local_alloc %x : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %ay = ttg.local_alloc %y : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %az = ttg.local_alloc %z : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %lx = ttg.local_load %ax : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %ly = ttg.local_load %ay : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %lz = ttg.local_load %az token %tok : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    tt.return %lx, %ly, %lz : tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>
  }
}

// -----

// A load into a non-dot-operand layout is not staging.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @keep_non_dot_load
  tt.func public @keep_non_dot_load(%x: tensor<64x256xf16, #blocked>, %y: tensor<64x256xf16, #blocked>, %z: tensor<64x256xf16, #blocked>) -> (tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #blocked>) {
    // CHECK: %[[Z:.*]] = ttg.local_alloc %arg2
    // CHECK-COUNT-2: ttg.convert_layout %{{.*}} : tensor<64x256xf16, #blocked> -> tensor<64x256xf16, #ttg.dot_op
    // CHECK: ttg.local_load %[[Z]] : {{.*}} -> tensor<64x256xf16, #blocked>
    %ax = ttg.local_alloc %x : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %ay = ttg.local_alloc %y : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %az = ttg.local_alloc %z : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %lx = ttg.local_load %ax : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %ly = ttg.local_load %ay : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %lz = ttg.local_load %az : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #blocked>
    tt.return %lx, %ly, %lz : tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #blocked>
  }
}

// -----

// A load carrying `tlx.rematerialize_coordinates` keeps it for lowering.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @keep_remat_load
  tt.func public @keep_remat_load(%x: tensor<64x256xf16, #blocked>, %y: tensor<64x256xf16, #blocked>, %z: tensor<64x256xf16, #blocked>) -> (tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>) {
    // CHECK: %[[Z:.*]] = ttg.local_alloc %arg2
    // CHECK-COUNT-2: ttg.convert_layout %{{.*}} : tensor<64x256xf16, #blocked> -> tensor<64x256xf16, #ttg.dot_op
    // CHECK: ttg.local_load %[[Z]] {tlx.rematerialize_coordinates
    %ax = ttg.local_alloc %x : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %ay = ttg.local_alloc %y : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %az = ttg.local_alloc %z : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %lx = ttg.local_load %ax : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %ly = ttg.local_load %ay : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %lz = ttg.local_load %az {tlx.rematerialize_coordinates} : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    tt.return %lx, %ly, %lz : tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>
  }
}

// -----

// A load carrying `tlx.rematerialize_coordinates_group` keeps it for lowering.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @keep_remat_group_load
  tt.func public @keep_remat_group_load(%x: tensor<64x256xf16, #blocked>, %y: tensor<64x256xf16, #blocked>, %z: tensor<64x256xf16, #blocked>) -> (tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>) {
    // CHECK: %[[Z:.*]] = ttg.local_alloc %arg2
    // CHECK-COUNT-2: ttg.convert_layout %{{.*}} : tensor<64x256xf16, #blocked> -> tensor<64x256xf16, #ttg.dot_op
    // CHECK: ttg.local_load %[[Z]] {tlx.rematerialize_coordinates_group
    %ax = ttg.local_alloc %x : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %ay = ttg.local_alloc %y : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %az = ttg.local_alloc %z : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %lx = ttg.local_load %ax : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %ly = ttg.local_load %ay : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %lz = ttg.local_load %az {tlx.rematerialize_coordinates_group = 3 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    tt.return %lx, %ly, %lz : tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>
  }
}

// -----

// With no gfx-arch option the budget comes from ttg.target: three live 32 KiB
// stagings fit in the 160 KiB gfx950 LDS.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @keep_staging_within_gfx950_budget
  tt.func public @keep_staging_within_gfx950_budget(%x: tensor<64x256xf16, #blocked>, %y: tensor<64x256xf16, #blocked>, %z: tensor<64x256xf16, #blocked>) -> (tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>) {
    // CHECK-COUNT-3: ttg.local_alloc
    // CHECK-NOT: ttg.convert_layout
    %ax = ttg.local_alloc %x : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %ay = ttg.local_alloc %y : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %az = ttg.local_alloc %z : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %lx = ttg.local_load %ax : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %ly = ttg.local_load %ay : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    %lz = ttg.local_load %az : !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf16, #dot>
    tt.return %lx, %ly, %lz : tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>, tensor<64x256xf16, #dot>
  }
}
