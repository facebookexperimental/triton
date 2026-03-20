// RUN: triton-opt %s -test-print-program-id-address -split-input-file 2>&1 | FileCheck %s

// Test basic program_id to address flow
// CHECK: remark: PID_AFFINE(pid[0] * 1)
// CHECK: remark: PID_AFFINE(pid[0] * 128)
// CHECK: remark: tt.load address pattern: PID_AFFINE
tt.func @basic_pid_to_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c128 = arith.constant 128 : i32
    %block_start = arith.muli %pid, %c128 : i32

    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>

    // Splat the scalar to tensor before adding
    %block_start_splat = tt.splat %block_start : i32 -> tensor<128xi32>
    %indices = arith.addi %block_start_splat, %offsets : tensor<128xi32>

    %ptr_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptrs = tt.addptr %ptr_base, %indices : tensor<128x!tt.ptr<f32>>, tensor<128xi32>

    %vals = tt.load %ptrs : tensor<128x!tt.ptr<f32>>

    tt.return
}

// -----

// Test multiple program_id axes
// CHECK: remark: PID_AFFINE(pid[0] * 64 + pid[1] * 128)
tt.func @multi_axis_pid(%arg0: !tt.ptr<f32>) {
    %pid_x = tt.get_program_id x : i32
    %pid_y = tt.get_program_id y : i32

    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32

    %x_offset = arith.muli %pid_x, %c64 : i32
    %y_offset = arith.muli %pid_y, %c128 : i32

    %combined = arith.addi %x_offset, %y_offset : i32

    tt.return
}

// -----

// Test pid-independent access (constant pattern)
// CHECK: remark: tt.load address pattern: CONSTANT
tt.func @constant_access(%arg0: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>

    %ptr_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>

    // Constant pattern: all programs access same memory
    %ptrs = tt.addptr %ptr_base, %offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>

    %vals = tt.load %ptrs : tensor<64x!tt.ptr<f32>>

    tt.return
}

// -----

// Test 2D blocking pattern
// CHECK: remark: PID_AFFINE(pid[0] * 64)
// CHECK: remark: PID_AFFINE(pid[1] * 64)
tt.func @matmul_pattern(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32

    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32

    %m_start = arith.muli %pid_m, %c64 : i32
    %n_start = arith.muli %pid_n, %c64 : i32

    // Row offsets within block
    %m_offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>

    // Splat scalar to tensor before adding
    %m_start_splat = tt.splat %m_start : i32 -> tensor<64xi32>
    %m_indices = arith.addi %m_start_splat, %m_offsets : tensor<64xi32>

    // Expand to 2D
    %m_indices_2d = tt.expand_dims %m_indices {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %m_indices_bc = tt.broadcast %m_indices_2d : tensor<64x1xi32> -> tensor<64x64xi32>

    // Multiply by stride (K dimension)
    // This shows row-major layout: addr = base + row * K + col
    %arg1_splat = tt.splat %arg1 : i32 -> tensor<64x64xi32>
    %row_offset = arith.muli %m_indices_bc, %arg1_splat : tensor<64x64xi32>

    tt.return
}

// -----

// Test store operation
// CHECK: remark: tt.store address pattern: PID_AFFINE
tt.func @store_pattern(%arg0: !tt.ptr<f32>, %arg1: tensor<128xf32>) {
    %pid = tt.get_program_id x : i32
    %c128 = arith.constant 128 : i32
    %block_start = arith.muli %pid, %c128 : i32
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>

    // Splat scalar to tensor before adding
    %block_start_splat = tt.splat %block_start : i32 -> tensor<128xi32>
    %indices = arith.addi %block_start_splat, %offsets : tensor<128xi32>

    %ptr_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptrs = tt.addptr %ptr_base, %indices : tensor<128x!tt.ptr<f32>>, tensor<128xi32>

    // Store pattern: same as load, PID_AFFINE with axis=0, stride=512
    tt.store %ptrs, %arg1 : tensor<128x!tt.ptr<f32>>

    tt.return
}
