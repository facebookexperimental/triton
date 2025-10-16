// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-assign-latencies=num-stages=3 -canonicalize | FileCheck %s

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @default_stages
tt.func @default_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @small_load
// We should *not* assign latency to the load of b_ptr.
tt.func @small_load(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL>) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}}
    // CHECK-NOT: tt.latency
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @load_into_shared
tt.func @load_into_shared(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #mma> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #mma>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.local_alloc %a_ : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory>

    %c = ttng.warp_group_dot %a, %b, %prev_c {maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory> -> tensor<128x128xf32, #mma>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>
  }
  tt.return %loop#2: tensor<128x128xf32, #mma>
}

// CHECK-LABEL: @load_into_lt_4b
tt.func @load_into_lt_4b(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #mma> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #mma>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.local_alloc %a_ : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>
    // Do not pipeline if cp.async would read less than 4 consecutive bytes
    // CHECK: tt.load
    // CHECK-NOT: {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #shared2, #ttg.shared_memory>

    %c = ttng.warp_group_dot %a, %b, %prev_c {maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<32x128xf16, #shared2, #ttg.shared_memory> -> tensor<128x128xf32, #mma>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>
  }
  tt.return %loop#2: tensor<128x128xf32, #mma>
}

// CHECK-LABEL: @intermediate_use
tt.func @intermediate_use(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %c2 = arith.constant dense<2.00> : tensor<32x128xf16, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_2 = arith.mulf %b_ , %c2 : tensor<32x128xf16, #BL>
    %b = ttg.convert_layout %b_2 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load
tt.func @indirect_load(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ind_ptr_init : tensor<32x128x!tt.ptr<i32>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_ind_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %b_ind_ptr = %b_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_off = tt.load %b_ind_ptr : tensor<32x128x!tt.ptr<i32>, #BL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ind_ptr = tt.addptr %b_ind_ptr, %b_ind_off : tensor<32x128x!tt.ptr<i32>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off {tt.divisibility = dense<16> : tensor<32x128xi32>, tt.contiguity = dense<32> : tensor<32x128xi32>, tt.constancy = dense<1> : tensor<32x128xi32>} : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_b_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @mixed_loads
tt.func @mixed_loads(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:4 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#3: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @per_loop_stages
tt.func @per_loop_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> (tensor<128x128xf32, #C>, tensor<128x128xf32, #C>) {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop_cust_stages:4 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init, %l_ptr = %a_ptr_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>, tensor<128x32x!tt.ptr<f16>, #AL>) {
    // CHECK: tt.load {{.*}} {tt.latency = 3 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 3 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    // CHECK: tt.load {{.*}} {tt.latency = 3 : i32}
    %l = tt.load %l_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    "use"(%l) : (tensor<128x32xf16, #AL>) -> ()
    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    %next_l_ptr = tt.addptr %l_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    scf.yield %next_a_ptr, %next_b_ptr, %c, %next_l_ptr : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>, tensor<128x32x!tt.ptr<f16>, #AL>
  } {tt.num_stages = 4 : i32}

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop_cust_stages#2, %loop#2: tensor<128x128xf32, #C>, tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load_cust_stages
tt.func @indirect_load_cust_stages(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ind_ptr_init : tensor<32x128x!tt.ptr<i32>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_ind_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %b_ind_ptr = %b_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_off = tt.load %b_ind_ptr : tensor<32x128x!tt.ptr<i32>, #BL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ind_ptr = tt.addptr %b_ind_ptr, %b_ind_off : tensor<32x128x!tt.ptr<i32>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off {tt.divisibility = dense<16> : tensor<32x128xi32>, tt.contiguity = dense<32> : tensor<32x128xi32>, tt.constancy = dense<1> : tensor<32x128xi32>} : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_b_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 5 : i32}
  tt.return %loop#4: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load_few_stages
tt.func @indirect_load_few_stages(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ind_ptr_init : tensor<32x128x!tt.ptr<i32>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_ind_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %b_ind_ptr = %b_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %b_off = tt.load %b_ind_ptr : tensor<32x128x!tt.ptr<i32>, #BL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ind_ptr = tt.addptr %b_ind_ptr, %b_ind_off : tensor<32x128x!tt.ptr<i32>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off {tt.divisibility = dense<16> : tensor<32x128xi32>, tt.contiguity = dense<32> : tensor<32x128xi32>, tt.constancy = dense<1> : tensor<32x128xi32>} : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_b_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 2 : i32}
  tt.return %loop#4: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @non_dot_pipeline
tt.func @non_dot_pipeline(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x32xf16, #A> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

    %c = arith.addf %a, %prev_c : tensor<128x32xf16, #A>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    scf.yield %next_a_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>
  } {tt.num_stages = 3 : i32}
  tt.return %loop#1: tensor<128x32xf16, #A>
}

// CHECK-LABEL: @no_pipeline
tt.func @no_pipeline(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x32xf16, #A> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

    %c = arith.addf %a, %prev_c : tensor<128x32xf16, #A>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    scf.yield %next_a_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>
  }
  tt.return %loop#1: tensor<128x32xf16, #A>
}

// CHECK-LABEL: @intermediate_use
tt.func @intermediate_use_cust_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %c2 = arith.constant dense<2.00> : tensor<32x128xf16, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_2 = arith.mulf %b_ , %c2 : tensor<32x128xf16, #BL>
    %b = ttg.convert_layout %b_2 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 3 : i32}
  tt.return %loop#2: tensor<128x128xf32, #C>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_overwrite_acc
tt.func @tc_gen5_mma_overwrite_acc(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_acc_use_false
tt.func @tc_gen5_mma_acc_use_false(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %false = arith.constant false
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %false, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_acc_use_false
tt.func @tc_gen5_mma_acc_use_false(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>,
                  %acc_use_init : i1) -> () {
  %true = arith.constant true
  %false = arith.constant false
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %acc_use = arith.xori %acc_use_init, %true : i1
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %acc_use, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_acc_use_false_dist_1
tt.func @tc_gen5_mma_acc_use_false_dist_1(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>,
                  %acc_use_init : i1) -> () {
  %true = arith.constant true
  %false = arith.constant false
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step iter_args(%acc_use = %acc_use_init) -> (i1) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %acc_use, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
    %acc_use_next = arith.xori %acc_use, %true : i1
    scf.yield %acc_use_next : i1
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_acc_use_false_outside_loop
tt.func @tc_gen5_mma_acc_use_false_outside_loop(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>,
                  %acc_use_init : i1) -> () {
  %true = arith.constant true
  %false = arith.constant false
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %acc_use = arith.xori %acc_use_init, %true : i1
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.self_latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %acc_use, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_overwrite_acc_outside_loop
tt.func @tc_gen5_mma_overwrite_acc_outside_loop(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.self_latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_overwrite_acc
tt.func @tc_gen5_mma_overwrite_acc_small_load(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    // CHECK-NOT: tt.self_latency
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_B_outside
tt.func @tc_gen5_mma_B_outside(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked1>,
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_disallow_multibuffer
tt.func @tc_gen5_mma_disallow_multibuffer(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked1>,
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.self_latency = 1 : i32}
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  } {tt.disallow_acc_multi_buffer}
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_B_outside2
tt.func @tc_gen5_mma_B_outside2(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_non_load_operand1
tt.func @tc_gen5_mma_non_load_operand1(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B = "producer"() : () -> tensor<128x128xf16, #blocked1>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_non_load_operand2
tt.func @tc_gen5_mma_non_load_operand2(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = "producer"() : () -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    // CHECK-NOT: tt.self_latency
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @select_after_mma
  tt.func public @select_after_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = "cnd"() : () -> i1
    %1 = ttng.tmem_alloc  : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %1, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32  : i32 {
      %4 = tt.load %arg0 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg1 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %7 = ttg.local_alloc %6 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
      ttng.tc_gen5_mma %5, %7, %1, %true, %true : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.xori %0, %true : i1
      ttng.tmem_store %cst_0, %1, %8 : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    } {tt.scheduled_max_stage = 3 : i32}
    %2 = ttng.tmem_load %1 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %3 = arith.truncf %2 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %3 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_scaled
tt.func @tc_gen5_mma_scaled(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %A_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>

    %A_sc = tt.load %A_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %A_sc_sh = ttg.local_alloc %A_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

    %B_sc = tt.load %B_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %B_sc_sh = ttg.local_alloc %B_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm, %A_sc_sh, %B_sc_sh, %true, %true lhs = e5m2 rhs = e5m2 : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#scales = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_scaled_tmem_scales
tt.func @tc_gen5_mma_scaled_tmem_scales(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %A_sc_ptr: tensor<128x8x!tt.ptr<i8>, #scales> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_sc_ptr: tensor<128x8x!tt.ptr<i8>, #scales> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>

    %A_sc = tt.load %A_sc_ptr : tensor<128x8x!tt.ptr<i8>, #scales>
    %A_sc_sh = ttg.local_alloc %A_sc : (tensor<128x8xi8, #scales>) -> !ttg.memdesc<128x8xi8, #shared1, #smem>

    %B_sc = tt.load %B_sc_ptr : tensor<128x8x!tt.ptr<i8>, #scales>
    %B_sc_tm = ttng.tmem_alloc %B_sc : (tensor<128x8xi8, #scales>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>

    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}
    // CHECK-NOT: tt.latency
    // CHECK-NOT: tt.self_latency
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm, %A_sc_sh, %B_sc_tm, %true, %true lhs = e5m2 rhs = e5m2 : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #shared1, #smem>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @block_scale_mxfp_matmul
  tt.func public @block_scale_mxfp_matmul(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<4> : tensor<128x256xi32, #blocked1>
    %cst_1 = arith.constant dense<4> : tensor<256x128xi32, #blocked2>
    %cst_2 = arith.constant dense<4> : tensor<1x2x32x4x4xi32, #blocked3>
    %0 = tt.splat %arg3 : !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>
    %1 = tt.splat %arg4 : !tt.ptr<f8E5M2> -> tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>
    %2 = tt.splat %arg5 : !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
    %3 = tt.splat %arg6 : !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
    %4 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %6 = tt.broadcast %5 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %7 = tt.addptr %0, %6 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<128x256xi32, #blocked1>
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %10 = tt.broadcast %9 : tensor<1x128xi32, #blocked2> -> tensor<256x128xi32, #blocked2>
    %11 = tt.addptr %1, %10 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<256x128xi32, #blocked2>
    %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>}>}>>
    %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>}>}>> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>}>> -> tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>>
    %15 = tt.expand_dims %14 {axis = 2 : i32} : tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>> -> tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #blocked3}>>
    %16 = tt.expand_dims %15 {axis = 3 : i32} : tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #blocked3}>> -> tensor<1x1x1x1x4xi32, #blocked3>
    %17 = tt.broadcast %16 : tensor<1x1x1x1x4xi32, #blocked3> -> tensor<1x2x32x4x4xi32, #blocked3>
    %18 = tt.addptr %2, %17 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4xi32, #blocked3>
    %19 = tt.addptr %3, %17 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4xi32, #blocked3>
    %20 = ttng.tmem_alloc  : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %20, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %21:4 = scf.for %arg7 = %arg0 to %arg1 step %arg2 iter_args(%arg8 = %7, %arg9 = %11, %arg10 = %18, %arg11 = %19) -> (tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>) {
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %22 = tt.load %arg8 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>
      %23 = ttg.local_alloc %22 : (tensor<128x256xf8E5M2, #blocked1>) -> !ttg.memdesc<128x256xf8E5M2, #shared, #smem>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %24 = tt.load %arg9 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>
      %25 = ttg.local_alloc %24 : (tensor<256x128xf8E5M2, #blocked2>) -> !ttg.memdesc<256x128xf8E5M2, #shared, #smem>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %26 = tt.load %arg10 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %27 = tt.load %arg11 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
      %28 = ttg.local_alloc %26 : (tensor<1x2x32x4x4xi8, #blocked3>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      %29 = ttg.local_alloc %27 : (tensor<1x2x32x4x4xi8, #blocked3>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      // CHECK: ttng.tc_gen5_mma_scaled {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
      ttng.tc_gen5_mma_scaled %23, %25, %20, %28, %29, %true, %true lhs = e5m2 rhs = e5m2 : !ttg.memdesc<128x256xf8E5M2, #shared, #smem>, !ttg.memdesc<256x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      %30 = tt.addptr %arg8, %cst_0 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<128x256xi32, #blocked1>
      %31 = tt.addptr %arg9, %cst_1 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<256x128xi32, #blocked2>
      %32 = tt.addptr %arg10, %cst_2 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4xi32, #blocked3>
      %33 = tt.addptr %arg11, %cst_2 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4xi32, #blocked3>
      scf.yield %30, %31, %32, %33 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
    } {tt.num_stages = 3 : i32}
    tt.return %cst : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @two_dots
  tt.func public @two_dots(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg4: i32) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ttng.tmem_alloc  : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_alloc  : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32  : i32 {
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %2 = tt.load %arg0 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %3 = ttg.local_alloc %2 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %4 = tt.load %arg1 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg2 : tensor<128x128x!tt.ptr<f32>, #blocked>
      ttng.tmem_store %6, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
      ttng.tc_gen5_mma %3, %5, %0, %true, %true : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %7 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      ttng.tmem_store %7, %1, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
      ttng.tc_gen5_mma %3, %5, %1, %true, %true : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = ttng.tmem_load %1 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      tt.store %arg3, %8 : tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @changed_acc_before_mma
  tt.func public @changed_acc_before_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %0[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      %3 = tt.load %arg0 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %4 = ttg.local_alloc %3 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %5 = tt.load %arg1 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %7, %load_tok = ttng.tmem_load %0[%tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %8 = arith.mulf %7, %cst_0 : tensor<128x128xf32, #blocked1>
      %store_tok = ttng.tmem_store %8, %0[%load_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
      %mma_tok = ttng.tc_gen5_mma %4, %6, %0[%store_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32}
    %1, %res_tok = ttng.tmem_load %0[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %2 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>

#smem = #ttg.shared_memory
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem_lhs = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @attention_forward
tt.func public @attention_forward(
  %Q_shared: !ttg.memdesc<256x64xf16, #shared, #smem>,
  %K_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
  %V_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
  %qk_scale: f32,
  %n_tiles: i32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32

  %neg_inf = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %zero = arith.constant dense<0.0> : tensor<256x64xf32, #blocked>
  %one = arith.constant dense<1.0> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

  %QK_tmem, %QK_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)

  %loop_outs:3 = scf.for %i = %c0_i32 to %n_tiles step %c64_i32 iter_args(
    %l_i = %one,
    %acc = %zero,
    %m_i = %neg_inf
  ) -> (
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
    tensor<256x64xf32, #blocked>,
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  ) : i32 {
    // CHECK: descriptor_load {{.*}} {tt.latency = 2 : i32}
    %K = tt.descriptor_load %K_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %K_shared = ttg.local_alloc %K : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %K_trans = ttg.memdesc_trans %K_shared {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared_T, #smem>
    // CHECK: tc_gen5_mma {{.*}} {tt.latency = 2 : i32, tt.self_latency = 1 : i32}
    %QK_mma_tok = ttng.tc_gen5_mma %Q_shared, %K_trans, %QK_tmem[%QK_tok], %false, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_T, #smem>, !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    %QK, %QK_load_tok = ttng.tmem_load %QK_tmem[%QK_mma_tok] : !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

    %alpha_1, %P, %next_l_i, %row_max = "softmax_work"(%QK, %l_i, %m_i, %qk_scale) : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> (tensor<256x64xf32, #blocked>, tensor<256x64xf16, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)

    %acc_corrected = arith.mulf %acc, %alpha_1 : tensor<256x64xf32, #blocked>

    // CHECK: descriptor_load {{.*}} {tt.latency = 2 : i32}
    %V = tt.descriptor_load %V_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %V_shared = ttg.local_alloc %V : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %P_tmem = ttng.tmem_alloc %P : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem_lhs, #ttng.tensor_memory>
    %acc_tmem, %acc_tok = ttng.tmem_alloc %acc_corrected : (tensor<256x64xf32, #blocked>) -> (!ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: tc_gen5_mma {{.*}} {tt.self_latency = 1 : i32}
    %PV_mma_tok = ttng.tc_gen5_mma %P_tmem, %V_shared, %acc_tmem[%acc_tok], %true, %true : !ttg.memdesc<256x64xf16, #tmem_lhs, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    %O, %O_tok = ttng.tmem_load %acc_tmem[%PV_mma_tok] : !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

    scf.yield %next_l_i, %O, %row_max : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  } {tt.warp_specialize}

  "use"(%loop_outs#0, %loop_outs#1, %loop_outs#2) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()

  tt.return
}

}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @_attn_fwd_persist
  tt.func public @_attn_fwd_persist(%sm_scale: f32 , %M: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %Z: i32 , %H: i32 {tt.divisibility = 16 : i32} , %desc_q: !tt.tensordesc<tensor<128x128xbf16, #shared>> , %desc_q_0: i32 , %desc_q_1: i32 , %desc_q_2: i64 , %desc_q_3: i64 , %desc_k: !tt.tensordesc<tensor<128x128xbf16, #shared>> , %desc_k_4: i32 , %desc_k_5: i32 , %desc_k_6: i64 , %desc_k_7: i64 , %desc_v: !tt.tensordesc<tensor<128x128xbf16, #shared>> , %desc_v_8: i32 , %desc_v_9: i32 , %desc_v_10: i64 , %desc_v_11: i64 , %desc_o: !tt.tensordesc<tensor<128x128xbf16, #shared>> , %desc_o_12: i32 , %desc_o_13: i32 , %desc_o_14: i64 , %desc_o_15: i64 , %N_CTX: i32 {tt.divisibility = 16 : i32})  attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %n_tile_num = arith.constant 255 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_17 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_18 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %n_tile_num_19 = arith.addi %N_CTX, %n_tile_num : i32
    %n_tile_num_20 = arith.divsi %n_tile_num_19, %c256_i32 : i32
    %prog_id = tt.get_program_id x : i32
    %num_progs = tt.get_num_programs x : i32
    %total_tiles = arith.muli %n_tile_num_20, %Z : i32
    %total_tiles_21 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_21, %num_progs : i32
    %0 = arith.remsi %total_tiles_21, %num_progs : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_22 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_22 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %offset_y = arith.muli %N_CTX, %H : i32
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %offs_m1 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked1>
    %qk_scale = arith.mulf %sm_scale, %cst : f32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #blocked>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_22 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_22, %n_tile_num_20 : i32
      %off_hz = arith.divsi %tile_idx_22, %n_tile_num_20 : i32
      %off_z = arith.divsi %off_hz, %H : i32
      %off_h = arith.remsi %off_hz, %H : i32
      %offset_y_23 = arith.muli %off_z, %offset_y : i32
      %offset_y_24 = arith.muli %off_h, %N_CTX : i32
      %offset_y_25 = arith.addi %offset_y_23, %offset_y_24 : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 : i32
      %qo_offset_y_26 = arith.addi %offset_y_25, %qo_offset_y : i32
      %offs_m0_27 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #blocked1>
      %offs_m0_28 = arith.addi %offs_m0_27, %offs_m0 : tensor<128xi32, #blocked1>
      %offs_m1_29 = arith.addi %offs_m0_27, %offs_m1 : tensor<128xi32, #blocked1>
      // CHECK: descriptor_load {{.*}} {tt.latency = 1 : i32}
      %q0 = tt.descriptor_load %desc_q[%qo_offset_y_26, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
      %q0_30 = ttg.local_alloc %q0 : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %q1 = arith.addi %qo_offset_y_26, %c128_i32 : i32
      // CHECK: descriptor_load {{.*}} {tt.latency = 0 : i32}
      %q1_31 = tt.descriptor_load %desc_q[%q1, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
      %q1_32 = ttg.local_alloc %q1_31 : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %qk_33, %qk_34 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc, %acc_35 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %qk_36, %qk_37 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_38, %acc_39 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_40 = ttng.tmem_store %cst_16, %acc_38[%acc_39], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_41 = ttng.tmem_store %cst_16, %acc[%acc_35], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y:10 = scf.for %offsetkv_y_56 = %c0_i32 to %N_CTX step %c128_i32 iter_args(%arg28 = %cst_18, %arg29 = %cst_18, %arg30 = %cst_17, %arg31 = %cst_17, %offset_y_57 = %offset_y_25, %arg33 = %false, %qk_58 = %qk_34, %acc_59 = %acc_41, %qk_60 = %qk_37, %acc_61 = %acc_40) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %k = tt.descriptor_load %desc_k[%offset_y_57, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
        %k_62 = ttg.local_alloc %k : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %k_63 = ttg.memdesc_trans %k_62 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
        %v = tt.descriptor_load %desc_v[%offset_y_57, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
        %v_64 = ttg.local_alloc %v : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %qk_65 = ttng.tc_gen5_mma %q0_30, %k_63, %qk_33[%qk_58], %false, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_66, %qk_67 = ttng.tmem_load %qk_33[%qk_65] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %m_ij_68 = "tt.reduce"(%qk_66) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_112: f32 , %m_ij_113: f32) :
          %m_ij_114 = arith.maxnumf %m_ij_112, %m_ij_113 : f32
          tt.reduce.return %m_ij_114 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_69 = arith.mulf %m_ij_68, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_70 = arith.maxnumf %arg30, %m_ij_69 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_71 = arith.mulf %qk_66, %qk : tensor<128x128xf32, #blocked>
        %qk_72 = tt.expand_dims %m_ij_70 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_73 = tt.broadcast %qk_72 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_74 = arith.subf %qk_71, %qk_73 : tensor<128x128xf32, #blocked>
        %p = math.exp2 %qk_74 : tensor<128x128xf32, #blocked>
        %alpha = arith.subf %arg30, %m_ij_70 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_75 = math.exp2 %alpha : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_112: f32 , %l_ij_113: f32) :
          %l_ij_114 = arith.addf %l_ij_112, %l_ij_113 : f32
          tt.reduce.return %l_ij_114 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_76 = tt.expand_dims %alpha_75 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc_77 = tt.broadcast %acc_76 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %acc_78, %acc_79 = ttng.tmem_load %acc[%acc_59] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc_80 = arith.mulf %acc_78, %acc_77 : tensor<128x128xf32, #blocked>
        %p_81 = arith.truncf %p : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %acc_82 = ttng.tmem_alloc %p_81 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>
        %acc_83 = ttng.tmem_store %acc_80, %acc[%acc_79], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_84 = ttng.tc_gen5_mma %acc_82, %v_64, %acc[%acc_83], %arg33, %true : !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i = arith.mulf %arg28, %alpha_75 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i_85 = arith.addf %l_i, %l_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_86 = ttng.tc_gen5_mma %q1_32, %k_63, %qk_36[%qk_60], %false, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_87, %qk_88 = ttng.tmem_load %qk_36[%qk_86] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %m_ij_89 = "tt.reduce"(%qk_87) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_112: f32 , %m_ij_113: f32) :
          %m_ij_114 = arith.maxnumf %m_ij_112, %m_ij_113 : f32
          tt.reduce.return %m_ij_114 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_90 = arith.mulf %m_ij_89, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_91 = arith.maxnumf %arg31, %m_ij_90 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_92 = arith.mulf %qk_87, %qk : tensor<128x128xf32, #blocked>
        %qk_93 = tt.expand_dims %m_ij_91 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_94 = tt.broadcast %qk_93 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_95 = arith.subf %qk_92, %qk_94 : tensor<128x128xf32, #blocked>
        %p_96 = math.exp2 %qk_95 : tensor<128x128xf32, #blocked>
        %alpha_97 = arith.subf %arg31, %m_ij_91 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_98 = math.exp2 %alpha_97 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij_99 = "tt.reduce"(%p_96) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_112: f32 , %l_ij_113: f32) :
          %l_ij_114 = arith.addf %l_ij_112, %l_ij_113 : f32
          tt.reduce.return %l_ij_114 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_100 = tt.expand_dims %alpha_98 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc_101 = tt.broadcast %acc_100 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %acc_102, %acc_103 = ttng.tmem_load %acc_38[%acc_61] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc_104 = arith.mulf %acc_102, %acc_101 : tensor<128x128xf32, #blocked>
        %p_105 = arith.truncf %p_96 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %acc_106 = ttng.tmem_alloc %p_105 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>
        %acc_107 = ttng.tmem_store %acc_104, %acc_38[%acc_103], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_108 = ttng.tc_gen5_mma %acc_106, %v_64, %acc_38[%acc_107], %arg33, %true : !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i_109 = arith.mulf %arg29, %alpha_98 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i_110 = arith.addf %l_i_109, %l_ij_99 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %offsetkv_y_111 = arith.addi %offset_y_57, %c128_i32 : i32
        scf.yield %l_i_85, %l_i_110, %m_ij_70, %m_ij_91, %offsetkv_y_111, %true, %qk_67, %acc_84, %qk_88, %acc_108 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {tt.disallow_acc_multi_buffer}
      %m_i0 = math.log2 %offsetkv_y#0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_42 = arith.addf %offsetkv_y#2, %m_i0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc0 = tt.expand_dims %offsetkv_y#0 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_43 = tt.broadcast %acc0 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc_44, %acc_45 = ttng.tmem_load %acc[%offsetkv_y#7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc0_46 = arith.divf %acc_44, %acc0_43 : tensor<128x128xf32, #blocked>
      %m_ptrs0 = arith.muli %off_hz, %N_CTX : i32
      %m_ptrs0_47 = tt.addptr %M, %m_ptrs0 : !tt.ptr<f32>, i32
      %m_ptrs0_48 = tt.splat %m_ptrs0_47 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
      %m_ptrs0_49 = tt.addptr %m_ptrs0_48, %offs_m0_28 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      %3 = ttg.convert_layout %m_i0_42 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      tt.store %m_ptrs0_49, %3 : tensor<128x!tt.ptr<f32>, #blocked1>
      %4 = arith.truncf %acc0_46 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %5 = ttg.convert_layout %4 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked2>
      tt.descriptor_store %desc_o[%qo_offset_y_26, %c0_i32], %5 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked2>
      %m_i1 = math.log2 %offsetkv_y#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i1_50 = arith.addf %offsetkv_y#3, %m_i1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc1 = tt.expand_dims %offsetkv_y#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc1_51 = tt.broadcast %acc1 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc_52, %acc_53 = ttng.tmem_load %acc_38[%offsetkv_y#9] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc1_54 = arith.divf %acc_52, %acc1_51 : tensor<128x128xf32, #blocked>
      %m_ptrs1 = tt.addptr %m_ptrs0_48, %offs_m1_29 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      %6 = ttg.convert_layout %m_i1_50 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      tt.store %m_ptrs1, %6 : tensor<128x!tt.ptr<f32>, #blocked1>
      %7 = arith.truncf %acc1_54 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %8 = ttg.convert_layout %7 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked2>
      tt.descriptor_store %desc_o[%q1, %c0_i32], %8 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked2>
      %tile_idx_55 = arith.addi %tile_idx_22, %num_progs : i32
      scf.yield %tile_idx_55 : i32
    } {tt.warp_specialize}
    tt.return
  }
}
