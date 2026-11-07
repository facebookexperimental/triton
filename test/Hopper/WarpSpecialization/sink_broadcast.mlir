// RUN: triton-opt %s -split-input-file --nvgpu-sink-broadcast | FileCheck %s

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 128]], warp = [[16, 0], [32, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @sink_after_tmem_load
  tt.func @sink_after_tmem_load(%bias: tensor<1x256xf32, #linear>,
                                %acc: !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>)
                                -> tensor<64x256xf32, #linear> {
    // CHECK: %[[LOAD:.*]] = ttng.tmem_load
    // CHECK-NEXT: %[[BCAST:.*]] = tt.broadcast %{{.*}} : tensor<1x256xf32, #linear> -> tensor<64x256xf32, #linear>
    // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[LOAD]], %[[BCAST]]
    %bcast = tt.broadcast %bias : tensor<1x256xf32, #linear> -> tensor<64x256xf32, #linear>
    %load = ttng.tmem_load %acc : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
    %add = arith.addf %load, %bcast : tensor<64x256xf32, #linear>
    tt.return %add : tensor<64x256xf32, #linear>
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 128]], warp = [[16, 0], [32, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @sink_broadcast_chain_through_descriptor_load
  tt.func @sink_broadcast_chain_through_descriptor_load(%bias_desc: !tt.tensordesc<1x256xf16>,
                                                        %acc: !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>)
                                                        -> tensor<64x256xf32, #linear> {
    // CHECK: %[[TMEM:.*]] = ttng.tmem_load
    // CHECK-NEXT: %[[BIAS:.*]] = tt.descriptor_load
    // CHECK-NEXT: %[[EXT:.*]] = arith.extf %[[BIAS]]
    // CHECK-NEXT: %[[BCAST:.*]] = tt.broadcast %[[EXT]]
    // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[TMEM]], %[[BCAST]]
    %c0 = arith.constant 0 : i32
    %bias = tt.descriptor_load %bias_desc[%c0, %c0] : !tt.tensordesc<1x256xf16> -> tensor<1x256xf16>
    %bias_f32 = arith.extf %bias : tensor<1x256xf16> to tensor<1x256xf32>
    %bcast = tt.broadcast %bias_f32 : tensor<1x256xf32> -> tensor<64x256xf32, #linear>
    %load = ttng.tmem_load %acc : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
    %add = arith.addf %load, %bcast : tensor<64x256xf32, #linear>
    tt.return %add : tensor<64x256xf32, #linear>
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 128]], warp = [[16, 0], [32, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @sink_multiple_broadcast_operands
  tt.func @sink_multiple_broadcast_operands(%lhs: tensor<64x1xf32, #linear>,
                                            %rhs: tensor<1x256xf32, #linear>,
                                            %x: tensor<64x256xf32, #linear>)
                                            -> tensor<64x256xf32, #linear> {
    // CHECK: %[[PREP:.*]] = arith.mulf
    // CHECK-NEXT: %[[LHS:.*]] = tt.broadcast
    // CHECK-NEXT: %[[RHS:.*]] = tt.broadcast
    // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[LHS]], %[[RHS]]
    // CHECK-NEXT: %[[SUB:.*]] = arith.subf %[[ADD]], %[[PREP]]
    %lhs_bcast = tt.broadcast %lhs : tensor<64x1xf32, #linear> -> tensor<64x256xf32, #linear>
    %rhs_bcast = tt.broadcast %rhs : tensor<1x256xf32, #linear> -> tensor<64x256xf32, #linear>
    %prep = arith.mulf %x, %x : tensor<64x256xf32, #linear>
    %add = arith.addf %lhs_bcast, %rhs_bcast : tensor<64x256xf32, #linear>
    %sub = arith.subf %add, %prep : tensor<64x256xf32, #linear>
    tt.return %sub : tensor<64x256xf32, #linear>
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 128]], warp = [[16, 0], [32, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @clone_for_shared_broadcast
  tt.func @clone_for_shared_broadcast(%bias: tensor<1x256xf32, #linear>,
                                      %x: tensor<64x256xf32, #linear>,
                                      %acc: !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>)
                                      -> tensor<64x256xf32, #linear> {
    // CHECK: %[[CLONE:.*]] = tt.broadcast
    // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[CLONE]], %{{.*}}
    // CHECK: %[[LOAD:.*]] = ttng.tmem_load
    // CHECK-NEXT: %[[ORIG:.*]] = tt.broadcast
    // CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[LOAD]], %[[ORIG]]
    %bcast = tt.broadcast %bias : tensor<1x256xf32, #linear> -> tensor<64x256xf32, #linear>
    %add = arith.addf %bcast, %x : tensor<64x256xf32, #linear>
    %load = ttng.tmem_load %acc : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
    %mul = arith.mulf %load, %bcast : tensor<64x256xf32, #linear>
    %ret = arith.addf %add, %mul : tensor<64x256xf32, #linear>
    tt.return %ret : tensor<64x256xf32, #linear>
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 128]], warp = [[16, 0], [32, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @do_not_sink_unary_op
  tt.func @do_not_sink_unary_op(%bias: tensor<1x256xf32, #linear>,
                                %x: tensor<64x256xf32, #linear>)
                                -> tensor<64x256xf32, #linear> {
    // CHECK: %[[BCAST:.*]] = tt.broadcast
    // CHECK-NEXT: %[[ABS:.*]] = math.absf %[[BCAST]]
    // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[ABS]], %{{.*}}
    %bcast = tt.broadcast %bias : tensor<1x256xf32, #linear> -> tensor<64x256xf32, #linear>
    %abs = math.absf %bcast : tensor<64x256xf32, #linear>
    %add = arith.addf %abs, %x : tensor<64x256xf32, #linear>
    tt.return %add : tensor<64x256xf32, #linear>
  }
}
