// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=90 -reconcile-unrealized-casts | FileCheck %s

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: init_barrier
  tt.func @init_barrier(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}

// -----

// Test that tensor select with warp-uniform condition (from vote_ballot via splat)
// is converted to branches instead of per-element select instructions.
// This is the pattern used in Flash Attention for conditional rescaling.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: uniform_tensor_select_to_branch
  // CHECK: nvvm.vote.sync  ballot
  // CHECK: llvm.icmp "ne"
  // CHECK: llvm.cond_br
  // CHECK: llvm.br
  // CHECK: llvm.br
  tt.func @uniform_tensor_select_to_branch(%mask: i32, %pred: i1, %true_val: tensor<16x32xf32, #blocked>, %false_val: tensor<16x32xf32, #blocked>, %ptr: !tt.ptr<f32>) {
    // Get warp-uniform ballot result
    %ballot = ttng.vote_ballot_sync %mask, %pred : i1 -> i32
    %c0 = arith.constant 0 : i32
    // Compare ballot result (scalar i32) - this is warp-uniform
    %scalar_cond = arith.cmpi ne, %ballot, %c0 : i32
    // Splat scalar condition to tensor shape to match tensor operands
    %cond = tt.splat %scalar_cond : i1 -> tensor<16x32xi1, #blocked>
    // Select with uniform tensor condition - should become branches
    %result = arith.select %cond, %true_val, %false_val : tensor<16x32xi1, #blocked>, tensor<16x32xf32, #blocked>
    // Store result (kernels can't return values)
    %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %result : tensor<16x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Test the full Flash Attention pattern: tensor predicate -> vote_ballot -> tensor condition -> select
// This matches the actual FA kernel pattern where pred = alpha_1 < 1.0 is a tensor.
#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: uniform_tensor_select_tensor_pred
  // CHECK: nvvm.vote.sync  ballot
  // CHECK: llvm.icmp "ne"
  // CHECK: llvm.cond_br
  // CHECK: llvm.br
  // CHECK: llvm.br
  tt.func @uniform_tensor_select_tensor_pred(%mask: i32, %alpha: tensor<128xf32, #blocked1d>, %acc: tensor<128xf32, #blocked1d>, %scaled_acc: tensor<128xf32, #blocked1d>, %ptr: !tt.ptr<f32>) {
    // pred = alpha < 1.0 - this is a tensor predicate
    %c1 = arith.constant dense<1.0> : tensor<128xf32, #blocked1d>
    %pred = arith.cmpf olt, %alpha, %c1 : tensor<128xf32, #blocked1d>
    // ballot_result is a tensor with the same shape, all elements contain warp ballot
    %ballot = ttng.vote_ballot_sync %mask, %pred : tensor<128xi1, #blocked1d> -> tensor<128xi32, #blocked1d>
    // should_rescale = ballot_result != 0
    %c0 = arith.constant dense<0> : tensor<128xi32, #blocked1d>
    %should_rescale = arith.cmpi ne, %ballot, %c0 : tensor<128xi32, #blocked1d>
    // Conditional select - condition is uniform since ballot result is same for all threads in warp
    %result = arith.select %should_rescale, %scaled_acc, %acc : tensor<128xi1, #blocked1d>, tensor<128xf32, #blocked1d>
    // Store result (kernels can't return values)
    %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1d>
    tt.store %ptrs, %result : tensor<128x!tt.ptr<f32>, #blocked1d>
    tt.return
  }
}

// -----

// Test 2D Flash Attention pattern: alpha is 128x1 (broadcast dim), acc/scaled_acc are 128x64
// This tests the broadcast scenario where alpha has a singleton dimension.
#blocked2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2d_alpha = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: uniform_tensor_select_2d_broadcast
  // CHECK: nvvm.vote.sync  ballot
  // CHECK: llvm.icmp "ne"
  // CHECK: llvm.cond_br
  // CHECK: llvm.br
  // CHECK: llvm.br
  tt.func @uniform_tensor_select_2d_broadcast(%mask: i32, %alpha: tensor<128x1xf32, #blocked2d_alpha>, %acc: tensor<128x64xf32, #blocked2d>, %scaled_acc: tensor<128x64xf32, #blocked2d>, %ptr: !tt.ptr<f32>) {
    // pred = alpha < 1.0 - alpha is 128x1, will be broadcast
    %c1 = arith.constant dense<1.0> : tensor<128x1xf32, #blocked2d_alpha>
    %pred = arith.cmpf olt, %alpha, %c1 : tensor<128x1xf32, #blocked2d_alpha>
    // ballot_result has same shape as pred (128x1)
    %ballot = ttng.vote_ballot_sync %mask, %pred : tensor<128x1xi1, #blocked2d_alpha> -> tensor<128x1xi32, #blocked2d_alpha>
    // should_rescale = ballot_result != 0 (128x1)
    %c0 = arith.constant dense<0> : tensor<128x1xi32, #blocked2d_alpha>
    %cond_small = arith.cmpi ne, %ballot, %c0 : tensor<128x1xi32, #blocked2d_alpha>
    // Broadcast condition from 128x1 to 128x64 to match acc/scaled_acc shape
    %should_rescale = tt.broadcast %cond_small : tensor<128x1xi1, #blocked2d_alpha> -> tensor<128x64xi1, #blocked2d>
    // Conditional select with broadcast condition
    %result = arith.select %should_rescale, %scaled_acc, %acc : tensor<128x64xi1, #blocked2d>, tensor<128x64xf32, #blocked2d>
    // Store result
    %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked2d>
    tt.store %ptrs, %result : tensor<128x64x!tt.ptr<f32>, #blocked2d>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: wait_barrier
  tt.func @wait_barrier(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %phase: i32, %pred: i1) {
    // CHECK: waitLoop:
    // CHECK: mbarrier.try_wait.parity.shared.b64
    // CHECK: @!complete bra.uni waitLoop
    // CHECK-NOT: skipWait
    // CHECK: %{{[0-9]+}}, %arg1 :
    ttng.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared0, #smem>
    %true = arith.constant true

    // CHECK: waitLoop:
    // CHECK: mbarrier.try_wait.parity.shared.b64
    // CHECK: @!complete bra.uni waitLoop
    // CHECK-NOT: skipWait
    // CHECK: %{{[0-9]+}}, %arg1 :
    ttng.wait_barrier %alloc, %phase, %true : !ttg.memdesc<1xi64, #shared0, #smem>

    // CHECK: @!$2 bra.uni skipWait
    // CHECK: waitLoop:
    // CHECK: mbarrier.try_wait.parity.shared.b64
    // CHECK: @!complete bra.uni waitLoop
    // CHECK: skipWait:
    // CHECK: %{{[0-9]+}}, %arg1, %arg2 :
    ttng.wait_barrier %alloc, %phase, %pred : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }

  // CHECK-LABEL: arrive_barrier
  tt.func @arrive_barrier(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK-NEXT: [[TID:%.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK-NEXT: [[C127:%.*]] = llvm.mlir.constant(127 : i32)
    // CHECK-NEXT: [[RTID:%.*]] = llvm.and [[TID]], [[C127]]
    // CHECK-NEXT: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK-NEXT: [[IS_ZERO:%.*]] = llvm.icmp "eq" [[RTID]], [[C0]]
    // CHECK-NEXT: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1], 2;", "b,r" [[IS_ZERO]], %arg0
    ttng.arrive_barrier %alloc, 2 : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }

  // CHECK-LABEL: arrive_barrier_pred
  tt.func @arrive_barrier_pred(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    // CHECK-NEXT: [[TID:%.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK-NEXT: [[C127:%.*]] = llvm.mlir.constant(127 : i32)
    // CHECK-NEXT: [[RTID:%.*]] = llvm.and [[TID]], [[C127]]
    // CHECK-NEXT: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK-NEXT: [[IS_ZERO:%.*]] = llvm.icmp "eq" [[RTID]], [[C0]]
    // CHECK-NEXT: [[PRED:%.*]] = llvm.and [[IS_ZERO]], %arg1
    // CHECK-NEXT: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1], 2;", "b,r" [[PRED]], %arg0
    ttng.arrive_barrier %alloc, 2, %pred : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }

  // CHECK-LABEL: arrive_barrier_named
  tt.func @arrive_barrier_named(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    %c9_i32 = arith.constant 9 : i32
    %c256_i32 = arith.constant 256 : i32
    // CHECK-NEXT: [[BAR_ID:%.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK-NEXT: [[NUM_THRADS:%.*]] = llvm.mlir.constant(256 : i32) : i32
    // CHECK-NEXT: "bar.arrive $0, $1;", "r,r" [[BAR_ID]], [[NUM_THRADS]]
    ttng.arrive_barrier_named %c9_i32, %c256_i32 : i32, i32
    tt.return
  }

  // CHECK-LABEL: arrive_barrier_remote
  tt.func @arrive_barrier_remote(%alloc: !ttg.memdesc<1xi64, #shared0, #ttng.shared_cluster_memory>, %pred: i1) {
    // CHECK: "@$0 mbarrier.arrive.shared::cluster.b64 _, [$1], 2;", "b,r" %{{.*}}
    ttng.arrive_barrier %alloc, 2, %pred : !ttg.memdesc<1xi64, #shared0, #ttng.shared_cluster_memory>
    tt.return
  }

  // CHECK-LABEL: wait_barrier_named
  tt.func @wait_barrier_named(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    %c9_i32 = arith.constant 9 : i32
    %c256_i32 = arith.constant 256 : i32
    // CHECK-NEXT: [[BAR_ID:%.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK-NEXT: [[NUM_THRADS:%.*]] = llvm.mlir.constant(256 : i32) : i32
    // CHECK-NEXT: "bar.sync $0, $1;", "r,r" [[BAR_ID]], [[NUM_THRADS]]
    ttng.wait_barrier_named %c9_i32, %c256_i32 : i32, i32
    tt.return
  }

}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: async_clc_try_cancel
  // CHECK: clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128
  tt.func @async_clc_try_cancel(%alloc: !ttg.memdesc<1xi64, #shared0, #smem, mutable>, %clc_response: !ttg.memdesc<1xui128, #shared0, #smem, mutable>) {
    ttng.async_clc_try_cancel %alloc, %clc_response : !ttg.memdesc<1xi64, #shared0, #smem, mutable>, !ttg.memdesc<1xui128, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_query_cancel
  // CHECK: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128
  // CHECK: clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
  tt.func @clc_query_cancel(%clc_response: !ttg.memdesc<1xui128, #shared0, #smem, mutable>) {
    %x = ttng.clc_query_cancel %clc_response : (!ttg.memdesc<1xui128, #shared0, #smem, mutable>) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: vote_ballot_sync
  // CHECK: nvvm.vote.sync  ballot
  tt.func @vote_ballot_sync(%mask: i32, %pred: i1) {
    %result = ttng.vote_ballot_sync %mask, %pred : i1 -> i32
    tt.return
  }
}

// -----

// Test that scalar select with warp-uniform condition (from vote_ballot) is
// converted to branches instead of select instruction.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: uniform_select_to_branch
  // CHECK: nvvm.vote.sync  ballot
  // CHECK: llvm.icmp "ne"
  // CHECK: llvm.cond_br
  // CHECK: llvm.br
  // CHECK: llvm.br
  tt.func @uniform_select_to_branch(%mask: i32, %pred: i1, %true_val: i32, %false_val: i32, %ptr: !tt.ptr<i32>) {
    %ballot = ttng.vote_ballot_sync %mask, %pred : i1 -> i32
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi ne, %ballot, %c0 : i32
    %result = arith.select %cond, %true_val, %false_val : i32
    // Store result (kernels can't return values)
    tt.store %ptr, %result : !tt.ptr<i32>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tma_copy_global_to_local
  // CHECK: elect.sync
  // CHECK: "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" {{.*}} : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
  // CHECK-NOT: cp.async.bulk.tensor.2d.shared
  // CHECK: return
  tt.func @tma_copy_global_to_local(%tma: !tt.tensordesc<tensor<128x128xf32, #shared1>>, %alloc: !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    ttng.async_tma_copy_global_to_local %tma[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<tensor<128x128xf32, #shared1>>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tma_copy_local_to_global
  // CHECK: elect.sync
  // CHECK: "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" {{.*}} : (i1, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
  // CHECK-NOT: cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
  // CHECK: nvvm.cp.async.bulk.commit.group
  tt.func @tma_copy_local_to_global(%tma: !tt.tensordesc<tensor<128x128xf32, #shared1>>, %alloc: !ttg.memdesc<128x128xf32, #shared1, #smem>, %x: i32) {
    ttng.async_tma_copy_local_to_global %tma[%x, %x] %alloc : !tt.tensordesc<tensor<128x128xf32, #shared1>>, !ttg.memdesc<128x128xf32, #shared1, #smem>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: async_tma_store_wait
  // CHECK: nvvm.cp.async.bulk.wait_group 0 {read}
  tt.func @async_tma_store_wait() {
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: expect_barrier
  // CHECK: @$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 16384;
  tt.func @expect_barrier(%barrier: !ttg.memdesc<1xi64, #shared0, #smem, mutable>, %pred: i1) {
    ttng.barrier_expect %barrier, 16384, %pred : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: byval_tma_desc
  // CHECK: llvm.align = 64
  // CHECK: llvm.byval = !llvm.array<128 x i8>
  // CHECK: nvvm.grid_constant
  tt.func @byval_tma_desc(%desc: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}) {
    tt.return
  }
}

// -----

// CHECK-LABEL: device_tensormap_create1d
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @device_tensormap_create1d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: st.shared.b32
    // CHECK: bar.warp.sync
    // CHECK: tensormap.replace.tile.global_address.shared::cta.b1024.b64 [ $0 + 0 ], $1;
    // CHECK: tensormap.replace.tile.rank.shared::cta.b1024.b32 [ $0 + 0 ], 0x0;
    // CHECK: tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [ $0 + 0 ], 0x3;
    // CHECK: tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [ $0 + 0 ], 0x0;
    // CHECK: tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x2;
    // CHECK: tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x1;
    // CHECK: tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [ $0 + 0 ], [ $1 + 0 ], 0x80;
    ttng.tensormap_create %arg1, %arg0, [%c256_i32], [%arg2], [], [%c1_i32] {elem_type = 3 : i32, fill_mode = 1 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32, allocation.offset = 0 : i32} : (!tt.ptr<i8>, !tt.ptr<i16>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// CHECK-LABEL: device_tensormap_create2d
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @device_tensormap_create2d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i64 = arith.constant 1024 : i64
    // CHECK: st.shared.b32
    // CHECK: bar.warp.sync
    // CHECK: tensormap.replace.tile.global_address.shared::cta.b1024.b64 [ $0 + 0 ], $1;
    // CHECK: tensormap.replace.tile.rank.shared::cta.b1024.b32 [ $0 + 0 ], 0x1;
    // CHECK: tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x1, $1;
    // CHECK: tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x1, $1;
    // CHECK: tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x1, $1;
    // CHECK: tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [ $0 + 0 ], 0x3;
    // CHECK: tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [ $0 + 0 ], 0x0;
    // CHECK: tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x2;
    // CHECK: tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x1;
    // CHECK: tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [ $0 + 0 ], [ $1 + 0 ], 0x80;
    ttng.tensormap_create %arg1, %arg0, [%c256_i32, %c256_i32], [%arg2, %arg2], [%c1024_i64], [%c1_i32, %c1_i32] {elem_type = 3 : i32, fill_mode = 1 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32, allocation.offset = 0 : i32} : (!tt.ptr<i8>, !tt.ptr<i16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    tt.return
  }
}

// -----

// CHECK-LABEL: tensormap_fenceproxy_acquire
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensormap_fenceproxy_acquire(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    // CHECK: fence.proxy.tensormap::generic.acquire.gpu [ $0 + 0 ], 0x80;
    // ptxas missing fence workaround:
    // CHECK: cp.async.bulk.commit_group
    // CHECK: cp.async.bulk.wait_group.read 0
    ttng.tensormap_fenceproxy_acquire %arg0 : !tt.ptr<i8>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// CHECK-LABEL: async_copy_mbarrier_arrive
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_copy_mbarrier_arrive(%arg0: !ttg.memdesc<1xi64, #shared, #ttg.shared_memory>)  attributes { noinline = false } {
    // CHECK: nvvm.cp.async.mbarrier.arrive.shared %{{.*}} : !llvm.ptr<3>
    ttng.async_copy_mbarrier_arrive %arg0 : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory>
    // CHECK: nvvm.cp.async.mbarrier.arrive.shared %{{.*}} {noinc = true} : !llvm.ptr<3>
    ttng.async_copy_mbarrier_arrive %arg0 { noIncrement } : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory>
    tt.return
  }
}

// -----

// CHECK-LABEL: map_smem_to_remote
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @map_smem_to_remote(%arg: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    %c1_i32 = arith.constant 1 : i32
    // CHECK: nvvm.mapa %{{.*}} : !llvm.ptr<3> -> !llvm.ptr<7>
    %0 = ttng.map_to_remote_buffer %arg, %c1_i32: !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }
}
