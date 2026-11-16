// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions=smem-budget=65536 | FileCheck %s --check-prefix=RLC

#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Register 1 keeps the lane bits and adds 32 for its odd digit; the sum
  // stays below twice the extent, so a compare-and-subtract reduces it. The
  // compare and the subtract are independent, so their order is not checked.
  // CHECK-LABEL: llvm.func @pointwise
  // CHECK-DAG: llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[ODD:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG: %[[N:.*]] = llvm.mlir.constant(96 : i32) : i32
  // CHECK: %[[LANE:.*]] = llvm.xor
  // CHECK: llvm.xor %[[LANE]], %{{.*}} : i32
  // CHECK: llvm.xor %{{.*}}, %[[ZERO]] : i32
  // CHECK: %[[BIN:.*]] = llvm.xor %{{.*}}, %[[ZERO]] : i32
  // CHECK: %[[SUM:.*]] = llvm.add %[[BIN]], %[[ODD]] : i32
  // CHECK: %[[WRAP:.*]] = llvm.icmp "uge" %[[SUM]], %[[N]] : i32
  // CHECK: %[[SUB:.*]] = llvm.sub %[[SUM]], %[[N]] : i32
  // CHECK: llvm.select %[[WRAP]], %[[SUB]], %[[SUM]] : i1, i32
  // CHECK-COUNT-3: llvm.fadd
  tt.func public @pointwise(%p: !tt.ptr<f32>) {
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %f = arith.sitofp %r : tensor<96xi32, #p96> to tensor<96xf32, #p96>
    %g = arith.addf %f, %f : tensor<96xf32, #p96>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %a = tt.addptr %s, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    tt.store %a, %g : tensor<96x!tt.ptr<f32>, #p96>
    tt.return
  }
}

// -----

#a = #ttg.linear<{register = [[32], [16]], lane = [[1], [2], [4], [8], [0]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
#b = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The idle buffer lifts the footprint over the budget, so the pass prices
  // each candidate's convert; a mixed-radix convert is priced by its plan.
  // RLC-LABEL: tt.func public @mixed_radix_budget
  // RLC: %[[C:.*]] = ttg.convert_layout %{{.*}} : tensor<96xf32, #linear1> -> tensor<96xf32, #linear>
  // RLC-NEXT: arith.addf %{{.*}}, %[[C]] : tensor<96xf32, #linear>
  // RLC-NOT: ttg.convert_layout
  // RLC: tt.return
  tt.func public @mixed_radix_budget(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256x128xf32, #shared, #smem, mutable>
    %ra = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #a>
    %sa = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #a>
    %pa = tt.addptr %sa, %ra : tensor<96x!tt.ptr<f32>, #a>, tensor<96xi32, #a>
    %x = tt.load %pa : tensor<96x!tt.ptr<f32>, #a>
    %rb = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #b>
    %sb = tt.splat %q : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #b>
    %pb = tt.addptr %sb, %rb : tensor<96x!tt.ptr<f32>, #b>, tensor<96xi32, #b>
    %y = tt.load %pb : tensor<96x!tt.ptr<f32>, #b>
    %yc = ttg.convert_layout %y : tensor<96xf32, #b> -> tensor<96xf32, #a>
    %z = arith.addf %x, %yc : tensor<96xf32, #a>
    tt.store %pa, %z : tensor<96x!tt.ptr<f32>, #a>
    tt.return
  }
}

// -----

#c = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // RLC-LABEL: tt.func public @pow2_budget
  // RLC: %[[C:.*]] = ttg.convert_layout %{{.*}} : tensor<512xf32, #blocked1> -> tensor<512xf32, #blocked>
  // RLC-NEXT: arith.addf %{{.*}}, %[[C]] : tensor<512xf32, #blocked>
  // RLC-NOT: ttg.convert_layout
  // RLC: tt.return
  tt.func public @pow2_budget(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256x128xf32, #shared, #smem, mutable>
    %ra = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #c>
    %sa = tt.splat %p : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #c>
    %pa = tt.addptr %sa, %ra : tensor<512x!tt.ptr<f32>, #c>, tensor<512xi32, #c>
    %x = tt.load %pa : tensor<512x!tt.ptr<f32>, #c>
    %rb = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #d>
    %sb = tt.splat %q : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #d>
    %pb = tt.addptr %sb, %rb : tensor<512x!tt.ptr<f32>, #d>, tensor<512xi32, #d>
    %y = tt.load %pb : tensor<512x!tt.ptr<f32>, #d>
    %yc = ttg.convert_layout %y : tensor<512xf32, #d> -> tensor<512xf32, #c>
    %z = arith.addf %x, %yc : tensor<512xf32, #c>
    tt.store %pa, %z : tensor<512x!tt.ptr<f32>, #c>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[1], [2], [128]], lane = [[4], [8], [16], [32], [64]], warp = [], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [384]}>
#dst = #ttg.linear<{register = [[2], [128], [1]], lane = [[4], [8], [16], [32], [64]], warp = [], block = [], inputRadices = {register = [2, 3, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [384]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Destination register j holds source register 2j mod 11 (11 for j = 11).
  // CHECK: module attributes {{.*}}ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @registers
  // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[V0:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V1:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V2:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V3:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V4:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V5:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V6:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V7:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V8:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V9:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V10:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK: %[[V11:.*]] = llvm.add %{{.+}}, %[[ZERO]] : i32
  // CHECK-NOT: {{ptr<3>|nvvm.shfl}}
  // CHECK: llvm.insertelement %[[V0]],
  // CHECK: llvm.insertelement %[[V2]],
  // CHECK: llvm.insertelement %[[V4]],
  // CHECK: llvm.insertelement %[[V6]],
  // CHECK: llvm.insertelement %[[V8]],
  // CHECK: llvm.insertelement %[[V10]],
  // CHECK: llvm.insertelement %[[V1]],
  // CHECK: llvm.insertelement %[[V3]],
  // CHECK: llvm.insertelement %[[V5]],
  // CHECK: llvm.insertelement %[[V7]],
  // CHECK: llvm.insertelement %[[V9]],
  // CHECK: llvm.insertelement %[[V11]],
  tt.func public @registers(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32, #src>
    %c = ttg.convert_layout %r : tensor<384xi32, #src> -> tensor<384xi32, #dst>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<384x!tt.ptr<i32>, #dst>
    tt.store %s, %c : tensor<384x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[1], [64]], lane = [[2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[64], [1]], lane = [[4], [2], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Lane bits 0 and 1 swap, and the register digits reverse after the shuffles.
  // CHECK: module attributes {{.*}}ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @warp_reorder
  // CHECK: %[[W0:.*]] = nvvm.shfl.sync idx
  // CHECK: %[[W1:.*]] = nvvm.shfl.sync idx
  // CHECK: %[[W2:.*]] = nvvm.shfl.sync idx
  // CHECK: nvvm.shfl.sync idx
  // CHECK: %[[W4:.*]] = nvvm.shfl.sync idx
  // CHECK-NOT: ptr<3>
  // CHECK: llvm.insertelement %[[W0]],
  // CHECK: llvm.insertelement %[[W2]],
  // CHECK: llvm.insertelement %[[W4]],
  // CHECK: llvm.insertelement %[[W1]],
  tt.func public @warp_reorder(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #src>
    %c = ttg.convert_layout %r {force_warp_shuffle} : tensor<192xi32, #src> -> tensor<192xi32, #dst>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<192x!tt.ptr<i32>, #dst>
    tt.store %s, %c : tensor<192x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[32], [16]], lane = [[1], [2], [4], [8], [0]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
#dst = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Register 1 moves to lane 16, which no factored map expresses. Each odd
  // tile is a 32-element pow2 transfer at its own offset, and a warp sync
  // suffices.
  // CHECK: module attributes {{.*}}ttg.shared = 384 : i32
  // CHECK-LABEL: llvm.func @warp_shared
  // CHECK-DAG: llvm.mlir.constant(-1 : i32) : i32
  // CHECK: nvvm.stmatrix
  // CHECK: llvm.getelementptr %{{.*}}[32] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i32
  // CHECK: nvvm.stmatrix
  // CHECK: llvm.getelementptr %{{.*}}[64] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i32
  // CHECK: nvvm.stmatrix
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK-COUNT-3: nvvm.ldmatrix {{.*}}, num = 1,
  // CHECK-NOT: {{nvvm.barrier|matrix}}
  // CHECK: llvm.return
  tt.func public @warp_shared(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #src>
    %c = ttg.convert_layout %r : tensor<96xi32, #src> -> tensor<96xi32, #dst>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>, #dst>
    tt.store %s, %c : tensor<96x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16]], warp = [[32]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[64], [0]], lane = [[2], [4], [8], [16], [32]], warp = [[1]], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Element 1 moves to warp 1, so CTA barriers order the scratch. Pointers
  // move as i64, and each broadcast destination register pair loads once.
  // Two odd tiles fit in the pow2 scratch, so the third takes a second round.
  // CHECK: module attributes {{.*}}ttg.shared = 1024 : i32
  // CHECK-LABEL: llvm.func @block_shared
  // CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr<1> to i64
  // CHECK-COUNT-2: llvm.store %{{.*}} : vector<1xi64>, !llvm.ptr<3>
  // CHECK-NEXT: nvvm.barrier
  // CHECK-COUNT-2: llvm.load %{{.*}} : !llvm.ptr<3> -> i64
  // CHECK: nvvm.barrier
  // CHECK: llvm.store %{{.*}} : vector<1xi64>, !llvm.ptr<3>
  // CHECK-NEXT: nvvm.barrier
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i64
  // CHECK-NOT: {{nvvm.barrier|ptr<3>}}
  // CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
  tt.func public @block_shared(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #src>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<192x!tt.ptr<i32>, #src>
    %a = tt.addptr %s, %r : tensor<192x!tt.ptr<i32>, #src>, tensor<192xi32, #src>
    %c = ttg.convert_layout %a : tensor<192x!tt.ptr<i32>, #src> -> tensor<192x!tt.ptr<i32>, #dst>
    %r2 = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #dst>
    tt.store %c, %r2 : tensor<192x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

#bl4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#bl1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A power-of-two conversion keeps the vectorized swizzled lowering.
  // CHECK-LABEL: llvm.func @pow2_control
  // CHECK: llvm.store {{.*}} : vector<4xi32>, !llvm.ptr<3>
  // CHECK: nvvm.ldmatrix
  tt.func public @pow2_control(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #bl4>
    %c = ttg.convert_layout %r : tensor<256xi32, #bl4> -> tensor<256xi32, #bl1>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #bl1>
    tt.store %s, %c : tensor<256x!tt.ptr<i32>, #bl1>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[0], [1], [128]], lane = [[2], [4], [8], [16], [32]], warp = [[64]], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [384]}>
#dst = #ttg.linear<{register = [[1], [2], [128]], lane = [[8], [16], [32], [64], [0]], warp = [[4]], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [384]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Source register 0 broadcasts, so each odd tile's two unique source values
  // store unpredicated as one 2-vector.
  // CHECK: module attributes {{.*}}ttg.shared = 1024 : i32
  // CHECK-LABEL: llvm.func @broadcast_source_register
  // CHECK-COUNT-2: llvm.store %{{.*}} : vector<2xi32>, !llvm.ptr<3>
  // CHECK-NEXT: nvvm.barrier
  // CHECK-COUNT-4: llvm.load {{.*}} : !llvm.ptr<3> -> vector<2xi32>
  // CHECK: nvvm.barrier
  // CHECK: llvm.store %{{.*}} : vector<2xi32>, !llvm.ptr<3>
  // CHECK-NEXT: nvvm.barrier
  // CHECK-COUNT-2: llvm.load {{.*}} : !llvm.ptr<3> -> vector<2xi32>
  // CHECK-NOT: {{nvvm.barrier|ptr<3> ->}}
  tt.func public @broadcast_source_register(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32, #src>
    %c = ttg.convert_layout %r : tensor<384xi32, #src> -> tensor<384xi32, #dst>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<384x!tt.ptr<i32>, #dst>
    tt.store %s, %c : tensor<384x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[8, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [0, 1], [0, 2]], warp = [], block = [], inputRadices = {register = [3, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [24, 12]}>
#dst = #ttg.linear<{register = [[0, 4], [8, 0]], lane = [[1, 0], [2, 0], [4, 0], [0, 1], [0, 2]], warp = [], block = [], inputRadices = {register = [3, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [24, 12]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The odd register digits swap, so the values go through scratch as 3 x 3
  // tiles, four per round, and a warp sync suffices.
  // CHECK: module attributes {{.*}}ttg.shared = 512 : i32
  // CHECK-LABEL: llvm.func @register_shared
  // CHECK-DAG: llvm.mlir.constant(-1 : i32) : i32
  // CHECK-COUNT-4: nvvm.stmatrix
  // CHECK-NEXT: nvvm.bar.warp.sync
  // CHECK-COUNT-4: nvvm.ldmatrix
  // CHECK-COUNT-4: nvvm.bar.warp.sync
  // CHECK: nvvm.ldmatrix
  // CHECK-NOT: {{nvvm.barrier|matrix}}
  // CHECK: llvm.return
  tt.func public @register_shared(%p: !tt.ptr<f32>, %f: f32) {
    %v = tt.splat %f : f32 -> tensor<24x12xf32, #src>
    %c = ttg.convert_layout %v : tensor<24x12xf32, #src> -> tensor<24x12xf32, #dst>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<24x12x!tt.ptr<f32>, #dst>
    tt.store %s, %c : tensor<24x12x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// -----

#p = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16]], warp = [[0], [32]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Warp bit 0 broadcasts, so only even warps store.
  // CHECK-LABEL: llvm.func @warp_broadcast_store
  // CHECK-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
  // CHECK: ttg.warp_id
  // CHECK: %[[W:.*]] = ttg.warp_id
  // CHECK: %[[BIT:.*]] = llvm.and %[[W]], %[[ONE]]
  // CHECK: %[[OWN:.*]] = llvm.icmp "eq" %[[BIT]], %{{.*}}
  // CHECK: st.global{{.*}}%[[OWN]]
  tt.func public @warp_broadcast_store(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #p>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<192x!tt.ptr<i32>, #p>
    %a = tt.addptr %s, %r : tensor<192x!tt.ptr<i32>, #p>, tensor<192xi32, #p>
    tt.store %a, %r : tensor<192x!tt.ptr<i32>, #p>
    tt.return
  }
}

// -----

#p = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A mixed-radix value carried through blocks lowers to its register struct.
  // CHECK-LABEL: llvm.func @carried
  // CHECK: llvm.br ^bb1({{.*}} : i32, !llvm.struct<(i32, i32, i32)>)
  // CHECK: ^bb1({{.*}}: i32, {{.*}}: !llvm.struct<(i32, i32, i32)>)
  tt.func public @carried(%x: !tt.ptr<i32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %z = arith.constant dense<0> : tensor<96xi32, #p>
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p>
    %s = tt.splat %x : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>, #p>
    %p = tt.addptr %s, %r : tensor<96x!tt.ptr<i32>, #p>, tensor<96xi32, #p>
    cf.br ^bb1(%c0, %z : i32, tensor<96xi32, #p>)
  ^bb1(%i: i32, %a: tensor<96xi32, #p>):
    %lt = arith.cmpi slt, %i, %n : i32
    cf.cond_br %lt, ^bb2, ^bb3
  ^bb2:
    %v = tt.load %p : tensor<96x!tt.ptr<i32>, #p>
    %b = arith.addi %a, %v : tensor<96xi32, #p>
    %j = arith.addi %i, %c1 : i32
    cf.br ^bb1(%j, %b : i32, tensor<96xi32, #p>)
  ^bb3:
    tt.store %p, %a : tensor<96x!tt.ptr<i32>, #p>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#dst = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Both sides move 8 f16 per access, so each 128x32 odd tile stores through
  // stmatrix; dim 0 bits 32 and 64 are registers on both sides, so 4 reps, and
  // one tile fits in the pow2 scratch, so 12 rounds.
  // CHECK: module attributes {{.*}}ttg.shared = 2048 : i32
  // CHECK-LABEL: llvm.func @block_shared_vectorized
  // CHECK: nvvm.stmatrix {{.*}} : !llvm.ptr<3>, i32, i32, i32, i32
  // CHECK-NEXT: nvvm.barrier
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xi32>
  // CHECK-COUNT-22: nvvm.barrier
  // CHECK-NOT: nvvm.barrier
  tt.func public @block_shared_vectorized(%p: !tt.ptr<f16>) {
    %s = tt.splat %p : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>, #src>
    %v = tt.load %s : tensor<128x96x!tt.ptr<f16>, #src>
    %c = ttg.convert_layout %v : tensor<128x96xf16, #src> -> tensor<128x96xf16, #dst>
    %d = tt.splat %p : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>, #dst>
    tt.store %d, %c : tensor<128x96x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[0, 1], [8, 0], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 8], [0, 16]], block = [], inputRadices = {register = [2, 2, 2, 3, 2, 2, 2], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 192]}>
#dst = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0], [64, 0], [0, 64]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 192]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // An mma.sync accumulator to a store layout: dim 0 bits 16-64 are registers
  // on both sides, so 8 reps of 16x64, and one tile fits in the pow2 scratch,
  // so 24 rounds.
  // CHECK: module attributes {{.*}}ttg.shared = 2048 : i32
  // CHECK-LABEL: llvm.func @accumulator_epilogue_tiles
  // CHECK: nvvm.stmatrix {{.*}} : !llvm.ptr<3>, i32, i32, i32, i32
  // CHECK-NEXT: nvvm.barrier
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xi32>
  // CHECK-COUNT-46: nvvm.barrier
  // CHECK-NOT: nvvm.barrier
  tt.func public @accumulator_epilogue_tiles(%p: !tt.ptr<f16>, %f: f16) {
    %v = tt.splat %f : f16 -> tensor<128x192xf16, #src>
    %c = ttg.convert_layout %v : tensor<128x192xf16, #src> -> tensor<128x192xf16, #dst>
    %d = tt.splat %p : !tt.ptr<f16> -> tensor<128x192x!tt.ptr<f16>, #dst>
    tt.store %d, %c : tensor<128x192x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32], [0, 64], [64, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[16, 0], [32, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [192, 128]}>
#dst = #ttg.linear<{register = [[0, 1], [0, 16], [0, 32], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 64], [1, 0]], warp = [[2, 0], [4, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [192, 128]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A wgmma accumulator to a layout with dim 1 bits 16 and 32 in registers:
  // the tiles move 4-vectors, dim 0 bit 8 makes 2 reps, and the scratch holds
  // only what the pow2 alloc shape needs, one tile, so 6 rounds.
  // CHECK: module attributes {{.*}}ttg.shared = 8192 : i32
  // CHECK-LABEL: llvm.func @accumulator_epilogue_middle_bit
  // CHECK-COUNT-4: llvm.store {{.*}} : vector<4xi32>, !llvm.ptr<3>
  // CHECK-NEXT: nvvm.barrier
  // CHECK-COUNT-4: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xi32>
  // CHECK-COUNT-10: nvvm.barrier
  // CHECK-NOT: nvvm.barrier
  tt.func public @accumulator_epilogue_middle_bit(%p: !tt.ptr<f16>, %f: f16) {
    %v = tt.splat %f : f16 -> tensor<192x128xf16, #src>
    %c = ttg.convert_layout %v : tensor<192x128xf16, #src> -> tensor<192x128xf16, #dst>
    %d = tt.splat %p : !tt.ptr<f16> -> tensor<192x128x!tt.ptr<f16>, #dst>
    tt.store %d, %c : tensor<192x128x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32], [0, 64], [64, 0], [128, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[16, 0], [32, 0]], block = []}>
#dst = #ttg.linear<{register = [[0, 1], [0, 16], [0, 32], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 64], [1, 0]], warp = [[2, 0], [4, 0]], block = []}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The pow2 alloc shape of the case above takes the same scratch.
  // CHECK: module attributes {{.*}}ttg.shared = 8192 : i32
  // CHECK-LABEL: llvm.func @accumulator_epilogue_middle_bit_pow2
  tt.func public @accumulator_epilogue_middle_bit_pow2(%p: !tt.ptr<f16>, %f: f16) {
    %v = tt.splat %f : f16 -> tensor<256x128xf16, #src>
    %c = ttg.convert_layout %v : tensor<256x128xf16, #src> -> tensor<256x128xf16, #dst>
    %d = tt.splat %p : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #dst>
    tt.store %d, %c : tensor<256x128x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 0]], warp = [], block = [], inputRadices = {register = [2, 2, 3, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [12, 32]}>
#dst = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [12, 32]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Peeling the odd register digit leaves 4x32 tiles, each one stmatrix
  // and one ldmatrix, two per round.
  // CHECK: module attributes {{.*}}ttg.shared = 512 : i32
  // CHECK-LABEL: llvm.func @run_ends_at_odd_digit
  // CHECK-COUNT-2: nvvm.stmatrix {{.*}} : !llvm.ptr<3>, i32, i32, i32, i32
  // CHECK: nvvm.bar.warp.sync
  // CHECK-COUNT-2: nvvm.ldmatrix {{.*}}, num = 2,
  // CHECK: nvvm.bar.warp.sync
  // CHECK: nvvm.stmatrix {{.*}} : !llvm.ptr<3>, i32, i32, i32, i32
  // CHECK: nvvm.bar.warp.sync
  // CHECK: nvvm.ldmatrix {{.*}}, num = 2,
  // CHECK-NOT: {{st.shared|ptr<3> ->}}
  tt.func public @run_ends_at_odd_digit(%p: !tt.ptr<f16>) {
    %s = tt.splat %p : !tt.ptr<f16> -> tensor<12x32x!tt.ptr<f16>, #src>
    %v = tt.load %s : tensor<12x32x!tt.ptr<f16>, #src>
    %c = ttg.convert_layout %v : tensor<12x32xf16, #src> -> tensor<12x32xf16, #dst>
    %d = tt.splat %p : !tt.ptr<f16> -> tensor<12x32x!tt.ptr<f16>, #dst>
    tt.store %d, %c : tensor<12x32x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[1], [4], [64]], lane = [[2], [8], [16], [32], [0]], warp = [], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[1], [64]], lane = [[2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The second register digit steps by 4, not 2, so each tile stores
  // 2-vectors, two tiles per round.
  // CHECK: module attributes {{.*}}ttg.shared = 512 : i32
  // CHECK-LABEL: llvm.func @run_ends_at_gap
  // CHECK-COUNT-4: llvm.store {{.*}} : vector<2xi32>, !llvm.ptr<3>
  // CHECK: nvvm.bar.warp.sync
  // CHECK-COUNT-2: llvm.load {{.*}} : !llvm.ptr<3> -> vector<2xi32>
  // CHECK: nvvm.bar.warp.sync
  // CHECK-COUNT-2: llvm.store {{.*}} : vector<2xi32>, !llvm.ptr<3>
  // CHECK: nvvm.bar.warp.sync
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<2xi32>
  // CHECK-NOT: {{nvvm.barrier|ptr<3> ->}}
  tt.func public @run_ends_at_gap(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #src>
    %v = tt.load %s : tensor<192x!tt.ptr<f32>, #src>
    %c = ttg.convert_layout %v : tensor<192xf32, #src> -> tensor<192xf32, #dst>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #dst>
    tt.store %d, %c : tensor<192x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[0, 16]], lane = [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 8]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [4, 48]}>
#dst = #ttg.linear<{register = [[0, 16]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0]], warp = [[2, 0]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [4, 48]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Both dims take one access per side, so each tile moves scalars.
  // CHECK: module attributes {{.*}}ttg.shared = 768 : i32
  // CHECK-LABEL: llvm.func @tie_keeps_dim0
  // CHECK-COUNT-3: llvm.store {{.*}} : vector<1xi32>, !llvm.ptr<3>
  // CHECK-NEXT: nvvm.barrier
  // CHECK-COUNT-3: llvm.load {{.*}} : !llvm.ptr<3> -> i32
  // CHECK-NOT: {{nvvm.barrier|ptr<3> ->}}
  tt.func public @tie_keeps_dim0(%p: !tt.ptr<i32>) {
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<4x48x!tt.ptr<i32>, #src>
    %v = tt.load %s : tensor<4x48x!tt.ptr<i32>, #src>
    %c = ttg.convert_layout %v : tensor<4x48xi32, #src> -> tensor<4x48xi32, #dst>
    %d = tt.splat %p : !tt.ptr<i32> -> tensor<4x48x!tt.ptr<i32>, #dst>
    tt.store %d, %c : tensor<4x48x!tt.ptr<i32>, #dst>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 96, 16]}>
#dst = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0], [32, 0], [64, 0], [0, 32]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A split-K wgmma epilogue: the pow2 alloc shape's (128x128) scratch holds
  // one 128x32 tile, so the three tiles move in three rounds.
  // CHECK: module attributes {{.*}}ttg.shared = 16384 : i32
  // CHECK-LABEL: llvm.func @odd_tiles_three_rounds
  // CHECK-COUNT-8: llvm.store {{.*}} : vector<4xi32>, !llvm.ptr<3>
  // CHECK-NEXT: nvvm.barrier
  // CHECK-COUNT-8: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xi32>
  // CHECK-COUNT-4: nvvm.barrier
  // CHECK-COUNT-8: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xi32>
  // CHECK-NOT: {{nvvm.barrier|ptr<3>}}
  // CHECK: llvm.return
  tt.func public @odd_tiles_three_rounds(%p: !tt.ptr<f32>, %f: f32) {
    %v = tt.splat %f : f32 -> tensor<128x96xf32, #mma>
    %c = ttg.convert_layout %v : tensor<128x96xf32, #mma> -> tensor<128x96xf32, #dst>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<128x96x!tt.ptr<f32>, #dst>
    tt.store %d, %c : tensor<128x96x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// -----

#a = #ttg.linear<{register = [[0, 32], [1, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [3, 64]}>
#b = #ttg.linear<{register = [[32, 0], [0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [64, 3]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A mixed-radix transpose keeps every register in place.
  // CHECK: module attributes {{.*}}ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @mixed_radix_transpose
  // CHECK-NOT: nvvm.shfl
  // CHECK-NOT: nvvm.barrier
  // CHECK: llvm.return
  tt.func public @mixed_radix_transpose(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<3x64x!tt.ptr<f32>, #a>
    %v = tt.load %s : tensor<3x64x!tt.ptr<f32>, #a>
    %t = tt.trans %v {order = array<i32: 1, 0>} : tensor<3x64xf32, #a> -> tensor<64x3xf32, #b>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<64x3x!tt.ptr<f32>, #b>
    tt.store %d, %t : tensor<64x3x!tt.ptr<f32>, #b>
    tt.return
  }
}

// -----

#a = #ttg.linear<{register = [[0, 32], [1, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [3, 64]}>
#flat = #ttg.linear<{register = [[32], [64]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A mixed-radix reshape to the regrouped source keeps every register in
  // place.
  // CHECK: module attributes {{.*}}ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @mixed_radix_reshape
  // CHECK-NOT: nvvm.shfl
  // CHECK-NOT: nvvm.barrier
  // CHECK: llvm.return
  tt.func public @mixed_radix_reshape(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<3x64x!tt.ptr<f32>, #a>
    %v = tt.load %s : tensor<3x64x!tt.ptr<f32>, #a>
    %r = tt.reshape %v : tensor<3x64xf32, #a> -> tensor<192xf32, #flat>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #flat>
    tt.store %d, %r : tensor<192x!tt.ptr<f32>, #flat>
    tt.return
  }
}

// -----

#b = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 0]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 3]}>
#flat = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [0]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // No 192 layout keeps these registers, so warps with digit 1 = 0 store each
  // element at row * 3 + column and every thread reloads after a barrier.
  // CHECK: module attributes {{.*}}ttg.shared = 768 : i32
  // CHECK-LABEL: llvm.func @moving_reshape
  // CHECK-DAG: llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG: %[[THREE:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %[[Q:.*]] = llvm.udiv %{{.*}}, %{{.*}} : i32
  // CHECK: %[[D:.*]] = llvm.urem %[[Q]], %{{.*}} : i32
  // CHECK: %[[OWN:.*]] = llvm.icmp "eq" %[[D]], %{{.*}} : i32
  // CHECK-NEXT: %[[PRED:.*]] = llvm.and %{{.*}}, %[[OWN]] : i1
  // CHECK: llvm.mul %{{.*}}, %[[THREE]] : i32
  // CHECK: "@$2 st.shared.b32 [ $0 + 0 ], $1;"{{.*}} %[[PRED]] :
  // CHECK-COUNT-2: "@$2 st.shared.b32 [ $0 + 0 ], $1;"
  // CHECK-NEXT: nvvm.barrier
  // CHECK-COUNT-3: llvm.load {{.*}} : !llvm.ptr<3> -> f32
  // CHECK-NOT: ptr<3>
  // CHECK: llvm.return
  tt.func public @moving_reshape(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<64x3x!tt.ptr<f32>, #b>
    %v = tt.load %s : tensor<64x3x!tt.ptr<f32>, #b>
    %r = tt.reshape %v : tensor<64x3xf32, #b> -> tensor<192xf32, #flat>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #flat>
    tt.store %d, %r : tensor<192x!tt.ptr<f32>, #flat>
    tt.return
  }
}

// -----

#b = #ttg.linear<{register = [[32, 0], [0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [64, 3]}>
#flat = #ttg.linear<{register = [[32], [64]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Shared memory holds i1 as i8 and pointers as i64.
  // CHECK: module attributes {{.*}}ttg.shared = 1536 : i32
  // CHECK-LABEL: llvm.func @moving_reshape_i1_ptr
  // CHECK: llvm.zext %{{.*}} : i1 to i8
  // CHECK: llvm.store {{.*}} : vector<1xi8>, !llvm.ptr<3>
  // CHECK: nvvm.barrier
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> i8
  // CHECK: llvm.trunc %{{.*}} : i8 to i1
  // CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr<1> to i64
  // CHECK: llvm.store {{.*}} : vector<1xi64>, !llvm.ptr<3>
  // CHECK: nvvm.barrier
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> i64
  // CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
  tt.func public @moving_reshape_i1_ptr(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<64x3x!tt.ptr<f32>, #b>
    %v = tt.load %s : tensor<64x3x!tt.ptr<f32>, #b>
    %m = arith.cmpf olt, %v, %v : tensor<64x3xf32, #b>
    %rm = tt.reshape %m : tensor<64x3xi1, #b> -> tensor<192xi1, #flat>
    %rp = tt.reshape %s : tensor<64x3x!tt.ptr<f32>, #b> -> tensor<192x!tt.ptr<f32>, #flat>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #flat>
    %x = tt.load %d : tensor<192x!tt.ptr<f32>, #flat>
    tt.store %rp, %x, %rm : tensor<192x!tt.ptr<f32>, #flat>
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A pow2 reorder reshape to a layout other than the regroup stays in registers.
  // CHECK: module attributes {{.*}}ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @pow2_reorder_reshape
  // CHECK-NOT: nvvm.barrier
  // CHECK: llvm.return
  tt.func public @pow2_reorder_reshape(%p: !tt.ptr<f32>) {
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #src>
    %v = tt.load %s : tensor<256x!tt.ptr<f32>, #src>
    %r = tt.reshape %v allow_reorder : tensor<256xf32, #src> -> tensor<128x2xf32, #dst>
    %d = tt.splat %p : !tt.ptr<f32> -> tensor<128x2x!tt.ptr<f32>, #dst>
    tt.store %d, %r : tensor<128x2x!tt.ptr<f32>, #dst>
    tt.return
  }
}
