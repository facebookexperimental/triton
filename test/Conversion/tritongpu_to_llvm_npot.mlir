// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions=smem-budget=65536 | FileCheck %s --check-prefix=RLC

#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Register 1 keeps the lane bits and adds 32 for its odd digit; the sum
  // stays below twice the extent, so a compare-and-subtract reduces it. The
  // compare and the subtract are independent, so their order is not checked.
  // CHECK-LABEL: llvm.func @pointwise
  // CHECK: llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[LANE:.*]] = llvm.xor
  // CHECK: llvm.xor %[[LANE]], %{{.*}} : i32
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[BIN:.*]] = llvm.xor %[[LANE]], %[[ZERO]] : i32
  // CHECK-NEXT: %[[ODD:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: %[[SUM:.*]] = llvm.add %[[BIN]], %[[ODD]] : i32
  // CHECK-DAG: %[[WRAP:.*]] = llvm.icmp "uge" %[[SUM]], %[[N:.*]] : i32
  // CHECK-DAG: %[[N]] = llvm.mlir.constant(96 : i32) : i32
  // CHECK-DAG: %[[SUB:.*]] = llvm.sub %[[SUM]], %[[M:.*]] : i32
  // CHECK-DAG: %[[M]] = llvm.mlir.constant(96 : i32) : i32
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
  // each candidate's convert; a Product convert is priced by its plan.
  // RLC-LABEL: tt.func public @product_budget
  // RLC: %[[C:.*]] = ttg.convert_layout %{{.*}} : tensor<96xf32, #linear1> -> tensor<96xf32, #linear>
  // RLC-NEXT: arith.addf %{{.*}}, %[[C]] : tensor<96xf32, #linear>
  // RLC-NOT: ttg.convert_layout
  // RLC: tt.return
  tt.func public @product_budget(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>) {
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
  // CHECK: %[[S0:.*]] = llvm.extractvalue %[[SRC:.*]][0] :
  // CHECK-NEXT: %[[S1:.*]] = llvm.extractvalue %[[SRC]][1] :
  // CHECK-NEXT: %[[S2:.*]] = llvm.extractvalue %[[SRC]][2] :
  // CHECK-NEXT: llvm.extractvalue %[[SRC]][3] :
  // CHECK-NEXT: %[[S4:.*]] = llvm.extractvalue %[[SRC]][4] :
  // CHECK-NOT: {{ptr<3>|nvvm.shfl}}
  // CHECK: llvm.insertvalue %[[S0]], %{{.*}}[0] :
  // CHECK-NEXT: llvm.insertvalue %[[S2]], %{{.*}}[1] :
  // CHECK-NEXT: llvm.insertvalue %[[S4]], %{{.*}}[2] :
  // CHECK-COUNT-3: llvm.insertvalue
  // CHECK-NEXT: llvm.insertvalue %[[S1]], %{{.*}}[6] :
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
  // CHECK: llvm.insertvalue %[[W0]], %{{.*}}[0] :
  // CHECK-NEXT: llvm.insertvalue %[[W2]], %{{.*}}[1] :
  // CHECK-NEXT: llvm.insertvalue %[[W4]], %{{.*}}[2] :
  // CHECK-NEXT: llvm.insertvalue %[[W1]], %{{.*}}[3] :
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
  // Register 1 moves to lane 16, which no factored map expresses. Of each
  // broadcast lane pair only lane bit 4 = 0 stores, and a warp sync suffices.
  // Register 1 stores at odd part 1, so its address adds 1 << 5.
  // CHECK: module attributes {{.*}}ttg.shared = 384 : i32
  // CHECK-LABEL: llvm.func @warp_shared
  // CHECK: "@$2 st.shared.b32 [ $0 + 0 ], $1;"
  // CHECK: %[[Q:.*]] = llvm.udiv %{{.*}}, %{{.*}} : i32
  // CHECK: %[[DIGIT:.*]] = llvm.urem %[[Q]], %{{.*}} : i32
  // CHECK: %[[OWNS:.*]] = llvm.icmp "eq" %[[DIGIT]], %{{.*}} : i32
  // CHECK: %[[PRED:.*]] = llvm.and %{{.*}}, %[[OWNS]] : i1
  // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[ODD:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[OFFSET:.*]] = llvm.shl %[[ODD]], %{{.*}} : i32
  // CHECK-NEXT: llvm.add %{{.*}}, %[[OFFSET]] : i32
  // CHECK: "@$2 st.shared.b32 [ $0 + 0 ], $1;"{{.*}} %[[PRED]] :
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> i32
  // CHECK-NOT: nvvm.barrier
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
  // Element 1 moves to warp 1, so a CTA barrier orders the scratch. Pointers
  // move as i64, and each broadcast destination register pair loads once.
  // CHECK: module attributes {{.*}}ttg.shared = 1536 : i32
  // CHECK-LABEL: llvm.func @block_shared
  // CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr<1> to i64
  // CHECK-COUNT-3: llvm.store %{{.*}} : vector<1xi64>, !llvm.ptr<3>
  // CHECK-NEXT: nvvm.barrier
  // CHECK-COUNT-3: llvm.load %{{.*}} : !llvm.ptr<3> -> i64
  // CHECK-NOT: llvm.load {{.*}}ptr<3>
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
  // Source register 0 broadcasts, so the six unique source values store
  // unpredicated, and the second one seeds its address with coordinate 1.
  // CHECK-LABEL: llvm.func @broadcast_source_register
  // CHECK: llvm.store %{{.*}} : vector<1xi32>, !llvm.ptr<3>
  // CHECK: llvm.mlir.constant(true) : i1
  // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[COORD:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[BIT:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.and %{{.*}}, %[[BIT]] : i32
  // CHECK: llvm.xor %[[COORD]], %{{.*}} : i32
  // CHECK-COUNT-5: llvm.store %{{.*}} : vector<1xi32>, !llvm.ptr<3>
  // CHECK-NEXT: nvvm.barrier
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
  // The odd register digits swap, so the values go through scratch, and each
  // thread reads back only its own stores.
  // CHECK-LABEL: llvm.func @register_shared
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
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
  // CHECK: ttg.warp_id
  // CHECK: %[[W:.*]] = ttg.warp_id
  // CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
  // CHECK-NEXT: %[[BIT:.*]] = llvm.and %[[W]], %[[ONE]]
  // CHECK-NEXT: %[[OWN:.*]] = llvm.icmp "eq" %[[BIT]], %{{.*}}
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
  // A Product value carried through blocks lowers to its register struct.
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
