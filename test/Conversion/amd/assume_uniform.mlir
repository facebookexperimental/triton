// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: assume_uniform_base
    tt.func @assume_uniform_base(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}) {
        // The assume makes the base uniform to LLVM, so the buffer op keeps a
        // scalar resource descriptor instead of a per-lane waterfall.
        // CHECK: %[[base:.*]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<1> to i64
        // CHECK: %[[uniform:.*]] = rocdl.readfirstlane %[[base]] : i64
        // CHECK: llvm.inttoptr %[[uniform]] : i64 to !llvm.ptr<1>
        // CHECK: rocdl.raw.ptr.buffer.load
        %ptr = amdg.assume_uniform %arg0 : !tt.ptr<f32>
        %ret = amdg.buffer_load %ptr[%offset] : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // Without the assume, lowering must not introduce a readfirstlane of its
    // own. A base that cannot be proven uniform has to stay on the safe
    // (waterfall) path that LLVM inserts.
    // CHECK-LABEL: no_assume_no_readfirstlane
    // CHECK-NOT: rocdl.readfirstlane
    // CHECK: rocdl.raw.ptr.buffer.load
    tt.func @no_assume_no_readfirstlane(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}) {
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
  }
}
