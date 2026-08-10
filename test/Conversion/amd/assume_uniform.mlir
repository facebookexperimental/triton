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

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // Non-pointer scalars go straight to the type-overloaded intrinsic.
    // CHECK-LABEL: assume_uniform_scalars
    tt.func @assume_uniform_scalars(%i16: i16, %i32: i32, %i64: i64,
                                    %f16: f16, %bf16: bf16, %f32: f32, %f64: f64) {
        // CHECK-NOT: llvm.ptrtoint
        // CHECK: rocdl.readfirstlane %{{.*}} : i16
        // CHECK: rocdl.readfirstlane %{{.*}} : i32
        // CHECK: rocdl.readfirstlane %{{.*}} : i64
        // CHECK: rocdl.readfirstlane %{{.*}} : f16
        // CHECK: rocdl.readfirstlane %{{.*}} : bf16
        // CHECK: rocdl.readfirstlane %{{.*}} : f32
        // CHECK: rocdl.readfirstlane %{{.*}} : f64
        %0 = amdg.assume_uniform %i16 : i16
        %1 = amdg.assume_uniform %i32 : i32
        %2 = amdg.assume_uniform %i64 : i64
        %3 = amdg.assume_uniform %f16 : f16
        %4 = amdg.assume_uniform %bf16 : bf16
        %5 = amdg.assume_uniform %f32 : f32
        %6 = amdg.assume_uniform %f64 : f64
        tt.return
  }
}
