// RUN: triton-opt -split-input-file %s -tritonamdgpu-pipeline | FileCheck %s

module attributes {
  "triton.skip_generic_pipeline",
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.target" = "hip:gfx942",
  "ttg.threads-per-warp" = 64 : i32
} {
  // CHECK-LABEL: tt.func @skip_generic_pipeline
  tt.func @skip_generic_pipeline(%lb: index, %ub: index, %step: index, %ptr: !tt.ptr<f32>, %init: f32) -> f32 {
    // CHECK-NEXT: %[[RESULT:.*]] = scf.for
    // CHECK-SAME: iter_args
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> f32 {
      // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
      %value = tt.load %ptr {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.ptr<f32>
      // CHECK: %[[SUM:.*]] = arith.addf %[[LOAD]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32}
      %sum = arith.addf %value, %acc {loop.cluster = 0 : i32, loop.stage = 1 : i32} : f32
      // CHECK: scf.yield %[[SUM]] : f32
      scf.yield %sum : f32
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32}
    // CHECK: tt.return %[[RESULT]] : f32
    tt.return %result : f32
  }
}

// -----

// gfx1250 uses the AMD pipeline to lower its explicitly scheduled TDM loops,
// even when TLX asks generic pipelines to leave the module alone.
module attributes {
  "triton.skip_generic_pipeline",
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.target" = "hip:gfx1250",
  "ttg.threads-per-warp" = 32 : i32
} {
  // CHECK-LABEL: tt.func @pipeline_gfx1250_despite_skip
  tt.func @pipeline_gfx1250_despite_skip(%lb: index, %ub: index, %step: index, %ptr: !tt.ptr<f32>, %init: f32) -> f32 {
    // CHECK: tt.load {{.*}} {amd.pipeliner_part = "prologue"}
    // CHECK: scf.for
    // CHECK-NOT: tt.scheduled_max_stage
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> f32 {
      %value = tt.load %ptr {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.ptr<f32>
      %sum = arith.addf %value, %acc {loop.cluster = 0 : i32, loop.stage = 1 : i32} : f32
      scf.yield %sum : f32
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return %result : f32
  }
}
