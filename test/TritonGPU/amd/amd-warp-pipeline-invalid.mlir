// RUN: triton-opt %s -split-input-file -tritonamdgpu-normalize-warp-pipeline -tritonamdgpu-warp-pipeline -verify-diagnostics

// Loops are not allowed inside a warp_pipeline_stage region; see isLoopOp
// in WarpPipeliner.cpp for the rationale (no scheduling benefit, opaque to
// MemoryEffectOpInterface, also covers the "no nested warp pipelines"
// rule).  Both the loop-form (createPipeline) and flat-form
// (createFlatPipeline) must reject loops between borders.

// ---- Loop-form: scf.for inside a stage ----

tt.func @loop_form_for_in_cluster(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "stage"}
    %a = arith.addi %i, %c1 : index
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}

    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "stage"}
    // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
    scf.for %j = %c0 to %n step %c1 {
      scf.yield
    }

    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "last_cluster"}
    %b = arith.addi %a, %i : index

    rocdl.sched.barrier none {triton.warp_pipeline.border = "last_cluster"}
    scf.yield
  }

  tt.return
}

// -----

// ---- Loop-form: scf.while inside a stage ----

tt.func @loop_form_while_in_cluster(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "stage"}
    %a = arith.addi %i, %c1 : index
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}

    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "stage"}
    // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
    scf.while (%w = %c0) : (index) -> index {
      %cond = arith.cmpi slt, %w, %n : index
      scf.condition(%cond) %w : index
    } do {
    ^bb0(%w: index):
      %wn = arith.addi %w, %c1 : index
      scf.yield %wn : index
    }

    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "last_cluster"}
    %b = arith.addi %a, %i : index

    rocdl.sched.barrier none {triton.warp_pipeline.border = "last_cluster"}
    scf.yield
  }

  tt.return
}

// -----

// ---- Loop-form: nested warp-pipelined scf.for is still a loop ----
//
// Even an already-pipelined inner loop is rejected: nesting warp pipelines
// is a hard constraint, and the loop-op check enforces it for free.

tt.func @loop_form_nested_pipelined_for(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "stage"}
    %a = arith.addi %i, %c1 : index
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}

    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "stage"}
    // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
    scf.for %j = %c0 to %n step %c1 {
      scf.execute_region {
        scf.yield
      } {triton.warp_pipeline.stage = "inner"}
      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "last_cluster"}
    %b = arith.addi %a, %i : index

    rocdl.sched.barrier none {triton.warp_pipeline.border = "last_cluster"}
    scf.yield
  }

  tt.return
}

// -----

// ---- Flat-form: scf.for between flat borders ----

tt.func @flat_form_for_in_cluster(%n: index, %ptr: !tt.ptr<f32>) {
  rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "stage0"}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v0 = arith.constant 0.0 : f32

  tt.store %ptr, %v0 : !tt.ptr<f32>
  rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}

  rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "stage1"}
  // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
  scf.for %j = %c0 to %n step %c1 {
    scf.yield
  }

  rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}
  tt.store %ptr, %v0 : !tt.ptr<f32>

  tt.return
}

// -----

// An async token producer cannot move past the wait that consumes it.
// Normalization must reject the wait inside the explicit stage.
tt.func @wait_token_producer_inside_stage(%n: index, %token: !ttg.async.token, %ptr: !tt.ptr<i32>, %value: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "first"}
    %next = arith.addi %value, %value : i32
    %group = ttg.async_commit_group tokens %token
    // expected-error @+1 {{barrier or wait op cannot appear inside a warp_pipeline_stage region}}
    ttg.async_wait %group {num = 0 : i32}
    tt.store %ptr, %next : !tt.ptr<i32>
    rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "second"}
    tt.store %ptr, %value : !tt.ptr<i32>
    rocdl.sched.barrier none {triton.warp_pipeline.border = "second"}
  }
  tt.return
}

// -----

// The final yield does not permit moving an input into a previous stage.
tt.func @input_after_last_stage(%ptr: !tt.ptr<i32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%carry = %c0) -> i32 : i32 {
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "first"}
    tt.store %ptr, %carry : !tt.ptr<i32>
    rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
    rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "last"}
    tt.store %ptr, %i : !tt.ptr<i32>
    rocdl.sched.barrier none {triton.warp_pipeline.border = "last"}
    // expected-error @+1 {{no legal warp-pipeline stage for input}}
    %next = arith.addi %carry, %i : i32
    // expected-note @+1 {{this use requires a stage before the input's original position}}
    scf.yield %next : i32
  }
  tt.return
}

// -----

// This conversion exchanges values between waves and requires shared memory.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#other = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @cross_wave_input(%input: tensor<256xf32, #blocked>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      // expected-error @+1 {{expected pure scalar computation or a wave-local layout conversion}}
      %converted = ttg.convert_layout %input : tensor<256xf32, #blocked> -> tensor<256xf32, #other>
      ttg.async_wait {num = 0 : i32}
      rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "first"}
      %sum = arith.addf %converted, %converted : tensor<256xf32, #other>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "first"}
      rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "last"}
      %next = arith.addf %sum, %sum : tensor<256xf32, #other>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "last"}
    }
    tt.return
  }
}

// -----

tt.func @unmatched_stage_begin() {
  // expected-error @+1 {{unmatched warp_pipeline_stage begin marker}}
  rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "first"}
  tt.return
}

// -----

tt.func @mismatched_stage_end() {
  rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "first"}
  // expected-error @+1 {{unmatched warp_pipeline_stage end marker}}
  rocdl.sched.barrier none {triton.warp_pipeline.border = "other"}
  tt.return
}

// -----

tt.func @nested_stages() {
  rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "outer"}
  // expected-error @+1 {{nested warp_pipeline_stage regions are not supported}}
  rocdl.sched.barrier none {triton.warp_pipeline.stage_begin = "inner"}
  rocdl.sched.barrier none {triton.warp_pipeline.border = "inner"}
  rocdl.sched.barrier none {triton.warp_pipeline.border = "outer"}
  tt.return
}
