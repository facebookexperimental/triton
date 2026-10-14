// RUN: triton-opt %s -split-input-file -tritonamdgpu-normalize-warp-pipeline | FileCheck %s --check-prefix=NORM
// RUN: triton-opt %s -split-input-file -tritonamdgpu-preserve-warp-pipeline-owners | FileCheck %s --check-prefix=PREP
// RUN: triton-opt %s -split-input-file -tritonamdgpu-preserve-warp-pipeline-owners -canonicalize -tritonamdgpu-normalize-warp-pipeline -tritonamdgpu-warp-pipeline | FileCheck %s --check-prefix=PIPE

module {
  tt.func @single_trip_owner_is_nop(%ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v0 = arith.constant 0 : i32
    %v1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c1 step %c1 {
      tt.store %ptr, %v0 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}
      tt.store %ptr, %v1 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}
    }
    tt.return
  }
}

// PREP-LABEL: tt.func @single_trip_owner_is_nop
// PREP-NOT: triton.warp_pipeline
// PREP: tt.return

// PIPE-LABEL: tt.func @single_trip_owner_is_nop
// PIPE-NOT: triton.warp_pipeline
// PIPE-NOT: scf.execute_region
// PIPE: tt.return

// -----

module {
  tt.func @multi_trip_owner_is_preserved(%ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %v0 = arith.constant 0 : i32
    %v1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c4 step %c1 {
      tt.store %ptr, %v0 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}
      tt.store %ptr, %v1 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}
    }
    tt.return
  }
}

// PREP-LABEL: tt.func @multi_trip_owner_is_preserved
// PREP: rocdl.sched.barrier none
// PREP-SAME: triton.warp_pipeline.owner_begin = 0 : i64
// PREP-SAME: triton.warp_pipeline.owner_stage_count = 2 : i64
// PREP-NEXT: scf.for
// PREP: triton.warp_pipeline.border = "stage0"
// PREP: triton.warp_pipeline.border = "stage1"
// PREP: rocdl.sched.barrier none {triton.warp_pipeline.owner_end = 0 : i64}

// PIPE-LABEL: tt.func @multi_trip_owner_is_preserved
// PIPE-NOT: owner_begin
// PIPE: scf.for
// PIPE: scf.execute_region
// PIPE: scf.execute_region
// PIPE: triton.warp_pipeline.pipelined_for
// PIPE-NOT: owner_end
// PIPE: tt.return

// -----

// This is the shape left behind when an owner loop is fully unrolled inside a
// persistent outer loop. The prologue/epilogue waits are outside the preserved
// owner scope and must not make the outer loop a warp-pipeline owner.
module {
  tt.func @flattened_owner_inside_persistent_loop(%n: index,
                                                   %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v0 = arith.constant 0 : i32
    %v1 = arith.constant 1 : i32
    scf.for %tile = %c0 to %n step %c1 {
      ttg.async_wait {num = 1 : i32}
      rocdl.sched.barrier none {triton.warp_pipeline.owner_begin = 7 : i64, triton.warp_pipeline.owner_stage_count = 2 : i64}
      tt.store %ptr, %v0 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}
      tt.store %ptr, %v1 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}
      tt.store %ptr, %v0 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}
      tt.store %ptr, %v1 : !tt.ptr<i32>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}
      rocdl.sched.barrier none {triton.warp_pipeline.owner_end = 7 : i64}
      ttg.async_wait {num = 0 : i32}
    }
    tt.return
  }
}

// PREP-LABEL: tt.func @flattened_owner_inside_persistent_loop

// PIPE-LABEL: tt.func @flattened_owner_inside_persistent_loop
// PIPE: scf.for
// PIPE-NOT: triton.warp_pipeline.pipelined_for
// PIPE: ttg.async_wait
// PIPE-NEXT: scf.execute_region
// PIPE: } {triton.warp_pipeline.flat
// PIPE: scf.execute_region
// PIPE: } {triton.warp_pipeline.flat
// PIPE: scf.execute_region
// PIPE: } {triton.warp_pipeline.flat
// PIPE: scf.execute_region
// PIPE: } {triton.warp_pipeline.flat
// PIPE-NEXT: ttg.async_wait
// PIPE-NOT: owner_begin
// PIPE-NOT: owner_end
// PIPE-NOT: triton.warp_pipeline.border
// PIPE: tt.return

// -----

// Layout inputs in an unrolled owner must move past waits. Inputs in the
// persistent loop's prologue and epilogue must keep their positions.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @flattened_owner_layout_inputs(%n: index, %input: tensor<256xf32, #blocked>, %other: tensor<256xf32, #blocked>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %tile = %c0 to %n step %c1 {
      %prologue = ttg.convert_layout %input : tensor<256xf32, #blocked> -> tensor<256xf32, #blocked2>
      ttg.async_wait {num = 1 : i32}
      rocdl.sched.barrier none {triton.warp_pipeline.owner_begin = 7 : i64, triton.warp_pipeline.owner_stage_count = 2 : i64}
      %first = ttg.convert_layout %other : tensor<256xf32, #blocked> -> tensor<256xf32, #blocked2>
      %ready0 = ttg.async_wait {num = 0 : i32}
      ttg.barrier local
      %a = arith.addf %first, %prologue : tensor<256xf32, #blocked2>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}
      %b = arith.addf %a, %prologue : tensor<256xf32, #blocked2>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}
      %second = ttg.convert_layout %b : tensor<256xf32, #blocked2> -> tensor<256xf32, #blocked>
      %ready1 = ttg.async_wait %ready0 {num = 0 : i32}
      %c = arith.addf %second, %input : tensor<256xf32, #blocked>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}
      %d = arith.addf %c, %input : tensor<256xf32, #blocked>
      rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}
      rocdl.sched.barrier none {triton.warp_pipeline.owner_end = 7 : i64}
      %epilogue = ttg.convert_layout %d : tensor<256xf32, #blocked> -> tensor<256xf32, #blocked2>
      ttg.async_wait %ready1 {num = 0 : i32}
      %sum = "tt.reduce"(%epilogue) <{axis = 0 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        tt.reduce.return %sum : f32
      }) : (tensor<256xf32, #blocked2>) -> f32
      tt.store %out, %sum : !tt.ptr<f32>
    }
    tt.return
  }
}

// PREP-LABEL: tt.func @flattened_owner_layout_inputs

// NORM-LABEL: tt.func @flattened_owner_layout_inputs
// NORM: scf.for
// NORM-NEXT: %[[PROLOGUE:.*]] = ttg.convert_layout
// NORM-NEXT: ttg.async_wait
// NORM-NEXT: rocdl.sched.barrier none {triton.warp_pipeline.owner_begin
// NORM-NEXT: %[[READY0:.*]] = ttg.async_wait
// NORM-NEXT: ttg.barrier local
// NORM-NEXT: %[[FIRST:.*]] = ttg.convert_layout
// NORM-NEXT: arith.addf %[[FIRST]]
// NORM: triton.warp_pipeline.border = "stage1"
// NORM-NEXT: %[[READY1:.*]] = ttg.async_wait %[[READY0]]
// NORM-NEXT: %[[SECOND:.*]] = ttg.convert_layout
// NORM-NEXT: arith.addf %[[SECOND]]
// NORM: triton.warp_pipeline.owner_end
// NORM-NEXT: ttg.convert_layout
// NORM-NEXT: ttg.async_wait %[[READY1]]

// PIPE-LABEL: tt.func @flattened_owner_layout_inputs
// PIPE: scf.for
// PIPE-NEXT: ttg.convert_layout
// PIPE-NEXT: ttg.async_wait
// PIPE-NEXT: ttg.async_wait
// PIPE-NEXT: ttg.barrier local
// PIPE-NEXT: {{.*}}scf.execute_region
// PIPE-NEXT: ttg.convert_layout
// PIPE: triton.warp_pipeline.stage = "stage0"
// PIPE: triton.warp_pipeline.stage = "stage1"
// PIPE-NEXT: ttg.async_wait
// PIPE-NEXT: {{.*}}scf.execute_region
// PIPE-NEXT: ttg.convert_layout
// PIPE: triton.warp_pipeline.stage = "stage0"
// PIPE: triton.warp_pipeline.stage = "stage1"
// PIPE-NEXT: ttg.convert_layout
// PIPE-NEXT: ttg.async_wait
// PIPE-NOT: triton.warp_pipeline.pipelined_for
// PIPE-NOT: owner_begin
// PIPE-NOT: owner_end
// PIPE: tt.return
