// RUN: triton-opt %s -split-input-file -tritonamdgpu-preserve-warp-pipeline-owners | FileCheck %s --check-prefix=PREP
// RUN: triton-opt %s -split-input-file -tritonamdgpu-preserve-warp-pipeline-owners -canonicalize -tritonamdgpu-warp-pipeline | FileCheck %s --check-prefix=PIPE

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
