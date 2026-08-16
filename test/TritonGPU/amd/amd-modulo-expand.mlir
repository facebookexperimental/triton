// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule=mode=modulo 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SCHEDULE
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule=mode=modulo 2>/dev/null \
// RUN:   | triton-opt -split-input-file -tritonamdgpu-pipeline 2>&1 \
// RUN:   | FileCheck %s
// RUN: triton-opt %s -tritonamdgpu-dump-plan-value-graph='output-path=- strict=true' \
// RUN:   -o /dev/null | FileCheck %s --check-prefix=PLAN
// RUN: triton-opt %s -tritonamdgpu-dump-plan-value-graph='output-path=- strict=true' \
// RUN:   -o /dev/null > %t.plan.original
// RUN: sed 's/%a_ptrs/%renamed_ptrs/g; s/%a_ld/%renamed_load/g' %s \
// RUN:   | triton-opt -tritonamdgpu-dump-plan-value-graph='output-path=- strict=true' \
// RUN:     -o /dev/null > %t.plan.renamed
// RUN: diff %t.plan.original %t.plan.renamed
// RUN: triton-opt %s -o %t.no-analysis.mlir
// RUN: triton-opt %s \
// RUN:   -tritonamdgpu-dump-plan-value-graph='output-path=%t.plan.json strict=true' \
// RUN:   -o %t.with-analysis.mlir
// RUN: diff %t.no-analysis.mlir %t.with-analysis.mlir
// RUN: triton-opt %S/../../Conversion/amd/in_thread_transpose.mlir \
// RUN:   -split-input-file \
// RUN:   -tritonamdgpu-dump-plan-value-graph='output-path=- strict=true' \
// RUN:   -o /dev/null | FileCheck %s --check-prefix=PLAN-TRANSPOSE
// RUN: triton-opt %s \
// RUN:   -tritonamdgpu-dump-plan-value-graph='output-path=%t.schedule.plan.json strict=true' \
// RUN:   -o /dev/null
// RUN: env PYTHONPATH=%S/../../../third_party/tlx/tools/plan_ir \
// RUN:   python3 -m tlx_plan.cli schedule-delta \
// RUN:   --value-graph %t.schedule.plan.json --kernel schedule_apply_fixture \
// RUN:   --output %t.schedule.identity.json
// RUN: python3 -c "import json; p=json.load(open('%t.schedule.identity.json')); o=p['blocks'][0]['desired_order']; o[0],o[1]=o[1],o[0]; open('%t.schedule.legal.json','w').write(json.dumps(p))"
// RUN: triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-schedule='input-path=%t.schedule.legal.json report-path=%t.schedule.report.json strict=true' \
// RUN:   | FileCheck %s --check-prefix=APPLY
// RUN: FileCheck %s --check-prefix=APPLY-REPORT < %t.schedule.report.json
// RUN: triton-opt %S/../../Conversion/amd/in_thread_transpose.mlir \
// RUN:   -split-input-file \
// RUN:   -tritonamdgpu-apply-plan-schedule='input-path=%t.schedule.legal.json strict=true allow-missing-kernel=true' \
// RUN:   -o /dev/null
// RUN: python3 -c "import json; p=json.load(open('%t.schedule.identity.json')); o=p['blocks'][0]['desired_order']; o[1],o[2]=o[2],o[1]; open('%t.schedule.invalid.json','w').write(json.dumps(p))"
// RUN: not triton-opt %s \
// RUN:   -tritonamdgpu-apply-plan-schedule='input-path=%t.schedule.invalid.json strict=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=APPLY-REJECT
//
// Modulo runs before the guarded legacy scheduler. A successful modulo schedule
// is preserved; the standard AMD pipeline lowers and expands it.

// SCHEDULE: remark: amd-modulo:{{.*}}II={{[0-9]+}} maxStage=1{{.*}}serialized num_stages=2
// SCHEDULE-NOT: triton.warp_pipeline.border
// CHECK-LABEL: tt.func @early_lower
// The standard pipeline may choose register pipelining when this single-load
// fixture is not profitable/legal for LDS async copy. Verify the load is peeled
// into the prologue and forwarded as a loop-carried tensor to the stage-1 dot.
// CHECK:       tt.load {{.*}}amd.pipeliner_part = "prologue"
// CHECK:       scf.for {{.*}}iter_args({{.*}}tensor<256x64xf16, #blocked>)
// CHECK:         tt.load
// CHECK:         ttg.convert_layout {{.*}}tensor<256x64xf16, #blocked>
// CHECK:         tt.dot

// PLAN-DAG: "schema_version": "plan-value-graph/0.4"
// PLAN-DAG: "kind": "loop_init"
// PLAN-DAG: "iteration_distance": 1
// PLAN-DAG: "kind": "loop_backedge"
// PLAN-DAG: "kind": "loop_exit"
// PLAN-DAG: "kind": "loop_forward"
// PLAN-DAG: "kind": "branch_yield"
// PLAN-DAG: "kind": "convert_layout"
// PLAN-DAG: "logical_bytes": 262144
// PLAN-DAG: "physical_register_bytes": null
// PLAN-DAG: "artifact_stage": "final_structured_ttgir"
// PLAN-DAG: "static_intervals_are_physical_cycles": false
// PLAN-DAG: "lds_logical_bytes_are_physical_allocation": false
// PLAN-DAG: "async_lifetime_extended_through_wait": true
// PLAN-DAG: "async_lifetimes_are_physical_cycles": false
// PLAN-DAG: "live_segments": [
// PLAN-DAG: "view_kind": "ttg.memdesc_index"
// PLAN-DAG: "kind": "modulo"
// PLAN-DAG: "modulus": 2
// PLAN-DAG: "possible_slots": [
// PLAN-DAG: "effect": "read"
// PLAN-DAG: "effect": "write"
// PLAN-DAG: "effect": "allocate"
// PLAN-DAG: "effect": "free"
// PLAN-DAG: "physical_lds_offset": null
// PLAN-DAG: "direction": "lds_write"
// PLAN-DAG: "retained_group_count": 1
// PLAN-DAG: "kind": "completion_wait"
// PLAN-DAG: "kind": "visibility_barrier"
// PLAN-DAG: "kind": "lds_consumer"
// PLAN-DAG: "kind": "reuse_release_barrier"
// PLAN-DAG: "kind": "slot_overwrite"
// PLAN-DAG: "iteration_distance": 2
// PLAN-DAG: "precision": "conservative_cross_region"
// PLAN-DAG: "code": "async_write_without_completion"
// PLAN-DAG: "kind": "loop_carried_ssa"
// PLAN-DAG: "kind": "memory_raw"
// PLAN-DAG: "kind": "memory_war"
// PLAN-DAG: "kind": "memory_waw"
// PLAN-DAG: "kind": "async_completion"
// PLAN-DAG: "kind": "barrier_visibility"
// PLAN-DAG: "kind": "consumer_release"
// PLAN-DAG: "kind": "slot_reuse"
// PLAN-DAG: "peak_live_sets": [
// PLAN-DAG: "logical_tensor_bytes_are_per_wave_vgpr_bytes": false
// PLAN-DAG: "physical_vgpr_peak": null
// PLAN-DAG: "physical_lds_bytes": null
// PLAN-DAG: "max_logical_slot_depth": 2
// PLAN-DAG: "importance": "important"
// PLAN-DAG: "status": "open"
// PLAN-TRANSPOSE: "kind": "in_thread_transpose"
// APPLY-LABEL: tt.func @schedule_apply_fixture
// APPLY: %[[C2:.*]] = arith.constant 2 : i32
// APPLY-NEXT: %[[C1:.*]] = arith.constant 1 : i32
// APPLY-NEXT: arith.addi %[[C1]], %[[C2]]
// APPLY-REPORT: "accepted": true
// APPLY-REPORT: "moved_operations": 2
// APPLY-REJECT: schedule delta reverses distance-zero dependency

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#slot_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#slot_shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @early_lower(
      %a_ptrs: tensor<256x64x!tt.ptr<f16>, #blocked>,
      %b: tensor<64x256xf16, #dot1>,
      %c_init: tensor<256x256xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<256x256xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c_init)
        -> (tensor<256x256xf32, #mma>) {
      %a_ld = tt.load %a_ptrs : tensor<256x64x!tt.ptr<f16>, #blocked>
      %a = ttg.convert_layout %a_ld : tensor<256x64xf16, #blocked>
              -> tensor<256x64xf16, #dot0>
      %d = tt.dot %a, %b, %acc :
          tensor<256x64xf16, #dot0> * tensor<64x256xf16, #dot1>
          -> tensor<256x256xf32, #mma>
      scf.yield %d : tensor<256x256xf32, #mma>
    }
    tt.return %res : tensor<256x256xf32, #mma>
  }

  tt.func @structured_control(%cond: i1, %initial: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %selected = scf.if %cond -> i32 {
      %then = arith.addi %initial, %c1 : i32
      scf.yield %then : i32
    } else {
      %else = arith.subi %initial, %c1 : i32
      scf.yield %else : i32
    }
    %result = scf.while (%iter = %selected) : (i32) -> i32 {
      %keep_going = arith.cmpi slt, %iter, %initial : i32
      scf.condition(%keep_going) %iter : i32
    } do {
    ^bb0(%iter: i32):
      %next = arith.addi %iter, %c1 : i32
      scf.yield %next : i32
    }
    tt.return %result : i32
  }

  tt.func @lds_modulo_slots(%data: tensor<16x16xf16, #slot_blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable>
    scf.for %i = %c0 to %c4 step %c1 {
      %i_i32 = arith.index_cast %i : index to i32
      %current_index = arith.remsi %i_i32, %c2_i32 : i32
      %previous_index = arith.subi %c1_i32, %current_index : i32
      %current = ttg.memdesc_index %alloc[%current_index] : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      %previous = ttg.memdesc_index %alloc[%previous_index] : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      ttg.local_store %data, %current : tensor<16x16xf16, #slot_blocked> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      %loaded = ttg.local_load %previous : !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable> -> tensor<16x16xf16, #slot_blocked>
    }
    ttg.local_dealloc %alloc : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable>
    tt.return
  }

  tt.func @async_lds_modulo_slots(
      %ptrs: tensor<16x16x!tt.ptr<f16>, #slot_blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable>
    scf.for %i = %c0 to %c4 step %c1 {
      %i_i32 = arith.index_cast %i : index to i32
      %current_index = arith.remsi %i_i32, %c2_i32 : i32
      %previous_index = arith.subi %c1_i32, %current_index : i32
      %current = ttg.memdesc_index %alloc[%current_index] : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      %previous = ttg.memdesc_index %alloc[%previous_index] : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
      %copy = ttg.async_copy_global_to_local %ptrs, %current : tensor<16x16x!tt.ptr<f16>, #slot_blocked> -> <16x16xf16, #slot_shared, #smem, mutable>
      %commit = ttg.async_commit_group tokens %copy
      %wait = ttg.async_wait {num = 1 : i32}
      ttg.barrier all
      %loaded = ttg.local_load %previous token %wait : !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable> -> tensor<16x16xf16, #slot_blocked>
      ttg.barrier all
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    ttg.barrier all
    ttg.local_dealloc %alloc : !ttg.memdesc<2x16x16xf16, #slot_shared, #smem, mutable>
    tt.return
  }

  tt.func @async_branch(
      %cond: i1, %ptrs: tensor<16x16x!tt.ptr<f16>, #slot_blocked>) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
    %outer_copy = ttg.async_copy_global_to_local %ptrs, %alloc : tensor<16x16x!tt.ptr<f16>, #slot_blocked> -> <16x16xf16, #slot_shared, #smem, mutable>
    %outer_commit = ttg.async_commit_group tokens %outer_copy
    scf.if %cond {
      %inner_copy = ttg.async_copy_global_to_local %ptrs, %alloc : tensor<16x16x!tt.ptr<f16>, #slot_blocked> -> <16x16xf16, #slot_shared, #smem, mutable>
      %inner_commit = ttg.async_commit_group tokens %inner_copy
    }
    %wait = ttg.async_wait %outer_commit {num = 0 : i32}
    ttg.barrier all
    ttg.local_dealloc %alloc : !ttg.memdesc<16x16xf16, #slot_shared, #smem, mutable>
    tt.return
  }

  tt.func @schedule_apply_fixture() -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %sum = arith.addi %c1, %c2 : i32
    tt.return %sum : i32
  }
}
