// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -nvgpu-modulo-schedule | FileCheck %s
// RUN: env TRITON_USE_MODULO_SCHEDULE=joint_solver triton-opt %s -split-input-file -allow-unregistered-dialect -nvgpu-modulo-schedule | FileCheck %s --check-prefix=JOINT
// RUN: env TRITON_MODULO_BASELINE_REPORT=1 triton-opt %s -split-input-file -allow-unregistered-dialect -nvgpu-modulo-schedule 2>&1 | FileCheck %s --check-prefix=BASELINE

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// Verify that the modulo schedule pass annotates the inner loop with its
// scheduling decision. For a single-MMA GEMM, all MMAs are in the same stage
// so tt.autows is skipped. The default run only checks the public attributes;
// the JOINT run freezes the schedule extracted from this DDG by native Z3.
//
// CHECK-LABEL: @gemm_inner_loop
// CHECK: scf.for
// CHECK-NOT: tt.autows
// CHECK: tt.modulo_ii = {{[0-9]+}} : i32
// CHECK-SAME: tt.scheduled_max_stage = {{[0-9]+}} : i32
//
// JOINT-LABEL: tt.func @gemm_inner_loop
// JOINT: arith.muli {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
// The two operand loads are symmetric on the TMA pipeline, so their relative
// cluster order is a tie-break, not a scheduling decision. This diff wraps
// every constraint as `assumption_literal -> constraint` and solves via
// Z3_solver_check_assumptions (see AssumptionTracker, added here for UNSAT
// core extraction), which changes the formula Z3 searches and therefore which
// of the two it places first. II, stages and partitions are unchanged, so the
// schedule is equivalent — only these cluster indices flip relative to the
// parent revision.
// JOINT: tt.descriptor_load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
// JOINT: tt.descriptor_load {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
// JOINT: ttg.local_alloc {{.*}} {buffer.id = 0 : i32, buffer.merge_group_id = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.num_buffers = 2 : i32, ttg.partition = array<i32: 1>}
// JOINT: ttg.local_alloc {{.*}} {buffer.id = 1 : i32, buffer.merge_group_id = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.num_buffers = 2 : i32, ttg.partition = array<i32: 2>}
// JOINT: ttng.tmem_alloc {{.*}} {buffer.id = 2 : i32, buffer.merge_group_id = 2 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.num_buffers = 2 : i32, ttg.partition = array<i32: 3>}
// JOINT: ttng.tc_gen5_mma {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
// The epilogue tmem_load gets a warp group of its own (4), separate from the
// MMA's (3), once the solver's warp-group assignment is applied — hence five
// partitions, not four.
// JOINT: ttng.tmem_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>}
// JOINT: } {tt.modulo_ii = 1091 : i32, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.partition_num_warps = array<i32: 4, 1, 1, 1, 4>, ttg.warp_specialize.tag = 0 : i32}
//
// M2 acceptance evidence on this same GEMM DDG. The IIs are solver- and
// machine-dependent so they are matched as numbers, not pinned: what is
// pinned is that the table is emitted, that both criteria are evaluated, and
// that neither reports FAIL. SKIP is accepted because the joint rows drop out
// in a build without Z3 (TRITON_ENABLE_Z3_JOINT_SOLVER off), where the
// baselines still run but there is nothing to compare them against.
//
// On this fixture every backend lands on MinII, so `strict improvement` is
// NO — the loop is MinII-bound and cannot separate schedulers. That is
// checked as-is rather than tolerated: if a future change makes the solver
// beat Rau here, this line must be updated deliberately, with the new number
// looked at, instead of silently flipping.
//
// BASELINE: modulo-baseline-report: gemm_inner_loop
// BASELINE: fixture{{ +}}MinII{{ +}}II_full{{ +}}relaxed_LB{{ +}}II_rau{{ +}}II_sms{{ +}}II_exhaustive
// BASELINE: gemm_inner_loop{{ +}}{{[0-9]+}}{{ +}}{{[0-9-]+}}{{ +}}{{[0-9-]+}}{{ +}}{{[0-9-]+}}
// BASELINE: soundness (II_full >= relaxed_LB): {{PASS|SKIP}}
// BASELINE: no-regression (II_full <= II_rau): {{PASS|SKIP}}
// BASELINE: strict improvement (II_full < II_rau): {{NO|SKIP}}
tt.func @gemm_inner_loop(
  %a_desc: !tt.tensordesc<128x64xf16>,
  %b_desc: !tt.tensordesc<64x128xf16>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %k_tiles = arith.constant 32 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> (tensor<128x128xf32, #acc_layout>) : i32 {
    %off_k = arith.muli %k, %c1_i32 : i32

    %a = tt.descriptor_load %a_desc[%c0_i32, %off_k] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
    %b = tt.descriptor_load %b_desc[%off_k, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked>

    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    scf.yield %c : tensor<128x128xf32, #acc_layout>
  }

  tt.return
}

}
