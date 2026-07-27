// Memory-planner-side companion to verify_reuse_group_decisions.mlir (which
// exercises the SAME unified predicate through the code-partition pass). Here we
// drive the predicate from inside the MEMORY PLANNER: TRITON_WS_MEM_PLAN_VERIFY_
// GROUPS makes the planner enumerate every candidate pair/triple of a loop's
// TMEM allocs and run orderReuseGroupChain on each, emitting one
// "[ws-mem-plan-verify] group {..} => ACCEPT/REJECT" line per group. It changes
// NO planning decision, so it is safe to keep on a regular RUN line.
//
// The planner calls orderReuseGroupChain with crossPartitionProgOrder=false --
// the SOUND group-FORMATION gate: an edge between two channels requires a real
// data dependency, or program order WITHIN a single partition; cross-partition
// textual order (which is not a happens-before) is NOT an edge. This refuses to
// order data-independent cross-partition siblings, which is exactly what makes
// the FA-bwd badgroups below rejectable. Code partitioning keeps the default
// (crossPartitionProgOrder=true) and is unaffected -- see
// verify_reuse_group_decisions.mlir.
//
// Fixture reuse: ws_memory_planner_bwd_buffer_reuse.mlir is the only bwd
// memory-planner fixture that puts dsT (the dk operand) in TMEM. Note dsT is NOT
// multi-buffered here: %dsT_0 is a rank-2 single-buffered TMEM copy read by the
// dk MMA, while dq reads a SEPARATE SMEM copy (%dsT_1) via memdesc_trans -- both
// derived from the same source value but distinct base buffers. So the dk side
// forms a real dpT->dsT_0->dk chain, while dq is a common-ancestor sibling with
// no dsT_0->dq edge (in the live BM128 kernel dq reads the TMEM dsT, so there
// {dpT,dsT,dq} ACCEPTs -- see verify_reuse_group_decisions.mlir CASE 3).
//
// No assertions build needed: the harness prints via llvm::errs(), not LDBG, so
// it runs on any build configuration.
//
// RUN: env TRITON_WS_MEM_PLAN_VERIFY_GROUPS=1 triton-opt %S/ws_memory_planner_bwd_buffer_reuse.mlir --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=231000" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// Real N-way chain still ACCEPTs (proves the sound gate is not rejecting
// everything): dpT -> dsT_0 -> dk are genuine data dependencies.
// CHECK-DAG: [ws-mem-plan-verify] group {dk,dpT,dsT_0} => ACCEPT order=dpT->dsT_0->dk

// Badgroup 1 -- qkT is independent of dpT: no data dep, and their producers are
// cross-partition, so cross-partition textual order is (correctly) not an edge.
// CHECK-DAG: [ws-mem-plan-verify] group {qkT,dpT,dq} => REJECT

// Badgroup 2 -- ppT is independent of dsT (the {qkT,ppT,dsT} group the planner
// wrongly formed for BM128): rejected for the same reason.
// CHECK-DAG: [ws-mem-plan-verify] group {qkT,ppT,dsT_0} => REJECT

// Target {dpT,dsT,dq}: rejected IN THIS FIXTURE because dq reads the SMEM copy
// (%dsT_1), so there is no dsT_0->dq edge. On the live BM128 kernel dq reads the
// TMEM dsT and this group ACCEPTs (dpT->dsT_0->dq).
// CHECK-DAG: [ws-mem-plan-verify] group {dpT,dq,dsT_0} => REJECT
