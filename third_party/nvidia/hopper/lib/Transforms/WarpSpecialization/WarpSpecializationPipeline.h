#ifndef NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATIONPIPELINE_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATIONPIPELINE_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

// Declarations for the internal sub-steps of the `nvgpu-warp-specialization`
// pipeline (orchestrated by WarpSpecialization.cpp). Each step is defined in
// its own .cpp under this directory. Declaring them here — and including this
// header from both the orchestrator and every definition site — gives the
// pipeline API a single, compiler-checked source of truth instead of the
// ad-hoc forward declarations that were previously duplicated across
// translation units. Because each definition now includes this header, a
// declaration that disagrees with its definition (e.g. on return type) is a
// compile error rather than a silently mismatched forward declaration.
//
// Signatures are normalized (WS-03): every step takes `triton::FuncOp` by value
// and reports failure via LogicalResult (or returns void when it cannot fail).
// There is no int/-1 sentinel.

namespace mlir {

// Assigns/normalizes async_task_id across the function.
LogicalResult doTaskIdPropagate(triton::FuncOp funcOp);

// Cross-partition run-once atomic support. Returns failure() when an atomic
// forces a graceful warp-specialization reject (kernel already de-specialized).
LogicalResult doDynamicTileBroadcast(triton::FuncOp funcOp,
                                     int tilePrefetchDepth);

// Test-only knobs for doMemoryPlanner. The production pipeline uses the
// defaults; only the -nvgpu-test-ws-memory-planner pass varies them:
// decision-file I/O to snapshot/replay planner decisions, an alternate SMEM
// allocation algorithm, and circular SMEM reuse. Grouped into a struct so the
// production entry point advertises only the parameters it actually uses
// (WS-10).
struct MemoryPlannerOptions {
  StringRef readDecisionFile = "";
  StringRef writeDecisionFile = "";
  int smemAllocAlgo = 1;
  bool smemCircularReuse = false;
};

// Plans SMEM/TMEM allocation (multi-buffering, liveness). Production passes
// only numBuffers and smemBudget; the test-only knobs default via
// MemoryPlannerOptions.
LogicalResult doMemoryPlanner(triton::FuncOp funcOp, unsigned numBuffers,
                              unsigned smemBudget,
                              const MemoryPlannerOptions &options = {});

void doBufferAllocation(triton::FuncOp funcOp);
void doHoistLoopInvariantTMEMStore(triton::FuncOp funcOp);
void removeRedundantTmemZeroStores(triton::FuncOp funcOp);
void doCodePartition(triton::FuncOp funcOp, unsigned numBuffers);
void doTokenLowering(triton::FuncOp funcOp, unsigned numConsumerGroups);
void doPingPongPrep(triton::FuncOp funcOp, unsigned numWarpGroups,
                    int capability, int defaultNumStages);
void doPingPongSync(triton::FuncOp funcOp, unsigned numWarpGroups,
                    int capability);
void doAnnotateTMAStoreWaits(triton::FuncOp funcOp);
void doValidateTMAStoreAnnotations(triton::FuncOp funcOp);
// Best-effort reordering of annotated TMA store waits; never fails (see the
// definition in WSTMAStoreLowering.cpp).
void doTMAStoreWaitReorder(triton::FuncOp funcOp);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATIONPIPELINE_H_
