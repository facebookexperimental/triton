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
// NOTE: this header intentionally does NOT change any signature — it only
// centralizes the existing declarations. Signature normalization (uniform
// return type / by-value FuncOp) is tracked separately (WS-03).

namespace mlir {

// Assigns/normalizes async_task_id across the function. Returns -1 on failure.
int doTaskIdPropagate(triton::FuncOp &funcOp);

// Cross-partition run-once atomic support. Returns failure() when an atomic
// forces a graceful warp-specialization reject (kernel already de-specialized).
LogicalResult doDynamicTileBroadcast(triton::FuncOp funcOp,
                                     int tilePrefetchDepth);

// Plans SMEM/TMEM allocation (multi-buffering, liveness). The decision-file and
// algorithm knobs are exercised only by the -nvgpu-test-ws-memory-planner test
// pass; the production pipeline uses the defaults declared here. Defaults live
// on this declaration only (not on the definition) so there is one source.
LogicalResult doMemoryPlanner(triton::FuncOp &funcOp, unsigned numBuffers,
                              StringRef readDecisionFile = "",
                              StringRef writeDecisionFile = "",
                              int smemAllocAlgo = 1, unsigned smemBudget = 0,
                              bool smemCircularReuse = false);

void doBufferAllocation(triton::FuncOp &funcOp);
void doHoistLoopInvariantTMEMStore(triton::FuncOp &funcOp);
void removeRedundantTmemZeroStores(triton::FuncOp &funcOp);
void doCodePartitionPost(triton::FuncOp &funcOp, unsigned numBuffers);
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doPingPongPrep(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability, int defaultNumStages);
void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability);
void doAnnotateTMAStoreWaits(triton::FuncOp &funcOp);
void doValidateTMAStoreAnnotations(triton::FuncOp &funcOp);
// Best-effort reordering of annotated TMA store waits; never fails (see the
// definition in WSTMAStoreLowering.cpp).
void doTMAStoreWaitReorder(triton::FuncOp &funcOp);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_WARPSPECIALIZATIONPIPELINE_H_
