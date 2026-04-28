#include "InsertSemas.h"
#include "MetaToNVWSConvert.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"

namespace mlir::triton {
#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {
using namespace nvws_semas;

LogicalResult runOnFunction(triton::FuncOp funcOp, bool useMetaPartitioner,
                            int lowerSemaphoreNumStages) {
  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(triton::kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted())
    return success();

  if (failed(validateNVWSManagedAllocationLocality(funcOp,
                                                    "nvws-insert-semas")))
    return failure();

  SmallVector<GroupDag, 0> candidate;
  for (Block &functionBlock : funcOp.getBody()) {
    FailureOr<SmallVector<GroupDag, 0>> groupsOr =
        collectGroups(funcOp, &functionBlock);
    if (failed(groupsOr))
      return failure();
    for (GroupDag &group : *groupsOr) {
      if (failed(buildAccessDag(group, funcOp, functionBlock)))
        return failure();
      candidate.push_back(std::move(group));
    }
  }

  int numTmemBlocks = 0;
  for (GroupDag &g : candidate)
    if (failed(buildSyncDag(g, useMetaPartitioner, lowerSemaphoreNumStages,
                            numTmemBlocks)))
      return failure();
  SmallVector<ScheduleUpdate> scheduleUpdates;
  if (failed(finalizeSyncSchedule(candidate, scheduleUpdates)))
    return failure();
  dumpSyncDags(candidate, funcOp);
  return emitIR(funcOp, candidate, scheduleUpdates);
}
} // namespace

class NVWSInsertSemas
    : public triton::impl::NVWSInsertSemasBase<NVWSInsertSemas> {
public:
  using NVWSInsertSemasBase::NVWSInsertSemasBase;
  void runOnOperation() override {
    auto walkResult = getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp, useMetaPartitioner, numStages)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace mlir::triton
