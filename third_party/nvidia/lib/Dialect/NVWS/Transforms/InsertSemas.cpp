// Four-stage dispatcher; see sema-docs/insert-semas/overview.md.
#include "InsertSemas.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"

namespace mlir::triton {
#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {
using namespace nvws_semas;

LogicalResult runOnFunction(triton::FuncOp funcOp, bool useMetaPartitioner, int lowerSemaphoreNumStages) {
  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(triton::kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted())
    return success();

  FailureOr<SmallVector<GroupDag, 0>> groupsOr = collectGroups(funcOp);
  if (failed(groupsOr))
    return failure();
  SmallVector<GroupDag, 0> groups = std::move(*groupsOr);
  for (GroupDag &g : groups)
    if (failed(buildAccessDag(g, funcOp)))
      return failure();
  for (GroupDag &g : groups)
    if (failed(buildOwnerDag(g)))
      return failure();
  int numTmemBlocks = 0;
  for (GroupDag &g : groups)
    if (failed(buildSyncDag(g, useMetaPartitioner, lowerSemaphoreNumStages, numTmemBlocks)))
      return failure();
  if (failed(finalizeSyncSchedule(groups)))
    return failure();
  if (shouldDumpDag()) {
    llvm::errs() << "==== NVWS InsertSemas (commit 4: ACCESS-DAG + " "OWNER-DAG + SYNC-DAG + EMIT) ====\n";
    llvm::errs() << "function: @" << funcOp.getName() << "\n";
    llvm::errs() << "groups: " << groups.size() << "\n";
    for (GroupDag &g : groups) {
      dumpGroupAccessDag(g, funcOp);
      dumpGroupOwnerDag(g, funcOp);
      dumpGroupSyncDag(g, funcOp);
    }
  }
  return emitIR(funcOp, groups);
}
} // namespace

class NVWSInsertSemas : public triton::impl::NVWSInsertSemasBase<NVWSInsertSemas> {
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
