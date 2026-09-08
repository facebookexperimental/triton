#include "InsertSemas.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"

namespace mlir::triton {
#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {
using namespace nvws_semas;

static Block *getTopLevelFunctionBlock(Operation *op, triton::FuncOp funcOp) {
  Block *block = op->getBlock();
  while (block && block->getParent() != &funcOp.getBody()) {
    Operation *parent = block->getParentOp();
    block = parent ? parent->getBlock() : nullptr;
  }
  return block;
}

static FailureOr<Block *> getManagedGroupUseBlock(const GroupDag &group,
                                                  triton::FuncOp funcOp) {
  Operation *anchor = group.pieceTable.members.front().allocOp;
  Block *fallbackBlock = nullptr;
  Block *useBlock = nullptr;
  SmallVector<Value, 8> worklist;
  DenseSet<Value> seen;
  for (const Member &member : group.pieceTable.members) {
    Block *block = getTopLevelFunctionBlock(member.allocOp, funcOp);
    if (!block)
      return semaError(member.allocOp)
             << "managed allocation is not nested in the function body";
    if (!fallbackBlock)
      fallbackBlock = block;
    for (Value result : member.allocOp->getResults())
      worklist.push_back(result);
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<BranchOpInterface>(user))
        return semaError(anchor)
               << "managed memdesc flow through function CFG block arguments "
                  "is unsupported";
      Block *block = getTopLevelFunctionBlock(user, funcOp);
      if (!block)
        return semaError(anchor)
               << "managed use is not nested in the function body";
      if (useBlock && useBlock != block)
        return semaError(anchor)
               << "managed memdesc flow across function CFG blocks is "
                  "unsupported";
      useBlock = block;

      if (isSupportedAliasOp(user))
        for (Value result : user->getResults())
          if (isa<gpu::MemDescType>(result.getType()))
            worklist.push_back(result);
      for (Value result : user->getResults())
        if (isa<gpu::AsyncTokenType>(result.getType()))
          worklist.push_back(result);

      if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        Operation *parent = yield->getParentOp();
        unsigned index = use.getOperandNumber();
        if (index < parent->getNumResults() &&
            isa<gpu::AsyncTokenType>(parent->getResult(index).getType()))
          worklist.push_back(parent->getResult(index));
      } else if (auto condition = dyn_cast<scf::ConditionOp>(user)) {
        unsigned index = use.getOperandNumber();
        auto whileOp = dyn_cast<scf::WhileOp>(condition->getParentOp());
        if (whileOp && index > 0 && index - 1 < whileOp->getNumResults() &&
            isa<gpu::AsyncTokenType>(whileOp->getResult(index - 1).getType()))
          worklist.push_back(whileOp->getResult(index - 1));
      }
    }
  }
  return useBlock ? useBlock : fallbackBlock;
}

static LogicalResult verifyManagedGroupBlockLocality(const GroupDag &group,
                                                     triton::FuncOp funcOp,
                                                     Block *expectedBlock) {
  Operation *anchor = group.pieceTable.members.front().allocOp;
  Block *definitionBlock = nullptr;
  for (const Member &member : group.pieceTable.members) {
    Block *block = getTopLevelFunctionBlock(member.allocOp, funcOp);
    if (!block)
      return semaError(member.allocOp)
             << "managed allocation is not nested in the function body";
    if (definitionBlock && definitionBlock != block)
      return semaError(anchor) << "one buffer group spans function CFG blocks";
    definitionBlock = block;
  }
  if (definitionBlock != expectedBlock)
    return semaError(anchor)
           << "managed memdesc flow across function CFG blocks is unsupported";
  return success();
}

// InsertSemas plans each top-level function CFG block independently. Verify
// that every managed allocation group and its complete memdesc/token use
// closure satisfy that block-local representation contract.
static LogicalResult validateManagedAllocationLocality(triton::FuncOp funcOp) {
  FailureOr<SmallVector<GroupDag, 0>> groupsOr = collectGroups(funcOp);
  if (failed(groupsOr))
    return failure();
  for (const GroupDag &group : *groupsOr) {
    FailureOr<Block *> useBlock = getManagedGroupUseBlock(group, funcOp);
    if (failed(useBlock) ||
        failed(verifyManagedGroupBlockLocality(group, funcOp, *useBlock)))
      return failure();
  }
  return success();
}

static LogicalResult validateDepth(Operation *op, StringRef name,
                                   int64_t depth) {
  if (depth >= 1 && depth <= 32)
    return success();
  semaError(op) << name << " must be in [1, 32], got " << depth;
  return failure();
}

static LogicalResult validateDepths(triton::FuncOp funcOp, int numStages) {
  if (failed(validateDepth(funcOp, "num-stages", numStages)))
    return failure();
  WalkResult result = funcOp.walk([&](Operation *op) {
    auto copies = op->getAttrOfType<IntegerAttr>(kBufferCopyAttrName);
    if (!copies)
      return WalkResult::advance();
    if (succeeded(validateDepth(op, "buffer.copy", copies.getInt())))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  return success(!result.wasInterrupted());
}

LogicalResult runOnFunction(triton::FuncOp funcOp, bool useMetaPartitioner,
                            int semaphoreOptimizeNumStages) {
  if (failed(validateDepths(funcOp, semaphoreOptimizeNumStages)))
    return failure();

  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(triton::kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted())
    return success();

  if (failed(validateManagedAllocationLocality(funcOp)))
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
    if (failed(buildSyncDag(g, useMetaPartitioner, semaphoreOptimizeNumStages,
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
