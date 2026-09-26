#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::AMD {
bool hasMutuallyExclusiveSuccessorUses(Value value) {
  auto blockArgument = dyn_cast<BlockArgument>(value);
  if (!blockArgument)
    return false;
  Block *header = blockArgument.getOwner();
  auto branch = dyn_cast<cf::CondBranchOp>(header->getTerminator());
  if (!branch || branch.getTrueDest() == branch.getFalseDest())
    return false;

  SmallVector<Block *, 2> consumers;
  for (OpOperand &use : value.getUses()) {
    Block *useBlock = use.getOwner()->getBlock();
    // Do not lift captures in nested regions to their enclosing CFG block.
    if (useBlock == header || useBlock->getParent() != header->getParent() ||
        llvm::is_contained(consumers, useBlock) || consumers.size() == 2)
      return false;
    consumers.push_back(useBlock);
  }
  if (consumers.size() != 2)
    return false;

  // SCF-to-CF may insert an unrelated diamond before the loop's accumulator
  // update. Each arm must still reach its own unique consumer on every path,
  // before any header re-entry. Only acyclic direct-branch prefixes are modeled.
  auto allPathsReachConsumer = [header](Block *from, Block *consumer) {
    llvm::SmallPtrSet<Block *, 16> active;
    llvm::SmallPtrSet<Block *, 16> proven;
    auto visit = [&](auto &&self, Block *block) -> bool {
      if (block == consumer)
        return true;
      if (block == header || block->getParent() != header->getParent())
        return false;
      if (proven.contains(block))
        return true;
      if (!active.insert(block).second)
        return false;
      Operation *terminator = block->getTerminator();
      if (!isa<cf::BranchOp, cf::CondBranchOp>(terminator))
        return false;
      for (Block *successor : terminator->getSuccessors()) {
        if (!self(self, successor))
          return false;
      }
      active.erase(block);
      proven.insert(block);
      return true;
    };
    return visit(visit, from);
  };
  auto matchesArms = [&](Block *first, Block *second) {
    return allPathsReachConsumer(branch.getTrueDest(), first) &&
           allPathsReachConsumer(branch.getFalseDest(), second);
  };
  if (!matchesArms(consumers[0], consumers[1]) &&
      !matchesArms(consumers[1], consumers[0]))
    return false;

  auto isReachableWithoutHeader = [header](Block *from, Block *to) {
    if (from == header)
      return false;
    llvm::SmallPtrSet<Block *, 16> excluded;
    excluded.insert(header);
    return from->isReachable(to, std::move(excluded));
  };
  if (isReachableWithoutHeader(consumers[0], consumers[1]) ||
      isReachableWithoutHeader(consumers[1], consumers[0]))
    return false;
  // An inner cycle must not consume the same dynamic accumulator again. Start
  // at successors so this tests a nonempty path, not reflexive reachability.
  for (Block *consumer : consumers) {
    for (Block *successor : consumer->getSuccessors()) {
      if (isReachableWithoutHeader(successor, consumer))
        return false;
    }
  }
  return true;
}

ElemLocationKey getElemCoordinatesFromRegisters(triton::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx) {
  StringAttr kReg = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  SmallVector<std::pair<StringAttr, int32_t>> hardwareLocation = {
      {kReg, static_cast<int32_t>(regId)},
      {kLane, 0},
      {kWarp, 0},
      {kBlock, 0},
  };

  return ll.apply(hardwareLocation);
}

std::optional<int> getRegFromCoordinates(triton::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx) {
  auto dims = ll.pseudoinvert().apply(coordinates);
  StringAttr kReg = StringAttr::get(ctx, "register");
  assert(dims[0].first == kReg && "First dimension must be 'register'");

  int regId = dims[0].second; // "register"
  if (dims[1].second != 0 || dims[2].second != 0 || dims[3].second != 0)
    return std::nullopt;
  return regId;
}
} // namespace mlir::triton::AMD
