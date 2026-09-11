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
  Operation *terminator = blockArgument.getOwner()->getTerminator();
  // This is the exact shape produced when SCF-to-CF lowers a runtime loop.
  // Do not generalize the exception to switches or other multiway CFGs.
  if (!isa<cf::CondBranchOp>(terminator) ||
      terminator->getNumSuccessors() != 2)
    return false;

  llvm::DenseSet<Block *> successorBlocks;
  for (unsigned index = 0; index < terminator->getNumSuccessors(); ++index)
    successorBlocks.insert(terminator->getSuccessor(index));

  llvm::DenseSet<Block *> usedSuccessors;
  for (OpOperand &use : value.getUses()) {
    Block *useBlock = use.getOwner()->getBlock();
    if (!successorBlocks.contains(useBlock) ||
        !usedSuccessors.insert(useBlock).second)
      return false;
  }
  if (usedSuccessors.size() != 2)
    return false;

  SmallVector<Block *> blocks(usedSuccessors.begin(), usedSuccessors.end());
  Block *header = blockArgument.getOwner();
  auto isReachableWithoutHeader = [header](Block *from, Block *to) {
    llvm::SmallPtrSet<Block *, 16> excluded;
    excluded.insert(header);
    return from->isReachable(to, std::move(excluded));
  };
  for (auto [index, lhs] : llvm::enumerate(blocks)) {
    for (Block *rhs : ArrayRef(blocks).drop_front(index + 1)) {
      if (isReachableWithoutHeader(lhs, rhs) ||
          isReachableWithoutHeader(rhs, lhs))
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
