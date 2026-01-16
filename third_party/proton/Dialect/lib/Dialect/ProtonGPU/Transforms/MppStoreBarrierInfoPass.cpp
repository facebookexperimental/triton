#include "Dialect/ProtonGPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "mpp-store-barrier-info"

namespace mlir::triton::proton::gpu {

#define GEN_PASS_DEF_MPPSTOREBARRIERINFOPASS
#include "Dialect/ProtonGPU/Transforms/Passes.h.inc"

namespace {

struct BarrierInfo {
  int64_t allocOpId = -1;
  bool hasIndex = false;
  int64_t constantIndex = -1;
  Value dynamicIndex = nullptr;
  int dynamicIndexYieldPosition = -1;
  bool hasAdjacentIndex = false;

  BarrierInfo() = default;
  explicit BarrierInfo(int64_t id) : allocOpId(id) {}

  BarrierInfo withConstantIndex(int64_t idx) const {
    BarrierInfo n = *this;
    n.hasIndex = true;
    n.constantIndex = idx;
    n.dynamicIndex = nullptr;
    return n;
  }

  BarrierInfo withDynamicIndex(Value idx, int yieldPos = -1) const {
    BarrierInfo n = *this;
    n.hasIndex = true;
    n.constantIndex = -1;
    n.dynamicIndex = idx;
    n.dynamicIndexYieldPosition = yieldPos;
    return n;
  }

  BarrierInfo withAdjacentIndex() const {
    BarrierInfo n = *this;
    n.hasAdjacentIndex = true;
    return n;
  }
};

int64_t getMppOpId(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("mpp.op.id"))
    return attr.getInt();
  return -1;
}

std::optional<int64_t> getConstantIntValue(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(c.getValue()))
      return intAttr.getInt();
  return std::nullopt;
}

bool isBarrierType(Type type) {
  auto memdescType = dyn_cast<triton::gpu::MemDescType>(type);
  return memdescType && memdescType.getElementType().isInteger(64);
}

Value getBarrierOperand(Operation *op, int idx) {
  if (auto o = dyn_cast<triton::nvidia_gpu::WaitBarrierOp>(op))
    return o.getAlloc();
  if (auto o = dyn_cast<triton::nvidia_gpu::ArriveBarrierOp>(op))
    return o.getAlloc();
  if (auto o = dyn_cast<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(op))
    return o.getBarrier();
  if (auto o = dyn_cast<triton::nvidia_gpu::TCGen5MMAOp>(op)) {
    auto b = o.getBarriers();
    return (idx >= 0 && idx < (int)b.size()) ? b[idx]
           : b.empty()                       ? nullptr
                                             : b[0];
  }
  if (auto o = dyn_cast<triton::nvidia_gpu::TCGen5CommitOp>(op))
    return o.getBarrier();
  return nullptr;
}

} // namespace

struct MppStoreBarrierInfoPass
    : public impl::MppStoreBarrierInfoPassBase<MppStoreBarrierInfoPass> {

  using impl::MppStoreBarrierInfoPassBase<
      MppStoreBarrierInfoPass>::MppStoreBarrierInfoPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    transformLoopsToTrackIndices(module, builder);
    transformCfBlocksToTrackIndices(module, builder);
    propagateBarrierInfo(module);

    for (auto func : module.getOps<triton::FuncOp>())
      if (failed(processFunction(func, builder)))
        return signalPassFailure();
  }

private:
  DenseMap<Value, BarrierInfo> valueToBarrierInfo;
  DenseSet<Value> visitedValues;

  //===--------------------------------------------------------------------===//
  // Loop Transformation - Track indices alongside barrier iter_args
  //===--------------------------------------------------------------------===//

  void transformLoopsToTrackIndices(ModuleOp module, OpBuilder &builder) {
    SmallVector<scf::ForOp> forOps;
    module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (auto forOp : forOps)
      transformSingleLoop(forOp, builder);
  }

  void transformSingleLoop(scf::ForOp forOp, OpBuilder &builder) {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

    // Find barrier iter_args that need index tracking
    SmallVector<std::tuple<unsigned, Value, Value>> indicesToInsert;

    for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
      Value yieldedValue = yieldOp.getOperand(i);
      auto yieldIndexOp =
          yieldedValue.getDefiningOp<triton::gpu::MemDescIndexOp>();
      if (!isBarrierType(yieldedValue.getType()) || !yieldIndexOp)
        continue;

      Value yieldIndexValue = yieldIndexOp.getIndex();
      if (i + 1 < yieldOp.getNumOperands() &&
          yieldOp.getOperand(i + 1) == yieldIndexValue)
        continue;

      auto initIndexOp =
          forOp.getInitArgs()[i].getDefiningOp<triton::gpu::MemDescIndexOp>();
      unsigned adjustedPos =
          i + 1 + llvm::count_if(indicesToInsert, [i](auto &t) {
            return std::get<0>(t) <= i;
          });
      indicesToInsert.push_back(
          {adjustedPos, yieldIndexValue,
           initIndexOp ? initIndexOp.getIndex() : nullptr});
    }

    if (indicesToInsert.empty())
      return;

    Location loc = forOp.getLoc();
    SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                   forOp.getInitArgs().end());
    SmallVector<Value> newYieldOperands(yieldOp.getOperands().begin(),
                                        yieldOp.getOperands().end());

    // Insert in reverse order
    for (auto it = indicesToInsert.rbegin(); it != indicesToInsert.rend();
         ++it) {
      auto [pos, yieldIndexValue, initIndexValue] = *it;
      if (pos <= newYieldOperands.size())
        newYieldOperands.insert(newYieldOperands.begin() + pos,
                                yieldIndexValue);

      builder.setInsertionPoint(forOp);
      Value initValue =
          initIndexValue
              ? (initIndexValue.getType().isInteger(32)
                     ? initIndexValue
                     : builder.create<arith::IndexCastOp>(
                           loc, builder.getI32Type(), initIndexValue))
              : builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                                  builder.getI32IntegerAttr(0));
      if (pos <= newInitArgs.size())
        newInitArgs.insert(newInitArgs.begin() + pos, initValue);
    }

    // Create new for loop
    builder.setInsertionPoint(forOp);
    auto newForOp = builder.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                               forOp.getUpperBound(),
                                               forOp.getStep(), newInitArgs);

    Block *newBlock = newForOp.getBody();
    Block *oldBlock = forOp.getBody();
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    SmallVector<unsigned> insertedPositions;
    for (auto &tup : indicesToInsert)
      insertedPositions.push_back(std::get<0>(tup));
    llvm::sort(insertedPositions);

    unsigned newIdx = 0, numInserted = 0;
    for (unsigned oldIdx = 0; oldIdx < forOp.getRegionIterArgs().size();
         ++oldIdx) {
      while (numInserted < insertedPositions.size() &&
             insertedPositions[numInserted] == newIdx) {
        newIdx++;
        numInserted++;
      }
      mapping.map(forOp.getRegionIterArgs()[oldIdx],
                  newForOp.getRegionIterArgs()[newIdx]);
      newIdx++;
    }

    builder.setInsertionPointToStart(newBlock);
    for (auto &op : oldBlock->without_terminator())
      builder.clone(op, mapping);

    SmallVector<Value> mappedYieldOperands;
    for (Value v : newYieldOperands)
      mappedYieldOperands.push_back(mapping.lookupOrDefault(v));

    builder.create<scf::YieldOp>(loc, mappedYieldOperands);

    numInserted = 0;
    newIdx = 0;
    for (unsigned oldIdx = 0; oldIdx < forOp.getNumResults(); ++oldIdx) {
      while (numInserted < insertedPositions.size() &&
             insertedPositions[numInserted] == newIdx) {
        newIdx++;
        numInserted++;
      }
      forOp.getResult(oldIdx).replaceAllUsesWith(newForOp.getResult(newIdx));
      newIdx++;
    }

    forOp.erase();
  }

  //===--------------------------------------------------------------------===//
  // CF Block Transformation - Track indices alongside barrier block args
  //===--------------------------------------------------------------------===//

  void transformCfBlocksToTrackIndices(ModuleOp module, OpBuilder &builder) {
    for (auto func : module.getOps<triton::FuncOp>())
      transformCfBlocksInFunction(func, builder);
  }

  void transformCfBlocksInFunction(FuncOp func, OpBuilder &builder) {
    DenseMap<std::pair<Block *, unsigned>, bool> barrierArgNeedsIndex;

    // Identify barrier arguments that need index tracking
    for (Block &block : func.getBody()) {
      if (block.isEntryBlock())
        continue;

      for (Block *pred : block.getPredecessors()) {
        Operation *terminator = pred->getTerminator();

        auto checkBranchOperands = [&](OperandRange operands, Block *dest) {
          if (dest != &block)
            return;

          for (unsigned i = 0; i < operands.size(); ++i) {
            Value operand = operands[i];
            if (!isBarrierType(operand.getType()))
              continue;

            auto indexOp = operand.getDefiningOp<triton::gpu::MemDescIndexOp>();
            if (!indexOp)
              continue;

            Value indexValue = indexOp.getIndex();
            auto key = std::make_pair(&block, i);

            bool hasIndexNext = false;
            if (i + 1 < operands.size()) {
              Value nextOperand = operands[i + 1];
              if (nextOperand == indexValue)
                hasIndexNext = true;
              else if (auto castOp =
                           nextOperand.getDefiningOp<arith::IndexCastOp>())
                if (castOp.getIn() == indexValue)
                  hasIndexNext = true;
            }

            if (!hasIndexNext)
              barrierArgNeedsIndex[key] = true;
          }
        };

        if (auto branchOp = dyn_cast<cf::BranchOp>(terminator)) {
          checkBranchOperands(branchOp.getDestOperands(), branchOp.getDest());
        } else if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(terminator)) {
          checkBranchOperands(condBranchOp.getTrueDestOperands(),
                              condBranchOp.getTrueDest());
          checkBranchOperands(condBranchOp.getFalseDestOperands(),
                              condBranchOp.getFalseDest());
        }
      }
    }

    if (barrierArgNeedsIndex.empty())
      return;

    DenseMap<Block *, SmallVector<unsigned>> blockToBarrierArgs;
    for (auto &[key, needsIndex] : barrierArgNeedsIndex) {
      if (needsIndex)
        blockToBarrierArgs[key.first].push_back(key.second);
    }

    for (auto &[block, barrierArgIndices] : blockToBarrierArgs) {
      llvm::sort(barrierArgIndices, std::greater<unsigned>());

      for (unsigned barrierArgIdx : barrierArgIndices) {
        Type indexType = builder.getI32Type();
        unsigned insertPos = barrierArgIdx + 1;
        block->insertArgument(insertPos, indexType, block->front().getLoc());

        for (Block *pred : block->getPredecessors()) {
          Operation *terminator = pred->getTerminator();

          auto getIndexForBarrier = [&](Value barrierOperand) -> Value {
            if (auto indexOp =
                    barrierOperand
                        .getDefiningOp<triton::gpu::MemDescIndexOp>()) {
              Value idx = indexOp.getIndex();
              builder.setInsertionPoint(terminator);
              if (idx.getType().isInteger(32))
                return idx;
              return builder.create<arith::IndexCastOp>(
                  terminator->getLoc(), builder.getI32Type(), idx);
            }
            builder.setInsertionPoint(terminator);
            return builder.create<arith::ConstantOp>(
                terminator->getLoc(), builder.getI32Type(),
                builder.getI32IntegerAttr(0));
          };

          if (auto branchOp = dyn_cast<cf::BranchOp>(terminator)) {
            if (branchOp.getDest() == block) {
              Value barrierOperand = branchOp.getDestOperands()[barrierArgIdx];
              Value newIndexValue = getIndexForBarrier(barrierOperand);

              SmallVector<Value> newOperands(branchOp.getDestOperands().begin(),
                                             branchOp.getDestOperands().end());
              newOperands.insert(newOperands.begin() + insertPos,
                                 newIndexValue);

              builder.setInsertionPoint(branchOp);
              builder.create<cf::BranchOp>(branchOp.getLoc(), block,
                                           newOperands);
              branchOp.erase();
            }
          } else if (auto condBranchOp =
                         dyn_cast<cf::CondBranchOp>(terminator)) {
            bool updateTrue = (condBranchOp.getTrueDest() == block);
            bool updateFalse = (condBranchOp.getFalseDest() == block);

            SmallVector<Value> newTrueOperands(
                condBranchOp.getTrueDestOperands().begin(),
                condBranchOp.getTrueDestOperands().end());
            SmallVector<Value> newFalseOperands(
                condBranchOp.getFalseDestOperands().begin(),
                condBranchOp.getFalseDestOperands().end());

            if (updateTrue && barrierArgIdx < newTrueOperands.size()) {
              Value newIndexValue =
                  getIndexForBarrier(newTrueOperands[barrierArgIdx]);
              newTrueOperands.insert(newTrueOperands.begin() + insertPos,
                                     newIndexValue);
            }
            if (updateFalse && barrierArgIdx < newFalseOperands.size()) {
              Value newIndexValue =
                  getIndexForBarrier(newFalseOperands[barrierArgIdx]);
              newFalseOperands.insert(newFalseOperands.begin() + insertPos,
                                      newIndexValue);
            }

            builder.setInsertionPoint(condBranchOp);
            builder.create<cf::CondBranchOp>(
                condBranchOp.getLoc(), condBranchOp.getCondition(),
                condBranchOp.getTrueDest(), newTrueOperands,
                condBranchOp.getFalseDest(), newFalseOperands);
            condBranchOp.erase();
          }
        }
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Barrier Info Propagation
  //===--------------------------------------------------------------------===//

  void propagateBarrierInfo(ModuleOp module) {
    valueToBarrierInfo.clear();
    module.walk([&](triton::gpu::LocalAllocOp allocOp) {
      if (!isBarrierType(allocOp.getType()))
        return;
      BarrierInfo info(getMppOpId(allocOp));
      valueToBarrierInfo[allocOp.getResult()] = info;
      visitedValues.clear();
      propagateToUses(allocOp.getResult(), info);
    });
  }

  void propagateToPartitions(triton::gpu::WarpSpecializePartitionsOp op,
                             unsigned argIdx, const BarrierInfo &info) {
    for (Region &r : op->getRegions())
      if (!r.empty() && argIdx < r.front().getNumArguments())
        propagateToUses(r.front().getArgument(argIdx), info);
  }

  void propagateToUses(Value value, const BarrierInfo &info) {
    if (!visitedValues.insert(value).second)
      return;
    valueToBarrierInfo[value] = info;

    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      unsigned idx = use.getOperandNumber();

      if (auto op = dyn_cast<triton::gpu::MemDescIndexOp>(user)) {
        Value idxVal = op.getIndex();
        auto constIdx = getConstantIntValue(idxVal);
        propagateToUses(op.getResult(), constIdx
                                            ? info.withConstantIndex(*constIdx)
                                            : info.withDynamicIndex(idxVal));
      } else if (auto op = dyn_cast<scf::ForOp>(user)) {
        handleScfForOp(op, use, info);
      } else if (auto op = dyn_cast<scf::YieldOp>(user)) {
        handleScfYieldOp(op, use, info);
      } else if (auto op = dyn_cast<cf::BranchOp>(user)) {
        Block *dest = op.getDest();
        if (idx >= dest->getNumArguments())
          continue;
        BarrierInfo newInfo = info;
        if (idx + 1 < dest->getNumArguments() &&
            dest->getArgument(idx + 1).getType().isInteger(32))
          newInfo = info.withDynamicIndex(dest->getArgument(idx + 1))
                        .withAdjacentIndex();
        propagateToUses(dest->getArgument(idx), newInfo);
      } else if (auto op = dyn_cast<cf::CondBranchOp>(user)) {
        if (idx == 0)
          continue;
        unsigned numTrue = op.getTrueDestOperands().size();
        Block *dest = (idx <= numTrue) ? op.getTrueDest() : op.getFalseDest();
        unsigned argIdx = (idx <= numTrue) ? idx - 1 : idx - 1 - numTrue;
        if (argIdx < dest->getNumArguments())
          propagateToUses(dest->getArgument(argIdx), info);
      } else if (auto op = dyn_cast<triton::gpu::WarpSpecializeOp>(user)) {
        if (op->getNumRegions() > 1 && !op->getRegion(1).empty())
          for (Operation &inner : op->getRegion(1).front())
            if (auto p =
                    dyn_cast<triton::gpu::WarpSpecializePartitionsOp>(&inner))
              propagateToPartitions(p, idx, info);
      } else if (auto op =
                     dyn_cast<triton::gpu::WarpSpecializePartitionsOp>(user)) {
        propagateToPartitions(op, idx, info);
      }
    }
  }

  void handleScfForOp(scf::ForOp forOp, OpOperand &use,
                      const BarrierInfo &info) {
    unsigned opIdx = use.getOperandNumber();
    if (opIdx < 3)
      return;

    unsigned iterIdx = opIdx - 3;
    Block &block = forOp.getRegion().front();
    if (iterIdx + 1 >= block.getNumArguments())
      return;

    BarrierInfo newInfo = info;
    if (info.hasIndex && info.constantIndex < 0 && info.dynamicIndex &&
        iterIdx + 1 < forOp.getInitArgs().size() &&
        iterIdx + 2 < block.getNumArguments()) {
      Value nextInit = forOp.getInitArgs()[iterIdx + 1];
      if (nextInit == info.dynamicIndex ||
          (nextInit.getDefiningOp<arith::IndexCastOp>() &&
           nextInit.getDefiningOp<arith::IndexCastOp>().getIn() ==
               info.dynamicIndex))
        newInfo = newInfo.withDynamicIndex(block.getArgument(iterIdx + 2));
    }

    propagateToUses(block.getArgument(iterIdx + 1), newInfo);
    if (iterIdx < forOp.getNumResults())
      propagateToUses(forOp.getResult(iterIdx), info);
  }

  void handleScfYieldOp(scf::YieldOp yieldOp, OpOperand &use,
                        const BarrierInfo &info) {
    unsigned opIdx = use.getOperandNumber();
    Operation *parentOp = yieldOp->getParentOp();

    // Find yield position once
    std::optional<unsigned> yieldPos;
    if (info.hasIndex && info.constantIndex < 0 && info.dynamicIndex)
      for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i)
        if (yieldOp.getOperand(i) == info.dynamicIndex) {
          yieldPos = i;
          break;
        }

    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      BarrierInfo updatedInfo =
          yieldPos ? info.withDynamicIndex(info.dynamicIndex, *yieldPos)
                         .withAdjacentIndex()
                   : info;

      if (opIdx < forOp.getNumResults())
        propagateToUses(forOp.getResult(opIdx), updatedInfo);

      Block &block = forOp.getRegion().front();
      if (opIdx + 1 < block.getNumArguments()) {
        Value newDynIdx = (yieldPos && *yieldPos + 1 < block.getNumArguments())
                              ? block.getArgument(*yieldPos + 1)
                              : nullptr;
        BarrierInfo blockInfo =
            (yieldPos && newDynIdx)
                ? info.withDynamicIndex(newDynIdx, *yieldPos)
                      .withAdjacentIndex()
                : info;
        propagateToUses(block.getArgument(opIdx + 1), blockInfo);
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
      if (opIdx < ifOp.getNumResults())
        propagateToUses(ifOp.getResult(opIdx), info);
    }
  }

  //===--------------------------------------------------------------------===//
  // Barrier Info Retrieval
  //===--------------------------------------------------------------------===//

  std::optional<BarrierInfo> getBarrierInfo(Value barrier, int depth = 0) {
    if (depth > 10)
      return std::nullopt;
    if (auto it = valueToBarrierInfo.find(barrier);
        it != valueToBarrierInfo.end())
      return it->second;

    std::optional<BarrierInfo> result;

    if (auto indexOp = barrier.getDefiningOp<triton::gpu::MemDescIndexOp>()) {
      if (auto src = getBarrierInfo(indexOp.getSrc(), depth + 1)) {
        auto idx = indexOp.getIndex();
        auto constIdx = getConstantIntValue(idx);
        result = constIdx ? src->withConstantIndex(*constIdx)
                          : src->withDynamicIndex(idx);
      }
    } else if (auto allocOp =
                   barrier.getDefiningOp<triton::gpu::LocalAllocOp>()) {
      if (isBarrierType(allocOp.getResult().getType()))
        result = BarrierInfo(getMppOpId(allocOp));
    } else if (auto blockArg = dyn_cast<BlockArgument>(barrier)) {
      result = getBarrierInfoForBlockArg(blockArg, depth);
    } else if (auto forOp = barrier.getDefiningOp<scf::ForOp>()) {
      unsigned idx =
          llvm::find(forOp.getResults(), barrier) - forOp.getResults().begin();
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (idx < yieldOp.getNumOperands())
        result = getBarrierInfo(yieldOp.getOperand(idx), depth + 1);
    }

    if (result)
      valueToBarrierInfo[barrier] = *result;
    return result;
  }

  std::optional<BarrierInfo> getBarrierInfoForBlockArg(BlockArgument blockArg,
                                                       int depth) {
    Block *block = blockArg.getOwner();
    unsigned argIdx = blockArg.getArgNumber();
    Operation *parentOp = block->getParentOp();

    // Check CF predecessors
    for (Block *pred : block->getPredecessors()) {
      Operation *term = pred->getTerminator();
      Value incoming = nullptr;
      if (auto br = dyn_cast<cf::BranchOp>(term)) {
        if (br.getDest() == block && argIdx < br.getDestOperands().size())
          incoming = br.getDestOperands()[argIdx];
      } else if (auto cbr = dyn_cast<cf::CondBranchOp>(term)) {
        if (cbr.getTrueDest() == block &&
            argIdx < cbr.getTrueDestOperands().size())
          incoming = cbr.getTrueDestOperands()[argIdx];
        else if (cbr.getFalseDest() == block &&
                 argIdx < cbr.getFalseDestOperands().size())
          incoming = cbr.getFalseDestOperands()[argIdx];
      }
      if (incoming)
        if (auto info = getBarrierInfo(incoming, depth + 1))
          return info;
    }

    // Check scf.for init args
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp))
      if (argIdx > 0 && argIdx - 1 < forOp.getInitArgs().size())
        return getBarrierInfo(forOp.getInitArgs()[argIdx - 1], depth + 1);

    // Check warp specialize partitions
    if (auto partitionsOp =
            dyn_cast<triton::gpu::WarpSpecializePartitionsOp>(parentOp))
      if (argIdx < partitionsOp.getParentOp().getNumOperands())
        return getBarrierInfo(partitionsOp.getParentOp().getOperand(argIdx),
                              depth + 1);

    return std::nullopt;
  }

  //===--------------------------------------------------------------------===//
  // Dominance and Index Extraction
  //===--------------------------------------------------------------------===//

  bool valueDominatesOp(Value value, Operation *op) {
    DominanceInfo dom(op->getParentOfType<FuncOp>());
    if (auto defOp = value.getDefiningOp())
      return dom.properlyDominates(defOp, op);
    if (auto arg = dyn_cast<BlockArgument>(value))
      return dom.dominates(arg.getOwner(), op->getBlock());
    return false;
  }

  Value findIndexValue(Value barrierValue, Operation *op, OpBuilder &builder) {
    auto toI32 = [&](Value v) {
      if (v.getType().isInteger(32))
        return v;
      builder.setInsertionPoint(op);
      return builder
          .create<arith::IndexCastOp>(op->getLoc(), builder.getI32Type(), v)
          .getResult();
    };

    // Direct memdesc_index
    if (auto indexOp =
            barrierValue.getDefiningOp<triton::gpu::MemDescIndexOp>()) {
      Value idx = indexOp.getIndex();
      if (valueDominatesOp(idx, op))
        return toI32(idx);
    }

    // Block arg from scf.for - check yield
    if (auto blockArg = dyn_cast<BlockArgument>(barrierValue)) {
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      unsigned argIdx = blockArg.getArgNumber();
      if (forOp && argIdx > 0) {
        auto yieldOp = cast<scf::YieldOp>(blockArg.getOwner()->getTerminator());
        unsigned iterIdx = argIdx - 1;
        if (iterIdx < yieldOp.getNumOperands())
          if (auto indexOp =
                  yieldOp.getOperand(iterIdx)
                      .getDefiningOp<triton::gpu::MemDescIndexOp>()) {
            Value idx = indexOp.getIndex();
            if (valueDominatesOp(idx, op))
              return toI32(idx);
          }
      }
    }
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Process Circular Store Pairs
  //===--------------------------------------------------------------------===//

  static bool isBarrierOp(Operation *op) {
    return isa<
        triton::nvidia_gpu::WaitBarrierOp, triton::nvidia_gpu::ArriveBarrierOp,
        triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
        triton::nvidia_gpu::TCGen5MMAOp, triton::nvidia_gpu::TCGen5CommitOp>(
        op);
  }

  struct StoreWithBarrierInfo {
    CircularStoreOp startStore;
    Value barrierValue;
    BarrierInfo info;
  };

  void walkBlockForStores(Block &block, SmallVectorImpl<CircularStoreOp> &stack,
                          SmallVectorImpl<StoreWithBarrierInfo> &results,
                          DenseMap<int, CircularStoreOp> &endMap) {
    for (Operation &op : block) {
      if (auto store = dyn_cast<CircularStoreOp>(&op)) {
        if (store.getIsStart())
          stack.push_back(store);
        else
          endMap[store.getScopeId()] = store;
        continue;
      }

      if (isBarrierOp(&op)) {
        int numBarriers = isa<triton::nvidia_gpu::TCGen5MMAOp>(&op)
                              ? cast<triton::nvidia_gpu::TCGen5MMAOp>(&op)
                                    .getBarriers()
                                    .size()
                              : 1;
        SmallVector<std::pair<CircularStoreOp, int>> pending;
        while (!stack.empty())
          pending.push_back(
              {stack.pop_back_val(),
               std::max(0, numBarriers - 1 - (int)pending.size())});

        for (auto &[startStore, barrierIdx] : llvm::reverse(pending)) {
          Value bv = getBarrierOperand(&op, barrierIdx);
          if (!bv)
            bv = getBarrierOperand(&op, -1);
          if (!bv)
            continue;

          auto infoOpt = getBarrierInfo(bv);
          BarrierInfo info =
              (infoOpt && infoOpt->allocOpId >= 0)
                  ? *infoOpt
                  : BarrierInfo(bv.getDefiningOp()
                                    ? getMppOpId(bv.getDefiningOp())
                                    : getMppOpId(&op));
          results.push_back({startStore, bv, info});
        }
        continue;
      }

      for (Region &r : op.getRegions())
        for (Block &b : r)
          walkBlockForStores(b, stack, results, endMap);
    }
  }

  Value computeIndexValue(const BarrierInfo &info, Value barrierValue,
                          CircularStoreOp endStore, ReadCounterOp readCounterOp,
                          OpBuilder &builder) {
    Location loc = readCounterOp.getLoc();
    Type counterType = readCounterOp.getCounter().getType();

    auto toCounterType = [&](Value v) -> Value {
      return v.getType() == counterType
                 ? v
                 : builder.create<arith::IndexCastOp>(loc, counterType, v)
                       .getResult();
    };

    // Try: CF block arg with adjacent tracked index
    if (auto blockArg = dyn_cast<BlockArgument>(barrierValue)) {
      Block *block = blockArg.getOwner();
      unsigned argIdx = blockArg.getArgNumber();
      if (!isa<scf::ForOp>(block->getParentOp()) &&
          argIdx + 1 < block->getNumArguments() &&
          block->getArgument(argIdx + 1).getType().isInteger(32))
        return block->getArgument(argIdx + 1);
    }

    // Try: Direct index from barrier value
    if (Value idx = findIndexValue(barrierValue, endStore, builder))
      return idx;

    // Try: Constant index from info
    if (info.hasIndex && info.constantIndex >= 0)
      return arith::ConstantIntOp::create(builder, loc, counterType,
                                          info.constantIndex);

    // Try: Dynamic index from info
    if (info.dynamicIndex && valueDominatesOp(info.dynamicIndex, endStore))
      return toCounterType(info.dynamicIndex);

    // Try: Loop result from yield position
    if (info.dynamicIndexYieldPosition >= 0 && info.hasAdjacentIndex)
      if (auto forOp = barrierValue.getDefiningOp<scf::ForOp>())
        if ((unsigned)info.dynamicIndexYieldPosition < forOp.getNumResults()) {
          Value result = forOp.getResult(info.dynamicIndexYieldPosition);
          if (valueDominatesOp(result, endStore))
            return toCounterType(result);
        }

    // Fallback: zero
    return arith::ConstantIntOp::create(builder, loc, counterType, 0);
  }

  LogicalResult processFunction(FuncOp func, OpBuilder &builder) {
    SmallVector<CircularStoreOp, 8> stack;
    SmallVector<StoreWithBarrierInfo, 8> stores;
    DenseMap<int, CircularStoreOp> endMap;

    for (Block &block : func.getBody())
      walkBlockForStores(block, stack, stores, endMap);

    auto replaceCounter = [](CircularStoreOp store, Value newVal) {
      if (auto rcOp = store.getCounter().getDefiningOp<ReadCounterOp>()) {
        store.getCounterMutable().assign(newVal);
        if (rcOp->use_empty())
          rcOp->erase();
      }
    };

    for (auto &si : stores) {
      auto endStore = endMap.lookup(si.startStore.getScopeId());
      if (!endStore)
        continue;

      if (auto rcOp =
              si.startStore.getCounter().getDefiningOp<ReadCounterOp>()) {
        builder.setInsertionPoint(rcOp);
        replaceCounter(si.startStore,
                       arith::ConstantIntOp::create(builder, rcOp.getLoc(),
                                                    rcOp.getCounter().getType(),
                                                    si.info.allocOpId));
      }

      if (auto rcOp = endStore.getCounter().getDefiningOp<ReadCounterOp>()) {
        builder.setInsertionPoint(rcOp);
        replaceCounter(endStore, computeIndexValue(si.info, si.barrierValue,
                                                   endStore, rcOp, builder));
      }
    }
    return success();
  }
};

} // namespace mlir::triton::proton::gpu
