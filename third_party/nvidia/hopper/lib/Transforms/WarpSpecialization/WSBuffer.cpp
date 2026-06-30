#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

static mlir::Location accumCntLoc(mlir::Location loc) {
  return mlir::NameLoc::get(
      mlir::StringAttr::get(loc.getContext(), "accum_cnt"));
}
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-buffer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static bool
enclosingAChannel(Operation *ctrlOp,
                  const DenseSet<Operation *> &regionsWithChannels) {
  for (auto *op : regionsWithChannels) {
    if (ctrlOp == op)
      return true;
    if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp))
      if (enclosing(forOp, op))
        return true;
    if (auto ifOp = dyn_cast<scf::IfOp>(ctrlOp))
      if (enclosing(ifOp, op))
        return true;
  }
  return false;
}

unsigned getLoopDepth(Operation *op) {
  unsigned depth = 0;
  auto pOp = op->getParentOfType<scf::ForOp>();
  while (pOp) {
    ++depth;
    pOp = pOp->getParentOfType<scf::ForOp>();
  }
  return depth;
}

// Update preOrderOps with a list of region Ops nested under ctrlOp that will
// need accumCnt. The list is in pre-order.
void getAccumCntsPreOrder(Operation *ctrlOp,
                          const DenseSet<Operation *> &regionsWithChannels,
                          SmallVector<Operation *> &preOrderOps) {
  ctrlOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    // This will walk ctrlOp itself.
    if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
      for (auto *op : regionsWithChannels) {
        if (subOp == op) {
          preOrderOps.push_back(subOp);
        }
      }
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(subOp)) {
      for (auto *op : regionsWithChannels) {
        if (subOp == op) {
          preOrderOps.push_back(subOp);
        }
      }
    }
    // A persistent scf.while can directly carry a channel (e.g. the
    // accumulator/epilogue channel that lives in its after region). Count it
    // so it gets its own accumCnt, advanced once per persistent iteration.
    if (auto whileOp = dyn_cast<scf::WhileOp>(subOp)) {
      for (auto *op : regionsWithChannels) {
        if (subOp == op) {
          preOrderOps.push_back(subOp);
        }
      }
    }
  });
}

// Go through all the regions in opList and correctly add accumCnt. taskTopOps
// will be updated if it is replaced in the process.
void updateAccumLoopCount(SmallVector<Operation *> &opList,
                          SmallVector<Operation *> &taskTopOps,
                          DenseSet<Operation *> &regionsWithChannels,
                          ReuseConfig *config);

// prevAccum is the accumCnt prior to the forOp. This function goes through
// the forOp and insert accumCnt when necessary.
scf::ForOp createNewLoopWrapper(scf::ForOp origForOp,
                                SmallVector<Operation *> &taskTopOps,
                                DenseSet<Operation *> &regionsWithChannels,
                                ReuseConfig *config);

// Thread accumCnt loop-carried values through a persistent scf.while loop so
// that buffer indices/phases stay in sync across persistent iterations.
scf::WhileOp createNewWhileWrapper(scf::WhileOp origWhileOp,
                                   SmallVector<Operation *> &taskTopOps,
                                   DenseSet<Operation *> &regionsWithChannels,
                                   ReuseConfig *config);

// If there is a channel directly inside IfOp, update endAccum and endAccumElse.
static void generateYieldCntsForIfOp(scf::IfOp ifOp, Value &endAccum,
                                     Value &endAccumElse,
                                     DenseSet<Operation *> &regionsWithChannels,
                                     ReuseConfig *config,
                                     OpBuilderWithAsyncTaskIds &ifBuilder,
                                     OpBuilderWithAsyncTaskIds &elseBuilder) {
  auto parentForOp = ifOp->getParentOfType<scf::ForOp>();
  auto *op = ifOp.getOperation();
  auto loc = ifOp.getLoc();
  if (parentForOp) {
    unsigned parentArgSize = parentForOp.getBody()->getArguments().size();
    // Get corresponding argument of accumCnt for "op" in parentForOp.
    unsigned accumArgId = getAccumArgIdx(parentForOp, ifOp.getOperation(),
                                         regionsWithChannels, config, -1);
    unsigned parentTCnts =
        getAccumCnts(parentForOp.getOperation(), regionsWithChannels, config);
    LDBG("rewrite ifOp: ifOp itself parentArg " << parentArgSize << " "
                                                << accumArgId);
    // All the accumCnts are at the end of argument list. When accumArgId
    // is parentTCnts - 1, the corresponding accumCnt will be the last
    // argument.
    Value arg = parentForOp.getBody()->getArgument(parentArgSize - parentTCnts +
                                                   accumArgId);
    // Either parent[accumCnt] + 1 or parent[accumCnt].
    Value one =
        ifBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
    endAccum = ifBuilder.createWithAsyncTaskIds<arith::AddIOp>(accumCntLoc(loc),
                                                               arg, one);
    endAccumElse = arg;
  } else {
    endAccum =
        ifBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
    endAccumElse =
        elseBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
  }
  LLVM_DEBUG({
    LDBG("Update yieldOperands ");
    endAccum.dump();
    endAccumElse.dump();
  });
}

// regionOp: inside thenBlock of ifOp.
// There can be a list of accumCnts associated with the regionOp, for which we
// need arguments on the ifOp.
static void generateYieldCntsForThenBlock(
    scf::IfOp ifOp, Operation *regionOp, SmallVector<Value> &thenYields,
    SmallVector<Value> &elseYields, DenseSet<Operation *> &regionsWithChannels,
    ReuseConfig *config, OpBuilderWithAsyncTaskIds &ifBuilder,
    OpBuilderWithAsyncTaskIds &elseBuilder) {
  SmallVector<Operation *> preOrderOps;
  getAccumCntsPreOrder(regionOp, regionsWithChannels, preOrderOps);
  if (preOrderOps.empty())
    return;
  auto numRes = regionOp->getNumResults();
  unsigned tCnts = preOrderOps.size();
  unsigned tCntsTotal = getAccumCnts(regionOp, regionsWithChannels, config);
  LDBG("rewrite ifOp: thenBlock " << tCnts << " accumCnts");

  unsigned accumArgId, parentArgSize, parentTCnts;
  auto parentForOp = ifOp->getParentOfType<scf::ForOp>();
  if (parentForOp) {
    parentArgSize = parentForOp.getBody()->getArguments().size();
    // Find accumArgId for preOrderOps[0] in parentForOp.
    accumArgId = getAccumArgIdx(parentForOp, preOrderOps[0],
                                regionsWithChannels, config, -1);
    parentTCnts =
        getAccumCnts(parentForOp.getOperation(), regionsWithChannels, config);
  }
  auto loc = ifOp.getLoc();

  // Set up value for thenYield and elseYield for accumCnts nested under "op".
  // Each accumCnt nested under "op", it will have a corresponding argument in
  // this "IfOp". If "op" has tCnts, this "IfOp" will have the same number of
  // corresponding accumCnts, in the same order.
  for (unsigned i = 0; i < tCnts; ++i) {
    // Handle each accumCnt for "op".
    Value endAccum = regionOp->getResult(numRes - tCntsTotal + i);
    thenYields.push_back(endAccum);

    // Find the corresponding accumArgId from parentForOp.
    Value elseVal;
    if (parentForOp) {
      elseVal = parentForOp.getBody()->getArgument(parentArgSize - parentTCnts +
                                                   accumArgId + i);
      LDBG("rewrite ifOp: elseYield parentArg " << parentArgSize << " "
                                                << accumArgId << " " << i);
    } else
      elseVal =
          elseBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
    elseYields.push_back(elseVal);
    LLVM_DEBUG({
      LDBG("Update yieldOperands ");
      endAccum.dump();
      elseVal.dump();
    });
  }
}

// Determine the per-iteration accumCnt increment for a ForOp.  When the loop
// body contains a SubtiledRegionOp, each iteration processes numTiles tiles,
// so the increment must be numTiles instead of 1.
static int64_t getAccumCntIncrement(scf::ForOp forOp) {
  int64_t increment = 1;
  forOp.walk([&](triton::nvidia_gpu::SubtiledRegionOp subOp) {
    unsigned numTiles = subOp.getNumTiles();
    if (numTiles > static_cast<unsigned>(increment))
      increment = numTiles;
  });
  return increment;
}

// Increment by the appropriate amount for unique channels.
static Value generateYieldCntsForForOp(scf::ForOp forOp, unsigned accumArgId) {
  Operation *yieldOp = forOp.getBody()->getTerminator();
  Value arg = forOp.getBody()->getArgument(accumArgId);
  OpBuilderWithAsyncTaskIds builder(forOp->getContext());
  builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(forOp));
  builder.setInsertionPoint(yieldOp);
  auto loc = forOp.getLoc();
  int64_t increment = getAccumCntIncrement(forOp);
  Value incVal =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, increment, 64);
  Value endAccum = builder.createWithAsyncTaskIds<arith::AddIOp>(
      accumCntLoc(loc), arg, incVal);
  return endAccum;
}

static bool isRegionOp(Operation *op) {
  return isa<scf::ForOp>(op) || isa<scf::IfOp>(op);
}

// op is in chList, chList is the list of operations under a ctrlOp enclosing
// channels for a given reuse group. Elements in chList can be region op or
// non-region op.
// Returns AccumCnt before or after op for a given reuse group.
Value getAccumForReuseGroup(Operation *op, SmallVector<Operation *> &chList,
                            DenseSet<Operation *> &regionsWithChannels,
                            ReuseConfig *config, int reuseGroupIdx,
                            bool before) {
  // If op is a region op, we can get its result at the matching ArgIdx.
  // Otherwise, we need to find the last region op prior to op and accumulate
  // from there.
  int opIdx = -1, idx = 0, lastRegionIdx = -1;
  for (auto *ch : chList) {
    if (isRegionOp(ch) && (!before || ch != op))
      // If checking before the op, we should exclude op.
      lastRegionIdx = idx;
    if (op == ch) {
      opIdx = idx;
      break;
    }
    ++idx;
  }
  assert(opIdx >= 0);
  if (before && lastRegionIdx >= 0 && lastRegionIdx == opIdx - 1) {
    auto *lastOp = chList[lastRegionIdx];
    auto numRes = lastOp->getNumResults();
    unsigned tCnts = getAccumCnts(lastOp, regionsWithChannels, config);
    auto reuseArgIdx =
        getReuseAccumArgIdx(lastOp, regionsWithChannels, config, reuseGroupIdx);
    return lastOp->getResult(numRes - tCnts + reuseArgIdx);
  }
  if (!before && lastRegionIdx == opIdx) {
    auto reuseArgIdx =
        getReuseAccumArgIdx(op, regionsWithChannels, config, reuseGroupIdx);
    auto numRes = op->getNumResults();
    unsigned tCnts = getAccumCnts(op, regionsWithChannels, config);
    return op->getResult(numRes - tCnts + reuseArgIdx);
  }
  Operation *ctrlOp = op->getParentOp();
  OpBuilderWithAsyncTaskIds builder(ctrlOp->getContext());
  auto taskIds = getNestedAsyncTaskIds(ctrlOp);
  // HACK
#if 0
  {
    auto parentForOp = ctrlOp->getParentOfType<scf::ForOp>();
    if (!parentForOp) {
      taskIds.pop_back();
      LDBG("getAccumForReuseGroup hack to remove last taskId");
    }
  }
#endif
  builder.setAsynTaskIdsFromArray(taskIds);
  builder.setInsertionPoint(op);
  if (lastRegionIdx >= 0) {
    auto *lastRegionOp = chList[lastRegionIdx];
    // Get the argment idx for accumCnt associated with lastRegionOp for the
    // specific reuse group.
    auto reuseArgIdx = getReuseAccumArgIdx(lastRegionOp, regionsWithChannels,
                                           config, reuseGroupIdx);
    auto numRes = lastRegionOp->getNumResults();
    unsigned tCnts = getAccumCnts(lastRegionOp, regionsWithChannels, config);
    Value res = lastRegionOp->getResult(numRes - tCnts + reuseArgIdx);
    auto loc = op->getLoc();
    // From the last region op, accumulate till before or after "op".
    Value lit = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        loc, before ? opIdx - lastRegionIdx - 1 : opIdx - lastRegionIdx, 64);
    Value endAccum = builder.createWithAsyncTaskIds<arith::AddIOp>(
        accumCntLoc(loc), res, lit);
    return endAccum;
  }
  // Here lastRegionIdx < 0: we need to start with the accumCnt value at the
  // start of ctrlOp.
  if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp)) {
    auto argIdx = getAccumArgIdx(forOp, nullptr, regionsWithChannels, config,
                                 reuseGroupIdx);
    auto numArgs = forOp.getBody()->getArguments().size();
    auto tCnts =
        getAccumCnts(forOp.getOperation(), regionsWithChannels, config);
    Value arg = forOp.getBody()->getArgument(numArgs - tCnts + argIdx);
    Value lit = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        forOp.getLoc(), before ? opIdx : opIdx + 1, 64);
    Value endAccum = builder.createWithAsyncTaskIds<arith::AddIOp>(
        accumCntLoc(forOp.getLoc()), arg, lit);
    return endAccum;
  }
  // A persistent scf.while carries the reuse-group accumCnt in its after
  // region arguments, mirroring the scf.for case above.
  if (auto whileOp = dyn_cast<scf::WhileOp>(ctrlOp)) {
    auto argIdx = getAccumArgIdx(whileOp, nullptr, regionsWithChannels, config,
                                 reuseGroupIdx);
    auto numArgs = whileOp.getAfterBody()->getArguments().size();
    auto tCnts =
        getAccumCnts(whileOp.getOperation(), regionsWithChannels, config);
    Value arg = whileOp.getAfterBody()->getArgument(numArgs - tCnts + argIdx);
    Value lit = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        whileOp.getLoc(), before ? opIdx : opIdx + 1, 64);
    Value endAccum = builder.createWithAsyncTaskIds<arith::AddIOp>(
        accumCntLoc(whileOp.getLoc()), arg, lit);
    return endAccum;
  }
  if (isa<scf::IfOp>(ctrlOp)) {
    // Find parentChList in parent scope and get value for the op
    // right before ctrlOp in parentChList.
    SmallVector<Operation *> parentChList;
    getReuseChannels(config->getGroup(reuseGroupIdx), ctrlOp->getParentOp(),
                     parentChList);
    Value startOfIf = getAccumForReuseGroup(
        ctrlOp, parentChList, regionsWithChannels, config, reuseGroupIdx, true);
    Value lit = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        ctrlOp->getLoc(), before ? opIdx : opIdx + 1, 64);
    Value endAccum = builder.createWithAsyncTaskIds<arith::AddIOp>(
        accumCntLoc(ctrlOp->getLoc()), startOfIf, lit);
    return endAccum;
  }
  assert(isa<tt::FuncOp>(ctrlOp));
  return builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
      op->getLoc(), before ? opIdx : opIdx + 1, 64);
}

scf::IfOp rewriteIfOp(scf::IfOp ifOp, SmallVector<Operation *> &taskTopOps,
                      DenseSet<Operation *> &regionsWithChannels,
                      ReuseConfig *config) {
  OpBuilderWithAsyncTaskIds ifBuilder(ifOp.getContext());
  ifBuilder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(ifOp));
  ifBuilder.setInsertionPoint(ifOp);

  // Calculate how many accumCnts we will need for this IfOp.
  unsigned numAccumCnts =
      getAccumCnts(ifOp.getOperation(), regionsWithChannels, config);
  // Add one i64 result value for each needed accumCnt.
  SmallVector<Type> newResultTypes(ifOp->getResultTypes());
  for (unsigned i = 0; i < numAccumCnts; ++i)
    newResultTypes.push_back(ifBuilder.getI64Type());

  LDBG("rewrite ifOp: add " << numAccumCnts << " accumCnts");
  assert(numAccumCnts > 0);
  // Create else block since we need to generate accumulated count for then and
  // else.
  auto newIfOp = ifBuilder.createWithAsyncTaskIds<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, ifOp.getCondition(), true, true);

  // Move the existing blocks to the new if.
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());

  ifBuilder.setInsertionPointToEnd(newIfOp.thenBlock());
  SmallVector<Operation *> opList;
  for (Operation &op : newIfOp.thenBlock()->getOperations()) {
    if (auto tOp = dyn_cast<scf::ForOp>(&op))
      opList.push_back(&op);
    if (auto tOp = dyn_cast<scf::IfOp>(&op))
      opList.push_back(&op);
  }

  // Create new Yield and erase original Yield.
  auto loc = ifOp.getLoc();
  auto updateYield = [&](scf::YieldOp yield, SmallVector<Value> &operands) {
    ifBuilder.setInsertionPoint(yield);
    ifBuilder.createWithAsyncTaskIds<scf::YieldOp>(loc, operands);
    yield.erase();
  };

  // Update regionsWithChannels withe newIfOp.
  auto tmpIter3 = std::find(regionsWithChannels.begin(),
                            regionsWithChannels.end(), ifOp.getOperation());
  if (tmpIter3 != regionsWithChannels.end()) {
    LDBG("rewrite ifOp: update regionsWithChannels "
         << ifOp.getOperation() << " --> " << newIfOp.getOperation());
    *tmpIter3 = newIfOp.getOperation();
  }

  // Go through region ops in the thenBlock. updateAccumLoopCount takes current
  // accumCnt value and returns the value at the end of the thenBlock.
  updateAccumLoopCount(opList, taskTopOps, regionsWithChannels, config);

  SmallVector<Value> ifYieldOperands = newIfOp.thenYield().getOperands();

  if (ifOp.elseBlock()) {
    ifBuilder.setInsertionPointToEnd(newIfOp.elseBlock());
    newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());

    SmallVector<Operation *> opListElse;
    for (Operation &op : newIfOp.elseBlock()->getOperations()) {
      if (auto tOp = dyn_cast<scf::ForOp>(&op))
        opListElse.push_back(&op);
      if (auto tOp = dyn_cast<scf::IfOp>(&op))
        opListElse.push_back(&op);
    }
    // We need to differentiate channels in then region vs. in else region.
    // For now, only handle the case where channels are in then region.
    for (auto *op : opListElse)
      assert(!enclosingAChannel(op, regionsWithChannels));
  } else {
    // Create an empty yield
    auto b = newIfOp.getElseBodyBuilder();
    auto yieldOp = scf::YieldOp::create(b, ifOp.getLoc());
  }

  SmallVector<Value> elseYieldOperands = newIfOp.elseYield().getOperands();
  OpBuilderWithAsyncTaskIds elseBuilder(ifOp.getContext());
  elseBuilder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(ifOp));
  elseBuilder.setInsertionPoint(newIfOp.elseYield());
  ifBuilder.setInsertionPoint(newIfOp.thenYield());

  auto parentForOp = newIfOp->getParentOfType<scf::ForOp>();
  SmallVector<Operation *> preOrderOpsOfParent;
  if (parentForOp) {
    getAccumCntsPreOrder(parentForOp.getOperation(), regionsWithChannels,
                         preOrderOpsOfParent);
  }

  // For this IfOp, add accumCnts in preorder, starting with the IfOp itself
  // if it contains a channel. It then goes through the body of thenBlock, add
  // accumCnts for each region op of the thenBlock.
  // Check to see if newIfOp has channels directly in.
  bool hasDirectChannel = false;
  for (auto *op : regionsWithChannels) {
    if (newIfOp.getOperation() == op) {
      hasDirectChannel = true;
      break;
    }
  }
  // We need to handle yield values for accumCnts of unique channels and reuse
  // channels.
  if (hasDirectChannel) {
    // Set up value for thenYield and elseYield for accumCnt associated with
    // "newIfOp".
    Value endAccum, endAccumElse;
    generateYieldCntsForIfOp(newIfOp, endAccum, endAccumElse,
                             regionsWithChannels, config, ifBuilder,
                             elseBuilder);
    ifYieldOperands.push_back(endAccum);
    elseYieldOperands.push_back(endAccumElse);
  }

  // Go through region ops in thenBlock.
  for (auto *op : opList) {
    if (!enclosingAChannel(op, regionsWithChannels))
      continue;
    SmallVector<Value> thenYields;
    SmallVector<Value> elseYields;
    generateYieldCntsForThenBlock(newIfOp, op, thenYields, elseYields,
                                  regionsWithChannels, config, ifBuilder,
                                  elseBuilder);
    for (auto V : thenYields)
      ifYieldOperands.push_back(V);
    for (auto V : elseYields)
      elseYieldOperands.push_back(V);
  }
  // Handle reuse groups.
  for (unsigned idx = 0; idx < config->getGroupSize(); ++idx) {
    // Find channels of reuse group that are inside ifOp. If the channel is
    // directly in ifOp, add the channel's DstOp, otherwise add the region Op
    // that is directly in ifOp.
    SmallVector<Operation *> chList;
    getReuseChannels(config->getGroup(idx), newIfOp.getOperation(), chList);
    if (chList.empty())
      continue;
    Operation *lastOp = chList.back();
    Value prevAccum;
    {
      SmallVector<Operation *> parentChList;
      Operation *parentOp = newIfOp->getParentOp();
      // Get a list of ops directly under parentOp that contain channels in the
      // reuse group.
      getReuseChannels(config->getGroup(idx), parentOp, parentChList);
      prevAccum = getAccumForReuseGroup(newIfOp.getOperation(), parentChList,
                                        regionsWithChannels, config, idx, true);
    }
    // Find accumValue after lastOp.
    auto thenYield = getAccumForReuseGroup(lastOp, chList, regionsWithChannels,
                                           config, idx, false);
    ifYieldOperands.push_back(thenYield);
    elseYieldOperands.push_back(prevAccum);
  }

  // Update Yields.
  updateYield(newIfOp.thenYield(), ifYieldOperands);
  updateYield(newIfOp.elseYield(), elseYieldOperands);

  int resultIdx = 0;
  // Replace old if with the new one.
  for (auto result : ifOp.getResults()) {
    result.replaceAllUsesWith(newIfOp->getResult(resultIdx++));
  }
  ifOp.erase();
  return newIfOp;
}

// Handle the forOp given initial accumCnts.
scf::ForOp createNewLoop(scf::ForOp forOp, scf::ForOp &parentForOp,
                         SmallVector<Value> &initialAccums) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();

  OpBuilderWithAsyncTaskIds builder(forOp.getContext());
  builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(forOp));
  builder.setInsertionPoint(forOp);

  unsigned numAccumCnts = initialAccums.size();

  // Step 1: Append accumCnts as forOp arguments.
  for (unsigned i = 0; i < numAccumCnts; i++)
    body->insertArgument(body->getNumArguments(), builder.getI64Type(),
                         accumCntLoc(loc));

  // Step 2: Add accumCnts to yieldOp.
  auto yieldOp = llvm::cast<scf::YieldOp>(body->getTerminator());
  builder.setInsertionPoint(yieldOp);
  unsigned tSize = body->getNumArguments();
  // Pass argument value as yield. This will be fixed in the caller.
  for (unsigned i = 0; i < numAccumCnts; i++)
    yieldOp->insertOperands(yieldOp.getNumOperands(),
                            {body->getArgument(tSize - numAccumCnts + i)});

  // Step 3: Create loop arguments for the new ForOp.
  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getInitArgs())
    newLoopArgs.push_back(operand);
  builder.setInsertionPoint(forOp);
  for (unsigned i = 0; i < numAccumCnts; i++) {
    newLoopArgs.append({initialAccums[i]});
  }

  // Step 4: Create newForOp and take the region of the original forOp.
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      newLoopArgs);
  newForOp.getRegion().takeBody(forOp.getRegion());

  // Set NameLoc("accum_cnt") on the accumCnt block arguments so they are
  // distinguishable from user-defined iter_args under
  // --mlir-use-nameloc-as-prefix.
  unsigned bodyArgSize = newForOp.getBody()->getNumArguments();
  for (unsigned i = 0; i < numAccumCnts; i++) {
    auto arg = newForOp.getBody()->getArgument(bodyArgSize - numAccumCnts + i);
    arg.setLoc(accumCntLoc(loc));
  }

  // Step 5: Copy over the existing attributes.
  // This is needed to preserve tt.warp_specialize.
  for (auto attr : forOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }

  // Step 6: Replace forOp with newForOp.
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  return newForOp;
}

// Here we assume the source and destination ops are in the same region op.
// Go through channels, and get a set of region ops containing channels.
void collectRegionsWithChannels(const SmallVector<Channel *> &channels,
                                DenseSet<Operation *> &regionsWithChannels) {
  for (auto *ch : channels) {
    auto *dst = ch->getDstOp();
    auto *pOp = dst->getParentOp();
    if (!pOp)
      continue;
    if (auto forOp = dyn_cast<scf::ForOp>(pOp))
      regionsWithChannels.insert(pOp);
    if (auto ifOp = dyn_cast<scf::IfOp>(pOp))
      regionsWithChannels.insert(pOp);
    // Channel directly in a persistent scf.while after region.
    if (auto whileOp = dyn_cast<scf::WhileOp>(pOp))
      regionsWithChannels.insert(pOp);
  }
}

void collectRegionsWithChannelsPost(
    const SmallVector<Channel *> &channels,
    DenseSet<Operation *> &regionsWithChannels) {
  for (auto *channel : channels) {
    if (channel->channelKind == DataChannelKind::TMEMPost) {
      ttng::TmemDataChannelPost *tmemChannel =
          static_cast<ttng::TmemDataChannelPost *>(channel);
      if (tmemChannel->isOperandD) {
        // Go through all dst ops and src ops.
        for (auto user : cast<ttng::TMEMAllocOp>(tmemChannel->allocOp)
                             .getResult()
                             .getUsers()) {
          auto *pOp = user->getParentOp();
          if (auto forOp = dyn_cast<scf::ForOp>(pOp)) {
            // Skip loops where the accumulator token is loop-carried —
            // the buffer doesn't rotate within such loops.
            if (hasLoopCarriedAccToken(tmemChannel->allocOp, forOp))
              continue;
            regionsWithChannels.insert(pOp);
          }
          if (auto ifOp = dyn_cast<scf::IfOp>(pOp))
            regionsWithChannels.insert(pOp);
          // Accumulator consumer/producer directly in a persistent scf.while
          // (e.g. the epilogue read in the after region). The accumulator
          // rotates once per persistent iteration, so the while carries it.
          if (auto whileOp = dyn_cast<scf::WhileOp>(pOp))
            regionsWithChannels.insert(pOp);
        }
        continue;
      }
    }
    auto *dst = channel->getDstOp();
    auto *src = channel->getSrcOp();
    auto *pOp = dst->getParentOp();
    if (!pOp)
      continue;
    if (auto forOp = dyn_cast<scf::ForOp>(pOp))
      regionsWithChannels.insert(pOp);
    if (auto ifOp = dyn_cast<scf::IfOp>(pOp))
      regionsWithChannels.insert(pOp);
    if (auto whileOp = dyn_cast<scf::WhileOp>(pOp))
      regionsWithChannels.insert(pOp);
    // When producer is in a different (outer) scope than consumer,
    // also register the producer's parent. This handles Q buffers in
    // persistent FA kernels: Q is produced in the outer tile loop but
    // consumed inside the inner KV loop. Without this, the outer loop
    // only gets 1 accumCnt (for the inner loop), and Q's phase uses
    // the inner loop's K/V counter instead of a separate Q counter.
    if (src) {
      auto *srcParent = src->getParentOp();
      if (srcParent && srcParent != pOp) {
        if (isa<scf::ForOp>(srcParent))
          regionsWithChannels.insert(srcParent);
        if (isa<scf::IfOp>(srcParent))
          regionsWithChannels.insert(srcParent);
        if (isa<scf::WhileOp>(srcParent))
          regionsWithChannels.insert(srcParent);
      }
    }
  }
}

// Go through a list of operations in opList, recursively call into
// createNewLoopWrapper or rewriteIfOp.
void updateAccumLoopCount(SmallVector<Operation *> &opList,
                          SmallVector<Operation *> &taskTopOps,
                          DenseSet<Operation *> &regionsWithChannels,
                          ReuseConfig *config) {
  DenseMap<Operation *, Operation *> oldToNew;
  for (Operation *op : opList) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto newForOp =
          createNewLoopWrapper(forOp, taskTopOps, regionsWithChannels, config);
      oldToNew[op] = newForOp.getOperation();
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      auto newWhileOp = createNewWhileWrapper(whileOp, taskTopOps,
                                              regionsWithChannels, config);
      oldToNew[op] = newWhileOp.getOperation();
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (enclosingAChannel(ifOp.getOperation(), regionsWithChannels)) {
        auto newIfOp =
            rewriteIfOp(ifOp, taskTopOps, regionsWithChannels, config);
        oldToNew[op] = newIfOp.getOperation();

        // Update prevAccum to be result of the new IfOp.
        assert(newIfOp.getNumResults() >= 1);
        auto numRes = newIfOp.getNumResults();
        Value prevAccum =
            newIfOp.getResult(numRes - 1); // accumCnt is the last result.
      } else {
        // Still need to process nested ForOps in pre-order.
        SmallVector<scf::ForOp> innerForOps;
        ifOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
          if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
            innerForOps.push_back(forOp);
          }
        });
        for (auto innerFor : innerForOps) {
          auto newFor = createNewLoopWrapper(innerFor, taskTopOps,
                                             regionsWithChannels, config);
          oldToNew[innerFor.getOperation()] = newFor.getOperation();
        }
      }
    }
  }
  for (unsigned i = 0; i < opList.size(); i++) {
    auto *oldOp = opList[i];
    if (oldToNew.find(oldOp) != oldToNew.end())
      opList[i] = oldToNew[oldOp];
  }
}

scf::ForOp createNewLoopWrapper(scf::ForOp origForOp,
                                SmallVector<Operation *> &taskTopOps,
                                DenseSet<Operation *> &regionsWithChannels,
                                ReuseConfig *config) {
  LLVM_DEBUG({
    LDBG("call createNewLoop on");
    origForOp.dump();
  });

  scf::ForOp parentForOp = origForOp->getParentOfType<scf::ForOp>();
  // The nearest enclosing loop that carries accumCnt can also be a persistent
  // scf.while: in that case the accumCnt is carried by the while's after
  // region, and the inner loop starts from the matching after-region argument.
  scf::WhileOp parentWhileOp;
  if (!parentForOp)
    parentWhileOp = origForOp->getParentOfType<scf::WhileOp>();

  // Block holding the parent's accumCnt arguments (for-loop body or while-loop
  // after body) and the parent op used for index computation.
  Block *parentAccumBlock = nullptr;
  Operation *parentAccumOp = nullptr;
  unsigned pArgSize = 0, pCnts = 0, accumArgId = 0;
  if (parentForOp) {
    parentAccumBlock = parentForOp.getBody();
    parentAccumOp = parentForOp.getOperation();
  } else if (parentWhileOp) {
    parentAccumBlock = parentWhileOp.getAfterBody();
    parentAccumOp = parentWhileOp.getOperation();
  }
  if (parentAccumBlock) {
    pArgSize = parentAccumBlock->getArguments().size();
    pCnts = getAccumCnts(parentAccumOp, regionsWithChannels, config);
  }

  SmallVector<Operation *> preOrderOps;
  getAccumCntsPreOrder(origForOp.getOperation(), regionsWithChannels,
                       preOrderOps);
  unsigned tCnts = preOrderOps.size();

  if (preOrderOps.size() > 0 && parentAccumOp) {
    // Find the accumArgId for preOrderOps[0] in the parent loop.
    accumArgId = getAccumArgIdx(parentAccumOp, preOrderOps[0],
                                regionsWithChannels, config, -1);
  }

  // Get initial value of accumCnts prior to the loop.
  SmallVector<Value> initialAccums;
  for (unsigned i = 0; i < tCnts; ++i) {
    // If there is an enclosing loop (for or while), use the corresponding
    // argument value so the counter continues across outer iterations.
    Value startAccum;
    if (parentAccumBlock)
      startAccum =
          parentAccumBlock->getArgument(pArgSize - pCnts + accumArgId + i);
    else {
      OpBuilderWithAsyncTaskIds builder(origForOp->getContext());
      builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(origForOp));
      builder.setInsertionPoint(origForOp);
      auto loc = origForOp.getLoc();
      startAccum =
          builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
    }
    initialAccums.push_back(startAccum);
  }
  // Handle reuse groups.
  for (unsigned idx = 0; idx < config->getGroupSize(); ++idx) {
    if (config->getGroup(idx)->channels.size() <= 1)
      continue;
    if (config->getGroup(idx)->channels[0]->getNumBuffers() <= 1)
      continue;
    Operation *parentOp = origForOp->getParentOp();
#if 0
    if (!isa<scf::ForOp>(parentOp) && !isa<scf::IfOp>(parentOp))
      continue;
#endif
    // Find channels of reuse group that are inside forOp. If the channel is
    // directly in forOp, add the channel's DstOp, otherwise add the region Op
    // that is directly in forOp.
    SmallVector<Operation *> chList;
    getReuseChannels(config->getGroup(idx), origForOp.getOperation(), chList);
    if (chList.empty())
      continue;

    // Find prevAccum right before the forOp.
    Value prevAccum;
    SmallVector<Operation *> parentChList;
    // Get a list of ops directly under parentOp that contain channels in the
    // reuse group.
    getReuseChannels(config->getGroup(idx), parentOp, parentChList);
    if (parentChList.empty())
      // There are channels in the reuse group that are under origForOp.
      parentChList.push_back(origForOp.getOperation());
    prevAccum = getAccumForReuseGroup(origForOp.getOperation(), parentChList,
                                      regionsWithChannels, config, idx, true);
    initialAccums.push_back(prevAccum);
  }

  scf::ForOp newForOp = createNewLoop(origForOp, parentForOp, initialAccums);
  LLVM_DEBUG({
    LDBG("after createNewLoop ");
    newForOp.dump();
  });

  // origForOp is erased in createNewLoop. Make sure taskTopOps is updated with
  // the newForOp.
  auto asyncTaskLoopForItr =
      std::find(taskTopOps.begin(), taskTopOps.end(), origForOp.getOperation());
  if (asyncTaskLoopForItr != taskTopOps.end()) {
    *asyncTaskLoopForItr = newForOp.getOperation();
  }
  auto tmpIter3 =
      std::find(regionsWithChannels.begin(), regionsWithChannels.end(),
                origForOp.getOperation());
  if (tmpIter3 != regionsWithChannels.end()) {
    *tmpIter3 = newForOp.getOperation();
  }

  // Handle ops in loop body, only IfOps and ForOps.
  SmallVector<Operation *> opList;
  for (Operation &op : newForOp.getBody()->without_terminator()) {
    if (auto tOp = dyn_cast<scf::ForOp>(&op))
      opList.push_back(&op);
    if (auto tOp = dyn_cast<scf::IfOp>(&op))
      opList.push_back(&op);
  }
  updateAccumLoopCount(opList, taskTopOps, regionsWithChannels, config);
  LLVM_DEBUG({
    LDBG("-- before replacing yieldOp ");
    newForOp.dump();
  });

  // Update yieldOp.
  Operation *yieldOp = newForOp.getBody()->getTerminator();
  unsigned tSize = newForOp.getBody()->getArguments().size();
  auto numAccumCnts = initialAccums.size();
  if (numAccumCnts == 0)
    return newForOp;

  // Start with the first accumCnt.
  accumArgId = tSize - numAccumCnts;
  LDBG("-- tSize, numAccumCnts, accumArgId " << tSize << " " << numAccumCnts
                                             << " " << accumArgId);

  // If there is a channel directly in forOp, it should be the first accumCnt.
  for (auto *op : regionsWithChannels) {
    if (newForOp.getOperation() == op) {
      Value endAccum = generateYieldCntsForForOp(newForOp, accumArgId);
      Value arg = newForOp.getBody()->getArgument(accumArgId);

      // Make sure accumCnt = argValue + 1, increment by 1.
      // In createNewLoop, yieldOp yields the argument value directly, it is
      // fixed here.
      yieldOp->replaceUsesOfWith(arg, endAccum);
      ++accumArgId;
      break;
    }
  }
  LDBG("-- accumArgId after channels directly under " << accumArgId);

  // Handle the loop body. This order should align with the preorder that is
  // used for accumCnts.
  SmallVector<Operation *> dummy;
  // Track seen ops for the reuse group section.
  DenseSet<Operation *> seenOps;
  for (auto *op : opList) {
    if (!enclosingAChannel(op, regionsWithChannels))
      continue;

    auto numRes = op->getNumResults();
    unsigned tCnts = getAccumCnts(op, regionsWithChannels, config);
    LDBG("-- numRes, tCnts, accumArgId " << numRes << " " << tCnts << " "
                                         << accumArgId);
    // Each accumCnt nested under "op", it will have a corresponding argument in
    // this "ForOp". If "op" has tCnts, this "ForOp" will have the same number
    // of corresponding accumCnts, in the same order.
    for (unsigned i = 0; i < tCnts; ++i) {
      Value arg = newForOp.getBody()->getArgument(accumArgId);
      Value endAccum = op->getResult(numRes - tCnts + i);
      LLVM_DEBUG({
        LDBG("-- replace use of arg with result " << numRes - tCnts + i);
        op->dump();
      });
      // In createNewLoop, yieldOp yields the argument value directly, it is
      // fixed here. Now, it will yield the accumCnt from the "op".
      yieldOp->replaceUsesOfWith(arg, endAccum);
      LLVM_DEBUG(yieldOp->dump());
      ++accumArgId;
    }
    // Insert ops for control flow to ensure they aren't also processed
    // in the reuse group section.
    if (tCnts > 0)
      seenOps.insert(op);
  }

  // Handle reuse groups.
  for (unsigned idx = 0; idx < config->getGroupSize(); ++idx) {
    // Find channels of reuse group that are inside forOp. If the channel is
    // directly in forOp, add the channel's DstOp, otherwise add the region Op
    // that is directly in forOp.
    SmallVector<Operation *> chList;
    getReuseChannels(config->getGroup(idx), newForOp.getOperation(), chList);
    if (chList.empty())
      continue;
    Operation *lastCh = chList.back();
    // Check if we have already accounted for this accumulator via nesting.
    if (seenOps.contains(lastCh))
      continue;
    auto forYield = getAccumForReuseGroup(lastCh, chList, regionsWithChannels,
                                          config, idx, false);
    Value arg = newForOp.getBody()->getArgument(accumArgId);
    yieldOp->replaceUsesOfWith(arg, forYield);
    ++accumArgId;
  }

  LLVM_DEBUG({
    LDBG("-- after all replacing ");
    newForOp.dump();
  });
  return newForOp;
}

// Recreate `whileOp` with `numAccumCnts` additional i64 loop-carried values
// threaded through its before region, condition op, after region, and yield.
// The new counters are appended:
//   - to the init operands (seeded from `initialAccums`),
//   - as the trailing before-region block arguments,
//   - forwarded verbatim by the scf.condition into the after region (and the
//     while results),
//   - as the trailing after-region block arguments, and
//   - yielded back as themselves (a placeholder the caller rewrites so the
//     after region yields the inner loop's final accumCnt).
static scf::WhileOp createNewWhileLoop(scf::WhileOp whileOp,
                                       SmallVector<Value> &initialAccums) {
  auto loc = whileOp.getLoc();
  OpBuilderWithAsyncTaskIds builder(whileOp.getContext());
  builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(whileOp));
  unsigned numAccumCnts = initialAccums.size();

  Block *beforeBlk = whileOp.getBeforeBody();
  Block *afterBlk = whileOp.getAfterBody();

  // Step 1: append i64 accumCnt arguments to the before and after blocks.
  for (unsigned i = 0; i < numAccumCnts; ++i) {
    beforeBlk->addArgument(builder.getI64Type(), accumCntLoc(loc));
    afterBlk->addArgument(builder.getI64Type(), accumCntLoc(loc));
  }

  // Step 2: forward the new before-region args through the condition op so they
  // reach the after region (and the while results).
  auto condOp = whileOp.getConditionOp();
  builder.setInsertionPoint(condOp);
  SmallVector<Value> condArgs(condOp.getArgs());
  unsigned beforeArgN = beforeBlk->getNumArguments();
  for (unsigned i = 0; i < numAccumCnts; ++i)
    condArgs.push_back(beforeBlk->getArgument(beforeArgN - numAccumCnts + i));
  builder.createWithAsyncTaskIds<scf::ConditionOp>(
      condOp.getLoc(), condOp.getCondition(), condArgs);
  condOp.erase();

  // Step 3: yield the new after-region args (placeholder; the caller replaces
  // these uses with the inner loop's final accumCnt value).
  auto yieldOp = whileOp.getYieldOp();
  builder.setInsertionPoint(yieldOp);
  SmallVector<Value> yieldArgs(yieldOp.getOperands());
  unsigned afterArgN = afterBlk->getNumArguments();
  for (unsigned i = 0; i < numAccumCnts; ++i)
    yieldArgs.push_back(afterBlk->getArgument(afterArgN - numAccumCnts + i));
  builder.createWithAsyncTaskIds<scf::YieldOp>(yieldOp.getLoc(), yieldArgs);
  yieldOp.erase();

  // Step 4: build the new while op with extended init operands / result types,
  // then move the (already-extended) before/after regions into it.
  SmallVector<Value> newInits(whileOp.getInits());
  for (auto v : initialAccums)
    newInits.push_back(v);
  SmallVector<Type> newResultTypes(whileOp->getResultTypes());
  for (unsigned i = 0; i < numAccumCnts; ++i)
    newResultTypes.push_back(builder.getI64Type());

  builder.setInsertionPoint(whileOp);
  auto newWhileOp = builder.createWithAsyncTaskIds<scf::WhileOp>(
      loc, newResultTypes, newInits);
  for (auto attr : whileOp->getAttrs())
    newWhileOp->setAttr(attr.getName(), attr.getValue());
  newWhileOp.getBefore().takeBody(whileOp.getBefore());
  newWhileOp.getAfter().takeBody(whileOp.getAfter());

  // Step 5: rewire the original results and erase the old loop.
  for (unsigned i = 0; i < whileOp.getNumResults(); ++i)
    whileOp.getResult(i).replaceAllUsesWith(newWhileOp.getResult(i));
  whileOp.erase();
  return newWhileOp;
}

// Thread accumCnt through a persistent scf.while loop. The while is assumed to
// be outermost (its parent is the func), so the counters are seeded to 0 and
// carried across persistent iterations. Channel-bearing region ops (the inner
// warp-specialized scf.for) live in the after region and are processed
// recursively; the final accumCnt they produce is yielded back so the next
// persistent iteration continues from the correct buffer index / phase.
scf::WhileOp createNewWhileWrapper(scf::WhileOp origWhileOp,
                                   SmallVector<Operation *> &taskTopOps,
                                   DenseSet<Operation *> &regionsWithChannels,
                                   ReuseConfig *config) {
  LLVM_DEBUG({
    LDBG("call createNewWhileWrapper on");
    origWhileOp.dump();
  });

  // One accumCnt per channel-bearing region op nested under the while.
  SmallVector<Operation *> preOrderOps;
  getAccumCntsPreOrder(origWhileOp.getOperation(), regionsWithChannels,
                       preOrderOps);
  unsigned tCnts = preOrderOps.size();
  if (tCnts == 0)
    return origWhileOp;

  OpBuilderWithAsyncTaskIds builder(origWhileOp->getContext());
  builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(origWhileOp));
  builder.setInsertionPoint(origWhileOp);
  auto loc = origWhileOp.getLoc();

  // The persistent while is outermost: seed every counter to 0.
  SmallVector<Value> initialAccums;
  for (unsigned i = 0; i < tCnts; ++i)
    initialAccums.push_back(
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64));

  // Reuse groups (e.g. the multi-buffered epilogue TMA-store staging that the
  // subtiled epilogue creates) get their own accumCnt, appended after the
  // per-region counters. Mirrors createNewLoopWrapper. The while is outermost,
  // so the prior value resolves to a constant 0.
  for (unsigned idx = 0; idx < config->getGroupSize(); ++idx) {
    if (config->getGroup(idx)->channels.size() <= 1)
      continue;
    if (config->getGroup(idx)->channels[0]->getNumBuffers() <= 1)
      continue;
    SmallVector<Operation *> chList;
    getReuseChannels(config->getGroup(idx), origWhileOp.getOperation(), chList);
    if (chList.empty())
      continue;
    Operation *parentOp = origWhileOp->getParentOp();
    SmallVector<Operation *> parentChList;
    getReuseChannels(config->getGroup(idx), parentOp, parentChList);
    if (parentChList.empty())
      parentChList.push_back(origWhileOp.getOperation());
    Value prevAccum =
        getAccumForReuseGroup(origWhileOp.getOperation(), parentChList,
                              regionsWithChannels, config, idx, true);
    initialAccums.push_back(prevAccum);
  }

  unsigned totalCnts = initialAccums.size();
  scf::WhileOp newWhileOp = createNewWhileLoop(origWhileOp, initialAccums);

  // Update bookkeeping references from origWhileOp to newWhileOp.
  auto asyncTaskItr = std::find(taskTopOps.begin(), taskTopOps.end(),
                                origWhileOp.getOperation());
  if (asyncTaskItr != taskTopOps.end())
    *asyncTaskItr = newWhileOp.getOperation();
  if (regionsWithChannels.contains(origWhileOp.getOperation())) {
    regionsWithChannels.erase(origWhileOp.getOperation());
    regionsWithChannels.insert(newWhileOp.getOperation());
  }

  // Recursively thread accumCnt through the channel-bearing region ops in the
  // after region (e.g. the inner warp-specialized scf.for). Their start value
  // is wired to the while's after-region accumCnt arguments by
  // createNewLoopWrapper.
  SmallVector<Operation *> opList;
  for (Operation &op : newWhileOp.getAfterBody()->without_terminator()) {
    if (isa<scf::ForOp>(&op) || isa<scf::IfOp>(&op))
      opList.push_back(&op);
  }
  updateAccumLoopCount(opList, taskTopOps, regionsWithChannels, config);

  // Wire the after-region yield so each accumCnt yields the final value
  // produced by its region op, instead of the placeholder pass-through created
  // in createNewWhileLoop. This mirrors the for-loop body handling in
  // createNewLoopWrapper.
  Operation *yieldOp = newWhileOp.getAfterBody()->getTerminator();
  unsigned afterArgSize = newWhileOp.getAfterBody()->getNumArguments();
  unsigned accumArgId = afterArgSize - totalCnts;

  // If the while directly carries a channel (the accumulator/epilogue channel
  // in the after region), its counter is first in preorder and advances by one
  // per persistent iteration. This mirrors the direct-channel handling in
  // createNewLoopWrapper.
  for (auto *op : regionsWithChannels) {
    if (newWhileOp.getOperation() == op) {
      Value arg = newWhileOp.getAfterBody()->getArgument(accumArgId);
      OpBuilderWithAsyncTaskIds yieldBuilder(newWhileOp.getContext());
      yieldBuilder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(newWhileOp));
      yieldBuilder.setInsertionPoint(yieldOp);
      Value one =
          yieldBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
      Value endAccum = yieldBuilder.createWithAsyncTaskIds<arith::AddIOp>(
          accumCntLoc(loc), arg, one);
      yieldOp->replaceUsesOfWith(arg, endAccum);
      ++accumArgId;
      break;
    }
  }

  // Nested channel-bearing region ops (e.g. the inner warp-specialized
  // scf.for): yield the final accumCnt each one produces.
  DenseSet<Operation *> seenOps;
  for (auto *op : opList) {
    if (!enclosingAChannel(op, regionsWithChannels))
      continue;
    auto numRes = op->getNumResults();
    unsigned cnts = getAccumCnts(op, regionsWithChannels, config);
    for (unsigned i = 0; i < cnts; ++i) {
      Value arg = newWhileOp.getAfterBody()->getArgument(accumArgId);
      Value endAccum = op->getResult(numRes - cnts + i);
      yieldOp->replaceUsesOfWith(arg, endAccum);
      ++accumArgId;
    }
    if (cnts > 0)
      seenOps.insert(op);
  }

  // Reuse-group counters: yield the staggered accumCnt for each multi-buffered
  // reuse group with channels in the while. Mirrors createNewLoopWrapper.
  for (unsigned idx = 0; idx < config->getGroupSize(); ++idx) {
    SmallVector<Operation *> chList;
    getReuseChannels(config->getGroup(idx), newWhileOp.getOperation(), chList);
    if (chList.empty())
      continue;
    Operation *lastCh = chList.back();
    if (seenOps.contains(lastCh))
      continue;
    Value whileYield = getAccumForReuseGroup(
        lastCh, chList, regionsWithChannels, config, idx, false);
    Value arg = newWhileOp.getAfterBody()->getArgument(accumArgId);
    yieldOp->replaceUsesOfWith(arg, whileYield);
    ++accumArgId;
  }

  LLVM_DEBUG({
    LDBG("-- after createNewWhileWrapper ");
    newWhileOp.dump();
  });
  return newWhileOp;
}

void appendAccumCntsForOps(SmallVector<Operation *> &taskTopOps,
                           const SmallVector<Channel *> &channels,
                           DenseSet<Operation *> &regionsWithChannels,
                           ReuseConfig *config) {

  SmallVector<Operation *> opList;
  for (auto &op : taskTopOps) {
    if (auto origIfOp = dyn_cast<scf::IfOp>(op)) {
      opList.push_back(op);
    }
    if (auto origForOp = dyn_cast<scf::ForOp>(op))
      opList.push_back(op);
    // A static persistent kernel's outer loop can be an scf.while; thread
    // accumCnt through it just like the persistent scf.for case.
    if (auto origWhileOp = dyn_cast<scf::WhileOp>(op))
      opList.push_back(op);
  }
  Value tmpAccumLoopCount;
  // Go through all the regions in opList and correctly add accumCnt. taskTopOps
  // will be updated if it is replaced in the process.
  // tmpAccumLoopCount is the current accumCnt;
  updateAccumLoopCount(opList, taskTopOps, regionsWithChannels, config);
}

} // namespace mlir
