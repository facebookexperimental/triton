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
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUWSCODEPARTITION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-warp-spec-code-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

std::pair<int, bool> scanRegUsage(Block *block, AsyncTaskId asyncTaskId,
                                  int regDecProducer, int regIncConsumer) {
  // TODO: scan ops to estimate register usage
  if (asyncTaskId == 0) {
    // deallocate registers
    return {regDecProducer == 0 ? 40 : regDecProducer, false};
  } else {
    // allocate registers
    return {regIncConsumer == 0 ? 232 : regIncConsumer, true};
  }
}

unsigned getNumBuffersOrDefault(scf::ForOp forOp, unsigned numBuffers) {
  // Use the attribute attached to the loop if it exists otherwise use the
  // global control.
  if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
    return numBuffers;
  return mlir::cast<IntegerAttr>(
             forOp->getAttr(mlir::triton::kNumStagesAttrName))
      .getInt();
}

// Collect argument indices that are used by the specific taskId.
static SmallVector<unsigned> collectBlockArgsForTask(scf::ForOp forOp,
                                                     int asyncTaskId) {

  // Collect argument indices that can be reached along the definition chain.
  SetVector<unsigned> argIndices;
  std::function<void(Value, unsigned)> dfs = [&](Value arg, unsigned argIdx) {
    for (auto user : arg.getUsers()) {
      // Skip ops that are not in the same async task
      if (!hasAsyncTaskId(user, asyncTaskId))
        continue;

      if (isa<scf::YieldOp>(user)) {
        if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp())) {
          // For block arguments, we need to check the initial value as well.
          if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
            auto initArg = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
            if (Operation *def = initArg.getDefiningOp()) {
              if (hasAsyncTaskId(def, asyncTaskId)) {
                argIndices.insert(argIdx);
              }
            } else {
              llvm_unreachable("Initial value should have a defining op");
            }
          }
        }

        // Skip control flow ops that are shared by all async tasks
        continue;
      }
      // If use is the initial value of ForOp argument.
      if (auto userFor = dyn_cast<scf::ForOp>(user)) {
        // For block arguments, we need to check the initial value as well.
        if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
          auto initArg = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
          if (Operation *def = initArg.getDefiningOp()) {
            if (hasAsyncTaskId(def, asyncTaskId)) {
              argIndices.insert(argIdx);
            }
          } else {
            llvm_unreachable("Initial value should have a defining op");
          }
        }
        return;
      }

      // Found a real user, the arg is needed
      if (user->getNumRegions() == 0) {
        argIndices.insert(argIdx);
        return;
      }

      // Iterate through all regions of the user operation
      for (auto &region : user->getRegions()) {
        for (auto regionArg : region.getArguments()) {
          if (arg == regionArg)
            dfs(regionArg, argIdx);
        }
      }
    }
  };

  // check dependency with DFS traversal for loop args and results.
  mlir::Block &block = forOp.getRegion().front();
  for (unsigned i = forOp.getNumInductionVars(); i < block.getNumArguments();
       ++i) {
    auto arg = block.getArgument(i);
    dfs(arg, i - forOp.getNumInductionVars());
  }
  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    auto result = forOp->getResult(i);
    dfs(result, i);
  }

  SmallVector<unsigned> args(argIndices.begin(), argIndices.end());
  llvm::sort(args);
  return args;
}

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilderWithAsyncTaskIds &builder,
                        AsyncTaskId asyncTaskId);

// Check to see if op is enclosed under ifOp.
static bool enclosing(scf::IfOp ifOp, Operation *op) {
  auto pOp = op->getParentOfType<scf::IfOp>();
  while (pOp) {
    if (pOp == ifOp)
      return true;
    pOp = pOp->getParentOfType<scf::IfOp>();
  }
  return false;
}

static bool enclosing(scf::ForOp forOp, Operation *op) {
  auto pOp = op->getParentOfType<scf::ForOp>();
  while (pOp) {
    if (pOp == forOp)
      return true;
    pOp = pOp->getParentOfType<scf::ForOp>();
  }
  return false;
}

// Check to see if there is no outer loop that is enclosed under ifOp.
static bool immediateEnclosing(scf::IfOp ifOp, Operation *subOp) {
  auto pOp = subOp->getParentOfType<scf::ForOp>();
  if (!pOp)
    return true;
  return !enclosing(ifOp, pOp.getOperation());
}

static bool enclosingAChannel(Operation *ctrlOp,
                              const DenseSet<Operation *> &opsWithChannels) {
  for (auto *op : opsWithChannels) {
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

// Return true if the IfOp contains a ForOp that is in opsWithBufferReuse.
// We want to support reuse between channels in a loop and channels in a IfOp.
static bool
needAccumulatedLoopCntForReuse(scf::IfOp ifOp,
                               SmallVector<Operation *> &opsWithBufferReuse) {
  bool needAccum = false;
  ifOp.walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    for (auto tOp : opsWithBufferReuse) {
      if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
        // For the case of ifOp contains forOp, which contains subOp, no need to
        // generate accumLoopCount for ifOp.
        if (subOp == tOp && immediateEnclosing(ifOp, tOp)) {
          needAccum = true;
          break;
        }
      } else {
        if (subOp == tOp) {
          needAccum = true;
          break;
        }
      }
    }
  });
  return needAccum;
}

// Return the argument that tracks accumLoopCount if there is an outer
// ForOp.
Value getReuseAccumCntArg(scf::ForOp parentForOp) {
  assert(parentForOp);
  auto tSize = parentForOp.getBody()->getArguments().size();
  assert(tSize >= 1); // With buffer reuse, a single accumCnt
  Value tmpAccumLoopCount = parentForOp.getBody()->getArgument(tSize - 1);
  return tmpAccumLoopCount;
}

static bool channelWithReuse(Operation *dstOp,
                             SmallVector<Operation *> &opsWithBufferReuse) {
  for (auto *op : opsWithBufferReuse) {
    if (dstOp == op) {
      return true;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      if (enclosing(forOp, dstOp))
        return true;
    if (auto ifOp = dyn_cast<scf::IfOp>(op))
      if (enclosing(ifOp, dstOp))
        return true;
  }
  return false;
}

// opsWithChannels: ctrl ops with channels directly under
void excludeChannelsWithReuse(const DenseSet<Operation *> &opsWithChannels,
                              SmallVector<Operation *> &opsWithBufferReuse,
                              DenseSet<Operation *> &excludeReuse) {
  for (auto *dstOp : opsWithChannels) {
    if (!channelWithReuse(dstOp, opsWithBufferReuse))
      excludeReuse.insert(dstOp);
  }
}

// Return number of AccumCnts for the given ctrlOp. Add a single
// AccumCnt for all channels under opsWithBufferReuse and it will be the
// last AccumCnt.
unsigned getAccumCnts(Operation *ctrlOp,
                      const DenseSet<Operation *> &opsWithChannels,
                      SmallVector<Operation *> &opsWithBufferReuse) {
  unsigned cnt = 0;
  // Add a single count for all channels under opsWithBufferReuse.
  DenseSet<Operation *> excludeReuse;
  excludeChannelsWithReuse(opsWithChannels, opsWithBufferReuse, excludeReuse);
  LDBG("getAccumCnts: " << ctrlOp);
  for (auto *op : opsWithBufferReuse) {
    LDBG("-- getAccumCnts: " << ctrlOp << " opsWithBufferReuse " << op);
  }
  for (auto *op : excludeReuse) {
    LDBG("-- getAccumCnts: " << ctrlOp << " excludeReuse " << op);
    if (ctrlOp == op) {
      ++cnt;
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp))
      if (enclosing(forOp, op))
        ++cnt;
    if (auto ifOp = dyn_cast<scf::IfOp>(ctrlOp))
      if (enclosing(ifOp, op))
        ++cnt;
  }
  if (auto tIf = dyn_cast<scf::IfOp>(ctrlOp))
    if (needAccumulatedLoopCntForReuse(tIf, opsWithBufferReuse))
      ++cnt;
  if (dyn_cast<scf::ForOp>(ctrlOp))
    if (opsWithBufferReuse.size() > 1)
      ++cnt;
  return cnt;
}

// Ignore channels under opsWithBufferReuse. Update preOrderOps with a list
// of Ctrl Ops that will need accumCnt as arguments/results of CtrlOp.
void getAccumCntsPreOrder(Operation *ctrlOp,
                          const DenseSet<Operation *> &opsWithChannels,
                          SmallVector<Operation *> &opsWithBufferReuse,
                          SmallVector<Operation *> &preOrderOps) {
  DenseSet<Operation *> excludeReuse;
  excludeChannelsWithReuse(opsWithChannels, opsWithBufferReuse, excludeReuse);
  for (auto *op : excludeReuse) {
    LDBG("getAccumCntsPreOrder: " << ctrlOp << " excludeReuse " << op);
  }
  ctrlOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    // This will walk ctrlOp.
    if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
      LDBG("-- getAccumCntsPreOrder: walk forOp " << subOp);
      LDBG("-- opsWithChannels: " << excludeReuse.size() << " "
                                  << excludeReuse.count(subOp));
      for (auto *op : excludeReuse) {
        if (subOp == op) {
          LDBG("-- opsWithChannels push to result");
          preOrderOps.push_back(subOp);
        }
      }
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(subOp)) {
      LDBG("-- getAccumCntsPreOrder: walk IfOp " << subOp);
      LDBG("-- opsWithChannels: " << excludeReuse.size() << " "
                                  << excludeReuse.count(subOp));
      for (auto *op : excludeReuse) {
        if (subOp == op) {
          preOrderOps.push_back(subOp);
          LDBG("-- opsWithChannels push to result");
        }
      }
    }
  });
  LDBG("-- getAccumCntsPreOrder: " << ctrlOp << " size " << preOrderOps.size());
}

// Assume parentForOp has accumCnt for the specified ctrlOp. For channels with
// reuse, use getReuseAccumCntArg.
unsigned getAccumArgIdx(scf::ForOp parentForOp, Operation *ctrlOp,
                        const DenseSet<Operation *> &opsWithChannels,
                        SmallVector<Operation *> &opsWithBufferReuse) {
  DenseSet<Operation *> excludeReuse;
  excludeChannelsWithReuse(opsWithChannels, opsWithBufferReuse, excludeReuse);
  // Walk parentForOp in preorder.
  unsigned preOrderId = 0, ctrlId = 0;
  bool found = false;
  parentForOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    // This will walk parentForOp.
    if (subOp == ctrlOp) {
      ctrlId = preOrderId;
      found = true;
    }
    for (auto *op : excludeReuse) {
      if (op == subOp) {
        LDBG("getAccumArgIdx: saw ctrlOp enclosing channel " << subOp);
        ++preOrderId;
      }
    }
  });
  assert(found && "error in getAccumArgIdx");
  LDBG("getAccumArgIdx: " << parentForOp.getOperation() << " " << ctrlOp << " "
                          << ctrlId);
  return ctrlId;
}

// Get the current accumulation count for the given op within its immediate
// scope.
// ForA (accumForA, accumIfA, accumForB, accumIfB)
//   IfA (accumIfA, accumForB)
//     Channel A --> uses ForA.arg[accumIfA]
//     ForB (accumForB)
//       Channel B --> uses ForB.arg[accumForB]
//   ThenYield ForA.arg[accumIfA] + 1, ForB.res[accumForB]
//   ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
//   ForC (accumForC, accumIfB)
//     IfB
//       Channel C --> uses ForC.arg[accumIfB]
//     ThenYield ForC.arg[accumIfB] + 1
//     ElseYield ForC.arg[accumIfB]
//   Channel D --> uses ForA.arg[accumForA]
// Right now, we only support a limited form of buffer reuse. We only allow
// reuses among a list of parallel control ops. And we will add a single
// AccumCnt as the last argument.
Value getAccumCount(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                    const DenseSet<Operation *> &opsWithChannels,
                    SmallVector<Operation *> &opsWithBufferReuse) {
  auto parentForOp = op->getParentOfType<scf::ForOp>();
  auto *pOp = op->getParentOp();
  // Get parentForOp.arg[pOp]
  unsigned accumArgId;
  unsigned tSize = parentForOp.getBody()->getArguments().size();
  unsigned parentTCnts =
      getAccumCnts(parentForOp, opsWithChannels, opsWithBufferReuse);
  Value accumCnt;
  bool partOfReuse = false;
  if (opsWithBufferReuse.size() > 1) {
    partOfReuse = channelWithReuse(op, opsWithBufferReuse);
  }
  if (opsWithBufferReuse.size() > 1 && partOfReuse) {
    // Check to see if the op is inside opsWithBufferReuse.
    accumCnt = parentForOp.getBody()->getArguments().back();
    accumArgId = parentTCnts - 1;
  } else {
    accumArgId =
        getAccumArgIdx(parentForOp, pOp, opsWithChannels, opsWithBufferReuse);
    accumCnt =
        parentForOp.getBody()->getArgument(tSize - parentTCnts + accumArgId);
  }

  LDBG("getAccumCount: parentForOp " << parentForOp.getOperation() << " pOp "
                                     << pOp << " " << tSize << " "
                                     << parentTCnts << " " << accumArgId);
  return accumCnt;
}

// Compute and return the buffer index and phase for a given accumulate count.
std::pair<Value, Value> getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                                             Location loc, Value accumCnt,
                                             unsigned numBuffers) {
  Value numBuffersVal =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, numBuffers, 32);
  numBuffersVal = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
      loc, builder.getI64Type(), numBuffersVal);
  // Calculate accumCnt / numBuffers
  // initBufferIdx = accumCnt - accumCnt / numBuffers * numBuffers
  // initPhase = (accumCnt / numBuffers) & 1
  Value bufferIdx = builder.createWithAsyncTaskIds<arith::DivUIOp>(
      loc, accumCnt, numBuffersVal);
  Value initBufferIdx = builder.createWithAsyncTaskIds<arith::SubIOp>(
      loc, accumCnt,
      builder.createWithAsyncTaskIds<arith::MulIOp>(loc, bufferIdx,
                                                    numBuffersVal));
  initBufferIdx = builder.createWithAsyncTaskIds<arith::TruncIOp>(
      loc, builder.getI32Type(), initBufferIdx);

  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
  bufferIdx =
      builder.createWithAsyncTaskIds<arith::AndIOp>(loc, bufferIdx, one);
  Value initPhase = builder.createWithAsyncTaskIds<arith::TruncIOp>(
      loc, builder.getI1Type(), bufferIdx);
  return {initBufferIdx, initPhase};
}

void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &opsWithChannels,
                          Value &bufferIdx, Value &phase,
                          SmallVector<Operation *> &opsWithBufferReuse) {
  Value accumCnt =
      getAccumCount(builder, op, opsWithChannels, opsWithBufferReuse);
  std::tie(bufferIdx, phase) =
      getBufferIdxAndPhase(builder, op->getLoc(), accumCnt, numBuffers);
}

// Get the bufferIdx and phase for the last iteration of the immediate scope.
std::pair<Value, Value>
getOutOfScopeBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                               Operation *op, unsigned numBuffers,
                               const DenseSet<Operation *> &opsWithChannels,
                               SmallVector<Operation *> &opsWithBufferReuse) {
  // Get the current in-scope accumulation count for op.
  Value accumCnt =
      getAccumCount(builder, op, opsWithChannels, opsWithBufferReuse);

  // Get the out-of-scope accumulation count.
  assert(isa<BlockArgument>(accumCnt) &&
         "Expected accumCnt to be a block argument");
  auto bbArg = dyn_cast<BlockArgument>(accumCnt);
  Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
  if (auto forOp = dyn_cast<scf::ForOp>(bbAargOwner)) {
    accumCnt = forOp.getResult(bbArg.getArgNumber() - 1);
  } else {
    llvm_unreachable("Unexpected block argument owner");
  }

  // The accumulation count is one past the last iteration. Subtract one to get
  // the last valid iteration index.
  auto loc = bbAargOwner->getLoc();
  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
  accumCnt = builder.createWithAsyncTaskIds<arith::SubIOp>(loc, accumCnt, one);

  return getBufferIdxAndPhase(builder, op->getLoc(), accumCnt, numBuffers);
}

static bool
needAccumulatedLoopCnt(scf::IfOp ifOp,
                       SmallVector<Operation *> &opsWithBufferReuse,
                       const DenseSet<Operation *> &opsWithChannels) {
  if (opsWithBufferReuse.size() > 1)
    return needAccumulatedLoopCntForReuse(ifOp, opsWithBufferReuse);
  return enclosingAChannel(ifOp.getOperation(), opsWithChannels);
}

// op is up-to-date (i.e will be updated when a control op is re-written).
Value updateAccumLoopCount(SmallVector<Operation *> &opList,
                           unsigned numBuffers,
                           SmallVector<Operation *> &taskTopOps,
                           Operation *commonOuterLoop,
                           SmallVector<Operation *> &opsWithBufferReuse,
                           DenseSet<Operation *> &opsWithChannels,
                           Value prevAccum);

scf::ForOp createNewLoopWrapper(scf::ForOp origForOp, unsigned numBuffers,
                                SmallVector<Operation *> &taskTopOps,
                                Operation *commonOuterLoop,
                                SmallVector<Operation *> &opsWithBufferReuse,
                                DenseSet<Operation *> &opsWithChannels,
                                Value prevAccum);

// For certain cases, we need to add an additional output for
// IfOp to track the accumulatedLoopCount, we may need to add
// a corresponding elseBlock with yieldOp.
scf::IfOp rewriteIfOp(scf::IfOp ifOp, unsigned numBuffers,
                      SmallVector<Operation *> &taskTopOps,
                      Operation *commonOuterLoop,
                      SmallVector<Operation *> &opsWithBufferReuse,
                      DenseSet<Operation *> &opsWithChannels, Value prevAccum) {
  LLVM_DEBUG({
    LDBG("rewrite ifOp for smem sharing ");
    ifOp.dump();
  });

  OpBuilderWithAsyncTaskIds ifBuilder(ifOp.getContext());
  ifBuilder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(ifOp));
  ifBuilder.setInsertionPoint(ifOp);

  unsigned numAccumCnts =
      getAccumCnts(ifOp.getOperation(), opsWithChannels, opsWithBufferReuse);

  SmallVector<Type> newResultTypes(ifOp->getResultTypes());
  bool hasBufferReuse = opsWithBufferReuse.size() > 1;
  for (unsigned i = 0; i < numAccumCnts; ++i)
    newResultTypes.push_back(ifBuilder.getI64Type());
  LDBG("rewrite ifOp: add " << numAccumCnts << " accumCnts");
  assert(numAccumCnts > 0);
  // Create else block if we need to generate accumulated loop count.
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

  // Update yields
  auto loc = ifOp.getLoc();
  auto updateYield = [&](scf::YieldOp yield, SmallVector<Value> &operands) {
    ifBuilder.setInsertionPoint(yield);
    ifBuilder.createWithAsyncTaskIds<scf::YieldOp>(loc, operands);
    yield.erase();
  };

  // Update opsWithChannels now that newIfOp takes over the body.
  auto tmpIter3 = std::find(opsWithChannels.begin(), opsWithChannels.end(),
                            ifOp.getOperation());
  if (tmpIter3 != opsWithChannels.end()) {
    LDBG("rewrite ifOp: update opsWithChannels "
         << ifOp.getOperation() << " --> " << newIfOp.getOperation());
    *tmpIter3 = newIfOp.getOperation();
  }

  // Add one more operand to then Yield.
  Value endAccum =
      updateAccumLoopCount(opList, numBuffers, taskTopOps, commonOuterLoop,
                           opsWithBufferReuse, opsWithChannels, prevAccum);
  Value endAccumReuseThen = endAccum, endAccumReuseElse;

  SmallVector<Value> ifYieldOperands = newIfOp.thenYield().getOperands();

  // Handle elseRegion of the IfOp.
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
    if (hasBufferReuse) {
      endAccumReuseElse = updateAccumLoopCount(
          opListElse, numBuffers, taskTopOps, commonOuterLoop,
          opsWithBufferReuse, opsWithChannels, prevAccum);
    } else {
      // We need to differentiate channels in then region vs. in else region.
      // For now, only handle the case where channels are in then region.
      for (auto *op : opListElse)
        assert(!enclosingAChannel(op, opsWithChannels));
    }
  } else {
    // Create an empty yield
    auto yieldOp =
        newIfOp.getElseBodyBuilder().create<scf::YieldOp>(ifOp.getLoc());
    endAccumReuseElse = prevAccum;
  }

  SmallVector<Value> elseYieldOperands = newIfOp.elseYield().getOperands();
  OpBuilderWithAsyncTaskIds elseBuilder(ifOp.getContext());
  elseBuilder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(ifOp));
  elseBuilder.setInsertionPoint(newIfOp.elseYield());
  ifBuilder.setInsertionPoint(newIfOp.thenYield());

  auto parentForOp = newIfOp->getParentOfType<scf::ForOp>();
  unsigned tSize, parentTCnts = 0;
  SmallVector<Operation *> preOrderOpsOfParent;
  if (parentForOp) {
    tSize = parentForOp.getBody()->getArguments().size();
    getAccumCntsPreOrder(parentForOp.getOperation(), opsWithChannels,
                         opsWithBufferReuse, preOrderOpsOfParent);
    parentTCnts =
        getAccumCnts(parentForOp.getOperation(), opsWithChannels,
                     opsWithBufferReuse); // preOrderOpsOfParent.size();
  }
  LDBG("rewrite ifOp: parentFor " << parentTCnts << " accumCnts");

  // else {
  //  Update both ifYieldOperands and elseYieldOperands.
  //  See below for an example of how to update yieldOp of IfA and IfB.
  //  ForA (accumForA, accumIfA, accumForB, accumIfB)
  //    IfA (accumIfA, accumForB)
  //      Channel A --> uses ForA.arg[accumIfA] to calculate (bufIdx, phase)
  //      ForB (accumForB)
  //        Channel B --> uses ForB.arg[accumForB]
  //    ThenYield ForA.arg[accumIfA] + 1, ForB.res[accumForB]
  //    ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
  //    ForC (accumForC, accumIfB)
  //      IfB
  //        Channel C --> uses ForC.arg[accumIfB]
  //      ThenYield ForC.arg[accumIfB] + 1
  //      ElseYield ForC.arg[accumIfB]
  //    Channel D --> uses ForA.arg[accumForA]
  //  Check to see if ifOp has a channel directly under. Both IfA and IfB fall
  //  into this case.
  DenseSet<Operation *> excludeReuse;
  excludeChannelsWithReuse(opsWithChannels, opsWithBufferReuse, excludeReuse);
  for (auto *op : excludeReuse) {
    if (newIfOp.getOperation() == op) {
      // Find enclosing Forop, use arg + 1; If no enclosing forOp, use 0.
      // arg is parentForOp.arg[newIfOp]
      Value endAccum, endAccumElse;
      if (parentForOp) {
        // Get corresponding argument of accumCnt: forOp.accumCnt[ifOp].
        unsigned accumArgId = getAccumArgIdx(parentForOp, op, opsWithChannels,
                                             opsWithBufferReuse);
        LDBG("rewrite ifOp: ifOp itself parentArg " << tSize << " "
                                                    << accumArgId);
        Value arg = parentForOp.getBody()->getArgument(tSize - parentTCnts +
                                                       accumArgId);
        Value one =
            ifBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
        endAccum =
            ifBuilder.createWithAsyncTaskIds<arith::AddIOp>(loc, arg, one);
        endAccumElse = arg;
      } else {
        endAccum =
            ifBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
        endAccumElse = elseBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(
            loc, 0, 64);
      }
      ifYieldOperands.push_back(endAccum);
      elseYieldOperands.push_back(endAccumElse);
      LLVM_DEBUG({
        LDBG("Update yieldOperands ");
        endAccum.dump();
        endAccumElse.dump();
      });
      break;
    }
  }
  // Go through ops in thenBlock, which should be preorder.
  for (auto *op : opList) {
    if (!enclosingAChannel(op, opsWithChannels))
      continue;
    // Push op.accumCnts as ifYield, push parentForOp.accumCnts[...] as
    // elseYield.
    SmallVector<Operation *> preOrderOps;
    getAccumCntsPreOrder(op, opsWithChannels, opsWithBufferReuse, preOrderOps);
    auto numRes = op->getNumResults();
    unsigned tCnts = preOrderOps.size();
    LDBG("rewrite ifOp: thenBlock " << tCnts << " accumCnts");
    unsigned accumArgId;
    if (parentForOp && preOrderOps.size() > 0)
      // arg is parentForOp.arg[preOrderOps[0]]
      accumArgId = getAccumArgIdx(parentForOp, preOrderOps[0], opsWithChannels,
                                  opsWithBufferReuse);
    for (unsigned i = 0; i < tCnts; ++i) {
      Value endAccum =
          op->getResult(numRes - tCnts + i - (hasBufferReuse ? 1 : 0));
      ifYieldOperands.push_back(endAccum);
      // Find the corresponding accumArgId from parentForOp.
      Value elseVal;
      if (parentForOp) {
        elseVal = parentForOp.getBody()->getArgument(tSize - parentTCnts +
                                                     accumArgId + i);
        LDBG("rewrite ifOp: elseYield parentArg " << tSize << " " << accumArgId
                                                  << " " << i);
      } else
        elseVal = elseBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(
            loc, 0, 64);
      elseYieldOperands.push_back(elseVal);
      LLVM_DEBUG({
        LDBG("Update yieldOperands ");
        endAccum.dump();
        elseVal.dump();
      });
    }
  }
  // Add one more operand to else Yield.
  if (hasBufferReuse) {
    ifYieldOperands.push_back(endAccumReuseThen);
    elseYieldOperands.push_back(endAccumReuseElse);
  }
  updateYield(newIfOp.thenYield(), ifYieldOperands);
  //}
  updateYield(newIfOp.elseYield(), elseYieldOperands);

  int resultIdx = 0;
  // Replace old if with the new one.
  for (auto result : ifOp.getResults()) {
    result.replaceAllUsesWith(newIfOp->getResult(resultIdx++));
  }

  // If ifOp is in opsWithBufferReuse, replace.
  auto tmpIter = std::find(opsWithBufferReuse.begin(), opsWithBufferReuse.end(),
                           ifOp.getOperation());
  if (tmpIter != opsWithBufferReuse.end()) {
    *tmpIter = newIfOp.getOperation();
  }

  ifOp.erase();
  return newIfOp;
}

Operation *SpecializeIfOp(scf::IfOp ifOp, IRMapping &mapping,
                          OpBuilderWithAsyncTaskIds &builder,
                          AsyncTaskId asyncTaskId) {
  LLVM_DEBUG({
    LDBG("specialize ifOp ");
    ifOp.dump();
  });

  // It is possible that we need to reduce the results. One example
  // is that the defining op for the yield operation is not for this
  // taskId and the defining op is not specialized, thus we should
  // remove the result.
  // We need to update the result types correctly here.
  unsigned resultIdx = 0;
  SmallVector<unsigned> keptResultVec;
  if (!ifOp->getResultTypes().empty()) {
    for (Value yieldV : ifOp.thenYield().getOperands()) {
      // Check the defining op for the corresponding result.
      if (Operation *def = yieldV.getDefiningOp()) {
        bool hasTaskId = hasAsyncTaskId(def, asyncTaskId);
        if (hasTaskId) {
          keptResultVec.push_back(resultIdx);
        }
      } else {
        assert(isa<BlockArgument>(yieldV) && "Unexpected yield value");
        auto bbArg = cast<BlockArgument>(yieldV);
        // Find transitive defining op for the block arg
        Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
        if (auto forOp = dyn_cast<scf::ForOp>(bbAargOwner)) {
          // track initial value
          auto initArg = forOp.getInitArgs()[bbArg.getArgNumber() - 1];
          if (Operation *def = initArg.getDefiningOp()) {
            if (hasAsyncTaskId(def, asyncTaskId))
              keptResultVec.push_back(resultIdx);
          } else {
            llvm_unreachable("Initial value should have a defining op");
          }
        } else {
          llvm_unreachable("Unexpected block argument owner");
        }
      }
      ++resultIdx;
    }
  }

  SmallVector<Type> newResultTypes;
  for (auto idx : keptResultVec) {
    newResultTypes.push_back(ifOp->getResultTypes()[idx]);
  }
  auto newIfOp = builder.createWithAsyncTaskIds<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, mapping.lookup(ifOp.getCondition()), true,
      ifOp.elseBlock());

  OpBuilderWithAsyncTaskIds ifBuilder(ifOp.getContext());
  ifBuilder.setAsynTaskIdsFromArray({asyncTaskId});

  // Handle thenRegion of this IfOp.
  ifBuilder.setInsertionPointToEnd(newIfOp.thenBlock());
  for (Operation &thenOp : ifOp.thenBlock()->getOperations()) {
    SpecializeOp(&thenOp, mapping, ifBuilder, asyncTaskId);
  }

  // Update yields
  auto updateYield = [&](scf::YieldOp yield, SmallVector<Value> &operands) {
    ifBuilder.setInsertionPoint(yield);
    ifBuilder.createWithAsyncTaskIds<scf::YieldOp>(yield.getLoc(), operands);
    yield.erase();
  };
  if (keptResultVec.size() < ifOp->getResultTypes().size()) {
    SmallVector<Value> ifYieldOperands;
    for (auto idx : keptResultVec) {
      ifYieldOperands.push_back(newIfOp.thenYield().getOperand(idx));
    }
    updateYield(newIfOp.thenYield(), ifYieldOperands);
  }

  // Handle elseRegion of the IfOp.
  if (ifOp.elseBlock()) {
    ifBuilder.setInsertionPointToEnd(newIfOp.elseBlock());
    for (Operation &elseOp : ifOp.elseBlock()->getOperations()) {
      SpecializeOp(&elseOp, mapping, ifBuilder, asyncTaskId);
    }
    if (keptResultVec.size() < ifOp->getResultTypes().size()) {
      SmallVector<Value> elseYieldOperands;
      for (auto idx : keptResultVec) {
        elseYieldOperands.push_back(newIfOp.elseYield().getOperand(idx));
      }
      updateYield(newIfOp.elseYield(), elseYieldOperands);
    }
  }

  unsigned newResIdx = 0;
  for (auto idx : keptResultVec) {
    mapping.map(ifOp.getResult(idx), newIfOp.getResult(newResIdx));
    ++newResIdx;
  }
  return newIfOp;
}

Operation *SpecializeForOp(scf::ForOp forOp, IRMapping &mapping,
                           OpBuilderWithAsyncTaskIds &builder,
                           AsyncTaskId asyncTaskId) {
  // Create newForOp for each task Id.
  auto usedArgs = collectBlockArgsForTask(forOp, asyncTaskId);

  // Prepare newLoopArgs.
  SmallVector<Value> newLoopArgs;
  for (unsigned argNumber : usedArgs) {
    auto arg = forOp.getInitArgs()[argNumber];
    auto newArg = mapping.lookupOrDefault(arg);
    assert(newArg && "Unexpected missing mapping");
    newLoopArgs.push_back(newArg);
  }

  // Prepare loop bounds.
  auto newLowerBound = mapping.lookupOrDefault(forOp.getLowerBound());
  auto newUpperBound = mapping.lookupOrDefault(forOp.getUpperBound());
  auto newStep = mapping.lookupOrDefault(forOp.getStep());

  // Create newForOp.
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      forOp.getLoc(), newLowerBound, newUpperBound, newStep, newLoopArgs);
  if (forOp->getAttr("tt.loop_schedule"))
    newForOp->setAttr("tt.loop_schedule", forOp->getAttr("tt.loop_schedule"));

  // Initialize Value mapping from forOp to newForOp
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < usedArgs.size(); ++i) {
    auto oldArg = forOp.getRegionIterArgs()[usedArgs[i]];
    auto newArg = newForOp.getRegionIterArgs()[i];
    mapping.map(oldArg, newArg);
  }

  // Recursively clone all operations with this asyncTaskId to newForOp.
  OpBuilderWithAsyncTaskIds forBuilder(forOp.getContext());
  forBuilder.setAsynTaskIdsFromArray({asyncTaskId});
  forBuilder.setInsertionPointToStart(newForOp.getBody());
  for (Operation &op : forOp.getBody()->without_terminator()) {
    SpecializeOp(&op, mapping, forBuilder, asyncTaskId);
  }

  // Create YieldOp for newForOp.
  auto yieldOp = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> newYieldOperands;
  for (unsigned i : usedArgs)
    newYieldOperands.push_back(mapping.lookup(yieldOp.getOperand(i)));

  bool createNewYield = true;
  if (newForOp.getBody()->mightHaveTerminator()) {
    auto initialYield =
        llvm::cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    if (newYieldOperands.size() == 0) {
      setAsyncTaskIds(initialYield, {asyncTaskId});
      createNewYield = false;
    }
  }
  if (createNewYield) {
    auto newYieldOp =
        forBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    setAsyncTaskIds(newYieldOp, {asyncTaskId});
  }

  // Replace results of forOp with results of newForOp.
  for (unsigned i = 0; i < usedArgs.size(); ++i) {
    auto oldResult = forOp.getResult(usedArgs[i]);
    auto newResult = newForOp.getResult(i);
    mapping.map(oldResult, newResult);
  }

  return newForOp;
}

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilderWithAsyncTaskIds &builder,
                        AsyncTaskId asyncTaskId) {
  auto taskIds = getAsyncTaskIds(op);
  // yieldOp are sometimes implict, meaning they do not necessarily have a task
  // id, but they should be shared by all async tasks.
  if (!hasAsyncTaskId(op, asyncTaskId) && !isa<scf::YieldOp>(op))
    return nullptr;

  if (op->getNumRegions() == 0) {
    Operation *newOp = builder.clone(*op, mapping);
    setAsyncTaskIds(newOp, asyncTaskId);
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      mapping.map(op->getResult(i), newOp->getResult(i));
    return newOp;
  } else {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return SpecializeIfOp(ifOp, mapping, builder, asyncTaskId);
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return SpecializeForOp(forOp, mapping, builder, asyncTaskId);
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      Operation *newOp = builder.clone(*op, mapping);
      // recursively set async task ids for child ops
      newOp->walk(
          [&](Operation *childOp) { setAsyncTaskIds(childOp, asyncTaskId); });
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
      return newOp;
    } else {
      llvm_unreachable("Unexpected Op with regions");
    }
  }

  return nullptr;
}

// Create IfOp for each ayncTaskId.
DenseMap<AsyncTaskId, scf::IfOp> SpecializeRegion(triton::FuncOp funcOp,
                                                  int regDecProducer,
                                                  int regIncConsumer) {

  LLVM_DEBUG({
    LDBG("\n\n");
    LDBG("Start specializing region");
  });

  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  auto loc = funcOp.getLoc();

  // Collect original operations
  SmallVector<Operation *> opList;
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &op : block.getOperations()) {
      auto taskIds = getAsyncTaskIds(&op);
      if (!taskIds.empty())
        opList.push_back(&op);
    }
  }

  LLVM_DEBUG({
    LDBG("ops to be specialized: ");
    for (Operation *op : opList) {
      op->dump();
    }
  });

  // Create GetAsyncTaskIdOp.
  Block *lastBlock = &funcOp.getBody().back();
  auto returnOp = llvm::cast<triton::ReturnOp>(lastBlock->getTerminator());
  builder.setInsertionPoint(returnOp);
  Value curAsyncTaskId = builder.create<ttng::GetAsyncTaskIdOp>(loc);

  DenseMap<AsyncTaskId, scf::IfOp> tasksToIfOp;

  // Clone all operations into the corresponding if blocks. If the operation
  // has multiple taskIds, it will be cloned for multiple if blocks.
  // If the original code has an IfOp, we should only clone its
  // body with the right asyncTaskId, instead of cloning the IfOp.
  for (AsyncTaskId asyncTaskId : getNestedAsyncTaskIds(funcOp)) {
    // Create IfOp for each asyncTaskId.
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, curAsyncTaskId,
        builder.create<arith::ConstantIntOp>(loc, asyncTaskId, 32));

    auto ifOp = builder.create<scf::IfOp>(loc, cond);
    tasksToIfOp[asyncTaskId] = ifOp;
    setAsyncTaskIds(ifOp, {asyncTaskId});

    OpBuilderWithAsyncTaskIds taskBuilder(context);
    taskBuilder.setAsynTaskIdsFromArray({asyncTaskId});

    // Set insertion point before yieldOp.
    auto yieldOp = ifOp.thenYield();
    setAsyncTaskIds(yieldOp, {asyncTaskId});
    taskBuilder.setInsertionPoint(yieldOp);

    IRMapping mapping;
    for (Operation *op : opList) {
      SpecializeOp(op, mapping, taskBuilder, asyncTaskId);
    }
  }

  // Decide if this taskId is a producer or a consumer, and create either
  // RegAllocOp or RegDeallocOp accordingly.
  for (auto ifOps : tasksToIfOp) {
    AsyncTaskId asyncTaskId = ifOps.first;
    auto ifOp = ifOps.second;
    OpBuilderWithAsyncTaskIds taskBuilder(ifOp.getContext());
    taskBuilder.setAsynTaskIdsFromArray({asyncTaskId});
    auto regAlloc = scanRegUsage(ifOp.thenBlock(), asyncTaskId, regDecProducer,
                                 regIncConsumer);
    taskBuilder.setInsertionPointToStart(&(ifOp.getThenRegion().front()));
    if (regAlloc.second)
      taskBuilder.create<ttng::RegAllocOp>(
          loc, taskBuilder.getI32IntegerAttr(regAlloc.first));
    else
      taskBuilder.create<ttng::RegDeallocOp>(
          loc, taskBuilder.getI32IntegerAttr(regAlloc.first));
  }

  LLVM_DEBUG({
    LDBG("\n\nWith task Id checks");
    funcOp.dump();
  });

  // Remove original operations that have been cloned in reverse order.
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    Operation *op = *it;
    LLVM_DEBUG({
      LDBG("erasing op ");
      op->dump();
    });
    // For debugging purposes, check to see if the original op is still in use.
    bool hasUse = false;
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      for (Operation *user : op->getResult(i).getUsers()) {
        hasUse = true;
        LLVM_DEBUG({
          LDBG("op has use ");
          user->dump();
        });
      }
    }
    op->erase();
  }
  return tasksToIfOp;
}

struct Channel {
public:
  using Relation = std::pair<int, SmallVector<int>>;

  Channel(int producer, SmallVector<int> &consumers, Operation *op,
          unsigned operandIdx, unsigned numBuffers)
      : relation(producer, consumers), op(op), operandIdx(operandIdx),
        numBuffers(numBuffers) {}

  bool operator==(const Channel &c) {
    return relation == c.relation && operandIdx == c.operandIdx && op == c.op;
  }

  Operation *getDstOp() { return op; }
  unsigned getDstOperandIdx() { return operandIdx; }
  Value getSrcOperand() { return op->getOperand(operandIdx); }
  Operation *getSrcOp() { return getSrcOperand().getDefiningOp(); }

  Relation relation; // producer task Id, a list of consumer task Ids
  Operation *op;
  unsigned operandIdx;
  unsigned numBuffers;
};

// Find transitive users of the root op. Track through control flow ops (such as
// yield) to get to the real users.
void getTransitiveUsers(Value root,
                        SetVector<std::pair<Operation *, unsigned>> &users) {
  for (Operation *userOp : root.getUsers()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(userOp)) {
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        if (operand.get() == root) {
          auto result =
              yieldOp->getParentOp()->getResult(operand.getOperandNumber());
          getTransitiveUsers(result, users);
        }
      }
    } else {
      // find operand index of root
      unsigned operandIndex = 0;
      for (OpOperand &operand : userOp->getOpOperands()) {
        if (operand.get() == root) {
          break;
        }
        operandIndex++;
      }
      assert(operandIndex < userOp->getNumOperands() &&
             "root is not an operand of userOp");
      users.insert({userOp, operandIndex});
    }
  }
}

// Loads will be in producer warp groups. For now, we only allow a single
// warp group/task for a producer. For each LoadOp, create a channel from it
// to any direct user which belongs to a different taskId.
void collectAsyncChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                          triton::FuncOp &funcOp, unsigned numBuffers) {
  funcOp.walk([&](Operation *op) {
    if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op) ||
        isa<mlir::triton::DotOpInterface>(op)) {
      auto producerTaskIds = getAsyncTaskIds(op);
      if (producerTaskIds.empty() || producerTaskIds.size() > 1) {
        LLVM_DEBUG({
          LDBG(" ignoring load ops without async task id or with multiple task "
               "ids: ");
          op->dump();
        });
        return;
      }
      auto producerTaskId = producerTaskIds.front();
      unsigned producerNumBuffers = numBuffers;
      if (auto forOp = op->getParentOfType<scf::ForOp>()) {
        producerNumBuffers = getNumBuffersOrDefault(forOp, numBuffers);
      }

      for (auto result : op->getResults()) {
        if (result.use_empty()) {
          continue;
        }

        SetVector<std::pair<Operation *, unsigned>> users;
        getTransitiveUsers(result, users);
        for (auto user : users) {
          auto userOp = user.first;
          auto consumerTaskIds = getAsyncTaskIds(userOp);
          if (consumerTaskIds.empty())
            continue;
          // Remove producer task id from consumerTaskIds.
          auto iter = std::remove(consumerTaskIds.begin(),
                                  consumerTaskIds.end(), producerTaskId);
          consumerTaskIds.erase(iter, consumerTaskIds.end());
          // Add a channel from the single producer task to consumerTaskIds.
          if (consumerTaskIds.size() > 0) {
            channels.push_back(std::make_unique<Channel>(
                producerTaskId, consumerTaskIds, userOp, user.second,
                producerNumBuffers));
          }
        }
      }
    }
  });

  LLVM_DEBUG({
    LDBG("Async channels:");
    for (auto &channel : channels) {
      LDBG("producer op: " << channel->relation.first);
      channel->getSrcOp()->dump();
      for (auto &asyncTaskId : channel->relation.second)
        LDBG("consumer: " << asyncTaskId);
      channel->getDstOp()->dump();
      LDBG("numBuffers: " << channel->numBuffers);
    }
  });
}

// Group channels in two ways:
//  - by producer ops. One producer corresponds to multiple channels. This
//    grouping will be used to create buffers per shared producer.
//  - by consumer ops. One consumer corresponds to multiple channels. This
//  grouping will be used to create barriers per shared consumer.
// Also compute orderedChannels, which will be keyed by getDstOp() of channels,
// to enforce deterministic order for map.
void groupChannels(
    SmallVector<Channel *> &channels,
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByProducers,
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByConsumers,
    SmallVector<Channel *> &orderedChannels) {

  // Group channels by producer op.
  DenseMap<Operation *, SmallVector<Channel *>> producerChannels;
  for (auto channel : channels) {
    producerChannels[channel->getSrcOp()].push_back(channel);
  }

#ifndef NDEBUG
  // Some sanity checks.
  for (auto &item : producerChannels) {
    auto &channels = item.second;
    unsigned numBuffers = channels.front()->numBuffers;
    for (auto c : channels) {
      assert(c->numBuffers == numBuffers && "Unmatched number of buffers");
    }
  }
#endif

  // Group channels by consumer op.
  DenseMap<Operation *, SmallVector<Channel *>> consumerChannels;

  // Two channels can be combined if
  //   src1 and src2 are in the same block and
  //   (dst1 == dst2 or
  //    (dst1 and dst2 are in the same block, both have a single user, and
  //     dst1User == dst2User and dst1User is in the same block as dst1))
  auto channelCanBeMerged = [](Channel *c1, Channel *c2) -> bool {
    // TODO: do not merge if one consumer is tcgen05_mma
    if (c1->getSrcOp()->getBlock() != c2->getSrcOp()->getBlock())
      return false;
    Operation *dst1 = c1->getDstOp(), *dst2 = c2->getDstOp();
    if (dst1 == dst2)
      return true;
    if (dst1->getBlock() != dst2->getBlock() || !dst1->hasOneUse() ||
        !dst2->hasOneUse())
      return false;
    // Check taskIds on dstOps.
    if (getAsyncTaskIds(dst1) != getAsyncTaskIds(dst2))
      return false;
    Operation *dst1User = *(dst1->getUsers().begin());
    Operation *dst2User = *(dst2->getUsers().begin());
    return dst1User == dst2User && dst1User->getBlock() == dst1->getBlock();
  };
  assert(channels.size() > 0 && "channel size is zero");
  // Compare with existing channels in the consumerChannels to see if
  // it can be combined.
  for (auto *c0 : channels) {
    bool merged = false;
    for (auto &kv : consumerChannels) {
      if (kv.second.size() > 0 && channelCanBeMerged(c0, kv.second.front())) {
        kv.second.push_back(c0);
        merged = true;
        break;
      }
    }
    if (!merged) { // Create a new entry.
      auto *keyOp = c0->getDstOp();
      if (!consumerChannels.count(keyOp))
        orderedChannels.push_back(c0);
      consumerChannels[keyOp].push_back(c0);
    }
  }

  // Reorder channels associated with one entry based on program order of the
  // producers.
  for (auto &kv : consumerChannels) {
    if (kv.second.size() > 1) {
      auto &allOps = kv.second.front()->getSrcOp()->getBlock()->getOperations();
      std::sort(
          kv.second.begin(), kv.second.end(), [&](Channel *a, Channel *b) {
            auto itrA =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == a->getSrcOp();
                });
            auto itrB =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == b->getSrcOp();
                });
            assert(itrA != allOps.end() && itrB != allOps.end());
            return std::distance(itrA, itrB) < 0;
          });
    }
  }

  // Switch to using channel as the key instead of ops as ops can be volatile.
  for (auto &kv : producerChannels) {
    channelsGroupedByProducers[kv.second.front()] = kv.second;
  }
  for (auto &kv : consumerChannels) {
    channelsGroupedByConsumers[kv.second.front()] = kv.second;
  }

  LLVM_DEBUG({
    DBGS() << "\n\n";
    LDBG("Grouped channels by producer:");
    unsigned i = 0;
    for (auto &kv : channelsGroupedByProducers) {
      DBGS() << "Channel  " << ++i << ":\n";
      DBGS() << "producer:  ";
      kv.getFirst()->getSrcOp()->dump();
      for (auto &channel : kv.second) {
        DBGS() << "consumer: ";
        channel->getDstOp()->dump();
        DBGS() << "] ";
        LDBG("numBuffers: " << channel->numBuffers);
        DBGS() << "\n";
      }
    }

    DBGS() << "\n\n";
    LDBG("Grouped channels by consumer:");
    i = 0;
    for (auto &kv : channelsGroupedByConsumers) {
      DBGS() << "Channel  " << ++i << ":\n";
      DBGS() << "consumer:  ";
      kv.getFirst()->getDstOp()->dump();
      for (auto &channel : kv.second) {
        DBGS() << "producer: ";
        channel->getSrcOp()->dump();
        for (auto &asyncTaskId : channel->relation.second)
          DBGS() << asyncTaskId << ", ";
        DBGS() << "] ";
        LDBG("numBuffers: " << channel->numBuffers);
        DBGS() << "\n";
      }
      DBGS() << "\n";
    }
  });
}

// Reorder producer ops to unblock consumers interleavingly.
void reorderProducerOps(SmallVector<Channel *> &channels) {
  if (channels.size() <= 1)
    return;

  // Bail out if channels are not in the same block
  auto block = channels.front()->getSrcOp()->getBlock();
  for (auto &channel : channels) {
    if (channel->getSrcOp()->getBlock() != block) {
      return;
    }
  }

  // Group channels by the first consumer taskId of each channel. Smaller taskId
  // has higher priority.
  // TODO: consider consumer priority
  std::map<AsyncTaskId, SmallVector<Channel *>> groupedProducerOps;
  for (auto &channel : channels) {
    auto asyncTaskId = channel->relation.second.front();
    groupedProducerOps[asyncTaskId].push_back(channel);
  }

  // No need to reorder if all channels are in the same group.
  if (groupedProducerOps.size() <= 1)
    return;

  // Sort each group by number of consumers.
  for (auto &group : groupedProducerOps) {
    std::sort(group.second.begin(), group.second.end(),
              [&](Channel *a, Channel *b) {
                return a->relation.second.size() < b->relation.second.size();
              });
  }

  // Start from the first producer in channels. Iterate through the groups
  // which are ordered by the first consumer taskId. Within each group, channels
  // are ordered by number of consumers.
  Operation *currOp = channels.front()->getSrcOp();
  for (auto &group : groupedProducerOps) {
    for (auto &channel : group.second) {
      channel->getSrcOp()->moveAfter(currOp);
      currOp = channel->getSrcOp();
    }
  }

  // Move backward dependency slice close to producer ops.
  // Start from the last producer op backwards and move backward slice to
  // before each op. This guarantees that the backward slice of each op is
  // scheduled as late as possible.
  for (auto &group : reverse(groupedProducerOps)) {
    for (auto &channel : reverse(group.second)) {
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      SetVector<Operation *> backwardSlice;
      getBackwardSlice(channel->getSrcOp(), &backwardSlice, opt);
      for (auto &op : backwardSlice) {
        if (op->getBlock() == block)
          op->moveBefore(channel->getSrcOp());
      }
    }
  }

  LLVM_DEBUG({
    LDBG("\n");
    LDBG("after reordering producer ops");
    currOp->getParentOfType<triton::FuncOp>().dump();
    LDBG("\n");
  });
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

// Generate code
//   numSteps = ((upperBound - lowerBound) + forOpStep - 1) / forOpStep
Value getNumSteps(scf::ForOp forOp, OpBuilderWithAsyncTaskIds &builder) {
  auto loc = forOp.getLoc();
  // numSteps = ((upperBound - lowerBound) + forOpStep - 1) / forOpStep
  Value numSteps = builder.createWithAsyncTaskIds<arith::SubIOp>(
      loc, forOp.getUpperBound(), forOp.getLowerBound());
  numSteps = builder.createWithAsyncTaskIds<arith::AddIOp>(loc, numSteps,
                                                           forOp.getStep());
  if (forOp.getStep().getType() != builder.getI64Type())
    numSteps = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
        loc, builder.getI64Type(), numSteps);

  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
  numSteps = builder.createWithAsyncTaskIds<arith::SubIOp>(loc, numSteps, one);
  Value innerForStep = forOp.getStep();
  if (forOp.getStep().getType() != builder.getI64Type())
    innerForStep = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
        loc, builder.getI64Type(), forOp.getStep());
  numSteps = builder.createWithAsyncTaskIds<arith::DivUIOp>(loc, numSteps,
                                                            innerForStep);
  return numSteps;
}

// When hasParallelReuse is true (i.e this is the innermost loop), we pass in
// accumulatedLoopCount, which is used to initialize initBufferIdx.
// When isOuterOfReuse is true, we add an additional arg for accumLoopCount.
scf::ForOp createNewLoop(scf::ForOp forOp, int numBuffers,
                         scf::ForOp &parentForOp,
                         SmallVector<Value> &initialAccum,
                         Value accumulatedLoopCount, bool hasParallelReuse,
                         bool isOuterOfReuse) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();

  OpBuilderWithAsyncTaskIds builder(forOp.getContext());
  builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(forOp));
  builder.setInsertionPoint(forOp);
  if (hasParallelReuse) {
    LLVM_DEBUG({
      LDBG("createNewLoop hasParallelReuse: ");
      accumulatedLoopCount.dump();
    });
  }

  // This doesn't include the accumCnt for reuse.
  unsigned numAccumCnts = initialAccum.size();

  // Step 1: Append accumCnts as forOp arguments.
  // With reuse, either isOuterOfReuse or hasParallelReuse is true.
  bool isBufferReuse = isOuterOfReuse || hasParallelReuse;
  // else {
  for (unsigned i = 0; i < numAccumCnts; i++)
    body->insertArgument(body->getNumArguments(), builder.getI64Type(), loc);
  //}
  // With reuse, the loops will get an additional accumCnt at the end.
  Value tmpAccumLoopCount;
  if (isBufferReuse) {
    // Add accumCnt for inner loops and outer loop.
    tmpAccumLoopCount = body->insertArgument(body->getNumArguments(),
                                             builder.getI64Type(), loc);
  }
  auto yieldOp = llvm::cast<scf::YieldOp>(body->getTerminator());
  builder.setInsertionPoint(yieldOp);
  // Step 3: Add accumCnts to yieldOp.
  // else {
  unsigned tSize = body->getNumArguments();
  // Will be fixed in the caller.
  for (unsigned i = 0; i < numAccumCnts; i++)
    yieldOp->insertOperands(yieldOp.getNumOperands(),
                            {body->getArgument(tSize - numAccumCnts + i -
                                               (isBufferReuse ? 1 : 0))});
  //}
  if (isOuterOfReuse) {
    // We have not iterated through the body yet, so do not have the right value
    // for nextTmpIdx. This will be fixed in the caller.
    yieldOp->insertOperands(yieldOp.getNumOperands(),
                            {tmpAccumLoopCount /*, nextPhase, nextBufferIdx*/});
  } else if (hasParallelReuse) {
    // Increment by 1.
    builder.setInsertionPoint(yieldOp);
    Value one =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
    Value nextCntIdx = builder.createWithAsyncTaskIds<arith::AddIOp>(
        loc, tmpAccumLoopCount, one);
    yieldOp->insertOperands(yieldOp.getNumOperands(),
                            {nextCntIdx /*, nextPhase, nextBufferIdx*/});
  }

  // Step 4: Create loop arguments for the new ForOp.
  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getInitArgs())
    newLoopArgs.push_back(operand);

  builder.setInsertionPoint(forOp);
  Value initCntIdx;
  for (unsigned i = 0; i < numAccumCnts; i++) {
    initCntIdx = initialAccum[i];
    newLoopArgs.append({initCntIdx /*, initPhase, initBufferIdx*/});
  }
  if (isBufferReuse) {
    if (hasParallelReuse) { // inner loops
      initCntIdx = accumulatedLoopCount;
    } else { // outer loop
      initCntIdx =
          builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
    }
    newLoopArgs.append({initCntIdx /*, initPhase, initBufferIdx*/});
  }

  // Step 5: Create newForOp and take the region of the original forOp.
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      newLoopArgs);
  if (forOp->getAttr("tt.loop_schedule"))
    newForOp->setAttr("tt.loop_schedule", forOp->getAttr("tt.loop_schedule"));
  newForOp.getRegion().takeBody(forOp.getRegion());

  // Step 6: Replace forOp with newForOp.
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  return newForOp;
}

// Find top-level ops which contain at least one channel. If a channel's
// getSrcOp() and getDstOp() belong to the inner loop, the outer loop will be
// part of asyncTaskOps.
SmallVector<Operation *>
getTaskTopRegion(triton::FuncOp funcOp,
                 const SmallVector<Channel *> &channels) {
  SmallVector<Operation *> asyncTaskOps;
  auto isAsyncTaskTopOp = [&](Operation *taskTopOp) -> bool {
    for (auto c : channels) {
      Operation *producer = c->getSrcOp(), *consumer = c->getDstOp();
      while (producer && !isa<triton::FuncOp>(producer->getParentOp())) {
        producer = producer->getParentOp();
      }
      while (consumer && !isa<triton::FuncOp>(consumer->getParentOp())) {
        consumer = consumer->getParentOp();
      }
      if (producer == taskTopOp && consumer == taskTopOp)
        return true;
    }
    return false;
  };
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &bodyOp : block.getOperations()) {
      Operation *op = &bodyOp;
      if (op->getNumRegions() <= 0)
        continue;
      // If this op does not contain both a producer taskId and a consumer
      // taskId, continue.
      if (getAsyncTaskIds(op).size() == 1)
        continue;
      if (isAsyncTaskTopOp(op))
        asyncTaskOps.push_back(op);
    }
  }

  LLVM_DEBUG({
    LDBG("\nTop Task Bodies");
    for (auto op : asyncTaskOps) {
      LDBG("\nTask Body:");
      op->dump();
    }
  });
  return asyncTaskOps;
}

static unsigned getNumChannelsInOp(Operation *op,
                                   const SmallVector<Channel *> &channels,
                                   SmallVector<Channel *> &channelsInOp) {
  unsigned num = 0;
  for (auto *ch : channels) {
    // Get the immediate parent.
    auto srcParent = ch->getSrcOp()->getParentOp();
    auto dstParent = ch->getDstOp()->getParentOp();
    if (srcParent == op && dstParent == op)
      channelsInOp.push_back(ch);
  }
  return channelsInOp.size();
}

void reuseBuffers(SmallVector<Operation *> &taskTopOps,
                  const SmallVector<Channel *> &channels,
                  DenseMap<Channel *, Channel *> &mapToRepresenting,
                  SmallVector<Operation *> &opsWithBufferReuse) {
  // For the case of multiple parallel ForOps with same number of channels,
  // we can try reusing the buffers across the parallel ForOps or across ForOps
  // and IfOps. Case 1:
  //   ForOp_A
  //   ForOp_B
  // --> opsWithBufferReuse: ForOp_A ForOp_B
  // Case 2:
  //   ForOp (persistent)
  //     ForOp_A
  //     ForOp_B
  // --> opsWithBufferReuse: ForOp_A ForOp_B
  // Case 3:
  //   ForOp (persistent)
  //     ForOp_A
  // --> --> opsWithBufferReuse: ForOp_A
  // Case 4:
  //   ForOp
  //   IfOp
  // --> opsWithBufferReuse: ForOp IfOp
  // We use accumLoopCount to update bufferIdx for the sharing groups. If there
  // is an outer loop, we will need to add an argument to it. Assume we handle
  // outer ForOp first, then inner ForOp in program order.
  unsigned maxDepth = 0;
  DenseMap<unsigned, SmallVector<Operation *>> loopDepthMap;
  for (auto &op : taskTopOps) {
    op->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
      if (dyn_cast<scf::ForOp>(subOp) || dyn_cast<scf::IfOp>(subOp)) {
        unsigned tDepth = getLoopDepth(subOp);
        loopDepthMap[tDepth].push_back(subOp);
        if (tDepth > maxDepth)
          maxDepth = tDepth;
      }
    });
  }
  // A list of IfOps/ForOps at the innermost level: loopDepthMap[maxDepth]
  auto &opsAtMaxDepth = loopDepthMap[maxDepth];
  LDBG("reuseBuffers number of inner ops: " << opsAtMaxDepth.size()
                                            << " at depth " << maxDepth);
  if (opsAtMaxDepth.empty() || opsAtMaxDepth.size() == 1)
    return;
  // Find ops that contain immediate channels. And the ops do not overlap
  // live range. For example
  // If
  //   For
  // --> If and For can overlap. But
  // For
  // If
  // --> can't overlap
  SmallVector<Operation *> innerOps;
  SmallVector<Operation *> innerLoops;
  for (auto *innerOp : opsAtMaxDepth) {
    SmallVector<Channel *> channelsInOp;
    getNumChannelsInOp(innerOp, channels, channelsInOp);
    if (channelsInOp.empty())
      continue;
    innerOps.push_back(innerOp);
    if (dyn_cast<scf::ForOp>(innerOp))
      innerLoops.push_back(innerOp);
  }
  // Make sure opsWithBufferReuse are under the same ForOp or at the top level.
  // Make sure opsWithBufferReuse contain the same number of channels, and the
  // same numBuffers for the channels. Channels in the first op will be the
  // representing channels. All sharing groups will span the same set of regions
  // in opsWithBufferReuse.
  bool firstOp = true;
  Operation *outerLoop = nullptr;
  unsigned numChannels = 0, numBuffers = 0;
  SmallVector<Channel *> channelsInOpOne;
  for (auto *innerOp : innerOps) {
    // Ignore IfOps that overlap with innerLoops.
    if (dyn_cast<scf::IfOp>(innerOp)) {
      bool ignore = false;
      for (auto *innerLoop : innerLoops) {
        if (innerOp == innerLoop->getParentOp()) {
          ignore = true;
          break;
        }
      }
      if (ignore)
        continue;
    }
    scf::ForOp parentForOp = innerOp->getParentOfType<scf::ForOp>();
    SmallVector<Channel *> channelsInOp;
    getNumChannelsInOp(innerOp, channels, channelsInOp);
    if (firstOp) {
      outerLoop = parentForOp.getOperation();
      numChannels = channelsInOp.size();
      channelsInOpOne = channelsInOp;
      numBuffers = channelsInOp[0]->numBuffers;
      opsWithBufferReuse.push_back(innerOp);
    } else {
      if (outerLoop != parentForOp.getOperation() ||
          numChannels != channelsInOp.size())
        // Not under the same outer loop.
        return;
      if (numBuffers != channelsInOp[0]->numBuffers)
        return;
      unsigned idx = 0;
      for (auto *ch : channelsInOp) {
        // TODO: sort the channels in the loop according to buffer size.
        mapToRepresenting[ch] = channelsInOpOne[idx++];
      }
      opsWithBufferReuse.push_back(innerOp);
    }
    firstOp = false;
  }
  if (opsWithBufferReuse.size() == 1)
    // A single op in buffer reuse and there is no outer loop.
    opsWithBufferReuse.clear();
  LLVM_DEBUG({
    LDBG("reuseBuffers: " << numChannels << " channels opsWithBufferReuse "
                          << opsWithBufferReuse.size());
    for (auto &kv : mapToRepresenting) {
      llvm::dbgs() << "---- from ";
      kv.first->getDstOp()->dump();
      llvm::dbgs() << "---- to ";
      kv.second->getDstOp()->dump();
    }
  });
  // opsWithBufferReuse = innerOps;
}

// Helper function to get a list of control Ops for which we need
// accumCnt. We go through all channels and find the enclosing controlOp X.
// For the case of
// ForA (accumForA, accumIfA, accumForB, accumIfB)
//   IfA (accumIfA, accumForB)
//     Channel A --> uses ForA.arg[accumIfA] to calculate (bufIdx, phase)
//     ForB (accumForB)
//       Channel B --> uses ForB.arg[accumForB]
//   ThenYield ForA.arg[accumIfA] + 1, ForB.res[accumForB]
//   ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
//   ForC (accumForC, accumIfB)
//     IfB
//       Channel C --> uses ForC.arg[accumIfB]
//     ThenYield ForC.arg[accumIfB] + 1
//     ElseYield ForC.arg[accumIfB]
//   Channel D --> uses ForA.arg[accumForA]
// opsWithChannels: ForA, IfA, ForB, IfB
// We start with ForA, then traverses IfA, ForB, ForC, IfB
void updateAccumRegions(SmallVector<Operation *> &opList,
                        const SmallVector<Channel *> &channels,
                        DenseSet<Operation *> &opsWithChannels) {
  for (auto *ch : channels) {
    auto *dst = ch->getDstOp();
    auto *pOp = dst->getParentOp();
    if (!pOp)
      continue;
    if (auto forOp = dyn_cast<scf::ForOp>(pOp))
      opsWithChannels.insert(pOp);
    if (auto ifOp = dyn_cast<scf::IfOp>(pOp))
      opsWithChannels.insert(pOp);
  }
}

// Go through a list of operations under one scope.
// prevAccum can be null if there is an outer loop for the reuse loops.
// -- prevAccum: for buffer reuse, opsWithBufferReuse.size() > 1
Value updateAccumLoopCount(SmallVector<Operation *> &opList,
                           unsigned numBuffers,
                           SmallVector<Operation *> &taskTopOps,
                           Operation *commonOuterLoop,
                           SmallVector<Operation *> &opsWithBufferReuse,
                           DenseSet<Operation *> &opsWithChannels,
                           Value prevAccum) {
  DenseMap<Operation *, Operation *> oldToNew;
  for (Operation *op : opList) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto newForOp =
          createNewLoopWrapper(forOp, numBuffers, taskTopOps, commonOuterLoop,
                               opsWithBufferReuse, opsWithChannels, prevAccum);
      oldToNew[op] = newForOp.getOperation();
      // Update prevAccum to be after the loop.
      // If the loop is in opsWithBufferReuse, generate prevAccum + numSteps.
      bool hasReuse = false;
      for (auto tLoop : opsWithBufferReuse)
        if (newForOp.getOperation() == tLoop) {
          hasReuse = true;
          break;
        }
      if (hasReuse && opsWithBufferReuse.size() > 1) {
        // Update accumLoopCount = prevAccum + numSteps.
        OpBuilderWithAsyncTaskIds builder(newForOp.getContext());
        builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(newForOp));
        builder.setInsertionPointAfter(newForOp);

        Value numSteps = getNumSteps(newForOp, builder);
        prevAccum = builder.createWithAsyncTaskIds<arith::AddIOp>(
            newForOp.getLoc(), prevAccum, numSteps);
      }
      // If the loop is the outer loop for a reuse loop, we are done.
      // At this point, op is no longer valid.
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (needAccumulatedLoopCnt(ifOp, opsWithBufferReuse, opsWithChannels)) {
        auto newIfOp =
            rewriteIfOp(ifOp, numBuffers, taskTopOps, commonOuterLoop,
                        opsWithBufferReuse, opsWithChannels, prevAccum);
        oldToNew[op] = newIfOp.getOperation();
        // update prevAccum to be result of the new IfOp.
        assert(newIfOp.getNumResults() >= 1);
        auto numRes = newIfOp.getNumResults();
        LDBG("update prevAccum with result from IfOp");
        prevAccum = newIfOp.getResult(numRes - 1); // last result
      } else {
        // Still need to process ForOps in pre-order.
        SmallVector<scf::ForOp> innerForOps;
        ifOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
          if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
            innerForOps.push_back(forOp);
          }
        });
        for (auto innerFor : innerForOps) {
          auto newFor = createNewLoopWrapper(
              innerFor, numBuffers, taskTopOps, commonOuterLoop,
              opsWithBufferReuse, opsWithChannels, prevAccum);
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
  return prevAccum;
}

scf::ForOp createNewLoopWrapper(scf::ForOp origForOp, unsigned numBuffers,
                                SmallVector<Operation *> &taskTopOps,
                                Operation *commonOuterLoop,
                                SmallVector<Operation *> &opsWithBufferReuse,
                                DenseSet<Operation *> &opsWithChannels,
                                Value prevAccum) {
  LLVM_DEBUG({
    LDBG("call createNewLoop on");
    origForOp.dump();
  });

  scf::ForOp parentForOp = origForOp->getParentOfType<scf::ForOp>();
  scf::ForOp newForOp;
  // for(...) -> for(..., phase, bufferIdx)
  unsigned loopNumBuffers = getNumBuffersOrDefault(origForOp, numBuffers);

  bool isOuterOfReuse = false;
  bool hasReuse = false;
  if (opsWithBufferReuse.size() > 1) {
    for (auto tLoop : opsWithBufferReuse)
      if (origForOp.getOperation() == tLoop) {
        hasReuse = true;
        break;
      }
    isOuterOfReuse =
        commonOuterLoop && commonOuterLoop == origForOp.getOperation();
  }
  // Set accumulatedLoopCount when this is a loop in opsWithBufferReuse. If
  // this loop has an outer loop, an extra arg for accumLoopCount should have
  // been added to the outer loop.
  Value accumulatedLoopCount = prevAccum;
  // In the case of no reuse, ForOp will have a list of accumCnts, starting with
  // argument value.
  // Get initial value of accumCnts prior to the loop.
  SmallVector<Value> initialAccum;
  unsigned tSize = 0, tNum = 0, accumArgId = 0;
  if (parentForOp) {
    tSize = parentForOp.getBody()->getArguments().size();
    tNum = getAccumCnts(parentForOp.getOperation(), opsWithChannels,
                        opsWithBufferReuse);
    LDBG("-- has parentForOp");
  }
  SmallVector<Operation *> preOrderOps;
  getAccumCntsPreOrder(origForOp.getOperation(), opsWithChannels,
                       opsWithBufferReuse, preOrderOps);
  unsigned tCnts = preOrderOps.size();
  if (preOrderOps.size() > 0 && parentForOp) {
    // Check for accumArgId in parentForOp for the first preOrderOp of the
    // ForOp.
    accumArgId = getAccumArgIdx(parentForOp, preOrderOps[0], opsWithChannels,
                                opsWithBufferReuse);
  }
  LDBG("-- isOuterOfReuse, hasReuse, tSizeForParent, tNumForParent, "
       "preOrderOps: "
       << isOuterOfReuse << " " << hasReuse << " " << tSize << " " << tNum
       << " " << tCnts);
  // Handle the case of no buffer reuse.
  for (unsigned i = 0; i < tCnts; ++i) {
    Value startAccum;
    if (parentForOp)
      startAccum =
          parentForOp.getBody()->getArgument(tSize - tNum + accumArgId + i);
    else {
      OpBuilderWithAsyncTaskIds builder(origForOp->getContext());
      builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(origForOp));
      builder.setInsertionPoint(origForOp);
      auto loc = origForOp.getLoc();
      startAccum =
          builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
    }
    initialAccum.push_back(startAccum);
  }

  newForOp = createNewLoop(origForOp, loopNumBuffers, parentForOp, initialAccum,
                           accumulatedLoopCount, hasReuse, isOuterOfReuse);
  LLVM_DEBUG({
    LDBG("after createNewLoop ");
    newForOp.dump();
  });
  // origForOp is erased in createNewLoop. If origForOp is a top operation
  // (i.e in taskTopOps), make sure taskTopOps is updated with the newForOp.
  auto asyncTaskLoopForItr =
      std::find(taskTopOps.begin(), taskTopOps.end(), origForOp.getOperation());
  if (asyncTaskLoopForItr != taskTopOps.end()) {
    // Update taskTopOps.
    *asyncTaskLoopForItr = newForOp.getOperation();
  }

  // origForOp is erased in createNewLoop. If origForOp is in
  // opsWithBufferReuse, replace.
  auto tmpIter = std::find(opsWithBufferReuse.begin(), opsWithBufferReuse.end(),
                           origForOp.getOperation());
  if (tmpIter != opsWithBufferReuse.end()) {
    *tmpIter = newForOp.getOperation();
  }
  // opsWithChannels
  auto tmpIter3 = std::find(opsWithChannels.begin(), opsWithChannels.end(),
                            origForOp.getOperation());
  if (tmpIter3 != opsWithChannels.end()) {
    LDBG("createNewLoopWrapper: update opsWithChannels "
         << origForOp.getOperation() << " --> " << newForOp.getOperation());
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
  Value endAccum = updateAccumLoopCount(
      opList, numBuffers, taskTopOps, commonOuterLoop, opsWithBufferReuse,
      opsWithChannels,
      isOuterOfReuse ? getReuseAccumCntArg(newForOp) : prevAccum);
  LLVM_DEBUG({
    LDBG("-- before replacing yieldOp ");
    newForOp.dump();
  });

  // Update yieldOp.
  if (isOuterOfReuse) {
    Value arg = getReuseAccumCntArg(newForOp);
    Operation *yieldOp = newForOp.getBody()->getTerminator();
    yieldOp->replaceUsesOfWith(arg, endAccum);
  }
  // else {
  Operation *yieldOp = newForOp.getBody()->getTerminator();
  // ForA (accumForA, accumIfA, accumForB, accumIfB)
  //   IfA (accumIfA, accumForB)
  //     Channel A --> uses ForA.arg[accumIfA] to calculate (bufIdx, phase)
  //     ForB (accumForB)
  //       Channel B --> uses ForB.arg[accumForB]
  //   ThenYield ForA.arg[accumIfA] + 1, ForB.res[accumForB]
  //   ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
  //   ForC (accumForC, accumIfB)
  //     IfB
  //       Channel C --> uses ForC.arg[accumIfB]
  //     ThenYield ForC.arg[accumIfB] + 1
  //     ElseYield ForC.arg[accumIfB]
  //   Channel D --> uses ForA.arg[accumForA]
  // Yield ForA.arg[accumForA]+1, IfA.res[accumIfA], IfA.res[accumForB],
  // ForC.res[accumIfB]
  tSize = newForOp.getBody()->getArguments().size();
  auto numAccumCnts = initialAccum.size();
  if (numAccumCnts == 0)
    return newForOp;
  if (isOuterOfReuse || hasReuse)
    numAccumCnts++;
  accumArgId = tSize - numAccumCnts; // first accumCnt: ForA.arg[accumForA]
  LDBG("-- tSize, numAccumCnts, accumArgId " << tSize << " " << numAccumCnts
                                             << " " << accumArgId);

  // If there is a channel directly in forOp, yield ForA.arg[accumForA]+1.
  DenseSet<Operation *> excludeReuse;
  excludeChannelsWithReuse(opsWithChannels, opsWithBufferReuse, excludeReuse);
  for (auto *op : excludeReuse) {
    if (newForOp.getOperation() == op) {
      Value arg = newForOp.getBody()->getArgument(accumArgId);
      OpBuilderWithAsyncTaskIds builder(newForOp->getContext());
      builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(newForOp));
      builder.setInsertionPoint(yieldOp);
      auto loc = newForOp.getLoc();
      Value one =
          builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
      Value endAccum =
          builder.createWithAsyncTaskIds<arith::AddIOp>(loc, arg, one);
      yieldOp->replaceUsesOfWith(arg, endAccum);
      ++accumArgId;
      break;
    }
  }
  LDBG("-- accumArgId after channels directly under " << accumArgId);
  // This order should align with the preorder that is used for accumCnts.
  SmallVector<Operation *> dummy;
  for (auto *op : opList) {
    if (!enclosingAChannel(op, excludeReuse))
      continue;
    auto numRes = op->getNumResults();
    // Ignore reuse AccumCnt here as it is handled earlier.
    unsigned tCnts = getAccumCnts(op, excludeReuse, dummy);
    // For now, we only supported limited form of reuse where we have double
    // loop nests.
    bool hasReuseCnt = false;
    if (auto tIf = dyn_cast<scf::IfOp>(op))
      hasReuseCnt = needAccumulatedLoopCntForReuse(tIf, opsWithBufferReuse);
    LDBG("-- hasReuseCnt, numRes, tCnts, accumArgId "
         << hasReuseCnt << " " << numRes << " " << tCnts << " " << accumArgId);
    for (unsigned i = 0; i < tCnts; ++i) {
      Value arg = newForOp.getBody()->getArgument(accumArgId);
      Value endAccum =
          op->getResult(numRes - tCnts + i - (hasReuseCnt ? 1 : 0));
      LLVM_DEBUG({
        LDBG("-- replace use of arg with result "
             << numRes - tCnts + i - (hasReuseCnt ? 1 : 0));
        op->dump();
      });
      yieldOp->replaceUsesOfWith(arg, endAccum);
      LLVM_DEBUG(yieldOp->dump());
      ++accumArgId;
    }
  }
  LLVM_DEBUG({
    LDBG("-- after all replacing ");
    newForOp.dump();
  });
  //}
  return newForOp;
}

// This function takes
// -- channels: a list of channels
// -- mapToRepresenting: a mapping from a channel to its representing channel if
// the channel shares smem space with the representing channel
// -- opsWithBufferReuse: a list of control ops that are sharing smem spaces.
// Note that every loop in opsWithBufferReuse either has the same outer loop or
// has no outer loop.
// We call updateAccumLoopCount on the list of top level Ops that are control
// ops (ForOps or IfOps). updateAccumLoopCount calls createNewLoopWrapper on
// ForOps, and rewriteIfOp on IfOps. Both will call updateAccumLoopCount on the
// list of Ops in the ForOp body or the thenBlock, elseBlock for IfOp.
// createNewLoopWrapper will create a new ForOp by adding phase,
// bufferIdx, and a list of accumLoopCnt to the arguments.
// In the case of sharing smem or persistent, we need to traverse and update
// IfOps via rewriteIfOp, when necessary.
Value appendBufferIdxArgs(
    SmallVector<Operation *> &taskTopOps, unsigned numBuffers,
    const SmallVector<Channel *> &channels,
    const DenseMap<Channel *, Channel *> &mapToRepresenting,
    SmallVector<Operation *> &opsWithBufferReuse,
    DenseSet<Operation *> &opsWithChannels) {
  // In order to handle sharing smem for a list of loops, we have two cases,
  // one is the top-level op containing all loops in opsWithBufferReuse is
  // a ForOp.
  bool genAccumLoopCount = !opsWithBufferReuse.empty();
  Operation *commonOuterLoop = nullptr;
  if (genAccumLoopCount) {
    auto oneFor = opsWithBufferReuse[0];
    scf::ForOp parentForOp = oneFor->getParentOfType<scf::ForOp>();
    if (parentForOp)
      commonOuterLoop = parentForOp.getOperation();
  }

  // When there is no outer loop, we need to create a place holder for
  // tmpAccumLoopCount. Every forOp in opsWithBufferReuse either has the same
  // outer loop or has no outer loop.
  Value tmpAccumLoopCount;
  if (opsWithBufferReuse.size() > 1 && !commonOuterLoop) {
    auto oneFor = opsWithBufferReuse[0];
    // Initialize tmpAccumLoopCount to be 0.
    OpBuilderWithAsyncTaskIds builder(taskTopOps[0]->getContext());
    builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(oneFor));
    builder.setInsertionPoint(taskTopOps[0]);
    tmpAccumLoopCount = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        oneFor->getLoc(), 0, 64);
  }

  SmallVector<Operation *> opList;
  for (auto &op : taskTopOps) {
    if (auto origIfOp = dyn_cast<scf::IfOp>(op)) {
      opList.push_back(op);
    }
    if (auto origForOp = dyn_cast<scf::ForOp>(op))
      opList.push_back(op);
  }
  updateAccumLoopCount(opList, numBuffers, taskTopOps, commonOuterLoop,
                       opsWithBufferReuse, opsWithChannels, tmpAccumLoopCount);

  return tmpAccumLoopCount;
}

// Create an allocation to hold the mbarriers.
static Value createBarrierAlloc(triton::FuncOp funcOp, unsigned distance) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(funcOp.getContext());
  Location loc = funcOp.getLoc();
  auto context = funcOp.getContext();
  auto barrierCTALayout =
      ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                              /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
      context, 1, 1, 1, {0}, barrierCTALayout);
  Type barrierMemDescType = ttg::MemDescType::get(
      {distance}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Type singleBarrierMemDescType =
      ttg::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loc, barrierMemDescType, Value());
  for (unsigned i = 0; i < distance; i++) {
    Value idx = builder.create<arith::ConstantIntOp>(loc, i, 32);
    Value barrierView = builder.create<ttg::MemDescSubviewOp>(
        loc, singleBarrierMemDescType, barrierAlloc, idx);
    builder.create<ttng::InitBarrierOp>(funcOp->getLoc(), barrierView, 1);
  }
  return barrierAlloc;
}

struct CommChannel {
  DenseMap<int, Value> tokens;
  // Producer barrier is only needed when the producer op itself can update the
  // barrier inline, such as the TMA load.
  std::optional<Value> producerBarrier;
  // Consumer barrier is only needed when the consumer op itself can update the
  // barrier inline, such as the TCGen5MMAOp.
  std::optional<Value> consumerBarrier;
};

static Operation *getRealConsumer(Operation *consumerOp) {
  // For a consumer of local_alloc, probe a step further to see if the
  // real consumer is a TCGen5MMAOp.
  if (isa<LocalAllocOp>(consumerOp)) {
    int userCount = 0;
    Operation *transitiveConsumer;
    for (auto user : consumerOp->getUsers()) {
      transitiveConsumer = user;
      userCount++;
      if (userCount > 1)
        break;
    }

    if (userCount == 1) {
      if (isa<nvidia_gpu::TCGen5MMAOp, nvidia_gpu::WarpGroupDotOp>(
              transitiveConsumer)) {
        return transitiveConsumer;
      }
    }
  }

  return consumerOp;
}

// channelsGroupedByConsumers: channels are grouped together.
// Go through each group, check the first channel in the group, create a token
// for each consumer taskId. Return a map that maps each channel + consumer
// taskId to a token. Also update barrierAllocMap that maps each channel +
// consumer taskId to a BarrierAlloc.
void createToken(
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByConsumers,
    const SmallVector<Channel *> &orderedChannels, triton::FuncOp funcOp,
    int numConsumerGroups,
    const DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseMap<Channel *, SmallVector<Channel *>> &channelReuse,
    DenseMap<Channel *, CommChannel> &tokenMap) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  for (auto *key : orderedChannels) {
    auto it = channelsGroupedByConsumers.find(key);
    Channel *channel = it->second.front();
    if (!channelReuse.count(channel))
      continue;

    CommChannel commChannel;
    auto producerOp = it->second.front()->getSrcOp();
    auto consumerOp = it->second.front()->getDstOp();
    consumerOp = getRealConsumer(consumerOp);

    if (isa<tt::ExperimentalDescriptorLoadOp>(producerOp)) {
      commChannel.producerBarrier =
          createBarrierAlloc(funcOp, channel->numBuffers);
    }

    for (auto consumerAsyncTaskId : channel->relation.second) {
      // No token is needed for a TMA <-> TCGen5MMAOp channel
      if (!isa<tt::ExperimentalDescriptorLoadOp>(producerOp) ||
          !isa<nvidia_gpu::TCGen5MMAOp>(consumerOp)) {
        ttng::TokenLoadType tokenLoadType;
        auto copyOp = copyOpMap.find(channel)->second.first;
        if (isa<ttg::AsyncCopyGlobalToLocalOp>(copyOp)) {
          tokenLoadType = ttng::TokenLoadType::AsyncLoadOp;
        } else if (isa<ExperimentalDescriptorLoadOp>(copyOp)) {
          tokenLoadType = ttng::TokenLoadType::TMALoadOp;
        } else if (isa<LocalStoreOp>(copyOp)) {
          tokenLoadType = ttng::TokenLoadType::LocalStoreOp;
        } else {
          llvm_unreachable("Unexpected load type");
        }
        Value v;
        if (it->second.front()->getSrcOp()->getParentOfType<scf::ForOp>())
          v = builder.create<ttng::CreateTokenOp>(
              funcOp.getLoc(), channel->numBuffers, tokenLoadType);
        else
          v = builder.create<ttng::CreateTokenOp>(funcOp.getLoc(), 1,
                                                  tokenLoadType);
        commChannel.tokens[consumerAsyncTaskId] = v;
      }

      if (isa<nvidia_gpu::TCGen5MMAOp>(consumerOp)) {
        Value v = createBarrierAlloc(funcOp, channel->numBuffers);
        commChannel.consumerBarrier = v;
      }
    }

    // Channels in the group share the same set of tokens.
    for (auto &c : it->second) {
      tokenMap[c] = commChannel;
    }
    for (auto *reuse : channelReuse[channel]) {
      tokenMap[reuse] = commChannel;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Communication Channels: \n";
    for (auto &item : tokenMap) {
      llvm::dbgs() << "\ndata channel: \n";
      llvm::dbgs() << *item.first->getSrcOp() << "\n";
      llvm::dbgs() << *item.first->getDstOp() << "\n";
      llvm::dbgs() << "communication channel: \n";
      for (auto &kv : item.second.tokens) {
        llvm::dbgs() << "token: " << kv.first << " " << kv.second << "\n";
      }
      if (item.second.producerBarrier)
        llvm::dbgs() << "producer barrier: " << *item.second.producerBarrier
                     << "\n";
      if (item.second.consumerBarrier)
        llvm::dbgs() << "consumer barrier: " << *item.second.consumerBarrier
                     << "\n";
    }
  });
}

// Create a buffer array for each producer op, if the producer is in a ForOp,
// the buffer array will contain numBuffers.
DenseMap<Channel *, Value> createBuffer(
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByProducers,
    triton::FuncOp funcOp, int numConsumerGroups,
    DenseMap<Channel *, Channel *> &mapToRepresenting,
    DenseMap<Channel *, SmallVector<Channel *>> &channelReuse) {

  DenseMap<Channel *, Value> bufferMap;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  DenseSet<Channel *> visited;
  for (auto &item : channelsGroupedByProducers) {
    auto &channels = item.second;
    for (auto c : channels) {
      assert(!visited.count(c));
      visited.insert(c);
      if (mapToRepresenting.count(c)) {
        channelReuse[mapToRepresenting[c]].push_back(c);
        LDBG("update channelReuse key " << mapToRepresenting[c] << " " << c);
      } else {
        channelReuse[c].push_back(c);
        LDBG("update channelReuse key " << c << " " << c);
      }
    }
  }
  for (auto &item : channelsGroupedByProducers) {
    auto &channels = item.second;
    auto srcValue = item.first->getSrcOperand();
    auto srcOp = item.first->getSrcOp();
    unsigned numBuffers = channels.front()->numBuffers;

    if (auto tensorType = dyn_cast<RankedTensorType>(srcValue.getType())) {
      // Get basic information from tensorType
      auto order = ttg::getOrder(tensorType);
      auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
      auto elemType = tensorType.getElementType();

      // Get shape, layout and type of a slice
      auto sliceShape = tensorType.getShape();
      auto sharedLayout = ttg::NVMMASharedEncodingAttr::get(
          context, sliceShape, order, CTALayout, elemType, /*fp4Padded*/ false);
      auto sliceType =
          RankedTensorType::get(sliceShape, elemType, sharedLayout);

      // Get shape, layout and type of the complete buffer
      SmallVector<int64_t> bufferShape(sliceShape.begin(), sliceShape.end());
      if (srcOp->getParentOfType<scf::ForOp>())
        bufferShape.insert(bufferShape.begin(), numBuffers);
      else
        bufferShape.insert(bufferShape.begin(), 1);
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(context);
      auto bufferType =
          RankedTensorType::get(bufferShape, elemType, sharedLayout);
      Type memdescType =
          ttg::MemDescType::get(bufferShape, elemType, sharedLayout,
                                sharedMemorySpace, /*mutableMemory*/ true);
      Value buffer =
          builder.create<ttg::LocalAllocOp>(funcOp.getLoc(), memdescType);

      // Channels in the group share the same buffer.
      for (auto c : channels)
        bufferMap[c] = buffer;
    } else {
      llvm_unreachable("Unexpected result type");
    }
  }
  unsigned groupId = 0;
  for (auto &kv : channelReuse) {
    if (kv.second.size() <= 1)
      continue;
    bufferMap[kv.first].getDefiningOp()->setAttr(
        "allocation.shareGroup",
        IntegerAttr::get(IntegerType::get(context, 32), groupId));
    for (auto *c : kv.second)
      bufferMap[c].getDefiningOp()->setAttr(
          "allocation.shareGroup",
          IntegerAttr::get(IntegerType::get(context, 32), groupId));
    ++groupId;
  }
  return bufferMap;
}

static std::pair<Operation *, Operation *>
createAsyncCopy(const DenseMap<Channel *, Value> &bufferMap, Channel *c,
                Operation *op, SmallVector<AsyncTaskId> &asyncTasksPC,
                Value bufferIdx, Value bufferIdxExtract) {
  auto loadOp = cast<triton::LoadOp>(op);
  auto buffer = bufferMap.find(c)->second;
  MLIRContext *context = loadOp->getContext();
  OpBuilderWithAsyncTaskIds builder(context);
  builder.setInsertionPoint(loadOp->getParentOp());
  builder.setAsynTaskIdsFromArray(asyncTasksPC);

  builder.setInsertionPoint(loadOp);
  Value loadResult = loadOp.getResult();
  auto tensorType = dyn_cast<RankedTensorType>(loadResult.getType());
  if (!tensorType)
    return {nullptr, nullptr};
  // Get basic information from tensorType
  auto order = ttg::getOrder(tensorType);
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout = ttg::NVMMASharedEncodingAttr::get(
      context, sliceShape, order, CTALayout, elemType, /*fp4Padded*/ false);
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType subviewTy =
      ttg::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                            sliceType.getEncoding(), sharedMemorySpace,
                            /*mutableMemory=*/true);
  Value zero = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
      loadOp.getLoc(), 0, 32);
  SmallVector<Value> copyOffsets(sliceType.getRank() + 1, zero);
  copyOffsets[0] = bufferIdx;
  builder.setAsyncTaskIdsFromOp(loadOp);
  builder.setInsertionPointAfter(loadOp);
  auto view = builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      loadOp.getLoc(), subviewTy, buffer, copyOffsets);
  // Create cp.async
  Operation *copy =
      builder.createWithAsyncTaskIds<ttg::AsyncCopyGlobalToLocalOp>(
          loadOp.getLoc(), loadOp.getPtr(), view, loadOp.getMask(),
          loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());

  // Extract part.
  builder.setAsyncTaskIdsFromValueUsers(loadResult);
  builder.setInsertionPoint(c->getDstOp());
  SmallVector<Value> loadOffsets(sliceType.getRank() + 1, zero);
  loadOffsets[0] = bufferIdxExtract;
  auto viewLoad = builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      loadOp.getLoc(), subviewTy, buffer, loadOffsets);
  auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
      loadOp.getLoc(), loadOp.getType(), viewLoad /*,wait->getResult(0)*/);
  // Replace all uses of loadResult
  loadResult.replaceAllUsesWith(sharedLoad.getResult());
  loadOp.erase();
  return {copy, sharedLoad};
}

// Create a local copy for a channel that is populated by the producer and
// accessed by the consumer.
static std::pair<Operation *, Operation *>
createLocalCopy(const DenseMap<Channel *, Value> &bufferMap, Channel *channel,
                Value srcBufferIdx, Value dstBufferIdx) {
  Operation *srcOp = channel->getSrcOp();
  Operation *dstOp = channel->getDstOp();
  MLIRContext *context = srcOp->getContext();
  auto buffer = bufferMap.find(channel)->second;

  Value srcValue = channel->getSrcOperand();
  auto tensorType = dyn_cast<RankedTensorType>(srcValue.getType());
  if (!tensorType)
    return {nullptr, nullptr};
  // Get basic information from tensorType
  auto order = ttg::getOrder(tensorType);
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout = ttg::NVMMASharedEncodingAttr::get(
      context, sliceShape, order, CTALayout, elemType, /*fp4Padded*/ false);
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType subviewTy =
      ttg::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                            sliceType.getEncoding(), sharedMemorySpace,
                            /*mutableMemory=*/true);

  // Consumer part.
  OpBuilderWithAsyncTaskIds builder(dstOp);
  builder.setAsyncTaskIdsFromOp(dstOp);
  builder.setInsertionPoint(dstOp);
  Value zero = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
      dstOp->getLoc(), 0, 32);
  SmallVector<Value> loadOffsets(sliceType.getRank() + 1, zero);
  loadOffsets[0] = dstBufferIdx;
  auto dstView = builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      dstOp->getLoc(), subviewTy, buffer, loadOffsets);
  auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
      dstOp->getLoc(), srcValue.getType(), dstView);
  srcValue.replaceAllUsesWith(sharedLoad.getResult());

  // Producer part. Create local_store for new producers.
  builder.setAsynTaskIdsFromArray(channel->relation.first);
  builder.setInsertionPoint(srcOp->getParentOp());
  zero = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(srcOp->getLoc(),
                                                              0, 32);
  SmallVector<Value> storeOffsets(sliceType.getRank() + 1, zero);
  storeOffsets[0] = srcBufferIdx;
  builder.setInsertionPointAfter(srcOp);
  auto srcView = builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      srcOp->getLoc(), subviewTy, buffer, storeOffsets);
  // Create local_alloc
  Operation *copy = builder.createWithAsyncTaskIds<ttg::LocalStoreOp>(
      srcOp->getLoc(), srcValue, srcView);
  return {copy, sharedLoad};
}

static int getTMALoadSize(tt::ExperimentalDescriptorLoadOp &tmaLoad) {
  auto tensorTy = cast<RankedTensorType>(tmaLoad->getResult(0).getType());
  int loadSize = product(tensorTy.getShape());
  return loadSize * tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
}

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx) {
  auto context = barrierAlloc.getContext();
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType barrierTy = ttg::MemDescType::get(
      {1}, builder.getI64Type(),
      cast<ttg::MemDescType>(barrierAlloc.getType()).getEncoding(),
      sharedMemorySpace,
      /*mutableMemory=*/true);

  // Create barrierForTMA from barrierAlloc.
  return builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      barrierAlloc.getLoc(), barrierTy, barrierAlloc,
      ArrayRef<Value>({bufferIdx}));
}

Value getBufferForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                Type loadType, Value buffer, Value bufferIdx,
                                bool mutableMem) {
  auto context = buffer.getContext();
  auto tensorType = dyn_cast<RankedTensorType>(loadType);
  assert(tensorType);

  auto order = ttg::getOrder(tensorType);
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout = ttg::NVMMASharedEncodingAttr::get(
      context, sliceShape, order, CTALayout, elemType, /*fp4Padded*/ false);
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType subviewTy =
      ttg::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                            sliceType.getEncoding(), sharedMemorySpace,
                            /*mutableMemOry=*/mutableMem);

  Value zero = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
      buffer.getLoc(), 0, 32);
  SmallVector<Value> copyOffsets(sliceType.getRank() + 1, zero);
  copyOffsets[0] = bufferIdx;

  return builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      buffer.getLoc(), subviewTy, buffer, copyOffsets);
}

Operation *
optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                 SmallVector<tt::ExperimentalDescriptorLoadOp> &tmaLoads,
                 SmallVector<Value> &buffers, Value barrierAlloc,
                 Value bufferIdx, Value bufferIdxExtract, Value phase,
                 Operation *headProducer, Operation *headConsumer) {
  auto loc = barrierAlloc.getLoc();

  // Compute the total size of the loads.
  int sizeInBytes = 0;
  for (auto &tmaLoad : tmaLoads) {
    sizeInBytes += getTMALoadSize(tmaLoad);
  }

  // For each of the following ops, we will operate on a subview of each value
  // according to the pipeline stage.

  // Create a barrier_expect with the appropriate size and insert it before the
  // first load.
  builder.setInsertionPoint(headProducer);
  builder.setAsyncTaskIdsFromOp(headProducer);
  auto prodBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  auto expect = builder.createWithAsyncTaskIds<ttng::BarrierExpectOp>(
      loc, prodBarrier, sizeInBytes, pred);

  // Convert all the producers to async_tma_copy_global_to_local
  Operation *copy = nullptr;
  for (auto [tmaLoad, buffer] : zip(tmaLoads, buffers)) {
    builder.setInsertionPoint(tmaLoad);
    auto pipelineBuffer = getBufferForPipelineStage(builder, tmaLoad.getType(),
                                                    buffer, bufferIdx, true);
    Value tmaPtr =
        builder
            .createWithAsyncTaskIds<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
                loc, tmaLoad.getDesc());
    copy = builder.createWithAsyncTaskIds<ttng::AsyncTMACopyGlobalToLocalOp>(
        loc, tmaPtr, tmaLoad.getIndices(), prodBarrier, pipelineBuffer, pred);
  }

  // Create a wait_barrier before the first consumer.
  builder.setInsertionPoint(headConsumer);
  builder.setAsyncTaskIdsFromOp(headConsumer);
  auto consBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdxExtract);
  phase = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
      loc, builder.getI32Type(), phase);
  auto wait = builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
      loc, consBarrier, phase);

  // Convert all the consumers to local_load
  for (auto [tmaLoad, buffer] : zip(tmaLoads, buffers)) {
    auto pipelineBuffer = getBufferForPipelineStage(
        builder, tmaLoad.getType(), buffer, bufferIdxExtract, false);
    auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
        loc, tmaLoad.getType(), pipelineBuffer);

    Value loadResult = tmaLoad.getResult();
    tmaLoad.getResult().replaceAllUsesWith(sharedLoad.getResult());
    tmaLoad.erase();
  }
  return copy;
}

void desyncTCGen5MMAOp(OpBuilderWithAsyncTaskIds &builder,
                       nvidia_gpu::TCGen5MMAOp mmaOp, Value barrierAlloc,
                       Value bufferIdx, Value phase, unsigned numBuffers,
                       Operation *headProducer,
                       DenseSet<Operation *> &opsWithChannels,
                       SmallVector<Operation *> &opsWithBufferReuse) {
  // attach the barrier as an operand of the mma op.
  builder.setInsertionPoint(mmaOp);
  builder.setAsyncTaskIdsFromOp(mmaOp);
  auto consumerBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  mmaOp.getBarrierMutable().assign(consumerBarrier);

  // Create a wait_barrier before the producer.
  builder.setInsertionPoint(headProducer);
  builder.setAsyncTaskIdsFromOp(headProducer);
  auto producerBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  // curPhase = curPhase xor True for emptyBarrier.
  auto loc = headProducer->getLoc();
  Value _1_1b = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  phase =
      builder.createWithAsyncTaskIds<mlir::arith::XOrIOp>(loc, phase, _1_1b);
  phase = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
      loc, builder.getI32Type(), phase);
  builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(loc, producerBarrier,
                                                      phase);

  // Create a wait_barrier before the tmem load.
  SetVector<std::pair<Operation *, unsigned>> users;
  getTransitiveUsers(mmaOp.getD(), users);
  for (auto item : users) {
    auto user = item.first;
    if (user == mmaOp)
      continue;
    builder.setInsertionPoint(user);
    builder.setAsyncTaskIdsFromOp(user);
    // If user and mmaOp are in the same block, we can use the same barrier.
    if (user->getBlock() != mmaOp->getBlock()) {
      // Compute the barrier from the last consumer instance
      // Extract the accum count from the consumer block.
      std::tie(bufferIdx, phase) = getOutOfScopeBufferIdxAndPhase(
          builder, mmaOp, numBuffers, opsWithChannels, opsWithBufferReuse);
      phase = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
          user->getLoc(), builder.getI32Type(), phase);
      consumerBarrier =
          getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
    }

    builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(user->getLoc(),
                                                        consumerBarrier, phase);
  }
}

// Lower producers for channels. Here channels are grouped in
// "channelsGroupedByConsumers". tokenMap tracks the set of tokens for each
// channel.
void insertAsyncComm(
    triton::FuncOp funcOp,
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByConsumers,
    const DenseMap<Channel *, CommChannel> &tokenMap,
    const DenseMap<Channel *, DenseMap<int, Value>> &barrierAllocMap,
    const DenseMap<Channel *, Value> &bufferMap,
    const DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    int numConsumerGroups, DenseSet<Operation *> &opsWithChannels,
    SmallVector<Operation *> &opsWithBufferReuse) {

  // Find the operation that is along producer's parent chain, and its parent
  // is the same op as producer's parent. Here p is producer, and c is consumer.
  auto getSameLevelOp = [](Operation *p, Operation *c) -> Operation * {
    while (!isa<triton::FuncOp>(c)) {
      if (c->getParentOp() == p->getParentOp()) {
        return c;
      }
      c = c->getParentOp();
    }
    llvm_unreachable("Failed to find consumer's same level Op with producer");
  };

  auto consumerReleaseHeuristic = [&](Operation *p, Operation *c,
                                      int consumerAsyncTaskId) -> Operation * {
    if (c->getBlock() != p->getBlock())
      return getSameLevelOp(p, c);

    // Find a common place for all users of the consumer, which would be the
    // common post dominator.
    mlir::PostDominanceInfo dom(funcOp);
    std::unordered_set<Operation *> mutuallyNonDominatingUsers;
    SmallVector<Operation *> users;
    for (auto user : c->getUsers()) {
      if (isa<TransOp, MemDescTransOp>(user)) {
        // TransOp is not a real consumer. It caculates the shared memory
        // address for the real consumer. Continue to find its transitive users
        // recursively.
        DenseSet<Operation *> visited;
        SmallVector<Operation *> transUsers;
        transUsers.push_back(user);
        while (!transUsers.empty()) {
          auto transUser = transUsers.pop_back_val();
          visited.insert(transUser);
          if (isa<TransOp, MemDescTransOp>(transUser)) {
            for (auto transitiveUser : transUser->getUsers()) {
              if (!visited.count(transitiveUser))
                transUsers.push_back(transitiveUser);
            }
          } else {
            users.push_back(transUser);
          }
        }
      } else {
        users.push_back(user);
      }
    }

    for (auto user : users) {
      auto it = mutuallyNonDominatingUsers.begin();
      while (it != mutuallyNonDominatingUsers.end()) {
        if (dom.properlyPostDominates(user, *it)) {
          it = mutuallyNonDominatingUsers.erase(it);
        } else if (dom.properlyPostDominates(*it, user)) {
          break;
        } else {
          ++it;
        }
      }
      if (it == mutuallyNonDominatingUsers.end())
        mutuallyNonDominatingUsers.insert(user);
    }

    if (mutuallyNonDominatingUsers.size() == 1) {
      // Find the common parent of this user and c
      auto user = *mutuallyNonDominatingUsers.begin();
      while (user && user->getParentOp() != c->getParentOp())
        user = user->getParentOp();
      assert(user && "Failed to find common parent of this user and c");
      return user;
    }

    for (auto &op : reverse(c->getBlock()->getOperations())) {
      auto asyncTasks = getAsyncTaskIds(&op);
      if (asyncTasks.size() == 1 && asyncTasks[0] == consumerAsyncTaskId)
        return &op;
    }

    return nullptr;
  };

  // Go through each channel group.
  for (auto kv : channelsGroupedByConsumers) {
    // Find head and tail ops.
    DenseSet<Operation *> producerOps;
    DenseSet<Operation *> consumerOps;
    for (auto &c : kv.second) {
      auto pcOp = copyOpMap.find(c)->second;
      producerOps.insert(pcOp.first);
      consumerOps.insert(pcOp.second);
      consumerOps.insert(c->getDstOp());
      consumerOps.insert(getRealConsumer(c->getDstOp()));
    }

    // Find head producer
    auto producerBlock = kv.second.front()->getSrcOp()->getBlock();
    Operation *headProducer = nullptr;
    for (auto &op : producerBlock->getOperations()) {
      if (producerOps.count(&op)) {
        headProducer = &op;
        break;
      }
    }
    // Find tail producer
    Operation *tailProducer = nullptr;
    for (auto &op : reverse(producerBlock->getOperations())) {
      if (producerOps.count(&op)) {
        tailProducer = &op;
        break;
      }
    }

    // Find head consumer and tail consumer
    auto consumerBlock = kv.second.front()->getDstOp()->getBlock();
    Operation *headConsumer = nullptr;
    for (auto &op : consumerBlock->getOperations()) {
      if (consumerOps.count(&op)) {
        headConsumer = &op;
        break;
      }
    }
    Operation *tailConsumer = nullptr;
    for (auto &op : reverse(consumerBlock->getOperations())) {
      if (consumerOps.count(&op)) {
        tailConsumer = &op;
        break;
      }
    }

    // We have one set of tokens for each channel group.
    auto &commChannel = tokenMap.find(kv.second.front())->second;
    auto masterChannel = kv.getFirst();

    SmallVector<AsyncTaskId> asyncTaskP;
    asyncTaskP.push_back(masterChannel->relation.first);
    SmallVector<AsyncTaskId> &asyncTaskC = masterChannel->relation.second;
    SmallVector<AsyncTaskId> asyncTasksPC = asyncTaskP;
    asyncTasksPC.insert(asyncTasksPC.end(), asyncTaskC.begin(),
                        asyncTaskC.end());

    OpBuilderWithAsyncTaskIds builder(headProducer->getContext());
    if (auto funcOp = dyn_cast<triton::FuncOp>(headProducer->getParentOp())) {
      builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    } else {
      builder.setInsertionPoint(headProducer->getParentOp());
    }
    builder.setAsynTaskIdsFromArray(asyncTasksPC);

    Value bufferIdx;
    Value phase = Value();
    if (auto forOp = headProducer->getParentOfType<scf::ForOp>()) {
      builder.setInsertionPoint(headProducer);
      LLVM_DEBUG({
        LDBG("call getBufferIdxAndPhase2 ");
        headProducer->dump();
      });
      getBufferIdxAndPhase(builder, headProducer, kv.second.front()->numBuffers,
                           opsWithChannels, bufferIdx, phase,
                           opsWithBufferReuse);
    } else {
      // Producer is not in a ForOp, create phase and bufferIdx here.
      bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 32);
      phase = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 1);
    }

    builder.setAsynTaskIdsFromArray(masterChannel->relation.first);
    for (const auto &token : commChannel.tokens) {
      if (!commChannel.consumerBarrier) {
        // Insert ProducerAcquireOp before the producer.
        builder.setInsertionPoint(headProducer);
        builder.createWithAsyncTaskIds<ttng::ProducerAcquireOp>(
            headProducer->getLoc(), token.second, bufferIdx, phase);
      }

      // Insert ProducerCommitOp if producer is not TMA. For TMA, TMA lowering
      // will handle the ProducerCommit.
      if (!commChannel.producerBarrier) {
        builder.setInsertionPointAfter(tailProducer);
        builder.createWithAsyncTaskIds<ttng::ProducerCommitOp>(
            tailProducer->getLoc(), token.second, bufferIdx);
      }
    }

    for (const auto &token : commChannel.tokens) {
      builder.setAsynTaskIdsFromArray(token.first);
      // Insert ConsumerWaitOp
      if (!commChannel.producerBarrier) {
        auto consumerWaitPoint = getSameLevelOp(headProducer, headConsumer);
        builder.setInsertionPoint(consumerWaitPoint);
        builder.createWithAsyncTaskIds<ttng::ConsumerWaitOp>(
            headConsumer->getLoc(), token.second, bufferIdx, phase);
      }

      // Insert ConsumerReleaseOp, if consumer is not a TCGen5MMAOp. For
      // TCGen5MMAOp, TCGen5MMAOp lowering will handle the ConsumerReleaseOp.
      if (!isa<nvidia_gpu::TCGen5MMAOp>(tailConsumer)) {
        auto consumerReleasePoint =
            consumerReleaseHeuristic(tailProducer, tailConsumer, token.first);
        builder.setInsertionPointAfter(consumerReleasePoint);
        builder.createWithAsyncTaskIds<ttng::ConsumerReleaseOp>(
            consumerReleasePoint->getLoc(), token.second, bufferIdx);
      }
    }

    SmallVector<tt::ExperimentalDescriptorLoadOp> tmaLoads;
    SmallVector<Value> buffers;
    DenseMap<Operation *, Operation *> producerCopyMap;
    // Go through all channels in this channel group.
    for (auto &c : kv.second) {
      if (auto tmaLoad =
              dyn_cast<tt::ExperimentalDescriptorLoadOp>(c->getSrcOp())) {
        tmaLoads.push_back(tmaLoad);
        buffers.push_back(bufferMap.find(c)->second);
      }
    }

    // Desynchronize TCGen5MMAOp. Set up consumer release and producer acquire.
    auto &c = kv.first;
    if (auto mmaOp =
            dyn_cast<nvidia_gpu::TCGen5MMAOp>(getRealConsumer(c->getDstOp()))) {
      desyncTCGen5MMAOp(builder, mmaOp, *commChannel.consumerBarrier, bufferIdx,
                        phase, c->numBuffers, headProducer, opsWithChannels,
                        opsWithBufferReuse);
    }

    // Optimize TMA loads.
    if (tmaLoads.size() > 0) {
      optimizeTMALoads(builder, tmaLoads, buffers, *commChannel.producerBarrier,
                       bufferIdx, bufferIdx, phase, headProducer, headConsumer);
    }
  }
}

// Lower producers for channels. Here channels are grouped in
// "channelsGroupedByProducers"
void insertAsyncCopy(
    triton::FuncOp funcOp,
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByProducers,
    const DenseMap<Channel *, Value> &bufferMap,
    DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseSet<Operation *> &opsWithChannels,
    SmallVector<Operation *> &opsWithBufferReuse) {
  // For each producer op, create a async_copy or local_store from the producer
  // to the buffer. Create a local_load from the buffer at the dominating
  // consumer.
  mlir::DominanceInfo dom(funcOp);

  for (auto kv : channelsGroupedByProducers) {
    // Finding the dominating channel if possible.
    std::unordered_set<Channel *> mutuallyNonDominatingChannels;
    for (auto &c : kv.second) {
      // check if c is dominating all other previous channels.
      auto it = mutuallyNonDominatingChannels.begin();
      while (it != mutuallyNonDominatingChannels.end()) {
        auto channel = *it;
        if (dom.properlyDominates(c->getDstOp(), channel->getDstOp())) {
          it = mutuallyNonDominatingChannels.erase(it);
        } else if (dom.properlyDominates(channel->getDstOp(), c->getDstOp())) {
          break;
        } else {
          ++it;
        }
      }
      if (it == mutuallyNonDominatingChannels.end())
        mutuallyNonDominatingChannels.insert(c);
    }

    auto srcOp = kv.getFirst()->getSrcOp();
    Value bufferIdx;
    Value phase = Value();
    OpBuilderWithAsyncTaskIds builder(srcOp);
    SmallVector<AsyncTaskId> asyncTasksPC = getAsyncTaskIds(srcOp);
    for (auto channel : mutuallyNonDominatingChannels)
      asyncTasksPC.append(getAsyncTaskIds(channel->getDstOp()));
    builder.setAsynTaskIdsFromArray(asyncTasksPC);
    if (auto forOp = srcOp->getParentOfType<scf::ForOp>()) {
      LLVM_DEBUG({
        LDBG("call getBufferIdxAndPhase ");
        srcOp->dump();
      });
      getBufferIdxAndPhase(builder, srcOp, kv.getFirst()->numBuffers,
                           opsWithChannels, bufferIdx, phase,
                           opsWithBufferReuse);
    } else {
      // Producer is not in a ForOp, create phase and bufferIdx here which will
      // be used by both producer and consumers.
      bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          srcOp->getLoc(), 0, 32);
    }

    assert(mutuallyNonDominatingChannels.size() == 1 &&
           "conditional consumers not supported");

    auto domininatingChannel = *mutuallyNonDominatingChannels.begin();
    std::pair<Operation *, Operation *> producerConsumerOps{nullptr, nullptr};

    // No need to create async copy for TMA load which will be handled in
    // insertAsyncComm.
    if (isa<tt::ExperimentalDescriptorLoadOp>(srcOp)) {
      producerConsumerOps = {srcOp, domininatingChannel->getDstOp()};
    } else if (isa<triton::LoadOp>(srcOp)) {
      SmallVector<AsyncTaskId> asyncTasksPC = getAsyncTaskIds(srcOp);
      asyncTasksPC.append(getAsyncTaskIds(domininatingChannel->getDstOp()));
      // After createAsyncCopy, c->getSrcOp()/headProducer are no longer
      // valid.
      producerConsumerOps = createAsyncCopy(bufferMap, domininatingChannel,
                                            domininatingChannel->getSrcOp(),
                                            asyncTasksPC, bufferIdx, bufferIdx);
    } else {
      assert(!isa<ttg::LocalLoadOp>(srcOp) &&
             "LocalLoadOp buffer should be reused");
      producerConsumerOps =
          createLocalCopy(bufferMap, domininatingChannel, bufferIdx, bufferIdx);
    }

    for (auto &channel : kv.second) {
      copyOpMap[channel] = producerConsumerOps;
    }
  }
}

void foldLocalLoads(triton::FuncOp funcOp) {
  // If loadResult has a single use which is LocalAlloc, we can get rid of
  // sharedLoad and replace all uses of LocalAlloc with viewLoad.
  DenseMap<Operation *, Value> opsToReplace;
  funcOp.walk([&](ttg::LocalAllocOp localAlloc) {
    if (auto src = localAlloc.getSrc()) {
      if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(src.getDefiningOp())) {
        // Only fold within the same tasks
        if (getAsyncTaskIds(localLoad) == getAsyncTaskIds(localAlloc)) {
          opsToReplace[localAlloc] = localLoad.getSrc();
        }
      }
    }
  });
  OpBuilderWithAsyncTaskIds builder(funcOp.getContext());
  for (auto kv : opsToReplace)
    replaceUsesAndPropagateType(builder, kv.getFirst(), kv.getSecond());
}

class TritonGPUWSCodePartitionPass
    : public impl::TritonGPUWSCodePartitionBase<TritonGPUWSCodePartitionPass> {
public:
  using impl::TritonGPUWSCodePartitionBase<
      TritonGPUWSCodePartitionPass>::TritonGPUWSCodePartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    // Disable code partitioning when numBuffers is 0.
    if (numBuffers == 0)
      return;

    // Step 1: collect all communications between producers and consumers.
    SmallVector<std::unique_ptr<Channel>> channelsOrigin;
    collectAsyncChannels(channelsOrigin, funcOp, numBuffers);
    SmallVector<Channel *> channels;
    for (const auto &c : channelsOrigin) {
      channels.push_back(c.get());
    }
    if (channels.empty()) {
      return;
    }

    // Step 2: group channels
    // -  each entry of the channelsGroupedByProducers is keyed by the srcOp.
    // -  each entry of the channelsGroupedByConsumers is keyed by the dstOp.
    DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByProducers;
    DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByConsumers;
    SmallVector<Channel *> orderedChannels;
    groupChannels(channels, channelsGroupedByProducers,
                  channelsGroupedByConsumers, orderedChannels);

    // Step 3: reorder producer ops and the backward slices of the producer ops.
    reorderProducerOps(channels);

    // Step 4: find top-level ops that contain a channel, also create new ForOps
    // by adding phase and bufferIdx to the original ForOps, erase the original
    // ForOps.
    SmallVector<Operation *> asyncTaskTopOps =
        getTaskTopRegion(funcOp, channels);
    // Update mapToRepresenting that maps a channel to the representing channel
    // in the sharing group.
    DenseMap<Channel *, Channel *> mapToRepresenting;
    SmallVector<Operation *> opsWithBufferReuse;
    reuseBuffers(asyncTaskTopOps, channels, mapToRepresenting,
                 opsWithBufferReuse);
    SmallVector<Operation *> opList;
    for (auto &op : asyncTaskTopOps) {
      if (auto origIfOp = dyn_cast<scf::IfOp>(op)) {
        opList.push_back(op);
      }
      if (auto origForOp = dyn_cast<scf::ForOp>(op))
        opList.push_back(op);
    }
    DenseSet<Operation *> opsWithChannels;
    updateAccumRegions(opList, channels, opsWithChannels);
    // Use and update opsWithBufferReuse.
    appendBufferIdxArgs(asyncTaskTopOps, numBuffers, channels,
                        mapToRepresenting, opsWithBufferReuse, opsWithChannels);
    LLVM_DEBUG({
      LDBG("\n\nafter appendBufferIdxArgs");
      funcOp.dump();
    });

    // Step 5: Create buffers. An array of buffers for each channel. Update
    // channelReuse that maps from a representing channel to the group of
    // channels that share buffers.
    DenseMap<Channel *, SmallVector<Channel *>> channelReuse;
    DenseMap<Channel *, Value> bufferMap =
        createBuffer(channelsGroupedByProducers, funcOp, numConsumerGroups,
                     mapToRepresenting, channelReuse);
    LLVM_DEBUG({
      LDBG("\n\nafter createBuffer");
      funcOp.dump();
    });

    // Step 6: Lower the loads. Also add local copy ops for non-load
    // producers.
    DenseMap<Channel *, std::pair<Operation *, Operation *>> copyOpMap;
    insertAsyncCopy(funcOp, channelsGroupedByProducers, bufferMap, copyOpMap,
                    opsWithChannels, opsWithBufferReuse);
    LLVM_DEBUG({
      LDBG("\n\nwith async copy");
      funcOp.dump();
    });

    // Step 7: Create tokens. A set of tokens for each group of channels for
    // each channel.
    DenseMap<Channel *, DenseMap<int, Value>> barrierAllocMap;
    DenseMap<Channel *, CommChannel> tokenMap;
    createToken(channelsGroupedByConsumers, orderedChannels, funcOp,
                numConsumerGroups, copyOpMap, channelReuse, tokenMap);
    LLVM_DEBUG({
      LDBG("\n\nafter createToken");
      funcOp.dump();
    });

    // Step 8: add async communication ops (ProducerAcquire etc). Also lower
    // TMA loads.
    insertAsyncComm(funcOp, channelsGroupedByConsumers, tokenMap,
                    barrierAllocMap, bufferMap, copyOpMap, numConsumerGroups,
                    opsWithChannels, opsWithBufferReuse);
    LLVM_DEBUG({
      LDBG("\n\nwith SyncOps");
      funcOp.dump();
    });

    // If loadResult has a single use which is LocalAlloc, we can get rid of
    // sharedLoad and replace all uses of LocalAlloc with viewLoad.
    foldLocalLoads(funcOp);
    LLVM_DEBUG({
      LDBG("\n\nsimplify localLoad + localAlloc");
      funcOp.dump();
    });

    auto ret = SpecializeRegion(funcOp, regDecProducer, regIncConsumer);
    LLVM_DEBUG({
      LDBG("\n\nwith SpecializeRegion");
      funcOp.dump();
    });
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
    LLVM_DEBUG({
      LDBG("post pass");
      getOperation()->dump();
    });
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
