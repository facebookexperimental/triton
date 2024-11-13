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
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"
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

std::pair<int, bool> scanRegUsage(ArrayRef<Operation *> opList,
                                  AsyncTaskId asyncTaskId, int regDecProducer,
                                  int regIncConsumer) {
  // TODO: scan ops to estimate register usage
  if (asyncTaskId == 0) {
    // deallocate registers
    return {regDecProducer == 0 ? 40 : regDecProducer, false};
  } else {
    // allocate registers
    return {regIncConsumer == 0 ? 232 : regIncConsumer, true};
  }
}

// Create IfOp for each ayncTaskId.
DenseMap<AsyncTaskId, scf::IfOp> SpecializeRegion(triton::FuncOp funcOp,
                                                  int regDecProducer,
                                                  int regIncConsumer) {
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  auto loc = funcOp.getLoc();

  // Collect original operations
  SmallVector<Operation *> opList;
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &op : block.getOperations())
      opList.push_back(&op);
  }

  // Create GetAsyncTaskIdOp.
  Block *lastBlock = &funcOp.getBody().back();
  auto returnOp = llvm::cast<triton::ReturnOp>(lastBlock->getTerminator());
  builder.setInsertionPoint(returnOp);
  Value curAsyncTaskId = builder.create<ttng::GetAsyncTaskIdOp>(loc);

  // Resources for each asyncTaskId: builder, IfOp, and IRMapping.
  DenseMap<AsyncTaskId, std::shared_ptr<OpBuilderWithAsyncTaskIds>>
      tasksToBuilders;
  DenseMap<AsyncTaskId, scf::IfOp> tasksToIfOp;
  DenseMap<AsyncTaskId, IRMapping> tasksToIRMappings;

  for (AsyncTaskId asyncTaskId : getNestedAsyncTaskIds(funcOp)) {
    // Create IfOp for each asyncTaskId.
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, curAsyncTaskId,
        builder.create<arith::ConstantIntOp>(loc, asyncTaskId, 32));

    auto ifOp = builder.create<scf::IfOp>(loc, cond);
    tasksToIfOp[asyncTaskId] = ifOp;
    setAsyncTaskIds(ifOp, {asyncTaskId});

    // Create OpBuilderWithAsyncTaskIds for each taskId.
    auto taskBuilder = std::make_shared<OpBuilderWithAsyncTaskIds>(context);
    tasksToBuilders[asyncTaskId] = taskBuilder;
    taskBuilder->setAsynTaskIdsFromArray({asyncTaskId});

    // Decide if this taskId is a producer or a consumer, and create either
    // RegAllocOp or RegDeallocOp accordingly.
    auto regAlloc =
        scanRegUsage(opList, asyncTaskId, regDecProducer, regIncConsumer);
    taskBuilder->setInsertionPointToStart(&(ifOp.getThenRegion().front()));
    if (regAlloc.second)
      taskBuilder->create<ttng::RegAllocOp>(
          loc, taskBuilder->getI32IntegerAttr(regAlloc.first));
    else
      taskBuilder->create<ttng::RegDeallocOp>(
          loc, taskBuilder->getI32IntegerAttr(regAlloc.first));

    // Set insertion point before yieldOp.
    auto yieldOp = ifOp.thenYield();
    setAsyncTaskIds(yieldOp, {asyncTaskId});
    taskBuilder->setInsertionPoint(yieldOp);
  }

  // Clone all operations into the corresponding if blocks. If the operation has
  // multiple taskIds, it will be cloned for multiple if blocks.
  // If the original code has an IfOp, we should only clone its
  // body with the right asyncTaskId, instead of cloning the IfOp.
  SmallVector<Operation *> cloned;
  for (Operation *op : opList) {
    auto asyncTaskIds = getAsyncTaskIds(op);
    if (asyncTaskIds.size() == 0)
      continue;
    cloned.push_back(op);
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      DenseMap<AsyncTaskId, scf::IfOp> tasksToThisIfOp;
      // TODO: handle outputs of this IfOp.
      for (AsyncTaskId asyncTaskId : getAsyncTaskIds(op)) {
        IRMapping &mapping = tasksToIRMappings[asyncTaskId];
        auto ifOpForTask = tasksToBuilders[asyncTaskId]->create<scf::IfOp>(
            loc, mapping.lookup(ifOp.getCondition()));
        tasksToThisIfOp[asyncTaskId] = ifOpForTask;
        auto newYieldOp = ifOpForTask.thenYield();
        tasksToBuilders[asyncTaskId]->setInsertionPoint(newYieldOp);
      }
      // Handle thenRegion of this IfOp.
      for (Operation &thenOp : ifOp.thenBlock()->without_terminator()) {
        LLVM_DEBUG({
          LDBG("specialize thenBlock inside ifOp ");
          thenOp.dump();
        });
        for (AsyncTaskId asyncTaskId : getAsyncTaskIds(&thenOp)) {
          IRMapping &mapping = tasksToIRMappings[asyncTaskId];
          Operation *newOp =
              tasksToBuilders[asyncTaskId]->clone(thenOp, mapping);
          for (unsigned i = 0; i < thenOp.getNumResults(); ++i)
            mapping.map(thenOp.getResult(i), newOp->getResult(i));
        }
      }
      if (!ifOp.elseBlock())
        continue; // Done with this IfOp, continue to the next op.
      // Handle elseRegion of the IfOp.
      for (AsyncTaskId asyncTaskId : getAsyncTaskIds(op)) {
        auto newYieldOp = tasksToThisIfOp[asyncTaskId].elseYield();
        tasksToBuilders[asyncTaskId]->setInsertionPoint(newYieldOp);
      }
      for (Operation &thenOp : ifOp.elseBlock()->without_terminator()) {
        LLVM_DEBUG({
          LDBG("specialize elseBlock inside ifOp ");
          thenOp.dump();
        });
        for (AsyncTaskId asyncTaskId : getAsyncTaskIds(&thenOp)) {
          IRMapping &mapping = tasksToIRMappings[asyncTaskId];
          Operation *newOp =
              tasksToBuilders[asyncTaskId]->clone(thenOp, mapping);
          for (unsigned i = 0; i < thenOp.getNumResults(); ++i)
            mapping.map(thenOp.getResult(i), newOp->getResult(i));
        }
      }
    } else {
      for (AsyncTaskId asyncTaskId : getAsyncTaskIds(op)) {
        IRMapping &mapping = tasksToIRMappings[asyncTaskId];
        Operation *newOp = tasksToBuilders[asyncTaskId]->clone(*op, mapping);
        for (unsigned i = 0; i < op->getNumResults(); ++i)
          mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }
  }

  LLVM_DEBUG({
    LDBG("\n\nWith task Id checks");
    funcOp.dump();
  });

  // Remove original operations that have been cloned in reverse order.
  for (auto it = cloned.rbegin(); it != cloned.rend(); ++it) {
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

  Channel(int producer, SmallVector<int> &consumers, Operation *src,
          Operation *dst, Value srcOperand)
      : relation(producer, consumers), srcOp(src), dstOp(dst),
        srcOperand(srcOperand) {}

  bool operator==(const Channel &c) {
    return relation == c.relation && srcOp == c.srcOp && dstOp == c.dstOp;
  }

  Relation relation; // producer task Id, a list of consumer task Ids
  Operation *srcOp;
  Operation *dstOp;
  Value srcOperand;
};

// Loads will be in producer warp groups. For now, we only allow a single
// warp group/task for a producer. For each LoadOp, create a channel from it
// to any direct user which belongs to a different taskId.
void collectAsyncChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                          triton::FuncOp &funcOp) {
  funcOp.walk([&](Operation *op) {
    if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op)) {
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

      for (auto result : op->getResults()) {
        if (result.use_empty()) {
          continue;
        }
        for (Operation *userOp : result.getUsers()) {
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
                producerTaskId, consumerTaskIds, op, userOp, result));
          }
        }
      }
    }
  });

  LLVM_DEBUG({
    LDBG("Async channels:");
    for (auto &channel : channels) {
      LDBG("producer op: " << channel->relation.first);
      channel->srcOp->dump();
      for (auto &asyncTaskId : channel->relation.second)
        LDBG("consumer: " << asyncTaskId);
      channel->dstOp->dump();
    }
  });
}

// Update map, which will be keyed by dstOp of the channel. Use mapKeyVec to
// enforce deterministic order for map.
void groupChannels(SmallVector<Channel *> &channels,
                   DenseMap<Operation *, SmallVector<Channel *>> &map,
                   SmallVector<Operation *> &mapKeyVec) {
  // Two channels can be combined if
  //   src1 and src2 are in the same block and
  //   (dst1 == dst2 or
  //    (dst1 and dst2 are in the same block, both have a single user, and
  //     dst1User == dst2User and dst1User is in the same block as dst1))
  auto channelCanBeMerged = [](Channel *c1, Channel *c2) -> bool {
    if (c1->srcOp->getBlock() != c2->srcOp->getBlock())
      return false;
    Operation *dst1 = c1->dstOp, *dst2 = c2->dstOp;
    if (dst1 == dst2)
      return true;
    if (dst1->getBlock() != dst2->getBlock() || !dst1->hasOneUse() ||
        !dst2->hasOneUse())
      return false;
    Operation *dst1User = *(dst1->getUsers().begin());
    Operation *dst2User = *(dst2->getUsers().begin());
    return dst1User == dst2User && dst1User->getBlock() == dst1->getBlock();
  };
  assert(channels.size() > 0 && "channel size is zero");
  // Compare with existing channels in the map to see if it can be combined.
  for (auto *c0 : channels) {
    bool merged = false;
    for (auto &kv : map) {
      if (kv.second.size() > 0 && channelCanBeMerged(c0, kv.second.front())) {
        kv.second.push_back(c0);
        merged = true;
        break;
      }
    }
    if (!merged) { // Create a new entry.
      auto *keyOp = c0->dstOp;
      if (!map.count(keyOp))
        mapKeyVec.push_back(keyOp);
      map[keyOp].push_back(c0);
    }
  }

  // Reorder channels associated with one entry based on program order of the
  // producers.
  for (auto &kv : map) {
    if (kv.second.size() > 1) {
      auto &allOps = kv.second.front()->srcOp->getBlock()->getOperations();
      std::sort(
          kv.second.begin(), kv.second.end(), [&](Channel *a, Channel *b) {
            auto itrA =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == a->srcOp;
                });
            auto itrB =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == b->srcOp;
                });
            assert(itrA != allOps.end() && itrB != allOps.end());
            return std::distance(itrA, itrB) < 0;
          });
    }
  }
}

// Reorder producer ops to unblock consumers interleavingly.
void reorderProducerOps(SmallVector<Channel *> &channels) {
  if (channels.size() <= 1)
    return;

  // Bail out if channels are not in the same block
  auto block = channels.front()->srcOp->getBlock();
  for (auto &channel : channels) {
    if (channel->srcOp->getBlock() != block) {
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
  Operation *currOp = channels.front()->srcOp;
  for (auto &group : groupedProducerOps) {
    for (auto &channel : group.second) {
      channel->srcOp->moveAfter(currOp);
      currOp = channel->srcOp;
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
      getBackwardSlice(channel->srcOp, &backwardSlice, opt);
      for (auto &op : backwardSlice) {
        if (op->getBlock() == block)
          op->moveBefore(channel->srcOp);
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

bool isInnermostLoop(scf::ForOp forOp) {
  for (Operation &nestedOp : forOp.getBody()->getOperations()) {
    if (isa<scf::ForOp>(nestedOp)) {
      return false;
    }
  }
  return true;
}

// Add phase and bufferIndex to be used when lowering the producer.
scf::ForOp createNewLoop(scf::ForOp forOp, int numBuffers,
                         scf::ForOp &parentForOp) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();

  OpBuilderWithAsyncTaskIds builder(forOp.getContext());
  builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(forOp));
  builder.setInsertionPoint(forOp);

  Value numBuffersVal =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, numBuffers, 32);

  // Step 1: Append bufferIdx and phase as forOp arguments.
  Value phase =
      body->insertArgument(body->getNumArguments(), builder.getI1Type(), loc);
  Value bufferIdx =
      body->insertArgument(body->getNumArguments(), builder.getI32Type(), loc);

  // Step 2: Generate bufferIdx and phase for next iteration:
  //   nextBufferIdx = bufferIdx + 1
  //   nextPhase = ((nextBufferIdx < numBuffers && curPhase) ||
  //                (nextBufferIdx >= numBuffers && curPhase^1))
  //   nextBufferIdx = nextBufferIdx >= numBuffers ? 0 : nextBufferIdx
  auto yieldOp = llvm::cast<scf::YieldOp>(body->getTerminator());
  builder.setInsertionPoint(yieldOp);
  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 32);
  Value zero = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 32);
  Value _1_1b = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  //   nextBufferIdx = bufferIdx + 1
  Value nextBufferIdx =
      builder.createWithAsyncTaskIds<arith::AddIOp>(loc, bufferIdx, one);
  Value bufferGECond = builder.createWithAsyncTaskIds<arith::CmpIOp>(
      loc, arith::CmpIPredicate::uge, nextBufferIdx, numBuffersVal);
  Value bufferLTCond = builder.createWithAsyncTaskIds<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, nextBufferIdx, numBuffersVal);
  if (isInnermostLoop(forOp)) {
    // nextBufferIdx >= numBuffers ? nextBufferIdx - numBuffers : nextBufferIdx
    Value moduloBufferIdx = builder.createWithAsyncTaskIds<arith::SubIOp>(
        loc, nextBufferIdx, numBuffersVal);
    nextBufferIdx = builder.createWithAsyncTaskIds<mlir::arith::SelectOp>(
        loc, bufferGECond, moduloBufferIdx, nextBufferIdx);
  }

  // nextPhase = ((nextBufferIdx < numBuffers && curPhase) ||
  //              (nextBufferIdx >= numBuffers && curPhase^1))
  Value flipPhase =
      builder.createWithAsyncTaskIds<mlir::arith::XOrIOp>(loc, phase, _1_1b);
  Value cond0 = builder.createWithAsyncTaskIds<mlir::arith::AndIOp>(
      loc, bufferGECond, flipPhase);
  Value cond1 = builder.createWithAsyncTaskIds<mlir::arith::AndIOp>(
      loc, bufferLTCond, phase);
  Value nextPhase =
      builder.createWithAsyncTaskIds<mlir::arith::OrIOp>(loc, cond0, cond1);

  // Step 3: Add nextBufferIdx and nextPhase to yieldOp.
  yieldOp->insertOperands(yieldOp.getNumOperands(), {nextPhase, nextBufferIdx});

  // Step 4: Create loop arguments for the new ForOp.
  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getInitArgs())
    newLoopArgs.push_back(operand);

  builder.setInsertionPoint(forOp);
  Value initBufferIdx, initPhase;
  zero = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 32);
  // Set initial values for bufferIdx and phase.
  if (parentForOp) {
    // Assume parent ForOp has bufferIdx as the last argument.
    initBufferIdx = parentForOp.getBody()->getArguments().back();

    // numSteps = ((upperBound - lowerBound) + forOpStep - 1) / forOpStep
    Value numSteps = builder.createWithAsyncTaskIds<arith::SubIOp>(
        loc, forOp.getUpperBound(), forOp.getLowerBound());
    numSteps = builder.createWithAsyncTaskIds<arith::AddIOp>(loc, numSteps,
                                                             forOp.getStep());
    Value one =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 32);
    Value two =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 2, 32);
    numSteps =
        builder.createWithAsyncTaskIds<arith::SubIOp>(loc, numSteps, one);
    numSteps = builder.createWithAsyncTaskIds<arith::DivUIOp>(loc, numSteps,
                                                              forOp.getStep());

    // initBufferIdx = (parentForOp.bufferIdx * numSteps) % numBuffers
    //   tmpIdx = parentForOp.bufferIdx * numSteps
    //   initBufferIdx = tmpIdx - tmpIdx / numBuffers * numBuffers
    // initPhase = (tmpIdx / numBuffers) & 1
    initBufferIdx = builder.createWithAsyncTaskIds<arith::MulIOp>(
        loc, initBufferIdx, numSteps);
    Value bufferIdx = builder.createWithAsyncTaskIds<arith::DivUIOp>(
        loc, initBufferIdx, numBuffersVal);
    initBufferIdx = builder.createWithAsyncTaskIds<arith::SubIOp>(
        loc, initBufferIdx,
        builder.createWithAsyncTaskIds<arith::MulIOp>(loc, bufferIdx,
                                                      numBuffersVal));
    bufferIdx =
        builder.createWithAsyncTaskIds<arith::AndIOp>(loc, bufferIdx, one);
    initPhase = builder.createWithAsyncTaskIds<arith::TruncIOp>(
        loc, builder.getI1Type(), bufferIdx);
  } else {
    // Set initial phase to false, and initial bufferIdx to 0.
    initBufferIdx = zero;
    initPhase = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 1);
  }
  newLoopArgs.append({initPhase, initBufferIdx});

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

// Find top-level ops which contain at least one channel. If a channel's srcOp
// and dstOp belong to the inner loop, the outer loop will be part of
// asyncTaskOps.
SmallVector<Operation *>
getTaskTopRegion(triton::FuncOp funcOp,
                 const SmallVector<Channel *> &channels) {
  SmallVector<Operation *> asyncTaskOps;
  auto isAsyncTaskTopOp = [&](Operation *taskTopOp) -> bool {
    for (auto c : channels) {
      Operation *producer = c->srcOp, *consumer = c->dstOp;
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
  return asyncTaskOps;
}

// For ForOps in taskTopOps, create new ForOp for each by adding phase,
// bufferIdx to the arguments.
void appendBufferIdxArgs(SmallVector<Operation *> &taskTopOps, int numBuffers) {
  SmallVector<scf::ForOp> orderedForOps;
  for (auto &op : taskTopOps) {
    op->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
      if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
        orderedForOps.push_back(forOp);
      }
    });
  }

  for (auto &origForOp : orderedForOps) {
    scf::ForOp parentForOp = origForOp->getParentOfType<scf::ForOp>();
    scf::ForOp newForOp;
    // for(...) -> for(..., phase, bufferIdx)
    newForOp = createNewLoop(origForOp, numBuffers, parentForOp);
    // origForOp is erased in createNewLoop. If origForOp is a top operation
    // (i.e in taskTopOps), make sure taskTopOps is updated with the newForOp.
    auto asyncTaskLoopForItr = std::find(taskTopOps.begin(), taskTopOps.end(),
                                         origForOp.getOperation());
    if (asyncTaskLoopForItr != taskTopOps.end()) {
      // Update taskTopOps.
      *asyncTaskLoopForItr = newForOp.getOperation();
    }
  }
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
  auto barrierEncoding =
      ttg::SharedEncodingAttr::get(context, 1, 1, 1, {0}, barrierCTALayout);
  Type barrierMemDescType = tt::MemDescType::get(
      {distance}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Type singleBarrierMemDescType =
      tt::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
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

// map: channels are grouped together.
// Go through each group, check the first channel in the group, create a token
// for each consumer taskId. Return a map that maps each channel + consumer
// taskId to a token. Also update barrierAllocMap that maps each channel +
// consumer taskId to a BarrierAlloc.
DenseMap<Channel *, DenseMap<int, Value>>
createToken(const DenseMap<Operation *, SmallVector<Channel *>> &map,
            const SmallVector<Operation *> &mapKeyVec, triton::FuncOp funcOp,
            int numBuffers, int numConsumerGroups,
            DenseMap<Channel *, DenseMap<int, Value>> &barrierAllocMap) {
  DenseMap<Channel *, DenseMap<int, Value>> ret;
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  for (auto *key : mapKeyVec) {
    auto it = map.find(key);
    for (auto consumerAsyncTaskId : it->second.front()->relation.second) {
      Value v;
      // TODO: remove numConsumerGroups here, it is only used for grid
      // partitioning.
      if (it->second.front()->srcOp->getParentOfType<scf::ForOp>()) {
        v = builder.create<ttng::CreateTokenOp>(funcOp.getLoc(),
                                                numBuffers * numConsumerGroups);
      } else {
        v = builder.create<ttng::CreateTokenOp>(funcOp.getLoc(),
                                                numConsumerGroups);
      }
      // Channels in the group share the same set of tokens.
      for (auto &c : it->second)
        ret[c][consumerAsyncTaskId] = v;

      auto producerOp = it->second.front()->srcOp;
      if (isa<tt::ExperimentalDescriptorLoadOp>(producerOp)) {
        Value bAlloc = createBarrierAlloc(funcOp, numBuffers);
        // Channels in the group share the same set of tokens.
        for (auto &c : it->second) {
          ret[c][consumerAsyncTaskId] = v;
          barrierAllocMap[c][consumerAsyncTaskId] = bAlloc;
        }
      }
    }
  }
  return ret;
}

// Create a buffer array for each channel, if the producer is in a ForOp,
// the buffer array will contain numBuffers.
DenseMap<Channel *, Value> createBuffer(const SmallVector<Channel *> &channels,
                                        triton::FuncOp funcOp, int numBuffers,
                                        int numConsumerGroups) {
  DenseMap<Channel *, Value> bufferMap;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  for (const auto &c : channels) {
    if (auto tensorType = dyn_cast<RankedTensorType>(c->srcOperand.getType())) {
      // Get basic information from tensorType
      auto order = ttg::getOrder(tensorType.getEncoding());
      auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
      auto elemType = tensorType.getElementType();

      // Get shape, layout and type of a slice
      auto sliceShape = tensorType.getShape();
      auto sharedLayout = ttg::SharedEncodingAttr::get(
          context, sliceShape, order, CTALayout, elemType);
      auto sliceType =
          RankedTensorType::get(sliceShape, elemType, sharedLayout);

      // Get shape, layout and type of the complete buffer
      SmallVector<int64_t> bufferShape(sliceShape.begin(), sliceShape.end());
      if (c->srcOp->getParentOfType<scf::ForOp>())
        bufferShape.insert(bufferShape.begin(), numBuffers);
      else
        bufferShape.insert(bufferShape.begin(), 1);
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(context);
      auto bufferType =
          RankedTensorType::get(bufferShape, elemType, sharedLayout);
      Type memdescType =
          tt::MemDescType::get(bufferShape, elemType, sharedLayout,
                               sharedMemorySpace, /*mutableMemory*/ true);
      Value buffer;
      if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(c->srcOp)) {
        buffer =
            builder.create<ttg::LocalAllocOp>(funcOp.getLoc(), memdescType);
      } else {
        buffer = builder.create<ttg::LocalAllocOp>(funcOp.getLoc(), memdescType,
                                                   c->srcOperand);
      }
      bufferMap[c] = buffer;
    } else {
      llvm_unreachable("Unexpected result type");
    }
  }
  return bufferMap;
}

static Operation *createAsyncCopy(const DenseMap<Channel *, Value> &bufferMap,
                                  Channel *c, Operation *op,
                                  SmallVector<AsyncTaskId> &asyncTasksPC,
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
    return nullptr;
  // Get basic information from tensorType
  auto order = ttg::getOrder(tensorType.getEncoding());
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout = ttg::SharedEncodingAttr::get(context, sliceShape, order,
                                                   CTALayout, elemType);
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  tt::MemDescType subviewTy =
      tt::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
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
  builder.setInsertionPoint(c->dstOp);
  SmallVector<Value> loadOffsets(sliceType.getRank() + 1, zero);
  loadOffsets[0] = bufferIdxExtract;
  auto viewLoad = builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      loadOp.getLoc(), subviewTy, buffer, loadOffsets);
  auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
      loadOp.getLoc(), loadOp.getType(), viewLoad /*,wait->getResult(0)*/);
  // Replace all uses of loadResult
  loadResult.replaceAllUsesWith(sharedLoad.getResult());
  loadOp.erase();
  return copy;
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
  tt::MemDescType barrierTy = tt::MemDescType::get(
      {1}, builder.getI64Type(),
      cast<tt::MemDescType>(barrierAlloc.getType()).getEncoding(),
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

  auto order = ttg::getOrder(tensorType.getEncoding());
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout = ttg::SharedEncodingAttr::get(context, sliceShape, order,
                                                   CTALayout, elemType);
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  tt::MemDescType subviewTy =
      tt::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
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
    auto pipelineBuffer = getBufferForPipelineStage(builder, tmaLoad.getType(),
                                                    buffer, bufferIdx, true);
    copy = builder.createWithAsyncTaskIds<ttng::AsyncTMACopyGlobalToLocalOp>(
        loc, tmaLoad.getDescPtr(), tmaLoad.getIndices(), prodBarrier,
        pipelineBuffer, pred);
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

// Lower producers for channels. Here channels are grouped in "map". tokenMap
// tracks the set of tokens for each channel.
void buildAsyncComm(
    const DenseMap<Operation *, SmallVector<Channel *>> &map,
    const DenseMap<Channel *, DenseMap<int, Value>> &tokenMap,
    const DenseMap<Channel *, DenseMap<int, Value>> &barrierAllocMap,
    const DenseMap<Channel *, Value> &bufferMap, int numBuffers,
    int numConsumerGroups) {

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

  auto consumerReleaseHeutistic = [&](Operation *p, Operation *c,
                                      int consumerAsyncTaskId) -> Operation * {
    if (c->getBlock() != p->getBlock())
      return getSameLevelOp(p, c);
    for (auto it = c->getBlock()->rbegin(); it != c->getBlock()->rend(); ++it) {
      if (!it->hasAttr("async_task_id"))
        continue;
      auto asyncAttr = it->getAttrOfType<DenseIntElementsAttr>("async_task_id")
                           .getValues<int>();
      if (asyncAttr.size() == 1 && asyncAttr[0] == consumerAsyncTaskId)
        return &(*it);
    }
    return nullptr;
  };

  auto getAsyncTasks = [&](Operation *p, Operation *c,
                           SmallVector<AsyncTaskId> &asyncTaskP,
                           SmallVector<AsyncTaskId> &asyncTaskC,
                           SmallVector<AsyncTaskId> &asyncTasksPC) -> void {
    asyncTaskP = getNestedAsyncTaskIds(p);
    asyncTaskC = getNestedAsyncTaskIds(c);
    asyncTasksPC.reserve(asyncTaskP.size() + asyncTaskC.size());
    asyncTasksPC.insert(asyncTasksPC.end(), asyncTaskP.begin(),
                        asyncTaskP.end());
    asyncTasksPC.insert(asyncTasksPC.end(), asyncTaskC.begin(),
                        asyncTaskC.end());
  };

  // Go through each channel group.
  for (auto kv : map) {
    auto headProducer = kv.second.front()->srcOp;
    auto tailProducer = kv.second.back()->srcOp;
    auto headConsumer = kv.second.front()->dstOp;
    auto tailConsumer = kv.second.back()->dstOp;
    // We have one set of tokens for each channel group.
    auto tokens = tokenMap.find(kv.second.front())->second;

    SmallVector<AsyncTaskId> asyncTaskP, asyncTaskC, asyncTasksPC;
    getAsyncTasks(headProducer, headConsumer, asyncTaskP, asyncTaskC,
                  asyncTasksPC);
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
      // We already added phase, bufferIdx to the ForOp.
      auto tSize = forOp.getBody()->getArguments().size();
      assert(tSize >= 2);
      bufferIdx = forOp.getBody()->getArguments().back();
      phase = forOp.getBody()->getArgument(tSize - 2); // next to last argument
    } else {
      // Producer is not in a ForOp, create phase and bufferIdx here.
      bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 32);
      phase = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 1);
    }

    assert((isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(headProducer)) &&
           "producer must be a LoadOp or tma LoadOp");
    builder.setAsynTaskIdsFromArray(asyncTaskP);
    for (auto token : tokens) {
      // Insert ProducerAcquireOp before the producer.
      builder.setInsertionPoint(headProducer);
      builder.createWithAsyncTaskIds<ttng::ProducerAcquireOp>(
          headProducer->getLoc(), token.second, bufferIdx, phase);

      // Insert ProducerCommitOp if producer is LoadOp. For TMA, TMA lowering
      // will handle the ProducerCommit.
      if (isa<tt::LoadOp>(headProducer)) {
        builder.setInsertionPointAfter(tailProducer);
        builder.createWithAsyncTaskIds<ttng::ProducerCommitOp>(
            tailProducer->getLoc(), token.second, bufferIdx);
      }
    }

    for (auto token : tokens) {
      builder.setAsynTaskIdsFromArray(token.first);
      // Insert ConsumerWaitOp
      if (!isa<tt::ExperimentalDescriptorLoadOp>(headProducer)) {
        auto consumerWaitPoint = getSameLevelOp(headProducer, headConsumer);
        builder.setInsertionPoint(consumerWaitPoint);
        builder.createWithAsyncTaskIds<ttng::ConsumerWaitOp>(
            headConsumer->getLoc(), token.second, bufferIdx, phase);
      }

      // Insert ConsumerReleaseOp.
      auto consumerReleasePoint =
          consumerReleaseHeutistic(tailProducer, tailConsumer, token.first);
      builder.setInsertionPointAfter(consumerReleasePoint);
      builder.createWithAsyncTaskIds<ttng::ConsumerReleaseOp>(
          consumerReleasePoint->getLoc(), token.second, bufferIdx);
    }

    SmallVector<tt::ExperimentalDescriptorLoadOp> tmaLoads;
    SmallVector<Value> buffers;
    // Go through all channels in this channel group.
    for (auto &c : kv.second) {
      assert(
          (isa<triton::LoadOp, tt::ExperimentalDescriptorLoadOp>(c->srcOp)) &&
          "producer must be a LoadOp or tma LoadOp");
      bool insideLoop = c->srcOp->getParentOfType<scf::ForOp>() != nullptr;
      if (isa<triton::LoadOp>(c->srcOp)) {
        // After createAsyncCopy, c->srcOp/headProducer are no longer valid.
        createAsyncCopy(bufferMap, c, c->srcOp, asyncTasksPC, bufferIdx,
                        bufferIdx);
      } else if (auto tmaLoad =
                     dyn_cast<tt::ExperimentalDescriptorLoadOp>(c->srcOp)) {
        tmaLoads.push_back(tmaLoad);
        buffers.push_back(bufferMap.find(c)->second);
      }
    }

    // Optimize TMA loads.
    if (tmaLoads.size() > 0) {
      auto barrierAllocs = barrierAllocMap.find(kv.second.front())->second;
      // TODO: we created one Alloc for each consumer taskId, but here, we
      // only use the first Alloc.
      auto barrierAlloc = barrierAllocs.begin()->second;
      optimizeTMALoads(builder, tmaLoads, buffers, barrierAlloc, bufferIdx,
                       bufferIdx, phase, headProducer, headConsumer);
    }
  }
}

// Collect argument indices that are used by the specific taskId.
static SmallVector<unsigned> collectBlockArgsForTask(
    scf::ForOp forOp, int asyncTaskId,
    DenseMap<BlockArgument, Value> &blockArgToYieldOperand) {
  DenseSet<Operation *> seen;
  // Collect argument indices that can be reached along the definition chain.
  // If reaching a BlockArgument, visit the corresponding yield operand.
  SetVector<unsigned> argIndices;
  std::function<void(Operation *)> dfs = [&](Operation *op) {
    if (!seen.insert(op).second)
      return;
    for (Value operand : op->getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (!blockArgToYieldOperand[blockArg])
          continue;
        argIndices.insert(blockArg.getArgNumber() -
                          forOp.getNumInductionVars());
        operand = blockArgToYieldOperand[blockArg];
      }
      Operation *depOp = operand.getDefiningOp();
      assert(depOp && "Unexpected Value with no defining op");
      if (depOp->getBlock() != forOp.getBody())
        continue;
      assert(hasAsyncTaskId(depOp, asyncTaskId) && "Dependency error");
      dfs(depOp);
    }
  };

  // Start from operations that are marked with this asyncTaskId explicitly and
  // check dependency with DFS traversal.
  forOp.walk([&](Operation *op) {
    if (hasAsyncTaskId(op, asyncTaskId) && !isa<scf::YieldOp>(op))
      dfs(op);
  });

  SmallVector<unsigned> args(argIndices.begin(), argIndices.end());
  llvm::sort(args);
  return args;
}

DenseMap<AsyncTaskId, scf::ForOp>
createForOpsForEachAsyncTaskId(scf::ForOp forOp) {
  // Collect operation list for each asyncTaskId.
  DenseMap<AsyncTaskId, SmallVector<Operation *>> opList;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto ids = getAsyncTaskIds(&op);
    for (AsyncTaskId asyncTaskId : ids)
      opList[asyncTaskId].push_back(&op);
  }

  // Prepare blockArgToYieldOperand mapping.
  DenseMap<BlockArgument, Value> blockArgToYieldOperand;
  auto yieldOp = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  assert(yieldOp.getNumOperands() == forOp.getNumRegionIterArgs());
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
    blockArgToYieldOperand[forOp.getRegionIterArg(i)] = yieldOp.getOperand(i);

  auto loc = forOp.getLoc();
  OpBuilderWithAsyncTaskIds builder(forOp.getContext());
  DenseMap<AsyncTaskId, scf::ForOp> asyncTasksToForOp;

  // Create newForOp for each task Id.
  for (AsyncTaskId asyncTaskId : getNestedAsyncTaskIds(forOp)) {
    auto usedArgs =
        collectBlockArgsForTask(forOp, asyncTaskId, blockArgToYieldOperand);

    // Prepare newLoopArgs.
    SmallVector<Value> newLoopArgs;
    for (unsigned argNumber : usedArgs)
      newLoopArgs.push_back(forOp.getInitArgs()[argNumber]);

    // Create newForOp.
    builder.setAsynTaskIdsFromArray({asyncTaskId});
    builder.setInsertionPoint(forOp);
    auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        newLoopArgs);
    if (forOp->getAttr("tt.loop_schedule"))
      newForOp->setAttr("tt.loop_schedule", forOp->getAttr("tt.loop_schedule"));

    // Initialize Value mapping from forOp to newForOp
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (unsigned i = 0; i < usedArgs.size(); ++i) {
      auto oldArg = forOp.getRegionIterArgs()[usedArgs[i]];
      auto newArg = newForOp.getRegionIterArgs()[i];
      mapping.map(oldArg, newArg);
    }

    // Clone all operations with this asyncTaskId to newForOp.
    builder.setInsertionPointToStart(newForOp.getBody());
    for (Operation *op : opList[asyncTaskId]) {
      Operation *newOp = builder.clone(*op, mapping);
      setAsyncTaskIds(newOp, {asyncTaskId});
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
    }

    // Create YieldOp for newForOp.
    SmallVector<Value> newYieldOperands;
    for (unsigned i : usedArgs) {
      LDBG("lookup operand " << i);
      newYieldOperands.push_back(mapping.lookup(yieldOp.getOperand(i)));
    }
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
          builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      setAsyncTaskIds(newYieldOp, {asyncTaskId});
    }

    // Replace results of forOp with results of newForOp.
    for (unsigned i = 0; i < usedArgs.size(); ++i) {
      auto oldResult = forOp.getResult(usedArgs[i]);
      auto newResult = newForOp.getResult(i);
      oldResult.replaceUsesWithIf(newResult, [&](OpOperand &operand) -> bool {
        return hasAsyncTaskId(operand.getOwner(), asyncTaskId);
      });
    }

    asyncTasksToForOp[asyncTaskId] = newForOp;
  }

  return asyncTasksToForOp;
}

// Input asyncTaskTopOp can be an IfOp that contains a ForOp. We clone
// the ForOp for each asyncTaskId.
DenseMap<AsyncTaskId, Operation *>
asyncTaskDivision(Operation *asyncTaskTopOp) {
  DenseMap<AsyncTaskId, Operation *> asyncTaskTopOpMap;
  Operation *mainForOp = asyncTaskTopOp;
  if (auto ifOp = dyn_cast<scf::IfOp>(asyncTaskTopOp)) {
    // Find the outmost ForOp inside. Assume only a single ForOp.
    Operation *nestedFor = nullptr;
    asyncTaskTopOp->walk([&](Operation *op) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assert(nestedFor == nullptr);
        nestedFor = op;
      }
    });
    assert(nestedFor && "can't find ForOp in a top-level IfOp");
    mainForOp = nestedFor;
  }
  asyncTaskTopOp->walk([&](Operation *op) {
    auto ids = getAsyncTaskIds(op);
    if (op->getNumRegions() > 0 && ids.size() > 1) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        // Create a cloned ForOp for each taskId and return the map.
        auto forOps = createForOpsForEachAsyncTaskId(forOp);
        if (op == mainForOp) {
          for (auto kv : forOps) {
            auto f = kv.second;
            auto id = getAsyncTaskIds(f.getOperation());
            assert(id.size() == 1 &&
                   "generated ForOp doesn't have one and only one asyncTaskId");
            asyncTaskTopOpMap[id.front()] = f.getOperation();
          }
        }
        // For debugging purposes, check to see if it is safe to erase the
        // original ForOp.
        bool hasIssue = false;
        for (Operation &opT : forOp.getBody()->without_terminator()) {
          // Check to see if opT is used in another block.
          for (unsigned i = 0; i < opT.getNumResults(); ++i)
            for (Operation *user : opT.getResult(i).getUsers()) {
              if (user->getBlock() != opT.getBlock()) {
                hasIssue = true;
                LLVM_DEBUG({
                  LDBG("-- op has user in another block");
                  opT.dump();
                  user->dump();
                });
              }
            }
        }
        if (hasIssue) {
          for (Operation &opT : forOp.getBody()->without_terminator()) {
            LLVM_DEBUG({
              LDBG("addr " << (&opT) << ": ");
              opT.dump();
            });
          }
        }
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
        ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
        LLVM_DEBUG({
          LDBG("erase ForOp");
          forOp.dump();
        });
        forOp.erase();
        LDBG("done erasing ForOp");
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        // The ForOp inside this ifOp will be cloned.
        LDBG("IfOp in asyncTaskDivision");
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
        LDBG("WhileOp in asyncTaskDivision");
      } else {
        llvm_unreachable("Unexpected Op with regions");
      }
    }
  });
  assert(asyncTaskTopOpMap.size() > 0 && "AsyncTask division failed");
  return asyncTaskTopOpMap;
}

void cloneAsyncTaskLoopForEachAsyncTaskId(
    SmallVector<Operation *> &asyncTaskTopOps) {
  SmallVector<Operation *> newBackBone;

  for (Operation *op : asyncTaskTopOps) {
    auto loc = op->getLoc();
    OpBuilderWithAsyncTaskIds builder(op->getContext());
    builder.setInsertionPoint(op);
    // Step 1: create a cloned forOp for each taskId based on the original
    // ForOp that is in this top-level operation.
    DenseMap<AsyncTaskId, Operation *> newAsyncTaskLoops =
        asyncTaskDivision(op);

    // Step 2: remove irrelevant Ops from the cloned ForOps.
    for (auto kv : newAsyncTaskLoops) {
      SmallVector<Operation *> deleteOps;
      AsyncTaskId targetId = kv.first;
      Operation *newAsyncTaskLoop = kv.second;
      newAsyncTaskLoop->walk([&](Operation *subOp) {
        auto ids = getAsyncTaskIds(subOp);
        if (std::find(ids.begin(), ids.end(), targetId) == ids.end()) {
          deleteOps.push_back(subOp);
        }
      });
      for (auto it = deleteOps.rbegin(); it != deleteOps.rend(); ++it) {
        (*it)->erase();
      }
    }
  }
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
    collectAsyncChannels(channelsOrigin, funcOp);
    SmallVector<Channel *> channels;
    for (const auto &c : channelsOrigin) {
      channels.push_back(c.get());
    }
    if (channels.empty()) {
      return;
    }

    // Step 2: group channels where each entry of the map is keyed by the dstOp.
    DenseMap<Operation *, SmallVector<Channel *>> map;
    SmallVector<Operation *> mapKeyVec;
    groupChannels(channels, map, mapKeyVec);

    // Step 3: reorder producer ops and the backward slices of the producer ops.
    reorderProducerOps(channels);

    // Step 4: find top-level ops that contain a channel, also create new ForOps
    // by adding phase and bufferIdx to the original ForOps, erase the original
    // ForOps.
    SmallVector<Operation *> asyncTaskTopOps =
        getTaskTopRegion(funcOp, channels);
    appendBufferIdxArgs(asyncTaskTopOps, numBuffers);

    // Step 5: Create tokens, and buffers. A set of tokens for each group of
    // channels and an array of buffers for each channel.
    DenseMap<Channel *, DenseMap<int, Value>> barrierAllocMap;
    DenseMap<Channel *, DenseMap<int, Value>> tokenMap = createToken(
        map, mapKeyVec, funcOp, numBuffers, numConsumerGroups, barrierAllocMap);
    DenseMap<Channel *, Value> bufferMap =
        createBuffer(channels, funcOp, numBuffers, numConsumerGroups);
    LLVM_DEBUG({
      LDBG("\n\nafter createBuffer");
      funcOp.dump();
    });

    // Step 6: add async communication ops (ProducerAcquire etc). Also lower the
    // loads.
    buildAsyncComm(map, tokenMap, barrierAllocMap, bufferMap, numBuffers,
                   numConsumerGroups);
    LLVM_DEBUG({
      LDBG("\n\nwith SyncOps");
      funcOp.dump();
    });

    // If loadResult has a single use which is LocalAlloc, we can get rid of
    // sharedLoad and replace all uses of LocalAlloc with viewLoad.
    DenseMap<Operation *, Value> opsToReplace;
    funcOp.walk([&](ttg::LocalAllocOp localAlloc) {
      if (auto src = localAlloc.getSrc()) {
        if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(src.getDefiningOp())) {
          opsToReplace[localAlloc] = localLoad.getSrc();
        }
      }
    });
    OpBuilderWithAsyncTaskIds builder(funcOp.getContext());
    for (auto kv : opsToReplace)
      replaceUsesAndPropagateType(builder, kv.getFirst(), kv.getSecond());
    LLVM_DEBUG({
      LDBG("\n\nsimplify localLoad + localAlloc");
      funcOp.dump();
    });

    // Clone taskTopOp, remove irrelevant blockArgument for {forOp, ifOp}
    cloneAsyncTaskLoopForEachAsyncTaskId(asyncTaskTopOps);
    LLVM_DEBUG({
      LDBG("\n\nwith Loop Split");
      funcOp.dump();
    });

    auto ret = SpecializeRegion(funcOp, regDecProducer, regIncConsumer);
    LLVM_DEBUG({
      LDBG("\n\nwith IfOps");
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
