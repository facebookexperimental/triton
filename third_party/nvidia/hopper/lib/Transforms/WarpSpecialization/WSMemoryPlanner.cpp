#include "CodePartitionUtility.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "nvgpu-ws-memory-planner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

using OperationListT = std::vector<Operation *>;

static bool isInnermostLoop(scf::ForOp forOp) {
  for (Operation &nestedOp : forOp.getBody()->getOperations()) {
    if (isa<scf::ForOp>(nestedOp)) {
      return false;
    }
  }
  return true;
}

static Channel *findChannelForOp(Operation *op,
                                 SmallVector<Channel *> &channels) {
  Channel *TheCh = nullptr;
  for (auto *ch : channels) {
    Operation *alloc = ch->getAllocOp();
    if (alloc == op) {
      TheCh = ch;
      break;
    }
  }
  return TheCh;
}

static Channel *findChannelForAlloc(Value value,
                                    SmallVector<Channel *> &channels) {
  Operation *op = value.getDefiningOp();
  Channel *TheCh = nullptr;
  for (auto *ch : channels) {
    Operation *alloc = ch->getAllocOp();
    if (alloc == op) {
      TheCh = ch;
      break;
    }
  }
  return TheCh;
}

static void getAllAcutalUsersForChannel(Channel *TheCh,
                                        DenseSet<Operation *> &users) {
  Operation *src = TheCh->getSrcOp();
  SmallVector<Operation *> dsts;
  TheCh->getDstOps(dsts);
  users.insert(src);
  for (auto *op : dsts) {
    auto actual = getActualConsumers(op);
    for (auto *tOp : actual)
      users.insert(tOp);
  }
}

static void updateLiveOpsInOneBlock(Channel *TheCh, OperationListT &liveOps) {
  assert(TheCh->channelKind == DataChannelKind::TMEMPost ||
         TheCh->channelKind == DataChannelKind::SMEMPost);
  Operation *src = TheCh->getSrcOp();
  SmallVector<Operation *> dsts;
  TheCh->getDstOps(dsts);
  Operation *lastDst = TheCh->getDstOpLast();
  // Assuming they are in the same block, insert ops from src to dsts.
  auto *block = src->getBlock();
  bool foundStart = false;
  for (auto &op : block->getOperations()) {
    if (&op == src) {
      foundStart = true;
      liveOps.push_back(&op);
      continue;
    }
    if (foundStart)
      liveOps.push_back(&op);
    if (&op == lastDst) {
      break;
    }
  }
}

static void updateLiveOpsAcrossScopes(DenseSet<Operation *> &users,
                                      OperationListT &liveOps) {
  DenseSet<Operation *> userScopes; // users in the same scope
  bool first = true;
  for (auto user : users) {
    if (first) {
      userScopes.insert(user);
    } else {
      // We may need to lift the scopes in userScopes.
      auto *scope = *(userScopes.begin());
      // If we can reach the same scope when lifting up "scope", return the
      // lifted "scope". Otherwise, we can lift up "user" to be in the same
      // scope as "scope", return scope.
      auto *sameLevel = getSameLevelOp(user, scope);
      if (sameLevel != scope) {
        // user stays unchanged, scope gets lifted to sameLevel.
        userScopes.clear();
        userScopes.insert(sameLevel);
        userScopes.insert(user);
      } else {
        // scope stays unchanged, user gets lifted.
        userScopes.insert(getSameLevelOp(scope, user));
      }
    }
    first = false;
  }
  // Find the block that contains all users
  bool foundStart = false;
  auto *scope = *(userScopes.begin());
  Operation *lastDst = nullptr;
  for (auto &op : scope->getBlock()->getOperations()) {
    if (userScopes.count(&op)) {
      lastDst = &op;
    }
  }
  for (auto &op : scope->getBlock()->getOperations()) {
    if (userScopes.count(&op) || foundStart) {
      foundStart = true;
      // Goes through nested regions.
      op.walk<WalkOrder::PostOrder>(
          [&](Operation *nestedOp) { liveOps.push_back(nestedOp); });
    }
    if (&op == lastDst) {
      break;
    }
  }
}

namespace triton {
// A simplified version of AllocationAnalysis.
class MemoryPlanner {
public:
  MemoryPlanner(Operation *operation, Allocation *allocation,
                SmallVector<Channel *> *channels)
      : operation(operation), allocation(allocation), channels(channels) {}

private:
  using BufferT = Allocation::BufferT;
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;
  Operation *operation;
  Allocation *allocation;
  SmallVector<Channel *> *channels;
  BufferRangeMapT bufferRange;

  void getExplicitValueSize(Operation *op) {
    auto alloc = dyn_cast<ttg::LocalAllocOp>(op);
    if (!alloc || !alloc.isSharedMemoryAlloc())
      return;
    auto allocType = alloc.getType();
    int64_t numElems = 0;
    if (auto paddedEnc =
            dyn_cast<ttg::PaddedSharedEncodingAttr>(allocType.getEncoding())) {
      SmallVector<int64_t> unpaddedShape = ttg::getShapePerCTA(allocType);
      numElems = paddedEnc.getPaddedSize(unpaddedShape);
    } else {
      auto shapePerCTA = ttg::getAllocationShapePerCTA(allocType);
      numElems = product<int64_t>(shapePerCTA);
    }
    int64_t bytes = numElems * allocType.getElementTypeBitWidth() / 8;

    auto alignment = alloc.getAlignmentOrDefault();
    allocation->addBuffer<BufferT::BufferKind::Explicit>(alloc, bytes,
                                                         alignment);
  }

  void getValuesAndSizes() {
    // Get the alloc values
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { getExplicitValueSize(op); });
  }

  void resolveExplicitBufferLiveness(
      function_ref<Interval<size_t>(Value value)> getLiveness) {
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto value = valueBufferIter.first;
      auto *buffer = valueBufferIter.second;
      bufferRange[buffer] = getLiveness(value);
      LLVM_DEBUG({
        llvm::dbgs() << "-- buffer " << buffer->id << "; value: ";
        value.dump();
      });
    }
  }

  OperationListT livenessForSmemChannel(Value value) {
    // Find the channel for value in channels.
    ChannelPost *TheCh =
        static_cast<ChannelPost *>(findChannelForAlloc(value, *channels));
    std::vector<Operation *> liveOps;
    DenseSet<Operation *> users;
    getAllAcutalUsersForChannel(TheCh, users);
    updateLiveOpsAcrossScopes(users, liveOps);
    return liveOps;
  }

  void resolveLiveness() {
    DenseMap<Operation *, size_t> operationId;
    operation->walk<WalkOrder::PostOrder>([&](Operation *op) {
      LLVM_DEBUG(
          op->setAttr("operation_id",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                       operationId.size())));
      operationId[op] = operationId.size();
    });

    // Analyze liveness of explicit buffers
    Liveness liveness(operation);
    auto getValueLivenessRange = [&](Value value) {
      Operation *defOp = value.getDefiningOp();
      LLVM_DEBUG({
        llvm::dbgs() << "-- getValueLivenessRange \n";
        value.dump();
      });
      auto liveOperations = livenessForSmemChannel(value);
      auto minId = std::numeric_limits<size_t>::max();
      auto maxId = std::numeric_limits<size_t>::min();
      llvm::for_each(liveOperations, [&](Operation *liveOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "---- liveOp " << operationId[liveOp] << "\n");
        if (defOp && isa<mlir::triton::gpu::WarpSpecializeOp>(defOp)) {
          minId = 0;
          maxId = operationId.size();
          return;
        }
        if (operationId[liveOp] < minId) {
          minId = operationId[liveOp];
        }
        if ((operationId[liveOp] + 1) > maxId) {
          maxId = operationId[liveOp] + 1;
        }
      });
      return Interval(minId, maxId);
    };

    resolveExplicitBufferLiveness(getValueLivenessRange);
  }

public:
  unsigned run(unsigned numBuffers) {
    getValuesAndSizes();
    resolveLiveness();
    // Try to set buffer.copy, buffer.id, heuristics: for channels in innermost
    // loop, set to maxStage Make sure the configuration will fit in SMEM.
    // FIXME: reuse for buffers in inner most loop, set copy to numBuffers.
    unsigned bufferId = 0;
    int bufferIdInnermost = -1;
    auto usedInnermostLoop = [&](Operation *alloc) -> bool {
      ChannelPost *TheCh =
          static_cast<ChannelPost *>(findChannelForOp(alloc, *channels));
      DenseSet<Operation *> users;
      getAllAcutalUsersForChannel(TheCh, users);
      // All users are in the same block and in the innermost loop.
      auto *first = *(users.begin());
      for (auto *user : users) {
        if (user->getBlock() != first->getBlock())
          return false;
      }
      return isInnermostLoop(first->getParentOfType<scf::ForOp>());
    };
    for (auto bufferIter : bufferRange) {
      Operation *owner = bufferIter.first->owner;
      if (usedInnermostLoop(owner)) {
        if (bufferIdInnermost < 0) {
          bufferIdInnermost = bufferId;
          ++bufferId;
        }
        owner->setAttr(
            "buffer.id",
            IntegerAttr::get(IntegerType::get(owner->getContext(), 32),
                             bufferIdInnermost));
        // FIXME: heuristics
        owner->setAttr(
            "buffer.copy",
            IntegerAttr::get(IntegerType::get(owner->getContext(), 32),
                             numBuffers));
      } else {
        owner->setAttr(
            "buffer.id",
            IntegerAttr::get(IntegerType::get(owner->getContext(), 32),
                             bufferId));
        // FIXME: heuristics
        owner->setAttr(
            "buffer.copy",
            IntegerAttr::get(IntegerType::get(owner->getContext(), 32), 1));
        ++bufferId;
      }
    }
    return bufferId;
  }
  void dumpBuffers() const {
    LDBG("Dump bufferRange: id size offset ---------");
    for (auto bufferIter : bufferRange) {
      llvm::dbgs() << "-- " << bufferIter.first->id << " "
                   << bufferIter.first->size << " " << bufferIter.first->offset;
      llvm::dbgs() << " interval " << bufferIter.second.start() << " "
                   << bufferIter.second.end() << "\n";
      bufferIter.first->owner->dump();
    }
  }
};
} // namespace triton

static void handleOperandD(ttng::TMEMAllocOp tmemAllocOp,
                           std::vector<Operation *> &liveOps) {
  DenseSet<Operation *> users;
  for (auto user : tmemAllocOp.getResult().getUsers()) {
    users.insert(user);
  }
  updateLiveOpsAcrossScopes(users, liveOps);
}

// Return the list of operations where value is live.
OperationListT livenessForTmemChannel(Value value,
                                      SmallVector<Channel *> &channels) {
  // Find the channel for value in channels.
  ttng::TmemDataChannelPost *TheCh = static_cast<ttng::TmemDataChannelPost *>(
      findChannelForAlloc(value, channels));
  std::vector<Operation *> liveOps;
  // Operand D can be associated with multiple channels. From first producer to
  // last consumer.
  if (TheCh->isOperandD) {
    handleOperandD(cast<ttng::TMEMAllocOp>(TheCh->getAllocOp()), liveOps);
  } else {
    DenseSet<Operation *> users;
    getAllAcutalUsersForChannel(TheCh, users);
    updateLiveOpsAcrossScopes(users, liveOps);
  }
  return liveOps;
}

// Copied from TensorMemoryAllocation.cpp
static Interval<int> getLiveIntervals(Value value, Liveness &liveness,
                                      DenseMap<Operation *, int> &operationId,
                                      SmallVector<Channel *> &channels) {
  auto liveOperations = livenessForTmemChannel(value, channels);
  // Merge the alloc liverange with the liverange of any subview of the
  // allocation.
  SmallVector<Operation *> users(value.getUsers());
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    if (!isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp>(user))
      continue;
    auto usersLivness = livenessForTmemChannel(user->getResult(0), channels);
    liveOperations.insert(liveOperations.end(), usersLivness.begin(),
                          usersLivness.end());
    users.append(user->getResult(0).getUsers().begin(),
                 user->getResult(0).getUsers().end());
  }
  auto minId = std::numeric_limits<int>::max();
  auto maxId = std::numeric_limits<int>::min();
  std::for_each(liveOperations.begin(), liveOperations.end(),
                [&](Operation *liveOp) {
                  if (operationId[liveOp] < minId) {
                    minId = operationId[liveOp];
                  }
                  if ((operationId[liveOp] + 1) > maxId) {
                    maxId = operationId[liveOp] + 1;
                  }
                });
  return Interval(minId, maxId);
}

static void allocateTMem(Operation *parentOp, SmallVector<Channel *> &channels,
                         unsigned bufferId) {
  SmallVector<triton::nvidia_gpu::TMEMAllocOp> allocs;
  DenseMap<Operation *, int> operationId;
  // Only consider allocs for channels.
  parentOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    operationId[op] = operationId.size();
    if (auto alloc = dyn_cast<triton::nvidia_gpu::TMEMAllocOp>(op)) {
      allocs.push_back(alloc);
    }
  });
  Liveness liveness(parentOp);
  DenseMap<Operation *, Interval<int>> allocToIntervals;
  DenseMap<Operation *, ttng::TMemAllocation> allocToSize;
  DenseMap<Operation *, ttng::TmemDataChannelPost *> allocToChannel;
  for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
    ttng::TMEMAllocOp alloc = *it;
    Interval<int> liveInterval =
        getLiveIntervals(alloc, liveness, operationId, channels);
    auto memDescType = alloc.getType();
    ttng::TMemAllocation allocSize = ttng::getTmemAllocSizes(memDescType);
    LLVM_DEBUG(alloc.dump());
    LDBG("tmem livenss: " << liveInterval.start() << " " << liveInterval.end());
    LDBG("tmem allocSize: " << allocSize.numCols << " " << allocSize.numRows);

    ttng::TmemDataChannelPost *TheCh = static_cast<ttng::TmemDataChannelPost *>(
        findChannelForAlloc(alloc, channels));
    if (TheCh->isOperandD)
      // Heuristics: do not reuse
      allocToIntervals[alloc.getOperation()] =
          Interval(0, (int)operationId.size());
    else
      allocToIntervals[alloc.getOperation()] = liveInterval;
    allocToSize.insert(
        {alloc.getOperation(),
         ttng::TMemAllocation(allocSize.numCols, allocSize.numRows)});
    allocToChannel[alloc.getOperation()] = TheCh;
  }
  sort(allocs, [&](ttng::TMEMAllocOp a, ttng::TMEMAllocOp b) {
    auto iter1 = allocToSize.find(a.getOperation());
    auto iter2 = allocToSize.find(b.getOperation());
    if (iter1->second.numRows == iter2->second.numRows)
      return iter1->second.numCols > iter2->second.numCols;
    if (iter1->second.numCols == iter2->second.numCols)
      return iter1->second.numRows > iter2->second.numRows;
    assert(false);
  });
  // Heuristics: one copy for each alloc
  // If liveness overlaps, we can't reuse the buffer.
  // Heuristics:
  // - no reuse if isOperandD is true
  // - allocate space for channels where isOperandDNoAcc or isOperandD is true
  // - extend live ranges for these channels in bufferSet
  // Sort allocs according to allocSize, try to allocate space according to
  // the sorted allocs, for each candidate alloc, decide if reuse is needed to
  // fit into TMEM.
  DenseMap<Operation *, Interval<int>> bufferSet;
  Operation *candidateAlloc = nullptr;
  DenseMap<Operation *, unsigned> allocToOffsets;
  SmallVector<Operation *> allocOrder;
  for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
    ttng::TMEMAllocOp alloc = *it;
    if (allocToChannel[alloc.getOperation()]->isOperandD ||
        allocToChannel[alloc.getOperation()]->isOperandDNoAcc) {
      bufferSet[alloc.getOperation()] = Interval(0, (int)operationId.size());
      alloc->setAttr("buffer.id",
                     IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                                      bufferId));
      // FIXME: heuristics
      alloc->setAttr(
          "buffer.copy",
          IntegerAttr::get(IntegerType::get(alloc->getContext(), 32), 1));
      allocToOffsets[alloc.getOperation()] = 0;
      allocOrder.push_back(alloc.getOperation());
      bufferId++;
    } else if (!candidateAlloc) {
      candidateAlloc = alloc.getOperation();
    }
  }
  auto findReuseChannel = [&](Operation *cand) -> Operation * {
    // Go through allocs with buffer.id (i.e allocated), check intervals
    // to find an allocated alloc without overlapping intervals and with enough
    // space.
    for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
      Operation *alloc = (*it).getOperation();
      if (allocToOffsets.count(alloc)) {
        auto iter1 = allocToSize.find(cand);
        auto iter2 = allocToSize.find(alloc);
        if (!allocToIntervals[alloc].intersects(allocToIntervals[cand]) &&
            iter2->second.numCols >=
                iter1->second.numCols + allocToOffsets[alloc]) {
          allocToOffsets[alloc] += iter1->second.numCols;
          return alloc;
        }
      }
    }
    return nullptr;
  };
  auto getBufferId = [&](Operation *op) -> int {
    auto stageAttr = op->getAttrOfType<IntegerAttr>("buffer.id");
    return stageAttr.getInt();
  };
  int totalMemorySize = ttng::allocateTMemWithInterval(bufferSet, allocOrder);
  LDBG(bufferSet.size() << " buffers with tmem size: " << totalMemorySize);
  if (totalMemorySize > 512)
    return;
  while (bufferSet.size() != allocs.size()) {
    // Decide if we need to reuse buffer for candidateAlloc.
    // Choose an interval for candidateAlloc based on the decision.
    LLVM_DEBUG({
      LDBG("try to allocate space for:");
      candidateAlloc->dump();
    });
    bufferSet[candidateAlloc] = Interval(0, (int)operationId.size());
    allocOrder.push_back(candidateAlloc);
    totalMemorySize = ttng::allocateTMemWithInterval(bufferSet, allocOrder);
    LDBG(bufferSet.size() << " buffers with tmem size: " << totalMemorySize);
    // FIXME: there are some issues in allocateTMemWithInterval where
    // it allocates overlapping address even though there is live range
    // intersect.
    if (false) { // totalMemorySize <= 512) {
      candidateAlloc->setAttr(
          "buffer.id",
          IntegerAttr::get(IntegerType::get(candidateAlloc->getContext(), 32),
                           bufferId));
      allocToOffsets[candidateAlloc] = 0;
      LLVM_DEBUG(candidateAlloc->dump());
      ++bufferId;
    } else {
      // need to reuse buffer, heuristics: do not reuse isOperandD channels
      // if can't find a reuse buffer, bail out
      auto *reuseAlloc = findReuseChannel(candidateAlloc);
      if (!reuseAlloc)
        assert(false && "can't find space");
      LLVM_DEBUG({
        LDBG("try to reuse space with:");
        reuseAlloc->dump();
      });
      // update intervals for representative channel and this channel so they
      // will not overlap
      auto origIntv = bufferSet[reuseAlloc];
      bufferSet[reuseAlloc] = Interval(origIntv.start(), origIntv.end() - 2);
      bufferSet[candidateAlloc] = Interval(origIntv.end() - 2, origIntv.end());
      totalMemorySize = ttng::allocateTMemWithInterval(bufferSet, allocOrder);
      LDBG(bufferSet.size() << " buffers with tmem size: " << totalMemorySize);

#if 0
      if (totalMemorySize > 512) {
        assert(false && "can't find space");
      }
#endif
      candidateAlloc->setAttr(
          "buffer.id",
          IntegerAttr::get(IntegerType::get(candidateAlloc->getContext(), 32),
                           getBufferId(reuseAlloc)));
      auto iter2 = allocToSize.find(candidateAlloc);
      candidateAlloc->setAttr(
          "buffer.offset",
          IntegerAttr::get(IntegerType::get(candidateAlloc->getContext(), 32),
                           allocToOffsets[reuseAlloc] - iter2->second.numCols));
      LLVM_DEBUG(candidateAlloc->dump());
    }
    for (auto *op_t : allocOrder) {
      LLVM_DEBUG(op_t->dump());
      LLVM_DEBUG(llvm::dbgs() << bufferSet[op_t].start() << " "
                              << bufferSet[op_t].end() << "\n");
    }
    // FIXME: heuristics
    candidateAlloc->setAttr(
        "buffer.copy",
        IntegerAttr::get(IntegerType::get(candidateAlloc->getContext(), 32),
                         1));
    // Find the next candidate.
    for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it)
      if (!bufferSet.count((*it).getOperation())) {
        candidateAlloc = (*it).getOperation();
        break;
      }
  }
}

void doMemoryPlanner(triton::FuncOp &funcOp, unsigned numBuffers) {

  // Step 1: collect all communications between producers and consumers.
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectPostChannels(channelsOrigin, funcOp);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    channels.push_back(c.get());
  }
  if (channels.empty()) {
    return;
  }
  // Step 2: figure out smem/tmem sizes and liveness.
  // If two buffers are sharing a multi-staged alloc, the liveness can overlap,
  // otherwise, the liveness can't overlap.
  Allocation allocation;
  triton::MemoryPlanner planner(funcOp, &allocation, &channels);
  unsigned bufferId = planner.run(numBuffers);
  LLVM_DEBUG(planner.dumpBuffers());
  allocateTMem(funcOp, channels, bufferId);
}

#define GEN_PASS_DEF_NVGPUTESTWSMEMORYPLANNER
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSMemoryPlannerPass
    : public impl::NVGPUTestWSMemoryPlannerBase<NVGPUTestWSMemoryPlannerPass> {
public:
  using impl::NVGPUTestWSMemoryPlannerBase<
      NVGPUTestWSMemoryPlannerPass>::NVGPUTestWSMemoryPlannerBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numBuffers >= 1)
      doMemoryPlanner(funcOp, numBuffers);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
