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
  return findChannelForOp(value.getDefiningOp(), channels);
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

// Lift up scope of b till it contains a.
static Operation *getLiftedScope(Operation *a, Operation *b) {
  DenseSet<Operation *> parentScopes;
  Operation *op = a;
  while (!isa<triton::FuncOp>(op)) {
    parentScopes.insert(op);
    op = op->getParentOp();
  }
  op = b;
  while (!isa<triton::FuncOp>(op)) {
    if (parentScopes.count(op))
      return op;
    op = op->getParentOp();
  }
  return nullptr;
}

// Return a list of users under the same scope, original users
// will be lifted up.
static void getUserScopes(DenseSet<Operation *> &users,
                          DenseSet<Operation *> &userScopes) {
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
      if (sameLevel && sameLevel != scope) {
        // user stays unchanged, scope gets lifted to sameLevel.
        userScopes.clear();
        userScopes.insert(sameLevel);
        userScopes.insert(user);
      } else if (sameLevel) {
        // scope stays unchanged, user gets lifted.
        userScopes.insert(getSameLevelOp(scope, user));
      } else { // user and scope in different blocks, lift both.
        // find the parent scope that include both scope and user
        auto *parentScope = getLiftedScope(scope, user);
        userScopes.clear();
        assert(parentScope);
        Operation *op = user;
        Operation *liftedUser = nullptr;
        while (!isa<triton::FuncOp>(op)) {
          if (op->getParentOp() == parentScope) {
            liftedUser = op;
            break;
          }
          op = op->getParentOp();
        }
        assert(liftedUser);
        userScopes.insert(liftedUser);
        op = scope;
        Operation *liftedScope = nullptr;
        while (!isa<triton::FuncOp>(op)) {
          if (op->getParentOp() == parentScope) {
            liftedScope = op;
            break;
          }
          op = op->getParentOp();
        }
        assert(liftedScope);
        userScopes.insert(liftedScope);
      }
    }
    first = false;
  }
}

static void updateLiveOpsAcrossScopes(DenseSet<Operation *> &users,
                                      OperationListT &liveOps) {
  DenseSet<Operation *> userScopes;
  getUserScopes(users, userScopes);
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
    // Keep track of unique element types. We don't support casting between
    // different element types.
    DenseMap<int, Type> idTypes;
    for (auto bufferIter : bufferRange) {
      Operation *owner = bufferIter.first->owner;
      auto sAlloc = cast<ttg::LocalAllocOp>(owner);
      auto aType = sAlloc.getType();
      auto allocDescType = cast<triton::gpu::MemDescType>(aType);
      auto elemType = aType.getElementType();
      // FIXME: reuse for buffers in inner most loop, set copy to numBuffers,
      // when the shape is 2D.
      unsigned numD = 0;
      for (int shape : allocDescType.getShape()) {
        if (shape > 1)
          ++numD;
      }
      if (usedInnermostLoop(owner) && numD >= 2) {
        if (bufferIdInnermost < 0) {
          bufferIdInnermost = bufferId;
          ++bufferId;
        }
        if (idTypes.count(bufferIdInnermost) == 0) {
          idTypes[bufferIdInnermost] = elemType;
        }
        if (idTypes[bufferIdInnermost] != elemType) {
          bufferIdInnermost = bufferId;
          idTypes[bufferIdInnermost] = elemType;
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
        if (idTypes.count(bufferId) == 0) {
          idTypes[bufferId] = elemType;
        }
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

static void getAllTmemUsers(ttng::TmemDataChannelPost *TheCh,
                            DenseSet<Operation *> &users) {
  ttng::TMEMAllocOp tmemAllocOp = cast<ttng::TMEMAllocOp>(TheCh->getAllocOp());
  if (TheCh->isOperandD) {
    for (auto user : tmemAllocOp.getResult().getUsers()) {
      users.insert(user);
    }
  } else {
    getAllAcutalUsersForChannel(TheCh, users);
  }
}

// Return the list of operations where value is live.
OperationListT livenessForTmemChannel(Value value,
                                      SmallVector<Channel *> &channels) {
  // Find the channel for value in channels.
  ttng::TmemDataChannelPost *TheCh = static_cast<ttng::TmemDataChannelPost *>(
      findChannelForAlloc(value, channels));
  std::vector<Operation *> liveOps;
  DenseSet<Operation *> users;
  getAllTmemUsers(TheCh, users);
  updateLiveOpsAcrossScopes(users, liveOps);

  return liveOps;
}

namespace triton {
// A simplified version of AllocationAnalysis.
class MemoryPlannerTmem {
public:
  MemoryPlannerTmem(Operation *operation, Allocation *allocation,
                    SmallVector<Channel *> *channels)
      : operation(operation), allocation(allocation), channels(channels) {}

private:
  using BufferT = Allocation::BufferT;
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<int>>;
  using GraphT = DenseMap<BufferT *, DenseSet<BufferT *>>;
  Operation *operation;
  Allocation *allocation;
  SmallVector<Channel *> *channels;
  BufferRangeMapT bufferRange;

  // Copied from TensorMemoryAllocation.cpp
  Interval<int> getLiveIntervals(Value value, Liveness &liveness,
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

  Interval<int> getIntervalForCtrlOp(Operation *ctrlOp,
                                     DenseMap<Operation *, int> &operationId) {
    // get the operationId of the first instruction
    auto forOp = cast<scf::ForOp>(ctrlOp);
    for (Operation &op : forOp.getBody()->without_terminator()) {
      return Interval(operationId[&op], operationId[ctrlOp]);
    }
    llvm_unreachable("getIntervalForCtrlOp");
  }

  // Return number of outer loops.
  unsigned getLoopDepth(Operation *op) {
    unsigned depth = 0;
    auto pOp = op->getParentOfType<scf::ForOp>();
    while (pOp) {
      ++depth;
      pOp = pOp->getParentOfType<scf::ForOp>();
    }
    return depth;
  }

  void buildInterferenceGraph(const SmallVector<BufferT *> &buffers,
                              GraphT &interference) {
    // Reset interference graph
    interference.clear();
    for (auto x : buffers) {
      for (auto y : buffers) {
        if (x == y)
          continue;
        auto xStart = x->offset;
        auto yStart = y->offset;
        auto xSize = x->size;
        auto ySize = y->size;
        Interval xSizeRange = {xStart, xStart + xSize};
        Interval ySizeRange = {yStart, yStart + ySize};
        auto xOpRange = bufferRange.lookup(x);
        auto yOpRange = bufferRange.lookup(y);

        // Buffers interfere if their allocation offsets overlap and they are
        // live at the same time.
        if (xOpRange.intersects(yOpRange) &&
            xSizeRange.intersects(ySizeRange)) {
          interference[x].insert(y);
        }

        // Buffers also interfere if their allocation offsets overlap and they
        // exist within regions that may execute simultaneously with respect to
        // each other.
        auto wsx = x->owner->getParentWithTrait<OpTrait::AsyncRegions>();
        auto wsy = y->owner->getParentWithTrait<OpTrait::AsyncRegions>();
        if (wsx && wsy && wsx == wsy &&
            x->owner->getParentRegion() != y->owner->getParentRegion() &&
            xSizeRange.intersects(ySizeRange)) {
          interference[x].insert(y);
        }
      }
    }

    // LLVM_DEBUG(dumpInterferenceGraph(interference));
  }

public:
  void run(unsigned bufferId) {
    Operation *parentOp = operation;
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
          getLiveIntervals(alloc, liveness, operationId, *channels);
      auto memDescType = alloc.getType();
      ttng::TMemAllocation allocSize = ttng::getTmemAllocSizes(memDescType);
      LLVM_DEBUG(alloc.dump());
      LDBG("tmem liveness: " << liveInterval.start() << " "
                             << liveInterval.end());
      LDBG("tmem allocSize: " << allocSize.numCols << " " << allocSize.numRows);

      ttng::TmemDataChannelPost *TheCh =
          static_cast<ttng::TmemDataChannelPost *>(
              findChannelForAlloc(alloc, *channels));
      allocToIntervals[alloc.getOperation()] = liveInterval;
      allocToSize.insert(
          {alloc.getOperation(),
           ttng::TMemAllocation(allocSize.numCols, allocSize.numRows)});
      allocToChannel[alloc.getOperation()] = TheCh;
    }
    // Sort allocs according to isOperandD, size, live interval.
    // This can be adjusted later on.
    sort(allocs, [&](ttng::TMEMAllocOp a, ttng::TMEMAllocOp b) {
      ttng::TmemDataChannelPost *aCh = static_cast<ttng::TmemDataChannelPost *>(
          findChannelForAlloc(a, *channels));
      ttng::TmemDataChannelPost *bCh = static_cast<ttng::TmemDataChannelPost *>(
          findChannelForAlloc(b, *channels));
      if (aCh->isOperandD && !bCh->isOperandD)
        return true;
      if (bCh->isOperandD && !aCh->isOperandD)
        return false;
      auto iter1 = allocToSize.find(a.getOperation());
      auto iter2 = allocToSize.find(b.getOperation());
      if (iter1->second.numRows == iter2->second.numRows &&
          iter1->second.numCols == iter2->second.numCols) {
        // check live interval length and offset.
        auto intv1 = allocToIntervals[a.getOperation()];
        auto intv2 = allocToIntervals[b.getOperation()];
#if 0
        // larger interval has higher priority
        if (intv1.size() > intv2.size())
          return true;
        if (intv1.size() < intv2.size())
          return false;
#endif
        // early interval has higher priority
        if (intv1.start() < intv2.start())
          return true;
        if (intv1.start() > intv2.start())
          return false;
        assert(false);
      }
      if (iter1->second.numRows == iter2->second.numRows)
        return iter1->second.numCols > iter2->second.numCols;
      if (iter1->second.numCols == iter2->second.numCols)
        return iter1->second.numRows > iter2->second.numRows;
      assert(false);
    });
    // Set up Vector<BufferT>, build interference graph.
    Allocation allocation;
    SmallVector<BufferT *> buffers;
    for (auto alloc : allocs) {
      // size is 0, alignment is default, offset is default
      allocation.addBuffer<BufferT::BufferKind::Explicit>(alloc, 0);
      BufferT *tBuf = allocation.valueBuffer[alloc];
      auto iter1 = allocToSize.find(alloc.getOperation());
      tBuf->rowSize = iter1->second.numRows;
      tBuf->colSize = iter1->second.numCols;
      tBuf->rowOffset = std::numeric_limits<size_t>::max();
      tBuf->colOffset = std::numeric_limits<size_t>::max();
      buffers.emplace_back(tBuf);
    }
    for (auto valueBufferIter : allocation.valueBuffer) {
      auto *buffer = valueBufferIter.second;
      // valueBuffer maps value to BufferT
      Operation *alloc = valueBufferIter.first.getDefiningOp();
      // bufferRange maps BufferT to interval
      bufferRange[buffer] = allocToIntervals[alloc];
    }
    GraphT interference;
    buildInterferenceGraph(buffers, interference);
    // For each innermost loop according to program order (via
    // getIntervalForCtrlOp)
    //   Go through all buffers that are live in the loop
    //   Start with buffers with longest span within the loop
    //   For each buffer
    //     either allocate new space (owner of a set of rows)
    //     or reuse an existing buffer's space
    //     if this buffer interferes with all allocated buffers, allocate new
    //     space if this buffer is along the dependency chain, reuse space if
    //     there is enough space, allocate new space otherwise, reuse space

    // Use BufferT to track rowSize/colSize/rowOffset etc, use bufferRange to
    // track intervals.
    SmallVector<Operation *> innermostLoops;
    parentOp->walk([&](Operation *subOp) {
      if (auto theForOp = dyn_cast<scf::ForOp>(subOp))
        if (isInnermostLoop(theForOp))
          innermostLoops.push_back(subOp);
    });
    DenseSet<Operation *> handledAllocs;
    for (auto *ctrlOp : innermostLoops) {
      SmallVector<triton::nvidia_gpu::TMEMAllocOp> allocsForThisLoop;
      unsigned allocIdx = 0;
      auto ctrlInt = getIntervalForCtrlOp(ctrlOp, operationId);
      for (auto alloc : allocs) {
        auto allocInt = bufferRange.lookup(buffers[allocIdx]);
        ++allocIdx;
        if (ctrlInt.intersects(allocInt)) {
          allocsForThisLoop.push_back(alloc);
          handledAllocs.insert(alloc.getOperation());
        }
      }
      LDBG("run allocation on innermost loop "
           << allocsForThisLoop.size() << " allocs " << ctrlInt.start() << " "
           << ctrlInt.end());
      bufferId = allocateTMemAllocs(
          allocsForThisLoop, buffers, // allocToIntervals,
          /*allocToSize,*/ allocToChannel, operationId, ctrlOp, bufferId);
    }
    SmallVector<triton::nvidia_gpu::TMEMAllocOp> lastAllocs;
    for (auto alloc : allocs) {
      if (handledAllocs.count(alloc))
        continue;
      lastAllocs.push_back(alloc);
    }
    if (!lastAllocs.empty()) {
      bufferId = allocateTMemAllocs(lastAllocs, buffers, // allocToIntervals,
                                    /*allocToSize,*/ allocToChannel,
                                    operationId, nullptr, bufferId);
    }
  }

  unsigned allocateTMemAllocs(
      SmallVector<triton::nvidia_gpu::TMEMAllocOp> &allocs,
      SmallVector<BufferT *> &buffers,
      DenseMap<Operation *, ttng::TmemDataChannelPost *> &allocToChannel,
      DenseMap<Operation *, int> &operationId, Operation *ctrlOp,
      unsigned bufferId) {
    auto alongDependencyChain = [&](Operation *src, Operation *dst) -> bool {
      // consumer of srcAlloc --> producer of dstAlloc
      // consumer partition of srcAllc vs. producer partition of dstAlloc
      auto *srcCh = allocToChannel[src];
      auto *dstCh = allocToChannel[dst];
      if (getAsyncTaskIds(dstCh->getSrcOp()) ==
          getAsyncTaskIds(srcCh->getDstOp()))
        return true;
      return false;
    };
    auto sameLoop = [&](BufferT *alloc, BufferT *cand) -> bool {
      if (ctrlOp) {
        auto ctrlInt = getIntervalForCtrlOp(ctrlOp, operationId);
        if (bufferRange[alloc].intersects(ctrlInt))
          return false;
        return true; // no overlapping for alloc and the current ctrlOp
      }
      return false;
    };
    auto getCombinedTasks = [&](BufferT *alloc) -> SmallVector<AsyncTaskId> {
      ttng::TmemDataChannelPost *TheCh =
          static_cast<ttng::TmemDataChannelPost *>(
              findChannelForOp(alloc->owner, *channels));
      DenseSet<Operation *> users;
      getAllTmemUsers(TheCh, users);
      SmallVector<AsyncTaskId> combinedTasks;
      DenseSet<AsyncTaskId> combinedSet;
      for (auto *user : users) {
        auto asyncTasksVec = getAsyncTaskIds(user);
        for (auto t : asyncTasksVec) {
          if (!combinedSet.count(t)) {
            combinedSet.insert(t);
            combinedTasks.push_back(t);
          }
        }
      }
      return combinedTasks;
    };
    auto samePartition = [&](BufferT *alloc, BufferT *cand) -> bool {
      auto aTasks = getCombinedTasks(alloc);
      auto bTasks = getCombinedTasks(cand);
      return aTasks == bTasks;
    };

    auto findReuseChannel = [&](BufferT *cand, bool relax) -> BufferT * {
      for (auto *alloc : buffers) {
        if (alloc->isOwnerOfSpace) {
          // The buffer owner owns a set of rows.
          // Each reuseOffset has at most one other buffer sharing the space.
          // If alloc and cand are in different loops, we can reuse as
          // long as they have the same partitions.
          // Otherwise, reuse when there is a dependency chain.
          // if relax is true, ignore the constraints.
          if (!bufferRange[alloc].intersects(bufferRange[cand]) &&
              alloc->colSize >= cand->colSize + alloc->reuseOffset &&
              (relax ||
               (!sameLoop(alloc, cand) && samePartition(alloc, cand)) ||
               (sameLoop(alloc, cand) &&
                alongDependencyChain(alloc->owner, cand->owner)))) {
            cand->isOwnerOfSpace = false; // redundant with reuseOwner?
            cand->rowOffset = alloc->rowOffset;
            cand->colOffset = alloc->reuseOffset;
            cand->reuseOwner = alloc;
            // Move reuseOffset for the buffer owner.
            alloc->reuseOffset = cand->colOffset + cand->colSize;
            LLVM_DEBUG({
              LDBG("set offset to " << cand->rowOffset << " " << cand->colOffset
                                    << " sameLoop " << sameLoop(alloc, cand)
                                    << " releax " << relax << ":");
              cand->owner->dump();
            });
            return alloc;
          }
        }
      }
      return nullptr;
    };
    // interferes with all allocated buffers
    auto allInterfere = [&](BufferT *cand) -> bool {
      for (auto *alloc : buffers) {
        if (alloc->rowOffset != std::numeric_limits<size_t>::max()) {
          if (!bufferRange[alloc].intersects(bufferRange[cand]))
            return false;
        }
      }
      return true;
    };
    auto allocateNewSpace = [&](BufferT *cand, bool allocate) -> bool {
      size_t maxRowOffset = 0;
      for (auto *alloc : buffers) {
        if (alloc->rowOffset != std::numeric_limits<size_t>::max()) {
          maxRowOffset =
              std::max(maxRowOffset, alloc->rowOffset + alloc->rowSize);
          LLVM_DEBUG({
            LDBG("\nbuffer is allocated "
                 << alloc->rowOffset << " " << alloc->rowSize << " "
                 << alloc->colOffset << " " << alloc->isOwnerOfSpace);
            alloc->owner->dump();
          });
        }
      }
      if (allocate) {
        cand->rowOffset = maxRowOffset;
        cand->colOffset = 0;
        cand->isOwnerOfSpace = true;
        cand->reuseOffset = 0;
        cand->reuseOwner = cand;
        cand->owner->setAttr(
            "buffer.id",
            IntegerAttr::get(IntegerType::get(cand->owner->getContext(), 32),
                             bufferId));
        ++bufferId;
      }
      if (maxRowOffset + cand->rowSize > 512)
        return false;
      return true;
    };
    auto getBufferId = [&](Operation *op) -> int {
      auto stageAttr = op->getAttrOfType<IntegerAttr>("buffer.id");
      return stageAttr.getInt();
    };
    auto getBuffer = [&](Operation *candAlloc) -> BufferT * {
      for (auto *alloc : buffers) {
        if (alloc->owner == candAlloc)
          return alloc;
      }
      return nullptr;
    };

    // Heuristics: num_buffers is one for each alloc
    // If liveness overlaps, we can't reuse the buffer.
    // Heuristics:
    //   if this buffer interferes with all allocated buffers, allocate new
    //   space; reuse buffers
    //   if belongs to the same loop and along the dependency chain
    //   or belongs to different loops and have the same partitions
    //   if there is enough space, allocate new space otherwise, reuse space
    DenseMap<Operation *, Interval<int>> bufferSet;
    Operation *candidateAlloc = nullptr;
    SmallVector<Operation *> allocOrder;
    for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
      ttng::TMEMAllocOp alloc = *it;
      auto *candBuf = getBuffer(alloc.getOperation());
      // if this is the first buffer to be allocated, allocate new space.
      // get a list of allocated buffers, check if it interferes
      if (allInterfere(candBuf)) {
        bool hasSpace = allocateNewSpace(candBuf, true);
        assert(hasSpace && "no new space for tmem alloc");
      } else if (auto *reuseBuf = findReuseChannel(candBuf, false)) {
        alloc.getOperation()->setAttr(
            "buffer.id",
            IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                             getBufferId(reuseBuf->owner)));
        alloc.getOperation()->setAttr(
            "buffer.offset",
            IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                             reuseBuf->reuseOffset - candBuf->colSize));
      } else {
        if (allocateNewSpace(candBuf, false))
          allocateNewSpace(candBuf, true);
        else {
          auto *reuseBuf = findReuseChannel(candBuf, true);
          assert(reuseBuf && "can't find tmem space");
          alloc.getOperation()->setAttr(
              "buffer.id",
              IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                               getBufferId(reuseBuf->owner)));
          alloc.getOperation()->setAttr(
              "buffer.offset",
              IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                               reuseBuf->reuseOffset - candBuf->colSize));
        }
      }
      LLVM_DEBUG({
        LDBG("\ntmem allocation " << candBuf->rowOffset << " "
                                  << candBuf->colOffset << " "
                                  << candBuf->isOwnerOfSpace);
        alloc.getOperation()->dump();
      });

      // FIXME: heuristics
      alloc.getOperation()->setAttr(
          "buffer.copy",
          IntegerAttr::get(IntegerType::get(alloc->getContext(), 32), 1));
    }
    return bufferId;
  }
};
} // namespace triton

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
  for (auto *ch : channels) {
    LLVM_DEBUG({
      LDBG("\nchannel with allocOp: " << static_cast<int>(ch->channelKind)
                                      << " " << ch->uniqID << " ");
      ch->getAllocOp()->dump();
    });
    if (ch->channelKind == DataChannelKind::TMEMPost) {
      ttng::TmemDataChannelPost *TheCh =
          static_cast<ttng::TmemDataChannelPost *>(ch);
      LDBG("channel type TMEM" << TheCh->isOperandD << " "
                               << TheCh->isOperandDNoAcc);
    }
  }
  // Step 2: figure out smem/tmem sizes and liveness.
  // If two buffers are sharing a multi-staged alloc, the liveness can overlap,
  // otherwise, the liveness can't overlap.
  Allocation allocation;
  triton::MemoryPlanner planner(funcOp, &allocation, &channels);
  unsigned bufferId = planner.run(numBuffers);
  LLVM_DEBUG(funcOp.dump());
  LLVM_DEBUG(planner.dumpBuffers());
  {
    Allocation allocation;
    triton::MemoryPlannerTmem planner(funcOp, &allocation, &channels);
    planner.run(bufferId);
  }
  // allocateTMem(funcOp, channels, bufferId);
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
