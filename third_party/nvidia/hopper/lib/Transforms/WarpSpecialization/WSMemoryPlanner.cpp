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

#if 0
// For allocs with users inside a ForOp and outside the ForOp under one
// scope, return the ForOp.
Operation *getInnermostCtrl(Value value, SmallVector<Channel *> &channels) {
  ttng::TmemDataChannelPost *TheCh = static_cast<ttng::TmemDataChannelPost *>(
      findChannelForAlloc(value, channels));
  auto tmemAllocOp = cast<ttng::TMEMAllocOp>(TheCh->getAllocOp());
  DenseSet<Operation *> users;
  if (TheCh->isOperandD) {
    for (auto user : tmemAllocOp.getResult().getUsers()) {
      users.insert(user);
    }
  } else {
    getAllAcutalUsersForChannel(TheCh, users);
  }
  DenseSet<Operation *> userScopes;
  getUserScopes(users, userScopes);
  Operation *scope = nullptr;
  unsigned numForOp = 0;
  for (auto *user : userScopes) {
    // if a single ForOp, return it
    if (isa<scf::ForOp>(user)) {
      ++numForOp;
      scope = user;
    }
  }
  // if multiple ForOp, return the parent scope.
  if (numForOp > 1)
    return scope->getParentOp();
  if (numForOp == 0) {
    auto *first = *(userScopes.begin());
    return first->getParentOp();
  }
  return scope;
}
#endif

namespace triton {
// A simplified version of AllocationAnalysis.
class MemoryPlannerTmem {
public:
  MemoryPlannerTmem(Operation *operation, Allocation *allocation,
                    SmallVector<Channel *> *channels)
      : operation(operation), allocation(allocation), channels(channels) {}

private:
  using BufferT = Allocation::BufferT;
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;
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
        // larger interval has higher priority
        if (intv1.size() > intv2.size())
          return true;
        if (intv1.size() < intv2.size())
          return false;
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
      buffers.emplace_back(allocation.valueBuffer[alloc]);
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

    DenseMap<Operation *, unsigned> allocToOffsets;
    // Start with the first candidate, allocate a space for it
    // Goes through all buffers that interfere with it and allocate
    // a new space or reuse an existing buffer depending on if we
    // have available space.
    bufferId = allocateTMemAllocs(allocs, allocs, allocToIntervals, allocToSize,
                                  allocToChannel, operationId, allocToOffsets,
                                  nullptr, bufferId, true /*firstRun*/);
#if 0
  DenseMap<unsigned, SmallVector<Operation *>> loopDepthMap;
  parentOp->walk([&](Operation *subOp) {
    if (dyn_cast<scf::ForOp>(subOp)) {
      unsigned tDepth = getLoopDepth(subOp);
      loopDepthMap[tDepth].push_back(subOp);
    }
  });
  // Perform allocation for each ForOp.
  llvm::MapVector<Operation *, SmallVector<triton::nvidia_gpu::TMEMAllocOp>>
      forOpToAllocs;
  auto getStartIdForLoop = [&](Operation *op) -> int {
    scf::ForOp forOp = cast<scf::ForOp>(op);
    // get the first instruction, return its operation id.
    for (Operation &op : forOp.getBody()->without_terminator())
      return operationId[&op];
    llvm_unreachable("empty ForOp");
  };
  // HACK for causal, we need to have a general packing algorithm. Even
  // if there is no liverange overlap, we can still use different space.
  for (auto alloc : allocs) {
    // Innermost forOp containing all users of the channel.
    Operation *ctrlOp = getInnermostCtrl(alloc, channels);
    LDBG("add alloc to forOp " << ctrlOp << " " << getLoopDepth(ctrlOp));
    if (getLoopDepth(ctrlOp) == 0) {
      // set ctrlOp to one of the inner ForOps
      if (loopDepthMap[1].size() == 1)
        ctrlOp = loopDepthMap[1][0];
      else {
        auto intv = allocToIntervals[alloc.getOperation()];
        assert(loopDepthMap[1].size() == 2);
        auto *secondLoop = loopDepthMap[1][1];
        if (intv.end() < getStartIdForLoop(secondLoop)) {
          // Before 2nd loop, comparing against first id of the loop.
          ctrlOp = loopDepthMap[1][0];
        } else {
          ctrlOp = loopDepthMap[1][1];
        }
        LDBG("set ctrlOp to " << ctrlOp);
      }
    }
    forOpToAllocs[ctrlOp].push_back(alloc);
  }
  bool firstRun = true;
  DenseMap<Operation *, unsigned> allocToOffsets;
  for (auto &kv : forOpToAllocs) {
    LDBG("run allocation on " << kv.second.size() << " allocs");
    bufferId = allocateTMemAllocs(kv.second, allocs, allocToIntervals,
                                  allocToSize, allocToChannel, operationId,
                                  allocToOffsets, kv.first, bufferId, firstRun);
    for (auto &kv_t : allocToOffsets)
      kv_t.second = 0; // reset offset
    firstRun = false;
  }
#endif
  }

  unsigned allocateTMemAllocs(
      SmallVector<triton::nvidia_gpu::TMEMAllocOp> &allocs,
      SmallVector<triton::nvidia_gpu::TMEMAllocOp> &allAllocs,
      DenseMap<Operation *, Interval<int>> &allocToIntervals,
      DenseMap<Operation *, ttng::TMemAllocation> &allocToSize,
      DenseMap<Operation *, ttng::TmemDataChannelPost *> &allocToChannel,
      DenseMap<Operation *, int> &operationId,
      DenseMap<Operation *, unsigned> &allocToOffsets, Operation *ctrlOp,
      unsigned bufferId, bool firstRun) {
    // Map from owner of a tmem location to the list of allocs that reuse the
    // space.
    DenseMap<Operation *, SmallVector<Operation *>> allocToReuses;
    // Map from an allocation to the owner of a tmem location + the offset.
    DenseMap<Operation *, std::pair<Operation *, int>> allocToAllocation;
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
    // FIXME: need to keep track of the actual allocations:
    // -- alloc: reuse baseAlloc with offset
    // -- allocToOffsets: all baseAllocs
    // For second forOp in causal mode, qk1 should reuse space of qk0, etc.
    auto findReuseChannel = [&](Operation *cand,
                                bool updateOffset) -> Operation * {
      // Go through allocs with buffer.id (i.e allocated), check intervals
      // to find an allocated alloc without overlapping intervals and with
      // enough space.
      // FIXME: try to find a buffer with a dependency chain. For FA, we want
      // p0/alpha0/l_i0/m_i0 to reuse.
      for (auto it = allAllocs.begin(), e = allAllocs.end(); it != e; ++it) {
        Operation *alloc = (*it).getOperation();
        if (allocToOffsets.count(alloc)) {
          auto iter1 = allocToSize.find(cand);
          auto iter2 = allocToSize.find(alloc);
          if (!allocToIntervals[alloc].intersects(allocToIntervals[cand]) &&
              iter2->second.numCols >=
                  iter1->second.numCols + allocToOffsets[alloc] &&
              alongDependencyChain(alloc, cand)) {
            allocToReuses[alloc].push_back(cand);
            allocToAllocation[cand] =
                std::make_pair(alloc, allocToOffsets[alloc]);
            if (updateOffset)
              allocToOffsets[alloc] += iter1->second.numCols;
            LLVM_DEBUG({
              LDBG("move offset to " << allocToOffsets[alloc] << " for:");
              alloc->dump();
            });
            return alloc;
          }
        }
      }
      // Do not enforce dependency
      for (auto it = allAllocs.begin(), e = allAllocs.end(); it != e; ++it) {
        Operation *alloc = (*it).getOperation();
        if (allocToOffsets.count(alloc)) {
          auto iter1 = allocToSize.find(cand);
          auto iter2 = allocToSize.find(alloc);
          if (!allocToIntervals[alloc].intersects(allocToIntervals[cand]) &&
              iter2->second.numCols >=
                  iter1->second.numCols + allocToOffsets[alloc]) {
            allocToAllocation[cand] =
                std::make_pair(alloc, allocToOffsets[alloc]);
            if (updateOffset)
              allocToOffsets[alloc] += iter1->second.numCols;
            allocToReuses[alloc].push_back(cand);
            LLVM_DEBUG({
              LDBG("move offset to " << allocToOffsets[alloc] << " for:");
              alloc->dump();
            });
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
    SmallVector<Operation *> allocOrder;
    for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
      ttng::TMEMAllocOp alloc = *it;
      if (allocToChannel[alloc.getOperation()]->isOperandD ||
          allocToChannel[alloc.getOperation()]->isOperandDNoAcc) {
        if (firstRun) {
          bufferSet[alloc.getOperation()] =
              getIntervalForCtrlOp(ctrlOp, operationId);
          alloc->setAttr(
              "buffer.id",
              IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                               bufferId));
          allocToOffsets[alloc.getOperation()] = 0;
          allocOrder.push_back(alloc.getOperation());
          LLVM_DEBUG({
            LDBG("new buffer.id for channel:");
            alloc.getOperation()->dump();
            auto liveInterval = allocToIntervals[alloc.getOperation()];
            LDBG("new buffer.id liveness: " << liveInterval.start() << " "
                                            << liveInterval.end());
          });
          bufferId++;
        } else {
          // handle 2nd loop.
          auto *reuseAlloc = findReuseChannel(alloc.getOperation(), true);
          if (!reuseAlloc)
            assert(false && "can't find space");
          LLVM_DEBUG({
            LDBG("2nd loop allocate space for channel:");
            alloc.getOperation()->dump();
          });
          bufferSet[alloc.getOperation()] =
              getIntervalForCtrlOp(ctrlOp, operationId);
          alloc->setAttr(
              "buffer.id",
              IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                               getBufferId(reuseAlloc)));
          auto iter2 = allocToSize.find(alloc.getOperation());
          alloc->setAttr(
              "buffer.offset",
              IntegerAttr::get(IntegerType::get(alloc->getContext(), 32),
                               allocToOffsets[reuseAlloc] -
                                   iter2->second.numCols));
          allocOrder.push_back(alloc.getOperation());
        }
        // FIXME: heuristics
        alloc->setAttr(
            "buffer.copy",
            IntegerAttr::get(IntegerType::get(alloc->getContext(), 32), 1));
      }
    }
    if (!firstRun) {
      for (auto &kv_t : allocToOffsets)
        kv_t.second = 0; // reset offset
    }
    for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
      if (bufferSet.count((*it).getOperation()))
        continue;
      if (!candidateAlloc) {
        candidateAlloc = (*it).getOperation();
        break;
      }
    }
    int totalMemorySize = ttng::allocateTMemWithInterval(bufferSet, allocOrder);
    LDBG(bufferSet.size() << " buffers with tmem size: " << totalMemorySize);
    if (totalMemorySize > 512)
      return bufferId;
    while (bufferSet.size() != allocs.size()) {
      // Decide if we need to reuse buffer for candidateAlloc.
      // Choose an interval for candidateAlloc based on the decision.
      LLVM_DEBUG({
        LDBG("try to allocate space for:");
        candidateAlloc->dump();
        auto liveInterval = allocToIntervals[candidateAlloc];
        LDBG("candidate liveness: " << liveInterval.start() << " "
                                    << liveInterval.end());
      });
      bufferSet[candidateAlloc] = getIntervalForCtrlOp(ctrlOp, operationId);
      allocOrder.push_back(candidateAlloc);
      totalMemorySize = ttng::allocateTMemWithInterval(bufferSet, allocOrder);
      LDBG(bufferSet.size() << " buffers with tmem size: " << totalMemorySize);
      for (auto *op_t : allocOrder) {
        LLVM_DEBUG(op_t->dump());
        LLVM_DEBUG(llvm::dbgs() << op_t << " " << bufferSet[op_t].start() << " "
                                << bufferSet[op_t].end() << "\n");
      }
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
        auto *reuseAlloc = findReuseChannel(candidateAlloc, true);
        if (!reuseAlloc)
          assert(false && "can't find space");
        LLVM_DEBUG({
          LDBG("try to reuse space with:");
          reuseAlloc->dump();
        });
        // update intervals for representative channel and this channel so they
        // will not overlap
        if (bufferSet.count(reuseAlloc)) {
          auto origIntv = bufferSet[reuseAlloc];
          bufferSet[reuseAlloc] =
              Interval(origIntv.start(), origIntv.end() - 2);
          bufferSet[candidateAlloc] =
              Interval(origIntv.end() - 2, origIntv.end());
        }
        totalMemorySize = ttng::allocateTMemWithInterval(bufferSet, allocOrder);
        LDBG(bufferSet.size()
             << " buffers with tmem size: " << totalMemorySize);

        candidateAlloc->setAttr(
            "buffer.id",
            IntegerAttr::get(IntegerType::get(candidateAlloc->getContext(), 32),
                             getBufferId(reuseAlloc)));
        auto iter2 = allocToSize.find(candidateAlloc);
        candidateAlloc->setAttr(
            "buffer.offset",
            IntegerAttr::get(IntegerType::get(candidateAlloc->getContext(), 32),
                             allocToOffsets[reuseAlloc] -
                                 iter2->second.numCols));
        LLVM_DEBUG(candidateAlloc->dump());
      }
      for (auto *op_t : allocOrder) {
        LLVM_DEBUG(op_t->dump());
        LLVM_DEBUG(llvm::dbgs() << op_t << " " << bufferSet[op_t].start() << " "
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
