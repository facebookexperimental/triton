#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/NamedBarrier.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallSet.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPROMOTEMBARRIERTONAMEDBARRIERPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

template <typename OpTy> struct IndexedBarrierUse {
  OpTy op;
  Value index;
};

struct BarrierCandidate {
  BarrierCandidate(ttg::LocalAllocOp alloc, unsigned numBarriers)
      : alloc(alloc), numBarriers(numBarriers) {}

  ttg::LocalAllocOp alloc;
  unsigned numBarriers;
  SmallVector<IndexedBarrierUse<ttng::InitBarrierOp>> inits;
  SmallVector<IndexedBarrierUse<ttng::InvalBarrierOp>> invalidations;
  SmallVector<IndexedBarrierUse<ttng::ArriveBarrierOp>> arrives;
  SmallVector<IndexedBarrierUse<ttng::WaitBarrierOp>> waits;
  SmallVector<ttg::LocalDeallocOp> deallocations;
  SmallVector<Operation *> views;
  SmallVector<std::pair<ttg::WarpSpecializePartitionsOp, Value>> captures;
  llvm::SmallDenseSet<Value> visitedValues;
  bool hasUnknownUse = false;
};

std::optional<unsigned> getBarrierCount(ttg::LocalAllocOp alloc) {
  ttg::MemDescType type = alloc.getType();
  if (alloc.getSrc() || !type.getElementType().isInteger(64))
    return std::nullopt;
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() == 1)
    return shape.front() == 1 ? std::optional<unsigned>(1) : std::nullopt;
  if (shape.size() != 2 || shape.back() != 1 || shape.front() <= 0)
    return std::nullopt;
  return static_cast<unsigned>(shape.front());
}

Region *getWarpSpecializePartition(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    // A top-level region (ModuleOp's) has no parent op. Any barrier not
    // nested in a warp-specialize partition reaches this null rather than a
    // partitions op, so return null here instead of dereferencing it in the
    // isa<> and the getParentRegion() below.
    if (!parent)
      return nullptr;
    if (isa<ttg::WarpSpecializePartitionsOp>(parent))
      return region;
    region = parent->getParentRegion();
  }
  return nullptr;
}

bool isWarpUniformValue(Value value, llvm::SmallDenseSet<Value> &visiting) {
  if (!visiting.insert(value).second)
    return true;
  if (Operation *def = value.getDefiningOp()) {
    if (def->hasTrait<OpTrait::ConstantLike>())
      return true;
    if (def->getName().getDialectNamespace() == "arith")
      return llvm::all_of(def->getOperands(), [&](Value operand) {
        return isWarpUniformValue(operand, visiting);
      });
    return isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(def);
  }

  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  Operation *parent = arg.getOwner()->getParentOp();
  if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent))
    return isWarpUniformValue(
        partitions.getExplicitCaptures()[arg.getArgNumber()], visiting);
  if (isa<triton::FuncOp>(parent))
    return true;
  if (auto loop = dyn_cast<scf::ForOp>(parent)) {
    if (arg != loop.getInductionVar())
      return false;
    return isWarpUniformValue(loop.getLowerBound(), visiting) &&
           isWarpUniformValue(loop.getUpperBound(), visiting) &&
           isWarpUniformValue(loop.getStep(), visiting);
  }
  return false;
}

bool isWarpUniform(Operation *op) {
  Region *partition = getWarpSpecializePartition(op);
  if (!partition)
    return false;
  for (Operation *parent = op->getParentOp();
       parent && parent != partition->getParentOp();
       parent = parent->getParentOp()) {
    // Only a loop is entered by every warp in the partition. Anything else --
    // scf.if, scf.while, scf.index_switch, an unstructured cf.cond_br region --
    // may run for a subset, and a promoted named barrier would then wait on
    // warps that never arrive. Allow-list rather than deny-list: a deny-list
    // silently over-promotes every time a new region op appears.
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      return false;
    // The trip count must also be the same for every warp, or the arrivals
    // will not pair up across iterations.
    llvm::SmallDenseSet<Value> visiting;
    if (!isWarpUniformValue(loop.getLowerBound(), visiting) ||
        !isWarpUniformValue(loop.getUpperBound(), visiting) ||
        !isWarpUniformValue(loop.getStep(), visiting))
      return false;
  }
  return true;
}

bool sameLoopBound(Value lhs, Value rhs,
                   const DenseMap<Value, Value> &inductionVars,
                   unsigned depth = 0) {
  lhs = ttg::resolveWarpSpecializeCapture(lhs);
  rhs = ttg::resolveWarpSpecializeCapture(rhs);
  if (lhs == rhs)
    return true;
  if (auto it = inductionVars.find(lhs); it != inductionVars.end())
    return it->second == rhs;
  if (lhs.getType() != rhs.getType() || depth > 8)
    return false;

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef || lhsDef->getNumRegions() || rhsDef->getNumRegions() ||
      !isMemoryEffectFree(lhsDef) || !isMemoryEffectFree(rhsDef) ||
      cast<OpResult>(lhs).getResultNumber() !=
          cast<OpResult>(rhs).getResultNumber())
    return false;
  return OperationEquivalence::isEquivalentTo(
      lhsDef, rhsDef,
      [&](Value lhsOperand, Value rhsOperand) {
        return success(sameLoopBound(lhsOperand, rhsOperand, inductionVars,
                                     depth + 1));
      },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

bool haveMatchingLoopNests(Operation *arrive, Operation *wait) {
  Region *arrivePartition = getWarpSpecializePartition(arrive);
  Region *waitPartition = getWarpSpecializePartition(wait);
  if (!arrivePartition || !waitPartition || arrivePartition == waitPartition ||
      arrivePartition->getParentOp() != waitPartition->getParentOp())
    return false;

  auto getLoops = [](Operation *op, Region *partition) {
    SmallVector<scf::ForOp> loops;
    for (Operation *parent = op->getParentOp();
         parent != partition->getParentOp(); parent = parent->getParentOp())
      loops.push_back(cast<scf::ForOp>(parent));
    return loops;
  };
  auto arriveLoops = getLoops(arrive, arrivePartition);
  auto waitLoops = getLoops(wait, waitPartition);
  if (arriveLoops.size() != waitLoops.size())
    return false;

  // Repeated waits may observe one completed mbarrier phase. A named wait
  // consumes a new arrival, so uniform execution alone does not establish
  // matching numbers of arrivals and waits.
  DenseMap<Value, Value> inductionVars;
  for (auto [arriveLoop, waitLoop] :
       llvm::zip(llvm::reverse(arriveLoops), llvm::reverse(waitLoops))) {
    if (!sameLoopBound(arriveLoop.getLowerBound(), waitLoop.getLowerBound(),
                       inductionVars) ||
        !sameLoopBound(arriveLoop.getUpperBound(), waitLoop.getUpperBound(),
                       inductionVars) ||
        !sameLoopBound(arriveLoop.getStep(), waitLoop.getStep(), inductionVars))
      return false;
    inductionVars[arriveLoop.getInductionVar()] = waitLoop.getInductionVar();
  }
  return true;
}

void traceBarrierUses(Value value, Value index, BarrierCandidate &candidate) {
  if (!candidate.visitedValues.insert(value).second)
    return;

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(user)) {
      if (index) {
        candidate.hasUnknownUse = true;
        continue;
      }
      candidate.views.push_back(user);
      traceBarrierUses(indexOp.getResult(), indexOp.getIndex(), candidate);
      continue;
    }
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      candidate.views.push_back(user);
      traceBarrierUses(user->getResult(0), index, candidate);
      continue;
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(user)) {
      unsigned operandIdx = use.getOperandNumber();
      candidate.captures.push_back({partitions, value});
      for (Region &region : partitions.getPartitionRegions())
        traceBarrierUses(region.getArgument(operandIdx), index, candidate);
      continue;
    }
    if (auto init = dyn_cast<ttng::InitBarrierOp>(user)) {
      candidate.inits.push_back({init, index});
      continue;
    }
    if (auto inval = dyn_cast<ttng::InvalBarrierOp>(user)) {
      candidate.invalidations.push_back({inval, index});
      continue;
    }
    if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(user)) {
      candidate.arrives.push_back({arrive, index});
      continue;
    }
    if (auto wait = dyn_cast<ttng::WaitBarrierOp>(user)) {
      candidate.waits.push_back({wait, index});
      continue;
    }
    if (auto dealloc = dyn_cast<ttg::LocalDeallocOp>(user)) {
      candidate.deallocations.push_back(dealloc);
      continue;
    }
    candidate.hasUnknownUse = true;
  }
}

std::optional<unsigned> getStaticSlot(Value index, unsigned numBarriers) {
  if (!index)
    return numBarriers == 1 ? std::optional<unsigned>(0) : std::nullopt;
  APInt value;
  if (!matchPattern(index, m_ConstantInt(&value)))
    return std::nullopt;
  int64_t slot = value.getSExtValue();
  if (slot < 0 || slot >= static_cast<int64_t>(numBarriers))
    return std::nullopt;
  return static_cast<unsigned>(slot);
}

template <typename RangeT>
bool coversEverySlot(const RangeT &uses, unsigned numBarriers) {
  llvm::SmallDenseSet<unsigned> slots;
  for (const auto &use : uses) {
    std::optional<unsigned> slot = getStaticSlot(use.index, numBarriers);
    if (!slot || !slots.insert(*slot).second)
      return false;
  }
  return slots.size() == numBarriers;
}

// Accepts either one constant-indexed use per slot or a single
// dynamically-indexed use. One dynamic use suffices because promotion rewrites
// it to a runtime select over the constant named-barrier handles, covering
// every slot as long as the index stays in range -- which holds since it
// already indexes the same N-slot mbarrier allocation.
template <typename RangeT>
bool hasValidSlotSelection(const RangeT &uses, unsigned numBarriers) {
  if (uses.size() == 1 && uses.front().index) {
    APInt value;
    if (!matchPattern(uses.front().index, m_ConstantInt(&value)))
      return true;
  }
  return coversEverySlot(uses, numBarriers);
}

std::optional<unsigned> getParticipantCount(BarrierCandidate &candidate) {
  if (candidate.hasUnknownUse ||
      !coversEverySlot(candidate.inits, candidate.numBarriers) ||
      (!candidate.invalidations.empty() &&
       !coversEverySlot(candidate.invalidations, candidate.numBarriers)) ||
      !hasValidSlotSelection(candidate.arrives, candidate.numBarriers) ||
      !hasValidSlotSelection(candidate.waits, candidate.numBarriers))
    return std::nullopt;

  for (const auto &use : candidate.arrives) {
    ttng::ArriveBarrierOp arrive = use.op;
    if (arrive.getPerThread() || arrive.isMulticast() || arrive.getPred() ||
        !isWarpUniform(arrive))
      return std::nullopt;
  }
  for (const auto &use : candidate.waits) {
    ttng::WaitBarrierOp wait = use.op;
    if (wait.getPred() || !wait.getDeps().empty() || !isWarpUniform(wait))
      return std::nullopt;
  }

  Region *arrivePartition = getWarpSpecializePartition(candidate.arrives[0].op);
  Region *waitPartition = getWarpSpecializePartition(candidate.waits[0].op);
  // Arrives and waits must live in different warp-specialize partitions. This
  // also rejects the both-nullptr case: a barrier used entirely outside any
  // partition has no warp-group structure to size the named barrier from.
  if (arrivePartition == waitPartition)
    return std::nullopt;
  if (llvm::any_of(candidate.arrives, [arrivePartition](const auto &use) {
        return getWarpSpecializePartition(use.op) != arrivePartition;
      }) ||
      llvm::any_of(candidate.waits, [waitPartition](const auto &use) {
        return getWarpSpecializePartition(use.op) != waitPartition;
      }))
    return std::nullopt;

  if (llvm::any_of(candidate.arrives, [&](const auto &use) {
        return !haveMatchingLoopNests(use.op, candidate.waits.front().op);
      }) ||
      llvm::any_of(candidate.waits, [&](const auto &use) {
        return !haveMatchingLoopNests(candidate.arrives.front().op, use.op);
      }))
    return std::nullopt;

  SmallVector<uint32_t> initCounts(candidate.numBarriers);
  for (const auto &use : candidate.inits) {
    auto init = use.op;
    initCounts[*getStaticSlot(use.index, candidate.numBarriers)] =
        init.getCount();
  }
  for (const auto &use : candidate.arrives) {
    auto arrive = use.op;
    if (std::optional<unsigned> slot =
            getStaticSlot(use.index, candidate.numBarriers)) {
      if (arrive.getCount() != initCounts[*slot])
        return std::nullopt;
      continue;
    }
    uint32_t arriveCount = arrive.getCount();
    if (llvm::any_of(initCounts, [&](uint32_t count) {
          return arriveCount != count;
        }))
      return std::nullopt;
  }

  auto isCTALocal = [](auto op) {
    ttg::MemDescType type = op.getAlloc().getType();
    // Ordinary shared memory can hold a barrier broadcast across CTAs.
    return !isa<ttng::SharedClusterMemorySpaceAttr>(type.getMemorySpace()) &&
           type.getShape().size() == 1 &&
           type.getShape().front() == ttg::lookupNumCTAs(op);
  };
  auto areCTALocal = [&](const auto &uses) {
    return llvm::all_of(uses, [&](const auto &use) { return isCTALocal(use.op); });
  };
  if (!areCTALocal(candidate.inits) || !areCTALocal(candidate.arrives) ||
      !areCTALocal(candidate.waits))
    return std::nullopt;

  unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(
      candidate.alloc->getParentOfType<ModuleOp>());
  unsigned numThreads =
      (ttg::lookupNumWarps(candidate.arrives[0].op) +
       ttg::lookupNumWarps(candidate.waits[0].op)) * threadsPerWarp;
  if (numThreads == 0)
    return std::nullopt;
  return numThreads;
}

void eraseBarrierStorage(BarrierCandidate &candidate) {
  for (auto use : candidate.inits)
    use.op.erase();
  for (auto use : candidate.invalidations)
    use.op.erase();
  for (ttg::LocalDeallocOp op : candidate.deallocations)
    op.erase();

  // Two passes: this one drops views the promotion just made dead, the one
  // below drops views that only stayed live as capture operands. `views` is
  // walked twice, so null each entry as it goes -- the second pass would
  // otherwise dereference an operation this pass already erased.
  for (Operation *&view : llvm::reverse(candidate.views)) {
    if (view->getResult(0).use_empty()) {
      view->erase();
      view = nullptr;
    }
  }

  // Erase every dead capture of a partitions op in one pass, scanning operand
  // positions rather than searching for each captured value. Two things make
  // the position the thing to key on: erasing an operand shifts the higher
  // ones down, so a stored index goes stale; and a value can be captured into
  // more than one operand, where searching by value keeps finding the first of
  // them and a later dead slot is never reached.
  llvm::MapVector<ttg::WarpSpecializePartitionsOp, llvm::SmallDenseSet<Value>>
      capturedByPartitions;

  for (auto [partitions, captured] : candidate.captures)
    capturedByPartitions[partitions].insert(captured);

  for (auto &[partitions, capturedValues] : capturedByPartitions) {
    llvm::BitVector toRemove(partitions.getNumOperands());
    for (unsigned idx = 0; idx < partitions->getNumOperands(); ++idx) {
      if (!capturedValues.contains(partitions->getOperand(idx)))
        continue;
      bool unused = llvm::all_of(partitions.getPartitionRegions(),
                                 [idx](Region &region) {
                                   return region.getArgument(idx).use_empty();
                                 });
      if (unused)
        toRemove.set(idx);
    }
    if (toRemove.none())
      continue;
    for (Region &region : partitions.getPartitionRegions())
      region.front().eraseArguments(toRemove);
    partitions->eraseOperands(toRemove);
  }

  for (Operation *view : llvm::reverse(candidate.views)) {
    if (view && view->getResult(0).use_empty())
      view->erase();
  }

  if (candidate.alloc.use_empty())
    candidate.alloc.erase();
}

Value createSelectedNamedBarrierId(OpBuilder &builder, Location loc,
                                   ArrayRef<int32_t> ids, Value index) {
  // A single handle needs no selection: the loop below would start at index
  // -1, emit nothing, and return it, but say so directly.
  if (ids.size() == 1)
    return createCompilerNamedBarrierId(builder, loc, ids.front());
  if (std::optional<unsigned> slot = getStaticSlot(index, ids.size()))
    return createCompilerNamedBarrierId(builder, loc, ids[*slot]);

  SmallVector<Value> handles;
  for (int32_t id : ids)
    handles.push_back(createCompilerNamedBarrierId(builder, loc, id));
  Value selected = handles.back();
  for (int32_t slot = static_cast<int32_t>(ids.size()) - 2; slot >= 0;
       --slot) {
    Value slotValue = arith::ConstantIntOp::create(builder, loc,
                                                   index.getType(), slot);
    Value isSlot = arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::eq, index,
                                         slotValue);
    selected = arith::SelectOp::create(builder, loc, isSlot, handles[slot],
                                       selected);
  }
  return selected;
}

void promoteBarrier(BarrierCandidate &candidate, ArrayRef<int32_t> ids,
                    unsigned numThreads) {
  for (const auto &use : candidate.arrives) {
    ttng::ArriveBarrierOp arrive = use.op;
    OpBuilder builder(arrive);
    Value namedId = createSelectedNamedBarrierId(builder, arrive.getLoc(), ids,
                                                  use.index);
    Value count =
        arith::ConstantIntOp::create(builder, arrive.getLoc(), numThreads, 32);
    ttng::NamedBarrierArriveOp::create(builder, arrive.getLoc(), namedId,
                                       count);
    arrive.erase();
  }
  for (const auto &use : candidate.waits) {
    ttng::WaitBarrierOp wait = use.op;
    OpBuilder builder(wait);
    Value namedId = createSelectedNamedBarrierId(builder, wait.getLoc(), ids,
                                                  use.index);
    Value count =
        arith::ConstantIntOp::create(builder, wait.getLoc(), numThreads, 32);
    ttng::NamedBarrierWaitOp::create(builder, wait.getLoc(), namedId, count);
    wait.erase();
  }
  eraseBarrierStorage(candidate);
}

} // namespace

class TritonNvidiaGPUPromoteMBarrierToNamedBarrierPass
    : public impl::TritonNvidiaGPUPromoteMBarrierToNamedBarrierPassBase<
          TritonNvidiaGPUPromoteMBarrierToNamedBarrierPass> {
public:
  using TritonNvidiaGPUPromoteMBarrierToNamedBarrierPassBase::
      TritonNvidiaGPUPromoteMBarrierToNamedBarrierPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    NamedBarrierIdAllocator allocator(module);
    if (failed(tryEnsureWarpSpecializeBarrierIds(module, allocator)))
      return;

    SmallVector<ttg::LocalAllocOp> allocs;
    module.walk([&](ttg::LocalAllocOp alloc) {
      if (getBarrierCount(alloc))
        allocs.push_back(alloc);
    });

    for (ttg::LocalAllocOp alloc : allocs) {
      unsigned numBarriers = *getBarrierCount(alloc);
      BarrierCandidate candidate(alloc, numBarriers);
      traceBarrierUses(alloc.getResult(), Value(), candidate);
      std::optional<unsigned> numThreads = getParticipantCount(candidate);
      if (!numThreads)
        continue;
      std::optional<SmallVector<int32_t>> ids =
          allocator.allocate(numBarriers);
      if (!ids)
        continue;
      promoteBarrier(candidate, *ids, *numThreads);
    }
  }
};

} // namespace mlir::triton::nvidia_gpu
