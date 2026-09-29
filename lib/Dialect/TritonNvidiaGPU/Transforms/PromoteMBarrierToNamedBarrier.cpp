#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/NamedBarrier.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPROMOTEMBARRIERTONAMEDBARRIERPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

struct BarrierCandidate {
  explicit BarrierCandidate(ttg::LocalAllocOp alloc) : alloc(alloc) {}

  ttg::LocalAllocOp alloc;
  SmallVector<ttng::InitBarrierOp> inits;
  SmallVector<ttng::InvalBarrierOp> invalidations;
  SmallVector<ttng::ArriveBarrierOp> arrives;
  SmallVector<ttng::WaitBarrierOp> waits;
  SmallVector<ttg::LocalDeallocOp> deallocations;
  SmallVector<Operation *> views;
  SmallVector<std::pair<ttg::WarpSpecializePartitionsOp, Value>> captures;
  llvm::SmallDenseSet<Value> visitedValues;
  bool hasUnknownUse = false;
};

bool isSingleBarrierAlloc(ttg::LocalAllocOp alloc) {
  ttg::MemDescType type = alloc.getType();
  return !alloc.getSrc() && type.getElementType().isInteger(64) &&
         !type.getShape().empty() && type.getShape().front() == 1;
}

Region *getWarpSpecializePartition(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    // A top-level region (ModuleOp's) has no parent op. Stop here rather than
    // dereferencing null in the isa<> and the getParentRegion() below, which
    // is reached by any barrier not nested in a warp-specialize partition.
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
  if (!lhsDef || !rhsDef || lhsDef->getNumRegions() ||
      rhsDef->getNumRegions() || !isMemoryEffectFree(lhsDef) ||
      !isMemoryEffectFree(rhsDef) ||
      cast<OpResult>(lhs).getResultNumber() !=
          cast<OpResult>(rhs).getResultNumber())
    return false;
  return OperationEquivalence::isEquivalentTo(
      lhsDef, rhsDef,
      [&](Value lhsOperand, Value rhsOperand) {
        return success(
            sameLoopBound(lhsOperand, rhsOperand, inductionVars, depth + 1));
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

void traceBarrierUses(Value value, BarrierCandidate &candidate) {
  if (!candidate.visitedValues.insert(value).second)
    return;

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      candidate.views.push_back(user);
      traceBarrierUses(user->getResult(0), candidate);
      continue;
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(user)) {
      unsigned operandIdx = use.getOperandNumber();
      candidate.captures.push_back({partitions, value});
      for (Region &region : partitions.getPartitionRegions())
        traceBarrierUses(region.getArgument(operandIdx), candidate);
      continue;
    }
    if (auto init = dyn_cast<ttng::InitBarrierOp>(user)) {
      candidate.inits.push_back(init);
      continue;
    }
    if (auto inval = dyn_cast<ttng::InvalBarrierOp>(user)) {
      candidate.invalidations.push_back(inval);
      continue;
    }
    if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(user)) {
      candidate.arrives.push_back(arrive);
      continue;
    }
    if (auto wait = dyn_cast<ttng::WaitBarrierOp>(user)) {
      candidate.waits.push_back(wait);
      continue;
    }
    if (auto dealloc = dyn_cast<ttg::LocalDeallocOp>(user)) {
      candidate.deallocations.push_back(dealloc);
      continue;
    }
    candidate.hasUnknownUse = true;
  }
}

std::optional<unsigned> getParticipantCount(BarrierCandidate &candidate) {
  if (candidate.inits.size() != 1 || candidate.arrives.size() != 1 ||
      candidate.waits.size() != 1 || candidate.hasUnknownUse)
    return std::nullopt;

  ttng::ArriveBarrierOp arrive = candidate.arrives.front();
  ttng::WaitBarrierOp wait = candidate.waits.front();
  if (arrive.getPerThread() || arrive.isMulticast() || arrive.getPred() ||
      wait.getPred() || !wait.getDeps().empty())
    return std::nullopt;
  if (candidate.inits.front().getCount() != arrive.getCount())
    return std::nullopt;
  if (!isWarpUniform(arrive) || !isWarpUniform(wait))
    return std::nullopt;
  if (!haveMatchingLoopNests(arrive, wait))
    return std::nullopt;

  auto isCTALocal = [](auto op) {
    ttg::MemDescType type = op.getAlloc().getType();
    // Ordinary shared memory can hold a barrier broadcast across CTAs.
    return !isa<ttng::SharedClusterMemorySpaceAttr>(type.getMemorySpace()) &&
           type.getShape().size() == 1 &&
           type.getShape().front() == ttg::lookupNumCTAs(op);
  };
  if (!isCTALocal(candidate.inits.front()) || !isCTALocal(arrive) ||
      !isCTALocal(wait))
    return std::nullopt;

  unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(
      candidate.alloc->getParentOfType<ModuleOp>());
  unsigned numThreads =
      (ttg::lookupNumWarps(arrive) + ttg::lookupNumWarps(wait)) *
      threadsPerWarp;
  if (numThreads == 0)
    return std::nullopt;
  return numThreads;
}

void eraseBarrierStorage(BarrierCandidate &candidate) {
  for (ttng::InitBarrierOp op : candidate.inits)
    op.erase();
  for (ttng::InvalBarrierOp op : candidate.invalidations)
    op.erase();
  for (ttg::LocalDeallocOp op : candidate.deallocations)
    op.erase();

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
      bool unused =
          llvm::all_of(partitions.getPartitionRegions(), [idx](Region &region) {
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
    if (view->getResult(0).use_empty())
      view->erase();
  }
  if (candidate.alloc.use_empty())
    candidate.alloc.erase();
}

void promoteBarrier(BarrierCandidate &candidate, int32_t id,
                    unsigned numThreads) {
  for (ttng::ArriveBarrierOp arrive : candidate.arrives) {
    OpBuilder builder(arrive);
    Value namedId = createCompilerNamedBarrierId(builder, arrive.getLoc(), id);
    Value count =
        arith::ConstantIntOp::create(builder, arrive.getLoc(), numThreads, 32);
    ttng::NamedBarrierArriveOp::create(builder, arrive.getLoc(), namedId,
                                       count);
    arrive.erase();
  }
  for (ttng::WaitBarrierOp wait : candidate.waits) {
    OpBuilder builder(wait);
    Value namedId = createCompilerNamedBarrierId(builder, wait.getLoc(), id);
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
      if (isSingleBarrierAlloc(alloc))
        allocs.push_back(alloc);
    });

    for (ttg::LocalAllocOp alloc : allocs) {
      BarrierCandidate candidate(alloc);
      traceBarrierUses(alloc.getResult(), candidate);
      std::optional<unsigned> numThreads = getParticipantCount(candidate);
      if (!numThreads)
        continue;
      std::optional<SmallVector<int32_t>> ids = allocator.allocate(1);
      if (!ids)
        continue;
      promoteBarrier(candidate, ids->front(), *numThreads);
    }
  }
};

} // namespace mlir::triton::nvidia_gpu
