#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_NVWSPACKEPILOGUESLICES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
} // namespace mlir::triton

namespace {

static Value getTMAStoreBuffer(Operation *op) {
  if (auto copy = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return copy.getSrc();
  return {};
}

static bool isSchedulingBoundary(Operation *op) {
  return isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAReduceOp,
             ttng::TMAStoreTokenWaitOp, ttng::WaitBarrierOp,
             ttng::ArriveBarrierOp, ttng::NamedBarrierArriveOp,
             ttng::NamedBarrierWaitOp, ttng::AsyncCopyMbarrierArriveOp>(op) ||
         op->getName().getStringRef() == "gpu.barrier";
}

static bool hasOnlyReadEffects(Operation *op) {
  if (isMemoryEffectFree(op))
    return true;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::all_of(instances, [](MemoryEffects::EffectInstance effect) {
    return isa<MemoryEffects::Read>(effect.getEffect());
  });
}

static bool isMovableProducer(Operation *op) {
  return op->getNumRegions() == 0 && !op->hasTrait<OpTrait::IsTerminator>() &&
         hasOnlyReadEffects(op);
}

static bool canCrossWhilePacking(Operation *op,
                                 const DenseSet<Operation *> &slice) {
  if (slice.contains(op) || isMemoryEffectFree(op))
    return true;
  if (isSchedulingBoundary(op) || op->getNumRegions() != 0)
    return false;
  // The moved slice reads TensorMemory and writes SharedMemory. A Triton
  // pointer store writes GlobalMemory, so preserving all global-store order
  // while moving the independent shared-memory store across it is safe.
  if (isa<triton::StoreOp>(op))
    return true;
  return hasOnlyReadEffects(op);
}

static LogicalResult packLoopEpilogue(scf::ForOp loop) {
  if (!loop->hasAttr(kWarpSpecializeAttrName) ||
      (!loop->hasAttr("tt.merge_epilogue") &&
       !loop->hasAttr("tt.merge_epilogue_to_computation")))
    return success();

  auto factor = loop->getAttrOfType<IntegerAttr>("tt.data_partition_factor");
  if (!factor || factor.getInt() != 2)
    return success();

  Block *body = loop.getBody();
  Operation *lastInnerLoop = nullptr;
  for (Operation &op : body->without_terminator())
    if (isa<scf::ForOp>(op))
      lastInnerLoop = &op;
  if (!lastInnerLoop)
    return success();

  SmallVector<Operation *> segment;
  for (Operation *op = lastInnerLoop->getNextNode();
       op && op != body->getTerminator(); op = op->getNextNode())
    segment.push_back(op);
  DenseSet<Operation *> segmentSet(segment.begin(), segment.end());

  SmallVector<ttng::AsyncTMACopyLocalToGlobalOp> tmaStores;
  for (Operation *op : segment)
    if (auto tma = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op))
      tmaStores.push_back(tma);
  if (tmaStores.size() != 2)
    return success();

  // Pair each TMA store with the unique preceding local_store to the exact
  // source-free staging allocation. TMA order is externally observable and
  // must agree with producer-store order.
  SmallVector<LocalStoreOp> roots;
  for (ttng::AsyncTMACopyLocalToGlobalOp tma : tmaStores) {
    Value buffer = getTMAStoreBuffer(tma);
    auto alloc = buffer.getDefiningOp<LocalAllocOp>();
    if (!alloc || alloc.getSrc() || !alloc.isSharedMemoryAlloc())
      return success();

    SmallVector<LocalStoreOp> matches;
    for (Operation *candidate : segment) {
      auto store = dyn_cast<LocalStoreOp>(candidate);
      if (store && store.getDst() == buffer)
        matches.push_back(store);
    }
    if (matches.size() != 1 ||
        !matches.front()->isBeforeInBlock(tma.getOperation()))
      return success();

    if (!tma.getToken().hasOneUse())
      return success();
    auto wait = dyn_cast<ttng::TMAStoreTokenWaitOp>(*tma.getToken().user_begin());
    if (!wait || wait->getBlock() != body ||
        !tma->isBeforeInBlock(wait.getOperation()) ||
        !wait.getBarriers().empty() || !wait.getBarrierPreds().empty() ||
        !wait.getNvwsTokens().empty() || !wait.getNvwsTokenIndices().empty())
      return success();
    roots.push_back(matches.front());
  }
  if (roots.size() != 2 ||
      !roots.front()->isBeforeInBlock(roots.back().getOperation()))
    return success();

  // Do not move a producer across synchronization or an already-issued TMA
  // operation. Such a boundary means this is not the straight-line
  // post-data-partition epilogue handled by this optimization.
  // Match the proven pre-TMA normalization exactly: only the first slice is
  // packed. Once it ends at its local_store, the second slice is the natural
  // tail and no longer overlaps another large output value.
  LocalStoreOp root = roots.front();
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  options.omitUsesFromAbove = true;
  options.filter = [&](Operation *op) { return segmentSet.contains(op); };
  SetVector<Operation *> backwardSlice;
  if (failed(getBackwardSlice(root.getOperation(), &backwardSlice, options)))
    return success();
  backwardSlice.insert(root.getOperation());

  SmallVector<Operation *> ordered;
  for (Operation *candidate : segment) {
    if (!backwardSlice.contains(candidate))
      continue;
    if (candidate != root.getOperation() && !isMovableProducer(candidate))
      return success();
    ordered.push_back(candidate);
  }
  if (ordered.empty() || ordered.back() != root.getOperation())
    return success();

  // Keep the earliest producer anchored. In real Meta-planned epilogues,
  // source-free TMEM loads in the slice are preceded by TMEM stores that
  // materialize loop results. Those stores are memory dependencies but are
  // not represented by an SSA edge to the tokenless loads, so moving the
  // complete slice to the beginning of the epilogue would be incorrect.
  // Packing only the remaining operations after the anchor preserves those
  // dependencies while still ending the first slice at its local_store before
  // the second large accumulator slice becomes live.
  Operation *anchor = ordered.front();
  DenseSet<Operation *> sliceSet(backwardSlice.begin(), backwardSlice.end());
  for (Operation *op = anchor->getNextNode(); op && op != root;
       op = op->getNextNode()) {
    if (!canCrossWhilePacking(op, sliceSet))
      return success();
  }

  Operation *insertAfter = anchor;
  for (Operation *op : ArrayRef<Operation *>(ordered).drop_front()) {
    op->moveAfter(insertAfter);
    insertAfter = op;
  }
  return success();
}

class NVWSPackEpilogueSlices
    : public mlir::triton::impl::NVWSPackEpilogueSlicesBase<
          NVWSPackEpilogueSlices> {
public:
  using mlir::triton::impl::NVWSPackEpilogueSlicesBase<
      NVWSPackEpilogueSlices>::NVWSPackEpilogueSlicesBase;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](scf::ForOp loop) {
      if (failed(packLoopEpilogue(loop))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    (void)result;
  }
};

} // namespace
