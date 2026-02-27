#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMASTOREBUFFERREUSEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;

struct CandidateInfo {
  ttg::LocalAllocOp alloc;
  Operation *tmaCopyOp;
  Operation *donePoint;
  unsigned blockPosition;
};

static bool isTMAStoreUser(Operation *op) {
  return isa<AsyncTMACopyLocalToGlobalOp, AsyncTMAScatterOp, AsyncTMAReduceOp>(
      op);
}

// A LocalAllocOp is a candidate for buffer reuse if:
// - It has a src operand (initialized alloc from TMA lowering)
// - Its result memdesc is in shared memory
// - It has exactly one user, which is a TMA store op
static bool isCandidate(ttg::LocalAllocOp alloc) {
  if (!alloc.getSrc())
    return false;

  auto memDescTy = cast<ttg::MemDescType>(alloc.getType());
  if (!isa<ttg::SharedMemorySpaceAttr>(memDescTy.getMemorySpace()))
    return false;

  if (!alloc->hasOneUse())
    return false;

  return isTMAStoreUser(*alloc->getUsers().begin());
}

// Walk forward from the TMA copy op to find a TMAStoreWaitOp with pendings=0
// in the same block.
static Operation *findDonePoint(Operation *tmaCopyOp) {
  for (Operation *op = tmaCopyOp->getNextNode(); op; op = op->getNextNode()) {
    if (auto waitOp = dyn_cast<TMAStoreWaitOp>(op)) {
      if (waitOp.getPendings() == 0)
        return op;
    }
  }
  return nullptr;
}

static ttg::MemDescType getMutableType(ttg::MemDescType ty) {
  return ttg::MemDescType::get(ty.getShape(), ty.getElementType(),
                               ty.getEncoding(), ty.getMemorySpace(),
                               /*mutableMemory=*/true);
}

static void processBlock(Block &block) {
  // Build position map for ordering checks.
  DenseMap<Operation *, unsigned> opPosition;
  unsigned pos = 0;
  for (Operation &op : block)
    opPosition[&op] = pos++;

  // Collect candidates in block order.
  SmallVector<CandidateInfo> candidates;
  for (Operation &op : block) {
    auto alloc = dyn_cast<ttg::LocalAllocOp>(&op);
    if (!alloc || !isCandidate(alloc))
      continue;

    Operation *tmaOp = *alloc->getUsers().begin();
    Operation *donePoint = findDonePoint(tmaOp);
    if (!donePoint)
      continue;

    candidates.push_back({alloc, tmaOp, donePoint, opPosition[alloc]});
  }

  if (candidates.size() < 2)
    return;

  // Group candidates by compatible mutable memdesc type.
  // MLIR types are uniqued, so pointer equality works for DenseMap keys.
  DenseMap<Type, SmallVector<unsigned>> groups;
  for (unsigned i = 0; i < candidates.size(); ++i) {
    auto mutableTy =
        getMutableType(cast<ttg::MemDescType>(candidates[i].alloc.getType()));
    groups[mutableTy].push_back(i);
  }

  for (auto &[ty, indices] : groups) {
    if (indices.size() < 2)
      continue;

    // Candidates are already in block order since we collected in order.
    // Build reuse chains: consecutive candidates where the previous
    // candidate's done point comes before the current candidate's alloc.
    SmallVector<SmallVector<unsigned>> chains;
    SmallVector<unsigned> currentChain = {indices[0]};

    for (unsigned i = 1; i < indices.size(); ++i) {
      auto &prev = candidates[currentChain.back()];
      auto &curr = candidates[indices[i]];

      if (opPosition[prev.donePoint] < curr.blockPosition) {
        currentChain.push_back(indices[i]);
      } else {
        if (currentChain.size() >= 2)
          chains.push_back(std::move(currentChain));
        currentChain = {indices[i]};
      }
    }
    if (currentChain.size() >= 2)
      chains.push_back(std::move(currentChain));

    // Rewrite each chain to share a single mutable buffer.
    auto mutableTy = cast<ttg::MemDescType>(ty);
    for (auto &chain : chains) {
      // First alloc: replace local_alloc %src with
      //   %buf = local_alloc (mutable, no src)
      //   local_store %src, %buf
      auto &first = candidates[chain[0]];
      OpBuilder builder(first.alloc);
      Value src = first.alloc.getSrc();
      auto buf =
          builder.create<ttg::LocalAllocOp>(first.alloc.getLoc(), mutableTy);
      builder.create<ttg::LocalStoreOp>(first.alloc.getLoc(), src, buf);
      first.alloc.replaceAllUsesWith(buf.getResult());
      first.alloc.erase();

      // Subsequent allocs: replace local_alloc %srcN with
      //   local_store %srcN, %buf
      // and RAUW the old alloc value with %buf.
      for (unsigned i = 1; i < chain.size(); ++i) {
        auto &cand = candidates[chain[i]];
        OpBuilder b(cand.alloc);
        Value srcN = cand.alloc.getSrc();
        b.create<ttg::LocalStoreOp>(cand.alloc.getLoc(), srcN, buf);
        cand.alloc.replaceAllUsesWith(buf.getResult());
        cand.alloc.erase();
      }
    }
  }
}

class TritonNvidiaGPUTMAStoreBufferReusePass
    : public impl::TritonNvidiaGPUTMAStoreBufferReusePassBase<
          TritonNvidiaGPUTMAStoreBufferReusePass> {
public:
  void runOnOperation() override {
    SmallVector<Block *> blocks;
    getOperation()->walk([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          blocks.push_back(&block);
    });
    for (Block *block : blocks)
      processBlock(*block);
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
