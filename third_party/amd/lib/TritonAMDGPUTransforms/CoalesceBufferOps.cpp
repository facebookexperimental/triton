#include "TritonAMDGPUTransforms/Passes.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCOALESCEBUFFEROPS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Compute the ideal number of elements per thread for a buffer op.
//
// Buffer ops use a scalar base ptr + i32 offset tensor.  The generic
// buildCoalescedEncoding / getNumElementsPerThread path only handles
// tensor-of-pointer loads and cannot be reused here.  Instead we mirror
// the same divisibility-based logic:
//
//   ptrAlign    = ptr_divisibility_bytes / elem_bytes
//   offsetAlign = offset_divisibility_bytes / elem_bytes
//   perThread   = min(min(ptrAlign, offsetAlign), 128 / elem_bits)
//
// Both divisibility values are layout-independent (they are derived from
// tt.divisibility function-argument attributes and propagated through
// arithmetic), so this gives the right answer regardless of the current
// sizePerThread in the encoding.
static unsigned
computePerThread(Value ptr, Value offsets, unsigned elemBitWidth,
                 triton::ModuleAxisInfoAnalysis &axisAnalysis) {
  unsigned elemNumBytes = std::max(elemBitWidth / 8, 1u);

  // Alignment from the scalar base pointer divisibility.
  unsigned ptrAlign = 1;
  if (auto *ptrInfo = axisAnalysis.getAxisInfo(ptr)) {
    unsigned ptrDivisibility = ptrInfo->getDivisibility(0);
    ptrAlign = std::max(ptrDivisibility / elemNumBytes, 1u);
  }

  // Alignment from the offset tensor's innermost (fast-varying) dimension.
  unsigned offsetAlign = 1;
  if (auto *offsetInfo = axisAnalysis.getAxisInfo(offsets)) {
    auto offsetTy = cast<RankedTensorType>(offsets.getType());
    auto contiguity = offsetInfo->getContiguity();
    SmallVector<unsigned> offsetOrder = getOrderFromContiguity(contiguity);
    unsigned innerDim = offsetOrder[0];
    unsigned divisibility = offsetInfo->getDivisibility(innerDim);
    offsetAlign = std::max(divisibility / elemNumBytes, 1u);
  }

  unsigned alignment = std::min(ptrAlign, offsetAlign);
  // Cap to the widest vectorized load/store (128 bits).
  unsigned maxVec = 128 / elemBitWidth;
  unsigned perThread = std::min(alignment, maxVec);
  LDBG("ptrAlign=" << ptrAlign << " offsetAlign=" << offsetAlign
                   << " perThread=" << perThread);
  return perThread;
}

// Build the optimal blocked encoding for a buffer op.
static ttg::BlockedEncodingAttr
buildBufferOpEncoding(MLIRContext *ctx, Value ptr, Value offsets,
                      RankedTensorType tensorTy, int numWarps, int threadsPerWarp,
                      triton::ModuleAxisInfoAnalysis &axisAnalysis) {
  unsigned elemBitWidth = triton::getPointeeBitWidth(ptr.getType());
  unsigned perThread =
      computePerThread(ptr, offsets, elemBitWidth, axisAnalysis);

  unsigned rank = tensorTy.getRank();
  SmallVector<unsigned> order(rank);
  std::iota(order.begin(), order.end(), 0);
  std::reverse(order.begin(), order.end());
  if (auto *offsetInfo = axisAnalysis.getAxisInfo(offsets)) {
    auto offsetTy = cast<RankedTensorType>(offsets.getType());
    auto contiguity = offsetInfo->getContiguity();
    order = getOrderFromContiguity(contiguity);
  }

  SmallVector<unsigned> sizePerThread(rank, 1);
  sizePerThread[order[0]] = perThread;

  auto shapePerCTA = ttg::getShapePerCTA(tensorTy);
  int numElems = 1;
  for (auto s : shapePerCTA)
    numElems *= s;
  int numThreads = numWarps * threadsPerWarp;
  // Cap so each thread gets at most its fair share.
  sizePerThread[order[0]] =
      std::min<unsigned>(sizePerThread[order[0]],
                         std::max<unsigned>(numElems / numThreads, 1u));

  auto cgaLayout = ttg::getCGALayout(tensorTy.getEncoding());
  return ttg::BlockedEncodingAttr::get(ctx, tensorTy.getShape(), sizePerThread,
                                       order, numWarps, threadsPerWarp,
                                       cgaLayout);
}

} // namespace

class TritonAMDGPUCoalesceBufferOpsPass
    : public impl::TritonAMDGPUCoalesceBufferOpsBase<
          TritonAMDGPUCoalesceBufferOpsPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    triton::AMD::ModuleAxisInfoAnalysis axisAnalysis(mod);
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    // Collect (encoding, op) pairs to avoid invalidating the walk iterator.
    llvm::MapVector<Operation *, Attribute> layoutMap;

    mod.walk([&](Operation *op) {
      auto bufOp = dyn_cast<triton::amdgpu::BufferOpInterface>(op);
      if (!bufOp)
        return;
      if (isa<triton::amdgpu::BufferLoadToLocalOp>(op))
        return;
      // Determine the tensor type that represents the data layout.
      // For loads it's the result; for stores find the first tensor operand.
      RankedTensorType tensorTy;
      if (op->getNumResults() == 1)
        tensorTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!tensorTy) {
        for (Value operand : op->getOperands()) {
          tensorTy = dyn_cast<RankedTensorType>(operand.getType());
          if (tensorTy)
            break;
        }
      }
      if (!tensorTy)
        return;

      // Only rewrite BlockedEncoding — other encodings (dot_op, etc.) are
      // managed by later passes.
      if (!isa<ttg::BlockedEncodingAttr>(tensorTy.getEncoding()))
        return;

      int numWarps = ttg::lookupNumWarps(op);
      Value ptr = bufOp.getPtr();
      Value offsets = bufOp.getOffsets();
      llvm::outs() << "op = " << bufOp << "\n";
      auto newEnc = buildBufferOpEncoding(ctx, ptr, offsets, tensorTy, numWarps,
                                          threadsPerWarp, axisAnalysis);
      // Nothing to do if the encoding is already optimal.
      llvm::outs() << "loc1\n";
      if (newEnc == tensorTy.getEncoding())
        return;

      llvm::outs() << "loc1\n";
      LDBG("Rewriting " << op->getName() << " with new encoding " << newEnc);
      layoutMap[op] = newEnc;
      llvm::outs() << "loc2\n";
    });

    llvm::outs() << "loc3\n";
    for (auto &kv : layoutMap) {
      llvm::outs() << "op = " << *kv.first << "\n";
      convertDistributedOpEncoding(kv.second, kv.first);
    }
    llvm::outs() << "loc4\n";
  }
};

} // namespace mlir
