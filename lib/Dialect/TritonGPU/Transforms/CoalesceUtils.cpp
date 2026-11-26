

#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceSectorUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"
#include <limits>

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {
BlockedEncodingAttr
buildCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       triton::gpu::CGAEncodingAttr cgaLayout,
                       SmallVector<int64_t> shapePerCTA, unsigned maxVecBits,
                       unsigned sectorBytes) {
  Value ptr = getMemAccessPtr(op);
  auto refTensorType = cast<RankedTensorType>(ptr.getType());

  LDBG("Considering op: " << *op);
  LLVM_DEBUG({
    DBGS() << "axis info of pointer: ";
    axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
  SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
  LDBG("order=[" << triton::join(order, ", ") << "]");

  auto matchesShape = [&refTensorType](const Value &val) {
    auto rttType = dyn_cast<RankedTensorType>(val.getType());
    return rttType && rttType.getShape() == refTensorType.getShape();
  };

  // The desired divisibility is the maximum divisibility among all dependent
  // pointers which have the same shape and order as `ptr`.
  llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
  memAccessesSameOrder.insert(op);
  if (ptr.getDefiningOp()) {
    for (Operation *use : mlir::getSlice(op)) {
      Value val = getMemAccessPtr(use);
      if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use))
        continue;
      auto currOrder = getOrderFromContiguity(
          axisInfoAnalysis.getAxisInfo(val)->getContiguity());
      if (order == currOrder) {
        LDBG("multi-root-slice: insert to memAccessesSameOrder " << *use);
        memAccessesSameOrder.insert(use);
      }
    }
  }

  LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

  int numElems = product<int64_t>(shapePerCTA);
  int numThreads = numWarps * threadsPerWarp;

  unsigned opPerThread = getNumElementsPerThread(op, order, axisInfoAnalysis,
                                                 shapePerCTA, maxVecBits);
  unsigned perThread = opPerThread;
  bool hasRelatedUnalignedVectorizedLoad = false;
  LDBG("perThread for op: " << perThread);

  for (Operation *opSameOrder : memAccessesSameOrder) {
    if (opSameOrder == op)
      continue;
    unsigned currPerThread = getNumElementsPerThread(
        opSameOrder, order, axisInfoAnalysis, shapePerCTA, maxVecBits);
    LDBG("perThread for opSameOrder: " << currPerThread);
    perThread = std::max(perThread, currPerThread);
    hasRelatedUnalignedVectorizedLoad |=
        canUseUnalignedVectorizedLoad(opSameOrder);
  }

  perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
  LDBG("perThread: " << perThread);

  auto load = dyn_cast<triton::LoadOp>(op);
  if (!load || (hasRelatedUnalignedVectorizedLoad &&
                (load.getIsVolatile() || load.getMask()))) {
    // For ops that can result in a global memory write, we should enforce
    // that each thread handles at most maxVecBits bits, which is the widest
    // available vectorized store op; otherwise, the store will have "gaps"
    // in the memory write at the warp level, resulting in worse performance.
    // For loads, we can expect that the gaps won't matter due to the L1
    // cache. Masked and volatile loads must retain their own safe transaction
    // width when a related unaligned load selects a wider layout.
    perThread = std::min(perThread, opPerThread);
  }
  SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
  sizePerThread[order[0]] = perThread;
  auto buildLayout = [&](ArrayRef<unsigned> layoutOrder) {
    return BlockedEncodingAttr::get(op->getContext(), refTensorType.getShape(),
                                    sizePerThread, layoutOrder, numWarps,
                                    threadsPerWarp, cgaLayout);
  };
  auto layout = buildLayout(order);
  if (!sectorBytes || !load || perThread != 1 || order.size() != 2 ||
      contiguity[order[0]] != 1)
    return layout;

  unsigned bits = getPointeeBitWidth(ptr.getType());
  if (bits < 8 || bits > 128 || bits % 8 != 0)
    return layout;

  uint64_t strides[2];
  for (unsigned dim = 0; dim < 2; ++dim) {
    if (shapePerCTA[dim] <= 1)
      return layout;
    unsigned compileTimeBudget = 64;
    auto step = inferLaneStep(ptr, dim, axisInfoAnalysis, compileTimeBudget);
    if (!step || *step <= 0 || *step > std::numeric_limits<uint32_t>::max())
      return layout;
    strides[dim] = *step;
  }

  auto alternate = buildLayout({order[1], order[0]});
  for (unsigned dim = 0; dim < 2; ++dim) {
    if (layout.getThreadsPerWarp()[dim] > shapePerCTA[dim] ||
        alternate.getThreadsPerWarp()[dim] > shapePerCTA[dim])
      return layout;
  }
  unsigned sectors =
      countScalarWaveSectors(layout, strides, bits / 8, sectorBytes);
  unsigned alternateSectors =
      countScalarWaveSectors(alternate, strides, bits / 8, sectorBytes);
  LDBG("scalar load sectors: " << sectors << " vs. " << alternateSectors);
  return alternateSectors < sectors ? alternate : layout;
}
} // namespace mlir::triton::gpu
