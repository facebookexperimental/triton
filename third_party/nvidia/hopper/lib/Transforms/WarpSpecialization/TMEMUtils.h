#ifndef NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/PassManager.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tt = mlir::triton;
namespace mlir {
// Generate code to reintepret a TMEM buffer operation by converting
// the N dimension to the given value that must be less the current size.
ttg::MemDescReinterpretOp sliceAndReinterpretTMEMBuffer(OpBuilder &builder,
                                                        Operation *allocOp,
                                                        int offset,
                                                        size_t blockN);
// Create a TMEM descriptor that is sufficient for the given
// TMEM Allocation Operator.
ttg::MemDescType createTMEMDesc(OpBuilder &builder, Type inputType,
                                int64_t blockM, int64_t blockN);

// Wrapper class to hold the context for handling
// 1D TMEM Allocation.
class TMEM1DAllocator {
private:
  OpBuilder &builder;
  // Intermediate info to minimize code reuse across functions.
  int numWarps = -1;
  tt::ExpandDimsOp _expandedInput = nullptr;
  // _allocOp should be one of the following types:
  // 1. ttng::TMEMAllocOp: A direct memory allocation
  // 2. ttng::MemDescReinterpretOp: A reinterpret of a
  // memory allocation.
  // 3. ttg.MemDescIndexOp: An index into a memory allocation.
  Operation *_allocOp = nullptr;

public:
  TMEM1DAllocator(OpBuilder &builder) : builder(builder) {}

private:
  void copyAttrs(Operation *oldOp, Operation *newOp) {
    // If you just want to wholesale replace the dictionary:
    newOp->setAttrs(oldOp->getAttrDictionary());
  }

  void setExpandedInput(tt::ExpandDimsOp expandedInput) {
    this->_expandedInput = expandedInput;
  }

  tt::ExpandDimsOp getExpandedInput() {
    assert(_expandedInput != nullptr && "Must call setExpandedInput");
    return _expandedInput;
  }

  void setAllocOp(Operation *allocOp) { this->_allocOp = allocOp; }

  Operation *getAllocOp() {
    assert(_allocOp != nullptr && "Must call getAllocOp()");
    return _allocOp;
  }

  RankedTensorType getResultTensorType(Operation *op, size_t expectedSize) {
    auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!outputType || outputType.getShape().size() != 2) {
      assert("Invalid tensor input");
    }
    return outputType;
  }

  ttng::TMEMAllocOp alloc1DTMEMBuffer();

  void TMEMStore1D(Operation *producer, Operation *allocOpBuffer);

  void TMEMLoad1D(Operation *producer, Operation *consumer);

public:
  void replaceWith1DTMEM(Operation *producer, Operation *consumer,
                         Operation *allocOpBuffer = nullptr) {
    this->numWarps = ttg::lookupNumWarps(producer);
    assert((numWarps == 4 || numWarps == 8) && "Only support 4 or 8 warps");
    TMEMStore1D(producer, allocOpBuffer);
    TMEMLoad1D(producer, consumer);
  }
};

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_
