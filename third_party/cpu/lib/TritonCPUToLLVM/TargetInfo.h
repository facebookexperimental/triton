#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::CPU {

class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo() {}

  bool supportMaximumMinimum() const override { return false; }

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override {
    llvm::report_fatal_error("Not supported on CPU");
  }

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Value val,
                    Value pred) const override {
    llvm::report_fatal_error("Not supported on CPU");
  }
  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Type elemTy,
                    Value pred) const override {
    llvm::report_fatal_error("Not supported on CPU");
  }

  bool canUseStMatrix(RankedTensorType tensorTy, ArrayRef<unsigned> repShape,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> order,
                      int swizzleByteSize) const override {
    llvm::report_fatal_error("Not supported on CPU");
  }

  void storeMatrixShared(RewriterBase &rewriter, Location loc, Value ptr,
                         Value val) const override {
    llvm::report_fatal_error("Not supported on CPU");
  }

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override {
    llvm::report_fatal_error("Not supported on CPU");
  }
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override {
    llvm::report_fatal_error("Not supported on CPU");
  }
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override {
    llvm::report_fatal_error("Not supported on CPU");
  }
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override {
    llvm::report_fatal_error("Not supported on CPU");
  }

  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  int axis) const override;

  bool warpReduce(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                  triton::ReduceOp op, unsigned numLaneToReduce,
                  unsigned interleave) const override {
    return false;
  }

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args) const override;

  void printf(RewriterBase &rewriter, StringRef msg,
              ValueRange args) const override;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const override;

  int getSharedAddressSpace() const override {
    llvm::report_fatal_error("Not supported on CPU");
  }

  bool supportVectorizedAtomics() const override { return false; };

  bool isCPUMode() const override { return true; }
};

} // namespace mlir::triton::CPU

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
