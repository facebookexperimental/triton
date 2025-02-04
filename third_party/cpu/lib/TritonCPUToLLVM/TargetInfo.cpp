#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;

namespace mlir::triton::CPU {

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return b.i32_val(0);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::CPU::llGetPid(loc, rewriter, moduleOp, axis);
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  llvm::report_fatal_error("not implemented");
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args) const {
  llvm::report_fatal_error("not implemented");
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg,
                        ValueRange args) const {
  llvm::report_fatal_error("not implemented");
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  llvm::report_fatal_error("not implemented");
}

} // namespace mlir::triton::CPU
