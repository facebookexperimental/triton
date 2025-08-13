#include "TargetInfo.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTAMDWARPSPECIALIZETOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;


//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertAMDWarpSpecializeToLLVM
    : public mlir::triton::impl::ConvertAMDWarpSpecializeToLLVMBase<
          ConvertAMDWarpSpecializeToLLVM> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::errs() << "<<<< Running ConvertAMDWarpSpecializeToLLVM >>>>\n\n";
    // AMD::TargetInfo targetInfo(archGenerationName);
  }
};
} // namespace

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertAMDWarpSpecializeToLLVMPass() {
  return std::make_unique<ConvertAMDWarpSpecializeToLLVM>();
}
} // namespace mlir::triton
