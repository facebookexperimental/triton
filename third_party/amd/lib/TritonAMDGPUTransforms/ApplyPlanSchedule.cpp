#include "TritonAMDGPUTransforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "triton/Analysis/PlanSchedule.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUAPPLYPLANSCHEDULE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

static LogicalResult writeReport(ModuleOp module, StringRef outputPath,
                                 StringRef payload) {
  if (outputPath.empty())
    return success();
  if (outputPath == "-") {
    llvm::outs() << payload;
    return success();
  }
  llvm::SmallString<256> path(outputPath);
  llvm::SmallString<256> parent = llvm::sys::path::parent_path(path);
  if (!parent.empty())
    if (std::error_code ec = llvm::sys::fs::create_directories(parent)) {
      module.emitError("cannot create plan-apply report directory '")
          << parent << "': " << ec.message();
      return failure();
    }
  std::error_code ec;
  llvm::ToolOutputFile output(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    module.emitError("cannot open plan-apply report '")
        << path << "': " << ec.message();
    return failure();
  }
  output.os() << payload;
  output.keep();
  return success();
}

struct TritonAMDGPUApplyPlanSchedulePass
    : impl::TritonAMDGPUApplyPlanScheduleBase<
          TritonAMDGPUApplyPlanSchedulePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    triton::plan::PlanScheduleApplyResult result;
    std::string error;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
        llvm::MemoryBuffer::getFile(inputPath);
    if (!input) {
      error = "cannot read schedule delta '" + inputPath +
              "': " + input.getError().message();
      result.error = error;
      (void)writeReport(module, reportPath,
                        triton::plan::serializePlanScheduleApplyReport(result));
      module.emitError(error);
      return signalPassFailure();
    }

    FailureOr<triton::plan::PlanScheduleDelta> delta =
        triton::plan::parsePlanScheduleDelta((*input)->getBuffer(), error);
    if (failed(delta)) {
      result.error = error;
      (void)writeReport(module, reportPath,
                        triton::plan::serializePlanScheduleApplyReport(result));
      module.emitError(error);
      return signalPassFailure();
    }

    triton::FuncOp target;
    for (triton::FuncOp function : module.getOps<triton::FuncOp>())
      if (function.getName() == delta->kernel) {
        target = function;
        break;
      }
    if (!target) {
      if (allowMissingKernel)
        return;
      result.kernel = delta->kernel;
      result.error = "schedule delta kernel is not present in the module";
      (void)writeReport(module, reportPath,
                        triton::plan::serializePlanScheduleApplyReport(result));
      module.emitError(result.error);
      return signalPassFailure();
    }

    if (failed(triton::plan::applyPlanSchedule(target, *delta, result, error,
                                               /*strict=*/strict))) {
      (void)writeReport(module, reportPath,
                        triton::plan::serializePlanScheduleApplyReport(result));
      target.emitError(error);
      return signalPassFailure();
    }
    if (failed(writeReport(
            module, reportPath,
            triton::plan::serializePlanScheduleApplyReport(result))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir
