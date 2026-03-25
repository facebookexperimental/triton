#include "triton/Analysis/BarrierAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONBARRIERANALYSISPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

class TritonBarrierAnalysisPass
    : public impl::TritonBarrierAnalysisPassBase<TritonBarrierAnalysisPass> {
public:
  using TritonBarrierAnalysisPassBase::TritonBarrierAnalysisPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    for (auto funcOp : mod.getOps<triton::FuncOp>()) {
      auto result = buildBarrierAnalysis(funcOp);

      // Dump if environment variable is set or if enabled via pass option.
      if (std::getenv("TRITON_DUMP_BARRIER_ANALYSIS") || dumpTable) {
        dumpBarrierTable(result, llvm::errs());
      }
    }
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
