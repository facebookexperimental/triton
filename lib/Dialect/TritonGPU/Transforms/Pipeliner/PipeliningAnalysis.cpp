#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPIPELININGANALYSIS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class PipeliningAnalysisPass
    : public impl::TritonGPUPipeliningAnalysisBase<PipeliningAnalysisPass> {
public:
  using TritonGPUPipeliningAnalysisBase::TritonGPUPipeliningAnalysisBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    moduleOp.walk([&](scf::ForOp forOp) {
      int stages = getNumStagesOrDefault(forOp, numStages);
      if (stages <= 1)
        return;

      if (isOuterLoop(forOp)) {
        forOp.emitRemark(
            "pipelining not applied: outer loop (contains nested loop)");
        return;
      }
    });

    // This is a read-only analysis pass; do not modify the IR.
    markAllAnalysesPreserved();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
