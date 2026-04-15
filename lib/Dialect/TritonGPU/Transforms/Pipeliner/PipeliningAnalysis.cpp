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
      // Skip successfully pipelined loops.
      if (forOp->hasAttr(kScheduledMaxStageAttrName))
        return;

      // Check common failure reasons.
      int stages = getNumStagesOrDefault(forOp, numStages);
      if (stages <= 1) {
        forOp.emitRemark("pipelining not applied: num_stages is ")
            << stages << ", no pipelining requested";
        return;
      }

      if (loopHasDistGreaterThanOne(forOp)) {
        forOp.emitRemark(
            "pipelining not applied: loop has distance > 1 (not supported)");
        return;
      }

      if (isOuterLoop(forOp)) {
        forOp.emitRemark(
            "pipelining not applied: outer loop (contains nested loop)");
        return;
      }

      // Check if any op in the loop body has a latency attribute, indicating
      // the assign-latencies pass identified pipelineable ops.
      bool hasLatencyOp = false;
      forOp.getBody()->walk([&](Operation *op) {
        if (op->hasAttr(kLoopStageAttrName)) {
          hasLatencyOp = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!hasLatencyOp) {
        forOp.emitRemark(
            "pipelining not applied: no latency assigned to any op in loop");
        return;
      }

      // If none of the above, emit a generic remark.
      forOp.emitRemark("pipelining not applied: unknown reason");
    });

    // This is a read-only analysis pass; do not modify the IR.
    markAllAnalysesPreserved();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
