//===- WSVerifyBarriers.cpp -----------------------------------------------===//
//
// Static verifier for autoWS mbarrier deadlocks and cross-partition races.
// Consumes post-code-partition / post-pipeline-expander IR. Verify-only: the
// module is never mutated.
//
// Design: lib/Transforms/WarpSpecialization/docs/WSAliasingCoverage.proposal.md
//
// Goal 1 (this file, incremental):
//   - run after the pipeline expander, autoWS only;
//   - barrier -> reuse group (buffer.id) is assumed given, carried as a
//     `buffer.id` integer attribute on each `ttng.init_barrier`.
//
// Phase 0 (current): parse the given barrier -> buffer.id mapping, group the
// WS-generated barriers by reuse group, and report it as remarks. Later phases
// add the Barrier Dependency Graph (deadlock SCC) and the per-reuse-group
// Access-Order Graph (race / under-buffering) described in the design doc.
//
//===----------------------------------------------------------------------===//

#include "CodePartitionUtility.h"
#include "WSBarrierAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-verify-ws-barriers"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {

#define GEN_PASS_DEF_NVGPUVERIFYWSBARRIERS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {

// A barrier whose guarded reuse group is known (Goal 1: given via buffer.id on
// the init_barrier). More fields (slot/phase, direction, dstTask, role) are
// added in the BDG-building phase.
struct KnownBarrier {
  ttng::InitBarrierOp init;
  int bufferId;
};

// Read the goal-1 `buffer.id` annotation off an init_barrier. Returns nullopt
// when absent (verifier then treats the barrier as buffer=unknown ->
// indeterminate, per the design doc).
static std::optional<int> getGuardedBufferId(ttng::InitBarrierOp op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("buffer.id"))
    return static_cast<int>(attr.getInt());
  return std::nullopt;
}

} // namespace

class NVGPUVerifyWSBarriersPass
    : public impl::NVGPUVerifyWSBarriersBase<NVGPUVerifyWSBarriersPass> {
public:
  using impl::NVGPUVerifyWSBarriersBase<
      NVGPUVerifyWSBarriersPass>::NVGPUVerifyWSBarriersBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    // Phase 0: collect the given barrier -> buffer.id mapping and report it,
    // grouped by reuse group. This establishes the input contract the later
    // BDG / AOG phases consume.
    DenseMap<int, SmallVector<KnownBarrier>> byGroup;
    unsigned numUnknown = 0;
    funcOp.walk([&](ttng::InitBarrierOp init) {
      if (auto id = getGuardedBufferId(init)) {
        byGroup[*id].push_back({init, *id});
      } else {
        ++numUnknown;
        init.emitRemark()
            << "WS barrier has no buffer.id (reuse group unknown) -> "
               "indeterminate";
      }
    });

    for (auto &kv : byGroup) {
      LDBG("reuse group buffer.id=" << kv.first << " has " << kv.second.size()
                                    << " WS barriers");
      for (KnownBarrier &kb : kv.second)
        kb.init.emitRemark()
            << "WS barrier guards reuse group buffer.id=" << kb.bufferId;
    }
    LDBG("verified func " << funcOp.getName() << ": " << byGroup.size()
                          << " reuse groups, " << numUnknown
                          << " unknown-buffer barriers");
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
