//===- DotDecomposeAndSchedule.cpp ----------------------------------------===//
//
// TTGIR-level replacement for the LLVM-IR `LLIRSchedule` pass on AMD MFMA
// matmul kernels (v8 / v10 family). Phase 0 — no-op scaffold.
//
// This pass is opt-in behind `TRITON_ENABLE_TTGIR_SCHED=1`. Phase 0 only
// walks the module, identifies inner-loop `tt.dot` ops that look like
// MFMA matmuls, and emits a remark counting them. Subsequent phases will
// (1) decompose the dots along M then N into MFMA-tile-sized sub-dots
// reusing the partition machinery from
// `nvidia/hopper/.../WSDataPartition.cpp` and the AMD shmem-subview
// helpers from `BlockPingpong.cpp::genLocalSlice`, then (2) reorder the
// resulting sub-dots and `ttg.local_load` ops with the
// 4-MFMA-per-GR / 1-MFMA-per-LR recipe currently implemented in
// `LLIRSchedule.cpp`, and (3) emit `triton_amdgpu.sched_barrier` markers
// that lower to `llvm.amdgcn.sched.barrier(0)`.
//
// See:
//   - ~/AMD/triton/claude/llir_sched_at_ttgir_design.md
//   - ~/AMD/triton/claude/llir_sched_at_ttgir_plan.md
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#define DEBUG_TYPE "tritonamdgpu-dot-decompose-and-schedule"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUDOTDECOMPOSEANDSCHEDULE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

/// Returns true if `dotOp`'s result tensor carries an `AMDMFMAEncodingAttr`,
/// which is the marker we use to identify dots that lower to MFMA intrinsics
/// (the targets of the LLIR scheduler today).
static bool isMFMADot(triton::DotOp dotOp) {
  auto resultType = dyn_cast<RankedTensorType>(dotOp.getResult().getType());
  if (!resultType)
    return false;
  return isa<triton::gpu::AMDMfmaEncodingAttr>(resultType.getEncoding());
}

/// Heuristic: is this `scf.for` the inner main-loop of an MFMA matmul that
/// the SCHED pass cares about? Phase 0 just looks for ≥1 MFMA-typed `tt.dot`
/// inside the for-body. Tighter guards (operand provenance, anchor presence)
/// land in Phase 1.
static bool isCandidateInnerLoop(scf::ForOp forOp, unsigned &numMfmaDots) {
  numMfmaDots = 0;
  forOp.getBody()->walk([&](triton::DotOp dotOp) {
    if (isMFMADot(dotOp))
      ++numMfmaDots;
  });
  return numMfmaDots > 0;
}

struct TritonAMDGPUDotDecomposeAndSchedulePass
    : impl::TritonAMDGPUDotDecomposeAndScheduleBase<
          TritonAMDGPUDotDecomposeAndSchedulePass> {

  void runOnOperation() override {
    // Belt-and-suspenders gate: even with the pass in the pipeline, only do
    // anything if the env var is set. compiler.py also gates the pass call
    // itself, but a lit-test invocation of the pass should be similarly
    // opt-in to make accidental triggering obvious.
    if (!triton::tools::getBoolEnv("TRITON_ENABLE_TTGIR_SCHED"))
      return;

    ModuleOp module = getOperation();
    unsigned numForOps = 0;
    unsigned numCandidateLoops = 0;
    unsigned totalMfmaDots = 0;
    module.walk([&](scf::ForOp forOp) {
      ++numForOps;
      unsigned numMfmaDots = 0;
      if (isCandidateInnerLoop(forOp, numMfmaDots)) {
        ++numCandidateLoops;
        totalMfmaDots += numMfmaDots;
        forOp.emitRemark()
            << "ttgir-sched: candidate inner loop with " << numMfmaDots
            << " MFMA tt.dot op(s) (phase 0: no-op)";
      }
    });
    module.emitRemark()
        << "ttgir-sched: visited " << numForOps << " scf.for op(s), "
        << numCandidateLoops << " candidate(s), " << totalMfmaDots
        << " MFMA tt.dot op(s) (phase 0: no-op)";
  }
};

} // namespace

} // namespace mlir
