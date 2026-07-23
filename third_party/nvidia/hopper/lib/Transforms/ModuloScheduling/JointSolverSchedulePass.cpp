// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Joint-solver scheduling pass — ModuloSchedulePass's complete-solver sibling.
//
// Runs the same shared driver (ModuloScheduleDriver.h) with the complete
// solver engaged end-to-end: the per-loop schedule comes from the joint-solver
// backend (TRITON_USE_MODULO_SCHEDULE is ignored; the algorithm is forced
// to "joint_solver", with Rau as warm-start incumbent and fallback), and the
// warp-group partition comes from the joint partition re-solve (v2: cycles +
// warp groups in one model → v1: warp groups only → exhaustive scorer).
// Every solver failure degrades down that chain, so the pass never fails
// where the modulo pass would have succeeded.
//
// The emitted annotation contract is byte-identical to the modulo pass
// (loop.stage / loop.cluster / tt.modulo_ii / tt.num_buffers / tt.autows),
// so data partitioning and every downstream WS consumer are unchanged —
// switching scheduler is purely a pass-selection decision in compiler.py
// (TRITON_USE_JOINT_SCHEDULE=1 vs TRITON_USE_MODULO_SCHEDULE).

#include "ModuloScheduleDriver.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"

using namespace mlir;
namespace ttg = triton::gpu;

namespace {

struct JointSolverSchedulePass
    : public PassWrapper<JointSolverSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(JointSolverSchedulePass)

  JointSolverSchedulePass() = default;
  JointSolverSchedulePass(const JointSolverSchedulePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "nvgpu-joint-solver-schedule";
  }

  StringRef getDescription() const override {
    return "Joint-solver schedule + warp-group partition (Pass A sibling)";
  }

  // Same test-only knob as the modulo pass (lit tests in opt builds).
  Option<bool> printScheduleGraph{
      *this, "print-schedule-graph",
      llvm::cl::desc("Dump the ScheduleGraph to stderr unconditionally "
                     "(test-only; bypasses LLVM_DEBUG)"),
      llvm::cl::init(false)};

  Option<int> dataPartitionFactor{
      *this, "data-partition-factor",
      llvm::cl::desc(
          "Pass A.5 data-partition factor N (M-split). 0 = use the "
          "tt.data_partition_factor loop attr / "
          "TRITON_DATA_PARTITION_N env; >1 forces N for all eligible "
          "MMAs."),
      llvm::cl::init(0)};

  // Replaces the retired TRITON_MODULO_CPSAT_JOINT env var 1:1 for A/B
  // measurement (see solver_regression.py).
  Option<int> jointMode{
      *this, "joint-mode",
      llvm::cl::desc("Joint partition solve: 0 = v2 then v1 fallback "
                     "(default), 1 = v1 only, 2 = v2 only"),
      llvm::cl::init(0)};

  void runOnOperation() override {
    ttg::ScheduleDriverOptions opts;
    opts.forceScheduleAlgo = "joint_solver";
    switch (jointMode) {
    case 1:
      opts.jointMode = ttg::JointSolverMode::V1Only;
      break;
    case 2:
      opts.jointMode = ttg::JointSolverMode::V2Only;
      break;
    default:
      opts.jointMode = ttg::JointSolverMode::V2ThenV1;
      break;
    }
    opts.printScheduleGraph = printScheduleGraph;
    opts.dataPartitionFactor = dataPartitionFactor;
    if (failed(ttg::runScheduleDriver(getOperation(), opts)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUJointSolverSchedule() {
  return std::make_unique<JointSolverSchedulePass>();
}

void registerNVGPUJointSolverSchedule() {
  PassRegistration<JointSolverSchedulePass>();
}
} // namespace mlir
