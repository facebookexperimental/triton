// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Stub for createNVGPUModuloSchedule — provides the symbol so that
// triton_nvidia.cc (ADD_PASS_WRAPPER_0) links before the real
// implementation lands in a child diff.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"

using namespace mlir;

namespace {
struct ModuloSchedulePass
    : public PassWrapper<ModuloSchedulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloSchedulePass)
  StringRef getArgument() const override { return "nvgpu-modulo-schedule"; }
  StringRef getDescription() const override {
    return "Modulo scheduling for warp specialization (stub)";
  }
  void runOnOperation() override {}
};
} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloSchedule() {
  return std::make_unique<ModuloSchedulePass>();
}
} // namespace mlir
