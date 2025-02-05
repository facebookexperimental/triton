#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace {

void lowerGetAsyncTaskIdOp(Operation *parentOp) {
  DenseSet<Operation *> eraseOps;
  parentOp->walk([&](ttng::GetAsyncTaskIdOp op) {
    auto loc = op.getLoc();
    OpBuilder builder(op);
    auto i32ty = builder.getIntegerType(32);
    auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
    auto constWaveSize = builder.create<arith::ConstantIntOp>(loc, 64, 32);
    auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWaveSize);
    op.getResult().replaceAllUsesWith(warpIDX);
    eraseOps.insert(op);
  });
  for (Operation *op : eraseOps)
    op->erase();
}

class TritonAMDGPUWSLoweringPass
    : public TritonAMDGPUWSLoweringBase<TritonAMDGPUWSLoweringPass> {
public:
  TritonAMDGPUWSLoweringPass() = default;
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::errs() << "pre: " << mod << "\n";
    lowerGetAsyncTaskIdOp(mod);
    llvm::errs() << "post: " << mod << "\n";
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUWSLoweringPass() {
  return std::make_unique<TritonAMDGPUWSLoweringPass>();
}
