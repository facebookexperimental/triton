#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"

#include "Utility.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_FIXWSBARRIER
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

struct FixWSBarrier
    : public mlir::triton::impl::FixWSBarrierBase<FixWSBarrier> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    if (!mod->hasAttr("triton_gpu.num-warp-groups-per-cta"))
      return;

    Operation *kernelFunc;
    int numKernel = 0;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      // We filter the libdevice functions
      if (!func.getName().starts_with("__nv")) {
        kernelFunc = func;
        numKernel++;
      }
    }

    // This warp specialization fix pass now only supports: all functions should
    // be inlined.
    if (numKernel != 1)
      return;

    assert(kernelFunc->getAttrOfType<mlir::IntegerAttr>("nvvm.kernel")
               .getValue()
               .getZExtValue() == 1);

    llvm::DenseMap<Block *, int> barIdReuse;
    llvm::SmallVector<llvm::StringRef> operands;
    llvm::SmallSet<int, 16> allocBarId;

    // Helper function to setup metadata for used barrier id
    auto processEachBarSync = [&](StringRef instruction, Block *block) {
      auto operandsStr = instruction.substr(instruction.find("bar.sync ") + 9);
      operandsStr = operandsStr.rtrim(";");
      operands.clear();
      operandsStr.split(operands, ',');
      int barId = -1;
      operands[0].trim().getAsInteger(0, barId);
      int threadCount = -1;
      operands[1].trim().getAsInteger(0, threadCount);
      if (threadCount == 128) {
        allocBarId.insert(barId);
        if (!barIdReuse.count(block)) {
          barIdReuse[block] = barId;
        }
      }
    };

    // Scan through the kernel function to find the used barrier id
    for (Block &block : kernelFunc->getRegion(0).getBlocks()) {
      for (LLVM::InlineAsmOp asmop : block.getOps<LLVM::InlineAsmOp>()) {
        StringRef instruction = asmop.getAsmString();
        if (instruction.starts_with("bar.sync")) {
          processEachBarSync(instruction, &block);
        }
      }
    }

    int curBarId = 1;
    OpBuilder builder(mod.getContext());
    for (Block &block : kernelFunc->getRegion(0).getBlocks()) {
      // We allow bar.sync 0 in the entry block. No warp specialization happens
      // yet in the entry block. For the rest, we need to rewrite the bar.sync 0
      // with a goal to reuse the barrier id.
      if (!block.isEntryBlock()) {
        for (NVVM::Barrier0Op barrier :
             llvm::make_early_inc_range(block.getOps<NVVM::Barrier0Op>())) {
          builder.setInsertionPoint(barrier);
          if (barIdReuse.count(&block))
            barSync(builder, barrier, barIdReuse[&block], 128);
          else {
            while (allocBarId.count(curBarId))
              curBarId++;

            if (curBarId > 15)
              llvm::report_fatal_error(
                  "Too many barriers, at most 16 barriers");

            barSync(builder, barrier, curBarId, 128);
            allocBarId.insert(curBarId);
            barIdReuse[&block] = curBarId;
            curBarId++;
          }
          barrier->erase();
        }
      }
    }
  }
};
} // namespace

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>> createFixWSBarrierPass() {
  return std::make_unique<FixWSBarrier>();
}

} // namespace mlir::triton
