#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Dominance.h"
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

    // Detect and get all WS local blocks, serving as the candidate for barrier
    // rewrite.
    SetVector<Operation *> forwardSlice;
    llvm::SmallPtrSet<Block *, 8> WSLocalImmediateBlock;
    llvm::SmallPtrSet<Block *, 16> WSLocalBlock;
    LLVM::InlineAsmOp canonicalWarpIdOp;
    // Hint for checking if the block is a WS local block. A prefix of the ptx
    // instructions in `Canonical_Warp_Id_Op`
    const std::string PrefixCanonicalWarpIdOpAsm =
        "{\n"
        ".reg .u32 a<5>;              \n"
        "mov.u32 a0, %tid.x;          \n" // x
        "mov.u32 a1, %tid.y;          \n" // y
        "mov.u32 a2, %tid.z;          \n" // z
        "mov.u32 a3, %ntid.x;         \n" // nx
        "mov.u32 a4, %ntid.y;         \n" // ny
        "mad.lo.u32 a1, a2, a4, a1;   \n";

    for (Block &block : kernelFunc->getRegion(0).getBlocks()) {
      for (LLVM::InlineAsmOp asmop : block.getOps<LLVM::InlineAsmOp>()) {
        StringRef instruction = asmop.getAsmString();
        if (instruction.starts_with(PrefixCanonicalWarpIdOpAsm)) {
          canonicalWarpIdOp = asmop;
          break;
        }
      }
      if (canonicalWarpIdOp)
        break;
    }

    if (!canonicalWarpIdOp)
      return;

    // Forward DFS to get all the immediate user blocks of the
    // canonicalWarpIdOp.
    mlir::getForwardSlice(canonicalWarpIdOp, &forwardSlice);
    for (Operation *op : forwardSlice) {
      if (auto condBrOp = dyn_cast<LLVM::CondBrOp>(op)) {
        WSLocalImmediateBlock.insert(condBrOp.getTrueDest());
        WSLocalImmediateBlock.insert(condBrOp.getFalseDest());
      }
    }

    // Get the full set of WS local blocks.
    mlir::DominanceInfo domInfo(kernelFunc);
    for (Block &block : kernelFunc->getRegion(0).getBlocks()) {
      for (Block *immBlock : WSLocalImmediateBlock)
        if (domInfo.dominates(immBlock, &block))
          WSLocalBlock.insert(&block);
    }

    // Scan through the kernel function to find the used barrier id
    llvm::DenseMap<Block *, int> barIdReuse;
    llvm::SmallVector<llvm::StringRef> operands;
    // The barrier id 9, 10, 12, 14 are reserved
    llvm::SmallSet<int, 16> allocBarId = {9, 10, 12, 14};

    // Helper function to setup metadata for used barrier id
    auto processEachBarSync = [&](StringRef instruction, Block *block) {
      auto operandsStr = instruction.substr(instruction.find("bar.sync ") + 9);
      operandsStr = operandsStr.rtrim(";");
      operands.clear();
      operandsStr.split(operands, ',');
      assert(operands.size() == 2);
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

    for (Block &block : kernelFunc->getRegion(0).getBlocks()) {
      for (LLVM::InlineAsmOp asmop : block.getOps<LLVM::InlineAsmOp>()) {
        StringRef instruction = asmop.getAsmString();
        if (instruction.starts_with("bar.sync")) {
          processEachBarSync(instruction, &block);
        }
      }
    }

    // Rewrite the problematic barrier.
    int curBarId = 1;
    OpBuilder builder(mod.getContext());
    for (Block *block : WSLocalBlock) {
      for (NVVM::Barrier0Op barrier :
           llvm::make_early_inc_range(block->getOps<NVVM::Barrier0Op>())) {
        builder.setInsertionPoint(barrier);
        if (barIdReuse.count(block))
          barSync(builder, barrier, barIdReuse[block], 128);
        else {
          while (allocBarId.count(curBarId))
            curBarId++;

          if (curBarId > 15)
            llvm::report_fatal_error("Too many barriers, at most 16 barriers");

          barSync(builder, barrier, curBarId, 128);
          allocBarId.insert(curBarId);
          barIdReuse[block] = curBarId;
          curBarId++;
        }
        barrier->erase();
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
