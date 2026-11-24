#include "Analysis/AMDGPUAllocation.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUFOLDLDSSTAGINGOVERBUDGET
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// ReduceDataDuplication's dot-operand staging: an immutable local_alloc of a
// register tensor read back only by local_loads in a dot-operand layout.
// Loads with coordinate-rematerialization metadata are left alone because
// their lowering needs it.
bool isDotOperandStaging(ttg::LocalAllocOp allocOp) {
  if (!allocOp.getSrc() || allocOp.getType().getMutableMemory() ||
      allocOp->use_empty())
    return false;
  return llvm::all_of(allocOp->getUsers(), [](Operation *user) {
    auto load = dyn_cast<ttg::LocalLoadOp>(user);
    return load && !load.getToken() &&
           !load->hasAttr("tlx.rematerialize_coordinates") &&
           !load->hasAttr("tlx.rematerialize_coordinates_group") &&
           isa<ttg::DotOperandEncodingAttr>(
               cast<RankedTensorType>(load.getType()).getEncoding());
  });
}

class TritonAMDGPUFoldLDSStagingOverBudgetPass
    : public impl::TritonAMDGPUFoldLDSStagingOverBudgetBase<
          TritonAMDGPUFoldLDSStagingOverBudgetPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // The late tlx_propagate_layout owns the staging of hand-placed modules.
    if (triton::placesLdsByHand(mod))
      return;
    SmallVector<ttg::LocalAllocOp> stagings;
    mod.walk([&](ttg::LocalAllocOp allocOp) {
      if (isDotOperandStaging(allocOp))
        stagings.push_back(allocOp);
    });
    if (stagings.empty())
      return;

    std::string arch = gfxArch;
    if (arch.empty())
      if (auto moduleArch = getAMDArch(mod))
        arch = moduleArch->split(':').first.str();
    triton::AMD::TargetInfo targetInfo(arch);
    auto scratchSizeFn = [&targetInfo](Operation *op) {
      return triton::AMD::AMDAllocationAnalysisScratchSizeFn(op, targetInfo);
    };
    // An estimate: passes between here and allocate_shared_memory (warp and
    // intra-wave pipelining, the sched-group scheduler, the sanitizers) can
    // still change LDS usage.
    ModuleAllocation allocation(mod, scratchSizeFn,
                                targetInfo.getSharedMemoryPartitionSize());
    if (allocation.getSharedMemorySize() <=
        static_cast<size_t>(targetInfo.getSharedMemorySize()))
      return;

    // Over budget: give the staging back to convert_layout, whose scratch is
    // reused across conversions, so the kernel still fits.
    // TODO: fold only what is needed, largest staging live at the peak first,
    // re-running the allocation until it fits; staging reads beat
    // convert_layout, so a kernel slightly over budget loses all of them today.
    OpBuilder builder(mod.getContext());
    for (ttg::LocalAllocOp allocOp : stagings) {
      for (Operation *user : llvm::make_early_inc_range(allocOp->getUsers())) {
        auto load = cast<ttg::LocalLoadOp>(user);
        builder.setInsertionPoint(load);
        Value cvt = ttg::ConvertLayoutOp::create(builder, load.getLoc(),
                                                 load.getType(),
                                                 allocOp.getSrc());
        load.replaceAllUsesWith(cvt);
        load.erase();
      }
      allocOp.erase();
    }
  }
};

} // namespace
} // namespace mlir
