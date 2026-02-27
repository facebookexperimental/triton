//===- AnnotatePidAddressPatterns.cpp - Annotate load/store patterns -----===//
//
// Part of the Triton Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This pass uses ProgramIdAddressAnalysis to annotate load/store operations
// with their access patterns relative to program_id. This information can be
// used by downstream passes for:
// - Memory coalescing optimization
// - Vectorization decisions
// - Out-of-bounds checking hints
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Analysis/ProgramIdToAddressAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-annotate-pid-address"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUANNOTATEPIDADDRESSPATTERNS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

/// Convert pattern enum to string for annotation
StringRef patternToString(ProgramIdAddressInfo::Pattern pattern) {
  switch (pattern) {
  case ProgramIdAddressInfo::Pattern::UNKNOWN:
    return "unknown";
  case ProgramIdAddressInfo::Pattern::CONSTANT:
    return "constant";
  case ProgramIdAddressInfo::Pattern::PID_AFFINE:
    return "pid_affine";
  case ProgramIdAddressInfo::Pattern::PID_NONLINEAR:
    return "pid_nonlinear";
  }
  llvm_unreachable("Unknown pattern type");
}

struct AnnotatePidAddressPatternsPass
    : public impl::TritonGPUAnnotatePidAddressPatternsBase<
          AnnotatePidAddressPatternsPass> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Run the ProgramIdAddressAnalysis
    ModuleProgramIdToAddressAnalysis pidAnalysis(moduleOp);

    LDBG("Running ProgramIdAddressAnalysis annotation pass");

    // Walk all load/store operations and annotate them
    moduleOp.walk([&](Operation *op) {
      if (!isa<triton::LoadOp, triton::StoreOp>(op))
        return;

      const LoadStoreAddressPattern *pattern =
          pidAnalysis.getAddressPattern(op);
      if (!pattern) {
        LDBG("No pattern found for op at " << op->getLoc());
        return;
      }

      LDBG("Annotating op at " << op->getLoc() << " with pattern "
                               << patternToString(pattern->pattern));

      // Add attribute indicating the address pattern type
      op->setAttr(
          "triton.pid_address_pattern",
          StringAttr::get(&getContext(), patternToString(pattern->pattern)));

      // For affine patterns, add stride information
      if (pattern->pattern == ProgramIdAddressInfo::Pattern::PID_AFFINE) {
        SmallVector<int64_t> axes;
        SmallVector<int64_t> strides;

        for (const auto &[axis, stride] : pattern->pidAxisStrides) {
          axes.push_back(axis);
          strides.push_back(stride);
        }

        if (!axes.empty()) {
          op->setAttr("triton.pid_axes",
                      DenseI64ArrayAttr::get(&getContext(), axes));
          op->setAttr("triton.pid_strides",
                      DenseI64ArrayAttr::get(&getContext(), strides));
        }

        // Add coalescing hint
        op->setAttr("triton.is_coalesced",
                    BoolAttr::get(&getContext(), pattern->isCoalesced));

        // Add block size if known
        if (pattern->blockSize.has_value()) {
          op->setAttr("triton.block_size",
                      IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                       pattern->blockSize.value()));
        }
      }
    });

    // Print statistics in debug mode
    LLVM_DEBUG({
      int numAnnotated = 0;
      int numCoalesced = 0;
      int numConstant = 0;
      int numAffine = 0;
      int numUnknown = 0;

      moduleOp.walk([&](Operation *op) {
        if (!isa<triton::LoadOp, triton::StoreOp>(op))
          return;

        if (auto patternAttr =
                op->getAttrOfType<StringAttr>("triton.pid_address_pattern")) {
          numAnnotated++;
          StringRef pattern = patternAttr.getValue();
          if (pattern == "constant")
            numConstant++;
          else if (pattern == "pid_affine")
            numAffine++;
          else
            numUnknown++;
        }

        if (auto coalescedAttr =
                op->getAttrOfType<BoolAttr>("triton.is_coalesced")) {
          if (coalescedAttr.getValue())
            numCoalesced++;
        }
      });

      DBGS() << "Annotated " << numAnnotated << " load/store ops\n";
      DBGS() << "  Constant patterns: " << numConstant << "\n";
      DBGS() << "  PID affine patterns: " << numAffine << "\n";
      DBGS() << "  Unknown patterns: " << numUnknown << "\n";
      DBGS() << "  Coalesced accesses: " << numCoalesced << "\n";
    });
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
