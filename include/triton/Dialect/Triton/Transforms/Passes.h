#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createCudaWarningsPass(int32_t computeCapability);

/// Collect CUDA-specific performance warnings for a module.
/// Returns a vector of warning messages that can be used to populate Python
/// warnings. The pass version (createCudaWarningsPass) also emits these as
/// MLIR warnings for lit testing purposes.
std::vector<std::string> collectCudaWarnings(ModuleOp module,
                                             int32_t computeCapability);

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
