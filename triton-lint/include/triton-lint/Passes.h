//===- Passes.h - Triton-Lint Pass Declarations ---------------*- C++ -*-===//
//
// Triton-Lint - Static analysis tool for Triton kernels
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for Triton-Lint's MLIR-based analysis.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINT_PASSES_H
#define TRITON_LINT_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton_lint {

/// Create a pass that verifies TMA usage for regression detection
std::unique_ptr<Pass> createTMAVerificationPass();

/// Register all Triton-Lint passes
void registerTritonLintPasses();

} // namespace triton_lint
} // namespace mlir

#endif // TRITON_LINT_PASSES_H
