// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_Z3_JOINT_SOLUTION_VALIDATOR_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_Z3_JOINT_SOLUTION_VALIDATOR_H

#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::triton::gpu {

/// Independently validate a joint-solver-0.1 or joint-solver-0.2 result.
/// The joint-solver-validation-0.1 JSON contains `valid`, `message`, and
/// `violations` fields.
std::string validateZ3JointSolution(llvm::StringRef problemJson,
                                    llvm::StringRef solutionJson);

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_Z3_JOINT_SOLUTION_VALIDATOR_H
