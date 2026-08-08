// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_Z3_JOINT_SOLVER_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_Z3_JOINT_SOLVER_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::triton::gpu {

/// Solve a joint-solver-0.1 scheduling problem in process. Returns failure
/// when Z3 support is disabled or the problem schema is unsupported.
FailureOr<std::string> runZ3JointSolver(llvm::StringRef problemJson);

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_Z3_JOINT_SOLVER_H
