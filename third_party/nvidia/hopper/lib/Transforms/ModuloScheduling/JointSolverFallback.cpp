// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// See JointSolverFallback.h for the policy contract.

#include "JointSolverFallback.h"

namespace mlir::triton::gpu {

llvm::StringRef getJointSolverTriggerName(JointSolverTrigger trigger) {
  switch (trigger) {
  case JointSolverTrigger::ScheduleSolve:
    return "schedule-solve";
  case JointSolverTrigger::PartitionSolve:
    return "partition-solve";
  case JointSolverTrigger::MemoryAudit:
    return "memory-audit";
  case JointSolverTrigger::AttemptFailed:
    return "attempt-failed";
  }
  return "unknown";
}

static JointSolverFallbackScope *&activeFallbackScope() {
  static thread_local JointSolverFallbackScope *scope = nullptr;
  return scope;
}

JointSolverFallbackScope::JointSolverFallbackScope() {
  previous = activeFallbackScope();
  activeFallbackScope() = this;
}

JointSolverFallbackScope::~JointSolverFallbackScope() {
  activeFallbackScope() = previous;
}

void JointSolverFallbackScope::record(JointSolverTrigger trigger,
                                      llvm::StringRef detail) {
  ++count;
  if (firstTrigger)
    return;
  firstTrigger = trigger;
  firstDetail = detail.str();
}

void reportJointSolverFailure(JointSolverTrigger trigger,
                              llvm::StringRef detail) {
  if (auto *scope = activeFallbackScope())
    scope->record(trigger, detail);
}

} // namespace mlir::triton::gpu
