// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Joint-solver terminal fallback policy.
//
// The joint solver can fail in two places separated by several call frames:
// the per-loop schedule solve (runJointSolverSchedule, reached from
// runModuloScheduling) and the warp-group partition plus the final memory
// audit (partitionJointSolver / applyGlobalWarpPartition). Neither stage can
// see the other, and neither may repair itself locally — a joint MinII
// schedule combined with a heuristic partition is exactly the mixed state the
// policy exists to prevent.
//
// So the failure handling lives in ONE place. `runScheduleDriver` installs a
// JointSolverFallbackScope around the whole attempt, every trigger site
// reports into it, and the driver decides once: rerun the complete baseline
// schedule + partition path (default), or fail the compilation
// (TRITON_MODULO_STRICT_ERROR).
//
// Reporting goes through a scoped thread-local rather than a process-global
// because Triton compiles independent modules concurrently in one process
// (triton.AsyncCompileMode submits compiles to a thread pool and the MLIR
// pass-manager binding releases the GIL). A process-global would let one
// compile's solver failure steer another compile's policy — the same reason
// the scheduling backend is an argument threaded through
// ScheduleDriverOptions rather than process state.
//
// A trigger cannot take that route: it originates several frames below the
// decision point, in functions whose return value the intermediate callers
// already use for something else. A thread-local scope is the narrowest thing
// that carries it up without rewriting those signatures.

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_JOINT_SOLVER_FALLBACK_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_JOINT_SOLVER_FALLBACK_H

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir::triton::gpu {

/// Why the joint solver could not deliver a complete verified result.
///
/// A proven UNSAT at ONE candidate II is deliberately absent: that is normal
/// progress in the solver's own II sweep, not a terminal outcome. Only a
/// global "no II is feasible" answer reaches ScheduleSolve.
enum class JointSolverTrigger {
  /// Schedule stage: backend unavailable, timeout, UNKNOWN, malformed model,
  /// global UNSAT, or a schedule that failed C++ re-verification.
  ScheduleSolve,
  /// Apply stage: the joint warp-group partition (v2 and v1) failed, so the
  /// only way to finish the loop would be a heuristic partition on top of a
  /// joint-solver schedule.
  PartitionSolve,
  /// Apply stage: the post-partition SMEM/TMEM audit rejected the result.
  MemoryAudit,
  /// The joint attempt returned failure without naming a trigger.
  AttemptFailed,
};

llvm::StringRef getJointSolverTriggerName(JointSolverTrigger trigger);

/// Collects terminal-policy triggers for one driver attempt. Exactly one is
/// alive per joint-solver attempt; the flag-off and heuristic paths install
/// none, so they pay nothing and can never take the fallback branch.
class JointSolverFallbackScope {
public:
  JointSolverFallbackScope();
  ~JointSolverFallbackScope();
  JointSolverFallbackScope(const JointSolverFallbackScope &) = delete;
  JointSolverFallbackScope &
  operator=(const JointSolverFallbackScope &) = delete;

  /// The FIRST trigger seen, which is the one that actually caused the
  /// fallback — later ones are downstream noise from an attempt that is
  /// already being discarded.
  std::optional<JointSolverTrigger> getTrigger() const { return firstTrigger; }
  llvm::StringRef getDetail() const { return firstDetail; }
  /// Total triggers seen, for the diagnostic.
  unsigned getCount() const { return count; }

  void record(JointSolverTrigger trigger, llvm::StringRef detail);

private:
  std::optional<JointSolverTrigger> firstTrigger;
  std::string firstDetail;
  unsigned count = 0;
  JointSolverFallbackScope *previous = nullptr;
};

/// Report a terminal-policy trigger to the innermost live scope, so trigger
/// sites need no guard.
///
/// A no-op when no scope is live. That is the common case — the heuristic
/// backends never report — but it also covers the callers of
/// `runModuloScheduling` that sit outside `runScheduleDriver` and so have no
/// policy layer at all: `ModuloWSPartitionPass`, the AMD scaffold in
/// `DotDecomposeAndSchedule.cpp`, and the baseline comparison. Reached only
/// with an explicit TRITON_USE_MODULO_SCHEDULE=joint_solver, they now get the
/// schedule stage's honest failure instead of a silent Rau substitution. That
/// is deliberate — a layer with no policy must not invent one — but it is a
/// behaviour change.
void reportJointSolverFailure(JointSolverTrigger trigger,
                              llvm::StringRef detail = {});

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_JOINT_SOLVER_FALLBACK_H
