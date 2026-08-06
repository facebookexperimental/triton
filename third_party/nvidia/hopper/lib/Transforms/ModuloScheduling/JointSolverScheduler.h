// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Joint-solver modulo scheduling backend — complete solver for joint schedule +
// buffer-depth feasibility, the successor of ExhaustiveScheduler's
// branch-and-bound (docs/SolverMigrationNotes.md, "Suggested sequencing"
// step 2). The model is solved in-process by the native Z3 backend; this side
// serializes the DDG, invokes the backend, parses the schedule back and
// RE-VERIFIES it against the reservation table and dependence constraints, so
// the solver is not part of the correctness TCB.
//
// Selected with TRITON_USE_MODULO_SCHEDULE=joint_solver. Because the search is
// complete, the II sweep runs from minII to a true feasibility bound
// (critical path + total serial work) with NO slack window — guard 2 of
// SolverMigrationNotes.md does not apply to this backend.

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_JOINT_SOLVER_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_JOINT_SOLVER_H

#include "DataDependenceGraph.h"
#include "ModuloReservationTable.h"

namespace mlir::triton::gpu {

/// Run joint-solver modulo scheduling. Returns failure if the native solver is
/// unavailable, errors, or returns a schedule that fails re-verification —
/// callers fall back to the heuristic backends.
FailureOr<ModuloScheduleResult>
runJointSolverSchedule(const DataDependenceGraph &ddg, int minII,
                       int smemBudget = 232448, int tmemColLimit = 512);

/// Run the native joint solver on an arbitrary problem JSON and return the raw
/// solution JSON text. Shared by the schedule backend above and the
/// joint-partition mode (ModuloSchedulePass's partitionJointSolver).
/// Proven results are cached by canonical request JSON in a bounded,
/// thread-safe LRU.
/// Inconclusive responses are retried with a fresh backend and a bounded,
/// increasing timeout.
FailureOr<std::string> runJointSolverBackend(llvm::StringRef problemJson);

/// Minimum II feasible under a relaxation of the joint model that keeps only
/// the resource-exclusivity constraints, i.e. `relaxed_lower_bound`.
///
/// A relaxation's feasible set is a superset of the full model's, so its
/// optimum can never exceed the full model's:
///
///     relaxed_lower_bound <= II_full     (always, by construction)
///
/// That direction is the useful one. A violation is not a performance
/// regression, it is a *soundness* failure: the full model found a schedule
/// the relaxed model calls impossible, which can only happen if a constraint
/// is mis-encoded or a "relaxation" is not actually a relaxation. Reporting
/// `II_full - relaxed_lower_bound` as an improvement would be backwards; the
/// gap is informational only, and a large gap can equally mean the dropped
/// constraints are necessary.
///
/// Named `relaxed_` rather than `resource_` deliberately: not every
/// constraint is assumption-gated, so this is not a pure resource-only bound.
/// The cycle-domain bounds and the canonical-root pin remain hard assertions
/// (the SMEM/TMEM ceilings also survive but go vacuous once their gated
/// contributors are dropped). The bound is therefore tighter than a textbook
/// resource-only relaxation — still a valid lower bound, and a stronger
/// soundness test, but not comparable to a published one.
FailureOr<int> runJointSolverRelaxedLowerBound(const DataDependenceGraph &ddg,
                                               int minII,
                                               int smemBudget = 232448,
                                               int tmemColLimit = 512);

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_JOINT_SOLVER_H
