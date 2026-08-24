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

#include "llvm/ADT/StringRef.h"

#include <functional>
#include <optional>
#include <string>

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

// ── Deterministic fake backend (tests) ──────────────────────────────────────
//
// The fallback policy has to be provable for failure modes a healthy solver
// never produces. These faults replace the transport with a canned response,
// one per terminal-policy trigger, so every trigger is reachable without a
// live Z3 backend and without depending on wall-clock timing. Diff 9's
// apply-stage rejection tests share the seam.

/// One canned backend response. Names match the `force-joint-solver-fault=`
/// pass option so lit tests read the same vocabulary as the C++ tests.
enum class JointSolverFault {
  /// Transport itself fails: backend missing or nonzero exit.
  Unavailable,
  /// Hard timeout — an inconclusive response, retried then given up on.
  Timeout,
  /// Z3 returned UNKNOWN.
  Unknown,
  /// Proven GLOBAL infeasibility. Distinct from a per-II UNSAT, which the
  /// solver's own II sweep steps past and which is never terminal.
  GlobalUnsat,
  /// Truncated, schema-violating response text.
  Malformed,
  /// Schema-valid response carrying an ILLEGAL schedule. Exercises C++
  /// re-verification rather than the transport or the parser.
  IllegalSchedule,
};

/// Which solver request a fault applies to. The two stages are told apart by
/// problem schema: joint-solver-0.1 is the per-loop schedule solve,
/// joint-solver-0.2 is the warp-group partition / joint re-solve. Requests
/// outside the selected stage reach the real backend, so a test can fail the
/// apply stage while the schedule stage genuinely succeeds — the only way to
/// prove the policy catches apply-stage failures at all.
enum class JointSolverFaultStage { All, Schedule, Partition };

struct JointSolverFaultSpec {
  JointSolverFault fault;
  JointSolverFaultStage stage = JointSolverFaultStage::All;
};

std::optional<JointSolverFault> parseJointSolverFault(llvm::StringRef name);
llvm::StringRef getJointSolverFaultName(JointSolverFault fault);

/// Parse a `force-joint-solver-fault=` value: a fault name, optionally
/// prefixed with `schedule:` or `partition:` (default: both stages).
std::optional<JointSolverFaultSpec>
parseJointSolverFaultSpec(llvm::StringRef spec);

/// The canned response for `fault` against `problemJson`.
///
/// `IllegalSchedule` is the one fault that cannot be forged from nothing: the
/// solution schema carries an objective, per-II statistics and buffer depths
/// that must agree with the problem, so a hand-built response would be
/// rejected as malformed long before any semantic check runs (and would need
/// re-forging on every schema change). It instead solves for real and then
/// collapses every cycle to 0 — schema-identical, and dependence-violating
/// for any problem with a positive-latency edge. It therefore needs a live
/// backend; without one it degrades to `Unavailable`.
FailureOr<std::string>
makeJointSolverFaultResponse(JointSolverFault fault,
                             llvm::StringRef problemJson);

/// While an instance is alive, `runJointSolverBackend` routes every request on
/// this thread to `respond` instead of the native backend, and bypasses the
/// result cache in both directions so a fake response can neither be served
/// from nor written to it. Thread-local and save/restore: nesting is
/// well-defined and concurrent compiles never observe each other's override.
class ScopedJointSolverBackendOverride {
public:
  using Responder = std::function<FailureOr<std::string>(llvm::StringRef)>;

  explicit ScopedJointSolverBackendOverride(Responder respond);
  explicit ScopedJointSolverBackendOverride(JointSolverFaultSpec spec);
  explicit ScopedJointSolverBackendOverride(JointSolverFault fault)
      : ScopedJointSolverBackendOverride(JointSolverFaultSpec{fault}) {}
  ~ScopedJointSolverBackendOverride();
  ScopedJointSolverBackendOverride(const ScopedJointSolverBackendOverride &) =
      delete;
  ScopedJointSolverBackendOverride &
  operator=(const ScopedJointSolverBackendOverride &) = delete;

private:
  Responder saved;
};

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
