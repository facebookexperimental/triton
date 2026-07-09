// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CP-SAT modulo scheduling backend — complete solver for joint schedule +
// buffer-depth feasibility, the successor of ExhaustiveScheduler's
// branch-and-bound (docs/SolverMigrationNotes.md, "Suggested sequencing"
// step 2). The model is solved by OR-Tools CP-SAT in a Python subprocess
// (python/triton/tools/modulo_cpsat.py — same subprocess pattern as
// LLMSchedulePass); this side serializes the DDG, launches the solver,
// parses the schedule back and RE-VERIFIES it against the reservation
// table and dependence constraints, so the subprocess is not part of the
// correctness TCB.
//
// Selected with TRITON_USE_MODULO_SCHEDULE=cpsat. Because the search is
// complete, the II sweep runs from minII to a true feasibility bound
// (critical path + total serial work) with NO slack window — guard 2 of
// SolverMigrationNotes.md does not apply to this backend.

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_CPSAT_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_CPSAT_H

#include "DataDependenceGraph.h"
#include "ModuloReservationTable.h"

namespace mlir::triton::gpu {

/// Run CP-SAT modulo scheduling. Returns failure if the solver subprocess
/// is unavailable, errors, or returns a schedule that fails re-verification
/// — callers fall back to the heuristic backends.
///
/// `incumbent` (optional) is a schedule from the heuristic path (Rau); it
/// seeds the solver as a warm-start hint so that among model-equivalent
/// optima the heuristic's placement discipline survives — the solver only
/// departs from it for a strictly better objective.
FailureOr<ModuloScheduleResult>
runCPSATSchedule(const DataDependenceGraph &ddg, int minII,
                 const ModuloScheduleResult *incumbent = nullptr,
                 int smemBudget = 232448, int tmemColLimit = 512);

/// Run the modulo_cpsat solver subprocess on an arbitrary problem JSON and
/// return the raw solution JSON text. Shared by the schedule backend above
/// and the joint-partition mode (ModuloSchedulePass's partitionJointCPSAT).
FailureOr<std::string> runCPSATSubprocess(llvm::StringRef problemJson);

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_CPSAT_H
