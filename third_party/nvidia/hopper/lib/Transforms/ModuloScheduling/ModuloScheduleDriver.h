// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Shared driver for the modulo-scheduling pass family.
//
// ModuloSchedulePass (heuristic backends, env-selected) and
// JointSolverSchedulePass (CP-SAT joint schedule + warp partition) are thin
// shells over the same orchestration: collect loops → schedule → global
// warp-group partition → emit loop.stage/loop.cluster + buffer annotations.
// ScheduleDriverOptions is the ONLY behavioural difference between them, so
// both passes stay in lockstep on everything downstream consumes.

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_MODULO_SCHEDULE_DRIVER_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_MODULO_SCHEDULE_DRIVER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace mlir::triton::gpu {

/// How the global warp-group partition (Pass B) is decided.
enum class JointSolverMode {
  /// Heuristic partitioners only (exhaustive scorer / greedy). The modulo
  /// pass runs here.
  Off,
  /// CP-SAT joint re-solve: v2 (cycles + warp groups in one model) first,
  /// v1 (warp groups only, cycles fixed) if v2 fails, exhaustive scorer if
  /// both fail. The joint-solver pass default.
  V2ThenV1,
  /// v1 only (then exhaustive scorer). For A/B measurement.
  V1Only,
  /// v2 only (then exhaustive scorer). For A/B measurement.
  V2Only,
};

struct ScheduleDriverOptions {
  JointSolverMode jointMode = JointSolverMode::Off;
  /// When non-empty, forces the per-loop scheduling algorithm for the whole
  /// driver run (held as a ScopedScheduleAlgoOverride), ignoring
  /// TRITON_USE_MODULO_SCHEDULE. The joint-solver pass forces "cpsat".
  std::string forceScheduleAlgo;
  /// Mirror of the passes' print-schedule-graph test option.
  bool printScheduleGraph = false;
  /// Mirror of the passes' data-partition-factor option (Pass A.5).
  int dataPartitionFactor = 0;
};

/// Run the full Pass A orchestration on `moduleOp`. Returns failure on a
/// diagnosed error (already emitted); callers map it to signalPassFailure.
LogicalResult runScheduleDriver(ModuleOp moduleOp,
                                const ScheduleDriverOptions &opts);

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_MODULO_SCHEDULE_DRIVER_H
