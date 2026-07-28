// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Exhaustive modulo scheduler — joint schedule + memory optimization.
//
// For small GPU inner loops (≤20 ops, ≤5 MMA ops), enumerates all valid
// MMA orderings on the TC pipeline, places remaining ops via constraint
// propagation, checks SMEM/TMEM budget feasibility for each candidate,
// and picks the schedule with minimum II and maximum buffering depth.

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_EXHAUSTIVE_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_EXHAUSTIVE_H

#include "DataDependenceGraph.h"
#include "ModuloReservationTable.h"

namespace mlir::triton::gpu {

/// One SMEM/TMEM buffer extracted from the DDG for joint schedule+memory
/// feasibility. Shared by the exhaustive and joint-solver backends.
struct SchedBufferInfo {
  unsigned allocNodeIdx;
  bool isTmem;
  int64_t sizeBytes; // SMEM buffers
  int64_t tmemCols;  // TMEM accumulators
  llvm::SmallVector<unsigned, 4> consumerNodes;
};

/// Extract SMEM (ttg.local_alloc) and TMEM (accumulator) buffers with their
/// distance-0 consumers from the DDG.
llvm::SmallVector<SchedBufferInfo>
extractSchedBuffers(const DataDependenceGraph &ddg);

/// Run exhaustive modulo scheduling with joint memory feasibility checking.
/// smemBudget and tmemColLimit are hardware constraints (bytes / columns).
FailureOr<ModuloScheduleResult>
runExhaustiveSearch(const DataDependenceGraph &ddg, int maxII = 0,
                    int smemBudget = 232448, int tmemColLimit = 512);

/// Run random sampling modulo scheduling. Randomly assigns stages to key ops
/// (MEM + TC), greedily places the rest, evaluates feasibility + score.
/// numSamples controls how many random candidates to try per II.
FailureOr<ModuloScheduleResult> runRandomSearch(const DataDependenceGraph &ddg,
                                                int maxII = 0,
                                                int smemBudget = 232448,
                                                int tmemColLimit = 512,
                                                int numSamples = 1000);

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_EXHAUSTIVE_H
