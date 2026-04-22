// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_SWING_SCHEDULER_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_SWING_SCHEDULER_H

#include "ModuloReservationTable.h"

namespace mlir::triton::gpu {

/// Swing Modulo Scheduling (SMS).
/// J. Llosa, A. González, E. Ayguadé, M. Valero,
/// "Swing Modulo Scheduling: A Lifetime-Sensitive Approach", PACT 1996.
FailureOr<ModuloScheduleResult> runSMS(const DataDependenceGraph &ddg,
                                       int minII, int maxII);

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_SWING_SCHEDULER_H
