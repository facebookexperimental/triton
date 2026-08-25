#ifndef TRITON_DIALECT_TRITON_IR_DISCARDABLE_ATTRIBUTES_H_
#define TRITON_DIALECT_TRITON_IR_DISCARDABLE_ATTRIBUTES_H_

#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

inline constexpr StringLiteral kNumStagesAttrName = "tt.num_stages";
inline constexpr StringLiteral kDisallowAccMultiBufferAttrName =
    "tt.disallow_acc_multi_buffer";
inline constexpr StringLiteral kWarpSpecializeAttrName = "tt.warp_specialize";
inline constexpr StringLiteral kScheduledMaxStageAttrName =
    "tt.scheduled_max_stage";
inline constexpr StringLiteral kDataPartitionFactorAttrName =
    "tt.data_partition_factor";

// AutoWS annotation on an MMA op: a JSON object carrying the desired schedule
// and, optionally, the channels the operands travel through, e.g.
//   {"stage": "0", "order": "2", "channels": ["opndD,tmem,1,0"]}
// The names below are the single source of truth for everyone who reads or
// writes that annotation (ScheduleLoops, AssignLatencies, WSMemoryPlanner).
inline constexpr StringLiteral kAutoWSAnnotationAttrName = "tt.autows";
inline constexpr StringLiteral kAutoWSStageKey = "stage";
inline constexpr StringLiteral kAutoWSOrderKey = "order";
inline constexpr StringLiteral kAutoWSChannelsKey = "channels";
inline constexpr StringLiteral kAutoWSOperandDTag = "opndD";

enum class AutoWSLoopAttrPropagation {
  NotForwarded,
  ForwardToInnerLoop,
};

struct AutoWSLoopAttrInfo {
  StringLiteral name;
  AutoWSLoopAttrPropagation propagation;
};

// Returns every loop attribute emitted by AutoWSLoopOptions and whether it must
// be propagated when an annotated scheduler loop is removed.
ArrayRef<AutoWSLoopAttrInfo> getAutoWSLoopAttrs();

[[nodiscard]] SmallVector<NamedAttribute>
filterAutoWSLoopAttrs(Operation *op, AutoWSLoopAttrPropagation propagation);

// Filter out attributes from the given operation that are not present in
// the allowList.
[[nodiscard]] SmallVector<NamedAttribute>
filterDiscardableAttrs(Operation *op, ArrayRef<StringRef> allowList);

} // namespace mlir::triton
#endif // TRITON_DIALECT_TRITON_IR_DISCARDABLE_ATTRIBUTES_H_
