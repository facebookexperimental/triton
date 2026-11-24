#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCESECTORUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCESECTORUTILS_H_

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>
#include <optional>

namespace mlir::triton {
class ModuleAxisInfoAnalysis;

namespace gpu {
class BlockedEncodingAttr;

std::optional<int64_t> inferLaneStep(Value value, unsigned dim,
                                     ModuleAxisInfoAnalysis &axisInfo,
                                     unsigned &compileTimeBudget);

// Counts the distinct sectors touched by a rank-2 scalar wave at a
// sector-aligned origin.
unsigned countScalarWaveSectors(BlockedEncodingAttr layout,
                                ArrayRef<uint64_t> strides,
                                unsigned elementBytes, unsigned sectorBytes);

} // namespace gpu
} // namespace mlir::triton

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCESECTORUTILS_H_
