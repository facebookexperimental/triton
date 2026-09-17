#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::AMD {
using ElemLocationKey = SmallVector<std::pair<StringAttr, int32_t>>;

// Returns true for the narrow runtime-loop CFG form where a block argument is
// used once in each arm of a two-way cf.cond_br and the arms cannot reach one
// another without returning through the defining block.
bool hasMutuallyExclusiveSuccessorUses(Value value);

// Build element coordinates for a given register ID.
// All other hardware dimensions (lane, warp, block) are set to 0.
ElemLocationKey getElemCoordinatesFromRegisters(LinearLayout ll, unsigned regId,
                                                MLIRContext *ctx);

// Extract register ID from element coordinates.
// Returns std::nullopt if non-register dimensions are non-zero.
std::optional<int> getRegFromCoordinates(LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_
