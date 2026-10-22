#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::AMD {
using ElemLocationKey = SmallVector<std::pair<StringAttr, int32_t>>;

// Returns true when a block argument or operation result has two mutually
// exclusive uses under its defining block's two-way cf.cond_br. Each arm must
// reach its consumer block through an acyclic branch prefix. A use may instead
// forward on a header edge if that successor cannot reach the other consumer
// without the header; the other consumer may reach a shared forwarding target.
// No consumer block may execute again before the defining block rebinds the
// argument or recomputes the result.
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
