#ifndef TRITON_DIALECT_TRITON_IR_TMAMULTICAST_H_
#define TRITON_DIALECT_TRITON_IR_TMAMULTICAST_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::triton {

inline constexpr llvm::StringLiteral kMulticastAttrName = "multicast";
inline constexpr llvm::StringLiteral kMulticastAxesAttrName =
    "tt.multicast_axes";

namespace gpu {

inline constexpr llvm::StringLiteral kPreferredCTAsPerCGAAttrName =
    "ttg.preferred-ctas-per-cga";

} // namespace gpu
} // namespace mlir::triton

#endif // TRITON_DIALECT_TRITON_IR_TMAMULTICAST_H_
