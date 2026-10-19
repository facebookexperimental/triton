#ifndef TRITON_ANALYSIS_INTEGERRANGE_H
#define TRITON_ANALYSIS_INTEGERRANGE_H

namespace mlir {
class DialectRegistry;
namespace triton {
// Allow MLIR's standard integer range analysis to follow scalar LLVM arithmetic
// and consume the range attributes on ROCDL IDs after GPU lowering.
void registerIntegerRangeExternalModels(DialectRegistry &registry);
} // namespace triton
} // namespace mlir

#endif
