#ifndef TRITON_ANALYSIS_REDUCTIONORDER_H
#define TRITON_ANALYSIS_REDUCTIONORDER_H

#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir {
class ModuleOp;
namespace triton {
class ReduceOp;
} // namespace triton
} // namespace mlir

namespace mlir::triton::bitequiv {

// MLIR-native reduction-order analysis for the bitwise-equivalence checker.
//
// These build a canonical, comparable *reduction-order signature* by inspecting
// the parsed TritonGPU (TTGIR) IR with the compiler's own layout machinery
// (`toLinearLayout` + the `tt.reduce` op accessors) — NOT by text/regex. Two
// ops with equal signatures perform the same floating-point association order,
// so the relation is sound (equal signature => same reduction tree => same
// bits).
//
// The same analysis is intended to be reused inside the compiler (e.g. a
// verification pass) to assert the reduction order is preserved across passes
// once the ordering switch (`inner_tree`) is on.

// Canonical signature of one `tt.reduce` op. Encodes: axis, the normalized
// `reduction_ordering` attribute, the combine-region structure, and — for
// `unordered` — the axis-projected `LinearLayout` of the operand (which fixes
// how elements are distributed across register/lane/warp/block along the reduce
// axis, hence the reduction tree). For `inner_tree` the layout is omitted (the
// order is layout-invariant by construction) and only the axis extent is kept.
std::string getReductionOrderSignature(::mlir::triton::ReduceOp op);

// One signature per `tt.reduce` in `module` (in walk/textual order). If the
// module contains a tensor-core/MMA accumulation (`tt.dot` / `*mma*`) — a
// reduction-like op this analysis does not model — a conservative entry is
// appended so two such modules are never falsely declared equivalent. Returns
// an empty vector only when the module has no reduction-like op at all.
llvm::SmallVector<std::string>
getReductionOrderSignatures(::mlir::ModuleOp module);

// Convenience for an in-compiler verification pass: true iff `a` and `b` have
// identical reduction-order signatures.
bool reductionOrdersEquivalent(::mlir::ModuleOp a, ::mlir::ModuleOp b);

} // namespace mlir::triton::bitequiv

#endif // TRITON_ANALYSIS_REDUCTIONORDER_H
