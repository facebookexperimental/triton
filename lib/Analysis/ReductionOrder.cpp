#include "triton/Analysis/ReductionOrder.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace {

// Structural key of the combine region: the ordered op-name sequence of a
// pre-order walk. Distinguishes add vs mul vs cmp+select (argmax/argmin) etc.,
// and works uniformly for single- and multi-operand reduces.
std::string combineKey(triton::ReduceOp op) {
  std::string s;
  llvm::raw_string_ostream os(s);
  op.getCombineOp().walk(
      [&](Operation *o) { os << o->getName().getStringRef() << ";"; });
  return os.str();
}

// Axis-projected LinearLayout of the operand, serialized. Two reduces with
// equal keys distribute elements across (register, lane, warp, block) along the
// reduce axis identically -> identical reduction tree (given the same
// ordering+combine). The sublayout keeps out-dim sizes, so a different axis
// extent (e.g. a different BLOCK_SIZE) yields a different key (shape-soundness,
// for free).
std::string axisLayoutKey(triton::ReduceOp op) {
  auto ty = dyn_cast<RankedTensorType>(op.getOperands()[0].getType());
  if (!ty || !ty.getEncoding()) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "noenc:";
    if (ty)
      ty.print(os);
    return os.str();
  }
  MLIRContext *ctx = ty.getContext();
  LinearLayout ll = gpu::toLinearLayout(ty);
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");
  auto kAxis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
  LinearLayout axisLL = ll.sublayout({kReg, kLane, kWarp, kBlock}, {kAxis});
  return axisLL.toString();
}

// A conservative, structural string for a tensor-core/MMA accumulation op:
// op name + sorted attributes + operand/result types (no SSA value names). It
// differs whenever the numerics could differ (precision attribute, operand
// layouts) yet is identical for identical IR.
std::string mmaOpKey(Operation *o) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << o->getName().getStringRef();
  for (NamedAttribute a : o->getAttrs()) {
    os << "|" << a.getName().str() << "=";
    a.getValue().print(os);
  }
  for (Type t : o->getOperandTypes()) {
    os << "|in:";
    t.print(os);
  }
  for (Type t : o->getResultTypes()) {
    os << "|out:";
    t.print(os);
  }
  return os.str();
}

bool isMmaLikeName(StringRef n) {
  return n.contains("dot") || n.contains("mma");
}

} // namespace

namespace mlir::triton::bitequiv {

std::string getReductionOrderSignature(triton::ReduceOp op) {
  unsigned axis = op.getAxis();
  StringAttr ordAttr = op.getReductionOrderingAttr();
  StringRef ord = (ordAttr && !ordAttr.getValue().empty())
                      ? ordAttr.getValue()
                      : StringRef("unordered");
  bool innerTree = ord == "inner_tree";

  std::string s;
  llvm::raw_string_ostream os(s);
  os << "reduce|axis=" << axis << "|ord=" << ord
     << "|nops=" << op.getNumOperands() << "|combine=" << combineKey(op)
     << "|layout=";
  if (innerTree) {
    // `inner_tree` enforces one canonical order over the original element
    // indices, independent of layout; key only on the axis extent (a different
    // number of leaves is a different canonical tree).
    int64_t sAxis = -1;
    if (auto ty = dyn_cast<RankedTensorType>(op.getOperands()[0].getType()))
      if (axis < ty.getRank())
        sAxis = ty.getShape()[axis];
    os << "inner_tree-invariant|sAxis=" << sAxis;
  } else {
    os << axisLayoutKey(op);
  }
  return os.str();
}

llvm::SmallVector<std::string> getReductionOrderSignatures(ModuleOp module) {
  llvm::SmallVector<std::string> sigs;
  module.walk([&](triton::ReduceOp op) {
    sigs.push_back(getReductionOrderSignature(op));
  });

  // Soundness guard: an MMA/dot accumulation is reduction-like but not modeled
  // here. Append a conservative entry so two modules that contain such ops are
  // never declared equivalent on an empty signature (they only match when the
  // dot ops' structure/types/attrs are identical).
  std::vector<std::string> mma;
  module.walk([&](Operation *o) {
    if (isMmaLikeName(o->getName().getStringRef()))
      mma.push_back(mmaOpKey(o));
  });
  if (!mma.empty()) {
    std::sort(mma.begin(), mma.end());
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "unanalyzed-mma";
    for (const auto &e : mma)
      os << "||" << e;
    sigs.push_back(os.str());
  }
  return sigs;
}

bool reductionOrdersEquivalent(ModuleOp a, ModuleOp b) {
  return getReductionOrderSignatures(a) == getReductionOrderSignatures(b);
}

} // namespace mlir::triton::bitequiv
