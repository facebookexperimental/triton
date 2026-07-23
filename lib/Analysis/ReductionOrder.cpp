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

// Bit-deciding key for a dot / wgmma accumulation. It splits on exactly the
// numeric-mode attributes that change the result bits -- the input precision
// (tf32 / ieee / tf32x3) and the fp8 partial-accumulator flush period
// (maxNumImpreciseAcc) -- plus the op name (a tt.dot ieee-FMA path vs an
// ttng.warp_group_dot wgmma path is itself a numeric change). It deliberately
// DROPS the operand/result types: their shapes are the free M/N/K tiling and their
// encodings are the free warpsPerCTA, while the element dtype is fixed per kernel,
// so keeping them would only over-split. A genuine K-decomposition (tf32x3 = 3
// passes) is recovered from the dot COUNT by the caller, not from the types.
std::string dotAccumKey(Operation *o) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << o->getName().getStringRef();
  for (NamedAttribute a : o->getAttrs()) {
    StringRef n = a.getName().getValue();
    if (n == "inputPrecision" || n == "maxNumImpreciseAcc") {
      os << "|" << n.str() << "=";
      a.getValue().print(os);
    }
  }
  return os.str();
}

bool isMmaLikeName(StringRef n) {
  // Match the accumulation ops (tt.dot, ttng.warp_group_dot, tcgen05_mma, ...) but NOT
  // their async completion barriers (ttng.warp_group_dot_wait / *_mma_wait): a wait is
  // pure synchronization with no numeric effect, and it only appears once the loop is
  // pipelined (num_stages > 1), so counting it would spuriously split num_stages (free).
  return (n.contains("dot") || n.contains("mma")) && !n.contains("wait");
}

// True if `o` is nested inside an `scf.for` (the K-loop). A dot in the loop body is
// the steady-state accumulation; the pipeliner peels copies OUT of the loop as
// prologue/epilogue (pure scheduling driven by num_stages, same bits), so counting
// only loop-body dots keeps num_stages free while still seeing an in-loop
// K-decomposition like tf32x3.
bool inForLoop(Operation *o) {
  for (Operation *p = o->getParentOp(); p; p = p->getParentOp())
    if (p->getName().getStringRef() == "scf.for")
      return true;
  return false;
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

  // A dot / wgmma accumulation sums over K in a structured `scf.for` with the
  // accumulator as an iter_arg -- the order is sequential and fixed by
  // construction, so (unlike PTX) no back-edge / reassociation modelling is needed.
  // Emit one bit-deciding key per accumulation and let the COUNT carry the
  // K-decomposition multiplicity (tf32x3 = 3 in-loop dots vs tf32 = 1). Count only
  // loop-body dots -- peeled pipeline copies live outside the loop -- so num_stages
  // stays free; if a GEMM has no K-loop at all, fall back to every dot so none is
  // dropped (soundness over tightness).
  std::vector<Operation *> loopDots, allDots;
  module.walk([&](Operation *o) {
    if (!isMmaLikeName(o->getName().getStringRef()))
      return;
    allDots.push_back(o);
    if (inForLoop(o))
      loopDots.push_back(o);
  });
  const std::vector<Operation *> &dotOps = loopDots.empty() ? allDots : loopDots;
  if (!dotOps.empty()) {
    std::vector<std::string> keys;
    for (Operation *o : dotOps)
      keys.push_back(dotAccumKey(o));
    std::sort(keys.begin(), keys.end()); // order-independent multiset; count matters
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "dot";
    for (const auto &e : keys)
      os << "||" << e;
    sigs.push_back(os.str());
  }
  return sigs;
}

bool reductionOrdersEquivalent(ModuleOp a, ModuleOp b) {
  return getReductionOrderSignatures(a) == getReductionOrderSignatures(b);
}

} // namespace mlir::triton::bitequiv
