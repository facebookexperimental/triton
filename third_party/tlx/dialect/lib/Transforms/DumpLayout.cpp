//===----------------------------------------------------------------------===//
// TLX Dump Layout Pass
//===----------------------------------------------------------------------===//
//
// Prints the resolved layout of every `tlx.dump_layout` operand to the compiler
// log (stderr) and then erases the op. This is a compile-time-only diagnostic:
// it emits no device code. The pass is scheduled at the end of the TTGIR
// pipeline so that the reported layout reflects all optimizations.
//
// The layout is rendered in CuTe (CUTLASS) `Shape:Stride` notation, with `_N`
// marking static integers:
//
//   - Register tensors -> a thread-value (TV) layout
//       ((thread...),(value...)):((thread...),(value...))
//     where the thread group is built from the hardware input dims
//     (lane, warp, block) and the value group from the per-thread register dim.
//   - Shared/tensor-memory buffers -> a single strided layout, e.g. _64:_1.
//   - Swizzled shared buffers -> a swizzle functor composed over the base
//       layout: Swizzle<B,M,S> o (base):(stride). B/M/S are derived from the
//       swizzled_shared encoding and verified against the linear layout.
//
// If the layout is not representable as a CuTe layout (e.g. an unsupported
// swizzle), the pass falls back to the raw linear-layout string.
//
// The pass also emits a remark for every convert_layout that must go through
// shared memory (shown with MLIR_ENABLE_DIAGNOSTICS=remarks). When a
// shared-memory load feeds the conversion, a note on that load suggests a
// layout that may remove the conversion.
//
//===----------------------------------------------------------------------===//

#include "IR/Dialect.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tlx-dump-layout"

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXDUMPLAYOUT
#include "tlx/dialect/include/Transforms/Passes.h.inc"

namespace {

// A CuTe layout mode: `size : stride` over the flattened element index.
using Mode = std::pair<int64_t, int64_t>;

// Collapse the per-bit strides of one input dimension into CuTe modes. A run of
// consecutive bits whose strides form a geometric progression s, 2s, 4s, ...
// (all-zero runs collapse too, for broadcast) becomes a single mode (2^run :
// s).
static void collapseModes(ArrayRef<int64_t> bitStrides,
                          SmallVectorImpl<Mode> &modes) {
  size_t i = 0, n = bitStrides.size();
  while (i < n) {
    int64_t s = bitStrides[i];
    int64_t size = 2;
    size_t j = i + 1;
    while (j < n && bitStrides[j] == s * size) {
      size *= 2;
      ++j;
    }
    modes.push_back({size, s});
    i = j;
  }
}

// Print one CuTe group: `_N` for a single mode, `(_a,_b,...)` for several, and
// `_1`/`_0` for an empty group (a single element with stride 0).
static void printGroup(llvm::raw_ostream &os, ArrayRef<Mode> modes,
                       bool wantSize) {
  if (modes.empty()) {
    os << (wantSize ? "_1" : "_0");
    return;
  }
  if (modes.size() == 1) {
    os << "_" << (wantSize ? modes[0].first : modes[0].second);
    return;
  }
  os << "(";
  for (size_t i = 0; i < modes.size(); ++i) {
    if (i)
      os << ",";
    os << "_" << (wantSize ? modes[i].first : modes[i].second);
  }
  os << ")";
}

// Row-major element strides for flattening a multi-dim coordinate into a single
// linear element index.
static SmallVector<int64_t> rowMajorStrides(ArrayRef<int64_t> shape) {
  int rank = shape.size();
  SmallVector<int64_t> rm(rank, 1);
  for (int d = rank - 2; d >= 0; --d)
    rm[d] = rm[d + 1] * shape[d + 1];
  return rm;
}

// CuTe modes of a layout: thread/value groups for a register tensor, a single
// flat group for a buffer. `blockModes` is the part of `threadModes` that comes
// from the CTA (block) dim.
struct CuteModes {
  SmallVector<Mode> threadModes, valueModes, flatModes;
  size_t blockModes = 0;
};

// Collect `type`'s layout as CuTe integer-strided modes. Returns false if the
// layout is not integer-strided (e.g. it uses XOR swizzling, which is linear
// over GF(2) but not over the integers).
static bool collectCuteModes(ttg::TensorOrMemDesc type, const LinearLayout &ll,
                             CuteModes &out) {
  MLIRContext *ctx = type.getContext();
  ArrayRef<int64_t> shape = type.getShape();
  int rank = shape.size();
  SmallVector<int64_t> rowMajor = rowMajorStrides(shape);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  auto flatten = [&](ArrayRef<int32_t> basis) -> int64_t {
    int64_t idx = 0;
    for (int d = 0; d < (int)outNames.size() && d < rank; ++d)
      idx += (int64_t)basis[d] * rowMajor[d];
    return idx;
  };

  // Collect one input dim's modes while checking the layout stays integer-
  // strided: every nonzero stride must be a distinct power of two, so that the
  // CuTe integer sum matches the GF(2) linear layout.
  bool faithful = true;
  llvm::SmallDenseSet<int64_t> seenStrides;
  auto modesFor = [&](StringRef name, SmallVectorImpl<Mode> &modes) {
    auto inDim = StringAttr::get(ctx, name);
    if (!ll.hasInDim(inDim))
      return;
    int bits = ll.getInDimSizeLog2(inDim);
    SmallVector<int64_t> bitStrides;
    for (int p = 0; p < bits; ++p) {
      int64_t s = flatten(ll.getBasis(inDim, p));
      bitStrides.push_back(s);
      if (s != 0 && (!llvm::isPowerOf2_64(s) || !seenStrides.insert(s).second))
        faithful = false;
    }
    collapseModes(bitStrides, modes);
  };

  if (isa<RankedTensorType>(type)) {
    modesFor("lane", out.threadModes);
    modesFor("warp", out.threadModes);
    size_t beforeBlock = out.threadModes.size();
    modesFor("block", out.threadModes);
    out.blockModes = out.threadModes.size() - beforeBlock;
    modesFor("register", out.valueModes);
  } else {
    modesFor("offset", out.flatModes);
    modesFor("block", out.flatModes);
  }
  return faithful;
}

// Emit `type`'s layout as a CuTe integer-strided Shape:Stride layout. Returns
// false if the layout is not integer-strided.
static bool emitCuteStrided(ttg::TensorOrMemDesc type, const LinearLayout &ll,
                            llvm::raw_ostream &os) {
  CuteModes m;
  if (!collectCuteModes(type, ll, m))
    return false;
  if (isa<RankedTensorType>(type)) {
    os << "(";
    printGroup(os, m.threadModes, true);
    os << ",";
    printGroup(os, m.valueModes, true);
    os << "):(";
    printGroup(os, m.threadModes, false);
    os << ",";
    printGroup(os, m.valueModes, false);
    os << ")";
  } else {
    printGroup(os, m.flatModes, true);
    os << ":";
    printGroup(os, m.flatModes, false);
  }
  return true;
}

// Spell a single-CTA register layout as a `tlx.layout(shape=..., stride=...)`
// expression, which uses the same (thread, value) Shape:Stride convention.
// Returns false if the layout has no such spelling.
static bool emitTlxLayout(RankedTensorType type, llvm::raw_ostream &os) {
  CuteModes m;
  if (!collectCuteModes(cast<ttg::TensorOrMemDesc>(type),
                        ttg::toLinearLayout(type), m) ||
      m.blockModes != 0)
    return false;
  auto print = [&](ArrayRef<Mode> modes, bool wantSize) {
    os << "(";
    for (size_t i = 0; i < modes.size(); ++i)
      os << (i ? ", " : "") << (wantSize ? modes[i].first : modes[i].second);
    if (modes.size() == 1)
      os << ",";
    os << ")";
  };
  os << "tlx.layout(shape=(";
  print(m.threadModes, true);
  os << ", ";
  print(m.valueModes, true);
  os << "), stride=(";
  print(m.threadModes, false);
  os << ", ";
  print(m.valueModes, false);
  os << "))";
  return true;
}

static std::string cuteOrEncoding(RankedTensorType type) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  if (!emitCuteStrided(cast<ttg::TensorOrMemDesc>(type),
                       ttg::toLinearLayout(type), os))
    os << type.getEncoding();
  return buf;
}

// Follow single-operand elementwise ops (e.g. extf) back to a shared-memory
// load, which could produce its value directly in any register layout.
static ttg::LocalLoadOp traceToLocalLoad(Value v) {
  for (int depth = 0; depth < 8; ++depth) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return nullptr;
    if (auto load = dyn_cast<ttg::LocalLoadOp>(def))
      return load;
    if (!def->hasTrait<OpTrait::Elementwise>() || def->getNumOperands() != 1)
      return nullptr;
    v = def->getOperand(0);
  }
  return nullptr;
}

// Remark on a convert_layout that exchanges data through shared memory.
// Suggest the destination layout for a load feeding the conversion, or the
// source layout for a peer load. Other users and downstream layout constraints
// may still require conversions.
static void remarkSharedMemoryConvert(ttg::ConvertLayoutOp cvt) {
  RankedTensorType srcTy = cvt.getSrc().getType();
  RankedTensorType dstTy = cvt.getType();
  // NPOT conversion diagnostics are not supported.
  if (hasNpotShape(srcTy) || hasNpotShape(dstTy))
    return;
  if (!cvtNeedsSharedMemory(cvt))
    return;
  bool inLoop = cvt->getParentOfType<LoopLikeOpInterface>() != nullptr;
  std::string shape;
  llvm::raw_string_ostream shapeOs(shape);
  for (int64_t dim : dstTy.getShape())
    shapeOs << dim << "x";
  shapeOs << dstTy.getElementType();
  InFlightDiagnostic diag = cvt.emitRemark()
                            << "convert_layout through shared memory"
                            << (inLoop ? " inside a loop" : "") << " for "
                            << shape << ": from " << cuteOrEncoding(srcTy)
                            << " to " << cuteOrEncoding(dstTy);

  auto suggest = [&](ttg::LocalLoadOp load, RankedTensorType wanted) {
    auto loadTy = cast<RankedTensorType>(load.getType());
    if (hasNpotShape(loadTy))
      return;
    Diagnostic &note = diag.attachNote(load.getLoc());
    note << "loading this shared-memory value in layout "
         << cuteOrEncoding(loadTy.cloneWithEncoding(wanted.getEncoding()))
         << " may remove the conversion";
    std::string tlx;
    llvm::raw_string_ostream os(tlx);
    if (emitTlxLayout(loadTy.cloneWithEncoding(wanted.getEncoding()), os))
      note << "; in TLX: tlx.local_load(..., layout=" << tlx << ")";
  };

  if (ttg::LocalLoadOp load = traceToLocalLoad(cvt.getSrc())) {
    suggest(load, dstTy);
    return;
  }
  llvm::SmallSetVector<Operation *, 4> seen;
  for (Operation *user : cvt->getUsers()) {
    if (!user->hasTrait<OpTrait::Elementwise>())
      continue;
    for (Value peer : user->getOperands()) {
      if (peer == cvt.getResult())
        continue;
      ttg::LocalLoadOp load = traceToLocalLoad(peer);
      if (load && seen.insert(load.getOperation()))
        suggest(load, srcTy);
    }
  }
}

// Emit a swizzled shared buffer as `Swizzle<B,M,S> o (base):(stride)`, derived
// from the swizzled_shared encoding and verified against the linear layout.
// Returns false if the encoding is not a supported swizzle or verification
// fails.
static bool emitCuteSwizzle(ttg::TensorOrMemDesc type, const LinearLayout &ll,
                            llvm::raw_ostream &os) {
  auto memdesc = dyn_cast<ttg::MemDescType>(type);
  if (!memdesc)
    return false;
  auto enc = dyn_cast<ttg::SwizzledSharedEncodingAttr>(memdesc.getEncoding());
  if (!enc)
    return false;

  int vec = enc.getVec(), perPhase = enc.getPerPhase(),
      maxPhase = enc.getMaxPhase();
  if (maxPhase <= 1)
    return false; // no swizzle; the strided path handles it
  if (!llvm::isPowerOf2_64(vec) || !llvm::isPowerOf2_64(perPhase) ||
      !llvm::isPowerOf2_64(maxPhase))
    return false;

  ArrayRef<int64_t> shape = type.getShape();
  ArrayRef<unsigned> order = enc.getOrder();
  int rank = shape.size();
  if (rank < 2 || (int)order.size() != rank)
    return false;

  int64_t numContig = shape[order[0]]; // fastest-varying (contiguous) dim
  int B = llvm::Log2_64(maxPhase);
  int M = llvm::Log2_64(vec);
  int S = llvm::Log2_64(numContig) + llvm::Log2_64(perPhase) - M;
  if (S < 0)
    return false;

  // Base (unswizzled) layout: contiguous along order[0], stacking outward.
  SmallVector<int64_t> baseStride(rank, 1);
  int64_t run = 1;
  for (int i = 0; i < rank; ++i) {
    baseStride[order[i]] = run;
    run *= shape[order[i]];
  }

  // Verify: for each physical offset bit, base^-1(swizzle(2^p)) must equal the
  // linear layout's logical coordinate. Swizzle and base are GF(2)-linear, so
  // checking the basis (powers of two) is sufficient.
  auto inDim = StringAttr::get(type.getContext(), "offset");
  if (!ll.hasInDim(inDim))
    return false;
  int bits = ll.getInDimSizeLog2(inDim);
  int64_t srcMask = ((int64_t(1) << B) - 1) << (M + S);
  for (int p = 0; p < bits; ++p) {
    int64_t phys = int64_t(1) << p;
    int64_t swz = phys ^ ((phys & srcMask) >> S);
    // base^-1: decompose swizzled offset into logical coords via `order`.
    SmallVector<int64_t> coord(rank, 0);
    int64_t rem = swz;
    for (int i = 0; i < rank; ++i) {
      coord[order[i]] = rem % shape[order[i]];
      rem /= shape[order[i]];
    }
    ArrayRef<int32_t> basis = ll.getBasis(inDim, p);
    for (int d = 0; d < rank; ++d)
      if ((int64_t)basis[d] != coord[d])
        return false;
  }

  os << "Swizzle<" << B << "," << M << "," << S << "> o (";
  for (int d = 0; d < rank; ++d)
    os << (d ? ",_" : "_") << shape[d];
  os << "):(";
  for (int d = 0; d < rank; ++d)
    os << (d ? ",_" : "_") << baseStride[d];
  os << ")";
  return true;
}

} // namespace

struct TLXDumpLayoutPass : public impl::TLXDumpLayoutBase<TLXDumpLayoutPass> {
public:
  using impl::TLXDumpLayoutBase<TLXDumpLayoutPass>::TLXDumpLayoutBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    SmallVector<Operation *> assertsToErase;
    bool assertionFailed = false;
    auto compare = [&](Operation *op, ttg::TensorOrMemDesc lhsType,
                       ttg::TensorOrMemDesc rhsType) {
      auto lhsEncoding = cast<ttg::LayoutEncodingTrait>(lhsType.getEncoding());
      auto rhsEncoding = cast<ttg::LayoutEncodingTrait>(rhsType.getEncoding());
      if (!ttg::areLayoutsEquivalent(lhsType.getShape(), lhsEncoding,
                                     rhsEncoding)) {
        LinearLayout lhs = ttg::toLinearLayout(lhsType);
        LinearLayout rhs = ttg::toLinearLayout(rhsType);
        op->emitError() << "layout assertion failed: LinearLayouts differ\n"
                        << "lhs: " << lhs.toString() << "\n"
                        << "rhs: " << rhs.toString();
        assertionFailed = true;
      }
      assertsToErase.push_back(op);
    };
    m.walk([&](AssertSameLayoutOp op) {
      compare(op, cast<ttg::TensorOrMemDesc>(op.getLhs().getType()),
              cast<ttg::TensorOrMemDesc>(op.getRhs().getType()));
    });
    m.walk([&](AssertSameLayoutExpectedOp op) {
      auto actualType = cast<ttg::TensorOrMemDesc>(op.getSrc().getType());
      Type expectedType;
      if (auto tensorType = dyn_cast<RankedTensorType>(actualType)) {
        expectedType = RankedTensorType::get(tensorType.getShape(),
                                             tensorType.getElementType(),
                                             op.getExpected());
      } else {
        auto memDescType = cast<ttg::MemDescType>(actualType);
        expectedType = ttg::MemDescType::get(
            memDescType.getShape(), memDescType.getElementType(),
            op.getExpected(), memDescType.getMemorySpace(),
            memDescType.getMutableMemory(), memDescType.getAllocShape());
      }
      compare(op, actualType, cast<ttg::TensorOrMemDesc>(expectedType));
    });
    for (Operation *op : assertsToErase)
      op->erase();
    if (assertionFailed) {
      signalPassFailure();
      return;
    }

    SmallVector<DumpLayoutOp> toErase;
    m.walk([&](DumpLayoutOp op) {
      Value src = op.getSrc();
      auto type = cast<ttg::TensorOrMemDesc>(src.getType());

      auto &os = llvm::errs();
      os << "// tlx.dump_layout @ " << op.getLoc() << "\n";
      os << "//   type: " << type << "\n";
      if (!type.getEncoding()) {
        os << "//   encoding: <none>\n";
        toErase.push_back(op);
        return;
      }

      LinearLayout ll = ttg::toLinearLayout(type);
      std::string buf;
      llvm::raw_string_ostream ss(buf);
      if (emitCuteStrided(type, ll, ss) || emitCuteSwizzle(type, ll, ss))
        os << "//   cute: " << buf << "\n";
      else
        os << "//   (not representable as a CuTe layout; linear layout "
              "follows)\n"
           << ll.toString() << "\n";

      toErase.push_back(op);
    });

    for (DumpLayoutOp op : toErase)
      op->erase();

    std::string diagnostics =
        triton::tools::getStrEnv("MLIR_ENABLE_DIAGNOSTICS");
    SmallVector<StringRef> enabledDiagnostics;
    StringRef(diagnostics).split(enabledDiagnostics, ',');
    if (llvm::is_contained(enabledDiagnostics, StringRef("remarks")))
      m.walk([](ttg::ConvertLayoutOp cvt) { remarkSharedMemoryConvert(cvt); });
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
