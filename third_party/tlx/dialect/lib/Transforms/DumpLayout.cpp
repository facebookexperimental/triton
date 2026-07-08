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
//===----------------------------------------------------------------------===//

#include "IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseSet.h"
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

// Emit `type`'s layout as a CuTe integer-strided Shape:Stride layout. Returns
// false if the layout is not integer-strided (e.g. it uses XOR swizzling, which
// is linear over GF(2) but not over the integers).
static bool emitCuteStrided(ttg::TensorOrMemDesc type, const LinearLayout &ll,
                            llvm::raw_ostream &os) {
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

  bool isTensor = isa<RankedTensorType>(type);
  SmallVector<Mode> threadModes, valueModes, flatModes;
  if (isTensor) {
    for (StringRef d : {"lane", "warp", "block"})
      modesFor(d, threadModes);
    modesFor("register", valueModes);
  } else {
    for (StringRef d : {"offset", "block"})
      modesFor(d, flatModes);
  }

  if (!faithful)
    return false;

  if (isTensor) {
    os << "(";
    printGroup(os, threadModes, true);
    os << ",";
    printGroup(os, valueModes, true);
    os << "):(";
    printGroup(os, threadModes, false);
    os << ",";
    printGroup(os, valueModes, false);
    os << ")";
  } else {
    printGroup(os, flatModes, true);
    os << ":";
    printGroup(os, flatModes, false);
  }
  return true;
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
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
