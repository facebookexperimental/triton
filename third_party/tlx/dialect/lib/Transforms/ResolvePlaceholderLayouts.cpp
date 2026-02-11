#include "IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "tlx-resolve-placeholder-layouts"
#define DBGS() (llvm::errs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) DBGS() << X << "\n"

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXRESOLVEPLACEHOLDERLAYOUTS

#include "tlx/dialect/include/Transforms/Passes.h.inc"

/// Check if an attribute is any of the dummy layout types
static bool isDummyLayoutAttr(Attribute attr) {
  return isa<DummyRegisterLayoutAttr>(attr);
}

/// Extract the dummy layout attribute from a type, if present
static Attribute getDummyLayoutFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto encoding = tensorType.getEncoding()) {
      if (isDummyLayoutAttr(encoding))
        return encoding;
    }
  }
  return nullptr;
}

/// Compute the resolved layout for a dummy register layout.
/// If tmemCompatible is true, creates a TMEM-compatible register layout using
/// getTmemCompatibleLayout (matches
/// make_default_tmem_compatible_tensor_layout_encoding). Otherwise, creates a
/// default BlockedEncodingAttr.
///
static Attribute resolveRegisterLayout(DummyRegisterLayoutAttr dummyLayout,
                                       Operation *contextOp,
                                       ModuleOp moduleOp) {
  auto shape = dummyLayout.getShape();
  auto elementType = dummyLayout.getElementType();
  auto rank = shape.size();

  // Use contextOp for lookupNumWarps to get partition-aware num_warps
  int numWarps = ttg::lookupNumWarps(contextOp);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(moduleOp);

  if (dummyLayout.getTmemCompatible()) {
    // Create a TMEM-compatible register layout
    // Matches make_default_tmem_compatible_tensor_layout_encoding
    //
    // Use the allocation shape (not the subsliced shape) for the
    // TMEM-compatible layout calculation. The allocation shape determines the
    // TMEM block dimensions.
    assert(rank == 2 &&
           "Only supporting 2D tensors for TMEM compatible layout.");
    // assert(!elementType.isInteger() && "Integer type not supported for
    // TMEM.");
    assert((numWarps == 4 || numWarps == 8) &&
           "Currently only support numWarps 4 or 8 for TMEM load and store.");

    ttg::BlockedEncodingAttr defaultBlockedEncoding =
        ttg::getDefaultBlockedEncoding(moduleOp.getContext(), shape, numWarps,
                                       threadsPerWarp, numCTAs);
    auto oldType =
        RankedTensorType::get(shape, elementType, defaultBlockedEncoding);

    auto result =
        ttng::getTmemCompatibleLayout(shape[0], shape[1], oldType, numWarps);
    return result;
  }

  // Default: create a standard blocked encoding
  // sizePerThread: all 1s (default)
  SmallVector<unsigned> sizePerThread(rank, 1);

  // order: reversed range [rank-1, rank-2, ..., 1, 0]
  SmallVector<unsigned> order(rank);
  std::iota(order.rbegin(), order.rend(), 0);

  return ttg::BlockedEncodingAttr::get(moduleOp.getContext(), shape,
                                       sizePerThread, order, numWarps,
                                       threadsPerWarp, numCTAs);
}

/// Resolve a dummy layout attribute to a concrete layout
/// For TMEM layouts and TMEM-compatible register layouts, allocShape is used
/// to determine the block dimensions.
/// For register layouts from TMEMLoadOp, definingOp is used to get the source
/// memdesc's allocation shape.
static Attribute resolveDummyLayout(Attribute dummyLayout,
                                    ArrayRef<int64_t> allocShape, Value value,
                                    ModuleOp moduleOp) {
  // Get the context operation for lookupNumWarps - this allows finding
  // partition-specific num_warps for warp specialized regions
  Operation *contextOp = nullptr;
  if (auto defOp = value.getDefiningOp()) {
    contextOp = defOp;
  } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    contextOp = blockArg.getOwner()->getParentOp();
  }
  if (!contextOp) {
    contextOp = moduleOp;
  }

  if (auto regLayout = dyn_cast<DummyRegisterLayoutAttr>(dummyLayout))
    return resolveRegisterLayout(regLayout, contextOp, moduleOp);

  llvm_unreachable("Unknown dummy layout type");
}

/// Replace the type of a value with a new encoding
static void replaceTypeWithNewEncoding(Value value, Attribute newEncoding) {
  Type oldType = value.getType();
  Type newType;

  if (auto tensorType = dyn_cast<RankedTensorType>(oldType)) {
    newType = RankedTensorType::get(tensorType.getShape(),
                                    tensorType.getElementType(), newEncoding);
  } else if (auto memDescType = dyn_cast<ttg::MemDescType>(oldType)) {
    // Preserve the allocation shape when replacing the encoding
    newType = ttg::MemDescType::get(
        memDescType.getShape(), memDescType.getElementType(), newEncoding,
        memDescType.getMemorySpace(), memDescType.getMutableMemory(),
        memDescType.getAllocShape());
  } else {
    return;
  }

  value.setType(newType);
}

LogicalResult resolvePlaceholderLayouts(ModuleOp moduleOp) {
  // Collect all values that have dummy layouts
  SmallVector<std::pair<Value, Attribute>> valuesToResolve;

  moduleOp.walk([&](Operation *op) {
    // Check all result types for dummy layouts
    for (Value result : op->getResults()) {
      if (Attribute dummyLayout = getDummyLayoutFromType(result.getType())) {
        valuesToResolve.emplace_back(result, dummyLayout);
      }
    }

    // Check block arguments in all regions (for ops like WarpSpecializeOp)
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (Attribute dummyLayout = getDummyLayoutFromType(arg.getType())) {
            valuesToResolve.emplace_back(arg, dummyLayout);
          }
        }
      }
    }
  });

  // Resolve each dummy layout
  for (auto &[value, dummyLayout] : valuesToResolve) {
    // Get allocation shape for TMEM layouts
    ArrayRef<int64_t> allocShape;
    if (auto memDescType = dyn_cast<ttg::MemDescType>(value.getType())) {
      allocShape = memDescType.getAllocShape();
    }
    Attribute resolvedLayout =
        resolveDummyLayout(dummyLayout, allocShape, value, moduleOp);
    LLVM_DEBUG({
      DBGS() << "Resolving dummy layout: ";
      dummyLayout.dump();
      DBGS() << "  to: ";
      resolvedLayout.dump();
    });
    replaceTypeWithNewEncoding(value, resolvedLayout);
  }

  return success();
}

struct TLXResolvePlaceholderLayoutsPass
    : public impl::TLXResolvePlaceholderLayoutsBase<
          TLXResolvePlaceholderLayoutsPass> {
public:
  using impl::TLXResolvePlaceholderLayoutsBase<
      TLXResolvePlaceholderLayoutsPass>::TLXResolvePlaceholderLayoutsBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (failed(tlx::resolvePlaceholderLayouts(m))) {
      signalPassFailure();
    }
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
