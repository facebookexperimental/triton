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
  return isa<DummyTMemLayoutAttr, DummySMemLayoutAttr, DummyRegisterLayoutAttr,
             DummyMMALayoutAttr, DummyDotOperandLayoutAttr>(attr);
}

/// Extract the dummy layout attribute from a type, if present
static Attribute getDummyLayoutFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto encoding = tensorType.getEncoding()) {
      if (isDummyLayoutAttr(encoding))
        return encoding;
    }
  }
  if (auto memDescType = dyn_cast<ttg::MemDescType>(type)) {
    if (auto encoding = memDescType.getEncoding()) {
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
/// For TMEM-compatible layouts, we need the allocation shape (not the subsliced
/// shape) to compute the correct layout. This is passed via allocShape
/// parameter which should be obtained from the TMEMLoadOp's source memdesc.
static Attribute resolveRegisterLayout(DummyRegisterLayoutAttr dummyLayout,
                                       ArrayRef<int64_t> allocShape,
                                       ModuleOp moduleOp) {
  auto shape = dummyLayout.getShape();
  auto elementType = dummyLayout.getElementType();
  auto rank = shape.size();

  int numWarps = ttg::lookupNumWarps(moduleOp);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(moduleOp);

  if (dummyLayout.getTmemCompatible()) {
    // Create a TMEM-compatible register layout
    // Matches make_default_tmem_compatible_tensor_layout_encoding
    //
    // Use the allocation shape (not the subsliced shape) for the
    // TMEM-compatible layout calculation. The allocation shape determines the
    // TMEM block dimensions.
    ArrayRef<int64_t> layoutShape = allocShape.empty() ? shape : allocShape;
    LDBG("resolveRegisterLayout: Taking TMEM-compatible path for shape ["
         << shape[0] << ", " << shape[1] << "], using allocShape ["
         << layoutShape[0] << ", " << layoutShape[1] << "]");
    assert(rank == 2 &&
           "Only supporting 2D tensors for TMEM compatible layout.");
    assert(!elementType.isInteger() && "Integer type not supported for TMEM.");
    assert((numWarps == 4 || numWarps == 8) &&
           "Currently only support numWarps 4 or 8 for TMEM load and store.");

    ttg::BlockedEncodingAttr defaultBlockedEncoding =
        ttg::getDefaultBlockedEncoding(moduleOp.getContext(), layoutShape,
                                       numWarps, threadsPerWarp, numCTAs);
    auto oldType =
        RankedTensorType::get(layoutShape, elementType, defaultBlockedEncoding);

    auto result = ttng::getTmemCompatibleLayout(layoutShape[0], layoutShape[1],
                                                oldType, numWarps);
    LDBG("resolveRegisterLayout: TMEM-compatible layout result: " << result);
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

/// Compute the resolved layout for a dummy SMEM layout.
/// Matches the Python defaults:
/// - For 1D: swizzled_shared_layout_encoding.make_default(rank=1)
///   vectorSize=1, perPhase=1, maxPhase=1, order=[0], CTAs all [1]
/// - For 2D+: nv_mma_shared_layout_encoding.make_default(shape, dtype)
///   order=reversed(range(rank)), CTAs all [1]*rank, fp4Padded=False,
///   swizzled=True
static Attribute resolveSMemLayout(DummySMemLayoutAttr dummyLayout,
                                   ModuleOp moduleOp) {
  auto shape = dummyLayout.getShape();
  auto elementType = dummyLayout.getElementType();
  auto rank = shape.size();

  // Build order as reversed range: [rank-1, rank-2, ..., 1, 0]
  SmallVector<unsigned> order(rank);
  std::iota(order.rbegin(), order.rend(), 0);

  // Build CTA layout with all 1s
  SmallVector<unsigned> ctasPerCGA(rank, 1);
  SmallVector<unsigned> ctaSplit(rank, 1);
  SmallVector<unsigned> ctaOrder(rank);
  std::iota(ctaOrder.rbegin(), ctaOrder.rend(), 0);

  auto ctaLayout = ttg::CTALayoutAttr::get(moduleOp.getContext(), ctasPerCGA,
                                           ctaSplit, ctaOrder);

  if (rank == 1) {
    // swizzled_shared_layout_encoding.make_default(rank=1)
    // vectorSize=1, perPhase=1, maxPhase=1
    return ttg::SwizzledSharedEncodingAttr::get(moduleOp.getContext(),
                                                /*vec=*/1,
                                                /*perPhase=*/1,
                                                /*maxPhase=*/1, order,
                                                ctaLayout);
  } else {
    // nv_mma_shared_layout_encoding.make_default(shape, dtype)
    // fp4Padded=False, swizzled=True
    return ttg::NVMMASharedEncodingAttr::get(moduleOp.getContext(), shape,
                                             order, ctaLayout, elementType,
                                             /*fp4Padded=*/false);
  }
}

/// Compute the resolved layout for a dummy TMEM layout.
/// This creates a TensorMemoryEncodingAttr for TMEM memory descriptors.
/// Matches the Python default:
/// tensor_memory_layout_encoding.make_default(shape) blockM=shape[0],
/// blockN=shape[1], CTASplitM=1, CTASplitN=1
/// TODO: Remove `unpacked` parameter and infer it in this pass based on usage
/// context.
static Attribute resolveTMemLayout(DummyTMemLayoutAttr dummyLayout,
                                   ArrayRef<int64_t> allocShape,
                                   ModuleOp moduleOp) {
  bool unpacked = dummyLayout.getUnpacked();

  // Use the actual allocation shape (from MemDescType) rather than the
  // shape stored in the dummy layout, because the tensor might be a subslice
  // of a larger allocation
  int64_t blockM =
      allocShape.size() > 0 ? allocShape[allocShape.size() - 2] : 1;
  int64_t blockN =
      allocShape.size() > 1 ? allocShape[allocShape.size() - 1] : 1;

  return ttng::TensorMemoryEncodingAttr::get(moduleOp.getContext(), blockM,
                                             blockN, unpacked,
                                             /*CTASplitM=*/1,
                                             /*CTASplitN=*/1);
}

/// Compute the instruction shape for Hopper MMA (version 3).
/// This is a simplified version - returns reasonable defaults.
static SmallVector<unsigned, 3> computeInstrShape(ArrayRef<int64_t> shape,
                                                  Type operandAElemType) {
  // Default Hopper wgmma instruction shape
  unsigned instrM = 16;
  unsigned instrN = 8;
  unsigned instrK = 16;

  // Adjust based on element type bit width
  unsigned bitWidth = operandAElemType.getIntOrFloatBitWidth();
  if (bitWidth == 16) {
    // FP16/BF16: 16x8x16
    instrM = 16;
    instrN = 8;
    instrK = 16;
  } else if (bitWidth == 32) {
    // FP32/TF32: 16x8x8
    instrM = 16;
    instrN = 8;
    instrK = 8;
  } else if (bitWidth == 8) {
    // FP8/INT8: 16x8x32
    instrM = 16;
    instrN = 8;
    instrK = 32;
  }

  return {instrM, instrN, instrK};
}

/// Compute the resolved layout for a dummy MMA layout.
/// Creates NvidiaMmaEncodingAttr for Hopper (version 3).
static Attribute resolveMMALayout(DummyMMALayoutAttr dummyLayout,
                                  ModuleOp moduleOp) {
  auto shape = dummyLayout.getShape();
  auto operandAElemType = dummyLayout.getOperandAElementType();

  int numWarps = ttg::lookupNumWarps(moduleOp);

  // Compute instruction shape based on operand A element type
  auto instrShape = computeInstrShape(shape, operandAElemType);

  // Default to row partitioning
  SmallVector<unsigned, 2> warpsPerCTA = {static_cast<unsigned>(numWarps), 1};
  SmallVector<unsigned, 2> CTAsPerCGA = {1, 1};
  SmallVector<unsigned, 2> CTASplitNum = {1, 1};
  SmallVector<unsigned, 2> CTAOrder = {1, 0};

  auto CTALayout = ttg::CTALayoutAttr::get(moduleOp.getContext(), CTAsPerCGA,
                                           CTASplitNum, CTAOrder);

  // Version 3 for Hopper, version minor 0
  return ttg::NvidiaMmaEncodingAttr::get(moduleOp.getContext(),
                                         /*versionMajor=*/3,
                                         /*versionMinor=*/0, warpsPerCTA,
                                         CTALayout, instrShape);
}

/// Compute the resolved layout for a dummy dot operand layout.
/// Creates DotOperandEncodingAttr with a default parent MMA layout.
static Attribute resolveDotOperandLayout(DummyDotOperandLayoutAttr dummyLayout,
                                         ModuleOp moduleOp) {
  auto shape = dummyLayout.getShape();
  auto elementType = dummyLayout.getElementType();
  unsigned opIdx = dummyLayout.getOpIdx();

  int numWarps = ttg::lookupNumWarps(moduleOp);

  // Create a default parent MMA layout for Hopper
  auto instrShape = computeInstrShape(shape, elementType);
  SmallVector<unsigned, 2> warpsPerCTA = {static_cast<unsigned>(numWarps), 1};
  SmallVector<unsigned, 2> CTAsPerCGA = {1, 1};
  SmallVector<unsigned, 2> CTASplitNum = {1, 1};
  SmallVector<unsigned, 2> CTAOrder = {1, 0};

  auto CTALayout = ttg::CTALayoutAttr::get(moduleOp.getContext(), CTAsPerCGA,
                                           CTASplitNum, CTAOrder);

  auto parentLayout = ttg::NvidiaMmaEncodingAttr::get(
      moduleOp.getContext(), /*versionMajor=*/3, /*versionMinor=*/0,
      warpsPerCTA, CTALayout, instrShape);

  return ttg::DotOperandEncodingAttr::get(moduleOp.getContext(), opIdx,
                                          parentLayout, elementType);
}

/// Resolve a dummy layout attribute to a concrete layout
/// For TMEM layouts and TMEM-compatible register layouts, allocShape is used
/// to determine the block dimensions.
/// For register layouts from TMEMLoadOp, definingOp is used to get the source
/// memdesc's allocation shape.
static Attribute resolveDummyLayout(Attribute dummyLayout,
                                    ArrayRef<int64_t> allocShape, Value value,
                                    ModuleOp moduleOp) {
  if (auto regLayout = dyn_cast<DummyRegisterLayoutAttr>(dummyLayout)) {
    // For TMEM-compatible register layouts, we need the allocation shape from
    // the source memdesc. If the value is from a TMEMLoadOp, get the allocShape
    // from its source.
    ArrayRef<int64_t> regAllocShape;
    if (regLayout.getTmemCompatible() && value) {
      if (auto defOp = value.getDefiningOp()) {
        if (auto tmemLoad = dyn_cast<ttng::TMEMLoadOp>(defOp)) {
          auto srcType = tmemLoad.getSrc().getType();
          regAllocShape = srcType.getAllocShape();
        } else if (auto releaseOp = dyn_cast<ReleaseLayoutOp>(defOp)) {
          // ReleaseLayoutOp wraps the actual load result, look through it
          if (auto srcDefOp = releaseOp.getSrc().getDefiningOp()) {
            if (auto tmemLoad = dyn_cast<ttng::TMEMLoadOp>(srcDefOp)) {
              auto srcType = tmemLoad.getSrc().getType();
              regAllocShape = srcType.getAllocShape();
            }
          }
        }
      }
    }
    return resolveRegisterLayout(regLayout, regAllocShape, moduleOp);
  }
  if (auto smemLayout = dyn_cast<DummySMemLayoutAttr>(dummyLayout))
    return resolveSMemLayout(smemLayout, moduleOp);
  if (auto tmemLayout = dyn_cast<DummyTMemLayoutAttr>(dummyLayout))
    return resolveTMemLayout(tmemLayout, allocShape, moduleOp);
  if (auto mmaLayout = dyn_cast<DummyMMALayoutAttr>(dummyLayout))
    return resolveMMALayout(mmaLayout, moduleOp);
  if (auto dotOpLayout = dyn_cast<DummyDotOperandLayoutAttr>(dummyLayout))
    return resolveDotOperandLayout(dotOpLayout, moduleOp);

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
  LDBG("resolvePlaceholderLayouts");

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

  LDBG("Found " << valuesToResolve.size() << " values with dummy layouts");

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
