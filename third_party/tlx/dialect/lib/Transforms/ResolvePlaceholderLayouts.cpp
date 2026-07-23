#include "IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AttrTypeSubElements.h"
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
#define GEN_PASS_DEF_TLXFINALIZEUSERLAYOUTS

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

/// Extract the no-verify layout attribute from a type, if present.
static NoVerifyLayoutAttr getNoVerifyLayoutFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return dyn_cast_or_null<NoVerifyLayoutAttr>(tensorType.getEncoding());
  }
  if (auto memDescType = dyn_cast<ttg::MemDescType>(type)) {
    return dyn_cast_or_null<NoVerifyLayoutAttr>(memDescType.getEncoding());
  }
  return nullptr;
}

/// Extract a user-layout wrapper on a *shared* (MemDescType) value, if present.
/// Register (RankedTensorType) user layouts are intentionally left wrapped
/// here: they must survive as anchors through remove-layout-conversions and are
/// unwrapped later by tlx-finalize-user-layouts.
static UserLayoutAttr getUserLayoutFromType(Type type) {
  if (auto memDescType = dyn_cast<ttg::MemDescType>(type)) {
    return dyn_cast_or_null<UserLayoutAttr>(memDescType.getEncoding());
  }
  return nullptr;
}

static bool containsNoVerifyLayout(Type type);

static bool containsNoVerifyLayout(Attribute attr) {
  if (!attr)
    return false;
  if (isa_and_nonnull<NoVerifyLayoutAttr>(attr))
    return true;

  bool found = false;
  attr.walkImmediateSubElements(
      [&](Attribute nestedAttr) {
        found |= containsNoVerifyLayout(nestedAttr);
      },
      [&](Type nestedType) { found |= containsNoVerifyLayout(nestedType); });
  return found;
}

static bool containsNoVerifyLayout(Type type) {
  if (!type)
    return false;
  if (getNoVerifyLayoutFromType(type))
    return true;

  bool found = false;
  type.walkImmediateSubElements(
      [&](Attribute attr) { found |= containsNoVerifyLayout(attr); },
      [&](Type nestedType) { found |= containsNoVerifyLayout(nestedType); });
  return found;
}

/// Compute the resolved layout for a dummy register layout.
/// If tmemCompatible is true, creates a TMEM-compatible register layout using
/// getTmemCompatibleLayout. Otherwise, creates a default
/// BlockedEncodingAttr.
///
static Attribute resolveRegisterLayout(DummyRegisterLayoutAttr dummyLayout,
                                       Operation *contextOp, ModuleOp moduleOp,
                                       Attribute tmemEncoding) {
  auto shape = dummyLayout.getShape();
  auto elementType = dummyLayout.getElementType();
  auto rank = shape.size();

  // Use contextOp for lookupNumWarps to get partition-aware num_warps
  int numWarps = ttg::lookupNumWarps(contextOp);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(moduleOp);

  if (dummyLayout.getTmemCompatible()) {
    // Create a TMEM-compatible register layout
    assert(tmemEncoding != nullptr &&
           "missing TMEM layout when finding compatible reg layouts");
    assert(rank == 2 &&
           "Only supporting 2D tensors for TMEM compatible layout.");
    assert((numWarps == 4 || numWarps == 8) &&
           "Currently only support numWarps 4 or 8 for TMEM load and store.");

    auto *ctx = moduleOp.getContext();
    auto memSpace = ttng::TensorMemorySpaceAttr::get(ctx);
    auto memDescType = ttg::MemDescType::get(shape, elementType, tmemEncoding,
                                             memSpace, /*mutableMemory=*/true);

    // Create a temporary RankedTensorType with a blocked encoding for
    // getTmemCompatibleLayout to use as a reference type.
    SmallVector<unsigned> spt(rank, 1);
    SmallVector<unsigned> order = {1, 0};
    auto blockedEnc = ttg::BlockedEncodingAttr::get(
        ctx, shape, spt, order, numWarps, threadsPerWarp, numCTAs);
    auto tmpType = RankedTensorType::get(shape, elementType, blockedEnc);
    auto compatibleLayouts =
        ttng::getTmemCompatibleLayouts(contextOp, tmpType, memDescType);
    assert(!compatibleLayouts.empty() && "No TMEM-compatible layout found");
    return compatibleLayouts.front();
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

/// Resolve a dummy layout attribute to a concrete layout.
static Attribute
resolveDummyLayout(Attribute dummyLayout, Value value, ModuleOp moduleOp,
                   const DenseMap<Value, Attribute> &valuesToRefTMEMLayoutMap) {
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

  auto tmemLayout = valuesToRefTMEMLayoutMap.lookup_or(value, nullptr);

  if (auto regLayout = dyn_cast<DummyRegisterLayoutAttr>(dummyLayout))
    return resolveRegisterLayout(regLayout, contextOp, moduleOp, tmemLayout);

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

static void
collectValuesWithEncodings(ModuleOp moduleOp,
                           function_ref<Attribute(Type)> getEncoding,
                           DenseMap<Value, Attribute> &valuesToResolve) {
  // Check module-like op results and nested op results.
  moduleOp.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      if (Attribute layout = getEncoding(result.getType()))
        valuesToResolve.try_emplace(result, layout);
    }

    // Check block arguments in all regions (for ops like WarpSpecializeOp).
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (Attribute layout = getEncoding(arg.getType()))
            valuesToResolve.try_emplace(arg, layout);
        }
      }
    }
  });
}

static LogicalResult unwrapNoVerifyLayouts(ModuleOp moduleOp) {
  // Strip #tlx.no_verify_layout everywhere it appears -- including nested
  // inside other encodings (e.g. slice<parent = ...>) and inside a user-layout
  // wrapper
  // (#tlx.user_layout<#tlx.no_verify_layout<L>> -> #tlx.user_layout<L>). The
  // no-verify marker only defers tensor verification through inlining; once
  // inlining is done it is no longer needed. A user-layout marker is preserved
  // (it is retired later by tlx-finalize-user-layouts).
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([](NoVerifyLayoutAttr wrapper) -> Attribute {
    return wrapper.getLayout();
  });

  moduleOp->walk([&](Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/false,
                               /*replaceTypes=*/true);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          arg.setType(replacer.replace(arg.getType()));
  });

  bool residual = false;
  moduleOp.walk([&](Operation *op) {
    for (Type type : op->getResultTypes())
      if (containsNoVerifyLayout(type)) {
        residual = true;
        op->emitError("unresolved TLX no-verify layout after placeholder "
                      "layout resolution");
      }
  });
  return failure(residual);
}

/// Unwrap every user-pinned layout (#tlx.user_layout<...>) back
/// to the concrete shared layout the user requested, and verify none remain.
static LogicalResult unwrapUserLayouts(ModuleOp moduleOp) {
  DenseMap<Value, Attribute> valuesToUnwrap;
  collectValuesWithEncodings(
      moduleOp,
      [](Type type) -> Attribute { return getUserLayoutFromType(type); },
      valuesToUnwrap);

  for (auto &[value, layout] : valuesToUnwrap) {
    auto userLayout = cast<UserLayoutAttr>(layout);
    replaceTypeWithNewEncoding(value, userLayout.getLayout());
  }

  bool foundResidual = false;
  moduleOp.walk([&](Operation *op) {
    for (Type type : op->getResultTypes()) {
      if (getUserLayoutFromType(type)) {
        foundResidual = true;
        op->emitError("unresolved TLX user shared layout after placeholder "
                      "layout resolution");
      }
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (getUserLayoutFromType(arg.getType())) {
            foundResidual = true;
            op->emitError("unresolved TLX user shared layout on block argument "
                          "after placeholder layout resolution");
          }
  });

  return failure(foundResidual);
}

LogicalResult resolvePlaceholderLayouts(ModuleOp moduleOp) {
  if (failed(unwrapNoVerifyLayouts(moduleOp)))
    return failure();

  if (failed(unwrapUserLayouts(moduleOp)))
    return failure();

  // Collect all values that have dummy layouts
  DenseMap<Value, Attribute> valuesToResolve;
  DenseMap<Value, Attribute> valuesToRefTMEMLayoutMap;
  collectValuesWithEncodings(moduleOp, getDummyLayoutFromType, valuesToResolve);

  moduleOp.walk([&](Operation *op) {
    if (auto tmemLdOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      auto v = tmemLdOp.getResult();
      if (valuesToResolve.contains(v)) {
        valuesToRefTMEMLayoutMap.try_emplace(
            v, tmemLdOp.getSrc().getType().getEncoding());
      }
    } else if (auto tmemStOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      auto v = tmemStOp.getSrc();
      if (valuesToResolve.contains(v)) {
        valuesToRefTMEMLayoutMap.try_emplace(
            v, tmemStOp.getDst().getType().getEncoding());
      }
    }
  });

  // Resolve each dummy layout
  for (auto &[value, dummyLayout] : valuesToResolve) {
    Attribute resolvedLayout = resolveDummyLayout(dummyLayout, value, moduleOp,
                                                  valuesToRefTMEMLayoutMap);
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

/// Recursively true if a type/attr embeds a #tlx.user_layout anywhere.
static bool containsUserLayout(Type type);
static bool containsUserLayout(Attribute attr) {
  if (!attr)
    return false;
  if (isa<UserLayoutAttr>(attr))
    return true;
  bool found = false;
  attr.walkImmediateSubElements(
      [&](Attribute a) { found |= containsUserLayout(a); },
      [&](Type t) { found |= containsUserLayout(t); });
  return found;
}
static bool containsUserLayout(Type type) {
  if (!type)
    return false;
  bool found = false;
  type.walkImmediateSubElements(
      [&](Attribute a) { found |= containsUserLayout(a); },
      [&](Type t) { found |= containsUserLayout(t); });
  return found;
}

/// Unwrap user-pinned register layouts (#tlx.user_layout on a RankedTensorType)
/// back to the concrete wrapped layout, and verify none remain. These are kept
/// wrapped through the layout-optimization passes (they anchor via
/// PinnedEncodingTrait) and retired here.
///
/// Uses an AttrTypeReplacer so the wrapper is stripped *everywhere* it appears,
/// including nested inside other encodings (e.g. a reduce's
/// `slice<parent = #tlx.user_layout<...>>`) and inside a constant's `value`
/// DenseElementsAttr type. Stripping only top-level result encodings would
/// leave nested wrappers behind and produce inconsistent op types.
// Recursively strip a chain of TLX wrapper attrs (user_layout / no_verify) down
// to the concrete layout. AttrTypeReplacer applies each replacement only once
// per matched attr, so a *nested* wrapper -- e.g.
// no_verify<user_layout<no_verify<L>>>, produced when a require_layout store pin
// also carries a tlx.assert_same_layout -- would keep an inner user_layout after
// a single getLayout(), which the residual check below then flags as
// "unresolved TLX user layout". Unwrapping the whole chain here resolves it
// (wrappers nested inside ttg encodings are still reached by the replacer's own
// sub-element recursion, which re-invokes this on the inner wrapper).
static Attribute deepUnwrapTlxWrappers(Attribute attr) {
  if (auto u = dyn_cast<UserLayoutAttr>(attr))
    return deepUnwrapTlxWrappers(u.getLayout());
  if (auto n = dyn_cast<NoVerifyLayoutAttr>(attr))
    return deepUnwrapTlxWrappers(n.getLayout());
  return attr;
}

static LogicalResult finalizeUserLayouts(ModuleOp moduleOp) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([](UserLayoutAttr wrapper) -> Attribute {
    return deepUnwrapTlxWrappers(wrapper);
  });
  // Defensively also strip any no-verify wrapper that reached this point (e.g.
  // nested under a user-layout that resolve-placeholder-layouts left in place).
  replacer.addReplacement([](NoVerifyLayoutAttr wrapper) -> Attribute {
    return deepUnwrapTlxWrappers(wrapper);
  });

  moduleOp->walk([&](Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/false,
                               /*replaceTypes=*/true);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          arg.setType(replacer.replace(arg.getType()));
  });

  // A constant's `value` DenseElementsAttr carries its own (shaped) type, which
  // AttrTypeReplacer does not rebuild; realign it with the unwrapped result
  // type.
  moduleOp->walk([&](arith::ConstantOp cst) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (!dense)
      return;
    auto resultType = dyn_cast<ShapedType>(cst.getType());
    if (resultType && dense.getType() != resultType)
      cst.setValueAttr(dense.reshape(resultType));
  });

  bool residual = false;
  moduleOp.walk([&](Operation *op) {
    for (Type type : op->getResultTypes())
      if (containsUserLayout(type)) {
        residual = true;
        op->emitError("unresolved TLX user layout after finalization");
      }
  });
  return failure(residual);
}

struct TLXFinalizeUserLayoutsPass
    : public impl::TLXFinalizeUserLayoutsBase<TLXFinalizeUserLayoutsPass> {
public:
  using impl::TLXFinalizeUserLayoutsBase<
      TLXFinalizeUserLayoutsPass>::TLXFinalizeUserLayoutsBase;

  void runOnOperation() override {
    if (failed(finalizeUserLayouts(getOperation())))
      signalPassFailure();
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
