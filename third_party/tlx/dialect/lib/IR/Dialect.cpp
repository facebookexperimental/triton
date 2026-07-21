#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

// clang-format off
#include "IR/Dialect.h"
#include "IR/Dialect.cpp.inc"
#include "IR/TLXTypesEnums.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::tlx;
namespace ttg = mlir::triton::gpu;

namespace {
ModuleOp getModuleOp(Operation *op) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<ModuleOp>();
  }
  return module;
}

const triton::DialectInferLayoutInterface *
getInferLayoutInterfaceFor(Attribute layout) {
  if (!layout)
    return nullptr;
  return dyn_cast<triton::DialectInferLayoutInterface>(&layout.getDialect());
}

// Strip all TLX layout wrappers (no-verify and user-pinned, in any nesting such
// as #tlx.user_layout<#tlx.no_verify_layout<...>>) to reach the concrete
// encoding, whose dialect provides the real layout-inference interface. Without
// this a wrapped layout (a TLX-dialect attribute) would resolve its delegate
// back to this same TLX interface and recurse forever (stack overflow).
static Attribute unwrapTlxLayoutWrappers(Attribute layout) {
  Attribute prev;
  do {
    prev = layout;
    layout = unwrapUserLayout(unwrapNoVerifyLayout(layout));
  } while (layout != prev);
  return layout;
}

LogicalResult delegateInferredLayout(
    Attribute srcLayout, Attribute &resultLayout,
    function_ref<LogicalResult(const triton::DialectInferLayoutInterface *,
                               Attribute, Attribute &)>
        infer) {
  Attribute unwrappedSrcLayout = unwrapTlxLayoutWrappers(srcLayout);
  const triton::DialectInferLayoutInterface *delegate =
      getInferLayoutInterfaceFor(unwrappedSrcLayout);
  if (!delegate)
    return failure();

  Attribute unwrappedResultLayout;
  if (failed(infer(delegate, unwrappedSrcLayout, unwrappedResultLayout)))
    return failure();
  // Re-apply only the TLX wrappers the source actually had (user_layout inside,
  // no_verify outside), so re-inference after resolve-placeholder-layouts --
  // which strips no_verify from operands -- stays consistent with the op's
  // stored result type instead of unconditionally re-adding no_verify.
  Attribute afterNoVerify = unwrapNoVerifyLayout(srcLayout);
  bool hadUserLayout = unwrapUserLayout(afterNoVerify) != afterNoVerify;
  // no_verify may be the top wrapper (TMEM pin: no_verify<user_layout<L>>) or
  // nested under user_layout (SMEM pin: user_layout<no_verify<L>>); check both
  // so the deferred placeholder is not silently downgraded to a plain anchor.
  bool hadNoVerify = hasNoVerifyLayout(srcLayout) ||
                     hasNoVerifyLayout(unwrapUserLayout(srcLayout));
  if (hadUserLayout)
    unwrappedResultLayout = wrapUserLayout(unwrappedResultLayout);
  if (hadNoVerify)
    unwrappedResultLayout = wrapNoVerifyLayout(unwrappedResultLayout);
  resultLayout = unwrappedResultLayout;
  return success();
}

struct TLXInferLayoutInterface : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const override {
    return delegateInferredLayout(
        operandEncoding, resultEncoding,
        [&](const triton::DialectInferLayoutInterface *delegate, Attribute enc,
            Attribute &result) {
          return delegate->inferTransOpEncoding(enc, shape, order, result, loc);
        });
  }

  // reduce/expand_dims produce or consume a `slice` encoding whose parent is
  // the full-tensor encoding. The TLX wrapper must stay on the slice's *parent*
  // (standard inference: slice<parent=user_layout<L>>) so the result is
  // structurally stable across resolve-placeholder-layouts (which strips
  // no_verify) and matches the op's own inference on both sides.
  //
  // When the operand carries the deferred no_verify pin, additionally keep
  // no_verify as the *outermost* wrapper of the result (peel only the outer
  // no_verify, run the standard inference on the user_layout<L> anchor, then
  // re-wrap): a build-time reduce_op.verify() runs before ttg.num-warps is set,
  // and a top-level no_verify makes it skip verifyTensorLayout. After resolve
  // strips no_verify from both operand and result, re-inference falls into the
  // plain branch and produces the same slice<parent=...> -- so they stay
  // consistent.
  template <typename InferFn>
  LogicalResult inferSliceLike(Attribute operandEncoding,
                               Attribute &resultEncoding, InferFn infer) const {
    // Strip #tlx.no_verify_layout at *every* nesting level (top, under a
    // user_layout wrapper, or inside a ttg encoding's parent), keeping
    // user_layout / concrete encodings intact. This yields the "anchor" the
    // standard inference runs on (so user_layout stays on the slice's parent),
    // and tells us whether the operand carried the deferred pin at all.
    mlir::AttrTypeReplacer stripper;
    stripper.addReplacement(
        [](NoVerifyLayoutAttr w) -> Attribute { return w.getLayout(); });
    Attribute anchor = stripper.replace(operandEncoding);
    bool deferred = anchor != operandEncoding;
    const triton::DialectInferLayoutInterface *delegate =
        getInferLayoutInterfaceFor(unwrapTlxLayoutWrappers(anchor));
    if (!delegate)
      return failure();
    Attribute result;
    if (failed(infer(delegate, anchor, result)))
      return failure();
    // Re-wrap no_verify as the outermost wrapper so a build-time verify (before
    // ttg.num-warps is set) skips verifyTensorLayout. After resolve strips
    // no_verify from operand and result alike, re-inference takes the plain
    // branch and produces the same slice<parent=...>, so they stay consistent.
    resultEncoding = deferred ? wrapNoVerifyLayout(result) : result;
    return success();
  }

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    return inferSliceLike(
        operandEncoding, resultEncoding,
        [&](const triton::DialectInferLayoutInterface *delegate, Attribute enc,
            Attribute &result) {
          return delegate->inferReduceOpEncoding(enc, axis, result, loc);
        });
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> loc) const override {
    return inferSliceLike(
        operandEncoding, resultEncoding,
        [&](const triton::DialectInferLayoutInterface *delegate, Attribute enc,
            Attribute &result) {
          return delegate->inferExpandDimsOpEncoding(enc, axis, result, loc);
        });
  }

  LogicalResult inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                                   Attribute retEncoding,
                                   std::optional<Location> loc) const override {
    Attribute unwrappedRetEncoding = unwrapTlxLayoutWrappers(retEncoding);
    const triton::DialectInferLayoutInterface *delegate =
        getInferLayoutInterfaceFor(unwrappedRetEncoding);
    if (!delegate)
      return failure();
    return delegate->inferDotOpEncoding(
        unwrapTlxLayoutWrappers(operandEncoding), opIdx, unwrappedRetEncoding,
        loc);
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         bool allowReorder,
                         std::optional<Location> loc) const override {
    return delegateInferredLayout(
        srcEnc, dstEnc,
        [&](const triton::DialectInferLayoutInterface *delegate, Attribute enc,
            Attribute &result) {
          return delegate->inferReshapeOpEncoding(srcShape, enc, dstShape,
                                                  result, allowReorder, loc);
        });
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    if (hasNoVerifyLayout(expected) || hasNoVerifyLayout(got))
      return success();

    Attribute unwrappedExpected = unwrapTlxLayoutWrappers(expected);
    Attribute unwrappedGot = unwrapTlxLayoutWrappers(got);
    const triton::DialectInferLayoutInterface *delegate =
        getInferLayoutInterfaceFor(unwrappedExpected);
    if (!delegate)
      return failure();
    return delegate->verifyLayoutsAreEqual(shape, unwrappedExpected,
                                           unwrappedGot, loc);
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    return delegateInferredLayout(
        srcEnc, dstEnc,
        [&](const triton::DialectInferLayoutInterface *delegate, Attribute enc,
            Attribute &result) {
          return delegate->inferDefaultJoinOpEncoding(enc, result, shape, loc);
        });
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    return delegateInferredLayout(
        srcEnc, dstEnc,
        [&](const triton::DialectInferLayoutInterface *delegate, Attribute enc,
            Attribute &result) {
          return delegate->inferSplitOpEncoding(enc, result, shape, loc);
        });
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    return success();
  }

  LogicalResult verifyCatOpEncodingCompatibility(Operation *op) const override {
    return success();
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    return delegateInferredLayout(
        inEnc, outEnc,
        [&](const triton::DialectInferLayoutInterface *delegate, Attribute enc,
            Attribute &result) {
          return delegate->inferFp4ToFpOpEncoding(shape, axis, enc, result,
                                                  fwdInference, loc);
        });
  }
};
} // namespace

void mlir::triton::tlx::TLXDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "IR/TLXAttrDefs.cpp.inc"
      >();

  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonInlinerInterface, TLXInferLayoutInterface>();
}

#define GET_ATTRDEF_CLASSES
#include "IR/TLXAttrDefs.cpp.inc"

LogicalResult mlir::triton::tlx::NoVerifyLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute layout) {
  if (isa<NoVerifyLayoutAttr>(layout)) {
    return emitError() << "nested no-verify layouts are not supported";
  }
  if (!isa<ttg::DistributedEncodingTrait>(layout)) {
    return emitError()
           << "no-verify layout must wrap a distributed tensor layout";
  }
  return success();
}

SmallVector<unsigned>
mlir::triton::tlx::NoVerifyLayoutAttr::getRepOrder() const {
  return cast<ttg::DistributedEncodingTrait>(getLayout()).getRepOrder();
}

::mlir::triton::LinearLayout
mlir::triton::tlx::NoVerifyLayoutAttr::toLinearLayout(
    ArrayRef<int64_t> shape) const {
  return cast<ttg::DistributedEncodingTrait>(getLayout()).toLinearLayout(shape);
}

ttg::CGAEncodingAttr
mlir::triton::tlx::NoVerifyLayoutAttr::getCGALayout() const {
  return cast<ttg::LayoutEncodingTrait>(getLayout()).getCGALayout();
}

Attribute mlir::triton::tlx::wrapNoVerifyLayout(Attribute layout) {
  if (!layout || isa<NoVerifyLayoutAttr>(layout))
    return layout;
  if (!isa<ttg::DistributedEncodingTrait>(layout))
    return layout;
  // AMD MFMA and dot-operand encodings are concrete hardware ownerships, not
  // unresolved Triton blocked layouts.  Gluon's CDNA MFMA path emits these
  // attributes bare, and the early Triton-to-TritonGPU conversion must see the
  // same concrete types before TLX's later placeholder cleanup runs.  Wrapping
  // them in #tlx.no_verify_layout makes a live tt.dot result look like an
  // unresolved blocked-to-MFMA materialization and fails conversion on gfx950.
  if (isa<ttg::AMDMfmaEncodingAttr>(layout))
    return layout;
  if (auto dotLayout = dyn_cast<ttg::DotOperandEncodingAttr>(layout))
    if (isa<ttg::AMDMfmaEncodingAttr>(dotLayout.getParent()))
      return layout;
  if (auto mmaLayout = dyn_cast<ttg::NvidiaMmaEncodingAttr>(layout);
      mmaLayout && mmaLayout.isHopper())
    return layout;
  if (auto dotLayout = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
    if (auto parentLayout =
            dyn_cast<ttg::NvidiaMmaEncodingAttr>(dotLayout.getParent());
        parentLayout && parentLayout.isHopper())
      return layout;
  }
  return NoVerifyLayoutAttr::get(layout.getContext(), layout);
}

Attribute mlir::triton::tlx::unwrapNoVerifyLayout(Attribute layout) {
  if (auto noVerify = dyn_cast_or_null<NoVerifyLayoutAttr>(layout))
    return noVerify.getLayout();
  return layout;
}

bool mlir::triton::tlx::hasNoVerifyLayout(Attribute layout) {
  if (!layout)
    return false;
  if (isa<NoVerifyLayoutAttr>(layout))
    return true;
  bool found = false;
  layout.walkImmediateSubElements(
      [&](Attribute sub) { found |= hasNoVerifyLayout(sub); }, [](Type) {});
  return found;
}

//-- UserLayoutAttr --

LogicalResult mlir::triton::tlx::UserLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute layout) {
  if (isa<UserLayoutAttr>(layout))
    return emitError() << "nested user layouts are not supported";
  if (!isa<ttg::DistributedEncodingTrait>(layout) &&
      !isa<ttg::SharedEncodingTrait>(layout))
    return emitError()
           << "user layout must wrap a distributed or shared layout";
  return success();
}

SmallVector<unsigned> mlir::triton::tlx::UserLayoutAttr::getRepOrder() const {
  if (auto distributed = dyn_cast<ttg::DistributedEncodingTrait>(getLayout()))
    return distributed.getRepOrder();
  // Shared inner: getRepOrder is a distributed-only concept; fall back to a
  // natural (row-major) order over the rank so any generic caller stays
  // well-formed.
  unsigned rank = getCGALayout().getRank();
  SmallVector<unsigned> order(rank);
  std::iota(order.rbegin(), order.rend(), 0);
  return order;
}

::mlir::triton::LinearLayout mlir::triton::tlx::UserLayoutAttr::toLinearLayout(
    ArrayRef<int64_t> shape) const {
  // Dispatch on the concrete inner layout (works for both distributed and
  // shared encodings) rather than re-entering through this wrapper.
  return ttg::toLinearLayout(shape, getLayout());
}

int32_t mlir::triton::tlx::UserLayoutAttr::getAlignment() const {
  if (auto shared = dyn_cast<ttg::SharedEncodingTrait>(getLayout()))
    return shared.getAlignment();
  return 16;
}

ttg::CGAEncodingAttr mlir::triton::tlx::UserLayoutAttr::getCGALayout() const {
  return cast<ttg::LayoutEncodingTrait>(getLayout()).getCGALayout();
}

Attribute mlir::triton::tlx::wrapUserLayout(Attribute layout) {
  if (!layout || isa<UserLayoutAttr>(layout))
    return layout;
  if (!isa<ttg::DistributedEncodingTrait>(layout) &&
      !isa<ttg::SharedEncodingTrait>(layout))
    return layout;
  return UserLayoutAttr::get(layout.getContext(), layout);
}

Attribute mlir::triton::tlx::unwrapUserLayout(Attribute layout) {
  if (auto userLayout = dyn_cast_or_null<UserLayoutAttr>(layout))
    return userLayout.getLayout();
  return layout;
}

bool mlir::triton::tlx::tlxEnablePairedMMA(Operation *op) {
  assert(op != nullptr && "expecting nonnull op for checking TLX 2cta mode");
  auto module = getModuleOp(op);
  assert(module != nullptr &&
         "expecting op nested in a module for checking TLX 2cta mode");
  auto attr = module->getAttrOfType<BoolAttr>(AttrTLXEnablePairedCTAMMAName);
  return attr != nullptr && attr.getValue();
}

bool mlir::triton::tlx::hasClusterSyncKernelInit(Operation *op) {
  assert(op != nullptr &&
         "expecting nonnull op for checking cluster sync kernel init");
  auto module = getModuleOp(op);
  assert(module != nullptr && "expecting op nested in a module for checking "
                              "cluster sync kernel init marker");
  auto attr = module->getAttrOfType<BoolAttr>(AttrClusterSyncKernelInitName);
  return attr != nullptr && attr.getValue();
}

void mlir::triton::tlx::setClusterSyncKernelInitOnMod(Operation *op,
                                                      bool value) {
  assert(op != nullptr &&
         "expecting nonnull op for setting cluster sync kernel init");
  auto module = getModuleOp(op);
  assert(module != nullptr && "expecting op nested in a module for setting "
                              "cluster sync kernel init marker");
  module->setAttr(AttrClusterSyncKernelInitName,
                  BoolAttr::get(module->getContext(), value));
}

bool mlir::triton::tlx::hasClusterSyncKernelCleanup(Operation *op) {
  assert(op != nullptr &&
         "expecting nonnull op for checking cluster sync kernel cleanup");
  auto module = getModuleOp(op);
  assert(module != nullptr && "expecting op nested in a module for checking "
                              "cluster sync kernel cleanup marker");
  auto attr = module->getAttrOfType<BoolAttr>(AttrClusterSyncKernelCleanupName);
  return attr != nullptr && attr.getValue();
}

void mlir::triton::tlx::setClusterSyncKernelCleanupOnMod(Operation *op,
                                                         bool value) {
  assert(op != nullptr &&
         "expecting nonnull op for setting cluster sync kernel cleanup");
  auto module = getModuleOp(op);
  assert(module != nullptr && "expecting op nested in a module for setting "
                              "cluster sync kernel cleanup marker");
  module->setAttr(AttrClusterSyncKernelCleanupName,
                  BoolAttr::get(module->getContext(), value));
}

bool mlir::triton::tlx::hasUserPostWsSync(Operation *op) {
  assert(op != nullptr &&
         "expecting nonnull op for checking user post-WS sync");
  auto module = getModuleOp(op);
  assert(module != nullptr && "expecting op nested in a module for checking "
                              "user post-WS sync marker");
  auto attr = module->getAttrOfType<BoolAttr>(AttrUserPostWsSyncName);
  return attr != nullptr && attr.getValue();
}

void mlir::triton::tlx::setUserPostWsSyncOnMod(Operation *op, bool value) {
  assert(op != nullptr && "expecting nonnull op for setting user post-WS sync");
  auto module = getModuleOp(op);
  assert(module != nullptr && "expecting op nested in a module for setting "
                              "user post-WS sync marker");
  module->setAttr(AttrUserPostWsSyncName,
                  BoolAttr::get(module->getContext(), value));
}

bool mlir::triton::tlx::tlxIsClustered(Operation *op) {
  assert(op != nullptr && "expecting nonnull op for checking cluster dims");
  auto moduleOp = getModuleOp(op);
  assert(moduleOp != nullptr &&
         "expecting op nested in a module for checking cluster dims");
  const SmallVector<int> clusterDims =
      triton::gpu::TritonGPUDialect::getClusterDims(moduleOp);
  int clusterSize = 1;
  for (int d : clusterDims)
    clusterSize *= d;
  return clusterSize > 1;
}
