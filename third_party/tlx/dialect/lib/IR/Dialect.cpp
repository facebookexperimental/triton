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
  resultLayout = wrapNoVerifyLayout(unwrappedResultLayout);
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
  // the full-tensor encoding. Keep any TLX wrapper on the slice's *parent*
  // (matching ReduceOp/ExpandDimsOp's own result inference) rather than
  // wrapping the whole slice, which would produce `no_verify<slice<parent=L>>`
  // where the IR has `slice<parent=no_verify<L>>` and fail the op verifier. So
  // we locate the concrete dialect from the unwrapped encoding but delegate
  // with the *wrapped* operand and do not re-wrap the result.
  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    const triton::DialectInferLayoutInterface *delegate =
        getInferLayoutInterfaceFor(unwrapTlxLayoutWrappers(operandEncoding));
    if (!delegate)
      return failure();
    return delegate->inferReduceOpEncoding(operandEncoding, axis,
                                           resultEncoding, loc);
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> loc) const override {
    const triton::DialectInferLayoutInterface *delegate =
        getInferLayoutInterfaceFor(unwrapTlxLayoutWrappers(operandEncoding));
    if (!delegate)
      return failure();
    return delegate->inferExpandDimsOpEncoding(operandEncoding, axis,
                                               resultEncoding, loc);
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
  return isa_and_nonnull<NoVerifyLayoutAttr>(layout);
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
