#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "llvm/ADT/TypeSwitch.h"

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

LogicalResult delegateInferredLayout(
    Attribute srcLayout, Attribute &resultLayout,
    function_ref<LogicalResult(const triton::DialectInferLayoutInterface *,
                               Attribute, Attribute &)>
        infer) {
  Attribute unwrappedSrcLayout = unwrapNoVerifyLayout(srcLayout);
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

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    return delegateInferredLayout(
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
    return delegateInferredLayout(
        operandEncoding, resultEncoding,
        [&](const triton::DialectInferLayoutInterface *delegate, Attribute enc,
            Attribute &result) {
          return delegate->inferExpandDimsOpEncoding(enc, axis, result, loc);
        });
  }

  LogicalResult inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                                   Attribute retEncoding,
                                   std::optional<Location> loc) const override {
    Attribute unwrappedRetEncoding = unwrapNoVerifyLayout(retEncoding);
    const triton::DialectInferLayoutInterface *delegate =
        getInferLayoutInterfaceFor(unwrappedRetEncoding);
    if (!delegate)
      return failure();
    return delegate->inferDotOpEncoding(unwrapNoVerifyLayout(operandEncoding),
                                        opIdx, unwrappedRetEncoding, loc);
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const override {
    return delegateInferredLayout(
        srcEnc, dstEnc,
        [&](const triton::DialectInferLayoutInterface *delegate, Attribute enc,
            Attribute &result) {
          return delegate->inferReshapeOpEncoding(srcShape, enc, dstShape,
                                                  result, loc);
        });
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    if (hasNoVerifyLayout(expected) || hasNoVerifyLayout(got))
      return success();

    Attribute unwrappedExpected = unwrapNoVerifyLayout(expected);
    Attribute unwrappedGot = unwrapNoVerifyLayout(got);
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
