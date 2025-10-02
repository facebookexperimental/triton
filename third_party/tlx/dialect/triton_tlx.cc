#include "IR/Dialect.h"
#include "Transforms/Passes.h"
#include "ir.h" // TritonOpBuilder
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "tlx/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Casting.h"

namespace py = pybind11;
using namespace ir;
using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace tlx = triton::tlx;

void init_triton_tlx_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
      ->def(
          "create_memdesc_subview",
          [](TritonOpBuilder &self, Value localAlloc,
             Value bufferIdx) -> mlir::Value {
            auto localAllocType = cast<ttg::MemDescType>(localAlloc.getType());
            auto localAllocShape = localAllocType.getShape();
            auto context = self.getBuilder().getContext();
            Type memDescType;
            if (localAllocShape.size() == 1) {
              memDescType = ttg::MemDescType::get(
                  {1}, localAllocType.getElementType(),
                  localAllocType.getEncoding(), localAllocType.getMemorySpace(),
                  /*mutableMemory=*/localAllocType.getMutableMemory());
            } else {
              memDescType = ttg::MemDescType::get(
                  localAllocShape.drop_front(), localAllocType.getElementType(),
                  localAllocType.getEncoding(), localAllocType.getMemorySpace(),
                  /*mutableMemory=*/localAllocType.getMutableMemory());
            }
            return self.create<ttg::MemDescIndexOp>(memDescType, localAlloc,
                                                    bufferIdx);
          })
      .def("create_require_layout",
           [](TritonOpBuilder &self, Value &v, Attribute &encoding) -> Value {
             Type newType;
             if (auto type = dyn_cast<ttg::MemDescType>(v.getType())) {
               newType = ttg::MemDescType::get(
                   type.getShape(), type.getElementType(), encoding,
                   type.getMemorySpace(), type.getMutableMemory());
               return self.create<tlx::RequireLayoutOp>(newType, v);
             } else if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
               newType = RankedTensorType::get(type.getShape(),
                                               type.getElementType(), encoding);
               return self.create<tlx::RequireLayoutOp>(newType, v);
             } else {
               throw std::runtime_error("Unsupported type");
             }
           })
      .def("create_release_layout",
           [](TritonOpBuilder &self, Value &v) -> Value {
             if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
               assert(type.getEncoding() && "Expect layout encoding");
               auto newType = RankedTensorType::get(type.getShape(),
                                                    type.getElementType());
               return self.create<tlx::ReleaseLayoutOp>(newType, v);
             } else {
               throw std::runtime_error("Unsupported type");
             }
           })
      .def("create_local_load",
           [](TritonOpBuilder &self, Value subView,
              std::optional<Value> asyncToken) -> mlir::Value {
             auto subViewType = cast<ttg::MemDescType>(subView.getType());
             auto newType = RankedTensorType::get(subViewType.getShape(),
                                                  subViewType.getElementType());
             return self.create<ttg::LocalLoadOp>(newType, subView,
                                                  asyncToken.value_or(Value()));
           })
      .def("create_local_store",
           [](TritonOpBuilder &self, Value &dst, Value &regValues) -> void {
             self.create<ttg::LocalStoreOp>(regValues, dst);
           })
      .def("make_swizzled_shared_encoding_attr",
           [](TritonOpBuilder &self, unsigned vectorSize, unsigned perPhase,
              unsigned maxPhase, std::vector<unsigned> order,
              std::vector<unsigned> CTAsPerCGA,
              std::vector<unsigned> CTASplitNum,
              std::vector<unsigned> CTAOrder) {
             assert(order.size() == CTAsPerCGA.size() && "shape mismatch");
             assert(order.size() == CTASplitNum.size() && "shape mismatch");
             assert(order.size() == CTAOrder.size() && "shape mismatch");
             auto context = self.getBuilder().getContext();
             auto CTALayout = ttg::CTALayoutAttr::get(context, CTAsPerCGA,
                                                      CTASplitNum, CTAOrder);
             return mlir::cast<Attribute>(ttg::SwizzledSharedEncodingAttr::get(
                 context, vectorSize, perPhase, maxPhase, order, CTALayout));
           })
      .def("make_tensor_memory_encoding_attr",
           [](TritonOpBuilder &self, unsigned blockM, unsigned blockN,
              bool unpacked, unsigned CTASplitM, unsigned CTASplitN) {
             auto context = self.getBuilder().getContext();
             return mlir::cast<Attribute>(ttng::TensorMemoryEncodingAttr::get(
                 context, blockM, blockN, unpacked, CTASplitM, CTASplitN));
           })
      .def("make_nv_mma_shared_encoding_attr",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              std::vector<unsigned> order, Type &elemType,
              std::vector<unsigned> CTAsPerCGA,
              std::vector<unsigned> CTASplitNum, std::vector<unsigned> CTAOrder,
              bool fp4Padded) {
             /* Validation logic for user defined layout encoding begin */
             assert(shape.size() == order.size());
             assert(order.size() == CTAsPerCGA.size());
             assert(CTAsPerCGA.size() == CTASplitNum.size());
             assert(CTASplitNum.size() == CTAOrder.size());
             /* Validation logic for user defined layout encoding end */

             auto context = self.getBuilder().getContext();
             auto CTALayout = ttg::CTALayoutAttr::get(context, CTAsPerCGA,
                                                      CTASplitNum, CTAOrder);
             return mlir::cast<Attribute>(ttg::NVMMASharedEncodingAttr::get(
                 context, shape, order, CTALayout, elemType, fp4Padded));
           })
      .def("make_nv_mma_encoding_attr",
           [](TritonOpBuilder &self, Value opndA, Value opndAcc,
              unsigned versionMajor, unsigned versionMinor,
              unsigned moduleNumWarps) {
             auto context = self.getBuilder().getContext();
             auto dtypeA =
                 cast<ttg::TensorOrMemDesc>(opndA.getType()).getElementType();
             auto retType = cast<RankedTensorType>(opndAcc.getType());
             auto retShapePerCTA = retType.getShape();
             Block *parentBlock = self.getBuilder().getInsertionBlock();
             unsigned numWarps =
                 ttg::maybeLookupNumWarps(parentBlock).value_or(moduleNumWarps);
             auto instrShape = mmaVersionToInstrShape(
                 versionMajor, retShapePerCTA, dtypeA, numWarps);
             // Default to row partitioning for now. Should be smarter.
             SmallVector<unsigned, 2> warpsPerCTA = {numWarps, 1};
             SmallVector<unsigned, 2> CTAsPerCGA = {1, 1};
             SmallVector<unsigned, 2> CTASplitNum = {1, 1};
             SmallVector<unsigned, 2> CTAOrder = {1, 0};
             auto CTALayout = ttg::CTALayoutAttr::get(context, CTAsPerCGA,
                                                      CTASplitNum, CTAOrder);
             return mlir::cast<Attribute>(ttg::NvidiaMmaEncodingAttr::get(
                 context, versionMajor, versionMinor, warpsPerCTA, CTALayout,
                 instrShape));
           })
      .def("make_dot_operand_encoding_attr",
           [](TritonOpBuilder &self, Value opnd, unsigned opIdx,
              Attribute parentEnc) -> Attribute {
             auto context = self.getBuilder().getContext();
             auto eltType =
                 cast<RankedTensorType>(opnd.getType()).getElementType();
             return ttg::DotOperandEncodingAttr::get(context, opIdx, parentEnc,
                                                     eltType);
           })
      .def("make_default_tmem_compatible_tensor_layout_encoding",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type elementType, int moduleNumWarps, int threadsPerWarp,
              int numCTAs) {
             // Include various assert to vet the input to make sure they're
             // valid for MMAv5. See also lib/Analysis/Utiity.cpp:supportMMA
             assert(shape.size() == 2 &&
                    "Only supporting 2D tensors for TMEM layout.");
             assert((!elementType.isInteger()) &&
                    "Integer type not supported.");

             Block *parentBlock = self.getBuilder().getInsertionBlock();
             int numWarps =
                 ttg::maybeLookupNumWarps(parentBlock).value_or(moduleNumWarps);
             assert((numWarps == 4 || numWarps == 8) &&
                    "Currently only support numWarps 4 or 8 for TMEM load and "
                    "store.");

             ttg::BlockedEncodingAttr defaultBlockedEncoding =
                 ttg::getDefaultBlockedEncoding(self.getContext(), shape,
                                                numWarps, threadsPerWarp,
                                                numCTAs);
             auto oldType = RankedTensorType::get(shape, elementType,
                                                  defaultBlockedEncoding);
             auto oldTypeShapePerCTA = ttg::getShapePerCTA(oldType);
             auto rank = oldTypeShapePerCTA.size();
             assert((oldTypeShapePerCTA[rank - 2] % 64 == 0 &&
                     ((oldTypeShapePerCTA[rank - 1] % 8 == 0) ||
                      oldTypeShapePerCTA[rank - 1] == 1)) &&
                    "Shape unsupported by TMEM ops.");

             Attribute newDistributedEncoding =
                 nvidia_gpu::getTmemCompatibleLayout(shape[0], shape[1],
                                                     oldType, numWarps);
             return newDistributedEncoding;
           })
      .def("create_fence_async_shared",
           [](TritonOpBuilder &self) -> void {
             self.create<ttng::FenceAsyncSharedOp>(false);
           })
      .def("create_warp_group_dot",
           [](TritonOpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &c, InputPrecision inputPrecision,
              int maxNumImpreciseAcc, bool isAsync) -> mlir::Value {
             return self.create<ttng::WarpGroupDotOp>(
                 c.getType(), a, b, c, nullptr, inputPrecision,
                 maxNumImpreciseAcc, isAsync);
           })
      .def("create_warp_group_dot_wait",
           [](TritonOpBuilder &self, std::vector<Value> inputs,
              unsigned pendings) -> std::vector<Value> {
             // Extract original sources for inputs wrapped in ReleaseLayoutOp.
             // These are the true operands to WarpGroupDotWaitOp.
             std::vector<Value> realInputs;
             realInputs.reserve(inputs.size());
             for (Value input : inputs) {
               if (auto releaseOp =
                       dyn_cast<tlx::ReleaseLayoutOp>(input.getDefiningOp()))
                 realInputs.push_back(releaseOp.getSrc());
               else
                 realInputs.push_back(input);
             }

             // Create the warp group wait op using the unwrapped input values.
             auto waitOp =
                 self.create<ttng::WarpGroupDotWaitOp>(realInputs, pendings);
             assert(waitOp.getNumResults() == inputs.size() &&
                    "Result count mismatch with inputs");

             // For each original input:
             // - If it was a ReleaseLayoutOp, move it after the wait op and
             // rewire it.
             // - Otherwise, return the raw wait result.
             std::vector<Value> outputs;
             outputs.reserve(inputs.size());
             for (unsigned i = 0; i < inputs.size(); ++i) {
               if (auto release = dyn_cast<tlx::ReleaseLayoutOp>(
                       inputs[i].getDefiningOp())) {
                 release->moveAfter(waitOp.getOperation());
                 release.getOperation()->setOperand(0, waitOp.getResult(i));
                 outputs.push_back(release.getResult());
               } else {
                 outputs.push_back(waitOp.getResult(i));
               }
             }
             return outputs;
           })
      // Barrier Ops
      .def("create_alloc_barriers",
           [](TritonOpBuilder &self, int numBarriers, int arriveCount,
              Attribute barrierEncoding) -> mlir::Value {
             auto context = self.getBuilder().getContext();
             auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             auto barriersMemDescType = ttg::MemDescType::get(
                 {numBarriers}, self.getBuilder().getI64Type(), barrierEncoding,
                 memorySpace, /*mutableMemory=*/true);

             auto singleBarrierMemDescType = ttg::MemDescType::get(
                 {1}, self.getBuilder().getI64Type(), barrierEncoding,
                 memorySpace, /*mutableMemory=*/true);

             // Allocate buffer in shared memory
             mlir::Value bufferViews =
                 self.create<ttg::LocalAllocOp>(barriersMemDescType);

             //  Init barrier in each slot
             for (auto i = 0; i < numBarriers; i++) {
               // Obtain the single buffer view
               Value idx = self.getBuilder().create<arith::ConstantIntOp>(
                   bufferViews.getLoc(), i, 32);
               mlir::Value buf = self.create<ttg::MemDescIndexOp>(
                   singleBarrierMemDescType, bufferViews, idx);

               // Initialize mbarrier at buf view
               self.create<ttng::InitBarrierOp>(buf,
                                                /*number of arrives*/
                                                arriveCount);
             }

             // Return mlir::Value
             return bufferViews;
           })
      .def("create_barrier_wait",
           [](TritonOpBuilder &self, Value mbarrerLoc, Value phase,
              Value pred) -> void {
             self.create<ttng::WaitBarrierOp>(mbarrerLoc, phase, pred);
           })
      .def(
          "create_barrier_arrive",
          [](TritonOpBuilder &self, Value mbarrerLoc, int arriveCount) -> void {
            self.create<ttng::ArriveBarrierOp>(mbarrerLoc, arriveCount);
          })
      .def("create_named_barrier_wait",
           [](TritonOpBuilder &self, Value barrier, Value numThreads) -> void {
             self.create<ttng::NamedBarrierWaitOp>(barrier, numThreads);
           })
      .def("create_named_barrier_arrive",
           [](TritonOpBuilder &self, Value barrier, Value numThreads) -> void {
             self.create<ttng::NamedBarrierArriveOp>(barrier, numThreads);
           })
      .def("create_barrier_expect",
           [](TritonOpBuilder &self, Value mbarrerLoc, int expectBytes,
              Value pred) -> void {
             self.create<ttng::BarrierExpectOp>(mbarrerLoc, expectBytes, pred);
           })
      .def("create_tmem_alloc",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding,
              std::optional<Value> alias) -> mlir::Value {
             auto context = self.getBuilder().getContext();
             auto memorySpace = ttng::TensorMemorySpaceAttr::get(context);
             auto memDesc =
                 ttg::MemDescType::get(shape, elementType, encoding,
                                       memorySpace, /*mutableMemory=*/true);
             if (alias)
               return self.create<tlx::LocalAliasOp>(memDesc, *alias);
             else
               return self.create<ttng::TMEMAllocOp>(memDesc, nullptr);
           })
      .def("create_tmem_load",
           [](TritonOpBuilder &self, Value subView, Attribute &layoutEncoding,
              std::optional<Value> asyncToken) -> mlir::Value {
             auto subViewType = cast<ttg::MemDescType>(subView.getType());

             // layoutEncoding must be TMEM compatible
             auto newType = RankedTensorType::get(subViewType.getShape(),
                                                  subViewType.getElementType(),
                                                  layoutEncoding);
             return self.create<ttng::TMEMLoadOp>(newType, subView,
                                                  asyncToken.value_or(Value()));
           })
      .def("create_tmem_store",
           [](TritonOpBuilder &self, Value &dst, Value &src) -> void {
             Value pred = self.create<arith::ConstantIntOp>(1, 1);
             self.create<ttng::TMEMStoreOp>(dst, src, pred);
           })
      .def("create_tmem_subslice",
           [](TritonOpBuilder &self, Value &src, int offset,
              int size) -> mlir::Value {
             // There're already checks for src and dst layouts in verifer
             // TMEMSubSliceOp::verify()
             // We do some reasonable extra checks here to make sure front end
             // only passes valid inputs to the op
             auto srcTy = dyn_cast<triton::gpu::MemDescType>(src.getType());
             assert(srcTy != nullptr && "Expect MemDescType for src");
             auto encoding =
                 dyn_cast<ttng::TensorMemoryEncodingAttr>(srcTy.getEncoding());
             auto blockN = encoding.getBlockN();
             assert(offset >= 0 && offset < blockN && "Invalid offset");
             assert(size > 0 && size <= blockN - offset && "Invalid size");
             return self.create<ttng::TMEMSubSliceOp>(src, offset, size);
           })
      .def("create_tcgen5_dot",
           [](TritonOpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &d, std::optional<Value> useD,
              std::optional<Value> pred, std::vector<Value> mBarriers) -> void {
             Value predTrue = self.create<arith::ConstantIntOp>(1, 1);
             std::vector<Value> barrierPreds(mBarriers.size(), predTrue);
             auto tokType = self.getBuilder().getType<ttg::AsyncTokenType>();
             self.create<ttng::TCGen5MMAOp>(
                 tokType, a, b, d, Value(),
                 useD.has_value() ? useD.value() : predTrue /*useD*/,
                 pred.has_value() ? pred.value() : predTrue /*pred */,
                 false /* two_ctas*/, ValueRange(mBarriers),
                 ValueRange(barrierPreds), !mBarriers.empty() /* is_async */);
           })
      .def("create_tcgen05_commit",
           [](TritonOpBuilder &self, Value &barrier) -> void {
             self.create<ttng::TCGen5CommitOp>(barrier);
           })
      .def("create_async_commit_group",
           [](TritonOpBuilder &self,
              std::vector<Value> asyncTokens) -> mlir::Value {
             return self.create<ttg::AsyncCommitGroupOp>(asyncTokens);
           })
      .def("create_async_wait",
           [](TritonOpBuilder &self, std::vector<Value> asyncTokens,
              unsigned pendings) -> mlir::Value {
             return self.create<ttg::AsyncWaitOp>(asyncTokens, pendings);
           })
      .def("create_memdesc_trans",
           [](TritonOpBuilder &self, Value &arg,
              std::vector<int32_t> order) -> mlir::Value {
             return self.create<ttg::MemDescTransOp>(arg, order);
           })
      .def("create_memdesc_reinterpret",
           [](TritonOpBuilder &self, Value &src, Type &newElementType,
              std::vector<int64_t> newShape) -> mlir::Value {
             auto oldType = cast<ttg::MemDescType>(src.getType());
             assert(oldType && "Expect MemDescType for src");
             auto encoding = oldType.getEncoding();
             if (!oldType.getShape().equals(newShape)) {
               // Only accept unswizzled encoding for now.
               if (auto mmaEncoding =
                       dyn_cast<ttg::NVMMASharedEncodingAttr>(encoding)) {
                 if (mmaEncoding.getSwizzlingByteWidth() != 0)
                   llvm_unreachable("Only accept unswizzled encoding");
               } else if (auto swizzledEncoding =
                              dyn_cast<ttg::SwizzledSharedEncodingAttr>(
                                  encoding)) {
                 if (!(swizzledEncoding.getVec() == 1 &&
                       swizzledEncoding.getPerPhase() == 1 &&
                       swizzledEncoding.getMaxPhase() == 1))
                   llvm_unreachable("Only accept unswizzled encoding");
               }
             }

             auto newType = ttg::MemDescType::get(
                 newShape, newElementType, encoding, oldType.getMemorySpace(),
                 oldType.getMutableMemory());
             return self.create<ttg::MemDescReinterpretOp>(newType, src);
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding,
              std::string storage) -> Type {
             auto context = self.getBuilder().getContext();
             Attribute memorySpace;
             if (storage == "tmem")
               memorySpace = ttng::TensorMemorySpaceAttr::get(context);
             else if (storage == "smem") {
               memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             } else {
               llvm_unreachable("Unknown storage type");
             }
             return ttg::MemDescType::get(shape, elementType, encoding,
                                          memorySpace, /*mutableMemory=*/true);
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding,
              std::optional<Value> alias) -> mlir::Value {
             auto context = self.getBuilder().getContext();
             auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             auto memDesc =
                 ttg::MemDescType::get(shape, elementType, encoding,
                                       memorySpace, /*mutableMemory=*/true);
             if (alias)
               return self.create<tlx::LocalAliasOp>(memDesc, *alias);
             else
               return self.create<ttg::LocalAllocOp>(memDesc);
           })
      .def("create_alloc_clc_responses",
           [](TritonOpBuilder &self, int numResponses,
              Attribute clcResEncoding) -> mlir::Value {
             auto context = self.getBuilder().getContext();
             auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             auto memDescType = ttg::MemDescType::get(
                 {numResponses},
                 self.getBuilder().getIntegerType(128, /*signed=*/false),
                 clcResEncoding, memorySpace, /*mutableMemory=*/true);

             mlir::Value bufferViews =
                 self.create<ttg::LocalAllocOp>(memDescType);

             return bufferViews;
           })
      .def("clc_issue",
           [](TritonOpBuilder &self, Value responseAddr, Value mbar) -> void {
             self.create<ttng::AsyncCLCTryCancelOp>(mbar, responseAddr);
           })
      .def("clc_query",
           [](TritonOpBuilder &self, Value responseAddr
              // Value valid,
              //   Value ctaIdX, Value ctaIdY, Value ctaIdZ
              // ) -> py::tuple {
              ) -> Value {
             auto op = self.create<ttng::AsyncCLCQueryCancelOp>(responseAddr);

             //  return op->getResult(0);
             //  return py::make_tuple(0, 1, 2, 3);
             //  op.getResult(0).dump();
             //  op.getValid().dump();
             //  op.getValid().dump();
             //  return py::make_tuple(op.getValid(), 0, 0, 0);
             return op.getValid();
             //  return py::make_tuple(op.getResult(0), op.getResult(1), 0, 0);
             // op->getResult(2), op->getResult(3));
           })
      .def("create_async_TMA_load",
           [](TritonOpBuilder &self, Value desc, std::vector<Value> &coord,
              Value mbarrier, Value pred, Value result,
              CacheModifier cacheModifier, EvictionPolicy evictionPolicy,
              bool isVolatile) -> void {
             self.create<ttng::AsyncTMACopyGlobalToLocalOp>(
                 desc, coord, mbarrier, result, pred, cacheModifier,
                 evictionPolicy, isVolatile);
           })
      .def("create_async_TMA_store",
           [](TritonOpBuilder &self, Value desc, std::vector<Value> &coord,
              Value source) -> void {
             self.create<ttng::AsyncTMACopyLocalToGlobalOp>(desc, coord,
                                                            source);
           })
      .def("create_async_TMA_store_wait",
           [](TritonOpBuilder &self, int pendings) {
             self.create<ttng::TMAStoreWaitOp>(pendings);
           })
      .def("create_fence_async_shared",
           [](TritonOpBuilder &self, bool bCluster) -> OpState {
             return self.create<ttng::FenceAsyncSharedOp>(bCluster);
           });
}

void init_triton_tlx_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_tlx_propagate_layout", tlx::createTlxPropagateLayout);
  ADD_PASS_WRAPPER_0("add_tlx_insert_require_layout",
                     tlx::createTLXInsertRequireLayout);
  ADD_PASS_WRAPPER_0("add_tlx_rewrite_local_alias",
                     tlx::createTLXRewriteLocalAlias);
  ADD_PASS_OPTION_WRAPPER_4("add_triton_tlx_fixup", tlx::createTritonTLXFixup,
                            std::string, int32_t, int32_t, int32_t);
}

void init_triton_tlx(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::tlx::TLXDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_tlx_ir(m.def_submodule("tlx_ir"));
  init_triton_tlx_passes(m.def_submodule("tlx_passes"));
}
