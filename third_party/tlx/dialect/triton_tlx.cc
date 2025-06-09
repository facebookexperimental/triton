#include "IR/Dialect.h"
#include "ir.h" // TritonOpBuilder
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace py = pybind11;
using namespace ir;
using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

void init_triton_tlx_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
      ->def("create_require_layout",
            [](TritonOpBuilder &self, Value &v, Attribute &encoding) -> Value {
              Type newType;
              if (auto type = dyn_cast<ttg::MemDescType>(v.getType())) {
                newType = ttg::MemDescType::get(
                    type.getShape(), type.getElementType(), encoding,
                    type.getMemorySpace(), type.getMutableMemory());
              } else {
                throw std::runtime_error("Unsupported type");
              }
              return self.create<tlx::RequireLayoutOp>(newType, v);
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
      .def("make_swizzled_shared_encoding_attr",
           [](TritonOpBuilder &self, unsigned vectorSize, unsigned perPhase,
              unsigned maxPhase, std::vector<unsigned> order,
              std::vector<unsigned> CTAsPerCGA,
              std::vector<unsigned> CTASplitNum,
              std::vector<unsigned> CTAOrder) {
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
      .def("create_tmem_alloc",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding) -> mlir::Value {
             auto context = self.getBuilder().getContext();
             auto memorySpace = ttng::TensorMemorySpaceAttr::get(context);
             auto memDesc =
                 ttg::MemDescType::get(shape, elementType, encoding,
                                       memorySpace, /*mutableMemory=*/true);
             return self.create<ttng::TMEMAllocOp>(memDesc, nullptr);
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
           });
}

void init_triton_tlx_passes(py::module &&m) {
  // TODO: add TLX passes
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
