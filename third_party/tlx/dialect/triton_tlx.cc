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
      ->def("make_tensor_memory_encoding_attr",
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
      .def("create_convert_layout",
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
              return self.create<ttg::ConvertLayoutOp>(newType, v);
            } else {
              throw std::runtime_error("Unsupported type");
            }
            newType.dump();
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
      .def(
          "create_require_layout",
          [](TritonOpBuilder &self, Value &v, int opIdx) -> Value {

            Value arg = v;

            auto argType = cast<RankedTensorType>(arg.getType());
            assert(argType.getEncoding() && "unexpected tensor type");

            auto order = ttg::getOrderForMemory(argType);
            llvm::SmallVector<unsigned> newOrder = order;
            if (opIdx == 1) {
              newOrder = {0, 1};
            } else {
              newOrder = {1, 0};
            }

            Attribute SharedMemorySpace =
                ttg::SharedMemorySpaceAttr::get(argType.getContext());
            auto CTALayout = ttg::getCTALayout(argType.getEncoding());
            auto newLayout = ttg::NVMMASharedEncodingAttr::get(
                argType.getContext(), argType.getShape(), newOrder, CTALayout,
                argType.getElementType(), false);
            auto newType = ttg::MemDescType::get(argType.getShape(), argType.getElementType(),
                                            newLayout, SharedMemorySpace);

            // if (auto type = dyn_cast<ttg::MemDescType>(v.getType())) {
            //   newType = ttg::MemDescType::get(
            //       type.getShape(), type.getElementType(), encoding,
            //       type.getMemorySpace(), type.getMutableMemory());
            // } else {
            //   throw std::runtime_error("Unsupported type");
            // }
            return self.create<tlx::RequireLayoutOp>(newType, v);
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
