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
      .def("create_async_TMA_load",
           [](TritonOpBuilder &self, Value desc, std::vector<Value> &coord,
              Value mbarrier, Value result, CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy, bool isVolatile) -> void {
             Value tmaPtr = self.create<ttng::TensorDescToTMAPtrOp>(desc);
             Value pred = self.create<arith::ConstantIntOp>(1, 1);
             self.create<ttng::AsyncTMACopyGlobalToLocalOp>(
                 tmaPtr, coord, mbarrier, result, pred, cacheModifier,
                 evictionPolicy, isVolatile);
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
