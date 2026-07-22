#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"
#include "triton/Tools/PluginUtils.h"
#include "triton/Tools/Sys/GetEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

mlir::FailureOr<bool> liftTritonControlFlowToSCF(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  context->getOrLoadDialect<mlir::ub::UBDialect>();

  mlir::ControlFlowToSCFTransformation transformation;
  bool changed = false;

  mlir::WalkResult result = module.walk([&](mlir::triton::FuncOp funcOp) {
    if (funcOp.getBody().empty())
      return mlir::WalkResult::advance();

    mlir::DominanceInfo domInfo(funcOp.getOperation());
    auto visitor = [&](mlir::Operation *innerOp) -> mlir::WalkResult {
      for (mlir::Region &region : innerOp->getRegions()) {
        mlir::FailureOr<bool> changedRegion =
            mlir::transformCFGToSCF(region, transformation, domInfo);
        if (mlir::failed(changedRegion))
          return mlir::WalkResult::interrupt();
        changed |= *changedRegion;
      }
      return mlir::WalkResult::advance();
    };

    if (funcOp->walk<mlir::WalkOrder::PostOrder>(visitor).wasInterrupted())
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  if (result.wasInterrupted())
    return mlir::failure();
  return changed;
}

} // namespace

void init_triton_analysis(py::module &&m) {
  py::class_<mlir::ModuleAllocation>(m, "allocation", py::module_local())
      .def(py::init<mlir::ModuleOp>());
  py::class_<mlir::ModuleMembarAnalysis>(m, "membar", py::module_local())
      .def(py::init<mlir::ModuleAllocation *>())
      .def("run", &mlir::ModuleMembarAnalysis::run);
}

void init_triton_passes_common(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
  ADD_PASS_WRAPPER_0("print_ir", createPrintIRPass);
}

void init_triton_passes_ttir(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_0("add_combine", createTritonCombineOps);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast", createTritonReorderBroadcast);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_descriptor_to_pointer",
                     createTritonRewriteTensorDescriptorToPointer);
  ADD_PASS_WRAPPER_0("add_loop_unroll", createTritonLoopUnroll);
  ADD_PASS_WRAPPER_0("add_simplify_single_trip_while",
                     createTritonSimplifySingleTripWhile);
  ADD_PASS_WRAPPER_0("add_uplift_while_to_for", createTritonUpliftWhileToFor);
  ADD_PASS_WRAPPER_0("add_triton_licm", createTritonLoopInvariantCodeMotion);
  ADD_PASS_WRAPPER_0("add_loop_aware_cse", createTritonLoopAwareCSE);
  ADD_PASS_OPTION_WRAPPER_4("add_convert_to_ttgpuir",
                            createConvertTritonToTritonGPU, const std::string &,
                            int, int, int);
}

void init_triton_passes_ttgpuir(py::module &&m) {
  using namespace mlir;
  using namespace mlir::triton::gpu;
  using namespace mlir::triton::instrument;
  ADD_PASS_OPTION_WRAPPER_1("add_coalesce", createTritonGPUCoalesce, unsigned);
  ADD_PASS_WRAPPER_0("add_optimize_thread_locality",
                     createTritonGPUOptimizeThreadLocality);
  ADD_PASS_OPTION_WRAPPER_1("add_hoist_tmem_alloc",
                            createTritonGPUHoistTMEMAlloc, bool);
  ADD_PASS_OPTION_WRAPPER_2("add_assign_latencies",
                            createTritonGPUAssignLatencies, int, bool);
  ADD_PASS_OPTION_WRAPPER_2("add_schedule_loops", createTritonGPUScheduleLoops,
                            int, bool);
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline", createTritonGPUPipeline, int, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_warp_specialize",
                            createTritonGPUAutomaticWarpSpecialization, int);
  ADD_PASS_WRAPPER_0("add_prefetch", createTritonGPUPrefetch);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul", createTritonGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     createTritonGPUReorderInstructions);
  ADD_PASS_OPTION_WRAPPER_1("add_f32_dot_tc", createTritonGPUF32DotTC, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_optimize_dot_operands",
                            createTritonGPUOptimizeDotOperands, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_remove_layout_conversions",
                            createTritonGPURemoveLayoutConversions, unsigned);
  ADD_PASS_WRAPPER_0("add_reduce_data_duplication",
                     createTritonGPUReduceDataDuplication);
  ADD_PASS_WRAPPER_0("add_allocate_warp_groups",
                     createTritonGPUAllocateWarpGroups);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory", createAllocateSharedMemory);
  ADD_PASS_WRAPPER_0("add_allocate_global_scratch_memory",
                     createTritonGPUGlobalScratchAllocationPass);
  ADD_PASS_WRAPPER_0("add_combine_tensor_select_and_if",
                     createTritonGPUCombineTensorSelectAndIf);
  ADD_PASS_WRAPPER_0("add_optimize_accumulator_init",
                     createTritonGPUOptimizeAccumulatorInit);
  ADD_PASS_WRAPPER_0("add_fuse_nested_loops", createTritonGPUFuseNestedLoops);
  ADD_PASS_WRAPPER_0("add_coalesce_async_copy",
                     createTritonGPUCoalesceAsyncCopy);
  ADD_PASS_WRAPPER_0("add_global_sanitizer",
                     createTritonInstrumentGlobalSanitizer);
  ADD_PASS_WRAPPER_0("add_concurrency_sanitizer",
                     createTritonInstrumentConcurrencySanitizer);
  ADD_PASS_WRAPPER_0("add_fp_sanitizer", createTritonInstrumentFpSanitizer);
  ADD_PASS_WRAPPER_0("add_optimize_partition_warps",
                     createTritonGPUOptimizePartitionWarps);
  ADD_PASS_WRAPPER_0("add_partition_scheduling",
                     createTritonGPUPartitionScheduling);
  m.def("add_canonicalize_llvm_ir", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(createCanonicalizeLLVMIR());
  });
}

void init_plugin_passes(py::module &&m) {
  for (const auto &plugin : mlir::triton::plugin::loadPlugins()) {
    for (const auto &pass : plugin.listPasses()) {
      m.def(
          pass.name,
          [pass](mlir::PassManager &pm, std::vector<std::string> args) {
            pass.addPass(&pm, args);
          },
          py::arg("pm"), py::arg("args") = std::vector<std::string>());
    }
  }
}

void init_triton_passes_convert(py::module &&m) {
  using namespace mlir;
  m.def("triton_lift_cf_to_scf", [](ModuleOp &mod) {
    FailureOr<bool> changed = liftTritonControlFlowToSCF(mod);
    if (failed(changed))
      throw std::runtime_error("failed to lift Triton control flow to SCF");
    return *changed;
  });
  ADD_PASS_WRAPPER_0("add_scf_to_cf", createSCFToControlFlowPass);
  ADD_PASS_WRAPPER_0("add_cf_to_llvmir", createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_index_to_llvmir", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_arith_to_llvmir", createArithToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("add_nvvm_to_llvm", createConvertNVVMToLLVMPass);
}

void init_triton_passes_llvmir(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_di_scope", mlir::createLLVMDIScope);
  ADD_PASS_WRAPPER_0("add_di_local_variable", mlir::createLLVMDILocalVariable);
}

void init_gluon_passes(py::module &&m) {
  using namespace mlir;
  namespace gluon = mlir::triton::gluon;
  ADD_PASS_WRAPPER_0("add_resolve_auto_encodings",
                     gluon::createGluonResolveAutoEncodingsPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", gluon::createGluonCanonicalize);
  ADD_PASS_WRAPPER_0("add_inliner", gluon::createGluonInline);
  ADD_PASS_WRAPPER_0("add_infer_coalesced_encodings",
                     gluon::createGluonInferCoalescedEncodingsPass);
}

void init_triton_passes(py::module &&m) {
  init_triton_analysis(m.def_submodule("analysis"));
  init_triton_passes_common(m.def_submodule("common"));
  init_triton_passes_convert(m.def_submodule("convert"));
  init_triton_passes_ttir(m.def_submodule("ttir"));
  init_triton_passes_ttgpuir(m.def_submodule("ttgpuir"));
  init_triton_passes_llvmir(m.def_submodule("llvmir"));
  init_gluon_passes(m.def_submodule("gluon"));
  init_plugin_passes(m.def_submodule("plugin"));
}
