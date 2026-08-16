#include "TritonAMDGPUTransforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "triton/Analysis/PlanValueGraph.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUDUMPPLANVALUEGRAPH
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

struct TritonAMDGPUDumpPlanValueGraphPass
    : impl::TritonAMDGPUDumpPlanValueGraphBase<
          TritonAMDGPUDumpPlanValueGraphPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<triton::plan::PlanValueGraph, 1> graphs;
    for (triton::FuncOp function : module.getOps<triton::FuncOp>()) {
      FailureOr<triton::plan::PlanValueGraph> graph =
          triton::plan::PlanValueGraph::build(function);
      if (failed(graph)) {
        function.emitError("failed to build plan value graph");
        return signalPassFailure();
      }
      if (strict && failed(graph->verify(/*strict=*/true))) {
        function.emitError("plan value graph failed strict verification");
        return signalPassFailure();
      }
      graphs.push_back(std::move(*graph));
    }

    std::string json = triton::plan::serializePlanValueGraphs(graphs, module);
    if (outputPath == "-") {
      llvm::outs() << json;
      return;
    }

    llvm::SmallString<256> path(outputPath);
    bool outputIsJSON = llvm::sys::path::extension(path) == ".json";
    if (!outputIsJSON) {
      if (std::error_code ec = llvm::sys::fs::create_directories(path)) {
        module.emitError("cannot create plan analysis directory '")
            << outputPath << "': " << ec.message();
        return signalPassFailure();
      }
      std::string fileName =
          graphs.empty()
              ? "empty.plan-values.json"
              : (graphs.front().getSemanticFingerprint() + ".plan-values.json")
                    .str();
      llvm::sys::path::append(path, fileName);
    } else {
      llvm::SmallString<256> parent = llvm::sys::path::parent_path(path);
      if (!parent.empty()) {
        if (std::error_code ec = llvm::sys::fs::create_directories(parent)) {
          module.emitError("cannot create plan analysis directory '")
              << parent << "': " << ec.message();
          return signalPassFailure();
        }
      }
    }

    std::error_code ec;
    llvm::ToolOutputFile output(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError("cannot open plan value graph sidecar '")
          << path << "': " << ec.message();
      return signalPassFailure();
    }
    output.os() << json;
    output.keep();
  }
};

} // namespace
} // namespace mlir
