#ifndef TRITON_TLX_PASSES
#define TRITON_TLX_PASSES

include "mlir/Pass/PassBase.td"

def TritonTLXAttachMetadata : Pass</*cli-arg*/"triton-tlx-attach-metadata", /*Op*/"mlir::ModuleOp"> {
  let summary = "Attach metadata to the module";
  let description = [{
    The pass attaches metadata to the TritonDialect module to help make TritonGPU or TritonNvidiaGPU integrate
    better into frontend DSL and TritonDialect.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];

  let options = [
      Option<"target", "target",
            "std::string", /*default*/"\"\"",
            "the GPU target, e.g., cuda:80, hip:gfx942">,
      Option<"numWarps", "num-warps",
             "int32_t", /*default*/"4",
             "number of warps">,
      Option<"threadsPerWarp", "threads-per-warp",
             "int32_t", /*default*/"32",
             "number of threads per warp">,
      Option<"numCTAs", "num-ctas",
             "int32_t", /*default*/"1",
             "number of ctas in a cga">,
   ];
}

#endif // TRITON_TLX_PASSES
