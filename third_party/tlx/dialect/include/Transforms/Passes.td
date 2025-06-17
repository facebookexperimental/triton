#ifndef TRITON_TLX_PASSES
#define TRITON_TLX_PASSES

include "mlir/Pass/PassBase.td"

def TritonTLXFixup : Pass</*cli-arg*/"triton-tlx-fixup", /*Op*/"mlir::ModuleOp"> {
  let summary = "Fixup the IR for TritonTLX";
  let description = [{
    The pass did some fixup to the TritonDialect module to help make TritonGPU or TritonNvidiaGPU integrate
    better into frontend DSL and TritonDialect, such as attaching metadata to the module.
  }];

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

def TlxPropagateLayout : Pass<"tlx-propagate-layout", "mlir::ModuleOp"> {
  let summary = "propagate layout information";

  let description = [{
    TODO
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

#endif // TRITON_TLX_PASSES
