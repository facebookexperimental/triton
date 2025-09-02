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
  let summary = "Propagate layout information";

  let description = [{
    This pass propagates layout information from the tlx::RequireLayoutOp and
    tlx::ReleaseLayoutOp by doing a backward and forward dataflow analysis. It
    is expected that these ops would be either completely eliminated or turned
    into ttg::ConvertLayoutOp(s).
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TLXInsertRequireLayout : Pass<"tlx-insert-require-layout", "mlir::ModuleOp"> {
  let summary = "Inserts a tlx::RequireLayoutOp op before the LocalLoad that feeds a tl.dot";

  let description = [{
    This pass inserts a tlx::RequireLayoutOp op before the LocalLoad that feeds a tl.dot.
    This layout will then be propagated to the local alloc, by the layout propagation pass.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];

}
#endif // TRITON_TLX_PASSES
