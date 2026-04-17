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
      ListOption<"clusterDims", "cluster-dims", "int32_t",
             "cluster dimensions (X, Y, Z)">,
   ];
}

def TlxPropagateLayout : Pass<"tlx-propagate-layout", "mlir::ModuleOp"> {
  let summary = "Propagate layout information";

  let description = [{
    This pass propagates explicit layout constraints from tlx::RequireLayoutOp
    and tlx::ReleaseLayoutOp using backward and forward dataflow analyses.

    The pass owns the downstream consequences of those constraints:
    - retagging memdesc producers and carriers when the memdesc lattice reaches
      a concrete encoding
    - retagging supported tensor producers (`ttg.local_load`) and region-carried
      tensor values when the tensor lattice reaches a concrete encoding
    - applying consensus-based updates for RegionBranchOpInterface carriers and
      `ttg.warp_specialize` partition captures/arguments
    - lowering residual tensor constraints to `ttg.convert_layout`
    - validating post-conditions such as the absence of unresolved
      DummyTMEMLayoutAttr allocations

    Tensor constraints are expected to be either eliminated or lowered to
    `ttg.convert_layout`. Memdesc constraints usually fold into retagged types,
    but may remain explicit when propagation conservatively falls back on an
    unknown/conflicting layout.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TLXInsertRequireLayout : Pass<"tlx-insert-require-layout", "mlir::ModuleOp"> {
  let summary = "Insert explicit TLX layout constraints for local_load-to-dot paths";

  let description = [{
    This pass discovers dot-fed ttg.local_load operations and inserts the
    missing tlx.require_layout constraints needed for the AMD local_load-to-dot
    path.

    The pass only synthesizes explicit TLX layout constraints:
    - memdesc-side tlx.require_layout constraints for shared-memory swizzling
    - tensor-side tlx.require_layout constraints in place of dot-path
      ttg.convert_layout operations

    It does not own tensor/register propagation or convert cleanup; those are
    handled by tlx-propagate-layout and downstream cleanup passes.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tlx::TLXDialect"];

}

def TLXRewriteLocalAlias : Pass<"tlx-rewrite-local-alias", "mlir::ModuleOp"> {
  let summary = "Replace tlx::LocalAliasOp with the aliased local mem_desc";

  let description = [{
    This pass replaces a tlx::LocalAliasOp op with the original aliased mem_desc.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];

}

def TLXResolvePlaceholderLayouts : Pass<"tlx-resolve-placeholder-layouts", "mlir::ModuleOp"> {
  let summary = "Resolve placeholder layouts after function inlining";

  let description = [{
    This pass resolves placeholder layout encodings that were generated during
    initial lowering. After function inlining, we have more context to determine
    the correct layouts for TMEM loads/stores and other TLX operations.

    The pass replaces:
    - DummyRegisterLayoutAttr -> BlockedEncodingAttr

    Each resolved layout uses the same default values as the corresponding
    Python make_default() methods.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TLXPrintTTGIRToTLX : Pass<"tlx-print-ttgir-to-tlx", "mlir::ModuleOp"> {
  let summary = "Print TTGIR operations with their TLX equivalents";

  let description = [{
    This pass walks through the TTGIR module and prints annotations showing
    the mapping from TTGIR operations back to their TLX API equivalents.
    This is useful for understanding the correspondence between the high-level
    TLX Python API and the low-level TTGIR operations.

    Example mappings:
    - ttng::InitBarrierOp -> tlx.alloc_barriers
    - ttng::WaitBarrierOp -> tlx.barrier_wait
    - ttng::WarpGroupDotOp -> tlx.async_dot (Hopper)
    - ttng::TCGen5MMAOp -> tlx.async_dot (Blackwell)
    - ttg::LocalAllocOp -> tlx.local_alloc (smem)
    - ttng::TMEMAllocOp -> tlx.local_alloc (tmem)
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def TLXStorageAliasLowering : Pass<"tlx-storage-alias-lowering", "mlir::ModuleOp"> {
  let summary = "Lower storage alias operations";

  let description = [{
    This pass lowers storage alias operations by:

    1. Computing or validating storage alias sizes - For each storage_alias_spec,
       computes the required buffer size as the maximum of all referencing
       storage_alias_local_alloc operations. If an explicit size is provided,
       validates it is sufficient.

    2. Materializing storage alias allocations - Creates LocalAllocOp/TMEMAllocOp
       for each storage_alias_spec and replaces storage_alias_local_alloc with
       local_alias referencing the allocation.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

#endif // TRITON_TLX_PASSES
