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

def TLXStorageAliasSizeDefinition : Pass<"tlx-storage-alias-size-definition", "mlir::ModuleOp"> {
  let summary = "Compute or validate storage alias sizes";

  let description = [{
    For each storage_alias_spec, this pass computes the required buffer size
    as the maximum of all referencing storage_alias_local_alloc operations.

    If the storage_alias_spec has an explicit buffer_size_bytes:
    - Validates that the size is sufficient for all referencing allocations
    - Emits an error if the size is too small

    If the storage_alias_spec has no explicit size:
    - Computes the size as the maximum of all referencing allocations
    - Updates the storage_alias_spec with the computed size
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TLXStorageAliasAllocation : Pass<"tlx-storage-alias-allocation", "mlir::ModuleOp"> {
  let summary = "Materialize storage alias allocations";

  let description = [{
    This pass eliminates storage_alias_spec operations by materializing actual
    memory allocations and updates all referencing storage_alias_local_alloc
    operations to use local_alias with the allocated buffer.

    This pass must run after TLXStorageAliasSizeDefinition to ensure that
    all storage alias specifications have their sizes set.

    The pass performs the following transformations:
    1. Creates a LocalAllocOp/TMEMAllocOp for each storage_alias_spec
    2. Replaces storage_alias_local_alloc with local_alias referencing the allocation
    3. Erases the original storage_alias_spec operations
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TLXStorageAliasLowering : Pass<"tlx-storage-alias-lowering", "mlir::ModuleOp"> {
  let summary = "Lower storage alias operations (combined size definition and allocation)";

  let description = [{
    This pass combines TLXStorageAliasSizeDefinition and TLXStorageAliasAllocation
    into a single pass for convenience. It performs all storage alias lowering
    in one step:

    1. Computes or validates storage alias sizes (TLXStorageAliasSizeDefinition)
    2. Materializes storage alias allocations (TLXStorageAliasAllocation)

    The individual passes (TLXStorageAliasSizeDefinition and TLXStorageAliasAllocation)
    remain available for testing and debugging purposes.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

#endif // TRITON_TLX_PASSES
