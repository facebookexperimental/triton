
#ifndef TLX_OPS
#define TLX_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/ControlFlowInterfaces.td" // RegionBranchOpInterface
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td"  // SameOperandsAndResultType
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "TLXDialect.td"


class TLX_Op<string mnemonic, list<Trait> traits = []> :
    Op<TLX_Dialect, mnemonic, traits>;

def TLX_RequireLayoutOp : TLX_Op<"require_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType,
                                  Pure]> {
  let summary = "require specific layout for a local memory buffer";

  let arguments = (ins TTG_TensorOrMemDesc:$src);

  let results = (outs TTG_TensorOrMemDesc:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

  let hasFolder = 1;
}

def TLX_ReleaseLayoutOp : TLX_Op<"release_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType,
                                  Pure]> {
  let summary = "release specific layout for a register buffer";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif
