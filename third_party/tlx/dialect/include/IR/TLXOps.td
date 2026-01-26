
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
include "TLXInterfaces.td"
include "TLXTypes.td"


class TLX_Op<string mnemonic, list<Trait> traits = []> :
    Op<TLX_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Storage Alias Spec Operation
//===----------------------------------------------------------------------===//

def TLX_StorageAliasSpecOp : TLX_Op<"storage_alias_spec", [Pure]> {
  let summary = "Define a storage alias specification";

  let description = [{
    Creates a storage alias specification that can be referenced by multiple
    `local_alloc` operations. This operation does not allocate memory itself;
    it defines a logical grouping for buffer sharing.

    The actual memory allocation is deferred until `local_alloc` operations
    reference this storage alias spec. The compiler will:
    - If `buffer_size_bytes` is specified: verify all references fit within
      the specified size.
    - Otherwise: compute the size as the maximum of all referencing allocations.

    Note: Only smem and tmem storage kinds are supported. smemCluster is not
    allowed.

    Example:
    ```mlir
    %alias = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %alias_sized = tlx.storage_alias_spec storage = tmem, size = 16384 : !tlx.storage_alias_spec<tmem, 16384>
    ```
  }];

  let arguments = (ins
    TLX_StorageKindAttr:$storage,
    OptionalAttr<I64Attr>:$buffer_size_bytes
  );

  let results = (outs TLX_StorageAliasSpecType:$result);

  // Use qualified() otherwise "!tlx.storage_alias_spec<X>" is printed as "<X>".
  let assemblyFormat = [{
    `storage` `=` $storage
    (`,` `size` `=` $buffer_size_bytes^)?
    attr-dict `:` qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Layout Operations
//===----------------------------------------------------------------------===//

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

def TLX_LocalAliasOp : TLX_Op<"local_alias",
                                 [SameOperandAndResultMemorySpace,
                                  Pure]> {
  let summary = "Create an alias of a local memory buffer";

  let arguments = (ins TTG_TensorOrMemDesc:$src);

  let results = (outs TTG_TensorOrMemDesc:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif
