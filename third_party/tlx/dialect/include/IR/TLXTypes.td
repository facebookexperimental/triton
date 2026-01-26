#ifndef TLX_TYPES
#define TLX_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "TLXDialect.td"

//===----------------------------------------------------------------------===//
// Storage Kind Enum
//===----------------------------------------------------------------------===//

def TLX_StorageKind_SMEM : I32EnumAttrCase<"smem", 0, "smem">;
def TLX_StorageKind_TMEM : I32EnumAttrCase<"tmem", 1, "tmem">;

def TLX_StorageKindAttr : I32EnumAttr<
    "StorageKind", "TLX storage kind for shared buffers",
    [TLX_StorageKind_SMEM, TLX_StorageKind_TMEM]> {
  let cppNamespace = "::mlir::triton::tlx";
}

//===----------------------------------------------------------------------===//
// TLX Type Base Class
//===----------------------------------------------------------------------===//

class TLXTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TLX_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

//===----------------------------------------------------------------------===//
// Storage Alias Spec Type
//===----------------------------------------------------------------------===//

def TLX_StorageAliasSpecType : TLXTypeDef<"StorageAliasSpec", "storage_alias_spec", []> {
  let summary = "A storage alias specification type";

  let description = [{
    Represents a storage alias specification that can be referenced by multiple
    local memory allocations. This type carries the storage kind and
    optional explicit size.

    This type is used by the `storage_alias_spec` operation to define a
    logical grouping for buffer sharing. Multiple `local_alloc` operations
    can reference the same storage alias specification via the `reuse` parameter.

    The actual memory allocation is deferred until `local_alloc` operations
    reference this storage alias spec. The compiler will:
    - If `bufferSizeBytes` is specified: verify all references fit within
      the specified size.
    - Otherwise: compute the size as the maximum of all referencing allocations.

    Note: Only smem and tmem storage kinds are supported. smemCluster is
    not allowed for storage alias specifications.

    Example:
    ```mlir
    %alias = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %alias_sized = tlx.storage_alias_spec storage = tmem, size = 16384 : !tlx.storage_alias_spec<tmem, 16384>
    ```
  }];

  let parameters = (ins
    EnumParameter<TLX_StorageKindAttr>:$storage,
    OptionalParameter<"std::optional<int64_t>">:$bufferSizeBytes
  );

  let assemblyFormat = "`<` $storage (`,` $bufferSizeBytes^)? `>`";

  let genVerifyDecl = 1;
}

#endif // TLX_TYPES
