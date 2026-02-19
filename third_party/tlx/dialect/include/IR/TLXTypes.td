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
// Reuse Group Kind Enum
//===----------------------------------------------------------------------===//

def TLX_ReuseGroupKind_Shared : I32EnumAttrCase<"shared", 0, "shared">;
def TLX_ReuseGroupKind_Distinct : I32EnumAttrCase<"distinct", 1, "distinct">;

def TLX_ReuseGroupKindAttr : I32EnumAttr<
    "ReuseGroupKind", "TLX reuse group kind for buffer overlap definitions",
    [TLX_ReuseGroupKind_Shared, TLX_ReuseGroupKind_Distinct]> {
  let cppNamespace = "::mlir::triton::tlx";
  let description = [{
    Defines the relationship between elements in a reuse group:

    - **shared**: Elements must logically occupy the same region in memory.
      There is no cross-index overlap, and elements share the memory at each
      buffer index. Useful when buffers are used at different times.
    - **distinct**: Elements must be placed into non-overlapping regions of
      memory. Elements can be accessed simultaneously without conflicts.
  }];
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

//===----------------------------------------------------------------------===//
// Reuse Group Type
//===----------------------------------------------------------------------===//

def TLX_ReuseGroupType : TLXTypeDef<"ReuseGroup", "reuse_group", []> {
  let summary = "A reuse group type for buffer overlap definitions";

  let description = [{
    Represents a reuse group that defines buffer overlap relationships for
    shared memory allocations. A reuse group organizes multiple buffers
    (or nested groups) with a specific relationship type:

    - **shared**: Elements logically occupy the same memory region at each
      buffer index. Useful when buffers are used at different times.
    - **distinct**: Elements must be in non-overlapping memory regions.
      Useful when buffers need to be accessed simultaneously.

    The reuse group forms a tree structure where leaf nodes are memory
    allocations and internal nodes are nested reuse groups.

    Constraints:
    - All elements must have the same buffer count (num).
    - All elements must use the same storage kind (smem or tmem).
      The storage kind is inferred from the elements and not stored in the type.

    Example:
    ```mlir
    // A and B share the same memory (used at different times)
    %group = tlx.reuse_group(%a, %b) {group_type = shared}
             : (!ttg.memdesc<...>, !ttg.memdesc<...>) -> !tlx.reuse_group<shared, 2>

    // Nested groups for complex sharing schemes
    %inner = tlx.reuse_group(%c, %d, %e) {group_type = distinct}
             : (...) -> !tlx.reuse_group<distinct, 2>
    %outer = tlx.reuse_group(%a, %inner) {group_type = shared}
             : (...) -> !tlx.reuse_group<shared, 2>
    ```
  }];

  let parameters = (ins
    EnumParameter<TLX_ReuseGroupKindAttr>:$groupKind
  );

  let assemblyFormat = "`<` $groupKind `>`";

  let genVerifyDecl = 1;
}

#endif // TLX_TYPES
