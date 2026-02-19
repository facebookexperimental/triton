# [WIP] TLX `storage_alias_spec` Design Proposal

**Author:** Nick Riasanovsky
**Updated:** 2026-02-07
**Status:** Draft - Phase 1, Phase 2, Phase 3, Phase 4 & Phase 5

---

## Background

The current TLX buffer reuse mechanism relies on passing an existing `buffered_tensor` to the `reuse` parameter of `local_alloc`. While functional, this approach has several limitations:

1. **Unclear ownership semantics**: The first buffer appears to "own" the underlying memory, creating confusion about which buffer is primary.
2. **Ordering dependencies**: Users must ensure the "owner" buffer is allocated before any reuse buffers.
3. **Manual size calculations**: When dtypes differ, users must manually calculate padding to ensure proper overlap.
4. **Complex autotuning**: When buffer parameters (shapes, dtypes, counts) are autotuned, the reuse relationship becomes fragile.

This design introduces `storage_alias_spec`, a new abstraction that explicitly represents shared buffer ownership, making the allocation semantics clearer and more robust. The name "storage alias spec" (storage alias specification) emphasizes that this object specifies how multiple allocations can alias the same storage region rather than performing an allocation itself.

> **Note:** This is Phase 1 of the broader buffer reuse refactoring. This phase focuses solely on defining the `storage_alias_spec` class and its IR lowering. Integration with `local_alloc` and the `set_buffer_overlap` API will be addressed in subsequent phases.

---

## Design Principles

Following TLX design principles:

- **Easy things stay easy**: The new API should not add unnecessary verbosity for common use cases.
- **The user has control**: Users can still fall back to explicit offset calculations when needed.
- **Backward compatibility**: Existing code using `reuse=buffered_tensor` continues to work during the deprecation period.

---

## Scope

This design covers **only**:
- The `storage_alias_spec` Python class definition
- The corresponding MLIR type and operation
- Lowering from Python to MLIR IR

**Out of scope** for this phase:
- Changes to `local_alloc` to accept `storage_alias_spec`
- `set_buffer_overlap` API
- Compiler passes that consume `storage_alias_spec`

---

## API Design

### `storage_alias_spec` Class Definition

```python
class storage_alias_spec:
    """
    Definition of a storage alias specification.

    This class represents ownership of an underlying memory buffer that can be
    shared by multiple `local_alloc` calls. It can be either unsized or sized:

    - **Unsized (default)**: The compiler sets the buffer size to accommodate
      the largest allocation that references it.
    - **Sized**: The user specifies an explicit size, and the compiler verifies
      all referencing allocations fit within it.

    All attributes are immutable after construction.

    Attributes:
        buffer_size_bytes: Optional explicit size in bytes. Must be a compile-time
            constant if provided. Immutable after construction.
        storage: The storage kind (smem or tmem) for this buffer.

    Note:
        smemCluster storage is not supported for storage alias specifications.

    Example:
        # Create an unsized storage alias spec (size determined by largest user)
        alias_spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

        # Create a sized storage alias spec with explicit padding
        alias_spec = tlx.storage_alias_spec(
            buffer_size_bytes=16384,
            storage=tlx.storage_kind.tmem
        )
    """

    def __init__(
        self,
        storage: tlx.storage_kind = tlx.storage_kind.smem,
        buffer_size_bytes: Optional[tl.constexpr[int]] = None,
    ):
        """
        Initialize a storage alias specification.

        Args:
            storage: The storage kind for this buffer. Must be smem or tmem.
                All local_alloc calls that reference this storage_alias_spec
                must use the same storage kind. smemCluster is not supported.
            buffer_size_bytes: Optional explicit size in bytes. If provided,
                must be a compile-time constant. The compiler will verify that
                all referencing allocations fit within this size. This value
                is immutable after construction.

        Raises:
            ValueError: If buffer_size_bytes is provided but is not a
                compile-time constant.
            ValueError: If storage is smemCluster (not supported).
        """
        if storage == tlx.storage_kind.smemCluster:
            raise ValueError("smemCluster storage is not supported for storage_alias_spec")
        self._storage = storage
        self._buffer_size_bytes = buffer_size_bytes
        self._handle = None  # IR handle, set during lowering

    @property
    def storage(self) -> tlx.storage_kind:
        """The storage kind for this buffer (read-only)."""
        return self._storage

    @property
    def buffer_size_bytes(self) -> Optional[int]:
        """The explicit buffer size in bytes, or None if unsized (read-only)."""
        return self._buffer_size_bytes
```

### Key Characteristics

1. **`storage_alias_spec` is not indexable**: It represents the memory region definition, not a usable buffer.
2. **No primary owner**: All allocations referencing the same `storage_alias_spec` are equal peers.
3. **Automatic size calculation**: The compiler determines the required buffer size as the maximum of all users (unless explicitly specified).
4. **Storage type enforcement**: All allocations sharing a buffer must have the same storage kind (smem or tmem only).
5. **Immutability**: All properties (`storage`, `buffer_size_bytes`) are immutable after construction.

---

## IR Design

### New MLIR Type: `TLX_StorageAliasSpecType`

A new MLIR type to represent the storage alias specification:

```tablegen
def TLX_StorageAliasSpecType : TLX_Type<"StorageAliasSpec", "storage_alias_spec"> {
  let summary = "A storage alias specification";
  let description = [{
    Represents a storage alias specification that can be referenced by multiple
    local memory allocations. This type carries the storage kind and
    optional explicit size.

    Note: Only smem and tmem storage kinds are supported. smemCluster is
    not allowed for storage alias specifications.
  }];

  let parameters = (ins
    "::mlir::triton::tlx::StorageKind":$storage,
    "std::optional<int64_t>":$bufferSizeBytes
  );

  let assemblyFormat = "`<` $storage (`,` $bufferSizeBytes^)? `>`";

  let genVerifyDecl = 1;
}
```

#### Type Verifier

```cpp
LogicalResult StorageAliasSpecType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    StorageKind storage,
    std::optional<int64_t> bufferSizeBytes) {
  // smemCluster is not supported
  if (storage == StorageKind::smemCluster) {
    return emitError() << "smemCluster storage is not supported for storage_alias_spec";
  }
  return success();
}
```

### New MLIR Operation: `TLX_StorageAliasSpecOp`

```tablegen
def TLX_StorageAliasSpecOp : TLX_Op<"storage_alias_spec", [Pure]> {
  let summary = "Define a storage alias specification";
  let description = [{
    Creates a storage alias specification that can be referenced by multiple
    local_alloc operations. This operation does not allocate memory itself;
    it defines a logical grouping for buffer sharing.

    The actual memory allocation is deferred until local_alloc operations
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

  let assemblyFormat = [{
    `storage` `=` $storage
    (`,` `size` `=` $buffer_size_bytes^)?
    attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins "StorageKind":$storage), [{
      build($_builder, $_state, storage, /*buffer_size_bytes=*/nullptr);
    }]>,
    OpBuilder<(ins "StorageKind":$storage, "int64_t":$bufferSizeBytes), [{
      build($_builder, $_state, storage,
            $_builder.getI64IntegerAttr(bufferSizeBytes));
    }]>
  ];

  let hasVerifier = 1;
}
```

### Operation Verifier

```cpp
LogicalResult StorageAliasSpecOp::verify() {
  // Verify buffer_size_bytes is positive if specified
  if (auto size = getBufferSizeBytes()) {
    if (*size <= 0) {
      return emitOpError("buffer_size_bytes must be positive, got ") << *size;
    }
  }

  // Verify storage kind is valid for storage alias specs (smemCluster not allowed)
  auto storage = getStorage();
  if (storage == StorageKind::smemCluster) {
    return emitOpError("smemCluster storage is not supported for storage_alias_spec");
  }
  if (storage != StorageKind::smem && storage != StorageKind::tmem) {
    return emitOpError("unsupported storage kind for storage_alias_spec, "
                       "expected smem or tmem");
  }

  return success();
}
```

---

## Python to MLIR Lowering

### Frontend Implementation

Location: `/data/users/njriasan/fbsource/third-party/triton/beta/triton/third_party/tlx/language/tlx/mem_ops.py`

```python
@tl.builtin
def storage_alias_spec(
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    buffer_size_bytes: Optional[tl.constexpr[int]] = None,
    _builder=None,
) -> tlx.storage_alias_spec:
    """
    Create a storage alias specification.

    Args:
        storage: The storage kind (smem or tmem). smemCluster is not supported.
        buffer_size_bytes: Optional explicit size in bytes (immutable after creation).
        _builder: Internal builder parameter.

    Returns:
        A storage_alias_spec object that can be passed to local_alloc.

    Raises:
        ValueError: If storage is not a valid storage_kind.
        ValueError: If storage is smemCluster (not supported).
        ValueError: If buffer_size_bytes is not a compile-time constant.
        ValueError: If buffer_size_bytes is not positive.
    """
    # Validate storage kind
    if not isinstance(storage, tlx.storage_kind):
        raise ValueError(f"storage must be a tlx.storage_kind, got {type(storage)}")

    # smemCluster is not supported
    if storage == tlx.storage_kind.smemCluster:
        raise ValueError("smemCluster storage is not supported for storage_alias_spec")

    # Validate buffer_size_bytes if provided
    if buffer_size_bytes is not None:
        if not isinstance(buffer_size_bytes, tl.constexpr):
            raise ValueError("buffer_size_bytes must be a compile-time constant")
        if buffer_size_bytes.value <= 0:
            raise ValueError(f"buffer_size_bytes must be positive, got {buffer_size_bytes.value}")

    # Create IR operation
    handle = _builder.create_storage_alias_spec(
        storage=storage,
        buffer_size_bytes=buffer_size_bytes.value if buffer_size_bytes else None,
    )

    # Return wrapper object (immutable)
    return tlx.storage_alias_spec(
        handle=handle,
        storage=storage,
        buffer_size_bytes=buffer_size_bytes,
    )
```

### Type Wrapper Class

Location: `/data/users/njriasan/fbsource/third-party/triton/beta/triton/third_party/tlx/language/tlx/types.py`

```python
class storage_alias_spec(tl.base_value):
    """
    Runtime wrapper for a storage alias specification.

    This class wraps the IR handle for a storage_alias_spec operation
    and provides the interface for passing to local_alloc.

    All properties are immutable after construction.
    """

    def __init__(
        self,
        handle,
        storage: storage_kind,
        buffer_size_bytes: Optional[int] = None,
    ):
        super().__init__(handle)
        self._handle = handle
        self._storage = storage
        self._buffer_size_bytes = buffer_size_bytes

    @property
    def handle(self):
        """The IR handle (read-only)."""
        return self._handle

    @property
    def storage(self) -> storage_kind:
        """The storage kind (read-only)."""
        return self._storage

    @property
    def buffer_size_bytes(self) -> Optional[int]:
        """The explicit buffer size in bytes, or None (read-only)."""
        return self._buffer_size_bytes

    def __repr__(self):
        size_str = f", size={self._buffer_size_bytes}" if self._buffer_size_bytes else ""
        return f"storage_alias_spec(storage={self._storage.value}{size_str})"
```

### C++ Builder Implementation

Location: `/data/users/njriasan/fbsource/third-party/triton/beta/triton/third_party/tlx/dialect/triton_tlx.cc`

```cpp
mlir::Value TritonTLXOpBuilder::create_storage_alias_spec(
    tlx::StorageKind storage,
    std::optional<int64_t> bufferSizeBytes) {

  // Validate storage kind (smemCluster not allowed)
  if (storage == tlx::StorageKind::smemCluster) {
    llvm::report_fatal_error(
        "smemCluster storage is not supported for storage_alias_spec");
  }

  // Create the result type
  auto resultType = tlx::StorageAliasSpecType::get(
      context, storage, bufferSizeBytes);

  // Create the operation
  if (bufferSizeBytes) {
    return builder.create<tlx::StorageAliasSpecOp>(
        loc, resultType, storage, *bufferSizeBytes);
  } else {
    return builder.create<tlx::StorageAliasSpecOp>(
        loc, resultType, storage);
  }
}
```

### Python Bindings

Location: `/data/users/njriasan/fbsource/third-party/triton/beta/triton/third_party/tlx/dialect/triton_tlx.cc` (pybind section)

```cpp
// In the pybind11 module definition
m.def("create_storage_alias_spec",
    [](TritonTLXOpBuilder &self,
       const std::string &storage,
       std::optional<int64_t> bufferSizeBytes) {
      auto storageKind = parseStorageKind(storage);
      if (storageKind == tlx::StorageKind::smemCluster) {
        throw std::invalid_argument(
            "smemCluster storage is not supported for storage_alias_spec");
      }
      return self.create_storage_alias_spec(storageKind, bufferSizeBytes);
    },
    py::arg("storage"),
    py::arg("buffer_size_bytes") = py::none(),
    "Create a storage alias specification");
```

---

## Example IR Output

### Unsized Storage Alias Spec

Python:
```python
alias_spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
```

MLIR:
```mlir
%0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
```

### Sized Storage Alias Spec

Python:
```python
alias_spec = tlx.storage_alias_spec(
    storage=tlx.storage_kind.tmem,
    buffer_size_bytes=16384,
)
```

MLIR:
```mlir
%0 = tlx.storage_alias_spec storage = tmem, size = 16384 : !tlx.storage_alias_spec<tmem, 16384>
```

### Invalid: smemCluster (Rejected)

Python:
```python
# This raises ValueError at JIT compile time
alias_spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smemCluster)
# ValueError: smemCluster storage is not supported for storage_alias_spec
```

---

## Testing Plan

### Unit Tests

1. **Python API tests**
   - Create unsized `storage_alias_spec`
   - Create sized `storage_alias_spec`
   - Verify storage kind validation
   - Verify `buffer_size_bytes` must be positive
   - Verify `buffer_size_bytes` must be constexpr
   - Verify smemCluster is rejected
   - Verify immutability (properties cannot be reassigned)

2. **IR lowering tests**
   - Verify correct MLIR operation is generated
   - Verify attributes are correctly set
   - Verify type is correctly constructed

3. **Operation verifier tests**
   - Negative `buffer_size_bytes` should fail
   - Zero `buffer_size_bytes` should fail
   - smemCluster storage should fail

### Example Test Cases

```python
def test_storage_alias_spec_unsized():
    @triton.jit
    def kernel():
        alias_spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        # Verify IR contains: tlx.storage_alias_spec storage = smem

def test_storage_alias_spec_sized():
    @triton.jit
    def kernel():
        alias_spec = tlx.storage_alias_spec(
            storage=tlx.storage_kind.tmem,
            buffer_size_bytes=tl.constexpr(8192),
        )
        # Verify IR contains: tlx.storage_alias_spec storage = tmem, size = 8192

def test_storage_alias_spec_invalid_size():
    @triton.jit
    def kernel():
        # Should raise ValueError
        alias_spec = tlx.storage_alias_spec(
            storage=tlx.storage_kind.smem,
            buffer_size_bytes=tl.constexpr(-100),
        )

def test_storage_alias_spec_smem_cluster_rejected():
    @triton.jit
    def kernel():
        # Should raise ValueError
        alias_spec = tlx.storage_alias_spec(
            storage=tlx.storage_kind.smemCluster,
        )

def test_storage_alias_spec_immutable():
    @triton.jit
    def kernel():
        alias_spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        # Should raise AttributeError - properties are read-only
        alias_spec.storage = tlx.storage_kind.tmem  # Error!
        alias_spec.buffer_size_bytes = 1024  # Error!
```

---

## Design Decisions

1. **Naming**: The class is named `shared_buffer_def` (definition) to emphasize that this object defines a shared memory region rather than performing an allocation itself.

2. **Storage validation**: `smemCluster` storage is **not supported** for `shared_buffer_def`. The semantics of sharing across cluster shared memory would require additional design consideration for distributed memory access patterns.

3. **Immutability**: All properties (`buffer_size_bytes`, `storage`) are **immutable** after construction. This ensures consistency and prevents accidental modifications that could lead to subtle bugs.

---
---

# Phase 2: `local_alloc` Integration and Compiler Passes

This phase extends Phase 1 by:
1. Updating `local_alloc` to accept `shared_buffer_def` via the `reuse` parameter
2. Implementing a compiler pass to determine and validate buffer sizes
3. Implementing a compiler pass to eliminate shared buffer definitions and materialize actual allocations

---

## Scope

This phase covers:
- Modifications to `local_alloc` to accept `shared_buffer_def` in the `reuse` parameter
- New MLIR operation `tlx.shared_buffer_alloc` to represent the actual allocation
- Compiler pass: `SharedBufferSizeDefinitionPass` - computes or validates buffer sizes
- Compiler pass: `SharedBufferAllocationPass` - eliminates `shared_buffer_def` and materializes allocations

**Out of scope** for this phase:
- `set_buffer_overlap` API (Phase 4)
- Buffer offset calculation based on overlap schemes (Phase 4)
- Buffer reuse analysis warnings (Phase 5)

---

## API Design

### Updated `local_alloc` Signature

The `reuse` parameter will be updated to accept either a `buffered_tensor` (existing behavior) or a `shared_buffer_def` (new behavior):

```python
@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    # Updated type definition: now accepts shared_buffer_def
    reuse: Optional[Union[tlx.buffered_tensor, tlx.shared_buffer_def]] = None,
    layout: Optional[tlx.shared_layout_encoding] = None,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Allocates buffer in shared memory and returns a view of the buffer.

    Args:
        shape: Shape of each buffer (excluding the num dimension).
        dtype: Data type of the buffer elements.
        num: Number of buffers to allocate (compile-time constant).
        storage: Storage kind (smem or tmem).
        reuse: Optional buffer reuse specification:
            - buffered_tensor: Reuse an existing buffer's memory (legacy).
            - shared_buffer_def: Reference a shared buffer definition.
        layout: Optional memory layout encoding.

    Returns:
        A buffered_tensor representing the allocated buffers.

    Raises:
        ValueError: If reuse storage kind doesn't match the specified storage.
    """
```

### Semantic Changes

When `reuse` is a `shared_buffer_def`:
1. The `storage` parameter is validated against the `shared_buffer_def.storage`
2. The `local_alloc` operation references the shared buffer definition
3. The compiler will later compute offsets and materialize the actual allocation

When `reuse` is a `buffered_tensor`:
1. Existing behavior is preserved for backward compatibility
2. The new buffer aliases the existing buffer's memory at offset 0

---

## IR Design

### New Operation: `TLX_SharedBufferLocalAllocOp`

This operation represents an allocation that references a shared buffer definition. It is produced by the Python frontend when `local_alloc` is called with a `shared_buffer_def` in the `reuse` parameter:

```tablegen
def TLX_SharedBufferLocalAllocOp : TLX_Op<"shared_buffer_local_alloc", [Pure]> {
  let summary = "Allocate local memory referencing a shared buffer definition";
  let description = [{
    Allocates local memory (shared memory or tensor memory) that references
    a shared buffer definition. Multiple allocations can reference the same
    shared buffer definition, and the compiler will:
    1. Compute the required buffer size (or validate the explicit size)
    2. Assign offsets to each allocation
    3. Materialize the actual memory allocation

    After the SharedBufferAllocationPass runs, this operation is replaced with
    a LocalAliasOp (or MemDescSubviewOp for non-zero offsets in Phase 3)
    pointing to a standard LocalAllocOp/TMEMAllocOp.
  }];

  let arguments = (ins TLX_SharedBufferDefType:$shared_buffer);
  let results = (outs TTG_MemDescType:$result);

  let assemblyFormat = [{
    $shared_buffer attr-dict `:`
    qualified(type($shared_buffer)) `->` type($result)
  }];

  let hasVerifier = 1;
}
```

### Design Decision: Reusing Existing Allocation Operations

The design deliberately **does not introduce a new allocation operation** (e.g., `shared_buffer_alloc`). Instead, the compiler passes transform the shared buffer operations into existing TritonGPU operations:

- **`LocalAllocOp`** (ttg.local_alloc) - For shared memory allocations
- **`TMEMAllocOp`** (ttng.tmem_alloc) - For tensor memory allocations
- **`LocalAliasOp`** (tlx.local_alias) - To reinterpret the allocation with the correct type
- **`MemDescSubviewOp`** (ttg.memdesc_subview) - For non-zero offsets (Phase 3)

This approach:
1. **Reduces complexity** - No new allocation operations to maintain
2. **Reuses existing infrastructure** - Leverages well-tested allocation and lowering paths
3. **Enables future extensibility** - Subviews can be used for Phase 3 offset support

#### Transformation Example

Before passes:
```mlir
%shared = tlx.shared_buffer_def storage = smem : !tlx.shared_buffer_def<smem>
%a = tlx.shared_buffer_local_alloc %shared : ... -> !ttg.memdesc<2x64x64xf32, ...>
%b = tlx.shared_buffer_local_alloc %shared : ... -> !ttg.memdesc<2x64x64xbf16, ...>
```

After `SharedBufferAllocationPass`:
```mlir
// Single allocation with computed max size (32768 bytes)
%alloc = ttg.local_alloc : !ttg.memdesc<32768xi8, #shared, #smem, mutable>

// Allocations become aliases pointing to the shared allocation
%a = tlx.local_alias %alloc : !ttg.memdesc<32768xi8, ...> -> !ttg.memdesc<2x64x64xf32, ...>
%b = tlx.local_alias %alloc : !ttg.memdesc<32768xi8, ...> -> !ttg.memdesc<2x64x64xbf16, ...>
```

---

## Compiler Passes

### Pass 1: `SharedBufferSizeDefinitionPass`

**Purpose**: For each `shared_buffer_def`, compute its required size as the maximum of all referencing allocations, or validate that an explicit size is sufficient.

**Location**: Run early in the pipeline, after type resolution but before layout assignment.

#### Algorithm

```
For each shared_buffer_def D in the module:
    1. Collect all local_alloc operations that reference D
    2. For each local_alloc L referencing D:
        a. Compute required_size(L) = element_size(L) * product(shape(L)) * num(L)
    3. max_required_size = max(required_size(L) for all L)
    4. If D has explicit buffer_size_bytes:
        a. If buffer_size_bytes < max_required_size:
            - Emit error: "shared_buffer_def size {buffer_size_bytes} is too small,
              requires at least {max_required_size} bytes"
        b. Else: use buffer_size_bytes (allows for explicit padding)
    5. Else (unsized):
        a. Set D.buffer_size_bytes = max_required_size
```

#### Implementation

```cpp
class SharedBufferSizeDefinitionPass
    : public PassWrapper<SharedBufferSizeDefinitionPass,
                         OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override {
    return "tlx-shared-buffer-size-definition";
  }

  StringRef getDescription() const override {
    return "Compute or validate shared buffer sizes";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Map from shared_buffer_def SSA value to list of referencing local_allocs
    DenseMap<Value, SmallVector<tlx::LocalAllocOp>> sharedBufferUsers;

    // Collect all local_alloc operations that reference shared buffers
    module.walk([&](tlx::LocalAllocOp allocOp) {
      if (Value sharedBuffer = allocOp.getSharedBuffer()) {
        sharedBufferUsers[sharedBuffer].push_back(allocOp);
      }
    });

    // Process each shared_buffer_def
    module.walk([&](tlx::SharedBufferDefOp defOp) {
      Value defValue = defOp.getResult();
      auto &users = sharedBufferUsers[defValue];

      if (users.empty()) {
        // Warn: shared_buffer_def has no users
        defOp.emitWarning("shared_buffer_def has no referencing local_alloc operations");
        return;
      }

      // Compute max required size
      int64_t maxRequiredSize = 0;
      for (auto allocOp : users) {
        int64_t size = computeAllocationSize(allocOp);
        maxRequiredSize = std::max(maxRequiredSize, size);
      }

      // Validate or set the size
      if (auto explicitSize = defOp.getBufferSizeBytes()) {
        if (*explicitSize < maxRequiredSize) {
          defOp.emitError()
              << "shared_buffer_def size " << *explicitSize
              << " is too small, requires at least " << maxRequiredSize
              << " bytes";
          signalPassFailure();
          return;
        }
        // Explicit size is sufficient, keep it (allows padding)
      } else {
        // Set computed size on the operation
        defOp.setBufferSizeBytesAttr(
            IntegerAttr::get(IndexType::get(getContext()), maxRequiredSize));
      }
    });
  }

private:
  int64_t computeAllocationSize(tlx::LocalAllocOp allocOp) {
    // Get element type size
    Type elemType = allocOp.getType()
                        .cast<RankedTensorType>()
                        .getElementType();
    int64_t elemSize = getElementSizeInBytes(elemType);

    // Get shape including num dimension
    ArrayRef<int64_t> shape = allocOp.getType()
                                  .cast<RankedTensorType>()
                                  .getShape();

    // Compute total size
    int64_t totalElements = 1;
    for (int64_t dim : shape) {
      totalElements *= dim;
    }

    return totalElements * elemSize;
  }

  int64_t getElementSizeInBytes(Type type) {
    if (auto floatType = type.dyn_cast<FloatType>()) {
      return floatType.getWidth() / 8;
    }
    if (auto intType = type.dyn_cast<IntegerType>()) {
      return (intType.getWidth() + 7) / 8;
    }
    llvm_unreachable("unsupported element type");
  }
};
```

#### Registration

```cpp
void registerSharedBufferSizeDefinitionPass() {
  PassRegistration<SharedBufferSizeDefinitionPass>();
}
```

---

### Pass 2: `SharedBufferAllocationPass`

**Purpose**: Eliminate `shared_buffer_def` operations by materializing actual memory allocations, and update all referencing `local_alloc` operations to use the allocated buffer.

**Location**: Run after `SharedBufferSizeDefinitionPass` and after layout assignment.

#### Algorithm

```
For each shared_buffer_def D in the module:
    1. Get the computed/validated buffer_size_bytes from D
    2. Create a new shared_buffer_alloc operation with:
        - storage = D.storage
        - buffer_size_bytes = D.buffer_size_bytes
    3. For each local_alloc L referencing D:
        a. Update L to alias the shared_buffer_alloc result
        b. Remove the shared_buffer reference from L
    4. Erase the shared_buffer_def D (it's now replaced)
```

#### Implementation

```cpp
class SharedBufferAllocationPass
    : public PassWrapper<SharedBufferAllocationPass,
                         OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override {
    return "tlx-shared-buffer-allocation";
  }

  StringRef getDescription() const override {
    return "Materialize shared buffer allocations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Map from shared_buffer_def to its materialized allocation
    DenseMap<Value, Value> defToAlloc;

    // First pass: create shared_buffer_alloc for each shared_buffer_def
    module.walk([&](tlx::SharedBufferDefOp defOp) {
      builder.setInsertionPoint(defOp);

      auto bufferSizeBytes = defOp.getBufferSizeBytes();
      if (!bufferSizeBytes) {
        defOp.emitError("shared_buffer_def has no size set; "
                        "run SharedBufferSizeDefinitionPass first");
        signalPassFailure();
        return;
      }

      // Determine the tensor type for the allocation
      // This is a 1D byte buffer
      auto elemType = IntegerType::get(getContext(), 8);
      auto tensorType = RankedTensorType::get({*bufferSizeBytes}, elemType);

      // Create the shared_buffer_alloc operation
      auto allocOp = builder.create<tlx::SharedBufferAllocOp>(
          defOp.getLoc(),
          tensorType,
          defOp.getStorage(),
          *bufferSizeBytes);

      defToAlloc[defOp.getResult()] = allocOp.getResult();
    });

    // Second pass: update local_alloc operations to reference the allocation
    module.walk([&](tlx::LocalAllocOp allocOp) {
      if (Value sharedBuffer = allocOp.getSharedBuffer()) {
        auto it = defToAlloc.find(sharedBuffer);
        if (it == defToAlloc.end()) {
          allocOp.emitError("shared_buffer_def not found");
          signalPassFailure();
          return;
        }

        // Update the local_alloc to alias the shared buffer allocation
        // The offset is 0 for now (overlap scheme will set offsets in Phase 4)
        allocOp.setAliasAttr(builder.getIndexAttr(0));
        allocOp.getSharedBufferMutable().clear();

        // Add the alias operand
        allocOp.getAliasMutable().assign(it->second);
      }
    });

    // Third pass: erase shared_buffer_def operations
    SmallVector<tlx::SharedBufferDefOp> toErase;
    module.walk([&](tlx::SharedBufferDefOp defOp) {
      toErase.push_back(defOp);
    });
    for (auto defOp : toErase) {
      defOp.erase();
    }
  }
};
```

#### Registration

```cpp
void registerSharedBufferAllocationPass() {
  PassRegistration<SharedBufferAllocationPass>();
}
```

---

## Pass Pipeline Integration

The passes should be registered in the Triton/TLX pass pipeline:

```cpp
void addTLXSharedBufferPasses(OpPassManager &pm) {
  // Phase 2 passes
  pm.addPass(createSharedBufferSizeDefinitionPass());
  // ... (other passes like layout assignment) ...
  pm.addPass(createSharedBufferAllocationPass());
}
```

**Pass ordering**:
1. `SharedBufferSizeDefinitionPass` - must run after type resolution, before layout assignment
2. Layout assignment passes
3. `SharedBufferAllocationPass` - must run after size definition and layout assignment

---

## Example IR Transformation

### Input (after Phase 1 lowering)

```mlir
tt.func @kernel() {
  // Create shared buffer definition
  %shared = tlx.shared_buffer_def storage = smem : !tlx.shared_buffer_def<smem>

  // Two allocations referencing the same shared buffer
  %a = tlx.local_alloc shape = [64, 64], dtype = f32, num = 2,
                       storage = smem, shared_buffer = %shared
                       : tensor<2x64x64xf32>

  %b = tlx.local_alloc shape = [64, 64], dtype = bf16, num = 2,
                       storage = smem, shared_buffer = %shared
                       : tensor<2x64x64xbf16>

  // ... use %a and %b ...
  tt.return
}
```

### After `SharedBufferSizeDefinitionPass`

```mlir
tt.func @kernel() {
  // Size computed as max(2*64*64*4, 2*64*64*2) = 32768 bytes
  %shared = tlx.shared_buffer_def storage = smem, size = 32768
            : !tlx.shared_buffer_def<smem, 32768>

  %a = tlx.local_alloc shape = [64, 64], dtype = f32, num = 2,
                       storage = smem, shared_buffer = %shared
                       : tensor<2x64x64xf32>

  %b = tlx.local_alloc shape = [64, 64], dtype = bf16, num = 2,
                       storage = smem, shared_buffer = %shared
                       : tensor<2x64x64xbf16>

  tt.return
}
```

### After `SharedBufferAllocationPass`

```mlir
tt.func @kernel() {
  // Shared buffer definition replaced with actual allocation
  %shared_mem = tlx.shared_buffer_alloc storage = smem, size = 32768
                : tensor<32768xi8>

  // Allocations now alias the shared memory
  %a = tlx.local_alloc shape = [64, 64], dtype = f32, num = 2,
                       storage = smem, alias = %shared_mem
                       : tensor<2x64x64xf32>

  %b = tlx.local_alloc shape = [64, 64], dtype = bf16, num = 2,
                       storage = smem, alias = %shared_mem
                       : tensor<2x64x64xbf16>

  tt.return
}
```

---

## Python Frontend Changes

### Updated `local_alloc` Implementation

```python
@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    reuse: Optional[Union[tlx.buffered_tensor, tlx.shared_buffer_def]] = None,
    layout: Optional[tlx.shared_layout_encoding] = None,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Allocates buffer in shared memory and return a view of the buffer.
    """
    # ... existing validation code ...

    alias_handle = None
    shared_buffer_handle = None

    if reuse is not None:
        if isinstance(reuse, tlx.buffered_tensor):
            # Legacy behavior: direct alias
            if reuse.type.storage != storage:
                raise ValueError("reuse tensor has different storage")
            alias_handle = reuse.handle

        elif isinstance(reuse, tlx.shared_buffer_def):
            # New behavior: reference shared buffer definition
            if reuse.storage != storage:
                raise ValueError(
                    f"shared_buffer_def storage ({reuse.storage.value}) "
                    f"doesn't match local_alloc storage ({storage.value})"
                )
            shared_buffer_handle = reuse.handle

        else:
            raise ValueError(
                f"reuse must be a buffered_tensor or shared_buffer_def, "
                f"got {type(reuse)}"
            )

    # Create IR operation with appropriate handle
    if storage == tlx.storage_kind.smem:
        tensor_handle = _semantic.builder.create_local_alloc(
            full_shape,
            elem_type,
            layout_handle,
            alias_handle,
            shared_buffer_handle,  # New parameter
        )
    else:
        tensor_handle = _semantic.builder.create_tmem_local_alloc(
            full_shape,
            elem_type,
            layout_handle,
            alias_handle,
            shared_buffer_handle,  # New parameter
        )

    # ... rest of implementation ...
```

---

## Testing Plan

### Unit Tests

1. **Size computation tests**
   - Single allocation referencing shared buffer
   - Multiple allocations with same size
   - Multiple allocations with different sizes (max wins)
   - Different dtypes affecting size calculation

2. **Size validation tests**
   - Explicit size that is sufficient
   - Explicit size that is too small (error)
   - Explicit size with padding (allowed)

3. **Allocation pass tests**
   - Single shared buffer replaced correctly
   - Multiple shared buffers in same function
   - Allocations updated to alias correctly

4. **Integration tests**
   - End-to-end from Python to final IR
   - Verify memory layout is correct
   - Verify barriers work correctly with shared buffers

### Example Test Cases

```python
def test_shared_buffer_size_computed():
    """Test that unsized shared buffer gets correct size."""
    @triton.jit
    def kernel():
        shared = tlx.shared_buffer_def(storage=tlx.storage_kind.smem)

        # 2 * 64 * 64 * 4 = 32768 bytes
        a = tlx.local_alloc((64, 64), tl.float32, 2,
                            tlx.storage_kind.smem, reuse=shared)
        # 2 * 64 * 64 * 2 = 16384 bytes
        b = tlx.local_alloc((64, 64), tl.bfloat16, 2,
                            tlx.storage_kind.smem, reuse=shared)

    # Verify IR shows size = 32768 after size definition pass


def test_shared_buffer_size_validated():
    """Test that explicit size is validated."""
    @triton.jit
    def kernel():
        # Explicit size of 16384, but allocation needs 32768
        shared = tlx.shared_buffer_def(
            storage=tlx.storage_kind.smem,
            buffer_size_bytes=16384
        )

        # This should fail: needs 32768 bytes
        a = tlx.local_alloc((64, 64), tl.float32, 2,
                            tlx.storage_kind.smem, reuse=shared)

    # Verify compilation error is raised


def test_shared_buffer_with_padding():
    """Test that explicit size with padding is allowed."""
    @triton.jit
    def kernel():
        # Explicit size of 65536 (more than needed)
        shared = tlx.shared_buffer_def(
            storage=tlx.storage_kind.smem,
            buffer_size_bytes=65536
        )

        # Only needs 32768 bytes, but 65536 is fine
        a = tlx.local_alloc((64, 64), tl.float32, 2,
                            tlx.storage_kind.smem, reuse=shared)

    # Verify compilation succeeds with size = 65536


def test_storage_mismatch_error():
    """Test that storage mismatch raises error."""
    @triton.jit
    def kernel():
        shared = tlx.shared_buffer_def(storage=tlx.storage_kind.smem)

        # Error: tmem doesn't match smem
        a = tlx.local_alloc((64, 64), tl.float32, 2,
                            tlx.storage_kind.tmem, reuse=shared)

    # Verify ValueError is raised
```

---
---

# Phase 3: Reuse Group IR API

This phase defines the IR interface for reuse groups, which are used to express buffer overlap relationships. Reuse groups organize multiple buffers (or nested groups) into a tree structure that defines how buffers share or are distinct from each other in memory. Reuse groups support both shared memory (smem) and tensor memory (tmem) allocations.

---

## Scope

This phase covers:
- New enum: `ReuseGroupKind` (`shared`, `distinct`)
- New MLIR type: `ReuseGroupType`
- New MLIR operation: `ReuseGroupOp`
- Python classes: `reuse_group_type`, `reuse_group`, `reuse_group_ir_type`

**Out of scope** for this phase:
- Offset calculation based on reuse groups (Phase 4)
- `set_buffer_overlap` API (Phase 4)

---

## Reuse Group Concepts

A **reuse group** defines buffer overlap relationships for memory allocations in shared memory (smem) or tensor memory (tmem):

- **shared**: Elements logically occupy the same memory region at each buffer index. Useful when buffers are used at different times and can share the same physical memory.
- **distinct**: Elements must be placed in non-overlapping memory regions. Useful when buffers need to be accessed simultaneously.

The reuse group forms a tree structure where:
- Leaf nodes are `buffered_tensor` objects (from `local_alloc`)
- Internal nodes are nested `reuse_group` objects
- The root defines the top-level sharing relationship

### Nesting Constraint

Nested reuse groups must alternate between `shared` and `distinct` types. This constraint ensures a well-formed tree structure where the relationship type changes at each level.

---

## IR Design

### New Enum: `ReuseGroupKind`

```tablegen
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
```

### New MLIR Type: `ReuseGroupType`

```tablegen
def TLX_ReuseGroupType : TLXTypeDef<"ReuseGroup", "reuse_group", []> {
  let summary = "A reuse group type for buffer overlap definitions";

  let description = [{
    Represents a reuse group that defines buffer overlap relationships for
    memory allocations (shared memory or tensor memory). A reuse group organizes multiple buffers
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
             : (!ttg.memdesc<...>, !ttg.memdesc<...>) -> !tlx.reuse_group<shared>

    // Nested groups for complex sharing schemes
    %inner = tlx.reuse_group(%c, %d, %e) {group_type = distinct}
             : (...) -> !tlx.reuse_group<distinct>
    %outer = tlx.reuse_group(%a, %inner) {group_type = shared}
             : (...) -> !tlx.reuse_group<shared>
    ```
  }];

  let parameters = (ins
    EnumParameter<TLX_ReuseGroupKindAttr>:$groupKind
  );

  let assemblyFormat = "`<` $groupKind `>`";

  let genVerifyDecl = 1;
}
```

### New MLIR Operation: `ReuseGroupOp`

```tablegen
def TLX_ReuseGroupOp : TLX_Op<"reuse_group", [Pure]> {
  let summary = "Define a reuse group for buffer overlap relationships";

  let description = [{
    Creates a reuse group that defines buffer overlap relationships for
    memory allocations (shared memory or tensor memory). A reuse group organizes multiple buffers (or nested
    groups) with a specific relationship:

    - **shared**: Elements logically occupy the same memory region at each
      buffer index. Useful when buffers are used at different times.
    - **distinct**: Elements must be in non-overlapping memory regions.
      Useful when buffers need to be accessed simultaneously.

    The operation takes a variadic list of elements (buffered tensors or
    nested reuse groups) and produces a reuse group.

    Note: The storage_alias_spec is NOT part of this operation. Validation
    that all elements reference the same storage_alias_spec is performed
    by the SetBufferOverlapOp verifier.

    Constraints:
    - Nested reuse_groups cannot have the same group_kind as their parent.
    - All elements must use the same storage kind (smem or tmem).

    Example:
    ```mlir
    // Create shared reuse group for A and B
    %group = tlx.reuse_group(%a, %b) group_kind = shared
             : (!ttg.memdesc<...>, !ttg.memdesc<...>)
             -> !tlx.reuse_group<shared>
    ```
  }];

  let arguments = (ins
    Variadic<TLX_ReuseGroupElement>:$elements,
    TLX_ReuseGroupKindAttr:$group_kind
  );

  let results = (outs TLX_ReuseGroupType:$result);

  let assemblyFormat = [{
    `(` $elements `)` `group_kind` `=` $group_kind attr-dict `:`
    `(` qualified(type($elements)) `)` `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}
```

---

## Python API

### `reuse_group_type` Enum

```python
class reuse_group_type(enum.Enum):
    """
    Type of buffer relationship within a reuse group.

    - **shared**: Elements must logically occupy the same region in memory.
      There is no cross-index overlap, and elements share the memory. Elements
      should be used at different times.
    - **distinct**: Elements must be placed into non-overlapping regions of
      memory. Elements can be accessed simultaneously without conflicts.
    """

    shared = "shared"
    distinct = "distinct"
```

### `reuse_group` Class

```python
class reuse_group:
    """
    Defines buffer overlap relationships for memory allocations (shared memory or tensor memory).

    A reuse_group organizes multiple buffers (or nested groups) into either:
    - **shared**: Elements logically occupy the same memory region at each
      buffer index. Useful when buffers are used at different times and can
      share the same physical memory.
    - **distinct**: Elements must be placed in non-overlapping memory regions.
      Useful when buffers need to be accessed simultaneously.

    The reuse_group forms a tree structure where:
    - Leaf nodes are `buffered_tensor` objects
    - Internal nodes are nested `reuse_group` objects
    - The root defines the top-level sharing relationship

Note: The storage_alias_spec is NOT passed to reuse_group directly in Python.
    Instead, the spec is associated with the reuse group tree when passed to
    `storage_alias_spec.set_buffer_overlap()`. During IR lowering, the spec is
    then attached to each `ReuseGroupOp` in the tree.

    Example - Flash Attention buffer sharing:
        ```python
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

        # Allocate buffers
        qk_tiles = tlx.local_alloc(..., reuse=spec)
        p_tiles = tlx.local_alloc(..., reuse=spec)
        alpha = tlx.local_alloc(..., reuse=spec)

        # QK and (P, alpha) share the same memory region
        # P and alpha are placed in distinct (non-overlapping) regions
        # Note: spec is passed to set_buffer_overlap, not to reuse_group
        spec.set_buffer_overlap(
            tlx.reuse_group(
                qk_tiles,
                tlx.reuse_group(p_tiles, alpha, group_type=tlx.reuse_group_type.distinct),
                group_type=tlx.reuse_group_type.shared,
            )
        )
        ```

    Constraints:
        - Nested reuse_groups must have different group_type than the parent.
    """

    def __init__(
        self,
        *args: "buffered_tensor | reuse_group",
        group_type: reuse_group_type,
    ):
        """
        Initialize a reuse group.

        Args:
            *args: buffered_tensor or reuse_group objects. Must not be empty.
            group_type: The relationship type for elements in this group.
                - shared: Elements occupy the same logical memory region.
                - distinct: Elements must be in non-overlapping regions.

        Raises:
            ValueError: If args is empty.
            ValueError: If a nested reuse_group has the same group_type as this group.
            TypeError: If any element is not a buffered_tensor or reuse_group.
        """
        ...

    @property
    def args(self) -> tuple:
        """The elements in this group (read-only)."""
        ...

    @property
    def group_type(self) -> reuse_group_type:
        """The relationship type for this group (read-only)."""
        ...
```

---

## Example IR Output

### Simple Shared Reuse Group

Python:
```python
spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
a = tlx.local_alloc((64, 64), tl.float32, 2, tlx.storage_kind.smem, reuse=spec)
b = tlx.local_alloc((64, 64), tl.bfloat16, 2, tlx.storage_kind.smem, reuse=spec)

# The spec is associated via set_buffer_overlap, not passed to reuse_group
spec.set_buffer_overlap(
    tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.shared)
)
```

MLIR:
```mlir
%spec = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
%a = ttg.local_alloc ... : () -> !ttg.memdesc<2x64x64xf32, ...>
%b = ttg.local_alloc ... : () -> !ttg.memdesc<2x64x64xbf16, ...>
%group = tlx.reuse_group(%a, %b) group_kind = shared
         : (!ttg.memdesc<2x64x64xf32, ...>, !ttg.memdesc<2x64x64xbf16, ...>)
         -> !tlx.reuse_group<shared>
tlx.set_buffer_overlap(%spec, %group)
         : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
```

### Nested Reuse Groups

Python:
```python
spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
qk = tlx.local_alloc(..., reuse=spec)
p = tlx.local_alloc(..., reuse=spec)
alpha = tlx.local_alloc(..., reuse=spec)

# P and alpha are distinct, then QK shares with (P, alpha)
# Note: spec is NOT passed to reuse_group - it's passed to set_buffer_overlap
spec.set_buffer_overlap(
    tlx.reuse_group(
        qk,
        tlx.reuse_group(p, alpha, group_type=tlx.reuse_group_type.distinct),
        group_type=tlx.reuse_group_type.shared,
    )
)
```

MLIR:
```mlir
%spec = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
%qk = ttg.local_alloc ... : () -> !ttg.memdesc<...>
%p = ttg.local_alloc ... : () -> !ttg.memdesc<...>
%alpha = ttg.local_alloc ... : () -> !ttg.memdesc<...>

%inner = tlx.reuse_group(%p, %alpha) group_kind = distinct
         : (!ttg.memdesc<...>, !ttg.memdesc<...>) -> !tlx.reuse_group<distinct>
%outer = tlx.reuse_group(%qk, %inner) group_kind = shared
         : (!ttg.memdesc<...>, !tlx.reuse_group<distinct>)
         -> !tlx.reuse_group<shared>
tlx.set_buffer_overlap(%spec, %outer)
         : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
```

---

## Testing Plan

### Unit Tests

1. **Python API tests**
   - Create shared reuse group
   - Create distinct reuse group
   - Verify nested groups with alternating types
   - Verify error on nested groups with same type
   - Verify error on empty elements
   - Verify error on invalid element types

2. **IR lowering tests**
   - Verify correct MLIR operation is generated
   - Verify group_kind attribute is correctly set
   - Verify type is correctly constructed

3. **Operation verifier tests**
   - Nested reuse_group with same group_kind should fail
   - Empty elements list should fail

---

## Summary

Phase 3 delivers the IR interface for reuse groups:

| Component | Description |
|-----------|-------------|
| `ReuseGroupKind` enum | `shared` and `distinct` relationship types |
| `ReuseGroupType` | MLIR type representing reuse group with its kind |
| `ReuseGroupOp` | Operation to create reuse groups from elements |
| `reuse_group_type` | Python enum for group kinds |
| `reuse_group` | Python class for defining buffer relationships (spec-free) |

**Design Principle**: Reuse groups form a tree structure that declaratively describes buffer overlap relationships. Neither the Python `reuse_group` class nor the IR `ReuseGroupOp` include the spec - validation is performed by `SetBufferOverlapOp`. The actual offset calculation and memory layout is deferred to Phase 5.

---
---

# Phase 4: `set_buffer_overlap` JIT API

This phase defines the JIT interface for `set_buffer_overlap`, which allows users to define buffer overlap schemes using reuse groups. This phase focuses solely on the frontend API and IR generation, without any lowering or offset calculations.

**Important**: Code using `set_buffer_overlap` will successfully lower to IR but is expected to fail during subsequent lowering passes until Phase 5 implements the offset calculation.

---

## Scope

This phase covers:
- `set_buffer_overlap` method on `storage_alias_spec`
- New MLIR operation: `SetBufferOverlapOp`
- Verification that reuse groups are valid

**Out of scope** for this phase:
- Offset calculation based on overlap definitions (Phase 5)
- `SharedBufferSizeDefinitionPass` integration (Phase 5)
- `ReusedBufferOffsetCalculationPass` (Phase 5)

---

## Motivation

The `set_buffer_overlap` API allows users to declaratively specify their buffer overlap scheme. This is crucial because:

1. **Simplicity**: Users define the logical relationship between buffers without manual offset calculations.
2. **Safety**: The compiler can verify that the overlap scheme is valid and achievable.
3. **Flexibility**: The overlap scheme can change based on compile-time constants (e.g., block sizes, data types).

By separating the overlap definition from the offset calculation, users can express their intent while the compiler handles the implementation details.

---

## API Design

### `set_buffer_overlap` Method

The `set_buffer_overlap` method is added to the `storage_alias_spec` class:

```python
class storage_alias_spec:
    """
    Definition of a storage alias specification.
    ...
    """

    def set_buffer_overlap(self, overlap_def: reuse_group) -> None:
        """
        Define the buffer overlap scheme for allocations using this storage alias spec.

        This method specifies how buffers should be laid out in memory relative to
        each other. The overlap_def is a reuse_group tree that defines:
        - **shared**: Elements logically occupy the same memory region
        - **distinct**: Elements must be in non-overlapping memory regions

        This function lowers to an IR operation that links the storage alias spec
        to its defined overlap scheme. The compiler will use this information to
        compute buffer offsets in subsequent passes.

        Note: This method should be called after all allocations using this
        storage_alias_spec have been created, and the reuse_group should contain
        all relevant buffered_tensor objects.

        Args:
            overlap_def: A reuse_group defining the buffer overlap relationships.
                         The reuse_group must use this storage_alias_spec.

        Raises:
            ValueError: If overlap_def does not use this storage_alias_spec.
            TypeError: If overlap_def is not a reuse_group.

        Example:
            ```python
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate buffers
            qk_tiles = tlx.local_alloc(..., reuse=spec)
            p_tiles = tlx.local_alloc(..., reuse=spec)
            alpha = tlx.local_alloc(..., reuse=spec)

            # Define overlap scheme: QK shares with (P and alpha which are distinct)
            # Note: spec is NOT passed to reuse_group in Python.
            # It gets attached to each ReuseGroupOp during IR lowering.
            spec.set_buffer_overlap(
                tlx.reuse_group(
                    qk_tiles,
                    tlx.reuse_group(p_tiles, alpha, group_type=tlx.reuse_group_type.distinct),
                    group_type=tlx.reuse_group_type.shared,
                )
            )
            ```
        """
        ...
```

### Design Rationale

The API is designed with these principles:

1. **Called in JIT code**: This allows the overlap definition to depend on compile-time constants (e.g., `tl.constexpr` values), enabling different overlap schemes based on autotuning parameters or data types.

2. **Method on `storage_alias_spec`**: This establishes a clear ownership relationship - the storage alias spec owns its overlap definition.

3. **Takes a `reuse_group`**: Reuses the existing reuse group infrastructure from Phase 3, providing a consistent tree-based overlap representation.

4. **Spec association during lowering**: The Python `reuse_group` class does not take the spec as a parameter. The spec is only associated at the `SetBufferOverlapOp` level, which validates that all leaf elements in the reuse group tree were allocated with the correct storage_alias_spec.

---

## IR Design

### New MLIR Operation: `SetBufferOverlapOp`

```tablegen
def TLX_SetBufferOverlapOp : TLX_Op<"set_buffer_overlap", []> {
  let summary = "Define the buffer overlap scheme for a storage alias spec";

  let description = [{
    Defines the buffer overlap scheme for allocations using a storage alias spec.
    This operation links a storage_alias_spec to its overlap definition (a reuse_group).

    The compiler will use this information in subsequent passes to:
    1. Validate that the overlap scheme is achievable
    2. Compute buffer offsets to satisfy the overlap requirements

    This operation is eliminated during the ReusedBufferOffsetCalculationPass
    after offsets have been computed and applied.

    Constraints:
    - The reuse_group must reference the same storage_alias_spec
    - All elements in the reuse_group must use the same storage kind
    - This operation should appear after all local_alloc operations that
      reference the storage_alias_spec

    Example:
    ```mlir
    %spec = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %qk = ttg.local_alloc ... : () -> !ttg.memdesc<...>
    %p = ttg.local_alloc ... : () -> !ttg.memdesc<...>

    %group = tlx.reuse_group(%spec : %qk, %p) group_kind = shared
             : (!tlx.storage_alias_spec<smem> : ...) -> !tlx.reuse_group<shared>

    tlx.set_buffer_overlap(%spec, %group)
             : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    ```
  }];

  let arguments = (ins
    TLX_StorageAliasSpecType:$storage_alias_spec,
    TLX_ReuseGroupType:$overlap_def
  );

  let results = (outs);

  let assemblyFormat = [{
    `(` $storage_alias_spec `,` $overlap_def `)`
    `:` `(` qualified(type($storage_alias_spec)) `,` qualified(type($overlap_def)) `)` `->` `(` `)`
    attr-dict
  }];

  let hasVerifier = 1;
}
```

### Verifier Checks

The `ReuseGroupOp` verifier should check:

1. Nested reuse_groups cannot have the same group_kind as their parent
2. All elements must use the same storage kind (smem or tmem)

The `SetBufferOverlapOp` verifier should check:

1. All leaf elements (buffered tensors) in the reuse group tree were allocated with the same `storage_alias_spec`
2. The storage kind of all elements matches the storage kind of the `storage_alias_spec`

---

## Python Implementation

### `set_buffer_overlap` Implementation

```python
class storage_alias_spec(tl.base_value):
    """..."""

    def set_buffer_overlap(self, overlap_def: reuse_group) -> None:
        """Define the buffer overlap scheme for this storage alias spec."""
        overlap_def = tl._unwrap_if_constexpr(overlap_def)
        # Validate input type
        if not isinstance(overlap_def, reuse_group):
            raise TypeError(f"overlap_def must be a reuse_group, got {type(overlap_def).__name__}")

        # Lower to IR operation
        # The semantic function will create the SetBufferOverlapOp
        # During lowering, this spec is attached to each ReuseGroupOp in the tree
        return tlx_language.set_buffer_overlap(self, overlap_def)


@builtin
def set_buffer_overlap(
    storage_alias_spec: storage_alias_spec,
    overlap_def: reuse_group,
    _builder: ir.builder = None,
) -> None:
    """
    Semantic function that lowers set_buffer_overlap to IR.

    The reuse_group tree is lowered to IR without the spec - each ReuseGroupOp
    only contains its elements and group_kind. The SetBufferOverlapOp then
    links the spec to the root reuse group, and its verifier validates that
    all leaf elements were allocated with this spec.
    """
    # Get the IR handles
    spec_handle = storage_alias_spec.handle

    # Lower the reuse_group tree to IR (creates ReuseGroupOp for each node)
    overlap_handle = _lower_reuse_group(overlap_def, _builder)

    # Create the set_buffer_overlap operation linking spec to the overlap tree
    _builder.create_set_buffer_overlap(spec_handle, overlap_handle)
```

---

## Example Usage

### Simple Shared Overlap

```python
@triton.jit
def kernel(...):
    spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

    # Allocate buffers that will share memory
    a_tiles = tlx.local_alloc((64, 64), tl.float32, 2, tlx.storage_kind.smem, reuse=spec)
    b_tiles = tlx.local_alloc((64, 64), tl.bfloat16, 2, tlx.storage_kind.smem, reuse=spec)

    # Define that a and b share the same memory region
    # Note: spec is not passed to reuse_group - it gets attached during lowering
    spec.set_buffer_overlap(
        tlx.reuse_group(a_tiles, b_tiles, group_type=tlx.reuse_group_type.shared)
    )

    # ... kernel code using a_tiles and b_tiles ...
```

Generated IR:
```mlir
%spec = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
%a = ttg.local_alloc ... : () -> !ttg.memdesc<2x64x64xf32, ...>
%b = ttg.local_alloc ... : () -> !ttg.memdesc<2x64x64xbf16, ...>

%group = tlx.reuse_group(%a, %b) group_kind = shared
         : (!ttg.memdesc<...>, !ttg.memdesc<...>)
         -> !tlx.reuse_group<shared>

tlx.set_buffer_overlap(%spec, %group)
         : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
```

### Complex Nested Overlap (Flash Attention Pattern)

```python
@triton.jit
def flash_attention_kernel(...):
    spec = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)

    # Allocate buffers
    qk_tiles = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.float32, NUM_BUFFERS, tlx.storage_kind.tmem, reuse=spec)
    p_tiles = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.bfloat16, NUM_BUFFERS, tlx.storage_kind.tmem, reuse=spec)
    alpha = tlx.local_alloc((BLOCK_M, 1), tl.float32, NUM_BUFFERS, tlx.storage_kind.tmem, reuse=spec)
    l = tlx.local_alloc((BLOCK_M, 1), tl.float32, NUM_BUFFERS, tlx.storage_kind.tmem, reuse=spec)
    m = tlx.local_alloc((BLOCK_M, 1), tl.float32, NUM_BUFFERS, tlx.storage_kind.tmem, reuse=spec)

    # Define overlap scheme:
    # - QK shares with (P and (alpha, l, m))
    # - P is distinct from (alpha, l, m)
    # - alpha, l, m are distinct from each other
    spec.set_buffer_overlap(
        tlx.reuse_group(
            spec,
            qk_tiles,
            tlx.reuse_group(
                spec,
                p_tiles,
                tlx.reuse_group(spec, alpha, l, m, group_type=tlx.reuse_group_type.distinct),
                group_type=tlx.reuse_group_type.distinct,
            ),
            group_type=tlx.reuse_group_type.shared,
        )
    )

    # ... kernel code ...
```

### Conditional Overlap Based on Block Size

```python
@triton.jit
def kernel(BLOCK_M: tl.constexpr, ...):
    spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

    a_tiles = tlx.local_alloc((BLOCK_M, 64), tl.float32, 2, tlx.storage_kind.smem, reuse=spec)
    b_tiles = tlx.local_alloc((BLOCK_M, 64), tl.float32, 2, tlx.storage_kind.smem, reuse=spec)

    # Only share buffers if block size is large (memory constrained)
    if BLOCK_M >= 128:
        spec.set_buffer_overlap(
            tlx.reuse_group(a_tiles, b_tiles, group_type=tlx.reuse_group_type.shared)
        )
    else:
        # With smaller blocks, buffers are distinct (no sharing needed)
        spec.set_buffer_overlap(
            tlx.reuse_group(a_tiles, b_tiles, group_type=tlx.reuse_group_type.distinct)
        )
```

---

## Testing Plan

### Unit Tests

1. **Python API tests**
   - Call `set_buffer_overlap` with valid reuse_group
   - Verify error on non-reuse_group argument
   - Verify error when reuse_group uses different storage_alias_spec

2. **IR lowering tests**
   - Verify `SetBufferOverlapOp` is correctly generated
   - Verify operation references correct storage_alias_spec and reuse_group
   - Verify assembly format is correct

3. **Verifier tests**
   - Test that mismatched storage_alias_spec in reuse_group fails verification

### Expected Behavior

In this phase, kernels using `set_buffer_overlap` will:
- ✅ Successfully parse Python code
- ✅ Successfully lower to TTIR
- ✅ Successfully lower to TTGIR
- ❌ Fail during `ReusedBufferOffsetCalculationPass` (not yet implemented)

This is the expected behavior until Phase 5 implements the offset calculation pass.

---

## Summary

Phase 4 delivers the JIT interface for defining buffer overlap schemes:

| Component | Description |
|-----------|-------------|
| `set_buffer_overlap` method | Method on `storage_alias_spec` to define overlap scheme |
| `SetBufferOverlapOp` | MLIR operation linking storage alias spec to overlap definition |
| Python validation | Type and consistency checks for the API |

**Design Principle**: This phase establishes the frontend API and IR representation without implementing the backend passes. This allows users to start writing code with the new API while the offset calculation logic is developed in Phase 5.

**Note**: Code using `set_buffer_overlap` will fail during lowering until Phase 5 is implemented. This is intentional and allows incremental development of the feature.

---
---

# Phase 5: Buffer Offset Calculation Pass

This phase implements the `ReusedBufferOffsetCalculationPass`, which computes the `initial_offset` and `bytes_between_buffers` for each allocation based on the reuse group definitions. This pass processes `SetBufferOverlapOp` operations and assigns memory layout attributes to allocations.

---

## Scope

This phase covers:
- Buffer offset calculation logic integrated into `TLXStorageAliasLowering` pass
- In-memory mapping from `LocalAliasOp` results to `(buffer_offset, bytes_between_buffers)` pairs
- Elimination of `SetBufferOverlapOp` after processing
- Error handling for insufficient space and duplicate `set_buffer_overlap` calls

**Implementation approach**: The buffer offset calculation is integrated into the `TLXStorageAliasLowering` pass. Offset information is NOT stored as IR attributes on `LocalAliasOp`. Instead, offsets are computed and stored in an in-memory `DenseMap<Value, std::pair<int64_t, int64_t>>` that is passed forward for use in subsequent lowering steps (Phase 6).

**Out of scope** for this phase:
- Integration with allocation lowering (Phase 6)
- Buffer reuse analysis warnings (Phase 7)

---

## Concepts

### Buffer Layout Model

Each `storage_alias_local_alloc` produces a multi-buffered allocation with shape `[num, ...]`. The memory layout for a single allocation is:

```
|<------ bytes_between_buffers ----->|
|                                    |
v                                    v
|<--- buffer 0 --->|<--- padding --->|<--- buffer 1 --->|<--- padding --->|...
^
buffer_offset
```

Where:
- **`buffer_offset`**: The starting offset (in bytes) from the base of the storage alias spec for buffer index 0
- **`bytes_between_buffers`**: The stride (in bytes) between the **starting points** of consecutive buffer indices. This is the distance from the start of buffer N to the start of buffer N+1.

For a simple allocation without overlap:
- `buffer_offset = 0`
- `bytes_between_buffers = ceil(element_size * product(shape[1:]))` (i.e., the allocation size per buffer)

### Reuse Group Offset Calculation

The reuse group tree defines how allocations are laid out relative to each other:

1. **shared**: All elements in the group start at the same `buffer_offset`. The group's size is the maximum element size.

2. **distinct**: Elements are placed sequentially in memory. Each element's `buffer_offset` is computed as `parent_offset + sum(sizes of preceding elements)`. The group's size is the sum of all element sizes.

---

## Algorithm

### Step 1: Calculate bytes_between_buffers (Static)

The `bytes_between_buffers` is computed once for the entire `storage_alias_spec` and is the same for all allocations:

```cpp
int64_t totalBytes = specOp.getBufferSizeBytes();
int64_t numBuffers = getNumBuffersForSpec(spec);  // From first allocation's shape[0]
int64_t bytesBetweenBuffers = totalBytes / numBuffers;
```

This value is static and represents the stride between buffer indices for all allocations sharing this spec.

### Step 2: Assign Offsets Recursively

The algorithm recursively walks the reuse group tree, assigning `buffer_offset` to each allocation in the in-memory map. The `bytes_between_buffers` remains constant throughout.

```cpp
LogicalResult assignOffsets(Value element, int64_t currentOffset,
                            int64_t bytesBetweenBuffers,
                            DenseMap<Value, std::pair<int64_t, int64_t>> &offsetMap) {
  if (auto allocOp = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    // Store the offset information in the map (NOT as IR attributes)
    offsetMap[allocOp.getResult()] = {currentOffset, bytesBetweenBuffers};
    return success();
  }

  if (auto reuseGroupOp = element.getDefiningOp<ReuseGroupOp>()) {
    if (reuseGroupOp.getGroupKind() == ReuseGroupKind::shared) {
      // All children start at the same offset
      for (auto child : reuseGroupOp.getElements()) {
        if (failed(assignOffsets(child, currentOffset, bytesBetweenBuffers, builder)))
          return failure();
      }
    } else {  // distinct
      // Children are placed sequentially
      int64_t runningOffset = currentOffset;
      for (auto child : reuseGroupOp.getElements()) {
        if (failed(assignOffsets(child, runningOffset, bytesBetweenBuffers, builder)))
          return failure();
        runningOffset += getElementSize(child);
      }

      // Verify we have enough space
      int64_t totalSize = runningOffset - currentOffset;
      if (totalSize > bytesBetweenBuffers) {
        return reuseGroupOp.emitError()
            << "not enough space for distinct allocations: need "
            << totalSize << " bytes, have " << bytesBetweenBuffers << " bytes";
      }
    }
    return success();
  }

  llvm_unreachable("unexpected element type in reuse group");
}
```

**Key insight**: The `bytes_between_buffers` is the available space per buffer index. For `distinct` groups, we verify that the sum of all allocation sizes fits within this space.

### Step 4: Eliminate SetBufferOverlapOp

After processing each `SetBufferOverlapOp`:
1. Mark the `storage_alias_spec` as processed
2. Erase the `SetBufferOverlapOp` from the IR
3. Erase all `ReuseGroupOp` operations that are no longer needed

### Step 5: Validation

After all `SetBufferOverlapOp` operations are processed:
1. **Error if duplicate**: If a `storage_alias_spec` has multiple `SetBufferOverlapOp` referencing it, emit an error
2. **Error if remaining**: If any `SetBufferOverlapOp` remains in the IR after the pass, emit an error

---

## IR Changes

### In-Memory Offset Mapping

Instead of storing `buffer_offset` and `bytes_between_buffers` as IR attributes on `LocalAliasOp`, the offset information is kept in an in-memory data structure. This is passed through the lowering pass steps:

```cpp
// In StorageAliasLowering.cpp
DenseMap<Value, std::pair<int64_t, int64_t>> localAliasOffsetMap;
// Maps: LocalAliasOp result -> (buffer_offset, bytes_between_buffers)
```

Where:
- **`buffer_offset`**: The starting offset (in bytes) from the base of the storage alias spec for buffer index 0
- **`bytes_between_buffers`**: The stride (in bytes) between the **starting points** of consecutive buffer indices

This approach has several advantages:
1. **Simpler IR**: `LocalAliasOp` remains a pure aliasing operation without additional attributes
2. **No verifier complexity**: No need to verify that attributes are zero until lowering runs
3. **Encapsulation**: Offset information is only visible within the lowering pass where it's needed

---

## C++ Implementation

The buffer offset calculation is implemented as C++ helper functions called from `add_tlx_storage_alias_lowering`. This is NOT a standalone MLIR pass.

### Integration Point: `add_tlx_storage_alias_lowering`

```cpp
// In triton_tlx.cc

void add_tlx_storage_alias_lowering(mlir::ModuleOp module) {
  // ... existing lowering logic ...

  // Process SetBufferOverlapOps and calculate offsets
  if (failed(processBufferOverlapOps(module))) {
    llvm::report_fatal_error("Failed to process buffer overlap definitions");
  }

  // ... continue with allocation lowering ...
}
```

### Buffer Offset Calculation

The main entry point `processBufferOverlapOps` does the following:

1. Collects all `SetBufferOverlapOp` operations
2. For each operation:
   - Check for duplicate `set_buffer_overlap` on the same spec (error if found)
   - Compute `bytes_between_buffers = totalBytes / numBuffers`
   - Recursively assign offsets to all allocations in the reuse group tree
   - Erase the `SetBufferOverlapOp`
3. Verify no unprocessed `SetBufferOverlapOp` remains
4. Clean up unused `ReuseGroupOp` operations

```cpp
mlir::LogicalResult assignOffsets(mlir::Value element, int64_t currentOffset,
                                  int64_t bytesBetweenBuffers,
                                  mlir::OpBuilder &builder) {
  if (auto allocOp = element.getDefiningOp<tlx::StorageAliasLocalAllocOp>()) {
    // Error if attributes are already set (may have been manually specified)
    if (allocOp.getBufferOffset() || allocOp.getBytesBetweenBuffers()) {
      return allocOp.emitError("buffer layout attributes already set; "
                               "cannot process set_buffer_overlap");
    }

    // Set the offset attributes
    allocOp.setBufferOffsetAttr(builder.getI64IntegerAttr(currentOffset));
    allocOp.setBytesBetweenBuffersAttr(builder.getI64IntegerAttr(bytesBetweenBuffers));
    return mlir::success();
  }

  if (auto reuseGroupOp = element.getDefiningOp<tlx::ReuseGroupOp>()) {
    if (reuseGroupOp.getGroupKind() == tlx::ReuseGroupKind::shared) {
      // All children start at the same offset
      for (auto child : reuseGroupOp.getElements()) {
        if (mlir::failed(assignOffsets(child, currentOffset, bytesBetweenBuffers, builder)))
          return mlir::failure();
      }
    } else {  // distinct
      // Children are placed sequentially
      int64_t runningOffset = currentOffset;
      for (auto child : reuseGroupOp.getElements()) {
        if (mlir::failed(assignOffsets(child, runningOffset, bytesBetweenBuffers, builder)))
          return mlir::failure();
        runningOffset += getElementSize(child);
      }

      // Verify we have enough space
      int64_t totalSize = runningOffset - currentOffset;
      if (totalSize > bytesBetweenBuffers) {
        return reuseGroupOp.emitError()
            << "not enough space for distinct allocations: need "
            << totalSize << " bytes, have " << bytesBetweenBuffers << " bytes";
      }
    }
    return mlir::success();
  }

  llvm_unreachable("unexpected element type in reuse group");
}
```

**Key behavior**:
- If `buffer_offset` or `bytes_between_buffers` is already set on an allocation, it's an error
- For `shared` groups: all children get the same offset
- For `distinct` groups: children are placed sequentially, with size validation

---

## Example IR Transformation

### Input (after Phase 4)

```mlir
tt.func @kernel() {
  %spec = tlx.storage_alias_spec storage = smem, size = 32768 : !tlx.storage_alias_spec<smem, 32768>

  // QK: 2 x 64 x 64 x f32 = 32768 bytes total, 16384 per buffer
  %qk = tlx.storage_alias_local_alloc %spec : !tlx.storage_alias_spec<smem, 32768>
        -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>

  // P: 2 x 64 x 64 x f16 = 16384 bytes total, 8192 per buffer
  %p = tlx.storage_alias_local_alloc %spec : !tlx.storage_alias_spec<smem, 32768>
       -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>

  // Alpha: 2 x 64 x f32 = 512 bytes total, 256 per buffer
  %alpha = tlx.storage_alias_local_alloc %spec : !tlx.storage_alias_spec<smem, 32768>
           -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>

  // P and alpha are distinct, QK shares with (P, alpha)
  %inner = tlx.reuse_group(%p, %alpha) group_kind = distinct : ... -> !tlx.reuse_group<distinct>
  %outer = tlx.reuse_group(%qk, %inner) group_kind = shared : ... -> !tlx.reuse_group<shared>

  tlx.set_buffer_overlap(%spec, %outer) : ...

  tt.return
}
```

### After Buffer Offset Calculation (within TLXStorageAliasLowering)

After the offset calculation step, the offsets are stored in an in-memory map (not as IR attributes). The `SetBufferOverlapOp` and `ReuseGroupOp` are eliminated:

```mlir
tt.func @kernel() {
  %spec = tlx.storage_alias_spec storage = smem, size = 32768 : !tlx.storage_alias_spec<smem, 32768>

  // space_per_buffer = 32768 / 2 = 16384 bytes
  // bytes_between_buffers = 16384 (stored in offsetMap, not as attributes)

  // QK: offset = 0 (shared group starts at 0)
  %qk = tlx.storage_alias_local_alloc %spec
        : !tlx.storage_alias_spec<smem, 32768> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>

  // P: offset = 0 (first in distinct group which shares with QK)
  %p = tlx.storage_alias_local_alloc %spec
       : !tlx.storage_alias_spec<smem, 32768> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>

  // Alpha: offset = 8192 (after P in distinct group: 0 + 8192 = 8192)
  %alpha = tlx.storage_alias_local_alloc %spec
           : !tlx.storage_alias_spec<smem, 32768> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>

  // SetBufferOverlapOp and ReuseGroupOps are eliminated
  // offsetMap contains:
  //   %qk -> (buffer_offset=0, bytes_between_buffers=16384)
  //   %p -> (buffer_offset=0, bytes_between_buffers=16384)
  //   %alpha -> (buffer_offset=8192, bytes_between_buffers=16384)

  tt.return
}
```

### Memory Layout Visualization

```
Buffer Index 0 (offset 0-16383):
|<-- QK (0-16383, 16384 bytes) ------------------------------------------->|
|<-- P (0-8191, 8192 bytes) ---------->|<-- Alpha (8192-8447, 256 bytes) ->|

Buffer Index 1 (offset 16384-32767):
|<-- QK (16384-32767, 16384 bytes) ------------------------------------------->|
|<-- P (16384-24575, 8192 bytes) ------>|<-- Alpha (24576-24831, 256 bytes) -->|
```

---

## Error Cases

### Error: Not Enough Space for Distinct Allocations

```python
spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem, buffer_size_bytes=1024)
a = tlx.local_alloc((64, 64), tl.float32, 2, reuse=spec)  # 16384 bytes per buffer
b = tlx.local_alloc((64, 64), tl.float32, 2, reuse=spec)  # 16384 bytes per buffer

# Error: distinct requires 32768 bytes but only 512 per buffer available
spec.set_buffer_overlap(
    tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.distinct)
)
```

Error message:
```
error: 'tlx.reuse_group' op not enough space for distinct allocations: need 32768 bytes, have 512 bytes
```

### Error: Duplicate set_buffer_overlap

```python
spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
a = tlx.local_alloc((64, 64), tl.float32, 2, reuse=spec)
b = tlx.local_alloc((64, 64), tl.float32, 2, reuse=spec)

# First overlap definition
spec.set_buffer_overlap(
    tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.shared)
)

# Error: duplicate overlap definition for same spec
spec.set_buffer_overlap(
    tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.distinct)
)
```

Error message:
```
error: 'tlx.set_buffer_overlap' op storage_alias_spec already has a set_buffer_overlap defined; each spec can only have one overlap definition
```

### Error: Unprocessed set_buffer_overlap

If for some reason a `SetBufferOverlapOp` is not processed (e.g., malformed IR):

```
error: 'tlx.set_buffer_overlap' op set_buffer_overlap was not processed by ReusedBufferOffsetCalculationPass
```

---

## Testing Plan

### Unit Tests

1. **Basic offset calculation tests**
   - Two allocations with `shared`: both get offset 0
   - Two allocations with `distinct`: sequential offsets
   - Three or more allocations with `distinct`

2. **Nested reuse group tests**
   - `shared` containing `distinct`: inner elements get sequential offsets
   - `distinct` containing `shared`: shared group counts as max size
   - Three-level nesting

3. **Error case tests**
   - Not enough space for `distinct` allocations
   - Duplicate `set_buffer_overlap` on same spec
   - Unprocessed `set_buffer_overlap` remaining after pass

4. **bytes_between_buffers tests**
   - Verify correct stride calculation
   - Different num values

### MLIR Lit Tests

```mlir
// RUN: triton-opt %s --tlx-reused-buffer-offset-calculation --verify-each=false | FileCheck %s

// Test: basic shared offset calculation
#shared = #ttg.swizzled_shared<{...}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @test_shared_offset
  tt.func @test_shared_offset() {
    %spec = tlx.storage_alias_spec storage = smem, size = 32768 : !tlx.storage_alias_spec<smem, 32768>
    // CHECK: tlx.storage_alias_local_alloc %{{.*}} {buffer_offset = 0 : i64, bytes_between_buffers = 16384 : i64}
    %a = tlx.storage_alias_local_alloc %spec : !tlx.storage_alias_spec<smem, 32768> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: tlx.storage_alias_local_alloc %{{.*}} {buffer_offset = 0 : i64, bytes_between_buffers = 16384 : i64}
    %b = tlx.storage_alias_local_alloc %spec : !tlx.storage_alias_spec<smem, 32768> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>

    %group = tlx.reuse_group(%a, %b) group_kind = shared : (...) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%spec, %group) : (...) -> ()

    // CHECK-NOT: tlx.set_buffer_overlap
    // CHECK-NOT: tlx.reuse_group
    tt.return
  }
}

// -----

// Test: distinct offset calculation
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @test_distinct_offset
  tt.func @test_distinct_offset() {
    %spec = tlx.storage_alias_spec storage = smem, size = 65536 : !tlx.storage_alias_spec<smem, 65536>
    // CHECK: tlx.storage_alias_local_alloc %{{.*}} {buffer_offset = 0 : i64, bytes_between_buffers = 32768 : i64}
    %a = tlx.storage_alias_local_alloc %spec : ... -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: tlx.storage_alias_local_alloc %{{.*}} {buffer_offset = 16384 : i64, bytes_between_buffers = 32768 : i64}
    %b = tlx.storage_alias_local_alloc %spec : ... -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>

    %group = tlx.reuse_group(%a, %b) group_kind = distinct : (...) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%spec, %group) : (...) -> ()

    tt.return
  }
}
```

### Error Test Cases

Create a separate file for error tests with `--verify-diagnostics`:

```mlir
// RUN: triton-opt %s --tlx-reused-buffer-offset-calculation --verify-diagnostics

// Test: duplicate set_buffer_overlap
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @test_duplicate_overlap() {
    %spec = tlx.storage_alias_spec storage = smem, size = 32768 : !tlx.storage_alias_spec<smem, 32768>
    %a = tlx.storage_alias_local_alloc %spec : ... -> !ttg.memdesc<2x64x64xf32, ...>
    %b = tlx.storage_alias_local_alloc %spec : ... -> !ttg.memdesc<2x64x64xf16, ...>

    %group1 = tlx.reuse_group(%a, %b) group_kind = shared : (...) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%spec, %group1) : (...) -> ()

    %group2 = tlx.reuse_group(%a, %b) group_kind = distinct : (...) -> !tlx.reuse_group<distinct>
    // expected-error @+1 {{storage_alias_spec already has a set_buffer_overlap defined}}
    tlx.set_buffer_overlap(%spec, %group2) : (...) -> ()

    tt.return
  }
}

// -----

// Test: not enough space for distinct allocations
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @test_insufficient_space() {
    // Only 1024 bytes total = 512 per buffer, but need 32768
    %spec = tlx.storage_alias_spec storage = smem, size = 1024 : !tlx.storage_alias_spec<smem, 1024>
    %a = tlx.storage_alias_local_alloc %spec : ... -> !ttg.memdesc<2x64x64xf32, ...>
    %b = tlx.storage_alias_local_alloc %spec : ... -> !ttg.memdesc<2x64x64xf32, ...>

    // expected-error @+1 {{not enough space for distinct allocations}}
    %group = tlx.reuse_group(%a, %b) group_kind = distinct : (...) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%spec, %group) : (...) -> ()

    tt.return
  }
}
```

---

## Pass Pipeline Integration

The pass should be added to the TLX pipeline after `SharedBufferSizeDefinitionPass`:

```cpp
void addTLXPasses(OpPassManager &pm) {
  // Phase 2 passes
  pm.addPass(createSharedBufferSizeDefinitionPass());

  // Phase 5 pass
  pm.addPass(createReusedBufferOffsetCalculationPass());

  // Phase 2 passes (continued)
  pm.addPass(createSharedBufferAllocationPass());
}
```

**Pass ordering**:
1. `SharedBufferSizeDefinitionPass` - compute buffer sizes
2. `ReusedBufferOffsetCalculationPass` - compute offsets based on reuse groups (NEW)
3. `SharedBufferAllocationPass` - materialize allocations

---

## Summary

Phase 5 delivers the buffer offset calculation based on reuse group definitions:

| Component | Description |
|-----------|-------------|
| `buffer_offset` attribute | Starting offset for buffer index 0 |
| `bytes_between_buffers` attribute | Stride between buffer indices |
| `ReusedBufferOffsetCalculationPass` | Computes offsets from reuse group tree |
| Verifier update | Ensures attributes are zero until pass runs |
| Error handling | Insufficient space and duplicate overlap errors |

### Key Design Decisions

1. **Greedy processing**: For `shared` groups, all elements start at offset 0. For `distinct` groups, elements are assigned offsets sequentially in order.

2. **Size calculation**: `shared` uses max element size, `distinct` uses sum of element sizes.

3. **Space validation**: The pass emits an error if `distinct` elements don't fit in the available `space_per_buffer`.

4. **Single overlap per spec**: Each `storage_alias_spec` can have at most one `set_buffer_overlap` definition.

5. **Clean elimination**: `SetBufferOverlapOp` and unused `ReuseGroupOp` operations are removed after processing.

---

# Phase 6: LLVM Lowering for Buffer Padding via Shape Transformation

This phase addresses how the `buffer_offset` and `bytes_between_buffers` values computed in Phase 5 are used during lowering. These values are stored in an in-memory hashmap (not as IR attributes) and used to transform accesses.

## Alternatives Considered

Two alternative approaches were considered but rejected:

1. **Extend `MemDescReinterpretOp` and `MemDescIndexOp`**: Add optional `buffer_offset` and `bytes_between_buffers` attributes to the existing TritonGPU operations and update their LLVM lowering to use these values.

2. **Introduce custom IR nodes**: Create new TLX-specific operations (`PaddedMemDescReinterpretOp`, `PaddedMemDescIndexOp`) that explicitly carry offset/stride information.

Both alternatives were rejected because they may violate assumptions in the `LinearLayout` system. The `LinearLayout` infrastructure computes physical addresses from logical coordinates based on the allocation shape. Introducing explicit offsets or non-standard strides that don't match the shape would break these assumptions, potentially causing incorrect address calculations in swizzled shared memory layouts. These could be harmless, but given the lack of a compelling use
case it seems safer to avoid them.

## Chosen Approach: Shape Transformation with Index Rewriting

The chosen approach transforms the IR by modifying the allocation shape to absorb padding, then rewrites all buffer index accesses to use scaled indices. This leverages the existing `MemDescIndexOp` stride computation without any changes to the lowering, ensuring full compatibility with `LinearLayout`.

In effect this is equivalent to the manual padding that users were previously required to apply to their allocation shapes. The difference is that the padding is now computed automatically based on the reuse group definitions, the reduce mistakes
and move complexity to the compiler.

### Core Concept

Padding between buffers can be represented as an expanded buffer dimension:

**Example 1: Padding doubles the buffer stride**

```
Original:
  Shape: 2 x 4 x 8 (2 buffers, each 4x8 = 32 elements = 64 bytes for fp16)
  bytes_between_buffers: 128 bytes (64 bytes padding per buffer)
  buffer_offset: 0

Transformed:
  Shape: 4 x 4 x 8 (doubled buffer dimension)
  Access buffer[i] becomes buffer[2*i]

Physical layout:
  [buffer 0 data][padding][buffer 1 data][padding]
  [   64 bytes  ][64 bytes][  64 bytes  ][64 bytes]
       ↑                         ↑
    index 0                   index 2
```

**Example 2: Padding with offset**

```
Original:
  Shape: 2 x 4 x 8 (2 buffers)
  bytes_between_buffers: 128 bytes
  buffer_offset: 64 bytes (starts at second "slot")

Transformed:
  Shape: 4 x 4 x 8 (doubled buffer dimension)
  Access buffer[i] becomes buffer[2*i + 1]

Physical layout:
  [padding][buffer 0 data][padding][buffer 1 data]
  [64 bytes][  64 bytes  ][64 bytes][  64 bytes  ]
       ↑          ↑            ↑          ↑
    index 0    index 1      index 2    index 3
```

### Mathematical Formulation

Given:
- `original_buffer_size`: size of one buffer in bytes = `product(shape[1:]) * element_size`
- `bytes_between_buffers`: stride between buffer starts in bytes
- `buffer_offset`: offset of buffer 0 from allocation base in bytes
- `num_buffers`: original shape[0]

Compute:
```
scale_factor = bytes_between_buffers / original_buffer_size
offset_slots = buffer_offset / original_buffer_size

new_buffer_dim = num_buffers * scale_factor + offset_slots
new_shape = [new_buffer_dim] + shape[1:]

// For access to logical buffer i:
physical_index = scale_factor * i + offset_slots
```

**Constraints**:
- `bytes_between_buffers` must be an integer multiple of `original_buffer_size`
- `buffer_offset` must be an integer multiple of `original_buffer_size`

### Implementation: Integrated into `TLXStorageAliasLowering` Pass

The shape transformation and index rewriting logic is added as a new step in the existing `TLXStorageAliasLowering` pass. This combined pass already handles storage alias size computation, buffer overlap processing, and allocation materialization. Adding padding transformation here ensures correct ordering and avoids a separate pass.

The `TLXStorageAliasLowering` pass is extended with a new Step 4:

1. **Step 1 (existing)**: Compute or validate storage alias sizes
2. **Step 2 (existing)**: Process buffer overlap operations (compute offsets)
3. **Step 3 (existing)**: Materialize storage alias allocations (creates `LocalAliasOp`)
4. **Step 4 (new)**: Transform padded accesses (rewrite `MemDescIndexOp` indices)

```cpp
// StorageAliasLowering.cpp (extended)

// Forward declaration for the new step
LogicalResult transformPaddedAccesses(ModuleOp m);

struct TLXStorageAliasLoweringPass
    : public impl::TLXStorageAliasLoweringBase<TLXStorageAliasLoweringPass> {
public:
  using impl::TLXStorageAliasLoweringBase<
      TLXStorageAliasLoweringPass>::TLXStorageAliasLoweringBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    LDBG("Running TLXStorageAliasLowering (combined pass)");

    // Step 1: Compute or validate storage alias sizes
    LDBG("Step 1: Computing/validating storage alias sizes");
    if (failed(computeOrValidateStorageAliasSizes(m))) {
      signalPassFailure();
      return;
    }

    // Step 2: Process buffer overlap operations (compute offsets)
    LDBG("Step 2: Processing buffer overlap operations");
    DenseMap<Value, std::pair<int64_t, int64_t>> offsetMap;
    if (failed(processBufferOverlapOps(m, offsetMap))) {
      signalPassFailure();
      return;
    }

    // Step 3: Materialize storage alias allocations
    LDBG("Step 3: Materializing storage alias allocations");
    if (failed(materializeStorageAliasAllocations(m, offsetMap))) {
      signalPassFailure();
      return;
    }

    // Step 4: Transform padded accesses (NEW)
    // This rewrites MemDescIndexOp indices to account for buffer_offset
    // and bytes_between_buffers by applying the formula:
    //   physical_index = scale_factor * logical_index + offset_slots
    LDBG("Step 4: Transforming padded accesses");
    if (failed(transformPaddedAccesses(m))) {
      signalPassFailure();
      return;
    }

      LDBG("TLXStorageAliasLowering completed successfully");
    }
  };
```

### Step 4: Transform Padded Accesses

The final step uses the offset information from `localAliasOffsetMap` to transform `MemDescIndexOp` indices:

```cpp
// NEW: Step 4 implementation - Transform padded accesses
LogicalResult transformPaddedAccesses(
    ModuleOp m,
    const DenseMap<Value, std::pair<int64_t, int64_t>> &localAliasOffsetMap) {
  OpBuilder builder(m.getContext());

  // Process LocalAliasOps with non-default offset information from the map
  for (auto &[aliasResult, offsets] : localAliasOffsetMap) {
    int64_t bufferOffset = offsets.first;
    int64_t bytesPerBuffer = offsets.second;

    // Get the LocalAliasOp
    auto aliasOp = aliasResult.getDefiningOp<tlx::LocalAliasOp>();
    if (!aliasOp)
      continue;

    auto aliasType = cast<ttg::MemDescType>(aliasOp.getType());

    // Compute original buffer size
    auto aliasShape = aliasType.getShape();
    int64_t elemBits = aliasType.getElementTypeBitWidth();
    int64_t bufferElements = 1;
    for (size_t i = 1; i < aliasShape.size(); ++i) {
      bufferElements *= aliasShape[i];
    }
    int64_t bufferBytes = (bufferElements * elemBits) / 8;

    // Validate constraints
    if (bytesPerBuffer % bufferBytes != 0) {
      return aliasOp.emitError(
          "bytes_between_buffers must be multiple of buffer size");
    }
    if (bufferOffset % bufferBytes != 0) {
      return aliasOp.emitError(
          "buffer_offset must be multiple of buffer size");
    }

    int64_t scaleFactor = bytesPerBuffer / bufferBytes;
    int64_t offsetSlots = bufferOffset / bufferBytes;

    if (scaleFactor == 1 && offsetSlots == 0) {
      // No transformation needed (default layout)
      continue;
    }

    // Rewrite all MemDescIndexOp users with scaled indices
    SmallVector<ttg::MemDescIndexOp> indexOps;
    collectIndexOps(aliasOp.getResult(), indexOps);

    for (auto indexOp : indexOps) {
      builder.setInsertionPoint(indexOp);
      Location loc = indexOp.getLoc();

      Value originalIndex = indexOp.getIndex();

      // Compute: newIndex = scaleFactor * originalIndex + offsetSlots
      Value scaledIndex = originalIndex;
      if (scaleFactor != 1) {
        Value scaleVal = builder.create<arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(scaleFactor));
        scaledIndex = builder.create<arith::MulIOp>(
            loc, originalIndex, scaleVal);
      }

      Value newIndex = scaledIndex;
      if (offsetSlots != 0) {
        Value offsetVal = builder.create<arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(offsetSlots));
        newIndex = builder.create<arith::AddIOp>(loc, scaledIndex, offsetVal);
      }

      // Update the index operand
      indexOp.getIndexMutable().assign(newIndex);
    }
  }

  return success();
}

void collectIndexOps(Value memDesc, SmallVector<ttg::MemDescIndexOp> &result) {
  // Walk through all users, following through MemDescReinterpretOp
  for (auto &use : memDesc.getUses()) {
    Operation *user = use.getOwner();
    if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(user)) {
      result.push_back(indexOp);
    } else if (auto reinterpret = dyn_cast<ttg::MemDescReinterpretOp>(user)) {
      // Follow through reinterpret ops
      collectIndexOps(reinterpret.getResult(), result);
    } else if (auto alias = dyn_cast<tlx::LocalAliasOp>(user)) {
      // Follow through nested aliases
      collectIndexOps(alias.getResult(), result);
    }
  }
}
```

### Identifying Accesses

The primary challenge is correctly identifying all `MemDescIndexOp` operations that need rewriting. The access patterns include:

#### Direct Access Pattern

```mlir
%alias = tlx.local_alias %alloc {buffer_offset = 64, bytes_between_buffers = 128}
%view = ttg.memdesc_index %alias[%i]  // Direct user - easy to find
```

**Detection**: Walk `aliasOp.getResult().getUses()` directly.

#### Access Through Reinterpret

```mlir
%alias = tlx.local_alias %alloc {buffer_offset = 64, bytes_between_buffers = 128}
%reinterp = ttg.memdesc_reinterpret %alias : !type1 -> !type2
%view = ttg.memdesc_index %reinterp[%i]  // Indirect user
```

**Detection**: Recursively follow through `MemDescReinterpretOp` users.

#### Access Through Control Flow

```mlir
%alias = tlx.local_alias %alloc {buffer_offset = 64, bytes_between_buffers = 128}
scf.if %cond {
  scf.yield %alias
} else {
  scf.yield %other
} -> %result
%view = ttg.memdesc_index %result[%i]  // Merged value
```

**Detection**: Use MLIR's data flow analysis or conservative approximation. For control flow, the algorithm must follow through `scf.yield`, `scf.condition`, and `cf.branch` operations to find all reachable `MemDescIndexOp` uses.

### IR Example: Before and After

**Before transformation:**

```mlir
// 2 buffers of 4x8 fp16, with 128-byte stride and 64-byte offset
%alloc = ttg.local_alloc : () -> !ttg.memdesc<4x4x8xf16, #shared>

%alias = tlx.local_alias %alloc {
  buffer_offset = 64,          // 32 elements
  bytes_between_buffers = 128  // 64 elements
} : !ttg.memdesc<4x4x8xf16> -> !ttg.memdesc<2x4x8xf16>

// Access buffer 0: should access physical slot 1 (offset by 1)
%c0 = arith.constant 0 : i32
%view0 = ttg.memdesc_index %alias[%c0]

// Access buffer 1: should access physical slot 3 (2*1 + 1)
%c1 = arith.constant 1 : i32
%view1 = ttg.memdesc_index %alias[%c1]
```

**After transformation:**

```mlir
// Allocation stays the same (4 slots available)
%alloc = ttg.local_alloc : () -> !ttg.memdesc<4x4x8xf16, #shared>

// Alias type updated to reflect physical layout
%alias = tlx.local_alias %alloc : !ttg.memdesc<4x4x8xf16> -> !ttg.memdesc<4x4x8xf16>

// Access buffer 0: newIndex = 2*0 + 1 = 1
%c0 = arith.constant 0 : i32
%scale = arith.constant 2 : i32
%offset = arith.constant 1 : i32
%scaled0 = arith.muli %c0, %scale : i32
%idx0 = arith.addi %scaled0, %offset : i32  // idx0 = 1
%view0 = ttg.memdesc_index %alias[%idx0]

// Access buffer 1: newIndex = 2*1 + 1 = 3
%c1 = arith.constant 1 : i32
%scaled1 = arith.muli %c1, %scale : i32
%idx1 = arith.addi %scaled1, %offset : i32  // idx1 = 3
%view1 = ttg.memdesc_index %alias[%idx1]
```

### Constant Folding

For constant indices (the common case), the arithmetic folds at compile time:

```mlir
// After canonicalization:
%idx0 = arith.constant 1 : i32
%view0 = ttg.memdesc_index %alias[%idx0]

%idx1 = arith.constant 3 : i32
%view1 = ttg.memdesc_index %alias[%idx1]
```

MLIR's built-in canonicalization patterns handle this automatically.

### Pass Pipeline Integration

The padding transformation is now part of Step 4 in the existing `TLXStorageAliasLowering` pass, so no additional pass is needed:

```cpp
void addTLXPasses(OpPassManager &pm) {
  // Phase 5 + 6 combined: TLXStorageAliasLowering now handles:
  //   Step 1: Compute/validate storage alias sizes
  //   Step 2: Process buffer overlap operations (compute offsets)
  //   Step 3: Materialize storage alias allocations (creates LocalAliasOp)
  //   Step 4: Transform padded accesses (rewrite MemDescIndexOp indices)
  pm.addPass(createTLXStorageAliasLoweringPass());

  // Standard canonicalization to fold constant index arithmetic
  pm.addPass(createCanonicalizerPass());

  // Existing Phase 2 passes
  pm.addPass(createTLXRewriteLocalAliasPass());
}
```

This integrated approach has several advantages:

1. **Correct ordering**: The transformation runs immediately after `LocalAliasOp` is created (Step 3), while the `buffer_offset` and `bytes_between_buffers` attributes are still available.
2. **Single combined pass**: All storage alias lowering logic is in one place, making it easier to understand and maintain.
3. **No pass ordering issues**: The transformation is guaranteed to run before any subsequent passes that might modify the IR.

### Constraints and Limitations

1. **Alignment requirement**: `bytes_between_buffers` and `buffer_offset` must be integer multiples of `buffer_size`. Non-aligned padding is not supported.

2. **Memory overhead**: The expanded shape includes unused "padding slots" which waste memory.

3. **Static buffer counts**: The number of buffers must be statically known to compute the expanded shape.

---

## Future Work

The following will be addressed in subsequent phases:

1. **Phase 7**: Buffer reuse analysis warnings (optional performance tool)
2. **Phase 8**: Deprecation path for existing `reuse=buffered_tensor` pattern

---

## Overall Summary

Phase 2 delivers the core functionality for shared buffer allocation:

| Component | Description |
|-----------|-------------|
| `local_alloc` update | Accept `storage_alias_spec` in `reuse` parameter |
| `shared_buffer_local_alloc` op | Intermediate IR for allocations referencing shared buffers |
| `SharedBufferSizeDefinitionPass` | Compute or validate buffer sizes |
| `SharedBufferAllocationPass` | Materialize allocations using existing `LocalAllocOp`/`TMEMAllocOp` |

### Key Design Decisions

1. **Naming**: The class is named `storage_alias_spec` (storage alias specification) to emphasize that this object specifies how multiple allocations can alias the same storage region rather than performing an allocation itself.

2. **Storage validation**: `smemCluster` storage is **not supported** for `storage_alias_spec`. The semantics of sharing across cluster shared memory would require additional design consideration for distributed memory access patterns.

**Design Principle**: This phase reuses existing TritonGPU allocation operations (`LocalAllocOp`, `TMEMAllocOp`, `LocalAliasOp`) rather than introducing new allocation IR. This keeps the design simple and leverages well-tested lowering paths.

This phase enables users to define shared buffers and have the compiler automatically compute the required size, while still allowing explicit size specification for advanced use cases.
