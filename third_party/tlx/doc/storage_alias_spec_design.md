# [WIP] TLX `storage_alias_spec` Design Proposal

**Author:** Nick Riasanovsky
**Updated:** 2026-01-26
**Status:** Draft - Phase 1 & Phase 2

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

## Future Work

The following will be addressed in subsequent phases:

1. **Phase 3**: Define the APIs for reuse groups.
1. **Phase 4**: Add `set_buffer_overlap` API for defining overlap schemes and implement offset calculation based on overlap definitions
2. **Phase 5**: Buffer reuse analysis warnings (optional performance tool)
3. **Phase 5**: Deprecation path for existing `reuse=buffered_tensor` pattern

---

## Summary

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
