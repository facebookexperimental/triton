import enum
from abc import abstractmethod
from typing import List, Optional, Tuple

import triton.language.core as tl
from triton._C.libtriton import ir
from triton.language.core import _aggregate as aggregate


class layout_encoding:

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.to_ir() must be overridden in subclasses")


class shared_layout_encoding(layout_encoding):

    def __init__(self):
        super().__init__()
        pass

    """
    Create a new layout object that is a permutation of the current layout.
    """

    @abstractmethod
    def make_permute(self, dims):
        raise NotImplementedError(f"{self.__class__.__name__}.make_permute() must be overridden in subclasses")

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.to_ir() must be overridden in subclasses")


class swizzled_shared_layout_encoding(shared_layout_encoding):

    def __init__(
        self,
        vectorSize,
        perPhase,
        maxPhase,
        order,
        numCTAs,
        numCTAsPerCGA,
        numCTASplit,
        numCTAOrder,
    ):
        super().__init__()
        self.vectorSize = vectorSize
        self.perPhase = perPhase
        self.maxPhase = maxPhase
        self.order = order
        self.numCTAs = numCTAs
        self.numCTAsPerCGA = numCTAsPerCGA
        self.numCTASplit = numCTASplit
        self.numCTAOrder = numCTAOrder

    """
    Make a default non-swizzled shared layout encoding.
    """

    @classmethod
    def make_default(cls, rank):
        return cls(
            vectorSize=1,
            perPhase=1,
            maxPhase=1,
            order=list(reversed(range(rank))),  # e.g, [1, 0] as a row-major order
            numCTAs=[1] * rank,
            numCTAsPerCGA=[1] * rank,
            numCTASplit=[1] * rank,
            numCTAOrder=[1] * rank,
        )

    """
    Create a new layout that is a permutation of the given layout.
    """

    def make_permute(self, dims):
        permuted_order = tuple(self.order[d] for d in dims)
        return swizzled_shared_layout_encoding(
            self.vectorSize,
            self.perPhase,
            self.maxPhase,
            permuted_order,
            self.numCTAs,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
        )

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_swizzled_shared_encoding_attr(
            self.vectorSize,
            self.perPhase,
            self.maxPhase,
            self.order,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
        )


class tensor_memory_layout_encoding(shared_layout_encoding):

    def __init__(self, blockM, blockN, colStride, CTASplitM, CTASplitN):
        super().__init__()
        self.blockM = blockM
        self.blockN = blockN
        self.colStride = colStride
        self.CTASplitM = CTASplitM
        self.CTASplitN = CTASplitN

    """
    Make a default tensor memory layout encoding.
    """

    @classmethod
    def make_default(cls, shape):
        return cls(
            blockM=shape[0],
            blockN=shape[1],
            colStride=1,
            CTASplitM=1,
            CTASplitN=1,
        )

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_tensor_memory_encoding_attr(
            self.blockM,
            self.blockN,
            self.colStride,
            self.CTASplitM,
            self.CTASplitN,
        )


class tensor_memory_scales_layout_encoding:
    """
    Tensor memory scales layout encoding for Blackwell.
    Used for scales in scaled MMA operations.
    """

    def __init__(
        self,
        CTASplitM: int = 1,
        CTASplitN: int = 1,
    ):
        self.CTASplitM = CTASplitM
        self.CTASplitN = CTASplitN

    @classmethod
    def make_default(cls):
        return cls(CTASplitM=1, CTASplitN=1)

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_tensor_memory_scales_encoding_attr(
            self.CTASplitM,
            self.CTASplitN,
        )


class nv_mma_shared_layout_encoding(shared_layout_encoding):

    def __init__(
        self,
        shape,
        order,
        elemType,
        numCTAsPerCGA,
        numCTASplit,
        numCTAOrder,
        fp4Padded,
        swizzled,
    ):
        super().__init__()
        self.shape = shape
        self.order = order
        self.elemType = elemType
        self.numCTAsPerCGA = numCTAsPerCGA
        self.numCTASplit = numCTASplit
        self.numCTAOrder = numCTAOrder
        self.fp4Padded = fp4Padded
        self.swizzled = swizzled

    """
    Make a default NVMMA shared layout encoding.
    """

    @classmethod
    def make_default(cls, shape, elemType, fp4Padded=False):
        rank = len(shape)
        return cls(
            shape=shape,
            order=list(reversed(range(rank))),  # e.g, [1, 0] as a row-major order
            elemType=elemType,
            numCTAsPerCGA=[1] * rank,
            numCTASplit=[1] * rank,
            numCTAOrder=[1] * rank,
            fp4Padded=fp4Padded,
            swizzled=True,
        )

    """
    Create a new layout that is a permutation of the given layout.
    """

    def make_permute(self, dims):
        permuted_order = tuple(self.order[d] for d in dims)
        return nv_mma_shared_layout_encoding(
            self.shape,
            permuted_order,
            self.elemType,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
            self.fp4Padded,
            self.swizzled,
        )

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_nv_mma_shared_encoding_attr(
            [int(x) for x in self.shape],
            self.order,
            self.elemType.to_ir(builder),
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
            self.fp4Padded,
            self.swizzled,
        )

    def __str__(self) -> str:
        return f"nv_mma_shared_layout_encoding<{self.shape}, {self.order}, {self.elemType}, {self.numCTAsPerCGA}, {self.numCTASplit}, {self.numCTAOrder}, {self.fp4Padded}, {self.swizzled}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.order == other.order
                and self.elemType == other.elemType and self.numCTAsPerCGA == other.numCTAsPerCGA
                and self.numCTASplit == other.numCTASplit and self.numCTAOrder == other.numCTAOrder
                and self.fp4Padded == other.fp4Padded and self.swizzled == other.swizzled)


class DummyRegisterLayoutEncoding(layout_encoding):
    """
    Placeholder layout for register-distributed tensors.
    Will be resolved to BlockedEncodingAttr, MmaEncodingAttr,
    DotOperandEncodingAttr, etc. after inlining.
    If tmem_compatible is True, the layout will be resolved to a
    TMEM-compatible register layout suitable for TMEM load/store.
    """

    def __init__(self, shape: List[int], element_type: tl.dtype, tmem_compatible: bool = False):
        super().__init__()
        self.shape = shape
        self.element_type = element_type
        self.tmem_compatible = tmem_compatible

    def to_ir(self, builder: ir.builder):
        return builder.make_dummy_register_layout_attr(self.shape, self.element_type.to_ir(builder),
                                                       self.tmem_compatible)

    def __repr__(self):
        return f"DummyRegisterLayoutEncoding<{self.shape}, {self.element_type}, tmem_compatible={self.tmem_compatible}>"

    def __eq__(self, other):
        return (isinstance(other, DummyRegisterLayoutEncoding) and self.shape == other.shape
                and self.element_type == other.element_type and self.tmem_compatible == other.tmem_compatible)

    def __hash__(self):
        return hash((tuple(self.shape), self.element_type, self.tmem_compatible))


class storage_kind(enum.Enum):
    smem = "smem"
    tmem = "tmem"
    smemCluster = "smemCluster"


class DummyTMEMLayoutEncoding(layout_encoding):
    """
    Placeholder layout for TMEM tensors that will be resolved during layout propagation.
    Used for sub-16-bit element types where the final layout depends on usage context
    (e.g., as scales in scaled MMA operations).
    """

    def __init__(self):
        super().__init__()

    def to_ir(self, builder: ir.builder):
        return builder.make_dummy_tmem_layout_attr()

    def __repr__(self):
        return "DummyTMEMLayoutEncoding<>"

    def __eq__(self, other):
        return isinstance(other, DummyTMEMLayoutEncoding)

    def __hash__(self):
        return hash(self.__class__.__name__)


class reuse_group_type(enum.Enum):
    """
    Type of buffer relationship within a reuse group.

    - **shared**: Elements must logically occupy the same region in memory.
      There is no cross-index overlap, and elements share the memory. Elements
      are guaranteed to overlap at the same buffer index.
    - **distinct**: Elements must be placed into non-overlapping regions of
      memory. Elements can be accessed simultaneously without conflicts.

    Example:
        In the Flash Attention buffer sharing scheme:
        - qk_tiles and (p_tiles, alpha, l, m) are **shared** because they
          occupy the same logical memory region at each buffer index.
        - p_tiles, alpha, l, and m are **distinct** because they must not
          overlap with each other within a buffer index.

    Note:
        The "shared" requirement does not mean elements are identical or must
        physically overlap. With infinite memory, elements could be placed in
        completely separate regions. However, when elements are shared, the
        user is responsible for proper synchronization via barriers.
    """

    shared = "shared"
    distinct = "distinct"


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

    Note: The storage_alias_spec is NOT passed to reuse_group. Instead, the
    spec is associated with the reuse group tree when passed to
    `storage_alias_spec.set_buffer_overlap()`. Validation that all elements
    reference the same storage_alias_spec is performed during that call.

    Example - Flash Attention buffer sharing:
        ```python
        spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
        qk_tiles = tlx.local_alloc(..., reuse=spec)
        p_tiles = tlx.local_alloc(..., reuse=spec)
        alpha = tlx.local_alloc(..., reuse=spec)
        l = tlx.local_alloc(..., reuse=spec)
        m = tlx.local_alloc(..., reuse=spec)

        # QK and (P, alpha) share the same memory region
        # P and alpha are placed in distinct (non-overlapping) regions
        # Note: spec is passed to set_buffer_overlap, not to reuse_group
        spec.set_buffer_overlap(
            tlx.reuse_group(
                qk_tiles,
                tlx.reuse_group(
                    p_tiles,
                    alpha,
                    l,
                    m,
                    group_type=tlx.reuse_group_type.distinct
                ),
            )
        )
        ```

    Example - Subtiling with group_size:
        ```python
        # P has 2 * NUM_SLICES buffers, QK has 2 buffers.
        # We need to be able to access NUM_SLICES buffers at once as logically
        # this subtiled buffer is a single iteration.
        # With NUM_SLICES=2, P's buffers [0,1] map to QK[0], [2,3] map to QK[1]
        spec.set_buffer_overlap(
            tlx.reuse_group(
                qk_tiles,
                tlx.reuse_group(
                    tlx.reuse_group(p_tiles, group_size=NUM_SLICES),  # Subtiling wrapper
                    alpha,
                    l,
                    m,
                    group_type=tlx.reuse_group_type.distinct,
                ),
            )
        )
        ```
    """

    def __init__(
        self,
        *args: "buffered_tensor | reuse_group",
        group_type: "reuse_group_type" = reuse_group_type.shared,
        group_size: int = 1,
    ):
        """
        Initialize a reuse group.

        Args:
            *args: buffered_tensor or reuse_group objects. Must not be empty.
            group_type: The relationship type for elements in this group.
                - shared: Elements occupy the same logical memory region.
                - distinct: Elements must be in non-overlapping regions.
                Defaults to shared.
            group_size: Multiplier for buffer grouping (subtiling). Defaults to 1.
                When > 1, K consecutive buffers are treated as a single logical
                group for offset calculation. This enables subtiling where a
                logical buffer is divided into smaller chunks.

                For example, with group_size=2 on a tensor with 4 buffers:
                - Buffers [0,1] are treated as logical group 0
                - Buffers [2,3] are treated as logical group 1

                This changes buffer count validation: after dividing by group_size,
                all elements at each level must have identical effective buffer counts.

        Raises:
            ValueError: If args is empty.
            ValueError: If group_size is not a positive integer.
            TypeError: If any element is not a buffered_tensor or reuse_group.
        """
        if len(args) == 0:
            raise ValueError("reuse_group requires at least one element")

        # Validate group_size
        group_size = tl._unwrap_if_constexpr(group_size)
        if not isinstance(group_size, int) or group_size < 1:
            raise ValueError(f"group_size must be a positive integer, got {group_size}")

        # Validate element types
        args = tuple(tl._unwrap_if_constexpr(elem) for elem in args)
        for elem in args:
            if not isinstance(elem, (reuse_group, buffered_tensor)):
                raise TypeError(f"reuse_group elements must be buffered_tensor or reuse_group, "
                                f"got {type(elem).__name__}")

        self._args = args
        self._group_type = group_type
        self._group_size = group_size

    @property
    def args(self) -> tuple:
        """The elements in this group (read-only)."""
        return self._args

    @property
    def group_type(self) -> reuse_group_type:
        """The relationship type for this group (read-only)."""
        return self._group_type

    @property
    def group_size(self) -> int:
        """The buffer grouping multiplier for subtiling (read-only).

        Defaults to 1 (no grouping). When > 1, K consecutive buffers are
        treated as a single logical group for offset calculation purposes.
        """
        return self._group_size

    def _flatten_ir(self, handles) -> None:
        """Recursively flatten IR handles from all elements in the group."""
        for elem in self._args:
            elem._flatten_ir(handles)

    def to_ir(self, builder) -> ir.value:
        """
        Recursively lower this reuse_group tree to IR.

        Args:
            builder: The IR builder.

        Returns:
            The IR value representing the reuse_group.
        """
        # Collect IR values for elements
        ir_elements = []
        for elem in self._args:
            if isinstance(elem, reuse_group):
                # Recursively lower nested reuse_group
                ir_elements.append(elem.to_ir(builder))
            elif isinstance(elem, buffered_tensor):
                # Get the memdesc handle from the buffered_tensor
                ir_elements.append(elem.handle)
            else:
                raise TypeError(f"reuse_group element must be buffered_tensor or reuse_group, "
                                f"got {type(elem).__name__}")

        # Create the reuse_group IR operation
        group_kind = self._group_type.value  # "shared" or "distinct"
        return builder.create_reuse_group(ir_elements, group_kind, self._group_size)

    def __repr__(self):
        if self._group_size == 1:
            return f"reuse_group({self._args}, group_type={self._group_type.value})"
        else:
            return f"reuse_group({self._args}, group_type={self._group_type.value}, group_size={self._group_size})"


class reuse_group_ir_type(tl.base_type):
    """
    Type for reuse group specifications in MLIR.

    This type represents the MLIR ReuseGroupType and carries
    the group kind (shared/distinct).
    The storage kind is inferred from the elements and not stored in the type.
    """

    def __init__(
        self,
        group_kind: reuse_group_type,
    ):
        self._group_kind = group_kind

    @property
    def group_kind(self) -> reuse_group_type:
        """The group kind (shared/distinct) (read-only)."""
        return self._group_kind

    def __eq__(self, other):
        return (isinstance(other, reuse_group_ir_type) and self._group_kind == other._group_kind)

    def __repr__(self) -> str:
        return f"reuse_group_ir_type(group_kind={self._group_kind.value})"

    def mangle(self) -> str:
        return f"reuse_group_{self._group_kind.value}"


class storage_alias_spec(tl.base_value):
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
        storage: The storage kind (smem or tmem) for this buffer.
        buffer_size_bytes: Optional explicit size in bytes. Must be a compile-time
            constant if provided. Immutable after construction.

    Note:
        smemCluster storage is not supported yet for storage alias specifications.

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
        handle,
        storage: storage_kind,
        buffer_size_bytes: Optional[int] = None,
    ):
        """
        Initialize a shared buffer definition.

        This constructor is internal. Use tlx.storage_alias_spec() builtin instead.

        Args:
            handle: The IR handle for this storage alias specification.
            storage: The storage kind for this buffer. Must be smem or tmem.
                smemCluster is not supported.
            buffer_size_bytes: Optional explicit size in bytes. If provided,
                the compiler will verify that all referencing allocations fit
                within this size. This value is immutable after construction.

        Raises:
            ValueError: If storage is smemCluster (not supported).
        """
        super().__init__()
        if storage == storage_kind.smemCluster:
            raise ValueError("smemCluster storage is not supported for storage_alias_spec")
        self._handle = handle
        self._storage = storage
        self._buffer_size_bytes = buffer_size_bytes
        self.type = storage_alias_spec_type(storage, buffer_size_bytes)

    @property
    def handle(self):
        """The IR handle (read-only)."""
        return self._handle

    @property
    def storage(self) -> storage_kind:
        """The storage kind for this buffer (read-only)."""
        return self._storage

    @property
    def buffer_size_bytes(self) -> Optional[int]:
        """The explicit buffer size in bytes, or None if unsized (read-only)."""
        return self._buffer_size_bytes

    @tl.builtin
    def set_buffer_overlap(self, overlap_def: "reuse_group", _semantic=None) -> None:
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
            _semantic: Internal semantic parameter (passed automatically in JIT context).

        Raises:
            TypeError: If overlap_def is not a reuse_group.

        Example:
            ```python
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate buffers
            qk_tiles = tlx.local_alloc(..., reuse=spec)
            p_tiles = tlx.local_alloc(..., reuse=spec)
            alpha = tlx.local_alloc(..., reuse=spec)

            # Define overlap scheme: QK shares with (P and alpha which are distinct)
            spec.set_buffer_overlap(
                tlx.reuse_group(
                    qk_tiles,
                    tlx.reuse_group(p_tiles, alpha, group_type=tlx.reuse_group_type.distinct),
                    group_type=tlx.reuse_group_type.shared,
                )
            )
            ```
        """
        overlap_def = tl._unwrap_if_constexpr(overlap_def)
        # Validate input type
        if not isinstance(overlap_def, reuse_group):
            raise TypeError(f"overlap_def must be a reuse_group, got {type(overlap_def).__name__}")

        # Recursively lower the reuse_group tree to IR
        overlap_def_ir = overlap_def.to_ir(_semantic.builder)

        # Create the set_buffer_overlap IR operation
        _semantic.builder.create_set_buffer_overlap(self._handle, overlap_def_ir)

    def _flatten_ir(self, handles) -> None:
        handles.append(self._handle)

    def __repr__(self):
        size_str = f", size={self._buffer_size_bytes}" if self._buffer_size_bytes else ""
        return f"storage_alias_spec(storage={self._storage.value}{size_str})"


class storage_alias_spec_type(tl.base_type):
    """
    Type for storage alias specifications.

    This type represents the MLIR StorageAliasSpecType and carries
    storage kind and optional explicit size information.
    """

    def __init__(
        self,
        storage: storage_kind,
        buffer_size_bytes: Optional[int] = None,
    ):
        self._storage = storage
        self._buffer_size_bytes = buffer_size_bytes

    @property
    def storage(self) -> storage_kind:
        """The storage kind (read-only)."""
        return self._storage

    @property
    def buffer_size_bytes(self) -> Optional[int]:
        """The explicit buffer size in bytes, or None (read-only)."""
        return self._buffer_size_bytes

    def __eq__(self, other):
        return (isinstance(other, storage_alias_spec_type) and self._storage == other._storage
                and self._buffer_size_bytes == other._buffer_size_bytes)

    def __repr__(self) -> str:
        size_str = f", size={self._buffer_size_bytes}" if self._buffer_size_bytes else ""
        return f"storage_alias_spec_type(storage={self._storage.value}{size_str})"

    def mangle(self) -> str:
        size_part = f"_{self._buffer_size_bytes}" if self._buffer_size_bytes else ""
        return f"storage_alias_spec_{self._storage.value}{size_part}"

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder):
        return builder.get_storage_alias_spec_type(
            self._storage.value,
            self._buffer_size_bytes,
        )

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple["storage_alias_spec", int]:
        value = storage_alias_spec(
            handles[cursor],
            self._storage,
            self._buffer_size_bytes,
        )
        return value, cursor + 1


class buffered_tensor(tl.base_value):
    """
    A symbolic type representing a tensor allocated in a manually managed buffer
    such as shared memory (SMEM).

    This type is to model data that is not stored in global memory or registers
    but instead resides in hardware-close memory spaces with specialized
    allocation, access, or swizzling patterns.

    Unlike regular `tl.tensor`, which models values computed by operations,
    `buffered_tensor` reflects a memory-backed buffer that may be explicitly
    allocated and reused across program regions. It is primarily used with
    low-level intrinsics such as `tlx.local_alloc()`.

    Examples:
        a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, num=4)

    Attributes:
        handle: The backing IR value representing the buffer allocation.
    """

    def __init__(
        self,
        handle,
        element_ty: tl.dtype,
        shape: List,
        num: int,
        storage: storage_kind,
        layout: Optional[shared_layout_encoding] = None,
    ):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = shape
        self.type = buffered_tensor_type(element_ty, shape, num, storage, layout)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = element_ty

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def make_permute(self, handle, dims):
        permuted_layout = self.type.layout.make_permute(dims)
        return buffered_tensor(
            handle,
            self.dtype,
            [self.shape[d] for d in dims],
            self.type.num,
            self.type.storage,
            permuted_layout,
        )


class buffered_tensor_type(tl.block_type):

    def __init__(
        self,
        element_ty: tl.dtype,
        shape: List,
        num: int,
        storage: storage_kind,
        layout: Optional[shared_layout_encoding] = None,
    ):
        super().__init__(element_ty, shape)
        # Storage
        self.storage = storage
        # Layout encoding
        self.layout = layout
        # Buffer number. 0 means a single buffer, 1+ means a buffer array.
        self.num = num

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[buffered_tensor, int]:
        value = buffered_tensor(
            handles[cursor],
            self.scalar,
            self.shape,
            self.num,
            self.storage,
            self.layout,
        )
        return value, cursor + 1

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = "_".join(map(str, self.shape))
        if self.num > 0:
            shape += f"_{self.num}"
        return f"buffered_{elt}S{shape}"

    def __str__(self) -> str:
        return f"buffered_tensor_<{self.element_ty}, {self.shape}, {self.layout}, {self.num}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.num == other.num)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> None:
        shape = self.shape
        if self.num >= 1:
            shape = [self.num] + list(shape)
        return builder.get_memdesc_type(
            shape,
            self.element_ty.to_ir(builder),
            self.layout.to_ir(builder),
            self.storage.value,
        )

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)


class mbarrier(tl.base_value):
    """
    Define a mbarrier object
    """

    def __init__(
        self,
        handle,
        num: int,
        layout: Optional[swizzled_shared_layout_encoding],
        storage: storage_kind = storage_kind.smem,
    ):
        assert storage == storage_kind.smem or storage == storage_kind.smemCluster, (
            "mbarrier requires storage to be smem or smemCluster")
        self.handle = handle
        self.type = mbarrier_type(num, layout, storage)
        self.num = num

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def _unflatten_ir(self, handles, cursor):
        """Build a frontend value with the current dtype, wrapping a list of existing handles.
        cursor is the index of the first handle relevant to this value, and the function
        should return the updated cursor position after any handles consumed by the created value.
        """
        raise NotImplementedError


class mbarrier_type(buffered_tensor_type):

    def __init__(self, num: int, layout: Optional[swizzled_shared_layout_encoding], storage):
        super().__init__(tl.int64, [1], num, storage, layout)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[mbarrier, int]:
        value = mbarrier(handles[cursor], self.num, self.layout, self.storage)
        return value, cursor + 1

    def to_ir(self, builder: ir.builder) -> None:
        if self.num >= 1:
            shape = [self.num]
        else:
            shape = self.shape
        return builder.get_memdesc_type(
            shape,
            self.element_ty.to_ir(builder),
            self.layout.to_ir(builder),
            self.storage.value,
        )


class clc_response(tl.base_value):
    """
    Define a CLC response object
    """

    def __init__(
        self,
        handle,
        num: int,
        layout: Optional[swizzled_shared_layout_encoding],
    ):
        self.handle = handle
        self.type = clc_response_type(num, layout)
        self.num = num

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def _unflatten_ir(self, handles, cursor):
        """Build a frontend value with the current dtype, wrapping a list of existing handles.
        cursor is the index of the first handle relevant to this value, and the function
        should return the updated cursor position after any handles consumed by the created value.
        """
        raise NotImplementedError


class clc_response_type(buffered_tensor_type):
    # TODO. a more generic design about buffered tensor type
    # since we have two concrete use cases now (mbarrier and clc_response)
    # both of which are opaque objects with fixed size

    def __init__(self, num: int, layout: Optional[swizzled_shared_layout_encoding]):
        super().__init__(tl.int64, [1], num, storage_kind.smem, layout)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[clc_response, int]:
        value = clc_response(handles[cursor], self.num, self.layout)
        return value, cursor + 1

    def to_ir(self, builder: ir.builder) -> None:
        if self.num >= 1:
            shape = [self.num]
        else:
            shape = self.shape
        return builder.get_memdesc_type(
            shape,
            self.element_ty.to_ir(builder),
            self.layout.to_ir(builder),
            self.storage.value,
        )


@aggregate
class CLCPipelineContext:
    _clc_mbars_empty: mbarrier
    _clc_mbars_full: mbarrier
    _clc_responses: clc_response

    def __init__(
        self,
        clc_mbars_empty: mbarrier,
        clc_mbars_full: mbarrier,
        clc_responses: clc_response,
    ):
        self._clc_mbars_empty = clc_mbars_empty
        self._clc_mbars_full = clc_mbars_full
        self._clc_responses = clc_responses


class async_token(tl.base_value):
    """
    Defines a type of value used to track and synchronize asynchronous operations.
    """

    def __init__(self, handle):
        self.handle = handle
        self.type = async_token_type(handle)

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def _unflatten_ir(self, handles, cursor):
        raise NotImplementedError


class async_token_type(tl.base_type):

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, async_token_type)

    def __repr__(self) -> str:
        return "async_token_type"

    def mangle(self) -> str:
        return repr(self)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        return

    def _unflatten_ir(self, handles: List[ir.value], cursor: int):
        return async_token(handles[cursor]), cursor + 1


class tensor_descriptor_ptr(tl.base_value):
    """
    A pointer type for tensor descriptors with 128-byte stride semantics.
    When performing pointer arithmetic (ptr + 1), the pointer advances by 128 bytes,
    which is the size of a single tensor descriptor.
    """

    def __init__(self, handle, num: int, descriptor_size: int):
        super().__init__()
        self.handle = handle
        self.type = tensor_descriptor_ptr_type(num, descriptor_size)

    @property
    def num(self) -> int:
        """Number of descriptors this pointer can access."""
        return self.type.num

    @property
    def descriptor_size(self) -> int:
        """Size of each descriptor in bytes."""
        return self.type.size

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def _unflatten_ir(self, handles, cursor):
        raise NotImplementedError


class tensor_descriptor_ptr_type(tl.pointer_type):
    """
    Type for pointers to tensor descriptors.
    Encodes size-byte stride semantics for pointer arithmetic.
    """

    def __init__(self, num: int, size: int = 128):
        # Initialize with a block type of size int8 elements to get size-byte stride
        element_type = tl.block_type(tl.int8, [size])
        super().__init__(element_type, address_space=1)
        # Number of descriptors this pointer can access (1 means single descriptor)
        self.num = num
        # Size of each descriptor in bytes
        self.size = size

    def __eq__(self, other):
        return isinstance(other, tensor_descriptor_ptr_type) and self.num == other.num and self.size == other.size

    def __repr__(self) -> str:
        return f"tensor_descriptor_ptr_type(num={self.num}, size={self.size})"

    def mangle(self) -> str:
        if self.num > 1:
            return f"tensor_desc_ptr_{self.num}_{self.size}"
        return f"tensor_desc_ptr_{self.size}"

    def _unflatten_ir(self, handles: List[ir.value], cursor: int):
        return tensor_descriptor_ptr(handles[cursor], self.num, self.size), cursor + 1
