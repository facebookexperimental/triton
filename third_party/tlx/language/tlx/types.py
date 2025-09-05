import triton.language.core as tl
from typing import Optional, Self, List, Tuple
import enum
from abc import abstractmethod
from triton._C.libtriton import ir
from triton.language.semantic import TritonSemantic


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
    def make_permute(self, dims) -> Self:
        raise NotImplementedError(f"{self.__class__.__name__}.make_permute() must be overridden in subclasses")

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.to_ir() must be overridden in subclasses")


class swizzled_shared_layout_encoding(shared_layout_encoding):

    def __init__(self, vectorSize, perPhase, maxPhase, order, numCTAs, numCTAsPerCGA, numCTASplit, numCTAOrder):
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

    def make_permute(self, dims) -> Self:
        permuted_order = tuple(self.order[d] for d in dims)
        return swizzled_shared_layout_encoding(self.vectorSize, self.perPhase, self.maxPhase, permuted_order,
                                               self.numCTAs, self.numCTAsPerCGA, self.numCTASplit, self.numCTAOrder)

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

    def __init__(self, blockM, blockN, unpacked, CTASplitM, CTASplitN):
        super().__init__()
        self.blockM = blockM
        self.blockN = blockN
        self.unpacked = unpacked
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
            unpacked=True,
            CTASplitM=1,
            CTASplitN=1,
        )

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_tensor_memory_encoding_attr(
            self.blockM,
            self.blockN,
            self.unpacked,
            self.CTASplitM,
            self.CTASplitN,
        )


class nv_mma_shared_layout_encoding(shared_layout_encoding):

    def __init__(self, shape, order, elemType, numCTAsPerCGA, numCTASplit, numCTAOrder, fp4Padded):
        super().__init__()
        self.shape = shape
        self.order = order
        self.elemType = elemType
        self.numCTAsPerCGA = numCTAsPerCGA
        self.numCTASplit = numCTASplit
        self.numCTAOrder = numCTAOrder
        self.fp4Padded = fp4Padded

    """
    Make a default NVMMA shared layout encoding.
    """

    @classmethod
    def make_default(cls, shape, elemType):
        rank = len(shape)
        return cls(shape=shape, order=list(reversed(range(rank))),  # e.g, [1, 0] as a row-major order
                   elemType=elemType, numCTAsPerCGA=[1] * rank, numCTASplit=[1] * rank, numCTAOrder=[1] * rank,
                   fp4Padded=False)

    """
    Create a new layout that is a permutation of the given layout.
    """

    def make_permute(self, dims) -> Self:
        permuted_order = tuple(self.order[d] for d in dims)
        return nv_mma_shared_layout_encoding(self.shape, permuted_order, self.elemType, self.numCTAsPerCGA,
                                             self.numCTASplit, self.numCTAOrder, self.fp4Padded)

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_nv_mma_shared_encoding_attr(
            [int(x) for x in self.shape],
            self.order,
            self.elemType.to_ir(builder),
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
            self.fp4Padded,
        )

    def __str__(self) -> str:
        return f"nv_mma_shared_layout_encoding<{self.shape}, {self.order}, {self.elemType}, {self.numCTAsPerCGA}, {self.numCTASplit}, {self.numCTAOrder}, {self.fp4Padded}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.order == other.order
                and self.elemType == other.elemType and self.numCTAsPerCGA == other.numCTAsPerCGA
                and self.numCTASplit == other.numCTASplit and self.numCTAOrder == other.numCTAOrder
                and self.fp4Padded == other.fp4Padded)


class storage_kind(enum.Enum):
    smem = "smem"
    tmem = "tmem"


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

    def __init__(self, handle, element_ty: tl.dtype, shape: List, num: int, storage: storage_kind,
                 layout: Optional[shared_layout_encoding] = None, semantic: TritonSemantic = None):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = shape
        self.type = buffered_tensor_type(element_ty, shape, num, storage, layout, semantic)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = element_ty

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def make_permute(self, handle, dims) -> Self:
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

    def __init__(self, element_ty: tl.dtype, shape: List, num: int, storage: storage_kind,
                 layout: Optional[shared_layout_encoding] = None, semantic: TritonSemantic = None):
        super().__init__(element_ty, shape)
        # Storage
        self.storage = storage
        # Layout encoding
        self.layout = layout
        # Buffer number. 0 means a single buffer, 1+ means a buffer array.
        self.num = num
        assert semantic or num == 0, "buffered_tensor array must be created with a builder"
        self.semantic = semantic

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[buffered_tensor, int]:
        value = buffered_tensor(handles[cursor], self.scalar, self.shape, self.num, self.storage, self.layout,
                                self.semantic)
        return value, cursor + 1

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        if self.num > 0:
            shape += f'_{self.num}'
        return f'buffered_{elt}S{shape}'

    def __str__(self) -> str:
        return f"buffered_tensor_<{self.element_ty}, {self.shape}, {self.layout}, {self.num}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.num == other.num)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> None:
        shape = self.shape
        builder = self.semantic.builder
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

    def __init__(self, handle, num: int, layout: Optional[swizzled_shared_layout_encoding],
                 semantics: TritonSemantic = None):
        self.handle = handle
        self.type = mbarrier_type(num, layout, semantics)
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

    def __init__(self, num: int, layout: Optional[swizzled_shared_layout_encoding], semantic: TritonSemantic):
        super().__init__(tl.int64, [1], num, storage_kind.smem, layout, semantic)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[mbarrier, int]:
        value = mbarrier(handles[cursor], self.num, self.layout, self.semantic)
        return value, cursor + 1

    def to_ir(self, builder: ir.builder) -> None:
        builder = self.semantic.builder
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

    def __init__(self, handle, num: int, layout: Optional[swizzled_shared_layout_encoding]):
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

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[mbarrier, int]:
        value = mbarrier(handles[cursor], self.num, self.layout)
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


class async_token(tl.base_value):
    """
    Defines a type of value used to track and synchronize asynchronous operations.
    """

    def __init__(self, handle):
        self.handle = handle

    @property
    def type(self):
        return None  # Python expects this to exist even if unused
