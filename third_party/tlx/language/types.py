import triton.language.core as tl
from typing import Optional, Self, List, Tuple
import enum
from abc import abstractmethod
from triton._C.libtriton import ir


class layout_encoding:

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__


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
    Create a new layout that is a permutation of the given layout.
    """

    def make_permute(self, dims) -> Self:
        permuted_order = tuple(self.order[d] for d in dims)
        return nv_mma_shared_layout_encoding(self.shape, permuted_order, self.elemType, self.numCTAsPerCGA, self.numCTASplit, self.numCTAOrder, self.fp4Padded)


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

    def __init__(self, handle, type: tl.dtype, storage: storage_kind, layout: Optional[shared_layout_encoding] = None):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = type.shape if type.is_block() else ()
        self.type = buffered_tensor_type(type, storage, layout)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def make_permute(self, handle, dims) -> Self:
        permuted_type = tl.block_type(self.type.dtype, [self.shape[d] for d in dims])
        permuted_layout = self.type.layout.make_permute(dims)
        return buffered_tensor(
            handle,
            permuted_type,
            self.type.storage,
            permuted_layout,
        )


class buffered_tensor_type(tl.base_type):

    def __init__(self, type: tl.dtype, storage: storage_kind, layout: Optional[shared_layout_encoding] = None):
        # Block shape
        self.shape = type.shape if type.is_block() else ()
        self.type = type  # Tensor type (can be block_type)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar
        self.scalar = type.scalar
        # Storage
        self.storage = storage
        # Layout encoding
        self.layout = layout

    def is_block(self):
        return self.type.is_block()

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[buffered_tensor, int]:
        value = buffered_tensor(handles[cursor], self.type, self.storage, self.layout)
        return value, cursor + 1


class buffered_tensors(tl.base_value):
    """
    Define a list of buffered_tensor
    """

    def __init__(self, base_tensor: buffered_tensor, num: tl.constexpr):
        self.base_tensor = base_tensor
        self.num = num
        self.handle = base_tensor.handle


class mbarrier(tl.base_value):
    """
    Define a mbarrier object
    """

    def __init__(self, handle):
        self.handle = handle
        self.type = mbarrier_type()

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def _unflatten_ir(self, handles, cursor):
        """Build a frontend value with the current dtype, wrapping a list of existing handles.
        cursor is the index of the first handle relevant to this value, and the function
        should return the updated cursor position after any handles consumed by the created value.
        """
        raise NotImplementedError


class mbarrier_type(buffered_tensor_type):

    def __init__(self):
        block_type = tl.block_type(tl.int64, [1])
        super().__init__(block_type, storage_kind.smem)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[mbarrier, int]:
        value = mbarrier(handles[cursor])
        return value, cursor + 1


class mbarriers(tl.base_value):
    """
    Define a list of mbarrier
    """

    def __init__(self, base_barrier: mbarrier, num: tl.constexpr):
        self.base_tensor = base_barrier
        self.num = num
        self.handle = base_barrier.handle


class async_token(tl.base_value):
    """
    Defines a type of value used to track and synchronize asynchronous operations.
    """

    def __init__(self, handle):
        self.handle = handle

    @property
    def type(self):
        return None  # Python expects this to exist even if unused
