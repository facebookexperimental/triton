import triton.language.core as tl
from typing import Optional, Self
import enum
from abc import abstractmethod


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
        raise NotImplementedError(
            f"{self.__class__.__name__}.make_permute() must be overridden in subclasses"
        )


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
            order = list(reversed(range(rank))), # e.g, [1, 0] as a row-major order
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
        return swizzled_shared_layout_encoding(
            self.vectorSize,
            self.perPhase,
            self.maxPhase,
            permuted_order,
            self.numCTAs,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder)


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
        self.type = type  # Tensor type (can be block_type)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar
        # Storage
        self.storage = storage
        # Layout encoding
        self.layout = layout


    def make_permute(self, handle, dims) -> Self:
        permuted_type = tl.block_type(self.type.scalar, [self.shape[d] for d in dims])
        permuted_layout = self.layout.make_permute(dims)
        return buffered_tensor(
            handle,
            permuted_type,
            self.storage,
            permuted_layout,
        )


class mbarriers(buffered_tensor):
    """
    Define mbarrier type derived from buffered_tensor to support barrier specific operations/validations
    """

    def __init__(self, handle):
        # Temporarily use 1, as the shape must be a power of 2.
        # TODO: use the actual barrier count to compute shape for precise boundary checks.
        block_type = tl.block_type(tl.int64, [1])
        super().__init__(handle, block_type, storage_kind.smem)
        pass


class async_token(tl.base_value):
    """
    Defines a type of value used to track and synchronize asynchronous operations.
    """

    def __init__(self, handle):
        self.handle = handle

    @property
    def type(self):
        return None  # Python expects this to exist even if unused
