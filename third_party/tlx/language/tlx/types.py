import triton.language.core as tl
from triton.language.core import _aggregate as aggregate
from typing import Optional, List, Tuple
import enum
from abc import abstractmethod
from triton._C.libtriton import ir


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

    def __init__(self, shape, order, elemType, numCTAsPerCGA, numCTASplit, numCTAOrder, fp4Padded, swizzled):
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
    def make_default(cls, shape, elemType):
        rank = len(shape)
        return cls(
            shape=shape,
            order=list(reversed(range(rank))),  # e.g, [1, 0] as a row-major order
            elemType=elemType,
            numCTAsPerCGA=[1] * rank,
            numCTASplit=[1] * rank,
            numCTAOrder=[1] * rank,
            fp4Padded=False,
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


class storage_kind(enum.Enum):
    smem = "smem"
    tmem = "tmem"
    smemCluster = "smemCluster"


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
        value = buffered_tensor(handles[cursor], self.scalar, self.shape, self.num, self.storage, self.layout)
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
