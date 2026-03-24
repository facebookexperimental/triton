"""Minimal TLX types for plugin memory ops.

Ported from triton-tlx/third_party/tlx/language/tlx/types.py.
Only includes types needed by local_alloc, local_view, local_store, local_load.
"""

import enum
from abc import abstractmethod
from typing import List, Optional

import triton.language.core as tl
from triton._C.libtriton import ir


class layout_encoding:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__}.to_ir() must be overridden in subclasses"
        )


class shared_layout_encoding(layout_encoding):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def make_permute(self, dims):
        raise NotImplementedError(
            f"{self.__class__.__name__}.make_permute() must be overridden in subclasses"
        )

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__}.to_ir() must be overridden in subclasses"
        )


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

    @classmethod
    def make_default(cls, rank):
        return cls(
            vectorSize=1,
            perPhase=1,
            maxPhase=1,
            order=list(reversed(range(rank))),
            numCTAs=[1] * rank,
            numCTAsPerCGA=[1] * rank,
            numCTASplit=[1] * rank,
            numCTAOrder=[1] * rank,
        )

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

    @classmethod
    def make_default(cls, shape, elemType, fp4Padded=False):
        rank = len(shape)
        return cls(
            shape=shape,
            order=list(reversed(range(rank))),
            elemType=elemType,
            numCTAsPerCGA=[1] * rank,
            numCTASplit=[1] * rank,
            numCTAOrder=[1] * rank,
            fp4Padded=fp4Padded,
            swizzled=True,
        )

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


class storage_kind(enum.Enum):
    smem = "smem"
    tmem = "tmem"
    smemCluster = "smemCluster"


class buffered_tensor(tl.base_value):
    """A tensor allocated in a manually managed buffer (SMEM or TMEM)."""

    def __init__(
        self,
        handle,
        element_ty: tl.dtype,
        shape: List,
        num: int,
        storage: storage_kind,
        layout: Optional[shared_layout_encoding] = None,
    ):
        super().__init__()
        self.handle = handle
        self.shape = shape
        self.type = buffered_tensor_type(element_ty, shape, num, storage, layout)
        self.dtype = element_ty

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)


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
        self.storage = storage
        self.layout = layout
        self.num = num


class mbarrier(buffered_tensor):
    """An mbarrier allocated in shared memory."""

    def __init__(
        self,
        handle,
        num_barriers: int,
        layout: Optional[shared_layout_encoding] = None,
        is_warp_barrier: bool = False,
    ):
        super().__init__(
            handle,
            element_ty=tl.int64,
            shape=[num_barriers],
            num=num_barriers,
            storage=storage_kind.smem,
            layout=layout,
        )
        self.is_warp_barrier = is_warp_barrier
