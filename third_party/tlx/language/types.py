import triton.language.core as tl


class buffered_tensor(tl.base_value):
    """
    A symbolic type representing a tensor allocated in a manually managed buffer
    such as shared memory (SMEM) or local memory (TMEM).

    This type is to model data that is not stored in global memory or registers
    but instead resides in hardware-close memory spaces with specialized allocation
    , access, or swizzling patterns.

    Unlike regular `tl.tensor`, which models values computed by operations,
    `buffered_tensor` reflects a memory-backed buffer that may be explicitly
    allocated and reused across program regions. It is primarily used with
    low-level intrinsics such as `tlx.local_alloc()`.

    Examples:
        a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, num=4)

    Attributes:
        handle: The backing IR value representing the buffer allocation.
    """

    def __init__(self, handle):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle

# class mbarriers(tl.tensor):
class mbarriers(tl.base_value):
    """
    A symbolic type representing an array of mbarriers (each mbarrier is 8-byte)
    with 1D tensor in uint64 dtype stored in shared memory.

    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment

    Examples:
        bars = tlx.alloc_barriers(num=4)

    Attributes:
        handle: The backing IR value representing the buffer allocation.
    """

    def __init__(self, handle):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle
