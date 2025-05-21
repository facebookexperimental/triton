import triton.language.core as tl

class buffered_tensor(tl.base_value):
    """
    A symbolic type representing a tensor allocated in a manually managed buffer
    such as shared memory (SMEM) or local memory (TMEM).

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

    def __init__(self, handle, type: tl.dtype):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle
        self.type = type


class mbarriers(buffered_tensor):
    """
    Define mbarrier type derived from buffered_tensor to support barrier specific operations/validations
    """
    def __init__(self, handle):
        super().__init__(handle)
        pass
