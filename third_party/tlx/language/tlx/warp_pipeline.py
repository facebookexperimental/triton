class warp_pipeline_stage:
    """Context manager marking an explicit warp-pipeline stage (AMD only).

    This is a partitioning marker, not an automatic optimization. The compiler
    splits the loop body at stage boundaries and inserts conditional barriers
    so that one warp group executes one stage ahead of the other. Correctness
    depends on the user's buffering and synchronization structure — the marker
    only defines where stages begin and end.

    Typical usage requires multi-buffered shared memory and explicit async
    waits to ensure data is ready before consumption. See the gfx1250
    warp-pipeline GEMM example for the full pattern with prefetch, async
    wait, and epilogue drain.

    Usage inside @triton.jit::

        for k in tl.range(NUM_BUFFERS - 1, K_ITERS):
            with tlx.warp_pipeline_stage("lds_load", priority=1):
                a_tile = tlx.local_load(tlx.local_view(buf_A, consumer))
            tlx.async_load_wait_group(0)
            with tlx.warp_pipeline_stage("compute_and_load", priority=0):
                tlx.async_load(a_ptrs, tlx.local_view(buf_A, producer), ...)
                acc = tl.dot(a_tile, b_tile, acc)

    Args:
        label: Stage name for diagnostics (e.g. "load", "compute").
        priority: Hardware scheduling hint (0-3), maps to s_setprio.
            Higher values indicate more urgent scheduling. Optional.
    """

    __slots__ = ("label", "priority")

    def __init__(self, label=None, *, priority=None):
        if label is not None and not isinstance(label, str):
            raise ValueError(f"label must be a string or None, got {type(label).__name__}")
        self.label = label
        if priority is not None and not (0 <= priority <= 3):
            raise ValueError(f"priority must be 0-3, got {priority}")
        self.priority = priority

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
