class warp_pipeline_stage:
    """Mark an explicit inter- or intra-wave pipeline stage (AMD only).

    The default ``scope="inter_wave"`` preserves the original warp-pipeline
    behavior: the compiler splits the loop body at stage boundaries and inserts
    conditional barriers so that one warp group executes one stage ahead of the
    other. Correctness depends on the user's buffering and synchronization
    structure; the marker only defines where stages begin and end.

    With ``scope="intra_wave"``, two regions carrying the same ``pair`` form
    one instruction-scheduling window. With ``auto_interleave=True``, one
    region may instead contain both independent memory and MFMA work. The
    compiler partitions that mixed region from SSA dependencies, counts the
    instructions produced by lowering, and stably interleaves its read-only
    memory and compute chunks before register allocation. A dependency between
    the streams, a memory write/async commit, control flow, or an unknown side
    effect is rejected rather than guessed.

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
        scope: ``"inter_wave"`` (the default) phase-shifts warp groups;
            ``"intra_wave"`` schedules instructions within each wave.
        priority: Hardware scheduling hint (0-3), maps to s_setprio.
            Higher values indicate more urgent scheduling. Inter-wave only.
        pair: Non-negative source relationship identifier for adjacent
            intra-wave regions. It may be reused after a window and does not
            select the machine scheduling-group ID.
        auto_interleave: Let the compiler partition and interleave an
            intra-wave pair or one mixed intra-wave region.
        cover_policy: In intra-wave auto-interleave mode, ``"balanced"`` gives
            active memory anchors equal shares of surplus compute, while
            ``"proportional"`` preserves the target issue-cost ratio.
    """

    __slots__ = (
        "label",
        "scope",
        "priority",
        "pair",
        "auto_interleave",
        "cover_policy",
    )

    def __init__(
        self,
        label=None,
        *,
        scope="inter_wave",
        priority=None,
        pair=0,
        auto_interleave=False,
        cover_policy="balanced",
    ):
        if label is not None and not isinstance(label, str):
            raise ValueError(f"label must be a string or None, got {type(label).__name__}")
        if scope not in ("inter_wave", "intra_wave"):
            raise ValueError(
                "scope must be 'inter_wave' or 'intra_wave', got "
                f"{scope!r}"
            )
        if priority is not None and not (0 <= priority <= 3):
            raise ValueError(f"priority must be 0-3, got {priority}")
        if not isinstance(pair, int) or isinstance(pair, bool) or pair < 0:
            raise ValueError(f"pair must be a non-negative integer, got {pair!r}")
        if not isinstance(auto_interleave, bool):
            raise ValueError(
                "auto_interleave must be a bool, got "
                f"{type(auto_interleave).__name__}"
            )
        if cover_policy not in ("balanced", "proportional"):
            raise ValueError(
                "cover_policy must be 'balanced' or 'proportional', got "
                f"{cover_policy!r}"
            )
        if cover_policy != "balanced" and not auto_interleave:
            raise ValueError(
                "a non-default cover_policy requires auto_interleave=True"
            )
        if scope == "inter_wave" and (
            pair != 0 or auto_interleave or cover_policy != "balanced"
        ):
            raise ValueError(
                "pair, auto_interleave, and cover_policy are only meaningful "
                "with scope='intra_wave'"
            )
        if scope == "intra_wave" and priority is not None:
            raise ValueError(
                "priority is only meaningful with scope='inter_wave'"
            )
        self.label = label
        self.scope = scope
        self.priority = priority
        self.pair = pair
        self.auto_interleave = auto_interleave
        self.cover_policy = cover_policy

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
