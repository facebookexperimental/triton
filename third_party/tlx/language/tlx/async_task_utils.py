import warnings

from triton.language import core


def _validate_num_regs(num_regs):
    if num_regs is not None and num_regs % 8 != 0:
        raise ValueError(f"num_regs must be divisible by 8, got {num_regs}")
    # `nvvm.setmaxregister` requires the per-thread register count to be at
    # most 256. Reject over-limit requests here: without this they are
    # silently clamped (or silently abandon register optimization) downstream.
    # Note the floor stays unchecked: below-minimum requests (e.g. 16) are
    # masked by the warp-group max when they share a group with a request
    # >= 24, and existing kernels rely on that.
    if num_regs is not None and num_regs > 256:
        raise ValueError(f"num_regs must be at most 256, got {num_regs}")


def _warn_missing_num_regs() -> None:
    warnings.warn(
        "async_task without num_regs is deprecated: register counts will be "
        "required for every warp-specialize task, including 'default'; "
        "specify num_regs for every task",
        DeprecationWarning,
        stacklevel=3,
    )


class async_task:
    """
    Context manager to run code fragments asynchronously.
    """

    def __init__(self, *args, _builder=None, **kwargs):
        self.builder = _builder
        # Handle either an explicit default task or a task id list.
        self.is_default = False
        self.is_explict = False
        self.task_ids = None
        self.num_warps = None
        self.num_regs = None
        self.replicate = None
        self.warp_group_start_id = None
        if args:
            assert len(args) == 1
            if core._unwrap_if_constexpr(args[0]) == "default":
                self.is_explict = True
                self.is_default = True
                self.num_regs = core._unwrap_if_constexpr(kwargs.get("num_regs", kwargs.get("registers", None)))
                _validate_num_regs(self.num_regs)
                self.replicate = core._unwrap_if_constexpr(kwargs.get("replicate", 1))
                self.warp_group_start_id = core._unwrap_if_constexpr(kwargs.get("warp_group_start_id", None))
            else:
                self.task_ids = list({core._unwrap_if_constexpr(tid) for tid in args[0]})
        else:
            self.is_explict = True
            self.num_warps = core._unwrap_if_constexpr(kwargs.get("num_warps", None))
            self.num_regs = core._unwrap_if_constexpr(kwargs.get("num_regs", kwargs.get("registers", None)))
            _validate_num_regs(self.num_regs)
            self.replicate = core._unwrap_if_constexpr(kwargs.get("replicate", 1))
            self.warp_group_start_id = core._unwrap_if_constexpr(kwargs.get("warp_group_start_id", None))
        if self.is_explict and self.num_regs is None:
            _warn_missing_num_regs()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class async_tasks:

    def __init__(
        self,
        *args,
        exclusive=False,
        no_ending_cluster_sync=False,
        mbarrier_try_wait_suspend_ns=None,
        less_reg_mma=False,
        no_ws_header_tid_reuse=False,
        **kwargs,
    ):
        self.exclusive = core._unwrap_if_constexpr(exclusive)
        self.no_ending_cluster_sync = core._unwrap_if_constexpr(no_ending_cluster_sync)
        self.mbarrier_try_wait_suspend_ns = core._unwrap_if_constexpr(mbarrier_try_wait_suspend_ns)
        # Lower each MMA's SMEM operand descriptor with its own side-effecting
        # address computation so LLVM cannot CSE it across distant MMAs. Trades
        # a few extra ALU ops for a much shorter live range, which avoids
        # register spilling in MMA-heavy tasks (e.g. FA4 backward).
        self.less_reg_mma = core._unwrap_if_constexpr(less_reg_mma)
        # Hacky tuning knob. For unknown reasons, ptxas may spill %tid.x to
        # local memory when it is computed in the warp-specialize header and
        # kept live all the way into the tasks. This hides the header's %tid.x
        # read in opaque inline PTX so each task reads %tid.x on its own.
        self.no_ws_header_tid_reuse = core._unwrap_if_constexpr(no_ws_header_tid_reuse)
        if self.mbarrier_try_wait_suspend_ns is not None:
            if not isinstance(self.mbarrier_try_wait_suspend_ns, int) or self.mbarrier_try_wait_suspend_ns < 0:
                raise ValueError("mbarrier_try_wait_suspend_ns must be a non-negative integer")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
