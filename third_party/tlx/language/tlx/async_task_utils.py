from triton.language import core


def _validate_num_regs(num_regs):
    if num_regs is not None and num_regs % 8 != 0:
        raise ValueError(f"num_regs must be divisible by 8, got {num_regs}")


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
                assert "num_regs" not in kwargs and "registers" not in kwargs, \
                    "Cannot specify registers for the default async_task; it receives leftover registers from the partition budget"
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class async_tasks:

    def __init__(self, *args, exclusive=False, **kwargs):
        self.exclusive = core._unwrap_if_constexpr(exclusive)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
