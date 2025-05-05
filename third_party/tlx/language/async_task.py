from triton.language import core


class async_task:
    """
    Context manager to run code fragments asynchronously.
    """
    def __init__(self, *args, _builder=None, **kwargs):
        self.builder = _builder
        # Handle the optional positional argument like [0]
        if args:
            self.task_ids = list({core._constexpr_to_value(tid) for tid in args[0]})
            self.num_warps = None
            self.explict = False
        else:
            self.task_ids = None
            self.num_warps = core._constexpr_to_value(kwargs.get("num_warps", None))
            self.explict = True

    def __enter__(self):
        if not self.explict:
            self.builder.set_async_task_ids(self.task_ids)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.builder.unset_async_task_ids()


class async_tasks:
    """
    Context manager to run code fragments asynchronously.
    """
    def __init__(self, _builder=None):
        # Support both: async_task([0]) and async_task(warp_id=4, num_warps=4)
        if args and isinstance(args[0], list):
            self.task_ids = list({core._constexpr_to_value(tid) for tid in task_ids})
            self.start_warp_id = None
            self.num_warps = None
            self.explict = False
        else:
            self.task_ids = None
            self.start_warp_id = kwargs.get("warp_id", None)
            self.num_warps = kwargs.get("num_warps", None)
            self.explict = True
        self.builder = _builder
