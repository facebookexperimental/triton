from triton.language import core


class async_task:
    """
    Context manager to run code fragments asynchronously.
    """

    def __init__(self, *args, _builder=None, **kwargs):
        self.builder = _builder
        # Handle the optional positional argument like [0]
        self.is_default = False
        self.is_explict = False
        self.task_ids = None
        self.num_warps = None
        if args:
            assert len(args) == 1
            if isinstance(args[0], core.constexpr) and args[0] == "default":
                self.is_explict = True
                self.is_default = True
            else:
                self.task_ids = list({core._constexpr_to_value(tid) for tid in args[0]})
        else:
            self.is_explict = True
            self.num_warps = core._constexpr_to_value(kwargs.get("num_warps", None))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class async_tasks:

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
