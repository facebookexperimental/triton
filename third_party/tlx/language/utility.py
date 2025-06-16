import triton.language.core as tl

from . import types as tlx
import re
import os


def cuda_parse_arch(arch):
    pattern = r"^sm(\d+)$"
    match = re.fullmatch(pattern, arch)
    if not match:
        raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
    return int(match.group(1))


@tl.builtin
def thread_id(axis, _builder=None):
    """
    Returns the id of the current thread instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = tl._unwrap_if_constexpr(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"thread_id axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(_builder.create_thread_id(axis), tl.int32)

@tl.builtin
def async_task_replica_id(_builder=None):
    return tl.constexpr(int(os.environ.get("TLX_ASYNC_TASK_REPLICA_ID", 0)))
