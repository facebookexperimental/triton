import triton.language.core as tl

from . import types as tlx


@tl.builtin
def thread_id(axis, _builder=None):
    """
    Returns the id of the current thread instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = tl._constexpr_to_value(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"thread_id axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(_builder.create_thread_id(axis), tl.int32)
