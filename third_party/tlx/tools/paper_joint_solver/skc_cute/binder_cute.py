"""Tombstone for the retired inherit-in-place FA4 backend.

The old functions returned a successful binding after dropping exact cycles,
physical warp placement, issue order, and solved geometry.  That behavior was
not schedule compilation.  Use
``skc.compiler.prepare_manual_cuda_handoff`` to emit the paper's pipelined,
warp-annotated IR for an expert CUDA C++ implementation.
"""


class InheritedFA4BackendRemoved(RuntimeError):
    pass


def _removed():
    raise InheritedFA4BackendRemoved(
        "the FA4 parameter-shim backend was removed because it did not "
        "compile the paper's schedule; prepare the manual CUDA handoff instead"
    )


def bind_fwd(*args, **kwargs):
    _removed()


def bind_bwd(*args, **kwargs):
    _removed()
