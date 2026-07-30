"""Tombstone for the retired FA4 monkey-patch driver."""

from .binder_cute import InheritedFA4BackendRemoved


def install(mode, binding):
    raise InheritedFA4BackendRemoved(
        "FA4 shims, including identity bindings, are not Twill-compiled "
        "kernels; prepare Twill IR for manual CUDA lowering instead"
    )
