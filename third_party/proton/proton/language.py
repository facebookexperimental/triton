from triton.language import core as tl
from triton.language.core import builtin
from triton._C.libtriton import proton as triton_proton
from triton.language.semantic import TritonSemantic
from triton.experimental.gluon.language._semantic import GluonSemantic

from .flags import flags

_ALL_SEMANTICS = {
    "triton": TritonSemantic,
    "gluon": GluonSemantic,
}
"""
By default **only Gluon** semantic is enabled.
Instrumenting kernels written in Triton DSL is disable because Triton's higher-level IR undergoes
aggressive compiler rewrites (loop pipelining, instruction re-ordering, IR duplication, etc.).
These transformations can invalidate naïve instrumentation and lead to misleading results.
"""
_SEMANTICS = {_ALL_SEMANTICS["gluon"]}


def _check_supported_semantic(semantic):
    pass


def enable_semantic(semantic_name: str):
    _SEMANTICS.add(_ALL_SEMANTICS[semantic_name])


def disable_semantic(semantic_name: str):
    _SEMANTICS.remove(_ALL_SEMANTICS[semantic_name])


@builtin
def record(is_start: tl.constexpr, scope_name: tl.constexpr, predicate=True, _semantic=None):
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(_semantic)
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)
    predicate = tl._unwrap_if_constexpr(predicate)
    if predicate is False:
        return
    predicate_handle = None if predicate is True else predicate.handle
    handle = triton_proton.create_proton_record(_semantic.builder, is_start, scope_name, predicate_handle)
    return tl.tensor(handle, tl.void)


@builtin
def enter_scope(name: tl.constexpr, predicate=True, _semantic=None):
    record(is_start=True, scope_name=name, predicate=predicate, _semantic=_semantic)


@builtin
def exit_scope(name: tl.constexpr, predicate=True, _semantic=None):
    record(is_start=False, scope_name=name, predicate=predicate, _semantic=_semantic)


class scope:

    def __init__(self, name: str, predicate=True, _semantic=None):
        self.name = name
        self.predicate = predicate
        self.semantic = _semantic

    def __enter__(self):
        enter_scope(self.name, predicate=self.predicate, _semantic=self.semantic)

    def __exit__(self, exc_type, exc_value, traceback):
        exit_scope(self.name, predicate=self.predicate, _semantic=self.semantic)
