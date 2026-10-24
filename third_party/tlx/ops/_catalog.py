"""Op catalog and dispatch."""

from __future__ import annotations

import dataclasses
import functools
import importlib
from typing import Any, Callable, Mapping, Optional

_PKG = __name__.rsplit(".", 1)[0]


class UnsupportedOp(RuntimeError):
    """No catalog entry for this op on the current target."""


class InvalidInput(ValueError):
    """An entry exists but cannot handle these inputs."""


@dataclasses.dataclass(frozen=True)
class OpSpec:
    op: str
    arch: str  # must equal Target.key
    variant: str  # label only; not a dispatch input, not user visible
    impl: str  # "module:attr", relative to this package, imported on first use
    dtypes: frozenset = frozenset()  # bare torch names, so the table needs no torch
    accepts: Optional[Callable[[Mapping[str, Any]], bool]] = None
    requires: frozenset = frozenset()
    invariant_vocab: frozenset = frozenset()  # canonical dim names a BitwiseInvarianceSpec may use
    enforces_invariance: bool = False  # the impl prunes schedules per the resolved spec
    default_space: str = "heuristic"  # used when the caller passes space=None

    def __str__(self) -> str:
        return f"{self.op}/{self.arch} ({self.variant})"


_FP16 = frozenset({"float16", "bfloat16"})

# Per-op dim vocabularies for BitwiseInvarianceSpec. Parallel dims tile the
# output; reduction dims order the accumulation.
_GEMM_DIMS = frozenset({"M", "N", "K"})

# A static table, not decorator self-registration: `impl` stays a string so
# `import triton.tlx` never imports a kernel module or builds autotune configs.
# A missing (op, arch) is deferred, not an error -- the op raises on that arch.
CATALOG: tuple[OpSpec, ...] = (
    OpSpec(
        op="mm",
        arch="gfx950",
        variant="heuristic",
        impl="kernels.mm.gfx950:mm",
        invariant_vocab=_GEMM_DIMS,
        # LocalSplitU and persistent plans remain FP16-only; the register and
        # general LDS paths also support BF16 and validate their own plans.
        dtypes=_FP16,
        requires=frozenset(),
    ),
)

_BY_KEY = {(s.op, s.arch): s for s in CATALOG}
assert len(_BY_KEY) == len(CATALOG), "duplicate (op, arch) in CATALOG"


def _target(device=None):
    # Lazy: hw.target imports torch.
    from triton.language.extra.tlx.hw.target import current_target, target_for_device

    return current_target() if device is None else target_for_device(device)


def _capabilities(target) -> frozenset:
    if target.spec is None:
        return frozenset()
    caps = set()
    if target.is_cuda and target.capability is not None and target.capability[0] >= 9:
        caps |= {"tma", "cluster"}
    if getattr(target, "has_tmem", False):
        caps.add("tmem")
    return frozenset(caps)


@functools.lru_cache(maxsize=None)
def _load(impl: str) -> Callable[..., Any]:
    mod, _, attr = impl.partition(":")
    return getattr(importlib.import_module(f".{mod}", package=_PKG), attr)


def _arches_for(op: str) -> list[str]:
    return sorted(s.arch for s in CATALOG if s.op == op)


def has_impl(op: str, arch: str) -> bool:
    """Is there a catalog entry for this pair, without importing the kernel?

    A table lookup, not a capability check: the benchmark suite uses it to skip
    an op cleanly on an arch it was never written for, rather than running every
    shape and reporting each one as an error.
    """
    return (op, arch) in _BY_KEY


def invariant_vocab_for(op: str) -> frozenset:
    """The dim vocabulary for `op`, shared by its arch entries; empty if unknown."""
    for spec in CATALOG:
        if spec.op == op:
            return spec.invariant_vocab
    return frozenset()


@functools.lru_cache(maxsize=None)
def any_enforces_invariance(op: str) -> bool:
    """Whether any arch entry for `op` enforces an invariance request yet."""
    return any(spec.op == op and spec.enforces_invariance for spec in CATALOG)


def impl_for(op: str, arch: Optional[str] = None, *, device=None) -> tuple[Callable[..., Any], OpSpec]:
    """The blessed callable for `op`, plus its spec.

    Raises rather than falling back: a silent fallback turns "TLX is not
    running here" into an unexplained performance cliff.

    Public op wrappers pass the input tensor's `device`, so dispatch follows
    the device that will execute the kernel rather than an ambient current
    device. An explicit `arch` pins the entry instead of detecting one. The
    capability check is then skipped -- the caller has asserted the target,
    and checking a pinned arch against the running device would reject the
    very case pinning exists for.
    """
    if arch is not None and device is not None:
        raise ValueError("impl_for accepts either arch or device, not both")
    available = ", ".join(_arches_for(op)) or "(nothing yet)"
    if arch is None:
        target = _target(device)
        if not target.key:
            location = f" for device={device}" if device is not None else ""
            raise UnsupportedOp(f"tlx.ops.{op}: no GPU visible{location}. Available on: {available}")
        spec = _BY_KEY.get((op, target.key))
        if spec is None:
            raise UnsupportedOp(f"tlx.ops.{op} has no implementation for {target.key}. "
                                f"Available on: {available}")
        missing = spec.requires - _capabilities(target)
        if missing:
            raise UnsupportedOp(f"{spec} needs {sorted(missing)}, which {target.key} does not report")
        return _load(spec.impl), spec

    spec = _BY_KEY.get((op, arch))
    if spec is None:
        raise UnsupportedOp(f"tlx.ops.{op} has no implementation for arch={arch!r}. Available on: {available}")
    return _load(spec.impl), spec


def check_inputs(spec: OpSpec, dtype=None, **dims) -> None:
    if dtype is not None and spec.dtypes:
        name = str(dtype).removeprefix("torch.")
        if name not in spec.dtypes:
            raise InvalidInput(f"{spec} does not support {name}; supported: {sorted(spec.dtypes)}")
    if spec.accepts is not None and not spec.accepts(dims):
        raise InvalidInput(f"{spec} does not support these inputs: {dims}")
