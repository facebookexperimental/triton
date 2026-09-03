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

    def __str__(self) -> str:
        return f"{self.op}/{self.arch} ({self.variant})"


_FP16 = frozenset({"float16", "bfloat16"})

# A static table, not decorator self-registration: `impl` stays a string so
# `import triton.tlx` never imports a kernel module or builds autotune configs.
# A missing (op, arch) is deferred, not an error -- the op raises on that arch.
CATALOG: tuple[OpSpec, ...] = (
    OpSpec(
        op="mm",
        arch="sm100",
        variant="ws",
        impl="kernels.mm.sm100:mm",
        dtypes=_FP16,
        # TMA needs 16-byte-aligned descriptor row strides. Checked against the
        # real strides, not M/N/K: a column-major operand is fed to its
        # descriptor transposed, which moves the constraint to another dim.
        accepts=lambda d: all(s * d["elem_bytes"] % 16 == 0 for s in d["row_strides"]),
        requires=frozenset({"tma", "tmem"}),
    ),
    OpSpec(
        op="mm",
        arch="gfx942",
        variant="lds_ring",
        impl="kernels.mm.gfx942:mm",
        dtypes=_FP16,
        # No `accepts`: operands are read through explicit strides rather than a
        # descriptor, so there is no alignment rule to fail. This arch therefore
        # admits shapes sm100 declines -- see kernels/mm/_shapes.py.
        requires=frozenset(),
    ),
    OpSpec(
        op="flash_attn",
        arch="sm100",
        variant="ws_pipelined_persistent",
        impl="kernels.flash_attn.sm100:flash_attn",
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") in (64, 128),
        requires=frozenset({"tma", "tmem"}),
    ),
    OpSpec(
        op="hstu_attn_dev",
        arch="sm100",
        variant="ws",
        impl="kernels.hstu_attn.sm100:hstu_attn",
        dtypes=_FP16,
        # Causal-only, non-causal is not supported yet
        accepts=lambda d: bool(d.get("causal", True)),
        requires=frozenset({"tma", "tmem"}),
    ),
    OpSpec(
        op="hstu_attn_dev",
        arch="gfx950",
        variant="tlx",
        impl="kernels.hstu_attn.gfx950:hstu_attn",
        dtypes=_FP16,
    ),
    OpSpec(
        op="kimi_delta_attention",
        arch="sm100",
        variant="ws",
        impl="kernels.kda.sm100:kimi_delta_attention",
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") == 128,
        requires=frozenset({"tma", "tmem"}),
    ),
)

_BY_KEY = {(s.op, s.arch): s for s in CATALOG}
assert len(_BY_KEY) == len(CATALOG), "duplicate (op, arch) in CATALOG"


def _target():
    # Lazy: hw.target imports torch.
    from triton.language.extra.tlx.hw.target import current_target

    return current_target()


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


def impl_for(op: str, arch: Optional[str] = None) -> tuple[Callable[..., Any], OpSpec]:
    """The blessed callable for `op`, plus its spec.

    Raises rather than falling back: a silent fallback turns "TLX is not
    running here" into an unexplained performance cliff.

    An explicit `arch` pins the entry instead of detecting one. The capability
    check is then skipped -- the caller has asserted the target, and checking a
    pinned arch against the running device would reject the very case pinning
    exists for.
    """
    available = ", ".join(_arches_for(op)) or "(nothing yet)"
    if arch is None:
        target = _target()
        if not target.key:
            raise UnsupportedOp(f"tlx.ops.{op}: no GPU visible. Available on: {available}")
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
