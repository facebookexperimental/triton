"""Cluster Launch Control (CLC) tile scheduler for core Triton.

CLC is a Blackwell (SM100+) hardware feature for dynamic persistent kernels: a
block that finishes early cancels a pending (not-yet-launched) cluster and takes
over its tile. This module exposes a minimal, opaque scheduler:

    sched = tl.clc_tile_scheduler()      # initialize
    while sched.is_valid():              # more work to steal?
        pid = sched.tile_id[0]           # current tile coord (x); [1]/[2] for y/z
        ... compute ...
        sched = sched.advance()          # fetch the next tile

The persistent loop carries only the 128-bit CLC result. The response buffer,
mbarrier, phase, first-tile seed, and the issue/wait overlap are ALL handled by a
later compiler lowering pass -- the kernel author writes none of that plumbing.
The API is standalone (it does not require warp specialization).

Ops emitted (all in the ``ttng`` dialect, barrier-free at this stage):
- ``clc_init``        -> i128 : seed for this CTA's own statically-launched tile
- ``clc_advance``     -> i128 : fetch the next tile's response
- ``clc_is_canceled`` -> i1   : is the current tile real work? (Pure)
- ``clc_get_program_id`` -> i32 : decode a tile coordinate (Pure)
"""
from typing import List, Tuple

import triton.language.core as tl
from triton.language.core import base_value, base_type, builtin, constexpr

__all__ = ["clc_tile_scheduler", "ClcTileScheduler"]


class _clc_result_type(base_type):
    """Type of the opaque 128-bit CLC response threaded through the loop.

    i128 is not a normal ``tl`` dtype, so we wrap the raw handle and remember its
    MLIR type to thread it through ``scf.while`` iter_args.
    """

    def __init__(self, ir_type):
        self.ir_type = ir_type

    def __eq__(self, other) -> bool:
        return isinstance(other, _clc_result_type) and self.ir_type == other.ir_type

    def __hash__(self):
        return hash(("clc_result", str(self.ir_type)))

    def __str__(self) -> str:
        return "clc_result"

    def to_ir(self, builder) -> None:
        return self.ir_type

    def _flatten_ir_types(self, builder, out: List) -> None:
        out.append(self.ir_type)

    def _unflatten_ir(self, handles: List, cursor: int) -> Tuple["_clc_result", int]:
        return _clc_result(handles[cursor], self.ir_type), cursor + 1

    def mangle(self) -> str:
        return "CLCres"


class _clc_result(base_value):
    """The opaque 128-bit CLC response value."""

    def __init__(self, handle, ir_type=None):
        self.handle = handle
        self.type = _clc_result_type(ir_type if ir_type is not None else handle.get_type())

    def _flatten_ir(self, handles: List) -> None:
        handles.append(self.handle)


class _tile_id_accessor(base_value):
    """Ephemeral proxy returned by ``scheduler.tile_id`` to support ``[dim]``.

    Created on attribute access and immediately subscripted; never loop-carried.
    """

    def __init__(self, scheduler):
        self._scheduler = scheduler
        self.type = None

    def _flatten_ir(self, handles: List) -> None:  # pragma: no cover - never called
        raise TypeError("tile_id is not a value; index it as tile_id[dim]")

    @builtin
    def __getitem__(self, dim, _semantic=None):
        if isinstance(dim, constexpr):
            dim = dim.value
        dim = int(dim)
        if dim < 0 or dim > 2:
            raise IndexError("tile_id dimension must be 0, 1 or 2")
        handle = _semantic.builder.create_clc_get_program_id(self._scheduler._clc_result.handle, dim)
        return tl.tensor(handle, tl.int32)


@tl._aggregate
class ClcTileScheduler:
    # The only carried state: the 128-bit CLC response for the current tile.
    _clc_result: _clc_result

    @builtin
    def is_valid(self, _semantic=None):
        """Whether the current tile holds real work. Use as the ``while`` condition."""
        handle = _semantic.builder.create_clc_is_canceled(self._clc_result.handle)
        return tl.tensor(handle, tl.int1)

    @builtin
    def advance(self, _semantic=None):
        """Fetch the next tile and return the updated scheduler."""
        return ClcTileScheduler(_clc_result(_semantic.builder.create_clc_advance()))


# `@_aggregate` only transfers methods (not properties) onto the value class, so
# attach the `tile_id` accessor property here.
ClcTileScheduler.tile_id = property(lambda self: _tile_id_accessor(self))


@builtin
def clc_tile_scheduler(_semantic=None):
    """Create a CLC tile scheduler for a dynamic persistent kernel (Blackwell).

    Seeds the scheduler with this CTA's own statically-launched tile; subsequent
    tiles are fetched via CLC work-stealing in ``advance``.
    """
    return ClcTileScheduler(_clc_result(_semantic.builder.create_clc_init()))
