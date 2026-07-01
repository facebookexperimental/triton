"""Cluster Launch Control (CLC) tile scheduler for core Triton.

CLC is a Blackwell (SM100+) hardware feature for dynamic persistent kernels: a
block that finishes early cancels a pending (not-yet-launched) cluster and takes
over its tile. This module exposes a minimal, opaque scheduler:

    sched = tl.clc_tile_scheduler()      # initialize
    while sched.is_valid():              # more work to steal?
        pid = sched.tile_id[0]           # current tile coord (x); [1]/[2] for y/z
        ... compute ...
        sched = sched.advance()          # fetch the next tile

The scheduler carries the *decoded* tile: an ``is_valid`` predicate and the
(x, y, z) program-id coordinates. The initial tile is simply this CTA's
``program_id`` (with ``is_valid = True``); every later tile comes from a single
high-level ``ttng.clc_advance`` op. The response buffer, mbarrier, phase, and the
issue/wait overlap are ALL handled by a later compiler lowering pass -- the kernel
author writes none of that plumbing. The API is standalone (it does not require
warp specialization).
"""
import triton.language.core as tl
from triton.language.core import builtin

__all__ = ["clc_tile_scheduler", "ClcTileScheduler"]


@tl._aggregate
class ClcTileScheduler:
    # Decoded current tile: validity + (x, y, z) program-id coordinates.
    _valid: tl.tensor
    _x: tl.tensor
    _y: tl.tensor
    _z: tl.tensor

    @builtin
    def is_valid(self, _semantic=None):
        """Whether the current tile holds real work. Use as the ``while`` condition."""
        return self._valid

    @builtin
    def advance(self, _semantic=None):
        """Fetch the next tile and return the updated scheduler."""
        is_valid, x, y, z = _semantic.builder.create_clc_advance()
        return ClcTileScheduler(tl.tensor(is_valid, tl.int1), tl.tensor(x, tl.int32), tl.tensor(y, tl.int32),
                                tl.tensor(z, tl.int32))


# `@_aggregate` only transfers methods (not properties) onto the value class, so
# attach the `tile_id` accessor property here. Returns a 3-tuple; index it as
# `tile_id[0|1|2]`.
ClcTileScheduler.tile_id = property(lambda self: tl.tuple([self._x, self._y, self._z]))


@builtin
def clc_tile_scheduler(_semantic=None):
    """Create a CLC tile scheduler for a dynamic persistent kernel (Blackwell).

    Seeds the scheduler with this CTA's own statically-launched tile
    (``program_id`` with ``is_valid = True``); subsequent tiles are fetched via
    CLC work-stealing in ``advance``.
    """
    b = _semantic.builder
    return ClcTileScheduler(
        tl.tensor(b.get_int1(True), tl.int1),
        tl.tensor(b.create_get_program_id(0), tl.int32),
        tl.tensor(b.create_get_program_id(1), tl.int32),
        tl.tensor(b.create_get_program_id(2), tl.int32),
    )
