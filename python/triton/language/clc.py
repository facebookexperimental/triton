"""Cluster Launch Control (CLC) tile scheduler for core Triton.

CLC is a Blackwell (SM100+) hardware feature for dynamic persistent kernels: a
block that finishes early cancels a pending (not-yet-launched) cluster and takes
over its tile. This module exposes a minimal, opaque scheduler:

    sched = tl.clc_tile_scheduler()      # initialize
    while sched.is_valid():              # more work to steal?
        pid = sched.tile_id[0]           # current tile coord (x); [1]/[2] for y/z
        ... compute ...
        sched = sched.advance()          # fetch the next tile

``tl.clc_tile_scheduler(num_stealing_ctas=N)`` lets only the first N CTAs steal
(e.g. N = SM count minus the SMs a concurrent NCCL kernel needs); the others run
their launched tile and exit, so at most N SMs stay occupied for more than one
tile at a time.

The scheduler carries the *decoded* tile: an ``is_valid`` predicate and the
(x, y, z) program-id coordinates. The initial tile is simply this CTA's
``program_id`` (with ``is_valid = True``); every later tile comes from a single
high-level ``ttng.clc_advance`` op. The response buffer, mbarrier, phase, and the
issue/wait overlap are ALL handled by a later compiler lowering pass -- the kernel
author writes none of that plumbing. The API is standalone (it does not require
warp specialization).

``ClcTileScheduler`` now lives in ``triton.language.schedule`` as one of the
unified tile schedulers (all share the ``is_valid`` / ``tile_id`` / ``advance``
API); it is re-exported here, and ``clc_tile_scheduler`` remains as the
convenience factory.
"""
from triton.language.core import builtin
from triton.language.schedule import ClcTileScheduler

__all__ = ["clc_tile_scheduler", "ClcTileScheduler"]


@builtin
def clc_tile_scheduler(num_stealing_ctas=None, _semantic=None):
    """Create a CLC tile scheduler for a dynamic persistent kernel (Blackwell).

    Seeds the scheduler with this CTA's own statically-launched tile
    (``program_id`` with ``is_valid = True``); subsequent tiles are fetched via
    CLC work-stealing in ``advance``.

    ``num_stealing_ctas`` (constexpr int) limits work-stealing to the first that
    many CTAs (whole clusters); the rest run one tile and exit. See
    ``ClcTileScheduler``.
    """
    return ClcTileScheduler.initialize(None, None, None, num_stealing_ctas, _semantic=_semantic)
