"""Unified tile-scheduler stdlib for core Triton.

A persistent kernel iterates over output *tiles*. Which tile a program processes,
how it advances to the next one, and how it knows when there is no more work all
depend on the *schedule*. Historically each schedule (non-persistent,
static-persistent, dynamic/atomic work-stealing, CLC) was hand-written inline in
every kernel with a subtly different loop shape.

This module unifies them behind one opaque loop API so the loop body is identical
regardless of schedule -- which lets the schedule be chosen as an autotunable
``constexpr``:

    sched = SCHEDULE.initialize(lowering_args, num_tiles_fn, tile_counter)
    while sched.is_valid():                 # more work?
        tile_id = sched.tile_id[0]          # current linear tile id ([1]/[2] = y/z)
        ... compute for this tile ...
        sched = sched.advance()             # claim the next tile

``SCHEDULE`` is one of the concrete scheduler classes below, passed to the kernel
as a ``tl.constexpr`` (autotuning simply varies which class is passed).

The schedule is *not* handed a precomputed tile count (that would not specialize
per autotune config). Instead it carries ``lowering_args`` -- the variables any
schedule variation needs (e.g. ``(M, N, BLOCK_M, BLOCK_N)``) -- and a user
``num_tiles_fn`` that computes the tile count from them. ``is_valid`` computes
``num_tiles`` in-kernel over the *tuned* block-size constexprs; because that is
pure loop-invariant integer arithmetic, CSE + LICM hoist it out of the loop, so
recomputing it each iteration is free.

The schedulers are ``@tl._aggregate`` value types, so instances flow through
``@triton.jit`` kernels (including the persistent ``scf.while`` loop-carried
state). ``TileScheduler`` is a plain base holding the shared ``is_valid`` body;
each concrete class re-declares its own fields (``@_aggregate`` reads only a
class's own annotations) and gets the ``tile_id`` accessor attached below.
"""
import triton.language.core as tl
from triton.language.core import builtin
from triton.runtime.jit import jit

__all__ = [
    "TileScheduler",
    "NonPersistentScheduler",
    "StaticPersistent1DScheduler",
    "DynamicPersistent1DScheduler",
    "ClcTileScheduler",
]


class TileScheduler:
    """Base tile-scheduler contract.

    All schedulers MUST implement the same API:
      * ``initialize(lowering_args, num_tiles_fn, tile_counter)`` -- construct the
        scheduler seeded with this program's first tile. Unused arguments (e.g.
        ``tile_counter`` for non-dynamic schedules) are ignored, so the call site
        is identical for every schedule.
      * ``is_valid()`` -- while-loop condition: does the current tile hold work?
      * ``tile_id`` -- 3-tuple ``(x, y, z)`` of the current tile's coordinates;
        read ``tile_id[0]`` for the linear id.
      * ``advance()`` -- claim the next tile and return the updated scheduler.

    This base provides no working ``is_valid`` -- there is no schedule-agnostic
    default (validity is hardware-driven for CLC, single-shot for non-persistent,
    and count-based for the persistent linear schedules). A subclass that fails to
    implement it fails at compile time via the ``static_assert`` below.
    """

    @jit
    def is_valid(self):
        tl.static_assert(False, "TileScheduler subclasses must implement is_valid()")
        return False


class _CountingTileScheduler(TileScheduler):
    """Base for count-limited persistent schedules (static / dynamic).

    ``is_valid`` recomputes ``num_tiles`` from the stored ``num_tiles_fn`` +
    ``lowering_args`` and tests the current tile against it. Recomputing every
    iteration is free: it is pure loop-invariant integer arithmetic that CSE + LICM
    hoist out of the ``scf.while``.
    """

    @jit
    def is_valid(self):
        num_tiles = self._num_tiles_fn(self._lowering_args)
        return self._x < num_tiles


def _attach_tile_id(value_cls):
    # `@_aggregate` transfers methods but not properties, so attach `tile_id`
    # onto each concrete value class. Returns a 3-tuple; index as tile_id[0|1|2].
    value_cls.tile_id = property(lambda self: tl.tuple([self._x, self._y, self._z]))
    return value_cls


@tl._aggregate
class NonPersistentScheduler(TileScheduler):
    """One tile per program (grid == num_tiles). The loop body runs exactly once.

    ``advance`` flips ``_valid`` to False so the shared ``while sched.is_valid()``
    loop terminates after the single tile.
    """
    _valid: tl.tensor
    _x: tl.tensor
    _y: tl.tensor
    _z: tl.tensor

    @jit
    def initialize(lowering_args, num_tiles_fn, tile_counter):
        return NonPersistentScheduler(tl.to_tensor(True), tl.program_id(0), tl.to_tensor(0), tl.to_tensor(0))

    @jit
    def is_valid(self):
        return self._valid

    @jit
    def advance(self):
        return NonPersistentScheduler(tl.to_tensor(False), self._x, self._y, self._z)


_attach_tile_id(NonPersistentScheduler)


@tl._aggregate
class StaticPersistent1DScheduler(_CountingTileScheduler):
    """Static persistent: grid == min(num_programs, num_tiles); each program walks
    tiles ``pid, pid + num_programs, pid + 2*num_programs, ...`` until the count is
    exhausted. ``is_valid`` is the count-based one from ``_CountingTileScheduler``.
    """
    _x: tl.tensor
    _y: tl.tensor
    _z: tl.tensor
    _stride: tl.tensor
    _lowering_args: tl.tuple
    _num_tiles_fn: tl.constexpr

    @jit
    def initialize(lowering_args, num_tiles_fn, tile_counter):
        return StaticPersistent1DScheduler(tl.program_id(0), tl.to_tensor(0), tl.to_tensor(0), tl.num_programs(0),
                                           lowering_args, num_tiles_fn)

    @jit
    def advance(self):
        return StaticPersistent1DScheduler(self._x + self._stride, self._y, self._z, self._stride, self._lowering_args,
                                           self._num_tiles_fn)


_attach_tile_id(StaticPersistent1DScheduler)


@tl._aggregate
class DynamicPersistent1DScheduler(_CountingTileScheduler):
    """Dynamic (work-stealing) persistent: grid == min(num_programs, num_tiles);
    each program claims the next tile from a shared global atomic counter, so tiles
    are distributed by demand rather than a fixed stride. The host allocates
    ``tile_counter`` (a 1-element int32 device tensor) initialized to the number of
    launched programs. ``is_valid`` is the count-based one from ``_CountingTileScheduler``.
    """
    _x: tl.tensor
    _y: tl.tensor
    _z: tl.tensor
    _counter: tl.tensor
    _lowering_args: tl.tuple
    _num_tiles_fn: tl.constexpr

    @jit
    def initialize(lowering_args, num_tiles_fn, tile_counter):
        return DynamicPersistent1DScheduler(tl.program_id(0), tl.to_tensor(0), tl.to_tensor(0), tile_counter,
                                            lowering_args, num_tiles_fn)

    @jit
    def advance(self):
        next_tile = tl.atomic_add(self._counter, 1)
        return DynamicPersistent1DScheduler(next_tile, self._y, self._z, self._counter, self._lowering_args,
                                            self._num_tiles_fn)


_attach_tile_id(DynamicPersistent1DScheduler)


@tl._aggregate
class ClcTileScheduler(TileScheduler):
    """Cluster Launch Control (Blackwell SM100+) dynamic persistent schedule.

    The grid is over-subscribed with one CTA per logical tile. With
    ``ctas_per_cga``, PTX groups adjacent CTA coordinates into indivisible
    clusters: one request atomically cancels a pending cluster, and each member
    derives its program id from the returned first CTA coordinate plus its local
    cluster coordinate. The scheduler carries the *decoded* current tile
    ``(is_valid, x, y, z)``: the seed is this program's static launch id, and
    every later tile comes from a single high-level
    ``ttng.clc_advance`` op (the response buffer, mbarrier, phase, and issue/wait
    overlap are all materialized by a later compiler lowering pass). ``num_tiles_fn``
    / ``tile_counter`` are ignored -- CLC termination is hardware-driven. Grid
    dimensions and logical tile geometry must be divisible by
    ``ctas_per_cga``; inactive padded CTAs are unsupported, and all members must
    reach ``advance`` uniformly.
    """
    _valid: tl.tensor
    _x: tl.tensor
    _y: tl.tensor
    _z: tl.tensor

    @builtin
    def initialize(lowering_args, num_tiles_fn, tile_counter, _semantic=None):
        b = _semantic.builder
        return ClcTileScheduler(
            tl.tensor(b.get_int1(True), tl.int1),
            tl.tensor(b.create_get_program_id(0), tl.int32),
            tl.tensor(b.create_get_program_id(1), tl.int32),
            tl.tensor(b.create_get_program_id(2), tl.int32),
        )

    @jit
    def is_valid(self):
        return self._valid

    @builtin
    def advance(self, _semantic=None):
        is_valid, x, y, z = _semantic.builder.create_clc_advance()
        return ClcTileScheduler(tl.tensor(is_valid, tl.int1), tl.tensor(x, tl.int32), tl.tensor(y, tl.int32),
                                tl.tensor(z, tl.int32))


_attach_tile_id(ClcTileScheduler)
