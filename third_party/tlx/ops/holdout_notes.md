# SM/CU holdout and green contexts in TLX ops

## Overview

TLX ops can leave SMs/CUs free for concurrent work, typically communication
overlap. TLX adds no setting of its own: you set the process-wide PyTorch
carveout once, and TLX launches respect it automatically. With no carveout
set, launches behave exactly as before.

## What you set

Set the holdout with `torch._C._set_sm_carveout_experimental(n)` to reserve
`n` SMs/CUs for other work. Any int up to `SM_COUNT − 1` is accepted (at
least one SM must remain); larger values are rejected. How many to hold
out is a model-level decision based on the overlap you need, often with
communication. The value is read on every launch, so changing it
mid-process (for example between phases) takes effect on the next launch
with no restart. There is one value per process, shared by all devices.

## What the ops do

- Persistent launches size their grid to the device count minus the
  holdout, floored at one program. Only tail waves are removed; the work
  per tile is unchanged. No extra stream is created.
- One-tile-per-program launches cannot shrink their grid without dropping
  work. On HIP they instead run confined to the remaining CUs; at holdout
  0 this is a no-op. CUDA has no equivalent confinement, so those launches
  are unchanged there.
- Launches whose grid is derived from the work itself (tiles, tasks) are
  unchanged: their size is correctness, not occupancy.

Grid sizing works from an expectation, not an exclusion: it assumes the
held-out SMs/CUs will be occupied by the concurrent work, so fewer
programs are launched to avoid oversubscribing the device. It does not
keep TLX programs off the held-out processors — a smaller grid's programs
may still be scheduled on any SM/CU, including held-out ones.

## NVIDIA: no guarantee, prefer green contexts

On NVIDIA there is no per-stream SM mask, so the carveout is not
recommended there and TLX does not guarantee the held-out SMs will be
unused. Persistent variants respect the count — they launch at most
`SM_COUNT − n` programs — but placement is still up to the hardware
scheduler, and non-persistent launches use the whole device. For a hard
reservation on NVIDIA, use green contexts instead.

Green-context support is opt-in, not automatic. A launch location that
runs on a green stream must ask for the stream check when sizing its
grid; the partition size then becomes the launch ceiling for that
location. Locations that do not opt in never query the stream (see the
overhead note below), and a green location that does not opt in will
oversubscribe its partition. If the partition cannot be queried, the
physical device count is used. The carveout remains a single global
reservation; it cannot express per-context reservations beyond the
partition itself. HIP has no green-context equivalent.

## Overhead

Sizing a persistent grid costs about 0.4 µs per launch on AMD: the
device count is cached, and the remaining work is reading the carveout
value (~0.1 µs) plus validation. The stream check adds about 1.3 µs on
CUDA, which is why it is opt-in. Caching the whole result per process
would reduce the cost to a dict lookup (~0.1 µs), but the carveout is
deliberately read on every launch so mid-process changes take effect;
only the static device properties are cached.

## Semantics and limits

This follows the documented intent — hold `n` out for others — matching
how Inductor sizes persistent grids, rather than mirroring PyTorch's HIP
BLAS mask, which currently confines BLAS to roughly the held-out set (the
inverse behavior, measured on MI350X). Autotuned config choices are not
re-picked when the carveout changes; only grid sizing reacts. MIG is
untested.
