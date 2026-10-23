# SM/CU holdout in TLX ops

Persistent TLX grids withhold SMs/CUs for concurrent work (typically NCCL
overlap) via `tlx.ops._holdout`. The count comes from the process-wide
PyTorch carveout (`torch._C._get_sm_carveout_experimental`, set with
`torch._C._set_sm_carveout_experimental`); 0 (or no API) keeps historical
behavior. Any non-negative int is accepted (overlap use cases hold out 1-2
SMs); tuning sweeps use multiples of 16 by convention. This mirrors
inductor, which sizes persistent grids as `max_sms - carveout`
(`torch._inductor.utils.get_num_sms`).

## How launchers consume it

- `cap_grid(grid)` subtracts the live carveout, floored at one program. All
  capped grids are `min(sm_count, tiles)`-style, so capping only removes
  tail waves; it never changes the per-tile work.
- The gfx950 dynamic mm caps its persistent pool before sizing the grid
  (`launch_cus = cap_grid(num_sms or 256)`, then `min(launch_cus, tiles)`).
- The gfx950 greedy mm caps only the persistent tail
  (`persistent_launch = cap_grid(num_persistent)`). The one-shot head count
  (`holdout_cus`, a multiple of 16) is unrelated to this carveout despite
  the name: it sizes how many programs grab once and exit, and is not
  reduced by it.
- Non-persistent static grids cannot shrink, so under a carveout the gfx950
  mm static dispatch (`_launch_dispatch`) runs on a CU-masked stream that
  drops the first-K holdout CUs (`carveout_stream`: masked stream created
  via the HIP library PyTorch loads — PyTorch exposes no Python API for
  this — wrapped with `ExternalStream`, pre/post event sync against the
  ambient stream; verified on MI350X including mask readback). Other non-persistent
  full-tile grids (gfx942 mm, sm100 mm/bwd full grids, addmm full-grid
  paths) launch unmasked — CUDA has no CU-mask equivalent; `cu_mask_words`
  computes the mask for the HIP ones.

## Coverage

Capped persistent grids: mm sm90/sm100 (incl. gfx950 dynamic and greedy),
addmm sm90/sm100, grouped_gemm sm100/gfx950, grouped_gemm_mxfp8
sm100/gfx950, mm_mxfp8 sm100, mm_bias_grad sm100, kda sm100 (fwd/bwd),
hstu_attn sm100, flash_attn sm90 and mxfp8 sm100, flash_attn_varlen +
hstu_cross_attn gfx950 (split planners), flash_attn gfx950 cluster
(rounded down to an XCD multiple, floored at `NUM_XCDS`).

Skipped deliberately: hstu_attn gfx950 XCD grouping and the gfx950 FA bwd
dispatch selector (physical-topology heuristics, not grid caps).

## The carveout is live, the device count is cached

`holdout_sms()` reads `torch._C` on every call, and every grid cap goes
through it at launch time, so setting or changing the carveout mid-process
(e.g. set after warmup, toggled between phases) takes effect on the next
launch with no cache flush. Only the static device query is cached: the
per-op SM-count helpers (`_get_num_sms`, `_num_sms`, `_num_cus`) cache
`get_device_properties` (keyed per device where applicable) and apply
`cap_grid` outside the cache. Residual
gap: already-autotuned config choices (the `_tuned_*` lru caches) are not
re-tuned when the carveout changes; only grid sizing reacts.

## PyTorch semantics and the HIP caveat

On CUDA, cuBLAS honors the carveout via
`cublasSetSmCountTarget(smCount - carveout)`; on HIP it runs BLAS on a
CU-masked stream (`hipExtStreamCreateWithCUMask`); inductor sizes persistent
grids as `max_sms - carveout`, which TLX mirrors. Measured on MI350X (4096²
fp16 matmul): carveout None/16/128/240 ran 0.11/1.15/0.23/0.23 ms —
larger values run *faster*, i.e. the HIP mask construction currently
confines BLAS to roughly the value CUs, the inverse of the CUDA semantic
and of inductor (plus a shift-by-32 edge at multiples of 32). TLX follows
the documented intent (hold N out for others) everywhere and builds its
masks per the HIP docs (active bit = CU used, low bit = low CU), not
mirroring PyTorch's construction.

## Green contexts (Nvidia)

`cap_grid` queries the launch stream's green partition
(`cuStreamGetGreenCtx` → SM count, via the driver bindings PyTorch loads)
and caps the grid at the partition size, so persistent grids no longer
oversubscribe a green-context stream. Regular streams are unaffected
(only `ExternalStream`s pay the driver query; any failure degrades to
the physical count). The carveout still cannot express per-context
reservations beyond the partition itself (single process-global int),
and HIP has no green-context equivalent.

## Not supported yet

- MIG: untested against TLX grids and masks.
- Multi-device: the carveout lives in `at::globalContext`, one value shared
  by all devices; per-device reservations are not expressible.
- Mask generality: drop-first-K only. TODO: generalize the carveout mask to
  arbitrary masks selectable per stream (holdout set vs complement,
  XCD-aligned subsets). A mask restricts *which* CUs run blocks, not how
  many, so each selection still needs a distribution rule for spreading
  consecutive programs across allowed CUs.
