# `tlx.ops.mm` on gfx942 — structure and invariants

Target guidance for `third_party/tlx/ops/kernels/mm/gfx942.py`. Read
`harnesses/gfx942/knowledge.md` first for the arch mechanism and the measurement
method. Current figures live in the run artifacts and
`python/test/tlx_benchmark/bench_mm.py`, not here.

## Invariants — changing these is a rewrite, not a tuning step

- **The operand path is register-staged, not direct-to-LDS:**
  `global --tl.load--> VGPR --tlx.local_store--> LDS --tlx.local_load--> MFMA`.
  The gfx950 tutorials all stage with `buffer_load_to_local`; that choice does
  not carry over to CDNA3. Treat the staging path as fixed unless you are
  deliberately testing it as the hypothesis.
- `matrix_instr_nonkdim=16`, and `num_stages=1` so the automatic software
  pipeliner bails out and leaves the manual LDS ring in charge. Raising
  `num_stages` re-enables a pipeliner that will fight the ring.
- `_prune_configs` rejects any config over the LDS budget before compilation.
  A candidate that widens a tile must keep that check honest rather than
  bypassing it.

## Structure of the search space

- **Per-shape tile selection dominates every other lever.** No single tile serves
  both ends of the range: at small outputs a wide tile decomposes into too few
  workgroups to fill the CUs, and at large outputs a narrow tile cannot feed the
  MFMAs. Any proposal that changes behaviour uniformly across shapes is
  attacking the smaller effect.
- **Ring depth trades against tile width for the same LDS bytes.** Single
  buffering wins on roughly half the measured shapes. Depth is not free
  headroom; deepening the ring means narrowing the tile.
- **The XCD remap and the GROUP_M swizzle target the same L2 reuse.** With the
  swizzle in place the remap measured neutral. Keep it as the standard grid
  transform, but do not propose it as an optimization.
- **Wide tiles win at 8 warps, narrow tiles at 4.** The warp count is coupled to
  the tile, not independently tunable.

## Scope — what the agent is actually measuring

`matmul` aliases `mm`, whose default `space="heuristic"` resolves to a **single
config per shape** from `_TILE_LADDER`, calibrated against a `space="full"`
sweep. Consequences:

- The gate is pinned by construction; a candidate cannot win by landing on a
  luckier tile, and there is no autotune search to control for.
- Agent numbers are not comparable to `space="full"` numbers. Do not cite one
  against the other.
- A candidate that changes `_TILE_LADDER` changes what every shape runs. That is
  a legitimate hypothesis, but it must be validated across the whole shape set,
  not the one shape that motivated it.

## Position relative to the gfx950 ladder

This kernel has the manual global-prefetch ring and the XCD remap, and none of
the loop-unroll, N-slice or warp-pipeline steps — roughly v4/v5 plus v9 of the
ladder in `harnesses/gfx942/knowledge.md` §3. That is the most concrete
statement available about where the headroom is, and it stays a hypothesis until
one of those steps is measured on this part.
