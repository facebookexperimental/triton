# Route A sub-tiling experiment (2026-07-06)

Assets for the sub-tiling experiment described in
`third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/docs/SubTilingDesign.md`
(see its "Experiment log" section for the full findings and measurements).

- `fa_subtiled_experiment.py` — the source-level Route A vehicle: FA fwd
  with BLOCK_M=256 written as two 128-row sub-tiles sharing each
  iteration's K/V tiles, no warp specialization in source. Runs the
  default pipeline (correctness + optional `--bench`); with the modulo
  env set it is the input for schedule-graph dumps.
- `fa_subtiled_rau_generated.py` — raw sched2tlx output for the Rau
  schedule (II=2487, 6-WG ping-pong-shaped partition). KNOWN BROKEN:
  fails at launch. Kept as the diff baseline.
- `fa_subtiled_rau_handpatched.py` — the same kernel hand-patched to run
  (correctness PASS; 206.7 TFLOPS at (1,32,8192)). The diff against the
  raw file is the spec of the three emitter defect classes for
  multi-instance graphs; see the file header. Run directly:
  `python fa_subtiled_rau_handpatched.py [--small] [--bench]`.

Headline result: correct end-to-end sub-tiled solver kernel achieved,
but 206.7 TFLOPS vs the 665 plateau — the two sub-tile chains execute in
lockstep and contend on the per-SM TMEM port/SFU, which the latency
model does not price (no shared-across-WG engine). Next steps live in
SubTilingDesign.md.
