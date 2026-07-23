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
  (correctness PASS; **703–720 TFLOPS at (1,32,8192)** — above the
  665–666 single-tile plateau). The diff against the raw file is the
  spec of the emitter defect classes for multi-instance graphs plus the
  spill-inducing load-placement fix; see the file header. Run directly:
  `python fa_subtiled_rau_handpatched.py [--small] [--bench]`.

Headline result: the Route A acceptance criterion is MET — the
sub-tiled solver kernel beats every committed single-tile config. The
decisive fix was register live-range, not TMEM contention: the emitted
schedule hoisted each softmax WG's 128-reg acc tmem_load ~5000 cycles
ahead of its use, and the resulting ptxas spills inflated the iteration
3.1× (206.7 TFLOPS as emitted). Full breakdown, microbench numbers, and
the re-ranked model/emitter work items live in SubTilingDesign.md.
