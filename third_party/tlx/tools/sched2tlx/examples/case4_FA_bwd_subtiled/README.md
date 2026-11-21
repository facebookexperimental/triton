# Sub-tiled FA backward fixture

This fixture models the paper's sub-tiled backward pass. Each loop iteration
shares a 128-row K/V tile across two 64-row Q/dO sub-tiles. The two sub-tiles
have independent exponential and five-MMA chains, while their dK/dV updates
accumulate into the same output tile.

Run the dump binary on a B200 or GB200 from `fbsource/fbcode`:

```bash
FIXTURE="$PWD/../third-party/triton/beta/triton/third_party/tlx/tools/sched2tlx/examples/case4_FA_bwd_subtiled"
TRITON_ALWAYS_COMPILE=1 TRITON_USE_MODULO_SCHEDULE=rau \
MLIR_ENABLE_DUMP=fa_bwd_dkdv_subtiled \
MLIR_DUMP_PATH=/tmp/fa_bwd_subtiled.mlir \
TRITON_MODULO_DUMP_DDG="$FIXTURE/ddg.json" \
TRITON_MODULO_DUMP_SCHEDULE="$FIXTURE/schedule_graph.json" \
buck2 run @mode/opt -m ovr_config//triton:beta \
  -c fbcode.nvcc_arch=b200a \
  -c fbcode.platform010_cuda_version=12.8 \
  fbsource//third-party/triton/beta/triton:py_fa_bwd_subtiled_dump
```

The runner tolerates the known downstream AutoWS failure only after checking
that every requested compiler dump exists. Verify the defining DDG shape from
the Triton root:

```bash
python3 third_party/tlx/tools/sched2tlx/examples/case4_FA_bwd_subtiled/verify_fixture.py \
  third_party/tlx/tools/sched2tlx/examples/case4_FA_bwd_subtiled/ddg.json
```

`ddg.json` and `schedule_graph.json` are compiler output and carry the standard
`@generated` marker. The verifier requires exactly two loop `math.exp2` nodes
and ten loop `ttng.tc_gen5_mma` nodes.

Run the no-WS correctness path against torch autograd by adding `-- --check`
to the Buck command and omitting the dump environment variables.
