# GEO OSS adapter

`geo_oss_adapter.py` lets GEO's non-GEO/custom-kernel workflow evaluate a
workspace with an OSS Triton checkout instead of building Triton and PyTorch
with Buck.

The adapter materializes the workspace's Buck `base_module` mappings as a
temporary Python package under `<workspace>/.oss_adapter/`. Its narrow `buck2`
shim intercepts:

- workspace `:test`, `:bench`, and `:repro` targets;
- `kernel_optimizer_setup_tool -- validate`.

All other Buck commands are delegated to `GEO_OSS_REAL_BUCK2` (default:
`/usr/local/bin/buck2`). Candidate edits therefore use the selected OSS
Python/Triton installation and normal Triton JIT caching; they do not rebuild
the compiler.

Run a GEO command through the adapter environment:

```bash
export GEO_OSS_PYTHON=/path/to/oss/triton/python
export GEO_OSS_FBCODE_ROOT=/path/to/fbsource/fbcode

third_party/tlx/tools/agents/with_geo_oss_adapter \
  /path/to/geo/python -m gem.next_gen.geo.agents.kernel_optimizer.resume \
  --workspace /path/to/workspace --gpu 0 --retry-setup
```

The GEO Python environment needs the GEO packages on `PYTHONPATH`. The OSS
Python environment needs the selected Triton checkout installed or on
`PYTHONPATH`. Performance runs should still be wrapped by the standard GPU
denoising command required by GEO.

The driver can also be used directly:

```bash
$GEO_OSS_PYTHON third_party/tlx/tools/agents/geo_oss_adapter.py test \
  --workspace /path/to/workspace

$GEO_OSS_PYTHON third_party/tlx/tools/agents/geo_oss_adapter.py validate \
  --workspace /path/to/workspace \
  --oss-root /path/to/triton \
  --fbcode-root "$GEO_OSS_FBCODE_ROOT" --gpu 0
```
