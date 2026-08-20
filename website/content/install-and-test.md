## Build and install TLX from source

```
git clone https://github.com/facebookexperimental/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

Run the tutorials after the build finishes, e.g,
```
python third_party/tlx/tutorials/hopper_fa_ws_pipelined_pingpong_test.py
```

To run Blackwell GEMM tutorial kernels, you can use the following command:

## Change 2: One correctness test script

`[TLX_VERSION=<kernel_name>] pytest third_party/tlx/tutorials/testing/test_correctness.py`

By default only one autotune config will be used by correctness test.

All kernels — Hopper, Blackwell, and AMD — share this one file; each test is
arch-gated with `@pytest.mark.skipif`, so on any given GPU only the relevant
cases run and the rest skip. To run just the AMD/IKBO cases:

`pytest third_party/tlx/tutorials/testing/test_correctness.py -k "amd or ikbo"`

(on gfx950 the gfx1250-only GEMM cases skip automatically).

## Change 3: One performance test script per op × arch (Hopper, Blackwell, AMD)

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_hopper_gemm_perf.py [--version {ws|pipelined}]`

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_hopper_fa_perf.py [--version {ws|ws_pipelined|ws_pipelined_pingpong|ws_pipelined_pingpong_persistent}]`

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_blackwell_gemm_perf.py [--version {ws|pipelined|clc|2cta}]`

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_blackwell_fa_perf.py [--version {ws|ws_persistent|ws_pipelined|ws_pipelined_persistent|clc}]`

`denoise.sh` wraps AMD runs too (it applies NUMA pinning and runs the
benchmark; the GPU clock/power lock is NVIDIA-only and is simply skipped on AMD).
gfx950 / CDNA4:

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_gemm_perf.py [--version {warp_pipeline|pipelined}]`

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_fa_perf.py [--version {simple|prefetch|persistent|cluster}]`

Without `--version`, AMD FA perf runs `simple`, `prefetch`, and `persistent`; select
`cluster` explicitly because it is D=128-only.

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_addmm_glu_perf.py [--version {tlx_baseline|tlx_simple_async|tlx_optimized_async|tlx_optimized|tlx_persistent}]`

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_ikbo_fa_perf.py`  (IKBO Flash Attention)

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_ikbo_lce_perf.py`  (IKBO LCE — distinct op, not attention)

gfx1250:

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_mxfp_gemm_perf.py [--transpose-b]`

## TLX-AMD CI

AMD tutorial kernels are exercised by `.github/workflows/mi350.yml` on a gfx950
(MI350 / CDNA4) runner, mirroring the H100 job in `.github/workflows/h100.yml`:

- **`mi350-tlx-test`** — TLX unit tests (`python/test/unit/language/test_tlx_*.py`)
  plus the tutorial correctness suite
  (`third_party/tlx/tutorials/testing/test_correctness.py`). AMD and IKBO cases
  run; Hopper/Blackwell and gfx1250 cases auto-skip via the arch gates.
- **`mi350-meta-triton-test`** — TritonBench performance coverage (the AMD perf
  scripts above are for local runs; perf-regression tracking lives in TritonBench).

Both run on push, PR, and the nightly schedule; nightly failures are filed as
issues via `report-nightly-failure.yml`.
