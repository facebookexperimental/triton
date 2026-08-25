## Correctness

`[TLX_VERSION=<kernel_name>] pytest third_party/tlx/tutorials/testing/test_correctness.py`

By default only one autotune config will be used by correctness test.

All kernels — Hopper, Blackwell, and AMD — share this one file; each test is
arch-gated with `@pytest.mark.skipif`, so on any given GPU only the relevant
cases run and the rest skip. To run just the AMD/IKBO cases:

`pytest third_party/tlx/tutorials/testing/test_correctness.py -k "amd or ikbo"`

(on gfx950 the gfx1250-only GEMM cases skip automatically).

## Performance

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
