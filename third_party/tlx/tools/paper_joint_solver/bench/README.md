# FMHA bar benchmarks (paper replication)

Bar-based benchmark harness replicating the paper's Blackwell FMHA evaluation.

Config: fp16, non-causal, BATCH=4, NUM_HEADS=32, HEAD_DIM=128, seqlens
2048 / 4096 / 8192 / 16384.  FLOPs use the FA convention shared with
`third_party/tlx/tutorials/testing/test_blackwell_fa_perf.py`
(flops_per_matmul = 2·B·H·S²·D; fwd = 2×, bwd = 5×, i.e. fwd = 4·B·H·S²·D and
bwd = 10·B·H·S²·D).

## Running

From `third_party/tlx/tools/paper_joint_solver`, with the main venv python and
`LD_LIBRARY_PATH` unset:

```bash
env -u LD_LIBRARY_PATH ../../../../.venv/bin/python -m bench.bench_bars \
    --mode fwd \
    --bars triton_ws_off,triton_ws_on,triton_tiled,tlx_default,jos,cudnn,fa4 \
    --out bench/results_fwd.json

env -u LD_LIBRARY_PATH ../../../../.venv/bin/python -m bench.bench_bars \
    --mode bwd \
    --bars cudnn_bwd,fa4_bwd,tlx_bwd_default,jos_bwd \
    --out bench/results_bwd.json
```

`--seqlens 2048,4096` restricts the sweep; `--bars` defaults to every bar for
the selected mode.  Results accumulate into `--out`
(`{bar: {seqlen: {tflops, ms, lo, hi, ok, ...}}}`) — existing entries for
other bars/seqlens are preserved, so partial reruns are fine.

Timing: `triton.testing.do_bench(fn, warmup=500, rep=500,
quantiles=[0.5, 0.2, 0.8])`.  The JSON records the median ms (`ms`), the
q20/q80 ms (`lo`/`hi`), and the median TFLOPS.  Every bar passes a correctness
gate before timing (fwd rel err < 1e-2 vs `torch.nn.functional.
scaled_dot_product_attention`; bwd grad rel err < 3e-2); a failed gate or an
unavailable dependency is recorded as a skip with a reason.

## Bars

Forward (`--mode fwd`):

| bar | what it runs |
|---|---|
| `triton_ws_off` | `python/tutorials/06-fused-attention.py` forward, fp16, causal=False, `warp_specialize=False` (output layout Z,H,N,D) |
| `triton_ws_on` | same, `warp_specialize=True` |
| `triton_tiled` | plain sub-tiled kernel `sched2tlx/examples/case3_FA_fp16_subtiled/fa_fwd_nows_subtiled.py` (SUB_M=64; its own launch recipe: num_warps=4, num_stages=2, maxRegAutoWS=152; grid cdiv(N,128)) |
| `tlx_default` | emitted baseline-schedule kernel `case3_FA_fp16/generated.py::fa_fwd_kernel_nows`, tensor prep per that dir's `fa_fwd_nows_fp16.py run()` (flattened [Z·H·N, D] tensors, grid (cdiv(N,128), Z·H)) |
| `jos` | solver-schedule kernel, default `case3_FA_fp16_subtiled/generated.py::fa_fwd_kernel_nows_subtiled`, same launch shape, grid cdiv(N,128).  Point `--jos-file` at a regenerated file (kernel name must stay `fa_fwd_kernel_nows_subtiled`) |
| `cudnn` | torch SDPA forced to `SDPBackend.CUDNN_ATTENTION`, inputs (B,H,S,D) fp16 |
| `fa4` | `flash_attn.cute.interface.flash_attn_func` via `bench/fa4_worker.py` under the separate venv `/projects/kzhou6/hwu27/baselines/.venv-fa4` (layout B,S,H,D) |

Backward (`--mode bwd`):

| bar | what it runs |
|---|---|
| `cudnn_bwd` | SDPA-cuDNN full fwd+bwd minus fwd (see deviations) |
| `fa4_bwd` | FA4 full fwd+bwd minus fwd, in the worker |
| `tlx_bwd_default` | emitted kernel `case4_FA_bwd/generated_hd128.py::fa_bwd_dkdv_5mma` with the M/D preprocessing recipe of `case4_FA_bwd/run_handwritten_nows.py` |
| `jos_bwd` | same loader; default `case4_FA_bwd/generated_hd128_jos.py` (does not exist until the solver regenerates it) — override with `--jos-bwd-file`; kernel name `fa_bwd_dkdv_5mma` |

## Bar → paper figure mapping

Neutral names for the paper's Blackwell FMHA figures:

| bench bar | paper figure bar |
|---|---|
| `triton_ws_off` | Triton (warp specialization off) — forward figure |
| `triton_ws_on` | Triton-WS — forward figure |
| `triton_tiled` | Triton at the JOS sub-tile granularity (tiling-vs-scheduling control; not a separate paper bar) |
| `tlx_default` | TLX-Default — forward figure |
| `jos` | JOS — forward figure |
| `cudnn` | cuDNN — forward figure |
| `fa4` | FA4 — forward figure |
| `cudnn_bwd` | cuDNN — backward figure |
| `fa4_bwd` | FA4 — backward figure |
| `tlx_bwd_default` | TLX-Default — backward figure |
| `jos_bwd` | JOS — backward figure |

## Deviations from the paper's setup

1. **No clock locking.** Locking SM clocks needs root, which we do not have.
   Instead `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu` is
   recorded before/after each bar into the JSON as `env_probe`; discard
   measurements whose clocks drifted between the two probes.
2. **Backward timing method (cudnn/fa4).** FA-style repos time the backward
   standalone from a saved forward context.  Here the full fwd+bwd is timed,
   the forward alone is timed (autograd graph still built), and
   (total − fwd median) is reported; `lo`/`hi` are the q20/q80 of the fwd+bwd
   distribution shifted by the fwd median.  The fwd median is stored as
   `fwd_ms`.
3. **TLX backward bars time the fused kernel only.**  `tlx_bwd_default` /
   `jos_bwd` time the dK/dV/dQ kernel launch; M (base-2 logsumexp) and
   D (rowsum(dO·O)) are precomputed on the host per
   `run_handwritten_nows.py`'s base-2 convention: sm_scale is pre-folded into
   Q and the kernel applies no softmax scale.  The preprocessing is excluded
   from the timed region.  dQ accumulates via TMA reduce-add, so timed
   repetitions re-accumulate into a stale buffer; correctness is checked on a
   freshly zeroed dQ before timing.
4. **Generated-kernel launch options.**  Emitted TLX kernels launch with
   num_warps=4, num_ctas=1, num_stages=1 (following
   `case4_FA_bwd/run_generated.py`) since pipelining/multibuffering is explicit
   in the emitted schedule; the plain `triton_tiled` kernel keeps its source
   recipe (num_stages=2, maxRegAutoWS=152).
5. **Correctness references.**  Forward bars gate against fp16 SDPA
   (auto backend); `cudnn_bwd` grads gate against default-backend SDPA
   autograd; the FA4 worker gates against fp32 SDPA; TLX backward bars gate
   against autograd of SDPA(scale=1.0) on the pre-scaled Q (the base-2
   reference model).  The TLX bwd reference runs SDPA in 4D `(BH, 1, S, D)` —
   3D inputs silently select the math backend (O(S²) memory).
6. **`triton_ws_on` pipeline.**  On this beta build the stock AutoWS pipeline
   fails to compile every tutorial ws=on config (`'ttng.tmem_alloc' op
   operation destroyed but still has uses`), so the bar sets
   `TRITON_USE_META_WS=1` and prunes the autotune list to the B200-verified
   configs in `_WS_SAFE_CONFIGS` (BLOCK_M=128, BLOCK_N∈{64,32},
   num_stages∈{3,4}, num_warps=8).  `triton_ws_off` keeps the stock pipeline
   and full config sweep.  See bench/RESULTS.md for the probe details.
