# Historical FMHA bar benchmark harness

This harness records local Triton, sched2tlx/TLX, cuDNN, and FA4 comparisons. It
does not replicate Twill §6.1.

Config: fp16, non-causal, BATCH=4, NUM_HEADS=32, HEAD_DIM=128, seqlens
2048 / 4096 / 8192 / 16384.  FLOPs use the FA convention shared with
`third_party/tlx/tutorials/testing/test_blackwell_fa_perf.py`
(flops_per_matmul = 2·B·H·S²·D; fwd = 2×, bwd = 5×, i.e. fwd = 4·B·H·S²·D and
bwd = 10·B·H·S²·D).

Twill §5 emits software-pipelined, warp-annotated IR for manual implementation.
The paper's §6.1 results use expert hand-compiled CUDA C++; automatic memory
allocation, layout, and synchronization lowering are out of scope. The current
paper-faithful path stops at the manual-CUDA handoff IR, and this checkout has no
paper handwritten CUDA source. Paper performance is therefore unreplicated.
TLX appears only in related work. All sched2tlx/TLX bars below are historical
non-paper experiments, not Twill/SKC codegen or performance evidence.

## Running

From `third_party/tlx/tools/paper_joint_solver`, with the main venv python and
`LD_LIBRARY_PATH` unset:

```bash
env -u LD_LIBRARY_PATH ../../../../.venv/bin/python -m bench.bench_bars \
    --mode fwd \
    --bars triton_ws_off,triton_ws_on,triton_tiled,tlx_default,cudnn,fa4 \
    --out bench/results_fwd.json

env -u LD_LIBRARY_PATH ../../../../.venv/bin/python -m bench.bench_bars \
    --mode bwd \
    --bars cudnn_bwd,fa4_bwd,tlx_bwd_default \
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

The former `jos`, `jos_bwd`, and TLX-skeleton entries were removed from the
registry so the current harness cannot execute them as Twill results. Their
old JSON keys remain only in archived result files.

Forward (`--mode fwd`):

| bar | what it runs |
|---|---|
| `triton_ws_off` | `python/tutorials/06-fused-attention.py` forward, fp16, causal=False, `warp_specialize=False` (output layout Z,H,N,D) |
| `triton_ws_on` | same, `warp_specialize=True` |
| `triton_tiled` | plain sub-tiled kernel `sched2tlx/examples/case3_FA_fp16_subtiled/fa_fwd_nows_subtiled.py` (SUB_M=64; its own launch recipe: num_warps=4, num_stages=2, maxRegAutoWS=152; grid cdiv(N,128)) |
| `tlx_default` | historical sched2tlx-emitted TLX baseline `case3_FA_fp16/generated.py::fa_fwd_kernel_nows`, tensor prep per that dir's `fa_fwd_nows_fp16.py run()` (flattened [Z·H·N, D] tensors, grid (cdiv(N,128), Z·H)) |
| `cudnn` | torch SDPA forced to `SDPBackend.CUDNN_ATTENTION`, inputs (B,H,S,D) fp16 |
| `fa4` | `flash_attn.cute.interface.flash_attn_func` via `bench/fa4_worker.py` under the separate FA4 venv (env `FA4_PYTHON`) (layout B,S,H,D) |

Backward (`--mode bwd`):

| bar | what it runs |
|---|---|
| `cudnn_bwd` | SDPA-cuDNN full fwd+bwd minus fwd (see limitations) |
| `fa4_bwd` | FA4 full fwd+bwd minus fwd, in the worker |
| `tlx_bwd_default` | historical sched2tlx-emitted TLX kernel `case4_FA_bwd/generated_hd128.py::fa_bwd_dkdv_5mma` with the M/D preprocessing recipe of `case4_FA_bwd/run_handwritten_nows.py` |

## Local bar provenance

There is no bar-to-paper-figure mapping. In particular, TLX is not a Twill
backend and the legacy `jos` bars do not stand in for the paper's hand-written
CUDA implementation.

| bench bars | provenance |
|---|---|
| `triton_ws_off`, `triton_ws_on`, `triton_tiled` | local Triton baselines and controls |
| `tlx_default`, `tlx_bwd_default` | historical sched2tlx/TLX baselines |
| `cudnn`, `cudnn_bwd` | local cuDNN baselines |
| `fa4`, `fa4_bwd` | local FA4 baselines |

## Local benchmark methodology and limitations

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
3. **TLX backward bars time the fused kernel only.**  `tlx_bwd_default` times
   the dK/dV/dQ kernel launch; M (base-2 logsumexp) and
   D (rowsum(dO·O)) are precomputed on the host per
   `run_handwritten_nows.py`'s base-2 convention: sm_scale is pre-folded into
   Q and the kernel applies no softmax scale.  The preprocessing is excluded
   from the timed region.  dQ accumulates via TMA reduce-add, so timed
   repetitions re-accumulate into a stale buffer; correctness is checked on a
   freshly zeroed dQ before timing.
4. **Historical generated-kernel launch options.**  The sched2tlx-emitted TLX
   experiments launch with
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
   and full config sweep. This is a local diagnostic baseline, not a
   reproduction of the paper's Triton-WS result. See bench/RESULTS.md for the
   probe details.
