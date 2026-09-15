# TLX op perf suite

Performance and compile-time guardrails for the blessed `tlx.ops` catalog.
Correctness lives in `python/test/unit/tlx_ops/`; this suite assumes it passes.

Depends on `torch` and `triton` only. No dependency on `tritonbench`

## Running

One shot command with extreme simplicity

python python/test/tlx_benchmark/bench_{op}.py

`{op}` is one of `mm`, `flash_attn`, `hstu_attn`, `kda`, `kda_prefill`,
`kda_decode`.

```
options:
  -h, --help            show this help message and exit
  --device DEVICE       GPU index, or 'auto' (default) for the least-used one
  --space {heuristic,full,smoke}
                        autotune search space; the default is each op's own, and measuring anything
                        else measures a path users do not take
  --head N              only the first N cases PER DIRECTION, for a quick look; on an op with a
                        backward, --head 10 is 10 fwd and 10 bwd
  --synthetic           run the correctness shapes instead of this arch's focus list; they are
                        mostly too small to time, so this is for looking, not for gating
  --fwd-only            skip the backward cases
  --bwd-only            skip the forward cases
  --latency-measure-mode {wallclock,gpu_events}
                        'wallclock' (default) times each call as a caller sees it; 'gpu_events'
                        pre-enqueues the batch behind a blocked stream to isolate device time
  --cold-compile {all,first,none}
                        how often to time a first call on a fresh cache; the default is each
                        op's own (mm: all, everything else: first)
  --json JSON           machine-readable artifact (default /tmp/tlx_benchmark/<op>.<arch>.json)
```

- built in (default on) denoise (freq-lock)
- built in (default on) GPU selection (least used)
- built in (default on) kernel selection by gpu arch

## Ops

| op | reference | gate | default space |
|----|-----------|------|---------------|
| `mm` | `torch.matmul` | speedup >= 0.9x | heuristic |
| `flash_attn` | `F.scaled_dot_product_attention` | speedup >= 0.9x | full |
| `hstu_attn` | `_reference.py::triton_hstu_mha` (production Triton) | speedup >= 0.9x | full |
| `kda` | none | absolute floor, currently unset -> reports only | full |
| `kda_prefill` | none | absolute floor, currently unset -> reports only | heuristic |
| `kda_decode` | none | absolute floor, currently unset -> reports only | heuristic |

Only `mm` has a `heuristic_config`, so it is the only op whose default is a
single analytically chosen config. The rest autotune a full space on their first
call, which is minutes rather than seconds; they raise `cap_s` accordingly rather
than measure a `smoke` space no user takes. Writing a `heuristic_config` for each
is the real fix and is tracked in the `tlx.ops` module docstring.

There is no vendor library for SiLU-scaled ragged attention, so `hstu_attn`
races the Triton kernel that ships today. The two are not tuned symmetrically --
the reference autotunes its full space regardless of `--space` -- which is
recorded per case as `ref_autotuned` rather than corrected for.

Ops with a backward (`flash_attn`, `hstu_attn`, `kda`) produce two cases per
shape. The backward is a different kernel at 2.5x the FLOPs, so folding it into
the forward's number would hide it. Both run by default and are reported as two
tables, `[fwd]` then `[bwd]`, under one summary and one artifact -- two
different amounts of work do not belong in one TFLOP/s column. `--fwd-only` /
`--bwd-only` skip a half.

## Metric report

1. Input info (varying between ops), e.g. for mm:
   `((), {'dtype': 'bf16', 'strides': '[[32768, 1], [12800, 1]]', 'M': '2304', 'N': '12800', 'K': '32768'})`
2. Core metrics: ref (TFLOP/s), TLX (TFLOP/s), speedup, compile time
3. Additional stats: samples, CV%, p50, p95, p99 (all based on TFLOP/s)
4. Op-specific columns, if the op declared any (see below)
5. status: ok/pip/noisy/error/...

### Common vs op-specific metrics

Everything in 2 and 3 is common and identical for every op: an op is normalised
onto them by supplying a `flop_count`, which is what makes a `speedup` or a `CV`
comparable across `mm` and `flash_attn`.

Anything else is one of two things, and which one decides where it goes:

- **What the run was asked for** -- shape, dtype, `causal`, direction, the
  seqlen distribution. These are inputs you chose, so they belong in `Case`:
  in `shape`, rendered by the op's own `label()`, which is the `input` column.
  `direction` is the one that is a real `Case` field, because it changes the
  FLOP count and has to reach the artifact key.
- **What the run found out** -- which SDPA backend torch dispatched, how many
  tokens a ragged batch actually carried, a rate that needs the measured mean.
  These go in `Result.extra`, a free-form JSON dict. They are artifact-only by
  default; an op puts one in the table by naming it in `EXTRA_COLUMNS`.

Today: `flash_attn` shows `ref_backend` (a 1.4x over `math` is a broken
benchmark, a 1.4x over cuDNN is a result), `hstu_attn` shows observed `tokens`,
`kda` shows `mtokens_per_s`, `mm` has none.

error: break, accuracy error
pip: speedup < 0.9, or TFLOP/s under an op's absolute floor when it has no
     reference, or compile time over that op's cap (2 min default; the ops with
     no heuristic raise it, see above)
noisy: CV% > 3%
ok: everything else

## How a call is timed

`wallclock`, the default, is `triton.testing.do_bench`: CUDA events around each
`fn()`, dispatched from the host one call at a time. It is the authoritative
mode here for two reasons. It measures what a caller actually gets -- if the
host cannot keep the GPU fed, that is a real cost of using the op, and
`tlx_host_us` sits in the artifact to say how much of the number it is. And it
is what every existing figure in this suite was taken with, so switching the
default would silently invalidate comparisons against them.

`gpu_events` is tritonbench's `--latency-measure-mode=gpu_events`: the whole
batch is enqueued behind a blocked stream so the host runs far ahead, and the
measured interval contains no dispatch gaps. Use it when the question is about
the kernels rather than the end-to-end path -- most usefully on the
multi-kernel backward passes (`flash_attn`, `hstu_attn`, `kda`), where a
wallclock reading folds in several launches' worth of host work. It flatters
both providers, so `speedup` moves much less than either absolute number; a
large gap between the two modes for one case is itself the finding, and it will
agree with a large `tlx_host_us`.

Neither mode is a profiler. There is no per-kernel breakdown here; use
`ir-debugging` or nsys for that.

## Compile time

The compile column is a real first call: a fresh Triton cache, so nothing is
reused. For `mm` that is under a second and every case pays it. For the ops with
no heuristic it compiles their whole autotune space -- `hstu_attn` is 48 forward
configs -- which per case is most of the suite's runtime, spent re-answering a
question that is not per-shape. Those ops sample it instead: one case per
direction, the rest report `-` and are not gated on compile.

`--cold-compile {all,first,none}` overrides. `all` is the honest per-case
reading and what to use when compile time is what you are investigating.

Stats are computed on per-iteration TFLOP/s, not converted from latency. So the
percentiles ascend: p99 is the best case, `min` (artifact only) the worst.

speedup = TLX / ref. >1 means TLX is faster.

Latency is not reported: it is `flop_count / TFLOP/s`, both in the artifact.
`tlx_host_us` stays in microseconds — host launch cost is not device work.

## Tests

- `test_harness.py` — `_harness` unit tests. No GPU, ~3s.
- `test_ops_perf.py` — pytest front end over each `bench_<op>.py`. Real kernels,
  minutes. CI's junitxml comes from here.

`pytest .` runs both, so it starts a benchmark.

## shapes

1. Synthetic (general): L1 only.
2. Focus shapes (arch specific): L2. Need to match the GPU arch. e.g. mm shapes sm100 and mm shapes gfx942

`--synthetic` runs list 1 under L2 instead. The focus list may be empty.

Each entry carries its own strides and dtype, so there is no dtype
cross-product. Strides, not a row/col flag: a leading stride wider than the row
is a padded slice, and 0 is a broadcast.

Shape lists live beside the kernel, in `tlx/ops/kernels/<op>/_shapes.py`, with
each arch module re-exporting its own as `PERF_SHAPES`. Not in this directory,
because the L1 correctness suites import the same lists -- one list, two
consumers, no drift. Only `mm`'s focus lists come from production captures;
the rest are hand-picked and marked provisional.

## Adding an op

Write `tlx/ops/kernels/<op>/_shapes.py` (`SYNTHETIC`, `<ARCH>_FOCUS`, `inputs()`,
`flops()`, `label()`), re-export `PERF_SHAPES` from each arch module, then a
`bench_<op>.py` with five names and one line of wiring:

```python
OP = "<op>"                    # catalog op name
REF_NAME = "..."               # what ref_fn is; "" when there is none
EXTRA_COLUMNS = ()             # ((header, Result.extra key), ...)

def cases(synthetic=False) -> list[Case]: ...
def prepare(case, space) -> Prepared: ...   # operands + closures + flop_count

supported, default_json, run, main = driver.bind(sys.modules[__name__])
```

Optional: `DEFAULT_SPACE` (defaults to `"heuristic"`), `COLD_COMPILE` (defaults
to `"all"`), and `annotate(result)`, a post-measurement hook for a metric that
needs the measured mean.

`Prepared.tlx_fn`/`ref_fn` must not allocate -- they are called hundreds of times
inside the measured window. Build the tensors in `prepare` and capture them.
`test_ops_perf.py` picks the new file up automatically.
