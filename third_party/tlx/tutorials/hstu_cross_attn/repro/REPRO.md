# Repro: autoWS HSTU reduce_dq wrong dq (task T278116159)

autoWS (meta warp specialization) produces wrong `dq` for the HSTU cross-attention
backward `reduce_dq` kernel; `dk`/`dv`, the non-WS `redq` kernel, and the TLX
`attn_bwd_ws` kernel are all correct. See the triage doc
`../../../../nvidia/hopper/lib/Transforms/WarpSpecialization/docs/HSTUReduceDqTriage.md`.

The **law**: wrong dq Q-blocks = `min(KV_blocks-1, num_stages)`, at the tail of the
Q range. Needs **>=2 Q-blocks** (the inner-Q peel) AND **>=2 KV-blocks** (the
in-kernel KV boundary). KV=1 or Q=1-block is correct.

## Environment

- Blackwell GPU (sm100 / GB200).
- A Python env with this triton built (editable). At Meta this was
  `~/.conda/envs/metamain2/bin/python` with triton at this checkout; rebuild after
  C++ changes: `ninja -C build/cmake.linux-aarch64-cpython-3.12`.
- `compute-sanitizer` (CUDA 12.8) for the barrier check (optional).

Run everything with `TRITON_ALWAYS_COMPILE=1` to avoid cache reuse. The scripts
add both this `repro/` dir and its parent (the kernels) to `sys.path`, so run them
from anywhere.

## Files

| file | what it does |
|---|---|
| `../bench_bwd.py` | 3-way accuracy + perf: redq / autows / tlx vs torch-float ref. **Primary oracle.** |
| `l10b.py`         | `go(Lq, Lkv, ns)` — autoWS-vs-redq dq, prints bad Q-block indices (the law). |
| `sweep2.py`       | shows the bug needs >=2 Q-blocks (Lq=1-block is correct even at KV>=2). |
| `decomp.py`       | accuracy decomposition (SILU): proves the bad block reads a **stale** dqT slot (not dropped, not clean arithmetic). synccheck-independent. |
| `rc.py`           | minimal single autoWS/tlx/redq bwd (KV=2, tiny) — for compute-sanitizer. `V=autows|tlx|redq`. |
| `singleloop_annotated_repro.mlir` | pre-WS TTGIR for `triton-opt` per-pass dumps (no GPU needed). |

## Commands

Accuracy (shows autows wrong at KV>=2, redq/tlx correct):
```bash
cd ..    # the hstu_cross_attn dir (bench_bwd.py lives there)
TRITON_ALWAYS_COMPILE=1 python bench_bwd.py --acc
```

The law + "needs >=2 Q-blocks":
```bash
python repro/sweep2.py     # Lq=64(1blk) KV=2/3 -> 0 bad;  Lq=128(2blk) KV=2 -> last block bad
```

Value decomposition (stale-read proof, no synccheck):
```bash
python repro/decomp.py
# bad Q blocks: [1];  kv1term is uncorrelated ~2x magnitude (not zero, not a clean partial)
```

Barrier check (optional; synccheck can false-positive, so treat as corroboration):
```bash
CS=/usr/local/cuda-12.8/bin/compute-sanitizer
V=autows TRITON_ALWAYS_COMPILE=1 python repro/rc.py            # warm/compile
V=autows $CS --tool synccheck --target-processes application-only python repro/rc.py
#   -> "Barrier error detected. Missing wait." at triton_bw_cross_attention.py:3272 (autows)
V=tlx    $CS --tool synccheck --target-processes application-only python repro/rc.py  # clean
V=autows $CS --tool racecheck --target-processes application-only python repro/rc.py  # 0 hazards
```

Per-pass IR (no GPU) — shows the reduction peel appears only after ExpandLoops:
```bash
OPT=build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt   # from repo root
$OPT repro/singleloop_annotated_repro.mlir \
  --nvgpu-warp-specialization="num-stages=2 smem-budget=232448" \
  --tritongpu-pipeline="num-stages=2 dump-intermediate-steps=true" 2>&1 | \
  grep -nE "SoftwarePipeliner internal IR Dump After: (LowerLoops|ExpandLoops)"
# reduction (task0/default) is a plain nested for KV { for Q } after LowerLoops,
# and PEELED (first-Q-block hoisted, %cmpi sgt seq_len_q,0 predicate) after ExpandLoops.
```

## Root cause (summary)

`dqT` is a single-copy, cross-partition, non-loop-carried TMEM accumulator, produced
by the gemm partition and consumed (read + `async_tma_reduce add`) by the reduction
partition every inner Q-iteration; `DQ[q]` accumulates across the **in-kernel KV
loop**. ExpandLoops peels the inner Q-loop of both partitions (the reduction's peel
is triggered by the stage-1 `async_tma_store_token_wait` that pipelines the TMA
reduce store). The consumer peels the FIRST Q-block, the producer peels the LAST —
so across the KV boundary the producer's next-KV writes overtake the consumer's
current-KV trailing reads on the single dqT slot -> stale read -> wrong dq.

FA bwd has the same store-wait/peel but is correct because its dq reduction is a flat
single loop (non-persistent; KV = separate grid launches) or nested over disjoint
outputs (persistent; tiles differ). HSTU is the only shape that nests a peeled,
single-slot, shared-`DQ[q]`-accumulating reduce in-kernel.

Fix directions: suppress/align the inner-Q peel so next-KV writes can't overtake the
trailing reads (or double-buffer dqT if TMEM fits), or restructure like FA
(grid-launch KV) / TLX (hand-managed handshake). Or use redq/TLX for reduce_dq.
