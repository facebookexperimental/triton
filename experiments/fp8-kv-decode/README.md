# FP8 KV decode experiments

This directory records the development of the gfx1250 TLX tutorial in
[`amd_pa_decode_fp8.py`](../../third_party/tlx/tutorials/amd_pa_decode_fp8.py).
The contract is E4M3FN KV with separate per-head scales, BF16 or FP8 Q, and BF16
attention output before projection.

- [Final optimization report](OPTIMIZATION_2026-11-09.md): measured gains,
  remaining AITER-favored cases, and fresh SGLang/vLLM controls.
- [Experiment ledger](FP8_KV_DECODE_ATTN.md): dated hypotheses, reasoning,
  validation, results, and rejected attempts.
- [Reproduction](REPRODUCE.md): environment, pinned sources, test commands,
  benchmarks, and chart generation.
- [References](REFERENCES.md): kernel capabilities and framework routing.

`charts/` and the tracked subset of `results/` contain the directly reviewable
measurements, CSV/JSON summaries, final test report, profiler analyses and
source snapshots needed to reproduce the final comparison.

The complete portable history is preserved in
[`history/portable-evidence-2026-11-09.tar.gz`](history/portable-evidence-2026-11-09.tar.gz).
Its [manifest](history/manifest.json) lists each archived file's path, size and
SHA256. It includes intermediate measurements, failed experiments, earlier
reports and dated source snapshots. Historical links in the ledger and older
reports resolve after restoring the archive from this directory:

```bash
tar --skip-old-files -xzf history/portable-evidence-2026-11-09.tar.gz
```

`--skip-old-files` preserves current source, documentation and tracked results.
Extract to a separate directory for an exact view of the original portable
history. Source snapshots under `results/` and `snapshots/` are immutable
evidence; do not reformat them.

Raw ATT/PMC sessions, decoder UI output, binary code objects, full counter
listings and duplicate external-source audits remain local to the experiment
host and are excluded from both Git and the archive. Their summaries, source
revisions and original local paths remain recorded.

The final complete timing matrix exceeds the usual 500 KiB hook limit but is
below 1 MiB. The pre-commit exception names only that matrix and the compressed
history; other files retain the repository's size check.
