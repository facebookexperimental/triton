# Jul 21 2026
TLX commit: df365a9f832bfc73fe0ce5c5a772e2e9baffe3e2
aiter commit: e6d04e3e2b4284d9f7e5b019e1776ca668d20f47

Initial testing results:
```
root@smci355-ccs-aus-m01-21:/workspace/projects/triton-tlx/third_party/tlx/tutorials/testing# python test_amd_pa_decode_perf.py 
[aiter] import [module_aiter_core] under /workspace/projects/aiter/aiter/jit/module_aiter_core.so
Running paged-decode benchmarks for: ['tlx', 'aiter'], qlens=[1]

=== query_length = 1 ===
paged-decode-performance-bf16-qlen1:
   BATCH     N_CTX  tlx (TB/s (effective HBM read))  aiter (TB/s (effective HBM read))  aiter/tlx speedup
0    1.0    8192.0                         1.001027                           0.622300           0.621662
1    8.0    8192.0                         4.122039                           4.112063           0.997580
2   32.0    8192.0                         7.075169                          10.677411           1.509139
3  128.0    8192.0                         6.875864                           6.535212           0.950457
4    1.0   32768.0                         2.053515                           1.501989           0.731423
5    8.0   32768.0                         4.808905                           4.714315           0.980330
6   32.0   32768.0                         6.429553                          10.407450           1.618689
7    8.0  131072.0                         6.206564                           5.119858           0.824910
```
We can see aither has better perf for later batch size, but worse perf for small batch size. MI355 has a peak bandwidth of 8TB/s, seems like something is wrong in the results

# Jul 22 2026

Three bugs fixed in `test_amd_pa_decode_perf.py`:

1. **Bandwidth byte count overcounted with pool sharing.** The formula used
   `2 * BATCH * NUM_KV_HEADS * N_CTX * HEAD_DIM * 2`, but the benchmark
   allocates a shared `pool_pages` physical page pool. For large BATCH the same
   physical pages are reused across sequences, so the true HBM footprint is much
   smaller than `BATCH * N_CTX` tokens. Fixed by using the actual tensor sizes:
   `(kc.numel() + vc.numel()) * kc.element_size()`.
   This explains the >8 TB/s numbers in the Jul 21 results (e.g. aiter at
   10.7 TB/s for BATCH=32) — those were artifacts of the overcounting.

2. **Min/max error bars swapped.** `do_bench` returns `(p50_ms, p20_ms, p80_ms)`
   for `quantiles=[0.5, 0.2, 0.8]`. Converting to bandwidth inverts the
   ordering, so the return was `tbps(p50), tbps(p80), tbps(p20)` — upper/lower
   bounds were flipped. Fixed to `tbps(ms), tbps(min_ms), tbps(max_ms)`.

3. **`DECODE_METHODS` hardcoded to `("tlx",)`.** aiter was excluded from
   `DECODE_METHODS` and filtered via `DEFAULT_DECODE_VERSIONS`. Now
   `DECODE_METHODS = ("tlx", "aiter") if _AITER_AVAILABLE else ("tlx",)`.

4. **`test_correctness` extended to cover both providers.** Previously only the
   TLX path was tested. Now `test_correctness` is parametrized over
   `DECODE_METHODS`, so aiter is validated against the reference whenever it is
   available. The aiter call goes through the existing `_make_decode_fn` helper
   (same path as the benchmark) and `pytest.skip`s gracefully if aiter fails to
   import at runtime. Run with:
   ```bash
   pytest third_party/tlx/tutorials/testing/test_amd_pa_decode_perf.py::test_correctness -v
   # selects by provider:
   pytest ... -k "tlx"
   pytest ... -k "aiter"
   ```

Re-run after fixes (same TLX/aiter commits as Jul 21):
```
root@smci355-ccs-aus-m01-21:/workspace/projects/triton-tlx/third_party/tlx/tutorials/testing# python test_amd_pa_decode_perf.py
[aiter] import [module_aiter_core] under /workspace/projects/aiter/aiter/jit/module_aiter_core.so
Running paged-decode benchmarks for: ['tlx', 'aiter'], qlens=[1]

=== query_length = 1 ===
paged-decode-performance-bf16-qlen1:
   BATCH     N_CTX  tlx (TB/s (effective HBM read))  aiter (TB/s (effective HBM read))  aiter/tlx speedup
0    1.0    8192.0                         1.018035                           0.738434           0.725352
1    8.0    8192.0                         2.500307                           2.383085           0.953117
2   32.0    8192.0                         3.929076                           4.129776           1.051081
3  128.0    8192.0                         5.200688                           5.161203           0.992408
4    1.0   32768.0                         2.424453                           1.425422           0.587935
5    8.0   32768.0                         3.919882                           3.666124           0.935264
6   32.0   32768.0                         5.192640                           4.783209           0.921152
7    8.0  131072.0                         5.207757                           4.215191           0.809406
```
Peak MI350 HBM bandwidth is ~8 TB/s. Numbers are well below that, indicating the
kernel is not yet fully utilizing memory bandwidth. TLX leads aiter for 1-seq and
large-context cases; aiter leads for mid-range batch (32 seqs). Performance
optimization is ongoing.

# Jul 23 2026
Implemented the 2-stage pipeline for PAGES_PER_TILE=4. Combined results showing perf vs aiter
and pipeline speedup (Jul 23 pipelined vs Jul 22 baseline), all in TB/s (effective HBM read):

| BATCH | N_CTX  | tlx (no pipeline) | tlx (2-stage pipeline) | aiter   | tlx/aiter speedup | pipeline speedup |
|------:|-------:|------------------:|-----------------------:|--------:|------------------:|-----------------:|
|     1 |   8192 |             1.018 |                  1.065 |   0.756 |             1.41x |            1.05x |
|     8 |   8192 |             2.500 |                  2.591 |   2.387 |             1.09x |            1.04x |
|    32 |   8192 |             3.929 |                  4.222 |   4.130 |             1.02x |            1.07x |
|   128 |   8192 |             5.201 |                  5.756 |   5.156 |             1.12x |            1.11x |
|     1 |  32768 |             2.424 |                  2.414 |   1.434 |             1.68x |            1.00x |
|     8 |  32768 |             3.920 |                  4.330 |   3.668 |             1.18x |            1.10x |
|    32 |  32768 |             5.193 |                  5.724 |   4.775 |             1.20x |            1.10x |
|     8 | 131072 |             5.208 |                  5.745 |   4.216 |             1.36x |            1.10x |

The 2-stage pipeline delivers 1.0–1.1x gains across all configurations. TLX outperforms aiter
across all configurations after pipelining.
