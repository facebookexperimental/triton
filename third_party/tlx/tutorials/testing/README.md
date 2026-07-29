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

The main branch just landed a new version with some optimization here are the perf numbers of the main branch:
```
   BATCH     N_CTX  tlx (TB/s (effective HBM read))  aiter (TB/s (effective HBM read))  aiter/tlx speedup
0    1.0    8192.0                         0.842230                           0.573776           0.681259
1    8.0    8192.0                         2.571221                           2.371338           0.922261
2   32.0    8192.0                         4.314261                           4.123401           0.955761
3  128.0    8192.0                         5.830760                           5.148329           0.882960
4    1.0   32768.0                         2.605158                           1.425422           0.547154
5    8.0   32768.0                         4.387634                           3.653152           0.832602
6   32.0   32768.0                         5.678139                           4.770882           0.840219
7    8.0  131072.0                         5.421255                           4.208248           0.776250
```
The two implementation has comparable perf numbers. Next step is for a three stage

# Jul 28 2026

- scenario 1: Performance of 2-stage pipeline, but without async_load_to_lds
```
=== query_length = 1 ===
paged-decode-performance-bf16-qlen1:
   BATCH     N_CTX  tlx (TB/s (effective HBM read))  aiter (TB/s (effective HBM read))  aiter/tlx speedup
0    1.0    8192.0                         1.053845                           0.735827           0.698230
1    8.0    8192.0                         2.559453                           2.338247           0.913573
2   32.0    8192.0                         4.267620                           4.045109           0.947860
3  128.0    8192.0                         5.688352                           5.054282           0.888532
4    1.0   32768.0                         2.393326                           1.427848           0.596596
5    8.0   32768.0                         4.340771                           3.549771           0.817774
6   32.0   32768.0                         5.673032                           4.765789           0.840078
7    8.0  131072.0                         5.559929                           4.203795           0.756088
```

IR dumps are in folder `third_party/tlx/tutorials/testing/ir_dumps/2-stage-no_async_copy`

- Scenario 2: Performance of 2-stage pipeline and with buffer_load_to_lds
```
=== query_length = 1 ===
paged-decode-performance-bf16-qlen1:
   BATCH     N_CTX  tlx (TB/s (effective HBM read))  aiter (TB/s (effective HBM read))  aiter/tlx speedup
0    1.0    8192.0                         0.961996                           0.619543           0.644018
1    8.0    8192.0                         2.553610                           2.335034           0.914405
2   32.0    8192.0                         4.222011                           4.029322           0.954361
3  128.0    8192.0                         5.580764                           5.048626           0.904648
4    1.0   32768.0                         2.449229                           1.438869           0.587479
5    8.0   32768.0                         4.101978                           3.554495           0.866532
6   32.0   32768.0                         5.585409                           4.772387           0.854438
7    8.0  131072.0                         5.559396                           4.214049           0.758005
```
IR dumps are in folder `third_party/tlx/tutorials/testing/ir_dumps/b1_n32768_2stage_buffer_load_to_lds`




- Main branch performance numbers: (commit #35a1f082c1 )
```
   BATCH     N_CTX  tlx (TB/s (effective HBM read))  aiter (TB/s (effective HBM read))  aiter/tlx speedup
0    1.0    8192.0                         0.795883                           0.551882           0.693421
1    8.0    8192.0                         2.541954                           2.343187           0.921805
2   32.0    8192.0                         4.328173                           3.941755           0.910720
3  128.0    8192.0                         5.848219                           5.050478           0.863593
4    1.0   32768.0                         2.601119                           1.416965           0.544752
5    8.0   32768.0                         4.301109                           3.567675           0.829478
6   32.0   32768.0                         5.626944                           4.758186           0.845608
7    8.0  131072.0                         5.351011                           4.199184           0.784746
```

Things are little complex for now:
1) Latest TLX commit (commit: f2b8d0b7ca802e51b2c2346b7b597bcee694cd57) has a warning message like: 
/workspace/projects/aiter/aiter/ops/triton/gluon/pa_decode_gluon.py:4441: UserWarning: [Triton] TRITON_USE_C_DISPATCHER=1 but kernel 'aiter.ops.triton.gluon.pa_decode_gluon.paged_attention_decode_v2_gluon_dot_kernel' has no C dispatcher, falling back to Python launch paged_attention_kernel[grid](

It is caused by (from Claude): 
```
TRITON_USE_C_DISPATCHER=1 is set in your environment (it's actually on by default in this codebase — see knobs.py:586). The C dispatcher is a fast kernel launch path that bypasses Python overhead.                                                                             
                                                                                                                                                                                                                                                                                   
  The C dispatcher requires a launch metadata schema ("launch_metadata" in the compiled kernel's ASM dict), which is generated by backend.make_launch_metadata(). Looking at the code:                                                                                             
                                                                                                                                                                                                                                                                                   
  - The NVIDIA backend (third_party/nvidia/backend/compiler.py:323) implements make_launch_metadata.                                                                                                                                                                               
  - The AMD/HIP backend does not implement make_launch_metadata.                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                   
  So when _pa_decode_reduce_kernel (a plain @triton.jit kernel) compiles for AMD, there's no launch metadata schema → no C dispatcher → Triton warns and falls back to Python launch. 
```

By settng the env var: `TRITON_USE_C_DISPATCHER=0 `, we got the following perf numbers:

```
[aiter] import [module_aiter_core] under /workspace/projects/aiter/aiter/jit/module_aiter_core.so
paged-decode-performance-bf16-qlen1:
   BATCH     N_CTX  tlx (TB/s (effective HBM read))  aiter (TB/s (effective HBM read))  aiter/tlx speedup
0    1.0    8192.0                         1.208733                           1.028016           0.850490
1    8.0    8192.0                         2.547793                           2.333410           0.915855
2   32.0    8192.0                         4.428093                           4.078296           0.921005
3  128.0    8192.0                         5.857791                           5.119346           0.873938
4    1.0   32768.0                         2.549729                           1.399267           0.548791
5    8.0   32768.0                         4.407735                           3.577673           0.811681
6   32.0   32768.0                         5.672110                           4.839683           0.853242
7    8.0  131072.0                         5.562219                           4.198856           0.754889
```


