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
