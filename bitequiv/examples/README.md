# bitequiv examples

Runnable, self-contained examples for the bitwise-equivalence project. Each script
compiles a kernel, prints the IR/PTX evidence for a specific compiler behavior, and
asserts the runtime result on a real GPU.

## numerical-inconsistency/ — autotuning configs that silently change the bits

A standalone repro kit (`n01`–`n05`) showing how ordinary autotuning choices
(`num_warps`, reduction ordering, dot/reduce) produce *different output bits* for the
same kernel — the problem the equivalence checker exists to catch. Each script is
self-contained (it carries its own IR-capture helpers in `_helpers.py` and imports
nothing from `bitequiv.*`, so it runs on a bare upstream checkout) and runs on any CUDA
box. See `numerical-inconsistency/README.md` for the catalogue.

| File | Shows |
|------|-------|
| `n01_autotuner_picks_silently.py` | the autotuner silently picking a bit-different config |
| `n02_sum_reduction_classes.py` | `num_warps` → distinct sum-reduction bit-classes |
| `n03_softmax_row_reduction.py` | softmax row-reduction bit-classes |
| `n04_layernorm_two_reductions.py` | layernorm's two reductions |
| `n05_dot_reduce.py` | a dot / `tl.sum(x*y)` reduction |

## Run

```bash
# one example
python bitequiv/examples/numerical-inconsistency/n02_sum_reduction_classes.py
# the example smoke test
pytest bitequiv/examples/numerical-inconsistency/test_smoke.py
```

The TTGIR equivalence checker that reasons about these cases is in `../reduction_tree.py`
/ `../equivalence_ttgir.py`; its GPU evaluation (static verdict vs `torch.equal` bits) is
in `../evaluation/`.
