"""GPU integration test for the equivalence example
(`bitequiv/examples/autotune_equivalence_pruning.py`).

Covers both parts: (1) the standalone checkers — the TTGIR checker is blind to FMA contraction
while the PTX checker is not; and (2) autotuner PTX-pruning — the default (first-config) and an
explicit reference keep different, internally-equivalent config sets. The pruning tests use a
small config subset (`_configs_quick`) so the test stays fast. GPU-gated (the example compiles +
benchmarks kernels); skips without a CUDA GPU. If a launch ever hangs, kill it with
`third_party/tlx/killgpu.sh`.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_standalone_checkers_ttgir_vs_ptx():
    from bitequiv.examples import autotune_equivalence_pruning as ex

    res = ex.run_checkers_standalone()
    # Pure-add row sum: unordered num_warps differ for BOTH checkers; inner_tree merge for both.
    assert res["rowsum_unordered_nw4_vs_nw8"] == (False, False)
    assert res["rowsum_inner_tree_nw4_vs_nw8"] == (True, True)
    # The headline contrast: TTGIR is blind to FMA contraction (equivalent), PTX is not.
    assert res["rowdot_fp_fusion_on_vs_off"] == (True, False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_default_reference_prunes_and_keeps_equivalent_set():
    from bitequiv.examples import autotune_equivalence_pruning as ex

    result = ex.run_default_reference(ex._configs_quick())
    # Some configs are pruned, some kept; the diverse grid means pruning has real work.
    assert result["pruned_labels"], "expected some configs to be pruned"
    assert result["kept_labels"], "expected at least one surviving config"
    # The first config is inner_tree, so every kept config is inner_tree (unordered can't match).
    assert all("inner_tree" in label for label in result["kept_labels"])
    # The best config the autotuner picked must be one it kept (equivalent to the reference).
    assert result["best_label"] in result["kept_labels"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_explicit_reference_changes_kept_set():
    from bitequiv.examples import autotune_equivalence_pruning as ex

    configs = ex._configs_quick()
    default = ex.run_default_reference(configs)
    explicit = ex.run_explicit_reference(configs)
    # Choosing an explicit (unordered) anchor changes which configs are kept vs the default mode.
    assert set(default["kept_labels"]) != set(explicit["kept_labels"])
    # ...and the survivors are anchored to that unordered choice.
    assert explicit["kept_labels"]
    assert all("unordered" in label for label in explicit["kept_labels"])
