"""End-to-end test for bitequiv's reduction-order *equivalence* pruning.

This is the equivalence *use* of the autotuner's general ``ir_config_prune`` hook. The
general (non-equivalence) showcases of that hook — PTX vectorization, TTGIR AutoWS, and
TTGIR ``tmem_load`` — live with the autotuner tests in
``python/test/unit/runtime/test_autotuner_constraint_prune.py``, since the hook is a general
IR filter, not bitequiv-specific.

Here ``bitequiv.equivalence.reduction_equivalence_prune`` keeps only configs whose compiled
TTGIR reduces in the same order as the reference (first) config — no kernel-output
comparison. The static reduction-order signature logic itself is unit-tested (no GPU) in
``test_equivalence.py``.
"""
import os
import sys

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    _IMPORT_OK = True
except Exception:  # pragma: no cover - torch/triton not importable
    _IMPORT_OK = False

# Make `bitequiv` importable when run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
if _IMPORT_OK:
    from bitequiv.equivalence import reduction_equivalence_prune


def is_cuda():
    return _IMPORT_OK and torch.cuda.is_available() and \
        triton.runtime.driver.active.get_current_target().backend == "cuda"


requires_cuda = pytest.mark.skipif(not is_cuda(), reason="requires a CUDA GPU")


# ---------------------------------------------------------------------------
# M1: STATIC bitwise-equivalence pruning at TTGIR level. No launch-output comparison:
# compile each config, read the reduce op's data layout from the TTGIR, and keep only
# configs whose reduction order matches the reference (first) config. num_warps changes
# warpsPerCTA along the reduce axis -> a different cross-warp tree -> different bits.
# ---------------------------------------------------------------------------
@requires_cuda
def test_static_reduction_equivalence():
    N = 4096
    src = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty(1, device="cuda", dtype=torch.float32)

    # First config (num_warps=4) defines the reference order; the second (num_warps=4)
    # shares it and is kept; num_warps 2 and 8 reduce differently and are pruned.
    configs = [
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ]

    prune = reduction_equivalence_prune("ttgir")

    @triton.autotune(configs=configs, key=["N"], prune_configs_by={"ir_config_prune": prune})
    @triton.jit
    def sum_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs, mask=offs < N, other=0.0)
        tl.store(dst, tl.sum(x, axis=0))

    sum_kernel[(1, )](src, out, N)

    # Reference (num_warps=4) class kept; num_warps 2 and 8 pruned as non-equivalent.
    assert sum_kernel.best_config.num_warps == 4
    assert {c.num_warps for c in prune.pruned} == {2, 8}
    assert len(prune.classes) == 3  # equivalence classes seen: num_warps 4, 2, 8
