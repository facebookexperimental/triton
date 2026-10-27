"""L1 for SM100 TorchTLX FlexAttention forward."""
from pathlib import Path

import pytest
import torch
from torch._inductor.choices import InductorChoices
from triton._internal_testing import is_blackwell

pytestmark = [
    pytest.mark.skipif(not is_blackwell(), reason="requires SM100"),
    pytest.mark.skipif(not hasattr(InductorChoices, "append_flex_attention_choices"),
                       reason="requires the FlexAttention choices hook"),
]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tlx_benchmark"))
    import bench_torchtlx_flex_attention as module
    with module.flex_adapter():
        yield module


@pytest.mark.parametrize(
    "mod", ("alibi", "head_bias", "document_mask", "causal", "prefix_lm", "sliding_window", "softcap", "rel", "noop"))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq", [256, 1024])
def test_flex_forward_mod(runner, tmp_path, mod, dtype, seq):
    tensors, score_mod, block_mask = runner.inputs(mod, seq, batch=1, heads=2, dim=128, dtype=dtype)
    stock, reference = runner.compile_arm(tensors, score_mod, block_mask, None, tmp_path / "stock.py")
    tlx, result = runner.compile_arm(tensors, score_mod, block_mask, "force", tmp_path / "tlx.py")
    if mod == "document_mask":
        independent = runner.document_reference(tensors, seq, 1)
        torch.testing.assert_close(reference, independent, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(result, reference, atol=2e-2, rtol=2e-2)
    torch._dynamo.reset()
    torch.testing.assert_close(stock(), reference, atol=0, rtol=0)
    torch.testing.assert_close(tlx(), result, atol=0, rtol=0)
