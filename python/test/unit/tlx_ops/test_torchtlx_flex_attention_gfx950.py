"""L1 for gfx950 FlexAttention forward with max_autotune disabled."""
from pathlib import Path

import pytest
import torch
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch.nn.attention.flex_attention import create_block_mask
from triton._internal_testing import is_hip_cdna4

pytestmark = [
    pytest.mark.skipif(not is_hip_cdna4(), reason="requires gfx950"),
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
    "mod", ("rel", "alibi", "document_mask", "causal", "prefix_lm", "sliding_window", "softcap", "head_bias", "noop"))
@pytest.mark.parametrize("seq,dim", [(129, 64), (257, 128)])
def test_flex_forward_mod_tail(runner, tmp_path, record_property, mod, seq, dim):
    tensors, score_mod, block_mask = runner.inputs(mod, seq, batch=3, heads=2, dim=dim)
    _, ref = runner.compile_arm(tensors, score_mod, block_mask, None, tmp_path / "stock.py")
    replay, got = runner.compile_arm(tensors, score_mod, block_mask, "force", tmp_path / "tlx.py")
    if mod == "document_mask":
        independent = runner.document_reference(tensors, seq, 3)
        torch.testing.assert_close(got, independent, atol=2e-2, rtol=2e-2)
        # Stock default also fails for uneven D=64 document tails on this wheel.
        record_property("stock_sdpa_max_abs", (ref.float() - independent.float()).abs().max().item())
    else:
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
    replay()
    torch.cuda.synchronize()


def test_flex_forward_empty_rows_cross_attention(runner, tmp_path):
    torch.manual_seed(0)
    q = torch.randn(2, 4, 257, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2, 193, 64, device="cuda", dtype=torch.bfloat16)
    mask = lambda b, h, m, n: (m % 11 != 0) & (m >= n)
    block_mask = torch.compile(create_block_mask)(mask, 2, 1, 257, 193, device="cuda")
    from torch.nn.attention.flex_attention import flex_attention
    from torch._inductor.utils import run_and_get_code
    outputs = []
    for mode in (None, "force"):
        torch._dynamo.reset()
        with config.patch(
            {"triton.tlx_mode": mode, "max_autotune": False, "max_autotune_gemm": False, "force_disable_caches": True}):
            fn = torch.compile(lambda q, k, v: flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True))
            result, source = run_and_get_code(fn, q, k, v)
            if mode:
                assert "tlx_gfx950_flex_attention" in "\n".join(source)
            outputs.append(result.clone())
    torch.testing.assert_close(outputs[1], outputs[0], atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(outputs[1][:, :, ::11]) == 0


def test_flex_forward_graph_replay_after_dynamo_reset(runner, tmp_path):
    import gc

    tensors, score_mod, block_mask = runner.inputs("head_bias", 128, batch=8, heads=16, dim=128)
    stock_replay, ref = runner.compile_arm(tensors, score_mod, block_mask, None, tmp_path / "stock.py")
    tlx_replay, got = runner.compile_arm(tensors, score_mod, block_mask, "force", tmp_path / "tlx.py")
    torch._dynamo.reset()
    gc.collect()
    for _ in range(3):
        torch.testing.assert_close(stock_replay(), ref, atol=0, rtol=0)
        torch.testing.assert_close(tlx_replay(), got, atol=0, rtol=0)
