"""L1 for SM100 TorchTLX FlexAttention forward."""
import re
from pathlib import Path

import pytest
import torch
from torch._inductor.choices import InductorChoices
from triton._internal_testing import is_blackwell

pytestmark = [
    pytest.mark.skipif(not hasattr(InductorChoices, "append_flex_attention_choices"),
                       reason="requires the FlexAttention choices hook"),
]


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tlx_benchmark"))
    import bench_torchtlx_flex_attention as module
    return module


@pytest.fixture
def runner(benchmark):
    if not is_blackwell():
        pytest.skip("requires sm100")
    with benchmark.flex_adapter():
        yield benchmark


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


@pytest.mark.parametrize(
    "mod,q_len,kv_len,dim",
    [
        ("noop", 32, 1920, 80),
        ("noop", 1, 77, 64),
        ("noop", 512, 320, 80),
        ("noop", 512, 320, 96),
        ("causal", 512, 320, 80),
        ("prefix_lm", 512, 320, 96),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flex_forward_one_group(runner, tmp_path, mod, q_len, kv_len, dim, dtype):
    tensors, score_mod, block_mask = runner.inputs(mod, kv_len, batch=1, heads=2, dim=dim, dtype=dtype, q_len=q_len,
                                                   block_size=(128, 64))
    assert tensors[0].shape == (1, 2, q_len, dim)
    assert tensors[1].shape == tensors[2].shape == (1, 2, kv_len, dim)
    if block_mask is not None:
        normal, full = block_mask.kv_num_blocks, block_mask.full_kv_num_blocks
        total = normal + full
        assert ((normal > 0) & (full > 0) & (total >= 3) & (total % 2 == 1)).any()
    options = {"FORCE_USE_FLEX_ATTENTION": True, "fwd_BLOCK_N": 64}
    stock, reference = runner.compile_arm(tensors, score_mod, block_mask, None, tmp_path / "stock.py",
                                          kernel_options=options)
    tlx, result = runner.compile_arm(tensors, score_mod, block_mask, "force", tmp_path / "tlx.py",
                                     kernel_options=options)
    source = (tmp_path / "tlx.py").read_text()
    assert re.search(r"['\"]NUM_MMA_GROUPS['\"]:\s*1\b", source)
    assert re.search(r"['\"]BLOCK_M['\"]:\s*64\b", source)
    assert re.search(r"['\"]BLOCK_N['\"]:\s*64\b", source)
    torch.testing.assert_close(result, reference, atol=2e-2, rtol=2e-2)
    torch._dynamo.reset()
    torch.testing.assert_close(stock(), reference, atol=0, rtol=0)
    torch.testing.assert_close(tlx(), result, atol=0, rtol=0)


def test_flex_adapter_preserves_blackwell_choice_space(monkeypatch, benchmark):
    from types import SimpleNamespace
    from torch._inductor import config
    from triton.language.extra.tlx.inductor import flex_attention_templates as templates

    def append(choices, **kwargs):
        choices.append(kwargs)

    monkeypatch.setattr(templates, "_use_amd_flex_template", lambda: False)
    monkeypatch.setattr(templates, "_blackwell_flex_fwd_configs", lambda configs: configs)
    monkeypatch.setattr(templates, "current_target",
                        lambda: SimpleNamespace(num_sms=148, is_blackwell=True, is_gfx1250=False))
    monkeypatch.setattr(torch._inductor.utils, "can_use_tma", lambda *args: True)
    monkeypatch.setattr(templates, "blackwell_flex_attention_template", SimpleNamespace(maybe_append_choice=append))
    configs = [
        SimpleNamespace(block_m=m, block_n=n, num_warps=w, num_stages=s)
        for m, n, w, s in ((128, 64, 4, 2), (256, 128, 8, 3))
    ]
    query = SimpleNamespace(get_size=lambda: [8, 16, 4096, 128], get_dtype=lambda: torch.bfloat16)
    choices = ["stock"]
    with benchmark.flex_adapter(), config.patch({"triton.tlx_mode": "force", "max_autotune": False}):
        config.inductor_choices_class().append_flex_attention_choices(choices, configs, [query] * 3 + [object()] * 6,
                                                                      [object(), object()], object(), {}, 256, 128)
    assert [(c["BLOCK_M"], c["BLOCK_N"], c["num_warps"], c["num_stages"]) for c in choices] == [(128, 64, 4, 1),
                                                                                                (256, 128, 8, 1)]
    assert all(c["USE_TMA"] and c["NUM_SMS"] == 148 for c in choices)
    assert all("waves_per_eu" not in c and "matrix_instr_nonkdim" not in c for c in choices)
