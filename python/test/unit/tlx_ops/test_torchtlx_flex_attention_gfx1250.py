"""L1 for the gfx1250 FlexAttention template and device-side recovery."""
from pathlib import Path

import pytest
import torch
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention
from triton._internal_testing import is_hip_gfx1250
from triton.tlx.ops.kernels.flash_attn._shapes import GFX1250_PRODUCTION


@pytest.fixture
def runner(monkeypatch):
    if not is_hip_gfx1250():
        pytest.skip("requires gfx1250")
    if not hasattr(InductorChoices, "append_flex_attention_choices"):
        pytest.skip("requires the FlexAttention choices hook")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tlx_benchmark"))
    import bench_torchtlx_flex_attention as benchmark
    with benchmark.flex_adapter():
        yield benchmark


def _reference(q, k, v, score_mod=None, mask_mod=None):
    if q.shape[1] != k.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    scores = (q.double() @ k.double().transpose(-1, -2)) * q.shape[-1]**-0.5
    b = torch.arange(q.shape[0], device=q.device)[:, None, None, None]
    h = torch.arange(q.shape[1], device=q.device)[None, :, None, None]
    m = torch.arange(q.shape[2], device=q.device)[None, None, :, None]
    n = torch.arange(k.shape[2], device=q.device)[None, None, None, :]
    if score_mod is not None:
        scores = score_mod(scores, b, h, m, n)
    if mask_mod is not None:
        scores = scores.masked_fill(~mask_mod(b, h, m, n), -float("inf"))
    output = torch.nan_to_num(scores.softmax(-1)) @ v.double()
    return output, scores.logsumexp(-1), scores.amax(-1)


def _compile(tensors, score_mod=None, mask_mod=None, aux=False, block_size=128, kernel_options=None):
    q, k, v = tensors
    block_mask = None
    if mask_mod is None and q.shape[0] != k.shape[0]:
        mask_mod = lambda b, h, m, n: m >= 0
    if mask_mod is not None:
        with config.patch({"triton.tlx_mode": None}):
            block_mask = torch.compile(create_block_mask)(mask_mod, q.shape[0], q.shape[1], q.shape[2], k.shape[2],
                                                          device=q.device, BLOCK_SIZE=block_size)
    torch._dynamo.reset()
    with config.patch(
        {"triton.tlx_mode": "force", "max_autotune": False, "max_autotune_gemm": False, "force_disable_caches": True}):

        def fn(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, enable_gqa=True,
                                  return_aux=AuxRequest(lse=True, max_scores=aux), kernel_options=kernel_options)

        compiled = torch.compile(fn, fullgraph=True)
        output, code = run_and_get_code(compiled, q, k, v)
        assert "tlx.async_amd_descriptor_load" in "\n".join(code)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(q, k, v)
    return graph, captured, compiled


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", ["dense", "causal", "gaps", "empty", "softcap", "relative"])
def test_flex_cross_attention_modifications(runner, dtype, case):
    torch.manual_seed(0)
    q = torch.randn((2, 4, 257, 128), device="cuda", dtype=dtype)
    k = torch.randn((1, 2, 1025, 128), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    score_mod = None
    mask_mod = None
    if case == "causal":
        mask_mod = lambda b, h, m, n: m >= n
    elif case == "gaps":
        mask_mod = lambda b, h, m, n: (n // 64) % 4 != 3
    elif case == "empty":
        mask_mod = lambda b, h, m, n: (m % 11 != 0) & (n < m)
    elif case == "softcap":
        score_mod = lambda s, b, h, m, n: 20 * torch.tanh(s / 20)
    elif case == "relative":
        score_mod = lambda s, b, h, m, n: s + (m - n) * .05 + h
    expected, lse, _ = _reference(q, k, v, score_mod, mask_mod)
    graph, (output, aux), keepalive = _compile((q, k, v), score_mod, mask_mod)
    for _ in range(3):
        graph.replay()
        tolerance = .005 if dtype == torch.bfloat16 else .0008
        assert (output.double() - expected).square().mean().sqrt() < tolerance * expected.square().mean().sqrt()
        torch.testing.assert_close(aux.lse.double(), lse, atol=1e-4, rtol=1e-4)
        if case == "empty":
            assert torch.count_nonzero(output[:, :, ::11]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset", [0, 1])
def test_flex_recovery_graph_replay(runner, dtype, offset):
    torch.manual_seed(1)

    def make(length):
        shape = (1, 2, length, 128)
        return torch.empty(2 * length * 128 + offset, device="cuda", dtype=dtype)[offset:].view(shape).normal_()

    q, k, v = make(129), make(1025), make(1025)
    v.abs_().add_(.25)
    graph, (output, aux), keepalive = _compile((q, k, v))
    for mode in ("ordinary", "exp_overflow", "probability_overflow", "common_offset", "ordinary"):
        if mode == "ordinary":
            q.normal_()
            k.normal_()
        elif mode == "exp_overflow":
            q.fill_(1)
            k.zero_()
            k[:, :, 128:].fill_(16)
        elif mode == "probability_overflow":
            q.zero_()
            q[..., 0] = 1
            k.zero_()
            k[:, :, 128:, 0] = 200
        else:
            q.zero_()
            q[..., 0] = 256
            q[..., 1] = 1
            k.zero_()
            k[..., 0] = 256
            k[..., 1] = torch.arange(k.shape[2], device=k.device) % 9 - 4
        expected, lse, _ = _reference(q, k, v)
        for _ in range(20):
            graph.replay()
            tolerance = .005 if dtype == torch.bfloat16 else .0008
            torch.testing.assert_close(output.double(), expected, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(aux.lse.double(), lse, atol=1e-3, rtol=1e-4)


def test_flex_auxiliary_max(runner):
    tensors = [torch.randn((1, 2, 257, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    score_mod = lambda s, b, h, m, n: s + n * .01
    expected, lse, maximum = _reference(*tensors, score_mod)
    graph, (output, aux), keepalive = _compile(tensors, score_mod, aux=True)
    graph.replay()
    torch.testing.assert_close(output.double(), expected, atol=.005, rtol=.005)
    torch.testing.assert_close(aux.lse.double(), lse, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(aux.max_scores.double(), maximum, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("value", [65504.0, -65504.0])
def test_flex_fp16_output_range(runner, value):
    q = torch.zeros((1, 1, 129, 128), device="cuda", dtype=torch.float16)
    q[..., 0] = 1
    k = torch.zeros((1, 1, 1025, 128), device="cuda", dtype=q.dtype)
    k[:, :, 1:, 0] = -7.7265625
    v = torch.full_like(k, value)
    graph, (output, aux), keepalive = _compile((q, k, v))
    graph.replay()
    torch.testing.assert_close(output, torch.full_like(output, value), atol=0, rtol=0)


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_flex_fp16_preserves_nonfinite_output(runner, value):
    q = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.float16)
    graph, (output, _), keepalive = _compile((q, q, torch.full_like(q, value)))
    graph.replay()
    torch.testing.assert_close(output, torch.full_like(output, value), equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size,tiles", [(64, (32, 32)), (64, (64, 64)), (256, (256, 64))])
def test_flex_sparse_tiles_and_strides(runner, dtype, block_size, tiles):

    def make(length):
        return torch.randn((2, 2, length, 129), device="cuda", dtype=dtype)[..., :128]

    tensors = (make(257), make(769), make(769))
    mask = lambda b, h, m, n: (m >= n) & ((n // 64) % 4 != 3)
    expected, lse, _ = _reference(*tensors, mask_mod=mask)
    options = {"fwd_BLOCK_M": tiles[0], "fwd_BLOCK_N": tiles[1]}
    graph, (output, aux), keepalive = _compile(tensors, mask_mod=mask, block_size=block_size, kernel_options=options)
    for _ in range(3):
        graph.replay()
        tolerance = .005 if dtype == torch.bfloat16 else .0008
        assert (output.double() - expected).square().mean().sqrt() < tolerance * expected.square().mean().sqrt()
        torch.testing.assert_close(aux.lse.double(), lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", GFX1250_PRODUCTION)
def test_flex_production_shapes(runner, shape):
    torch.manual_seed(1250)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[shape.dtype]
    tensors = [
        torch.randn((shape.batch, shape.heads, shape.context, shape.head_dim), device="cuda", dtype=dtype)
        for _ in range(3)
    ]
    q, k, v = tensors
    rows = torch.tensor([0, 31, 32, 63, 64, 127, 128, 255, 256, 65535, 65536, shape.context - 1], device=q.device)
    expected, lse, _ = _reference(q.index_select(2, rows), k, v)
    graph, (output, aux), keepalive = _compile(tensors)
    graph.replay()
    assert torch.isfinite(output).all()
    actual = output.index_select(2, rows).double()
    tolerance = .005 if dtype == torch.bfloat16 else .0008
    assert (actual - expected).square().mean().sqrt() < tolerance * expected.square().mean().sqrt()
    torch.testing.assert_close(aux.lse.index_select(2, rows).double(), lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("axis", [0, 1])
def test_flex_large_outer_stride(runner, dtype, axis):
    shape = (3, 1, 128, 128) if axis == 0 else (1, 3, 128, 128)
    strides = list(torch.empty(shape).stride())
    strides[axis] = 2**30 + 128
    tensors = [torch.empty_strided(shape, strides, dtype=dtype, device="cuda") for _ in range(3)]
    for tensor in tensors:
        tensor.copy_(torch.randn(shape, dtype=dtype, device="cuda"))
    expected, lse, _ = _reference(*tensors)
    graph, (output, aux), keepalive = _compile(tensors)
    graph.replay()
    tolerance = .005 if dtype == torch.bfloat16 else .0008
    assert (output.double() - expected).square().mean().sqrt() < tolerance * expected.square().mean().sqrt()
    torch.testing.assert_close(aux.lse.double(), lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flex_short_query(runner, dtype):
    q = torch.randn((1, 2, 2, 128), device="cuda", dtype=dtype)
    k = torch.randn((1, 2, 65, 128), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    expected, lse, _ = _reference(q, k, v)
    graph, (output, aux), keepalive = _compile((q, k, v), kernel_options={"FORCE_USE_FLEX_ATTENTION": True})
    graph.replay()
    tolerance = .005 if dtype == torch.bfloat16 else .0008
    assert (output.double() - expected).square().mean().sqrt() < tolerance * expected.square().mean().sqrt()
    torch.testing.assert_close(aux.lse.double(), lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mode", ["allow", "force"])
@pytest.mark.parametrize("case", ["fp16", "bf16", "fp32", "mixed", "head_dim", "stride", "invalid_tile"])
def test_flex_dispatch_preserves_unsupported_choices(monkeypatch, mode, case):
    from types import SimpleNamespace
    from triton.language.extra.tlx.inductor import flex_attention_templates as templates

    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case, torch.bfloat16)
    tensors = [torch.empty((1, 2, 129, 128), dtype=dtype) for _ in range(3)]
    if case == "mixed":
        tensors[2] = tensors[2].half()
    elif case == "head_dim":
        tensors[2] = tensors[2][..., :64]
    elif case == "stride":
        tensors[0] = torch.empty((1, 2, 129, 256), dtype=dtype)[..., ::2]
    nodes = [
        SimpleNamespace(get_size=lambda t=t: list(t.shape), get_stride=lambda t=t: list(t.stride()),
                        get_dtype=lambda t=t: t.dtype) for t in tensors
    ]
    monkeypatch.setattr(templates, "current_target", lambda: SimpleNamespace(key="gfx1250", num_cus=256))
    template = SimpleNamespace(maybe_append_choice=lambda choices, **kw: choices.append(kw))
    monkeypatch.setattr(templates, "gfx1250_flex_attention_template", template)
    choices = ["stock"]
    options = {"fwd_BLOCK_M": 16} if case == "invalid_tile" else {"fwd_BLOCK_M": 32, "fwd_BLOCK_N": 64}
    with config.patch({"triton.tlx_mode": mode, "max_autotune": False}):
        templates.append_tlx_flex(choices, [], nodes + [object() for _ in range(6)], [], object(), options, 128, 128)
    if case in ("fp16", "bf16"):
        assert len(choices) == (1 if mode == "force" else 2)
        assert (choices[-1]["BLOCK_M"], choices[-1]["BLOCK_N"]) == (32, 64)
    else:
        assert choices == ["stock"]
