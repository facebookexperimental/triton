"""L1 correctness for gfx1250 dense attention and its numerical recovery."""

import pytest
import torch
import triton
from triton._internal_testing import is_hip_gfx1250
from triton.tlx.ops import flash_attn
from triton.tlx.ops.kernels.flash_attn._shapes import CORRECTNESS_SHAPES
from triton.tlx.ops.kernels.flash_attn import gfx1250 as _gfx1250_attention


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("BATCH,H,SEQLEN_Q,SEQLEN_K", [
    (1, 8, 1024, 1024),
    (2, 4, 1024, 1024),
    (1, 16, 2048, 2048),
    (1, 2, 896, 896),
    (1, 1, 640, 640),
    (1, 8, 2048, 2048),
    (1, 2, 4096, 4096),  # underfilled long-K grids
    (1, 2, 63, 65),  # one partial key tile
    (1, 2, 128, 128),  # one full key tile
    (1, 2, 129, 129),  # two key tiles with partial Q/K boundaries
    (2, 2, 129, 257),  # partial Q and K tiles, independent lengths
    (1, 2, 897, 897),  # final K tile contains a single valid row
    (8, 32, 1, 4096),
    (8, 32, 33, 4096),
    (2, 3, 129, 65537),  # long-K fallback across batches/heads
    (1, 32, 65537, 65537),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gfx1250_attn_fwd_tdm_pipelined(BATCH, H, SEQLEN_Q, SEQLEN_K, dtype):
    _check_attention(BATCH, H, SEQLEN_Q, SEQLEN_K, dtype)


def _check_attention(BATCH, H, SEQLEN_Q, SEQLEN_K, dtype):
    torch.manual_seed(0)
    D = 128
    q = torch.randn((BATCH, H, SEQLEN_Q, D), device=triton.runtime.driver.active.get_active_torch_device(), dtype=dtype)
    k = torch.randn((BATCH, H, SEQLEN_K, D), device=triton.runtime.driver.active.get_active_torch_device(), dtype=dtype)
    v = torch.randn_like(k)
    sm_scale = 1.0 / (D**0.5)
    out = flash_attn(q, k, v, sm_scale=sm_scale)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    # Avoid a quadratic reference allocation for long attention, but check every
    # output element for finiteness and sample both sides of tile boundaries.
    if SEQLEN_Q > 2048:
        boundaries = [0, 31, 32, 63, 64, 127, 128, 255, 256, 65535, 65536, SEQLEN_Q - 2, SEQLEN_Q - 1]
        rows = torch.tensor(sorted({row for row in boundaries if 0 <= row < SEQLEN_Q}), device=q.device)
        q_ref = q.index_select(2, rows)
        actual = out.index_select(2, rows)
    else:
        q_ref, actual = q, out
    scores = (q_ref.double() @ k.double().transpose(-1, -2)) * sm_scale
    ref = scores.softmax(dim=-1) @ v.double()
    atol, rtol = (2e-2, 1e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)
    torch.testing.assert_close(actual.double(), ref, atol=atol, rtol=rtol)
    # A loose absolute bound alone misses large relative errors at long lengths.
    tolerance = 0.005 if dtype == torch.bfloat16 else 0.0008
    assert (actual.double() - ref).square().mean().sqrt() < tolerance * ref.square().mean().sqrt()


# Automatic narrow/long-K routes plus forced medium/wide tiles exercise replay
# without allocating an unusually large adversarial query sequence.
@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seqlen_k,block_m,block_n", [(512, None, None), (1025, None, None), (1025, 128, 64),
                                                      (1025, 256, 64), (65537, None, None)])
@pytest.mark.parametrize("recovery_case", ["exp_overflow", "probability_overflow", "common_offset"])
def test_gfx1250_attn_fwd_tdm_pipelined_recovery(dtype, seqlen_k, block_m, block_n, recovery_case, monkeypatch):
    torch.manual_seed(1)
    device = triton.runtime.driver.active.get_active_torch_device()
    heads = 32 if seqlen_k >= 65536 else 2
    q = torch.randn((1, heads, 257, 128), device=device, dtype=dtype)
    k = torch.randn((1, heads, seqlen_k, 128), device=device, dtype=dtype)
    v = torch.randn_like(k).abs().add_(0.25)
    scale = 128**-0.5
    if block_m is not None:
        selected = next(config for config in _gfx1250_attention._CONFIGS
                        if (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"]) == (block_m, block_n))
        monkeypatch.setattr(_gfx1250_attention, "heuristic_config", lambda *args: selected)
    # Mix safe and unsafe heads so replaying the wrong tile cannot repair all
    # outputs accidentally through a different permutation of workgroups.
    q_bad = q[:, -1:] if heads >= 32 else q
    k_bad = k[:, -1:] if heads >= 32 else k
    flash_attn(q, k, v, sm_scale=scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flash_attn(q, k, v, sm_scale=scale)
    assert out.dtype == dtype

    # Reuse one graph across ordinary -> unsafe -> ordinary inputs. The
    # numerical decision must be recomputed on the device for every replay.
    for unsafe in (False, True, False):
        if not unsafe:
            q.normal_()
            k.normal_()
        elif recovery_case == "exp_overflow":
            q_bad.fill_(1)
            k_bad.zero_()
            k_bad[:, :, 128:].fill_(16)
        elif recovery_case == "probability_overflow":
            q_bad.zero_()
            q_bad[..., 0] = 1
            k_bad.zero_()
            k_bad[:, :, 128:, 0] = 200
        else:
            q_bad.zero_()
            q_bad[..., 0] = 256
            q_bad[..., 1] = 1
            k_bad.zero_()
            k_bad[..., 0] = 256
            k_bad[..., 1] = torch.arange(k_bad.shape[2], device=device) % 9 - 4
        ref = ((q.double() @ k.double().transpose(-1, -2)) * scale).softmax(-1) @ v.double()
        for _ in range(3):
            graph.replay()
            assert torch.isfinite(out).all()
            tolerance = 0.005 if dtype == torch.bfloat16 else 0.0008
            torch.testing.assert_close(out.double(), ref, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["offset", "padded", "odd_stride"])
@pytest.mark.parametrize("seqlen_k,block_m,block_n,query_tiles_first", [
    (1025, 32, 128, False),
    (1025, 64, 128, False),
    (1025, 128, 64, False),
    (1025, 256, 64, False),
    pytest.param(1025, 128, 64, True, id="query_first"),
    # Q prefetch overlaps K/V in both the two-tile and longer startup paths.
    (129, 64, 128, False),
    (257, 64, 128, False),
    (257, 32, 128, False),
])
def test_gfx1250_attn_fwd_tdm_pipelined_strides(dtype, layout, seqlen_k, block_m, block_n, query_tiles_first):
    torch.manual_seed(2)
    device = triton.runtime.driver.active.get_active_torch_device()
    batch, heads, sq, sk, dim = 2, 2, 257, seqlen_k, 128

    def inputs(length):
        shape = (batch, heads, length, dim)
        if layout == "offset":
            tensor = torch.empty(batch * heads * length * dim + 1, dtype=dtype, device=device)[1:].view(shape)
        else:
            padding = 1 if layout == "odd_stride" else 16
            tensor = torch.empty((*shape[:-1], dim + padding), dtype=dtype, device=device)[..., :dim]
        return tensor.normal_()

    q, k, v = inputs(sq), inputs(sk), inputs(sk)
    scale = dim**-0.5
    ref = ((q.double() @ k.double().transpose(-1, -2)) * scale).softmax(-1) @ v.double()
    # A partial query tile must preserve storage on both sides of the output.
    storage = torch.full((q.numel() + 512, ), -123.0, device=device, dtype=dtype)
    out = storage[256:-256].view(q.shape)
    grid = (batch, heads, triton.cdiv(sq, block_m))

    def launch():
        _gfx1250_attention.attn_fwd_tdm_pipelined_kernel[grid](q, k, v, out, *q.stride(), *k.stride(), *v.stride(),
                                                               *out.stride(), scale, sq, sk, block_m, block_n, dim,
                                                               USE_FIXED_REFERENCE=True,
                                                               QUERY_TILES_FIRST=query_tiles_first,
                                                               num_warps=2 if block_m == 32 else 4, waves_per_eu=1)

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(5):
        out.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(out).all()
        tolerance = 0.005 if dtype == torch.bfloat16 else 0.0008
        torch.testing.assert_close(out.double(), ref, atol=tolerance, rtol=tolerance)
        assert (out.double() - ref).square().mean().sqrt() < tolerance * ref.square().mean().sqrt()
        assert (storage[:256] == -123.0).all()
        assert (storage[-256:] == -123.0).all()


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("shape", [shape for shape in CORRECTNESS_SHAPES if shape.head_dim == 128 and not shape.causal])
def test_flash_attn_shared_shapes(shape):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[shape.dtype]
    _check_attention(shape.batch, shape.heads, shape.context, shape.context, dtype)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
def test_flash_attn_batch_head_changes_reuse_kernel(monkeypatch):
    compiled = []
    monkeypatch.setattr(triton.knobs.compilation, "listener", lambda **kwargs: compiled.append(kwargs["src"]))
    for batch, heads in ((1, 3), (2, 3), (3, 5)):
        q, k, v = [torch.randn((batch, heads, 769, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        flash_attn(q, k, v)
        if batch == 1:
            first_call = len(compiled)
        else:
            assert len(compiled) == first_call


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("space", ["heuristic", "full", "smoke"])
def test_flash_attn_public_spaces(dtype, space):
    q = torch.randn((1, 2, 129, 128), device="cuda", dtype=dtype)
    k = torch.randn((1, 2, 257, 128), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    out = flash_attn(q, k, v, space=space)
    ref = ((q.double() @ k.double().transpose(-1, -2)) * 128**-0.5).softmax(-1) @ v.double()
    assert out.shape == q.shape and out.dtype == q.dtype and out.device == q.device
    tolerance = .005 if dtype == torch.bfloat16 else .0008
    assert (out.double() - ref).square().mean().sqrt() < tolerance * ref.square().mean().sqrt()


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("invalid", ["causal", "gqa", "dtype", "stride", "scale", "empty", "head_dim", "backward"])
def test_flash_attn_public_rejects_unsupported(invalid):
    from triton.tlx.ops import InvalidInput, UnsupportedBackward
    q, k, v = [torch.randn((1, 2, 129, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    kwargs = {}
    if invalid == "causal":
        kwargs["causal"] = True
    elif invalid == "gqa":
        k, v = k[:, :1], v[:, :1]
    elif invalid == "dtype":
        v = v.half()
    elif invalid == "stride":
        q = q.transpose(2, 3).contiguous().transpose(2, 3)
    elif invalid == "scale":
        kwargs["sm_scale"] = float("nan")
    elif invalid == "empty":
        k, v = k[:, :, :0], v[:, :, :0]
    elif invalid == "head_dim":
        q, k, v = [t[..., :64] for t in (q, k, v)]
    else:
        q.requires_grad_()
    with pytest.raises(UnsupportedBackward if invalid == "backward" else InvalidInput):
        flash_attn(q, k, v, **kwargs)


@pytest.mark.parametrize("batch,heads,seqlen_q,seqlen_k,expected", [
    (8, 32, 1, 4096, (32, 128)),
    (8, 32, 32, 4096, (32, 128)),
    (8, 32, 33, 4096, (64, 128)),
    (8, 32, 64, 4096, (64, 128)),
    (8, 32, 65, 4096, (128, 64)),
    (8, 32, 256, 4096, (256, 64)),
    (8, 32, 1, 128, (32, 128)),
    (1, 1, 32, 128, (32, 128)),
    (1, 1, 33, 4096, (32, 128)),
])
def test_flash_attn_heuristic_respects_query_length(batch, heads, seqlen_q, seqlen_k, expected):
    config = _gfx1250_attention.heuristic_config(batch, heads, seqlen_q, seqlen_k, 256)
    assert (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"]) == expected
    assert config in _gfx1250_attention._prune_configs(_gfx1250_attention._CONFIGS, {"SEQLEN_Q": seqlen_q})


@pytest.mark.parametrize("arch,expected_space", [("gfx1250", "heuristic"), ("gfx950", "full"), ("sm90", "full"),
                                                 ("sm100", "full")])
def test_flash_attn_default_space_is_architecture_specific(monkeypatch, arch, expected_space):
    import contextlib
    from unittest.mock import Mock
    from triton.tlx import ops
    from triton.tlx.ops._catalog import CATALOG

    spec = next(entry for entry in CATALOG if entry.op == "flash_attn" and entry.arch == arch)
    kernel = Mock(return_value=object())
    monkeypatch.setattr(ops, "impl_for", lambda *args, **kwargs: (kernel, spec))
    monkeypatch.setattr(torch.cuda, "device", lambda *args: contextlib.nullcontext())
    q = torch.empty((1, 1, 2, 128), dtype=torch.bfloat16)
    assert ops.flash_attn(q, q, q) is kernel.return_value
    assert kernel.call_args.kwargs == {"space": expected_space}
    if expected_space == "full":
        kernel.reset_mock()
        with pytest.raises(ops.InvalidInput, match="space='heuristic'"):
            ops.flash_attn(q, q, q, space="heuristic")
        kernel.assert_not_called()


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("value", [65504.0, -65504.0])
@pytest.mark.parametrize("seqlen_k", [129, 1025])
def test_flash_attn_fp16_output_range(value, seqlen_k):
    q = torch.zeros((1, 1, 2, 128), device="cuda", dtype=torch.float16)
    q[..., 0] = 1
    k = torch.zeros((1, 1, seqlen_k, 128), device="cuda", dtype=q.dtype)
    k[:, :, 1:, 0] = -7.7265625
    v = torch.full_like(k, value)
    output = flash_attn(q, k, v)
    torch.testing.assert_close(output, torch.full_like(output, value), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_flash_attn_fp16_preserves_nonfinite_output(value):
    q = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.float16)
    output = flash_attn(q, q, torch.full_like(q, value))
    torch.testing.assert_close(output, torch.full_like(output, value), equal_nan=True)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("axis", [0, 1])
def test_flash_attn_large_outer_stride(dtype, axis):
    shape = (3, 1, 128, 128) if axis == 0 else (1, 3, 128, 128)
    strides = list(torch.empty(shape).stride())
    strides[axis] = 2**30 + 128
    tensors = [torch.empty_strided(shape, strides, dtype=dtype, device="cuda") for _ in range(3)]
    for tensor in tensors:
        tensor.copy_(torch.randn(shape, dtype=dtype, device="cuda"))
    q, k, v = tensors
    expected = ((q.double() @ k.double().transpose(-1, -2)) * 128**-.5).softmax(-1) @ v.double()
    output = flash_attn(q, k, v)
    tolerance = .005 if dtype == torch.bfloat16 else .0008
    assert (output.double() - expected).square().mean().sqrt() < tolerance * expected.square().mean().sqrt()
