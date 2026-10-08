"""Public ``tlx.ops.flash_attn`` coverage for gfx950."""

import contextlib
import subprocess
import sys
from unittest import mock

import pytest
import torch

from triton._internal_testing import is_hip_cdna4


@pytest.fixture
def flash_attention_registry():
    attention = pytest.importorskip("torch.nn.attention")
    names = (
        "activate_flash_attention_impl",
        "current_flash_attention_impl",
        "register_flash_attention_impl",
        "restore_flash_attention_impl",
    )
    if not all(hasattr(attention, name) for name in names) or not hasattr(torch.library, "get_kernel"):
        pytest.skip("requires PyTorch's conditional FlashAttention provider APIs")


@pytest.mark.usefixtures("flash_attention_registry")
@pytest.mark.parametrize("case", ("measured", "different_lengths", "unknown_shape", "shifted_q", "shifted_v"))
def test_varlen_provider_admission_requires_measured_workload(monkeypatch, case):
    from triton.tlx import pytorch as provider

    q = torch.empty((96, 2, 128), dtype=torch.bfloat16)
    k = torch.empty((192, 2, 128), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    cu_q = torch.tensor([0, 32, 96], dtype=torch.int32)
    q_offsets, k_offsets = [0, 32, 96], [0, 64, 192]
    # This measured fixture has the same totals as [0, 64, 96], but a
    # different query-length distribution. Totals alone must not admit it.
    signature = ((96, 2, 128), (192, 2, 128), 2, 64, 128, False)
    monkeypatch.setattr(provider, "_PERFORMANCE_VALIDATED_VARLEN_SIGNATURES", {
        signature: "22c09c3283a266bfb0745905f4d1a625bdbeedd59a3ece10c333921b3b163b3b",
    })
    if case == "different_lengths":
        q_offsets = [0, 64, 96]
    elif case == "unknown_shape":
        q = q[:80]
    elif case == "shifted_q":
        q = torch.empty(q.numel() + 1, dtype=q.dtype)[1:].view_as(q)
    elif case == "shifted_v":
        v = torch.empty(v.numel() + 1, dtype=v.dtype)[1:].view_as(v)

    admitted = provider._is_varlen_performance_validated(q, k, cu_q, 64, 128, False, q_offsets, k_offsets, (v, ))
    assert admitted is (case == "measured")


@contextlib.contextmanager
def _varlen_provider():
    from triton.tlx import pytorch as provider
    import torch.nn.attention as attention

    previous = attention.current_flash_attention_impl()
    attention.activate_flash_attention_impl(provider.PROVIDER_NAME)
    try:
        yield provider
    finally:
        attention.restore_flash_attention_impl()
        if previous is not None:
            attention.activate_flash_attention_impl(previous)


def _varlen_aten_case(*, causal=False, gqa=False, empty=False, expanded_grad=False):
    cu_q = torch.tensor([0, 0 if empty else 32, 96], device="cuda", dtype=torch.int32)
    cu_k = cu_q if causal else torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)
    max_q = 96 if empty else 64
    max_k = max_q if causal else 128
    generator = torch.Generator(device="cuda").manual_seed(3775)
    q = torch.randn((96, 2, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn((96 if causal else 192, 1 if gqa else 2, 128), device="cuda", dtype=q.dtype, generator=generator)
    v = torch.randn(k.shape, device="cuda", dtype=q.dtype, generator=generator)
    out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(q, k, v, cu_q, cu_k, max_q, max_k, 0.0, causal,
                                                                       False)
    do = torch.ones((), dtype=q.dtype, device=q.device).expand_as(q) if expanded_grad else torch.randn(
        q.shape, device=q.device, dtype=q.dtype, generator=generator)
    return (do, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, 0.0, causal, rng, unused)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.usefixtures("flash_attention_registry")
@pytest.mark.parametrize(("causal", "gqa", "expanded_grad"),
                         ((False, False, False), (False, True, False), (True, False, True)))
def test_varlen_provider_routes_fp32_backward_gfx950(monkeypatch, causal, gqa, expanded_grad):
    from triton.tlx import pytorch as provider
    from triton.tlx.ops.kernels.flash_attn_varlen import gfx950_bwd as gfx950_varlen_bwd

    args = _varlen_aten_case(causal=causal, gqa=gqa, expanded_grad=expanded_grad)
    kwargs = {"window_size_left": -1, "window_size_right": 0 if causal else -1}
    reference = torch.ops.aten._flash_attention_backward(*args, **kwargs)
    # Exercise the real operator adapter independently of which production
    # benchmark signatures have earned transparent routing.
    monkeypatch.setattr(provider, "_is_varlen_performance_validated", lambda *args: True, raising=False)
    calls = []
    original = gfx950_varlen_bwd.fa_varlen_backward

    def observe(*args, **kwargs):
        calls.append(kwargs["dq_atomic_fp32"])
        return original(*args, **kwargs)

    monkeypatch.setattr(gfx950_varlen_bwd, "fa_varlen_backward", observe)
    with _varlen_provider():
        actual = torch.ops.aten._flash_attention_backward(*args, **kwargs)
    assert calls == [True]
    for result, expected in zip(actual, reference, strict=True):
        error = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(expected.float())
        assert error.item() < 1e-2


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.usefixtures("flash_attention_registry")
def test_varlen_provider_empty_sequence_uses_native_gfx950(monkeypatch):
    from triton.tlx import pytorch as provider
    from triton.tlx.ops.kernels.flash_attn_varlen import gfx950_bwd as gfx950_varlen_bwd

    args = _varlen_aten_case(empty=True)
    reference = torch.ops.aten._flash_attention_backward(*args)
    monkeypatch.setattr(provider, "_is_varlen_performance_validated", lambda *args: True, raising=False)

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported empty sequence reached TLX")

    monkeypatch.setattr(gfx950_varlen_bwd, "fa_varlen_backward", unexpected)
    with _varlen_provider():
        actual = torch.ops.aten._flash_attention_backward(*args)
    torch.testing.assert_close(actual[0], reference[0], rtol=1e-2, atol=1e-2)
    # Native CK can leave dK/dV rows for an empty query sequence unwritten.
    # Compare its nonempty sequence; the guard above checks that TLX was not called.
    first_nonempty_kv = args[7][1].item()
    for result, expected in zip(actual[1:], reference[1:], strict=True):
        torch.testing.assert_close(result[first_nonempty_kv:], expected[first_nonempty_kv:], rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.usefixtures("flash_attention_registry")
@pytest.mark.parametrize("reason", ("unmeasured", "deterministic", "causal_gqa", "window", "capture"))
def test_varlen_provider_preserves_native_fallback_gfx950(monkeypatch, reason):
    from triton.tlx import pytorch as provider
    from triton.tlx.ops.kernels.flash_attn_varlen import gfx950_bwd as gfx950_varlen_bwd

    args = _varlen_aten_case(causal=reason == "causal_gqa", gqa=reason == "causal_gqa")
    kwargs = {"window_size_left": 16, "window_size_right": 16} if reason == "window" else {}
    reference = torch.ops.aten._flash_attention_backward(*args, **kwargs)
    monkeypatch.setattr(provider, "_is_varlen_performance_validated", lambda *args: reason != "unmeasured")
    if reason == "deterministic":
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    if reason == "capture":
        # Verify the branch before any host offset copy. Real graph replay
        # has separate coverage; CK's native capture support varies by build.
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported call prepared a TLX plan")

    monkeypatch.setattr(gfx950_varlen_bwd, "prepare_varlen_backward", unexpected)
    with _varlen_provider():
        actual = torch.ops.aten._flash_attention_backward(*args, **kwargs)
    for result, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.usefixtures("flash_attention_registry")
@pytest.mark.parametrize("stage", ("backward", "preparation"))
@pytest.mark.parametrize("error_type", (RuntimeError, TypeError, ValueError))
def test_varlen_provider_does_not_retry_kernel_failure_gfx950(monkeypatch, stage, error_type):
    from triton.tlx import pytorch as provider
    from triton.tlx.ops.kernels.flash_attn_varlen import gfx950_bwd as gfx950_varlen_bwd

    args = _varlen_aten_case()
    monkeypatch.setattr(provider, "_is_varlen_performance_validated", lambda *args: True)

    def fail(*args, **kwargs):
        raise error_type("varlen kernel failure")

    name = "fa_varlen_backward" if stage == "backward" else "prepare_varlen_backward"
    monkeypatch.setattr(gfx950_varlen_bwd, name, fail)
    with _varlen_provider(), pytest.raises(error_type, match="varlen kernel failure"):
        torch.ops.aten._flash_attention_backward(*args)


def _gfx950_device_indices():
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").startswith("gfx950")
    ]


def _has_two_gfx950_devices():
    return len(_gfx950_device_indices()) >= 2


def test_gfx950_device_indices_ignore_non_rocm_and_other_amd_devices():
    properties = (
        mock.Mock(spec=["name"]),
        mock.Mock(gcnArchName="gfx950:sramecc+:xnack-"),
        mock.Mock(gcnArchName="gfx942:sramecc+:xnack-"),
        mock.Mock(gcnArchName="gfx950"),
    )
    with (
            mock.patch.object(torch.cuda, "device_count", return_value=len(properties)),
            mock.patch.object(torch.cuda, "get_device_properties", side_effect=lambda index: properties[index]),
    ):
        assert _gfx950_device_indices() == [1, 3]
        assert _has_two_gfx950_devices()


def test_flash_attn_gfx950_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs

    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("flash_attn", "gfx950")]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.impl == "triton.tlx.ops.kernels.flash_attn.gfx950:flash_attn"
    assert spec.supports_backward

    check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=64)
    check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=128)
    with pytest.raises(InvalidInput, match="does not support float16"):
        check_inputs(spec, dtype=torch.float16, HEAD_DIM=64)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=256)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_catalog_resolves_public_implementation():
    from triton.tlx.ops._catalog import impl_for
    from triton.tlx.ops.kernels.flash_attn.gfx950 import flash_attn

    implementation, spec = impl_for("flash_attn", arch="gfx950")
    assert implementation is flash_attn
    assert spec.supports_backward


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize(
    ("requires_grad", "grad_enabled"),
    ((False, True), (True, False)),
)
def test_flash_attn_gfx950_inference_skips_backward_state(requires_grad, grad_enabled):
    from triton.tlx.ops.kernels.flash_attn import gfx950

    q, k, v = _qkv(64, requires_grad=requires_grad, context=256)
    sentinel = object()
    with (
            mock.patch.object(gfx950, "_flash_attn_inference", return_value=sentinel) as inference,
            mock.patch.object(gfx950._attention, "apply") as differentiable,
            torch.set_grad_enabled(grad_enabled),
    ):
        assert gfx950.flash_attn(q, k, v, space="smoke") is sentinel

    inference.assert_called_once_with(q, k, v, 64**-0.5, False, "smoke")
    differentiable.assert_not_called()


@pytest.mark.skipif(not _has_two_gfx950_devices(), reason="requires two gfx950 GPUs")
def test_flash_attn_gfx950_uses_input_device_when_current_device_differs():
    previous_device, query_device = _gfx950_device_indices()[:2]
    # Keep an illegal cross-device launch from poisoning the pytest process if
    # this regresses. The child deliberately leaves one gfx950 current while
    # the public op's inputs live on another gfx950.
    script = r"""
import sys

import torch

from triton.tlx.ops import flash_attn

previous_device = int(sys.argv[1])
query_device = int(sys.argv[2])
torch.cuda.set_device(previous_device)
q, k, v = [
    torch.randn(
        (1, 1, 256, 64),
        device=f"cuda:{query_device}",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    for _ in range(3)
]
assert torch.cuda.current_device() == previous_device

out = flash_attn(q, k, v, space="smoke")
assert out.device == q.device
assert torch.cuda.current_device() == previous_device
torch.cuda.synchronize(query_device)
torch.cuda.synchronize(previous_device)

grad_out = torch.randn_like(out)
out.backward(grad_out)
torch.cuda.synchronize(query_device)
torch.cuda.synchronize(previous_device)
assert torch.cuda.current_device() == previous_device
assert q.grad.device == k.grad.device == v.grad.device == q.device

rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
with torch.cuda.device(query_device):
    reference = torch.nn.functional.scaled_dot_product_attention(rq, rk, rv)
    reference.backward(grad_out)
torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)
for actual, expected in zip((q.grad, k.grad, v.grad), (rq.grad, rk.grad, rv.grad), strict=True):
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(expected.float())
    assert relative_l2.item() < 5e-3
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(previous_device),
         str(query_device)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _qkv(head_dim, *, requires_grad, context=None):
    context = context or (512 if head_dim == 64 else 1024)
    return [
        torch.randn(
            (1, 4, context, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        ).requires_grad_(requires_grad) for _ in range(3)
    ]


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_gfx950_forward_returns_natural_log_lse(causal):
    from triton.tlx.ops.kernels.flash_attn.gfx950 import _flash_attn_forward

    torch.manual_seed(1)
    q, k, v = _qkv(64, requires_grad=False, context=256)
    scale = q.shape[-1]**-0.5

    _out, lse = _flash_attn_forward(q, k, v, scale, causal, "smoke")
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        causal_mask = torch.ones((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool).triu(1)
        scores.masked_fill_(causal_mask, float("-inf"))
    expected = torch.logsumexp(scores, dim=-1)

    assert lse.dtype is torch.float32
    assert lse.shape == q.shape[:-1]
    torch.testing.assert_close(lse, expected, atol=2e-2, rtol=2e-3)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_gfx950_forward_and_backward(head_dim, causal):
    from triton.tlx.ops import flash_attn

    torch.manual_seed(0)
    q, k, v = _qkv(head_dim, requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    if head_dim == 64 and not causal:
        grad_out = torch.randn((*q.shape[:-1], 2 * head_dim), device=q.device, dtype=q.dtype)[..., ::2]
        assert not grad_out.is_contiguous()
    else:
        grad_out = torch.randn_like(q)
    scale = None if causal else 0.75 * head_dim**-0.5

    if head_dim == 64 and not causal:
        # Exercise the public default and the STORE_LSE/ALLOW_PRESCALE_Q
        # autotuner-key ABI; the remaining cases use the smaller smoke space.
        out = flash_attn(q, k, v, causal=causal, sm_scale=scale)
    else:
        out = flash_attn(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    reference = torch.nn.functional.scaled_dot_product_attention(
        rq,
        rk,
        rv,
        is_causal=causal,
        scale=scale,
    )

    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)
    out.backward(grad_out)
    reference.backward(grad_out)

    for name, actual, expected in zip(
        ("dq", "dk", "dv"),
        (q.grad, k.grad, v.grad),
        (rq.grad, rk.grad, rv.grad),
            strict=True,
    ):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        tolerance = 5e-3 if head_dim == 64 else 1e-2
        assert relative_l2.item() < tolerance, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_honors_deterministic_algorithms():
    from triton.tlx.ops import flash_attn
    from triton.tlx.ops.kernels.flash_attn import gfx950, gfx950_bwd

    q, k, v = _qkv(128, requires_grad=True, context=256)
    forward_state = (
        torch.empty_like(q),
        torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32),
    )
    backward_result = tuple(torch.empty_like(tensor) for tensor in (q, k, v))
    grad_out = torch.empty_like(q)
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with (
                mock.patch.object(gfx950, "_flash_attn_forward", return_value=forward_state),
                mock.patch.object(gfx950_bwd, "fa_backward", return_value=backward_result) as backward,
        ):
            torch.use_deterministic_algorithms(False)
            with mock.patch.object(gfx950_bwd, "fa_backward_is_deterministic") as classify_determinism:
                out = flash_attn(q, k, v, space="smoke")
                out.backward(grad_out)
            classify_determinism.assert_not_called()
            backward.assert_called_once()

            backward.reset_mock()
            torch.use_deterministic_algorithms(True)
            out = flash_attn(q, k, v, space="smoke")
            with pytest.raises(RuntimeError, match="does not have a deterministic implementation"):
                out.backward(grad_out)
            backward.assert_not_called()

            torch.use_deterministic_algorithms(True, warn_only=True)
            out = flash_attn(q, k, v, space="smoke")
            with pytest.warns(UserWarning, match="does not have a deterministic implementation"):
                out.backward(grad_out)
            backward.assert_called_once()

            backward.reset_mock()
            torch.use_deterministic_algorithms(True)
            with mock.patch.object(gfx950_bwd, "fa_backward_is_deterministic", return_value=True):
                out = flash_attn(q, k, v, space="smoke")
                out.backward(grad_out)
            backward.assert_called_once()
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "family", "expected"),
    (
        ((1, 4, 256, 64), (1, 4, 256, 64), "noncausal_direct_n256", True),
        ((1, 4, 4096, 64), (1, 4, 4096, 64), "noncausal_fused_n256", False),
        ((16, 27, 256, 128), (16, 27, 256, 128), None, False),
        ((16, 27, 200, 128), (16, 27, 200, 128), None, True),
        ((32, 1, 2600, 256), (32, 1, 2600, 256), None, True),
    ),
)
def test_flash_attn_gfx950_backward_determinism_matches_route(q_shape, k_shape, family, expected):
    from triton.tlx.ops.kernels.flash_attn import gfx950_bwd

    q = mock.Mock(shape=q_shape)
    k = mock.Mock(shape=k_shape)
    tensors = (q, k, mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock())
    dispatch = mock.Mock(family=family)
    with (
            mock.patch.object(gfx950_bwd, "_validate_inputs"),
            mock.patch.object(gfx950_bwd, "_select_d64_dispatch_for_device", return_value=dispatch) as select_d64,
    ):
        actual = gfx950_bwd.fa_backward_is_deterministic(*tensors, 0.125, False)

    assert actual is expected
    assert select_d64.call_count == int(q_shape[-1] == 64)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("exact", (False, True), ids=("split", "exact"))
def test_flash_attn_gfx950_short_d128_forward_and_backward(causal, exact, monkeypatch):
    from triton.tlx.ops import flash_attn
    from triton.tlx.ops.kernels.flash_attn import gfx950_bwd

    route_options = (
        gfx950_bwd._D128_EXACT_ENABLE_ENV,
        gfx950_bwd._D128_PERSISTENT_ENABLE_ENV,
        gfx950_bwd._D128_PERSISTENT_PIPE_ENABLE_ENV,
        gfx950_bwd._D128_SINK_INSTS_ENV,
        gfx950_bwd._D128_REGCLASS_PRIORITY_ENV,
        gfx950_bwd._D128_REVERSE_LOCAL_ENV,
    )
    for option in route_options:
        monkeypatch.setenv(option, "0")
    monkeypatch.setenv(gfx950_bwd._D128_EXACT_ENABLE_ENV, str(int(exact)))

    shape = (16, 27, 200, 128)
    generator = torch.Generator(device="cuda").manual_seed(3735 + 2 * int(exact) + int(causal))
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator).requires_grad_() for _ in range(3)
    ]
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    grad_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    out = flash_attn(q, k, v, causal=causal, space="smoke")
    reference = torch.nn.functional.scaled_dot_product_attention(rq, rk, rv, is_causal=causal)
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)

    out.backward(grad_out)
    reference.backward(grad_out)
    for name, actual, expected in zip(
        ("dq", "dk", "dv"),
        (q.grad, k.grad, v.grad),
        (rq.grad, rk.grad, rv.grad),
            strict=True,
    ):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_rejects_fp16():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = [torch.randn((1, 1, 128, 64), device="cuda", dtype=torch.float16) for _ in range(3)]
    with pytest.raises(InvalidInput, match="does not support float16"):
        flash_attn(q, k, v, space="smoke")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_rejects_unknown_space():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(64, requires_grad=False)
    with pytest.raises(InvalidInput, match="does not provide space"):
        flash_attn(q, k, v, space="heuristic")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("requires_grad", (False, True))
@pytest.mark.parametrize(
    ("invalid", "match"),
    (
        ("rank", "rank-4"),
        ("shape", "same shape"),
        ("dtype", "same dtype"),
    ),
)
def test_flash_attn_gfx950_invalid_inputs_use_public_exception(invalid, match, requires_grad):
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(64, requires_grad=requires_grad, context=256)
    if invalid == "rank":
        q = q[0]
        k = k[0]
        v = v[0]
    elif invalid == "shape":
        k = k[..., :-1, :].contiguous()
    else:
        k = k.to(torch.float16).requires_grad_(requires_grad)

    with pytest.raises(InvalidInput, match=match):
        flash_attn(q, k, v, space="smoke")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_keeps_inference_only_shapes():
    from triton.tlx.ops import flash_attn

    q, k, v = _qkv(64, requires_grad=False, context=192)
    out = flash_attn(q, k, v, space="smoke")
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_rejects_unsupported_backward_shape():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(64, requires_grad=True, context=192)
    with pytest.raises(InvalidInput, match="supported MHA shapes"):
        flash_attn(q, k, v, space="smoke")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("operand", ("q", "k", "v"))
def test_flash_attn_gfx950_rejects_noncontiguous_backward_inputs(operand):
    from triton.tlx.ops import InvalidInput, flash_attn

    tensors = _qkv(64, requires_grad=True, context=256)
    padded = torch.randn(
        (*tensors[0].shape[:-1], 128),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    tensors[("q", "k", "v").index(operand)] = padded[..., ::2]
    assert not tensors[("q", "k", "v").index(operand)].is_contiguous()

    with pytest.raises(InvalidInput, match=rf"{operand} must be contiguous"):
        flash_attn(*tensors, space="smoke")
