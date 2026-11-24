"""L1 correctness for ``tlx.ops.swiglu_ffn`` on Hopper. Forward only."""

import functools
import math

import pytest
import torch
from triton._internal_testing import is_hopper

pytestmark = pytest.mark.skipif(not is_hopper(), reason="tlx.ops.swiglu_ffn on sm90 requires Hopper")


def _weights(d_model, seed):
    """Scaled so the feed-forward branch is of order 1, well above the FP16 tolerance of the residual sum."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    bound = 1.6 / math.sqrt(d_model)
    w1, w2, w3 = (((torch.rand(d_model, d_model, device="cuda", generator=g) * 2 - 1) * bound).half() for _ in range(3))
    norm = (0.5 + torch.rand(1, d_model, device="cuda", generator=g)).half()
    return norm, w1, w2, w3


def _assert_matches(out, x, ref):
    """The branch, out - x, to a relative tolerance, then the sum."""
    torch.testing.assert_close(out.float() - x.float(), ref.float() - x.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def _reference(x, norm, w1, w2, w3, eps=1e-5):
    """The eager FP16 module (`NormWithInit` then a SwiGLU feed-forward, with residual)."""
    xf = x.float()
    n = (xf / torch.sqrt(xf.square().mean(-1, keepdim=True) + eps) * norm.float()).half()
    return x + (torch.nn.functional.silu(n @ w1.t()) * (n @ w3.t())) @ w2.t()


@pytest.mark.parametrize("tokens", [0, 1, 63, 64, 65, 128, 129, 1000, 40000])
@pytest.mark.parametrize("d_model", [128, 256])
def test_swiglu_ffn_matches_eager(tokens, d_model):
    from triton.tlx.ops import swiglu_ffn
    torch.manual_seed(tokens)
    norm, w1, w2, w3 = _weights(d_model, tokens)
    x = torch.randn(tokens, d_model, device="cuda", dtype=torch.float16)
    _assert_matches(swiglu_ffn(x, norm, w1, w2, w3), x, _reference(x, norm, w1, w2, w3))


def test_swiglu_ffn_strided_inputs():
    """Row-strided x and norm_weight views, as slices of wider tensors."""
    from triton.tlx.ops import swiglu_ffn
    norm, w1, w2, w3 = _weights(256, 4)
    x = torch.randn(1000, 512, device="cuda", dtype=torch.float16)[:, 128:384]
    norm = torch.stack([norm.reshape(-1), norm.reshape(-1)], 1)[:, 0]
    out = swiglu_ffn(x, norm, w1, w2, w3)
    assert out.is_contiguous()
    _assert_matches(out, x, _reference(x, norm, w1, w2, w3))


def test_swiglu_ffn_rejects_misaligned_weights():
    """TMA reads the weights, so a weight whose data does not start on a 16-byte boundary is rejected."""
    from triton.tlx.ops import InvalidInput, swiglu_ffn
    norm, w1, w2, w3 = _weights(256, 0)
    w2_offset = torch.empty(256 * 256 + 1, device="cuda", dtype=torch.float16)[1:].view(256, 256)
    w2_offset.copy_(w2)
    x = torch.randn(10, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput):
        swiglu_ffn(x, norm, w1, w2_offset, w3)


def test_swiglu_ffn_fake_output_matches_real_layout():
    """The launcher is the fake implementation under `triton_op`: for a transposed x it reports the real strides."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from triton.tlx.ops.kernels.swiglu_ffn.sm90 import swiglu_ffn
    norm, w1, w2, w3 = _weights(128, 5)
    x = torch.randn(128, 3, device="cuda", dtype=torch.float16).t()
    args = (x, norm.reshape(-1), w1, w2, w3)
    real = swiglu_ffn(*args, 1e-5)
    with FakeTensorMode() as mode:
        fake = swiglu_ffn(*(mode.from_tensor(t) for t in args), 1e-5)
    assert fake.shape == real.shape and fake.stride() == real.stride() == (128, 1)


def test_swiglu_ffn_epsilon():
    from triton.tlx.ops import swiglu_ffn
    norm, w1, w2, w3 = _weights(256, 1)
    x = torch.randn(300, 256, device="cuda", dtype=torch.float16) * 1e-3
    _assert_matches(swiglu_ffn(x, norm, w1, w2, w3, eps=1e-4), x, _reference(x, norm, w1, w2, w3, eps=1e-4))


def test_swiglu_ffn_rejects_invalid_inputs():
    from triton.tlx.ops import InvalidInput, swiglu_ffn
    norm, w1, w2, w3 = _weights(256, 0)
    x = torch.randn(10, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput):
        swiglu_ffn(x, norm, w1[:128], w2, w3)
    with pytest.raises(InvalidInput):
        swiglu_ffn(x.float(), norm, w1, w2, w3)
    with pytest.raises(InvalidInput):
        swiglu_ffn(x[:, :192], norm[:, :192], w1[:192, :192], w2[:192, :192], w3[:192, :192])
    with pytest.raises(InvalidInput):
        swiglu_ffn(x, norm[:, :128], w1, w2, w3)


@functools.cache
def _traceable_op():
    from triton.tlx.ops.kernels.swiglu_ffn.sm90 import swiglu_ffn

    @torch.library.triton_op("tlx_ops_test::swiglu_ffn_sm90", mutates_args=())
    def swiglu_ffn_sm90(x: torch.Tensor, norm: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                        w3: torch.Tensor) -> torch.Tensor:
        return swiglu_ffn(x, norm.reshape(-1), w1, w2, w3, 1e-5)

    return torch.ops.tlx_ops_test.swiglu_ffn_sm90


class _Ffn(torch.nn.Module):
    """Calls the kernel through a custom op, as a served model would."""

    def __init__(self, weights):
        super().__init__()
        self.op = _traceable_op()
        self.weights = torch.nn.ParameterList(torch.nn.Parameter(t, requires_grad=False) for t in weights)

    def forward(self, x):
        return self.op(x, *self.weights)


def test_swiglu_ffn_compiles_with_dynamic_shapes():
    from torch._dynamo.testing import CompileCounterWithBackend

    weights = _weights(256, 2)
    ffn = _Ffn(weights)
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(ffn, backend=counter, fullgraph=True, dynamic=True)
    for tokens in (300, 1000, 40000):
        x = torch.randn(tokens, 256, device="cuda", dtype=torch.float16)
        torch._dynamo.mark_static(x, 1)
        eager = ffn(x)
        _assert_matches(eager, x, _reference(x, *weights))
        torch.testing.assert_close(compiled(x), eager)
    assert counter.frame_count == 1


def test_swiglu_ffn_compiles_with_transposed_input():
    ffn = _Ffn(_weights(256, 7))
    x = torch.randn(256, 300, device="cuda", dtype=torch.float16).t()
    torch._dynamo.reset()
    torch.testing.assert_close(torch.compile(ffn, fullgraph=True)(x), ffn(x))


@pytest.mark.skipif(torch.__version__ < "2.16.0.dev20261029",
                    reason="older AOTInductor autotunes with a tensor in place of a TMA descriptor argument")
def test_swiglu_ffn_aoti_package_matches_eager(tmp_path):
    """AOTInductor with the autotune settings production lowering uses."""
    ffn = _Ffn(_weights(256, 3))
    x = torch.randn(9000, 256, device="cuda", dtype=torch.float16)
    exported = torch.export.export(ffn, (x, ))
    with torch._inductor.config.patch({
            "triton.autotune_at_compile_time": True,
            "triton.autotune_with_sample_inputs": True,
    }):
        package = torch._inductor.aoti_compile_and_package(exported, package_path=str(tmp_path / "ffn.pt2"))
    torch.testing.assert_close(torch._inductor.aoti_load_package(package)(x), ffn(x))
