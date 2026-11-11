"""L1 correctness for ``tlx.ops.mla_window_attn`` on Hopper. Forward only."""

import functools
import itertools
import math

import pytest
import torch
from triton._internal_testing import is_hopper

pytestmark = pytest.mark.skipif(not is_hopper(), reason="tlx.ops.mla_window_attn on sm90 requires Hopper")

LENGTHS = [
    [1],
    [7, 1, 3],
    [200, 0, 77, 1],
    [64, 65, 63, 128, 129],
    [5120, 4000, 1],
]


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _weights(d_model, ranks, seed):
    """Projection weights of the op's shapes, with norm weights away from 1 so their indexing is checked."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def uniform(*shape, bound):
        return ((torch.rand(shape, device="cuda", generator=g) * 2 - 1) * bound).half()

    def norm(n):
        return (0.5 + torch.rand(n, device="cuda", generator=g)).half()

    def lora(rank, out):
        return (uniform(rank, d_model, bound=math.sqrt(1 / (d_model + rank))), norm(rank),
                uniform(out, rank, bound=math.sqrt(1 / (rank + out))))

    q_rank, kv_rank, g_rank = ranks
    return dict(norm=norm(d_model), q=lora(q_rank, d_model), kv=lora(kv_rank, 2 * d_model), gate=lora(g_rank, d_model))


def _rms(x, w, eps):
    xf = x.float()
    return (w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))).to(x.dtype)


def _reference(x, cu, w, heads, window, norm_eps=1e-6, lora_eps=1e-6):
    """The eager FP16 module; attention per sequence in FP32."""
    d_model = x.shape[1]
    head_dim = d_model // heads
    h = _rms(x, w["norm"], norm_eps)

    def lora(p):
        down, norm, up = p
        return _rms(h @ down.t(), norm, lora_eps) @ up.t()

    q = lora(w["q"]).view(-1, heads, head_dim)
    kv = lora(w["kv"]).view(-1, heads, 2 * head_dim)
    gate = torch.sigmoid(lora(w["gate"]))
    attn = torch.zeros_like(q)
    offsets = cu.tolist()
    for a, b in zip(offsets[:-1], offsets[1:]):
        if b == a:
            continue
        qs, ks, vs = (t[a:b].float().transpose(0, 1) for t in (q, kv[..., :head_dim], kv[..., head_dim:]))
        i = torch.arange(b - a, device="cuda")
        outside = (i[None, :] < i[:, None] - window[0]) | (i[None, :] > i[:, None] + window[1])
        scores = (qs @ ks.transpose(1, 2) * head_dim**-0.5).masked_fill(outside, float("-inf"))
        attn[a:b] = (torch.softmax(scores, -1) @ vs).transpose(0, 1).half()
    return x + attn.reshape(-1, d_model) * gate


def _assert_matches(out, x, ref):
    """The attention branch, out - x, to a relative tolerance, then the sum."""
    torch.testing.assert_close(out.float() - x.float(), ref.float() - x.float(), atol=4e-3, rtol=1e-2)
    torch.testing.assert_close(out, ref, atol=4e-3, rtol=1e-2)


def _run(x, cu, w, heads, window, **kwargs):
    from triton.tlx.ops import mla_window_attn
    return mla_window_attn(x, cu, w["norm"], w["q"], w["kv"], w["gate"], num_heads=heads, window_size=window, **kwargs)


def _check(lengths, window=(4, 4), d_model=256, heads=2, ranks=(64, 32, 64), seed=0):
    torch.manual_seed(seed)
    w = _weights(d_model, ranks, seed)
    x = torch.randn(sum(lengths), d_model, device="cuda", dtype=torch.float16)
    cu = _cu(lengths)
    _assert_matches(_run(x, cu, w, heads, window), x, _reference(x, cu, w, heads, window))


@pytest.mark.parametrize("lengths", LENGTHS, ids=lambda lens: "-".join(map(str, lens[:5])))
@pytest.mark.parametrize("window", [(4, 4), (0, 0), (3, 1), (0, 5), (16, 16)], ids=str)
def test_mla_window_attn_lengths_and_windows(lengths, window):
    _check(lengths, window)


@pytest.mark.parametrize("d_model, heads, ranks", [(256, 2, (64, 32, 64)), (256, 4, (16, 16, 32)),
                                                   (128, 2, (32, 64, 16)), (128, 1, (128, 128, 128)),
                                                   (256, 2, (128, 64, 128))])
def test_mla_window_attn_shapes(d_model, heads, ranks):
    _check([300, 5, 900], d_model=d_model, heads=heads, ranks=ranks)


@pytest.mark.parametrize("d_model, heads, ranks, tlx_kernel", [
    (256, 2, (64, 32, 64), True),
    (128, 2, (32, 64, 16), True),
    (256, 2, (128, 64, 128), False),
    (256, 4, (16, 16, 32), False),
])
def test_mla_window_attn_kernel_choice(d_model, heads, ranks, tlx_kernel):
    """Two heads whose weights fit in shared memory take the TLX kernel; other shapes take the per-block kernel."""
    from triton.tlx.ops.kernels.mla_window_attn import sm90

    def launches():
        return [
            sum(len(cache[0])
                for cache in kernel.device_caches.values())
            for kernel in (sm90._mla_window_attn_ws, sm90._mla_window_attn_fwd)
        ]

    before = launches()
    w = _weights(d_model, ranks, 9)
    for kernel in (sm90._mla_window_attn_ws, sm90._mla_window_attn_fwd):
        kernel.device_caches.clear()
    _run(torch.randn(100, d_model, device="cuda", dtype=torch.float16), _cu([100]), w, heads, (4, 4))
    assert launches() == ([1, 0] if tlx_kernel else [0, 1]), before


def test_mla_window_attn_split_kv_descriptor(monkeypatch):
    """The ptxas 12 form of the TLX kernel, which gives k and v their own shared-memory descriptor, under any ptxas."""
    from triton.tlx.ops.kernels.mla_window_attn import sm90
    monkeypatch.setattr(sm90, "_ptxas_before_13", lambda: True)
    _check([300, 5, 900])
    _check([5120, 4000, 1])


def test_mla_window_attn_norm_epsilons():
    w = _weights(256, (64, 32, 64), 3)
    x = torch.randn(500, 256, device="cuda", dtype=torch.float16) * 1e-3
    cu = _cu([250, 250])
    out = _run(x, cu, w, 2, (4, 4), norm_eps=1e-5, lora_eps=1e-4)
    _assert_matches(out, x, _reference(x, cu, w, 2, (4, 4), 1e-5, 1e-4))


def test_mla_window_attn_zero_rows():
    """All-zero rows, as padding gives, score exactly 0 against every key, and their neighbours score 0 against them."""
    w = _weights(256, (64, 32, 64), 4)
    x = torch.randn(1205, 256, device="cuda", dtype=torch.float16)
    x[100:103] = 0
    cu = _cu([300, 5, 900])
    _assert_matches(_run(x, cu, w, 2, (4, 4)), x, _reference(x, cu, w, 2, (4, 4)))


def test_mla_window_attn_strided_inputs():
    """Row-strided x and norm_weight views, as slices of wider tensors."""
    w = _weights(256, (64, 32, 64), 7)
    x = torch.randn(1205, 512, device="cuda", dtype=torch.float16)[:, 128:384]
    w["norm"] = torch.stack([w["norm"], w["norm"]], 1)[:, 0]
    cu = _cu([300, 5, 900])
    out = _run(x, cu, w, 2, (4, 4))
    assert out.is_contiguous()
    _assert_matches(out, x, _reference(x, cu, w, 2, (4, 4)))


def test_mla_window_attn_small_gates():
    """Strongly negative gate logits next to large values. Every token is its own sequence, so the attention output
    is exactly v and the error comes from the gate alone: an FP16 tanh-based sigmoid rounds its small probabilities
    to zero (sigmoid(-10) is 4.5e-5)."""
    w = _weights(256, (64, 32, 64), 9)
    down, norm, up = w["gate"]
    w["gate"] = (down, norm, up * 12)
    down, norm, up = w["kv"]
    up = up.clone()
    for head in range(2):
        up[head * 256 + 128:(head + 1) * 256] *= 256
    w["kv"] = (down, norm, up)
    x = torch.randn(600, 256, device="cuda", dtype=torch.float16)
    cu = _cu([1] * 600)
    out, expected = _run(x, cu, w, 2, (4, 4)), _reference(x, cu, w, 2, (4, 4))
    # With up-projection weights this large, a small v is a cancellation that magnifies FP16 rounding differences in
    # the low-rank projection; compare where |v| > 4, which leaves the gate's precision as the only error.
    down, norm, up = w["kv"]
    v = (_rms(_rms(x, w["norm"], 1e-6) @ down.t(), norm, 1e-6) @ up.t()).view(600, 2, 2, 128)[:, :, 1].reshape(600, 256)
    keep = v.abs() > 4
    torch.testing.assert_close((out.float() - x.float())[keep], (expected.float() - x.float())[keep], atol=4e-3,
                               rtol=1e-2)


def test_mla_window_attn_rejects_misaligned_weights():
    """TMA reads the projection weights, so one whose data does not start on a 16-byte boundary is rejected."""
    from triton.tlx.ops import InvalidInput
    w = _weights(256, (64, 32, 64), 0)
    up = torch.empty(2 * 256 * 32 + 1, device="cuda", dtype=torch.float16)[1:].view(2 * 256, 32)
    up.copy_(w["kv"][2])
    w["kv"] = (w["kv"][0], w["kv"][1], up)
    with pytest.raises(InvalidInput):
        _run(torch.randn(10, 256, device="cuda", dtype=torch.float16), _cu([10]), w, 2, (4, 4))


def test_mla_window_attn_fake_output_matches_real_layout():
    """The launcher is the fake implementation under `triton_op`: for a transposed x it reports the real strides."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from triton.tlx.ops.kernels.mla_window_attn.sm90 import mla_window_attn
    w = _weights(256, (64, 32, 64), 1)
    x = torch.randn(256, 40, device="cuda", dtype=torch.float16).t()
    flat = [x, _cu([40]), w["norm"], *w["q"], *w["kv"], *w["gate"]]

    def call(t):
        return mla_window_attn(t[0], t[1], t[2], tuple(t[3:6]), tuple(t[6:9]), tuple(t[9:12]), 2, (4, 4), 1e-6, 1e-6)

    real = call(flat)
    with FakeTensorMode() as mode:
        fake = call([mode.from_tensor(t) for t in flat])
    assert fake.shape == real.shape and fake.stride() == real.stride() == (256, 1)


def test_mla_window_attn_empty_input():
    w = _weights(256, (64, 32, 64), 0)
    x = torch.empty(0, 256, device="cuda", dtype=torch.float16)
    assert _run(x, _cu([0]), w, 2, (4, 4)).shape == (0, 256)


def test_mla_window_attn_rejects_invalid_inputs():
    from triton.tlx.ops import InvalidInput
    w = _weights(256, (64, 32, 64), 0)
    x = torch.randn(10, 256, device="cuda", dtype=torch.float16)
    cu = _cu([10])
    for window in [(-1, 4), (17, 16), 4, (2.0, 1)]:
        with pytest.raises(InvalidInput):
            _run(x, cu, w, 2, window)
    with pytest.raises(InvalidInput):
        _run(x, cu, w, 3, (4, 4))
    with pytest.raises(InvalidInput):
        _run(x, cu.long(), w, 2, (4, 4))
    with pytest.raises(InvalidInput):
        _run(x.float(), cu, w, 2, (4, 4))
    bad = dict(w, kv=(w["kv"][0], w["kv"][1], w["kv"][2][:256]))
    with pytest.raises(InvalidInput):
        _run(x, cu, bad, 2, (4, 4))
    bad = dict(w, q=(w["q"][0][:48], w["q"][1][:48], w["q"][2][:, :48]))
    with pytest.raises(InvalidInput):
        _run(x, cu, bad, 2, (4, 4))


@functools.cache
def _traceable_op():
    from triton.tlx.ops.kernels.mla_window_attn.sm90 import mla_window_attn

    @torch.library.triton_op("tlx_ops_test::mla_window_attn_sm90", mutates_args=())
    def mla_window_attn_sm90(x: torch.Tensor, cu: torch.Tensor, norm: torch.Tensor, q0: torch.Tensor, q1: torch.Tensor,
                             q2: torch.Tensor, kv0: torch.Tensor, kv1: torch.Tensor, kv2: torch.Tensor,
                             g0: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        return mla_window_attn(x, cu, norm, (q0, q1, q2), (kv0, kv1, kv2), (g0, g1, g2), 2, (4, 4), 1e-6, 1e-6)

    return torch.ops.tlx_ops_test.mla_window_attn_sm90


class _Block(torch.nn.Module):
    """An attention block that calls the kernel through a custom op, as a served model would."""

    def __init__(self, w):
        super().__init__()
        self.op = _traceable_op()
        self.weights = torch.nn.ParameterList(
            torch.nn.Parameter(t, requires_grad=False) for t in (w["norm"], *w["q"], *w["kv"], *w["gate"]))

    def forward(self, x, cu):
        return self.op(x, cu, *self.weights)


def test_mla_window_attn_compiles_with_dynamic_shapes():
    """One compiled graph serves several token counts and batch sizes."""
    from torch._dynamo.testing import CompileCounterWithBackend

    w = _weights(256, (64, 32, 64), 5)
    block = _Block(w)
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(block, backend=counter, fullgraph=True, dynamic=True)
    for lens in ([200, 300], [65, 129, 257, 0], [5120, 4000, 1]):
        x = torch.randn(sum(lens), 256, device="cuda", dtype=torch.float16)
        torch._dynamo.mark_static(x, 1)
        cu = _cu(lens)
        eager = block(x, cu)
        _assert_matches(eager, x, _reference(x, cu, w, 2, (4, 4)))
        torch.testing.assert_close(compiled(x, cu), eager)
    assert counter.frame_count == 1


def test_mla_window_attn_compiles_with_transposed_input():
    block = _Block(_weights(256, (64, 32, 64), 8))
    x = torch.randn(256, 300, device="cuda", dtype=torch.float16).t()
    cu = _cu([100, 200])
    torch._dynamo.reset()
    torch.testing.assert_close(torch.compile(block, fullgraph=True)(x, cu), block(x, cu))


@pytest.mark.skipif(torch.__version__ < "2.16.0.dev20261029",
                    reason="older AOTInductor autotunes with a tensor in place of a TMA descriptor argument")
def test_mla_window_attn_aoti_package_matches_eager(tmp_path):
    """AOTInductor with the autotune settings production lowering uses."""
    block = _Block(_weights(256, (64, 32, 64), 6))
    lens = [5120, 3000, 700, 1]
    x = torch.randn(sum(lens), 256, device="cuda", dtype=torch.float16)
    cu = _cu(lens)
    exported = torch.export.export(block, (x, cu))
    with torch._inductor.config.patch({
            "triton.autotune_at_compile_time": True,
            "triton.autotune_with_sample_inputs": True,
    }):
        package = torch._inductor.aoti_compile_and_package(exported, package_path=str(tmp_path / "mla.pt2"))
    torch.testing.assert_close(torch._inductor.aoti_load_package(package)(x, cu), block(x, cu))
