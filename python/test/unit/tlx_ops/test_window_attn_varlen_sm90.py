"""L1 correctness for ``tlx.ops.window_attn_varlen`` on Hopper. Forward only."""

import functools
import itertools

import pytest
import torch
from triton._internal_testing import is_hopper
from triton.tlx.ops.kernels.window_attn_varlen._shapes import CORRECTNESS_SHAPES, seqlens

pytestmark = pytest.mark.skipif(not is_hopper(), reason="tlx.ops.window_attn_varlen on sm90 requires Hopper")

TOLERANCE = {torch.float16: dict(atol=3e-3, rtol=1e-2), torch.bfloat16: dict(atol=2e-2, rtol=2e-2)}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _reference(q, k, v, cu, window_size, scale):
    """FP32 attention per sequence; row ``i`` sees keys ``i - left .. i + right`` of its sequence."""
    left, right = window_size
    out = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    group = q.shape[1] // k.shape[1]
    bounds = cu.tolist()
    for b in range(len(bounds) - 1):
        s, e = bounds[b], bounds[b + 1]
        if s == e:
            continue
        rows = torch.arange(e - s, device=q.device)[:, None]
        cols = torch.arange(e - s, device=q.device)[None, :]
        scores = torch.einsum("qhd,khd->hqk", q[s:e].float(), k[s:e].float().repeat_interleave(group, 1)) * scale
        scores = scores.masked_fill((cols < rows - left) | (cols > rows + right), float("-inf"))
        out[s:e] = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), v[s:e].float().repeat_interleave(group, 1))
    return out


def _inputs(lengths, heads, kv_heads, head_dim, dtype):
    tokens = sum(lengths)
    q = torch.randn(tokens, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=dtype)
    return q, k, v, _cu(lengths)


def _check(lengths, window_size, heads=1, kv_heads=None, head_dim=128, dtype=torch.float16, sm_scale=None):
    from triton.tlx.ops import window_attn_varlen

    q, k, v, cu = _inputs(lengths, heads, kv_heads or heads, head_dim, dtype)
    out = window_attn_varlen(q, k, v, cu, window_size, sm_scale)
    ref = _reference(q, k, v, cu, window_size, head_dim**-0.5 if sm_scale is None else sm_scale)
    assert out.dtype == dtype and out.shape == q.shape and out.is_contiguous()
    torch.testing.assert_close(out.float(), ref, **TOLERANCE[dtype])


@pytest.mark.parametrize("batch,min_seqlen,max_seqlen,heads,kv_heads,head_dim,left,right,dtype_name",
                         CORRECTNESS_SHAPES)
def test_window_attn_varlen_shapes(batch, min_seqlen, max_seqlen, heads, kv_heads, head_dim, left, right, dtype_name):
    torch.manual_seed(20)
    _check(seqlens(batch, min_seqlen, max_seqlen), (left, right), heads=heads, kv_heads=kv_heads, head_dim=head_dim,
           dtype=DTYPES[dtype_name])


@pytest.mark.parametrize(
    "lengths",
    [
        [1],
        [2],
        # Sequences shorter than, equal to and one past the window and the 64-row query and key blocks.
        [1, 2, 63, 64, 65, 127, 128, 129, 130, 300],
        # Empty sequences, first, inside and last.
        [0, 5, 0, 300, 0],
        # The longest sequences the ranking model sends, next to a one-row one.
        [12460, 1, 5000],
        # More sequences than the row lookup reads per load.
        [700] * 40,
    ],
)
def test_window_attn_varlen_lengths(lengths):
    torch.manual_seed(20)
    _check(lengths, (64, 64))


@pytest.mark.parametrize("window_size", [(0, 0), (5, 0), (0, 77), (31, 200), (128, 128), (1, 255), (256, 0)])
def test_window_attn_varlen_windows(window_size):
    torch.manual_seed(20)
    _check([1000, 37, 2000, 1], window_size)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,kv_heads", [(2, 2), (4, 2), (8, 1)])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_window_attn_varlen_heads(heads, kv_heads, head_dim, dtype):
    torch.manual_seed(20)
    _check([300, 700, 64], (16, 48), heads=heads, kv_heads=kv_heads, head_dim=head_dim, dtype=dtype)


def test_window_attn_varlen_matches_flash_attention_2():
    """Same window semantics as PyTorch's FlashAttention-2 varlen forward with `window_size_left/right`."""
    from triton.tlx.ops import window_attn_varlen

    torch.manual_seed(20)
    lengths = [12460, 1, 5000, 129, 64]
    q, k, v, cu = _inputs(lengths, 1, 1, 128, torch.float16)
    ref = torch.ops.aten._flash_attention_forward(q, k, v, cu, cu, max(lengths), max(lengths), 0.0, False, False,
                                                  window_size_left=64, window_size_right=64)[0]
    torch.testing.assert_close(window_attn_varlen(q, k, v, cu, (64, 64)), ref, atol=4e-3, rtol=1e-2)


@pytest.mark.parametrize("sm_scale", [0.05, 1.0, 11.313708498984761])
def test_window_attn_varlen_custom_scale(sm_scale):
    torch.manual_seed(20)
    _check([700, 300], (64, 64), sm_scale=sm_scale)


def test_window_attn_varlen_scale_underflow_gives_uniform_weights():
    """A scale whose float32 ``sm_scale * log2(e)`` underflows to 0 gives every visible key the same weight."""
    from triton.tlx.ops import window_attn_varlen

    ones = torch.ones(100, 1, 128, device="cuda", dtype=torch.float16)
    out = window_attn_varlen(ones, ones, ones, _cu([100]), (64, 64), 1e-50)
    torch.testing.assert_close(out, ones, atol=0, rtol=0)
    torch.manual_seed(20)
    lengths = [300, 5, 1, 200]
    q, k, v, cu = _inputs(lengths, 1, 1, 128, torch.float16)
    out = window_attn_varlen(q, k, v, cu, (64, 64), 1e-50)
    torch.testing.assert_close(out.float(), _reference(q, k, v, cu, (64, 64), 0.0), **TOLERANCE[torch.float16])


def test_window_attn_varlen_scale_underflow_ignores_score_size():
    """Equal weights also when the scores are large: BF16 scores 0 and 2**126 at a scale of 1e-50."""
    from triton.tlx.ops import window_attn_varlen

    q = torch.full((2, 1, 64), 2.0**60, device="cuda", dtype=torch.bfloat16)
    k = torch.stack([torch.zeros_like(q[0]), q[0]])
    v = torch.stack([torch.zeros_like(q[0]), torch.ones_like(q[0])])
    out = window_attn_varlen(q, k, v, _cu([2]), (1, 1), 1e-50)
    torch.testing.assert_close(out, torch.full_like(q, 0.5), atol=0, rtol=0)


def test_window_attn_varlen_large_scale_keeps_the_maximum():
    """A finite scale whose scaled scores overflow float32 still gives all weight to the largest scores."""
    from triton.tlx.ops import window_attn_varlen

    ones = torch.ones(1, 1, 128, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(window_attn_varlen(ones, ones, ones, _cu([1]), (0, 0), 1e37), ones, atol=0, rtol=0)
    q = torch.ones(2, 1, 128, device="cuda", dtype=torch.float16)
    k = torch.stack([q[0], 2 * q[0]])
    v = torch.stack([torch.zeros_like(q[0]), q[0]])
    out = window_attn_varlen(q, k, v, _cu([2]), (1, 1), 1e37)
    torch.testing.assert_close(out, torch.ones_like(q), atol=0, rtol=0)


def test_window_attn_varlen_strided_qkv():
    """Q/K/V as head slices of one packed `(tokens, 3 * heads, 128)` projection, and strided offsets."""
    from triton.tlx.ops import window_attn_varlen

    torch.manual_seed(20)
    lengths = [3000, 129, 1]
    cu = _cu(lengths)
    qkv = torch.randn(sum(lengths), 6, 128, device="cuda", dtype=torch.float16)
    q, k, v = qkv[:, 0:2], qkv[:, 2:4], qkv[:, 4:6]
    out = window_attn_varlen(q, k, v, torch.stack([cu, cu.flip(0)], 1)[:, 0], (64, 64))
    torch.testing.assert_close(out.float(), _reference(q, k, v, cu, (64, 64), 128**-0.5), **TOLERANCE[torch.float16])


def test_window_attn_varlen_contiguous_but_unaligned_base():
    """A contiguous view whose storage offset leaves the data pointer off a 16-byte boundary."""
    from triton.tlx.ops import window_attn_varlen

    torch.manual_seed(20)
    lengths = [60, 40]
    cu = _cu(lengths)

    def unaligned(tokens):
        flat = torch.randn(tokens * 128 + 1, device="cuda", dtype=torch.float16)
        x = flat[1:].view(tokens, 1, 128)
        assert x.is_contiguous() and x.data_ptr() % 16 != 0
        return x

    q, k, v = (unaligned(sum(lengths)) for _ in range(3))
    out = window_attn_varlen(q, k, v, cu, (64, 64))
    torch.testing.assert_close(out.float(), _reference(q, k, v, cu, (64, 64), 128**-0.5), **TOLERANCE[torch.float16])


def test_window_attn_varlen_no_tokens():
    from triton.tlx.ops import window_attn_varlen

    q = torch.empty(0, 1, 128, device="cuda", dtype=torch.float16)
    assert window_attn_varlen(q, q, q, _cu([0, 0]), (64, 64)).shape == q.shape


def test_window_attn_varlen_repeat_and_graph():
    """Launches are deterministic, eager or replayed from a CUDA graph."""
    from triton.tlx.ops import window_attn_varlen

    torch.manual_seed(20)
    lengths = [5000, 3000, 1, 2000]
    cu = _cu(lengths)
    sets = [_inputs(lengths, 1, 1, 128, torch.float16)[:3] for _ in range(3)]
    eager = [window_attn_varlen(*s, cu, (64, 64)) for s in sets]
    for _ in range(3):
        torch.testing.assert_close(window_attn_varlen(*sets[0], cu, (64, 64)), eager[0], atol=0, rtol=0)
    static = [x.clone() for x in sets[0]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = window_attn_varlen(*static, cu, (64, 64))
    for i in (1, 2, 0, 1):
        for dst, src in zip(static, sets[i]):
            dst.copy_(src)
        graph.replay()
        torch.testing.assert_close(out, eager[i], atol=0, rtol=0)


def test_window_attn_varlen_rejects_invalid_inputs():
    from triton.tlx.ops import InvalidInput, UnsupportedBackward, window_attn_varlen

    q = torch.randn(100, 1, 128, device="cuda", dtype=torch.float16)
    cu = _cu([100])
    with pytest.raises(InvalidInput, match="int32"):
        window_attn_varlen(q, q, q, cu.long(), (64, 64))
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        window_attn_varlen(q[..., :96], q[..., :96], q[..., :96], cu, (64, 64))
    with pytest.raises(InvalidInput, match="dtype"):
        window_attn_varlen(q, q.bfloat16(), q, cu, (64, 64))
    with pytest.raises(InvalidInput, match="rank-3"):
        window_attn_varlen(q[None], q[None], q[None], cu, (64, 64))
    with pytest.raises(InvalidInput, match="head count"):
        window_attn_varlen(q.expand(100, 3, 128), q.expand(100, 2, 128), q.expand(100, 2, 128), cu, (64, 64))
    with pytest.raises(InvalidInput, match="tokens"):
        window_attn_varlen(q, q[:50], q[:50], cu, (64, 64))
    with pytest.raises(InvalidInput, match="does not support float32"):
        window_attn_varlen(q.float(), q.float(), q.float(), cu, (64, 64))
    for window_size in [(-1, 64), (64, -1), (200, 57), (64, ), 64, (64.0, 64)]:
        with pytest.raises(InvalidInput, match="window_size"):
            window_attn_varlen(q, q, q, cu, window_size)
    for sm_scale in [0.0, -0.1, float("nan"), float("inf")]:
        with pytest.raises(InvalidInput, match="positive finite sm_scale"):
            window_attn_varlen(q, q, q, cu, (64, 64), sm_scale)
    with pytest.raises(InvalidInput, match="overflows float32"):
        window_attn_varlen(q, q, q, cu, (64, 64), 1e40)
    for position in range(3):
        operands = [q, q, q]
        operands[position] = q.clone().requires_grad_()
        with pytest.raises(UnsupportedBackward):
            window_attn_varlen(*operands, cu, (64, 64))


@functools.cache
def _traceable_op():
    from triton.tlx.ops.kernels.window_attn_varlen.sm90 import window_attn_varlen

    @torch.library.triton_op("tlx_ops_test::window_attn_varlen_sm90", mutates_args=())
    def window_attn_varlen_sm90(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
        return window_attn_varlen(q, k, v, cu, (64, 64), None)

    return torch.ops.tlx_ops_test.window_attn_varlen_sm90


def test_window_attn_varlen_traces_for_export_and_compile():
    """The launch has no host-only step, so export keeps it as one op and Inductor compiles it."""
    torch.manual_seed(20)
    op = _traceable_op()
    lengths = [3000, 129, 1, 2000]
    q, k, v, cu = _inputs(lengths, 1, 1, 128, torch.float16)
    eager = op(q, k, v, cu)
    torch.testing.assert_close(eager.float(), _reference(q, k, v, cu, (64, 64), 128**-0.5), **TOLERANCE[torch.float16])

    class Wrapper(torch.nn.Module):

        def forward(self, q, k, v, cu):
            return op(q, k, v, cu)

    graph = torch.export.export(Wrapper(), (q, k, v, cu)).graph
    assert any("window_attn_varlen_sm90" in str(node.target) for node in graph.nodes if node.op == "call_function")
    torch.testing.assert_close(torch.compile(op, fullgraph=True)(q, k, v, cu), eager)


def test_window_attn_varlen_compiles_with_dynamic_shapes():
    """Reuse one compiled graph across token counts and batch sizes."""
    from torch._dynamo.testing import CompileCounterWithBackend

    torch.manual_seed(20)
    op = _traceable_op()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(op, backend=counter, fullgraph=True, dynamic=True)
    for lengths in ([200, 300], [65, 129, 257], [5000, 1, 3]):
        q, k, v, cu = _inputs(lengths, 1, 1, 128, torch.float16)
        for tensor in (q, k, v):
            torch._dynamo.mark_static(tensor, 1)
            torch._dynamo.mark_static(tensor, 2)
        torch.testing.assert_close(compiled(q, k, v, cu), op(q, k, v, cu))
    assert counter.frame_count == 1


def test_window_attn_varlen_aoti_package_matches_eager(tmp_path):
    """AOTInductor with the autotune settings production lowering uses; the kernel takes no host descriptor."""
    torch.manual_seed(20)
    op = _traceable_op()
    lengths = [3000, 129, 1, 2000]
    q, k, v, cu = _inputs(lengths, 1, 1, 128, torch.float16)

    class Wrapper(torch.nn.Module):

        def forward(self, q, k, v, cu):
            return op(q, k, v, cu)

    eager = Wrapper()(q, k, v, cu)
    tokens = torch.export.Dim("tokens", min=2, max=1 << 20)
    exported = torch.export.export(Wrapper(), (q, k, v, cu),
                                   dynamic_shapes=({0: tokens}, {0: tokens}, {0: tokens}, {0: torch.export.Dim.AUTO}))
    with torch._inductor.config.patch({
            "triton.autotune_at_compile_time": True,
            "triton.autotune_with_sample_inputs": True,
    }):
        package = torch._inductor.aoti_compile_and_package(exported, package_path=str(tmp_path / "window.pt2"))
    compiled = torch._inductor.aoti_load_package(package)
    torch.testing.assert_close(compiled(q, k, v, cu), eager)
    lengths = [700, 9000, 64]
    q, k, v, cu = _inputs(lengths, 1, 1, 128, torch.float16)
    torch.testing.assert_close(compiled(q, k, v, cu), Wrapper()(q, k, v, cu))
