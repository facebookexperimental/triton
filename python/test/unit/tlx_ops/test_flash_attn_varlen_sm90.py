"""L1 correctness for ``tlx.ops.flash_attn_varlen`` on Hopper. Forward only."""

import contextvars
import functools
import itertools

import pytest
import torch
import triton
from triton._internal_testing import is_hopper
from triton.tlx.ops.kernels.flash_attn_varlen._shapes import CORRECTNESS_SHAPES, is_cross_attention, lengths, seqlens

SELF_SHAPES = [shape[:7] for shape in CORRECTNESS_SHAPES if not is_cross_attention(shape)]
RAGGED_SHAPES = [shape for shape in CORRECTNESS_SHAPES if is_cross_attention(shape)]

pytestmark = pytest.mark.skipif(not is_hopper(), reason="tlx.ops.flash_attn_varlen on sm90 requires Hopper")

TOLERANCE = {torch.float16: dict(atol=3e-3, rtol=1e-2), torch.bfloat16: dict(atol=2e-2, rtol=2e-2)}


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _reference(q, k, v, cu_q, cu_k, causal, scale):
    """FP32 attention per sequence, bottom-right causal; a row with no key is 0."""
    out = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    group = q.shape[1] // k.shape[1]
    for b in range(cu_q.numel() - 1):
        q0, q1, k0, k1 = cu_q[b].item(), cu_q[b + 1].item(), cu_k[b].item(), cu_k[b + 1].item()
        if q1 == q0:
            continue
        qs = q[q0:q1].float()
        ks = k[k0:k1].float().repeat_interleave(group, 1)
        vs = v[k0:k1].float().repeat_interleave(group, 1)
        s = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if causal:
            rows = torch.arange(q1 - q0, device=q.device)[:, None]
            cols = torch.arange(k1 - k0, device=q.device)[None, :]
            s = s.masked_fill(cols > rows + (k1 - k0) - (q1 - q0), float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)
        out[q0:q1] = torch.einsum("hqk,khd->qhd", p, vs)
    return out


def _check(lens_q, lens_k, causal, heads=2, kv_heads=None, head_dim=128, dtype=torch.bfloat16, sm_scale=None,
           max_seqlen_q=None):
    from triton.tlx.ops import flash_attn_varlen

    kv_heads = kv_heads or heads
    q = torch.randn(sum(lens_q), heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(sum(lens_k), kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(sum(lens_k), kv_heads, head_dim, device="cuda", dtype=dtype)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    out = flash_attn_varlen(q, k, v, cu_q, cu_k, max_seqlen_q or max(lens_q), max(lens_k), sm_scale=sm_scale,
                            causal=causal)
    ref = _reference(q, k, v, cu_q, cu_k, causal, head_dim**-0.5 if sm_scale is None else sm_scale)
    assert out.dtype == dtype and out.shape == q.shape
    torch.testing.assert_close(out.float(), ref, **TOLERANCE[dtype])


@pytest.mark.parametrize("batch,min_seqlen,max_seqlen,heads,head_dim,causal,dtype_name", SELF_SHAPES)
def test_flash_attn_varlen_shapes(batch, min_seqlen, max_seqlen, heads, head_dim, causal, dtype_name):
    torch.manual_seed(20)
    lens = seqlens(batch, min_seqlen, max_seqlen)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    _check(lens, lens, causal, heads=heads, head_dim=head_dim, dtype=dtype)


@pytest.mark.parametrize("shape", RAGGED_SHAPES)
def test_flash_attn_varlen_ragged_shapes(shape):
    """Query and key/value lengths that differ, as in the ranking models' attention over a user history."""
    torch.manual_seed(20)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[shape.dtype]
    lens_q, lens_k = lengths(shape)
    _check(lens_q, lens_k, shape.causal, heads=shape.heads, head_dim=shape.head_dim, dtype=dtype)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "lens_q,lens_k",
    [
        ([1], [1]),
        # Partial, exact and one-past tiles of the 128-row / 128-key blocks.
        ([5, 64, 65, 127, 128, 129, 255, 256, 257, 300], [5, 64, 65, 127, 128, 129, 255, 256, 257, 300]),
        # Empty sequences are skipped without disturbing their neighbours.
        ([0, 100, 0, 257], [0, 100, 0, 257]),
        # More keys than queries: causal rows see the whole key prefix.
        ([100, 37], [250, 600]),
        # Fewer keys than queries: causal rows before the offset see no key.
        ([300, 600], [120, 50]),
        ([2000, 1500], [2000, 1500]),
        # A key block that straddles a sequence end must not weigh the next sequence's rows.
        ([700, 700, 700], [129, 1, 257]),
    ],
)
def test_flash_attn_varlen_lengths(lens_q, lens_k, causal):
    torch.manual_seed(20)
    _check(lens_q, lens_k, causal)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lens_k", [[0, 400, 0], [400, 0, 400], [0, 0, 400], [400, 0, 0]])
def test_flash_attn_varlen_sequences_without_keys(lens_k, causal):
    """A query row with no key writes 0, and its neighbours are untouched."""
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    # The query lengths are not tile multiples, so a neighbour's last partial tile ends inside or next to a
    # key-less sequence's rows.
    lens_q = [200, 300, 150]
    q = torch.randn(sum(lens_q), 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens_k), 2, 128, device="cuda", dtype=torch.bfloat16)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    out = flash_attn_varlen(q, k, k, cu_q, cu_k, max(lens_q), max(lens_k), causal=causal)
    starts = cu_q.tolist()
    for b, keys in enumerate(lens_k):
        rows = out[starts[b]:starts[b + 1]]
        assert (torch.count_nonzero(rows) == 0) == (keys == 0), f"sequence {b} with {keys} keys"
    ref = _reference(q, k, k, cu_q, cu_k, causal, 128**-0.5)
    torch.testing.assert_close(out.float(), ref, **TOLERANCE[torch.bfloat16])


def test_flash_attn_varlen_no_keys_at_all():
    from triton.tlx.ops import flash_attn_varlen

    q = torch.randn(100, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(0, 2, 128, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_varlen(q, k, k, _cu([60, 40]), _cu([0, 0]), 60, 0)
    assert torch.count_nonzero(out) == 0


def test_flash_attn_varlen_oversized_max_seqlen():
    """`max_seqlen_q` only sizes the grid, so a loose bound changes nothing."""
    torch.manual_seed(20)
    _check([100, 300], [200, 500], False, max_seqlen_q=1000)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (4, 2), (8, 1)])
def test_flash_attn_varlen_heads(heads, kv_heads, dtype):
    torch.manual_seed(20)
    _check([300, 700, 64], [300, 700, 64], True, heads=heads, kv_heads=kv_heads, dtype=dtype, sm_scale=0.05)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_varlen_head_dim_64(causal, dtype):
    torch.manual_seed(20)
    _check([300, 200, 129], [300, 640, 129], causal, head_dim=64, dtype=dtype)


@pytest.mark.parametrize("sm_scale", [0.05, 11.313708498984761, 25.298221281347036])
def test_flash_attn_varlen_custom_scale(sm_scale):
    """Scales above 1 sharpen the softmax; they are used per layer by ranking models."""
    torch.manual_seed(20)
    _check([700, 300], [1500, 900], False, sm_scale=sm_scale)


@pytest.mark.parametrize("sm_scale", [0.0, -0.05, -11.313708498984761, float("nan"), float("inf")])
def test_flash_attn_varlen_rejects_non_positive_scale(sm_scale):
    """Masking and the running max assume a positive finite scale, as in FlashAttention."""
    from triton.tlx.ops import InvalidInput, flash_attn_varlen

    q = torch.randn(100, 1, 128, device="cuda", dtype=torch.float16)
    cu = _cu([100])
    with pytest.raises(InvalidInput, match="positive finite sm_scale"):
        flash_attn_varlen(q, q, q, cu, cu, 100, 100, sm_scale=sm_scale)


def test_flash_attn_varlen_ranking_attention_shape():
    """24 query rows per candidate over a long user history."""
    torch.manual_seed(20)
    _check([2160] * 3, [3072, 1500, 2900], False, sm_scale=16.0)


def _split_inputs(lengths, n=1):
    q, k, v = (torch.randn(n, sum(lengths), 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    return q, k, v, _cu(lengths)


def test_flash_attn_varlen_repeat_and_graph():
    """Launches are deterministic, eager or replayed from a CUDA graph."""
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    lengths = [1000] * 6
    q, k, v, cu = _split_inputs(lengths, 3)
    eager = [flash_attn_varlen(q[i], k[i], v[i], cu, cu, 1000, 1000, causal=True) for i in range(3)]
    for _ in range(3):
        torch.testing.assert_close(flash_attn_varlen(q[0], k[0], v[0], cu, cu, 1000, 1000, causal=True), eager[0],
                                   atol=0, rtol=0)
    sq, sk, sv = q[0].clone(), k[0].clone(), v[0].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flash_attn_varlen(sq, sk, sv, cu, cu, 1000, 1000, causal=True)
    for i in (1, 2, 0, 1):
        sq.copy_(q[i])
        sk.copy_(k[i])
        sv.copy_(v[i])
        graph.replay()
        torch.testing.assert_close(out, eager[i], atol=0, rtol=0)


def test_flash_attn_varlen_streams():
    """Launches on two streams at once do not interfere."""
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    lengths = [1000] * 4
    q, k, v, cu = _split_inputs(lengths, 2)
    expected = [flash_attn_varlen(q[i], k[i], v[i], cu, cu, 1000, 1000, causal=True) for i in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    outs = [[], []]
    for _ in range(20):
        for i, s in enumerate(streams):
            with torch.cuda.stream(s):
                outs[i].append(flash_attn_varlen(q[i], k[i], v[i], cu, cu, 1000, 1000, causal=True))
    torch.cuda.synchronize()
    for i in range(2):
        for out in outs[i]:
            torch.testing.assert_close(out, expected[i], atol=0, rtol=0)


def test_flash_attn_varlen_strided_qkv():
    """Q/K/V as head slices of one packed `(tokens, 3 * heads, 128)` projection."""
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    lengths = [700, 1000, 1200]
    cu = _cu(lengths)
    qkv = torch.randn(sum(lengths), 6, 128, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv[:, 0:2], qkv[:, 2:4], qkv[:, 4:6]
    out = flash_attn_varlen(q, k, v, cu, cu, max(lengths), max(lengths), causal=True)
    ref = _reference(q, k, v, cu, cu, True, 128**-0.5)
    torch.testing.assert_close(out.float(), ref, **TOLERANCE[torch.bfloat16])


def test_flash_attn_varlen_contiguous_but_unaligned_base():
    """A contiguous view whose storage offset leaves the data pointer off a 16-byte boundary."""
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    lens = [60, 40]
    cu = _cu(lens)

    def unaligned(tokens):
        flat = torch.randn(tokens * 2 * 128 + 1, device="cuda", dtype=torch.float16)
        x = flat[1:].view(tokens, 2, 128)
        assert x.is_contiguous() and x.data_ptr() % 16 != 0
        return x

    q, k, v = (unaligned(sum(lens)) for _ in range(3))
    out = flash_attn_varlen(q, k, v, cu, cu, max(lens), max(lens))
    torch.testing.assert_close(out.float(), _reference(q, k, v, cu, cu, False, 128**-0.5), **TOLERANCE[torch.float16])


def test_flash_attn_varlen_k_and_v_may_be_one_tensor():
    """Ranking models attend with the same history as keys and values."""
    from triton.tlx.ops import flash_attn_varlen

    torch.manual_seed(20)
    lens_q, lens_k = [600, 400], [1000, 300]
    q = torch.randn(sum(lens_q), 2, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(sum(lens_k), 2, 128, device="cuda", dtype=torch.bfloat16)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)
    out = flash_attn_varlen(q, kv, kv, cu_q, cu_k, 600, 1000)
    torch.testing.assert_close(out.float(), _reference(q, kv, kv, cu_q, cu_k, False, 128**-0.5),
                               **TOLERANCE[torch.bfloat16])


def test_flash_attn_varlen_rejects_invalid_inputs():
    from triton.tlx.ops import InvalidInput, UnsupportedBackward, flash_attn_varlen

    q = torch.randn(100, 1, 128, device="cuda", dtype=torch.float16)
    cu = _cu([100])
    with pytest.raises(InvalidInput, match="int32"):
        flash_attn_varlen(q, q, q, cu.long(), cu.long(), 100, 100)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        flash_attn_varlen(q[..., :96], q[..., :96], q[..., :96], cu, cu, 100, 100)
    with pytest.raises(InvalidInput, match="dtype"):
        flash_attn_varlen(q, q.bfloat16(), q, cu, cu, 100, 100)
    with pytest.raises(InvalidInput, match="rank-3"):
        flash_attn_varlen(q[None], q[None], q[None], cu, cu, 100, 100)
    with pytest.raises(InvalidInput, match="head count"):
        flash_attn_varlen(q.expand(100, 3, 128), q.expand(100, 2, 128), q.expand(100, 2, 128), cu, cu, 100, 100)
    with pytest.raises(InvalidInput, match="does not support float32"):
        flash_attn_varlen(q.float(), q.float(), q.float(), cu, cu, 100, 100)
    for position in range(3):
        operands = [q, q, q]
        operands[position] = q.clone().requires_grad_()
        with pytest.raises(UnsupportedBackward):
            flash_attn_varlen(*operands, cu, cu, 100, 100)


@functools.cache
def _traceable_op():
    from triton.tlx.ops.kernels.flash_attn_varlen.sm90 import flash_attn_varlen

    @torch.library.triton_op("tlx_ops_test::varlen_sm90", mutates_args=())
    def varlen_sm90(q: torch.Tensor, kv: torch.Tensor, cu_q: torch.Tensor, cu_k: torch.Tensor, max_seqlen_q: int,
                    scale: float) -> torch.Tensor:
        return flash_attn_varlen(q, kv, kv, cu_q, cu_k, max_seqlen_q, scale, False)

    return torch.ops.tlx_ops_test.varlen_sm90


def test_flash_attn_varlen_traces_for_export_and_compile():
    """The launch has no host-only step, so export keeps it as one op and Inductor compiles it."""
    torch.manual_seed(20)
    op = _traceable_op()
    lens_q, lens_k = [300, 200, 128, 450], [900, 1500, 700, 1100]
    q = torch.randn(sum(lens_q), 2, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(sum(lens_k), 2, 128, device="cuda", dtype=torch.bfloat16)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)

    def attend(q, kv, cu_q, cu_k):
        return op(q, kv, cu_q, cu_k, max(lens_q), 16.0)

    eager = attend(q, kv, cu_q, cu_k)
    ref = _reference(q, kv, kv, cu_q, cu_k, False, 16.0)
    torch.testing.assert_close(eager.float(), ref, **TOLERANCE[torch.bfloat16])

    class Wrapper(torch.nn.Module):

        def forward(self, q, kv, cu_q, cu_k):
            return attend(q, kv, cu_q, cu_k)

    graph = torch.export.export(Wrapper(), (q, kv, cu_q, cu_k)).graph
    assert any("varlen_sm90" in str(node.target) for node in graph.nodes if node.op == "call_function")
    torch.testing.assert_close(torch.compile(attend, fullgraph=True)(q, kv, cu_q, cu_k), eager)


@pytest.mark.skip(reason="Re-enable when the H100 CI PyTorch wheel includes pytorch#201258 (D125369038).")
def test_flash_attn_varlen_aoti_package_matches_eager(tmp_path):
    """AOTInductor with the autotune settings production lowering uses; full tiles keep the in-kernel TMA store."""
    torch.manual_seed(20)
    op = _traceable_op()
    lens_q, lens_k = [300, 200, 128, 450], [900, 1500, 700, 1100]
    q = torch.randn(sum(lens_q), 2, 128, device="cuda", dtype=torch.float16)
    kv = torch.randn(sum(lens_k), 2, 128, device="cuda", dtype=torch.float16)
    cu_q, cu_k = _cu(lens_q), _cu(lens_k)

    class Wrapper(torch.nn.Module):

        def forward(self, q, kv, cu_q, cu_k):
            return op(q, kv, cu_q, cu_k, max(lens_q), 11.313708498984761)

    eager = Wrapper()(q, kv, cu_q, cu_k)
    exported = torch.export.export(Wrapper(), (q, kv, cu_q, cu_k))
    with torch._inductor.config.patch({
            "triton.autotune_at_compile_time": True,
            "triton.autotune_with_sample_inputs": True,
    }):
        package = torch._inductor.aoti_compile_and_package(exported, package_path=str(tmp_path / "varlen.pt2"))
    compiled = torch._inductor.aoti_load_package(package)
    torch.testing.assert_close(compiled(q, kv, cu_q, cu_k), eager)


def test_flash_attn_varlen_compiles_with_dynamic_shapes():
    """Reuse one compiled graph across token counts, batch sizes, and maximum query lengths."""
    from torch._dynamo.testing import CompileCounterWithBackend

    torch.manual_seed(20)
    op = _traceable_op()

    def attend(q, kv, cu_q, cu_k, max_seqlen_q):
        return op(q, kv, cu_q, cu_k, max_seqlen_q, 128**-0.5)

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(attend, backend=counter, fullgraph=True, dynamic=True)
    for lens_q, lens_k in [([200, 300], [400, 500]), ([65, 129, 257], [200, 400, 350])]:
        q = torch.randn(sum(lens_q), 2, 128, device="cuda", dtype=torch.float16)
        kv = torch.randn(sum(lens_k), 2, 128, device="cuda", dtype=torch.float16)
        for tensor in (q, kv):
            torch._dynamo.mark_static(tensor, 1)
            torch._dynamo.mark_static(tensor, 2)
        cu_q, cu_k = _cu(lens_q), _cu(lens_k)
        eager = attend(q, kv, cu_q, cu_k, max(lens_q))
        ref = _reference(q, kv, kv, cu_q, cu_k, False, 128**-0.5)
        torch.testing.assert_close(eager.float(), ref, **TOLERANCE[torch.float16])
        torch.testing.assert_close(compiled(q, kv, cu_q, cu_k, max(lens_q)), eager)
    assert counter.frame_count == 1


@pytest.mark.skip(reason="Re-enable when the H100 CI PyTorch wheel includes pytorch#202409 (D127915035).")
def test_flash_attn_varlen_compiles_with_unaligned_base():
    """A compiled launch whose example inputs already sit off a 16-byte boundary; Inductor does not realign them."""
    torch.manual_seed(20)
    op = _traceable_op()
    lens = [60, 40]
    cu = _cu(lens)

    def unaligned(tokens):
        flat = torch.randn(tokens * 2 * 128 + 1, device="cuda", dtype=torch.float16)
        x = flat[1:].view(tokens, 2, 128)
        assert x.is_contiguous() and x.data_ptr() % 16 != 0
        return x

    q, kv = unaligned(sum(lens)), unaligned(sum(lens))

    def attend(q, kv, cu_q, cu_k):
        return op(q, kv, cu_q, cu_k, max(lens), 128**-0.5)

    eager = attend(q, kv, cu, cu)
    torch.testing.assert_close(eager.float(), _reference(q, kv, kv, cu, cu, False, 128**-0.5),
                               **TOLERANCE[torch.float16])
    torch.testing.assert_close(torch.compile(attend, fullgraph=True)(q, kv, cu, cu), eager)


def test_flash_attn_varlen_needs_no_triton_allocator():
    """The in-kernel output descriptor lives in a buffer the launch allocates, so no Triton allocator is used or set."""
    from triton.runtime import _allocation
    from triton.tlx.ops import flash_attn_varlen

    lens = [200, 300]
    cu = _cu(lens)
    q = torch.randn(sum(lens), 2, 128, device="cuda", dtype=torch.float16)

    def without_allocator():
        out = flash_attn_varlen(q, q, q, cu, cu, max(lens), max(lens))
        return out, _allocation._allocator.get()

    out, after = contextvars.Context().run(without_allocator)
    assert isinstance(after, _allocation.NullAllocator)
    torch.testing.assert_close(out.float(), _reference(q, q, q, cu, cu, False, 128**-0.5), **TOLERANCE[torch.float16])

    calls = []

    def counting(size, align, stream):
        calls.append(size)
        return torch.empty(size, dtype=torch.int8, device="cuda")

    def with_allocator():
        triton.set_allocator(counting)
        flash_attn_varlen(q, q, q, cu, cu, max(lens), max(lens))

    contextvars.Context().run(with_allocator)
    assert not calls


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_flash_attn_varlen_backend_on_non_current_device():
    """The backend selects its input device for both scratch allocation and kernel launch."""
    from triton.tlx.ops.kernels.flash_attn_varlen.sm90 import flash_attn_varlen

    current = torch.cuda.current_device()
    device = torch.device("cuda", (current + 1) % torch.cuda.device_count())
    lens = [65, 129]
    with torch.cuda.device(device):
        q = torch.randn(sum(lens), 2, 128, device=device, dtype=torch.float16)
        cu = _cu(lens)
        ref = _reference(q, q, q, cu, cu, False, 128**-0.5)

    out = contextvars.Context().run(flash_attn_varlen, q, q, q, cu, cu, max(lens), 128**-0.5, False)
    assert out.device == device and torch.cuda.current_device() == current
    torch.testing.assert_close(out.float(), ref, **TOLERANCE[torch.float16])
