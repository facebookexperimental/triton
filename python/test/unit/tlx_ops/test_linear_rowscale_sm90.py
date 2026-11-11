"""L1 correctness for ``tlx.ops.linear_rowscale`` on Hopper. Forward only."""

import pytest
import torch
from triton._internal_testing import is_hopper

pytestmark = pytest.mark.skipif(not is_hopper(), reason="tlx.ops.linear_rowscale on sm90 requires Hopper")


def _inputs(M, N, K, dtype=torch.float16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", generator=g).to(dtype)
    bound = K**-0.5
    weight = ((torch.rand(N, K, device="cuda", generator=g) * 2 - 1) * bound).to(dtype)
    bias = ((torch.rand(N, device="cuda", generator=g) * 2 - 1) * bound).to(dtype)
    # A wide range of row scales, as the reciprocal standard deviations of a deferred normalization are.
    row_scale = torch.exp(torch.randn(M, device="cuda", generator=g) * 2)
    return x, weight, bias, row_scale


def _reference(x, weight, bias, row_scale):
    y = x.float() @ weight.float().t()
    if bias is not None:
        y = y + bias.float()
    if row_scale is not None:
        y = y * row_scale[:, None]
    return y.to(x.dtype)


def _assert_matches(out, ref):
    """FP32 accumulation and one rounding: the products differ from the reference only by summation order.

    The absolute tolerance is per row: the row scales span orders of magnitude, and a tolerance scaled to the largest
    output would accept any error in the rows with small scales.
    """
    assert out.shape == ref.shape and out.dtype == ref.dtype
    if ref.numel() == 0:
        return
    out, ref = out.float(), ref.float()
    atol = 1e-3 * ref.abs().amax(dim=1, keepdim=True) + 1e-6
    bad = (out - ref).abs() > atol + 8e-3 * ref.abs()
    assert not bad.any(), f"{int(bad.sum())} of {bad.numel()} outputs, in {int(bad.any(1).sum())} rows, differ"


def _shifted(t):
    """A copy of ``t`` whose base sits 8 bytes past a 16-byte boundary."""
    pad = 8 // t.element_size()
    storage = torch.empty(t.numel() + pad, device=t.device, dtype=t.dtype)
    shifted = storage[pad:].view(t.shape)
    shifted.copy_(t)
    return shifted


# The deployed product (N = 58880, K = 368) over its row counts plus edges, and shapes that exercise the K tail, one
# column tile and N narrower than a tile.
@pytest.mark.parametrize("M", [0, 1, 2, 63, 127, 128, 129, 600, 671, 704, 768, 1000, 4097])
def test_linear_rowscale_deployed_shape(M):
    from triton.tlx.ops import linear_rowscale
    x, weight, bias, row_scale = _inputs(M, 58880, 368, seed=M)
    _assert_matches(linear_rowscale(x, weight, bias, row_scale), _reference(x, weight, bias, row_scale))


# The other two deployed products: N = 35328 with the same K, and N = 30720 with K = 128 (two x blocks, ten weight
# stages).
@pytest.mark.parametrize("N, K", [(35328, 368), (30720, 128)])
@pytest.mark.parametrize("M", [1, 129, 600, 768])
def test_linear_rowscale_other_deployed_shapes(M, N, K):
    from triton.tlx.ops import linear_rowscale
    x, weight, bias, row_scale = _inputs(M, N, K, seed=M + N)
    _assert_matches(linear_rowscale(x, weight, bias, row_scale), _reference(x, weight, bias, row_scale))


@pytest.mark.parametrize("N, K", [(1024, 512), (136, 64), (8, 8), (256, 200), (4096, 128)])
@pytest.mark.parametrize("M", [1, 129, 1000])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_rowscale_shapes(M, N, K, dtype):
    from triton.tlx.ops import linear_rowscale
    x, weight, bias, row_scale = _inputs(M, N, K, dtype, seed=N + K)
    _assert_matches(linear_rowscale(x, weight, bias, row_scale), _reference(x, weight, bias, row_scale))


@pytest.mark.parametrize("with_bias, with_scale", [(False, False), (True, False), (False, True)])
def test_linear_rowscale_optional_terms(with_bias, with_scale):
    from triton.tlx.ops import linear_rowscale
    x, weight, bias, row_scale = _inputs(700, 58880, 368, seed=3)
    bias = bias if with_bias else None
    row_scale = row_scale if with_scale else None
    _assert_matches(linear_rowscale(x, weight, bias, row_scale), _reference(x, weight, bias, row_scale))


def test_linear_rowscale_many_row_blocks():
    """More 128-row blocks than SMs: each CTA walks several row blocks, and the consumers' tile parity carries over."""
    from triton.tlx.ops import linear_rowscale
    # N = 264 is three column tiles, so the second row block starts on the other consumer.
    x, weight, bias, row_scale = _inputs(17000, 264, 64, seed=4)
    _assert_matches(linear_rowscale(x, weight, bias, row_scale), _reference(x, weight, bias, row_scale))


def test_linear_rowscale_rows_do_not_depend_on_the_batch():
    """A row's result is bitwise the same whether it is computed in the full batch or in a slice of it."""
    from triton.tlx.ops import linear_rowscale
    x, weight, bias, row_scale = _inputs(700, 58880, 368, seed=8)
    full = linear_rowscale(x, weight, bias, row_scale)
    for lo, hi in ((0, 1), (5, 6), (129, 300), (699, 700)):
        part = linear_rowscale(x[lo:hi], weight, bias, row_scale[lo:hi])
        torch.testing.assert_close(part, full[lo:hi], atol=0, rtol=0)


def test_linear_rowscale_strided_inputs():
    """Row-strided x and strided vectors are read as they are; a column-major weight is copied."""
    from triton.tlx.ops import linear_rowscale
    x, weight, bias, row_scale = _inputs(300, 1024, 368, seed=5)
    x_rows = torch.zeros(300, 400, device="cuda", dtype=x.dtype)
    x_rows[:, :368] = x
    weight_t = weight.t().contiguous().t()
    row_scale_strided = torch.stack([row_scale, row_scale], 1)[:, 0]
    bias_strided = torch.stack([bias, bias], 1)[:, 0]
    out = linear_rowscale(x_rows[:, :368], weight_t, bias_strided, row_scale_strided)
    assert out.is_contiguous()
    _assert_matches(out, _reference(x, weight, bias, row_scale))


@pytest.mark.parametrize("operand", ["x_base", "x_row_stride", "weight_base"])
def test_linear_rowscale_copies_operands_tma_cannot_read(operand):
    """A base or row stride that is not 16-byte aligned cannot be read by TMA; the op copies the operand."""
    from triton.tlx.ops import linear_rowscale
    x, weight, bias, row_scale = _inputs(300, 1024, 368, seed=6)
    x_in, weight_in = x, weight
    if operand == "x_base":
        x_in = _shifted(x)
        assert x_in.data_ptr() % 16 != 0
    elif operand == "x_row_stride":
        x_in = torch.zeros(300, 372, device="cuda", dtype=x.dtype)[:, :368]
        x_in.copy_(x)
    else:
        weight_in = _shifted(weight)
        assert weight_in.data_ptr() % 16 != 0
    _assert_matches(linear_rowscale(x_in, weight_in, bias, row_scale), _reference(x, weight, bias, row_scale))


def test_linear_rowscale_repeated_calls_on_a_side_stream():
    from triton.tlx.ops import linear_rowscale
    x, weight, bias, row_scale = _inputs(671, 58880, 368, seed=6)
    ref = _reference(x, weight, bias, row_scale)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        outs = [linear_rowscale(x, weight, bias, row_scale) for _ in range(3)]
    torch.cuda.current_stream().wait_stream(stream)
    for out in outs:
        torch.testing.assert_close(out, outs[0], atol=0, rtol=0)
        _assert_matches(out, ref)


def test_linear_rowscale_rejects_invalid_inputs():
    from triton.tlx.ops import InvalidInput, UnsupportedBackward, linear_rowscale
    x, weight, bias, row_scale = _inputs(10, 256, 64)
    with pytest.raises(InvalidInput):
        linear_rowscale(x, weight[:, :32], bias, row_scale)
    with pytest.raises(InvalidInput):
        linear_rowscale(x.float(), weight.float(), bias.float(), row_scale)
    with pytest.raises(InvalidInput):
        linear_rowscale(x, weight, bias, row_scale.half())
    with pytest.raises(InvalidInput):
        linear_rowscale(x, weight, bias[:128], row_scale)
    with pytest.raises(InvalidInput):
        linear_rowscale(x, weight[:0])
    x_odd, weight_odd, _, _ = _inputs(10, 256, 60)
    with pytest.raises(InvalidInput):
        linear_rowscale(x_odd, weight_odd)
    x_long, weight_long, _, _ = _inputs(10, 256, 576)
    with pytest.raises(InvalidInput):
        linear_rowscale(x_long, weight_long)
    with pytest.raises(UnsupportedBackward):
        linear_rowscale(x, weight.requires_grad_())


def _launch(x, weight, bias, row_scale):
    from triton.tlx.ops import linear_rowscale_launch

    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    launch = linear_rowscale_launch(x, weight, bias, row_scale, out=out)
    torch.library.wrap_triton(launch.kernel)[launch.grid](*launch.args, **launch.kwargs)
    return out


@torch.library.triton_op("tlx_ops_test::linear_rowscale", mutates_args=())
def _linear_rowscale_op(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                        row_scale: torch.Tensor) -> torch.Tensor:
    return _launch(x, weight, bias, row_scale)


def test_linear_rowscale_launch_matches_op():
    from triton.tlx.ops import linear_rowscale
    x, weight, bias, row_scale = _inputs(671, 4096, 368, seed=9)
    torch.testing.assert_close(_launch(x, weight, bias, row_scale), linear_rowscale(x, weight, bias, row_scale), atol=0,
                               rtol=0)


def test_linear_rowscale_launch_writes_a_row_strided_out():
    """The TMA store takes out's row stride, so out need not be contiguous; the columns past N stay untouched."""
    from triton.tlx.ops import linear_rowscale, linear_rowscale_launch
    x, weight, bias, row_scale = _inputs(300, 1024, 368, seed=12)
    rows = torch.full((300, 1032), 7.0, device="cuda", dtype=x.dtype)
    out = rows[:, :1024]
    launch = linear_rowscale_launch(x, weight, bias, row_scale, out=out)
    launch.kernel[launch.grid](*launch.args, **launch.kwargs)
    torch.testing.assert_close(out, linear_rowscale(x, weight, bias, row_scale), atol=0, rtol=0)
    assert torch.all(rows[:, 1024:] == 7.0)


@pytest.mark.parametrize("operand",
                         ["x_base", "x_row_stride", "weight_column_major", "out_shape", "out_column_major", "out_base"])
def test_linear_rowscale_launch_rejects_operands_tma_cannot_read(operand):
    """The launch copies nothing, so an operand that TMA cannot read as it is raises instead."""
    from triton.tlx.ops import InvalidInput, linear_rowscale_launch
    x, weight, bias, row_scale = _inputs(300, 1024, 368, seed=10)
    out = torch.empty(300, 1024, device="cuda", dtype=x.dtype)
    if operand == "x_base":
        x = _shifted(x)
    elif operand == "x_row_stride":
        x = torch.zeros(300, 372, device="cuda", dtype=x.dtype)[:, :368]
    elif operand == "weight_column_major":
        weight = weight.t().contiguous().t()
    elif operand == "out_column_major":
        out = torch.empty(1024, 300, device="cuda", dtype=x.dtype).t()
    elif operand == "out_base":
        out = _shifted(out)
    else:
        out = out[:, :512]
    with pytest.raises(InvalidInput):
        linear_rowscale_launch(x, weight, bias, row_scale, out=out)


def test_linear_rowscale_launch_rejects_traced_misaligned_x():
    """Under tracing x has no address, but its storage offset shows the misalignment."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from triton.tlx.ops import InvalidInput, linear_rowscale_launch
    x, weight, bias, row_scale = _inputs(300, 1024, 368, seed=11)
    shifted = _shifted(x)
    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(t) for t in (shifted, weight, bias, row_scale)]
        out = torch.empty(300, 1024, device="cuda", dtype=x.dtype)
        with pytest.raises(InvalidInput):
            linear_rowscale_launch(*fake, out=out)


class _ScaledLinear(torch.nn.Module):
    """Calls the kernel through a custom op, as a served model would; the row scale is computed in the graph."""

    def __init__(self, weight, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x, var):
        return _linear_rowscale_op(x, self.weight, self.bias, torch.rsqrt(var.clamp_min(0.0) + 1e-5))


def test_linear_rowscale_compiles_with_dynamic_shapes():
    from torch._dynamo.testing import CompileCounterWithBackend

    _, weight, bias, _ = _inputs(1, 58880, 368)
    module = _ScaledLinear(weight, bias)
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(module, backend=counter, fullgraph=True, dynamic=True)
    for M in (600, 704, 768):
        x, _, _, row_scale = _inputs(M, 58880, 368, seed=M)
        var = row_scale.reciprocal().square() - 1e-5
        torch._dynamo.mark_static(x, 1)
        eager = module(x, var)
        _assert_matches(eager, _reference(x, weight, bias, torch.rsqrt(var.clamp_min(0.0) + 1e-5)))
        torch.testing.assert_close(compiled(x, var), eager)
    assert counter.frame_count == 1


def test_linear_rowscale_aoti_package_matches_eager(tmp_path):
    """AOTInductor with the autotune settings production lowering uses, and a dynamic row count."""
    _, weight, bias, _ = _inputs(1, 58880, 368)
    module = _ScaledLinear(weight, bias)
    x, _, _, row_scale = _inputs(700, 58880, 368, seed=7)
    var = row_scale.reciprocal().square()
    rows = torch.export.Dim("rows", min=1, max=4096)
    exported = torch.export.export(module, (x, var), dynamic_shapes=({0: rows}, {0: rows}))
    with torch._inductor.config.patch({
            "triton.autotune_at_compile_time": True,
            "triton.autotune_with_sample_inputs": True,
    }):
        package = torch._inductor.aoti_compile_and_package(exported, package_path=str(tmp_path / "linear.pt2"))
    loaded = torch._inductor.aoti_load_package(package)
    for M in (700, 1, 333):
        x, _, _, row_scale = _inputs(M, 58880, 368, seed=M)
        var = row_scale.reciprocal().square()
        torch.testing.assert_close(loaded(x, var), module(x, var))
