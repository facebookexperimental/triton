"""L1 correctness for ``tlx.ops.rmsnorm_cat`` on Hopper. Forward only."""

import functools

import pytest
import torch
from torch._inductor.utils import run_and_get_code
from torch.utils._python_dispatch import TorchDispatchMode
from triton._internal_testing import is_hopper
from triton.tlx.ops.kernels.norm._cat_shapes import CORRECTNESS_SHAPES, RmsNormCatShape, inputs, reference
from triton.tlx.ops.kernels.norm._cat_sm90 import tlx_layout

pytestmark = pytest.mark.skipif(not is_hopper(), reason="tlx.ops.rmsnorm_cat on sm90 requires Hopper")

# The kernel takes the final mean over the unrounded concatenation and the eager reference over the rounded one, so
# the two can differ by an output ulp or so.
TOLERANCE = {torch.float16: dict(atol=1e-2, rtol=1e-2), torch.bfloat16: dict(atol=5e-2, rtol=2e-2)}


def _check(parts, part_weights, weights, **kwargs):
    from triton.tlx.ops import rmsnorm_cat

    outs = rmsnorm_cat(parts, weights, part_weights=part_weights, **kwargs)
    refs = reference(parts, part_weights, weights, **kwargs)
    assert len(outs) == len(weights)
    for out, ref in zip(outs, refs):
        assert out.dtype == parts[0].dtype and out.shape == ref.shape and out.is_contiguous()
        torch.testing.assert_close(out, ref, **TOLERANCE[parts[0].dtype])
    return outs


@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=lambda s: f"{s.layout}-B{s.batch}-o{s.outputs}-{s.dtype}")
def test_rmsnorm_cat_shapes(shape):
    torch.manual_seed(0)
    _check(*inputs(shape))


@pytest.mark.parametrize("rows", [1, 3, 64, 100, 4352])
@pytest.mark.parametrize("transposed", [False, True])
def test_rmsnorm_cat_part_rows(rows, transposed):
    """Row counts that are and are not powers of two, read along either axis."""
    torch.manual_seed(1)
    x = torch.randn(9, 2, rows, device="cuda", dtype=torch.float16).permute(0, 2, 1) if transposed else torch.randn(
        9, rows, 2, device="cuda", dtype=torch.float16)
    head = torch.randn(9, 128, device="cuda", dtype=torch.float16)
    pw = torch.rand(2 * rows, device="cuda", dtype=torch.float16) + 0.5
    w = torch.rand(128 + 2 * rows, device="cuda", dtype=torch.float16) + 0.5
    _check([head, x], [None, pw], [w])


def test_rmsnorm_cat_2d_part_of_any_width():
    torch.manual_seed(7)
    a = torch.randn(10, 384, device="cuda", dtype=torch.float16)
    b = torch.randn(10, 1000, device="cuda", dtype=torch.float16)[:, 100:]
    pw = torch.rand(900, device="cuda", dtype=torch.float16) + 0.5
    w = torch.rand(1284, device="cuda", dtype=torch.float16) + 0.5
    _check([a, b], [None, pw], [w])


def test_rmsnorm_cat_no_part_norm_is_plain_rms_norm():
    from triton.tlx.ops import rmsnorm_cat

    torch.manual_seed(2)
    x = torch.randn(33, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(4096, device="cuda", dtype=torch.bfloat16)
    (out, ) = rmsnorm_cat([x], [w])
    ref = torch.nn.functional.rms_norm(x.float(), (4096, ), w.float(), eps=1e-5).bfloat16()
    torch.testing.assert_close(out, ref, **TOLERANCE[torch.bfloat16])


def test_rmsnorm_cat_fp32_weights_and_eps():
    """Weights may be fp32 parameters, as in a module that was not converted to half precision."""
    torch.manual_seed(3)
    parts, part_weights, weights = inputs(RmsNormCatShape(6, "mixed_small", 2, "fp16"))
    _check(parts, [w if w is None else w.float() for w in part_weights], [w.float() for w in weights], eps=1e-3)


def _strided(w):
    buf = torch.zeros(2 * w.numel(), device=w.device, dtype=w.dtype)
    buf[::2] = w
    return buf[::2]


@pytest.mark.parametrize("layout,batch", [("mixed_small", 5), ("ranking_over_arch", 300)])
def test_rmsnorm_cat_strided_weights(layout, batch):
    """1-D weights and part weights that are strided views, e.g. one column of a packed parameter."""
    torch.manual_seed(9)
    parts, part_weights, weights = inputs(RmsNormCatShape(batch, layout, 2, "fp16"))
    _check(parts, [w if w is None else _strided(w) for w in part_weights], [_strided(w) for w in weights])


def test_rmsnorm_cat_rejects_weights_that_need_grad():
    from triton.tlx.ops import rmsnorm_cat, UnsupportedBackward

    parts, part_weights, weights = inputs(RmsNormCatShape(5, "mixed_small", 1, "fp16"))
    w = weights[0].clone().requires_grad_(True)
    with pytest.raises(UnsupportedBackward):
        rmsnorm_cat(parts, [w], part_weights=part_weights)
    pw = part_weights[1].clone().requires_grad_(True)
    with pytest.raises(UnsupportedBackward):
        rmsnorm_cat(parts, weights, part_weights=[None, pw, part_weights[2]])
    with torch.no_grad():
        _check(parts, part_weights, [w])


class _PassThroughMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("batch", [16, 300])
def test_rmsnorm_cat_under_user_dispatch_mode(batch):
    """A dispatch mode that does not trace (a profiler, a flop counter) gets the eager launch."""
    from triton.tlx.ops import rmsnorm_cat

    torch.manual_seed(10)
    parts, part_weights, weights = inputs(RmsNormCatShape(batch, "ranking_over_arch", 2, "fp16"))
    eager = rmsnorm_cat(parts, weights, part_weights=part_weights)
    with _PassThroughMode():
        outs = rmsnorm_cat(parts, weights, part_weights=part_weights)
    for out, ref in zip(outs, eager):
        assert torch.equal(out, ref)


@pytest.mark.parametrize("batch", [16, 300])
def test_rmsnorm_cat_matches_inductor_rounding(batch):
    """Bit-for-bit what Inductor's fused kernel computes for the over-arch expression (fp32, unrounded final mean)."""
    from triton.tlx.ops import rmsnorm_cat

    torch.manual_seed(4)
    parts, part_weights, weights = inputs(RmsNormCatShape(batch, "ranking_over_arch", 1, "fp16"))
    (out, ) = rmsnorm_cat(parts, weights, part_weights=part_weights)

    def rms(x, w):
        return x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-5) * w.float()

    flat = [p.reshape(batch, -1).float() for p in parts]
    cat = torch.cat([flat[0], rms(flat[1], part_weights[1]), rms(flat[2], part_weights[2])], 1)
    ref = (cat.half().float() / torch.sqrt(cat.square().mean(-1, keepdim=True) + 1e-5) * weights[0].float()).half()
    mismatched = (out != ref).float().mean().item()
    # Reduction order still differs from torch's, which moves a handful of values by one ulp.
    assert mismatched < 1e-3, mismatched
    torch.testing.assert_close(out, ref, atol=4e-3, rtol=1e-3)


def _three_d(parts):
    return [parts[0].unsqueeze(1), *parts[1:]]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("batch", [1, 133, 264, 692])
def test_rmsnorm_cat_over_arch_kernel_choice(batch, outputs, dtype, monkeypatch):
    """The over-arch layout runs the TLX kernel at every batch size, with at most one CTA per SM."""
    from triton.tlx.ops import rmsnorm_cat
    from triton.tlx.ops.kernels.norm import _cat_sm90 as sm90

    torch.manual_seed(8)
    parts, part_weights, weights = inputs(RmsNormCatShape(batch, "ranking_over_arch", outputs, dtype))
    assert tlx_layout(_three_d(parts), part_weights, weights)
    launches = []
    launch_tlx = sm90._launch_tlx
    monkeypatch.setattr(sm90, "_launch_tlx", lambda *args: launches.append(args) or launch_tlx(*args))
    outs = _check(parts, part_weights, weights)
    assert [args[5] for args in launches] == [min(batch, torch.cuda.get_device_properties(0).multi_processor_count)]
    # A contiguous copy of dots is not the TLX layout, so this is the register kernel on the same values.
    copied = [parts[0], parts[1], parts[2].contiguous()]
    assert not tlx_layout(_three_d(copied), part_weights, weights)
    for out, reg in zip(outs, rmsnorm_cat(copied, weights, part_weights=part_weights)):
        # Same arithmetic in another fp32 summation order: a few outputs may move by an ulp.
        assert (out != reg).float().mean().item() < 1e-3
        torch.testing.assert_close(out, reg, **TOLERANCE[parts[0].dtype])


@pytest.mark.parametrize("layout", ["ranking_over_arch", "mixed_small"])
def test_rmsnorm_cat_rows_do_not_depend_on_batch(layout):
    """A row's outputs are the same bits in any batch, small or large."""
    from triton.tlx.ops import rmsnorm_cat

    torch.manual_seed(11)
    parts, part_weights, weights = inputs(RmsNormCatShape(692, layout, 2, "fp16"))
    full = rmsnorm_cat(parts, weights, part_weights=part_weights)
    for start, size in ((0, 1), (100, 133), (392, 300)):
        rows = [p[start:start + size] for p in parts]
        assert tlx_layout(_three_d(rows), part_weights, weights) == (layout == "ranking_over_arch")
        for out, ref in zip(rmsnorm_cat(rows, weights, part_weights=part_weights), full):
            assert torch.equal(out, ref[start:start + size])


def test_rmsnorm_cat_tlx_layout_rejects():
    """Other layouts and three outputs stay on the register kernel."""
    parts, part_weights, weights = inputs(RmsNormCatShape(64, "ranking_over_arch", 3, "fp16"))
    assert not tlx_layout(_three_d(parts), part_weights, weights)
    assert not tlx_layout(_three_d(parts), part_weights, [w.float() for w in weights[:2]])
    assert not tlx_layout(_three_d(parts), [None, part_weights[1], None], weights[:2])
    shifted = torch.randn(64 * 8704 + 8, device="cuda", dtype=torch.float16)[4:]
    misaligned = [parts[0], parts[1], shifted.as_strided(parts[2].shape, (8704, 1, 4352))]
    assert not tlx_layout(_three_d(misaligned), part_weights, weights[:2])
    parts, part_weights, weights = inputs(RmsNormCatShape(64, "mixed_small", 2, "fp16"))
    assert not tlx_layout(_three_d(parts), part_weights, weights)


def test_rmsnorm_cat_empty_batch():
    from triton.tlx.ops import rmsnorm_cat

    x = torch.empty(0, 2, 64, device="cuda", dtype=torch.float16)
    outs = rmsnorm_cat([x, x], [torch.ones(256, device="cuda", dtype=torch.float16)])
    assert outs[0].shape == (0, 256)


@pytest.mark.parametrize("batch", [64, 300])
def test_rmsnorm_cat_repeat_and_graph(batch):
    """Launches are deterministic, eager or replayed from a CUDA graph."""
    from triton.tlx.ops import rmsnorm_cat

    torch.manual_seed(5)
    shape = RmsNormCatShape(batch, "ranking_over_arch", 2, "fp16")
    sets = [inputs(shape) for _ in range(3)]
    _, part_weights, weights = sets[0]
    eager = [rmsnorm_cat(p, weights, part_weights=part_weights) for p, _, _ in sets]
    for _ in range(3):
        for out, ref in zip(rmsnorm_cat(sets[0][0], weights, part_weights=part_weights), eager[0]):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
    # Static inputs with the same strided layout, refilled before each replay.
    parts = inputs(shape)[0]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = rmsnorm_cat(parts, weights, part_weights=part_weights)
    for i in (1, 2, 0):
        for dst, src in zip(parts, sets[i][0]):
            dst.copy_(src)
        graph.replay()
        for out, ref in zip(outs, eager[i]):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_rmsnorm_cat_rejects_invalid_inputs():
    from triton.tlx.ops import InvalidInput, UnsupportedBackward, rmsnorm_cat

    x = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    w = torch.ones(64, device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="non-empty list"):
        rmsnorm_cat([], [w])
    with pytest.raises(InvalidInput, match="one to three"):
        rmsnorm_cat([x], [w] * 4)
    with pytest.raises(InvalidInput, match="one part weight"):
        rmsnorm_cat([x], [w], part_weights=[None, None])
    with pytest.raises(InvalidInput, match="batch size, dtype"):
        rmsnorm_cat([x, x[:3]], [torch.ones(128, device="cuda", dtype=torch.float16)])
    with pytest.raises(InvalidInput, match="batch size, dtype"):
        rmsnorm_cat([x, x.bfloat16()], [torch.ones(128, device="cuda", dtype=torch.float16)])
    with pytest.raises(InvalidInput, match="at least one element"):
        rmsnorm_cat([x, x[:, :0]], [w])
    with pytest.raises(InvalidInput, match="hold 64 elements"):
        rmsnorm_cat([x], [w[:32]])
    with pytest.raises(InvalidInput, match="row size"):
        rmsnorm_cat([x], [w], part_weights=[w[:32]])
    with pytest.raises(InvalidInput, match="power of two"):
        rmsnorm_cat([x.view(4, 2, 32)[..., :24]], [w[:48]])
    with pytest.raises(InvalidInput, match="fit in registers"):
        big = torch.randn(2, 40000, device="cuda", dtype=torch.float16)
        rmsnorm_cat([big.view(2, 625, 64)], [torch.ones(40000, device="cuda", dtype=torch.float16)])
    with pytest.raises(InvalidInput, match="segments"):
        odd = torch.randn(2, 127, 2, device="cuda", dtype=torch.float16)
        rmsnorm_cat([odd], [torch.ones(254, device="cuda", dtype=torch.float16)])
    with pytest.raises(InvalidInput, match="does not support float32"):
        rmsnorm_cat([x.float()], [w.float()])
    with pytest.raises(InvalidInput, match="eps"):
        rmsnorm_cat([x], [w], eps=0.0)
    with pytest.raises(UnsupportedBackward):
        rmsnorm_cat([x.clone().requires_grad_()], [w])


@functools.cache
def _traceable_op():
    from triton.tlx.ops.kernels.norm._cat_sm90 import rmsnorm_cat

    @torch.library.triton_op("tlx_ops_test::rmsnorm_cat_sm90", mutates_args=())
    def rmsnorm_cat_sm90(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, wb: torch.Tensor, wc: torch.Tensor,
                         w0: torch.Tensor, w1: torch.Tensor) -> list[torch.Tensor]:
        return rmsnorm_cat([a.unsqueeze(1), b, c], [w0, w1], [None, wb, wc], 1e-5)

    return torch.ops.tlx_ops_test.rmsnorm_cat_sm90


@pytest.mark.parametrize("batch", [40, 300])
def test_rmsnorm_cat_traces_for_export_and_compile(batch):
    """The launch has no host-only step, so export keeps it as one op and Inductor compiles it."""
    torch.manual_seed(6)
    op = _traceable_op()
    (a, b, c), (_, wb, wc), (w0, w1) = inputs(RmsNormCatShape(batch, "ranking_over_arch", 2, "fp16"))

    def norm(a, b, c):
        return op(a, b, c, wb, wc, w0, w1)

    eager = norm(a, b, c)
    for out, ref in zip(eager, reference([a, b, c], [None, wb, wc], [w0, w1])):
        torch.testing.assert_close(out, ref, **TOLERANCE[torch.float16])

    class Wrapper(torch.nn.Module):

        def forward(self, a, b, c):
            return norm(a, b, c)

    graph = torch.export.export(Wrapper(), (a, b, c)).graph
    assert any("rmsnorm_cat_sm90" in str(node.target) for node in graph.nodes if node.op == "call_function")
    torch._dynamo.reset()
    outs, code = run_and_get_code(torch.compile(norm, fullgraph=True, dynamic=False), a, b, c)
    assert "_rmsnorm_cat_tlx_fwd" in code[0] and "_rmsnorm_cat_fwd" not in code[0]
    for out, ref in zip(outs, eager):
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_rmsnorm_cat_dynamic_batch_compiles_tlx_kernel():
    """With a dynamic batch the graph holds the TLX launch alone, with a grid of at most one CTA per SM.

    The views are taken inside the graph, as in the model, so their strides and offsets are multiples of the batch
    and the TLX kernel's alignment needs are known without guards.
    """
    op = _traceable_op()

    def norm(float_proj, gemm, dots, wb, wc, w0, w1):
        return op(float_proj[:, :8192], gemm[64:].permute(1, 0, 2), dots.permute(0, 2, 1), wb, wc, w0, w1)

    torch._dynamo.reset()
    compiled = torch.compile(norm, fullgraph=True, dynamic=False)
    for batch in (296, 133, 40, 700):
        torch.manual_seed(batch)
        bases = [
            torch.randn(*size, device="cuda", dtype=torch.float16)
            for size in ((batch, 20992), (96, batch, 128), (batch, 2, 4352))
        ]
        for base, dim in zip(bases, (0, 1, 0)):
            torch._dynamo.mark_dynamic(base, dim)
        weights = [torch.rand(n, device="cuda", dtype=torch.float16) + 0.5 for n in (4096, 8704, 20992, 20992)]
        eager = norm(*bases, *weights)
        if batch == 296:
            outs, code = run_and_get_code(compiled, *bases, *weights)
            assert "_rmsnorm_cat_tlx_fwd" in code[0] and "_rmsnorm_cat_fwd" not in code[0]
        else:
            outs = compiled(*bases, *weights)
        for out, ref in zip(outs, eager):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.skipif(torch.__version__ < "2.16.0.dev20261029",
                    reason="older AOTInductor autotunes with a tensor in place of a TMA descriptor argument")
def test_rmsnorm_cat_aoti_package_matches_eager(tmp_path):
    """AOTInductor with a dynamic batch and the autotune settings production lowering uses, small and large batches."""
    op = _traceable_op()

    class Norm(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterList(
                torch.nn.Parameter(torch.rand(n, device="cuda", dtype=torch.float16) + 0.5, requires_grad=False)
                for n in (4096, 8704, 20992, 20992))

        def forward(self, float_proj, gemm, dots):
            return op(float_proj[:, :8192], gemm[64:].permute(1, 0, 2), dots.permute(0, 2, 1), *self.weights)

    def bases(batch):
        torch.manual_seed(batch)
        return [
            torch.randn(*size, device="cuda", dtype=torch.float16)
            for size in ((batch, 20992), (96, batch, 128), (batch, 2, 4352))
        ]

    norm = Norm()
    batch = torch.export.Dim("batch", min=2, max=1024)
    exported = torch.export.export(norm, tuple(bases(296)), dynamic_shapes=({0: batch}, {1: batch}, {0: batch}))
    with torch._inductor.config.patch({
            "triton.autotune_at_compile_time": True,
            "triton.autotune_with_sample_inputs": True,
    }):
        package = torch._inductor.aoti_compile_and_package(exported, package_path=str(tmp_path / "rmsnorm_cat.pt2"))
    compiled = torch._inductor.aoti_load_package(package)
    for n in (296, 40, 700):
        args = bases(n)
        for out, ref in zip(compiled(*args), norm(*args)):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
