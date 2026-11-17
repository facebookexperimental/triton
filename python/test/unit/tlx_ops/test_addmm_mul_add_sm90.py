"""L1 correctness for the Hopper ``tlx.ops.addmm_mul_add`` backend."""

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hopper
from triton.tlx.ops import InvalidInput, UnsupportedBackward
from triton.tlx.ops.kernels.fused.addmm_mul_add._shapes import CORRECTNESS_SHAPES, inputs

pytestmark = pytest.mark.skipif(not is_hopper(), reason="requires an sm90 GPU")

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
TOL = {torch.float16: 2e-2, torch.bfloat16: 6e-2}


def reference(bias, a, b, x0, res):
    """The unfused sequence: ``linear`` in the input dtype, then mul/add in fp32."""
    lin = a @ b if bias is None else torch.addmm(bias, a, b)
    res = x0 if res is None else res
    return (x0.float() * lin.float() + res.float()).to(x0.dtype)


def assert_matches(actual, expected):
    tol = TOL[expected.dtype]
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("m,n,k,has_res,dtype_name", CORRECTNESS_SHAPES)
def test_addmm_mul_add(m, n, k, has_res, dtype_name):
    from triton.tlx.ops import addmm_mul_add

    dtype = DTYPES[dtype_name]
    torch.manual_seed(m + n + k)
    bias, a, b, x0, res = inputs((m, n, k, has_res, dtype_name), dtype)
    out = torch.empty_like(x0)
    actual = addmm_mul_add(bias, a, b, x0, res, out=out)

    assert actual is out
    assert_matches(actual, reference(bias, a, b, x0, res))


@pytest.mark.parametrize(
    "m,n,k",
    [(1, 128, 64), (65, 136, 48),  # K below one K tile, N not a multiple of the tile
     (650, 520, 200),  # last M tile has a consumer half entirely past M
     (130, 1000, 520),  # K not a multiple of 64 with a five-stage ring
     ],
)
@pytest.mark.parametrize("has_res", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ragged_shapes(m, n, k, has_res, dtype):
    from triton.tlx.ops import addmm_mul_add

    torch.manual_seed(0)
    bias, a, b, x0, res = inputs((m, n, k, has_res, None), dtype)
    assert_matches(addmm_mul_add(bias, a, b, x0, res), reference(bias, a, b, x0, res))


@pytest.mark.parametrize("alias", ["x0", "res"])
@pytest.mark.parametrize("space", ["heuristic", "full"])
def test_out_aliases_input(alias, space):
    from triton.tlx.ops import addmm_mul_add

    torch.manual_seed(1)
    bias, a, b, x0, res = inputs((706, 2560, 512, True, None), torch.float16)
    expected = addmm_mul_add(bias, a, b, x0, res)
    target = x0 if alias == "x0" else res
    actual = addmm_mul_add(bias, a, b, x0, res, out=target, space=space)

    assert actual is target
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_no_bias_and_row_major_b():
    from triton.tlx.ops import addmm_mul_add

    torch.manual_seed(2)
    _, a, b, x0, res = inputs((300, 768, 256, True, None), torch.float16)
    b_row_major = b.contiguous()
    assert b_row_major.stride(1) == 1
    expected = reference(None, a, b, x0, res)
    assert_matches(addmm_mul_add(None, a, b, x0, res), expected)
    assert_matches(addmm_mul_add(None, a, b_row_major, x0, res), expected)


def test_dcn_block_chain():
    """Two DCNv2 blocks, ``x = x0 * proj(x) + x``, as the model runs them."""
    from triton.tlx.ops import addmm_mul_add

    torch.manual_seed(3)
    dtype = torch.float16
    m, n, rank = 706, 2048, 256
    x0 = torch.randn(m, n, device="cuda", dtype=dtype)
    blocks = [(torch.nn.Linear(n, rank).cuda().to(dtype), torch.nn.Linear(rank, n).cuda().to(dtype)) for _ in range(2)]
    with torch.no_grad():
        x_ref = x0
        for down, up in blocks:
            x_ref = x0 * up(down(x_ref)) + x_ref
        x = x0
        for i, (down, up) in enumerate(blocks):
            x = addmm_mul_add(up.bias, down(x), up.weight.t(), x0, None if i == 0 else x)
    assert_matches(x, x_ref)


@triton.jit
def _fused_triton_gate(h_ptr, w_ptr, b_ptr, x0_ptr, x_ptr, out_ptr, M, K: tl.constexpr, N: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """A plain Triton DCN gate as deployed in production today (BM 64, BN 256, BK 32)."""
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n: tl.constexpr = N // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < M
    rm_safe = tl.where(mask_m, rm, 0).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(h_ptr + (rm_safe[:, None] * K + (k0 + rk)[None, :]))
        b = tl.load(w_ptr + (rn[None, :].to(tl.int64) * K + (k0 + rk)[:, None]))
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
    bias = tl.load(b_ptr + rn).to(tl.float32)
    z = (acc + bias[None, :]).to(out_ptr.dtype.element_ty).to(tl.float32)
    off = rm[:, None].to(tl.int64) * N + rn[None, :]
    x0 = tl.load(x0_ptr + off, mask=mask_m[:, None], other=0.0).to(tl.float32)
    x = tl.load(x_ptr + off, mask=mask_m[:, None], other=0.0).to(tl.float32)
    tl.store(out_ptr + off, (x0 * z + x).to(out_ptr.dtype.element_ty), mask=mask_m[:, None])


def _gate_inputs(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.relu(torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g))
    w = torch.randn(n, k, device="cuda", dtype=torch.float16, generator=g) / k**0.5
    bias = torch.randn(n, device="cuda", dtype=torch.float16, generator=g) * 0.1
    x0 = torch.randn(m, n, device="cuda", dtype=torch.float16, generator=g)
    x = torch.randn(m, n, device="cuda", dtype=torch.float16, generator=g)
    return h, w, bias, x0, x


@pytest.mark.parametrize("m", [1, 320, 525, 741, 767])
@pytest.mark.parametrize("same", [True, False])
def test_bitwise_equal_to_fused_triton_gate(m, same):
    """Drop-in for the deployed DCN gate kernel: bitwise equal at its production shape (N 22016, K 512)."""
    from triton.tlx.ops import addmm_mul_add

    n, k = 22016, 512
    h, w, bias, x0, x = _gate_inputs(m, n, k, seed=m)
    x = x0 if same else x
    expected = torch.empty_like(x0)
    _fused_triton_gate[(triton.cdiv(m, 64) * (n // 256), )](h, w, bias, x0, x, expected, m, K=k, N=n, BLOCK_M=64,
                                                            BLOCK_N=256, BLOCK_K=32, GROUP_M=8, num_warps=8,
                                                            num_stages=3)
    actual = addmm_mul_add(bias, h, w.t(), x0, None if same else x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("has_res", [False, True])
def test_launch_spec_matches_op(has_res):
    from triton.tlx.ops import addmm_mul_add, addmm_mul_add_launch

    h, w, bias, x0, x = _gate_inputs(300, 1024, 512, seed=4)
    res = x if has_res else None
    out = torch.empty_like(x0)
    launch = addmm_mul_add_launch(bias, h, w.t(), x0, res, out=out)
    launch.kernel[launch.grid](*launch.args, **launch.kwargs)
    torch.testing.assert_close(out, addmm_mul_add(bias, h, w.t(), x0, res), atol=0, rtol=0)


def _launch_kernel(kind, bias, a, b, x0, res, out):
    from triton.tlx.ops.kernels.fused.addmm_mul_add import sm90

    M, N, K, b_src, b_row_major = sm90._validate(bias, a, b, x0, res, out)
    fn = sm90._resident_launch if kind == "resident" else sm90._cooperative_launch
    launch = fn(bias, a, b_src, b_row_major, x0, res, out, M, N, K)
    launch.kernel[launch.grid](*launch.args, **launch.kwargs)
    return out


@pytest.mark.parametrize("m", [1, 7, 129, 600, 671, 704, 767, 768])
@pytest.mark.parametrize("same", [True, False])
def test_resident_bitwise_equal_to_cooperative(m, same):
    """The A-resident kernel gives the cooperative kernel's output bit for bit (N 22016, K 512)."""
    h, w, bias, x0, x = _gate_inputs(m, 22016, 512, seed=m + 11)
    res = None if same else x
    expected = _launch_kernel("cooperative", bias, h, w.t(), x0, res, torch.empty_like(x0))
    actual = _launch_kernel("resident", bias, h, w.t(), x0, res, torch.full_like(x0, float("nan")))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("m,n,k", [(706, 20992, 256), (300, 1000, 136), (17920, 1024, 256)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_resident_repeated_calls_bitwise_stable(m, n, k, has_bias):
    """Repeated calls agree bit for bit (handoff/bias slot reuse), incl. more M blocks than SMs (M = 140 x 128)."""
    h, w, bias, x0, x = _gate_inputs(m, n, k, seed=k)
    bias = bias if has_bias else None
    expected = _launch_kernel("cooperative", bias, h, w.t(), x0, x, torch.empty_like(x0))
    out = torch.empty_like(x0)
    for _ in range(10):
        out.fill_(float("nan"))
        _launch_kernel("resident", bias, h, w.t(), x0, x, out)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("k,kernel", [(64, "resident"), (512, "resident"), (520, "cooperative"), (1024, "cooperative")])
def test_launch_selects_kernel_by_k(k, kernel):
    from triton.tlx.ops import addmm_mul_add_launch
    from triton.tlx.ops.kernels.fused.addmm_mul_add import sm90

    h, w, bias, x0, _ = _gate_inputs(256, 1024, k, seed=6)
    launch = addmm_mul_add_launch(bias, h, w.t(), x0, out=torch.empty_like(x0))
    expected = sm90.addmm_mul_add_resident_kernel if kernel == "resident" else sm90.addmm_mul_add_kernel
    assert launch.kernel is expected


def test_misaligned_bias_uses_cooperative_kernel():
    """A bias whose base is not 16-byte aligned cannot be TMA-loaded; the launch falls back and stays exact."""
    from triton.tlx.ops import addmm_mul_add, addmm_mul_add_launch
    from triton.tlx.ops.kernels.fused.addmm_mul_add import sm90

    h, w, bias, x0, x = _gate_inputs(300, 1024, 512, seed=7)
    storage = torch.empty(1025, device="cuda", dtype=torch.float16)
    storage[1:].copy_(bias)
    shifted = storage[1:]
    assert shifted.data_ptr() % 16 != 0
    launch = addmm_mul_add_launch(shifted, h, w.t(), x0, x, out=torch.empty_like(x0))
    assert launch.kernel is sm90.addmm_mul_add_kernel
    torch.testing.assert_close(addmm_mul_add(shifted, h, w.t(), x0, x), addmm_mul_add(bias, h, w.t(), x0, x), atol=0,
                               rtol=0)


@torch.library.triton_op("tlx_ops_test::dcn_gate", mutates_args=())
def _dcn_gate_op(h: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, x0: torch.Tensor,
                 x: torch.Tensor) -> torch.Tensor:
    from triton.tlx.ops import addmm_mul_add_launch

    out = torch.empty_like(x0)
    launch = addmm_mul_add_launch(bias, h, w.t(), x0, None if x is x0 else x, out=out)
    torch.library.wrap_triton(launch.kernel)[launch.grid](*launch.args, **launch.kwargs)
    return out


def test_launch_spec_exports_with_dynamic_m():
    """The launch spec traces under torch.export with a symbolic M, as a triton_op for AOTInductor."""
    from triton.tlx.ops import addmm_mul_add

    n, k = 2560, 512
    h, w, bias, x0, _ = _gate_inputs(300, n, k, seed=5)

    class TwoBlocks(torch.nn.Module):

        def forward(self, h, x0):
            x1 = _dcn_gate_op(h, w, bias, x0, x0)
            return _dcn_gate_op(h, w, bias, x0, x1)

    m_dim = torch.export.Dim("M", min=1, max=4096)
    ep = torch.export.export(TwoBlocks(), (h, x0), dynamic_shapes=({0: m_dim}, {0: m_dim}))
    for m in (1, 129, 741):
        h = torch.randn(m, k, device="cuda", dtype=torch.float16)
        x0 = torch.randn(m, n, device="cuda", dtype=torch.float16)
        x1 = addmm_mul_add(bias, h, w.t(), x0)
        expected = addmm_mul_add(bias, h, w.t(), x0, x1)
        torch.testing.assert_close(ep.module()(h, x0), expected, atol=0, rtol=0)


def test_rejects_invalid_inputs():
    from triton.tlx.ops import addmm_mul_add

    bias, a, b, x0, res = inputs((128, 256, 64, True, None), torch.float16)
    with pytest.raises(InvalidInput):
        addmm_mul_add(bias.float(), a.float(), b.float(), x0.float(), res.float())
    with pytest.raises(InvalidInput):
        addmm_mul_add(bias, a, b, x0[:, :128], res)
    with pytest.raises(InvalidInput):
        addmm_mul_add(bias, a, b, x0, res.t().contiguous().t())
    with pytest.raises(InvalidInput):
        addmm_mul_add(bias[:128], a, b, x0, res)
    with pytest.raises(InvalidInput):
        addmm_mul_add(bias, a, b, x0, res, space="smoke")
    with pytest.raises(UnsupportedBackward):
        addmm_mul_add(bias, a, b, x0.requires_grad_(), res)


def _shifted(t):
    """A copy of ``t`` whose base sits one element (2 bytes) past a 16-byte boundary."""
    storage = torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype)
    shifted = storage[1:].view(t.shape)
    shifted.copy_(t)
    return shifted


def test_traced_misaligned_bias_uses_cooperative_kernel():
    """Under tracing the bias has no address, but its storage offset shows the misalignment; the launch falls back."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from triton.tlx.ops import addmm_mul_add, addmm_mul_add_launch
    from triton.tlx.ops.kernels.fused.addmm_mul_add import sm90

    h, w, bias, x0, x = _gate_inputs(300, 1024, 512, seed=8)
    shifted = _shifted(bias)
    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(t) for t in (shifted, h, w, x0, x)]
        launch = addmm_mul_add_launch(fake[0], fake[1], fake[2].t(), fake[3], fake[4], out=torch.empty_like(fake[3]))
    assert launch.kernel is sm90.addmm_mul_add_kernel

    compiled = torch.compile(_dcn_gate_op, fullgraph=True)
    torch.testing.assert_close(compiled(h, w, shifted, x0, x), addmm_mul_add(bias, h, w.t(), x0, x), atol=0, rtol=0)


@pytest.mark.parametrize("name", ["a", "x0"])
def test_traced_misaligned_operand_rejected(name):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from triton.tlx.ops import addmm_mul_add_launch

    h, w, bias, x0, x = _gate_inputs(300, 1024, 512, seed=9)
    if name == "a":
        h = _shifted(h)
    else:
        x0 = _shifted(x0)
    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(t) for t in (bias, h, w, x0, x)]
        with pytest.raises(InvalidInput):
            addmm_mul_add_launch(fake[0], fake[1], fake[2].t(), fake[3], fake[4], out=torch.empty_like(fake[3]))


def test_rejects_row_stride_past_int32_tile_offsets():
    from triton.tlx.ops import addmm_mul_add

    h, w, bias, _, _ = _gate_inputs(2, 64, 64, seed=14)
    x0 = torch.empty_strided((2, 64), (2**31 // 64, 1), device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput):
        addmm_mul_add(bias, h, w.t(), x0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("space", ["heuristic", "full"])
def test_inputs_on_a_non_current_device(space):
    """Inputs on cuda:1 while cuda:0 is current run on cuda:1 (with peer access, a launch on cuda:0 would also be
    numerically right, so the test checks the device of the kernel)."""
    from triton.tlx.ops import addmm_mul_add

    device = torch.device("cuda", 1)
    h, w, bias, x0, x = (t.to(device) for t in _gate_inputs(300, 1024, 512, seed=10))
    with torch.cuda.device(0):
        addmm_mul_add(bias, h, w.t(), x0, x, space=space)  # compile and tune outside the profile
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            actual = addmm_mul_add(bias, h, w.t(), x0, x, space=space)
            torch.cuda.synchronize(device)
    kernels = [
        e for e in prof.events() if "addmm_mul_add" in e.name and e.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert kernels and all(e.device_index == 1 for e in kernels), [(e.name, e.device_index) for e in kernels]
    assert actual.device == device
    assert_matches(actual, reference(bias, h, w.t(), x0, x))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 40 * 2**30,
                    reason="needs about 14 GiB of free device memory")
@pytest.mark.parametrize("k,kernel", [(512, "resident"), (1024, "cooperative")])
def test_rows_past_int32_element_offsets(k, kernel):
    """Rows whose element offset ``row * N`` exceeds INT32_MAX are read and written at the right address."""
    m, n = 97543, 22016  # (m - 1) * n > 2**31 - 1
    h, w, bias, _, _ = _gate_inputs(1, n, k, seed=12)
    g = torch.Generator(device="cuda").manual_seed(13)
    h = torch.relu(torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g))
    x0 = torch.randn(m, n, device="cuda", dtype=torch.float16, generator=g)
    x = torch.randn(m, n, device="cuda", dtype=torch.float16, generator=g)
    out = torch.full_like(x0, float("nan"))
    _launch_kernel(kernel, bias, h, w.t(), x0, x, out)
    rows = slice(m - 64, m)
    expected = _launch_kernel(kernel, bias, h[rows].contiguous(), w.t(), x0[rows].contiguous(), x[rows].contiguous(),
                              torch.empty_like(x0[rows]))
    torch.testing.assert_close(out[rows], expected, atol=0, rtol=0)
