"""L1 correctness for ``tlx.ops.flash_attn_mxfp8`` on gfx950."""

import subprocess
import sys
from unittest import mock

import pytest
import torch
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.flash_attn_mxfp8._shapes import CORRECTNESS_SHAPES

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="tlx.ops.flash_attn_mxfp8 on gfx950 requires CDNA4")

MULTI_WAVE_SHAPE = (1, 64, 1024, 128)


def _qkv(shape, *, requires_grad=False):
    return [(torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_(requires_grad)
            for _ in range(3)]


def _sdpa(q, k, v, causal, scale):
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
        scale=scale,
    )


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_flash_attn_mxfp8_fwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv((Z, H, N_CTX, HEAD_DIM))
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    ref = _sdpa(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=0.2, rtol=0)


def _blackwell_mxfp8(x, rows, cols):
    # Blackwell's MXFP8 rule for each rows x cols block of x's last two dims:
    # E8M0 byte RCEIL(amax * fp32(1 / 448)) (cvt.rp.satfinite.ue8m0x2.f32),
    # E4M3 data x / 2**(byte - 127) rounded to nearest.
    z, h, n, d = x.shape
    blocks = x.float().reshape(z, h, n // rows, rows, d // cols, cols)
    amax = blocks.abs().amax(dim=(3, 5), keepdim=True)
    byte = ((amax * (1.0 / 448.0)).view(torch.int32) + 0x7FFFFF) >> 23
    data = (blocks * torch.exp2(127.0 - byte.float())).clamp(-448, 448).to(torch.float8_e4m3fn)
    return data.reshape(z, h, n, d), byte.reshape(z, h, n // rows, d // cols).to(torch.uint8)


def test_flash_attn_mxfp8_quantizers_match_blackwell():
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950 import quantize_mxfp8_head, quantize_mxfp8_v

    torch.manual_seed(20)
    shape = (2, 3, 512, 128)
    # Token magnitudes that vary across each 32-row block, and an all-zero block.
    x, v = [torch.randn(shape, device="cuda") * torch.randn(shape[:3] + (1, ), device="cuda").exp() for _ in range(2)]
    x[0, 0, :32, :32] = 0
    x, v = x.to(torch.bfloat16), v.to(torch.bfloat16)
    data, scale = quantize_mxfp8_head(x)
    ref_data, ref_scale = _blackwell_mxfp8(x, 32, 32)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(scale, ref_scale.repeat_interleave(32, 2))
    data, scale = quantize_mxfp8_v(v)
    ref_data, ref_scale = _blackwell_mxfp8(v, 32, 1)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(scale, ref_scale)


def test_flash_attn_mxfp8_fast_exp_bound():
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950

    torch.manual_seed(20)
    q, k, v = _qkv((1, 4, 1024, 128))
    # A flat softmax, where the bit-trick exp2 errs the most against the rest
    # of the MXFP8 error (about 9% more error here, 0-6% at sharper ones).
    sm_scale = 128**-0.5
    assert gfx950._default_config(False, q.shape[2])["pingpong"]
    q8, qs = gfx950.quantize_mxfp8_head(q)
    k8, ks = gfx950.quantize_mxfp8_head(k, pack_k=True)
    v8, vs = gfx950.quantize_mxfp8_v(v, transposed=True)
    fast = gfx950._launch_quantized(q8, k8, v8, qs, ks, vs, False, sm_scale).float()
    exact = gfx950._launch_quantized(q8, k8, v8, qs, ks, vs, False, sm_scale, fast_exp=False).float()
    ref = _sdpa(q.float(), k.float(), v.float(), False, sm_scale)

    def rms(t):
        return t.pow(2).mean().sqrt().item()

    assert rms(fast - ref) < 1.15 * rms(exact - ref)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_mxfp8_fwd_multiple_cta_waves(causal):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(20)
    q, k, v = _qkv(MULTI_WAVE_SHAPE)
    scale = 0.5
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    ref = _sdpa(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0)


def test_flash_attn_mxfp8_supports_backward():
    from triton.tlx.ops import flash_attn_mxfp8

    q, k, v = _qkv((1, 1, 256, 128), requires_grad=True)
    out = flash_attn_mxfp8(q, k, v, space="smoke")
    grads = torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
    assert all(x.dtype == torch.bfloat16 and bool(x.isfinite().all()) for x in grads)


@pytest.mark.parametrize(
    "shape,dtype,match",
    [
        ((1, 1, 256, 64), torch.bfloat16, "does not support"),
        ((1, 1, 384, 128), torch.bfloat16, "does not support"),
        ((1, 1, 256, 128), torch.float16, "does not support"),
    ],
)
def test_flash_attn_mxfp8_rejects_unsupported_inputs(shape, dtype, match):
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
    with pytest.raises(InvalidInput, match=match):
        flash_attn_mxfp8(q, k, v, space="smoke")


def test_flash_attn_mxfp8_rejects_mismatched_shapes():
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q = torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with pytest.raises(InvalidInput, match="identical shapes"):
        flash_attn_mxfp8(q, k, v, space="smoke")


def test_flash_attn_mxfp8_rejects_unknown_space():
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q, k, v = _qkv((1, 1, 256, 128))
    with pytest.raises(InvalidInput, match="does not provide space"):
        flash_attn_mxfp8(q, k, v, space="heuristic")


def _gfx950_device_indices():
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").startswith("gfx950")
    ]


def _has_two_gfx950_devices():
    return len(_gfx950_device_indices()) >= 2


@pytest.mark.parametrize(
    "n_ctx,causal,scale",
    [(n, causal, 0.5) for n in (1024, 2048, 4096, 8192) for causal in (False, True)] + [(8192, False, 1.3)],
)
def test_flash_attn_mxfp8_shared_backward_dispatch_and_graph(n_ctx, causal, scale):
    from triton.tlx.ops import flash_attn_mxfp8
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd, gfx950_bwd_shared

    torch.manual_seed(20)
    inputs = _qkv((4, 32, n_ctx, 128), requires_grad=True)
    do = torch.randn_like(inputs[0]) * 0.5
    out = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale)
    reference = _sdpa(*inputs, causal, scale)
    expected = torch.autograd.grad(reference, inputs, do)
    _, _, v, q8, k8, saved_out, lse, qs, ks = out.grad_fn.saved_tensors
    args = (q8, k8, qs, ks, v, do, saved_out, lse, scale)
    assert gfx950_bwd_shared.can_use_shared_square(*args, causal=causal)
    assert not gfx950_bwd_shared.can_use_shared_square(*args, causal=1)
    for unsupported_scale in (0.7, float("nan"), float("inf")):
        assert not gfx950_bwd_shared.can_use_shared_square(*args[:-1], unsupported_scale, causal=causal)
    if scale == 1.3:
        assert not gfx950_bwd_shared.can_use_shared_square(*args, causal=True)
    # A contiguous view of a larger allocation is outside the fast-path
    # whole-storage contract; it must remain on the general implementation.
    oversized = torch.empty(v.numel() + 1, device=v.device, dtype=v.dtype)
    view = oversized[1:].view_as(v)
    assert not gfx950_bwd_shared.can_use_shared_square(*args[:4], view, *args[5:], causal=causal)
    del oversized, view

    def check(grads):
        for actual, ref in zip(grads, expected):
            actual, ref = actual.float(), ref.float()
            assert bool(actual.isfinite().all())
            rrms = ((actual - ref).square().mean() / ref.square().mean()).sqrt()
            cosine = torch.nn.functional.cosine_similarity(actual.flatten(), ref.flatten(), dim=0)
            assert rrms < 0.15 and cosine >= 0.98

    def backward():
        return torch.autograd.grad(out, inputs, do, retain_graph=True)

    with mock.patch.object(gfx950_bwd, "launch_backward", side_effect=AssertionError("unexpected general path")):
        with mock.patch.object(gfx950_bwd_shared, "_check_shared_square_inputs",
                               wraps=gfx950_bwd_shared._check_shared_square_inputs) as validate:
            with mock.patch.object(gfx950_bwd_shared, "_is_gfx950", wraps=gfx950_bwd_shared._is_gfx950) as arch:
                eager = backward()
        assert validate.call_count == arch.call_count == 1
        check(eager)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        workspaces = {}
        original_empty = torch.empty
        arena_bytes = 2 * (128 * n_ctx * 128) + 4 * (128 * n_ctx * 4)

        def observe_empty(*allocation_args, **kwargs):
            tensor = original_empty(*allocation_args, **kwargs)
            if tuple(tensor.shape) == (arena_bytes, ) and tensor.dtype == torch.uint8:
                assert "arena" not in workspaces, "expected one private preparation arena"
                workspaces["arena"] = tensor
            elif tuple(tensor.shape) == (4, 32, n_ctx, n_ctx) and tensor.dtype == torch.float8_e4m3fn:
                assert "ds" not in workspaces, "expected exactly one captured backward"
                workspaces["ds"] = tensor
            elif "ds" in workspaces and "dss" not in workspaces:
                # DSS is allocated immediately after DS. Shape alone is not
                # unique: at N4096 it equals the preparation scale shapes.
                assert tuple(tensor.shape) == (4, 32, n_ctx // 32, n_ctx // 32)
                assert tensor.dtype == torch.uint8 and tensor.device == inputs[0].device
                workspaces["dss"] = tensor
            return tensor

        try:
            with torch.cuda.stream(stream):
                # Autograd records the forward stream on its nodes. Create
                # fresh leaves and this graph on the capture stream.
                inputs = tuple(x.detach().clone().requires_grad_() for x in inputs)
                out = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale)
                for _ in range(2):
                    backward()
                stream.synchronize()
                # Observe only the one captured invocation, not warmups, so
                # large temporary allocations from other calls are not kept.
                with mock.patch.object(gfx950_bwd_shared.torch, "empty", side_effect=observe_empty):
                    with torch.cuda.graph(graph, stream=stream):
                        captured = backward()
                assert set(workspaces) == ({"arena", "ds", "dss"} if n_ctx == 1024 else {"ds", "dss"})
                for _ in range(2):
                    # Poison before each replay, outside the captured graph.
                    # E4M3 byte0x7f and E8M0 byte255 represent NaN. Every
                    # consumed workspace value must come from this replay.
                    # Also poison both private payloads, their scales, and
                    # FP32 Delta. Views retain this storage for graph replay.
                    if n_ctx == 1024:
                        workspaces["arena"].fill_(255)
                    workspaces["ds"].view(torch.uint8).fill_(0x7f)
                    workspaces["dss"].fill_(255)
                    for grad in captured:
                        grad.fill_(float("nan"))
                    graph.replay()
                    stream.synchronize()
                    if causal:
                        ds_bytes = workspaces["ds"].view(torch.uint8)
                        # Physical [key,query] blocks wholly above the
                        # causal domain are not initialized or consumed.
                        assert bool((ds_bytes[0, 0, 128:192, :64] == 0x7f).all())
                        # Causal scales pack each Q128/K64 record as [4,2]
                        # query/key groups, despite the capacity-only shape.
                        dss_records = workspaces["dss"].view(4, 32, n_ctx // 128, n_ctx // 64, 4, 2)
                        assert bool((dss_records[0, 0, 0, 2, :2, :] == 255).all())
                        # Within diagonal128, even masked key64/query64
                        # blocks are fully written as zeros before consumption.
                        assert bool((ds_bytes[0, 0, 64:128, :64] == 0).all())
                        assert bool((dss_records[0, 0, 0, 1, :2, :] != 255).all())
                    check(captured)
                    for actual, ref in zip(captured, eager):
                        torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        finally:
            stream.synchronize()
            graph.reset()
            workspaces.clear()


@pytest.mark.parametrize("n_ctx", (1024, 2048, 4096, 8192))
def test_flash_attn_mxfp8_shared_backward_arena_storage(n_ctx):
    """Check the actual launch allocation contract without compiling kernels."""
    from triton import knobs
    from triton._C.libtriton import native_specialize_impl
    from triton.backends.amd.compiler import HIPBackend
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    shape = (4, 32, n_ctx, 128)
    packed = n_ctx == 1024
    preparation_allocations = 1 if packed else 6
    original_empty = torch.empty
    allocations = []

    def allocate(allocation_shape, *, dtype, device):
        # Materialize preparation storage on CPU. DS and gradients can be
        # large, and mocked launchers need only their metadata.
        actual_device = "cpu" if len(allocations) < preparation_allocations else "meta"
        tensor = original_empty(allocation_shape, dtype=dtype, device=actual_device)
        allocations.append(tensor)
        return tensor

    prepare_entry = shared._prepare_fused_arena if packed else shared._prepare_fused
    kv_entry = shared._bwd_kv_owner_arena if packed else shared._bwd_kv_owner
    q_entry = shared._bwd_q_consume_arena if packed else shared._bwd_q_consume
    with mock.patch.object(shared.torch, "empty", side_effect=allocate):
        with mock.patch.object(prepare_entry, "run") as prepare:
            with mock.patch.object(kv_entry, "run") as kv:
                with mock.patch.object(q_entry, "run") as query:
                    with mock.patch.object(torch.Tensor, "narrow", side_effect=AssertionError("host narrow")):
                        with mock.patch.object(torch.Tensor, "view", side_effect=AssertionError("host view")):
                            gradients = shared._launch_backward_shared_square(*([object()] * 8), 0.5, False,
                                                                              torch.device("cpu"), shape)
                            if packed:
                                first_arena = allocations[0]
                                allocations.clear()
                                gradients = shared._launch_backward_shared_square(*([object()] * 8), 0.5, False,
                                                                                  torch.device("cpu"), shape)
                                assert allocations[0].data_ptr() != first_arena.data_ptr()
    assert len(allocations) == preparation_allocations + 5  # Preparation, DS, DSS, and three outputs.
    args = prepare.call_args.args
    expected_shapes = (shape, shape, (4, 32, n_ctx, 4), (4, 32, n_ctx, 4), (4, 32, 128, n_ctx // 32), shape[:-1])
    expected_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.uint8, torch.uint8, torch.uint8, torch.float32)
    if packed:
        arena = allocations[0]
        assert args[4] is kv.call_args.args[5] is query.call_args.args[3] is arena
        assert kv.call_args.kwargs["ARENA_N"] == query.call_args.kwargs["ARENA_N"] == n_ctx
        assert arena.dtype == torch.uint8 and arena.ndim == 1 and arena.storage_offset() == 0
        offsets = shared._preparation_arena_layout(n_ctx)
        assert offsets[-1] == arena.numel()
        # These test-only views model the JIT's typed pointer segments.
        private = [
            arena.narrow(0, start, end - start).view(dtype).view(expected_shape)
            for start, end, dtype, expected_shape in zip(offsets, offsets[1:], expected_dtypes, expected_shapes)
        ]
        with pytest.raises(AssertionError):
            shared._preparation_arena_layout(2048)
    else:
        private = (*args[3:8], args[9])  # V8, dO8, VS, dOS, KDQS, Delta.
    if not packed:
        assert all(tensor is allocated for tensor, allocated in zip(private, allocations[:6]))
        assert len({tensor.untyped_storage().data_ptr() for tensor in private}) == 6
    offset = 0
    byte_views = []
    with mock.patch.object(knobs.amd, "use_buffer_ops", True):
        for index, (tensor, expected_shape, dtype) in enumerate(zip(private, expected_shapes, expected_dtypes)):
            assert tuple(tensor.shape) == expected_shape and tensor.dtype == dtype
            assert tensor.is_contiguous() and tensor.device.type == "cpu"
            tensor_bytes = tensor.numel() * tensor.element_size()
            if packed:
                arena = allocations[0]
                assert tensor.untyped_storage().data_ptr() == arena.data_ptr()
                assert tensor.data_ptr() == arena.data_ptr() + offset
                assert tensor.untyped_storage().nbytes() == arena.numel() < 2**31
            else:
                assert tensor.storage_offset() == 0
                assert tensor.untyped_storage().data_ptr() == tensor.data_ptr()
                assert tensor.untyped_storage().nbytes() == tensor_bytes < 2**31
            assert tensor.data_ptr() % 16 == 0
            _, attributes = native_specialize_impl(HIPBackend, tensor, False, True, True)
            assert "D" in attributes and "S" in attributes
            byte_view = tensor.view(torch.uint8).reshape(-1)
            byte_view[0] = byte_view[-1] = index + 1
            byte_views.append(byte_view)
            offset += tensor_bytes
    assert offset == 34816 * n_ctx
    if packed:
        assert offset == allocations[0].numel()
    for index, byte_view in enumerate(byte_views):
        assert int(byte_view[0]) == int(byte_view[-1]) == index + 1
    assert all(gradient is allocation for gradient, allocation in zip(gradients, allocations[-3:]))


@pytest.mark.parametrize("n_ctx", (1024, 2048, 4096, 8192))
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_mxfp8_shared_backward_live_metadata(n_ctx, causal):
    """Admission rechecks each tensor even when its identity is unchanged."""
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    shape = (4, 32, n_ctx, 128)
    scale_shape = (4, 32, n_ctx, 4)
    device = torch.device("cuda:0")
    metadata = ((shape, torch.float8_e4m3fn, 1), (shape, torch.float8_e4m3fn, 1), (scale_shape, torch.uint8, 1),
                (scale_shape, torch.uint8, 1), (shape, torch.bfloat16, 2), (shape, torch.bfloat16, 2),
                (shape, torch.bfloat16, 2), (shape[:-1], torch.float32, 4))
    tensors = []
    for index, (tensor_shape, dtype, item_bytes) in enumerate(metadata):
        tensor = mock.Mock(spec=torch.Tensor)
        tensor.shape, tensor.dtype, tensor.device, tensor.layout = torch.Size(
            tensor_shape), dtype, device, torch.strided
        tensor.is_contiguous.return_value = True
        tensor.storage_offset.return_value = 0
        tensor.is_conj.return_value = tensor.is_neg.return_value = False
        tensor.numel.side_effect = tensor.element_size.side_effect = AssertionError("redundant tensor metadata")
        tensor.data_ptr.return_value = (index + 1) * 16
        elements = 1
        for extent in tensor_shape:
            elements *= extent
        tensor.untyped_storage.return_value.nbytes.return_value = elements * item_bytes
        tensor.untyped_storage.return_value.data_ptr.return_value = tensor.data_ptr.return_value
        tensors.append(tensor)

    def admit():
        return shared._check_shared_square_inputs(*tensors, 0.5, causal)

    assert admit() == (device, shape)
    for tensor in tensors:
        storage = tensor.untyped_storage.return_value
        for target, attribute, invalid_value in (
            (tensor, "shape", torch.Size((*tensor.shape[:-1], tensor.shape[-1] + 1))),
            (tensor.storage_offset, "return_value", 1),
            (tensor.is_contiguous, "return_value", False),
            (tensor.is_conj, "return_value", True),
            (tensor.is_neg, "return_value", True),
            (storage.nbytes, "return_value", storage.nbytes.return_value + 1),
            (storage.data_ptr, "return_value", storage.data_ptr.return_value + 16),
            (tensor.data_ptr, "return_value", tensor.data_ptr.return_value + 1),
        ):
            original = getattr(target, attribute)
            setattr(target, attribute, invalid_value)
            with pytest.raises(ValueError):
                admit()
            setattr(target, attribute, original)
            assert admit() == (device, shape)


def test_flash_attn_mxfp8_shared_backward_host_dispatch():
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    shape = (4, 32, 1024, 128)
    scale_shape = (4, 32, 1024, 4)
    q, k = [torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn) for _ in range(2)]
    qs, ks = [torch.empty(scale_shape, device="cuda", dtype=torch.uint8) for _ in range(2)]
    v, do, out = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    args = (q, k, qs, ks, v, do, out, lse, 0.5)
    result = object()

    # Both routes validate once; only the internal caller owns the device
    # context. Stub the common launch body, not the metadata checks.
    with torch.cuda.device(q.device):
        for entry in (shared._try_launch_backward_shared_square, shared.launch_backward_shared_square):
            with mock.patch.object(shared, "_launch_backward_shared_square", return_value=result) as launch:
                with mock.patch.object(shared, "_check_shared_square_inputs",
                                       wraps=shared._check_shared_square_inputs) as validate:
                    assert entry(*args) is result
                assert validate.call_count == launch.call_count == 1
                assert launch.call_args.args[-2:] == (q.device, shape)
            # A ValueError from launch is not an admission failure. Neither
            # it nor a runtime/allocation error may be silently retried.
            for error in (ValueError("launch failed"), RuntimeError("allocation failed")):
                with mock.patch.object(shared, "_launch_backward_shared_square", side_effect=error):
                    with pytest.raises(type(error), match=str(error)):
                        entry(*args)

        with mock.patch.object(shared, "_launch_backward_shared_square") as launch:
            assert shared._try_launch_backward_shared_square(*args, causal=1) is None
            with pytest.raises(ValueError, match="actual bool"):
                shared.launch_backward_shared_square(*args, causal=1)
            assert shared._try_launch_backward_shared_square(*args[:-1], 1) is None
            with pytest.raises(ValueError, match="Python float"):
                shared.launch_backward_shared_square(*args[:-1], 1)
            with mock.patch.object(shared, "_is_gfx950", return_value=False):
                assert shared._try_launch_backward_shared_square(*args) is None
                with pytest.raises(ValueError, match="gfx950-only"):
                    shared.launch_backward_shared_square(*args)
            launch.assert_not_called()


def _mxfp8_reference_quantize(x, *, sequence=False, square=False):
    # Independent mathematical RCEIL, without the implementation's bit tricks.
    x = x.float().transpose(-1, -2).contiguous() if sequence else x.float()
    rows, cols = x.shape[-2:]
    if square:
        blocks = x.reshape(*x.shape[:-2], rows // 32, 32, cols // 32, 32)
        maximum = blocks.abs().amax((-3, -1), keepdim=True)
    else:
        blocks = x.reshape(*x.shape[:-1], cols // 32, 32)
        maximum = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(maximum * (1.0 / 448.0))).clamp(-127, 127)
    scale = torch.exp2(exponent)
    payload = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    decoded = (payload.float() * scale).reshape(x.shape)
    if square:
        byte = (exponent.squeeze(-1).squeeze(-2) + 127).to(torch.uint8).repeat_interleave(32, -2)
    else:
        byte = (exponent.squeeze(-1) + 127).to(torch.uint8)
    payload = payload.reshape(x.shape)
    if sequence:
        payload = payload.transpose(-1, -2).contiguous()
        decoded = decoded.transpose(-1, -2).contiguous()
    return payload, byte, decoded


def _shared_fp8_boundary_fixture(causal):
    from triton.tlx.ops import flash_attn_mxfp8
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_quant import quantize_mxfp8

    torch.manual_seed(20)
    inputs = _qkv((4, 32, 1024, 128), requires_grad=True)
    do = torch.randn_like(inputs[0]) * .5
    out = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=.5)
    _, _, v, q8, k8, saved_out, lse, qs, ks = out.grad_fn.saved_tensors
    # Backward V uses feature32, not forward V's sequence32 quantization.
    v8, vs = quantize_mxfp8(v, transpose_for_reduction=False)
    args = (q8, k8, v8, qs, ks, vs, do, saved_out, lse, .5)
    return inputs, out, args


@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_mxfp8_shared_fp8_boundary_graph(causal):
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    inputs, out, args = _shared_fp8_boundary_fixture(causal)
    q8, k8, v8, qs, ks, vs, do, saved_out, lse, scale = args
    reference = _sdpa(*inputs, causal, scale)
    bf16_grads = torch.autograd.grad(reference, inputs, do)
    legacy = shared.launch_backward_shared_square(q8, k8, qs, ks, inputs[2], do, saved_out, lse, scale, causal=causal)
    frozen_inputs = tuple(t.clone() for t in args[:-1])

    def equal(a, b):
        assert a.shape == b.shape and a.dtype == b.dtype and a.is_contiguous() and b.is_contiguous()
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    def check(grads):
        for actual, old, ref in zip(grads, legacy, bf16_grads):
            equal(actual, old)
            a, b = actual.float(), ref.float()
            assert bool(a.isfinite().all())
            assert ((a - b).square().mean() / b.square().mean()).sqrt() < .15
            assert torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0) >= .98

    # Compare every new preparation output with the unchanged fused producer.
    vb, do8 = [torch.empty_like(v8) for _ in range(2)]
    old_vs, dos = [torch.empty_like(vs) for _ in range(2)]
    kdqs = torch.empty((4, 32, 128, 32), device=v8.device, dtype=torch.uint8)
    delta = torch.empty_like(lse)
    shared._prepare_fused[(32, 128)](inputs[2], do, ks, vb, do8, old_vs, dos, kdqs, saved_out, delta, 1024, 128, 32,
                                     num_warps=4, num_stages=2)
    equal(v8, vb)
    equal(vs, old_vs)
    prepared = shared.prepare_backward_shared_square_mxfp8(ks, do, saved_out, scale, causal=causal)
    for actual, expected in zip(prepared, (do8, dos, delta, kdqs)):
        equal(actual, expected)
    prepared_before = tuple(t.clone() for t in prepared)
    core_args = (q8, k8, v8, qs, ks, vs, *prepared, lse, scale)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    prep_graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            shared.prepare_backward_shared_square_mxfp8(ks, do, saved_out, scale, causal=causal)
            stream.synchronize()
            with torch.cuda.graph(prep_graph, stream=stream):
                captured_prep = shared.prepare_backward_shared_square_mxfp8(ks, do, saved_out, scale, causal=causal)
            for _ in range(2):
                for tensor in captured_prep:
                    if tensor.dtype == torch.float8_e4m3fn:
                        tensor.view(torch.uint8).fill_(0x7f)
                    elif tensor.dtype == torch.uint8:
                        tensor.fill_(255)
                    else:
                        tensor.fill_(float("nan"))
                prep_graph.replay()
                stream.synchronize()
                for actual, expected in zip(captured_prep, prepared_before):
                    equal(actual, expected)
    finally:
        stream.synchronize()
        prep_graph.reset()
    entries = ((shared.launch_backward_shared_square_mxfp8, args), (shared.launch_backward_shared_square_mxfp8_core,
                                                                    core_args))
    for entry, entry_args in entries:
        graph = torch.cuda.CUDAGraph()
        workspaces = {}
        original_empty = torch.empty

        def observe_empty(*positional, **kwargs):
            tensor = original_empty(*positional, **kwargs)
            if tuple(tensor.shape) == (4, 32, 1024, 1024) and tensor.dtype == torch.float8_e4m3fn:
                assert not workspaces, "expected one captured core"
                workspaces["ds"] = tensor
            elif "ds" in workspaces and "dss" not in workspaces:
                assert tuple(tensor.shape) == (4, 32, 32, 32) and tensor.dtype == torch.uint8
                workspaces["dss"] = tensor
            return tensor

        try:
            with torch.cuda.stream(stream):
                check(entry(*entry_args, causal=causal))
                entry(*entry_args, causal=causal)
                stream.synchronize()
                with mock.patch.object(shared.torch, "empty", side_effect=observe_empty):
                    with torch.cuda.graph(graph, stream=stream):
                        captured = entry(*entry_args, causal=causal)
                assert set(workspaces) == {"ds", "dss"}
                for _ in range(2):
                    workspaces["ds"].view(torch.uint8).fill_(0x7f)
                    workspaces["dss"].fill_(255)
                    for grad in captured:
                        grad.fill_(float("nan"))
                    graph.replay()
                    stream.synchronize()
                    check(captured)
                    if causal:
                        assert bool((workspaces["ds"].view(torch.uint8)[0, 0, 128:192, :64] == 0x7f).all())
                        records = workspaces["dss"].view(4, 32, 8, 16, 4, 2)
                        assert bool((records[0, 0, 0, 2, :2, :] == 255).all())
                        assert bool((workspaces["ds"].view(torch.uint8)[0, 0, 64:128, :64] == 0).all())
                        assert bool((records[0, 0, 0, 1, :2, :] != 255).all())
        finally:
            stream.synchronize()
            graph.reset()
            workspaces.clear()
    for actual, expected in zip(args[:-1], frozen_inputs):
        equal(actual, expected)
    for actual, expected in zip(prepared, prepared_before):
        equal(actual, expected)


@pytest.mark.parametrize("bad", ("q_dtype", "k_dtype", "v_dtype", "scale_dtype", "forward_v_scale", "strided", "offset",
                                 "oversized", "cpu", "do_dtype", "out_dtype", "lse_dtype", "causal", "scale"))
def test_flash_attn_mxfp8_shared_fp8_boundary_rejects_metadata(bad):
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    shape = (4, 32, 1024, 128)
    payload = [torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn) for _ in range(3)]
    scales = [torch.empty((4, 32, 1024, 4), device="cuda", dtype=torch.uint8) for _ in range(3)]
    do, out = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    args = [*payload, *scales, do, out, lse, .5]
    causal = False
    if bad in ("q_dtype", "k_dtype", "v_dtype"):
        args[("q_dtype", "k_dtype", "v_dtype").index(bad)] = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    elif bad == "scale_dtype":
        args[5] = scales[2].float()
    elif bad == "forward_v_scale":
        args[5] = torch.empty((4, 32, 32, 128), device="cuda", dtype=torch.uint8)
    elif bad == "strided":
        args[2] = torch.empty((4, 32, 1024, 256), device="cuda", dtype=payload[2].dtype)[..., ::2]
    elif bad in ("offset", "oversized"):
        backing = torch.empty(payload[2].numel() + 1, device="cuda", dtype=payload[2].dtype)
        args[2] = (backing[1:] if bad == "offset" else backing[:-1]).view(shape)
    elif bad == "cpu":
        args[2] = torch.empty(shape, dtype=payload[2].dtype)
    elif bad in ("do_dtype", "out_dtype", "lse_dtype"):
        index = {"do_dtype": 6, "out_dtype": 7, "lse_dtype": 8}[bad]
        args[index] = args[index].to(torch.float16)
    elif bad == "causal":
        causal = 1
    else:
        args[-1] = .7
    with mock.patch.object(shared.torch, "empty", side_effect=AssertionError("invalid input allocated workspace")):
        with pytest.raises(ValueError):
            shared.launch_backward_shared_square_mxfp8(*args, causal=causal)


@pytest.mark.parametrize("bad", ("prep_scale", "prep_do", "prep_causal", "core_do", "core_delta", "core_kdqs"))
def test_flash_attn_mxfp8_shared_fp8_split_rejects_metadata(bad):
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    shape = (4, 32, 1024, 128)
    q8, k8, v8, do8 = [torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn) for _ in range(4)]
    qs, ks, vs, dos = [torch.empty((4, 32, 1024, 4), device="cuda", dtype=torch.uint8) for _ in range(4)]
    do, out = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    delta, lse = [torch.empty(shape[:-1], device="cuda", dtype=torch.float32) for _ in range(2)]
    kdqs = torch.empty((4, 32, 128, 32), device="cuda", dtype=torch.uint8)
    if bad.startswith("prep"):
        entry = shared.prepare_backward_shared_square_mxfp8
        args, causal = [ks, do, out, .5], False
        if bad == "prep_scale":
            args[0] = qs.float()
        elif bad == "prep_do":
            args[1] = do8
        else:
            causal = 1
    else:
        entry = shared.launch_backward_shared_square_mxfp8_core
        args, causal = [q8, k8, v8, qs, ks, vs, do8, dos, delta, kdqs, lse, .5], False
        if bad == "core_do":
            args[6] = do
        elif bad == "core_delta":
            args[8] = delta.to(torch.bfloat16)
        else:
            args[9] = kdqs.transpose(-1, -2)
    with mock.patch.object(shared.torch, "empty", side_effect=AssertionError("invalid input allocated workspace")):
        with pytest.raises(ValueError):
            entry(*args, causal=causal)


def test_flash_attn_mxfp8_gfx950_catalog_buffer_span():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs

    spec, = [spec for spec in CATALOG if (spec.op, spec.arch) == ("flash_attn_mxfp8", "gfx950")]
    assert spec.supports_backward
    check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=128, N_CTX=2**24 - 256)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=128, N_CTX=2**24)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("sequence", (False, True))
@pytest.mark.parametrize("strided", (False, True))
def test_flash_attn_mxfp8_gfx950_quantization_axes(sequence, strided):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_quant import quantize_mxfp8

    torch.manual_seed(731)
    source = torch.randn((2, 3, 96, 256 if strided else 128), device="cuda", dtype=torch.bfloat16)
    x = source[..., ::2] if strided else source
    x[:, :, :32, :32] = 0
    x[:, :, 32:64] *= 2.0**-30
    x[:, :, 64:] *= 2.0**20
    actual, scales = quantize_mxfp8(x, transpose_for_reduction=sequence)
    expected, expected_scales, _ = _mxfp8_reference_quantize(x, sequence=sequence)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("sequence", (False, True))
def test_flash_attn_mxfp8_gfx950_quantization_scale_boundaries(sequence):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_quant import quantize_mxfp8

    maxima = torch.tensor([448 * 2.0**-10, 448, 450, 448 * 2.0**10], device="cuda", dtype=torch.bfloat16)
    x = maxima.repeat_interleave(32).expand(1, 1, 128, 128).contiguous()
    if sequence:
        x = x.transpose(-1, -2).contiguous()
    actual, scales = quantize_mxfp8(x, transpose_for_reduction=sequence)
    expected, expected_scales, _ = _mxfp8_reference_quantize(x, sequence=sequence)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("sequence", (False, True))
@pytest.mark.parametrize("forward", (False, True))
def test_flash_attn_mxfp8_gfx950_quantization_minimum_scale(sequence, forward):
    import triton
    import triton.language as tl
    import triton.language.extra.tlx as tlx
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950 import quantize_mxfp8_head, quantize_mxfp8_v
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_quant import _scale_exponent, quantize_mxfp8
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_bwd_shared import _quantize_ds_square, _scaled_e4m3_words

    # The minimum E8M0 scale is subnormal FP32 (2**-127), not zero.
    # Include adjacent floats at its RCEIL transition, zero, and large values.
    maxima = torch.tensor([
        0, 1, 0x00800000, 0x04000000, 0x04600000, 0x04600001, 0x04600002, 0x04dfffff, 0x04e00000, 0x04e00001, 0x7f7fffff
    ], dtype=torch.int32).view(torch.float32)
    x = maxima[None, :, None, None].expand(1, len(maxima), 32, 128).contiguous()
    x[..., 1::2] *= -1
    # CPU double log/division avoids reference-side log rounding and FTZ.
    exponent = torch.ceil(torch.log2((maxima * (1.0 / 448.0)).double())).clamp(-127, 127)
    scales = torch.exp2(exponent)
    expected = (x.double() / scales[None, :, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    if forward:
        actual, actual_scales = (quantize_mxfp8_v if sequence else quantize_mxfp8_head)(x.cuda())
    else:
        actual, actual_scales = quantize_mxfp8(x.cuda(), transpose_for_reduction=sequence)
    expected_scales = (exponent + 127).to(torch.uint8)[None, :, None, None].expand(actual_scales.shape)
    torch.testing.assert_close(actual_scales.cpu(), expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(actual.cpu().view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)

    @triton.jit
    def shared_quantize(X, Y, S, E, REFERENCE: tl.constexpr):
        tile = tl.program_id(0)
        rows = tl.arange(0, 64)
        cols = tl.arange(0, 64)
        mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 64], transposed=True,
                                                warps_per_cta=[2, 1])
        ds = tl.load(X + tile * 64 * 64 + rows[:, None] * 64 + cols[None, :])
        ds = tlx.release_layout(tlx.require_layout(ds, mma, pin=False))
        if REFERENCE:
            # Keep the original reduction association and integer exponent
            # rule, independently of the shared helper's fused scale bits.
            square = ds.reshape(2, 32, 2, 32)
            amax = tl.max(tl.max(tl.abs(square), 3), 1)
            exponent = _scale_exponent(amax)
            scale_bits = tl.where(exponent == 0, 0x00400000, exponent << 23)
            scale_bits = tl.where(exponent == 255, 0x7fc00000, scale_bits)
            scale = scale_bits.to(tl.float32, bitcast=True)
            pairs = tl.broadcast_to(scale[:, None, :, None], (2, 32, 2, 16))
            payload = _scaled_e4m3_words(ds, pairs.reshape(64, 32))
            scales = tl.broadcast_to(exponent[:, None, :], (2, 32, 2)).reshape(64, 2).to(tl.uint8)
        else:
            payload, scales, exponent = _quantize_ds_square(ds, 64, 64)
        groups = tl.arange(0, 2)
        tl.store(Y + tile * 64 * 64 + rows[:, None] * 64 + cols[None, :], payload.to(tl.uint8, bitcast=True))
        tl.store(S + tile * 64 * 2 + rows[:, None] * 2 + groups[None, :], scales)
        tl.store(E + tile * 4 + groups[:, None] * 2 + groups[None, :], exponent)

    # Exercise the actual square32 dS helper, not just the forward/general
    # quantizers above. Bit patterns preserve subnormals and adjacent floats.
    shared_maxima = torch.tensor([
        0,
        1,
        0x04600000,
        0x04600001,
        0x04600002,
        0x43dfffff,
        0x43e00000,
        0x43e00001,
        0x7f7fffff,
        0x7f800000,
        0x7fc00000,
        0x43e00000,
    ], dtype=torch.int32).view(torch.float32)
    shared_input = shared_maxima[:, None, None].expand(-1, 64, 64).contiguous()
    shared_input[..., 1::2] *= -1
    # Mixed NaN/finite blocks use the same tl.max semantics in both paths;
    # the default reduction does not promise to propagate every input NaN.
    shared_input[-1, ::32, ::32] = float('nan')
    shared_input = shared_input.cuda()
    results = []
    for reference in (False, True):
        payload = torch.empty(shared_input.shape, device="cuda", dtype=torch.uint8)
        scales = torch.empty((len(shared_maxima), 64, 2), device="cuda", dtype=torch.uint8)
        exponents = torch.empty((len(shared_maxima), 2, 2), device="cuda", dtype=torch.int32)
        shared_quantize[(len(shared_maxima), )](shared_input, payload, scales, exponents, reference, num_warps=2,
                                                num_stages=1, matrix_instr_nonkdim=32)
        results.append((payload, scales, exponents))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("n_ctx", (96, 256, 1024))
@pytest.mark.parametrize("transpose_storage", (False, True))
def test_flash_attn_mxfp8_gfx950_fused_quantization(n_ctx, transpose_storage):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_quant import quantize_backward_operands

    torch.manual_seed(631)
    tensors = [torch.randn((1, 2, n_ctx, 128), device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    tensors[0][:, :, :32, :32] = 0
    result = quantize_backward_operands(*tensors, transpose_storage=transpose_storage)
    for (actual, scales), (index, sequence) in zip(result, ((0, True), (1, True), (2, False), (3, False), (3, True))):
        expected, expected_scales, _ = _mxfp8_reference_quantize(tensors[index], sequence=sequence)
        torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)
        if transpose_storage and sequence:
            assert actual.stride(-2) == 1 and actual.stride(-1) == n_ctx


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("scale", (0.0, -0.5, 0.5, 1.3))
def test_flash_attn_mxfp8_gfx950_forward_returns_base2_lse(causal, scale):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950 import (_launch_quantized, quantize_mxfp8_head,
                                                                quantize_mxfp8_v)

    torch.manual_seed(20)
    q, k, v = [torch.randn((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(3)]
    q8, qs = quantize_mxfp8_head(q)
    k8, ks = quantize_mxfp8_head(k)
    v8, vs = quantize_mxfp8_v(v)
    _, lse = _launch_quantized(q8, k8, v8, qs, ks, vs, causal, scale, return_lse=True)
    q_ref = q8.float() * torch.exp2(qs.float() - 127).repeat_interleave(32, -1)
    k_ref = k8.float() * torch.exp2(ks.float() - 127).repeat_interleave(32, -1)
    scores = (q_ref @ k_ref.transpose(-1, -2)) * scale
    if causal:
        row = torch.arange(256, device=q.device)
        scores.masked_fill_(row[:, None] < row[None, :], -float("inf"))
    expected = torch.logsumexp(scores, -1) * 1.4426950408889634
    torch.testing.assert_close(lse, expected, atol=3e-4, rtol=1e-5)


@pytest.mark.skipif(not _has_two_gfx950_devices(), reason="requires two gfx950 GPUs")
def test_flash_attn_mxfp8_gfx950_uses_input_device_when_current_device_differs():
    previous_device, query_device = _gfx950_device_indices()[:2]
    script = r"""
import sys
import torch
from triton.tlx.ops import flash_attn_mxfp8
previous, target = map(int, sys.argv[1:])
torch.cuda.set_device(previous)
q, k, v = [(torch.randn((1, 2, 256, 128), device=f"cuda:{target}", dtype=torch.bfloat16) * .5)
           .requires_grad_() for _ in range(3)]
out = flash_attn_mxfp8(q, k, v)
assert out.device == q.device and torch.cuda.current_device() == previous
out.backward(torch.randn_like(out))
assert torch.cuda.current_device() == previous
assert all(x.grad.device == q.device and bool(x.grad.isfinite().all()) for x in (q, k, v))
"""
    subprocess.run([sys.executable, "-c", script, str(previous_device), str(query_device)], check=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("n_ctx", (96, 256))
def test_flash_attn_mxfp8_gfx950_backward_matches_recipe(causal, n_ctx):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_bwd import launch_backward

    torch.manual_seed(20)
    q, k, v, do = [torch.randn((1, 2, n_ctx, 128), device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(4)]
    q8, qs, qf = _mxfp8_reference_quantize(q)
    k8, ks, kf = _mxfp8_reference_quantize(k)
    qdk, qdks, qdkf = _mxfp8_reference_quantize(q, sequence=True)
    kdq, kdqs, kdqf = _mxfp8_reference_quantize(k, sequence=True)
    vb, vs, vf = _mxfp8_reference_quantize(v)
    do8, dos, dof = _mxfp8_reference_quantize(do)
    dodv, dodvs, dodvf = _mxfp8_reference_quantize(do, sequence=True)
    sm_scale = 0.5
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        scores = (qf @ kf.transpose(-1, -2)) * sm_scale
        if causal:
            row = torch.arange(n_ctx, device=q.device)
            scores.masked_fill_(row[:, None] < row[None, :], -float("inf"))
        lse = torch.logsumexp(scores, -1) * 1.4426950408889634
        p = scores.softmax(-1)
        out = (p @ vf).to(torch.bfloat16)
        delta = (out.float() * do.float()).sum(-1)
        dp = dof @ vf.transpose(-1, -2)
        _, _, ds = _mxfp8_reference_quantize(p * (dp - delta[..., None]), square=True)
        p_quant = (p * 256).to(torch.float8_e4m3fn).float() / 256
        expected = (ds @ kdqf * sm_scale, ds.transpose(-1, -2) @ qdkf * sm_scale, p_quant.transpose(-1, -2) @ dodvf)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    dq = torch.empty_like(q, dtype=torch.float32)
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    kernel = launch_backward(do8, dodv, q8, qdk, k8, kdq, vb, out, lse, qs, qdks, ks, kdqs, vs, dos, dodvs, sm_scale,
                             do, dq, dk, dv, torch.empty_like(lse), causal=causal, block_m=64, block_n=128)
    assert "v_mfma_scale_f32" in kernel.asm["amdgcn"]
    for actual, reference in zip((dq, dk, dv), expected):
        assert torch.isfinite(actual).all()
        # BF16 output rounding and near-tie FP8 dS rounding can change a few
        # elements; bound total error, not just gradient direction.
        relative_rms = ((actual.float() - reference).square().mean() / reference.square().mean()).sqrt()
        assert relative_rms < 0.01


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("n_ctx", (1024, 2048, 4096, 8192))
@pytest.mark.parametrize("scale", (128**-0.5, 0.5, 1.3))
@pytest.mark.parametrize("seed", (20, 21))
def test_flash_attn_mxfp8_gfx950_backward_meta_shapes(causal, n_ctx, scale, seed):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(seed)
    q, k, v = [(torch.randn((4, 32, n_ctx, 128), device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_()
               for _ in range(3)]
    rq, rk, rv = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    do = torch.randn_like(q)
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    reference = torch.nn.functional.scaled_dot_product_attention(rq, rk, rv, is_causal=causal, scale=scale)
    out.backward(do)
    reference.backward(do)
    if scale <= 0.5:
        torch.testing.assert_close(out, reference, atol=0.2, rtol=0)
    # At scale 1.3, Q/K FP8 quantization changes sharp softmax peaks: the
    # decoded-input FP32 reference also exceeds 0.2 at isolated elements.
    # Bound direction and magnitude against BF16 for every scale and tensor.
    for actual, expected in ((out, reference), (q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        assert actual.dtype == torch.bfloat16
        assert torch.isfinite(actual).all()
        actual, expected = actual.float(), expected.float()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        relative_rms = ((actual - expected).square().mean() / expected.square().mean()).sqrt()
        assert cosine >= 0.98
        assert relative_rms < 0.15


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_mxfp8_gfx950_backward_zero_blocks(causal):
    from triton.tlx.ops import flash_attn_mxfp8

    q, k, v = [torch.zeros((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    out = flash_attn_mxfp8(q, k, v, causal=causal)
    out.backward(torch.zeros_like(out))
    for tensor in (out, q.grad, k.grad, v.grad):
        torch.testing.assert_close(tensor, torch.zeros_like(tensor), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("scale", (0.0, -0.5))
@pytest.mark.parametrize("requires_grad", (False, True))
def test_flash_attn_mxfp8_gfx950_nonpositive_scale(causal, scale, requires_grad):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(21)
    q, k, v = [(torch.randn((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_(requires_grad)
               for _ in range(3)]
    rq, rk, rv = [x.detach().float().requires_grad_(requires_grad) for x in (q, k, v)]
    # Some SDPA implementations return NaN for negative scale; use explicit
    # FP32 softmax attention so the test does not bless a broken reference.
    scores = (rq @ rk.transpose(-1, -2)) * scale
    if causal:
        row = torch.arange(256, device=q.device)
        scores = scores.masked_fill(row[:, None] < row[None, :], -float("inf"))
    reference = scores.softmax(-1) @ rv
    actual = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale)
    torch.testing.assert_close(actual.float(), reference, atol=0.2, rtol=0)
    if requires_grad:
        do = torch.randn_like(actual)
        actual.backward(do)
        reference.backward(do.float())
        for got, ref in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
            assert torch.isfinite(got).all()
            relative_rms = ((got.float() - ref).square().mean() / ref.square().mean().clamp_min(1e-30)).sqrt()
            assert relative_rms < 0.15


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("shape,scale,match", [
    ((0, 1, 256, 128), 0.5, "nonempty"),
    ((1, 0, 256, 128), 0.5, "nonempty"),
    ((1, 1, 0, 128), 0.5, "nonempty"),
    ((1, 1, 256, 128), float("nan"), "finite"),
    ((1, 1, 256, 128), float("inf"), "finite"),
    ((1, 1, 256, 128), -float("inf"), "finite"),
])
def test_flash_attn_mxfp8_gfx950_invalid_inputs(shape, scale, match):
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q, k, v = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    with pytest.raises(InvalidInput, match=match):
        flash_attn_mxfp8(q, k, v, sm_scale=scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("warn_only", (False, True))
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_mxfp8_gfx950_determinism(warn_only, causal):
    from triton.tlx.ops import flash_attn_mxfp8

    q, k, v = [(torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_()
               for _ in range(3)]
    out = flash_attn_mxfp8(q, k, v, causal=causal)
    do = torch.randn_like(out)
    enabled = torch.are_deterministic_algorithms_enabled()
    warned = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        expected = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
        for _ in range(3):
            actual = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
            for got, ref in zip(actual, expected):
                torch.testing.assert_close(got, ref, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warned)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("kind", ("head", "v", "v_transposed"))
def test_flash_attn_mxfp8_gfx950_quantization_large_head_offset_codegen(kind):
    import re
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950 import (_quantize_mxfp8_kernel, _quantize_mxfp8_v_kernel)

    x = torch.empty((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((1, 1, 64, 4), device="cuda", dtype=torch.uint8)
    # Compile only: head 2's source offset would be 2**31 elements. Dummy
    # tensors stay tiny and the deliberately oversized strides are never run.
    args = (x, y, scales, 2**30, 2**30, 128, 1, 3, 64)
    if kind == "head":
        kernel = _quantize_mxfp8_kernel.warmup(*args, HEAD_DIM=128, BLOCK_N=64, PACK_K=False, num_warps=4, grid=(1, 3))
    else:
        kernel = _quantize_mxfp8_v_kernel.warmup(*args, HEAD_DIM=128, BLOCK_N=64, TRANSPOSED=kind == "v_transposed",
                                                 num_warps=1, grid=(1, 3))
    ir = kernel.asm["ttir"]
    pid = re.search(r"(%[\w]+) = tt.get_program_id y : i32", ir).group(1)
    widened = re.search(rf"(%[\w]+) = arith.extsi {re.escape(pid)} : i32 to i64", ir)
    assert widened is not None, "widen head index before multiplying tensor strides"
    assert re.search(r"arith.muli .* : i64", ir)
