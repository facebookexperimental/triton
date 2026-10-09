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


def _mxfp8_saved_square_inputs(out, *, whole=False):
    """Test adapter; the separate-tensor public API still requires whole storage."""
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    saved = out.grad_fn.saved_tensors
    if len(saved) == 6:
        q, k, v, arena, saved_out, lse = saved
        assert out.grad_fn.saved_format == shared._SAVED_QK_ARENA_FORMAT
        shared._check_saved_qk_arena(arena, q.device, q.shape[2])
        q8, k8, qs, ks = shared._saved_qk_arena_views(arena, q.shape[2])
        if whole:
            q8, k8, qs, ks = (tensor.clone() for tensor in (q8, k8, qs, ks))
        return q, k, v, q8, k8, saved_out, lse, qs, ks
    assert out.grad_fn.saved_format is None and len(saved) == 9
    return saved


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
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950, gfx950_bwd, gfx950_bwd_shared

    torch.manual_seed(20)
    inputs = _qkv((4, 32, n_ctx, 128), requires_grad=True)
    do = torch.randn_like(inputs[0]) * 0.5
    out = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale)
    reference = _sdpa(*inputs, causal, scale)
    expected = torch.autograd.grad(reference, inputs, do)
    original_general = gfx950_bwd.launch_backward
    _, _, v, q8, k8, saved_out, lse, qs, ks = _mxfp8_saved_square_inputs(out, whole=True)
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
        validator = ("_check_shared_square_qk_arena_inputs" if n_ctx in (1024, 2048) else "_check_shared_square_inputs")
        with mock.patch.object(gfx950_bwd_shared, validator, wraps=getattr(gfx950_bwd_shared, validator)) as validate:
            with mock.patch.object(gfx950_bwd_shared, "_is_gfx950", wraps=gfx950_bwd_shared._is_gfx950) as arch:
                eager = backward()
        assert validate.call_count == arch.call_count == 1
        check(eager)
        if n_ctx in (1024, 2048):
            # Independently force the original forward and nonarena BN64
            # backward. Check all four saved byte regions and O/LSE first.
            with mock.patch.object(gfx950, "_PUBLIC_QK_ARENA_LENGTHS", ()):
                original_out = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale)
            original_saved = _mxfp8_saved_square_inputs(original_out)
            packed_saved = _mxfp8_saved_square_inputs(out)
            assert len(out.grad_fn.saved_tensors) == 6
            assert len(original_out.grad_fn.saved_tensors) == 9
            for index in (3, 4, 5, 6, 7, 8):
                assert torch.equal(packed_saved[index].view(torch.uint8), original_saved[index].view(torch.uint8))
            with mock.patch.object(gfx950_bwd_shared, "_PREPARATION_ARENA_BYTES", {}):
                original_stores = torch.autograd.grad(original_out, inputs, do, retain_graph=True)
            for actual, original in zip(eager, original_stores):
                assert torch.equal(actual.view(torch.uint8), original.view(torch.uint8))
            del original_out, original_saved, packed_saved, original_stores

            # Hooks must own all six real saved tensors, including the arena.
            packed_shapes = []

            def pack(tensor):
                packed_shapes.append((tuple(tensor.shape), tensor.dtype))
                return tensor.detach().clone()

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor.clone()):
                hooked_out = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale)
            assert len(packed_shapes) == 6
            assert packed_shapes[3] == ((gfx950_bwd_shared._saved_qk_arena_layout(n_ctx)[-1], ), torch.uint8)
            hooked = torch.autograd.grad(hooked_out, inputs, do, retain_graph=True)
            for actual, expected_bytes in zip(hooked, eager):
                assert torch.equal(actual.view(torch.uint8), expected_bytes.view(torch.uint8))
            del hooked_out, hooked

            versioned_out = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale)
            with torch.no_grad():
                versioned_out.grad_fn.saved_tensors[3][0].add_(1)
            with pytest.raises(RuntimeError, match="modified by an inplace operation"):
                torch.autograd.grad(versioned_out, inputs, do)
            del versioned_out

            def offset_hook(tensor):
                # Restore a valid dense view with natural, not 16-byte,
                # alignment. The whole uint8 Q/K arena stays strict.
                if tensor.dtype == torch.uint8 or tensor.dtype == torch.float8_e4m3fn:
                    return tensor.clone()
                storage = torch.empty(tensor.numel() + 3, device=tensor.device, dtype=tensor.dtype)
                restored = storage[1:1 + tensor.numel()].view_as(tensor)
                restored.copy_(tensor)
                return restored

            with torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor.detach().clone(), offset_hook):
                fallback_out = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale)
                with mock.patch.object(gfx950, "_PUBLIC_QK_ARENA_LENGTHS", ()):
                    fallback_original = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale)
            with mock.patch.object(gfx950_bwd, "launch_backward", new=original_general):
                restored = torch.autograd.grad(fallback_out, inputs, do, retain_graph=True)
                original_restored = torch.autograd.grad(fallback_original, inputs, do, retain_graph=True)
            for actual, original in zip(restored, original_restored):
                assert torch.equal(actual.view(torch.uint8), original.view(torch.uint8))
            del fallback_out, fallback_original, restored, original_restored

            def invalid_arena_hook(tensor):
                if tensor.dtype != torch.uint8 or tensor.ndim != 1:
                    return tensor.clone()
                storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
                storage[1:].copy_(tensor)
                return storage[1:]

            with torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor.detach().clone(), invalid_arena_hook):
                malformed_out = flash_attn_mxfp8(*inputs, causal=causal, sm_scale=scale)
            with mock.patch.object(gfx950_bwd_shared, "_saved_qk_arena_views",
                                   side_effect=AssertionError("invalid arena reached views")):
                with pytest.raises(ValueError, match="arena"):
                    torch.autograd.grad(malformed_out, inputs, do)
            del malformed_out

            streams = (torch.cuda.Stream(), torch.cuda.Stream())
            concurrent = []
            for current in streams:
                current.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(current):
                    leaves = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
                    concurrent_out = flash_attn_mxfp8(*leaves, causal=causal, sm_scale=scale)
                    grads = torch.autograd.grad(concurrent_out, leaves, do, retain_graph=True)
                    concurrent.append((concurrent_out, grads))
            for current in streams:
                current.synchronize()
            arenas = [current_out.grad_fn.saved_tensors[3] for current_out, _ in concurrent]
            assert arenas[0].data_ptr() != arenas[1].data_ptr()
            for _, grads in concurrent:
                for actual, original in zip(grads, eager):
                    assert torch.equal(actual.view(torch.uint8), original.view(torch.uint8))
            del concurrent, arenas
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        workspaces = {}
        original_empty = torch.empty
        arena_bytes = gfx950_bwd_shared._BACKWARD_ARENA_BYTES.get(n_ctx)

        def observe_empty(*allocation_args, **kwargs):
            tensor = original_empty(*allocation_args, **kwargs)
            if tuple(tensor.shape) == (arena_bytes, ) and tensor.dtype == torch.uint8:
                assert "arena" not in workspaces, "expected one private backward arena"
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
                assert set(workspaces) == ({"arena"} if n_ctx in (1024, 2048) else {"ds", "dss"})
                if n_ctx in (1024, 2048):
                    ds_offset, dss_offset, total_bytes = gfx950_bwd_shared._backward_arena_layout(n_ctx)
                    # Test-only views expose the JIT pointer regions for the
                    # causal untouched-region checks below.
                    workspaces["ds"] = workspaces["arena"].narrow(0, ds_offset, dss_offset - ds_offset).view(
                        torch.float8_e4m3fn).view(4, 32, n_ctx, n_ctx)
                    workspaces["dss"] = workspaces["arena"].narrow(0, dss_offset, total_bytes - dss_offset).view(
                        4, 32, n_ctx // 32, n_ctx // 32)
                for _ in range(2):
                    # Poison before each replay, outside the captured graph.
                    # E4M3 byte0x7f and E8M0 byte255 represent NaN. Every
                    # consumed workspace value must come from this replay.
                    # Also poison both private payloads, their scales, and
                    # FP32 Delta. Views retain this storage for graph replay.
                    if n_ctx in (1024, 2048):
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
    packed = n_ctx in (1024, 2048)
    preparation_allocations = 1 if packed else 6
    original_empty = torch.empty
    allocations = []

    def allocate(allocation_shape, *, dtype, device):
        # Materialize temporary storage on CPU to check byte-region bounds.
        # Only region endpoints are touched; output metadata can use meta.
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
    assert len(allocations) == (4 if packed else 11)
    args = prepare.call_args.args
    expected_shapes = (shape, shape, (4, 32, n_ctx, 4), (4, 32, n_ctx, 4), (4, 32, 128, n_ctx // 32), shape[:-1])
    expected_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.uint8, torch.uint8, torch.uint8, torch.float32)
    if packed:
        arena = allocations[0]
        assert args[4] is kv.call_args.args[5] is query.call_args.args[1] is arena
        assert kv.call_args.kwargs["ARENA_N"] == query.call_args.kwargs["ARENA_N"] == n_ctx
        assert arena.dtype == torch.uint8 and arena.ndim == 1 and arena.storage_offset() == 0
        offsets = (*shared._preparation_arena_layout(n_ctx)[:-1], *shared._backward_arena_layout(n_ctx))
        expected_shapes += ((4, 32, n_ctx, n_ctx), (4, 32, n_ctx // 32, n_ctx // 32))
        expected_dtypes += (torch.float8_e4m3fn, torch.uint8)
        assert offsets[-1] == arena.numel()
        # These test-only views model the JIT's typed pointer segments.
        private = [
            arena.narrow(0, start, end - start).view(dtype).view(expected_shape)
            for start, end, dtype, expected_shape in zip(offsets, offsets[1:], expected_dtypes, expected_shapes)
        ]
        with pytest.raises(AssertionError):
            shared._preparation_arena_layout(4096)
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
    assert offset == 34816 * n_ctx + (128 * n_ctx * n_ctx + 128 * (n_ctx // 32)**2 if packed else 0)
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


@pytest.mark.parametrize(
    "n_ctx,qk_format,plan_kind",
    [(n, fmt, "ordinary") for n in (1024, 2048) for fmt in (False, True)]
    + [(1024, True, "inline"), (1024, True, "partial")],
)
def test_flash_attn_mxfp8_shared_backward_launch_plan_invalidation(n_ctx, qk_format, plan_kind):
    from contextlib import ExitStack
    from types import SimpleNamespace
    from triton import knobs
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    device = torch.device("cuda:2")
    globals_dict = {"value": 7}
    names = {
        "ordinary": ("_prepare_fused_arena", "_bwd_kv_owner_arena", "_bwd_q_consume_arena"),
        "inline": ("_bwd_kv_owner_inline_arena", "_bwd_q_consume_arena"),
        "partial": ("_prepare_do_arena", "_bwd_kv_owner_partial_arena", "_bwd_q_consume_arena"),
    }[plan_kind]
    jits = [
        SimpleNamespace(device_caches={}, pre_run_hooks=[], launch_metadata=None, debug=False, hash="source",
                        used_global_vals={("value", 0): (7, globals_dict)}) for _ in names
    ]
    kernels = [mock.MagicMock(spec=shared.CompiledKernel) for _ in names]
    for jit, kernel in zip(jits, kernels):
        kernel.src = SimpleNamespace(fn=jit)
        kernel._compile_iq_acf_cubin = None
        jit.device_caches[device.index] = ({"compiled-key": kernel}, {}, None, None, None)

    def remember(fmt=qk_format):
        if plan_kind == "ordinary":
            shared._remember_arena_launch_plan(device, n_ctx, False, 0.5, kernels, fmt)
        else:
            getattr(shared, "_remember_" + plan_kind + "_arena_launch_plan")(device, n_ctx, 0.5, kernels)

    def plan(selected_device=device, n=n_ctx, scale=0.5, fmt=qk_format):
        if plan_kind == "ordinary":
            return shared._arena_launch_plan(selected_device, n, False, scale, fmt)
        return getattr(shared, "_" + plan_kind + "_arena_launch_plan")(selected_device, n, scale)

    with ExitStack() as stack:
        for name, jit in zip(names, jits):
            stack.enter_context(mock.patch.object(shared, name, jit))
        stack.enter_context(mock.patch.object(shared, "_ARENA_LAUNCH_PLANS", {}))
        controls = stack.enter_context(mock.patch.object(shared, "_arena_launch_controls", return_value=True))
        remember()
        if plan_kind == "ordinary":
            assert plan(fmt=not qk_format) is None
            remember(not qk_format)
            assert len(shared._ARENA_LAUNCH_PLANS) == 2
            assert shared._arena_launch_plan(device, n_ctx, True, 0.5, qk_format) is None
        else:
            other = "partial" if plan_kind == "inline" else "inline"
            assert getattr(shared, "_" + other + "_arena_launch_plan")(device, n_ctx, 0.5) is None
        expected = tuple(kernel.__getitem__.return_value for kernel in kernels)
        assert plan() == expected
        assert plan(selected_device=torch.device("cuda:3")) is None
        assert plan(scale=1.3) is None
        controls.return_value = False
        assert plan() is None
        controls.return_value = True
        for jit in jits:
            jit.pre_run_hooks.append(lambda: None)
            assert plan() is None
            jit.pre_run_hooks.clear()
            jit.hash = "changed source"
            assert plan() is None
            jit.hash = "source"
            old_cache = jit.device_caches[device.index]
            jit.device_caches.clear()
            assert plan() is None
            jit.device_caches[device.index] = old_cache
        globals_dict["value"] = 8
        assert plan() is None
        globals_dict["value"] = 7
        assert plan() == expected
        if plan_kind == "partial":
            key = (device.index, n_ctx, False, 0.5, ("partial_saved_qk_arena", 1))
            original = shared._ARENA_LAUNCH_PLANS[key]
            shared._ARENA_LAUNCH_PLANS[key] = original[:2]
            assert plan() is None
            shared._ARENA_LAUNCH_PLANS[key] = original
            replacement = SimpleNamespace(**vars(jits[1]))
            with mock.patch.object(shared, "_bwd_kv_owner_partial_arena", replacement):
                assert plan() is None

    # Controls are tested independently of the mocked validity metadata.
    with mock.patch.object(shared, "get_cache_invalidating_env_vars", return_value={}):
        with mock.patch.dict("os.environ", {}, clear=True):
            assert shared._arena_launch_controls()
            with knobs.compilation.scope():
                knobs.compilation.always_compile = True
                assert not shared._arena_launch_controls()
            with knobs.runtime.scope():
                knobs.runtime.debug = True
                assert not shared._arena_launch_controls()
            with knobs.runtime.scope():
                knobs.runtime.jit_cache_hook = lambda **kwargs: None
                assert not shared._arena_launch_controls()
            with knobs.compilation.scope():
                knobs.compilation.instrumentation_mode = "test instrumentation"
                assert not shared._arena_launch_controls()


@pytest.mark.parametrize("n_ctx", (1024, 2048))
@pytest.mark.parametrize("qk_format", (False, True))
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_mxfp8_shared_backward_launch_plan_fresh_arguments(n_ctx, qk_format, causal):
    from contextlib import ExitStack
    from types import SimpleNamespace
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    device = torch.device("cuda:2")
    shape = (4, 32, n_ctx, 128)
    inputs = [torch.empty(shape, device="meta") for _ in range(8)]
    if qk_format:
        inputs[:4] = [
            torch.empty(shared._saved_qk_arena_layout(n_ctx)[-1], device="meta", dtype=torch.uint8), None, None, None
        ]
    partial_prepare = qk_format and n_ctx in (1024, 2048) and not causal
    jits = ((shared._prepare_do_arena, shared._bwd_kv_owner_partial_arena, shared._bwd_q_consume_arena) if partial_prepare else
            (shared._prepare_fused_arena, shared._bwd_kv_owner_arena, shared._bwd_q_consume_arena))
    # Populate real transitive source hashes without compiling or using a GPU.
    assert all(jit.cache_key for jit in jits)
    kernels = [mock.MagicMock(spec=shared.CompiledKernel) for _ in jits]
    original_empty = torch.empty
    allocations = []

    def allocate(allocation_shape, *, dtype, device):
        tensor = original_empty(allocation_shape, dtype=dtype, device="meta")
        allocations.append(tensor)
        return tensor

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(shared, "_arena_launch_controls", return_value=True))
        stack.enter_context(mock.patch.object(shared, "_ARENA_LAUNCH_PLANS", {}))
        stack.enter_context(mock.patch.object(shared.torch, "empty", side_effect=allocate))
        stack.enter_context(
            mock.patch.object(shared.driver, "_active", SimpleNamespace(get_current_stream=lambda device: 101)))
        runs = []
        for jit, kernel in zip(jits, kernels):
            kernel.src = SimpleNamespace(fn=jit)
            kernel._compile_iq_acf_cubin = None
            stack.enter_context(mock.patch.object(jit, "device_caches", {2: ({"compiled-key": kernel}, )}))
            runs.append(stack.enter_context(mock.patch.object(jit, "run", return_value=kernel)))
        cold = shared._launch_backward_shared_square(*inputs, 0.5, causal, device, shape, qk_format)
        warm = shared._launch_backward_shared_square(*inputs, 0.5, causal, device, shape, qk_format)
        assert all(run.call_count == 1 for run in runs)
        assert all(first is not second for first, second in zip(cold, warm))
        kv_index = 1
        kv_args = kernels[kv_index].__getitem__.return_value.call_args.args
        query_args = kernels[-1].__getitem__.return_value.call_args.args
        assert query_args[0] is inputs[0 if qk_format else 1]
        assert query_args[2] is warm[0]
        assert query_args[5:] == (n_ctx, 128, 128, 64, causal, 128, qk_format)
        assert kv_args[0] is inputs[0]
        if partial_prepare:
            # Cold and cached launches bind the runtime N ABI to ARENA_N.
            cold_kv_args = runs[kv_index].call_args.args
            cold_arena_n = runs[kv_index].call_args.kwargs["ARENA_N"]
            assert cold_kv_args[6] == cold_arena_n == n_ctx
            assert kv_args[6] == kv_args[8] == n_ctx
            prepare_args = kernels[0].__getitem__.return_value.call_args.args
            assert prepare_args[0] is inputs[5] and prepare_args[1] is inputs[6]
            assert prepare_args[2] is kv_args[3] is query_args[1] is allocations[4]
            assert prepare_args[3:] == (n_ctx, 128, 32)
            assert kv_args[1] is inputs[4] and kv_args[2] is inputs[7]
            assert kv_args[4] is warm[1] and kv_args[5] is warm[2]
            assert kv_args[8:] == (n_ctx, 128, 64, 64)
            arguments = (prepare_args, kv_args, query_args)
            expected_tensors = 12
        else:
            # Cold and cached launches bind the runtime N ABI to ARENA_N.
            cold_kv_args = runs[kv_index].call_args.args
            cold_arena_n = runs[kv_index].call_args.kwargs["ARENA_N"]
            assert cold_kv_args[8] == cold_arena_n == n_ctx
            assert kv_args[8] == kv_args[10] == n_ctx
            assert kv_args[5] is query_args[1] is allocations[4]
            assert kv_args[6] is warm[1] and kv_args[7] is warm[2]
            prepare_args = kernels[0].__getitem__.return_value.call_args.args
            assert prepare_args[4] is kv_args[5]
            assert prepare_args[5:] == (n_ctx, 128, 32, qk_format)
            assert kv_args[10:] == (n_ctx, 128, 64, 64, causal, qk_format)
            arguments = (prepare_args, kv_args, query_args)
            expected_tensors = 13 if qk_format else 16
            if qk_format:
                assert kv_args[1:4] == (None, None, None)
                assert prepare_args[2] is query_args[0] is inputs[0]
        assert sum(isinstance(arg, torch.Tensor) for args in arguments for arg in args) == expected_tensors
        assert all(kernel.__getitem__.call_count == 1 for kernel in kernels)
        assert all(kernel.__getitem__.return_value.call_args.kwargs == {"stream": 101} for kernel in kernels)
        # A supported source edit invalidates each caller's hash. The live
        # real JIT objects must reject a retained compiled plan afterward.
        with mock.patch.object(jits[kv_index], "hash", None):
            if partial_prepare:
                assert shared._partial_arena_launch_plan(device, n_ctx, 0.5) is None
            else:
                assert shared._arena_launch_plan(device, n_ctx, causal, 0.5, qk_format) is None
        kernels[kv_index].__getitem__.return_value.side_effect = RuntimeError("compiled launch failed")
        with pytest.raises(RuntimeError, match="compiled launch failed"):
            shared._launch_backward_shared_square(*inputs, 0.5, causal, device, shape, qk_format)
        assert all(run.call_count == 1 for run in runs), "compiled launch errors must not retry JIT"


@pytest.mark.parametrize("qk_format", (False, True))
def test_flash_attn_mxfp8_shared_backward_runner_plan_current_stream(qk_format):
    """Use actual public compiled runners with CPU-modeled launch handles."""
    from contextlib import ExitStack
    from types import SimpleNamespace
    from triton.backends.compiler import GPUTarget
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    shape = (4, 32, 1024, 128)
    device = torch.device("cuda:2")
    inputs = [torch.empty(shape, device="meta") for _ in range(8)]
    if qk_format:
        inputs[:4] = [
            torch.empty(shared._saved_qk_arena_layout(1024)[-1], device="meta", dtype=torch.uint8), None, None, None
        ]
    jits = ((shared._prepare_do_arena, shared._bwd_kv_owner_partial_arena, shared._bwd_q_consume_arena) if qk_format else
            (shared._prepare_fused_arena, shared._bwd_kv_owner_arena, shared._bwd_q_consume_arena))
    kernels = []
    for jit in jits:
        kernel = shared.CompiledKernel.__new__(shared.CompiledKernel)
        kernel.module = object()
        kernel._unload_module = None
        kernel._init_handles = lambda: None
        kernel._dispatcher = None
        kernel._run = mock.Mock()
        kernel.function, kernel.packed_metadata = object(), ()
        kernel.src = SimpleNamespace(fn=jit)
        kernel.metadata = SimpleNamespace(target=GPUTarget("hip", "gfx950", 64))
        kernel.name, kernel.metadata_group, kernel.hash = "modeled", {}, "modeled"
        kernels.append(kernel)
    active = SimpleNamespace(stream=101, get_current_stream=mock.Mock(side_effect=lambda device: active.stream))
    original_empty = torch.empty

    def allocate(allocation_shape, *, dtype, device):
        return original_empty(allocation_shape, dtype=dtype, device="meta")

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(shared, "_arena_launch_controls", return_value=True))
        stack.enter_context(mock.patch.object(shared, "_ARENA_LAUNCH_PLANS", {}))
        stack.enter_context(mock.patch.object(shared.driver, "_active", active))
        stack.enter_context(mock.patch.object(shared.torch, "empty", side_effect=allocate))
        for jit, kernel in zip(jits, kernels):
            stack.enter_context(mock.patch.object(jit, "device_caches", {2: ({"compiled-key": kernel}, )}))
            stack.enter_context(mock.patch.object(jit, "run", return_value=kernel))
        shared._launch_backward_shared_square(*inputs, 0.5, False, device, shape, qk_format)
        first = shared._launch_backward_shared_square(*inputs, 0.5, False, device, shape, qk_format)
        active.stream = 202
        second = shared._launch_backward_shared_square(*inputs, 0.5, False, device, shape, qk_format)
        assert active.get_current_stream.call_count == 2
        assert all(kernel._run.call_args_list[0].args[3] == 101 for kernel in kernels)
        assert all(kernel._run.call_args_list[1].args[3] == 202 for kernel in kernels)
        assert all(a is not b for a, b in zip(first, second))
        arena_index = 11 if qk_format else 13
        assert kernels[0]._run.call_args_list[0].args[arena_index] is not kernels[0]._run.call_args_list[1].args[
            arena_index]
        kernels[0 if qk_format else 1]._run.side_effect = RuntimeError("modeled runner failure")
        with pytest.raises(RuntimeError, match="modeled runner failure"):
            shared._launch_backward_shared_square(*inputs, 0.5, False, device, shape, qk_format)
        assert kernels[-1]._run.call_count == 2, "a failed producer must not launch its consumer"


def test_flash_attn_mxfp8_shared_backward_compiled_runner_current_stream():
    from types import SimpleNamespace
    from triton import knobs
    from triton.runtime import driver
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    active = SimpleNamespace(get_current_device=lambda: 2, get_current_stream=lambda device: active.stream, stream=101)
    kernel = SimpleNamespace(_init_handles=lambda: None, _dispatcher=None, launch_metadata=lambda *args: None,
                             run=mock.Mock(), function=object(), packed_metadata=())
    with mock.patch.object(driver, "_active", active):
        with knobs.nvidia.scope():
            knobs.nvidia.use_triton_dispatcher = False
            runner = shared.CompiledKernel.__getitem__(kernel, (32, 128, 1))
            runner("first call")
            active.stream = 202
            runner("second call")
    assert kernel.run.call_args_list[0].args[3] == 101
    assert kernel.run.call_args_list[1].args[3] == 202


@pytest.mark.parametrize("n_ctx", (1024, 2048))
@pytest.mark.parametrize("offset", (1, 8))
def test_flash_attn_mxfp8_public_qk_arena_admission(n_ctx, offset):
    """Real CPU storage proves strict arena and general dense-view contracts."""
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared

    shape = (4, 32, n_ctx, 128)
    elements = 128 * n_ctx * 128
    arena = torch.empty(shared._saved_qk_arena_layout(n_ctx)[-1], device="cpu", dtype=torch.uint8)
    whole = torch.empty(shape, device="cpu", dtype=torch.bfloat16)
    whole_lse = torch.empty(shape[:-1], device="cpu", dtype=torch.float32)
    view = torch.empty(elements + offset + 3, device="cpu", dtype=torch.bfloat16)[offset:offset + elements].view(shape)
    view_lse = torch.empty(128 * n_ctx + offset + 3, device="cpu",
                           dtype=torch.float32)[offset:offset + 128 * n_ctx].view(shape[:-1])
    device = torch.device("cuda:2")
    with mock.patch.object(torch.Tensor, "device", property(lambda tensor: device)):
        assert shared._check_saved_qk_backward_tensors(arena, whole, whole, whole, whole_lse) == (device, shape)
        with pytest.raises(ValueError, match="metadata"):
            shared._check_saved_qk_backward_tensors(arena, view, view, view, view_lse)
        assert shared._check_saved_qk_backward_tensors(arena, view, view, view, view_lse,
                                                       allow_views=True) == (device, shape)
        bad_arena = torch.empty(arena.numel() + 1, device="cpu", dtype=torch.uint8)[1:]
        for allow_views in (False, True):
            with pytest.raises(ValueError, match="arena"):
                shared._check_saved_qk_backward_tensors(bad_arena, whole, whole, whole, whole_lse,
                                                        allow_views=allow_views)
        with mock.patch.object(shared, "_is_gfx950", return_value=True):
            with mock.patch.object(shared, "_launch_backward_shared_square", return_value="launched") as launch:
                assert shared._try_launch_backward_shared_square(arena, None, None, None, whole, whole, whole,
                                                                 whole_lse, 0.5) == "launched"
                assert launch.call_args.kwargs == {"qk_format": True}
                for slots in ((None, whole, None), (whole, None, None), (None, None, whole)):
                    assert shared._try_launch_backward_shared_square(arena, *slots, whole, whole, whole, whole_lse,
                                                                     0.5) is None
                launch.side_effect = RuntimeError("allocation failed")
                with pytest.raises(RuntimeError, match="allocation failed"):
                    shared._try_launch_backward_shared_square(arena, None, None, None, whole, whole, whole, whole_lse,
                                                              0.5)
        regions = shared._saved_qk_arena_views(arena, n_ctx)
        offsets = shared._saved_qk_arena_layout(n_ctx)
        for index, region in enumerate(regions):
            assert region.data_ptr() == arena.data_ptr() + offsets[index]
            byte_view = region.view(torch.uint8).reshape(-1)
            byte_view[0] = byte_view[-1] = index + 1
        for index, region in enumerate(regions):
            byte_view = region.view(torch.uint8).reshape(-1)
            assert int(byte_view[0]) == int(byte_view[-1]) == index + 1


def test_flash_attn_mxfp8_public_qk_arena_fallback_validation():
    """Fallback checks precede view derivation and preserve valid offset inputs."""
    from contextlib import ExitStack, nullcontext
    from types import SimpleNamespace
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950, gfx950_bwd, gfx950_bwd_shared as shared, gfx950_quant

    n_ctx = 1024
    shape = (4, 32, n_ctx, 128)
    device = torch.device("cuda:2")
    q = torch.empty(shape, device="cpu", dtype=torch.bfloat16)
    view = torch.empty(q.numel() + 3, device="cpu", dtype=q.dtype)[1:1 + q.numel()].view(shape)
    lse = torch.empty(128 * n_ctx + 3, device="cpu", dtype=torch.float32)[1:1 + 128 * n_ctx].view(shape[:-1])
    arena = torch.empty(shared._saved_qk_arena_layout(n_ctx)[-1], device="cpu", dtype=torch.uint8)
    ctx = SimpleNamespace(saved_format=shared._SAVED_QK_ARENA_FORMAT, saved_tensors=(q, q, view, arena, view, lse),
                          sm_scale=0.5, causal=False)
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(torch.Tensor, "device", property(lambda tensor: device)))
        stack.enter_context(mock.patch.object(torch.cuda, "device", lambda device: nullcontext()))
        stack.enter_context(mock.patch.object(shared, "_try_launch_backward_shared_square", return_value=None))
        views = stack.enter_context(
            mock.patch.object(shared, "_saved_qk_arena_views", wraps=shared._saved_qk_arena_views))
        quantize = stack.enter_context(
            mock.patch.object(gfx950_quant, "quantize_backward_operands", return_value=((object(), object()), ) * 5))
        launch = stack.enter_context(mock.patch.object(gfx950_bwd, "launch_backward"))
        gradients = gfx950._MXFP8Attention.backward(ctx, view)
        assert len(gradients) == 5 and gradients[-2:] == (None, None)
        assert views.call_count == quantize.call_count == launch.call_count == 1
        assert quantize.call_args.args[:3] == (q, q, view)
        saved = ctx.saved_tensors
        for bad_index, malformed in ((3, arena[1:]), (4, view.transpose(-1, -2)), (5, lse.to(torch.bfloat16))):
            changed = list(saved)
            changed[bad_index] = malformed
            ctx.saved_tensors = tuple(changed)
            with pytest.raises(ValueError):
                gfx950._MXFP8Attention.backward(ctx, view)
            assert views.call_count == quantize.call_count == launch.call_count == 1
        ctx.saved_tensors = saved
        launch.side_effect = RuntimeError("general launch failed")
        with pytest.raises(RuntimeError, match="general launch failed"):
            gfx950._MXFP8Attention.backward(ctx, view)
        ctx.saved_format = ("unknown", 99)
        with pytest.raises(ValueError, match="Unknown"):
            gfx950._MXFP8Attention.backward(ctx, view)


def test_flash_attn_mxfp8_public_qk_arena_live_helpers():
    """Changes to live producer controls retain the original quantizer route."""
    from contextlib import ExitStack
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950, gfx950_bwd_shared as shared

    inputs = tuple(torch.empty((4, 32, 1024, 128), device="meta", dtype=torch.bfloat16) for _ in range(3))
    device = torch.device("cuda:2")
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(torch.Tensor, "device", property(lambda tensor: device)))
        stack.enter_context(mock.patch.object(shared, "_is_gfx950", return_value=True))
        controls = stack.enter_context(mock.patch.object(shared, "_arena_launch_controls", return_value=True))

        def selected():
            return gfx950._can_save_public_qk_arena(*inputs, 0.5, False)

        assert selected()
        controls.return_value = False
        assert not selected()
        controls.return_value = True
        for helper_name, expected in gfx950._PUBLIC_QK_FORWARD_HELPERS:
            helper = getattr(gfx950, helper_name)
            with mock.patch.object(gfx950, helper_name, lambda *args, **kwargs: None):
                assert not selected()
            code, defaults, kwdefaults = helper.__code__, helper.__defaults__, helper.__kwdefaults__
            try:
                helper.__code__ = (lambda *args: None).__code__
                assert not selected()
            finally:
                helper.__code__ = code
            try:
                helper.__defaults__ = ("changed", )
                assert not selected()
            finally:
                helper.__defaults__ = defaults
            keywords = dict(expected[3])
            keywords[0] = False  # Legal Python metadata with incomparable key types.
            try:
                helper.__kwdefaults__ = keywords
                assert not selected()
            finally:
                helper.__kwdefaults__ = kwdefaults
            assert selected()
        for name, jit, _ in gfx950._PUBLIC_QK_JIT_HELPERS:
            for attribute, changed in (("pre_run_hooks", [lambda: None]), ("launch_metadata", lambda *args: None),
                                       ("debug", True), ("_src", jit.src + "\n# changed")):
                with mock.patch.object(jit, attribute, changed):
                    assert not selected()
            assert getattr(gfx950, name) is jit
        with mock.patch.object(gfx950, "_FP8", torch.float8_e5m2):
            assert not selected()
        assert not gfx950._can_save_public_qk_arena(*inputs, 0.5, 1)
        assert not gfx950._can_save_public_qk_arena(*inputs, 0.7, False)

        class Subclass(torch.Tensor):
            pass

        assert not gfx950._can_save_public_qk_arena(inputs[0].as_subclass(Subclass), *inputs[1:], 0.5, False)


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
    _, _, v, q8, k8, saved_out, lse, qs, ks = _mxfp8_saved_square_inputs(out, whole=True)
    # Backward V uses feature32, not forward V's sequence32 quantization.
    v8, vs = quantize_mxfp8(v, transpose_for_reduction=False)
    args = (q8, k8, v8, qs, ks, vs, do, saved_out, lse, .5)
    return inputs, out, args


@pytest.mark.parametrize("fixture", ("random", "boundary", "extreme"))
def test_flash_attn_mxfp8_shared_inline_prepare_matches_fused(fixture):
    """Compare inline preparation bytes with the unchanged fused producer."""
    import triton
    import triton.language as tl
    import triton.language.extra.tlx as tlx
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950_bwd_shared as shared
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_bwd_shared import (
        _inline_prepare_do32,
        _inline_prepare_v64,
        _store_inline_kdqs,
    )

    @triton.jit
    def inline_prepare_probe(V, DO, O, KS, VB, DO8, VS, DOS, KDQS, Delta, N: tl.constexpr, D: tl.constexpr):
        tl.static_assert(N == 64 and D == 128)
        head = tl.program_id(0).to(tl.int64)
        base = head * N * D
        rows = tl.arange(0, 64)
        features = tl.arange(0, D)
        groups = tl.arange(0, 4)
        value, vscale = _inline_prepare_v64(V, base, rows, D)
        do0, delta0, word0 = _inline_prepare_do32(DO, O, base, 0, D)
        do1, delta1, word1 = _inline_prepare_do32(DO, O, base, 32, D)
        offsets = base + rows[:, None] * D + features[None, :]
        tl.store(VB + offsets, tlx.release_layout(value))
        tl.store(DO8 + offsets, tl.cat(do0, do1, dim=0))
        scale_offsets = (head * N + rows[:, None]) * 4 + groups[None, :]
        tl.store(VS + scale_offsets, tlx.release_layout(vscale).to(tl.uint8))
        scale0 = tl.broadcast_to(((word0 >> (groups * 8)) & 255)[None, :], (32, 4))
        scale1 = tl.broadcast_to(((word1 >> (groups * 8)) & 255)[None, :], (32, 4))
        tl.store(DOS + scale_offsets, tl.cat(scale0, scale1, dim=0).to(tl.uint8))
        tl.store(Delta + head * N + rows, tl.cat(delta0, delta1, dim=0))
        _store_inline_kdqs(KS, KDQS, head, 0, N)

    torch.manual_seed(20)
    shape = (2, 3, 64, 128)
    v, do, out = [(torch.randn(shape, device="cuda") * .5).to(torch.bfloat16) for _ in range(3)]
    if fixture == "boundary":
        # Give each feature32 group a separate scale-transition maximum.
        maxima = torch.tensor([0., .8671875, .875, .87890625], device="cuda", dtype=torch.bfloat16)
        signs = torch.tensor([1., -1.], device="cuda", dtype=torch.bfloat16).repeat(64)
        row = maxima.repeat_interleave(32) * signs
        do = row[None, None, None, :].expand(shape).contiguous()
        do[:, :, 32:] = do[:, :, 32:].flip(-1)
        factors = torch.tensor([2**-8, 1., 2**8, 2**16], device="cuda", dtype=torch.bfloat16)
        v = (do * factors.repeat_interleave(32)).contiguous()
    elif fixture == "extreme":
        # Include signed zero, BF16 subnormals, product underflow, and overflow.
        bits = torch.tensor([
            0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0,
            0xffc0, 0x3e80, 0xbe80
        ], device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        do = bits.repeat(4)[None, None, :, None].expand(shape).contiguous()
        out = do.clone()
        v = do.flip(-2).contiguous()
    ks = torch.randint(115, 135, (2, 3, 2, 4), device="cuda", dtype=torch.uint8).repeat_interleave(32, -2)
    inputs = (v, do, out, ks)
    frozen = tuple(tensor.clone() for tensor in inputs)
    expected = (torch.empty_like(v, dtype=torch.float8_e4m3fn), torch.empty_like(do, dtype=torch.float8_e4m3fn),
                torch.empty_like(ks), torch.empty_like(ks),
                torch.empty((2, 3, 128, 2), device=v.device,
                            dtype=torch.uint8), torch.empty(shape[:-1], device=v.device, dtype=torch.float32))
    vb, do8, vs, dos, kdqs, delta = expected
    shared._prepare_fused[(2, 6)](v, do, ks, vb, do8, vs, dos, kdqs, out, delta, 64, 128, 32, PACKED_STORES=True,
                                  num_warps=4, num_stages=2)
    actual = tuple(torch.empty_like(tensor) for tensor in expected)

    def probe():
        inline_prepare_probe[(6, )](v, do, out, ks, *actual, 64, 128, num_warps=2, num_stages=1,
                                    matrix_instr_nonkdim=32)

    def equal_bytes(left, right):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))

    probe()
    for tensor, reference in zip(actual, expected):
        equal_bytes(tensor, reference)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            probe()
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                probe()
            for _ in range(2):
                for tensor in actual:
                    tensor.view(torch.uint8).fill_(255)
                graph.replay()
                stream.synchronize()
                for tensor, reference in zip(actual, expected):
                    equal_bytes(tensor, reference)
    finally:
        stream.synchronize()
        graph.reset()
    for tensor, original in zip(inputs, frozen):
        equal_bytes(tensor, original)

    # Exercise the actual N1024 producer with all existing numerical fixtures.
    partial_inputs = tuple(tensor.repeat(1, 1, 16, 1) for tensor in (v, do, out, ks))
    partial_frozen = tuple(tensor.clone() for tensor in partial_inputs)
    partial_v, partial_do, partial_out, partial_ks = partial_inputs
    arena_bytes = shared._PREPARATION_ARENA_BYTES[1024]
    expected_arena = torch.empty((arena_bytes, ), device=v.device, dtype=torch.uint8)
    actual_arena = torch.empty_like(expected_arena)
    shared._prepare_fused_arena[(32, 6)](partial_v, partial_do, partial_ks, partial_out, expected_arena,
                                         1024, 128, 32, QK_FORMAT=False, num_warps=4, num_stages=2)
    offsets = shared._preparation_arena_layout(1024)
    selected = ((1, 6 * 1024 * 128), (3, 6 * 1024 * 4), (5, 6 * 1024 * 4))

    def partial_probe():
        shared._prepare_do_arena[(32, 6)](partial_do, partial_out, actual_arena, 1024, 128, 32,
                                           num_warps=2, num_stages=1)

    def check_partial():
        for index, length in selected:
            equal_bytes(actual_arena.narrow(0, offsets[index], length),
                        expected_arena.narrow(0, offsets[index], length))
        for index in (0, 2, 4):
            assert bool((actual_arena.narrow(0, offsets[index], 16) == 255).all())

    actual_arena.fill_(255)
    partial_probe()
    check_partial()
    partial_stream = torch.cuda.Stream()
    partial_stream.wait_stream(torch.cuda.current_stream())
    partial_graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(partial_stream):
            partial_probe()
            partial_stream.synchronize()
            with torch.cuda.graph(partial_graph, stream=partial_stream):
                partial_probe()
            for _ in range(2):
                actual_arena.fill_(255)
                partial_graph.replay()
                partial_stream.synchronize()
                check_partial()
    finally:
        partial_stream.synchronize()
        partial_graph.reset()
    for tensor, original in zip(partial_inputs, partial_frozen):
        equal_bytes(tensor, original)


@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_mxfp8_shared_fp8_boundary_graph(causal):
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import gfx950, gfx950_bwd_shared as shared

    # Compile and execute the exact new public Q/K wrappers on zero squares
    # and adjacent BF16 values around the RCEIL scale boundary at 448/512.
    boundary = torch.tensor([0., 0., .8671875, .875, .87890625, -.875, -.87890625, 2**-15], device="cuda",
                            dtype=torch.bfloat16).repeat(4 * 32 * 1024 * 128 // 8).view(4, 32, 1024, 128)
    boundary[:, :, :32, :32] = 0
    saved_arena = torch.empty(shared._saved_qk_arena_layout(1024)[-1], device=boundary.device, dtype=torch.uint8)
    for key, operand in ((False, boundary), (True, boundary.flip(-1).contiguous())):
        gfx950._quantize_saved_qk_arena[(16, 128)](operand, saved_arena, *operand.stride(), ARENA_N=1024, KEY=key,
                                                   num_warps=4)
        old_payload, old_scale = gfx950.quantize_mxfp8_head(operand)
        regions = shared._saved_qk_arena_views(saved_arena, 1024)
        payload, exponent = (regions[1], regions[3]) if key else (regions[0], regions[2])
        assert torch.equal(payload.view(torch.uint8), old_payload.view(torch.uint8))
        assert torch.equal(exponent, old_scale)
    del boundary, saved_arena, regions, payload, exponent, old_payload, old_scale

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
    # Check every byte written by the arena's packed producer against the
    # original quantization and reduction, including its scale metadata.
    original_prep = (vb, do8, old_vs, dos, kdqs, delta)
    packed_prep = tuple(torch.empty_like(tensor) for tensor in original_prep)
    pvb, pdo8, pvs, pdos, pkdqs, pdelta = packed_prep
    shared._prepare_fused[(32, 128)](inputs[2], do, ks, pvb, pdo8, pvs, pdos, pkdqs, saved_out, pdelta, 1024, 128, 32,
                                     PACKED_STORES=True, num_warps=4, num_stages=2)
    for actual, expected in zip(packed_prep, original_prep):
        equal(actual, expected)
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
