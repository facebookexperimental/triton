"""Public ``tlx.ops.qk_norm_rope_kv_cache`` coverage for gfx950."""

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedBackward, qk_norm_rope_kv_cache
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.fused.qk_norm_rope_kv_cache._shapes import (CORRECTNESS_SHAPES, QKNormRopeShape,
                                                                        head_sum_squares, inputs, label, reference)

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
PRODUCTION = (16, 2, 64)


@triton.jit
def _rsqrt_kernel(X, Y, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + offs, tl.rsqrt(tl.load(X + offs, mask=offs < N)), mask=offs < N)


def hw_rsqrt(x):
    """The hardware reciprocal square root the op uses."""
    x = x.contiguous()
    y = torch.empty_like(x)
    _rsqrt_kernel[(triton.cdiv(x.numel(), 1024), )](x, y, x.numel(), BLOCK=1024)
    return y


def _bits(x):
    return x.view(torch.uint8) if x.element_size() == 1 else x.view(torch.int16)


def _exact_reference(args, rope_bf16=True):
    return reference(**args, rope_bf16=rope_bf16, rsqrt=hw_rsqrt, sum_squares=head_sum_squares)


def _run(args, **kw):
    """The op on clones of the caches; returns ``(q_out, k_cache, v_cache)``."""
    args = dict(args, k_cache=args["k_cache"].clone(), v_cache=args["v_cache"].clone(), **kw)
    q = qk_norm_rope_kv_cache(**args)
    return q, args["k_cache"], args["v_cache"]


def _assert_equal(got, ref):
    for g, r, name in zip(got, ref, ("q", "k_cache", "v_cache")):
        assert g.dtype == r.dtype and g.shape == r.shape, name
        diff = int((_bits(g) != _bits(r)).sum())
        assert diff == 0, f"{name}: {diff} of {g.numel()} differ"


def _assert_close(got, ref, max_fraction):
    """Equal up to one rounding step of the output dtype, in at most ``max_fraction`` of the elements."""
    for g, r, name in zip(got, ref, ("q", "k_cache", "v_cache")):
        assert g.dtype == r.dtype and g.shape == r.shape, name
        differ = _bits(g) != _bits(r)
        assert differ.float().mean() <= max_fraction, name
        gi, ri = _bits(g).int(), _bits(r).int()
        assert ((gi - ri).abs()[differ] <= 1).all(), f"{name}: more than one rounding step apart"


def test_catalog_entry():
    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("qk_norm_rope_kv_cache", "gfx950")]
    assert len(specs) == 1
    spec, = specs
    assert spec.dtypes == frozenset({"bfloat16"})
    assert not spec.supports_backward
    assert spec.accepts({"HQ": 16, "HK": 2, "D": 64}) and spec.accepts({"HQ": 16, "HK": 16, "D": 256})
    for dims in ({"HQ": 12, "HK": 2, "D": 64}, {"HQ": 16, "HK": 2, "D": 96}, {"HQ": 16, "HK": 2, "D":
                                                                              8}, {"HQ": 32, "HK": 2, "D": 256}):
        assert not spec.accepts(dims)


@requires_gfx950
@pytest.mark.parametrize("shape", [s for s in CORRECTNESS_SHAPES if s[1:4] == PRODUCTION], ids=label)
def test_production_shapes_bitwise(shape):
    args = inputs(shape)
    _assert_equal(_run(args), _exact_reference(args))


@requires_gfx950
@pytest.mark.parametrize("shape", [s for s in CORRECTNESS_SHAPES if s[1:4] != PRODUCTION], ids=label)
def test_other_head_shapes(shape):
    # The per-head sums use the compiler's order at these shapes: rstd may differ in the last bit.
    args = inputs(shape)
    _assert_close(_run(args), reference(**args, rsqrt=hw_rsqrt), max_fraction=0.01)


@requires_gfx950
@pytest.mark.parametrize("T", [1, 2, 5, 64, 1000])
@pytest.mark.parametrize("rope,remap,q_fp8", [(True, True, True), (True, False, False), (False, True, False),
                                              (False, False, True)])
def test_token_counts_and_modes(T, rope, remap, q_fp8):
    args = inputs(QKNormRopeShape(T, *PRODUCTION, rope, remap, q_fp8), seed=T)
    _assert_equal(_run(args), _exact_reference(args))


@requires_gfx950
@pytest.mark.parametrize("rope,q_fp8", [(True, True), (False, False)])
def test_head_sums_in_fixed_order(rope, q_fp8):
    # Elements spread over 2**-12 .. 2**12 make the per-head sums depend on the summation order.
    args = inputs(QKNormRopeShape(4096, *PRODUCTION, rope, False, q_fp8), seed=5)
    gen = torch.Generator(device="cuda").manual_seed(5)
    for name in ("q", "k"):
        x = args[name]
        x.mul_(torch.exp2(torch.randint(-12, 13, x.shape, device="cuda", generator=gen)).to(x.dtype))
    _assert_equal(_run(args), _exact_reference(args))


@requires_gfx950
@pytest.mark.parametrize("k_scale,v_scale", [(0.5, 2.0), (0.25, 0.125)])
def test_power_of_two_scales(k_scale, v_scale):
    args = inputs(QKNormRopeShape(300, *PRODUCTION, True, True, True), k_scale=k_scale, v_scale=v_scale)
    _assert_equal(_run(args), _exact_reference(args))


@requires_gfx950
def test_other_scales():
    # The FP8 q uses the hardware reciprocal of q_out_scale; the reference divides exactly.
    args = inputs(QKNormRopeShape(300, *PRODUCTION, True, False, True), k_scale=0.37, v_scale=1.7)
    _assert_close(_run(args), _exact_reference(args), max_fraction=0.01)


@requires_gfx950
@pytest.mark.parametrize("rope,remap,q_fp8", [(True, True, False), (True, False, True), (False, True, True)])
def test_without_kv_scales(rope, remap, q_fp8):
    # The default: no k_scale and v_scale, so K and V are converted unscaled. Dividing by one is exact, so the
    # result equals the one with unit scales.
    args = inputs(QKNormRopeShape(300, *PRODUCTION, rope, remap, q_fp8))
    unscaled = dict(args, k_scale=None, v_scale=None)
    got = _run(unscaled)
    _assert_equal(got, _exact_reference(unscaled))
    _assert_equal(got, _run(args))


@requires_gfx950
def test_rope_fp32():
    # One rounding per RoPE output; the kernel may fuse the products into FMAs.
    args = inputs(QKNormRopeShape(300, *PRODUCTION, True, False, False))
    got = _run(args, rope_bf16=False)
    _assert_close(got, _exact_reference(args, rope_bf16=False), max_fraction=0.01)
    assert (_bits(got[0]) != _bits(_run(args)[0])).any()  # differs from the BF16 RoPE


@requires_gfx950
def test_saturation():
    args = inputs(QKNormRopeShape(64, *PRODUCTION, True, False, True))
    v = args["v"]
    v[:, :4] = torch.tensor([500.0, -1e4, 448.0, -464.0], device="cuda", dtype=torch.bfloat16)
    got = _run(args)
    _assert_equal(got, _exact_reference(args))
    s = args["slot_mapping"][0]
    row = _bits(got[2][s // 16, s % 16].flatten()[:4]).tolist()
    assert row == [0x7E, 0xFE, 0x7E, 0xFE]


@requires_gfx950
def test_q_fp8_saturates_and_maps_nan_to_min():
    # q * 64 passes 448; a NaN head of q turns into -448, as a clamp with max(x, -448) first does.
    args = inputs(QKNormRopeShape(64, *PRODUCTION, True, False, True), k_scale=2.0**-6, v_scale=2.0**-6)
    args["q"][3, 64:128] = float("nan")
    got = _run(args)
    _assert_equal(got, _exact_reference(args))
    q = got[0].float()
    assert (q == 448).any() and (q == -448).any()
    assert (_bits(got[0][3, 64:128]) == 0xFE).all()


@requires_gfx950
def test_skipped_slots_leave_cache_untouched():
    args = inputs(QKNormRopeShape(100, *PRODUCTION, True, True, False))
    args["slot_mapping"][:10] = -1
    sentinel = torch.full_like(args["k_cache"], 3.0)
    args = dict(args, k_cache=sentinel, v_cache=sentinel.clone())
    got = _run(args)
    _assert_equal(got, _exact_reference(args))
    slots = args["slot_mapping"].long()
    slots = torch.where(slots >= 0, args["slot_remap"].long()[slots.clamp_min(0)], slots)
    written = torch.zeros(args["k_cache"].shape[:2], dtype=torch.bool, device="cuda").flatten()
    written[slots[slots >= 0]] = True
    assert (slots < 0).sum() > 10  # padding and the remap's -1 entries
    for cache in got[1:]:
        assert (cache.flatten(0, 1)[~written].float() == 3.0).all()


@requires_gfx950
def test_int32_indices_and_q_out():
    args = inputs(QKNormRopeShape(77, *PRODUCTION, True, True, True))
    ref = _exact_reference(args)
    q_out = torch.empty(77, 1024, device="cuda", dtype=torch.float8_e4m3fn)
    got = _run(args, positions=args["positions"].int(), slot_mapping=args["slot_mapping"].int(),
               slot_remap=args["slot_remap"].int(), q_out=q_out)
    assert got[0].data_ptr() == q_out.data_ptr()
    _assert_equal(got, ref)


@requires_gfx950
def test_row_strided_q_and_cache_block_stride():
    args = inputs(QKNormRopeShape(50, *PRODUCTION, True, False, False))
    ref = _exact_reference(args)
    wide = torch.zeros(50, 4096, device="cuda", dtype=torch.bfloat16)
    wide[:, 8:8 + 1024] = args["q"]
    # Blocks padded to twice their size: block stride 2 * 16 * 2 * 64.
    padded = lambda: torch.zeros(args["k_cache"].shape[0], 2, 16, 2, 64, device="cuda",  # noqa: E731
                                 dtype=torch.float8_e4m3fn)[:, 0]
    k_cache, v_cache = padded(), padded()
    q = qk_norm_rope_kv_cache(**dict(args, q=wide[:, 8:8 + 1024], k_cache=k_cache, v_cache=v_cache))
    _assert_equal((q, k_cache, v_cache), ref)


@requires_gfx950
def test_batch_invariance():
    args = inputs(QKNormRopeShape(4096, *PRODUCTION, True, True, False))
    full = _run(args)[0]
    for lo, hi in ((0, 1), (1000, 1333)):
        part = dict(args, q=args["q"][lo:hi], k=args["k"][lo:hi], v=args["v"][lo:hi],
                    slot_mapping=args["slot_mapping"][lo:hi], positions=args["positions"][lo:hi])
        assert torch.equal(_bits(_run(part)[0]), _bits(full[lo:hi]))


@requires_gfx950
def test_zero_tokens():
    args = inputs(QKNormRopeShape(3, *PRODUCTION, True, False, True))
    args = dict(args, q=args["q"][:0], k=args["k"][:0], v=args["v"][:0], slot_mapping=args["slot_mapping"][:0],
                positions=args["positions"][:0])
    q, k_cache, _ = _run(args)
    assert q.shape == (0, 1024) and q.dtype == torch.float8_e4m3fn
    assert torch.equal(_bits(k_cache), _bits(args["k_cache"]))


@requires_gfx950
def test_graph_replay_with_new_inputs():
    first = inputs(QKNormRopeShape(16, *PRODUCTION, True, True, True), seed=1)
    second = inputs(QKNormRopeShape(16, *PRODUCTION, True, True, True), seed=2)
    args = dict(first, k_cache=first["k_cache"].clone(), v_cache=first["v_cache"].clone())
    q_out = torch.empty(16, 1024, device="cuda", dtype=torch.float8_e4m3fn)
    qk_norm_rope_kv_cache(**args, q_out=q_out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qk_norm_rope_kv_cache(**args, q_out=q_out)
    for name in ("q", "k", "v", "positions", "slot_mapping", "slot_remap"):
        args[name].copy_(second[name])
    args["k_cache"].zero_()
    args["v_cache"].zero_()
    graph.replay()
    _assert_equal((q_out, args["k_cache"], args["v_cache"]), _exact_reference(second))


@requires_gfx950
def test_torch_compile():
    args = inputs(QKNormRopeShape(40, *PRODUCTION, True, True, True))
    ref = _exact_reference(args)
    k_cache, v_cache = args["k_cache"].clone(), args["v_cache"].clone()

    @torch.compile
    def fn(q, k, v, k_cache, v_cache, slot_mapping, positions, cos_sin_cache, slot_remap, scale):
        return qk_norm_rope_kv_cache(q, k, v, k_cache, v_cache, slot_mapping, eps=args["eps"], q_scale=args["q_scale"],
                                     positions=positions, cos_sin_cache=cos_sin_cache, slot_remap=slot_remap,
                                     k_scale=scale, v_scale=scale, q_out_scale=scale)

    q = fn(args["q"], args["k"], args["v"], k_cache, v_cache, args["slot_mapping"], args["positions"],
           args["cos_sin_cache"], args["slot_remap"], args["k_scale"])
    _assert_equal((q, k_cache, v_cache), ref)


@requires_gfx950
def test_invalid_inputs():
    args = inputs(QKNormRopeShape(8, *PRODUCTION, True, True, True))
    bad = [
        dict(q=args["q"].float()),
        dict(k=args["k"][:, :64]),
        dict(v=args["v"][:4]),
        dict(q=args["q"].t().contiguous().t()),
        dict(k_cache=args["k_cache"].to(torch.bfloat16)),
        dict(v_cache=args["v_cache"][:2]),
        dict(k_cache=args["k_cache"].transpose(1, 2)),
        dict(slot_mapping=args["slot_mapping"].float()),
        dict(slot_mapping=args["slot_mapping"][:4]),
        dict(slot_remap=args["slot_remap"].view(2, -1)),
        dict(positions=None),
        dict(cos_sin_cache=args["cos_sin_cache"][:, :32]),
        dict(cos_sin_cache=args["cos_sin_cache"].bfloat16()),
        dict(positions=args["positions"].float()),
        dict(v_scale=None),
        dict(k_scale=torch.ones(2, device="cuda")),
        dict(q_out_scale=torch.ones(1, device="cuda", dtype=torch.float16)),
        dict(eps=torch.tensor(1e-5)),
        dict(q_out=torch.empty(8, 1024, device="cuda", dtype=torch.bfloat16)),
        dict(q_out=args["q"]),
        dict(space="full"),
    ]
    for override in bad:
        with pytest.raises(InvalidInput):
            qk_norm_rope_kv_cache(**dict(args, **override))
    # Envelope: 32 query heads of 256 is past one program's tile.
    with pytest.raises(InvalidInput):
        qk_norm_rope_kv_cache(**inputs(QKNormRopeShape(2, 32, 2, 256, False, False, False)))
    # Outputs must not alias the inputs or each other.
    q = args["q"].contiguous()
    with pytest.raises(InvalidInput):
        qk_norm_rope_kv_cache(**dict(args, q=q, q_out=q.view(-1).view(torch.float8_e4m3fn)[:8 * 1024].view(8, 1024)))
    with pytest.raises(InvalidInput):
        qk_norm_rope_kv_cache(**dict(args, q_out=args["k_cache"].view(-1)[:8 * 1024].view(8, 1024)))
    with pytest.raises(InvalidInput):
        qk_norm_rope_kv_cache(**dict(args, v_cache=args["k_cache"]))


@requires_gfx950
def test_autograd_rejected():
    args = inputs(QKNormRopeShape(8, *PRODUCTION, True, False, False))
    with pytest.raises(UnsupportedBackward):
        qk_norm_rope_kv_cache(**dict(args, q=args["q"].clone().requires_grad_()))
    q_out = torch.empty(8, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(UnsupportedBackward):
        qk_norm_rope_kv_cache(**args, q_out=q_out)
    with torch.no_grad():
        qk_norm_rope_kv_cache(**args, q_out=q_out)
