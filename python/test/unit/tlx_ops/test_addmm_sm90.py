"""L1 correctness for the Hopper ``tlx.ops.addmm`` backend."""

import pytest
import torch

from triton._internal_testing import is_hopper
from triton.tlx.ops.kernels.addmm import sm90
from triton.tlx.ops.kernels.addmm._shapes import CORRECTNESS_SHAPES, inputs

pytestmark = pytest.mark.skipif(not is_hopper(), reason="requires an sm90 GPU")


@pytest.mark.parametrize("m,n,k,a_strides,b_strides,bias_strides,dtype_name", CORRECTNESS_SHAPES)
def test_addmm(m, n, k, a_strides, b_strides, bias_strides, dtype_name):
    from triton.tlx.ops import addmm as tlx_addmm

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    torch.manual_seed(m + n + k)
    bias, a, b = inputs((m, n, k, a_strides, b_strides, bias_strides, dtype_name), dtype)
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    actual = tlx_addmm(bias, a, b, out=out)
    expected = torch.addmm(bias, a, b)

    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)

    del bias, a, b, out, actual, expected
    torch.cuda.empty_cache()


@pytest.mark.parametrize("bias_shape", [(37, ), (1, 37), (23, 1), (23, 37)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_generic_addmm_broadcast(bias_shape, dtype):
    from triton.tlx.ops import addmm as tlx_addmm

    torch.manual_seed(0)
    a = torch.randn((23, 19), device="cuda", dtype=dtype)
    b = torch.randn((19, 37), device="cuda", dtype=dtype)
    bias = torch.randn(bias_shape, device="cuda", dtype=dtype)
    actual = tlx_addmm(bias, a, b, space="heuristic")
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("bias_2d", [False, True])
@pytest.mark.parametrize("b_col_major", [False, True])
def test_addmm_2cta_paths(bias_2d, b_col_major):
    """Exercise the 2-CTA cluster paths: multicast, solo/ghost, split pairs.

    19 x 7 tiles over 132 CTAs strands pair (0, 1) at 2 vs 1 tiles (solo +
    ghost) and the odd last M-group splits pairs across N-tiles; the rest
    run plain multicast. Autotune benchmarks every surviving smoke config,
    so the 2-CTA kernel runs during tuning. The prune assertion pins the
    2-CTA config as surviving so this coverage cannot silently vanish.
    The column-major arm routes B through the transposed descriptor.
    """
    from triton.tlx.ops import addmm as tlx_addmm

    m, n, k = 2432, 896, 768
    assert not sm90._use_direct_path(m, n, k)
    kept = sm90._ws_prune_configs(sm90._ws_smoke_configs(), {"M": m, "N": n, "K": k})
    assert any(c.kwargs["NUM_CTAS"] == 2 for c in kept)

    torch.manual_seed(m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    if b_col_major:
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).t()
        assert b.stride(0) == 1
    else:
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n, ), device="cuda", dtype=torch.bfloat16)
    if bias_2d:
        bias = bias.unsqueeze(0).expand(m, n).contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    assert sm90._tma_eligible(a, b, out, n, k)
    actual = tlx_addmm(bias, a, b, out=out, space="smoke")
    assert actual is out
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)

    del bias, a, b, actual, expected
    torch.cuda.empty_cache()


@pytest.mark.parametrize("bias_2d", [False, True])
@pytest.mark.parametrize("b_col_major", [False, True])
def test_addmm_epilogue_partition(bias_2d, b_col_major):
    """Exercise the partitioned epilogue: handoff rings + 2-replica stores.

    Same shape as the 2-CTA test so tuning artifacts are shared. Autotune
    benchmarks every surviving smoke config, so the partitioned kernel
    (plain, subtile, and 2-CTA-combo) runs during tuning. The prune
    assertions pin those configs as surviving so this coverage cannot
    silently vanish.
    """
    from triton.tlx.ops import addmm as tlx_addmm

    m, n, k = 2432, 896, 768
    assert not sm90._use_direct_path(m, n, k)
    kept = sm90._ws_prune_configs(sm90._ws_smoke_configs(), {"M": m, "N": n, "K": k})
    assert any(c.kwargs["EPILOGUE_PARTITION"] for c in kept)
    assert any(c.kwargs["EPILOGUE_PARTITION"] and c.kwargs["EPILOGUE_SUBTILE"] for c in kept)
    assert any(c.kwargs["EPILOGUE_PARTITION"] and c.kwargs["NUM_CTAS"] == 2 for c in kept)

    torch.manual_seed(m + n + k + 1)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    if b_col_major:
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).t()
        assert b.stride(0) == 1
    else:
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n, ), device="cuda", dtype=torch.bfloat16)
    if bias_2d:
        bias = bias.unsqueeze(0).expand(m, n).contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    assert sm90._tma_eligible(a, b, out, n, k)
    actual = tlx_addmm(bias, a, b, out=out, space="smoke")
    assert actual is out
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)

    del bias, a, b, actual, expected
    torch.cuda.empty_cache()


def test_addmm_heuristic_2cta_space():
    """The ``heuristic_2cta`` space pins the heuristic tile to two CTAs."""
    from triton.tlx.ops import addmm as tlx_addmm

    cfg = sm90._ws_heuristic_2cta_config(2432, 896, 256)[0]
    assert cfg.kwargs["NUM_CTAS"] == 2
    assert cfg.kwargs["BM"] == sm90._ws_heuristic_config(2432, 896, 256)[0].kwargs["BM"]
    # Few tiles: the twin is pruned, so the space degrades to one CTA
    # instead of handing autotune an empty config list.
    cfg = sm90._ws_heuristic_2cta_config(64, 64, 64)[0]
    assert cfg.kwargs["NUM_CTAS"] == 1

    torch.manual_seed(0)
    m, n, k = 2432, 896, 768
    assert not sm90._use_direct_path(m, n, k)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n, ), device="cuda", dtype=torch.bfloat16)
    actual = tlx_addmm(bias, a, b, space="heuristic_2cta")
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)

    del bias, a, b, actual, expected
    torch.cuda.empty_cache()


def test_addmm_heuristic_routes():
    """Pin the data-fit heuristic routing (pure config selection, no GPU)."""
    # Large-M s4 pocket: K>=512 with N>=1024 takes partitioned s4 ...
    kw = sm90._ws_heuristic_config(21347, 1024, 768)[0].kwargs
    assert (kw["BM"], kw["BN"], kw["BK"], kw["NUM_STAGES"]) == (128, 128, 64, 4)
    assert kw["EPILOGUE_PARTITION"] is True
    # ... as does K<512 large-M ...
    kw = sm90._ws_heuristic_config(20480, 768, 256)[0].kwargs
    assert (kw["BK"], kw["NUM_STAGES"]) == (64, 4)
    assert kw["EPILOGUE_PARTITION"] is True
    # ... while N<=768 with K>=512 keeps fused BK=128.
    kw = sm90._ws_heuristic_config(160853, 768, 768)[0].kwargs
    assert (kw["BK"], kw["NUM_STAGES"]) == (128, 3)
    assert kw["EPILOGUE_PARTITION"] is False
    # ... except small-M, where the wide shallow-K tile wins.
    kw = sm90._ws_heuristic_config(21197, 768, 768)[0].kwargs
    assert (kw["BN"], kw["BK"], kw["NUM_STAGES"]) == (256, 64, 3)
    assert kw["EPILOGUE_PARTITION"] is False
    assert kw["NUM_CTAS"] == 1
    # Giant-M past the 2-CTA crossover takes the wide twin ...
    kw = sm90._ws_heuristic_config(705448, 1536, 768)[0].kwargs
    assert (kw["BN"], kw["NUM_CTAS"]) == (256, 2)
    # ... while giant-M at/below it stays on one CTA.
    kw = sm90._ws_heuristic_config(262144, 1536, 768)[0].kwargs
    assert (kw["BN"], kw["NUM_CTAS"]) == (256, 1)


@pytest.mark.parametrize("bias_2d", [False, True])
@pytest.mark.parametrize("b_col_major", [False, True])
def test_addmm_pingpong_supported(bias_2d, b_col_major):
    """Pin the supported-but-unselected ping-pong mode via direct launch.

    The shape routes direct through the op, so this also covers WS
    correctness on direct-routed shapes.
    """
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    m, n, k = 2432, 896, 256
    torch.manual_seed(m + n + k + 2)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    if b_col_major:
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).t()
        assert b.stride(0) == 1
    else:
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n, ), device="cuda", dtype=torch.bfloat16)
    if bias_2d:
        bias = bias.unsqueeze(0).expand(m, n).contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    assert sm90._tma_eligible(a, b, out, n, k)

    b_row_major = b.stride(1) == 1
    b_src = b if b_row_major else b.T
    a_desc = TensorDescriptor(a, a.shape, a.stride(), [1, 1])
    b_desc = TensorDescriptor(b_src, b_src.shape, b_src.stride(), [1, 1])
    c_desc = TensorDescriptor(out, out.shape, out.stride(), [1, 1])
    num_sms = sm90._get_num_sms()
    cfg = sm90._ws_config(128, 128, 64, 8, 4, False, 1, False, True)
    assert cfg.kwargs["USE_PINGPONG"] is True
    kernel = triton.autotune(configs=[cfg], key=["M", "N", "K"])(sm90.addmm_kernel_tma_ws)
    grid = (min(triton.cdiv(m, 128) * triton.cdiv(n, 128), num_sms), )
    smy, sny = (0, bias.stride(0)) if bias.ndim == 1 else (bias.stride(0), bias.stride(1))
    kernel[grid](a_desc, b_desc, c_desc, bias, m, n, k, smy, sny, NUM_SMS=num_sms, B_ROW_MAJOR=b_row_major)
    torch.cuda.synchronize()

    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(out, expected, atol=0.05, rtol=0.05)

    del bias, a, b, out, expected
    torch.cuda.empty_cache()


@pytest.mark.parametrize("bias_2d", [False, True])
@pytest.mark.parametrize("b_col_major", [False, True])
def test_addmm_np_path(bias_2d, b_col_major):
    """Pin the non-persistent path (TMA loads, direct store) op-level."""
    from triton.tlx.ops import addmm as tlx_addmm

    m, n, k = 192, 128, 128
    assert sm90._use_np_path(m, n, k) is True
    torch.manual_seed(m + n + k + 3)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    if b_col_major:
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).t()
        assert b.stride(0) == 1
    else:
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n, ), device="cuda", dtype=torch.bfloat16)
    if bias_2d:
        bias = bias.unsqueeze(0).expand(m, n).contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    assert sm90._tma_eligible(a, b, out, n, k)

    actual = tlx_addmm(bias, a, b, out=out)
    expected = torch.addmm(bias, a, b)
    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)

    del bias, a, b, out, actual, expected
    torch.cuda.empty_cache()


def test_addmm_np_routing():
    """Pin NP/WS routing boundaries (pure selection, no GPU)."""
    assert sm90._use_np_path(652, 1536, 768) is True
    assert sm90._use_np_path(160, 256, 2560) is True
    assert sm90._use_np_path(20480, 768, 256) is False
    assert sm90._use_np_path(21197, 768, 768) is False
    assert sm90._use_np_path(23, 128, 128) is False
    assert sm90._use_np_path(256, 32, 128) is False
    assert sm90._use_np_path(256, 128, 64) is False

    kw = sm90._np_heuristic_config(867, 768, 256)[0].kwargs
    assert (kw["BM"], kw["BN"], kw["BK"]) == (64, 128, 128)
    kw = sm90._np_heuristic_config(652, 768, 768)[0].kwargs
    assert (kw["BM"], kw["BN"], kw["BK"]) == (64, 64, 256)
    # Sub-wave shapes take narrow-N/deep-K for CTA count ...
    kw = sm90._np_heuristic_config(160, 256, 2560)[0].kwargs
    assert (kw["BM"], kw["BN"], kw["BK"], kw["NUM_STAGES"]) == (64, 32, 256, 4)
    kw = sm90._np_heuristic_config(652, 256, 256)[0].kwargs
    assert (kw["BM"], kw["BN"], kw["BK"]) == (64, 32, 256)
    # ... while at/above a wave keeps BN=64 with deep K.
    kw = sm90._np_heuristic_config(717, 512, 768)[0].kwargs
    assert (kw["BM"], kw["BN"], kw["BK"]) == (64, 64, 256)


def test_addmm_np_space_covers_small_tiles():
    """Pin the BN=32/BK=256 family in the NP search space (no GPU)."""
    have = {(c.kwargs["BM"], c.kwargs["BN"], c.kwargs["BK"], c.kwargs["NUM_STAGES"]) for c in sm90._np_configs()}
    assert {(64, 32, 256, 4), (64, 32, 256, 3), (64, 64, 256, 3)} <= have
    kept = sm90._np_prune_configs(sm90._np_configs(), {"M": 160, "N": 256, "K": 2560})
    kept_tags = {(c.kwargs["BM"], c.kwargs["BN"], c.kwargs["BK"]) for c in kept}
    assert (64, 32, 256) in kept_tags


@pytest.mark.parametrize("cfg_idx", range(6))
def test_addmm_np_fused_barriers(cfg_idx):
    """Pin fused-barrier soundness on every NP space config.

    The fused full barrier (arrive_count=2, one expect per TMA)
    hangs on any expect pairing mistake, so each config launches
    directly. K=1024 flips stage phases on every config.
    """
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    m, n, k = 192, 256, 1024
    cfgs = sm90._np_configs()
    assert len(cfgs) == 6
    cfg = cfgs[cfg_idx]
    assert cfg in sm90._np_prune_configs(cfgs, {"M": m, "N": n, "K": k})
    torch.manual_seed(m + n + k + cfg_idx)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n, ), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    assert sm90._tma_eligible(a, b, out, n, k)

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [1, 1])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [1, 1])
    c_desc = TensorDescriptor(out, out.shape, out.stride(), [1, 1])
    num_sms = sm90._get_num_sms()
    kernel = triton.autotune(configs=[cfg], key=["M", "N", "K"])(sm90.addmm_kernel_tma_np)
    bm, bn = cfg.kwargs["BM"], cfg.kwargs["BN"]
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), )
    kernel[grid](a_desc, b_desc, c_desc, bias, out, m, n, k, 0, bias.stride(0), out.stride(0), out.stride(1),
                 NUM_SMS=num_sms, B_ROW_MAJOR=True)
    torch.cuda.synchronize()

    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(out, expected, atol=0.05, rtol=0.05)

    del bias, a, b, out, expected
    torch.cuda.empty_cache()


def test_addmm_direct_routing():
    """Pin small-shape direct routing + configs (pure selection, no GPU)."""
    assert sm90._use_direct_path(652, 256, 256) is True
    assert sm90._use_direct_path(4096, 768, 256) is True
    assert sm90._use_direct_path(256, 1024, 512) is True
    assert sm90._use_direct_path(8192, 768, 256) is False
    assert sm90._use_direct_path(652, 512, 512) is False
    assert sm90._use_direct_path(652, 768, 768) is False
    assert sm90._use_direct_path(20480, 768, 256) is False

    kw = sm90.heuristic_config(652, 256, 256)[0].kwargs
    assert (kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_K"]) == (128, 32, 32)
    kw = sm90.heuristic_config(4096, 768, 256)[0].kwargs
    assert (kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_K"]) == (128, 64, 32)
    kw = sm90.heuristic_config(256, 1024, 512)[0].kwargs
    assert (kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_K"]) == (64, 128, 32)
