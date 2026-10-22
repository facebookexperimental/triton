"""sm90 L1 correctness for ``tlx.ops.mm``."""

import contextlib

import pytest
import torch
from triton._internal_testing import is_hopper
from triton.tlx.ops import InvalidInput
from triton.tlx.ops.kernels.mm import sm90
from triton.tlx.ops.kernels.mm._shapes import SM90_FOCUS, SYNTHETIC

from mm_test_utils import REL_PRECISION, run_mm_case

pytestmark = pytest.mark.skipif(not is_hopper(), reason="Requires sm90")

ARCH = "sm90"
COOPERATIVE_CONFIGS = (
    (128, 256, 3, 1),
    (128, 256, 4, 1),
    (256, 128, 3, 1),
    (256, 128, 4, 1),
    (128, 128, 6, 1),
    (128, 64, 8, 1),
    (128, 256, 3, 2),
    (256, 128, 3, 3),
    (128, 128, 6, 5),
    (128, 64, 8, 2),
)


@pytest.mark.parametrize(
    "M, N, K, a_strides, b_strides, dtype_name",
    list(SYNTHETIC) + list(SM90_FOCUS),
)
def test_mm(M, N, K, a_strides, b_strides, dtype_name):
    run_mm_case(
        ARCH,
        M,
        N,
        K,
        a_strides,
        b_strides,
        dtype_name,
        allow_decline=False,
    )


def _inputs(M=256, N=256, K=128, dtype=torch.float16):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    return a, b


def _assert_close(actual, expected):
    precision = REL_PRECISION[actual.dtype]
    torch.testing.assert_close(
        actual,
        expected,
        atol=precision * expected.abs().max().item(),
        rtol=precision,
    )


def test_smoke_space_is_registered_and_runs():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs()
    actual = tlx_mm(a, b, space="smoke")
    _assert_close(actual, torch.matmul(a, b))


def test_out():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs(dtype=torch.bfloat16)
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    actual = tlx_mm(a, b, out=out, space="heuristic")
    assert actual is out
    _assert_close(actual, torch.matmul(a, b))


def test_rejects_invalid_out():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs()
    out = torch.empty((a.shape[0] - 1, b.shape[1]), device=a.device, dtype=a.dtype)
    with pytest.raises(InvalidInput, match="out must be"):
        tlx_mm(a, b, out=out)


def test_rejects_unknown_space():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs()
    with pytest.raises(InvalidInput, match="does not provide search space"):
        tlx_mm(a, b, space="unknown")


def test_rejects_unaligned_operand():
    from triton.tlx.ops import mm as tlx_mm

    storage = torch.randn((256 * 128 + 1, ), device="cuda", dtype=torch.float16)
    a = storage[1:].view(256, 128)
    _, b = _inputs()
    assert a.is_contiguous() and a.data_ptr() % 16 != 0
    with pytest.raises(InvalidInput, match="16-byte-aligned operands"):
        tlx_mm(a, b)


def test_rejects_unaligned_out():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs()
    storage = torch.empty((256 * 256 + 1, ), device="cuda", dtype=torch.float16)
    out = storage[1:].view(256, 256)
    assert out.is_contiguous() and out.data_ptr() % 16 != 0
    with pytest.raises(InvalidInput, match="16-byte-aligned output"):
        tlx_mm(a, b, out=out)


@contextlib.contextmanager
def _pinned_full_config(block_m, block_n, num_stages, split_k=1):
    original = sm90.CONFIGS
    sm90.CONFIGS = lambda: [sm90._config(block_m, block_n, num_stages, split_k=split_k)]
    sm90._tuned.cache_clear()
    try:
        yield
    finally:
        sm90.CONFIGS = original
        sm90._tuned.cache_clear()


@pytest.mark.parametrize("block_m, block_n, num_stages, split_k", COOPERATIVE_CONFIGS)
def test_cooperative_config(block_m, block_n, num_stages, split_k):
    from triton.tlx.ops import mm as tlx_mm

    # Five K tiles per split wrap both stage rings; 153+ work units also make
    # some SMs revisit the persistent loop on an H100. M is not a tile multiple,
    # so the last split slab has padding rows, and 41 K tiles leave a short
    # last split.
    k = 64 * (5 * split_k if split_k == 1 else 41)
    a, b = _inputs(M=2176 - 40, N=2176, K=k)
    with _pinned_full_config(block_m, block_n, num_stages, split_k):
        actual = tlx_mm(a, b, space="full")

    _assert_close(actual, torch.matmul(a, b))


@pytest.mark.parametrize("M, N, K", [(706, 256, 20992), (7, 4096, 4096), (706, 128, 3536), (706, 2560, 2560)])
def test_full_space_prune_fills_the_gpu(M, N, K):
    configs = sm90._prune_full_configs(sm90.CONFIGS(), {"M": M, "N": N, "K": K})
    num_sms = sm90._get_num_sms()
    assert configs
    for c in configs:
        kw = c.kwargs
        assert sm90._smem_bytes(kw["BM"], kw["BN"], kw["BK"], kw["NUM_STAGES"], kw["SPLIT_K"]) <= sm90._MAX_SMEM
        work = sm90._mn_tiles(M, N, kw["BM"], kw["BN"]) * kw["SPLIT_K"]
        assert kw["SPLIT_K"] == 1 or work <= num_sms
    best = max(sm90._mn_tiles(M, N, c.kwargs["BM"], c.kwargs["BN"]) * c.kwargs["SPLIT_K"] for c in configs)
    assert best > num_sms // 2


def test_split_k_is_deterministic():
    # Guards the fixed-order workspace reduction: switching split-K to fp32
    # atomics or TMA reduce-add would make results depend on CTA scheduling.
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs(M=706, N=256, K=20992)
    assert sm90.heuristic_config(706, 256, 20992)[0].kwargs["SPLIT_K"] > 1
    first = tlx_mm(a, b, space="heuristic")
    for _ in range(3):
        assert torch.equal(tlx_mm(a, b, space="heuristic"), first)
    _assert_close(first, torch.matmul(a, b))


@contextlib.contextmanager
def _pinned_swap_config(block_m, block_n, block_k, split_k):
    original = sm90._swap_configs
    num_stages = sm90._deepest_ring(block_m, block_n, block_k, split_k)
    sm90._swap_configs = lambda: [sm90._config(block_m, block_n, num_stages, 1, split_k, block_k, True)]
    sm90._tuned.cache_clear()
    try:
        yield
    finally:
        sm90._swap_configs = original
        sm90._tuned.cache_clear()


def _operand(rows, cols, row_major, dtype=torch.float16):
    t = torch.randn((rows, cols) if row_major else (cols, rows), device="cuda", dtype=dtype)
    return t if row_major else t.T


@pytest.mark.parametrize("M", [1, 7, 33, 64])
@pytest.mark.parametrize("a_row_major, b_row_major", [(True, False), (True, True), (False, True), (False, False)])
@pytest.mark.parametrize("block_m, block_k, split_k", [(128, 128, 1), (128, 256, 1), (128, 128, 3), (128, 256, 2),
                                                       (256, 64, 1)])
def test_swap_ab_config(M, a_row_major, b_row_major, block_m, block_k, split_k):
    from triton.tlx.ops import mm as tlx_mm

    if not a_row_major and M % 8:
        pytest.skip("column-major A needs a 16-byte-aligned leading stride")
    torch.manual_seed(0)
    # N is not a tile multiple; 41 * 64 K leaves a short last split.
    N, K = 1000, 64 * 41
    a = _operand(M, K, a_row_major)
    b = _operand(K, N, b_row_major)
    assert sm90._use_swap_ab(M, "full")
    with _pinned_swap_config(block_m, sm90._swap_block_n(M), block_k, split_k):
        actual = tlx_mm(a, b, space="full")

    _assert_close(actual, torch.matmul(a, b))


@pytest.mark.parametrize("M, N, K", [(7, 16384, 4096), (7, 4096, 4096), (7, 1024, 4992), (7, 4992, 128),
                                     (64, 4096, 4096)])
def test_swap_prune_fills_the_gpu(M, N, K):
    configs = sm90._prune_swap_configs(sm90._swap_configs(), {"M": N, "N": M, "K": K})
    num_sms = sm90._get_num_sms()
    assert configs
    for c in configs:
        kw = c.kwargs
        assert kw["SWAP_AB"] and kw["BN"] == sm90._swap_block_n(M)
        assert sm90._fits_smem(kw["BM"], kw["BN"], kw["NUM_STAGES"], kw["SPLIT_K"], kw["BK"], swap_ab=True)
        assert kw["SPLIT_K"] == 1 or sm90._mn_tiles(N, M, kw["BM"], kw["BN"]) * kw["SPLIT_K"] <= num_sms


def test_swap_ab_split_k_is_deterministic():
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs(M=7, N=4096, K=4096)
    assert sm90._swap_heuristic_config(7, 4096, 4096)[0].kwargs["SPLIT_K"] > 1
    first = tlx_mm(a, b, space="heuristic")
    for _ in range(3):
        assert torch.equal(tlx_mm(a, b, space="heuristic"), first)
    _assert_close(first, torch.matmul(a, b))
