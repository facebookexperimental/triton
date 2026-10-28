"""sm90 L1 correctness for ``tlx.ops.mm``."""

import contextlib

import pytest
import torch
from triton._internal_testing import is_hopper
from triton.tlx.ops import BitwiseInvarianceSpec, InvalidInput
from triton.tlx.ops.kernels.mm import sm90
from triton.tlx.ops.kernels.mm._shapes import SM90_FOCUS, SYNTHETIC

from mm_test_utils import REL_PRECISION, run_mm_case, run_mm_invariance_case

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
    sm90._FORCE_PATH = "ws"
    sm90._tuned.cache_clear()
    try:
        yield
    finally:
        sm90.CONFIGS = original
        sm90._FORCE_PATH = None
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


@pytest.mark.parametrize("M, N, K, group_size_m", [(706, 2560, 2560, 8), (706, 2560, 20992, 8),
                                                   (706, 512, 20992, 1),  # split-K
                                                   (706, 512, 512, 1),  # N <= 512
                                                   ])
def test_heuristic_tile_order_matches_full_space(M, N, K, group_size_m):
    # The sub-wave heuristic tile follows the full-space prune: row-major order
    # only for N <= 512 or split-K, grouped order otherwise.
    kw = sm90.heuristic_config(M, N, K)[0].kwargs
    assert kw["GROUP_SIZE_M"] == group_size_m
    full = sm90._prune_full_configs(sm90.CONFIGS(), {"M": M, "N": N, "K": K})
    assert any(c.kwargs["GROUP_SIZE_M"] == group_size_m and c.kwargs["SPLIT_K"] == kw["SPLIT_K"] for c in full)


@pytest.mark.parametrize("M, N, K, b_row_major, tile", [
    # Non-power-of-two tiles for a K-major B (measured best on H100).
    (706, 3072, 12288, False, (128, 192, 1)),
    (706, 6144, 3072, False, (128, 192, 1)),
    # ... but not where a power-of-two tile ties it.
    (706, 2048, 7072, False, (128, 128, 1)),
    (706, 20992, 512, False, (128, 256, 1)),
    (706, 8192, 2560, False, (128, 192, 1)),
    # ... never for a row-major B.
    (706, 8192, 2560, True, (128, 128, 1)),
    # A narrow whole-K tile over a split one when both fill the GPU alike.
    (706, 1280, 2560, True, (128, 64, 1)),
    (706, 288, 2048, True, (128, 64, 4)),
    (706, 256, 20992, True, (128, 128, 11)),
    (706, 2560, 20992, True, (128, 128, 1)),
])
def test_sub_wave_heuristic_picks(M, N, K, b_row_major, tile):
    kw = sm90.heuristic_config(M, N, K, b_row_major)[0].kwargs
    assert (kw["BM"], kw["BN"], kw["SPLIT_K"]) == tile
    assert sm90._fits_smem(kw["BM"], kw["BN"], kw["NUM_STAGES"], kw["SPLIT_K"])


@pytest.mark.parametrize("M, N, K", [(38400, 256, 640), (81920, 256, 704), (2252800, 256, 256)])
def test_tall_narrow_n_takes_128x128(M, N, K):
    """Past two waves of the wide tile, a tall product with N <= 256 runs
    128x128 in both spaces: the heuristic picks it and the full-space prune
    keeps it."""
    kw = sm90.heuristic_config(M, N, K)[0].kwargs
    assert (kw["BM"], kw["BN"], kw["SPLIT_K"], kw["GROUP_SIZE_M"]) == (128, 128, 1, 1)
    full = sm90._prune_full_configs(sm90.CONFIGS(), {"M": M, "N": N, "K": K})
    assert any((c.kwargs["BM"], c.kwargs["BN"]) == (128, 128) for c in full)
    # A wider N still prefers the wide tile.
    kw = sm90.heuristic_config(M, 1024, K)[0].kwargs
    assert (kw["BM"], kw["BN"]) in ((256, 128), (128, 256))


@pytest.mark.parametrize("M, N, K, tile", [
    # Few-tile, K up to 1024: 64x64 tiles, deep K step within two waves.
    (600, 128, 576, (64, 64, 128)),
    (706, 512, 1024, (64, 64, 128)),
    (600, 2048, 576, (64, 64, 64)),
    # Past three waves of 64x64 tiles, or past K = 1024, or skinny M: not small-K.
    (2048, 1024, 768, None),
    (706, 256, 1536, None),
    (32, 512, 768, None),
])
def test_few_tile_small_k_heuristic(M, N, K, tile):
    assert sm90._use_small_k(M, N, K, "heuristic") == (tile is not None)
    assert sm90._use_small_k(M, N, K, "smoke") is False
    if tile is not None:
        kw = sm90._small_k_heuristic_config(M, N, K)[0].kwargs
        assert (kw["BM"], kw["BN"], kw["BK"]) == tile
        # The full space times the small-K kernel against WS there too.
        assert sm90._small_k_candidate(M, N, K, "full")


def test_few_tile_small_k_is_correct():
    from triton.tlx.ops import addmm as tlx_addmm
    from triton.tlx.ops import mm as tlx_mm

    torch.manual_seed(0)
    for M, N, K in ((600, 128, 576), (600, 2048, 576)):
        a = _operand(M, K, True)
        b = _operand(K, N, False)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        _assert_close(tlx_mm(a, b, space="heuristic"), torch.matmul(a, b))
        torch.testing.assert_close(tlx_addmm(bias, a, b, space="heuristic"), torch.addmm(bias, a, b), atol=5e-2,
                                   rtol=5e-2)


def test_small_k_space_has_deep_k_64x64():
    tiles = {
        sm90._small_k_tile(c)
        for c in sm90._prune_small_k_configs(sm90._small_k_configs(), {"M": 600, "N": 128, "K": 576})
    }
    assert (64, 64, 128, 4, 3) in tiles


@pytest.mark.parametrize("M, N, K, tile", [
    (706, 256, 256, (64, 64)),
    (706, 3536, 128, (64, 128)),
    (8192, 8192, 128, (128, 128)),
])
def test_small_k_heuristic_fills_two_waves(M, N, K, tile):
    kw = sm90._small_k_heuristic_config(M, N, K)[0].kwargs
    assert (kw["BM"], kw["BN"]) == tile


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
    sm90._FORCE_PATH = "ws"
    sm90._tuned.cache_clear()
    try:
        yield
    finally:
        sm90._swap_configs = original
        sm90._FORCE_PATH = None
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


_INVARIANCE_SPECS = [
    BitwiseInvarianceSpec.dims("M"),
    BitwiseInvarianceSpec.dims("K"),
    BitwiseInvarianceSpec.all(),
]

_INVARIANCE_SHAPES = [
    (1024, 1024, 1024, (1024, 1), (1024, 1), "bf16"),
    (512, 2048, 512, (512, 1), (2048, 1), "bf16"),
]


@pytest.mark.parametrize("spec", _INVARIANCE_SPECS)
@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", _INVARIANCE_SHAPES)
def test_mm_invariance(spec, M, N, K, a_strides, b_strides, dtype_name):
    """A non-empty spec is accepted and gives exact M-slice equality."""
    run_mm_invariance_case(ARCH, M, N, K, a_strides, b_strides, dtype_name, spec)


@pytest.mark.parametrize("tile", sm90._SMALL_K_TILES)
@pytest.mark.parametrize("a_row_major, b_row_major", [(True, False), (True, True), (False, True), (False, False)])
def test_small_k_config(monkeypatch, tile, a_row_major, b_row_major):
    from triton.tlx.ops import mm as tlx_mm

    torch.manual_seed(0)
    # Ragged M and N, and a K that is not a multiple of any BK (TMA zero-fills the tail).
    M, N, K = 712, 1000, 200
    a = _operand(M, K, a_row_major)
    b = _operand(K, N, b_row_major)
    cfg = sm90._small_k_config(*tile)
    monkeypatch.setattr(sm90, "_small_k_configs", lambda: [cfg])
    monkeypatch.setattr(sm90, "_FORCE_PATH", "small_k")
    sm90._tuned_small_k.cache_clear()
    try:
        actual = tlx_mm(a, b, space="full")
    finally:
        sm90._tuned_small_k.cache_clear()
    _assert_close(actual, torch.matmul(a, b))


@pytest.mark.parametrize(
    "M, N, K, small_k",
    [
        (706, 3536, 128, True),
        (8192, 8192, 128, True),
        (706, 2560, 256, True),
        (706, 20992, 128, False),
        (706, 20992, 256, False),
        (4096, 4096, 256, False),
        (706, 3536, 512, False),
        (7, 4992, 128, True),
        (64, 1024, 128, True),
        (7, 4096, 4096, False),
        (7, 4992, 256, False),
        (7, 20992, 128, True),
    ],
)
def test_small_k_dispatch(M, N, K, small_k):
    assert sm90._use_small_k(M, N, K, "full") == small_k
    assert not sm90._use_small_k(M, N, K, "smoke")


@pytest.mark.parametrize("tile", sm90._SKINNY_SMALL_K_TILES)
@pytest.mark.parametrize("M", [1, 7, 64])
@pytest.mark.parametrize("a_row_major, b_row_major", [(True, False), (True, True), (False, True), (False, False)])
def test_small_k_skinny_m(monkeypatch, tile, M, a_row_major, b_row_major):
    from triton.tlx.ops import mm as tlx_mm

    if not a_row_major and M % 8:
        pytest.skip("column-major A needs a 16-byte-aligned leading stride")
    torch.manual_seed(0)
    N, K = 1000, 120
    a = _operand(M, K, a_row_major)
    b = _operand(K, N, b_row_major)
    assert sm90._use_small_k(M, N, K, "full")
    monkeypatch.setattr(sm90, "_FORCE_PATH", "small_k")
    monkeypatch.setattr(sm90, "_small_k_configs", lambda: [sm90._small_k_config(*tile)])
    sm90._tuned_small_k.cache_clear()
    try:
        actual = tlx_mm(a, b, space="full")
    finally:
        sm90._tuned_small_k.cache_clear()
    _assert_close(actual, torch.matmul(a, b))


# Not production shapes: tiles are (BM, BN, BK, warps, stages) entries of
# _SMALL_K_TILES (M = 72, ragged and > 64) and _SKINNY_SMALL_K_TILES (M = 7, and
# M = 64 at the swap-AB threshold). Production shapes run in test_mm through the
# sm90_1 focus suite.
@pytest.mark.parametrize("M, tile", [
    (72, (64, 128, 64, 4, 3)),
    (72, (128, 128, 64, 8, 3)),
    (72, (64, 64, 64, 4, 3)),
    (7, (16, 64, 64, 2, 2)),
    (7, (32, 64, 64, 4, 2)),
    (64, (16, 128, 64, 4, 2)),
])
@pytest.mark.parametrize("split_k", [1, 3])
@pytest.mark.parametrize("a_row_major, b_row_major", [(True, False), (True, True), (False, True), (False, False)])
def test_small_k_split_k_config(monkeypatch, M, tile, split_k, a_row_major, b_row_major):
    from triton.tlx.ops import mm as tlx_mm

    if not a_row_major and M % 8:
        pytest.skip("column-major A needs a 16-byte-aligned leading stride")
    torch.manual_seed(0)
    # Ragged N; 41 K tiles wrap the pipeline and leave a short last split.
    N, K = 1000, 64 * 41
    a = _operand(M, K, a_row_major)
    b = _operand(K, N, b_row_major)
    monkeypatch.setattr(sm90, "_FORCE_PATH", "small_k")
    monkeypatch.setattr(sm90, "_small_k_configs", lambda: [sm90._small_k_config(*tile, split_k=split_k)])
    sm90._tuned_small_k.cache_clear()
    try:
        actual = tlx_mm(a, b, space="full")
        assert sm90._tuned_small_k("full", None).best_config.kwargs["SPLIT_K"] == split_k
    finally:
        sm90._tuned_small_k.cache_clear()
    _assert_close(actual, torch.matmul(a, b))


def test_small_k_split_k_is_deterministic(monkeypatch):
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs(M=7, N=4096, K=4096)
    monkeypatch.setattr(sm90, "_FORCE_PATH", "small_k")
    monkeypatch.setattr(sm90, "_small_k_configs", lambda: [sm90._small_k_config(16, 64, 64, 2, 2, split_k=2)])
    sm90._tuned_small_k.cache_clear()
    try:
        first = tlx_mm(a, b, space="full")
        assert sm90._tuned_small_k("full", None).best_config.kwargs["SPLIT_K"] == 2
        for _ in range(3):
            assert torch.equal(tlx_mm(a, b, space="full"), first)
    finally:
        sm90._tuned_small_k.cache_clear()
    _assert_close(first, torch.matmul(a, b))


def test_small_k_prune_splits_only_past_small_k():
    num_sms = sm90._get_num_sms()
    small = sm90._prune_small_k_configs(sm90._small_k_configs(), {"M": 706, "N": 3536, "K": 128})
    assert small and all(c.kwargs["SPLIT_K"] == 1 for c in small)
    skinny = sm90._prune_small_k_configs(sm90._small_k_configs(), {"M": 7, "N": 4096, "K": 4096})
    assert any(c.kwargs["SPLIT_K"] > 1 for c in skinny)
    for c in skinny:
        kw = c.kwargs
        assert kw["BM"] <= 64
        assert sm90._mn_tiles(7, 4096, kw["BM"], kw["BN"]) * kw["SPLIT_K"] <= num_sms or kw["SPLIT_K"] == 1


@pytest.mark.parametrize("M, N, K, candidates", [
    (7, 4992, 128, {"ws", "small_k"}),
    (7, 4096, 4096, {"ws", "small_k"}),
    (706, 3536, 128, {"ws", "small_k"}),
    (706, 2560, 2560, {"ws"}),
])
def test_full_space_selects_fastest_candidate(monkeypatch, M, N, K, candidates):
    from triton.tlx.ops import mm as tlx_mm

    a, b = _inputs(M=M, N=N, K=K)
    seen = {}
    real_select = sm90._select_path

    def spy(space, key, launchers):
        seen["candidates"] = set(launchers)
        return real_select(space, key, launchers)

    monkeypatch.setattr(sm90, "_select_path", spy)
    monkeypatch.setattr(sm90, "_PATH_CACHE", {})
    actual = tlx_mm(a, b, space="full")
    assert seen["candidates"] == candidates
    key = (M, N, K, True, True, a.dtype)
    if len(candidates) > 1:
        assert sm90._PATH_CACHE[key] in candidates
    _assert_close(actual, torch.matmul(a, b))
    # A second call reuses the cached choice.
    _assert_close(tlx_mm(a, b, space="full"), torch.matmul(a, b))


def test_npot_config_not_reused_across_b_layouts(monkeypatch):
    """The autotune cache is keyed on the operand layouts: a non-power-of-two
    tile tuned for a K-major B must not serve a row-major B of the same shape,
    which that tile cannot run."""
    from triton.tlx.ops import mm as tlx_mm

    M, N, K = 2136, 1000, 64 * 7
    npot = next(c for c in sm90._npot_configs() if c.kwargs["BN"] == 192)
    monkeypatch.setattr(sm90, "CONFIGS", lambda: [npot, sm90._config(128, 128, 4)])
    real_prune = sm90._prune_full_configs

    def prune(configs, nargs, **kwargs):
        # Make the K-major call pick the 192 tile deterministically.
        if not kwargs.get("B_ROW_MAJOR", True):
            return [c for c in configs if c.kwargs["BN"] == 192]
        return real_prune(configs, nargs, **kwargs)

    monkeypatch.setattr(sm90, "_prune_full_configs", prune)
    monkeypatch.setattr(sm90, "_FORCE_PATH", "ws")
    sm90._tuned.cache_clear()
    try:
        torch.manual_seed(0)
        a = _operand(M, K, True)
        b_col = _operand(K, N, False)
        _assert_close(tlx_mm(a, b_col, space="full"), torch.matmul(a, b_col))
        assert sm90._tuned("full", None, False).best_config.kwargs["BN"] == 192
        b_row = b_col.contiguous()
        _assert_close(tlx_mm(a, b_row, space="full"), torch.matmul(a, b_row))
        assert sm90._tuned("full", None, False).best_config.kwargs["BN"] == 128
    finally:
        sm90._tuned.cache_clear()


@pytest.mark.parametrize("block_n", [96, 160, 192, 224])
@pytest.mark.parametrize("a_row_major", [True, False])
def test_npot_block_n_config(block_n, a_row_major):
    from triton.tlx.ops import mm as tlx_mm

    torch.manual_seed(0)
    # Ragged M and N, more tiles than SMs (persistent revisits), and a K that
    # wraps the stage ring; B is K-major (W^T), the only layout NPOT tiles use.
    M, N, K = 2136, 1000, 64 * 7
    a = _operand(M, K, a_row_major)
    b = _operand(K, N, False)
    num_stages = {c.kwargs["BN"]: c.kwargs["NUM_STAGES"] for c in sm90._npot_configs()}[block_n]
    with _pinned_full_config(128, block_n, num_stages):
        actual = tlx_mm(a, b, space="full")
        assert sm90._tuned("full", None, False).best_config.kwargs["BN"] == block_n

    _assert_close(actual, torch.matmul(a, b))


def test_n_piece_decomposition():
    assert [sm90._n_piece(96, i) for i in range(3)] == [64, 32, 0]
    assert [sm90._n_piece(224, i) for i in range(4)] == [128, 64, 32, 0]
    assert [sm90._n_piece(128, i) for i in range(2)] == [128, 0]


def test_npot_prune_rules():
    num_sms = sm90._get_num_sms()

    def npot_kept(M, N, K, b_row_major):
        kept = sm90._prune_full_configs(sm90.CONFIGS(), {"M": M, "N": N, "K": K}, B_ROW_MAJOR=b_row_major)
        return {c.kwargs["BN"] for c in kept if not sm90._is_pow2(c.kwargs["BN"])}

    # 2048 / 96 = 22 tiles across N fill exactly one wave with M = 706.
    assert 96 in npot_kept(706, 2048, 7072, False)
    # 128x192 only ties 128x128 on per-SM work here (2 waves x 192 vs 3 x 128),
    # but the two-piece tile is kept: 128x128 is SMEM-bandwidth bound.
    assert 192 in npot_kept(706, 8192, 2560, False)
    # Against a 128x256 best tile an NPOT tile still needs a 3% cut.
    assert not npot_kept(4096, 4096, 4096, False)
    # N-major B never takes N pieces; a shape that tiles evenly gains nothing.
    assert not npot_kept(706, 2048, 7072, True)
    assert not npot_kept(8192, 8192, 8192, False)
    for bn in sm90._NPOT_BLOCK_NS:
        pieces = [sm90._n_piece(bn, i) for i in range(4)]
        assert pieces[3] == 0 and min(p for p in pieces if p) >= 32
        assert sm90._fits_smem(128, bn, 2, 1)
    assert num_sms > 0
