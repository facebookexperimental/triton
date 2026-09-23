"""sm100 L1 correctness for ``tlx.ops.mm``.

Runs the hardware-agnostic synthetic list plus every MM focus suite. L2 perf
selects only the running host's configured default.

A shape the op declines is reported as a skip with the reason, never as a
pass.

TODO: cover the USE_WARP_BARRIER config variant dropped with the tutorial copy
of this kernel. Only the heuristic-selected config runs today.

The split-K tests include white-box workspace-layout coverage because the
public API cannot pin ``SPLIT_K``/``NUM_CTAS``. A split's region must cover the
whole padded tile grid; otherwise an epilogue tile can overwrite the next
split's partials.
"""

import contextlib
import time
from types import SimpleNamespace

import pytest
import torch
import triton
from triton._internal_testing import is_blackwell
from triton.tlx.ops.kernels.mm import sm100
from triton.tlx.ops.kernels.mm._shapes import CORRECTNESS_SHAPES, operand

from mm_test_utils import MAX_SECONDS_PER_CASE, REL_PRECISION, run_mm_case

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="Requires sm100")

ARCH = "sm100"


@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", CORRECTNESS_SHAPES)
def test_mm(M, N, K, a_strides, b_strides, dtype_name):
    run_mm_case(ARCH, M, N, K, a_strides, b_strides, dtype_name)


GEOMETRIES = [
    # No overhang: M is a whole number of tiles and the count already divides
    # NUM_CTAS. These must stay at zero -- they are the canary for the fix
    # over-padding and wasting memory on the common case.
    (1024, 256, 1, 0),
    (1024, 256, 2, 0),
    (256, 256, 1, 0),
    # M % BLOCK_SIZE_M != 0.
    (1000, 256, 1, 24),
    (136, 128, 1, 120),
    (64, 128, 1, 64),
    # M % BLOCK_SIZE_M == 0, but an odd tile count padded up to NUM_CTAS=2.
    (384, 128, 2, 128),
    (768, 256, 2, 256),
]

SPLIT_KS = [1, 2, 3, 4, 8]


def _geometry(M, BLOCK_SIZE_M, NUM_CTAS):
    num_pid_m = sm100._padded_num_pid_m(M, BLOCK_SIZE_M, NUM_CTAS)
    rows = sm100._workspace_rows_per_split(M, BLOCK_SIZE_M, NUM_CTAS)
    # Exclusive upper bound on rows the epilogue stores within one split.
    written = num_pid_m * BLOCK_SIZE_M
    return num_pid_m, rows, written


@pytest.mark.parametrize("M, BLOCK_SIZE_M, NUM_CTAS, overhang", GEOMETRIES)
def test_geometry_is_what_the_case_claims(M, BLOCK_SIZE_M, NUM_CTAS, overhang):
    """Pin the overhang, so a config change cannot quietly defang these cases."""
    _, _, written = _geometry(M, BLOCK_SIZE_M, NUM_CTAS)
    assert written - M == overhang


@pytest.mark.parametrize("M, BLOCK_SIZE_M, NUM_CTAS, overhang", GEOMETRIES)
def test_rows_per_split_covers_the_tile_grid(M, BLOCK_SIZE_M, NUM_CTAS, overhang):
    """A split's region must hold every row the epilogue can store into it."""
    _, rows, written = _geometry(M, BLOCK_SIZE_M, NUM_CTAS)
    assert rows >= written, (f"split region is {rows} rows but the epilogue writes up to {written}; "
                             f"the last tile overhangs by {written - rows} rows into the next split")
    # And the reduction reads M rows back out of that region.
    assert rows >= M


@pytest.mark.parametrize("SPLIT_K", SPLIT_KS)
@pytest.mark.parametrize("M, BLOCK_SIZE_M, NUM_CTAS, overhang", GEOMETRIES)
def test_splits_do_not_alias(M, BLOCK_SIZE_M, NUM_CTAS, overhang, SPLIT_K):
    """No split may write into the next split's region."""
    _, rows, written = _geometry(M, BLOCK_SIZE_M, NUM_CTAS)
    for s in range(SPLIT_K - 1):
        end_of_writes = s * rows + written
        start_of_next = (s + 1) * rows
        assert end_of_writes <= start_of_next, (f"split {s} writes through row {end_of_writes}, "
                                                f"but split {s + 1} starts at row {start_of_next}")


@pytest.mark.parametrize("SPLIT_K", SPLIT_KS)
@pytest.mark.parametrize("M, BLOCK_SIZE_M, NUM_CTAS, overhang", GEOMETRIES)
def test_allocation_covers_every_written_row(M, BLOCK_SIZE_M, NUM_CTAS, overhang, SPLIT_K):
    """The workspace must be at least as tall as the highest row written."""
    _, rows, written = _geometry(M, BLOCK_SIZE_M, NUM_CTAS)
    allocated = SPLIT_K * rows
    highest = (SPLIT_K - 1) * rows + written
    assert allocated >= highest


@pytest.mark.parametrize("M, N, K", [
    (1000, 1000, 1024),
    (64, 4096, 4096),
    (256, 256, 16384),
    (136, 256, 128),
])
def test_heuristic_configs_have_a_sound_workspace(M, N, K):
    """Tie the invariant to configs the heuristic actually emits in production."""
    for num_sms in (148, 132):  # B200 and H100-class SM counts
        cfg = sm100.get_heuristic_config(M, N, K, num_sms)
        if cfg is None or cfg.get("SPLIT_K", 1) == 1:
            continue
        _, rows, written = _geometry(M, cfg["BLOCK_SIZE_M"], cfg.get("NUM_CTAS", 1))
        assert rows >= written, f"heuristic config for {M}x{N}x{K} on {num_sms} SMs has an aliasing workspace: {cfg}"


@pytest.mark.parametrize(
    "M, N, K",
    [
        (64512, 128, 512),
        (1000000, 512, 512),
        (3159809, 384, 384),
    ],
)
def test_tall_m_heuristic_cluster_fits_one_group(M, N, K):
    cfg = sm100.get_heuristic_config(M, N, K, num_sms=148)
    assert cfg["NUM_CTAS"] == 2
    assert cfg["GROUP_SIZE_M"] % cfg["NUM_CTAS"] == 0
    assert cfg["GROUP_SIZE_M"] >= cfg["NUM_CTAS"]


@pytest.mark.parametrize("M, N", [(1, 1), (1280, 1280), (128, 10000)])
def test_group_size_fits_padded_cluster_grid(M, N):
    block_m = 256
    num_ctas = 2
    group_size = sm100._select_group_size_m(M, N, block_m, num_ctas)
    padded_tiles = sm100._padded_num_pid_m(M, block_m, num_ctas)
    assert num_ctas <= group_size <= padded_tiles
    assert group_size % num_ctas == 0


def test_split_k_workspace_uses_fp32_partials():
    c = torch.empty((384, 384), device="cuda", dtype=torch.bfloat16)
    workspace_desc = SimpleNamespace(base=c, shape=list(c.shape), block_shape=[1, 1])
    nargs = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "NUM_MMA_GROUPS": 2,
        "NUM_CTAS": 1,
        "SPLIT_K": 4,
        "M": 384,
        "N": 384,
        "a_desc": SimpleNamespace(block_shape=[1, 1]),
        "b_desc": SimpleNamespace(block_shape=[1, 1]),
        "c_desc": SimpleNamespace(base=c, block_shape=[1, 1]),
        "workspace_desc": workspace_desc,
    }
    sm100.matmul_tma_set_block_size_hook(nargs)
    assert workspace_desc.base.dtype == torch.float32
    assert workspace_desc.shape == [4 * 384, 384]


def test_preprocess_prunes_oversized_fp32_epilogue():
    config = triton.Config({
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "NUM_SMEM_BUFFERS": 3,
        "NUM_TMEM_BUFFERS": 2,
        "NUM_MMA_GROUPS": 1,
        "EPILOGUE_SUBTILE": 2,
        "NUM_CTAS": 1,
        "SPLIT_K": 4,
        "INTERLEAVE_EPILOGUE": 0,
    })
    assert sm100.preprocess_configs([config], {"M": 128, "N": 256, "K": 4096}) == []


@contextlib.contextmanager
def _pinned_config(overrides):
    """Force ``space="heuristic"`` to compile exactly one config."""
    original = sm100.heuristic_config

    def one_config(M, N, K):
        cfg = sm100.get_heuristic_config(M, N, K, sm100._get_num_sms())
        assert cfg is not None, f"no heuristic config for {M}x{N}x{K}"
        cfg = dict(cfg)
        cfg.pop("ctas_per_cga", None)
        pre_hook = cfg.pop("pre_hook", None) or sm100.matmul_tma_set_block_size_hook
        cfg.update(overrides)
        num_ctas = cfg.get("NUM_CTAS", 1)
        return [
            triton.Config(cfg, num_warps=4, num_stages=1, pre_hook=pre_hook,
                          ctas_per_cga=(num_ctas, 1, 1) if num_ctas > 1 else None)
        ]

    sm100.heuristic_config = one_config
    sm100._tuned.cache_clear()
    try:
        yield
    finally:
        sm100.heuristic_config = original
        sm100._tuned.cache_clear()


GPU_SHAPES = [
    # M % BLOCK_SIZE_M != 0 -- the reported bug (BLOCK_SIZE_M=256 -> 24 rows over).
    (1000, 1000, 1024, 1, torch.float16),
    # Whole region overhangs: one tile of 128 rows holds only 64 real rows.
    (64, 4096, 4096, 1, torch.float16),
    # Long-K bf16 case needs fp32 partials before the split reduction.
    (384, 384, 19459, 1, torch.bfloat16),
]

SPLIT_KS_GPU = [1, 4]


@pytest.mark.parametrize("SPLIT_K", SPLIT_KS_GPU)
@pytest.mark.parametrize("M, N, K, NUM_CTAS, dtype", GPU_SHAPES)
def test_output_is_independent_of_split_k(M, N, K, NUM_CTAS, dtype, SPLIT_K):
    """Splitting the reduction must not change the result."""
    from triton.tlx.ops import mm as tlx_mm

    # A row-major descriptor would have row stride K, which is not necessarily
    # 16-byte aligned (for example K=19459). Use layouts accepted by SM100 TMA.
    a = operand(M, K, (1, M), dtype)
    b = operand(K, N, (N, 1), dtype)

    with _pinned_config({"SPLIT_K": SPLIT_K, "NUM_CTAS": NUM_CTAS}):
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = tlx_mm(a, b, space="heuristic")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        # The autotuner widens back to the full space when pruning empties a
        # reduced one, which would silently run a different config than the one
        # under test. Confirm the pin survived.
        chosen = sm100._tuned("heuristic", (M, N, K)).best_config.kwargs
        assert chosen["SPLIT_K"] == SPLIT_K and chosen["NUM_CTAS"] == NUM_CTAS, \
            f"pin was defeated by config pruning; ran {chosen}"

    assert elapsed < MAX_SECONDS_PER_CASE, (f"mm({M}x{N}x{K}, SPLIT_K={SPLIT_K}) took {elapsed:.1f}s, "
                                            f"over the {MAX_SECONDS_PER_CASE}s budget")

    ref = torch.matmul(a, b)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)


def test_mm_rejects_unsupported_backward():
    from triton.tlx.ops import UnsupportedBackward
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((16, 16), device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    with pytest.raises(UnsupportedBackward, match="tlx.ops.mm does not support backward on sm100"):
        tlx_mm(a, b)

    with torch.no_grad():
        out = tlx_mm(a, b)
    assert not out.requires_grad
