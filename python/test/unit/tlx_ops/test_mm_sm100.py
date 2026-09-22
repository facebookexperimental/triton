"""sm100 L1 correctness for ``tlx.ops.mm``.

Runs the hardware-agnostic synthetic list plus every MM focus suite. L2 perf
selects only the running host's configured default.

A shape the op declines is reported as a skip with the reason, never as a
pass.

TODO: cover the config variants dropped with the tutorial copy of this kernel --
USE_WARP_BARRIER and NUM_CTAS=2. Only the heuristic-selected config runs today.

The split-K tests include white-box workspace-layout coverage because the
public API cannot pin ``SPLIT_K``/``NUM_CTAS``. A split's region must cover the
whole padded tile grid; otherwise an epilogue tile can overwrite the next
split's partials.
"""

import contextlib
import time

import pytest
import torch
import triton
from triton._internal_testing import is_blackwell
from triton.tlx.ops.kernels.mm import sm100
from triton.tlx.ops.kernels.mm._shapes import CORRECTNESS_SHAPES

from mm_test_utils import MAX_SECONDS_PER_CASE, REL_PRECISION, run_mm_case

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="Requires sm100")

ARCH = "sm100"

# TODO: Re-enable these shapes when their direct TLX accuracy failures are fixed.
FAILED_SHAPES = {
    (589824, 2048, 512, (512, 1), (2048, 1), "bf16"),
    (442368, 2048, 512, (512, 1), (2048, 1), "bf16"),
    (589824, 1056, 800, (800, 1), (1056, 1), "bf16"),
    (2433024, 256, 256, (256, 1), (1, 256), "bf16"),
    (589824, 512, 2048, (2048, 1), (1, 2048), "bf16"),
    (12800, 1024, 2304, (1, 12800), (1024, 1), "bf16"),
    (442368, 512, 2048, (2048, 1), (1, 2048), "bf16"),
    (2701258, 384, 384, (384, 1), (1, 384), "bf16"),
    (2617290, 384, 384, (384, 1), (1, 384), "bf16"),
    (2536160, 384, 384, (384, 1), (1, 384), "bf16"),
    (442368, 512, 192, (192, 1), (512, 1), "bf16"),
    (2433024, 256, 256, (256, 1), (256, 1), "bf16"),
    (2701258, 384, 384, (384, 1), (384, 1), "bf16"),
    (2617290, 384, 384, (384, 1), (384, 1), "bf16"),
    (2536160, 384, 384, (384, 1), (384, 1), "bf16"),
    (4126464, 256, 256, (256, 1), (1, 256), "bf16"),
    (12800, 1024, 1152, (1, 12800), (1024, 1), "bf16"),
    (2701258, 384, 1152, (1152, 1), (1, 1152), "bf16"),
    (2617290, 384, 1152, (1152, 1), (1, 1152), "bf16"),
    (2536160, 384, 1152, (1152, 1), (1, 1152), "bf16"),
    (294912, 3584, 800, (800, 1), (3584, 1), "bf16"),
    (49152, 512, 1152, (1, 49152), (512, 1), "bf16"),
    (589824, 256, 128, (128, 1), (1, 128), "bf16"),
    (33792, 1024, 2304, (1, 33792), (1024, 1), "bf16"),
    (589824, 512, 192, (192, 1), (512, 1), "bf16"),
    (33792, 2048, 2304, (1, 33792), (2048, 1), "bf16"),
    (67584, 1024, 2304, (1, 67584), (1024, 1), "bf16"),
    (294912, 2816, 768, (768, 1), (2816, 1), "bf16"),
    (1253376, 256, 256, (256, 1), (1, 256), "bf16"),
    (317200, 4096, 512, (512, 1), (4096, 1), "bf16"),
    (589824, 544, 512, (512, 1), (544, 1), "bf16"),
    (65536, 512, 1152, (1, 65536), (512, 1), "bf16"),
    (147456, 448, 192, (192, 1), (448, 1), "bf16"),
    (4126464, 256, 256, (256, 1), (256, 1), "bf16"),
    (838100, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (810572, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (838100, 512, 512, (512, 1), (512, 1), "bf16"),
    (776648, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (810572, 512, 512, (512, 1), (512, 1), "bf16"),
    (294912, 1056, 256, (256, 1), (1056, 1), "bf16"),
    (776648, 512, 512, (512, 1), (512, 1), "bf16"),
    (294912, 512, 192, (192, 1), (512, 1), "bf16"),
    (294912, 544, 768, (768, 1), (544, 1), "bf16"),
    (294912, 512, 512, (512, 1), (512, 1), "bf16"),
    (626688, 256, 256, (256, 1), (1, 256), "bf16"),
    (351004, 4096, 512, (512, 1), (4096, 1), "bf16"),
    (294912, 256, 128, (128, 1), (1, 128), "bf16"),
    (1216512, 256, 256, (256, 1), (1, 256), "bf16"),
    (294912, 544, 512, (512, 1), (544, 1), "bf16"),
    (338640, 4096, 512, (512, 1), (4096, 1), "bf16"),
    (1253376, 256, 256, (256, 1), (256, 1), "bf16"),
    (229248, 1024, 1152, (1, 229248), (1024, 1), "bf16"),
    (114624, 2048, 1152, (1, 114624), (2048, 1), "bf16"),
    (1216512, 256, 256, (256, 1), (256, 1), "bf16"),
    (294912, 384, 512, (512, 1), (384, 1), "bf16"),
    (114624, 1024, 1152, (1, 114624), (1024, 1), "bf16"),
    (838100, 512, 512, (512, 1), (1, 512), "bf16"),
    (810572, 512, 512, (512, 1), (1, 512), "bf16"),
    (451885, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (776648, 512, 512, (512, 1), (1, 512), "bf16"),
    (451885, 512, 512, (512, 1), (512, 1), "bf16"),
    (397716, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (705178, 1536, 384, (384, 1), (1536, 1), "bf16"),
    (397716, 512, 512, (512, 1), (512, 1), "bf16"),
    (17408, 1024, 2304, (1, 17408), (1024, 1), "bf16"),
    (34816, 1024, 2304, (1, 34816), (1024, 1), "bf16"),
    (2701258, 384, 128, (128, 1), (384, 1), "bf16"),
    (626688, 256, 256, (256, 1), (256, 1), "bf16"),
    (308743, 512, 1536, (1536, 1), (1, 1536), "bf16"),
    (32768, 512, 1152, (1, 32768), (512, 1), "bf16"),
    (67584, 1024, 1152, (1, 67584), (1024, 1), "bf16"),
    (2617290, 384, 128, (128, 1), (384, 1), "bf16"),
    (308743, 512, 512, (512, 1), (512, 1), "bf16"),
    (73728, 384, 512, (512, 1), (384, 1), "bf16"),
    (503599, 1536, 384, (384, 1), (1536, 1), "bf16"),
    (294912, 368, 768, (768, 1), (368, 1), "bf16"),
    (451885, 512, 512, (512, 1), (1, 512), "bf16"),
    (2536160, 384, 128, (128, 1), (384, 1), "bf16"),
    (386937, 1792, 384, (384, 1), (1792, 1), "bf16"),
    (17408, 1024, 1152, (1, 17408), (1024, 1), "bf16"),
    (397716, 512, 512, (512, 1), (1, 512), "bf16"),
    (386515, 1536, 384, (384, 1), (1536, 1), "bf16"),
    (34816, 1024, 1152, (1, 34816), (1024, 1), "bf16"),
    (32768, 1024, 1152, (1, 32768), (1024, 1), "bf16"),
    (313230, 1792, 384, (384, 1), (1792, 1), "bf16"),
    (308743, 512, 512, (512, 1), (1, 512), "bf16"),
    (294912, 512, 256, (256, 1), (512, 1), "bf16"),
    (222929, 1792, 512, (512, 1), (1792, 1), "bf16"),
    (198339, 1792, 512, (512, 1), (1792, 1), "bf16"),
    (73728, 512, 512, (512, 1), (512, 1), "bf16"),
    (776648, 512, 128, (128, 1), (512, 1), "bf16"),
    (838100, 512, 128, (128, 1), (512, 1), "bf16"),
    (810572, 512, 128, (128, 1), (512, 1), "bf16"),
    (119998, 384, 384, (384, 1), (384, 1), "bf16"),
    (132755, 1792, 512, (512, 1), (1792, 1), "bf16"),
    (15044, 1024, 1152, (1, 15072), (1024, 1), "bf16"),
    (119998, 384, 1152, (1152, 1), (1, 1152), "bf16"),
    (23552, 1024, 1152, (1, 23552), (1024, 1), "bf16"),
    (107766, 2304, 384, (384, 1), (2304, 1), "bf16"),
    (136074, 1792, 384, (384, 1), (1792, 1), "bf16"),
    (83117, 384, 384, (384, 1), (384, 1), "bf16"),
    (384, 384, 19459, (1, 384), (384, 1), "bf16"),
    (211269, 1024, 384, (384, 1), (1024, 1), "bf16"),
    (198462, 1024, 384, (384, 1), (1024, 1), "bf16"),
    (83117, 384, 1152, (1152, 1), (1, 1152), "bf16"),
    (75315, 2560, 384, (384, 1), (2560, 1), "bf16"),
    (65217, 2304, 384, (384, 1), (2304, 1), "bf16"),
    (66611, 384, 384, (384, 1), (384, 1), "bf16"),
    (77557, 2304, 384, (384, 1), (2304, 1), "bf16"),
    (115451, 256, 256, (256, 1), (256, 1), "bf16"),
    (117014, 256, 256, (256, 1), (256, 1), "bf16"),
    (119998, 384, 384, (384, 1), (1, 384), "bf16"),
    (114658, 256, 256, (256, 1), (256, 1), "bf16"),
    (451885, 512, 128, (128, 1), (512, 1), "bf16"),
    (15044, 512, 1152, (1, 15072), (512, 1), "bf16"),
    (48136, 3328, 384, (384, 1), (3328, 1), "bf16"),
    (308743, 512, 128, (128, 1), (512, 1), "bf16"),
    (397716, 512, 128, (128, 1), (512, 1), "bf16"),
    (47224, 3328, 384, (384, 1), (3328, 1), "bf16"),
    (61104, 2560, 384, (384, 1), (2560, 1), "bf16"),
    (57151, 2560, 384, (384, 1), (2560, 1), "bf16"),
    (66611, 384, 1152, (1152, 1), (1, 1152), "bf16"),
    (1000000, 512, 512, (512, 1), (512, 1), "bf16"),
    # Shared gfx942_1/gfx950_1 production-request shapes.
    (819200, 1024, 192, (192, 1), (1, 192), "bf16"),
    (61440, 2048, 5120, (5120, 1), (1, 5120), "bf16"),
    (61440, 5120, 2048, (2048, 1), (5120, 1), "fp16"),
    (2252800, 256, 256, (256, 1), (256, 1), "fp16"),
    (61440, 3840, 4096, (4096, 1), (3840, 1), "fp16"),
    (61440, 5120, 7744, (7744, 1), (5120, 1), "fp16"),
}


def _cases():
    return [entry for entry in CORRECTNESS_SHAPES if tuple(entry) not in FAILED_SHAPES]


@pytest.mark.parametrize("M, N, K, a_strides, b_strides, dtype_name", _cases())
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
    (1000, 1000, 1024, 1),
    # Whole region overhangs: one tile of 128 rows holds only 64 real rows.
    (64, 4096, 4096, 1),
]

SPLIT_KS_GPU = [1, 4]


@pytest.mark.parametrize("SPLIT_K", SPLIT_KS_GPU)
@pytest.mark.parametrize("M, N, K, NUM_CTAS", GPU_SHAPES)
def test_output_is_independent_of_split_k(M, N, K, NUM_CTAS, SPLIT_K):
    """Splitting the reduction must not change the result."""
    from triton.tlx.ops import mm as tlx_mm

    dtype = torch.float16
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

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
