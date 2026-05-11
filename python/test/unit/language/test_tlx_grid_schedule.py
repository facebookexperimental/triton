"""Tests for tlx.grid_schedule: GridSchedule enum and compute_grid."""

import math
import pytest

from triton.third_party.tlx.language.tlx.grid_schedule import (
    GridSchedule,
    compute_grid,
    _get_max_num_sms,
)


# ---------------------------------------------------------------------------
# GridSchedule enum
# ---------------------------------------------------------------------------


class TestGridScheduleEnum:
    def test_values(self):
        assert GridSchedule.NON_PERSISTENT.value == "non_persistent"
        assert GridSchedule.STATIC_PERSISTENT.value == "static_persistent"
        assert GridSchedule.CLC.value == "clc"

    def test_members(self):
        assert set(GridSchedule) == {
            GridSchedule.NON_PERSISTENT,
            GridSchedule.STATIC_PERSISTENT,
            GridSchedule.CLC,
        }

    def test_from_value(self):
        assert GridSchedule("non_persistent") is GridSchedule.NON_PERSISTENT
        assert GridSchedule("static_persistent") is GridSchedule.STATIC_PERSISTENT
        assert GridSchedule("clc") is GridSchedule.CLC


# ---------------------------------------------------------------------------
# compute_grid — Non-Persistent
# ---------------------------------------------------------------------------


class TestComputeGridNonPersistent:
    """NON_PERSISTENT: grid = (total_tiles,)."""

    def test_basic(self):
        grid = compute_grid(
            GridSchedule.NON_PERSISTENT,
            total_tiles_fn=lambda: 200,
        )
        assert grid == (200,)

    def test_single_tile(self):
        grid = compute_grid(
            GridSchedule.NON_PERSISTENT,
            total_tiles_fn=lambda: 1,
        )
        assert grid == (1,)

    def test_ignores_num_sms_fn(self):
        """num_sms_fn should never be called for NON_PERSISTENT."""
        grid = compute_grid(
            GridSchedule.NON_PERSISTENT,
            total_tiles_fn=lambda: 500,
            num_sms_fn=lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        assert grid == (500,)

    def test_gemm_example(self):
        """Usage example 4.1: Non-Persistent GEMM."""
        M, N, BLOCK_M, BLOCK_N = 1024, 2048, 128, 128
        grid = compute_grid(
            GridSchedule.NON_PERSISTENT,
            total_tiles_fn=lambda: math.ceil(M / BLOCK_M) * math.ceil(N / BLOCK_N),
        )
        assert grid == (8 * 16,)  # 128 tiles


# ---------------------------------------------------------------------------
# compute_grid — Static Persistent
# ---------------------------------------------------------------------------


class TestComputeGridStaticPersistent:
    """STATIC_PERSISTENT: grid = (min(num_sms, total_tiles),)."""

    def test_tiles_exceed_sms(self):
        grid = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: 500,
            num_sms_fn=lambda: 148,
        )
        assert grid == (148,)

    def test_tiles_fewer_than_sms(self):
        grid = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: 64,
            num_sms_fn=lambda: 148,
        )
        assert grid == (64,)

    def test_tiles_equal_sms(self):
        grid = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: 148,
            num_sms_fn=lambda: 148,
        )
        assert grid == (148,)

    def test_single_tile(self):
        grid = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: 1,
            num_sms_fn=lambda: 148,
        )
        assert grid == (1,)

    def test_gemm_split_k_2cta(self):
        """Usage example 4.2: Static Persistent GEMM with Split-K and 2-CTA."""
        M, N, K = 4096, 4096, 8192
        BLOCK_M, BLOCK_N = 256, 256
        NUM_CTAS = 2
        SPLIT_K = 4
        NUM_SMS = 148

        def total_tiles():
            num_m = math.ceil(M / BLOCK_M)  # 16
            num_n = math.ceil(N / BLOCK_N)  # 16
            # Pad M-tiles for 2-CTA cluster alignment
            num_m = ((num_m + NUM_CTAS - 1) // NUM_CTAS) * NUM_CTAS  # 16 (already aligned)
            return num_m * num_n * SPLIT_K  # 16 * 16 * 4 = 1024

        grid = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=total_tiles,
            num_sms_fn=lambda: NUM_SMS,
        )
        assert grid == (148,)  # min(148, 1024) = 148

    def test_sm_pruning_percentage(self):
        """Usage example 4.6: Explicit SM pruning (75%)."""
        grid = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: 500,
            num_sms_fn=lambda: int(148 * 0.75),
        )
        assert grid == (111,)  # int(148 * 0.75) = 111

    def test_sm_pruning_fixed(self):
        """Usage example 4.6: Fixed SM count."""
        grid = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: 500,
            num_sms_fn=lambda: 128,
        )
        assert grid == (128,)


# ---------------------------------------------------------------------------
# compute_grid — CLC
# ---------------------------------------------------------------------------


class TestComputeGridCLC:
    """CLC: grid = (total_tiles,)."""

    def test_basic(self):
        grid = compute_grid(
            GridSchedule.CLC,
            total_tiles_fn=lambda: 200,
        )
        assert grid == (200,)

    def test_same_as_non_persistent_from_host(self):
        """CLC and NON_PERSISTENT produce the same grid from the host side."""
        tiles = 300
        grid_clc = compute_grid(
            GridSchedule.CLC,
            total_tiles_fn=lambda: tiles,
        )
        grid_np = compute_grid(
            GridSchedule.NON_PERSISTENT,
            total_tiles_fn=lambda: tiles,
        )
        assert grid_clc == grid_np

    def test_ignores_num_sms_fn(self):
        """num_sms_fn should never be called for CLC."""
        grid = compute_grid(
            GridSchedule.CLC,
            total_tiles_fn=lambda: 500,
            num_sms_fn=lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        assert grid == (500,)

    def test_gemm_example(self):
        """Usage example 4.3: CLC GEMM."""
        M, N, BLOCK_M, BLOCK_N = 4096, 4096, 128, 256
        grid = compute_grid(
            GridSchedule.CLC,
            total_tiles_fn=lambda: math.ceil(M / BLOCK_M) * math.ceil(N / BLOCK_N),
        )
        assert grid == (32 * 16,)  # 512 tiles


# ---------------------------------------------------------------------------
# compute_grid — Autotuning pattern
# ---------------------------------------------------------------------------


class TestComputeGridAutotuning:
    """Usage example 4.4: Autotuned Persistent FA."""

    def test_autotuned_fa_pattern(self):
        seq_len, batch, heads = 2048, 4, 32
        NUM_SMS = 148

        # Simulate the autotuning META dict
        META = {"BLOCK_M": 128}

        grid = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: math.ceil(seq_len / META["BLOCK_M"]) * batch * heads,
            num_sms_fn=lambda: NUM_SMS,
        )
        # cdiv(2048, 128) * 4 * 32 = 16 * 128 = 2048 tiles -> min(148, 2048) = 148
        assert grid == (148,)

    def test_autotuned_grid_lambda(self):
        """The grid lambda pattern for @triton.autotune."""
        seq_len, batch, heads = 2048, 4, 32
        NUM_SMS = 148

        # This is how it would be used with autotune:
        # grid = lambda META: compute_grid(...)
        grid_fn = lambda META: compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: math.ceil(seq_len / META["BLOCK_M"]) * batch * heads,
            num_sms_fn=lambda: NUM_SMS,
        )

        assert grid_fn({"BLOCK_M": 128}) == (148,)
        assert grid_fn({"BLOCK_M": 256}) == (148,)  # cdiv(2048,256)*128 = 1024 > 148
        assert grid_fn({"BLOCK_M": 2048}) == (128,)  # cdiv(2048,2048)*128 = 128 < 148


# ---------------------------------------------------------------------------
# compute_grid — Schedule switching
# ---------------------------------------------------------------------------


class TestComputeGridScheduleSwitching:
    """Usage example 4.5: Switching schedule types."""

    def test_switch_persistent_to_clc(self):
        num_m, num_n = 16, 16
        NUM_SMS = 148

        grid_persistent = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: num_m * num_n,
            num_sms_fn=lambda: NUM_SMS,
        )
        grid_clc = compute_grid(
            GridSchedule.CLC,
            total_tiles_fn=lambda: num_m * num_n,
        )

        assert grid_persistent == (148,)  # min(148, 256)
        assert grid_clc == (256,)  # all tiles

    def test_switch_non_persistent_to_persistent(self):
        tiles = 64
        NUM_SMS = 148

        grid_np = compute_grid(
            GridSchedule.NON_PERSISTENT,
            total_tiles_fn=lambda: tiles,
        )
        grid_sp = compute_grid(
            GridSchedule.STATIC_PERSISTENT,
            total_tiles_fn=lambda: tiles,
            num_sms_fn=lambda: NUM_SMS,
        )

        assert grid_np == (64,)
        assert grid_sp == (64,)  # fewer tiles than SMs


# ---------------------------------------------------------------------------
# compute_grid — Error handling
# ---------------------------------------------------------------------------


class TestComputeGridErrors:
    def test_invalid_schedule_string(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            compute_grid(
                "not_a_schedule",  # type: ignore
                total_tiles_fn=lambda: 100,
            )


# ---------------------------------------------------------------------------
# _get_max_num_sms
# ---------------------------------------------------------------------------


class TestGetMaxNumSms:
    def test_returns_positive_int(self):
        """_get_max_num_sms should return a positive integer on any CUDA device."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("No CUDA device available")
            result = _get_max_num_sms()
            assert isinstance(result, int)
            assert result > 0
        except ImportError:
            pytest.skip("torch not available")

    def test_is_cached(self):
        """Repeated calls should return the same object (lru_cache)."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("No CUDA device available")
            a = _get_max_num_sms()
            b = _get_max_num_sms()
            assert a == b
        except ImportError:
            pytest.skip("torch not available")


# ---------------------------------------------------------------------------
# compute_grid — default num_sms_fn
# ---------------------------------------------------------------------------


class TestComputeGridDefaultNumSms:
    def test_static_persistent_uses_default(self):
        """STATIC_PERSISTENT should work without explicit num_sms_fn on CUDA."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("No CUDA device available")
            grid = compute_grid(
                GridSchedule.STATIC_PERSISTENT,
                total_tiles_fn=lambda: 10000,
            )
            # Should be capped at the device SM count
            assert grid[0] == _get_max_num_sms()
        except ImportError:
            pytest.skip("torch not available")
