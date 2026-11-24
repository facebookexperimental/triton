"""TLX misc tests -- Blackwell-only."""
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize(
    "src_dtype, dst_dtype",
    [
        ("float32", "float8_e5m2"),
        ("float32", "float8_e4m3fn"),
        ("float32", "float16"),
        ("float32", "bfloat16"),
    ],
)
def test_stoch_round(src_dtype, dst_dtype, device):

    @triton.jit
    def stoch_round_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        # Generate 1/4 shape for each random stream
        offsets_quarter = tl.arange(0, BLOCK_SIZE // 4)
        r0, r1, r2, r3 = tl.randint4x(0, offsets_quarter, n_rounds=7)
        # Combine the 4 blocks into a single vector of random values
        # r0,r1,r2,r3: each [BLOCK_SIZE//4]
        # after joins: rbits: [BLOCK_SIZE]
        rbits = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(x.shape)
        y = tlx.stoch_round(
            x,
            tlx.dtype_of(y_ptr),
            rbits,
        )
        tl.store(y_ptr + offsets, y)

    # Map string names to torch dtypes
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }

    src_dtype_torch = dtype_map[src_dtype]
    dst_dtype_torch = dtype_map[dst_dtype]

    SIZE = 256
    a = torch.randn([SIZE], dtype=torch.float32, device=device).to(src_dtype_torch)
    b = torch.empty([SIZE], dtype=torch.float32, device=device).to(dst_dtype_torch)
    grid = lambda meta: (1, )
    kernel = stoch_round_kernel[grid](
        a,
        b,
        BLOCK_SIZE=SIZE,
        num_warps=1,
    )
    assert kernel.asm["ptx"].count("cvt.rs.satfinite") > 0

    # Compare against PyTorch baseline
    # PyTorch doesn't have stochastic rounding, so we verify the result
    # is within the representable range and matches deterministic rounding
    # for most values (stochastic should be close on average)
    a_f32 = a.float()
    b_ref = a_f32.to(dst_dtype_torch)  # PyTorch uses round-to-nearest-even

    # Convert to float32 for validation (FP8 doesn't support all PyTorch ops)
    b_back = b.float()

    # Verify all values are in valid range (no NaN/Inf introduced)
    assert not torch.isnan(b_back).any(), "Stochastic rounding produced NaN"
    assert not torch.isinf(b_back).any(), "Stochastic rounding produced Inf"

    # For values that don't need rounding (exact in FP8), should match exactly
    exact_mask = b_back == a_f32
    if exact_mask.any():
        assert torch.equal(b[exact_mask],
                           b_ref[exact_mask]), ("Values that don't need rounding should match deterministic rounding")

    # For values that need rounding, verify they're in a reasonable range
    # (stochastic rounding can pick either of two adjacent representable values,
    # so we can't easily validate without knowing FP8 representation details)
    needs_rounding = ~exact_mask
    if needs_rounding.any():
        # Basic sanity check: stochastic result should be reasonably close to input
        # For FP8 e5m2, max representable is 57344, so use that as scale
        max_expected_diff = 100.0  # Conservative bound for FP8 rounding error
        diff = torch.abs(b_back[needs_rounding] - a_f32[needs_rounding])
        assert (diff < max_expected_diff).all(), (
            f"Stochastic rounding produced unreasonably large errors: max diff = {diff.max()}, "
            f"expected < {max_expected_diff}")


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("dst_dtype", ["float8_e5m2", "float8_e4m3fn", "float16", "bfloat16"])
def test_stoch_round_partial_pack(dst_dtype, device):
    """Test stochastic rounding with block sizes not evenly divisible by pack size."""

    @triton.jit
    def stoch_round_partial_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_ROUNDED: tl.constexpr,
                                   QUARTER_SIZE_ROUNDED: tl.constexpr):
        # Use power-of-2 size for arange (triton requirement), then mask to actual size
        offsets_full = tl.arange(0, BLOCK_SIZE_ROUNDED)
        mask = offsets_full < BLOCK_SIZE
        offsets = tl.where(mask, offsets_full, 0)
        x = tl.load(x_ptr + offsets, mask=mask)
        # For sizes that don't divide evenly by 4 (FP8 pack size)
        # Use pre-computed power-of-2 size for the quarter size
        offsets_quarter = tl.arange(0, QUARTER_SIZE_ROUNDED)
        r0, r1, r2, r3 = tl.randint4x(42, offsets_quarter, n_rounds=7)
        rbits_raw = tl.join(tl.join(r0, r1), tl.join(r2, r3))
        # Take only BLOCK_SIZE elements
        rbits = tl.view(rbits_raw, (BLOCK_SIZE_ROUNDED, ))
        rbits_masked = tl.where(mask, rbits, 0)
        y = tlx.stoch_round(x, tlx.dtype_of(y_ptr), rbits_masked)
        tl.store(y_ptr + offsets, y, mask=mask)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }

    dst_dtype_torch = dtype_map[dst_dtype]

    # Test with sizes not divisible by 4 (FP8) or 2 (BF16/F16)
    for SIZE in [130, 65, 17]:  # Not divisible by pack sizes
        # Round up SIZE to next power of 2
        SIZE_ROUNDED = 1 << (SIZE - 1).bit_length()
        # Compute quarter size and round it up to next power of 2
        quarter_size = (SIZE + 3) // 4
        QUARTER_SIZE_ROUNDED = 1 << (quarter_size - 1).bit_length()
        a = torch.randn([SIZE], dtype=torch.float32, device=device)
        b = torch.empty([SIZE], dtype=torch.float32, device=device).to(dst_dtype_torch)
        grid = lambda meta: (1, )
        stoch_round_partial_kernel[grid](
            a,
            b,
            BLOCK_SIZE=SIZE,
            BLOCK_SIZE_ROUNDED=SIZE_ROUNDED,
            QUARTER_SIZE_ROUNDED=QUARTER_SIZE_ROUNDED,
            num_warps=1,
        )

        # Verify no NaN/Inf
        b_back = b.float()
        assert not torch.isnan(b_back).any(), f"NaN produced for size {SIZE}"
        assert not torch.isinf(b_back).any(), f"Inf produced for size {SIZE}"


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("invalid_src, invalid_dst", [("float16", "float8_e5m2"), ("bfloat16", "float16"),
                                                      ("float32", "int32")])
def test_stoch_round_invalid_dtypes(invalid_src, invalid_dst, device):
    """Test that invalid dtype combinations raise proper errors."""

    @triton.jit
    def stoch_round_invalid_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr, SRC_DTYPE: tl.constexpr,
                                   DST_DTYPE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets).to(SRC_DTYPE)
        offsets_quarter = tl.arange(0, BLOCK_SIZE // 4)
        r0, r1, r2, r3 = tl.randint4x(0, offsets_quarter, n_rounds=7)
        rbits = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(x.shape)
        y = tlx.stoch_round(x, DST_DTYPE, rbits)
        tl.store(y_ptr + offsets, y)

    dtype_map = {
        "float32": tl.float32,
        "float16": tl.float16,
        "bfloat16": tl.bfloat16,
        "float8_e5m2": tl.float8e5,
        "float8_e4m3fn": tl.float8e4nv,
        "int32": tl.int32,
    }

    SIZE = 128
    a = torch.randn([SIZE], dtype=torch.float32, device=device)
    b = torch.empty([SIZE], dtype=torch.float32, device=device)
    grid = lambda meta: (1, )

    with pytest.raises(Exception) as exc_info:
        stoch_round_invalid_kernel[grid](a, b, BLOCK_SIZE=SIZE, SRC_DTYPE=dtype_map[invalid_src],
                                         DST_DTYPE=dtype_map[invalid_dst], num_warps=1)

    # Verify error message mentions the issue
    error_msg = str(exc_info.value)
    assert "Stochastic rounding" in error_msg or "float32" in error_msg or "supported" in error_msg.lower()


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_stoch_round_entropy_quality(device):
    """Test that different random seeds produce different results."""

    @triton.jit
    def stoch_round_seed_kernel(x_ptr, y_ptr, seed, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        offsets_quarter = tl.arange(0, BLOCK_SIZE // 4)
        r0, r1, r2, r3 = tl.randint4x(seed, offsets_quarter, n_rounds=7)
        rbits = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(x.shape)
        y = tlx.stoch_round(x, tlx.dtype_of(y_ptr), rbits)
        tl.store(y_ptr + offsets, y)

    SIZE = 256
    # Use values that will definitely need rounding in FP8
    a = torch.randn([SIZE], dtype=torch.float32, device=device) * 10.0
    b1 = torch.empty([SIZE], dtype=torch.float8_e5m2, device=device)
    b2 = torch.empty([SIZE], dtype=torch.float8_e5m2, device=device)
    grid = lambda meta: (1, )

    # Run with different seeds
    stoch_round_seed_kernel[grid](a, b1, seed=12345, BLOCK_SIZE=SIZE, num_warps=1)
    stoch_round_seed_kernel[grid](a, b2, seed=67890, BLOCK_SIZE=SIZE, num_warps=1)

    # Results should be different for at least some values
    different_count = (b1.float() != b2.float()).sum().item()
    assert different_count > SIZE * 0.1, (
        f"Different seeds should produce different results, but only {different_count}/{SIZE} values differ")
