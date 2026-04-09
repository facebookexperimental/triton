import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_blackwell
import triton.language.extra.tlx as tlx
import traceback


def test_thread_id(device):

    @triton.jit
    def store_from_thread_0_kernel(
        output_ptr,
        value,
        n_elements,
        axis: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tid = tlx.thread_id(axis)
        if tid == 0:
            tl.store(output_ptr + offsets, value, mask=mask)

    output = torch.zeros(32, dtype=torch.int32, device="cuda")
    n_elements = output.numel()
    value = 42
    store_from_thread_0_kernel[(1, )](output, value, n_elements, 0, 32, num_warps=1)
    torch.cuda.synchronize()
    expected_output = torch.zeros(32, dtype=torch.int32, device="cuda")
    expected_output[0] = value
    torch.testing.assert_close(output, expected_output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_clock64(device):

    @triton.jit
    def clock64_from_thread_0_kernel(
        output_ptr,
        value,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tid = tlx.thread_id(0)
        if pid == 0 and tid == 0:
            start = tlx.clock64()
            tl.store(output_ptr + offsets, value, mask=mask)
            end = tlx.clock64()
            tl.device_print("Cycles elapsed: ", end - start)

    output = torch.zeros(32, dtype=torch.int32, device="cuda")
    n_elements = output.numel()
    value = 42
    kernel = clock64_from_thread_0_kernel[(1, )](output, value, n_elements, 32, num_warps=1)
    assert kernel.asm["ttgir"].count("ttg.clock64") == 2
    assert kernel.asm["ptx"].count("%clock64") == 2


def test_loop_carry_var_check(device):

    @triton.jit
    def loop_carry_shadow():
        x = tlx.local_alloc((16, 16), tl.int16, tl.constexpr(2))
        y = x
        for _ in range(0, 128):
            zeros = tl.zeros((16, 16), dtype=tl.int16)
            # shadow x with different type
            x = tlx.local_view(y, 0)
            tlx.local_store(x, zeros)

    grid = lambda meta: (1, 1)

    with pytest.raises(triton.CompilationError) as e:
        loop_carry_shadow[grid]()
    list_msg = traceback.format_exception(e.type, e.value, e.tb, chain=True)
    assert "Please make sure that the type stays consistent" in "\n".join(list_msg)


def test_size_of(device):

    @triton.jit
    def size_of_kernel(output_ptr):
        # Test size_of for various dtypes
        size_fp32 = tlx.size_of(tl.float32)
        size_fp16 = tlx.size_of(tl.float16)
        size_int32 = tlx.size_of(tl.int32)
        size_int8 = tlx.size_of(tl.int8)
        size_int64 = tlx.size_of(tl.int64)

        # Store results
        tl.store(output_ptr + 0, size_fp32)
        tl.store(output_ptr + 1, size_fp16)
        tl.store(output_ptr + 2, size_int32)
        tl.store(output_ptr + 3, size_int8)
        tl.store(output_ptr + 4, size_int64)

    # Expected sizes in bytes
    expected_sizes = torch.tensor([4, 2, 4, 1, 8], dtype=torch.int32, device=device)
    output = torch.zeros(5, dtype=torch.int32, device=device)

    grid = lambda meta: (1, )
    size_of_kernel[grid](output)

    torch.testing.assert_close(output, expected_sizes)


def test_size_of_constexpr(device):

    @triton.jit
    def size_of_constexpr_kernel(output_ptr, DTYPE: tl.constexpr):
        # Test size_of with constexpr dtype argument
        size = tlx.size_of(DTYPE)
        tl.store(output_ptr, size)

    output = torch.zeros(1, dtype=torch.int32, device=device)

    # Test with float32 (4 bytes)
    grid = lambda meta: (1, )
    size_of_constexpr_kernel[grid](output, tl.float32)
    assert output.item() == 4, f"Expected 4 for float32, got {output.item()}"

    # Test with float16 (2 bytes)
    size_of_constexpr_kernel[grid](output, tl.float16)
    assert output.item() == 2, f"Expected 2 for float16, got {output.item()}"

    # Test with int8 (1 byte)
    size_of_constexpr_kernel[grid](output, tl.int8)
    assert output.item() == 1, f"Expected 1 for int8, got {output.item()}"

    # Test with int64 (8 bytes)
    size_of_constexpr_kernel[grid](output, tl.int64)
    assert output.item() == 8, f"Expected 8 for int64, got {output.item()}"


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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_buffer_indexing_in_function_call(device):
    """Test that buffer indexing with [] syntax works correctly in function calls"""

    @triton.jit
    def helper_function(buffers, idx, data):
        """Helper function that receives buffers and performs indexing inside"""
        tlx.local_store(buffers[idx], data)  # Indexing happens inside the helper
        result = tlx.local_load(buffers[idx])  # Indexing again
        return result

    @triton.jit
    def kernel_with_indexing(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Allocate buffer with multiple stages
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, num=tl.constexpr(4))

        # Load data
        x = tl.load(x_ptr + offsets, mask=mask)

        # Pass buffers to helper function which performs ALL indexing
        result = helper_function(buffers, 0, x)

        # Store result
        tl.store(y_ptr + offsets, result, mask=mask)

    torch.manual_seed(0)
    size = 1024
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(size, BLOCK_SIZE), )
    kernel_with_indexing[grid](x, y, size, BLOCK_SIZE)

    # Verify correctness
    assert torch.allclose(y, x), "Buffer indexing in function call failed"


@triton.jit
def _test_get_fp8_format_name_kernel(
    output_ptr,
    DTYPE: tl.constexpr,
    EXPECTED: tl.constexpr,
):
    result: tl.constexpr = tlx.get_fp8_format_name(DTYPE)
    if result == EXPECTED:
        tl.store(output_ptr, 1)
    else:
        tl.store(output_ptr, 0)


@triton.jit
def _test_get_fp8_format_name_unsupported_kernel(
    output_ptr,
    DTYPE: tl.constexpr,
):
    result: tl.constexpr = tlx.get_fp8_format_name(DTYPE)
    tl.store(output_ptr, result == "e5m2")


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (tl.float8e5, "e5m2"),
        (tl.float8e4nv, "e4m3"),
    ],
)
def test_get_fp8_format_name(dtype, expected, device):
    """Test that FP8 dtypes return correct format strings."""
    output = torch.zeros(1, dtype=torch.int32, device=device)
    _test_get_fp8_format_name_kernel[(1, )](output, DTYPE=dtype, EXPECTED=expected)
    assert output.item() == 1


@pytest.mark.parametrize(
    "dtype",
    [
        tl.float32,
        tl.float16,
        tl.int32,
    ],
)
def test_get_fp8_format_name_unsupported_dtype_raises_error(dtype, device):
    """Test that non-FP8 dtypes raise a CompilationError during compilation."""
    output = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(triton.CompilationError) as exc_info:
        _test_get_fp8_format_name_unsupported_kernel[(1, )](output, DTYPE=dtype)
    # Check that the underlying cause mentions the supported types
    assert "only supports tl.float8e5" in str(exc_info.value.__cause__)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_vote_ballot_sync(device):
    """Test vote_ballot_sync TLX operation for warp-level voting."""

    @triton.jit
    def vote_ballot_kernel(
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each thread's lane ID (use x-axis thread ID)
        tid = tlx.thread_id(0)

        # Create a predicate: lanes 0-15 vote True, lanes 16-31 vote False
        pred = tid < 16

        # Perform warp-level ballot vote
        # 0xFFFFFFFF means all 32 threads in the warp participate
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)

        # Store the ballot result from thread 0 only
        if tid == 0:
            tl.store(output_ptr, ballot_result)

    output = torch.zeros(1, dtype=torch.int32, device=device)

    # Run the kernel with 1 warp
    vote_ballot_kernel[(1, )](output, BLOCK_SIZE=32, num_warps=1)
    torch.cuda.synchronize()

    # Expected ballot result: threads 0-15 have pred=True, threads 16-31 have pred=False
    # So ballot should be 0x0000FFFF (lower 16 bits set)
    expected_ballot = 0x0000FFFF
    assert output.item() == expected_ballot, f"Expected {hex(expected_ballot)}, got {hex(output.item())}"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_vote_ballot_sync_ir_emission(device):
    """Test that vote_ballot_sync generates the correct IR."""

    @triton.jit
    def vote_ballot_ir_kernel(output_ptr, ):
        tid = tlx.thread_id(0)
        pred = tid < 16  # First 16 threads True
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
        if tid == 0:
            tl.store(output_ptr, ballot_result)

    output = torch.zeros(1, dtype=torch.int32, device=device)
    kernel = vote_ballot_ir_kernel[(1, )](output, num_warps=1)

    # Verify the TTGIR contains the vote_ballot_sync op
    ttgir = kernel.asm["ttgir"]
    assert "vote_ballot_sync" in ttgir, "Expected vote_ballot_sync in TTGIR"

    # Verify the LLVM IR contains the NVVM vote instruction
    llir = kernel.asm["llir"]
    assert "nvvm.vote.ballot.sync" in llir or "vote.sync.ballot" in llir, (
        "Expected nvvm.vote.ballot.sync or vote.sync.ballot in LLVM IR")


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("CHUNK_SIZE", [256, 1024])
def test_async_bulk_copy_roundtrip(CHUNK_SIZE, device):
    """Test gmem->smem->gmem roundtrip using async_load(bulk=True) and async_store."""

    @triton.jit
    def bulk_copy_kernel(
        src_ptr,
        dst_ptr,
        CHUNK_SIZE: tl.constexpr,
    ):
        smem = tlx.local_alloc((CHUNK_SIZE, ), tl.uint8, num=1)
        bars = tlx.alloc_barriers(1, arrive_count=1)
        bar = bars[0]
        buf = smem[0]

        # gmem -> smem (bulk async_load)
        tlx.barrier_expect_bytes(bar, CHUNK_SIZE)
        tlx.async_load(src_ptr, buf, bulk=True, barrier=bar)
        tlx.barrier_wait(bar, 0)

        # smem -> gmem
        tlx.async_store(dst_ptr, buf, CHUNK_SIZE)
        tlx.async_descriptor_store_wait(0)

    size = CHUNK_SIZE
    src = torch.randint(0, 256, (size, ), dtype=torch.uint8, device=device)
    dst = torch.zeros(size, dtype=torch.uint8, device=device)

    kernel = bulk_copy_kernel[(1, )](src, dst, CHUNK_SIZE, num_warps=1)

    # Verify IR uses async_copy_global_to_local with bulk mode
    ttgir = kernel.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" in ttgir, "Expected async_copy_global_to_local in TTGIR"
    assert "useBulk = true" in ttgir, "Expected useBulk = true in TTGIR"
    assert "ttng.async_store" in ttgir, "Expected async_store in TTGIR"

    # Verify PTX contains the bulk copy instructions
    ptx = kernel.asm["ptx"]
    assert "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes" in ptx, (
        "Expected cp.async.bulk gmem->smem in PTX")
    assert "cp.async.bulk.global.shared::cta.bulk_group" in ptx, "Expected cp.async.bulk smem->gmem in PTX"

    # Verify correctness
    torch.testing.assert_close(src, dst)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("CHUNK_SIZE", [256, 1024])
def test_async_load_bulk(CHUNK_SIZE, device):
    """Test async_load with bulk=True (1D bulk copy via mbarrier)."""

    @triton.jit
    def bulk_load_kernel(
        src_ptr,
        dst_ptr,
        CHUNK_SIZE: tl.constexpr,
    ):
        smem = tlx.local_alloc((CHUNK_SIZE, ), tl.uint8, num=1)
        bars = tlx.alloc_barriers(1, arrive_count=1)
        bar = bars[0]
        buf = smem[0]

        # Bulk async_load: no explicit pred needed (auto-generated in lowering)
        tlx.barrier_expect_bytes(bar, CHUNK_SIZE)
        tlx.async_load(src_ptr, buf, bulk=True, barrier=bar)
        tlx.barrier_wait(bar, 0)

        # Write back to gmem via smem->gmem bulk copy
        tlx.async_store(dst_ptr, buf, CHUNK_SIZE)
        tlx.async_descriptor_store_wait(0)

    size = CHUNK_SIZE
    src = torch.randint(0, 256, (size, ), dtype=torch.uint8, device=device)
    dst = torch.zeros(size, dtype=torch.uint8, device=device)

    kernel = bulk_load_kernel[(1, )](src, dst, CHUNK_SIZE, num_warps=1)

    # Verify IR: should use async_copy_global_to_local with useBulk/bulk_size/barrier
    ttgir = kernel.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" in ttgir, "Expected async_copy_global_to_local in TTGIR"
    assert "bulk_size" in ttgir, "Expected bulk_size operand in TTGIR"
    assert "barrier" in ttgir, "Expected barrier operand in TTGIR"
    assert "useBulk = true" in ttgir, "Expected useBulk = true in TTGIR"

    # Verify PTX contains the bulk copy instruction
    ptx = kernel.asm["ptx"]
    assert "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes" in ptx, (
        "Expected cp.async.bulk gmem->smem in PTX")

    # Verify correctness
    torch.testing.assert_close(src, dst)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("CHUNK_SIZE", [256, 1024])
def test_async_load_bulk_auto_size(CHUNK_SIZE, device):
    """Test async_load bulk=True with explicit bulk_size parameter."""

    @triton.jit
    def bulk_load_explicit_size_kernel(
        src_ptr,
        dst_ptr,
        CHUNK_SIZE: tl.constexpr,
    ):
        smem = tlx.local_alloc((CHUNK_SIZE, ), tl.uint8, num=1)
        bars = tlx.alloc_barriers(1, arrive_count=1)
        bar = bars[0]
        buf = smem[0]

        # Pass explicit bulk_size
        tlx.barrier_expect_bytes(bar, CHUNK_SIZE)
        tlx.async_load(src_ptr, buf, bulk=True, bulk_size=CHUNK_SIZE, barrier=bar)
        tlx.barrier_wait(bar, 0)

        tlx.async_store(dst_ptr, buf, CHUNK_SIZE)
        tlx.async_descriptor_store_wait(0)

    size = CHUNK_SIZE
    src = torch.randint(0, 256, (size, ), dtype=torch.uint8, device=device)
    dst = torch.zeros(size, dtype=torch.uint8, device=device)

    kernel = bulk_load_explicit_size_kernel[(1, )](src, dst, CHUNK_SIZE, num_warps=1)

    # Verify IR uses the bulk path
    ttgir = kernel.asm["ttgir"]
    assert "bulk_size" in ttgir, "Expected bulk_size operand in TTGIR"
    assert "useBulk = true" in ttgir, "Expected useBulk = true in TTGIR"

    # Verify correctness
    torch.testing.assert_close(src, dst)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_fence_gpu(device):

    @triton.jit
    def fence_gpu_kernel(ptr):
        tl.atomic_add(ptr, 1)
        tlx.fence("gpu")
        tl.atomic_add(ptr + 1, 1)

    x = torch.zeros(2, dtype=torch.int32, device=device)
    kernel = fence_gpu_kernel[(1, )](x, num_warps=1)

    # Verify TTGIR contains the fence op with gpu scope
    ttgir = kernel.asm["ttgir"]
    assert 'ttng.fence {scope = "gpu"}' in ttgir

    # Verify PTX contains the correct fence instruction
    ptx = kernel.asm["ptx"]
    assert "fence.acq_rel.gpu" in ptx

    # Verify correctness
    assert x[0].item() == 1
    assert x[1].item() == 1


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_fence_sys(device):

    @triton.jit
    def fence_sys_kernel(ptr):
        tl.atomic_add(ptr, 1)
        tlx.fence(scope="sys")
        tl.atomic_add(ptr + 1, 1)

    x = torch.zeros(2, dtype=torch.int32, device=device)
    kernel = fence_sys_kernel[(1, )](x, num_warps=1)

    # Verify TTGIR contains the fence op with sys scope
    ttgir = kernel.asm["ttgir"]
    assert 'ttng.fence {scope = "sys"}' in ttgir

    # Verify PTX contains the correct fence instruction
    ptx = kernel.asm["ptx"]
    assert "fence.acq_rel.sys" in ptx

    # Verify correctness
    assert x[0].item() == 1
    assert x[1].item() == 1
