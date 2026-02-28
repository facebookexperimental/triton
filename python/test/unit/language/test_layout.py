# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Test to verify that the rmsnorm kernel uses the expected layout.

This test compiles a Triton kernel and checks the generated ttgir to verify
that the layout matches the expected pattern.
"""

import os
import re
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def parse_layout_params(layout_str: str) -> dict | None:
    """
    Parse a blocked layout string and extract its parameters.

    Args:
        layout_str: A layout string like
            "#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], ...}>"

    Returns:
        A dict with extracted parameters, or None if no parameters found.
    """
    params = {}

    # Extract sizePerThread
    match = re.search(r"sizePerThread\s*=\s*\[([^\]]+)\]", layout_str)
    if match:
        params["sizePerThread"] = [int(x.strip()) for x in match.group(1).split(",")]

    # Extract threadsPerWarp
    match = re.search(r"threadsPerWarp\s*=\s*\[([^\]]+)\]", layout_str)
    if match:
        params["threadsPerWarp"] = [int(x.strip()) for x in match.group(1).split(",")]

    # Extract warpsPerCTA
    match = re.search(r"warpsPerCTA\s*=\s*\[([^\]]+)\]", layout_str)
    if match:
        params["warpsPerCTA"] = [int(x.strip()) for x in match.group(1).split(",")]

    # Extract order
    match = re.search(r"order\s*=\s*\[([^\]]+)\]", layout_str)
    if match:
        params["order"] = [int(x.strip()) for x in match.group(1).split(",")]

    return params if params else None


def parse_slice_layout(layout_str: str) -> dict | None:
    """
    Parse a slice layout string and extract its parameters.

    Args:
        layout_str: A layout string like "#ttg.slice<{dim = 1, parent = #blocked}>"

    Returns:
        A dict with 'dim' and 'parent' keys, or None if parsing fails.
    """
    params = {}

    # Extract dim
    dim_match = re.search(r"dim\s*=\s*(\d+)", layout_str)
    if dim_match:
        params["dim"] = int(dim_match.group(1))

    # Extract parent layout name
    parent_match = re.search(r"parent\s*=\s*(#\w+)", layout_str)
    if parent_match:
        params["parent"] = parent_match.group(1)

    return params if params else None


def extract_blocked_layout(ttgir_content: str) -> str | None:
    """
    Extract the primary blocked layout definition from ttgir content.

    Returns the full blocked layout string or None if not found.
    """
    blocked_pattern = r"(#blocked\s*=\s*#ttg\.blocked<\{[^}]+\}>)"
    match = re.search(blocked_pattern, ttgir_content)
    if match:
        return match.group(1)
    return None


def extract_reduce_output_layout(ttgir_content: str) -> dict | None:
    """
    Extract the output layout from tt.reduce operations in ttgir content.

    The tt.reduce operation outputs a tensor with a sliced layout like:
        tensor<512xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    The tt.reduce operation spans multiple lines:
        %variance = "tt.reduce"(%x_squared) <{axis = 1 : i32}> ({
        ^bb0(...):
          ...
          tt.reduce.return %result : f32 loc(...)
        }) : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(...)

    Returns:
        A dict with 'dim' and 'parent' keys describing the slice layout,
        or None if no reduce operation found.
    """
    # Pattern to match tt.reduce operation including multi-line body
    # Using re.DOTALL to make . match newlines
    # The pattern captures:
    # 1. "tt.reduce" - the operation name
    # 2. Everything up to the closing }) which ends the reduce body
    # 3. The type signature : (input) -> output with slice layout
    reduce_pattern = (
        r'"tt\.reduce"'  # Match the tt.reduce operation
        r"[\s\S]*?"  # Match any characters including newlines (non-greedy)
        r"\}\)\s*:\s*"  # Match the closing }) :
        r"\([^)]+\)\s*->\s*"  # Match (input_type) ->
        r"tensor<[^,]+,\s*(#ttg\.slice<\{[^}]+\}>)>"  # Match output tensor with slice layout
    )

    match = re.search(reduce_pattern, ttgir_content)
    if match:
        slice_layout = match.group(1)
        return parse_slice_layout(slice_layout)

    return None


def get_expected_slice_params(reduce_axis: int) -> dict:
    """
    Calculate expected slice layout parameters for a reduce operation.

    When reducing along an axis, the output layout is a slice of the parent
    blocked layout with that dimension removed.

    Args:
        reduce_axis: The axis along which the reduction is performed (0 or 1)

    Returns:
        Dictionary with expected slice layout parameters
    """
    return {
        "dim": reduce_axis,
        "parent": "#blocked",
    }


def check_slice_layout_matches(
    ttgir_content: str, expected_params: dict
) -> tuple[bool, str]:
    """
    Check if the tt.reduce output layout matches the expected slice layout.

    Args:
        ttgir_content: The ttgir content string
        expected_params: Dict with expected slice layout parameters (dim, parent)

    Returns:
        (matches, message) tuple
    """
    actual_params = extract_reduce_output_layout(ttgir_content)

    if actual_params is None:
        return False, "No tt.reduce operation with slice layout found in ttgir"

    # Compare parameters
    mismatches = []
    for key, expected_value in expected_params.items():
        if key not in actual_params:
            mismatches.append(f"  {key}: expected {expected_value}, but key not found")
        elif actual_params[key] != expected_value:
            mismatches.append(
                f"  {key}: expected {expected_value}, got {actual_params[key]}"
            )

    if mismatches:
        return False, (
            f"Slice layout mismatch:\n"
            + "\n".join(mismatches)
            + f"\nActual slice params: {actual_params}"
        )

    return True, f"Slice layout matches: {actual_params}"


def check_layout_matches(ttgir_content: str, expected_params: dict) -> tuple[bool, str]:
    """
    Check if the ttgir content contains the expected blocked layout parameters.

    Args:
        ttgir_content: The ttgir content string
        expected_params: Dict with expected layout parameters

    Returns:
        (matches, message) tuple
    """
    actual_layout = extract_blocked_layout(ttgir_content)

    if actual_layout is None:
        return False, "No blocked layout found in ttgir"

    actual_params = parse_layout_params(actual_layout)

    if actual_params is None:
        return False, f"Failed to parse actual layout: {actual_layout}"

    # Compare each parameter that exists in expected_params
    mismatches = []
    for key, expected_value in expected_params.items():
        if key not in actual_params:
            mismatches.append(f"  {key}: expected {expected_value}, but key not found")
        elif actual_params[key] != expected_value:
            mismatches.append(
                f"  {key}: expected {expected_value}, got {actual_params[key]}"
            )

    if mismatches:
        return False, f"Layout mismatch:\n" + "\n".join(
            mismatches
        ) + f"\nActual layout: {actual_layout}"

    return True, f"Layout matches: {actual_params}"


# Define the RMSNorm kernel
@triton.jit
def _apply_rmsnorm_tile(
    output_tile,
    ln_weight,
    eps,
    HEAD_DIM: tl.constexpr,
):
    """Apply RMSNorm to a tile."""
    x_squared = output_tile * output_tile
    variance = tl.sum(x_squared, axis=1) / HEAD_DIM
    rrms = libdevice.rsqrt(variance + eps)
    normalized_tile = output_tile * rrms[:, None] * ln_weight[None, :]
    return normalized_tile


@triton.jit
def rmsnorm_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    M,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    eps: tl.constexpr,
):
    """Wrapper kernel that loads data, calls _apply_rmsnorm_tile, and stores results."""
    pid = tl.program_id(0)

    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, HEAD_DIM)

    mask = row_offsets[:, None] < M

    offsets = row_offsets[:, None] * HEAD_DIM + col_offsets[None, :]
    x_tile = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    ln_weight = tl.load(W_ptr + col_offsets)

    normalized_tile = _apply_rmsnorm_tile(x_tile, ln_weight, eps, HEAD_DIM)

    tl.store(Out_ptr + offsets, normalized_tile, mask=mask)


# Constants for layout calculation
SIZE_PER_THREAD_FEATURE = 4  # Elements processed per thread in feature dimension
NUM_WARPS = 8  # Number of warps per CTA (Cooperative Thread Array)


def get_warp_size() -> int:
    """
    Get the warp size for the current GPU.

    Returns:
        Warp size: 64 for AMD GPUs (wavefront), 32 for NVIDIA GPUs

    Raises:
        RuntimeError: If CUDA/ROCm is not available or GPU type is not recognized
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm not available")

    device_name = torch.cuda.get_device_name(0).lower()

    # AMD GPUs use 64-thread wavefronts
    # Common identifiers: "amd", "radeon", "mi" (e.g., MI250, MI300), "gfx" (e.g., gfx90a)
    if (
        "amd" in device_name
        or "radeon" in device_name
        or "mi" in device_name
        or "gfx" in device_name
    ):
        return 64

    # NVIDIA GPUs use 32-thread warps
    # Common identifiers: "nvidia", "geforce", "tesla", "quadro", "rtx", "gtx",
    # Data center GPUs: "a100", "h100", "h200", "b100", "b200", "v100", "p100"
    if (
        "nvidia" in device_name
        or "geforce" in device_name
        or "tesla" in device_name
        or "quadro" in device_name
        or "rtx" in device_name
        or "gtx" in device_name
        or "a100" in device_name
        or "h100" in device_name
        or "h200" in device_name
        or "b100" in device_name
        or "b200" in device_name
        or "v100" in device_name
        or "p100" in device_name
    ):
        return 32

    raise RuntimeError(
        f"Unsupported GPU type: '{device_name}'. "
        "Expected AMD GPU (warp size 64) or NVIDIA GPU (warp size 32). "
        "Please update get_warp_size() to support this GPU."
    )


def get_expected_params(D: int, warp_size: int) -> dict:
    """
    Calculate expected layout parameters based on dimension D and warp size.

    The Triton compiler deterministically calculates the blocked layout based on
    the block dimensions and target hardware. For a 2D blocked layout:

    Layout Constraints:
    ------------------
    1. Total threads per warp must equal warp_size:
       - AMD GPUs: warp_size = 64 (wavefront)
       - NVIDIA GPUs: warp_size = 32
       threadsPerWarp[0] × threadsPerWarp[1] = warp_size

    2. Each warp must cover the full feature dimension D:
       sizePerThread[1] × threadsPerWarp[1] = D
       (where sizePerThread[1] = SIZE_PER_THREAD_FEATURE = 4)

    Calculation:
    -----------
    Given sizePerThread = [1, 4] (each thread processes 4 elements in feature dim):

    - threadsPerWarp[1] = D / sizePerThread[1] = D / 4
      (threads needed in feature dimension to cover D elements)

    - threadsPerWarp[0] = warp_size / threadsPerWarp[1]
      (remaining threads distributed to batch dimension)

    Examples (AMD GPU, warp_size=64):
    ---------------------------------
    | D   | threadsPerWarp[1] | threadsPerWarp[0] | Layout       |
    |-----|-------------------|-------------------|--------------|
    | 16  | 16 / 4 = 4        | 64 / 4 = 16       | [16, 4]      |
    | 32  | 32 / 4 = 8        | 64 / 8 = 8        | [8, 8]       |
    | 64  | 64 / 4 = 16       | 64 / 16 = 4       | [4, 16]      |
    | 128 | 128 / 4 = 32      | 64 / 32 = 2       | [2, 32]      |

    Examples (NVIDIA GPU, warp_size=32):
    ------------------------------------
    | D   | threadsPerWarp[1] | threadsPerWarp[0] | Layout       |
    |-----|-------------------|-------------------|--------------|
    | 16  | 16 / 4 = 4        | 32 / 4 = 8        | [8, 4]       |
    | 32  | 32 / 4 = 8        | 32 / 8 = 4        | [4, 8]       |
    | 64  | 64 / 4 = 16       | 32 / 16 = 2       | [2, 16]      |
    | 128 | 128 / 4 = 32      | 32 / 32 = 1       | [1, 32]      |

    Args:
        D: Feature dimension size (must be a power of 2, >= 16)
        warp_size: Number of threads per warp (64 for AMD, 32 for NVIDIA)

    Returns:
        Dictionary with expected layout parameters
    """
    # Calculate threads needed in feature dimension to cover D elements
    threads_per_warp_feature = D // SIZE_PER_THREAD_FEATURE

    # Remaining threads go to batch dimension
    threads_per_warp_batch = warp_size // threads_per_warp_feature

    return {
        "sizePerThread": [1, SIZE_PER_THREAD_FEATURE],
        "threadsPerWarp": [threads_per_warp_batch, threads_per_warp_feature],
        "warpsPerCTA": [NUM_WARPS, 1],
        "order": [1, 0],
    }


@pytest.mark.parametrize("T", [128, 256])
@pytest.mark.parametrize("D", [16, 32, 64, 128])
def test_rmsnorm_layout(T, D):
    """
    Test that the rmsnorm kernel uses the expected uniform layout.

    This test compiles the rmsnorm kernel, retrieves the generated ttgir,
    and verifies that the blocked layout matches the expected pattern.

    Uses the same kernel launch parameter configs from:
    genai/msl/ops/kernels/triton/norm/rms_norm.py (lines 195-229)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32
    eps = 1e-6

    # Configure kernel launch parameters (from rms_norm.py lines 195-229)
    NUM_ELEMENTS = 8192  # Target elements per thread block
    NUM_WARPS = 8  # Number of warps per thread block
    BLOCK_D = min(
        triton.next_power_of_2(D), NUM_ELEMENTS
    )  # Block size in feature dimension
    BLOCK_T = max(
        1, triton.next_power_of_2(NUM_ELEMENTS // BLOCK_D)
    )  # Block size in batch dimension

    # Create input tensors
    x = torch.randn(T, D, device=device, dtype=dtype)
    weight = torch.randn(D, device=device, dtype=dtype)
    output = torch.empty_like(x)

    # Compile and run the kernel
    grid = (triton.cdiv(T, BLOCK_T),)
    k = rmsnorm_kernel[grid](
        x,
        weight,
        output,
        T,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_T,
        eps=eps,
        num_warps=NUM_WARPS,
    )

    # Verify correctness first
    variance = (x**2).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    expected = x * rrms * weight
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    # Check the ttgir for expected layout pattern
    ttgir = k.asm["ttgir"]

    # Get warp size for current GPU and expected parameters based on dimension D
    warp_size = get_warp_size()
    expected_params = get_expected_params(D, warp_size)

    # Verify the blocked layout matches expected pattern
    matches, message = check_layout_matches(ttgir, expected_params)
    assert matches, f"The TTGIR layout does not match the expected pattern.\n{message}"

    # Verify the reduce output layout (slice layout) matches expected pattern
    # The RMSNorm kernel reduces along axis=1 (the feature dimension)
    expected_slice_params = get_expected_slice_params(reduce_axis=1)
    slice_matches, slice_message = check_slice_layout_matches(
        ttgir, expected_slice_params
    )
    assert slice_matches, (
        f"The tt.reduce output layout does not match the expected slice pattern.\n"
        f"{slice_message}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
