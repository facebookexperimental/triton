"""TLX AMD tests -- CDNA3 (gfx942) -- declared, no OSS runner yet."""
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hip_cdna3


@triton.jit
def _amd_scheduled_mfma_gfx942_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    PERSISTENT: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    INSTR_K: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[16, 16, INSTR_K],
        transposed=TRANSPOSED,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    rows = tl.arange(0, 32)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tl.full((32, 128), 7.0, tl.float32)
    acc = tlx.require_layout(acc, mma, pin=False)
    if PERSISTENT:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="persistent",
            # On CDNA3 the compiler-generated AGPR read is not ordered
            # against the MFMA drain, so the accumulator stays in VGPRs.
            accumulator_register_class="vgpr",
            initialize=True,
        )
    else:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="transient",
            initialize=True,
        )
        result, _ = tlx.amd_mfma_commit(result, b)
    offsets = output_ptr + rows[:, None] * 128 + cols[None, :]
    offsets = tlx.require_layout(offsets, mma, pin=False)
    tl.store(offsets, result)


@triton.jit
def _amd_scheduled_mfma_32x32_gfx942_kernel(a_ptr, b_ptr, output_ptr, PERSISTENT: tl.constexpr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[32, 32, 8],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    rows = tl.arange(0, 32)
    reduction = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tlx.zeros((32, 128), tl.float32, layout=mma)
    if PERSISTENT:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="persistent",
            # On CDNA3 the compiler-generated AGPR read is not ordered
            # against the MFMA drain, so the accumulator stays in VGPRs.
            accumulator_register_class="vgpr",
            initialize=True,
        )
    else:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="transient",
            initialize=True,
        )
    offsets = output_ptr + rows[:, None] * 128 + cols[None, :]
    offsets = tlx.require_layout(offsets, mma, pin=False)
    tl.store(offsets, result)


@triton.jit
def _amd_scheduled_mfma_large_gfx942_kernel(a_ptr, b_ptr, output_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[2, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    rows = tl.arange(0, 256)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 256)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 256 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tlx.zeros((256, 256), tl.float32, layout=mma)
    acc = tlx.amd_scheduled_mfma(a, b, acc, accumulator_role="persistent", accumulator_register_class="vgpr")
    acc = tlx.amd_scheduled_mfma(a, b, acc, accumulator_role="persistent", accumulator_register_class="vgpr")
    offsets = output_ptr + rows[:, None] * 256 + cols[None, :]
    offsets = tlx.require_layout(offsets, mma, pin=False)
    tl.store(offsets, acc)


@pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942 hardware")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("persistent", [False, True])
def test_amd_scheduled_mfma_32x32_correct_gfx942(persistent, dtype):
    """The mnemonic check above cannot see a fragment packing or ordering bug.

    A 32x32x8 fragment is 16 accumulator registers per lane against the
    16x16x16 path's 4, so it exercises a different packing.
    """
    torch.manual_seed(0)
    a = torch.randn((32, 16), device="cuda", dtype=dtype)
    b = torch.randn((16, 128), device="cuda", dtype=dtype)
    actual = torch.empty((32, 128), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_32x32_gfx942_kernel[(1, )](
        a,
        b,
        actual,
        PERSISTENT=persistent,
        num_warps=4,
        matrix_instr_nonkdim=32,
    )
    torch.testing.assert_close(actual, a.float() @ b.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942 hardware")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_amd_scheduled_mfma_multifragment_correct_gfx942(persistent, transposed, dtype):
    torch.manual_seed(0)
    a = torch.randn((32, 32), device="cuda", dtype=dtype)
    b = torch.randn((32, 128), device="cuda", dtype=dtype)
    actual = torch.empty((32, 128), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_gfx942_kernel[(1, )](
        a,
        b,
        actual,
        PERSISTENT=persistent,
        TRANSPOSED=transposed,
        INSTR_K=16,  # 16x16x16
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    torch.testing.assert_close(actual, a.float() @ b.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not is_hip_cdna3(), reason="Requires gfx942 hardware")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_amd_scheduled_mfma_large_persistent_correct_gfx942(dtype):
    torch.manual_seed(0)
    a = torch.randn((256, 32), device="cuda", dtype=dtype)
    b = torch.randn((32, 256), device="cuda", dtype=dtype)
    actual = torch.empty((256, 256), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_large_gfx942_kernel[(1, )](
        a,
        b,
        actual,
        num_warps=8,
        matrix_instr_nonkdim=16,
    )
    torch.testing.assert_close(actual, 2 * (a.float() @ b.float()), atol=4e-3, rtol=2e-3)
