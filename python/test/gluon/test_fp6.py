import pytest
import torch

from triton._internal_testing import fp6_values, is_blackwell, pack_fp6
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("K", [128])
@pytest.mark.parametrize("offset", [0, 128])
@pytest.mark.parametrize("a_format,b_format", [("e2m3", "e3m2"), ("e3m2", "e2m3")])
@pytest.mark.parametrize("code_side", ["a", "b"])
def test_tcgen05_mma_scaled_fp6_tma(K, offset, a_format, b_format, code_side):

    @gluon.jit
    def kernel(a_desc, b_desc, out, K: ttgl.constexpr, OFFSET: ttgl.constexpr, A_FORMAT: ttgl.constexpr,
               B_FORMAT: ttgl.constexpr):
        a = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
        b = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)
        a_scale = allocate_tensor_memory(ttgl.uint8, [128, K // 32], TensorMemoryScalesLayout())
        b_scale = allocate_tensor_memory(ttgl.uint8, [128, K // 32], TensorMemoryScalesLayout())
        a_scale.store(ttgl.full([128, K // 32], 127, ttgl.uint8, a_scale.get_reg_layout()))
        b_scale.store(ttgl.full([128, K // 32], 127, ttgl.uint8, b_scale.get_reg_layout()))
        acc = allocate_tensor_memory(ttgl.float32, [128, 128], TensorMemoryLayout([128, 128], col_stride=1))
        load_bar = mbarrier.allocate_mbarrier()
        mma_bar = mbarrier.allocate_mbarrier()
        mbarrier.init(load_bar, count=1)
        mbarrier.init(mma_bar, count=1)
        ttgl.static_assert(a_desc.nbytes_per_cta == 128 * K * 3 // 4)
        ttgl.static_assert(b_desc.nbytes_per_cta == 128 * K * 3 // 4)
        mbarrier.expect(load_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        # OFFSET is in logical FP6 values, not bytes: the descriptor
        # translates it to dense byte offset 3*OFFSET//4 in global memory
        # (and padded containers in smem). Host data below is filled for
        # the first 3*(K+OFFSET)//4 byte-columns for the same reason.
        tma.async_load(a_desc, [0, OFFSET], load_bar, a)
        tma.async_load(b_desc, [0, OFFSET], load_bar, b)
        mbarrier.wait(load_bar, phase=0, deps=[a, b])
        tcgen05_mma_scaled(a, b.permute((1, 0)), acc, a_scale, b_scale, A_FORMAT, B_FORMAT, use_acc=False,
                           mbarriers=[mma_bar])
        mbarrier.wait(mma_bar, phase=0)
        values = acc.load()
        layout: ttgl.constexpr = values.type.layout
        rows = ttgl.arange(0, 128, layout=ttgl.SliceLayout(1, layout))[:, None]
        cols = ttgl.arange(0, 128, layout=ttgl.SliceLayout(0, layout))[None, :]
        ttgl.store(out + rows * 128 + cols, values)
        mbarrier.invalidate(load_bar)
        mbarrier.invalidate(mma_bar)

    def operand(fmt, all_codes):
        cols = K + offset
        base = torch.arange(128, device="cuda", dtype=torch.int32) % 64
        if not all_codes:
            base.fill_(8 if fmt == "e2m3" else 12)
        codes_2d = base[:, None].repeat(1, cols)
        packed = pack_fp6(codes_2d)
        values = fp6_values(fmt, base)
        return packed, values

    a, a_values = operand(a_format, code_side == "a")
    b, b_values = operand(b_format, code_side == "b")
    layout = ttgl.NVMMASharedLayout.get_default_for([128, K], ttgl.uint8, fp6_padded=True)
    a_desc = TensorDescriptor.from_tensor(a, [128, K], layout)
    b_desc = TensorDescriptor.from_tensor(b, [128, K], layout)
    out = torch.empty((128, 128), device="cuda", dtype=torch.float32)
    compiled = kernel[(1, )](a_desc, b_desc, out, K, offset, a_format, b_format, num_warps=4)
    torch.testing.assert_close(out, a_values[:, None] * b_values[None, :] * K, atol=0, rtol=0)
    assert "kind::mxf8f6f4" in compiled.asm["ptx"]


def test_nvmma_shared_layout_fp6_padded():
    plain = ttgl.NVMMASharedLayout(128, 8)
    padded = ttgl.NVMMASharedLayout(128, 8, fp6_padded=True)
    # Legacy fp6_padded=True maps to canonical b6x16_p32 padding.
    assert padded.padding == "b6x16_p32"
    assert padded.fp6_padded
    assert "b6x16_p32" in padded.mangle()
    assert "b6x16_p32" not in plain.mangle()
    assert hash(padded) != hash(plain)
    assert padded != plain
    canonical = ttgl.NVMMASharedLayout(128, 8, padding="b6x16_p32")
    assert canonical == padded
    assert canonical.mangle() == padded.mangle()
    with pytest.raises(AssertionError, match="mutually exclusive"):
        ttgl.NVMMASharedLayout(128, 8, fp4_padded=True, fp6_padded=True)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("misalignment", ["base", "stride"])
def test_fp6_tma_descriptor_alignment(misalignment):
    layout = ttgl.NVMMASharedLayout(128, 8, fp6_padded=True)
    # Packed FP6 rows hold 128 logical values in 96 bytes; use packed
    # shapes so the shape check passes and the alignment check is exercised.
    if misalignment == "base":
        data = torch.empty(128 * 96 + 16, device="cuda", dtype=torch.uint8)[16:].reshape(128, 96)
    else:
        data = torch.empty((128, 112), device="cuda", dtype=torch.uint8)[:, :96]
    with pytest.raises(AssertionError, match="32-byte aligned"):
        TensorDescriptor.from_tensor(data, [128, 128], layout)
