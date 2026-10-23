"""TLX layout tests -- CDNA4 (gfx950)."""
import math
import re
import importlib.util
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip_cdna4
import triton.language.extra.tlx as tlx
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.tlx.ops.kernels.flash_attn.gfx950 import _sum_rows_chain4 as _cluster_sum_rows_chain4
from triton.language.extra.tlx.tutorials.amd_gemm_warp_pipeline import matmul_kernel_warp_pipeline

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("bits", [16, 32])
@pytest.mark.parametrize("iterations", [0, 3])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("shared_mask", [False, True])
def test_buffer_loop_offset_wrapping(bits, iterations, negative, shared_mask):

    @triton.jit
    def kernel(X, Y, base, start, count, BITS: tl.constexpr, SHARED_MASK: tl.constexpr):
        lane = tl.arange(0, 64) - 32
        acc = tl.full((64, ), 0, tl.int32)
        for i in range(count):
            dynamic = start + i * 32
            if BITS == 16:
                term = (dynamic.to(tl.int16) + lane.to(tl.int16)).to(tl.int16)
            else:
                term = dynamic + lane
            offset = (base.to(tl.int64) + term.to(tl.int64)).to(tl.int32)
            mask = offset < 65536 if SHARED_MASK else tl.full((64, ), True, tl.int1)
            acc += tlx.buffer_load(X, offset, mask, other=0)
        tl.store(Y + tl.arange(0, 64), acc)

    start = -(1 << (bits - 1)) if negative else (1 << (bits - 1)) - 16
    base = -start + 65536 + (1 << 32)
    x = torch.arange(131072, device=DEVICE, dtype=torch.int32)
    y = torch.empty((64, ), device=DEVICE, dtype=torch.int32)
    kernel[(1, )](x, y, base, start, iterations, bits, shared_mask, num_warps=1)
    lanes = torch.arange(64, device=DEVICE, dtype=torch.int64) - 32
    expected = torch.zeros_like(y)
    for i in range(iterations):
        term = (start + i * 32 + lanes + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
        offset = (base + term).to(torch.int32).to(torch.int64)
        valid = offset < 65536 if shared_mask else torch.ones_like(offset, dtype=torch.bool)
        expected += torch.where(valid, x[offset.clamp(0, x.numel() - 1)], 0)
    torch.testing.assert_close(y, expected, atol=0, rtol=0)


@pytest.fixture(scope="module")
def split_k_tutorial():
    path = Path(__file__).resolve().parents[3] / "tutorials" / "12-split-k-matmul.py"
    spec = importlib.util.spec_from_file_location("split_k_tutorial_test", path)
    tutorial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tutorial)
    for kernel in vars(tutorial).values():
        if isinstance(kernel, triton.runtime.Autotuner):
            kernel.configs = kernel.configs[:1]
    return tutorial


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("method", ["stock_triton_matmul", "skinny_atomic_matmul", "skinny_twopass_matmul"])
@pytest.mark.parametrize("m,n", [(64, 96), (288, 160), (1, 32), (32, 1), (1, 1), (0, 32), (32, 0)])
def test_split_k_grouped_coordinates(split_k_tutorial, method, m, n):
    a = torch.randn((m, 256), device=DEVICE, dtype=torch.float16) * 0.2
    b = torch.randn((256, n), device=DEVICE, dtype=torch.float16) * 0.2
    output = getattr(split_k_tutorial, method)(a, b)
    torch.testing.assert_close(output, a @ b, atol=0.015, rtol=0.015)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("m,n", [(256, 256), (768, 1280), (1, 256), (256, 1), (1, 1), (0, 256), (256, 0), (-1, 256)])
def test_grouped_gemm_ignores_outside_tiles(m, n):
    k = 256
    a = torch.randn((max(m, 1), k), device=DEVICE, dtype=torch.float16) * 0.2
    b = torch.randn((k, max(n, 1)), device=DEVICE, dtype=torch.float16) * 0.2
    output = torch.zeros((max(m, 1), max(n, 1)), device=DEVICE, dtype=torch.float16)
    tiles = triton.cdiv(max(m, 0), 256) * triton.cdiv(max(n, 0), 256)
    matmul_kernel_warp_pipeline[(tiles + 3, )](a, b, output, m, n, k,
                                               a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                                               output.stride(1), BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, GROUP_M=4,
                                               NUM_BUFFERS=2, NUM_XCDS=3, XCD_CHUNK=5, num_warps=8, num_stages=1,
                                               matrix_instr_nonkdim=16)
    expected = a @ b if m > 0 and n > 0 else torch.zeros_like(output)
    torch.testing.assert_close(output, expected, atol=0.015, rtol=0.015)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("start,stride", [(0, 3), (2147483640, 3), (-2147483648, -1), (1073741824, 7)])
def test_address_distribution_preserves_wrapping(start, stride):

    @triton.jit(do_not_specialize=["start", "stride"])
    def kernel(table, output, start, stride):
        lane = tl.arange(0, 128)
        acc = tl.full((128, ), 0, tl.int32)
        for i in range(5):
            product = ((start + i + lane) * stride).to(tl.int64)
            index = (product.to(tl.uint64) // 3) % 2
            acc += tl.load(table + index)
        tl.store(output + lane, acc)

    table = torch.tensor([17, 41], dtype=torch.int32, device=DEVICE)
    output = torch.empty((128, ), dtype=torch.int32, device=DEVICE)
    compiled = kernel[(1, )](table, output, start, stride)
    expected = []
    for lane in range(128):
        total = 0
        for i in range(5):
            product = ((start + i + lane) * stride + 2**31) % 2**32 - 2**31
            total += (17, 41)[(product % 2**64 // 3) % 2]
        expected.append(total)
    torch.testing.assert_close(output, torch.tensor(expected, dtype=torch.int32, device=DEVICE), rtol=0, atol=0)
    assert re.search(r"arith.muli .* : i32", compiled.asm["ttgir"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("m,n", [(13, 17), (1025, 2049), (2147483647, 2147483647)])
@pytest.mark.parametrize("xcd,chunk,group", [(8, 4, 4), (3, 5, 7)])
def test_integer_facts_swizzle_coordinates(m, n, xcd, chunk, group):

    @triton.jit
    def permute(tile, count, XCD: tl.constexpr, CHUNK: tl.constexpr):
        aligned = (count // (XCD * CHUNK)) * (XCD * CHUNK)
        if tile >= aligned:
            return tile
        minor = tile % XCD
        major = tile // XCD
        return (major // CHUNK) * XCD * CHUNK + minor * CHUNK + major % CHUNK

    @triton.jit(do_not_specialize=["m", "n"])
    def kernel(tiles, row_out, col_out, m, n, XCD: tl.constexpr, CHUNK: tl.constexpr, GROUP: tl.constexpr):
        if (m > 0) & (n > 0):
            wide_m = m.to(tl.uint64)
            wide_n = n.to(tl.uint64)
            rows = tl.cdiv(wide_m, 32)
            cols = tl.cdiv(wide_n, 32)
            count = rows * cols
            tile = tl.load(tiles + tl.program_id(0)).to(tl.uint64)
            if tile < count:
                mapped = permute(tile, count, XCD, CHUNK)
                width = GROUP * cols
                base = (mapped // width) * GROUP
                active = tl.minimum(rows - base, GROUP)
                row = base + (mapped % width) % active
                col = (mapped % width) // active
                lane = tl.arange(0, 32).to(tl.uint64)
                r = (row * 32 + lane) % wide_m
                c = (col * 32 + lane) % wide_n
                out = tl.program_id(0) * 32 + tl.arange(0, 32)
                tl.store(row_out + out, r)
                tl.store(col_out + out, c)

    rows, cols = triton.cdiv(m, 32), triton.cdiv(n, 32)
    count = rows * cols
    tiles = sorted({0, min(1, count - 1), count // 2, count - 1})
    tile_tensor = torch.tensor(tiles, dtype=torch.int64, device=DEVICE)
    row_out = torch.empty((len(tiles), 32), dtype=torch.int64, device=DEVICE)
    col_out = torch.empty_like(row_out)
    compiled = kernel[(len(tiles), )](tile_tensor, row_out, col_out, m, n, xcd, chunk, group)
    expected_rows, expected_cols = [], []
    for tile in tiles:
        aligned = count // (xcd * chunk) * (xcd * chunk)
        major, minor = divmod(tile, xcd)
        mapped = tile if tile >= aligned else major // chunk * xcd * chunk + minor * chunk + major % chunk
        base = mapped // (group * cols) * group
        active = min(rows - base, group)
        row = base + mapped % (group * cols) % active
        col = mapped % (group * cols) // active
        expected_rows.append([(row * 32 + lane) % m for lane in range(32)])
        expected_cols.append([(col * 32 + lane) % n for lane in range(32)])
    torch.testing.assert_close(row_out, torch.tensor(expected_rows, device=DEVICE), rtol=0, atol=0)
    torch.testing.assert_close(col_out, torch.tensor(expected_cols, device=DEVICE), rtol=0, atol=0)
    assert re.search(r"arith.remui .* : tensor<32xi32>", compiled.asm["ttir"])
    assert not re.search(r"arith.remui .* : tensor<32xi64>", compiled.asm["ttir"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("value", [-17, 0, 1023, 1024, 2**40])
def test_integer_facts_narrow_branch(value):

    @triton.jit
    def kernel(input, output):
        x = tl.load(input)
        if (x >= 0) & (x < 1024):
            result = x // 3
        else:
            result = x // 5
        tl.store(output, result)

    input = torch.tensor([value], dtype=torch.int64, device=DEVICE)
    output = torch.empty_like(input)
    compiled = kernel[(1, )](input, output)
    expected = torch.div(input, 3 if 0 <= value < 1024 else 5, rounding_mode="trunc")
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert re.search(r"arith.divsi .* : i32", compiled.asm["ttir"])
    assert re.search(r"arith.divsi .* : i64", compiled.asm["ttir"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("value", [-2147483648, -17, -1, 0, 2147483647])
def test_integer_facts_unsigned_sign_extension(value):

    @triton.jit
    def kernel(input, output):
        x = tl.load(input).to(tl.int64).to(tl.uint64)
        tl.store(output, x % 7)
        tl.store(output + 1, x >> 33)

    input = torch.tensor([value], dtype=torch.int32, device=DEVICE)
    output = torch.empty((2, ), dtype=torch.int64, device=DEVICE)
    kernel[(1, )](input, output)
    unsigned = value % 2**64
    expected = torch.tensor([unsigned % 7, unsigned >> 33], dtype=torch.int64, device=DEVICE)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("shift", [0, 31, 32, 33, 63])
def test_integer_facts_wide_shift(shift):

    @triton.jit
    def kernel(input, amount, output):
        x = tl.load(input).to(tl.uint32).to(tl.uint64)
        s = tl.load(amount).to(tl.uint64)
        tl.store(output, x >> s)

    input = torch.tensor([-1], dtype=torch.int32, device=DEVICE)
    amount = torch.tensor([shift], dtype=torch.int32, device=DEVICE)
    output = torch.empty((1, ), dtype=torch.int64, device=DEVICE)
    kernel[(1, )](input, amount, output)
    assert output.item() == (2**32 - 1) >> shift


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("base", [-128, -64, -4, 0, 128, 2147483584])
@pytest.mark.parametrize("divisor", [-64, -4, 4, 64])
@pytest.mark.parametrize("op", ["div", "rem"])
@pytest.mark.parametrize("indirect", [False, True])
def test_signed_div_rem_contiguous_boundary(base, divisor, op, indirect):

    @triton.jit
    def kernel(output, table, base, DIVISOR: tl.constexpr, OP: tl.constexpr, INDIRECT: tl.constexpr, L: tl.constexpr):
        lanes = tlx.require_layout(tl.arange(0, 1024), L)
        values = base + lanes
        if OP == "div":
            result = values // DIVISOR
        else:
            result = values % DIVISOR
        if INDIRECT:
            if OP == "rem":
                index = result + 64
            else:
                index = result & 255
            result = tl.load(table + index)
        tl.store(output + lanes, result)

    layout = tlx.layout(shape=((64, 4), (4, )), stride=((4, 256), (1, )))
    output = torch.empty((1024, ), dtype=torch.int32, device=DEVICE)
    table = torch.arange(256, dtype=torch.int32, device=DEVICE) * 3 + 7
    kernel[(1, )](output, table, base, divisor, op, indirect, layout)
    values = (torch.arange(1024, dtype=torch.int64, device=DEVICE) + base).to(torch.int32)
    quotient = torch.div(values, divisor, rounding_mode="trunc")
    expected = quotient if op == "div" else values - quotient * divisor
    if indirect:
        index = expected + 64 if op == "rem" else expected & 255
        expected = table[index.to(torch.int64)]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("modulus", [-17, 17])
@pytest.mark.parametrize("group", [-7, 7])
@pytest.mark.parametrize("base", [-128, 0, 2147483584])
def test_signed_div_rem_dynamic_program_mapping(modulus, group, base):

    @triton.jit
    def kernel(output, table, modulus, group, base, L: tl.constexpr):
        pid = tl.program_id(0)
        mapped = (pid % modulus) // group
        lanes = tlx.require_layout(tl.arange(0, 1024), L)
        values = mapped * 128 + base + lanes
        remainder = values % 64
        quotient = values // 64
        result = tl.load(table + (remainder + 64)) + tl.load(table + (quotient & 255))
        tl.store(output + pid * 1024 + lanes, result)

    layout = tlx.layout(shape=((64, 4), (4, )), stride=((4, 256), (1, )))
    output = torch.empty((32, 1024), dtype=torch.int32, device=DEVICE)
    table = torch.arange(256, dtype=torch.int32, device=DEVICE) * 3 + 7
    kernel[(32, )](output, table, modulus, group, base, layout)
    pid = torch.arange(32, dtype=torch.int64, device=DEVICE)
    mapped = torch.div(torch.fmod(pid, modulus), group, rounding_mode="trunc")
    values = (mapped[:, None] * 128 + base + torch.arange(1024, device=DEVICE)).to(torch.int32)
    remainder = torch.fmod(values, 64)
    quotient = torch.div(values, 64, rounding_mode="trunc")
    expected = table[(remainder + 64).long()] + table[(quotient & 255).long()]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("groups,columns", [(3, 5), (7, 13), (2, 19)])
def test_integer_facts_dynamic_tile_column(groups, columns):

    @triton.jit(do_not_specialize=["groups", "columns"])
    def kernel(output, table, groups, columns, L: tl.constexpr):
        tl.assume(groups > 0)
        tl.assume(groups <= 7)
        tl.assume(columns > 0)
        tl.assume(columns <= 8388608)
        pid = tl.program_id(0)
        tl.assume(pid < groups * columns)
        column = pid // groups
        lanes = tlx.require_layout(tl.arange(0, 256), L)
        offsets = column * 256 + lanes
        values = tl.load(table + offsets % 256)
        tl.store(output + pid * 256 + lanes, values)

    layout = tlx.layout(shape=((64, ), (4, )), stride=((4, ), (1, )))
    table = torch.arange(256, dtype=torch.int32, device=DEVICE) * 3 + 7
    output = torch.empty((groups * columns, 256), dtype=torch.int32, device=DEVICE)
    compiled = kernel[(groups * columns, )](output, table, groups, columns, layout, num_warps=1)
    torch.testing.assert_close(output, table.expand_as(output), rtol=0, atol=0)
    assert "tt.refine" in compiled.asm["ttir"]
    assert "tt.refine" not in compiled.asm["llir"]
    assert re.search(r"(?:global|buffer)_load_dwordx4", compiled.asm["amdgcn"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("rows,columns,group", [(13, 17, 8), (7, 19, 16), (24, 23, 4), (1, 5, 8)])
@pytest.mark.parametrize("persistent", [False, True])
def test_integer_facts_partial_group(rows, columns, group, persistent):

    @triton.jit(do_not_specialize=["rows", "columns"])
    def kernel(output, table, rows, columns, GROUP: tl.constexpr, L: tl.constexpr):
        tl.assume(rows > 0)
        tl.assume(rows <= 16777215)
        tl.assume(columns > 0)
        tl.assume(columns <= 8388608)
        tl.assume(rows <= 2147483647 // columns)
        pid = tl.program_id(0)
        for tile in range(pid, rows * columns, tl.num_programs(0)):
            tl.assume(tile >= 0)
            width = GROUP * columns
            base = (tile // width) * GROUP
            active = tl.minimum(rows - base, GROUP)
            column = (tile % width) // active
            lanes = tlx.require_layout(tl.arange(0, 256), L)
            offsets = column * 256 + lanes
            values = tl.load(table + offsets % 256)
            tl.store(output + tile * 256 + lanes, values)

    layout = tlx.layout(shape=((64, ), (4, )), stride=((4, ), (1, )))
    table = torch.arange(256, dtype=torch.int32, device=DEVICE) * 3 + 7
    output = torch.empty((rows * columns, 256), dtype=torch.int32, device=DEVICE)
    grid = 3 if persistent else rows * columns
    compiled = kernel[(grid, )](output, table, rows, columns, group, layout, num_warps=1)
    torch.testing.assert_close(output, table.expand_as(output), rtol=0, atol=0)
    assert re.search(r"(?:global|buffer)_load_dwordx4", compiled.asm["amdgcn"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_integer_facts_gluon_dynamic_tile_column():

    @gluon.jit(do_not_specialize=["groups", "columns"])
    def kernel(output, table, groups, columns, layout: ttgl.constexpr):
        ttgl.assume(groups > 0)
        ttgl.assume(groups <= 7)
        ttgl.assume(columns > 0)
        ttgl.assume(columns <= 8388608)
        pid = ttgl.program_id(0)
        ttgl.assume(pid < groups * columns)
        column = pid // groups
        lanes = ttgl.arange(0, 256, layout=layout)
        values = ttgl.load(table + (column * 256 + lanes) % 256)
        ttgl.store(output + pid * 256 + lanes, values)

    layout = ttgl.BlockedLayout([4], [64], [1], [0])
    table = torch.arange(256, dtype=torch.int32, device=DEVICE) * 3 + 7
    output = torch.empty((35, 256), dtype=torch.int32, device=DEVICE)
    compiled = kernel[(35, )](output, table, 7, 5, layout, num_warps=1)
    torch.testing.assert_close(output, table.expand_as(output), rtol=0, atol=0)
    assert "tt.refine" in compiled.asm["ttgir"]
    assert "tt.refine" not in compiled.asm["llir"]
    assert "load <4 x i32>" in compiled.asm["llir"]


@triton.jit
def _pinned_add_combine(a, b):
    return a + b


def _assert_no_layout_residue(ttgir):
    # Match the *encoding* form (#tlx.user_layout<...>) specifically: the TMEM
    # register-layout path sets an unrelated op attribute literally named
    # `tlx.user_layout` (see triton_tlx.cc), which must not trip this check.
    assert "#tlx.user_layout" not in ttgir, "user-layout wrapper encoding leaked into final IR"
    assert "#tlx.no_verify_layout" not in ttgir, "no-verify wrapper encoding leaked into final IR"
    assert "ttg.require_layout" not in ttgir, "require_layout boundary leaked into final IR"
    assert "ttg.release_layout" not in ttgir, "release_layout boundary leaked into final IR"


_A16W16_SHARED_INTERVALS = [(512, 16)]

_A16W16_SHARED_OFFSET_BASES = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0],
                               [2, 0], [4, 0], [8, 0]]

_A16W16_TILE = [128, 64]

_A16W16_LOAD_REG = [[0, 1], [0, 2], [0, 4], [8, 0]]

_A16W16_LOAD_LANE = [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]]

_A16W16_LOAD_WARP = [[1, 0], [2, 0], [4, 0]]

_A16W16_STORE_SHAPE = ((16, 4, 8), (8, 4))

_A16W16_STORE_STRIDE = ((8, 128, 512), (1, 4096))


@triton.jit
def _fa_pin_helper_result(value, layout: tl.constexpr):
    # The pin originates inside the helper, so its return operand is the only
    # authoritative layout witness for the helper result ABI.
    return tlx.require_layout(value, layout)


@triton.jit
def _fa_workitems_to_mfma_rows(workitems):
    rows, _ = workitems.reshape([8, 2, 32]).permute(0, 2, 1).split()
    return rows.reshape([256])


@triton.jit
def _fa_mfma_rows_to_workitems(rows):
    per_warp_rows = rows.reshape([8, 32])
    return tl.broadcast_to(per_warp_rows[:, None, :], (8, 2, 32)).reshape([512])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_shared_linear_layout_lowers_on_cdna4():
    """The Gluon K-tile mapping lowers to a native shared_linear attribute."""
    bases = [
        [0, 1],
        [0, 2],
        [0, 4],
        [0, 8],
        [1, 0],
        [2, 0],
        [4, 0],
        [8, 0],
        [0, 16],
        [0, 32],
        [0, 64],
        [16, 0],
        [32, 0],
    ]
    layout = tlx.shared_linear_layout_encoding(bases, [], 16)

    @triton.jit
    def kernel(PAD: tl.constexpr):
        buf = tlx.local_alloc((64, 128), tl.bfloat16, 1, layout=PAD)
        view = tlx.local_view(buf, 0)
        x = tlx.local_load(view)
        tlx.local_store(view, x)

    ttgir = kernel.warmup(layout, grid=(1, ), num_warps=4).asm["ttgir"]
    assert "#ttg.shared_linear" in ttgir


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_shared_linear_raw_physical_stage_compiles_on_cdna4():
    """A row-major rank-3 physical image can be written by direct-to-LDS."""
    raw_bases = [
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 4],
        [0, 1, 0],
        [0, 2, 0],
        [0, 4, 0],
        [0, 8, 0],
        [1, 0, 0],
        [2, 0, 0],
        [4, 0, 0],
        [8, 0, 0],
        [16, 0, 0],
        [32, 0, 0],
        [64, 0, 0],
        [128, 0, 0],
    ]
    raw_layout = tlx.shared_linear_layout_encoding(raw_bases, [], 16)
    k_layout = tlx.shared_linear_layout_encoding([
        [0, 1],
        [0, 2],
        [0, 4],
        [0, 8],
        [0, 64],
        [1, 0],
        [2, 0],
        [4, 0],
        [8, 64],
        [0, 16],
        [0, 32],
        [16, 0],
        [32, 0],
        [64, 0],
        [128, 0],
    ], [], 16)
    raw_async_layout = tlx.layout(
        # 256 threads cover the N dimension (six lane bits plus two warp
        # bits), while each thread owns 128 values: seven register bits for
        # Dgroup/V.  The final value mode is the N bit at 128.
        shape=((64, 4), (8, 8, 2)),
        stride=((8, 512), (1, 2048, 16384)),
    )

    @triton.jit
    def kernel(X, Y, RAW: tl.constexpr, K_LAYOUT: tl.constexpr):
        rows = tl.arange(0, 256)
        groups = tl.arange(0, 16)
        values = tl.arange(0, 8)
        offsets = (rows[:, None, None] * 128 + groups[None, :, None] * 8 + values[None, None, :])
        mask = rows[:, None, None] < 256
        mask = tl.broadcast_to(mask, offsets.shape)
        offsets = tlx.require_layout(offsets, raw_async_layout)
        mask = tlx.require_layout(mask, raw_async_layout)
        buf = tlx.local_alloc((256, 16, 8), tl.bfloat16, 1, layout=RAW)
        token = tlx.buffer_load_to_local(
            tlx.local_view(buf, 0), X, offsets,
            # Every row is valid in this compiler probe, so no fallback value
            # is needed; a scalar `other` would otherwise carry a default
            # register layout that the AMD verifier correctly rejects.
            mask=mask)
        tlx.async_load_commit_group([token])
        wait = tlx.async_load_wait_group(0)
        # The rank-3 physical image is reinterpreted as the rank-2 K tile
        # without copying; this is the descriptor half of Gluon's
        # direct-to-LDS transpose-read staging.
        k_view = tlx.local_reinterpret(tlx.local_view(buf, 0), tl.bfloat16, [256, 128], layout=K_LAYOUT)
        x = tlx.local_load(k_view, token=wait)
        x = tl.sum(x.to(tl.float32), axis=1)
        tl.store(Y + rows, x)

    x = torch.zeros((256 * 128, ), device=DEVICE, dtype=torch.bfloat16)
    y = torch.zeros((256, ), device=DEVICE, dtype=torch.float32)
    compiled = kernel.warmup(x, y, raw_layout, k_layout, grid=(1, ), num_warps=4)
    assert "#ttg.shared_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_require_layout_pin_modes_on_cdna4():
    """pin=False keeps a soft requirement; pin=True creates a user anchor."""
    layout = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((4, 256), (1, )),
    )

    @triton.jit
    def kernel(X, Y, L: tl.constexpr, PIN: tl.constexpr):
        offsets = tl.arange(0, 1024)
        values = tl.load(X + offsets)
        values = tlx.require_layout(values, L, pin=PIN)
        tl.store(Y + offsets, values)

    x = torch.arange(1024, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    soft = kernel.warmup(x, y, layout, False, grid=(1, ), num_warps=4)
    hard = kernel.warmup(x, y, layout, True, grid=(1, ), num_warps=4)
    kernel[(1, )](x, y, layout, False, num_warps=4)
    torch.testing.assert_close(y, x, atol=0, rtol=0)
    assert "tlx.require_layout" in soft.asm["ttir"]
    assert "#tlx.no_verify_layout<#linear>" in soft.asm["ttir"]
    assert "#tlx.user_layout" not in soft.asm["ttir"]
    assert "#tlx.user_layout" in hard.asm["ttir"]


@triton.jit
def _wave_local_scratch_reuse_kernel(X, Y, valid_rows, iterations):
    # Flat offsets are row * 128 + col. The low six thread bits are lanes;
    # both layouts keep the warp bases [[0, 32], [32, 0]].
    source: tl.constexpr = tlx.layout(
        shape=((32, 2, 2, 2), (4, 4, 2)),
        stride=((128, 4, 32, 4096), (1, 8, 64)),
    )
    dest: tl.constexpr = tlx.layout(
        shape=((4, 16, 2, 2), (8, 2, 2)),
        stride=((8, 128, 32, 4096), (1, 2048, 64)),
    )
    rows = tl.arange(0, 64)[:, None]
    cols = tl.arange(0, 128)[None, :]
    base = tl.program_id(0) * 4 * 64 * 128
    for iteration in range(iterations):
        for part in tl.static_range(4):
            offsets = base + part * 64 * 128 + rows * 128 + cols
            src_offsets = tlx.require_layout(offsets, source)
            src_mask = tlx.require_layout(tl.broadcast_to(rows < valid_rows, (64, 128)), source)
            # Preserve source ownership through the load; each thread owns
            # groups of four contiguous BF16 elements.
            value = tlx.buffer_load(X, src_offsets, src_mask, other=0, contiguity=4)
            value = (value.to(tl.float32) + iteration).to(tl.bfloat16)
            # Both boundaries are hard pins so layout propagation cannot remove
            # the conversion or change its per-wave scratch partition.
            value = tlx.require_layout(value, source)
            converted = tlx.require_layout(value, dest)
            tl.store(Y + offsets, converted, rows < valid_rows)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("valid_rows", [1, 17, 31, 32, 33, 63, 64])
@pytest.mark.parametrize("iterations", [1, 3, 11])
def test_convert_layout_wave_local_scratch_reuse(valid_rows, iterations):
    # Repeated conversions must order each wave's loads before reusing its
    # physical scratch partition, including across the runtime loop backedge.
    shape = (32, 4, 64, 128)
    count = math.prod(shape)
    inp_cpu = ((torch.arange(count, dtype=torch.int32) * 17 + 31) % 251 - 125).to(torch.bfloat16).reshape(shape)
    inp = inp_cpu.to(DEVICE)
    raw_output = torch.full((count + 128, ), 256, dtype=torch.bfloat16, device=DEVICE)
    output = raw_output[64:-64].view(shape)
    compiled = _wave_local_scratch_reuse_kernel[(32, )](inp, output, valid_rows, iterations, num_warps=4)

    expected = torch.full((count + 128, ), 256, dtype=torch.bfloat16)
    expected[64:-64].view(shape)[:, :, :valid_rows, :] = inp_cpu[:, :, :valid_rows, :] + (iterations - 1)
    # Every value is an exactly representable BF16 integer. Check valid outputs,
    # untouched masked rows, and 64-element guards at both ends.
    torch.testing.assert_close(raw_output.cpu(), expected, atol=0, rtol=0)
    _assert_no_layout_residue(compiled.asm["ttgir"])
    # Keep this on the wave-local LDS path if conversion lowering changes.
    asm = compiled.asm["amdgcn"]
    assert re.search(r"\bds_write", asm)
    assert re.search(r"\bds_read", asm)
    assert "call void @llvm.amdgcn.wave.barrier()" in compiled.asm["llir"]
    # All conversions must reuse one tile's scratch image. Extra conversions
    # or CTA rendezvous between unrolled parts could hide the reuse hazard;
    # only the runtime loop backedge may require a CTA barrier.
    assert compiled.metadata.shared == 64 * 128 * inp.element_size()
    assert len(re.findall(r"\bs_barrier\b", asm)) <= 1


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_fp_cast_preserves_explicit_mfma_layout_on_cdna4():
    """Ordinary FP casts keep concrete source ownership until a new anchor."""
    mma = tlx.amd_mfma_layout(4, [16, 16, 32], True, [4, 1])
    store = tlx.layout(shape=((64, 4), (16, )), stride=((16, 1024), (1, )))

    @triton.jit
    def kernel(Y, MMA: tl.constexpr, STORE: tl.constexpr):
        acc = tlx.zeros((256, 16), tl.float32, layout=MMA)
        narrowed = acc.to(tl.bfloat16)
        extended = narrowed.to(tl.float32)
        extended = tlx.require_layout(extended, STORE)
        rows = tl.arange(0, 256)
        cols = tl.arange(0, 16)
        tl.store(Y + rows[:, None] * 16 + cols[None, :], extended)

    y = torch.full((256 * 16, ), float("nan"), device=DEVICE, dtype=torch.float32)
    kernel[(1, )](y, mma, store, num_warps=4)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_fp_cast_preserves_layout_for_equal_width_types_on_cdna4():
    """BF16/FP16 conversions retain ownership and produce numeric casts."""
    mma = tlx.amd_mfma_layout(4, [32, 32, 16], True, [4, 1])
    store = tlx.layout(
        shape=((64, 4), (2, 2, 2, 2, 2, 2, 2)),
        stride=((128, 8192), (1, 2, 4, 8, 16, 32, 64)),
    )

    @triton.jit
    def kernel(src_bf16, src_f16, dst_from_f16, dst_from_bf16, MMA: tl.constexpr, STORE: tl.constexpr):
        rows = tl.arange(0, 256)[:, None]
        cols = tl.arange(0, 128)[None, :]
        offsets = rows * 128 + cols
        bf16 = tlx.require_layout(tl.load(src_bf16 + offsets), MMA)
        f16 = tlx.require_layout(tl.load(src_f16 + offsets), MMA)
        from_f16 = tlx.require_layout(f16.to(tl.bfloat16), STORE)
        from_bf16 = tlx.require_layout(bf16.to(tl.float16), STORE)
        tl.store(dst_from_f16 + offsets, from_f16)
        tl.store(dst_from_bf16 + offsets, from_bf16)

    numel = 256 * 128
    values = torch.linspace(-1.0, 1.0, numel, device=DEVICE, dtype=torch.float32)
    src_bf16 = values.to(torch.bfloat16)
    src_f16 = (values * 0.75 + 0.125).to(torch.float16)
    dst_from_f16 = torch.full_like(src_bf16, float("nan"))
    dst_from_bf16 = torch.full_like(src_f16, float("nan"))
    kernel[(1, )](src_bf16, src_f16, dst_from_f16, dst_from_bf16, mma, store, num_warps=4)
    torch.testing.assert_close(dst_from_f16, src_f16.to(torch.bfloat16), atol=0, rtol=0)
    torch.testing.assert_close(dst_from_bf16, src_bf16.to(torch.float16), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_amd_mfma_layout_anchors_on_cdna4():
    """The Gluon MFMA/dot layout handles lower through store anchors.

    This is a compiler/API probe, not a performance claim: the FA pipeline
    keeps the generic register layout until TLX can form compatible C and
    transposed dS operands for the full dot chain.
    """
    shared = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [256, 128],
    )
    mma = tlx.amd_mfma_layout(4, [16, 16, 32], True, [4, 1])
    dot0 = tlx.dot_operand_layout(0, mma, 8)
    x_store = tlx.layout(shape=((64, 4), (128, )), stride=((128, 8192), (1, )))
    acc_store = tlx.layout(shape=((64, 4), (16, )), stride=((16, 1024), (1, )))

    @triton.jit
    def kernel(
        X,
        Y,
        SHARED: tl.constexpr,
        DOT0: tl.constexpr,
        MMA: tl.constexpr,
        X_STORE: tl.constexpr,
        ACC_STORE: tl.constexpr,
    ):
        buf = tlx.local_alloc((256, 128), tl.bfloat16, 1, layout=SHARED)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, tl.zeros((256, 128), tl.bfloat16))
        x = tlx.local_load(view, layout=DOT0)
        x = tlx.require_layout(x, X_STORE).to(tl.float32)
        rows = tl.arange(0, 256)
        cols = tl.arange(0, 128)
        tl.store(Y + rows[:, None] * 128 + cols[None, :], x)
        acc = tlx.zeros((256, 16), tl.float32, layout=MMA)
        acc = tlx.require_layout(acc, ACC_STORE)
        cols_acc = tl.arange(0, 16)
        tl.store(Y + 256 * 128 + rows[:, None] * 16 + cols_acc[None, :], acc)

    x = torch.zeros((256 * 128, ), device=DEVICE, dtype=torch.bfloat16)
    y = torch.zeros((256 * 128 + 256 * 16, ), device=DEVICE, dtype=torch.float32)
    compiled = kernel.warmup(x, y, shared, dot0, mma, x_store, acc_store, grid=(1, ), num_warps=4)
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    # Explicit-layout producers and public pins use the same SSA boundary.
    assert ttir.count("require_layout") >= 4
    assert "#ttg.amd_mfma" in ttir
    assert "#ttg.dot_op" in ttir
    assert "#tlx.user_layout" not in ttgir


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_tlx_dot_preserves_explicit_accumulator_layout_on_cdna4():
    """A standard ``tl.dot`` keeps an explicit AMD accumulator layout live.

    The semantic ``tl.dot`` path propagates an explicitly laid-out accumulator
    type, so the score dot can feed elementwise operations without an unresolved
    blocked-to-MFMA materialization.
    """
    mma = tlx.amd_mfma_layout(4, [16, 16, 32], True, [4, 1])
    dot0 = tlx.dot_operand_layout(0, mma, 8)
    dot1 = tlx.dot_operand_layout(1, mma, 8)
    shared = tlx.swizzled_shared_layout_encoding.make_default(2)
    store = tlx.layout(shape=((64, 4), (16, )), stride=((16, 1024), (1, )))

    @triton.jit
    def kernel(
        X,
        Y,
        SHARED: tl.constexpr,
        DOT0: tl.constexpr,
        DOT1: tl.constexpr,
        MMA: tl.constexpr,
        STORE: tl.constexpr,
    ):
        a_buf = tlx.local_alloc((256, 128), tl.bfloat16, 1, layout=SHARED)
        b_buf = tlx.local_alloc((128, 16), tl.bfloat16, 1, layout=SHARED)
        tlx.local_store(tlx.local_view(a_buf, 0), tl.zeros((256, 128), tl.bfloat16))
        tlx.local_store(tlx.local_view(b_buf, 0), tl.zeros((128, 16), tl.bfloat16))
        a = tlx.local_load(tlx.local_view(a_buf, 0), layout=DOT0)
        b = tlx.local_load(tlx.local_view(b_buf, 0), layout=DOT1)
        acc = tlx.zeros((256, 16), tl.float32, layout=MMA)
        out = tl.dot(a, b, acc=acc, out_dtype=acc.dtype)
        out = tlx.require_layout(out, STORE)
        rows = tl.arange(0, 256)
        cols = tl.arange(0, 16)
        tl.store(Y + rows[:, None] * 16 + cols[None, :], out)

    x = torch.zeros((256 * 128, ), device=DEVICE, dtype=torch.bfloat16)
    y = torch.zeros((256 * 16, ), device=DEVICE, dtype=torch.float32)
    compiled = kernel.warmup(x, y, shared, dot0, dot1, mma, store, grid=(1, ), num_warps=4)
    assert "#ttg.amd_mfma" in compiled.asm["ttir"]
    assert "#tlx.no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
@pytest.mark.parametrize("rotate_final", [False, True], ids=["stage-three", "stage-four-rotated"])
def test_mfma_split_concat_preserves_logical_columns_on_cdna4(rotate_final):
    """Order-preserving reshape/split/join reconstructs an MFMA score tile.

    Flash attention carries N8 probability fragments across source stages and
    later reassembles them for its row sum and P-by-V dot.  Marking these
    reshapes reorderable changes their logical register interpretation: the
    shapes still verify, but every reconstructed row can contain wrong values.
    Both orders are production schedules: stage three uses the ordinary chain,
    while the N8192+ causal stage-four schedule rotates the final two additions.
    """

    @triton.jit
    def split_cols(x):
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return x0, x1

    @triton.jit
    def concat_cols(x0, x1):
        return tl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] + x1.shape[1]])

    @triton.jit
    def kernel(X, Recon, Chain, Direct, MMA: tl.constexpr, ROTATE_FINAL: tl.constexpr):
        rows = tl.arange(0, 256)
        cols = tl.arange(0, 64)
        offsets = rows[:, None] * 64 + cols[None, :]
        x = tlx.require_layout(tl.load(X + offsets), MMA)
        x_lo, x_hi = split_cols(x)
        reconstructed = concat_cols(x_lo, x_hi)
        chain = _cluster_sum_rows_chain4(x, ROTATE_FINAL)
        direct = tl.sum(x, 1)
        tl.store(Recon + offsets, reconstructed)
        tl.store(Chain + rows, chain)
        tl.store(Direct + rows, direct)

    mma = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])
    torch.manual_seed(7)
    x = torch.rand((256, 64), device=DEVICE, dtype=torch.float32)
    reconstructed = torch.empty_like(x)
    chain = torch.empty((256, ), device=DEVICE, dtype=torch.float32)
    direct = torch.empty_like(chain)
    compiled = kernel[(1, )](
        x,
        reconstructed,
        chain,
        direct,
        mma,
        rotate_final,
        num_warps=8,
        enable_tree_reduction=True,
    )

    ttir = compiled.asm["ttir"]
    release_line = next(line for line in ttir.splitlines() if "tlx.release_layout" in line)
    release_operand = release_line.split("tlx.release_layout", 1)[1].split()[0]
    assert any(f'{release_operand} = "tt.reduce"' in line for line in ttir.splitlines())

    reference = x.sum(1)
    torch.testing.assert_close(reconstructed, x, atol=0, rtol=0)
    torch.testing.assert_close(chain, reference, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(direct, reference, atol=1e-5, rtol=1e-6)
    _assert_no_layout_residue(compiled.asm["ttgir"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_slice_layout_matches_mfma_row_reduction_on_cdna4():
    """The public slice layout names the rank-1 result of an MFMA row sum."""
    mma = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])
    rows = tlx.slice_layout(mma, dim=1)

    @triton.jit
    def kernel(X, Y, MMA: tl.constexpr, ROWS: tl.constexpr):
        offs_m = tl.arange(0, 256)
        offs_n = tl.arange(0, 64)
        offsets = offs_m[:, None] * 64 + offs_n[None, :]
        x = tlx.require_layout(tl.load(X + offsets), MMA)
        reduced = tl.reduce(x, 1, _pinned_add_combine)
        reduced = tlx.require_layout(reduced, ROWS)
        tlx.assert_same_layout(reduced, ROWS)
        tl.store(Y + offs_m, reduced)

    x = torch.rand((256, 64), device=DEVICE, dtype=torch.float32)
    y = torch.empty((256, ), device=DEVICE, dtype=torch.float32)
    compiled = kernel[(1, )](x, y, mma, rows, num_warps=8, enable_tree_reduction=True)
    torch.testing.assert_close(y, x.sum(1), atol=1e-5, rtol=1e-6)
    _assert_no_layout_residue(compiled.asm["ttgir"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_user_pinned_swizzled_padded_survives_amd():
    """A user-pinned *swizzled* padded_shared (built with `with_bases`) survives
    to final TTGIR as #ttg.padded_shared with the explicit {offset = ...} form,
    not the identity {order, shape}, and leaves no #tlx.user_layout residue."""

    @triton.jit
    def kernel(PAD: tl.constexpr, M: tl.constexpr, K: tl.constexpr):
        x = tl.zeros((M, K), tl.float16)
        buf = tlx.local_alloc((M, K), tl.float16, tl.constexpr(1), layout=PAD)
        v = tlx.local_view(buf, 0)
        tlx.local_store(v, x)
        y = tlx.local_load(v)
        tlx.local_store(v, y)

    pad = tlx.padded_shared_layout_encoding.with_bases(_A16W16_SHARED_INTERVALS, _A16W16_SHARED_OFFSET_BASES,
                                                       _A16W16_TILE)
    ttgir = kernel.warmup(pad, _A16W16_TILE[0], _A16W16_TILE[1], grid=(1, ), num_warps=8).asm["ttgir"]
    assert "#ttg.padded_shared" in ttgir
    assert "offset = [" in ttgir  # the explicit bases form, not {order, shape}
    _assert_no_layout_residue(ttgir)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_buffer_load_to_local_infers_offset_layout_amd():
    """With only the swizzled shared layout pinned on the alloc (no explicit
    offset layout), tlx-insert-require-layout infers the matching offset
    tensor's #linear so the direct-to-LDS load coalesces and lowers to amdgcn.
    The inferred #linear must equal the hand-derived a16w16 load layout."""

    @triton.jit
    def kernel(a_ptr, SHARED: tl.constexpr, M: tl.constexpr, K: tl.constexpr, STRIDE_M: tl.constexpr):
        offs_m = tl.arange(0, M)
        offs_k = tl.arange(0, K)
        off = offs_m[:, None] * STRIDE_M + offs_k[None, :]
        smem = tlx.local_alloc((M, K), tl.float16, tl.constexpr(1), layout=SHARED)
        tlx.buffer_load_to_local(smem[0], a_ptr, off, contiguity=8)

    pad = tlx.padded_shared_layout_encoding.with_bases(_A16W16_SHARED_INTERVALS, _A16W16_SHARED_OFFSET_BASES,
                                                       _A16W16_TILE)
    M, K = _A16W16_TILE
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    compiled = kernel.warmup(a, pad, M, K, K, grid=(1, ), num_warps=8)
    ttgir = compiled.asm["ttgir"]
    # The offset layout is inferred (not authored) and matches the hand-derived
    # a16w16 load layout, and the load stays a single direct-to-LDS op.
    expected = f"register = {_A16W16_LOAD_REG}, lane = {_A16W16_LOAD_LANE}, warp = {_A16W16_LOAD_WARP}"
    assert "#ttg.linear" in ttgir
    assert expected in ttgir, f"inferred offset layout mismatch; expected substring:\n{expected}\n\nttgir:\n{ttgir}"
    assert "amdg.buffer_load_to_local" in ttgir
    assert "contiguity = 8" in ttgir
    # It lowers all the way to amdgcn (the direct-to-LDS width/alignment
    # requirements are met by the inferred offset layout).
    assert compiled.asm.get("amdgcn")


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_require_layout_pins_epilogue_store_amd():
    """A store *value* pinned via tlx.require_layout survives coalesce /
    remove-layout-conversions / optimize-epilogue as the exact coalesced #linear,
    so the FP16 epilogue store stays a wide buffer_store_dwordx4 (the a16w16
    scenario) instead of being narrowed to the MMA-accumulator layout. The
    in-kernel tlx.assert_same_layout(c, L) compares final LinearLayouts and fails
    compilation if the pin is dropped."""

    @triton.jit
    def kernel(a_ptr, b_ptr, c_ptr, K: tl.constexpr, L: tl.constexpr):
        offs_m = tl.arange(0, 128)
        offs_n = tl.arange(0, 128)
        offs_k = tl.arange(0, K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * 128 + offs_n[None, :])
        acc = tl.dot(a, b)  # MMA accumulator -> the store OptimizeEpilogue rewrites
        c = tlx.require_layout(acc.to(tl.float16), L)  # pin the store value to L
        tlx.assert_same_layout(c, L)  # fails compilation if the pin didn't survive
        tl.store(c_ptr + offs_m[:, None] * 128 + offs_n[None, :], c)

    L = tlx.layout(shape=_A16W16_STORE_SHAPE, stride=_A16W16_STORE_STRIDE)
    a = torch.randn((128, 64), device=DEVICE, dtype=torch.float16)
    b = torch.randn((64, 128), device=DEVICE, dtype=torch.float16)
    c = torch.empty((128, 128), device=DEVICE, dtype=torch.float16)
    compiled = kernel.warmup(a, b, c, 64, L, grid=(1, ), num_warps=8)
    # assert_same_layout would have failed compilation if the pin were dropped; the
    # epilogue store lowers to the wide coalesced dwordx4, not the narrow dwordx2
    # fallback OptimizeEpilogue would otherwise produce.
    amdgcn = compiled.asm["amdgcn"]
    assert "buffer_store_dwordx4" in amdgcn
    assert "buffer_store_dwordx2" not in amdgcn


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_pinned_online_softmax_amd():
    """A pinned mfma `tl.dot` result feeds a blocked-init online-softmax
    (max/sub/exp2) and lowers correctly on AMD: the blocked m_i meets the
    mfma-derived reduce without an unresolved blocked->no_verify materialization,
    and the result stores directly (tl.store converts the pinned mfma layout)."""
    mma = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])
    dot0 = tlx.dot_operand_layout(0, mma, 8)
    dot1 = tlx.dot_operand_layout(1, mma, 8)

    @triton.jit
    def kernel(A, B, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, MMA: tl.constexpr, DOT0: tl.constexpr,
               DOT1: tl.constexpr):
        a = tlx.require_layout(tl.load(A + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]), DOT0)
        b = tlx.require_layout(tl.load(B + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]), DOT1)
        acc = tlx.require_layout(tlx.zeros([M, N], tl.float32, layout=MMA), MMA)
        qk = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
        m_i = tl.zeros([M], tl.float32) - float("inf")  # blocked init
        m_new = tl.maximum(m_i, tl.max(qk, 1))  # blocked vs slice<mfma>
        p = tl.exp2(qk - m_new[:, None])  # mfma vs expand(slice<mfma>)
        tl.store(Out + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], p)

    M, N, K = 256, 64, 64
    a = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=DEVICE, dtype=torch.float32)
    compiled = kernel[(1, )](a, b, out, M, N, K, mma, dot0, dot1, num_warps=8)
    torch.cuda.synchronize()
    qk = a.float() @ b.float()
    ref = torch.exp2(qk - qk.max(1, keepdim=True).values)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    assert "#ttg.amd_mfma" in compiled.asm["ttir"]
    _assert_no_layout_residue(compiled.asm["ttgir"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_pinned_softmax_scf_loop_amd():
    """A full online-softmax body with loop-carried acc/m_i inside an scf.for
    (mirrors tier5, no warp_pipeline). The ConvertLayoutOp source materialization
    keeps the pinned acc/m_i live across the loop back-edge instead of leaving an
    unresolvable blocked->no_verify materialization."""
    mma = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])
    dot0 = tlx.dot_operand_layout(0, mma, 8)
    dot1 = tlx.dot_operand_layout(1, mma, 8)

    @triton.jit
    def kernel(A, B, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, NBLK: tl.constexpr, MMA: tl.constexpr,
               DOT0: tl.constexpr, DOT1: tl.constexpr):
        a = tlx.require_layout(tl.load(A + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]), DOT0)
        b = tlx.require_layout(tl.load(B + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]), DOT1)
        acc = tlx.require_layout(tlx.zeros([M, N], tl.float32, layout=MMA), MMA)
        m_i = tl.zeros([M], tl.float32) - float("inf")
        for _ in range(NBLK):
            qk = tl.dot(a, b, acc=tlx.require_layout(tlx.zeros([M, N], tl.float32, layout=MMA), MMA),
                        out_dtype=tl.float32)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(qk - m_safe[:, None])
            alpha = tl.exp2(m_i - m_safe)
            acc = acc * alpha[:, None]
            acc = tl.dot(tlx.require_layout(p.to(tl.bfloat16), DOT0), b, acc=acc, out_dtype=tl.float32)
            m_i = m_new
        tl.store(Out + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], acc)

    M, N, K, NBLK = 256, 64, 64, 4
    a = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=DEVICE, dtype=torch.float32)
    compiled = kernel[(1, )](a, b, out, M, N, K, NBLK, mma, dot0, dot1, num_warps=8)
    torch.cuda.synchronize()
    # Every block is identical, so m stabilizes after iter 0 (alpha==1) and acc
    # accumulates NBLK copies of p@b (p = exp2(qk - rowmax(qk))).
    qk = a.float() @ b.float()
    p = torch.exp2(qk - qk.max(1, keepdim=True).values).to(torch.bfloat16).float()
    ref = NBLK * (p @ b.float())
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=ref.abs().max().item() * 3e-2, rtol=3e-2)
    ttgir = compiled.asm["ttgir"]
    assert "#ttg.amd_mfma" in ttgir
    assert "scf.for" in ttgir
    _assert_no_layout_residue(ttgir)
    assert compiled.asm.get("amdgcn")


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_pinned_loop_carried_dot_operand_amd():
    """A dot operand (b) is loop-carried and re-pinned each iteration (mirrors
    tier5's prefetched kt). The pinned dot_operand<mfma> must survive as a
    loop-carried value across the scf.for back-edge."""
    mma = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])
    dot0 = tlx.dot_operand_layout(0, mma, 8)
    dot1 = tlx.dot_operand_layout(1, mma, 8)

    @triton.jit
    def kernel(A, B, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, NBLK: tl.constexpr, MMA: tl.constexpr,
               DOT0: tl.constexpr, DOT1: tl.constexpr):
        a = tlx.require_layout(tl.load(A + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]), DOT0)
        b = tlx.require_layout(tl.load(B + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]), DOT1)
        acc = tlx.require_layout(tlx.zeros([M, N], tl.float32, layout=MMA), MMA)
        m_i = tl.zeros([M], tl.float32) - float("inf")
        for _ in range(NBLK):
            qk = tl.dot(a, b, acc=tlx.require_layout(tlx.zeros([M, N], tl.float32, layout=MMA), MMA),
                        out_dtype=tl.float32)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(qk - m_safe[:, None])
            alpha = tl.exp2(m_i - m_safe)
            acc = acc * alpha[:, None]
            acc = tl.dot(tlx.require_layout(p.to(tl.bfloat16), DOT0), b, acc=acc, out_dtype=tl.float32)
            m_i = m_new
            # re-pin b -> b is a loop-carried dot_operand<mfma>
            b = tlx.require_layout(tl.load(B + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]), DOT1)
        tl.store(Out + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], acc)

    M, N, K, NBLK = 256, 64, 64, 4
    a = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=DEVICE, dtype=torch.float32)
    compiled = kernel[(1, )](a, b, out, M, N, K, NBLK, mma, dot0, dot1, num_warps=8)
    torch.cuda.synchronize()
    qk = a.float() @ b.float()
    p = torch.exp2(qk - qk.max(1, keepdim=True).values).to(torch.bfloat16).float()
    ref = NBLK * (p @ b.float())
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=ref.abs().max().item() * 3e-2, rtol=3e-2)
    ttgir = compiled.asm["ttgir"]
    assert "#ttg.amd_mfma" in ttgir
    _assert_no_layout_residue(ttgir)
    assert compiled.asm.get("amdgcn")


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_cute_layout_with_standard_casts_on_cdna4():
    physical = tlx.layout(
        shape=((64, ), ()),
        stride=((1, ), ()),
    )

    @triton.jit
    def kernel(X, Y, Bits, PHYSICAL: tl.constexpr):
        offsets = tl.arange(0, 64)
        values = tl.load(X + offsets)
        pinned = tlx.require_layout(values, PHYSICAL)
        narrowed = pinned.to(tl.bfloat16)
        widened = narrowed.to(tl.float32)
        bits = pinned.to(tl.int32, bitcast=True)
        tl.store(Y + offsets, widened)
        tl.store(Bits + offsets, bits)

    x = torch.linspace(-3.0, 3.0, 64, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    bits = torch.empty(64, device=DEVICE, dtype=torch.int32)
    compiled = kernel.warmup(x, y, bits, physical, grid=(1, ), num_warps=1)
    kernel[(1, )](x, y, bits, physical, num_warps=1)

    torch.testing.assert_close(y, x.to(torch.bfloat16).float(), atol=0, rtol=0)
    torch.testing.assert_close(bits, x.view(torch.int32), atol=0, rtol=0)
    assert "#ttg.linear" in compiled.asm["ttir"]
    assert "#tlx.no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_internally_pinned_helper_result_specializes_abi_on_cdna4():
    layout = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])

    @triton.jit
    def kernel(Y, LAYOUT: tl.constexpr):
        row = tl.arange(0, 256)
        col = tl.arange(0, 32)
        value = tl.full((256, 32), 3.0, tl.float32)
        pinned = _fa_pin_helper_result(value, LAYOUT)
        tl.store(Y + row[:, None] * 32 + col[None, :], pinned)

    y = torch.empty((256 * 32, ), device=DEVICE, dtype=torch.float32)
    compiled = kernel.warmup(y, layout, grid=(1, ), num_warps=8)
    kernel[(1, )](y, layout, num_warps=8)
    torch.testing.assert_close(y, torch.full_like(y, 3.0), atol=0, rtol=0)
    assert "#tlx.no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_concrete_mfma_layout_reconciles_elementwise_broadcast_on_cdna4():
    mma = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])
    store = tlx.layout(
        shape=((64, 8), (16, )),
        stride=((16, 1024), (1, )),
    )

    @triton.jit
    def kernel(Y, MMA: tl.constexpr, STORE: tl.constexpr):
        acc = tlx.require_layout(tl.full((256, 32), 2.0, tl.float32), MMA)
        source_rows = tl.arange(0, 256).to(tl.float32) + 1.0
        workitems = _fa_mfma_rows_to_workitems(source_rows)
        rows = _fa_workitems_to_mfma_rows(workitems)
        out = (acc * rows[:, None]).to(tl.bfloat16)
        out = tlx.require_layout(out, STORE)
        row = tl.arange(0, 256)
        col = tl.arange(0, 32)
        tl.store(Y + row[:, None] * 32 + col[None, :], out)

    y = torch.full((256 * 32, ), float("nan"), device=DEVICE, dtype=torch.bfloat16)
    compiled = kernel.warmup(y, mma, store, grid=(1, ), num_warps=8)
    kernel[(1, )](y, mma, store, num_warps=8)
    expected = (2 * torch.arange(1, 257, device=DEVICE, dtype=torch.float32)).to(torch.bfloat16)
    expected = expected[:, None].broadcast_to((256, 32)).flatten()
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    assert "#tlx.no_verify_layout" not in compiled.asm["ttgir"]
