"""TLX layout tests -- CDNA4 (gfx950)."""
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip_cdna4
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


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


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_user_register_layout_smem_loop_carried_amd():
    """Alternating user/no-verify wrapper order must disappear across an AMD loop edge."""

    @triton.jit
    def kernel(REG: tl.constexpr, n):
        initial = tl.zeros((256, 8), tl.int8)
        buf = tlx.local_alloc((256, 8), tl.int8, tl.constexpr(1))
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, initial)
        value = tlx.local_load(view, layout=REG)
        for _ in tl.range(0, n, num_stages=1):
            value += 1
        tlx.local_store(view, value)

    reg = tlx.layout(
        shape=((32, 2, 4), (8, )),
        stride=((64, 1, 2), (8, )),
    )
    compiled = kernel.warmup(reg, 4, grid=(1, ), num_warps=4)
    _assert_no_layout_residue(compiled.asm["ttgir"])


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


@triton.jit
def _release_reshaped_helper_input(value):
    reshaped = value.reshape([256, 2, 16]).permute(0, 2, 1).reshape([256, 32])
    return tlx.release_layout(reshaped)


@triton.jit
def _shape_preserving_reduce_broadcast(value):
    rows = tl.sum(value, axis=1)
    return tl.broadcast_to(rows[:, None], value.shape)


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
    # Hard destination anchors express both conversions without a public
    # release-layout operation.
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
def test_mfma_split_concat_preserves_logical_columns_on_cdna4():
    """Order-preserving reshape/split/join reconstructs an MFMA score tile.

    Flash attention carries N8 probability fragments across source stages and
    later reassembles them for its row sum and P-by-V dot.  Marking these
    reshapes reorderable changes their logical register interpretation: the
    shapes still verify, but every reconstructed row can contain wrong values.
    """

    @triton.jit
    def split_cols(x):
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return x0, x1

    @triton.jit
    def concat_cols(x0, x1):
        return tl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] + x1.shape[1]])

    @triton.jit
    def sum_rows_chain4(x):
        x_01, x_23 = split_cols(x)
        x_0, x_1 = split_cols(x_01)
        x_2, x_3 = split_cols(x_23)
        return tl.sum(x_0 + x_1 + x_2 + x_3, 1)

    @triton.jit
    def kernel(X, Recon, Chain, Direct, MMA: tl.constexpr):
        rows = tl.arange(0, 256)
        cols = tl.arange(0, 64)
        offsets = rows[:, None] * 64 + cols[None, :]
        x = tlx.require_layout(tl.load(X + offsets), MMA)
        x_lo, x_hi = split_cols(x)
        reconstructed = concat_cols(x_lo, x_hi)
        chain = sum_rows_chain4(x)
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
    compiled = kernel[(1, )](x, reconstructed, chain, direct, mma, num_warps=8, enable_tree_reduction=True)

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
        tlx.buffer_load_to_local(smem[0], a_ptr, off)

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
def test_release_layout_accepts_deferred_helper_input_on_cdna4():
    layout = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])

    @triton.jit
    def kernel(Y, LAYOUT: tl.constexpr):
        row = tl.arange(0, 256)
        col = tl.arange(0, 32)
        value = tl.full((256, 32), 5.0, tl.float32)
        pinned = tlx.require_layout(value, LAYOUT)
        released = _release_reshaped_helper_input(pinned)
        tl.store(Y + row[:, None] * 32 + col[None, :], released)

    y = torch.empty((256 * 32, ), device=DEVICE, dtype=torch.float32)
    compiled = kernel.warmup(y, layout, grid=(1, ), num_warps=8)
    kernel[(1, )](y, layout, num_warps=8)
    torch.testing.assert_close(y, torch.full_like(y, 5.0), atol=0, rtol=0)
    assert "#tlx.no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_pinned_helper_does_not_copy_layout_across_shape_cycle_on_cdna4():
    layout = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])

    @triton.jit
    def kernel(Y, LAYOUT: tl.constexpr):
        row = tl.arange(0, 256)
        col = tl.arange(0, 32)
        value = tl.full((256, 32), 1.0, tl.float32)
        pinned = tlx.require_layout(value, LAYOUT)
        result = _shape_preserving_reduce_broadcast(pinned)
        tl.store(Y + row[:, None] * 32 + col[None, :], result)

    y = torch.empty((256 * 32, ), device=DEVICE, dtype=torch.float32)
    compiled = kernel.warmup(y, layout, grid=(1, ), num_warps=8)
    kernel[(1, )](y, layout, num_warps=8)
    torch.testing.assert_close(y, torch.full_like(y, 32.0), atol=0, rtol=0)
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


@pytest.mark.skipif(not is_hip_cdna4(), reason="Need gfx950 (CDNA4)")
def test_concrete_elementwise_conflicting_explicit_layouts_fail_on_cdna4():
    mma = tlx.amd_mfma_layout(4, [32, 32, 16], True, [8, 1])
    other = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        warp_bases=[[64, 0], [128, 0], [0, 16]],
        block_bases=[],
        shape=[256, 32],
    )

    @triton.jit
    def kernel(X, Y, MMA: tl.constexpr, OTHER: tl.constexpr):
        row = tl.arange(0, 256)
        col = tl.arange(0, 32)
        offsets = row[:, None] * 32 + col[None, :]
        value = tl.load(X + offsets)
        lhs = tlx.require_layout(value, MMA)
        rhs = tlx.require_layout(value, OTHER)
        out = tlx.require_layout(lhs + rhs, OTHER)
        tl.store(Y + offsets, out)

    x = torch.zeros((256 * 32, ), device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    with pytest.raises(RuntimeError, match="conflicting user-pinned layouts meet on this operation"):
        kernel.warmup(x, y, mma, other, grid=(1, ), num_warps=8)
