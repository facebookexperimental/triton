import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
from triton.tools.mxfp import MXFP4Tensor
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def quantized_matmul_tma_ws(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REP_M: tl.constexpr,
    REP_N: tl.constexpr,
    REP_K: tl.constexpr,
    A_TYPE: tl.constexpr,
    B_TYPE: tl.constexpr,
    PACK_FACTOR: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    TWO_CTAS: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k = 0
    offs_scale_m = pid_m * REP_M
    offs_scale_n = pid_n * REP_N
    offs_scale_k = 0

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=True):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = (scale_a.reshape(REP_M, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE))
        scale_b = (scale_b.reshape(REP_N, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE))

        accumulator = tl.dot_scaled(a, scale_a, A_TYPE, b.T, scale_b, B_TYPE, accumulator, two_ctas=TWO_CTAS)
        offs_k += BLOCK_K // PACK_FACTOR
        offs_scale_k += REP_K

    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator)


def _make_unit_scale_5d(scale_kind, rows, k, vec_size, device):
    scale_cols = k // vec_size
    if scale_kind in ("mxfp4", "mxfp8"):
        raw_scale = torch.full((rows, scale_cols), 127, dtype=torch.uint8, device=device)
    else:
        raw_scale = torch.ones((rows, scale_cols), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    return raw_scale.reshape(1, rows // 128, scale_cols // 4, 2, 256).contiguous()


def _make_varied_e8m0_scale_5d(rows, k, vec_size, device):
    """A 5D MXFP8 scale whose value differs per (row, K-group).

    Unit scales cannot validate scale *addressing*: every element is the same,
    so reading the wrong row or K-group still yields the right answer. This
    matters most for the 2-CTA path, where the B operand is split across the
    pair but the B scale is deliberately kept full width and each CTA addresses
    its own N-half out of it.

    Exponents stay in [125, 130] (0.25x .. 8x) so the reference stays
    well-conditioned against an e5m2 MMA.
    """
    scale_cols = k // vec_size
    raw = torch.randint(125, 131, (rows, scale_cols), dtype=torch.uint8, device=device)
    return raw.reshape(1, rows // 128, scale_cols // 4, 2, 256).contiguous()


def _decode_e8m0_scale_5d(scale_5d, rows, k, vec_size):
    """Undo the 5D blocked layout the same way the kernel does.

    Mirrors the in-kernel `reshape(REP, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4)`
    exactly, so the reference consumes whatever the kernel consumes rather than
    a separately re-derived swizzle.
    """
    scale_cols = k // vec_size
    rep = rows // 128
    rep_k = scale_cols // 4
    logical = (scale_5d.reshape(rep, rep_k, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, scale_cols))
    return torch.exp2(logical.to(torch.float32) - 127.0)


def _scaled_reference(a_ref, b_ref, scale_a_5d, scale_b_5d, k, vec_size):
    """Row-scaled fp32 reference: (a * sa) @ (b * sb).T with per-K-group scales."""
    sa = _decode_e8m0_scale_5d(scale_a_5d, a_ref.shape[0], k, vec_size)
    sb = _decode_e8m0_scale_5d(scale_b_5d, b_ref.shape[0], k, vec_size)
    a_scaled = a_ref * sa.repeat_interleave(vec_size, dim=1)
    b_scaled = b_ref * sb.repeat_interleave(vec_size, dim=1)
    return torch.matmul(a_scaled, b_scaled.T)


def _make_quantized_input(data_kind, size, device):
    if data_kind in ("mxfp4", "nvfp4"):
        tensor = MXFP4Tensor(size=size, device=device).random()
        return tensor.to_packed_tensor(dim=1).contiguous(), tensor.to(torch.float32)
    if data_kind == "mxfp8":
        tensor = torch.randint(20, 40, size, dtype=torch.uint8, device=device).view(torch.float8_e5m2)
        return tensor.contiguous(), tensor.to(torch.float32)
    raise AssertionError(f"Unsupported data kind: {data_kind}")


@pytest.mark.parametrize(
    ("data_kind", "scale_kind", "vec_size", "a_type", "b_type", "expected_ptx"),
    [
        ("mxfp4", "mxfp4", 32, "e2m1", "e2m1", "kind::mxf4.block_scale"),
        ("nvfp4", "nvfp4", 16, "e2m1", "e2m1", "kind::mxf4nvf4.block_scale"),
        ("mxfp8", "mxfp8", 32, "e5m2", "e5m2", "kind::mxf8f6f4.block_scale"),
    ],
)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_quantized_matmul_tma(data_kind, scale_kind, vec_size, a_type, b_type, expected_ptx, device):
    if scale_kind == "nvfp4" and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("NVFP4 scales require torch.float8_e4m3fn")
    if data_kind == "mxfp8" and not hasattr(torch, "float8_e5m2"):
        pytest.skip("MXFP8 inputs require torch.float8_e5m2")

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.disable_wsbarrier_reorder = True

        M, N, K = 128, 128, 256
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
        rep_m = BLOCK_M // 128
        rep_n = BLOCK_N // 128
        rep_k = BLOCK_K // vec_size // 4
        pack_factor = 2 if data_kind in ("mxfp4", "nvfp4") else 1

        torch.manual_seed(42)
        a, a_ref = _make_quantized_input(data_kind, (M, K), device)
        b, b_ref = _make_quantized_input(data_kind, (N, K), device)
        scale_a = _make_unit_scale_5d(scale_kind, M, K, vec_size, device)
        scale_b = _make_unit_scale_5d(scale_kind, N, K, vec_size, device)
        c = torch.empty((M, N), dtype=torch.float32, device=device)

        def alloc_fn(size, _align, _stream):
            return torch.empty(size, dtype=torch.int8, device=device)

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // pack_factor])
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // pack_factor])
        c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])
        a_scale_desc = TensorDescriptor.from_tensor(scale_a, [1, rep_m, rep_k, 2, 256])
        b_scale_desc = TensorDescriptor.from_tensor(scale_b, [1, rep_n, rep_k, 2, 256])

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
        kernel = quantized_matmul_tma_ws[grid](
            a_desc,
            a_scale_desc,
            b_desc,
            b_scale_desc,
            c_desc,
            M,
            N,
            K,
            vec_size,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            rep_m,
            rep_n,
            rep_k,
            a_type,
            b_type,
            pack_factor,
            NUM_STAGES=2,
            num_warps=4,
        )

        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert ("ttng.tc_gen5_mma_scaled" in ttgir), "Expected scaled Blackwell MMA instruction"
        assert expected_ptx in kernel.asm["ptx"]

        ref = torch.matmul(a_ref, b_ref.T)
        torch.testing.assert_close(ref, c, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_quantized_matmul_tma_2cta(device):
    """MXFP8 scaled matmul on a 2-CTA (cta_group::2) collective MMA.

    Launched with ctas_per_cga=(2, 1, 1), so each CTA is its own program and the
    pair cooperates only through the collective MMA. M is sized to two BLOCK_M
    tiles so the grid is an exact multiple of the cluster and adjacent programs
    pair up along M, matching how the compiler splits the B operand.
    """
    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("MXFP8 inputs require torch.float8_e5m2")

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.disable_wsbarrier_reorder = True

        vec_size = 32
        M, N, K = 256, 128, 256
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
        rep_m = BLOCK_M // 128
        rep_n = BLOCK_N // 128
        rep_k = BLOCK_K // vec_size // 4

        torch.manual_seed(42)
        a, a_ref = _make_quantized_input("mxfp8", (M, K), device)
        b, b_ref = _make_quantized_input("mxfp8", (N, K), device)
        # Nonuniform scales: the compiler splits B across the CTA pair but keeps
        # the B scale full width, so a wrong N-half or K-group in the scale
        # addressing must change the result. Unit scales cannot show that.
        scale_a = _make_varied_e8m0_scale_5d(M, K, vec_size, device)
        scale_b = _make_varied_e8m0_scale_5d(N, K, vec_size, device)
        c = torch.empty((M, N), dtype=torch.float32, device=device)

        def alloc_fn(size, _align, _stream):
            return torch.empty(size, dtype=torch.int8, device=device)

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K])
        c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])
        a_scale_desc = TensorDescriptor.from_tensor(scale_a, [1, rep_m, rep_k, 2, 256])
        b_scale_desc = TensorDescriptor.from_tensor(scale_b, [1, rep_n, rep_k, 2, 256])

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
        assert grid[0] % 2 == 0, "ctas_per_cga=(2,1,1) requires an even CTA grid"

        kernel = quantized_matmul_tma_ws[grid](
            a_desc,
            a_scale_desc,
            b_desc,
            b_scale_desc,
            c_desc,
            M,
            N,
            K,
            vec_size,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            rep_m,
            rep_n,
            rep_k,
            "e5m2",
            "e5m2",
            1,
            NUM_STAGES=2,
            TWO_CTAS=True,
            num_warps=4,
            ctas_per_cga=(2, 1, 1),
        )

        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma_scaled" in ttgir, "Expected scaled Blackwell MMA instruction"
        assert "two_ctas" in ttgir, "Expected the scaled MMA to stay 2-CTA (no 1-CTA fallback)"
        assert "cta_group::2" in kernel.asm["ptx"], "Expected a collective 2-CTA MMA in PTX"

        ref = _scaled_reference(a_ref, b_ref, scale_a, scale_b, K, vec_size)
        # Scales span 0.25x..8x per side, so the magnitude range is much wider
        # than in the unit-scale tests; scale the absolute tolerance with it.
        torch.testing.assert_close(ref, c, rtol=1e-2, atol=1e-2 * ref.abs().max().item())
