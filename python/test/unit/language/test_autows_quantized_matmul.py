import pytest
import torch
import triton
import triton.language as tl
from test_mxfp import MXFP4Tensor
from triton._internal_testing import is_blackwell
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

        accumulator = tl.dot_scaled(a, scale_a, A_TYPE, b.T, scale_b, B_TYPE, accumulator)
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
