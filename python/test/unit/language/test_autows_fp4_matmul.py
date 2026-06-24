import pytest
import torch
import triton
import triton.language as tl
from test_mxfp import MXFP4Tensor
from triton._internal_testing import is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def fp4_matmul_tma_ws(
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
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=True, num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(REP_M, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(REP_N, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        offs_k += BLOCK_K // 2
        offs_scale_k += REP_K

    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator)


def _make_unit_scale_5d(scale_kind, rows, k, vec_size, device):
    scale_cols = k // vec_size
    if scale_kind == "mxfp4":
        raw_scale = torch.full((rows, scale_cols), 127, dtype=torch.uint8, device=device)
    else:
        raw_scale = torch.ones((rows, scale_cols), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    return raw_scale.reshape(1, rows // 128, scale_cols // 4, 2, 256).contiguous()


@pytest.mark.parametrize(
    ("scale_kind", "vec_size", "expected_ptx"),
    [
        ("mxfp4", 32, "kind::mxf4.block_scale"),
        ("nvfp4", 16, "kind::mxf4nvf4.block_scale"),
    ],
)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_autows_fp4_matmul_tma(scale_kind, vec_size, expected_ptx, device):
    if scale_kind == "nvfp4" and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("NVFP4 scales require torch.float8_e4m3fn")

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.disable_wsbarrier_reorder = True

        M, N, K = 128, 128, 256
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
        rep_m = BLOCK_M // 128
        rep_n = BLOCK_N // 128
        rep_k = BLOCK_K // vec_size // 4

        torch.manual_seed(42)
        a_fp4 = MXFP4Tensor(size=(M, K), device=device).random()
        b_fp4 = MXFP4Tensor(size=(N, K), device=device).random()
        a = a_fp4.to_packed_tensor(dim=1).contiguous()
        b = b_fp4.to_packed_tensor(dim=1).contiguous()
        scale_a = _make_unit_scale_5d(scale_kind, M, K, vec_size, device)
        scale_b = _make_unit_scale_5d(scale_kind, N, K, vec_size, device)
        c = torch.empty((M, N), dtype=torch.float32, device=device)

        def alloc_fn(size, _align, _stream):
            return torch.empty(size, dtype=torch.int8, device=device)

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // 2])
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // 2])
        c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])
        a_scale_desc = TensorDescriptor.from_tensor(scale_a, [1, rep_m, rep_k, 2, 256])
        b_scale_desc = TensorDescriptor.from_tensor(scale_b, [1, rep_n, rep_k, 2, 256])

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
        kernel = fp4_matmul_tma_ws[grid](
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
            NUM_STAGES=2,
            num_warps=4,
        )

        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma_scaled" in ttgir, "Expected scaled Blackwell MMA instruction"
        assert expected_ptx in kernel.asm["ptx"]

        ref = torch.matmul(a_fp4.to(torch.float32), b_fp4.to(torch.float32).T)
        torch.testing.assert_close(ref, c, atol=1e-2, rtol=1e-2)
