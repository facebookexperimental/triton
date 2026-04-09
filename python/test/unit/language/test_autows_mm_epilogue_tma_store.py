"""
Unit tests for mm + pointwise epilogue fusion with TMA descriptor store
and automatic warp specialization.

Tests that tt.descriptor_store inside a warp-specialized loop compiles
and runs correctly with Meta WS on Blackwell. The epilogue computes
x * (2 * sigmoid(mm(x, W.T))) — a fused mm + sigmoid + mul pattern.

Authored with Claude.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _subtile_accumulator(
    acc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_FACTOR: tl.constexpr,
):
    tl.static_assert(SUBTILE_FACTOR > 0, "SUBTILE_FACTOR must be positive")
    tl.static_assert(
        (SUBTILE_FACTOR & (SUBTILE_FACTOR - 1)) == 0,
        "SUBTILE_FACTOR must be a power of 2",
    )
    if SUBTILE_FACTOR == 1:
        return (acc,)
    else:
        tl.static_assert(BLOCK_N % 2 == 0)
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        left, right = tl.split(acc)
        left_subtiles = _subtile_accumulator(left, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2)
        right_subtiles = _subtile_accumulator(right, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2)
        return left_subtiles + right_subtiles


@triton.jit
def mm_sigmoid_tma_store_kernel(
    x_ptr,
    A,
    B,
    out_ptr,
    ws_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    FLATTEN: tl.constexpr,
    A_COL_MAJOR: tl.constexpr,
    B_COL_MAJOR: tl.constexpr,
):
    """mm + sigmoid epilogue with TMA descriptor store.

    Computes: x * (2 * sigmoid(mm(x, W.T)))
    The epilogue loads x via pointer, applies sigmoid/mul, stores via TMA.
    """
    GROUP_M: tl.constexpr = 8
    NUM_SMS: tl.constexpr = 148

    stride_ak = 1
    stride_bn = 1
    if A_COL_MAJOR:
        a_desc = tl.make_tensor_descriptor(
            A, shape=[K, M], strides=[stride_am, 1],
            block_shape=[BLOCK_K, BLOCK_M],
        )
    else:
        a_desc = tl.make_tensor_descriptor(
            A, shape=[M, K], strides=[stride_am, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )
    if B_COL_MAJOR:
        b_desc = tl.make_tensor_descriptor(
            B, shape=[K, N], strides=[stride_bk, 1],
            block_shape=[BLOCK_K, BLOCK_N],
        )
    else:
        b_desc = tl.make_tensor_descriptor(
            B, shape=[N, K], strides=[stride_bk, 1],
            block_shape=[BLOCK_N, BLOCK_K],
        )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE],
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=True
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS)
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            if A_COL_MAJOR:
                a = tl.load_tensor_descriptor(a_desc, [offs_k, offs_am]).T
            else:
                a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
            if B_COL_MAJOR:
                b = tl.load_tensor_descriptor(b_desc, [offs_k, offs_bn]).T
            else:
                b = tl.load_tensor_descriptor(b_desc, [offs_bn, offs_k])
            accumulator += tl.dot(a, b.T, allow_tf32=False)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_M
        offs_cn = pid_n * BLOCK_N

        subtiles = _subtile_accumulator(accumulator, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE)
        for i in tl.static_range(EPILOGUE_SUBTILE):
            subtile = subtiles[i]
            offs_cn_i = offs_cn + i * (BLOCK_N // EPILOGUE_SUBTILE)

            # Epilogue: load x, compute sigmoid(mm) * 2 * x, TMA store
            xindex = (offs_cm + tl.arange(0, BLOCK_M))[:, None]
            yindex = (offs_cn_i + tl.arange(0, BLOCK_N // EPILOGUE_SUBTILE))[None, :]
            ymask = yindex < N
            x_val = tl.load(x_ptr + (yindex + N * xindex), ymask).to(tl.float32)
            fused = x_val * (2.0 * tl.sigmoid(subtile))
            out_desc.store([offs_cm, offs_cn_i], fused.to(tl.bfloat16))


@pytest.mark.parametrize("M, N, K", [(512, 256, 256), (4096, 256, 256)])
@pytest.mark.parametrize("BLOCK_M", [128, 256])
@pytest.mark.parametrize("BLOCK_N", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("FLATTEN", [False])  # True is blocked by fb-triton WS bug
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2])
@pytest.mark.parametrize("A_col_major", [False])
@pytest.mark.parametrize("B_col_major", [False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_mm_sigmoid_epilogue_tma_store(
    M, N, K,
    BLOCK_M, BLOCK_N, BLOCK_K,
    num_stages, num_warps,
    FLATTEN,
    EPILOGUE_SUBTILE,
    A_col_major, B_col_major,
):
    """Test mm + sigmoid epilogue with TMA descriptor store and Meta WS."""
    # Skip configs that exceed hardware resource limits
    if BLOCK_M == 256 and BLOCK_N == 256:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if BLOCK_N == 256 and not FLATTEN:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if BLOCK_N == 256 and BLOCK_K == 128 and num_stages == 3:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if not FLATTEN and BLOCK_K == 128 and B_col_major and not A_col_major:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    if BLOCK_M == 256 and not FLATTEN and (BLOCK_K == 128 or num_stages == 3):
        pytest.skip("Out of resources: shared memory exceeded")

    if BLOCK_M == 256 and FLATTEN and BLOCK_K == 128 and num_stages == 3 and EPILOGUE_SUBTILE != 4:
        pytest.skip("Out of resources: shared memory exceeded")

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True

        dtype = torch.bfloat16
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            x = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            x = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            W = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            W = torch.randn((N, K), dtype=dtype, device=device)
        out = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        workspace = torch.empty(37888, dtype=torch.uint8, device=device)

        grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

        kernel = mm_sigmoid_tma_store_kernel[grid](
            x, x, W, out, workspace,
            M, N, K,
            x.stride(0),  # stride_am
            W.stride(0),  # stride_bk
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            FLATTEN=FLATTEN,
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        # Verify TTGIR contains structurally partitioned WS + TMA store
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp_specialize in TTGIR"
        assert "partition0(" in ttgir, "Expected partition0 region in TTGIR"
        assert "partition1(" in ttgir, "Expected partition1 region in TTGIR"
        assert "partition2(" in ttgir, "Expected partition2 region in TTGIR"
        assert "async_tma_copy_local_to_global" in ttgir, "Expected TMA store in TTGIR"

        # Verify PTX contains structural WS indicators
        ptx = kernel.asm["ptx"]
        assert "setmaxnreg" in ptx, "Expected setmaxnreg (WS register allocation) in PTX"

        # Verify correctness: x * (2 * sigmoid(mm(x, W.T)))
        mm_ref = torch.matmul(x.float(), W.T.float())
        ref_out = (x.float() * (2.0 * torch.sigmoid(mm_ref))).to(dtype)
        torch.testing.assert_close(ref_out, out, atol=1e-1, rtol=1e-1)
