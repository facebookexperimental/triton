import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda
import triton.tlx.language as tlx
from typing import Optional


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_async_tasks(BLOCK_SIZE, device):

    @triton.jit
    def add2_warp_specialized_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            with tlx.async_task("default"):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                output = x + y
                tl.store(z_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=4, registers=100):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                output = a + b
                tl.store(c_ptr + offsets, output, mask=mask)

    def dual_add(x, y, a, b):
        return x + y, a + b

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)

    output1 = torch.empty_like(x)
    output2 = torch.empty_like(a)
    n_elements = output1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = add2_warp_specialized_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE)
    ttgir = kernel.asm["ttgir"]
    assert "ttg.warp_specialize" in ttgir
    assert "requestedRegisters" in ttgir

    ref_out1, ref_out2 = dual_add(x, y, a, b)
    torch.testing.assert_close(output1, ref_out1, check_dtype=False)
    torch.testing.assert_close(output2, ref_out2, check_dtype=False)


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_local_load(BLOCK_SIZE, device):

    @triton.jit
    def local_load(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_ptr_offsets = x_ptr + offsets
        y_ptr_offsets = y_ptr + offsets

        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(2))
        buffer0 = tlx.local_view(buffers, 0)
        buffer1 = tlx.local_view(buffers, 1)
        tlx.async_load(x_ptr_offsets, buffer0, mask=mask)
        tlx.async_load(y_ptr_offsets, buffer1, mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        x_local = tlx.local_load(buffer0)
        y_local = tlx.local_load(buffer1)
        local_add = x_local + y_local
        tl.store(output_ptr + offsets, local_add, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = local_load[grid](x, y, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_subview") == 2
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 2
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    assert kernel.asm["ttgir"].count("ttg.local_load") == 2
    torch.testing.assert_close(x + y, output)


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_local_store(BLOCK_SIZE, device):

    @triton.jit
    def local_load_store(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_ptr_offsets = x_ptr + offsets
        y_ptr_offsets = y_ptr + offsets

        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(4))
        buffer0 = tlx.local_view(buffers, 0)
        buffer1 = tlx.local_view(buffers, 1)
        buffer2 = tlx.local_view(buffers, 2)
        tlx.async_load(x_ptr_offsets, buffer0, mask=mask)
        tlx.async_load(y_ptr_offsets, buffer1, mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        x_local = tlx.local_load(buffer0)
        y_local = tlx.local_load(buffer1)
        local_add = x_local + y_local
        # store result into buffer2 and then load it
        tlx.local_store(buffer2, local_add)
        result = tlx.local_load(buffer2)
        tl.store(output_ptr + offsets, result, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = local_load_store[grid](x, y, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_subview") == 3
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 2
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    assert kernel.asm["ttgir"].count("ttg.local_load") == 3
    assert kernel.asm["ttgir"].count("ttg.local_store") == 1
    torch.testing.assert_close(x + y, output)


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires compute capability >= 10 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_tmem_alloc_index(BLOCK_SIZE, device):

    @triton.jit
    def kernel(BLOCK_SIZE: tl.constexpr, ):
        buffers = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.tmem)
        buffer0 = tlx.local_view(buffers, 0)  # noqa: F841
        buffer1 = tlx.local_view(buffers, 1)  # noqa: F841

    grid = lambda meta: (1, )
    kerenl_info = kernel[grid](BLOCK_SIZE)
    # TODO: check numerics once tmem load/store is ready
    kerenl_info.asm["ttgir"]
    assert kerenl_info.asm["ttgir"].count("kernel") == 1


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

    output = torch.zeros(32, dtype=torch.int32, device='cuda')
    n_elements = output.numel()
    value = 42
    store_from_thread_0_kernel[(1, )](output, value, n_elements, 0, 32, num_warps=1)
    torch.cuda.synchronize()
    expected_output = torch.zeros(32, dtype=torch.int32, device='cuda')
    expected_output[0] = value
    torch.testing.assert_close(output, expected_output)


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_async_wait(BLOCK_SIZE, device):

    @triton.jit
    def async_wait_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input_ptr_offsets = input_ptr + offsets
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        tlx.async_load(input_ptr_offsets, buffer, mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        x = tlx.local_load(buffer)
        tl.store(output_ptr + offsets, x, mask=mask)

    @triton.jit
    def async_wait_token_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input_ptr_offsets = input_ptr + offsets
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        token = tlx.async_load(input_ptr_offsets, buffer, mask=mask)
        token = tlx.async_load_commit_group([token])
        tlx.async_load_wait_group(tl.constexpr(0), [token])
        x = tlx.local_load(buffer)
        tl.store(output_ptr + offsets, x, mask=mask)

    torch.manual_seed(0)
    size = 64
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = async_wait_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 1
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    torch.testing.assert_close(x, output)
    kernel = async_wait_token_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    torch.testing.assert_close(x, output)


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
def test_local_trans(device):

    @triton.jit
    def local_trans_kernel(
        input_ptr,
        output_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Compute global offsets
        input_offset = off_m[:, None] * N + off_n[None, :]
        output_offset = off_n[:, None] * M + off_m[None, :]

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1))
        buffer0 = tlx.local_view(buffers, 0)
        tlx.async_load(input_ptr + input_offset, buffer0)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        buffer1 = tlx.local_trans(buffer0)
        transposed = tlx.local_load(buffer1)
        tl.store(output_ptr + output_offset, transposed)

    torch.manual_seed(0)
    M, N = 32, 64
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 64
    x = torch.rand((M, N), dtype=torch.float32, device=device)
    y = torch.empty((N, M), dtype=torch.float32, device=device)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    kernel = local_trans_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=1)
    assert kernel.asm["ttgir"].count("ttg.memdesc_trans") == 1
    torch.testing.assert_close(y, x.T)


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires compute capability == 9 for NV",
)
def test_async_dot(device):
    """
    Define a unit test with similar schema with tl.dot
    @triton.jit
    def ref_kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, INPUT_PRECISION: tl.constexpr, out_dtype: tl.constexpr = tl.float32):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y, input_precision=INPUT_PRECISION, out_dtype=out_dtype)
        tl.store(Zs, z)
    """

    @triton.jit
    def tgt_kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, Z, stride_zm, stride_zn, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, INPUT_PRECISION: tl.constexpr, out_dtype: tl.constexpr,
                   COL_INPUT: tl.constexpr, COL_OTHER: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)

        buf_alloc_x = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float32, 1)
        buf_alloc_y = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float32, 1)
        a_smem = tlx.local_view(buf_alloc_x, 0)
        b_smem = tlx.local_view(buf_alloc_y, 0)

        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn

        # TODO. initialize values or async load

        z = tlx.async_dot(a_smem, b_smem, input_precision=INPUT_PRECISION, out_dtype=out_dtype, col_input=COL_INPUT,
                          col_other=COL_OTHER)
        tl.store(Zs, z)

    M, N, K = (64, 64, 64)
    x = torch.ones((M, K), device=device, dtype=torch.float32)
    y = torch.ones((K, N), device=device, dtype=torch.float32)
    z = torch.empty_like(x, device=device, dtype=torch.float32)

    kern_kwargs = {
        'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N, 'INPUT_PRECISION': "tf32", 'out_dtype': tl.float32, 'COL_INPUT': 0,
        'COL_OTHER': 1
    }
    with pytest.raises(RuntimeError) as _:
        _ = tgt_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0), z.stride(1),
                               **kern_kwargs)

    # TODO. assert "ttng.warp_group_dot" in pgm["ttir"] but not accessible due to thrown RuntimeError
    # Following snippet can be found in the printed TTIR.
    # %49 = "tlx.require_layout"(%10) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>) -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    # %50 = "tlx.require_layout"(%13) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>) -> !ttg.memdesc<64x64xf32, #shared2, #smem, mutable>
    # %51 = "ttg.convert_layout"(%3) : (tensor<64x64xf32>) -> tensor<64x64xf32, #mma>
    # "ttng.fence_async_shared"() <{bCluster = false}> : () -> ()
    # %52 = "ttng.warp_group_dot"(%49, %50, %51) <{inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32}> : (!ttg.memdesc<64x64xf32, #shared1, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared2, #smem, mutable>, tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #mma>


@triton.jit
def tlx_square_non_ws(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Test pairs of arrive/wait using different phases
    with a few random misc operations interleaved between them.

    To learn more about mbarrier phase, refer to:
    https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier

    Following patterns will cause mbarrier deadlock.
    TODO. add unit tests demonstrating mbarrier deadlock

    Case 1:
    arrive => wait(phase=1)

    Case 2:
    arrive => arrive => wait(phase=0)

    Case 3:
    wait(phase=0) => arrive
    """

    # prologue
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # mbarrier ops

    bars = tlx.alloc_barriers(num_barriers=1)  # create
    bar = tlx.local_view(bars, 0)

    x = tl.load(x_ptr + offsets, mask=mask)  # Do something

    p = 0
    tlx.barrier_arrive(bar=bar)  # Release
    tlx.barrier_wait(bar=bar, phase=p)  # Wait (proceed immediately)

    z = x * x  # Do something

    p = p ^ 1
    tlx.barrier_arrive(bar=bar)  # Release
    tlx.barrier_wait(bar=bar, phase=p)  # Wait (proceed immediately)

    tl.store(z_ptr + offsets, z, mask=mask)  # Do something

    p = p ^ 1
    tlx.barrier_arrive(bar=bar)  # Release
    tlx.barrier_wait(bar=bar, phase=0)  # Wait (proceed immediately)


@triton.jit
def tlx_square_ws(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # prologue
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # mbarrier ops
    bars = tlx.alloc_barriers(num_barriers=2)  # create
    b0 = tlx.local_view(bars, 0)
    b1 = tlx.local_view(bars, 1)

    phase = 0
    with tlx.async_tasks():
        with tlx.async_task("default"):

            tlx.barrier_wait(bar=b1, phase=phase ^ 1)

            # Placeholder block to do something

            tlx.barrier_arrive(bar=b0)  # Release

        with tlx.async_task(num_warps=4):

            tlx.barrier_wait(bar=b0, phase=phase)  # Wait

            # Some arith ops TODO. add WS
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            z = x * x
            tl.store(z_ptr + offsets, z, mask=mask)

            tlx.barrier_arrive(bar=b0)  # Wait


def run_tlx_square(func, BLOCK_SIZE, device):

    # prepare inputs
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    z = torch.empty_like(x)
    z_ref = torch.empty_like(x)

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    kernel = func[grid](x, z, n_elements, BLOCK_SIZE)

    z_ref = x * x

    torch.testing.assert_close(z, z_ref, check_dtype=False)
    return kernel


# Unit test for arrive/wait
@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
# def test_mbarriers(BLOCK_SIZE, device):
def test_wait_arrive_non_ws(BLOCK_SIZE, device):
    kernel = run_tlx_square(tlx_square_non_ws, BLOCK_SIZE, device)

    # ASSERT in ttgir
    ttgir = kernel.asm["ttgir"]
    assert (ttgir.count("ttng.init_barrier") == 1) and (ttgir.count("ttng.wait_barrier") == 3) and (
        ttgir.count("ttng.barrier_expect") == 0) and (ttgir.count("ttng.arrive_barrier") == 3), f"TTGIR {ttgir}"


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
# def test_mbarriers(BLOCK_SIZE, device):
def test_wait_arrive_ws(BLOCK_SIZE, device):
    kernel = run_tlx_square(tlx_square_ws, BLOCK_SIZE, device)

    # ASSERT in ttgir
    ttgir = kernel.asm["ttgir"]
    assert (ttgir.count("ttng.init_barrier")
            == 2) and (ttgir.count("ttng.wait_barrier") == 2) and (ttgir.count("ttng.barrier_expect") == 0) and (
                ttgir.count("ttng.arrive_barrier")
                == 2) and (ttgir.count("default {") == 1) and (ttgir.count("partition0") == 1), f"TTGIR {ttgir}"


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
def test_descriptor_load(device):

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def descriptor_load_kernel(
        input_ptr,
        output_ptr,
        M, N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int16, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n])

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
    x = torch.ones((M, N), dtype=torch.int16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N)
    )

    # TODO: remove exception handling once layout propagation is implemented
    with pytest.raises(RuntimeError) as _:
        kernel = descriptor_load_kernel[grid](
            x, y, M, N,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
        assert kernel.asm["ttgir"].count("ttng.async_tma_copy_global_to_local") == 1
        assert kernel.asm["ttgir"].count("ttng.async_tma_copy_local_to_global") == 1
        torch.testing.assert_close(x, y)
