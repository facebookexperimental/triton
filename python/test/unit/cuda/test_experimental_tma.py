import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import (
    numpy_random,
    requires_tma,
    supports_tma,
    tma_skip_msg,
    to_triton,
    unwrap_tensor,
)
from triton.tools.experimental_descriptor import (
    create_1d_tma_descriptor,
    create_2d_tma_descriptor,
    create_1d_tma_descriptor_type,
    create_2d_tma_descriptor_type,
)


def create_tma_desc_gmem_ptr(ptr, dims, block_dims, element_size):
    cpu_desc = torch.empty(128, device="cpu")
    if len(dims) == 1:
        triton.runtime.driver.active.utils.fill_1d_tma_descriptor(
            ptr, dims[0], block_dims[0], element_size, cpu_desc.data_ptr()
        )
    else:
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
            ptr,
            dims[0],
            dims[1],
            block_dims[0],
            block_dims[1],
            element_size,
            cpu_desc.data_ptr(),
        )
    return cpu_desc.cuda()


tma_dtypes = [
    "uint8",
    "uint16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "float16",
    "float32",
    "float64",
    "bfloat16",
]


@pytest.mark.parametrize("byval_tma", [True, False])
def test_experimetal_descriptor_load(byval_tma):
    if not supports_tma(byval_tma):
        pytest.skip(tma_skip_msg(byval_tma))

    device = "cuda"
    SIZE = 128

    @triton.jit
    def kernel(Z, desc, SIZE: tl.constexpr, BYVAL_TMA: tl.constexpr):
        if not BYVAL_TMA:
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc)
        off_desc = 0
        off = tl.arange(0, SIZE)
        x = tl._experimental_descriptor_load(
            desc, [off_desc], [SIZE], Z.dtype.element_ty
        )
        tl.store(Z + off, x)

    x = torch.randn(SIZE, dtype=torch.float32, device=device)
    if byval_tma:
        desc = create_1d_tma_descriptor(x.data_ptr(), SIZE, SIZE, x.element_size())
    else:
        desc = create_tma_desc_gmem_ptr(x.data_ptr(), [SIZE], [SIZE], x.element_size())
    z_tri = torch.empty_like(x)
    compiled_kernel = kernel[(1,)](
        z_tri, desc, SIZE=SIZE, BYVAL_TMA=byval_tma, num_warps=4
    )
    assert torch.equal(x, z_tri)
    if byval_tma:
        assert ".param .align 64 .b8" in compiled_kernel.asm["ptx"]


@triton.jit
def matmul_kernel_tma(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,  #
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BYVAL_TMA: tl.constexpr,
    dtype: tl.constexpr,
):
    if not BYVAL_TMA:
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], dtype
        )
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(dtype)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


@pytest.mark.parametrize("num_stages", [1, 4])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K",
    [(32, 32, 32), (128, 64, 64), (128, 128, 64), (128, 256, 64)],
)
@pytest.mark.parametrize("byval_tma", [True, False])
def test_experimental_tma_matmul(num_stages, BLOCK_M, BLOCK_N, BLOCK_K, byval_tma):
    if not supports_tma(byval_tma):
        pytest.skip(tma_skip_msg(byval_tma))

    device = "cuda"
    M, N, K = 8192, 8192, 1024
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    if byval_tma:
        desc_a = create_2d_tma_descriptor(
            A.data_ptr(), M, K, BLOCK_M, BLOCK_K, A.element_size()
        )
        desc_b = create_2d_tma_descriptor(
            B.data_ptr(), K, N, BLOCK_K, BLOCK_N, B.element_size()
        )
        desc_c = create_2d_tma_descriptor(
            C.data_ptr(), M, N, BLOCK_M, BLOCK_N, C.element_size()
        )
    else:
        desc_a = create_tma_desc_gmem_ptr(
            A.data_ptr(), [M, K], [BLOCK_M, BLOCK_K], A.element_size()
        )
        desc_b = create_tma_desc_gmem_ptr(
            B.data_ptr(), [K, N], [BLOCK_K, BLOCK_N], B.element_size()
        )
        desc_c = create_tma_desc_gmem_ptr(
            C.data_ptr(), [M, N], [BLOCK_M, BLOCK_N], C.element_size()
        )
    kernel = matmul_kernel_tma[
        (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
    ](
        desc_a,
        desc_b,
        desc_c,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        BYVAL_TMA=byval_tma,
        num_warps=8,
        num_stages=num_stages,
        dtype=tl.float16,
    )
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
    if BLOCK_M >= 64 and BLOCK_N >= 64 and torch.cuda.get_device_capability()[0] == 9:
        # TODO: The use of stmatrix for Blackwell is currently not supported.
        # Only a subset of TMEM and stmatrix layout pairs are compatible, for example 16x256bx2 and m8n8x4.
        assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in kernel.asm["ptx"]
    if byval_tma:
        assert ".param .align 64 .b8" in kernel.asm["ptx"]


@triton.jit
def device_tensormap_kernel2d(
    in_ptr,
    out_ptr,
    in_desc,
    out_desc,
    ready_flag,
    M,
    N,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    if pid_m == 0 and pid_n == 0:
        # Write out descriptor
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=in_desc,
            global_address=in_ptr,
            load_size=[M_BLOCK, N_BLOCK],
            global_size=[M, N],
            element_ty=in_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=out_desc,
            global_address=out_ptr,
            load_size=[M_BLOCK, N_BLOCK],
            global_size=[M, N],
            element_ty=out_ptr.dtype.element_ty,
        )
        tl.atomic_xchg(ready_flag, 1, sem="release")
    else:
        # Spin until descriptor is ready
        flag = tl.full([], 0, tl.int32)
        while flag == 0:
            flag = tl.atomic_add(ready_flag, 0, sem="acquire")
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(in_desc)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(out_desc)

    moffset = pid_m * M_BLOCK
    noffset = pid_n * N_BLOCK

    x = tl._experimental_descriptor_load(
        in_desc, [moffset, noffset], [M_BLOCK, N_BLOCK], in_ptr.dtype.element_ty
    )
    tl._experimental_descriptor_store(out_desc, x, [moffset, noffset])


@requires_tma
@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_device_tensormap2d(dtype_str):
    M_BLOCK, N_BLOCK = 32, 64
    M_GRID, N_GRID = 2, 4

    shape = (M_BLOCK * M_GRID, M_BLOCK * N_GRID)
    device = "cuda"
    inp = to_triton(
        numpy_random(shape, dtype_str=dtype_str), device=device, dst_type=dtype_str
    )
    inp_copy = inp.clone()
    out = to_triton(
        numpy_random(shape, dtype_str=dtype_str), device=device, dst_type=dtype_str
    )

    in_desc = torch.randint(0, 256, size=(128,), dtype=torch.uint8, device="cuda")
    out_desc = torch.randint(0, 256, size=(128,), dtype=torch.uint8, device="cuda")
    ready_flag = torch.zeros((), dtype=torch.int32, device="cuda")

    device_tensormap_kernel2d[M_GRID, N_GRID](
        inp,
        out,
        in_desc,
        out_desc,
        ready_flag,
        *shape,
        M_BLOCK=M_BLOCK,
        N_BLOCK=N_BLOCK
    )

    # Check results are correct
    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(out))
    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(inp_copy))


@triton.jit
def device_tensormap_kernel1d(
    in_ptr, out_ptr, in_desc, out_desc, ready_flag, numel, BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    if pid == 0:
        # Write out descriptor
        tl.extra.cuda.experimental_device_tensormap_create1d(
            desc_ptr=in_desc,
            global_address=in_ptr,
            load_size=BLOCK,
            global_size=numel,
            element_ty=in_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_device_tensormap_create1d(
            desc_ptr=out_desc,
            global_address=out_ptr,
            load_size=BLOCK,
            global_size=numel,
            element_ty=out_ptr.dtype.element_ty,
        )
        tl.atomic_xchg(ready_flag, 1, sem="release")
    else:
        # Spin until descriptor is ready
        flag = tl.full([], 0, tl.int32)
        while flag == 0:
            flag = tl.atomic_add(ready_flag, 0, sem="acquire")
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(in_desc)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(out_desc)

    offset = pid * BLOCK

    x = tl._experimental_descriptor_load(
        in_desc, [offset], [BLOCK], in_ptr.dtype.element_ty
    )
    tl._experimental_descriptor_store(out_desc, x, [offset])


@requires_tma
@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_device_tensormap1d(dtype_str):
    BLOCK = 256
    GRID = 8

    shape = (BLOCK * GRID,)
    device = "cuda"
    inp = to_triton(
        numpy_random(shape, dtype_str=dtype_str), device=device, dst_type=dtype_str
    )
    inp_copy = inp.clone()
    out = to_triton(
        numpy_random(shape, dtype_str=dtype_str), device=device, dst_type=dtype_str
    )

    in_desc = torch.randint(0, 256, size=(128,), dtype=torch.uint8, device="cuda")
    out_desc = torch.randint(0, 256, size=(128,), dtype=torch.uint8, device="cuda")
    ready_flag = torch.zeros((), dtype=torch.int32, device="cuda")

    device_tensormap_kernel1d[1,](
        inp, out, in_desc, out_desc, ready_flag, *shape, BLOCK=BLOCK
    )

    # Check results are correct
    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(out))
    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(inp_copy))

####################################################################################################
# TMA Reduce
####################################################################################################


def map_dtype_to_triton(dtype: torch.dtype) -> int:
    """
    Maps torch dtype to triton dtype.
    Args:
        dtype (torch.dtype): input dtype.
    Returns:
        tl.dtype: triton dtype.
    """
    if dtype == torch.float16:
        return 0
    elif dtype == torch.bfloat16:
        return 1
    elif dtype == torch.float32:
        return 2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


tma_reduce_dtypes = [torch.float16, torch.bfloat16, torch.float32]

####################################################################################################
# Vector Reduce-add with on-host TMA
####################################################################################################


@triton.jit
def vector_add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    x_desc,
    y_ptr,  # *Pointer* to second input vector.
    y_desc,
    output_desc,
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    # Load x through TMA.
    x = tl._experimental_descriptor_load(
        x_desc, [block_start], [BLOCK_SIZE], x_ptr.dtype.element_ty
    )
    # Store x to through TMA.
    tl._experimental_descriptor_store(output_desc, x, [block_start])
    # Load y through TMA.
    y = tl._experimental_descriptor_load(
        y_desc, [block_start], [BLOCK_SIZE], y_ptr.dtype.element_ty
    )
    tl.debug_barrier()
    # Store y to through TMA reduce add.
    tl._experimental_descriptor_store(output_desc, y, [block_start], store_reduce="add")


@requires_tma
@pytest.mark.parametrize("dtype", tma_reduce_dtypes)
def test_vector_add_host_tma_reduce(dtype):
    if torch.version.cuda < "12.4":
        pytest.skip("Test requires CUDA 12.4+")
        return
    BLOCK_SIZE = 256
    size = 1024
    x = torch.rand(size, dtype=dtype, device="cuda")
    y = torch.rand(size, dtype=dtype, device="cuda")
    output_triton = torch.empty_like(x)
    x_desc = create_1d_tma_descriptor_type(
        x.data_ptr(), size, BLOCK_SIZE, map_dtype_to_triton(x.dtype)
    )
    y_desc = create_1d_tma_descriptor_type(
        y.data_ptr(), size, BLOCK_SIZE, map_dtype_to_triton(y.dtype)
    )
    output_desc = create_1d_tma_descriptor_type(
        output_triton.data_ptr(),
        size,
        BLOCK_SIZE,
        map_dtype_to_triton(output_triton.dtype),
    )
    n_elements = output_triton.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, x_desc, y, y_desc, output_desc, BLOCK_SIZE=BLOCK_SIZE)
    # Check results are correct
    output_torch = x + y
    torch.testing.assert_close(output_triton, output_torch)


####################################################################################################
# Tile Reduce-add with on-host TMA
####################################################################################################


@triton.jit
def tile_add_kernel(
    x_ptr,
    x_desc,
    y_ptr,
    y_desc,
    output_desc,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE
    BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE
    GROUP_SIZE_M: tl.constexpr = GROUP_SIZE
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N

    # Load x through TMA.
    x = tl._experimental_descriptor_load(
        x_desc, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], x_ptr.dtype.element_ty
    )
    # Store x to through TMA.
    tl._experimental_descriptor_store(output_desc, x, [offs_m, offs_n])
    # Load y through TMA.
    y = tl._experimental_descriptor_load(
        y_desc, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], y_ptr.dtype.element_ty
    )
    tl.debug_barrier()
    # Store y to through TMA reduce add.
    tl._experimental_descriptor_store(
        output_desc, y, [offs_m, offs_n], store_reduce="add"
    )


@requires_tma
@pytest.mark.parametrize("dtype", tma_reduce_dtypes)
def test_tile_add_host_tma_reduce(dtype):
    BLOCK_SIZE = 128
    size = 512
    x = torch.rand((size, size), dtype=dtype, device="cuda")
    y = torch.rand((size, size), dtype=dtype, device="cuda")
    M, N = x.shape
    x_desc = create_2d_tma_descriptor_type(
        x.data_ptr(), M, N, BLOCK_SIZE, BLOCK_SIZE, map_dtype_to_triton(x.dtype)
    )
    y_desc = create_2d_tma_descriptor_type(
        y.data_ptr(), M, N, BLOCK_SIZE, BLOCK_SIZE, map_dtype_to_triton(y.dtype)
    )
    output_triton = torch.empty((M, N), device=x.device, dtype=dtype)
    output_desc = triton.tools.experimental_descriptor.create_2d_tma_descriptor_type(
        output_triton.data_ptr(),
        M,
        N,
        BLOCK_SIZE,
        BLOCK_SIZE,
        map_dtype_to_triton(x.dtype),
    )

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]) * triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    tile_add_kernel[grid](
        x,
        x_desc,
        y,
        y_desc,
        output_desc,
        M,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=8,
    )
    # Check results are correct
    output_torch = x + y
    torch.testing.assert_close(output_triton, output_torch)


####################################################################################################
# Tile Reduce-add with on-device TMA
####################################################################################################


@triton.jit
def add_kernel_device_tma_reduce(
    workspace_ptr,
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE
    BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE
    GROUP_SIZE_M: tl.constexpr = GROUP_SIZE
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N

    TMA_SIZE: tl.constexpr = 128
    workspace_base = workspace_ptr + pid * 3 * TMA_SIZE
    x_desc_ptr = workspace_base
    y_desc_ptr = workspace_base + TMA_SIZE
    output_desc_ptr = workspace_base + 2 * TMA_SIZE

    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=x_desc_ptr,
        global_address=x_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=x_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=y_desc_ptr,
        global_address=y_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=y_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=output_desc_ptr,
        global_address=output_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=output_ptr.dtype.element_ty,
    )

    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(x_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(y_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(output_desc_ptr)

    # Load x through TMA.
    x = tl._experimental_descriptor_load(
        x_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], x_ptr.dtype.element_ty
    )
    # Store x to through TMA.
    tl.debug_barrier()
    tl._experimental_descriptor_store(output_desc_ptr, x, [offs_m, offs_n])
    # Load y through TMA.
    y = tl._experimental_descriptor_load(
        y_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], y_ptr.dtype.element_ty
    )
    # Store y to through TMA reduce add.
    tl.debug_barrier()
    tl._experimental_descriptor_store(
        output_desc_ptr, y, [offs_m, offs_n], store_reduce="add"
    )


@requires_tma
@pytest.mark.parametrize("dtype", tma_reduce_dtypes)
def test_tile_add_device_tma_reduce(dtype):
    BLOCK_SIZE = 128
    size = 512
    x = torch.rand((size, size), dtype=dtype, device="cuda")
    y = torch.rand((size, size), dtype=dtype, device="cuda")
    M, N = x.shape
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    TMA_SIZE = 128
    workspace = torch.empty(NUM_SMS * 3 * TMA_SIZE, dtype=torch.uint8, device="cuda")
    output_triton = torch.zeros((M, N), device=x.device, dtype=dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]) * triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    add_kernel_device_tma_reduce[grid](
        workspace,
        x,
        y,
        output_triton,
        M,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=8,
    )
    # Check results are correct
    output_torch = x + y
    torch.testing.assert_close(output_triton, output_torch)
