import pytest
import torch
import re
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_blackwell
import triton.language.extra.tlx as tlx
from typing import Optional


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
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
            with tlx.async_task("default", replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                x1 = x + replica_id
                y1 = y - replica_id
                output = x1 + y1
                tl.store(z_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=1, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                # This no-op is just to test that replica_id
                # is correctly passed to the kernel
                a1 = a + replica_id
                b1 = b - replica_id
                output = a1 + b1
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
    kernel = add2_warp_specialized_kernel[grid](
        x,
        y,
        output1,
        a,
        b,
        output2,
        n_elements,
        BLOCK_SIZE,
        num_warps=4,
    )
    ttgir = kernel.asm["ttgir"]
    pattern_p0 = r"partition0\([^\n]*\)\s+num_warps\(4\)"
    assert re.search(pattern_p0, ttgir, flags=re.DOTALL)
    pattern_p1 = r"partition1\([^\n]*\)\s+num_warps\(1\)"
    assert re.search(pattern_p1, ttgir, flags=re.DOTALL)
    pattern_p2 = r"partition2\([^\n]*\)\s+num_warps\(1\)"
    assert re.search(pattern_p2, ttgir, flags=re.DOTALL)

    # Check that the replica_id is correctly passed to non-default regions
    # TTIR/TTGIR should be something like:
    #  partition0(...) {
    #   %a1 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
    #   ...
    #   %13 = arith.addf %9, %cst
    #   ...}
    #  partition1(...) {
    #   %cst = arith.constant dense<1.000000e+00> : tensor<1024xf32, #blocked>
    #   ...
    #   %13 = arith.addf %9, %cst
    #   %14 = arith.subf %12, %cst
    #   ...}
    pattern_cst = r"= arith.constant dense\<.*\>"
    found = re.findall(pattern_cst, ttgir)
    assert len(found) == 4, "Expected 4 cst by calling `tlx.async_task_replica_id()` in all regions"
    assert found[0] != found[1], "Two matches MUST be different"
    assert "dense<0.0" in found[0] and "dense<1.0" in found[1], "Expected 0.0 and 1.0 as replica_id"

    ref_out1, ref_out2 = dual_add(x, y, a, b)
    torch.testing.assert_close(output1, ref_out1, check_dtype=False)
    torch.testing.assert_close(output2, ref_out2, check_dtype=False)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
@pytest.mark.parametrize("ENABLE_SECOND_TASK", [True, False])
def test_async_tasks_constexpr_guard(BLOCK_SIZE, ENABLE_SECOND_TASK, device):
    """Test that a tl.constexpr if-check can guard an async_task within async_tasks.

    The first async_task (default) is always present. The second async_task
    is conditionally included based on the ENABLE_SECOND_TASK constexpr flag.
    Both configurations should produce the correct result.
    """

    @triton.jit
    def add_kernel_conditional_task(
        x_ptr,
        y_ptr,
        z_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        ENABLE_SECOND_TASK: tl.constexpr,
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
            if ENABLE_SECOND_TASK:
                with tlx.async_task(num_warps=1, registers=100):
                    offsets = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    a = tl.load(a_ptr + offsets, mask=mask)
                    b = tl.load(b_ptr + offsets, mask=mask)
                    output = a + b
                    tl.store(c_ptr + offsets, output, mask=mask)

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)
    output_z = torch.empty_like(x)
    output_c = torch.empty_like(a)
    n_elements = output_z.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = add_kernel_conditional_task[grid](
        x,
        y,
        output_z,
        a,
        b,
        output_c,
        n_elements,
        BLOCK_SIZE,
        ENABLE_SECOND_TASK,
        num_warps=4,
    )

    ttgir = kernel.asm["ttgir"]
    if ENABLE_SECOND_TASK:
        assert re.search(r"ttg.warp_specialize",
                         ttgir), ("Expected warp_specialize in TTGIR when ENABLE_SECOND_TASK=True")
        assert re.search(r"partition0\([^\n]*\)\s+num_warps\(1\)", ttgir,
                         flags=re.DOTALL), ("Expected partition0 with num_warps(1) when ENABLE_SECOND_TASK=True")
    else:
        assert not re.search(r"ttg.warp_specialize",
                             ttgir), ("Did not expect warp_specialize in TTGIR when ENABLE_SECOND_TASK=False")

    torch.testing.assert_close(output_z, x + y, check_dtype=False)
    if ENABLE_SECOND_TASK:
        torch.testing.assert_close(output_c, a + b, check_dtype=False)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
@pytest.mark.parametrize("USE_LARGE_DEFAULT", [True, False])
def test_async_tasks_constexpr_select_default(BLOCK_SIZE, USE_LARGE_DEFAULT, device):
    """Test that a constexpr if/else can select between two different default tasks.

    Both branches of the if/else contain a default async_task, but only one
    survives constexpr resolution. This exercises the num_default == 1 assertion
    which must hold after resolution, not before.
    """

    @triton.jit
    def kernel_select_default(
        x_ptr,
        y_ptr,
        z_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        USE_LARGE_DEFAULT: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            if USE_LARGE_DEFAULT:
                with tlx.async_task("default", warp_group_start_id=0):
                    offsets = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    x = tl.load(x_ptr + offsets, mask=mask)
                    y = tl.load(y_ptr + offsets, mask=mask)
                    tl.store(z_ptr + offsets, x + y, mask=mask)
            else:
                with tlx.async_task("default", warp_group_start_id=1):
                    offsets = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    x = tl.load(x_ptr + offsets, mask=mask)
                    y = tl.load(y_ptr + offsets, mask=mask)
                    tl.store(z_ptr + offsets, x * y, mask=mask)
            with tlx.async_task(num_warps=1, registers=100):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                tl.store(c_ptr + offsets, a + b, mask=mask)

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)
    output_z = torch.empty_like(x)
    output_c = torch.empty_like(a)
    n_elements = output_z.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = kernel_select_default[grid](
        x,
        y,
        output_z,
        a,
        b,
        output_c,
        n_elements,
        BLOCK_SIZE,
        USE_LARGE_DEFAULT,
        num_warps=4,
    )

    ttgir = kernel.asm["ttgir"]
    assert re.search(r"ttg.warp_specialize", ttgir), "Expected warp_specialize in TTGIR"
    # Verify the non-default task always ran (a + b → c)
    torch.testing.assert_close(output_c, a + b, check_dtype=False)
    # Verify which default was selected by the constexpr condition
    if USE_LARGE_DEFAULT:
        torch.testing.assert_close(output_z, x + y, check_dtype=False)
    else:
        torch.testing.assert_close(output_z, x * y, check_dtype=False)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_async_tasks_region_error(device):

    @triton.jit
    def ws_error_kernel():
        with tlx.async_tasks():
            with tlx.async_task("default"):
                _z = 1 + 2
            with tlx.async_task(num_warps=1):
                _x = 1 / 0

    grid = lambda meta: (1, )
    with pytest.raises(triton.CompilationError) as e:
        ws_error_kernel[grid]()
    exc_msg = str(e.value)
    assert "division by zero" in exc_msg, "\n\nExpected 'division by zero' but got: \n\n" + exc_msg + "\n\n"


def test_default_task_rejects_registers():
    """Specifying registers on the default async_task is banned because the
    default always receives leftover registers from the partition budget."""
    with pytest.raises(AssertionError, match="Cannot specify registers"):
        tlx.async_task("default", registers=128)

    with pytest.raises(AssertionError, match="Cannot specify registers"):
        tlx.async_task("default", num_regs=128)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_async_token_error(device):

    @triton.jit
    def asycn_copy_kernel(x_ptr, y_ptr, cond):
        buffers = tlx.local_alloc((128, ), tl.float32, 1)
        offsets = tl.arange(0, 128)
        if cond:
            token = tlx.async_load(x_ptr + offsets, buffers[0])
        else:
            token = tlx.async_load(y_ptr + offsets, buffers[0])
        tlx.async_load_commit_group([token])

    x = torch.tensor([128], dtype=torch.float32, device=device)
    y = torch.tensor([128], dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    kernel = asycn_copy_kernel[grid](x, y, True)
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 2
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_async_tasks_warp_group_start_ids(BLOCK_SIZE, device):
    """Test that warp_group_start_id is correctly passed to warp_specialize op."""

    @triton.jit
    def warp_specialized_kernel_with_start_ids(
        x_ptr,
        y_ptr,
        z_ptr,
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
            with tlx.async_task(num_warps=2, warp_group_start_id=4, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                tl.store(z_ptr + offsets, x, mask=mask)
            with tlx.async_task(num_warps=1, warp_group_start_id=8):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                y = tl.load(y_ptr + offsets, mask=mask)
                tl.store(z_ptr + offsets, y, mask=mask)

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = warp_specialized_kernel_with_start_ids[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE,
        num_warps=4,
    )
    ttgir = kernel.asm["ttgir"]

    # Verify that warpGroupStartIds attribute is present in the IR with the correct values
    pattern_ws = r"ttg.warp_specialize.*warpGroupStartIds = array<i32: 4, 6, 8>"
    assert re.search(pattern_ws, ttgir,
                     flags=re.DOTALL), (f"Expected warpGroupStartIds = array<i32: 4, 6, 8> in ttgir, got:\n{ttgir}")

    # Verify partition structure
    # Task 1 has replicate=2 with num_warps=2, so partition0 and partition1 both have 2 warps
    # Task 2 has replicate=1 with num_warps=1, so partition2 has 1 warp
    pattern_p0 = r"partition0\([^\n]*\)\s+num_warps\(2\)"
    assert re.search(pattern_p0, ttgir, flags=re.DOTALL)
    pattern_p1 = r"partition1\([^\n]*\)\s+num_warps\(2\)"
    assert re.search(pattern_p1, ttgir, flags=re.DOTALL)
    pattern_p2 = r"partition2\([^\n]*\)\s+num_warps\(1\)"
    assert re.search(pattern_p2, ttgir, flags=re.DOTALL)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell for TMEM")
def test_dummy_layout_function_inlining(device):
    """Test that dummy layouts are correctly resolved when helper functions are inlined into async tasks.

    This test verifies that:
    1. Helper functions with TMA+TMEM operations get properly inlined into async task regions
    2. The dummy layout resolution uses the correct num_warps from the async task context
       (not the global num_warps)
    3. TMA load/store and TMEM operations work correctly when in separate helper functions
       with different warp counts than the async task
    """

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def load_helper(desc, smem_buffer, tmem_buffer, offset_m, offset_n, bar, tmem_full_bar):
        """Helper function: TMA load from global to SMEM, then store to TMEM."""
        tlx.async_descriptor_load(desc, smem_buffer, [offset_m, offset_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        # Load from SMEM to registers, then store to TMEM
        reg_data = tlx.local_load(smem_buffer)
        tlx.local_store(tmem_buffer, reg_data)
        # Signal that TMEM is ready
        tlx.barrier_arrive(tmem_full_bar)

    @triton.jit
    def store_helper(desc, smem_buffer, tmem_buffer, offset_m, offset_n, tmem_full_bar):
        """Helper function: Load from TMEM, then TMA store to global."""
        # Wait for TMEM to be ready
        tlx.barrier_wait(tmem_full_bar, phase=0)
        # Load from TMEM to registers, then store to SMEM
        reg_data = tlx.local_load(tmem_buffer)
        tlx.local_store(smem_buffer, reg_data)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc, smem_buffer, [offset_m, offset_n])
        tlx.async_descriptor_store_wait(0)

    @triton.jit
    def kernel(input_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        # SMEM buffer for TMA operations
        smem_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, tl.constexpr(1))
        smem_buffer = tlx.local_view(smem_buffers, 0)

        # TMEM buffer for intermediate storage
        tmem_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, tl.constexpr(1), tlx.storage_kind.tmem)
        tmem_buffer = tlx.local_view(tmem_buffers, 0)

        # Barrier for TMA load completion
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_M * BLOCK_N * 2)

        # Barrier for TMEM write completion (producer-consumer sync between async tasks)
        tmem_full_bars = tlx.alloc_barriers(tl.constexpr(1))
        tmem_full_bar = tlx.local_view(tmem_full_bars, 0)

        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        with tlx.async_tasks():
            with tlx.async_task("default"):
                # Load from TMA + store to TMEM
                load_helper(desc_in, smem_buffer, tmem_buffer, off_m, off_n, bar, tmem_full_bar)
            with tlx.async_task(num_warps=8):
                # Load from TMEM + store to TMA
                store_helper(desc_out, smem_buffer, tmem_buffer, off_m, off_n, tmem_full_bar)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_M, BLOCK_N = 64, 64
    x = torch.randn((M, N), dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    compiled_kernel = kernel[grid](x, y, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)

    ttgir = compiled_kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 1
    assert ttgir.count("ttng.async_tma_copy_local_to_global") == 1
    assert ttgir.count("ttng.tmem_alloc") == 1
    assert ttgir.count("ttng.tmem_store") == 1
    assert ttgir.count("ttng.tmem_load") == 1

    assert torch.equal(x, y), "Data copy through TMA+TMEM should be exact"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_async_tasks_thread_safety(device):
    """Verify that concurrent compilation of warp-specialized kernels is thread-safe.

    The TLX code generator uses thread-local storage for region_replica_id_stack
    and sub_region_has_exception. This test compiles two different kernels using
    async_tasks() + async_task_replica_id() from separate threads simultaneously
    to verify no cross-thread state corruption occurs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    @triton.jit
    def ws_add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            with tlx.async_task("default", replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = x + y + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=1, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = x + y + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def ws_mul_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            with tlx.async_task("default", replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = a * b + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=1, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = a * b + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)

    size = 98432
    BLOCK_SIZE = 1024

    def compile_and_run_add():
        torch.manual_seed(42)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        out = torch.empty_like(x)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        ws_add_kernel[grid](x, y, out, n, BLOCK_SIZE, num_warps=4)
        torch.testing.assert_close(out, x + y, check_dtype=False)
        return True

    def compile_and_run_mul():
        torch.manual_seed(43)
        a = torch.rand(size, device=device)
        b = torch.rand(size, device=device)
        out = torch.empty_like(a)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        ws_mul_kernel[grid](a, b, out, n, BLOCK_SIZE, num_warps=4)
        torch.testing.assert_close(out, a * b, check_dtype=False)
        return True

    # Use 4 workers: 2 run ws_add_kernel, 2 run ws_mul_kernel.
    # This tests both different-kernel and same-kernel concurrent compilation.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(compile_and_run_add),
            executor.submit(compile_and_run_mul),
            executor.submit(compile_and_run_add),
            executor.submit(compile_and_run_mul),
        ]
        for future in as_completed(futures):
            assert future.result() is True


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_async_tasks_thread_exception_isolation(device):
    """Verify that a compilation exception in one thread doesn't affect others."""
    from concurrent.futures import ThreadPoolExecutor

    @triton.jit
    def ws_good_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            with tlx.async_task("default", replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = x + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=1, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = x + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def ws_bad_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            # Missing "default" task — this should fail during compilation
            with tlx.async_task(num_warps=1, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                tl.store(out_ptr + offsets, x, mask=mask)

    size = 98432
    BLOCK_SIZE = 1024

    def compile_and_run_good():
        x = torch.rand(size, device=device)
        out = torch.empty_like(x)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        ws_good_kernel[grid](x, out, n, BLOCK_SIZE, num_warps=4)
        torch.testing.assert_close(out, x, check_dtype=False)
        return True

    def compile_and_run_bad():
        x = torch.rand(size, device=device)
        out = torch.empty_like(x)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        try:
            ws_bad_kernel[grid](x, out, n, BLOCK_SIZE, num_warps=4)
        except Exception:
            pass  # Expected to fail
        return True

    # Run bad kernel first to set exception flag, then verify good kernel
    # still works on a thread that may be reused from the pool.
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit bad first, then good
        bad_future = executor.submit(compile_and_run_bad)
        bad_future.result()  # Wait for bad to finish
        good_future = executor.submit(compile_and_run_good)
        assert good_future.result() is True


@triton.jit
def _store_ws_kernel(
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Warp-specialized store kernel for PlanCTA regression test.

    Tests tl.store in a warp-specialized context where the store partition
    has fewer warps (1) than the default partition, with num_ctas=2 to
    ensure PlanCTA actually runs (it skips when num_ctas=1).

    This exercises PlanCTA's per-op numWarps lookup: the store's layout
    must be planned with 1 warp (the partition's warp count), not the
    function-level total. Without the fix (lookupNumWarps(store) instead
    of lookupNumWarps(funcOp)), PlanCTA would assign warpsPerCTA=[4]
    inside the 1-warp partition, producing an invalid layout.
    """
    pid = tl.program_id(axis=0)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            _ = tl.arange(0, BLOCK_SIZE)

        with tlx.async_task(num_warps=1):
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            data = offsets.to(tl.float32)
            tl.store(output_ptr + offsets, data)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_store_ws(device):
    BLOCK_SIZE = 256
    n_elements = 1024
    n_blocks = n_elements // BLOCK_SIZE

    output = torch.empty(n_elements, device=device, dtype=torch.float32)
    # num_ctas=2 ensures PlanCTA runs (it skips when num_ctas=1).
    _store_ws_kernel[(n_blocks, )](output, BLOCK_SIZE=BLOCK_SIZE, num_ctas=2)

    expected = torch.arange(n_elements, device=device, dtype=torch.float32)
    torch.testing.assert_close(output, expected)
