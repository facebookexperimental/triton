"""TLX warp specialization tests -- Blackwell-only."""
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx
from typing import Optional


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


@pytest.mark.skipif(not is_blackwell(), reason="single_warp_specialize lowering targets Blackwell")
def test_single_warp_specialize(device):
    # A single warp_specialize op (captures the pointers/size): the `default`
    # warp group computes x + y, one worker warp group computes a + b. Marking
    # the async_tasks block `exclusive=True` makes the Fixup pass set the
    # ttg.single-warp-specialize module attribute (after validating there is
    # exactly one warp_specialize op), which makes the warp-specialize -> LLVM
    # lowering dispatch worker warps from `wid` (no SMEM state id / switch table)
    # and reclaim the state SMEM.
    @triton.jit
    def add2_ws_kernel(x_ptr, y_ptr, z_ptr, a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                       EXCLUSIVE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks(exclusive=EXCLUSIVE):
            with tlx.async_task("default"):
                offs = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offs < n_elements
                tl.store(z_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)
            with tlx.async_task(num_warps=4, num_regs=232):
                offs = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offs < n_elements
                tl.store(c_ptr + offs, tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask), mask=mask)

    def run(exclusive):
        torch.manual_seed(0)
        n = 98432
        x, y, a, b = (torch.rand(n, device=device) for _ in range(4))
        z, c = torch.empty_like(x), torch.empty_like(a)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        compiled = add2_ws_kernel[grid](x, y, z, a, b, c, n, BLOCK_SIZE=1024, EXCLUSIVE=exclusive)
        # Same result regardless of the dispatch strategy.
        torch.testing.assert_close(z, x + y, check_dtype=False)
        torch.testing.assert_close(c, a + b, check_dtype=False)
        return compiled

    single = run(exclusive=True)
    multi = run(exclusive=False)

    # The exclusive marker is propagated to the module attribute, and there is
    # exactly one warp_specialize op (a precondition of the fast path).
    assert '"ttg.single-warp-specialize" = true' in single.asm["ttgir"]
    assert single.asm["ttgir"].count("ttg.warp_specialize(") == 1
    assert "single-warp-specialize" not in multi.asm["ttgir"]

    # The fast path dispatches worker warps from `wid`: no SMEM state-id switch
    # table. The general path still lowers to a switch.
    assert "switch" not in single.asm["llir"]
    assert "switch" in multi.asm["llir"]

    # The worker task requested 232 registers, so the lowering emits register
    # (setmaxnreg) and barrier instructions in the PTX. The fast path drops the
    # end-of-region register restoration and the state-id/switch barriers, so it
    # emits strictly fewer of both than the general (multi-WS) lowering.
    single_ptx, multi_ptx = single.asm["ptx"], multi.asm["ptx"]
    assert single_ptx.count("setmaxnreg") < multi_ptx.count("setmaxnreg")
    assert single_ptx.count("barrier.sync") < multi_ptx.count("barrier.sync")
