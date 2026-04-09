import pytest
import torch
import re
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_hip
import triton.language.extra.tlx as tlx


@triton.jit
def tlx_square_non_ws(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXPECTED_ARRIVAL_COUNT: tl.constexpr,
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

    bars = tlx.alloc_barriers(num_barriers=1, arrive_count=EXPECTED_ARRIVAL_COUNT)  # create
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
    EXPECTED_ARRIVAL_COUNT: tl.constexpr,
):
    # prologue
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # mbarrier ops
    bars = tlx.alloc_barriers(num_barriers=2, arrive_count=EXPECTED_ARRIVAL_COUNT)  # create
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


def run_tlx_square(func, BLOCK_SIZE, device, expected_arrival_count=1):
    # prepare inputs
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    z = torch.empty_like(x)
    z_ref = torch.empty_like(x)

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    kernel = func[grid](x, z, n_elements, BLOCK_SIZE, expected_arrival_count)

    z_ref = x * x

    torch.testing.assert_close(z, z_ref, check_dtype=False)
    return kernel


# Unit test for arrive/wait
@pytest.mark.skipif(not (is_hip() or is_hopper_or_newer()), reason="Need Hopper or newer or AMD")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_wait_arrive_non_ws(BLOCK_SIZE, device):
    expected_arrival_count = 4 if is_hip() else 1
    kernel = run_tlx_square(tlx_square_non_ws, BLOCK_SIZE, device, expected_arrival_count=expected_arrival_count)
    # ASSERT in ttgir
    ttgir = kernel.asm["ttgir"]
    if is_hip():
        assert ((ttgir.count("amdgpu.init_barrier") == 1) and (ttgir.count("amdgpu.read_barrier_phase") == 3)
                and (ttgir.count("amdgpu.arrive_barrier") == 3)), f"TTGIR {ttgir}"
    else:
        assert ((ttgir.count("ttng.init_barrier") == 1) and (ttgir.count("ttng.wait_barrier") == 3)
                and (ttgir.count("ttng.barrier_expect") == 0)
                and (ttgir.count("ttng.arrive_barrier") == 3)), f"TTGIR {ttgir}"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_wait_arrive_ws(BLOCK_SIZE, device):
    kernel = run_tlx_square(tlx_square_ws, BLOCK_SIZE, device)

    # ASSERT in ttgir
    ttgir = kernel.asm["ttgir"]
    assert ((ttgir.count("ttng.init_barrier") == 2) and (ttgir.count("ttng.wait_barrier") == 2)
            and (ttgir.count("ttng.barrier_expect") == 0) and (ttgir.count("ttng.arrive_barrier") == 2)
            and (ttgir.count("default {") == 1) and (ttgir.count("partition0") == 1)), f"TTGIR {ttgir}"


@triton.jit
def tlx_square_warp_barrier(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    """
    Warp-specialized kernel demonstrating perThread barrier arrives with SMEM.
    Producer loads global → stores SMEM → arrives (perThread, no bar.sync).
    Consumer waits → loads SMEM → computes z=x*x → stores global → arrives.

    This mirrors the GEMM epilogue pattern where local_load from shared memory
    is followed by barrier_arrive to signal the buffer is consumed.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Warp barriers: each thread arrives independently (no leader sync)
    bars = tlx.alloc_warp_barrier(num_barriers=2, num_warps=NUM_WARPS)
    b0 = tlx.local_view(bars, 0)
    b1 = tlx.local_view(bars, 1)

    # Shared memory buffer for producer-consumer data transfer
    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    smem = tlx.local_view(buf, 0)

    phase = 0
    with tlx.async_tasks():
        with tlx.async_task("default"):
            tlx.barrier_wait(bar=b1, phase=phase ^ 1)
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            # Producer: load from global, store to SMEM
            x = tl.load(x_ptr + offsets, mask=mask)
            tlx.local_store(smem, x)
            # KEY PATTERN: SMEM write → perThread arrive (no bar.sync)
            tlx.barrier_arrive(bar=b0)

        with tlx.async_task(num_warps=4):
            tlx.barrier_wait(bar=b0, phase=phase)

            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            # Consumer: load from SMEM, compute, store to global
            data = tlx.local_load(smem)
            z = data * data
            tl.store(z_ptr + offsets, z, mask=mask)
            # KEY PATTERN: SMEM read → perThread arrive (no bar.sync)
            tlx.barrier_arrive(bar=b0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
@pytest.mark.parametrize("num_warps", [4])
def test_alloc_warp_barrier(BLOCK_SIZE, num_warps, device):
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    z = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = tlx_square_warp_barrier[grid](
        x,
        z,
        n_elements,
        BLOCK_SIZE,
        num_warps,
        num_warps=num_warps,
    )

    z_ref = x * x
    torch.testing.assert_close(z, z_ref, check_dtype=False)

    # Verify TTGIR: warp-specialized with perThread arrives
    ttgir = kernel.asm["ttgir"]
    assert "perThread" in ttgir, f"Expected perThread attrs in TTGIR:\n{ttgir}"
    assert "ttng.arrive_barrier" in ttgir, f"Expected arrive_barrier in TTGIR:\n{ttgir}"

    # Verify LLIR: perThread arrives use per-thread lowering (no leader predicate)
    llir = kernel.asm["llir"]
    # Per-thread arrive emits unpredicated: mbarrier.arrive.shared::cta.b64 _, [$0]
    assert "mbarrier.arrive.shared::cta.b64 _, [$0]" in llir, (
        f"Expected unpredicated per-thread mbarrier.arrive in LLIR:\n{llir}")
    # Leader pattern would emit predicated: @$0 mbarrier.arrive
    assert "@$0 mbarrier.arrive" not in llir, f"Unexpected leader-predicated mbarrier.arrive in LLIR:\n{llir}"
    # No bar.sync immediately before mbarrier.arrive (membar pass should skip
    # perThread arrives for both full-range and per-buffer SMEM hazards).
    # Other bar.sync may exist (e.g. before wait_barrier) — that's fine.

    assert not re.search(r"barrier\.cta\.sync.*\n.*mbarrier\.arrive",
                         llir), (f"Unexpected bar.sync before mbarrier.arrive in LLIR:\n{llir}")


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_barrier_live_range(device):

    @triton.jit
    def bar_live_kernel():
        # an intentional early return here to check that we're considering dominance when inserting inval bar ops
        pid = tl.program_id(axis=0)
        if pid == 258:
            return

        # use bars1 after bars2/3 init
        bars1 = tlx.alloc_barriers(num_barriers=tl.constexpr(1), arrive_count=1)

        bars2 = tlx.alloc_barriers(num_barriers=tl.constexpr(1), arrive_count=2)
        tlx.barrier_arrive(bars2[0])
        # No-op wait to avoid pruning.
        tlx.barrier_wait(bar=bars2[0], phase=1)

        bars3 = tlx.alloc_barriers(num_barriers=tl.constexpr(1), arrive_count=3)
        tlx.barrier_arrive(bars3[0])
        # No-op wait to avoid pruning.
        tlx.barrier_wait(bar=bars3[0], phase=1)

        # bars1 and bars2 should both be live here
        tlx.barrier_arrive(bars1[0])
        # No-op wait to avoid pruning.
        tlx.barrier_wait(bar=bars1[0], phase=0)

    torch.manual_seed(0)
    kernel = bar_live_kernel[(2, 1)]()
    ptx = kernel.asm["ptx"]

    # e.g. extract %1 and 1 from "mbarrier.init.shared::cta.b64 [%r1], 1;"
    pattern = r"mbarrier\.init\..*\.b64 \[(%r\d+)\], (\d+);"
    matches = re.findall(pattern, ptx)

    arrive_count_to_reg = {int(arrive_count): reg for reg, arrive_count in matches}
    assert len(arrive_count_to_reg) == 3, f"Expected 3 mbarrier init, got ptx: \n{ptx}"
    # Make sure they all have different registers (different SMEM addresses)
    assert arrive_count_to_reg[1] != arrive_count_to_reg[2], f"invalid reuse of SMEM, full ptx: \n{ptx}"
    assert arrive_count_to_reg[2] != arrive_count_to_reg[3], f"invalid reuse of SMEM, full ptx: \n{ptx}"
    assert arrive_count_to_reg[1] != arrive_count_to_reg[3], f"invalid reuse of SMEM, full ptx: \n{ptx}"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_named_wait_arrive(BLOCK_SIZE, device):

    @triton.jit
    def add2_warp_specialized_pingpong_kernel(
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
                tlx.named_barrier_wait(9, 256)
                tlx.named_barrier_arrive(10, 256)
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                output = x + y
                tl.store(z_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=4, registers=100):
                tlx.named_barrier_arrive(9, 256)
                tlx.named_barrier_wait(10, 256)
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
    kernel = add2_warp_specialized_pingpong_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE)
    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.wait_barrier_named %c9_i32, %c256_i32") == 1
    assert ttgir.count("ttng.arrive_barrier_named %c10_i32, %c256_i32") == 1
    assert ttgir.count("ttng.arrive_barrier_named %c9_i32_1, %c256_i32") == 1
    assert ttgir.count("ttng.wait_barrier_named %c10_i32_0, %c256_i32") == 1

    ref_out1, ref_out2 = dual_add(x, y, a, b)
    torch.testing.assert_close(output1, ref_out1, check_dtype=False)
    torch.testing.assert_close(output2, ref_out2, check_dtype=False)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_barrier_wait_no_remote_view(device):
    """Test that barrier_wait does not allow remote_view of mbarrier."""

    @triton.jit
    def barrier_wait_remote_view_kernel():
        bars = tlx.alloc_barriers(num_barriers=tl.constexpr(1), arrive_count=1)
        bar = tlx.local_view(bars, 0)
        # Get remote view of the barrier
        remote_bar = tlx.remote_view(bar, 0)
        # This should raise an assertion error because barrier_wait does not support remote_view
        tlx.barrier_wait(remote_bar, phase=0)

    grid = lambda meta: (1, )
    with pytest.raises(triton.CompilationError) as e:
        barrier_wait_remote_view_kernel[grid](ctas_per_cga=(2, 1, 1))
    exc_msg = str(e.value)
    assert "barrier_wait" in exc_msg, f"Expected error about barrier_wait, but got: {exc_msg}"


# =============================================================================
# Test: named_barrier_wait in 1-warp async_task (DEADLOCKS)
# =============================================================================


def _run_kernel_diverge_both_1warp(result_queue):
    """Subprocess target: runs the deadlocking kernel and reports back."""
    try:
        import torch
        import triton
        import triton.language as tl
        import triton.language.extra.tlx as tlx

        @triton.jit
        def _kernel_diverge_both_1warp(output_ptr):
            """1-warp task, divergence on both sides -> DEADLOCKS."""
            with tlx.async_tasks():
                with tlx.async_task(num_warps=1):
                    if tlx.thread_id(axis=0) % 32 == 0:
                        tl.store(output_ptr + 1, 99)  # divergence BEFORE
                    tlx.named_barrier_wait(14, 32)
                    if tlx.thread_id(axis=0) % 32 == 0:
                        tl.store(output_ptr + 0, 5)  # divergence AFTER
                with tlx.async_task("default"):
                    pass

        output = torch.zeros(2, dtype=torch.int32, device="cuda")
        _kernel_diverge_both_1warp[(1, )](output, num_warps=4)
        torch.cuda.synchronize()
        result_queue.put(("PASS", output.cpu().tolist()))
    except Exception as e:
        result_queue.put(("ERROR", str(e)))


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_named_barrier_wait_1warp_async_deadlock(device):
    """Test that named_barrier_wait(14, 32) in 1-warp async_task deadlocks.

    This test demonstrates a known deadlock scenario where a named barrier
    with divergent code on both sides deadlocks inside an async_task.
    The kernel is run in a subprocess with a timeout so a deadlock doesn't
    hang the entire test suite.
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_run_kernel_diverge_both_1warp, args=(result_queue, ))
    proc.start()
    proc.join(timeout=15)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=10)
        pytest.xfail("Kernel deadlocked as expected (known issue: named_barrier_wait "
                     "with divergent code on both sides inside async_task)")
    elif result_queue.empty():
        pytest.fail("Subprocess exited without producing a result")
    else:
        status, detail = result_queue.get()
        if status == "PASS":
            # If this passes, the bug has been fixed!
            pass
        else:
            pytest.fail(f"Kernel raised an error: {detail}")


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_named_barrier_wait_1warp_async_deadlock_single_proc(device):
    """Same as test_named_barrier_wait_1warp_async_deadlock but runs in the
    current process for easier IR debugging. WARNING: will hang if the bug
    is present — use with a timeout (e.g. ``pytest --timeout=15``)."""

    @triton.jit
    def _kernel_diverge_both_1warp_sp(output_ptr):
        if tlx.thread_id(axis=0) % 32 == 0:
            tl.store(output_ptr + 1, 99)  # divergence BEFORE
        tlx.named_barrier_wait(14, 32)
        if tlx.thread_id(axis=0) % 32 == 0:
            tl.store(output_ptr + 0, 5)  # divergence AFTER

    output = torch.zeros(2, dtype=torch.int32, device=device)
    _kernel_diverge_both_1warp_sp[(1, )](output, num_warps=4)
    torch.cuda.synchronize()
    result = output.cpu().tolist()
    assert result[0] == 5, f"Expected output[0]=5, got {result[0]}"
    assert result[1] == 99, f"Expected output[1]=99, got {result[1]}"
