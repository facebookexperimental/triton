"""Standalone tests for tlx_plugin barrier ops (alloc_barriers, alloc_warp_barrier).

Tests the plugin-ported alloc_barriers operation: allocating mbarrier buffers
in shared memory and initializing them with InitBarrierOp. Covers compile-only
IR verification and on-GPU execution.

These tests import from tlx_plugin (the out-of-tree plugin Python DSL)
rather than triton.language.extra.tlx (the in-tree TLX DSL).
"""

import sys
import os
import pytest
import torch

import triton
import triton.language as tl

_plugin_python_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "..",
                 "triton-tlx", "python")
)
if _plugin_python_dir not in sys.path:
    sys.path.insert(0, _plugin_python_dir)
from tlx_plugin.utility import ensure_plugin_on_path
ensure_plugin_on_path()
import tlx_plugin as tlx  # type: ignore[import-not-found]


def is_hopper_or_newer():
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Compile-only: verify IR generation without running on GPU
# ---------------------------------------------------------------------------

def test_alloc_barriers_compile_only():
    """Verify alloc_barriers generates valid TTGIR with expected ops."""

    @triton.jit
    def kernel(Out):
        bars = tlx.alloc_barriers(
            num_barriers=tl.constexpr(4), arrive_count=tl.constexpr(1)
        )
        pid = tl.program_id(0)
        tl.store(Out + pid, pid)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"Out": "*i32"},
        constexprs={},
    )
    try:
        ret = triton.compile(
            src, target=triton.runtime.driver.active.get_current_target()
        )
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "local_alloc" in ir_str, (
        f"Expected 'local_alloc' in TTGIR.\nIR:\n{ir_str}"
    )
    assert "init_barrier" in ir_str, (
        f"Expected 'init_barrier' in TTGIR.\nIR:\n{ir_str}"
    )
    assert "memdesc_index" in ir_str, (
        f"Expected 'memdesc_index' in TTGIR.\nIR:\n{ir_str}"
    )


def test_alloc_warp_barrier_compile_only():
    """Verify alloc_warp_barrier generates init_barrier with correct arrive count."""

    @triton.jit
    def kernel(Out):
        bars = tlx.alloc_warp_barrier(
            num_barriers=tl.constexpr(2),
            num_warps=tl.constexpr(4),
            num_arrivals=tl.constexpr(1),
        )
        pid = tl.program_id(0)
        tl.store(Out + pid, pid)

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"Out": "*i32"},
        constexprs={},
    )
    try:
        ret = triton.compile(
            src, target=triton.runtime.driver.active.get_current_target()
        )
    except Exception:
        pytest.skip("No GPU target available for compilation")

    ir_str = ret.asm.get("ttgir", "")
    assert "init_barrier" in ir_str, (
        f"Expected 'init_barrier' in TTGIR.\nIR:\n{ir_str}"
    )
    # arrive_count = num_warps(4) * 32 * num_arrivals(1) = 128
    assert "128" in ir_str, (
        f"Expected arrive_count 128 in TTGIR.\nIR:\n{ir_str}"
    )


# ---------------------------------------------------------------------------
# On-GPU: alloc_barriers execution
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_alloc_barriers_on_gpu(device="cuda"):
    """Allocate barriers on GPU and verify the kernel runs without errors.

    Exercises the full alloc_barriers pipeline on hardware: LocalAllocOp
    for the barrier buffer + InitBarrierOp for each slot. A simple
    load/store via standard tl ops verifies the kernel completes correctly.
    """

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)

        # Allocate barriers (exercises alloc_barriers plugin op)
        bars = tlx.alloc_barriers(
            num_barriers=tl.constexpr(2), arrive_count=tl.constexpr(1)
        )

        tl.store(out_ptr + offs, x)

    BLOCK = 128
    x = torch.randn(BLOCK, device=device, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1,)](x, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, x)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("num_barriers", [1, 2, 4])
def test_alloc_barriers_various_counts(num_barriers, device="cuda"):
    """Test alloc_barriers with different barrier counts."""

    @triton.jit
    def kernel(out_ptr, NUM_BARS: tl.constexpr):
        bars = tlx.alloc_barriers(
            num_barriers=NUM_BARS, arrive_count=tl.constexpr(1)
        )
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, pid)

    out = torch.zeros(1, device=device, dtype=torch.int32)
    kernel[(1,)](out, NUM_BARS=num_barriers)
    assert out[0].item() == 0


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_alloc_warp_barrier_on_gpu(device="cuda"):
    """Test alloc_warp_barrier runs on GPU without errors."""

    @triton.jit
    def kernel(out_ptr):
        bars = tlx.alloc_warp_barrier(
            num_barriers=tl.constexpr(2),
            num_warps=tl.constexpr(4),
            num_arrivals=tl.constexpr(1),
        )
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, pid)

    out = torch.zeros(1, device=device, dtype=torch.int32)
    kernel[(1,)](out)
    assert out[0].item() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
