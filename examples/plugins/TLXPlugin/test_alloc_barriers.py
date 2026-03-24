"""Test for the tlx_alloc_barriers plugin custom op.

Verifies that the alloc_barriers op generates correct TTIR containing:
  - ttg.local_alloc for the barrier buffer
  - ttg.memdesc_index for each barrier slot
  - ttng.init_barrier for each slot

Usage:
    # Set TRITON_PASS_PLUGIN_PATH to point to the built plugin .so, then:
    TRITON_PASS_PLUGIN_PATH=<path-to>/libTLXMemOpsPlugin.so python test_alloc_barriers.py

    # Or if built via pip install -e, the .so is typically at:
    #   triton/_C/plugins/libTLXMemOpsPlugin.so
"""

import os
import sys

# Ensure the plugin Python package is importable
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "python")
)

import triton
import triton.language as tl
import tlx_plugin as tlx


@triton.jit
def alloc_barriers_kernel(Out):
    """Minimal kernel that allocates barriers to exercise the plugin op."""
    bars = tlx.alloc_barriers(num_barriers=tl.constexpr(4), arrive_count=tl.constexpr(1))
    # Store a sentinel value to prevent the kernel from being entirely optimized away
    pid = tl.program_id(0)
    tl.store(Out + pid, pid)


@triton.jit
def alloc_warp_barrier_kernel(Out):
    """Minimal kernel that allocates warp barriers."""
    bars = tlx.alloc_warp_barrier(
        num_barriers=tl.constexpr(2),
        num_warps=tl.constexpr(4),
        num_arrivals=tl.constexpr(1),
    )
    pid = tl.program_id(0)
    tl.store(Out + pid, pid)


def get_ttgir(kernel, *args, grid=(1,), num_warps=4, **kwargs):
    """Compile the kernel and return TTGIR as a string."""
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.runtime import driver

    target = driver.active.get_current_target()
    src = ASTSource(fn=kernel, signature={"Out": "*i32"}, constexprs={})
    compiled = triton_compile(src, target=target, options={"num_warps": num_warps})
    return compiled.asm.get("ttgir", "")


def test_alloc_barriers_ir():
    """Check that alloc_barriers generates the expected MLIR ops."""
    ir_str = get_ttgir(alloc_barriers_kernel)

    # The IR should contain a local_alloc for the barrier buffer (3 x i64)
    assert "local_alloc" in ir_str, (
        f"Expected 'local_alloc' in TTGIR but not found.\nIR:\n{ir_str}"
    )

    # Should have init_barrier ops (one per barrier slot)
    assert "init_barrier" in ir_str, (
        f"Expected 'init_barrier' in TTGIR but not found.\nIR:\n{ir_str}"
    )

    # Should have memdesc_index ops for indexing into barrier slots
    assert "memdesc_index" in ir_str or "MemDescIndex" in ir_str.replace(" ", ""), (
        f"Expected 'memdesc_index' in TTGIR but not found.\nIR:\n{ir_str}"
    )

    print("PASS: test_alloc_barriers_ir")
    print(f"  Generated TTGIR ({len(ir_str)} chars) contains local_alloc, "
          "init_barrier, and memdesc_index ops.")


def test_alloc_warp_barrier_ir():
    """Check that alloc_warp_barrier generates init_barrier with correct arrive count."""
    ir_str = get_ttgir(alloc_warp_barrier_kernel)

    assert "local_alloc" in ir_str, (
        f"Expected 'local_alloc' in TTGIR but not found.\nIR:\n{ir_str}"
    )
    assert "init_barrier" in ir_str, (
        f"Expected 'init_barrier' in TTGIR but not found.\nIR:\n{ir_str}"
    )

    # arrive_count = num_warps(4) * 32 * num_arrivals(1) = 128
    assert "128" in ir_str, (
        f"Expected arrive_count 128 (4 warps * 32 threads) in TTGIR but not found.\n"
        f"IR:\n{ir_str}"
    )

    print("PASS: test_alloc_warp_barrier_ir")
    print(f"  Generated TTGIR ({len(ir_str)} chars) contains init_barrier with "
          "arrive_count=128.")


if __name__ == "__main__":
    plugin_path = os.environ.get("TRITON_PASS_PLUGIN_PATH", "")
    if not plugin_path:
        print("ERROR: TRITON_PASS_PLUGIN_PATH is not set.", file=sys.stderr)
        print("Set it to the path of the built libTLXMemOpsPlugin.so", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(plugin_path):
        print(f"ERROR: Plugin not found at {plugin_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Using plugin: {plugin_path}")
    test_alloc_barriers_ir()
    test_alloc_warp_barrier_ir()
    print("\nAll tests passed.")
