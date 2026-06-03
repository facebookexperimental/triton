"""Compile-only smoke test for tlx.dot_scaled `tiles_per_warp` attribute.

Verifies that passing `tiles_per_warp=[m, n]` to tlx.dot_scaled materializes
the `amdg.wmma_tiles_per_warp` attribute on the produced tt.dot_scaled op.
This is the Python -> C++ contract consumed by AccelerateAMDMatmul.
"""
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _dot_scaled_tiles_per_warp_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // SCALE_BLOCK
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_ks = tl.arange(0, BLOCK_K_SCALE)

    a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    a_scale = tl.load(a_scale_ptr + offs_m[:, None] * BLOCK_K_SCALE + offs_ks[None, :])
    b_scale = tl.load(b_scale_ptr + offs_n[:, None] * BLOCK_K_SCALE + offs_ks[None, :])

    acc = tlx.dot_scaled(a, a_scale, "e5m2", b, b_scale, "e5m2", tiles_per_warp=[2, 2])
    tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc)


def test_dot_scaled_tiles_per_warp_attr_in_ttir():
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    src = ASTSource(
        fn=_dot_scaled_tiles_per_warp_kernel,
        signature={
            "a_ptr": "*fp8e5",
            "b_ptr": "*fp8e5",
            "a_scale_ptr": "*i8",
            "b_scale_ptr": "*i8",
            "c_ptr": "*fp32",
        },
        constexprs={
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "SCALE_BLOCK": 32,
        },
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))
    ttir = compiled.asm["ttir"]
    assert "amdg.wmma_tiles_per_warp = array<i32: 2, 2>" in ttir, ttir
