"""TLX layout tests -- Hopper-only."""
import pytest
import triton
import triton.language as tl
from triton._internal_testing import is_hopper
import triton.language.extra.tlx as tlx


@pytest.mark.skipif(not is_hopper(), reason="Need Hopper")
def test_nv_mma_tiled_shared_linear_reinterpret_pin_false_on_hopper():
    """A non-pinned shared-linear view does not retag its NVMMA backing buffer."""
    score_layout = tlx.nv_mma_shared_layout_encoding(
        (64, 64),
        [1, 0],
        tl.bfloat16,
        [1, 1],
        [1, 1],
        [1, 0],
        False,
        True,
    ).tile_to_shape((64, 128))

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        backing = tlx.local_alloc((64, 128), tl.bfloat16, 1)
        view = tlx.local_reinterpret(backing[0], tl.bfloat16, [64, 128], layout=LAYOUT, pin=False)
        subview = tlx.local_slice(view, [0, 0], [64, 64])
        values = tl.full((64, 64), 1.0, tl.bfloat16)
        tlx.local_store(subview, values)

    compiled = kernel.warmup(score_layout, grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_reinterpret" in ttgir
    assert "#ttg.shared_linear" in ttgir
    assert "#tlx.user_layout" not in ttgir
