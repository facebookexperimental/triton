import pytest
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx


# The FA4 "separable" layout for a 128x128 TMEM tile, written purely as
# shape/stride (a CuTe thread-value layout). The two top-level modes are
# (thread, value); strides are flat row-major offsets into the tile
# (offset = n * 128 + m). The compiler splits the thread bits into lane/warp and
# the value bits into registers, producing the #linear encoding below (the
# "value -> M, thread -> N" separable layout used by the MXFP8 attention
# backward).
def _separable_qk_layout():
    return tlx.layout(
        shape=((32, 4, 2), (32, 2)),  # (thread, value)
        stride=((128, 4096, 32), (1, 64)),
    )


_SEPARABLE_QK_LINEAR = (
    "#ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 64]], "
    "lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], "
    "warp = [[32, 0], [64, 0], [0, 32]], block = []}>")


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_layout_shape_stride_maps_to_linear():
    """A shape/stride `tlx.layout` passed to `local_load(layout=...)` lowers to
    the expected #linear encoding (no register/lane/warp on the user surface)."""

    @triton.jit
    def kernel(LAYOUT: tl.constexpr):
        qk = tlx.local_alloc((128, 128), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        v = tlx.local_view(qk, 0)
        x = tlx.local_load(v, layout=LAYOUT)
        tlx.local_store(v, x)

    # 3 warp bases -> 2**3 = 8 warps; the layout requires num_warps == 8.
    compiled = kernel.warmup(_separable_qk_layout(), grid=(1, ), num_warps=8)
    ttir = compiled.asm["ttir"]
    assert _SEPARABLE_QK_LINEAR in ttir
