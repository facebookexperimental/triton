import triton
import triton.language as tl
from triton._filecheck import run_parser


@triton.jit
def _deep_binop_template():
    x = tl.arange(0, 1)
    y = x  # REPLACE_WITH_DEEP_BINOP
    tl.static_assert(y.shape[0] == 1)


def test_deeply_nested_binop_does_not_recurse():
    expr = "x" + " + x" * 1500
    kernel = triton.JITFunction(_deep_binop_template.fn)
    kernel._unsafe_update_src(kernel.src.replace("y = x  # REPLACE_WITH_DEEP_BINOP", f"y = {expr}"))

    run_parser(kernel)
