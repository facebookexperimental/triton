import pytest

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._C.libtriton import ir
from triton._internal_testing import is_hopper
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend


@triton.jit
def _raw_ttir_async_dot_kernel():
    a_tiles = tlx.local_alloc((64, 32), tl.float16, 1)
    b_tiles = tlx.local_alloc((32, 64), tl.float16, 1)
    acc = tlx.async_dot(a_tiles[0], b_tiles[0])
    tlx.async_dot_wait(0, acc)


@triton.jit
def _raw_ttir_ws_async_dot_kernel():
    a_tiles = tlx.local_alloc((64, 32), tl.float16, 1)
    b_tiles = tlx.local_alloc((32, 64), tl.float16, 1)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            _ = tl.arange(0, 1)
        with tlx.async_task(num_warps=4):
            acc = tlx.async_dot(a_tiles[0], b_tiles[0])
            tlx.async_dot_wait(0, acc)


@pytest.mark.skipif(not is_hopper(), reason="Need Hopper")
def test_raw_ttir_async_dot_has_num_warps():
    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 8})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(fn=_raw_ttir_async_dot_kernel, signature={}, constexprs={})

    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )

    assert module.get_int_attr("ttg.num-warps") == 8
    assert module.get_int_attr("ttg.threads-per-warp") == 32
    assert module.get_int_attr("ttg.num-ctas") == 1
    assert module.get_bool_attr("tlx.has_tlx_ops")
    assert module.verify()


@pytest.mark.skipif(not is_hopper(), reason="Need Hopper")
def test_raw_ttir_async_dot_uses_partition_num_warps():
    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 8})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(fn=_raw_ttir_ws_async_dot_kernel, signature={}, constexprs={})

    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    module_text = str(module)

    assert module.get_int_attr("ttg.num-warps") == 8
    assert module_text.count("partition0(") == 1
    assert module_text.count("num_warps(4)") == 1
    assert module_text.count("warpsPerCTA = [4, 1]") == 1
    assert module.verify()
