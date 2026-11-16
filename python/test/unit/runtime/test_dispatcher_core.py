"""Test native CUDA and HIP dispatchers and their shared JIT launch contract."""

import contextlib
import json
import os
from types import SimpleNamespace

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs
from triton._internal_testing import is_cuda, is_hip, is_hip_cdna4
from triton.compiler.compiler import ASTSource

_LAUNCHER_SOURCE_GLOBAL = tl.constexpr(None)

pytestmark = pytest.mark.skipif(not (is_cuda() or is_hip()), reason="Requires CUDA or HIP backend")


@contextlib.contextmanager
def force_dispatcher():
    prev_env = os.environ.get("TRITON_USE_C_DISPATCHER")
    with knobs.runtime.scope():
        knobs.runtime.use_triton_dispatcher = True
        os.environ["TRITON_USE_C_DISPATCHER"] = "1"
        try:
            yield
        finally:
            if prev_env is None:
                os.environ.pop("TRITON_USE_C_DISPATCHER", None)
            else:
                os.environ["TRITON_USE_C_DISPATCHER"] = prev_env


@triton.jit
def _disp_add(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _disp_scalar_mix(x_ptr, out_ptr, fscale, iadd, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * fscale + iadd, mask=mask)


@triton.jit
def _disp_nop():
    pass


def test_dispatcher_core_add_numerics():
    N = 4096
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    with force_dispatcher():
        _disp_add[grid](x, y, out, N, BLOCK=256)
    torch.testing.assert_close(out, x + y)


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA consan instrumentation")
def test_dispatcher_core_uses_default_profile_allocator(fresh_knobs):
    from triton.runtime import _allocation

    N = 256
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    previous_profile_allocator = _allocation._profile_allocator.get()

    try:
        _allocation.set_profile_allocator(None)
        fresh_knobs.compilation.instrumentation_mode = "consan"
        with force_dispatcher():
            compiled = _disp_add.warmup(x, y, out, N, BLOCK=256, grid=(1, ))
            compiled._init_handles()
            assert compiled.metadata.global_scratch_size == 0
            assert compiled.metadata.profile_scratch_size > 0
            assert compiled._dispatcher is not None
            stream = triton.runtime.driver.active.get_current_stream(0)
            compiled._dispatcher(1, 1, 1, stream, x, y, out, N)
    finally:
        _allocation.set_profile_allocator(previous_profile_allocator)

    torch.testing.assert_close(out, x + y)


def test_dispatcher_core_mixed_scalar_args():
    N = 2048
    x = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    with force_dispatcher():
        _disp_scalar_mix[grid](x, out, 2.5, 3, N, BLOCK=128)
    torch.testing.assert_close(out, x * 2.5 + 3)


def test_dispatcher_core_nop():
    with force_dispatcher():
        _disp_nop[(1, )]()
    torch.cuda.synchronize()


def test_dispatcher_core_empty_grid():
    N = 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.zeros(N, device="cuda")
    with force_dispatcher():
        _disp_add[(0, )](x, y, out, N, BLOCK=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.zeros(N, device="cuda"))


def test_dispatcher_built_and_converged_path():
    """Directly verify a dispatcher is created (proves the dispatcher path is
    active, not a launchKernel fallback) and that calling it launches correctly
    through the shared core."""
    N = 2048
    BLOCK = 128
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    g = triton.cdiv(N, BLOCK)

    with force_dispatcher():
        compiled = _disp_add.warmup(x, y, out, N, BLOCK=BLOCK, grid=(g, ))
        if hasattr(compiled, "_init_handles"):
            compiled._init_handles()

        disp = getattr(compiled, "_dispatcher", None)
        assert disp is not None, ("expected a C dispatcher when TRITON_USE_C_DISPATCHER=1")

        dev = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(dev)
        # Dispatcher ABI: (grid_x, grid_y, grid_z, stream, *kernel_args)
        disp(g, 1, 1, stream, x, y, out, N)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y)


@triton.jit
def _disp_pid_write(out_ptr, GX: tl.constexpr, GY: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    offset = pid_x + GX * (pid_y + GY * pid_z)
    tl.store(out_ptr + offset, offset)


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA clusters")
def test_dispatcher_core_multidim_cluster():
    """An explicit multi-dimensional cluster (ctas_per_cga, which forces
    cluster_dims=(x,y,z) and num_ctas==1) must launch correctly through the
    shared core. With the dispatcher forced on, a non-TMA kernel now sets
    use_core_launch even for multi-dim clusters (previously gated off), so
    triton_launch_kernel builds the explicit
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION. A wrong/missing cluster dim would
    conflict with the PTX .reqnctapercluster baked into the cubin and fail the
    launch; every program writing its linear id proves all CTAs in the
    (2,1,3) cluster ran."""
    from triton._internal_testing import is_hopper_or_newer

    if not is_hopper_or_newer():
        pytest.skip("clusters need Hopper or newer")

    GX, GY, GZ = 4, 2, 3  # each divisible by the (2,1,3) cluster dims
    out = torch.full((GX * GY * GZ, ), -1, device="cuda", dtype=torch.int32)
    with force_dispatcher():
        _disp_pid_write[(GX, GY, GZ)](out, GX, GY, ctas_per_cga=(2, 1, 3))
    torch.cuda.synchronize()
    expected = torch.arange(GX * GY * GZ, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(out, expected)


@triton.jit
def _disp_scalar_write(out_ptr, value):
    tl.store(out_ptr, value)


@pytest.mark.parametrize("scalar_type,output_dtype,values", [
    ("i1", torch.bool, (False, True)),
    ("i8", torch.int8, (-117, 123)),
    ("i16", torch.int16, (-31001, 32001)),
    ("i32", torch.int32, (-2**30 - 3, 2**30 + 7)),
    ("i64", torch.int64, (-2**60 - 3, 2**60 + 7)),
    ("u8", torch.uint8, (7, 251)),
    ("u16", torch.uint16, (7, 65001)),
    ("u32", torch.uint32, (7, 2**32 - 3)),
    ("u64", torch.uint64, (7, 2**64 - 3)),
    ("fp16", torch.float16, (-1.5, 2.25)),
    ("bf16", torch.bfloat16, (-1.5, 2.25)),
    ("fp32", torch.float32, (-1.5, 2.25)),
    ("fp64", torch.float64, (-1.5, 2.25)),
])
def test_dispatcher_core_scalar_abi(scalar_type, output_dtype, values):
    out = torch.empty((), dtype=output_dtype, device="cuda")
    source = ASTSource(_disp_scalar_write, signature={"out_ptr": "*" + scalar_type, "value": scalar_type})
    with force_dispatcher():
        compiled = triton.compile(source)
        compiled._init_handles()
        assert compiled._dispatcher is not None
        device = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(device)
        for value in values:
            compiled._dispatcher(1, 1, 1, stream, out, value)
            assert out.item() == value


@triton.jit
def _disp_optional_write(out_ptr, optional_ptr):
    if optional_ptr.to(tl.uint64) == 0:
        value = 0
    else:
        value = tl.load(optional_ptr)
    tl.store(out_ptr, value)


def test_dispatcher_core_pointer_abi():
    out = torch.empty((), device="cuda", dtype=torch.int32)
    source = torch.full((), 17, device="cuda", dtype=torch.int32)
    with force_dispatcher():
        compiled = _disp_optional_write.warmup(out, source, grid=(1, ))
        compiled._init_handles()
        assert compiled._dispatcher is not None
        device = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(device)
        for pointer, expected in [(None, 0), (0, 0), (source, 17), (source.data_ptr(), 17)]:
            compiled._dispatcher(1, 1, 1, stream, out, pointer)
            assert out.item() == expected


@pytest.mark.parametrize("grid", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_dispatcher_core_direct_empty_grid(grid):
    out = torch.full((), 19, device="cuda", dtype=torch.int32)
    with force_dispatcher():
        compiled = _disp_scalar_write.warmup(out, 7, grid=(1, ))
        compiled._init_handles()
        assert compiled._dispatcher is not None
        device = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(device)
        compiled._dispatcher(*grid, stream, out, 7)
    assert out.item() == 19


def test_dispatcher_core_graph_capture():

    @triton.jit(c_cache=True)
    def add(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        tl.store(out_ptr + offsets, tl.load(x_ptr + offsets) + tl.load(y_ptr + offsets), mask=offsets < N)

    N = 256
    x = torch.ones(N, device="cuda")
    y = torch.full((N, ), 2.0, device="cuda")
    out = torch.empty_like(x)
    with force_dispatcher():
        compiled = add[(1, )](x, y, out, N, N)
        assert compiled._dispatcher is not None
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            add[(1, )](x, y, out, N, N)
        x.fill_(5)
        out.fill_(-1)
        graph.replay()
    torch.testing.assert_close(out, torch.full_like(out, 7))


@pytest.mark.parametrize("invalidation", ["all_devices", "kernel_cache", "device"])
def test_dispatcher_core_cache_clear(invalidation, fresh_knobs):

    @triton.jit(c_cache=True)
    def write(out_ptr, value):
        tl.store(out_ptr, value)

    compilations = []

    def cache_hook(**kwargs):
        compilations.append(kwargs["fn"])

    fresh_knobs.runtime.jit_cache_hook = cache_hook
    out = torch.empty((), device="cuda", dtype=torch.int32)
    with force_dispatcher():
        compiled = write[(1, )](out, 7)
        assert compiled._dispatcher is not None
        runner = write[(1, )]
        assert type(runner).__name__ == "_JITCacheProxy"
        runner(out, 9)
        assert len(compilations) == 1
        assert out.item() == 9
        device = triton.runtime.driver.active.get_current_device()
        if invalidation == "all_devices":
            write.device_caches.clear()
        elif invalidation == "kernel_cache":
            write.device_caches[device][0].clear()
        else:
            write.device_caches.pop(device)
        recompiled = runner(out, 11)
        assert len(compilations) == 2
        assert any(kernel is recompiled for kernel in write.device_caches[device][0].values())
        assert out.item() == 11


_DISPATCHER_GLOBAL = tl.constexpr(7)


@pytest.mark.parametrize("mutation", ["change", "delete"])
def test_dispatcher_core_global_invalidation(mutation, monkeypatch):

    @triton.jit(c_cache=True)
    def write(out_ptr):
        tl.store(out_ptr, _DISPATCHER_GLOBAL)

    out = torch.empty((), device="cuda", dtype=torch.int32)
    with force_dispatcher():
        compiled = write[(1, )](out)
        assert compiled._dispatcher is not None
        write[(1, )](out)
        assert out.item() == 7
        if mutation == "change":
            monkeypatch.setitem(globals(), "_DISPATCHER_GLOBAL", tl.constexpr(11))
        else:
            monkeypatch.delitem(globals(), "_DISPATCHER_GLOBAL")
        out.fill_(-1)
        with pytest.raises(RuntimeError, match="Global variable _DISPATCHER_GLOBAL has changed"):
            write[(1, )](out)
        assert out.item() == -1


def test_dispatcher_core_pre_run_hook():

    @triton.jit(c_cache=True)
    def add(x_ptr, out_ptr):
        tl.store(out_ptr, tl.load(x_ptr) + 1)

    x = torch.full((), 7, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    calls = []

    def pre_run_hook(x_ptr, out_ptr, **_):
        calls.append((x_ptr, out_ptr))
        x_ptr.fill_(11)

    with force_dispatcher():
        compiled = add[(1, )](x, out)
        assert compiled._dispatcher is not None
        assert out.item() == 8
        add.add_pre_run_hook(pre_run_hook)
        add[(1, )](x, out)
        add[(1, )](x, out)
    assert len(calls) == 2
    assert out.item() == 12


def test_dispatcher_core_launch_hooks(fresh_knobs):

    def launch_metadata(grid, kernel, args):
        return {"value": args["value"], "grid": grid}

    @triton.jit(c_cache=True, launch_metadata=launch_metadata)
    def write(out_ptr, value):
        tl.store(out_ptr, value)

    calls = []

    def enter_hook(metadata):
        calls.append(("enter", metadata.get()["value"], metadata.get()["grid"]))

    def exit_hook(metadata):
        calls.append(("exit", metadata.get()["value"], metadata.get()["grid"]))

    enter_chain = fresh_knobs.runtime.launch_enter_hook
    exit_chain = fresh_knobs.runtime.launch_exit_hook
    out = torch.empty((), device="cuda", dtype=torch.int32)
    with force_dispatcher():
        compiled = write[(1, )](out, 7)
        assert compiled._dispatcher is not None
        enter_chain.add(enter_hook)
        exit_chain.add(exit_hook)
        try:
            write[(1, )](out, 9)
            write[(1, )](out, 11)
            compiled[(1, 1, 1)](out, 13)
        finally:
            enter_chain.remove(enter_hook)
            exit_chain.remove(exit_hook)
    assert calls == [("enter", 9, (1, )), ("exit", 9, (1, )), ("enter", 11, (1, )), ("exit", 11, (1, )),
                     ("enter", 13, (1, 1, 1)), ("exit", 13, (1, 1, 1))]
    assert out.item() == 13


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPU devices")
def test_dispatcher_core_device_switch():

    @triton.jit(c_cache=True)
    def write(out_ptr, value):
        tl.store(out_ptr, value)

    first_device = torch.cuda.current_device()
    second_device = (first_device + 1) % torch.cuda.device_count()
    with force_dispatcher():
        for device, value in [(first_device, 7), (second_device, 9), (first_device, 11)]:
            with torch.cuda.device(device):
                out = torch.empty((), device="cuda", dtype=torch.int32)
                compiled = write[(1, )](out, value)
                assert compiled._dispatcher is not None
                assert out.item() == value


@pytest.mark.skipif(not is_hip(), reason="Requires HIP dispatcher")
@pytest.mark.parametrize("grid", [(-1, 1, 1), (1, -1, 1), (1, 1, -1), (2**31, 1, 1)])
def test_hip_dispatcher_rejects_invalid_grid(grid):
    with force_dispatcher():
        compiled = _disp_nop.warmup(grid=(1, ))
        compiled._init_handles()
        assert compiled._dispatcher is not None
        device = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(device)
        with pytest.raises(ValueError, match="HIP grid dimensions"):
            compiled._dispatcher(*grid, stream)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP compiled runner")
def test_hip_dispatcher_compiled_runner_specializations():

    @triton.jit
    def write(out_ptr, value, equal_one, BLOCK: tl.constexpr):
        tl.store(out_ptr, value + equal_one + BLOCK)

    out = torch.empty((), device="cuda", dtype=torch.int32)
    with force_dispatcher():
        compiled = write.warmup(out, 7, 1, 16, grid=(1, ))
        compiled._init_handles()
        assert compiled._dispatcher is not None
        runner = compiled[(1, 1, 1)]
        for value in [7, 9]:
            runner(out, value, 1, 16)
            assert out.item() == value + 17


@pytest.mark.skipif(not is_hip(), reason="Requires HIP launch metadata")
def test_hip_dispatcher_omitted_constexpr_signature():

    @triton.jit
    def write(out_ptr, BLOCK: tl.constexpr, value):
        tl.store(out_ptr, value + BLOCK)

    source = ASTSource(write, signature={"out_ptr": "*i32", "value": "i32"}, constexprs={"BLOCK": 16})
    out = torch.empty((), device="cuda", dtype=torch.int32)
    with force_dispatcher():
        compiled = triton.compile(source)
        compiled._init_handles()
        assert compiled._dispatcher is not None
        device = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(device)
        compiled._dispatcher(1, 1, 1, stream, out, 7)
        assert out.item() == 23
        compiled[(1, 1, 1)](out, 9)
        assert out.item() == 25


@pytest.mark.skipif(not is_hip(), reason="Requires HIP compiled runner")
@pytest.mark.parametrize("control", ["dispatcher_flag", "launch_hook"])
def test_hip_dispatcher_compiled_runner_dynamic_controls(control, fresh_knobs, monkeypatch):
    out = torch.empty((), device="cuda", dtype=torch.int32)
    native_calls = []
    hook_calls = []

    def launch_hook(metadata):
        hook_calls.append(metadata.get()["name"])

    hook_chain = fresh_knobs.runtime.launch_enter_hook
    with force_dispatcher():
        compiled = _disp_scalar_write.warmup(out, 7, grid=(1, ))
        compiled._init_handles()
        native_dispatcher = compiled._dispatcher
        assert native_dispatcher is not None

        def record_dispatch(*args):
            native_calls.append(args)
            return native_dispatcher(*args)

        monkeypatch.setattr(compiled, "_dispatcher", record_dispatch)
        runner = compiled[(1, 1, 1)]
        runner(out, 7)
        assert len(native_calls) == 1
        native_calls.clear()
        try:
            if control == "dispatcher_flag":
                knobs.runtime.use_triton_dispatcher = False
            else:
                hook_chain.add(launch_hook)
            runner(out, 9)
            assert native_calls == []
            assert len(hook_calls) == (control == "launch_hook")
            assert out.item() == 9
        finally:
            hook_chain.remove(launch_hook)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP dispatcher")
@pytest.mark.parametrize("field,value", [
    ("num_warps", 0),
    ("warp_size", 0),
    ("num_ctas", 0),
    ("shared_mem", -1),
    ("global_scratch_size", -1),
    ("profile_scratch_size", -1),
    ("global_scratch_align", 0),
    ("profile_scratch_align", 0),
])
def test_hip_dispatcher_rejects_invalid_metadata(field, value):
    from triton.backends.amd.triton_dispatcher_factory import make_triton_dispatcher

    with force_dispatcher():
        compiled = _disp_nop.warmup(grid=(1, ))
        compiled._init_handles()
        assert compiled._dispatcher is not None
        schema = json.loads(compiled.asm["launch_metadata"])
        schema[field] = value
        with pytest.raises(ValueError, match="Invalid HIP"):
            make_triton_dispatcher(schema, compiled.function)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP dispatcher")
def test_hip_dispatcher_rejects_argument_count():
    with force_dispatcher():
        compiled = _disp_nop.warmup(grid=(1, ))
        compiled._init_handles()
        assert compiled._dispatcher is not None
        device = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(device)
        with pytest.raises(TypeError, match="positional HIP dispatcher arguments"):
            compiled._dispatcher(1, 1, 1)
        with pytest.raises(TypeError, match="positional HIP dispatcher arguments"):
            compiled._dispatcher(1, 1, 1, stream, 7)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP dispatcher")
def test_hip_dispatcher_scratch_allocators():
    allocations = []

    def allocate_global(size, alignment, stream):
        allocations.append(("global", size, alignment, stream))
        return torch.empty(size, dtype=torch.uint8, device="cuda")

    def allocate_profile(size, alignment, stream):
        allocations.append(("profile", size, alignment, stream))
        return torch.empty(size, dtype=torch.uint8, device="cuda")

    with force_dispatcher():
        compiled = _disp_nop.warmup(grid=(1, ))
        compiled._init_handles()
        assert compiled._dispatcher is not None
        schema = json.loads(compiled.asm["launch_metadata"])
        schema.update(global_scratch_size=16, global_scratch_align=16, profile_scratch_size=32,
                      profile_scratch_align=32)
        utils = triton.runtime.driver.active.utils
        dispatcher = utils._TritonDispatcher(schema, compiled.function, b"",
                                             SimpleNamespace(get=lambda: allocate_global),
                                             SimpleNamespace(get=lambda: allocate_profile))
        device = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(device)
        dispatcher(2, 3, 4, stream)
        torch.cuda.synchronize()
    assert allocations == [("global", 24 * 16, 16, stream), ("profile", 24 * 32, 32, stream)]


@pytest.mark.skipif(not is_hip(), reason="Requires HIP dispatcher")
def test_hip_dispatcher_default_profile_allocator(monkeypatch):
    from triton.backends.amd.triton_dispatcher_factory import make_triton_dispatcher
    from triton.runtime import _allocation

    allocations = []

    def allocate_profile(size, alignment, stream):
        allocations.append((size, alignment, stream))
        return torch.empty(size, dtype=torch.uint8, device="cuda")

    active_driver = triton.runtime.driver.active
    monkeypatch.setattr(active_driver, "allocate_default_profile_scratch", allocate_profile)
    previous_allocator = _allocation._profile_allocator.get()
    try:
        _allocation.set_profile_allocator(None)
        with force_dispatcher():
            compiled = _disp_nop.warmup(grid=(1, ))
            compiled._init_handles()
            assert compiled._dispatcher is not None
            schema = json.loads(compiled.asm["launch_metadata"])
            schema.update(profile_scratch_size=32, profile_scratch_align=16)
            dispatcher = make_triton_dispatcher(schema, compiled.function)
            stream = active_driver.get_current_stream(active_driver.get_current_device())
            dispatcher(2, 3, 1, stream)
            torch.cuda.synchronize()
    finally:
        _allocation.set_profile_allocator(previous_allocator)
    assert allocations == [(6 * 32, 16, stream)]


@pytest.mark.skipif(not is_hip(), reason="Requires HIP pointer extraction")
@pytest.mark.parametrize("pointer_value,error", [("invalid", TypeError), (-1, OverflowError), (2**64, OverflowError)])
def test_hip_dispatcher_rejects_invalid_pointer(pointer_value, error):
    out = torch.full((), 19, device="cuda", dtype=torch.int32)
    invalid_pointer = SimpleNamespace(data_ptr=lambda: pointer_value)
    with force_dispatcher():
        compiled = _disp_scalar_write.warmup(out, 7, grid=(1, ))
        compiled._init_handles()
        assert compiled._dispatcher is not None
        device = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(device)
        with pytest.raises(error):
            compiled._dispatcher(1, 1, 1, stream, invalid_pointer, 7)
        compiled._dispatcher(1, 1, 1, stream, out, 11)
    assert out.item() == 11


def test_dispatcher_core_structured_argument_fallback():

    @triton.jit(c_cache=True)
    def write(arguments):
        tl.store(arguments[0], arguments[1])

    out = torch.empty((), device="cuda", dtype=torch.int32)
    with force_dispatcher():
        for value in [7, 9]:
            with pytest.warns(UserWarning, match="has no C dispatcher"):
                compiled = write[(1, )]((out, value))
            assert compiled._dispatcher is None
            assert out.item() == value


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("hook_kind", ["pre_run", "runtime_cache"])
def test_hip_dispatcher_jit_hooks_after_warmup(monkeypatch, hook_kind):

    @triton.jit
    def add(X, Y, BIAS: tl.constexpr):
        offsets = tl.arange(0, 64)
        tl.store(Y + offsets, tl.load(X + offsets) + BIAS)

    launch = add[(1, )]
    x = torch.arange(64, device="cuda", dtype=torch.int32)
    y = torch.empty_like(x)
    launch(x, y, 2)
    launch(x, y, 2)
    torch.testing.assert_close(y, x + 2, atol=0, rtol=0)
    direct = torch.empty_like(y)
    add[(1, )](x, direct, 2)
    torch.testing.assert_close(y, direct, atol=0, rtol=0)
    calls = []
    if hook_kind == "pre_run":

        def hook(*args, **kwargs):
            calls.append(True)
            args[0].fill_(7)

        add.add_pre_run_hook(hook)
        bias = 2
    else:

        def hook(*args, **kwargs):
            calls.append(True)

        monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", hook)
        bias = 3
    launch(x, y, bias)
    torch.testing.assert_close(y, x + bias, atol=0, rtol=0)
    assert calls == [True]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_hip_dispatcher_jit_rejects_deleted_none_global(monkeypatch):
    """Preserve ordinary JIT rejection after deleting a tracked source global."""
    monkeypatch.setitem(globals(), "_LAUNCHER_SOURCE_GLOBAL", tl.constexpr(None))

    @triton.jit
    def copy_with_global(X, Y):
        offsets = tl.arange(0, 64)
        if _LAUNCHER_SOURCE_GLOBAL is None:
            values = tl.load(X + offsets)
        else:
            values = tl.load(X + offsets) + 1
        tl.store(Y + offsets, values)

    x = torch.arange(64, device="cuda", dtype=torch.int32)
    direct = torch.empty_like(x)
    ordinary = copy_with_global[(1, )](x, direct)
    torch.testing.assert_close(direct, x, atol=0, rtol=0)
    launch = copy_with_global[(1, )]
    y = torch.empty_like(x)
    compiled = launch(x, y)
    assert compiled is ordinary
    launch(x, y)
    torch.testing.assert_close(y, direct, atol=0, rtol=0)

    monkeypatch.delitem(globals(), "_LAUNCHER_SOURCE_GLOBAL")
    direct.fill_(-1)
    y.fill_(-1)
    diagnostic = "Global variable _LAUNCHER_SOURCE_GLOBAL has changed since we compiled this kernel"
    with pytest.raises(RuntimeError, match=diagnostic):
        copy_with_global[(1, )](x, direct)
    with pytest.raises(RuntimeError, match=diagnostic):
        launch(x, y)
    torch.testing.assert_close(direct, torch.full_like(direct, -1), atol=0, rtol=0)
    torch.testing.assert_close(y, direct, atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("invalidate", ["entry", "device_cache"])
def test_hip_dispatcher_jit_cache_invalidation(invalidate):

    @triton.jit
    def copy(X, Y):
        offsets = tl.arange(0, 64)
        tl.store(Y + offsets, tl.load(X + offsets))

    launch = copy[(1, )]
    x = torch.arange(64, device="cuda", dtype=torch.int32)
    y = torch.empty_like(x)
    launch(x, y)
    launch(x, y)
    device = torch.cuda.current_device()
    if invalidate == "entry":
        copy.device_caches[device][0].clear()
    else:
        copy.device_caches.pop(device)
    replacement = x + 11
    compiled = launch(replacement, y)
    torch.testing.assert_close(y, replacement, atol=0, rtol=0)
    direct = torch.empty_like(y)
    ordinary = copy[(1, )](replacement, direct)
    torch.testing.assert_close(y, direct, atol=0, rtol=0)
    assert compiled is ordinary
    assert any(compiled is value for value in copy.device_caches[device][0].values())
    launch(x, y)
    torch.testing.assert_close(y, x, atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_hip_dispatcher_jit_specialization_and_graph_capture():

    @triton.jit
    def affine(X, Y, scale, BIAS: tl.constexpr):
        offsets = tl.arange(0, 64)
        tl.store(Y + offsets, tl.load(X + offsets) * scale + BIAS)

    launch = affine[(1, )]
    specializations = [
        (torch.int32, 0, 2, 1, 4),
        (torch.int32, 1, 2, 1, 4),
        (torch.int32, 1, 3, 1, 4),
        (torch.int32, 1, 3, 4, 4),
        (torch.float32, 1, 3, 4, 4),
        (torch.float32, 1, 3, 4, 8),
        (torch.int32, 0, 2, 1, 4),
    ]
    for dtype, shift, scale, bias, warps in specializations:
        storage = torch.arange(65, device="cuda", dtype=dtype)
        x = storage[shift:shift + 64]
        y = torch.empty_like(x)
        compiled = launch(x, y, scale, bias, num_warps=warps)
        torch.testing.assert_close(y, x * scale + bias, atol=0, rtol=0)
        direct = torch.empty_like(y)
        ordinary = affine[(1, )](x, direct, scale, bias, num_warps=warps)
        torch.testing.assert_close(y, direct, atol=0, rtol=0)
        assert compiled is ordinary
    replacement = torch.arange(64, device="cuda", dtype=torch.int32) + 17
    launch(replacement, y, 2, 1, num_warps=4)
    torch.testing.assert_close(y, replacement * 2 + 1, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        launch(replacement, y, 2, 1, num_warps=4)
    replacement.add_(5)
    y.zero_()
    graph.replay()
    torch.testing.assert_close(y, replacement * 2 + 1, atol=0, rtol=0)
