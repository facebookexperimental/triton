import ast
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

import triton
from triton._C.libtriton import ir
from triton._C.libtriton.linear_layout import LinearLayout
import triton.language as tl
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile, make_backend
from triton.runtime.jit import MockTensor

if "tlx_wave" in backends:
    from triton.backends.tlx_wave import compiler as tlx_wave_compiler
    from triton.backends.tlx_wave import driver as tlx_wave_driver
    from triton.backends.tlx_wave import wave_bridge_tools
    from triton.backends.tlx_wave.converter import barrier_order as converter_barrier_order
    from triton.backends.tlx_wave.converter import canonicalize as converter_canonicalize
    from triton.backends.tlx_wave.converter import diagnostics as converter_diagnostics
    from triton.backends.tlx_wave.converter import domains as converter_domains
    from triton.backends.tlx_wave.converter import emission as converter_emission
    from triton.backends.tlx_wave.converter import facts as converter_facts
    from triton.backends.tlx_wave.converter import coordinates as converter_coordinates
    from triton.backends.tlx_wave.converter import layouts as converter_layouts
    from triton.backends.tlx_wave.converter import op_conversion as converter_op_conversion
    from triton.backends.tlx_wave.converter import pipeline as converter_pipeline
    from triton.backends.tlx_wave.converter import source_import as converter_source_import
    from triton.backends.tlx_wave.converter import source_ir as converter_source_ir
    from triton.backends.tlx_wave.converter import target_ir as converter_target_ir
    from triton.backends.tlx_wave.converter import tokens as converter_tokens
    from triton.backends.tlx_wave.converter import types as converter_types
    from triton.backends.tlx_wave.converter import verifier as converter_verifier
else:
    tlx_wave_compiler = None
    tlx_wave_driver = None
    wave_bridge_tools = None
    converter_barrier_order = None
    converter_canonicalize = None
    converter_diagnostics = None
    converter_domains = None
    converter_emission = None
    converter_facts = None
    converter_coordinates = None
    converter_layouts = None
    converter_op_conversion = None
    converter_pipeline = None
    converter_source_import = None
    converter_source_ir = None
    converter_target_ir = None
    converter_tokens = None
    converter_types = None
    converter_verifier = None

pytestmark = pytest.mark.skipif("tlx_wave" not in backends, reason="tlx_wave backend is not installed")

GFX942_WAVE = GPUTarget("tlx_wave", "gfx942", 64)
GFX950_WAVE = GPUTarget("tlx_wave", "gfx950", 64)
_TLX_WAVE_RUNTIME_ARCHES = {"gfx942", "gfx950"}
_LEGACY_LAYOUT_PLAN_ATTRS = frozenset({
    "block_count",
    "component_sources",
    "cta_thread_count",
    "mode",
    "packet_source_indices",
    "register_payload_source_slots",
    "registers",
    "relation_bases",
    "relation_out_dims",
    "result_component_count",
    "result_packet_width",
    "result_registers_per_component",
    "result_slot_count",
    "result_value_mode",
    "scalar_source_slots",
    "scalar_sources",
    "source_component_count",
    "source_packet_width",
    "source_registers_per_component",
    "source_slot_count",
})


def _fake_layout(
        layout_map_id,
        value_id,
        *,
        kind="blocked",
        shape=(8, 8),
        element_type="i32",
        component_count=1,
        lane_width=64,
        properties=None,
):
    return converter_layouts.LayoutMap(
        layout_map_id,
        value_id,
        kind,
        tuple(shape),
        element_type,
        int(component_count),
        int(lane_width),
        dict(properties or {}),
    )


def _converted_value(
    value_id,
    *,
    kind="tensor",
    representation="simd",
    element_type="i32",
    component_count=1,
    layout_map_id=None,
):
    return converter_types.ConvertedValue(
        value_id,
        converter_types.ConvertedType(
            kind,
            representation,
            element_type,
            64,
            int(component_count),
        ),
        layout_map_id,
    )


def _identity_target_layout(layout_map_id=0, *, shape=(64, ), element_type="i32"):
    assert tuple(shape) == (64, )
    linear = converter_target_ir.TargetLinearLayout(
        ("lane", ),
        (("dim0", 64), ),
        (("lane", ((1, ), (2, ), (4, ), (8, ), (16, ), (32, ))), ),
    )
    return converter_target_ir.TargetLayout(
        int(layout_map_id),
        "linear",
        tuple(shape),
        element_type,
        1,
        64,
        linear_layout=linear,
    )


def _cross_wave_target_layouts():
    source = converter_target_ir.TargetLinearLayout(
        ("lane", "warp"),
        (("dim0", 128), ),
        (
            ("lane", ((1, ), (2, ), (4, ), (8, ), (16, ), (32, ))),
            ("warp", ((64, ), )),
        ),
    )
    result = converter_target_ir.TargetLinearLayout(
        ("lane", "warp"),
        (("dim0", 128), ),
        (
            ("lane", ((1, ), (2, ), (4, ), (8, ), (16, ), (64, ))),
            ("warp", ((32, ), )),
        ),
    )
    return tuple(
        converter_target_ir.TargetLayout(
            layout_map_id,
            "linear",
            (128, ),
            "f32",
            1,
            64,
            linear_layout=linear,
        ) for layout_map_id, linear in enumerate((source, result)))


def _asm_text(compiled, artifact):
    text = compiled.asm[artifact]
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return text


def _tlx_wave_physical_arch(properties):
    return str(properties.get("arch", "")).split(":")[0]


def _tlx_wave_runtime_skip_reason(arch):
    supported = "/".join(sorted(_TLX_WAVE_RUNTIME_ARCHES))
    return (f"requires physical {supported} hardware for TLX Wave launch tests, "
            f"got {arch or 'unknown'}; this is a runtime launch guard, not a "
            "Wave HSACO generation failure. Compile-only TLX Wave tests may target "
            "gfx942/gfx950 without matching local hardware.")


def _with_target_op_attrs(target_program, target_op_id, **attrs):
    updated_ops = []
    found = False
    for op in target_program.ops:
        if op.target_op_id != target_op_id:
            updated_ops.append(op)
            continue
        found = True
        updated_attrs = converter_target_ir.attrs_dict(op)
        updated_attrs.update(attrs)
        updated_ops.append(replace(
            op,
            attrs=converter_target_ir._attrs_tuple(
                updated_attrs,
                op.target_op_id,
            ),
        ))
    assert found, target_op_id
    return replace(target_program, ops=tuple(updated_ops))


def _target_value_producer(target_program, target_value_id, *, kind=None):
    producers = [
        op for op in target_program.ops if int(target_value_id) in op.results and (kind is None or op.kind == kind)
    ]
    assert len(producers) == 1, (target_value_id, kind, producers)
    return producers[0]


def _assert_mechanical_layout_transform(
    target_program,
    op,
    *,
    transform=None,
    axis=None,
    order=None,
):
    if op.kind == "layout_convert":
        expected = {
            "fact_policy": "invalidate_layout_sensitive",
            "transform": transform or "identity",
        }
        if order is not None:
            expected["order"] = tuple(order)
    elif op.kind == "expand_dims":
        expected = {"axis": int(axis)}
    else:
        expected = {}
    attrs = converter_target_ir.attrs_dict(op)
    assert attrs == expected
    assert not _LEGACY_LAYOUT_PLAN_ATTRS.intersection(attrs)
    for target_value_id in (*op.operands, *op.results):
        target_value = target_program.values[int(target_value_id)]
        assert target_value.layout_map_id is not None
        layout = target_program.layouts[int(target_value.layout_map_id)]
        assert layout.linear_layout is not None
        assert tuple(size for _name, size in layout.linear_layout.out_dims) == layout.shape


def _assert_layout_transforms_reach_wave(output):
    expected = sum(2 if op.kind == "split" else 1
                   for op in output.target_program.ops
                   if op.kind in {"broadcast", "expand_dims", "join", "layout_convert", "split"})
    assert output.emitted_module.text.count("wave.redistribute") == expected


def _packetized_local_load_attrs(output):
    attrs = [converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "local_load"]
    return [entry for entry in attrs if int(entry.get("packet_count", 1)) > 1]


def _memory_offset_edge(target_program, memory_op):
    offset_operand_index = {
        "buffer_load": 1,
        "buffer_load_to_local": 2,
        "buffer_store": 2,
    }[memory_op.kind]
    memory_attrs = converter_target_ir.attrs_dict(memory_op)
    assert memory_attrs.get("offset_mode", "operand") == "operand"
    assert "offset_terms" not in memory_attrs
    assert "source_offset_terms" not in memory_attrs
    offset_target_id = memory_op.operands[offset_operand_index]
    edge = _target_value_producer(target_program, offset_target_id)
    edge_attrs = converter_target_ir.attrs_dict(edge)
    assert edge.kind != "type_convert"
    live_op_ids = {int(op_id) for region in target_program.regions for op_id in region.op_ids}
    assert edge.target_op_id in live_op_ids
    return edge, edge_attrs


def _memory_mask_edge(target_program, memory_op):
    mask_operand_index = {
        "buffer_load": 2,
        "buffer_load_to_local": 3,
        "buffer_store": 3,
    }[memory_op.kind]
    attrs = converter_target_ir.attrs_dict(memory_op)
    assert attrs["has_mask"] is True
    assert attrs.get("mask_operand_mode", "operand") == "operand"
    assert "mask_predicate_plans" not in attrs
    return _target_value_producer(
        target_program,
        memory_op.operands[mask_operand_index],
    )


def _require_tlx_wave_runtime_target():
    torch = pytest.importorskip("torch")
    try:
        active_driver = triton.runtime.driver.active
        device = active_driver.get_current_device()
        properties = active_driver.utils.get_device_properties(device)
    except Exception as exc:
        pytest.skip(f"requires an active HIP runtime for TLX Wave launch tests: {exc}")
    arch = _tlx_wave_physical_arch(properties)
    if arch not in _TLX_WAVE_RUNTIME_ARCHES:
        pytest.skip(_tlx_wave_runtime_skip_reason(arch))
    if not torch.cuda.is_available():
        pytest.skip("requires torch.cuda/ROCm for TLX Wave launch tests")
    return torch, arch


@contextmanager
def _active_tlx_wave_driver():
    previous_driver = triton.runtime.driver.active
    triton.runtime.driver.set_active(tlx_wave_driver.TLXWaveDriver())
    try:
        yield
    finally:
        triton.runtime.driver.set_active(previous_driver)


@contextmanager
def _active_amd_driver():
    from triton.backends.amd import driver as amd_driver

    previous_driver = triton.runtime.driver.active
    triton.runtime.driver.set_active(amd_driver.HIPDriver())
    try:
        yield
    finally:
        triton.runtime.driver.set_active(previous_driver)


@contextmanager
def _tlx_wave_compile_driver(monkeypatch):
    previous_default = triton.runtime.driver._default
    previous_active = triton.runtime.driver._active
    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "tlx_wave")
    try:
        triton.runtime.driver._default = None
        triton.runtime.driver._active = None
        active_driver = triton.runtime.driver.active
    except RuntimeError as exc:
        pytest.skip(f"requires active TLX Wave compile driver: {exc}")
    try:
        yield active_driver
    finally:
        triton.runtime.driver._default = previous_default
        triton.runtime.driver._active = previous_active


def _load_tlx_gfx9_gemm_module(version_dir, module_name=None):
    repo_root = Path(__file__).resolve().parents[4]
    kernel_path = (repo_root / "third_party" / "tlx" / "tutorials" / "gfx9_gemm" / "a16w16" / version_dir /
                   "matmul_kernel.py")
    spec = importlib.util.spec_from_file_location(
        module_name or f"_tlx_wave_test_{version_dir}",
        kernel_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tlx_gfx9_gemm_kernel(version_dir, function_name):
    module = _load_tlx_gfx9_gemm_module(
        version_dir,
        f"_tlx_wave_test_{version_dir}_{function_name}",
    )
    return getattr(module, function_name)


def _load_tlx_gfx9_gemm_bench_module(module_name="_tlx_wave_test_gfx9_bench"):
    repo_root = Path(__file__).resolve().parents[4]
    bench_path = (repo_root / "third_party" / "tlx" / "tutorials" / "gfx9_gemm" / "a16w16" / "bench.py")
    spec = importlib.util.spec_from_file_location(module_name, bench_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tlx_gfx9_inter_wave_bench_module(module_name="_tlx_wave_test_gfx9_inter_wave_bench"):
    repo_root = Path(__file__).resolve().parents[4]
    bench_dir = repo_root / "third_party" / "tlx" / "tutorials" / "gfx9_gemm" / "inter_wave" / "a16w16"
    bench_path = bench_dir / "bench.py"
    before_path = list(sys.path)
    previous_kernel_module = sys.modules.pop("matmul_kernel", None)
    try:
        sys.path.insert(0, str(bench_dir))
        spec = importlib.util.spec_from_file_location(module_name, bench_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = before_path
        if previous_kernel_module is None:
            sys.modules.pop("matmul_kernel", None)
        else:
            sys.modules["matmul_kernel"] = previous_kernel_module


def _load_tlx_gfx9_a4w4_module(module_name="_tlx_wave_test_gfx9_a4w4"):
    repo_root = Path(__file__).resolve().parents[4]
    kernel_path = (repo_root / "third_party" / "tlx" / "tutorials" / "gfx9_gemm" / "intra_wave" / "a4w4" /
                   "matmul_kernel.py")
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tlx_gfx9_a4w4_bench_module(module_name="_tlx_wave_test_gfx9_a4w4_bench"):
    repo_root = Path(__file__).resolve().parents[4]
    bench_dir = repo_root / "third_party" / "tlx" / "tutorials" / "gfx9_gemm" / "intra_wave" / "a4w4"
    bench_path = bench_dir / "bench.py"
    before_path = list(sys.path)
    previous_kernel_module = sys.modules.get("matmul_kernel")
    try:
        sys.path.insert(0, str(bench_dir))
        spec = importlib.util.spec_from_file_location(module_name, bench_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = before_path
        if previous_kernel_module is None:
            sys.modules.pop("matmul_kernel", None)
        else:
            sys.modules["matmul_kernel"] = previous_kernel_module


def _load_tlx_glu_bench_module(module_name="_tlx_wave_test_glu_bench"):
    repo_root = Path(__file__).resolve().parents[4]
    bench_path = repo_root / "third_party" / "tlx" / "tutorials" / "amd-addmm-glu-opt_test.py"
    spec = importlib.util.spec_from_file_location(module_name, bench_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tlx_fa_bench_module(module_name="_tlx_wave_test_fa_bench"):
    repo_root = Path(__file__).resolve().parents[4]
    bench_path = repo_root / "third_party" / "tlx" / "tutorials" / "amd-fa-pipelined_test.py"
    spec = importlib.util.spec_from_file_location(module_name, bench_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tlx_fa_wave_module(module_name="_tlx_wave_test_fa_wave"):
    repo_root = Path(__file__).resolve().parents[4]
    kernel_path = repo_root / "third_party" / "tlx" / "tutorials" / "amd_fa_wave.py"
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tlx_perf_sweep_module(module_name="_tlx_wave_test_perf_sweep"):
    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "third_party" / "tlx" / "tutorials" / "run_wave_perf_sweeps.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


_MXFP4_SCALE_GROUP_SIZE = 32


def _random_mxfp4_inputs(torch, m, n, k, device, *, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    a_low = torch.randint(0, 16, (m, k // 2), dtype=torch.uint8, device=device, generator=generator)
    a_high = torch.randint(0, 16, (m, k // 2), dtype=torch.uint8, device=device, generator=generator)
    b_low = torch.randint(0, 16, (n, k // 2), dtype=torch.uint8, device=device, generator=generator)
    b_high = torch.randint(0, 16, (n, k // 2), dtype=torch.uint8, device=device, generator=generator)

    a = (a_high << 4) | a_low
    b = (b_high << 4) | b_low
    k_scales = k // _MXFP4_SCALE_GROUP_SIZE
    m_pad = triton.cdiv(m, 256) * 256
    a_scales = torch.randint(
        124,
        128,
        (k_scales, m_pad),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    ).T[:m]
    b_scales = torch.randint(
        124,
        128,
        (k_scales, n),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    ).T
    return a, b, a_scales, b_scales


def _mxfp4_to_f32(torch, packed):
    unpacked = packed.repeat_interleave(2, dim=1)
    unpacked[:, ::2] = unpacked[:, ::2] & 0xF
    unpacked[:, 1::2] = unpacked[:, 1::2] >> 4
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    return values[unpacked.long()]


def _e8m0_to_f32(torch, scales):
    return torch.pow(2.0, scales.to(torch.int16).to(torch.float32) - 127.0)


def _mxfp4_gemm_reference(torch, a, b, a_scales, b_scales):
    a_f32 = _mxfp4_to_f32(torch, a) * _e8m0_to_f32(torch, a_scales).repeat_interleave(
        _MXFP4_SCALE_GROUP_SIZE,
        dim=1,
    )
    b_f32 = _mxfp4_to_f32(torch, b) * _e8m0_to_f32(torch, b_scales).repeat_interleave(
        _MXFP4_SCALE_GROUP_SIZE,
        dim=1,
    )
    return torch.mm(a_f32, b_f32.T).to(torch.bfloat16)


def test_tlx_gfx9_a4w4_random_inputs_use_gluon_scale_strides():
    torch = pytest.importorskip("torch")

    _a, _b, a_scales, b_scales = _random_mxfp4_inputs(
        torch,
        384,
        512,
        1024,
        torch.device("cpu"),
        seed=0,
    )

    assert a_scales.shape == (384, 32)
    assert b_scales.shape == (512, 32)
    assert a_scales.stride() == (1, 512)
    assert b_scales.stride() == (1, 512)


def test_tlx_gfx9_a4w4_precompile_uses_gluon_scale_strides(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    bench = _load_tlx_gfx9_a4w4_bench_module("_tlx_wave_test_gfx9_a4w4_bench_strides")
    call = {}

    def warmup(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs

    monkeypatch.setattr(bench._a4w4_kernel, "warmup", warmup)

    bench.compile_shape((384, 512, 1024), tmp_path)

    a_scales = call["args"][3]
    b_scales = call["args"][4]
    assert tuple(a_scales.shape) == (384, 32)
    assert tuple(b_scales.shape) == (512, 32)
    assert a_scales.stride() == (1, 512)
    assert b_scales.stride() == (1, 512)
    assert call["args"][14:18] == (1, 512, 1, 512)


def test_tlx_gfx9_a4w4_bench_launch_reuses_output(monkeypatch):
    torch = pytest.importorskip("torch")
    bench = _load_tlx_gfx9_a4w4_bench_module("_tlx_wave_test_gfx9_a4w4_bench_output")
    call = {}

    class FakeKernel:

        def __getitem__(self, grid):
            call["grid"] = grid

            def launch(*args, **kwargs):
                call["args"] = args
                call["kwargs"] = kwargs

            return launch

    monkeypatch.setattr(bench, "_a4w4_kernel", FakeKernel())
    a = torch.empty((256, 512), dtype=torch.uint8)
    b = torch.empty((256, 512), dtype=torch.uint8)
    a_scales = torch.empty((256, 32), dtype=torch.uint8)
    b_scales = torch.empty((256, 32), dtype=torch.uint8)
    out = torch.empty((256, 256), dtype=torch.bfloat16)

    result = bench.launch_matmul(a, b, a_scales, b_scales, out=out)

    assert result is out
    assert call["args"][2] is out
    assert call["grid"] == (1, )


def test_tlx_gfx9_a4w4_bench_batched_timing_uses_one_event_span_per_repeat():
    bench = _load_tlx_gfx9_a4w4_bench_module("_tlx_wave_test_gfx9_a4w4_bench_timing")
    state = {"launches": 0, "synchronizes": 0, "events": 0}

    class FakeEvent:

        def __init__(self):
            self.launch = None

        def record(self):
            self.launch = state["launches"]

        def elapsed_time(self, other):
            return (other.launch - self.launch) * 0.25

    class FakeDeviceInterface:

        def Event(self, *, enable_timing):
            assert enable_timing
            state["events"] += 1
            return FakeEvent()

        def synchronize(self):
            state["synchronizes"] += 1

    def launch():
        state["launches"] += 1

    ms = bench.do_bench_batched(
        launch,
        warmup_launches=2,
        timed_launches=4,
        repeats=3,
        device_interface=FakeDeviceInterface(),
    )

    assert ms == 0.25
    assert state == {"launches": 18, "synchronizes": 6, "events": 6}


def test_tlx_gfx9_a4w4_bench_triton_timing_reports_median(monkeypatch):
    bench = _load_tlx_gfx9_a4w4_bench_module("_tlx_wave_test_gfx9_a4w4_bench_median")
    call = {}

    def do_bench(fn, **kwargs):
        call["fn"] = fn
        call["kwargs"] = kwargs
        return 0.75

    monkeypatch.setattr(bench.triton.testing, "do_bench", do_bench)
    fn = lambda: None
    ms = bench.measure_matmul(
        SimpleNamespace(timing_mode="triton", warmup=13, rep=29),
        fn,
    )

    assert ms == 0.75
    assert call == {
        "fn": fn,
        "kwargs": {"warmup": 13, "rep": 29, "return_mode": "median"},
    }


def test_tlx_glu_precompile_reuses_variant_launch_configuration(monkeypatch):
    pytest.importorskip("torch")
    bench = _load_tlx_glu_bench_module("_tlx_wave_test_glu_bench_precompile")
    calls = {}
    kernels = {
        "tlx_baseline": bench.tlx_addmm_glu_kernel_baseline,
        "tlx_simple_async": bench.tlx_addmm_glu_kernel_simple_async,
        "tlx_optimized_async": bench.tlx_addmm_glu_kernel_optimized_async,
        "tlx_optimized": bench.tlx_addmm_glu_kernel_optimized,
        "tlx_persistent": bench.tlx_addmm_glu_kernel_persistent,
    }

    for kernel_name, kernel in kernels.items():

        def warmup(*args, _kernel_name=kernel_name, **kwargs):
            calls[_kernel_name] = (args, kwargs)

        monkeypatch.setattr(kernel, "warmup", warmup)

    shape = (1024, 21568, 256)
    for kernel_name in kernels:
        bench.compile_kernel_shape(kernel_name, shape, num_cus=304)

    assert set(calls) == set(kernels)
    for args, kwargs in calls.values():
        assert all(isinstance(arg, bench.MockTensor) for arg in args[:5])
        assert args[5:8] == shape
        assert args[8:] == (256, 1, 21568, 1, 21568, 1, 21568, 1)
        assert kwargs["grid"]
    assert calls["tlx_baseline"][1]["NUM_STAGES"] == 2
    assert calls["tlx_optimized_async"][1]["NUM_BUFFERS"] == 4
    assert calls["tlx_persistent"][1]["NUM_CUS"] == 304
    assert 1 <= bench.DEFAULT_COMPILE_WORKERS <= 8


def test_tlx_fa_precompile_reuses_runtime_launch_configuration(monkeypatch):
    pytest.importorskip("torch")
    bench = _load_tlx_fa_bench_module("_tlx_wave_test_fa_bench_precompile")
    calls = {}
    kernels = {
        "async_simple": bench._attn_fwd_async_simple,
        "async_prefetch": bench._attn_fwd_async_prefetch,
        "persistent": bench._attn_fwd_persistent,
        "cluster": bench._attn_fwd_cluster_pipeline,
    }

    for kernel_name, kernel in kernels.items():

        def warmup(*args, _kernel_name=kernel_name, **kwargs):
            calls[_kernel_name] = (args, kwargs)

        monkeypatch.setattr(kernel, "warmup", warmup)

    jobs = [
        ("async_simple", 1, 64, 4096, 64, False, "bf16"),
        ("async_prefetch", 1, 64, 4096, 64, False, "bf16"),
        ("persistent", 2, 64, 8192, 128, True, "bf16"),
        ("cluster", 1, 64, 4096, 128, True, "bf16"),
    ]
    for job in jobs:
        bench.compile_kernel_config(job, num_sms=304)

    assert set(calls) == set(kernels)
    for args, kwargs in calls.values():
        assert all(isinstance(arg, MockTensor) for arg in args[:4])
        assert kwargs["grid"]

    simple_args, simple_kwargs = calls["async_simple"]
    assert simple_args[20:23] == (1, 64, 4096)
    assert simple_kwargs["grid"] == (16, 64)
    assert simple_kwargs["BLOCK_N"] == 64
    assert simple_kwargs["HEAD_DIM"] == 64
    assert simple_kwargs["IS_CAUSAL"] is False

    prefetch_kwargs = calls["async_prefetch"][1]
    assert prefetch_kwargs["BLOCK_N"] == 128

    persistent_args, persistent_kwargs = calls["persistent"]
    assert persistent_args[20:25] == (2, 64, 8192, 8192, 0)
    assert persistent_kwargs["grid"] == (304, )
    assert persistent_kwargs["BLOCK_N"] == 64
    assert persistent_kwargs["NUM_M_BLOCKS"] == 32
    assert persistent_kwargs["NUM_SMS"] == 304
    assert persistent_kwargs["IS_CAUSAL"] is True

    cluster_args, cluster_kwargs = calls["cluster"]
    assert cluster_args[20] == 4096
    assert cluster_kwargs["grid"] == (64, 16, 1)
    assert cluster_kwargs["BLOCK_N"] == 32
    assert cluster_kwargs["num_warps"] == 8
    assert cluster_kwargs["waves_per_eu"] == 0

    perf_jobs = bench.compilation_jobs(SimpleNamespace(mode="perf_test"))
    assert len(perf_jobs) == len(bench.PERF_BASELINE_TFLOPS) == 42
    assert 1 <= bench.DEFAULT_COMPILE_WORKERS <= 8


def test_tlx_glu_parallel_precompile_defers_device_query_to_worker(monkeypatch):
    pytest.importorskip("torch")
    bench = _load_tlx_glu_bench_module("_tlx_wave_test_glu_bench_worker_device")

    class PoolCreated(Exception):
        pass

    def fail_device_query(*_args, **_kwargs):
        pytest.fail("parallel precompile initialized HIP in the parent process")

    def create_pool(*_args, **_kwargs):
        raise PoolCreated

    monkeypatch.setattr(bench.torch.cuda, "get_device_properties", fail_device_query)
    monkeypatch.setattr(bench.concurrent.futures, "ProcessPoolExecutor", create_pool)
    args = SimpleNamespace(
        mode="benchmark",
        M=[1024],
        N=[21568],
        K=[256],
        kernel=["tlx_baseline", "tlx_persistent"],
        compile_workers=2,
    )
    with pytest.raises(PoolCreated):
        bench.precompile_kernels(args)

    queried_devices = []
    compile_calls = []

    def get_device_properties(device):
        queried_devices.append(device)
        return SimpleNamespace(multi_processor_count=304)

    monkeypatch.setattr(bench, "active_torch_device", lambda: "worker-device")
    monkeypatch.setattr(bench.torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(
        bench,
        "compile_kernel_shape",
        lambda kernel_name, shape, num_cus: compile_calls.append((kernel_name, shape, num_cus)),
    )
    bench._device_num_cus.cache_clear()
    shape = (1024, 21568, 256)
    bench.compile_kernel_shape_worker("tlx_persistent", shape, None)
    bench.compile_kernel_shape_worker("tlx_persistent", shape, None)

    assert queried_devices == ["worker-device"]
    assert compile_calls == [
        ("tlx_persistent", shape, 304),
        ("tlx_persistent", shape, 304),
    ]


def test_tlx_fa_parallel_precompile_defers_device_query_to_worker(monkeypatch):
    pytest.importorskip("torch")
    bench = _load_tlx_fa_bench_module("_tlx_wave_test_fa_bench_worker_device")

    class PoolCreated(Exception):
        pass

    def fail_device_query(*_args, **_kwargs):
        pytest.fail("parallel precompile initialized HIP in the parent process")

    def create_pool(*_args, **_kwargs):
        raise PoolCreated

    monkeypatch.setattr(bench.torch.cuda, "get_device_properties", fail_device_query)
    monkeypatch.setattr(bench.concurrent.futures, "ProcessPoolExecutor", create_pool)
    args = SimpleNamespace(
        mode="benchmark",
        kernel=["async_simple", "persistent"],
        b=[1],
        hq=[64],
        d=[128],
        sq=[4096],
        causal=["false"],
        dtype="bf16",
        compile_workers=2,
    )
    with pytest.raises(PoolCreated):
        bench.precompile_kernels(args)

    queried_devices = []
    compile_calls = []

    def get_device_properties(device):
        queried_devices.append(device)
        return SimpleNamespace(multi_processor_count=306)

    monkeypatch.setattr(bench, "active_torch_device", lambda: "worker-device")
    monkeypatch.setattr(bench.torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(
        bench,
        "compile_kernel_config",
        lambda job, num_sms: compile_calls.append((job, num_sms)),
    )
    bench._device_num_sms.cache_clear()
    job = ("persistent", 1, 64, 4096, 128, False, "bf16")
    assert bench.compile_kernel_config_worker(job, None)[2] is None
    assert bench.compile_kernel_config_worker(job, None)[2] is None

    assert queried_devices == ["worker-device"]
    assert compile_calls == [(job, 304), (job, 304)]


def test_tlx_perf_sweep_forwards_compile_workers_to_fa(tmp_path):
    runner = _load_tlx_perf_sweep_module()
    args = SimpleNamespace(
        sweeps=["fa"],
        rep=1,
        warmup=0,
        compile_workers=3,
        wave_split_barriers=False,
    )

    specs = runner.build_run_specs(args, tmp_path)

    assert [spec.name for spec in specs] == [
        "fa-llvm",
        "fa-wave",
        "fa-eight-wave-llvm",
        "fa-eight-wave-wave",
    ]
    assert [spec.backend for spec in specs] == ["llvm", "wave", "llvm", "wave"]
    assert all(spec.command[-2:] == ("--compile-workers", "3") for spec in specs[:2])
    for spec, minimum in zip(specs[2:], ("0", "1000")):
        assert spec.command[-6:] == (
            "--rep",
            "1",
            "--warmup",
            "0",
            "--min-tflops",
            minimum,
        )


def test_tlx_perf_sweep_forwards_f16_input_and_timing_options(tmp_path):
    runner = _load_tlx_perf_sweep_module("_tlx_wave_test_perf_sweep_f16_options")
    args = SimpleNamespace(
        sweeps=["f16"],
        rep=31,
        warmup=7,
        compile_workers=2,
        f16_input_mode="rand-int",
        f16_input_seed=5,
        f16_timing_mode="batched",
        f16_warmup_launches=11,
        f16_timed_launches=101,
        f16_timing_repeats=3,
        wave_split_barriers=False,
    )

    v9_spec, v10_spec, v11_spec, inter_wave_llvm_spec, inter_wave_wave_spec = runner.build_run_specs(args, tmp_path)
    assert v9_spec.name == "f16"
    assert v10_spec.name == "f16-v10"
    command = v9_spec.command

    expected_options = {
        "--rep": "31",
        "--warmup": "7",
        "--compile-workers": "2",
        "--input-mode": "rand-int",
        "--seed": "5",
        "--timing-mode": "batched",
        "--warmup-launches": "11",
        "--timed-launches": "101",
        "--timing-repeats": "3",
    }
    for option, expected in expected_options.items():
        assert command[command.index(option) + 1] == expected

    v10_command = v10_spec.command
    v10_expected_options = {
        "--version": "10",
        "--shape": "8192x8192x8192",
        "--rep": "31",
        "--warmup": "7",
        "--compile-workers": "2",
        "--input-mode": "rand-int",
        "--seed": "5",
        "--timing-mode": "batched",
        "--warmup-launches": "11",
        "--timed-launches": "101",
        "--timing-repeats": "3",
    }
    for option, expected in v10_expected_options.items():
        assert v10_command[v10_command.index(option) + 1] == expected
    providers_index = v10_command.index("--providers")
    assert v10_command[providers_index + 1:providers_index + 3] == ("tlx", "wave")

    assert v11_spec.name == "f16-v11"
    v11_command = v11_spec.command
    v11_expected_options = {
        "--version": "11",
        "--shape": "8192x8192x8192",
        "--rep": "31",
        "--warmup": "7",
        "--compile-workers": "2",
        "--input-mode": "rand-int",
        "--seed": "5",
        "--timing-mode": "batched",
        "--warmup-launches": "11",
        "--timed-launches": "101",
        "--timing-repeats": "3",
    }
    for option, expected in v11_expected_options.items():
        assert v11_command[v11_command.index(option) + 1] == expected
    providers_index = v11_command.index("--providers")
    assert v11_command[providers_index + 1:providers_index + 3] == ("tlx", "wave")

    assert inter_wave_llvm_spec.name == "f16-inter-wave-llvm"
    assert inter_wave_llvm_spec.backend == "llvm"
    assert inter_wave_wave_spec.name == "f16-inter-wave-wave"
    assert inter_wave_wave_spec.backend == "wave"
    for spec in (inter_wave_llvm_spec, inter_wave_wave_spec):
        assert spec.command[1].endswith("gfx9_gemm/inter_wave/a16w16/bench.py")
        shape_index = spec.command.index("--shape")
        assert spec.command[shape_index + 1:shape_index + 4] == ("8192", "8192", "8192")
        assert spec.command[spec.command.index("--rep") + 1] == "31"
        assert spec.command[spec.command.index("--warmup") + 1] == "7"
        assert spec.command[spec.command.index("--input-mode") + 1] == "rand-int"
        assert spec.command[spec.command.index("--seed") + 1] == "5"


def test_tlx_perf_sweep_forwards_mxfp_batched_timing_options(tmp_path):
    runner = _load_tlx_perf_sweep_module("_tlx_wave_test_perf_sweep_mxfp_options")
    args = SimpleNamespace(
        sweeps=["mxfp"],
        rep=31,
        warmup=7,
        compile_workers=2,
        mxfp_timing_mode="batched",
        mxfp_warmup_launches=13,
        mxfp_timed_launches=103,
        mxfp_timing_repeats=5,
        wave_split_barriers=False,
    )

    specs = runner.build_run_specs(args, tmp_path)

    assert [spec.backend for spec in specs] == ["llvm", "wave", "llvm", "wave"]
    expected_options = {
        "--rep": "31",
        "--warmup": "7",
        "--timing-mode": "batched",
        "--warmup-launches": "13",
        "--timed-launches": "103",
        "--timing-repeats": "5",
    }
    for spec in specs:
        for option, expected in expected_options.items():
            assert spec.command[spec.command.index(option) + 1] == expected

    intra_wave_llvm, intra_wave_wave, inter_wave_llvm, inter_wave_wave = specs
    assert intra_wave_llvm.name == "mxfp-llvm"
    assert intra_wave_wave.name == "mxfp-wave"
    for spec in (intra_wave_llvm, intra_wave_wave):
        assert spec.command[1].endswith("gfx9_gemm/intra_wave/a4w4/bench.py")
        assert spec.command[spec.command.index("--compile-workers") + 1] == "2"

    assert inter_wave_llvm.name == "mxfp-inter-wave-llvm"
    assert inter_wave_wave.name == "mxfp-inter-wave-wave"
    for spec in (inter_wave_llvm, inter_wave_wave):
        assert spec.command[1].endswith("gfx9_gemm/inter_wave/a4w4/bench.py")
        assert "--compile-workers" not in spec.command


def test_tlx_wave_gfx9_a4w4_scale_loads_keep_requested_packet_layouts(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    bench = _load_tlx_gfx9_a4w4_bench_module("_tlx_wave_test_gfx9_a4w4_bench_scale_loads")
    cache_dir = tmp_path / "a4w4-scale-load-cache"

    with (
            _tlx_wave_compile_driver(monkeypatch),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(cache_dir)
        triton.knobs.runtime.override_arch = "gfx950"
        bench.compile_shape((256, 256, 1024), cache_dir)

    ttgir = next(cache_dir.rglob("_a4w4_kernel.ttgir")).read_text()
    wave = next(cache_dir.rglob("_a4w4_kernel.wave")).read_text(errors="ignore")

    a_load = re.search(
        r"amdg\.buffer_load %a_scales_ptr\[[^\]]+\] "
        r"\{tlx\.layout_is_explicit\} : tensor<256x8xi8, (#[^>]+)>",
        ttgir,
    )
    b_load = re.search(
        r"amdg\.buffer_load %b_scales_ptr\[[^\]]+\] "
        r"\{tlx\.layout_is_explicit\} : tensor<128x8xi8, (#[^>]+)>",
        ttgir,
    )
    assert a_load is not None
    assert b_load is not None
    assert re.search(
        rf"{re.escape(a_load.group(1))} = #ttg\.linear<\{{register = "
        r"\[\[1, 0\], \[2, 0\], \[4, 0\]\],",
        ttgir,
    )
    assert re.search(
        rf"{re.escape(b_load.group(1))} = #ttg\.linear<\{{register = "
        r"\[\[1, 0\], \[2, 0\]\],",
        ttgir,
    )

    global_i8_scalar_load = (r"wave\.gather .* : \(!wave\.ptr<#waveamd\.buffer, i8>.*-> "
                             r"\(!wave\.simd<vector<1xi8>, 64>")
    global_i8x8_load = (r"wave\.gather .* : \(!wave\.ptr<#waveamd\.buffer, i8>.*-> "
                        r"\(!wave\.simd<vector<8xi8>, 64>")
    global_i8x4_load = (r"wave\.gather .* : \(!wave\.ptr<#waveamd\.buffer, i8>.*-> "
                        r"\(!wave\.simd<vector<4xi8>, 64>")
    assert re.search(global_i8_scalar_load, wave) is None
    assert re.search(global_i8x8_load, wave) is not None
    assert re.search(global_i8x4_load, wave) is not None


def _compile_tlx_gfx9_gemm_kernel(tmp_path, monkeypatch, case):
    torch = pytest.importorskip("torch")
    if case.get("disables_post_misched", False):
        monkeypatch.setenv("TRITON_DISABLE_POST_MISCHED", "1")
    kernel = _load_tlx_gfx9_gemm_kernel(case["version_dir"], case["function_name"])

    m, n, k = case.get("shape", (256, 256, 256))
    a = MockTensor(torch.float16, [m, k])
    b = MockTensor(torch.float16, [k, n])
    c = MockTensor(torch.float16, [m, n])
    a_strides = a.stride()
    b_strides = case.get("b_strides", b.stride())
    c_strides = c.stride()

    with (
            _tlx_wave_compile_driver(monkeypatch),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / f"{case['version_dir']}-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        return kernel.warmup(
            a,
            b,
            c,
            m,
            n,
            k,
            a_strides[0],
            a_strides[1],
            b_strides[0],
            b_strides[1],
            c_strides[0],
            c_strides[1],
            BLOCK_M=256,
            BLOCK_N=256,
            BLOCK_K=64,
            num_warps=case["num_warps"],
            num_stages=1,
            matrix_instr_nonkdim=16,
            grid=case.get("grid", (case.get("extra_meta", {}).get("GRID_MN", 1), )),
            **case.get("compile_options", {}),
            **case.get("extra_meta", {}),
        )


def test_tlx_wave_converter_import_stage_boundary_is_static():
    package_root = (Path(__file__).resolve().parents[4] / "third_party" / "tlx_wave" / "backend" / "converter")
    forbidden_prefixes = (
        "triton.backends.tlx_wave.wave_bridge",
        "third_party.tlx_wave.backend.wave_bridge",
    )
    for path in package_root.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
        assert not any(module.startswith(forbidden_prefixes) for module in imports), (path, imports)

    assert "wave_bridge" not in converter_source_ir.__dict__
    assert "wave_bridge" not in converter_diagnostics.__dict__
    assert "wave_bridge" not in converter_domains.__dict__
    assert "wave_bridge" not in converter_canonicalize.__dict__
    assert "wave_bridge" not in converter_source_import.__dict__
    assert "wave_bridge" not in converter_tokens.__dict__
    assert "wave_bridge" not in converter_target_ir.__dict__
    assert "wave_bridge" not in converter_op_conversion.__dict__
    assert "wave_bridge" not in converter_verifier.__dict__
    assert "wave_bridge" not in converter_emission.__dict__
    assert "wave_bridge" not in converter_pipeline.__dict__


def test_tlx_wave_converter_lowering_domains_are_pure_policy():
    tree = ast.parse(
        Path(converter_domains.__file__).read_text(),
        filename=converter_domains.__file__,
    )
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)

    assert imports == ["dataclasses"]


def test_tlx_wave_converter_lowering_domains_cover_dispatch():
    assert converter_domains.DOMAIN_NAMES == (
        "arithmetic_control",
        "memory_dma",
        "generic_memory",
        "local_memory_layout",
        "mfma_fragment",
        "store_epilogue",
    )
    assert converter_domains.source_domains_for_op("arith.addi") == ("arithmetic_control", )
    assert converter_domains.source_domains_for_op("arith.constant") == (
        "arithmetic_control",
        "mfma_fragment",
    )
    assert converter_domains.source_domains_for_op("amdg.buffer_load_to_local") == ("memory_dma", )
    assert converter_domains.source_domains_for_op("tt.load") == ("generic_memory", )
    assert converter_domains.source_domains_for_op("tt.store") == ("generic_memory", )
    assert converter_domains.source_domains_for_op("rocdl.sched.barrier") == ("arithmetic_control", )
    assert converter_domains.target_domain_for_op("local_load") == ("local_memory_layout")
    assert converter_domains.target_domain_for_op("mma") == "mfma_fragment"
    assert converter_domains.target_domain_for_op("load") == "generic_memory"
    assert converter_domains.target_domain_for_op("store") == "generic_memory"
    assert converter_domains.target_domain_for_op("buffer_store") == "store_epilogue"
    assert (converter_op_conversion._SUPPORTED_SOURCE_OPS == converter_domains.all_source_ops())
    assert set(converter_emission._TARGET_EMITTERS) == (converter_domains.all_target_ops())


def test_tlx_wave_converter_op_rewriters_do_not_accept_source_program():
    tree = ast.parse(
        Path(converter_op_conversion.__file__).read_text(),
        filename=converter_op_conversion.__file__,
    )
    allowed_source_program_functions = {
        "convert_ops",
        "_build_conversion_input",
        "_wait_publication_barrier_by_op",
        "_memdesc_infos",
        "_constant_ints",
    }
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in allowed_source_program_functions:
            continue
        arg_names = [arg.arg for arg in node.args.args]
        if "source_program" in arg_names:
            offenders.append(f"{node.name}:argument")
        if any(isinstance(child, ast.Name) and child.id == "source_program" for child in ast.walk(node)):
            offenders.append(f"{node.name}:body")

    assert not offenders


def test_tlx_wave_converter_import_stage_builds_source_snapshot(tmp_path):
    local_func = """
  tt.func public @converter_import(
      %arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32, tt.divisibility = 16 : i32},
      %arg1: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %positive = arith.cmpi sgt, %arg1, %zero : i32
    %value = scf.if %positive -> (i32) {
      %one = arith.constant 1 : i32
      scf.yield %one : i32
    } else {
      %two = arith.constant 2 : i32
      scf.yield %two : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=2)

    program = converter_source_import.import_source_program(mod)

    assert isinstance(program, converter_source_ir.SourceProgram)
    assert program.kernel.name == "converter_import"
    assert program.kernel.num_warps == 2
    assert len(program.kernel.arg_ids) == 2
    pointer_arg = program.values[program.kernel.arg_ids[0]]
    assert pointer_arg.type.kind == "pointer"
    assert pointer_arg.type.pointer_range == 32
    assert pointer_arg.type.divisibility == 16

    if_op = next(op for op in program.ops if op.name == "scf.if")
    assert len(if_op.region_ids) == 2
    assert all(program.regions[region_id].parent_op_index == if_op.index for region_id in if_op.region_ids)
    assert [program.ops[index].name
            for index in program.regions[if_op.region_ids[0]].op_indices] == ["arith.constant", "scf.yield"]
    assert [program.ops[index].name
            for index in program.regions[if_op.region_ids[1]].op_indices] == ["arith.constant", "scf.yield"]
    assert all(program.values[result_id].owner_op_index == op.index for op in program.ops for result_id in op.results)
    assert not any(hasattr(value, "users") for value in program.values.values())
    del ctx


def test_tlx_wave_converter_import_stage_reports_structured_diagnostics(tmp_path):
    local_func = """
  tt.func private @not_public() attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_source_import.import_source_program(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_IMPORT_KERNEL_COUNT"
    assert diagnostic.stage == "import"
    assert diagnostic.no_fallback is True
    assert "no_fallback" in str(diagnostic)
    del ctx


def test_tlx_wave_converter_type_layout_stage_converts_source_snapshot(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_type_layout(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<128xi1, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)

    converted = converter_types.convert_source_program(source)

    range_op = next(op for op in source.ops if op.name == "tt.make_range")
    range_value = converted.values[range_op.results[0]]
    range_layout = converted.layouts[range_value.layout_map_id]
    assert range_value.type.representation == "simd_tuple"
    assert range_value.type.component_count == 2
    assert range_layout.kind == "blocked"
    assert range_layout.properties["size_per_thread"] == (2, )

    mask_op = next(op for op in source.ops
                   if op.name == "tt.splat" and source.values[op.results[0]].type.element_type == "i1")
    assert converted.values[mask_op.results[0]].type.representation == "mask_tuple"

    addptr_op = next(op for op in source.ops if op.name == "tt.addptr")
    assert converted.values[addptr_op.results[0]].type.representation == "pointer_tuple"
    assert not hasattr(converted, "target_ops")
    del ctx


def test_tlx_wave_converter_type_layout_reuses_equivalent_layout_analysis(
    tmp_path,
    monkeypatch,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
"""
    local_func = """
  tt.func public @converter_repeated_layouts() attributes {noinline = false} {
    %first = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %second = arith.constant dense<1> : tensor<8x8xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )
    source = converter_source_import.import_source_program(mod)
    original = converter_types.build_layout_map
    calls = 0

    def counted_build_layout_map(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        converter_types,
        "build_layout_map",
        counted_build_layout_map,
    )
    converted = converter_types.convert_source_program(source)

    assert calls == 1
    assert len(converted.layouts) == 2
    assert converted.layouts[0].layout_map_id == 0
    assert converted.layouts[1].layout_map_id == 1
    assert converted.layouts[0].value_id != converted.layouts[1].value_id
    assert converted.layouts[0].properties == converted.layouts[1].properties
    del ctx


def test_tlx_wave_converter_type_layout_unwraps_tlx_layout_markers(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [[1]], lane = [[2], [4], [8], [16], [32], [64]], warp = [], block = []}>
#user_no_verify = #tlx.user_layout<#tlx.no_verify_layout<#linear>>
#no_verify_user = #tlx.no_verify_layout<#tlx.user_layout<#linear>>
"""
    local_func = """
  tt.func public @converter_wrapped_layouts() attributes {noinline = false} {
    %a = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #user_no_verify>
    %b = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #no_verify_user>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)

    converted = converter_types.convert_source_program(source)

    ranges = [op for op in source.ops if op.name == "tt.make_range"]
    assert len(ranges) == 2
    for range_op in ranges:
        value = converted.values[range_op.results[0]]
        layout = converted.layouts[value.layout_map_id]
        assert layout.kind == "linear"
        assert layout.properties["register_bases"] == ((1, ), )
        assert layout.properties["lane_bases"] == ((2, ), (4, ), (8, ), (16, ), (32, ), (64, ))
        assert value.type.representation == "simd_tuple"
        assert value.type.component_count == 2
    del ctx


def test_tlx_wave_converter_type_layout_stage_rejects_unknown_encoding():

    class UnknownEncoding:
        pass

    source_type = converter_source_ir.SourceType(
        "tensor<4xf32, #unknown>",
        "tensor",
        shape=(4, ),
        element_type="f32",
        encoding_attr=UnknownEncoding(),
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("bad_layout", threads_per_warp=64),
        (),
        {1: converter_source_ir.SourceValue(1, source_type)},
        (converter_source_ir.SourceRegion(0, ()), ),
        0,
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_types.convert_source_program(program)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_TYPE_UNSUPPORTED_LAYOUT"
    assert diagnostic.stage == "type_layout"
    assert diagnostic.source_value_id == 1
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_type_layout_stage_supports_slice_encoding(tmp_path):
    preamble = """
#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [8, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
"""
    local_func = """
  tt.func public @converter_slice_layout() attributes {noinline = false} {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #slice>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)
    source = converter_source_import.import_source_program(mod)

    converted = converter_types.convert_source_program(source)

    range_op = next(op for op in source.ops if op.name == "tt.make_range")
    range_value = converted.values[range_op.results[0]]
    layout = converted.layouts[range_value.layout_map_id]
    assert layout.kind == "slice"
    assert layout.properties["dim"] == 1
    assert layout.properties["parent_kind"] == "blocked"
    assert layout.properties["parent_properties"]["warps_per_cta"] == (8, 1)
    del ctx


def test_tlx_wave_converter_type_layout_stage_supports_nested_slice_encoding(tmp_path):
    preamble = """
#parent = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 8, 8], warpsPerCTA = [1, 1, 1, 1], order = [3, 2, 1, 0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #parent}>
#slice1 = #ttg.slice<{dim = 0, parent = #slice0}>
"""
    local_func = """
  tt.func public @converter_nested_slice_layout() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<8x8xi32, #slice1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)

    converted = converter_types.convert_source_program(source)

    constant_op = next(op for op in source.ops if op.name == "arith.constant")
    value = converted.values[constant_op.results[0]]
    layout = converted.layouts[value.layout_map_id]
    linear = converter_layouts.distributed_linear_layout(layout)
    assert value.type.component_count == 1
    assert layout.properties["parent_kind"] == "slice"
    assert converter_layouts.linear_layout_coords(linear, 0, 9, warp=0) == (1, 1)
    del ctx


def test_tlx_wave_converter_make_range_uses_slice_coordinates(tmp_path):
    preamble = """
#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #parent}>
"""
    local_func = """
  tt.func public @converter_slice_make_range() attributes {noinline = false} {
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    fact_program = converter_facts.analyze_facts(source, converted)
    token_program = converter_tokens.build_token_program(source, converted)

    target = converter_op_conversion.convert_ops(
        source,
        converted,
        fact_program,
        token_program,
    )

    (range_op, ) = [op for op in target.ops if op.kind == "make_range"]
    attrs = converter_target_ir.attrs_dict(range_op)
    assert attrs["coordinate_mode"] == "bit_affine_workitem"
    assert attrs["component_bases"] == (0, )
    assert attrs["workitem_coefficients"] == (1, 2, 4, 8, 16, 0)
    del ctx


def test_tlx_wave_converter_fact_stage_extracts_provenance_facts(tmp_path):
    local_func = """
  tt.func public @converter_facts(
      %arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %nonnegative = arith.cmpi sge, %stride, %zero : i32
    llvm.intr.assume %nonnegative : i1
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    fact_program = converter_facts.analyze_facts(source, converted)

    pointer_arg_id, stride_id = source.kernel.arg_ids
    pointer_facts = converter_facts.facts_for_value(fact_program, pointer_arg_id)
    assert any(fact.kind == "pointer_byte_range" and fact.lower == 0 and fact.upper == (1 << 31) -
               1 and fact.width == 32 and fact.signedness == "signed" and fact.provenance == "arg:tt.pointer_range"
               for fact in pointer_facts)

    stride_facts = converter_facts.facts_for_value(fact_program, stride_id)
    assert any(fact.kind == "range" and fact.predicate == "signed_width" and fact.lower == -(1 << 31) and fact.upper ==
               (1 << 31) - 1 and fact.width == 32 and fact.provenance == "type:i32" for fact in stride_facts)
    assume_op = next(op for op in source.ops if op.name == "llvm.intr.assume")
    assert any(
        fact.kind == "range" and fact.predicate == "sge" and fact.lower == 0 and fact.upper is None and fact.width == 32
        and fact.provenance == "llvm.intr.assume" and fact.source_op_index == assume_op.index for fact in stride_facts)
    assert not hasattr(fact_program, "target_ops")
    del ctx


def test_tlx_wave_converter_fact_stage_does_not_infer_overflowing_mul(tmp_path):
    local_func = """
  tt.func public @converter_no_mul_fact(%x: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %x_nonnegative = arith.cmpi sge, %x, %zero : i32
    llvm.intr.assume %x_nonnegative : i1
    %prod = arith.muli %x, %x : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    fact_program = converter_facts.analyze_facts(source, converted)

    product_op = next(op for op in source.ops if op.name == "arith.muli")
    product_facts = converter_facts.facts_for_value(fact_program, product_op.results[0])
    assert not any(
        fact.kind == "range" and fact.predicate != "signed_width" and fact.lower is not None and fact.lower >= 0
        for fact in product_facts)
    del ctx


def test_tlx_wave_converter_preserves_nonnegative_signed_div_rem(tmp_path):
    local_func = """
  tt.func public @converter_nonnegative_div_rem(%limit: i32) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %limit_positive = arith.cmpi sgt, %limit, %c0 : i32
    llvm.intr.assume %limit_positive : i1
    %first_rem = arith.remsi %pid, %c8 : i32
    %first_div = arith.divsi %pid, %c8 : i32
    %left = arith.addi %first_rem, %first_div : i32
    %right = arith.addi %first_div, %c4 : i32
    %use_left = arith.cmpi slt, %first_rem, %c4 : i32
    %swizzled = scf.if %use_left -> (i32) {
      scf.yield %left : i32
    } else {
      scf.yield %right : i32
    }
    %group_positive = arith.cmpi sgt, %c1, %c0 : i32
    llvm.intr.assume %group_positive : i1
    %inner = arith.remsi %swizzled, %c1 : i32
    %tile = arith.divsi %swizzled, %c1 : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    operations = [
        converter_target_ir.attrs_dict(op)["operation"] for op in output.target_program.ops if op.kind == "binary"
    ]
    assert "divsi" in operations
    assert "remsi" in operations
    assert "divui" not in operations
    assert "remui" not in operations
    del ctx


def test_tlx_wave_converter_preserves_make_range_signed_div_rem(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_make_range_div_rem_unsigned() attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %divisor = arith.constant dense<16> : tensor<64xi32, #blocked>
    %remainder = arith.remsi %range, %divisor : tensor<64xi32, #blocked>
    %quotient = arith.divsi %range, %divisor : tensor<64xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    operations = [
        converter_target_ir.attrs_dict(op)["operation"] for op in output.target_program.ops if op.kind == "binary"
    ]
    assert operations == ["remsi", "divsi"]
    assert "divui" not in operations
    assert "remui" not in operations
    _run_wave_verify(output.emitted_module.text)
    _run_waveamd_to_machine(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_preserves_shared_dynamic_divisor(tmp_path):
    local_func = """
  tt.func public @converter_exact_symbolic_divisor(%limit: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %limit_nonnegative = arith.cmpi sge, %limit, %c0 : i32
    llvm.intr.assume %limit_nonnegative : i1
    %bounded = arith.minsi %limit, %c4 : i32
    %bounded_positive = arith.cmpi sgt, %bounded, %c0 : i32
    llvm.intr.assume %bounded_positive : i1
    %bounded_at_least_two = arith.cmpi sge, %bounded, %c2 : i32
    llvm.intr.assume %bounded_at_least_two : i1
    %bounded_at_most_two = arith.cmpi sle, %bounded, %c2 : i32
    llvm.intr.assume %bounded_at_most_two : i1
    %pid = tt.get_program_id x : i32
    %remainder = arith.remsi %pid, %bounded : i32
    %quotient = arith.divsi %pid, %bounded : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    binary_ops = [op for op in output.target_program.ops if op.kind == "binary"]
    div_rem_ops = [op for op in binary_ops if converter_target_ir.attrs_dict(op)["operation"] in {"divsi", "remsi"}]
    assert [converter_target_ir.attrs_dict(op)["operation"] for op in div_rem_ops] == ["remsi", "divsi"]
    assert div_rem_ops[0].operands == div_rem_ops[1].operands

    producer_by_result = {result: op for op in output.target_program.ops for result in op.results}
    divisor = producer_by_result[div_rem_ops[0].operands[1]]
    assert divisor.kind != "constant"
    _run_wave_verify(output.emitted_module.text)
    _run_waveamd_to_machine(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_preserves_nonnegative_for_iv_signed_div_rem(tmp_path):
    local_func = """
  tt.func public @converter_for_iv_rem_unsigned(%upper: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %sum = scf.for %i = %c0 to %upper step %c1 iter_args(%acc = %c0) -> (i32)  : i32 {
      %r = arith.remsi %i, %c4 : i32
      %q = arith.divsi %i, %c4 : i32
      %next0 = arith.addi %acc, %r : i32
      %next1 = arith.addi %next0, %q : i32
      scf.yield %next1 : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    fact_program = converter_facts.analyze_facts(source, converted)

    for_op = next(op for op in source.ops if op.name == "scf.for")
    induction_arg = source.regions[for_op.region_ids[0]].block_arg_ids[0]
    assert any(fact.kind == "range" and fact.lower == 0 and fact.provenance == "derived:scf.for"
               for fact in converter_facts.facts_for_value(fact_program, induction_arg))

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    operations = [
        converter_target_ir.attrs_dict(op)["operation"] for op in output.target_program.ops if op.kind == "binary"
    ]
    assert "remsi" in operations
    assert "divsi" in operations
    assert "remui" not in operations
    assert "divui" not in operations
    del ctx


@pytest.mark.parametrize(("lower", "step", "divisor"), ((0, 16, 16), (8, 4, 4), (6, 4, 2)))
def test_tlx_wave_converter_passes_loop_induction_facts_to_wave(tmp_path, lower, step, divisor):
    local_func = f"""
  tt.func public @converter_for_iv_facts(%upper: i32) attributes {{noinline = false}} {{
    %lower = arith.constant {lower} : i32
    %step = arith.constant {step} : i32
    %sum = scf.for %i = %lower to %upper step %step iter_args(%acc = %lower) -> (i32)  : i32 {{
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }}
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    fact_program = converter_facts.analyze_facts(source, converted)

    for_op = next(op for op in source.ops if op.name == "scf.for")
    induction_arg = source.regions[for_op.region_ids[0]].block_arg_ids[0]
    induction_facts = converter_facts.facts_for_value(fact_program, induction_arg)
    assert any(fact.kind == "divisible" and fact.divisor == divisor and fact.provenance == "derived:scf.for.step"
               for fact in induction_facts)

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    loop_body = output.emitted_module.text.split("scf.for", maxsplit=1)[1]
    assert f'#wave.pred<"Mod(x, {divisor}) == 0">' in loop_body
    assert loop_body.index("wave.assume") < loop_body.index("wave.binary addi")
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_keeps_unproven_signed_div_signed(tmp_path):
    local_func = """
  tt.func public @converter_unproven_signed_div(%arg0: i32) attributes {noinline = false} {
    %one = arith.constant 1 : i32
    %q = arith.divsi %arg0, %one : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    operations = [
        converter_target_ir.attrs_dict(op)["operation"] for op in output.target_program.ops if op.kind == "binary"
    ]
    assert "divsi" in operations
    assert "divui" not in operations
    del ctx


def test_tlx_wave_converter_keeps_branch_assume_out_of_later_div(tmp_path):
    local_func = """
  tt.func public @converter_branch_assume_later_div(%arg0: i32, %cond: i1) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    scf.if %cond {
      %nonnegative = arith.cmpi sge, %arg0, %zero : i32
      llvm.intr.assume %nonnegative : i1
      scf.yield
    } else {
      scf.yield
    }
    %one = arith.constant 1 : i32
    %q = arith.divsi %arg0, %one : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    operations = [
        converter_target_ir.attrs_dict(op)["operation"] for op in output.target_program.ops if op.kind == "binary"
    ]
    assert "divsi" in operations
    assert "divui" not in operations
    del ctx


def test_tlx_wave_converter_keeps_branch_assume_out_of_if_result_range(tmp_path):
    local_func = """
  tt.func public @converter_if_result_assume_scope(%arg0: i32, %cond: i1) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %selected = scf.if %cond -> (i32) {
      %nonnegative = arith.cmpi sge, %arg0, %zero : i32
      llvm.intr.assume %nonnegative : i1
      scf.yield %arg0 : i32
    } else {
      scf.yield %arg0 : i32
    }
    %q = arith.divsi %selected, %one : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    operations = [
        converter_target_ir.attrs_dict(op)["operation"] for op in output.target_program.ops if op.kind == "binary"
    ]
    assert "divsi" in operations
    assert "divui" not in operations
    del ctx


def test_tlx_wave_converter_fact_stage_preserves_ranges_without_affine_graph(tmp_path):
    preamble = """
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_fact_layout_invalidation() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
    %converted = ttg.convert_layout %range : tensor<128xi32, #blocked0> -> tensor<128xi32, #blocked1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    fact_program = converter_facts.analyze_facts(source, converted)

    convert_op = next(op for op in source.ops if op.name == "ttg.convert_layout")
    source_value_id = convert_op.operands[0]
    result_value_id = convert_op.results[0]
    assert converted.values[source_value_id].layout_map_id != converted.values[result_value_id].layout_map_id
    assert not hasattr(fact_program, "tensor_affine")
    source_ranges = [
        fact for fact in converter_facts.facts_for_value(fact_program, source_value_id) if fact.kind == "range"
    ]
    result_ranges = [
        fact for fact in converter_facts.facts_for_value(fact_program, result_value_id) if fact.kind == "range"
    ]
    assert any(fact.lower == 0 and fact.upper == 127 and fact.provenance == "derived:tt.make_range"
               for fact in source_ranges)
    assert any(fact.lower == 0 and fact.upper == 127 and fact.provenance == "derived:ttg.convert_layout"
               for fact in result_ranges)
    del ctx


def test_tlx_wave_converter_token_stage_builds_async_groups_and_effects(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_tokens(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f16>, #blocked>, tensor<64xi32, #blocked>
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<64xi1, #blocked>
    %token = ttg.async_copy_global_to_local %ptr, %alloc mask %mask : tensor<64x!tt.ptr<f16>, #blocked> -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    copy_op = next(op for op in source.ops if op.name == "ttg.async_copy_global_to_local")
    group_op = next(op for op in source.ops if op.name == "ttg.async_commit_group")
    wait_op = next(op for op in source.ops if op.name == "ttg.async_wait")
    copy_node = token_program.node_for_value(copy_op.results[0])
    group_node = token_program.node_for_value(group_op.results[0])
    wait_node = token_program.node_for_value(wait_op.results[0])
    assert copy_node.source_address_value_id == copy_op.operands[0]
    assert copy_node.memdesc_value_id == copy_op.operands[1]
    assert copy_node.mask_value_id == copy_op.operands[2]
    group = token_program.groups[group_node.committed_group_id]
    assert group.member_token_ids == copy_op.results
    assert group.next_same_region_wait_op_index == wait_op.index
    assert wait_node.input_token_ids == group_op.results
    assert token_program.users_for_value(copy_op.results[0]) == (group_node, )
    assert token_program.users_for_value(group_op.results[0]) == (wait_node, )

    assert [(effect.kind, effect.address_space) for effect in token_program.memory_effects] == [
        ("read", "global"),
        ("write", "local"),
    ]
    read_effect, write_effect = token_program.memory_effects
    assert read_effect.address_value_id == copy_op.operands[0]
    assert write_effect.address_value_id == copy_op.operands[1]
    assert read_effect.mask_value_id == write_effect.mask_value_id == copy_op.operands[2]
    assert read_effect.token_node_id == write_effect.token_node_id == copy_node.node_id
    assert write_effect.depends_on_effect_ids == (read_effect.effect_id, )
    assert not hasattr(token_program, "target_ops")
    del ctx


def test_tlx_wave_converter_materializes_implicit_async_group_fifo(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_implicit_async_fifo(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %copy = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group
    %wait = ttg.async_wait {num = 0 : i32}
    %loaded = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<64xf16, #shared, #smem, mutable> -> tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    source_commit = next(op for op in output.source_program.ops if op.name == "ttg.async_commit_group")
    source_wait = next(op for op in output.source_program.ops if op.name == "ttg.async_wait")
    assert not source_commit.operands
    assert not source_wait.operands
    copy_op = next(op for op in output.target_program.ops if op.kind == "buffer_load_to_local")
    commit_op = next(op for op in output.target_program.ops if op.kind == "async_commit_group")
    wait_op = next(op for op in output.target_program.ops if op.kind == "async_wait")
    load_op = next(op for op in output.target_program.ops if op.kind == "local_load")
    assert commit_op.operands[-1:] == copy_op.results[-1:]
    assert wait_op.operands == commit_op.results
    assert load_op.operands[1:] == wait_op.results
    assert converter_target_ir.attrs_dict(load_op)["explicit_dependency_count"] == 1
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_token_stage_records_loop_async_issue_carry(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_token_loop_carry(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_group = ttg.async_commit_group tokens %warmup
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %wait = ttg.async_wait {num = 1 : i32}
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    for_op = next(op for op in source.ops if op.name == "scf.for")
    warmup_group_op, body_group_op = [op for op in source.ops if op.name == "ttg.async_commit_group"]
    _warmup_load_op, _body_load_op = [op for op in source.ops if op.name == "amdg.buffer_load_to_local"]
    (carry, ) = token_program.loop_token_carries_by_op[for_op.index]
    assert carry.loop_op_index == for_op.index
    assert carry.init_source_value_id == warmup_group_op.results[0]
    assert carry.yield_source_value_id == body_group_op.results[0]
    assert carry.add_issue_dependency is False
    assert carry.issue_dependency_op_indices == ()
    del ctx


def test_tlx_wave_converter_token_stage_shifts_async_queue_across_loop(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_token_loop_queue_shift(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup0 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup0_group = ttg.async_commit_group tokens %warmup0
    %warmup1 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup1_group = ttg.async_commit_group tokens %warmup1
    %warmup2 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup2_group = ttg.async_commit_group tokens %warmup2
    %warmup3 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup3_group = ttg.async_commit_group tokens %warmup3
    %prewait = ttg.async_wait {num = 2 : i32}
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %wait = ttg.async_wait {num = 2 : i32}
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    for_op = next(op for op in source.ops if op.name == "scf.for")
    group_ops = [op for op in source.ops if op.name == "ttg.async_commit_group"]
    carries = token_program.loop_token_carries_by_op[for_op.index]
    assert tuple(carry.init_source_value_id for carry in carries) == (
        group_ops[2].results[0],
        group_ops[3].results[0],
    )
    assert tuple(carry.yield_source_value_id for carry in carries) == (
        group_ops[3].results[0],
        group_ops[4].results[0],
    )
    assert tuple(carry.add_issue_dependency for carry in carries) == (False, False)
    assert tuple(carry.issue_dependency_op_indices for carry in carries) == ((), ())

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    (for_target_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    target_region = output.target_program.regions[for_target_op.region_ids[0]]
    target_region_ops = tuple(output.target_program.ops[op_id] for op_id in target_region.op_ids)
    (body_wait_op, ) = [op for op in target_region_ops if op.kind == "async_wait"]
    commit_ops = [op for op in output.target_program.ops if op.kind == "async_commit_group"]
    final_wait_op = [op for op in output.target_program.ops if op.kind == "async_wait"][-1]
    assert converter_target_ir.attrs_dict(for_target_op)["init_arg_count"] == 3
    body_wait_attrs = converter_target_ir.attrs_dict(body_wait_op)
    assert body_wait_attrs["completed_group_dependency_count"] == 2
    assert body_wait_op.operands[:2] == (
        target_region.block_arg_ids[-2],
        commit_ops[2].results[0],
    )
    assert final_wait_op.operands == for_target_op.results[1:]
    _run_wave_verify(output.emitted_module.text)
    del ctx


@pytest.mark.parametrize("warmup_count", (2, 3))
def test_tlx_wave_converter_token_stage_pads_drained_loop_queue_with_neutral_tokens(
    tmp_path,
    warmup_count,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    warmups = "\n".join(f"""
    %warmup{index} = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup{index}_group = ttg.async_commit_group tokens %warmup{index}
""" for index in range(warmup_count))
    local_func = f"""
  tt.func public @converter_token_loop_drains_queue(
      %arg0: !tt.ptr<f16> {{tt.pointer_range = 32 : i32}},
      %arg1: i32) attributes {{noinline = false}} {{
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {{end = 512 : i32, start = 0 : i32}} : tensor<512xi32, #blocked>
{warmups}
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {{
      %wait = ttg.async_wait {{num = 0 : i32}}
      %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }}
    %final_wait = ttg.async_wait {{num = 0 : i32}}
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    source_for_op = next(op for op in source.ops if op.name == "scf.for")
    group_ops = [op for op in source.ops if op.name == "ttg.async_commit_group"]
    carries = token_program.loop_token_carries_by_op[source_for_op.index]
    assert tuple(carry.init_source_value_id for carry in carries) == tuple(op.results[0]
                                                                           for op in group_ops[:warmup_count])
    assert tuple(carry.yield_source_value_id for carry in carries) == (
        group_ops[-1].results[0],
        *((None, ) * (warmup_count - 1)),
    )
    assert all(not carry.add_issue_dependency for carry in carries)

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    (target_for_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    target_region = output.target_program.regions[target_for_op.region_ids[0]]
    target_region_ops = tuple(output.target_program.ops[op_id] for op_id in target_region.op_ids)
    (body_wait_op, ) = [op for op in target_region_ops if op.kind == "async_wait"]
    target_commit_ops = [op for op in output.target_program.ops if op.kind == "async_commit_group"]
    final_wait_op = [op for op in output.target_program.ops if op.kind == "async_wait"][-1]
    assert converter_target_ir.attrs_dict(target_for_op)["init_arg_count"] == 1 + warmup_count
    assert body_wait_op.operands == (
        *target_region.block_arg_ids[-warmup_count:],
        *(op.results[0] for op in target_commit_ops[:warmup_count]),
    )
    assert converter_target_ir.attrs_dict(body_wait_op)["completed_group_dependency_count"] == 2 * warmup_count
    assert final_wait_op.operands == (target_for_op.results[1], )
    assert len([op for op in target_region_ops if op.kind == "token"]) == warmup_count - 1
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_token_stage_merges_conditional_group_before_loop_carry(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_token_conditional_loop_group(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %limit: i32,
      %cond: i1) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_group = ttg.async_commit_group tokens %warmup
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %limit step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %wait = ttg.async_wait {num = 0 : i32}
      scf.if %cond {
        %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
        %body_group = ttg.async_commit_group tokens %body
        scf.yield
      } else {
        scf.yield
      }
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    token_program = converter_tokens.build_token_program(source, converted)

    source_for_op = next(op for op in source.ops if op.name == "scf.for")
    source_if_op = next(op for op in source.ops if op.name == "scf.if")
    warmup_group_op, body_group_op = [op for op in source.ops if op.name == "ttg.async_commit_group"]
    (loop_carry, ) = token_program.loop_token_carries_by_op[source_for_op.index]
    (if_carry, ) = token_program.if_token_carries_by_op[source_if_op.index]
    assert loop_carry.init_source_value_id == warmup_group_op.results[0]
    assert loop_carry.yield_source_value_id == body_group_op.results[0]
    assert if_carry.then_source_value_id == body_group_op.results[0]
    assert if_carry.else_source_value_id is None

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (target_for_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    (target_if_op, ) = [op for op in output.target_program.ops if op.kind == "if"]
    target_wait_ops = [op for op in output.target_program.ops if op.kind == "async_wait"]
    target_for_region = output.target_program.regions[target_for_op.region_ids[0]]
    assert len(target_if_op.results) == 1
    assert target_for_region.yield_value_ids[-1] == target_if_op.results[0]
    target_commit_ops = [op for op in output.target_program.ops if op.kind == "async_commit_group"]
    assert target_wait_ops[0].operands == (
        target_for_region.block_arg_ids[-1],
        target_commit_ops[0].results[0],
    )
    assert converter_target_ir.attrs_dict(target_wait_ops[0])["completed_group_dependency_count"] == 2
    assert target_wait_ops[-1].operands == (target_for_op.results[-1], )
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_token_stage_records_loop_async_final_wait_carry(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_token_loop_final_wait(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    for_op = next(op for op in source.ops if op.name == "scf.for")
    (body_group_op, ) = [op for op in source.ops if op.name == "ttg.async_commit_group"]
    (carry, ) = token_program.loop_token_carries_by_op[for_op.index]
    assert carry.loop_op_index == for_op.index
    assert carry.init_source_value_id is None
    assert carry.yield_source_value_id == body_group_op.results[0]
    assert carry.add_issue_dependency is False
    assert carry.issue_dependency_op_indices == ()
    del ctx


def test_tlx_wave_converter_token_stage_records_multi_group_loop_carries(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_token_multi_async_for(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup0 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup0_group = ttg.async_commit_group tokens %warmup0
    %warmup1 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup1_group = ttg.async_commit_group tokens %warmup1
    %warmup2 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup2_group = ttg.async_commit_group tokens %warmup2
    %warmup3 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup3_group = ttg.async_commit_group tokens %warmup3
    %prewait = ttg.async_wait {num = 3 : i32}
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %wait0 = ttg.async_wait {num = 2 : i32}
      %body0 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body0_group = ttg.async_commit_group tokens %body0
      %wait1 = ttg.async_wait {num = 2 : i32}
      %body1 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body1_group = ttg.async_commit_group tokens %body1
      %wait2 = ttg.async_wait {num = 2 : i32}
      %body2 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body2_group = ttg.async_commit_group tokens %body2
      %wait3 = ttg.async_wait {num = 2 : i32}
      %body3 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body3_group = ttg.async_commit_group tokens %body3
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    for_op = next(op for op in source.ops if op.name == "scf.for")
    group_ops = [op for op in source.ops if op.name == "ttg.async_commit_group"]
    carries = token_program.loop_token_carries_by_op[for_op.index]
    assert tuple(carry.loop_op_index for carry in carries) == (for_op.index, ) * 3
    assert tuple(carry.init_source_value_id for carry in carries) == tuple(op.results[0] for op in group_ops[1:4])
    assert tuple(carry.yield_source_value_id for carry in carries) == tuple(op.results[0] for op in group_ops[5:8])
    assert tuple(carry.add_issue_dependency for carry in carries) == (False, False, False)
    assert tuple(carry.issue_dependency_op_indices for carry in carries) == ((), (), ())
    del ctx


def test_tlx_wave_converter_token_stage_pairs_wait_consumed_body_issue(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_token_consumed_body_issue(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_group = ttg.async_commit_group tokens %warmup
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %body0 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body0_group = ttg.async_commit_group tokens %body0
      %body1 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body1_group = ttg.async_commit_group tokens %body1
      %wait = ttg.async_wait {num = 1 : i32}
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    for_op = next(op for op in source.ops if op.name == "scf.for")
    warmup_group_op, body0_group_op, body1_group_op = [op for op in source.ops if op.name == "ttg.async_commit_group"]
    _warmup_load_op, _body0_load_op, _body1_load_op = [
        op for op in source.ops if op.name == "amdg.buffer_load_to_local"
    ]
    (carry, ) = token_program.loop_token_carries_by_op[for_op.index]
    assert carry.init_source_value_id == warmup_group_op.results[0]
    assert carry.yield_source_value_id == body1_group_op.results[0]
    assert carry.yield_source_value_id != body0_group_op.results[0]
    assert carry.add_issue_dependency is False
    assert carry.issue_dependency_op_indices == ()
    del ctx


def test_tlx_wave_converter_token_stage_orders_generic_memory_effects(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_memory_order(%ptr: tensor<64x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %loaded = tt.load %ptr : tensor<64x!tt.ptr<f32>, #blocked>
    tt.store %ptr, %loaded : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    load_effect, store_effect = token_program.memory_effects
    assert (load_effect.op_name, load_effect.kind) == ("tt.load", "read")
    assert (store_effect.op_name, store_effect.kind) == ("tt.store", "write")
    assert store_effect.depends_on_effect_ids == (load_effect.effect_id, )
    assert load_effect.alias_class == store_effect.alias_class == "unknown"
    assert token_program.nodes == ()
    del ctx


def test_tlx_wave_converter_token_stage_uses_memory_frontier(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_memory_frontier(%ptr: tensor<64x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %loaded0 = tt.load %ptr : tensor<64x!tt.ptr<f32>, #blocked>
    %loaded1 = tt.load %ptr : tensor<64x!tt.ptr<f32>, #blocked>
    tt.store %ptr, %loaded0 : tensor<64x!tt.ptr<f32>, #blocked>
    %loaded2 = tt.load %ptr : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    read0, read1, write, read2 = token_program.memory_effects
    assert (read0.kind, read1.kind, write.kind, read2.kind) == (
        "read",
        "read",
        "write",
        "read",
    )
    assert read0.depends_on_effect_ids == ()
    assert read1.depends_on_effect_ids == ()
    assert write.depends_on_effect_ids == (read0.effect_id, read1.effect_id)
    assert read2.depends_on_effect_ids == (write.effect_id, )
    del ctx


def test_tlx_wave_converter_token_stage_orders_local_memory_effects(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_local_effects(%value: tensor<64xf32, #blocked>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    ttg.local_store %value, %alloc : tensor<64xf32, #blocked> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %loaded = ttg.local_load %alloc : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    store_effect, load_effect = token_program.memory_effects
    assert (store_effect.op_name, store_effect.kind, store_effect.address_space) == (
        "ttg.local_store",
        "write",
        "local",
    )
    assert (load_effect.op_name, load_effect.kind, load_effect.address_space) == (
        "ttg.local_load",
        "read",
        "local",
    )
    assert load_effect.depends_on_effect_ids == (store_effect.effect_id, )
    assert token_program.nodes == ()
    del ctx


def test_tlx_wave_converter_vectorizes_dense_i8_local_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_vector_local_store(%value: tensor<512xi8, #blocked>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xi8, #shared, #smem, mutable>
    ttg.local_store %value, %alloc : tensor<512xi8, #blocked> -> !ttg.memdesc<512xi8, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    store_lines = [line for line in output.emitted_module.text.splitlines() if "wave.scatter" in line]
    assert len(store_lines) == 1
    assert "!wave.simd<vector<8xi8>, 64>" in store_lines[0]
    assert "packet_bindings []()" in store_lines[0]
    assert "bit_offset" in store_lines[0]
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.ds_store_tuple_b32" in machine
    assert "waveamdmachine.ds_store_b8" not in machine
    del ctx


def test_tlx_wave_converter_vectorizes_swizzled_i8_local_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_vector_swizzled_scale_store(%value: tensor<256x8xi8, #blocked>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x8xi8, #shared, #smem, mutable>
    ttg.local_store %value, %alloc : tensor<256x8xi8, #blocked> -> !ttg.memdesc<256x8xi8, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    store_lines = [line for line in output.emitted_module.text.splitlines() if "wave.scatter" in line]
    assert len(store_lines) == 1
    assert "!wave.simd<vector<8xi8>, 64>" in store_lines[0]
    assert "packet_bindings []()" in store_lines[0]
    assert "bit_offset" in store_lines[0]
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.ds_store_tuple_b32" in machine
    assert "waveamdmachine.ds_store_b8" not in machine
    del ctx


def test_tlx_wave_converter_vectorizes_swizzled_f16_local_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_vector_swizzled_f16_store(%value: tensor<128x64xf16, #blocked>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttg.local_store %value, %alloc : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    store_lines = [line for line in output.emitted_module.text.splitlines() if "wave.scatter" in line]
    assert len(store_lines) == 1
    assert "!wave.simd<vector<32xf16>, 64>" in store_lines[0]
    assert "packet_bindings []()" in store_lines[0]
    assert "bit_offset" in store_lines[0]
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert machine.count("waveamdmachine.ds_store_tuple_b32") == 4
    assert "waveamdmachine.ds_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_token_stage_records_buffer_store_effect():
    value_type = converter_source_ir.SourceType(
        "tensor<64xf16>",
        "tensor",
        shape=(64, ),
        element_type="f16",
    )
    pointer_type = converter_source_ir.SourceType(
        "!tt.ptr<f16>",
        "pointer",
        pointee_type="f16",
        address_space=1,
    )
    offset_type = converter_source_ir.SourceType(
        "tensor<64xi32>",
        "tensor",
        shape=(64, ),
        element_type="i32",
    )
    mask_type = converter_source_ir.SourceType(
        "tensor<64xi1>",
        "tensor",
        shape=(64, ),
        element_type="i1",
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("buffer_store_effect"),
        (converter_source_ir.SourceOp(
            0,
            "amdg.buffer_store",
            operands=(1, 2, 3, 4),
            attrs={
                "cacheModifier": "none",
                "operandSegmentSizes": (1, 1, 1, 0, 1),
            },
        ), ),
        {
            1: converter_source_ir.SourceValue(1, value_type, producer_name="arg0"),
            2: converter_source_ir.SourceValue(2, pointer_type, producer_name="arg1"),
            3: converter_source_ir.SourceValue(3, offset_type, producer_name="arg2"),
            4: converter_source_ir.SourceValue(4, mask_type, producer_name="arg3"),
        },
        (converter_source_ir.SourceRegion(0, (0, )), ),
        0,
    )

    token_program = converter_tokens.build_token_program(program, None)

    (effect, ) = token_program.memory_effects
    assert (effect.op_name, effect.kind, effect.address_space) == (
        "amdg.buffer_store",
        "write",
        "buffer",
    )
    assert effect.value_value_id == 1
    assert effect.address_value_id == 2
    assert effect.offset_value_id == 3
    assert effect.mask_value_id == 4
    assert effect.cache_modifier == "none"


def test_tlx_wave_converter_token_stage_records_buffer_load_effect():
    result_type = converter_source_ir.SourceType(
        "tensor<64xf32>",
        "tensor",
        shape=(64, ),
        element_type="f32",
    )
    pointer_type = converter_source_ir.SourceType(
        "!tt.ptr<f32>",
        "pointer",
        pointee_type="f32",
        address_space=1,
    )
    offset_type = converter_source_ir.SourceType(
        "tensor<64xi32>",
        "tensor",
        shape=(64, ),
        element_type="i32",
    )
    mask_type = converter_source_ir.SourceType(
        "tensor<64xi1>",
        "tensor",
        shape=(64, ),
        element_type="i1",
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("buffer_load_effect"),
        (converter_source_ir.SourceOp(
            0,
            "amdg.buffer_load",
            operands=(1, 2, 3, 4),
            results=(5, ),
            attrs={
                "cache": 1,
                "operandSegmentSizes": (1, 1, 0, 1, 1),
            },
        ), ),
        {
            1: converter_source_ir.SourceValue(1, pointer_type, producer_name="arg0"),
            2: converter_source_ir.SourceValue(2, offset_type, producer_name="arg1"),
            3: converter_source_ir.SourceValue(3, mask_type, producer_name="arg2"),
            4: converter_source_ir.SourceValue(4, result_type, producer_name="arg3"),
            5: converter_source_ir.SourceValue(5, result_type, producer_name="load"),
        },
        (converter_source_ir.SourceRegion(0, (0, )), ),
        0,
    )

    token_program = converter_tokens.build_token_program(program, None)

    (effect, ) = token_program.memory_effects
    assert (effect.op_name, effect.kind, effect.address_space) == (
        "amdg.buffer_load",
        "read",
        "buffer",
    )
    assert effect.address_value_id == 1
    assert effect.offset_value_id == 2
    assert effect.value_value_id is None
    assert effect.mask_value_id == 3


def test_tlx_wave_converter_token_stage_treats_global_and_buffer_as_may_alias():
    value_type = converter_source_ir.SourceType(
        "tensor<64xf32>",
        "tensor",
        shape=(64, ),
        element_type="f32",
    )
    pointer_type = converter_source_ir.SourceType(
        "!tt.ptr<f32>",
        "pointer",
        pointee_type="f32",
        address_space=1,
    )
    tensor_pointer_type = converter_source_ir.SourceType(
        "tensor<64x!tt.ptr<f32>>",
        "tensor",
        shape=(64, ),
        element_type="!tt.ptr<f32>",
        address_space=1,
    )
    offset_type = converter_source_ir.SourceType(
        "tensor<64xi32>",
        "tensor",
        shape=(64, ),
        element_type="i32",
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("global_buffer_alias"),
        (
            converter_source_ir.SourceOp(
                0,
                "tt.load",
                operands=(1, ),
                results=(4, ),
            ),
            converter_source_ir.SourceOp(
                1,
                "amdg.buffer_store",
                operands=(5, 2, 3),
                attrs={
                    "cacheModifier": "none",
                    "operandSegmentSizes": (1, 1, 1, 0, 0),
                },
            ),
        ),
        {
            1: converter_source_ir.SourceValue(1, tensor_pointer_type, producer_name="arg0"),
            2: converter_source_ir.SourceValue(2, pointer_type, producer_name="arg1"),
            3: converter_source_ir.SourceValue(3, offset_type, producer_name="arg2"),
            4: converter_source_ir.SourceValue(4, value_type, producer_name="load"),
            5: converter_source_ir.SourceValue(5, value_type, producer_name="arg3"),
        },
        (converter_source_ir.SourceRegion(0, (0, 1)), ),
        0,
    )

    token_program = converter_tokens.build_token_program(program, None)

    load_effect, store_effect = token_program.memory_effects
    assert (load_effect.address_space, store_effect.address_space) == (
        "global",
        "buffer",
    )
    assert store_effect.depends_on_effect_ids == (load_effect.effect_id, )


def test_tlx_wave_converter_token_stage_treats_unknown_space_as_may_alias():
    value_type = converter_source_ir.SourceType(
        "tensor<64xf32>",
        "tensor",
        shape=(64, ),
        element_type="f32",
    )
    unknown_tensor_pointer_type = converter_source_ir.SourceType(
        "tensor<64x!tt.ptr<f32>>",
        "tensor",
        shape=(64, ),
        element_type="!tt.ptr<f32>",
    )
    buffer_pointer_type = converter_source_ir.SourceType(
        "!tt.ptr<f32>",
        "pointer",
        pointee_type="f32",
        address_space=1,
    )
    offset_type = converter_source_ir.SourceType(
        "tensor<64xi32>",
        "tensor",
        shape=(64, ),
        element_type="i32",
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("unknown_alias"),
        (
            converter_source_ir.SourceOp(
                0,
                "tt.load",
                operands=(1, ),
                results=(4, ),
            ),
            converter_source_ir.SourceOp(
                1,
                "amdg.buffer_store",
                operands=(5, 2, 3),
                attrs={
                    "cacheModifier": "none",
                    "operandSegmentSizes": (1, 1, 1, 0, 0),
                },
            ),
            converter_source_ir.SourceOp(
                2,
                "tt.store",
                operands=(1, 5),
            ),
        ),
        {
            1: converter_source_ir.SourceValue(
                1,
                unknown_tensor_pointer_type,
                producer_name="arg0",
            ),
            2: converter_source_ir.SourceValue(2, buffer_pointer_type, producer_name="arg1"),
            3: converter_source_ir.SourceValue(3, offset_type, producer_name="arg2"),
            4: converter_source_ir.SourceValue(4, value_type, producer_name="load"),
            5: converter_source_ir.SourceValue(5, value_type, producer_name="arg3"),
        },
        (converter_source_ir.SourceRegion(0, (0, 1, 2)), ),
        0,
    )

    token_program = converter_tokens.build_token_program(program, None)

    unknown_read, buffer_write, unknown_write = token_program.memory_effects
    assert unknown_read.address_space == "unknown"
    assert buffer_write.address_space == "buffer"
    assert unknown_write.address_space == "unknown"
    assert buffer_write.depends_on_effect_ids == (unknown_read.effect_id, )
    assert unknown_write.depends_on_effect_ids == (
        unknown_read.effect_id,
        buffer_write.effect_id,
    )


def test_tlx_wave_converter_token_stage_reports_malformed_segments():
    pointer_type = converter_source_ir.SourceType(
        "!tt.ptr<f16>",
        "pointer",
        pointee_type="f16",
        address_space=1,
    )
    memdesc_type = converter_source_ir.SourceType(
        "!ttg.memdesc<64xf16>",
        "memdesc",
        shape=(64, ),
        element_type="f16",
    )
    token_type = converter_source_ir.SourceType("!tt.async.token", "token")
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("bad_segments"),
        (converter_source_ir.SourceOp(
            0,
            "ttg.async_copy_global_to_local",
            operands=(1, 2),
            results=(3, ),
            attrs={"operandSegmentSizes": (1, 1, 0)},
        ), ),
        {
            1:
            converter_source_ir.SourceValue(1, pointer_type, producer_name="arg0"),
            2:
            converter_source_ir.SourceValue(2, memdesc_type, producer_name="alloc"),
            3:
            converter_source_ir.SourceValue(
                3,
                token_type,
                owner_op_index=0,
                producer_name="ttg.async_copy_global_to_local",
            ),
        },
        (converter_source_ir.SourceRegion(0, (0, )), ),
        0,
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_tokens.build_token_program(program, None)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS"
    assert diagnostic.stage == "tokens"
    assert diagnostic.source_op_index == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_token_stage_rejects_non_token_dependencies():
    scalar_type = converter_source_ir.SourceType("i32", "scalar")
    token_type = converter_source_ir.SourceType("!tt.async.token", "token")
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("bad_token_dep"),
        (converter_source_ir.SourceOp(
            0,
            "ttg.async_commit_group",
            operands=(1, ),
            results=(2, ),
        ), ),
        {
            1: converter_source_ir.SourceValue(1, scalar_type, producer_name="arg0"),
            2: converter_source_ir.SourceValue(
                2,
                token_type,
                owner_op_index=0,
                producer_name="ttg.async_commit_group",
            ),
        },
        (converter_source_ir.SourceRegion(0, (0, )), ),
        0,
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_tokens.build_token_program(program, None)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_TOKEN_NON_TOKEN_DEPENDENCY"
    assert diagnostic.stage == "tokens"
    assert diagnostic.source_op_index == 0
    assert diagnostic.source_value_id == 1
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_op_stage_lowers_basic_dataflow(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_op(%arg0: i32, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %sum = arith.addi %arg0, %zero : i32
    %pred = arith.cmpi sge, %sum, %zero : i32
    llvm.intr.assume %pred : i1
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    facts = converter_facts.analyze_facts(source, converted)
    tokens = converter_tokens.build_token_program(source, converted)

    target = converter_op_conversion.convert_ops(source, converted, facts, tokens)

    assert target.contract == converter_target_ir.TargetContract(
        schema_version=converter_target_ir.TARGET_SCHEMA_VERSION,
        address_arithmetic=converter_target_ir.ADDRESS_ARITHMETIC_NO_OVERFLOW,
        enable_fp_fusion=False,
    )
    assert target.layouts
    assert target.assumptions
    assert any(layout.properties for layout in target.layouts)
    assert all(
        isinstance(prop, converter_target_ir.TargetAttr) for layout in target.layouts for prop in layout.properties)
    assert all(assumption.subject_target_ids for assumption in target.assumptions)
    assert all(value.layout_map_id is None or target.layouts[value.layout_map_id].layout_map_id == value.layout_map_id
               for value in target.values)
    assert converter_verifier.verify_target_program(
        target,
        source_program=source,
        token_program=tokens,
    )
    assert [op.kind for op in target.ops] == [
        "assume",
        "constant",
        "binary",
        "cmpi",
        "assume",
        "make_range",
        "splat",
        "addptr",
        "return",
    ]
    binary_op = next(op for op in target.ops if op.kind == "binary")
    assert converter_target_ir.attrs_dict(binary_op) == {
        "operation": "addi",
        "source_width": 32,
    }
    assume_op = next(op for op in target.ops if op.kind == "assume")
    assert assume_op.fact_ids
    assert assume_op.fact_target_ids
    range_op = next(op for op in target.ops if op.kind == "make_range")
    assert converter_target_ir.attrs_dict(range_op) == {"end": 64, "start": 0}
    assert not any(callable(attr.value) for op in target.ops for attr in op.attrs)
    assert not hasattr(target, "source_program")
    assert not hasattr(target, "target_ops")
    del ctx


def test_tlx_wave_converter_materializes_operand_assumes_before_arithmetic(tmp_path):
    local_func = """
  tt.func public @converter_early_assume(%arg0: i32) attributes {noinline = false} {
    %c255 = arith.constant 255 : i32
    %sum = arith.addi %arg0, %c255 : i32
    %zero = arith.constant 0 : i32
    %positive = arith.cmpi sgt, %arg0, %zero : i32
    llvm.intr.assume %positive : i1
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    add_op = next(op for op in output.target_program.ops
                  if op.kind == "binary" and converter_target_ir.attrs_dict(op)["operation"] == "addi")
    assert add_op.fact_ids
    assert add_op.fact_target_ids == (output.target_program.kernel.arg_target_ids[0], )
    wave = output.emitted_module.text
    assert wave.index("wave.assume %arg0") < wave.index("wave.binary addi")
    del ctx


def test_tlx_wave_converter_materializes_divisibility_only_on_integer_values(tmp_path):
    local_func = """
  tt.func public @converter_divisibility_facts(
      %pointer: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %stride: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert '#wave.pred<"Mod(x, 8) == 0">' in wave
    assert "wave.assume %arg0" not in wave
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_emits_facts_without_source_provenance(tmp_path):
    local_func = """
  tt.func public @converter_stripped_facts(%arg0: i32) attributes {noinline = false} {
    %c255 = arith.constant 255 : i32
    %sum = arith.addi %arg0, %c255 : i32
    %zero = arith.constant 0 : i32
    %positive = arith.cmpi sgt, %arg0, %zero : i32
    llvm.intr.assume %positive : i1
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    stripped_values = tuple(
        converter_target_ir.TargetValue(
            value.target_value_id,
            value.type,
            layout_map_id=value.layout_map_id,
        ) for value in output.target_program.values)
    stripped_target = converter_target_ir.TargetProgram(
        stripped_values,
        output.target_program.ops,
        output.target_program.regions,
        {},
        {},
        output.target_program.kernel,
        output.target_program.layouts,
        output.target_program.assumptions,
        output.target_program.contract,
    )

    assert converter_verifier.verify_target_program(stripped_target)
    stripped_emitted = converter_emission.emit_wave_module(stripped_target)

    assert stripped_emitted.text == output.emitted_module.text
    del ctx


def test_tlx_wave_converter_preserves_explicit_arith_overflow_flags(tmp_path):
    local_func = """
  tt.func public @converter_overflow_flags(%arg0: i32, %arg1: i32) attributes {noinline = false} {
    %sum = arith.addi %arg0, %arg1 overflow<nsw> : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    binary_op = next(op for op in output.target_program.ops if op.kind == "binary")
    assert converter_target_ir.attrs_dict(binary_op)["nsw"] is True
    assert "wave.index_expr" not in output.emitted_module.text
    assert "overflow<nsw>" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_preserves_wrapping_arith_with_scoped_ranges(tmp_path):
    local_func = """
  tt.func public @converter_range_nsw(%arg0: i32, %arg1: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c31 = arith.constant 31 : i32
    %arg0_nonnegative = arith.cmpi sge, %arg0, %c0 : i32
    llvm.intr.assume %arg0_nonnegative : i1
    %arg0_bounded = arith.cmpi sle, %arg0, %c31 : i32
    llvm.intr.assume %arg0_bounded : i1
    %arg1_nonnegative = arith.cmpi sge, %arg1, %c0 : i32
    llvm.intr.assume %arg1_nonnegative : i1
    %arg1_bounded = arith.cmpi sle, %arg1, %c31 : i32
    llvm.intr.assume %arg1_bounded : i1
    %sum = arith.addi %arg0, %arg1 : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    add_op = next(op for op in output.target_program.ops
                  if op.kind == "binary" and converter_target_ir.attrs_dict(op)["operation"] == "addi")
    add_attrs = converter_target_ir.attrs_dict(add_op)
    assert add_attrs == {"operation": "addi", "source_width": 32}
    assert "wave.binary addi" in output.emitted_module.text
    add_line = next(line for line in output.emitted_module.text.splitlines() if "wave.binary addi" in line)
    assert "overflow" not in add_line
    del ctx


def test_tlx_wave_converter_preserves_wrapping_layout_integer_math(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_layout_math_nsw(%stride: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<64xi32, #blocked>
    %scaled = arith.muli %range, %stride_splat : tensor<64xi32, #blocked>
    %offset = arith.addi %scaled, %range : tensor<64xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    layout_math = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "binary" and converter_target_ir.attrs_dict(op)["operation"] in {"muli", "addi"}
    ]
    assert {attrs["operation"] for attrs in layout_math} == {"muli", "addi"}
    assert all(set(attrs) == {"operation", "source_width"} for attrs in layout_math)
    assert "overflow<" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_marks_only_symbolic_address_math_nsw(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_scalar_address_math_nsw(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %tile: i32) attributes {noinline = false} {
    %c256 = arith.constant 256 : i32
    %base = arith.muli %tile, %c256 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %base_splat = tt.splat %base : i32 -> tensor<64xi32, #blocked>
    %offset = arith.addi %base_splat, %range : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    amdg.buffer_store %value, %arg0[%offset] {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    scalar_mul_attrs = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "binary" and not op.layout_map_ids and converter_target_ir.attrs_dict(op)["operation"] == "muli"
    ]
    assert scalar_mul_attrs
    assert all(set(attrs) == {"operation", "source_width"} for attrs in scalar_mul_attrs)
    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    offset_edge, offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    assert offset_edge.kind == "binary"
    assert offset_attrs["operation"] == "addi"
    assert "wave.address_arithmetic_no_overflow" in output.emitted_module.text
    assert "wave.binary" in output.emitted_module.text
    assert "overflow<" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_does_not_walk_loop_carried_address_producers(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_loop_carried_scalar_address_math_nsw(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %tile: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c64 = arith.constant 64 : i32
    %base0 = arith.muli %tile, %c64 : i32
    %base = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %base0) -> (i32)  : i32 {
      %next = arith.addi %acc, %c64 : i32
      scf.yield %next : i32
    }
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %base_splat = tt.splat %base : i32 -> tensor<64xi32, #blocked>
    %offset = arith.addi %base_splat, %range : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    amdg.buffer_store %value, %arg0[%offset] {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    scalar_binary_attrs = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "binary" and not op.layout_map_ids
    ]
    assert {attrs["operation"] for attrs in scalar_binary_attrs} >= {"muli", "addi"}
    assert all(set(attrs) == {"operation", "source_width"} for attrs in scalar_binary_attrs)
    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    offset_edge, offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    assert offset_edge.kind == "binary"
    assert offset_attrs["operation"] == "addi"
    assert "wave.address_arithmetic_no_overflow" in output.emitted_module.text
    assert "wave.binary" in output.emitted_module.text
    assert "overflow<" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_float_add(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_float_add() attributes {noinline = false} {
    %lhs = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    %rhs = arith.constant dense<2.000000e+00> : tensor<64xf32, #blocked>
    %sum = arith.addf %lhs, %rhs : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    float_op = next(op for op in output.target_program.ops if op.kind == "float_binary")
    attrs = converter_target_ir.attrs_dict(float_op)
    assert attrs == {"operation": "addf"}
    assert re.search(r"wave\.fadd (?!.*fastmath)", output.emitted_module.text)
    assert "wave.enable_fp_fusion" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_translates_global_fp_fusion_to_fastmath(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_float_mul_add(
      %out: !tt.ptr<f32> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %x = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    %y = arith.constant dense<2.000000e+00> : tensor<64xf32, #blocked>
    %prod = arith.mulf %x, %y fastmath<nnan> : tensor<64xf32, #blocked>
    %sum = arith.addf %x, %prod : tensor<64xf32, #blocked>
    %offset = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    amdg.buffer_store %sum, %out[%offset] {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(
        mod,
        enable_fp_fusion=True,
    )

    float_attrs = [converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "float_binary"]
    assert float_attrs == [
        {"operation": "mulf", "fastmath": ("nnan", "contract")},
        {"operation": "addf", "fastmath": ("contract", )},
    ]
    assert output.target_program.contract.enable_fp_fusion is True
    assert "wave.enable_fp_fusion" not in output.emitted_module.text
    canonical = _run_wave_canonicalize(output.emitted_module.text)
    assert "wave.fma" in canonical
    assert "fastmath<contract>" in canonical
    del ctx


def test_tlx_wave_converter_pipeline_preserves_source_fp_contract(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_source_float_contract(
      %out: !tt.ptr<f32> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %x = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    %y = arith.constant dense<2.000000e+00> : tensor<64xf32, #blocked>
    %prod = arith.mulf %x, %y fastmath<nnan,contract> : tensor<64xf32, #blocked>
    %sum = arith.addf %x, %prod fastmath<contract> : tensor<64xf32, #blocked>
    %offset = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    amdg.buffer_store %sum, %out[%offset] {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    float_attrs = [converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "float_binary"]
    assert float_attrs == [
        {"operation": "mulf", "fastmath": ("nnan", "contract")},
        {"operation": "addf", "fastmath": ("contract", )},
    ]
    assert "wave.enable_fp_fusion" not in output.emitted_module.text
    canonical = _run_wave_canonicalize(output.emitted_module.text)
    assert "wave.fma" in canonical
    assert "fastmath<contract>" in canonical
    del ctx


def test_tlx_wave_converter_pipeline_keeps_plain_mul_add_split(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_plain_float_mul_add(
      %out: !tt.ptr<f32> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %x = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    %y = arith.constant dense<2.000000e+00> : tensor<64xf32, #blocked>
    %prod = arith.mulf %x, %y : tensor<64xf32, #blocked>
    %sum = arith.addf %x, %prod : tensor<64xf32, #blocked>
    %offset = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    amdg.buffer_store %sum, %out[%offset] {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    canonical = _run_wave_canonicalize(output.emitted_module.text)
    assert "wave.fma" not in canonical
    assert "wave.fmul" in canonical
    assert "wave.fadd" in canonical
    del ctx


def test_tlx_wave_converter_pipeline_preserves_explicit_float_fastmath(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_explicit_float_fastmath() attributes {noinline = false} {
    %lhs = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    %rhs = arith.constant dense<2.000000e+00> : tensor<64xf32, #blocked>
    %sum = arith.addf %lhs, %rhs fastmath<nnan> : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    float_op = next(op for op in output.target_program.ops if op.kind == "float_binary")
    attrs = converter_target_ir.attrs_dict(float_op)
    assert attrs["operation"] == "addf"
    assert attrs["fastmath"] == ("nnan", )
    assert re.search(r"wave\.fadd .* fastmath<nnan>", output.emitted_module.text)
    assert "fastmath<nnan,contract>" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_sched_barrier(tmp_path):
    local_func = """
  tt.func public @converter_sched_barrier() attributes {noinline = false} {
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage0", triton.warp_pipeline.priority = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == ["sched_barrier", "return"]
    sched_barrier = output.target_program.ops[0]
    assert converter_target_ir.attrs_dict(sched_barrier) == {"border": "stage0", "mask": 0}
    assert "rocdl.sched.barrier" not in output.emitted_module.text
    assert output.emitted_module.text.count("wave.sched_barrier") == 1
    assert "wave.barrier" not in output.emitted_module.text
    assert "s_barrier" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_partial_sched_barrier(tmp_path):
    local_func = """
  tt.func public @converter_partial_sched_barrier() attributes {noinline = false} {
    rocdl.sched.barrier 2
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert output.emitted_module.text.count("wave.sched_barrier") == 1
    assert "wave.barrier" not in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_lowers_converted_warp_pipeline_primitives(tmp_path):
    local_func = """
  tt.func public @converter_warp_pipeline_primitives() attributes {noinline = false} {
    %c256 = arith.constant 256 : i32
    %c0 = arith.constant 0 : i32
    %wi = rocdl.workitem.id.x : i32
    %wave = arith.divsi %wi, %c256 : i32
    %high = arith.cmpi ne, %wave, %c0 : i32
    amdg.cond_barrier %high
    rocdl.s.setprio 2
    rocdl.sched.barrier 0
    rocdl.s.barrier
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    kinds = [op.kind for op in output.target_program.ops]
    assert "thread_id" in kinds
    assert "cond_barrier" in kinds
    assert "set_priority" in kinds
    assert "sched_barrier" in kinds
    assert "barrier" in kinds
    wave = output.emitted_module.text
    assert "wave.workitem_id 0" in wave
    assert "wave.where" in wave
    assert "waveamd.set_priority 2" in wave
    assert wave.count("wave.sched_barrier") == 1
    assert wave.count("wave.barrier") == 2
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.s_setprio" in machine
    assert machine.count("waveamdmachine.s_barrier") == 2
    assert machine.count("waveamdmachine.sched_barrier") == 1
    del ctx


@pytest.mark.parametrize(
    "barrier_scope,address_space",
    [("local", 1), ("all", 31)],
)
def test_tlx_wave_converter_pipeline_lowers_explicit_cta_barrier(
    tmp_path,
    barrier_scope,
    address_space,
):
    local_func = f"""
  tt.func public @converter_explicit_barrier() attributes {{noinline = false}} {{
    ttg.barrier {barrier_scope}
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=2)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == ["barrier", "return"]
    barrier = output.target_program.ops[0]
    assert converter_target_ir.attrs_dict(barrier) == {
        "address_space": address_space,
        "dependency_count": 0,
        "orders_memory_issue": barrier_scope == "all",
    }
    wave = output.emitted_module.text
    assert wave.count("wave.barrier") == 1
    assert "wave.sched_barrier" not in wave
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.s_barrier") == 1
    assert "waveamdmachine.sched_barrier" not in machine
    del ctx


def test_tlx_wave_converter_imports_compiler_membar_issue_order(tmp_path):
    local_func = """
  tt.func public @converter_compiler_membar() attributes {noinline = false} {
    ttg.barrier local
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4)
    barriers = []

    def collect_barriers(op):
        if op.get_name() == "ttg.barrier":
            barriers.append(op)
        return True

    mod.walk(collect_barriers)
    assert len(barriers) == 1

    output = converter_pipeline.convert_ttgir_to_wave(
        mod,
        compiler_membar_barriers=tuple(barriers),
    )

    source_barrier = next(op for op in output.source_program.ops if op.name == "ttg.barrier")
    assert source_barrier.attrs["tlx.compiler_membar_barrier"] is True
    assert source_barrier.attrs["tlx.orders_memory_issue"] is True
    target_barrier = next(op for op in output.target_program.ops if op.kind == "barrier")
    target_attrs = converter_target_ir.attrs_dict(target_barrier)
    assert target_attrs["compiler_membar_barrier"] is True
    assert target_attrs["orders_memory_issue"] is True
    # Compiler-selected barriers remain literal source operations even when
    # there is no bridge-owned memory epoch to attach to them.
    assert output.emitted_module.text.count("wave.barrier") == 1
    del ctx


def test_tlx_wave_full_barrier_orders_memory_issue_without_blocking_redistribute(tmp_path, ):
    preamble = """
#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#result = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_full_barrier_issue_order() attributes {noinline = false} {
    %source_alloc = ttg.local_alloc : () -> !ttg.memdesc<8x8xi32, #shared, #smem, mutable>
    %result_alloc = ttg.local_alloc : () -> !ttg.memdesc<8x8xi32, #shared, #smem, mutable>
    %loaded = ttg.local_load %source_alloc : !ttg.memdesc<8x8xi32, #shared, #smem, mutable> -> tensor<8x8xi32, #source>
    ttg.barrier all
    %converted = ttg.convert_layout %loaded : tensor<8x8xi32, #source> -> tensor<8x8xi32, #result>
    ttg.local_store %converted, %result_alloc : tensor<8x8xi32, #result> -> !ttg.memdesc<8x8xi32, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "local_load"]
    (barrier_op, ) = [op for op in output.target_program.ops if op.kind == "barrier"]
    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    (store_op, ) = [op for op in output.target_program.ops if op.kind == "local_store"]
    issue_ops = [op for op in output.target_program.ops if op.kind == "issue_token"]
    assert len(issue_ops) == 1
    (post_issue, ) = issue_ops
    assert converter_target_ir.attrs_dict(post_issue)["projection_domain"] == (
        converter_target_ir.EVENT_DOMAIN_BARRIER_ISSUE)
    assert converter_target_ir.attrs_dict(barrier_op)["lds_read_dependency_count"] == 1
    assert barrier_op.operands == (load_op.results[-1], )
    assert post_issue.operands == barrier_op.results
    assert store_op.operands[-1] == post_issue.results[0]
    assert converter_target_ir.attrs_dict(store_op)["barrier_order_dependency_count"] == 1
    assert post_issue.results[0] not in convert_op.operands
    _assert_mechanical_layout_transform(output.target_program, convert_op)

    wave = output.emitted_module.text
    assert wave.count("wave.issue_token") == 1
    assert "wave.sched_barrier" not in wave
    assert wave.count("wave.redistribute") == 1
    _run_wave_verify(wave)
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.issue_token") == 1
    assert "waveamdmachine.sched_barrier" not in machine
    del ctx


def test_tlx_wave_full_barrier_orders_generic_and_buffer_memory_siblings(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_full_barrier_global_issue_order(
      %generic_src: !tt.ptr<f32>,
      %generic_dst: !tt.ptr<f32>,
      %buffer_src: !tt.ptr<f32> {tt.pointer_range = 32 : i32},
      %buffer_dst: !tt.ptr<f32> {tt.pointer_range = 32 : i32},
      %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %generic_src_base = tt.splat %generic_src : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %generic_dst_base = tt.splat %generic_dst : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %generic_src_ptr = tt.addptr %generic_src_base, %range : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %generic_dst_ptr = tt.addptr %generic_dst_base, %range : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %generic_loaded = tt.load %generic_src_ptr, %mask, %other : tensor<64x!tt.ptr<f32>, #blocked>
    %buffer_loaded = amdg.buffer_load %buffer_src[%range], %mask, %other {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    ttg.barrier all
    %sum = arith.addf %generic_loaded, %buffer_loaded : tensor<64xf32, #blocked>
    tt.store %generic_dst_ptr, %sum, %mask : tensor<64x!tt.ptr<f32>, #blocked>
    amdg.buffer_store %sum, %buffer_dst[%range], %mask {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    ttg.barrier all
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    loads = [op for op in output.target_program.ops if op.kind in {"load", "buffer_load"}]
    stores = [op for op in output.target_program.ops if op.kind in {"store", "buffer_store"}]
    assert len(loads) == len(stores) == 2
    assert all(converter_target_ir.attrs_dict(op)["issue_order_result_count"] == 1 for op in (*loads, *stores))
    assert all(
        output.target_program.values[op.results[-1]].event_domain == converter_target_ir.EVENT_DOMAIN_MEMORY_COMPLETION
        for op in (*loads, *stores))
    store_epochs = {op.operands[-1] for op in stores}
    assert len(store_epochs) == 1
    (store_epoch, ) = store_epochs
    assert output.target_program.values[store_epoch].event_domain == (converter_target_ir.EVENT_DOMAIN_BARRIER_ISSUE)
    assert all(converter_target_ir.attrs_dict(op)["barrier_order_dependency_count"] == 1 for op in stores)
    barriers = [op for op in output.target_program.ops if op.kind == "barrier"]
    assert len(barriers) == 2
    memory_issue_ops = [
        op for op in output.target_program.ops if op.kind == "issue_token"
        and converter_target_ir.attrs_dict(op)["projection_domain"] == converter_target_ir.EVENT_DOMAIN_MEMORY_ISSUE
    ]
    assert len(memory_issue_ops) == 2
    assert set(memory_issue_ops[0].operands) == {op.results[-1] for op in loads}
    assert set(memory_issue_ops[1].operands) == {op.results[-1] for op in stores}
    (sum_op, ) = [op for op in output.target_program.ops if op.kind == "float_binary"]
    assert store_epoch not in sum_op.operands

    wave = output.emitted_module.text
    assert wave.count("wave.issue_token") == 3
    assert "wave.sched_barrier" not in wave
    assert wave.count("wave.barrier") == 2
    _run_wave_verify(wave)
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.issue_token") == 3
    assert "waveamdmachine.sched_barrier" not in machine
    del ctx


@pytest.mark.parametrize("num_warps", (1, 4))
def test_tlx_wave_converter_preserves_adjacent_wait_and_barrier(
    tmp_path,
    num_warps,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [NUM_WARPS], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
""".replace("NUM_WARPS", str(num_warps))
    local_func = """
  tt.func public @converter_wait_adjacent_barrier(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<256xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %copy = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<256xi32, #blocked>] -> <256xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %copy
    %wait = ttg.async_wait %group {num = 0 : i32}
    ttg.barrier local
    %loaded = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<256xf16, #shared, #smem, mutable> -> tensor<256xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=num_warps,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (wait_op, ) = [op for op in output.target_program.ops if op.kind == "async_wait"]
    wait_attrs = converter_target_ir.attrs_dict(wait_op)
    assert set(wait_attrs) >= {
        "completed_group_dependency_count",
        "retained_issue_dependency_count",
    }
    assert "publication_mode" not in wait_attrs
    assert "coalesced_source_barrier_op_index" not in wait_attrs
    source_barrier_ops = [op for op in output.source_program.ops if op.name == "ttg.barrier"]
    assert len(source_barrier_ops) == 1
    source_barrier_index = source_barrier_ops[0].index
    target_barrier_ops = [
        op for op in output.target_program.ops if op.kind == "barrier" and op.source_op_index == source_barrier_index
    ]
    assert len(target_barrier_ops) == 1
    (barrier_op, ) = target_barrier_ops
    (load_op, ) = [op for op in output.target_program.ops if op.kind == "local_load"]
    assert barrier_op.operands == wait_op.results
    assert converter_target_ir.attrs_dict(barrier_op)["dependency_count"] == 1
    assert load_op.operands[1:] == wait_op.results
    # The wait remains an SSA readiness edge and the source barrier remains a
    # distinct barrier. Neither operation is synthesized from the other.
    assert output.emitted_module.text.count("wave.barrier") == 1
    assert "wave.sched_barrier" not in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_explicit_barrier_preserves_structural_lds_read_completion(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_explicit_barrier_lds_frontier(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<256xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %copy = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<256xi32, #blocked>] -> <256xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %copy
    %wait = ttg.async_wait %group {num = 0 : i32}
    %loaded = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<256xf16, #shared, #smem, mutable> -> tensor<256xf16, #blocked>
    ttg.barrier local
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=4,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "local_load"]
    (barrier_op, ) = [
        op for op in output.target_program.ops if op.kind == "barrier" and op.source_op_index is not None
    ][-1:]
    (wait_op, ) = [op for op in output.target_program.ops if op.kind == "async_wait"]
    assert load_op.operands[1:] == wait_op.results
    assert barrier_op.operands == (
        *wait_op.results,
        load_op.results[-1],
    )
    assert len(barrier_op.results) == 1
    barrier_attrs = converter_target_ir.attrs_dict(barrier_op)
    assert barrier_attrs["dependency_count"] == 1
    assert barrier_attrs["lds_read_dependency_count"] == 1
    assert barrier_attrs["lds_completion_result_count"] == 1

    wave = output.emitted_module.text
    explicit_barrier_line = [line for line in wave.splitlines() if "wave.barrier" in line][-1]
    assert "wave.barrier %" in explicit_barrier_line
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_rejects_scalar_float_add(tmp_path):
    local_func = """
  tt.func public @converter_scalar_float_add(%arg0: f32, %arg1: f32) attributes {noinline = false} {
    %sum = arith.addf %arg0, %arg1 : f32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_FLOAT_BINARY"
    assert diagnostic.stage == "op_conversion"
    assert diagnostic.no_fallback is True
    del ctx


@pytest.mark.parametrize(
    "operations",
    (
        ("divsi", "remsi"),
        ("remsi", "divsi"),
        ("divui", "remui"),
        ("remui", "divui"),
    ),
)
def test_tlx_wave_converter_preserves_div_rem_source_order(operations):
    target = _target_div_rem_program(operations)

    ordered = converter_barrier_order.thread_barrier_issue_order(target)
    assert [converter_target_ir.attrs_dict(op)["operation"]
            for op in ordered.ops
            if op.kind == "binary"] == list(operations)


def test_tlx_wave_converter_preserves_async_wait_source_order():
    target = _target_async_wait_local_load_program("local_load")

    ordered = converter_barrier_order.thread_barrier_issue_order(target)

    op_ids = ordered.regions[0].op_ids
    assert [ordered.ops[op_id].kind for op_id in op_ids] == [
        "async_commit_group",
        "memdesc_index",
        "local_load",
        "async_wait",
    ]


def test_tlx_wave_converter_orders_compiler_membar_between_dma_epochs():
    builder = converter_target_ir.TargetBuilder()
    token_type = converter_target_ir.TargetType("token", "token")
    first_completion = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
    )
    second_completion = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
    )
    first_group = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_GROUP,
    )
    first_dma_id = builder.add_op(
        "buffer_load_to_local",
        results=(first_completion, ),
        attrs={"mode": "symbolic_copy"},
    )
    first_commit_id = builder.add_op(
        "async_commit_group",
        operands=(first_completion, ),
        results=(first_group, ),
    )
    barrier_id = builder.add_op(
        "barrier",
        attrs={
            "address_space": 1,
            "compiler_membar_barrier": True,
            "dependency_count": 0,
            "orders_memory_issue": True,
        },
    )
    second_dma_id = builder.add_op(
        "buffer_load_to_local",
        results=(second_completion, ),
        attrs={"mode": "symbolic_copy"},
    )

    target = converter_canonicalize.eliminate_redundant_compiler_membar_barriers(builder.build())
    target = converter_barrier_order.thread_barrier_issue_order(target)

    ordered_ops = [target.ops[op_id] for op_id in target.regions[0].op_ids]
    assert [op.kind for op in ordered_ops] == [
        "buffer_load_to_local",
        "async_commit_group",
        "issue_token",
        "barrier",
        "issue_token",
        "buffer_load_to_local",
    ]
    first_dma = target.ops[first_dma_id]
    assert target.ops[first_commit_id] == ordered_ops[1]
    pre_issue = ordered_ops[2]
    barrier = target.ops[barrier_id]
    post_issue = ordered_ops[4]
    second_dma = target.ops[second_dma_id]
    assert pre_issue.operands == (first_dma.results[0], )
    assert barrier.operands == pre_issue.results
    assert post_issue.operands == barrier.results
    assert second_dma.operands == post_issue.results
    assert (target.values[barrier.results[0]].event_domain == converter_target_ir.EVENT_DOMAIN_MEMORY_BARRIER)


def test_tlx_wave_converter_elides_compiler_membar_within_open_dma_group():
    builder = converter_target_ir.TargetBuilder()
    token_type = converter_target_ir.TargetType("token", "token")
    completions = tuple(
        builder.add_value(
            token_type,
            event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
        ) for _ in range(2))
    group = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_GROUP,
    )
    builder.add_op(
        "buffer_load_to_local",
        results=(completions[0], ),
        attrs={"mode": "symbolic_copy"},
    )
    builder.add_op(
        "barrier",
        attrs={
            "address_space": 1,
            "compiler_membar_barrier": True,
            "dependency_count": 0,
            "orders_memory_issue": True,
        },
    )
    builder.add_op(
        "buffer_load_to_local",
        results=(completions[1], ),
        attrs={"mode": "symbolic_copy"},
    )
    builder.add_op(
        "async_commit_group",
        operands=completions,
        results=(group, ),
    )

    target = converter_canonicalize.eliminate_redundant_compiler_membar_barriers(builder.build())

    assert [target.ops[op_id].kind for op_id in target.regions[0].op_ids] == [
        "buffer_load_to_local",
        "buffer_load_to_local",
        "async_commit_group",
    ]


def test_tlx_wave_converter_elides_compiler_membar_before_wait_ready_load():
    builder = converter_target_ir.TargetBuilder()
    token_type = converter_target_ir.TargetType("token", "token")
    completion = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
    )
    group = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_GROUP,
    )
    builder.add_op(
        "buffer_load_to_local",
        results=(completion, ),
        attrs={"mode": "symbolic_copy"},
    )
    builder.add_op(
        "async_commit_group",
        operands=(completion, ),
        results=(group, ),
    )
    builder.add_op(
        "barrier",
        attrs={
            "address_space": 1,
            "compiler_membar_barrier": True,
            "dependency_count": 0,
            "orders_memory_issue": True,
        },
    )
    builder.add_op(
        "local_load",
        attrs={"synced_via_async_wait": True},
    )

    target = converter_canonicalize.eliminate_redundant_compiler_membar_barriers(builder.build())

    assert [target.ops[op_id].kind for op_id in target.regions[0].op_ids] == [
        "buffer_load_to_local",
        "async_commit_group",
        "local_load",
    ]


def _target_async_wait_local_load_program(load_kind):
    builder = converter_target_ir.TargetBuilder()
    token_type = converter_target_ir.TargetType("token", "token")
    memdesc_type = converter_target_ir.TargetType("memdesc", "memdesc", "f16")
    tensor_type = converter_target_ir.TargetType("tensor", "simd_tuple", "f16", 64, 1)
    seed_token = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
    )
    commit_token = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_GROUP,
    )
    wait_token = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
    )
    memdesc = builder.add_value(memdesc_type)
    indexed_memdesc = builder.add_value(memdesc_type)
    payload = builder.add_value(tensor_type)
    builder.add_op("async_commit_group", operands=(seed_token, ), results=(commit_token, ))
    builder.add_op("memdesc_index", operands=(memdesc, ), results=(indexed_memdesc, ))
    builder.add_op(load_kind, operands=(indexed_memdesc, ), results=(payload, ))
    builder.add_op(
        "async_wait",
        operands=(commit_token, ),
        results=(wait_token, ),
        attrs={
            "completed_group_dependency_count": 1,
            "retained_issue_dependency_count": 0,
        },
    )
    return builder.build()


def _target_div_rem_program(operations):
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    values = tuple(converter_target_ir.TargetValue(value_id, scalar_i32) for value_id in range(4))
    result_ids = {"divsi": 2, "divui": 2, "remsi": 3, "remui": 3}
    if operations[0].startswith("rem"):
        result_ids = {"remsi": 2, "remui": 2, "divsi": 3, "divui": 3}
    ops = tuple(
        converter_target_ir.TargetOp(
            index,
            "binary",
            operands=(0, 1),
            results=(result_ids[operation], ),
            attrs=(
                converter_target_ir.TargetAttr("operation", operation),
                converter_target_ir.TargetAttr("source_width", 32),
            ),
        ) for index, operation in enumerate(operations))
    return converter_target_ir.TargetProgram(
        values,
        ops,
        (converter_target_ir.TargetRegion(0, tuple(range(len(ops)))), ),
        {},
        {},
    )


def test_tlx_wave_converter_op_stage_lowers_generic_load():
    pointer_type = converter_source_ir.SourceType(
        "tensor<64x!tt.ptr<f32>>",
        "tensor",
        shape=(64, ),
        pointee_type="f32",
        address_space=1,
    )
    value_type = converter_source_ir.SourceType(
        "tensor<64xf32>",
        "tensor",
        shape=(64, ),
        element_type="f32",
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("unsupported_load", arg_ids=(1, )),
        (converter_source_ir.SourceOp(
            0,
            "tt.load",
            operands=(1, ),
            results=(2, ),
        ), ),
        {
            1: converter_source_ir.SourceValue(
                1,
                pointer_type,
                producer_name="arg0",
                argument_index=0,
            ),
            2: converter_source_ir.SourceValue(
                2,
                value_type,
                owner_op_index=0,
                producer_name="tt.load",
            ),
        },
        (converter_source_ir.SourceRegion(0, (0, )), ),
        0,
    )
    converted = converter_types.convert_source_program(program)
    facts = converter_facts.analyze_facts(program, converted)
    tokens = converter_tokens.build_token_program(program, converted)

    target = converter_op_conversion.convert_ops(program, converted, facts, tokens)

    (load_op, ) = target.ops
    assert load_op.kind == "load"
    assert load_op.operands == (0, )
    assert load_op.results == (1, )
    assert converter_target_ir.attrs_dict(load_op) == {
        "component_count": 1,
        "element_type": "f32",
        "has_mask": False,
        "has_other": False,
        "lane_width": 64,
        "mask_mode": "none",
    }


def test_tlx_wave_converter_verifier_rejects_missing_fact():
    target = converter_target_ir.TargetProgram(
        (converter_target_ir.TargetValue(
            0,
            converter_target_ir.TargetType("mask", "mask", "i1"),
        ), ),
        (converter_target_ir.TargetOp(
            0,
            "assume",
            operands=(0, ),
        ), ),
        (converter_target_ir.TargetRegion(0, (0, )), ),
        {},
        {},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_MISSING_FACT"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_unknown_value_layout():
    target = converter_target_ir.TargetProgram(
        values=(converter_target_ir.TargetValue(
            0,
            converter_target_ir.TargetType("tensor", "simd", "i32", 64),
            layout_map_id=0,
        ), ),
        ops=(),
        regions=(converter_target_ir.TargetRegion(0), ),
        source_value_targets={},
        erased_source_values={},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_UNKNOWN_LAYOUT"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_value_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_wrapping_address_contract():
    target = converter_target_ir.TargetProgram(
        values=(),
        ops=(),
        regions=(converter_target_ir.TargetRegion(0), ),
        source_value_targets={},
        erased_source_values={},
        contract=converter_target_ir.TargetContract(address_arithmetic="wrapping", ),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_ADDRESS_ARITHMETIC"
    assert diagnostic.stage == "verification"
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_non_boolean_fp_fusion_contract():
    target = converter_target_ir.TargetProgram(
        values=(),
        ops=(),
        regions=(converter_target_ir.TargetRegion(0), ),
        source_value_targets={},
        erased_source_values={},
        contract=converter_target_ir.TargetContract(enable_fp_fusion="yes", ),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_FP_FUSION"
    assert diagnostic.stage == "verification"
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_unknown_contract_version():
    target = converter_target_ir.TargetProgram(
        values=(),
        ops=(),
        regions=(converter_target_ir.TargetRegion(0), ),
        source_value_targets={},
        erased_source_values={},
        contract=converter_target_ir.TargetContract(schema_version=converter_target_ir.TARGET_SCHEMA_VERSION + 1, ),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_SCHEMA_VERSION"
    assert diagnostic.stage == "verification"
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_missing_fact_target():
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    target = converter_target_ir.TargetProgram(
        (converter_target_ir.TargetValue(0, scalar_i32, source_value_id=0), ),
        (converter_target_ir.TargetOp(
            0,
            "assume",
            fact_ids=(0, ),
        ), ),
        (converter_target_ir.TargetRegion(0, (0, )), ),
        {0: (0, )},
        {},
    )
    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_FACT_TARGET_COUNT"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_incompatible_fact_target():
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    target = converter_target_ir.TargetProgram(
        (
            converter_target_ir.TargetValue(0, scalar_i32, source_value_id=0),
            converter_target_ir.TargetValue(1, scalar_i32, source_value_id=1),
        ),
        (converter_target_ir.TargetOp(
            0,
            "assume",
            fact_ids=(0, ),
            fact_target_ids=(1, ),
        ), ),
        (converter_target_ir.TargetRegion(0, (0, )), ),
        {0: (0, ), 1: (1, )},
        {},
        assumptions=(converter_target_ir.TargetAssumption(
            0,
            "range",
            "sge",
            (0, ),
            lower=0,
        ), ),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_FACT_TARGET"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.target_value_id == 1
    assert diagnostic.fact_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_layout_convert_without_fact_policy():
    builder = converter_target_ir.TargetBuilder()
    tensor = converter_target_ir.TargetType("tensor", "simd", "f32", 64, 1)
    operand = builder.add_value(tensor, source_value_id=0)
    result = builder.add_value(tensor, source_value_id=1)
    builder.add_op(
        "layout_convert",
        operands=(operand, ),
        results=(result, ),
        attrs={"transform": "identity"},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_LAYOUT_FACT_POLICY"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_invalidating_layout_convert_facts():
    builder = converter_target_ir.TargetBuilder()
    tensor = converter_target_ir.TargetType("tensor", "simd", "i32", 64, 1)
    operand = builder.add_value(tensor, source_value_id=0)
    result = builder.add_value(tensor, source_value_id=1)
    builder.add_op(
        "layout_convert",
        operands=(operand, ),
        results=(result, ),
        attrs={
            "fact_policy": "invalidate_layout_sensitive",
            "transform": "identity",
        },
        fact_ids=(0, ),
        fact_target_ids=(operand, ),
    )
    fact_program = converter_facts.FactProgram(
        (converter_facts.Fact(0, "range", 0, "signed_width", lower=0), ),
        {0: (0, )},
    )
    builder.set_assumptions(
        converter_target_ir.target_assumptions_from_facts(
            fact_program,
            builder.source_value_targets,
        ))

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_LAYOUT_FACT_POLICY"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_physical_layout_convert_attrs():
    builder = converter_target_ir.TargetBuilder()
    tensor = converter_target_ir.TargetType("tensor", "simd", "i32", 64, 1)
    operand = builder.add_value(tensor, source_value_id=0)
    result = builder.add_value(tensor, source_value_id=1)
    builder.add_op(
        "layout_convert",
        operands=(operand, ),
        results=(result, ),
        attrs={
            "fact_policy": "invalidate_layout_sensitive",
            "transform": "identity",
            "mode": "same_lane_register_remap",
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_LAYOUT_ATTRS"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_non_convert_layout_source():
    builder = converter_target_ir.TargetBuilder(layouts=(_identity_target_layout(), ), )
    tensor = converter_target_ir.TargetType("tensor", "simd", "i32", 64, 1)
    operand = builder.add_value(tensor, source_value_id=0, layout_map_id=0)
    result = builder.add_value(tensor, source_value_id=1, layout_map_id=0)
    builder.add_op(
        "layout_convert",
        operands=(operand, ),
        results=(result, ),
        attrs={
            "fact_policy": "invalidate_layout_sensitive",
            "transform": "identity",
        },
        source_op_index=0,
    )
    source_program = SimpleNamespace(ops=(SimpleNamespace(name="tt.dot"), ))

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(
            builder.build(),
            source_program=source_program,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_LAYOUT_CONVERT_SOURCE"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.source_op_index == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_unknown_target_value():
    target = converter_target_ir.TargetProgram(
        (),
        (converter_target_ir.TargetOp(
            0,
            "return",
            operands=(99, ),
        ), ),
        (converter_target_ir.TargetRegion(0, (0, )), ),
        {},
        {},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_UNKNOWN_TARGET_VALUE"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.target_value_id == 99
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_semantic_memory_mask_edges():
    base_type = converter_target_ir.TargetType(
        "pointer",
        "uniform_pointer",
        "f32",
    )
    offset_type = converter_target_ir.TargetType(
        "tensor",
        "simd",
        "index",
        64,
        1,
    )
    mask_type = converter_target_ir.TargetType(
        "mask",
        "mask",
        "i1",
        64,
        1,
    )
    target = converter_target_ir.TargetProgram(
        (
            converter_target_ir.TargetValue(0, base_type),
            converter_target_ir.TargetValue(1, offset_type),
            converter_target_ir.TargetValue(2, mask_type),
        ),
        (converter_target_ir.TargetOp(
            0,
            "buffer_load",
            operands=(0, 1, 2),
            attrs=(
                converter_target_ir.TargetAttr("has_mask", True),
                converter_target_ir.TargetAttr("component_count", 1),
                converter_target_ir.TargetAttr("access_component_count", 1),
                converter_target_ir.TargetAttr(
                    "mask_operand_mode",
                    "structural_predicate",
                ),
                converter_target_ir.TargetAttr("mask_scalar_count", 0),
                converter_target_ir.TargetAttr(
                    "mask_predicate_plans",
                    (("scalar_predicate", 0), ),
                ),
                converter_target_ir.TargetAttr(
                    "mask_predicate_component_map",
                    (0, ),
                ),
            ),
        ), ),
        (converter_target_ir.TargetRegion(0, (0, )), ),
        {},
        {},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    assert exc_info.value.code == "TLXW_VERIFY_MASK_EDGE"
    assert "unsupported memory mask operand mode" in str(exc_info.value)


def test_tlx_wave_converter_verifier_rejects_f32_memory_offset_edges():
    target = converter_target_ir.TargetProgram(
        (
            converter_target_ir.TargetValue(
                0,
                converter_target_ir.TargetType(
                    "pointer",
                    "uniform_pointer",
                    "i32",
                ),
            ),
            converter_target_ir.TargetValue(
                1,
                converter_target_ir.TargetType(
                    "tensor",
                    "simd",
                    "f32",
                    64,
                    1,
                ),
            ),
        ),
        (converter_target_ir.TargetOp(
            0,
            "buffer_load",
            operands=(0, 1),
            attrs=(
                converter_target_ir.TargetAttr("has_mask", False),
                converter_target_ir.TargetAttr("component_count", 1),
                converter_target_ir.TargetAttr("access_component_count", 1),
            ),
        ), ),
        (converter_target_ir.TargetRegion(0, (0, )), ),
        {},
        {},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    assert exc_info.value.code == "TLXW_VERIFY_MEMORY_EDGE"
    assert "SIMD i32 or index representation" in str(exc_info.value)


def test_tlx_wave_converter_verifier_rejects_unknown_target_op():
    target = converter_target_ir.TargetProgram(
        (),
        (converter_target_ir.TargetOp(
            0,
            "layout_convert_like",
        ), ),
        (converter_target_ir.TargetRegion(0, (0, )), ),
        {},
        {},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_UNKNOWN_TARGET_OP"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_fragment_target_value():
    target = converter_target_ir.TargetProgram(
        (converter_target_ir.TargetValue(
            0,
            converter_target_ir.TargetType(
                "tensor",
                "fragment",
                "f32",
                64,
                1,
            ),
        ), ),
        (),
        (converter_target_ir.TargetRegion(0), ),
        {},
        {},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_FRAGMENT_BOUNDARY"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_value_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_issue_event_as_wait_completion():
    builder = converter_target_ir.TargetBuilder()
    token_type = converter_target_ir.TargetType("token", "token")
    issue = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_ISSUE,
    )
    ready = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
    )
    builder.add_op(
        "async_wait",
        operands=(issue, ),
        results=(ready, ),
        attrs={
            "completed_group_dependency_count": 1,
            "retained_issue_dependency_count": 0,
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    assert exc_info.value.code == "TLXW_VERIFY_ASYNC_PROTOCOL_DOMAIN"


def test_tlx_wave_converter_verifier_rejects_malformed_wait_segments():
    builder = converter_target_ir.TargetBuilder()
    token_type = converter_target_ir.TargetType("token", "token")
    group = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_GROUP,
    )
    ready = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
    )
    builder.add_op(
        "async_wait",
        operands=(group, ),
        results=(ready, ),
        attrs={
            "completed_group_dependency_count": 0,
            "retained_issue_dependency_count": 0,
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    assert exc_info.value.code == "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS"


def test_tlx_wave_converter_verifier_rejects_non_dma_issue_projection():
    builder = converter_target_ir.TargetBuilder()
    token_type = converter_target_ir.TargetType("token", "token")
    completion = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
    )
    issue = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_ISSUE,
    )
    builder.add_op(
        "issue_token",
        operands=(completion, ),
        results=(issue, ),
        attrs={
            "input_count": 1,
            "projection_domain": "lds_issue",
            "projection_provenance": "lds_release_publication",
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    assert exc_info.value.code == "TLXW_VERIFY_ASYNC_PROTOCOL_DOMAIN"


def test_tlx_wave_converter_verifier_rejects_memory_completion_as_dma_issue():
    builder = converter_target_ir.TargetBuilder()
    token_type = converter_target_ir.TargetType("token", "token")
    completion = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
    )
    issue = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_ISSUE,
    )
    builder.add_op(
        "issue_token",
        operands=(completion, ),
        results=(issue, ),
        attrs={
            "input_count": 1,
            "projection_domain": converter_target_ir.EVENT_DOMAIN_DMA_ISSUE,
            "projection_provenance": "partial_wait_retained_group",
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    assert exc_info.value.code == "TLXW_VERIFY_ASYNC_PROTOCOL_DOMAIN"


def test_tlx_wave_converter_verifier_rejects_unproven_barrier_issue_projection():
    builder = converter_target_ir.TargetBuilder()
    token_type = converter_target_ir.TargetType("token", "token")
    raw_barrier = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_MEMORY_BARRIER,
    )
    barrier_issue = builder.add_value(
        token_type,
        event_domain=converter_target_ir.EVENT_DOMAIN_BARRIER_ISSUE,
    )
    builder.add_op(
        "issue_token",
        operands=(raw_barrier, ),
        results=(barrier_issue, ),
        attrs={
            "input_count": 1,
            "projection_domain": (converter_target_ir.EVENT_DOMAIN_BARRIER_ISSUE),
            "projection_provenance": "memory_barrier_successors",
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    assert exc_info.value.code == "TLXW_VERIFY_BARRIER_ORDER_PROVENANCE"


def test_tlx_wave_converter_verifier_requires_compiler_membar_issue_assumption():
    builder = converter_target_ir.TargetBuilder()
    builder.add_op(
        "barrier",
        attrs={
            "address_space": 1,
            "compiler_membar_barrier": True,
            "dependency_count": 0,
            "orders_memory_issue": False,
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    assert exc_info.value.code == "TLXW_VERIFY_BARRIER_ORDER_ASSUMPTION"


def test_tlx_wave_converter_verifier_accepts_wait_order_loop_block_arg():
    builder = converter_target_ir.TargetBuilder()
    ready = builder.add_value(
        converter_target_ir.TargetType("token", "token"),
        event_domain=converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
    )
    body = builder.add_region(block_arg_ids=(ready, ))
    with builder.insertion_region(body):
        builder.add_op(
            "barrier",
            operands=(ready, ),
            attrs={
                "address_space": 1,
                "dependency_count": 1,
                "orders_memory_issue": False,
            },
        )
    builder.add_op("for_loop", region_ids=(body, ))

    assert converter_verifier.verify_target_program(builder.build())


def test_tlx_wave_converter_verifier_accepts_dominating_wait_in_nested_region():
    builder = converter_target_ir.TargetBuilder()
    ready = builder.add_value(
        converter_target_ir.TargetType("token", "token"),
        event_domain=converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
    )
    builder.add_op(
        "async_wait",
        results=(ready, ),
        attrs={
            "completed_group_dependency_count": 0,
            "retained_issue_dependency_count": 0,
        },
    )
    body = builder.add_region()
    with builder.insertion_region(body):
        builder.add_op(
            "barrier",
            operands=(ready, ),
            attrs={
                "address_space": 1,
                "dependency_count": 1,
                "orders_memory_issue": False,
            },
        )
    builder.add_op("for_loop", region_ids=(body, ))

    assert converter_verifier.verify_target_program(builder.build())


def test_tlx_wave_converter_verifier_rejects_dma_as_barrier_lds_read():
    builder = converter_target_ir.TargetBuilder()
    dma_completion = builder.add_value(
        converter_target_ir.TargetType("token", "token"),
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
    )
    builder.add_op(
        "barrier",
        operands=(dma_completion, ),
        attrs={
            "address_space": 1,
            "compiler_membar_barrier": True,
            "dependency_count": 0,
            "lds_read_dependency_count": 1,
            "orders_memory_issue": True,
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    assert exc_info.value.code == "TLXW_VERIFY_ASYNC_PROTOCOL_DOMAIN"


def test_tlx_wave_converter_verifier_rejects_raw_barrier_memory_dependency():
    builder = converter_target_ir.TargetBuilder()
    raw_barrier = builder.add_value(
        converter_target_ir.TargetType("token", "token"),
        event_domain=converter_target_ir.EVENT_DOMAIN_MEMORY_BARRIER,
    )
    builder.add_op(
        "store",
        operands=(raw_barrier, ),
        attrs={"barrier_order_dependency_count": 1},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    assert exc_info.value.code == "TLXW_VERIFY_BARRIER_ORDER_DOMAIN"


def test_tlx_wave_converter_emission_stage_emits_basic_wave_module(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_emit(%arg0: i32, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %sum = arith.addi %arg0, %zero : i32
    %pred = arith.cmpi sge, %sum, %zero : i32
    llvm.intr.assume %pred : i1
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    output = converter_pipeline.convert_ttgir_to_wave(mod)
    emitted = output.emitted_module

    assert "@converter_emit" in emitted.text
    assert "tlx_wave.new_converter" in emitted.text
    assert "gpu.module @kernels" in emitted.text
    assert "gpu.kernel" in emitted.text
    assert "wave.binary" in emitted.text
    assert "wave.assume" in emitted.text
    assert "wave.ptr_add" in emitted.text
    assert "wave_bridge" not in emitted.text
    assert output.target_program.kernel.name == "converter_emit"
    del ctx


def _dense_f32_local_access_attrs():
    return {
        "component_count": 1,
        "coordinate_shape": (64, ),
        "packet_coordinate_bases": ((0, ), ),
        "packet_count": 1,
        "packet_width": 1,
        "slot_coordinate_coefficients": (),
        "workitem_coordinate_coefficients": (
            (1, ),
            (2, ),
            (4, ),
            (8, ),
            (16, ),
            (32, ),
        ),
        "memdesc_logical_origin": (0, ),
        "memdesc_physical_shape": (64, ),
        "memdesc_shape": (64, ),
        "shared_physical_element_byte_width": 4,
        "shared_physical_offset_plan": "dense_row_major",
        "shared_physical_offset_unit": "element",
        "source_shape": (64, ),
        "element_byte_width": 4,
        "element_type": "f32",
        "lane_width": 64,
    }


def _dense_f16_mma_local_access_attrs():
    return {
        "component_count": 1,
        "coordinate_shape": (512, ),
        "packet_coordinate_bases": ((0, ), ),
        "packet_count": 1,
        "packet_width": 2,
        "slot_coordinate_coefficients": ((1, ), ),
        "workitem_coordinate_coefficients": (
            (2, ),
            (4, ),
            (8, ),
            (16, ),
            (32, ),
            (64, ),
            (128, ),
            (256, ),
        ),
        "memdesc_logical_origin": (0, ),
        "memdesc_physical_shape": (512, ),
        "memdesc_shape": (512, ),
        "shared_physical_element_byte_width": 2,
        "shared_physical_offset_plan": "dense_row_major",
        "shared_physical_offset_unit": "element",
        "source_shape": (512, ),
        "element_byte_width": 2,
        "element_type": "f16",
        "lane_width": 64,
    }


def _dense_f32_dma_attrs():
    return {
        "cache_modifier": 1,
        "component_count": 1,
        "destination_component_coordinate_bases": ((0, ), ),
        "destination_coordinate_shape": (64, ),
        "destination_logical_origin": (0, ),
        "destination_offset_mode": "layout_coordinates",
        "destination_physical_element_byte_width": 4,
        "destination_physical_offset_plan": "dense_row_major",
        "destination_physical_offset_unit": "element",
        "destination_physical_shape": (64, ),
        "destination_workitem_coordinate_coefficients": (
            (1, ),
            (2, ),
            (4, ),
            (8, ),
            (16, ),
            (32, ),
        ),
        "element_byte_width": 4,
        "element_type": "f32",
        "has_mask": False,
        "has_stride_operand": False,
        "issue_dependency_count": 0,
        "source_issue_dependency_count": 0,
        "lane_width": 64,
        "mask_mode": "none",
        "mode": "symbolic_copy",
        "range_bytes": 256,
    }


def _dense_f16_symbolic_copy_attrs():
    return {
        "cache_modifier": 1,
        "component_count": 1,
        "destination_component_coordinate_bases": ((0, ), ),
        "destination_coordinate_shape": (512, ),
        "destination_logical_origin": (0, ),
        "destination_offset_mode": "layout_coordinates",
        "destination_physical_element_byte_width": 2,
        "destination_physical_offset_plan": "dense_row_major",
        "destination_physical_offset_unit": "element",
        "destination_physical_shape": (512, ),
        "destination_workitem_coordinate_coefficients": (
            (1, ),
            (2, ),
            (4, ),
            (8, ),
            (16, ),
            (32, ),
        ),
        "element_byte_width": 2,
        "element_type": "f16",
        "has_mask": False,
        "has_stride_operand": False,
        "issue_dependency_count": 0,
        "source_issue_dependency_count": 0,
        "lane_width": 64,
        "mask_mode": "none",
        "mode": "symbolic_copy",
        "range_bytes": 1024,
    }


def _build_two_lds_store_target(*, select_load=False):
    builder = converter_target_ir.TargetBuilder(kernel=converter_target_ir.TargetKernel(
        name="converter_two_lds_tokens",
        num_warps=1,
        threads_per_warp=64,
    ))
    scalar_f32 = converter_target_ir.TargetType("scalar", "scalar", "f32")
    scalar_i1 = converter_target_ir.TargetType("scalar", "scalar", "i1")
    simd_f32 = converter_target_ir.TargetType("tensor", "simd", "f32", 64, 1)
    memdesc_f32 = converter_target_ir.TargetType("memdesc", "memdesc", "f32")
    value = builder.add_value(scalar_f32)
    pred = builder.add_value(scalar_i1)
    alloc_a = builder.add_value(memdesc_f32)
    alloc_b = builder.add_value(memdesc_f32)
    load_a = builder.add_value(simd_f32)
    load_b = builder.add_value(simd_f32)
    selected = builder.add_value(memdesc_f32)
    load_selected = builder.add_value(simd_f32)

    builder.add_op("constant", results=(value, ), attrs={"value": 1.0})
    builder.add_op("constant", results=(pred, ), attrs={"value": True})
    for alloc in (alloc_a, alloc_b):
        builder.add_op(
            "local_alloc",
            results=(alloc, ),
            attrs={
                "align": 16,
                "allocation_bytes": 256,
                "element_type": "f32",
                "shape": (64, ),
            },
        )
    attrs = _dense_f32_local_access_attrs()
    builder.add_op("local_store", operands=(value, alloc_a), attrs=attrs)
    builder.add_op("local_store", operands=(value, alloc_b), attrs=attrs)
    if select_load:
        builder.add_op("select", operands=(pred, alloc_a, alloc_b), results=(selected, ))
        builder.add_op("local_load", operands=(selected, ), results=(load_selected, ), attrs=attrs)
    else:
        builder.add_op("local_load", operands=(alloc_a, ), results=(load_a, ), attrs=attrs)
        builder.add_op("local_load", operands=(alloc_b, ), results=(load_b, ), attrs=attrs)
    builder.add_op("return")
    return builder.build()


def _build_static_memdesc_view_store_target(*, parent_load=False):
    builder = converter_target_ir.TargetBuilder(kernel=converter_target_ir.TargetKernel(
        name="converter_static_memdesc_view_tokens",
        num_warps=1,
        threads_per_warp=64,
    ))
    scalar_f32 = converter_target_ir.TargetType("scalar", "scalar", "f32")
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    simd_f32 = converter_target_ir.TargetType("tensor", "simd", "f32", 64, 1)
    memdesc_f32 = converter_target_ir.TargetType("memdesc", "memdesc", "f32")
    value = builder.add_value(scalar_f32)
    slot0 = builder.add_value(scalar_i32)
    slot1 = builder.add_value(scalar_i32)
    alloc = builder.add_value(memdesc_f32)
    view0 = builder.add_value(memdesc_f32)
    view1 = builder.add_value(memdesc_f32)
    parent_loaded = builder.add_value(simd_f32)

    builder.add_op("constant", results=(value, ), attrs={"value": 1.0})
    builder.add_op("constant", results=(slot0, ), attrs={"value": 0})
    builder.add_op("constant", results=(slot1, ), attrs={"value": 1})
    builder.add_op(
        "local_alloc",
        results=(alloc, ),
        attrs={
            "align": 16,
            "allocation_bytes": 512,
            "element_type": "f32",
            "shape": (2, 64),
        },
    )
    for view, slot, static_byte_offset in ((view0, slot0, 0), (view1, slot1, 256)):
        builder.add_op(
            "memdesc_index",
            operands=(alloc, slot),
            results=(view, ),
            attrs={
                "element_byte_width": 4,
                "elements_per_slot": 64,
                "static_byte_offset": static_byte_offset,
            },
        )
    attrs = _dense_f32_local_access_attrs()
    builder.add_op("local_store", operands=(value, view0), attrs=attrs)
    builder.add_op("local_store", operands=(value, view1), attrs=attrs)
    if parent_load:
        builder.add_op("local_load", operands=(alloc, ), results=(parent_loaded, ), attrs=attrs)
    builder.add_op("return")
    return builder.build()


def _build_circular_memdesc_read_refill_target(
    *,
    slot_count=2,
    slot_stride_bytes=256,
    next_phase=1,
    independent_index=False,
    phase_add_nuw=True,
    phase_add_nsw=False,
):
    builder = converter_target_ir.TargetBuilder(kernel=converter_target_ir.TargetKernel(
        name="converter_circular_memdesc_read_refill",
        num_warps=4,
        threads_per_warp=64,
    ))
    pointer_f32 = converter_target_ir.TargetType("pointer", "uniform_pointer", "f32")
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    simd_i32 = converter_target_ir.TargetType("tensor", "simd", "i32", 64, 1)
    simd_f32 = converter_target_ir.TargetType("tensor", "simd", "f32", 64, 1)
    memdesc_f32 = converter_target_ir.TargetType("memdesc", "memdesc", "f32")
    token = converter_target_ir.TargetType("token", "token")

    source = builder.add_value(pointer_f32)
    phase = builder.add_value(scalar_i32)
    other_phase = builder.add_value(scalar_i32) if independent_index else None
    zero = builder.add_value(scalar_i32)
    depth = builder.add_value(scalar_i32)
    phase_delta = builder.add_value(scalar_i32)
    current_slot = builder.add_value(scalar_i32)
    next_base = builder.add_value(scalar_i32)
    next_slot = builder.add_value(scalar_i32)
    offsets = builder.add_value(simd_i32)
    alloc = builder.add_value(memdesc_f32)
    current_view = builder.add_value(memdesc_f32)
    next_view = builder.add_value(memdesc_f32)
    loaded = builder.add_value(simd_f32)
    refill = builder.add_value(token)

    kernel_args = (source, phase)
    if independent_index:
        kernel_args = (*kernel_args, other_phase)
    builder.set_kernel_arg_targets(kernel_args)
    builder.add_op("constant", results=(zero, ), attrs={"value": 0})
    builder.add_op("constant", results=(depth, ), attrs={"value": int(slot_count)})
    builder.add_op("constant", results=(phase_delta, ), attrs={"value": int(next_phase)})
    builder.add_op(
        "binary",
        operands=(phase, depth),
        results=(current_slot, ),
        attrs={"operation": "remui", "source_width": 32},
    )
    if independent_index:
        next_base_operand = other_phase
    else:
        add_attrs = {"operation": "addi", "source_width": 32}
        if phase_add_nuw:
            add_attrs["nuw"] = True
        if phase_add_nsw:
            add_attrs["nsw"] = True
        builder.add_op(
            "binary",
            operands=(phase, phase_delta),
            results=(next_base, ),
            attrs=add_attrs,
        )
        next_base_operand = next_base
    builder.add_op(
        "binary",
        operands=(next_base_operand, depth),
        results=(next_slot, ),
        attrs={"operation": "remui", "source_width": 32},
    )
    builder.add_op("splat", operands=(zero, ), results=(offsets, ), attrs={"lane_width": 64})
    builder.add_op(
        "local_alloc",
        results=(alloc, ),
        attrs={
            "align": 16,
            "allocation_bytes": int(slot_count) * int(slot_stride_bytes),
            "element_type": "f32",
            "shape": (int(slot_count), int(slot_stride_bytes) // 4),
        },
    )
    view_attrs = {
        "element_byte_width": 4,
        "elements_per_slot": int(slot_stride_bytes) // 4,
        "slot_count": int(slot_count),
        "static_byte_offset": None,
    }
    builder.add_op(
        "memdesc_index",
        operands=(alloc, current_slot),
        results=(current_view, ),
        attrs=view_attrs,
    )
    builder.add_op(
        "memdesc_index",
        operands=(alloc, next_slot),
        results=(next_view, ),
        attrs=view_attrs,
    )
    builder.add_op(
        "local_load",
        operands=(current_view, ),
        results=(loaded, ),
        attrs=_dense_f32_local_access_attrs(),
    )
    builder.add_op(
        "buffer_load_to_local",
        operands=(next_view, source, offsets),
        results=(refill, ),
        attrs=_dense_f32_dma_attrs(),
    )
    builder.add_op("return")
    return builder.build()


def _build_independent_async_dma_wait_target():
    builder = converter_target_ir.TargetBuilder(kernel=converter_target_ir.TargetKernel(
        name="converter_independent_async_dma_wait",
        num_warps=1,
        threads_per_warp=64,
    ))
    pointer_f32 = converter_target_ir.TargetType("pointer", "uniform_pointer", "f32")
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    simd_i32 = converter_target_ir.TargetType("tensor", "simd", "i32", 64, 1)
    memdesc_f32 = converter_target_ir.TargetType("memdesc", "memdesc", "f32")
    token = converter_target_ir.TargetType("token", "token")

    source = builder.add_value(pointer_f32)
    zero = builder.add_value(scalar_i32)
    offsets = builder.add_value(simd_i32)
    allocs = tuple(builder.add_value(memdesc_f32) for _ in range(2))
    dma_tokens = tuple(
        builder.add_value(
            token,
            event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
        ) for _ in range(2))
    group_tokens = tuple(
        builder.add_value(
            token,
            event_domain=converter_target_ir.EVENT_DOMAIN_DMA_GROUP,
        ) for _ in range(2))
    wait_token = builder.add_value(
        token,
        event_domain=converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
    )

    builder.set_kernel_arg_targets((source, ))
    builder.add_op("constant", results=(zero, ), attrs={"value": 0})
    builder.add_op("splat", operands=(zero, ), results=(offsets, ), attrs={"lane_width": 64})
    for alloc in allocs:
        builder.add_op(
            "local_alloc",
            results=(alloc, ),
            attrs={
                "align": 16,
                "allocation_bytes": 256,
                "element_type": "f32",
                "shape": (64, ),
            },
        )
    for alloc, dma_token, group_token in zip(allocs, dma_tokens, group_tokens):
        builder.add_op(
            "buffer_load_to_local",
            operands=(alloc, source, offsets),
            results=(dma_token, ),
            attrs=_dense_f32_dma_attrs(),
        )
        builder.add_op(
            "async_commit_group",
            operands=(dma_token, ),
            results=(group_token, ),
        )
    builder.add_op(
        "async_wait",
        operands=(group_tokens[0], ),
        results=(wait_token, ),
        attrs={
            "completed_group_dependency_count": 1,
            "retained_issue_dependency_count": 0,
        },
    )
    builder.add_op("return")
    return builder.build()


def _build_lds_read_then_scratch_exchange_target(
    *,
    future_load=False,
    conditional_reload=False,
    loop_reload=False,
):
    builder = converter_target_ir.TargetBuilder(
        kernel=converter_target_ir.TargetKernel(
            name="converter_lds_read_then_scratch_exchange",
            num_warps=2,
            threads_per_warp=64,
        ),
        layouts=_cross_wave_target_layouts(),
    )
    scalar_f32 = converter_target_ir.TargetType("scalar", "scalar", "f32")
    scalar_i1 = converter_target_ir.TargetType("scalar", "scalar", "i1")
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    simd_f32 = converter_target_ir.TargetType("tensor", "simd", "f32", 64, 1)
    simd_tuple_f32 = converter_target_ir.TargetType("tensor", "simd_tuple", "f32", 64, 1)
    memdesc_f32 = converter_target_ir.TargetType("memdesc", "memdesc", "f32")
    value = builder.add_value(scalar_f32)
    alloc = builder.add_value(memdesc_f32)
    loaded = builder.add_value(simd_f32, layout_map_id=0)
    converted = builder.add_value(simd_tuple_f32, layout_map_id=1)
    future_loaded = builder.add_value(simd_f32) if future_load else None
    condition = builder.add_value(scalar_i1) if conditional_reload else None
    conditional_loaded = builder.add_value(simd_f32) if conditional_reload else None
    loop_lower = builder.add_value(scalar_i32) if loop_reload else None
    loop_upper = builder.add_value(scalar_i32) if loop_reload else None
    loop_step = builder.add_value(scalar_i32) if loop_reload else None
    loop_induction = builder.add_value(scalar_i32) if loop_reload else None
    loop_loaded = builder.add_value(simd_f32) if loop_reload else None

    builder.add_op(
        "constant",
        results=(value, ),
        attrs={"value": 1.0},
        source_op_index=0,
    )
    builder.add_op(
        "local_alloc",
        results=(alloc, ),
        attrs={
            "align": 16,
            "allocation_bytes": 256,
            "element_type": "f32",
            "shape": (64, ),
        },
        source_op_index=1,
    )
    builder.add_op(
        "local_store",
        operands=(value, alloc),
        attrs=_dense_f32_local_access_attrs(),
        source_op_index=2,
    )
    builder.add_op(
        "local_load",
        operands=(alloc, ),
        results=(loaded, ),
        attrs=_dense_f32_local_access_attrs(),
        source_op_index=3,
    )
    layout_source_index = 4
    if conditional_reload:
        builder.add_op(
            "constant",
            results=(condition, ),
            attrs={"value": True},
            source_op_index=4,
        )
        then_region = builder.add_region()
        else_region = builder.add_region()
        with builder.insertion_region(then_region):
            builder.add_op(
                "local_load",
                operands=(alloc, ),
                results=(conditional_loaded, ),
                attrs=_dense_f32_local_access_attrs(),
                source_op_index=5,
            )
        builder.add_op(
            "if",
            operands=(condition, ),
            region_ids=(then_region, else_region),
            source_op_index=6,
        )
        layout_source_index = 7
    elif loop_reload:
        builder.add_op(
            "constant",
            results=(loop_lower, ),
            attrs={"value": 0},
            source_op_index=4,
        )
        builder.add_op(
            "constant",
            results=(loop_upper, ),
            attrs={"value": 1},
            source_op_index=5,
        )
        builder.add_op(
            "constant",
            results=(loop_step, ),
            attrs={"value": 1},
            source_op_index=6,
        )
        loop_region = builder.add_region(block_arg_ids=(loop_induction, ))
        with builder.insertion_region(loop_region):
            builder.add_op(
                "local_load",
                operands=(alloc, ),
                results=(loop_loaded, ),
                attrs=_dense_f32_local_access_attrs(),
                source_op_index=7,
            )
        builder.add_op(
            "for_loop",
            operands=(loop_lower, loop_upper, loop_step),
            attrs={"init_arg_count": 0, "nonzero_trip": False},
            region_ids=(loop_region, ),
            source_op_index=8,
        )
        layout_source_index = 9
    builder.add_op(
        "layout_convert",
        operands=(loaded, ),
        results=(converted, ),
        attrs={
            "fact_policy": "invalidate_layout_sensitive",
            "transform": "identity",
        },
        source_op_index=layout_source_index,
    )
    if future_load:
        builder.add_op(
            "local_load",
            operands=(alloc, ),
            results=(future_loaded, ),
            attrs=_dense_f32_local_access_attrs(),
            source_op_index=layout_source_index + 1,
        )
    builder.add_op("return")
    return builder.build()


def _build_mma_read_then_dma_reuse_target(
    num_warps=1,
    *,
    add_mfma_boundary=False,
    mfma_boundary_mask=0,
    add_dominating_barrier=False,
    reload_after_boundary=False,
    add_refill=True,
    second_root=False,
    loop_refill=False,
    defer_group_read_to_wait=False,
):
    builder = converter_target_ir.TargetBuilder(kernel=converter_target_ir.TargetKernel(
        name="converter_mma_read_then_dma_reuse",
        num_warps=num_warps,
        threads_per_warp=64,
    ))
    pointer_f16 = converter_target_ir.TargetType("pointer", "uniform_pointer", "f16")
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    simd_i32 = converter_target_ir.TargetType("tensor", "simd", "i32", 64, 1)
    memdesc_f16 = converter_target_ir.TargetType("memdesc", "memdesc", "f16")
    fragment_f16 = converter_target_ir.TargetType("tensor", "simd_packet", "f16", 64, 1)
    token = converter_target_ir.TargetType("token", "token")
    source = builder.add_value(pointer_f16)
    zero = builder.add_value(scalar_i32)
    offset = builder.add_value(simd_i32)
    alloc = builder.add_value(memdesc_f16)
    payload = builder.add_value(fragment_f16)
    second_alloc = builder.add_value(memdesc_f16) if second_root else None
    second_payload = builder.add_value(fragment_f16) if second_root else None
    reloaded_payload = builder.add_value(fragment_f16) if reload_after_boundary else None
    barrier_seed_token = (builder.add_value(
        token,
        event_domain=converter_target_ir.EVENT_DOMAIN_EMPTY,
    ) if add_dominating_barrier else None)
    barrier_wait_token = (builder.add_value(
        token,
        event_domain=converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
    ) if add_dominating_barrier else None)
    dma_token = (builder.add_value(
        token,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
    ) if add_refill else None)
    second_dma_token = (builder.add_value(
        token,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
    ) if add_refill and second_root else None)
    commit_token = (builder.add_value(
        token,
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_GROUP,
    ) if defer_group_read_to_wait else None)
    wait_result_token = (builder.add_value(
        token,
        event_domain=converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
    ) if defer_group_read_to_wait else None)
    loop_lower = builder.add_value(scalar_i32) if loop_refill else None
    loop_upper = builder.add_value(scalar_i32) if loop_refill else None
    loop_step = builder.add_value(scalar_i32) if loop_refill else None
    loop_induction = builder.add_value(scalar_i32) if loop_refill else None
    loop_payload = builder.add_value(fragment_f16) if loop_refill else None

    builder.set_kernel_arg_targets((source, ))
    builder.add_op("constant", results=(zero, ), attrs={"value": 0})
    builder.add_op("splat", operands=(zero, ), results=(offset, ), attrs={"lane_width": 64})
    builder.add_op(
        "local_alloc",
        results=(alloc, ),
        attrs={
            "align": 16,
            "allocation_bytes": 1024,
            "element_type": "f16",
            "shape": (512, ),
        },
    )
    builder.add_op(
        "local_load",
        operands=(alloc, ),
        results=(payload, ),
        attrs=_dense_f16_mma_local_access_attrs(),
    )
    if second_root:
        builder.add_op(
            "local_alloc",
            results=(second_alloc, ),
            attrs={
                "align": 16,
                "allocation_bytes": 1024,
                "element_type": "f16",
                "shape": (512, ),
            },
        )
        builder.add_op(
            "local_load",
            operands=(second_alloc, ),
            results=(second_payload, ),
            attrs=_dense_f16_mma_local_access_attrs(),
        )
    if add_mfma_boundary:
        builder.add_op(
            "sched_barrier",
            attrs={"border": "mfma", "mask": int(mfma_boundary_mask)},
        )
    if add_dominating_barrier:
        builder.add_op("token", results=(barrier_seed_token, ))
        builder.add_op(
            "async_wait",
            operands=(barrier_seed_token, ),
            results=(barrier_wait_token, ),
            attrs={
                "completed_group_dependency_count": 1,
                "retained_issue_dependency_count": 0,
            },
        )
    if reload_after_boundary:
        builder.add_op(
            "local_load",
            operands=(alloc, ),
            results=(reloaded_payload, ),
            attrs=_dense_f16_mma_local_access_attrs(),
        )
    if loop_refill:
        assert add_refill and not second_root
        builder.add_op("constant", results=(loop_lower, ), attrs={"value": 0})
        builder.add_op("constant", results=(loop_upper, ), attrs={"value": 2})
        builder.add_op("constant", results=(loop_step, ), attrs={"value": 1})
        loop_region = builder.add_region(block_arg_ids=(loop_induction, ))
        with builder.insertion_region(loop_region):
            builder.add_op(
                "buffer_load_to_local",
                operands=(alloc, source, offset),
                results=(dma_token, ),
                attrs=_dense_f16_symbolic_copy_attrs(),
            )
            builder.add_op(
                "local_load",
                operands=(alloc, ),
                results=(loop_payload, ),
                attrs=_dense_f16_mma_local_access_attrs(),
            )
        builder.add_op(
            "for_loop",
            operands=(loop_lower, loop_upper, loop_step),
            attrs={"init_arg_count": 0, "nonzero_trip": True},
            region_ids=(loop_region, ),
        )
    elif add_refill:
        refills = [(alloc, dma_token)]
        if second_root:
            refills.append((second_alloc, second_dma_token))
        for destination, result_token in refills:
            builder.add_op(
                "buffer_load_to_local",
                operands=(destination, source, offset),
                results=(result_token, ),
                attrs={
                    **_dense_f16_symbolic_copy_attrs(),
                    **({"async_group_id": 0} if defer_group_read_to_wait else {}),
                },
            )
        if defer_group_read_to_wait:
            assert not second_root
            builder.add_op(
                "async_commit_group",
                operands=(dma_token, ),
                results=(commit_token, ),
                attrs={
                    "group_id": 0,
                    "member_count": 1,
                    "issue_group_size": 0,
                    "issue_delay_cycles": 0,
                    "issue_delay_overlap_cycles": 0,
                    "issue_delay_skip_thread_threshold": 0,
                },
            )
            builder.add_op(
                "async_wait",
                operands=(commit_token, ),
                results=(wait_result_token, ),
                attrs={
                    "wait_group": 1,
                    "waited_group_ids": (),
                    "completed_group_dependency_count": 1,
                    "retained_issue_dependency_count": 0,
                },
                source_op_index=9001,
            )
    builder.add_op("return")
    return builder.build()


def _ssa_result_name(line):
    return line.strip().split("=", 1)[0].strip()


def _ssa_second_result_name(line):
    return line.strip().split("=", 1)[0].split(",")[1].strip()


def _ssa_value_is_used(line, value):
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])",
        line.split("=", 1)[-1],
    ) is not None


def _ssa_after_uses(line, value):
    return re.search(
        rf"\bafter\s+{re.escape(value)}(?![A-Za-z0-9_])",
        line,
    ) is not None


def test_tlx_wave_converter_emission_does_not_infer_distinct_lds_dependencies():
    emitted = converter_emission.emit_wave_module(_build_two_lds_store_target())
    lines = emitted.text.splitlines()
    store_indices = [index for index, line in enumerate(lines) if "wave.scatter" in line]
    assert len(store_indices) == 2
    first_store, second_store = store_indices
    assert "wave.barrier" not in emitted.text
    first_store_token = _ssa_result_name(lines[first_store])
    second_store_token = _ssa_result_name(lines[second_store])
    load_lines = [line for line in lines if "wave.gather" in line]
    assert len(load_lines) == 2
    for line in load_lines:
        dependency = re.search(r"\bafter (?P<token>%[\w#]+)", line)
        assert dependency is not None
        assert not _wave_token_depends_on(
            emitted.text,
            dependency.group("token"),
            first_store_token,
        )
        assert not _wave_token_depends_on(
            emitted.text,
            dependency.group("token"),
            second_store_token,
        )
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_does_not_infer_selected_memdesc_alias():
    emitted = converter_emission.emit_wave_module(_build_two_lds_store_target(select_load=True))
    lines = emitted.text.splitlines()
    load_line = next(line for line in lines if "wave.gather" in line)
    store_tokens = [_ssa_result_name(line) for line in lines if "wave.scatter" in line]
    assert len(store_tokens) == 2
    assert "wave.barrier" not in emitted.text
    load_dependency = re.search(r"\bafter (?P<token>%[\w#]+)", load_line)
    assert load_dependency is not None
    assert all(not _wave_token_depends_on(
        emitted.text,
        load_dependency.group("token"),
        store_token,
    ) for store_token in store_tokens)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_keeps_disjoint_static_memdesc_views_independent():
    emitted = converter_emission.emit_wave_module(_build_static_memdesc_view_store_target())
    lines = emitted.text.splitlines()
    store_indices = [index for index, line in enumerate(lines) if "wave.scatter" in line]
    assert len(store_indices) == 2
    first_store, second_store = store_indices
    assert "wave.barrier" not in emitted.text
    assert f"after {_ssa_result_name(lines[first_store])}" not in lines[second_store]
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_does_not_infer_parent_view_alias():
    emitted = converter_emission.emit_wave_module(_build_static_memdesc_view_store_target(parent_load=True))
    lines = emitted.text.splitlines()
    load_line = next(line for line in lines if "wave.gather" in line)
    store_tokens = [_ssa_result_name(line) for line in lines if "wave.scatter" in line]
    assert len(store_tokens) == 2
    assert "wave.barrier" not in emitted.text
    load_dependency = re.search(r"\bafter (?P<token>%[\w#]+)", load_line)
    assert load_dependency is not None
    assert all(not _wave_token_depends_on(
        emitted.text,
        load_dependency.group("token"),
        store_token,
    ) for store_token in store_tokens)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_verifier_rejects_dma_lds_release_segment():
    builder = converter_target_ir.TargetBuilder()
    memdesc = builder.add_value(converter_target_ir.TargetType("memdesc", "memdesc", "f32"), )
    source = builder.add_value(converter_target_ir.TargetType(
        "pointer",
        "uniform_pointer",
        "f32",
    ), )
    offsets = builder.add_value(converter_target_ir.TargetType(
        "tensor",
        "simd",
        "index",
        64,
        1,
    ), )
    completion = builder.add_value(
        converter_target_ir.TargetType("token", "token"),
        event_domain=converter_target_ir.EVENT_DOMAIN_DMA_COMPLETION,
    )
    builder.set_kernel_arg_targets((memdesc, source, offsets))
    builder.add_op(
        "buffer_load_to_local",
        operands=(memdesc, source, offsets),
        results=(completion, ),
        attrs={
            "issue_dependency_count": 0,
            "source_issue_dependency_count": 0,
            "lds_release_dependency_count": 0,
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    assert exc_info.value.code == "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS"


@pytest.mark.parametrize(
    "slot_count,slot_stride_bytes,next_phase",
    ((2, 256, 1), (3, 272, 2), (4, 320, 3)),
)
def test_tlx_wave_converter_emission_keeps_disjoint_circular_slots_independent(
    slot_count,
    slot_stride_bytes,
    next_phase,
):
    emitted = converter_emission.emit_wave_module(
        _build_circular_memdesc_read_refill_target(
            slot_count=slot_count,
            slot_stride_bytes=slot_stride_bytes,
            next_phase=next_phase,
        ))

    assert "wave.barrier" not in emitted.text
    load_line = next(line for line in emitted.text.splitlines() if "wave.gather" in line)
    copy_line = next(line for line in emitted.text.splitlines() if "wave.gather" in line and "#waveamd.buffer" in line)
    assert not _ssa_value_is_used(copy_line, _ssa_second_result_name(load_line))
    _run_wave_verify(emitted.text)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"next_phase": 0},
        {"independent_index": True},
        {"phase_add_nuw": False, "phase_add_nsw": True},
    ),
)
def test_tlx_wave_converter_emission_does_not_infer_dma_dependency_for_unproven_circular_slots(kwargs):
    emitted = converter_emission.emit_wave_module(_build_circular_memdesc_read_refill_target(**kwargs))
    lines = emitted.text.splitlines()
    load_index = next(index for index, line in enumerate(lines) if "wave.gather" in line)
    copy_index = next(index for index, line in enumerate(lines) if "wave.gather" in line and "#waveamd.buffer" in line)

    read_token = _ssa_second_result_name(lines[load_index])
    assert load_index < copy_index
    assert "wave.barrier" not in emitted.text
    assert not _ssa_value_is_used(lines[copy_index], read_token)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_waits_only_explicit_async_dma_groups():
    emitted = converter_emission.emit_wave_module(_build_independent_async_dma_wait_target())
    join_lines = [line for line in emitted.text.splitlines() if "wave.join" in line]
    wait_lines = [line for line in emitted.text.splitlines() if "wave.after" in line]

    assert len(join_lines) == 2
    assert len(wait_lines) == 1
    assert "wave.barrier" not in emitted.text
    waited_group_token, live_group_token = map(_ssa_result_name, join_lines)
    assert waited_group_token in wait_lines[0]
    assert live_group_token not in wait_lines[0]
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_barriers_pending_lds_read_before_redistribute():
    emitted = converter_emission.emit_wave_module(_build_lds_read_then_scratch_exchange_target())
    lines = emitted.text.splitlines()
    load_index = next(index for index, line in enumerate(lines) if "wave.gather" in line)
    redistribute_index = next(index for index, line in enumerate(lines) if "wave.redistribute" in line)
    assert load_index < redistribute_index
    assert "wave.alloc_release" not in emitted.text
    _run_wave_verify(emitted.text)
    _run_waveamd_to_machine(emitted.text)


def test_tlx_wave_converter_emission_keeps_live_lds_before_redistribute():
    emitted = converter_emission.emit_wave_module(_build_lds_read_then_scratch_exchange_target(future_load=True))
    lines = emitted.text.splitlines()
    redistribute_index = next(index for index, line in enumerate(lines) if "wave.redistribute" in line)

    assert not any("wave.alloc_release" in line for line in lines[:redistribute_index])
    assert "wave.alloc_release" not in emitted.text
    _run_wave_verify(emitted.text)
    _run_waveamd_to_machine(emitted.text)


def test_tlx_wave_converter_emission_merges_conditional_lds_access_before_release():
    emitted = converter_emission.emit_wave_module(_build_lds_read_then_scratch_exchange_target(conditional_reload=True))
    lines = emitted.text.splitlines()
    if_index = next(index for index, line in enumerate(lines) if "scf.if" in line)
    redistribute_index = next(index for index, line in enumerate(lines) if "wave.redistribute" in line)

    assert if_index < redistribute_index
    assert "wave.alloc_release" not in emitted.text
    _run_wave_verify(emitted.text)
    _run_waveamd_to_machine(emitted.text)


def test_tlx_wave_converter_emission_carries_loop_lds_history_before_release():
    emitted = converter_emission.emit_wave_module(_build_lds_read_then_scratch_exchange_target(loop_reload=True))
    lines = emitted.text.splitlines()
    loop_index = next(index for index, line in enumerate(lines) if "scf.for" in line)
    redistribute_index = next(index for index, line in enumerate(lines) if "wave.redistribute" in line)

    assert loop_index < redistribute_index
    assert "wave.alloc_release" not in emitted.text
    _run_wave_verify(emitted.text)
    _run_waveamd_to_machine(emitted.text)


def test_tlx_wave_converter_emission_does_not_add_mma_read_dependency_to_dma():
    emitted = converter_emission.emit_wave_module(_build_mma_read_then_dma_reuse_target())
    lines = emitted.text.splitlines()
    load_index = next(index for index, line in enumerate(lines) if "wave.gather" in line)
    copy_index = next(index for index, line in enumerate(lines) if "wave.gather" in line and "#waveamd.buffer" in line)
    read_token = _ssa_second_result_name(lines[load_index])

    assert "wave.barrier" not in "\n".join(lines[load_index + 1:copy_index])
    assert not _ssa_value_is_used(lines[copy_index], read_token)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_does_not_barrier_multi_wave_mma_read_before_dma():
    emitted = converter_emission.emit_wave_module(_build_mma_read_then_dma_reuse_target(num_warps=4))
    lines = emitted.text.splitlines()
    load_index = next(index for index, line in enumerate(lines) if "wave.gather" in line)
    copy_index = next(index for index, line in enumerate(lines) if "wave.gather" in line and "#waveamd.buffer" in line)
    read_token = _ssa_second_result_name(lines[load_index])

    assert emitted.text.count("wave.barrier") == 0
    assert load_index < copy_index
    assert not _ssa_value_is_used(lines[copy_index], read_token)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_does_not_carry_mma_read_dependency_to_loop_dma():
    emitted = converter_emission.emit_wave_module(_build_mma_read_then_dma_reuse_target(
        num_warps=8,
        loop_refill=True,
    ))
    lines = emitted.text.splitlines()
    load_index = next(index for index, line in enumerate(lines) if "wave.gather" in line)
    loop_index = next(index for index, line in enumerate(lines) if "scf.for" in line)
    copy_index = next(index for index, line in enumerate(lines)
                      if index > loop_index and "wave.gather" in line and "#waveamd.buffer" in line)
    read_token = _ssa_second_result_name(lines[load_index])

    assert load_index < loop_index < copy_index
    assert "iter_args" not in lines[loop_index]
    assert "!wave.mem.token" not in lines[loop_index]
    assert read_token not in lines[loop_index]
    assert "wave.barrier" not in emitted.text
    assert not _ssa_value_is_used(lines[copy_index], read_token)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_keeps_group_wait_separate_from_mma_read_frontier():
    emitted = converter_emission.emit_wave_module(
        _build_mma_read_then_dma_reuse_target(
            num_warps=8,
            defer_group_read_to_wait=True,
        ))
    lines = emitted.text.splitlines()
    load_index = next(index for index, line in enumerate(lines) if "wave.gather" in line)
    copy_index = next(index for index, line in enumerate(lines) if "wave.gather" in line and "#waveamd.buffer" in line)
    wait_indices = [index for index, line in enumerate(lines) if "wave.after" in line]
    (wait_index, ) = wait_indices
    read_token = _ssa_second_result_name(lines[load_index])
    wait_token = _ssa_result_name(lines[wait_index])
    commit_token = _ssa_result_name(next(line for line in lines if "wave.join" in line))

    assert "wave.barrier" not in emitted.text
    assert load_index < copy_index < wait_index
    assert not _ssa_value_is_used(lines[copy_index], read_token)
    assert f"{wait_token} = wave.after {commit_token}" in lines[wait_index]
    assert not _wave_token_depends_on(emitted.text, wait_token, read_token)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_does_not_materialize_mfma_boundary_for_dma():
    emitted = converter_emission.emit_wave_module(
        _build_mma_read_then_dma_reuse_target(
            num_warps=4,
            add_mfma_boundary=True,
        ))
    lines = emitted.text.splitlines()
    load_index = next(index for index, line in enumerate(lines) if "wave.gather" in line)
    copy_index = next(index for index, line in enumerate(lines) if "wave.gather" in line and "#waveamd.buffer" in line)
    read_token = _ssa_second_result_name(lines[load_index])

    assert load_index < copy_index
    assert "wave.barrier" not in emitted.text
    assert not _ssa_value_is_used(lines[copy_index], read_token)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_drops_unused_mfma_boundary():
    emitted = converter_emission.emit_wave_module(
        _build_mma_read_then_dma_reuse_target(
            num_warps=4,
            add_mfma_boundary=True,
            add_refill=False,
        ))

    assert "wave.barrier" not in emitted.text
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_does_not_materialize_mfma_boundaries_for_dma_roots():
    emitted = converter_emission.emit_wave_module(
        _build_mma_read_then_dma_reuse_target(
            num_warps=4,
            add_mfma_boundary=True,
            second_root=True,
        ))
    lines = emitted.text.splitlines()
    copy_lines = [line for line in lines if "wave.gather" in line and "#waveamd.buffer" in line]
    read_tokens = [_ssa_second_result_name(line) for line in lines if "wave.gather" in line and "#wave.shared" in line]

    assert emitted.text.count("wave.barrier") == 0
    assert len(copy_lines) == 2
    assert all(all(not _ssa_value_is_used(line, token) for token in read_tokens) for line in copy_lines)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_does_not_use_dominating_wait_as_dma_dependency():
    emitted = converter_emission.emit_wave_module(
        _build_mma_read_then_dma_reuse_target(
            num_warps=4,
            add_mfma_boundary=True,
            add_dominating_barrier=True,
            second_root=True,
        ))
    lines = emitted.text.splitlines()
    wait_lines = [line for line in lines if "wave.after" in line]
    copy_lines = [line for line in lines if "wave.gather" in line and "#waveamd.buffer" in line]

    assert len(wait_lines) == 1
    assert len(copy_lines) == 2
    assert "wave.barrier" not in emitted.text
    wait_token = _ssa_result_name(wait_lines[0])
    assert all(f"after {wait_token}" not in line for line in copy_lines)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_does_not_add_dma_wait_after_new_mma_read():
    emitted = converter_emission.emit_wave_module(
        _build_mma_read_then_dma_reuse_target(
            num_warps=4,
            add_mfma_boundary=True,
            add_dominating_barrier=True,
            reload_after_boundary=True,
        ))
    lines = emitted.text.splitlines()
    load_indices = [index for index, line in enumerate(lines) if "wave.gather" in line and "#wave.shared" in line]
    wait_indices = [index for index, line in enumerate(lines) if "wave.after" in line]
    copy_index = next(index for index, line in enumerate(lines) if "wave.gather" in line and "#waveamd.buffer" in line)

    assert len(load_indices) == 2
    assert len(wait_indices) == 1
    assert load_indices[0] < wait_indices[0] < load_indices[1] < copy_index
    second_read_token = _ssa_second_result_name(lines[load_indices[1]])
    wait_token = _ssa_result_name(lines[wait_indices[0]])
    assert not _ssa_value_is_used(lines[copy_index], second_read_token)
    assert f"after {wait_token}" not in lines[copy_index]
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_drops_mfma_boundary_before_dma_after_new_read():
    emitted = converter_emission.emit_wave_module(
        _build_mma_read_then_dma_reuse_target(
            num_warps=4,
            add_mfma_boundary=True,
            reload_after_boundary=True,
        ))
    lines = emitted.text.splitlines()
    load_indices = [index for index, line in enumerate(lines) if "wave.gather" in line and "#wave.shared" in line]
    copy_index = next(index for index, line in enumerate(lines) if "wave.gather" in line and "#waveamd.buffer" in line)
    second_read_token = _ssa_second_result_name(lines[load_indices[-1]])

    assert emitted.text.count("wave.barrier") == 0
    assert load_indices[0] < load_indices[-1] < copy_index
    assert not _ssa_value_is_used(lines[copy_index], second_read_token)
    _run_wave_verify(emitted.text)


def test_tlx_wave_converter_emission_ignores_masked_mfma_boundary():
    emitted = converter_emission.emit_wave_module(
        _build_mma_read_then_dma_reuse_target(
            num_warps=4,
            add_mfma_boundary=True,
            mfma_boundary_mask=1,
        ))
    lines = emitted.text.splitlines()
    load_index = next(index for index, line in enumerate(lines) if "wave.gather" in line)
    copy_index = next(index for index, line in enumerate(lines) if "wave.gather" in line and "#waveamd.buffer" in line)
    read_token = _ssa_second_result_name(lines[load_index])

    assert emitted.text.count("wave.barrier") == 0
    assert load_index < copy_index
    assert not _ssa_value_is_used(lines[copy_index], read_token)
    _run_wave_verify(emitted.text)


@pytest.mark.parametrize("prestore", [False, True])
def test_tlx_wave_converter_does_not_carry_implicit_lds_token_across_for(
    tmp_path,
    prestore,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    store = "ttg.local_store %value, %alloc : tensor<64xf32, #blocked> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>"
    local_func = "\n".join([
        "  tt.func public @converter_loop_carried_local_token(%value: tensor<64xf32, #blocked>) attributes {noinline = false} {",
        "    %c0 = arith.constant 0 : index",
        "    %c1 = arith.constant 1 : index",
        "    %c2 = arith.constant 2 : index",
        "    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>",
        f"    {store}" if prestore else "",
        "    scf.for %i = %c0 to %c2 step %c1 {",
        f"      {store}",
        "    }",
        "    %loaded = ttg.local_load %alloc : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked>",
        "    tt.return",
        "  }",
    ])
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (loop_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    assert converter_target_ir.attrs_dict(loop_op)["init_arg_count"] == 0
    assert not loop_op.results
    local_ops = [op for op in output.target_program.ops if op.kind in {"local_load", "local_store"}]
    assert all(converter_target_ir.attrs_dict(op)["explicit_dependency_count"] == 0 for op in local_ops)
    assert all(len(op.operands) == (2 if op.kind == "local_store" else 1) for op in local_ops)
    wave = output.emitted_module.text
    loop_line = next(line for line in wave.splitlines() if "scf.for" in line)
    assert "iter_args" not in loop_line
    assert "wave.barrier" not in wave
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_emits_workgroup_shape_from_ttgir(tmp_path):
    local_func = """
  tt.func public @converter_workgroup_shape() attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=4,
        threads_per_warp=64,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    wave_artifact = output.emitted_module.text

    assert "wave.workgroup_size = array<i32: 256, 1, 1>" in wave_artifact
    assert "gpu.known_block_size = array<i32: 256, 1, 1>" in wave_artifact
    assert "wave.waves_per_workgroup = 4 : i64" in wave_artifact
    assert "waveamdmachine.target_waves = 1 : i64" in wave_artifact
    del ctx


@pytest.mark.parametrize(
    "waves_per_eu,expected_target_waves",
    [(1, 2), (4, 4)],
)
def test_tlx_wave_converter_applies_waves_per_eu_target(
    tmp_path,
    waves_per_eu,
    expected_target_waves,
):
    local_func = """
  tt.func public @converter_waves_per_eu() attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=8,
        threads_per_warp=64,
    )

    output = converter_pipeline.convert_ttgir_to_wave(
        mod,
        waves_per_eu=waves_per_eu,
    )

    assert (f"waveamdmachine.target_waves = {expected_target_waves} : i64" in output.emitted_module.text)
    del ctx


def test_tlx_wave_backend_wave_stage_uses_staged_converter(tmp_path, monkeypatch):
    local_func = """
  tt.func public @backend_wave_stage(%arg0: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %sum = arith.addi %arg0, %zero : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)
    metadata = {}

    wave_artifact = tlx_wave_compiler.TLXWaveBackend.make_wave(
        mod,
        metadata,
        _tlx_wave_options(),
    )

    assert "tlx_wave.new_converter" in wave_artifact
    assert "gpu.module @kernels" in wave_artifact
    assert "gpu.kernel" in wave_artifact
    assert "wave_bridge" not in wave_artifact
    assert metadata["name"] == "backend_wave_stage"
    assert metadata["tlx_wave_status"] == "emitted_wave_staged_converter"
    assert metadata["tlx_wave_bridge_stage"] == "staged-converter"
    assert metadata["tlx_wave_wave_builder"] == "staged-converter"
    assert metadata["tlx_wave_plan_kind"] == "staged-converter"
    assert metadata["tlx_wave_ttgir_target"] == "hip:gfx950"
    assert metadata["tlx_wave_num_kernel_args"] == 1
    assert metadata["tlx_wave_num_scalar_args"] == 1
    assert metadata["tlx_wave_num_pointer_args"] == 0
    assert metadata["tlx_wave_workgroup_size"] == 64
    _run_wave_verify(wave_artifact)
    del ctx


def test_tlx_wave_backend_wave_stage_forwards_waves_per_eu(tmp_path):
    local_func = """
  tt.func public @backend_wave_waves_per_eu() attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8)

    wave_artifact = tlx_wave_compiler.TLXWaveBackend.make_wave(
        mod,
        {},
        _tlx_wave_options(waves_per_eu=4),
    )

    assert "waveamdmachine.target_waves = 4 : i64" in wave_artifact
    _run_wave_verify(wave_artifact)
    del ctx


def test_tlx_wave_backend_wave_stage_emits_split_barrier_attr(tmp_path):
    local_func = """
  tt.func public @backend_wave_split_barriers() attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8)
    metadata = {}

    wave_artifact = tlx_wave_compiler.TLXWaveBackend.make_wave(
        mod,
        metadata,
        _tlx_wave_options(tlx_wave_enable_split_barriers=True),
    )

    assert "waveamdmachine.enable_split_barriers" in wave_artifact
    assert "waveamdmachine.enable_multi_wave_specialization" not in wave_artifact
    assert metadata["tlx_wave_enable_split_barriers"] is True
    _run_wave_verify(wave_artifact)
    del ctx


def test_tlx_wave_backend_wave_stage_emits_multi_wave_specialization_attr(tmp_path):
    local_func = """
  tt.func public @backend_wave_multi_wave_specialization() attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8)
    metadata = {}

    wave_artifact = tlx_wave_compiler.TLXWaveBackend.make_wave(
        mod,
        metadata,
        _tlx_wave_options(tlx_wave_enable_multi_wave_specialize=True),
    )

    assert "waveamdmachine.enable_multi_wave_specialization" in wave_artifact
    assert "waveamdmachine.enable_split_barriers" not in wave_artifact
    assert metadata["tlx_wave_enable_multi_wave_specialize"] is True
    _run_wave_verify(wave_artifact)
    del ctx


@pytest.mark.parametrize(
    "source_attr,kernel_field,wave_attr,other_wave_attr",
    [
        (
            "tlx_wave.enable_split_barriers",
            "enable_split_barriers",
            "waveamdmachine.enable_split_barriers",
            "waveamdmachine.enable_multi_wave_specialization",
        ),
        (
            "tlx_wave.enable_multi_wave_specialization",
            "enable_multi_wave_specialization",
            "waveamdmachine.enable_multi_wave_specialization",
            "waveamdmachine.enable_split_barriers",
        ),
    ],
)
def test_tlx_wave_converter_carries_function_policy_attr(
    tmp_path,
    source_attr,
    kernel_field,
    wave_attr,
    other_wave_attr,
):
    local_func = f"""
  tt.func public @backend_wave_function_policy() attributes {{
    noinline = false,
    {source_attr} = true
  }} {{
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert getattr(output.source_program.kernel, kernel_field) is True
    assert getattr(output.target_program.kernel, kernel_field) is True
    assert wave_attr in output.emitted_module.text
    assert other_wave_attr not in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_backend_honors_function_specialization_without_global_option(tmp_path, ):
    local_func = """
  tt.func public @backend_wave_function_specialization() attributes {
    noinline = false,
    tlx_wave.enable_multi_wave_specialization = true
  } {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8)
    metadata = {}

    wave_artifact = tlx_wave_compiler.TLXWaveBackend.make_wave(
        mod,
        metadata,
        _tlx_wave_options(),
    )

    assert "waveamdmachine.enable_multi_wave_specialization" in wave_artifact
    assert metadata["tlx_wave_enable_multi_wave_specialize"] is True
    _run_wave_verify(wave_artifact)
    del ctx


def test_tlx_wave_backend_wave_stage_keeps_fixed_lds_out_of_launch_shared(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @backend_wave_stage_fixed_lds() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    ttg.local_store %value, %alloc : tensor<64xf16, #blocked> -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    metadata = {}

    wave_artifact = tlx_wave_compiler.TLXWaveBackend.make_wave(
        mod,
        metadata,
        _tlx_wave_options(),
    )

    assert "wave.alloc" in wave_artifact
    assert "wave.lds_size" not in wave_artifact
    assert metadata["shared"] == 0
    assert metadata["tlx_wave_launch_shared_bytes"] == 0
    assert metadata["tlx_wave_lds_size_bytes"] == 0
    _run_wave_verify(wave_artifact)
    del ctx


def test_tlx_wave_backend_hash_includes_wave_opt_sha(monkeypatch):
    backend = tlx_wave_compiler.TLXWaveBackend(GFX950_WAVE)
    first_sha = "1" * 64
    second_sha = "2" * 64

    monkeypatch.setattr(tlx_wave_compiler, "_wave_opt_sha256", lambda: first_sha)
    first_hash = backend.hash()

    monkeypatch.setattr(tlx_wave_compiler, "_wave_opt_sha256", lambda: second_sha)
    second_hash = backend.hash()

    assert "stage15-symbolic-arithmetic-contract" in first_hash
    assert f"wave-opt-sha256={first_sha}" in first_hash
    assert f"wave-opt-sha256={second_sha}" in second_hash
    assert first_hash != second_hash


def test_tlx_wave_driver_load_binary_delegates_hsaco():
    calls = []

    class FakeHIPUtils:

        def load_binary(self, name, kernel, shared, device):
            calls.append((name, kernel, shared, device))
            return "module", "function", 10, 0, 1024

    utils = tlx_wave_driver._TLXWaveUtils(FakeHIPUtils())

    result = utils.load_binary("kernel_name", b"\x7fELFpayload", 128, 0)

    assert result == ("module", "function", 10, 0, 1024)
    assert calls == [("kernel_name", b"\x7fELFpayload", 128, 0)]


def test_tlx_wave_driver_load_binary_rejects_wave_text():

    class FakeHIPUtils:

        def load_binary(self, name, kernel, shared, device):
            raise AssertionError("non-HSACO artifact should not reach HIP")

    utils = tlx_wave_driver._TLXWaveUtils(FakeHIPUtils())

    with pytest.raises(RuntimeError, match="expected HSACO bytes"):
        utils.load_binary("kernel_name", "gpu.module @kernels {}", 0, 0)

    with pytest.raises(RuntimeError, match="expected an ELF HSACO object"):
        utils.load_binary("kernel_name", b"gpu.module @kernels {}", 0, 0)


def test_tlx_wave_runtime_skip_reason_is_test_only_targeting_policy():
    assert _TLX_WAVE_RUNTIME_ARCHES == {"gfx942", "gfx950"}
    assert _tlx_wave_physical_arch({"arch": "gfx1100:sramecc+:xnack-"}) == "gfx1100"

    reason = _tlx_wave_runtime_skip_reason("gfx1100")

    assert "physical gfx942/gfx950 hardware" in reason
    assert "runtime launch guard" in reason
    assert "not a Wave HSACO generation failure" in reason
    assert "Compile-only TLX Wave tests may target gfx942/gfx950" in reason


def test_tlx_wave_backend_compile_uses_staged_converter():
    src = ASTSource(fn=_tlx_wave_stage_only_kernel, signature={}, constexprs={})

    compiled = triton_compile(src, target=GFX950_WAVE)
    wave_artifact = _asm_text(compiled, "wave")
    hsaco = compiled.asm["hsaco"]

    assert "tlx_wave.new_converter" in wave_artifact
    assert "gpu.module @kernels" in wave_artifact
    assert "gpu.kernel" in wave_artifact
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert compiled.kernel == hsaco
    assert compiled.metadata.tlx_wave_status == "emitted_wave_staged_converter"
    assert compiled.metadata.tlx_wave_binary_stage == "wave-compile-kernels"
    assert compiled.metadata.tlx_wave_hsaco_size_bytes == len(hsaco)
    assert compiled.metadata.tlx_wave_bridge_stage == "staged-converter"
    assert compiled.metadata.tlx_wave_wave_builder == "staged-converter"
    assert compiled.metadata.tlx_wave_plan_kind == "staged-converter"
    assert compiled.metadata.tlx_wave_num_kernel_args == 0
    assert compiled.metadata.tlx_wave_plan_num_ops >= 1
    _run_wave_verify(wave_artifact)
    binary_module = _run_wave_compile_kernels(wave_artifact)
    assert "gpu.binary @kernels" in binary_module
    assert "gpu.module @kernels" not in binary_module


def test_tlx_wave_backend_compile_forwards_waves_per_eu():
    src = ASTSource(fn=_tlx_wave_stage_only_kernel, signature={}, constexprs={})

    compiled = triton_compile(
        src,
        target=GFX950_WAVE,
        options={"num_warps": 8, "waves_per_eu": 4},
    )
    wave_artifact = _asm_text(compiled, "wave")

    assert "waveamdmachine.target_waves = 4 : i64" in wave_artifact
    assert compiled.metadata.waves_per_eu == 4


def test_tlx_wave_backend_compile_lowers_masked_global_load_store():
    src = ASTSource(
        fn=_tlx_wave_add_one_kernel,
        signature={
            "x": "*fp32",
            "y": "*fp32",
            "n": "i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": 64},
        attrs={
            (0, ): [["tt.pointer_range", 32]],
            (1, ): [["tt.pointer_range", 32]],
        },
    )

    compiled = triton_compile(src, target=GFX950_WAVE)
    wave_artifact = _asm_text(compiled, "wave")
    hsaco = compiled.asm["hsaco"]

    assert wave_artifact.count("wave.where") == 2
    assert "otherwise" in wave_artifact
    assert "wave.select" not in wave_artifact
    assert "wave.fadd" in wave_artifact
    assert "wave.gather" in wave_artifact
    assert "wave.scatter" in wave_artifact
    assert "waveamd.make_buffer" in wave_artifact
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert compiled.metadata.tlx_wave_status == "emitted_wave_staged_converter"
    assert compiled.metadata.tlx_wave_binary_stage == "wave-compile-kernels"


def test_tlx_gfx9_gemm_bench_parses_shapes_and_defaults():
    bench = _load_tlx_gfx9_gemm_bench_module()

    assert not hasattr(bench, "DEVICE")
    assert bench.provider_defaults(9) == ["tlx", "wave"]
    assert bench.provider_defaults(10) == ["wave"]
    assert bench.provider_defaults(11) == ["wave"]
    assert bench.provider_defaults(0) == ["rocblas", "tlx"]
    assert bench.parse_shape("128x256x64") == (128, 256, 64)
    assert bench.parse_shape("128,256,64") == (128, 256, 64)
    with pytest.raises(Exception, match="shape dimensions must be positive"):
        bench.parse_shape("128x0x64")
    with pytest.raises(Exception, match="shape must be MxNxK"):
        bench.parse_shape("128x256")
    bench.validate_shape_for_providers((256, 256, 64), 0, ["tlx"])
    bench.validate_shape_for_providers((128, 128, 64), 9, ["rocblas"])
    with pytest.raises(Exception, match="M to be a multiple of 256"):
        bench.validate_shape_for_providers((128, 256, 64), 9, ["wave"])
    with pytest.raises(Exception, match="N to be a multiple of 256"):
        bench.validate_shape_for_providers((256, 128, 64), 9, ["tlx"])
    with pytest.raises(Exception, match="K to be a multiple of 64"):
        bench.validate_shape_for_providers((256, 256, 96), 2, ["tlx"])
    with pytest.raises(Exception, match="prefetch two 64-wide K tiles"):
        bench.validate_shape_for_providers((256, 256, 64), 9, ["tlx", "wave"])
    bench.validate_shape_for_providers((256, 256, 128), 9, ["tlx", "wave"])


def test_tlx_gfx9_gemm_bench_input_modes_are_deterministic():
    torch = pytest.importorskip("torch")
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_wave_test_gfx9_bench_inputs")
    inter_wave = _load_tlx_gfx9_inter_wave_bench_module("_tlx_wave_test_gfx9_inter_wave_bench_inputs")
    assert inter_wave.INPUT_MODES == bench.INPUT_MODES
    normal_seed_zero = None

    for input_mode in bench.INPUT_MODES:
        a, b = bench.make_inputs(
            2,
            4,
            8,
            torch.device("cpu"),
            "transposed",
            input_mode=input_mode,
            seed=0,
        )
        repeat_a, repeat_b = bench.make_inputs(
            2,
            4,
            8,
            torch.device("cpu"),
            "transposed",
            input_mode=input_mode,
            seed=0,
        )
        torch.testing.assert_close(a, repeat_a)
        torch.testing.assert_close(b, repeat_b)
        inter_wave_a, inter_wave_b = inter_wave.make_inputs(
            2,
            4,
            8,
            torch.device("cpu"),
            "transposed",
            input_mode=input_mode,
            seed=0,
        )
        torch.testing.assert_close(inter_wave_a, a)
        torch.testing.assert_close(inter_wave_b, b)
        assert b.shape == (8, 4)
        assert b.stride() == (1, 8)
        if input_mode == "normal":
            normal_seed_zero = a

    normal_a, _ = bench.make_inputs(2, 4, 8, "cpu", "transposed", input_mode="normal", seed=1)
    assert not torch.equal(normal_seed_zero, normal_a)


def test_tlx_gfx9_gemm_bench_reproduces_wave_rand_int_inputs():
    torch = pytest.importorskip("torch")
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_wave_test_gfx9_bench_rand_int")

    a, b = bench.make_inputs(
        2,
        4,
        8,
        torch.device("cpu"),
        "transposed",
        input_mode="rand-int",
        seed=0,
    )

    expected_a = torch.tensor(
        [
            [-2, -2, 0, 0, 1, 0, 1, 2],
            [-1, -2, 0, -1, -2, -2, 0, -1],
        ],
        dtype=torch.float16,
    )
    expected_b_storage = torch.tensor(
        [
            [2, -2, 0, 0, -1, 0, -1, 2],
            [-1, 2, 0, 1, -2, 2, 0, 1],
            [2, 0, -2, -1, 1, 2, 0, 2],
            [2, -1, -1, 1, 2, 2, 1, 2],
        ],
        dtype=torch.float16,
    )
    torch.testing.assert_close(a, expected_a)
    torch.testing.assert_close(b.T, expected_b_storage)


def test_tlx_gfx9_gemm_bench_launch_reuses_output():
    torch = pytest.importorskip("torch")
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_wave_test_gfx9_bench_output")
    call = {}

    class FakeKernel:

        def __getitem__(self, grid):
            call["grid"] = grid

            def launch(*args, **kwargs):
                call["args"] = args
                call["kwargs"] = kwargs

            return launch

    module = SimpleNamespace(v9_beyond_hotloop=FakeKernel())
    a = torch.empty((256, 128), dtype=torch.float16)
    b = torch.empty((128, 256), dtype=torch.float16)
    out = torch.empty((256, 256), dtype=torch.float16)

    result = bench.launch_tutorial_matmul(module, "v9_beyond_hotloop", a, b, out=out)

    assert result is out
    assert call["args"][2] is out
    assert call["grid"] == (1, )


def test_tlx_gfx9_gemm_bench_enables_four_wave_specialization_only_for_wave():
    torch = pytest.importorskip("torch")
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_wave_test_gfx9_bench_four_wave")
    calls = []

    class FakeKernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    module = SimpleNamespace(wave_4wave_specialized=FakeKernel())
    a = torch.empty((256, 128), dtype=torch.float16)
    b = torch.empty((256, 128), dtype=torch.float16).T
    args = SimpleNamespace(wave_split_barriers=False)

    bench.provider_matmul(args, "wave", module, "wave_4wave_specialized", a, b)
    bench.provider_matmul(args, "tlx", module, "wave_4wave_specialized", a, b)

    assert [call[0] for call in calls] == [(1, 1), (1, 1)]
    assert calls[0][2]["num_warps"] == 4
    assert calls[0][2]["tlx_wave_enable_multi_wave_specialize"] is True
    assert "tlx_wave_enable_multi_wave_specialize" not in calls[1][2]


def test_tlx_gfx9_gemm_bench_batched_timing_uses_one_event_span_per_repeat():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_wave_test_gfx9_bench_timing")
    state = {"launches": 0, "synchronizes": 0, "events": 0}

    class FakeEvent:

        def __init__(self):
            self.launch = None

        def record(self):
            self.launch = state["launches"]

        def elapsed_time(self, other):
            return (other.launch - self.launch) * 0.25

    class FakeDeviceInterface:

        def Event(self, *, enable_timing):
            assert enable_timing
            state["events"] += 1
            return FakeEvent()

        def synchronize(self):
            state["synchronizes"] += 1

    def launch():
        state["launches"] += 1

    ms = bench.do_bench_batched(
        launch,
        warmup_launches=2,
        timed_launches=4,
        repeats=3,
        device_interface=FakeDeviceInterface(),
    )

    assert ms == 0.25
    assert state == {"launches": 18, "synchronizes": 6, "events": 6}


def test_tlx_gfx9_gemm_bench_triton_timing_reports_median(monkeypatch):
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_wave_test_gfx9_bench_median")
    call = {}

    def do_bench(fn, **kwargs):
        call["fn"] = fn
        call["kwargs"] = kwargs
        return 0.75

    monkeypatch.setattr(bench.triton.testing, "do_bench", do_bench)
    fn = lambda: None
    ms = bench.measure_provider(
        SimpleNamespace(timing_mode="triton", warmup=13, rep=29),
        fn,
    )

    assert ms == 0.75
    assert call == {
        "fn": fn,
        "kwargs": {"warmup": 13, "rep": 29, "return_mode": "median"},
    }


def test_tlx_gfx9_gemm_bench_loads_modules_without_import_leaks():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_wave_test_gfx9_bench_imports")
    before_path = list(sys.path)

    module = bench.load_matmul_module("v0_naive", "test")

    assert hasattr(module, "matmul")
    assert list(sys.path) == before_path
    assert module.__name__ not in sys.modules


def test_tlx_gfx9_gemm_bench_active_driver_restores(monkeypatch):
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_wave_test_gfx9_bench_driver")

    class FakeDriverState:

        def __init__(self):
            self.active = "previous"
            self.transitions = []

        def set_active(self, driver):
            self.transitions.append(driver)
            self.active = driver

    driver_state = FakeDriverState()
    monkeypatch.setattr(
        bench.triton,
        "runtime",
        SimpleNamespace(driver=driver_state),
    )

    with pytest.raises(RuntimeError, match="boom"):
        with bench.active_driver("next"):
            assert driver_state.active == "next"
            raise RuntimeError("boom")

    assert driver_state.active == "previous"
    assert driver_state.transitions == ["next", "previous"]
    with bench.active_driver(None):
        assert driver_state.active == "previous"
    assert driver_state.transitions == ["next", "previous"]


@pytest.mark.parametrize(
    "case",
    [
        {
            "version_dir": "v2_async_copy",
            "function_name": "v2_async_copy",
            "num_warps": 4,
        },
        {
            "version_dir": "v3_lds",
            "function_name": "v3_lds",
            "num_warps": 4,
        },
        {
            "version_dir": "v4_global_prefetch",
            "function_name": "v4_global_prefetch",
            "num_warps": 4,
        },
        {
            "version_dir": "v6_loop_unroll",
            "function_name": "v6_loop_unroll",
            "num_warps": 4,
        },
        {
            "version_dir": "v7_slice",
            "function_name": "v7_slice",
            "num_warps": 4,
        },
        {
            "version_dir": "v8_warp_pipeline",
            "function_name": "v8_warp_pipeline",
            "num_warps": 8,
            "disables_post_misched": True,
        },
        {
            "version_dir": "v9_beyond_hotloop",
            "function_name": "v9_beyond_hotloop",
            "num_warps": 8,
            "disables_post_misched": True,
            "extra_meta": {"GROUP_SIZE_M": 4, "NUM_XCDS": 8, "GRID_MN": 1},
        },
        {
            "version_dir": "wave_8wave",
            "function_name": "wave_8wave",
            "num_warps": 8,
            "expected_dma_load_lds": 24,
            "b_strides": (1, 256),
            "extra_meta": {"GROUP_SIZE_M": 4, "NUM_XCDS": 8, "GRID_MN": 1},
        },
        {
            "id": "wave_8wave_multi_wave_specialize",
            "version_dir": "wave_8wave",
            "function_name": "wave_8wave",
            "num_warps": 8,
            "expected_dma_load_lds": 24,
            "b_strides": (1, 256),
            "compile_options": {"tlx_wave_enable_multi_wave_specialize": True},
            "extra_meta": {"GROUP_SIZE_M": 4, "NUM_XCDS": 8, "GRID_MN": 1},
        },
        {
            "version_dir": "wave_4wave_specialized",
            "function_name": "wave_4wave_specialized",
            "num_warps": 4,
            "expected_dma_load_lds": 48,
            "b_strides": (1, 256),
            "compile_options": {"tlx_wave_enable_multi_wave_specialize": True},
            "extra_meta": {"GROUP_SIZE_M": 4, "NUM_XCDS": 8, "GRID_MN": 1},
        },
        {
            "id": "v9_beyond_hotloop_grouped_pid_multi_n",
            "version_dir": "v9_beyond_hotloop",
            "function_name": "v9_beyond_hotloop",
            "num_warps": 8,
            "disables_post_misched": True,
            "shape": (512, 1024, 256),
            "extra_meta": {"GROUP_SIZE_M": 4, "NUM_XCDS": 8, "GRID_MN": 8},
        },
        {
            "id": "v9_beyond_hotloop_transposed_b",
            "version_dir": "v9_beyond_hotloop",
            "function_name": "v9_beyond_hotloop",
            "num_warps": 8,
            "disables_post_misched": True,
            "b_strides": (1, 256),
            "extra_meta": {"GROUP_SIZE_M": 4, "NUM_XCDS": 8, "GRID_MN": 1},
        },
    ],
    ids=lambda case: case.get("id", case["version_dir"]),
)
def test_tlx_wave_backend_compiles_gfx9_gemm_passing_variants_to_hsaco(
    tmp_path,
    monkeypatch,
    case,
):
    compiled = _compile_tlx_gfx9_gemm_kernel(tmp_path, monkeypatch, case)
    ttgir_artifact = _asm_text(compiled, "ttgir")
    wave_artifact = _asm_text(compiled, "wave")
    hsaco = compiled.asm["hsaco"]

    assert "tlx_wave.new_converter" in wave_artifact
    assert "gpu.kernel" in wave_artifact
    expected_target_waves = max(1, (case["num_warps"] + 3) // 4)
    assert (f"waveamdmachine.target_waves = {expected_target_waves} : i64" in wave_artifact)
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert compiled.kernel == hsaco
    assert compiled.metadata.tlx_wave_status == "emitted_wave_staged_converter"
    assert compiled.metadata.tlx_wave_binary_stage == "wave-compile-kernels"
    assert compiled.metadata.tlx_wave_hsaco_size_bytes == len(hsaco)
    assert compiled.metadata.tlx_wave_ttgir_target == "hip:gfx950"
    assert compiled.metadata.shared == 0
    assert compiled.metadata.tlx_wave_launch_shared_bytes == 0
    assert compiled.metadata.tlx_wave_lds_size_bytes == 0
    assert compiled.metadata.tlx_wave_num_mmas > 0
    enable_multi_wave_specialization = bool(
        case.get("compile_options", {}).get(
            "tlx_wave_enable_multi_wave_specialize",
            False,
        ))
    assert ("tlx_wave.enable_multi_wave_specialization" in ttgir_artifact) is enable_multi_wave_specialization
    assert ("waveamdmachine.enable_multi_wave_specialization" in wave_artifact) is enable_multi_wave_specialization
    assert (compiled.metadata.tlx_wave_enable_multi_wave_specialize is enable_multi_wave_specialization)
    # The bridge records semantic gather/scatter copies; packetization and
    # direct-to-LDS selection happen later in Wave.
    assert compiled.metadata.tlx_wave_num_dma_load_lds == 0
    if case["version_dir"] == "v9_beyond_hotloop":
        assert "layout_convert" not in wave_artifact
    if case["version_dir"] in {"wave_8wave", "wave_4wave_specialized"}:
        assert "wave.sched_barrier" not in wave_artifact
        assert "memdesc_subslice" not in wave_artifact
    if case.get("id") == "v9_beyond_hotloop_transposed_b":
        lowered_wave = _run_wave_lower_symbolic_memory(wave_artifact)
        barrier_lines = [line for line in lowered_wave.splitlines() if "wave.barrier" in line]
        barrier_tokens = {_ssa_result_name(line) for line in barrier_lines}
        shared_load_lines = [
            line for line in lowered_wave.splitlines() if "wave.load" in line and "#wave.shared" in line
        ]
        dma_lines = [line for line in lowered_wave.splitlines() if "waveamd.dma_load_lds" in line]
        issue_lines = [line for line in lowered_wave.splitlines() if "wave.issue_token" in line]
        # Source barriers remain literal barriers. Implicit wait readiness is
        # represented by SSA dependencies on the loads, not bridge-created
        # publication barriers.
        assert barrier_lines
        assert shared_load_lines
        assert all(re.search(r"\bafter %[A-Za-z0-9_]+", line) for line in shared_load_lines)
        dma_barrier_tokens = {token for token in barrier_tokens if any(f"after {token}" in line for line in dma_lines)}
        barrier_issue_tokens = {
            _ssa_result_name(line)
            for line in issue_lines
            if any(_wave_token_depends_on(
                lowered_wave,
                _ssa_result_name(line),
                token,
            ) for token in barrier_tokens)
        }
        dma_barrier_issue_tokens = {
            token
            for token in barrier_issue_tokens
            if any(f"after {token}" in line for line in dma_lines)
        }
        assert dma_barrier_tokens == set()
        assert dma_barrier_issue_tokens


@pytest.mark.parametrize(
    "case_name,b_layout",
    [
        ("contiguous_b", "contiguous"),
        ("transposed_b", "transposed"),
    ],
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (256, 256, 256),
        (1024, 1024, 1024),
    ],
    ids=["256", "1024"],
)
def test_tlx_wave_runtime_gfx950_v9_e2e(tmp_path, case_name, b_layout, m, n, k):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for gfx950 v9 e2e, got {arch}")
    tutorial = _load_tlx_gfx9_gemm_module(
        "v9_beyond_hotloop",
        f"_tlx_wave_v9_runtime_{case_name}",
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    if b_layout == "contiguous":
        b = torch.randn((k, n), device=device, dtype=torch.float16)
    else:
        b = torch.randn((n, k), device=device, dtype=torch.float16).T
    assert b.shape == (k, n)

    with (
            _active_tlx_wave_driver(),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / f"{case_name}-{m}x{n}x{k}-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        got = tutorial.matmul(a, b)
        torch.cuda.synchronize()

    expected = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(got, expected, atol=1e-1, rtol=0)


@pytest.mark.parametrize("backend", ["llvm", "wave", "wave-multi-wave"])
def test_tlx_wave_runtime_gfx950_wave_8wave_e2e(tmp_path, monkeypatch, backend):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for wave_8wave e2e, got {arch}")
    tutorial = _load_tlx_gfx9_gemm_module(
        "wave_8wave",
        f"_tlx_wave_8wave_runtime_{backend}",
    )

    m, n, k = 256, 256, 256
    device = torch.device("cuda")
    torch.manual_seed(17)
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((n, k), device=device, dtype=torch.float16).T

    if backend == "wave-multi-wave":
        monkeypatch.setenv("TRITON_TLX_WAVE_ENABLE_MULTI_WAVE_SPECIALIZE", "1")
    else:
        monkeypatch.delenv("TRITON_TLX_WAVE_ENABLE_MULTI_WAVE_SPECIALIZE", raising=False)
    driver_context = _active_amd_driver if backend == "llvm" else _active_tlx_wave_driver
    with (
            driver_context(),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / f"{backend}-wave-8wave-runtime-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        got = tutorial.matmul(a, b)
        torch.cuda.synchronize()

    expected = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(got, expected, atol=3e-1, rtol=0)


@pytest.mark.parametrize("backend", ["llvm", "wave"])
def test_tlx_wave_runtime_gfx950_wave_4wave_specialized_e2e(
    tmp_path,
    backend,
):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for specialized 4-wave e2e, got {arch}")
    tutorial = _load_tlx_gfx9_gemm_module(
        "wave_4wave_specialized",
        f"_tlx_wave_4wave_specialized_runtime_{backend}",
    )
    bench = _load_tlx_gfx9_gemm_bench_module(f"_tlx_wave_4wave_specialized_bench_{backend}")

    m, n, k = 256, 256, 256
    device = torch.device("cuda")
    torch.manual_seed(23)
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((n, k), device=device, dtype=torch.float16).T
    expected = torch.matmul(a, b)
    torch.cuda.synchronize()

    compile_options = ({"tlx_wave_enable_multi_wave_specialize": True} if backend == "wave" else None)
    driver_context = _active_amd_driver if backend == "llvm" else _active_tlx_wave_driver
    with (
            driver_context(),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / f"{backend}-wave-4wave-specialized-runtime-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        # Repeated launches cover the independently reused A/B LDS epochs; a
        # missing workgroup rendezvous manifests as a nondeterministic race.
        for _ in range(3):
            got = bench.launch_tutorial_matmul(
                tutorial,
                "wave_4wave_specialized",
                a,
                b,
                extra_compile_options=compile_options,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(got, expected, atol=3e-1, rtol=0)


@pytest.mark.parametrize("backend", ["llvm", "wave"])
def test_tlx_wave_runtime_gfx950_fa_eight_wave_e2e(tmp_path, backend):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for eight-wave FA e2e, got {arch}")
    tutorial = _load_tlx_fa_wave_module(f"_tlx_wave_fa_eight_wave_runtime_{backend}")

    shape = (1, 1, 512, 128)
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(0)

    def bounded_tensor():
        values = torch.randint(
            -8,
            9,
            shape,
            dtype=torch.int8,
            device=device,
            generator=generator,
        ).to(torch.bfloat16)
        return values.mul_(0.125)

    q = bounded_tensor()
    k = bounded_tensor()
    v = torch.rand(shape, device=device, generator=generator).mul_(2.0).sub_(1.0).to(torch.bfloat16)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=1.0 / math.sqrt(shape[-1]),
    )

    driver_context = _active_amd_driver if backend == "llvm" else _active_tlx_wave_driver
    with (
            driver_context(),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / f"{backend}-fa-eight-wave-runtime-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        got = tutorial.attention(q, k, v)
        torch.cuda.synchronize()

    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)


def test_tlx_wave_runtime_gfx950_v9_group_swizzle_multi_n_e2e(tmp_path):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for gfx950 v9 e2e, got {arch}")
    tutorial = _load_tlx_gfx9_gemm_module(
        "v9_beyond_hotloop",
        "_tlx_wave_v9_runtime_group_swizzle_multi_n",
    )

    m, n, k = 512, 1024, 256
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((n, k), device=device, dtype=torch.float16).T
    assert b.shape == (k, n)
    assert b.stride() == (1, k)

    with (
            _active_tlx_wave_driver(),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / "group-swizzle-multi-n-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        got = tutorial.matmul(a, b)
        torch.cuda.synchronize()

    expected = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(got, expected, atol=1e-1, rtol=0)


def test_tlx_runtime_gfx950_v9_amd_backend_transposed_b_e2e(tmp_path):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for gfx950 v9 e2e, got {arch}")
    tutorial = _load_tlx_gfx9_gemm_module(
        "v9_beyond_hotloop",
        "_tlx_v9_amd_backend_transposed_b",
    )

    m = n = k = 256
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((n, k), device=device, dtype=torch.float16).T
    assert b.shape == (k, n)
    assert b.stride() == (1, k)

    with (
            _active_amd_driver(),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / "amd-backend-transposed-b-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        got = tutorial.matmul(a, b)
        torch.cuda.synchronize()

    expected = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(got, expected, atol=1e-1, rtol=0)


def test_tlx_wave_runtime_gfx950_glu_optimized_async_e2e(tmp_path):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for gfx950 GLU e2e, got {arch}")
    tutorial = _load_tlx_glu_bench_module("_tlx_wave_glu_optimized_async_runtime")

    m, n, k = 128, 128, 256
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16)
    bias = torch.randn((n, ), device=device, dtype=torch.float16)
    y = torch.randn((m, n), device=device, dtype=torch.float16)

    expected = tutorial.pytorch_baseline(bias, a, b, y)
    torch.cuda.synchronize()

    with (
            _active_tlx_wave_driver(),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / "glu-optimized-async-runtime-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        got = tutorial.run_optimized_async(a, b, bias, y)
        torch.cuda.synchronize()

    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("k", [1024, 2048])
def test_tlx_wave_runtime_gfx950_a4w4_mxfp_e2e_random(tmp_path, k):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for gfx950 MXFP e2e, got {arch}")
    tutorial = _load_tlx_gfx9_a4w4_module("_tlx_wave_a4w4_mxfp_runtime_random")

    m = n = 256
    device = torch.device("cuda")
    a, b, a_scales, b_scales = _random_mxfp4_inputs(
        torch,
        m,
        n,
        k,
        device,
        seed=0,
    )

    with (
            _active_tlx_wave_driver(),
            triton.knobs.cache.scope(),
            triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / f"a4w4-mxfp-random-cache-k{k}")
        triton.knobs.runtime.override_arch = "gfx950"
        got = tutorial.matmul(a, b, a_scales, b_scales)
        torch.cuda.synchronize()

    expected = _mxfp4_gemm_reference(torch, a, b, a_scales, b_scales)
    torch.cuda.synchronize()
    torch.testing.assert_close(got, expected, atol=1e-1, rtol=0)


def test_tlx_wave_runtime_launches_no_memory_kernel():
    torch, _arch = _require_tlx_wave_runtime_target()

    with _active_tlx_wave_driver():
        _tlx_wave_stage_only_kernel[(1, )]()
        torch.cuda.synchronize()


def test_tlx_wave_runtime_launches_masked_global_memory_kernel():
    torch, _arch = _require_tlx_wave_runtime_target()
    block = 64
    n = 37
    x = torch.arange(block, device="cuda", dtype=torch.float32)
    y = torch.full((block, ), -1.0, device="cuda", dtype=torch.float32)
    expected = torch.full_like(y, -1.0)
    expected[:n] = x[:n] + 1.0

    with _active_tlx_wave_driver():
        _tlx_wave_add_one_kernel[(1, )](x, y, n, BLOCK=block)
        torch.cuda.synchronize()

    torch.testing.assert_close(y, expected)


def test_tlx_wave_converter_pipeline_lowers_program_id(tmp_path):
    local_func = """
  tt.func public @converter_program_id() attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == ["program_id", "return"]
    assert "wave.workgroup_id" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_thread_id(tmp_path):
    local_func = """
  tt.func public @converter_thread_id() attributes {noinline = false} {
    %tid = gpu.thread_id x
    %tid_i32 = arith.index_cast %tid : index to i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == ["thread_id", "type_convert", "return"]
    assert "wave.workitem_id 0" in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_pipeline_lowers_pure_value_if(tmp_path):
    local_func = """
  tt.func public @converter_pure_if(%arg0: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %positive = arith.cmpi sgt, %arg0, %zero : i32
    %value = scf.if %positive -> (i32) {
      %one = arith.constant 1 : i32
      %then = arith.addi %arg0, %one : i32
      scf.yield %then : i32
    } else {
      %two = arith.constant 2 : i32
      %else = arith.addi %arg0, %two : i32
      scf.yield %else : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    root_region = output.target_program.regions[0]
    root_kinds = [output.target_program.ops[op_id].kind for op_id in root_region.op_ids]
    assert [kind for kind in root_kinds if kind != "assume"] == ["constant", "cmpi", "if", "return"]
    if_op = next(op for op in output.target_program.ops if op.kind == "if")
    assert len(if_op.region_ids) == 2
    assert [
        output.target_program.ops[op_id].kind for op_id in output.target_program.regions[if_op.region_ids[0]].op_ids
    ] == ["constant", "binary"]
    assert [
        output.target_program.ops[op_id].kind for op_id in output.target_program.regions[if_op.region_ids[1]].op_ids
    ] == ["constant", "binary"]
    assert "scf.if" in output.emitted_module.text
    assert "wave.select" not in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_pipeline_preserves_mma_packet_through_if(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mma_packet_if(%cond: i1) attributes {noinline = false} {
    %value = scf.if %cond -> (tensor<16x16xf32, #mma>) {
      %then = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #mma>
      scf.yield %then : tensor<16x16xf32, #mma>
    } else {
      %else = arith.constant dense<2.000000e+00> : tensor<16x16xf32, #mma>
      scf.yield %else : tensor<16x16xf32, #mma>
    }
    %truncated = arith.truncf %value : tensor<16x16xf32, #mma> to tensor<16x16xf16, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (if_op, ) = [op for op in output.target_program.ops if op.kind == "if"]
    assert converter_target_ir.attrs_dict(if_op)["result_packet_registers"] == (4, )
    wave = output.emitted_module.text
    if_line = next(line for line in wave.splitlines() if "scf.if" in line)
    yield_lines = [line for line in wave.splitlines() if "scf.yield" in line]
    assert "!wave.simd<vector<4xf32>, 64>" in if_line
    assert len(yield_lines) == 2
    assert all("!wave.simd<vector<4xf32>, 64>" in line for line in yield_lines)
    assert all("!waveamd.fragment" not in line for line in (if_line, *yield_lines))
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_pipeline_keeps_if_stores_in_branches(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_if_stores(
      %cond: i1,
      %ptr: tensor<64x!tt.ptr<f32>, #blocked>,
      %then_value: tensor<64xf32, #blocked>,
      %else_value: tensor<64xf32, #blocked>) attributes {noinline = false} {
    scf.if %cond {
      tt.store %ptr, %then_value : tensor<64x!tt.ptr<f32>, #blocked>
    } else {
      tt.store %ptr, %else_value : tensor<64x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    root_region = output.target_program.regions[0]
    root_kinds = [output.target_program.ops[op_id].kind for op_id in root_region.op_ids]
    assert root_kinds == ["if", "return"]
    if_op = next(op for op in output.target_program.ops if op.kind == "if")
    assert [
        output.target_program.ops[op_id].kind for op_id in output.target_program.regions[if_op.region_ids[0]].op_ids
    ] == ["store"]
    assert [
        output.target_program.ops[op_id].kind for op_id in output.target_program.regions[if_op.region_ids[1]].op_ids
    ] == ["store"]
    wave = output.emitted_module.text
    assert "wave.store" not in wave.split("scf.if", 1)[0]
    assert wave.count("wave.scatter") == 2
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_pipeline_lowers_result_free_if_without_else(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_one_sided_if(
      %cond: i1,
      %ptr: tensor<64x!tt.ptr<f32>, #blocked>,
      %value: tensor<64xf32, #blocked>) attributes {noinline = false} {
    scf.if %cond {
      tt.store %ptr, %value : tensor<64x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (if_op, ) = [op for op in output.target_program.ops if op.kind == "if"]
    then_region, else_region = (output.target_program.regions[region_id] for region_id in if_op.region_ids)
    assert [output.target_program.ops[op_id].kind for op_id in then_region.op_ids] == ["store"]
    assert else_region.op_ids == ()
    assert output.emitted_module.text.count("wave.scatter") == 1
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_pipeline_keeps_branch_assumes_scoped(tmp_path):
    local_func = """
  tt.func public @converter_if_assume_scope(%arg0: i32, %cond: i1) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    scf.if %cond {
      %positive = arith.cmpi sgt, %arg0, %zero : i32
      llvm.intr.assume %positive : i1
      scf.yield
    } else {
      scf.yield
    }
    %one = arith.constant 1 : i32
    %sum = arith.addi %arg0, %one : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    post_if_add = next(op for op in output.target_program.ops
                       if op.kind == "binary" and converter_target_ir.attrs_dict(op)["operation"] == "addi")
    assert post_if_add.fact_ids == ()
    if_op = next(op for op in output.target_program.ops if op.kind == "if")
    then_kinds = [
        output.target_program.ops[op_id].kind for op_id in output.target_program.regions[if_op.region_ids[0]].op_ids
    ]
    assert then_kinds == ["cmpi", "assume"]
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_merges_implicit_if_async_wait_escape_with_neutral_else(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_if_async_wait_escape(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %cond: i1) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    scf.if %cond {
      %token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
      %group = ttg.async_commit_group tokens %token
      scf.yield
    } else {
      scf.yield
    }
    %wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    token_program = converter_tokens.build_token_program(source, converted)

    source_if_op = next(op for op in source.ops if op.name == "scf.if")
    (source_group_op, ) = [op for op in source.ops if op.name == "ttg.async_commit_group"]
    (carry, ) = token_program.if_token_carries_by_op[source_if_op.index]
    assert carry.then_source_value_id == source_group_op.results[0]
    assert carry.else_source_value_id is None

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (target_if_op, ) = [op for op in output.target_program.ops if op.kind == "if"]
    (target_wait_op, ) = [op for op in output.target_program.ops if op.kind == "async_wait"]
    then_region, else_region = (output.target_program.regions[region_id] for region_id in target_if_op.region_ids)
    assert len(target_if_op.results) == 1
    assert target_wait_op.operands == target_if_op.results
    assert "token" not in [output.target_program.ops[op_id].kind for op_id in then_region.op_ids]
    assert [output.target_program.ops[op_id].kind for op_id in else_region.op_ids] == ["token"]
    _run_wave_verify(output.emitted_module.text)
    del ctx


@pytest.mark.parametrize("wait_group", (0, 1), ids=("drain", "partial"))
def test_tlx_wave_converter_handles_alternative_if_async_groups(
    tmp_path,
    wait_group,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = f"""
  tt.func public @converter_if_alternative_async_groups(
      %arg0: !tt.ptr<f16> {{tt.pointer_range = 32 : i32}},
      %arg1: !tt.ptr<f16> {{tt.pointer_range = 32 : i32}},
      %cond: i1) attributes {{noinline = false}} {{
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #blocked>
    scf.if %cond {{
      %then_token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
      %then_group = ttg.async_commit_group tokens %then_token
      scf.yield
    }} else {{
      %else_token = amdg.buffer_load_to_local %arg1[%range] into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
      %else_group = ttg.async_commit_group tokens %else_token
      scf.yield
    }}
    %wait = ttg.async_wait {{num = {wait_group} : i32}}
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    token_program = converter_tokens.build_token_program(source, converted)

    source_if_op = next(op for op in source.ops if op.name == "scf.if")
    then_group_op, else_group_op = [op for op in source.ops if op.name == "ttg.async_commit_group"]
    (carry, ) = token_program.if_token_carries_by_op[source_if_op.index]
    assert carry.then_source_value_id == then_group_op.results[0]
    assert carry.else_source_value_id == else_group_op.results[0]

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (target_if_op, ) = [op for op in output.target_program.ops if op.kind == "if"]
    (target_wait_op, ) = [op for op in output.target_program.ops if op.kind == "async_wait"]
    wait_attrs = converter_target_ir.attrs_dict(target_wait_op)
    assert len(target_if_op.results) == 1
    if wait_group:
        (issue_op, ) = [op for op in output.target_program.ops if op.kind == "issue_token"]
        assert issue_op.operands == target_if_op.results
        assert target_wait_op.operands == issue_op.results
        assert wait_attrs["completed_group_dependency_count"] == 0
        assert wait_attrs["retained_issue_dependency_count"] == 1
    else:
        assert target_wait_op.operands == target_if_op.results
        assert wait_attrs["completed_group_dependency_count"] == 1
        assert wait_attrs["retained_issue_dependency_count"] == 0
    assert all("token" not in
               [output.target_program.ops[op_id].kind
                for op_id in output.target_program.regions[region_id].op_ids]
               for region_id in target_if_op.region_ids)
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_scopes_implicit_wait_to_its_if_branch(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_if_sibling_async_wait(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %cond: i1) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    scf.if %cond {
      %then_token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
      %then_group = ttg.async_commit_group tokens %then_token
      scf.yield
    } else {
      %else_token = amdg.buffer_load_to_local %arg1[%range] into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
      %else_group = ttg.async_commit_group tokens %else_token
      %else_wait = ttg.async_wait {num = 0 : i32}
      scf.yield
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (target_if_op, ) = [op for op in output.target_program.ops if op.kind == "if"]
    then_region, else_region = (output.target_program.regions[region_id] for region_id in target_if_op.region_ids)
    then_ops = tuple(output.target_program.ops[op_id] for op_id in then_region.op_ids)
    else_ops = tuple(output.target_program.ops[op_id] for op_id in else_region.op_ids)
    assert not target_if_op.results
    assert [op.kind for op in then_ops] == [
        "buffer_load_to_local",
        "async_commit_group",
    ]
    assert [op.kind for op in else_ops] == [
        "buffer_load_to_local",
        "async_commit_group",
        "async_wait",
    ]
    else_group = next(op for op in else_ops if op.kind == "async_commit_group")
    else_wait = next(op for op in else_ops if op.kind == "async_wait")
    assert else_wait.operands == else_group.results
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_pipeline_lowers_dynamic_for_with_iter_args(tmp_path):
    local_func = """
  tt.func public @converter_dynamic_for(%arg0: i32, %arg1: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg0 step %c1_i32 iter_args(%acc = %arg1) -> (i32)  : i32 {
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops if op.kind != "assume"] == [
        "constant",
        "constant",
        "binary",
        "for_loop",
        "return",
    ]
    for_op = next(op for op in output.target_program.ops if op.kind == "for_loop")
    assert len(for_op.region_ids) == 1
    assert len(output.target_program.regions[for_op.region_ids[0]].block_arg_ids) == 2
    assert "scf.for" in output.emitted_module.text
    assert "iter_args" in output.emitted_module.text
    assert "scf.yield" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_keeps_loop_carried_mma_values_as_payloads(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_loop_carried_mma_payload() attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> tensor<16x32xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> tensor<32x16xf16, #dot1>
    %init = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %loop = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32, #mma>) {
      %dot = tt.dot %lhs, %rhs, %acc : tensor<16x32xf16, #dot0> * tensor<32x16xf16, #dot1> -> tensor<16x16xf32, #mma>
      scf.yield %dot : tensor<16x16xf32, #mma>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert "scf.for" in wave
    assert "waveamdmachine.isolate_invariant_loop_inits" not in wave
    assert "waveamd.fragment_pack" in wave
    assert "waveamd.fragment_unpack" in wave
    (for_line, ) = [line for line in wave.splitlines() if "scf.for" in line]
    init_pack = re.search(
        r"(?P<pack>%\d+) = wave\.pack (?P<scalar>%\d+), (?P=scalar), (?P=scalar), (?P=scalar).*"
        r"!wave\.simd<vector<4xf32>, 64>",
        wave,
    )
    assert init_pack is not None
    assert init_pack.group("pack") in for_line
    assert init_pack.group("scalar") not in for_line
    assert f"wave.extract {init_pack.group('pack')}" not in wave
    assert for_line.count("!wave.simd<vector<4xf32>, 64>") == 1
    assert "!wave.simd<f32, 64>" not in for_line
    yield_lines = [line for line in wave.splitlines() if "scf.yield" in line]
    assert yield_lines
    assert all("!waveamd.fragment" not in line for line in yield_lines)
    assert all(line.count("!wave.simd<vector<4xf32>, 64>") == 1 for line in yield_lines)
    assert all("!wave.simd<f32, 64>" not in line for line in yield_lines)
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_keeps_redistribute_scratch_internal_in_loop(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
"""
    local_func = """
  tt.func public @converter_looped_direct_dot_internal_scratch(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %step_a = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %step_b = arith.constant dense<8192> : tensor<64x128xi32, #blocked1>
    %offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_k_a = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_e = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %offs_m_b = tt.broadcast %offs_m_e : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %offs_m_s = arith.muli %offs_m_b, %step_a : tensor<128x64xi32, #blocked>
    %offs_k_a_e = tt.expand_dims %offs_k_a {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %offs_k_a_b = tt.broadcast %offs_k_a_e : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %a_offsets = arith.addi %offs_m_s, %offs_k_a_b : tensor<128x64xi32, #blocked>
    %offs_k_b = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_k_b_e = tt.expand_dims %offs_k_b {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %offs_k_b_b = tt.broadcast %offs_k_b_e : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
    %offs_k_b_s = arith.muli %offs_k_b_b, %step_b : tensor<64x128xi32, #blocked1>
    %offs_n_e = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %offs_n_b = tt.broadcast %offs_n_e : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
    %b_offsets = arith.addi %offs_k_b_s, %offs_n_b : tensor<64x128xi32, #blocked1>
    %init = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %loop = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> (tensor<128x128xf32, #mma>) {
      %a_ptrs = arith.addi %a_offsets, %step_a : tensor<128x64xi32, #blocked>
      %b_ptrs = arith.addi %b_offsets, %step_b : tensor<64x128xi32, #blocked1>
      %a = amdg.buffer_load %arg0[%a_ptrs] {contiguity = 8 : i32} : tensor<128x64xf16, #blocked>
      %b = amdg.buffer_load %arg1[%b_ptrs] {contiguity = 8 : i32} : tensor<64x128xf16, #blocked1>
      %lhs = ttg.convert_layout %a : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dot0>
      %rhs = ttg.convert_layout %b : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #dot1>
      %dot = tt.dot %lhs, %rhs, %acc : tensor<128x64xf16, #dot0> * tensor<64x128xf16, #dot1> -> tensor<128x128xf32, #mma>
      scf.yield %dot : tensor<128x128xf32, #mma>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    (for_line, ) = [line for line in wave.splitlines() if "scf.for" in line]
    assert "!wave.mem.token" not in for_line
    assert all("!wave.mem.token" not in line for line in wave.splitlines() if "scf.yield" in line)
    _assert_layout_transforms_reach_wave(output)
    assert sum(op.kind == "layout_convert" for op in output.target_program.ops) == 2
    assert "waveamd.mma" in wave
    _run_wave_verify(wave)
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_pipeline_carries_symbolic_mask_across_if(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_if_symbolic_mask(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %limit: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %positive = arith.cmpi sgt, %limit, %c0 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %active = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %selected = scf.if %positive -> (tensor<64xi1, #blocked>) {
      scf.yield %active : tensor<64xi1, #blocked>
    } else {
      %inactive = arith.constant dense<false> : tensor<64xi1, #blocked>
      scf.yield %inactive : tensor<64xi1, #blocked>
    }
    %other = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %selected, %other {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    if_line = next(line for line in wave.splitlines() if "scf.if" in line)
    assert "-> (!wave.mask<64>)" in if_line
    assert "wave.where" in wave
    assert "wave.gather" in wave
    _run_wave_verify(wave)
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.uniform_if" in machine
    assert "waveamdmachine.exec_if" in machine
    del ctx


def test_tlx_wave_converter_pipeline_carries_symbolic_mask_across_dynamic_for(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_dynamic_for_symbolic_mask(%limit: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %init = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %carried = scf.for %i = %c0 to %c64 step %c1 iter_args(%mask = %init) -> (tensor<64xi1, #blocked>) {
      %next = arith.andi %mask, %init : tensor<64xi1, #blocked>
      scf.yield %next : tensor<64xi1, #blocked>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    (for_line, ) = [line for line in wave.splitlines() if "scf.for" in line]
    (yield_line, ) = [line for line in wave.splitlines() if "scf.yield" in line]
    assert "iter_args" in for_line
    assert "-> (!wave.mask<64>)" in for_line
    assert ": !wave.mask<64>" in yield_line
    assert "arith.andi" not in wave
    _run_wave_verify(wave)
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_preserves_loop_carried_symbolic_mask_for_buffer_load(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_loop_carried_mask_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %limit: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %init = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %carried = scf.for %i = %c0 to %c2 step %c1 iter_args(%mask = %init) -> (tensor<64xi1, #blocked>) {
      %next = arith.andi %mask, %init : tensor<64xi1, #blocked>
      scf.yield %next : tensor<64xi1, #blocked>
    }
    %other = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %carried, %other {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert "scf.for" in wave
    assert "wave.where" in wave
    assert "wave.gather" in wave
    assert "otherwise" in wave
    assert "wave.select" in wave
    for_line = next(line for line in wave.splitlines() if "scf.for" in line)
    assert "iter_args" in for_line
    assert "-> (!wave.mask<64>)" in for_line
    assert "wave.cmpi eq" not in wave
    assert "wave.cmpi ne" not in wave
    _run_wave_verify(wave)
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.uniform_loop carries" in machine
    assert "waveamdmachine.s_and_b64" in machine
    assert "waveamdmachine.continue_if" in machine
    assert "waveamdmachine.exec_if" in machine
    assert "waveamdmachine.v_cmp_eq_u32_vcc" not in machine
    del ctx


def test_tlx_wave_converter_pipeline_carries_constant_symbolic_mask_init(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_dynamic_for_mask_constant_init(%limit: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %init = arith.constant dense<true> : tensor<64xi1, #blocked>
    %carried = scf.for %i = %c0 to %c64 step %c1 iter_args(%mask = %init) -> (tensor<64xi1, #blocked>) {
      %active = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
      %next = arith.andi %mask, %active : tensor<64xi1, #blocked>
      scf.yield %next : tensor<64xi1, #blocked>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    (for_line, ) = [line for line in wave.splitlines() if "scf.for" in line]
    assert "-> (!wave.mask<64>)" in for_line
    assert "scf.yield" in wave
    assert "arith.andi" not in wave
    _run_wave_verify(wave)
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_preserves_selected_symbolic_mask_for_buffer_load(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_select_mask_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %active = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %true = arith.constant dense<true> : tensor<64xi1, #blocked>
    %false = arith.constant dense<false> : tensor<64xi1, #blocked>
    %mask = arith.select %active, %true, %false : tensor<64xi1, #blocked>, tensor<64xi1, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %mask, %other {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert "wave.cmpi slt" in wave
    assert "wave.where" in wave
    assert "wave.gather" in wave
    assert "otherwise" in wave
    assert "wave.select" in wave
    assert "wave.cmpi ne" not in wave
    assert (wave.index("wave.cmpi slt") < wave.index("wave.select") < wave.index("wave.where") <
            wave.index("wave.gather"))
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.exec_if" in machine
    assert "waveamdmachine.v_cmp_ne_u32_vcc" not in machine
    assert "waveamdmachine.v_cmp_lt_i32_vcc" in machine
    assert "waveamdmachine.buffer_load_b16" in machine
    del ctx


def test_tlx_wave_converter_pipeline_lowers_nested_dynamic_for_with_iter_args(tmp_path, ):
    local_func = """
  tt.func public @converter_nested_dynamic_for(
      %arg0: i32,
      %arg1: i32,
      %arg2: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg0 step %c1_i32 iter_args(%outer_acc = %arg2) -> (i32)  : i32 {
      %inner = scf.for %j = %c0_i32 to %arg1 step %c1_i32 iter_args(%inner_acc = %outer_acc) -> (i32)  : i32 {
        %ij = arith.addi %i, %j : i32
        %next_inner = arith.addi %inner_acc, %ij : i32
        scf.yield %next_inner : i32
      }
      %next_outer = arith.addi %inner, %i : i32
      scf.yield %next_outer : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    for_ops = [op for op in output.target_program.ops if op.kind == "for_loop"]
    assert len(for_ops) == 2
    assert len(output.target_program.regions) == 3
    assert len(output.target_program.regions[for_ops[0].region_ids[0]].block_arg_ids) == 2
    assert len(output.target_program.regions[for_ops[1].region_ids[0]].block_arg_ids) == 2
    assert output.emitted_module.text.count("scf.for") == 2
    assert output.emitted_module.text.count("scf.yield") == 2
    del ctx


def _scf_for_matches(wave):
    return tuple(re.finditer(r"%\d+(?::\d+)? = scf\.for [^\n]+", wave))


def _loop_iter_arg_names(loop_header):
    iter_args = re.search(r"iter_args\((?P<args>.*?)\) ->", loop_header)
    assert iter_args is not None
    return tuple(re.findall(r"(%arg\d+)\s*=", iter_args.group("args")))


def _loop_explicit_and_hidden_token_args(for_op, loop_header):
    iter_args = _loop_iter_arg_names(loop_header)
    attrs = converter_target_ir.attrs_dict(for_op)
    source_arg_count = int(attrs["source_result_count"])
    explicit_arg_count = int(attrs["init_arg_count"])
    return iter_args[source_arg_count:explicit_arg_count], iter_args[explicit_arg_count:]


def _wave_token_depends_on(wave, token, dependency):
    worklist = [token]
    seen = set()
    while worklist:
        current = worklist.pop()
        if current == dependency:
            return True
        if current in seen:
            continue
        seen.add(current)
        match = re.search(
            rf"{re.escape(current)} = wave\."
            rf"(?:join|barrier|after|issue_token) "
            rf"(?P<operands>[^\n]*)-> !wave\.mem\.token",
            wave,
        )
        if match is None:
            continue
        worklist.extend(re.findall(r"%[\w#]+", match.group("operands")))
    return False


def _dma_load_lds_matches(wave):
    return tuple(re.finditer(
        r"(?P<token>%\d+) = waveamd\.dma_load_lds [^\n]* after (?P<after>%[\w#]+)",
        wave,
    ))


def _dma_after_dependency_count(wave, dependency):
    return sum(1 for match in _dma_load_lds_matches(wave)
               if _wave_token_depends_on(wave, match.group("after"), dependency))


def _barrier_mentions_any(wave, dependencies):
    for match in re.finditer(r"%\d+ = wave\.barrier (?P<operands>[^\n]+)", wave):
        operands = match.group("operands")
        if any(dependency in operands for dependency in dependencies):
            yield match


def _after_mentions_any(wave, dependencies):
    for match in re.finditer(r"%\d+ = wave\.after (?P<operands>[^\n]+)", wave):
        operands = match.group("operands")
        if any(dependency in operands for dependency in dependencies):
            yield match


def test_tlx_wave_converter_pipeline_carries_async_token_across_dynamic_for(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_async_for(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_group = ttg.async_commit_group tokens %warmup
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %wait = ttg.async_wait {num = 1 : i32}
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    body_dma = [
        op for op in output.target_program.ops if op.kind == "buffer_load_to_local" and op.source_op_index is not None
    ][1]
    attrs = converter_target_ir.attrs_dict(body_dma)
    assert attrs["mode"] == "symbolic_copy"
    assert attrs["issue_dependency_count"] == 0
    (for_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    (loop_match, ) = _scf_for_matches(wave)
    dma_issue_token_args, hidden_local_token_args = _loop_explicit_and_hidden_token_args(for_op, loop_match.group(0))
    assert len(dma_issue_token_args) == 1
    loop_body = wave[loop_match.end():]
    assert "waveamd.dma_load_lds" in loop_body
    assert _dma_after_dependency_count(loop_body, dma_issue_token_args[0]) == 0
    # Queue groups are already explicit loop values. A write-only loop has no
    # synchronous LDS state, read frontier, or future allocation release to
    # justify additional bridge-owned token carries.
    assert hidden_local_token_args == ()
    wait_matches = tuple(_after_mentions_any(loop_body, dma_issue_token_args))
    assert wait_matches
    assert loop_body.index("waveamd.dma_load_lds") < wait_matches[0].start()
    del ctx


def test_tlx_wave_converter_loop_dma_group_is_not_an_implicit_ds_dependency(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_loop_dma_group_not_ds_dependency(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup0 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group0 = ttg.async_commit_group tokens %warmup0
    %warmup1 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group1 = ttg.async_commit_group tokens %warmup1
    %ready = ttg.async_wait {num = 1 : i32}
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %loaded = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
      %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %next_ready = ttg.async_wait {num = 1 : i32}
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (for_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    region = output.target_program.regions[for_op.region_ids[0]]
    block_arg_domains = [output.target_program.values[target_id].event_domain for target_id in region.block_arg_ids[1:]]
    queue_index = block_arg_domains.index(converter_target_ir.EVENT_DOMAIN_DMA_GROUP)
    ready_index = block_arg_domains.index(converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY)

    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    (loop_match, ) = _scf_for_matches(wave)
    iter_args = _loop_iter_arg_names(loop_match.group(0))
    queue_arg = iter_args[queue_index]
    ready_arg = iter_args[ready_index]
    loop_body = wave[loop_match.end():]
    load_line = next(line for line in loop_body.splitlines() if "wave.gather" in line or "wave.load" in line)
    load_dependency = re.search(r"after (?P<token>%[\w#]+)", load_line)
    assert load_dependency is not None
    load_dependency = load_dependency.group("token")
    assert _wave_token_depends_on(wave, load_dependency, ready_arg)
    assert not _wave_token_depends_on(wave, load_dependency, queue_arg)
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_pipeline_carries_loop_issued_async_token_to_final_wait(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_async_for_final_wait(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops].count("token") == 1
    (dma_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(dma_op)
    assert attrs["mode"] == "symbolic_copy"
    assert attrs["issue_dependency_count"] == 0
    (for_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    (loop_match, ) = _scf_for_matches(wave)
    dma_issue_token_args, hidden_local_token_args = _loop_explicit_and_hidden_token_args(for_op, loop_match.group(0))
    assert len(dma_issue_token_args) == 1
    loop_body = wave[loop_match.end():]
    assert _dma_after_dependency_count(loop_body, dma_issue_token_args[0]) == 0
    assert hidden_local_token_args == ()
    assert re.search(
        r"}\n\s+%\d+ = wave\.after %\d+#\d+ : !wave\.mem\.token -> !wave\.mem\.token",
        wave,
    )
    del ctx


def test_tlx_wave_converter_pipeline_carries_async_tokens_through_nested_for(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_nested_async_for(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg2: i32,
      %arg3: i32) attributes {noinline = false} {
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup_a = amdg.buffer_load_to_local %arg0[%range] into %alloc_a : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_b = amdg.buffer_load_to_local %arg1[%range] into %alloc_b : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_group = ttg.async_commit_group tokens %warmup_a, %warmup_b
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg2 step %c1_i32 iter_args(%outer_acc = %c0_i32) -> (i32)  : i32 {
      %inner = scf.for %j = %c0_i32 to %arg3 step %c1_i32 iter_args(%inner_acc = %outer_acc) -> (i32)  : i32 {
        %body_a = amdg.buffer_load_to_local %arg0[%range] into %alloc_a : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
        %body_b = amdg.buffer_load_to_local %arg1[%range] into %alloc_b : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
        %body_group = ttg.async_commit_group tokens %body_a, %body_b
        %wait = ttg.async_wait {num = 1 : i32}
        %ij = arith.addi %i, %j : i32
        %next_inner = arith.addi %inner_acc, %ij : i32
        scf.yield %next_inner : i32
      }
      scf.yield %inner : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    for_ops = [op for op in output.target_program.ops if op.kind == "for_loop"]
    assert len(for_ops) == 2
    dma_ops = [
        op for op in output.target_program.ops if op.kind == "buffer_load_to_local" and op.source_op_index is not None
    ]
    assert len(dma_ops) == 4
    assert [converter_target_ir.attrs_dict(op)["issue_dependency_count"] for op in dma_ops] == [0, 0, 0, 0]
    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    loop_matches = _scf_for_matches(wave)
    assert len(loop_matches) == 2
    dma_issue_token_args, hidden_local_token_args = _loop_explicit_and_hidden_token_args(
        for_ops[-1],
        loop_matches[-1].group(0),
    )
    inner_body = wave[loop_matches[-1].end():]
    inner_dma_matches = _dma_load_lds_matches(inner_body)
    assert len(inner_dma_matches) == 2
    assert len(dma_issue_token_args) == 1
    # The explicit queue group is sufficient for a write-only inner loop. The
    # enclosing structured region remains conservative about nested effects.
    assert hidden_local_token_args == ()
    assert not _wave_token_depends_on(inner_body, inner_dma_matches[0].group("after"), dma_issue_token_args[0])
    assert not _wave_token_depends_on(inner_body, inner_dma_matches[1].group("after"), dma_issue_token_args[0])
    assert not _wave_token_depends_on(inner_body, inner_dma_matches[1].group("after"),
                                      inner_dma_matches[0].group("token"))
    wait_matches = tuple(_after_mentions_any(inner_body, dma_issue_token_args))
    assert len(wait_matches) == 1
    assert inner_body.index("waveamd.dma_load_lds") < wait_matches[0].start()
    assert wave.count("waveamd.dma_load_lds") == 4
    assert wave.count("scf.for") == 2
    del ctx


def test_tlx_wave_converter_pipeline_carries_multiple_async_groups_across_for(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_multi_async_for(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup0 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup0_group = ttg.async_commit_group tokens %warmup0
    %warmup1 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup1_group = ttg.async_commit_group tokens %warmup1
    %warmup2 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup2_group = ttg.async_commit_group tokens %warmup2
    %warmup3 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup3_group = ttg.async_commit_group tokens %warmup3
    %prewait = ttg.async_wait {num = 3 : i32}
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %wait0 = ttg.async_wait {num = 2 : i32}
      %body0 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body0_group = ttg.async_commit_group tokens %body0
      %wait1 = ttg.async_wait {num = 2 : i32}
      %body1 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body1_group = ttg.async_commit_group tokens %body1
      %wait2 = ttg.async_wait {num = 2 : i32}
      %body2 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body2_group = ttg.async_commit_group tokens %body2
      %wait3 = ttg.async_wait {num = 2 : i32}
      %body3 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body3_group = ttg.async_commit_group tokens %body3
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (for_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    for_attrs = converter_target_ir.attrs_dict(for_op)
    assert for_attrs["source_result_count"] == 1
    assert for_attrs["init_arg_count"] == 4
    dma_ops = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    assert [converter_target_ir.attrs_dict(op)["issue_dependency_count"] for op in dma_ops] == [0] * 8
    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    (loop_match, ) = _scf_for_matches(wave)
    iter_args = _loop_iter_arg_names(loop_match.group(0))
    assert len(iter_args) == 4
    assert re.search(
        r"scf\.for .*-> \(i32, !wave\.mem\.token, !wave\.mem\.token, "
        r"!wave\.mem\.token\)",
        wave,
    )
    body = wave[loop_match.end():]
    assert [_dma_after_dependency_count(body, arg) for arg in iter_args[1:]] == [0, 0, 0]
    assert wave.count("waveamd.dma_load_lds") == 8
    del ctx


def test_tlx_wave_converter_pipeline_preserves_waited_slot_carry_for_dynamic_load(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_dynamic_slot_wait_carry(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<4x512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %slot0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<4x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %warmup0 = amdg.buffer_load_to_local %arg0[%range] into %slot0 : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup0_group = ttg.async_commit_group tokens %warmup0
    %slot1 = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<4x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %warmup1 = amdg.buffer_load_to_local %arg0[%range] into %slot1 : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup1_group = ttg.async_commit_group tokens %warmup1
    %slot2 = ttg.memdesc_index %alloc[%c2_i32] : !ttg.memdesc<4x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %warmup2 = amdg.buffer_load_to_local %arg0[%range] into %slot2 : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup2_group = ttg.async_commit_group tokens %warmup2
    %slot3 = ttg.memdesc_index %alloc[%c3_i32] : !ttg.memdesc<4x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %warmup3 = amdg.buffer_load_to_local %arg0[%range] into %slot3 : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup3_group = ttg.async_commit_group tokens %warmup3
    %prewait = ttg.async_wait {num = 2 : i32}
    %seed = ttg.local_load %slot0 {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %prefetch_buf = arith.remsi %i, %c4_i32 : i32
      %next_buf = arith.addi %i, %c1_i32 : i32
      %load_buf = arith.remsi %next_buf, %c4_i32 : i32
      %prefetch_slot = ttg.memdesc_index %alloc[%prefetch_buf] : !ttg.memdesc<4x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
      %body = amdg.buffer_load_to_local %arg0[%range] into %prefetch_slot : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %load_slot = ttg.memdesc_index %alloc[%load_buf] : !ttg.memdesc<4x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
      %loaded = ttg.local_load %load_slot {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
      amdg.buffer_store %loaded, %arg0[%range] {contiguity = 1 : i32} : tensor<512xf16, #blocked>
      %wait = ttg.async_wait {num = 2 : i32}
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    (loop_match, ) = _scf_for_matches(wave)
    iter_args = _loop_iter_arg_names(loop_match.group(0))
    loop_body = wave[loop_match.end():]
    load_pos = min(position for marker in ("wave.load", "waveamd.transpose_load")
                   if (position := loop_body.find(marker)) >= 0)
    wait_ready = next(re.finditer(r"(?P<token>%\d+) = wave\.after (?P<operands>[^\n]+)", loop_body[load_pos:]))
    wait_token = wait_ready.group("token")
    yield_values = re.findall(r"%[\w#]+", re.search(r"scf\.yield (?P<values>[^\n]+?) :", loop_body).group("values"))
    wait_arg_positions = [
        index for index, value in enumerate(yield_values) if _wave_token_depends_on(loop_body, value, wait_token)
    ]
    assert wait_arg_positions
    carried_wait_args = [iter_args[index] for index in wait_arg_positions]
    preload_dma_loads = _dma_load_lds_matches(loop_body[:load_pos])
    assert preload_dma_loads
    # A carried wait may order later DS consumers, but it must never become an
    # inferred destination dependency on a direct-to-LDS DMA issue.
    for dma_load in preload_dma_loads:
        assert not any(
            _wave_token_depends_on(loop_body[:load_pos], dma_load.group("after"), arg) for arg in carried_wait_args)
    del ctx


def _circular_refill_ttgir(
    slot_count,
    *,
    independent_refill=False,
    explicit_warp_pipeline_protocol=False,
):
    refill_arg = ",\n      %refill_phase: i32" if independent_refill else ""
    refill_base = "%refill_phase" if independent_refill else "%next_phase"
    protocol_head = """
      rocdl.s.setprio 1
      rocdl.sched.barrier 0
""" if explicit_warp_pipeline_protocol else ""
    protocol_tail = """
      rocdl.s.setprio 0
      rocdl.sched.barrier 0
      rocdl.s.barrier
      rocdl.sched.barrier 0
""" if explicit_warp_pipeline_protocol else ""
    return f"""
  tt.func public @converter_circular_refill(
      %arg0: !tt.ptr<f16> {{tt.pointer_range = 32 : i32}},
      %trip_count: i32{refill_arg}) attributes {{noinline = false}} {{
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<{slot_count}x512xf16, #shared, #smem, mutable>
    %range = tt.make_range {{end = 512 : i32, start = 0 : i32}} : tensor<512xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %depth = arith.constant {slot_count} : i32
    %slot0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<{slot_count}x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %warmup = amdg.buffer_load_to_local %arg0[%range] into %slot0 : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_group = ttg.async_commit_group tokens %warmup
    %sum = scf.for %i = %c0_i32 to %trip_count step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {{
{protocol_head}
      %wait = ttg.async_wait {{num = 0 : i32}}
      %current_phase = arith.remui %i, %depth : i32
      %next_phase = arith.addi %i, %c1_i32 overflow<nsw> : i32
      %refill_phase_mod = arith.remui {refill_base}, %depth : i32
      %current = ttg.memdesc_index %alloc[%current_phase] : !ttg.memdesc<{slot_count}x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
      %loaded = ttg.local_load %current token %wait : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
      ttg.barrier all
      %refill = ttg.memdesc_index %alloc[%refill_phase_mod] : !ttg.memdesc<{slot_count}x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
      %body = amdg.buffer_load_to_local %arg0[%range] into %refill : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
{protocol_tail}
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }}
    %final_wait = ttg.async_wait {{num = 0 : i32}}
    tt.return
  }}
"""


@pytest.mark.parametrize("slot_count", (2, 3, 4))
def test_tlx_wave_converter_pipeline_uses_compiler_barrier_for_circular_refill(
    tmp_path,
    slot_count,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        _circular_refill_ttgir(slot_count),
        num_warps=4,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    dynamic_views = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "memdesc_index" and converter_target_ir.attrs_dict(op)["static_byte_offset"] is None
    ]
    assert dynamic_views
    assert all(attrs["slot_count"] == slot_count for attrs in dynamic_views)
    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    (loop_match, ) = _scf_for_matches(wave)
    loop_body = wave[loop_match.end():]
    load_index = loop_body.index("wave.load")
    dma_index = loop_body.index("waveamd.dma_load_lds")
    barrier_indices = [match.start() for match in re.finditer(r"wave\.barrier", loop_body[:dma_index])]
    sched_barrier_indices = [match.start() for match in re.finditer(r"wave\.sched_barrier", loop_body[:dma_index])]
    assert len(barrier_indices) == 1
    assert not sched_barrier_indices
    assert load_index < barrier_indices[0] < dma_index
    assert all(op.kind != "lds_release" for op in output.target_program.ops)
    dma_ops = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    assert len(dma_ops) == 2
    body_dma = dma_ops[-1]
    body_dma_attrs = converter_target_ir.attrs_dict(body_dma)
    assert body_dma_attrs["issue_dependency_count"] == 0
    assert body_dma_attrs["source_issue_dependency_count"] == 0
    assert body_dma_attrs["barrier_order_dependency_count"] == 1
    assert "lds_release_dependency_count" not in body_dma_attrs
    order_token_id = body_dma.operands[-1]
    assert output.target_program.values[order_token_id].event_domain == (converter_target_ir.EVENT_DOMAIN_BARRIER_ISSUE)
    order_projection = _target_value_producer(
        output.target_program,
        order_token_id,
        kind="issue_token",
    )
    full_barrier = _target_value_producer(
        output.target_program,
        order_projection.operands[0],
        kind="barrier",
    )
    assert converter_target_ir.attrs_dict(full_barrier)["address_space"] == 31
    _run_wave_verify(wave)
    if slot_count == 2:
        machine = _run_waveamd_to_machine(wave)
        assert machine.count("waveamdmachine.s_barrier") == 1
    del ctx


def test_tlx_wave_converter_pipeline_does_not_infer_dma_dependency_from_circular_alias(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        _circular_refill_ttgir(4, independent_refill=True),
        num_warps=4,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    (loop_match, ) = _scf_for_matches(wave)
    loop_body = wave[loop_match.end():]
    load_index = loop_body.index("wave.load")
    dma_index = loop_body.index("waveamd.dma_load_lds")
    barrier_indices = [match.start() for match in re.finditer(r"wave\.barrier", loop_body[:dma_index])]
    sched_barrier_indices = [match.start() for match in re.finditer(r"wave\.sched_barrier", loop_body[:dma_index])]
    assert len(barrier_indices) == 1
    assert not sched_barrier_indices
    assert load_index < barrier_indices[0] < dma_index
    dma_ops = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    dma_attrs = converter_target_ir.attrs_dict(dma_ops[-1])
    assert dma_attrs["issue_dependency_count"] == 0
    assert dma_attrs["source_issue_dependency_count"] == 0
    assert dma_attrs["barrier_order_dependency_count"] == 1
    assert "lds_release_dependency_count" not in dma_attrs
    order_token_id = dma_ops[-1].operands[-1]
    assert output.target_program.values[order_token_id].event_domain == (converter_target_ir.EVENT_DOMAIN_BARRIER_ISSUE)
    order_projection = _target_value_producer(
        output.target_program,
        order_token_id,
        kind="issue_token",
    )
    full_barrier = _target_value_producer(
        output.target_program,
        order_projection.operands[0],
        kind="barrier",
    )
    assert converter_target_ir.attrs_dict(full_barrier)["address_space"] == 31
    assert all(op.kind != "lds_release" for op in output.target_program.ops)
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_keeps_compiler_reuse_barrier_with_warp_pipeline(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        _circular_refill_ttgir(2, explicit_warp_pipeline_protocol=True),
        num_warps=4,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (loop_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    assert converter_target_ir.attrs_dict(loop_op)["explicit_warp_pipeline_protocol"] is True
    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    (loop_match, ) = _scf_for_matches(wave)
    loop_body = wave[loop_match.end():]
    load_index = loop_body.index("wave.load")
    dma_index = loop_body.index("waveamd.dma_load_lds")
    barrier_indices = [match.start() for match in re.finditer(r"wave\.barrier", loop_body)]
    assert len(barrier_indices) == 2
    assert load_index < barrier_indices[0] < dma_index < barrier_indices[1]
    dma_ops = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    assert converter_target_ir.attrs_dict(dma_ops[-1])["issue_dependency_count"] == 0
    assert all(op.kind != "lds_release" for op in output.target_program.ops)
    assert "waveamd.set_priority 1" in loop_body
    assert "waveamd.set_priority 0" in loop_body
    _run_wave_verify(wave)
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.s_barrier") == 2
    del ctx


def test_tlx_wave_converter_pipeline_orders_wait_consumed_body_issue(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_consumed_body_issue(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_group = ttg.async_commit_group tokens %warmup
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %body0 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body0_group = ttg.async_commit_group tokens %body0
      %body1 = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body1_group = ttg.async_commit_group tokens %body1
      %wait = ttg.async_wait {num = 1 : i32}
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    dma_ops = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    assert [converter_target_ir.attrs_dict(op)["issue_dependency_count"] for op in dma_ops] == [0, 0, 0]
    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    assert wave.count("waveamd.dma_load_lds") == 3
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.s_waitcnt") <= 2
    assert "vmcnt(0)" not in machine
    del ctx


def test_tlx_wave_converter_pipeline_lowers_signed_extrema(tmp_path):
    local_func = """
  tt.func public @converter_signed_extrema(%arg0: i32, %arg1: i32) attributes {noinline = false} {
    %min = arith.minsi %arg0, %arg1 : i32
    %max = arith.maxsi %arg0, %arg1 : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops if op.kind != "assume"] == ["minsi", "maxsi", "return"]
    assert output.emitted_module.text.count("arith.cmpi") == 2
    assert output.emitted_module.text.count("wave.select") == 2
    del ctx


def test_tlx_wave_converter_pipeline_lowers_local_alloc(tmp_path):
    preamble = """
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_local_alloc() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == ["local_alloc", "return"]
    local_alloc = output.target_program.ops[0]
    assert converter_target_ir.attrs_dict(local_alloc) == {
        "allocation_bytes": 128,
        "align": 16,
        "element_type": "f16",
        "shape": (64, ),
    }
    assert output.emitted_module.lds_size == 0
    assert "wave.alloc" in output.emitted_module.text
    assert "wave.lds_size" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_memdesc_index(tmp_path):
    preamble = """
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_memdesc_index() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x64xf16, #shared, #smem, mutable>
    %slot = arith.constant 1 : i32
    %view = ttg.memdesc_index %alloc[%slot] : !ttg.memdesc<2x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "local_alloc",
        "constant",
        "memdesc_index",
        "return",
    ]
    assert converter_target_ir.attrs_dict(output.target_program.ops[2]) == {
        "element_byte_width": 2,
        "elements_per_slot": 64,
        "slot_count": 2,
        "static_byte_offset": 128,
    }
    assert output.emitted_module.lds_size == 0
    assert "wave.lds_size" not in output.emitted_module.text
    assert "wave.alloc" in output.emitted_module.text
    assert "wave.ptr_add" in output.emitted_module.text
    del ctx


@pytest.mark.parametrize(
    "shared_a,shared_b,expected_plan",
    [
        (
            "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>",
            "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>",
            "dense_row_major",
        ),
        (
            "#ttg.padded_shared<[512:+16] {order = [1, 0], shape = [256, 64]}>",
            "#ttg.padded_shared<[512:+16] {order = [1, 0], shape = [64, 256]}>",
            "padded_linear",
        ),
    ],
    ids=["identity-swizzled", "padded"],
)
def test_tlx_wave_converter_lowers_memdesc_subslice_as_physical_parent_view(
    tmp_path,
    shared_a,
    shared_b,
    expected_plan,
):
    preamble = f"""
#mma = #ttg.amd_mfma<{{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}}>
#dot0 = #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>
#dot1 = #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = 8}}>
#shared_a = {shared_a}
#shared_b = {shared_b}
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_memdesc_subslice() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared_a, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared_b, #smem, mutable>
    %a_view = ttg.memdesc_subslice %a_alloc[0, 32] : !ttg.memdesc<256x64xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable, 256x64>
    %b_view = ttg.memdesc_subslice %b_alloc[32, 0] : !ttg.memdesc<64x256xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable, 64x256>
    %lhs = ttg.local_load %a_view : !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable, 256x64> -> tensor<256x32xf16, #dot0>
    %rhs = ttg.local_load %b_view : !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable, 64x256> -> tensor<32x256xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<256x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<256x256xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    source_subslices = [op for op in output.source_program.ops if op.name == "ttg.memdesc_subslice"]
    assert [op.attrs["offsets"] for op in source_subslices] == [(0, 32), (32, 0)]
    view_attrs = [converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "memdesc_view"]
    assert view_attrs == [
        {
            "logical_origin": (0, 32),
            "physical_shape": (256, 64),
            "view": "subslice",
        },
        {
            "logical_origin": (32, 0),
            "physical_shape": (64, 256),
            "view": "subslice",
        },
    ]
    load_attrs = _packetized_local_load_attrs(output)
    assert [attrs["memdesc_shape"] for attrs in load_attrs] == [(256, 32), (32, 256)]
    assert [attrs["memdesc_physical_shape"] for attrs in load_attrs] == [(256, 64), (64, 256)]
    assert [attrs["memdesc_logical_origin"] for attrs in load_attrs] == [(0, 32), (32, 0)]
    assert [attrs["shared_physical_offset_plan"] for attrs in load_attrs] == [
        expected_plan,
        expected_plan,
    ]
    assert "waveamd.mma" in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_carries_shared_memdescs_as_structural_pointers(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_loop_carried_memdesc() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %slot0 = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %slot1 = ttg.memdesc_index %alloc[%c1] : !ttg.memdesc<2x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %loop:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%current = %slot0, %next = %slot1) -> (!ttg.memdesc<64xf32, #shared, #smem, mutable>, !ttg.memdesc<64xf32, #shared, #smem, mutable>) : i32 {
      %loaded = ttg.local_load %current : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked>
      scf.yield %next, %current : !ttg.memdesc<64xf32, #shared, #smem, mutable>, !ttg.memdesc<64xf32, #shared, #smem, mutable>
    }
    %final = ttg.local_load %loop#0 : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    (loop_match, ) = _scf_for_matches(wave)
    loop_header = loop_match.group(0)
    assert loop_header.count("!wave.ptr<#wave.shared") >= 2
    assert "wave.gather" in wave or "wave.load" in wave
    _run_wave_verify(wave)
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.uniform_loop" in machine
    del ctx


def test_tlx_wave_converter_applies_memdesc_subslice_to_local_load_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_memdesc_subslice_local_access() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %alloc[64] : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable, 128>
    %value = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    ttg.local_store %value, %view : tensor<64xf32, #blocked> -> !ttg.memdesc<64xf32, #shared, #smem, mutable, 128>
    %loaded = ttg.local_load %view : !ttg.memdesc<64xf32, #shared, #smem, mutable, 128> -> tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    accesses = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind in {"local_store", "local_load"}
    ]
    assert len(accesses) == 2
    assert [attrs["memdesc_logical_origin"] for attrs in accesses] == [(64, ), (64, )]
    assert [attrs["memdesc_physical_shape"] for attrs in accesses] == [(128, ), (128, )]
    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert wave.count("wave.gather") == 1
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_scalarizes_async_copy_into_memdesc_subslice(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_memdesc_subslice_async_copy(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<128xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %alloc[64] : !ttg.memdesc<128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64xf16, #shared, #smem, mutable, 128>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %view : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable, 128>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (copy_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(copy_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    wave = output.emitted_module.text
    assert "waveamd.dma_load_lds" not in wave
    assert wave.count("wave.gather") == 1
    assert wave.count("wave.scatter") == 1
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_packetizes_padded_dma_into_memdesc_subslice(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.padded_shared<[512:+8] {order = [0], shape = [4096]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_padded_subslice_dma(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<4096xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %alloc[2048] : !ttg.memdesc<4096xf16, #shared, #smem, mutable> -> !ttg.memdesc<2048xf16, #shared, #smem, mutable, 4096>
    %range = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %view : <f16>[tensor<2048xi32, #blocked>] -> <2048xf16, #shared, #smem, mutable, 4096>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (copy_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(copy_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert wave.count("waveamd.dma_load_lds") == 1
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_rejects_out_of_bounds_memdesc_subslice():
    parent_type = converter_source_ir.SourceType(
        raw="parent",
        kind="memdesc",
        shape=(8, 8),
        element_type="f16",
        element_byte_width=2,
        encoding="shared",
        memory_space="smem",
        mutable=True,
        alloc_shape=(8, 8),
    )
    child_type = converter_source_ir.SourceType(
        raw="child",
        kind="memdesc",
        shape=(4, 4),
        element_type="f16",
        element_byte_width=2,
        encoding="shared",
        memory_space="smem",
        mutable=True,
        alloc_shape=(8, 8),
    )
    source_program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("bad_subslice"),
        (
            converter_source_ir.SourceOp(0, "ttg.local_alloc", results=(0, )),
            converter_source_ir.SourceOp(
                1,
                "ttg.memdesc_subslice",
                operands=(0, ),
                results=(1, ),
                attrs={"offsets": (6, 0)},
            ),
        ),
        {
            0: converter_source_ir.SourceValue(0, parent_type, owner_op_index=0),
            1: converter_source_ir.SourceValue(1, child_type, owner_op_index=1),
        },
        (),
        0,
    )
    memdescs = converter_op_conversion._memdesc_infos(source_program)

    with pytest.raises(converter_diagnostics.Diagnostic, match="outside the parent logical shape"):
        converter_op_conversion._compute_memdesc_view_infos(
            source_program.values,
            source_program.ops,
            source_program.kernel.arg_ids,
            memdescs,
        )


def test_tlx_wave_converter_static_memdesc_offsets_are_relative_to_operand():
    ops = (
        SimpleNamespace(name="ttg.local_alloc", operands=(), results=(0, ), index=0),
        SimpleNamespace(name="ttg.memdesc_index", operands=(0, 10), results=(1, ), index=1),
        SimpleNamespace(name="ttg.memdesc_index", operands=(1, 10), results=(2, ), index=2),
    )
    memdescs = {
        1:
        converter_op_conversion.MemdescInfo(
            value_id=1,
            element_type="f16",
            element_byte_width=2,
            shape=(2, 64),
            alloc_shape=(2, 64),
            allocation_bytes=256,
        ),
        2:
        converter_op_conversion.MemdescInfo(
            value_id=2,
            element_type="f16",
            element_byte_width=2,
            shape=(64, ),
            alloc_shape=(64, ),
            allocation_bytes=128,
        ),
    }

    offsets = converter_op_conversion._compute_static_memdesc_byte_offsets(
        ops,
        memdescs,
        {1: 256, 2: 128},
        {10: 1},
    )

    assert offsets == {1: 256, 2: 128}


def test_tlx_wave_converter_pipeline_lowers_padded_memdesc_index_stride(tmp_path):
    preamble = """
#shared = #ttg.padded_shared<[512:+16] {order = [1, 0], shape = [256, 64]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_padded_memdesc_index() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable>
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %view0 = ttg.memdesc_index %alloc[%zero] : !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %view1 = ttg.memdesc_index %alloc[%one] : !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    local_alloc_attrs = converter_target_ir.attrs_dict(output.target_program.ops[0])
    # Preserve the full physical stride for both indexed slots, including
    # the otherwise-unused terminal padding of the final slot.
    assert local_alloc_attrs["allocation_bytes"] == 67584
    assert output.emitted_module.lds_size == 0
    assert "wave.lds_size" not in output.emitted_module.text
    assert "wave.alloc" in output.emitted_module.text

    first_view_attrs = converter_target_ir.attrs_dict(output.target_program.ops[3])
    second_view_attrs = converter_target_ir.attrs_dict(output.target_program.ops[4])
    assert first_view_attrs["elements_per_slot"] == 16896
    assert first_view_attrs["static_byte_offset"] == 0
    assert second_view_attrs["elements_per_slot"] == 16896
    assert second_view_attrs["static_byte_offset"] == 33792
    assert "wave.ptr_add" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_sizes_three_slot_padded_dot_buffers(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared_a = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0], [8, 0], [16, 0], [1, 0], [32, 0], [64, 0]], block = []}>
#shared_b = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [4, 0], [16, 0], [1, 0], [2, 0], [8, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_three_slot_padded_dot_buffers(%stage: i32) attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #shared_a, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<3x32x128xf16, #shared_b, #smem, mutable>
    %a_view = ttg.memdesc_index %a_alloc[%stage] : !ttg.memdesc<3x128x32xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared_a, #smem, mutable>
    %b_view = ttg.memdesc_index %b_alloc[%stage] : !ttg.memdesc<3x32x128xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared_b, #smem, mutable>
    %lhs = ttg.local_load %a_view : !ttg.memdesc<128x32xf16, #shared_a, #smem, mutable> -> tensor<128x32xf16, #dot0>
    %rhs = ttg.local_load %b_view : !ttg.memdesc<32x128xf16, #shared_b, #smem, mutable> -> tensor<32x128xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<128x32xf16, #dot0> * tensor<32x128xf16, #dot1> -> tensor<128x128xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    local_alloc_attrs = [
        converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "local_alloc"
    ]
    assert [attrs["allocation_bytes"] for attrs in local_alloc_attrs] == [25344, 25344]
    memdesc_index_attrs = [
        converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "memdesc_index"
    ]
    assert [attrs["elements_per_slot"] for attrs in memdesc_index_attrs] == [4224, 4224]
    assert all(attrs["static_byte_offset"] is None for attrs in memdesc_index_attrs)
    local_load_attrs = _packetized_local_load_attrs(output)
    assert [attrs["shared_physical_offset_plan"] for attrs in local_load_attrs] == [
        "padded_linear",
        "padded_linear",
    ]
    assert "waveamd.mma" in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_pipeline_sizes_non_glu_padded_memdesc_slots(tmp_path):
    preamble = """
#shared = #ttg.padded_shared<[512:+16] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_five_slot_padded_memdesc(%stage: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<5x64x64xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_index %alloc[%stage] : !ttg.memdesc<5x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    local_alloc_attrs = [
        converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "local_alloc"
    ]
    assert [attrs["allocation_bytes"] for attrs in local_alloc_attrs] == [42240]
    (memdesc_index_attrs, ) = [
        converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "memdesc_index"
    ]
    assert memdesc_index_attrs["elements_per_slot"] == 4224
    assert memdesc_index_attrs["static_byte_offset"] is None
    assert "wave.ptr_add" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_buffer_load_to_local_dma(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_buffer_load_to_local(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "local_alloc",
        "make_range",
        "buffer_load_to_local",
        "async_commit_group",
        "async_wait",
        "return",
    ]
    (dma_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(dma_op)
    assert attrs["mode"] == "symbolic_copy"
    assert attrs["component_count"] == 8
    assert attrs["destination_offset_mode"] == "layout_coordinates"
    assert "waveamd.make_buffer" in output.emitted_module.text
    assert "wave.gather" in output.emitted_module.text
    assert "wave.scatter" in output.emitted_module.text
    assert "wave.load" not in output.emitted_module.text
    assert "wave.store" not in output.emitted_module.text
    assert "waveamd.dma_load_lds" not in output.emitted_module.text
    assert "wave.wait" not in output.emitted_module.text
    assert "wave.after" in output.emitted_module.text
    assert "wave.barrier" not in output.emitted_module.text
    lowered = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    assert lowered.count("waveamd.dma_load_lds") == 1
    assert "wave.gather" not in lowered
    assert "wave.scatter" not in lowered
    del ctx


def test_tlx_wave_converter_pipeline_preserves_pointer_range_through_addptr_dma(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_buffer_load_to_local_derived_base(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %base_offset = arith.constant 128 : i32
    %base = tt.addptr %arg0, %base_offset : !tt.ptr<f16>, i32
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %token = amdg.buffer_load_to_local %base[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (dma_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(dma_op)
    assert attrs["mode"] == "symbolic_copy"
    assert attrs["range_bytes"] == 2147483647
    assert "waveamd.make_buffer" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_keeps_dma_pointer_conversion_inside_loop(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_loop_carried_dma_pointer(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %loop = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%base_offset = %c64_i32) -> (i32)  : i32 {
      %next_offset = arith.addi %base_offset, %c64_i32 : i32
      %base = tt.addptr %arg0, %next_offset : !tt.ptr<f16>, i32
      %token = amdg.buffer_load_to_local %base[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %group = ttg.async_commit_group tokens %token
      %wait = ttg.async_wait %group {num = 0 : i32}
      scf.yield %next_offset : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (dma_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    dma_attrs = converter_target_ir.attrs_dict(dma_op)
    _assert_mechanical_symbolic_copy(dma_attrs, masked=False)
    assert "source_pointer_mode" not in dma_attrs
    pointer_type = output.target_program.values[dma_op.operands[1]].type
    assert pointer_type.representation == "uniform_pointer"
    assert pointer_type.component_count == 1
    assert not [op for op in output.target_program.ops if op.kind == "make_buffer"]

    (for_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    region = output.target_program.regions[for_op.region_ids[0]]
    assert all(output.target_program.values[value_id].type.representation != "buffer_pointer"
               for value_id in for_op.operands[3:])
    assert all(output.target_program.values[value_id].type.representation != "buffer_pointer"
               for value_id in region.block_arg_ids[1:])

    raw_wave = output.emitted_module.text
    assert raw_wave.count("waveamd.make_buffer") == 1
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.reg_after" not in machine
    _run_wave_compile_kernels(wave)
    del ctx


def test_tlx_wave_converter_dma_affine_offset_marks_layout_math_nsw(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[512:+16] {order = [1, 0], shape = [256, 64]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_dma_affine_nsw(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %rows = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<256x1xi32, #blocked>
    %row_scaled = arith.muli %row, %stride_splat : tensor<256x1xi32, #blocked>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %row_b = tt.broadcast %row_scaled : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %col_b = tt.broadcast %col : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %offset = arith.addi %row_b, %col_b : tensor<256x64xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%offset] into %alloc : <f16>[tensor<256x64xi32, #blocked>] -> <256x64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 1
    assert wave.count("wave.scatter") == 1
    del ctx


def test_tlx_wave_converter_lowers_dynamic_memdesc_index_packet_dma_destination(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_dynamic_memdesc_dma(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stage: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x512xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_index %alloc[%stage] : !ttg.memdesc<2x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %view : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops if op.kind != "assume"] == [
        "local_alloc",
        "memdesc_index",
        "make_range",
        "buffer_load_to_local",
        "async_commit_group",
        "async_wait",
        "return",
    ]
    memdesc_op = next(op for op in output.target_program.ops if op.kind == "memdesc_index")
    memdesc_attrs = converter_target_ir.attrs_dict(memdesc_op)
    assert memdesc_attrs["element_byte_width"] == 2
    assert memdesc_attrs["elements_per_slot"] == 512
    assert memdesc_attrs["static_byte_offset"] is None
    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    load_attrs = converter_target_ir.attrs_dict(load_op)
    _assert_mechanical_symbolic_copy(load_attrs, masked=False)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_lds_b128" in machine
    del ctx


def test_tlx_wave_converter_lowers_dynamic_padded_memdesc_index_packet_dma_destination(tmp_path, ):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [32, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [1, 0]], block = []}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0], [8, 0], [16, 0], [1, 0], [32, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_dynamic_padded_memdesc_dma(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stage: i32,
      %row_limit: i32,
      %stride: i32,
      %limit: i32 {tt.divisibility = 2 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<4x64x32xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_index %alloc[%stage] : !ttg.memdesc<4x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi32, #linear>
    %row_limit_splat = tt.splat %row_limit : i32 -> tensor<64x1xi32, #linear>
    %row_mod = arith.remsi %row, %row_limit_splat : tensor<64x1xi32, #linear>
    %stride_splat = tt.splat %stride : i32 -> tensor<64x1xi32, #linear>
    %row_scaled = arith.muli %row_mod, %stride_splat : tensor<64x1xi32, #linear>
    %row_b = tt.broadcast %row_scaled : tensor<64x1xi32, #linear> -> tensor<64x32xi32, #linear>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x32xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x32xi32, #linear> -> tensor<64x32xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<64x32xi32, #linear>
    %limit_splat = tt.splat %limit : i32 -> tensor<1x32xi32, #linear>
    %mask = arith.cmpi slt, %col, %limit_splat : tensor<1x32xi32, #linear>
    %mask_b = tt.broadcast %mask : tensor<1x32xi1, #linear> -> tensor<64x32xi1, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] mask = %mask_b stride = %stride into %view {contiguity = 2 : i32} : <f16>[tensor<64x32xi32, #linear>] -> <64x32xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (local_alloc_op, ) = [op for op in output.target_program.ops if op.kind == "local_alloc"]
    # Four indexed slots retain the full 4224-byte physical stride, including
    # the final slot's terminal padding.
    assert converter_target_ir.attrs_dict(local_alloc_op)["allocation_bytes"] == 16896
    (memdesc_index_op, ) = [op for op in output.target_program.ops if op.kind == "memdesc_index"]
    memdesc_attrs = converter_target_ir.attrs_dict(memdesc_index_op)
    assert memdesc_attrs["elements_per_slot"] == 2112
    assert memdesc_attrs["static_byte_offset"] is None
    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.where") == 1
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.ds_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_lowers_bit_affine_padded_dot_dma_packet(tmp_path, ):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [32, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [1, 0]], block = []}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0], [8, 0], [16, 0], [1, 0], [32, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_dynamic_padded_memdesc_dot_dma(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stage: i32,
      %row_limit: i32,
      %stride: i32,
    %limit: i32 {tt.divisibility = 2 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<4x64x32xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_index %alloc[%stage] : !ttg.memdesc<4x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %c1_i32 = arith.constant 1 : i32
    %prefetch = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<4x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi32, #linear>
    %row_limit_splat = tt.splat %row_limit : i32 -> tensor<64x1xi32, #linear>
    %row_mod = arith.remsi %row, %row_limit_splat : tensor<64x1xi32, #linear>
    %stride_splat = tt.splat %stride : i32 -> tensor<64x1xi32, #linear>
    %row_scaled = arith.muli %row_mod, %stride_splat : tensor<64x1xi32, #linear>
    %row_b = tt.broadcast %row_scaled : tensor<64x1xi32, #linear> -> tensor<64x32xi32, #linear>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x32xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x32xi32, #linear> -> tensor<64x32xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<64x32xi32, #linear>
    %limit_splat = tt.splat %limit : i32 -> tensor<1x32xi32, #linear>
    %mask = arith.cmpi slt, %col, %limit_splat : tensor<1x32xi32, #linear>
    %mask_b = tt.broadcast %mask : tensor<1x32xi1, #linear> -> tensor<64x32xi1, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] mask = %mask_b stride = %stride into %prefetch {contiguity = 2 : i32} : <f16>[tensor<64x32xi32, #linear>] -> <64x32xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    %tile = ttg.local_load %view {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    local_alloc = next(op for op in output.target_program.ops if op.kind == "local_alloc")
    assert converter_target_ir.attrs_dict(local_alloc)["allocation_bytes"] == 16896
    memdesc_index_ops = [op for op in output.target_program.ops if op.kind == "memdesc_index"]
    assert [converter_target_ir.attrs_dict(op)["elements_per_slot"] for op in memdesc_index_ops] == [2112, 2112]
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.where") == 1
    assert raw_wave.count("wave.gather") == 3
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.ds_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_keeps_padded_dot_b_dma_packet(tmp_path, ):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [4, 0], [16, 0]], warp = [[1, 0], [2, 0], [8, 0]], block = []}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [4, 0], [16, 0], [1, 0], [2, 0], [8, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_dynamic_padded_memdesc_dot_b_dma(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stage: i32,
      %stride: i32,
      %limit: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<4x32x128xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_index %alloc[%stage] : !ttg.memdesc<4x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %c1_i32 = arith.constant 1 : i32
    %prefetch = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<4x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<32x1xi32, #linear>
    %stride_splat = tt.splat %stride : i32 -> tensor<32x1xi32, #linear>
    %row_scaled = arith.muli %row, %stride_splat : tensor<32x1xi32, #linear>
    %row_b = tt.broadcast %row_scaled : tensor<32x1xi32, #linear> -> tensor<32x128xi32, #linear>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x128xi32, #linear> -> tensor<32x128xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<32x128xi32, #linear>
    %limit_splat = tt.splat %limit : i32 -> tensor<1x128xi32, #linear>
    %mask = arith.cmpi slt, %col, %limit_splat : tensor<1x128xi32, #linear>
    %mask_b = tt.broadcast %mask : tensor<1x128xi1, #linear> -> tensor<32x128xi1, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] mask = %mask_b stride = %stride into %prefetch {contiguity = 8 : i32} : <f16>[tensor<32x128xi32, #linear>] -> <32x128xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    %tile = ttg.local_load %view {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.where") == 1
    assert sum("wave.gather" in line and "#waveamd.buffer" in line for line in raw_wave.splitlines()) == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.ds_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_keeps_partial_masked_buffer_load_to_local_scalar(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_masked_buffer_load_to_local(%arg0: !tt.ptr<f16> {tt.pointer_range = 2 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %one = arith.constant dense<1> : tensor<512xi32, #blocked>
    %mask = arith.cmpi slt, %range, %one : tensor<512xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    assert attrs["component_count"] == 8
    wave = output.emitted_module.text
    assert "waveamd.dma_load_lds" not in wave
    assert "wave.gather" in wave
    assert "wave.scatter" in wave
    assert "wave.load" not in wave
    assert "wave.store" not in wave
    assert wave.count("wave.where") == 1
    assert wave.count("wave.gather") == 1
    assert wave.count("wave.scatter") == 1
    assert wave.index("wave.where") < wave.index("wave.gather")
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b16" in machine
    assert machine.count("waveamdmachine.exec_if") == 1
    del ctx


def test_tlx_wave_converter_lowers_aligned_masked_buffer_load_to_local_dma(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_aligned_masked_buffer_load_to_local(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %limit: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<512xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<512xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    assert attrs["mask_component_count"] == attrs["component_count"]
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.where") == 1
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert wave.count("waveamd.dma_load_lds") == 1
    assert "zero_fill_inactive" in wave
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_lds_b128" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.ds_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_lowers_broadcast_masked_2d_buffer_load_to_local_dma(tmp_path, ):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [8, 0]], lane = [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]], warp = [[1, 0], [2, 0], [4, 0]], block = []}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_broadcast_masked_2d_buffer_load_to_local(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %row_limit: i32,
      %limit: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
    %row_limit_splat = tt.splat %row_limit : i32 -> tensor<128x1xi32, #linear>
    %row_mod = arith.remsi %row, %row_limit_splat : tensor<128x1xi32, #linear>
    %row_b = tt.broadcast %row_mod : tensor<128x1xi32, #linear> -> tensor<128x64xi32, #linear>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x64xi32, #linear> -> tensor<128x64xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<128x64xi32, #linear>
    %limit_splat = tt.splat %limit : i32 -> tensor<1x64xi32, #linear>
    %mask = arith.cmpi slt, %col, %limit_splat : tensor<1x64xi32, #linear>
    %mask_b = tt.broadcast %mask : tensor<1x64xi1, #linear> -> tensor<128x64xi1, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] mask = %mask_b into %alloc {contiguity = 8 : i32} : <f16>[tensor<128x64xi32, #linear>] -> <128x64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.where") == 1
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    assert "zero_fill_inactive" in wave
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_lds_b128" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.ds_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_lowers_masked_narrow_padded_buffer_load_to_local_dma(tmp_path, ):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [32, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [1, 0]], block = []}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0], [8, 0], [16, 0], [1, 0], [32, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_masked_narrow_padded_buffer_load_to_local(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stride: i32,
      %limit: i32 {tt.divisibility = 2 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi32, #linear>
    %stride_splat = tt.splat %stride : i32 -> tensor<64x1xi32, #linear>
    %row_scaled = arith.muli %row, %stride_splat : tensor<64x1xi32, #linear>
    %row_b = tt.broadcast %row_scaled : tensor<64x1xi32, #linear> -> tensor<64x32xi32, #linear>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x32xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x32xi32, #linear> -> tensor<64x32xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<64x32xi32, #linear>
    %limit_splat = tt.splat %limit : i32 -> tensor<1x32xi32, #linear>
    %mask = arith.cmpi slt, %col, %limit_splat : tensor<1x32xi32, #linear>
    %mask_b = tt.broadcast %mask : tensor<1x32xi1, #linear> -> tensor<64x32xi1, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] mask = %mask_b stride = %stride into %alloc {contiguity = 2 : i32} : <f16>[tensor<64x32xi32, #linear>] -> <64x32xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.where") == 1
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    assert "#waveamd.buffer, f16" in wave
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_lds_b32" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.ds_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_lowers_glu_like_masked_narrow_padded_buffer_load_to_local_dma(tmp_path, ):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [32, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [1, 0]], block = []}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0], [8, 0], [16, 0], [1, 0], [32, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_glu_like_masked_narrow_padded_buffer_load_to_local(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %row_limit: i32,
      %stride: i32,
      %limit: i32 {tt.divisibility = 2 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi32, #linear>
    %row_limit_splat = tt.splat %row_limit : i32 -> tensor<64x1xi32, #linear>
    %row_mod = arith.remsi %row, %row_limit_splat : tensor<64x1xi32, #linear>
    %stride_splat = tt.splat %stride : i32 -> tensor<64x1xi32, #linear>
    %row_scaled = arith.muli %row_mod, %stride_splat : tensor<64x1xi32, #linear>
    %row_b = tt.broadcast %row_scaled : tensor<64x1xi32, #linear> -> tensor<64x32xi32, #linear>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x32xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x32xi32, #linear> -> tensor<64x32xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<64x32xi32, #linear>
    %limit_splat = tt.splat %limit : i32 -> tensor<1x32xi32, #linear>
    %mask = arith.cmpi slt, %col, %limit_splat : tensor<1x32xi32, #linear>
    %mask_b = tt.broadcast %mask : tensor<1x32xi1, #linear> -> tensor<64x32xi1, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] mask = %mask_b stride = %stride into %alloc {contiguity = 2 : i32} : <f16>[tensor<64x32xi32, #linear>] -> <64x32xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.where") == 1
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    assert "#waveamd.buffer, f16" in wave
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_lds_b32" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.ds_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_lowers_contiguous_modulo_2d_buffer_load_to_local_dma(tmp_path, ):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [4, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]], warp = [[1, 0], [2, 0], [8, 0]], block = []}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [1, 0], [2, 0], [8, 0], [4, 0], [32, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_contiguous_modulo_2d_buffer_load_to_local(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %tile_n: i32 {tt.divisibility = 8 : i32},
      %n: i32 {tt.divisibility = 8 : i32},
      %stride: i32 {tt.divisibility = 8 : i32},
      %limit: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %ks = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %k = tt.expand_dims %ks {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi32, #linear>
    %stride_splat = tt.splat %stride : i32 -> tensor<64x1xi32, #linear>
    %row = arith.muli %k, %stride_splat : tensor<64x1xi32, #linear>
    %row_b = tt.broadcast %row : tensor<64x1xi32, #linear> -> tensor<64x256xi32, #linear>
    %tile_n_splat = tt.splat %tile_n : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %col_add = arith.addi %tile_n_splat, %cols : tensor<256xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %n_splat = tt.splat %n : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %col_mod = arith.remsi %col_add, %n_splat : tensor<256xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %col = tt.expand_dims %col_mod {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x256xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x256xi32, #linear> -> tensor<64x256xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<64x256xi32, #linear>
    %limit_splat = tt.splat %limit : i32 -> tensor<64x1xi32, #linear>
    %mask = arith.cmpi slt, %k, %limit_splat : tensor<64x1xi32, #linear>
    %mask_b = tt.broadcast %mask : tensor<64x1xi1, #linear> -> tensor<64x256xi1, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] mask = %mask_b stride = %stride into %alloc {contiguity = 8 : i32} : <f16>[tensor<64x256xi32, #linear>] -> <64x256xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.where") == 1
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_lds_b128" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.ds_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_lowers_masked_scalar_buffer_load_to_local_fallback(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_masked_scalar_buffer_load_to_local(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit = arith.constant dense<32> : tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit : tensor<64xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    assert attrs["mode"] == "symbolic_copy"
    assert attrs["has_mask"] is True
    assert attrs["component_count"] == 1
    wave = output.emitted_module.text
    assert "waveamd.dma_load_lds" not in wave
    assert wave.count("wave.gather") == 1
    assert wave.count("wave.scatter") == 1
    assert "wave.load" not in wave
    assert "wave.store" not in wave
    assert "wave.where" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b16" in machine
    del ctx


def test_tlx_wave_converter_scalarized_buffer_load_to_local_uses_structural_source_offsets(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_scalarized_dma_structural_source_i8(
      %arg0: !tt.ptr<i8> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x8xi8, #shared, #smem, mutable>
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<8x1xi32, #blocked>
    %row_scaled = arith.muli %row, %stride_splat : tensor<8x1xi32, #blocked>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %row_b = tt.broadcast %row_scaled : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %col_b = tt.broadcast %col : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %offset = arith.addi %row_b, %col_b : tensor<8x8xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<8x8xi1, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%offset] mask = %mask into %alloc : <i8>[tensor<8x8xi32, #blocked>] -> <8x8xi8, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    wave = output.emitted_module.text
    assert "waveamd.dma_load_lds" not in wave
    assert "wave.gather" in wave
    assert "wave.scatter" in wave
    assert "wave.ptr_add" not in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_u8" in machine
    del ctx


def test_tlx_wave_converter_rejects_buffer_load_to_local_mask_layout_mismatch():
    offset_layout = _fake_layout(0, 2, element_type="i32", properties={"order": (0, )})
    mask_layout = _fake_layout(1, 3, element_type="i1", properties={"order": (1, )})
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(
                0,
                kind="memdesc",
                representation="memdesc",
                element_type="f16",
            ),
            1: _converted_value(
                1,
                kind="pointer",
                representation="uniform_pointer",
                element_type="f16",
            ),
            2: _converted_value(2, element_type="i32", layout_map_id=0),
            3: _converted_value(
                3,
                kind="mask",
                representation="mask",
                element_type="i1",
                layout_map_id=1,
            ),
            4: _converted_value(
                4,
                kind="token",
                representation="token",
                element_type=None,
            ),
        },
        (offset_layout, mask_layout),
    )
    builder = converter_target_ir.TargetBuilder()
    for source_value_id, value in type_layout_program.values.items():
        if source_value_id == 4:
            continue
        builder.add_value(
            converter_target_ir.target_type_from_converted(value.type),
            source_value_id=source_value_id,
        )
    conversion_input = SimpleNamespace(
        async_issue_dependency_target_ids_by_op={},
        memdescs={0: converter_op_conversion.MemdescInfo(
            0,
            "f16",
            2,
            (64, ),
            (64, ),
            128,
        )},
        token_groups_by_id={},
        token_nodes_by_op={0: SimpleNamespace(value_id=4)},
    )
    fact_program = converter_facts.FactProgram(
        (converter_facts.Fact(0, "pointer_byte_range", 1, "pointer_range", upper=128), ),
        {1: (0, )},
    )
    op = converter_source_ir.SourceOp(
        0,
        "amdg.buffer_load_to_local",
        operands=(0, 1, 2, 3),
        results=(4, ),
        attrs={"operandSegmentSizes": (1, 1, 1, 1, 0, 0)},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_buffer_load_to_local(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "amdg.buffer_load_to_local mask layouts must match" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_lowers_splat_i1_buffer_load_to_local_mask(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_splat_i1_buffer_load_to_local_mask(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stage: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %two = arith.constant 2 : i32
    %three = arith.constant 3 : i32
    %active_a = arith.cmpi ne, %stage, %two : i32
    %active_b = arith.cmpi ne, %stage, %three : i32
    %mask_a = tt.splat %active_a : i1 -> tensor<64xi1, #blocked>
    %mask_b = tt.splat %active_b : i1 -> tensor<64xi1, #blocked>
    %mask = arith.andi %mask_a, %mask_b : tensor<64xi1, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    mask_edge = _memory_mask_edge(output.target_program, load_to_local_op)
    assert mask_edge.kind == "binary"
    assert converter_target_ir.attrs_dict(mask_edge)["operation"] == "andi"
    wave = output.emitted_module.text
    assert "scf.if" not in wave
    assert "wave.select" in wave
    assert "arith.andi" not in wave
    assert "i1 -> !wave.simd<i1" not in wave
    assert wave.count("wave.where") == 1
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b16" in machine
    del ctx


def test_tlx_wave_converter_scalarized_buffer_load_to_local_swizzled_order01(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 2, order = [0, 1]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_masked_swizzled_scalarized_dma(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x8xf16, #shared, #smem, mutable>
    %zero = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<8x8xi1, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%zero] mask = %mask into %alloc : <f16>[tensor<8x8xi32, #blocked>] -> <8x8xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    assert attrs["destination_coordinate_shape"] == (8, 8)
    assert attrs["destination_physical_offset_plan"] == "swizzled_xor"
    assert attrs["destination_physical_offset_unit"] == "element"
    assert attrs["destination_physical_element_byte_width"] == 2
    assert attrs["destination_physical_layout_kind"] == "swizzled_shared"
    assert attrs["destination_physical_order"] == (0, 1)
    assert attrs["destination_physical_bindings"] == ("logical_coords", )
    assert attrs["destination_physical_assumptions"] == ("minor_extent_divisible_by_vec", )
    assert attrs["destination_physical_proof_status"] == "symbolic_verified"
    assert attrs["destination_physical_provenance"] == "swizzled_shared"
    assert attrs["destination_physical_swizzled_vec"] == 2
    assert attrs["destination_physical_swizzled_per_phase"] == 1
    assert attrs["destination_physical_swizzled_max_phase"] == 2
    assert "destination_component_offsets" not in attrs
    assert "destination_shared_layout" not in attrs
    assert "destination_swizzled_order" not in attrs
    wave = output.emitted_module.text
    assert "waveamd.dma_load_lds" not in wave
    assert wave.count("wave.gather") == 1
    assert wave.count("wave.scatter") == 1
    assert "wave.load" not in wave
    assert "wave.store" not in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b16" in machine
    del ctx


def test_tlx_wave_converter_scalarized_buffer_load_to_local_replicated_waves(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_replicated_wave_scalarized_dma(
      %arg0: !tt.ptr<i8> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xi8, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #linear>
    %token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <i8>[tensor<64xi32, #linear>] -> <64xi8, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=2, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    assert output.emitted_module.text.count("wave.gather") == 1
    assert output.emitted_module.text.count("wave.scatter") == 1
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_packetizes_replicated_eight_wave_i8_copy(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [[1, 0], [2, 0]], lane = [[4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 1]], warp = [[0, 0], [0, 2], [0, 4]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_replicated_eight_wave_i8_dma(
      %arg0: !tt.ptr<i8> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<128x8xi8, #shared, #smem, mutable>
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
    %rows_b = tt.broadcast %rows_2d : tensor<128x1xi32, #linear> -> tensor<128x8xi32, #linear>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x8xi32, #linear>
    %cols_b = tt.broadcast %cols_2d : tensor<1x8xi32, #linear> -> tensor<128x8xi32, #linear>
    %stride = arith.constant dense<128> : tensor<128x8xi32, #linear>
    %col_offsets = arith.muli %cols_b, %stride : tensor<128x8xi32, #linear>
    %offset = arith.addi %rows_b, %col_offsets : tensor<128x8xi32, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] into %alloc {amdgpu.redundant_wave_mask = 1 : i32, contiguity = 4 : i32} : <i8>[tensor<128x8xi32, #linear>] -> <128x8xi8, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=8,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (copy_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(copy_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert wave.count("waveamd.dma_load_lds") == 1
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_lds_b32" in machine
    del ctx


def test_tlx_wave_converter_packetizes_replicated_four_wave_i8_copy(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [[1, 0], [2, 0]], lane = [[4, 0], [8, 0], [16, 0], [32, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 4]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_replicated_four_wave_i8_dma(
      %arg0: !tt.ptr<i8> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x8xi8, #shared, #smem, mutable>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi32, #linear>
    %rows_b = tt.broadcast %rows_2d : tensor<64x1xi32, #linear> -> tensor<64x8xi32, #linear>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x8xi32, #linear>
    %cols_b = tt.broadcast %cols_2d : tensor<1x8xi32, #linear> -> tensor<64x8xi32, #linear>
    %stride = arith.constant dense<64> : tensor<64x8xi32, #linear>
    %col_offsets = arith.muli %cols_b, %stride : tensor<64x8xi32, #linear>
    %offset = arith.addi %rows_b, %col_offsets : tensor<64x8xi32, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] into %alloc {amdgpu.redundant_wave_mask = 1 : i32, contiguity = 4 : i32} : <i8>[tensor<64x8xi32, #linear>] -> <64x8xi8, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=4,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (copy_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(copy_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert wave.count("waveamd.dma_load_lds") == 1
    machine = _run_waveamd_to_machine(raw_wave)
    assert "waveamdmachine.buffer_load_lds_b32" in machine
    del ctx


def test_tlx_wave_converter_lowers_scalarized_swizzled_vec4_layout(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_rejects_swizzled_scalarized_dma(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x8xf16, #shared, #smem, mutable>
    %zero = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<8x8xi1, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%zero] mask = %mask into %alloc : <f16>[tensor<8x8xi32, #blocked>] -> <8x8xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    assert attrs["mode"] == "symbolic_copy"
    assert attrs["destination_physical_offset_plan"] == "swizzled_xor"
    assert attrs["destination_physical_proof_status"] == "symbolic_verified"
    assert attrs["destination_physical_order"] == (0, 1)
    assert attrs["destination_physical_swizzled_vec"] == 4
    assert attrs["destination_physical_swizzled_per_phase"] == 1
    assert attrs["destination_physical_swizzled_max_phase"] == 4
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 1
    assert wave.count("wave.scatter") == 1
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_rejects_buffer_load_to_local_other_fallback(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_buffer_load_to_local_other(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %true = arith.constant dense<true> : tensor<64xi1, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %true other = %other into %alloc : <f16>[tensor<64xi32, #blocked>] tensor<64xf16, #blocked> -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC"
    assert "other fallback is not converted yet" in str(diagnostic)
    del ctx


def test_tlx_wave_converter_rejects_buffer_load_to_local_cache_modifier(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_buffer_load_to_local_cache(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %true = arith.constant dense<true> : tensor<64xi1, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %true cacheModifier = cv into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_CACHE_MODIFIER"
    assert "cacheModifier=" in str(diagnostic)
    del ctx


def test_tlx_wave_converter_pipeline_joins_independent_dma_packets(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_buffer_load_to_local_join(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<1024xi32, #blocked>] -> <1024xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (dma_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(dma_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    dma_matches = _dma_load_lds_matches(wave)
    assert len(dma_matches) == 2
    assert dma_matches[1].group("after") == dma_matches[0].group("after")
    assert not _wave_token_depends_on(wave, dma_matches[1].group("after"), dma_matches[0].group("token"))
    assert not tuple(_barrier_mentions_any(wave, (dma_matches[0].group("token"), )))
    del ctx


def test_tlx_wave_converter_pipeline_orders_independent_dma_load_lds_issues(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_independent_dma_load_lds_issues(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %token_a = amdg.buffer_load_to_local %arg0[%range] into %alloc_a : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %token_b = amdg.buffer_load_to_local %arg1[%range] into %alloc_b : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token_a, %token_b
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    dma_ops = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    assert len(dma_ops) == 2
    for dma_op in dma_ops:
        _assert_mechanical_symbolic_copy(
            converter_target_ir.attrs_dict(dma_op),
            masked=False,
        )
    wave = _run_wave_lower_symbolic_memory(output.emitted_module.text)
    dma_matches = _dma_load_lds_matches(wave)
    assert len(dma_matches) == 2
    assert not _wave_token_depends_on(wave, dma_matches[1].group("after"), dma_matches[0].group("token"))
    barrier_matches = tuple(_barrier_mentions_any(wave, (dma_matches[0].group("token"), )))
    assert not barrier_matches
    del ctx


def test_tlx_wave_converter_lowers_mult_warp_blocked_make_range_structurally(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
"""
    local_func = """
  tt.func public @converter_mult_warp_blocked_range() attributes {noinline = false} {
    %range = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (range_op, ) = [op for op in output.target_program.ops if op.kind == "make_range"]
    attrs = converter_target_ir.attrs_dict(range_op)
    (result_id, ) = range_op.results
    result_type = output.target_program.values[result_id].type
    assert result_type.component_count == 8
    assert attrs["coordinate_mode"] == "affine_workitem"
    assert attrs["component_bases"] == tuple(range(8))
    assert attrs["workitem_stride"] == 8
    assert "wave.binary muli" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_lowers_linear_make_range_with_block_basis(tmp_path, ):
    preamble = """
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [[64]]}>
"""
    local_func = """
  tt.func public @converter_linear_range_block_basis() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_ctas=2,
        num_warps=1,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (range_op, ) = [op for op in output.target_program.ops if op.kind == "make_range"]
    (result_id, ) = range_op.results
    assert output.target_program.values[result_id].type.component_count == 1
    assert "wave.workitem_id" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_lowers_slice_of_explicit_linear_layout(tmp_path):
    preamble = """
#parent = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [128, 0]], lane = [[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]], warp = [[1, 0], [2, 0]], block = []}>
#rows = #ttg.slice<{dim = 1, parent = #parent}>
#cols = #ttg.slice<{dim = 0, parent = #parent}>
"""
    local_func = """
  tt.func public @converter_explicit_linear_slice_range() attributes {noinline = false} {
    %rows = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #rows>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #cols>
    %row_tile = tt.expand_dims %rows {axis = 1 : i32} : tensor<256xi32, #rows> -> tensor<256x1xi32, #parent>
    %col_tile = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #cols> -> tensor<1x128xi32, #parent>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    range_ops = [op for op in source.ops if op.name == "tt.make_range"]
    row_layout = converted.layouts[converted.values[range_ops[0].results[0]].layout_map_id]
    col_layout = converted.layouts[converted.values[range_ops[1].results[0]].layout_map_id]
    row_linear = converter_layouts.distributed_linear_layout(row_layout)
    col_linear = converter_layouts.distributed_linear_layout(col_layout)

    assert converted.values[range_ops[0].results[0]].type.component_count == 8
    assert converter_layouts.linear_layout_bases(row_linear, "register") == (
        (4, ),
        (8, ),
        (128, ),
    )
    assert converted.values[range_ops[1].results[0]].type.component_count == 16
    assert converter_layouts.linear_layout_bases(col_linear, "register") == (
        (1, ),
        (2, ),
        (4, ),
        (8, ),
    )
    expand_ops = [op for op in source.ops if op.name == "tt.expand_dims"]
    assert converted.values[expand_ops[0].results[0]].type.component_count == 8
    assert converted.values[expand_ops[1].results[0]].type.component_count == 16

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    assert [op.kind for op in output.target_program.ops].count("make_range") == 2
    del ctx


def test_tlx_wave_explicit_linear_slice_restricts_replicated_bases():
    linear = converter_layouts._project_explicit_linear_slice_layout(
        (1, ),
        {
            "register_bases": ((1, 0), (2, 0), (4, 0), (0, 1), (0, 0)),
            "lane_bases": (),
            "warp_bases": (),
            "block_bases": (),
        },
        1,
        stage="test",
        source_op_index=0,
        source_value_id=0,
    )

    # Shape restriction preserves one canonical replicated register slot and
    # removes bases that no longer name distinct elements of the slice.
    assert converter_layouts.linear_layout_bases(linear, "register") == ((0, ), )


def test_tlx_wave_converter_lowers_bit_affine_linear_make_range(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [], lane = [[32], [16], [8], [4], [2], [1]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_bit_affine_linear_range() attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (range_op, ) = [op for op in output.target_program.ops if op.kind == "make_range"]
    attrs = converter_target_ir.attrs_dict(range_op)
    assert attrs["coordinate_mode"] == "bit_affine_workitem"
    assert attrs["component_bases"] == (0, )
    assert attrs["workitem_coefficients"] == (32, 16, 8, 4, 2, 1)
    assert "wave.binary divui" in output.emitted_module.text
    assert "wave.binary remui" in output.emitted_module.text
    assert "wave.binary shrui" not in output.emitted_module.text
    assert "wave.binary andi" not in output.emitted_module.text
    del ctx


def test_tlx_wave_layout_query_records_padded_physical_offset():
    layout = converter_layouts.LayoutMap(
        0,
        7,
        "padded_shared",
        (64, 128),
        "f16",
        1,
        64,
        {
            "intervals": (4, ),
            "order": (1, 0),
            "paddings": (16, ),
        },
    )

    record = converter_layouts.shared_physical_offset(
        layout,
        (64, 128),
        (0, 4),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=12,
    )

    assert record.layout_kind == "padded_shared"
    assert record.order == (1, 0)
    assert record.logical_linear_offset == 4
    assert record.element_offset == 20
    assert record.byte_offset == 40
    assert record.dword_offset == 10
    assert record.element_byte_width == 2
    assert record.logical_coords == (0, 4)
    assert record.assumptions == ("valid_padded_intervals", )
    assert record.provenance == "padded_shared"

    plan = converter_layouts.shared_physical_offset_expression_plan(
        layout,
        (64, 128),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=12,
    )
    assert plan.expression_kind == "padded_linear"
    assert plan.offset_unit == "element"
    assert plan.element_byte_width == 2
    assert plan.layout_kind == "padded_shared"
    assert plan.order == (1, 0)
    assert plan.intervals == (4, )
    assert plan.paddings == (16, )
    assert plan.assumptions == ("valid_padded_intervals", )

    attrs = converter_layouts.physical_offset_expression_plan_attrs(
        plan,
        "destination",
    )
    assert attrs["destination_physical_offset_plan"] == "padded_linear"
    assert attrs["destination_physical_offset_unit"] == "element"
    assert attrs["destination_physical_element_byte_width"] == 2
    assert attrs["destination_physical_intervals"] == (4, )
    assert attrs["destination_physical_paddings"] == (16, )

    empty_layout = converter_layouts.LayoutMap(
        2,
        9,
        "padded_shared",
        (8, 8),
        "f16",
        1,
        64,
        {
            "intervals": (),
            "order": (1, 0),
            "paddings": (),
        },
    )
    empty_plan = converter_layouts.shared_physical_offset_expression_plan(
        empty_layout,
        (8, 8),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=14,
    )
    empty_attrs = converter_layouts.physical_offset_expression_plan_attrs(
        empty_plan,
        "destination",
    )
    assert empty_plan.expression_kind == "padded_linear"
    assert empty_attrs["destination_physical_offset_plan"] == "padded_linear"
    assert empty_attrs["destination_physical_intervals"] == ()
    assert empty_attrs["destination_physical_paddings"] == ()


def test_tlx_wave_layout_query_supports_high_rank_padded_order():
    layout = converter_layouts.LayoutMap(
        0,
        7,
        "padded_shared",
        (2, 2, 2, 8, 16, 32),
        "f16",
        1,
        64,
        {
            "intervals": (512, ),
            "order": (5, 1, 3, 4, 2, 0),
            "paddings": (8, ),
        },
    )

    record = converter_layouts.shared_physical_offset(
        layout,
        layout.shape,
        (0, 0, 0, 0, 1, 0),
        2,
        stage="test",
        diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    )

    assert record.order == (5, 1, 3, 4, 2, 0)
    assert record.logical_linear_offset == 512
    assert record.element_offset == 520
    assert record.byte_offset == 1040


def test_tlx_wave_layout_query_records_transposed_padded_physical_offset():
    linear_component = LinearLayout.from_bases(
        [
            (
                "offset",
                (
                    (1, 0),
                    (2, 0),
                    (4, 0),
                    (0, 1),
                    (0, 2),
                    (0, 4),
                ),
            ),
            ("block", ()),
        ],
        ("dim0", "dim1"),
        (8, 8),
        False,
    )
    layout = converter_layouts.LayoutMap(
        3,
        10,
        "padded_shared",
        (8, 8),
        "f16",
        1,
        64,
        {
            "intervals": (4, ),
            "linear_component": linear_component,
            "order": (0, 1),
            "paddings": (16, ),
        },
    )

    record = converter_layouts.shared_physical_offset(
        layout,
        (8, 8),
        (2, 1),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=15,
    )
    assert record.logical_linear_offset == 10
    assert record.element_offset == 42
    assert record.byte_offset == 84
    assert record.dword_offset == 21

    plan = converter_layouts.shared_physical_offset_expression_plan(
        layout,
        (8, 8),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=15,
    )
    assert plan.expression_kind == "padded_linear"
    assert plan.linear_component_bases == (
        (1, 0),
        (2, 0),
        (4, 0),
        (0, 1),
        (0, 2),
        (0, 4),
    )
    attrs = converter_layouts.physical_offset_expression_plan_attrs(
        plan,
        "shared",
    )
    assert attrs["shared_physical_linear_component_bases"] == plan.linear_component_bases


def test_tlx_wave_layout_query_rejects_shared_linear_as_dense():
    layout = converter_layouts.LayoutMap(
        4,
        11,
        "linear",
        (8, 8),
        "f16",
        1,
        64,
        {
            "block_bases": (),
            "lane_bases": ((0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (0, 32)),
            "register_bases": (),
            "warp_bases": (),
        },
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_layouts.shared_physical_offset_expression_plan(
            layout,
            (8, 8),
            2,
            stage="op_conversion",
            diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            source_op_index=16,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_LOCAL_LOAD"
    assert "linear_shared" in str(diagnostic)


def test_tlx_wave_layout_query_imports_shared_linear_inverse_map():
    linear_component = LinearLayout.from_bases(
        [
            (
                "offset",
                (
                    (0, 1),
                    (0, 2),
                    (0, 4),
                    (0, 8),
                    (0, 16),
                    (1, 0),
                    (2, 8),
                    (4, 16),
                    (8, 0),
                    (16, 0),
                    (32, 0),
                    (64, 0),
                    (128, 0),
                    (0, 32),
                ),
            ),
            ("block", ()),
        ],
        ("dim0", "dim1"),
        (256, 64),
        False,
    )
    layout = converter_layouts.LayoutMap(
        5,
        12,
        "shared_linear",
        (256, 64),
        "f16",
        1,
        64,
        {
            "alignment": 16,
            "linear_component": linear_component,
            "order": (1, 0),
        },
    )

    record = converter_layouts.shared_physical_offset(
        layout,
        (256, 64),
        (3, 32),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=17,
    )
    assert record.layout_kind == "shared_linear"
    assert record.order == (1, 0)
    assert record.logical_linear_offset == 224
    assert record.element_offset == 8296
    assert record.byte_offset == 16592
    assert record.provenance == "shared_linear_inverse"

    plan = converter_layouts.shared_physical_offset_expression_plan(
        layout,
        (256, 64),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=17,
    )
    assert plan.expression_kind == "linear_shared"
    assert plan.linear_inverse_offset_bases == (
        (32, 72, 144, 256, 512, 1024, 2048, 4096),
        (1, 2, 4, 8, 16, 8192),
    )
    attrs = converter_layouts.physical_offset_expression_plan_attrs(
        plan,
        "shared",
    )
    assert attrs["shared_physical_linear_inverse_offset_bases"] == (
        (32, 72, 144, 256, 512, 1024, 2048, 4096),
        (1, 2, 4, 8, 16, 8192),
    )


def test_tlx_wave_layout_query_records_swizzled_physical_offset():
    layout = converter_layouts.LayoutMap(
        1,
        8,
        "swizzled_shared",
        (8, 8),
        "f16",
        1,
        64,
        {
            "max_phase": 2,
            "order": (0, 1),
            "per_phase": 1,
            "vec": 2,
        },
    )

    record = converter_layouts.shared_physical_offset(
        layout,
        (8, 8),
        (2, 1),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        source_op_index=13,
    )

    assert record.layout_kind == "swizzled_shared"
    assert record.order == (0, 1)
    assert record.logical_linear_offset == 10
    assert record.element_offset == 8
    assert record.byte_offset == 16
    assert record.dword_offset == 4
    assert record.assumptions == ("minor_extent_divisible_by_vec", )
    assert record.provenance == "swizzled_shared"

    plan = converter_layouts.shared_physical_offset_expression_plan(
        layout,
        (8, 8),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        source_op_index=13,
    )
    assert plan.expression_kind == "swizzled_xor"
    assert plan.offset_unit == "element"
    assert plan.element_byte_width == 2
    assert plan.layout_kind == "swizzled_shared"
    assert plan.order == (0, 1)
    assert plan.swizzled_vec == 2
    assert plan.swizzled_per_phase == 1
    assert plan.swizzled_max_phase == 2
    assert plan.assumptions == ("minor_extent_divisible_by_vec", )


def test_tlx_wave_layout_query_records_rank4_swizzled_physical_offset():
    layout = converter_layouts.LayoutMap(
        6,
        13,
        "swizzled_shared",
        (16, 16, 2, 32),
        "f16",
        1,
        64,
        {
            "max_phase": 8,
            "order": (3, 1, 0, 2),
            "per_phase": 2,
            "vec": 8,
        },
    )

    record = converter_layouts.shared_physical_offset(
        layout,
        (16, 16, 2, 32),
        (1, 2, 1, 24),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        source_op_index=18,
    )
    assert record.order == (3, 1, 0, 2)
    assert record.logical_linear_offset == 8792
    assert record.element_offset == 8784
    assert record.byte_offset == 17568

    wrapped_phase = converter_layouts.shared_physical_offset(
        layout,
        (16, 16, 2, 32),
        (0, 8, 0, 31),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        source_op_index=18,
    )
    assert wrapped_phase.logical_linear_offset == 287
    assert wrapped_phase.element_offset == 287

    plan = converter_layouts.shared_physical_offset_expression_plan(
        layout,
        (16, 16, 2, 32),
        2,
        stage="op_conversion",
        diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        source_op_index=18,
    )
    assert plan.expression_kind == "swizzled_xor"
    assert plan.order == (3, 1, 0, 2)
    assert converter_layouts.swizzled_shared_linear_component_bases(
        layout,
        (16, 16, 2, 32),
    ) == (
        (0, 0, 0, 1),
        (0, 0, 0, 2),
        (0, 0, 0, 4),
        (0, 0, 0, 8),
        (0, 0, 0, 16),
        (0, 1, 0, 0),
        (0, 2, 0, 8),
        (0, 4, 0, 16),
        (0, 8, 0, 0),
        (1, 0, 0, 0),
        (2, 0, 0, 0),
        (4, 0, 0, 0),
        (8, 0, 0, 0),
        (0, 0, 1, 0),
    )


def test_tlx_wave_converter_preserves_memdesc_reshape_as_structural_view(tmp_path):
    preamble = """
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [3, 1, 0, 2]}>
#shared1 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 8], [4, 16], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 32]]}, alignment = 16>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_memdesc_reshape() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16x2x32xf16, #shared, #smem, mutable>
    %reshape = ttg.memdesc_reshape %alloc : !ttg.memdesc<16x16x2x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared1, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (reshape_op, ) = [op for op in output.target_program.ops if op.kind == "memdesc_view"]
    assert converter_target_ir.attrs_dict(reshape_op)["view"] == "reshape"
    reshape_layout = output.type_layout_program.layouts[reshape_op.layout_map_ids[0]]
    assert reshape_layout.kind == "shared_linear"
    assert reshape_layout.shape == (256, 64)
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_lowers_replicated_generic_linear_make_range(tmp_path):
    preamble = """
#linear = #ttg.generic_linear<{register = [[1], [2]], lane = [[4], [8], [16], [32], [64], [0]], warp = [[64], [128]], block = []}>
"""
    local_func = """
  tt.func public @converter_replicated_generic_linear_range() attributes {noinline = false} {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    range_op = next(op for op in source.ops if op.name == "tt.make_range")
    value = converted.values[range_op.results[0]]
    layout = converted.layouts[value.layout_map_id]

    assert layout.kind == "generic_linear"
    assert value.type.component_count == 4
    assert layout.properties["coordinate_domain"]["coverage"] == "replicated"
    assert layout.properties["coordinate_domain"]["covered_elements"] == 256
    assert layout.properties["coordinate_domain"]["duplicate_slots"] > 0

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (target_range_op, ) = [op for op in output.target_program.ops if op.kind == "make_range"]
    attrs = converter_target_ir.attrs_dict(target_range_op)
    assert attrs["coordinate_mode"] == "layout_coordinates"
    assert attrs["component_coordinate_bases"] == ((0, ), (1, ), (2, ), (3, ))
    assert attrs["workitem_coordinate_coefficients"] == (
        (4, ),
        (8, ),
        (16, ),
        (32, ),
        (64, ),
        (0, ),
        (64, ),
        (128, ),
    )
    assert "wave.binary xori" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_keeps_overlapping_xor_basis_out_of_affine_path(tmp_path):
    preamble = """
#linear = #ttg.generic_linear<{register = [[1]], lane = [[3], [6], [12], [24], [48], [96]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_overlapping_xor_linear_range() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (range_op, ) = [op for op in output.target_program.ops if op.kind == "make_range"]
    attrs = converter_target_ir.attrs_dict(range_op)
    assert attrs["coordinate_mode"] == "layout_coordinates"
    assert attrs["component_coordinate_bases"] == ((0, ), (1, ))
    assert attrs["workitem_coordinate_coefficients"] == (
        (3, ),
        (6, ),
        (12, ),
        (24, ),
        (48, ),
        (96, ),
    )
    assert "wave.binary xori" in output.emitted_module.text
    assert attrs.get("workitem_stride") is None
    del ctx


def test_tlx_wave_converter_materializes_rank2_blocked_coordinates(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
"""
    local_func = """
  tt.func public @converter_rank2_coordinate_layout() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    constant_op = next(op for op in source.ops if op.name == "arith.constant")
    value = converted.values[constant_op.results[0]]
    layout = converted.layouts[value.layout_map_id]

    plan = converter_coordinates.layout_coordinate_plan(
        layout,
        value.type.component_count,
        value.type.lane_width,
        1,
        constant_op,
        value.value_id,
    )

    assert plan.shape == (8, 8)
    assert plan.component_bases == ((0, 0), )
    assert plan.workitem_coefficients == (
        (1, 0),
        (2, 0),
        (4, 0),
        (0, 1),
        (0, 2),
        (0, 4),
    )

    target = converter_target_ir.TargetProgram(
        (converter_target_ir.TargetValue(
            0,
            converter_target_ir.TargetType("tensor", "simd", "i32", 64, 1),
        ), ),
        (converter_target_ir.TargetOp(
            0,
            "make_range",
            results=(0, ),
            attrs=(
                converter_target_ir.TargetAttr("start", 0),
                converter_target_ir.TargetAttr("end", 64),
                converter_target_ir.TargetAttr("coordinate_mode", "layout_coordinates"),
                converter_target_ir.TargetAttr("coordinate_shape", plan.shape),
                converter_target_ir.TargetAttr(
                    "component_coordinate_bases",
                    plan.component_bases,
                ),
                converter_target_ir.TargetAttr(
                    "workitem_coordinate_coefficients",
                    plan.workitem_coefficients,
                ),
            ),
        ), ),
        (converter_target_ir.TargetRegion(0, (0, )), ),
        {},
        {},
    )
    emitted = converter_emission.emit_wave_module(target)

    assert "wave.binary divui" in emitted.text
    assert "wave.binary remui" in emitted.text
    assert "wave.binary shrui" not in emitted.text
    assert "wave.binary andi" not in emitted.text
    assert "wave.binary muli" in emitted.text
    assert "overflow<nsw>" in emitted.text
    del ctx


@pytest.mark.parametrize(
    "is_transposed,expected_coefficients",
    [
        (True, ((1, 0), (2, 0), (4, 0), (8, 0), (0, 4), (0, 8))),
        (False, ((0, 1), (0, 2), (0, 4), (0, 8), (4, 0), (8, 0))),
    ],
)
def test_tlx_wave_layout_coordinate_plan_handles_mfma_out_dim_order(
    is_transposed,
    expected_coefficients,
):
    layout = converter_layouts.LayoutMap(
        0,
        7,
        "amd_mfma",
        (16, 16),
        "f32",
        1,
        64,
        {
            "element_bit_width": 32,
            "instr_shape": (16, 16, 32),
            "is_transposed": is_transposed,
            "tiles_per_warp": (1, 1),
            "version": 4,
            "warps_per_cta": (1, 1),
        },
    )

    plan = converter_coordinates.layout_coordinate_plan(
        layout,
        1,
        64,
        1,
        SimpleNamespace(index=0),
        7,
    )

    assert plan.shape == (16, 16)
    assert plan.component_bases == ((0, 0), )
    assert plan.workitem_coefficients == expected_coefficients


def test_tlx_wave_mfma_coordinate_plan_uses_logical_dim_order():
    layout = _fake_layout(
        0,
        0,
        kind="amd_mfma",
        shape=(256, 128),
        component_count=16,
        lane_width=64,
        properties={
            "version": 4,
            "warps_per_cta": (4, 2),
            "instr_shape": (16, 16, 32),
            "is_transposed": True,
            "tiles_per_warp": (1, 1),
            "element_bit_width": 32,
        },
    )

    plan = converter_coordinates.layout_coordinate_plan(
        layout,
        layout.component_count,
        layout.lane_width,
        8,
        SimpleNamespace(index=0),
        0,
    )

    assert plan.component_bases[:4] == ((0, 0), (0, 32), (0, 64), (0, 96))
    assert plan.workitem_coefficients == (
        (1, 0),
        (2, 0),
        (4, 0),
        (8, 0),
        (0, 4),
        (0, 8),
        (0, 16),
        (16, 0),
        (32, 0),
    )


def test_tlx_wave_converter_rejects_non_injective_linear_make_range(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [0], [0]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_non_injective_linear_range() attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_TYPE_UNSUPPORTED_LAYOUT"
    assert diagnostic.stage == "type_layout"
    text = str(diagnostic)
    assert "unsupported distributed layout coordinate domain duplicate_partial" in text
    assert "bases={'register': (), 'lane': ((0,), (0,), (0,), (0,), (0,), (0,))" in text
    del ctx


def test_tlx_wave_converter_lowers_blocked_to_linear_same_lane_payloads(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_blocked_to_linear_payloads(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %converted_value = ttg.convert_layout %value : tensor<64xf32, #blocked> -> tensor<64xf32, #linear>
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<64xi1, #blocked>
    %converted_mask = ttg.convert_layout %mask : tensor<64xi1, #blocked> -> tensor<64xi1, #linear>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %converted_ptr = ttg.convert_layout %base : tensor<64x!tt.ptr<f32>, #blocked> -> tensor<64x!tt.ptr<f32>, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    convert_ops = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    assert len(convert_ops) == 3
    for convert_op in convert_ops:
        _assert_mechanical_layout_transform(output.target_program, convert_op)
    assert output.emitted_module.text.count("wave.redistribute") == 3
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.ds_bpermute_b32" not in machine
    del ctx


def test_tlx_wave_converter_lowers_blocked_component_reorder(tmp_path):
    preamble = """
#source = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [1, 1, 64], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#result = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [1, 1, 64], warpsPerCTA = [1, 1, 1], order = [2, 0, 1]}>
"""
    local_func = """
  tt.func public @converter_blocked_component_reorder() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<2x2x64xi32, #source>
    %converted = ttg.convert_layout %value : tensor<2x2x64xi32, #source> -> tensor<2x2x64xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    _assert_layout_transforms_reach_wave(output)
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.ds_bpermute_b32" not in machine
    del ctx


def test_tlx_wave_converter_lowers_blocked_cross_lane_transpose(tmp_path):
    preamble = """
#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#result = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
"""
    local_func = """
  tt.func public @converter_blocked_cross_lane_transpose() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<8x8xi32, #source>
    %converted = ttg.convert_layout %value : tensor<8x8xi32, #source> -> tensor<8x8xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    _assert_layout_transforms_reach_wave(output)
    lowered = _run_wave_lower_redistribute(output.emitted_module.text)
    assert "wave.shuffle" in lowered
    del ctx


@pytest.mark.parametrize(
    "elements,size_per_thread",
    [(64, 1), (128, 2)],
    ids=["one-register", "two-register"],
)
def test_tlx_wave_converter_keeps_join_split_structural(
    tmp_path,
    elements,
    size_per_thread,
):
    preamble = f"""
#source = #ttg.blocked<{{sizePerThread = [{size_per_thread}], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}}>
#joined = #ttg.blocked<{{sizePerThread = [{size_per_thread}, 2], threadsPerWarp = [64, 1], warpsPerCTA = [1, 1], order = [1, 0]}}>
"""
    local_func = f"""
  tt.func public @converter_join_split() attributes {{noinline = false}} {{
    %first = arith.constant dense<1> : tensor<{elements}xi32, #source>
    %second = arith.constant dense<2> : tensor<{elements}xi32, #source>
    %joined = tt.join %first, %second : tensor<{elements}xi32, #source> -> tensor<{elements}x2xi32, #joined>
    %low, %high = tt.split %joined : tensor<{elements}x2xi32, #joined> -> tensor<{elements}xi32, #source>
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (join_op, ) = [op for op in output.target_program.ops if op.kind == "join"]
    (split_op, ) = [op for op in output.target_program.ops if op.kind == "split"]
    _assert_mechanical_layout_transform(output.target_program, join_op)
    _assert_mechanical_layout_transform(output.target_program, split_op)
    _assert_layout_transforms_reach_wave(output)
    lowered = _run_wave_lower_redistribute(output.emitted_module.text)
    assert "wave.redistribute" not in lowered
    assert "wave.shuffle" not in lowered
    _run_wave_verify(lowered)
    del ctx


def test_tlx_wave_converter_keeps_reshape_transpose_structural(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
#reordered = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [], block = []}>
#transposed = #ttg.linear<{register = [], lane = [[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_reshape_transpose() attributes {noinline = false} {
    %source = arith.constant dense<1> : tensor<64xi32, #linear>
    %reshaped = tt.reshape %source allow_reorder : tensor<64xi32, #linear> -> tensor<8x8xi32, #reordered>
    %transposed_value = tt.trans %reshaped {order = array<i32: 1, 0>} : tensor<8x8xi32, #reordered> -> tensor<8x8xi32, #transposed>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )

    output = _convert_ttgir_to_wave_keep_dead(mod)

    convert_ops = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    assert len(convert_ops) == 2
    _assert_mechanical_layout_transform(
        output.target_program,
        convert_ops[0],
        transform="reshape",
    )
    _assert_mechanical_layout_transform(
        output.target_program,
        convert_ops[1],
        transform="trans",
        order=(1, 0),
    )
    assert output.emitted_module.text.count("wave.redistribute") == 2
    lowered = _run_wave_lower_redistribute(output.emitted_module.text)
    # Wave proves the transpose is an alias and lowers only the reordered
    # reshape to movement.
    assert "wave.redistribute" not in lowered
    assert "wave.shuffle" in lowered
    _run_wave_verify(lowered)
    del ctx


def test_tlx_wave_converter_serializes_equivalent_symbolic_layouts(tmp_path):
    preamble = """
#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#result = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
"""
    local_func = """
  tt.func public @converter_repeated_redistribution() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<8x8xi32, #source>
    %first = ttg.convert_layout %value : tensor<8x8xi32, #source> -> tensor<8x8xi32, #result>
    %second = ttg.convert_layout %value : tensor<8x8xi32, #source> -> tensor<8x8xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )
    output = _convert_ttgir_to_wave_keep_dead(mod)

    convert_ops = tuple(op for op in output.target_program.ops if op.kind == "layout_convert")
    assert len(convert_ops) == 2
    for convert_op in convert_ops:
        _assert_mechanical_layout_transform(output.target_program, convert_op)
    first_layout_ids = tuple(output.target_program.values[value_id].layout_map_id
                             for value_id in (*convert_ops[0].operands, *convert_ops[0].results))
    second_layout_ids = tuple(output.target_program.values[value_id].layout_map_id
                              for value_id in (*convert_ops[1].operands, *convert_ops[1].results))
    assert first_layout_ids[0] == second_layout_ids[0]
    assert (output.target_program.layouts[first_layout_ids[1]].linear_layout == output.target_program.layouts[
        second_layout_ids[1]].linear_layout)
    del ctx


def test_tlx_wave_converter_reuses_equivalent_fragment_local_load_plans(monkeypatch, ):
    shared0 = _fake_layout(
        0,
        1,
        kind="padded_shared",
        shape=(128, 128),
        element_type="bf16",
        properties={
            "intervals": (1, 16),
            "paddings": (0, 1),
            "order": (1, 0),
        },
    )
    shared1 = _fake_layout(
        1,
        2,
        kind="padded_shared",
        shape=(128, 128),
        element_type="bf16",
        properties=shared0.properties,
    )
    dot_properties = {
        "op_idx": 0,
        "k_width": 16,
        "parent_kind": "amd_mfma",
        "parent_properties": {
            "instr_shape": (32, 32, 16),
            "warps_per_cta": (4, 2),
        },
    }
    dot0 = _fake_layout(
        2,
        3,
        kind="dot_operand",
        shape=(128, 128),
        element_type="bf16",
        component_count=4,
        properties=dot_properties,
    )
    dot1 = _fake_layout(
        3,
        4,
        kind="dot_operand",
        shape=(128, 128),
        element_type="bf16",
        component_count=4,
        properties=dot_properties,
    )
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            1: _converted_value(
                1,
                kind="memdesc",
                representation="memdesc",
                element_type="bf16",
                layout_map_id=0,
            ),
            2: _converted_value(
                2,
                kind="memdesc",
                representation="memdesc",
                element_type="bf16",
                layout_map_id=1,
            ),
        },
        (shared0, shared1),
    )
    conversion_input = SimpleNamespace(
        memdescs={
            value_id: converter_op_conversion.MemdescInfo(
                value_id,
                "bf16",
                2,
                (128, 128),
                (128, 136),
                34816,
            )
            for value_id in (1, 2)
        },
        memdesc_views={
            value_id: converter_op_conversion.MemdescViewInfo(
                value_id,
                (0, 0),
                (128, 136),
            )
            for value_id in (1, 2)
        },
        fragment_local_load_plans={},
        coordinate_plans={},
    )
    calls = 0

    def fake_symbolic_plan(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"load_mode": "symbolic_mma_payload_load"}

    monkeypatch.setattr(
        converter_op_conversion,
        "_symbolic_mma_payload_gather_plan",
        fake_symbolic_plan,
    )
    first = converter_op_conversion._fragment_local_load_plan(
        conversion_input,
        type_layout_program,
        1,
        dot0,
        4,
        4,
        SimpleNamespace(index=10),
    )
    second = converter_op_conversion._fragment_local_load_plan(
        conversion_input,
        type_layout_program,
        2,
        dot1,
        4,
        4,
        SimpleNamespace(index=11),
    )

    assert calls == 1
    assert first is second


def test_tlx_wave_converter_lowers_linear_alias_convert_layout(tmp_path):
    preamble = """
#source = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
#result = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_linear_alias() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<64xi32, #source>
    %converted = ttg.convert_layout %value : tensor<64xi32, #source> -> tensor<64xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    _assert_layout_transforms_reach_wave(output)
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.ds_bpermute_b32" not in machine
    del ctx


def test_tlx_wave_converter_lowers_generic_multi_warp_linear_remap(tmp_path):
    preamble = """
#source = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[64, 0]], block = []}>
#result = #ttg.generic_linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[64, 1]], block = []}>
"""
    local_func = """
  tt.func public @converter_generic_multi_warp_linear_remap() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<128x2xi32, #source>
    %converted = ttg.convert_layout %value : tensor<128x2xi32, #source> -> tensor<128x2xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=2, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    del ctx


def test_tlx_wave_converter_lowers_bit_reversed_linear_lane_remap(tmp_path):
    preamble = """
#source = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
#result = #ttg.linear<{register = [], lane = [[32], [16], [8], [4], [2], [1]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_non_affine_linear_remap() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<64xi32, #source>
    %converted = ttg.convert_layout %value : tensor<64xi32, #source> -> tensor<64xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    result_value = output.target_program.values[convert_op.results[0]]
    result_layout = output.target_program.layouts[result_value.layout_map_id]
    lane_bases = dict(result_layout.linear_layout.bases)["lane"]
    assert tuple(basis[0] for basis in lane_bases) == (32, 16, 8, 4, 2, 1)
    del ctx


def test_tlx_wave_converter_lowers_lane_mux_register_remap(tmp_path):
    preamble = """
#src = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_lane_mux_remap(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #dst>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #dst>
    %ptr = tt.addptr %base, %range : tensor<128x!tt.ptr<f16>, #dst>, tensor<128xi32, #dst>
    %value = arith.constant dense<0.000000e+00> : tensor<128xf16, #src>
    %converted = ttg.convert_layout %value : tensor<128xf16, #src> -> tensor<128xf16, #dst>
    tt.store %ptr, %converted : tensor<128x!tt.ptr<f16>, #dst>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    assert output.emitted_module.text.count("wave.redistribute") == 1
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_lowers_slice_parent_layout_remap(tmp_path):
    preamble = """
#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #parent}>
#slice1 = #ttg.slice<{dim = 1, parent = #parent}>
"""
    local_func = """
  tt.func public @converter_slice_parent_layout_remap() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<64xi32, #slice0>
    %converted = ttg.convert_layout %value : tensor<64xi32, #slice0> -> tensor<64xi32, #slice1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_dispatches_blocked_to_mfma_base_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_blocked_to_mfma_base_remap() attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %converted = ttg.convert_layout %value : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    fact_program = converter_facts.analyze_facts(source, converted)
    token_program = converter_tokens.build_token_program(source, converted)

    target = converter_op_conversion.convert_ops(
        source,
        converted,
        fact_program,
        token_program,
    )

    (convert_op, ) = [op for op in target.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(target, convert_op)
    del ctx


def test_tlx_wave_converter_dispatches_tiled_blocked_to_mfma_base_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_tiled_blocked_to_mfma_base_remap() attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #blocked>
    %converted = ttg.convert_layout %value : tensor<256x128xf32, #blocked> -> tensor<256x128xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_mfma_layout_import_preserves_extended_metadata(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true, tilesPerWarp = [2, 2], elementBitWidth = 64}>
"""
    local_func = """
  tt.func public @converter_mfma_extended_metadata() attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<128x128xf64, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    constant_op = next(op for op in source.ops if op.name == "arith.constant")
    converted_value = converted.values[constant_op.results[0]]
    layout = converted.layouts[converted_value.layout_map_id]

    assert layout.kind == "amd_mfma"
    assert layout.shape == (128, 128)
    assert layout.element_type == "f64"
    assert layout.properties["version"] == 4
    assert layout.properties["warps_per_cta"] == (2, 2)
    assert layout.properties["instr_shape"] == (16, 16, 32)
    assert layout.properties["is_transposed"] is True
    assert layout.properties["tiles_per_warp"] == (2, 2)
    assert layout.properties["element_bit_width"] == 64
    del ctx


def test_tlx_wave_mfma_linear_layout_logical_coordinates_match_native_samples():
    cases = (
        (
            "mfma16_t_warps4x2",
            (256, 128),
            {
                "version": 4,
                "warps_per_cta": (4, 2),
                "instr_shape": (16, 16, 32),
                "is_transposed": True,
                "tiles_per_warp": (1, 1),
                "element_bit_width": 32,
            },
            (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 16, 0),
                (0, 0, 1),
                (2, 7, 3),
                (4, 31, 1),
                (8, 63, 3),
                (15, 63, 7),
                (31, 63, 7),
                (63, 63, 7),
            ),
            (
                (0, 0),
                (0, 1),
                (1, 0),
                (0, 4),
                (0, 16),
                (23, 18),
                (15, 52),
                (31, 92),
                (63, 127),
                (127, 127),
                (255, 127),
            ),
        ),
        (
            "mfma16_nt_warps4x2",
            (256, 128),
            {
                "version": 4,
                "warps_per_cta": (4, 2),
                "instr_shape": (16, 16, 32),
                "is_transposed": False,
                "tiles_per_warp": (1, 1),
                "element_bit_width": 32,
            },
            (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 16, 0),
                (0, 0, 1),
                (2, 7, 3),
                (4, 31, 1),
                (8, 63, 3),
                (15, 63, 7),
                (31, 63, 7),
                (63, 63, 7),
            ),
            (
                (0, 0),
                (1, 0),
                (0, 1),
                (4, 0),
                (0, 16),
                (18, 23),
                (4, 63),
                (28, 95),
                (63, 127),
                (127, 127),
                (255, 127),
            ),
        ),
        (
            "mfma32_t_warps2x2",
            (128, 128),
            {
                "version": 4,
                "warps_per_cta": (2, 2),
                "instr_shape": (32, 32, 16),
                "is_transposed": True,
                "tiles_per_warp": (1, 1),
                "element_bit_width": 32,
            },
            (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 16, 0),
                (0, 0, 1),
                (2, 7, 3),
                (4, 31, 1),
                (8, 63, 3),
            ),
            (
                (0, 0),
                (0, 1),
                (1, 0),
                (16, 0),
                (0, 32),
                (39, 34),
                (31, 40),
                (63, 52),
            ),
        ),
        (
            "mfma32_scaled_t_warps2x2",
            (128, 128),
            {
                "version": 4,
                "warps_per_cta": (2, 2),
                "instr_shape": (32, 32, 64),
                "is_transposed": True,
                "tiles_per_warp": (1, 1),
                "element_bit_width": 32,
            },
            (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 16, 0),
                (0, 0, 1),
                (2, 7, 3),
                (4, 31, 1),
                (8, 63, 3),
            ),
            (
                (0, 0),
                (0, 1),
                (1, 0),
                (16, 0),
                (0, 32),
                (39, 34),
                (31, 40),
                (63, 52),
            ),
        ),
        (
            "mfma32_nt_warps2x2",
            (128, 128),
            {
                "version": 4,
                "warps_per_cta": (2, 2),
                "instr_shape": (32, 32, 16),
                "is_transposed": False,
                "tiles_per_warp": (1, 1),
                "element_bit_width": 32,
            },
            (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 16, 0),
                (0, 0, 1),
                (2, 7, 3),
                (4, 31, 1),
                (8, 63, 3),
            ),
            (
                (0, 0),
                (1, 0),
                (0, 1),
                (0, 16),
                (0, 32),
                (34, 39),
                (8, 63),
                (52, 63),
            ),
        ),
        (
            "mfma16_t_tiles2x2",
            (256, 256),
            {
                "version": 4,
                "warps_per_cta": (2, 2),
                "instr_shape": (16, 16, 32),
                "is_transposed": True,
                "tiles_per_warp": (2, 2),
                "element_bit_width": 32,
            },
            (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 16, 0),
                (0, 0, 1),
                (2, 7, 3),
                (4, 31, 1),
                (8, 63, 3),
            ),
            (
                (0, 0),
                (0, 1),
                (1, 0),
                (0, 4),
                (0, 32),
                (39, 34),
                (15, 52),
                (47, 108),
            ),
        ),
        (
            "mfma16_t_f64height",
            (128, 128),
            {
                "version": 4,
                "warps_per_cta": (2, 2),
                "instr_shape": (16, 16, 32),
                "is_transposed": True,
                "tiles_per_warp": (1, 1),
                "element_bit_width": 64,
            },
            (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 16, 0),
                (0, 0, 1),
                (2, 7, 3),
                (4, 31, 1),
                (8, 63, 3),
            ),
            (
                (0, 0),
                (0, 4),
                (1, 0),
                (0, 1),
                (0, 16),
                (23, 24),
                (15, 49),
                (31, 83),
            ),
        ),
    )

    for name, shape, properties, samples, expected in cases:
        linear = converter_layouts.distributed_linear_layout_from_parts(
            "amd_mfma",
            shape,
            properties,
            64,
        )
        assert tuple(name for name, _size in linear.out_dims) == (
            "dim0",
            "dim1",
        ), name
        actual = tuple(
            converter_layouts.linear_layout_coords(
                linear,
                register,
                lane,
                warp=warp,
            ) for register, lane, warp in samples)
        assert actual == expected, name

        register_count = converter_layouts.linear_layout_in_dim_size(
            linear,
            "register",
        )
        warp_count = converter_layouts.linear_layout_in_dim_size(linear, "warp")
        all_coords = tuple(
            converter_layouts.linear_layout_coords(
                linear,
                register,
                lane,
                warp=warp,
            ) for warp in range(warp_count) for lane in range(64) for register in range(register_count))
        assert len(all_coords) == shape[0] * shape[1], name
        assert len(set(all_coords)) == len(all_coords), name
        assert all(0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1] for coord in all_coords), name


def test_tlx_wave_converter_packs_blocked_accumulator_remap_for_dot(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_blocked_accumulator_remap_for_dot() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> tensor<16x32xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> tensor<32x16xf16, #dot1>
    %base = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %acc = ttg.convert_layout %base : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<16x32xf16, #dot0> * tensor<32x16xf16, #dot1> -> tensor<16x16xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    dot_layout_convert_ops = [
        op for op in output.target_program.ops
        if op.kind == "layout_convert" and output.source_program.ops[op.source_op_index].name == "tt.dot"
    ]
    assert dot_layout_convert_ops == []
    (mma_op, ) = [op for op in output.target_program.ops if op.kind == "mma"]
    assert convert_op.results[0] == mma_op.operands[2]
    mma_attrs = converter_target_ir.attrs_dict(mma_op)
    assert mma_attrs["swap_operands_for_transposed_result"] is True
    dot_source_op = next(op for op in output.source_program.ops if op.name == "tt.dot")
    assert (output.target_program.values[mma_op.results[0]].source_value_id == dot_source_op.results[0])
    wave = output.emitted_module.text
    assert wave.count("wave.redistribute") == 1
    assert 'waveamd.fragment_pack' in wave
    assert 'waveamd.mma "mfma.f32.16x16x32.f16"' in wave
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_emits_mfma32_vector_accumulator_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mfma32_vector_accumulator_remap() attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %converted = ttg.convert_layout %value : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    wave = output.emitted_module.text
    assert wave.count("wave.redistribute") == 1
    assert "waveamd.fragment_pack" not in wave
    assert "vector<16xf32>" in wave
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_classifies_mfma_to_blocked_epilogue_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mfma_to_blocked_epilogue_remap() attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %c = arith.truncf %acc : tensor<256x256xf32, #mma> to tensor<256x256xf16, #mma>
    %converted = ttg.convert_layout %c : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    convert_layout_op = next(op for op in source.ops if op.name == "ttg.convert_layout")
    converted_result = converted.values[convert_layout_op.results[0]]
    assert converted_result.type.component_count == 256

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    assert output.emitted_module.lds_size == 0
    assert output.emitted_module.text.count("wave.redistribute") == 1
    assert "wave.alloc" not in output.emitted_module.text
    assert "wave.store" not in output.emitted_module.text
    assert "wave.load" not in output.emitted_module.text
    assert "xor(" in output.emitted_module.text
    assert "wave.wait" not in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.ds_store" in machine
    assert "waveamdmachine.ds_load" in machine
    del ctx


def test_tlx_wave_converter_lowers_mfma_epilogue_convert_before_buffer_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 128], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mfma_epilogue_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %rows = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<256x1xi32, #blocked>
    %row_scaled = arith.muli %row, %stride_splat : tensor<256x1xi32, #blocked>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %row_b = tt.broadcast %row_scaled : tensor<256x1xi32, #blocked> -> tensor<256x128xi32, #blocked>
    %col_b = tt.broadcast %col : tensor<1x128xi32, #blocked> -> tensor<256x128xi32, #blocked>
    %offset = arith.addi %row_b, %col_b : tensor<256x128xi32, #blocked>
    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c = arith.truncf %acc : tensor<256x128xf32, #mma> to tensor<256x128xf16, #mma>
    %converted = ttg.convert_layout %c : tensor<256x128xf16, #mma> -> tensor<256x128xf16, #blocked>
    amdg.buffer_store %converted, %arg0[%offset] {contiguity = 1 : i32} : tensor<256x128xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    live_ops = [
        output.target_program.ops[int(op_id)] for region in output.target_program.regions for op_id in region.op_ids
    ]
    (convert_op, ) = [op for op in live_ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    _assert_layout_transforms_reach_wave(output)
    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert "access_element_count" not in attrs
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    assert output.emitted_module.lds_size == 0
    assert "wave.alloc" not in output.emitted_module.text
    assert output.emitted_module.text.count("wave.scatter") == 1
    assert "vector<1024xf16>" in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.buffer_store_tuple_b32" in machine
    del ctx


def test_tlx_wave_converter_preserves_generic_epilogue_convert_before_buffer_store(tmp_path):
    preamble = """
#src = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_generic_epilogue_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #dst>
    %value = arith.constant dense<0.000000e+00> : tensor<128xf16, #src>
    %converted = ttg.convert_layout %value : tensor<128xf16, #src> -> tensor<128xf16, #dst>
    amdg.buffer_store %converted, %arg0[%range] {contiguity = 1 : i32} : tensor<128xf16, #dst>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    live_ops = [
        output.target_program.ops[int(op_id)] for region in output.target_program.regions for op_id in region.op_ids
    ]
    (convert_op, ) = [op for op in live_ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert "value_mode" not in attrs
    assert attrs["component_count"] == 2
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert "!wave.simd<vector<2xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_store_b16" in machine
    assert "waveamdmachine.buffer_store_b32" not in machine
    del ctx


def test_tlx_wave_converter_composes_blocked_to_mfma_metadata_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_blocked_to_mfma_metadata_remap() attributes {noinline = false} {
    %offsets = arith.constant dense<0> : tensor<256x128xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<256x128xi1, #blocked>
    %converted_offsets = ttg.convert_layout %offsets : tensor<256x128xi32, #blocked> -> tensor<256x128xi32, #mma>
    %converted_mask = ttg.convert_layout %mask : tensor<256x128xi1, #blocked> -> tensor<256x128xi1, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    convert_ops = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    assert len(convert_ops) == 2
    for convert_op in convert_ops:
        _assert_mechanical_layout_transform(output.target_program, convert_op)
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_rejects_mma_packet_truncf_layout_relabel():
    operand_layout = _fake_layout(
        0,
        0,
        kind="amd_mfma",
        shape=(16, 16),
        element_type="f32",
        properties={
            "instr_shape": (16, 16, 32),
            "is_transposed": True,
            "warps_per_cta": (1, 1),
        },
    )
    result_layout = _fake_layout(
        1,
        1,
        kind="amd_mfma",
        shape=(16, 16),
        element_type="f16",
        properties={
            "instr_shape": (16, 16, 32),
            "is_transposed": False,
            "warps_per_cta": (1, 1),
        },
    )
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(
                0,
                kind="tensor",
                representation="simd_packet",
                element_type="f32",
                layout_map_id=0,
            ),
            1: _converted_value(
                1,
                kind="tensor",
                representation="simd_packet",
                element_type="f16",
                layout_map_id=1,
            ),
        },
        (operand_layout, result_layout),
    )
    op = converter_source_ir.SourceOp(
        0,
        "arith.truncf",
        operands=(0, ),
        results=(1, ),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_mma_packet_truncf(
            converter_target_ir.TargetBuilder(),
            type_layout_program,
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "MMA packet truncf operand and result layouts must match" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_rejects_simple_op_layout_relabel():
    lhs_layout = _fake_layout(0, 0, element_type="i32", properties={"order": (0, )})
    rhs_layout = _fake_layout(1, 1, element_type="i32", properties={"order": (1, )})
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(0, element_type="i32", layout_map_id=0),
            1: _converted_value(1, element_type="i32", layout_map_id=1),
            2: _converted_value(2, element_type="i32", layout_map_id=0),
        },
        (lhs_layout, rhs_layout),
    )
    op = converter_source_ir.SourceOp(
        0,
        "arith.addi",
        operands=(0, 1),
        results=(2, ),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_source_op(
            converter_target_ir.TargetBuilder(),
            SimpleNamespace(fact_ids_by_op={}),
            type_layout_program,
            converter_facts.FactProgram((), {}),
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "arith.addi operand and result layouts must match" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_rejects_mma_packet_result_from_arbitrary_source_op():
    result_layout = _fake_layout(
        0,
        2,
        kind="amd_mfma",
        shape=(16, 16),
        element_type="f32",
        properties={
            "instr_shape": (16, 16, 32),
            "is_transposed": True,
            "warps_per_cta": (1, 1),
        },
    )
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(
                0,
                kind="scalar",
                representation="scalar",
                element_type="f32",
            ),
            1: _converted_value(
                1,
                kind="scalar",
                representation="scalar",
                element_type="f32",
            ),
            2: _converted_value(
                2,
                representation="simd_packet",
                element_type="f32",
                layout_map_id=0,
            ),
        },
        (result_layout, ),
    )
    op = converter_source_ir.SourceOp(
        0,
        "arith.minimumf",
        operands=(0, 1),
        results=(2, ),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_source_op(
            converter_target_ir.TargetBuilder(),
            SimpleNamespace(fact_ids_by_op={}),
            type_layout_program,
            converter_facts.FactProgram((), {}),
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_MMA_PACKET_PRODUCER"
    assert diagnostic.stage == "op_conversion"
    assert diagnostic.source_value_id == 2
    assert diagnostic.source_op_index == 0
    assert "arith.minimumf" in str(diagnostic)


def test_tlx_wave_converter_rejects_dot_operand_parent_layout_mismatch():
    result_properties = {
        "instr_shape": (16, 16, 32),
        "is_transposed": True,
        "warps_per_cta": (1, 1),
    }
    other_parent_properties = {
        "instr_shape": (32, 32, 16),
        "is_transposed": True,
        "warps_per_cta": (1, 1),
    }
    lhs_layout = _fake_layout(
        0,
        0,
        kind="dot_operand",
        shape=(16, 32),
        element_type="f16",
        properties={
            "k_width": 8,
            "op_idx": 0,
            "parent_kind": "amd_mfma",
            "parent_properties": other_parent_properties,
        },
    )
    rhs_layout = _fake_layout(
        1,
        1,
        kind="dot_operand",
        shape=(32, 16),
        element_type="f16",
        properties={
            "k_width": 8,
            "op_idx": 1,
            "parent_kind": "amd_mfma",
            "parent_properties": other_parent_properties,
        },
    )
    acc_layout = _fake_layout(
        2,
        2,
        kind="amd_mfma",
        shape=(16, 16),
        element_type="f32",
        properties=result_properties,
    )
    result_layout = _fake_layout(
        3,
        3,
        kind="amd_mfma",
        shape=(16, 16),
        element_type="f32",
        properties=result_properties,
    )
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(
                0,
                representation="simd_packet",
                element_type="f16",
                layout_map_id=0,
            ),
            1: _converted_value(
                1,
                representation="simd_packet",
                element_type="f16",
                layout_map_id=1,
            ),
            2: _converted_value(
                2,
                representation="simd_packet",
                element_type="f32",
                layout_map_id=2,
            ),
            3: _converted_value(
                3,
                representation="simd_packet",
                element_type="f32",
                layout_map_id=3,
            ),
        },
        (lhs_layout, rhs_layout, acc_layout, result_layout),
    )
    op = converter_source_ir.SourceOp(
        0,
        "tt.dot",
        operands=(0, 1, 2),
        results=(3, ),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_dot(
            converter_target_ir.TargetBuilder(),
            SimpleNamespace(),
            type_layout_program,
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_DOT"
    assert "parent MFMA layout must match the result layout" in str(diagnostic)


def test_tlx_wave_converter_rejects_if_yield_layout_relabel():
    result_layout = _fake_layout(0, 5, element_type="i32", properties={"order": (0, )})
    else_layout = _fake_layout(1, 2, element_type="i32", properties={"order": (1, )})
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(
                0,
                kind="scalar",
                representation="scalar",
                element_type="i1",
            ),
            1: _converted_value(1, element_type="i32", layout_map_id=0),
            2: _converted_value(2, element_type="i32", layout_map_id=1),
            5: _converted_value(5, element_type="i32", layout_map_id=0),
        },
        (result_layout, else_layout),
    )
    ops = (
        converter_source_ir.SourceOp(
            0,
            "scf.if",
            operands=(0, ),
            results=(5, ),
            region_ids=(1, 2),
        ),
        converter_source_ir.SourceOp(1, "arith.constant", results=(1, ), attrs={"value": 0}),
        converter_source_ir.SourceOp(2, "scf.yield", operands=(1, )),
        converter_source_ir.SourceOp(3, "arith.constant", results=(2, ), attrs={"value": 0}),
        converter_source_ir.SourceOp(4, "scf.yield", operands=(2, )),
    )
    conversion_input = SimpleNamespace(
        ops=ops,
        regions=(
            converter_source_ir.SourceRegion(0, (0, )),
            converter_source_ir.SourceRegion(1, (1, 2), parent_op_index=0),
            converter_source_ir.SourceRegion(2, (3, 4), parent_op_index=0),
        ),
        fact_ids_by_op={},
        token_nodes_by_op={},
        token_groups_by_id={},
        if_token_carries_by_op={},
    )
    builder = converter_target_ir.TargetBuilder()
    builder.add_value(
        converter_target_ir.target_type_from_converted(type_layout_program.values[0].type),
        source_value_id=0,
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_if(
            builder,
            conversion_input,
            type_layout_program,
            converter_facts.FactProgram((), {}),
            ops[0],
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "scf.if else yield and result" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_rejects_for_iter_arg_layout_relabel():
    result_layout = _fake_layout(0, 1, element_type="i32", properties={"order": (0, )})
    block_arg_layout = _fake_layout(1, 2, element_type="i32", properties={"order": (1, )})
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(0, element_type="i32", layout_map_id=0),
            1: _converted_value(1, element_type="i32", layout_map_id=0),
            2: _converted_value(2, element_type="i32", layout_map_id=1),
        },
        (result_layout, block_arg_layout),
    )
    op = converter_source_ir.SourceOp(
        0,
        "scf.for",
        operands=(10, 11, 12, 0),
        results=(1, ),
        region_ids=(1, ),
    )
    conversion_input = SimpleNamespace(
        regions=(
            converter_source_ir.SourceRegion(0, (0, )),
            converter_source_ir.SourceRegion(1, (), block_arg_ids=(20, 2)),
        ), )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_for(
            converter_target_ir.TargetBuilder(),
            conversion_input,
            type_layout_program,
            converter_facts.FactProgram((), {}),
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "scf.for block argument and result" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_aliases_same_lane_mfma_packet_to_blocked_slots(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_same_lane_mfma_to_blocked_remap(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %row_b = tt.broadcast %row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %col_b = tt.broadcast %col : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %stride = arith.constant dense<16> : tensor<16x16xi32, #blocked>
    %row_scaled = arith.muli %row_b, %stride : tensor<16x16xi32, #blocked>
    %offset = arith.addi %row_scaled, %col_b : tensor<16x16xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offset : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c = arith.truncf %acc : tensor<16x16xf32, #mma> to tensor<16x16xf16, #mma>
    %converted = ttg.convert_layout %c : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #blocked>
    tt.store %ptr, %converted : tensor<16x16x!tt.ptr<f16>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    _assert_layout_transforms_reach_wave(output)
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.ds_bpermute_b32" not in machine
    del ctx


def test_tlx_wave_converter_lowers_cross_lane_mfma_to_blocked_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_cross_lane_mfma_to_blocked_remap(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %row_b = tt.broadcast %row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %col_b = tt.broadcast %col : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %stride = arith.constant dense<16> : tensor<16x16xi32, #blocked>
    %row_scaled = arith.muli %row_b, %stride : tensor<16x16xi32, #blocked>
    %offset = arith.addi %row_scaled, %col_b : tensor<16x16xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offset : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c = arith.truncf %acc : tensor<16x16xf32, #mma> to tensor<16x16xf16, #mma>
    %converted = ttg.convert_layout %c : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #blocked>
    tt.store %ptr, %converted : tensor<16x16x!tt.ptr<f16>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    _assert_layout_transforms_reach_wave(output)
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert machine.count("waveamdmachine.ds_bpermute_b32") == 2
    del ctx


def test_tlx_wave_converter_lowers_fragment_f32_mfma_to_blocked_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_fragment_f32_mfma_to_blocked_remap() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> tensor<16x32xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> tensor<32x16xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %acc_mma = ttg.convert_layout %acc : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc_mma : tensor<16x32xf16, #dot0> * tensor<32x16xf16, #dot1> -> tensor<16x16xf32, #mma>
    %converted = ttg.convert_layout %dot : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    dot_result = next(source_op.results[0] for source_op in output.source_program.ops if source_op.name == "tt.dot")
    (convert_op, ) = [
        op for op in output.target_program.ops
        if op.kind == "layout_convert" and output.source_program.ops[op.source_op_index].name == "ttg.convert_layout"
        and output.source_program.ops[op.source_op_index].operands[0] == dot_result
    ]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    wave = output.emitted_module.text
    # Both source convert_layout operations remain structural conversions.
    # The bridge does not recover producer provenance to erase the accumulator
    # conversion merely because it feeds this dot.
    assert wave.count("wave.redistribute") == 2
    assert "waveamd.fragment_unpack" in wave
    _run_wave_verify(wave)
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_lowers_blocked_truncf(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
"""
    local_func = """
  tt.func public @converter_blocked_truncf() attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %truncated = arith.truncf %value : tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (cast_op, ) = [op for op in output.target_program.ops if op.kind == "float_cast"]
    attrs = converter_target_ir.attrs_dict(cast_op)
    assert attrs["operation"] == "fp_convert"
    assert "result_value_mode" not in attrs
    wave = output.emitted_module.text
    assert "arith.truncf" not in wave
    _run_wave_verify(wave)
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_preserves_rotated_linear_register_packet_cast(tmp_path):
    preamble = """
#linear = #ttg.linear<{
  register = [[0, 8], [0, 1], [0, 2], [0, 4]],
  lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
  warp = [[64, 0], [128, 0], [256, 0]],
  block = []
}>
"""
    local_func = """
  tt.func public @converter_rotated_linear_register_packet_cast() attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<512x16xf32, #linear>
    %truncated = arith.truncf %value : tensor<512x16xf32, #linear> to tensor<512x16xbf16, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (cast_op, ) = [op for op in output.target_program.ops if op.kind == "float_cast"]
    attrs = converter_target_ir.attrs_dict(cast_op)
    assert "result_value_mode" not in attrs
    assert "register_packet_width" not in attrs
    assert "register_packet_scalar_slots" not in attrs
    wave = output.emitted_module.text
    assert wave.count("wave.cast fpconvert") == 1
    assert "!wave.simd<vector<16xf32>, 64>" not in wave
    assert "!wave.simd<vector<16xbf16>, 64>" not in wave
    _run_wave_verify(wave)
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_pipeline_lowers_masked_buffer_store_with_where(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_buffer_store(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}, %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert attrs["has_mask"] is True
    assert attrs["mask_mode"] == "exec_where"
    assert "inactive_byte_offset" not in attrs
    assert "inactive_offset" not in attrs
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    assert output.emitted_module.text.count("wave.where") == 1
    assert output.emitted_module.text.count("wave.scatter") == 1
    assert 'bit_offset = <"16*offset">' in output.emitted_module.text
    assert "wave.select" not in output.emitted_module.text

    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.buffer_store_b16" in machine
    assert "waveamdmachine.exec_if" in machine
    del ctx


def test_tlx_wave_converter_applies_generic_buffer_store_ownership(tmp_path):
    preamble = """
#linear = #ttg.linear<{
  register = [[1], [0]],
  lane = [[0], [2], [4], [8], [16], [32]],
  warp = [[0], [0], [0]],
  block = []
}>
"""
    local_func = """
  tt.func public @converter_owned_buffer_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #linear>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #linear>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #linear>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #linear>
    amdg.buffer_store %value, %arg0[%range], %mask {
      amdgpu.redundant_lane_mask = 1 : i32,
      amdgpu.redundant_register_mask = 2 : i32,
      amdgpu.redundant_wave_mask = 7 : i32,
      contiguity = 2 : i32
    } : tensor<64xf16, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=8,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert attrs["redundant_register_mask"] == 2
    assert attrs["redundant_lane_mask"] == 1
    assert attrs["redundant_wave_mask"] == 7
    assert attrs["wave_count"] == 8
    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert wave.count("wave.where") == 1
    assert "wave.binary andi" in wave
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.buffer_store_b16") == 2
    assert machine.count("waveamdmachine.exec_if") == 2
    del ctx


def test_tlx_wave_converter_buffer_store_dynamic_scalar_offset_stays_structural(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_buffer_store_structural_offset(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %stride_nonnegative = arith.cmpi sge, %stride, %zero : i32
    llvm.intr.assume %stride_nonnegative : i1
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<64xi32, #blocked>
    %offset = arith.addi %range, %stride_splat : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    amdg.buffer_store %value, %arg0[%offset] {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    offset_edge, offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    assert len(store_op.operands) == 3
    assert offset_edge.kind == "binary"
    assert offset_attrs["operation"] == "addi"
    assert "wave.assume" in output.emitted_module.text
    assert output.emitted_module.text.count("wave.scatter") == 1
    del ctx


def test_tlx_wave_converter_keeps_dynamic_offset_ssa_at_store_edges(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_structural_store_offsets(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<512xi32, #blocked>
    %offset = arith.muli %range, %stride_splat : tensor<512xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<512xf16, #blocked>
    amdg.buffer_store %value, %arg0[%offset] {contiguity = 1 : i32} : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    offset_edge, offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    assert offset_edge.kind == "binary"
    assert offset_attrs["operation"] == "muli"

    wave_lines = output.emitted_module.text.splitlines()
    multiply_line = max(index for index, line in enumerate(wave_lines) if "wave.binary muli" in line)
    scatter_lines = [index for index, line in enumerate(wave_lines) if "wave.scatter " in line]
    assert len(scatter_lines) == 1
    bounded_offset_lines = [
        index for index, line in enumerate(wave_lines)
        if multiply_line < index < scatter_lines[0] and "wave.assume" in line and 'as "x"' in line
    ]
    assert len(bounded_offset_lines) == 8
    assert bounded_offset_lines[-1] < scatter_lines[0]
    assert "wave.binary muli" in output.emitted_module.text
    _run_waveamd_to_machine(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_preserves_remui_memory_offset_ssa(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_remui_store_offset(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %divisor = arith.constant dense<32> : tensor<64xi32, #blocked>
    %offset = arith.remui %range, %divisor : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    amdg.buffer_store %value, %arg0[%offset] {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=1,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    offset_op = _target_value_producer(
        output.target_program,
        store_op.operands[2],
    )
    offset_attrs = converter_target_ir.attrs_dict(offset_op)
    assert offset_op.kind == "binary"
    assert offset_attrs["operation"] == "remui"
    result_type = output.target_program.values[offset_op.results[0]].type
    assert result_type.element_type == "i32"
    assert "offset_range" not in converter_target_ir.attrs_dict(store_op)
    wave = output.emitted_module.text
    assert "wave.binary remui" in wave
    assert "wave.assume" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_store_b16" in machine
    del ctx


def test_tlx_wave_converter_vectorizes_mfma_vector_payload_buffer_store(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mfma_vector_payload_buffer_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
    %row_b = tt.broadcast %row : tensor<128x1xi32, #mma> -> tensor<128x128xi32, #mma>
    %col_b = tt.broadcast %col : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
    %stride = arith.constant dense<128> : tensor<128x128xi32, #mma>
    %row_scaled = arith.muli %row_b, %stride : tensor<128x128xi32, #mma>
    %offset = arith.addi %row_scaled, %col_b : tensor<128x128xi32, #mma>
    %acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %value = arith.truncf %acc : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    amdg.buffer_store %value, %arg0[%offset] {contiguity = 4 : i32} : tensor<128x128xf16, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert "access_element_count" not in attrs
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert "!wave.simd<vector<4xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_vectorizes_mfma_vector_payload_dynamic_stride_buffer_store(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mfma_vector_payload_dynamic_stride_buffer_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %stride_nonnegative = arith.cmpi sge, %stride, %zero : i32
    llvm.intr.assume %stride_nonnegative : i1
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
    %row_b = tt.broadcast %row : tensor<128x1xi32, #mma> -> tensor<128x128xi32, #mma>
    %col_b = tt.broadcast %col : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
    %stride_splat = tt.splat %stride : i32 -> tensor<128x128xi32, #mma>
    %row_scaled = arith.muli %row_b, %stride_splat : tensor<128x128xi32, #mma>
    %offset = arith.addi %row_scaled, %col_b : tensor<128x128xi32, #mma>
    %acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %value = arith.truncf %acc : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    amdg.buffer_store %value, %arg0[%offset] {contiguity = 4 : i32} : tensor<128x128xf16, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert "access_element_count" not in attrs
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert "!wave.simd<vector<4xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_buffer_store_dynamic_branch_fact_stays_structural(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_buffer_store_branch_fact(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stride: i32,
      %cond: i1) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    scf.if %cond {
      %stride_nonnegative = arith.cmpi sge, %stride, %zero : i32
      llvm.intr.assume %stride_nonnegative : i1
      scf.yield
    } else {
      scf.yield
    }
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<64xi32, #blocked>
    %offset = arith.addi %range, %stride_splat : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    amdg.buffer_store %value, %arg0[%offset] {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    del ctx


def test_tlx_wave_converter_buffer_store_exec_mask_uses_where_without_barrier(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_buffer_store_modes(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}, %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert attrs["mask_mode"] == "exec_where"

    exec_wave = output.emitted_module.text
    exec_machine = _run_waveamd_to_machine(exec_wave)

    assert exec_wave.count("wave.where") == 1
    assert exec_wave.count("wave.scatter") == 1
    assert "wave.select" not in exec_wave
    assert exec_machine.count("waveamdmachine.buffer_store_b16") == 1
    assert exec_machine.count("waveamdmachine.exec_if") == 1
    assert "waveamdmachine.s_barrier" not in exec_machine
    assert "waveamdmachine.s_waitcnt" not in exec_machine
    del ctx


def test_tlx_wave_converter_masks_wide_buffer_store_with_where(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_wide_masked_buffer_store(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}, %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask {contiguity = 4 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert attrs["mask_mode"] == "exec_where"
    assert "access_element_count" not in attrs
    assert "inactive_byte_offset" not in attrs
    assert "inactive_offset" not in attrs
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    assert "wave.where" in output.emitted_module.text
    assert output.emitted_module.text.count("wave.scatter") == 1
    assert "wave.select" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_vectorizes_structural_contiguous_f16_buffer_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_vector_buffer_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<512xf16, #blocked>
    amdg.buffer_store %value, %arg0[%range] : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert "access_element_count" not in attrs
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        store_op,
    )
    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_store_tuple_b32" in machine
    assert "waveamdmachine.buffer_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_vectorizes_uniform_masked_structural_buffer_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_vector_buffer_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<512xf16, #blocked>
    %mask = arith.constant dense<true> : tensor<512xi1, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert "wave.where" in wave
    assert "wave.select" not in wave
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_store_tuple_b32" in machine
    assert "waveamdmachine.buffer_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_vectorizes_packet_aligned_masked_buffer_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_packet_aligned_masked_vector_buffer_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %limit: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<512xf16, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<512xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<512xi32, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert "mask_alignment" not in attrs
    mask_edge = _memory_mask_edge(output.target_program, store_op)
    mask_attrs = converter_target_ir.attrs_dict(mask_edge)
    assert mask_edge.kind == "cmpi"
    assert "component_sources" not in mask_attrs
    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert "wave.where" in wave
    assert "wave.select" not in wave
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_store_tuple_b32" in machine
    assert "waveamdmachine.buffer_store_b16" not in machine
    del ctx


def test_tlx_wave_converter_keeps_unaligned_masked_buffer_store_scalar(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_unaligned_masked_buffer_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<512xf16, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<512xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<512xi32, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert "mask_alignment" not in attrs
    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.buffer_store_b16") == 8
    assert "waveamdmachine.buffer_store_tuple_b32" not in machine
    del ctx


def test_tlx_wave_converter_keeps_noncontiguous_structural_buffer_store_scalar(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_strided_buffer_store(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %two = arith.constant dense<2> : tensor<512xi32, #blocked>
    %offsets = arith.muli %range, %two : tensor<512xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<512xf16, #blocked>
    amdg.buffer_store %value, %arg0[%offsets] : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert wave.count("wave.scatter") == 1
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.buffer_store_b16") == 8
    assert "waveamdmachine.buffer_store_tuple_b32" not in machine
    del ctx


def test_tlx_wave_converter_rejects_buffer_store_layout_mismatch():
    offset_layout = _fake_layout(0, 2, element_type="i32", properties={"order": (0, )})
    value_layout = _fake_layout(1, 0, element_type="f16", properties={"order": (1, )})
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(0, element_type="f16", layout_map_id=1),
            1: _converted_value(
                1,
                kind="pointer",
                representation="uniform_pointer",
                element_type="f16",
            ),
            2: _converted_value(2, element_type="i32", layout_map_id=0),
        },
        (offset_layout, value_layout),
    )
    op = converter_source_ir.SourceOp(
        0,
        "amdg.buffer_store",
        operands=(0, 1, 2),
        attrs={"operandSegmentSizes": (1, 1, 1, 0, 0)},
    )
    fact_program = converter_facts.FactProgram(
        (converter_facts.Fact(0, "pointer_byte_range", 1, "pointer_range", upper=128), ),
        {1: (0, )},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_buffer_store(
            converter_target_ir.TargetBuilder(),
            SimpleNamespace(value_element_byte_widths={0: 2}),
            type_layout_program,
            fact_program,
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "amdg.buffer_store value and offsets layouts must match" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_rejects_buffer_store_mask_layout_mismatch():
    value_layout = _fake_layout(0, 0, element_type="f16", properties={"order": (0, )})
    mask_layout = _fake_layout(1, 3, element_type="i1", properties={"order": (1, )})
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(0, element_type="f16", layout_map_id=0),
            1: _converted_value(
                1,
                kind="pointer",
                representation="uniform_pointer",
                element_type="f16",
            ),
            2: _converted_value(2, element_type="i32", layout_map_id=0),
            3: _converted_value(
                3,
                kind="mask",
                representation="mask",
                element_type="i1",
                layout_map_id=1,
            ),
        },
        (value_layout, mask_layout),
    )
    builder = converter_target_ir.TargetBuilder()
    for source_value_id, value in type_layout_program.values.items():
        builder.add_value(
            converter_target_ir.target_type_from_converted(value.type),
            source_value_id=source_value_id,
        )
    op = converter_source_ir.SourceOp(
        0,
        "amdg.buffer_store",
        operands=(0, 1, 2, 3),
        attrs={"operandSegmentSizes": (1, 1, 1, 0, 1)},
    )
    fact_program = converter_facts.FactProgram(
        (converter_facts.Fact(0, "pointer_byte_range", 1, "pointer_range", upper=128), ),
        {1: (0, )},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_buffer_store(
            builder,
            SimpleNamespace(value_element_byte_widths={0: 2}),
            type_layout_program,
            fact_program,
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "amdg.buffer_store mask layouts must match" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_rejects_buffer_load_layout_mismatch():
    result_layout = _fake_layout(0, 0, element_type="f16", properties={"order": (0, )})
    offset_layout = _fake_layout(1, 2, element_type="i32", properties={"order": (1, )})
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(0, element_type="f16", layout_map_id=0),
            1: _converted_value(
                1,
                kind="pointer",
                representation="uniform_pointer",
                element_type="f16",
            ),
            2: _converted_value(2, element_type="i32", layout_map_id=1),
        },
        (result_layout, offset_layout),
    )
    op = converter_source_ir.SourceOp(
        0,
        "amdg.buffer_load",
        operands=(1, 2),
        results=(0, ),
        attrs={"operandSegmentSizes": (1, 1, 0, 0, 0)},
    )
    fact_program = converter_facts.FactProgram(
        (converter_facts.Fact(0, "pointer_byte_range", 1, "pointer_range", upper=128), ),
        {1: (0, )},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_buffer_load(
            converter_target_ir.TargetBuilder(),
            SimpleNamespace(value_element_byte_widths={0: 2}),
            type_layout_program,
            fact_program,
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "amdg.buffer_load result and offsets layouts must match" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_rejects_raw_load_layout_mismatch():
    pointer_layout = _fake_layout(0, 0, element_type="f32", properties={"order": (0, )})
    result_layout = _fake_layout(1, 1, element_type="f32", properties={"order": (1, )})
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(
                0,
                kind="pointer",
                representation="per_lane_pointer",
                element_type="f32",
                layout_map_id=0,
            ),
            1: _converted_value(1, element_type="f32", layout_map_id=1),
        },
        (pointer_layout, result_layout),
    )
    op = converter_source_ir.SourceOp(
        0,
        "tt.load",
        operands=(0, ),
        results=(1, ),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_load(
            converter_target_ir.TargetBuilder(),
            SimpleNamespace(),
            type_layout_program,
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "tt.load pointer and result layouts must match" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_rejects_raw_store_layout_mismatch():
    pointer_layout = _fake_layout(0, 0, element_type="f32", properties={"order": (0, )})
    value_layout = _fake_layout(1, 1, element_type="f32", properties={"order": (1, )})
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0: _converted_value(
                0,
                kind="pointer",
                representation="per_lane_pointer",
                element_type="f32",
                layout_map_id=0,
            ),
            1: _converted_value(1, element_type="f32", layout_map_id=1),
        },
        (pointer_layout, value_layout),
    )
    op = converter_source_ir.SourceOp(
        0,
        "tt.store",
        operands=(0, 1),
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion._convert_store(
            converter_target_ir.TargetBuilder(),
            SimpleNamespace(),
            type_layout_program,
            op,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_LAYOUT_MISMATCH"
    assert "tt.store value and pointer layouts must match" in str(diagnostic)
    assert "ttg.convert_layout" in str(diagnostic)


def test_tlx_wave_converter_vectorizes_contiguous_buffer_store_components(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_independent_buffer_store_components(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<128xf16, #blocked>
    amdg.buffer_store %value, %arg0[%range] {contiguity = 1 : i32} : tensor<128xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    scatter_lines = [line for line in wave.splitlines() if "wave.scatter" in line]
    assert len(scatter_lines) == 1
    assert "!wave.simd<vector<2xf16>, 64>" in scatter_lines[0]
    assert " after " not in scatter_lines[0]

    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.buffer_store_b32") == 1
    assert "waveamdmachine.buffer_store_b16" not in machine
    assert machine.count("waveamdmachine.s_waitcnt_vscnt") <= 1
    del ctx


def test_tlx_wave_converter_masks_byte_buffer_store_with_where(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_byte_masked_buffer_store(%arg0: !tt.ptr<i8> {tt.pointer_range = 32 : i32}, %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %value = arith.constant dense<0> : tensor<64xi8, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask {contiguity = 1 : i32} : tensor<64xi8, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert attrs["mask_mode"] == "exec_where"
    assert "inactive_byte_offset" not in attrs
    assert "inactive_offset" not in attrs
    wave = output.emitted_module.text
    assert "wave.where" in wave
    assert wave.count("wave.scatter") == 1
    assert "wave.select" not in wave
    assert "arith.constant -2147483648" not in wave

    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_store_b8" in machine
    assert "waveamdmachine.exec_if" in machine
    del ctx


def test_tlx_wave_converter_pipeline_lowers_raw_masked_load_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_raw_load_store(
      %arg0: !tt.ptr<f32>,
      %arg1: !tt.ptr<f32>,
      %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %src_ptr = tt.addptr %src_base, %range : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %loaded = tt.load %src_ptr, %mask, %other : tensor<64x!tt.ptr<f32>, #blocked>
    tt.store %dst_ptr, %loaded, %mask : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "load"]
    (store_op, ) = [op for op in output.target_program.ops if op.kind == "store"]
    load_attrs = converter_target_ir.attrs_dict(load_op)
    store_attrs = converter_target_ir.attrs_dict(store_op)
    assert load_attrs["has_mask"] is True
    assert load_attrs["has_other"] is True
    assert load_attrs["mask_mode"] == "exec_where"
    assert store_attrs["has_mask"] is True
    assert store_attrs["mask_mode"] == "exec_where"
    wave = output.emitted_module.text
    assert "waveamd.make_buffer" not in wave
    assert wave.count("wave.gather") == 1
    assert wave.count("wave.scatter") == 1
    assert "base = <\"slot\">" in wave
    assert "bit_offset = <\"0\">" in wave
    assert wave.count("wave.where") == 2
    assert wave.count("otherwise") == 1
    assert "wave.select" not in wave
    binary_module = _run_wave_compile_kernels(wave)
    assert "gpu.binary @kernels" in binary_module
    del ctx


def test_tlx_wave_converter_pipeline_lowers_masked_buffer_load_with_other(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_buffer_load(
      %arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<f32> {tt.pointer_range = 32 : i32},
      %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %mask, %other {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    amdg.buffer_store %loaded, %arg1[%range], %mask {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert attrs["has_mask"] is True
    assert attrs["has_other"] is True
    assert attrs["mask_mode"] == "exec_where"
    assert "inactive_byte_offset" not in attrs
    assert "inactive_offset" not in attrs
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        load_op,
    )
    assert output.emitted_module.text.count("waveamd.make_buffer") == 2
    assert output.emitted_module.text.count("wave.gather") == 1
    assert output.emitted_module.text.count("wave.scatter") == 1
    assert output.emitted_module.text.count("wave.where") == 2
    assert output.emitted_module.text.count('bit_offset = <"32*offset">') == 2
    assert "otherwise" in output.emitted_module.text

    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.buffer_load_b32" in machine
    assert "waveamdmachine.buffer_store_b32" in machine
    assert "waveamdmachine.exec_if" in machine
    del ctx


def test_tlx_wave_converter_vectorizes_contiguous_f16_buffer_load(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_vector_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range] {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert "access_element_count" not in attrs
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 1
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    assert wave.count("wave.extract") == 8
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_tuple_b32" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    del ctx


def test_tlx_wave_converter_vectorizes_dynamic_stride_contiguous_f16_buffer_load(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_m = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked_n = #ttg.slice<{dim = 0, parent = #blocked}>
"""
    local_func = """
  tt.func public @converter_dynamic_stride_vector_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %sy0: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_m>
    %offs_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_n>
    %offs_m_e = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<128xi32, #blocked_m> -> tensor<128x1xi32, #blocked>
    %offs_n_e = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<128xi32, #blocked_n> -> tensor<1x128xi32, #blocked>
    %stride = tt.splat %sy0 : i32 -> tensor<128x1xi32, #blocked>
    %row_offset = arith.muli %offs_m_e, %stride : tensor<128x1xi32, #blocked>
    %row_offset_b = tt.broadcast %row_offset : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %col_offset_b = tt.broadcast %offs_n_e : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %offset = arith.addi %row_offset_b, %col_offset_b : tensor<128x128xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%offset] {contiguity = 8 : i32} : tensor<128x128xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        load_op,
    )
    assert "source_access_element_count" not in attrs
    assert "access_element_count" not in attrs
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 1
    assert "!wave.simd<vector<64xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    # The complete per-thread packet contains eight contiguous transactions;
    # Wave, rather than the bridge, discovers and lowers all eight.
    assert machine.count("waveamdmachine.buffer_load_tuple_b32") == 8
    assert "waveamdmachine.buffer_load_b16" not in machine
    del ctx


def test_tlx_wave_converter_preserves_direct_buffer_cache_modifiers(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_buffer_cache_modifiers(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %ca = amdg.buffer_load %arg0[%range] cacheModifier = ca {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    %cg = amdg.buffer_load %arg0[%range] cacheModifier = cg {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    %cs = amdg.buffer_load %arg0[%range] cacheModifier = cs {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    %cv = amdg.buffer_load %arg0[%range] cacheModifier = cv {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    amdg.buffer_store %ca, %arg1[%range] cacheModifier = wb {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    amdg.buffer_store %cg, %arg1[%range] cacheModifier = cg {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    amdg.buffer_store %cs, %arg1[%range] cacheModifier = cs {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    amdg.buffer_store %cv, %arg1[%range] cacheModifier = wt {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    load_ops = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    store_ops = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    assert [converter_target_ir.attrs_dict(op)["cache_modifier"] for op in load_ops] == [2, 3, 5, 7]
    assert [converter_target_ir.attrs_dict(op)["cache_modifier"] for op in store_ops] == [4, 3, 5, 6]
    wave = output.emitted_module.text
    for modifier in ("ca", "cg", "cs", "cv"):
        assert wave.count(f"#waveamd.load_cache<{modifier}>") == 1
    for modifier in ("wb", "cg", "cs", "wt"):
        assert wave.count(f"#waveamd.store_cache<{modifier}>") == 1
    machine = _run_waveamd_to_machine(wave)
    for modifier in ("ca", "cg", "cs", "cv"):
        assert machine.count(f"#waveamd.load_cache<{modifier}>") == 1
    for modifier in ("wb", "cg", "cs", "wt"):
        assert machine.count(f"#waveamd.store_cache<{modifier}>") == 1
    del ctx


def test_tlx_wave_converter_vectorizes_contiguous_i8_buffer_load_from_structural_offsets(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_structural_vector_buffer_load(
      %arg0: !tt.ptr<i8> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range] {contiguity = 1 : i32} : tensor<512xi8, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert "access_element_count" not in attrs
    assert "result_value_mode" not in attrs
    assert "result_packet_width" not in attrs
    assert "semantic_role" not in attrs
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 1
    assert "!wave.simd<vector<8xi8>, 64>" in wave
    assert wave.count("wave.extract") == 8
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_u8" not in machine
    del ctx


def test_tlx_wave_converter_preserves_generic_i8_buffer_load_packets_into_local_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_i8_packet_load_store(
      %arg0: !tt.ptr<i8> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range] {contiguity = 8 : i32} : tensor<512xi8, #blocked>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xi8, #shared, #smem, mutable>
    ttg.local_store %loaded, %alloc : tensor<512xi8, #blocked> -> !ttg.memdesc<512xi8, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert "result_value_mode" not in attrs
    assert "result_packet_width" not in attrs
    assert "semantic_role" not in attrs
    wave = output.emitted_module.text
    assert "!wave.simd<vector<8xi8>, 64>" in wave
    assert wave.count("wave.scatter") == 1
    assert wave.count("wave.store") == 0
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_u8" not in machine
    assert "waveamdmachine.ds_store_tuple_b32" in machine
    del ctx


def test_tlx_wave_converter_vectorizes_packet_uniform_masked_f16_buffer_load(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_vector_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<512xi1, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %mask {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 1
    assert "wave.select" not in wave
    assert "wave.where" in wave
    assert "otherwise" in wave
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    assert wave.count("wave.extract") == 8
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_tuple_b32" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.buffer_load_tuple_b32" in machine
    del ctx


def test_tlx_wave_converter_vectorizes_packet_aligned_masked_f16_buffer_load(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_packet_aligned_masked_vector_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %limit: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<512xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<512xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %mask {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert "access_element_count" not in attrs
    assert "mask_alignment" not in attrs
    mask_edge = _memory_mask_edge(output.target_program, load_op)
    mask_attrs = converter_target_ir.attrs_dict(mask_edge)
    assert mask_edge.kind == "cmpi"
    assert "component_sources" not in mask_attrs
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 1
    assert "wave.where" in wave
    assert "otherwise" in wave
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_tuple_b32" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    del ctx


def test_tlx_wave_converter_zero_fills_masked_buffer_load_without_other(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_buffer_load_zero_fill(
      %arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit = arith.constant dense<32> : tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit : tensor<64xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %mask {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert attrs["has_mask"] is True
    assert attrs["has_other"] is False
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 1
    assert "wave.select" not in wave
    assert "wave.where" in wave
    assert "otherwise" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b32" in machine
    assert "waveamdmachine.exec_if" in machine
    del ctx


def test_tlx_wave_converter_packs_glu_bias_slice_into_mfma_fragment(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#bias = #ttg.slice<{dim = 0, parent = #mma}>
"""
    local_func = """
  tt.func public @converter_glu_bias_fragment(
      %bias_ptr: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #bias>
    %bias = amdg.buffer_load %bias_ptr[%offs] {contiguity = 4 : i32} : tensor<256xf16, #bias>
    %bias_f32 = arith.extf %bias : tensor<256xf16, #bias> to tensor<256xf32, #bias>
    %expanded = tt.expand_dims %bias_f32 {axis = 0 : i32} : tensor<256xf32, #bias> -> tensor<1x256xf32, #mma>
    %broadcast = tt.broadcast %expanded : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
    %zero = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %sum = arith.addf %zero, %broadcast : tensor<128x256xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)
    source_program = converter_source_import.import_source_program(mod)
    type_layout_program = converter_types.convert_source_program(source_program)
    fact_program = converter_facts.analyze_facts(source_program, type_layout_program)
    token_program = converter_tokens.build_token_program(source_program, type_layout_program)
    target_program = converter_op_conversion.convert_ops(
        source_program,
        type_layout_program,
        fact_program,
        token_program,
    )

    (load_op, ) = [op for op in target_program.ops if op.kind == "buffer_load"]
    load_attrs = converter_target_ir.attrs_dict(load_op)
    assert "result_value_mode" not in load_attrs
    (cast_op, ) = [op for op in target_program.ops if op.kind == "float_cast"]
    assert "result_value_mode" not in converter_target_ir.attrs_dict(cast_op)
    (expand_op, ) = [op for op in target_program.ops if op.kind == "expand_dims"]
    _assert_mechanical_layout_transform(
        target_program,
        expand_op,
        axis=0,
    )
    (broadcast_op, ) = [op for op in target_program.ops if op.kind == "broadcast"]
    _assert_mechanical_layout_transform(target_program, broadcast_op)

    wave = converter_emission.emit_wave_module(target_program).text
    assert wave.count("!wave.simd<vector<4xf32>, 64>") >= 4
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_broadcasts_scalar_mfma_slice_into_register_payload(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#rows = #ttg.slice<{dim = 1, parent = #mma}>
"""
    local_func = """
  tt.func public @converter_mfma_scalar_broadcast() attributes {noinline = false} {
    %row = arith.constant dense<1.000000e+00> : tensor<256xf32, #rows>
    %expanded = tt.expand_dims %row {axis = 1 : i32} : tensor<256xf32, #rows> -> tensor<256x1xf32, #mma>
    %broadcast = tt.broadcast %expanded : tensor<256x1xf32, #mma> -> tensor<256x64xf32, #mma>
    %zero = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %sum = arith.addf %zero, %broadcast : tensor<256x64xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    expand_op = next(op for op in output.target_program.ops if op.kind == "expand_dims")
    expand_type = output.target_program.values[expand_op.results[0]].type
    assert expand_type.representation == "simd_packet_tuple"
    _assert_mechanical_layout_transform(
        output.target_program,
        expand_op,
        axis=1,
    )
    broadcast_op = next(op for op in output.target_program.ops if op.kind == "broadcast")
    _assert_mechanical_layout_transform(output.target_program, broadcast_op)
    _assert_layout_transforms_reach_wave(output)
    wave = output.emitted_module.text
    assert wave.count("!wave.simd<vector<16xf32>, 64>") >= 4
    assert "!waveamd.fragment" not in wave
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_fragment_masked_load_preserves_andi_predicates(tmp_path, ):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#mma_m = #ttg.slice<{dim = 1, parent = #mma}>
#mma_n = #ttg.slice<{dim = 0, parent = #mma}>
"""
    local_func = """
  tt.func public @converter_fragment_masked_load_andi_predicates(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %M: i32,
      %N: i32,
      %sy0: i32) attributes {noinline = false} {
    %offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #mma_m>
    %offs_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #mma_n>
    %offs_m_e = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<128xi32, #mma_m> -> tensor<128x1xi32, #mma>
    %offs_n_e = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<128xi32, #mma_n> -> tensor<1x128xi32, #mma>
    %m_limit = tt.splat %M : i32 -> tensor<128x1xi32, #mma>
    %n_limit = tt.splat %N : i32 -> tensor<1x128xi32, #mma>
    %mask_m = arith.cmpi slt, %offs_m_e, %m_limit : tensor<128x1xi32, #mma>
    %mask_n = arith.cmpi slt, %offs_n_e, %n_limit : tensor<1x128xi32, #mma>
    %mask_m_b = tt.broadcast %mask_m : tensor<128x1xi1, #mma> -> tensor<128x128xi1, #mma>
    %mask_n_b = tt.broadcast %mask_n : tensor<1x128xi1, #mma> -> tensor<128x128xi1, #mma>
    %mask = arith.andi %mask_m_b, %mask_n_b : tensor<128x128xi1, #mma>
    %stride = tt.splat %sy0 : i32 -> tensor<128x1xi32, #mma>
    %row_offset = arith.muli %offs_m_e, %stride : tensor<128x1xi32, #mma>
    %row_offset_b = tt.broadcast %row_offset : tensor<128x1xi32, #mma> -> tensor<128x128xi32, #mma>
    %col_offset_b = tt.broadcast %offs_n_e : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
    %offset = arith.addi %row_offset_b, %col_offset_b : tensor<128x128xi32, #mma>
    %loaded = amdg.buffer_load %arg0[%offset], %mask {contiguity = 8 : i32} : tensor<128x128xf16, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert attrs["result_value_mode"] == "mma_packet_payload"
    assert attrs["has_mask"] is True
    mask_target_id = load_op.operands[2]
    assert not any(op.kind == "type_convert" and mask_target_id in op.results for op in output.target_program.ops)
    mask_binary = next(op for op in output.target_program.ops if op.kind == "binary" and mask_target_id in op.results)
    assert converter_target_ir.attrs_dict(mask_binary)["operation"] == "andi"
    wave = output.emitted_module.text
    # Boolean composition and per-slot predicates remain structural around one
    # complete symbolic packet.
    assert "wave.cmpi eq" not in wave
    assert wave.count("wave.where") == 1
    assert "wave.select" in wave
    assert wave.count("wave.gather") == 1
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_derives_buffer_load_packets_from_symbolic_mapping(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_nondividing_contiguity_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range] {contiguity = 5 : i32} : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 1
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_tuple_b32" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    del ctx


def test_tlx_wave_converter_masks_buffer_load_offset_assumes(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_small_buffer_load(
      %arg0: !tt.ptr<f32> {tt.pointer_range = 3 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %one = arith.constant dense<1> : tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %one : tensor<64xi32, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %mask, %other {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    _offset_edge, _offset_attrs = _memory_offset_edge(
        output.target_program,
        load_op,
    )
    assert attrs["mask_mode"] == "exec_where"
    assert "inactive_byte_offset" not in attrs
    assert "inactive_offset" not in attrs
    wave = output.emitted_module.text
    assert "wave.where" in wave
    assert "otherwise" in wave
    gather_index = wave.index("wave.gather")
    where_index = wave.rindex("wave.where", 0, gather_index)
    assume_index = wave.index("wave.assume", where_index)
    assert where_index < assume_index < gather_index
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b32" in machine
    assert "waveamdmachine.exec_if" in machine
    del ctx


def test_tlx_wave_converter_pipeline_groups_mult_warp_padded_dma(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#shared = #ttg.padded_shared<[512:+16] {order = [0], shape = [4096]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_padded_dma(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<4096xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<4096xi32, #blocked>] -> <4096xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (dma_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(dma_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert wave.count("waveamd.dma_load_lds") == 1
    del ctx


def test_tlx_wave_converter_pipeline_groups_2d_padded_dma_physical_linear_slots(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [128, 0]], lane = [[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]], warp = [[1, 0], [2, 0]], block = []}>
#shared = #ttg.padded_shared<[1024:+32] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_2d_padded_dma_physical_linear_slots(
      %arg0: !tt.ptr<i8> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x128xi8, #shared, #smem, mutable>
    %rows = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<256x1xi32, #linear>
    %stride_splat = tt.splat %stride : i32 -> tensor<256x1xi32, #linear>
    %row_scaled = arith.muli %row, %stride_splat : tensor<256x1xi32, #linear>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xi32, #linear>
    %row_b = tt.broadcast %row_scaled : tensor<256x1xi32, #linear> -> tensor<256x128xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x128xi32, #linear> -> tensor<256x128xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<256x128xi32, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] into %alloc : <i8>[tensor<256x128xi32, #linear>] -> <256x128xi8, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (dma_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(dma_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" in wave
    _run_waveamd_to_machine(raw_wave)
    del ctx


def test_tlx_wave_converter_pipeline_does_not_assume_runtime_modulo_contiguity(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], warp = [[1, 0], [2, 0], [4, 0]], block = []}>
#shared = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [128, 128]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_selects_narrow_padded_dma_packet(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %row0: i32,
      %col0: i32,
      %stride: i32,
      %m: i32,
      %n: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %row0_s = tt.splat %row0 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %row_raw = arith.addi %row0_s, %rows : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %m_s = tt.splat %m : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %row = arith.remsi %row_raw, %m_s : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %row_2d = tt.expand_dims %row {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
    %stride_s = tt.splat %stride : i32 -> tensor<128x1xi32, #linear>
    %row_scaled = arith.muli %row_2d, %stride_s : tensor<128x1xi32, #linear>
    %row_b = tt.broadcast %row_scaled : tensor<128x1xi32, #linear> -> tensor<128x128xi32, #linear>
    %col0_s = tt.splat %col0 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %col_raw = arith.addi %col0_s, %cols : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %n_s = tt.splat %n : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %col = arith.remsi %col_raw, %n_s : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %col_2d = tt.expand_dims %col {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xi32, #linear>
    %col_b = tt.broadcast %col_2d : tensor<1x128xi32, #linear> -> tensor<128x128xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<128x128xi32, #linear>
    %token = amdg.buffer_load_to_local %arg0[%offset] stride = %stride into %alloc {contiguity = 2 : i32} : <f16>[tensor<128x128xi32, #linear>] -> <128x128xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (dma_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(dma_op)
    _assert_mechanical_symbolic_copy(attrs, masked=False)
    raw_wave = output.emitted_module.text
    assert raw_wave.count("wave.gather") == 1
    assert raw_wave.count("wave.scatter") == 1
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    assert "waveamd.dma_load_lds" not in wave
    _run_waveamd_to_machine(raw_wave)
    del ctx


def test_tlx_wave_converter_scalarized_masked_dma_preserves_rank1_padded_offsets(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#shared = #ttg.padded_shared<[512:+16] {order = [0], shape = [4096]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_masked_padded_scalarized_dma(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<4096xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %limit = arith.constant dense<4096> : tensor<4096xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit : tensor<4096xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask into %alloc : <f16>[tensor<4096xi32, #blocked>] -> <4096xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    _assert_mechanical_symbolic_copy(attrs, masked=True)
    assert attrs["component_count"] == 8
    assert output.emitted_module.text.count("wave.gather") == 1
    assert output.emitted_module.text.count("wave.scatter") == 1
    assert "wave.load" not in output.emitted_module.text
    assert "wave.store" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_warp_tiled_mfma_dot(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_warp_tiled_mfma_dot() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<256x64xf16, #dot0> * tensor<64x128xf16, #dot1> -> tensor<256x128xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    attrs_by_kind = {
        op.kind: converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind in {"mma_packet_constant", "mma"}
    }
    assert attrs_by_kind["mma_packet_constant"]["component_count"] == 16
    assert attrs_by_kind["mma"]["m_tiles"] == 4
    assert attrs_by_kind["mma"]["n_tiles"] == 4
    assert attrs_by_kind["mma"]["k_tiles"] == 2
    local_load_attrs = _packetized_local_load_attrs(output)
    assert [attrs["component_count"] for attrs in local_load_attrs] == [8, 8]
    assert [attrs["packet_count"] for attrs in local_load_attrs] == [8, 8]
    assert [attrs["packet_width"] for attrs in local_load_attrs] == [8, 8]
    assert [attrs["shared_physical_offset_plan"] for attrs in local_load_attrs] == [
        "dense_row_major",
        "dense_row_major",
    ]
    assert all("shared_layout_kind" not in attrs for attrs in local_load_attrs)
    assert [attrs["coordinate_shape"] for attrs in local_load_attrs] == [
        (256, 64),
        (64, 128),
    ]
    assert local_load_attrs[1]["memdesc_shape"] == (64, 128)
    assert local_load_attrs[0]["packet_coordinate_bases"] == (
        (0, 0),
        (0, 32),
        (64, 0),
        (64, 32),
        (128, 0),
        (128, 32),
        (192, 0),
        (192, 32),
    )
    assert local_load_attrs[1]["packet_coordinate_bases"] == (
        (0, 0),
        (32, 0),
        (0, 32),
        (32, 32),
        (0, 64),
        (32, 64),
        (0, 96),
        (32, 96),
    )
    assert all("load_mode" not in attrs for attrs in local_load_attrs)
    assert all("wave_tile_axis" not in attrs for attrs in local_load_attrs)
    assert [op.kind for op in output.target_program.ops].count("layout_convert") == 0
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == sum(attrs["packet_count"] for attrs in local_load_attrs)
    assert "vector<8xf16>" in wave
    assert "native_register_layout" not in wave
    assert "waveamd.fragment_fill" not in wave
    assert wave.count('waveamd.mma "mfma.f32.16x16x32.f16"') == 32
    fragment_lines = [line for line in wave.splitlines() if "!waveamd.fragment" in line]
    assert fragment_lines
    assert all(
        any(operation in line for operation in (
            "waveamd.fragment_pack",
            "waveamd.fragment_unpack",
            "waveamd.mma",
        )) for line in fragment_lines)
    del ctx


def test_tlx_wave_converter_pipeline_lowers_typed_mfma_fragment_constants(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_typed_mfma_fragment_constants() attributes {noinline = false} {
    %finite = arith.constant dense<1.250000e+00> : tensor<16x16xf32, #mma>
    %negative_inf = arith.constant dense<0xFF800000> : tensor<16x16xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    constants = [
        converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "mma_packet_constant"
    ]
    assert [attrs["value"] for attrs in constants] == [1.25, float("-inf")]
    assert all(attrs["element_type"] == "f32" for attrs in constants)
    assert output.emitted_module.text.count("wave.constant") == 2
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_pipeline_keeps_mfma_fragment_math_structural(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mfma_fragment_splat_math() attributes {noinline = false} {
    %lhs = arith.constant dense<1.250000e+00> : tensor<16x16xf32, #mma>
    %rhs = arith.constant dense<2.500000e+00> : tensor<16x16xf32, #mma>
    %maximum = arith.maxnumf %lhs, %rhs : tensor<16x16xf32, #mma>
    %product = arith.mulf %maximum, %lhs : tensor<16x16xf32, #mma>
    %sum = arith.addf %maximum, %maximum : tensor<16x16xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert not re.search(r"wave\.fmul .* -> !wave\.simd<f32, 64>", wave)
    assert re.search(r"wave\.fmul .*vector<4xf32>", wave)
    assert re.search(r"wave\.fadd .*vector<4xf32>", wave)
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_pipeline_selects_mfma_fragment_payloads(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_select_mfma_fragment_payloads() attributes {noinline = false} {
    %condition = arith.constant dense<true> : tensor<16x16xi1, #mma>
    %finite = arith.constant dense<1.250000e+00> : tensor<16x16xf32, #mma>
    %negative_inf = arith.constant dense<0xFF800000> : tensor<16x16xf32, #mma>
    %selected = arith.select %condition, %finite, %negative_inf : tensor<16x16xi1, #mma>, tensor<16x16xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops].count("select") == 1
    assert "wave.select" in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    del ctx


@pytest.mark.parametrize(
    ("preamble", "shape", "layout"),
    (
        (
            """
#layout = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
""",
            "128",
            "#layout",
        ),
        (
            """
#layout = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
""",
            "16x16",
            "#layout",
        ),
    ),
)
def test_tlx_wave_converter_preserves_distributed_compare_selects_for_wave(
    tmp_path,
    preamble,
    shape,
    layout,
):
    local_func = f"""
  tt.func public @converter_structural_compare_select() attributes {{noinline = false}} {{
    %lhs = arith.constant dense<0> : tensor<{shape}xi32, {layout}>
    %rhs = arith.constant dense<1> : tensor<{shape}xi32, {layout}>
    %condition = arith.cmpi slt, %lhs, %rhs : tensor<{shape}xi32, {layout}>
    %finite = arith.constant dense<1.250000e+00> : tensor<{shape}xf32, {layout}>
    %negative_inf = arith.constant dense<0xFF800000> : tensor<{shape}xf32, {layout}>
    %selected = arith.select %condition, %finite, %negative_inf : tensor<{shape}xi1, {layout}>, tensor<{shape}xf32, {layout}>
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    kinds = [op.kind for op in output.target_program.ops]
    assert kinds.count("cmpi") == 1
    assert kinds.count("select") == 1
    compare = next(op for op in output.target_program.ops if op.kind == "cmpi")
    select = next(op for op in output.target_program.ops if op.kind == "select")
    assert compare.target_op_id < select.target_op_id
    assert compare.source_op_index < select.source_op_index
    assert select.operands[0] == compare.results[0]

    compare_select_lines = [
        line for line in output.emitted_module.text.splitlines() if "wave.cmpi slt" in line or "wave.select" in line
    ]
    assert compare_select_lines
    first_select = next(index for index, line in enumerate(compare_select_lines) if "wave.select" in line)
    assert all("wave.cmpi slt" in line for line in compare_select_lines[:first_select])
    assert all("wave.select" in line for line in compare_select_lines[first_select:])

    _run_wave_verify(output.emitted_module.text)
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "cmp_lt_i32" in machine
    assert "cselect_b32" in machine or "cndmask_b32" in machine
    del ctx


def test_tlx_wave_converter_pipeline_lowers_mfma_fragment_softmax_math(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mfma_fragment_softmax_math() attributes {noinline = false} {
    %lhs = arith.constant dense<1.250000e+00> : tensor<16x16xf32, #mma>
    %rhs = arith.constant dense<2.500000e+00> : tensor<16x16xf32, #mma>
    %maximum = arith.maxnumf %lhs, %rhs : tensor<16x16xf32, #mma>
    %product = arith.mulf %lhs, %rhs fastmath<contract> : tensor<16x16xf32, #mma>
    %sum = arith.addf %product, %maximum fastmath<contract> : tensor<16x16xf32, #mma>
    %exponential = math.exp2 %sum : tensor<16x16xf32, #mma>
    %quotient = arith.divf %exponential, %lhs : tensor<16x16xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    operations = [
        converter_target_ir.attrs_dict(op)["operation"]
        for op in output.target_program.ops
        if op.kind in {"float_binary", "float_unary"}
    ]
    assert operations == ["maxnumf", "mulf", "addf", "exp2", "divf"]
    wave = output.emitted_module.text
    assert "wave.fmax" in wave
    assert "wave.fma " not in wave
    assert "wave.fexp2" in wave
    assert "wave.frcp" in wave
    _run_wave_verify(wave)
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_pipeline_reduces_mfma_fragments_within_waves(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_reduce_mfma_fragments() attributes {noinline = false} {
    %input = arith.constant dense<1.250000e+00> : tensor<256x64xf32, #mma>
    %maximum_propagate = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.maximumf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<256x64xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %maximum_number = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<256x64xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %sum = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<256x64xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    reductions = [op for op in output.target_program.ops if op.kind == "reduction"]
    assert [converter_target_ir.attrs_dict(op)["operation"] for op in reductions] == [
        "maximumf",
        "maxnumf",
        "addf",
    ]
    assert all(len(converter_target_ir.attrs_dict(op)["component_terms"]) == 2 for op in reductions)
    assert all(
        all(len(terms) == 64 for terms in converter_target_ir.attrs_dict(op)["component_terms"]) for op in reductions)
    wave = output.emitted_module.text
    assert "wave.shuffle" in wave
    assert "wave.fmax" in wave
    assert "wave.fadd" in wave
    _run_wave_verify(wave)
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_preserves_explicit_local_load_wait_token(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_explicit_local_load_wait_token() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %wait = ttg.async_wait {num = 0 : i32}
    %value = ttg.local_load %alloc token %wait : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> tensor<16x32xf16, #dot0>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (wait_op, ) = [op for op in output.target_program.ops if op.kind == "async_wait"]
    (load_op, ) = [op for op in output.target_program.ops if op.kind == "local_load"]
    assert load_op.operands[1:] == wait_op.results
    assert converter_target_ir.attrs_dict(load_op)["explicit_dependency_count"] == 1
    load_lines = [line for line in output.emitted_module.text.splitlines() if "wave.gather" in line]
    assert load_lines
    assert all(" after " in line for line in load_lines)
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_partial_wait_keeps_retained_group_issue_only(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_partial_wait_issue_only(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x256xf16, #shared, #smem, mutable>
    %slot0 = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<256xf16, #shared, #smem, mutable>
    %slot1 = ttg.memdesc_index %alloc[%c1] : !ttg.memdesc<2x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<256xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %copy0 = amdg.buffer_load_to_local %arg0[%range] into %slot0 : <f16>[tensor<256xi32, #blocked>] -> <256xf16, #shared, #smem, mutable>
    %group0 = ttg.async_commit_group tokens %copy0
    %copy1 = amdg.buffer_load_to_local %arg0[%range] into %slot1 : <f16>[tensor<256xi32, #blocked>] -> <256xf16, #shared, #smem, mutable>
    %group1 = ttg.async_commit_group tokens %copy1
    %wait = ttg.async_wait {num = 1 : i32}
    %loaded = ttg.local_load %slot0 {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<256xf16, #shared, #smem, mutable> -> tensor<256xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    commit_ops = [op for op in output.target_program.ops if op.kind == "async_commit_group"]
    (issue_op, ) = [op for op in output.target_program.ops if op.kind == "issue_token"]
    (wait_op, ) = [op for op in output.target_program.ops if op.kind == "async_wait"]
    wait_attrs = converter_target_ir.attrs_dict(wait_op)
    assert wait_attrs["waited_group_ids"] != wait_attrs["retained_group_ids"]
    assert wait_attrs["completed_group_dependency_count"] == 1
    assert wait_attrs["retained_issue_dependency_count"] == 1
    assert "lds_release_dependency_count" not in wait_attrs
    assert wait_op.operands == (commit_ops[0].results[0], issue_op.results[0])
    assert issue_op.operands == commit_ops[1].results
    assert output.target_program.values[issue_op.results[0]].event_domain == (
        converter_target_ir.EVENT_DOMAIN_DMA_ISSUE)
    wave = output.emitted_module.text
    assert wave.count("wave.issue_token") == 1
    issue_line = next(line for line in wave.splitlines() if "wave.issue_token" in line)
    issue_token = _ssa_result_name(issue_line)
    assert any("wave.after" in line and issue_token in line for line in wave.splitlines())
    _run_wave_verify(wave)
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.issue_token" in machine
    del ctx


def test_tlx_wave_converter_threads_dominating_wait_without_relaxing_ordinary_load(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_dominating_local_load_wait_token() attributes {noinline = false} {
    %relaxed_alloc = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %ordinary_alloc = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %wait = ttg.async_wait {num = 0 : i32}
    %stored = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #blocked>
    ttg.local_store %stored, %relaxed_alloc : tensor<16x32xf16, #blocked> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %relaxed = ttg.local_load %relaxed_alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> tensor<16x32xf16, #dot0>
    ttg.local_store %stored, %ordinary_alloc : tensor<16x32xf16, #blocked> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %ordinary = ttg.local_load %ordinary_alloc : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> tensor<16x32xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (wait_op, ) = [op for op in output.target_program.ops if op.kind == "async_wait"]
    relaxed_store, ordinary_store = [op for op in output.target_program.ops if op.kind == "local_store"]
    local_loads = [op for op in output.target_program.ops if op.kind == "local_load"]
    (relaxed_load, ) = [op for op in local_loads if converter_target_ir.attrs_dict(op)["synced_via_async_wait"]]
    (ordinary_load, ) = [op for op in local_loads if not converter_target_ir.attrs_dict(op)["synced_via_async_wait"]]
    assert relaxed_load.operands[1:] == wait_op.results
    assert relaxed_store.operands[2:] == ()
    assert ordinary_store.operands[2:] == ()
    assert ordinary_load.operands[1:] == ()
    relaxed_attrs = converter_target_ir.attrs_dict(relaxed_load)
    ordinary_attrs = converter_target_ir.attrs_dict(ordinary_load)
    assert relaxed_attrs["synced_via_async_wait"] is True
    assert relaxed_attrs["explicit_dependency_count"] == 1
    assert ordinary_attrs["synced_via_async_wait"] is False
    assert ordinary_attrs["explicit_dependency_count"] == 0
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_marked_tokenless_load_consumes_wait_once(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_wait_ready_ordinary_local_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %copy = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %copy
    %wait = ttg.async_wait %group {num = 0 : i32}
    %loaded = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<512xf16, #shared, #smem, mutable> -> tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=4,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (wait_op, ) = [op for op in output.target_program.ops if op.kind == "async_wait"]
    (dma_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    (load_op, ) = [op for op in output.target_program.ops if op.kind == "local_load"]
    assert converter_target_ir.attrs_dict(dma_op)["mode"] == "symbolic_copy"
    load_attrs = converter_target_ir.attrs_dict(load_op)
    assert load_attrs["synced_via_async_wait"] is True
    assert load_attrs["explicit_dependency_count"] == 1
    assert load_op.operands[1:] == wait_op.results

    wave = output.emitted_module.text
    load_line = next(line for line in wave.splitlines() if "wave.gather" in line and "#wave.shared" in line)
    assert " after " in load_line
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_carries_implicit_wait_readiness_across_loop(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_wait_ds_loop_epoch() attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<256xf16, #shared, #smem, mutable>
    %preheader_wait = ttg.async_wait {num = 0 : i32}
    %preheader_load = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<256xf16, #shared, #smem, mutable> -> tensor<256xf16, #blocked>
    scf.for %i = %c0 to %c2 step %c1 {
      %body_load = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<256xf16, #shared, #smem, mutable> -> tensor<256xf16, #blocked>
      %body_wait = ttg.async_wait {num = 0 : i32}
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    token_program = converter_tokens.build_token_program(source, converted)
    (source_loop, ) = [op for op in source.ops if op.name == "scf.for"]
    source_waits = [op for op in source.ops if op.name == "ttg.async_wait"]
    readiness_carries = [
        carry for carry in token_program.loop_token_carries_by_op[source_loop.index] if carry.readiness_carry
    ]
    assert readiness_carries == [
        converter_tokens.LoopTokenCarry(
            loop_op_index=source_loop.index,
            init_source_value_id=source_waits[0].results[0],
            yield_source_value_id=source_waits[1].results[0],
            add_issue_dependency=False,
            issue_dependency_op_indices=(),
            readiness_carry=True,
        )
    ]

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (loop_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    loop_attrs = converter_target_ir.attrs_dict(loop_op)
    wait_ops = [op for op in output.target_program.ops if op.kind == "async_wait"]
    load_ops = [op for op in output.target_program.ops if op.kind == "local_load"]
    target_region = output.target_program.regions[loop_op.region_ids[0]]
    assert loop_attrs["source_result_count"] == 0
    assert loop_attrs["init_arg_count"] == 1
    assert loop_op.operands[-1:] == wait_ops[0].results
    assert output.target_program.values[loop_op.results[-1]].event_domain == (
        converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY)
    assert output.target_program.values[target_region.block_arg_ids[-1]].event_domain == (
        converter_target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY)
    assert load_ops[0].operands[1:] == wait_ops[0].results
    assert load_ops[1].operands[1:] == target_region.block_arg_ids[-1:]
    assert target_region.yield_value_ids[-1:] == wait_ops[1].results
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_closes_sequential_loop_wait_carries(tmp_path):
    local_func = """
  tt.func public @converter_sequential_wait_loop_epochs() attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %first_wait = ttg.async_wait {num = 0 : i32}
      ttg.barrier all
    }
    scf.for %i = %c0 to %c2 step %c1 {
      %second_wait = ttg.async_wait {num = 0 : i32}
      ttg.barrier all
    }
    ttg.barrier all
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4)

    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    token_program = converter_tokens.build_token_program(source, converted)
    source_loops = [op for op in source.ops if op.name == "scf.for"]
    source_waits = [op for op in source.ops if op.name == "ttg.async_wait"]
    assert [
        tuple(carry
              for carry in token_program.loop_token_carries_by_op[loop.index]
              if carry.readiness_carry)
        for loop in source_loops
    ] == [
        (converter_tokens.LoopTokenCarry(
            loop_op_index=source_loops[0].index,
            init_source_value_id=None,
            yield_source_value_id=source_waits[0].results[0],
            add_issue_dependency=False,
            issue_dependency_op_indices=(),
            readiness_carry=True,
        ), ),
        (converter_tokens.LoopTokenCarry(
            loop_op_index=source_loops[1].index,
            init_source_value_id=source_waits[0].results[0],
            yield_source_value_id=source_waits[1].results[0],
            add_issue_dependency=False,
            issue_dependency_op_indices=(),
            readiness_carry=True,
        ), ),
    ]

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    target_loops = [op for op in output.target_program.ops if op.kind == "for_loop"]
    assert target_loops[1].operands[-1:] == target_loops[0].results[-1:]
    assert all(converter_target_ir.attrs_dict(loop)["source_result_count"] == 0 for loop in target_loops)
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_does_not_carry_readiness_without_backedge_wait(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_wait_ds_open_loop_epoch() attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<256xf16, #shared, #smem, mutable>
    %first_wait = ttg.async_wait {num = 0 : i32}
    scf.for %i = %c0 to %c2 step %c1 {
      %loaded = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<256xf16, #shared, #smem, mutable> -> tensor<256xf16, #blocked>
    }
    %release_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    token_program = converter_tokens.build_token_program(source, converted)
    (source_loop, ) = [op for op in source.ops if op.name == "scf.for"]
    assert not [
        carry for carry in token_program.loop_token_carries_by_op.get(
            source_loop.index,
            (),
        ) if carry.readiness_carry
    ]

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wait_ops = [op for op in output.target_program.ops if op.kind == "async_wait"]
    (loop_op, ) = [op for op in output.target_program.ops if op.kind == "for_loop"]
    (load_op, ) = [op for op in output.target_program.ops if op.kind == "local_load"]
    assert not loop_op.results
    assert load_op.operands[1:] == wait_ops[0].results
    assert not wait_ops[1].operands
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_does_not_merge_readiness_when_wait_precedes_if(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_wait_ds_if_epoch(%cond: i1) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<256xf16, #shared, #smem, mutable>
    %first_wait = ttg.async_wait {num = 0 : i32}
    scf.if %cond {
      %loaded = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<256xf16, #shared, #smem, mutable> -> tensor<256xf16, #blocked>
    }
    %release_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    token_program = converter_tokens.build_token_program(source, converted)
    (source_if, ) = [op for op in source.ops if op.name == "scf.if"]
    assert not token_program.if_token_carries_by_op.get(source_if.index)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wait_ops = [op for op in output.target_program.ops if op.kind == "async_wait"]
    (if_op, ) = [op for op in output.target_program.ops if op.kind == "if"]
    (load_op, ) = [op for op in output.target_program.ops if op.kind == "local_load"]
    assert not if_op.results
    assert load_op.operands[1:] == wait_ops[0].results
    assert not wait_ops[1].operands
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_merges_branch_wait_for_dominated_ds_load(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_branch_wait_ds_dependency(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %cond: i1) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    scf.if %cond {
      %copy = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
      %group = ttg.async_commit_group tokens %copy
      %wait = ttg.async_wait %group {num = 0 : i32}
      scf.yield
    } else {
      scf.yield
    }
    %loaded = ttg.local_load %alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<64xf16, #shared, #smem, mutable> -> tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    token_program = converter_tokens.build_token_program(source, converted)
    source_if_op = next(op for op in source.ops if op.name == "scf.if")
    source_wait_op = next(op for op in source.ops if op.name == "ttg.async_wait")
    (carry, ) = token_program.if_token_carries_by_op[source_if_op.index]
    assert carry.then_source_value_id == source_wait_op.results[0]
    assert carry.else_source_value_id is None

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (if_op, ) = [op for op in output.target_program.ops if op.kind == "if"]
    (load_op, ) = [op for op in output.target_program.ops if op.kind == "local_load"]
    assert len(if_op.results) == 1
    assert load_op.operands[1:] == if_op.results
    then_region, else_region = (output.target_program.regions[region_id] for region_id in if_op.region_ids)
    assert "async_wait" in [output.target_program.ops[op_id].kind for op_id in then_region.op_ids]
    assert [output.target_program.ops[op_id].kind for op_id in else_region.op_ids] == ["token"]
    assert output.target_program.values[else_region.yield_value_ids[0]].event_domain == (
        converter_target_ir.EVENT_DOMAIN_EMPTY)
    load_lines = [line for line in output.emitted_module.text.splitlines() if "wave.gather" in line]
    assert load_lines and all(" after " in line for line in load_lines)
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_pipeline_lowers_glu_b_swizzle_transpose_load(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_glu_b_swizzle_transpose_load() attributes {noinline = false} {
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared_b, #smem, mutable>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<64x128xf16, #shared_b, #smem, mutable> -> tensor<64x128xf16, #dot1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (attrs, ) = _packetized_local_load_attrs(output)
    assert attrs["component_count"] == 8
    assert attrs["packet_count"] == 8
    assert attrs["packet_width"] == 8
    assert len(attrs["packet_coordinate_bases"]) == attrs["packet_count"]
    assert attrs["coordinate_shape"] == (64, 128)
    assert attrs["memdesc_shape"] == (64, 128)
    assert attrs["shared_physical_offset_plan"] == "swizzled_xor"
    assert attrs["shared_physical_swizzled_vec"] == 8
    assert attrs["shared_physical_swizzled_per_phase"] == 1
    assert attrs["shared_physical_swizzled_max_phase"] == 16
    assert "load_mode" not in attrs
    assert "mma_access_lane_layout" not in attrs
    assert "component_tile_offsets" not in attrs
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == attrs["packet_count"]
    assert "waveamd.transpose_load" not in wave
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.ds_read_tr_b64_b16") == 16
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_pipeline_keeps_mma_payload_read_tokens_out_of_barriers(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared_a = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_mma_payload_read_tokens_not_barriered() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared_b, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable> -> tensor<128x64xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<64x128xf16, #shared_b, #smem, mutable> -> tensor<64x128xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<128x64xf16, #dot0> * tensor<64x128xf16, #dot1> -> tensor<128x128xf32, #mma>
    %wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    read_tokens = tuple(
        match.group("token") for match in re.finditer(
            r"%[\w#]+,\s*(?P<token>%[\w#]+)\s*=\s*wave\.(?:gather|load)",
            wave,
        ))
    assert len(read_tokens) == sum(attrs["packet_count"] for attrs in _packetized_local_load_attrs(output))
    token_consumers = [
        line for line in wave.splitlines() if "wave.barrier" in line if any(
            re.search(rf"(?<![\w#]){re.escape(token)}(?![\w#])", line) for token in read_tokens)
    ]
    assert token_consumers == []
    assert 'waveamd.mma "mfma.f32.16x16x32.f16"' in wave
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_pipeline_uses_compiler_barrier_for_async_refill(tmp_path, ):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [32, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [1, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared_a = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0], [8, 0], [16, 0], [1, 0], [32, 0]], block = []}>
#shared_b = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [4, 0], [16, 0], [1, 0], [2, 0], [8, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_mma_payload_async_refill_barrier(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %row_limit: i32,
      %stride: i32,
      %limit: i32 {tt.divisibility = 2 : i32}) attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #shared_a, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #shared_b, #smem, mutable>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi32, #linear>
    %row_limit_splat = tt.splat %row_limit : i32 -> tensor<64x1xi32, #linear>
    %row_mod = arith.remsi %row, %row_limit_splat : tensor<64x1xi32, #linear>
    %stride_splat = tt.splat %stride : i32 -> tensor<64x1xi32, #linear>
    %row_scaled = arith.muli %row_mod, %stride_splat : tensor<64x1xi32, #linear>
    %row_b = tt.broadcast %row_scaled : tensor<64x1xi32, #linear> -> tensor<64x32xi32, #linear>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x32xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x32xi32, #linear> -> tensor<64x32xi32, #linear>
    %offset = arith.addi %row_b, %col_b : tensor<64x32xi32, #linear>
    %limit_splat = tt.splat %limit : i32 -> tensor<1x32xi32, #linear>
    %mask = arith.cmpi slt, %col, %limit_splat : tensor<1x32xi32, #linear>
    %mask_b = tt.broadcast %mask : tensor<1x32xi1, #linear> -> tensor<64x32xi1, #linear>
    %warmup = amdg.buffer_load_to_local %arg0[%offset] mask = %mask_b stride = %stride into %a_alloc {contiguity = 2 : i32} : <f16>[tensor<64x32xi32, #linear>] -> <64x32xf16, #shared_a, #smem, mutable>
    %group = ttg.async_commit_group tokens %warmup
    %wait = ttg.async_wait %group {num = 0 : i32}
    %lhs = ttg.local_load %a_alloc {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<64x32xf16, #shared_a, #smem, mutable> -> tensor<64x32xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x128xf16, #shared_b, #smem, mutable> -> tensor<32x128xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<64x32xf16, #dot0> * tensor<32x128xf16, #dot1> -> tensor<64x128xf32, #mma>
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "mfma", triton.warp_pipeline.priority = 0 : i32}
    ttg.barrier all
    %refill = amdg.buffer_load_to_local %arg0[%offset] mask = %mask_b stride = %stride into %a_alloc {contiguity = 2 : i32} : <f16>[tensor<64x32xi32, #linear>] -> <64x32xf16, #shared_a, #smem, mutable>
    %refill_group = ttg.async_commit_group tokens %refill
    %final_wait = ttg.async_wait %refill_group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    raw_wave = output.emitted_module.text
    dma_ops = [op for op in output.target_program.ops if op.kind == "buffer_load_to_local"]
    assert [converter_target_ir.attrs_dict(op)["mode"] for op in dma_ops] == ["symbolic_copy", "symbolic_copy"]
    assert [converter_target_ir.attrs_dict(op)["issue_dependency_count"] for op in dma_ops] == [0, 0]
    assert [converter_target_ir.attrs_dict(op)["source_issue_dependency_count"] for op in dma_ops] == [0, 0]
    assert all("lds_release_dependency_count" not in converter_target_ir.attrs_dict(op) for op in dma_ops)
    assert all(op.kind != "lds_release" for op in output.target_program.ops)
    assert [converter_target_ir.attrs_dict(op)["mask_mode"] for op in dma_ops] == [
        "exec_where",
        "exec_where",
    ]
    assert [(converter_target_ir.attrs_dict(op)["border"], converter_target_ir.attrs_dict(op)["mask"])
            for op in output.target_program.ops
            if op.kind == "sched_barrier"] == [("mfma", 0)]
    assert raw_wave.count("wave.where") == 2
    assert sum("wave.gather" in line and "#waveamd.buffer" in line for line in raw_wave.splitlines()) == 2
    assert raw_wave.count("wave.scatter") == 2
    wave = _run_wave_lower_symbolic_memory(raw_wave)
    lines = wave.splitlines()
    assert "zero_fill_inactive" in wave
    mma_index = next(index for index, line in enumerate(lines) if "waveamd.mma" in line)
    refill_index = max(index for index, line in enumerate(lines) if "waveamd.dma_load_lds" in line)
    barrier_indices = [
        index for index, line in enumerate(lines[mma_index + 1:refill_index], mma_index + 1) if "wave.barrier" in line
    ]
    sched_barrier_indices = [index for index, line in enumerate(lines) if "wave.sched_barrier" in line]
    assert len(barrier_indices) == 1
    assert len(sched_barrier_indices) == 1
    assert sched_barrier_indices[0] < barrier_indices[0] < refill_index
    barrier_lines = [lines[index] for index in barrier_indices]
    assert len(barrier_lines) == 1
    release_token = _ssa_result_name(barrier_lines[0])
    assert f"after {release_token}" not in lines[refill_index]
    assert converter_target_ir.attrs_dict(dma_ops[-1])["barrier_order_dependency_count"] == 1
    order_token_id = dma_ops[-1].operands[-1]
    assert output.target_program.values[order_token_id].event_domain == (converter_target_ir.EVENT_DOMAIN_BARRIER_ISSUE)
    order_projection = _target_value_producer(
        output.target_program,
        order_token_id,
        kind="issue_token",
    )
    full_barrier = _target_value_producer(
        output.target_program,
        order_projection.operands[0],
        kind="barrier",
    )
    assert order_projection.operands == full_barrier.results
    assert converter_target_ir.attrs_dict(full_barrier)["address_space"] == 31
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_pipeline_preserves_mma_packet_vectors_across_for(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared_a = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_f16_mfma_payload_vector_for_carry() attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared_b, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable> -> tensor<128x64xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<64x128xf16, #shared_b, #smem, mutable> -> tensor<64x128xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %rhs_carried, %acc_carried = scf.for %i = %c0 to %c2 step %c1 iter_args(%rhs_iter = %rhs, %acc_iter = %acc) -> (tensor<64x128xf16, #dot1>, tensor<128x128xf32, #mma>) {
      %dot = tt.dot %lhs, %rhs_iter, %acc_iter : tensor<128x64xf16, #dot0> * tensor<64x128xf16, #dot1> -> tensor<128x128xf32, #mma>
      scf.yield %rhs_iter, %dot : tensor<64x128xf16, #dot1>, tensor<128x128xf32, #mma>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    loop_lines = [line for line in wave.splitlines() if "scf.for" in line and "iter_args" in line]
    assert loop_lines
    assert loop_lines[0].count("!wave.simd<vector<8xf16>, 64>") == 8
    assert loop_lines[0].count("!wave.simd<vector<4xf32>, 64>") == 16
    assert "!wave.simd<f16, 64>" not in loop_lines[0]
    assert "!wave.simd<f32, 64>" not in loop_lines[0]
    assert "waveamd.fragment_pack" in wave
    assert all("!waveamd.fragment" not in line
               for line in wave.splitlines()
               if "scf.for" in line or "scf.yield" in line)
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.mfma_f32_16x16x32_f16" in machine
    del ctx


def test_tlx_wave_converter_pipeline_lowers_scaled_mfma_i8_local_loads(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 128], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [0, 0]], block = []}>
#shared_a = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_scaled_mfma_i8_local_loads() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<256x128xi8, #shared_a, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x64xi8, #shared_b, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<256x128xi8, #shared_a, #smem, mutable> -> tensor<256x128xi8, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<128x64xi8, #shared_b, #smem, mutable> -> tensor<128x64xi8, #dot1>
    %a_scale = arith.constant dense<127> : tensor<256x8xi8, #linear>
    %b_scale = arith.constant dense<127> : tensor<64x8xi8, #linear1>
    %acc = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %dot = tt.dot_scaled %lhs scale %a_scale, %rhs scale %b_scale, %acc lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<256x128xi8, #dot0>, tensor<256x8xi8, #linear> * tensor<128x64xi8, #dot1>, tensor<64x8xi8, #linear1> -> tensor<256x64xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    local_load_attrs = _packetized_local_load_attrs(output)
    assert [attrs["component_count"] for attrs in local_load_attrs] == [16, 4]
    assert [attrs["packet_count"] for attrs in local_load_attrs] == [16, 4]
    assert [attrs["packet_width"] for attrs in local_load_attrs] == [16, 16]
    assert [attrs["coordinate_shape"] for attrs in local_load_attrs] == [(256, 128), (128, 64)]
    assert [attrs["memdesc_shape"] for attrs in local_load_attrs] == [(256, 128), (128, 64)]
    assert all(len(attrs["packet_coordinate_bases"]) == attrs["packet_count"] for attrs in local_load_attrs)
    assert all("load_mode" not in attrs for attrs in local_load_attrs)
    assert all("mma_access_lane_layout" not in attrs for attrs in local_load_attrs)
    assert [attrs["shared_physical_offset_plan"] for attrs in local_load_attrs] == [
        "dense_row_major",
        "swizzled_xor",
    ]
    assert [attrs["shared_physical_order"] for attrs in local_load_attrs] == [(1, 0), (0, 1)]
    assert local_load_attrs[0]["packet_coordinate_bases"] == (
        (0, 0),
        (0, 64),
        (32, 0),
        (32, 64),
        (64, 0),
        (64, 64),
        (96, 0),
        (96, 64),
        (128, 0),
        (128, 64),
        (160, 0),
        (160, 64),
        (192, 0),
        (192, 64),
        (224, 0),
        (224, 64),
    )
    assert local_load_attrs[1]["packet_coordinate_bases"] == (
        (0, 0),
        (64, 0),
        (0, 32),
        (64, 32),
    )
    (mma_attrs, ) = [converter_target_ir.attrs_dict(op) for op in output.target_program.ops if op.kind == "mma_scaled"]
    assert (mma_attrs["m_tiles"], mma_attrs["n_tiles"], mma_attrs["k_tiles"]) == (8, 2, 2)
    assert mma_attrs["has_scales"]
    assert mma_attrs["kind"] == "mfma.scale.f32.16x16x128.f4.f4"
    assert (
        mma_attrs["lhs_scale_group_count"],
        mma_attrs["lhs_scale_pack_width"],
        mma_attrs["lhs_scale_k_packed_vals"],
        mma_attrs["lhs_scale_non_k_packed_vals"],
    ) == (4, 4, 2, 2)
    assert (
        mma_attrs["rhs_scale_group_count"],
        mma_attrs["rhs_scale_pack_width"],
        mma_attrs["rhs_scale_k_packed_vals"],
        mma_attrs["rhs_scale_non_k_packed_vals"],
    ) == (1, 4, 2, 2)
    assert [op.kind for op in output.target_program.ops].count("layout_convert") == 0
    del ctx


def test_tlx_wave_converter_pipeline_lowers_scaled_mfma_32x32x64_local_loads(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [32, 32, 64], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
#shared_a = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_scaled_mfma_32x32x64_local_loads() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x128xi8, #shared_a, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x128xi8, #shared_b, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<128x128xi8, #shared_a, #smem, mutable> -> tensor<128x128xi8, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<128x128xi8, #shared_b, #smem, mutable> -> tensor<128x128xi8, #dot1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    local_load_attrs = _packetized_local_load_attrs(output)
    assert [attrs["component_count"] for attrs in local_load_attrs] == [8, 4]
    assert [attrs["packet_count"] for attrs in local_load_attrs] == [8, 4]
    assert [attrs["packet_width"] for attrs in local_load_attrs] == [16, 16]
    assert [attrs["coordinate_shape"] for attrs in local_load_attrs] == [(128, 128), (128, 128)]
    assert all(len(attrs["packet_coordinate_bases"]) == attrs["packet_count"] for attrs in local_load_attrs)
    assert all("load_mode" not in attrs for attrs in local_load_attrs)
    assert all("mma_access_lane_layout" not in attrs for attrs in local_load_attrs)
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert machine.count("waveamdmachine.ds_load_tuple_b32") == 12
    del ctx


def test_tlx_wave_converter_lowers_scaled_mfma_32x32x64():
    mfma_properties = {
        "version": 4,
        "warps_per_cta": (2, 4),
        "instr_shape": (32, 32, 64),
        "is_transposed": True,
    }
    layouts = (
        _fake_layout(
            0,
            0,
            kind="dot_operand",
            shape=(128, 128),
            element_type="i8",
            component_count=8,
            properties={
                "k_width": 16,
                "op_idx": 0,
                "parent_kind": "amd_mfma",
                "parent_properties": mfma_properties,
            },
        ),
        _fake_layout(
            1,
            1,
            kind="dot_operand",
            shape=(128, 128),
            element_type="i8",
            component_count=4,
            properties={
                "k_width": 16,
                "op_idx": 1,
                "parent_kind": "amd_mfma",
                "parent_properties": mfma_properties,
            },
        ),
        _fake_layout(
            2,
            2,
            kind="amd_mfma",
            shape=(128, 128),
            element_type="f32",
            component_count=2,
            properties=mfma_properties,
        ),
        _fake_layout(3, 3, kind="linear", shape=(128, 8), element_type="i8", component_count=8),
        _fake_layout(4, 4, kind="linear", shape=(128, 8), element_type="i8", component_count=4),
    )
    type_layout_program = converter_types.TypeLayoutProgram(
        {
            0:
            _converted_value(
                0,
                representation="simd_packet_tuple",
                element_type="i8",
                component_count=8,
                layout_map_id=0,
            ),
            1:
            _converted_value(
                1,
                representation="simd_packet_tuple",
                element_type="i8",
                component_count=4,
                layout_map_id=1,
            ),
            2:
            _converted_value(
                2,
                representation="simd_packet_tuple",
                element_type="f32",
                component_count=2,
                layout_map_id=2,
            ),
            3:
            _converted_value(
                3,
                representation="simd_tuple",
                element_type="i8",
                component_count=8,
                layout_map_id=3,
            ),
            4:
            _converted_value(
                4,
                representation="simd_tuple",
                element_type="i8",
                component_count=4,
                layout_map_id=4,
            ),
            5:
            _converted_value(
                5,
                representation="simd_packet_tuple",
                element_type="f32",
                component_count=2,
                layout_map_id=2,
            ),
        },
        layouts,
    )
    builder = converter_target_ir.TargetBuilder()
    for value_id in range(5):
        converted = type_layout_program.values[value_id]
        builder.add_value(
            converter_target_ir.target_type_from_converted(converted.type),
            source_value_id=value_id,
        )
    op = converter_source_ir.SourceOp(
        0,
        "tt.dot_scaled",
        operands=(0, 1, 2, 3, 4),
        results=(5, ),
        attrs={"a_elem_type": 4, "b_elem_type": 4},
    )

    converter_op_conversion._convert_dot_scaled(
        builder,
        SimpleNamespace(),
        type_layout_program,
        op,
    )

    mma_attrs = converter_target_ir.attrs_dict(builder.ops[-1])
    assert (mma_attrs["m_tiles"], mma_attrs["n_tiles"], mma_attrs["k_tiles"]) == (2, 1, 4)
    assert mma_attrs["kind"] == "mfma.scale.f32.32x32x64.f4.f4"
    assert mma_attrs["lhs_registers"] == mma_attrs["rhs_registers"] == 4
    assert mma_attrs["acc_registers"] == 16


def test_tlx_wave_converter_pipeline_lowers_scaled_mfma_i8_scale_transpose_loads(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 128], isTransposed = true}>
  #dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
  #dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
  #linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [16, 0]], block = []}>
  #linear1 = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [0, 0]], block = []}>
  #blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
  #shared_a = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
  #shared_b = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
  #shared_scale = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
  #smem = #ttg.shared_memory
  """
    local_func = """
    tt.func public @converter_scaled_mfma_i8_scale_transpose_loads(
        %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
        %stride: i32) attributes {noinline = false} {
      %rows = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
      %stride_splat = tt.splat %stride : i32 -> tensor<256x1xi32, #blocked>
      %row_scaled = arith.muli %row, %stride_splat : tensor<256x1xi32, #blocked>
      %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %row_b = tt.broadcast %row_scaled : tensor<256x1xi32, #blocked> -> tensor<256x128xi32, #blocked>
      %col_b = tt.broadcast %col : tensor<1x128xi32, #blocked> -> tensor<256x128xi32, #blocked>
      %offset = arith.addi %row_b, %col_b : tensor<256x128xi32, #blocked>
      %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<256x128xi8, #shared_a, #smem, mutable>
      %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x128xi8, #shared_b, #smem, mutable>
      %as_alloc = ttg.local_alloc : () -> !ttg.memdesc<256x8xi8, #shared_scale, #smem, mutable>
      %bs_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x8xi8, #shared_scale, #smem, mutable>
      %lhs = ttg.local_load %a_alloc : !ttg.memdesc<256x128xi8, #shared_a, #smem, mutable> -> tensor<256x128xi8, #dot0>
      %rhs = ttg.local_load %b_alloc : !ttg.memdesc<128x128xi8, #shared_b, #smem, mutable> -> tensor<128x128xi8, #dot1>
      %a_scale_store = arith.constant dense<127> : tensor<256x8xi8, #linear>
      %b_scale_store = arith.constant dense<127> : tensor<128x8xi8, #linear1>
      ttg.local_store %a_scale_store, %as_alloc : tensor<256x8xi8, #linear> -> !ttg.memdesc<256x8xi8, #shared_scale, #smem, mutable>
      ttg.local_store %b_scale_store, %bs_alloc : tensor<128x8xi8, #linear1> -> !ttg.memdesc<128x8xi8, #shared_scale, #smem, mutable>
      %a_scale = ttg.local_load %as_alloc : !ttg.memdesc<256x8xi8, #shared_scale, #smem, mutable> -> tensor<256x8xi8, #linear>
      %b_scale = ttg.local_load %bs_alloc : !ttg.memdesc<128x8xi8, #shared_scale, #smem, mutable> -> tensor<128x8xi8, #linear1>
      %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
      %dot = tt.dot_scaled %lhs scale %a_scale, %rhs scale %b_scale, %acc lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<256x128xi8, #dot0>, tensor<256x8xi8, #linear> * tensor<128x128xi8, #dot1>, tensor<128x8xi8, #linear1> -> tensor<256x128xf32, #mma>
      %c = arith.truncf %dot : tensor<256x128xf32, #mma> to tensor<256x128xf16, #mma>
      %converted = ttg.convert_layout %c : tensor<256x128xf16, #mma> -> tensor<256x128xf16, #blocked>
      amdg.buffer_store %converted, %arg0[%offset] {contiguity = 1 : i32} : tensor<256x128xf16, #blocked>
      tt.return
    }
  """
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    scale_load_attrs = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "local_load" and converter_target_ir.attrs_dict(op)["element_type"] == "i8"
        and int(converter_target_ir.attrs_dict(op)["packet_count"]) == 1
    ]
    assert [attrs["component_count"] for attrs in scale_load_attrs] == [16, 8]
    assert [attrs["packet_count"] for attrs in scale_load_attrs] == [1, 1]
    assert [attrs["packet_width"] for attrs in scale_load_attrs] == [16, 8]
    assert all(attrs["shared_physical_offset_plan"] == "swizzled_xor" for attrs in scale_load_attrs)
    assert all(len(attrs["packet_coordinate_bases"]) == 1 for attrs in scale_load_attrs)
    assert all("result_value_mode" not in attrs for attrs in scale_load_attrs)
    assert all("result_packet_width" not in attrs for attrs in scale_load_attrs)
    assert all("semantic_role" not in attrs for attrs in scale_load_attrs)
    wave = output.emitted_module.text
    transpose_gathers = [line for line in wave.splitlines() if "wave.gather" in line and "vector<8xi8>" in line]
    assert len(transpose_gathers) == 1
    assert "waveamd.transpose_load" not in wave
    mma_scale_lines = [line for line in wave.splitlines() if "waveamd.mma_scale" in line]
    assert mma_scale_lines
    assert all("!wave.simd<vector<4xi8>, 64>" in line for line in mma_scale_lines)
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.ds_read_tr_b64_b8") == 3
    del ctx


def test_tlx_wave_converter_pipeline_carries_scaled_mfma_i8_scale_packets_across_for(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 128], isTransposed = true}>
  #dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
  #dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
  #linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [16, 0]], block = []}>
  #linear1 = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [0, 0]], block = []}>
  #shared_a = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
  #shared_b = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
  #shared_scale = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
  #smem = #ttg.shared_memory
  """
    local_func = """
    tt.func public @converter_scaled_mfma_i8_scale_packet_loop() attributes {noinline = false} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<256x128xi8, #shared_a, #smem, mutable>
      %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x128xi8, #shared_b, #smem, mutable>
      %as_alloc = ttg.local_alloc : () -> !ttg.memdesc<256x8xi8, #shared_scale, #smem, mutable>
      %bs_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x8xi8, #shared_scale, #smem, mutable>
      %lhs = ttg.local_load %a_alloc : !ttg.memdesc<256x128xi8, #shared_a, #smem, mutable> -> tensor<256x128xi8, #dot0>
      %rhs = ttg.local_load %b_alloc : !ttg.memdesc<128x128xi8, #shared_b, #smem, mutable> -> tensor<128x128xi8, #dot1>
      %a_scale_store = arith.constant dense<127> : tensor<256x8xi8, #linear>
      %b_scale_store = arith.constant dense<127> : tensor<128x8xi8, #linear1>
      ttg.local_store %a_scale_store, %as_alloc : tensor<256x8xi8, #linear> -> !ttg.memdesc<256x8xi8, #shared_scale, #smem, mutable>
      ttg.local_store %b_scale_store, %bs_alloc : tensor<128x8xi8, #linear1> -> !ttg.memdesc<128x8xi8, #shared_scale, #smem, mutable>
      %a_scale = ttg.local_load %as_alloc : !ttg.memdesc<256x8xi8, #shared_scale, #smem, mutable> -> tensor<256x8xi8, #linear>
      %b_scale = ttg.local_load %bs_alloc : !ttg.memdesc<128x8xi8, #shared_scale, #smem, mutable> -> tensor<128x8xi8, #linear1>
      %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
      %a_carried, %b_carried, %acc_carried = scf.for %i = %c0 to %c2 step %c1 iter_args(%a_iter = %a_scale, %b_iter = %b_scale, %acc_iter = %acc) -> (tensor<256x8xi8, #linear>, tensor<128x8xi8, #linear1>, tensor<256x128xf32, #mma>) {
        %dot = tt.dot_scaled %lhs scale %a_iter, %rhs scale %b_iter, %acc_iter lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<256x128xi8, #dot0>, tensor<256x8xi8, #linear> * tensor<128x128xi8, #dot1>, tensor<128x8xi8, #linear1> -> tensor<256x128xf32, #mma>
        scf.yield %a_iter, %b_iter, %dot : tensor<256x8xi8, #linear>, tensor<128x8xi8, #linear1>, tensor<256x128xf32, #mma>
      }
      tt.return
    }
  """
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    loop_lines = [line for line in wave.splitlines() if "scf.for" in line and "iter_args" in line]
    assert loop_lines
    assert "!wave.simd<i8, 64>" in loop_lines[0]
    mma_scale_lines = [line for line in wave.splitlines() if "waveamd.mma_scale" in line]
    assert mma_scale_lines
    assert all("!wave.simd<vector<4xi8>, 64>" in line for line in mma_scale_lines)
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_pipeline_lowers_transposed_scaled_mfma_i8_load(tmp_path):
    preamble = """
  #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 128], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [0, 0]], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared_a = #ttg.padded_shared<[1024:+32] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]], block = []}>
#shared_b = #ttg.padded_shared<[1024:+32] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0]], block = []}>
#shared_b_t = #ttg.padded_shared<[1024:+32] {offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8]], block = []}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_transposed_scaled_mfma_i8_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %rows = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<256x1xi32, #blocked>
    %row_scaled = arith.muli %row, %stride_splat : tensor<256x1xi32, #blocked>
    %col = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %row_b = tt.broadcast %row_scaled : tensor<256x1xi32, #blocked> -> tensor<256x128xi32, #blocked>
    %col_b = tt.broadcast %col : tensor<1x128xi32, #blocked> -> tensor<256x128xi32, #blocked>
    %offset = arith.addi %row_b, %col_b : tensor<256x128xi32, #blocked>
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<256x128xi8, #shared_a, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<128x128xi8, #shared_b, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<256x128xi8, #shared_a, #smem, mutable> -> tensor<256x128xi8, #dot0>
    %b_t = ttg.memdesc_trans %b_alloc {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xi8, #shared_b, #smem, mutable> -> !ttg.memdesc<128x128xi8, #shared_b_t, #smem, mutable>
    %rhs = ttg.local_load %b_t : !ttg.memdesc<128x128xi8, #shared_b_t, #smem, mutable> -> tensor<128x128xi8, #dot1>
    %a_scale = arith.constant dense<127> : tensor<256x8xi8, #linear>
    %b_scale = arith.constant dense<127> : tensor<128x8xi8, #linear1>
    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %dot = tt.dot_scaled %lhs scale %a_scale, %rhs scale %b_scale, %acc lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<256x128xi8, #dot0>, tensor<256x8xi8, #linear> * tensor<128x128xi8, #dot1>, tensor<128x8xi8, #linear1> -> tensor<256x128xf32, #mma>
    %c = arith.truncf %dot : tensor<256x128xf32, #mma> to tensor<256x128xf16, #mma>
    %converted = ttg.convert_layout %c : tensor<256x128xf16, #mma> -> tensor<256x128xf16, #blocked>
    amdg.buffer_store %converted, %arg0[%offset] {contiguity = 1 : i32} : tensor<256x128xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    local_load_attrs = _packetized_local_load_attrs(output)
    assert [attrs["packet_count"] for attrs in local_load_attrs] == [16, 8]
    assert [attrs["packet_width"] for attrs in local_load_attrs] == [16, 16]
    assert local_load_attrs[1]["element_type"] == "i8"
    assert local_load_attrs[1]["shared_physical_offset_plan"] == "padded_linear"
    assert all("load_mode" not in attrs for attrs in local_load_attrs)
    assert all("mma_access_lane_layout" not in attrs for attrs in local_load_attrs)
    wave = output.emitted_module.text
    assert "waveamd.transpose_load" not in wave
    assert "vector<16xi8>" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.ds_read_tr_b64_b8" not in machine
    del ctx


def test_tlx_wave_converter_packs_blocked_dot_operand_parent_layout(tmp_path):
    preamble = """
#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
"""
    local_func = """
  tt.func public @converter_blocked_dot_operand_parent_layout() attributes {noinline = false} {
    %a = arith.constant dense<0.000000e+00> : tensor<256x64xf16, #blocked_a>
    %b = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked_b>
    %a_dot = ttg.convert_layout %a : tensor<256x64xf16, #blocked_a> -> tensor<256x64xf16, #dot0>
    %b_dot = ttg.convert_layout %b : tensor<64x256xf16, #blocked_b> -> tensor<64x256xf16, #dot1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    layout_converts = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    assert len(layout_converts) == 2
    for convert_op in layout_converts:
        _assert_mechanical_layout_transform(output.target_program, convert_op)
    wave = output.emitted_module.text
    assert wave.count("wave.redistribute") == 2
    assert "wave.alloc" not in wave
    assert "waveamd.transpose_load" not in wave
    assert "waveamd.fragment_pack" not in wave
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_preserves_typed_f16_packets_into_redistribute(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_m = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked_k = #ttg.slice<{dim = 0, parent = #blocked}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
"""
    local_func = """
  tt.func public @converter_raw_f16_packets_into_dot_operand(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_m>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked_k>
    %rows_e = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #blocked_m> -> tensor<128x1xi32, #blocked>
    %cols_e = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #blocked_k> -> tensor<1x64xi32, #blocked>
    %stride = arith.constant 64 : i32
    %stride_splat = tt.splat %stride : i32 -> tensor<128x1xi32, #blocked>
    %row_offsets = arith.muli %rows_e, %stride_splat : tensor<128x1xi32, #blocked>
    %row_offsets_b = tt.broadcast %row_offsets : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %col_offsets_b = tt.broadcast %cols_e : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %offsets = arith.addi %row_offsets_b, %col_offsets_b : tensor<128x64xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%offsets] {contiguity = 8 : i32} : tensor<128x64xf16, #blocked>
    %dot = ttg.convert_layout %loaded : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dot0>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (load_op, ) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    load_attrs = converter_target_ir.attrs_dict(load_op)
    assert "result_value_mode" not in load_attrs
    assert "result_packet_width" not in load_attrs
    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    wave = output.emitted_module.text
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    _assert_layout_transforms_reach_wave(output)
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_tuple_b32" in machine
    assert "waveamdmachine.ds_store_tuple_b32" in machine
    assert "waveamdmachine.ds_load_tuple_b32" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    assert "waveamdmachine.ds_store_b16" not in machine
    assert "waveamdmachine.ds_load_b16" not in machine
    del ctx


def test_tlx_wave_converter_lowers_chunked_blocked_dot_operand_pack(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
"""
    local_func = """
  tt.func public @converter_chunked_dot_operand_parent_layout() attributes {noinline = false} {
    %a = arith.constant dense<0.000000e+00> : tensor<256x64xf16, #blocked>
    %a_dot = ttg.convert_layout %a : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #dot0>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    wave = output.emitted_module.text
    assert wave.count("wave.redistribute") == 1
    assert "vector<8xf16>" in wave
    _run_wave_verify(wave)
    del ctx


@pytest.mark.parametrize(
    "instr_shape,k_width,tensor_shape,source_registers",
    [
        ((32, 32, 16), 4, (256, 64), 16),
        ((16, 16, 32), 8, (128, 64), 4),
    ],
    ids=["mfma32", "mfma16"],
)
def test_tlx_wave_converter_lowers_mfma_fragment_to_dot_operand(
    tmp_path,
    instr_shape,
    k_width,
    tensor_shape,
    source_registers,
):
    preamble = f"""
#mma = #ttg.amd_mfma<{{version = 4, warpsPerCTA = [4, 1], instrShape = [{instr_shape[0]}, {instr_shape[1]}, {instr_shape[2]}], isTransposed = true}}>
#dot0 = #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {k_width}}}>
"""
    local_func = f"""
  tt.func public @converter_mfma_fragment_to_dot_operand() attributes {{noinline = false}} {{
    %acc = arith.constant dense<1.250000e+00> : tensor<{tensor_shape[0]}x{tensor_shape[1]}xf32, #mma>
    %truncated = arith.truncf %acc : tensor<{tensor_shape[0]}x{tensor_shape[1]}xf32, #mma> to tensor<{tensor_shape[0]}x{tensor_shape[1]}xbf16, #mma>
    %dot = ttg.convert_layout %truncated : tensor<{tensor_shape[0]}x{tensor_shape[1]}xbf16, #mma> -> tensor<{tensor_shape[0]}x{tensor_shape[1]}xbf16, #dot0>
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = _convert_ttgir_to_wave_keep_dead(mod)

    (convert_op, ) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    _assert_mechanical_layout_transform(output.target_program, convert_op)
    wave = output.emitted_module.text
    assert f"vector<{source_registers}xbf16>" in wave
    assert "vector<8xbf16>" in wave
    _assert_layout_transforms_reach_wave(output)
    assert "wave.alloc" not in wave
    assert "wave.barrier" not in wave
    assert "wave.store" not in wave
    assert "wave.load" not in wave
    lowered = _run_wave_lower_redistribute(wave)
    if instr_shape == (32, 32, 16):
        assert "wave.shuffle" not in lowered
    else:
        assert "wave.shuffle" in lowered
    assert "wave.alloc" not in lowered
    assert "wave.barrier" not in lowered
    assert "wave.store" not in lowered
    assert "wave.load" not in lowered
    del ctx


def test_tlx_wave_converter_pipeline_lowers_mfma32_transpose_load(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_mfma32_transpose_load() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    local_load_attrs = _packetized_local_load_attrs(output)
    assert [attrs["packet_count"]
            for attrs in local_load_attrs] == [attrs["component_count"] for attrs in local_load_attrs]
    assert all(attrs["packet_width"] == 8 for attrs in local_load_attrs)
    assert [attrs["shared_physical_offset_plan"] for attrs in local_load_attrs] == [
        "swizzled_xor",
        "swizzled_xor",
    ]
    assert all(attrs["shared_physical_swizzled_vec"] == 8 for attrs in local_load_attrs)
    assert [attrs["slot_coordinate_coefficients"] for attrs in local_load_attrs] == [
        ((0, 1), (0, 2), (0, 8)),
        ((1, 0), (2, 0), (8, 0)),
    ]
    assert all("shared_layout_kind" not in attrs for attrs in local_load_attrs)
    assert [op.kind for op in output.target_program.ops].count("layout_convert") == 0
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 4
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.ds_load_tuple_b32") == 4
    assert machine.count("waveamdmachine.ds_read_tr_b64_b16") == 4
    assert "waveamdmachine.ds_bpermute_b32" not in machine
    assert "waveamdmachine.v_permlane32_b32" not in machine
    del ctx


@pytest.mark.parametrize(
    "element_type,per_phase,max_phase",
    (("f16", 4, 4), ("bf16", 2, 8)),
)
def test_tlx_wave_converter_uses_physical_minor_k_order_for_swizzled_mfma_b_load(
    tmp_path,
    element_type,
    per_phase,
    max_phase,
):
    preamble = f"""
#mma = #ttg.amd_mfma<{{version = 4, warpsPerCTA = [8, 1], instrShape = [32, 32, 16], isTransposed = true}}>
#dot1 = #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = 8}}>
#shared = #ttg.swizzled_shared<{{vec = 8, perPhase = {per_phase}, maxPhase = {max_phase}, order = [1, 0]}}>
#shared_t = #ttg.swizzled_shared<{{vec = 8, perPhase = {per_phase}, maxPhase = {max_phase}, order = [0, 1]}}>
#smem = #ttg.shared_memory
"""
    local_func = f"""
  tt.func public @converter_swizzled_mfma_b_physical_order() attributes {{noinline = false}} {{
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x128x{element_type}, #shared, #smem, mutable>
    %transposed = ttg.memdesc_trans %alloc {{order = array<i32: 1, 0>}} : !ttg.memdesc<32x128x{element_type}, #shared, #smem, mutable> -> !ttg.memdesc<128x32x{element_type}, #shared_t, #smem, mutable>
    %rhs = ttg.local_load %transposed : !ttg.memdesc<128x32x{element_type}, #shared_t, #smem, mutable> -> tensor<128x32x{element_type}, #dot1>
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (attrs, ) = _packetized_local_load_attrs(output)
    assert attrs["shared_physical_order"] == (0, 1)
    assert attrs["packet_width"] == 8
    assert len(attrs["packet_coordinate_bases"]) == attrs["packet_count"]
    assert "load_mode" not in attrs
    assert "mma_access_lane_layout" not in attrs
    assert f"vector<8x{element_type}>" in output.emitted_module.text
    _run_waveamd_to_machine(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_records_b16_transpose_chunk_deltas(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#shared = #ttg.padded_shared<[4:+16] {order = [1, 0], shape = [64, 128]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_b16_transpose_chunk_deltas() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %rhs = ttg.local_load %alloc : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #dot1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    local_load_attrs = _packetized_local_load_attrs(output)

    assert len(local_load_attrs) == 1
    assert local_load_attrs[0]["packet_count"] == local_load_attrs[0]["component_count"]
    assert local_load_attrs[0]["packet_width"] == 8
    assert "load_mode" not in local_load_attrs[0]
    assert local_load_attrs[0]["shared_physical_offset_plan"] == "padded_linear"
    assert local_load_attrs[0]["shared_physical_intervals"] == (4, )
    assert local_load_attrs[0]["shared_physical_paddings"] == (16, )
    assert local_load_attrs[0]["shared_physical_linear_component_bases"] == (
        (0, 1),
        (0, 2),
        (0, 4),
        (0, 8),
        (0, 16),
        (0, 32),
        (0, 64),
        (1, 0),
        (2, 0),
        (4, 0),
        (8, 0),
        (16, 0),
        (32, 0),
    )
    assert "shared_layout_kind" not in local_load_attrs[0]
    assert local_load_attrs[0]["slot_coordinate_coefficients"] == (
        (1, 0),
        (2, 0),
        (8, 0),
    )
    wave = output.emitted_module.text
    assert wave.count("wave.gather") == 8
    assert "waveamd.transpose_load" not in wave
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.ds_read_tr_b64_b16") == 16
    assert "waveamdmachine.ds_load" not in machine
    assert "waveamdmachine.ds_bpermute_b32" not in machine
    del ctx


def test_tlx_wave_converter_pipeline_lowers_same_representation_expand_dims(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
"""
    local_func = """
  tt.func public @converter_expand_dims() attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32, #slice> -> tensor<1x64xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "make_range",
        "expand_dims",
        "return",
    ]
    (expand_op, ) = [op for op in output.target_program.ops if op.kind == "expand_dims"]
    _assert_mechanical_layout_transform(
        output.target_program,
        expand_op,
        axis=0,
    )
    _assert_layout_transforms_reach_wave(output)
    assert "tt.expand_dims" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_pointer_splat_expand_dims(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
"""
    local_func = """
  tt.func public @converter_pointer_expand_dims(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #slice>
    %expanded = tt.expand_dims %base {axis = 0 : i32} : tensor<64x!tt.ptr<f32>, #slice> -> tensor<1x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "splat",
        "expand_dims",
        "return",
    ]
    (expand_op, ) = [op for op in output.target_program.ops if op.kind == "expand_dims"]
    _assert_mechanical_layout_transform(
        output.target_program,
        expand_op,
        axis=0,
    )
    _assert_layout_transforms_reach_wave(output)
    assert "tt.expand_dims" not in output.emitted_module.text
    assert "wave.splat" in output.emitted_module.text
    assert "!wave.simd<!wave.ptr<#wave.global" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_blocked_broadcast(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
"""
    local_func = """
  tt.func public @converter_broadcast() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice>
    %expanded = tt.expand_dims %range {axis = 1 : i32} : tensor<128xi32, #slice> -> tensor<128x1xi32, #blocked>
    %broadcast = tt.broadcast %expanded : tensor<128x1xi32, #blocked> -> tensor<128x2xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=2, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "make_range",
        "expand_dims",
        "broadcast",
        "return",
    ]
    expand_op = next(op for op in output.target_program.ops if op.kind == "expand_dims")
    broadcast_op = next(op for op in output.target_program.ops if op.kind == "broadcast")
    _assert_mechanical_layout_transform(
        output.target_program,
        expand_op,
        axis=1,
    )
    _assert_mechanical_layout_transform(output.target_program, broadcast_op)
    _assert_layout_transforms_reach_wave(output)
    assert "tt.broadcast" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_blocked_column_broadcast_components(tmp_path, ):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
"""
    local_func = """
  tt.func public @converter_column_broadcast() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<128xi32, #slice> -> tensor<1x128xi32, #blocked>
    %broadcast = tt.broadcast %expanded : tensor<1x128xi32, #blocked> -> tensor<256x128xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (expand_op, ) = [op for op in output.target_program.ops if op.kind == "expand_dims"]
    (broadcast_op, ) = [op for op in output.target_program.ops if op.kind == "broadcast"]
    _assert_mechanical_layout_transform(
        output.target_program,
        expand_op,
        axis=0,
    )
    _assert_mechanical_layout_transform(output.target_program, broadcast_op)
    _assert_layout_transforms_reach_wave(output)
    assert "tt.broadcast" not in output.emitted_module.text
    del ctx


@triton.jit
def _tlx_wave_stage_only_kernel():
    pid = tl.program_id(0)
    tl.assume(pid >= 0)


@triton.jit
def _tlx_wave_add_one_kernel(x, y, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(x + offs, mask=mask, other=0.0)
    tl.store(y + offs, vals + 1.0, mask=mask)


def _minimal_ttgir(
    public_funcs,
    target="hip:gfx950",
    threads_per_warp=64,
    num_ctas=1,
    num_warps=4,
    preamble="",
):
    return f"""
{preamble}
module attributes {{tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = {num_ctas} : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "{target}", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
{public_funcs}
}}
"""


def _tlx_wave_options(
    arch="gfx950",
    warp_size=64,
    tlx_wave_enable_split_barriers=False,
    tlx_wave_enable_multi_wave_specialize=False,
    waves_per_eu=0,
):
    return SimpleNamespace(
        arch=arch,
        warp_size=warp_size,
        tlx_wave_enable_split_barriers=tlx_wave_enable_split_barriers,
        tlx_wave_enable_multi_wave_specialize=tlx_wave_enable_multi_wave_specialize,
        waves_per_eu=waves_per_eu,
    )


def _convert_ttgir_to_wave_keep_dead(
    mod,
    *,
    kernel_name=None,
    verify=True,
):
    source_program = converter_source_import.import_source_program(mod, kernel_name=kernel_name)
    type_layout_program = converter_types.convert_source_program(source_program)
    fact_program = converter_facts.analyze_facts(source_program, type_layout_program)
    token_program = converter_tokens.build_token_program(source_program, type_layout_program)
    target_program = converter_op_conversion.convert_ops(
        source_program,
        type_layout_program,
        fact_program,
        token_program,
    )
    target_program = converter_barrier_order.thread_barrier_issue_order(target_program)
    if verify:
        converter_verifier.verify_target_program(
            target_program,
            source_program=source_program,
            token_program=token_program,
        )
    emitted_module = converter_emission.emit_wave_module(target_program)
    return converter_pipeline.ConversionOutput(
        source_program,
        type_layout_program,
        fact_program,
        token_program,
        target_program,
        emitted_module,
    )


def test_tlx_wave_backend_defaults_and_accepts_mfma_options(monkeypatch):
    backend = make_backend(GFX950_WAVE)
    monkeypatch.delenv("TRITON_TLX_WAVE_ENABLE_SPLIT_BARRIERS", raising=False)
    monkeypatch.delenv("TRITON_TLX_WAVE_ENABLE_MULTI_WAVE_SPECIALIZE", raising=False)

    default_options = backend.parse_options({})
    assert default_options.matrix_instr_nonkdim == 0
    assert not hasattr(backend.parse_options({}), "tlx_wave_schedule_max_region_ops")
    assert default_options.tlx_wave_enable_split_barriers is False
    assert default_options.tlx_wave_enable_multi_wave_specialize is False
    assert backend.parse_options({"matrix_instr_nonkdim": 32}).matrix_instr_nonkdim == 32
    assert backend.parse_options({"tlx_wave_enable_split_barriers": True}).tlx_wave_enable_split_barriers is True
    assert backend.parse_options({"tlx_wave_enable_split_barriers": "off"}).tlx_wave_enable_split_barriers is False
    monkeypatch.setenv("TRITON_TLX_WAVE_ENABLE_SPLIT_BARRIERS", "1")
    assert backend.parse_options({}).tlx_wave_enable_split_barriers is True
    assert backend.parse_options({"tlx_wave_enable_split_barriers": False}).tlx_wave_enable_split_barriers is False
    with pytest.raises(ValueError, match="tlx_wave_enable_split_barriers"):
        backend.parse_options({"tlx_wave_enable_split_barriers": "maybe"})
    enabled_multi_wave = backend.parse_options({"tlx_wave_enable_multi_wave_specialize": True})
    disabled_multi_wave = backend.parse_options({"tlx_wave_enable_multi_wave_specialize": "off"})
    assert enabled_multi_wave.tlx_wave_enable_multi_wave_specialize is True
    assert disabled_multi_wave.tlx_wave_enable_multi_wave_specialize is False
    assert enabled_multi_wave.hash() != default_options.hash()
    monkeypatch.setenv("TRITON_TLX_WAVE_ENABLE_MULTI_WAVE_SPECIALIZE", "1")
    assert backend.parse_options({}).tlx_wave_enable_multi_wave_specialize is True
    explicit_disabled_multi_wave = backend.parse_options({"tlx_wave_enable_multi_wave_specialize": False})
    assert explicit_disabled_multi_wave.tlx_wave_enable_multi_wave_specialize is False
    with pytest.raises(ValueError, match="tlx_wave_enable_multi_wave_specialize"):
        backend.parse_options({"tlx_wave_enable_multi_wave_specialize": "maybe"})
    with pytest.warns(UserWarning, match="kpack is deprecated"):
        assert backend.parse_options({"kpack": 2}).kpack == 1
    assert make_backend(GFX942_WAVE).parse_options({}).matrix_instr_nonkdim == 0
    assert make_backend(GFX942_WAVE).parse_options({"kpack": 2}).kpack == 2


def test_tlx_wave_uses_standard_wave_pipeline():
    wave_opt = wave_bridge_tools._wave_opt()
    pipeline = wave_bridge_tools._wave_hsaco_pipeline_args(
        wave_opt,
        "gfx950",
    )[0]

    assert "entry-point=compile_kernels" in pipeline
    assert "waveamd_backend_multi_wave" not in pipeline


def _parse_ttgir(
    tmp_path,
    public_funcs,
    target="hip:gfx950",
    threads_per_warp=64,
    num_ctas=1,
    num_warps=4,
    preamble="",
):
    ctx = ir.context()
    ir.load_dialects(ctx)
    make_backend(GFX950_WAVE).load_dialects(ctx)
    path = tmp_path / "tlx_wave_test.mlir"
    path.write_text(_minimal_ttgir(
        public_funcs,
        target,
        threads_per_warp,
        num_ctas,
        num_warps,
        preamble,
    ))
    mod = ir.parse_mlir_module(str(path), ctx)
    # Compiler stage inputs carry their owning context. Direct parser fixtures
    # retain it explicitly so make_wave can run the same preparation passes.
    mod.context = ctx
    return mod, ctx


def _run_waveamd_to_machine(wave_artifact):
    wave_opt = wave_bridge_tools._wave_opt()
    result = subprocess.run(
        [
            wave_opt,
            "-",
            "--wave-lower-redistribute",
            "--wave-lower-symbolic-memory",
            "--wave-promote-global-to-buffer",
            "--wave-expand-integer-div-rem",
            "--wave-resolve-allocs",
            "--waveamd-dma-zero-fill",
            "--waveamd-to-machine",
        ],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _run_wave_verify(wave_artifact):
    result = subprocess.run(
        [wave_bridge_tools._wave_opt(), "-", "--verify-diagnostics"],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _run_wave_canonicalize(wave_artifact):
    result = subprocess.run(
        [wave_bridge_tools._wave_opt(), "-", "--canonicalize"],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _run_wave_lower_redistribute(wave_artifact):
    result = subprocess.run(
        [wave_bridge_tools._wave_opt(), "-", "--wave-lower-redistribute"],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _run_wave_lower_symbolic_memory(wave_artifact):
    result = subprocess.run(
        [
            wave_bridge_tools._wave_opt(),
            "-",
            "--wave-lower-redistribute",
            "--wave-lower-symbolic-memory",
        ],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _assert_mechanical_symbolic_copy(attrs, *, masked=None):
    assert attrs["mode"] == "symbolic_copy"
    assert attrs["destination_offset_mode"] == "layout_coordinates"
    assert attrs["component_count"] > 0
    assert attrs["destination_coordinate_shape"]
    assert len(attrs["destination_component_coordinate_bases"]) == attrs["component_count"]
    for policy_attr in (
            "packet_bytes",
            "packet_elements",
            "mask_alignment",
            "redundant_wave_mask",
            "destination_component_offsets",
            "destination_lane_stride_elements",
            "destination_wave_stride_elements",
            "destination_wave_stride_dwords",
            "destination_wave_offset_coefficients_dwords",
    ):
        assert policy_attr not in attrs
    if masked is not None:
        assert attrs["has_mask"] is masked
        assert attrs["mask_mode"] == ("exec_where" if masked else "none")


def _run_wave_compile_kernels(wave_artifact):
    wave_opt = wave_bridge_tools._wave_opt()
    result = subprocess.run(
        [
            wave_opt,
            "-",
            *wave_bridge_tools._wave_hsaco_pipeline_args(
                wave_opt,
                "gfx950",
            ),
        ],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout
