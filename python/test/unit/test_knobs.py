import os
import pytest
import shutil
import sys
import triton
from concurrent.futures import ThreadPoolExecutor
from triton._C.libtriton import get_cache_invalidating_env_vars  # type: ignore[attr-defined]
from triton._internal_testing import is_hip

from pathlib import Path


def test_knobs_utils(fresh_knobs) -> None:
    triton.knobs.propagate_env = False

    class test_knobs(triton.knobs.base_knobs):
        foo: triton.knobs.env_str = triton.knobs.env_str("FOO", "triton")
        bar: triton.knobs.env_bool = triton.knobs.env_bool("BAR", True)
        baz: triton.knobs.env_opt_str = triton.knobs.env_opt_str("BAZ")
        quux: triton.knobs.env_opt_bool = triton.knobs.env_opt_bool("QUUX")

    instance = test_knobs()

    # Make sure knobs works
    assert instance.knobs == {
        "foo": "triton",
        "bar": True,
        "baz": None,
        "quux": None,
    }

    # Now make sure copying works properly, otherwise all other tests in this
    # file aren't trustworthy.
    instance.bar = False
    instance.quux = True
    assert instance.foo == "triton"
    assert not instance.bar
    assert instance.baz is None
    assert instance.quux
    assert instance.knobs == {
        "foo": "triton",
        "bar": False,
        "baz": None,
        "quux": True,
    }

    second = instance.copy()
    assert second.foo == "triton"
    assert not second.bar
    assert second.baz is None
    assert second.quux

    second.foo = "tritium"
    assert instance.foo != "tritium"
    assert second.foo == "tritium"

    # Ditto on trustworthiness if reset() doesn't work.
    second.reset()
    assert second.knobs == {
        "foo": "triton",
        "bar": True,
        "baz": None,
        "quux": None,
    }
    # Triple check original instance didn't change.
    assert instance.knobs == {
        "foo": "triton",
        "bar": False,
        "baz": None,
        "quux": True,
    }


# Beta enables expert scheduling on gfx1250 by default; upstream does not.
@pytest.mark.parametrize(("arch", "enable_fp_fusion", "disable_opt", "expected_flags", "disable_optimization"), [
    ("gfx90a", True, None, [], False),
    ("gfx942", False, "1", ["amdgpu-use-amdgpu-trackers"], True),
    ("gfx950", True, "disable-lsr", ["amdgpu-use-amdgpu-trackers"], False),
    ("gfx1250", True, "0", ["amdgpu-expert-scheduling-mode"], False),
])
def test_amd_codegen_options(arch, enable_fp_fusion, disable_opt, expected_flags, disable_optimization, fresh_knobs,
                             monkeypatch):
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd import compiler

    calls = []

    def run_codegen(source, triple, processor, features, **kwargs):
        calls.append((source, triple, processor, features, kwargs))
        return ".text\n.globl test_kernel\ntest_kernel:\n  s_endpgm\n"

    monkeypatch.setattr(compiler, "compile_amdgpu", run_codegen)
    if disable_opt is not None:
        monkeypatch.setenv("DISABLE_LLVM_OPT", disable_opt)
    else:
        monkeypatch.delenv("DISABLE_LLVM_OPT", raising=False)

    warp_size = 32 if arch == "gfx1250" else 64
    backend = compiler.HIPBackend(GPUTarget("hip", arch, warp_size))
    options = compiler.HIPOptions(arch=arch, enable_fp_fusion=enable_fp_fusion)
    source = "define amdgpu_kernel void @test_kernel() { ret void }"
    metadata = {}
    assembly = backend.make_amdgcn(source, metadata, options)

    assert len(calls) == 1
    actual_source, triple, processor, features, arguments = calls[0]
    assert actual_source == source
    assert triple == compiler.amd.TARGET_TRIPLE
    assert processor == arch
    assert features == ""
    assert arguments["flags"] == expected_flags
    assert arguments["enable_fp_fusion"] is enable_fp_fusion
    assert arguments["disable_optimization"] is disable_optimization
    assert arguments["disabled_passes"] == ("disable-lsr" if disable_opt == "disable-lsr" else "")
    assert metadata["name"] == "test_kernel"
    assert "s_endpgm" in assembly


def test_amd_codegen_path_override(fresh_knobs, monkeypatch):
    from triton.backends.amd import compiler

    monkeypatch.delenv("TRITON_AMD_CODEGEN_PATH", raising=False)
    library = Path(compiler.get_amd_codegen_path())
    assert library.parent == Path(compiler.__file__).parent / "lib"
    assert "triton_amd_codegen" in library.name

    monkeypatch.setenv("TRITON_AMD_CODEGEN_PATH", "/custom/amd/codegen.so")
    assert compiler.get_amd_codegen_path() == "/custom/amd/codegen.so"


def test_amd_codegen_revision_invalidates_backend_hash(fresh_knobs, monkeypatch):
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd import compiler

    monkeypatch.setattr(compiler, "get_amd_codegen_revision", lambda: "first-revision-1")
    backend = compiler.HIPBackend(GPUTarget("hip", "gfx942", 64))
    first_hash = backend.hash()

    monkeypatch.setattr(compiler, "get_amd_codegen_revision", lambda: "second-revision-1")
    backend.hash.cache_clear()
    second_hash = backend.hash()

    assert first_hash != second_hash


def test_amd_codegen_inlines_functions_from_bitcode(fresh_knobs, monkeypatch):
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd import compiler

    source = '''
target triple = "amdgcn-amd-amdhsa"

define internal i32 @helper(i32 %value) {
  %result = add i32 %value, 1
  ret i32 %result
}

define amdgpu_kernel void @inline_kernel(ptr addrspace(1) %out, i32 %value) {
  %result = call i32 @helper(i32 %value)
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}
'''
    library = compiler._load_amd_codegen(compiler.get_amd_codegen_path())
    compile_bitcode = library.triton_amdgpu_compile
    inputs = []

    def check_bitcode(bitcode, size, *args):
        inputs.append(bitcode)
        assert bitcode.startswith(b"BC\xc0\xde")
        assert b"\x00" in bitcode
        assert size == len(bitcode)
        return compile_bitcode(bitcode, size, *args)

    monkeypatch.setattr(library, "triton_amdgpu_compile", check_bitcode)
    backend = compiler.HIPBackend(GPUTarget("hip", "gfx942", 64))
    assembly = backend.make_amdgcn(source, {}, compiler.HIPOptions(arch="gfx942"))

    assert len(inputs) == 1
    assert "inline_kernel:" in assembly
    assert "s_endpgm" in assembly
    assert "helper:" not in assembly


@pytest.mark.parametrize("option", ["dump_ir", "enable_timing"])
def test_amd_codegen_options_restored(option, capfd, fresh_knobs):
    from triton.backends.amd import compiler

    source = "define amdgpu_kernel void @test_kernel() { ret void }"
    options = dict(flags=[], enable_fp_fusion=True, disable_optimization=False, canonicalize_gep=False,
                   disabled_passes="", dump_ir=False, enable_timing=False)
    options[option] = True
    compiler.compile_amdgpu(source, compiler.amd.TARGET_TRIPLE, "gfx942", "", **options)
    assert capfd.readouterr().err

    options[option] = False
    compiler.compile_amdgpu(source, compiler.amd.TARGET_TRIPLE, "gfx942", "", **options)
    assert not capfd.readouterr().err


def test_amd_codegen_reports_invalid_llvm_ir(fresh_knobs):
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd import compiler

    source = "define amdgpu_kernel void @invalid_kernel() { invalid }"
    backend = compiler.HIPBackend(GPUTarget("hip", "gfx942", 64))
    with pytest.raises(RuntimeError, match="failed to parse LLVM IR"):
        backend.make_amdgcn(source, {}, compiler.HIPOptions(arch="gfx942"))


def test_amd_codegen_assembles_object_with_private_llvm(fresh_knobs):
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd import compiler

    source = '''
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @object_kernel() {
  ret void
}
'''
    backend = compiler.HIPBackend(GPUTarget("hip", "gfx942", 64))
    assembly = backend.make_amdgcn(source, {}, compiler.HIPOptions(arch="gfx942"))
    object_file = compiler.assemble_amdgcn(assembly, "gfx942", "")

    assert object_file.startswith(b"\x7fELF")


@pytest.mark.parametrize("enabled", [False, True])
def test_amd_codegen_respects_mir_dump_knob(enabled, fresh_knobs, monkeypatch, tmp_path):
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd import compiler

    events = []
    monkeypatch.setattr(compiler.llvm, "translate_to_mir", lambda *args: events.append("mir"))
    monkeypatch.setattr(compiler.llvm, "dump_sched_dag", lambda *args: events.append("dag"))
    monkeypatch.setattr(compiler, "compile_amdgpu", lambda *args, **kwargs: "s_endpgm")
    backend = compiler.HIPBackend(GPUTarget("hip", "gfx942", 64))
    source = "define amdgpu_kernel void @test_kernel() { ret void }"

    with fresh_knobs.amd.scope():
        if enabled:
            fresh_knobs.amd.dump_mir = str(tmp_path)
        else:
            fresh_knobs.amd.dump_mir = None
        backend.make_amdgcn(source, {}, compiler.HIPOptions(arch="gfx942"))

    assert events == (["mir", "dag"] if enabled else [])


def test_amd_codegen_preserves_mir_replacement(fresh_knobs, monkeypatch):
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd import compiler

    replacement_calls = []

    def replace_mir(path, *args):
        replacement_calls.append((path, args))
        return ".text\n.globl test_kernel\ntest_kernel:\n  s_endpgm\n"

    monkeypatch.setattr(compiler.llvm, "translate_mir_to_asm", replace_mir)
    monkeypatch.setattr(compiler, "compile_amdgpu", lambda *args, **kwargs: pytest.fail("MIR replacement bypassed"))
    backend = compiler.HIPBackend(GPUTarget("hip", "gfx942", 64))
    source = "define amdgpu_kernel void @test_kernel() { ret void }"

    with fresh_knobs.amd.scope():
        fresh_knobs.amd.swap_mir = "/custom/mir"
        assembly = backend.make_amdgcn(source, {}, compiler.HIPOptions(arch="gfx942"))

    assert len(replacement_calls) == 1
    assert replacement_calls[0][0].startswith("/custom/mir/test_kernel_")
    assert replacement_calls[0][0].endswith(".txt")
    assert "s_endpgm" in assembly


def test_knobs_scope(fresh_knobs, monkeypatch):
    fresh_knobs.amd.use_buffer_atomics = True

    # Update env *after* the __set__() does
    monkeypatch.setenv("AMDGCN_USE_BUFFER_ATOMICS", "0")

    assert fresh_knobs.amd.use_buffer_atomics

    # Just to prove that use_buffer_ops is coming from env
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    assert not fresh_knobs.amd.use_buffer_ops
    monkeypatch.delenv("AMDGCN_USE_BUFFER_OPS")
    assert fresh_knobs.amd.use_buffer_ops

    with fresh_knobs.amd.scope():
        # Use the environment
        del fresh_knobs.amd.use_buffer_atomics
        fresh_knobs.amd.use_buffer_ops = False

        assert not fresh_knobs.amd.use_buffer_atomics
        assert not fresh_knobs.amd.use_buffer_ops

    assert fresh_knobs.amd.use_buffer_atomics
    assert fresh_knobs.amd.use_buffer_ops

    # Just to prove that use_buffer_ops is coming from env
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    assert not fresh_knobs.amd.use_buffer_ops
    monkeypatch.delenv("AMDGCN_USE_BUFFER_OPS")
    assert fresh_knobs.amd.use_buffer_ops


def test_env_updated(fresh_knobs, monkeypatch):
    fresh_knobs.amd.use_buffer_ops = False
    assert os.getenv("AMDGCN_USE_BUFFER_OPS") == "0"
    # Just triple checking both APIs give us what we expect
    assert os.environ["AMDGCN_USE_BUFFER_OPS"] == "0"

    fresh_knobs.cache.home_dir = "/foo/bar"
    assert os.getenv("TRITON_HOME") == "/foo/bar"
    assert os.environ["TRITON_HOME"] == "/foo/bar"


def test_scalarize_packed_fops_invalidates_cache(monkeypatch):
    monkeypatch.setenv("AMDGCN_SCALARIZE_PACKED_FOPS", "0")
    disabled = get_cache_invalidating_env_vars()

    monkeypatch.setenv("AMDGCN_SCALARIZE_PACKED_FOPS", "1")
    enabled = get_cache_invalidating_env_vars()

    assert disabled["AMDGCN_SCALARIZE_PACKED_FOPS"] == "false"
    assert enabled["AMDGCN_SCALARIZE_PACKED_FOPS"] == "true"


@pytest.mark.parametrize("truthy, falsey", [("1", "0"), ("true", "false"), ("True", "False"), ("TRUE", "FALSE"),
                                            ("y", "n"), ("YES", "NO"), ("ON", "OFF")])
def test_read_env(truthy, falsey, fresh_knobs_including_libraries, monkeypatch):
    fresh_knobs = fresh_knobs_including_libraries
    # bool defaulting to False
    assert not fresh_knobs.runtime.debug
    # bool defaulting to True
    assert fresh_knobs.language.default_fp_fusion
    # str defaulting to None
    assert fresh_knobs.compilation.use_ir_loc is None
    # str defaulting to not None
    assert fresh_knobs.cache.dir.endswith(".triton/cache")
    # class defaulting to None
    assert fresh_knobs.cache.manager_class is None
    # set[str] defaulting to empty
    assert len(fresh_knobs.build.backend_dirs) == 0

    monkeypatch.setenv("TRITON_DEFAULT_FP_FUSION", falsey)
    monkeypatch.setenv("TRITON_DEBUG", truthy)
    monkeypatch.setenv("USE_IR_LOC", "ttir")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/triton_cache")
    monkeypatch.setenv("TRITON_HOME", "/tmp/triton_home")
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "triton.runtime.cache:FileCacheManager")
    monkeypatch.setenv("TRITON_CUDACRT_PATH", "/tmp/cuda/crt")
    monkeypatch.setenv("TRITON_CUDART_PATH", "/tmp/cuda/rt")

    triton.knobs.refresh_knobs()
    assert fresh_knobs.runtime.debug
    assert not fresh_knobs.language.default_fp_fusion
    assert fresh_knobs.compilation.use_ir_loc == "ttir"
    assert fresh_knobs.cache.home_dir == "/tmp/triton_home"
    assert fresh_knobs.cache.dir == "/tmp/triton_cache"
    assert fresh_knobs.cache.dump_dir == "/tmp/triton_home/.triton/dump"
    assert fresh_knobs.cache.override_dir == "/tmp/triton_home/.triton/override"

    from triton.runtime.cache import FileCacheManager

    assert fresh_knobs.cache.manager_class == FileCacheManager

    assert fresh_knobs.build.backend_dirs == {"/tmp/cuda/crt", "/tmp/cuda/rt"}


def _register_pressure_scheduler_kernel():
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import compiler

    compiler.llvm.init_targets()
    backend = compiler.CUDABackend(GPUTarget("cuda", 90, 32))
    arithmetic = []
    stores = []
    for lane in range(32):
        arithmetic.extend([
            f"  %seed{lane} = add i32 %seed, {lane + 1}",
            f"  %high{lane} = mul i32 %seed{lane}, -766435501",
            f"  %low{lane} = mul i32 %seed{lane}, -845247145",
            f"  %value{lane} = xor i32 %high{lane}, %low{lane}",
        ])
        stores.extend([
            f"  %out{lane} = getelementptr i32, ptr addrspace(1) %out, i64 {lane}",
            f"  store i32 %value{lane}, ptr addrspace(1) %out{lane}, align 4",
        ])

    source = ('target triple = "nvptx64-nvidia-cuda"\n'
              'define ptx_kernel void @scheduler_kernel(ptr addrspace(1) %out, i32 %seed) {\n' +
              "\n".join(arithmetic + stores) + "\n  ret void\n}\n")
    return backend, source


@pytest.mark.skipif(is_hip(), reason="NVPTX code generation is unavailable on AMD")
@pytest.mark.parametrize("link_hip_first", [False, True])
def test_nvidia_register_pressure_scheduler_hook(link_hip_first, fresh_triton_cache):
    backend, source = _register_pressure_scheduler_kernel()
    if link_hip_first:
        # HIP linking resets LLVM command-line options.
        _compile_empty_hip_kernel("gfx942", 64)

    default_options = backend.parse_options({"ptx_version": 80, "sched4reg": False})
    pressure_options = backend.parse_options({"ptx_version": 80, "sched4reg": True})
    default_ptx = backend.make_ptx(source, {}, default_options, 90)
    pressure_ptx = backend.make_ptx(source, {}, pressure_options, 90)
    assert default_ptx != pressure_ptx


def _nvidia_short_pointer_kernel():
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import compiler

    compiler.llvm.init_targets()
    backend = compiler.CUDABackend(GPUTarget("cuda", 90, 32))
    source = '''
target triple = "nvptx64-nvidia-cuda"
@shared = external addrspace(3) global [0 x i8]
define ptx_kernel void @short_pointer_kernel(ptr addrspace(1) %out, i64 %offset) {
  %ptr = getelementptr i8, ptr addrspace(3) @shared, i64 %offset
  %value = ptrtoint ptr addrspace(3) %ptr to i64
  store i64 %value, ptr addrspace(1) %out
  ret void
}
'''
    return backend, source


@pytest.mark.skipif(is_hip(), reason="NVPTX code generation is unavailable on AMD")
@pytest.mark.parametrize("link_hip_first", [False, True])
def test_nvidia_short_pointer_option(link_hip_first, fresh_triton_cache):
    from triton.backends.compiler import GPUTarget

    if link_hip_first:
        # HIP linking resets every LLVM command-line option.
        _compile_empty_hip_kernel("gfx942", 64)

    @triton.jit
    def empty_kernel():
        return

    compiled = triton.compile(triton.compiler.ASTSource(empty_kernel, {}), target=GPUTarget("cuda", 90, 32))
    data_layout = next(line for line in compiled.asm["llir"].splitlines() if line.startswith("target datalayout"))
    assert "p3:32:32" in data_layout

    backend, source = _nvidia_short_pointer_kernel()
    options = backend.parse_options({"ptx_version": 80})
    ptx = backend.make_ptx(source, {}, options, 90)
    assert "\tadd.s32 " in ptx
    assert "\tadd.s64 " not in ptx


def _compile_empty_hip_kernel(arch, warp_size):
    from triton.backends.compiler import GPUTarget

    @triton.jit
    def empty_kernel():
        return

    return triton.compile(triton.compiler.ASTSource(empty_kernel, {}), target=GPUTarget("hip", arch, warp_size))


def _amd_scheduler_kernel():
    arithmetic = []
    stores = []
    for lane in range(32):
        arithmetic.extend([
            f"  %seed{lane} = add i32 %seed, {lane + 1}",
            f"  %high{lane} = mul i32 %seed{lane}, -766435501",
            f"  %low{lane} = mul i32 %seed{lane}, -845247145",
            f"  %value{lane} = xor i32 %high{lane}, %low{lane}",
        ])
        stores.extend([
            f"  %out{lane} = getelementptr i32, ptr addrspace(1) %out, i64 {lane}",
            f"  store i32 %value{lane}, ptr addrspace(1) %out{lane}, align 4",
        ])

    return ('target triple = "amdgcn-amd-amdhsa"\n'
            'define amdgpu_kernel void @scheduler_kernel(ptr addrspace(1) %out, i32 %seed) {\n' +
            "\n".join(arithmetic + stores) + "\n  ret void\n}\n")


def test_amd_llvm_options_concurrent():
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd import compiler

    compiler.llvm.init_targets()
    source = _amd_scheduler_kernel()
    # gfx950 codegen needs a process-wide LLVM option that gfx1250 codegen must
    # not see, and every HSACO link resets all LLVM options.
    backends = {
        arch: compiler.HIPBackend(GPUTarget("hip", arch, warp_size))
        for arch, warp_size in [("gfx950", 64), ("gfx1250", 32)]
    }

    def emit(arch):
        backend = backends[arch]
        options = backend.parse_options({})
        metadata = {}
        amdgcn = backend.make_amdgcn(source, metadata, options)
        backend.make_hsaco(amdgcn, metadata, options)
        return amdgcn

    expected = {arch: emit(arch) for arch in backends}
    assert expected["gfx950"] != expected["gfx1250"]

    archs = [("gfx950", "gfx1250")[index % 2] for index in range(24)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(emit, archs))

    assert results == [expected[arch] for arch in archs]


def test_triton_home(fresh_knobs, monkeypatch):
    initial_home = fresh_knobs.cache.home_dir
    assert initial_home == os.path.expanduser("~/")
    assert fresh_knobs.cache.dir == os.path.join(initial_home, ".triton/cache")
    assert fresh_knobs.cache.dump_dir == os.path.join(initial_home, ".triton/dump")
    assert fresh_knobs.cache.override_dir == os.path.join(initial_home, ".triton/override")

    monkeypatch.setenv("TRITON_HOME", "/tmp/triton_home")
    assert fresh_knobs.cache.dir == "/tmp/triton_home/.triton/cache"
    assert fresh_knobs.cache.dump_dir == "/tmp/triton_home/.triton/dump"
    assert fresh_knobs.cache.override_dir == "/tmp/triton_home/.triton/override"

    fresh_knobs.cache.home_dir = "/tmp/user/triton_home"
    assert fresh_knobs.cache.dir == "/tmp/user/triton_home/.triton/cache"
    assert fresh_knobs.cache.dump_dir == "/tmp/user/triton_home/.triton/dump"
    assert fresh_knobs.cache.override_dir == "/tmp/user/triton_home/.triton/override"


def test_set_knob_directly(fresh_knobs_including_libraries, monkeypatch):
    fresh_knobs = fresh_knobs_including_libraries
    assert fresh_knobs.cache.dir.endswith(".triton/cache")

    fresh_knobs.cache.dir = "/tmp/triton_cache"
    assert fresh_knobs.cache.dir == "/tmp/triton_cache"

    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/other_triton_cache")
    assert fresh_knobs.cache.dir == "/tmp/triton_cache"

    # Disable propagation to verify resetting/del behavior
    triton.knobs.propagate_env = False

    fresh_knobs.cache.dir = fresh_knobs.env
    assert fresh_knobs.cache.dir == "/tmp/other_triton_cache"

    fresh_knobs.cache.dir = "/tmp/triton_cache"
    fresh_knobs.cache.reset()
    assert fresh_knobs.cache.dir == "/tmp/other_triton_cache"

    triton.knobs.propagate_env = True

    # Just in case, lets check all the other datatypes too
    fresh_knobs.language.default_fp_fusion = False
    fresh_knobs.amd.use_block_pingpong = True
    fresh_knobs.redis.port = 6380
    fresh_knobs.nvidia.mock_ptx_version = "42.0.1"

    from triton.runtime.cache import FileCacheManager

    class TestManagerClass(FileCacheManager):
        pass

    fresh_knobs.cache.manager_class = TestManagerClass

    monkeypatch.setenv("TRITON_CUDART_PATH", "/tmp/the/real/cudart")
    monkeypatch.setenv("TRITON_DEFAULT_FP_FUSION", "1")
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0")
    monkeypatch.setenv("TRITON_REDIS_PORT", "6381")
    monkeypatch.setenv("TRITON_MOCK_PTX_VERSION", "1.0.0")
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "triton.runtime.cache:FileCacheManager")

    assert not fresh_knobs.language.default_fp_fusion
    assert fresh_knobs.amd.use_block_pingpong
    assert fresh_knobs.redis.port == 6380
    assert fresh_knobs.nvidia.mock_ptx_version == "42.0.1"
    assert fresh_knobs.cache.manager_class == TestManagerClass

    # Make sure both setting `.env` or deleting resets to env vars.
    fresh_knobs.language.default_fp_fusion = fresh_knobs.env
    fresh_knobs.amd.use_block_pingpong = fresh_knobs.env
    fresh_knobs.redis.port = fresh_knobs.env
    del fresh_knobs.nvidia.mock_ptx_version
    del fresh_knobs.cache.manager_class

    assert fresh_knobs.build.backend_dirs == {"/tmp/the/real/cudart"}
    assert fresh_knobs.language.default_fp_fusion
    assert not fresh_knobs.amd.use_block_pingpong
    assert fresh_knobs.redis.port == 6381
    assert fresh_knobs.nvidia.mock_ptx_version == "1.0.0"
    assert fresh_knobs.cache.manager_class == FileCacheManager


@pytest.mark.skipif(
    is_hip(),
    reason="PTXAS is not installed on AMD",
)
def test_nvidia_tool(fresh_knobs, tmp_path, monkeypatch):
    triton_root = Path(fresh_knobs.__file__).parent
    default_ptxas = triton_root / "backends/nvidia/bin/ptxas"

    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options is None

    tmp_ptxas = tmp_path / "ptxas-special"
    shutil.copy(default_ptxas, tmp_ptxas)
    monkeypatch.setenv("TRITON_PTXAS_PATH", str(tmp_ptxas))
    monkeypatch.setenv("PTXAS_OPTIONS", "--verbose")
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options == "--verbose"

    # Don't prop so that the `del` is correctly tested
    fresh_knobs.propagate_env = False
    fresh_knobs.nvidia.ptxas = str(default_ptxas)
    fresh_knobs.nvidia.ptxas_options = "--device-debug"
    fresh_knobs.propagate_env = True
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options == "--device-debug"

    del fresh_knobs.nvidia.ptxas
    del fresh_knobs.nvidia.ptxas_options
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options == "--verbose"

    # Triple check scope works
    with fresh_knobs.nvidia.scope():
        fresh_knobs.nvidia.ptxas = str(default_ptxas)
        fresh_knobs.nvidia.ptxas_options = "--device-debug"
        assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()
        assert fresh_knobs.nvidia.ptxas_options == "--device-debug"

    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options == "--verbose"

    monkeypatch.delenv("TRITON_PTXAS_PATH")
    monkeypatch.delenv("PTXAS_OPTIONS")
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options is None


def _fake_tool(path, body):
    """A stand-in for ptxas: a script running `body` under this interpreter."""
    path.write_text(f"#!{sys.executable}\nimport sys\n{body}\n")
    path.chmod(0o755)
    return path


def test_nvidia_tool_probe_reports_reason(tmp_path):
    probe = triton.knobs.NvidiaTool.probe

    tool, reason = probe(str(tmp_path / "absent"))
    assert tool is None
    assert reason == "no such file"

    # The case that motivated this: present and executable, but killed at exec.
    broken = _fake_tool(tmp_path / "broken", 'sys.stderr.write("undefined symbol: unw_backtrace\\n"); sys.exit(127)')
    tool, reason = probe(str(broken))
    assert tool is None
    assert "exited with status 127" in reason
    assert "undefined symbol: unw_backtrace" in reason

    # Executable, but not a runnable image: exec fails with ENOEXEC, an OSError
    # that is not FileNotFoundError, and it has to come back as a reason rather
    # than propagate out of the candidate loop. ENOEXEC rather than a cleared
    # execute bit so the case does not depend on who runs the test.
    not_a_binary = tmp_path / "not-a-binary"
    not_a_binary.write_bytes(b"\x7fnot an ELF header\n")
    not_a_binary.chmod(0o755)
    tool, reason = probe(str(not_a_binary))
    assert tool is None
    assert reason.startswith("cannot execute: ")

    unparseable = _fake_tool(tmp_path / "unparseable", 'print("not a version string")')
    tool, reason = probe(str(unparseable))
    assert tool is None
    assert "release X.Y" in reason

    good = _fake_tool(tmp_path / "good", 'print("Cuda compilation tools, release 12.8, V12.8.93")')
    tool, reason = probe(str(good))
    assert reason is None
    assert tool.version == "12.8"


def test_nvidia_tool_probe_truncates_output(tmp_path):
    noisy = _fake_tool(tmp_path / "noisy", 'sys.stderr.write("E" * 100000); sys.exit(1)')
    _, reason = triton.knobs.NvidiaTool.probe(str(noisy))
    assert len(reason) < 4096
    assert reason.startswith("`--version` exited with status 1: ...")


def test_nvidia_tool_error_lists_all_candidates(fresh_knobs, tmp_path, monkeypatch):
    broken = _fake_tool(tmp_path / "broken-ptxas", 'sys.stderr.write("boom\\n"); sys.exit(127)')
    absent = tmp_path / "absent-ptxas"

    # Both candidates must fail for transform() to raise.
    monkeypatch.setattr(type(fresh_knobs.nvidia).__dict__["ptxas"], "default_path", str(absent))
    monkeypatch.setenv("TRITON_PTXAS_PATH", str(broken))

    with pytest.raises(RuntimeError) as excinfo:
        fresh_knobs.nvidia.ptxas

    msg = str(excinfo.value)
    assert "Cannot find ptxas" in msg
    # The interpreter running the fake tool is free to add its own noise to the
    # captured stream, so `boom` is not necessarily flush against the status --
    # only assert it lands in the broken candidate's reason.
    broken_prefix = f"{broken}: `--version` exited with status 127:"
    absent_reason = f"{absent}: no such file"
    assert broken_prefix in msg
    assert absent_reason in msg
    broken_reason = msg.split(broken_prefix, 1)[1].split(absent_reason, 1)[0]
    assert "boom" in broken_reason


def test_nvidia_tool_env_hook(fresh_knobs, tmp_path, monkeypatch):
    # A tool that dies unless a specific variable is scrubbed from its
    # environment -- the shape of an inherited LD_PRELOAD that only resolves
    # inside the host interpreter.
    poison = "TRITON_TEST_POISONED_ENV"
    tool = _fake_tool(
        tmp_path / "ptxas", 'import os\n'
        f'if os.environ.get("{poison}"):\n'
        '    sys.stderr.write("undefined symbol: unw_backtrace\\n"); sys.exit(127)\n'
        'print("Cuda compilation tools, release 12.8, V12.8.93")')
    monkeypatch.setenv(poison, "1")

    # Drive both directions explicitly rather than relying on the ambient value:
    # `fresh_knobs` deliberately does not reset `nvidia`, and a packager may have
    # installed a hook at import (fbcode's `set_configs()` does). `scope()`
    # snapshots `nvidia.__dict__` and restores it on exit, so whatever was
    # installed survives this test.
    with fresh_knobs.nvidia.scope():
        # No hook: the environment is inherited as-is, so the tool dies.
        fresh_knobs.nvidia.tool_env = None
        assert fresh_knobs.nvidia.get_tool_env() is None
        triton.knobs.NvidiaTool.probe.cache_clear()
        tool_obj, reason = triton.knobs.NvidiaTool.probe(str(tool))
        assert tool_obj is None
        assert "unw_backtrace" in reason

        # With a hook that scrubs the marker, the same tool resolves.
        fresh_knobs.nvidia.tool_env = lambda: {k: v for k, v in os.environ.items() if k != poison}
        triton.knobs.NvidiaTool.probe.cache_clear()
        tool_obj, reason = triton.knobs.NvidiaTool.probe(str(tool))
        assert reason is None
        assert tool_obj.version == "12.8"

    # `probe` is lru_cached per path and the cache is process-wide; don't leave
    # entries behind that were resolved under this test's hook.
    triton.knobs.NvidiaTool.probe.cache_clear()


def test_opt_bool(fresh_knobs_including_libraries, monkeypatch):
    fresh_knobs = fresh_knobs_including_libraries
    assert fresh_knobs.amd.use_block_pingpong is None
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0")
    assert not fresh_knobs.amd.use_block_pingpong
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "1")
    assert fresh_knobs.amd.use_block_pingpong
    monkeypatch.delenv("TRITON_HIP_USE_BLOCK_PINGPONG")
    assert fresh_knobs.amd.use_block_pingpong is None


def test_autotune_warmup_rep_defaults(fresh_knobs):
    assert fresh_knobs.autotuning.warmup == 25
    assert fresh_knobs.autotuning.rep == 100


def test_autotune_warmup_rep_env(fresh_knobs, monkeypatch):
    monkeypatch.setenv("TRITON_AUTOTUNE_WARMUP_MS", "50")
    monkeypatch.setenv("TRITON_AUTOTUNE_REP_MS", "200")
    assert fresh_knobs.autotuning.warmup == 50
    assert fresh_knobs.autotuning.rep == 200


def test_autotune_warmup_rep_set_directly(fresh_knobs):
    fresh_knobs.autotuning.warmup = 10
    fresh_knobs.autotuning.rep = 40
    assert fresh_knobs.autotuning.warmup == 10
    assert fresh_knobs.autotuning.rep == 40


def test_autotune_warmup_rep_reset(fresh_knobs, monkeypatch):
    triton.knobs.propagate_env = False
    fresh_knobs.autotuning.warmup = 10
    fresh_knobs.autotuning.rep = 40
    fresh_knobs.autotuning.reset()
    assert fresh_knobs.autotuning.warmup == 25
    assert fresh_knobs.autotuning.rep == 100
    triton.knobs.propagate_env = True


def test_autotune_warmup_rep_scope(fresh_knobs, monkeypatch):
    fresh_knobs.autotuning.warmup = 10
    fresh_knobs.autotuning.rep = 40

    with fresh_knobs.autotuning.scope():
        fresh_knobs.autotuning.warmup = 77
        fresh_knobs.autotuning.rep = 88
        assert fresh_knobs.autotuning.warmup == 77
        assert fresh_knobs.autotuning.rep == 88

    assert fresh_knobs.autotuning.warmup == 10
    assert fresh_knobs.autotuning.rep == 40
