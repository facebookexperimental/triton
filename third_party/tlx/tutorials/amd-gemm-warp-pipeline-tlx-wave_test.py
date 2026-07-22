import importlib.util
import subprocess
from pathlib import Path

import pytest
import torch


def _load_gfx9_v9_module(module_name="tlx_wave_gfx9_v9_tutorial"):
    path = (Path(__file__).parent / "gfx9_gemm" / "a16w16" / "v9_beyond_hotloop" / "matmul_kernel.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TLX_WAVE_STOP_AFTER_WAVE_KEY = "tlx-wave-v9-asm-stop-after-wave-v1"


def _tlx_wave_stop_after_wave_hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(arg is None for arg in (stages, options, language, capability)):
        return _TLX_WAVE_STOP_AFTER_WAVE_KEY, _TLX_WAVE_STOP_AFTER_WAVE_KEY

    # This tutorial test compares the Wave handoff and text assembly against
    # the existing AMD backend. Full v9 HSACO assembly is tracked separately
    # because Wave currently emits a constant-bus-illegal instruction there.
    def keep_wave_as_test_binary(src, metadata):
        metadata["tlx_wave_binary_stage"] = "wave-inspection-only"
        return src

    stages["hsaco"] = keep_wave_as_test_binary
    return _TLX_WAVE_STOP_AFTER_WAVE_KEY, _TLX_WAVE_STOP_AFTER_WAVE_KEY


def _warmup_gfx9_v9_backend(
    tmp_path,
    monkeypatch,
    *,
    backend_name,
    expected_target_backend,
    cache_suffix,
    module_suffix,
    m=256,
    n=256,
    k=128,
    block_m=256,
    block_n=256,
    block_k=64,
    group_m=4,
    num_xcds=8,
):
    import triton
    from triton import knobs
    from triton.backends import backends
    from triton.runtime.jit import MockTensor

    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", backend_name)
    if backend_name == "tlx_wave":
        wave_opt = (Path(__file__).parents[2] / "wave" / "build" / "wave-build" / "bin" / "wave-opt")
        if wave_opt.exists():
            monkeypatch.setenv("TRITON_WAVE_OPT", str(wave_opt))

    if backend_name not in backends:
        pytest.skip(f"{backend_name} backend is not installed")

    with knobs.cache.scope(), knobs.runtime.scope():
        knobs.cache.dir = str(tmp_path / f"triton-cache-v9-{cache_suffix}")
        knobs.runtime.override_arch = "gfx950"
        if backend_name == "tlx_wave":
            knobs.runtime.add_stages_inspection_hook = _tlx_wave_stop_after_wave_hook
        triton.runtime.driver._default = None
        triton.runtime.driver._active = None
        try:
            target = triton.runtime.driver.active.get_current_target()
        except RuntimeError as exc:
            pytest.skip(f"{backend_name} backend is not active: {exc}")
        if target.backend != expected_target_backend or target.arch != "gfx950":
            pytest.skip(f"requires {expected_target_backend}:gfx950, got {target}")

        tutorial = _load_gfx9_v9_module(f"tlx_wave_gfx9_v9_tutorial_{module_suffix}")
        grid_mn = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)

        a = MockTensor(torch.float16, [m, k])
        b = MockTensor(torch.float16, [k, n])
        c = MockTensor(torch.float16, [m, n])
        a_strides = a.stride()
        b_strides = b.stride()
        c_strides = c.stride()
        compiled = tutorial.v9_beyond_hotloop.warmup(
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_SIZE_M=group_m,
            NUM_XCDS=num_xcds,
            GRID_MN=grid_mn,
            num_warps=8,
            num_stages=1,
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            grid=(grid_mn, ),
        )
    return compiled


def _warmup_gfx9_v9_tlx_wave(tmp_path, monkeypatch, **kwargs):
    return _warmup_gfx9_v9_backend(
        tmp_path,
        monkeypatch,
        backend_name="tlx_wave",
        expected_target_backend="tlx_wave",
        cache_suffix="tlx-wave",
        module_suffix="tlx_wave",
        **kwargs,
    )


def _warmup_gfx9_v9_amd(tmp_path, monkeypatch, **kwargs):
    return _warmup_gfx9_v9_backend(
        tmp_path,
        monkeypatch,
        backend_name="amd",
        expected_target_backend="hip",
        cache_suffix="amd",
        module_suffix="amd",
        **kwargs,
    )


def _asm_text(compiled, artifact):
    text = compiled.asm[artifact]
    if isinstance(text, bytes):
        text = text.decode()
    return text


def _wave_text(compiled):
    return _asm_text(compiled, "wave")


def _run_wave_promote_buffer_to_machine(wave_artifact):
    from triton.backends.tlx_wave.wave_bridge_tools import _wave_machine_pipeline_args

    wave_opt = (Path(__file__).parents[2] / "wave" / "build" / "wave-build" / "bin" / "wave-opt")
    if not wave_opt.exists():
        pytest.skip("wave-opt is not built")
    result = subprocess.run(
        [str(wave_opt), "-", *_wave_machine_pipeline_args(str(wave_opt), "gfx950")],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _run_wave_to_amdgpu_asm(wave_artifact):
    wave_bin = Path(__file__).parents[2] / "wave" / "build" / "wave-build" / "bin"
    wave_translate = wave_bin / "wave-translate"
    if not wave_translate.exists():
        pytest.skip("wave-translate is not built")
    asm = subprocess.run(
        [str(wave_translate), "--wave-to-amdgpu-asm", "-"],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert asm.returncode == 0, asm.stderr or asm.stdout
    return asm.stdout, asm.stderr


def test_gfx9_v9_tlx_wave_warmup_lowers_to_machine(monkeypatch, tmp_path):
    amd_compiled = _warmup_gfx9_v9_amd(tmp_path, monkeypatch)
    amd_asm = _asm_text(amd_compiled, "amdgcn")

    compiled = _warmup_gfx9_v9_tlx_wave(tmp_path, monkeypatch)

    wave = _wave_text(compiled)
    machine = _run_wave_promote_buffer_to_machine(wave)

    assert "gpu.module @kernels" in wave
    assert "gpu.kernel" in wave
    assert compiled.metadata.tlx_wave_status == "emitted_wave_staged_converter"
    assert compiled.metadata.tlx_wave_wave_builder == "staged-converter"
    assert compiled.metadata.tlx_wave_num_mmas == 128
    assert compiled.metadata.tlx_wave_num_dma_load_lds == 16
    assert wave.count("wave.index_expr") < 1_900
    assert wave.count("wave.cast fpconvert") == 32
    # The symbolic bridge keeps the output as two masked scatters.  Check the
    # structural path rather than the pre-gather/scatter scalar store form.
    assert wave.count("wave.scatter") == 2
    assert "wave.store" not in wave
    assert wave.count("wave.where") == 2
    assert wave.count("wave.join") <= 20
    assert wave.count("wave.barrier") == 1
    assert wave.count("wave.extract") < 512
    assert "waveamdmachine.mfma_f32_16x16x32_f16" in machine
    assert machine.count("waveamdmachine.v_cvt_pk_f16_f32") == 64
    assert "waveamdmachine.v_cvt_f16_f32" not in machine
    assert machine.count("waveamdmachine.buffer_load_lds_b128") == 16
    assert "waveamdmachine.global_load_lds_b128" not in machine
    assert machine.count("waveamdmachine.ds_load_b128") == 16
    assert machine.count("waveamdmachine.token_join") <= 20
    assert machine.count("waveamdmachine.buffer_store_b64") == 32
    assert machine.count("waveamdmachine.v_cndmask_b32_tuple") == 0
    assert amd_asm.count("v_cndmask_b32") == 32
    assert machine.count("waveamdmachine.exec_if") == 32
    assert "waveamdmachine.buffer_store_b16" not in machine
    assert "waveamdmachine.global_store_b16_addr64" not in machine
    assert "waveamdmachine.s_addc_u32" not in machine
    assert "waveamdmachine.v_addc" not in machine
    asm, _diagnostics = _run_wave_to_amdgpu_asm(wave)
    assert asm.count("v_mfma") == amd_asm.count("v_mfma") == 128
    assert asm.count("buffer_load") == amd_asm.count("buffer_load") == 16
    assert asm.count("buffer_store") == amd_asm.count("buffer_store") == 32
    assert asm.count("v_cvt_pk_f16_f32") == amd_asm.count("v_cvt_pk_f16_f32") == 64
    assert asm.count("s_barrier") == 1
    assert amd_asm.count("s_barrier") == 2
    assert "global_load" not in asm
    assert "global_load" not in amd_asm
    assert "global_store" not in asm
    assert "global_store" not in amd_asm
    assert "v_cvt_f16_f32" not in asm
    assert "v_cvt_f16_f32" not in amd_asm
    assert "s_addc" not in asm
    assert "s_addc" not in amd_asm
    assert "v_addc" not in asm
    assert "v_addc" not in amd_asm
    assert asm.count("s_waitcnt") < 2 * amd_asm.count("s_waitcnt")
    assert "waveamdmachine.s_lshl_b64" not in machine
    assert "waveamdmachine.v_lshlrev_b64" not in machine
    assert "waveamdmachine.v_lshrrev_b64" not in machine
    assert "waveamdmachine.v_add_u64" not in machine
    assert machine.count("waveamdmachine.v_cmp") <= 24
    assert machine.count("waveamdmachine.s_cmp_lg_u32") <= 32
    assert machine.count("waveamdmachine.s_cselect_b32") <= 64
    assert machine.count("waveamdmachine.s_xor_b32") <= 32
    assert machine.count("waveamdmachine.s_lshr_b64") <= 16
    assert machine.count("waveamdmachine.s_add_u64") <= 64
    assert len(machine.splitlines()) < 5_000
    assert machine.count("waveamdmachine.tuple_to_elements") < 384
    assert machine.count("waveamdmachine.tuple_from_elements") < 640
    assert machine.count("waveamdmachine.v_mov_b32_tuple") < 256


def test_gfx9_v9_tlx_wave_hot_loop_waits_are_not_full_drains(monkeypatch, tmp_path):
    amd_compiled = _warmup_gfx9_v9_amd(tmp_path, monkeypatch, k=256)
    amd_asm = _asm_text(amd_compiled, "amdgcn")

    compiled = _warmup_gfx9_v9_tlx_wave(tmp_path, monkeypatch, k=256)
    wave = _wave_text(compiled)
    machine = _run_wave_promote_buffer_to_machine(wave)
    asm, _diagnostics = _run_wave_to_amdgpu_asm(wave)

    assert compiled.metadata.tlx_wave_status == "emitted_wave_staged_converter"
    assert compiled.metadata.tlx_wave_num_mmas == 256
    assert compiled.metadata.tlx_wave_num_dma_load_lds == 32
    assert asm.count("v_mfma") == amd_asm.count("v_mfma") == 256
    assert asm.count("buffer_load") == amd_asm.count("buffer_load") == 32
    assert asm.count("buffer_store") == amd_asm.count("buffer_store") == 32
    assert wave.count("wave.wait") == 0
    assert wave.count("wave.barrier") == 11
    assert asm.count("s_barrier") == 11
    assert machine.count("waveamdmachine.s_waitcnt vmcnt(10)") == 0
    assert machine.count("waveamdmachine.s_waitcnt vmcnt(8)") == 4
    assert asm.count("s_waitcnt vmcnt(10)") == 0
    assert asm.count("s_waitcnt vmcnt(8)") == 4
    assert asm.count("s_waitcnt vmcnt(0)") == 1
    assert asm.count("s_waitcnt vmcnt(0)") < amd_asm.count("s_waitcnt vmcnt(0)")
