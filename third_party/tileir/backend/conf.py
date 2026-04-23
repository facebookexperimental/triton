from contextlib import contextmanager
import os
import warnings
import triton
import functools


_tileir_info_msg = """
The Triton-TileIR backend is an experimental feature and is not supported for production use.
If you would like to use this for benchmarking/investigation purposes, please set
TRITON_TILEIRAS_PATH to the path of the tileiras binary. This can be found in the
Cuda 13.1 toolkit under bin and is available by default on most devservers.

If you find substantial performance results when benchmarking please report your results
to the Triton team.
"""

_tileir_enabled_msg = """
The Triton-TileIR backend is enabled. This is an experimental feature and is
not supported for production use.

If you are using this solely for benchmarking purposes you can ignore this message.
If you find substantial performance results when benchmarking please report your results
to the Triton team.
"""


class TileIREnvConf:

    @staticmethod
    def enable_approx():
        # Enable approximate calculation, trading off numerical precision for performance gains
        return os.getenv("TILEIR_ENABLE_APPROX", "0") == "1"

    @staticmethod
    def enable_ftz():
        # Enable flush denormal to zero, trading off numerical precision for performance gains
        return os.getenv("TILEIR_ENABLE_FTZ", "0") == "1"

    @staticmethod
    def enable_autogen_alias_mem_token():
        return os.getenv("TILEIR_ENABLE_AUTOGEN_ALIAS_MEM_TOKEN", "1") == "1"

    @staticmethod
    def get_fmad_flag():
        # Default to True, but allow disabling via env var
        return os.getenv("TILE_IR_DISABLE_FMAD", "0") != "1"

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def get_tileiras_path():
        env_path = os.getenv("TRITON_TILEIRAS_PATH")
        if env_path:
            warnings.warn(_tileir_enabled_msg)
            return os.path.join(env_path, "tileiras")
        cuda_home = os.getenv("CUDA_HOME")
        if cuda_home:
            path = os.path.join(cuda_home, "bin", "tileiras")
            if os.path.exists(path):
                import subprocess
                version_output = subprocess.check_output([path, "--version"], encoding="utf-8",
                                                         stderr=subprocess.STDOUT)
                if "release 13.1" in version_output:
                    warnings.warn(_tileir_enabled_msg)
                    return path
        from shutil import which
        tileiras_path = which("tileiras")
        if tileiras_path:
            warnings.warn(_tileir_enabled_msg)
            return tileiras_path
        # TODO: FIXME HACK: FBCODE FALLBACK.
        # Buck does not always propagate environment variables to subprocesses,
        # so fall back to a well-known devserver path when no tileiras is found.
        from triton.runtime.fbcode_gating import is_fbcode_dependant
        if is_fbcode_dependant():
            warnings.warn(_tileir_info_msg)
            return "/usr/local/cuda/bin/tileiras"
        return None

    # todo: DKG CI related, need to be removed
    @staticmethod
    def get_device():
        return 'cpu' if os.environ.get("ENABLE_CPU_TORCH", False) else 'cuda'

    @staticmethod
    def in_nightly_pipeline():
        return os.getenv("RUN_FULL_TEST", "0") == "1"

    @staticmethod
    def in_release_pipeline():
        """Check if running in release pipeline environment"""
        return os.getenv("NVT_RUN_RELEASE_PIPELINE", "0") == "1"

    @staticmethod
    def get_sm_arch():
        import torch

        device = "cuda"
        cc = torch.cuda.get_device_capability(device)
        sm_arch = f"sm{cc[0]}{cc[1]}"
        return sm_arch

    @staticmethod
    def enable_tma_offset_assert_check():
        return os.getenv("NVT_TMA_OFFSET_CHECK", "0") == "1"


@contextmanager
def set_env_var(var_name, new_value):
    # Save the original value of the environment variable
    original_value = os.getenv(var_name, None)

    # Set the new value
    if new_value is None and var_name in os.environ:
        del os.environ[var_name]
    elif new_value is not None:
        os.environ[var_name] = str(new_value)
    try:
        yield
    finally:
        # Reset to the original value or remove the variable
        if original_value is not None:
            os.environ[var_name] = original_value
        elif var_name in os.environ:
            del os.environ[var_name]
