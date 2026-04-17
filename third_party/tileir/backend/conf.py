from contextlib import contextmanager
import os
import triton
import functools


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
            return os.path.join(env_path, "tileiras")
        cuda_home = os.getenv("CUDA_HOME")
        if cuda_home:
            path = os.path.join(cuda_home, "bin", "tileiras")
            if os.path.exists(path):
                import subprocess
                version_output = subprocess.check_output([path, "--version"], encoding="utf-8", stderr=subprocess.STDOUT)
                if "release 13.1" in version_output:
                    return path
        from shutil import which
        tileiras_path = which("tileiras")
        if tileiras_path:
            return tileiras_path
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
