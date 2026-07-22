import os

import triton.backends.amd.driver as amd_driver
from triton.backends.compiler import GPUTarget
from triton import knobs


class _TLXWaveUtils:

    def __init__(self, hip_utils):
        self._hip_utils = hip_utils

    def __getattr__(self, name):
        return getattr(self._hip_utils, name)

    def load_binary(self, name, kernel, shared, device):
        if not isinstance(kernel, (bytes, bytearray, memoryview)):
            raise RuntimeError("tlx_wave expected HSACO bytes for executable loading, got "
                               f"{type(kernel).__name__}. Inspect compiled.asm['wave'] and "
                               "compiled.asm['hsaco'] to verify the compiler pipeline reached "
                               "the hsaco stage.")
        kernel = bytes(kernel)
        if not kernel.startswith(b"\x7fELF"):
            raise RuntimeError("tlx_wave expected an ELF HSACO object for executable loading. "
                               "The cached artifact is not executable; inspect compiled.asm['wave'] "
                               "and compiled.asm['hsaco'].")
        return self._hip_utils.load_binary(name, kernel, shared, device)


class TLXWaveDriver(amd_driver.HIPDriver):
    """HIP runtime shim for selecting the TLX Wave compiler via tl.jit.

    This driver is active only when explicitly selected with
    TRITON_DEFAULT_BACKEND=tlx_wave, avoiding conflicts with the normal HIP
    driver during backend auto-discovery.
    """

    def __init__(self):
        super().__init__()
        self.utils = _TLXWaveUtils(self.utils)

    @staticmethod
    def is_active():
        return os.environ.get("TRITON_DEFAULT_BACKEND") == "tlx_wave" and amd_driver.HIPDriver.is_active()

    def get_current_target(self):
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        arch = knobs.runtime.override_arch or device_properties["arch"]
        arch = arch.split(":")[0]
        if arch not in {"gfx942", "gfx950"}:
            raise RuntimeError(f"tlx_wave stage-1 scaffold only supports gfx942/gfx950, got {arch}")
        return GPUTarget("tlx_wave", arch, 64)
