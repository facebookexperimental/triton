"""SM/CU holdout shared by TLX ops.

The count comes from the process-wide PyTorch carveout
(`torch._C._get_sm_carveout_experimental`, set with
`torch._C._set_sm_carveout_experimental`); 0 (or no API) keeps historical
behavior. This matches inductor, which sizes persistent grids as
`max_sms - carveout`.

Caveat: on HIP, PyTorch's own cuBLAS carveout stream currently confines BLAS
to roughly the carveout value CUs (measured: larger values run faster), the
inverse of the CUDA semantic (`smCount - carveout`) and of inductor. TLX
follows the documented intent everywhere (hold N out for others, e.g. NCCL
overlap); see holdout_notes.md.

Everything here goes through PyTorch APIs (carveout read, device queries,
stream wrap and sync) except masked-stream creation/destruction: PyTorch
exposes no Python API for a CU-masked stream (its own ROCm carveout stream
is a static C++ helper), so those two calls use the HIP library handle
PyTorch itself loads.
"""

import atexit
import contextlib
import ctypes
import functools

import torch

from ._catalog import InvalidInput

# (device_index, holdout, total_cus) -> (torch stream, raw handle, HIP lib).
_MASKED_STREAMS = {}


def _torch_carveout():
    get = getattr(torch._C, "_get_sm_carveout_experimental", None)
    if get is None:
        return None
    value = get()
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidInput(f"torch SM carveout must be a non-negative int; got {value!r}")
    return value


def holdout_sms() -> int:
    """SMs/CUs reserved for concurrent work outside this launch."""
    return _torch_carveout() or 0


def _green_driver():
    """CUDA driver bindings via PyTorch, or None when unavailable."""
    try:
        from torch.cuda._utils import _cuda_bindings_driver

        return _cuda_bindings_driver
    except Exception:
        return None


def _stream_green_sms(stream) -> int | None:
    """Green-context SM count for a stream, or None if it is not green."""
    drv = _green_driver()
    get_green = getattr(drv, "cuStreamGetGreenCtx", None) if drv is not None else None
    if get_green is None:
        return None
    try:
        from torch.cuda._utils import _check_cuda_bindings

        green = _check_cuda_bindings(get_green(stream.cuda_stream))
        resource = _check_cuda_bindings(
            drv.cuGreenCtxGetDevResource(green, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
        )
        return int(resource.sm.smCount) or None
    except Exception:
        return None


def green_sm_count(device=None) -> int | None:
    """SM count of the current stream's green context, or None.

    None means no green partition constrains this launch: not on CUDA, the
    current stream is a regular PyTorch stream, or the driver query is
    unavailable. Only green-context streams (ExternalStream) pay the query.
    """
    if torch.version.hip is not None or torch.version.cuda is None:
        return None
    try:
        stream = torch.cuda.current_stream(device)
    except Exception:
        return None
    if not isinstance(stream, torch.cuda.ExternalStream):
        return None
    return _stream_green_sms(stream)


def cap_grid(grid: int) -> int:
    """Withhold held-out SMs from a persistent grid, floored at one program.

    On a green-context stream the grid also fits the partition: the
    partition size replaces the physical device as the launch ceiling.
    """
    capped = max(1, grid - holdout_sms())
    green = green_sm_count()
    return min(capped, green) if green is not None else capped


def cu_mask_words(holdout: int, total_cus: int) -> tuple[int, ...]:
    """CU mask words enabling exactly the first ``holdout`` CUs.

    Per the HIP docs, an active bit uses its CU and low bits map to low CUs,
    so this is the mask for running a kernel on the holdout set. It
    deliberately does not mirror PyTorch's own HIP carveout mask, whose
    construction currently reads as use-N (see holdout_notes.md).
    """
    words = [0] * ((total_cus + 31) // 32)
    for cu in range(min(holdout, total_cus)):
        words[cu // 32] |= 1 << (cu % 32)
    return tuple(words)


def cu_mask_drop_first(holdout: int, total_cus: int) -> tuple[int, ...]:
    """CU mask words dropping the first ``holdout`` CUs of the part."""
    words = [0] * ((total_cus + 31) // 32)
    for cu in range(min(holdout, total_cus), total_cus):
        words[cu // 32] |= 1 << (cu % 32)
    return tuple(words)


@functools.lru_cache(maxsize=1)
def _hip_lib():
    """HIP runtime loaded through PyTorch, or None when unavailable.

    Cached: the loaded library never changes in-process, so the lookup and
    the ctypes signatures are set up once, not per masked-stream cache miss.
    """
    try:
        from torch.cuda._utils import _get_hip_runtime_library

        lib = _get_hip_runtime_library()
    except Exception:
        return None
    create = getattr(lib, "hipExtStreamCreateWithCUMask", None)
    destroy = getattr(lib, "hipStreamDestroy", None)
    if create is None or destroy is None:
        return None
    create.restype = ctypes.c_int32
    create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    destroy.restype = ctypes.c_int32
    destroy.argtypes = [ctypes.c_void_p]
    return lib


def _destroy_masked_streams():
    for _stream, handle, lib in list(_MASKED_STREAMS.values()):
        try:
            lib.hipStreamDestroy(ctypes.c_void_p(handle))
        except Exception:
            # Teardown must not raise: FFI failures (e.g. ctypes.ArgumentError)
            # during interpreter shutdown would surface as unraisable.
            pass
    _MASKED_STREAMS.clear()


atexit.register(_destroy_masked_streams)


def _masked_stream(device_index, holdout, total_cus):
    """CU-masked torch stream dropping the first ``holdout`` CUs, or None."""
    key = (device_index, holdout, total_cus)
    cached = _MASKED_STREAMS.get(key)
    if cached is not None:
        return cached[0]
    lib = _hip_lib()
    if lib is None:
        return None
    words = cu_mask_drop_first(holdout, total_cus)
    arr = (ctypes.c_uint32 * len(words))(*words)
    handle = ctypes.c_void_p(None)
    with torch.cuda.device(device_index):
        ok = lib.hipExtStreamCreateWithCUMask(ctypes.byref(handle), len(words), arr) == 0 and handle.value
    if not ok:
        return None
    stream = torch.cuda.ExternalStream(handle.value, device_index)
    _MASKED_STREAMS[key] = (stream, handle.value, lib)
    return stream


@contextlib.contextmanager
def carveout_stream(device=None):
    """Run a launch dropping the first-K-CU holdout when carved out.

    No-op (yields immediately) when the carveout is 0, the build is not HIP,
    or no masked stream can be created. Otherwise the body launches on a
    CU-masked stream with pre/post event sync against the ambient stream.

    TODO: generalize beyond drop-first-K to arbitrary masks selectable per
    stream (holdout set vs complement, XCD-aligned subsets). A mask restricts
    *which* CUs run blocks, not how many, so each selection still needs a
    distribution rule for spreading consecutive programs across allowed CUs.
    """
    holdout = holdout_sms()
    if holdout == 0 or torch.version.hip is None or not torch.cuda.is_available():
        yield
        return
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).multi_processor_count
    masked = _masked_stream(idx, min(holdout, total - 1), total) if total > 0 else None
    if masked is None:
        yield
        return
    current = torch.cuda.current_stream(dev)
    if current.cuda_stream == masked.cuda_stream:
        # Already on the CU-masked stream (e.g. a nested carveout_stream):
        # work is already ordered, so no stream switch or sync is needed.
        yield
        return
    # Order the masked stream after the ambient stream via PyTorch's
    # stream-sync helper (records an event on `current`, `masked` waits).
    masked.wait_stream(current)
    try:
        with torch.cuda.stream(masked):
            yield
    finally:
        # Always sync the ambient stream against masked-stream work, even if
        # the body raised after enqueueing.
        current.wait_stream(masked)
