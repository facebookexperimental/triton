"""HIP inter-process memory for the all-reduce kernels.

Each rank allocates one buffer with ``hipExtMallocWithFlags``, exports it with ``hipIpcGetMemHandle``, and opens the
other ranks' buffers with ``hipIpcOpenMemHandle``. Buffers that carry cross-GPU flags or are read by a peer after a
flag are allocated uncached (``hipDeviceMallocUncached``), so a poll never hits a stale L2 line.
"""

from __future__ import annotations

import ctypes
import functools

import torch
import torch.distributed as dist

HIP_DEVICE_MALLOC_DEFAULT = 0x0
HIP_DEVICE_MALLOC_UNCACHED = 0x3
_HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
_HIP_IPC_HANDLE_BYTES = 64


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * _HIP_IPC_HANDLE_BYTES)]


def _loaded_hip_path():
    """The HIP runtime torch already loaded; a second copy would have its own device context."""
    torch.cuda.init()
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = line.split()[-1]
            if "libamdhip64.so" in path:
                return path
    return "libamdhip64.so"


@functools.cache
def _hip():
    lib = ctypes.CDLL(_loaded_hip_path())
    lib.hipExtMallocWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
    lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    lib.hipFree.argtypes = [ctypes.c_void_p]
    lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.hipDeviceSynchronize.argtypes = []
    for name in ("hipExtMallocWithFlags", "hipIpcGetMemHandle", "hipIpcOpenMemHandle", "hipIpcCloseMemHandle",
                 "hipFree", "hipMemset", "hipDeviceSynchronize"):
        getattr(lib, name).restype = ctypes.c_int
    return lib


def _check(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed with hipError {err}")


class DevicePtr:
    """A raw device address that Triton accepts as a pointer argument."""

    def __init__(self, ptr: int, dtype: torch.dtype):
        self.ptr = int(ptr)
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self.ptr


class IpcBuffer:
    """One buffer per rank, mapped into every rank's address space.

    ``ptrs[r]`` is rank ``r``'s buffer as seen from this rank; ``ptrs[rank]`` is the local allocation. Collective over
    ``group``: every rank must construct it with the same ``nbytes`` and ``flags``.
    """

    def __init__(self, nbytes: int, *, rank: int, world_size: int, group, flags: int = HIP_DEVICE_MALLOC_UNCACHED):
        hip = _hip()
        self.nbytes = nbytes
        self.rank = rank
        local = ctypes.c_void_p()
        _check(hip.hipExtMallocWithFlags(ctypes.byref(local), nbytes, flags), "hipExtMallocWithFlags")
        _check(hip.hipMemset(local, 0, nbytes), "hipMemset")
        _check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        handle = _IpcHandle()
        _check(hip.hipIpcGetMemHandle(ctypes.byref(handle), local), "hipIpcGetMemHandle")
        handles = [None] * world_size
        dist.all_gather_object(handles, ctypes.string_at(ctypes.addressof(handle), _HIP_IPC_HANDLE_BYTES), group=group)
        self._opened = []
        self.ptrs = []
        for r, raw in enumerate(handles):
            if r == rank:
                self.ptrs.append(local.value)
                continue
            peer = ctypes.c_void_p()
            h = _IpcHandle()
            ctypes.memmove(ctypes.byref(h), raw, _HIP_IPC_HANDLE_BYTES)
            _check(hip.hipIpcOpenMemHandle(ctypes.byref(peer), h, _HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
                   "hipIpcOpenMemHandle")
            self._opened.append(peer.value)
            self.ptrs.append(peer.value)
        self._local = local.value

    def close(self):
        hip = _hip()
        for p in self._opened:
            hip.hipIpcCloseMemHandle(ctypes.c_void_p(p))
        self._opened = []
        if self._local is not None:
            hip.hipFree(ctypes.c_void_p(self._local))
            self._local = None

    def local_tensor(self, numel: int, dtype: torch.dtype, offset_bytes: int = 0) -> torch.Tensor:
        """A torch view of the local allocation (no copy, not owned by the caching allocator)."""
        elem = torch.empty((), dtype=dtype).element_size()
        if numel < 0 or offset_bytes < 0 or offset_bytes + numel * elem > self.nbytes:
            raise ValueError(f"local_tensor {numel}x{dtype} @ {offset_bytes} exceeds the {self.nbytes}-byte buffer")
        return _wrap(self.ptrs[self.rank] + offset_bytes, numel, dtype, torch.cuda.current_device())


def _wrap(ptr: int, numel: int, dtype: torch.dtype, device: int) -> torch.Tensor:
    elem = torch.empty((), dtype=dtype).element_size()

    class _Arr:
        __cuda_array_interface__ = {
            "shape": (numel, ),
            "typestr": {1: "|u1", 2: "<u2", 4: "<u4", 8: "<u8"}[elem],
            "data": (ptr, False),
            "version": 3,
        }

    return torch.as_tensor(_Arr(), device=torch.device("cuda", device)).view(dtype)
