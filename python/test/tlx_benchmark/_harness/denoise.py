"""Environment verification and capture.

``third_party/tlx/denoise.sh`` remains the thing that *applies* the lock. This
module deliberately does not re-implement it in Python: duplicating privileged
shell logic would give two sources of truth for what "denoised" means, and the
copy would drift.

What it does is verify the environment, watch the GPU's operating point while a
case is measured, and record all of it into the artifact -- a latency reported
without its environment is not comparable to anything, and phase 1 measured how
large that difference is (13.9% vs 1.7% across-run spread on one kernel).

**Do not check for a locked SM clock.** Measured on B200: under a sustained
8192x8192x8192 fp16 GEMM the card runs at ~830 MHz and reports ``sw_power_cap``
whether or not ``nvidia-smi -lgc 1965`` has been applied -- the operating point
is set by the power budget long before the clock cap binds, so ``-lgc`` is
close to a no-op for compute-bound work. A "clocks are not locked" check would
have fired on every correct run. What ``denoise.sh`` actually contributes for
this workload is a *fixed* power cap, persistence mode, and NUMA binding; the
rest of the stability comes from warming to steady state (see
``measure.DEFAULT_WARMUP_MS``).

So verification here is behavioural: watch the operating point rather than the
configuration that was meant to produce it.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import gc
import os
import re
import shutil
import socket
import statistics
import subprocess
import threading
import time
from typing import Optional

from .measure import relative_interdecile_range

#: nvml.h clock event ("throttle") reason bits.
EVENT_REASONS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}

#: Reasons that mean the card was degraded, as opposed to merely governed.
#: ``sw_power_cap`` is excluded on purpose: on B200 a compute-bound GEMM sits
#: against the power cap continuously, denoised or not, so counting it would
#: flag every healthy run. ``gpu_idle`` is excluded because it says nothing
#: about the measured window.
DEGRADING_REASONS = 0x8 | 0x20 | 0x40 | 0x80

#: Relative interdecile range of the SM clock across a measured window, above
#: which the operating point moved enough to be worth reporting. The trace
#: covers warmup as well as measurement, so some ramp is expected and this is
#: deliberately loose -- it is a diagnostic, not the gate. The gate is the
#: latency's between-run figure against ``measure.MAX_REPLICATE_DEVIATION``.
MAX_CLOCK_IDR = 0.10

#: Interval between operating-point samples. NVML costs ~16us per sample, so
#: 20 Hz is free even next to a host-bound case.
SAMPLE_INTERVAL_S = 0.05


def decode_event_reasons(mask: int) -> list[str]:
    return sorted(name for bit, name in EVENT_REASONS.items() if mask & bit)


# --------------------------------------------------------------------------
# NVML, via ctypes.
#
# nvidia-smi is fine for one-shot inventory but forks a process per reading,
# which is far too expensive to run inside a measurement window. NVML is
# already on any machine with a driver, and pynvml is not a dependency we want.
# --------------------------------------------------------------------------

_NVML_CLOCK_SM = 1


class _Nvml:

    def __init__(self):
        self.lib = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError:
            return
        if lib.nvmlInit_v2() != 0:
            return
        self.lib = lib

    def handle(self, uuid: Optional[str]):
        """Resolve by UUID, never by index: NVML indexes physical devices while
        torch indexes whatever ``CUDA_VISIBLE_DEVICES`` left visible, so the two
        numbering schemes routinely disagree."""
        if self.lib is None or not uuid:
            return None
        h = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(h)) != 0:
            return None
        return h

    def sample(self, handle) -> Optional[tuple[int, int]]:
        """``(SM clock MHz, active event-reason mask)``."""
        if self.lib is None or handle is None:
            return None
        sm, reasons = ctypes.c_uint(), ctypes.c_ulonglong()
        if self.lib.nvmlDeviceGetClockInfo(handle, _NVML_CLOCK_SM, ctypes.byref(sm)) != 0:
            return None
        if self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(handle, ctypes.byref(reasons)) != 0:
            return None
        return sm.value, reasons.value


_nvml: Optional[_Nvml] = None


def nvml() -> _Nvml:
    global _nvml
    if _nvml is None:
        _nvml = _Nvml()
    return _nvml


# --------------------------------------------------------------------------
# One-shot inventory, via nvidia-smi.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GpuState:
    """A point-in-time reading. Every field is optional so a machine without
    ``nvidia-smi`` degrades to "unknown" rather than crashing."""

    available: bool
    index: Optional[int] = None
    uuid: Optional[str] = None
    name: Optional[str] = None
    driver: Optional[str] = None
    sm_clock_mhz: Optional[float] = None
    max_sm_clock_mhz: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    max_power_limit_w: Optional[float] = None
    temperature_c: Optional[float] = None
    persistence_mode: Optional[str] = None
    event_reasons: Optional[int] = None

    @property
    def event_reason_names(self) -> list[str]:
        return decode_event_reasons(self.event_reasons) if self.event_reasons is not None else []

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["event_reason_names"] = self.event_reason_names
        return d


@dataclasses.dataclass(frozen=True)
class ClockTrace:
    """The GPU's operating point over a measured window.

    This is the honest replacement for a "are the clocks locked?" question: it
    reports where the card actually ran and whether it stayed there.
    """

    samples: int
    min_mhz: Optional[int] = None
    median_mhz: Optional[int] = None
    max_mhz: Optional[int] = None
    #: Relative interdecile range, ``(p90 - p10) / median``, not ``(max - min)``.
    #:
    #: The window necessarily contains the ramp from idle -- measured on B200,
    #: the card sits at 990 MHz until the first launch and settles to ~830 under
    #: load -- so a min/max spread is ~25% on every run, healthy or not, and
    #: carries no information. Quantiles ignore the handful of ramp samples for
    #: the same reason the latency path rejects outliers.
    rel_idr: Optional[float] = None
    #: Union of every reason seen, and the degrading subset of it.
    reasons: tuple = ()
    degrading: tuple = ()

    @property
    def stable(self) -> Optional[bool]:
        if self.rel_idr is None:
            return None
        return self.rel_idr <= MAX_CLOCK_IDR and not self.degrading

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["reasons"] = list(self.reasons)
        d["degrading"] = list(self.degrading)
        d["stable"] = self.stable
        return d


_QUERY = ("index,uuid,name,driver_version,clocks.current.sm,clocks.max.sm,power.draw,power.limit,"
          "power.max_limit,temperature.gpu,persistence_mode,clocks_event_reasons.active")
_QUERY_FIELDS = 12


def _smi(args: list[str]) -> Optional[str]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def _num(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None  # nvidia-smi prints "[N/A]" for unsupported fields


def gpu_uuid(device: int = 0) -> Optional[str]:
    """The physical UUID of a torch device index.

    Indices are not usable for talking to NVML or ``nvidia-smi``: torch sees
    indices remapped by ``CUDA_VISIBLE_DEVICES``, so torch device 0 is routinely
    a different card than physical device 0. The UUID is stable across all three.
    """
    try:
        import torch

        raw = torch.cuda.get_device_properties(device).uuid
    except Exception:
        return None
    # torch returns a ``_CUuuid`` whose str() is the bare hyphenated UUID;
    # NVML and nvidia-smi both prefix it with "GPU-".
    text = str(raw).strip()
    return text if text.startswith("GPU-") else f"GPU-{text}"


def gpu_state(uuid: Optional[str] = None, device: int = 0) -> GpuState:
    """Read the current state of the GPU this process is using."""
    uuid = uuid or gpu_uuid(device)
    out = _smi([f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"])
    if out is None:
        return GpuState(available=False)
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) != _QUERY_FIELDS or (uuid is not None and f[1] != uuid):
            continue
        return GpuState(
            available=True,
            index=int(f[0]),
            uuid=f[1],
            name=f[2],
            driver=f[3],
            sm_clock_mhz=_num(f[4]),
            max_sm_clock_mhz=_num(f[5]),
            power_draw_w=_num(f[6]),
            power_limit_w=_num(f[7]),
            max_power_limit_w=_num(f[8]),
            temperature_c=_num(f[9]),
            persistence_mode=f[10],
            event_reasons=int(f[11], 16) if f[11].startswith("0x") else None,
        )
    return GpuState(available=False)


def foreign_processes(uuid: Optional[str] = None, device: int = 0) -> list[dict]:
    """Other processes holding memory on our GPU.

    A co-tenant makes every number on the card meaningless, and on a shared dev
    box it is the failure mode most likely to go unnoticed.
    """
    uuid = uuid or gpu_uuid(device)
    out = _smi(["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
    if not out:
        return []
    mine = os.getpid()
    found = []
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) != 3 or (uuid is not None and f[0] != uuid):
            continue
        if int(f[1]) != mine:
            found.append({"pid": int(f[1]), "used_mib": _num(f[2])})
    return found


# --------------------------------------------------------------------------
# NUMA.
# --------------------------------------------------------------------------


def parse_cpulist(text: str) -> set[int]:
    """Parse a sysfs cpulist such as ``0-95,192-287``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def numa_node(uuid: Optional[str] = None, device: int = 0) -> Optional[int]:
    """The NUMA node local to our GPU, via sysfs."""
    uuid = uuid or gpu_uuid(device)
    out = _smi(["--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"])
    if not out:
        return None
    bus = None
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 2 and (uuid is None or f[0] == uuid):
            bus = f[1].lower()
            break
    if bus is None:
        return None
    # nvidia-smi prints an eight-digit PCI domain; sysfs uses four.
    bus = re.sub(r"^[0-9a-f]{4}([0-9a-f]{4}:)", r"\1", bus)
    try:
        with open(f"/sys/bus/pci/devices/{bus}/numa_node") as fh:
            node = int(fh.read().strip())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def numa_bound(node: Optional[int]) -> Optional[bool]:
    """Whether this process is confined to the GPU-local node's CPUs.

    This is also the most reliable signal that ``denoise.sh`` wrapped the run,
    since it is the one effect that is unambiguous and machine-independent.
    """
    if node is None:
        return None
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as fh:
            local = parse_cpulist(fh.read())
        affinity = os.sched_getaffinity(0)
    except (OSError, ValueError, AttributeError):
        return None
    return bool(affinity) and affinity <= local


# --------------------------------------------------------------------------
# Capture, check, and the measurement context.
# --------------------------------------------------------------------------


def capture_env(device: int = 0) -> dict:
    """The ``env`` block of the artifact: everything the numbers depend on."""
    import torch

    import triton

    uuid = gpu_uuid(device)
    state = gpu_state(uuid, device)
    node = numa_node(uuid, device)
    return {
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": state.to_dict(),
        "numa_node": node,
        "numa_bound": numa_bound(node),
        "foreign_processes": foreign_processes(uuid, device),
        "nvml": nvml().lib is not None,
    }


def check(device: int = 0) -> list[str]:
    """Everything wrong with the environment, in plain language.

    Empty means the numbers are trustworthy. Each entry names the fix, because
    whoever reads it is usually not whoever wrote it.
    """
    problems = []
    uuid = gpu_uuid(device)
    state = gpu_state(uuid, device)
    if not state.available:
        return ["nvidia-smi unavailable: cannot verify NUMA, tenancy, or the GPU operating point"]

    others = foreign_processes(uuid, device)
    if others:
        who = ", ".join(f"pid {p['pid']} ({p['used_mib']:.0f} MiB)" for p in others)
        problems.append(f"{len(others)} other process(es) on this GPU: {who}")

    node = numa_node(uuid, device)
    if numa_bound(node) is False:
        problems.append(f"CPU affinity is not confined to GPU-local NUMA node {node} -- "
                        f"run under third_party/tlx/denoise.sh")

    if state.persistence_mode and state.persistence_mode.lower() != "enabled":
        problems.append("persistence mode is disabled: driver teardown between launches adds latency -- "
                        "run under third_party/tlx/denoise.sh")

    degraded = (state.event_reasons or 0) & DEGRADING_REASONS
    if degraded:
        problems.append(f"GPU is already degraded before measuring: {', '.join(decode_event_reasons(degraded))}")

    if not nvml().lib:
        problems.append("NVML unavailable: the operating point cannot be watched during measurement")
    return problems


class _Sampler(threading.Thread):
    """Watches the operating point in the background at ~20 Hz."""

    def __init__(self, uuid: Optional[str]):
        # Three names here belong to Thread and must not be reused: ``daemon``
        # (set via the constructor), ``_handle`` (hence ``_nvml_handle``), and
        # ``_stop``. The last one is not a flag but a *method* CPython calls
        # from inside ``join()``; shadowing it with an Event made every
        # successful join raise ``TypeError: 'Event' object is not callable``,
        # which discarded a whole run's results at the very end.
        super().__init__(daemon=True)
        self._nvml_handle = nvml().handle(uuid)
        self._stopped = threading.Event()
        self.clocks: list[int] = []
        self.reasons = 0

    def run(self):
        while not self._stopped.is_set():
            got = nvml().sample(self._nvml_handle)
            if got is not None:
                self.clocks.append(got[0])
                self.reasons |= got[1]
            self._stopped.wait(SAMPLE_INTERVAL_S)

    def finish(self) -> ClockTrace:
        self._stopped.set()
        self.join(timeout=5)
        if not self.clocks:
            return ClockTrace(samples=0)
        med = int(statistics.median(self.clocks))
        return ClockTrace(
            samples=len(self.clocks),
            min_mhz=min(self.clocks),
            median_mhz=med,
            max_mhz=max(self.clocks),
            rel_idr=relative_interdecile_range(self.clocks, med),
            reasons=tuple(decode_event_reasons(self.reasons)),
            degrading=tuple(decode_event_reasons(self.reasons & DEGRADING_REASONS)),
        )


@contextlib.contextmanager
def stable(device: int = 0, strict: bool = False, watch: bool = True):
    """Hold the process still, and watch what the GPU does meanwhile.

    Freezes the GC so a collection cannot land inside a measurement window, and
    samples the SM clock and event reasons throughout, so the artifact records
    where the card actually ran rather than where it was configured to run.

    Yields a dict that gains ``clock_trace`` and ``elapsed_s`` on exit.
    ``strict`` turns environment problems into an error instead of a warning.
    """
    import warnings

    problems = check(device)
    if problems:
        message = "environment is not denoised:\n  - " + "\n  - ".join(problems)
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)

    uuid = gpu_uuid(device)
    info = {"problems": problems, "gpu_before": gpu_state(uuid, device).to_dict()}

    sampler = _Sampler(uuid) if watch else None
    gc.collect()
    gc.freeze()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    started = time.perf_counter()
    if sampler is not None:
        sampler.start()
    try:
        yield info
    finally:
        info["elapsed_s"] = time.perf_counter() - started
        info["clock_trace"] = (sampler.finish() if sampler is not None else ClockTrace(samples=0)).to_dict()
        if gc_was_enabled:
            gc.enable()
        gc.unfreeze()
        info["gpu_after"] = gpu_state(uuid, device).to_dict()


# --------------------------------------------------------------------------
# Device selection and governing.
#
# This duplicates what ``third_party/tlx/denoise.sh`` does, so that the suite
# is one command with no wrapper. The cost is two definitions of "denoised"
# that can drift; the mitigation is that the verification above is behavioural
# and will notice if this stops working.
#
# The AMD path mirrors denoise.sh's rocm-smi handling and has NOT been run on
# hardware -- there is no AMD card on the box this was written on. It degrades
# to "governed=False" rather than failing, and ``check()`` still reports the
# environment truthfully either way.
# --------------------------------------------------------------------------

NVIDIA, AMD = "nvidia", "amd"

#: Sustained power target per part, in watts. Same values as denoise.sh. The
#: point is a *fixed* cap rather than a high one: an unpinned cap is one more
#: thing that can differ between two runs being compared.
POWER_TARGET_W = {
    "H100": 700,
    "GB200": 1200,
    "GB300": 1400,
    "B200": 750,
    "MI300X": 750,
    "MI355X": 1400,
    "MI350X": 1000,
}

#: gfx clock to pin on AMD via perf-determinism. CDNA4 does not support
#: ``--setperflevel high`` (rocm-smi returns "Not supported"), so that silently
#: did nothing and determinism is the working knob.
AMD_DETERMINISM_MHZ = 2100

#: Part name -> ``tlx.ops`` catalog arch key.
#:
#: Resolved from the part rather than from torch because the arch is needed
#: *before* the first CUDA call: the visibility variable that pins the device is
#: read once at context creation, so anything that initializes a context first
#: silently defeats ``--device``. ``list_devices`` is smi-based and does not.
#:
#: Longer names first, as in ``POWER_TARGET_W``: the lookup is a substring test,
#: so "B200" would otherwise swallow "GB200".
ARCH_BY_PART = {
    "GB200": "sm100",
    "GB300": "sm100",
    "B200": "sm100",
    "MI300X": "gfx942",
}


@dataclasses.dataclass(frozen=True)
class Device:
    vendor: str
    index: int
    name: str
    uuid: Optional[str] = None
    memory_used_mib: float = 0.0

    @property
    def visibility_env(self) -> str:
        return "CUDA_VISIBLE_DEVICES" if self.vendor == NVIDIA else "HIP_VISIBLE_DEVICES"

    @property
    def power_target_w(self) -> Optional[int]:
        for part, watts in POWER_TARGET_W.items():
            if part.lower() in self.name.lower().replace(" ", ""):
                return watts
        return None

    @property
    def arch(self) -> Optional[str]:
        """``tlx.ops`` catalog key for this part, or None if it has no entry."""
        for part, key in ARCH_BY_PART.items():
            if part.lower() in self.name.lower().replace(" ", ""):
                return key
        return None


def _rocm(args: list[str]) -> Optional[str]:
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.run(["rocm-smi", *args], capture_output=True, text=True, timeout=30, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def vendor() -> Optional[str]:
    if _smi(["--query-gpu=index", "--format=csv,noheader"]) is not None:
        return NVIDIA
    if _rocm(["--showid"]) is not None:
        return AMD
    return None


def list_devices() -> list[Device]:
    """Every GPU on the box, with how much memory is already in use."""
    which = vendor()
    if which == NVIDIA:
        out = _smi(["--query-gpu=index,name,uuid,memory.used", "--format=csv,noheader,nounits"]) or ""
        found = []
        for line in out.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) == 4:
                found.append(Device(NVIDIA, int(f[0]), f[1], f[2], _num(f[3]) or 0.0))
        return found
    if which == AMD:
        out = _rocm(["--showmeminfo", "vram", "--csv"]) or ""
        found = []
        for line in out.splitlines()[1:]:
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 3 and f[0].lower().startswith("card"):
                used = _num(f[2])
                idx = "".join(c for c in f[0] if c.isdigit())
                if idx:
                    found.append(Device(AMD, int(idx), _amd_name(int(idx)), None, (used or 0.0) / 1024 / 1024))
        return found
    return []


def _amd_name(index: int) -> str:
    """Identify by device id and GFX version, not by "Card Series".

    On CDNA data-center parts the series string is often just "AMD Radeon
    Graphics", which distinguishes nothing.
    """
    info = _rocm(["-d", str(index), "--showproductname"]) or ""
    blob = info.lower()
    for marker, part in (("0x74a0", "MI300X"), ("0x74a1", "MI300X"), ("mi300", "MI300X"), ("gfx942", "MI300X"),
                         ("0x75a1", "MI355X"), ("0x75a3", "MI355X"), ("mi355", "MI355X"), ("0x75a0", "MI350X"),
                         ("mi350", "MI350X"), ("gfx950", "MI350X")):
        if marker in blob:
            return part
    return f"AMD GPU {index}"


def select_device(spec: str = "auto") -> Optional[Device]:
    """Resolve ``--device``. ``auto`` picks the least-used GPU.

    By free memory rather than by index: on a shared box a co-tenant is the
    failure mode most likely to go unnoticed, and index 0 is as likely to be
    busy as any other.
    """
    devices = list_devices()
    if not devices:
        return None
    if spec != "auto":
        wanted = int(spec)
        for d in devices:
            if d.index == wanted:
                return d
        raise ValueError(f"--device {spec}: no such GPU (have {[d.index for d in devices]})")
    return min(devices, key=lambda d: d.memory_used_mib)


#: Where sysfs enumerates PCI devices. A module constant so the resolution above
#: can be tested without a GPU.
_PCI_DEVICES = "/sys/bus/pci/devices"


def _amd_numa_node(index: int) -> Optional[int]:
    """GPU-local NUMA node for a *rocm-smi* device index.

    Resolved through the PCI bus id, not through ``/sys/class/drm/card<index>``.
    Those two indices are different numbering schemes and only coincide at 0: on
    an 8-GPU MI300X box rocm-smi's card6 is ``0000:c8:00.0``, which sysfs
    enumerates as ``card40``. Reading ``card6`` there finds a node with no
    ``numa_node`` at all, so binding was silently skipped for every device but
    the first -- and NUMA binding is the one lever here that needs no privilege
    and was measured to matter.

    This mirrors the UUID indirection the NVIDIA path already needs, for the
    same underlying reason: a device index is only meaningful to the tool that
    issued it.
    """
    bus = _rocm(["-d", str(index), "--showbus", "--csv"]) or ""
    for line in bus.splitlines():
        fields = [f.strip() for f in line.split(",")]
        # e.g. "card6,0000:C8:00.0"; sysfs spells the domain in lowercase.
        if len(fields) >= 2 and fields[0].lower() == f"card{index}":
            try:
                with open(f"{_PCI_DEVICES}/{fields[1].lower()}/numa_node") as fh:
                    node = int(fh.read().strip())
            except (OSError, ValueError):
                return None
            return node if node >= 0 else None
    return None


class Governor(contextlib.AbstractContextManager):
    """Pin the machine's operating point for the duration of a run, and put it
    back afterwards.

    Everything is best-effort and each step is independent: without passwordless
    sudo the power and clock steps are skipped, but NUMA binding -- which needs
    no privilege and is the one lever measured to matter here -- still applies.
    A partially governed run is still worth taking, and ``check()`` reports what
    actually held rather than what was attempted.
    """

    def __init__(self, device: Optional[Device], enable: bool = True):
        self.device = device
        self.enable = enable
        self.applied: list[str] = []
        self.skipped: list[str] = []
        self._restore: list[list[str]] = []
        self._affinity: Optional[set] = None

    def _sudo(self, args: list[str]) -> bool:
        try:
            done = subprocess.run(["sudo", "-n", *args], capture_output=True, text=True, timeout=60)
        except (subprocess.SubprocessError, OSError):
            return False
        return done.returncode == 0

    def _govern_nvidia(self, d: Device):
        state = gpu_state(d.uuid)
        if self._sudo(["nvidia-smi", "-i", str(d.index), "-pm", "1"]):
            self.applied.append("persistence mode")
        else:
            self.skipped.append("persistence mode (needs passwordless sudo)")
            return  # no sudo: the rest will fail the same way

        target = d.power_target_w
        cap = state.max_power_limit_w
        if target and cap:
            watts = min(target, cap)
            if self._sudo(["nvidia-smi", "-i", str(d.index), f"--power-limit={watts}"]):
                self.applied.append(f"power cap {watts:.0f} W")
                if state.power_limit_w:
                    self._restore.append(["nvidia-smi", "-i", str(d.index), f"--power-limit={state.power_limit_w}"])

        if state.max_sm_clock_mhz and self._sudo(
            ["nvidia-smi", "-lgc", str(int(state.max_sm_clock_mhz)), "-i",
             str(d.index)]):
            # Measured on B200: this changes nothing for compute-bound work,
            # which is power-governed well below the cap. Applied anyway because
            # it does bind for smaller kernels, and recorded so the artifact is
            # honest about what was attempted.
            self.applied.append(f"clock lock {state.max_sm_clock_mhz:.0f} MHz")
            self._restore.append(["nvidia-smi", "-rgc", "-i", str(d.index)])

    def _govern_amd(self, d: Device):
        target = d.power_target_w or 500
        if self._sudo(["rocm-smi", "-d", str(d.index), "--setperfdeterminism", str(AMD_DETERMINISM_MHZ)]):
            self.applied.append(f"perf determinism {AMD_DETERMINISM_MHZ} MHz")
            self._restore.append(["rocm-smi", "-d", str(d.index), "--resetperfdeterminism"])
        else:
            self.skipped.append("perf determinism (needs passwordless sudo)")
        if self._sudo(["rocm-smi", "-d", str(d.index), "--setpoweroverdrive", str(target)]):
            self.applied.append(f"power cap {target} W")
            self._restore.append(["rocm-smi", "-d", str(d.index), "--resetpoweroverdrive"])

    def _bind_numa(self, d: Device):
        node = _amd_numa_node(d.index) if d.vendor == AMD else numa_node(d.uuid)
        if node is None:
            self.skipped.append("NUMA binding (GPU-local node unknown)")
            return
        try:
            with open(f"/sys/devices/system/node/node{node}/cpulist") as fh:
                local = parse_cpulist(fh.read())
            self._affinity = set(os.sched_getaffinity(0))
            os.sched_setaffinity(0, local)
        except (OSError, ValueError, AttributeError):
            self.skipped.append(f"NUMA binding to node {node}")
            return
        # CPU affinity only; memory follows by first-touch, which is enough
        # because every allocation here happens after this point.
        self.applied.append(f"NUMA node {node} ({len(local)} CPUs)")

    def __enter__(self):
        if not self.enable or self.device is None:
            self.skipped.append("all governing (disabled or no GPU)")
            return self
        if self.device.vendor == NVIDIA:
            self._govern_nvidia(self.device)
        else:
            self._govern_amd(self.device)
        self._bind_numa(self.device)
        return self

    def __exit__(self, *exc):
        for args in reversed(self._restore):
            self._sudo(args)
        if self._affinity:
            try:
                os.sched_setaffinity(0, self._affinity)
            except OSError:
                pass
        return False

    def to_dict(self) -> dict:
        return {"applied": list(self.applied), "skipped": list(self.skipped)}
