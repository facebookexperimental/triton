from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

_ARCH_ALIASES = {
    "blackwell": "sm100",
    "b200": "sm100",
    "gb200": "sm100",
    "gb300": "sm100",
    "hopper": "sm90",
    "h100": "sm90",
    "cdna3": "gfx942",
    "mi300x": "gfx942",
    "cdna4": "gfx950",
    "mi350x": "gfx950",
    "mi355x": "gfx950",
}


def canonical_arch(arch: str) -> str:
    return _ARCH_ALIASES.get(arch.lower(), arch.lower().replace("_", ""))


def _select_device(devices: Sequence[Any], spec: str, arch: str) -> Any:
    if not devices:
        raise SystemExit("no GPU was found")
    if spec != "auto":
        try:
            index = int(spec)
        except ValueError as error:
            raise SystemExit(f"--device must be 'auto' or a physical GPU index, got {spec!r}") from error
        for device in devices:
            if device.index == index:
                return device
        raise SystemExit(f"--device {spec}: no such GPU (have {[device.index for device in devices]})")

    expected = canonical_arch(arch)
    matching = [device for device in devices if device.arch and canonical_arch(device.arch) == expected]
    if matching:
        devices = matching
    elif any(device.arch for device in devices):
        available = sorted({device.arch for device in devices if device.arch})
        raise SystemExit(f"--arch {arch} has no matching GPU (available: {available})")
    return min(devices, key=lambda device: device.memory_used_mib)


def _load_benchmark_environment(repository: Path):
    benchmark_root = repository / "python" / "test" / "tlx_benchmark"
    if not benchmark_root.is_dir():
        raise SystemExit(f"TLX benchmark harness not found at {benchmark_root}")
    root = str(benchmark_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from _harness.denoise import Governor, list_devices

    return Governor, list_devices


def _matching_devices(devices: Sequence[Any], arch: str) -> list[Any]:
    expected = canonical_arch(arch)
    matching = [device for device in devices if device.arch and canonical_arch(device.arch) == expected]
    if matching:
        return sorted(matching, key=lambda device: (device.memory_used_mib, device.index))
    if any(device.arch for device in devices):
        available = sorted({device.arch for device in devices if device.arch})
        raise SystemExit(f"--arch {arch} has no matching GPU (available: {available})")
    return sorted(devices, key=lambda device: (device.memory_used_mib, device.index))


def _numa_cpus(device: Any) -> str:
    benchmark_root = Path(__file__).resolve().parents[6] / "python" / "test" / "tlx_benchmark"
    root = str(benchmark_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from _harness.denoise import AMD, _amd_numa_node, numa_node, parse_cpulist

    node = _amd_numa_node(device.index) if device.vendor == AMD else numa_node(device.uuid)
    if node is None:
        return ""
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as stream:
            return ",".join(map(str, sorted(parse_cpulist(stream.read()))))
    except OSError:
        return ""


@contextlib.contextmanager
def governed_benchmark_devices(
    repository: Path,
    arch: str,
    spec: str = "auto",
    *,
    govern: bool = True,
) -> Iterator[tuple[Any, ...]]:
    """Select and govern an exclusive pool of matching GPUs for tuning."""
    Governor, list_devices = _load_benchmark_environment(repository)
    devices = _matching_devices(list_devices(), arch)
    if not devices:
        raise SystemExit("no GPU was found")
    if spec != "auto":
        selected = _select_device(devices, spec, arch)
        devices = [selected]
    else:
        baseline = devices[0].memory_used_mib
        devices = [device for device in devices if device.memory_used_mib <= baseline + 1024]
    print(
        "[tlx-agent] device pool: "
        + ", ".join(f"gpu{device.index} {device.name} ({device.memory_used_mib:.0f} MiB in use)" for device in devices),
        file=sys.stderr,
        flush=True,
    )
    with ExitStack() as stack:
        for device in devices:
            governor = stack.enter_context(Governor(device, enable=govern))
            for step in governor.applied:
                print(f"[tlx-agent] gpu{device.index}: {step}", file=sys.stderr, flush=True)
            for step in governor.skipped:
                print(f"[tlx-agent] gpu{device.index}: SKIPPED {step}", file=sys.stderr, flush=True)
        yield tuple(devices)


def device_environment(device: Any) -> dict[str, str]:
    environment = {
        device.visibility_env: str(device.index),
        "TLX_AGENT_PHYSICAL_GPU": str(device.index),
    }
    cpus = _numa_cpus(device)
    if cpus:
        environment["TLX_AGENT_NUMA_CPUS"] = cpus
    return environment


@contextlib.contextmanager
def governed_benchmark_device(
    repository: Path,
    arch: str,
    spec: str = "auto",
    *,
    govern: bool = True,
) -> Iterator[Any]:
    """Select and govern a GPU before any framework initializes its context."""
    Governor, list_devices = _load_benchmark_environment(repository)
    device = _select_device(list_devices(), spec, arch)
    visibility_key = device.visibility_env
    previous_visibility = os.environ.get(visibility_key)
    os.environ[visibility_key] = str(device.index)
    selection = "least used" if spec == "auto" else "selected"
    print(
        f"[tlx-agent] device: gpu{device.index} {device.name} ({selection}, {device.memory_used_mib:.0f} MiB in use)",
        file=sys.stderr,
        flush=True,
    )
    try:
        with Governor(device, enable=govern) as governor:
            for step in governor.applied:
                print(f"[tlx-agent] denoise: {step}", file=sys.stderr, flush=True)
            for step in governor.skipped:
                print(f"[tlx-agent] denoise: SKIPPED {step}", file=sys.stderr, flush=True)
            yield device
    finally:
        if previous_visibility is None:
            os.environ.pop(visibility_key, None)
        else:
            os.environ[visibility_key] = previous_visibility
