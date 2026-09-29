from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator, Sequence
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
