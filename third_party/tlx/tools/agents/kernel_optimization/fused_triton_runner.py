#!/usr/bin/env python3
"""Run a generated fused-kernel comparison with a selected Triton checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _load_benchmark(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_fused_triton_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated benchmark from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _time_us(
    torch: Any,
    call: Callable[[], Any],
    warmup_ms: int,
    benchmark_ms: int,
) -> dict[str, Any]:
    """Use Triton's duration-based benchmark and retain every device sample."""
    del torch
    import triton

    times_ms = triton.testing.do_bench(
        call,
        warmup=warmup_ms,
        rep=benchmark_ms,
        return_mode="all",
    )
    measurements = [float(sample) * 1000.0 for sample in times_ms]
    if not measurements:
        raise RuntimeError("triton.testing.do_bench returned no timing samples")
    return {
        "latency_us": statistics.median(measurements),
        "min_us": min(measurements),
        "max_us": max(measurements),
        "samples": len(measurements),
        "samples_us": measurements,
    }


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--triton-dir", type=Path, required=True)
    parser.add_argument("--fbsource-root", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmup-ms", type=int, required=True)
    parser.add_argument("--benchmark-ms", type=int, required=True)
    return parser.parse_args(arguments)


def _install_generated_wrapper_compat(torch: Any) -> None:
    """Bridge small fbcode/generated-wrapper API differences in OSS PyTorch."""
    guards = torch._C._dynamo.guards
    if not hasattr(guards, "copy_if_misaligned") and hasattr(guards, "copy_misaligned"):
        guards.copy_if_misaligned = guards.copy_misaligned
    if not hasattr(guards, "assert_size_stride_grouped"):

        def assert_size_stride_grouped(
            tensors: tuple[Any, ...],
            sizes: tuple[tuple[int, ...], ...],
            strides: tuple[tuple[int, ...], ...],
            *labels: object,
        ) -> None:
            for tensor, size, stride in zip(tensors, sizes, strides):
                guards.assert_size_stride(tensor, size, stride)

        guards.assert_size_stride_grouped = assert_size_stride_grouped
    if not hasattr(guards, "reinterpret_tensor"):

        def reinterpret_tensor(
            tensor: Any,
            size: tuple[int, ...],
            stride: tuple[int, ...],
            offset_increment: int,
        ) -> Any:
            return torch.as_strided(
                tensor,
                size,
                stride,
                tensor.storage_offset() + offset_increment,
            )

        guards.reinterpret_tensor = reinterpret_tensor


def main(arguments: list[str] | None = None) -> int:
    args = _parse_args(arguments)
    expected_package = (args.triton_dir / "python/triton").resolve()
    try:
        import torch
        import triton
    except ImportError as error:
        raise SystemExit(
            "the selected Python environment must provide torch and the selected "
            f"Triton checkout: {error}"
        ) from error
    loaded_package = Path(triton.__file__).resolve().parent
    if loaded_package != expected_package:
        raise SystemExit(
            f"selected {args.triton_dir}, but Python imported Triton from "
            f"{loaded_package}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("the selected Python environment has no available CUDA GPU")
    _install_generated_wrapper_compat(torch)

    benchmark = _load_benchmark(args.benchmark)
    benchmark.time_us = lambda torch, call, _warmup, _samples: _time_us(
        torch,
        call,
        args.warmup_ms,
        args.benchmark_ms,
    )
    result = benchmark.main(
        [
            "--fbsource-root",
            str(args.fbsource_root),
            "--fused",
            str(args.fused),
            "--original",
            str(args.original),
            "--json",
            str(args.json),
            # The generated benchmark requires these count-shaped arguments,
            # but the override above uses duration-based do_bench settings.
            "--warmup",
            "1",
            "--samples",
            "1",
        ]
    )
    if result == 0:
        payload = json.loads(args.json.read_text())
        payload["reference"] = payload.pop("geo_tlx")
        payload["triton_package"] = str(loaded_package)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
