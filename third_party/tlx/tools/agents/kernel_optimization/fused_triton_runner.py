#!/usr/bin/env python3
"""Run a generated fused-kernel comparison with a selected Triton checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
    parser.add_argument("--reference-config-json", type=Path, default=None)
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--save-output", type=Path, default=None)
    parser.add_argument("--compare-output", type=Path, default=None)
    parser.add_argument(
        "--only",
        choices=("both", "fused", "reference"),
        default="both",
        help="Run both benchmark legs or one leg for process-isolated autoWS timing.",
    )
    return parser.parse_args(arguments)


def _cpu_output(torch: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_cpu_output(torch, item) for item in value)
    if isinstance(value, list):
        return [_cpu_output(torch, item) for item in value]
    if isinstance(value, dict):
        return {key: _cpu_output(torch, item) for key, item in value.items()}
    raise TypeError(f"unsupported output leaf for precision comparison: {type(value)}")


def _precision_comparison(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    """Return direct per-output Triton-vs-TLX numerical differences."""
    rows: list[dict[str, Any]] = []

    def visit(lhs: Any, rhs: Any, path: str) -> None:
        if isinstance(lhs, (tuple, list)) and len(lhs) == 1 and isinstance(
            rhs, torch.Tensor
        ):
            visit(lhs[0], rhs, f"{path}[0]")
            return
        if isinstance(rhs, (tuple, list)) and len(rhs) == 1 and isinstance(
            lhs, torch.Tensor
        ):
            visit(lhs, rhs[0], f"{path}[0]")
            return
        if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            lhs_cpu = lhs.detach().cpu()
            rhs_cpu = rhs.detach().cpu()
            if lhs_cpu.shape != rhs_cpu.shape:
                raise ValueError(
                    f"output {path} shape differs: {lhs_cpu.shape} vs {rhs_cpu.shape}"
                )
            if lhs_cpu.dtype != rhs_cpu.dtype:
                raise ValueError(
                    f"output {path} dtype differs: {lhs_cpu.dtype} vs {rhs_cpu.dtype}"
                )
            total = lhs_cpu.numel()
            equal = torch.eq(lhs_cpu, rhs_cpu)
            if lhs_cpu.is_floating_point() or lhs_cpu.is_complex():
                equal = equal | (torch.isnan(lhs_cpu) & torch.isnan(rhs_cpu))
                lhs_float = lhs_cpu.float()
                rhs_float = rhs_cpu.float()
                absolute = (lhs_float - rhs_float).abs()
                denominator = rhs_float.abs().clamp_min(torch.finfo(torch.float32).tiny)
                relative = absolute / denominator
                finite_relative = relative[torch.isfinite(relative)]
                max_relative = (
                    float(finite_relative.max().item())
                    if finite_relative.numel()
                    else math.inf
                )
                max_absolute = float(absolute.max().item()) if total else 0.0
                mean_absolute = float(absolute.mean().item()) if total else 0.0
                reference_max = float(rhs_float.abs().max().item()) if total else 0.0
                relative_linf = (
                    max_absolute / reference_max
                    if reference_max
                    else (0.0 if max_absolute == 0.0 else math.inf)
                )
                reference_norm = float(torch.linalg.vector_norm(rhs_float).item())
                difference_norm = float(torch.linalg.vector_norm(lhs_float - rhs_float).item())
                relative_l2 = (
                    difference_norm / reference_norm
                    if reference_norm
                    else (0.0 if difference_norm == 0.0 else math.inf)
                )
            else:
                max_relative = 0.0
                max_absolute = float((lhs_cpu != rhs_cpu).any().item())
                mean_absolute = float((lhs_cpu != rhs_cpu).float().mean().item())
                relative_linf = max_absolute
                relative_l2 = mean_absolute
            exact = int(equal.sum().item())
            rows.append(
                {
                    "path": path,
                    "shape": list(lhs_cpu.shape),
                    "dtype": str(lhs_cpu.dtype),
                    "elements": total,
                    "exact_elements": exact,
                    "exact_fraction": exact / total if total else 1.0,
                    "max_abs_error": max_absolute,
                    "mean_abs_error": mean_absolute,
                    "max_relative_error": max_relative,
                    "relative_linf": relative_linf,
                    "relative_l2": relative_l2,
                }
            )
            return
        if isinstance(lhs, (tuple, list)) and isinstance(rhs, type(lhs)):
            if len(lhs) != len(rhs):
                raise ValueError(f"output {path} length differs")
            for index, (lhs_item, rhs_item) in enumerate(zip(lhs, rhs)):
                visit(lhs_item, rhs_item, f"{path}[{index}]")
            return
        if isinstance(lhs, dict) and isinstance(rhs, dict):
            if lhs.keys() != rhs.keys():
                raise ValueError(f"output {path} keys differ")
            for key in lhs:
                visit(lhs[key], rhs[key], f"{path}[{key!r}]")
            return
        raise TypeError(f"output {path} structures differ")

    visit(actual, expected, "output")
    return {"outputs": rows}


def _install_output_capture(
    benchmark: ModuleType,
    torch: Any,
    *,
    save_output: Path | None,
    compare_output: Path | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if save_output is None and compare_output is None:
        return metrics
    expected = torch.load(compare_output, map_location="cpu", weights_only=True) if compare_output else None
    captured = False
    original_measure_legs = benchmark.measure_legs

    def measure_legs(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        nonlocal captured
        legs = kwargs.get("legs", args[6] if len(args) > 6 else ())
        wrapped = []
        for name, call in legs:
            def capture(call: Callable[[], Any] = call) -> Any:
                nonlocal captured
                value = call()
                if not captured:
                    cpu_value = _cpu_output(torch, value)
                    if save_output is not None:
                        save_output.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(cpu_value, save_output)
                    if expected is not None:
                        metrics.update(_precision_comparison(torch, cpu_value, expected))
                    captured = True
                return value

            wrapped.append((name, capture))
        if "legs" in kwargs:
            kwargs["legs"] = wrapped
        else:
            positional = list(args)
            positional[6] = wrapped
            args = tuple(positional)
        return original_measure_legs(*args, **kwargs)

    benchmark.measure_legs = measure_legs
    return metrics


def _install_reference_config(benchmark: ModuleType, path: Path) -> None:
    """Pass one fixed configuration through a generated benchmark's kernel leg."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("reference configuration must be a JSON object")
    if payload.get("schema_version") == 1:
        overrides = payload.get("module_overrides", [])
        if not isinstance(overrides, list):
            raise TypeError("module_overrides must be a list")
        if overrides:
            original_build_benchmark = benchmark.build_benchmark

            def build_benchmark(*args: Any, **kwargs: Any) -> Any:
                instance = original_build_benchmark(*args, **kwargs)
                from gem.next_gen.geo.kernel_library.kernels.path_loader import (
                    load_kernel_module,
                )

                for override in overrides:
                    if not isinstance(override, dict):
                        raise TypeError("each module override must be an object")
                    relative_path = override.get("relative_path")
                    values = override.get("values")
                    if not isinstance(relative_path, str) or not isinstance(
                        values, dict
                    ):
                        raise TypeError(
                            "module override requires relative_path and values"
                        )
                    module = load_kernel_module(relative_path, "reference_tuning")
                    for name, value in values.items():
                        if not isinstance(name, str) or not hasattr(module, name):
                            raise ValueError(
                                f"reference module {relative_path} has no global "
                                f"{name!r}"
                            )
                        setattr(module, name, value)
                return instance

            benchmark.build_benchmark = build_benchmark
        config = payload.get("kernel_config")
        if config is None:
            return
    else:
        config = payload
    if not isinstance(config, dict):
        raise TypeError("kernel_config must be an object or null")
    if isinstance(config.get("ctas_per_cga"), list):
        config["ctas_per_cga"] = tuple(config["ctas_per_cga"])
    original_method_by_mro = benchmark.method_by_mro

    def method_by_mro(instance: Any, names: Any, what: str) -> Callable[..., Any]:
        method = original_method_by_mro(instance, names, what)
        if what != "kernel":
            return method

        def configured(inputs: Any) -> Any:
            return method(inputs, config=dict(config))

        return configured

    benchmark.method_by_mro = method_by_mro


def _select_benchmark_leg(benchmark: ModuleType, only: str) -> None:
    if only == "both":
        return
    selected_name = "fused" if only == "fused" else "geo_tlx"
    original_measure_legs = benchmark.measure_legs

    def measure_legs(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        all_legs = kwargs.get("legs", args[6] if len(args) > 6 else ())
        if "legs" in kwargs:
            kwargs["legs"] = [leg for leg in kwargs["legs"] if leg[0] == selected_name]
        else:
            positional = list(args)
            positional[6] = [leg for leg in positional[6] if leg[0] == selected_name]
            args = tuple(positional)
        results = original_measure_legs(*args, **kwargs)
        for name, _ in all_legs:
            results.setdefault(name, {"accuracy": "PASS", "skipped": True})
        return results

    benchmark.measure_legs = measure_legs
    # The generated display assumes both legs are present.
    benchmark.print_results = lambda results: None


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
    torch.manual_seed(args.input_seed)
    torch.cuda.manual_seed_all(args.input_seed)

    benchmark = _load_benchmark(args.benchmark)
    if args.reference_config_json is not None:
        _install_reference_config(benchmark, args.reference_config_json)
    _select_benchmark_leg(benchmark, args.only)
    precision_comparison = _install_output_capture(
        benchmark,
        torch,
        save_output=args.save_output,
        compare_output=args.compare_output,
    )
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
        if args.only == "fused":
            payload.pop("geo_tlx", None)
        elif "geo_tlx" in payload:
            payload["reference"] = payload.pop("geo_tlx")
            if args.only == "reference":
                payload.pop("fused", None)
        payload["triton_package"] = str(loaded_package)
        if precision_comparison:
            payload["tlx_precision_comparison"] = precision_comparison
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
