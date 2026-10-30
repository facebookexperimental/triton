"""Replay a TTGIR-to-TLX dump against the kernel it was generated from.

`_test_standalone.py` (written by TRITON_DUMP_TLX_BENCHMARK=1) times the
generated TLX kernel, but it cannot say whether that kernel computes what the
source did. This runs both kernels on identical inputs, compares every tensor
argument bitwise afterwards (outputs are written in place), and times both.

Everything comes from `_kernel_args.json`: the source kernel's module, the
backend knobs it compiled under (AutoWS kernels need use_meta_ws and friends),
and its launch options, including the cluster shape. Point TLX_DUMP_DIR at the
dump directory. Exits nonzero when the outputs differ.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import sys

import torch
import triton
import triton.language as tl
from triton.runtime.jit import JITFunction
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = "cuda"
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "float8_e5m2": torch.float8_e5m2,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e8m0fnu": torch.float8_e8m0fnu,
}


def _make_tensor(shape, dtype, gen):
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return torch.randint(0, 127, shape, dtype=dtype, device=DEVICE, generator=gen)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=DEVICE, generator=gen).bool()
    if dtype == torch.float8_e8m0fnu:
        # Unsigned exponent-only scales: keep them near 2**0 rather than NaN.
        return torch.randint(120, 135, shape, dtype=torch.uint8, device=DEVICE, generator=gen).view(dtype)
    return (torch.randn(shape, device=DEVICE, generator=gen) * 0.5).to(dtype)


def _constexpr_value(entry):
    value, kind = entry.get("value"), entry.get("scalar_type")
    if kind == "NoneType":
        return None
    if kind == "dtype":
        return tl.dtype(value)
    return value


def _build_args(meta, seed):
    """Every argument's value by name, and every tensor the kernel can write."""
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    values, tensors = {}, {}
    for entry in meta["args"]:
        kind, name = entry["kind"], entry["name"]
        if kind == "tensor":
            values[name] = tensors[name] = _make_tensor(entry["shape"], _DTYPES[entry["dtype"]], gen)
        elif kind == "tensor_descriptor":
            tensors[name] = _make_tensor(entry["base_shape"], _DTYPES[entry["base_dtype"]], gen)
            values[name] = TensorDescriptor(tensors[name], shape=entry["desc_shape"], strides=entry["desc_strides"],
                                            block_shape=entry["block_shape"])
        elif kind == "constexpr":
            values[name] = _constexpr_value(entry)
        else:
            convert = {"float": float, "int": int, "bool": bool}.get(entry.get("scalar_type"))
            values[name] = convert(entry["value"]) if convert else entry["value"]
    return values, tensors


def _load_source_kernel(origin):
    try:
        module = importlib.import_module(origin["module"])
    except ImportError as e:
        # Tutorials such as 06-fused-attention-ws.py are not importable by name.
        if not origin.get("file"):
            raise SystemExit(f"cannot import source module {origin['module']!r}, and the dump recorded no source "
                             "file to load it from") from e
        spec = importlib.util.spec_from_file_location("_tlx_roundtrip_source", origin["file"])
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load source module {origin['module']!r} from {origin['file']!r}") from e
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    fn = getattr(module, origin["name"])
    # Autotuner and Heuristics wrap the JITFunction; launch it directly with the
    # captured config instead of re-tuning.
    while not isinstance(fn, JITFunction):
        fn = fn.fn
    return fn


def _load_tlx_kernel(dump_dir, name):
    sys.path.insert(0, dump_dir)
    path = os.path.join(dump_dir, name + "_kernel.py")
    spec = importlib.util.spec_from_file_location("_tlx_roundtrip_generated", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the generated TLX kernel from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


@contextlib.contextmanager
def _knobs(overrides):
    with contextlib.ExitStack() as stack:
        for section, values in overrides.items():
            group = getattr(triton.knobs, section)
            stack.enter_context(group.scope())
            for name, value in values.items():
                setattr(group, name, value)
        yield


def _compiled_stats(fn):
    for cache in fn.device_caches.values():
        for kernel in cache[0].values():
            kernel = kernel.result() if hasattr(kernel, "result") else kernel
            kernel._init_handles()
            return f"regs={kernel.n_regs} spills={kernel.n_spills} shared={kernel.metadata.shared}"
    return "unknown"


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _median(xs):
    return sorted(xs)[len(xs) // 2]


def main() -> None:
    dump_dir = os.environ.get("TLX_DUMP_DIR")
    if not dump_dir:
        raise SystemExit("TLX_DUMP_DIR is unset; point it at a TRITON_TLX_DUMP_DIR from a dump")
    with open(os.path.join(dump_dir, "_kernel_args.json")) as f:
        meta = json.load(f)
    origin = meta.get("kernel", {})
    if "module" not in origin:
        raise SystemExit("_kernel_args.json does not record the source kernel; re-dump with this Triton")

    tlx_kernel = _load_tlx_kernel(dump_dir, origin["name"])
    src_kernel = _load_source_kernel(origin)
    launch = {k: tuple(v) if isinstance(v, list) else v for k, v in meta.get("launch_options", {}).items()}
    knobs = meta.get("knobs", {})
    grid = tuple(meta["grid"])
    print(f"[roundtrip] {origin['name']} grid={grid} launch={launch} knobs={knobs}", flush=True)
    triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device=DEVICE))

    # The generated kernel bakes every constexpr in, including runtime scalars
    # Triton specialized to a constant (a unit stride); the source still takes
    # those positionally, which _kernel_args.json cannot tell apart.
    tlx_values, tlx_tensors = _build_args(meta, seed=0)
    tlx_args = [tlx_values[e["name"]] for e in meta["args"] if e["kind"] != "constexpr"]
    src_values, src_tensors = _build_args(meta, seed=0)
    if not src_tensors:
        raise SystemExit("the kernel takes no tensor arguments, so there is nothing to compare")
    src_args = [src_values[p.name] for p in src_kernel.params if not p.is_constexpr]
    constexprs = {p.name: src_values[p.name] for p in src_kernel.params if p.is_constexpr and p.name in src_values}
    initial = {name: t.clone() for name, t in src_tensors.items()}

    def run_src():
        with _knobs(knobs):
            src_kernel[grid](*src_args, **constexprs, **launch)

    def run_tlx():
        # Knobs only steer AutoWS, which the generated kernel no longer goes through.
        tlx_kernel[grid](*tlx_args, **launch)

    # Synchronize after each so a fault is attributed to the right kernel.
    run_src()
    torch.cuda.synchronize()
    run_tlx()
    torch.cuda.synchronize()
    print(f"[roundtrip] source: {_compiled_stats(src_kernel)}  tlx: {_compiled_stats(tlx_kernel)}", flush=True)

    all_equal, any_written = True, False
    for name, src in src_tensors.items():
        tlx_out = tlx_tensors[name]
        # Compare bits, so a NaN both kernels produced still counts as equal.
        equal = torch.equal(_bits(src), _bits(tlx_out))
        all_equal &= equal
        src_written = not torch.equal(_bits(src), _bits(initial[name]))
        tlx_written = not torch.equal(_bits(tlx_out), _bits(initial[name]))
        written = src_written or tlx_written
        any_written |= written
        diff = torch.nan_to_num((src.float() - tlx_out.float()).abs(), nan=float("inf"))
        print(f"[roundtrip]   {name:12s} written={written} bitwise_equal={equal} max_abs_diff={diff.max().item():.3e}",
              flush=True)
    # Unwritten outputs match trivially; equality only means something once a
    # kernel has produced something.
    if not any_written:
        raise SystemExit("neither kernel wrote any of its tensor arguments, so the comparison checked nothing")
    print(f"[roundtrip] OUTPUTS_BITWISE_EQUAL={all_equal}", flush=True)

    # Eager timing is CPU-bound for kernels under ~100us: the source launch binds
    # every constexpr as a kwarg and enters a knobs scope. CUDA graphs remove that.
    # Interleave the two so clock drift hits both alike. TLX_ROUNDTRIP_REPS=0
    # checks correctness only, e.g. under ncu or compute-sanitizer.
    reps = int(os.environ.get("TLX_ROUNDTRIP_REPS", "3"))
    times = {"src": [], "tlx": [], "src_graph": [], "tlx_graph": []}
    for _ in range(reps):
        times["src"].append(triton.testing.do_bench(run_src, warmup=100, rep=500))
        times["tlx"].append(triton.testing.do_bench(run_tlx, warmup=100, rep=500))
        times["src_graph"].append(triton.testing.do_bench_cudagraph(run_src, rep=500))
        times["tlx_graph"].append(triton.testing.do_bench_cudagraph(run_tlx, rep=500))
    modes = (("eager", "src", "tlx"), ("graph", "src_graph", "tlx_graph")) if reps > 0 else ()
    for mode, src_key, tlx_key in modes:
        src_ms, tlx_ms = _median(times[src_key]), _median(times[tlx_key])
        # A kernel below event resolution can time at 0; the bitwise verdict
        # above must still decide the exit code.
        ratio = f"{tlx_ms / src_ms:.4f}" if src_ms > 0 else "n/a"
        print(f"[roundtrip] {mode:5s} source {src_ms:.4f} ms  tlx {tlx_ms:.4f} ms  tlx/source={ratio}", flush=True)
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
