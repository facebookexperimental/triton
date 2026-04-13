"""Utilities for capturing kernel arguments and generating standalone TLX benchmark tests.

When TRITON_DUMP_TLX_BENCHMARK is set, the JIT runtime calls capture_kernel_args()
before compilation to serialize argument metadata (tensor shapes, dtypes, strides,
TensorDescriptor configs, scalar values, constexprs) to _kernel_args.json in the
TLX dump directory. After grid evaluation, capture_grid() appends the actual grid.

_generate_standalone_test() reads this JSON and produces a generic _test_standalone.py
that works for any kernel — no hardcoded attention-specific inputs.
"""

import json
import logging
import os
import tempfile

log = logging.getLogger(__name__)


def _ensure_dump_dir():
    """Return the TLX dump directory, creating it if necessary."""
    dump_dir = os.environ.get("TRITON_TLX_DUMP_DIR")
    if dump_dir and os.path.isdir(dump_dir):
        return dump_dir
    dump_dir = tempfile.mkdtemp(prefix="triton_tlx_")
    os.environ["TRITON_TLX_DUMP_DIR"] = dump_dir
    log.warning("TLX benchmark dump dir: %s", dump_dir)
    return dump_dir


# ---------------------------------------------------------------------------
# Helpers called from CUDABackend.make_llir() in compiler.py
# ---------------------------------------------------------------------------


def setup_tlx_dump(pm, tlx_passes):
    """Set up TLX benchmark dump before ``pm.run()``.

    Adds the TLX print pass to *pm*, creates the dump directory, and redirects
    fd 1 (C++ ``llvm::outs()``) to a capture file so that older code-paths
    that still print to stdout are also caught.

    Returns ``(dump_dir, saved_fd, capture_file)`` — pass these to
    :func:`finalize_tlx_dump` after ``pm.run()`` completes.
    """
    import sys

    dump_dir = _ensure_dump_dir()
    tlx_passes.add_tlx_print_ttgir_to_tlx(pm)

    sys.stdout.flush()
    capture_file = os.path.join(dump_dir, "_stdout_capture.txt")
    saved_fd = os.dup(1)
    fd = os.open(capture_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.close(fd)

    return dump_dir, saved_fd, capture_file


def finalize_tlx_dump(dump_dir, saved_fd, capture_file, metadata):
    """Process TLX dump artifacts after ``pm.run()``.

    Restores stdout, collects ``.tlx`` files from *dump_dir*, copies the
    original kernel source (if found), and generates ``_test_standalone.py``.
    """
    import re as _re
    import shutil
    import sys
    from glob import glob

    # Restore stdout
    if saved_fd is not None:
        sys.stdout.flush()
        os.fsync(1)
        os.dup2(saved_fd, 1)
        os.close(saved_fd)

    try:
        tlx_files = glob(os.path.join(dump_dir, "*.tlx"))

        # Fall back to captured stdout if the C++ pass didn't write a file
        if not tlx_files and capture_file and os.path.exists(capture_file):
            with open(capture_file, "r") as f:
                captured = f.read()
            if captured.strip():
                kernel_name = "kernel"
                for line in captured.splitlines():
                    if line.startswith("func "):
                        parts = line.split("(")[0].split()
                        if len(parts) >= 2:
                            kernel_name = parts[1]
                            break
                tlx_file = os.path.join(dump_dir, kernel_name + ".tlx")
                with open(tlx_file, "w") as f:
                    f.write(captured)
                tlx_files = [tlx_file]

        for tlx_file in tlx_files:
            with open(tlx_file, "r") as f:
                tlx_dump = f.read()
            kernel_name = os.path.splitext(os.path.basename(tlx_file))[0]
            kernel_path = os.path.join(dump_dir, kernel_name + "_kernel.py")
            shutil.copy2(tlx_file, kernel_path)

            # Try to find and copy the original kernel source module
            source_origin = None
            source_module = None
            for _line in tlx_dump.splitlines():
                _m = _re.search(r'#\s+(\w+)\.py:\d+', _line)
                if _m:
                    source_module = _m.group(1)
                    break
            if source_module:
                try:
                    import importlib.util
                    for mod_name in [f"tritonbench.kernels.{source_module}", source_module]:
                        spec = importlib.util.find_spec(mod_name)
                        if spec and spec.origin:
                            source_dest = os.path.join(dump_dir, kernel_name + "_source.py")
                            shutil.copy2(spec.origin, source_dest)
                            source_origin = spec.origin
                            break
                except Exception as exc:
                    log.warning("Could not copy kernel source (%s): %s", source_module, exc)

            generate_standalone_test(dump_dir, kernel_name, source_origin, metadata)

            # Log per-file details on first compilation only
            if not os.environ.get("_TRITON_TLX_LOGGED"):
                test_path = os.path.join(dump_dir, "_test_standalone.py")
                if os.path.exists(test_path):
                    log.warning("TLX test:      %s", test_path)
                log.warning("TLX kernel:    %s", kernel_path)
                log.warning("TLX dump:      %s", tlx_file)

        if not os.environ.get("_TRITON_TLX_LOGGED"):
            os.environ["_TRITON_TLX_LOGGED"] = "1"
        if not tlx_files:
            log.warning("No TLX output captured in %s", dump_dir)
    except Exception as e:
        log.warning("Error generating TLX benchmark: %s", e)


def _dtype_str(dtype):
    """Convert a torch dtype to a serialisable string like 'bfloat16'."""
    return str(dtype).replace("torch.", "")


def capture_kernel_args(bound_args, signature, constexprs, _params=None):
    """Serialize kernel call argument metadata to *_kernel_args.json*.

    Parameters
    ----------
    bound_args : OrderedDict[str, Any]
        Mapping from parameter name to actual value (tensors, scalars,
        TensorDescriptor objects, …).
    signature : dict[str, str]
        Mapping from parameter name to Triton type string (e.g. ``"*bf16"``,
        ``"i32"``, ``"constexpr"``).
    constexprs : dict[tuple, Any]
        Mapping from path-tuples ``(index,)`` to constexpr values.
    params : list
        The ``JITFunction.params`` list (used for positional ordering).
    """
    import torch

    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except ImportError:
        TensorDescriptor = None

    dump_dir = _ensure_dump_dir()
    arg_names = list(bound_args.keys())

    # Build constexpr name→value mapping
    constexpr_map = {}
    for path, val in constexprs.items():
        if isinstance(path, tuple) and len(path) == 1:
            idx = path[0]
            if idx < len(arg_names):
                constexpr_map[arg_names[idx]] = val

    args_list = []
    for name, val in bound_args.items():
        sig_type = signature.get(name, "")
        entry = {"name": name, "sig_type": sig_type}

        if name in constexpr_map:
            entry["kind"] = "constexpr"
            v = constexpr_map[name]
            if isinstance(v, bool):
                entry["value"] = v
                entry["scalar_type"] = "bool"
            elif isinstance(v, int):
                entry["value"] = v
                entry["scalar_type"] = "int"
            elif isinstance(v, float):
                entry["value"] = v
                entry["scalar_type"] = "float"
            else:
                entry["value"] = str(v)
                entry["scalar_type"] = type(v).__name__
        elif TensorDescriptor is not None and isinstance(val, TensorDescriptor):
            entry["kind"] = "tensor_descriptor"
            entry["base_shape"] = list(val.base.shape)
            entry["base_dtype"] = _dtype_str(val.base.dtype)
            entry["base_strides"] = list(val.base.stride())
            entry["desc_shape"] = [int(s) for s in val.shape]
            entry["desc_strides"] = [int(s) for s in val.strides]
            entry["block_shape"] = [int(s) for s in val.block_shape]
        elif isinstance(val, torch.Tensor):
            entry["kind"] = "tensor"
            entry["shape"] = list(val.shape)
            entry["dtype"] = _dtype_str(val.dtype)
            entry["strides"] = list(val.stride())
        elif isinstance(val, bool):
            entry["kind"] = "scalar"
            entry["scalar_type"] = "bool"
            entry["value"] = val
        elif isinstance(val, int):
            entry["kind"] = "scalar"
            entry["scalar_type"] = "int"
            entry["value"] = val
        elif isinstance(val, float):
            entry["kind"] = "scalar"
            entry["scalar_type"] = "float"
            entry["value"] = val
        else:
            entry["kind"] = "scalar"
            entry["scalar_type"] = type(val).__name__
            entry["value"] = str(val)

        args_list.append(entry)

    meta = {
        "args": args_list,
        "constexprs": constexpr_map,
    }

    json_path = os.path.join(dump_dir, "_kernel_args.json")
    try:
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
    except Exception as exc:
        log.warning("Could not write _kernel_args.json: %s", exc)


def capture_grid(grid_tuple):
    """Append the evaluated grid to *_kernel_args.json*."""
    dump_dir = os.environ.get("TRITON_TLX_DUMP_DIR")
    if not dump_dir:
        return
    json_path = os.path.join(dump_dir, "_kernel_args.json")
    if not os.path.exists(json_path):
        return
    try:
        import fcntl
        with open(json_path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            meta = json.load(f)
            meta["grid"] = list(grid_tuple) if hasattr(grid_tuple, "__iter__") else [grid_tuple]
            f.seek(0)
            f.truncate()
            json.dump(meta, f, indent=2, default=str)
    except Exception as exc:
        log.warning("Could not update grid in _kernel_args.json: %s", exc)


# ---------------------------------------------------------------------------
# Standalone test generation
# ---------------------------------------------------------------------------

_TORCH_DTYPE_MAP = {
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "float32": "torch.float32",
    "float64": "torch.float64",
    "int8": "torch.int8",
    "int16": "torch.int16",
    "int32": "torch.int32",
    "int64": "torch.int64",
    "bool": "torch.bool",
    "float8_e5m2": "torch.float8_e5m2",
    "float8_e4m3fn": "torch.float8_e4m3fn",
}


def generate_standalone_test(dump_dir, kernel_name, _source_origin=None, _metadata=None):
    """Generate ``_test_standalone.py`` that runs the dumped TLX kernel.

    Reads ``_kernel_args.json`` (written by :func:`capture_kernel_args`) and
    produces a self-contained benchmark script that works for *any* kernel.
    """
    json_path = os.path.join(dump_dir, "_kernel_args.json")
    if not os.path.exists(json_path):
        log.warning("Cannot generate standalone test: _kernel_args.json not found in %s", dump_dir)
        return

    with open(json_path) as f:
        _meta = json.load(f)  # validate JSON is readable

    # Determine if source module exists (for pre-hook support)
    source_file = os.path.join(dump_dir, kernel_name + "_source.py")
    has_source = os.path.exists(source_file)

    lines = [
        "#!/usr/bin/env python3",
        '"""Standalone benchmark for generated TLX kernel: ' + kernel_name + '.',
        "",
        "Auto-generated from captured kernel arguments.",
        "Input shapes, dtypes, and TensorDescriptor configs are read from",
        "_kernel_args.json (produced during the original compilation).",
        '"""',
        "import json, os, sys, argparse, importlib.util",
        "",
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))",
        "from " + kernel_name + "_kernel import " + kernel_name,
        "",
        "import torch",
        "import triton",
        "from triton.tools.tensor_descriptor import TensorDescriptor",
        "",
        'DEVICE = "cuda"',
        "",
        "",
    ]

    # --- _load_source_module helper (only if source exists) ---
    if has_source:
        lines += [
            "def _load_source_module():",
            '    """Load the original source module to access its warmup hook."""',
            "    source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),",
            '                               "' + kernel_name + '_source.py")',
            '    spec = importlib.util.spec_from_file_location("_source", source_path)',
            "    mod = importlib.util.module_from_spec(spec)",
            "    spec.loader.exec_module(mod)",
            "    return mod",
            "",
            "",
        ]

    # --- benchmark function ---
    lines += [
        "def _make_tensor(shape, dtype, device):",
        "    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):",
        "        return torch.randint(0, 127, shape, dtype=dtype, device=device)",
        "    elif dtype == torch.bool:",
        "        return torch.randint(0, 2, shape, device=device).bool()",
        "    else:",
        "        return torch.randn(shape, device=device).to(dtype)",
        "",
        "",
        "def benchmark():",
        "    # Load captured argument metadata",
        "    meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_kernel_args.json')",
        "    with open(meta_path) as f:",
        "        meta = json.load(f)",
        "",
        "    args_meta = meta['args']",
        "    constexprs = meta.get('constexprs', {})",
        "    grid_vals = meta.get('grid', [1, 1, 1])",
        "",
        "    # Allocator for Triton scratch memory",
        "    triton.set_allocator(lambda s, a, _: torch.empty(s, dtype=torch.int8, device=DEVICE))",
        "",
        "    _DTYPE_MAP = {",
        '        "float16": torch.float16, "bfloat16": torch.bfloat16,',
        '        "float32": torch.float32, "float64": torch.float64,',
        '        "int8": torch.int8, "int16": torch.int16,',
        '        "int32": torch.int32, "int64": torch.int64,',
        '        "bool": torch.bool,',
        '        "float8_e5m2": torch.float8_e5m2,',
        '        "float8_e4m3fn": torch.float8_e4m3fn,',
        "    }",
        "",
        "    # Build kernel arguments from metadata",
        "    # TLX kernel: constexprs are baked in, only positional args needed.",
        "    # Source kernel: constexprs must be passed as kwargs.",
        "    kernel_args = []       # positional args for TLX kernel",
        "    source_kwargs = {}     # constexprs passed as kwargs to source kernel",
        "    desc_args = {}",
        "    scalar_map = {}        # name -> value for FLOPS computation",
        "",
        "    for entry in args_meta:",
        "        name = entry['name']",
        "        kind = entry['kind']",
        "",
        "        if kind == 'tensor':",
        "            dtype = _DTYPE_MAP.get(entry['dtype'], torch.float32)",
        "            t = _make_tensor(entry['shape'], dtype, DEVICE)",
        "            kernel_args.append(t)",
        "",
        "        elif kind == 'tensor_descriptor':",
        "            dtype = _DTYPE_MAP.get(entry['base_dtype'], torch.bfloat16)",
        "            base = _make_tensor(entry['base_shape'], dtype, DEVICE)",
        "            desc = TensorDescriptor(",
        "                base,",
        "                shape=entry['desc_shape'],",
        "                strides=entry['desc_strides'],",
        "                block_shape=entry['block_shape'],",
        "            )",
        "            kernel_args.append(desc)",
        "            desc_args[name] = desc",
        "",
        "        elif kind == 'constexpr':",
        "            source_kwargs[name] = entry['value']",
        "            scalar_map[name] = entry['value']",
        "",
        "        elif kind == 'scalar':",
        "            val = entry['value']",
        "            if entry.get('scalar_type') == 'float':",
        "                val = float(val)",
        "            elif entry.get('scalar_type') == 'int':",
        "                val = int(val)",
        "            elif entry.get('scalar_type') == 'bool':",
        "                val = bool(val)",
        "            kernel_args.append(val)",
        "            scalar_map[name] = val",
        "",
    ]

    # --- Apply pre-hook if source module exists ---
    if has_source:
        lines += [
            "    # Apply source module's warmup hook (sets TensorDescriptor block_shapes)",
            "    src = _load_source_module()",
            "    try:",
            "        if hasattr(src, '_fwd_host_descriptor_pre_hook'):",
            "            nargs = dict(constexprs)",
            "            nargs.update(desc_args)",
            "            src._fwd_host_descriptor_pre_hook(nargs)",
            "    except Exception as e:",
            "        print(f'Warning: could not apply pre-hook: {e}')",
            "",
        ]

    # --- FLOPS computation ---
    lines += [
        "    # Compute FLOPS: try TensorDescriptor base shapes (generic for attention-like kernels)",
        "    flops = 0",
        "    desc_base_shapes = {e['name']: e['base_shape'] for e in args_meta if e['kind'] == 'tensor_descriptor'}",
        "    q_shape = desc_base_shapes.get('desc_q')",
        "    k_shape = desc_base_shapes.get('desc_k')",
        "    if q_shape and len(q_shape) == 4 and k_shape and len(k_shape) == 4:",
        "        B, H, S_Q, D = q_shape",
        "        _, _, S_KV, _ = k_shape",
        "        flops = 4.0 * B * H * S_Q * S_KV * D",
        "",
    ]

    # --- TLX kernel benchmark ---
    lines += [
        "    grid = tuple(grid_vals)",
        "",
        '    print(f"--- TLX kernel ---", flush=True)',
        "    " + kernel_name + "[grid](*kernel_args)",
        "    torch.cuda.synchronize()",
        "",
        "    ms_tlx = triton.testing.do_bench(",
        "        lambda: " + kernel_name + "[grid](*kernel_args),",
        "        warmup=100, rep=500)",
        "    tflops_tlx = flops / ms_tlx / 1e9 if flops and ms_tlx else 0",
        "    if flops:",
        '        print(f"TLX kernel:    {ms_tlx:.3f} ms | {tflops_tlx:.1f} TFLOPS")',
        "    else:",
        '        print(f"TLX kernel:    {ms_tlx:.3f} ms")',
        "",
    ]

    # --- Source kernel benchmark (only if source exists) ---
    if has_source:
        lines += [
            '    print(f"--- Source kernel ---", flush=True)',
            "    src_kernel = getattr(src, " + repr(kernel_name) + ", None)",
            "    if src_kernel is not None:",
            "        # Source kernel is autotuned — pass user-level constexprs as kwargs.",
            "        # Filter out autotuner-managed config keys (BLOCK_*, NUM_*, USE_*).",
            "        user_kwargs = {k: v for k, v in source_kwargs.items()",
            "                       if not k.startswith(('BLOCK_', 'NUM_', 'USE_'))}",
            "        # Grid: use the captured grid (works for any kernel).",
            "        # Wrap in a lambda accepting META so the autotuner can call it.",
            "        _captured_grid = tuple(grid_vals)",
            "        def src_grid(META):",
            "            return _captured_grid",
            "        src_kernel[src_grid](*kernel_args, **user_kwargs)",
            "        torch.cuda.synchronize()",
            "",
            "        ms_src = triton.testing.do_bench(",
            "            lambda: src_kernel[src_grid](*kernel_args, **user_kwargs),",
            "            warmup=100, rep=500)",
            "        tflops_src = flops / ms_src / 1e9 if flops and ms_src else 0",
            "        if flops:",
            '            print(f"Source kernel: {ms_src:.3f} ms | {tflops_src:.1f} TFLOPS")',
            "        else:",
            '            print(f"Source kernel: {ms_src:.3f} ms")',
            "    else:",
            '        print(f"Source kernel {' + repr(kernel_name) + '} not found in source module")',
            "",
        ]

    lines += [
        "",
        'if __name__ == "__main__":',
        "    benchmark()",
    ]

    test_script = "\n".join(lines) + "\n"
    test_path = os.path.join(dump_dir, "_test_standalone.py")
    try:
        with open(test_path, "w") as f:
            f.write(test_script)
    except Exception as e:
        log.warning("Could not generate standalone test: %s", e)
