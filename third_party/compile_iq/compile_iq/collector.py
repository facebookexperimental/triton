"""Stage 1 — collection. A zero-user-surface hook (called from jit.py, gated by
FBTRITON_COMPILE_IQ_COLLECT) that, when a kernel is launched, dumps a self-contained
"compileIQ task" to disk for the offline factory.

A task dir ($COMPILE_IQ_TASK_DIR/<ptx_sha[:16]>/) contains:
    kernel.ptx        the emitted PTX (the ACF will be tuned for this)
    task.json         identity, launch dims, arch, ptxas, arg metadata, grid
    source.py         copy of the kernel's defining source (for replay)

PINNED: arg shapes + kernel identity are captured here so the factory can replay,
even though the store key (see store.py) is sha256(PTX) alone for now.
"""

import inspect
import json
import os
import re
import shutil

from .store import dlog, ptx_sha256


def task_root() -> str:
    return os.environ.get("COMPILE_IQ_TASK_DIR", os.path.expanduser("~/.compile_iq/tasks"))


def _dtype_str(dt) -> str:
    return str(dt).replace("torch.", "")


def _serialize_args(bound_args, signature, constexpr_map):
    import torch

    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except Exception:
        TensorDescriptor = None

    out = []
    for name, val in bound_args.items():
        e = {"name": name, "sig_type": signature.get(name, "")}
        if name in constexpr_map:
            e["kind"] = "constexpr"
            e["value"] = constexpr_map[name]
        elif TensorDescriptor is not None and isinstance(val, TensorDescriptor):
            e.update(kind="tensor_descriptor", base_shape=list(val.base.shape), base_dtype=_dtype_str(val.base.dtype),
                     base_strides=list(val.base.stride()), block_shape=[int(s) for s in val.block_shape])
        elif isinstance(val, torch.Tensor):
            e.update(kind="tensor", shape=list(val.shape), dtype=_dtype_str(val.dtype), strides=list(val.stride()))
        elif isinstance(val, (bool, int, float)):
            e.update(kind="scalar", value=val, scalar_type=type(val).__name__)
        else:
            e.update(kind="scalar", value=str(val), scalar_type=type(val).__name__)
        out.append(e)
    return out


def _arch_from_ptx(ptx: str) -> str:
    m = re.search(r"\.target\s+(sm_\w+)", ptx)
    return m.group(1) if m else "unknown"


def capture(*, jitfn, kernel, bound_args, signature, constexprs, grid):
    """Best-effort task dump. Never raises into the user's launch path."""
    try:
        ptx = kernel.asm["ptx"]
        sha = ptx_sha256(ptx)
        tdir = os.path.join(task_root(), sha[:16])
        done = os.path.join(tdir, "task.json")
        if os.path.exists(done):
            dlog("collector", f"skip {sha[:16]} {_arch_from_ptx(ptx)} (already collected)")
            return  # dedup: this PTX already collected
        os.makedirs(tdir, exist_ok=True)

        with open(os.path.join(tdir, "kernel.ptx"), "w") as f:
            f.write(ptx)

        # constexpr path-tuples -> name map (mirrors tlx_benchmark_gen)
        names = list(bound_args.keys())
        cmap = {}
        for path, v in (constexprs or {}).items():
            if isinstance(path, tuple) and len(path) == 1 and path[0] < len(names):
                cmap[names[path[0]]] = v

        md = kernel.metadata
        ptxas = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH") or os.environ.get("TRITON_PTXAS_PATH", "")
        task = {
            "kernel_name": getattr(md, "name", getattr(jitfn, "__name__", "kernel")),
            "fn_name": getattr(jitfn, "__name__", None),
            # Defining module of the kernel (and its wrapper) -- the factory must import the
            # SAME installed module, not the source.py copy: Triton bakes module identity into
            # the compile, so a copy yields a different PTX (-> wrong store key).
            "module": getattr(getattr(jitfn, "fn", jitfn), "__module__", None),
            "ptx_sha256": sha,
            "arch": _arch_from_ptx(ptx),
            "ptxas_path": ptxas,
            "num_warps": getattr(md, "num_warps", None),
            "num_ctas": getattr(md, "num_ctas", None),
            "shared": getattr(md, "shared", None),
            "cluster_dims": list(getattr(md, "cluster_dims", []) or []),
            "grid": list(grid),
            # Kernel-selecting env knobs (e.g. TLX_GEMM_USE_HEURISTIC) so the offline
            # factory reproduces the SAME single-config launch that was collected --
            # otherwise the factory autotunes a different (and slower) config and the
            # tuned PTX no longer matches the collected key.
            "env": {k: v for k, v in os.environ.items() if k.startswith("TLX_")},
            "args": _serialize_args(bound_args, signature, cmap),
            "constexprs": {k: (v if isinstance(v, (bool, int, float, str)) else str(v))
                           for k, v in cmap.items()},
        }

        # copy source file for replay
        try:
            src = inspect.getsourcefile(jitfn.fn)
            if src and os.path.exists(src):
                shutil.copy(src, os.path.join(tdir, "source.py"))
                task["source_file"] = "source.py"
                task["source_origin"] = os.path.basename(src)
        except Exception:
            pass

        with open(done, "w") as f:
            json.dump(task, f, indent=2, default=str)
        dlog("collector", f"collected {sha[:16]} {task['arch']} kernel={task['kernel_name']} "
             f"grid={task['grid']} args={len(task['args'])} -> {tdir}")
    except Exception as e:  # collection must never break the user's run
        dlog("collector", f"capture failed: {type(e).__name__}: {e}")
