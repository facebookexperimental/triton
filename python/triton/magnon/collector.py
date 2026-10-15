"""Stage 1 — collection. A zero-user-surface hook (called from jit.py, gated by
TRITON_COMPILE_IQ_COLLECT) that, when a kernel is compiled, dumps a self-contained, SOURCE-FREE
"compileIQ task" to disk for the offline (PTX-direct) factory.

A task dir ($COMPILE_IQ_TASK_DIR/<ptx_sha[:16]>/) contains:
    kernel.ptx            the emitted PTX (the ACF is tuned for this; ptxas assembles it -> SASS)
    spec.json             the source-free launch description -- identity, arch, entry, block/grid,
                          shared, and the post-specialization kernel-param layout (see
                          ptx_launch.build_spec)
    <sha256>.pkl          the pre-launch data of each tensor arg under 1MB (a pickled CPU
                          torch.Tensor), named by the sha256 of the file and referenced by
                          spec.json's "pickle_sha256"
    launch_metadata.json  (optional) the compiler's Level 0 launch schema (asm["launch_metadata"])
    launcher.c            (optional) the compiler-generated launcher (asm["launcher_src"]) that the
                          factory can compile + drive via launch.h (see ptx_anneal.level1)

NO source.py is dumped: the factory never recompiles from source, it only runs ptxas (PTX->SASS)
and launches the cubin via the CUDA driver API, so the Python source is not needed. The store key
(see store.py) is sha256(normalized PTX) x arch.

Only kernels the PTX-direct path covers are collected; anything build_spec can't express yet
(non-null global/profile scratch, multi-CTA/cluster, tensordesc args, or any param mismatch) is
skipped fail-open -- the user's run is never affected. WS/TMA support extends build_spec later.
"""

import hashlib
import json
import os
import pickle
import re

from triton.backends.nvidia.compiler import get_ptxas
from triton.runtime.jit import _compile_iq_state, _compile_iq_suppress_competition

from . import ptx_launch
from .store import dlog, ptx_sha256

_PICKLE_MAX_BYTES = 1 << 20


def benchmarking():
    """Context manager marking launches inside it as autotuning benchmark runs.

    Frameworks that benchmark candidate kernels outside Triton's autotuner (e.g. Inductor) enter it
    so that only the winning config is collected, on its first launch outside benchmarking. Shares
    the flag Triton's autotuner sets around do_bench, which also defers the ACF A/B."""
    return _compile_iq_suppress_competition()


def task_root() -> str:
    return os.environ.get("COMPILE_IQ_TASK_DIR", os.path.expanduser("~/.compile_iq/tasks"))


def _dtype_str(dt) -> str:
    return str(dt).replace("torch.", "")


def _tensor_info(t, blobs):
    """Describe a tensor for build_spec. Tensors under _PICKLE_MAX_BYTES also get their data pickled
    into `blobs` (sha256 -> bytes) and the hash recorded as "pickle_sha256"."""
    info = {"id": t.data_ptr(), "shape": list(t.shape), "dtype": _dtype_str(t.dtype), "strides": list(t.stride())}
    if t.numel() * t.element_size() < _PICKLE_MAX_BYTES:
        # clone() so a small view doesn't pickle its whole (possibly huge) backing storage.
        data = pickle.dumps(t.detach().cpu().clone())
        sha = hashlib.sha256(data).hexdigest()
        blobs[sha] = data
        info["pickle_sha256"] = sha
    return info


def _ordered_args(bound_args, constexpr_names, blobs):
    """Classify each bound arg (in order) for build_spec: constexpr / tensor / tensordesc / scalar."""
    import torch
    ordered = []
    for name, val in bound_args.items():
        if name in constexpr_names:
            ordered.append(("constexpr", ))
        elif isinstance(val, torch.Tensor):
            ordered.append(("tensor", _tensor_info(val, blobs)))
        elif type(val).__name__ == "TensorDescriptor":  # host-side TMA descriptor (no hard import)
            ordered.append(("tensordesc", {
                "base": _tensor_info(val.base, blobs),
                "desc_shape": list(val.shape),
                "desc_strides": list(val.strides),
                "block_shape": list(val.block_shape),
                "padding": 1 if getattr(val, "padding", "zero") == "nan" else 0,
            }))
        elif isinstance(val, (bool, int, float)):
            ordered.append(("scalar", val))
        else:
            raise NotImplementedError(f"unsupported arg type {type(val).__name__} for {name!r}")
    return ordered


def capture(*, jitfn, kernel, bound_args, signature, constexprs, grid):
    """Best-effort, source-free task dump. Never raises into the user's launch path."""
    try:
        kernel_name = getattr(kernel.metadata, "name", getattr(jitfn, "__name__", "kernel"))
        # e.g. "^triton_tem_" keeps Inductor's final template kernels and drops its autotune candidates.
        pattern = os.environ.get("TRITON_COMPILE_IQ_COLLECT_KERNELS")
        if pattern and not re.search(pattern, kernel_name):
            dlog("collector", f"skip {kernel_name} (no match for TRITON_COMPILE_IQ_COLLECT_KERNELS)")
            return
        ptx = kernel.asm["ptx"]
        sha = ptx_sha256(ptx)
        tdir = os.path.join(task_root(), sha[:16])
        done = os.path.join(tdir, "spec.json")
        if os.path.exists(done):
            dlog("collector", f"skip {sha[:16]} (already collected)")
            return

        # constexpr path-tuples -> the set of constexpr arg names.
        names = list(bound_args.keys())
        constexpr_names = set()
        for path in (constexprs or {}):
            if isinstance(path, tuple) and len(path) == 1 and path[0] < len(names):
                constexpr_names.add(names[path[0]])

        ptxas = get_ptxas(kernel.metadata.target.arch)
        blobs = {}
        spec = ptx_launch.build_spec(ptx, kernel.metadata, tuple(grid),
                                     _ordered_args(bound_args, constexpr_names, blobs), ptxas.path, ptxas.version)
        spec.update(
            ptx_sha256=sha,
            kernel_name=kernel_name,
            fn_name=getattr(jitfn, "__name__", None),
        )

        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "kernel.ptx"), "w") as f:
            f.write(ptx)
        # Before spec.json: its existence marks the task complete (see the skip check above). Only
        # write blobs the spec references -- build_spec dedupes tensors by data_ptr, dropping the rest.
        for blob_sha in {t["pickle_sha256"] for t in spec["tensors"] if "pickle_sha256" in t}:
            with open(os.path.join(tdir, f"{blob_sha}.pkl"), "wb") as f:
                f.write(blobs[blob_sha])
        with open(done, "w") as f:
            json.dump(spec, f, indent=2, default=str)
        dlog(
            "collector", f"collected {sha[:16]} {spec['arch']} entry={spec['entry']} grid={spec['grid']} "
            f"tensors={len(spec['tensors'])} args={len(spec['args'])} -> {tdir}")

        # Level 1 (prototype): additionally capture the compiler-owned launcher + Level 0 schema when
        # the backend emits them, so the factory can drive triton_launch_<kernel>() (launch.h) rather
        # than the hand-rolled PTX-direct launcher (see ptx_anneal.level1). Best-effort and additive:
        # absence just means the factory falls back to spec.json. Isolated so it can never perturb the
        # already-written spec.json / kernel.ptx.
        try:
            asm = kernel.asm
            lm = asm["launch_metadata"] if "launch_metadata" in asm else None
            if lm is not None:
                with open(os.path.join(tdir, "launch_metadata.json"), "w") as f:
                    f.write(lm if isinstance(lm, str) else json.dumps(lm))
            ls = asm["launcher_src"] if "launcher_src" in asm else None
            if ls is not None:
                with open(os.path.join(tdir, "launcher.c"), "w") as f:
                    f.write(ls)
            if lm is not None or ls is not None:
                dlog(
                    "collector", f"captured level1 {sha[:16]} "
                    f"(launch_metadata={lm is not None}, launcher_src={ls is not None})")
        except Exception as e:
            dlog("collector", f"level1 capture skipped: {type(e).__name__}: {e}")
    except NotImplementedError as e:
        dlog("collector", f"skip (unsupported by PTX-direct spec): {e}")
    except Exception as e:  # collection must never break the user's run
        dlog("collector", f"capture failed: {type(e).__name__}: {e}")


def wrap_launcher(kernel, run):
    """Wrap a CompiledKernel's launcher so its first launch dumps a task.

    Covers kernels compiled with triton.compile() and launched through CompiledKernel.run (e.g.
    Inductor's generated launchers), which never reach the JITFunction.run hook. Callers may hold on
    to the returned function, so it stays installed and only the first launch collects. Launches
    inside benchmarking() are skipped, so autotune candidates that never run outside a benchmark are
    never collected."""
    collected = False

    def launcher(grid_0, grid_1, grid_2, stream, function, metadata, launch_metadata, enter_hook, exit_hook, *args):
        nonlocal collected
        if not collected and not getattr(_compile_iq_state, "suppressed", False):
            collected = True
            src = kernel.src
            fn = getattr(src, "fn", None)
            if fn is not None:
                capture(jitfn=fn, kernel=kernel, bound_args=dict(zip(fn.arg_names, args)), signature=src.signature,
                        constexprs=src.constants, grid=(grid_0, grid_1, grid_2))
        return run(grid_0, grid_1, grid_2, stream, function, metadata, launch_metadata, enter_hook, exit_hook, *args)

    return launcher
