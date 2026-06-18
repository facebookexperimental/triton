"""Stage 2a — replay. Re-run a collected kernel from its on-disk task (source + args + grid) and
benchmark it, recompiling through Triton so the run is representative of the real kernel. The ACF
(when tuning) is applied via the `ptx_options` launch kwarg (-> opt.ptx_options -> ptxas), the same
path the consume hook uses. Naive (non-warp-specialized) kernels only; WS/TMA replay is a later diff.
"""

import importlib.util
import json
import os

import torch
import triton  # noqa: F401

_DT = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
}


def load_task(task_dir: str) -> dict:
    with open(os.path.join(task_dir, "task.json")) as f:
        return json.load(f)


def load_kernel(task_dir: str, task: dict):
    src = os.path.join(task_dir, task.get("source_file", "source.py"))
    spec = importlib.util.spec_from_file_location("ciq_replay_src", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, task["fn_name"])


def _make_tensor(shape, strides, dt, device):
    t = torch.empty_strided(shape, strides, dtype=dt, device=device)
    t.normal_() if dt.is_floating_point else t.zero_()
    return t


def build_args(task: dict, device="cuda"):
    """Reconstruct positional args in bound order. Returns (args, tensors) where `tensors` is the
    list of torch.Tensors used for the self-consistency comparison."""
    torch.manual_seed(0)
    args, tensors = [], []
    for a in task["args"]:
        kind = a["kind"]
        if kind == "tensor":
            t = _make_tensor(a["shape"], a["strides"], _DT[a["dtype"]], device)
            args.append(t)
            tensors.append(t)
        elif kind in ("scalar", "constexpr"):
            args.append(a["value"])
        else:
            raise NotImplementedError(f"arg kind {kind}")
    return args, tensors


def _ptxas_ge_133(p: str) -> bool:
    import re
    import subprocess
    try:
        out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+)\.(\d+)", out)
        return bool(m) and (int(m.group(1)), int(m.group(2))) >= (13, 3)
    except Exception:
        return False


def find_ptxas() -> str | None:
    """Locate a ptxas >= 13.3 (needed for --apply-controls). Resolution order: explicit env
    (trusted) > the pip `nvidia-cuda-nvcc` wheel > PATH; non-env candidates are gated to >= 13.3."""
    import glob
    import importlib.util
    import shutil

    p = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH") or os.environ.get("TRITON_PTXAS_PATH")
    if p and os.path.exists(p):
        return p  # explicit choice, trusted as-is
    cands = []
    spec = importlib.util.find_spec("nvidia")  # nvidia-cuda-nvcc -> nvidia/cuNN/bin/ptxas
    for root in (spec.submodule_search_locations or []) if spec else []:
        cands += sorted(glob.glob(os.path.join(root, "cu*/bin/ptxas")), reverse=True)
    if (w := shutil.which("ptxas")):
        cands.append(w)
    return next((c for c in cands if _ptxas_ge_133(c)), None)


def _set_ptxas(task):
    p = (os.environ.get("TRITON_PTXAS_BLACKWELL_PATH") or find_ptxas() or task.get("ptxas_path") or "")
    if p:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = p
        os.environ["TRITON_PTXAS_PATH"] = p
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"  # force recompile so each candidate's ACF is applied


def _launch_kwargs(acf_path):
    # Apply the ACF via the per-compile `ptx_options` launch kwarg -> opt.ptx_options -> ptxas. This
    # is the SAME mechanism the consume hook uses (it appends --apply-controls to ptx_extra_options),
    # so what the factory benchmarks is byte-identical to what consume produces. (The PTXAS_OPTIONS
    # env can't be used: Triton reads it into a knob once at import, so it can't vary per candidate.)
    return {"ptx_options": f"--apply-controls={acf_path}"} if acf_path else {}


def run_once(kernel, task, args, acf_path=None):
    grid = tuple(task["grid"])
    kernel[grid](*args, **_launch_kwargs(acf_path))
    torch.cuda.synchronize()


def benchmark(kernel, task, args, acf_path=None, warmup=25, rep=100):
    grid = tuple(task["grid"])
    kw = _launch_kwargs(acf_path)
    return triton.testing.do_bench(lambda: kernel[grid](*args, **kw), warmup=warmup, rep=rep, return_mode="mean")
