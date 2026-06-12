"""Stage 2a — replay. Re-run a collected kernel from its task, optionally applying
an ACF, and benchmark it. Decoupled from the live user process: works purely from
the on-disk task (source + args + grid + ptx), recompiling deterministically (the
regenerated PTX matches the captured ptx_sha) with the ACF applied at ptxas.
"""

import importlib.util
import json
import os

import torch
import triton  # noqa: F401


def load_task(task_dir: str) -> dict:
    with open(os.path.join(task_dir, "task.json")) as f:
        return json.load(f)


def load_kernel(task_dir: str, task: dict):
    src = os.path.join(task_dir, task.get("source_file", "source.py"))
    spec = importlib.util.spec_from_file_location("ciq_replay_src", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, task["fn_name"])


_DT = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
}


def build_args(task: dict, device="cuda"):
    """Reconstruct positional args in bound order. Returns (args, tensor_indices)."""
    torch.manual_seed(0)
    args, tensor_idx = [], []
    for i, a in enumerate(task["args"]):
        kind = a["kind"]
        if kind == "tensor":
            dt = _DT[a["dtype"]]
            if dt.is_floating_point:
                t = torch.randn(a["shape"], device=device, dtype=dt)
            else:
                t = torch.zeros(a["shape"], device=device, dtype=dt)
            args.append(t)
            tensor_idx.append(i)
        elif kind in ("scalar", "constexpr"):
            args.append(a["value"])
        else:
            raise NotImplementedError(f"arg kind {kind} (tensor_descriptor replay TODO)")
    return args, tensor_idx


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
    """Locate a ptxas >= 13.3 (needed for --apply-controls). Resolution order:
    explicit env (trusted) > the pip `nvidia-cuda-nvcc` wheel > PATH; candidates
    other than the env are version-gated to >= 13.3. Repro is then a one-liner:
    `pip install nvidia-cuda-nvcc` (no long path / env var needed)."""
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
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"


def run_once(kernel, task, args, acf_path: str | None):
    """Launch the kernel once with the captured grid; ACF applied via PTX_OPTIONS."""
    grid = tuple(task["grid"])
    if acf_path:
        os.environ["PTX_OPTIONS"] = f"--apply-controls={acf_path}"
    else:
        os.environ.pop("PTX_OPTIONS", None)
    try:
        kernel[grid](*args)
        torch.cuda.synchronize()
    finally:
        os.environ.pop("PTX_OPTIONS", None)


def benchmark(kernel, task, args, acf_path, warmup=25, rep=100):
    if acf_path:
        os.environ["PTX_OPTIONS"] = f"--apply-controls={acf_path}"
    try:
        return triton.testing.do_bench(lambda: kernel[tuple(task["grid"])](*args), warmup=warmup, rep=rep,
                                       return_mode="mean")
    finally:
        os.environ.pop("PTX_OPTIONS", None)
