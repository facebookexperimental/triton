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


def _make_tensor(shape, strides, dt, device):
    t = torch.empty_strided(shape, strides, dtype=dt, device=device)
    t.normal_() if dt.is_floating_point else t.zero_()
    return t


def build_args(task: dict, device="cuda"):
    """Reconstruct positional args in bound order. Returns (args, tensors) where
    `tensors` is the list of underlying torch.Tensors (plain tensors + the base of each
    TensorDescriptor) used for correctness comparison. Handles TMA descriptors so
    autotuned/warp-specialized kernels (e.g. blackwell_gemm_ws) replay generically."""
    torch.manual_seed(0)
    args, tensors = [], []
    for a in task["args"]:
        kind = a["kind"]
        if kind == "tensor":
            t = _make_tensor(a["shape"], a["strides"], _DT[a["dtype"]], device)
            args.append(t)
            tensors.append(t)
        elif kind == "tensor_descriptor":
            from triton.tools.tensor_descriptor import TensorDescriptor
            base = _make_tensor(a["base_shape"], a["base_strides"], _DT[a["base_dtype"]], device)
            args.append(TensorDescriptor(base, a["base_shape"], a["base_strides"], a["block_shape"]))
            tensors.append(base)
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


def _alloc_fn(size, align, stream):  # TMA descriptor scratch allocator
    return torch.empty(size, dtype=torch.int8, device="cuda")


def _raw_jit(kernel):
    """The launchable JITFunction. If the loaded symbol is an Autotuner, use its `.fn`
    so the captured config (already in the args) is launched directly (no re-autotuning)."""
    return kernel.fn if hasattr(kernel, "configs") else kernel


def _launch_kwargs(task):
    """Triton launch options (distinct from the kernel's constexprs) needed when bypassing
    the autotuner. Clusters use `ctas_per_cga` (CUDA semantics; == cluster_dims), NOT
    `num_ctas` — TLX/cluster kernels reject num_ctas and require ctas_per_cga."""
    kw = {}
    if task.get("num_warps"):
        kw["num_warps"] = task["num_warps"]
    cd = tuple(task.get("cluster_dims") or ())
    if cd and any(d > 1 for d in cd):
        kw["ctas_per_cga"] = cd
    return kw


def run_once(kernel, task, args, acf_path: str | None):
    """Launch the kernel once with the captured grid; ACF applied via PTX_OPTIONS."""
    grid = tuple(task["grid"])
    kfn = _raw_jit(kernel)
    try:
        triton.set_allocator(_alloc_fn)
    except Exception:
        pass
    if acf_path:
        os.environ["PTX_OPTIONS"] = f"--apply-controls={acf_path}"
    else:
        os.environ.pop("PTX_OPTIONS", None)
    try:
        kfn[grid](*args, **_launch_kwargs(task))
        torch.cuda.synchronize()
    finally:
        os.environ.pop("PTX_OPTIONS", None)


def benchmark(kernel, task, args, acf_path, warmup=25, rep=100):
    kfn = _raw_jit(kernel)
    try:
        triton.set_allocator(_alloc_fn)
    except Exception:
        pass
    if acf_path:
        os.environ["PTX_OPTIONS"] = f"--apply-controls={acf_path}"
    kw = _launch_kwargs(task)
    try:
        return triton.testing.do_bench(lambda: kfn[tuple(task["grid"])](*args, **kw), warmup=warmup, rep=rep,
                                       return_mode="mean")
    finally:
        os.environ.pop("PTX_OPTIONS", None)
