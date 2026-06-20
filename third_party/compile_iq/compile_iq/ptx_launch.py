"""PTX-direct launch: compile a FIXED PTX with ptxas (optionally applying an ACF) and run the
resulting cubin via the CUDA driver API -- no Triton frontend recompile from source.

This is the core of the PTX-direct factory objective (an alternative to replay.py, which
recompiles the kernel from its Python source per candidate). Tuning here means:

    ptxas <fixed kernel.ptx> [--apply-controls=<acf>] -o cand.cubin   # the per-candidate compile
    cuModuleLoadData(cand.cubin) -> cuLaunchKernel(...)               # the objective body (run + time)

Because the PTX is frozen, what we tune is byte-identical to production and the whole
"did source recompile to the same PTX" parity problem disappears.

Launch ABI (discovered from third_party/nvidia/backend/driver.py): the kernel params are the
NON-constexpr, NON-`equal_to_1`-specialized scalar/pointer args in source order, followed by two
trailing internal pointers -- `global_scratch` then `profile_scratch` -- which are NULL when their
sizes are 0. The PTX `.visible .entry` param list is authoritative for the count/types; callers
build `kernel_args` to match and we assert parity against the PTX before launching.
"""

import ctypes
import re
import subprocess
import tempfile

# cuda-python is needed only to LAUNCH (driver API). Import lazily so the pure helpers
# (parse_entry / build_spec / ptxas_compile) can be used from the collection path -- which runs
# in the user's process and must stay dependency-light -- without requiring cuda-python.
try:
    from cuda.bindings import driver as _drv
except Exception:  # pragma: no cover - only the launch path needs it
    _drv = None


# --------------------------------------------------------------------------------------
# driver error handling
# --------------------------------------------------------------------------------------
def _chk(ret):
    """Unwrap a cuda.bindings (CUresult, *rest) tuple, raising on a non-success status."""
    err = ret[0]
    if err != _drv.CUresult.CUDA_SUCCESS:
        name = _drv.cuGetErrorString(err)[1]
        try:
            name = name.decode()
        except Exception:
            pass
        raise RuntimeError(f"CUDA driver error: {err} ({name})")
    rest = ret[1:]
    return rest[0] if len(rest) == 1 else rest


_INITED = False


def _ensure_init(device: int = 0):
    """cuInit once and make sure a CUDA context is current on this thread. We retain+bind the
    device PRIMARY context -- the same one torch uses -- so driver launches and torch tensors share
    a context (and so this works even before torch has run its first device op in a fresh process)."""
    global _INITED
    if _INITED:
        return
    if _drv is None:
        raise RuntimeError("cuda-python (cuda.bindings.driver) is required to launch cubins; "
                           "the launch path needs it even though build_spec/ptxas do not.")
    _chk(_drv.cuInit(0))
    cur = _drv.cuCtxGetCurrent()
    has_ctx = cur[0] == _drv.CUresult.CUDA_SUCCESS and int(getattr(cur[1], "_ptr", cur[1]) or 0) != 0
    if not has_ctx:
        dev = _chk(_drv.cuDeviceGet(device))
        ctx = _chk(_drv.cuDevicePrimaryCtxRetain(dev))
        _chk(_drv.cuCtxSetCurrent(ctx))
    _INITED = True


# --------------------------------------------------------------------------------------
# PTX parsing
# --------------------------------------------------------------------------------------
_ENTRY_RE = re.compile(r"\.visible\s+\.entry\s+([\w$]+)\s*\(([^)]*)\)", re.S)
_TARGET_RE = re.compile(r"\.target\s+(sm_\w+)")


def parse_arch(ptx: str) -> str:
    m = _TARGET_RE.search(ptx)
    if not m:
        raise ValueError("no .target sm_* found in PTX")
    return m.group(1)


def _param_kind(p: str) -> str:
    """Map a PTX `.param` declaration to a launch-arg kind: 'ptr' | 'i64' | 'f64' | 'f32' | 'i32'."""
    if ".ptr" in p:
        return "ptr"
    if ".u64" in p or ".s64" in p or ".b64" in p:
        return "i64"
    if ".f64" in p:
        return "f64"
    if ".f32" in p:
        return "f32"
    return "i32"  # .u32 / .s32 / .b32 / narrower ints (passed in a 32-bit slot)


def parse_entry(ptx: str):
    """Return (entry_name, [kind, ...]) for the single .visible .entry, kind per _param_kind."""
    m = _ENTRY_RE.search(ptx)
    if not m:
        raise ValueError("no .visible .entry found in PTX")
    name = m.group(1)
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    return name, [_param_kind(p) for p in params]


def build_spec(ptx: str, metadata, grid, ordered_args, ptxas: str) -> dict:
    """Build the source-free launch spec from a compiled kernel + its (bound-order) call args.

    ordered_args: list in source/bound order, each one of:
        ("tensor", {"shape", "dtype", "strides"}) | ("scalar", value) | ("constexpr",)
    Applies Triton's specialization (drop constexprs and `equal_to_1` integer scalars), maps the
    survivors onto the PTX `.entry` param slots, and appends the two trailing null scratch pointers.
    Raises NotImplementedError for kernels this PTX-direct prototype does not yet cover (non-null
    global/profile scratch, multi-CTA/cluster, tensordesc/other arg types, or any param mismatch) so
    the caller fails open and simply does not collect that kernel."""
    name, kinds = parse_entry(ptx)
    gss = int(getattr(metadata, "global_scratch_size", 0) or 0)
    pss = int(getattr(metadata, "profile_scratch_size", 0) or 0)
    num_ctas = int(getattr(metadata, "num_ctas", 1) or 1)
    cluster = list(getattr(metadata, "cluster_dims", getattr(metadata, "cluster", [1, 1, 1])) or [1, 1, 1])
    if gss or pss:
        raise NotImplementedError(f"non-null scratch (global={gss} profile={pss})")
    if num_ctas != 1 or any(int(c) != 1 for c in cluster):
        raise NotImplementedError(f"multi-CTA/cluster (num_ctas={num_ctas} cluster={cluster})")
    if len(kinds) < 2 or kinds[-2] != "ptr" or kinds[-1] != "ptr":
        raise NotImplementedError("expected two trailing scratch pointer params")
    user_kinds = kinds[:-2]  # drop global_scratch + profile_scratch

    survivors = []  # (sk, value) post-specialization, in param order
    for a in ordered_args:
        if a[0] == "constexpr":
            continue
        if a[0] == "tensor":
            survivors.append(("tensor", a[1]))
        elif a[0] == "scalar":
            v = a[1]
            if isinstance(v, bool):
                survivors.append(("scalar", int(v)))
            elif isinstance(v, int) and v == 1:
                continue  # equal_to_1 specialization: dropped from the kernel signature
            else:
                survivors.append(("scalar", v))
        else:
            raise NotImplementedError(f"unsupported arg kind {a[0]!r}")
    if len(survivors) != len(user_kinds):
        raise NotImplementedError(f"param parity: {len(survivors)} survivors != {len(user_kinds)} PTX user params")

    tensors, args = [], []
    for (sk, sv), k in zip(survivors, user_kinds):
        if sk == "tensor":
            if k != "ptr":
                raise NotImplementedError("tensor arg mapped to a non-pointer PTX param")
            args.append({"t": len(tensors)})
            tensors.append({"shape": list(sv["shape"]), "dtype": sv["dtype"], "strides": list(sv["strides"])})
        else:  # scalar
            if k == "ptr":
                raise NotImplementedError("scalar arg mapped to a pointer PTX param")
            args.append({k: int(sv) if k in ("i32", "i64") else float(sv)})
    args += [{"null": True}, {"null": True}]  # global_scratch, profile_scratch

    g = list(grid)
    return {
        "entry": name,
        "arch": parse_arch(ptx),
        "shared": int(getattr(metadata, "shared", 0) or 0),
        "block": [int(getattr(metadata, "num_warps", 1) or 1) * 32, 1, 1],
        "grid": [int(g[0]), int(g[1]) if len(g) > 1 else 1,
                 int(g[2]) if len(g) > 2 else 1],
        "ptxas": ptxas,
        "tensors": tensors,
        "args": args,
    }


# --------------------------------------------------------------------------------------
# ptxas
# --------------------------------------------------------------------------------------
def ptxas_compile(ptx: str, ptxas: str, arch: str | None = None, acf_path: str | None = None, extra_args=()) -> bytes:
    """Assemble PTX -> cubin with ptxas. If acf_path is given, append --apply-controls=<acf>
    (the consume mechanism). Raises RuntimeError with ptxas stderr on failure."""
    arch = arch or parse_arch(ptx)
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=True) as pf, \
            tempfile.NamedTemporaryFile(suffix=".cubin", delete=True) as cf:
        pf.write(ptx)
        pf.flush()
        cmd = [ptxas, f"-arch={arch}", pf.name, "-o", cf.name, *extra_args]
        if acf_path:
            cmd.append(f"--apply-controls={acf_path}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ptxas failed (rc={r.returncode}): {r.stderr.strip()}\n  cmd: {' '.join(cmd)}")
        with open(cf.name, "rb") as f:
            return f.read()


# --------------------------------------------------------------------------------------
# driver load + launch
# --------------------------------------------------------------------------------------
class LoadedKernel:
    """A cubin loaded into the current context, ready to launch."""

    def __init__(self, cubin: bytes, name: str, shared: int):
        _ensure_init()
        self.module = _chk(_drv.cuModuleLoadData(cubin))
        self.func = _chk(_drv.cuModuleGetFunction(self.module, name.encode()))
        self.shared = int(shared)
        if self.shared > 48 * 1024:  # opt-in to >48KB dynamic smem
            _chk(
                _drv.cuFuncSetAttribute(self.func,
                                        _drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        self.shared))

    def unload(self):
        if getattr(self, "module", None) is not None:
            _drv.cuModuleUnload(self.module)
            self.module = None

    @classmethod
    def from_cubin(cls, cubin: bytes, name: str, shared: int) -> "LoadedKernel":
        return cls(cubin, name, shared)

    def launch(self, grid, block, kernel_args, stream=0):
        """kernel_args: ordered list of ('ptr', int_addr) | ('i32', int) | ('i64', int) | ('null',).
        Builds the void** kernelParams array Triton's launcher would build and calls cuLaunchKernel."""
        gx, gy, gz = (tuple(grid) + (1, 1, 1))[:3]
        bx, by, bz = (tuple(block) + (1, 1, 1))[:3]
        holders = []  # keep ctypes objects alive until after launch
        for a in kernel_args:
            kind = a[0]
            if kind == "ptr":
                holders.append(ctypes.c_void_p(int(a[1])))
            elif kind == "null":
                holders.append(ctypes.c_void_p(0))
            elif kind == "i32":
                holders.append(ctypes.c_int32(int(a[1])))
            elif kind == "i64":
                holders.append(ctypes.c_int64(int(a[1])))
            elif kind == "f32":
                holders.append(ctypes.c_float(float(a[1])))
            else:
                raise ValueError(f"unsupported kernel arg kind: {kind}")
        arr = (ctypes.c_void_p * len(holders))()
        for i, h in enumerate(holders):
            arr[i] = ctypes.cast(ctypes.pointer(h), ctypes.c_void_p)
        _chk(_drv.cuLaunchKernel(self.func, gx, gy, gz, bx, by, bz, self.shared, stream, int(ctypes.addressof(arr)), 0))


def load_cubin(cubin: bytes, name: str, shared: int) -> LoadedKernel:
    """Load cubin bytes into the current CUDA context and return a launchable handle."""
    return LoadedKernel(cubin, name, shared)


# --------------------------------------------------------------------------------------
# Launch spec -- the self-contained description a PTX-direct run needs WITHOUT any source.
#
# This is what build_spec emits and what the collector (collector.py) writes as spec.json: the full
# launch ABI (entry/arch/shared/block/grid + the post-specialization arg order), so a run needs only
# kernel.ptx + spec.json -- no Python source. A spec is:
#   {
#     "entry": "matmul_kernel", "arch": "sm_100a", "shared": 32784, "block": [128,1,1],
#     "grid": [4,4,1],
#     "tensors": [{"shape":[256,256],"dtype":"float16","strides":[256,1]}, ...],
#     "args": [ {"t":0}, {"t":1}, {"t":2}, {"i32":256}, ..., {"null":true}, {"null":true} ],
#   }
# "args" is the FINAL kernel-param order (post equal_to_1 / constexpr specialization) including the
# two trailing null scratch params; {"t":i} is a pointer to tensors[i].
# --------------------------------------------------------------------------------------
_DT = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "float64": "float64",
    "int32": "int32",
    "int64": "int64",
    "int8": "int8",
}


def alloc_tensors(spec, device="cuda", seed=0):
    """Allocate the spec's tensors (deterministic, so baseline and ACF runs see identical inputs)."""
    import torch
    torch.manual_seed(seed)
    dts = {
        "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "float64": torch.float64,
        "int32": torch.int32, "int64": torch.int64, "int8": torch.int8
    }
    out = []
    for t in spec["tensors"]:
        dt = dts[t["dtype"]]
        x = torch.empty_strided(t["shape"], t["strides"], dtype=dt, device=device)
        x.normal_() if dt.is_floating_point else x.zero_()
        out.append(x)
    return out


def kernel_args_from_spec(spec, tensors):
    """Turn the spec's final arg list + allocated tensors into the launch() kernel_args."""
    ka = []
    for a in spec["args"]:
        if "t" in a:
            ka.append(("ptr", tensors[a["t"]].data_ptr()))
        elif a.get("null"):
            ka.append(("null", ))
        elif "i32" in a:
            ka.append(("i32", a["i32"]))
        elif "i64" in a:
            ka.append(("i64", a["i64"]))
        elif "f32" in a:
            ka.append(("f32", a["f32"]))
        else:
            raise ValueError(f"unsupported spec arg: {a}")
    return ka
