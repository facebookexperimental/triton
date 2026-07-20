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
    """Map a PTX `.param` declaration to a launch-arg kind: 'tma' | 'ptr' | 'i64' | 'f64' | 'f32' | 'i32'."""
    if ".b8" in p and "[" in p:
        return "tma"  # a by-value byte-array param == a 128B CUtensorMap (TMA descriptor)
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
        ("tensor", {"id", "shape", "dtype", "strides"})
        ("tensordesc", {"base": {"id","shape","dtype","strides"}, "desc_shape", "desc_strides",
                        "block_shape", "padding"})   # host-side TMA TensorDescriptor
        ("scalar", value) | ("constexpr",)
    Applies Triton's specialization (drop constexprs and `equal_to_1` integer scalars), maps the
    survivors onto the PTX `.entry` param slots -- including expanding each TensorDescriptor into its
    [tensormap, *shape, *strides] params (the TMA ABI, see nvidia/backend/driver.py
    make_tensordesc_arg) -- and appends the two trailing null scratch pointers. Backing tensors are
    deduped by `id` so descriptors that share a tensor share one allocation.

    Cluster shape is read from ``metadata.ctas_per_cga`` ONLY: that is the sole field TLX kernels set to
    request a cluster (the CUDA-native path), and the nvidia backend mirrors it into ``cluster_dims`` in
    ``Metadata.__post_init__`` -- so ctas_per_cga is authoritative and cluster_dims/cluster are not read.

    Raises NotImplementedError for kernels this PTX-direct path does not cover yet (non-null
    global/profile scratch, multi-CTA/cluster, fp4-padded/im2col TMA, other arg types, or any param
    mismatch) so the caller fails open and simply does not collect that kernel."""
    name, kinds = parse_entry(ptx)
    gss = int(getattr(metadata, "global_scratch_size", 0) or 0)
    pss = int(getattr(metadata, "profile_scratch_size", 0) or 0)
    num_ctas = int(getattr(metadata, "num_ctas", 1) or 1)
    # Cluster shape from ctas_per_cga only (see docstring): the field TLX kernels populate; the backend
    # mirrors it into cluster_dims, so ctas_per_cga is authoritative.
    cluster = list(getattr(metadata, "ctas_per_cga", None) or [1, 1, 1])
    tdmeta = list(getattr(metadata, "tensordesc_meta", None) or [])
    if gss or pss:
        raise NotImplementedError(f"non-null scratch (global={gss} profile={pss})")
    # Cluster support is limited to the TLX / CUDA-native path: num_ctas == 1 with the cluster shape
    # carried by the cubin's `.reqnctapercluster` directive, so a plain launch with the total-CTA grid
    # re-forms the clusters (matches the frontend launcher, which sets NO CLUSTER_DIMENSION attr for
    # this path). The Triton num_ctas>1 path (grid multiplied by num_ctas + an explicit
    # CLUSTER_DIMENSION launch attr) is not covered.
    if num_ctas != 1:
        raise NotImplementedError(f"multi-CTA via num_ctas>1 (num_ctas={num_ctas})")
    cluster = [int(c) for c in cluster]
    cluster_size = 1
    for c in cluster:
        cluster_size *= c
    if len(kinds) < 2 or kinds[-2] != "ptr" or kinds[-1] != "ptr":
        raise NotImplementedError("expected two trailing scratch pointer params")
    user_kinds = kinds[:-2]  # drop global_scratch + profile_scratch

    tensors = []
    _id2idx = {}

    def _tensor_idx(info):
        key = info.get("id")
        if key is not None and key in _id2idx:
            return _id2idx[key]
        idx = len(tensors)
        tensors.append({"shape": list(info["shape"]), "dtype": info["dtype"], "strides": list(info["strides"])})
        if key is not None:
            _id2idx[key] = idx
        return idx

    # Specialize: drop constexprs and equal_to_1 int scalars (what Triton removes from kernel params).
    survivors = []
    for a in ordered_args:
        if a[0] == "constexpr":
            continue
        if a[0] in ("tensor", "tensordesc"):
            survivors.append(a)
        elif a[0] == "scalar":
            v = a[1]
            if isinstance(v, bool):
                survivors.append(("scalar", int(v)))
            elif isinstance(v, int) and v == 1:
                continue
            else:
                survivors.append(("scalar", v))
        else:
            raise NotImplementedError(f"unsupported arg kind {a[0]!r}")

    args, tensordescs, ki, tdmi = [], [], 0, 0

    def _kind_at(i):
        if i >= len(user_kinds):
            raise NotImplementedError("param parity: ran past PTX user params")
        return user_kinds[i]

    for s in survivors:
        if s[0] == "tensor":
            if _kind_at(ki) != "ptr":
                raise NotImplementedError("tensor arg mapped to a non-pointer PTX param")
            args.append({"t": _tensor_idx(s[1])})
            ki += 1
        elif s[0] == "scalar":
            k = _kind_at(ki)
            if k in ("ptr", "tma"):
                raise NotImplementedError("scalar arg mapped to a pointer/tma PTX param")
            args.append({k: int(s[1]) if k in ("i32", "i64") else float(s[1])})
            ki += 1
        else:  # tensordesc -> [tensormap, *desc_shape, *desc_strides]
            info = s[1]
            if tdmi >= len(tdmeta):
                raise NotImplementedError("more tensordesc args than tensordesc_meta entries")
            meta = tdmeta[tdmi]
            tdmi += 1
            if meta.get("fp4_padded") or meta.get("is_im2col"):
                raise NotImplementedError("fp4-padded / im2col TMA not supported yet")
            if _kind_at(ki) != "tma":
                raise NotImplementedError("tensordesc arg not aligned to a tensormap PTX param")
            rank = len(info["desc_shape"])
            args.append({"tma": len(tensordescs)})
            tensordescs.append({
                "base": _tensor_idx(info["base"]),
                "desc_shape": list(info["desc_shape"]),
                "desc_strides": list(info["desc_strides"]),
                "swizzle": int(meta["swizzle"]),
                "elem_size": int(meta["elem_size"]),
                "elem_type": int(meta["elem_type"]),
                "block_size": list(meta["block_size"]),
                "padding": int(info.get("padding", 0)),
            })
            ki += 1
            for d in range(rank):  # *shape params
                k = _kind_at(ki)
                args.append({k: int(info["desc_shape"][d])})
                ki += 1
            for d in range(rank):  # *stride params
                k = _kind_at(ki)
                args.append({k: int(info["desc_strides"][d])})
                ki += 1

    if ki != len(user_kinds):
        raise NotImplementedError(f"param parity: consumed {ki} != {len(user_kinds)} PTX user params")
    args += [{"null": True}, {"null": True}]  # global_scratch, profile_scratch

    g = [int(grid[0]), int(grid[1]) if len(grid) > 1 else 1, int(grid[2]) if len(grid) > 2 else 1]
    # A clustered launch re-forms clusters from the total-CTA grid via `.reqnctapercluster`; the driver
    # requires each grid dim to be a multiple of its cluster dim (as it was in the collected run).
    if cluster_size != 1 and any(gd % cd != 0 for gd, cd in zip(g, cluster, strict=True)):
        raise NotImplementedError(f"grid {g} not divisible by cluster {cluster}")
    spec = {
        "entry": name,
        "arch": parse_arch(ptx),
        "shared": int(getattr(metadata, "shared", 0) or 0),
        "block": [int(getattr(metadata, "num_warps", 1) or 1) * 32, 1, 1],
        "grid": g,
        "ptxas": ptxas,
        "tensors": tensors,
        "args": args,
    }
    if cluster_size != 1:
        spec["cluster"] = cluster
    if tensordescs:
        spec["tensordescs"] = tensordescs
    return spec


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
        """kernel_args: ordered list of ('ptr', addr) | ('i32', int) | ('i64', int) | ('f32', float) |
        ('null',) | ('tma', addr_of_128B_tensormap). Builds the void** kernelParams array Triton's
        launcher would build and calls cuLaunchKernel. For 'tma' the param IS passed by value (128
        bytes), so the kernelParams slot points straight at the tensormap bytes -- the caller must keep
        the backing PyCUtensorMap objects alive across the launch."""
        gx, gy, gz = (tuple(grid) + (1, 1, 1))[:3]
        bx, by, bz = (tuple(block) + (1, 1, 1))[:3]
        arr = (ctypes.c_void_p * len(kernel_args))()
        holders = []  # keep ctypes value objects alive until after launch
        for i, a in enumerate(kernel_args):
            kind = a[0]
            if kind == "tma":  # by-value tensormap: slot points directly at the 128B descriptor
                arr[i] = ctypes.c_void_p(int(a[1]))
                continue
            if kind == "ptr":
                h = ctypes.c_void_p(int(a[1]))
            elif kind == "null":
                h = ctypes.c_void_p(0)
            elif kind == "i32":
                h = ctypes.c_int32(int(a[1]))
            elif kind == "i64":
                h = ctypes.c_int64(int(a[1]))
            elif kind == "f32":
                h = ctypes.c_float(float(a[1]))
            else:
                raise ValueError(f"unsupported kernel arg kind: {kind}")
            holders.append(h)
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


def build_tensormaps(spec, tensors):
    """Build the 128B CUtensorMap for each spec tensordesc via Triton's tensormap encoder, and return
    (objs, addrs): the PyCUtensorMap objects (KEEP ALIVE across launch) and the address of each one's
    128-byte tensormap (id(obj) + (basicsize - 128)). Empty for non-TMA kernels."""
    tds = spec.get("tensordescs")
    if not tds:
        return [], []
    import triton
    try:
        from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST as _D2H
    except Exception:
        _D2H = {i: i for i in range(16)}
    fill = triton.runtime.driver.active.utils.fill_tma_descriptor_tiled
    objs, addrs = [], []
    for td in tds:
        base = tensors[td["base"]]
        tm = fill(base.data_ptr(), td["swizzle"], td["elem_size"], _D2H[td["elem_type"]], list(td["block_size"]),
                  list(td["desc_shape"]), list(td["desc_strides"]), td["padding"])
        objs.append(tm)
        addrs.append(id(tm) + (type(tm).__basicsize__ - 128))
    return objs, addrs


def kernel_args_from_spec(spec, tensors, tensormap_addrs=None):
    """Turn the spec's final arg list + allocated tensors (+ tensormap addrs) into launch() args."""
    tensormap_addrs = tensormap_addrs or []
    ka = []
    for a in spec["args"]:
        if "t" in a:
            ka.append(("ptr", tensors[a["t"]].data_ptr()))
        elif "tma" in a:
            ka.append(("tma", tensormap_addrs[a["tma"]]))
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
