"""Pinned-source decode comparisons, explicit contracts, numerical gate, GPU graph timing.

Framework imports are replaced by small host-side adapters for standalone source
loading. Kernel function bodies from the reference checkouts remain unchanged.
"""

import argparse
import ast
from enum import IntEnum
from functools import lru_cache
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import types

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = Path(os.environ.get("FP8_REFERENCE_ROOT", "/references"))


def load_source(name, path, namespace=None, omit_prefixes=()):
    """Only alter framework import nodes; keep source filenames/line numbers."""
    path = Path(path)
    tree = ast.parse(path.read_text(), filename=str(path))

    class FrameworkImports(ast.NodeTransformer):

        def visit_FunctionDef(self, node):
            return node  # never alter GPU or host function bodies

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ImportFrom(self, node):
            return None if node.module and node.module.startswith(tuple(omit_prefixes)) else node

        def visit_Import(self, node):
            return None if any(alias.name.startswith(tuple(omit_prefixes)) for alias in node.names) else node

        def visit_If(self, node):
            self.generic_visit(node)
            if not node.body:
                node.body = [ast.copy_location(ast.Pass(), node)]
            return node

    tree = FrameworkImports().visit(tree)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(namespace or {})
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@lru_cache(maxsize=1)
def tlx_source():
    return load_source("tlx_decode_candidate", ROOT / "third_party/tlx/tutorials/amd_pa_decode_fp8.py")


PLATFORM = types.SimpleNamespace(is_rocm=lambda: True, fp8_dtype=lambda: torch.float8_e4m3fn,
                                 is_device_capability_family=lambda _: False)


@lru_cache(maxsize=1)
def sglang_source():
    score = load_source("sglang_score_reference",
                        REFERENCES / "sglang/python/sglang/kernels/ops/attention/score_mod.py")
    return load_source(
        "sglang_decode_reference", REFERENCES / "sglang/python/sglang/kernels/ops/attention/decode_attention.py", {
            "is_hip":
            lambda: True,
            "is_gfx1250_supported":
            lambda: True,
            "current_arch":
            lambda: types.SimpleNamespace(num_sms=torch.cuda.get_device_properties(0).multi_processor_count),
            "is_gfx95_supported":
            lambda: False,
            "get_device_core_count":
            lambda _: torch.cuda.get_device_properties(0).multi_processor_count,
            "unpack_aux_tensors":
            score.unpack_aux_tensors,
            "envs":
            types.SimpleNamespace(SGLANG_MLA_DECODE_TUNE=types.SimpleNamespace(get=lambda: False),
                                  SGLANG_FORCE_LEAN_GRID_CU_MULT=types.SimpleNamespace(get=lambda: 1.0)),
        }, ("sglang", ))


@lru_cache(maxsize=2)
def vllm_source(unified=False):
    namespace = {
        "triton": triton, "tl": tl, "current_platform": PLATFORM, "init_logger": logging.getLogger, "on_gfx1100":
        lambda: False, "on_gfx1201": lambda: False, "envs": types.SimpleNamespace(VLLM_BATCH_INVARIANT=False)
    }
    if unified:
        helper = load_source("vllm_attention_helpers_reference",
                             REFERENCES / "vllm/vllm/v1/attention/ops/triton_attention_helpers.py", namespace,
                             ("vllm", ))
        namespace.update({name: value for name, value in vars(helper).items() if not name.startswith("__")})
        path = REFERENCES / "vllm/vllm/v1/kv_cache_interface.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KVQuantMode")
        enum_ns = {"IntEnum": IntEnum}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), enum_ns)
        namespace["KVQuantMode"] = enum_ns["KVQuantMode"]
    filename = "triton_unified_attention.py" if unified else "triton_decode_attention.py"
    return load_source("vllm_unified_reference" if unified else "vllm_decode_reference",
                       REFERENCES / "vllm/vllm/v1/attention/ops" / filename, namespace, ("vllm", ))


@lru_cache(maxsize=1)
def tokenspeed_source():
    from triton.experimental import gluon
    import triton.experimental.gluon.language as gl
    return load_source(
        "tokenspeed_decode_reference", REFERENCES /
        "tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/mha/decode.py",
        {"triton": triton, "tl": tl, "gluon": gluon, "gl": gl}, ("tokenspeed", ))


@lru_cache(maxsize=1)
def aiter_sources():
    import aiter
    import aiter.ops
    # Keep the image's compiled core; load pinned Python ops/configs, rather
    # than rebuilding unrelated AITER C++ modules just to import attention.
    for name in list(sys.modules):
        if name.startswith("aiter.ops.triton"):
            sys.modules.pop(name)
    aiter.ops.__path__.insert(0, str(REFERENCES / "aiter/aiter/ops"))
    from aiter.ops.triton.attention.unified_attention import unified_attention
    from aiter.ops.triton.attention.mha_fp8 import mha_fwd
    return unified_attention, mha_fwd


def inputs(args):
    torch.manual_seed(182)
    ps, hkv, dim = args.page, args.kv_heads, args.dim
    pages = triton.cdiv(args.length, ps)
    q = torch.randn(args.tokens, args.heads, dim, device="cuda") * 0.4
    if args.scales == "unit":
        ks = torch.ones(hkv, device="cuda")
        vs = torch.ones(hkv, device="cuda")
    elif args.scales == "shared":
        ks = torch.full((hkv, ), 0.05, device="cuda")
        vs = torch.full((hkv, ), 0.08, device="cuda")
    else:
        ks = torch.linspace(0.025, 0.08, hkv, device="cuda")
        vs = torch.linspace(0.04, 0.12, hkv, device="cuda").flip(0).contiguous()
    qs = torch.ones(args.heads, device="cuda") if args.scales == "unit" else torch.full(
        (args.heads, ), 0.04, device="cuda")
    q = (q / qs[None, :, None]).to(torch.float8_e4m3fn) if args.fp8_q else q.to(torch.bfloat16)
    k = torch.randn(pages, hkv, ps, dim, device="cuda") * 0.5
    v = torch.randn_like(k) * 0.8
    k = (k / ks[None, :, None, None]).to(torch.float8_e4m3fn)
    v = (v / vs[None, :, None, None]).to(torch.float8_e4m3fn)
    table = torch.randperm(pages, device="cuda").to(torch.int32)[None, :]
    kp = k.reshape(pages, hkv, ps, dim // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    vp = v.reshape(pages, hkv, ps // 16, 16, dim).permute(0, 1, 2, 4, 3).contiguous()
    return types.SimpleNamespace(q=q, k=k, v=v, kp=kp, vp=vp, table=table, ks=ks, vs=vs, qs=qs,
                                 contexts=torch.tensor([args.length], device="cuda", dtype=torch.int32))


def reference(x, args):
    # Chunk heads/queries so the 256K and T=8192 cases do not materialize a
    # full [T,H,L] probability tensor. This runs outside timed regions.
    page_ids = x.table[0].long()
    result = torch.empty_like(x.q, dtype=torch.float32)
    positions = torch.arange(args.length, device="cuda")
    for kvh in range(args.kv_heads):
        k = x.k[page_ids, kvh].float().reshape(-1, args.dim)[:args.length] * x.ks[kvh]
        v = x.v[page_ids, kvh].float().reshape(-1, args.dim)[:args.length] * x.vs[kvh]
        group = args.heads // args.kv_heads
        for first in range(0, args.tokens, 64):
            last = min(first + 64, args.tokens)
            qr = x.q[first:last, kvh * group:(kvh + 1) * group].float()
            if args.fp8_q:
                qr *= x.qs[kvh * group:(kvh + 1) * group][None, :, None]
            scores = torch.matmul(qr, k.T) * args.dim**-0.5
            qpos = args.length - args.tokens + torch.arange(first, last, device="cuda")
            if args.tokens == 1 or args.causal:
                scores = scores.masked_fill(positions[None, None, :] > qpos[:, None, None], -float("inf"))
            if args.window:
                scores = scores.masked_fill(positions[None, None, :] <= qpos[:, None, None] - args.window,
                                            -float("inf"))
            p = torch.softmax(scores, dim=-1).nan_to_num()
            result[first:last, kvh * group:(kvh + 1) * group] = torch.matmul(p, v)
    return result


def make_backend(name, x, a):
    tokens, hq, dim = x.q.shape
    out = torch.empty_like(x.q, dtype=torch.bfloat16)
    if name.startswith("tlx"):
        module = tlx_source()
        chosen_splits = a.splits
        if name == "tlx_auto":
            cfg = module.select_decode_config(x.q, x.kp, max_context_len=a.length, query_length=tokens,
                                              window_size=a.window)
            chosen_splits = cfg["num_splits"]
        kwargs = {
            "variant": name.removeprefix("tlx_"), "block_n": a.block_n, "block_m": a.block_m, "num_warps": a.warps
        }
        if name != "tlx_baseline":
            kwargs.update(
                out=out,
                workspace=(torch.empty((tokens, hq, chosen_splits, dim),
                                       device=x.q.device), torch.empty((tokens, hq, chosen_splits), device=x.q.device)))
        return lambda: module.fp8_kv_decode(x.q, x.kp, x.vp, x.contexts, x.table, x.ks, x.vs, max_context_len=a.length,
                                            q_scale=x.qs if a.fp8_q else None, query_length=tokens, causal=a.causal or
                                            tokens == 1, window_size=a.window, num_splits=None
                                            if name == "tlx_auto" else a.splits, **kwargs)
    if a.scales == "per-head" and not name.startswith("vllm_unified"):
        raise NotImplementedError("Reference accepts only scalar/unit scales; use a declared restricted control")
    if name in ("aiter_unified", "aiter_dense", "aiter_gluon2d"):
        ua, dense = aiter_sources()
        if name == "aiter_gluon2d":
            if not a.window or tokens != 1 or a.fp8_q:
                raise NotImplementedError("Forced mixed-dtype Gluon2D control covers BF16-Q single-token SWA")
            host_globals = dict(ua.__globals__)
            if host_globals.get("_unified_attention_kernel_2d_gfx1250") is None:
                raise NotImplementedError("Pinned Gluon2D source did not import")
            host_globals["is_2d_gluon_available"] = lambda params, backend: True
            original = ua
            ua = types.FunctionType(original.__code__, host_globals, original.__name__, original.__defaults__,
                                    original.__closure__)
            ua.__kwdefaults__ = original.__kwdefaults__
        if name == "aiter_dense":
            if not a.fp8_q or a.window:
                raise NotImplementedError("Dense reference requires FP8 Q and has no SWA API")
            kd = x.k[x.table[0].long()].permute(1, 0, 2, 3).reshape(1, a.kv_heads, -1,
                                                                    dim)[:, :, :a.length].contiguous()
            vd = x.v[x.table[0].long()].permute(1, 0, 2, 3).reshape(1, a.kv_heads, -1,
                                                                    dim)[:, :, :a.length].contiguous()
            qd = x.q.transpose(0, 1)[None].contiguous()
            od = torch.empty(qd.shape, device="cuda", dtype=torch.bfloat16)
            return lambda: dense(qd, kd, vd, x.qs[:1], x.ks[:1], x.vs[:1], o=od, causal=a.causal or tokens == 1
                                 ).squeeze(0).transpose(0, 1)
        if not a.causal and tokens != 1:
            raise NotImplementedError("AITER unified is causal-only")
        indptr = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
        return lambda: (ua(x.q, x.kp, x.vp, out, indptr, tokens, x.contexts, a.length, dim**-0.5, True,
                           (a.window - 1 if a.window else -1, 0), x.table, 0, x.qs[:1]
                           if a.fp8_q else None, x.ks[:1], x.vs[:1], shuffled_kv_cache=True, backend="gluon"), out)[1]
    if name == "tokenspeed":
        if not a.fp8_q or tokens != 1 or a.window or a.scales != "unit":
            raise NotImplementedError("TokenSpeed requires same-dtype QKV, unit scales, q_len=1 and no SWA")
        mod = tokenspeed_source()
        kn = x.k.permute(0, 2, 1, 3).contiguous()
        vn = x.v.permute(0, 2, 1, 3).contiguous()
        return lambda: mod.launch_gluon_mha_decode_gfx1250(x.q, kn, vn, x.table, x.contexts, a.length)
    if name.startswith("vllm_unified"):
        if a.fp8_q and a.scales == "per-head":
            raise NotImplementedError("Full-FP8 Q with per-token-head descales is not established")
        split_decode = name in ("vllm_unified_3d", "vllm_unified_td3d", "vllm_unified_3d_tuned", "vllm_unified_kvtd3d")
        td = name in ("vllm_unified_td2d", "vllm_unified_td3d", "vllm_unified_kvtd3d", "vllm_unified_kvtd2d")
        if split_decode and tokens != 1:
            raise NotImplementedError("This explicit3D adapter covers one query per request")
        m = vllm_source(unified=True)
        unified = m.unified_attention
        if name in ("vllm_unified_3d_tuned", "vllm_unified_kvtd3d", "vllm_unified_kvtd2d"):
            original_tile = m._get_tile_size

            def tile_config(head_size, sliding_window, element_size, is_prefill=False):
                return original_tile(head_size, sliding_window, element_size,
                                     is_prefill=True) if is_prefill and name != "vllm_unified_kvtd2d" else a.block_n

            host_globals = dict(unified.__globals__, _get_tile_size=tile_config)
            unified = types.FunctionType(unified.__code__, host_globals, unified.__name__, unified.__defaults__,
                                         unified.__closure__)
            unified.__kwdefaults__ = m.unified_attention.__kwdefaults__
            if name in ("vllm_unified_kvtd3d", "vllm_unified_kvtd2d"):
                # Only the Python launcher's existing Q/O flag changes. The
                # original GPU JIT functions in host_globals remain untouched.
                tree = ast.parse(Path(m.__file__).read_text())
                launch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "unified_attention")
                for node in ast.walk(launch):
                    if isinstance(node, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == "use_td_qo" for t in node.targets):
                        node.value = ast.copy_location(ast.Constant(False), node.value)
                exec(compile(ast.Module(body=[launch], type_ignores=[]), m.__file__, "exec"), host_globals)
                unified = host_globals["unified_attention"]
        kn = x.k.permute(0, 2, 1, 3).contiguous()
        vn = x.v.permute(0, 2, 1, 3).contiguous()
        ip = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
        per_head = a.scales == "per-head"
        mode = m.KVQuantMode.FP8_PER_TOKEN_HEAD if per_head else m.KVQuantMode.FP8_PER_TENSOR
        ksc = x.ks[None, None, :].expand(kn.shape[:3]) if per_head else None
        vsc = x.vs[None, None, :].expand(vn.shape[:3]) if per_head else None
        scratch = {}
        if split_decode:
            scratch = dict(
                seq_threshold_3D=max(1, 128 // a.kv_heads), num_par_softmax_segments=a.splits,
                softmax_segm_output=torch.empty((tokens, hq, a.splits, dim), device=x.q.device, dtype=torch.float32),
                softmax_segm_max=torch.empty((tokens, hq, a.splits), device=x.q.device,
                                             dtype=torch.float32), softmax_segm_expsum=torch.empty(
                                                 (tokens, hq, a.splits), device=x.q.device, dtype=torch.float32))
        invoke = lambda: (unified(x.q, kn, vn, out, ip, tokens, x.contexts, a.length, dim**-0.5, a.causal or tokens ==
                                  1, (a.window - 1 if a.window else -1, 0), x.table, 0, x.qs[:1]
                                  if a.fp8_q else None, x.ks[:1], x.vs[:1], kv_quant_mode=mode, k_scale_cache=ksc,
                                  v_scale_cache=vsc, use_td=td, **scratch), out)[1]
        tile = a.block_n if name in ("vllm_unified_3d_tuned",
                                     "vllm_unified_kvtd3d", "vllm_unified_kvtd2d") else m._get_tile_size(
                                         dim, a.window, x.q.element_size(), is_prefill=not split_decode)
        invoke.reference_config = dict(
            dispatch="3D split-KV" if split_decode else "2D", kv_tile=min(tile, a.page) if td else tile,
            segments=a.splits if split_decode else None, td_kv=td, td_qo=td
            and ((hq // a.kv_heads) & (hq // a.kv_heads - 1)) == 0 and dim == triton.next_power_of_2(dim)
            and name not in ("vllm_unified_kvtd3d", "vllm_unified_kvtd2d"), launcher_tuned=name
            in ("vllm_unified_3d_tuned", "vllm_unified_kvtd3d", "vllm_unified_kvtd2d"))
        return invoke
    if name in ("sglang", "sglang_lean", "vllm_decode"):
        if a.window or (tokens > 1 and a.causal):
            raise NotImplementedError("Single-query decode adapter does not cover causal multi-query/SWA")
        if a.fp8_q and not torch.all(x.qs == x.qs[0]).item():
            raise NotImplementedError("Legacy FP8-Q controls require a common scalar Q scale")
        kn = x.k.permute(0, 2, 1, 3).contiguous()
        vn = x.v.permute(0, 2, 1, 3).contiguous()
        table = x.table.expand(tokens, -1).contiguous()
        ctx = x.contexts.expand(tokens).contiguous()
        if name == "vllm_decode":
            # For these scalar controls QS is identical across query heads.
            # Folding it into sm_scale preserves the incoming FP8 Q payload.
            query_scale = float(x.qs[0]) if a.fp8_q else 1.0
            m = vllm_source()
            mid = torch.empty((tokens, hq, a.splits, dim + 1), device="cuda")
            lse = torch.empty((tokens, hq), device="cuda")
            return lambda: (m.decode_attention_fwd(x.q, kn, vn, out, lse, table, ctx, mid, a.splits, dim**-0.5 *
                                                   query_scale, a.page, k_scale=x.ks[:1], v_scale=x.vs[:1]), out)[1]
        query_scale = float(x.qs[0]) if a.fp8_q else 1.0
        m = sglang_source()
        locations = (table[:, :, None] * a.page + torch.arange(a.page, device="cuda")).reshape(
            tokens, -1)[:, :a.length].contiguous()
        ip = torch.arange(tokens + 1, device="cuda", dtype=torch.int32) * a.length
        mid = torch.empty((tokens, hq, a.splits, dim), device="cuda")
        lse = torch.empty((tokens, hq, a.splits), device="cuda")
        ns = torch.full((tokens, ), a.splits, dtype=torch.int32, device="cuda")
        kt = kn.reshape(-1, a.kv_heads, dim)
        vt = vn.reshape(-1, a.kv_heads, dim)
        kscalar, vscalar = float(x.ks[0]), float(x.vs[0])
        lean = {}
        if name == "sglang_lean":
            programs, _, _ = m._lean_decode_launch_params(a.kv_heads, hq // a.kv_heads)
            bm = m._LEAN_BLOCK_M
            lean = dict(lean_Mp=torch.empty((programs, bm), device=x.q.device), lean_Lp=torch.empty(
                (programs, bm), device=x.q.device), lean_Op=torch.empty((programs, bm, dim), device=x.q.device),
                        lean_locks=torch.zeros(programs, device=x.q.device, dtype=torch.int32))
        return lambda: (m.decode_attention_fwd(x.q, kt, vt, out, ip, locations.flatten(
        ), mid, lse, ns, a.splits, dim**-0.5 * query_scale, kscalar, vscalar, enable_lean=name == "sglang_lean", **lean
                                               ), out)[1]
    raise ValueError(name)


def tlx_resources():
    """Compiled specializations seen in this process; not hardware counters."""
    module = tlx_source()._optimized_module()
    records = []
    for name in ("_vector_decode", "_matrix_decode", "_merge"):
        fn = getattr(module, name)
        for device_data in fn.device_caches.values():
            for kernel in device_data[0].values():
                meta = kernel.metadata
                records.append(
                    dict(kernel=name, n_regs=getattr(kernel, "n_regs",
                                                     None), n_spills=getattr(kernel, "n_spills", None),
                         **{k: getattr(meta, k, None)
                            for k in ("num_warps", "waves_per_eu", "shared")}))
    return records


def timing(fn, samples, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    measurements = []
    for _ in range(samples):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        measurements.append(start.elapsed_time(stop) * 1000 / repeats)
    caller = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        caller.append((time.perf_counter_ns() - start) / 1000)
    return {
        "graph_batch_mean_us": measurements, "median_us": statistics.median(measurements), "p95_batch_mean_us":
        sorted(measurements)[min(len(measurements) - 1, int(len(measurements) * 0.95))], "caller_median_us":
        statistics.median(caller), "graph_repeats": repeats, "timing_scope":
        "GPU event time per invocation within a graph; p95 is across batch means"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["tlx_baseline"])
    parser.add_argument("--length", type=int, default=2304)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--page", type=int, default=64)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--fp8-q", action="store_true")
    parser.add_argument("--scales", choices=["unit", "shared", "per-head"], default="per-head")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.heads % args.kv_heads == 0
    x = inputs(args)
    oracle = reference(x, args)
    records = []
    for name in args.backends:
        row = {
            "backend": name, "workload": {k: v
                                          for k, v in vars(args).items()
                                          if k not in ("output", "backends")}, "torch": torch.__version__, "triton":
            triton.__version__, "references_root": str(REFERENCES), "device": str(torch.cuda.get_device_properties(0))
        }
        try:
            if name == "tlx_auto":
                row["selected_config"] = tlx_source().select_decode_config(x.q, x.kp, max_context_len=args.length,
                                                                           query_length=args.tokens,
                                                                           window_size=args.window)
            if name.startswith("vllm_unified"):
                row["dispatch"] = ("3D split-KV" if name in ("vllm_unified_3d", "vllm_unified_td3d",
                                                             "vllm_unified_3d_tuned", "vllm_unified_kvtd3d") else "2D")
                row["decode_tile_override"] = args.block_n if name in ("vllm_unified_3d_tuned", "vllm_unified_kvtd3d",
                                                                       "vllm_unified_kvtd2d") else None
                row["tensor_descriptors"] = name in ("vllm_unified_td2d", "vllm_unified_td3d", "vllm_unified_kvtd3d",
                                                     "vllm_unified_kvtd2d")
                row["qo_descriptors_disabled"] = name in ("vllm_unified_kvtd3d", "vllm_unified_kvtd2d")
            fn = make_backend(name, x, args)
            if hasattr(fn, "reference_config"):
                row["reference_config"] = fn.reference_config
            actual = fn()
            torch.cuda.synchronize()
            error = actual.float() - oracle
            row.update(max_abs_error=float(error.abs().max()), rms_error=float(error.square().mean().sqrt()))
            try:
                torch.testing.assert_close(actual.float(), oracle, atol=5e-4, rtol=1e-2)
                row.update(status="pass", performance_eligible=True)
            except AssertionError as exc:
                row.update(status="failed_accuracy", performance_eligible=False, reason=str(exc))
            # Failed numerical gates remain visibly ineligible. Their latency
            # is useful for diagnosing the accuracy/performance tradeoff.
            row.update(timing(fn, args.samples, args.repeats))
            if name.startswith("tlx") and name != "tlx_baseline":
                try:
                    row["compiled_specializations_seen"] = tlx_resources()
                except Exception as exc:
                    row["resource_metadata_note"] = str(exc)
        except NotImplementedError as exc:
            row.update(status="unsupported", reason=str(exc))
        except Exception as exc:
            row.update(status="failed", reason=str(exc), traceback=traceback.format_exc())
        records.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("traceback", "device", "workload")}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
