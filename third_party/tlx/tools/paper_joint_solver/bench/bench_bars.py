"""Bar-based benchmark runner replicating the paper's Blackwell FMHA evaluation.

Config: fp16, non-causal, BATCH=4, NUM_HEADS=32, HEAD_DIM=128, seqlens
{2048, 4096, 8192, 16384}.  FLOPs follow the FA convention used by
third_party/tlx/tutorials/testing/test_blackwell_fa_perf.py:
flops_per_matmul = 2*B*H*S^2*D, fwd = 2x, bwd = 5x (so fwd = 4*B*H*S^2*D,
bwd = 10*B*H*S^2*D).

Run from the paper_joint_solver directory with the MAIN venv python and
LD_LIBRARY_PATH unset:

    env -u LD_LIBRARY_PATH ../../../../.venv/bin/python -m bench.bench_bars \
        --bars triton_ws_on,tlx_default,jos --mode fwd --out bench/results_fwd.json

The `fa4` / `fa4_bwd` bars shell out to bench/fa4_worker.py under the separate
FA4 venv (set env FA4_PYTHON to its python binary).

GPU clocks cannot be locked (non-root); instead
`nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu` is recorded
before/after each bar into the JSON as `env_probe`.

Backward timing for cudnn/fa4: the full fwd+bwd is timed, fwd alone is timed
(autograd graph still built), and (total - fwd median) is reported.  See
bench/README.md for all deviation notes.
"""

import argparse
import gc
import importlib.util
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.testing

BENCH_DIR = Path(__file__).resolve().parent
PKG_DIR = BENCH_DIR.parent  # paper_joint_solver/
TOOLS_DIR = PKG_DIR.parent  # third_party/tlx/tools/
REPO_ROOT = TOOLS_DIR.parents[2]
EXAMPLES = TOOLS_DIR / "sched2tlx" / "examples"
TUTORIAL_FA = REPO_ROOT / "python" / "tutorials" / "06-fused-attention.py"
FA4_PYTHON = Path(os.environ.get("FA4_PYTHON", ""))  # FA4 venv python

BATCH = 4
NUM_HEADS = 32
HEAD_DIM = 128
SEQLENS = [2048, 4096, 8192, 16384]
DTYPE = torch.float16
LOG2E = 1.4426950408889634
QUANTILES = [0.5, 0.2, 0.8]
EXTERNAL_BARS = {"fa4", "fa4_bwd"}

_MOD_CACHE = {}


@dataclass
class ShapeCfg:
    batch: int
    heads: int
    seqlen: int
    head_dim: int
    sm_scale: float


class BarUnavailable(Exception):
    pass


def total_flops(mode, seqlen):
    per_matmul = 2.0 * BATCH * NUM_HEADS * seqlen * seqlen * HEAD_DIM
    return 2 * per_matmul if mode == "fwd" else 5 * per_matmul


def _alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _load_module(path, name=None):
    path = Path(path)
    key = str(path)
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    if not path.exists():
        raise BarUnavailable(f"missing file: {path}")
    name = name or f"bench_{path.parent.name}_{path.stem}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOD_CACHE[key] = mod
    return mod


def _rel(a, b):
    return (a.float() - b.float()).abs().max().item() / max(
        b.float().abs().max().item(), 1e-9)


def _sdpa(q, k, v, scale):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                            scale=scale)


def _make_qkv(seqlen, requires_grad=False):
    torch.manual_seed(0)
    ts = [
        torch.randn(BATCH, NUM_HEADS, seqlen, HEAD_DIM, device="cuda",
                    dtype=DTYPE) for _ in range(3)
    ]
    if requires_grad:
        for t in ts:
            t.requires_grad_(True)
    return ts


def _nvsmi():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10)
        return proc.stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


# ── forward bars ─────────────────────────────────────────────────────────────


# On this beta build the tutorial's warp_specialize=True path is broken in the
# stock (upstream OAI) AutoWS pipeline: every autotune config dies in
# TritonGPULoadMMASpecialization with "'ttng.tmem_alloc' op operation destroyed
# but still has uses".  Under TRITON_USE_META_WS=1 (this repo's Meta WS
# pipeline) the configs below compile and pass correctness (B200-probed);
# BLOCK_M=64 configs fail ("Only supported for scales as we pad the
# allocation"), num_stages=2 fails ("pipeliner doesn't know how to predicate"),
# and num_warps=4 exceeds the thread budget imposed by the tutorial's
# maxnreg=168.  triton_ws_on therefore runs with Meta WS and this pruned
# autotune list; triton_ws_off runs the stock pipeline and full list.
_WS_SAFE_CONFIGS = [(128, 64, 3, 8), (128, 64, 4, 8), (128, 32, 3, 8)]


def build_triton(cfg, warp_specialize):
    # python/tutorials/06-fused-attention.py; tri_out layout is (Z, H, N, D).
    tut = _load_module(TUTORIAL_FA, "bench_fused_attention_tutorial")
    at = tut._attn_fwd  # Autotuner around the fwd kernel
    if not hasattr(at, "_bench_orig_configs"):
        at._bench_orig_configs = list(at.configs)
    if warp_specialize:
        os.environ["TRITON_USE_META_WS"] = "1"
        at.configs = [
            triton.Config(dict(BLOCK_M=bm, BLOCK_N=bn), num_stages=s,
                          num_warps=w, pre_hook=tut._host_descriptor_pre_hook)
            for bm, bn, s, w in _WS_SAFE_CONFIGS
        ]
    else:
        os.environ.pop("TRITON_USE_META_WS", None)
        at.configs = list(at._bench_orig_configs)
    at.cache.clear()
    q, k, v = _make_qkv(cfg.seqlen)

    def fn():
        return tut.attention(q, k, v, False, cfg.sm_scale, warp_specialize)

    def check():
        out = fn()
        torch.cuda.synchronize()
        rel = _rel(out, _sdpa(q, k, v, cfg.sm_scale))
        return None if rel < 1e-2 else f"fwd rel {rel:.2e} >= 1e-2"

    return fn, check


def build_triton_tiled(cfg):
    # Plain sub-tiled kernel; launch recipe from its own run() (SUB_M=64).
    mod = _load_module(EXAMPLES / "case3_FA_fp16_subtiled" /
                       "fa_fwd_nows_subtiled.py")
    kernel = mod.fa_fwd_kernel_nows_subtiled
    q, k, v = _make_qkv(cfg.seqlen)
    bh = cfg.batch * cfg.heads
    qf, kf, vf = (t.contiguous().view(-1, cfg.head_dim) for t in (q, k, v))
    of = torch.empty_like(qf)
    m_lse = torch.empty(bh, cfg.seqlen, device="cuda", dtype=torch.float32)
    SUB_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(cfg.seqlen, 2 * SUB_M), bh)

    def fn():
        kernel[grid](qf, kf, vf, of, m_lse, cfg.sm_scale, bh, cfg.seqlen,
                     SUB_M=SUB_M, BLOCK_N=BLOCK_N, HEAD_DIM=cfg.head_dim,
                     num_warps=4, num_ctas=1, num_stages=2, maxRegAutoWS=152)

    def check():
        fn()
        torch.cuda.synchronize()
        out = of.view(cfg.batch, cfg.heads, cfg.seqlen, cfg.head_dim)
        rel = _rel(out, _sdpa(q, k, v, cfg.sm_scale))
        return None if rel < 1e-2 else f"fwd rel {rel:.2e} >= 1e-2"

    return fn, check


def build_tlx_fwd(cfg, path, kernel_name):
    # sched2tlx-emitted fwd kernel; tensor prep from case3_FA_fp16's
    # fa_fwd_nows_fp16.py run(); grid = (cdiv(N, 128), Z*H).  Generated
    # kernels pipeline explicitly, so the launch follows
    # case4_FA_bwd/run_generated.py (num_stages=1, no maxRegAutoWS).
    mod = _load_module(path)
    kernel = getattr(mod, kernel_name, None)
    if kernel is None:
        raise BarUnavailable(f"{path} has no kernel {kernel_name}")
    q, k, v = _make_qkv(cfg.seqlen)
    bh = cfg.batch * cfg.heads
    qf, kf, vf = (t.contiguous().view(-1, cfg.head_dim) for t in (q, k, v))
    of = torch.empty_like(qf)
    m_lse = torch.empty(bh, cfg.seqlen, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(cfg.seqlen, 128), bh)

    def fn():
        kernel[grid](qf, kf, vf, of, m_lse, cfg.sm_scale, bh, cfg.seqlen,
                     num_warps=4, num_ctas=1, num_stages=1)

    def check():
        fn()
        torch.cuda.synchronize()
        out = of.view(cfg.batch, cfg.heads, cfg.seqlen, cfg.head_dim)
        rel = _rel(out, _sdpa(q, k, v, cfg.sm_scale))
        return None if rel < 1e-2 else f"fwd rel {rel:.2e} >= 1e-2"

    return fn, check


def build_skc(cfg, path):
    # SKC skeleton instance (skc/ package); (Z, H, N, D) tensors like the
    # tutorial bars, attention() returns (o, M_lse).
    mod = _load_module(path)
    q, k, v = _make_qkv(cfg.seqlen)

    def fn():
        return mod.attention(q, k, v, cfg.sm_scale)

    def check():
        out, _ = fn()
        torch.cuda.synchronize()
        rel = _rel(out, _sdpa(q, k, v, cfg.sm_scale))
        return None if rel < 1e-2 else f"fwd rel {rel:.2e} >= 1e-2"

    return fn, check


def build_cudnn(cfg):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q, k, v = _make_qkv(cfg.seqlen)

    def fn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return _sdpa(q, k, v, cfg.sm_scale)

    def check():
        out = fn()
        torch.cuda.synchronize()
        rel = _rel(out, _sdpa(q, k, v, cfg.sm_scale))
        return None if rel < 1e-2 else f"fwd rel {rel:.2e} >= 1e-2"

    return fn, check


# ── backward bars ────────────────────────────────────────────────────────────


def build_cudnn_bwd(cfg):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q, k, v = _make_qkv(cfg.seqlen, requires_grad=True)
    do = torch.randn_like(q)

    def total():
        q.grad = k.grad = v.grad = None
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            o = _sdpa(q, k, v, cfg.sm_scale)
        o.backward(do)

    def fwd_only():
        # graph is still built (inputs require grad) so total - fwd isolates
        # the backward
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            _sdpa(q, k, v, cfg.sm_scale)

    def check():
        total()
        torch.cuda.synchronize()
        got = [t.grad.clone() for t in (q, k, v)]
        q.grad = k.grad = v.grad = None
        o = _sdpa(q, k, v, cfg.sm_scale)  # default-backend reference grads
        o.backward(do)
        rels = [_rel(g, t.grad) for g, t in zip(got, (q, k, v))]
        q.grad = k.grad = v.grad = None
        bad = max(rels)
        if bad >= 3e-2:
            return ("grad rel dQ={:.2e} dK={:.2e} dV={:.2e} (>= 3e-2)".format(
                *rels))
        return None

    return total, check, fwd_only


def build_skc_bwd(cfg, path):
    # SKC bwd skeleton instance; same data prep / conventions as
    # build_tlx_bwd (see its docstring), entry is bwd_attention.
    mod = _load_module(path)
    fn_entry = getattr(mod, "bwd_attention", None)
    if fn_entry is None:
        raise BarUnavailable(f"{path} has no bwd_attention")
    bh = cfg.batch * cfg.heads
    s = cfg.seqlen
    torch.manual_seed(0)
    q = (torch.randn(bh, s, cfg.head_dim, device="cuda", dtype=DTYPE) *
         cfg.sm_scale).detach().requires_grad_(True)
    k = torch.randn(bh, s, cfg.head_dim, device="cuda",
                    dtype=DTYPE).detach().requires_grad_(True)
    v = torch.randn(bh, s, cfg.head_dim, device="cuda",
                    dtype=DTYPE).detach().requires_grad_(True)
    do = torch.randn(bh, s, cfg.head_dim, device="cuda", dtype=DTYPE)

    o = _sdpa(q.view(bh, 1, s, cfg.head_dim), k.view(bh, 1, s, cfg.head_dim),
              v.view(bh, 1, s, cfg.head_dim), 1.0).view(bh, s, cfg.head_dim)
    o.backward(do)
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad

    M = _lse_base2(q.detach(), k.detach())
    D = (do.float() * o.detach().float()).sum(-1).contiguous()
    qf, kf, vf, dof = (t.detach().contiguous() for t in (q, k, v, do))
    dq = torch.zeros_like(qf)
    dk = torch.empty_like(kf)
    dv = torch.empty_like(vf)

    def fn():
        fn_entry(qf, kf, vf, dof, dq, dk, dv, M, D, s)

    def check():
        dq.zero_()
        fn()
        torch.cuda.synchronize()
        rq, rk, rv = _rel(dq, ref_dq), _rel(dk, ref_dk), _rel(dv, ref_dv)
        if max(rq, rk, rv) >= 3e-2:
            return (f"grad rel dQ={rq:.2e} dK={rk:.2e} dV={rv:.2e}"
                    " (>= 3e-2)")
        return None

    return fn, check


def _lse_base2(q, k, row_chunk=1024, bh_chunk=8):
    # logsumexp(q @ k^T) * log2(e), chunked to stay linear in memory.
    bh, s, _ = q.shape
    out = torch.empty(bh, s, device=q.device, dtype=torch.float32)
    for b in range(0, bh, bh_chunk):
        kb = k[b:b + bh_chunk].float().transpose(-1, -2)
        for i in range(0, s, row_chunk):
            sblk = torch.matmul(q[b:b + bh_chunk, i:i + row_chunk].float(), kb)
            out[b:b + bh_chunk, i:i + row_chunk] = \
                torch.logsumexp(sblk, dim=-1) * LOG2E
    return out.contiguous()


def build_tlx_bwd(cfg, path):
    # Base-2 convention from case4_FA_bwd/run_handwritten_nows.py: sm_scale is
    # pre-folded into Q and the kernel applies no softmax scale, so the
    # reference forward is plain softmax(q_scaled @ k^T) == SDPA(scale=1.0).
    # M = logsumexp * log2(e), D = rowsum(dO*O); both precomputed here (the
    # timed bar is the fused dK/dV/dQ kernel only — see README).
    mod = _load_module(path)
    kernel = getattr(mod, "fa_bwd_dkdv_5mma", None)
    if kernel is None:
        raise BarUnavailable(f"{path} has no kernel fa_bwd_dkdv_5mma")
    bh = cfg.batch * cfg.heads
    s = cfg.seqlen
    torch.manual_seed(0)
    q = (torch.randn(bh, s, cfg.head_dim, device="cuda", dtype=DTYPE) *
         cfg.sm_scale).detach().requires_grad_(True)
    k = torch.randn(bh, s, cfg.head_dim, device="cuda",
                    dtype=DTYPE).detach().requires_grad_(True)
    v = torch.randn(bh, s, cfg.head_dim, device="cuda",
                    dtype=DTYPE).detach().requires_grad_(True)
    do = torch.randn(bh, s, cfg.head_dim, device="cuda", dtype=DTYPE)

    # SDPA on 3D tensors falls back to the math backend (O(S^2) memory: 128 GiB
    # at S=16384); run it as 4D (BH, 1, S, D) so the flash backend applies.
    o = _sdpa(q.view(bh, 1, s, cfg.head_dim), k.view(bh, 1, s, cfg.head_dim),
              v.view(bh, 1, s, cfg.head_dim), 1.0).view(bh, s, cfg.head_dim)
    o.backward(do)
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad

    M = _lse_base2(q.detach(), k.detach())
    D = (do.float() * o.detach().float()).sum(-1).contiguous()
    qf, kf, vf, dof = (t.detach().contiguous() for t in (q, k, v, do))
    dq = torch.zeros_like(qf)  # TMA reduce-add accumulates
    dk = torch.empty_like(kf)
    dv = torch.empty_like(vf)

    stride = cfg.head_dim  # per-(b,h) [N_CTX, HEAD_DIM] row stride
    grid = (s // 128, bh)

    def fn():
        # Timed reps re-accumulate into a stale dq; values are irrelevant to
        # timing, correctness runs on a freshly zeroed dq in check().
        kernel[grid](qf, kf, vf, dof, dq, dk, dv, M, D, stride, stride, s,
                     num_warps=4, num_ctas=1, num_stages=1)

    def check():
        dq.zero_()
        fn()
        torch.cuda.synchronize()
        rq, rk, rv = _rel(dq, ref_dq), _rel(dk, ref_dk), _rel(dv, ref_dv)
        if max(rq, rk, rv) >= 3e-2:
            return (f"grad rel dQ={rq:.2e} dK={rk:.2e} dV={rv:.2e}"
                    " (>= 3e-2)")
        return None

    return fn, check


# ── runner ───────────────────────────────────────────────────────────────────


def make_registry(args):
    fwd = {
        "triton_ws_off": lambda cfg: build_triton(cfg, False),
        "triton_ws_on": lambda cfg: build_triton(cfg, True),
        "triton_tiled": build_triton_tiled,
        "tlx_default": lambda cfg: build_tlx_fwd(
            cfg, EXAMPLES / "case3_FA_fp16" / "generated.py",
            "fa_fwd_kernel_nows"),
        "jos": lambda cfg: build_tlx_fwd(cfg, Path(args.jos_file),
                                         "fa_fwd_kernel_nows_subtiled"),
        "skc_default": lambda cfg: build_skc(cfg, Path(args.skc_default_file)),
        "skc_jos": lambda cfg: build_skc(cfg, Path(args.skc_jos_file)),
        "skc_var_bn64": lambda cfg: build_skc(cfg, PKG_DIR / "skc_fwd_var_bn64.py"),
        "skc_var_qk2": lambda cfg: build_skc(cfg, PKG_DIR / "skc_fwd_var_qk2.py"),
        "cudnn": build_cudnn,
        "fa4": None,
    }
    bwd = {
        "cudnn_bwd": build_cudnn_bwd,
        "fa4_bwd": None,
        "tlx_bwd_default": lambda cfg: build_tlx_bwd(
            cfg, EXAMPLES / "case4_FA_bwd" / "generated_hd128.py"),
        "jos_bwd": lambda cfg: build_tlx_bwd(cfg, Path(args.jos_bwd_file)),
        "skc_bwd_default": lambda cfg: build_skc_bwd(
            cfg, PKG_DIR / "skc_bwd_default.py"),
        "skc_bwd_jos": lambda cfg: build_skc_bwd(
            cfg, PKG_DIR / "skc_bwd_jos.py"),
        "skc_bwd_q3": lambda cfg: build_skc_bwd(cfg, PKG_DIR / "skc_bwd_q3.py"),
        "skc_bwd_q4": lambda cfg: build_skc_bwd(cfg, PKG_DIR / "skc_bwd_q4.py"),
    }
    return fwd if args.mode == "fwd" else bwd


def run_fa4(cfg, mode, args):
    if not str(FA4_PYTHON) or not FA4_PYTHON.exists():
        return {"ok": False,
                "reason": "set FA4_PYTHON to the FA4 venv python binary"}
    cmd = [
        str(FA4_PYTHON), str(BENCH_DIR / "fa4_worker.py"), "--mode", mode,
        "--b", str(cfg.batch), "--h", str(cfg.heads), "--s", str(cfg.seqlen),
        "--d", str(cfg.head_dim), "--warmup", str(args.warmup), "--rep",
        str(args.rep)
    ]
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=3600)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "fa4 worker timed out"}
    for line in reversed(proc.stdout.strip().splitlines()):
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("ok"):
            return {"ok": False,
                    "reason": r.get("reason", "fa4 worker reported failure")}
        rec = {"ok": True, "ms": r["ms_median"], "lo": r["ms_lo"],
               "hi": r["ms_hi"]}
        if "fwd_ms" in r:
            rec["fwd_ms"] = r["fwd_ms"]
        return rec
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    return {"ok": False,
            "reason": f"fa4 worker produced no JSON (rc={proc.returncode}): "
                      + " | ".join(tail)}


def run_one(name, builder, cfg, mode, args):
    rec = {"env_probe": {"before": _nvsmi()}}
    try:
        if name in EXTERNAL_BARS:
            rec.update(run_fa4(cfg, mode, args))
        else:
            built = builder(cfg)
            fn, check = built[0], built[1]
            fwd_baseline = built[2] if len(built) > 2 else None
            reason = check()
            torch.cuda.synchronize()
            if reason:
                rec.update(ok=False, reason=f"correctness gate: {reason}")
            else:
                ms, lo, hi = triton.testing.do_bench(
                    fn, warmup=args.warmup, rep=args.rep, quantiles=QUANTILES)
                if fwd_baseline is not None:
                    # bwd = (fwd+bwd) - fwd; quantiles of the total shifted by
                    # the fwd median (see README).  This Triton's do_bench
                    # returns a bare float for a single quantile.
                    fwd_ms = triton.testing.do_bench(
                        fwd_baseline, warmup=args.warmup, rep=args.rep,
                        quantiles=[0.5])
                    if isinstance(fwd_ms, (list, tuple)):
                        fwd_ms = fwd_ms[0]
                    ms, lo, hi = ms - fwd_ms, lo - fwd_ms, hi - fwd_ms
                    rec["fwd_ms"] = fwd_ms
                rec.update(ok=True, ms=ms, lo=lo, hi=hi)
    except BarUnavailable as e:
        rec.update(ok=False, reason=str(e))
    except Exception as e:  # keep the sweep going
        rec.update(ok=False, reason=f"{type(e).__name__}: {e}")
    if rec.get("ok") and rec["ms"] <= 0:
        rec.update(ok=False, reason=f"non-positive time {rec['ms']:.4f} ms")
    if rec.get("ok"):
        rec["tflops"] = total_flops(mode, cfg.seqlen) * 1e-12 / (rec["ms"] *
                                                                 1e-3)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    rec["env_probe"]["after"] = _nvsmi()
    return rec


def _report(name, seqlen, rec, mode):
    if rec.get("ok"):
        flops = total_flops(mode, seqlen)
        t_lo = flops * 1e-12 / (max(rec["hi"], 1e-6) * 1e-3)
        t_hi = flops * 1e-12 / (max(rec["lo"], 1e-6) * 1e-3)
        print(f"[{mode}] {name:16s} S={seqlen:6d}  {rec['tflops']:8.1f} TFLOPS"
              f"  (q80/q20: {t_lo:.1f}/{t_hi:.1f})  {rec['ms']:.4f} ms",
              flush=True)
    else:
        print(f"[{mode}] {name:16s} S={seqlen:6d}  SKIP: {rec.get('reason')}",
              flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench_bars")
    ap.add_argument("--bars", default=None,
                    help="comma-separated; default: all bars for --mode")
    ap.add_argument("--mode", choices=["fwd", "bwd"], default="fwd")
    ap.add_argument("--seqlens", default=",".join(str(s) for s in SEQLENS))
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--rep", type=int, default=500)
    ap.add_argument("--jos-file",
                    default=str(EXAMPLES / "case3_FA_fp16_subtiled" /
                                "generated.py"))
    ap.add_argument("--jos-bwd-file",
                    default=str(EXAMPLES / "case4_FA_bwd" /
                                "generated_hd128_jos.py"))
    ap.add_argument("--skc-default-file",
                    default=str(PKG_DIR / "skc_fwd_default.py"))
    ap.add_argument("--skc-jos-file",
                    default=str(PKG_DIR / "skc_fwd_jos.py"))
    args = ap.parse_args(argv)

    registry = make_registry(args)
    bars = list(registry) if args.bars is None else args.bars.split(",")
    unknown = [b for b in bars if b not in registry]
    if unknown:
        ap.error(f"unknown bars {unknown}; available for --mode {args.mode}: "
                 f"{list(registry)}")
    seqlens = [int(s) for s in args.seqlens.split(",")]

    triton.set_allocator(_alloc_fn)
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in bars:
        for s in seqlens:
            cfg = ShapeCfg(BATCH, NUM_HEADS, s, HEAD_DIM,
                           1.0 / math.sqrt(HEAD_DIM))
            rec = run_one(name, registry[name], cfg, args.mode, args)
            results.setdefault(name, {})[str(s)] = rec
            out_path.write_text(json.dumps(results, indent=1))
            _report(name, s, rec, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
