"""
Fixed-config timing for Meta AutoWS GEMMs on Blackwell.

Times one GEMM kernel across fixed (not autotuned) AutoWS configurations so that
the "[measured]" rules in
third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/docs/perf_tuning/
can be re-run on any build. Each named regime is a set of shapes and configs
that one rule cites, e.g.

    TRITON_USE_META_WS=1 python python/test/microbenchmark/autows_gemm.py --regime stages --json out.json
    TRITON_USE_META_WS=1 python python/test/microbenchmark/autows_gemm.py --list
    TRITON_USE_META_WS=1 python python/test/microbenchmark/autows_gemm.py --shape 4096,4096,4096 \
        --config bm=128,bn=256,bk=64,ns=4,es=2

Every arm is checked before it is timed: bitwise against torch on integer-valued
inputs, and with torch.testing.assert_close (bf16 rtol 1.6e-2, atol 1e-2) on
unit-variance randn inputs. Arms whose TTGIR is
identical to an earlier arm are reported as duplicates and timed once. Timing
flushes L2 before every launch and replays a CUDA graph of launches, so short
kernels are not dominated by event resolution. Lock SM clocks before running.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from typing import NamedTuple

import torch

import triton
import triton.language as tl
from triton.language.extra.subtile_ops import _split_n_2D

SMEM_BYTES = 232448  # sm100 opt-in limit used by the WS memory planner
AUX_BOUND = 4096  # upper bound used for the planner's auxiliary reservation
NUM_SMS = 148


class _TileArgs(NamedTuple):
    M: tl.tensor
    N: tl.tensor
    BM: tl.constexpr
    BN: tl.constexpr
    PAIR: tl.constexpr
    SPLIT_K: tl.constexpr


@triton.jit
def _num_tiles(args):
    num_pid_m = tl.cdiv(tl.cdiv(args.M, args.BM), args.PAIR) * args.PAIR
    return num_pid_m * tl.cdiv(args.N, args.BN) * args.SPLIT_K


@triton.jit
def _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def autows_gemm_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, tile_counter, M, N, K,  #
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ES: tl.constexpr,  #
                       TWO_CTAS: tl.constexpr, DP: tl.constexpr, SES: tl.constexpr, SCHEDULE: tl.constexpr,  #
                       GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, BIAS: tl.constexpr, EPI: tl.constexpr,  #
                       ARM_TAG: tl.constexpr):
    # ARM_TAG only gives arms that differ in env-only knobs their own specialization.
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BN, BK])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN // ES])
    PAIR: tl.constexpr = 2 if TWO_CTAS else 1
    num_pid_m = tl.cdiv(tl.cdiv(M, BM), PAIR) * PAIR  # 2-CTA pairs adjacent M tiles
    num_pid_n = tl.cdiv(N, BN)
    num_mn = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BK)
    k_per_split = tl.cdiv(k_tiles, SPLIT_K)
    sched = SCHEDULE.initialize(_TileArgs(M, N, BM, BN, PAIR, SPLIT_K), _num_tiles, tile_counter)
    while tl.condition(sched.is_valid(), warp_specialize=True, data_partition_factor=DP, separate_epilogue_store=SES):
        tile_id = sched.tile_id[0]
        split = tile_id // num_mn
        pid_m, pid_n = _compute_pid(tile_id % num_mn, num_pid_m, num_pid_n, GROUP_M)
        k_lo = split * k_per_split
        k_hi = tl.minimum(k_lo + k_per_split, k_tiles)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_lo, k_hi):
            a = a_desc.load([pid_m * BM, ki * BK])
            b = b_desc.load([pid_n * BN, ki * BK])
            acc = tl.dot(a, b.T, acc, two_ctas=TWO_CTAS)
        slices = _split_n_2D(acc, ES)
        for s in tl.static_range(0, ES):
            off_n = pid_n * BN + s * (BN // ES)
            out = slices[s]
            if BIAS:
                out = out + tl.load(bias_ptr + off_n + tl.arange(0, BN // ES)).to(tl.float32)[None, :]
            if EPI == "tanh":
                out = 2.0 * tl.sigmoid(2.0 * out) - 1.0
            if SPLIT_K > 1:
                c_desc.atomic_add([pid_m * BM, off_n], out)
            else:
                c_desc.store([pid_m * BM, off_n], out.to(c_ptr.dtype.element_ty))
        sched = sched.advance()


@dataclasses.dataclass(frozen=True)
class Config:
    bm: int = 128
    bn: int = 256
    bk: int = 64
    ns: int = 0  # 0 = d_fit, negative = d_fit + ns
    nw: int = 4
    es: int = 0  # 0 = smallest legal ES with R <= 128 that fits
    two: int = 0
    dp: int = 1
    ses: int = 1
    sched: str = "static"  # static | clc | dynamic | nonpers
    gm: int = 8
    splitk: int = 1
    bias: int = 0
    epi: str = "none"
    staging: int = 0  # TRITON_WS_STAGING_COPIES, 0 = unset
    pingpong: int = 0

    def name(self):
        d = Config()
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in dataclasses.fields(self)
                        if getattr(self, f.name) != getattr(d, f.name)) or "default"


def parse_config(s):
    kw = {}
    for item in s.split(","):
        k, v = item.split("=")
        kw[k] = v if k in ("sched", "epi") else int(v)
    return Config(**kw)


SCHEDULES = {
    "static": tl.StaticPersistent1DScheduler, "clc": tl.ClcTileScheduler, "dynamic": tl.DynamicPersistent1DScheduler,
    "nonpers": tl.NonPersistentScheduler
}


def quantities(M, N, K, c, es=None):
    """Step-2 quantities from the perf_tuning README, for one (shape, config)."""
    es = es or max(c.es, 1)
    pair = 2 if c.two else 1
    tiles = math.ceil(math.ceil(M / c.bm) / pair) * pair * math.ceil(N / c.bn) * c.splitk
    slot = (c.bm + (c.bn // 2 if c.two else c.bn)) * c.bk * 2
    staging = c.bm * (c.bn // es) * (4 if c.splitk > 1 else 2)
    return dict(T=tiles, W=tiles / NUM_SMS, E_tail=tiles / (NUM_SMS * math.ceil(tiles / NUM_SMS)),
                S_prog=math.ceil(tiles / NUM_SMS) * c.bm * c.bn, k_iters=math.ceil(K / (c.bk * c.splitk)),
                pad_waste=1 - M * N / (math.ceil(M / c.bm) * c.bm * math.ceil(N / c.bn) * c.bn),
                AI_min=2 * M * N * K / (2 * (M * K + N * K + M * N)), d_fit=(SMEM_BYTES - AUX_BOUND - staging) // slot,
                R=c.bm // c.dp * (c.bn // es) // (32 * c.nw))


def resolve(M, N, K, c):
    """Fill in ES and num_stages defaults (d_fit-based)."""
    es = c.es
    if es == 0:
        es = next((e for e in (1, 2, 4, 8) if c.bn // e >= 32 and quantities(M, N, K, c, e)["R"] <= 128
                   and quantities(M, N, K, c, e)["d_fit"] >= 2), 8)
    q = quantities(M, N, K, c, es)
    ns = c.ns if c.ns > 0 else max(1, q["d_fit"] + c.ns)
    return dataclasses.replace(c, es=es, ns=ns), q


# Shapes from the evidence plan (M, N, K).
SHAPES = {
    "R1a": (8192, 8192, 8192), "R1b": (4096, 4096, 4096), "R1c": (5139, 4096, 2048), "R2a": (1152, 12800, 1024),
    "R2b": (512, 9472, 4096), "R3a": (256, 256, 65536), "R3b": (1024, 1024, 16384), "R3c": (2048, 2048, 8192),
    "R4a": (8192, 8192, 128), "R4b": (16384, 4096, 256), "R5a": (16384, 128, 16384), "R5b": (65536, 256, 512),
    "R6a": (192, 16384, 4096), "R6b": (320, 8192, 4096)
}

C = Config
TILES = {
    "128x128x64": C(bm=128, bn=128, bk=64), "128x128x128": C(bm=128, bn=128, bk=128), "128x256x64": C(),
    "64x256x64": C(bm=64), "64x128x64": C(bm=64, bn=128), "256x128x64dp2": C(bm=256, bn=128, dp=2),
    "256x256x64dp2": C(bm=256, bn=256, dp=2), "2cta128x256x64": C(two=1), "2cta128x256x128": C(two=1, bk=128),
    "2cta128x128x64": C(two=1, bn=128)
}


def _regime(shapes, configs):
    return [(s, c) for s in shapes for c in configs]


def _ns_pair(cfgs):
    return [dataclasses.replace(c, ns=ns) for c in cfgs for ns in (0, -1)]


def _all_ns(cfgs):
    return [dataclasses.replace(c, ns=ns) for c in cfgs for ns in (1, 2, 3, 4, 5, 6)]


T = TILES
REGIMES = {
    "tile": _regime(["R1a", "R1b", "R1c", "R2a", "R2b"],
                    _ns_pair([T[k] for k in ("128x128x64", "128x128x128", "128x256x64", "64x256x64", "256x128x64dp2",
                                             "2cta128x256x64", "2cta128x256x128", "2cta128x128x64")])),
    "mtail": _regime(["R6a", "R6b", "R1c", "R2b"],
                     _ns_pair([T[k] for k in ("64x256x64", "128x256x64", "64x128x64", "128x128x64")])),
    "stages": _regime(["R1a", "R1b", "R2a", "R3c", "R4a", "R4b"],
                      _all_ns([T["128x256x64"], T["128x128x128"], T["2cta128x256x128"]])),
    "bk": _regime(["R1a", "R1b", "R2a", "R4b", "R3b"],
                  _ns_pair([dataclasses.replace(c, bk=bk) for c in (T["128x128x64"], T["128x256x64"], T["2cta128x256x64"])
                            for bk in (64, 128)])),
    "wes": _regime(["R1a", "R1b", "R2a"], [
        dataclasses.replace(c, nw=nw, es=es) for c in (T["128x128x64"], T["128x256x64"], T["256x128x64dp2"],
                                                       T["2cta128x256x64"]) for nw in (4, 8) for es in (1, 2, 4, 8)
        if c.bn // es >= 32
    ]) + _regime(["R1b"], [C(bias=1, nw=nw, es=es) for nw in (4, 8) for es in (1, 2, 4)]),
    "ses": _regime(["R1a", "R1b", "R2a"], [
        dataclasses.replace(c, ses=ses, staging=st) for c in (C(es=2), C(es=4), C(bm=128, bn=128, bk=128, es=2),
                                                             C(two=1, es=4)) for ses in (0, 1) for st in (0, 1, 2)
        if not (st and c.two) and not (st == 2 and ses == 0 and c.es % 2)
    ]),
    "2cta": _regime(["R2b", "R3b", "R3c"],
                    _ns_pair([T["128x256x64"], T["2cta128x256x64"], T["128x128x128"], T["2cta128x128x64"]])),
    "dp": _regime(["R1a", "R1b", "R2a", "R4a", "R3c"],
                  _ns_pair([C(bm=256, bn=128, dp=1), T["256x128x64dp2"], T["256x256x64dp2"], T["2cta128x256x64"],
                            T["2cta128x128x64"]])),
    "dp_k": [(f"M8192xN8192xK{k}", c) for k in (256, 1024, 4096, 16384) for c in (T["256x256x64dp2"], T["256x128x64dp2"])],
    "sched": _regime(["R1a", "R1b", "R2a", "R2b", "R3b", "R3c", "R6a"], [
        dataclasses.replace(c, sched=s, gm=g) for c in (T["128x256x64"], T["2cta128x256x64"])
        for s, g in (("nonpers", 8), ("static", 8), ("static", 1), ("static", 16), ("clc", 8))
    ]) + _regime(["R1b", "R2a", "R3b", "R6a"], [C(sched="dynamic")]),
    "splitk": _regime(["R3a", "R3b", "R3c", "R5a", "R5b", "R1b"],
                      [dataclasses.replace(c, splitk=s) for c in (T["128x128x64"], T["128x256x64"], T["64x128x64"])
                       for s in (1, 2, 4, 8, 16, 32)]),
    "ridge": [(f"M8192xN8192xK{k}", C()) for k in (128, 256, 512, 1024, 2048, 4096, 8192)] +
    [(f"M16384xN{n}xK4096", C(bn=min(n, 256))) for n in (128, 256, 512, 1024, 2048, 4096)],
    "pingpong": _regime(["R1b", "R2a"], [C(bm=256, bn=128, dp=2, bias=1, epi="tanh", pingpong=p) for p in (0, 1)]),
}


def shape_of(name):
    if name in SHAPES:
        return SHAPES[name]
    m = re.fullmatch(r"M(\d+)xN(\d+)xK(\d+)", name)
    return tuple(int(x) for x in m.groups())


def skip_reason(M, N, K, c, q):
    """Arms the plan excludes up front (never-try rows and the split-K bound)."""
    if c.splitk > 1 and (q["T"] // c.splitk * (c.splitk - 1) >= NUM_SMS or q["k_iters"] < 4):
        return "split-K outside T*(S-1) < P and k_iters/S >= 4"
    if c.two and c.sched == "dynamic":
        return "2-CTA with the atomic dynamic scheduler (never try)"
    if q["d_fit"] < 1:
        return "does not fit SMEM"
    if c.bm == 256 and c.dp == 1 and c.bn == 256:
        return "TMEM"
    return None


class Arm:

    def __init__(self, M, N, K, cfg, tag):
        self.M, self.N, self.K = M, N, K
        self.cfg, self.q = resolve(M, N, K, cfg)
        self.tag = tag
        self.kernel = None

    def grid_and_counter(self):
        T = self.q["T"]
        persistent = self.cfg.sched in ("static", "dynamic")
        g = min(NUM_SMS, T) if persistent else T
        if self.cfg.two:
            g = g // 2 * 2
        return (g, ), torch.full((1, ), g, dtype=torch.int32, device="cuda")

    def launch(self, a, b, c, bias):
        cfg = self.cfg
        grid, counter = self.grid_and_counter()
        opts = dict(num_warps=cfg.nw, num_stages=cfg.ns, pingpongAutoWS=bool(cfg.pingpong))
        if cfg.two:
            opts["ctas_per_cga"] = (2, 1, 1)
        if cfg.splitk > 1:
            c.zero_()
        self.kernel = autows_gemm_kernel[grid](a, b, c, bias, counter, self.M, self.N, self.K, cfg.bm, cfg.bn, cfg.bk,
                                               cfg.es, bool(cfg.two), cfg.dp, bool(cfg.ses), SCHEDULES[cfg.sched],
                                               cfg.gm, cfg.splitk, bool(cfg.bias), cfg.epi, self.tag, **opts)

    def features(self):
        ttgir, ptx = self.kernel.asm["ttgir"], self.kernel.asm["ptx"]
        copies = sorted(set(re.findall(r"!ttg.memdesc<(\d+)x(\d+)x(\d+)x(bf16|f32)", ttgir)))
        body = "\n".join(re.sub(r"\s*loc\(.*\)$", "", ln) for ln in ttgir.splitlines() if not ln.startswith("#loc"))
        return dict(ws="ttg.warp_specialize" in ttgir, cta_group2="cta_group::2" in ptx, clc="clc_" in ttgir,
                    dp_halves=ttgir.count("computation"), smem_buffers=["x".join(x) for x in copies[:8]],
                    smem=self.kernel.metadata.shared, ttgir_hash=hashlib.sha1(body.encode()).hexdigest()[:12])


def make_inputs(M, N, K, integer):
    if integer:
        a = torch.randint(-3, 4, (M, K), device="cuda").to(torch.bfloat16)
        b = torch.randint(-3, 4, (N, K), device="cuda").to(torch.bfloat16)
        bias = torch.randint(-3, 4, (N, ), device="cuda").to(torch.bfloat16)
    else:
        # Unit-variance outputs, so one absolute tolerance fits every K.
        a = (torch.randn((M, K), device="cuda") * K**-0.25).to(torch.bfloat16)
        b = (torch.randn((N, K), device="cuda") * K**-0.25).to(torch.bfloat16)
        bias = torch.randn((N, ), device="cuda", dtype=torch.bfloat16)
    return a, b, bias


def reference(a, b, bias, cfg):
    out = a.float() @ b.float().T
    if cfg.bias:
        out = out + bias.float()
    if cfg.epi == "tanh":
        out = torch.tanh(out)
    return out


def check(arm):
    """Bitwise on integer inputs (exact while 9*K < 2^24), then assert_close (bf16 rtol, atol 1e-2) on randn."""
    cfg = arm.cfg
    out_dtype = torch.float32 if cfg.splitk > 1 else torch.bfloat16
    a, b, bias = make_inputs(arm.M, arm.N, arm.K, integer=True)
    ref = reference(a, b, bias, cfg).to(out_dtype)
    for _ in range(3):
        c = torch.full((arm.M, arm.N), float("nan"), device="cuda", dtype=out_dtype)
        arm.launch(a, b, c, bias)
        torch.cuda.synchronize()
        exact = cfg.epi != "none" or torch.equal(c, ref)
        if not exact:
            return False, f"bitwise: {int((c != ref).sum())} of {c.numel()} elements differ"
    a, b, bias = make_inputs(arm.M, arm.N, arm.K, integer=False)
    c = torch.full((arm.M, arm.N), float("nan"), device="cuda", dtype=out_dtype)
    arm.launch(a, b, c, bias)
    try:
        torch.testing.assert_close(c, reference(a, b, bias, cfg).to(out_dtype), atol=1e-2, rtol=1.6e-2)
    except AssertionError as e:
        return False, " ".join(line.strip() for line in str(e).splitlines() if line.strip())[:300]
    return True, ""


def time_graph(fn, flush, reps=10):
    """ms per launch: a CUDA graph of `reps` (flush, fn) pairs minus a graph of `reps` flushes."""

    def capture(body):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                body()
        torch.cuda.current_stream().wait_stream(s)
        return g

    def pairs():
        for _ in range(reps):
            flush()
            fn()

    def flushes():
        for _ in range(reps):
            flush()

    gp, gf = capture(pairs), capture(flushes)

    def run(g):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    samples = [(run(gp) - run(gf)) / reps for _ in range(20)]
    samples.sort()
    keep = samples[2:-2]  # 10% trimmed on each side
    return sum(keep) / len(keep)


def gpu_state():
    out = subprocess.run([
        "nvidia-smi", "-i",
        os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], "--query-gpu=clocks.sm,power.draw,"
        "clocks_event_reasons.active", "--format=csv,noheader,nounits"
    ], capture_output=True, text=True).stdout.strip().split(", ")
    return dict(sm_mhz=int(out[0]), power_w=float(out[1]), reasons=out[2]) if len(out) == 3 else {}


def run_shape(shape_name, cfgs, rounds, compile_only=False):
    M, N, K = shape_of(shape_name)
    results, arms, seen = [], [], {}
    for i, cfg in enumerate(cfgs):
        base = Arm(M, N, K, cfg, tag=0)
        rec = dict(shape=shape_name, M=M, N=N, K=K, config=cfg.name(), resolved=dataclasses.asdict(base.cfg),
                   quantities=base.q)
        reason = skip_reason(M, N, K, base.cfg, base.q)
        if reason:
            results.append(dict(rec, status="skipped", reason=reason))
            continue
        arm = Arm(M, N, K, cfg, tag=i if base.cfg.staging else 0)
        os.environ.pop("TRITON_WS_STAGING_COPIES", None)
        if arm.cfg.staging:
            os.environ["TRITON_WS_STAGING_COPIES"] = str(arm.cfg.staging)
        try:
            ok, msg = check(arm)
        except Exception as e:  # compile errors and launch failures are results too
            results.append(dict(rec, status="error", reason=f"{type(e).__name__}: {str(e)[:300]}"))
            continue
        feats = arm.features()
        rec.update(features=feats)
        if not ok:
            results.append(dict(rec, status="WRONG", reason=msg))
            print(f"WRONG ANSWER {shape_name} {cfg.name()}: {msg}", file=sys.stderr, flush=True)
            continue
        if not feats["ws"] or bool(feats["cta_group2"]) != bool(arm.cfg.two) or (feats["clc"] != (arm.cfg.sched == "clc")):
            results.append(dict(rec, status="feature-missing"))
            continue
        if feats["ttgir_hash"] in seen and not arm.cfg.staging:
            results.append(dict(rec, status="duplicate", duplicate_of=seen[feats["ttgir_hash"]]))
            continue
        seen[feats["ttgir_hash"]] = cfg.name()
        results.append(dict(rec, status="timed", times_ms=[]))
        arms.append((arm, results[-1]))
    if compile_only or not arms:
        return results
    a, b, bias = make_inputs(M, N, K, integer=False)
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    flush = lambda: triton.runtime.driver.active.clear_cache(cache)
    c16 = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    c32 = torch.empty((M, N), device="cuda", dtype=torch.float32)
    cublas = dict(shape=shape_name, M=M, N=N, K=K, config="cublas", status="timed", times_ms=[])
    results.append(cublas)
    for _ in range(rounds):
        state0 = gpu_state()
        for arm, rec in arms:
            fn = lambda arm=arm: arm.launch(a, b, c32 if arm.cfg.splitk > 1 else c16, bias)
            rec["times_ms"].append(time_graph(fn, flush))
        cublas["times_ms"].append(time_graph(lambda: torch.matmul(a, b.T, out=c16), flush))
        state1 = gpu_state()
        for _, rec in arms + [(None, cublas)]:
            rec.setdefault("gpu_state", []).append([state0, state1])
    for _, rec in arms + [(None, cublas)]:
        ts = sorted(rec["times_ms"])
        rec["median_ms"] = ts[len(ts) // 2]
        rec["tflops"] = 2 * M * N * K / rec["median_ms"] / 1e9
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regime", help="named regime (see --list)")
    p.add_argument("--shapes", help="comma-separated shape ids to restrict a regime to")
    p.add_argument("--shape", help="M,N,K for a one-off run")
    p.add_argument("--config", action="append", default=[], help="one-off config, e.g. bm=128,bn=256,ns=4")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--json", help="write results here")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        for name, arms in REGIMES.items():
            print(f"{name}: {len(arms)} arms over {sorted(set(s for s, _ in arms))}")
        return
    if os.environ.get("TRITON_USE_META_WS") != "1":
        sys.exit("set TRITON_USE_META_WS=1")
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    if args.shape:
        work = {"M{}xN{}xK{}".format(*args.shape.split(",")): [parse_config(c) for c in args.config or ["bm=128"]]}
    else:
        work = {}
        for s, c in REGIMES[args.regime]:
            if not args.shapes or s in args.shapes.split(","):
                work.setdefault(s, []).append(c)
    results = []
    for shape_name, cfgs in work.items():
        results += run_shape(shape_name, cfgs, args.rounds, args.compile_only)
        for r in results:
            if r["shape"] == shape_name:
                t = f"{r['median_ms'] * 1e3:9.1f} us" if "median_ms" in r else f"{r['status']:>12}"
                print(f"{shape_name:>20} {r['config']:<48} {t} {r.get('reason', '')}", flush=True)
    meta = dict(triton=triton.__version__, commit=os.environ.get("AUTOWS_GEMM_COMMIT", ""), gpu_state=gpu_state(),
                regime=args.regime)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(meta=meta, results=results), f, indent=1)


if __name__ == "__main__":
    main()
