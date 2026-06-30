"""Evaluate the M1 TTGIR reduction-equivalence check on LARGER / MORE COMPLEX kernels.

Same static-vs-empirical confusion framework as evaluate_reduction_equivalence.py,
extended to two groups:

  IN-SCOPE (the checker should handle; soundness FP==0 is gated):
    - sum_looped      : large N with a runtime loop + tail masking (single final reduce)
    - reduce2d_axis0  : genuine 2-D tile reduced along axis 0
    - reduce2d_axis1  : genuine 2-D tile reduced along axis 1
    - sum_bf16        : bf16 in/out (f32-upcast reduce path); 16-bit bit-compare
    - sum_fp16        : fp16 in/out

  BOUNDARY (out of the current checker's scope; we MAP the limitation, not gate it):
    - argmin          : multi-operand (value+index) reduce -> parser returns () (empty)
    - welford         : multi-operand (mean,M2,weight) reduce -> () AND order-sensitive
                        => the empty descriptor unsoundly equates differing configs (FP)
    - gemm            : tl.dot / MMA -> no tt.reduce -> () => precision modes that differ
                        are unsoundly equated (the M3 sub-problem)

Per pair: stat = reduction_descriptor(ttgir_a)==reduction_descriptor(ttgir_b);
emp = bit-identical on every one of R random inputs.

                 emp-equal      emp-different
  stat-equiv     TP             FP  (unsound)
  stat-noteq     FN (cons.)     TN  (detected)

In-scope FP / problems hard-fail the run. Boundary FP / empty descriptors are
EXPECTED findings (the point of including them) — reported, not gated.

Run:
  cd <m1 worktree>
  PYTHONPATH=/home/youngzt/bitwise-equiv/triton/python \\
    python bitequiv/evaluation/evaluate_complex_kernels.py
"""
import itertools
import os
import sys
from collections import namedtuple

import torch

import triton
import triton.language as tl

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "bitequiv", "examples", "numerical-inconsistency"))

from bitequiv.reduction_tree import reduction_descriptor  # noqa: E402
from _helpers import adversarial_1d, adversarial_2d, bitclass_key, compile_only, is_cuda  # noqa: E402

R_SEEDS = list(range(5))
_DEV = "cuda"
_ORD = {"unordered": tl.ReductionOrdering.UNORDERED, "inner_tree": tl.ReductionOrdering.INNER_TREE}

RCfg = namedtuple("RCfg", "ordering num_warps num_stages")
GCfg = namedtuple("GCfg", "prec num_warps block_k")
Spec = namedtuple("Spec", "name in_scope cfgs label compile run")


def _rcfgs(stages=(2, )):
    return [RCfg(o, nw, ns) for o in ("unordered", "inner_tree") for nw in (1, 2, 4, 8) for ns in stages]


def _rlabel(c):
    return f"{c.ordering[:3]}/nw{c.num_warps}/ns{c.num_stages}"


# --------------------------------------------------------------------------- #
# Combine fns (verbatim from python/test/unit/language/test_core.py)
# --------------------------------------------------------------------------- #
@triton.jit
def _welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = weight_2 / new_weight
    return (mean_1 + delta * w2_over_w, m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w, new_weight)


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
@triton.jit
def sum_looped_kernel(src, dst, N, BLOCK: tl.constexpr, ORD: tl.constexpr):
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        acc += tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(acc, axis=0, reduction_ordering=ORD))


@triton.jit
def reduce2d_kernel(src, dst, sr, sc, BM: tl.constexpr, BN: tl.constexpr, AXIS: tl.constexpr, ORD: tl.constexpr):
    offm = tl.arange(0, BM)
    offn = tl.arange(0, BN)
    x = tl.load(src + offm[:, None] * sr + offn[None, :] * sc)
    z = tl.sum(x, axis=AXIS, reduction_ordering=ORD)
    if AXIS == 1:
        tl.store(dst + offm, z)
    else:
        tl.store(dst + offn, z)


@triton.jit
def sum_cast_kernel(src, dst, N, BLOCK: tl.constexpr, ORD: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    s = tl.sum(x, axis=0, reduction_ordering=ORD)
    tl.store(dst, s.to(dst.dtype.element_ty))


@triton.jit
def argmin_kernel(src, dst, N, BLOCK: tl.constexpr, ORD: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=float("inf"))
    tl.store(dst, tl.argmin(x, axis=0, reduction_ordering=ORD))


@triton.jit
def welford_kernel(src, dst, BLOCK: tl.constexpr, ORD: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs)
    mean = x
    m2 = tl.zeros_like(x)
    weight = tl.full(x.shape, 1.0, tl.float32)
    (mean, m2, weight) = tl.reduce((mean, m2, weight), 0, _welford_combine, reduction_ordering=ORD)
    tl.store(dst, m2 / weight)


@triton.jit
def gemm_kernel(a, b, c, M, N, K, sa0, sa1, sb0, sb1, sc0, sc1, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr):
    offm = tl.arange(0, BM)
    offn = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offk = k0 + tl.arange(0, BK)
        at = tl.load(a + offm[:, None] * sa0 + offk[None, :] * sa1)
        bt = tl.load(b + offk[:, None] * sb0 + offn[None, :] * sb1)
        acc = tl.dot(at, bt, acc, input_precision=PREC)
    tl.store(c + offm[:, None] * sc0 + offn[None, :] * sc1, acc)


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
def _sum_looped_spec():
    N, BLOCK = 1 << 20, 2048

    def comp(c):
        ck = compile_only(sum_looped_kernel, adversarial_1d(N, 0, device=_DEV), torch.empty(1, device=_DEV), N,
                          BLOCK=BLOCK, ORD=_ORD[c.ordering], num_warps=c.num_warps, num_stages=c.num_stages, grid=(1, ))
        return ck.asm["ttgir"]

    def run(c, seed):
        dst = torch.empty(1, device=_DEV)
        sum_looped_kernel[(1, )](adversarial_1d(N, seed, device=_DEV), dst, N, BLOCK=BLOCK, ORD=_ORD[c.ordering],
                                 num_warps=c.num_warps, num_stages=c.num_stages)
        torch.cuda.synchronize()
        return dst

    return Spec("sum_looped (large N, loop+mask)", True, _rcfgs(stages=(2, 3)), _rlabel, comp, run)


def _reduce2d_spec(axis):
    BM = BN = 128

    def comp(c):
        src = adversarial_2d(BM, BN, 0, device=_DEV)
        out = torch.empty(BM if axis == 1 else BN, device=_DEV)
        ck = compile_only(reduce2d_kernel, src, out, src.stride(0), src.stride(1), BM=BM, BN=BN, AXIS=axis,
                          ORD=_ORD[c.ordering], num_warps=c.num_warps, num_stages=c.num_stages, grid=(1, ))
        return ck.asm["ttgir"]

    def run(c, seed):
        src = adversarial_2d(BM, BN, seed, device=_DEV)
        out = torch.empty(BM if axis == 1 else BN, device=_DEV)
        reduce2d_kernel[(1, )](src, out, src.stride(0), src.stride(1), BM=BM, BN=BN, AXIS=axis, ORD=_ORD[c.ordering],
                               num_warps=c.num_warps, num_stages=c.num_stages)
        torch.cuda.synchronize()
        return out

    return Spec(f"reduce2d_axis{axis}", True, _rcfgs(), _rlabel, comp, run)


def _sum_cast_spec(dtype, name):
    N = 8192

    def comp(c):
        src = adversarial_1d(N, 0, device=_DEV, dtype=dtype)
        ck = compile_only(sum_cast_kernel, src, torch.empty(1, device=_DEV, dtype=dtype), N, BLOCK=N,
                          ORD=_ORD[c.ordering], num_warps=c.num_warps, num_stages=c.num_stages, grid=(1, ))
        return ck.asm["ttgir"]

    def run(c, seed):
        dst = torch.empty(1, device=_DEV, dtype=dtype)
        sum_cast_kernel[(1, )](adversarial_1d(N, seed, device=_DEV, dtype=dtype), dst, N, BLOCK=N, ORD=_ORD[c.ordering],
                               num_warps=c.num_warps, num_stages=c.num_stages)
        torch.cuda.synchronize()
        return dst

    return Spec(name, True, _rcfgs(), _rlabel, comp, run)


def _argmin_spec():
    N = 8192

    def comp(c):
        ck = compile_only(argmin_kernel, adversarial_1d(N, 0, device=_DEV),
                          torch.empty(1, device=_DEV, dtype=torch.int32), N, BLOCK=N, ORD=_ORD[c.ordering],
                          num_warps=c.num_warps, num_stages=c.num_stages, grid=(1, ))
        return ck.asm["ttgir"]

    def run(c, seed):
        dst = torch.empty(1, device=_DEV, dtype=torch.int32)
        argmin_kernel[(1, )](adversarial_1d(N, seed, device=_DEV), dst, N, BLOCK=N, ORD=_ORD[c.ordering],
                             num_warps=c.num_warps, num_stages=c.num_stages)
        torch.cuda.synchronize()
        return dst

    return Spec("argmin (multi-operand value+index)", False, _rcfgs(), _rlabel, comp, run)


def _welford_spec():
    N = 8192

    def comp(c):
        ck = compile_only(welford_kernel, adversarial_1d(N, 0, device=_DEV), torch.empty(1, device=_DEV), BLOCK=N,
                          ORD=_ORD[c.ordering], num_warps=c.num_warps, num_stages=c.num_stages, grid=(1, ))
        return ck.asm["ttgir"]

    def run(c, seed):
        dst = torch.empty(1, device=_DEV)
        welford_kernel[(1, )](adversarial_1d(N, seed, device=_DEV), dst, BLOCK=N, ORD=_ORD[c.ordering],
                              num_warps=c.num_warps, num_stages=c.num_stages)
        torch.cuda.synchronize()
        return dst

    return Spec("welford (multi-operand, order-sensitive)", False, _rcfgs(), _rlabel, comp, run)


def _gemm_spec():
    M = N = K = 128
    cfgs = [GCfg(p, nw, bk) for p in ("ieee", "tf32") for nw, bk in ((4, 128), (8, 128), (4, 64))]

    def _ab(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.randn(M, K, generator=g).to(_DEV, torch.float32)
        b = torch.randn(K, N, generator=g).to(_DEV, torch.float32)
        return a, b

    def comp(c):
        a, b = _ab(0)
        out = torch.empty(M, N, device=_DEV)
        ck = compile_only(gemm_kernel, a, b, out, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                          out.stride(0), out.stride(1), PREC=c.prec, BM=M, BN=N, BK=c.block_k, num_warps=c.num_warps,
                          num_stages=1, grid=(1, ))
        return ck.asm["ttgir"]

    def run(c, seed):
        a, b = _ab(seed)
        out = torch.empty(M, N, device=_DEV)
        gemm_kernel[(1, )](a, b, out, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0),
                           out.stride(1), PREC=c.prec, BM=M, BN=N, BK=c.block_k, num_warps=c.num_warps, num_stages=1)
        torch.cuda.synchronize()
        return out

    return Spec("gemm (tl.dot / MMA)", False, cfgs, lambda c: f"{c.prec}/nw{c.num_warps}/bk{c.block_k}", comp, run)


# --------------------------------------------------------------------------- #
# Generic evaluation
# --------------------------------------------------------------------------- #
def _reduce_lines(ttgir):
    return [ln.strip() for ln in ttgir.splitlines() if '"tt.reduce"' in ln]


def evaluate(spec):
    ttgir, desc, bits = {}, {}, {}
    problems = []
    ok = []
    for c in spec.cfgs:
        try:
            g = spec.compile(c)
            b = tuple(bitclass_key(spec.run(c, s)) for s in R_SEEDS)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{spec.label(c)}: {type(e).__name__}: {e}")
            continue
        ttgir[c], desc[c], bits[c], = g, reduction_descriptor(g), b
        ok.append(c)
        if desc[c] == ():
            problems.append(f"{spec.label(c)}: EMPTY descriptor (reduce lines in TTGIR: {len(_reduce_lines(g))})")

    TP = FP = TN = FN = 0
    fp_pairs = []
    for a, b in itertools.combinations(ok, 2):
        stat = desc[a] == desc[b]
        emp = bits[a] == bits[b]
        if stat and emp:
            TP += 1
        elif stat and not emp:
            FP += 1
            fp_pairs.append((a, b))
        elif not stat and not emp:
            TN += 1
        else:
            FN += 1
    return dict(spec=spec, ok=ok, ttgir=ttgir, desc=desc, TP=TP, FP=FP, TN=TN, FN=FN, fp_pairs=fp_pairs,
                problems=problems, n_stat=len({desc[c]
                                               for c in ok}), n_emp=len({bits[c]
                                                                         for c in ok}))


def report(res):
    spec = res["spec"]
    tag = "IN-SCOPE" if spec.in_scope else "BOUNDARY (out of scope)"
    print("\n" + "=" * 78)
    print(f"[{tag}] {spec.name}   ({len(res['ok'])}/{len(spec.cfgs)} configs compiled+ran)")
    print("=" * 78)
    if res["ok"]:
        s0 = res["ok"][0]
        print(f"  sample descriptor [{spec.label(s0)}]: {res['desc'][s0]}")
        for ln in _reduce_lines(res["ttgir"][s0])[:2]:
            print(f"    tt.reduce: {ln[:120]}")
    TP, FP, TN, FN = res["TP"], res["FP"], res["TN"], res["FN"]
    print(f"  static classes={res['n_stat']}  empirical classes={res['n_emp']}")
    print(f"  pairs: TP={TP} FP={FP} TN={TN} FN={FN}")
    if TN + FP:
        print(f"    detection (TN/(TN+FP)) = {100.0 * TN / (TN + FP):.1f}% over {TN + FP} emp-different pairs")
    if TP + FP:
        print(f"    soundness FP rate (FP/(TP+FP)) = {100.0 * FP / (TP + FP):.1f}% over {TP + FP} stat-equiv pairs")
    if FN:
        print(f"    {FN} conservative over-splits (emp-equal, flagged different) — acceptable")
    for a, b in res["fp_pairs"][:6]:
        print(f"    FP: {spec.label(a)} ~ {spec.label(b)} declared equivalent but bits differ")
    for p in res["problems"]:
        print(f"    PROBLEM: {p}")


def main():
    print(f"device: {torch.cuda.get_device_name()}  cc={torch.cuda.get_device_capability()}  "
          f"triton={triton.__version__}  R={len(R_SEEDS)} inputs/config")
    specs = [
        _sum_looped_spec(),
        _reduce2d_spec(0),
        _reduce2d_spec(1),
        _sum_cast_spec(torch.bfloat16, "sum_bf16 (bf16 in/out)"),
        _sum_cast_spec(torch.float16, "sum_fp16 (fp16 in/out)"),
        _argmin_spec(),
        _welford_spec(),
        _gemm_spec(),
    ]
    results = [evaluate(s) for s in specs]
    for res in results:
        report(res)

    inscope = [r for r in results if r["spec"].in_scope]
    boundary = [r for r in results if not r["spec"].in_scope]
    in_fp = sum(r["FP"] for r in inscope)
    in_prob = sum(len(r["problems"]) for r in inscope)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  IN-SCOPE: total FP (soundness) = {in_fp};  problems = {in_prob}")
    print("  BOUNDARY (expected limitations, not gated):")
    for r in boundary:
        empty = sum(1 for c in r["ok"] if r["desc"][c] == ())
        print(f"    - {r['spec'].name}: FP={r['FP']}  empty-descriptors={empty}/{len(r['ok'])}")
    print("=" * 78)
    if in_fp or in_prob:
        raise SystemExit(f"FAIL (in-scope): {in_fp} soundness violation(s), {in_prob} problem(s).")
    print("PASS (in-scope): no soundness violations. Boundary findings mapped above.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
