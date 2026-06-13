"""Evaluate the M1 TTGIR reduction-equivalence check against REAL compiled kernels.

Two-way evaluation over a config matrix, per kernel:

  Part 1 — detection (recall):  for config pairs whose outputs differ bit-for-bit,
           does our static check flag them as NOT equivalent (via IR inspection)?
  Part 2 — soundness (FP rate): for config pairs our static check declares
           EQUIVALENT, run R random inputs and measure the rate they are NOT
           actually bit-equal. This MUST be 0; any FP is a soundness bug.

Static relation : bitequiv.reduction_tree.reduction_descriptor over the compiled
                  TTGIR (two configs equivalent iff descriptors are equal).
Empirical truth : two configs are equivalent iff their outputs are bit-identical on
                  *every* one of R adversarial random inputs.

Confusion per pair:
                 emp-equal      emp-different
  stat-equiv     TP (correct)   FP  <-- UNSOUND, must be 0
  stat-noteq     FN (cons.)     TN  (detected)

This is the FIRST time the parser runs on real compiled TTGIR, so it also flags
implementation/design mismatches (empty descriptor => regex didn't match; an
inner_tree pair that is emp-different => inner_tree not actually layout-invariant).
Per the plan: we REPORT such problems and exit nonzero — fixes are the next step.

Scope: simple 1-D reductions (sum, softmax, layernorm). dot/MMA + precision are
deferred (bits there move via input_precision, not reduction order — M3 territory).

Run:
  cd <m1 worktree>
  PYTHONPATH=/home/youngzt/bitwise-equiv/triton/python \\
    python bitequiv/evaluation/evaluate_reduction_equivalence.py
"""
import itertools
import os
import sys
from collections import namedtuple

import torch

import triton
import triton.language as tl

# Our checker lives in the m1 worktree root; the repro helpers in the (materialized)
# numerical-inconsistency example folder. Put both on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "bitequiv", "examples", "numerical-inconsistency"))

from bitequiv.reduction_tree import reduction_descriptor  # noqa: E402
from _helpers import adversarial_1d, adversarial_2d, bitclass_key, compile_only, is_cuda  # noqa: E402

R_SEEDS = list(range(8))  # number of random inputs for the empirical relation
_ORD = {"unordered": tl.ReductionOrdering.UNORDERED, "inner_tree": tl.ReductionOrdering.INNER_TREE}

Config = namedtuple("Config", "ordering num_warps num_stages")


def configs():
    return [Config(o, nw, ns) for o in ("unordered", "inner_tree") for nw in (1, 2, 4, 8) for ns in (2, 3)]


# --------------------------------------------------------------------------- #
# Kernels — ordering-parameterized so the inner_tree path is exercised.
# --------------------------------------------------------------------------- #
@triton.jit
def sum_kernel(src, dst, N, BLOCK: tl.constexpr, ORD: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0, reduction_ordering=ORD))


@triton.jit
def softmax_kernel(src, dst, n_cols, stride, BLOCK: tl.constexpr, ORD: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(src + row * stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0, reduction_ordering=ORD)
    num = tl.exp(x)
    den = tl.sum(num, axis=0, reduction_ordering=ORD)
    tl.store(dst + row * stride + offs, num / den, mask=mask)


@triton.jit
def layernorm_kernel(src, dst, n_cols, stride, eps, BLOCK: tl.constexpr, ORD: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(src + row * stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0, reduction_ordering=ORD) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0, reduction_ordering=ORD) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(dst + row * stride + offs, xc * rstd, mask=mask)


# --------------------------------------------------------------------------- #
# Per-kernel specs: compile-to-TTGIR (input-independent) and run-on-seed.
# --------------------------------------------------------------------------- #
Spec = namedtuple("Spec", "name compile run")
_DEV = "cuda"


def _sum_spec():
    N = 8192

    def comp(c):
        src = adversarial_1d(N, seed=0, device=_DEV)
        dst = torch.empty(1, device=_DEV)
        ck = compile_only(sum_kernel, src, dst, N, BLOCK=N, ORD=_ORD[c.ordering], num_warps=c.num_warps,
                          num_stages=c.num_stages, grid=(1, ))
        return ck.asm["ttgir"]

    def run(c, seed):
        src = adversarial_1d(N, seed=seed, device=_DEV)
        dst = torch.empty(1, device=_DEV)
        sum_kernel[(1, )](src, dst, N, BLOCK=N, ORD=_ORD[c.ordering], num_warps=c.num_warps, num_stages=c.num_stages)
        torch.cuda.synchronize()
        return dst

    return Spec("sum", comp, run)


def _softmax_spec():
    rows, cols = 64, 8192

    def _inp(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(rows, cols, generator=g, dtype=torch.float32).to(_DEV)  # unit-scale (see n03)

    def comp(c):
        src = _inp(0)
        dst = torch.empty_like(src)
        ck = compile_only(softmax_kernel, src, dst, cols, src.stride(0), BLOCK=cols, ORD=_ORD[c.ordering],
                          num_warps=c.num_warps, num_stages=c.num_stages, grid=(rows, ))
        return ck.asm["ttgir"]

    def run(c, seed):
        src = _inp(seed)
        dst = torch.empty_like(src)
        softmax_kernel[(rows, )](src, dst, cols, src.stride(0), BLOCK=cols, ORD=_ORD[c.ordering], num_warps=c.num_warps,
                                 num_stages=c.num_stages)
        torch.cuda.synchronize()
        return dst

    return Spec("softmax", comp, run)


def _layernorm_spec():
    rows, cols, eps = 64, 8192, 1e-5

    def comp(c):
        src = adversarial_2d(rows, cols, seed=0, device=_DEV)
        dst = torch.empty_like(src)
        ck = compile_only(layernorm_kernel, src, dst, cols, src.stride(0), eps, BLOCK=cols, ORD=_ORD[c.ordering],
                          num_warps=c.num_warps, num_stages=c.num_stages, grid=(rows, ))
        return ck.asm["ttgir"]

    def run(c, seed):
        src = adversarial_2d(rows, cols, seed=seed, device=_DEV)
        dst = torch.empty_like(src)
        layernorm_kernel[(rows, )](src, dst, cols, src.stride(0), eps, BLOCK=cols, ORD=_ORD[c.ordering],
                                   num_warps=c.num_warps, num_stages=c.num_stages)
        torch.cuda.synchronize()
        return dst

    return Spec("layernorm", comp, run)


def _reduce_lines(ttgir):
    return [ln.strip() for ln in ttgir.splitlines() if '"tt.reduce"' in ln]


def evaluate(spec):
    cfgs = configs()
    ttgir, desc, bits = {}, {}, {}
    problems = []
    for c in cfgs:
        try:
            g = spec.compile(c)
        except Exception as e:  # noqa: BLE001
            problems.append(f"compile failed for {c}: {type(e).__name__}: {e}")
            continue
        ttgir[c] = g
        desc[c] = reduction_descriptor(g)
        if desc[c] == ():
            problems.append(f"EMPTY descriptor for {c} — parser did not match real TTGIR "
                            f"(reduce lines present: {len(_reduce_lines(g))})")
        bits[c] = tuple(bitclass_key(spec.run(c, s)) for s in R_SEEDS)

    cfgs = [c for c in cfgs if c in ttgir]  # keep only those that compiled
    TP = FP = TN = FN = 0
    fp_pairs = []
    for a, b in itertools.combinations(cfgs, 2):
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

    n_stat = len({desc[c] for c in cfgs})
    n_emp = len({bits[c] for c in cfgs})
    # per-ordering empirical bit-class counts (sanity vs the branch's observed numbers)
    emp_by_ord = {}
    for o in ("unordered", "inner_tree"):
        members = [bits[c] for c in cfgs if c.ordering == o]
        emp_by_ord[o] = len(set(members))

    return dict(name=spec.name, cfgs=cfgs, ttgir=ttgir, desc=desc, n_stat=n_stat, n_emp=n_emp, emp_by_ord=emp_by_ord,
                TP=TP, FP=FP, TN=TN, FN=FN, fp_pairs=fp_pairs, problems=problems)


def _short(c):
    return f"{c.ordering[:3]}/nw{c.num_warps}/ns{c.num_stages}"


def report(res):
    print("\n" + "=" * 78)
    print(f"KERNEL: {res['name']}   ({len(res['cfgs'])} configs, {len(R_SEEDS)} random inputs each)")
    print("=" * 78)
    sample = res["cfgs"][0]
    print(f"  sample descriptor [{_short(sample)}]: {res['desc'][sample]}")
    for ln in _reduce_lines(res["ttgir"][sample])[:3]:
        print(f"    tt.reduce: {ln[:130]}")
    print(f"  empirical bit-classes: unordered={res['emp_by_ord']['unordered']}, "
          f"inner_tree={res['emp_by_ord']['inner_tree']}  "
          f"(expect unordered>1 split, inner_tree==1 collapsed)")
    print(f"  static classes={res['n_stat']}, empirical classes={res['n_emp']}")
    TP, FP, TN, FN = res["TP"], res["FP"], res["TN"], res["FN"]
    print(f"  confusion over {TP + FP + TN + FN} pairs:  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    emp_diff = TN + FP
    stat_eq = TP + FP
    det = (100.0 * TN / emp_diff) if emp_diff else float("nan")
    fpr = (100.0 * FP / stat_eq) if stat_eq else float("nan")
    print(f"  Part 1 detection (TN/(TN+FP)) = {det:.1f}%   over {emp_diff} emp-different pairs")
    print(f"  Part 2 soundness FP rate (FP/(TP+FP)) = {fpr:.1f}%   over {stat_eq} stat-equiv pairs"
          if stat_eq else "  Part 2: no stat-equiv pairs (unexpected)")
    if FN:
        print(f"  note: {FN} conservative over-splits (emp-equal but flagged different) — acceptable")
    for a, b in res["fp_pairs"]:
        print(f"  *** FP (UNSOUND): {_short(a)} ~ {_short(b)} declared equivalent but bits differ")
    for p in res["problems"]:
        print(f"  *** PROBLEM: {p}")


def main():
    print(f"device: {torch.cuda.get_device_name()}  cc={torch.cuda.get_device_capability()}  "
          f"triton={triton.__version__}")
    results = [evaluate(s()) for s in (_sum_spec, _softmax_spec, _layernorm_spec)]
    for res in results:
        report(res)

    total_fp = sum(r["FP"] for r in results)
    total_problems = sum(len(r["problems"]) for r in results)
    print("\n" + "=" * 78)
    print(f"SUMMARY: total FP (soundness violations) = {total_fp};  flagged problems = {total_problems}")
    print("=" * 78)
    if total_fp or total_problems:
        raise SystemExit(f"FAIL: {total_fp} soundness violation(s), {total_problems} problem(s) — "
                         "see report above. Fixes are the next step (per plan).")
    print("PASS: no soundness violations; detection as expected.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
