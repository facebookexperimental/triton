"""Correctness test for the HSTU SELF-attention forward under automatic warp
specialization (meta-WS).

Enables autoWS on the hammer-template Triton self-attn kernel via the
`warp_specialize=True` annotation on the main KV loop (env HSTU_SELF_AUTOWS=1,
baked as a tl.constexpr at import) and asserts the fwd output AND the dq/dk/dv
gradients match a torch-autograd float causal-SiLU reference.

Four configs are covered:
- The DEFAULT autoWS config (in-process): `test_self_attention_fwd_autows`.
- The TLX-matching dq-reduce config (BM=BN=128, num_stages=2, TMEM reuse, dsT in
  SMEM): `test_self_attention_bwd_autows_dqreduce`. Its flags
  (HSTU_SELF_DQ_REDUCE / HSTU_SELF_DQ_REUSE / the BWD tile knobs) are baked as
  tl.constexpr / read into the autotune configs AT IMPORT, so they can't coexist
  with the default config in one interpreter -> that case runs in a SUBPROCESS
  (this same file re-invoked with `--run-dqreduce`) with the env set first.
- The FA-style manual data-partition FWD config (HSTU_SELF_FA_DP=1, DP=2, WARPS=4):
  `test_self_attention_fwd_autows_fadp`. Splits BLOCK_M=256 into two 128-row halves
  sharing one K/V load, warp-specialized into load + 2 MMA groups -- the working
  alternative to the broken compiler data_partition_factor=2 path. Also
  constexpr-baked at import -> SUBPROCESS (`--run-fadp`); fwd-only output check.
- The compiler DP=2 config (HSTU_SELF_DP=2, no FA_DP):
  `test_self_attention_fwd_autows_compiler_dp2`. The compiler's
  data_partition_factor=2 splits the 256-row tile into two 128-row groups. Was
  broken (TMA descriptor box_dim left at 256 -> SMEM overrun -> stall/OOB); fixed
  in WSDataPartition by scaling box_dim by the partition factor. SUBPROCESS,
  `--run-fwd`, fwd-only output check.

Notes:
- autoWS needs TRITON_USE_META_WS=1 + TRITON_DISABLE_WSBARRIER_REORDER=1; these
  are set BEFORE importing the kernel so the constexpr/config pick see them.
- HSTU_SELF_DP=1: data_partition_factor=2 would need BLOCK_M=256 (each slice
  >=128 TMEM rows) which OOMs TMEM on this kernel, so DP is off here.
- The TLX self-attn *fwd* uses num_stages=0 and asserts under meta-WS, so it
  cannot be compiled in the same (META_WS) process; the accuracy oracle here is
  the torch reference (as in test_cross_attention_bwd_autows.py). The dq-reduce
  config's grads match this torch ref to bf16 precision, i.e. the same numerics
  the hand-written TLX bwd produces.

Run: pytest third_party/tlx/tutorials/test_self_attention_autows.py
"""
import os
import subprocess
import sys

_HSTU_DIR = os.path.join(os.path.dirname(__file__), "hstu_self_attn")
sys.path.insert(0, _HSTU_DIR)

# HSTU autoWS config as dicts (replaces the HSTU_SELF_* env vars), applied via
# hstu_autows_config.set_config() BEFORE importing the kernel -- the autoWS
# structural flags and the autotune tile config are read at import. Only the
# Triton *compiler* knobs (meta-WS on, wsbarrier-reorder off) stay as env; they
# are generic triton knobs, not HSTU_SELF_* config.
import hstu_autows_config as _C  # noqa: E402

# DEFAULT autoWS config (in-process fwd+grads test).
_DEFAULT_CFG = dict(autows=True, dp=1, pin=True)
# TLX-matching dq-reduce config: BM=BN=128, num_stages=2 (annotation-driven SWP),
# FA-style TMEM reuse (id2={qk,act}, id5={dp,dq}, id7=dv, id10=dk) with dsT pinned
# to SMEM, dq via TMA reduce-add subtiled x4 -- final ttgir matches the hand-written
# TLX bwd and its grads match torch/TLX numerics.
_DQREDUCE_CFG = dict(
    autows=True, dq_reduce=True, dq_reuse=True, dp=1,
    bwd_bm=128, bwd_bn=128, bwd_stages=2, warps=4, dq_iters=4, pin=True,
)
# FA-style manual data-partition fwd: split BLOCK_M=256 into two 128-row halves
# sharing one K/V load, warp-specialized (load + 2 MMA groups).
_FADP_CFG = dict(autows=True, fa_dp=True, dp=2, warps=4, pin=True)
# Compiler data_partition_factor=2 fwd (no FA_DP): the compiler splits the 256 tile
# (fixed via the WSDataPartition box_dim scaling).
_COMPILER_DP2_CFG = dict(autows=True, dp=2, warps=4, pin=True)

# The dq-reduce / fadp / compiler-dp2 cases re-invoke this file as a subprocess;
# select the config (before the kernel import below) from argv.
if "--run-dqreduce" in sys.argv:
    _C.set_config(**_DQREDUCE_CFG)
    os.environ["TRITON_WS_SMEM_PLAN_SEARCH"] = "1"
elif "--run-fadp" in sys.argv:
    _C.set_config(**_FADP_CFG)
elif "--run-fwd" in sys.argv:
    _C.set_config(**_COMPILER_DP2_CFG)
else:
    _C.set_config(**_DEFAULT_CFG)
os.environ["TRITON_USE_META_WS"] = "1"
os.environ["TRITON_DISABLE_WSBARRIER_REORDER"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

import triton_hstu_attention as A  # noqa: E402

D = 128
H = 2


def _torch_ref(q, k, v, do, so, asc):
    """Float autograd HSTU-SiLU causal self-attention reference."""
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    alpha, scale = 1.0 / D, asc.item()
    outs = []
    for z in range(so.numel() - 1):
        s, e = int(so[z]), int(so[z + 1])
        n = e - s
        qk = torch.einsum("qhd,khd->hqk", qf[s:e], kf[s:e]) * alpha
        sig = qk * torch.sigmoid(qk) * scale
        i = torch.arange(n, device=qk.device)
        sig = sig * (i[:, None] >= i[None, :]).float()[None]
        outs.append(torch.einsum("hqk,khd->qhd", sig, vf[s:e]))
    torch.cat(outs, 0).backward(do.float())
    return qf.grad, kf.grad, vf.grad


def _rel_l2(a, b):
    return (torch.norm(a.float() - b.float()) / (torch.norm(b.float()) + 1e-12)).item()


def _run_autows_bwd(L, Z):
    """Run the (already-imported) autoWS kernel fwd+bwd and return the grads plus
    the torch-float reference grads. Config is whatever env was baked at import."""
    torch.manual_seed(0)
    t = Z * L
    g = lambda: torch.randn(t, H, D, device="cuda", dtype=torch.bfloat16)  # noqa: E731
    q, k, v = g().requires_grad_(True), g().requires_grad_(True), g().requires_grad_(True)
    so = torch.arange(0, t + 1, L, device="cuda", dtype=torch.int64)
    asc = torch.tensor(1.0 / L, device="cuda", dtype=torch.float32)
    do = g()

    rq, rk, rv = _torch_ref(q, k, v, do, so, asc)

    for tsr in (q, k, v):
        tsr.grad = None
    o = A.triton_hstu_mha(
        max_seq_len=L, alpha=1.0 / D, q=q, k=k, v=v, seq_offsets=so,
        attn_scale=asc, enable_tma=True,
    )
    o.backward(do)
    return (q.grad.clone(), k.grad.clone(), v.grad.clone()), (rq, rk, rv)


def _torch_ref_fwd(q, k, v, so, asc):
    """Float HSTU-SiLU causal self-attention forward reference (output only)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    alpha, scale = 1.0 / D, asc.item()
    outs = []
    for z in range(so.numel() - 1):
        s, e = int(so[z]), int(so[z + 1])
        n = e - s
        qk = torch.einsum("qhd,khd->hqk", qf[s:e], kf[s:e]) * alpha
        sig = qk * torch.sigmoid(qk) * scale
        i = torch.arange(n, device=qk.device)
        sig = sig * (i[:, None] >= i[None, :]).float()[None]
        outs.append(torch.einsum("hqk,khd->qhd", sig, vf[s:e]))
    return torch.cat(outs, 0)


def _run_autows_fwd(L, Z):
    """Run the (already-imported) autoWS kernel forward-only and return the fwd
    output plus the torch-float reference output. Config = env baked at import."""
    torch.manual_seed(0)
    t = Z * L
    g = lambda: torch.randn(t, H, D, device="cuda", dtype=torch.bfloat16)  # noqa: E731
    q, k, v = g(), g(), g()
    so = torch.arange(0, t + 1, L, device="cuda", dtype=torch.int64)
    asc = torch.tensor(1.0 / L, device="cuda", dtype=torch.float32)
    ref = _torch_ref_fwd(q, k, v, so, asc)
    o = A.triton_hstu_mha(
        max_seq_len=L, alpha=1.0 / D, q=q, k=k, v=v, seq_offsets=so,
        attn_scale=asc, enable_tma=True,
    )
    return o, ref


@pytest.mark.parametrize("L,Z", [(256, 4), (512, 2)])
def test_self_attention_fwd_autows(L, Z):
    """DEFAULT autoWS config (in-process): fwd + grads vs torch reference."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert bool(A._HSTU_SELF_AUTOWS), "autoWS flag not baked on"

    (dq, dk, dv), (rq, rk, rv) = _run_autows_bwd(L, Z)

    for name, got, want in (("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv)):
        rl2 = _rel_l2(got, want)
        assert rl2 < 1e-2, f"autoWS {name} rel-L2 {rl2:.2e} too high (L={L} Z={Z})"


@pytest.mark.parametrize("L,Z", [(256, 2)])
def test_self_attention_bwd_autows_dqreduce(L, Z):
    """TLX-matching dq-reduce config (BM=BN=128, ns=2, TMEM reuse, dsT-in-SMEM).

    Runs in a SUBPROCESS because its flags are constexpr-baked at import and
    cannot share an interpreter with the default config above. Asserts the bwd
    grads match the torch-float reference (== TLX numerics) to bf16 precision;
    this is the case that previously produced dv rel-L2 ~0.5 before the
    dsT-in-SMEM / mmaSelfLatency fixes.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # The subprocess re-runs this file with --run-dqreduce, which selects
    # _DQREDUCE_CFG via hstu_autows_config.set_config() before importing the kernel
    # (no HSTU_SELF_* env needed).
    r = subprocess.run(
        [sys.executable, __file__, "--run-dqreduce", str(L), str(Z)],
        env=dict(os.environ), capture_output=True, text=True, timeout=900,
    )
    # Surface the child's REL_L2 line (and any error) in the pytest output.
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    assert r.returncode == 0, (
        f"dq-reduce autoWS bwd failed (L={L} Z={Z}):\n{r.stdout}\n{r.stderr}"
    )


@pytest.mark.parametrize("L,Z", [(256, 4), (512, 2)])
def test_self_attention_fwd_autows_fadp(L, Z):
    """FA-style manual data-partition fwd (split BLOCK_M=256 -> 2x128, shared K/V,
    warp-specialized 2 MMA groups; env HSTU_SELF_FA_DP=1 + HSTU_SELF_DP=2).

    Runs in a SUBPROCESS because HSTU_SELF_* are constexpr-baked at import and
    cannot share an interpreter with the default (DP=1) config above. FA-DP is a
    forward-only change, so this checks the forward OUTPUT vs the torch-float
    reference (the compiler data_partition_factor=2 path is broken on this kernel;
    this manual split is the working, warp-specialized alternative).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # The subprocess selects _FADP_CFG via set_config() (argv --run-fadp) before
    # importing the kernel; no HSTU_SELF_* env needed.
    r = subprocess.run(
        [sys.executable, __file__, "--run-fadp", str(L), str(Z)],
        env=dict(os.environ), capture_output=True, text=True, timeout=900,
    )
    # Surface the child's REL_L2 line (and any error) in the pytest output.
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    assert r.returncode == 0, (
        f"FA-DP autoWS fwd failed (L={L} Z={Z}):\n{r.stdout}\n{r.stderr}"
    )


@pytest.mark.parametrize("L,Z", [(256, 4), (512, 2)])
def test_self_attention_fwd_autows_compiler_dp2(L, Z):
    """Compiler data-partition fwd (HSTU_SELF_DP=2, no FA_DP): the compiler splits
    the 2*BLOCK_M=256 tile into two 128-row groups. Runs in a SUBPROCESS
    (constexpr-baked config). Forward-only output check vs the torch reference.

    This path was previously broken -- the DP transform left the device-side TMA
    descriptor box_dim at the un-partitioned 256, so the TMA overran the 128-row
    SMEM buffer (tcgen05 TMEM-alloc stall / epilogue-store OOB). Fixed in
    WSDataPartition by scaling the tensormap box_dim by the partition factor.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # The subprocess selects _COMPILER_DP2_CFG via set_config() (argv --run-fwd)
    # before importing the kernel; no HSTU_SELF_* env needed.
    r = subprocess.run(
        [sys.executable, __file__, "--run-fwd", str(L), str(Z)],
        env=dict(os.environ), capture_output=True, text=True, timeout=900,
    )
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    assert r.returncode == 0, (
        f"compiler DP=2 fwd failed (L={L} Z={Z}):\n{r.stdout}\n{r.stderr}"
    )


if __name__ == "__main__":
    # Subprocess entry point for the dq-reduce config. _DQREDUCE_CFG was applied
    # via set_config() at the top of this file (argv --run-dqreduce) before the
    # kernel import, so the dq-reduce constexprs / autotune config are baked on.
    if len(sys.argv) >= 4 and sys.argv[1] == "--run-dqreduce":
        _L, _Z = int(sys.argv[2]), int(sys.argv[3])
        assert bool(A._HSTU_DQ_REUSE), "dq-reduce reuse flag not baked on"
        (dq, dk, dv), (rq, rk, rv) = _run_autows_bwd(_L, _Z)
        rls = {n: _rel_l2(g_, w) for n, g_, w in
               (("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv))}
        print(
            f"REL_L2 dq/dk/dv = {rls['dq']:.2e} / {rls['dk']:.2e} / {rls['dv']:.2e} "
            f"(L={_L} Z={_Z})"
        )
        bad = {n: v for n, v in rls.items() if not (v < 1e-2)}
        if bad:
            print(f"FAIL: rel-L2 too high: {bad}")
            sys.exit(1)
        print("OK")
        sys.exit(0)
    # Subprocess entry point for the FA-style manual-DP fwd config (_FADP_CFG
    # applied via set_config() at import). Forward-only check vs the torch ref.
    if len(sys.argv) >= 4 and sys.argv[1] == "--run-fadp":
        _L, _Z = int(sys.argv[2]), int(sys.argv[3])
        assert bool(A._HSTU_SELF_FA_DP), "FA-DP flag not baked on"
        o, ref = _run_autows_fwd(_L, _Z)
        rl2 = _rel_l2(o, ref)
        print(f"REL_L2 fwd = {rl2:.2e} (L={_L} Z={_Z})")
        if not (rl2 < 1e-2):
            print(f"FAIL: fwd rel-L2 too high: {rl2:.2e}")
            sys.exit(1)
        print("OK")
        sys.exit(0)
    # Generic fwd-only entry (config = whatever env was baked at import). Used by
    # the known-broken compiler-DP=2 xfail; _rel_l2 forces a sync so a device
    # fault (illegal address) surfaces here as a nonzero exit.
    if len(sys.argv) >= 4 and sys.argv[1] == "--run-fwd":
        _L, _Z = int(sys.argv[2]), int(sys.argv[3])
        o, ref = _run_autows_fwd(_L, _Z)
        rl2 = _rel_l2(o, ref)
        print(f"REL_L2 fwd = {rl2:.2e} (L={_L} Z={_Z})")
        if not (rl2 < 1e-2):
            print(f"FAIL: fwd rel-L2 too high: {rl2:.2e}")
            sys.exit(1)
        print("OK")
        sys.exit(0)
    sys.exit(
        "usage: test_self_attention_autows.py "
        "--run-dqreduce|--run-fadp|--run-fwd L Z"
    )
