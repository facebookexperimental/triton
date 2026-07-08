"""Correctness test for the HSTU cross-attention backward (reduce_dq) under
automatic warp specialization (meta-WS).

Regression coverage for the premature k/v EMPTY-release bug: k/v are TMA-loaded
per KV block in the outer loop and read by the inner Q-loop MMAs on a single
copy=1 buffer; releasing the buffer every inner iteration (instead of only on
the last) corrupted the trailing min(KV_blocks-1, num_stages) Q-blocks of dq for
KV>=2. This test runs the autoWS kernel across KV=1..4 and num_stages=1/2 and
asserts it matches both the hand-written TLX kernel (byte-reference for the dq
Q-blocks) and a torch-autograd float reference (dq/dk/dv rel-L2).

Run: pytest third_party/tlx/tutorials/test_cross_attention_bwd_autows.py
"""
import os
import sys

# The HSTU kernels live in the hstu_cross_attn/ subpackage; its modules use bare
# imports (e.g. `import stubs`, `import triton_bw_cross_attention`) that assume
# the subdir is on sys.path (as the repro/bench scripts do). Put it on the path
# and import them as top-level modules (avoids triggering the subpackage
# __init__ before the path is set).
_HSTU_DIR = os.path.join(os.path.dirname(__file__), "hstu_cross_attn")
sys.path.insert(0, _HSTU_DIR)
os.environ.pop("TRITON_USE_META_WS", None)

import pytest
import torch
import triton.language as tl
from triton.runtime.errors import OutOfResources

import bench_bwd as bb
import triton_bw_cross_attention as xa


# (Lq, Lkv, Z): KV=1 is always correct; KV>=2 exercised the bug (bad Q-blocks at
# the tail of the Q range). Lq=256 -> 4 Q-blocks (needs >=2 for the inner peel).
_SHAPES = [(256, 64, 2), (256, 128, 2), (256, 192, 2), (256, 256, 2)]


@pytest.mark.parametrize("ns", [1, 2])
@pytest.mark.parametrize("Lq,Lkv,Z", _SHAPES)
def test_cross_attention_bwd_autows(Lq, Lkv, Z, ns):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    bb.force(ns)
    torch.manual_seed(0)
    q, k, v, do, so_kv, so_q, asc = bb.make(Lq, Lkv, Z)

    # torch-autograd float reference (accuracy oracle).
    rq, rk, rv = bb.torch_ref(q, k, v, do, so_kv, so_q, asc)

    # Hand-written TLX kernel: byte-reference for the dq bad-Q-block count.
    for t in (q, k, v):
        t.grad = None
    bb._fwd(xa.BwdVariant.TLX, "0", q, k, v, so_kv, so_q, Lkv, Lq, asc).backward(do)
    ref_dq = q.grad.clone()

    # autoWS kernel under test.
    for t in (q, k, v):
        t.grad = None
    bb._fwd(
        xa.BwdVariant.TRITON_AUTOWS, "1", q, k, v, so_kv, so_q, Lkv, Lq, asc
    ).backward(do)
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # No dq Q-block may diverge from the TLX reference (this is what the bug hit).
    n_bad, bad_blocks = bb.bad_qblocks(dq, ref_dq, Lq)
    assert n_bad == 0, (
        f"autoWS dq diverges from TLX in Q-blocks {bad_blocks} "
        f"({n_bad} rows) for Lq={Lq} Lkv={Lkv} ns={ns}"
    )

    # And all three grads must match the float reference to bf16 precision.
    for name, got, want in (("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv)):
        rl2 = bb.rel_l2(got, want)
        assert rl2 < 1e-2, f"{name} rel-L2 {rl2:.2e} too high (Lq={Lq} Lkv={Lkv} ns={ns})"


# Shared-KV (V aliases K) variants. Shapes are Lkv multiples of 128 (2/3/4 KV
# blocks of 128), including an odd count to exercise the partial tail pair.
_SHARED_SHAPES = [(256, 256, 2), (256, 384, 2), (256, 512, 2)]


@pytest.mark.parametrize("ns", [1, 2])
@pytest.mark.parametrize("Lq,Lkv,Z", _SHARED_SHAPES)
def test_cross_attention_bwd_tlx_2kv(Lq, Lkv, Z, ns):
    """TLX 2-KV-block data-partitioned reduce_dq (BwdVariant.TLX_2KV), shared-KV.
    Must match the torch-float reference and be run-to-run deterministic."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    bb.force(ns)
    torch.manual_seed(0)
    q, k, v, do, so_kv, so_q, asc = bb.make(Lq, Lkv, Z, shared=True)
    # shared-KV: K and V are one leaf, so the returned dk = dk + dv (dv is None).
    rq, rk, _ = bb.torch_ref(q, k, v, do, so_kv, so_q, asc, shared=True)

    grads = []
    for _ in range(2):  # determinism: two identical runs
        for t in (q, k, v):
            t.grad = None
        bb._fwd(xa.BwdVariant.TLX_2KV, "0", q, k, v, so_kv, so_q, Lkv, Lq, asc,
                shared=True).backward(do)
        grads.append((q.grad.clone(), k.grad.clone()))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(
        grads[0][1], grads[1][1]
    ), f"TLX_2KV nondeterministic (Lq={Lq} Lkv={Lkv} ns={ns})"

    dq, dk = grads[0]
    for name, got, want in (("dq", dq, rq), ("dk", dk, rk)):
        rl2 = bb.rel_l2(got, want)
        assert rl2 < 1e-2, (
            f"TLX_2KV {name} rel-L2 {rl2:.2e} too high (Lq={Lq} Lkv={Lkv} ns={ns})"
        )


@pytest.mark.xfail(
    raises=OutOfResources,
    strict=True,
    reason="T278685041 (handleOperandD chained shared-opndD accumulator) is FIXED: "
    "the coalesced dv/dk_attn MMAs now form a producer chain in the autoWS "
    "channel model and the autoWS+DP=2 fold compiles. The kernel still overflows "
    "shared memory at launch (Required ~576KB > 232KB HW limit) under BLOCK_N=256 "
    "(DP=2) -- a separate downstream memory-planner layer. Drop the xfail once "
    "SMEM fits; the body already asserts grads vs the torch reference.",
)
@pytest.mark.parametrize("ns", [1, 2])
@pytest.mark.parametrize("Lq,Lkv,Z", _SHARED_SHAPES)
def test_cross_attention_bwd_autows_dp_fold(Lq, Lkv, Z, ns, monkeypatch):
    """autoWS reduce_dq with geometric DP (BLOCK_N=256 -> data_partition_factor=2)
    and the compute fold (dk_attn folded into the dk accumulator, TLX 2kv shape).
    Currently xfails at compile in handleOperandD; the body still asserts grads vs
    the torch reference so this flips to XPASS -- and surfaces any wrong grad --
    once the block is fixed."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    # The fold gate is a module-level constexpr evaluated at import; override the
    # attribute (setting the env var now is too late).
    monkeypatch.setattr(xa, "_HSTU_COMPUTE_FOLD", tl.constexpr(True))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    xa.set_bwd_dot_attrs(xa._HSTU_BWD_DOT_ATTRS_TLX)

    bb.force(ns, bm=128, bn=256)  # BLOCK_N=256 -> data_partition_factor=2
    torch.manual_seed(0)
    q, k, v, do, so_kv, so_q, asc = bb.make(Lq, Lkv, Z, shared=True)
    rq, rk, _ = bb.torch_ref(q, k, v, do, so_kv, so_q, asc, shared=True)

    for t in (q, k, v):
        t.grad = None
    bb._fwd(xa.BwdVariant.TRITON_AUTOWS, "1", q, k, v, so_kv, so_q, Lkv, Lq, asc,
            shared=True).backward(do)
    dq, dk = q.grad.clone(), k.grad.clone()
    for name, got, want in (("dq", dq, rq), ("dk", dk, rk)):
        rl2 = bb.rel_l2(got, want)
        assert rl2 < 1e-2, (
            f"autoWS+DP fold {name} rel-L2 {rl2:.2e} (Lq={Lq} Lkv={Lkv} ns={ns})"
        )
