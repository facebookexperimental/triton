"""Correctness test for the HSTU SELF-attention BACKWARD: plain-Triton bwd vs the
hand-written TLX bwd, and both vs a torch-autograd float causal-SiLU reference.

Runs WITHOUT meta-WS (TRITON_USE_META_WS unset): the TLX self-attn fwd uses
num_stages=1 and asserts under meta-WS, so this is the process where plain-Triton
and TLX can coexist. (The autoWS variant is a separate process /
test_self_attention_autows.py.) HSTU_SELF_PIN=1 pins the bwd autotune to one
config so it compiles fast instead of building the ~29-config bwd space.

Run: pytest third_party/tlx/tutorials/test_self_attention_bwd.py
"""
import os
import sys

_HSTU_DIR = os.path.join(os.path.dirname(__file__), "hstu_self_attn")
sys.path.insert(0, _HSTU_DIR)

# HSTU autoWS config as a dict (replaces HSTU_SELF_* env vars), applied via
# hstu_autows_config.set_config() before importing the kernel (config is read at
# import). This is the non-autoWS path (plain-Triton vs TLX): autoWS off, pinned
# for fast compile. The Triton compiler knobs (use_meta_ws off, wsbarrier-reorder
# off for TLX) are applied per-run via a knobs scope in bench_self.grads().
import hstu_autows_config as _C  # noqa: E402

_C.set_config(autows=False, pin=True)

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
from triton._internal_testing import is_blackwell  # noqa: E402

import bench_self as bs  # noqa: E402

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell (sm100)")


def _run_tlx_raw_bwd(q, k, v, do, so, asc, L, num_targets, use_persistent):
    """Run one TLX backward variant without coupling the test to TLX forward."""
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    # SiLU backward does not consume M/Delta, but the common wrapper requires
    # stable pointer arguments.
    M = torch.empty((1, ), device=q.device, dtype=torch.float32)
    Delta = torch.empty((1, ), device=q.device, dtype=torch.float32)
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = False
        triton.knobs.nvidia.disable_wsbarrier_reorder = True
        return bs.T.tlx_hstu_attention_bwd(
            dout=do,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=so,
            attn_scale=asc,
            max_seq_len=L,
            alpha=1.0 / bs.D,
            M=M,
            Delta=Delta,
            stride_mm=1,
            num_softmax_heads=0,
            num_targets=num_targets,
            causal=True,
            use_persistent=use_persistent,
        )


@pytest.mark.parametrize("L,Z", [(256, 4), (512, 2)])
def test_self_attention_bwd_triton_vs_tlx(L, Z):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    torch.manual_seed(0)
    q, k, v, do, so, asc = bs.make(L, Z)
    rq, rk, rv = bs.torch_ref(q, k, v, do, so, asc, causal=True)

    _, dq_t, dk_t, dv_t = bs.grads(bs.run_triton, q, k, v, so, L, asc, do)
    _, dq_x, dk_x, dv_x = bs.grads(bs.run_tlx, q, k, v, so, L, asc, do, causal=True)

    # Each kernel's grads must match the torch-float reference (bf16 precision).
    for name, got in (
        ("triton dq", dq_t),
        ("triton dk", dk_t),
        ("triton dv", dv_t),
        ("tlx dq", dq_x),
        ("tlx dk", dk_x),
        ("tlx dv", dv_x),
    ):
        want = {"dq": rq, "dk": rk, "dv": rv}[name.split()[1]]
        rl2 = bs.rel_l2(got, want)
        assert rl2 < 1e-2, f"{name} rel-L2 {rl2:.2e} too high (L={L} Z={Z})"

    # TLX bwd must match the plain-Triton bwd (same math family).
    for name, a, b in (("dq", dq_x, dq_t), ("dk", dk_x, dk_t), ("dv", dv_x, dv_t)):
        rl2 = bs.rel_l2(a, b)
        assert rl2 < 1e-2, f"tlx-vs-triton {name} rel-L2 {rl2:.2e} too high (L={L} Z={Z})"


@pytest.mark.parametrize("use_persistent", [True, False], ids=["persistent", "non-persistent"])
@pytest.mark.parametrize("L,target_count", [(257, 20), (384, 200)])
def test_self_attention_bwd_tlx_targets(L, target_count, use_persistent):
    """PART 2 must retain the target clamp when its K/V tile reaches targets."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    Z = 2
    torch.manual_seed(0)
    q, k, v, do, so, asc = bs.make(L, Z)
    num_targets = torch.full((Z, ), target_count, device=q.device, dtype=torch.int64)
    rq, rk, rv = bs.torch_ref(q, k, v, do, so, asc, causal=True, num_targets=num_targets)
    dq, dk, dv = _run_tlx_raw_bwd(q, k, v, do, so, asc, L, num_targets, use_persistent)

    for name, got, want in (("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv)):
        rl2 = bs.rel_l2(got, want)
        assert rl2 < 1e-2, (f"TLX {name} rel-L2 {rl2:.2e} too high "
                            f"(L={L}, targets={target_count}, persistent={use_persistent})")
